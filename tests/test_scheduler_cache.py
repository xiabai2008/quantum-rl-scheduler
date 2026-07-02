"""
量子RL调度系统 - 调度决策缓存单元测试
Unit Tests for src/scheduler/cache.py

测试覆盖：
- TestSchedulerCacheBasic      : put/get/clear/len 基本操作
- TestSchedulerCacheSimilarity : 相似状态命中、不相似未命中、阈值边界
- TestSchedulerCacheLRU        : max_size 淘汰、LRU 顺序正确
- TestSchedulerCacheTTL        : TTL 过期后未命中、未过期命中
- TestSchedulerCacheStats      : hits/misses/hit_rate/size/evictions 统计
- TestSchedulerCacheThreadSafety: 多线程并发 put/get 不出错
- TestSchedulerCacheEdgeCases  : 空状态、全零状态、单元素状态
"""

import os
import sys
import threading
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import src.scheduler.cache as cache_mod
from src.scheduler.cache import SchedulerCache


# ============================================================
# 基本操作测试
# ============================================================
class TestSchedulerCacheBasic(unittest.TestCase):
    """测试 SchedulerCache 的 put/get/clear/len 基本操作。"""

    def test_empty_cache_len_zero(self):
        """新建缓存长度应为 0。"""
        cache = SchedulerCache()
        self.assertEqual(len(cache), 0)

    def test_put_increases_len(self):
        """put 后缓存长度应增加。"""
        cache = SchedulerCache()
        cache.put(np.array([1.0, 0.0]), action=1)
        self.assertEqual(len(cache), 1)

    def test_get_exact_match_returns_action(self):
        """精确匹配应返回缓存的 action。"""
        cache = SchedulerCache()
        cache.put(np.array([1.0, 2.0, 3.0]), action=2)
        result = cache.get(np.array([1.0, 2.0, 3.0]))
        self.assertEqual(result, 2)

    def test_get_empty_cache_returns_none(self):
        """空缓存 get 应返回 None。"""
        cache = SchedulerCache()
        self.assertIsNone(cache.get(np.array([1.0, 0.0])))

    def test_put_same_state_updates_action(self):
        """对同一状态再次 put 应更新 action。"""
        cache = SchedulerCache()
        cache.put(np.array([1.0, 0.0]), action=1)
        cache.put(np.array([1.0, 0.0]), action=5)
        self.assertEqual(len(cache), 1)
        self.assertEqual(cache.get(np.array([1.0, 0.0])), 5)

    def test_clear_empties_cache(self):
        """clear 应清空缓存条目。"""
        cache = SchedulerCache()
        cache.put(np.array([1.0]), 1)
        cache.put(np.array([2.0]), 2)
        self.assertEqual(len(cache), 2)
        cache.clear()
        self.assertEqual(len(cache), 0)
        self.assertIsNone(cache.get(np.array([1.0])))

    def test_invalid_constructor_args(self):
        """非法构造参数应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            SchedulerCache(max_size=0)
        with self.assertRaises(ValueError):
            SchedulerCache(max_size=-1)
        with self.assertRaises(ValueError):
            SchedulerCache(similarity_threshold=1.5)
        with self.assertRaises(ValueError):
            SchedulerCache(similarity_threshold=-0.1)
        with self.assertRaises(ValueError):
            SchedulerCache(ttl_seconds=0.0)
        with self.assertRaises(ValueError):
            SchedulerCache(ttl_seconds=-5.0)

    def test_get_accepts_multidimensional_state(self):
        """get/put 应接受多维状态（内部 flatten）。"""
        cache = SchedulerCache()
        cache.put(np.array([[1.0, 0.0], [0.0, 1.0]]), action=7)
        # 同一数据不同形状应能精确命中
        result = cache.get(np.array([1.0, 0.0, 0.0, 1.0]))
        self.assertEqual(result, 7)


# ============================================================
# 相似度匹配测试
# ============================================================
class TestSchedulerCacheSimilarity(unittest.TestCase):
    """测试余弦相似度匹配逻辑。"""

    def test_similar_state_hits(self):
        """高度相似的状态应命中缓存。"""
        cache = SchedulerCache(similarity_threshold=0.9)
        cache.put(np.array([1.0, 0.0, 0.0]), action=1)
        # 余弦相似度 ≈ 0.99995，超过 0.9
        result = cache.get(np.array([1.0, 0.01, 0.0]))
        self.assertEqual(result, 1)

    def test_dissimilar_state_misses(self):
        """正交（不相似）状态不应命中。"""
        cache = SchedulerCache(similarity_threshold=0.95)
        cache.put(np.array([1.0, 0.0, 0.0]), action=1)
        result = cache.get(np.array([0.0, 1.0, 0.0]))
        self.assertIsNone(result)

    def test_threshold_boundary_hit(self):
        """相似度刚好超过阈值应命中。"""
        # [1,0] vs [1,1]: cos = 1/sqrt(2) ≈ 0.7071
        cache = SchedulerCache(similarity_threshold=0.5)
        cache.put(np.array([1.0, 0.0]), action=3)
        result = cache.get(np.array([1.0, 1.0]))
        self.assertEqual(result, 3)

    def test_threshold_boundary_miss(self):
        """相似度低于阈值不应命中。"""
        # [1,0] vs [1,1]: cos ≈ 0.7071
        cache = SchedulerCache(similarity_threshold=0.8)
        cache.put(np.array([1.0, 0.0]), action=3)
        result = cache.get(np.array([1.0, 1.0]))
        self.assertIsNone(result)

    def test_opposite_direction_misses(self):
        """方向相反的状态余弦相似度为负，不应命中。"""
        cache = SchedulerCache(similarity_threshold=0.5)
        cache.put(np.array([1.0, 0.0]), action=1)
        # cos([1,0], [-1,0]) = -1
        result = cache.get(np.array([-1.0, 0.0]))
        self.assertIsNone(result)

    def test_picks_most_similar(self):
        """存在多个缓存条目时应选相似度最高的命中。"""
        cache = SchedulerCache(similarity_threshold=0.7)
        cache.put(np.array([1.0, 0.0]), action=10)  # 与 [1,1] 相似度 0.707
        cache.put(np.array([0.0, 1.0]), action=20)  # 与 [1,1] 相似度 0.707
        # 查询 [1,1]，两个条目相似度相同，应命中其一
        result = cache.get(np.array([1.0, 1.0]))
        self.assertIn(result, (10, 20))

    def test_different_dimension_states(self):
        """不同维度的状态不应导致崩溃，且不命中。"""
        cache = SchedulerCache(similarity_threshold=0.9)
        cache.put(np.array([1.0, 0.0, 0.0]), action=1)
        # 维度不同，相似度返回 0.0
        result = cache.get(np.array([1.0, 0.0]))
        self.assertIsNone(result)


# ============================================================
# LRU 淘汰测试
# ============================================================
class TestSchedulerCacheLRU(unittest.TestCase):
    """测试 LRU 淘汰策略与顺序维护。"""

    def test_max_size_evicts_oldest(self):
        """超过 max_size 时应淘汰最久未访问的条目。"""
        cache = SchedulerCache(max_size=2, similarity_threshold=0.99)
        # 使用相互正交的 2D 向量，避免相似度误命中
        cache.put(np.array([1.0, 0.0]), 10)
        cache.put(np.array([0.0, 1.0]), 20)
        # 新增第三个，应淘汰 [1,0]（最久未访问）
        cache.put(np.array([1.0, 1.0]), 30)
        self.assertEqual(len(cache), 2)
        # [1,0] 被淘汰，且与剩余 [0,1]/[1,1] 相似度均 < 0.99
        self.assertIsNone(cache.get(np.array([1.0, 0.0])))

    def test_lru_order_after_get(self):
        """get 访问应更新 LRU 顺序，被访问的条目不被淘汰。"""
        cache = SchedulerCache(max_size=2, similarity_threshold=0.99)
        cache.put(np.array([1.0, 0.0]), 10)
        cache.put(np.array([0.0, 1.0]), 20)
        # 访问 [1,0]，使其成为最近使用
        self.assertEqual(cache.get(np.array([1.0, 0.0])), 10)
        # 新增第三个，应淘汰 [0,1]（最久未访问）
        cache.put(np.array([1.0, 1.0]), 30)
        self.assertEqual(len(cache), 2)
        # [1,0] 仍应存在
        self.assertEqual(cache.get(np.array([1.0, 0.0])), 10)
        # [0,1] 应被淘汰
        self.assertIsNone(cache.get(np.array([0.0, 1.0])))

    def test_lru_order_after_put_update(self):
        """对已存在状态再次 put 应更新其 LRU 顺序。"""
        cache = SchedulerCache(max_size=2, similarity_threshold=0.99)
        cache.put(np.array([1.0, 0.0]), 10)
        cache.put(np.array([0.0, 1.0]), 20)
        # 重新 put [1,0]，使其成为最近使用
        cache.put(np.array([1.0, 0.0]), 11)
        # 新增第三个，应淘汰 [0,1]
        cache.put(np.array([1.0, 1.0]), 30)
        self.assertEqual(len(cache), 2)
        self.assertEqual(cache.get(np.array([1.0, 0.0])), 11)
        self.assertIsNone(cache.get(np.array([0.0, 1.0])))

    def test_eviction_counter(self):
        """淘汰条目应累加 evictions 计数。"""
        cache = SchedulerCache(max_size=1, similarity_threshold=0.99)
        cache.put(np.array([1.0, 0.0]), 1)
        cache.put(np.array([0.0, 1.0]), 2)  # 淘汰 1 个
        cache.put(np.array([1.0, 1.0]), 3)  # 淘汰 1 个
        stats = cache.stats()
        self.assertEqual(stats["evictions"], 2)

    def test_large_max_size_no_eviction(self):
        """未达 max_size 时不应发生淘汰。"""
        cache = SchedulerCache(max_size=1000)
        for i in range(50):
            cache.put(np.array([float(i), float(i + 1)]), i)
        self.assertEqual(cache.stats()["evictions"], 0)
        self.assertEqual(len(cache), 50)


# ============================================================
# TTL 过期测试
# ============================================================
class TestSchedulerCacheTTL(unittest.TestCase):
    """测试 TTL 过期逻辑（通过 mock time.monotonic 控制时间）。"""

    def test_ttl_not_expired_hits(self):
        """TTL 未过期时应命中。"""
        cache = SchedulerCache(ttl_seconds=100.0, similarity_threshold=0.99)
        # put 时刻 1000.0，get 时刻 1050.0，差 50 < 100
        times = [1000.0, 1050.0]
        with patch.object(cache_mod.time, "monotonic", side_effect=times):
            cache.put(np.array([1.0, 0.0]), action=1)
            result = cache.get(np.array([1.0, 0.0]))
        self.assertEqual(result, 1)

    def test_ttl_expired_misses(self):
        """TTL 过期后应未命中。"""
        cache = SchedulerCache(ttl_seconds=10.0, similarity_threshold=0.99)
        # put 时刻 1000.0，get 时刻 1100.0，差 100 > 10
        times = [1000.0, 1100.0]
        with patch.object(cache_mod.time, "monotonic", side_effect=times):
            cache.put(np.array([1.0, 0.0]), action=1)
            result = cache.get(np.array([1.0, 0.0]))
        self.assertIsNone(result)

    def test_ttl_exact_boundary_hits(self):
        """TTL 边界（now - ts == ttl）应视为未过期命中。"""
        cache = SchedulerCache(ttl_seconds=10.0, similarity_threshold=0.99)
        # put 1000.0，get 1010.0，差 10 == ttl
        times = [1000.0, 1010.0]
        with patch.object(cache_mod.time, "monotonic", side_effect=times):
            cache.put(np.array([1.0, 0.0]), action=1)
            result = cache.get(np.array([1.0, 0.0]))
        self.assertEqual(result, 1)

    def test_ttl_expired_falls_through_to_similarity(self):
        """精确匹配过期后，相似度匹配仍可命中其他未过期条目。"""
        cache = SchedulerCache(ttl_seconds=10.0, similarity_threshold=0.9)
        # put [1,0] @ 1000（将过期），put [1,0.001] @ 1095（未过期），get [1,0] @ 1100
        # [1,0] 精确匹配已过期（1100-1000=100 > 10），但 [1,0.001] 相似度 ≈ 0.9999995
        # >= 0.9 且未过期（1100-1095=5 <= 10），应通过相似度命中
        times = [1000.0, 1095.0, 1100.0]
        with patch.object(cache_mod.time, "monotonic", side_effect=times):
            cache.put(np.array([1.0, 0.0]), action=1)
            cache.put(np.array([1.0, 0.001]), action=2)
            result = cache.get(np.array([1.0, 0.0]))
        # 应通过相似度命中 [1,0.001]
        self.assertEqual(result, 2)

    def test_ttl_refreshed_on_put(self):
        """对同一状态再次 put 应刷新其 TTL 时间戳。"""
        cache = SchedulerCache(ttl_seconds=10.0, similarity_threshold=0.99)
        # put @ 1000，put @ 1095（刷新），get @ 1105
        # 第一次 put 的 TTL 已被覆盖，1105 - 1095 = 10 <= 10，应命中
        times = [1000.0, 1095.0, 1105.0]
        with patch.object(cache_mod.time, "monotonic", side_effect=times):
            cache.put(np.array([1.0, 0.0]), action=1)
            cache.put(np.array([1.0, 0.0]), action=2)
            result = cache.get(np.array([1.0, 0.0]))
        self.assertEqual(result, 2)


# ============================================================
# 统计信息测试
# ============================================================
class TestSchedulerCacheStats(unittest.TestCase):
    """测试 stats() 返回的统计信息正确性。"""

    def test_initial_stats(self):
        """新建缓存统计应为零值。"""
        cache = SchedulerCache()
        s = cache.stats()
        self.assertEqual(s["hits"], 0)
        self.assertEqual(s["misses"], 0)
        self.assertEqual(s["hit_rate"], 0.0)
        self.assertEqual(s["size"], 0)
        self.assertEqual(s["evictions"], 0)

    def test_hits_and_misses_counted(self):
        """命中与未命中应分别计数。"""
        cache = SchedulerCache(similarity_threshold=0.99)
        cache.put(np.array([1.0, 0.0]), 1)
        cache.get(np.array([1.0, 0.0]))  # hit
        cache.get(np.array([0.0, 1.0]))  # miss
        s = cache.stats()
        self.assertEqual(s["hits"], 1)
        self.assertEqual(s["misses"], 1)
        self.assertAlmostEqual(s["hit_rate"], 0.5)

    def test_hit_rate_calculation(self):
        """hit_rate 应等于 hits / (hits + misses)。"""
        cache = SchedulerCache(similarity_threshold=0.99)
        cache.put(np.array([1.0, 0.0]), 1)
        cache.get(np.array([1.0, 0.0]))  # hit
        cache.get(np.array([1.0, 0.0]))  # hit
        cache.get(np.array([1.0, 0.0]))  # hit
        cache.get(np.array([0.0, 1.0]))  # miss
        s = cache.stats()
        self.assertEqual(s["hits"], 3)
        self.assertEqual(s["misses"], 1)
        self.assertAlmostEqual(s["hit_rate"], 0.75)

    def test_stats_reflects_size_and_evictions(self):
        """stats 应反映当前 size 与 evictions。"""
        cache = SchedulerCache(max_size=2, similarity_threshold=0.99)
        cache.put(np.array([1.0, 0.0]), 1)
        cache.put(np.array([0.0, 1.0]), 2)
        cache.put(np.array([1.0, 1.0]), 3)  # 触发 1 次淘汰
        s = cache.stats()
        self.assertEqual(s["size"], 2)
        self.assertEqual(s["evictions"], 1)

    def test_stats_keys(self):
        """stats 应包含所有约定的键。"""
        cache = SchedulerCache()
        s = cache.stats()
        expected_keys = {"hits", "misses", "hit_rate", "size", "evictions"}
        self.assertEqual(set(s.keys()), expected_keys)

    def test_stats_value_types(self):
        """stats 各字段类型应符合声明。"""
        cache = SchedulerCache()
        cache.put(np.array([1.0]), 1)
        cache.get(np.array([1.0]))
        s = cache.stats()
        self.assertIsInstance(s["hits"], int)
        self.assertIsInstance(s["misses"], int)
        self.assertIsInstance(s["hit_rate"], float)
        self.assertIsInstance(s["size"], int)
        self.assertIsInstance(s["evictions"], int)

    def test_clear_preserves_stats(self):
        """clear 仅清除条目，不重置统计计数器。"""
        cache = SchedulerCache(similarity_threshold=0.99)
        cache.put(np.array([1.0, 0.0]), 1)
        cache.get(np.array([1.0, 0.0]))  # hit
        cache.get(np.array([0.0, 1.0]))  # miss
        cache.clear()
        s = cache.stats()
        self.assertEqual(s["size"], 0)
        self.assertEqual(s["hits"], 1)
        self.assertEqual(s["misses"], 1)


# ============================================================
# 线程安全测试
# ============================================================
class TestSchedulerCacheThreadSafety(unittest.TestCase):
    """测试多线程并发访问的安全性。"""

    def test_concurrent_put_get_no_error(self):
        """多线程并发 put/get 不应抛出异常。"""
        cache = SchedulerCache(max_size=200, similarity_threshold=0.99)
        errors: list[Exception] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(100):
                    state = np.array([float(worker_id * 100 + i), 0.0])
                    cache.put(state, worker_id)
                    cache.get(state)
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(wid,)) for wid in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])

    def test_concurrent_respects_max_size(self):
        """并发写入后缓存大小不应超过 max_size。"""
        max_size = 100
        cache = SchedulerCache(max_size=max_size, similarity_threshold=0.99)

        def worker(worker_id: int) -> None:
            for i in range(50):
                # 各 worker 使用不同维度空间避免相似度误命中
                state = np.array([float(worker_id), float(i)])
                cache.put(state, worker_id * 1000 + i)

        threads = [threading.Thread(target=worker, args=(wid,)) for wid in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertLessEqual(len(cache), max_size)

    def test_concurrent_stats_consistent(self):
        """并发访问后 stats 计数应一致（hits + misses = 总访问数）。"""
        cache = SchedulerCache(max_size=500, similarity_threshold=0.99)
        total_gets = 200

        def worker() -> None:
            for i in range(total_gets):
                cache.get(np.array([float(i), 0.0]))

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        s = cache.stats()
        # 4 worker × 200 gets = 800 次访问，hits + misses 应等于 800
        self.assertEqual(s["hits"] + s["misses"], 4 * total_gets)

    def test_concurrent_clear_and_put(self):
        """并发 clear 与 put 不应导致数据结构损坏。"""
        cache = SchedulerCache(max_size=50, similarity_threshold=0.99)
        errors: list[Exception] = []

        def putter() -> None:
            try:
                for i in range(200):
                    cache.put(np.array([float(i), float(i)]), i)
            except Exception as exc:
                errors.append(exc)

        def clearer() -> None:
            try:
                for _ in range(20):
                    cache.clear()
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=putter) for _ in range(3)]
        threads.append(threading.Thread(target=clearer))
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(errors, [])
        self.assertLessEqual(len(cache), 50)


# ============================================================
# 边界情况测试
# ============================================================
class TestSchedulerCacheEdgeCases(unittest.TestCase):
    """测试空状态、全零状态、单元素状态等边界情况。"""

    def test_empty_state(self):
        """空状态向量应能正常存取（精确匹配快速路径）。"""
        cache = SchedulerCache(similarity_threshold=0.95)
        empty = np.array([], dtype=np.float64)
        cache.put(empty, action=0)
        self.assertEqual(len(cache), 1)
        # 空向量范数为 0，相似度返回 0，但精确匹配快速路径应命中
        result = cache.get(empty)
        self.assertEqual(result, 0)

    def test_all_zero_state(self):
        """全零状态向量应能正常存取（精确匹配快速路径）。"""
        cache = SchedulerCache(similarity_threshold=0.95)
        zero = np.array([0.0, 0.0, 0.0])
        cache.put(zero, action=2)
        # 全零向量余弦相似度无法定义（返回 0），但精确匹配应命中
        result = cache.get(zero)
        self.assertEqual(result, 2)

    def test_all_zero_state_no_false_similarity_hit(self):
        """全零查询状态不应通过相似度误命中非零条目。"""
        cache = SchedulerCache(similarity_threshold=0.5)
        cache.put(np.array([1.0, 0.0]), action=1)
        # 查询全零向量：与 [1,0] 余弦相似度为 0（零向量保护）
        result = cache.get(np.array([0.0, 0.0]))
        self.assertIsNone(result)

    def test_single_element_state(self):
        """单元素状态向量应能正常存取。"""
        cache = SchedulerCache(similarity_threshold=0.99)
        cache.put(np.array([5.0]), action=1)
        result = cache.get(np.array([5.0]))
        self.assertEqual(result, 1)

    def test_single_element_similar_hits(self):
        """单元素正数状态间余弦相似度为 1，应相互命中。"""
        cache = SchedulerCache(similarity_threshold=0.9)
        cache.put(np.array([3.0]), action=1)
        # cos([3], [5]) = 1（同向正数）
        result = cache.get(np.array([5.0]))
        self.assertEqual(result, 1)

    def test_single_element_opposite_misses(self):
        """单元素反方向状态相似度为 -1，不应命中。"""
        cache = SchedulerCache(similarity_threshold=0.9)
        cache.put(np.array([3.0]), action=1)
        # cos([3], [-5]) = -1
        result = cache.get(np.array([-5.0]))
        self.assertIsNone(result)

    def test_high_dimensional_state(self):
        """高维状态向量应能正常工作。"""
        cache = SchedulerCache(similarity_threshold=0.99)
        state = np.linspace(0.0, 1.0, num=100)
        cache.put(state, action=42)
        self.assertEqual(cache.get(state), 42)

    def test_integer_state_input(self):
        """整数类型状态输入应被接受（内部转 float64）。"""
        cache = SchedulerCache(similarity_threshold=0.99)
        cache.put(np.array([1, 0, 0]), action=1)
        # 整数与浮点表示应能精确命中（均归一为 float64 bytes）
        result = cache.get(np.array([1.0, 0.0, 0.0]))
        self.assertEqual(result, 1)

    def test_zero_action_value(self):
        """action=0 应能被正确缓存与返回（不与 None 混淆）。"""
        cache = SchedulerCache(similarity_threshold=0.99)
        cache.put(np.array([1.0, 0.0]), action=0)
        result = cache.get(np.array([1.0, 0.0]))
        self.assertIsNotNone(result)
        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
