"""调度决策缓存性能基准 — 命中率-延迟曲线（Issue #363）

验证目标：
    明确 SchedulerCache 仅在高命中率场景下对推理热路径有益。
    缓存 miss 时 get() 走向量化余弦相似度慢路径（O(N·d)），命中率越低，
    慢路径占比越高，单次 get 延迟越高。

命中率-延迟曲线：
    本模块以参数化基准给出曲线数据点。运行方式：

        python -m pytest tests/benchmarks/test_scheduler_cache_benchmark.py \
            --benchmark-only \
            --benchmark-columns=min,mean,median,max,ops

    预期观察（曲线解读）：
        - hit_rate=1.0：全部精确命中快速路径，mean 延迟最低
        - hit_rate=0.9：10% miss 触发慢路径扫描，mean 略升
        - hit_rate=0.5：半数 miss，mean 显著上升
        - hit_rate=0.0：全部 miss，每次 get 都扫描全量 N 条目，mean 最高
        - dim=128 相比 dim=16：miss 路径延迟随维度 d 上升（O(N·d)）

    结论：缓存收益与命中率强相关；命中率低时慢路径开销可能抵消甚至超过
    缓存带来的收益，此时应缩小 max_size 或直接关闭缓存。

覆盖维度：
    - hit_rate ∈ {0.0, 0.5, 0.9, 1.0}：控制精确命中占比
    - dim ∈ {16, 128}：16 为原生观测维度（OBS_DIM=16），128 为高维场景
    - max_size=500：典型缓存容量，miss 时扫描 500 条目
"""

import os
import sys
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.scheduler.cache import SchedulerCache


def _get_stat(stats: Any, key: str, default: float = 0.0) -> float:
    """兼容 pytest-benchmark 4.x/5.x 的 stats 属性访问。

    4.x: stats.mean / stats.median（直接属性）
    5.x: stats 是 Metadata 对象，需通过 stats['mean'] 或 stats.stats.mean 访问；
         在某些 5.x 版本下 stats 可能为 None，此时返回 default。
    """
    # 4.x: 直接属性访问
    val = getattr(stats, key, None)
    if val is not None:
        return val
    # 5.x: dict-like 访问
    try:
        return stats[key]
    except (KeyError, TypeError, IndexError):
        pass
    # 5.x: 嵌套 Stats 对象
    inner = getattr(stats, "stats", None)
    if inner is not None:
        val = getattr(inner, key, None)
        if val is not None:
            return val
    return default


def _build_populated_cache(
    dim: int, n_entries: int, seed: int
) -> tuple[SchedulerCache, list[np.ndarray]]:
    """构造一个已填充 n_entries 条目的缓存，返回 (cache, base_states)。

    base_states 为已写入缓存的原始状态列表，查询其中任一状态均精确命中。

    Args:
        dim       : 状态向量维度
        n_entries : 预填充条目数
        seed      : 随机种子

    Returns:
        (SchedulerCache, base_states)
    """
    # similarity_threshold 设为 0.999：随机 miss 查询（独立高斯向量）在 dim>=16
    # 下余弦相似度集中于 0，几乎不可能 >= 0.999，确保 miss 不被误判为相似命中
    cache = SchedulerCache(max_size=n_entries, similarity_threshold=0.999, ttl_seconds=1e9)
    rng = np.random.default_rng(seed)
    base_states: list[np.ndarray] = []
    for i in range(n_entries):
        state = rng.standard_normal(dim)
        cache.put(state, i)
        base_states.append(state)
    return cache, base_states


@pytest.mark.benchmark
class TestSchedulerCacheBenchmark:
    """SchedulerCache.get() 命中率-延迟曲线基准。"""

    @pytest.mark.parametrize("hit_rate", [0.0, 0.5, 0.9, 1.0])
    @pytest.mark.parametrize("dim", [16, 128])
    def test_get_latency_vs_hit_rate(self, benchmark, hit_rate: float, dim: int) -> None:
        """测量不同命中率下的单次 get() 延迟，构成命中率-延迟曲线。

        Args:
            benchmark : pytest-benchmark 提供的基准 fixture
            hit_rate  : 精确命中占比（0=全 miss，1=全 hit）
            dim       : 状态向量维度
        """
        n_entries = 500
        cache, base_states = _build_populated_cache(dim=dim, n_entries=n_entries, seed=42)
        rng = np.random.default_rng(123)

        # 预生成查询序列：hit_rate 比例为已缓存状态的精确重复（快速路径命中），
        # 其余为全新随机向量（慢路径 miss，触发 500 条目向量化扫描）
        n_queries = 1000
        queries: list[np.ndarray] = []
        for _ in range(n_queries):
            if rng.random() < hit_rate:
                queries.append(base_states[int(rng.integers(0, n_entries))])
            else:
                queries.append(rng.standard_normal(dim))

        idx = [0]

        def get_once() -> int | None:
            query = queries[idx[0] % n_queries]
            idx[0] += 1
            return cache.get(query)

        result = benchmark(get_once)

        # 轻量断言：get 返回值为 int（命中）或 None（未命中），缓存大小不超 max_size
        assert result is None or isinstance(result, int)
        assert len(cache) <= n_entries
        # 性能回归阈值断言（Issue #729）：缓存 get() 中位数应 < 100ms（含全 miss 慢路径）
        # pytest-benchmark 5.x 兼容：stats 可能为 None / Metadata 对象，用 _get_stat 安全访问
        median = _get_stat(benchmark.stats, "median", default=0.0)
        assert median < 0.1, (
            f"cache.get() 中位数超阈值 (hit_rate={hit_rate}, dim={dim}): {median:.4f}s"
        )
