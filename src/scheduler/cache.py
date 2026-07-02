"""
调度决策缓存模块
Scheduler Decision Cache Module

RL 推理（model.predict）在相似状态下会重复计算，本模块提供一个线程安全的
LRU + TTL + 余弦相似度缓存，用于复用相似状态的决策结果以降低推理延迟。

设计要点：
- 缓存键：状态向量 flatten 后的 bytes（用于精确匹配快速路径与 LRU 顺序维护）
- 相似度匹配：对缓存中的状态向量做余弦相似度扫描，命中阈值则返回缓存的 action
- LRU 淘汰：基于 OrderedDict，超容量时移除最久未访问的条目
- TTL 过期：每个条目记录写入时间戳，get 时校验是否在 TTL 有效期内
- 线程安全：所有公开方法通过 threading.Lock 串行化
"""

import threading
import time
from collections import OrderedDict

import numpy as np

# ---------------------------------------------------------------------------
# 缓存条目类型：(action, cached_flat_state, put_timestamp)
# ---------------------------------------------------------------------------
_CacheEntry = tuple[int, np.ndarray, float]

# 零向量范数保护阈值，避免除零
_EPSILON = 1e-12


class SchedulerCache:
    """
    调度决策缓存（线程安全、LRU + TTL + 余弦相似度）。

    用于缓存 RL 智能体在相似状态下的决策结果，减少重复推理耗时。
    查找策略：
        1. 快速路径：状态向量 bytes 精确命中且未过 TTL，直接返回
        2. 慢速路径：遍历缓存计算余弦相似度，取最高相似度条目，
           若 >= similarity_threshold 且未过 TTL，返回该条目 action

    Args:
        max_size             : 缓存最大条目数，超出后按 LRU 淘汰
        similarity_threshold : 相似度命中阈值（0-1），越高越严格
        ttl_seconds          : 条目生存时间（秒），超过则视为过期

    Attributes:
        无公开属性，请通过 stats()/__len__() 查询运行状态。
    """

    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.95,
        ttl_seconds: float = 300.0,
    ) -> None:
        """
        初始化调度决策缓存。

        Args:
            max_size             : 缓存最大条目数（必须 > 0）
            similarity_threshold : 余弦相似度命中阈值，范围 [0, 1]
            ttl_seconds          : 条目生存时间（秒，必须 > 0）
        """
        if max_size <= 0:
            raise ValueError(f"max_size 必须为正整数，收到 {max_size}")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold 必须在 [0, 1] 范围内，收到 {similarity_threshold}"
            )
        if ttl_seconds <= 0.0:
            raise ValueError(f"ttl_seconds 必须为正数，收到 {ttl_seconds}")

        self._max_size: int = max_size
        self._similarity_threshold: float = similarity_threshold
        self._ttl_seconds: float = ttl_seconds

        # OrderedDict 维护 LRU 顺序：末尾为最近访问，头部为最久未访问
        self._cache: OrderedDict[bytes, _CacheEntry] = OrderedDict()
        self._lock: threading.Lock = threading.Lock()

        # 统计计数器
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def get(self, state: np.ndarray) -> int | None:
        """
        查找相似状态的缓存决策。

        先尝试精确匹配（bytes 相等且未过 TTL），再进行余弦相似度扫描。
        命中则将对应条目标记为最近访问（LRU move_to_end）并返回 action；
        未命中返回 None 并累加 miss 计数。

        Args:
            state: RL 状态向量（任意形状，内部会 flatten 处理）

        Returns:
            命中时返回缓存的 action（int），未命中返回 None
        """
        flat = self._flatten(state)
        key = flat.tobytes()
        now = time.monotonic()

        with self._lock:
            # 快速路径：精确匹配
            entry = self._cache.get(key)
            if entry is not None:
                action, cached_state, ts = entry
                if now - ts <= self._ttl_seconds:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return action
                # 精确匹配但 TTL 过期：移除过期条目后继续相似度扫描
                self._cache.pop(key, None)

            # 慢速路径：余弦相似度扫描
            best_key: bytes | None = None
            best_sim: float = 0.0
            for k, (_, cached_state, _) in self._cache.items():
                sim = self._cosine_similarity(flat, cached_state)
                if sim > best_sim:
                    best_sim = sim
                    best_key = k

            if best_key is not None and best_sim >= self._similarity_threshold:
                action, _, ts = self._cache[best_key]
                if now - ts <= self._ttl_seconds:
                    self._cache.move_to_end(best_key)
                    self._hits += 1
                    return action

            self._misses += 1
            return None

    def put(self, state: np.ndarray, action: int) -> None:
        """
        存入一条调度决策缓存。

        若状态已存在（bytes 精确相等）则更新 action 与时间戳并标记为最近访问；
        否则新增条目。当缓存大小超过 max_size 时，按 LRU 策略淘汰最久未访问
        的条目并累加 evictions 计数。

        Args:
            state : RL 状态向量（任意形状，内部会 flatten 处理）
            action: 缓存的决策动作（int）
        """
        flat = self._flatten(state)
        key = flat.tobytes()
        now = time.monotonic()

        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = (action, flat, now)

            # LRU 淘汰：从头部移除最久未访问的条目
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)
                self._evictions += 1

    def clear(self) -> None:
        """
        清空缓存条目。

        仅清除缓存的决策条目，保留累计的统计计数器（hits/misses/evictions），
        以便观察缓存整个生命周期的命中情况。
        """
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict[str, int | float]:
        """
        返回缓存运行统计信息。

        Returns:
            包含以下键的字典：
                - hits      : 命中次数（int）
                - misses    : 未命中次数（int）
                - hit_rate  : 命中率（float，0-1，无访问时为 0.0）
                - size      : 当前缓存条目数（int）
                - evictions : LRU 淘汰次数（int）
        """
        with self._lock:
            total = self._hits + self._misses
            hit_rate: float = self._hits / total if total > 0 else 0.0
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "size": len(self._cache),
                "evictions": self._evictions,
            }

    def __len__(self) -> int:
        """返回当前缓存条目数。"""
        with self._lock:
            return len(self._cache)

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------
    @staticmethod
    def _flatten(state: np.ndarray) -> np.ndarray:
        """
        将状态向量转换为 float64 一维数组。

        Args:
            state: 任意形状的 numpy 数组

        Returns:
            float64 一维数组（flatten 后的副本）
        """
        flat: np.ndarray = np.asarray(state, dtype=np.float64).flatten()
        return flat

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """
        计算两个一维向量的余弦相似度。

        cos(a, b) = dot(a, b) / (||a|| * ||b||)

        边界处理：
            - 形状不一致：返回 0.0（视为不相似）
            - 任一向量为零向量（范数 < _EPSILON）：返回 0.0（避免除零）

        Args:
            a: 一维 float64 向量
            b: 一维 float64 向量

        Returns:
            余弦相似度（float，范围 -1 到 1）
        """
        if a.shape != b.shape:
            return 0.0
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a < _EPSILON or norm_b < _EPSILON:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
