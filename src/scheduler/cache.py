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
- 向量化相似度：预分配 _state_matrix，慢速路径用 numpy 矩阵运算一次性计算
  所有缓存条目与查询向量的余弦相似度，消除 O(n) Python 循环
"""

import threading
import time
from collections import OrderedDict
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# 缓存条目类型：(action, cached_flat_state, put_timestamp)
# ---------------------------------------------------------------------------
_CacheEntry = tuple[int, NDArray[Any], float]

# 零向量范数保护阈值，避免除零
_EPSILON = 1e-12


class SchedulerCache:
    """
    调度决策缓存（线程安全、LRU + TTL + 余弦相似度）。

    用于缓存 RL 智能体在相似状态下的决策结果，减少重复推理耗时。
    查找策略：
        1. 快速路径：状态向量 bytes 精确命中且未过 TTL，直接返回
        2. 慢速路径：通过 numpy 矩阵向量化运算一次性计算所有缓存条目
           与查询向量的余弦相似度，取最高相似度条目，
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

        # 向量化相似度支持：预分配状态矩阵（首次 put 时惰性初始化）
        self._state_matrix: NDArray[Any] | None = None
        self._dim: int = 0
        self._size: int = 0
        self._key_to_index: dict[bytes, int] = {}
        self._index_to_key: dict[int, bytes] = {}

        # 统计计数器
        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def get(self, state: NDArray[Any]) -> int | None:
        """
        查找相似状态的缓存决策。

        先尝试精确匹配（bytes 相等且未过 TTL），再进行余弦相似度扫描。
        命中则将对应条目标记为最近访问（LRU move_to_end）并返回 action；
        未命中返回 None 并累加 miss 计数。

        慢速路径使用 numpy 矩阵向量化运算一次性计算所有缓存条目的余弦
        相似度，消除逐条 Python 循环。

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
                action, _, ts = entry
                if now - ts <= self._ttl_seconds:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    return action
                # 精确匹配但 TTL 过期：移除过期条目后继续相似度扫描
                self._cache.pop(key, None)
                self._remove_matrix_row_locked(key)

            # 慢速路径：向量化余弦相似度扫描
            if self._state_matrix is None or self._size == 0:
                self._misses += 1
                return None

            # 维度不匹配时无法计算相似度（与原实现行为一致）
            if flat.shape[0] != self._dim:
                self._misses += 1
                return None

            # 向量化余弦相似度计算
            active = self._state_matrix[: self._size]  # (n, dim)
            norms = np.linalg.norm(active, axis=1)  # (n,)
            dots = active @ flat  # (n,)
            query_norm = float(np.linalg.norm(flat))
            # 初始化为 -1.0，保证与原实现 best_sim=0.0 + 严格 > 比较语义一致
            sims = np.full(self._size, -1.0, dtype=np.float64)
            if query_norm >= _EPSILON:
                valid = norms >= _EPSILON
                sims[valid] = dots[valid] / (norms[valid] * query_norm)

            best_idx = int(np.argmax(sims))
            best_sim = float(sims[best_idx])

            if best_sim > 0.0 and best_sim >= self._similarity_threshold:
                best_key = self._index_to_key[best_idx]
                action, _, ts = self._cache[best_key]
                if now - ts <= self._ttl_seconds:
                    self._cache.move_to_end(best_key)
                    self._hits += 1
                    return action

            self._misses += 1
            return None

    def put(self, state: NDArray[Any], action: int) -> None:
        """
        存入一条调度决策缓存。

        若状态已存在（bytes 精确相等）则更新 action 与时间戳并标记为最近访问；
        否则新增条目。当缓存大小超过 max_size 时，按 LRU 策略淘汰最久未访问
        的条目并累加 evictions 计数。

        新状态会同步写入预分配的 _state_matrix，淘汰时采用 swap-last 策略
        避免数组移位。

        Args:
            state : RL 状态向量（任意形状，内部会 flatten 处理）
            action: 缓存的决策动作（int）
        """
        flat = self._flatten(state)
        key = flat.tobytes()
        now = time.monotonic()

        with self._lock:
            self._ensure_matrix(flat.shape[0])

            if key in self._cache:
                # 更新已存在条目
                self._cache.move_to_end(key)
                if key in self._key_to_index:
                    idx = self._key_to_index[key]
                    self._state_matrix[idx] = flat  # type: ignore[index]
            else:
                # 新增条目：先淘汰至有空位，再写入矩阵行
                while len(self._cache) >= self._max_size:
                    old_key, _ = self._cache.popitem(last=False)
                    self._evictions += 1
                    self._remove_matrix_row_locked(old_key)

                idx = self._size
                self._state_matrix[idx] = flat  # type: ignore[index]
                self._key_to_index[key] = idx
                self._index_to_key[idx] = key
                self._size += 1

            self._cache[key] = (action, flat, now)

    def clear(self) -> None:
        """
        清空缓存条目。

        仅清除缓存的决策条目，保留累计的统计计数器（hits/misses/evictions），
        以便观察缓存整个生命周期的命中情况。同时重置状态矩阵和行计数器。
        """
        with self._lock:
            self._cache.clear()
            self._key_to_index.clear()
            self._index_to_key.clear()
            self._size = 0
            self._state_matrix = None
            self._dim = 0

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
    def _ensure_matrix(self, dim: int) -> None:
        """
        惰性分配状态矩阵（首次 put 时调用）。

        Args:
            dim: 状态向量维度
        """
        if self._state_matrix is None:
            self._dim = dim
            self._state_matrix = np.zeros((self._max_size, dim), dtype=np.float64)

    def _remove_matrix_row_locked(self, key: bytes) -> None:
        """
        从状态矩阵中移除指定 key 对应的行（swap-last 策略）。

        将待删行与最后一行交换后截断 _size，避免数组整体移位。
        调用者必须已持有 self._lock。

        Args:
            key: 缓存条目的 bytes 键
        """
        evicted_idx = self._key_to_index.pop(key, None)
        if evicted_idx is None or self._state_matrix is None or self._size == 0:
            return
        last_idx = self._size - 1
        if evicted_idx < last_idx:
            # 将最后一行移动到被删行的位置
            self._state_matrix[evicted_idx] = self._state_matrix[last_idx]
            last_key = self._index_to_key.pop(last_idx)
            self._index_to_key[evicted_idx] = last_key
            self._key_to_index[last_key] = evicted_idx
        else:
            # 被删行恰为最后一行，直接移除
            self._index_to_key.pop(evicted_idx)
        self._size -= 1

    @staticmethod
    def _flatten(state: NDArray[Any]) -> NDArray[Any]:
        """
        将状态向量转换为 float64 一维数组。

        Args:
            state: 任意形状的 numpy 数组

        Returns:
            float64 一维数组（flatten 后的副本）
        """
        flat: NDArray[Any] = np.asarray(state, dtype=np.float64).flatten()
        return flat

    @staticmethod
    def _cosine_similarity(a: NDArray[Any], b: NDArray[Any]) -> float:
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
