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

性能优化（Issue #363）：
- 精确匹配优先：bytes 哈希 O(1) 命中后提前返回，跳过相似度扫描
- 慢路径向量化：miss 时将同维度条目堆叠为矩阵，用一次 matmul + 一次批量
  norm 替代逐条 np.dot/np.linalg.norm 调用，查询向量范数仅计算一次
- 同维度过滤：跨维度条目余弦相似度定义为 0.0，扫描时直接跳过
- 周期日志：每 _LOG_INTERVAL 次访问输出一次命中率统计，便于评估缓存收益
"""

import logging
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

# 命中率统计日志输出间隔（每 N 次 get 访问输出一次）
_LOG_INTERVAL = 1000

_logger = logging.getLogger(__name__)


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
        # get 访问计数（用于周期性命中率日志触发）
        self._access_count: int = 0

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def get(self, state: NDArray[Any]) -> int | None:
        """
        查找相似状态的缓存决策。

        先尝试精确匹配（bytes 相等且未过 TTL），再进行余弦相似度扫描。
        命中则将对应条目标记为最近访问（LRU move_to_end）并返回 action；
        未命中返回 None 并累加 miss 计数。

        性能特性（Issue #363）：
            - 精确命中走 O(1) 哈希快速路径，无 numpy 计算
            - 未命中时慢路径为 O(N·d) 但已向量化（单次 matmul + 批量 norm），
              查询向量范数仅计算一次；命中率低时慢路径开销仍显著，
              建议在低命中率场景缩小 max_size 或关闭缓存
            - 每 _LOG_INTERVAL 次访问输出一次命中率统计日志

        Args:
            state: RL 状态向量（任意形状，内部会 flatten 处理）

        Returns:
            命中时返回缓存的 action（int），未命中返回 None
        """
        flat = self._flatten(state)
        key = flat.tobytes()
        now = time.monotonic()

        log_payload: dict[str, int | float] | None = None
        with self._lock:
            self._access_count += 1
            result = self._lookup(flat, key, now)
            if self._access_count % _LOG_INTERVAL == 0:
                total = self._hits + self._misses
                rate: float = self._hits / total if total > 0 else 0.0
                log_payload = {
                    "hits": self._hits,
                    "misses": self._misses,
                    "hit_rate": rate,
                    "size": len(self._cache),
                }

        if log_payload is not None:
            _logger.info(
                "SchedulerCache 周期统计: hits=%d misses=%d hit_rate=%.4f size=%d "
                "— 低命中率下慢路径开销显著，建议提高命中率或缩小 max_size",
                log_payload["hits"],
                log_payload["misses"],
                log_payload["hit_rate"],
                log_payload["size"],
            )
        return result

    def _lookup(self, flat: NDArray[Any], key: bytes, now: float) -> int | None:
        """
        执行缓存查找（调用方须持有 self._lock）。

        查找顺序：
            1. 精确匹配：bytes 哈希命中且未过 TTL → 提前返回
            2. 慢路径：向量化余弦相似度扫描同维度条目，取最高相似度，
               若 >= similarity_threshold 且未过 TTL 则返回

        慢路径向量化策略：
            - 查询向量范数 norm_a 仅计算一次（原实现每条目重复计算）
            - 同维度条目堆叠为矩阵 M，dots = M @ flat 单次 matmul
            - norms = np.linalg.norm(M, axis=1) 单次批量计算
            - 跨维度条目相似度定义为 0.0，扫描时直接跳过
            - tie-breaking 复刻原语义：best_sim 初值 0.0 + 严格 > 更新，
              故仅正相似度可被选中（等价于将非正相似度屏蔽为 -inf 后取 argmax）

        Args:
            flat: 已 flatten 的 float64 一维状态向量
            key : flat.tobytes() 精确匹配键
            now : 当前单调时钟时间戳

        Returns:
            命中时返回缓存的 action（int），未命中返回 None
        """
        # 快速路径：精确匹配
        entry = self._cache.get(key)
        if entry is not None:
            action, _cached_state, ts = entry
            if now - ts <= self._ttl_seconds:
                self._cache.move_to_end(key)
                self._hits += 1
                return action
            # 精确匹配但 TTL 过期：移除过期条目后继续相似度扫描
            self._cache.pop(key, None)

        # 慢速路径：余弦相似度批量扫描（向量化，避免逐条 numpy 调用开销）
        # 查询向量范数仅计算一次（原实现每条目重复计算一次 np.linalg.norm(flat)）
        norm_a = float(np.linalg.norm(flat))
        # 零向量查询：余弦相似度无定义，无法通过相似度命中
        if norm_a < _EPSILON:
            self._misses += 1
            return None

        # 仅扫描同维度条目（跨维度相似度定义为 0.0，必不命中）
        # 保持 OrderedDict 迭代顺序以复刻原 tie-breaking 语义
        same_dim_states: list[NDArray[Any]] = []
        same_dim_keys: list[bytes] = []
        for k, (_, cached_state, _) in self._cache.items():
            if cached_state.shape == flat.shape:
                same_dim_keys.append(k)
                same_dim_states.append(cached_state)

        if not same_dim_states:
            self._misses += 1
            return None

        # 单次 matmul + 单次批量 norm 替代逐条 np.dot / np.linalg.norm
        states_mat = np.stack(same_dim_states)
        dots = states_mat @ flat
        norms = np.linalg.norm(states_mat, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sims = np.where(norms >= _EPSILON, dots / (norm_a * norms), -np.inf)
        # 原实现 best_sim 初值 0.0 且用严格 > 更新，仅正相似度可被选中
        sims = np.where(sims > 0.0, sims, -np.inf)

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        if best_sim >= self._similarity_threshold:
            best_key = same_dim_keys[best_idx]
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
