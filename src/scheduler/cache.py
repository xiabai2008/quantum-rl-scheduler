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
- 线程安全：通过细粒度锁策略保护 dict/list 操作，numpy 运算在锁外执行

性能优化（Issue #363）：
- 精确匹配优先：bytes 哈希 O(1) 命中后提前返回，跳过相似度扫描
- 慢路径向量化：miss 时将同维度条目堆叠为矩阵，用一次 matmul + 一次批量
  norm 替代逐条 np.dot/np.linalg.norm 调用，查询向量范数仅计算一次
- 同维度过滤：跨维度条目余弦相似度定义为 0.0，扫描时直接跳过
- 周期日志：每 _LOG_INTERVAL 次访问输出一次命中率统计，便于评估缓存收益

锁粒度优化（Issue #738）：
- get 慢路径采用三阶段锁策略：持锁做精确匹配+快照收集，锁外做 numpy
  相似度矩阵运算，再持锁确认命中条目状态。避免 numpy 运算期间阻塞所有线程。

TTL 主动清理（Issue #737）：
- get 慢路径收集快照时顺便清理所有过期条目（不仅是查询条目）
- put 每 cleanup_interval 次触发一次全量过期扫描
- 解决仅惰性清理导致缓存空间被过期条目占用的问题
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

# TTL 过期条目主动清理间隔（每 N 次 put 触发一次全量过期扫描，Issue #737）
_DEFAULT_CLEANUP_INTERVAL = 64

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
        cleanup_interval     : TTL 主动清理间隔（每 N 次 put 触发一次全量过期扫描）

    Attributes:
        无公开属性，请通过 stats()/__len__() 查询运行状态。
    """

    def __init__(
        self,
        max_size: int = 1000,
        similarity_threshold: float = 0.95,
        ttl_seconds: float = 300.0,
        cleanup_interval: int = _DEFAULT_CLEANUP_INTERVAL,
    ) -> None:
        """
        初始化调度决策缓存。

        Args:
            max_size             : 缓存最大条目数（必须 > 0）
            similarity_threshold : 余弦相似度命中阈值，范围 [0, 1]
            ttl_seconds          : 条目生存时间（秒，必须 > 0）
            cleanup_interval     : TTL 主动清理间隔（每 N 次 put 触发一次全量
                过期扫描，必须 > 0，默认 _DEFAULT_CLEANUP_INTERVAL）
        """
        if max_size <= 0:
            raise ValueError(f"max_size 必须为正整数，收到 {max_size}")
        if not 0.0 <= similarity_threshold <= 1.0:
            raise ValueError(
                f"similarity_threshold 必须在 [0, 1] 范围内，收到 {similarity_threshold}"
            )
        if ttl_seconds <= 0.0:
            raise ValueError(f"ttl_seconds 必须为正数，收到 {ttl_seconds}")
        if cleanup_interval <= 0:
            raise ValueError(f"cleanup_interval 必须为正整数，收到 {cleanup_interval}")

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
        # put 计数（用于周期性 TTL 过期清理触发，Issue #737）
        self._put_count: int = 0
        # TTL 主动清理间隔（每 N 次 put 触发一次全量过期扫描）
        self._cleanup_interval: int = cleanup_interval

    # ------------------------------------------------------------------
    # 公开接口
    # ------------------------------------------------------------------
    def get(self, state: NDArray[Any]) -> int | None:
        """
        查找相似状态的缓存决策。

        先尝试精确匹配（bytes 相等且未过 TTL），再进行余弦相似度扫描。
        命中则将对应条目标记为最近访问（LRU move_to_end）并返回 action；
        未命中返回 None 并累加 miss 计数。

        性能特性（Issue #363 / #738）：
            - 精确命中走 O(1) 哈希快速路径，无 numpy 计算
            - 慢路径采用三阶段细粒度锁策略：持锁做精确匹配+快照收集+过期
              清理，锁外做 numpy 相似度矩阵运算，再持锁确认命中条目状态。
              避免 numpy 运算期间阻塞并发 put/get（Issue #738）
            - 慢路径收集快照时主动清理所有过期条目（Issue #737）
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
        # 慢路径快照：在持锁阶段收集，用于锁外的 numpy 运算
        same_dim_states: list[NDArray[Any]] = []
        same_dim_keys: list[bytes] = []
        result: int | None = None
        hit = False

        # ---- 阶段1：持锁 - 精确匹配 + 快照收集 + 顺便清理过期条目（Issue #737）----
        with self._lock:
            self._access_count += 1

            # 快速路径：精确匹配
            entry = self._cache.get(key)
            if entry is not None:
                action, _cached_state, ts = entry
                if now - ts <= self._ttl_seconds:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    result = action
                    hit = True
                else:
                    # 精确匹配但 TTL 过期：移除后继续相似度扫描
                    self._cache.pop(key, None)

            if not hit:
                # 收集同维度未过期条目快照，同时主动清理所有过期条目（Issue #737）
                for k, (_, cached_state, ts) in list(self._cache.items()):
                    if now - ts > self._ttl_seconds:
                        self._cache.pop(k, None)
                        continue
                    if cached_state.shape == flat.shape:
                        same_dim_keys.append(k)
                        same_dim_states.append(cached_state)

            # 周期性命中率日志
            if self._access_count % _LOG_INTERVAL == 0:
                total = self._hits + self._misses
                rate: float = self._hits / total if total > 0 else 0.0
                log_payload = {
                    "hits": self._hits,
                    "misses": self._misses,
                    "hit_rate": rate,
                    "size": len(self._cache),
                }

        # 日志输出（锁外，避免持锁时 I/O 阻塞）
        if log_payload is not None:
            _logger.info(
                "SchedulerCache 周期统计: hits=%d misses=%d hit_rate=%.4f size=%d "
                "— 低命中率下慢路径开销显著，建议提高命中率或缩小 max_size",
                log_payload["hits"],
                log_payload["misses"],
                log_payload["hit_rate"],
                log_payload["size"],
            )

        if hit:
            return result

        # ---- 阶段2：锁外 - numpy 相似度矩阵运算（不阻塞并发读写，Issue #738）----
        # 查询向量范数仅计算一次
        norm_a = float(np.linalg.norm(flat))
        # 零向量查询：余弦相似度无定义，无法通过相似度命中
        if norm_a < _EPSILON:
            with self._lock:
                self._misses += 1
            return None

        if not same_dim_states:
            with self._lock:
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
        if best_sim < self._similarity_threshold:
            with self._lock:
                self._misses += 1
            return None

        # ---- 阶段3：持锁 - 确认命中条目仍未过期且未被淘汰 ----
        best_key = same_dim_keys[best_idx]
        with self._lock:
            entry = self._cache.get(best_key)
            if entry is None:
                # 条目在阶段2期间被淘汰或清理
                self._misses += 1
                return None
            action, _, ts = entry
            if now - ts > self._ttl_seconds:
                # 条目在阶段2期间过期
                self._cache.pop(best_key, None)
                self._misses += 1
                return None
            self._cache.move_to_end(best_key)
            self._hits += 1
            return action

    def put(self, state: NDArray[Any], action: int) -> None:
        """
        存入一条调度决策缓存。

        若状态已存在（bytes 精确相等）则更新 action 与时间戳并标记为最近访问；
        否则新增条目。当缓存大小超过 max_size 时，按 LRU 策略淘汰最久未访问
        的条目并累加 evictions 计数。

        每 cleanup_interval 次 put 会触发一次全量 TTL 过期清理（Issue #737），
        主动移除所有过期条目，避免缓存空间被无效条目占用。

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

            # Issue #737: 周期性主动清理所有过期条目
            self._put_count += 1
            if self._put_count % self._cleanup_interval == 0:
                self._cleanup_expired_locked(now)

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
    def _cleanup_expired_locked(self, now: float) -> None:
        """
        主动清理所有 TTL 过期条目（调用方须持有 self._lock）。

        Issue #737: 配合 get 慢路径的顺便清理，put 周期性触发全量扫描，
        避免仅惰性清理导致缓存空间被过期条目占用。

        Args:
            now: 当前单调时钟时间戳
        """
        expired_keys = [k for k, (_, _, ts) in self._cache.items() if now - ts > self._ttl_seconds]
        for k in expired_keys:
            self._cache.pop(k, None)

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
