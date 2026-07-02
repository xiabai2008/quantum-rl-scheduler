"""
熔断器模块
Circuit Breaker Module

实现熔断器模式（Circuit Breaker Pattern）：当连续调用失败达到阈值时
自动熔断，避免对故障服务的持续压力；经过恢复超时后进入 HALF_OPEN
状态放行一次试探性调用，根据结果决定恢复或继续熔断。

熔断器模式概述
----------------
熔断器模式是一种工程韧性（Resilience）设计模式，用于防止分布式系统中
故障级联扩散。其核心思想是：当对下游服务的调用连续失败达到阈值时，
"熔断"（断开）调用链路，让后续请求直接快速失败（Fast Fail），
而不是继续等待超时，从而保护系统资源不被耗尽。

经过一段恢复超时（recovery_timeout）后，熔断器进入 HALF_OPEN 状态，
放行一次试探性调用以探测下游服务是否恢复：
- 试探成功 → 熔断器完全恢复（CLOSED）
- 试探失败 → 重新熔断（OPEN），继续等待下一个恢复周期

三态转换图
----------
::

    ┌──────────┐  连续失败达阈值  ┌──────────┐
    │  CLOSED  │ ───────────────→ │   OPEN   │
    │ (正常放行) │                  │ (拒绝调用) │
    └──────────┘                  └──────────┘
         ↑                              │
         │ 试探成功                     │ 超过 recovery_timeout
         │                              ↓
         │                        ┌────────────┐
         └────────────────────────│ HALF_OPEN  │
                                  │(试探性放行) │
                                  └────────────┘
                                         │
                                         │ 试探失败
                                         ↓
                                  ┌──────────┐
                                  │   OPEN   │
                                  └──────────┘

何时使用
--------
- 调用外部依赖（HTTP API、数据库、第三方服务）且希望在故障时快速失败
- 需要避免故障级联扩散、雪崩效应的场景
- 需要与重试逻辑配合使用：重试负责瞬时故障恢复，熔断器负责持续性故障隔离

与重试逻辑的关系
----------------
熔断器与重试（Retry）是互补的两种韧性机制：
- **重试**：处理瞬时故障（如网络抖动），对单次请求透明
- **熔断器**：处理持续性故障（如服务下线），在一段时间内直接拒绝请求

典型组合：先经熔断器判断是否放行，放行后的请求再由重试逻辑处理瞬时失败；
当重试耗尽仍失败时，熔断器累加失败计数（本模块的 :meth:`CircuitBreaker.call`
已内置此反馈）。

参考
----
- Martin Fowler: CircuitBreaker (https://martinfowler.com/bliki/CircuitBreaker.html)
- 本项目中的使用：``src/api/tianyan_client.py`` 中 ``TianyanClient`` 内置
  熔断器保护对天衍云平台的请求
"""

import time
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from loguru import logger

from src.exceptions import CircuitOpenError

__all__ = ["CircuitBreaker", "CircuitState"]

T = TypeVar("T")


class CircuitState(Enum):
    """熔断器状态枚举

    表示熔断器在任意时刻所处的三态之一，状态间转换由 ``CircuitBreaker``
    内部根据调用结果与时间自动驱动。

    Attributes:
        CLOSED: 闭合（正常放行）。初始状态，所有调用直接放行；
            调用成功时清零失败计数，失败时累加失败计数。
        OPEN: 打开（熔断拒绝）。连续失败达到 ``failure_threshold`` 时进入；
            所有调用被直接拒绝并抛出 :class:`~src.exceptions.CircuitOpenError`，
            直到超过 ``recovery_timeout`` 才转入 ``HALF_OPEN``。
        HALF_OPEN: 半开（试探性恢复）。恢复超时到期后进入；
            仅放行一次试探性调用，根据其结果决定回到 ``CLOSED`` 或 ``OPEN``。
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """熔断器

    在函数调用外层包裹熔断逻辑，实现对下游服务故障的自动隔离与恢复。

    状态机
    ------
    - **CLOSED**：正常放行；失败时累加计数，达到 ``failure_threshold`` 转为 OPEN
    - **OPEN**：拒绝调用并抛出 :class:`~src.exceptions.CircuitOpenError`；
      超过 ``recovery_timeout`` 后转为 HALF_OPEN
    - **HALF_OPEN**：放行一次试探性调用；成功则重置为 CLOSED，失败则重回 OPEN

    使用场景
    --------
    适用于包裹对外部依赖（如天衍云 API、数据库、第三方服务）的调用，
    在下游持续故障时快速失败，避免资源被长时间占用或故障扩散。

    Args:
        failure_threshold: 连续失败触发熔断的阈值。值越小越敏感（更快熔断），
            值越大越宽容（容忍更多瞬时抖动）。典型值 5~10。
        recovery_timeout: OPEN 状态恢复到 HALF_OPEN 的等待秒数。值过短会
            频繁试探故障服务，值过长会延迟恢复。典型值 30~120 秒。

    Attributes:
        failure_threshold: 连续失败触发熔断的阈值。
        recovery_timeout: OPEN 状态恢复到 HALF_OPEN 的等待秒数（单调时钟）。
        state: 当前熔断状态（:class:`CircuitState`）。
        failure_count: 当前连续失败计数（CLOSED 状态下累加，成功时清零）。
        last_failure_time: 最近一次失败的单调时间戳（``time.monotonic()``），
            用于判断 OPEN 状态是否已过恢复超时。

    基本用法
    --------
    直接包裹任意可调用对象::

        >>> from src.api.circuit_breaker import CircuitBreaker
        >>> cb = CircuitBreaker(failure_threshold=3, recovery_timeout=60.0)
        >>> # 正常调用
        >>> result = cb.call(lambda x: x + 1, 41)
        >>> result
        42
        >>> # 查询当前状态
        >>> cb.state.value
        'closed'
        >>> cb.is_available()
        True

    与 TianyanClient 配合使用
    -------------------------
    本项目 ``TianyanClient`` 内部已集成熔断器（默认启用），无需手动构造本类。
    如需手动使用本类包裹其他客户端调用，可参考以下模式::

        >>> from src.api.circuit_breaker import CircuitBreaker
        >>> from src.exceptions import CircuitOpenError
        >>> cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)
        >>> def call_api(endpoint):
        ...     return cb.call(requests.get, endpoint)
        >>> try:
        ...     call_api("https://api.tianyanyun.cn/v1/tasks")
        ... except CircuitOpenError:
        ...     print("熔断器已打开，请求被拒绝，请稍后重试")

    在测试中禁用熔断器
    ------------------
    测试场景下若不希望熔断器介入，有两种方式：

    1. 直接不使用 ``CircuitBreaker``（最干净的方式）。
    2. 将 ``failure_threshold`` 设置为极大值，使熔断器永远不会触发::

        >>> cb = CircuitBreaker(failure_threshold=10**9, recovery_timeout=0.0)
        >>> # 此配置下熔断器形同"始终放行"

    注：``TianyanClient`` 通过构造参数 ``enable_circuit_breaker=False``
    可直接禁用熔断器，无需手动构造本类。

    编程式查询状态
    --------------
    可通过以下属性/方法在运行时查询熔断器状态，便于监控与告警::

        >>> cb.state.value          # 'closed' / 'open' / 'half_open'
        'closed'
        >>> cb.is_available()       # 当前是否放行调用（布尔语义）
        True
        >>> cb.failure_count        # 当前连续失败计数
        0

    与重试逻辑的关系
    ----------------
    本类的 :meth:`call` 方法本身不重试，仅负责状态判断与结果反馈。
    外层调用方可在 ``call`` 之外自行实现重试；当 ``func`` 抛出异常时，
    :meth:`call` 会自动累加失败计数（驱动状态机），成功时清零计数。
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        """初始化熔断器

        Args:
            failure_threshold: 连续失败触发熔断的阈值，默认 5。
                达到该阈值时 CLOSED 状态转为 OPEN。
                - 调大：更宽容，容忍更多瞬时抖动（适合网络不稳定的场景）
                - 调小：更敏感，更快熔断（适合故障代价高的场景）
            recovery_timeout: OPEN 状态恢复到 HALF_OPEN 的等待秒数，默认 60.0。
                使用单调时钟（``time.monotonic``）计时，不受系统时间回拨影响。
                - 调大：恢复更慢，对故障服务更"耐心"
                - 调小：恢复更快，但可能频繁试探故障服务

        Note:
            初始状态为 :attr:`CircuitState.CLOSED`，失败计数为 0。
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """通过熔断器执行函数

        根据当前熔断状态决定是否放行调用，并自动记录成功/失败结果以驱动状态机。

        状态机行为
        ----------
        - **OPEN**：若已过 ``recovery_timeout`` 则转入 HALF_OPEN 并放行一次试探；
          否则直接抛出 :class:`~src.exceptions.CircuitOpenError`，不执行 ``func``。
        - **HALF_OPEN**：放行一次试探性调用。
          - 成功 → 调用 :meth:`reset` 回到 CLOSED
          - 失败 → 回到 OPEN，重新计时
        - **CLOSED**：放行调用。
          - 成功 → 清零 ``failure_count``
          - 失败 → 累加 ``failure_count``，达到 ``failure_threshold`` 时转为 OPEN

        Args:
            func: 待调用的可调用对象。
            *args: 透传给 ``func`` 的位置参数。
            **kwargs: 透传给 ``func`` 的关键字参数。

        Returns:
            ``func`` 的返回值。

        Raises:
            CircuitOpenError: 熔断器处于 OPEN 状态且未到恢复超时时抛出，
                携带 ``code="CIRCUIT_OPEN"``、``retryable=True``。
            Exception: ``func`` 自身抛出的任何异常将被透传（同时记录为失败）。

        Note:
            本方法不实现重试逻辑，仅做单次调用与状态反馈。
            若需重试，应由调用方在外层实现。
        """
        # OPEN 状态：判断是否已过恢复超时
        if self.state == CircuitState.OPEN:
            if time.monotonic() - self.last_failure_time >= self.recovery_timeout:
                # 进入 HALF_OPEN，放行一次试探性调用
                self.state = CircuitState.HALF_OPEN
            else:
                raise CircuitOpenError(
                    "熔断器处于 OPEN 状态，拒绝调用",
                    code="CIRCUIT_OPEN",
                    retryable=True,
                )

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            # 熔断器需捕获所有异常以记录失败计数，原异常重新抛出由上层处理
            logger.debug(f"熔断器记录失败: {type(e).__name__}: {e}")
            # 失败：累加计数并更新失败时间
            self.failure_count += 1
            self.last_failure_time = time.monotonic()
            if self.state == CircuitState.HALF_OPEN:
                # 试探性调用失败，重回 OPEN
                self.state = CircuitState.OPEN
            elif self.state == CircuitState.CLOSED and self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
            raise

        # 成功：HALF_OPEN 试探通过则重置，CLOSED 则清零连续失败计数
        if self.state == CircuitState.HALF_OPEN:
            self.reset()
        elif self.state == CircuitState.CLOSED:
            self.failure_count = 0
        return result

    def reset(self) -> None:
        """手动重置熔断器为 CLOSED 状态并清零失败计数

        将 ``state`` 置为 :attr:`CircuitState.CLOSED`，``failure_count`` 与
        ``last_failure_time`` 置为 0。适用于：

        - 运维人员确认下游服务已恢复后手动恢复调用链路
        - 测试用例在每个用例之间的状态隔离
        - HALF_OPEN 状态下试探性调用成功后的自动恢复（由 :meth:`call` 内部调用）

        Note:
            本方法不区分当前状态，任何状态调用都会立即重置为 CLOSED。
        """
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def is_available(self) -> bool:
        """判断熔断器当前是否放行调用

        用于在不实际执行调用的情况下预判熔断器是否会拒绝，便于监控、
        健康检查或日志上报。

        判定规则
        --------
        - **CLOSED**：返回 ``True``（始终放行）
        - **HALF_OPEN**：返回 ``True``（放行试探性调用）
        - **OPEN**：若已过 ``recovery_timeout`` 返回 ``True``（即将转 HALF_OPEN），
          否则返回 ``False``

        Returns:
            ``True`` 表示调用可放行；``False`` 表示调用会被拒绝
            （但注意：OPEN 状态到达恢复超时后，下一次 :meth:`call` 会自动
            转为 HALF_OPEN 并放行，本方法返回 ``True`` 与之一致）。

        Note:
            本方法不会触发状态转换，仅做只读判定。实际的状态转换发生在
            :meth:`call` 调用时。
        """
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.HALF_OPEN:
            return True
        # OPEN 状态：判断是否已过恢复超时
        return time.monotonic() - self.last_failure_time >= self.recovery_timeout
