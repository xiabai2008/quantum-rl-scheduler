"""
熔断器模块
Circuit Breaker Module

实现熔断器模式（Circuit Breaker Pattern）：当连续调用失败达到阈值时
自动熔断，避免对故障服务的持续压力；经过恢复超时后进入 HALF_OPEN
状态放行一次试探性调用，根据结果决定恢复或继续熔断。
"""

import time
from collections.abc import Callable
from enum import Enum
from typing import Any, TypeVar

from src.exceptions import CircuitOpenError

__all__ = ["CircuitBreaker", "CircuitState"]

T = TypeVar("T")


class CircuitState(Enum):
    """熔断器状态枚举

    Attributes:
        CLOSED: 正常放行
        OPEN: 熔断拒绝
        HALF_OPEN: 试探性恢复
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """熔断器

    在函数调用外层包裹熔断逻辑：

    - CLOSED：正常放行；失败时累加计数，达到阈值转为 OPEN
    - OPEN：拒绝调用并抛出 CircuitOpenError；超过 recovery_timeout 后转为 HALF_OPEN
    - HALF_OPEN：放行一次试探性调用；成功则重置为 CLOSED，失败则重回 OPEN

    Args:
        failure_threshold: 连续失败触发熔断的阈值
        recovery_timeout: OPEN 状态恢复到 HALF_OPEN 的等待秒数
    """

    def __init__(self, failure_threshold: int = 5, recovery_timeout: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.state: CircuitState = CircuitState.CLOSED
        self.failure_count: int = 0
        self.last_failure_time: float = 0.0

    def call(self, func: Callable[..., T], *args: Any, **kwargs: Any) -> T:
        """通过熔断器执行函数

        Args:
            func: 待调用的可调用对象
            *args: 透传给 func 的位置参数
            **kwargs: 透传给 func 的关键字参数

        Returns:
            func 的返回值

        Raises:
            CircuitOpenError: 熔断器处于 OPEN 状态且未到恢复超时
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
        except Exception:
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
        """手动重置熔断器为 CLOSED 状态并清零失败计数"""
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0.0

    def is_available(self) -> bool:
        """判断熔断器当前是否放行调用

        Returns:
            True 表示调用可放行（CLOSED / HALF_OPEN，或 OPEN 已过恢复超时）
        """
        if self.state == CircuitState.CLOSED:
            return True
        if self.state == CircuitState.HALF_OPEN:
            return True
        # OPEN 状态：判断是否已过恢复超时
        return time.monotonic() - self.last_failure_time >= self.recovery_timeout
