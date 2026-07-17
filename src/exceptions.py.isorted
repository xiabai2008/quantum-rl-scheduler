"""
统一异常层次结构模块
Unified Exception Hierarchy Module

为整个 quantum-rl-scheduler 系统提供统一的异常基类与具体异常类型。
所有自定义异常均派生自 QuantumSchedulerError，携带错误码（code）与
可重试标志（retryable），便于上层（熔断器、重试器、API 客户端）统一处理。
"""

__all__ = [
    "CircuitOpenError",
    "ConfigurationError",
    "QuantumAnnealingError",
    "QuantumSchedulerError",
    "RateLimitError",
    "ResourceExhaustedError",
    "SchedulingError",
    "TaskParseError",
    "TianyanAPIError",
]


class QuantumSchedulerError(Exception):
    """系统基础异常

    所有 quantum-rl-scheduler 自定义异常的基类，携带错误码与可重试标志，
    便于熔断器、重试器等上层组件统一决策。

    Args:
        message: 异常描述信息
        code: 错误码（关键字参数，默认 "UNKNOWN"）
        retryable: 该异常是否可重试（关键字参数，默认 False）
    """

    def __init__(self, message: str, *, code: str = "UNKNOWN", retryable: bool = False) -> None:
        self.code = code
        self.retryable = retryable
        super().__init__(message)


class TianyanAPIError(QuantumSchedulerError):
    """天衍云 API 异常

    用于天衍云平台 API 调用失败的场景，如鉴权失败、请求超时、服务端错误等。
    """


class CircuitOpenError(QuantumSchedulerError):
    """熔断器打开异常

    当熔断器处于 OPEN 状态且尚未到达恢复超时时间时，调用将被拒绝并抛出此异常。
    """


class ConfigurationError(QuantumSchedulerError):
    """配置错误（不可重试）

    用于配置文件缺失、字段非法、环境变量未设置等场景，通常不可通过重试解决。
    """


class TaskParseError(QuantumSchedulerError):
    """任务解析错误

    用于 QASM 电路解析失败、任务字段缺失或格式非法等场景。
    """


class SchedulingError(QuantumSchedulerError):
    """调度错误

    用于调度引擎内部状态错误、无可用机器、动作非法等场景。
    """


class QuantumAnnealingError(QuantumSchedulerError):
    """量子退火错误

    用于 QUBO 矩阵构造失败、退火求解异常、结果不收敛等场景。
    """


class ResourceExhaustedError(QuantumSchedulerError):
    """资源耗尽错误

    用于量子比特、机器队列、连接池等资源耗尽，无法接受新任务的场景。
    """


class RateLimitError(QuantumSchedulerError):
    """API 限流错误

    当天衍云平台返回 429 Too Many Requests 或本地令牌桶限流触发时抛出。
    该异常默认可重试（``retryable=True``），且不计入熔断器失败计数，
    以避免限流导致的连续失败误触发熔断。

    Args:
        message: 异常描述信息
        code: 错误码（关键字参数，默认 "RATE_LIMIT"）
        retryable: 该异常是否可重试（关键字参数，默认 True）
        retry_after: 服务端建议的等待时间（秒），若响应中包含则透传
    """

    def __init__(
        self,
        message: str,
        *,
        code: str = "RATE_LIMIT",
        retryable: bool = True,
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, code=code, retryable=retryable)
