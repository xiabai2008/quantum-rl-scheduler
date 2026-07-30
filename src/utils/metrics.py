"""
Prometheus 指标收集模块
Prometheus Metrics Collection Module

使用 prometheus_client 定义系统全局可观测指标，覆盖调度引擎、天衍云 API、
量子退火与运行时状态。所有指标为模块级单例，可直接 import 使用或通过
helper 函数批量记录。
"""

from typing import Any

from prometheus_client import Counter, Gauge, Histogram

__all__ = [
    "GRADIENT_NORM",
    "POLICY_ENTROPY",
    "POLICY_LOSS",
    "TRAINING_STEPS",
    "VALUE_LOSS",
    "active_connections",
    "annealing_iterations",
    "api_calls",
    "api_errors",
    "api_latency",
    "api_request_duration",
    "circuit_breaker_state",
    "qubit_utilization",
    "queue_length",
    "record_api_call",
    "record_scheduled_task",
    "task_wait_time",
    "tasks_scheduled",
    "tianyan_cb_state",
    "update_runtime_gauges",
]


# ===== 调度指标 =====
# 按策略与目标机器统计累计调度任务数
tasks_scheduled = Counter(
    "scheduler_tasks_total",
    "Total tasks scheduled",
    ["strategy", "target"],
)
# 任务等待时间分布（秒）
task_wait_time = Histogram(
    "scheduler_wait_seconds",
    "Task wait time",
    buckets=[1, 5, 10, 30, 60, 120, 300],
)

# ===== API 调用指标 =====
# 按方法与端点统计天衍云 API 请求总数
api_calls = Counter(
    "tianyan_api_requests_total",
    "Total API requests",
    ["method", "endpoint"],
)
# 天衍云 API 请求耗时分布（秒），按方法与端点维度
api_request_duration = Histogram(
    "tianyan_api_request_duration_seconds",
    "API request duration in seconds",
    ["method", "endpoint"],
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120],
)
# 天衍云 API 错误总数，按方法、端点与错误类型统计
api_errors = Counter(
    "tianyan_api_errors_total",
    "Total API errors",
    ["method", "endpoint", "error_type"],
)
# 保留旧版延迟直方图（向后兼容，无标签）
api_latency = Histogram(
    "tianyan_api_latency_seconds",
    "API call latency",
    buckets=[1, 5, 10, 30, 60, 120],
)

# ===== 量子退火指标 =====
# 量子退火迭代次数分布
annealing_iterations = Histogram(
    "annealing_iterations",
    "Annealing iterations",
    buckets=[100, 500, 1000, 5000, 10000],
)

# ===== 运行时状态指标 =====
# 当前量子比特利用率（0-1）
qubit_utilization = Gauge(
    "scheduler_qubit_utilization",
    "Current qubit utilization ratio 0-1",
)
# 当前任务队列长度
queue_length = Gauge(
    "scheduler_queue_length",
    "Current task queue length",
)
# 当前活跃 WebSocket 连接数
active_connections = Gauge(
    "websocket_active_connections",
    "Active WebSocket connections",
)
# 熔断器状态（0=closed, 1=open, 2=half_open）
circuit_breaker_state = Gauge(
    "circuit_breaker_state",
    "Circuit breaker state (0=closed, 1=open, 2=half_open)",
)
# 天衍云客户端熔断器状态（0=closed, 1=open, 2=half_open）
tianyan_cb_state = Gauge(
    "tianyan_circuit_breaker_state",
    "Tianyan client circuit breaker state (0=closed, 1=open, 2=half_open)",
)

# ── RL 训练过程指标（Issue #411）──
TRAINING_STEPS = Counter(
    "scheduler_training_steps_total",
    "Total RL training steps",
)
POLICY_ENTROPY = Gauge(
    "scheduler_policy_entropy",
    "Current policy entropy (exploration ability)",
)
GRADIENT_NORM = Gauge(
    "scheduler_gradient_norm",
    "Gradient norm (training stability)",
)
POLICY_LOSS = Gauge(
    "scheduler_policy_loss",
    "Current policy loss",
)
VALUE_LOSS = Gauge(
    "scheduler_value_loss",
    "Current value function loss",
)


def record_api_call(method: str, endpoint: str, latency: float) -> None:
    """记录一次 API 调用

    同时更新 API 请求计数器与延迟直方图。

    Args:
        method: 调用方法名称（如 "submit_quantum_task"）
        endpoint: API 端点名称（如 "quantum_task"）
        latency: 调用耗时（秒）
    """
    api_calls.labels(method=method, endpoint=endpoint).inc()
    api_request_duration.labels(method=method, endpoint=endpoint).observe(latency)


def record_scheduled_task(strategy: str, target: str, wait_seconds: float) -> None:
    """记录一次任务调度

    同时更新任务调度计数器与等待时间直方图。

    Args:
        strategy: 调度策略名称（如 "PPO" / "FCFS"）
        target: 调度目标（如机器名 "tianyan_s"）
        wait_seconds: 任务等待时间（秒）
    """
    tasks_scheduled.labels(strategy=strategy, target=target).inc()
    task_wait_time.observe(wait_seconds)


def update_runtime_gauges(system_status: dict[str, Any]) -> None:
    """从系统状态字典更新运行时 Gauge 指标（Issue #679）。

    Prometheus Gauge 此前定义后从未被 ``.set()`` 赋值，导致 ``/metrics``
    端点输出的 gauge 指标恒为初始值 0。本函数将 ``system_status`` 中的
    运行时数值同步到对应的 Prometheus Gauge，使监控采集获得真实数据。

    Args:
        system_status: 共享系统状态字典，包含 ``qubit_utilization``、
            ``queue_length`` 等键。
    """
    qubit_utilization.set(float(system_status.get("qubit_utilization", 0.0)))
    queue_length.set(float(system_status.get("queue_length", 0)))
