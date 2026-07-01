"""
Prometheus 指标收集模块
Prometheus Metrics Collection Module

使用 prometheus_client 定义系统全局可观测指标，覆盖调度引擎、天衍云 API、
量子退火与运行时状态。所有指标为模块级单例，可直接 import 使用或通过
helper 函数批量记录。
"""

from prometheus_client import Counter, Gauge, Histogram

__all__ = [
    "active_connections",
    "annealing_iterations",
    "api_calls",
    "api_latency",
    "circuit_breaker_state",
    "qubit_utilization",
    "queue_length",
    "record_api_call",
    "record_scheduled_task",
    "task_wait_time",
    "tasks_scheduled",
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
# 按端点与状态统计天衍云 API 请求总数
api_calls = Counter(
    "tianyan_api_requests_total",
    "Total API requests",
    ["endpoint", "status"],
)
# 天衍云 API 调用延迟分布（秒）
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


def record_api_call(endpoint: str, status: str, latency: float) -> None:
    """记录一次 API 调用

    同时更新 API 请求计数器与延迟直方图。

    Args:
        endpoint: API 端点名称（如 "submit_task"）
        status: 调用状态（如 "success" / "error"）
        latency: 调用耗时（秒）
    """
    api_calls.labels(endpoint=endpoint, status=status).inc()
    api_latency.observe(latency)


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
