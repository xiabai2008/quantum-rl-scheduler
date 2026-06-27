"""
调度引擎核心模块
Scheduler Engine Core Module

包含：
- SchedulerAgent: RL智能体
- QuantumSchedulingEnv: 量子任务调度环境（Gymnasium接口）
- SchedulingEnv: QuantumSchedulingEnv 的别名（向后兼容）
- TaskParser: 任务解析器（新版，字典 → Task）
- LegacyTaskParser: 旧版字符串解析器（向后兼容）
- Task: 规范化任务数据类（env.py 中用于调度队列）
- TaskBuilder: Task 的 Builder 模式构造器
- TaskFeatures: 旧版特征向量（向后兼容）
- QuantumTask: env.py 中的量子任务数据结构
"""

# 直接导入环境模块（不依赖 stable_baselines3）
from src.scheduler.env import (
    QuantumSchedulingEnv,
    Task as EnvTask,
    QuantumResource,
    ClassicalResource,
    register_env,
)

# SchedulerAgent 依赖 stable_baselines3，延迟导入
try:
    from src.scheduler.agent import SchedulerAgent
except ImportError:
    SchedulerAgent = None  # type: ignore[assignment, misc]

# parser 模块
try:
    from src.scheduler.parser import (
        Task,
        TaskBuilder,
        TaskParser,
        TaskFeatures,
        LegacyTaskParser,
    )
except ImportError:
    Task = EnvTask  # fallback to env Task
    TaskBuilder = None  # type: ignore[assignment, misc]
    TaskParser = None  # type: ignore[assignment, misc]
    TaskFeatures = None  # type: ignore[assignment, misc]
    LegacyTaskParser = None  # type: ignore[assignment, misc]

# 向后兼容别名
SchedulingEnv = QuantumSchedulingEnv

__all__ = [
    "SchedulerAgent",
    "QuantumSchedulingEnv",
    "SchedulingEnv",
    "QuantumResource",
    "ClassicalResource",
    "register_env",
    "Task",
    "TaskBuilder",
    "TaskParser",
    "TaskFeatures",
    "LegacyTaskParser",
]
