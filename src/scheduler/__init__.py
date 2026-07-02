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

import logging
from typing import Any

logger = logging.getLogger(__name__)

# 直接导入环境模块（不依赖 stable_baselines3）
from src.scheduler.env import (
    DEFAULT_MACHINE_CONFIGS,
    ClassicalResource,
    QuantumMachine,
    QuantumResource,
    QuantumSchedulingEnv,
    register_env,
)
from src.scheduler.env import Task as EnvTask

# SchedulerAgent 依赖 stable_baselines3，延迟导入
try:
    from src.scheduler.agent import (
        PPOAgent,
        RealMachineCallback,
        SchedulerAgent,
    )
except ImportError:
    logger.warning(
        "scheduler.agent 导入失败（可能缺少 stable_baselines3），"
        "SchedulerAgent / PPOAgent / RealMachineCallback 不可用。"
        " 如需 RL 训练功能，请执行: pip install stable-baselines3[extra]"
    )
    SchedulerAgent = None  # type: ignore[assignment, misc]
    PPOAgent = None  # type: ignore[assignment, misc]
    RealMachineCallback = None  # type: ignore[assignment, misc]

# parser 模块
try:
    from src.scheduler.parser import (
        LegacyTaskParser,
        Task,
        TaskBuilder,
        TaskFeatures,
        TaskParser,
    )
except ImportError:
    logger.warning(
        "scheduler.parser 导入失败，将回退到 env.Task。"
        " TaskBuilder / TaskParser / TaskFeatures / LegacyTaskParser 不可用。"
    )
    Task = EnvTask  # type: ignore[assignment, misc]  # fallback to env Task
    TaskBuilder = None  # type: ignore[assignment, misc]
    TaskParser = None  # type: ignore[assignment, misc]
    TaskFeatures = None  # type: ignore[assignment, misc]
    LegacyTaskParser = None  # type: ignore[assignment, misc]

# 多目标奖励包装器
MultiObjectiveRewardWrapper: Any | None = None
make_mo_env: Any | None = None
MO_DEFAULT_WEIGHTS: Any = {}

try:
    from src.scheduler.multi_objective_env import (
        DEFAULT_WEIGHTS as MO_DEFAULT_WEIGHTS,
    )
    from src.scheduler.multi_objective_env import (
        MultiObjectiveRewardWrapper,
        make_mo_env,
    )
except ImportError:
    logger.warning(
        "scheduler.multi_objective_env 导入失败，"
        "MultiObjectiveRewardWrapper / make_mo_env 不可用。"
    )

# 向后兼容别名
SchedulingEnv = QuantumSchedulingEnv

__all__ = [
    "DEFAULT_MACHINE_CONFIGS",
    "MO_DEFAULT_WEIGHTS",
    "ClassicalResource",
    "LegacyTaskParser",
    "MultiObjectiveRewardWrapper",
    "PPOAgent",
    "QuantumMachine",
    "QuantumResource",
    "QuantumSchedulingEnv",
    "RealMachineCallback",
    "SchedulerAgent",
    "SchedulingEnv",
    "Task",
    "TaskBuilder",
    "TaskFeatures",
    "TaskParser",
    "make_mo_env",
    "register_env",
]
