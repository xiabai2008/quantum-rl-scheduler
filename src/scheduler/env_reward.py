"""
量子任务调度环境的奖励计算模块
Reward Computation for Quantum-Classical Hybrid Task Scheduling Environment

本模块将调度环境的奖励计算逻辑抽离为独立函数，便于单元测试与复用：
    - compute_execution_reward : 计算任务执行成功后的即时奖励
    - compute_wait_penalty      : 计算队列中所有任务的等待超时惩罚

依赖关系：仅依赖 env_types.py 中的常量与数据类，不依赖 env.py。
"""

import numpy as np

from src.scheduler.env_types import (
    ACTION_CLASSICAL,
    ACTION_QUANTUM,
    MAX_WAIT_STEPS,
    QUANTUM_SPEEDUP_RANGE,
    REWARD_CLASSICAL,
    REWARD_HYBRID,
    REWARD_QUANTUM_BASE,
    REWARD_SUCCESS_BONUS,
    REWARD_WAIT_OVER_THRESHOLD,
    Task,
)


def compute_execution_reward(
    task: Task,
    action: int,
    rng: np.random.Generator,
    quantum_fidelity: float,
    quantum_available_ratio: float,
) -> float:
    """
    计算任务执行成功后的即时奖励。

    这个函数只处理"任务已经被安排执行"的正向收益，不处理错误分配、
    等待超时和低利用率惩罚；那些全局项在 step() 中统一累加。

    奖励规则：
        - 经典执行 (action=0):
          REWARD_CLASSICAL + REWARD_SUCCESS_BONUS，作为稳定基准。
        - 量子执行 (action=1):
          REWARD_QUANTUM_BASE * speedup + REWARD_SUCCESS_BONUS。
          speedup 从 QUANTUM_SPEEDUP_RANGE 随机采样，并乘以保真度因子；
          当保真度低于 0.9 时再乘 0.6，表示低质量量子结果的折扣。
        - 混合执行 (action=2):
          REWARD_HYBRID * hybrid_factor + REWARD_SUCCESS_BONUS。
          hybrid_factor 随量子可用率从 0.5 到 1.0 变化，表示量子资源越充足，
          混合执行越接近完整收益。

    Args:
        task                    : 被执行的任务（保留用于未来扩展，当前未参与计算）
        action                  : 执行方式（0=经典，1=量子，2=混合）
        rng                     : 随机数生成器（用于采样量子加速比）
        quantum_fidelity        : 当前量子资源聚合保真度（0-1）
        quantum_available_ratio : 当前量子资源聚合可用比率（0-1）

    Returns:
        float: 计算得到的即时奖励
    """
    if action == ACTION_CLASSICAL:
        # 经典执行不依赖量子机器状态，奖励最稳定，用作所有策略的基准线。
        return REWARD_CLASSICAL + REWARD_SUCCESS_BONUS

    elif action == ACTION_QUANTUM:
        # 基础加速比在 [2, 5] 之间随机
        speedup = rng.uniform(*QUANTUM_SPEEDUP_RANGE)
        # 保真度加成：保真度越高，加速比越大
        fidelity_factor = quantum_fidelity / 0.99  # 归一化到 ~1.0
        speedup *= fidelity_factor
        reward = REWARD_QUANTUM_BASE * speedup
        # 保真度过低时打折，避免智能体盲目偏向低质量量子资源。
        if quantum_fidelity < 0.9:
            reward *= 0.6
        return reward + REWARD_SUCCESS_BONUS

    else:  # ACTION_HYBRID
        # 混合执行奖励介于经典和量子之间，并随量子可用率动态调整。
        base = REWARD_HYBRID
        # available_ratio=0 时 factor=0.5，available_ratio=1 时 factor=1.0。
        hybrid_factor = 0.5 + 0.5 * quantum_available_ratio
        return base * hybrid_factor + REWARD_SUCCESS_BONUS


def compute_wait_penalty(task_queue: list[Task]) -> float:
    """
    计算队列中所有任务的等待超时惩罚。

    当任务的等待步数超过 MAX_WAIT_STEPS 时，每超一步惩罚
    REWARD_WAIT_OVER_THRESHOLD * overtime_ratio。惩罚与超时比例成正比。

    Args:
        task_queue: 当前任务队列（含 wait_steps 字段）

    Returns:
        float: 总等待惩罚（通常为负值或零）
    """
    penalty = 0.0
    for task in task_queue:
        if task.wait_steps > MAX_WAIT_STEPS:
            overtime_ratio = (task.wait_steps - MAX_WAIT_STEPS) / MAX_WAIT_STEPS
            penalty += REWARD_WAIT_OVER_THRESHOLD * overtime_ratio
    return penalty
