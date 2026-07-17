"""
量子任务调度环境的动态演化模块
Dynamics Module for Quantum-Classical Hybrid Task Scheduling Environment

本模块封装环境的动态演化逻辑，将依赖环境内部状态的方法抽离为独立函数：
    - generate_random_task : 生成具有真实差异性的随机任务（异质化版本）
    - check_compatibility   : 检查任务类型与所选资源的兼容性
    - advance_time          : 推进一个仿真时间步，更新所有资源状态
    - pick_next_task        : 从队列中取出下一个待调度任务

依赖关系：仅依赖 env_types.py 中的常量与数据类，不依赖 env.py。
通过 ``env`` 参数访问环境内部状态，通过 ``env._recompute_aggregate`` /
``env._generate_random_task`` 等薄包装回调到 env.py，避免循环导入。
"""

from typing import TYPE_CHECKING

import numpy as np

from src.scheduler.env_types import (
    ACTION_CLASSICAL,
    ACTION_HYBRID,
    ACTION_QUANTUM,
    MAX_QUEUE_SIZE,
    Task,
)

if TYPE_CHECKING:
    # 仅用于类型标注，避免运行时循环导入
    from src.scheduler.env import QuantumSchedulingEnv


def generate_random_task(rng: np.random.Generator, task_id: int) -> Task:
    """
    生成具有真实差异性的随机任务（异质化版本）。

    任务参数采用不均匀分布，模拟真实场景中的任务多样性：
        - qubits: 偏向中小规模，但偶发大规模（长尾分布）
        - urgency: 大部分正常，少数紧急
        - execution_time: 与 qubits 正相关
        - task_type: 混合分布，量子任务居多

    Args:
        rng     : NumPy 随机数生成器
        task_id : 任务编号

    Returns:
        Task: 生成的随机任务对象
    """
    qubit_options = [2, 3, 5, 5, 8, 10, 10, 15, 20, 30, 50, 100]
    qubit_probs = [0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02]
    qubits = int(rng.choice(qubit_options, p=qubit_probs))

    urgency_options = [0.1, 0.3, 0.5, 0.5, 0.5, 0.7, 0.8, 0.9, 0.95, 1.0]
    urgency_probs = [0.05, 0.10, 0.20, 0.20, 0.15, 0.10, 0.08, 0.05, 0.05, 0.02]
    urgency = float(rng.choice(urgency_options, p=urgency_probs))

    base_time = int(qubits**0.6)
    execution_time = max(1, base_time + int(rng.choice([-1, 0, 0, 0, 1, 2])))

    task_type_options = ["quantum", "quantum", "classical", "classical", "universal"]
    task_type_probs = [0.35, 0.35, 0.10, 0.10, 0.10]
    task_type = str(rng.choice(task_type_options, p=task_type_probs))

    qubit_count = 0 if task_type == "classical" else qubits

    return Task(
        task_id=f"T{task_id:04d}",
        task_type=task_type,
        qubit_count=qubit_count,
        wait_steps=0,
        urgency=urgency,
        priority=int(rng.integers(1, 6)),
        execution_time=execution_time,
    )


def check_compatibility(task: Task, action: int) -> bool:
    """
    检查任务类型与所选资源的兼容性。

    兼容性规则（修改后）：
        - classical 任务 → 经典资源 (action=0) 或混合执行 (action=2)
        - quantum 任务   → 量子资源 (action=1) 或混合执行 (action=2)
        - universal 任务 → 三种动作均兼容

    注意：混合执行 (action=2) 现在对所有任务类型都兼容，
    因为它会根据实际情况灵活选择资源。

    Args:
        task   : 待检查的任务
        action : 智能体选择的动作

    Returns:
        bool: True 表示兼容，False 表示不兼容
    """
    if task.task_type == "classical":
        # classical任务：经典资源或混合执行都兼容
        return action in (ACTION_CLASSICAL, ACTION_HYBRID)
    elif task.task_type == "quantum":
        # quantum任务：量子资源或混合执行都兼容
        return action in (ACTION_QUANTUM, ACTION_HYBRID)
    else:  # universal
        return True


def advance_time(env: "QuantumSchedulingEnv", rng: np.random.Generator) -> None:
    """
    推进一个仿真时间步，更新所有资源状态。

    更新内容：
        1. 量子资源：每台机器独立波动（可用比特、保真度、在线状态、队列完成）
        2. 经典资源：负载随机波动，任务完成后释放
        3. 队列中任务：每个任务等待步数 +1
        4. 随机生成 0-3 个新任务（泊松分布，均值 1.2）
        5. 聚合所有机器状态到 env._quantum（保证旧版 obs 一致）

    Args:
        env: 调度环境实例
        rng: 随机数生成器
    """
    # ---- 每台量子机器独立波动 ----
    for m in env._machines:
        # 可用比特比率波动
        m.available_ratio = float(
            np.clip(m.available_ratio + rng.uniform(-0.1, 0.1), 0.05, 1.0)
        )
        # 保真度衰减 + 随机恢复
        m.fidelity = float(np.clip(m.fidelity - 0.002 + rng.uniform(0.0, 0.01), 0.7, 0.999))
        # 更新物理噪声特征（基于新的 fidelity）
        m.update_noise_features(rng)
        # 在线状态随机波动（模拟真机维护/校准，5% 概率翻转）
        if rng.random() < 0.05:
            m.available = not m.available
        # 该机器队列完成任务
        completed_m = 0
        for _ in range(m.quantum_queue):
            if rng.random() < 0.15:  # 15% 概率完成一个量子任务
                completed_m += 1
        m.quantum_queue = max(0, m.quantum_queue - completed_m)

    # 聚合到 env._quantum（保持旧版 obs/reward 逻辑不变）
    env._recompute_aggregate()

    # 经典资源波动
    env._classical.load = np.clip(env._classical.load + rng.uniform(-0.15, 0.15), 0.0, 1.0)
    # 经典队列完成一些任务
    completed_c = 0
    for _ in range(env._classical.queue):
        if rng.random() < 0.2:  # 20% 概率完成一个经典任务
            completed_c += 1
    env._classical.queue = max(0, env._classical.queue - completed_c)
    # 完成任务后释放负载
    env._classical.load = np.clip(env._classical.load - 0.05 * completed_c, 0.0, 1.0)

    # 队列中任务等待步数 +1
    for task in env._task_queue:
        task.wait_steps += 1

    # 随机生成新任务（泊松分布，均值 1.2）
    new_task_count = int(rng.poisson(1.2))
    for _ in range(new_task_count):
        if len(env._task_queue) < MAX_QUEUE_SIZE:
            new_id = env._total_scheduled + len(env._task_queue)
            env._task_queue.append(env._generate_random_task(rng, task_id=new_id))


def pick_next_task(env: "QuantumSchedulingEnv") -> None:
    """
    从队列中取出下一个待调度任务作为当前任务。

    优先调度紧急程度最高的任务（priority 降序、wait_steps 降序）。

    Args:
        env: 调度环境实例
    """
    if not env._task_queue:
        env._current_task = None
        return

    # 按紧急程度和等待时间排序，选出最紧急的任务
    env._task_queue.sort(key=lambda t: (-t.priority, -t.wait_steps, -t.urgency))
    env._current_task = env._task_queue.pop(0)
