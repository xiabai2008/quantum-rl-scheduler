"""
量子任务调度环境的多机器调度模块
Multi-Machine Scheduling Module for Quantum-Classical Hybrid Task Scheduling

本模块封装多机器调度的核心启发式逻辑，将依赖环境内部状态的方法抽离为
独立函数，便于单测与复用：
    - select_best_machine      : 为任务选择最合适的量子机器（评分启发式）
    - machine_supports_task    : 检查机器门集合是否兼容任务需求
    - route_to_machine         : 将任务路由到选定机器，更新队列与调度记录
    - recompute_aggregate      : 根据所有机器状态重算聚合视图

依赖关系：仅依赖 env_types.py 中的常量与数据类，不依赖 env.py。
真机提交通过 ``env._submit_to_real_machine`` 薄包装调用，避免循环导入。
"""

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from src.scheduler.env_types import QuantumMachine, Task

if TYPE_CHECKING:
    # 仅用于类型标注，避免运行时循环导入
    from src.scheduler.env import QuantumSchedulingEnv


def select_best_machine(env: "QuantumSchedulingEnv", task: Task) -> QuantumMachine | None:
    """
    为给定任务选择最合适的量子机器（多机器调度核心启发式）。

    选择策略：
        1. 过滤：机器在线、可用比特数 >= 任务需求、支持任务所需门集合
        2. 评分：score = fidelity * available_ratio / (1 + quantum_queue)
        3. 返回得分最高的机器；若无机器通过过滤，返回 None

    评分兼顾保真度、可用资源与队列负载，倾向于把任务分发给
    当前质量最好且最空闲的机器，实现负载均衡与质量择优。

    Args:
        env:  调度环境实例（提供 _machines 列表）
        task: 待调度的任务

    Returns:
        QuantumMachine 或 None（无机器可承接）
    """
    candidates = []
    for m in env._machines:
        if not m.available:
            continue
        # 比特数检查（available_ratio * total_qubits 为估算可用比特）
        usable_qubits = int(m.total_qubits * m.available_ratio)
        if usable_qubits < task.qubit_count:
            continue
        # 门集合检查（任务所需门需被机器支持；这里按常见量子门粗粒度匹配）
        if not machine_supports_task(m, task):
            continue
        candidates.append(m)

    if not candidates:
        return None

    # 评分：保真度 * 可用比率 / (1 + 队列长度)
    def _score(m: QuantumMachine) -> float:
        return m.fidelity * m.available_ratio / (1.0 + m.quantum_queue)

    return max(candidates, key=_score)


def machine_supports_task(machine: QuantumMachine, task: Task) -> bool:
    """检查机器门集合是否兼容任务需求。

    当前实现：tianyan_s 仅支持 H/CZ/M（真机实测约束），
    含参数化门（RX/RY/RZ）的任务需路由到 tianyan_tn 等支持该门集的机器。
    任务的 required_gates 字段若未声明，默认按机器声明集合宽松放行。

    Args:
        machine: 候选机器
        task:    待执行任务

    Returns:
        bool: 机器能否执行该任务
    """
    required = getattr(task, "required_gates", None)
    if not required:
        # 任务未声明门需求，默认放行（保持与旧版行为一致）
        return True
    return set(required).issubset(set(machine.supported_gates))


def route_to_machine(
    env: "QuantumSchedulingEnv",
    machine: QuantumMachine | None,
    task: Task,
    rng: np.random.Generator,
) -> None:
    """将任务路由到选定的量子机器，更新队列与调度记录。

    若机器为真机模式（is_real=True 且已 attach 客户端），则以
    ``real_submit_probability`` 概率真正提交到天衍云真机，控制机时消耗。

    Args:
        env     : 调度环境实例
        machine : 选定的量子机器（None 时不做任何操作）
        task    : 被执行的任务
        rng     : 随机数生成器（用于抽样是否上真机）
    """
    if machine is None:
        env._last_selected_machine = None
        return

    # 多租户配额检查（Issue #97）：配额不足时拒绝路由
    if env._tenant_manager is not None:
        tenant_id = getattr(task, "tenant_id", None)
        if not env._tenant_manager.consume(
            tenant_id=tenant_id,
            qubits=getattr(task, "qubit_count", 0),
            tasks=1,
        ):
            logger.warning(
                f"[Tenant] 租户 {tenant_id} 配额不足，任务 {task.task_id} 调度被拒绝"
            )
            env._last_selected_machine = None
            return

    machine.quantum_queue += 1
    env._last_selected_machine = machine.name
    env._machine_schedule_count[machine.name] = (
        env._machine_schedule_count.get(machine.name, 0) + 1
    )

    # 选择性真机提交（控制机时成本）
    if (
        machine.is_real
        and machine.name in env._real_clients
        and env.real_submit_probability > 0.0
        and float(rng.random()) < env.real_submit_probability
    ):
        env._submit_to_real_machine(machine, task)


def recompute_aggregate(env: "QuantumSchedulingEnv") -> None:
    """根据所有机器状态重算 env._quantum 聚合视图。

    聚合规则：
        - available_ratio : 各机器可用比率的加权均值（按 total_qubits 加权）
        - fidelity        : 各机器保真度的加权均值
        - quantum_queue   : 所有机器队列长度之和
        - _quantum_available : 任一机器在线即为 True

    该聚合保证旧版 10 维观测与奖励计算在多机器模式下仍然有效。

    Args:
        env: 调度环境实例
    """
    if not env._machines:
        return
    total_q = sum(m.total_qubits for m in env._machines)
    if total_q <= 0:
        total_q = 1
    env._quantum.available_ratio = float(
        sum(m.available_ratio * m.total_qubits for m in env._machines) / total_q
    )
    env._quantum.fidelity = float(
        sum(m.fidelity * m.total_qubits for m in env._machines) / total_q
    )
    env._quantum.quantum_queue = sum(m.quantum_queue for m in env._machines)
    env._quantum_available = any(m.available for m in env._machines)
