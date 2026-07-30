"""
量子任务调度环境的观测构建模块
Observation Builder Module for Quantum-Classical Hybrid Task Scheduling Environment

本模块封装环境的观测向量与信息字典构建逻辑，将依赖环境内部状态的方法
抽离为独立函数：
    - get_observation : 构建并返回当前 17 维状态向量（含物理噪声、拓扑特征、串扰风险、公平性指数）
    - get_info        : 构建环境信息字典，供调试和监控使用

依赖关系：仅依赖 env_types.py 中的常量与数据类，不依赖 env.py。
通过 ``env`` 参数访问环境内部状态，避免循环导入。
"""

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.scheduler.env_types import (
    MAX_QUEUE_SIZE,
    MAX_WAIT_STEPS,
    OBS_ARRIVAL_RATE_MA,
    OBS_AVG_CONNECTIVITY,
    OBS_AVG_WAIT_TIME,
    OBS_CLASSICAL_LOAD,
    OBS_COUPLING_DENSITY,
    OBS_CROSSTALK_RISK,
    OBS_DIM,
    OBS_FAIRNESS_INDEX,
    OBS_FIDELITY,
    OBS_QUANTUM_QUEUE_RATIO,
    OBS_QUBIT_AVAILABILITY,
    OBS_QUEUE_LENGTH,
    OBS_SINGLE_GATE_FIDELITY,
    OBS_TASK_TYPE_CLASSICAL,
    OBS_TASK_TYPE_QUANTUM,
    OBS_TIME_OF_DAY,
    OBS_TWO_GATE_FIDELITY,
    OBS_URGENCY_LEVEL,
)

if TYPE_CHECKING:
    # 仅用于类型标注，避免运行时循环导入
    from src.scheduler.env import QuantumSchedulingEnv


def get_observation(env: "QuantumSchedulingEnv") -> NDArray[Any]:
    """
    构建并返回当前 17 维状态向量（扩展版：包含物理噪声、拓扑特征、串扰风险、时序流量、公平性指数）。

    各维度含义及计算方式：
        [0] qubit_availability  : 量子比特可用比率（直接取值）
        [1] queue_length        : 队列长度 / MAX_QUEUE_SIZE
        [2] avg_wait_time       : 队列中任务平均等待步数 / MAX_WAIT_STEPS
        [3] fidelity            : 量子比特平均保真度
        [4] classical_load     : 经典计算负载（直接取值）
        [5] quantum_queue_ratio : 量子队列 / (量子队列 + 经典队列 + 1)
        [6] time_of_day        : 当前模拟时刻
        [7] urgency_level      : 当前任务紧急程度（无任务时为 0）
        [8] task_type_quantum  : 当前任务是quantum类型（1=是，0=不是）
        [9] task_type_classical: 当前任务是classical类型（1=是，0=不是）
        [10] single_gate_fidelity : 单比特门平均保真度（所有机器加权平均）
        [11] two_gate_fidelity : 两比特门平均保真度（所有机器加权平均）
        [12] coupling_density  : 耦合图密度（所有机器加权平均）
        [13] avg_connectivity  : 量子比特平均连通度（所有机器加权平均）
        [14] crosstalk_risk    : 串扰风险（基于空间并发的任务密度）
        [15] arrival_rate_ma   : 任务到达率滑动平均（流量突发感知）
        [16] fairness_index    : Jain 公平性指数（基于多租户等待时间，Issue #588）

    Args:
        env: 调度环境实例

    Returns:
        NDArray[Any]: 形状 (17,)，dtype=float32，值域 [0, 1]
    """
    obs: NDArray[Any] = np.zeros(OBS_DIM, dtype=np.float32)

    obs[OBS_QUBIT_AVAILABILITY] = float(np.clip(env._quantum.available_ratio, 0.0, 1.0))

    obs[OBS_QUEUE_LENGTH] = float(np.clip(len(env._task_queue) / MAX_QUEUE_SIZE, 0.0, 1.0))

    if env._task_queue:
        avg_wait = sum(t.wait_steps for t in env._task_queue) / len(env._task_queue)
        obs[OBS_AVG_WAIT_TIME] = float(np.clip(avg_wait / MAX_WAIT_STEPS, 0.0, 1.0))
    else:
        obs[OBS_AVG_WAIT_TIME] = 0.0

    obs[OBS_FIDELITY] = float(np.clip(env._quantum.fidelity, 0.0, 1.0))

    obs[OBS_CLASSICAL_LOAD] = float(np.clip(env._classical.load, 0.0, 1.0))

    total_running = env._quantum.quantum_queue + env._classical.queue + 1
    obs[OBS_QUANTUM_QUEUE_RATIO] = float(
        np.clip(env._quantum.quantum_queue / total_running, 0.0, 1.0)
    )

    obs[OBS_TIME_OF_DAY] = float(np.clip(env._time_of_day, 0.0, 1.0))

    if env._current_task is not None:
        obs[OBS_URGENCY_LEVEL] = float(np.clip(env._current_task.urgency, 0.0, 1.0))
    else:
        obs[OBS_URGENCY_LEVEL] = 0.0

    # 添加任务类型编码
    if env._current_task is not None:
        obs[OBS_TASK_TYPE_QUANTUM] = 1.0 if env._current_task.task_type == "quantum" else 0.0
        obs[OBS_TASK_TYPE_CLASSICAL] = 1.0 if env._current_task.task_type == "classical" else 0.0
    else:
        obs[OBS_TASK_TYPE_QUANTUM] = 0.0
        obs[OBS_TASK_TYPE_CLASSICAL] = 0.0

    # 阶段1：物理噪声特征（所有机器加权平均）
    # 性能优化（Issue #219）：
    # - total_qubits 是机器静态属性，初始化后不变，使用 env._total_qubits_cache 缓存值
    # - 加权平均改用 numpy 向量化（np.average + weights），替代 Python for 循环
    # Issue #746：coupling_density / avg_connectivity / qubits_arr 在 episode 内不变，
    # 缓存到 env._obs_static_cache，reset() 时置脏重建
    if env._machines:
        total_q = getattr(env, "_total_qubits_cache", None) or sum(
            m.total_qubits for m in env._machines
        )
        if total_q > 0:
            # 惰性构建 episode 级静态缓存（首次调用、reset 后、或机器列表变更时重建）
            static_cache: dict[str, Any] | None = getattr(env, "_obs_static_cache", None)
            if static_cache is None or len(static_cache["qubits_arr"]) != len(env._machines):
                qubits_arr = np.array([m.total_qubits for m in env._machines], dtype=np.float64)
                coupling_arr = np.array(
                    [m.coupling_density for m in env._machines], dtype=np.float64
                )
                conn_arr = np.array([m.avg_connectivity for m in env._machines], dtype=np.float64)
                static_cache = {
                    "qubits_arr": qubits_arr,
                    "coupling_density": float(
                        np.clip(np.average(coupling_arr, weights=qubits_arr), 0.0, 1.0)
                    ),
                    "avg_connectivity": float(
                        np.clip(np.average(conn_arr, weights=qubits_arr), 0.0, 1.0)
                    ),
                }
                env._obs_static_cache = static_cache
                # Issue #746: 机器列表变更时同步 _total_qubits_cache
                total_q = int(qubits_arr.sum())
                env._total_qubits_cache = total_q

            # 动态特征仍每步重算（fidelity / used_qubits 随时间变化）
            single_arr = np.array([m.single_gate_fidelity for m in env._machines], dtype=np.float64)
            two_arr = np.array([m.two_gate_fidelity for m in env._machines], dtype=np.float64)
            obs[OBS_SINGLE_GATE_FIDELITY] = float(
                np.clip(
                    np.average(single_arr, weights=static_cache["qubits_arr"]),
                    0.0,
                    1.0,
                )
            )
            obs[OBS_TWO_GATE_FIDELITY] = float(
                np.clip(
                    np.average(two_arr, weights=static_cache["qubits_arr"]),
                    0.0,
                    1.0,
                )
            )
            # 阶段2：拓扑特征（episode 内静态，直接读缓存）
            obs[OBS_COUPLING_DENSITY] = static_cache["coupling_density"]
            obs[OBS_AVG_CONNECTIVITY] = static_cache["avg_connectivity"]

            # 计算串扰风险 (Crosstalk Risk): 所有机器中 (used_qubits / total_qubits) 的最大值或加权平均
            # 这里我们使用所有机器使用的 qubits 总和 / 所有机器的总 qubits
            used_q = sum(m.used_qubits for m in env._machines)
            obs[OBS_CROSSTALK_RISK] = float(np.clip(used_q / total_q, 0.0, 1.0))

    # 计算任务到达率滑动平均
    if hasattr(env, "arrival_history") and env.arrival_history:
        avg_arrival = sum(env.arrival_history) / len(env.arrival_history)
        # 假设最大预期单步到达任务数为 10（可调整）
        max_expected_arrival = 10.0
        obs[OBS_ARRIVAL_RATE_MA] = float(np.clip(avg_arrival / max_expected_arrival, 0.0, 1.0))
    elif hasattr(env, "current_time_window_arrivals"):
        # 如果历史为空，使用当前窗口的值
        max_expected_arrival = 10.0
        obs[OBS_ARRIVAL_RATE_MA] = float(
            np.clip(env.current_time_window_arrivals / max_expected_arrival, 0.0, 1.0)
        )

    # Issue #588: Jain 公平性指数（基于多租户等待时间）
    # 从公平性跟踪器获取实时 Jain 指数，无跟踪器或无数据时为 0.0
    fairness_tracker = getattr(env, "_fairness_tracker", None)
    if fairness_tracker is not None:
        try:
            jain_index = fairness_tracker.jain_wait_fairness()
            obs[OBS_FAIRNESS_INDEX] = float(np.clip(jain_index, 0.0, 1.0))
        except (ValueError, AttributeError):
            obs[OBS_FAIRNESS_INDEX] = 0.0

    return obs


def get_info(env: "QuantumSchedulingEnv") -> dict[str, Any]:
    """
    构建环境信息字典，供调试和监控使用。

    Args:
        env: 调度环境实例

    Returns:
        dict: 包含当前步数、统计摘要、资源状态、多机器调度详情等信息
    """
    info: dict[str, Any] = {
        "current_step": env._current_step,
        "max_steps": env._max_steps,
        "task_queue_length": len(env._task_queue),
        "total_scheduled": env._total_scheduled,
        "quantum_success": env._quantum_success,
        "classical_success": env._classical_success,
        "hybrid_success": env._hybrid_success,
        "mismatch_count": env._mismatch_count,
        "episode_reward": env._episode_reward,
        "qubit_availability": env._quantum.available_ratio,
        "fidelity": env._quantum.fidelity,
        "classical_load": env._classical.load,
        "time_of_day": env._time_of_day,
        # 多机器调度信息
        "num_machines": len(env._machines),
        "last_selected_machine": env._last_selected_machine,
        "machine_schedule_count": dict(env._machine_schedule_count),
        "machine_real_submits": dict(env._machine_real_submits),
        # 真机闭环统计（Issue #64）
        "real_machine_degraded": env._real_machine_degraded,
        "real_machine_stats": env.get_real_machine_stats(),
        "machines": [
            {
                "name": m.name,
                "total_qubits": m.total_qubits,
                "available_ratio": m.available_ratio,
                "fidelity": m.fidelity,
                "quantum_queue": m.quantum_queue,
                "available": m.available,
                "is_real": m.is_real,
                "supported_gates": list(m.supported_gates),
            }
            for m in env._machines
        ],
    }
    # 任务完成率（Issue #400）：成功调度数 / (成功调度数 + 队列剩余任务数)
    total_seen = env._total_scheduled + len(env._task_queue)
    info["completion_rate"] = env._total_scheduled / total_seen if total_seen > 0 else 0.0
    if env._current_task is not None:
        info["current_task"] = {
            "task_id": env._current_task.task_id,
            "task_type": env._current_task.task_type,
            "urgency": env._current_task.urgency,
            "wait_steps": env._current_task.wait_steps,
        }
    return info
