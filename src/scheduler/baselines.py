"""
经典调度策略基线模块
Classic Scheduling Strategy Baselines

提供 FCFS / SPTF / EDF / Priority / RoundRobin / LIFO 等经典调度算法，
作为 RL 调度策略（PPO/DQN）的对比基准。

任务以 dict 表示，常用字段：
    - task_id        : 任务唯一标识
    - priority       : 优先级 1-5（5 最高）
    - estimated_time : 预估执行时间
    - arrival_time   : 到达时间
    - deadline       : 截止时间（可选，缺失时按 arrival_time + estimated_time*2 推算）
    - qubit_count    : 所需量子比特数

每个策略实现 ``select_action(tasks, available_resources) -> int`` 接口，
返回所选任务在 tasks 列表中的索引；若 tasks 为空返回 -1。
"""

from typing import Any

import numpy as np

__all__ = [
    "BaselineScheduler",
    "EDFScheduler",
    "EnvBasedEDFScheduler",
    "EnvBasedFCFSScheduler",
    "EnvBasedGreedyScheduler",
    "EnvBasedSPTFScheduler",
    "EnvBasedScheduler",
    "FCFSScheduler",
    "LIFOScheduler",
    "PriorityScheduler",
    "RoundRobinScheduler",
    "SPTFScheduler",
    "get_all_baseline_schedulers",
    "get_all_env_based_schedulers",
    "run_baseline_comparison",
]


# ---------------------------------------------------------------------------
# 任务字段默认值（字段缺失或类型异常时回退）
# ---------------------------------------------------------------------------
_DEFAULT_PRIORITY = 3
_DEFAULT_ESTIMATED_TIME = 1.0
_DEFAULT_ARRIVAL_TIME = 0.0


def _get_float(task: dict[str, Any], key: str, default: float) -> float:
    """安全读取任务字典中的 float 字段。

    字段缺失、为 None 或无法转换为 float 时返回默认值。

    Args:
        task    : 任务字典
        key     : 字段名
        default : 默认值

    Returns:
        字段对应的 float 值
    """
    value = task.get(key, default)
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _get_int(task: dict[str, Any], key: str, default: int) -> int:
    """安全读取任务字典中的 int 字段。

    字段缺失、为 None 或无法转换为 int 时返回默认值。

    Args:
        task    : 任务字典
        key     : 字段名
        default : 默认值

    Returns:
        字段对应的 int 值
    """
    value = task.get(key, default)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# 基类
# ---------------------------------------------------------------------------


class BaselineScheduler:
    """经典调度策略基类。

    所有具体策略继承本类并实现 ``select_action``。
    """

    def __init__(self, name: str) -> None:
        """初始化基类。

        Args:
            name : 策略名称（如 "FCFS"）
        """
        self.name = name

    def select_action(
        self, tasks: list[dict[str, Any]], available_resources: dict[str, Any]
    ) -> int:
        """从任务列表中选择一个任务，返回其索引。

        Args:
            tasks               : 待调度任务列表，每个任务为 dict
            available_resources : 可用资源字典（如 {"qubits": 10, "classical_load": 0.5}）

        Returns:
            被选中任务在 tasks 中的索引；若 tasks 为空返回 -1。
        """
        if not tasks:
            return -1
        raise NotImplementedError("子类必须实现 select_action")

    def reset(self) -> None:
        """重置调度器内部状态（如 RoundRobin 指针）。基类默认无操作。"""

    def __repr__(self) -> str:
        """返回策略的字符串表示。"""
        return f"{self.__class__.__name__}(name={self.name!r})"


# ---------------------------------------------------------------------------
# 具体策略
# ---------------------------------------------------------------------------


class FCFSScheduler(BaselineScheduler):
    """先来先服务（First-Come First-Served）。

    按 arrival_time 升序选择最先到达的任务。
    """

    def __init__(self) -> None:
        """初始化 FCFS 策略。"""
        super().__init__("FCFS")

    def select_action(
        self, tasks: list[dict[str, Any]], available_resources: dict[str, Any]
    ) -> int:
        """选择到达时间最早的任务。

        Args:
            tasks               : 待调度任务列表
            available_resources : 可用资源字典（本策略未使用）

        Returns:
            最早到达任务的索引；空列表返回 -1。
        """
        if not tasks:
            return -1
        return min(
            range(len(tasks)),
            key=lambda i: _get_float(tasks[i], "arrival_time", _DEFAULT_ARRIVAL_TIME),
        )


class SPTFScheduler(BaselineScheduler):
    """最短处理时间优先（Shortest Processing Time First）。

    按 estimated_time 升序选择耗时最短的任务。
    """

    def __init__(self) -> None:
        """初始化 SPTF 策略。"""
        super().__init__("SPTF")

    def select_action(
        self, tasks: list[dict[str, Any]], available_resources: dict[str, Any]
    ) -> int:
        """选择预估执行时间最短的任务。

        Args:
            tasks               : 待调度任务列表
            available_resources : 可用资源字典（本策略未使用）

        Returns:
            最短耗时任务的索引；空列表返回 -1。
        """
        if not tasks:
            return -1
        return min(
            range(len(tasks)),
            key=lambda i: _get_float(tasks[i], "estimated_time", _DEFAULT_ESTIMATED_TIME),
        )


class EDFScheduler(BaselineScheduler):
    """最早截止时间优先（Earliest Deadline First）。

    按 deadline 升序选择；若 deadline 缺失则按
    arrival_time + estimated_time * 2 推算截止时间。
    """

    def __init__(self) -> None:
        """初始化 EDF 策略。"""
        super().__init__("EDF")

    def select_action(
        self, tasks: list[dict[str, Any]], available_resources: dict[str, Any]
    ) -> int:
        """选择有效截止时间最早的任务。

        Args:
            tasks               : 待调度任务列表
            available_resources : 可用资源字典（本策略未使用）

        Returns:
            最早截止任务的索引；空列表返回 -1。
        """
        if not tasks:
            return -1
        return min(range(len(tasks)), key=lambda i: self._effective_deadline(tasks[i]))

    @staticmethod
    def _effective_deadline(task: dict[str, Any]) -> float:
        """计算任务的有效截止时间。

        deadline 缺失或非法时，按 arrival_time + estimated_time * 2 推算。

        Args:
            task : 任务字典

        Returns:
            有效截止时间（float）
        """
        raw = task.get("deadline")
        if raw is not None:
            try:
                return float(raw)
            except (TypeError, ValueError):
                pass
        arrival = _get_float(task, "arrival_time", _DEFAULT_ARRIVAL_TIME)
        est = _get_float(task, "estimated_time", _DEFAULT_ESTIMATED_TIME)
        return arrival + est * 2.0


class PriorityScheduler(BaselineScheduler):
    """优先级调度（Priority Scheduling）。

    按 priority 降序选择（priority 1-5，5 最高）；同优先级按到达时间升序
    （先到先服务）作为稳定 tiebreaker。
    """

    def __init__(self) -> None:
        """初始化 Priority 策略。"""
        super().__init__("Priority")

    def select_action(
        self, tasks: list[dict[str, Any]], available_resources: dict[str, Any]
    ) -> int:
        """选择优先级最高的任务。

        Args:
            tasks               : 待调度任务列表
            available_resources : 可用资源字典（本策略未使用）

        Returns:
            最高优先级任务的索引；空列表返回 -1。
        """
        if not tasks:
            return -1
        # priority 降序（5 最高），同优先级按 arrival_time 升序（先到先服务）
        return max(
            range(len(tasks)),
            key=lambda i: (
                _get_int(tasks[i], "priority", _DEFAULT_PRIORITY),
                -_get_float(tasks[i], "arrival_time", _DEFAULT_ARRIVAL_TIME),
            ),
        )


class RoundRobinScheduler(BaselineScheduler):
    """轮询调度（Round Robin）。

    维护内部指针，每次调用按指针返回当前任务索引并推进指针。
    """

    def __init__(self) -> None:
        """初始化 RoundRobin 策略，指针归零。"""
        super().__init__("RoundRobin")
        self._pointer = 0

    def select_action(
        self, tasks: list[dict[str, Any]], available_resources: dict[str, Any]
    ) -> int:
        """按内部指针轮转选择任务，并推进指针。

        Args:
            tasks               : 待调度任务列表
            available_resources : 可用资源字典（本策略未使用）

        Returns:
            指针当前位置对应的任务索引；空列表返回 -1。
        """
        if not tasks:
            return -1
        n = len(tasks)
        idx = self._pointer % n
        self._pointer = (self._pointer + 1) % n
        return idx

    def reset(self) -> None:
        """重置轮询指针为 0。"""
        self._pointer = 0


class LIFOScheduler(BaselineScheduler):
    """后来先服务（Last-In-First-Out）。

    按 arrival_time 降序选择最后到达的任务。
    """

    def __init__(self) -> None:
        """初始化 LIFO 策略。"""
        super().__init__("LIFO")

    def select_action(
        self, tasks: list[dict[str, Any]], available_resources: dict[str, Any]
    ) -> int:
        """选择到达时间最晚的任务。

        Args:
            tasks               : 待调度任务列表
            available_resources : 可用资源字典（本策略未使用）

        Returns:
            最晚到达任务的索引；空列表返回 -1。
        """
        if not tasks:
            return -1
        return max(
            range(len(tasks)),
            key=lambda i: _get_float(tasks[i], "arrival_time", _DEFAULT_ARRIVAL_TIME),
        )


# ---------------------------------------------------------------------------
# 模块级工具函数
# ---------------------------------------------------------------------------


def get_all_baseline_schedulers() -> list[BaselineScheduler]:
    """返回所有基线调度策略的实例列表。

    Returns:
        包含 6 个基线策略实例的列表
    """
    return [
        FCFSScheduler(),
        SPTFScheduler(),
        EDFScheduler(),
        PriorityScheduler(),
        RoundRobinScheduler(),
        LIFOScheduler(),
    ]


def run_baseline_comparison(
    tasks: list[dict[str, Any]],
    num_steps: int = 100,
    use_env: bool = False,
    env_config: dict[str, Any] | None = None,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """用所有基线策略调度给定任务列表，返回对比结果（Issue #231）。

    模拟流程：每步从剩余任务队列中按策略选一个任务执行，完成后从队列移除，
    累计奖励 / 等待时间 / 完成数；最多执行 num_steps 步或队列清空为止。

    **use_env=False（默认，向后兼容）**：
        使用独立模拟流程，奖励公式：reward = 10.0 + priority * 2.0 - wait * 0.1

    **use_env=True（Issue #230/#231）**：
        使用 ``EnvBasedScheduler`` 在 ``QuantumSchedulingEnv`` 中运行基线，
        reward 由 ``env.step(action)`` 返回（即 ``env_reward.py`` 计算），
        确保基线与 PPO 在相同奖励函数下对比。

    Args:
        tasks       : 待调度任务列表
        num_steps   : 最大调度步数（默认 100）
        use_env     : 是否使用 Gymnasium 环境模式（默认 False，向后兼容）
        env_config  : 环境配置字典（仅 use_env=True 时使用），如 max_steps, max_qubits
        seed        : 随机种子（仅 use_env=True 时使用）

    Returns:
        ``{策略名: {total_reward, completed_tasks, avg_wait_time, throughput, comparison_mode}}``
    """
    if not use_env:
        # 向后兼容：独立模拟流程
        results: dict[str, dict[str, Any]] = {}
        available_resources: dict[str, Any] = {"qubits": 20, "classical_load": 0.0}
        schedulers = get_all_baseline_schedulers()

        for scheduler in schedulers:
            scheduler.reset()
            # 深拷贝任务，避免跨策略污染
            queue: list[dict[str, Any]] = [dict(t) for t in tasks]

            total_reward = 0.0
            completed = 0
            total_wait = 0.0
            current_time = 0.0

            for _step in range(num_steps):
                if not queue:
                    break
                idx = scheduler.select_action(queue, available_resources)
                if idx < 0 or idx >= len(queue):
                    break
                task = queue.pop(idx)
                est = _get_float(task, "estimated_time", _DEFAULT_ESTIMATED_TIME)
                arrival = _get_float(task, "arrival_time", _DEFAULT_ARRIVAL_TIME)
                priority = _get_int(task, "priority", _DEFAULT_PRIORITY)

                # 等待时间 = 当前时间 - 到达时间（不小于 0）
                wait = max(0.0, current_time - arrival)
                total_wait += wait
                # 奖励：基础完成 + 优先级加权 - 等待惩罚
                total_reward += 10.0 + priority * 2.0 - wait * 0.1
                completed += 1
                # 推进模拟时钟
                current_time += est

            avg_wait = total_wait / completed if completed > 0 else 0.0
            throughput = completed / num_steps if num_steps > 0 else 0.0
            results[scheduler.name] = {
                "total_reward": total_reward,
                "completed_tasks": completed,
                "avg_wait_time": avg_wait,
                "throughput": throughput,
                "comparison_mode": "legacy",
            }

        return results

    # use_env=True：Gymnasium 环境模式（Issue #230/#231）
    from src.scheduler.env import QuantumSchedulingEnv

    config = env_config or {}
    max_steps = config.get("max_steps", num_steps)
    max_qubits = config.get("max_qubits", 287)

    # 导入观测索引常量（用于从 obs 读取 avg_wait_time）
    from src.scheduler.env_types import MAX_WAIT_STEPS, OBS_AVG_WAIT_TIME

    env_results: dict[str, dict[str, Any]] = {}
    env_schedulers = get_all_env_based_schedulers()

    for env_scheduler in env_schedulers:
        env_scheduler.reset()
        env = QuantumSchedulingEnv(
            max_steps=max_steps,
            max_qubits=max_qubits,
            seed=seed,
        )

        obs = env.reset(seed=seed)[0]
        total_reward = 0.0
        completed = 0
        total_wait = 0.0
        done = False

        while not done:
            action = env_scheduler.select_action(obs, env)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            # 从观测向量读取队列平均等待步数（OBS_AVG_WAIT_TIME 已归一化）
            total_wait += float(obs[OBS_AVG_WAIT_TIME]) * MAX_WAIT_STEPS
            completed += 1
            done = terminated or truncated

        avg_wait = total_wait / completed if completed > 0 else 0.0
        env_results[env_scheduler.name] = {
            "total_reward": total_reward,
            "completed_tasks": completed,
            "avg_wait_time": avg_wait,
            "throughput": completed / max_steps if max_steps > 0 else 0.0,
            "comparison_mode": "env_based",
        }

    return env_results


# ---------------------------------------------------------------------------
# EnvBasedScheduler：将基线策略封装为 Gymnasium 环境动作选择器（Issue #230）
# ---------------------------------------------------------------------------
# 与 BaselineScheduler 的区别：
#   - BaselineScheduler.select_action(tasks, resources) -> int  操作任务列表索引
#   - EnvBasedScheduler.select_action(observation, env) -> int  操作 Gymnasium 动作空间
#
# 在 Gymnasium 环境中运行时，reward 由 env.step(action) 返回（env_reward.py 计算），
# 而非独立公式，确保基线与 PPO 在相同奖励函数下对比。


class EnvBasedScheduler:
    """基线策略的 Gymnasium 环境适配器（Issue #230）。

    将经典调度策略封装为 Gymnasium 环境的动作选择器，
    使基线策略在相同的 ``QuantumSchedulingEnv`` 环境中运行，
    reward 由 ``env.step(action)`` 返回（即 ``env_reward.py`` 计算）。

    子类需实现 ``select_action(observation, env) -> int``，
    在合法动作空间 [0, 1, 2] 中选择动作：
        - 0 (ACTION_CLASSICAL): 分配到经典计算资源
        - 1 (ACTION_QUANTUM)  : 分配到量子计算资源
        - 2 (ACTION_HYBRID)    : 混合执行

    Attributes:
        name: 策略名称，用于结果标识
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def select_action(self, observation: np.ndarray, env: Any) -> int:
        """根据观测和环境状态选择动作。

        Args:
            observation: Gymnasium 环境的观测向量（14维）
            env        : QuantumSchedulingEnv 实例

        Returns:
            动作值（0=classical, 1=quantum, 2=hybrid）
        """
        raise NotImplementedError("子类必须实现 select_action")

    def reset(self) -> None:
        """重置调度器内部状态。基类默认无操作。"""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class EnvBasedFCFSScheduler(EnvBasedScheduler):
    """FCFS 策略的 Gymnasium 环境适配器。

    策略逻辑：优先处理当前任务（先来先服务）。
    若当前任务是量子类型且量子资源可用，选择量子动作；
    若是经典类型，选择经典动作；
    无当前任务时选择经典动作（空操作）。
    """

    def __init__(self) -> None:
        super().__init__("FCFS")

    def select_action(self, observation: np.ndarray, env: Any) -> int:
        # OBS_TASK_TYPE_QUANTUM=8, OBS_TASK_TYPE_CLASSICAL=9
        is_quantum = observation[8] > 0.5
        is_classical = observation[9] > 0.5
        qubit_avail = observation[0]  # 量子比特可用比率

        if is_quantum and qubit_avail > 0.1:
            return 1  # ACTION_QUANTUM
        if is_classical:
            return 0  # ACTION_CLASSICAL
        # 无明确类型或量子资源不足，走经典
        return 0


class EnvBasedSPTFScheduler(EnvBasedScheduler):
    """SPTF 策略的 Gymnasium 环境适配器。

    策略逻辑：最短处理时间优先。
    量子任务通常有加速比（speedup 2-5x），等效处理时间更短，
    因此量子任务可用时优先选择量子动作。
    """

    def __init__(self) -> None:
        super().__init__("SPTF")

    def select_action(self, observation: np.ndarray, env: Any) -> int:
        is_quantum = observation[8] > 0.5
        qubit_avail = observation[0]
        fidelity = observation[3]  # 量子保真度

        # 量子任务且资源充足且保真度合格 -> 量子（等效时间最短）
        if is_quantum and qubit_avail > 0.2 and fidelity > 0.85:
            return 1  # ACTION_QUANTUM
        # 量子任务但资源不足 -> 混合
        if is_quantum and qubit_avail > 0.05:
            return 2  # ACTION_HYBRID
        # 默认经典
        return 0  # ACTION_CLASSICAL


class EnvBasedEDFScheduler(EnvBasedScheduler):
    """EDF 策略的 Gymnasium 环境适配器。

    策略逻辑：最早截止时间优先。
    紧急任务（高 urgency）优先选择量子加速以尽快完成。
    """

    def __init__(self) -> None:
        super().__init__("EDF")

    def select_action(self, observation: np.ndarray, env: Any) -> int:
        urgency = observation[7]  # 当前任务紧急程度
        is_quantum = observation[8] > 0.5
        qubit_avail = observation[0]
        fidelity = observation[3]

        # 紧急任务且量子可用 -> 量子加速
        if urgency > 0.5 and qubit_avail > 0.1 and fidelity > 0.8:
            return 1  # ACTION_QUANTUM
        # 量子任务且资源可用 -> 量子
        if is_quantum and qubit_avail > 0.2:
            return 1  # ACTION_QUANTUM
        # 量子任务但资源紧张 -> 混合
        if is_quantum and qubit_avail > 0.05:
            return 2  # ACTION_HYBRID
        # 默认经典
        return 0  # ACTION_CLASSICAL


class EnvBasedGreedyScheduler(EnvBasedScheduler):
    """Greedy 策略的 Gymnasium 环境适配器。

    策略逻辑：贪心选择当前收益最高的动作。
    量子动作在保真度足够时有更高基础奖励（10*speedup），
    但保真度低于 0.9 时奖励打 6 折。
    """

    def __init__(self) -> None:
        super().__init__("Greedy")

    def select_action(self, observation: np.ndarray, env: Any) -> int:
        is_quantum = observation[8] > 0.5
        qubit_avail = observation[0]
        fidelity = observation[3]

        # 量子任务且保真度高且资源充足 -> 量子（贪心追求最高奖励）
        if is_quantum and fidelity > 0.9 and qubit_avail > 0.2:
            return 1  # ACTION_QUANTUM
        # 量子任务且保真度中等 -> 混合（避免低保真度打折）
        if is_quantum and fidelity > 0.85 and qubit_avail > 0.1:
            return 2  # ACTION_HYBRID
        # 量子任务但条件不足 -> 经典
        if is_quantum:
            return 0  # ACTION_CLASSICAL
        # 经典任务
        return 0  # ACTION_CLASSICAL


def get_all_env_based_schedulers() -> list[EnvBasedScheduler]:
    """返回所有 Gymnasium 环境适配的基线策略实例列表（Issue #230）。

    Returns:
        包含 4 个 EnvBasedScheduler 子类实例的列表
    """
    return [
        EnvBasedFCFSScheduler(),
        EnvBasedSPTFScheduler(),
        EnvBasedEDFScheduler(),
        EnvBasedGreedyScheduler(),
    ]
