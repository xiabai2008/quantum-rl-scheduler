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

__all__ = [
    "BaselineScheduler",
    "EDFScheduler",
    "FCFSScheduler",
    "LIFOScheduler",
    "PriorityScheduler",
    "RoundRobinScheduler",
    "SPTFScheduler",
    "get_all_baseline_schedulers",
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

    def select_action(self, tasks: list[dict], available_resources: dict) -> int:
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

    def select_action(self, tasks: list[dict], available_resources: dict) -> int:
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

    def select_action(self, tasks: list[dict], available_resources: dict) -> int:
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

    def select_action(self, tasks: list[dict], available_resources: dict) -> int:
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

    def select_action(self, tasks: list[dict], available_resources: dict) -> int:
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

    def select_action(self, tasks: list[dict], available_resources: dict) -> int:
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

    def select_action(self, tasks: list[dict], available_resources: dict) -> int:
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


def run_baseline_comparison(tasks: list[dict], num_steps: int = 100) -> dict[str, dict]:
    """用所有基线策略调度给定任务列表，返回对比结果。

    模拟流程：每步从剩余任务队列中按策略选一个任务执行，完成后从队列移除，
    累计奖励 / 等待时间 / 完成数；最多执行 num_steps 步或队列清空为止。

    奖励公式：reward = 10.0 + priority * 2.0 - wait * 0.1

        - 基础完成奖励 10.0
        - 优先级加权（priority 1-5）
        - 等待惩罚（每单位等待时间 -0.1）

    Args:
        tasks     : 待调度任务列表
        num_steps : 最大调度步数（默认 100）

    Returns:
        ``{策略名: {total_reward, completed_tasks, avg_wait_time, throughput}}``
    """
    results: dict[str, dict] = {}
    available_resources: dict = {"qubits": 20, "classical_load": 0.0}
    schedulers = get_all_baseline_schedulers()

    for scheduler in schedulers:
        scheduler.reset()
        # 深拷贝任务，避免跨策略污染
        queue: list[dict] = [dict(t) for t in tasks]

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
        }

    return results
