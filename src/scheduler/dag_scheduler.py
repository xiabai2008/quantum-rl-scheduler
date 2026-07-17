"""
DAG 调度器模块
DAG Scheduler with Task Dependency Graph Support

支持任务依赖图的调度器，提供：
- DAG 构建与合法性校验（环检测）
- Kahn 算法拓扑排序
- 就绪任务查询
- CPM 关键路径分析
- 资源约束调度（量子比特 + 多机器）

适用于需要表达任务间依赖关系的量子/经典混合调度场景。
"""

from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

__all__ = [
    "DAGScheduler",
    "DAGTask",
]

# DFS 三色标记常量（模块级，用于环检测）
_WHITE = 0  # 未访问
_GRAY = 1  # 访问中（在递归栈内）
_BLACK = 2  # 已完成


@dataclass
class DAGTask:
    """DAG 调度任务数据类。

    表示调度图中的一个节点，包含任务元数据、资源需求与依赖关系。

    Attributes:
        task_id: 任务唯一标识。
        task_type: 任务类型（quantum/classical/hybrid），默认 quantum。
        qubits_required: 所需量子比特数，默认 0。
        estimated_time: 预估执行时长，默认 0.0。
        priority: 优先级（1-5，5 最高），默认 3。
        dependencies: 前驱任务 ID 列表，默认空。
        status: 任务状态（pending/running/completed/failed），默认 pending。
    """

    task_id: str
    task_type: str = "quantum"
    qubits_required: int = 0
    estimated_time: float = 0.0
    priority: int = 3
    dependencies: list[str] = field(default_factory=list)
    status: str = "pending"


class DAGScheduler:
    """基于 DAG 的任务调度器。

    维护任务依赖图，提供拓扑排序、关键路径分析与资源约束调度能力。

    Attributes:
        tasks: 任务 ID 到 DAGTask 的映射。
        max_qubits: 单台机器最大可用量子比特数。
        completed: 已完成任务 ID 集合。
        failed: 已失败任务 ID 集合。
    """

    def __init__(self, tasks: list[DAGTask] | None = None, max_qubits: int = 287) -> None:
        """初始化 DAG 调度器。

        Args:
            tasks: 初始任务列表，默认 None 表示空图。
            max_qubits: 单台机器最大量子比特数，默认 287（天衍-287）。
        """
        self.tasks: dict[str, DAGTask] = {}
        self.max_qubits: int = max_qubits
        self.completed: set[str] = set()
        self.failed: set[str] = set()
        if tasks:
            for task in tasks:
                self.add_task(task)

    # ----------------------------------------------------------
    # DAG 构建
    # ----------------------------------------------------------

    def add_task(self, task: DAGTask) -> None:
        """添加任务到 DAG。

        Args:
            task: 待添加的任务对象。

        Raises:
            ValueError: 任务 ID 已存在。
        """
        if task.task_id in self.tasks:
            raise ValueError(f"任务 ID '{task.task_id}' 已存在，不可重复添加。")
        self.tasks[task.task_id] = task

    def add_dependency(self, task_id: str, depends_on: str) -> None:
        """添加依赖关系（task_id 依赖于 depends_on）。

        Args:
            task_id: 后继任务 ID。
            depends_on: 前驱任务 ID。

        Raises:
            ValueError: 任务不存在或自引用依赖。
        """
        if task_id not in self.tasks:
            raise ValueError(f"任务 '{task_id}' 不存在，无法添加依赖。")
        if depends_on not in self.tasks:
            raise ValueError(f"依赖任务 '{depends_on}' 不存在。")
        if task_id == depends_on:
            raise ValueError(f"任务 '{task_id}' 不能依赖自身。")
        deps = self.tasks[task_id].dependencies
        if depends_on not in deps:
            deps.append(depends_on)

    # ----------------------------------------------------------
    # DAG 校验
    # ----------------------------------------------------------

    def validate_dag(self) -> bool:
        """校验 DAG 合法性。

        检查所有依赖引用均存在且无环。

        Returns:
            True 表示 DAG 合法。

        Raises:
            ValueError: 存在缺失依赖或环。
        """
        # 检查依赖引用是否存在
        for task_id, task in self.tasks.items():
            for dep in task.dependencies:
                if dep not in self.tasks:
                    raise ValueError(f"任务 '{task_id}' 依赖不存在的任务 '{dep}'。")
        # 检查环
        if self._detect_cycle():
            raise ValueError("DAG 中存在环，无法进行拓扑排序。")
        return True

    def _detect_cycle(self) -> bool:
        """DFS 检测图中是否存在环。

        使用三色标记法（白/灰/黑）检测回边。

        Returns:
            True 表示存在环。
        """
        color: dict[str, int] = dict.fromkeys(self.tasks, _WHITE)

        def dfs(node: str) -> bool:
            """从指定节点开始深度优先搜索，检测回边。"""
            color[node] = _GRAY
            for dep in self.tasks[node].dependencies:
                if dep not in color:
                    continue
                if color[dep] == _GRAY:
                    return True
                if color[dep] == _WHITE and dfs(dep):
                    return True
            color[node] = _BLACK
            return False

        return any(color[t] == _WHITE and dfs(t) for t in self.tasks)

    def _build_adjacency(self) -> dict[str, list[str]]:
        """构建正向邻接表（前驱 → 后继列表）。

        Returns:
            任务 ID 到其后继任务 ID 列表的映射。
        """
        adj: dict[str, list[str]] = {tid: [] for tid in self.tasks}
        for tid, task in self.tasks.items():
            for dep in task.dependencies:
                if dep in adj:
                    adj[dep].append(tid)
        return adj

    # ----------------------------------------------------------
    # 拓扑排序
    # ----------------------------------------------------------

    def topological_sort(self) -> list[str]:
        """Kahn 算法拓扑排序。

        Returns:
            拓扑顺序的任务 ID 列表。

        Raises:
            ValueError: DAG 存在环。
        """
        if self._detect_cycle():
            raise ValueError("DAG 中存在环，无法进行拓扑排序。")

        # 计算入度（仅统计存在的依赖）
        in_degree: dict[str, int] = dict.fromkeys(self.tasks, 0)
        for tid, task in self.tasks.items():
            for dep in task.dependencies:
                if dep in in_degree:
                    in_degree[tid] += 1

        adj = self._build_adjacency()
        queue: deque[str] = deque(tid for tid, deg in in_degree.items() if deg == 0)
        order: list[str] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for succ in adj[node]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        return order

    # ----------------------------------------------------------
    # 就绪任务
    # ----------------------------------------------------------

    def get_ready_tasks(self) -> list[DAGTask]:
        """返回依赖已全部完成的就绪任务。

        Returns:
            状态为 pending 且所有依赖均 completed 的任务列表，
            按优先级降序、task_id 升序排列。
        """
        ready: list[DAGTask] = []
        for task in self.tasks.values():
            if task.status != "pending":
                continue
            if all(dep in self.completed for dep in task.dependencies if dep in self.tasks):
                ready.append(task)
        ready.sort(key=lambda t: (-t.priority, t.task_id))
        return ready

    # ----------------------------------------------------------
    # 状态流转
    # ----------------------------------------------------------

    def mark_completed(self, task_id: str) -> None:
        """标记任务完成并更新后继就绪状态。

        Args:
            task_id: 待标记完成的任务 ID。

        Raises:
            ValueError: 任务不存在。
        """
        if task_id not in self.tasks:
            raise ValueError(f"任务 '{task_id}' 不存在。")
        self.tasks[task_id].status = "completed"
        self.completed.add(task_id)
        self.failed.discard(task_id)

    def mark_failed(self, task_id: str) -> None:
        """标记任务失败。

        Args:
            task_id: 待标记失败的任务 ID。

        Raises:
            ValueError: 任务不存在。
        """
        if task_id not in self.tasks:
            raise ValueError(f"任务 '{task_id}' 不存在。")
        self.tasks[task_id].status = "failed"
        self.failed.add(task_id)
        self.completed.discard(task_id)

    # ----------------------------------------------------------
    # 关键路径
    # ----------------------------------------------------------

    def critical_path(self) -> list[str]:
        """CPM 关键路径分析（基于 estimated_time 最长路径）。

        Returns:
            关键路径上的任务 ID 列表（拓扑顺序）。空图返回空列表。
        """
        if not self.tasks:
            return []
        order = self.topological_sort()
        # earliest_finish[task] = max(ef of deps) + estimated_time
        earliest_finish: dict[str, float] = {}
        predecessor: dict[str, str | None] = dict.fromkeys(self.tasks, None)
        for tid in order:
            task = self.tasks[tid]
            max_dep_ef = 0.0
            for dep in task.dependencies:
                if dep in earliest_finish and earliest_finish[dep] >= max_dep_ef:
                    max_dep_ef = earliest_finish[dep]
                    predecessor[tid] = dep
            earliest_finish[tid] = max_dep_ef + task.estimated_time

        # 找到最大 EF 的节点（取拓扑序最晚的，确保路径最长）
        end_node = order[0]
        for tid in order:
            if earliest_finish[tid] >= earliest_finish[end_node]:
                end_node = tid
        # 回溯路径
        path: list[str] = []
        node: str | None = end_node
        while node is not None:
            path.append(node)
            node = predecessor[node]
        path.reverse()
        return path

    # ----------------------------------------------------------
    # 资源约束调度
    # ----------------------------------------------------------

    def schedule_with_resources(
        self, available_qubits: int, available_machines: int = 1
    ) -> list[dict[str, Any]]:
        """拓扑排序 + 资源约束调度。

        在满足依赖关系的前提下，按拓扑顺序将任务分配到多台机器，
        遵守每台机器的量子比特容量约束。

        Args:
            available_qubits: 每台机器可用量子比特数。
            available_machines: 可用机器数，默认 1。

        Returns:
            调度结果列表，每项为
            ``{task_id, start_time, machine_id, estimated_finish}``，
            按开始时间、机器 ID、任务 ID 升序排列。
        """
        order = self.topological_sort()
        # 每台机器的已调度区间列表：(start, end, qubits)
        machines: list[list[tuple[float, float, int]]] = [
            [] for _ in range(max(1, available_machines))
        ]
        finish_time: dict[str, float] = {}
        schedule: list[dict[str, Any]] = []

        for tid in order:
            task = self.tasks[tid]
            # 最早开始时间 = 依赖中最大完成时间
            est = 0.0
            for dep in task.dependencies:
                if dep in finish_time:
                    est = max(est, finish_time[dep])
            qubits_needed = max(0, task.qubits_required)
            duration = max(0.0, task.estimated_time)

            # 选最早可开始的机器
            best_start = float("inf")
            best_machine = 0
            for mid, intervals in enumerate(machines):
                start = self._earliest_slot(
                    intervals, est, qubits_needed, duration, available_qubits
                )
                if start < best_start:
                    best_start = start
                    best_machine = mid

            finish = best_start + duration
            machines[best_machine].append((best_start, finish, qubits_needed))
            finish_time[tid] = finish
            schedule.append(
                {
                    "task_id": tid,
                    "start_time": best_start,
                    "machine_id": best_machine,
                    "estimated_finish": finish,
                }
            )

        schedule.sort(key=lambda x: (x["start_time"], x["machine_id"], x["task_id"]))
        return schedule

    @staticmethod
    def _earliest_slot(
        intervals: list[tuple[float, float, int]],
        est: float,
        qubits_needed: int,
        duration: float,
        capacity: int,
    ) -> float:
        """计算机器上最早可容纳任务的起始时间。

        Args:
            intervals: 该机器已占用的区间列表 (start, end, qubits)。
            est: 最早允许开始时间（依赖约束）。
            qubits_needed: 任务所需比特数。
            duration: 任务执行时长。
            capacity: 机器比特总容量。

        Returns:
            最早可开始的时刻。
        """
        if qubits_needed > capacity:
            # 比特需求超出容量，只能串行排在所有任务之后
            last_end = max((e for _, e, _ in intervals), default=0.0)
            return max(est, last_end)
        # 候选起始时间：est 及各占用区间的结束时间
        candidates = {est}
        for s, e, _q in intervals:
            if e >= est:
                candidates.add(e)
            if s >= est:
                candidates.add(s)
        for t in sorted(candidates):
            end = t + duration
            if DAGScheduler._can_fit(intervals, t, end, qubits_needed, capacity):
                return t
        # 兜底：所有任务结束后
        last_end = max((e for _, e, _ in intervals), default=0.0)
        return max(est, last_end)

    @staticmethod
    def _can_fit(
        intervals: list[tuple[float, float, int]],
        start: float,
        end: float,
        qubits_needed: int,
        capacity: int,
    ) -> bool:
        """检查 ``[start, end]`` 区间内能否容纳任务。

        将区间按已有任务边界切分为子段，逐段校验比特占用峰值。

        Args:
            intervals: 已占用区间列表。
            start: 待插入任务开始时间。
            end: 待插入任务结束时间。
            qubits_needed: 待插入任务所需比特数。
            capacity: 机器比特总容量。

        Returns:
            True 表示可容纳。
        """
        if qubits_needed > capacity:
            return False
        # 构造事件点：区间端点 + 已有任务边界
        points = {start, end}
        for s, e, _q in intervals:
            if start < s < end:
                points.add(s)
            if start < e < end:
                points.add(e)
        sorted_points = sorted(points)
        for i in range(len(sorted_points) - 1):
            seg_start = sorted_points[i]
            seg_end = sorted_points[i + 1]
            if seg_end <= start or seg_start >= end:
                continue
            # 该子段内已用比特 = 完全覆盖该子段的所有任务比特之和
            used = sum(q for s, e, q in intervals if s <= seg_start and e >= seg_end)
            if used + qubits_needed > capacity:
                return False
        return True

    # ----------------------------------------------------------
    # 执行顺序
    # ----------------------------------------------------------

    def get_execution_order(self) -> list[str]:
        """返回考虑资源约束的执行顺序。

        使用调度器配置的 max_qubits 与单台机器进行资源约束调度，
        返回任务按开始时间排序的 ID 序列。

        Returns:
            任务 ID 执行顺序列表。
        """
        schedule = self.schedule_with_resources(self.max_qubits, 1)
        return [item["task_id"] for item in schedule]

    # ----------------------------------------------------------
    # 序列化
    # ----------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """序列化 DAG 的节点与边。

        Returns:
            包含 ``nodes``（任务字典列表）与 ``edges``（依赖边列表）的字典。
        """
        nodes = [asdict(task) for task in self.tasks.values()]
        edges: list[list[str]] = []
        for tid, task in self.tasks.items():
            for dep in task.dependencies:
                edges.append([dep, tid])
        return {"nodes": nodes, "edges": edges}

    @classmethod
    def from_tasks(cls, tasks: list[dict[str, Any]]) -> "DAGScheduler":
        """从字典列表构建调度器。

        Args:
            tasks: 任务字典列表，字段与 DAGTask 一致。

        Returns:
            构建完成的 DAGScheduler 实例。
        """
        scheduler = cls()
        for item in tasks:
            task = DAGTask(
                task_id=item["task_id"],
                task_type=item.get("task_type", "quantum"),
                qubits_required=item.get("qubits_required", 0),
                estimated_time=item.get("estimated_time", 0.0),
                priority=item.get("priority", 3),
                dependencies=list(item.get("dependencies", [])),
                status=item.get("status", "pending"),
            )
            scheduler.add_task(task)
        return scheduler
