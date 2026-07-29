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
from itertools import pairwise
from typing import Any

import numpy as np
from loguru import logger

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

    def __init__(
        self,
        tasks: list[DAGTask] | None = None,
        max_qubits: int = 287,
        seed: int | None = None,
    ) -> None:
        """初始化 DAG 调度器。

        Args:
            tasks: 初始任务列表，默认 None 表示空图。
            max_qubits: 单台机器最大量子比特数，默认 287（天衍-287）。
            seed: 随机种子（Issue #354）。固定后内置 NumPy 模拟退火结果可复现；
                  None 时使用默认值 42 以保证兜底退火行为的稳定性。
        """
        self.tasks: dict[str, DAGTask] = {}
        self.max_qubits: int = max_qubits
        # Issue #354: 退火种子，统一接入 set_seed 体系
        self._annealing_seed: int = 42 if seed is None else int(seed)
        self.completed: set[str] = set()
        self.failed: set[str] = set()
        self._last_annealing_solver: str | None = None
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

    # ----------------------------------------------------------
    # 时间索引 QUBO 调度（Issue #286）
    # ----------------------------------------------------------

    def build_scheduling_qubo(
        self,
        time_horizon: int,
    ) -> tuple[np.ndarray, dict[int, tuple[str, int]]]:
        """构建带唯一性、前驱与容量约束的时间索引 QUBO。

        决策变量 ``x[i, t]`` 表示任务 ``i`` 是否在时间槽 ``t`` 开始。
        矩阵采用对称 ``x.T @ Q @ x`` 约定，因此非对角有效系数平均
        分配到上下三角。

        Args:
            time_horizon: 可选开始时间槽数量，必须为正整数。

        Returns:
            QUBO 对称矩阵，以及变量索引到 ``(task_id, slot)`` 的映射。

        Raises:
            ValueError: 时间范围非法或 DAG 不合法。
        """
        if time_horizon <= 0:
            raise ValueError("time_horizon 必须为正整数")
        self.validate_dag()

        task_ids = list(self.tasks)
        n_variables = len(task_ids) * time_horizon
        if n_variables > 512:
            logger.warning(
                "[DAG-QUBO] 变量数 {} 超过 512，建议缩小任务数或 time_horizon",
                n_variables,
            )

        variable_map = {
            task_index * time_horizon + slot: (task_id, slot)
            for task_index, task_id in enumerate(task_ids)
            for slot in range(time_horizon)
        }
        qubo = np.zeros((n_variables, n_variables), dtype=np.float64)
        if not task_ids:
            return qubo, variable_map

        max_priority = max(max(0, task.priority) for task in self.tasks.values())
        constraint_penalty = float(max(100, 10 * time_horizon * (max_priority + 2) * len(task_ids)))

        def variable_index(task_id: str, slot: int) -> int:
            return task_ids.index(task_id) * time_horizon + slot

        def add_pair(left: int, right: int, effective_coefficient: float) -> None:
            if left == right:
                qubo[left, left] += effective_coefficient
                return
            half = effective_coefficient / 2.0
            qubo[left, right] += half
            qubo[right, left] += half

        # 目标：晚开始成本 + 高优先级任务更强的提前倾向。
        for task_id in task_ids:
            task = self.tasks[task_id]
            priority_weight = 1.0 + max(0, task.priority)
            for slot in range(time_horizon):
                index = variable_index(task_id, slot)
                qubo[index, index] += priority_weight * slot

        # 唯一性：P * (sum_t x[i,t] - 1)^2。
        for task_id in task_ids:
            indices = [variable_index(task_id, slot) for slot in range(time_horizon)]
            for index in indices:
                qubo[index, index] -= constraint_penalty
            for left_offset, left in enumerate(indices):
                for right in indices[left_offset + 1 :]:
                    add_pair(left, right, 2.0 * constraint_penalty)

        # 前驱：后继开始早于前驱完成的组合增加硬惩罚。
        for successor_id in task_ids:
            successor = self.tasks[successor_id]
            for predecessor_id in successor.dependencies:
                predecessor = self.tasks[predecessor_id]
                duration = max(0.0, predecessor.estimated_time)
                for predecessor_slot in range(time_horizon):
                    earliest_successor = predecessor_slot + duration
                    for successor_slot in range(time_horizon):
                        if successor_slot + 1e-12 >= earliest_successor:
                            continue
                        add_pair(
                            variable_index(predecessor_id, predecessor_slot),
                            variable_index(successor_id, successor_slot),
                            2.0 * constraint_penalty,
                        )

        # 容量：所有重叠任务加入轻量二次拥塞成本；两任务合计已超容量时
        # 再加入硬惩罚。拥塞项能表达三项及以上并发负载的二次增长，
        # 最终仍由解码后的独立校验保证总和不超过容量。
        capacity_scale = max(1, self.max_qubits) ** 2
        for left_offset, left_id in enumerate(task_ids):
            left_task = self.tasks[left_id]
            left_qubits = max(0, left_task.qubits_required)
            left_duration = max(0.0, left_task.estimated_time)
            for right_id in task_ids[left_offset + 1 :]:
                right_task = self.tasks[right_id]
                right_qubits = max(0, right_task.qubits_required)
                right_duration = max(0.0, right_task.estimated_time)
                congestion_cost = (
                    0.05 * constraint_penalty * left_qubits * right_qubits / capacity_scale
                )
                hard_conflict = left_qubits + right_qubits > self.max_qubits
                for left_slot in range(time_horizon):
                    left_finish = left_slot + left_duration
                    for right_slot in range(time_horizon):
                        right_finish = right_slot + right_duration
                        overlaps = left_slot < right_finish and right_slot < left_finish
                        if overlaps:
                            effective_coefficient = congestion_cost
                            if hard_conflict:
                                effective_coefficient += 2.0 * constraint_penalty
                            add_pair(
                                variable_index(left_id, left_slot),
                                variable_index(right_id, right_slot),
                                effective_coefficient,
                            )

        return qubo, variable_map

    def schedule_with_annealing(
        self,
        time_horizon: int = 10,
        num_reads: int = 100,
        fallback: bool = True,
    ) -> list[dict[str, Any]]:
        """使用 neal 或内置 NumPy 模拟退火求解时间索引 DAG QUBO。

        Args:
            time_horizon: 可选开始时间槽数量。
            num_reads: 独立退火读取次数，必须为正整数。
            fallback: 不可行或求解异常时是否回退到经典资源调度。

        Returns:
            与 :meth:`schedule_with_resources` 相同结构的调度列表。

        Raises:
            RuntimeError: 求解失败或结果不可行且 ``fallback=False``。
            ValueError: ``time_horizon`` 或 ``num_reads`` 非法。
        """
        if num_reads <= 0:
            raise ValueError("num_reads 必须为正整数")
        qubo, variable_map = self.build_scheduling_qubo(time_horizon)
        if not variable_map:
            self._last_annealing_solver = None
            return []

        try:
            bits = self._solve_scheduling_qubo(qubo, num_reads)
            schedule = self._decode_scheduling_solution(bits, variable_map)
            if self._is_scheduling_solution_feasible(schedule, time_horizon):
                return schedule
            raise RuntimeError("退火解未通过前驱或容量可行性校验")
        except Exception as exc:
            if not fallback:
                raise RuntimeError("DAG QUBO 退火调度失败") from exc
            logger.warning("[DAG-QUBO] {}，回退到经典资源调度", exc)
            self._last_annealing_solver = "classical_fallback"
            return self.schedule_with_resources(self.max_qubits, available_machines=1)

    def _solve_scheduling_qubo(self, qubo: np.ndarray, num_reads: int) -> np.ndarray:
        """优先使用 neal，导入或求解失败时使用 NumPy 模拟退火。"""
        try:
            import neal

            qubo_dict: dict[tuple[int, int], float] = {}
            for row in range(qubo.shape[0]):
                diagonal = float(qubo[row, row])
                if abs(diagonal) > 1e-12:
                    qubo_dict[(row, row)] = diagonal
                for column in range(row + 1, qubo.shape[0]):
                    coefficient = float(qubo[row, column] + qubo[column, row])
                    if abs(coefficient) > 1e-12:
                        qubo_dict[(row, column)] = coefficient
            sampleset = neal.SimulatedAnnealingSampler().sample_qubo(
                qubo_dict,
                num_reads=num_reads,
                seed=self._annealing_seed,
            )
            sample = sampleset.first.sample
            self._last_annealing_solver = "neal"
            return np.array(
                [int(sample.get(index, 0)) for index in range(qubo.shape[0])],
                dtype=np.int8,
            )
        except Exception as exc:
            logger.warning("[DAG-QUBO] neal 不可用或求解失败：{}；使用 NumPy SA", exc)
            self._last_annealing_solver = "numpy_sa"
            return self._numpy_scheduling_annealing(qubo, num_reads)

    def _numpy_scheduling_annealing(
        self,
        qubo: np.ndarray,
        num_reads: int,
    ) -> np.ndarray:
        """轻量 NumPy 模拟退火兜底，不依赖 torch。

        Issue #354: 使用 ``self._annealing_seed`` 作为 RNG 种子，
        同 seed 两次运行结果一致，可复现。
        """
        n_variables = qubo.shape[0]
        rng = np.random.default_rng(self._annealing_seed)
        best = np.zeros(n_variables, dtype=np.int8)
        best_energy = float(best @ qubo @ best)
        sweeps = max(100, 20 * n_variables)
        initial_temperature = max(1.0, float(np.max(np.abs(qubo))))

        for _ in range(num_reads):
            current: np.ndarray = np.asarray(
                rng.integers(0, 2, size=n_variables),
                dtype=np.int8,
            )
            current_energy = float(current @ qubo @ current)
            for sweep in range(sweeps):
                temperature = initial_temperature * (0.01 ** (sweep / max(1, sweeps - 1)))
                index = int(rng.integers(0, n_variables))
                diagonal = qubo[index, index]
                local_field = diagonal + 2.0 * (
                    float(np.dot(qubo[index], current)) - diagonal * current[index]
                )
                delta_energy = (1.0 - 2.0 * current[index]) * local_field
                if delta_energy <= 0 or rng.random() < np.exp(-delta_energy / temperature):
                    current[index] = 1 - current[index]
                    current_energy += float(delta_energy)
                    if current_energy < best_energy:
                        best = current.copy()
                        best_energy = current_energy
        return best

    def _decode_scheduling_solution(
        self,
        bits: np.ndarray,
        variable_map: dict[int, tuple[str, int]],
    ) -> list[dict[str, Any]]:
        """将开始变量解码为单机调度；缺失或重复选择保留为不可行。"""
        starts: dict[str, list[int]] = {task_id: [] for task_id in self.tasks}
        for index, value in enumerate(bits):
            if int(value) == 1 and index in variable_map:
                task_id, slot = variable_map[index]
                starts[task_id].append(slot)

        schedule: list[dict[str, Any]] = []
        for task_id, slots in starts.items():
            for slot in slots:
                finish = float(slot) + max(0.0, self.tasks[task_id].estimated_time)
                schedule.append(
                    {
                        "task_id": task_id,
                        "start_time": float(slot),
                        "machine_id": 0,
                        "estimated_finish": finish,
                    }
                )
        schedule.sort(key=lambda item: (item["start_time"], item["task_id"]))
        return schedule

    def _is_scheduling_solution_feasible(
        self,
        schedule: list[dict[str, Any]],
        time_horizon: int,
    ) -> bool:
        """独立校验唯一性、前驱时序、开始范围与总容量。"""
        if len(schedule) != len(self.tasks):
            return False
        by_id: dict[str, dict[str, Any]] = {}
        for item in schedule:
            task_id = str(item["task_id"])
            if task_id not in self.tasks or task_id in by_id:
                return False
            start = float(item["start_time"])
            if start < 0 or start >= time_horizon:
                return False
            by_id[task_id] = item

        for task_id, task in self.tasks.items():
            start = float(by_id[task_id]["start_time"])
            for dependency in task.dependencies:
                dependency_finish = float(by_id[dependency]["estimated_finish"])
                if start + 1e-12 < dependency_finish:
                    return False

        event_points = sorted(
            {float(item[key]) for item in schedule for key in ("start_time", "estimated_finish")}
        )
        for left, right in pairwise(event_points):
            if right <= left:
                continue
            midpoint = (left + right) / 2.0
            used_qubits = sum(
                max(0, self.tasks[str(item["task_id"])].qubits_required)
                for item in schedule
                if float(item["start_time"]) <= midpoint < float(item["estimated_finish"])
            )
            if used_qubits > self.max_qubits:
                return False
        return True

    # ----------------------------------------------------------
    # 量子退火辅助调度
    # ----------------------------------------------------------

    def schedule_with_quantum_assist(
        self,
        available_qubits: int,
        available_machines: int = 1,
        optimizer: Any = None,
    ) -> list[dict[str, Any]]:
        """量子退火辅助的多机任务调度。

        在满足依赖关系的前提下，调用量子退火求解多机任务分配 QUBO 问题，
        将任务分配到具体机器，再基于分配结果计算开始时间与预估完成时间。

        退火功能受全局开关 ``QUANTUM_ACCELERATION_ENABLED`` 控制（定义于
        :mod:`src.quantum.annealing`，默认关闭）。开关关闭时直接回退到
        :meth:`schedule_with_resources` 经典调度，避免无谓的退火调用。
        退火失败或异常时同样回退到 :meth:`schedule_with_resources`，保证调度可用。

        Args:
            available_qubits: 每台机器可用量子比特数（容量）。
            available_machines: 可用机器数，默认 1。
            optimizer: 量子退火优化器实例；为 None 时由退火模块创建默认仿真器。

        Returns:
            调度结果列表，每项为
            ``{task_id, start_time, machine_id, estimated_finish}``，
            格式与 :meth:`schedule_with_resources` 一致，
            按开始时间、机器 ID、任务 ID 升序排列。
        """
        # 延迟导入，避免 dag_scheduler 模块加载时引入 annealing 的 torch 依赖
        try:
            from src.quantum.annealing import (
                QUANTUM_ACCELERATION_ENABLED,
                solve_task_assignment,
            )
        except ImportError:
            return self.schedule_with_resources(available_qubits, available_machines)

        # Issue #590: 退火功能开关检查——关闭时回退到经典资源约束调度，
        # 避免在 QUANTUM_ACCELERATION_ENABLED=False（默认）时无谓地调用退火。
        if not QUANTUM_ACCELERATION_ENABLED:
            logger.debug(
                "量子加速已关闭（QUANTUM_ACCELERATION_ENABLED=False），"
                "schedule_with_quantum_assist 回退到经典资源约束调度。"
            )
            return self.schedule_with_resources(available_qubits, available_machines)

        # 空图直接返回
        if not self.tasks:
            return []

        n_machines = max(1, available_machines)
        tasks_list: list[dict[str, Any]] = [
            {
                "task_id": tid,
                "qubits_required": max(0, task.qubits_required),
                "estimated_time": max(0.0, task.estimated_time),
                "priority": task.priority,
            }
            for tid, task in self.tasks.items()
        ]
        machines_list: list[dict[str, Any]] = [
            {"machine_id": str(m), "capacity": available_qubits} for m in range(n_machines)
        ]

        try:
            assignment, _energy = solve_task_assignment(
                tasks_list, machines_list, optimizer=optimizer
            )
        except Exception as e:
            # Issue #389: 退火异常时记录日志后回退到经典资源约束调度，保证调度可用
            # 原实现完全吞掉异常，退火配置错误/依赖缺失/QUBO 构建异常将无法被发现
            logger.warning(
                "量子退火辅助调度失败，回退到经典资源约束调度: {}: {}",
                type(e).__name__,
                e,
                exc_info=True,
            )
            return self.schedule_with_resources(available_qubits, available_machines)

        # 机器 ID → 索引映射
        machine_id_to_idx: dict[str, int] = {
            machines_list[m]["machine_id"]: m for m in range(n_machines)
        }

        order = self.topological_sort()
        machines_intervals: list[list[tuple[float, float, int]]] = [[] for _ in range(n_machines)]
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

            # 量子退火分配的机器索引
            assigned_mid_str = assignment.get(tid, "0")
            mid = machine_id_to_idx.get(assigned_mid_str, 0)
            if mid < 0 or mid >= n_machines:
                mid = 0

            start = self._earliest_slot(
                machines_intervals[mid], est, qubits_needed, duration, available_qubits
            )
            finish = start + duration
            machines_intervals[mid].append((start, finish, qubits_needed))
            finish_time[tid] = finish
            schedule.append(
                {
                    "task_id": tid,
                    "start_time": start,
                    "machine_id": mid,
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
