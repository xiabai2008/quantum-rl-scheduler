"""

经典调度策略基线模块

Classic Scheduling Strategy Baselines

提供 FCFS / SPTF / EDF / Priority / RoundRobin / LIFO / HEFT / Min-Min 等经典调度算法，

作为 RL 调度策略（PPO/DQN）的对比基准。

任务以 dict 表示，常用字段：

    - task_id        : 任务唯一标识

    - priority       : 优先级 1-5（5 最高）

    - estimated_time : 预估执行时间

    - arrival_time   : 到达时间

    - deadline       : 截止时间（可选，缺失时按 arrival_time + estimated_time*2 推算）

    - qubit_count    : 所需量子比特数

经典策略实现 ``select_action(tasks, available_resources) -> int`` 接口，

返回所选任务在 tasks 列表中的索引；若 tasks 为空返回 -1。

环境适配策略（Issue #230/#270）继承 ``EnvBasedScheduler``，

实现 ``select_action(observation, env) -> int`` 在 Gymnasium 环境中运行。

"""

from collections.abc import Callable

from typing import Any, NamedTuple

import numpy as np

from loguru import logger

from src.scheduler.env_types import (

    OBS_FIDELITY,

    OBS_QUANTUM_QUEUE_RATIO,

    OBS_QUBIT_AVAILABILITY,

    OBS_TASK_TYPE_CLASSICAL,

    OBS_TASK_TYPE_QUANTUM,

    OBS_URGENCY_LEVEL,

)

__all__ = [

    "BaselineScheduler",

    "EDFScheduler",

    "EnvBasedEDFScheduler",

    "EnvBasedFCFSScheduler",

    "EnvBasedGreedyScheduler",

    "EnvBasedHEFTScheduler",

    "EnvBasedMinMinScheduler",

    "EnvBasedSPTFScheduler",

    "EnvBasedScheduler",

    "FCFSScheduler",

    "HEFTScheduler",

    "LIFOScheduler",

    "MinMinScheduler",

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

# 观测解析结果（EnvBasedScheduler 公共解析，Issue #387）

# ---------------------------------------------------------------------------

class ObsInfo(NamedTuple):

    """从观测向量解析出的公共字段（EnvBasedScheduler 使用，Issue #387）。"""

    is_quantum: bool

    is_classical: bool

    qubit_avail: float

    fidelity: float

    urgency: float

    quantum_queue: float

# ---------------------------------------------------------------------------

# 基类

# ---------------------------------------------------------------------------

class BaselineScheduler:

    """经典调度策略基类。

    简单排序策略只需定义 ``sort_key`` 与 ``reverse`` 类属性；

    复杂策略（Priority/RoundRobin）覆盖 ``select_action``。

    """

    # 子类可定义 sort_key（task -> 可比较值）与 reverse 来复用排序式选择

    sort_key: Callable[[dict[str, Any]], Any] | None = None

    reverse: bool = False

    def __init__(self, name: str) -> None:

        """初始化基类。

        Args:

            name : 策略名称（如 "FCFS"）

        """

        self.name = name

    def select_action(self, tasks: list[dict], available_resources: dict) -> int:

        """从任务列表中选择一个任务，返回其索引。

        子类定义 ``sort_key`` 时按排序键选择，否则需覆盖本方法；空 tasks 返回 -1。

        """

        if not tasks:

            return -1

        key_func = self.sort_key

        if key_func is None:

            raise NotImplementedError("子类必须实现 select_action 或定义 sort_key/reverse")

        return self._select_by_sort_key(tasks, key_func, self.reverse)

    def _select_by_sort_key(

        self,

        tasks: list[dict],

        key_func: Callable[[dict[str, Any]], Any],

        reverse: bool = False,

    ) -> int:

        """按排序键选择任务索引（reverse=False 升序，True 降序）；空列表返回 -1。"""

        if not tasks:

            return -1

        return (

            max(range(len(tasks)), key=lambda i: key_func(tasks[i]))

            if reverse

            else min(range(len(tasks)), key=lambda i: key_func(tasks[i]))

        )

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

    sort_key = staticmethod(lambda task: _get_float(task, "arrival_time", _DEFAULT_ARRIVAL_TIME))

    reverse = False

    def __init__(self) -> None:

        """初始化 FCFS 策略。"""

        super().__init__("FCFS")

class SPTFScheduler(BaselineScheduler):

    """最短处理时间优先（Shortest Processing Time First）。

    按 estimated_time 升序选择耗时最短的任务。

    """

    sort_key = staticmethod(

        lambda task: _get_float(task, "estimated_time", _DEFAULT_ESTIMATED_TIME)

    )

    reverse = False

    def __init__(self) -> None:

        """初始化 SPTF 策略。"""

        super().__init__("SPTF")

class EDFScheduler(BaselineScheduler):

    """最早截止时间优先（Earliest Deadline First）。

    按 deadline 升序选择；若 deadline 缺失则按

    arrival_time + estimated_time * 2 推算截止时间。

    """

    reverse = False

    def __init__(self) -> None:

        """初始化 EDF 策略。"""

        super().__init__("EDF")

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

            except (TypeError, ValueError) as e:

                logger.debug(f"deadline 字段转换为 float 失败，使用估算值: {e}")

        arrival = _get_float(task, "arrival_time", _DEFAULT_ARRIVAL_TIME)

        est = _get_float(task, "estimated_time", _DEFAULT_ESTIMATED_TIME)

        return arrival + est * 2.0

    sort_key = _effective_deadline

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

    sort_key = staticmethod(lambda task: _get_float(task, "arrival_time", _DEFAULT_ARRIVAL_TIME))

    reverse = True

    def __init__(self) -> None:

        """初始化 LIFO 策略。"""

        super().__init__("LIFO")

class HEFTScheduler(BaselineScheduler):

    """HEFT（异构最早完成时间）经典调度策略（Issue #270/#582）。

    实现标准 HEFT upward rank 计算：

    - upward_rank[node] = avg_comp_time[node] + max(upward_rank[successor])

    - 退出节点（无后继）: upward_rank = avg_comp_time

    - 按 upward rank 降序选择任务

    任务字典可通过 ``successors`` 字段指定后继任务 ID 列表，

    缺失时视为退出节点（upward_rank = estimated_time）。

    """

    def __init__(self) -> None:

        """初始化 HEFT 策略。"""

        super().__init__("HEFT")

    def _compute_upward_ranks(

        self,

        tasks: list[dict],

        successors_map: dict[str, list[str]],

        avg_comp_times: dict[str, float],

    ) -> dict[str, float]:

        """计算 DAG 中每个任务的 upward rank（Issue #582）。

        upward_rank[node] = avg_comp_time[node] + max(upward_rank[successor])

        退出节点（无后继）: upward_rank = avg_comp_time[node]

        使用记忆化递归计算，含环检测保护（遇到正在计算的节点按退出节点处理）。

        Args:

            tasks           : 任务列表（用于遍历，实际计算依赖映射）

            successors_map  : {task_id: [successor_id, ...]} 后继映射

            avg_comp_times  : {task_id: avg_comp_time} 平均计算时间

        Returns:

            {task_id: upward_rank} 字典

        """

        ranks: dict[str, float] = {}

        # 正在计算中的节点（用于环检测，避免无限递归）

        computing: set[str] = set()

        def _rank(task_id: str) -> float:

            if task_id in ranks:

                return ranks[task_id]

            # 环检测：遇到正在计算的节点，按退出节点处理（断开环）

            if task_id in computing:

                return avg_comp_times.get(task_id, _DEFAULT_ESTIMATED_TIME)

            computing.add(task_id)

            comp = avg_comp_times.get(task_id, _DEFAULT_ESTIMATED_TIME)

            successors = successors_map.get(task_id, [])

            if not successors:

                ranks[task_id] = comp

            else:

                # 取所有后继 upward rank 的最大值

                max_succ = 0.0

                for s in successors:

                    max_succ = max(max_succ, _rank(s))

                ranks[task_id] = comp + max_succ

            computing.discard(task_id)

            return ranks[task_id]

        for task in tasks:

            tid = str(task.get("task_id", ""))

            if tid:

                _rank(tid)

        return ranks

    def select_action(self, tasks: list[dict], available_resources: dict) -> int:

        """按 upward rank 降序选择任务（Issue #582）。

        计算每个任务的 upward rank（基于 DAG 后继和平均计算时间），

        返回 rank 最高的任务索引。无后继信息的任务视为退出节点。

        Args:

            tasks               : 待调度任务列表

            available_resources : 可用资源字典（本策略未使用）

        Returns:

            upward rank 最高的任务索引；空列表返回 -1。

        """

        if not tasks:

            return -1

        # 构造后继映射和平均计算时间

        task_id_set = {str(t.get("task_id", i)) for i, t in enumerate(tasks)}

        successors_map: dict[str, list[str]] = {}

        avg_comp_times: dict[str, float] = {}

        for i, task in enumerate(tasks):

            tid = str(task.get("task_id", i))

            avg_comp_times[tid] = _get_float(task, "estimated_time", _DEFAULT_ESTIMATED_TIME)

            succ = task.get("successors", [])

            if succ is None:

                succ = []

            # 只保留在当前任务列表中的后继（避免引用不存在的任务）

            successors_map[tid] = [str(s) for s in succ if str(s) in task_id_set]

        ranks = self._compute_upward_ranks(tasks, successors_map, avg_comp_times)

        # 按 upward rank 降序选择（最大值索引）

        return max(

            range(len(tasks)),

            key=lambda i: ranks.get(str(tasks[i].get("task_id", i)), _DEFAULT_ESTIMATED_TIME),

        )

class MinMinScheduler(BaselineScheduler):

    """Min-Min 经典调度策略（Issue #270/#583）。

    标准迭代贪心 Min-Min 算法：

    1. 对每个未分配任务，计算在每个机器上的最小完成时间 (MCT)

    2. 选择所有任务中最小 MCT 对应的任务-机器对

    3. 将该任务分配到最佳机器，更新机器可用时间

    4. 重复直到所有任务分配完毕

    在 ``select_action`` 接口中，返回 Min-Min 算法第一轮选中的任务索引

    （即全局最小 MCT 对应的任务）。

    资源字典可通过 ``machines`` 字段提供机器列表，每个机器为含

    ``available_time`` 字段的 dict；缺失时按单机器（available_time=0）处理。

    """

    def __init__(self) -> None:

        """初始化 Min-Min 策略。"""

        super().__init__("MinMin")

    def _min_min_iterative(

        self,

        tasks: list[dict],

        machines: list[dict],

    ) -> list[tuple[int, int]]:

        """迭代贪心 Min-Min 分配（Issue #583）。

        每轮选择 (task, machine) 对中最小 MCT（Minimum Completion Time）的组合，

        分配该任务到该机器并更新机器可用时间，直到所有任务分配完毕。

        MCT = machine_available_time + task_estimated_time

        Args:

            tasks    : 待调度任务列表

            machines : 机器列表，每个机器为 dict，含 "available_time" 字段

        Returns:

            分配顺序列表 [(task_idx, machine_idx), ...]，按分配顺序排列

        """

        if not tasks or not machines:

            return []

        # 复制机器可用时间，避免污染输入

        machine_avail: list[float] = []

        for m in machines:

            avail = m.get("available_time", 0.0) if isinstance(m, dict) else 0.0

            machine_avail.append(float(avail))

        remaining = list(range(len(tasks)))

        assignment_order: list[tuple[int, int]] = []

        while remaining:

            best_pair: tuple[int, int] | None = None

            best_mct = float("inf")

            for ti in remaining:

                est = _get_float(tasks[ti], "estimated_time", _DEFAULT_ESTIMATED_TIME)

                for mi, m_avail in enumerate(machine_avail):

                    mct = m_avail + est

                    if mct < best_mct:

                        best_mct = mct

                        best_pair = (ti, mi)

            if best_pair is None:

                break

            ti, mi = best_pair

            assignment_order.append((ti, mi))

            # 更新机器可用时间

            machine_avail[mi] = best_mct

            remaining.remove(ti)

        return assignment_order

    def select_action(self, tasks: list[dict], available_resources: dict) -> int:

        """按 Min-Min 迭代贪心选择任务（Issue #583）。

        运行完整 Min-Min 迭代算法，返回第一轮选中的任务索引

        （即全局最小 MCT 对应的任务）。

        Args:

            tasks               : 待调度任务列表

            available_resources : 可用资源字典，可含 "machines" 列表

        Returns:

            第一轮选中的任务索引；空列表返回 -1。

        """

        if not tasks:

            return -1

        machines = available_resources.get("machines") if available_resources else None

        if not machines:

            # 单机器模式：使用单个默认机器

            machines = [{"available_time": 0.0}]

        order = self._min_min_iterative(tasks, machines)

        if not order:

            return -1

        return order[0][0]

# ---------------------------------------------------------------------------

# 模块级工具函数

# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# QAOA 量子近似优化调度基线（Issue #599）
# ---------------------------------------------------------------------------

class QAOAScheduler(BaselineScheduler):
    """QAOA 量子近似优化算法调度基线（Issue #599）。

    使用 QAOA（Quantum Approximate Optimization Algorithm）求解任务分配 QUBO，
    作为量子-经典混合调度方法的 SOTA 对比基线。

    工作流程：
        1. 将任务选择问题编码为 QUBO（二次无约束二元优化）
        2. 将 QUBO 转换为 Ising 模型
        3. 使用 p=1 QAOA 电路（经典模拟）求解
        4. 从测量结果中解码最优任务分配

    QUBO 设计：
        - 变量：x[i] = 1 表示选择第 i 个任务
        - 目标：最小化 (-priority * w_p + est_time * w_t + resource_penalty * w_r)
        - 约束：恰好选择一个任务（sum(x) = 1）

    QAOA p=1 电路：
        |psi(gamma, beta)> = U_M(beta) * U_C(gamma) |+>^n

    其中：
        - U_C(gamma) = exp(-i * gamma * H_C)，H_C 为 Ising 成本哈密顿量
        - U_M(beta) = exp(-i * beta * H_M)，H_M = sum(X_i) 为混合哈密顿量

    通过网格搜索优化 (gamma, beta) 参数，使有效解（恰好一个 1）的概率最大化。

    注意：经典模拟复杂度为 O(4^n)，因此限制最多 ``max_qubits`` 个候选任务。
    超出时按优先级降序预选前 ``max_qubits`` 个任务。

    Attributes:
        p          : QAOA 层数（默认 1）
        grid_size  : 参数网格搜索分辨率（默认 10）
        max_qubits : 最大模拟量子比特数（默认 10）
        seed       : 随机种子
    """

    def __init__(
        self,
        p: int = 1,
        grid_size: int = 10,
        max_qubits: int = 10,
        seed: int = 42,
    ) -> None:
        """初始化 QAOA 调度器。

        Args:
            p         : QAOA 层数，默认 1
            grid_size : 参数网格搜索分辨率，默认 10
            max_qubits: 最大模拟量子比特数，默认 10
            seed      : 随机种子，默认 42
        """
        super().__init__("QAOA")
        self.p: int = p
        self.grid_size: int = grid_size
        self.max_qubits: int = max_qubits
        self._rng = np.random.default_rng(seed)
        # 调试信息
        self._last_qubo: np.ndarray | None = None
        self._last_solution: np.ndarray | None = None
        self._last_gamma: float = 0.0
        self._last_beta: float = 0.0
        self._last_valid_prob: float = 0.0

    def select_action(self, tasks: list[dict], available_resources: dict) -> int:
        """使用 QAOA 求解任务分配 QUBO，返回选中的任务索引。

        Args:
            tasks              : 任务列表
            available_resources : 可用资源字典（如 {"qubits": 20}）

        Returns:
            被选中任务的索引；空列表返回 -1
        """
        if not tasks:
            return -1

        n = len(tasks)

        # 任务数超过模拟上限时，按优先级降序预选
        if n > self.max_qubits:
            candidate_indices = sorted(
                range(n),
                key=lambda i: _get_int(tasks[i], "priority", _DEFAULT_PRIORITY),
                reverse=True,
            )[: self.max_qubits]
            candidate_tasks = [tasks[i] for i in candidate_indices]
        else:
            candidate_indices = list(range(n))
            candidate_tasks = tasks

        # 构建 QUBO
        qubo = self._build_assignment_qubo(candidate_tasks, available_resources)
        self._last_qubo = qubo

        # QAOA 求解
        solution = self._solve_qaoa(qubo)
        self._last_solution = solution

        # 解码
        local_idx = self._decode_assignment(solution, len(candidate_tasks))

        # 回退：选择优先级最高的任务
        if local_idx < 0:
            local_idx = max(
                range(len(candidate_tasks)),
                key=lambda i: _get_int(candidate_tasks[i], "priority", _DEFAULT_PRIORITY),
            )

        return candidate_indices[local_idx]

    def _build_assignment_qubo(self, tasks: list[dict], available_resources: dict) -> np.ndarray:
        """构建任务分配 QUBO 矩阵。

        QUBO 编码：
            - 目标：选择高优先级、短执行时间、资源匹配的任务
            - 约束：恰好选择一个任务

        Args:
            tasks              : 候选任务列表
            available_resources : 可用资源字典

        Returns:
            n x n 的 QUBO 矩阵（对称）
        """
        n = len(tasks)
        qubo = np.zeros((n, n), dtype=np.float64)
        available_qubits = int(available_resources.get("qubits", 20))

        # 约束惩罚权重：确保恰好选择一个任务
        penalty = 10.0 * max(1, n)

        for i in range(n):
            priority = _get_int(tasks[i], "priority", _DEFAULT_PRIORITY)
            est_time = _get_float(tasks[i], "estimated_time", _DEFAULT_ESTIMATED_TIME)
            qubit_count = _get_int(tasks[i], "qubit_count", 0)

            # 资源不匹配惩罚：所需比特超过可用比特时加重惩罚
            resource_penalty = max(0, qubit_count - available_qubits) * 5.0

            # 目标成本（最小化）：高优先级 → 低成本，长执行时间 → 高成本
            objective_cost = -float(priority) * 3.0 + est_time * 0.3 + resource_penalty

            # 对角线：目标成本 + 约束（-penalty 鼓励 x_i=1）
            qubo[i, i] = objective_cost - penalty

            # 非对角线：约束（+2*penalty 惩罚同时选择两个任务）
            for j in range(i + 1, n):
                qubo[i, j] = penalty
                qubo[j, i] = penalty

        return qubo

    def _solve_qaoa(self, qubo: np.ndarray) -> np.ndarray:
        """使用 QAOA 求解 QUBO。

        Args:
            qubo: n x n QUBO 矩阵

        Returns:
            长度 n 的二值解向量（0/1）
        """
        n = qubo.shape[0]

        if n == 0:
            return np.array([], dtype=np.int8)
        if n == 1:
            return np.array([1], dtype=np.int8)

        # QUBO → Ising
        h, coupling = self._qubo_to_ising(qubo)

        # 预计算所有基态的 Ising 能量
        n_states = 2**n
        all_bits = np.array(
            [[(x >> i) & 1 for i in range(n)] for x in range(n_states)], dtype=np.float64
        )
        spins = 1.0 - 2.0 * all_bits  # 0→+1, 1→-1

        # E(x) = sum_i h_i * s_i + sum_{i<j} J_{ij} * s_i * s_j
        energies = spins @ h
        for i in range(n):
            for j in range(i + 1, n):
                if abs(coupling[i, j]) > 1e-15:
                    energies += coupling[i, j] * spins[:, i] * spins[:, j]

        # 网格搜索最优 (gamma, beta)
        best_prob = -1.0
        best_gamma = 0.0
        best_beta = 0.0
        best_probs: np.ndarray | None = None

        gammas = np.linspace(0.01, np.pi - 0.01, self.grid_size)
        betas = np.linspace(0.01, np.pi / 2 - 0.01, self.grid_size)

        for gamma in gammas:
            cost_phases = np.exp(-1j * gamma * energies)
            for beta in betas:
                probs = self._compute_probabilities_fast(cost_phases, all_bits, beta, n, n_states)
                # 有效解概率（恰好一个 1）
                valid_mask = np.sum(all_bits, axis=1) == 1
                valid_prob = float(np.sum(probs[valid_mask]))

                if valid_prob > best_prob:
                    best_prob = valid_prob
                    best_gamma = gamma
                    best_beta = beta
                    best_probs = probs

        self._last_gamma = best_gamma
        self._last_beta = best_beta
        self._last_valid_prob = best_prob

        if best_probs is None:
            # 回退：选择 QUBO 对角线最小值
            return self._greedy_solution(qubo)

        # 从有效解中选择概率最高的
        valid_indices = np.where(np.sum(all_bits, axis=1) == 1)[0]
        valid_probs_arr = best_probs[valid_indices]
        best_z = int(valid_indices[np.argmax(valid_probs_arr)])

        return np.array([(best_z >> i) & 1 for i in range(n)], dtype=np.int8)

    def _compute_probabilities_fast(
        self,
        cost_phases: np.ndarray,
        all_bits: np.ndarray,
        beta: float,
        n: int,
        n_states: int,
    ) -> np.ndarray:
        """快速计算 QAOA 概率分布。

        使用向量化操作计算所有基态的测量概率。

        Args:
            cost_phases : exp(-i * gamma * E(x))，形状 (n_states,)
            all_bits    : 所有基态的比特串，形状 (n_states, n)
            beta        : 混合参数
            n           : 量子比特数
            n_states    : 基态总数 (2^n)

        Returns:
            概率数组，形状 (n_states,)
        """
        cos_beta = np.cos(beta)
        sin_factor = -1j * np.sin(beta)
        amplitudes = np.zeros(n_states, dtype=np.complex128)

        for z in range(n_states):
            z_bits = all_bits[z]
            # 每个比特匹配时贡献 cos(beta)，不匹配时贡献 -i*sin(beta)
            matches = all_bits == z_bits[np.newaxis, :]  # (n_states, n)
            n_match = np.sum(matches, axis=1)  # (n_states,)
            n_mismatch = n - n_match
            mixer_vals = (cos_beta**n_match) * (sin_factor**n_mismatch)
            amplitudes[z] = np.sum(mixer_vals * cost_phases) / np.sqrt(n_states)

        probs = np.abs(amplitudes) ** 2
        total = float(probs.sum())
        if total > 1e-15:
            probs /= total
        return probs

    def _qubo_to_ising(self, qubo: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """将 QUBO 矩阵转换为 Ising 模型参数。

        转换关系：x_i = (1 - z_i) / 2，z_i ∈ {+1, -1}

        Args:
            qubo: n x n QUBO 矩阵

        Returns:
            (h, coupling): Ising 线性项和耦合项
        """
        n = qubo.shape[0]
        h = np.zeros(n, dtype=np.float64)
        coupling = np.zeros((n, n), dtype=np.float64)

        for i in range(n):
            h[i] = -qubo[i, i] / 2.0
            for j in range(n):
                if i != j:
                    h[i] -= qubo[i, j] / 4.0
                    coupling[i, j] = qubo[i, j] / 4.0

        return h, coupling

    def _decode_assignment(self, solution: np.ndarray, n_tasks: int) -> int:
        """将 QAOA 解码为任务索引。

        Args:
            solution : 二值解向量
            n_tasks  : 任务数

        Returns:
            被选中任务的索引；无有效解返回 -1
        """
        if len(solution) == 0:
            return -1

        ones = np.where(solution == 1)[0]
        if len(ones) == 0:
            return -1
        if len(ones) == 1:
            return int(ones[0])

        # 多个 1：取第一个（QAOA 近似解可能不完美）
        return int(ones[0])

    def _greedy_solution(self, qubo: np.ndarray) -> np.ndarray:
        """贪心回退：选择 QUBO 对角线最小值对应的变量。

        Args:
            qubo: QUBO 矩阵

        Returns:
            二值解向量
        """
        n = qubo.shape[0]
        solution = np.zeros(n, dtype=np.int8)
        if n > 0:
            solution[int(np.argmin(np.diag(qubo)))] = 1
        return solution

def get_all_baseline_schedulers() -> list[BaselineScheduler]:

    """返回所有基线调度策略的实例列表。

    Returns:

        包含 9 个基线策略实例的列表（FCFS/SPTF/EDF/Priority/RoundRobin/LIFO/HEFT/MinMin/QAOA）

    """

    return [

        FCFSScheduler(),

        SPTFScheduler(),

        EDFScheduler(),

        PriorityScheduler(),

        RoundRobinScheduler(),

        LIFOScheduler(),

        HEFTScheduler(),

        MinMinScheduler(),
        QAOAScheduler(),

    ]

def run_baseline_comparison(

    tasks: list[dict],

    num_steps: int = 100,

    use_env: bool = False,

    env_config: dict | None = None,

    seed: int = 42,

) -> dict[str, dict]:

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

                "comparison_mode": "legacy",

            }

        return results

    # use_env=True：Gymnasium 环境模式（Issue #230/#231）

    from src.scheduler.env import QuantumSchedulingEnv

    config = env_config or {}

    max_steps = config.get("max_steps", num_steps)

    max_qubits = config.get("max_qubits", 287)

    env_results: dict[str, dict] = {}

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

        done = False

        while not done:

            action = env_scheduler.select_action(obs, env)

            obs, reward, terminated, truncated, _info = env.step(action)

            total_reward += float(reward)

            completed += 1

            done = terminated or truncated

        env_results[env_scheduler.name] = {

            "total_reward": total_reward,

            "completed_tasks": completed,

            "avg_wait_time": 0.0,

            "throughput": completed / max_steps if max_steps > 0 else 0.0,

            "comparison_mode": "env_based",

        }

    return env_results

# ---------------------------------------------------------------------------

# EnvBasedScheduler：将基线策略封装为 Gymnasium 环境动作选择器（Issue #230/#270）

# ---------------------------------------------------------------------------

# 与 BaselineScheduler 的区别：

#   - BaselineScheduler.select_action(tasks, resources) -> int  操作任务列表索引

#   - EnvBasedScheduler.select_action(observation, env) -> int  操作 Gymnasium 动作空间

#

# 在 Gymnasium 环境中运行时，reward 由 env.step(action) 返回（env_reward.py 计算），

# 而非独立公式，确保基线与 PPO 在相同奖励函数下对比。

# 动作常量（与 env_types.py 保持一致）

_ACTION_CLASSICAL = 0

_ACTION_QUANTUM = 1

_ACTION_HYBRID = 2

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

    公共的观测解析逻辑由 ``_parse_obs`` 提供（Issue #387），子类复用以消除重复。

    Attributes:

        name: 策略名称，用于结果标识

    """

    def __init__(self, name: str) -> None:

        """初始化环境适配策略。

        Args:

            name : 策略名称

        """

        self.name = name

    def select_action(self, observation: np.ndarray, env: Any) -> int:

        """根据观测和环境状态选择动作。

        Args:

            observation: Gymnasium 环境的观测向量（16维）

            env        : QuantumSchedulingEnv 实例

        Returns:

            动作值（0=classical, 1=quantum, 2=hybrid）

        """

        raise NotImplementedError("子类必须实现 select_action")

    def _parse_obs(self, observation: np.ndarray) -> ObsInfo:

        """从观测向量解析公共字段（使用 OBS_ 常量，Issue #387）。"""

        obs = list(observation) if hasattr(observation, "__iter__") else []

        def _get(idx: int, default: float) -> float:

            return float(obs[idx]) if len(obs) > idx else default

        return ObsInfo(

            is_quantum=_get(OBS_TASK_TYPE_QUANTUM, 0.0) > 0.5,

            is_classical=_get(OBS_TASK_TYPE_CLASSICAL, 0.0) > 0.5,

            qubit_avail=_get(OBS_QUBIT_AVAILABILITY, 0.5),

            fidelity=_get(OBS_FIDELITY, 1.0),

            urgency=_get(OBS_URGENCY_LEVEL, 0.5),

            quantum_queue=_get(OBS_QUANTUM_QUEUE_RATIO, 0.0),

        )

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

        info = self._parse_obs(observation)

        if info.is_quantum and info.qubit_avail > 0.1:

            return _ACTION_QUANTUM

        if info.is_classical:

            return _ACTION_CLASSICAL

        # 无明确类型或量子资源不足，走经典

        return _ACTION_CLASSICAL

class EnvBasedSPTFScheduler(EnvBasedScheduler):

    """SPTF 策略的 Gymnasium 环境适配器。

    策略逻辑：最短处理时间优先。

    量子任务通常有加速比（speedup 2-5x），等效处理时间更短，

    因此量子任务可用时优先选择量子动作。

    """

    def __init__(self) -> None:

        super().__init__("SPTF")

    def select_action(self, observation: np.ndarray, env: Any) -> int:

        info = self._parse_obs(observation)

        # 量子任务且资源充足且保真度合格 -> 量子（等效时间最短）

        if info.is_quantum and info.qubit_avail > 0.2 and info.fidelity > 0.85:

            return _ACTION_QUANTUM

        # 量子任务但资源不足 -> 混合

        if info.is_quantum and info.qubit_avail > 0.05:

            return _ACTION_HYBRID

        # 默认经典

        return _ACTION_CLASSICAL

class EnvBasedEDFScheduler(EnvBasedScheduler):

    """EDF 策略的 Gymnasium 环境适配器。

    策略逻辑：最早截止时间优先。

    紧急任务（高 urgency）优先选择量子加速以尽快完成。

    """

    def __init__(self) -> None:

        super().__init__("EDF")

    def select_action(self, observation: np.ndarray, env: Any) -> int:

        info = self._parse_obs(observation)

        # 紧急任务且量子可用 -> 量子加速

        if info.urgency > 0.5 and info.qubit_avail > 0.1 and info.fidelity > 0.8:

            return _ACTION_QUANTUM

        # 量子任务且资源可用 -> 量子

        if info.is_quantum and info.qubit_avail > 0.2:

            return _ACTION_QUANTUM

        # 量子任务但资源紧张 -> 混合

        if info.is_quantum and info.qubit_avail > 0.05:

            return _ACTION_HYBRID

        # 默认经典

        return _ACTION_CLASSICAL

class EnvBasedGreedyScheduler(EnvBasedScheduler):

    """Greedy 策略的 Gymnasium 环境适配器。

    策略逻辑：贪心选择当前收益最高的动作。

    量子动作在保真度足够时有更高基础奖励（10*speedup），

    但保真度低于 0.9 时奖励打 6 折。

    """

    def __init__(self) -> None:

        super().__init__("Greedy")

    def select_action(self, observation: np.ndarray, env: Any) -> int:

        info = self._parse_obs(observation)

        # 量子任务且保真度高且资源充足 -> 量子（贪心追求最高奖励）

        if info.is_quantum and info.fidelity > 0.9 and info.qubit_avail > 0.2:

            return _ACTION_QUANTUM

        # 量子任务且保真度中等 -> 混合（避免低保真度打折）

        if info.is_quantum and info.fidelity > 0.85 and info.qubit_avail > 0.1:

            return _ACTION_HYBRID

        # 量子任务但条件不足 -> 经典

        if info.is_quantum:

            return _ACTION_CLASSICAL

        # 经典任务

        return _ACTION_CLASSICAL

class EnvBasedHEFTScheduler(EnvBasedScheduler):

    """HEFT（Heterogeneous Earliest Finish Time）策略的环境适配器（Issue #270）。

    HEFT 算法步骤：

    1. 计算每个任务的 upward rank（基于估算执行时间和后继依赖）

    2. 按 upward rank 降序排列任务

    3. 对每个任务，计算在每种"处理器"（classical/quantum/hybrid）上的最早完成时间

    4. 选择最早完成时间对应的动作

    在 Gymnasium 环境中，每步只需为当前任务选择一个动作，

    因此 HEFT 简化为：根据当前任务的估算执行时间和资源可用性，

    选择能最早完成该任务的动作。

    Note:

        传统 HEFT 用于 DAG 任务图调度，多处理器异构环境。

        本实现将其适配为单步动作选择，保留"最早完成时间优先"的核心思想。

    """

    # 量子加速比范围（与 env_types.py QUANTUM_SPEEDUP_RANGE 一致）

    _QUANTUM_SPEEDUP_MIN = 2.0

    _QUANTUM_SPEEDUP_MAX = 5.0

    def __init__(self) -> None:

        """初始化 HEFT 环境策略。"""

        super().__init__("EnvBased-HEFT")

    def _compute_upward_rank(

        self, estimated_time: float, is_quantum: bool, quantum_speedup: float

    ) -> float:

        """计算任务的 upward rank（简化版）。

        upward rank = 估算执行时间 + max(后继 upward rank)

        在单步决策中，后继信息不可用，简化为估算执行时间本身。

        Args:

            estimated_time  : 任务的估算执行时间

            is_quantum      : 是否为量子任务

            quantum_speedup : 量子加速比

        Returns:

            upward rank 值

        """

        equiv_time = estimated_time / max(quantum_speedup, 0.1) if is_quantum else estimated_time

        return equiv_time

    def _estimate_finish_time(

        self, action: int, estimated_time: float, quantum_speedup: float

    ) -> float:

        """估算任务在指定动作下的完成时间（Issue #584 修复）。

        修复点：增加负值校验，EFT < 0 时记录警告并回退到经典完成时间。

        Args:

            action          : 动作（0=classical, 1=quantum, 2=hybrid）

            estimated_time  : 任务的估算执行时间

            quantum_speedup : 量子加速比

        Returns:

            估算完成时间（保证非负）

        """

        if action == _ACTION_QUANTUM:

            ft = estimated_time / max(quantum_speedup, 0.1)

        elif action == _ACTION_HYBRID:

            # 混合执行：部分加速

            ft = estimated_time / max(quantum_speedup * 0.5, 0.1)

        else:

            # classical

            ft = estimated_time

        # EFT 校验：负值回退到经典完成时间（Issue #584）

        if ft < 0:

            logger.warning(

                f"HEFT EFT 计算异常为负值: action={action}, ft={ft}, "

                f"estimated_time={estimated_time}, speedup={quantum_speedup}, 回退到经典"

            )

            return estimated_time

        return ft

    def select_action(self, observation: np.ndarray, env: Any) -> int:

        """根据 HEFT 策略选择最早完成时间的动作（Issue #584 修复）。

        修复要点（根因：原实现对量子任务选经典动作导致 mismatch -2.0 惩罚累积）：

        1. 非量子任务直接走经典（兼容 classical/universal，避免量子 mismatch）

        2. 量子任务在资源充足时选量子，否则选混合（始终兼容，无机器时降级为经典执行）

        3. 永远不对量子任务选经典（mismatch）、不对非量子任务选量子（mismatch）

        4. EFT 校验：负值回退到 FCFS 式行为

        Args:

            observation: 16维观测向量

            env        : QuantumSchedulingEnv 实例

        Returns:

            动作值（0=classical, 1=quantum, 2=hybrid）

        """

        info = self._parse_obs(observation)

        # 非量子任务（classical/universal）：直接走经典（兼容，避免 mismatch）

        if not info.is_quantum:

            return _ACTION_CLASSICAL

        # 量子任务但资源严重不足：走混合（始终兼容，无机器时降级为经典执行）

        # 永远不对量子任务选经典（mismatch -2.0 惩罚）

        if info.qubit_avail < 0.1:

            return _ACTION_HYBRID

        # 估算量子加速比（从观测范围推算）

        speedup = self._QUANTUM_SPEEDUP_MIN + info.qubit_avail * (

            self._QUANTUM_SPEEDUP_MAX - self._QUANTUM_SPEEDUP_MIN

        )

        # 估算执行时间（从 urgency 推算，高 urgency = 短时间）

        estimated_time = max(1.0, 10.0 * (1.0 - info.urgency))

        # 计算兼容动作的完成时间（量子任务仅可选量子/混合，经典不兼容）

        finish_times: dict[int, float] = {}

        for action in (_ACTION_QUANTUM, _ACTION_HYBRID):

            ft = self._estimate_finish_time(action, estimated_time, speedup)

            # 量子动作需要考虑队列等待

            if action == _ACTION_QUANTUM:

                ft += info.quantum_queue * 5.0  # 队列惩罚

            elif action == _ACTION_HYBRID:

                ft += info.quantum_queue * 2.5

            finish_times[action] = ft

        # 选择最早完成时间的动作

        best_action = min(finish_times, key=lambda a: finish_times[a])

        # 量子动作的额外资源校验：资源或保真度不足时降级到混合（兼容）

        if best_action == _ACTION_QUANTUM and (info.qubit_avail < 0.3 or info.fidelity < 0.85):

            best_action = _ACTION_HYBRID

        return best_action

class EnvBasedMinMinScheduler(EnvBasedScheduler):

    """Min-Min 策略的环境适配器（Issue #270）。

    Min-Min 算法步骤：

    1. 对每个未分配任务，计算在每种处理器上的最小完成时间

    2. 选择所有任务中最小完成时间对应的任务-处理器对

    3. 将该任务分配到最佳处理器

    4. 重复直到所有任务分配完毕

    在 Gymnasium 环境中，每步只需为当前任务选择一个动作，

    因此 Min-Min 简化为：选择使当前任务完成时间最小的动作。

    与 HEFT 的区别：

    - HEFT 按 upward rank 排序后分配

    - Min-Min 每轮选择全局最小完成时间的任务

    在单步决策中，两者都简化为"选择最小完成时间的动作"，

    但 Min-Min 更倾向于选择短任务优先。

    """

    def __init__(self) -> None:

        """初始化 Min-Min 环境策略。"""

        super().__init__("EnvBased-MinMin")

    def select_action(self, observation: np.ndarray, env: Any) -> int:

        """根据 Min-Min 策略选择最小完成时间的动作。

        Min-Min 倾向于选择能最快完成的动作，与 HEFT 类似但

        更强调"最小"而非"最早"——即优先短任务和快速处理器。

        Args:

            observation: 16维观测向量

            env        : QuantumSchedulingEnv 实例

        Returns:

            动作值（0=classical, 1=quantum, 2=hybrid）

        """

        info = self._parse_obs(observation)

        # Min-Min: 选择"最小"完成时间的动作

        # 量子动作的等效时间最短（加速比 2-5x），但需考虑可用性和队列

        if info.is_quantum and info.qubit_avail > 0.3 and info.quantum_queue < 0.5:

            return _ACTION_QUANTUM

        if info.is_quantum and info.qubit_avail > 0.1:

            return _ACTION_HYBRID

        return _ACTION_CLASSICAL

def get_all_env_based_schedulers() -> list[EnvBasedScheduler]:

    """返回所有 Gymnasium 环境适配的基线策略实例列表（Issue #230/#270）。

    Returns:

        包含 6 个 EnvBasedScheduler 子类实例的列表（FCFS/SPTF/EDF/Greedy/HEFT/MinMin）

    """

    return [

        EnvBasedFCFSScheduler(),

        EnvBasedSPTFScheduler(),

        EnvBasedEDFScheduler(),

        EnvBasedGreedyScheduler(),

        EnvBasedHEFTScheduler(),

        EnvBasedMinMinScheduler(),

    ]

