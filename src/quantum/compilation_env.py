"""
QuantumCompilationEnv — PPO驱动的量子比特映射环境 (14维/可配置动作)

修复记录（Issue #404, #406, #594）：
    - #404: 耦合图从线性链改为 4x4 2D网格拓扑，匹配天衍真机 nearest-neighbor 结构
    - #406: SWAP距离使用图上最短路径而非 abs(p1-p2)；保真度模型改为距离感知
    - #594: PHYSICAL_QUBITS 从硬编码改为构造函数参数，支持自定义耦合图和天衍-287拓扑
"""

import warnings
from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

# 8.8 修复（P1-5）：qiskit 为编译实验可选依赖（requirements-dev.txt），
# 模块级 import 会使标准环境（仅装 requirements.txt）import 本模块即崩。
# 改为延迟导入：仅编译功能实际使用时才 import，并给出明确提示。
# from qiskit.converters import circuit_to_dag

# ---------------------------------------------------------------------------
# 默认常量（向后兼容，Issue #594 后可通过构造函数覆盖）
# ---------------------------------------------------------------------------
PHYSICAL_QUBITS = 16
GRID_ROWS, GRID_COLS = 4, 4

# 天衍-287 编译预设拓扑 10x11 网格（110 节点，可配置编译环境预设）；
# 天衍-287 物理比特口径为 105 数据 + 182 耦合（287 总比特），
# 该 110 节点网格为编译层的抽象耦合图预设，与物理比特拆解无一一对应关系。
TIANYAN_287_ROWS, TIANYAN_287_COLS = 10, 11
TIANYAN_287_QUBITS = TIANYAN_287_ROWS * TIANYAN_287_COLS  # 110


def _build_2d_grid_coupling(rows: int, cols: int) -> dict[int, set[int]]:
    """构建 2D 网格耦合图（nearest-neighbor 拓扑），匹配天衍真机结构。"""
    graph: dict[int, set[int]] = {i: set() for i in range(rows * cols)}
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                graph[q].add(q + 1)
                graph[q + 1].add(q)
            if r + 1 < rows:
                graph[q].add(q + cols)
                graph[q + cols].add(q)
    return graph


# 默认耦合图（4x4 网格，向后兼容）
COUPLING_GRAPH: dict[int, set[int]] = _build_2d_grid_coupling(GRID_ROWS, GRID_COLS)

# 天衍-287 耦合图（10x11 网格）
TIANYAN_287_COUPLING_GRAPH: dict[int, set[int]] = _build_2d_grid_coupling(
    TIANYAN_287_ROWS, TIANYAN_287_COLS
)


def _graph_distance(g: dict[int, set[int]], src: int, dst: int) -> int:
    """BFS 计算耦合图上两点的最短路径距离（SWAP 开销下界）。"""
    if src == dst:
        return 0
    visited = {src}
    queue: deque[tuple[int, int]] = deque([(src, 0)])
    while queue:
        node, dist = queue.popleft()
        for nb in g.get(node, set()):
            if nb == dst:
                return dist + 1
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, dist + 1))
    return len(g)


def _all_pairs_distances(g: dict[int, set[int]]) -> dict[tuple[int, int], int]:
    """预计算所有点对最短路径。"""
    dists: dict[tuple[int, int], int] = {}
    for src in g:
        for dst in g:
            if src == dst:
                dists[(src, dst)] = 0
            elif dst in g.get(src, set()):
                dists[(src, dst)] = 1
            else:
                dists[(src, dst)] = _graph_distance(g, src, dst)
    return dists


# 默认距离缓存（4x4 网格，向后兼容）
_DISTANCE_CACHE = _all_pairs_distances(COUPLING_GRAPH)


class QuantumCompilationEnv(gym.Env):
    """PPO驱动的量子比特映射环境，14维观测空间（编译层独立维度，区别于调度层OBS_DIM=16），可配置物理比特数和耦合图。

    Issue #594: PHYSICAL_QUBITS 从硬编码改为构造函数参数，支持自定义耦合图和天衍-287拓扑。

    Args:
        circuit      : Qiskit QuantumCircuit，None 时使用默认 8 逻辑比特
        max_steps    : 每个 episode 的最大步数
        n_physical   : 物理比特数（默认 PHYSICAL_QUBITS=16，向后兼容）
        coupling_graph: 自定义耦合图 {qubit: {neighbors}}，None 时根据 n_physical 推导
                        若 n_physical == TIANYAN_287_QUBITS 则使用 10x11 网格，
                        否则使用正方形网格（sqrt(n) x sqrt(n)）
        grid_rows    : 网格行数，与 grid_cols 一起用于构建耦合图（优先于自动推导）
        grid_cols    : 网格列数
    """

    metadata = {"render_modes": ["human"]}  # noqa: RUF012

    def __init__(
        self,
        circuit: Any | None = None,
        max_steps: int = 200,
        n_physical: int | None = None,
        coupling_graph: dict[int, set[int]] | None = None,
        grid_rows: int | None = None,
        grid_cols: int | None = None,
        swap_penalty: float = 2.0,
        distance_penalty: float = 2.0,
        no_free_qubit_penalty: float = 50.0,
        mapping_reward: float = 1.0,
        completion_reward_scale: float = 10.0,
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.circuit = circuit
        self.n_logical = circuit.num_qubits if circuit else 8

        # Issue #889: 奖励系数参数化（默认值保持既有行为，支持奖励消融实验）
        self.swap_penalty = float(swap_penalty)
        self.distance_penalty = float(distance_penalty)
        self.no_free_qubit_penalty = float(no_free_qubit_penalty)
        self.mapping_reward = float(mapping_reward)
        self.completion_reward_scale = float(completion_reward_scale)

        # Issue #594: 可配置物理比特数和耦合图
        self.n_physical = n_physical if n_physical is not None else PHYSICAL_QUBITS
        # 编译层观测空间为14维（逻辑/物理映射状态），与调度层OBS_DIM=16独立

        # 确定耦合图：优先使用自定义输入，其次根据网格参数构建，最后自动推导
        if coupling_graph is not None:
            self.coupling_graph = coupling_graph
            # 确保 n_physical 与耦合图一致
            self.n_physical = len(coupling_graph)
        elif grid_rows is not None and grid_cols is not None:
            self.coupling_graph = _build_2d_grid_coupling(grid_rows, grid_cols)
            self.n_physical = grid_rows * grid_cols
        elif self.n_physical == TIANYAN_287_QUBITS:
            # 天衍-287 预设：10x11 网格
            self.coupling_graph = TIANYAN_287_COUPLING_GRAPH
        else:
            # 自动推导正方形网格（如 16→4x4, 9→3x3）
            side = int(np.sqrt(self.n_physical))
            if side * side == self.n_physical:
                self.coupling_graph = _build_2d_grid_coupling(side, side)
            else:
                # Issue #659: 非完全平方数时发出警告，避免静默截断用户输入
                warnings.warn(
                    f"n_physical={self.n_physical} 不是完全平方数，"
                    f"请显式提供 grid_rows 和 grid_cols。"
                    f"回退到 {GRID_ROWS}x{GRID_COLS}={PHYSICAL_QUBITS} 比特网格。",
                    stacklevel=2,
                )
                self.coupling_graph = _build_2d_grid_coupling(GRID_ROWS, GRID_COLS)
                self.n_physical = PHYSICAL_QUBITS

        # 预计算距离缓存（实例级别，支持自定义耦合图）
        self._distance_cache = _all_pairs_distances(self.coupling_graph)
        # Issue #889: 预计算邻接矩阵，供 _get_obs 向量化连通性/边界/碎片化计算
        self._adj_matrix: NDArray[np.float64] = np.zeros(
            (self.n_physical, self.n_physical), dtype=np.float64
        )
        for q, neighbors in self.coupling_graph.items():
            for nb in neighbors:
                self._adj_matrix[q, nb] = 1.0
        # Issue #889: 预计算距离矩阵（n×n），供 avg_swap_dist 向量化计算
        self._dist_matrix: NDArray[np.float64] = np.array(
            [
                [self._distance_cache.get((i, j), self.n_physical) for j in range(self.n_physical)]
                for i in range(self.n_physical)
            ],
            dtype=np.float64,
        )
        self.observation_space = spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.n_physical)
        self._gates: list[Any] = []
        self._n_gates: int = 0
        self._two_q_ratio: float = 0.0
        self._mapping: dict[int, int] = {}
        self._reverse_map: dict[int, int] = {}
        self._mapped_gates: int = 0
        self._swap_count: int = 0
        self._step_count: int = 0
        self._current_depth: int = 0
        self._init_state()

    def _init_state(self) -> None:
        if self.circuit:
            try:
                from qiskit.converters import circuit_to_dag
            except ImportError as e:
                raise ImportError(
                    "编译功能需要 qiskit：请执行 pip install -r requirements-dev.txt"
                ) from e
            dag = circuit_to_dag(self.circuit)
            self._gates = list(dag.topological_op_nodes())
            self._n_gates = len(self._gates)
            two_q = sum(1 for g in self._gates if len(g.qargs) == 2)
            self._two_q_ratio = two_q / max(1, self._n_gates)
        else:
            self._gates, self._n_gates, self._two_q_ratio = [], 0, 0.0
        self._mapping, self._reverse_map = {}, {}
        self._mapped_gates, self._swap_count = 0, 0
        self._step_count, self._current_depth = 0, 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[NDArray[np.float32], dict[str, Any]]:
        super().reset(seed=seed)
        # Issue #883: 通过 _init_state 完整重置（含 _gates/_n_gates/_two_q_ratio），
        # 避免复用同一 env 实例重置时残留上一电路的门列表。
        self._init_state()
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        # Issue #883: 校验动作范围（Discrete(n_physical)），非法动作直接报错，
        # 避免映射到不存在的物理比特后观测/奖励静默失真。
        if not isinstance(action, (int, np.integer)) or not (0 <= int(action) < self.n_physical):
            raise ValueError(f"非法动作 {action!r}：动作必须在 [0, {self.n_physical}) 范围内")
        action = int(action)
        self._step_count += 1
        logical_idx = len(self._mapping)
        reward: float = 0.0
        terminated = False
        truncated = self._step_count >= self.max_steps
        if action in self._reverse_map:
            self._swap_count += 1
            reward -= self.swap_penalty
            free = [q for q in range(self.n_physical) if q not in self._reverse_map]
            if free:
                dist = min(self._distance_cache.get((action, fq), self.n_physical) for fq in free)
                self._swap_count += dist
                reward -= dist * self.distance_penalty
                actual = min(
                    free, key=lambda fq: self._distance_cache.get((action, fq), self.n_physical)
                )
            else:
                reward -= self.no_free_qubit_penalty
                terminated = True
                actual = action
        else:
            actual = action
        self._mapping[logical_idx] = actual
        self._reverse_map[actual] = logical_idx
        self._mapped_gates += 1
        self._current_depth += 1
        reward += self.mapping_reward
        if len(self._mapping) >= self.n_logical:
            terminated = True
            swap_ratio = self._swap_count / max(1, self._n_gates)
            reward += self.completion_reward_scale * (1 - min(swap_ratio, 1))
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> NDArray[np.float32]:
        """构建14维观测向量，包含电路特征、映射状态、拓扑指标和效率指标。

        Issue #889: 连通性/边界碎片化/平均SWAP距离均用预计算矩阵向量化，
        将 _get_obs 的 O(n²) 逐元素循环改为 numpy 矩阵运算（天衍-287
        n_physical=110 时从 ~12100 次 Python 迭代降至矩阵乘法），
        数值结果与旧的循环实现完全一致，不改变观测语义。
        """
        nq_n = min(self.n_logical / 100.0, 1.0)
        gate_n = min(self._n_gates / 500.0, 1.0)
        two_q_n = self._two_q_ratio
        conn = 0.0
        if len(self._mapping) >= 2:
            mapped = np.asarray(list(self._mapping.values()), dtype=np.int64)
            k = mapped.size
            # 向量化连通性：sub_adj 对称且对角为 0，每条无向边被计两次；
            # 原循环 total_pairs=k(k-1)/2、matched=边数，conn=matched/total_pairs
            sub_adj = self._adj_matrix[np.ix_(mapped, mapped)]
            conn = float(sub_adj.sum() / max(1, k * (k - 1)))
        alloc = len(self._mapping) / self.n_physical
        occupied = set(self._reverse_map.keys())
        occ_mask = np.zeros(self.n_physical, dtype=bool)
        if occupied:
            occ_mask[list(occupied)] = True
        # 向量化 free_n：空闲且至少有一个被占用邻居的物理比特占比
        has_occ_neighbor = self._adj_matrix @ occ_mask > 0
        free_n = float(np.sum(~occ_mask & has_occ_neighbor) / max(1, self.n_physical))
        # 向量化 frag：被占用状态不同的边界边数 / 总边数
        upper = np.triu(self._adj_matrix, 1)
        total_edges = float(upper.sum())
        boundary_count = float(np.sum(upper * (occ_mask[:, None] != occ_mask[None, :])))
        frag = boundary_count / max(1, total_edges)
        depth_n = min(self._current_depth / 100.0, 1.0)
        mapped_r = min(self._mapped_gates / max(1, self._n_gates), 1.0)
        swap_n = min(self._swap_count / max(1, self._n_gates), 1.0)
        avg_swap_dist = 0.0
        if self._swap_count > 0 and len(self._mapping) >= 2:
            occ_idx = np.asarray(sorted(occupied), dtype=np.int64)
            sub_dist = self._dist_matrix[np.ix_(occ_idx, occ_idx)]
            np.fill_diagonal(sub_dist, np.inf)
            avg_swap_dist = float(np.mean(sub_dist.min(axis=1)))
        fid = max(1.0 - 0.01 * self._swap_count - 0.005 * avg_swap_dist, 0.0)
        # Issue #656/#841: 暂保留旧反义维度(1-mapped_r/1-alloc/1-conn)以匹配已训练的 ppo_compilation_agent.zip
        # 新维度(avg_swap_dist_n/swap_efficiency/isolated_occupied_n)虽更合理，但需重训练模型，8/15冻结前回退
        return np.array(
            [
                nq_n,
                gate_n,
                two_q_n,
                conn,
                alloc,
                free_n,
                frag,
                depth_n,
                mapped_r,
                swap_n,
                fid,
                1.0 - mapped_r,
                1.0 - alloc,
                1.0 - conn,
            ],
            dtype=np.float32,
        )

    def get_stats(self) -> dict[str, int]:
        return {
            "n_logical": self.n_logical,
            "n_physical": self.n_physical,
            "swap_count": self._swap_count,
            "mapped_gates": self._mapped_gates,
        }

    def get_metrics(self) -> dict[str, float]:
        """返回 episode 关键指标（Issue #889 可观测性补充）。

        包含映射完成率、SWAP 率、连通性、边界碎片化与保真度估计，
        供训练循环/日志记录 episode 级性能。
        """
        mapped_r = min(self._mapped_gates / max(1, self._n_gates), 1.0)
        swap_ratio = self._swap_count / max(1, self._n_gates)
        conn = 0.0
        if len(self._mapping) >= 2:
            mapped = np.asarray(list(self._mapping.values()), dtype=np.int64)
            k = mapped.size
            sub_adj = self._adj_matrix[np.ix_(mapped, mapped)]
            conn = float(sub_adj.sum() / max(1, k * (k - 1)))
        occ_mask = np.asarray([q in self._reverse_map for q in range(self.n_physical)], dtype=bool)
        upper = np.triu(self._adj_matrix, 1)
        frag = float(np.sum(upper * (occ_mask[:, None] != occ_mask[None, :])) / max(1, upper.sum()))
        return {
            "mapped_gates": float(self._mapped_gates),
            "mapped_ratio": mapped_r,
            "swap_count": float(self._swap_count),
            "swap_ratio": swap_ratio,
            "connectivity": conn,
            "fragmentation": frag,
            "fidelity": max(1.0 - 0.01 * self._swap_count, 0.0),
        }
