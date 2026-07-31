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
from qiskit.converters import circuit_to_dag

# ---------------------------------------------------------------------------
# 默认常量（向后兼容，Issue #594 后可通过构造函数覆盖）
# ---------------------------------------------------------------------------
PHYSICAL_QUBITS = 16
GRID_ROWS, GRID_COLS = 4, 4

# 天衍-287 拓扑配置（105 数据比特，10x11 网格 = 110 物理比特）
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
        obs_version  : 观测语义版本，兼容 PR #759 前后模型权重。
                      "v759_post" (默认): 维度11-13 = avg_swap_dist_n/swap_efficiency/isolated_occupied_n (新特征)
                      "v759_pre"         : 维度11-13 = 1-mapped_r/1-alloc/1-conn (旧模型训练语义，
                                           加载 deliverable_models/ppo_compilation_agent.zip 时必须使用)
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
        obs_version: str = "v759_post",
    ) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.circuit = circuit
        self.n_logical = circuit.num_qubits if circuit else 8

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
        self.observation_space = spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32)
        # Issue #772: 观测语义版本化，兼容 PR #759 前的旧模型权重
        # "v759_post" (默认): 维度11-13 = 新特征 (avg_swap_dist_n/swap_efficiency/isolated_occupied_n)
        # "v759_pre"         : 维度11-13 = 1-mapped_r/1-alloc/1-conn (旧模型训练语义)
        if obs_version not in ("v759_pre", "v759_post"):
            raise ValueError(
                f"obs_version 必须为 'v759_pre' 或 'v759_post'，收到 {obs_version!r}"
            )
        self._obs_version = obs_version
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
        self._mapping, self._reverse_map = {}, {}
        self._mapped_gates, self._swap_count = 0, 0
        self._step_count, self._current_depth = 0, 0
        return self._get_obs(), {}

    def step(self, action: int) -> tuple[NDArray[np.float32], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        logical_idx = len(self._mapping)
        reward: float = 0.0
        terminated = False
        truncated = self._step_count >= self.max_steps
        if action in self._reverse_map:
            self._swap_count += 1
            reward -= 2
            free = [q for q in range(self.n_physical) if q not in self._reverse_map]
            if free:
                dist = min(self._distance_cache.get((action, fq), self.n_physical) for fq in free)
                self._swap_count += dist
                reward -= dist * 2
                actual = min(
                    free, key=lambda fq: self._distance_cache.get((action, fq), self.n_physical)
                )
            else:
                reward -= 50
                terminated = True
                actual = action
        else:
            actual = action
        self._mapping[logical_idx] = actual
        self._reverse_map[actual] = logical_idx
        self._mapped_gates += 1
        self._current_depth += 1
        reward += 1
        if len(self._mapping) >= self.n_logical:
            terminated = True
            swap_ratio = self._swap_count / max(1, self._n_gates)
            reward += 10 * (1 - min(swap_ratio, 1))
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> NDArray[np.float32]:
        """构建14维观测向量，包含电路特征、映射状态、拓扑指标和效率指标。"""
        nq_n = min(self.n_logical / 100.0, 1.0)
        gate_n = min(self._n_gates / 500.0, 1.0)
        two_q_n = self._two_q_ratio
        conn = 0.0
        if len(self._mapping) >= 2:
            matched = 0
            total_pairs = 0
            mapped_physical = list(self._mapping.values())
            for i, p1 in enumerate(mapped_physical):
                for p2 in mapped_physical[i + 1 :]:
                    total_pairs += 1
                    if p2 in self.coupling_graph.get(p1, set()):
                        matched += 1
            conn = matched / max(1, total_pairs)
        alloc = len(self._mapping) / self.n_physical
        occupied = set(self._reverse_map.keys())
        free_n = sum(
            1
            for q in range(self.n_physical)
            if q not in occupied and any(n in occupied for n in self.coupling_graph.get(q, set()))
        ) / max(1, self.n_physical)
        boundary_count = 0
        total_edges = 0
        for q in range(self.n_physical):
            for nb in self.coupling_graph.get(q, set()):
                if q < nb:
                    total_edges += 1
                    if (q in occupied) != (nb in occupied):
                        boundary_count += 1
        frag = boundary_count / max(1, total_edges)
        depth_n = min(self._current_depth / 100.0, 1.0)
        mapped_r = min(self._mapped_gates / max(1, self._n_gates), 1.0)
        swap_n = min(self._swap_count / max(1, self._n_gates), 1.0)
        avg_swap_dist = 0.0
        if self._swap_count > 0 and len(self._mapping) >= 2:
            dists = []
            for q in occupied:
                min_d = min(
                    (self._distance_cache.get((q, o), self.n_physical) for o in occupied if o != q),
                    default=0,
                )
                dists.append(min_d)
            avg_swap_dist = float(np.mean(dists)) if dists else 0.0
        fid = max(1.0 - 0.01 * self._swap_count - 0.005 * avg_swap_dist, 0.0)
        # Issue #656: 替换冗余反义维度(1-mapped_r/1-alloc/1-conn)为有信息量的特征
        avg_swap_dist_n = min(avg_swap_dist / max(1, self.n_physical - 1), 1.0)
        swap_efficiency = min(
            self._mapped_gates / max(1, self._mapped_gates + self._swap_count), 1.0
        )
        isolated_occupied = sum(
            1 for q in occupied if not any(n in occupied for n in self.coupling_graph.get(q, set()))
        )
        isolated_occupied_n = isolated_occupied / max(1, self.n_physical)
        # Issue #772: 观测语义版本化，兼容 PR #759 前的旧模型权重
        # 维度11-13 在 v759_post 为新特征，在 v759_pre 为旧模型训练语义(反义冗余特征)
        if self._obs_version == "v759_pre":
            dim11 = 1.0 - mapped_r
            dim12 = 1.0 - alloc
            dim13 = 1.0 - conn
        else:  # v759_post (默认)
            dim11 = avg_swap_dist_n
            dim12 = swap_efficiency
            dim13 = isolated_occupied_n
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
                dim11,
                dim12,
                dim13,
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
