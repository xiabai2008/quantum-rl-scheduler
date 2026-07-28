"""
QuantumCompilationEnv — PPO驱动的量子比特映射环境 (14维/16动作)

修复记录（Issue #404, #406）：
    - #404: 耦合图从线性链改为 4x4 2D网格拓扑，匹配天衍真机 nearest-neighbor 结构
    - #406: SWAP距离使用图上最短路径而非 abs(p1-p2)；保真度模型改为距离感知
"""

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from qiskit.converters import circuit_to_dag

PHYSICAL_QUBITS = 16
GRID_ROWS, GRID_COLS = 4, 4


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


COUPLING_GRAPH: dict[int, set[int]] = _build_2d_grid_coupling(GRID_ROWS, GRID_COLS)


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


_DISTANCE_CACHE = _all_pairs_distances(COUPLING_GRAPH)


class QuantumCompilationEnv(gym.Env):
    """PPO驱动的量子比特映射环境，14维观测空间，16个离散动作。"""

    metadata = {"render_modes": ["human"]}  # noqa: RUF012

    def __init__(self, circuit: Any | None = None, max_steps: int = 200) -> None:
        super().__init__()
        self.max_steps = max_steps
        self.circuit = circuit
        self.n_logical = circuit.num_qubits if circuit else 8
        self.n_physical = PHYSICAL_QUBITS
        self.observation_space = spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32)
        self.action_space = spaces.Discrete(PHYSICAL_QUBITS)
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
                dist = min(_DISTANCE_CACHE.get((action, fq), self.n_physical) for fq in free)
                self._swap_count += dist
                reward -= dist * 2
                actual = min(
                    free, key=lambda fq: _DISTANCE_CACHE.get((action, fq), self.n_physical)
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
        reward += 1
        if len(self._mapping) >= self.n_logical:
            terminated = True
            swap_ratio = self._swap_count / max(1, self._n_gates)
            reward += 10 * (1 - min(swap_ratio, 1))
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self) -> NDArray[np.float32]:
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
                    if p2 in COUPLING_GRAPH.get(p1, set()):
                        matched += 1
            conn = matched / max(1, total_pairs)
        alloc = len(self._mapping) / self.n_physical
        occupied = set(self._reverse_map.keys())
        free_n = sum(
            1
            for q in range(self.n_physical)
            if q not in occupied and any(n in occupied for n in COUPLING_GRAPH.get(q, set()))
        ) / max(1, self.n_physical)
        boundary_count = 0
        total_edges = 0
        for q in range(self.n_physical):
            for nb in COUPLING_GRAPH.get(q, set()):
                if q < nb:
                    total_edges += 1
                    if (q in occupied) != (nb in occupied):
                        boundary_count += 1
        frag = boundary_count / max(1, total_edges)
        depth_n = min(self._current_depth / 100.0, 1.0)
        mapped_r = self._mapped_gates / max(1, self._n_gates)
        swap_n = min(self._swap_count / max(1, self._n_gates), 1.0)
        avg_swap_dist = 0.0
        if self._swap_count > 0 and len(self._mapping) >= 2:
            dists = []
            for q in occupied:
                min_d = min(
                    (_DISTANCE_CACHE.get((q, o), self.n_physical) for o in occupied if o != q),
                    default=0,
                )
                dists.append(min_d)
            avg_swap_dist = float(np.mean(dists)) if dists else 0.0
        fid = max(1.0 - 0.01 * self._swap_count - 0.005 * avg_swap_dist, 0.0)
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
