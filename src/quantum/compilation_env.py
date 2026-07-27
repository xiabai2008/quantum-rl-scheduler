"""
QuantumCompilationEnv — PPO驱动的量子比特映射环境 (14维/16动作)
"""

from typing import ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from qiskit.converters import circuit_to_dag

PHYSICAL_QUBITS = 16
COUPLING_GRAPH = {i: set() for i in range(PHYSICAL_QUBITS)}
for i in range(PHYSICAL_QUBITS - 1):
    COUPLING_GRAPH[i].add(i + 1)
    COUPLING_GRAPH[i + 1].add(i)


class QuantumCompilationEnv(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": ["human"]}

    def __init__(self, circuit=None, max_steps=200):
        super().__init__()
        self.max_steps = max_steps
        self.circuit = circuit
        self.n_logical = circuit.num_qubits if circuit else 8
        self.n_physical = PHYSICAL_QUBITS
        self.observation_space = spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32)
        self.action_space = spaces.Discrete(PHYSICAL_QUBITS)
        self._init_state()

    def _init_state(self):
        if self.circuit:
            dag = circuit_to_dag(self.circuit)
            self._gates = list(dag.topological_op_nodes())
            self._n_gates = len(self._gates)
            two_q = sum(1 for g in self._gates if len(g.qargs) == 2)
            self._two_q_ratio = two_q / max(1, self._n_gates)
        else:
            self._gates, self._n_gates, self._two_q_ratio = [], 0, 0
        self._mapping, self._reverse_map = {}, {}
        self._mapped_gates, self._swap_count = 0, 0
        self._step_count, self._current_depth = 0, 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._mapping, self._reverse_map = {}, {}
        self._mapped_gates, self._swap_count = 0, 0
        self._step_count, self._current_depth = 0, 0
        return self._get_obs(), {}

    def step(self, action: int):
        self._step_count += 1
        logical_idx = len(self._mapping)
        reward, terminated = 0, False
        truncated = self._step_count >= self.max_steps
        if action in self._reverse_map:
            self._swap_count += 1
            reward -= 2
            free = [q for q in range(self.n_physical) if q not in self._reverse_map]
            if free:
                dist = min(abs(action - fq) for fq in free)
                self._swap_count += dist
                reward -= dist * 2
                actual = free[0]
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

    def _get_obs(self):
        nq_n = min(self.n_logical / 100.0, 1.0)
        gate_n = min(self._n_gates / 500.0, 1.0)
        two_q_n = self._two_q_ratio
        conn = 0.0
        if len(self._mapping) >= 2:
            matched = sum(
                1
                for l1, p1 in self._mapping.items()
                for l2, p2 in self._mapping.items()
                if l1 < l2 and abs(p1 - p2) <= 1
            )
            total = len(self._mapping) * (len(self._mapping) - 1) // 2
            conn = matched / max(1, total)
        alloc = len(self._mapping) / self.n_physical
        free_n = sum(
            1
            for q in range(self.n_physical)
            if q not in self._reverse_map
            and any(n in self._reverse_map for n in COUPLING_GRAPH.get(q, set()))
        ) / max(1, self.n_physical)
        frag = (
            1.0
            if len(self._reverse_map) == 0
            else sum(
                1
                for q in range(self.n_physical - 1)
                if (q in self._reverse_map) != (q + 1 in self._reverse_map)
            )
            / (self.n_physical - 1)
        )
        depth_n = min(self._current_depth / 100.0, 1.0)
        mapped_r = self._mapped_gates / max(1, self._n_gates)
        swap_n = min(self._swap_count / max(1, self._n_gates), 1.0)
        fid = max(1.0 - 0.01 * self._swap_count, 0.0)
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

    def get_stats(self):
        return {
            "n_logical": self.n_logical,
            "n_physical": self.n_physical,
            "swap_count": self._swap_count,
            "mapped_gates": self._mapped_gates,
        }
