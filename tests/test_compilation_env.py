"""QuantumCompilationEnv 核心功能与边界场景测试 (Issue #382)。

覆盖 reset/step/reward/终止条件/边界场景，验证 PPO 驱动的
量子比特映射环境的 14 维观测、16 动作空间、SWAP 计数与奖励逻辑。
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from src.quantum.compilation_env import (
    COUPLING_GRAPH,
    PHYSICAL_QUBITS,
    QuantumCompilationEnv,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def env_no_circuit() -> QuantumCompilationEnv:
    """无电路环境（n_logical=8，无门）。"""
    return QuantumCompilationEnv(circuit=None, max_steps=200)


@pytest.fixture
def env_short() -> QuantumCompilationEnv:
    """短 max_steps 环境，用于测试截断。"""
    return QuantumCompilationEnv(circuit=None, max_steps=3)


# ---------------------------------------------------------------------------
# reset 测试
# ---------------------------------------------------------------------------


class TestReset:
    """reset() 行为验证。"""

    def test_reset_returns_14dim_observation(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """reset 应返回 14 维 float32 观测向量。"""
        obs, _info = env_no_circuit.reset(seed=42)
        assert obs.shape == (14,)
        assert obs.dtype == np.float32
        assert env_no_circuit.observation_space.contains(obs)

    def test_reset_returns_empty_info(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """reset 的 info 应为空字典。"""
        _, info = env_no_circuit.reset(seed=42)
        assert info == {}

    def test_reset_clears_internal_state(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """reset 后映射、SWAP 计数、步数应归零。"""
        env_no_circuit.reset(seed=1)
        env_no_circuit.step(0)
        env_no_circuit.step(1)
        env_no_circuit.reset(seed=2)
        assert env_no_circuit._mapping == {}
        assert env_no_circuit._reverse_map == {}
        assert env_no_circuit._swap_count == 0
        assert env_no_circuit._step_count == 0
        assert env_no_circuit._mapped_gates == 0

    def test_reset_observation_values_in_range(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """所有观测值应在 [0, 1] 范围内。"""
        obs, _ = env_no_circuit.reset(seed=42)
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)


# ---------------------------------------------------------------------------
# 动作空间测试
# ---------------------------------------------------------------------------


class TestActionSpace:
    """动作空间验证。"""

    def test_action_space_is_discrete_16(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """动作空间应为 Discrete(16)。"""
        assert isinstance(env_no_circuit.action_space, spaces.Discrete)
        assert env_no_circuit.action_space.n == PHYSICAL_QUBITS
        assert env_no_circuit.action_space.n == 16


# ---------------------------------------------------------------------------
# step 测试
# ---------------------------------------------------------------------------


class TestStep:
    """step() 行为验证。"""

    def test_step_valid_action_maps_logical_to_physical(
        self, env_no_circuit: QuantumCompilationEnv
    ) -> None:
        """合法动作应将逻辑比特映射到物理比特。"""
        env_no_circuit.reset(seed=42)
        obs, reward, terminated, truncated, _info = env_no_circuit.step(5)

        assert env_no_circuit._mapping[0] == 5
        assert env_no_circuit._reverse_map[5] == 0
        assert env_no_circuit._step_count == 1
        assert env_no_circuit.observation_space.contains(obs)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)

    def test_step_conflict_triggers_swap(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """重复映射已占用的物理比特应触发 SWAP。"""
        env_no_circuit.reset(seed=42)
        env_no_circuit.step(3)  # 映射 logical 0 -> physical 3
        _, reward, _, _, _ = env_no_circuit.step(3)  # 冲突

        assert env_no_circuit._swap_count > 0
        assert reward < 0  # SWAP 惩罚

    def test_swap_count_increments_correctly(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """SWAP 计数应正确递增（含距离惩罚）。"""
        env_no_circuit.reset(seed=42)
        env_no_circuit.step(0)  # logical 0 -> physical 0
        # 冲突：physical 0 已占用，最近空闲是 1，距离=1
        env_no_circuit.step(0)
        # 基础 SWAP +1，距离 SWAP +1 => swap_count=2
        assert env_no_circuit._swap_count == 2

    def test_step_returns_14dim_observation(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """step 返回的观测应为 14 维。"""
        env_no_circuit.reset(seed=42)
        obs, _, _, _, _ = env_no_circuit.step(0)
        assert obs.shape == (14,)
        assert obs.dtype == np.float32

    def test_step_increments_mapped_gates(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """每次成功映射应递增 mapped_gates。"""
        env_no_circuit.reset(seed=42)
        assert env_no_circuit._mapped_gates == 0
        env_no_circuit.step(0)
        assert env_no_circuit._mapped_gates == 1
        env_no_circuit.step(1)
        assert env_no_circuit._mapped_gates == 2


# ---------------------------------------------------------------------------
# 电路深度观测维度测试 (Issue #652)
# ---------------------------------------------------------------------------


class TestDepthObservation:
    """观测维度7（depth_n）应随映射进度递增，而非恒为0。"""

    def test_depth_zero_after_reset(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """reset 后 _current_depth 应为0。"""
        env_no_circuit.reset(seed=42)
        assert env_no_circuit._current_depth == 0

    def test_depth_increments_on_step(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """每次成功映射后 _current_depth 应递增。"""
        env_no_circuit.reset(seed=42)
        assert env_no_circuit._current_depth == 0
        env_no_circuit.step(0)
        assert env_no_circuit._current_depth == 1
        env_no_circuit.step(1)
        assert env_no_circuit._current_depth == 2

    def test_depth_observation_nonzero_after_step(
        self, env_no_circuit: QuantumCompilationEnv
    ) -> None:
        """step 后观测维度7（depth_n）应 > 0。"""
        env_no_circuit.reset(seed=42)
        obs_before, _, _, _, _ = env_no_circuit.step(0)
        # depth_n 是 obs[7]，归一化为 min(depth/100, 1.0)
        assert obs_before[7] > 0.0, "depth_n should be > 0 after a step"

    def test_depth_observation_increases_with_steps(
        self, env_no_circuit: QuantumCompilationEnv
    ) -> None:
        """多次 step 后 depth_n 应单调递增。"""
        env_no_circuit.reset(seed=42)
        depths: list[float] = []
        for i in range(5):
            obs, _, _, _, _ = env_no_circuit.step(i)
            depths.append(float(obs[7]))
        for i in range(len(depths) - 1):
            assert depths[i + 1] > depths[i], "depth_n should monotonically increase"

    def test_depth_resets_on_reset(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """reset 应将 _current_depth 归零。"""
        env_no_circuit.reset(seed=42)
        env_no_circuit.step(0)
        env_no_circuit.step(1)
        assert env_no_circuit._current_depth > 0
        env_no_circuit.reset(seed=42)
        assert env_no_circuit._current_depth == 0


# ---------------------------------------------------------------------------
# 终止条件测试
# ---------------------------------------------------------------------------


class TestTermination:
    """episode 终止条件验证。"""

    def test_episode_terminates_on_full_mapping(
        self, env_no_circuit: QuantumCompilationEnv
    ) -> None:
        """所有逻辑比特映射完成后应 terminated=True。"""
        env_no_circuit.reset(seed=42)
        n_logical = env_no_circuit.n_logical
        terminated = False
        for i in range(n_logical):
            _, _, terminated, _, _ = env_no_circuit.step(i)
        assert terminated is True

    def test_episode_truncates_on_max_steps(self, env_short: QuantumCompilationEnv) -> None:
        """到达 max_steps 应 truncated=True。"""
        env_short.reset(seed=42)
        truncated = False
        for _ in range(env_short.max_steps):
            _, _, _, truncated, _ = env_short.step(0)
        assert truncated is True

    def test_not_terminated_before_full_mapping(
        self, env_no_circuit: QuantumCompilationEnv
    ) -> None:
        """未完成全部映射时不应 terminated。"""
        env_no_circuit.reset(seed=42)
        _, _, terminated, _, _ = env_no_circuit.step(0)
        assert terminated is False


# ---------------------------------------------------------------------------
# 奖励函数测试
# ---------------------------------------------------------------------------


class TestReward:
    """奖励函数数值正确性验证。"""

    def test_reward_mapping_success_positive(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """成功映射应给出正奖励（+1）。"""
        env_no_circuit.reset(seed=42)
        _, reward, _, _, _ = env_no_circuit.step(0)
        assert reward == pytest.approx(1.0)

    def test_reward_conflict_swap_negative(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """冲突 SWAP 应给出负奖励（-2 基础 + 距离惩罚 + 1 映射奖励）。"""
        env_no_circuit.reset(seed=42)
        env_no_circuit.step(0)  # 首次映射 reward=+1
        # 冲突：base -2 + dist*2 penalty + 1 mapping = -2 + 2 + 1 = -1（dist=1）
        _, reward, _, _, _ = env_no_circuit.step(0)
        assert reward < 0

    def test_reward_completion_scales_with_swap_ratio(
        self, env_no_circuit: QuantumCompilationEnv
    ) -> None:
        """完成奖励应与 SWAP 比率反相关（swap 越少奖励越高）。"""
        # 无 SWAP 完成：全部用不同物理比特
        env_clean = QuantumCompilationEnv(circuit=None, max_steps=200)
        env_clean.reset(seed=42)
        total_reward_clean = 0.0
        for i in range(env_clean.n_logical):
            _, r, _, _, _ = env_clean.step(i)
            total_reward_clean += r

        # 有 SWAP 完成：重复使用同一物理比特
        env_swap = QuantumCompilationEnv(circuit=None, max_steps=200)
        env_swap.reset(seed=42)
        total_reward_swap = 0.0
        for _i in range(env_swap.n_logical):
            _, r, _, _, _ = env_swap.step(0)
            total_reward_swap += r

        assert total_reward_clean > total_reward_swap

    def test_no_free_qubit_terminates_with_large_penalty(
        self, env_no_circuit: QuantumCompilationEnv
    ) -> None:
        """所有物理比特被占用且再次冲突时应 terminated=True 并给大惩罚。"""
        env = QuantumCompilationEnv(circuit=None, max_steps=200)
        env.n_logical = PHYSICAL_QUBITS + 1  # 强制超过物理比特数
        env.reset(seed=42)
        # 映射全部 16 个物理比特
        for i in range(PHYSICAL_QUBITS):
            env.step(i)
        # 第 17 次映射：无空闲物理比特
        _, reward, terminated, _, _ = env.step(0)
        assert terminated is True
        assert reward < -40  # -50 大惩罚 + 1 映射奖励


# ---------------------------------------------------------------------------
# 边界场景测试
# ---------------------------------------------------------------------------


class TestBoundaryCases:
    """空电路、单门电路等边界场景。"""

    def test_empty_circuit_boundary(self) -> None:
        """空电路（circuit=None）应正常初始化，n_logical=8。"""
        env = QuantumCompilationEnv(circuit=None, max_steps=10)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (14,)
        assert env.n_logical == 8
        assert env._n_gates == 0
        assert env._two_q_ratio == 0.0
        # 空电路仍可正常 step
        obs, reward, _terminated, _truncated, _ = env.step(0)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)

    def test_single_gate_circuit_boundary(self) -> None:
        """单门电路应正确解析门数和双比特门比率。"""
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            pytest.skip("qiskit not installed")

        qc = QuantumCircuit(3)
        qc.h(0)
        env = QuantumCompilationEnv(circuit=qc, max_steps=50)
        env.reset(seed=42)
        assert env.n_logical == 3
        assert env._n_gates == 1
        assert env._two_q_ratio == 0.0  # H 是单比特门

    def test_two_qubit_gate_circuit(self) -> None:
        """包含双比特门的电路应正确计算 two_q_ratio。"""
        try:
            from qiskit import QuantumCircuit
        except ImportError:
            pytest.skip("qiskit not installed")

        qc = QuantumCircuit(3)
        qc.h(0)
        qc.cx(0, 1)  # 双比特门
        qc.cx(1, 2)  # 双比特门
        env = QuantumCompilationEnv(circuit=qc, max_steps=50)
        env.reset(seed=42)
        assert env._n_gates == 3
        assert env._two_q_ratio == pytest.approx(2 / 3)

    def test_coupling_graph_is_2d_grid(self) -> None:
        """耦合图应为 4x4 2D网格拓扑（Issue #404 修复）。"""
        assert PHYSICAL_QUBITS == 16
        rows, cols = 4, 4
        for r in range(rows):
            for c in range(cols):
                q = r * cols + c
                neighbors = COUPLING_GRAPH[q]
                if c + 1 < cols:
                    assert (q + 1) in neighbors
                    assert q in COUPLING_GRAPH[q + 1]
                if r + 1 < rows:
                    assert (q + cols) in neighbors
                    assert q in COUPLING_GRAPH[q + cols]
        corners = [0, cols - 1, (rows - 1) * cols, rows * cols - 1]
        for corner in corners:
            assert len(COUPLING_GRAPH[corner]) == 2


# ---------------------------------------------------------------------------
# 观测空间内容测试
# ---------------------------------------------------------------------------


class TestObservation:
    """观测向量各维度含义验证。"""

    def test_observation_dims_11_13_non_redundant(
        self, env_no_circuit: QuantumCompilationEnv
    ) -> None:
        """观测维度11-13应提供非冗余信息，不再是1-x反义维度（Issue #656）。"""
        env_no_circuit.reset(seed=42)
        # step(0) 映射 logical 0 -> physical 0；step(0) 冲突 SWAP 到 physical 1
        env_no_circuit.step(0)
        obs, _, _, _, _ = env_no_circuit.step(0)
        # 维度11: avg_swap_dist_n（SWAP距离归一化，>0 因有SWAP）
        assert obs[11] > 0.0, "avg_swap_dist_n 应在SWAP后 > 0"
        assert obs[11] != pytest.approx(1.0 - obs[8]), "obs[11] 不应等于 1-mapped_r"
        # 维度12: swap_efficiency = mapped_gates / (mapped_gates + swap_count)
        assert obs[12] == pytest.approx(0.5), "swap_efficiency 应为 0.5（2次映射/4次总操作）"
        assert obs[12] != pytest.approx(1.0 - obs[4]), "obs[12] 不应等于 1-alloc"
        # 维度13: isolated_occupied_n（0和1相邻，无隔离占用）
        assert obs[13] == pytest.approx(0.0), "相邻映射的 isolated_occupied_n 应为 0"

    def test_observation_isolated_occupied_scattered(
        self, env_no_circuit: QuantumCompilationEnv
    ) -> None:
        """分散映射时维度13（隔离占用比例）应 > 0，区别于1-conn（Issue #656）。"""
        env_no_circuit.reset(seed=42)
        # physical 0 和 5 在4x4网格中不相邻
        env_no_circuit.step(0)
        obs, _, _, _, _ = env_no_circuit.step(5)
        assert obs[13] > 0.0, "分散映射时 isolated_occupied_n 应 > 0"
        assert obs[13] != pytest.approx(1.0 - obs[3]), "obs[13] 不应等于 1-conn"

    def test_observation_updates_after_step(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """step 后观测应反映映射状态变化。"""
        env_no_circuit.reset(seed=42)
        obs_before = env_no_circuit._get_obs()
        env_no_circuit.step(0)
        obs_after = env_no_circuit._get_obs()
        # alloc 维度（索引 4）应增加
        assert obs_after[4] > obs_before[4]

    def test_get_stats_returns_correct_dict(self, env_no_circuit: QuantumCompilationEnv) -> None:
        """get_stats 应返回正确的统计字典。"""
        env_no_circuit.reset(seed=42)
        env_no_circuit.step(0)
        env_no_circuit.step(1)
        stats = env_no_circuit.get_stats()
        assert stats["n_logical"] == 8
        assert stats["n_physical"] == 16
        assert stats["mapped_gates"] == 2
        assert stats["swap_count"] == 0
