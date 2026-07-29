"""公平性惩罚和观测的单元测试（Issue #587, #588, #585）。"""

from __future__ import annotations

import numpy as np
import pytest

from src.scheduler.env_reward import compute_execution_reward, compute_fairness_penalty
from src.scheduler.env_types import (
    ACTION_CLASSICAL,
    FAIRNESS_PENALTY_FACTOR,
    FAIRNESS_PENALTY_THRESHOLD,
    OBS_DIM,
    OBS_DIM_WITH_FAIRNESS,
    Task,
)


def _make_task() -> Task:
    """创建测试用任务。"""
    return Task(
        task_id="test-001",
        task_type="quantum",
        qubit_count=4,
        urgency=0.5,
        priority=3,
        tenant_id="tenant_a",
    )


class TestComputeFairnessPenalty:
    """Issue #587: 公平性惩罚函数测试。"""

    def test_no_penalty_when_single_tenant(self) -> None:
        """单个租户时无惩罚（数据不足）。"""
        result = compute_fairness_penalty("tenant_a", {"tenant_a": 5.0})
        assert result == 0.0

    def test_no_penalty_when_empty_dict(self) -> None:
        """空字典时无惩罚。"""
        result = compute_fairness_penalty("tenant_a", None)
        assert result == 0.0

    def test_no_penalty_when_low_deviation(self) -> None:
        """偏离均值低于阈值时无惩罚。"""
        wait_times = {"tenant_a": 10.0, "tenant_b": 11.0}
        # deviation = |10 - 10.5| / 10.5 ≈ 0.048 < 0.3
        result = compute_fairness_penalty("tenant_a", wait_times)
        assert result == 0.0

    def test_penalty_when_high_deviation(self) -> None:
        """偏离均值超过阈值时施加惩罚。"""
        wait_times = {"tenant_a": 20.0, "tenant_b": 5.0}
        # mean = 12.5, deviation = |20 - 12.5| / 12.5 = 0.6 > 0.3
        # penalty = -2.0 * 0.6 = -1.2
        result = compute_fairness_penalty("tenant_a", wait_times)
        expected = -FAIRNESS_PENALTY_FACTOR * 0.6
        assert result == pytest.approx(expected, abs=1e-6)

    def test_penalty_for_low_wait_tenant(self) -> None:
        """等待时间远低于均值的租户也受惩罚。"""
        wait_times = {"tenant_a": 2.0, "tenant_b": 20.0}
        # mean = 11.0, deviation = |2 - 11| / 11 ≈ 0.818 > 0.3
        result = compute_fairness_penalty("tenant_a", wait_times)
        assert result < 0.0

    def test_no_penalty_when_zero_mean(self) -> None:
        """均值为零时无惩罚（避免除零）。"""
        wait_times = {"tenant_a": 0.0, "tenant_b": 0.0}
        result = compute_fairness_penalty("tenant_a", wait_times)
        assert result == 0.0

    def test_unknown_tenant_uses_zero_wait(self) -> None:
        """未知租户使用等待时间0。"""
        wait_times = {"tenant_a": 10.0, "tenant_b": 10.0}
        # tenant_c 不在字典中，wait=0
        # mean=10, deviation=|0-10|/10=1.0 > 0.3
        result = compute_fairness_penalty("tenant_c", wait_times)
        expected = -FAIRNESS_PENALTY_FACTOR * 1.0
        assert result == pytest.approx(expected, abs=1e-6)

    def test_none_tenant_id_returns_zero(self) -> None:
        """None 租户 ID 无租户上下文，返回0（不施加公平性惩罚）。"""
        wait_times = {"t1": 10.0, "t2": 20.0}
        result = compute_fairness_penalty(None, wait_times)
        assert result == 0.0


class TestFairnessRewardIntegration:
    """Issue #587: 公平性惩罚在奖励函数中的集成测试。"""

    def test_fairness_penalty_added_to_reward(self) -> None:
        """公平性惩罚应累加到执行奖励上。"""
        rng = np.random.default_rng(42)
        task = _make_task()
        reward_without = compute_execution_reward(
            task, ACTION_CLASSICAL, rng, 0.99, 1.0, fairness_penalty=0.0
        )
        reward_with = compute_execution_reward(
            task, ACTION_CLASSICAL, rng, 0.99, 1.0, fairness_penalty=-1.5
        )
        assert reward_with == pytest.approx(reward_without - 1.5, abs=1e-6)

    def test_fairness_penalty_zero_by_default(self) -> None:
        """默认公平性惩罚为0，不影响奖励。"""
        rng = np.random.default_rng(42)
        task = _make_task()
        reward_default = compute_execution_reward(task, ACTION_CLASSICAL, rng, 0.99, 1.0)
        reward_explicit = compute_execution_reward(
            task, ACTION_CLASSICAL, rng, 0.99, 1.0, fairness_penalty=0.0
        )
        assert reward_default == pytest.approx(reward_explicit, abs=1e-6)


class TestFairnessObservation:
    """Issue #588: 公平性观测测试。"""

    def test_default_obs_dim_unchanged(self) -> None:
        """默认 OBS_DIM 仍为 16（向后兼容）。"""
        assert OBS_DIM == 16

    def test_fairness_obs_dim(self) -> None:
        """启用公平性观测时维度为 17。"""
        assert OBS_DIM_WITH_FAIRNESS == 17

    def test_env_without_fairness_obs(self) -> None:
        """默认环境观测空间为 16 维。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv()
        assert env.observation_space.shape == (16,)

    def test_env_with_fairness_obs(self) -> None:
        """启用公平性观测后观测空间为 17 维。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(include_fairness_obs=True)
        assert env.observation_space.shape == (17,)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (17,)


class TestObservationDimAblation:
    """Issue #585: 观测维度消融实验测试。"""

    def test_observation_dim_truncation(self) -> None:
        """observation_dim=8 时观测空间为 8 维。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(observation_dim=8)
        assert env.observation_space.shape == (8,)
        obs, _ = env.reset(seed=42)
        assert obs.shape == (8,)

    def test_observation_dim_none_defaults_to_full(self) -> None:
        """observation_dim=None 时使用默认 16 维。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(observation_dim=None)
        assert env.observation_space.shape == (16,)

    def test_observation_dim_larger_than_default_no_truncation(self) -> None:
        """observation_dim 大于 OBS_DIM 时不截断。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(observation_dim=20)
        assert env.observation_space.shape == (16,)
