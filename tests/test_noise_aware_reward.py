#!/usr/bin/env python
"""噪声感知奖励整形测试（Issue #577）。

测试 _compute_noise_gradient_feedback 和 apply_noise_penalty_to_reward
形成的"真机保真度→后续步奖励惩罚"闭环。
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_real_machine import (
    NOISE_PENALTY_DECAY,
    NOISE_PENALTY_DURATION,
    NOISE_PENALTY_FIDELITY_THRESHOLD,
    NOISE_PENALTY_SCALE,
    _compute_noise_gradient_feedback,
    apply_noise_penalty_to_reward,
)


def _make_env(noise_aware: bool = True) -> QuantumSchedulingEnv:
    """构造测试用环境实例。"""
    env = QuantumSchedulingEnv(noise_aware_reward_shaping=noise_aware)
    return env


# =============================================================================
# _compute_noise_gradient_feedback 测试
# =============================================================================


class TestComputeNoiseGradientFeedback:
    """噪声梯度反馈触发逻辑测试。"""

    def test_high_fidelity_no_penalty(self):
        """保真度 >= 0.9 时不触发惩罚。"""
        env = _make_env()
        penalty = _compute_noise_gradient_feedback(env, 0.95)
        assert penalty == 0.0
        assert env._noise_penalty_remaining == 0

    def test_invalid_fidelity_no_penalty(self):
        """保真度 = -1（未计算）时不触发惩罚。"""
        env = _make_env()
        penalty = _compute_noise_gradient_feedback(env, -1.0)
        assert penalty == 0.0
        assert env._noise_penalty_remaining == 0

    def test_low_fidelity_triggers_penalty(self):
        """保真度 < 0.9 时触发惩罚并设置后续步计数。"""
        env = _make_env()
        penalty = _compute_noise_gradient_feedback(env, 0.5)
        assert penalty > 0.0
        assert env._noise_penalty_remaining == NOISE_PENALTY_DURATION
        assert env._noise_penalty_initial == penalty

    def test_penalty_scales_with_fidelity_gap(self):
        """惩罚强度与保真度差距成正比。"""
        env1 = _make_env()
        penalty_low = _compute_noise_gradient_feedback(env1, 0.5)
        env2 = _make_env()
        penalty_high = _compute_noise_gradient_feedback(env2, 0.85)
        # 0.5 距离 0.9 的差距更大，惩罚应更大
        assert penalty_low > penalty_high

    def test_penalty_value_correct(self):
        """惩罚值 = (threshold - fidelity) * scale。"""
        env = _make_env()
        fidelity = 0.7
        expected = (NOISE_PENALTY_FIDELITY_THRESHOLD - fidelity) * NOISE_PENALTY_SCALE
        penalty = _compute_noise_gradient_feedback(env, fidelity)
        assert pytest.approx(penalty) == expected

    def test_disabled_when_noise_aware_false(self):
        """noise_aware_reward_shaping=False 时不触发惩罚。"""
        env = _make_env(noise_aware=False)
        penalty = _compute_noise_gradient_feedback(env, 0.5)
        assert penalty == 0.0
        assert env._noise_penalty_remaining == 0

    def test_boundary_fidelity_no_penalty(self):
        """保真度恰好 = 0.9 时不触发（边界条件）。"""
        env = _make_env()
        penalty = _compute_noise_gradient_feedback(env, 0.9)
        assert penalty == 0.0


# =============================================================================
# apply_noise_penalty_to_reward 测试
# =============================================================================


class TestApplyNoisePenaltyToReward:
    """奖励衰减应用逻辑测试。"""

    def test_no_penalty_when_remaining_zero(self):
        """_noise_penalty_remaining=0 时奖励不变。"""
        env = _make_env()
        env._noise_penalty_remaining = 0
        reward = 10.0
        shaped = apply_noise_penalty_to_reward(env, reward)
        assert shaped == reward

    def test_reward_decayed_when_penalty_active(self):
        """_noise_penalty_remaining > 0 时奖励被衰减。"""
        env = _make_env()
        env._noise_penalty_remaining = 5
        reward = 10.0
        shaped = apply_noise_penalty_to_reward(env, reward)
        assert shaped < reward

    def test_counter_decrements(self):
        """每次应用后计数器递减。"""
        env = _make_env()
        env._noise_penalty_remaining = 5
        apply_noise_penalty_to_reward(env, 10.0)
        assert env._noise_penalty_remaining == 4
        apply_noise_penalty_to_reward(env, 10.0)
        assert env._noise_penalty_remaining == 3

    def test_penalty_expires_after_duration(self):
        """NOISE_PENALTY_DURATION 步后惩罚消失。"""
        env = _make_env()
        env._noise_penalty_remaining = NOISE_PENALTY_DURATION
        reward = 10.0
        for i in range(NOISE_PENALTY_DURATION):
            shaped = apply_noise_penalty_to_reward(env, reward)
            assert shaped < reward, f"Step {i+1}: reward should be decayed"
        # 第 6 步不应再衰减
        shaped = apply_noise_penalty_to_reward(env, reward)
        assert shaped == reward
        assert env._noise_penalty_remaining == 0

    def test_decay_factor_decreases_over_steps(self):
        """衰减强度随步数递减（越往后惩罚越小）。"""
        env = _make_env()
        env._noise_penalty_remaining = NOISE_PENALTY_DURATION
        reward = 10.0
        shaped_values = []
        for _ in range(NOISE_PENALTY_DURATION):
            shaped_values.append(apply_noise_penalty_to_reward(env, reward))
        # shaped_values 应递增（越往后衰减越小，奖励越大）
        for i in range(len(shaped_values) - 1):
            assert shaped_values[i] <= shaped_values[i + 1], (
                f"Step {i+1}: decay should decrease over time"
            )

    def test_penalty_total_accumulates(self):
        """累计衰减量正确累积。"""
        env = _make_env()
        env._noise_penalty_remaining = 3
        env._noise_penalty_total = 0.0
        reward = 10.0
        total_decay = 0.0
        for _ in range(3):
            shaped = apply_noise_penalty_to_reward(env, reward)
            total_decay += (reward - shaped)
        assert pytest.approx(env._noise_penalty_total) == total_decay
        assert env._noise_penalty_total > 0

    def test_minimum_reward_floor(self):
        env = _make_env()
        env._noise_penalty_remaining = NOISE_PENALTY_DURATION
        env._noise_penalty_initial = 0.0  # 显式设置，避免 getattr 默认值问题
        reward = 100.0
        shaped = apply_noise_penalty_to_reward(env, reward)
        # 衰减公式：decay = 1 - 0.1 * remaining / duration
        # 第一步：decay = 1 - 0.1 * 5/5 = 0.9, shaped = 90
        # floor = 0.1, 确保不低于 reward * 0.1
        assert shaped >= reward * 0.1
        assert shaped == reward * 0.9  # 第一步衰减为 0.9
class TestNoiseAwareRewardIntegration:
    """噪声感知奖励整形集成测试。"""

    def test_full_flow_low_fidelity_to_expiry(self):
        """完整流程：低保真度触发→5步衰减→惩罚消失。"""
        env = _make_env()
        env.reset(seed=42)

        # 模拟真机返回低保真度
        _compute_noise_gradient_feedback(env, 0.6)
        assert env._noise_penalty_remaining == NOISE_PENALTY_DURATION

        # 5 步内奖励被衰减
        for _ in range(NOISE_PENALTY_DURATION):
            assert env._noise_penalty_remaining > 0
            shaped = apply_noise_penalty_to_reward(env, 10.0)
            assert shaped < 10.0

        # 第 6 步无衰减
        assert env._noise_penalty_remaining == 0
        shaped = apply_noise_penalty_to_reward(env, 10.0)
        assert shaped == 10.0

    def test_disabled_shaping_no_effect(self):
        """noise_aware_reward_shaping=False 时整个机制无效。"""
        env = _make_env(noise_aware=False)
        env.reset(seed=42)

        # 即使低保真度也不触发
        _compute_noise_gradient_feedback(env, 0.3)
        assert env._noise_penalty_remaining == 0

        # 奖励不被衰减
        shaped = apply_noise_penalty_to_reward(env, 10.0)
        assert shaped == 10.0

    def test_reset_clears_penalty_state(self):
        """reset() 清除 _noise_penalty_remaining。"""
        env = _make_env()
        env.reset(seed=42)
        _compute_noise_gradient_feedback(env, 0.5)
        assert env._noise_penalty_remaining > 0

        env.reset(seed=99)
        assert env._noise_penalty_remaining == 0

    def test_stats_include_noise_penalty(self):
        """get_real_machine_stats 包含噪声惩罚信息。"""
        env = _make_env()
        env.reset(seed=42)
        _compute_noise_gradient_feedback(env, 0.5)
        apply_noise_penalty_to_reward(env, 10.0)

        stats = env.get_real_machine_stats()
        assert "noise_penalty_remaining" in stats
        assert "noise_penalty_total" in stats
        assert stats["noise_penalty_remaining"] == NOISE_PENALTY_DURATION - 1
        assert stats["noise_penalty_total"] > 0

    def test_multiple_triggers_reset_counter(self):
        """多次触发低保真度重置计数器为满。"""
        env = _make_env()
        env.reset(seed=42)

        # 第一次触发
        _compute_noise_gradient_feedback(env, 0.5)
        assert env._noise_penalty_remaining == NOISE_PENALTY_DURATION
        apply_noise_penalty_to_reward(env, 10.0)
        assert env._noise_penalty_remaining == NOISE_PENALTY_DURATION - 1

        # 第二次触发重置
        _compute_noise_gradient_feedback(env, 0.3)
        assert env._noise_penalty_remaining == NOISE_PENALTY_DURATION

    def test_execution_reward_affected_by_noise(self):
        """env._compute_execution_reward 在噪声惩罚激活时返回更低奖励。"""
        env = _make_env()
        env.reset(seed=42)
        rng = np.random.default_rng(42)

        from src.scheduler.env_types import ACTION_QUANTUM, Task

        task = Task(task_id="test", task_type="quantum", qubit_count=3, priority=3)

        # 无噪声惩罚时的奖励
        reward_normal = env._compute_execution_reward(task, ACTION_QUANTUM, rng)

        # 触发噪声惩罚
        env._noise_penalty_remaining = NOISE_PENALTY_DURATION
        reward_decayed = env._compute_execution_reward(task, ACTION_QUANTUM, rng)

        assert reward_decayed < reward_normal, (
            "噪声惩罚激活时执行奖励应更低"
        )

    def test_execution_reward_unaffected_when_disabled(self):
        """noise_aware_reward_shaping=False 时执行奖励不受影响。"""
        env = _make_env(noise_aware=False)
        env.reset(seed=42)
        rng = np.random.default_rng(42)

        from src.scheduler.env_types import ACTION_QUANTUM, Task

        task = Task(task_id="test", task_type="quantum", qubit_count=3, priority=3)

        # 即使手动设置 _noise_penalty_remaining，也不会应用
        env._noise_penalty_remaining = 5
        env._compute_execution_reward(task, ACTION_QUANTUM, rng)
        # _noise_penalty_remaining 不变（apply_noise_penalty_to_reward 未被调用）
        # 但因为 noise_aware_reward_shaping=False，_compute_execution_reward 不调用 apply
        assert env._noise_penalty_remaining == 5  # 未被消费


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
