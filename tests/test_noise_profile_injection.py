#!/usr/bin/env python
"""噪声参数注入仿真环境测试（Issue #591）。

测试 noise_profile 参数注入到 QuantumSchedulingEnv，
使奖励函数感知真机噪声水平（readout_error / gate_error / t1）。

测试链路：
    NoiseModelExtractor.extract_all() → env.inject_noise_profile()
    → compute_execution_reward(noise_profile=...) → 奖励差异
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_real_machine import NoiseModelExtractor
from src.scheduler.env_reward import _compute_noise_discount, compute_execution_reward
from src.scheduler.env_types import (
    ACTION_CLASSICAL,
    ACTION_HYBRID,
    ACTION_QUANTUM,
    NOISE_GATE_PENALTY_WEIGHT,
    NOISE_KEY_GATE_ERROR,
    NOISE_KEY_READOUT_ERROR,
    NOISE_PENALTY_FLOOR,
    NOISE_READOUT_PENALTY_WEIGHT,
    REWARD_SUCCESS_BONUS,
    Task,
)


def _make_rng(seed: int = 42) -> np.random.Generator:
    return np.random.default_rng(seed)


# =============================================================================
# _compute_noise_discount 单元测试
# =============================================================================


class TestComputeNoiseDiscount:
    """噪声折扣因子计算测试。"""

    def test_none_profile_returns_unity(self):
        """None 噪声配置返回 1.0（无折扣）。"""
        assert _compute_noise_discount(None) == 1.0

    def test_empty_profile_returns_unity(self):
        """空字典返回 1.0。"""
        assert _compute_noise_discount({}) == 1.0

    def test_zero_noise_returns_unity(self):
        """零噪声返回 1.0。"""
        profile = {"readout_error": 0.0, "gate_error": 0.0}
        assert _compute_noise_discount(profile) == 1.0

    def test_readout_error_discount(self):
        """读出误差产生等比例折扣。"""
        profile = {"readout_error": 0.1, "gate_error": 0.0}
        expected = 1.0 - 0.1 * NOISE_READOUT_PENALTY_WEIGHT
        assert _compute_noise_discount(profile) == pytest.approx(expected)

    def test_gate_error_discount(self):
        """门误差产生 2 倍折扣。"""
        profile = {"readout_error": 0.0, "gate_error": 0.05}
        expected = 1.0 - 0.05 * NOISE_GATE_PENALTY_WEIGHT
        assert _compute_noise_discount(profile) == pytest.approx(expected)

    def test_combined_discount(self):
        """读出+门误差组合折扣。"""
        profile = {"readout_error": 0.1, "gate_error": 0.05}
        expected = 1.0 - (
            0.1 * NOISE_READOUT_PENALTY_WEIGHT + 0.05 * NOISE_GATE_PENALTY_WEIGHT
        )
        assert _compute_noise_discount(profile) == pytest.approx(expected)

    def test_half_weight_for_hybrid(self):
        """混合执行半权重折扣。"""
        profile = {"readout_error": 0.2, "gate_error": 0.1}
        full = _compute_noise_discount(profile, full_weight=True)
        half = _compute_noise_discount(profile, full_weight=False)
        # 半权重折扣更少（折扣因子更大）
        assert half > full
        # 验证数值
        expected_half = 1.0 - 0.5 * (
            0.2 * NOISE_READOUT_PENALTY_WEIGHT + 0.1 * NOISE_GATE_PENALTY_WEIGHT
        )
        assert half == pytest.approx(expected_half)

    def test_extreme_noise_floored(self):
        """极端噪声不低于 NOISE_PENALTY_FLOOR。"""
        profile = {"readout_error": 1.0, "gate_error": 1.0}
        discount = _compute_noise_discount(profile)
        assert discount >= NOISE_PENALTY_FLOOR
        assert discount == NOISE_PENALTY_FLOOR

    def test_missing_keys_default_zero(self):
        """缺少键时默认 0（不产生折扣）。"""
        profile = {"t1": 50.0}
        assert _compute_noise_discount(profile) == 1.0


# =============================================================================
# compute_execution_reward 噪声感知测试
# =============================================================================


class TestRewardWithNoise:
    """奖励函数噪声感知测试。"""

    def test_classical_unaffected_by_noise(self):
        """经典执行不受噪声影响。"""
        task = Task(task_id="t", task_type="classical", qubit_count=1, priority=1)
        rng = _make_rng()
        reward_clean = compute_execution_reward(
            task, ACTION_CLASSICAL, rng, quantum_fidelity=0.95,
            quantum_available_ratio=0.8, noise_profile=None,
        )
        reward_noisy = compute_execution_reward(
            task, ACTION_CLASSICAL, rng, quantum_fidelity=0.95,
            quantum_available_ratio=0.8,
            noise_profile={"readout_error": 0.5, "gate_error": 0.3},
        )
        assert reward_clean == reward_noisy

    def test_quantum_reward_decreased_by_noise(self):
        """量子执行奖励受噪声折扣。"""
        task = Task(task_id="t", task_type="quantum", qubit_count=2, priority=3)
        rng = _make_rng()
        noise = {"readout_error": 0.2, "gate_error": 0.1}
        reward_clean = compute_execution_reward(
            task, ACTION_QUANTUM, rng, quantum_fidelity=0.95,
            quantum_available_ratio=0.8, noise_profile=None,
        )
        reward_noisy = compute_execution_reward(
            task, ACTION_QUANTUM, rng, quantum_fidelity=0.95,
            quantum_available_ratio=0.8, noise_profile=noise,
        )
        assert reward_noisy < reward_clean

    def test_hybrid_reward_decreased_by_noise(self):
        """混合执行奖励受噪声折扣。"""
        task = Task(task_id="t", task_type="hybrid", qubit_count=2, priority=3)
        rng = _make_rng()
        noise = {"readout_error": 0.3, "gate_error": 0.2}
        reward_clean = compute_execution_reward(
            task, ACTION_HYBRID, rng, quantum_fidelity=0.95,
            quantum_available_ratio=0.8, noise_profile=None,
        )
        reward_noisy = compute_execution_reward(
            task, ACTION_HYBRID, rng, quantum_fidelity=0.95,
            quantum_available_ratio=0.8, noise_profile=noise,
        )
        assert reward_noisy < reward_clean

    def test_quantum_noise_stronger_than_hybrid(self):
        """量子执行的噪声折扣强于混合执行。"""
        noise = {"readout_error": 0.2, "gate_error": 0.1}

        # 量子的折扣因子 < 混合的折扣因子
        discount_q = _compute_noise_discount(noise, full_weight=True)
        discount_h = _compute_noise_discount(noise, full_weight=False)
        assert discount_q < discount_h

    def test_higher_noise_lower_reward(self):
        """噪声越大，量子奖励越低。"""
        task = Task(task_id="t", task_type="quantum", qubit_count=2, priority=3)

        rng1 = _make_rng()
        rng2 = _make_rng()
        low_noise = {"readout_error": 0.05, "gate_error": 0.02}
        high_noise = {"readout_error": 0.3, "gate_error": 0.2}

        reward_low = compute_execution_reward(
            task, ACTION_QUANTUM, rng1, quantum_fidelity=0.95,
            quantum_available_ratio=0.8, noise_profile=low_noise,
        )
        reward_high = compute_execution_reward(
            task, ACTION_QUANTUM, rng2, quantum_fidelity=0.95,
            quantum_available_ratio=0.8, noise_profile=high_noise,
        )
        assert reward_high < reward_low

    def test_zero_noise_same_as_none(self):
        """零噪声与无噪声注入效果相同。"""
        task = Task(task_id="t", task_type="quantum", qubit_count=2, priority=3)
        rng1 = _make_rng()
        rng2 = _make_rng()
        reward_none = compute_execution_reward(
            task, ACTION_QUANTUM, rng1, quantum_fidelity=0.95,
            quantum_available_ratio=0.8, noise_profile=None,
        )
        reward_zero = compute_execution_reward(
            task, ACTION_QUANTUM, rng2, quantum_fidelity=0.95,
            quantum_available_ratio=0.8,
            noise_profile={"readout_error": 0.0, "gate_error": 0.0},
        )
        assert reward_none == pytest.approx(reward_zero)


# =============================================================================
# env.inject_noise_profile 测试
# =============================================================================


class TestInjectNoiseProfile:
    """env.inject_noise_profile 方法测试。"""

    def test_default_no_noise_profile(self):
        """默认构造时 noise_profile 为 None。"""
        env = QuantumSchedulingEnv()
        assert env.noise_profile is None

    def test_init_with_noise_profile(self):
        """构造时传入 noise_profile。"""
        profile = {"readout_error": 0.1, "gate_error": 0.05}
        env = QuantumSchedulingEnv(noise_profile=profile)
        assert env.noise_profile is not None
        assert env.noise_profile["readout_error"] == 0.1
        assert env.noise_profile["gate_error"] == 0.05

    def test_inject_from_extractor_results(self):
        """从 NoiseModelExtractor.extract_all() 结果注入。"""
        extractor = NoiseModelExtractor()
        results = extractor.extract_all(
            measurement_results={"0": 0.4, "1": 0.6},
            rb_results=[
                {"m": 1, "fidelity": 0.99},
                {"m": 5, "fidelity": 0.95},
                {"m": 10, "fidelity": 0.90},
            ],
            delay_results=[
                {"t": 10, "p1": 0.9},
                {"t": 50, "p1": 0.6},
                {"t": 100, "p1": 0.3},
            ],
        )
        env = QuantumSchedulingEnv()
        env.inject_noise_profile(results)
        assert env.noise_profile is not None
        assert NOISE_KEY_READOUT_ERROR in env.noise_profile
        assert NOISE_KEY_GATE_ERROR in env.noise_profile
        assert "t1" in env.noise_profile
        assert env.noise_profile[NOISE_KEY_READOUT_ERROR] > 0

    def test_inject_partial_data(self):
        """仅部分数据时只注入对应参数。"""
        env = QuantumSchedulingEnv()
        env.inject_noise_profile({"readout_error": 0.15})
        assert env.noise_profile is not None
        assert env.noise_profile["readout_error"] == 0.15
        assert "gate_error" not in env.noise_profile

    def test_inject_empty_dict_clears(self):
        """空字典注入后 noise_profile 为 None。"""
        env = QuantumSchedulingEnv(noise_profile={"readout_error": 0.1})
        env.inject_noise_profile({})
        assert env.noise_profile is None

    def test_inject_decoherence_subdict(self):
        """decoherence 子字典正确提取 t1。"""
        env = QuantumSchedulingEnv()
        env.inject_noise_profile({
            "readout_error": 0.1,
            "decoherence": {"t1": 75.5, "amplitude": 0.98, "fit_quality": 0.99},
        })
        assert env.noise_profile["t1"] == 75.5

    def test_noise_profile_persists_across_reset(self):
        """reset() 不清除 noise_profile（跨 episode 持久化）。"""
        env = QuantumSchedulingEnv(noise_profile={"readout_error": 0.1})
        env.reset(seed=42)
        assert env.noise_profile is not None
        assert env.noise_profile["readout_error"] == 0.1
        env.reset(seed=99)
        assert env.noise_profile is not None

    def test_stats_include_noise_profile(self):
        """get_real_machine_stats 包含 noise_profile。"""
        env = QuantumSchedulingEnv()
        env.inject_noise_profile({"readout_error": 0.12, "gate_error": 0.08})
        stats = env.get_real_machine_stats()
        assert "noise_profile" in stats
        assert stats["noise_profile"] is not None
        assert stats["noise_profile"]["readout_error"] == 0.12


# =============================================================================
# 端到端集成测试
# =============================================================================


class TestNoiseInjectionIntegration:
    """端到端：噪声注入→step→奖励差异。"""

    def test_step_reward_lower_with_noise(self):
        """注入噪声后，量子 step 的奖励低于无噪声时。"""
        env_clean = QuantumSchedulingEnv(noise_aware_reward_shaping=False)
        env_noisy = QuantumSchedulingEnv(
            noise_aware_reward_shaping=False,
            noise_profile={"readout_error": 0.3, "gate_error": 0.2},
        )
        env_clean.reset(seed=42)
        env_noisy.reset(seed=42)

        # 确保两个环境有相同的初始状态和任务
        # 手动设置相同的当前任务
        from src.scheduler.env_types import ACTION_QUANTUM, Task

        task = Task(task_id="test", task_type="quantum", qubit_count=3, priority=3)
        env_clean._current_task = task
        env_noisy._current_task = task

        # 确保量子资源可用且保真度相同
        env_clean._quantum.fidelity = 0.95
        env_noisy._quantum.fidelity = 0.95
        env_clean._quantum.available_ratio = 0.8
        env_noisy._quantum.available_ratio = 0.8

        rng_clean = _make_rng(42)
        rng_noisy = _make_rng(42)

        reward_clean = env_clean._compute_execution_reward(task, ACTION_QUANTUM, rng_clean)
        reward_noisy = env_noisy._compute_execution_reward(task, ACTION_QUANTUM, rng_noisy)

        assert reward_noisy < reward_clean, (
            f"噪声注入后奖励应更低: clean={reward_clean}, noisy={reward_noisy}"
        )

    def test_classical_step_unaffected_by_noise(self):
        """经典 step 奖励不受噪声影响。"""
        env_clean = QuantumSchedulingEnv(noise_aware_reward_shaping=False)
        env_noisy = QuantumSchedulingEnv(
            noise_aware_reward_shaping=False,
            noise_profile={"readout_error": 0.3, "gate_error": 0.2},
        )
        env_clean.reset(seed=42)
        env_noisy.reset(seed=42)

        from src.scheduler.env_types import ACTION_CLASSICAL, Task

        task = Task(task_id="test", task_type="classical", qubit_count=1, priority=1)
        env_clean._current_task = task
        env_noisy._current_task = task

        rng_clean = _make_rng(42)
        rng_noisy = _make_rng(42)

        reward_clean = env_clean._compute_execution_reward(task, ACTION_CLASSICAL, rng_clean)
        reward_noisy = env_noisy._compute_execution_reward(task, ACTION_CLASSICAL, rng_noisy)

        assert reward_clean == reward_noisy

    def test_full_flow_extractor_to_env(self):
        """完整流程：NoiseModelExtractor → env.inject → step reward。"""
        # 1. 用 NoiseModelExtractor 提取噪声参数
        extractor = NoiseModelExtractor()
        noise_results = extractor.extract_all(
            measurement_results={"0": 0.35, "1": 0.65},  # readout_error = 0.3
            rb_results=[
                {"m": 1, "fidelity": 0.99},
                {"m": 5, "fidelity": 0.92},
                {"m": 10, "fidelity": 0.85},
                {"m": 20, "fidelity": 0.70},
            ],
            delay_results=[
                {"t": 10, "p1": 0.9},
                {"t": 50, "p1": 0.6},
                {"t": 100, "p1": 0.3},
            ],
        )

        # 2. 注入到环境
        env = QuantumSchedulingEnv(noise_aware_reward_shaping=False)
        env.inject_noise_profile(noise_results)
        assert env.noise_profile is not None
        assert env.noise_profile[NOISE_KEY_READOUT_ERROR] > 0

        # 3. 验证 step 奖励被噪声折扣
        env.reset(seed=42)
        from src.scheduler.env_types import ACTION_QUANTUM, Task

        task = Task(task_id="test", task_type="quantum", qubit_count=2, priority=3)
        env._current_task = task
        env._quantum.fidelity = 0.95
        rng = _make_rng(42)

        reward_with_noise = env._compute_execution_reward(task, ACTION_QUANTUM, rng)

        # 4. 移除噪声后奖励应更高
        env.noise_profile = None
        rng2 = _make_rng(42)
        reward_without_noise = env._compute_execution_reward(task, ACTION_QUANTUM, rng2)

        assert reward_with_noise < reward_without_noise

    def test_quantifiable_difference(self):
        """有/无噪声注入的奖励差异可量化。"""
        task = Task(task_id="t", task_type="quantum", qubit_count=2, priority=3)
        noise = {"readout_error": 0.2, "gate_error": 0.1}

        rng1 = _make_rng(42)
        rng2 = _make_rng(42)

        reward_clean = compute_execution_reward(
            task, ACTION_QUANTUM, rng1, quantum_fidelity=0.95,
            quantum_available_ratio=0.8, noise_profile=None,
        )
        reward_noisy = compute_execution_reward(
            task, ACTION_QUANTUM, rng2, quantum_fidelity=0.95,
            quantum_available_ratio=0.8, noise_profile=noise,
        )

        # 差异应大于 0（可量化）
        diff = reward_clean - reward_noisy
        assert diff > 0.0

        # 验证差异约等于 reward_clean_success_bonus * (1 - discount)
        # reward_noisy = (reward_clean - bonus) * discount + bonus
        # diff = (reward_clean - bonus) * (1 - discount)
        discount = _compute_noise_discount(noise, full_weight=True)
        expected_diff = (reward_clean - REWARD_SUCCESS_BONUS) * (1.0 - discount)
        assert diff == pytest.approx(expected_diff, rel=0.01)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
