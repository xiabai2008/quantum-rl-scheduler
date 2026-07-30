"""Issue #746 回归测试：env.step 热路径性能优化。

验证内容：
1. 行为保持：固定 seed 下 env.step() 产出（obs/reward/terminated/truncated）逐位一致
2. 调用次数：单步 _get_observation() 调用次数由 3 降为 2（消除 crosstalk_risk 同态重复计算）
3. 静态缓存：episode 内 coupling_density / avg_connectivity 不变，缓存正确失效重建
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_types import (
    OBS_AVG_CONNECTIVITY,
    OBS_COUPLING_DENSITY,
)


class TestIssue746BehaviorPreservation:
    """验证性能优化不改变 env.step() 的行为输出。"""

    def test_step_output_deterministic(self):
        """固定 seed 下 env.step() 产出应逐位一致（行为保持）。"""
        env = QuantumSchedulingEnv()
        env.reset(seed=42)

        # Step 1: ACTION_CLASSICAL (action=0) — 走兼容分配分支
        obs1, reward1, term1, trunc1, _ = env.step(0)
        assert reward1 == pytest.approx(7.800000190734863)
        assert term1 is False
        assert trunc1 is False
        expected_obs1 = np.array(
            [
                0.5297490358352661,
                0.1666666716337204,
                0.01600000075995922,
                0.9165992140769958,
                0.3988058269023895,
                0.0,
                0.7467621564865112,
                0.5,
                0.0,
                0.0,
                0.8948423862457275,
                0.8595514893531799,
                0.30000001192092896,
                0.75,
                0.0,
                0.10000000149011612,
                0.0,
            ],
            dtype=np.float32,
        )
        assert_allclose(obs1, expected_obs1, rtol=1e-6)

        # Step 2: ACTION_QUANTUM (action=1)
        _obs2, reward2, term2, trunc2, _ = env.step(1)
        assert reward2 == pytest.approx(23.3950252532959)
        assert term2 is False
        assert trunc2 is False

        # Step 3: ACTION_HYBRID (action=2)
        _obs3, reward3, term3, trunc3, _ = env.step(2)
        assert reward3 == pytest.approx(7.190027713775635)
        assert term3 is False
        assert trunc3 is False

    def test_repeated_run_identical(self):
        """两次独立创建的 env（同 seed）应产出完全一致的结果。"""
        results_a = []
        env_a = QuantumSchedulingEnv()
        env_a.reset(seed=42)
        for action in (0, 1, 2):
            obs, reward, term, trunc, _ = env_a.step(action)
            results_a.append((obs.copy(), reward, term, trunc))

        results_b = []
        env_b = QuantumSchedulingEnv()
        env_b.reset(seed=42)
        for action in (0, 1, 2):
            obs, reward, term, trunc, _ = env_b.step(action)
            results_b.append((obs.copy(), reward, term, trunc))

        for (obs_a, r_a, t_a, tr_a), (obs_b, r_b, t_b, tr_b) in zip(
            results_a, results_b, strict=True
        ):
            assert_allclose(obs_a, obs_b, rtol=1e-7)
            assert r_a == pytest.approx(r_b)
            assert t_a == t_b
            assert tr_a == tr_b


class TestIssue746CallCountReduction:
    """验证 _get_observation() 调用次数由 3 降为 2。"""

    def test_get_observation_called_twice_in_compatible_branch(self):
        """兼容分配分支单步 _get_observation() 应只调用 2 次。

        改动前：558 行 obs 构建 + 748 行 reward 冗余重算 + 783 行外部 obs = 3 次
        改动后：558 行 obs 构建 + 783 行外部 obs = 2 次（748 行冗余调用已消除）
        """
        env = QuantumSchedulingEnv()
        env.reset(seed=42)

        call_count = 0
        original = env._get_observation

        def counting_wrapper():
            nonlocal call_count
            call_count += 1
            return original()

        env._get_observation = counting_wrapper

        # action=0 走兼容分配分支
        env.step(0)

        assert call_count == 2, (
            f"Expected 2 _get_observation() calls, got {call_count}. "
            "The redundant call in _compute_execution_reward should be eliminated."
        )

    def test_crosstalk_risk_passed_to_compute_reward(self):
        """_compute_execution_reward 应接受外部传入的 crosstalk_risk。"""
        env = QuantumSchedulingEnv()
        env.reset(seed=42)

        # 先构建一次 obs 获取 crosstalk_risk
        obs = env._get_observation()
        from src.scheduler.env_types import OBS_CROSSTALK_RISK

        crosstalk_risk = float(obs[OBS_CROSSTALK_RISK])

        # 传入 crosstalk_risk 时不应额外调用 _get_observation()
        call_count = 0
        original = env._get_observation

        def counting_wrapper():
            nonlocal call_count
            call_count += 1
            return original()

        env._get_observation = counting_wrapper

        from src.scheduler.env_types import ACTION_CLASSICAL

        # 传入 crosstalk_risk，内部不应再调用 _get_observation()
        env._compute_execution_reward(
            env._current_task, ACTION_CLASSICAL, env.np_random, crosstalk_risk=crosstalk_risk
        )
        assert call_count == 0, (
            f"Expected 0 _get_observation() calls when crosstalk_risk is passed, got {call_count}."
        )


class TestIssue746StaticCache:
    """验证 episode 内静态观测分量缓存。"""

    def test_static_components_constant_within_episode(self):
        """coupling_density / avg_connectivity 在 episode 内应保持不变。"""
        env = QuantumSchedulingEnv()
        env.reset(seed=42)

        obs1, _, _, _, _ = env.step(0)
        obs2, _, _, _, _ = env.step(1)
        obs3, _, _, _, _ = env.step(2)

        # 静态分量在 episode 内不变
        assert obs1[OBS_COUPLING_DENSITY] == obs2[OBS_COUPLING_DENSITY]
        assert obs2[OBS_COUPLING_DENSITY] == obs3[OBS_COUPLING_DENSITY]
        assert obs1[OBS_AVG_CONNECTIVITY] == obs2[OBS_AVG_CONNECTIVITY]
        assert obs2[OBS_AVG_CONNECTIVITY] == obs3[OBS_AVG_CONNECTIVITY]

    def test_cache_populated_after_step(self):
        """步进后静态缓存应被填充。"""
        env = QuantumSchedulingEnv()
        env.reset(seed=42)
        env.step(0)

        assert env._obs_static_cache is not None
        assert "qubits_arr" in env._obs_static_cache
        assert "coupling_density" in env._obs_static_cache
        assert "avg_connectivity" in env._obs_static_cache

    def test_cache_rebuilt_after_reset(self):
        """reset() 后缓存应被重建（置脏后由 reset 末尾的 obs 构建重建）。"""
        env = QuantumSchedulingEnv()
        env.reset(seed=42)
        env.step(0)
        old_cache = env._obs_static_cache
        assert old_cache is not None

        # reset 会先置 _obs_static_cache = None，再由末尾
        # _get_external_observation() → get_observation() 重建
        env.reset(seed=42)
        new_cache = env._obs_static_cache
        assert new_cache is not None
        # 同 seed 同机器配置，缓存值应一致
        assert new_cache["coupling_density"] == old_cache["coupling_density"]
        assert new_cache["avg_connectivity"] == old_cache["avg_connectivity"]
        np.testing.assert_array_equal(new_cache["qubits_arr"], old_cache["qubits_arr"])

    def test_qubits_arr_is_static(self):
        """qubits_arr 缓存值应等于机器 total_qubits 数组（episode 内不变）。"""
        env = QuantumSchedulingEnv()
        env.reset(seed=42)
        env.step(0)

        expected = np.array([m.total_qubits for m in env._machines], dtype=np.float64)
        np.testing.assert_array_equal(env._obs_static_cache["qubits_arr"], expected)
