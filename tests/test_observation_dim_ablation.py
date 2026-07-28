"""
D2 观测维度消融实验配置测试（Issue #585）
Unit Tests for Observation Dimension Ablation

测试覆盖：
- TestObservationDimParam      : observation_dim 参数验证与观测空间截断
- TestD2AblationConfig         : D2 配置正确设置 observation_dim=8
- TestD2AblationRun            : D2 消融运行使用 8 维观测
- TestObservationDimValidation : 参数边界校验
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.ablation import AblationRunner
from src.scheduler.env import OBS_DIM, QuantumSchedulingEnv


# ============================================================
# TestObservationDimParam
# ============================================================
class TestObservationDimParam(unittest.TestCase):
    """测试 observation_dim 参数对观测空间的影响。"""

    def test_default_uses_full_dim(self):
        """不指定 observation_dim 时使用完整 OBS_DIM 维。"""
        env = QuantumSchedulingEnv(max_qubits=20, seed=42)
        self.assertEqual(env.observation_space.shape, (OBS_DIM,))
        env.close()

    def test_dim_8_truncates_observation_space(self):
        """observation_dim=8 时观测空间为 (8,)。"""
        env = QuantumSchedulingEnv(max_qubits=20, seed=42, observation_dim=8)
        self.assertEqual(env.observation_space.shape, (8,))
        env.close()

    def test_dim_8_reset_returns_8dim_obs(self):
        """observation_dim=8 时 reset 返回 8 维观测。"""
        env = QuantumSchedulingEnv(max_qubits=20, seed=42, observation_dim=8)
        obs, _info = env.reset()
        self.assertEqual(obs.shape, (8,))
        env.close()

    def test_dim_8_step_returns_8dim_obs(self):
        """observation_dim=8 时 step 返回 8 维观测。"""
        env = QuantumSchedulingEnv(max_qubits=20, seed=42, observation_dim=8)
        env.reset()
        obs, _reward, _terminated, _truncated, _info = env.step(0)
        self.assertEqual(obs.shape, (8,))
        env.close()

    def test_dim_12_truncates_correctly(self):
        """observation_dim=12 时截断到 12 维。"""
        env = QuantumSchedulingEnv(max_qubits=20, seed=42, observation_dim=12)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (12,))
        env.close()

    def test_internal_observation_remains_full(self):
        """内部 _get_observation 始终返回完整维度。"""
        env = QuantumSchedulingEnv(max_qubits=20, seed=42, observation_dim=8)
        env.reset()
        full_obs = env._get_observation()
        self.assertEqual(full_obs.shape, (OBS_DIM,))
        env.close()

    def test_truncated_obs_matches_prefix(self):
        """截断观测等于完整观测的前 N 维。"""
        env = QuantumSchedulingEnv(max_qubits=20, seed=42, observation_dim=8)
        env.reset()
        full_obs = env._get_observation()
        external_obs = env._get_external_observation()
        np.testing.assert_array_equal(external_obs, full_obs[:8])
        env.close()


# ============================================================
# TestD2AblationConfig
# ============================================================
class TestD2AblationConfig(unittest.TestCase):
    """测试 D2 消融配置正确设置 observation_dim。"""

    def setUp(self):
        self.runner = AblationRunner()
        self.configs = self.runner.define_configs()

    def test_d2_has_observation_dim_in_env_params(self):
        """D2 配置的 env_params 应包含 observation_dim=8。"""
        d2 = next(c for c in self.configs if c.name.startswith("D2"))
        self.assertIn("observation_dim", d2.env_params)
        self.assertEqual(d2.env_params["observation_dim"], 8)

    def test_other_configs_do_not_have_observation_dim(self):
        """非 D2 配置不应设置 observation_dim。"""
        for cfg in self.configs:
            if cfg.name.startswith("D2"):
                continue
            self.assertNotIn(
                "observation_dim",
                cfg.env_params,
                f"{cfg.name} 不应设置 observation_dim",
            )

    def test_d2_disables_state_14dim(self):
        """D2 应关闭 state_14dim 组件。"""
        d2 = next(c for c in self.configs if c.name.startswith("D2"))
        self.assertFalse(d2.components["state_14dim"])


# ============================================================
# TestD2AblationRun
# ============================================================
class TestD2AblationRun(unittest.TestCase):
    """测试 D2 消融实验运行时使用 8 维观测。"""

    def test_d2_run_completes(self):
        """D2 配置运行应成功完成。"""
        runner = AblationRunner()
        configs = runner.define_configs()
        d2 = next(c for c in configs if c.name.startswith("D2"))
        result = runner.run_single(d2, n_episodes=2)
        self.assertEqual(result.n_episodes, 2)
        self.assertIsNotNone(result.mean_reward)

    def test_d2_env_has_8dim_observation(self):
        """D2 构建的环境观测空间应为 8 维。"""
        runner = AblationRunner()
        configs = runner.define_configs()
        d2 = next(c for c in configs if c.name.startswith("D2"))

        env_params = {"max_steps": 50, "max_qubits": 20, "seed": 42}
        env_params.update(d2.env_params)
        env = runner._build_env(d2, env_params)
        self.assertEqual(env.observation_space.shape, (8,))
        env.close()

    def test_d2_run_all_completes(self):
        """run_all 包含 D2 时应全部成功。"""
        runner = AblationRunner()
        results = runner.run_all(n_episodes=2)
        self.assertEqual(len(results), 5)
        # 确保所有结果有有效奖励值
        for r in results:
            self.assertFalse(math.isnan(r.mean_reward))


# ============================================================
# TestObservationDimValidation
# ============================================================
class TestObservationDimValidation(unittest.TestCase):
    """测试 observation_dim 参数校验。"""

    def test_zero_raises_error(self):
        """observation_dim=0 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            QuantumSchedulingEnv(max_qubits=20, observation_dim=0)

    def test_negative_raises_error(self):
        """observation_dim=-1 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            QuantumSchedulingEnv(max_qubits=20, observation_dim=-1)

    def test_exceeds_obs_dim_raises_error(self):
        """observation_dim 超过 OBS_DIM 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            QuantumSchedulingEnv(max_qubits=20, observation_dim=OBS_DIM + 1)

    def test_observation_dim_equal_to_obs_dim(self):
        """observation_dim == OBS_DIM 应正常工作。"""
        env = QuantumSchedulingEnv(max_qubits=20, observation_dim=OBS_DIM)
        self.assertEqual(env.observation_space.shape, (OBS_DIM,))
        env.close()

    def test_observation_dim_one(self):
        """observation_dim=1 应正常工作。"""
        env = QuantumSchedulingEnv(max_qubits=20, observation_dim=1)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (1,))
        env.close()


if __name__ == "__main__":
    unittest.main()
