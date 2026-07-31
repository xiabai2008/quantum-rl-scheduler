"""
Unit Tests for src/scheduler/ablation.ObsMaskWrapper 与 D2 消融配置
测试覆盖（Issue #875）：
- 屏蔽维度置零、未屏蔽维度保持
- reset/step 均应用掩码
- 越界 mask 维度安全忽略
- 观测空间保持 Box 且范围继承底层环境
- D2_OBSERVATION_CONFIGS 配置完整性（dim/mask 维度与 env_types 常量一致）
"""

from __future__ import annotations

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym

from src.scheduler.ablation import D2_OBSERVATION_CONFIGS, ObsMaskWrapper
from src.scheduler.env_types import (
    OBS_ARRIVAL_RATE_MA,
    OBS_CROSSTALK_RISK,
    OBS_DIM,
    OBS_SINGLE_GATE_FIDELITY,
    OBS_TWO_GATE_FIDELITY,
)


class _DummyEnv(gym.Env):
    """极简环境：固定观测 + 简单 step。"""

    def __init__(self, obs_dim: int = 16) -> None:
        super().__init__()
        self.observation_space = gym.spaces.Box(
            low=0.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = gym.spaces.Discrete(4)
        self._step = 0

    def reset(self, **kwargs):
        super().reset(seed=kwargs.get("seed"))
        return np.full(self.observation_space.shape, 1.0, dtype=np.float32), {}

    def step(self, action):
        self._step += 1
        obs = np.full(self.observation_space.shape, float(self._step + 1), dtype=np.float32)
        return obs, 1.0, False, False, {}


class TestObsMaskWrapper(unittest.TestCase):
    """掩码行为。"""

    def setUp(self) -> None:
        self.env = _DummyEnv(obs_dim=16)
        self.wrapper = ObsMaskWrapper(self.env, mask_dims=[2, 5, 14])

    def test_reset_masks_specified_dims(self) -> None:
        """reset 返回的观测中指定维度应为 0，其余保持。"""
        obs, _ = self.wrapper.reset()
        self.assertEqual(obs[2], 0.0)
        self.assertEqual(obs[5], 0.0)
        self.assertEqual(obs[14], 0.0)
        self.assertEqual(obs[0], 1.0)
        self.assertEqual(obs[15], 1.0)

    def test_step_masks_specified_dims(self) -> None:
        """step 返回的观测同样应被屏蔽。"""
        self.wrapper.reset()
        obs, reward, terminated, truncated, _ = self.wrapper.step(0)
        self.assertEqual(obs[2], 0.0)
        self.assertEqual(obs[5], 0.0)
        self.assertEqual(obs[0], 2.0)  # 未屏蔽维度保持 step 后的值
        self.assertEqual(reward, 1.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)

    def test_out_of_range_mask_dims_ignored(self) -> None:
        """越界 mask 索引应被安全忽略。"""
        wrapper = ObsMaskWrapper(_DummyEnv(obs_dim=16), mask_dims=[100, -1, 3])
        obs, _ = wrapper.reset()
        self.assertEqual(obs[3], 0.0)
        self.assertEqual(obs[0], 1.0)

    def test_observation_space_inherits_box_bounds(self) -> None:
        """观测空间应为 Box 且继承底层 env 的 low/high。"""
        self.assertIsInstance(self.wrapper.observation_space, gym.spaces.Box)
        np.testing.assert_array_equal(self.wrapper.observation_space.low, np.zeros(16))
        np.testing.assert_array_equal(self.wrapper.observation_space.high, np.full(16, 2.0))

    def test_no_mask_dims_passthrough(self) -> None:
        """空 mask 列表应原样透传观测。"""
        wrapper = ObsMaskWrapper(_DummyEnv(obs_dim=16), mask_dims=[])
        obs, _ = wrapper.reset()
        np.testing.assert_array_equal(obs, np.ones(16, dtype=np.float32))

    def test_mask_does_not_mutate_underlying_obs(self) -> None:
        """掩码应返回副本，不修改底层环境返回的数组。"""
        env = _DummyEnv(obs_dim=16)
        wrapper = ObsMaskWrapper(env, mask_dims=[0])
        raw_obs, _ = env.reset()
        masked_obs, _ = wrapper.reset()
        self.assertEqual(raw_obs[0], 1.0)  # 底层观测未被修改
        self.assertEqual(masked_obs[0], 0.0)


class TestD2ObservationConfigs(unittest.TestCase):
    """D2 观测消融配置完整性（Issue #875）。"""

    def test_all_configs_have_required_keys(self) -> None:
        """每个 D2 配置应包含 mask_dims/description/dim。"""
        for name, cfg in D2_OBSERVATION_CONFIGS.items():
            self.assertIn("mask_dims", cfg, name)
            self.assertIn("description", cfg, name)
            self.assertIn("dim", cfg, name)
            self.assertIsInstance(cfg["mask_dims"], list, name)

    def test_config_dims_match_mask_count(self) -> None:
        """dim 应为 OBS_DIM 减去有效屏蔽维度数。"""
        for name, cfg in D2_OBSERVATION_CONFIGS.items():
            effective = [d for d in cfg["mask_dims"] if 0 <= d < OBS_DIM]
            self.assertEqual(
                cfg["dim"],
                OBS_DIM - len(effective),
                f"{name}: dim={cfg['dim']} 与屏蔽维度数不一致",
            )

    def test_key_ablation_configs_reference_real_obs_indices(self) -> None:
        """关键消融配置应引用真实观测索引（串扰/到达率/物理维度）。"""
        self.assertEqual(D2_OBSERVATION_CONFIGS["no_crosstalk"]["mask_dims"], [OBS_CROSSTALK_RISK])
        self.assertEqual(
            D2_OBSERVATION_CONFIGS["no_arrival_rate"]["mask_dims"], [OBS_ARRIVAL_RATE_MA]
        )
        self.assertIn(OBS_SINGLE_GATE_FIDELITY, D2_OBSERVATION_CONFIGS["no_physical"]["mask_dims"])
        self.assertIn(OBS_TWO_GATE_FIDELITY, D2_OBSERVATION_CONFIGS["no_physical"]["mask_dims"])

    def test_full_config_masks_nothing(self) -> None:
        """full 配置应不屏蔽任何维度且 dim=OBS_DIM。"""
        self.assertEqual(D2_OBSERVATION_CONFIGS["full"]["mask_dims"], [])
        self.assertEqual(D2_OBSERVATION_CONFIGS["full"]["dim"], OBS_DIM)

    def test_wrapper_applies_each_config(self) -> None:
        """每个 D2 配置都应能实例化 wrapper 并正常 reset/step。"""
        for name, cfg in D2_OBSERVATION_CONFIGS.items():
            with self.subTest(config=name):
                env = _DummyEnv(obs_dim=OBS_DIM)
                wrapper = ObsMaskWrapper(env, mask_dims=cfg["mask_dims"])
                obs, _ = wrapper.reset()
                self.assertEqual(len(obs), OBS_DIM)
                for dim in cfg["mask_dims"]:
                    if 0 <= dim < OBS_DIM:
                        self.assertEqual(obs[dim], 0.0, f"{name} 维度 {dim} 未屏蔽")


if __name__ == "__main__":
    unittest.main()
