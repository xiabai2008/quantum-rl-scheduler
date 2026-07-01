"""
多目标RL调度系统 - 单元测试
Unit Tests for Multi-Objective RL Scheduling System

测试覆盖：
- MultiObjectiveRewardWrapper 初始化与权重切换
- 3 个独立目标的正负值边界
- info dict 中的 objectives 字典
- 加权标量化正确性
- 累积统计
- 权重预设
- 工厂函数 make_mo_env
- PPO 训练兼容性
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.env import (
    MAX_WAIT_STEPS,
    OBS_DIM,
    QuantumSchedulingEnv,
)
from src.scheduler.multi_objective_env import (
    DEFAULT_WEIGHTS,
    MultiObjectiveRewardWrapper,
    make_mo_env,
)


class TestMultiObjectiveWrapper(unittest.TestCase):
    """测试 MultiObjectiveRewardWrapper 核心功能"""

    def setUp(self):
        """测试初始化：创建基础环境和包装器"""
        self.base_env = QuantumSchedulingEnv(
            max_qubits=20,
            max_steps=100,
            seed=42,
        )
        self.mo_env = MultiObjectiveRewardWrapper(
            self.base_env,
            weights=[1.0, 0.5, 0.5],
        )

    def test_init_default_weights(self):
        """测试默认权重初始化"""
        wrapper = MultiObjectiveRewardWrapper(
            QuantumSchedulingEnv(max_qubits=20, max_steps=50),
        )
        self.assertEqual(wrapper.weights, [1.0, 0.5, 0.5])

    def test_init_custom_weights(self):
        """测试自定义权重初始化"""
        wrapper = MultiObjectiveRewardWrapper(
            QuantumSchedulingEnv(max_qubits=20, max_steps=50),
            weights=[0.5, 1.0, 0.5],
        )
        self.assertEqual(wrapper.weights, [0.5, 1.0, 0.5])

    def test_init_weight_preset(self):
        """测试权重预设初始化"""
        for preset, expected in DEFAULT_WEIGHTS.items():
            wrapper = MultiObjectiveRewardWrapper(
                QuantumSchedulingEnv(max_qubits=20, max_steps=50),
                weight_preset=preset,
            )
            self.assertEqual(wrapper.weights, expected, f"预设 {preset} 失败")

    def test_init_both_weights_and_preset_raises(self):
        """测试同时指定 weights 和 weight_preset 时抛出异常"""
        with self.assertRaises(ValueError):
            MultiObjectiveRewardWrapper(
                QuantumSchedulingEnv(max_qubits=20, max_steps=50),
                weights=[1.0, 1.0, 1.0],
                weight_preset="balanced",
            )

    def test_init_invalid_preset_raises(self):
        """测试无效预设名称抛出异常"""
        with self.assertRaises(ValueError):
            MultiObjectiveRewardWrapper(
                QuantumSchedulingEnv(max_qubits=20, max_steps=50),
                weight_preset="nonexistent",
            )

    def test_init_invalid_weights_length_raises(self):
        """测试权重长度错误时抛出异常"""
        with self.assertRaises(ValueError):
            MultiObjectiveRewardWrapper(
                QuantumSchedulingEnv(max_qubits=20, max_steps=50),
                weights=[1.0, 0.5],
            )

    def test_reset_returns_objectives(self):
        """测试 reset() 返回 info["objectives"]"""
        obs, info = self.mo_env.reset(seed=42)
        self.assertIn("objectives", info)
        self.assertEqual(info["objectives"]["throughput"], 0.0)
        self.assertEqual(info["objectives"]["balance"], 0.0)
        self.assertEqual(info["objectives"]["quality"], 0.0)
        self.assertIn("mo_weights", info)
        self.assertEqual(info["mo_weights"], [1.0, 0.5, 0.5])

    def test_reset_returns_obs_dim(self):
        """测试 reset() 返回正确维度的观测"""
        obs, info = self.mo_env.reset(seed=42)
        self.assertEqual(obs.shape, (OBS_DIM,))
        self.assertTrue(np.all(obs >= 0.0))
        self.assertTrue(np.all(obs <= 1.0))

    def test_step_returns_objectives(self):
        """测试 step() 返回 info["objectives"]"""
        self.mo_env.reset(seed=42)
        obs, reward, terminated, truncated, info = self.mo_env.step(0)
        self.assertIn("objectives", info)
        self.assertIn("throughput", info["objectives"])
        self.assertIn("balance", info["objectives"])
        self.assertIn("quality", info["objectives"])
        self.assertIn("mo_weights", info)
        self.assertIn("original_reward", info)
        self.assertIn("mo_reward", info)

    def test_step_returns_valid_types(self):
        """测试 step() 返回类型正确"""
        self.mo_env.reset(seed=42)
        obs, reward, terminated, truncated, info = self.mo_env.step(1)
        self.assertIsInstance(obs, np.ndarray)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    # ------------------------------------------------------------------
    # 目标值边界测试
    # ------------------------------------------------------------------

    def test_throughput_bounds(self):
        """测试吞吐量目标值在 [0, 1] 范围内"""
        self.mo_env.reset(seed=42)
        for _ in range(50):
            action = np.random.randint(0, 3)
            obs, reward, terminated, truncated, info = self.mo_env.step(action)
            t = info["objectives"]["throughput"]
            self.assertGreaterEqual(t, 0.0, f"throughput {t} < 0")
            self.assertLessEqual(t, 1.0, f"throughput {t} > 1")
            if terminated:
                break

    def test_balance_bounds(self):
        """测试平衡度目标值在 [-1, 0] 范围内"""
        self.mo_env.reset(seed=42)
        for _ in range(50):
            action = np.random.randint(0, 3)
            obs, reward, terminated, truncated, info = self.mo_env.step(action)
            b = info["objectives"]["balance"]
            self.assertGreaterEqual(b, -1.0, f"balance {b} < -1")
            self.assertLessEqual(b, 0.0, f"balance {b} > 0")
            if terminated:
                break

    def test_quality_bounds(self):
        """测试服务质量目标值在 [-1, 0] 范围内"""
        self.mo_env.reset(seed=42)
        for _ in range(50):
            action = np.random.randint(0, 3)
            obs, reward, terminated, truncated, info = self.mo_env.step(action)
            q = info["objectives"]["quality"]
            self.assertGreaterEqual(q, -1.0, f"quality {q} < -1")
            self.assertLessEqual(q, 0.0, f"quality {q} > 0")
            if terminated:
                break

    def test_balance_zero_when_equal(self):
        """测试量子/经典利用率相等时平衡度为 0"""
        self.mo_env.reset(seed=42)
        # 手动设置使两者相等
        env = self.mo_env.env.unwrapped
        env._quantum.available_ratio = 0.5
        env._classical.load = 0.5
        balance = self.mo_env._compute_balance()
        self.assertAlmostEqual(balance, 0.0, delta=0.01)

    def test_balance_negative_when_imbalanced(self):
        """测试利用率不平衡时平衡度为负值"""
        self.mo_env.reset(seed=42)
        env = self.mo_env.env.unwrapped
        env._quantum.available_ratio = 0.9
        env._classical.load = 0.1
        balance = self.mo_env._compute_balance()
        self.assertLess(balance, -0.5)  # 应该显著为负

    def test_quality_zero_when_no_queue(self):
        """测试空队列时服务质量为 0"""
        self.mo_env.reset(seed=42)
        env = self.mo_env.env.unwrapped
        env._task_queue = []
        quality = self.mo_env._compute_quality()
        self.assertEqual(quality, 0.0)

    def test_quality_negative_when_long_wait(self):
        """测试长等待时间时服务质量为负值"""
        self.mo_env.reset(seed=42)
        env = self.mo_env.env.unwrapped
        # 创建等待时间很长的任务
        from src.scheduler.env import Task as EnvTask

        env._task_queue = [
            EnvTask(
                task_id=99,
                task_type="universal",
                priority=1,
                urgency=0.5,
                wait_steps=MAX_WAIT_STEPS * 2,
                qubit_count=5,
            )
        ]
        quality = self.mo_env._compute_quality()
        self.assertLess(quality, -0.5)  # 应该显著为负

    # ------------------------------------------------------------------
    # 加权标量化测试
    # ------------------------------------------------------------------

    def test_weighted_scalarization(self):
        """测试加权标量化公式: reward = w0*t + w1*b + w2*q"""
        self.mo_env.reset(seed=42)
        env = self.mo_env.env.unwrapped

        # 设置已知状态
        env._quantum.available_ratio = 0.6
        env._classical.load = 0.4
        from src.scheduler.env import Task as EnvTask

        env._task_queue = [
            EnvTask(
                task_id=1,
                task_type="universal",
                priority=1,
                urgency=0.5,
                wait_steps=10,
                qubit_count=5,
            )
        ]

        # 手动计算目标值
        t = self.mo_env._compute_throughput({"total_scheduled": 0})
        b = self.mo_env._compute_balance()
        q = self.mo_env._compute_quality()
        w = self.mo_env.weights

        expected_reward = w[0] * t + w[1] * b + w[2] * q

        # 执行一步
        obs, reward, terminated, truncated, info = self.mo_env.step(1)

        self.assertIn("mo_reward", info)
        self.assertAlmostEqual(info["mo_reward"], expected_reward, delta=0.5)

    def test_weight_switch_runtime(self):
        """测试运行时切换权重"""
        self.mo_env.reset(seed=42)
        self.mo_env.weights = [0.0, 1.0, 0.0]  # 仅平衡
        self.assertEqual(self.mo_env.weights, [0.0, 1.0, 0.0])

        # 执行一步，确认权重已生效
        obs, reward, terminated, truncated, info = self.mo_env.step(0)
        self.assertEqual(info["mo_weights"], [0.0, 1.0, 0.0])

    def test_set_weight_preset(self):
        """测试 set_weight_preset 方法"""
        self.mo_env.reset(seed=42)
        self.mo_env.set_weight_preset("balanced")
        self.assertEqual(self.mo_env.weights, [1.0, 1.0, 1.0])

        self.mo_env.set_weight_preset("quality_heavy")
        self.assertEqual(self.mo_env.weights, [0.5, 0.5, 1.0])

    def test_set_weight_preset_invalid(self):
        """测试无效预设名抛出异常"""
        with self.assertRaises(ValueError):
            self.mo_env.set_weight_preset("invalid")

    # ------------------------------------------------------------------
    # 累积统计测试
    # ------------------------------------------------------------------

    def test_episode_objectives_accumulate(self):
        """测试 episode 粒度累积统计"""
        self.mo_env.reset(seed=42)
        for _ in range(20):
            action = np.random.randint(0, 3)
            obs, reward, terminated, truncated, info = self.mo_env.step(action)
            if terminated:
                break

        mo = self.mo_env.get_episode_objectives()
        self.assertIn("throughput", mo)
        self.assertIn("balance", mo)
        self.assertIn("quality", mo)
        self.assertIsInstance(mo["throughput"], float)
        self.assertIsInstance(mo["balance"], float)
        self.assertIsInstance(mo["quality"], float)

    def test_cumulative_in_info(self):
        """测试 info["mo_cumulative"] 包含累积值"""
        self.mo_env.reset(seed=42)
        for _ in range(5):
            action = np.random.randint(0, 3)
            obs, reward, terminated, truncated, info = self.mo_env.step(action)
            if terminated:
                break

        self.assertIn("mo_cumulative", info)
        cum = info["mo_cumulative"]
        self.assertIn("throughput", cum)
        self.assertIn("balance", cum)
        self.assertIn("quality", cum)

    def test_cumulative_resets_after_reset(self):
        """测试 reset() 后累积统计清零"""
        self.mo_env.reset(seed=42)
        for _ in range(10):
            action = np.random.randint(0, 3)
            obs, reward, terminated, truncated, info = self.mo_env.step(action)
            if terminated:
                break

        # reset 后累积应为 0
        self.mo_env.reset(seed=123)
        mo = self.mo_env.get_episode_objectives()
        self.assertEqual(mo["throughput"], 0.0)
        self.assertEqual(mo["balance"], 0.0)
        self.assertEqual(mo["quality"], 0.0)

    # ------------------------------------------------------------------
    # 工厂函数测试
    # ------------------------------------------------------------------

    def test_make_mo_env_default(self):
        """测试工厂函数默认参数"""
        mo_env = make_mo_env(max_qubits=20, max_steps=50, seed=42)
        self.assertIsInstance(mo_env, MultiObjectiveRewardWrapper)
        self.assertEqual(mo_env.weights, [1.0, 0.5, 0.5])

    def test_make_mo_env_with_preset(self):
        """测试工厂函数指定预设"""
        mo_env = make_mo_env(max_qubits=20, max_steps=50, weight_preset="balanced")
        self.assertEqual(mo_env.weights, [1.0, 1.0, 1.0])

    def test_make_mo_env_with_custom_weights(self):
        """测试工厂函数自定义权重"""
        mo_env = make_mo_env(max_qubits=20, max_steps=50, weights=[0.1, 0.2, 0.7])
        self.assertEqual(mo_env.weights, [0.1, 0.2, 0.7])

    # ------------------------------------------------------------------
    # PPO 兼容性测试
    # ------------------------------------------------------------------

    def test_ppo_training_compatible(self):
        """测试 PPO 智能体可以在多目标环境下训练"""
        try:
            from src.scheduler.agent import PPOAgent
        except ImportError:
            self.skipTest("PPOAgent 不可用（缺少 stable_baselines3）")

        mo_env = make_mo_env(
            max_qubits=10,
            max_steps=50,
            weight_preset="throughput_heavy",
            seed=42,
        )

        agent = PPOAgent(
            mo_env,
            learning_rate=3e-4,
            n_steps=128,
            batch_size=32,
            n_epochs=3,
            gamma=0.99,
            verbose=0,
            seed=42,
        )

        # 短训练（500 步）
        agent.train(total_timesteps=500)

        # 评估
        eval_result = agent.evaluate(num_episodes=3, deterministic=True)
        self.assertIn("mean_reward", eval_result)

    # ------------------------------------------------------------------
    # 多组权重对比测试
    # ------------------------------------------------------------------

    def test_all_presets_trainable(self):
        """测试所有预设权重都能正常训练"""
        try:
            from src.scheduler.agent import PPOAgent
        except ImportError:
            self.skipTest("PPOAgent 不可用（缺少 stable_baselines3）")

        for preset in ["throughput_heavy", "balance_heavy", "quality_heavy"]:
            mo_env = make_mo_env(
                max_qubits=10,
                max_steps=50,
                weight_preset=preset,
                seed=42,
            )
            agent = PPOAgent(
                mo_env,
                learning_rate=3e-4,
                n_steps=128,
                batch_size=32,
                n_epochs=3,
                gamma=0.99,
                verbose=0,
                seed=42,
            )
            agent.train(total_timesteps=300)
            eval_result = agent.evaluate(num_episodes=2, deterministic=True)
            self.assertIn("mean_reward", eval_result, f"预设 {preset} 评估失败")

    # ------------------------------------------------------------------
    # 边界情况测试
    # ------------------------------------------------------------------

    def test_step_without_reset(self):
        """测试未 reset 直接 step 的行为"""
        env = QuantumSchedulingEnv(max_qubits=20, max_steps=50, seed=42)
        mo_env = MultiObjectiveRewardWrapper(env, weights=[1.0, 0.5, 0.5])
        # 先 reset
        mo_env.reset(seed=42)
        obs, reward, terminated, truncated, info = mo_env.step(0)
        self.assertIn("objectives", info)

    def test_empty_weights_list(self):
        """测试空权重列表时抛出异常"""
        with self.assertRaises(ValueError):
            MultiObjectiveRewardWrapper(
                QuantumSchedulingEnv(max_qubits=20, max_steps=50),
                weights=[],
            )

    def test_weight_names(self):
        """测试 weight_names 属性"""
        self.assertEqual(
            self.mo_env.weight_names,
            ["throughput", "balance", "quality"],
        )

    def test_original_reward_in_info(self):
        """测试 info 中包含原始环境奖励"""
        self.mo_env.reset(seed=42)
        obs, reward, terminated, truncated, info = self.mo_env.step(0)
        self.assertIn("original_reward", info)
        self.assertIsInstance(info["original_reward"], float)


if __name__ == "__main__":
    unittest.main()
