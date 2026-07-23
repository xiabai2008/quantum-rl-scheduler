"""
量子RL调度系统 - 调度环境 seed 可复现性验证测试

验证 QuantumSchedulingEnv 的 reset(seed=) / step() 在相同 seed 下产生可复现的:
- 初始任务队列
- 观测序列
- episode 累计奖励

Issue #74: 补齐调度环境 seed 可复现性验证测试
"""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.env import QuantumSchedulingEnv


class TestEnvReproducibility(unittest.TestCase):
    """验证 QuantumSchedulingEnv 的 seed 可复现性。"""

    def test_reset_same_seed_produces_same_initial_queue(self) -> None:
        """相同 seed 的 reset 应产生相同的初始任务队列。

        验证 _task_queue 中每个 Task 的关键字段（task_id / task_type /
        qubit_count / priority）逐项一致。
        """
        env1 = QuantumSchedulingEnv(max_steps=50, seed=42)
        env2 = QuantumSchedulingEnv(max_steps=50, seed=42)
        env1.reset(seed=42)
        env2.reset(seed=42)

        self.assertEqual(len(env1._task_queue), len(env2._task_queue))
        for t1, t2 in zip(env1._task_queue, env2._task_queue, strict=False):
            self.assertEqual(t1.task_id, t2.task_id)
            self.assertEqual(t1.task_type, t2.task_type)
            self.assertEqual(t1.qubit_count, t2.qubit_count)
            self.assertEqual(t1.priority, t2.priority)

    def test_same_seed_produces_identical_observation_sequence(self) -> None:
        """相同 seed 跑 10 步，观测序列应完全相同。

        每步 action=0，用 np.array_equal 逐帧比较观测向量。
        """
        env1 = QuantumSchedulingEnv(max_steps=50, seed=42)
        env2 = QuantumSchedulingEnv(max_steps=50, seed=42)
        env1.reset(seed=42)
        env2.reset(seed=42)

        for _ in range(10):
            obs1, _, _, _, _ = env1.step(0)
            obs2, _, _, _, _ = env2.step(0)
            self.assertTrue(np.array_equal(obs1, obs2))

    def test_same_seed_produces_same_episode_reward(self) -> None:
        """相同 seed 跑完整 episode，episode_reward 和 _total_scheduled 应相同。

        跑到 terminated=True，比较累计奖励和已调度任务数。
        """
        env1 = QuantumSchedulingEnv(max_steps=20, seed=42)
        env2 = QuantumSchedulingEnv(max_steps=20, seed=42)
        env1.reset(seed=42)
        env2.reset(seed=42)

        terminated1 = False
        while not terminated1:
            _, _, terminated1, _, _ = env1.step(0)

        terminated2 = False
        while not terminated2:
            _, _, terminated2, _, _ = env2.step(0)

        self.assertEqual(env1._episode_reward, env2._episode_reward)
        self.assertEqual(env1._total_scheduled, env2._total_scheduled)

    def test_different_seeds_produce_different_trajectories(self) -> None:
        """不同 seed 应产生不同轨迹（5步后观测不完全相同）。

        seed=42 vs seed=123，跑 5 步后用 not np.array_equal 验证差异。
        """
        env1 = QuantumSchedulingEnv(max_steps=50, seed=42)
        env2 = QuantumSchedulingEnv(max_steps=50, seed=123)
        env1.reset(seed=42)
        env2.reset(seed=123)

        for _ in range(5):
            obs1, _, _, _, _ = env1.step(0)
            obs2, _, _, _, _ = env2.step(0)

        self.assertFalse(np.array_equal(obs1, obs2))


if __name__ == "__main__":
    unittest.main()
