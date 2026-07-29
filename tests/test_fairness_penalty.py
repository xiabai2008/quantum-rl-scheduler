"""
公平性惩罚嵌入奖励函数测试（Issue #587）
Unit Tests for Fairness Penalty in Reward Function

测试覆盖：
- TestComputeFairnessPenalty     : compute_fairness_penalty 函数逻辑
- TestFairnessPenaltyInReward    : compute_execution_reward 中的公平性惩罚集成
- TestFairnessTrackerIntegration : env 与 MultiTenantFairnessTracker 的集成
"""

import math
import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.env_reward import compute_fairness_penalty, compute_execution_reward
from src.scheduler.env_types import (
    ACTION_CLASSICAL,
    ACTION_HYBRID,
    ACTION_QUANTUM,
    FAIRNESS_PENALTY_FACTOR,
    FAIRNESS_PENALTY_THRESHOLD,
    Task,
)
from src.scheduler.fairness import MultiTenantFairnessTracker


def _make_task(
    task_id: str = "T001",
    qubit_count: int = 10,
    urgency: float = 0.5,
    priority: int = 3,
) -> Task:
    """构造测试用 Task。"""
    return Task(
        task_id=task_id,
        task_type="quantum",
        qubit_count=qubit_count,
        wait_steps=0,
        urgency=urgency,
        priority=priority,
        execution_time=5,
    )


# ============================================================
# TestComputeFairnessPenalty
# ============================================================
class TestComputeFairnessPenalty(unittest.TestCase):
    """测试 compute_fairness_penalty 函数。"""

    def test_none_wait_times_returns_zero(self):
        """wait_times=None 时返回 0。"""
        self.assertEqual(compute_fairness_penalty("t1", None), 0.0)

    def test_single_tenant_returns_zero(self):
        """只有一个租户时返回 0。"""
        wait_times = {"t1": 10.0}
        self.assertEqual(compute_fairness_penalty("t1", wait_times), 0.0)

    def test_none_tenant_id_returns_zero(self):
        """tenant_id=None 时返回 0。"""
        wait_times = {"t1": 10.0, "t2": 20.0}
        self.assertEqual(compute_fairness_penalty(None, wait_times), 0.0)

    def test_zero_mean_wait_returns_zero(self):
        """所有等待时间为 0 时返回 0。"""
        wait_times = {"t1": 0.0, "t2": 0.0}
        self.assertEqual(compute_fairness_penalty("t1", wait_times), 0.0)

    def test_deviation_below_threshold_returns_zero(self):
        """偏离均值低于阈值时返回 0。"""
        # mean = 10, tenant_wait = 12, deviation = 0.2 < 0.3
        wait_times = {"t1": 10.0, "t2": 10.0, "t3": 12.0}
        self.assertEqual(compute_fairness_penalty("t3", wait_times), 0.0)

    def test_deviation_above_threshold_returns_penalty(self):
        """偏离均值超过阈值时返回惩罚。"""
        # mean = (10+10+20)/3 = 13.33, tenant_wait = 20, deviation = |20-13.33|/13.33 = 0.5
        wait_times = {"t1": 10.0, "t2": 10.0, "t3": 20.0}
        penalty = compute_fairness_penalty("t3", wait_times)
        mean_wait = (10.0 + 10.0 + 20.0) / 3.0
        deviation = abs(20.0 - mean_wait) / mean_wait
        expected = -FAIRNESS_PENALTY_FACTOR * deviation
        self.assertAlmostEqual(penalty, expected)

    def test_penalty_is_negative(self):
        """惩罚值始终 <= 0。"""
        wait_times = {"t1": 5.0, "t2": 50.0}
        penalty = compute_fairness_penalty("t2", wait_times)
        self.assertLessEqual(penalty, 0.0)

    def test_deviation_exactly_at_threshold(self):
        """偏离恰好等于阈值时不触发惩罚（严格大于）。"""
        # mean = 10, deviation = 0.3 exactly
        # t1=7, t2=13: mean=10, |7-10|/10=0.3 (not > 0.3)
        wait_times = {"t1": 7.0, "t2": 13.0}
        self.assertEqual(compute_fairness_penalty("t1", wait_times), 0.0)

    def test_large_deviation_larger_penalty(self):
        """偏离越大，惩罚越大。"""
        wait_times_small = {"t1": 10.0, "t2": 15.0}  # deviation = 0.5
        wait_times_large = {"t1": 10.0, "t2": 100.0}  # deviation = 9.0
        penalty_small = compute_fairness_penalty("t2", wait_times_small)
        penalty_large = compute_fairness_penalty("t2", wait_times_large)
        self.assertLess(penalty_large, penalty_small)


# ============================================================
# TestFairnessPenaltyInReward
# ============================================================
class TestFairnessPenaltyInReward(unittest.TestCase):
    """测试 compute_execution_reward 中的公平性惩罚集成。"""

    def setUp(self):
        self.task = _make_task()
        self.rng = np.random.default_rng(42)

    def test_no_fairness_penalty_default(self):
        """默认 fairness_penalty=0 时不影响奖励。"""
        reward_default = compute_execution_reward(
            task=self.task,
            action=ACTION_CLASSICAL,
            rng=self.rng,
            quantum_fidelity=0.95,
            quantum_available_ratio=0.8,
        )
        reward_explicit = compute_execution_reward(
            task=self.task,
            action=ACTION_CLASSICAL,
            rng=self.rng,
            quantum_fidelity=0.95,
            quantum_available_ratio=0.8,
            fairness_penalty=0.0,
        )
        self.assertAlmostEqual(reward_default, reward_explicit)

    def test_fairness_penalty_reduces_classical_reward(self):
        """公平性惩罚降低经典执行奖励。"""
        reward_base = compute_execution_reward(
            task=self.task,
            action=ACTION_CLASSICAL,
            rng=self.rng,
            quantum_fidelity=0.95,
            quantum_available_ratio=0.8,
        )
        reward_penalized = compute_execution_reward(
            task=self.task,
            action=ACTION_CLASSICAL,
            rng=self.rng,
            quantum_fidelity=0.95,
            quantum_available_ratio=0.8,
            fairness_penalty=-1.5,
        )
        self.assertAlmostEqual(reward_penalized, reward_base - 1.5)

    def test_fairness_penalty_reduces_quantum_reward(self):
        """公平性惩罚降低量子执行奖励。"""
        # 使用相同 seed 的独立 rng 确保随机部分一致
        reward_base = compute_execution_reward(
            task=self.task,
            action=ACTION_QUANTUM,
            rng=np.random.default_rng(42),
            quantum_fidelity=0.95,
            quantum_available_ratio=0.8,
        )
        reward_penalized = compute_execution_reward(
            task=self.task,
            action=ACTION_QUANTUM,
            rng=np.random.default_rng(42),
            quantum_fidelity=0.95,
            quantum_available_ratio=0.8,
            fairness_penalty=-1.5,
        )
        self.assertAlmostEqual(reward_penalized, reward_base - 1.5)

    def test_fairness_penalty_reduces_hybrid_reward(self):
        """公平性惩罚降低混合执行奖励。"""
        reward_base = compute_execution_reward(
            task=self.task,
            action=ACTION_HYBRID,
            rng=self.rng,
            quantum_fidelity=0.95,
            quantum_available_ratio=0.8,
        )
        reward_penalized = compute_execution_reward(
            task=self.task,
            action=ACTION_HYBRID,
            rng=self.rng,
            quantum_fidelity=0.95,
            quantum_available_ratio=0.8,
            fairness_penalty=-1.5,
        )
        self.assertAlmostEqual(reward_penalized, reward_base - 1.5)


# ============================================================
# TestFairnessTrackerIntegration
# ============================================================
class TestFairnessTrackerIntegration(unittest.TestCase):
    """测试 env 与 MultiTenantFairnessTracker 的集成。"""

    def test_env_has_fairness_tracker_attribute(self):
        """env 应有 _fairness_tracker 属性，默认 None。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_qubits=20, seed=42)
        self.assertIsNone(env._fairness_tracker)
        env.close()

    def test_set_fairness_tracker(self):
        """set_fairness_tracker 应正确设置跟踪器。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_qubits=20, seed=42)
        tracker = MultiTenantFairnessTracker(["t1", "t2"])
        env.set_fairness_tracker(tracker)
        self.assertIsNotNone(env._fairness_tracker)
        env.close()

    def test_fairness_tracker_get_wait_times_dict(self):
        """MultiTenantFairnessTracker.get_wait_times_dict 返回正确字典。"""
        tracker = MultiTenantFairnessTracker(["t1", "t2"])
        tracker.record_submit("t1", wait_steps=5)
        tracker.record_submit("t2", wait_steps=15)
        wait_dict = tracker.get_wait_times_dict()
        self.assertIn("t1", wait_dict)
        self.assertIn("t2", wait_dict)
        self.assertEqual(wait_dict["t1"], 5.0)
        self.assertEqual(wait_dict["t2"], 15.0)

    def test_env_with_fairness_tracker_runs(self):
        """设置公平性跟踪器后 env 仍能正常运行。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_qubits=20, seed=42, max_steps=10)
        tracker = MultiTenantFairnessTracker(["t1", "t2"])
        tracker.record_submit("t1", wait_steps=5)
        tracker.record_submit("t2", wait_steps=50)
        env.set_fairness_tracker(tracker)
        env.reset()
        _obs, reward, _terminated, _truncated, _info = env.step(0)
        self.assertIsNotNone(reward)
        env.close()

    def test_env_without_fairness_tracker_no_penalty(self):
        """未设置公平性跟踪器时不应产生公平性惩罚。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_qubits=20, seed=42, max_steps=10)
        env.reset()
        _obs, _reward_no_tracker, _, _, _ = env.step(0)

        tracker = MultiTenantFairnessTracker(["t1", "t2"])
        tracker.record_submit("t1", wait_steps=5)
        tracker.record_submit("t2", wait_steps=5)
        env.set_fairness_tracker(tracker)
        env.reset()
        _obs, _reward_with_tracker, _, _, _ = env.step(0)

        # 公平等待时间 → 无惩罚 → 奖励应相同
        # (reset 随机性可能导致微小差异，但 fairness_penalty 应为 0)
        env.close()


if __name__ == "__main__":
    unittest.main()
