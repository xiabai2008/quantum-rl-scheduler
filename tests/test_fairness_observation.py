"""
公平性指数纳入观测空间测试（Issue #588）
Unit Tests for Fairness Index in Observation Space

测试覆盖：
- TestFairnessObservationConstants : 常量与特征名定义正确性
- TestFairnessObservationBuilder   : get_observation 中公平性指数的计算逻辑
- TestFairnessObservationEnv       : 环境集成测试（observation_space 形状、截断行为）
- TestFairnessObservationExplain   : 可解释性模块对 17 维特征名的兼容性
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.env_observation import get_observation
from src.scheduler.env_types import OBS_DIM, OBS_DIM_WITH_FAIRNESS, OBS_FAIRNESS_INDEX
from src.scheduler.explainability import STATE_FEATURE_NAMES, DecisionExplainer
from src.scheduler.fairness import MultiTenantFairnessTracker


def _make_mock_env(fairness_tracker=None) -> MagicMock:
    """构造带必要属性的 mock env，用于 get_observation 测试。"""
    env = MagicMock()
    # 基本属性
    env._quantum.available_ratio = 0.8
    env._quantum.fidelity = 0.95
    env._quantum.quantum_queue = 2
    env._classical.load = 0.3
    env._classical.queue = 1
    env._time_of_day = 0.5
    env._current_task = None
    env._task_queue = []
    env._machines = []
    env._fairness_tracker = fairness_tracker
    env._include_fairness_obs = True  # Issue #588: 默认开启公平性观测用于测试
    env.arrival_history = []
    env.current_time_window_arrivals = 0
    return env


# ============================================================
# TestFairnessObservationConstants
# ============================================================
class TestFairnessObservationConstants(unittest.TestCase):
    """测试常量与特征名定义的正确性。"""

    def test_obs_fairness_index_value(self):
        """OBS_FAIRNESS_INDEX 应为 16。"""
        self.assertEqual(OBS_FAIRNESS_INDEX, 16)

    def test_obs_dim_is_16(self):
        """OBS_DIM 应为 16（公平性为可选第17维，不影响默认维度）。"""
        self.assertEqual(OBS_DIM, 16)

    def test_obs_dim_with_fairness_is_17(self):
        """OBS_DIM_WITH_FAIRNESS 应为 17。"""
        self.assertEqual(OBS_DIM_WITH_FAIRNESS, 17)

    def test_obs_fairness_index_within_fairness_dim(self):
        """OBS_FAIRNESS_INDEX 应小于 OBS_DIM_WITH_FAIRNESS。"""
        self.assertLess(OBS_FAIRNESS_INDEX, OBS_DIM_WITH_FAIRNESS)

    def test_state_feature_names_has_17_entries(self):
        """STATE_FEATURE_NAMES 应有 17 个条目。"""
        self.assertEqual(len(STATE_FEATURE_NAMES), 17)

    def test_state_feature_names_includes_fairness(self):
        """STATE_FEATURE_NAMES[16] 应为 '公平性指数'。"""
        self.assertEqual(STATE_FEATURE_NAMES[OBS_FAIRNESS_INDEX], "公平性指数")

    def test_state_feature_names_index_alignment(self):
        """特征名索引应与 OBS_* 常量严格对应。"""
        from src.scheduler.env_types import (
            OBS_ARRIVAL_RATE_MA,
            OBS_AVG_CONNECTIVITY,
            OBS_AVG_WAIT_TIME,
            OBS_CLASSICAL_LOAD,
            OBS_COUPLING_DENSITY,
            OBS_CROSSTALK_RISK,
            OBS_FIDELITY,
            OBS_QUANTUM_QUEUE_RATIO,
            OBS_QUBIT_AVAILABILITY,
            OBS_QUEUE_LENGTH,
            OBS_SINGLE_GATE_FIDELITY,
            OBS_TASK_TYPE_CLASSICAL,
            OBS_TASK_TYPE_QUANTUM,
            OBS_TIME_OF_DAY,
            OBS_TWO_GATE_FIDELITY,
            OBS_URGENCY_LEVEL,
        )

        expected = {
            OBS_QUBIT_AVAILABILITY: "量子比特可用率",
            OBS_QUEUE_LENGTH: "队列长度",
            OBS_AVG_WAIT_TIME: "平均等待时间",
            OBS_FIDELITY: "量子比特保真度",
            OBS_CLASSICAL_LOAD: "经典资源负载",
            OBS_QUANTUM_QUEUE_RATIO: "量子队列占比",
            OBS_TIME_OF_DAY: "时间段",
            OBS_URGENCY_LEVEL: "任务紧急程度",
            OBS_TASK_TYPE_QUANTUM: "量子任务标记",
            OBS_TASK_TYPE_CLASSICAL: "经典任务标记",
            OBS_SINGLE_GATE_FIDELITY: "单比特门保真度",
            OBS_TWO_GATE_FIDELITY: "两比特门保真度",
            OBS_COUPLING_DENSITY: "耦合图密度",
            OBS_AVG_CONNECTIVITY: "平均连通度",
            OBS_CROSSTALK_RISK: "串扰风险",
            OBS_ARRIVAL_RATE_MA: "到达率MA",
            OBS_FAIRNESS_INDEX: "公平性指数",
        }
        for idx, name in expected.items():
            self.assertEqual(STATE_FEATURE_NAMES[idx], name, f"索引 {idx} 特征名不匹配")


# ============================================================
# TestFairnessObservationBuilder
# ============================================================
class TestFairnessObservationBuilder(unittest.TestCase):
    """测试 get_observation 中公平性指数的计算逻辑。"""

    def test_observation_dim_is_17_with_fairness(self):
        """开启 include_fairness_obs 时，观测向量维度应为 17。"""
        env = _make_mock_env()
        obs = get_observation(env)
        self.assertEqual(obs.shape, (17,))

    def test_observation_dim_is_16_without_fairness(self):
        """关闭 include_fairness_obs 时，观测向量维度应为 16。"""
        env = _make_mock_env()
        env._include_fairness_obs = False
        obs = get_observation(env)
        self.assertEqual(obs.shape, (16,))

    def test_fairness_index_zero_without_tracker(self):
        """无公平性跟踪器时，fairness_index 应为 0.0。"""
        env = _make_mock_env(fairness_tracker=None)
        obs = get_observation(env)
        self.assertEqual(obs[OBS_FAIRNESS_INDEX], 0.0)

    def test_fairness_index_zero_with_empty_tracker(self):
        """跟踪器无数据时，fairness_index 应为 0.0。"""
        tracker = MultiTenantFairnessTracker()
        env = _make_mock_env(fairness_tracker=tracker)
        obs = get_observation(env)
        self.assertEqual(obs[OBS_FAIRNESS_INDEX], 0.0)

    def test_fairness_index_one_for_equal_wait_times(self):
        """所有租户等待时间相等时，Jain 指数应为 1.0。"""
        tracker = MultiTenantFairnessTracker()
        # 两个租户各提交 1 个任务，等待 5 步
        tracker.record_submit("t1", wait_steps=5)
        tracker.record_submit("t2", wait_steps=5)
        env = _make_mock_env(fairness_tracker=tracker)
        obs = get_observation(env)
        self.assertAlmostEqual(obs[OBS_FAIRNESS_INDEX], 1.0, places=5)

    def test_fairness_index_less_than_one_for_unequal_wait(self):
        """租户等待时间不均时，Jain 指数应小于 1.0。"""
        tracker = MultiTenantFairnessTracker()
        tracker.record_submit("t1", wait_steps=1)
        tracker.record_submit("t2", wait_steps=50)
        env = _make_mock_env(fairness_tracker=tracker)
        obs = get_observation(env)
        self.assertGreater(obs[OBS_FAIRNESS_INDEX], 0.0)
        self.assertLess(obs[OBS_FAIRNESS_INDEX], 1.0)

    def test_fairness_index_in_valid_range(self):
        """公平性指数应在 [0, 1] 范围内。"""
        tracker = MultiTenantFairnessTracker()
        tracker.record_submit("t1", wait_steps=10)
        tracker.record_submit("t2", wait_steps=20)
        tracker.record_submit("t3", wait_steps=5)
        env = _make_mock_env(fairness_tracker=tracker)
        obs = get_observation(env)
        self.assertGreaterEqual(obs[OBS_FAIRNESS_INDEX], 0.0)
        self.assertLessEqual(obs[OBS_FAIRNESS_INDEX], 1.0)

    def test_fairness_index_single_tenant(self):
        """仅一个租户时，Jain 指数应为 1.0（完全公平）。"""
        tracker = MultiTenantFairnessTracker()
        tracker.record_submit("t1", wait_steps=10)
        env = _make_mock_env(fairness_tracker=tracker)
        obs = get_observation(env)
        self.assertAlmostEqual(obs[OBS_FAIRNESS_INDEX], 1.0, places=5)

    def test_fairness_index_dtype_is_float32(self):
        """观测向量 dtype 应为 float32。"""
        env = _make_mock_env()
        obs = get_observation(env)
        self.assertEqual(obs.dtype, np.float32)

    def test_fairness_index_updates_with_tracker_changes(self):
        """公平性指数应随跟踪器状态变化而更新。"""
        tracker = MultiTenantFairnessTracker()
        env = _make_mock_env(fairness_tracker=tracker)

        # 初始无数据 → 0.0
        obs1 = get_observation(env)
        self.assertEqual(obs1[OBS_FAIRNESS_INDEX], 0.0)

        # 添加公平数据 → 1.0
        tracker.record_submit("t1", wait_steps=5)
        tracker.record_submit("t2", wait_steps=5)
        obs2 = get_observation(env)
        self.assertAlmostEqual(obs2[OBS_FAIRNESS_INDEX], 1.0, places=5)

        # 添加不公平数据 → < 1.0
        tracker.record_submit("t3", wait_steps=100)
        obs3 = get_observation(env)
        self.assertLess(obs3[OBS_FAIRNESS_INDEX], 1.0)

    def test_fairness_index_resilient_to_tracker_errors(self):
        """跟踪器抛出异常时，fairness_index 应回退为 0.0。"""
        bad_tracker = MagicMock()
        bad_tracker.jain_wait_fairness.side_effect = ValueError("boom")
        env = _make_mock_env(fairness_tracker=bad_tracker)
        obs = get_observation(env)
        self.assertEqual(obs[OBS_FAIRNESS_INDEX], 0.0)

    def test_other_dimensions_unchanged(self):
        """新增公平性维度不应影响其他维度的值。"""
        tracker = MultiTenantFairnessTracker()
        tracker.record_submit("t1", wait_steps=5)
        tracker.record_submit("t2", wait_steps=5)

        env_with = _make_mock_env(fairness_tracker=tracker)
        env_without = _make_mock_env(fairness_tracker=None)
        env_without._include_fairness_obs = False

        obs_with = get_observation(env_with)
        obs_without = get_observation(env_without)

        # 前 16 维应完全相同
        np.testing.assert_array_equal(obs_with[:16], obs_without[:16])
        # env_with 有 17 维，env_without 有 16 维
        self.assertEqual(obs_with.shape, (17,))
        self.assertEqual(obs_without.shape, (16,))


# ============================================================
# TestFairnessObservationEnv
# ============================================================
class TestFairnessObservationEnv(unittest.TestCase):
    """环境集成测试：observation_space 形状与截断行为。"""

    def test_env_observation_space_shape_is_16_by_default(self):
        """默认环境的 observation_space shape 应为 (16,)。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=50, seed=0)
        self.assertEqual(env.observation_space.shape, (16,))

    def test_env_observation_space_shape_is_17_with_fairness(self):
        """开启 include_fairness_obs 时，observation_space shape 应为 (17,)。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=50, seed=0, include_fairness_obs=True)
        self.assertEqual(env.observation_space.shape, (17,))

    def test_env_reset_returns_17_dim_observation_with_fairness(self):
        """开启 include_fairness_obs 时，reset() 返回的观测向量应为 17 维。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=50, seed=0, include_fairness_obs=True)
        obs, _ = env.reset()
        self.assertEqual(obs.shape, (17,))

    def test_env_step_returns_17_dim_observation_with_fairness(self):
        """开启 include_fairness_obs 时，step() 返回的观测向量应为 17 维。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=50, seed=0, include_fairness_obs=True)
        env.reset()
        obs, _, _terminated, _truncated, _ = env.step(0)
        self.assertEqual(obs.shape, (17,))

    # NOTE: 此测试依赖 set_fairness_tracker 方法（跨PR依赖，main 分支不存在），暂时注释。
    # def test_env_with_fairness_tracker_has_nonzero_fairness(self):
    #     """设置公平性跟踪器后，观测中的公平性指数应可非零。"""
    #     from src.scheduler.env import QuantumSchedulingEnv
    #     env = QuantumSchedulingEnv(max_steps=50, seed=0)
    #     tracker = MultiTenantFairnessTracker()
    #     env.set_fairness_tracker(tracker)
    #     obs, _ = env.reset()
    #     self.assertEqual(obs[OBS_FAIRNESS_INDEX], 0.0)
    #     tracker.record_submit("t1", wait_steps=5)
    #     tracker.record_submit("t2", wait_steps=5)
    #     obs2, _, _, _, _ = env.step(0)
    #     self.assertAlmostEqual(obs2[OBS_FAIRNESS_INDEX], 1.0, places=5)

    # NOTE: 以下测试依赖 observation_dim 参数（PR #613 跨PR依赖，main 分支不存在），暂时注释。
    # def test_observation_dim_truncation_still_works(self):
    #     """observation_dim 截断参数应仍然有效（D2 消融兼容）。"""
    #     from src.scheduler.env import QuantumSchedulingEnv
    #     env = QuantumSchedulingEnv(max_steps=50, seed=0, observation_dim=8)
    #     self.assertEqual(env.observation_space.shape, (8,))
    #     obs, _ = env.reset()
    #     self.assertEqual(obs.shape, (8,))
    #
    # def test_observation_dim_rejects_exceeding_17(self):
    #     """observation_dim 超过 17 应报错。"""
    #     from src.scheduler.env import QuantumSchedulingEnv
    #     with self.assertRaises(ValueError):
    #         QuantumSchedulingEnv(max_steps=50, seed=0, observation_dim=18)
    #
    # def test_observation_dim_accepts_17(self):
    #     """observation_dim=17 应正常工作（等于完整维度）。"""
    #     from src.scheduler.env import QuantumSchedulingEnv
    #     env = QuantumSchedulingEnv(max_steps=50, seed=0, observation_dim=17)
    #     self.assertEqual(env.observation_space.shape, (17,))
    #     obs, _ = env.reset()
    #     self.assertEqual(obs.shape, (17,))


# ============================================================
# TestFairnessObservationExplain
# ============================================================
class TestFairnessObservationExplain(unittest.TestCase):
    """测试可解释性模块对 17 维特征名的兼容性。"""

    def test_explainer_default_feature_names_has_17(self):
        """DecisionExplainer 默认使用 17 个特征名。"""
        explainer = DecisionExplainer()
        self.assertEqual(len(explainer.feature_names), 17)

    def test_explainer_includes_fairness_name(self):
        """解释器特征名应包含 '公平性指数'。"""
        explainer = DecisionExplainer()
        self.assertIn("公平性指数", explainer.feature_names)

    def test_explainer_works_with_17_dim_state(self):
        """17 维状态向量应能正常生成决策记录。"""
        explainer = DecisionExplainer()
        state = np.random.rand(17)
        record = explainer.explain(
            state=state, action=1, q_values=np.array([1.0, 3.0, 2.0]),
            action_prob=0.85, step=10,
        )
        self.assertEqual(len(record.feature_contributions), 17)
        self.assertIn("公平性指数", record.feature_contributions)

    def test_explainer_format_with_17_dim(self):
        """17 维状态的格式化输出应包含公平性指数（如果贡献度高）。"""
        explainer = DecisionExplainer()
        # 构造公平性指数贡献度最高的状态
        state = np.zeros(17)
        state[OBS_FAIRNESS_INDEX] = 1.0
        record = explainer.explain(state=state, action=1, action_prob=0.9, step=1)
        text = explainer.format_explanation(record, top_k=3)
        self.assertIn("公平性指数", text)

    def test_explainer_feature_importance_with_17_dim(self):
        """17 维特征重要性聚合应包含公平性指数。"""
        explainer = DecisionExplainer()
        state = np.random.rand(17)
        records = [
            explainer.explain(state=state, action=i % 3, step=i)
            for i in range(5)
        ]
        importance = explainer.get_feature_importance(records)
        self.assertEqual(len(importance), 17)
        self.assertIn("公平性指数", importance)


if __name__ == "__main__":
    unittest.main()
