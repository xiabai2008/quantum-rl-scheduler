"""
MAPPO 可学习路由测试（Issue #598）

测试覆盖：
- TestLearnableMachineScorer    : LearnableMachineScorer 类
- TestWrapperWithScorer         : 包装器集成可学习评分器
- TestMultiAgentPPOWithScorer   : MAPPO 智能体集成可学习评分器
- TestBackwardCompatibility     : 无评分器时行为不变
- TestFeatureExtraction         : 特征提取
- TestEdgeCases                 : 边界场景
"""

import os
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.env import MAX_QUEUE_SIZE, QuantumSchedulingEnv
from src.scheduler.marl import (
    LearnableMachineScorer,
    MultiAgentEnvWrapper,
    MultiAgentPPO,
)


# ============================================================
# TestLearnableMachineScorer
# ============================================================
class TestLearnableMachineScorer(unittest.TestCase):
    """测试 LearnableMachineScorer 类。"""

    def test_is_nn_module(self):
        """应是 nn.Module 子类。"""
        scorer = LearnableMachineScorer()
        self.assertIsInstance(scorer, torch.nn.Module)

    def test_default_hidden_dims(self):
        """默认隐藏层应为 (32, 16)。"""
        scorer = LearnableMachineScorer()
        self.assertEqual(scorer.hidden_dims, (32, 16))

    def test_custom_hidden_dims(self):
        """应支持自定义隐藏层。"""
        scorer = LearnableMachineScorer(hidden_dims=(64, 32, 16))
        self.assertEqual(scorer.hidden_dims, (64, 32, 16))

    def test_input_dim_is_3(self):
        """输入维度应为 3。"""
        scorer = LearnableMachineScorer()
        self.assertEqual(scorer.input_dim, 3)

    def test_forward_returns_tensor(self):
        """forward 应返回张量。"""
        scorer = LearnableMachineScorer()
        features = torch.randn(4, 3)
        output = scorer(features)
        self.assertIsInstance(output, torch.Tensor)
        self.assertEqual(output.shape, (4, 1))

    def test_forward_single_input(self):
        """单个输入应返回 (1,) 张量。"""
        scorer = LearnableMachineScorer()
        features = torch.tensor([0.5, 0.8, 0.1], dtype=torch.float32)
        output = scorer(features.unsqueeze(0))
        self.assertEqual(output.shape, (1, 1))

    def test_score_machine_returns_float(self):
        """score_machine 应返回 float。"""
        scorer = LearnableMachineScorer()
        score = scorer.score_machine(0.9, 0.8, 0.1)
        self.assertIsInstance(score, float)

    def test_score_machine_different_inputs_different_scores(self):
        """不同输入应产生不同评分。"""
        scorer = LearnableMachineScorer()
        score1 = scorer.score_machine(0.9, 0.8, 0.1)
        score2 = scorer.score_machine(0.1, 0.1, 0.9)
        self.assertNotAlmostEqual(score1, score2, places=4)

    def test_parameters_are_learnable(self):
        """参数应可学习（requires_grad=True）。"""
        scorer = LearnableMachineScorer()
        for param in scorer.parameters():
            self.assertTrue(param.requires_grad)

    def test_gradient_flow(self):
        """梯度应能通过前向传播流回。"""
        scorer = LearnableMachineScorer()
        features = torch.randn(2, 3)
        output = scorer(features)
        loss = output.sum()
        loss.backward()
        for param in scorer.parameters():
            self.assertIsNotNone(param.grad)


# ============================================================
# TestFeatureExtraction
# ============================================================
class TestFeatureExtraction(unittest.TestCase):
    """测试 LearnableMachineScorer.extract_features。"""

    def test_returns_numpy_array(self):
        """应返回 numpy 数组。"""
        features = LearnableMachineScorer.extract_features(0.9, 0.8, 0.1)
        self.assertIsInstance(features, np.ndarray)

    def test_shape_is_3(self):
        """形状应为 (3,)。"""
        features = LearnableMachineScorer.extract_features(0.9, 0.8, 0.1)
        self.assertEqual(features.shape, (3,))

    def test_dtype_is_float32(self):
        """数据类型应为 float32。"""
        features = LearnableMachineScorer.extract_features(0.9, 0.8, 0.1)
        self.assertEqual(features.dtype, np.float32)

    def test_values_clipped(self):
        """值应被裁剪到 [0, 1]。"""
        features = LearnableMachineScorer.extract_features(1.5, -0.2, 2.0)
        self.assertAlmostEqual(features[0], 1.0)
        self.assertAlmostEqual(features[1], 0.0)
        self.assertAlmostEqual(features[2], 1.0)

    def test_values_correct(self):
        """值应正确。"""
        features = LearnableMachineScorer.extract_features(0.5, 0.3, 0.7)
        np.testing.assert_array_almost_equal(features, [0.5, 0.3, 0.7])


# ============================================================
# TestWrapperWithScorer
# ============================================================
class TestWrapperWithScorer(unittest.TestCase):
    """测试包装器集成可学习评分器。"""

    @classmethod
    def setUpClass(cls):
        """创建测试环境。"""
        from src.scheduler.env import DEFAULT_MACHINE_CONFIGS

        cls.env = QuantumSchedulingEnv(machine_configs=DEFAULT_MACHINE_CONFIGS)

    def test_wrapper_without_scorer(self):
        """无评分器时 machine_scorer 应为 None。"""
        wrapper = MultiAgentEnvWrapper(self.env)
        self.assertIsNone(wrapper.machine_scorer)

    def test_wrapper_with_scorer(self):
        """有评分器时应正确设置。"""
        scorer = LearnableMachineScorer()
        wrapper = MultiAgentEnvWrapper(self.env, machine_scorer=scorer)
        self.assertIsNotNone(wrapper.machine_scorer)
        self.assertIs(wrapper.machine_scorer, scorer)

    def test_machine_score_without_scorer(self):
        """无评分器时应使用静态公式。"""
        wrapper = MultiAgentEnvWrapper(self.env)
        m = self.env._machines[0]
        expected = m.fidelity * m.available_ratio / (1.0 + m.quantum_queue)
        actual = wrapper._machine_score(0)
        self.assertAlmostEqual(actual, expected, places=6)

    def test_machine_score_with_scorer(self):
        """有评分器时应使用学习型评分。"""
        scorer = LearnableMachineScorer()
        wrapper = MultiAgentEnvWrapper(self.env, machine_scorer=scorer)
        score = wrapper._machine_score(0)
        self.assertIsInstance(score, float)

    def test_machine_score_with_scorer_differs_from_static(self):
        """学习型评分应与静态评分不同。"""
        wrapper_static = MultiAgentEnvWrapper(self.env)
        wrapper_learnable = MultiAgentEnvWrapper(self.env, machine_scorer=LearnableMachineScorer())
        # 至少一台机器的评分应不同
        any_diff = any(
            abs(wrapper_static._machine_score(i) - wrapper_learnable._machine_score(i)) > 1e-6
            for i in range(len(self.env._machines))
        )
        self.assertTrue(any_diff, "学习型评分应与静态评分不同")

    def test_aggregate_actions_with_scorer(self):
        """有评分器时 aggregate_actions 应正常工作。"""
        scorer = LearnableMachineScorer()
        wrapper = MultiAgentEnvWrapper(self.env, machine_scorer=scorer)
        # 所有机器投票量子执行
        actions = dict.fromkeys(wrapper.machine_names, 1)
        env_action, chosen = wrapper.aggregate_actions(actions)
        self.assertIn(env_action, (0, 1, 2))
        if env_action in (1, 2):
            self.assertIsNotNone(chosen)


# ============================================================
# TestMultiAgentPPOWithScorer
# ============================================================
class TestMultiAgentPPOWithScorer(unittest.TestCase):
    """测试 MAPPO 智能体集成可学习评分器。"""

    @classmethod
    def setUpClass(cls):
        """创建测试环境。"""
        from src.scheduler.env import DEFAULT_MACHINE_CONFIGS

        cls.env = QuantumSchedulingEnv(machine_configs=DEFAULT_MACHINE_CONFIGS)

    def test_default_no_scorer(self):
        """默认不启用评分器。"""
        agent = MultiAgentPPO(self.env, n_steps=32)
        self.assertIsNone(agent.machine_scorer)
        self.assertIsNone(agent.scorer_optimizer)

    def test_enable_scorer(self):
        """启用评分器时应正确创建。"""
        agent = MultiAgentPPO(self.env, n_steps=32, learnable_scorer=True)
        self.assertIsNotNone(agent.machine_scorer)
        self.assertIsInstance(agent.machine_scorer, LearnableMachineScorer)
        self.assertIsNotNone(agent.scorer_optimizer)

    def test_scorer_passed_to_wrapper(self):
        """评分器应传递给包装器。"""
        agent = MultiAgentPPO(self.env, n_steps=32, learnable_scorer=True)
        self.assertIs(agent.wrapper.machine_scorer, agent.machine_scorer)

    def test_custom_scorer_hidden(self):
        """应支持自定义评分器隐藏层。"""
        agent = MultiAgentPPO(self.env, n_steps=32, learnable_scorer=True, scorer_hidden=(64, 32))
        self.assertEqual(agent.machine_scorer.hidden_dims, (64, 32))

    def test_scorer_optimizer_created(self):
        """评分器优化器应被创建。"""
        agent = MultiAgentPPO(self.env, n_steps=32, learnable_scorer=True)
        self.assertIsNotNone(agent.scorer_optimizer)

    def test_scorer_on_device(self):
        """评分器应在正确设备上。"""
        agent = MultiAgentPPO(self.env, n_steps=32, learnable_scorer=True, device="cpu")
        # 检查评分器参数设备
        for param in agent.machine_scorer.parameters():
            self.assertEqual(param.device.type, "cpu")


# ============================================================
# TestBackwardCompatibility
# ============================================================
class TestBackwardCompatibility(unittest.TestCase):
    """测试向后兼容性。"""

    @classmethod
    def setUpClass(cls):
        """创建测试环境。"""
        from src.scheduler.env import DEFAULT_MACHINE_CONFIGS

        cls.env = QuantumSchedulingEnv(machine_configs=DEFAULT_MACHINE_CONFIGS)

    def test_wrapper_without_scorer_works(self):
        """无评分器的包装器应正常工作。"""
        wrapper = MultiAgentEnvWrapper(self.env)
        actions = dict.fromkeys(wrapper.machine_names, 0)
        env_action, chosen = wrapper.aggregate_actions(actions)
        self.assertEqual(env_action, 0)
        self.assertIsNone(chosen)

    def test_agent_without_scorer_works(self):
        """无评分器的智能体应正常工作。"""
        agent = MultiAgentPPO(self.env, n_steps=32)
        self.assertIsNone(agent.machine_scorer)

    def test_machine_score_static_formula_unchanged(self):
        """静态评分公式应不变。"""
        wrapper = MultiAgentEnvWrapper(self.env)
        m = self.env._machines[0]
        expected = m.fidelity * m.available_ratio / (1.0 + m.quantum_queue)
        self.assertAlmostEqual(wrapper._machine_score(0), expected, places=6)


# ============================================================
# TestEdgeCases
# ============================================================
class TestEdgeCases(unittest.TestCase):
    """测试边界场景。"""

    def test_scorer_with_zero_features(self):
        """零特征应能产生有效评分。"""
        scorer = LearnableMachineScorer()
        score = scorer.score_machine(0.0, 0.0, 0.0)
        self.assertIsInstance(score, float)

    def test_scorer_with_max_features(self):
        """最大特征应能产生有效评分。"""
        scorer = LearnableMachineScorer()
        score = scorer.score_machine(1.0, 1.0, 1.0)
        self.assertIsInstance(score, float)

    def test_scorer_with_single_hidden_layer(self):
        """单隐藏层应能正常工作。"""
        scorer = LearnableMachineScorer(hidden_dims=(8,))
        features = torch.randn(2, 3)
        output = scorer(features)
        self.assertEqual(output.shape, (2, 1))

    def test_scorer_with_empty_hidden(self):
        """空隐藏层（线性映射）应能工作。"""
        scorer = LearnableMachineScorer(hidden_dims=())
        features = torch.randn(2, 3)
        output = scorer(features)
        self.assertEqual(output.shape, (2, 1))

    def test_scorer_batch_inference(self):
        """批量推理应能工作。"""
        scorer = LearnableMachineScorer()
        features = torch.randn(10, 3)
        output = scorer(features)
        self.assertEqual(output.shape, (10, 1))

    def test_extract_features_with_zero(self):
        """extract_features 零值应正确处理。"""
        features = LearnableMachineScorer.extract_features(0.0, 0.0, 0.0)
        np.testing.assert_array_almost_equal(features, [0.0, 0.0, 0.0])

    def test_extract_features_with_negative(self):
        """extract_features 负值应被裁剪为 0。"""
        features = LearnableMachineScorer.extract_features(-0.5, -1.0, -0.1)
        np.testing.assert_array_almost_equal(features, [0.0, 0.0, 0.0])


# ============================================================
# TestScorerConsistency
# ============================================================
class TestScorerConsistency(unittest.TestCase):
    """测试评分器一致性。"""

    def test_same_input_same_output(self):
        """相同输入应产生相同输出。"""
        scorer = LearnableMachineScorer()
        score1 = scorer.score_machine(0.5, 0.5, 0.5)
        score2 = scorer.score_machine(0.5, 0.5, 0.5)
        self.assertAlmostEqual(score1, score2, places=6)

    def test_eval_mode_no_dropout(self):
        """eval 模式下不应有随机性。"""
        scorer = LearnableMachineScorer()
        scorer.eval()
        features = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
        with torch.no_grad():
            out1 = scorer(features).item()
            out2 = scorer(features).item()
        self.assertAlmostEqual(out1, out2, places=6)

    def test_scorer_can_be_saved_loaded(self):
        """评分器应可保存和加载。"""
        import tempfile

        scorer1 = LearnableMachineScorer()
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
        torch.save(scorer1.state_dict(), path)
        scorer2 = LearnableMachineScorer()
        scorer2.load_state_dict(torch.load(path, weights_only=True))
        os.unlink(path)

        features = torch.tensor([[0.5, 0.5, 0.5]], dtype=torch.float32)
        with torch.no_grad():
            out1 = scorer1(features)
            out2 = scorer2(features)
        torch.testing.assert_close(out1, out2)


if __name__ == "__main__":
    unittest.main()
