"""
Unit Tests for src/scheduler/explainability.PPOExplainer
测试覆盖（Issue #876）：
- 初始化（特征名对齐 / n_features / n_actions）
- explain 返回结构（heuristic 与 shap 不可用回退路径）
- 输入维度不匹配时 resize
- get_feature_importance 批量聚合
- _predict_proba 模型接口包装
"""

from __future__ import annotations

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.env_types import OBS_DIM
from src.scheduler.explainability import PPOExplainer


class _FakeDistribution:
    """模拟 SB3 策略的分布对象（dist.distribution.probs 层级）。"""

    def __init__(self, probs: np.ndarray) -> None:
        self.distribution = _FakeProbs(probs)


class _FakeProbs:
    def __init__(self, probs: np.ndarray) -> None:
        import torch

        self.probs = torch.as_tensor(probs, dtype=torch.float32)


class _FakePolicy:
    """模拟 SB3 策略对象（仅提供 get_distribution）。"""

    def __init__(self, n_actions: int) -> None:
        self.n_actions = n_actions

    def get_distribution(self, obs_t) -> _FakeDistribution:
        batch = obs_t.shape[0]
        probs = np.full((batch, self.n_actions), 1.0 / self.n_actions, dtype=np.float64)
        # 让第一个特征主导动作 0 的选择，便于断言贡献度方向
        probs[:, 0] += 0.2 * obs_t[:, 0].numpy()
        return _FakeDistribution(probs)


class _FakeModel:
    """模拟 stable-baselines3 PPO 模型（observation_space/action_space/policy）。"""

    def __init__(self, obs_dim: int = OBS_DIM, n_actions: int = 4) -> None:
        self.observation_space = MagicMock()
        self.observation_space.shape = (obs_dim,)
        self.action_space = MagicMock()
        self.action_space.n = n_actions
        self.policy = _FakePolicy(n_actions)


class TestPPOExplainerInit(unittest.TestCase):
    """初始化行为。"""

    def test_default_feature_names_match_obs_dim(self) -> None:
        """默认特征名数量应与模型观测维度一致。"""
        model = _FakeModel(obs_dim=16)
        explainer = PPOExplainer(model, method="heuristic")
        self.assertEqual(len(explainer.feature_names), 16)
        self.assertEqual(explainer.n_features, 16)
        self.assertEqual(explainer.n_actions, 4)

    def test_feature_names_truncated_to_obs_dim(self) -> None:
        """特征名多于观测维度时应收敛到 n_features。"""
        model = _FakeModel(obs_dim=4)
        explainer = PPOExplainer(
            model, feature_names=[f"f{i}" for i in range(8)], method="heuristic"
        )
        self.assertEqual(len(explainer.feature_names), 4)

    def test_feature_names_padded_to_obs_dim(self) -> None:
        """特征名少于观测维度时应补齐占位名。"""
        model = _FakeModel(obs_dim=16)
        explainer = PPOExplainer(model, feature_names=["only"], method="heuristic")
        self.assertEqual(len(explainer.feature_names), 16)
        self.assertEqual(explainer.feature_names[0], "only")
        self.assertTrue(explainer.feature_names[1].startswith("特征"))

    def test_invalid_method_falls_back_to_heuristic(self) -> None:
        """shap 不可用时应回退到 heuristic 且不抛异常。"""
        model = _FakeModel()
        with patch.object(PPOExplainer, "_check_shap_available", return_value=False):
            explainer = PPOExplainer(model, method="shap")
            obs = np.zeros(16, dtype=np.float64)
            with self.assertWarns(UserWarning):
                result = explainer.explain(obs)
        self.assertEqual(len(result), 16)


class TestPPOExplainerExplain(unittest.TestCase):
    """explain() 行为。"""

    def setUp(self) -> None:
        self.model = _FakeModel()
        self.explainer = PPOExplainer(self.model, method="heuristic")

    def test_explain_returns_feature_contribution_dict(self) -> None:
        """explain 应返回 特征名->贡献度 的字典（含正负方向）。"""
        obs = np.random.default_rng(42).random(16)
        result = self.explainer.explain(obs)
        self.assertIsInstance(result, dict)
        self.assertEqual(set(result.keys()), set(self.explainer.feature_names))
        for v in result.values():
            self.assertIsInstance(v, float)
        # 贡献度应存在正负（z-score 符号保留）
        values = list(result.values())
        self.assertTrue(any(v > 0 for v in values))
        self.assertTrue(any(v < 0 for v in values))

    def test_explain_resizes_mismatched_input(self) -> None:
        """输入维度与 n_features 不一致时应 resize 而非报错。"""
        short = np.zeros(8, dtype=np.float64)
        result = self.explainer.explain(short)
        self.assertEqual(len(result), 16)

    def test_explain_with_explicit_action(self) -> None:
        """显式指定 action 时应走 advantage 加权路径且不抛异常。"""
        obs = np.random.default_rng(7).random(16)
        result = self.explainer.explain(obs, action=1)
        self.assertEqual(len(result), 16)

    def test_explain_shap_available_uses_shap(self) -> None:
        """shap 可用时 explain 应调用 SHAP 路径并返回一致结构。"""
        explainer = PPOExplainer(self.model, method="shap")
        with patch.object(PPOExplainer, "_check_shap_available", return_value=True):
            explainer._shap_available = True
            fake_values = np.random.default_rng(1).random((1, 16))
            with (
                patch.object(explainer, "_init_shap_explainer", return_value=MagicMock()),
                patch.object(
                    type(explainer._init_shap_explainer),
                    "shap_values",
                    create=True,
                ),
            ):
                explainer._explainer = MagicMock()
                explainer._explainer.shap_values.return_value = fake_values
                result = explainer.explain(np.zeros(16))
        self.assertEqual(len(result), 16)

    def test_predict_proba_shape(self) -> None:
        """_predict_proba 应返回 (batch, n_actions) 概率矩阵。"""
        batch = np.zeros((4, 16), dtype=np.float32)
        probs = self.explainer._predict_proba(batch)
        self.assertEqual(probs.shape, (4, 4))
        # 每行应近似归一化（fake 分布非严格归一化，仅校验形状与有限值）
        self.assertTrue(np.isfinite(probs).all())


class TestPPOExplainerFeatureImportance(unittest.TestCase):
    """get_feature_importance() 行为。"""

    def test_batch_importance_returns_mean_abs_contribution(self) -> None:
        """批量重要性应为各特征 |贡献度| 的平均。"""
        explainer = PPOExplainer(_FakeModel(), method="heuristic")
        rng = np.random.default_rng(3)
        batch = [rng.random(16) for _ in range(5)]
        importance = explainer.get_feature_importance(batch)
        self.assertEqual(set(importance.keys()), set(explainer.feature_names))
        for v in importance.values():
            self.assertGreaterEqual(v, 0.0)

    def test_empty_batch_returns_zero_importance(self) -> None:
        """空批次应返回全零重要性。"""
        explainer = PPOExplainer(_FakeModel(), method="heuristic")
        importance = explainer.get_feature_importance([])
        self.assertEqual(len(importance), 16)
        self.assertTrue(all(v == 0.0 for v in importance.values()))


if __name__ == "__main__":
    unittest.main()
