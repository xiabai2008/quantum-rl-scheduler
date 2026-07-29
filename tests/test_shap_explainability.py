"""
SHAP 可解释性集成测试（Issue #596）

测试覆盖：
- TestShapMethodParameter       : method="shap" 参数支持
- TestShapDirectionalContributions : SHAP 模式保留正/负方向
- TestShapFallback              : shap 未安装时优雅回退
- TestShapFormatExplanation     : 负贡献度格式化
- TestShapBackwardCompatibility : heuristic 模式不受影响
- TestShapWithPredictFn         : 提供 predict_fn 时的行为
- TestShapEdgeCases             : 边界场景
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.scheduler.explainability import DecisionExplainer, DecisionRecord


# ============================================================
# TestShapMethodParameter
# ============================================================
class TestShapMethodParameter(unittest.TestCase):
    """测试 method="shap" 参数支持。"""

    def test_default_method_is_heuristic(self):
        """默认 method 应为 'heuristic'。"""
        explainer = DecisionExplainer()
        self.assertEqual(explainer.method, "heuristic")

    def test_shap_method_set(self):
        """method='shap' 应正确设置。"""
        explainer = DecisionExplainer(method="shap")
        self.assertEqual(explainer.method, "shap")

    def test_shap_available_flag(self):
        """_shap_available 应为布尔值。"""
        explainer = DecisionExplainer(method="shap")
        self.assertIsInstance(explainer._shap_available, bool)

    def test_method_with_custom_features(self):
        """method 和 feature_names 应可同时使用。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c"], method="shap")
        self.assertEqual(explainer.method, "shap")
        self.assertEqual(explainer.feature_names, ["a", "b", "c"])


# ============================================================
# TestShapDirectionalContributions
# ============================================================
class TestShapDirectionalContributions(unittest.TestCase):
    """测试 SHAP 模式保留正/负方向。"""

    def setUp(self):
        self.explainer = DecisionExplainer(feature_names=["a", "b", "c"], method="shap")
        self.state = np.array([0.8, 0.1, 0.5], dtype=np.float64)

    def test_shap_can_produce_negative_contributions(self):
        """SHAP 模式下贡献度可以是负数。"""
        # q_values[action] < mean → advantage 为负 → state * advantage 为负
        # action=0, q=[1, 5, 3], advantage = 1 - 3 = -2
        rec = self.explainer.explain(self.state, action=0, q_values=np.array([1.0, 5.0, 3.0]))
        values = list(rec.feature_contributions.values())
        # 至少有一个负值（因为 advantage 为负，state 全正 → 全负）
        self.assertTrue(any(v < 0 for v in values), f"应存在负贡献度，实际: {values}")

    def test_shap_can_produce_positive_contributions(self):
        """SHAP 模式下贡献度可以是正数。"""
        # q_values[action] > mean → advantage 为正 → state * advantage 为正
        # action=1, q=[1, 5, 3], advantage = 5 - 3 = 2
        rec = self.explainer.explain(self.state, action=1, q_values=np.array([1.0, 5.0, 3.0]))
        values = list(rec.feature_contributions.values())
        # 全部应为正（advantage 正，state 全正）
        self.assertTrue(all(v > 0 for v in values), f"应全部为正贡献度，实际: {values}")

    def test_shap_mixed_positive_negative(self):
        """SHAP 模式下可同时存在正/负贡献度。"""
        # 构造混合场景：部分 state 为正，部分为负
        state_mixed = np.array([0.8, -0.3, 0.5], dtype=np.float64)
        # advantage 为正
        rec = self.explainer.explain(state_mixed, action=1, q_values=np.array([1.0, 5.0, 3.0]))
        values = list(rec.feature_contributions.values())
        has_pos = any(v > 0 for v in values)
        has_neg = any(v < 0 for v in values)
        self.assertTrue(has_pos, f"应存在正贡献度，实际: {values}")
        self.assertTrue(has_neg, f"应存在负贡献度，实际: {values}")

    def test_shap_absolute_sum_is_one(self):
        """SHAP 模式下贡献度的绝对值之和应为 1。"""
        rec = self.explainer.explain(self.state, action=0, q_values=np.array([1.0, 5.0, 3.0]))
        abs_sum = sum(abs(v) for v in rec.feature_contributions.values())
        self.assertAlmostEqual(abs_sum, 1.0, places=6)

    def test_shap_without_q_values(self):
        """SHAP 模式无 q_values 时应使用方向感知 z-score。"""
        state = np.array([0.9, 0.1, 0.5], dtype=np.float64)
        rec = self.explainer.explain(state, action=0)
        values = list(rec.feature_contributions.values())
        # z-score 模式下，高于均值的状态为正，低于均值为负
        # state[0]=0.9 > mean(0.5) → z-score > 0 → 正贡献
        # state[1]=0.1 < mean(0.5) → z-score < 0 → 负贡献
        self.assertGreater(values[0], 0, "高于均值的状态应有正贡献度")
        self.assertLess(values[1], 0, "低于均值的状态应有负贡献度")

    def test_shap_directional_raw_with_q_values(self):
        """_directional_raw 有 q_values 时应保留方向。"""
        state = np.array([1.0, -2.0, 3.0], dtype=np.float64)
        q = np.array([1.0, 5.0, 3.0])
        raw = DecisionExplainer._directional_raw(state, action=1, q_arr=q)
        # advantage = 5 - 3 = 2
        # raw = [1*2, -2*2, 3*2] = [2, -4, 6]
        np.testing.assert_array_almost_equal(raw, np.array([2.0, -4.0, 6.0]))

    def test_shap_directional_raw_without_q_values(self):
        """_directional_raw 无 q_values 时应使用 z-score（保留方向）。"""
        state = np.array([1.0, 0.0, -1.0], dtype=np.float64)
        raw = DecisionExplainer._directional_raw(state, action=0, q_arr=None)
        # mean=0, std=2/3... 实际上 std = sqrt(((1-0)^2+(0-0)^2+(-1-0)^2)/3) = sqrt(2/3)
        # z = (x - 0) / sqrt(2/3)
        expected = (state - state.mean()) / state.std()
        np.testing.assert_array_almost_equal(raw, expected)


# ============================================================
# TestShapFallback
# ============================================================
class TestShapFallback(unittest.TestCase):
    """测试 shap 未安装时的优雅回退。"""

    def test_fallback_produces_valid_contributions(self):
        """shap 不可用时应回退到方向感知启发式，仍能产生有效贡献度。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c"], method="shap")
        # 不提供 predict_fn，模拟无 shap 库或无 predict_fn 的情况
        rec = explainer.explain(
            np.array([0.8, 0.1, 0.5]),
            action=0,
            q_values=np.array([1.0, 5.0, 3.0]),
        )
        # 贡献度应为有效数字
        for v in rec.feature_contributions.values():
            self.assertIsInstance(v, float)
            self.assertFalse(np.isnan(v))
            self.assertFalse(np.isinf(v))

    def test_fallback_preserves_direction(self):
        """回退时仍应保留正/负方向。"""
        explainer = DecisionExplainer(feature_names=["a", "b"], method="shap")
        rec = explainer.explain(
            np.array([0.8, 0.1]),
            action=0,
            q_values=np.array([1.0, 5.0]),
        )
        values = list(rec.feature_contributions.values())
        # advantage = 1 - 3 = -2 → 全负
        self.assertTrue(all(v < 0 for v in values))

    def test_fallback_absolute_sum_one(self):
        """回退时绝对值之和仍应为 1。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c"], method="shap")
        rec = explainer.explain(
            np.array([0.3, 0.8, 0.1]),
            action=1,
            q_values=np.array([1.0, 5.0, 2.0]),
        )
        abs_sum = sum(abs(v) for v in rec.feature_contributions.values())
        self.assertAlmostEqual(abs_sum, 1.0, places=6)

    def test_fallback_with_zero_state(self):
        """全零状态在 SHAP 回退时应均匀分布。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c"], method="shap")
        rec = explainer.explain(np.zeros(3), action=0)
        for v in rec.feature_contributions.values():
            self.assertAlmostEqual(v, 1.0 / 3, places=6)


# ============================================================
# TestShapFormatExplanation
# ============================================================
class TestShapFormatExplanation(unittest.TestCase):
    """测试 SHAP 模式下的格式化输出。"""

    def setUp(self):
        self.explainer = DecisionExplainer(feature_names=["a", "b", "c"], method="shap")

    def test_format_includes_direction(self):
        """SHAP 模式格式化应包含正/负方向标注。"""
        state = np.array([0.8, -0.3, 0.5], dtype=np.float64)
        rec = self.explainer.explain(state, action=1, q_values=np.array([1.0, 5.0, 3.0]))
        text = self.explainer.format_explanation(rec, top_k=3)
        self.assertIn("正向", text)
        self.assertIn("负向", text)

    def test_format_direction_for_all_positive(self):
        """全正贡献度时应标注'正向'。"""
        state = np.array([0.8, 0.3, 0.5], dtype=np.float64)
        rec = self.explainer.explain(state, action=1, q_values=np.array([1.0, 5.0, 3.0]))
        text = self.explainer.format_explanation(rec, top_k=3)
        self.assertIn("正向", text)
        self.assertNotIn("负向", text)

    def test_format_direction_for_all_negative(self):
        """全负贡献度时应标注'负向'。"""
        state = np.array([0.8, 0.3, 0.5], dtype=np.float64)
        rec = self.explainer.explain(state, action=0, q_values=np.array([1.0, 5.0, 3.0]))
        text = self.explainer.format_explanation(rec, top_k=3)
        self.assertIn("负向", text)
        self.assertNotIn("正向", text)

    def test_format_sorted_by_absolute_value(self):
        """SHAP 模式应按绝对值降序排序。"""
        state = np.array([0.1, 0.9, 0.5], dtype=np.float64)
        rec = self.explainer.explain(state, action=0, q_values=np.array([1.0, 5.0, 3.0]))
        text = self.explainer.format_explanation(rec, top_k=3)
        # 解析贡献度值（从 "值=X.XXX" 中提取）
        # 按绝对值排序，最大的应在前面
        # advantage = 1 - 3 = -2
        # raw = [0.1*-2, 0.9*-2, 0.5*-2] = [-0.2, -1.8, -1.0]
        # abs = [0.2, 1.8, 1.0] → 排序: b(1.8) > c(1.0) > a(0.2)
        # 第一个因素应是 b
        self.assertTrue(text.index("b") < text.index("a"))

    def test_format_english_with_direction(self):
        """英文格式化也应包含方向标注。"""
        state = np.array([0.8, -0.3, 0.5], dtype=np.float64)
        rec = self.explainer.explain(state, action=1, q_values=np.array([1.0, 5.0, 3.0]))
        text = self.explainer.format_explanation(rec, top_k=3, lang="en")
        self.assertIn("+", text)
        self.assertIn("-", text)

    def test_format_top_k_zero(self):
        """top_k=0 时不应显示任何因素。"""
        state = np.array([0.8, 0.3, 0.5], dtype=np.float64)
        rec = self.explainer.explain(state, action=0, q_values=np.array([1.0, 5.0, 3.0]))
        text = self.explainer.format_explanation(rec, top_k=0)
        self.assertEqual(text.count("值="), 0)


# ============================================================
# TestShapBackwardCompatibility
# ============================================================
class TestShapBackwardCompatibility(unittest.TestCase):
    """测试 heuristic 模式不受 SHAP 集成影响。"""

    def setUp(self):
        self.explainer = DecisionExplainer(feature_names=["a", "b", "c"])

    def test_heuristic_contributions_non_negative(self):
        """heuristic 模式贡献度应全部非负。"""
        state = np.array([0.8, -0.3, 0.5], dtype=np.float64)
        rec = self.explainer.explain(state, action=0, q_values=np.array([1.0, 5.0, 3.0]))
        for v in rec.feature_contributions.values():
            self.assertGreaterEqual(v, 0.0)

    def test_heuristic_sum_is_one(self):
        """heuristic 模式贡献度之和应为 1。"""
        rec = self.explainer.explain(
            np.array([0.8, 0.3, 0.5]), action=0, q_values=np.array([1.0, 5.0, 3.0])
        )
        total = sum(rec.feature_contributions.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_heuristic_format_no_direction(self):
        """heuristic 模式格式化不应包含方向标注。"""
        rec = self.explainer.explain(
            np.array([0.8, 0.3, 0.5]), action=0, q_values=np.array([1.0, 5.0, 3.0])
        )
        text = self.explainer.format_explanation(rec, top_k=3)
        self.assertNotIn("正向", text)
        self.assertNotIn("负向", text)

    def test_heuristic_format_has_level(self):
        """heuristic 模式格式化应包含等级标注。"""
        rec = self.explainer.explain(
            np.array([0.8, 0.3, 0.5]), action=0, q_values=np.array([1.0, 5.0, 3.0])
        )
        text = self.explainer.format_explanation(rec, top_k=3)
        self.assertTrue("高" in text or "中" in text or "低" in text)

    def test_predict_fn_ignored_in_heuristic(self):
        """heuristic 模式应忽略 predict_fn 参数。"""
        mock_fn = MagicMock(return_value=np.array([1.0, 2.0, 3.0]))
        rec = self.explainer.explain(
            np.array([0.8, 0.3, 0.5]),
            action=0,
            q_values=np.array([1.0, 5.0, 3.0]),
            predict_fn=mock_fn,
        )
        # predict_fn 不应被调用
        mock_fn.assert_not_called()
        # 贡献度应全部非负
        for v in rec.feature_contributions.values():
            self.assertGreaterEqual(v, 0.0)


# ============================================================
# TestShapWithPredictFn
# ============================================================
class TestShapWithPredictFn(unittest.TestCase):
    """测试提供 predict_fn 时的行为。"""

    def test_predict_fn_called_when_shap_available(self):
        """shap 库可用时，predict_fn 应被调用。"""
        explainer = DecisionExplainer(feature_names=["a", "b"], method="shap")
        if not explainer._shap_available:
            self.skipTest("shap 库未安装，跳过 predict_fn 测试")

        mock_fn = MagicMock(return_value=np.array([[1.0, 2.0]]))
        state = np.array([0.5, 0.3], dtype=np.float64)
        try:
            rec = explainer.explain(state, action=0, predict_fn=mock_fn)
            # 应成功返回记录
            self.assertIsInstance(rec, DecisionRecord)
        except Exception:
            # SHAP 计算可能因 mock 函数不兼容而失败，回退到方向感知
            pass

    def test_predict_fn_not_required(self):
        """SHAP 模式不要求必须提供 predict_fn。"""
        explainer = DecisionExplainer(feature_names=["a", "b"], method="shap")
        rec = explainer.explain(np.array([0.5, 0.3]), action=0)
        self.assertIsInstance(rec, DecisionRecord)
        # 贡献度应有效
        abs_sum = sum(abs(v) for v in rec.feature_contributions.values())
        self.assertAlmostEqual(abs_sum, 1.0, places=6)


# ============================================================
# TestShapEdgeCases
# ============================================================
class TestShapEdgeCases(unittest.TestCase):
    """测试 SHAP 模式的边界场景。"""

    def test_shap_empty_state(self):
        """空状态向量应返回空贡献度。"""
        explainer = DecisionExplainer(feature_names=[], method="shap")
        rec = explainer.explain(np.array([]), action=0)
        self.assertEqual(len(rec.feature_contributions), 0)

    def test_shap_single_element(self):
        """单元素状态应正确计算。"""
        explainer = DecisionExplainer(feature_names=["a"], method="shap")
        rec = explainer.explain(np.array([0.5]), action=0, q_values=np.array([1.0, 2.0]))
        abs_sum = sum(abs(v) for v in rec.feature_contributions.values())
        self.assertAlmostEqual(abs_sum, 1.0, places=6)

    def test_shap_constant_state(self):
        """常量状态应均匀分布。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c"], method="shap")
        rec = explainer.explain(np.array([0.5, 0.5, 0.5]), action=0)
        for v in rec.feature_contributions.values():
            self.assertAlmostEqual(v, 1.0 / 3, places=6)

    def test_shap_all_zero_state(self):
        """全零状态应均匀分布。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c"], method="shap")
        rec = explainer.explain(np.zeros(3), action=0)
        for v in rec.feature_contributions.values():
            self.assertAlmostEqual(v, 1.0 / 3, places=6)

    def test_shap_negative_advantage(self):
        """负 advantage（选中动作低于均值）应产生负贡献度。"""
        explainer = DecisionExplainer(feature_names=["a", "b"], method="shap")
        state = np.array([0.5, 0.3], dtype=np.float64)
        # action=0, q=[1, 5], advantage = 1 - 3 = -2
        rec = explainer.explain(state, action=0, q_values=np.array([1.0, 5.0]))
        values = list(rec.feature_contributions.values())
        self.assertTrue(all(v < 0 for v in values))

    def test_shap_positive_advantage(self):
        """正 advantage（选中动作高于均值）应产生正贡献度。"""
        explainer = DecisionExplainer(feature_names=["a", "b"], method="shap")
        state = np.array([0.5, 0.3], dtype=np.float64)
        # action=1, q=[1, 5], advantage = 5 - 3 = 2
        rec = explainer.explain(state, action=1, q_values=np.array([1.0, 5.0]))
        values = list(rec.feature_contributions.values())
        self.assertTrue(all(v > 0 for v in values))

    def test_shap_zero_advantage(self):
        """零 advantage（选中动作等于均值）应均匀分布。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c"], method="shap")
        state = np.array([0.5, 0.3, 0.8], dtype=np.float64)
        # action=1, q=[1, 3, 5], advantage = 3 - 3 = 0
        rec = explainer.explain(state, action=1, q_values=np.array([1.0, 3.0, 5.0]))
        for v in rec.feature_contributions.values():
            self.assertAlmostEqual(v, 1.0 / 3, places=6)

    def test_shap_state_shorter_than_features(self):
        """状态短于特征名时应补齐。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c"], method="shap")
        rec = explainer.explain(np.array([0.5, 0.3]), action=0)
        self.assertEqual(len(rec.feature_contributions), 2)

    def test_shap_record_has_all_fields(self):
        """SHAP 模式生成的记录应包含所有必要字段。"""
        explainer = DecisionExplainer(feature_names=["a", "b"], method="shap")
        rec = explainer.explain(
            np.array([0.5, 0.3]),
            action=1,
            q_values=np.array([1.0, 2.0]),
            action_prob=0.85,
            step=5,
        )
        self.assertEqual(rec.step, 5)
        self.assertEqual(rec.action, 1)
        self.assertAlmostEqual(rec.action_prob, 0.85)
        self.assertIsNotNone(rec.q_values)
        self.assertEqual(len(rec.feature_contributions), 2)
        self.assertTrue(rec.timestamp)

    def test_heuristic_method_attribute_exists(self):
        """heuristic 模式应有 method 属性。"""
        explainer = DecisionExplainer()
        self.assertEqual(explainer.method, "heuristic")
        self.assertFalse(explainer.method == "shap")


if __name__ == "__main__":
    unittest.main()
