"""
Power analysis 统一接口测试（Issue #597）

测试覆盖：
- TestPowerAnalysis           : power_analysis() 统一入口
- TestGeneratePreregistration : generate_preregistration() 预注册生成
- TestPlotPowerCurve          : plot_power_curve() 功效曲线
- TestInterpolateThreshold    : _interpolate_threshold() 辅助函数
- TestBackwardCompatibility   : 原有函数不受影响
- TestEdgeCases               : 边界场景
- TestIntegration             : 集成场景
"""

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.stats_significance import (
    _MANN_WHITNEY_ARE,
    _interpolate_threshold,
    generate_preregistration,
    minimum_detectable_effect,
    plot_power_curve,
    power_analysis,
    power_ttest,
    sample_size_for_effect,
)


# ============================================================
# TestPowerAnalysis
# ============================================================
class TestPowerAnalysis(unittest.TestCase):
    """测试 power_analysis() 统一入口。"""

    def test_returns_dict_with_required_keys(self):
        """应返回包含所有必需键的字典。"""
        result = power_analysis(effect_size=0.5)
        expected_keys = {
            "sample_size_per_group",
            "total_sample_size",
            "effect_size",
            "alpha",
            "power",
            "test_type",
            "ratio",
            "are_correction",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_default_test_type_is_mann_whitney(self):
        """默认 test_type 应为 'mann_whitney'。"""
        result = power_analysis(effect_size=0.5)
        self.assertEqual(result["test_type"], "mann_whitney")

    def test_mann_whitney_applies_are_correction(self):
        """Mann-Whitney 应应用 ARE 校正。"""
        result_mw = power_analysis(effect_size=0.5, test_type="mann_whitney")
        result_t = power_analysis(effect_size=0.5, test_type="t_test")
        self.assertTrue(result_mw["are_correction"])
        self.assertFalse(result_t["are_correction"])
        # Mann-Whitney 需要更多样本
        self.assertGreater(result_mw["sample_size_per_group"], result_t["sample_size_per_group"])

    def test_t_test_no_are_correction(self):
        """t_test 不应应用 ARE 校正。"""
        result = power_analysis(effect_size=0.5, test_type="t_test")
        self.assertFalse(result["are_correction"])

    def test_t_test_matches_sample_size_for_effect(self):
        """t_test 结果应与 sample_size_for_effect 一致。"""
        result = power_analysis(effect_size=0.5, test_type="t_test")
        expected_n = sample_size_for_effect(0.5, 0.05, 0.80)
        self.assertEqual(result["sample_size_per_group"], expected_n)

    def test_mann_whitney_sample_size_ceiling(self):
        """Mann-Whitney 样本量应向上取整。"""
        result = power_analysis(effect_size=0.5, test_type="t_test")
        n_t = result["sample_size_per_group"]
        result_mw = power_analysis(effect_size=0.5, test_type="mann_whitney")
        expected = math.ceil(n_t / _MANN_WHITNEY_ARE)
        self.assertEqual(result_mw["sample_size_per_group"], expected)

    def test_total_sample_size_with_ratio_1(self):
        """ratio=1 时总样本量应为 2 * n_per_group。"""
        result = power_analysis(effect_size=0.5, ratio=1.0)
        self.assertEqual(result["total_sample_size"], 2 * result["sample_size_per_group"])

    def test_total_sample_size_with_ratio_2(self):
        """ratio=2 时总样本量应为 n1 + 2*n1。"""
        result = power_analysis(effect_size=0.5, ratio=2.0, test_type="t_test")
        n1 = result["sample_size_per_group"]
        expected_total = n1 + int(n1 * 2.0)
        self.assertEqual(result["total_sample_size"], expected_total)

    def test_effect_size_uses_absolute_value(self):
        """效应量应取绝对值。"""
        result_pos = power_analysis(effect_size=0.5, test_type="t_test")
        result_neg = power_analysis(effect_size=-0.5, test_type="t_test")
        self.assertEqual(result_pos["sample_size_per_group"], result_neg["sample_size_per_group"])
        self.assertEqual(result_neg["effect_size"], 0.5)

    def test_default_alpha_and_power(self):
        """默认 alpha=0.05, power=0.80。"""
        result = power_analysis(effect_size=0.5)
        self.assertAlmostEqual(result["alpha"], 0.05)
        self.assertAlmostEqual(result["power"], 0.80)

    def test_custom_alpha(self):
        """应支持自定义 alpha。"""
        result = power_analysis(effect_size=0.5, alpha=0.01)
        self.assertAlmostEqual(result["alpha"], 0.01)

    def test_custom_power(self):
        """应支持自定义 power。"""
        result = power_analysis(effect_size=0.5, power=0.90)
        self.assertAlmostEqual(result["power"], 0.90)

    def test_higher_power_needs_more_samples(self):
        """更高的 power 需要更多样本。"""
        r80 = power_analysis(effect_size=0.5, power=0.80, test_type="t_test")
        r90 = power_analysis(effect_size=0.5, power=0.90, test_type="t_test")
        self.assertGreaterEqual(r90["sample_size_per_group"], r80["sample_size_per_group"])

    def test_larger_effect_needs_fewer_samples(self):
        """更大的效应量需要更少样本。"""
        r_small = power_analysis(effect_size=0.2, test_type="t_test")
        r_large = power_analysis(effect_size=0.8, test_type="t_test")
        self.assertGreater(r_small["sample_size_per_group"], r_large["sample_size_per_group"])

    def test_invalid_effect_size_zero(self):
        """effect_size=0 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            power_analysis(effect_size=0)

    def test_invalid_effect_size_negative(self):
        """effect_size < 0 应抛出 ValueError（虽然取绝对值，但 0 不行）。"""
        with self.assertRaises(ValueError):
            power_analysis(effect_size=-0.0)

    def test_invalid_alpha_zero(self):
        """alpha=0 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            power_analysis(effect_size=0.5, alpha=0.0)

    def test_invalid_alpha_one(self):
        """alpha=1 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            power_analysis(effect_size=0.5, alpha=1.0)

    def test_invalid_power_zero(self):
        """power=0 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            power_analysis(effect_size=0.5, power=0.0)

    def test_invalid_power_one(self):
        """power=1 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            power_analysis(effect_size=0.5, power=1.0)

    def test_invalid_test_type(self):
        """无效 test_type 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            power_analysis(effect_size=0.5, test_type="unknown")


# ============================================================
# TestGeneratePreregistration
# ============================================================
class TestGeneratePreregistration(unittest.TestCase):
    """测试 generate_preregistration()。"""

    def test_returns_dict_with_required_keys(self):
        """应返回包含所有必需键的字典。"""
        result = generate_preregistration(effect_size=0.5)
        expected_keys = {
            "experiment_name",
            "timestamp",
            "hypotheses",
            "strategies",
            "design",
            "sample_size_plan",
            "multiple_comparison",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_json_serializable(self):
        """结果应可 JSON 序列化。"""
        result = generate_preregistration(effect_size=0.5)
        s = json.dumps(result, ensure_ascii=False)
        self.assertIsInstance(s, str)
        # 反序列化应成功
        restored = json.loads(s)
        self.assertEqual(restored["experiment_name"], result["experiment_name"])

    def test_default_experiment_name(self):
        """未提供实验名时应使用默认值。"""
        result = generate_preregistration(effect_size=0.5)
        self.assertEqual(result["experiment_name"], "未命名实验")

    def test_custom_experiment_name(self):
        """应支持自定义实验名。"""
        result = generate_preregistration(effect_size=0.5, experiment_name="VQE对比实验")
        self.assertEqual(result["experiment_name"], "VQE对比实验")

    def test_default_hypotheses(self):
        """未提供假设时应使用默认值。"""
        result = generate_preregistration(effect_size=0.5)
        self.assertEqual(len(result["hypotheses"]), 1)
        self.assertIn("H1", result["hypotheses"][0])

    def test_custom_hypotheses(self):
        """应支持自定义假设。"""
        hyps = ["H1: PPO优于FCFS", "H2: PPO优于DQN"]
        result = generate_preregistration(effect_size=0.5, hypotheses=hyps)
        self.assertEqual(result["hypotheses"], hyps)

    def test_default_strategies(self):
        """未提供策略时应使用默认值。"""
        result = generate_preregistration(effect_size=0.5)
        self.assertEqual(result["strategies"], ["策略A", "策略B"])

    def test_custom_strategies(self):
        """应支持自定义策略列表。"""
        strategies = ["PPO", "DQN", "FCFS"]
        result = generate_preregistration(effect_size=0.5, strategies=strategies)
        self.assertEqual(result["strategies"], strategies)

    def test_timestamp_is_iso_format(self):
        """时间戳应为 ISO 格式字符串。"""
        result = generate_preregistration(effect_size=0.5)
        ts = result["timestamp"]
        self.assertIsInstance(ts, str)
        self.assertIn("T", ts)

    def test_design_contains_parameters(self):
        """design 字典应包含所有设计参数。"""
        result = generate_preregistration(
            effect_size=0.5, alpha=0.01, power=0.90, test_type="t_test", ratio=2.0
        )
        design = result["design"]
        self.assertAlmostEqual(design["effect_size"], 0.5)
        self.assertAlmostEqual(design["alpha"], 0.01)
        self.assertAlmostEqual(design["power"], 0.90)
        self.assertEqual(design["test_type"], "t_test")
        self.assertAlmostEqual(design["ratio"], 2.0)

    def test_sample_size_plan_matches_power_analysis(self):
        """sample_size_plan 应与 power_analysis 结果一致。"""
        pa = power_analysis(effect_size=0.5)
        result = generate_preregistration(effect_size=0.5)
        self.assertEqual(
            result["sample_size_plan"]["sample_size_per_group"],
            pa["sample_size_per_group"],
        )

    def test_multiple_comparison_bonferroni(self):
        """multiple_comparison 应使用 Bonferroni 校正。"""
        result = generate_preregistration(effect_size=0.5, strategies=["A", "B", "C"])
        mc = result["multiple_comparison"]
        self.assertEqual(mc["method"], "Bonferroni")
        # 3 strategies → 3 comparisons
        self.assertEqual(mc["n_comparisons"], 3)
        self.assertAlmostEqual(mc["corrected_alpha"], 0.05 / 3)

    def test_two_strategies_one_comparison(self):
        """2 个策略应有 1 次比较。"""
        result = generate_preregistration(effect_size=0.5, strategies=["A", "B"])
        mc = result["multiple_comparison"]
        self.assertEqual(mc["n_comparisons"], 1)
        self.assertAlmostEqual(mc["corrected_alpha"], 0.05)

    def test_one_strategy_one_comparison(self):
        """1 个策略应退化到 1 次比较。"""
        result = generate_preregistration(effect_size=0.5, strategies=["A"])
        mc = result["multiple_comparison"]
        self.assertEqual(mc["n_comparisons"], 1)


# ============================================================
# TestPlotPowerCurve
# ============================================================
class TestPlotPowerCurve(unittest.TestCase):
    """测试 plot_power_curve()。"""

    def test_returns_dict_with_required_keys(self):
        """应返回包含所有必需键的字典。"""
        result = plot_power_curve(n_per_group=30)
        expected_keys = {
            "effect_sizes",
            "powers",
            "n_per_group",
            "alpha",
            "test_type",
            "threshold_80",
            "threshold_90",
        }
        self.assertEqual(set(result.keys()), expected_keys)

    def test_default_effect_sizes(self):
        """默认效应量应为 [0.1, 0.2, ..., 2.0]。"""
        result = plot_power_curve(n_per_group=30)
        self.assertEqual(len(result["effect_sizes"]), 20)
        self.assertAlmostEqual(result["effect_sizes"][0], 0.1)
        self.assertAlmostEqual(result["effect_sizes"][-1], 2.0)

    def test_custom_effect_sizes(self):
        """应支持自定义效应量列表。"""
        custom = [0.1, 0.5, 1.0, 1.5]
        result = plot_power_curve(effect_sizes=custom, n_per_group=30)
        self.assertEqual(result["effect_sizes"], custom)

    def test_powers_length_matches_effect_sizes(self):
        """powers 长度应与 effect_sizes 一致。"""
        result = plot_power_curve(n_per_group=30)
        self.assertEqual(len(result["powers"]), len(result["effect_sizes"]))

    def test_powers_in_zero_one_range(self):
        """所有功效值应在 [0, 1] 范围内。"""
        result = plot_power_curve(n_per_group=30)
        for p in result["powers"]:
            self.assertGreaterEqual(p, 0.0)
            self.assertLessEqual(p, 1.0)

    def test_power_increases_with_effect_size(self):
        """功效应随效应量增大而递增。"""
        result = plot_power_curve(n_per_group=30)
        powers = result["powers"]
        for i in range(len(powers) - 1):
            self.assertGreaterEqual(powers[i + 1], powers[i] - 0.01)  # 允许微小数值误差

    def test_large_effect_achieves_high_power(self):
        """大效应量应达到高功效（>0.8）。"""
        result = plot_power_curve(n_per_group=30)
        self.assertGreater(result["powers"][-1], 0.8)

    def test_small_effect_low_power(self):
        """小效应量应有较低功效。"""
        result = plot_power_curve(n_per_group=10)
        self.assertLess(result["powers"][0], 0.2)

    def test_default_test_type_t_test(self):
        """默认 test_type 应为 't_test'。"""
        result = plot_power_curve(n_per_group=30)
        self.assertEqual(result["test_type"], "t_test")

    def test_mann_whitney_lower_power_than_t_test(self):
        """相同样本量下 Mann-Whitney 功效应低于 t 检验。"""
        result_t = plot_power_curve(n_per_group=30, test_type="t_test")
        result_mw = plot_power_curve(n_per_group=30, test_type="mann_whitney")
        # 至少有一些点 MW 功效 <= t 检验
        has_lower = any(
            mw <= t for mw, t in zip(result_mw["powers"], result_t["powers"], strict=True)
        )
        self.assertTrue(has_lower, "Mann-Whitney 应至少有一处功效低于 t 检验")

    def test_threshold_80_exists_for_large_n(self):
        """大样本量时应存在 0.8 功效阈值。"""
        result = plot_power_curve(n_per_group=50)
        self.assertIsNotNone(result["threshold_80"])

    def test_threshold_90_greater_than_80(self):
        """0.9 阈值应大于 0.8 阈值。"""
        result = plot_power_curve(n_per_group=50)
        if result["threshold_80"] is not None and result["threshold_90"] is not None:
            self.assertGreaterEqual(result["threshold_90"], result["threshold_80"])

    def test_threshold_none_for_small_n(self):
        """小样本量时可能无法达到 0.9 功效。"""
        result = plot_power_curve(n_per_group=5)
        # 小样本量时，最大功效可能低于 0.9
        if max(result["powers"]) < 0.9:
            self.assertIsNone(result["threshold_90"])

    def test_more_samples_higher_power(self):
        """更多样本量应产生更高功效。"""
        r_small = plot_power_curve(n_per_group=10)
        r_large = plot_power_curve(n_per_group=100)
        # 在中等效应量（如 0.5）处的功效
        idx = r_small["effect_sizes"].index(0.5)
        self.assertGreater(r_large["powers"][idx], r_small["powers"][idx])


# ============================================================
# TestInterpolateThreshold
# ============================================================
class TestInterpolateThreshold(unittest.TestCase):
    """测试 _interpolate_threshold()。"""

    def test_exact_match(self):
        """y 值恰好等于阈值时应返回对应 x。"""
        xs = [1.0, 2.0, 3.0]
        ys = [0.5, 0.8, 0.95]
        result = _interpolate_threshold(xs, ys, 0.5)
        self.assertAlmostEqual(result, 1.0)

    def test_linear_interpolation(self):
        """应在两个点之间线性插值。"""
        xs = [1.0, 2.0]
        ys = [0.5, 1.0]
        result = _interpolate_threshold(xs, ys, 0.75)
        self.assertAlmostEqual(result, 1.5)

    def test_threshold_above_all(self):
        """阈值高于所有 y 值时应返回 None。"""
        xs = [1.0, 2.0, 3.0]
        ys = [0.1, 0.2, 0.3]
        result = _interpolate_threshold(xs, ys, 0.5)
        self.assertIsNone(result)

    def test_threshold_at_last_point(self):
        """阈值等于最后一个 y 值时应返回最后的 x。"""
        xs = [1.0, 2.0, 3.0]
        ys = [0.1, 0.2, 0.8]
        result = _interpolate_threshold(xs, ys, 0.8)
        self.assertAlmostEqual(result, 3.0)

    def test_empty_lists(self):
        """空列表应返回 None。"""
        result = _interpolate_threshold([], [], 0.5)
        self.assertIsNone(result)

    def test_single_point(self):
        """单点列表应返回 None（不足 2 点）。"""
        result = _interpolate_threshold([1.0], [0.5], 0.5)
        self.assertIsNone(result)

    def test_mismatched_lengths(self):
        """x 和 y 长度不一致应返回 None。"""
        result = _interpolate_threshold([1.0, 2.0], [0.5], 0.5)
        self.assertIsNone(result)

    def test_first_point_above_threshold(self):
        """第一个点就超过阈值时应返回第一个 x。"""
        xs = [1.0, 2.0, 3.0]
        ys = [0.9, 0.95, 0.99]
        result = _interpolate_threshold(xs, ys, 0.8)
        self.assertAlmostEqual(result, 1.0)

    def test_flat_segment(self):
        """平坦段（y1 == y2）时应返回 x1。"""
        xs = [1.0, 2.0, 3.0]
        ys = [0.3, 0.3, 0.8]
        result = _interpolate_threshold(xs, ys, 0.3)
        self.assertAlmostEqual(result, 1.0)


# ============================================================
# TestBackwardCompatibility
# ============================================================
class TestBackwardCompatibility(unittest.TestCase):
    """测试原有函数不受影响。"""

    def test_power_ttest_still_works(self):
        """power_ttest 应正常工作。"""
        p = power_ttest(0.5, 30, 30)
        self.assertGreater(p, 0.0)
        self.assertLess(p, 1.0)

    def test_minimum_detectable_effect_still_works(self):
        """minimum_detectable_effect 应正常工作。"""
        mde = minimum_detectable_effect(30, 30)
        self.assertGreater(mde, 0.0)

    def test_sample_size_for_effect_still_works(self):
        """sample_size_for_effect 应正常工作。"""
        n = sample_size_for_effect(0.5)
        self.assertGreater(n, 0)

    def test_are_constant_value(self):
        """_MANN_WHITNEY_ARE 应为 0.955。"""
        self.assertAlmostEqual(_MANN_WHITNEY_ARE, 0.955)


# ============================================================
# TestEdgeCases
# ============================================================
class TestEdgeCases(unittest.TestCase):
    """测试边界场景。"""

    def test_power_analysis_large_effect(self):
        """大效应量应只需少量样本。"""
        result = power_analysis(effect_size=2.0, test_type="t_test")
        self.assertGreater(result["sample_size_per_group"], 0)
        self.assertLess(result["sample_size_per_group"], 10)

    def test_power_analysis_small_effect(self):
        """小效应量应需要大量样本。"""
        result = power_analysis(effect_size=0.1, test_type="t_test")
        self.assertGreater(result["sample_size_per_group"], 100)

    def test_preregistration_with_many_strategies(self):
        """多策略预注册应正确计算比较次数。"""
        strategies = [f"S{i}" for i in range(5)]
        result = generate_preregistration(effect_size=0.5, strategies=strategies)
        # 5 strategies → C(5,2) = 10 comparisons
        self.assertEqual(result["multiple_comparison"]["n_comparisons"], 10)
        self.assertAlmostEqual(result["multiple_comparison"]["corrected_alpha"], 0.05 / 10)

    def test_power_curve_single_effect_size(self):
        """单一效应量应能生成功效曲线。"""
        result = plot_power_curve(effect_sizes=[0.5], n_per_group=30)
        self.assertEqual(len(result["powers"]), 1)
        # 单点无法插值
        self.assertIsNone(result["threshold_80"])

    def test_power_curve_empty_effect_sizes(self):
        """空效应量列表应返回空功效列表。"""
        result = plot_power_curve(effect_sizes=[], n_per_group=30)
        self.assertEqual(len(result["powers"]), 0)

    def test_power_analysis_ratio_non_integer(self):
        """非整数比例应正确计算。"""
        result = power_analysis(effect_size=0.5, ratio=1.5, test_type="t_test")
        n1 = result["sample_size_per_group"]
        n2 = int(n1 * 1.5)
        self.assertEqual(result["total_sample_size"], n1 + n2)


# ============================================================
# TestIntegration
# ============================================================
class TestIntegration(unittest.TestCase):
    """集成场景测试。"""

    def test_full_workflow(self):
        """完整工作流：power_analysis → preregistration → power_curve。"""
        # 1. 功效分析
        pa = power_analysis(effect_size=0.5, test_type="mann_whitney")
        self.assertGreater(pa["sample_size_per_group"], 0)

        # 2. 预注册
        prereg = generate_preregistration(
            effect_size=0.5,
            test_type="mann_whitney",
            strategies=["PPO", "FCFS"],
            experiment_name="PPO vs FCFS 对比实验",
        )
        self.assertEqual(
            prereg["sample_size_plan"]["sample_size_per_group"], pa["sample_size_per_group"]
        )

        # 3. 功效曲线
        curve = plot_power_curve(n_per_group=pa["sample_size_per_group"], test_type="t_test")
        self.assertGreater(len(curve["powers"]), 0)

    def test_preregistration_json_round_trip(self):
        """预注册 JSON 应可完整往返。"""
        original = generate_preregistration(
            effect_size=0.8,
            alpha=0.01,
            power=0.90,
            test_type="t_test",
            experiment_name="高效应量实验",
            hypotheses=["H1: PPO显著优于Random"],
            strategies=["PPO", "Random"],
        )
        # 序列化
        json_str = json.dumps(original, ensure_ascii=False)
        # 反序列化
        restored = json.loads(json_str)
        self.assertEqual(restored["experiment_name"], original["experiment_name"])
        self.assertEqual(restored["strategies"], original["strategies"])
        self.assertEqual(restored["hypotheses"], original["hypotheses"])
        self.assertEqual(
            restored["sample_size_plan"]["sample_size_per_group"],
            original["sample_size_plan"]["sample_size_per_group"],
        )

    def test_power_curve_thresholds_reasonable(self):
        """功效曲线阈值应在合理范围内。"""
        result = plot_power_curve(n_per_group=64, test_type="t_test")
        if result["threshold_80"] is not None:
            self.assertGreater(result["threshold_80"], 0.0)
            self.assertLess(result["threshold_80"], 2.0)
        if result["threshold_90"] is not None:
            self.assertGreater(result["threshold_90"], 0.0)
            self.assertLess(result["threshold_90"], 2.0)


if __name__ == "__main__":
    unittest.main()
