"""
统计显著性检验模块测试
Tests for src/utils/stats_significance.py

覆盖：
- 正态分布数据 → t 检验
- 非正态数据 → Mann-Whitney U
- 方差不齐 → Welch t
- 显著性判断
- 效应量计算正确性
- Bonferroni 校正
- 多策略两两比较
- 边界（单策略 / 空输入 / 样本不足）
- 置信区间
"""

import math

import numpy as np
import pytest

from src.utils.stats_significance import (
    cohen_d,
    compare_strategies,
    minimum_detectable_effect,
    normality_test,
    power_ttest,
    rank_biserial,
    sample_size_for_effect,
)


@pytest.fixture
def rng() -> np.random.Generator:
    """可复现的随机数生成器"""
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# 1. 正态分布数据 → t 检验
# ---------------------------------------------------------------------------


def test_normal_data_uses_ttest(rng: np.random.Generator) -> None:
    """两组正态数据且方差齐 → 应选择独立样本 t 检验"""
    a = rng.normal(loc=100, scale=10, size=30).tolist()
    b = rng.normal(loc=90, scale=10, size=30).tolist()
    results = compare_strategies({"A": a, "B": b}, alpha=0.05)
    pair = results["A vs B"]
    assert pair["test"] == "独立样本 t 检验"
    assert pair["effect_size_type"] == "Cohen's d"
    # 均值差 10、样本 30、标准差 10，差异应显著
    assert pair["significant"] is True


# ---------------------------------------------------------------------------
# 2. 非正态数据 → Mann-Whitney U
# ---------------------------------------------------------------------------


def test_nonnormal_data_uses_mannwhitney(rng: np.random.Generator) -> None:
    """非正态数据（指数分布）→ 应选择 Mann-Whitney U 检验"""
    # 指数分布偏态强，n>=80 时 normaltest 必拒绝正态性
    a = rng.exponential(scale=1.0, size=80).tolist()
    b = rng.exponential(scale=5.0, size=80).tolist()
    results = compare_strategies({"A": a, "B": b}, alpha=0.05)
    pair = results["A vs B"]
    assert pair["test"] == "Mann-Whitney U 检验"
    assert pair["effect_size_type"] == "rank-biserial correlation"


# ---------------------------------------------------------------------------
# 3. 方差不齐 → Welch t
# ---------------------------------------------------------------------------


def test_unequal_variance_uses_welch(rng: np.random.Generator) -> None:
    """两组正态数据但方差不齐 → 应选择 Welch t 检验"""
    a = rng.normal(loc=100, scale=2, size=30).tolist()
    b = rng.normal(loc=100, scale=20, size=30).tolist()
    results = compare_strategies({"A": a, "B": b}, alpha=0.05)
    pair = results["A vs B"]
    # 两组均正态，方差比 100:1，Levene 必拒绝方差齐
    assert pair["test"] == "Welch t 检验"
    assert pair["effect_size_type"] == "Cohen's d"


# ---------------------------------------------------------------------------
# 4. 显著性判断（p_value < alpha）
# ---------------------------------------------------------------------------


def test_significant_when_large_difference(rng: np.random.Generator) -> None:
    """均值差大 → p_value < alpha 且 significant=True"""
    a = rng.normal(loc=100, scale=5, size=30).tolist()
    b = rng.normal(loc=80, scale=5, size=30).tolist()
    results = compare_strategies({"A": a, "B": b}, alpha=0.05)
    pair = results["A vs B"]
    assert pair["p_value"] < 0.05
    # 单次比较，bonferroni_alpha == alpha
    assert math.isclose(pair["bonferroni_alpha"], 0.05, rel_tol=1e-9)
    assert pair["significant"] is True


def test_nonsignificant_when_identical_data() -> None:
    """两组完全相同数据 → p_value=1.0、不显著"""
    data = [100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 99.8]
    results = compare_strategies({"A": data, "B": list(data)}, alpha=0.05)
    pair = results["A vs B"]
    assert pair["p_value"] >= 0.05
    assert pair["significant"] is False


# ---------------------------------------------------------------------------
# 5. 效应量计算正确性
# ---------------------------------------------------------------------------


def test_cohen_d_correctness() -> None:
    """Cohen's d 与手算公式一致"""
    x = [10.0, 12.0, 14.0, 16.0, 18.0]
    y = [5.0, 7.0, 9.0, 11.0, 13.0]
    arr_x = np.array(x)
    arr_y = np.array(y)
    n1, n2 = len(arr_x), len(arr_y)
    mean_diff = float(arr_x.mean() - arr_y.mean())
    var_x = float(arr_x.var(ddof=1))
    var_y = float(arr_y.var(ddof=1))
    pooled_var = ((n1 - 1) * var_x + (n2 - 1) * var_y) / (n1 + n2 - 2)
    expected = mean_diff / math.sqrt(pooled_var)
    assert math.isclose(cohen_d(x, y), expected, rel_tol=1e-9)


def test_cohen_d_sign() -> None:
    """Cohen's d 符号：x 均值高 → 正"""
    assert cohen_d([10.0, 20.0, 30.0], [1.0, 2.0, 3.0]) > 0
    assert cohen_d([1.0, 2.0, 3.0], [10.0, 20.0, 30.0]) < 0


def test_rank_biserial_extremes() -> None:
    """rank-biserial：x 全小于 y → -1；x 全大于 y → +1"""
    low = [1.0, 2.0, 3.0, 4.0, 5.0]
    high = [6.0, 7.0, 8.0, 9.0, 10.0]
    assert math.isclose(rank_biserial(low, high), -1.0, abs_tol=1e-9)
    assert math.isclose(rank_biserial(high, low), 1.0, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# 6. Bonferroni 校正
# ---------------------------------------------------------------------------


def test_bonferroni_correction_three_strategies() -> None:
    """3 策略 → 3 次比较 → 校正 α = 0.05/3"""
    data = {
        "A": [1.0, 2.0, 3.0, 4.0, 5.0],
        "B": [2.0, 3.0, 4.0, 5.0, 6.0],
        "C": [10.0, 11.0, 12.0, 13.0, 14.0],
    }
    results = compare_strategies(data, alpha=0.05)
    for pair in results.values():
        assert math.isclose(pair["bonferroni_alpha"], 0.05 / 3, rel_tol=1e-9)
        assert pair["n_comparisons"] == 3
    # A vs C 差异巨大，即使校正后也应显著
    assert results["A vs C"]["significant"] is True


# ---------------------------------------------------------------------------
# 7. 多策略两两比较
# ---------------------------------------------------------------------------


def test_multiple_strategies_pairwise() -> None:
    """4 策略 → C(4,2)=6 次两两比较，键齐全"""
    data = {
        "A": [1.0, 2.0, 3.0],
        "B": [4.0, 5.0, 6.0],
        "C": [7.0, 8.0, 9.0],
        "D": [10.0, 11.0, 12.0],
    }
    results = compare_strategies(data, alpha=0.05)
    assert len(results) == 6
    expected_keys = {"A vs B", "A vs C", "A vs D", "B vs C", "B vs D", "C vs D"}
    assert set(results.keys()) == expected_keys
    for pair in results.values():
        assert pair["n_comparisons"] == 6


# ---------------------------------------------------------------------------
# 8. 边界：单策略输入、空输入、样本不足
# ---------------------------------------------------------------------------


def test_empty_input_returns_empty() -> None:
    """空输入 → 返回空字典"""
    assert compare_strategies({}) == {}


def test_single_strategy_returns_empty() -> None:
    """单策略输入 → 返回空字典（无法比较）"""
    assert compare_strategies({"A": [1.0, 2.0, 3.0]}) == {}


def test_insufficient_samples_marked() -> None:
    """样本不足（<2）→ 标记为无法检验"""
    data = {"A": [1.0, 2.0, 3.0], "B": [5.0]}
    results = compare_strategies(data, alpha=0.05)
    pair = results["A vs B"]
    assert pair["test"] == "无法检验"
    assert pair["significant"] is False
    assert "样本量不足" in pair["interpretation"]


# ---------------------------------------------------------------------------
# 9. 置信区间
# ---------------------------------------------------------------------------


def test_confidence_interval_brackets_mean_diff(rng: np.random.Generator) -> None:
    """95% CI 应包含均值差，且 lower < mean_diff < upper"""
    a = rng.normal(loc=100, scale=10, size=30).tolist()
    b = rng.normal(loc=90, scale=10, size=30).tolist()
    results = compare_strategies({"A": a, "B": b}, alpha=0.05)
    pair = results["A vs B"]
    assert pair["ci_lower"] < pair["mean_diff"] < pair["ci_upper"]
    # 均值差应明显大于 0（约 10）
    assert pair["mean_diff"] > 5.0


def test_confidence_interval_covers_zero_when_identical() -> None:
    """不显著（相同数据）时 95% CI 应包含 0"""
    data = [100.0, 101.0, 99.0, 100.5, 99.5, 100.2, 99.8]
    results = compare_strategies({"A": data, "B": list(data)}, alpha=0.05)
    pair = results["A vs B"]
    assert pair["ci_lower"] <= 0.0 <= pair["ci_upper"]


# ---------------------------------------------------------------------------
# 正态性检验辅助函数
# ---------------------------------------------------------------------------


def test_normality_test_shapiro_path(rng: np.random.Generator) -> None:
    """n < 50 → 使用 Shapiro-Wilk"""
    data = rng.normal(loc=0, scale=1, size=49).tolist()
    is_normal, _, test_name = normality_test(data, alpha=0.05)
    assert test_name == "Shapiro-Wilk"
    assert is_normal is True


def test_normality_test_normaltest_path(rng: np.random.Generator) -> None:
    """n >= 50 → 使用 D'Agostino K²"""
    data = rng.normal(loc=0, scale=1, size=80).tolist()
    is_normal, _, test_name = normality_test(data, alpha=0.05)
    assert test_name == "D'Agostino K²"
    assert is_normal is True


def test_normality_test_rejects_nonnormal() -> None:
    """明显非正态数据 → is_normal=False"""
    # 双峰分布：一半 0、一半 100
    data = [0.0] * 40 + [100.0] * 40
    is_normal, p_value, _ = normality_test(data, alpha=0.05)
    assert is_normal is False
    assert p_value < 0.05


def test_normality_test_too_few_samples() -> None:
    """n < 3 → 保守判为非正态"""
    is_normal, _, test_name = normality_test([1.0, 2.0], alpha=0.05)
    assert is_normal is False
    assert "不足" in test_name


# ---------------------------------------------------------------------------
# 结果结构完整性
# ---------------------------------------------------------------------------


def test_result_contains_all_required_fields(rng: np.random.Generator) -> None:
    """每个对比结果应包含规范字段"""
    a = rng.normal(loc=100, scale=5, size=20).tolist()
    b = rng.normal(loc=90, scale=5, size=20).tolist()
    results = compare_strategies({"A": a, "B": b}, alpha=0.05)
    pair = results["A vs B"]
    required = {
        "test",
        "statistic",
        "p_value",
        "significant",
        "effect_size",
        "effect_size_type",
        "mean_diff",
        "ci_lower",
        "ci_upper",
        "bonferroni_alpha",
        "n_comparisons",
        "interpretation",
    }
    assert required.issubset(pair.keys())
    # interpretation 应为非空中文
    assert isinstance(pair["interpretation"], str)
    assert len(pair["interpretation"]) > 0


# ---------------------------------------------------------------------------
# 10. 检验力分析（Power Analysis）
# ---------------------------------------------------------------------------


class TestPowerAnalysis:
    """统计检验力分析函数测试"""

    def test_power_ttest_large_effect(self) -> None:
        """大效应量（d=1.5）且 n=30+30，检验力应 > 0.95"""
        power = power_ttest(d=1.5, n1=30, n2=30, alpha=0.05)
        assert not math.isnan(power)
        assert power > 0.95

    def test_power_ttest_small_sample(self) -> None:
        """中等效应量（d=0.5）且小样本 n=5+5，检验力应 < 0.5"""
        power = power_ttest(d=0.5, n1=5, n2=5, alpha=0.05)
        assert not math.isnan(power)
        assert power < 0.5

    def test_power_ttest_boundary(self) -> None:
        """边界：n < 2 或 d 为 nan → 返回 nan"""
        assert math.isnan(power_ttest(d=0.5, n1=1, n2=30))
        assert math.isnan(power_ttest(d=0.5, n1=30, n2=1))
        assert math.isnan(power_ttest(d=float("nan"), n1=30, n2=30))

    def test_mde_decreases_with_n(self) -> None:
        """样本量增大时，最小可检测效应量应单调递减"""
        mde_small = minimum_detectable_effect(n1=10, n2=10, alpha=0.05, power=0.8)
        mde_large = minimum_detectable_effect(n1=100, n2=100, alpha=0.05, power=0.8)
        assert not math.isnan(mde_small)
        assert not math.isnan(mde_large)
        assert mde_large < mde_small

    def test_mde_boundary(self) -> None:
        """边界：n < 2 → 返回 nan"""
        assert math.isnan(minimum_detectable_effect(n1=1, n2=10))
        assert math.isnan(minimum_detectable_effect(n1=10, n2=1))

    def test_sample_size_for_large_effect(self) -> None:
        """大效应量（d=1.5）所需样本量应较小（< 10）"""
        n = sample_size_for_effect(d=1.5, alpha=0.05, power=0.8, ratio=1.0)
        assert n > 0
        assert n < 10

    def test_sample_size_for_small_effect(self) -> None:
        """小效应量（d=0.2）所需样本量应较大（> 50）"""
        n = sample_size_for_effect(d=0.2, alpha=0.05, power=0.8, ratio=1.0)
        assert n > 50

    def test_sample_size_boundary(self) -> None:
        """边界：d=0 或 nan → 返回 0"""
        assert sample_size_for_effect(d=0.0) == 0
        assert sample_size_for_effect(d=float("nan")) == 0
