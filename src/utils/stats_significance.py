"""
统计显著性检验模块
Statistical Significance Testing Module

为策略对比提供统计显著性检验，确保结论科学性。支持：
- 正态性检验（Shapiro-Wilk / D'Agostino K²）
- 方差齐性检验（Levene）
- 两两比较（独立样本 t 检验 / Welch t / Mann-Whitney U）
- 效应量计算（Cohen's d / rank-biserial correlation / Cliff's delta）
- 多重比较校正（Bonferroni）
- 均值差的 95% 置信区间
- 统计功效分析（post-hoc power / 所需样本量 / 一站式功效报告）
- 中文解释文本

典型用法：
    from src.utils.stats_significance import compare_strategies
    results = compare_strategies({"PPO": [2747, 2850, ...], "FCFS": [1462, ...]})

统一 Power Analysis 接口（Issue #597）：
    from src.utils.stats_significance import (
        compute_effect_size, compute_statistical_power,
        required_sample_size, power_analysis_report,
    )
    es = compute_effect_size(group_a, group_b)
    power = compute_statistical_power(es["cohens_d"], n1=30, n2=30)
    n = required_sample_size(effect_size=0.5, power=0.8)
    report = power_analysis_report({"PPO": [...], "DQN": [...], "FCFS": [...]})
"""

import math
from itertools import combinations
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy import stats


def normality_test(samples: list[float], alpha: float = 0.05) -> tuple[bool, float, str]:
    """正态性检验

    根据样本量自动选择检验方法：
    - n < 50：Shapiro-Wilk 检验（对小样本更有效）
    - n >= 50：D'Agostino K² 检验（对大样本更稳健）
    - n < 3：样本量过小，保守判为非正态

    Args:
        samples: 样本数据列表
        alpha: 显著性水平（p >= alpha 时接受正态性假设）

    Returns:
        (is_normal, p_value, test_name) 元组：
        - is_normal: 是否通过正态性检验
        - p_value: 检验 p 值
        - test_name: 使用的检验名称
    """
    n = len(samples)
    if n < 3:
        # 样本量过小无法可靠检验正态性，保守判为非正态
        return False, 0.0, "样本量不足(n<3)"
    arr = np.asarray(samples, dtype=float)
    if n < 50:
        result = stats.shapiro(arr)
        test_name = "Shapiro-Wilk"
    else:
        result = stats.normaltest(arr)
        test_name = "D'Agostino K²"
    p_value = float(result.pvalue)
    return (p_value >= alpha), p_value, test_name


def cohen_d(x: list[float], y: list[float]) -> float:
    """计算 Cohen's d 效应量

    公式：d = (mean_x - mean_y) / pooled_std
    其中 pooled_std = sqrt(((n1-1)*s1² + (n2-1)*s2²) / (n1+n2-2))

    Args:
        x: 第一组样本
        y: 第二组样本

    Returns:
        Cohen's d 效应量（正值表示 x 均值高于 y）；方差为零或样本不足时返回 nan
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    n1, n2 = len(arr_x), len(arr_y)
    if n1 < 2 or n2 < 2:
        return float("nan")
    mean_diff = float(arr_x.mean() - arr_y.mean())
    var_x = float(arr_x.var(ddof=1))
    var_y = float(arr_y.var(ddof=1))
    pooled_var = ((n1 - 1) * var_x + (n2 - 1) * var_y) / (n1 + n2 - 2)
    if pooled_var <= 0:
        return float("nan")
    return mean_diff / math.sqrt(pooled_var)


def rank_biserial(x: list[float], y: list[float]) -> float:
    """计算 rank-biserial correlation 效应量（非参数检验的效应量）

    公式：r = (2 * U_x) / (n1 * n2) - 1
    其中 U_x 为 Mann-Whitney U 统计量（针对 x 组）。
    正值表示 x 倾向高于 y，负值表示 x 倾向低于 y，取值范围 [-1, 1]。

    Args:
        x: 第一组样本
        y: 第二组样本

    Returns:
        rank-biserial 相关系数；样本为空时返回 nan
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    n1, n2 = len(arr_x), len(arr_y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    result = stats.mannwhitneyu(arr_x, arr_y, alternative="two-sided")
    u_x = float(result.statistic)
    return (2.0 * u_x) / (n1 * n2) - 1.0


def _mean_diff_ci(
    x: list[float],
    y: list[float],
    equal_var: bool = True,
    confidence: float = 0.95,
) -> tuple[float, float, float]:
    """计算均值差的置信区间

    Args:
        x: 第一组样本
        y: 第二组样本
        equal_var: 是否假设方差齐性（True 用合并方差，False 用 Welch-Satterthwaite）
        confidence: 置信水平（默认 0.95）

    Returns:
        (mean_diff, ci_lower, ci_upper) 元组；样本不足时 CI 为 nan
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    n1, n2 = len(arr_x), len(arr_y)
    mean_diff = float(arr_x.mean() - arr_y.mean())
    if n1 < 2 or n2 < 2:
        return mean_diff, float("nan"), float("nan")

    var_x = float(arr_x.var(ddof=1))
    var_y = float(arr_y.var(ddof=1))

    df: float
    if equal_var:
        pooled_var = ((n1 - 1) * var_x + (n2 - 1) * var_y) / (n1 + n2 - 2)
        se = math.sqrt(pooled_var * (1.0 / n1 + 1.0 / n2))
        df = float(n1 + n2 - 2)
    else:
        se = math.sqrt(var_x / n1 + var_y / n2)
        # Welch-Satterthwaite 自由度
        num = (var_x / n1 + var_y / n2) ** 2
        den = (var_x / n1) ** 2 / (n1 - 1) + (var_y / n2) ** 2 / (n2 - 1)
        df = num / den if den > 0 else float(n1 + n2 - 2)

    alpha_ci = 1.0 - confidence
    t_crit = float(stats.t.ppf(1.0 - alpha_ci / 2.0, df))
    margin = t_crit * se
    return mean_diff, mean_diff - margin, mean_diff + margin


def bootstrap_improvement_ci(
    target: list[float],
    baseline: list[float],
    confidence: float = 0.95,
    n_bootstrap: int = 10000,
    seed: int | None = 42,
) -> tuple[float, float, float]:
    """计算策略相对于基线的提升百分比的 Bootstrap 95% 置信区间。

    提升百分比定义：(mean(target) - mean(baseline)) / |mean(baseline)| * 100

    使用百分位 Bootstrap 法：有放回地从两组独立抽样，计算每次的提升百分比，
    然后取对应置信水平的百分位数作为 CI 上下界。

    Args:
        target: 目标策略奖励列表
        baseline: 基线策略奖励列表
        confidence: 置信水平（默认 0.95）
        n_bootstrap: Bootstrap 重抽样次数（默认 10000）
        seed: 随机种子（默认 42，保证可复现）

    Returns:
        (improvement_pct, ci_lower, ci_upper) 元组；基线均值为 0 或样本不足时 CI 为 nan
    """
    arr_target = np.asarray(target, dtype=float)
    arr_baseline = np.asarray(baseline, dtype=float)
    n1, n2 = len(arr_target), len(arr_baseline)

    baseline_mean = float(np.mean(arr_baseline))
    target_mean = float(np.mean(arr_target))
    if n1 < 2 or n2 < 2 or baseline_mean == 0:
        if baseline_mean == 0:
            improvement = (
                float("inf") if target_mean > 0 else (float("-inf") if target_mean < 0 else 0.0)
            )
        else:
            improvement = (
                (target_mean - baseline_mean) / abs(baseline_mean) * 100
                if baseline_mean != 0
                else 0.0
            )
        return improvement, float("nan"), float("nan")

    improvement = (target_mean - baseline_mean) / abs(baseline_mean) * 100

    rng = np.random.default_rng(seed)
    boot_improvements: NDArray[Any] = np.empty(n_bootstrap, dtype=np.float64)

    for i in range(n_bootstrap):
        idx1 = rng.integers(0, n1, size=n1)
        idx2 = rng.integers(0, n2, size=n2)
        boot_target_mean = float(np.mean(arr_target[idx1]))
        boot_baseline_mean = float(np.mean(arr_baseline[idx2]))
        if boot_baseline_mean == 0:
            boot_improvements[i] = float("nan")
        else:
            boot_improvements[i] = (
                (boot_target_mean - boot_baseline_mean) / abs(boot_baseline_mean) * 100
            )

    valid = boot_improvements[~np.isnan(boot_improvements)]
    if len(valid) < 10:
        return improvement, float("nan"), float("nan")

    alpha_ci = 1.0 - confidence
    ci_lower = float(np.percentile(valid, alpha_ci / 2.0 * 100))
    ci_upper = float(np.percentile(valid, (1.0 - alpha_ci / 2.0) * 100))
    return improvement, ci_lower, ci_upper


def _effect_level(effect: float, effect_type: str) -> str:
    """根据效应量类型与大小判定等级中文描述

    Args:
        effect: 效应量数值
        effect_type: 效应量类型（"Cohen's d" 或 "rank-biserial correlation"）

    Returns:
        等级描述（无法计算 / 可忽略 / 小效应 / 中效应 / 大效应）
    """
    if math.isnan(effect):
        return "无法计算"
    abs_e = abs(effect)
    if effect_type == "Cohen's d":
        if abs_e < 0.2:
            return "可忽略"
        elif abs_e < 0.5:
            return "小效应"
        elif abs_e < 0.8:
            return "中效应"
        else:
            return "大效应"
    # rank-biserial correlation 等级阈值
    if abs_e < 0.1:
        return "可忽略"
    elif abs_e < 0.3:
        return "小效应"
    elif abs_e < 0.5:
        return "中效应"
    else:
        return "大效应"


def _build_interpretation(
    name_a: str,
    name_b: str,
    test_name: str,
    statistic: float,
    p_value: float,
    significant: bool,
    effect: float,
    effect_type: str,
    mean_diff: float,
    ci_lo: float,
    ci_hi: float,
    adjusted_alpha: float,
    n_comparisons: int,
) -> str:
    """生成中文解释文本

    Args:
        name_a: 策略 A 名称
        name_b: 策略 B 名称
        test_name: 使用的检验名称
        statistic: 检验统计量
        p_value: 检验 p 值
        significant: 经 Bonferroni 校正后是否显著
        effect: 效应量数值
        effect_type: 效应量类型
        mean_diff: 均值差（A - B）
        ci_lo: 95% CI 下界
        ci_hi: 95% CI 上界
        adjusted_alpha: Bonferroni 校正后的 α
        n_comparisons: 比较总次数

    Returns:
        中文解释字符串
    """
    if mean_diff > 0:
        direction = "高于"
    elif mean_diff < 0:
        direction = "低于"
    else:
        direction = "等于"
    diff_abs = abs(mean_diff)
    sig_text = "显著" if significant else "不显著"
    effect_level = _effect_level(effect, effect_type)
    return (
        f"使用 {test_name} 比较 {name_a} 与 {name_b}："
        f"{name_a} 平均奖励{direction}{name_b} {diff_abs:.2f}"
        f"（95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]）；"
        f"统计量={statistic:.4f}，p={p_value:.4g}。"
        f"经 Bonferroni 校正（{n_comparisons} 次比较，校正 α={adjusted_alpha:.4f}），"
        f"差异{sig_text}。"
        f"效应量 {effect_type}={effect:.4f}（{effect_level}）。"
    )


def compare_strategies(
    data: dict[str, list[float]], alpha: float = 0.05
) -> dict[str, dict[str, Any]]:
    """策略两两统计显著性比较主函数

    对所有策略两两组合执行统计检验，自动选择合适的检验方法：
    - 两组均正态且方差齐 → 独立样本 t 检验
    - 两组均正态但方差不齐 → Welch t 检验
    - 任一组非正态 → Mann-Whitney U 检验

    同时计算效应量、均值差 95% 置信区间，并使用 Bonferroni 校正多重比较。

    Args:
        data: ``{策略名: [多次运行的奖励列表]}``，例如
            ``{"PPO": [2747, 2850, ...], "FCFS": [1462, ...]}``
        alpha: 显著性水平（默认 0.05）

    Returns:
        结果字典：``{对比名: {字段}}``。对比名格式为 ``"策略A vs 策略B"``。
        每个对比包含字段：test / statistic / p_value / significant /
        effect_size / effect_size_type / mean_diff / ci_lower / ci_upper /
        bonferroni_alpha / n_comparisons / normality_a / normality_b /
        interpretation。
        空输入或单策略输入返回空字典。
    """
    # 边界：空输入或单策略
    if not data or len(data) < 2:
        return {}

    strategies = list(data.keys())
    pairs = list(combinations(strategies, 2))
    n_comparisons = len(pairs)
    adjusted_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha

    results: dict[str, dict[str, Any]] = {}

    for name_a, name_b in pairs:
        samples_a = list(data[name_a])
        samples_b = list(data[name_b])
        pair_key = f"{name_a} vs {name_b}"

        # 边界：样本不足，无法执行检验
        if len(samples_a) < 2 or len(samples_b) < 2:
            results[pair_key] = {
                "test": "无法检验",
                "statistic": float("nan"),
                "p_value": float("nan"),
                "significant": False,
                "effect_size": float("nan"),
                "effect_size_type": "N/A",
                "mean_diff": float("nan"),
                "ci_lower": float("nan"),
                "ci_upper": float("nan"),
                "bonferroni_alpha": adjusted_alpha,
                "n_comparisons": n_comparisons,
                "normality_a": {"is_normal": False, "p_value": 0.0, "test": "样本不足"},
                "normality_b": {"is_normal": False, "p_value": 0.0, "test": "样本不足"},
                "interpretation": (
                    f"样本量不足（{name_a}={len(samples_a)}, "
                    f"{name_b}={len(samples_b)}），至少需要 2 个样本才能执行统计检验。"
                ),
            }
            continue

        # 正态性检验
        normal_a, p_norm_a, test_norm_a = normality_test(samples_a, alpha)
        normal_b, p_norm_b, test_norm_b = normality_test(samples_b, alpha)
        both_normal = normal_a and normal_b

        arr_a = np.asarray(samples_a, dtype=float)
        arr_b = np.asarray(samples_b, dtype=float)

        if both_normal:
            # 方差齐性检验
            lev = stats.levene(arr_a, arr_b)
            equal_var = bool(lev.pvalue >= alpha)
            if equal_var:
                # 独立样本 t 检验（方差齐）
                res = stats.ttest_ind(arr_a, arr_b, equal_var=True)
                test_name = "独立样本 t 检验"
                effect = cohen_d(samples_a, samples_b)
                effect_type = "Cohen's d"
                mean_diff, ci_lo, ci_hi = _mean_diff_ci(samples_a, samples_b, equal_var=True)
            else:
                # Welch t 检验（方差不齐）
                res = stats.ttest_ind(arr_a, arr_b, equal_var=False)
                test_name = "Welch t 检验"
                effect = cohen_d(samples_a, samples_b)
                effect_type = "Cohen's d"
                mean_diff, ci_lo, ci_hi = _mean_diff_ci(samples_a, samples_b, equal_var=False)
        else:
            # 非正态 → Mann-Whitney U 检验
            res = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
            test_name = "Mann-Whitney U 检验"
            effect = rank_biserial(samples_a, samples_b)
            effect_type = "rank-biserial correlation"
            # 非参数检验仍报告均值差 CI 作为参考
            mean_diff, ci_lo, ci_hi = _mean_diff_ci(samples_a, samples_b, equal_var=True)

        p_value = float(res.pvalue)
        statistic = float(res.statistic)
        significant = bool(p_value < adjusted_alpha)

        imp_pct, imp_ci_lo, imp_ci_hi = bootstrap_improvement_ci(
            samples_a, samples_b, confidence=0.95
        )

        interpretation = _build_interpretation(
            name_a=name_a,
            name_b=name_b,
            test_name=test_name,
            statistic=statistic,
            p_value=p_value,
            significant=significant,
            effect=effect,
            effect_type=effect_type,
            mean_diff=mean_diff,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            adjusted_alpha=adjusted_alpha,
            n_comparisons=n_comparisons,
        )

        results[pair_key] = {
            "test": test_name,
            "statistic": statistic,
            "p_value": p_value,
            "significant": significant,
            "effect_size": effect,
            "effect_size_type": effect_type,
            "mean_diff": mean_diff,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "improvement_pct": imp_pct,
            "improvement_pct_ci_lower": imp_ci_lo,
            "improvement_pct_ci_upper": imp_ci_hi,
            "bonferroni_alpha": adjusted_alpha,
            "n_comparisons": n_comparisons,
            "normality_a": {"is_normal": normal_a, "p_value": p_norm_a, "test": test_norm_a},
            "normality_b": {"is_normal": normal_b, "p_value": p_norm_b, "test": test_norm_b},
            "interpretation": interpretation,
        }

    return results


def power_ttest(d: float, n1: int, n2: int, alpha: float = 0.05) -> float:
    """计算两样本 t 检验的事后检验力（post-hoc power）

    基于非中心 t 分布（non-central t distribution）计算在给定效应量 Cohen's d、
    样本量 n1/n2 与显著性水平 α 下的双侧 t 检验检验力。

    Args:
        d: Cohen's d 效应量（取绝对值，符号不影响检验力）
        n1: 第一组样本量
        n2: 第二组样本量
        alpha: 显著性水平（默认 0.05）

    Returns:
        检验力（0-1 之间）；样本不足（n<2）或 d 为 NaN 时返回 nan
    """
    if n1 < 2 or n2 < 2 or math.isnan(d):
        return float("nan")
    # 非中心参数 ncp = |d| * sqrt(n1*n2 / (n1+n2))
    ncp = abs(d) * math.sqrt(n1 * n2 / (n1 + n2))
    df = n1 + n2 - 2
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    # 双侧检验力 = P(T > t_crit | ncp) + P(T < -t_crit | ncp)
    # 注意：当 ncp 较大时 scipy.stats.nct.cdf 在 -t_crit 处可能返回 nan（数值下溢），
    # 用 survival function (sf = 1 - cdf) 与 cdf 分别计算并容错。
    right_tail = float(stats.nct.sf(t_crit, df, ncp))
    left_tail_cdf = stats.nct.cdf(-t_crit, df, ncp)
    # ncp 远大于 -t_crit 时左尾概率可忽略（scipy 在数值下溢时返回 nan）
    left_tail = 0.0 if math.isnan(left_tail_cdf) else float(left_tail_cdf)
    power = right_tail + left_tail
    # 数值精度保护
    if power > 1.0:
        power = 1.0
    if power < 0.0:
        power = 0.0
    return power


def minimum_detectable_effect(n1: int, n2: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """计算给定样本量下的最小可检测效应量（MDES, Cohen's d）

    使用经典近似公式：MDES ≈ (t_{α/2} + t_{power}) * sqrt((n1+n2)/(n1*n2))

    Args:
        n1: 第一组样本量
        n2: 第二组样本量
        alpha: 显著性水平（默认 0.05）
        power: 目标检验力（默认 0.8）

    Returns:
        在指定检验力下可检测的最小 Cohen's d；样本不足时返回 nan
    """
    if n1 < 2 or n2 < 2:
        return float("nan")
    df = n1 + n2 - 2
    t_alpha = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    t_beta = float(stats.t.ppf(power, df))
    mde = (t_alpha + t_beta) * math.sqrt((n1 + n2) / (n1 * n2))
    return mde


def sample_size_for_effect(
    d: float, alpha: float = 0.05, power: float = 0.8, ratio: float = 1.0
) -> int:
    """计算检测指定效应量所需的每组样本量

    通过迭代求解：找到最小的 n1，使得 power_ttest(d, n1, n2=ratio*n1, alpha) >= power。

    Args:
        d: 目标 Cohen's d 效应量（取绝对值）
        alpha: 显著性水平（默认 0.05）
        power: 目标检验力（默认 0.8）
        ratio: n2/n1 的比例（默认 1.0，即等样本量）

    Returns:
        第一组所需样本量 n1；d 为 0/NaN 或无法达到目标检验力时返回 0
    """
    if math.isnan(d) or d == 0:
        return 0
    for n in range(2, 100000):
        n1, n2 = n, int(n * ratio)
        if n2 < 2:
            continue
        p = power_ttest(d, n1, n2, alpha)
        if not math.isnan(p) and p >= power:
            return n1
    return 0


# ---------------------------------------------------------------------------
# Issue #597: 统一 Power Analysis 接口
# ---------------------------------------------------------------------------

# Mann-Whitney U 检验相对于 t 检验的渐近相对效率（ARE ≈ 3/π ≈ 0.955）
_MANN_WHITNEY_ARE: float = 0.955


def power_analysis(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    test_type: str = "mann_whitney",
    ratio: float = 1.0,
) -> dict[str, Any]:
    """统一功效分析入口函数（Issue #597）

    根据效应量、显著性水平和目标功效，计算所需样本量。
    内部复用已有的 ``sample_size_for_effect`` 函数，并支持 Mann-Whitney 校正。

    Args:
        effect_size: 目标效应量（Cohen's d），取绝对值
        alpha      : 显著性水平（默认 0.05）
        power      : 目标功效（默认 0.80）
        test_type  : 检验类型，``"mann_whitney"``（默认）或 ``"t_test"``
        ratio      : n2/n1 的比例（默认 1.0，即等样本量）

    Returns:
        包含以下键的字典：
        - ``sample_size_per_group`` : 每组所需样本量
        - ``total_sample_size``     : 总样本量（两组之和）
        - ``effect_size``           : 输入的效应量
        - ``alpha``                 : 显著性水平
        - ``power``                 : 目标功效
        - ``test_type``             : 检验类型
        - ``ratio``                 : n2/n1 比例
        - ``are_correction``        : 是否应用了 ARE 校正

    Raises:
        ValueError: 当参数不合法时（effect_size <= 0, alpha/power 不在 (0,1) 区间）
    """
    if effect_size == 0:
        raise ValueError("effect_size 不能为 0")
    if not 0 < alpha < 1:
        raise ValueError(f"alpha 必须在 (0, 1) 区间，当前: {alpha}")
    if not 0 < power < 1:
        raise ValueError(f"power 必须在 (0, 1) 区间，当前: {power}")
    if test_type not in ("mann_whitney", "t_test"):
        raise ValueError(f"test_type 必须为 'mann_whitney' 或 't_test'，当前: {test_type}")

    # 基于 t 检验计算基础样本量
    n_t = sample_size_for_effect(abs(effect_size), alpha, power, ratio)

    # Mann-Whitney 检验需要更多样本（ARE ≈ 0.955）
    are_corrected = False
    n_final = n_t
    if test_type == "mann_whitney" and n_t > 0:
        n_final = int(np.ceil(n_t / _MANN_WHITNEY_ARE))
        are_corrected = True

    n2 = int(n_final * ratio)
    total = n_final + n2

    return {
        "sample_size_per_group": n_final,
        "total_sample_size": total,
        "effect_size": abs(effect_size),
        "alpha": alpha,
        "power": power,
        "test_type": test_type,
        "ratio": ratio,
        "are_correction": are_corrected,
    }


def generate_preregistration(
    effect_size: float,
    alpha: float = 0.05,
    power: float = 0.80,
    test_type: str = "mann_whitney",
    ratio: float = 1.0,
    experiment_name: str = "",
    hypotheses: list[str] | None = None,
    strategies: list[str] | None = None,
) -> dict[str, Any]:
    """生成实验预注册 JSON（Issue #597）

    在实验执行前生成预注册文档，明确实验设计、假设和样本量规划，
    防止 p-hacking 和选择性报告。

    Args:
        effect_size    : 预期效应量（Cohen's d）
        alpha          : 显著性水平（默认 0.05）
        power          : 目标功效（默认 0.80）
        test_type      : 检验类型（默认 ``"mann_whitney"``）
        ratio          : n2/n1 比例（默认 1.0）
        experiment_name: 实验名称
        hypotheses     : 假设列表
        strategies     : 参与比较的策略名称列表

    Returns:
        可 JSON 序列化的预注册字典，包含：
        - ``experiment_name``  : 实验名称
        - ``timestamp``        : 生成时间（ISO 格式）
        - ``hypotheses``       : 假设列表
        - ``strategies``       : 策略列表
        - ``design``           : 实验设计参数
        - ``sample_size_plan`` : 样本量规划（来自 power_analysis）
        - ``multiple_comparison``: 多重比较校正方法
    """
    from datetime import datetime

    pa = power_analysis(
        effect_size=effect_size,
        alpha=alpha,
        power=power,
        test_type=test_type,
        ratio=ratio,
    )

    n_strategies = len(strategies) if strategies else 2
    n_comparisons = n_strategies * (n_strategies - 1) // 2 if n_strategies > 1 else 1
    bonferroni_alpha = alpha / n_comparisons if n_comparisons > 0 else alpha

    return {
        "experiment_name": experiment_name or "未命名实验",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "hypotheses": hypotheses or ["H1: 策略间奖励均值存在显著差异"],
        "strategies": strategies or ["策略A", "策略B"],
        "design": {
            "effect_size": abs(effect_size),
            "alpha": alpha,
            "power": power,
            "test_type": test_type,
            "ratio": ratio,
        },
        "sample_size_plan": pa,
        "multiple_comparison": {
            "method": "Bonferroni",
            "n_comparisons": n_comparisons,
            "corrected_alpha": bonferroni_alpha,
        },
    }


def plot_power_curve(
    effect_sizes: list[float] | None = None,
    n_per_group: int = 30,
    alpha: float = 0.05,
    test_type: str = "t_test",
) -> dict[str, Any]:
    """生成功效曲线数据（Issue #597）

    计算不同效应量下的检验功效，返回可用于绘图的曲线数据。
    功效曲线帮助研究者直观理解效应量与检验力之间的关系。

    Args:
        effect_sizes  : 效应量列表，为 None 时自动生成 [0.1, 0.2, ..., 2.0]
        n_per_group   : 每组样本量（默认 30）
        alpha         : 显著性水平（默认 0.05）
        test_type     : 检验类型，``"t_test"``（默认）或 ``"mann_whitney"``

    Returns:
        包含以下键的字典：
        - ``effect_sizes`` : 效应量数组
        - ``powers``       : 对应功效数组
        - ``n_per_group``  : 每组样本量
        - ``alpha``        : 显著性水平
        - ``test_type``    : 检验类型
        - ``threshold_80`` : 0.8 功效阈值对应的效应量（插值）
        - ``threshold_90`` : 0.9 功效阈值对应的效应量（插值）
    """
    if effect_sizes is None:
        effect_sizes = [round(0.1 * i, 1) for i in range(1, 21)]  # 0.1 to 2.0

    n2 = n_per_group
    powers: list[float] = []
    for d in effect_sizes:
        if test_type == "mann_whitney":
            # Mann-Whitney 功效近似：使用 ARE 校正后的有效样本量
            n_eff = int(np.ceil(n_per_group * _MANN_WHITNEY_ARE))
            p = power_ttest(d, n_eff, n_eff, alpha)
        else:
            p = power_ttest(d, n_per_group, n2, alpha)
        powers.append(p)

    # 插值计算 0.8 和 0.9 功效阈值对应的效应量
    threshold_80 = _interpolate_threshold(effect_sizes, powers, 0.80)
    threshold_90 = _interpolate_threshold(effect_sizes, powers, 0.90)

    return {
        "effect_sizes": effect_sizes,
        "powers": powers,
        "n_per_group": n_per_group,
        "alpha": alpha,
        "test_type": test_type,
        "threshold_80": threshold_80,
        "threshold_90": threshold_90,
    }


def cliffs_delta(x: list[float], y: list[float]) -> float:
    """计算 Cliff's delta 效应量（非参数）

    Cliff's delta 衡量两组数据重叠程度的非参数效应量，不依赖正态性或方差齐性假设。
    公式：δ = (#(x_i > y_j) - #(x_i < y_j)) / (n1 * n2)
    取值范围 [-1, 1]，正值表示 x 倾向大于 y。

    解释阈值（Romano et al., 2006）：
        |δ| < 0.147 → 可忽略
        |δ| < 0.33  → 小效应
        |δ| < 0.474 → 中效应
        否则        → 大效应

    Args:
        x: 第一组样本
        y: 第二组样本

    Returns:
        Cliff's delta 值；任一组为空时返回 nan
    """
    arr_x = np.asarray(x, dtype=float)
    arr_y = np.asarray(y, dtype=float)
    n1, n2 = len(arr_x), len(arr_y)
    if n1 == 0 or n2 == 0:
        return float("nan")
    # Issue #689: 原实现用 n1×n2 差分矩阵（arr_x[:,None]-arr_y[None,:]），
    # n1=n2=10000 时约 800MB、n=20000 时约 3.2GB，大样本 OOM。
    # 改为排序 + searchsorted 的 O((n1+n2) log n2) 内存高效算法：
    #   对每个 x_i，#(y_j < x_i) = searchsorted(y_sorted, x_i, 'left')
    #                  #(y_j > x_i) = n2 - searchsorted(y_sorted, x_i, 'right')
    #   平局（y_j == x_i）既不计入 greater 也不计入 less，与原定义一致。
    arr_y_sorted = np.sort(arr_y)
    lo = np.searchsorted(arr_y_sorted, arr_x, side="left")  # #(y_j < x_i) = #(x_i > y_j)
    hi = np.searchsorted(arr_y_sorted, arr_x, side="right")  # #(y_j <= x_i)
    n_greater = float(lo.sum())
    n_less = float((n2 - hi).sum())  # #(y_j > x_i) = #(x_i < y_j)
    return (n_greater - n_less) / (n1 * n2)


def _cliffs_delta_level(d: float) -> str:
    """Cliff's delta 效应量等级判定"""
    if math.isnan(d):
        return "无法计算"
    abs_d = abs(d)
    if abs_d < 0.147:
        return "可忽略"
    elif abs_d < 0.33:
        return "小效应"
    elif abs_d < 0.474:
        return "中效应"
    else:
        return "大效应"


def _resolve_test_type(test: str) -> str:
    """将用户输入的 test 参数归一化为内部标准名称。

    支持别名：'mannwhitney' / 'mann_whitney' / 'mw' → 'mannwhitney'
              'ttest' / 't_test' / 't' → 'ttest'

    Raises:
        ValueError: 不认识的 test 名称
    """
    normalized = test.lower().replace("_", "").replace("-", "")
    if normalized in ("mannwhitney", "mw", "mannwhitneyu"):
        return "mannwhitney"
    if normalized in ("ttest", "t", "studentt"):
        return "ttest"
    raise ValueError(f"不支持的 test 类型 '{test}'，请使用 'mannwhitney' 或 'ttest'")


def compute_effect_size(group1: list[float], group2: list[float]) -> dict[str, Any]:
    """统一效应量计算入口

    一次性计算两组之间的三种效应量：
    - Cohen's d（参数，基于均值差/合并标准差）
    - rank-biserial correlation（非参数，基于 Mann-Whitney U）
    - Cliff's delta（非参数，基于优势度，不依赖正态性）

    同时返回每种效应量的中文等级解释，并自动推荐适合的数据分布类型。

    Args:
        group1: 第一组样本数据
        group2: 第二组样本数据

    Returns:
        包含以下键的字典：
        - ``cohens_d``           : Cohen's d 值
        - ``cohens_d_level``     : Cohen's d 等级（可忽略/小/中/大）
        - ``rank_biserial``      : rank-biserial 值
        - ``rank_biserial_level``: rank-biserial 等级
        - ``cliffs_delta``       : Cliff's delta 值
        - ``cliffs_delta_level`` : Cliff's delta 等级
        - ``n1``, ``n2``         : 两组样本量
        - ``recommended``        : 推荐使用的效应量类型
          （正态且方差齐 → Cohen's d；否则 → Cliff's delta）
        - ``normality``          : 两组正态性检验结果
    """
    d = cohen_d(group1, group2)
    rb = rank_biserial(group1, group2)
    cd = cliffs_delta(group1, group2)

    normal1, p1, test1 = normality_test(group1)
    normal2, p2, test2 = normality_test(group2)
    both_normal = normal1 and normal2

    arr1 = np.asarray(group1, dtype=float)
    arr2 = np.asarray(group2, dtype=float)
    if both_normal and len(group1) >= 3 and len(group2) >= 3:
        try:
            lev = stats.levene(arr1, arr2)
            equal_var = bool(lev.pvalue >= 0.05)
        except Exception:  # noqa: BLE001
            equal_var = False
        recommended = "cohens_d" if equal_var else "cliffs_delta"
    else:
        recommended = "cliffs_delta"

    return {
        "cohens_d": d,
        "cohens_d_level": _effect_level(d, "Cohen's d"),
        "rank_biserial": rb,
        "rank_biserial_level": _effect_level(rb, "rank-biserial correlation"),
        "cliffs_delta": cd,
        "cliffs_delta_level": _cliffs_delta_level(cd),
        "n1": len(group1),
        "n2": len(group2),
        "recommended": recommended,
        "normality": {
            "group1": {"is_normal": normal1, "p_value": p1, "test": test1},
            "group2": {"is_normal": normal2, "p_value": p2, "test": test2},
        },
    }


def _power_mannwhitney(effect_size: float, n1: int, n2: int, alpha: float) -> float:
    """Mann-Whitney U 检验的统计功效近似计算

    使用 ARE（渐近相对效率）校正法：将 Mann-Whitney 功效近似为
    有效样本量 n_eff = n * ARE（ARE ≈ 3/π ≈ 0.955）下的 t 检验功效。

    Args:
        effect_size: Cohen's d 效应量（取绝对值）
        n1: 第一组样本量
        n2: 第二组样本量
        alpha: 显著性水平

    Returns:
        近似功效值 [0, 1]
    """
    n1_eff = max(2, int(np.ceil(n1 * _MANN_WHITNEY_ARE)))
    n2_eff = max(2, int(np.ceil(n2 * _MANN_WHITNEY_ARE)))
    return power_ttest(abs(effect_size), n1_eff, n2_eff, alpha)


def compute_statistical_power(
    effect_size: float,
    n1: int,
    n2: int | None = None,
    alpha: float = 0.05,
    test: str = "mannwhitney",
) -> float:
    """计算给定效应量和样本量下的统计功效（post-hoc power）

    Args:
        effect_size: 效应量（Cohen's d 尺度，取绝对值）
        n1: 第一组样本量
        n2: 第二组样本量（为 None 时默认与 n1 相等）
        alpha: 显著性水平（默认 0.05）
        test: 检验类型，``'mannwhitney'``（默认）或 ``'ttest'``

    Returns:
        统计功效值（0-1 之间）；参数非法或样本不足时返回 nan

    Raises:
        ValueError: test 类型不认识或 alpha 不在 (0,1)
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha 必须在 (0,1) 区间，当前: {alpha}")
    test_type = _resolve_test_type(test)
    if n2 is None:
        n2 = n1
    if n1 < 2 or n2 < 2 or math.isnan(effect_size):
        return float("nan")
    es = abs(effect_size)
    if es == 0:
        return alpha  # 效应量为 0 时，功效等于 Type I error rate
    if test_type == "ttest":
        return power_ttest(es, n1, n2, alpha)
    return _power_mannwhitney(es, n1, n2, alpha)


def required_sample_size(
    effect_size: float,
    power: float = 0.8,
    alpha: float = 0.05,
    test: str = "mannwhitney",
) -> int:
    """计算达到指定功效所需的每组样本量（等样本量设计）

    通过迭代搜索找到最小 n，使得 ``compute_statistical_power(effect_size, n, n, alpha, test) >= power``。

    Args:
        effect_size: 目标效应量（Cohen's d 尺度，取绝对值）
        power: 目标统计功效（默认 0.8）
        alpha: 显著性水平（默认 0.05）
        test: 检验类型，``'mannwhitney'``（默认）或 ``'ttest'``

    Returns:
        每组所需样本量（整数）；效应量为 0/NaN 或无法在合理范围内达到时返回 0

    Raises:
        ValueError: alpha/power 不在 (0,1) 或 test 类型不认识
    """
    if not 0 < alpha < 1:
        raise ValueError(f"alpha 必须在 (0,1) 区间，当前: {alpha}")
    if not 0 < power < 1:
        raise ValueError(f"power 必须在 (0,1) 区间，当前: {power}")
    test_type = _resolve_test_type(test)
    if math.isnan(effect_size) or effect_size == 0:
        return 0

    es = abs(effect_size)
    # 从已有 t 检验样本量估算起点，加速迭代
    n_start = sample_size_for_effect(es, alpha, power, ratio=1.0)
    n_start = 2 if n_start == 0 else max(2, n_start - 5)

    for n in range(n_start, 200000):
        p = compute_statistical_power(es, n, n, alpha, test_type)
        if not math.isnan(p) and p >= power:
            return n
    return 0


def power_analysis_report(
    groups: dict[str, list[float]],
    alpha: float = 0.05,
    target_power: float = 0.8,
) -> dict[str, Any]:
    """一站式统计功效分析报告

    对多组策略数据自动执行两两功效分析，包括：
    - 每组基本描述统计（n, mean, std）
    - 两两效应量（Cohen's d / rank-biserial / Cliff's delta）
    - 基于当前样本量的事后功效
    - 达到目标功效所需的样本量建议
    - 功效充足/不足标记
    - 总体摘要

    Args:
        groups: ``{组名: [样本数据列表]}`` 字典
        alpha: 显著性水平（默认 0.05）
        target_power: 目标统计功效（默认 0.8）

    Returns:
        完整的功效分析报告字典，包含：
        - ``summary``: 总体摘要（总组数、充足功效对数、不足对数等）
        - ``group_stats``: 每组描述统计
        - ``pairwise``: 两两对比详情（键为 "A vs B" 格式）
        - ``alpha``, ``target_power``: 使用的参数
    """
    if not groups or len(groups) < 2:
        return {
            "summary": {
                "n_groups": len(groups) if groups else 0,
                "n_pairs": 0,
                "sufficient_power_pairs": 0,
                "insufficient_power_pairs": 0,
                "message": "至少需要 2 组数据才能进行两两功效分析",
            },
            "group_stats": {},
            "pairwise": {},
            "alpha": alpha,
            "target_power": target_power,
        }

    # ---- 每组描述统计 ----
    group_stats: dict[str, dict[str, Any]] = {}
    for name, samples in groups.items():
        arr = np.asarray(samples, dtype=float)
        n = len(arr)
        group_stats[name] = {
            "n": n,
            "mean": float(np.mean(arr)) if n > 0 else float("nan"),
            "std": float(np.std(arr, ddof=1)) if n >= 2 else float("nan"),
            "min": float(np.min(arr)) if n > 0 else float("nan"),
            "max": float(np.max(arr)) if n > 0 else float("nan"),
        }

    # ---- 两两功效分析 ----
    names = list(groups.keys())
    pairs = list(combinations(names, 2))
    pairwise: dict[str, dict[str, Any]] = {}
    sufficient_count = 0
    insufficient_count = 0

    for name_a, name_b in pairs:
        samples_a = list(groups[name_a])
        samples_b = list(groups[name_b])
        pair_key = f"{name_a} vs {name_b}"

        n_a, n_b = len(samples_a), len(samples_b)
        es = compute_effect_size(samples_a, samples_b)

        # 根据数据分布选择推荐的检验类型
        rec = es["recommended"]
        test_type = "ttest" if rec == "cohens_d" else "mannwhitney"

        d_abs = abs(es["cohens_d"])
        d_valid = not math.isnan(es["cohens_d"])
        d_nonzero = d_valid and es["cohens_d"] != 0

        current_power_ttest = (
            compute_statistical_power(d_abs, n_a, n_b, alpha, "ttest") if d_valid else float("nan")
        )

        current_power_mw = (
            compute_statistical_power(d_abs, n_a, n_b, alpha, "mannwhitney")
            if d_valid
            else float("nan")
        )

        # 所需样本量（等样本量）
        req_n_ttest = required_sample_size(d_abs, target_power, alpha, "ttest") if d_nonzero else 0

        req_n_mw = (
            required_sample_size(d_abs, target_power, alpha, "mannwhitney") if d_nonzero else 0
        )

        # 判断当前功效是否充足（使用推荐检验类型的功效）
        current_p = current_power_ttest if test_type == "ttest" else current_power_mw
        is_sufficient = (not math.isnan(current_p)) and current_p >= target_power
        if is_sufficient:
            sufficient_count += 1
        else:
            insufficient_count += 1

        # 功效解释
        if math.isnan(current_p):
            power_msg = "无法计算（样本不足）"
        elif current_p < 0.5:
            power_msg = "功效严重不足"
        elif current_p < target_power:
            power_msg = f"功效不足（<{target_power:.0%}）"
        elif current_p < 0.9:
            power_msg = "功效充足"
        else:
            power_msg = "功效很高"

        pairwise[pair_key] = {
            "n_a": n_a,
            "n_b": n_b,
            "effect_sizes": {
                "cohens_d": es["cohens_d"],
                "cohens_d_level": es["cohens_d_level"],
                "rank_biserial": es["rank_biserial"],
                "rank_biserial_level": es["rank_biserial_level"],
                "cliffs_delta": es["cliffs_delta"],
                "cliffs_delta_level": es["cliffs_delta_level"],
            },
            "recommended_effect_size": rec,
            "recommended_test": test_type,
            "current_power": {
                "ttest": current_power_ttest,
                "mannwhitney": current_power_mw,
            },
            "current_power_recommended": current_p,
            "power_sufficient": is_sufficient,
            "required_n_per_group": {
                "ttest": req_n_ttest,
                "mannwhitney": req_n_mw,
            },
            "required_n_recommended": req_n_ttest if test_type == "ttest" else req_n_mw,
            "power_interpretation": power_msg,
            "normality": es["normality"],
        }

    n_pairs = len(pairs)
    all_sufficient = sufficient_count == n_pairs
    if all_sufficient:
        summary_msg = f"所有 {n_pairs} 对比较的统计功效均达到 {target_power:.0%}，当前样本量充足。"
    elif sufficient_count > 0:
        summary_msg = (
            f"{sufficient_count}/{n_pairs} 对比较功效充足，"
            f"{insufficient_count} 对不足。建议对功效不足的对比增加样本量。"
        )
    else:
        summary_msg = (
            f"所有 {n_pairs} 对比较功效均不足（<{target_power:.0%}）。"
            "建议增加样本量以获得可靠的统计结论。"
        )

    summary: dict[str, Any] = {
        "n_groups": len(names),
        "n_pairs": n_pairs,
        "sufficient_power_pairs": sufficient_count,
        "insufficient_power_pairs": insufficient_count,
        "all_pairs_sufficient": all_sufficient,
        "alpha": alpha,
        "target_power": target_power,
        "message": summary_msg,
    }

    return {
        "summary": summary,
        "group_stats": group_stats,
        "pairwise": pairwise,
        "alpha": alpha,
        "target_power": target_power,
    }


def _interpolate_threshold(xs: list[float], ys: list[float], threshold: float) -> float | None:
    """线性插值找到 y 达到 threshold 时的 x 值。

    Args:
        xs       : x 值列表（升序）
        ys       : y 值列表
        threshold: 目标 y 值

    Returns:
        插值得到的 x 值；如果所有 y 都低于 threshold 则返回 None
    """
    if len(xs) != len(ys) or len(xs) < 2:
        return None

    for i in range(len(ys) - 1):
        y1, y2 = ys[i], ys[i + 1]
        x1, x2 = xs[i], xs[i + 1]
        if y1 >= threshold:
            return x1
        if y1 < threshold <= y2:
            # 线性插值: x = x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)
            if y2 - y1 == 0:
                return x1
            return x1 + (threshold - y1) * (x2 - x1) / (y2 - y1)

    # 检查最后一个点
    if ys[-1] >= threshold:
        return xs[-1]
    return None
