"""
统计显著性检验模块
Statistical Significance Testing Module

为策略对比提供统计显著性检验，确保结论科学性。支持：
- 正态性检验（Shapiro-Wilk / D'Agostino K²）
- 方差齐性检验（Levene）
- 两两比较（独立样本 t 检验 / Welch t / Mann-Whitney U）
- 效应量计算（Cohen's d / rank-biserial correlation）
- 多重比较校正（Bonferroni）
- 均值差的 95% 置信区间
- 中文解释文本

典型用法：
    from src.utils.stats_significance import compare_strategies
    results = compare_strategies({"PPO": [2747, 2850, ...], "FCFS": [1462, ...]})
"""

import math
from itertools import combinations
from typing import Any

import numpy as np
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
    boot_improvements: np.ndarray = np.empty(n_bootstrap, dtype=np.float64)

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
