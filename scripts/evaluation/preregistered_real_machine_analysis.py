#!/usr/bin/env python
"""
预注册真机闭环实验分析脚本（Pre-Registered Analysis）

遵循 results/reports/real_machine_preregistration.md 中定义的分析计划：
1. 按 §5.4 预定义标准执行数据清洗
2. 按 §6 分析计划执行两两比较
3. 以效应量（Cohen's d）+ 95% CI 为主要决策依据
4. p 值为辅助参考，禁止追 p 值
5. 按 §9 报告模板生成标准报告

输入 JSON 格式：
    {
        "status_only": [reward1, reward2, ...],     # C1 基线
        "result_aware": [reward1, reward2, ...],     # C2 实验组
        "shuffled": [reward1, reward2, ...]          # C3 消融对照
    }

用法：
    python scripts/evaluation/preregistered_real_machine_analysis.py \\
        --input results/real_machine/preregistered_data.json \\
        --output results/reports/preregistered_analysis_report.md

    # 验证模式（不生成报告，仅检查数据质量）
    python scripts/evaluation/preregistered_real_machine_analysis.py \\
        --input results/real_machine/preregistered_data.json \\
        --validate-only
"""

import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import click
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scipy import stats as sp_stats

from src.utils.stats_significance import (
    _mean_diff_ci,
    bootstrap_improvement_ci,
    cohen_d,
    normality_test,
    rank_biserial,
)

# ── 预注册常量（与 real_machine_preregistration.md 一致） ──

#: 预注册的三个条件名称
CONDITION_STATUS_ONLY = "status_only"
CONDITION_RESULT_AWARE = "result_aware"
CONDITION_SHUFFLED = "shuffled"

#: 预注册的假设
HYPOTHESES = {
    "H1": {
        "description": "result_aware 高于 status_only",
        "comparison": (CONDITION_RESULT_AWARE, CONDITION_STATUS_ONLY),
        "direction": "one-sided",
    },
    "H2": {
        "description": "result_aware 高于 shuffled",
        "comparison": (CONDITION_RESULT_AWARE, CONDITION_SHUFFLED),
        "direction": "one-sided",
    },
    "H3": {
        "description": "status_only 与 shuffled 无实质差异 (|d| < 0.2)",
        "comparison": (CONDITION_STATUS_ONLY, CONDITION_SHUFFLED),
        "direction": "two-sided",
    },
}

#: 最低样本量（每组）
MIN_SAMPLE_SIZE = 18

#: 最高样本量（每组）
MAX_SAMPLE_SIZE = 30

#: 异常值剔除标准（标准差倍数）
OUTLIER_SIGMA = 4.0

#: 显著性水平
ALPHA = 0.05

#: 比较次数（3 组两两比较 = 3 次）
N_COMPARISONS = 3

#: Bonferroni 校正后的 α
BONFERRONI_ALPHA = ALPHA / N_COMPARISONS


@dataclass
class CleaningRecord:
    """数据清洗记录"""

    condition: str
    original_count: int
    removed_outliers: list[float] = field(default_factory=list)
    removed_count: int = 0
    final_count: int = 0
    removal_reasons: list[str] = field(default_factory=list)


@dataclass
class ComparisonResult:
    """两两比较结果"""

    name: str
    hypothesis: str
    group_a: str
    group_b: str
    mean_a: float
    mean_b: float
    std_a: float
    std_b: float
    n_a: int
    n_b: int
    cohen_d: float
    effect_level: str
    rank_biserial: float
    mean_diff: float
    ci_lower: float
    ci_upper: float
    improvement_pct: float
    improvement_ci_lower: float
    improvement_ci_upper: float
    p_value: float
    test_name: str
    ci_crosses_zero: bool
    hypothesis_judgment: str


def load_data(input_path: Path) -> dict[str, list[float]]:
    """从 JSON 文件加载预注册实验数据

    Args:
        input_path: JSON 文件路径

    Returns:
        ``{条件名: [奖励列表]}`` 字典

    Raises:
        click.ClickException: 文件格式错误或缺少必要条件
    """
    with input_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise click.ClickException("输入 JSON 顶层必须是对象 {条件名: [奖励列表]}")

    required = {CONDITION_STATUS_ONLY, CONDITION_RESULT_AWARE, CONDITION_SHUFFLED}
    missing = required - set(raw.keys())
    if missing:
        raise click.ClickException(f"缺少预注册条件: {missing}")

    cleaned: dict[str, list[float]] = {}
    for name in required:
        rewards = raw[name]
        if not isinstance(rewards, list):
            raise click.ClickException(f"条件 {name!r} 的奖励不是列表")
        try:
            cleaned[name] = [float(r) for r in rewards]
        except (TypeError, ValueError) as e:
            raise click.ClickException(f"条件 {name!r} 含非数值奖励: {e}") from e

    return cleaned


def clean_data(
    data: dict[str, list[float]],
) -> tuple[dict[str, list[float]], list[CleaningRecord]]:
    """按预注册 §5.4 标准执行数据清洗

    清洗规则（预定义，不依赖分析结果）：
    1. 异常值：奖励超出 [mean ± 4σ] 的数据点
    2. 最低样本量检查：每组不少于 18 个数据点

    Args:
        data: 原始数据 ``{条件名: [奖励列表]}``

    Returns:
        (清洗后数据, 清洗记录列表)
    """
    cleaned: dict[str, list[float]] = {}
    records: list[CleaningRecord] = []

    for condition, rewards in data.items():
        arr = np.asarray(rewards, dtype=float)
        original_count = len(arr)

        if original_count == 0:
            record = CleaningRecord(
                condition=condition,
                original_count=0,
                final_count=0,
                removal_reasons=["数据为空"],
            )
            records.append(record)
            cleaned[condition] = []
            continue

        mean = float(arr.mean())
        std = float(arr.std(ddof=1)) if original_count > 1 else 0.0

        # 4σ 异常值剔除
        if std > 0:
            lower = mean - OUTLIER_SIGMA * std
            upper = mean + OUTLIER_SIGMA * std
            mask = (arr >= lower) & (arr <= upper)
            removed = arr[~mask].tolist()
        else:
            removed = []
            mask = np.ones(len(arr), dtype=bool)

        final_rewards = arr[mask].tolist()
        reasons: list[str] = []
        if removed:
            reasons.append(
                f"异常值剔除（{OUTLIER_SIGMA}σ）: {len(removed)} 个数据点 "
                f"超出 [{mean - OUTLIER_SIGMA * std:.2f}, {mean + OUTLIER_SIGMA * std:.2f}]"
            )

        if len(final_rewards) < MIN_SAMPLE_SIZE:
            reasons.append(
                f"样本量不足: {len(final_rewards)} < {MIN_SAMPLE_SIZE}（预注册最低要求）"
            )

        record = CleaningRecord(
            condition=condition,
            original_count=original_count,
            removed_outliers=removed,
            removed_count=len(removed),
            final_count=len(final_rewards),
            removal_reasons=reasons,
        )
        records.append(record)
        cleaned[condition] = final_rewards

    return cleaned, records


def _effect_level_cohen_d(d: float) -> str:
    """判定 Cohen's d 效应量等级"""
    if math.isnan(d):
        return "无法计算"
    abs_d = abs(d)
    if abs_d < 0.2:
        return "可忽略"
    if abs_d < 0.5:
        return "小效应"
    if abs_d < 0.8:
        return "中效应"
    return "大效应"


def _judge_hypothesis(
    hypothesis_id: str,
    d: float,
    ci_lower: float,
    ci_upper: float,
) -> str:
    """按预注册 §6.4 标准判定假设

    Args:
        hypothesis_id: 假设编号（H1/H2/H3）
        d: Cohen's d 效应量
        ci_lower: 均值差 95% CI 下界
        ci_upper: 均值差 95% CI 上界

    Returns:
        判定结果: "支持" / "不支持" / "不确定，需更多数据"
    """
    if math.isnan(d):
        return "无法判定（效应量无法计算）"

    abs_d = abs(d)
    ci_has_nan = math.isnan(ci_lower) or math.isnan(ci_upper)
    ci_crosses_zero = (not ci_has_nan) and (ci_lower <= 0 <= ci_upper)

    if hypothesis_id in ("H1", "H2"):
        # 单侧假设：d ≥ 0.5 且 CI 下界 > 0 → 支持
        if d >= 0.5 and not ci_crosses_zero and not ci_has_nan and ci_lower > 0:
            return "支持"
        # d < 0.2 或 CI 跨 0 → 不支持
        if d < 0.2 or ci_crosses_zero:
            return "不支持"
        # CI 无法计算但效应量足够大 → 弱支持（需更多数据验证 CI）
        if ci_has_nan and d >= 0.5:
            return "不确定，需更多数据（CI 无法计算）"
        # 中间区域
        return "不确定，需更多数据"

    # H3: |d| < 0.2 → 支持；|d| ≥ 0.5 → 不支持
    if abs_d < 0.2:
        return "支持"
    if abs_d >= 0.5:
        return "不支持"
    return "不确定，需更多数据"


def run_comparisons(
    data: dict[str, list[float]],
) -> list[ComparisonResult]:
    """执行预注册 §6.2 定义的两两比较

    直接计算 CI 和 p 值，不依赖 compare_strategies 的 pair key 匹配。

    Args:
        data: 清洗后的数据

    Returns:
        比较结果列表
    """
    results: list[ComparisonResult] = []

    for hyp_id, hyp_info in HYPOTHESES.items():
        group_a_name, group_b_name = hyp_info["comparison"]
        pair_key = f"{group_a_name} vs {group_b_name}"

        rewards_a = data.get(group_a_name, [])
        rewards_b = data.get(group_b_name, [])

        if len(rewards_a) < 2 or len(rewards_b) < 2:
            results.append(
                ComparisonResult(
                    name=pair_key,
                    hypothesis=hyp_id,
                    group_a=group_a_name,
                    group_b=group_b_name,
                    mean_a=float("nan"),
                    mean_b=float("nan"),
                    std_a=float("nan"),
                    std_b=float("nan"),
                    n_a=len(rewards_a),
                    n_b=len(rewards_b),
                    cohen_d=float("nan"),
                    effect_level="无法计算",
                    rank_biserial=float("nan"),
                    mean_diff=float("nan"),
                    ci_lower=float("nan"),
                    ci_upper=float("nan"),
                    improvement_pct=float("nan"),
                    improvement_ci_lower=float("nan"),
                    improvement_ci_upper=float("nan"),
                    p_value=float("nan"),
                    test_name="样本不足",
                    ci_crosses_zero=True,
                    hypothesis_judgment="无法判定（样本不足）",
                )
            )
            continue

        arr_a = np.asarray(rewards_a, dtype=float)
        arr_b = np.asarray(rewards_b, dtype=float)

        mean_a = float(arr_a.mean())
        mean_b = float(arr_b.mean())
        std_a = float(arr_a.std(ddof=1)) if len(rewards_a) > 1 else 0.0
        std_b = float(arr_b.std(ddof=1)) if len(rewards_b) > 1 else 0.0

        d = cohen_d(rewards_a, rewards_b)
        rb = rank_biserial(rewards_a, rewards_b)

        # Bootstrap CI（提升百分比）
        imp_pct, imp_ci_lo, imp_ci_hi = bootstrap_improvement_ci(
            rewards_a, rewards_b, confidence=0.95, n_bootstrap=10000
        )

        # 正态性检验 → 选择检验方法
        normal_a, _, _ = normality_test(rewards_a, ALPHA)
        normal_b, _, _ = normality_test(rewards_b, ALPHA)
        both_normal = normal_a and normal_b

        if both_normal:
            # 方差齐性检验
            lev = sp_stats.levene(arr_a, arr_b)
            equal_var = bool(lev.pvalue >= ALPHA)
            if equal_var:
                res = sp_stats.ttest_ind(arr_a, arr_b, equal_var=True)
                test_name = "独立样本 t 检验"
                mean_diff, ci_lo, ci_hi = _mean_diff_ci(rewards_a, rewards_b, equal_var=True)
            else:
                res = sp_stats.ttest_ind(arr_a, arr_b, equal_var=False)
                test_name = "Welch t 检验"
                mean_diff, ci_lo, ci_hi = _mean_diff_ci(rewards_a, rewards_b, equal_var=False)
        else:
            # 非正态 → Mann-Whitney U 检验
            res = sp_stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
            test_name = "Mann-Whitney U 检验"
            mean_diff, ci_lo, ci_hi = _mean_diff_ci(rewards_a, rewards_b, equal_var=True)

        p_val = float(res.pvalue)

        ci_has_nan = math.isnan(ci_lo) or math.isnan(ci_hi)
        ci_crosses = (not ci_has_nan) and (ci_lo <= 0 <= ci_hi)

        judgment = _judge_hypothesis(hyp_id, d, ci_lo, ci_hi)

        results.append(
            ComparisonResult(
                name=pair_key,
                hypothesis=hyp_id,
                group_a=group_a_name,
                group_b=group_b_name,
                mean_a=mean_a,
                mean_b=mean_b,
                std_a=std_a,
                std_b=std_b,
                n_a=len(rewards_a),
                n_b=len(rewards_b),
                cohen_d=d,
                effect_level=_effect_level_cohen_d(d),
                rank_biserial=rb,
                mean_diff=mean_diff,
                ci_lower=ci_lo,
                ci_upper=ci_hi,
                improvement_pct=imp_pct,
                improvement_ci_lower=imp_ci_lo,
                improvement_ci_upper=imp_ci_hi,
                p_value=p_val,
                test_name=test_name,
                ci_crosses_zero=ci_crosses,
                hypothesis_judgment=judgment,
            )
        )

    return results


def generate_report(
    data: dict[str, list[float]],
    cleaned_data: dict[str, list[float]],
    cleaning_records: list[CleaningRecord],
    comparisons: list[ComparisonResult],
    input_path: Path,
) -> str:
    """按预注册 §9 报告模板生成 Markdown 报告

    Args:
        data: 原始数据
        cleaned_data: 清洗后数据
        cleaning_records: 清洗记录
        comparisons: 比较结果
        input_path: 输入文件路径

    Returns:
        Markdown 格式的报告字符串
    """
    lines: list[str] = []
    lines.append("# 预注册真机闭环实验分析报告")
    lines.append("")
    lines.append("> **遵循预注册方案**: `results/reports/real_machine_preregistration.md`")
    lines.append(f"> **分析日期**: 基于 {input_path.name}")
    lines.append("> **预注册日期**: 2026-07-21（数据收集前）")
    lines.append("> **主要决策依据**: 效应量（Cohen's d）+ 95% CI")
    lines.append("> **p 值角色**: 辅助参考，非主要决策依据")
    lines.append("")
    lines.append("---")
    lines.append("")

    # §1 实验概述
    lines.append("## 一、实验概述")
    lines.append("")
    lines.append("| 条件 | `real_feedback_mode` | 角色 |")
    lines.append("|:-----|:---------------------|:-----|")
    lines.append(f"| C1: `{CONDITION_STATUS_ONLY}` | `status_only` | 基线 |")
    lines.append(f"| C2: `{CONDITION_RESULT_AWARE}` | `result_aware` | 实验组 |")
    lines.append(f"| C3: `{CONDITION_SHUFFLED}` | `shuffled` | 消融对照 |")
    lines.append("")
    lines.append(f"- **最低样本量要求**: {MIN_SAMPLE_SIZE} 组/条件")
    lines.append(f"- **异常值剔除标准**: {OUTLIER_SIGMA}σ")
    lines.append(f"- **Bonferroni 校正 α**: {BONFERRONI_ALPHA:.4f}（{N_COMPARISONS} 次比较）")
    lines.append("")
    lines.append("---")
    lines.append("")

    # §2 数据清洗报告
    lines.append("## 二、数据清洗报告（按预注册 §5.4 标准执行）")
    lines.append("")
    lines.append("| 条件 | 原始数据量 | 剔除异常值 | 最终数据量 | 状态 |")
    lines.append("|:-----|:----------:|:----------:|:----------:|:-----|")
    for rec in cleaning_records:
        status = "✅ 通过" if rec.final_count >= MIN_SAMPLE_SIZE else "⚠️ 样本不足"
        lines.append(
            f"| {rec.condition} | {rec.original_count} | {rec.removed_count} | "
            f"{rec.final_count} | {status} |"
        )
    lines.append("")
    if any(rec.removal_reasons for rec in cleaning_records):
        lines.append("### 清洗详情")
        lines.append("")
        for rec in cleaning_records:
            if rec.removal_reasons:
                lines.append(f"**{rec.condition}**:")
                for reason in rec.removal_reasons:
                    lines.append(f"- {reason}")
                if rec.removed_outliers:
                    lines.append(f"- 剔除的异常值: {[f'{v:.2f}' for v in rec.removed_outliers]}")
                lines.append("")
    lines.append("---")
    lines.append("")

    # §3 描述性统计
    lines.append("## 三、描述性统计")
    lines.append("")
    lines.append("| 条件 | N | 均值 | 标准差 | 中位数 | IQR |")
    lines.append("|:-----|:--:|:----:|:------:|:------:|:---:|")
    for condition in [CONDITION_STATUS_ONLY, CONDITION_RESULT_AWARE, CONDITION_SHUFFLED]:
        rewards = cleaned_data.get(condition, [])
        if len(rewards) == 0:
            lines.append(f"| {condition} | 0 | — | — | — | — |")
            continue
        arr = np.asarray(rewards, dtype=float)
        median = float(np.median(arr))
        q1 = float(np.percentile(arr, 25))
        q3 = float(np.percentile(arr, 75))
        iqr = q3 - q1
        lines.append(
            f"| {condition} | {len(rewards)} | {arr.mean():.2f} | "
            f"{arr.std(ddof=1):.2f} | {median:.2f} | {iqr:.2f} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # §4 正态性检验
    lines.append("## 四、正态性检验")
    lines.append("")
    lines.append("| 条件 | 检验方法 | p 值 | 结论 |")
    lines.append("|:-----|:---------|:----:|:-----|")
    for condition in [CONDITION_STATUS_ONLY, CONDITION_RESULT_AWARE, CONDITION_SHUFFLED]:
        rewards = cleaned_data.get(condition, [])
        if len(rewards) < 3:
            lines.append(f"| {condition} | — | — | 样本不足 |")
            continue
        is_normal, p_val, test_name = normality_test(rewards, ALPHA)
        conclusion = "正态" if is_normal else "非正态"
        lines.append(f"| {condition} | {test_name} | {p_val:.4f} | {conclusion} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # §5 效应量报告（主要）
    lines.append("## 五、效应量报告（主要决策依据）")
    lines.append("")
    lines.append("### 5.1 Cohen's d + 95% CI")
    lines.append("")
    lines.append(
        "| 比较 | 假设 | 均值A | 均值B | 均值差 | Cohen's d | 等级 | "
        "95% CI [下界, 上界] | CI 跨0? |"
    )
    lines.append(
        "|:-----|:----:|:----:|:----:|:------:|:---------:|:----:|:-------------------:|:------:|"
    )
    for comp in comparisons:
        ci_str = (
            f"[{comp.ci_lower:.2f}, {comp.ci_upper:.2f}]"
            if not math.isnan(comp.ci_lower)
            else "无法计算"
        )
        lines.append(
            f"| {comp.name} | {comp.hypothesis} | {comp.mean_a:.2f} | "
            f"{comp.mean_b:.2f} | {comp.mean_diff:.2f} | {comp.cohen_d:.4f} | "
            f"{comp.effect_level} | {ci_str} | "
            f"{'是' if comp.ci_crosses_zero else '否'} |"
        )
    lines.append("")

    lines.append("### 5.2 Bootstrap 提升百分比 95% CI")
    lines.append("")
    lines.append("| 比较 | 提升百分比 | 95% CI [下界, 上界] | CI 下界 > 0? |")
    lines.append("|:-----|:----------:|:-------------------:|:-----------:|")
    for comp in comparisons:
        ci_str = (
            f"[{comp.improvement_ci_lower:.2f}%, {comp.improvement_ci_upper:.2f}%]"
            if not math.isnan(comp.improvement_ci_lower)
            else "无法计算"
        )
        ci_positive = (
            "是"
            if not math.isnan(comp.improvement_ci_lower) and comp.improvement_ci_lower > 0
            else "否"
        )
        lines.append(f"| {comp.name} | {comp.improvement_pct:.2f}% | {ci_str} | {ci_positive} |")
    lines.append("")

    lines.append("### 5.3 rank-biserial correlation（非参数效应量）")
    lines.append("")
    lines.append("| 比较 | rank-biserial |")
    lines.append("|:-----|:------------:|")
    for comp in comparisons:
        rb_str = f"{comp.rank_biserial:.4f}" if not math.isnan(comp.rank_biserial) else "无法计算"
        lines.append(f"| {comp.name} | {rb_str} |")
    lines.append("")
    lines.append("---")
    lines.append("")

    # §6 假设检验报告（辅助）
    lines.append("## 六、假设检验报告（辅助参考）")
    lines.append("")
    lines.append(
        f"| 比较 | 检验方法 | 统计量 | p 值 | Bonferroni α（{BONFERRONI_ALPHA:.4f}） | 显著? |"
    )
    lines.append("|:-----|:---------|:------:|:----:|:-------------:|:------:|")
    for comp in comparisons:
        if math.isnan(comp.p_value):
            lines.append(f"| {comp.name} | {comp.test_name} | — | — | — | — |")
        else:
            significant = comp.p_value < BONFERRONI_ALPHA
            lines.append(
                f"| {comp.name} | {comp.test_name} | — | {comp.p_value:.6f} | "
                f"{'p < α' if significant else 'p ≥ α'} | "
                f"{'显著' if significant else '不显著'} |"
            )
    lines.append("")
    lines.append(
        "> **注意**: p 值为辅助参考。根据预注册方案，主要决策依据是效应量 + 95% CI，而非 p 值。"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # §7 假设判定
    lines.append("## 七、假设判定（基于效应量 + CI）")
    lines.append("")
    lines.append("| 假设 | 描述 | Cohen's d | CI 跨0? | 判定 |")
    lines.append("|:----:|:-----|:---------:|:------:|:-----|")
    for comp in comparisons:
        ci_cross_str = "是" if comp.ci_crosses_zero else "否"
        lines.append(
            f"| {comp.hypothesis} | {HYPOTHESES[comp.hypothesis]['description']} | "
            f"{comp.cohen_d:.4f} | {ci_cross_str} | **{comp.hypothesis_judgment}** |"
        )
    lines.append("")
    lines.append("### 判定标准回顾（预注册 §6.4）")
    lines.append("")
    lines.append("| 假设 | 支持 | 不支持 |")
    lines.append("|:-----|:-----|:-------|")
    lines.append("| H1/H2 | d ≥ 0.5 且 CI 下界 > 0 | d < 0.2 或 CI 跨 0 |")
    lines.append("| H3 | \\|d\\| < 0.2 | \\|d\\| ≥ 0.5 |")
    lines.append("")
    lines.append(
        '> 中间区域（0.2 ≤ d < 0.5 或 CI 跨 0）: 判定为 "不确定，需更多数据"，不强行得出结论。'
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    # §8 功效分析
    lines.append("## 八、功效分析（Post-Hoc）")
    lines.append("")
    for comp in comparisons:
        if math.isnan(comp.cohen_d) or comp.n_a < 2 or comp.n_b < 2:
            lines.append(f"- **{comp.name}**: 无法计算（样本不足）")
            continue
        # 简化功效估计：基于 Cohen's d 和样本量
        n_total = comp.n_a + comp.n_b
        d_abs = abs(comp.cohen_d)
        # 近似功效（正态近似）
        z = d_abs * math.sqrt(n_total / 2)
        # 使用正态 CDF 近似
        from scipy.stats import norm

        power = float(norm.cdf(z - 1.96))  # 单侧 α=0.025 近似
        lines.append(
            f"- **{comp.name}**: d={comp.cohen_d:.4f}, N={n_total}, "
            f"近似功效={power:.1%}（目标 80%）"
        )
    lines.append("")
    lines.append("---")
    lines.append("")

    # §9 探索性分析
    lines.append("## 九、探索性分析")
    lines.append("")
    lines.append("> 本节为探索性分析，**不作为正式结论**。仅用于生成后续研究假设。")
    lines.append("")
    lines.append('（本模板预留探索性分析位置，如有请在此填写并明确标注为"探索性"。）')
    lines.append("")
    lines.append("---")
    lines.append("")

    # §10 偏离声明
    lines.append("## 十、偏离声明")
    lines.append("")
    lines.append("> 如分析过程中有任何偏离预注册方案的行为，必须在此显式声明。")
    lines.append("")
    lines.append("（本模板预留偏离声明位置，如有请在此填写偏离原因及影响。）")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 核心信条
    lines.append("## 核心信条")
    lines.append("")
    lines.append('> 在小样本真机实验中，诚实的"不确定"比虚假的"显著"更有科学价值。')
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        "*预注册分析报告 | 基于 `results/reports/real_machine_preregistration.md` | "
        "使用 `src/utils/stats_significance.py` 统计基础设施*"
    )

    return "\n".join(lines)


@click.command()
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="输入 JSON 文件路径（{条件名: [奖励列表]}）",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(path_type=Path),
    default=Path("results/reports/preregistered_analysis_report.md"),
    help="输出 Markdown 报告路径",
)
@click.option(
    "--validate-only",
    is_flag=True,
    default=False,
    help="仅验证数据质量，不生成报告",
)
def main(
    input_path: Path,
    output_path: Path,
    validate_only: bool,
) -> None:
    """预注册真机闭环实验分析

    遵循 results/reports/real_machine_preregistration.md 中定义的分析计划，
    以效应量 + CI 为主要决策依据。
    """
    click.echo(f"加载实验数据: {input_path}")
    data = load_data(input_path)

    click.echo("执行数据清洗（预注册 §5.4 标准）...")
    cleaned_data, cleaning_records = clean_data(data)

    # 输出清洗摘要
    for rec in cleaning_records:
        status = "✅" if rec.final_count >= MIN_SAMPLE_SIZE else "⚠️"
        click.echo(
            f"  {rec.condition}: {rec.original_count} → {rec.final_count} "
            f"（剔除 {rec.removed_count}）{status}"
        )

    if validate_only:
        click.echo("\n验证模式：仅检查数据质量，不生成报告")
        all_pass = all(rec.final_count >= MIN_SAMPLE_SIZE for rec in cleaning_records)
        if all_pass:
            click.echo("✅ 所有条件满足最低样本量要求")
        else:
            click.echo("⚠️ 部分条件样本量不足，详见上方清洗记录")
        return

    click.echo("\n执行两两比较（预注册 §6.2 计划）...")
    comparisons = run_comparisons(cleaned_data)

    click.echo("生成报告...")
    report = generate_report(
        data=data,
        cleaned_data=cleaned_data,
        cleaning_records=cleaning_records,
        comparisons=comparisons,
        input_path=input_path,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write(report)

    click.echo(f"\n✅ 报告已生成: {output_path}")

    # 输出假设判定摘要
    click.echo("\n=== 假设判定摘要 ===")
    for comp in comparisons:
        click.echo(
            f"  {comp.hypothesis} ({comp.name}): d={comp.cohen_d:.4f} → {comp.hypothesis_judgment}"
        )


if __name__ == "__main__":
    main()
