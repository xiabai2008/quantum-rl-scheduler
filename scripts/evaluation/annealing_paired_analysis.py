#!/usr/bin/env python
"""
退火消融实验配对检验分析脚本 (Issue #319)

对消融实验数据执行配对 Wilcoxon signed-rank 检验，并与独立样本检验对比，
量化配对设计带来的功效提升。

功能：
1. 读取消融实验 JSON 数据（5 seeds 或 20 seeds）
2. 对每个 checkpoint 执行配对 Wilcoxon signed-rank 检验
3. 对每个 checkpoint 执行独立样本 Mann-Whitney U 检验（对比口径）
4. 计算配对效应量（matched-pairs rank-biserial correlation）
5. 计算事后统计功效（post-hoc power）
6. 计算达到 80% 功效所需样本量
7. 输出对比表格和 Markdown 报告
"""

import json
import math
import os
import sys
from datetime import datetime
from typing import Any

import numpy as np
from scipy import stats

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.stats_significance import (
    cohen_d,
    power_ttest,
    rank_biserial,
)

RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")


def matched_pairs_rank_biserial(diffs: list[float]) -> float:
    """计算配对检验的 rank-biserial 效应量。

    对于 Wilcoxon signed-rank 检验，效应量 r = W / (n*(n+1)/2)，
    其中 W 是正秩和。r 的范围 [0, 1]，转换为 [-1, 1] 形式。

    Args:
        diffs: 配对差值列表（x - y）

    Returns:
        rank-biserial 相关系数 [-1, 1]
    """
    arr = np.asarray(diffs, dtype=float)
    n = len(arr)
    if n < 2:
        return float("nan")
    # 移除零差值
    nonzero = arr[arr != 0]
    n_nz = len(nonzero)
    if n_nz < 2:
        return float("nan")
    # Wilcoxon signed-rank
    result = stats.wilcoxon(nonzero, alternative="greater")
    # 正秩和 W+ / 总秩和 T = n(n+1)/2
    total_ranks = n_nz * (n_nz + 1) / 2
    # W 统计量对应正秩和
    w_plus = float(result.statistic)
    # rank-biserial = (W+ - W-) / (W+ + W-) = (2*W+ - T) / T
    r = (2 * w_plus - total_ranks) / total_ranks
    return float(r)


def paired_power_analysis(
    diffs: list[float],
    n: int,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """配对检验的事后功效分析。

    基于配对差值的均值和标准差计算效应量 d_z = mean_diff / std_diff，
    然后使用单样本 t 检验的功效公式估算检验力。

    Args:
        diffs: 配对差值列表
        n: 样本量（配对数）
        alpha: 显著性水平

    Returns:
        包含效应量、功效、所需样本量的字典
    """
    arr = np.asarray(diffs, dtype=float)
    if n < 2 or len(arr) < 2:
        return {
            "d_z": float("nan"),
            "power": float("nan"),
            "n_needed_80": 0,
            "n_needed_90": 0,
        }

    mean_diff = float(np.mean(arr))
    std_diff = float(np.std(arr, ddof=1))
    if std_diff == 0:
        return {
            "d_z": float("inf") if mean_diff != 0 else 0.0,
            "power": 1.0 if mean_diff != 0 else 0.0,
            "n_needed_80": 2 if mean_diff != 0 else 0,
            "n_needed_90": 2 if mean_diff != 0 else 0,
        }

    # 配对效应量 d_z = mean_diff / std_diff
    d_z = mean_diff / std_diff

    # 使用 power_ttest（单样本等价于配对 t 检验）
    # 对于配对检验，n2 可以视为无穷大（单样本），但 power_ttest 需要两组
    # 使用 n1=n, n2=n（保守估计）或用单样本功效公式
    # 单样本 t 检验功效: power = P(T > t_crit | ncp = d_z * sqrt(n))
    ncp = abs(d_z) * math.sqrt(n)
    df = n - 1
    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    right_tail = float(stats.nct.sf(t_crit, df, ncp))
    left_tail_cdf = stats.nct.cdf(-t_crit, df, ncp)
    left_tail = 0.0 if math.isnan(left_tail_cdf) else float(left_tail_cdf)
    power = min(1.0, max(0.0, right_tail + left_tail))

    # 计算达到 80% 和 90% 功效所需样本量
    n_needed_80 = 0
    n_needed_90 = 0
    if not math.isnan(d_z) and d_z != 0:
        for test_n in range(3, 100000):
            test_ncp = abs(d_z) * math.sqrt(test_n)
            test_df = test_n - 1
            test_t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, test_df))
            test_right = float(stats.nct.sf(test_t_crit, test_df, test_ncp))
            test_left_cdf = stats.nct.cdf(-test_t_crit, test_df, test_ncp)
            test_left = 0.0 if math.isnan(test_left_cdf) else float(test_left_cdf)
            test_power = min(1.0, max(0.0, test_right + test_left))
            if n_needed_80 == 0 and test_power >= 0.80:
                n_needed_80 = test_n
            if n_needed_90 == 0 and test_power >= 0.90:
                n_needed_90 = test_n
                break

    return {
        "d_z": d_z,
        "power": power,
        "n_needed_80": n_needed_80,
        "n_needed_90": n_needed_90,
    }


def analyze_checkpoint(
    no_anneal_vals: list[float],
    with_anneal_vals: list[float],
    checkpoint_label: str,
) -> dict[str, Any]:
    """对单个 checkpoint 执行配对检验和独立检验对比。

    Args:
        no_anneal_vals: 无退火组各 seed 的奖励
        with_anneal_vals: 有退火组各 seed 的奖励
        checkpoint_label: checkpoint 标签（如 "50k"）

    Returns:
        包含所有检验结果的字典
    """
    n = len(no_anneal_vals)
    diffs = [with_anneal_vals[i] - no_anneal_vals[i] for i in range(n)]

    result: dict[str, Any] = {
        "checkpoint": checkpoint_label,
        "n": n,
        "no_anneal_mean": float(np.mean(no_anneal_vals)),
        "no_anneal_std": float(np.std(no_anneal_vals, ddof=1)) if n > 1 else 0.0,
        "with_anneal_mean": float(np.mean(with_anneal_vals)),
        "with_anneal_std": float(np.std(with_anneal_vals, ddof=1)) if n > 1 else 0.0,
        "mean_diff": float(np.mean(diffs)),
        "improvement_pct": 0.0,
    }
    result["improvement_pct"] = result["mean_diff"] / (abs(result["no_anneal_mean"]) + 1e-8) * 100

    # === 独立样本检验（原口径） ===
    # Mann-Whitney U 检验
    arr_a = np.asarray(with_anneal_vals, dtype=float)
    arr_b = np.asarray(no_anneal_vals, dtype=float)
    if n >= 2:
        mwu_result = stats.mannwhitneyu(arr_a, arr_b, alternative="two-sided")
        result["independent_mwu_p"] = float(mwu_result.pvalue)
        result["independent_mwu_stat"] = float(mwu_result.statistic)
        result["independent_rank_biserial"] = rank_biserial(with_anneal_vals, no_anneal_vals)

        # 独立样本 t 检验（Welch）
        t_result = stats.ttest_ind(arr_a, arr_b, equal_var=False)
        result["independent_t_p"] = float(t_result.pvalue)
        result["independent_t_stat"] = float(t_result.statistic)
        result["independent_cohen_d"] = cohen_d(with_anneal_vals, no_anneal_vals)

        # 独立样本功效
        d_ind = abs(result["independent_cohen_d"])
        result["independent_power"] = power_ttest(d_ind, n, n)
    else:
        result["independent_mwu_p"] = float("nan")
        result["independent_t_p"] = float("nan")
        result["independent_cohen_d"] = float("nan")
        result["independent_power"] = float("nan")

    # === 配对检验（新口径） ===
    nonzero_diffs = [d for d in diffs if d != 0]
    if len(nonzero_diffs) >= 5:
        # Wilcoxon signed-rank 检验
        # 注意：scipy 要求 n >= 5 且差值不全为 0
        try:
            wilcox_result = stats.wilcoxon(
                [float(x) for x in with_anneal_vals],
                [float(x) for x in no_anneal_vals],
                alternative="greater",
            )
            result["paired_wilcoxon_p"] = float(wilcox_result.pvalue)
            result["paired_wilcoxon_stat"] = float(wilcox_result.statistic)
        except ValueError:
            result["paired_wilcoxon_p"] = float("nan")
            result["paired_wilcoxon_stat"] = float("nan")
    elif len(nonzero_diffs) >= 1:
        # 对于 n < 5，使用精确 p 值
        try:
            wilcox_result = stats.wilcoxon(
                [float(x) for x in with_anneal_vals],
                [float(x) for x in no_anneal_vals],
                alternative="greater",
                zero_method="wilcox",
            )
            result["paired_wilcoxon_p"] = float(wilcox_result.pvalue)
            result["paired_wilcoxon_stat"] = float(wilcox_result.statistic)
        except (ValueError, TypeError):
            result["paired_wilcoxon_p"] = float("nan")
            result["paired_wilcoxon_stat"] = float("nan")
    else:
        result["paired_wilcoxon_p"] = float("nan")
        result["paired_wilcoxon_stat"] = float("nan")

    # 配对效应量
    result["paired_rank_biserial"] = matched_pairs_rank_biserial(diffs)

    # 配对 t 检验
    if n >= 2:
        paired_t = stats.ttest_rel(arr_a, arr_b)
        result["paired_t_p"] = float(paired_t.pvalue)
        result["paired_t_stat"] = float(paired_t.statistic)
    else:
        result["paired_t_p"] = float("nan")

    # 配对功效分析
    pa = paired_power_analysis(diffs, n)
    result["paired_d_z"] = pa["d_z"]
    result["paired_power"] = pa["power"]
    result["n_needed_80"] = pa["n_needed_80"]
    result["n_needed_90"] = pa["n_needed_90"]

    return result


def generate_report(
    all_results: list[dict[str, Any]],
    n_seeds: int,
    data_source: str,
) -> str:
    """生成 Markdown 格式的配对检验分析报告。

    Args:
        all_results: 各 checkpoint 的分析结果列表
        n_seeds: seed 数量
        data_source: 数据来源描述

    Returns:
        Markdown 格式的报告字符串
    """
    lines = [
        f"# 退火消融实验配对检验分析报告（{n_seeds} Seeds）",
        "",
        f"> **数据来源**: `{data_source}`",
        f"> **分析时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "> **Issue**: #319",
        "",
        "## 1. 分析背景",
        "",
        "原始 5 seed 消融实验使用独立样本检验（Mann-Whitney U / Welch t），",
        "但实验设计使用相同 seeds 进行有退火/无退火对比，本质上是配对设计。",
        "配对 Wilcoxon signed-rank 检验可消除 seed 间方差，大幅提升统计功效。",
        "",
        "## 2. 各 Checkpoint 检验结果对比",
        "",
        "| Checkpoint | 无退火 (mean±std) | 有退火 (mean±std) | 提升% |",
        "独立 MWU p | 配对 Wilcoxon p | 独立功效 | 配对功效 |",
        "|:--|:--|:--|:--|:--|:--|:--|:--|",
    ]

    for r in all_results:
        cp = r["checkpoint"]
        no_str = f"{r['no_anneal_mean']:.1f}±{r['no_anneal_std']:.1f}"
        an_str = f"{r['with_anneal_mean']:.1f}±{r['with_anneal_std']:.1f}"
        imp = f"{r['improvement_pct']:+.1f}%"
        ind_p = f"{r['independent_mwu_p']:.4f}" if not math.isnan(r["independent_mwu_p"]) else "N/A"
        paired_p = (
            f"{r['paired_wilcoxon_p']:.4f}" if not math.isnan(r["paired_wilcoxon_p"]) else "N/A"
        )
        ind_pow = (
            f"{r['independent_power']:.1%}" if not math.isnan(r["independent_power"]) else "N/A"
        )
        paired_pow = f"{r['paired_power']:.1%}" if not math.isnan(r["paired_power"]) else "N/A"
        lines.append(
            f"| {cp} | {no_str} | {an_str} | {imp} | {ind_p} | {paired_p} | {ind_pow} | {paired_pow} |"
        )

    lines.extend(
        [
            "",
            "## 3. 功效分析（最终 checkpoint 50k）",
            "",
        ]
    )

    # 找到 50k 的结果
    final = next((r for r in all_results if "50k" in r["checkpoint"]), all_results[-1])
    lines.append(f"### 最终 checkpoint ({final['checkpoint']}) 功效分析")
    lines.append("")
    lines.append("| 指标 | 独立样本检验 | 配对检验 |")
    lines.append("|:--|:--|:--|")
    lines.append(
        f"| p 值 | {final['independent_mwu_p']:.4f} (MWU) | "
        f"{final['paired_wilcoxon_p']:.4f} (Wilcoxon) |"
    )
    lines.append(
        f"| 效应量 | Cohen's d={final['independent_cohen_d']:.4f} | d_z={final['paired_d_z']:.4f} |"
    )
    lines.append(
        f"| rank-biserial | {final['independent_rank_biserial']:.4f} | "
        f"{final['paired_rank_biserial']:.4f} |"
    )
    lines.append(f"| 统计功效 | {final['independent_power']:.1%} | {final['paired_power']:.1%} |")
    lines.append("")
    lines.append("### 达到 80% 功效所需样本量")
    lines.append("")
    if final["n_needed_80"] > 0:
        lines.append(
            f"- 基于当前配对效应量 d_z={final['paired_d_z']:.4f}，"
            f"达到 80% 功效需 **{final['n_needed_80']}** 个配对样本"
        )
    else:
        lines.append("- 无法计算（效应量过小或为 0）")
    lines.append("")

    # 最优 checkpoint
    best_cp = max(all_results, key=lambda r: r["improvement_pct"])
    lines.append("## 4. 最优 Checkpoint")
    lines.append("")
    lines.append(
        f"最大提升出现在 **{best_cp['checkpoint']}**：提升 {best_cp['improvement_pct']:+.1f}%"
    )
    lines.append(
        f"- 独立 MWU p={best_cp['independent_mwu_p']:.4f}，"
        f"配对 Wilcoxon p={best_cp['paired_wilcoxon_p']:.4f}"
    )
    lines.append(
        f"- 配对功效={best_cp['paired_power']:.1%}，独立功效={best_cp['independent_power']:.1%}"
    )
    lines.append("")

    lines.append("## 5. 结论")
    lines.append("")
    # 动态结论
    final_paired_p = final["paired_wilcoxon_p"]
    if not math.isnan(final_paired_p) and final_paired_p < 0.05:
        lines.append(
            f"配对 Wilcoxon signed-rank 检验 p={final_paired_p:.4f} < 0.05，"
            "退火效果在配对设计下达到统计显著。"
        )
        lines.append(
            "配对检验通过消除 seed 间方差，成功将 p 值从"
            f" {final['independent_mwu_p']:.4f}（独立 MWU）"
            f" 降至 {final_paired_p:.4f}（配对 Wilcoxon）。"
        )
    else:
        p_str = f"{final_paired_p:.4f}" if not math.isnan(final_paired_p) else "N/A"
        lines.append(
            f"配对 Wilcoxon signed-rank 检验 p={p_str} ≥ 0.05，"
            "退火效果在配对设计下仍未达到统计显著。"
        )
        if final["n_needed_80"] > 0:
            lines.append(
                f"基于当前配对效应量 d_z={final['paired_d_z']:.4f}，"
                f"达到 80% 功效需 {final['n_needed_80']} 个配对样本。"
            )

    return "\n".join(lines)


def analyze_data_file(data_path: str) -> None:
    """分析指定的消融实验数据文件。

    Args:
        data_path: JSON 数据文件路径
    """
    with open(data_path, encoding="utf-8") as f:
        data = json.load(f)

    # 提取 per-seed 数据
    no_anneal_seeds = data["no_anneal"]["per_seed"]
    with_anneal_seeds = data["with_anneal"]["per_seed"]
    timesteps = data["no_anneal"]["timesteps"]
    n_seeds = len(no_anneal_seeds)
    n_evals = len(timesteps)

    print(f"Loaded: {data_path}")
    print(f"  Seeds: {n_seeds}, Checkpoints: {n_evals}")
    print(f"  Timesteps: {timesteps}")
    print()

    # 对每个 checkpoint 执行分析
    all_results: list[dict[str, Any]] = []
    for j, ts in enumerate(timesteps):
        cp_label = f"{int(ts / 1000)}k"
        no_vals = [s["rewards"][j] for s in no_anneal_seeds if s["rewards"]]
        an_vals = [s["rewards"][j] for s in with_anneal_seeds if s["rewards"]]

        if len(no_vals) < 2 or len(an_vals) < 2:
            print(f"  Skipping {cp_label}: insufficient data")
            continue

        result = analyze_checkpoint(no_vals, an_vals, cp_label)
        all_results.append(result)

        # 打印简要结果
        ind_sig = "✓" if result["independent_mwu_p"] < 0.05 else "✗"
        pair_sig = "✓" if result["paired_wilcoxon_p"] < 0.05 else "✗"
        print(
            f"  {cp_label}: ind_p={result['independent_mwu_p']:.4f}{ind_sig} "
            f"pair_p={result['paired_wilcoxon_p']:.4f}{pair_sig} "
            f"ind_pow={result['independent_power']:.1%} "
            f"pair_pow={result['paired_power']:.1%}"
        )

    print()

    # 生成报告
    report = generate_report(all_results, n_seeds, data_path)

    # 保存报告
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_name = f"annealing_paired_analysis_{n_seeds}seeds.md"
    report_path = os.path.join(REPORTS_DIR, report_name)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"Report saved to: {report_path}")

    # 同时保存 JSON 格式的完整结果
    json_result = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": data_path,
        "n_seeds": n_seeds,
        "checkpoints": all_results,
    }
    json_path = os.path.join(RESULTS_DIR, f"annealing_paired_analysis_{n_seeds}seeds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_result, f, indent=2, ensure_ascii=False, default=str)
    print(f"JSON results saved to: {json_path}")


def main() -> None:
    """主函数：分析所有可用的消融数据文件。"""
    # 查找所有消融数据文件
    data_files = []

    # 优先查找 20 seeds 数据
    path_20 = os.path.join(RESULTS_DIR, "ablation_annealing_multiseed_20seeds.json")
    if os.path.exists(path_20):
        data_files.append(("20 seeds", path_20))

    # 查找原始 5 seeds 数据
    for fname in sorted(os.listdir(RESULTS_DIR)):
        if (
            fname.startswith("ablation_annealing_multiseed_")
            and fname.endswith(".json")
            and "20seeds" not in fname
        ):
            data_files.append(("5 seeds (original)", os.path.join(RESULTS_DIR, fname)))

    if not data_files:
        print("No ablation data files found in results/")
        print("Please run ablation_annealing_20seeds.py first to generate data.")
        sys.exit(1)

    for label, path in data_files:
        print(f"\n{'=' * 60}")
        print(f"Analyzing: {label}")
        print(f"{'=' * 60}")
        analyze_data_file(path)


if __name__ == "__main__":
    main()
