#!/usr/bin/env python
"""Issue #39：多负载模式×8策略对比的 seed 级配对统计显著性检验。

对 ``run_workload_pattern_comparison.py`` 生成的逐 episode reward 数据做：

    1. 每 seed 聚合 5 episodes 均值 → N=10 seed 级样本
    2. 相同 seed 的策略比较使用 paired t-test（主分析）
    3. Wilcoxon signed-rank（敏感性分析）
    4. 配对效应量 d_z = mean(diffs) / sd(diffs)
    5. Holm-Bonferroni 校正（7 比较）

输出：
    - JSON：完整统计结果
    - Markdown 报告：可追溯的统计检验摘要

用法：
    python scripts/evaluation/workload_pattern_stats.py \\
        --input results/workload_pattern_evaluation/workload_pattern_results.json \\
        --output-data results/workload_pattern_evaluation/workload_pattern_stats.json \\
        --output-report results/reports/workload_pattern_stats.md
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

ALPHA = 0.05
REFERENCE_STRATEGY = "PPO"
BONFERRONI_COMPARISONS = 7  # PPO vs 其余 7 策略
BONFERRONI_ALPHA = ALPHA / BONFERRONI_COMPARISONS
MIN_PAIRED_SAMPLE = 5  # 配对检验最少需要 5 对


def holm_bonferroni(p_values: list[float], alpha: float = ALPHA) -> tuple[list[bool], list[float]]:
    """Holm-Bonferroni 逐步校正。

    返回 (rejected, adjusted_p)，顺序与输入一致。
    """
    m = len(p_values)
    if m == 0:
        return [], []
    order = sorted(range(m), key=lambda i: p_values[i])
    rejected = [False] * m
    adj_p_sorted = [0.0] * m
    prev_adj = 0.0
    for rank, idx in enumerate(order):
        raw = p_values[idx]
        correction = (m - rank) * raw
        adj = max(correction, prev_adj)  # 单调性
        adj = min(adj, 1.0)
        adj_p_sorted[rank] = adj
        rejected[rank] = raw <= alpha / (m - rank)
        prev_adj = adj
    final_rejected = [False] * m
    final_adj = [0.0] * m
    for rank, idx in enumerate(order):
        final_adj[idx] = adj_p_sorted[rank]
        final_rejected[idx] = rejected[rank]
    return final_rejected, final_adj


def cohens_d_z(diffs: np.ndarray) -> float:
    """配对效应量 d_z = mean(diffs) / sd(diffs)。"""
    if len(diffs) < 2:
        return 0.0
    sd = float(np.std(diffs, ddof=1))
    if sd == 0:
        return 0.0 if float(np.mean(diffs)) == 0 else math.inf
    return float(np.mean(diffs)) / sd


def mean_diff_ci_paired(diffs: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """返回 (mean_diff, lower, upper) 配对 95% CI。"""
    n = len(diffs)
    if n < 2:
        m = float(np.mean(diffs)) if n == 1 else 0.0
        return m, m, m
    mean_diff = float(np.mean(diffs))
    se = float(np.std(diffs, ddof=1)) / math.sqrt(n)
    if se == 0:
        return mean_diff, mean_diff, mean_diff
    tcrit = stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean_diff, mean_diff - tcrit * se, mean_diff + tcrit * se


def effect_size_label(d: float) -> str:
    """Cohen's d 效应等级。"""
    ad = abs(d)
    if ad < 0.2:
        return "可忽略"
    if ad < 0.5:
        return "小效应"
    if ad < 0.8:
        return "中效应"
    return "大效应"


def validate_input(raw: dict[str, Any], seeds: int, episodes_per_seed: int) -> None:
    """校验输入数据结构，失败时抛出 ValueError。"""
    if "patterns" not in raw:
        raise ValueError("输入数据缺少 'patterns' 字段")
    if not isinstance(raw["patterns"], dict) or not raw["patterns"]:
        raise ValueError("'patterns' 必须是非空字典")
    expected_len = seeds * episodes_per_seed
    for pattern_key, pattern_data in raw["patterns"].items():
        strategies = pattern_data.get("strategies")
        if not isinstance(strategies, dict) or not strategies:
            raise ValueError(f"模式 '{pattern_key}' 缺少 'strategies' 或为空")
        if REFERENCE_STRATEGY not in strategies:
            raise ValueError(f"模式 '{pattern_key}' 缺少参考策略 '{REFERENCE_STRATEGY}'")
        ref_rewards = strategies[REFERENCE_STRATEGY].get("episode_rewards")
        if ref_rewards is None:
            raise ValueError(f"模式 '{pattern_key}' 的 {REFERENCE_STRATEGY} 缺少 'episode_rewards'")
        if len(ref_rewards) != expected_len:
            raise ValueError(
                f"模式 '{pattern_key}' 的 {REFERENCE_STRATEGY} "
                f"episode_rewards 长度 {len(ref_rewards)} "
                f"与 seeds×episodes={expected_len} 不一致"
            )
        for name, data in strategies.items():
            er = data.get("episode_rewards")
            if er is None:
                raise ValueError(f"模式 '{pattern_key}' 的策略 '{name}' 缺少 'episode_rewards'")
            if len(er) != expected_len:
                raise ValueError(
                    f"模式 '{pattern_key}' 的策略 '{name}' "
                    f"episode_rewards 长度 {len(er)} "
                    f"与预期 {expected_len} 不一致"
                )
    if seeds < MIN_PAIRED_SAMPLE:
        raise ValueError(f"配对检验至少需要 {MIN_PAIRED_SAMPLE} seeds，当前 {seeds}")


def aggregate_to_seed_level(
    episode_rewards: list[float], seeds: int, episodes_per_seed: int
) -> np.ndarray:
    """将扁平 episode_rewards 按 seed 聚合为 seed 级均值。

    生成顺序为 ``for seed: for episode: rewards.append()``，
    因此连续 episodes_per_seed 个元素属于同一 seed。
    """
    arr = np.array(episode_rewards, dtype=float).reshape(seeds, episodes_per_seed)
    return arr.mean(axis=1)


def analyze_pattern(
    pattern_key: str,
    strategies: dict[str, dict[str, Any]],
    seeds: int,
    episodes_per_seed: int,
) -> dict[str, Any]:
    """对单个负载模式做 PPO vs 其余 7 策略的 seed 级配对检验。"""
    ref_seed_means = aggregate_to_seed_level(
        strategies[REFERENCE_STRATEGY]["episode_rewards"],
        seeds,
        episodes_per_seed,
    )

    comparisons: list[dict[str, Any]] = []
    for name, data in strategies.items():
        if name == REFERENCE_STRATEGY:
            continue
        other_seed_means = aggregate_to_seed_level(
            data["episode_rewards"], seeds, episodes_per_seed
        )
        diffs = ref_seed_means - other_seed_means

        # 主分析：paired t-test
        t_stat, p_ttest = stats.ttest_rel(ref_seed_means, other_seed_means)
        # 敏感性分析：Wilcoxon signed-rank
        try:
            w_stat, p_wilcoxon = stats.wilcoxon(ref_seed_means, other_seed_means)
            wilcoxon_w = float(w_stat)
            wilcoxon_p = float(p_wilcoxon)
        except ValueError:
            wilcoxon_w = None
            wilcoxon_p = None

        d_z = cohens_d_z(diffs)
        diff, lo, hi = mean_diff_ci_paired(diffs)

        comparisons.append(
            {
                "comparison": f"{REFERENCE_STRATEGY} vs {name}",
                "reference": {
                    "name": REFERENCE_STRATEGY,
                    "n_seeds": seeds,
                    "seed_mean": float(np.mean(ref_seed_means)),
                    "seed_std": float(np.std(ref_seed_means, ddof=1)),
                },
                "other": {
                    "name": name,
                    "n_seeds": seeds,
                    "seed_mean": float(np.mean(other_seed_means)),
                    "seed_std": float(np.std(other_seed_means, ddof=1)),
                },
                "paired_t": float(t_stat),
                "p_value": float(p_ttest),
                "wilcoxon_w": wilcoxon_w,
                "wilcoxon_p": wilcoxon_p,
                "cohens_d_z": float(d_z),
                "effect_size": effect_size_label(d_z),
                "mean_diff": float(diff),
                "ci_95": [float(lo), float(hi)],
            }
        )

    # Holm-Bonferroni 校正
    raw_p_values = [c["p_value"] for c in comparisons]
    rejected, adj_p = holm_bonferroni(raw_p_values, ALPHA)
    for comp, rej, ap, bp in zip(comparisons, rejected, adj_p, raw_p_values, strict=True):
        comp["bonferroni_alpha"] = BONFERRONI_ALPHA
        comp["bonferroni_significant"] = bool(bp < BONFERRONI_ALPHA)
        comp["holm_adjusted_p"] = float(ap)
        comp["holm_significant"] = bool(rej)
        comp["judgment"] = "支持" if rej else "不支持"

    comparisons.sort(key=lambda c: c["p_value"])
    return {
        "pattern": pattern_key,
        "n_seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "alpha": ALPHA,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "reference_strategy": REFERENCE_STRATEGY,
        "comparisons": comparisons,
    }


def generate_report(
    results: dict[str, Any],
    report_path: Path,
    data_path: Path,
) -> None:
    """生成 Markdown 统计检验报告。"""
    lines = [
        "# 多负载模式统计显著性检验报告",
        "",
        f"> 生成时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"> 参考策略：{REFERENCE_STRATEGY}",
        f"> 样本口径：N={results['n_seeds']} seeds（每 seed 聚合 {results['episodes_per_seed']} episodes 均值）",
        f"> α = {ALPHA}，Bonferroni 校正 α = {BONFERRONI_ALPHA:.5f}（{BONFERRONI_COMPARISONS} 比较）",
        "> 数据源：`workload_pattern_results.json` 逐 episode reward",
        "",
        "## 检验方法",
        "",
        "- **主分析**：paired t-test（同 seed 配对，消除 seed 间变异）",
        "- **敏感性分析**：Wilcoxon signed-rank（非参数）",
        "- **效应量**：配对效应量 d_z = mean(diffs) / sd(diffs)",
        "- **多重比较校正**：Holm-Bonferroni（7 比较）",
        "- **置信区间**：配对均值差的 95% CI",
        "- **判定**：Holm 校正后 p < α → 支持；否则 → 不支持",
        "",
    ]

    for pattern_key, pattern_data in results["patterns"].items():
        label = pattern_data.get("label", pattern_key)
        lines.extend(
            [
                f"## {label} 模式",
                "",
                "| 比较 | PPO seed均值 | 对手 seed均值 | paired t | p 值 | Holm adj.p | Holm 显著 | d_z | 效应等级 | 95% CI | 判定 |",
                "|:--|--:|--:|--:|--:|--:|:--|--:|:--|--:|:--|",
            ]
        )
        for comp in pattern_data["comparisons"]:
            sig = "✅ 是" if comp["holm_significant"] else "❌ 否"
            ci_str = f"[{comp['ci_95'][0]:.1f}, {comp['ci_95'][1]:.1f}]"
            lines.append(
                f"| {comp['comparison']} | "
                f"{comp['reference']['seed_mean']:.2f} | "
                f"{comp['other']['seed_mean']:.2f} | "
                f"{comp['paired_t']:.3f} | "
                f"{comp['p_value']:.2e} | "
                f"{comp['holm_adjusted_p']:.2e} | {sig} | "
                f"{comp['cohens_d_z']:.3f} | {comp['effect_size']} | {ci_str} | "
                f"{comp['judgment']} |"
            )
        lines.append("")

    sig_counts = []
    for pattern_key, pattern_data in results["patterns"].items():
        sc = sum(1 for c in pattern_data["comparisons"] if c["holm_significant"])
        sig_counts.append((pattern_key, sc))
    summary_line = "；".join(f"{pk}: {sc}/7 Holm 显著" for pk, sc in sig_counts)

    lines.extend(
        [
            "## 结论",
            "",
            f"- 样本口径：N={results['n_seeds']} seeds（配对设计），"
            f"非 N={results['n_seeds'] * results['episodes_per_seed']} 独立样本。",
            f"- Holm 校正后显著性：{summary_line}。",
            "- 不显著的比较判定为「不支持」，不得解读为「相等」。",
            "- 本报告复用 `workload_pattern_results.json` 的逐 episode reward，不重新跑仿真。",
            "",
            f"数据文件：`{data_path.as_posix()}`",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #39 多负载模式 seed 级配对统计检验")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/workload_pattern_evaluation/workload_pattern_results.json"),
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=Path("results/workload_pattern_evaluation/workload_pattern_stats.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("results/reports/workload_pattern_stats.md"),
    )
    parser.add_argument("--seeds", type=int, default=10, help="seed 数量")
    parser.add_argument("--episodes-per-seed", type=int, default=5, help="每 seed 的 episode 数")
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        raw = json.load(f)

    # 从 config 读取 seed 信息（如果与命令行参数一致）
    config = raw.get("config", {})
    seeds = config.get("seeds", args.seeds)
    episodes_per_seed = config.get("episodes_per_seed", args.episodes_per_seed)
    # 优先使用数据文件中的 seed_list 长度
    seed_list = config.get("seed_list")
    if seed_list:
        seeds = len(seed_list)
    episodes_per_seed = config.get("episodes_per_seed", episodes_per_seed)

    validate_input(raw, seeds, episodes_per_seed)

    pattern_stats: dict[str, Any] = {}
    for pattern_key, pattern_data in raw["patterns"].items():
        pattern_stats[pattern_key] = analyze_pattern(
            pattern_key,
            pattern_data["strategies"],
            seeds,
            episodes_per_seed,
        )

    results = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_data": str(args.input),
        "n_seeds": seeds,
        "episodes_per_seed": episodes_per_seed,
        "sample_unit": "seed",
        "alpha": ALPHA,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "reference_strategy": REFERENCE_STRATEGY,
        "patterns": pattern_stats,
    }

    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_data, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    generate_report(results, args.output_report, args.output_data)

    print(f"统计检验完成：{len(pattern_stats)} 模式 × 7 比较（N={seeds} seeds 配对）")
    print(f"数据：{args.output_data}")
    print(f"报告：{args.output_report}")

    for pattern_key, pdata in pattern_stats.items():
        sig_count = sum(1 for c in pdata["comparisons"] if c["holm_significant"])
        print(
            f"  {pattern_key}: {sig_count}/7 Holm 显著，"
            f"最强效应 d_z={max(abs(c['cohens_d_z']) for c in pdata['comparisons']):.3f}"
        )


if __name__ == "__main__":
    main()
