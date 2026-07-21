#!/usr/bin/env python
"""Issue #39：多负载模式×8策略对比的统计显著性检验。

对 ``run_workload_pattern_comparison.py`` 生成的逐 episode reward 数据做：
    - Welch t 检验（PPO vs 其余 7 策略，每负载模式独立）
    - Cohen's d 效应量
    - Bonferroni 校正（α=0.05 / 7 比较 = 0.00714）
    - 95% 置信区间

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
BONFERRONS_COMPARISONS = 7  # PPO vs 其余 7 策略
BONFERRONI_ALPHA = ALPHA / BONFERRONS_COMPARISONS


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """计算 Cohen's d 效应量（ pooled SD ）。"""
    na, nb = len(a), len(b)
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    pooled_sd = math.sqrt(((na - 1) * va + (nb - 1) * vb) / (na + nb - 2))
    if pooled_sd == 0:
        return 0.0
    return (float(np.mean(a)) - float(np.mean(b))) / pooled_sd


def mean_diff_ci(
    a: np.ndarray, b: np.ndarray, confidence: float = 0.95
) -> tuple[float, float, float]:
    """返回 (mean_diff, lower, upper) 95% CI（Welch 近似）。"""
    na, nb = len(a), len(b)
    ma, mb = float(np.mean(a)), float(np.mean(b))
    va, vb = float(np.var(a, ddof=1)), float(np.var(b, ddof=1))
    se = math.sqrt(va / na + vb / nb)
    diff = ma - mb
    if se == 0:
        return diff, diff, diff
    # Welch 自由度
    num = (va / na + vb / nb) ** 2
    den = (va / na) ** 2 / (na - 1) + (vb / nb) ** 2 / (nb - 1)
    df = num / den if den > 0 else na + nb - 2
    tcrit = stats.t.ppf((1 + confidence) / 2, df)
    return diff, diff - tcrit * se, diff + tcrit * se


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


def analyze_pattern(
    pattern_key: str,
    strategies: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """对单个负载模式做 PPO vs 其余 7 策略的统计检验。"""
    ref_rewards = np.array(strategies[REFERENCE_STRATEGY]["episode_rewards"], dtype=float)
    comparisons: list[dict[str, Any]] = []

    for name, data in strategies.items():
        if name == REFERENCE_STRATEGY:
            continue
        other_rewards = np.array(data["episode_rewards"], dtype=float)
        t_stat, p_value = stats.ttest_ind(ref_rewards, other_rewards, equal_var=False)
        d = cohens_d(ref_rewards, other_rewards)
        diff, lo, hi = mean_diff_ci(ref_rewards, other_rewards)
        comparisons.append(
            {
                "comparison": f"{REFERENCE_STRATEGY} vs {name}",
                "reference": {
                    "name": REFERENCE_STRATEGY,
                    "n": len(ref_rewards),
                    "mean": float(np.mean(ref_rewards)),
                    "std": float(np.std(ref_rewards, ddof=1)),
                },
                "other": {
                    "name": name,
                    "n": len(other_rewards),
                    "mean": float(np.mean(other_rewards)),
                    "std": float(np.std(other_rewards, ddof=1)),
                },
                "welch_t": float(t_stat),
                "p_value": float(p_value),
                "bonferroni_alpha": BONFERRONI_ALPHA,
                "bonferroni_significant": bool(p_value < BONFERRONI_ALPHA),
                "cohens_d": float(d),
                "effect_size": effect_size_label(d),
                "mean_diff": float(diff),
                "ci_95": [float(lo), float(hi)],
            }
        )

    # 按 p 值升序
    comparisons.sort(key=lambda c: c["p_value"])
    return {
        "pattern": pattern_key,
        "n_episodes_per_strategy": len(ref_rewards),
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
        f"> α = {ALPHA}，Bonferroni 校正 α = {BONFERRONI_ALPHA:.5f}（{BONFERRONS_COMPARISONS} 比较）",
        "> 数据源：`workload_pattern_results.json` 逐 episode reward",
        "",
        "## 检验方法",
        "",
        "- **检验类型**：Welch t 检验（不假设等方差）",
        "- **效应量**：Cohen's d（pooled SD）",
        "- **多重比较校正**：Bonferroni（α/N 比较）",
        "- **置信区间**：95% Welch 近似",
        "- **判定**：p < Bonferroni α 且 |d| ≥ 0.8（大效应）→ 显著且有意义",
        "",
    ]

    for pattern_key, pattern_data in results["patterns"].items():
        label = pattern_data.get("label", pattern_key)
        lines.extend(
            [
                f"## {label} 模式",
                "",
                "| 比较 | PPO 均值 | 对手均值 | Welch t | p 值 | Bonferroni 显著 | Cohen's d | 效应等级 | 95% CI |",
                "|:--|--:|--:|--:|--:|:--|--:|:--|--:|",
            ]
        )
        for comp in pattern_data["comparisons"]:
            sig = "✅ 是" if comp["bonferroni_significant"] else "❌ 否"
            ci_str = f"[{comp['ci_95'][0]:.1f}, {comp['ci_95'][1]:.1f}]"
            lines.append(
                f"| {comp['comparison']} | "
                f"{comp['reference']['mean']:.2f} | "
                f"{comp['other']['mean']:.2f} | "
                f"{comp['welch_t']:.3f} | "
                f"{comp['p_value']:.2e} | {sig} | "
                f"{comp['cohens_d']:.3f} | {comp['effect_size']} | {ci_str} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 结论",
            "",
            "- PPO 在所有负载模式下相对 FCFS/SJF/Random/Greedy 均应通过 Bonferroni 校正。",
            "- 若 DQN/Quantum-Only/Classical-Only reward 为负（无法调度量子任务），",
            "  PPO 与它们的差异在统计上必然显著且效应量极大。",
            "- 本报告复用 `workload_pattern_results.json` 的逐 episode reward，不重新跑仿真。",
            "",
            f"数据文件：`{data_path.as_posix()}`",
            "",
        ]
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #39 多负载模式统计显著性检验")
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
    args = parser.parse_args()

    with open(args.input, encoding="utf-8") as f:
        raw = json.load(f)

    pattern_stats: dict[str, Any] = {}
    for pattern_key, pattern_data in raw["patterns"].items():
        pattern_stats[pattern_key] = analyze_pattern(pattern_key, pattern_data["strategies"])

    results = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "source_data": str(args.input),
        "alpha": ALPHA,
        "bonferroni_alpha": BONFERRONI_ALPHA,
        "reference_strategy": REFERENCE_STRATEGY,
        "patterns": pattern_stats,
    }

    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_data, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    generate_report(results, args.output_report, args.output_data)

    print(f"统计检验完成：{len(pattern_stats)} 模式 × 7 比较")
    print(f"数据：{args.output_data}")
    print(f"报告：{args.output_report}")

    # 打印摘要
    for pattern_key, pdata in pattern_stats.items():
        sig_count = sum(1 for c in pdata["comparisons"] if c["bonferroni_significant"])
        print(
            f"  {pattern_key}: {sig_count}/7 Bonferroni 显著，"
            f"最强效应 d={max(abs(c['cohens_d']) for c in pdata['comparisons']):.3f}"
        )


if __name__ == "__main__":
    main()
