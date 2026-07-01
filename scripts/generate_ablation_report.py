#!/usr/bin/env python
"""
消融实验分析报告生成器 — 学术级别自动分析
Ablation Report Generator — Academic-Level Analysis

输入 ablation_study_TIMESTAMP.json，自动生成：
    1. LaTeX 兼容表格（含边际贡献、效应量）
    2. 多维度雷达图对比
    3. 贡献度排序 / 关键发现
    4. 消融决策树（各消融维度的最优/最差配置路由）

用法：
    python scripts/generate_ablation_report.py results/ablation_study_XXX.json
    python scripts/generate_ablation_report.py results/ablation_study_XXX.json -o report.md
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 延迟导入 matplotlib
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 中文字体配置
try:
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass


# =========================================================================
# 报告生成核心
# =========================================================================


def generate_ablation_report(json_path: str, output_path: str | None = None) -> str:
    """
    从消融实验 JSON 生成学术级别的分析报告。

    Args:
        json_path   : 消融实验结果 JSON 文件路径
        output_path : 输出 Markdown 路径（可选）

    Returns:
        str: 生成的 Markdown 报告全文
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    ts = data.get("timestamp", datetime.now().strftime("%Y%m%d_%H%M%S"))
    config = data.get("config", {})
    dimensions = data.get("dimensions", {})

    report = _build_report_header(ts, config)
    report += _build_executive_summary(dimensions)
    report += _build_dimension_details(dimensions)
    report += _build_cross_dimension_analysis(dimensions)
    report += _build_academic_tables(dimensions)
    report += _build_conclusions(dimensions)

    # 生成图表
    chart_paths = _generate_charts(dimensions, json_path)

    output_dir = os.path.dirname(json_path)
    if output_path is None:
        output_path = os.path.join(output_dir, f"ablation_report_{ts}.md")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[报告] 消融分析报告已保存: {output_path}")
    if chart_paths:
        print(f"[图表] 已生成 {len(chart_paths)} 张分析图表")

    return report


def _build_report_header(ts: str, config: dict) -> str:
    lines = [
        "# 多维消融实验分析报告",
        "",
        f"**生成时间**: {ts}",
        f"**实验配置**: {config.get('timesteps', 'N/A')} 步, "
        f"{len(config.get('seeds', []))} seeds, "
        f"DryRun={'是' if config.get('dry_run') else '否'}",
        f"**覆盖维度**: {', '.join(config.get('dimensions', []))}",
        "",
        "---",
        "",
    ]
    return "\n".join(lines)


def _build_executive_summary(dimensions: dict) -> str:
    """生成执行摘要，列出每个维度的最佳配置和关键发现。"""
    lines = [
        "## 执行摘要",
        "",
        "| 维度 | 最佳配置 | 最优均值 | 最差配置 | 效应量(Range) |",
        "|------|----------|----------|----------|---------------|",
    ]

    dim_names = {
        "D1": "算法组件",
        "D2": "状态空间",
        "D3": "奖励函数",
        "D4": "机器规模",
        "D5": "退火策略",
    }

    for dim_key, dim_data in dimensions.items():
        configs = dim_data.get("configs", [])
        if not configs:
            continue
        best = max(configs, key=lambda c: c.get("mean_reward", -999))
        worst = min(configs, key=lambda c: c.get("mean_reward", 999))
        rng = best.get("mean_reward", 0) - worst.get("mean_reward", 0)
        dim_label = dim_names.get(dim_key, dim_key)
        lines.append(
            f"| {dim_label} | **{best['name']}** | {best.get('mean_reward', 0):.1f} "
            f"| {worst['name']} | {rng:.1f} |"
        )

    lines.extend(["", "---", ""])
    return "\n".join(lines)


def _build_dimension_details(dimensions: dict) -> str:
    """为每个消融维度生成详细分析。"""
    lines = ["## 分维度详细分析", ""]

    dim_labels = {
        "D1": "### D1: 算法组件消融",
        "D2": "### D2: 状态空间消融",
        "D3": "### D3: 奖励函数消融",
        "D4": "### D4: 机器规模消融",
        "D5": "### D5: 退火策略消融",
    }

    dim_descriptions = {
        "D1": ("逐步移除算法组件（退火 → PPO → 量子），" "量化每个核心模块的边际贡献。"),
        "D2": ("从完整10维观测逐步裁剪到3维基线，" "分析增加状态信息的边际效益是否递减。"),
        "D3": (
            "对比四种奖励函数设计（标准/加速比加权/公平性/能耗感知），"
            "验证哪种设计最有利于学习效率。"
        ),
        "D4": (
            "从单台287q机器扩展到三台异构机器集群，"
            "量化多机器调度带来的吞吐量提升和边际递减效应。"
        ),
        "D5": (
            "对比无退火/模拟退火/真机退火三种策略，" "量化退火加速对RL收敛速度和最终质量的增益。"
        ),
    }

    for dim_key in ["D1", "D2", "D3", "D4", "D5"]:
        if dim_key not in dimensions:
            continue
        dim_data = dimensions[dim_key]
        configs = dim_data.get("configs", [])
        if not configs:
            continue

        lines.append(dim_labels.get(dim_key, f"### {dim_key}"))
        lines.append("")
        lines.append(dim_descriptions.get(dim_key, ""))
        lines.append("")

        # 结果表格
        lines.append("| 配置 | Mean | Std | Max | Min | 中位数 | 相对% |")
        lines.append("|------|------|-----|-----|-----|--------|-------|")
        for c in configs:
            rel = c.get("relative_to_baseline_pct", 0)
            lines.append(
                f"| {c['name']} | {c.get('mean_reward', 0):.1f} "
                f"| {c.get('std_reward', 0):.1f} "
                f"| {c.get('max_reward', 0):.1f} "
                f"| {c.get('min_reward', 0):.1f} "
                f"| {c.get('median_reward', 0):.1f} "
                f"| {rel:.0f}% |"
            )
        lines.append("")

        # 边际贡献
        marginal = dim_data.get("summary", {}).get("marginal_contributions", {})
        if marginal:
            lines.append("**边际贡献**:")
            for label, val in marginal.items():
                lines.append(f"- {label}: **{val:+.1f}**")
            lines.append("")

        # 关键发现
        lines.append(_derive_key_findings(dim_key, configs))
        lines.append("")

    return "\n".join(lines)


def _derive_key_findings(dim_key: str, configs: list) -> str:
    """基于数据自动推导关键发现。"""
    if len(configs) < 2:
        return ""

    best = max(configs, key=lambda c: c.get("mean_reward", -999))
    worst = min(configs, key=lambda c: c.get("mean_reward", 999))
    delta = best.get("mean_reward", 0) - worst.get("mean_reward", 0)
    pct = abs(delta / max(abs(worst.get("mean_reward", 1)), 1e-8)) * 100

    findings = [
        "> **关键发现**:",
        f"> - 最优配置 **{best['name']}** (mean={best.get('mean_reward', 0):.1f})",
        f"> - 相对于最差配置 **{worst['name']}** (mean={worst.get('mean_reward', 0):.1f}) "
        f"提升 **{delta:+.1f}** ({pct:.1f}%)",
    ]

    # 维度特定发现
    if dim_key == "D1":
        findings.append("> - 退火模块贡献 ≈ 移除退火前后的reward差值")
        findings.append("> - RL学习贡献 ≈ PPO-Only vs Pure-Annealing 的reward差值")
    elif dim_key == "D2":
        findings.append("> - 维度增益是否递减：观察相邻配置之间的reward增量是否变小")
    elif dim_key == "D4":
        findings.append("> - 多机器扩展效率：观察每增加一台机器的边际增益是否递减")

    return "\n".join(findings)


def _build_cross_dimension_analysis(dimensions: dict) -> str:
    """跨维度综合分析。"""
    lines = [
        "## 跨维度综合分析",
        "",
        "### 贡献度排序",
        "",
        "下表汇总所有维度的效应量（最优-最差），按影响力从大到小排序：",
        "",
        "| 排名 | 维度 | 效应量(Range) | 影响力评级 |",
        "|------|------|---------------|------------|",
    ]

    dim_ranges = []
    for dim_key, dim_data in dimensions.items():
        configs = dim_data.get("configs", [])
        if len(configs) >= 2:
            best = max(configs, key=lambda c: c.get("mean_reward", -999))
            worst = min(configs, key=lambda c: c.get("mean_reward", 999))
            rng = best.get("mean_reward", 0) - worst.get("mean_reward", 0)
            dim_ranges.append((dim_key, rng))

    dim_ranges.sort(key=lambda x: abs(x[1]), reverse=True)

    for rank, (dim_key, rng) in enumerate(dim_ranges, 1):
        if abs(rng) > 1000:
            rating = "非常显著"
        elif abs(rng) > 500:
            rating = "显著"
        elif abs(rng) > 100:
            rating = "中等"
        else:
            rating = "轻微"
        lines.append(f"| {rank} | {dim_key} | {rng:.1f} | {rating} |")

    lines.extend(
        [
            "",
            "### 消融决策树",
            "",
            "以下决策树总结了沿每个消融维度的最优路径：",
            "",
            "```",
        ]
    )

    for dim_key, dim_data in dimensions.items():
        configs = dim_data.get("configs", [])
        if not configs:
            continue
        best = max(configs, key=lambda c: c.get("mean_reward", -999))
        lines.append(f"  {dim_key}: {' > '.join(c['name'] for c in configs)}")
        lines.append(f"    => 最佳: {best['name']}")

    lines.extend(
        [
            "```",
            "",
            "---",
            "",
        ]
    )

    return "\n".join(lines)


def _build_academic_tables(dimensions: dict) -> str:
    """生成 LaTeX 兼容的学术表格。"""
    lines = [
        "## 学术表格（LaTeX 兼容）",
        "",
        "以下表格可直接嵌入 LaTeX 论文/技术白皮书。",
        "",
    ]

    for dim_key, dim_data in dimensions.items():
        configs = dim_data.get("configs", [])
        if not configs:
            continue
        dim_label = dim_data.get("dimension", dim_key)

        lines.append(f"### 表: {dim_label}")
        lines.append("")
        lines.append("\\begin{table}[htbp]")
        lines.append("  \\centering")
        lines.append(
            "  \\caption{"
            + f"{dim_label}消融实验结果 (mean ± std over {configs[0].get('num_seeds', 0)} seeds)"
            + "}"
        )
        lines.append("  \\label{tab:" + dim_key.lower() + "}")
        lines.append("  \\begin{tabular}{lrrr}")
        lines.append("    \\toprule")
        lines.append("    Configuration & Mean Reward & Std & Effect Size \\\\")
        lines.append("    \\midrule")

        baseline = configs[0]["mean_reward"] if configs else 1.0
        for c in configs:
            effect = (c.get("mean_reward", 0) - baseline) / max(abs(baseline), 1e-8) * 100
            lines.append(
                f"    {c['name']} & {c.get('mean_reward', 0):.2f} & "
                f"$\\pm${c.get('std_reward', 0):.2f} & {effect:+.1f}\\% \\\\"
            )

        lines.append("    \\bottomrule")
        lines.append("  \\end{tabular}")
        lines.append("\\end{table}")
        lines.append("")

    return "\n".join(lines)


def _build_conclusions(dimensions: dict) -> str:
    """综合结论和建议。"""
    # 收集所有配置的最佳
    all_best = []
    for dim_key, dim_data in dimensions.items():
        configs = dim_data.get("configs", [])
        if configs:
            all_best.append((dim_key, max(configs, key=lambda c: c.get("mean_reward", -999))))

    lines = [
        "## 结论与建议",
        "",
        "### 综合推荐配置",
        "",
        "基于消融实验结果，推荐的系统配置组合：",
        "",
    ]

    for dim_key, best in all_best:
        lines.append(f"- **{dim_key}**: 使用 **{best['name']}**")

    lines.extend(
        [
            "",
            "### 对参赛材料的建议",
            "",
            "1. **核心创新点佐证**: 消融实验数据可作为'双向赋能'创新性的定量证据",
            "   - D1数据: 退火的边际贡献量化了'量子赋能AI'的价值",
            "   - D2数据: 状态空间维度分析量化了'AI赋能量子'的信息增益",
            "2. **技术白皮书引用**: LaTeX 表格可直接嵌入技术白皮书",
            "3. **答辩Q&A储备**: 效应量排序可作为'哪个组件最重要'问题的定量回答",
            "",
            "### 后续研究方向",
            "",
            "1. 在真机上进行 D5 (退火策略) 的完整消融，获取真机退火 vs 模拟退火的真实差距",
            "2. 扩展 D2 到更多维度（如20维），探索信息增益的饱和点",
            "3. 设计交互消融实验（如 D1×D4），分析算法与机器规模之间的交互效应",
            "",
            "---",
            "",
            f"*本报告由 `scripts/generate_ablation_report.py` 自动生成于 "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        ]
    )

    return "\n".join(lines)


# =========================================================================
# 图表生成
# =========================================================================


def _generate_charts(dimensions: dict, json_path: str) -> list[str]:
    """生成多维度图表并保存到 results/。"""
    chart_paths = []
    output_dir = os.path.dirname(json_path)
    ts = os.path.basename(json_path).replace("ablation_study_", "").replace(".json", "")

    try:
        # 图表1: 效应量条形图
        dim_names = []
        best_names = []
        ranges = []

        for dim_key in ["D1", "D2", "D3", "D4", "D5"]:
            if dim_key not in dimensions:
                continue
            configs = dimensions[dim_key].get("configs", [])
            if len(configs) < 2:
                continue
            best = max(configs, key=lambda c: c.get("mean_reward", -999))
            worst = min(configs, key=lambda c: c.get("mean_reward", 999))
            dim_names.append(dim_key)
            best_names.append(best["name"])
            ranges.append(best.get("mean_reward", 0) - worst.get("mean_reward", 0))

        if dim_names:
            fig, ax = plt.subplots(figsize=(10, 5))
            colors = [f"C{i}" for i in range(len(ranges))]
            bars = ax.barh(dim_names, ranges, color=colors, alpha=0.8)

            for bar, val, best_name in zip(bars, ranges, best_names, strict=False):
                ax.text(
                    bar.get_width() + max(ranges) * 0.02,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.0f} ({best_name})",
                    va="center",
                    fontsize=9,
                )

            ax.set_xlabel("Effect Size (Best - Worst)")
            ax.set_title("Ablation Impact Ranking\n(larger = bigger ablation effect)")
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()

            chart1 = os.path.join(output_dir, f"ablation_impact_{ts}.png")
            fig.savefig(chart1, dpi=150, bbox_inches="tight")
            plt.close(fig)
            chart_paths.append(chart1)

        # 图表2: 每个维度的对比条形图
        if len(dimensions) >= 2:
            fig, axes = plt.subplots(1, len(dimensions), figsize=(5 * len(dimensions), 5))
            if len(dimensions) == 1:
                axes = [axes]

            for idx, (dim_key, dim_data) in enumerate(dimensions.items()):
                ax = axes[idx]
                configs = dim_data.get("configs", [])
                names = [c["name"] for c in configs]
                means = [c.get("mean_reward", 0) for c in configs]
                stds = [c.get("std_reward", 0) for c in configs]

                bars = ax.bar(
                    range(len(names)),
                    means,
                    yerr=stds,
                    capsize=5,
                    color=plt.cm.Set2(np.linspace(0, 1, len(names))),
                )
                ax.set_xticks(range(len(names)))
                ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
                ax.set_title(dim_key, fontsize=10)
                ax.grid(True, alpha=0.3, axis="y")

            plt.tight_layout()
            chart2 = os.path.join(output_dir, f"ablation_dimensions_{ts}.png")
            fig.savefig(chart2, dpi=150, bbox_inches="tight")
            plt.close(fig)
            chart_paths.append(chart2)

    except Exception as e:
        print(f"  [WARN] 图表生成失败: {e}")

    return chart_paths


# =========================================================================
# CLI
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="消融实验分析报告生成器",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("json_path", help="消融实验结果 JSON 文件路径")
    parser.add_argument(
        "-o", "--output", default=None, help="输出 Markdown 路径（默认同目录自动命名）"
    )
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"[错误] 文件不存在: {args.json_path}")
        sys.exit(1)

    generate_ablation_report(args.json_path, args.output)


if __name__ == "__main__":
    main()
