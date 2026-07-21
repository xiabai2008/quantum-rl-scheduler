#!/usr/bin/env python
"""
统计显著性检验 CLI 入口
Statistical Significance Testing CLI

从 JSON 文件加载多策略多次运行的奖励数据，执行统计显著性两两比较，
生成中文 Markdown 报告到 results/reports/statistical_significance.md。

输入 JSON 格式（{策略名: [多次运行奖励列表]}）：
    {
        "PPO": [2746.94, 2801.4, 2826.9, ...],
        "FCFS": [1462.48, 1450.2, ...],
        "Random": [1275.91, 1284.0, ...]
    }

用法：
    python scripts/evaluation/statistical_significance.py \\
        --input results/rewards.json \\
        --output results/reports/statistical_significance.md \\
        --alpha 0.05
"""

import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click

from src.utils.stats_significance import compare_strategies


def _load_rewards(input_path: Path) -> dict[str, list[float]]:
    """从 JSON 文件加载策略奖励数据

    Args:
        input_path: JSON 文件路径，顶层为 {策略名: [奖励列表]}

    Returns:
        ``{策略名: [float, ...]}`` 字典；非列表/非数值项会被跳过并告警

    Raises:
        click.ClickException: 顶层非对象，或可用策略不足 2 个
    """
    with input_path.open(encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, dict):
        raise click.ClickException("输入 JSON 顶层必须是对象 {策略名: [奖励列表]}")

    cleaned: dict[str, list[float]] = {}
    for name, rewards in raw.items():
        if not isinstance(rewards, list):
            click.echo(f"警告：策略 {name!r} 的奖励不是列表，已跳过", err=True)
            continue
        try:
            cleaned[name] = [float(r) for r in rewards]
        except (TypeError, ValueError):
            click.echo(f"警告：策略 {name!r} 含非数值奖励，已跳过", err=True)
            continue

    if len(cleaned) < 2:
        raise click.ClickException("至少需要 2 个策略才能执行比较")
    return cleaned


def _generate_markdown_report(
    data: dict[str, list[float]],
    results: dict[str, dict[str, Any]],
    alpha: float,
    input_path: Path,
) -> str:
    """生成中文 Markdown 报告

    Args:
        data: 原始策略奖励数据
        results: compare_strategies 返回的结果字典
        alpha: 显著性水平
        input_path: 输入文件路径（用于报告头部引用）

    Returns:
        Markdown 字符串
    """
    lines: list[str] = []
    lines.append("# 策略对比统计显著性检验报告")
    lines.append("")
    lines.append(f"> **数据来源**: `{input_path}`")
    lines.append(f"> **显著性水平 α**: {alpha}")

    # 取第一个对比的校正信息（所有对比一致）
    first = next(iter(results.values()))
    n_comp = first["n_comparisons"]
    bonf = first["bonferroni_alpha"]
    lines.append(f"> **比较次数**: {n_comp}（Bonferroni 校正后 α = {bonf:.4f}）")
    lines.append("")
    lines.append("---")
    lines.append("")

    # 一、各策略奖励统计
    lines.append("## 一、各策略奖励统计")
    lines.append("")
    lines.append("| 策略 | 样本数 | 平均奖励 | 标准差 | 最小值 | 最大值 |")
    lines.append("|:--|:--:|:--:|:--:|:--:|:--:|")
    for name, rewards in data.items():
        n = len(rewards)
        if n == 0:
            lines.append(f"| {name} | 0 | - | - | - | - |")
            continue
        mean = statistics.mean(rewards)
        std = statistics.stdev(rewards) if n >= 2 else 0.0
        lines.append(
            f"| {name} | {n} | {mean:.2f} | {std:.2f} | {min(rewards):.2f} | {max(rewards):.2f} |"
        )
    lines.append("")

    # 二、两两比较结果
    lines.append("## 二、两两比较结果")
    lines.append("")
    lines.append(
        "| 对比 | 检验方法 | 统计量 | p 值 | 显著? | 效应量 | 均值差 | 95% CI | 提升% 95% CI |"
    )
    lines.append("|:--|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
    for pair, info in results.items():
        ci = f"[{info['ci_lower']:.2f}, {info['ci_upper']:.2f}]"
        info.get("improvement_pct", float("nan"))
        imp_lo = info.get("improvement_pct_ci_lower", float("nan"))
        imp_hi = info.get("improvement_pct_ci_upper", float("nan"))
        if math.isnan(imp_lo) or math.isnan(imp_hi):
            imp_ci_str = "N/A"
        else:
            imp_ci_str = f"[{imp_lo:+.1f}%, {imp_hi:+.1f}%]"
        sig = "✅ 是" if info["significant"] else "❌ 否"
        lines.append(
            f"| {pair} | {info['test']} | {info['statistic']:.4f} | "
            f"{info['p_value']:.4g} | {sig} | "
            f"{info['effect_size_type']}={info['effect_size']:.4f} | "
            f"{info['mean_diff']:.2f} | {ci} | {imp_ci_str} |"
        )
    lines.append("")

    # 三、详细解释
    lines.append("## 三、详细解释")
    lines.append("")
    for pair, info in results.items():
        lines.append(f"### {pair}")
        lines.append("")
        lines.append(f"> {info['interpretation']}")
        lines.append("")

    # 四、检验方法说明
    lines.append("## 四、检验方法说明")
    lines.append("")
    lines.append("- **正态性检验**：n < 50 使用 Shapiro-Wilk，n ≥ 50 使用 D'Agostino K²")
    lines.append("- **方差齐性检验**：Levene 检验")
    lines.append("- **检验选择**：")
    lines.append("  - 两组均正态且方差齐 → 独立样本 t 检验")
    lines.append("  - 两组均正态但方差不齐 → Welch t 检验")
    lines.append("  - 任一组非正态 → Mann-Whitney U 检验")
    lines.append("- **效应量**：正态用 Cohen's d，非参数用 rank-biserial correlation")
    lines.append("- **多重比较校正**：Bonferroni（校正 α = α / 比较次数）")
    lines.append("- **置信区间**：均值差的 95% CI")
    lines.append("- **Cohen's d 等级**：< 0.2 可忽略，0.2-0.5 小，0.5-0.8 中，≥ 0.8 大")
    lines.append("- **rank-biserial 等级**：< 0.1 可忽略，0.1-0.3 小，0.3-0.5 中，≥ 0.5 大")
    lines.append("")
    lines.append("---")
    lines.append(f"*报告自动生成 | 数据源: {input_path}*")

    return "\n".join(lines)


def _print_summary(results: dict[str, dict[str, Any]], alpha: float) -> None:
    """打印检验摘要到控制台

    Args:
        results: compare_strategies 返回的结果字典
        alpha: 显著性水平
    """
    click.echo("")
    click.echo("=" * 60)
    click.echo("统计显著性检验摘要")
    click.echo("=" * 60)
    sig_count = sum(1 for r in results.values() if r["significant"])
    click.echo(
        f"共 {len(results)} 次两两比较，其中 {sig_count} 次显著（Bonferroni 校正后，α={alpha}）"
    )
    click.echo("")
    for pair, info in results.items():
        sig = "✅" if info["significant"] else "❌"
        click.echo(
            f"  {sig} {pair}: {info['test']}, "
            f"p={info['p_value']:.4g}, {info['effect_size_type']}={info['effect_size']:.4f}"
        )
    click.echo("=" * 60)


@click.command(name="statistical-significance")
@click.option(
    "--input",
    "input_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="输入 JSON 文件路径，格式为 {策略名: [多次运行奖励列表]}",
)
@click.option(
    "--output",
    "output_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("results/reports/statistical_significance.md"),
    show_default=True,
    help="输出 Markdown 报告路径",
)
@click.option(
    "--alpha",
    type=float,
    default=0.05,
    show_default=True,
    help="显著性水平 α（默认 0.05）",
)
def main(input_path: Path, output_path: Path, alpha: float) -> None:
    """对策略对比实验数据执行统计显著性检验并生成 Markdown 报告

    自动根据数据分布选择 t 检验 / Welch t / Mann-Whitney U，
    计算 Cohen's d 或 rank-biserial 效应量，并做 Bonferroni 多重比较校正。
    """
    data = _load_rewards(input_path)
    click.echo(f"已加载 {len(data)} 个策略：{', '.join(data.keys())}")

    results = compare_strategies(data, alpha=alpha)
    if not results:
        raise click.ClickException("未能生成任何比较结果")

    report = _generate_markdown_report(data, results, alpha, input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    click.echo(f"报告已生成：{output_path}")

    _print_summary(results, alpha)


if __name__ == "__main__":
    main()
