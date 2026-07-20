#!/usr/bin/env python3
"""
Benchmark 性能趋势可视化工具

读取 results/benchmark_history.jsonl，生成性能趋势图用于答辩材料。

使用示例：
    python scripts/ci/visualize_benchmark_trend.py
    python scripts/ci/visualize_benchmark_trend.py --output results/reports/performance_trend.png
    python scripts/ci/visualize_benchmark_trend.py --history results/benchmark_history.jsonl --dpi 150

作者：量子RL调度系统团队
日期：2026-07-20
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_history(history_path: str) -> list[dict[str, Any]]:
    """加载 benchmark 历史记录。

    Args:
        history_path: JSONL 历史文件路径

    Returns:
        记录列表，按时间排序
    """
    path = Path(history_path)
    if not path.exists():
        print(f"❌ 历史文件不存在: {history_path}", file=sys.stderr)
        return []

    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def generate_trend_chart(
    records: list[dict[str, Any]],
    output_path: str = "results/reports/performance_trend.png",
    dpi: int = 150,
) -> str:
    """生成性能趋势图。

    Args:
        records: 历史记录列表
        output_path: 输出图片路径
        dpi: 图片 DPI

    Returns:
        输出文件路径
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
    except ImportError:
        print("⚠️  matplotlib 未安装，跳过图片生成。尝试生成 ASCII 报告...")
        return _generate_ascii_report(records, output_path)

    if len(records) < 2:
        print(f"⚠️  只有 {len(records)} 条记录，需要至少 2 条才能画趋势图", file=sys.stderr)
        return ""

    # 提取所有 benchmark 名称
    all_names: list[str] = []
    for rec in records:
        for bench in rec.get("benchmarks", []):
            name = bench["name"]
            if name not in all_names:
                all_names.append(name)

    if not all_names:
        print("⚠️  没有 benchmark 数据", file=sys.stderr)
        return ""

    # 解析时间序列数据
    timestamps: list[datetime] = []
    series: dict[str, list[float]] = {name: [] for name in all_names}
    commits: list[str] = []

    for rec in records:
        try:
            ts = datetime.fromisoformat(rec["timestamp"])
        except (ValueError, KeyError):
            continue
        timestamps.append(ts)
        commits.append(rec.get("commit", "?")[:7])

        rec_benchmarks = {b["name"]: b["mean"] for b in rec.get("benchmarks", [])}
        for name in all_names:
            series[name].append(rec_benchmarks.get(name, float("nan")))

    # 创建图表
    n = len(all_names)
    cols = min(n, 2)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 4 * rows), squeeze=False)
    fig.suptitle(
        "Quantum-RL Scheduler — Benchmark Performance Trends",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )

    for i, name in enumerate(all_names):
        row, col = i // cols, i % cols
        ax = axes[row][col]

        y = series[name]
        x = timestamps

        # 过滤 NaN
        valid = [(t, v) for t, v in zip(x, y) if v == v]
        if not valid:
            ax.set_title(name, fontsize=9)
            ax.text(0.5, 0.5, "无数据", ha="center", va="center", transform=ax.transAxes)
            continue

        tx, ty = zip(*valid)
        ax.plot(tx, ty, "o-", linewidth=1.5, markersize=4, color="#2c7fb8")

        # 标注增长率
        if len(ty) >= 2 and ty[0] > 0:
            pct = ((ty[-1] - ty[0]) / ty[0]) * 100
            color = "#e31a1c" if pct > 5 else ("#33a02c" if pct < -5 else "#666666")
            ax.set_title(
                f"{name}\n({pct:+.1f}% vs baseline, n={len(valid)})",
                fontsize=9,
                color=color,
            )
        else:
            ax.set_title(f"{name} (n={len(valid)})", fontsize=9)

        ax.set_ylabel("Mean (s)")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(True, alpha=0.3)

        # X axis formatting (keep English to avoid CJK font issues on CI)
        ax.set_xlabel("Timestamp")
        try:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        except Exception:
            pass

    # 隐藏多余子图
    for j in range(n, rows * cols):
        row, col = j // cols, j % cols
        axes[row][col].set_visible(False)

    fig.tight_layout(rect=[0, 0, 1, 0.95])

    # 底部统计摘要
    total_commits = len(set(commits))
    first_ts = timestamps[0].strftime("%Y-%m-%d") if timestamps else "?"
    last_ts = timestamps[-1].strftime("%Y-%m-%d") if timestamps else "?"
    fig.text(
        0.5, 0.01,
        f"{len(records)} runs · {total_commits} commits · {first_ts} ~ {last_ts}",
        ha="center", fontsize=8, color="#888888",
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"✅ 趋势图已保存: {out}")
    return str(out)


def _generate_ascii_report(records: list[dict[str, Any]], output_path: str) -> str:
    """降级方案：生成 ASCII 文本趋势报告。"""
    if len(records) < 2:
        return ""

    # 简单计算每项的变化
    first = records[0].get("benchmarks", [])
    last = records[-1].get("benchmarks", [])
    first_map = {b["name"]: b["mean"] for b in first}
    last_map = {b["name"]: b["mean"] for b in last}

    lines = [
        "# Benchmark 性能趋势 (文本报告)",
        "",
        f"记录数: {len(records)}",
        f"首次: {records[0].get('timestamp', '?')[:19]}",
        f"末次: {records[-1].get('timestamp', '?')[:19]}",
        "",
        "| Benchmark | 首次 (s) | 末次 (s) | 变化 |",
        "|-----------|----------|----------|------|",
    ]

    for name in sorted(first_map):
        f_val = first_map[name]
        l_val = last_map.get(name, float("nan"))
        if f_val > 0:
            pct = ((l_val - f_val) / f_val) * 100
            lines.append(f"| {name} | {f_val:.6f} | {l_val:.6f} | {pct:+.1f}% |")

    out = Path(output_path).with_suffix(".md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"✅ 文本报告已保存: {out}")
    return str(out)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark 性能趋势可视化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--history",
        type=str,
        default="results/benchmark_history.jsonl",
        help="历史记录文件路径",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/reports/performance_trend.png",
        help="输出图片路径",
    )
    parser.add_argument("--dpi", type=int, default=150, help="图片 DPI")

    args = parser.parse_args()

    records = load_history(args.history)
    print(f"📊 加载 {len(records)} 条历史记录")

    if len(records) < 2:
        print(f"⚠️  需要至少 2 条记录才能生成趋势图，当前 {len(records)} 条")
        sys.exit(0)

    result = generate_trend_chart(records, args.output, args.dpi)
    if result:
        print(f"\n✅ 完成: {result}")
    else:
        print("\n❌ 生成失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
