#!/usr/bin/env python
"""
帕累托前沿可视化
Pareto Frontier Visualization for Multi-Objective RL

读取训练结果 JSON，生成 3 张可视化图：
    1. 3D 散点图：3 个目标在三维空间中的分布
    2. 2D 投影图：3 组两两投影（throughput-balance, throughput-quality, balance-quality）
    3. 权重雷达图：不同权重组合在 3 个目标上的雷达图对比

使用示例:
    python scripts/compare_pareto.py --input results/multi_objective/mo_training_results.json
    python scripts/compare_pareto.py --input results/multi_objective/mo_training_results.json --output results/pareto
"""

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ============================================================================
# 命令行参数解析
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="帕累托前沿可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default="results/multi_objective/mo_training_results.json",
        help="训练结果 JSON 文件路径",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="results/pareto",
        help="输出目录（默认: results/pareto）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="图片 DPI（默认: 150）",
    )
    return parser.parse_args()


# ============================================================================
# 数据加载
# ============================================================================
def load_results(input_path: str) -> list[dict[str, Any]]:
    """
    加载训练结果 JSON。

    Args:
        input_path: JSON 文件路径

    Returns:
        list: 结果列表
    """
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    if "results" in data:
        return data["results"]
    return data


# ============================================================================
# 帕累托前沿计算
# ============================================================================
def find_pareto_front(points: np.ndarray) -> np.ndarray:
    """
    找出帕累托前沿（所有目标都最大化）。

    Args:
        points: (N, 3) 数组，每行是 [throughput, balance, quality]

    Returns:
        np.ndarray: 布尔数组，True 表示帕累托前沿点
    """
    n = len(points)
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            # 如果 j 在所有维度上都不差于 i，且至少一个维度严格优于 i
            if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
                is_pareto[i] = False
                break
    return is_pareto


# ============================================================================
# 可视化函数
# ============================================================================
def plot_3d_scatter(
    results: list[dict[str, Any]],
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    生成 3D 散点图：throughput × balance × quality。

    Args:
        results: 训练结果列表
        output_path: 输出文件路径
        dpi: 图片 DPI
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    # 提取数据
    points = []
    labels = []
    colors_map = {
        "throughput_heavy": "#e74c3c",
        "balance_heavy": "#2ecc71",
        "quality_heavy": "#3498db",
        "balanced": "#9b59b6",
    }

    for r in results:
        mo = r.get("mo_metrics", {})
        if not mo:
            continue
        t = mo.get("avg_throughput", 0)
        b = mo.get("avg_balance", 0)
        q = mo.get("avg_quality", 0)
        points.append([t, b, q])
        labels.append(r.get("weight_preset", "unknown"))

    if not points:
        print("警告: 没有可用的多目标数据")
        return

    points = np.array(points)
    pareto_mask = find_pareto_front(points)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 绘制所有点
    for _i, (preset, color) in enumerate(colors_map.items()):
        idxs = [j for j, lbl in enumerate(labels) if lbl == preset]
        if not idxs:
            continue
        pts = points[idxs]
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            pts[:, 2],
            c=color,
            label=preset,
            s=80,
            alpha=0.85,
            edgecolors="black",
            linewidth=0.5,
        )

    # 标记帕累托前沿
    pareto_pts = points[pareto_mask]
    if len(pareto_pts) > 0:
        ax.scatter(
            pareto_pts[:, 0],
            pareto_pts[:, 1],
            pareto_pts[:, 2],
            c="gold",
            marker="*",
            s=200,
            alpha=1.0,
            edgecolors="black",
            linewidth=1.0,
            label="Pareto Front",
        )

    ax.set_xlabel("Throughput", fontsize=12)
    ax.set_ylabel("Balance", fontsize=12)
    ax.set_zlabel("Quality", fontsize=12)
    ax.set_title("Multi-Objective Pareto Frontier (3D)", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"3D 散点图已保存至: {output_path}")


def plot_2d_projections(
    results: list[dict[str, Any]],
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    生成 2D 投影图：3 组两两投影。

    Args:
        results: 训练结果列表
        output_path: 输出文件路径
        dpi: 图片 DPI
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = {}
    colors_map = {
        "throughput_heavy": "#e74c3c",
        "balance_heavy": "#2ecc71",
        "quality_heavy": "#3498db",
        "balanced": "#9b59b6",
    }

    for r in results:
        mo = r.get("mo_metrics", {})
        if not mo:
            continue
        preset = r.get("weight_preset", "unknown")
        if preset not in points:
            points[preset] = []
        points[preset].append(
            [
                mo.get("avg_throughput", 0),
                mo.get("avg_balance", 0),
                mo.get("avg_quality", 0),
            ]
        )

    _fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    projections = [
        (0, 1, "Throughput", "Balance", "Throughput vs Balance"),
        (0, 2, "Throughput", "Quality", "Throughput vs Quality"),
        (1, 2, "Balance", "Quality", "Balance vs Quality"),
    ]

    for ax_idx, (dim_x, dim_y, xlabel, ylabel, title) in enumerate(projections):
        ax = axes[ax_idx]
        for preset, pts in points.items():
            pts_arr = np.array(pts)
            color = colors_map.get(preset, "#888888")
            ax.scatter(
                pts_arr[:, dim_x],
                pts_arr[:, dim_y],
                c=color,
                label=preset,
                s=60,
                alpha=0.8,
                edgecolors="black",
                linewidth=0.3,
            )

        ax.set_xlabel(xlabel, fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=8, loc="lower right")

    plt.suptitle("Multi-Objective 2D Projections", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"2D 投影图已保存至: {output_path}")


def plot_radar(
    results: list[dict[str, Any]],
    output_path: str,
    dpi: int = 150,
) -> None:
    """
    生成权重雷达图：不同权重组合在 3 个目标上的表现。

    Args:
        results: 训练结果列表
        output_path: 输出文件路径
        dpi: 图片 DPI
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 按权重预设聚合（取平均）
    aggregated = {}
    for r in results:
        mo = r.get("mo_metrics", {})
        if not mo:
            continue
        preset = r.get("weight_preset", "unknown")
        if preset not in aggregated:
            aggregated[preset] = {"throughput": [], "balance": [], "quality": []}
        aggregated[preset]["throughput"].append(mo.get("avg_throughput", 0))
        aggregated[preset]["balance"].append(mo.get("avg_balance", 0))
        aggregated[preset]["quality"].append(mo.get("avg_quality", 0))

    if not aggregated:
        print("警告: 没有可用的聚合数据")
        return

    # 归一化到 [0, 1]（对 balance 和 quality 做平移）
    max_vals = {"throughput": 0, "balance": 1e-6, "quality": 1e-6}
    for preset, vals in aggregated.items():
        for k in ["throughput", "balance", "quality"]:
            raw = np.mean(vals[k])
            # balance 和 quality 是负数，平移使其为正
            if k in ("balance", "quality") and raw < 0:
                raw = raw + 1.0  # 从 [-1,0] 映射到 [0,1]
            max_vals[k] = max(max_vals[k], raw)

    categories = ["Throughput", "Balance", "Quality"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合

    _fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    colors_map = {
        "throughput_heavy": "#e74c3c",
        "balance_heavy": "#2ecc71",
        "quality_heavy": "#3498db",
        "balanced": "#9b59b6",
    }

    for preset, vals in aggregated.items():
        values = []
        for k in ["throughput", "balance", "quality"]:
            raw = np.mean(vals[k])
            if k in ("balance", "quality") and raw < 0:
                raw = raw + 1.0
            # 归一化
            norm = raw / max(max_vals[k], 1e-6)
            values.append(norm)
        values += values[:1]  # 闭合

        color = colors_map.get(preset, "#888888")
        ax.fill(angles, values, alpha=0.15, color=color)
        ax.plot(angles, values, "o-", linewidth=2, color=color, label=preset, markersize=6)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8", "1.0"], fontsize=8)
    ax.set_title("Multi-Objective Radar Chart", fontsize=14, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"雷达图已保存至: {output_path}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    args = parse_args()

    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        print("请先运行 train_multi_objective.py 生成训练结果")
        sys.exit(1)

    os.makedirs(args.output, exist_ok=True)

    results = load_results(args.input)
    print(f"加载了 {len(results)} 条结果记录")

    # 生成 3 张图
    plot_3d_scatter(
        results,
        os.path.join(args.output, "pareto_3d_scatter.png"),
        dpi=args.dpi,
    )
    plot_2d_projections(
        results,
        os.path.join(args.output, "pareto_2d_projections.png"),
        dpi=args.dpi,
    )
    plot_radar(
        results,
        os.path.join(args.output, "pareto_radar.png"),
        dpi=args.dpi,
    )

    print(f"\n所有可视化图片已保存至: {args.output}/")
    print("生成的文件:")
    print("  - pareto_3d_scatter.png     (3D 散点图)")
    print("  - pareto_2d_projections.png (2D 投影图)")
    print("  - pareto_radar.png          (权重雷达图)")


if __name__ == "__main__":
    main()
