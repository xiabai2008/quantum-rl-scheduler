"""
读取 simulation_results_*.json，输出 Markdown 格式对比报告
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def generate_report(json_path: str, output_path: str | None = None) -> str:
    """
    从仿真结果 JSON 生成 Markdown 格式对比报告。

    Args:
        json_path   : 仿真结果 JSON 文件路径
        output_path : 输出的 Markdown 文件路径（可选，默认同目录下 strategy_comparison_report.md）

    Returns:
        str: 生成的 Markdown 报告内容
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    sorted_strategies = sorted(
        data.items(), key=lambda x: x[1].get("avg_reward", 0.0), reverse=True
    )

    report = "# 调度策略对比报告\n\n"
    report += f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    report += f"**数据来源**: `{os.path.basename(json_path)}`\n\n"

    report += "## 一、核心指标对比\n\n"
    report += "| 排名 | 策略 | 平均奖励 | 平均等待时间(步) | 完成率 | 量子利用率 | 经典利用率 | 平均执行时间 |\n"
    report += "|------|------|----------|-----------------|--------|-----------|----------|-------------|\n"

    for rank, (name, metrics) in enumerate(sorted_strategies, 1):
        avg_reward = metrics.get("avg_reward", 0.0)
        avg_wait = metrics.get("avg_wait_time", 0.0)
        completion = metrics.get("completion_rate", 0.0)
        qubit_util = metrics.get("qubit_utilization", 0.0)
        classical_util = metrics.get("classical_utilization", 0.0)
        avg_exec = metrics.get("avg_execution_time", 0.0)

        medal = ""
        if rank == 1:
            medal = " 🥇"
        elif rank == 2:
            medal = " 🥈"
        elif rank == 3:
            medal = " 🥉"

        report += f"| {rank} | **{name}**{medal} | {avg_reward:.2f} | {avg_wait:.2f} | "
        report += f"{completion:.2%} | {qubit_util:.2%} | {classical_util:.2%} | {avg_exec:.2f} |\n"

    report += "\n## 二、策略分析\n\n"

    best = sorted_strategies[0]
    report += f"### 2.1 最佳策略: {best[0]}\n\n"
    report += f"- 平均奖励: **{best[1].get('avg_reward', 0.0):.2f}**\n"
    report += f"- 完成率: **{best[1].get('completion_rate', 0.0):.2%}**\n"
    report += f"- 量子利用率: **{best[1].get('qubit_utilization', 0.0):.2%}**\n\n"

    if len(sorted_strategies) >= 2:
        dqn_entry = None
        random_entry = None
        for name, metrics in sorted_strategies:
            if "dqn" in name.lower():
                dqn_entry = (name, metrics)
            if "random" in name.lower():
                random_entry = (name, metrics)

        if dqn_entry and random_entry:
            dqn_reward = dqn_entry[1].get("avg_reward", 0.0)
            random_reward = random_entry[1].get("avg_reward", 0.0)
            diff = dqn_reward - random_reward
            diff_pct = (diff / abs(random_reward) * 100) if random_reward != 0 else 0.0

            report += "### 2.2 DQN vs Random 对比\n\n"
            report += f"- DQN 平均奖励: **{dqn_reward:.2f}**\n"
            report += f"- Random 平均奖励: **{random_reward:.2f}**\n"
            report += f"- 差值: **{diff:+.2f}** ({diff_pct:+.1f}%)\n"
            if diff > 0:
                report += "- ✅ **DQN 优于 Random**\n\n"
            else:
                report += "- ⚠️ **DQN 尚未超越 Random**\n\n"

        ppo_entry = None
        for name, metrics in sorted_strategies:
            if "ppo" in name.lower():
                ppo_entry = (name, metrics)
                break

        if ppo_entry and random_entry:
            ppo_reward = ppo_entry[1].get("avg_reward", 0.0)
            random_reward = random_entry[1].get("avg_reward", 0.0)
            diff = ppo_reward - random_reward
            diff_pct = (diff / abs(random_reward) * 100) if random_reward != 0 else 0.0

            report += "### 2.3 PPO vs Random 对比\n\n"
            report += f"- PPO 平均奖励: **{ppo_reward:.2f}**\n"
            report += f"- Random 平均奖励: **{random_reward:.2f}**\n"
            report += f"- 差值: **{diff:+.2f}** ({diff_pct:+.1f}%)\n"
            if diff > 0:
                report += "- ✅ **PPO 显著优于 Random**\n\n"
            else:
                report += "- ⚠️ **PPO 尚未超越 Random**\n\n"

    report += "## 三、各策略详细说明\n\n"

    strategy_descriptions = {
        "DQN": "深度 Q 网络强化学习策略，通过与环境交互学习最优调度决策。",
        "PPO": "近端策略优化强化学习策略，在连续动作空间和高维状态空间中表现更稳定。",
        "FCFS": "先来先服务策略，按任务到达顺序分配到经典资源。",
        "Random": "随机策略，随机选择经典/量子/混合执行方式。",
        "Quantum-Only": "仅量子资源策略，尽可能将任务分配到量子资源。",
        "Classical-Only": "仅经典资源策略，所有任务都分配到经典资源。",
        "Greedy": "贪心调度策略，基于当前资源利用率和任务紧急程度做局部最优选择。",
        "SJF": "最短作业优先策略，优先调度预估执行时间短的任务。",
    }

    for name, metrics in sorted_strategies:
        desc = strategy_descriptions.get(name, "自定义调度策略。")
        report += f"### {name}\n\n"
        report += f"{desc}\n\n"
        report += f"- 平均奖励: {metrics.get('avg_reward', 0.0):.2f}\n"
        report += f"- 平均等待时间: {metrics.get('avg_wait_time', 0.0):.2f} 步\n"
        report += f"- 任务完成率: {metrics.get('completion_rate', 0.0):.2%}\n"
        report += f"- 量子比特利用率: {metrics.get('qubit_utilization', 0.0):.2%}\n"
        report += f"- 经典资源利用率: {metrics.get('classical_utilization', 0.0):.2%}\n\n"

    report += "---\n\n"
    report += "*本报告由 `scripts/generate_report.py` 自动生成*\n"

    if output_path is None:
        output_path = os.path.join(os.path.dirname(json_path), "strategy_comparison_report.md")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"[报告生成] Markdown 报告已保存到: {output_path}")
    return report


def main():
    parser = argparse.ArgumentParser(
        description="从仿真结果 JSON 生成 Markdown 对比报告",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/generate_report.py results/simulation_results_20240101_120000.json
  python scripts/generate_report.py results/sim.json -o results/report.md
        """,
    )
    parser.add_argument(
        "json_path",
        type=str,
        help="仿真结果 JSON 文件路径",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="输出 Markdown 文件路径（默认同目录下 strategy_comparison_report.md）",
    )

    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"[错误] 文件不存在: {args.json_path}")
        sys.exit(1)

    generate_report(args.json_path, args.output)


if __name__ == "__main__":
    main()
