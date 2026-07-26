#!/usr/bin/env python
"""
等待惩罚权重扫描实验（Issue #120）

测试不同 REWARD_WAIT_OVER_THRESHOLD 值对 PPO 等待时间的影响，
寻找总奖励与等待时间之间的 Pareto 最优配置。

实验设计：
    - 4 种 wait_penalty 配置：-0.1（基线）, -0.3, -0.5, -1.0
    - 每种配置训练 PPO 50000 步，3 个 seeds
    - 评估 10 个 episodes：总奖励、平均等待时间、完成数
    - FCFS 基线对比
    - 生成 Pareto 前沿图和 Markdown 报告

用法:
    python scripts/evaluation/wait_penalty_sweep.py
    python scripts/evaluation/wait_penalty_sweep.py --timesteps 50000 --seeds 42 123 456
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import matplotlib

matplotlib.use("Agg")
# ============================================================================
# 日志配置
# ============================================================================
import logging

import matplotlib.pyplot as plt

# 需要动态修改的模块
import src.scheduler.env_reward as env_reward_mod
from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_types import ACTION_CLASSICAL, ACTION_QUANTUM
from src.scheduler.ppo_agent import PPOAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 实验配置
# ============================================================================
WAIT_PENALTY_CONFIGS = [
    {"name": "baseline_-0.1", "penalty": -0.1, "label": "基线(-0.1)"},
    {"name": "moderate_-0.3", "penalty": -0.3, "label": "中等(-0.3)"},
    {"name": "strong_-0.5", "penalty": -0.5, "label": "强(-0.5)"},
    {"name": "aggressive_-1.0", "penalty": -1.0, "label": "激进(-1.0)"},
]

DEFAULT_SEEDS = [42, 123, 456]
DEFAULT_TIMESTEPS = 50000
EVAL_EPISODES = 10


# ============================================================================
# 评估函数
# ============================================================================
def evaluate_ppo(
    agent: PPOAgent,
    env: QuantumSchedulingEnv,
    num_episodes: int = 10,
) -> dict[str, float]:
    """
    评估 PPO 智能体，记录总奖励、平均等待时间、完成数。

    Args:
        agent: 训练好的 PPO 智能体
        env: 调度环境
        num_episodes: 评估 episode 数

    Returns:
        dict: 评估指标
    """
    all_rewards = []
    all_avg_waits = []
    all_max_waits = []
    all_scheduled = []
    all_completion = []

    for ep in range(num_episodes):
        obs, _info = env.reset(seed=10000 + ep)
        total_reward = 0.0
        steps = 0
        done = False
        ep_wait_sum = 0.0
        ep_wait_count = 0
        ep_max_wait = 0

        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated

            # 收集等待时间统计
            for task in env._task_queue:
                ep_wait_sum += task.wait_steps
                ep_wait_count += 1
                ep_max_wait = max(ep_max_wait, task.wait_steps)

        all_rewards.append(total_reward)
        all_avg_waits.append(ep_wait_sum / max(ep_wait_count, 1))
        all_max_waits.append(ep_max_wait)
        all_scheduled.append(info.get("total_scheduled", 0))
        # 完成率 = scheduled / max_steps
        all_completion.append(info.get("total_scheduled", 0) / max(steps, 1))

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_avg_wait": float(np.mean(all_avg_waits)),
        "std_avg_wait": float(np.std(all_avg_waits)),
        "mean_max_wait": float(np.mean(all_max_waits)),
        "mean_scheduled": float(np.mean(all_scheduled)),
        "mean_completion": float(np.mean(all_completion)),
    }


def evaluate_fcfs(
    env: QuantumSchedulingEnv,
    num_episodes: int = 10,
) -> dict[str, float]:
    """
    FCFS 基线策略评估。

    Args:
        env: 调度环境
        num_episodes: 评估 episode 数

    Returns:
        dict: 评估指标
    """
    all_rewards = []
    all_avg_waits = []
    all_max_waits = []
    all_scheduled = []
    all_completion = []

    for ep in range(num_episodes):
        _obs, _info = env.reset(seed=10000 + ep)
        total_reward = 0.0
        steps = 0
        done = False
        ep_wait_sum = 0.0
        ep_wait_count = 0
        ep_max_wait = 0

        while not done:
            # FCFS 策略：按任务类型选择最兼容的动作
            task = env._current_task
            if task is None:
                action = ACTION_CLASSICAL
            elif task.task_type == "quantum":
                action = ACTION_QUANTUM
            elif task.task_type == "classical":
                action = ACTION_CLASSICAL
            else:
                action = ACTION_CLASSICAL  # universal 默认经典

            _obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = terminated or truncated

            for task in env._task_queue:
                ep_wait_sum += task.wait_steps
                ep_wait_count += 1
                ep_max_wait = max(ep_max_wait, task.wait_steps)

        all_rewards.append(total_reward)
        all_avg_waits.append(ep_wait_sum / max(ep_wait_count, 1))
        all_max_waits.append(ep_max_wait)
        all_scheduled.append(info.get("total_scheduled", 0))
        all_completion.append(info.get("total_scheduled", 0) / max(steps, 1))

    return {
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_avg_wait": float(np.mean(all_avg_waits)),
        "std_avg_wait": float(np.std(all_avg_waits)),
        "mean_max_wait": float(np.mean(all_max_waits)),
        "mean_scheduled": float(np.mean(all_scheduled)),
        "mean_completion": float(np.mean(all_completion)),
    }


# ============================================================================
# Pareto 前沿图
# ============================================================================
def plot_pareto(
    results: dict[str, Any],
    output_path: str,
) -> None:
    """
    生成 Pareto 前沿图：总奖励 vs 平均等待时间。

    Args:
        results: 实验结果
        output_path: 图片保存路径
    """
    # 设置中文字体
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    _fig, ax = plt.subplots(figsize=(10, 7))

    # 绘制每个配置的点
    colors = ["#2196F3", "#4CAF50", "#FF9800", "#F44336"]
    markers = ["o", "s", "^", "D"]

    for i, config in enumerate(WAIT_PENALTY_CONFIGS):
        name = config["name"]
        if name not in results["configs"]:
            continue
        cfg_data = results["configs"][name]
        reward = cfg_data["mean_reward"]
        reward_std = cfg_data["std_reward"]
        wait = cfg_data["mean_avg_wait"]
        wait_std = cfg_data["std_avg_wait"]

        ax.errorbar(
            wait,
            reward,
            xerr=wait_std,
            yerr=reward_std,
            fmt=markers[i],
            color=colors[i],
            markersize=12,
            capsize=5,
            capthick=2,
            linewidth=2,
            label=config["label"],
        )

    # FCFS 基线
    if "fcfs" in results:
        fcfs = results["fcfs"]
        ax.errorbar(
            fcfs["mean_avg_wait"],
            fcfs["mean_reward"],
            xerr=fcfs["std_avg_wait"],
            yerr=fcfs["std_reward"],
            fmt="*",
            color="black",
            markersize=18,
            capsize=5,
            capthick=2,
            linewidth=2,
            label="FCFS 基线",
        )

    # 标注 Pareto 最优
    ax.set_xlabel("平均等待时间（步）", fontsize=14)
    ax.set_ylabel("平均总奖励", fontsize=14)
    ax.set_title("等待惩罚权重扫描：奖励 vs 等待时间 Pareto 前沿", fontsize=16)
    ax.legend(fontsize=12, loc="best")
    ax.grid(True, alpha=0.3)

    # 添加箭头指示好方向
    ax.annotate(
        "← 更低等待时间更好",
        xy=(0.02, 0.5),
        xycoords="axes fraction",
        fontsize=11,
        color="green",
        alpha=0.7,
    )
    ax.annotate(
        "↑ 更高奖励更好",
        xy=(0.7, 0.98),
        xycoords="axes fraction",
        fontsize=11,
        color="green",
        alpha=0.7,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Pareto 前沿图已保存: {output_path}")


# ============================================================================
# 报告生成
# ============================================================================
def generate_report(
    results: dict[str, Any],
    output_path: str,
) -> None:
    """
    生成 Markdown 实验报告。

    Args:
        results: 实验结果
        output_path: 报告保存路径
    """
    lines = [
        "# 等待惩罚权重扫描实验报告（Issue #120）",
        "",
        f"> **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "> **实验目的**: 通过调整 REWARD_WAIT_OVER_THRESHOLD，寻找降低 PPO 等待时间的最佳配置",
        f"> **训练步数**: {results['config']['timesteps']}",
        f"> **Seeds**: {results['config']['seeds']}",
        f"> **评估 Episodes**: {EVAL_EPISODES}",
        f"> **总耗时**: {results['total_time_seconds']:.1f}s ({results['total_time_seconds'] / 60:.1f}min)",
        "",
        "## 实验背景",
        "",
        "Issue #120 指出 PPO 的平均等待时间比 FCFS 高 47%。原因分析：",
        "- 当前 `REWARD_WAIT_OVER_THRESHOLD = -0.1`，惩罚强度较弱",
        "- PPO 为了最大化总奖励，倾向于让任务排队等待（以获得更高的量子执行奖励）",
        "- 需要增强等待惩罚，迫使 PPO 更快地调度任务",
        "",
        "## 实验配置",
        "",
        "| 配置 | wait_penalty | 说明 |",
        "|:--:|:--:|:--|",
    ]

    for config in WAIT_PENALTY_CONFIGS:
        lines.append(f"| {config['label']} | {config['penalty']} | {config['name']} |")

    lines.extend(
        [
            "",
            "## 实验结果",
            "",
            "| 配置 | 平均奖励 | 标准差 | 平均等待时间 | 标准差 | 最大等待 | 完成数 | 完成率 |",
            "|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|",
        ]
    )

    # FCFS 基线
    if "fcfs" in results:
        fcfs = results["fcfs"]
        lines.append(
            f"| **FCFS 基线** | **{fcfs['mean_reward']:.2f}** | {fcfs['std_reward']:.2f} | "
            f"**{fcfs['mean_avg_wait']:.2f}** | {fcfs['std_avg_wait']:.2f} | "
            f"{fcfs['mean_max_wait']:.1f} | {fcfs['mean_scheduled']:.1f} | "
            f"{fcfs['mean_completion']:.1%} |"
        )

    # 各配置结果
    for config in WAIT_PENALTY_CONFIGS:
        name = config["name"]
        if name not in results["configs"]:
            continue
        cfg = results["configs"][name]
        lines.append(
            f"| {config['label']} | {cfg['mean_reward']:.2f} | {cfg['std_reward']:.2f} | "
            f"{cfg['mean_avg_wait']:.2f} | {cfg['std_avg_wait']:.2f} | "
            f"{cfg['mean_max_wait']:.1f} | {cfg['mean_scheduled']:.1f} | "
            f"{cfg['mean_completion']:.1%} |"
        )

    lines.extend(
        [
            "",
            "## 关键发现",
            "",
        ]
    )

    # 分析关键发现
    fcfs_wait = results.get("fcfs", {}).get("mean_avg_wait", 0)
    baseline_wait = results.get("configs", {}).get("baseline_-0.1", {}).get("mean_avg_wait", 0)
    best_config = None
    best_wait = float("inf")
    for config in WAIT_PENALTY_CONFIGS:
        name = config["name"]
        if name in results["configs"]:
            cfg = results["configs"][name]
            if cfg["mean_avg_wait"] < best_wait:
                best_wait = cfg["mean_avg_wait"]
                best_config = config

    if best_config:
        best_name = best_config["name"]
        best_data = results["configs"][best_name]
        improvement_vs_baseline = (
            (baseline_wait - best_wait) / baseline_wait * 100 if baseline_wait > 0 else 0
        )
        improvement_vs_fcfs = (fcfs_wait - best_wait) / fcfs_wait * 100 if fcfs_wait > 0 else 0

        lines.extend(
            [
                f"1. **最佳配置**: `{best_config['label']}`（penalty={best_config['penalty']}）",
                f"2. **等待时间改善**: vs 基线 {improvement_vs_baseline:+.1f}%，vs FCFS {improvement_vs_fcfs:+.1f}%",
                f"3. **奖励变化**: {best_data['mean_reward']:.2f} vs 基线 {results['configs']['baseline_-0.1']['mean_reward']:.2f}",
                "",
            ]
        )

        # 判断是否解决了Issue #120
        if best_wait <= fcfs_wait:
            lines.append(
                f"✅ **Issue #120 已解决**: `{best_config['label']}` 配置下 PPO 等待时间({best_wait:.2f}) ≤ FCFS({fcfs_wait:.2f})"
            )
        else:
            ratio = (best_wait - fcfs_wait) / fcfs_wait * 100
            lines.append(
                f"⚠️ **Issue #120 部分改善**: `{best_config['label']}` 配置下 PPO 等待时间仍比 FCFS 高 {ratio:.1f}%，但相比基线已改善 {improvement_vs_baseline:.1f}%"
            )

    lines.extend(
        [
            "",
            "## Pareto 前沿分析",
            "",
            "![Pareto 前沿](wait_penalty_pareto.png)",
            "",
            "Pareto 前沿图展示了总奖励与等待时间之间的权衡关系：",
            "- 左上角为理想区域（高奖励、低等待时间）",
            "- 随着等待惩罚增强，PPO 会降低等待时间，但可能牺牲部分总奖励",
            "- 选择 Pareto 最优配置，在可接受的奖励范围内最小化等待时间",
            "",
            "## 结论与建议",
            "",
        ]
    )

    if best_config and best_data:
        lines.extend(
            [
                f"建议采用 `{best_config['label']}` 配置（REWARD_WAIT_OVER_THRESHOLD = {best_config['penalty']}），",
                f"在保持总奖励 {best_data['mean_reward']:.2f} 的同时，将平均等待时间降至 {best_data['mean_avg_wait']:.2f}。",
                "",
                "### 后续行动",
                f"1. 将 `env_types.py` 中 `REWARD_WAIT_OVER_THRESHOLD` 更新为 `{best_config['penalty']}`",
                "2. 用新配置重新训练权威 PPO 模型（50000步，多seed）",
                "3. 重新运行 50 seeds × 5 episodes 多seed评估验证统计显著性",
                "4. 更新 PPT/白皮书中的实验数据",
            ]
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"报告已保存: {output_path}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description="等待惩罚权重扫描实验")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"每组训练的总步数（默认: {DEFAULT_TIMESTEPS}）",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"随机种子列表（默认: {DEFAULT_SEEDS}）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/wait_penalty_sweep",
        help="结果输出目录",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    original_penalty = env_reward_mod.REWARD_WAIT_OVER_THRESHOLD

    logger.info("=" * 70)
    logger.info("等待惩罚权重扫描实验（Issue #120）")
    logger.info("=" * 70)
    logger.info(f"配置数: {len(WAIT_PENALTY_CONFIGS)}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"训练步数: {args.timesteps}")
    logger.info(f"评估 Episodes: {EVAL_EPISODES}")
    logger.info(f"当前 REWARD_WAIT_OVER_THRESHOLD: {original_penalty}")
    logger.info("=" * 70)

    all_results: dict[str, Any] = {
        "config": {
            "timesteps": args.timesteps,
            "seeds": list(args.seeds),
            "eval_episodes": EVAL_EPISODES,
            "wait_penalty_configs": [c["penalty"] for c in WAIT_PENALTY_CONFIGS],
        },
        "configs": {},
        "fcfs": None,
        "timestamp": datetime.now().isoformat(),
    }

    start_time = time.time()

    # ========================================================================
    # FCFS 基线评估
    # ========================================================================
    logger.info("\n--- FCFS 基线评估 ---")
    # 恢复原始penalty
    env_reward_mod.REWARD_WAIT_OVER_THRESHOLD = original_penalty
    fcfs_env = QuantumSchedulingEnv(max_qubits=20, max_steps=200, seed=42)
    fcfs_result = evaluate_fcfs(fcfs_env, num_episodes=EVAL_EPISODES)
    all_results["fcfs"] = fcfs_result
    logger.info(
        f"FCFS: reward={fcfs_result['mean_reward']:.2f}±{fcfs_result['std_reward']:.2f}, "
        f"wait={fcfs_result['mean_avg_wait']:.2f}±{fcfs_result['std_avg_wait']:.2f}, "
        f"scheduled={fcfs_result['mean_scheduled']:.1f}"
    )

    # ========================================================================
    # PPO 训练 + 评估（每种配置）
    # ========================================================================
    for config in WAIT_PENALTY_CONFIGS:
        cfg_name = config["name"]
        penalty = config["penalty"]

        logger.info(f"\n{'=' * 60}")
        logger.info(f"配置: {config['label']} (penalty={penalty})")
        logger.info(f"{'=' * 60}")

        # 动态修改等待惩罚
        env_reward_mod.REWARD_WAIT_OVER_THRESHOLD = penalty
        logger.info(
            f"已设置 REWARD_WAIT_OVER_THRESHOLD = {env_reward_mod.REWARD_WAIT_OVER_THRESHOLD}"
        )

        seed_results = []
        for seed in args.seeds:
            logger.info(f"\n--- 训练 {cfg_name} seed={seed} ---")

            # 创建环境和智能体
            env = QuantumSchedulingEnv(max_qubits=20, max_steps=200, seed=seed)
            agent = PPOAgent(
                env,
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                n_epochs=10,
                gamma=0.99,
                verbose=0,
                seed=seed,
                log_dir=f"./logs/wait_sweep/{cfg_name}_seed{seed}",
            )

            # 训练
            train_start = time.time()
            agent.train(total_timesteps=args.timesteps, eval_freq=0, n_eval_episodes=0)
            train_time = time.time() - train_start
            logger.info(f"训练完成，耗时 {train_time:.1f}s")

            # 评估
            eval_env = QuantumSchedulingEnv(max_qubits=20, max_steps=200, seed=seed)
            eval_result = evaluate_ppo(agent, eval_env, num_episodes=EVAL_EPISODES)
            eval_result["train_time"] = train_time
            seed_results.append(eval_result)

            logger.info(
                f"seed={seed}: reward={eval_result['mean_reward']:.2f}±{eval_result['std_reward']:.2f}, "
                f"wait={eval_result['mean_avg_wait']:.2f}±{eval_result['std_avg_wait']:.2f}, "
                f"scheduled={eval_result['mean_scheduled']:.1f}, "
                f"耗时={train_time:.1f}s"
            )

        # 汇总该配置的所有seed结果
        agg_result = {
            "mean_reward": float(np.mean([r["mean_reward"] for r in seed_results])),
            "std_reward": float(np.std([r["mean_reward"] for r in seed_results])),
            "mean_avg_wait": float(np.mean([r["mean_avg_wait"] for r in seed_results])),
            "std_avg_wait": float(np.std([r["mean_avg_wait"] for r in seed_results])),
            "mean_max_wait": float(np.mean([r["mean_max_wait"] for r in seed_results])),
            "mean_scheduled": float(np.mean([r["mean_scheduled"] for r in seed_results])),
            "mean_completion": float(np.mean([r["mean_completion"] for r in seed_results])),
            "seed_results": seed_results,
        }
        all_results["configs"][cfg_name] = agg_result

        logger.info(f"\n{cfg_name} 汇总:")
        logger.info(
            f"  reward={agg_result['mean_reward']:.2f}±{agg_result['std_reward']:.2f}, "
            f"wait={agg_result['mean_avg_wait']:.2f}±{agg_result['std_avg_wait']:.2f}"
        )

    # 恢复原始penalty
    env_reward_mod.REWARD_WAIT_OVER_THRESHOLD = original_penalty
    logger.info(f"\n已恢复 REWARD_WAIT_OVER_THRESHOLD = {original_penalty}")

    total_time = time.time() - start_time
    all_results["total_time_seconds"] = total_time
    logger.info(f"\n总实验耗时: {total_time:.1f}s ({total_time / 60:.1f}min)")

    # ========================================================================
    # 保存结果和生成报告
    # ========================================================================
    # JSON 结果
    json_path = os.path.join(args.output_dir, "wait_penalty_sweep_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"结果已保存: {json_path}")

    # Pareto 前沿图
    plot_path = os.path.join(args.output_dir, "wait_penalty_pareto.png")
    plot_pareto(all_results, plot_path)

    # Markdown 报告
    report_path = os.path.join(args.output_dir, "wait_penalty_sweep_report.md")
    generate_report(all_results, report_path)

    # 打印最终汇总
    logger.info("\n" + "=" * 80)
    logger.info("实验汇总")
    logger.info("=" * 80)
    header = f"{'配置':20s} {'Reward':>12s} {'Wait':>10s} {'MaxWait':>8s} {'Scheduled':>10s}"
    logger.info(header)
    logger.info("-" * 80)

    fcfs = all_results["fcfs"]
    logger.info(
        f"{'FCFS 基线':20s} {fcfs['mean_reward']:12.2f} "
        f"{fcfs['mean_avg_wait']:10.2f} {fcfs['mean_max_wait']:8.1f} "
        f"{fcfs['mean_scheduled']:10.1f}"
    )

    for config in WAIT_PENALTY_CONFIGS:
        name = config["name"]
        if name in all_results["configs"]:
            cfg = all_results["configs"][name]
            logger.info(
                f"{config['label']:20s} {cfg['mean_reward']:12.2f} "
                f"{cfg['mean_avg_wait']:10.2f} {cfg['mean_max_wait']:8.1f} "
                f"{cfg['mean_scheduled']:10.1f}"
            )

    logger.info("=" * 80)
    logger.info("实验完成！")


if __name__ == "__main__":
    main()
