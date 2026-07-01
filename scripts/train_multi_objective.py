#!/usr/bin/env python
"""
多目标RL训练脚本
Multi-Objective RL Training Script for Quantum Scheduling

对 3 组不同权重组合分别训练 PPO 智能体，对比多目标权衡效果。

使用示例:
    # 使用默认的 3 组权重训练
    python scripts/train_multi_objective.py

    # 自定义训练步数和权重
    python scripts/train_multi_objective.py --timesteps 30000 --seeds 42 123

    # 仅训练单组权重
    python scripts/train_multi_objective.py --weights throughput_heavy
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
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ============================================================================
# 日志配置
# ============================================================================
import logging

from src.scheduler.agent import PPOAgent
from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.multi_objective_env import (
    DEFAULT_WEIGHTS,
    MultiObjectiveRewardWrapper,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 命令行参数解析
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="多目标RL调度训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认 3 组权重训练
  python scripts/train_multi_objective.py

  # 指定单一权重
  python scripts/train_multi_objective.py --weights throughput_heavy

  # 自定义步数和种子
  python scripts/train_multi_objective.py --timesteps 30000 --seeds 42 123
        """,
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        default=["throughput_heavy", "balance_heavy", "quality_heavy"],
        help="权重预设名称列表，可选: throughput_heavy, balance_heavy, quality_heavy, balanced",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=20000,
        help="每组权重的训练总步数（默认: 20000）",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="随机种子列表（默认: [42]）",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=200,
        help="每个 episode 的最大步数（默认: 200）",
    )
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=20,
        help="最大量子比特数（默认: 20）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/multi_objective",
        help="结果输出目录（默认: results/multi_objective）",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/multi_objective",
        help="模型保存目录（默认: models/multi_objective）",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO 学习率（默认: 3e-4）",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="评估时的 episode 数量（默认: 10）",
    )
    return parser.parse_args()


# ============================================================================
# 训练函数
# ============================================================================
def train_mo(
    weight_preset: str,
    weights: list[float],
    seed: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    """
    在给定权重下训练 PPO 智能体并评估。

    Args:
        weight_preset: 权重预设名称
        weights: 实际权重值 [w_t, w_b, w_q]
        seed: 随机种子
        args: 命令行参数

    Returns:
        dict: 包含训练结果、评估指标和多目标累积值的字典
    """
    tag = f"{weight_preset}_seed{seed}"
    logger.info(f"[{tag}] 开始训练，权重={weights}")

    # 创建多目标环境
    env = QuantumSchedulingEnv(
        max_qubits=args.max_qubits,
        max_steps=args.max_steps,
        seed=seed,
    )
    mo_env = MultiObjectiveRewardWrapper(env, weights=weights)

    # 构建 PPO 智能体
    agent = PPOAgent(
        mo_env,
        learning_rate=args.learning_rate,
        n_steps=min(2048, args.timesteps // 4),
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=0,
        seed=seed,
        log_dir=f"./logs/multi_objective/{tag}",
    )

    # 训练
    start_time = time.time()
    agent.train(total_timesteps=args.timesteps)
    train_time = time.time() - start_time

    # 评估
    eval_results = agent.evaluate(num_episodes=args.eval_episodes, deterministic=True)

    # 收集多目标指标
    mo_metrics = collect_mo_metrics(mo_env, agent, num_episodes=args.eval_episodes, seed=seed)

    result = {
        "weight_preset": weight_preset,
        "weights": weights,
        "seed": seed,
        "timesteps": args.timesteps,
        "train_time_seconds": round(train_time, 1),
        "mean_reward": round(eval_results.get("mean_reward", 0.0), 4),
        "std_reward": round(eval_results.get("std_reward", 0.0), 4),
        "success_rate": round(eval_results.get("success_rate", 0.0), 4),
        "mo_metrics": mo_metrics,
        "timestamp": datetime.now().isoformat(),
    }

    logger.info(
        f"[{tag}] 训练完成: "
        f"mean_reward={result['mean_reward']:.2f}, "
        f"success_rate={result['success_rate']:.2%}, "
        f"mo={mo_metrics['avg_throughput']:.2f}/{mo_metrics['avg_balance']:.2f}/{mo_metrics['avg_quality']:.2f}, "
        f"耗时={train_time:.1f}s"
    )

    # 保存模型
    model_path = os.path.join(args.model_dir, tag)
    agent.save(model_path)

    return result


def collect_mo_metrics(
    mo_env: MultiObjectiveRewardWrapper,
    agent: PPOAgent,
    num_episodes: int = 10,
    seed: int | None = None,
) -> dict[str, float]:
    """
    收集多目标指标，运行多个 episode 取平均值。

    Args:
        mo_env: 多目标环境
        agent: PPO 智能体
        num_episodes: episode 数量
        seed: 随机种子

    Returns:
        dict: 平均多目标指标
    """
    all_throughputs = []
    all_balances = []
    all_qualities = []
    all_rewards = []

    for ep in range(num_episodes):
        obs, info = mo_env.reset(seed=seed + ep if seed is not None else None)
        done = False
        ep_reward = 0.0

        while not done:
            action, _ = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = mo_env.step(action)
            done = terminated or truncated
            ep_reward += reward

        objectives = mo_env.get_episode_objectives()
        all_throughputs.append(objectives["throughput"])
        all_balances.append(objectives["balance"])
        all_qualities.append(objectives["quality"])
        all_rewards.append(ep_reward)

    return {
        "avg_throughput": round(float(np.mean(all_throughputs)), 4),
        "std_throughput": round(float(np.std(all_throughputs)), 4),
        "avg_balance": round(float(np.mean(all_balances)), 4),
        "std_balance": round(float(np.std(all_balances)), 4),
        "avg_quality": round(float(np.mean(all_qualities)), 4),
        "std_quality": round(float(np.std(all_qualities)), 4),
        "avg_episode_reward": round(float(np.mean(all_rewards)), 4),
        "std_episode_reward": round(float(np.std(all_rewards)), 4),
    }


# ============================================================================
# 主函数
# ============================================================================
def main():
    args = parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)

    logger.info("=" * 60)
    logger.info("多目标RL训练 - 开始")
    logger.info(f"权重预设: {args.weights}")
    logger.info(f"种子: {args.seeds}")
    logger.info(f"训练步数: {args.timesteps}")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info("=" * 60)

    all_results = []

    for weight_preset in args.weights:
        if weight_preset not in DEFAULT_WEIGHTS:
            logger.error(f"未知权重预设 '{weight_preset}'，跳过")
            continue

        weights = DEFAULT_WEIGHTS[weight_preset]
        for seed in args.seeds:
            result = train_mo(weight_preset, weights, seed, args)
            all_results.append(result)

    # 保存汇总结果
    summary = {
        "config": {
            "weights_tested": args.weights,
            "seeds": args.seeds,
            "timesteps": args.timesteps,
            "max_steps": args.max_steps,
            "max_qubits": args.max_qubits,
            "learning_rate": args.learning_rate,
        },
        "results": all_results,
        "timestamp": datetime.now().isoformat(),
    }

    output_path = os.path.join(args.output_dir, "mo_training_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 打印对比表
    logger.info("\n" + "=" * 80)
    logger.info("多目标训练结果对比")
    logger.info("=" * 80)
    header = (
        f"{'权重预设':20s} {'Seed':5s} {'MeanReward':>12s} {'Success%':>9s} "
        f"{'Throughput':>12s} {'Balance':>10s} {'Quality':>10s} {'耗时':>8s}"
    )
    logger.info(header)
    logger.info("-" * 80)

    for r in all_results:
        mo = r["mo_metrics"]
        line = (
            f"{r['weight_preset']:20s} {r['seed']:5d} "
            f"{r['mean_reward']:12.2f} {r['success_rate']:8.1%} "
            f"{mo['avg_throughput']:12.2f} {mo['avg_balance']:10.2f} "
            f"{mo['avg_quality']:10.2f} {r['train_time_seconds']:7.1f}s"
        )
        logger.info(line)

    logger.info("=" * 80)
    logger.info(f"结果已保存至: {output_path}")
    logger.info("训练完成!")


if __name__ == "__main__":
    main()
