#!/usr/bin/env python
"""
量子RL调度系统 — MAPPO 多智能体训练脚本
Quantum RL Scheduler - Multi-Agent PPO Training Script

为每台量子机器配置独立 Actor，集中式 Critic 协调多机器调度。

使用示例：
    # 默认三机配置训练
    python scripts/train_marl.py --timesteps 50000

    # 单机模式（退化为单 Agent PPO，用于基线对比）
    python scripts/train_marl.py --single-machine --timesteps 20000

    # 自定义参数
    python scripts/train_marl.py --timesteps 100000 --n-steps 2048 \\
        --learning-rate 3e-4 --seed 42

    # 训练后立即评估
    python scripts/train_marl.py --timesteps 30000 --eval-episodes 20
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv
from src.scheduler.marl import MultiAgentPPO

# ============================================================================
# 日志配置
# ============================================================================
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
    """
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数
    """
    parser = argparse.ArgumentParser(
        description="量子RL调度系统 - MAPPO 多智能体训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 默认三机配置训练（3 台真机配置）
  python scripts/train_marl.py --timesteps 50000

  # 单机模式（退化为单 Agent，用于基线对比）
  python scripts/train_marl.py --single-machine --timesteps 20000

  # 自定义超参数
  python scripts/train_marl.py --timesteps 100000 --n-steps 2048 \\
      --learning-rate 3e-4 --seed 42 --eval-episodes 20

  # 训练 + 保存 + 评估
  python scripts/train_marl.py --timesteps 30000 --save-path ./models/mappo
        """,
    )

    # 训练参数
    training = parser.add_argument_group("训练参数")
    training.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="训练总步数（默认: 50000）",
    )
    training.add_argument(
        "--n-steps",
        type=int,
        default=1024,
        help="每次 rollout 收集的步数（默认: 1024）",
    )
    training.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="小批量大小（默认: 64）",
    )
    training.add_argument(
        "--n-epochs",
        type=int,
        default=10,
        help="每次更新的 epoch 数（默认: 10）",
    )
    training.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="学习率（默认: 3e-4）",
    )
    training.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="折扣因子（默认: 0.99）",
    )
    training.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="GAE lambda 参数（默认: 0.95）",
    )
    training.add_argument(
        "--clip-range",
        type=float,
        default=0.2,
        help="PPO 裁剪范围（默认: 0.2）",
    )
    training.add_argument(
        "--ent-coef",
        type=float,
        default=0.01,
        help="熵正则系数（默认: 0.01）",
    )
    training.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="单个 episode 最大步数（默认: 500）",
    )

    # 机器配置参数
    machine_group = parser.add_argument_group("机器配置")
    machine_group.add_argument(
        "--single-machine",
        action="store_true",
        help="单机模式（退化为单 Agent PPO，用于基线对比）",
    )
    machine_group.add_argument(
        "--num-machines",
        type=int,
        default=3,
        help="自定义机器数量（1-3，从 DEFAULT_MACHINE_CONFIGS 截取）",
    )

    # 评估参数
    evaluation = parser.add_argument_group("评估参数")
    evaluation.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="评估频率，每隔多少步评估一次（默认: 5000）",
    )
    evaluation.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="每次评估的 episode 数（默认: 10）",
    )

    # 输出参数
    output = parser.add_argument_group("输出参数")
    output.add_argument(
        "--save-path",
        type=str,
        default="./models/mappo",
        help="模型保存路径（默认: ./models/mappo）",
    )
    output.add_argument(
        "--log-dir",
        type=str,
        default="./logs/marl/",
        help="TensorBoard/训练日志目录（默认: ./logs/marl/）",
    )
    output.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="评估结果保存目录（默认: ./results）",
    )
    output.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认: 42）",
    )
    output.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="日志详细程度 0=静默 1=进度（默认: 1）",
    )
    output.add_argument(
        "--no-save",
        action="store_true",
        help="不保存模型（仅训练用于快速验证）",
    )
    output.add_argument(
        "--no-eval",
        action="store_true",
        help="训练后不进行评估",
    )

    return parser.parse_args()


# ============================================================================
# 机器配置构建
# ============================================================================
def build_machine_configs(single_machine: bool, num_machines: int) -> list | None:
    """
    根据 CLI 参数构建机器配置列表。

    Args:
        single_machine: 是否单机模式
        num_machines: 自定义机器数量

    Returns:
        机器配置列表，None 表示单机模式（env 内部退化为默认单机）
    """
    if single_machine:
        # 单机模式：返回 None，env 内部会退化为默认单机配置
        return None
    n = max(1, min(num_machines, len(DEFAULT_MACHINE_CONFIGS)))
    return DEFAULT_MACHINE_CONFIGS[:n]


# ============================================================================
# 主流程
# ============================================================================
def main():
    """训练主流程。"""
    args = parse_args()
    start_time = time.time()

    print("=" * 64)
    print("  量子RL调度系统 — MAPPO 多智能体训练")
    print("=" * 64)
    print(f"  训练步数   : {args.timesteps}")
    print(f"  rollout步数: {args.n_steps}")
    print(f"  batch_size : {args.batch_size}")
    print(f"  学习率     : {args.learning_rate}")
    print(f"  种子       : {args.seed}")

    # 构建机器配置
    machine_configs = build_machine_configs(args.single_machine, args.num_machines)
    if machine_configs is None:
        print("  机器配置   : 单机模式（退化为单 Agent）")
    else:
        names = [c["name"] for c in machine_configs]
        qubits = [c["total_qubits"] for c in machine_configs]
        print(f"  机器配置   : {len(machine_configs)} 台 -> {names}")
        print(f"  量子比特   : {qubits}")

    print("=" * 64 + "\n")

    # 创建环境
    env = QuantumSchedulingEnv(
        max_steps=args.max_steps,
        machine_configs=machine_configs,
        seed=args.seed,
    )

    # 创建 MAPPO 智能体
    agent = MultiAgentPPO(
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        seed=args.seed,
        log_dir=args.log_dir,
        verbose=args.verbose,
    )
    print(agent)
    print()

    # ---- 训练 ----
    print("--- 开始训练 ---")
    train_start = time.time()
    agent.train(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
    )
    train_duration = time.time() - train_start
    print(f"\n训练完成，耗时 {train_duration:.1f}s")

    # ---- 保存模型 ----
    if not args.no_save:
        os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
        agent.save(args.save_path)
        print(f"模型已保存: {args.save_path}.pt")

    # ---- 评估 ----
    if not args.no_eval:
        print("\n--- 最终评估 ---")
        eval_start = time.time()
        eval_result = agent.evaluate(
            num_episodes=args.eval_episodes,
            deterministic=True,
        )
        eval_duration = time.time() - eval_start
        print(f"评估结果 ({eval_result['num_episodes']} episodes, " f"耗时 {eval_duration:.1f}s):")
        print(
            f"  平均奖励 : {eval_result['mean_reward']:.2f} " f"± {eval_result['std_reward']:.2f}"
        )
        print(f"  成功率   : {eval_result['success_rate'] * 100:.1f}%")

        # 保存评估结果
        os.makedirs(args.results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_tag = "single" if args.single_machine else f"multi{env.num_machines}"
        results_path = os.path.join(args.results_dir, f"mappo_eval_{mode_tag}_{timestamp}.json")
        result_data: dict[str, Any] = {
            "timestamp": timestamp,
            "mode": "single_machine" if args.single_machine else "multi_machine",
            "num_machines": env.num_machines,
            "machine_names": env.machine_names,
            "config": agent.get_config(),
            "eval_result": eval_result,
            "train_duration_s": round(train_duration, 2),
            "total_timesteps": args.timesteps,
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, ensure_ascii=False, indent=2)
        print(f"\n评估结果已保存: {results_path}")

    total_duration = time.time() - start_time
    print(f"\n总耗时: {total_duration:.1f}s")
    print("=" * 64)
    print("  MAPPO 训练流程完成")
    print("=" * 64)


if __name__ == "__main__":
    main()
