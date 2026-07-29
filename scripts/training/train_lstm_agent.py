"""
LSTM 策略训练脚本
Train PPO Agent with LSTM Policy for Quantum Task Scheduling

使用 LSTM 策略增强 PPO 智能体的时序记忆能力，使其能够利用历史观测序列
做出更优的调度决策。

使用方法:
    python scripts/train_lstm_agent.py --timesteps 50000 --seed 42
"""

import argparse
import os
import sys
from datetime import datetime

# 添加项目根目录到路径
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scheduler.agent import PPOAgent
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练 PPO-LSTM 智能体进行量子任务调度")
    parser.add_argument("--timesteps", type=int, default=50000, help="训练总步数（默认 50000）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    parser.add_argument("--n-lstm-layers", type=int, default=1, help="LSTM 层数（默认 1）")
    parser.add_argument(
        "--lstm-hidden-size", type=int, default=64, help="LSTM 隐藏层大小（默认 64）"
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="学习率（默认 3e-4）")
    parser.add_argument("--n-steps", type=int, default=2048, help="每次更新的步数（默认 2048）")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小（默认 64）")
    parser.add_argument(
        "--multi-machine", action="store_true", help="使用多机器调度模式（3台真机）"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs/lstm_ppo", help="日志目录（默认 ./logs/lstm_ppo）"
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./models/lstm_ppo_agent",
        help="模型保存路径（默认 ./models/lstm_ppo_agent）",
    )
    parser.add_argument("--eval-freq", type=int, default=5000, help="评估频率（默认每 5000 步）")
    parser.add_argument("--eval-episodes", type=int, default=10, help="每次评估的回合数（默认 10）")

    return parser.parse_args()


def main():
    """主训练流程"""
    args = parse_args()

    print("=" * 70)
    print("PPO-LSTM 量子任务调度智能体训练")
    print("=" * 70)
    print(f"训练步数: {args.timesteps}")
    print(f"随机种子: {args.seed}")
    print(f"LSTM 层数: {args.n_lstm_layers}")
    print(f"LSTM 隐藏层: {args.lstm_hidden_size}")
    print(f"学习率: {args.learning_rate}")
    print(f"多机器模式: {'是' if args.multi_machine else '否'}")
    print(f"日志目录: {args.log_dir}")
    print(f"模型保存: {args.save_path}")
    print("=" * 70)

    # 创建环境
    if args.multi_machine:
        print("\n[环境] 使用多机器调度模式（3台真机）")
        env = QuantumSchedulingEnv(
            max_steps=500,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
            seed=args.seed,
        )
    else:
        print("\n[环境] 使用单机调度模式")
        env = QuantumSchedulingEnv(
            max_steps=500,
            max_qubits=287,
            seed=args.seed,
        )

    print(f"[环境] 观测空间维度: {env.observation_space.shape[0]}")
    print(f"[环境] 动作空间: {env.action_space.n} 个离散动作")

    # 创建 PPO-LSTM 智能体
    print("\n[智能体] 初始化 PPO-LSTM 智能体...")
    agent = PPOAgent(
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        use_lstm=True,  # 启用 LSTM 策略
        n_lstm_layers=args.n_lstm_layers,
        lstm_hidden_size=args.lstm_hidden_size,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        log_dir=args.log_dir,
    )

    # 训练
    print(f"\n[训练] 开始训练 {args.timesteps} 步...")
    start_time = datetime.now()

    agent.train(
        total_timesteps=args.timesteps,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        log_dir=args.log_dir,
    )

    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()

    print(f"\n[训练] 训练完成！耗时: {training_duration:.2f} 秒")

    # 保存模型
    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    agent.save(args.save_path)
    print(f"[保存] 模型已保存至: {args.save_path}.zip")

    # 最终评估
    print("\n[评估] 进行最终性能评估...")
    eval_results = agent.evaluate(num_episodes=args.eval_episodes, deterministic=True)

    print("\n" + "=" * 70)
    print("训练结果摘要")
    print("=" * 70)
    print(f"平均奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"成功率: {eval_results['success_rate']:.2%}")
    print(f"训练时长: {training_duration:.2f} 秒")
    print(f"模型路径: {args.save_path}.zip")
    print("=" * 70)

    # 保存训练结果到 JSON
    results = {
        "algorithm": "PPO-LSTM",
        "timesteps": args.timesteps,
        "seed": args.seed,
        "n_lstm_layers": args.n_lstm_layers,
        "lstm_hidden_size": args.lstm_hidden_size,
        "learning_rate": args.learning_rate,
        "multi_machine": args.multi_machine,
        "training_duration_s": training_duration,
        "mean_reward": eval_results["mean_reward"],
        "std_reward": eval_results["std_reward"],
        "success_rate": eval_results["success_rate"],
        "timestamp": datetime.now().isoformat(),
    }

    results_path = os.path.join(args.log_dir, "lstm_ppo_results.json")
    os.makedirs(args.log_dir, exist_ok=True)

    import json

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n[结果] 训练结果已保存至: {results_path}")


if __name__ == "__main__":
    main()
