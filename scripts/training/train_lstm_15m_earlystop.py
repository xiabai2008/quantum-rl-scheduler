"""
PPO-LSTM 训练脚本（150万步 + 早停）
Early stopping: 连续5次评估（25万步）平均奖励无提升则停止训练
"""

import argparse
import os
import sys
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.scheduler.agent import PPOAgent
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv


class EarlyStoppingCallback(BaseCallback):
    """早停回调：连续patience次评估mean_reward无提升则停止训练。

    通过检查 EvalCallback 保存的 evaluations.npz 文件来获取最近的评估结果。
    """

    def __init__(self, log_path: str, patience: int = 5, min_delta: float = 10.0, verbose: int = 1):
        super().__init__(verbose)
        self.log_path = log_path
        self.patience = patience
        self.min_delta = min_delta
        self.best_mean_reward = -float("inf")
        self.no_improve_count = 0
        self.evals_path = os.path.join(log_path, "evaluations.npz")
        self._last_eval_count = 0

    def _on_step(self) -> bool:
        if not os.path.exists(self.evals_path):
            return True
        try:
            data = np.load(self.evals_path)
            timesteps = data["timesteps"]
            results = data["results"]
            if len(timesteps) <= self._last_eval_count:
                return True
            for i in range(self._last_eval_count, len(timesteps)):
                mean_r = float(results[i].mean())
                t = int(timesteps[i])
                if mean_r > self.best_mean_reward + self.min_delta:
                    self.best_mean_reward = mean_r
                    self.no_improve_count = 0
                    if self.verbose:
                        print(f"\n[早停] step={t}: 新最佳奖励={mean_r:.2f}，重置计数器")
                else:
                    self.no_improve_count += 1
                    if self.verbose:
                        print(
                            f"\n[早停] step={t}: 奖励={mean_r:.2f}（最佳={self.best_mean_reward:.2f}），"
                            f"连续{self.no_improve_count}/{self.patience}次无提升"
                        )
                    if self.no_improve_count >= self.patience:
                        print(
                            f"\n[早停] 连续{self.patience}次评估无提升，停止训练。"
                            f"最佳奖励={self.best_mean_reward:.2f}"
                        )
                        return False
            self._last_eval_count = len(timesteps)
        except Exception:
            pass
        return True


def parse_args():
    parser = argparse.ArgumentParser(description="PPO-LSTM训练（150万步+早停）")
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5, help="早停耐心值（连续N次评估无提升停）")
    parser.add_argument("--min-delta", type=float, default=10.0, help="最小提升幅度")
    return parser.parse_args()


def main():
    args = parse_args()

    total_timesteps = args.timesteps
    eval_freq = 50_000
    patience = args.patience
    min_delta = args.min_delta

    print("=" * 70)
    print("PPO-LSTM 量子任务调度智能体训练（150万步 + 早停）")
    print("=" * 70)
    print(f"训练步数: {total_timesteps:,}")
    print(f"随机种子: {args.seed}")
    print("LSTM: 1层, 隐藏层64")
    print("学习率: 0.0003, n_steps=2048, batch_size=64")
    print("多机器模式: 是（3台真机）")
    print(f"早停: 连续{patience}次评估（每{eval_freq:,}步1次）奖励提升<{min_delta}则停止")
    print("=" * 70)

    env = QuantumSchedulingEnv(
        max_steps=500,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=args.seed,
    )
    print(f"\n[环境] 观测维度: {env.observation_space.shape[0]}, 动作空间: {env.action_space.n}")

    log_dir = "./logs/advanced_ppo_lstm"
    save_path = "./models/advanced_ppo_lstm_agent"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    print("\n[智能体] 初始化 PPO-LSTM 智能体...")
    agent = PPOAgent(
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        use_lstm=True,
        n_lstm_layers=1,
        lstm_hidden_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        log_dir=log_dir,
    )

    eval_results_path = os.path.join(log_dir, "eval_results")
    early_stop_cb = EarlyStoppingCallback(
        log_path=eval_results_path,
        patience=patience,
        min_delta=min_delta,
        verbose=1,
    )

    print(f"\n[训练] 开始训练 {total_timesteps:,} 步（早停耐心值={patience}）...")
    start_time = datetime.now()

    agent.train(
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
        n_eval_episodes=10,
        log_dir=log_dir,
        extra_callbacks=[early_stop_cb],
    )

    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds()

    print(
        f"\n[训练] 训练完成！耗时: {training_duration:.2f} 秒 ({training_duration / 3600:.2f}小时)"
    )

    agent.save(save_path)
    print(f"[保存] 模型已保存至: {save_path}.zip")

    print("\n[评估] 进行最终性能评估...")
    eval_results = agent.evaluate(num_episodes=10, deterministic=True)

    print("\n" + "=" * 70)
    print("训练结果摘要")
    print("=" * 70)
    print(f"平均奖励: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"成功率: {eval_results['success_rate']:.2%}")
    print(f"训练时长: {training_duration:.2f} 秒 ({training_duration / 3600:.2f}小时)")
    print(f"模型路径: {save_path}.zip")
    print("=" * 70)

    import json

    results = {
        "algorithm": "PPO-LSTM",
        "timesteps": total_timesteps,
        "early_stopped": early_stop_cb.no_improve_count >= patience,
        "best_mean_reward": early_stop_cb.best_mean_reward,
        "patience": patience,
        "seed": args.seed,
        "training_duration_s": training_duration,
        "mean_reward": eval_results["mean_reward"],
        "std_reward": eval_results["std_reward"],
        "success_rate": eval_results["success_rate"],
        "timestamp": datetime.now().isoformat(),
    }
    with open(os.path.join(log_dir, "lstm_ppo_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
