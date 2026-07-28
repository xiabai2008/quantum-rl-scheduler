"""
PPO-MLP 训练脚本（150万步 + 早停，用于LSTM消融对比）
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from src.scheduler.agent import PPOAgent
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv


class EarlyStoppingCallback(BaseCallback):
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
                        print(f"\n[早停] step={t}: 奖励={mean_r:.2f}（最佳={self.best_mean_reward:.2f}），"
                              f"连续{self.no_improve_count}/{self.patience}次无提升")
                    if self.no_improve_count >= self.patience:
                        print(f"\n[早停] 连续{self.patience}次评估无提升，停止训练。最佳={self.best_mean_reward:.2f}")
                        return False
            self._last_eval_count = len(timesteps)
        except Exception:
            pass
        return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--use-annealing", action="store_true", help="启用退火优化")
    parser.add_argument("--use-lstm", action="store_true", help="使用LSTM策略")
    parser.add_argument("--label", type=str, default="mlp", help="模型标签(用于日志/保存路径)")
    args = parser.parse_args()

    log_dir = f"./logs/ppo_{args.label}"
    save_path = f"./models/ppo_{args.label}_agent"
    total_timesteps = args.timesteps
    eval_freq = 50_000
    use_lstm = args.use_lstm or ("lstm" in args.label)

    print("=" * 70)
    print(f"PPO 训练 - {args.label} 变体（150万步 + 早停）")
    print("=" * 70)
    print(f"训练步数: {total_timesteps:,}, seed: {args.seed}")
    print(f"策略网络: {'LSTM' if use_lstm else 'MLP'}")
    print(f"退火: {'启用' if args.use_annealing else '禁用'}")
    print(f"早停: 连续{args.patience}次评估无提升则停止")
    print("=" * 70)

    env = QuantumSchedulingEnv(
        max_steps=500,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=args.seed,
    )
    print(f"\n[环境] obs_dim={env.observation_space.shape[0]}, act_dim={env.action_space.n}")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    agent_kwargs = dict(
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        use_lstm=use_lstm,
        n_lstm_layers=1,
        lstm_hidden_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=1,
        seed=args.seed,
        log_dir=log_dir,
    )
    if args.use_annealing:
        agent_kwargs["use_annealing"] = True
        agent_kwargs["anneal_interval"] = eval_freq

    agent = PPOAgent(**agent_kwargs)

    eval_results_path = os.path.join(log_dir, "eval_results")
    early_stop_cb = EarlyStoppingCallback(
        log_path=eval_results_path,
        patience=args.patience,
        min_delta=10.0,
        verbose=1,
    )

    print(f"\n[训练] 开始 {total_timesteps:,} 步...")
    t0 = datetime.now()
    agent.train(
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
        n_eval_episodes=10,
        log_dir=log_dir,
        extra_callbacks=[early_stop_cb],
    )
    elapsed = (datetime.now() - t0).total_seconds()

    print(f"\n[训练] 完成! 耗时: {elapsed:.0f}s ({elapsed/3600:.2f}h)")
    agent.save(save_path)
    print(f"[保存] {save_path}.zip")

    results = agent.evaluate(num_episodes=10, deterministic=True)
    print(f"\n最终评估: reward={results['mean_reward']:.2f}±{results['std_reward']:.2f}, "
          f"success={results['success_rate']:.2%}")

    import json
    summary = {
        "label": args.label,
        "seed": args.seed,
        "use_annealing": args.use_annealing,
        "use_lstm": use_lstm,
        "timesteps": total_timesteps,
        "early_stopped": early_stop_cb.no_improve_count >= args.patience,
        "best_mean_reward": early_stop_cb.best_mean_reward,
        "training_duration_s": elapsed,
        "mean_reward": results["mean_reward"],
        "std_reward": results["std_reward"],
        "success_rate": results["success_rate"],
    }
    import pathlib
    rdir = pathlib.Path("results")
    rdir.mkdir(exist_ok=True)
    with open(rdir / f"ablation_{args.label}_seed{args.seed}.json", "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
