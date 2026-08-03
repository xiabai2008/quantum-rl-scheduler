"""
3 独立 PPO 训练脚本（wrapper 严格对比第三路，Issue #928）

对每台机器训练一个独立 SB3 PPO（单机 env，仅含该机器），评估时轮流决策
（无协同投票），与 MAPPO（协同投票）严格对照。

训练量：每机 16K 步（3 机合计 48K，与 MAPPO 50K 对等）。

产出: models/independent_ppo_m{i}.zip (i=0,1,2)
用法: python scripts/training/train_independent_ppo.py [--timesteps-per-machine 16000]
"""

from __future__ import annotations

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from stable_baselines3 import PPO

from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="3 独立 PPO 训练")
    parser.add_argument("--timesteps-per-machine", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    machines = DEFAULT_MACHINE_CONFIGS[:3]
    os.makedirs("models", exist_ok=True)
    for i, m in enumerate(machines):
        env = QuantumSchedulingEnv(
            max_steps=100,
            machine_configs=[m],  # 单机 env（仅该机器）
            seed=args.seed,
        )
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            seed=args.seed,
            verbose=0,
        )
        t0 = time.time()
        model.learn(total_timesteps=args.timesteps_per_machine)
        save_path = f"models/independent_ppo_m{i}"
        model.save(save_path)
        print(
            f"[独立PPO-{i}] {m['name']} 训练完成 "
            f"({args.timesteps_per_machine} 步, {time.time() - t0:.1f}s) → {save_path}.zip"
        )


if __name__ == "__main__":
    main()
