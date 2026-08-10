#!/usr/bin/env python
"""round5-C: PPO-LSTM 变体训练与评估（与权威 PPO 对比）。

目标：验证 LSTM 策略（RecurrentPPO）在部分可观测调度问题上是否优于
标准 MLP-PPO（权威 +20.2% 模型）。代码已支持 use_lstm=True（ppo_agent.py）。

用法:
    python scripts/training/train_lstm_variant.py --train      # 训练 10 seeds
    python scripts/training/train_lstm_variant.py --eval       # 评估对比
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

import torch

from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv
from src.scheduler.ppo_agent import PPOAgent

TRAIN_TIMESTEPS = 500_000
EVAL_EPISODES = 5
MAX_STEPS = 500
EVAL_MAX_STEPS = 200  # 权威协议：200 步/episode（与 +20.2% 权威实验一致）
SEEDS = [42, 123, 456, 789, 1024, 2025, 3141, 5678, 8765, 9999]
MODEL_DIR = _PROJECT_ROOT / "models" / "lstm_variant"
RESULTS_DIR = _PROJECT_ROOT / "results" / "lstm_variant"
AUTHORITATIVE_MODEL = _PROJECT_ROOT / "deliverable_models" / "ppo_best_model_16dim.zip"


def train_one(seed: int, timesteps: int = TRAIN_TIMESTEPS) -> str:
    path = MODEL_DIR / f"ppo_lstm_seed{seed}"
    if path.with_suffix(".zip").exists():
        print(f"  [SKIP] {path}")
        return str(path)
    env = QuantumSchedulingEnv(
        max_steps=MAX_STEPS,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=seed,
    )
    agent = PPOAgent(
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=0,
        seed=seed,
        log_dir=str(_PROJECT_ROOT / "logs" / "lstm_variant" / f"seed{seed}"),
        use_lstm=True,
        n_lstm_layers=1,
        lstm_hidden_size=64,
    )
    agent.train(total_timesteps=timesteps, log_dir=str(_PROJECT_ROOT / "logs" / "lstm_variant"))
    agent.save(str(path))
    print(f"  [TRAINED] seed={seed}")
    return str(path)


def evaluate_model(model_path: str, seed: int) -> dict:
    env = QuantumSchedulingEnv(
        max_steps=EVAL_MAX_STEPS,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=seed + 10000,
    )
    agent = PPOAgent(env, verbose=0, seed=seed, use_lstm="lstm" in model_path)
    agent.load(model_path)
    results = agent.evaluate(num_episodes=EVAL_EPISODES, deterministic=True)
    return {
        "seed": seed,
        "mean_reward": float(results["mean_reward"]),
        "std_reward": float(results["std_reward"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="PPO-LSTM 变体训练与评估")
    parser.add_argument("--train", action="store_true", help="训练 LSTM 模型")
    parser.add_argument("--eval", action="store_true", help="评估 LSTM vs 权威 PPO")
    parser.add_argument("--timesteps", type=int, default=TRAIN_TIMESTEPS)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    args = parser.parse_args()

    seeds = args.seeds or SEEDS
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.train:
        torch.set_num_threads(4)
        for seed in seeds:
            t0 = time.time()
            train_one(seed, args.timesteps)
            print(f"  seed={seed} 耗时 {time.time() - t0:.0f}s", flush=True)

    if args.eval:
        lstm_results = []
        ppo_results = []
        for seed in seeds:
            lstm_results.append(evaluate_model(str(MODEL_DIR / f"ppo_lstm_seed{seed}.zip"), seed))
            ppo_results.append(evaluate_model(str(AUTHORITATIVE_MODEL), seed))
            print(
                f"  seed={seed}: LSTM={lstm_results[-1]['mean_reward']:.1f} "
                f"PPO={ppo_results[-1]['mean_reward']:.1f}",
                flush=True,
            )

        from scipy import stats

        l_means = [r["mean_reward"] for r in lstm_results]
        p_means = [r["mean_reward"] for r in ppo_results]
        # 配对检验（同 seed）
        try:
            _stat, p = stats.wilcoxon(p_means, l_means, alternative="two-sided")
        except ValueError:
            p = 1.0
        d = (sum(l_means) - sum(p_means)) / len(l_means)
        summary = {
            "experiment": "lstm_vs_ppo_authoritative",
            "timestamp": datetime.now().isoformat(),
            "n_seeds": len(seeds),
            "lstm_mean": round(sum(l_means) / len(l_means), 2),
            "ppo_mean": round(sum(p_means) / len(p_means), 2),
            "mean_diff_lstm_minus_ppo": round(d, 2),
            "wilcoxon_p": float(p),
            "lstm_per_seed": lstm_results,
            "ppo_per_seed": ppo_results,
        }
        out = RESULTS_DIR / f"lstm_vs_ppo_{datetime.now().strftime('%Y%m%d')}.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(
            f"\n结果: LSTM={summary['lstm_mean']:.1f} vs PPO={summary['ppo_mean']:.1f} "
            f"(diff={d:.1f}, Wilcoxon p={p:.3f})"
        )
        print(f"已保存: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
