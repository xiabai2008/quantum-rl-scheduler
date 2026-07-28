"""
消融实验统一评估脚本
对比所有PPO变体：PPO-LSTM / PPO-MLP / PPO-LSTM+Annealing
多seed × 多episode统计显著性检验
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduler.ppo_agent import PPOAgent
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv

NUM_SEEDS = 10
EPISODES_PER_SEED = 15
MAX_STEPS = 500
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

MODEL_VARIANTS = [
    {
        "name": "PPO-LSTM (Full)",
        "path": PROJECT_ROOT / "logs" / "advanced_ppo_lstm" / "best_model" / "best_model.zip",
        "use_lstm": True,
        "use_annealing": False,
    },
    {
        "name": "PPO-MLP (No LSTM)",
        "path": PROJECT_ROOT / "logs" / "ppo_mlp" / "best_model" / "best_model.zip",
        "use_lstm": False,
        "use_annealing": False,
    },
    {
        "name": "PPO-LSTM+Annealing",
        "path": PROJECT_ROOT / "logs" / "ppo_lstm_anneal" / "best_model" / "best_model.zip",
        "use_lstm": True,
        "use_annealing": True,
    },
]


def evaluate_model(
    model_path: Path,
    use_lstm: bool,
    use_annealing: bool,
    num_seeds: int = NUM_SEEDS,
    episodes_per_seed: int = EPISODES_PER_SEED,
) -> dict:
    """Evaluate a trained PPO model across seeds (using agent.predict consistently)."""
    if not model_path.exists():
        return {"name": str(model_path), "exists": False}

    env0 = QuantumSchedulingEnv(
        max_steps=MAX_STEPS, machine_configs=DEFAULT_MACHINE_CONFIGS, seed=42
    )
    agent = PPOAgent(env=env0, use_lstm=use_lstm, use_annealing=use_annealing)
    agent.load(str(model_path))

    all_rewards = []
    all_completion_rates = []
    all_lengths = []
    all_action_counts = []

    for seed in range(num_seeds):
        env = QuantumSchedulingEnv(
            max_steps=MAX_STEPS, machine_configs=DEFAULT_MACHINE_CONFIGS, seed=10000 + seed
        )
        for ep in range(episodes_per_seed):
            obs, info = env.reset(seed=20000 + seed * 100 + ep)
            done = False
            total_reward = 0.0
            length = 0
            action_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            while not done:
                action = agent.predict(obs, deterministic=True)
                action_counts[action] += 1
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                length += 1
                done = terminated or truncated
            all_rewards.append(total_reward)
            all_completion_rates.append(info.get("completion_rate", 0.0))
            all_lengths.append(length)
            all_action_counts.append(action_counts)

    avg_counts = {k: float(np.mean([ac[k] for ac in all_action_counts])) for k in range(4)}

    return {
        "exists": True,
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "median_reward": float(np.median(all_rewards)),
        "min_reward": float(np.min(all_rewards)),
        "max_reward": float(np.max(all_rewards)),
        "mean_completion_rate": float(np.mean(all_completion_rates)),
        "std_completion_rate": float(np.std(all_completion_rates)),
        "mean_length": float(np.mean(all_lengths)),
        "n_episodes": len(all_rewards),
        "action_distribution": {
            "classical": avg_counts[0],
            "quantum": avg_counts[1],
            "hybrid": avg_counts[2],
            "qem": avg_counts[3],
        },
        "rewards": [float(r) for r in all_rewards],
    }


def compare_models(baseline: dict, variant: dict) -> dict:
    """Statistical comparison between baseline and variant."""
    if not baseline.get("exists") or not variant.get("exists"):
        return {"error": "model missing"}

    bl_r = np.array(baseline["rewards"])
    var_r = np.array(variant["rewards"])

    u_stat, p_mwu = stats.mannwhitneyu(bl_r, var_r, alternative="two-sided")
    t_stat, p_t = stats.ttest_ind(bl_r, var_r, equal_var=False)
    n1, n2 = len(bl_r), len(var_r)
    pooled_std = np.sqrt(((n1 - 1) * bl_r.std() ** 2 + (n2 - 1) * var_r.std() ** 2) / (n1 + n2 - 2))
    cohens_d = (bl_r.mean() - var_r.mean()) / pooled_std if pooled_std > 0 else 0.0

    return {
        "baseline": baseline["name"],
        "variant": variant["name"],
        "baseline_mean": baseline["mean_reward"],
        "variant_mean": variant["mean_reward"],
        "delta": variant["mean_reward"] - baseline["mean_reward"],
        "delta_pct": (variant["mean_reward"] - baseline["mean_reward"])
        / abs(baseline["mean_reward"])
        * 100,
        "mann_whitney_u": float(u_stat),
        "p_value_mwu": float(p_mwu),
        "p_value_t": float(p_t),
        "cohens_d": float(cohens_d),
        "significant_005": bool(p_mwu < 0.05),
        "significant_001": bool(p_mwu < 0.01),
    }


def main():
    print("=" * 70)
    print("消融实验: PPO策略网络组件对比评估")
    print(
        f"  {NUM_SEEDS} seeds × {EPISODES_PER_SEED} episodes = {NUM_SEEDS * EPISODES_PER_SEED} episodes/model"
    )
    print("=" * 70)

    results = {}
    for variant in MODEL_VARIANTS:
        name = variant["name"]
        print(f"\n评估: {name} ...")
        r = evaluate_model(variant["path"], variant["use_lstm"], variant["use_annealing"])
        r["name"] = name
        r["use_lstm"] = variant["use_lstm"]
        r["use_annealing"] = variant["use_annealing"]
        results[name] = r
        if r["exists"]:
            print(f"  Reward: {r['mean_reward']:.1f} +/- {r['std_reward']:.1f}")
            print(f"  Completion: {r['mean_completion_rate'] * 100:.1f}%")
            ad = r["action_distribution"]
            total_a = ad["classical"] + ad["quantum"] + ad["hybrid"] + ad["qem"]
            print(
                f"  Actions: Q={ad['quantum'] / total_a * 100:.0f}% H={ad['hybrid'] / total_a * 100:.0f}% "
                f"C={ad['classical'] / total_a * 100:.0f}% QEM={ad['qem'] / total_a * 100:.0f}%"
            )
        else:
            print(f"  [跳过] 模型不存在: {variant['path']}")

    baseline_name = "PPO-LSTM (Full)"
    baseline = results.get(baseline_name)
    comparisons = []

    if baseline and baseline.get("exists"):
        print(f"\n{'=' * 70}")
        print(f"统计显著性检验 (vs {baseline_name}, Mann-Whitney U)")
        print(f"{'=' * 70}")
        print(
            f"{'对比模型':<30} {'ΔReward':>10} {'Δ%':>8} {'p-value':>12} {'效应量d':>8} {'显著性':>8}"
        )
        print("-" * 82)

        for name, r in results.items():
            if name == baseline_name or not r.get("exists"):
                continue
            cmp = compare_models(baseline, r)
            comparisons.append(cmp)
            sig = "***" if cmp["significant_001"] else "**" if cmp["significant_005"] else "n.s."
            print(
                f"{name:<30} {cmp['delta']:>+10.1f} {cmp['delta_pct']:>+7.2f}% "
                f"{cmp['p_value_mwu']:>12.8f} {cmp['cohens_d']:>+8.3f} {sig:>8}"
            )

    print(f"\n{'=' * 70}")
    print("消融实验结果汇总")
    print(f"{'=' * 70}")
    print(f"{'模型':<30} {'平均奖励':>12} {'成功率':>10} {'Quantum%':>9} {'Hybrid%':>9}")
    print("-" * 72)
    for name, r in results.items():
        if r.get("exists"):
            ad = r["action_distribution"]
            total_a = ad["classical"] + ad["quantum"] + ad["hybrid"] + ad["qem"]
            print(
                f"{name:<30} {r['mean_reward']:>12.1f} {r['mean_completion_rate'] * 100:>9.1f}% "
                f"{ad['quantum'] / total_a * 100:>8.1f}% {ad['hybrid'] / total_a * 100:>8.1f}%"
            )

    output = {
        "experiment": "ppo_ablation_study",
        "timestamp": datetime.now().isoformat(),
        "num_seeds": NUM_SEEDS,
        "episodes_per_seed": EPISODES_PER_SEED,
        "model_results": results,
        "statistical_comparisons": comparisons,
    }
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"ablation_ppo_variants_{ts}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
