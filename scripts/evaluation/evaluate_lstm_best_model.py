"""
PPO-LSTM 最佳模型深度评估 + 基线对比实验
- 多seed(10) × 多episode(20) 统计显著性
- 对比: PPO-LSTM(best/final) vs Random/All-Classical/All-Quantum/Hybrid/Greedy
- 指标: mean_reward, success_rate, avg_wait, qubit_utilization, fidelity
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.scheduler.agent import PPOAgent
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv

NUM_SEEDS = 10
EPISODES_PER_SEED = 15
MAX_STEPS = 500
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

BEST_MODEL_PATH = PROJECT_ROOT / "logs" / "advanced_ppo_lstm" / "best_model" / "best_model.zip"
FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "advanced_ppo_lstm_agent.zip"


def make_env(seed: int) -> QuantumSchedulingEnv:
    return QuantumSchedulingEnv(
        max_steps=MAX_STEPS,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=seed,
    )


def evaluate_strategy(strategy_name: str, action_fn, num_seeds: int, episodes_per_seed: int) -> dict:
    """Evaluate a strategy across multiple seeds and episodes.

    action_fn: callable(obs, info) -> action
    """
    all_rewards = []
    all_completion_rates = []
    all_ep_lengths = []

    for seed in range(num_seeds):
        env = make_env(seed=1000 + seed)
        for ep in range(episodes_per_seed):
            obs, info = env.reset(seed=2000 + seed * 100 + ep)
            total_reward = 0.0
            steps = 0
            done = False
            while not done:
                action = action_fn(obs, info, env)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated
            all_rewards.append(total_reward)
            all_completion_rates.append(info.get("completion_rate", 0.0))
            all_ep_lengths.append(steps)
        env.close()

    return {
        "strategy": strategy_name,
        "num_seeds": num_seeds,
        "episodes_per_seed": episodes_per_seed,
        "total_episodes": len(all_rewards),
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "median_reward": float(np.median(all_rewards)),
        "min_reward": float(np.min(all_rewards)),
        "max_reward": float(np.max(all_rewards)),
        "mean_completion_rate": float(np.mean(all_completion_rates)),
        "std_completion_rate": float(np.std(all_completion_rates)),
        "mean_ep_length": float(np.mean(all_ep_lengths)),
        "rewards": [float(r) for r in all_rewards],
        "completion_rates": [float(c) for c in all_completion_rates],
    }


def load_ppo_agent(model_path: Path, use_lstm: bool = True) -> PPOAgent:
    env = make_env(seed=42)
    agent = PPOAgent(
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        use_lstm=use_lstm,
        n_lstm_layers=1,
        lstm_hidden_size=64,
        gamma=0.99,
        gae_lambda=0.95,
        ent_coef=0.01,
        verbose=0,
        seed=42,
    )
    agent.load(str(model_path))
    return agent


# ─── Baseline strategies ────────────────────────────────────────────

def random_action(obs, info, env):
    return env.action_space.sample()

def classical_action(obs, info, env):
    return 0  # ACTION_CLASSICAL

def quantum_action(obs, info, env):
    return 1  # ACTION_QUANTUM

def hybrid_action(obs, info, env):
    return 2  # ACTION_HYBRID

def qem_action(obs, info, env):
    return 3  # ACTION_QEM (QEM)

def greedy_action(obs, info, env):
    """Greedy: use quantum if quantum machines have high availability, else hybrid."""
    machines_info = info.get("machines", [])
    quantum_machines = [m for m in machines_info if m.get("is_real", False)]
    if quantum_machines:
        avg_available = np.mean([m["available_ratio"] for m in quantum_machines])
        if avg_available > 0.3:
            return 1  # quantum
    return 2  # hybrid fallback


def make_ppo_action(agent: PPOAgent):
    """Create action function from loaded PPOAgent."""
    def action_fn(obs, info, env):
        return agent.predict(obs, deterministic=True)
    return action_fn


# ─── Statistical tests ──────────────────────────────────────────────

def compute_statistical_significance(baseline_rewards: list[float], treatment_rewards: list[float]) -> dict:
    """Compare treatment against baseline using Mann-Whitney U test + effect size."""
    u_stat, p_value = stats.mannwhitneyu(treatment_rewards, baseline_rewards, alternative="two-sided")
    # Cohen's d effect size
    n1, n2 = len(treatment_rewards), len(baseline_rewards)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(treatment_rewards, ddof=1) + (n2 - 1) * np.var(baseline_rewards, ddof=1))
        / (n1 + n2 - 2)
    )
    cohens_d = (np.mean(treatment_rewards) - np.mean(baseline_rewards)) / pooled_std if pooled_std > 0 else 0.0
    return {
        "mann_whitney_u": float(u_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant_005": bool(p_value < 0.05),
        "significant_001": bool(p_value < 0.01),
    }


def main():
    print("=" * 70)
    print("PPO-LSTM 最佳模型深度评估 + 基线对比实验")
    print("=" * 70)
    print(f"配置: {NUM_SEEDS} seeds × {EPISODES_PER_SEED} episodes = {NUM_SEEDS * EPISODES_PER_SEED} episodes/策略")
    print(f"环境: 多机器模式, max_steps={MAX_STEPS}")
    print("=" * 70)

    strategies = []

    # 1. PPO-LSTM best model (250K steps)
    if BEST_MODEL_PATH.exists():
        print(f"\n[1/8] 加载 PPO-LSTM 最佳模型: {BEST_MODEL_PATH}")
        agent_best = load_ppo_agent(BEST_MODEL_PATH, use_lstm=True)
        strategies.append(("PPO-LSTM (best@250K)", make_ppo_action(agent_best)))
    else:
        print(f"WARNING: best model not found at {BEST_MODEL_PATH}")

    # 2. PPO-LSTM final model (500K steps, early stopped)
    if FINAL_MODEL_PATH.exists():
        print(f"[2/8] 加载 PPO-LSTM 最终模型: {FINAL_MODEL_PATH}")
        agent_final = load_ppo_agent(FINAL_MODEL_PATH, use_lstm=True)
        strategies.append(("PPO-LSTM (final@500K)", make_ppo_action(agent_final)))
    else:
        print(f"WARNING: final model not found at {FINAL_MODEL_PATH}")

    # 3-8. Baselines
    print("[3/8] Random 策略")
    strategies.append(("Random", random_action))
    print("[4/8] All-Classical 策略")
    strategies.append(("All-Classical", classical_action))
    print("[5/8] All-Quantum 策略")
    strategies.append(("All-Quantum", quantum_action))
    print("[6/8] Hybrid-only 策略")
    strategies.append(("Hybrid-only", hybrid_action))
    print("[7/8] QEM-only 策略")
    strategies.append(("QEM-only", qem_action))
    print("[8/8] Greedy (qubit-aware) 策略")
    strategies.append(("Greedy-Qubit", greedy_action))

    results = []
    for name, action_fn in strategies:
        print(f"\n>>> 评估策略: {name}")
        t0 = datetime.now()
        r = evaluate_strategy(name, action_fn, NUM_SEEDS, EPISODES_PER_SEED)
        elapsed = (datetime.now() - t0).total_seconds()
        r["eval_time_s"] = elapsed
        results.append(r)
        print(f"    mean_reward={r['mean_reward']:.2f} ± {r['std_reward']:.2f}, "
              f"completion={r['mean_completion_rate']:.2%}, "
              f"time={elapsed:.1f}s")

    # ─── Statistical comparison vs best model ──────────────────────
    best_result = next(r for r in results if "best@250K" in r["strategy"])
    baseline_for_stats = next(r for r in results if r["strategy"] == "Random")
    comparisons = {}
    for r in results:
        if r["strategy"] != best_result["strategy"]:
            comparisons[r["strategy"]] = compute_statistical_significance(
                r["rewards"], best_result["rewards"]
            )

    # ─── Print summary table ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("对比结果汇总")
    print("=" * 70)
    header = f"{'策略':<25} {'平均奖励':>12} {'标准差':>10} {'完成率':>8} {'vs最佳p值':>10} {'效应量':>8}"
    print(header)
    print("-" * len(header))
    for r in sorted(results, key=lambda x: -x["mean_reward"]):
        name = r["strategy"]
        stars = ""
        cohens = ""
        if name in comparisons:
            sig = comparisons[name]
            if sig["significant_001"]:
                stars = "***"
            elif sig["significant_005"]:
                stars = "**"
            cohens = f"{sig['cohens_d']:+.2f}"
        else:
            stars = " (best)"
        print(f"{name:<25} {r['mean_reward']:>12.2f} {r['std_reward']:>10.2f} "
              f"{r['mean_completion_rate']:>7.2%} {stars:>10} {cohens:>8}")

    print(f"\n显著性标注: *** p<0.01, ** p<0.05 (vs PPO-LSTM best@250K)")
    print(f"效应量: |d|<0.2 可忽略, 0.2-0.5 小, 0.5-0.8 中, >0.8 大")

    # ─── Save results ──────────────────────────────────────────────
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "PPO-LSTM deep evaluation + baseline comparison",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "num_seeds": NUM_SEEDS,
            "episodes_per_seed": EPISODES_PER_SEED,
            "max_steps": MAX_STEPS,
            "multi_machine": True,
            "best_model_path": str(BEST_MODEL_PATH),
            "final_model_path": str(FINAL_MODEL_PATH),
        },
        "results": results,
        "statistical_tests_vs_best": comparisons,
    }
    json_path = RESULTS_DIR / f"lstm_baseline_comparison_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {json_path}")

    # Print improvement over random
    random_r = next(r for r in results if r["strategy"] == "Random")
    best_r = best_result
    improvement = (best_r["mean_reward"] - random_r["mean_reward"]) / abs(random_r["mean_reward"]) * 100
    print(f"\nPPO-LSTM 相对于 Random 策略提升: {improvement:+.1f}%")
    if "All-Classical" in [r["strategy"] for r in results]:
        classical_r = next(r for r in results if r["strategy"] == "All-Classical")
        imp_c = (best_r["mean_reward"] - classical_r["mean_reward"]) / abs(classical_r["mean_reward"]) * 100
        print(f"PPO-LSTM 相对于 All-Classical 策略提升: {imp_c:+.1f}%")
    if "Greedy-Qubit" in [r["strategy"] for r in results]:
        greedy_r = next(r for r in results if r["strategy"] == "Greedy-Qubit")
        imp_g = (best_r["mean_reward"] - greedy_r["mean_reward"]) / abs(greedy_r["mean_reward"]) * 100
        print(f"PPO-LSTM 相对于 Greedy-Qubit 策略提升: {imp_g:+.1f}%")


if __name__ == "__main__":
    main()
