#!/usr/bin/env python
"""
退火 L2 正则效应分解对照实验（Issue #353 科研诚信）

问题：退火接受门控的质量评估默认使用权重 L2 范数，可能将 L2 正则化效应
误归因于"退火对 RL 目标的优化"。本实验通过三组对照分解两类贡献：

    A. PPO-no-anneal    : 不使用退火（纯 PPO baseline）
    B. PPO-anneal-λ0    : 退火但 reg_lambda=0（QUBO 无 L2 正则，纯梯度方向）
    C. PPO-anneal-λ01   : 退火 reg_lambda=0.1（当前默认，含 L2 正则）

对比 B-A 隔离退火梯度引导效应（无正则），对比 C-B 隔离 L2 正则效应。
实验 N=20 seeds，使用 Mann-Whitney U 检验 + Cliff's delta 效应量。

使用方法：
    python scripts/evaluation/annealing_l2_decomposition.py
    python scripts/evaluation/annealing_l2_decomposition.py --quick  # 快速模式 5 seeds

输出：
    results/annealing_l2_decomposition.json  原始数据
    results/annealing_l2_decomposition_report.md  分析报告
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"
os.environ["ANNEALING_ENABLED"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import numpy as np

import src.quantum.annealing as _anneal_module

_anneal_module._DWAVE_AVAILABLE = False

from src.scheduler.agent import PPOAgent
from src.scheduler.env import QuantumSchedulingEnv

SEEDS_FULL = [
    42, 123, 456, 789, 1024,
    2026, 314, 271, 828, 5566,
    7788, 1234, 2345, 3456, 4567,
    5678, 6789, 7890, 8901, 1122,
]
SEEDS_QUICK = [42, 123, 456, 789, 1024]

TOTAL_TIMESTEPS = 50000
EVAL_FREQ = 10000
N_EVAL_EPISODES = 10
MAX_STEPS = 100
ANNEAL_INTERVAL = 5000
ANNEAL_QUBITS = 16


def train_one(seed: int, condition: str) -> dict:
    """训练单组 PPO，返回最终评估指标。

    Args:
        seed: 随机种子
        condition: "no_anneal" | "anneal_l0" | "anneal_l01"

    Returns:
        包含 mean_reward, std_reward, success_rate, train_time_s 的字典
    """
    use_annealing = condition != "no_anneal"
    reg_lambda = 0.0 if condition == "anneal_l0" else 0.1

    env = QuantumSchedulingEnv(max_steps=MAX_STEPS, seed=seed)
    agent_kwargs: dict = {
        "env": env,
        "use_annealing": use_annealing,
        "anneal_interval": ANNEAL_INTERVAL,
        "anneal_qubits": ANNEAL_QUBITS,
        "verbose": 0,
        "seed": seed,
        "n_steps": 2048,
        "batch_size": 64,
        "log_dir": os.path.join(
            PROJECT_ROOT, "logs", f"l2decomp_{condition}_seed{seed}"
        ),
    }
    if use_annealing:
        agent_kwargs["anneal_reg_lambda"] = reg_lambda

    agent = PPOAgent(**agent_kwargs)

    if use_annealing and hasattr(agent, "annealing_optimizer"):
        agent.annealing_optimizer._reg_lambda = reg_lambda

    t0 = time.time()
    agent.train(
        total_timesteps=TOTAL_TIMESTEPS,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
    )
    train_time = time.time() - t0

    eval_results = agent.evaluate(num_episodes=N_EVAL_EPISODES, deterministic=True)

    return {
        "seed": seed,
        "condition": condition,
        "mean_reward": float(eval_results["mean_reward"]),
        "std_reward": float(eval_results["std_reward"]),
        "success_rate": float(eval_results.get("success_rate", 0.0)),
        "train_time_s": train_time,
    }


def mann_whitney_u(a: list[float], b: list[float]) -> tuple[float, float]:
    """Mann-Whitney U 检验，返回 (u_stat, p_value)。"""
    from scipy import stats

    u_stat, p_value = stats.mannwhitneyu(a, b, alternative="two-sided")
    return float(u_stat), float(p_value)


def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Cliff's delta 效应量：|d|<0.147 negligible, <0.33 small, <0.474 medium, else large。"""
    n1, n2 = len(a), len(b)
    gt = sum(1 for x in a for y in b if x > y)
    lt = sum(1 for x in a for y in b if x < y)
    return (gt - lt) / (n1 * n2)


def run_experiment(seeds: list[int]) -> dict:
    """运行完整对照实验。"""
    conditions = ["no_anneal", "anneal_l0", "anneal_l01"]
    all_results: list[dict] = []

    total = len(seeds) * len(conditions)
    done = 0

    for condition in conditions:
        for seed in seeds:
            done += 1
            print(
                f"[{done}/{total}] condition={condition}, seed={seed} ...",
                flush=True,
            )
            try:
                result = train_one(seed, condition)
                all_results.append(result)
                print(
                    f"  reward={result['mean_reward']:.1f}±{result['std_reward']:.1f}, "
                    f"success={result['success_rate']:.2%}, "
                    f"time={result['train_time_s']:.1f}s"
                )
            except Exception as e:
                print(f"  FAILED: {e}")
                all_results.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "mean_reward": None,
                        "std_reward": None,
                        "success_rate": None,
                        "train_time_s": None,
                        "error": str(e),
                    }
                )

    groups: dict[str, list[float]] = {}
    for cond in conditions:
        groups[cond] = [
            r["mean_reward"]
            for r in all_results
            if r["condition"] == cond and r["mean_reward"] is not None
        ]

    stats_results = {}
    pairs = [
        ("anneal_l0", "no_anneal", "梯度引导效应(λ=0 vs baseline)"),
        ("anneal_l01", "anneal_l0", "L2正则效应(λ=0.1 vs λ=0)"),
        ("anneal_l01", "no_anneal", "退火总效应(λ=0.1 vs baseline)"),
    ]
    for a_name, b_name, desc in pairs:
        a_vals = groups.get(a_name, [])
        b_vals = groups.get(b_name, [])
        if len(a_vals) >= 2 and len(b_vals) >= 2:
            u, p = mann_whitney_u(a_vals, b_vals)
            d = cliffs_delta(a_vals, b_vals)
            stats_results[f"{a_name}_vs_{b_name}"] = {
                "description": desc,
                "u_stat": u,
                "p_value": p,
                "cliffs_delta": d,
                "significant_005": p < 0.05,
                "mean_a": float(np.mean(a_vals)),
                "mean_b": float(np.mean(b_vals)),
                "n_a": len(a_vals),
                "n_b": len(b_vals),
            }

    return {
        "timestamp": datetime.now().isoformat(),
        "total_timesteps": TOTAL_TIMESTEPS,
        "max_steps": MAX_STEPS,
        "seeds": seeds,
        "conditions": conditions,
        "raw_results": all_results,
        "statistics": stats_results,
    }


def generate_report(data: dict) -> str:
    """生成 Markdown 分析报告。"""
    lines = [
        "# 退火 L2 正则效应分解报告（Issue #353）",
        "",
        f"实验时间: {data['timestamp']}",
        f"训练步数: {data['total_timesteps']}",
        f"Seed 数量: {len(data['seeds'])}",
        "",
        "## 实验设计",
        "",
        "| 条件 | reg_lambda | 说明 |",
        "|------|-----------|------|",
        "| A: no_anneal | N/A | 纯 PPO baseline |",
        "| B: anneal_l0 | 0.0 | 退火无 L2 正则（纯梯度引导） |",
        "| C: anneal_l01 | 0.1 | 退火默认配置（含 L2 正则） |",
        "",
        "- **B-A** → 退火梯度引导的真实优化效应（无正则干扰）",
        "- **C-B** → 纯 L2 正则化效应（权重衰减）",
        "- **C-A** → 退火总效应（当前报告声称的\"退火+88.3%\"）",
        "",
        "## 描述统计",
        "",
        "| 条件 | N | Mean Reward | Std |",
        "|------|---|------------|-----|",
    ]

    for cond in data["conditions"]:
        vals = [
            r["mean_reward"]
            for r in data["raw_results"]
            if r["condition"] == cond and r["mean_reward"] is not None
        ]
        if vals:
            lines.append(
                f"| {cond} | {len(vals)} | {np.mean(vals):.1f} | {np.std(vals):.1f} |"
            )

    lines.extend([
        "",
        "## 统计检验（Mann-Whitney U + Cliff's delta）",
        "",
        "| 对比 | U | p-value | Cliff's δ | 显著(p<0.05) | 效应量 | 解读 |",
        "|------|---|---------|-----------|-------------|--------|------|",
    ])

    for _key, s in data["statistics"].items():
        effect = "negligible"
        if abs(s["cliffs_delta"]) >= 0.474:
            effect = "large"
        elif abs(s["cliffs_delta"]) >= 0.33:
            effect = "medium"
        elif abs(s["cliffs_delta"]) >= 0.147:
            effect = "small"
        lines.append(
            f"| {s['description']} | {s['u_stat']:.1f} | {s['p_value']:.4f} | "
            f"{s['cliffs_delta']:.3f} | {'✅' if s['significant_005'] else '❌'} | "
            f"{effect} | Δ={s['mean_a'] - s['mean_b']:.1f} |"
        )

    lines.extend([
        "",
        "## 结论",
        "",
        "（自动实验完成后填写）",
        "",
        "### 诚实声明",
        "",
        "本实验严格分解了退火优化中的两类效应，报告中引用的\"退火对 RL 性能提升\"数据",
        "必须基于 B-A 对比（排除 L2 正则效应）。若 B-A 不显著，则所谓\"退火加速 RL\"",
        "实质为 L2 正则化（权重衰减）带来的训练稳定性改善，并非量子退火对 RL 目标的",
        "直接优化。无论结果正负均如实报告，不做选择性报道。",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="退火 L2 正则效应分解对照实验")
    parser.add_argument(
        "--quick", action="store_true", help="快速模式：仅 5 seeds 用于验证"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "results"),
        help="输出目录",
    )
    args = parser.parse_args()

    seeds = SEEDS_QUICK if args.quick else SEEDS_FULL
    print(f"{'='*60}")
    print("退火 L2 分解对照实验 (Issue #353)")
    print(f"Seeds: {len(seeds)}, Timesteps: {TOTAL_TIMESTEPS}")
    print(f"{'='*60}")

    os.makedirs(args.output_dir, exist_ok=True)

    data = run_experiment(seeds)

    json_path = os.path.join(args.output_dir, "annealing_l2_decomposition.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n原始数据已保存: {json_path}")

    report = generate_report(data)
    report_path = os.path.join(args.output_dir, "annealing_l2_decomposition_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"分析报告已保存: {report_path}")


if __name__ == "__main__":
    main()
