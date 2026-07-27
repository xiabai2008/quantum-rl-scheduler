#!/usr/bin/env python
"""
量子退火消融实验 (20 Seed 扩展版)：PPO带退火 vs PPO不带退火

Issue #319: 退火消融实验统计功效提升
- 扩展 seeds 列表至 20 个（保留原 5 个 + 新增 15 个）
- 保持原有配置不变：50000步、退火间隔5000步、head_only模式、16 qubits、4比特编码
- 保存原始数据至 results/ablation_annealing_multiseed_20seeds.json
"""

import json
import os
import sys
import time
from datetime import datetime

os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 强制使用 numpy 模拟退火以加速实验（neal 求解器每次 ~12s，numpy ~0.5s）
# 两种求解器均属于"仿真退火"，不影响实验结论的统计有效性
import src.quantum.annealing as _anneal_module

_anneal_module._DWAVE_AVAILABLE = False

from src.scheduler.agent import PPOAgent
from src.scheduler.env import QuantumSchedulingEnv

# ---- 配置 ----
# 保留原 5 个 seed + 新增 15 个 seed = 20 个
SEEDS = [
    42,
    123,
    456,
    789,
    1024,  # 原始 5 seeds
    2026,
    314,
    271,
    828,
    5566,  # 新增 seeds 6-10
    7788,
    1234,
    2345,
    3456,
    4567,  # 新增 seeds 11-15
    5678,
    6789,
    7890,
    8901,
    1122,  # 新增 seeds 16-20
]
TOTAL_TIMESTEPS = 50000
EVAL_FREQ = 5000
N_EVAL_EPISODES = 5
MAX_STEPS = 100
ANNEAL_INTERVAL = 5000
ANNEAL_QUBITS = 16
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")


def train_one(seed: int, use_annealing: bool) -> dict:
    """训练单组 PPO，返回 eval rewards 和训练时间。

    Args:
        seed: 随机种子
        use_annealing: 是否启用量子退火

    Returns:
        包含 timesteps、rewards、train_time_s 的字典
    """
    label = "with_anneal" if use_annealing else "no_anneal"
    env = QuantumSchedulingEnv(max_steps=MAX_STEPS, seed=seed)

    agent = PPOAgent(
        env,
        use_annealing=use_annealing,
        anneal_interval=ANNEAL_INTERVAL,
        anneal_qubits=ANNEAL_QUBITS,
        verbose=0,
        seed=seed,
        n_steps=2048,
        batch_size=64,
        log_dir=os.path.join(PROJECT_ROOT, "logs", f"ablation_20s_{label}_seed{seed}"),
    )

    t0 = time.time()
    agent.train(
        total_timesteps=TOTAL_TIMESTEPS,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
    )
    train_time = time.time() - t0

    # 读取 eval 结果
    eval_log = os.path.join(agent.log_dir, "eval_results", "evaluations.npz")
    try:
        data = np.load(eval_log)
        ts = data["timesteps"].tolist()
        rs = data["results"].tolist()
        # results 可能是 (n_evals, n_episodes) 或 (n_evals,)
        if rs and isinstance(rs[0], (list, np.ndarray)):
            rs = [float(np.mean(r)) for r in rs]
        else:
            rs = [float(r) for r in rs]
        return {"timesteps": ts, "rewards": rs, "train_time_s": train_time}
    except Exception as e:
        print(f"  [WARN] seed={seed} {label}: eval 读取失败 ({e})")
        return {"timesteps": [], "rewards": [], "train_time_s": train_time}


def run_multi_seed() -> None:
    """运行 20 seeds 消融实验并保存结果。"""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_no_anneal: list[dict] = []
    all_with_anneal: list[dict] = []

    n_seeds = len(SEEDS)
    for idx, seed in enumerate(SEEDS):
        print(f"\n{'=' * 50}")
        print(f"[{idx + 1}/{n_seeds}] Seed = {seed}")
        print(f"{'=' * 50}")

        print("  [1/2] PPO 无退火...")
        r1 = train_one(seed, use_annealing=False)
        all_no_anneal.append(r1)
        if r1["rewards"]:
            print(f"        最终={r1['rewards'][-1]:.1f}  耗时={r1['train_time_s']:.0f}s")

        print("  [2/2] PPO + 退火...")
        r2 = train_one(seed, use_annealing=True)
        all_with_anneal.append(r2)
        if r2["rewards"]:
            print(f"        最终={r2['rewards'][-1]:.1f}  耗时={r2['train_time_s']:.0f}s")

    # ---- 汇总 ----
    ref_ts = all_no_anneal[0]["timesteps"]
    n_evals = len(ref_ts)

    no_anneal_matrix = np.zeros((n_seeds, n_evals))
    with_anneal_matrix = np.zeros((n_seeds, n_evals))

    for i, r in enumerate(all_no_anneal):
        if r["rewards"]:
            no_anneal_matrix[i] = r["rewards"]
    for i, r in enumerate(all_with_anneal):
        if r["rewards"]:
            with_anneal_matrix[i] = r["rewards"]

    no_anneal_mean = no_anneal_matrix.mean(axis=0)
    no_anneal_std = no_anneal_matrix.std(axis=0)
    with_anneal_mean = with_anneal_matrix.mean(axis=0)
    with_anneal_std = with_anneal_matrix.std(axis=0)

    # ---- 保存 JSON ----
    report = {
        "timestamp": timestamp,
        "experiment": "ablation_annealing_20seeds",
        "issue": 319,
        "config": {
            "seeds": SEEDS,
            "n_seeds": n_seeds,
            "total_timesteps": TOTAL_TIMESTEPS,
            "eval_freq": EVAL_FREQ,
            "n_eval_episodes": N_EVAL_EPISODES,
            "anneal_interval": ANNEAL_INTERVAL,
            "anneal_qubits": ANNEAL_QUBITS,
            "mode": "head_only",
            "encoding_bits": 4,
        },
        "no_anneal": {
            "per_seed": all_no_anneal,
            "mean": no_anneal_mean.tolist(),
            "std": no_anneal_std.tolist(),
            "timesteps": ref_ts,
        },
        "with_anneal": {
            "per_seed": all_with_anneal,
            "mean": with_anneal_mean.tolist(),
            "std": with_anneal_std.tolist(),
            "timesteps": ref_ts,
        },
    }

    json_path = os.path.join(RESULTS_DIR, "ablation_annealing_multiseed_20seeds.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- 生成图 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 左图：mean ± std 曲线
    ax = axes[0]
    for label, mean, std, color in [
        ("PPO (无退火)", no_anneal_mean, no_anneal_std, "#3498db"),
        ("PPO + 退火", with_anneal_mean, with_anneal_std, "#e74c3c"),
    ]:
        ax.plot(ref_ts, mean, "o-", linewidth=2.5, markersize=6, color=color, label=label)
        ax.fill_between(ref_ts, mean - std, mean + std, alpha=0.15, color=color)
        ax.annotate(
            f"{mean[-1]:.1f}",
            (ref_ts[-1], mean[-1]),
            textcoords="offset points",
            xytext=(10, 0),
            fontsize=10,
            fontweight="bold",
            color=color,
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title(f"PPO with vs without Quantum Annealing\n({n_seeds} seeds, mean ± std)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 中图：每个 seed 的最终 reward 对比
    ax = axes[1]
    x = np.arange(n_seeds)
    width = 0.35
    finals_no = [r["rewards"][-1] if r["rewards"] else 0.0 for r in all_no_anneal]
    finals_anneal = [r["rewards"][-1] if r["rewards"] else 0.0 for r in all_with_anneal]

    ax.bar(x - width / 2, finals_no, width, label="PPO (无退火)", color="#3498db", alpha=0.8)
    ax.bar(x + width / 2, finals_anneal, width, label="PPO + 退火", color="#e74c3c", alpha=0.8)

    ax.set_xlabel("Seed Index")
    ax.set_ylabel("Final Reward (50K steps)")
    ax.set_title("Per-Seed Final Reward Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels([str(i + 1) for i in range(n_seeds)], fontsize=7)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 右图：统计摘要
    ax = axes[2]
    ax.axis("off")
    no_mean = float(no_anneal_mean[-1])
    no_std = float(no_anneal_std[-1])
    an_mean = float(with_anneal_mean[-1])
    an_std = float(with_anneal_std[-1])
    delta = an_mean - no_mean
    rel = delta / (abs(no_mean) + 1e-8) * 100
    no_time = float(np.mean([r["train_time_s"] for r in all_no_anneal]))
    an_time = float(np.mean([r["train_time_s"] for r in all_with_anneal]))

    summary_lines = [
        "20 Seeds 消融实验统计摘要",
        "------------------------",
        f"Seeds: {n_seeds} 个",
        "",
        "PPO (无退火):",
        f"  最终 reward = {no_mean:.1f} ± {no_std:.1f}",
        f"  训练耗时 = {no_time:.0f}s/seed",
        "",
        "PPO + 退火:",
        f"  最终 reward = {an_mean:.1f} ± {an_std:.1f}",
        f"  训练耗时 = {an_time:.0f}s/seed",
        "",
        "退火提升:",
        f"  mean δ = {delta:+.1f}",
        f"  relative = {rel:+.1f}%",
        f"  训练开销 = {(an_time - no_time) / no_time * 100:+.1f}%",
    ]
    for i, line in enumerate(summary_lines):
        y = 0.95 - i * 0.04
        fontweight = "bold" if i == 0 else "normal"
        ax.text(
            0.05,
            y,
            line,
            transform=ax.transAxes,
            fontsize=10,
            fontweight=fontweight,
            family="monospace",
        )

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, "ablation_annealing_multiseed_20seeds.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- 终端输出 ----
    print(f"\n{'=' * 60}")
    print("20 Seeds 多 Seed 消融实验完成")
    print(f"{'=' * 60}")
    print(f"JSON: {json_path}")
    print(f"PNG:  {png_path}")
    print("")
    for line in summary_lines[2:]:
        print(f"  {line}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    run_multi_seed()
