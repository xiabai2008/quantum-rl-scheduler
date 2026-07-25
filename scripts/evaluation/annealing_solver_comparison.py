#!/usr/bin/env python
"""
QUBO 求解器对比实验 (Issue #111)

对比 4 种策略的 QUBO 求解器效果：
    A: QUBO + neal 退火求解
    B: QUBO + numpy 模拟退火
    C: QUBO + 随机采样
    D: 直接梯度下降（纯 PPO，不含量子）

实验配置：
    - 5 个随机种子 [42, 123, 456, 789, 1024]
    - 小型 PPO 网络（隐藏层 16-16）
    - 每 seed 运行 4 种策略，记录 eval reward 曲线
    - 策略 D 不涉及量子退火 QUBO，仅 A/B/C 比较
    - Welch t 检验比较最终 reward 差异

输出：
    - results/reports/annealing_solver_comparison.md
    - results/annealing_solver_comparison_*.json
    - results/annealing_solver_comparison_*.png
"""

import argparse
import json
import math
import os
import logging
import sys
import time
from datetime import datetime
from typing import Any

import matplotlib
import numpy as np
from scipy import stats
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

from src.quantum import annealing as _annealing_mod
from src.quantum.annealing import QuantumAnnealingOptimizer
from src.scheduler.agent import PPOAgent
from src.scheduler.env import QuantumSchedulingEnv

# ---- 实验配置 ----
DEFAULT_SEEDS = [42, 123, 456, 789, 1024]
DEFAULT_TIMESTEPS = 50000
EVAL_FREQ = 5000
N_EVAL_EPISODES = 5
MAX_STEPS = 100
ANNEAL_INTERVAL = 5000
ANNEAL_QUBITS = 16
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
REPORTS_DIR = os.path.join(RESULTS_DIR, "reports")

STRATEGIES = ["A", "B", "C", "D"]
STRATEGY_LABELS = {
    "A": "QUBO+neal",
    "B": "QUBO+numpy SA",
    "C": "QUBO+random",
    "D": "Direct GD (PPO)",
}
STRATEGY_COLORS = {
    "A": "#e74c3c",
    "B": "#3498db",
    "C": "#2ecc71",
    "D": "#9b59b6",
}


def random_sample_qubo(qubo_matrix: np.ndarray, num_samples: int = 1000) -> str:
    """随机采样求解 QUBO 矩阵，返回最优比特串

    Args:
        qubo_matrix: QUBO 矩阵，形状 (n, n)。
        num_samples: 采样次数。

    Returns:
        最优比特字符串，长度 n。
    """
    n = qubo_matrix.shape[0]
    best_energy = float("inf")
    best_bits = np.zeros(n, dtype=np.float64)
    for _ in range(num_samples):
        bits = np.random.randint(0, 2, n).astype(np.float64)
        energy = QuantumAnnealingOptimizer._compute_qubo_energy(  # type: ignore[attr-defined]  依赖内部实现，后续重构时需同步更新
            bits, qubo_matrix
        )
        if energy < best_energy:
            best_energy = energy
            best_bits = bits.copy()
    return "".join(str(int(b)) for b in best_bits)


class _AnnealingCallbackProxy(BaseCallback):
    """退火回调代理：在训练过程中定期执行退火优化

    根据策略选择不同求解方式（通过 strategy 参数控制）：
        - "A": 调用 optimizer.anneal（neal 求解）
        - "B": 调用 numpy 模拟退火
        - "C": 调用随机采样求解
    """

    def __init__(
        self,
        optimizer: QuantumAnnealingOptimizer,
        interval: int = 1000,
        strategy: str = "A",
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.optimizer = optimizer
        self.interval = interval
        self.strategy = strategy

    def _on_step(self) -> bool:
        if self.n_calls % self.interval == 0 and self.n_calls > 0:
            try:
                if self.strategy == "B":
                    original_use_dw = self.optimizer.use_dw
                    self.optimizer.use_dw = False
                    self.optimizer.optimize_policy(self.model, head_only=True)
                    self.optimizer.use_dw = original_use_dw
                elif self.strategy == "C":
                    original_anneal = self.optimizer.anneal
                    self.optimizer.anneal = lambda qubo: random_sample_qubo(qubo, num_samples=1000)
                    self.optimizer.optimize_policy(self.model, head_only=True)
                    self.optimizer.anneal = original_anneal
                    # monkey-patch 不会设置 _last_solver，手动修正
                    self.optimizer._last_solver = "random"  # type: ignore[attr-defined]
                    if hasattr(self.optimizer, "_last_anneal_stats"):
                        self.optimizer._last_anneal_stats["solver"] = "random"  # type: ignore[attr-defined]
                else:
                    self.optimizer.optimize_policy(self.model, head_only=True)
            except Exception as e:
                if self.verbose:
                    print(f"[WARN] 退火失败 ({e})")
        return True


def _build_small_ppo(agent: PPOAgent, env: QuantumSchedulingEnv) -> PPO:
    """用 agent 参数构建小型 [16, 16] 隐藏层 PPO 模型

    Args:
        agent: PPOAgent 实例（用于提取超参数）。
        env:   训练环境。

    Returns:
        构建好的 PPO 模型实例。
    """
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=agent.learning_rate,
        n_steps=agent.n_steps,
        batch_size=agent.batch_size,
        n_epochs=agent.n_epochs,
        gamma=agent.gamma,
        gae_lambda=agent.gae_lambda,
        clip_range=agent.clip_range,
        ent_coef=agent.ent_coef,
        vf_coef=agent.vf_coef,
        max_grad_norm=agent.max_grad_norm,
        verbose=agent.verbose,
        seed=agent.seed,
        tensorboard_log=agent.log_dir,
        policy_kwargs={"net_arch": [16, 16]},
    )


def _read_eval(log_dir: str) -> dict[str, Any]:
    """从 eval_results/evaluations.npz 读取评估结果"""
    eval_log = os.path.join(log_dir, "eval_results", "evaluations.npz")
    try:
        data = np.load(eval_log)
        ts = data["timesteps"].tolist()
        rs = data["results"].tolist()
        if rs and isinstance(rs[0], (list, np.ndarray)):
            rs = [float(np.mean(r)) for r in rs]
        else:
            rs = [float(r) for r in rs]
        return {"timesteps": ts, "rewards": rs, "train_time_s": 0.0}
    except Exception as e:
        print(f"  [WARN] eval 读取失败 ({e})")
        return {"timesteps": [], "rewards": [], "train_time_s": 0.0}


def _train_with_strategy(
    env: QuantumSchedulingEnv, seed: int, strategy: str, total_timesteps: int
) -> dict[str, Any]:
    """用指定策略训练 PPO 并返回结果"""
    agent = PPOAgent(
        env,
        use_annealing=True,
        anneal_interval=ANNEAL_INTERVAL,
        anneal_qubits=ANNEAL_QUBITS,
        verbose=0,
        seed=seed,
        n_steps=2048,
        batch_size=64,
        log_dir=os.path.join(PROJECT_ROOT, "logs", f"solver_cmp_{strategy}_seed{seed}"),
    )
    agent.model = _build_small_ppo(agent, env)

    cb = _AnnealingCallbackProxy(
        optimizer=agent.annealing_optimizer,
        interval=ANNEAL_INTERVAL,
        strategy=strategy,
        verbose=0,
    )

    t0 = time.time()
    agent.train(
        total_timesteps=total_timesteps,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        extra_callbacks=[cb],
    )
    train_time = time.time() - t0

    r = _read_eval(agent.log_dir)
    r["train_time_s"] = train_time
    # 记录退火诊断信息（接受率、求解器等）
    stats = getattr(cb.optimizer, "_last_anneal_stats", None)
    if stats:
        r["anneal_stats"] = stats
    return r


def evaluate_qubo_solvers(policy_net: Any, optimizer: QuantumAnnealingOptimizer) -> dict[str, Any]:
    """评估给定策略下 A/B/C 三种 QUBO 求解器表现

    Args:
        policy_net:  策略网络（nn.Module）。
        optimizer:   QuantumAnnealingOptimizer 实例。

    Returns:
        包含各求解器能量、gap、比特串的字典。
    """
    weights, _shapes = optimizer._extract_weights(  # type: ignore[attr-defined]  依赖内部实现，后续重构时需同步更新
        policy_net
    )
    num_weights = sum(w.size for w in weights)
    n_bits_per_weight = max(1, optimizer.num_qubits // 4)
    qubo = optimizer.network_to_qubo(weights)
    n = qubo.shape[0]
    print(
        f"[QUBO诊断] 权重数={num_weights}, 每权重比特={n_bits_per_weight}, "
        f"QUBO规模={n}x{n} (= {num_weights} x {n_bits_per_weight})"
    )

    # A: neal（默认），失败则回退 numpy SA
    energy_a: float | None = None
    bitstring_a = ""
    solver_a = "unknown"
    try:
        bitstring_a = optimizer.anneal(qubo)
        solver_a = getattr(optimizer, "_last_solver", "unknown")
        bits_a = np.array([int(b) for b in bitstring_a], dtype=np.float64)
        energy_a = optimizer._compute_qubo_energy(  # type: ignore[attr-defined]  依赖内部实现，后续重构时需同步更新
            bits_a, qubo
        )
    except Exception:
        energy_a = None

    # B: numpy SA
    bitstring_b = optimizer._numpy_simulated_annealing(  # type: ignore[attr-defined]  依赖内部实现，后续重构时需同步更新
        qubo
    )
    bits_b = np.array([int(b) for b in bitstring_b], dtype=np.float64)
    energy_b = optimizer._compute_qubo_energy(  # type: ignore[attr-defined]  依赖内部实现，后续重构时需同步更新
        bits_b, qubo
    )

    # C: random
    bitstring_c = random_sample_qubo(qubo, num_samples=1000)
    bits_c = np.array([int(b) for b in bitstring_c], dtype=np.float64)
    energy_c = optimizer._compute_qubo_energy(  # type: ignore[attr-defined]  依赖内部实现，后续重构时需同步更新
        bits_c, qubo
    )

    # 计算各求解器相对于最优解的 gap
    energies = [e for e in (energy_a, energy_b, energy_c) if e is not None]
    best_energy = min(energies) if energies else 1.0

    def _gap(energy: float | None) -> float | None:
        if energy is None or best_energy == 0.0:
            return None
        return (energy - best_energy) / abs(best_energy)

    return {
        "qubo_size": n,
        "qubo_diagnostic": {
            "num_weights": num_weights,
            "n_bits_per_weight": n_bits_per_weight,
            "qubo_size_explained": f"{num_weights} x {n_bits_per_weight} = {n}",
        },
        "best_energy": best_energy,
        "solver_a": solver_a,
        "A": {
            "energy": energy_a,
            "gap": _gap(energy_a),
            "bitstring": bitstring_a,
        },
        "B": {
            "energy": energy_b,
            "gap": _gap(energy_b),
            "bitstring": bitstring_b,
        },
        "C": {
            "energy": energy_c,
            "gap": _gap(energy_c),
            "bitstring": bitstring_c,
        },
    }


def _plot_results(report: dict[str, Any], timestamp: str) -> None:
    """绘制 reward 曲线和 QUBO 能量对比图"""
    ref_ts = report["timesteps"]
    n_seeds = len(report["config"]["seeds"])

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # 左图：mean +- std reward 曲线
    ax = axes[0]
    for s in STRATEGIES:
        mean = np.array(report["rewards"][s]["mean"])
        std = np.array(report["rewards"][s]["std"])
        ax.plot(
            ref_ts,
            mean,
            "o-",
            linewidth=2.5,
            markersize=6,
            color=STRATEGY_COLORS[s],
            label=STRATEGY_LABELS[s],
        )
        ax.fill_between(ref_ts, mean - std, mean + std, alpha=0.15, color=STRATEGY_COLORS[s])
        ax.annotate(
            f"{mean[-1]:.1f}",
            (ref_ts[-1], mean[-1]),
            textcoords="offset points",
            xytext=(10, 0),
            fontsize=10,
            fontweight="bold",
            color=STRATEGY_COLORS[s],
        )

    ax.set_xlabel("Training Steps")
    ax.set_ylabel("Mean Eval Reward")
    ax.set_title(f"QUBO Solver Comparison\n({n_seeds} seeds, mean +- std)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：QUBO 能量对比（每 seed 一组柱）
    ax = axes[1]
    qubo_data = report["qubo"]
    seeds = [q["seed"] for q in qubo_data]
    x = np.arange(len(seeds))
    width = 0.25

    energies_a = [q["A"]["energy"] for q in qubo_data]
    energies_b = [q["B"]["energy"] for q in qubo_data]
    energies_c = [q["C"]["energy"] for q in qubo_data]

    ax.bar(x - width, energies_a, width, label="A: neal", color=STRATEGY_COLORS["A"], alpha=0.8)
    ax.bar(x, energies_b, width, label="B: numpy SA", color=STRATEGY_COLORS["B"], alpha=0.8)
    ax.bar(x + width, energies_c, width, label="C: random", color=STRATEGY_COLORS["C"], alpha=0.8)

    ax.set_xlabel("Seed")
    ax.set_ylabel("QUBO Energy (lower is better)")
    ax.set_title("QUBO Solving Quality per Seed")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds])
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    png_path = os.path.join(RESULTS_DIR, f"annealing_solver_comparison_{timestamp}.png")
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"PNG: {png_path}")


def _generate_markdown(report: dict[str, Any], path: str) -> None:
    """生成 Markdown 报告文件"""
    cfg = report["config"]

    lines: list[str] = [
        "# QUBO 求解器对比实验 (Issue #111)",
        "",
        f"- **生成时间**: {report['timestamp']}",
        f"- **Seeds**: {cfg['seeds']}",
        f"- **训练总步数**: {cfg['total_timesteps']}",
        f"- **评估频率**: {cfg['eval_freq']}",
        f"- **退火间隔**: {cfg['anneal_interval']}",
        f"- **QUBO 比特数**: {cfg['anneal_qubits']}",
        f"- **PPO 隐藏层**: {cfg['hidden_layers']}",
        "",
        "## 1. 实验设计",
        "",
        "对比 4 种策略的 QUBO 求解器效果：",
        "",
        "| 策略 | 描述 |",
        "|------|------|",
        "| A | QUBO + neal 退火求解 |",
        "| B | QUBO + numpy 模拟退火 |",
        "| C | QUBO + 随机采样 |",
        "| D | 直接梯度下降（纯 PPO，不含量子） |",
        "",
        "## 2. QUBO 求解质量对比",
        "",
    ]

    # QUBO 表格
    lines.extend(
        [
            "| Seed | QUBO 大小 | A (neal) 能量 | A gap | B (numpy SA) 能量 | B gap | C (random) 能量 | C gap |",
            "|------|-----------|---------------|-------|-------------------|-------|-----------------|-------|",
        ]
    )
    for q in report["qubo"]:
        a_e = q["A"]["energy"]
        a_g = q["A"]["gap"]
        b_e = q["B"]["energy"]
        b_g = q["B"]["gap"]
        c_e = q["C"]["energy"]
        c_g = q["C"]["gap"]
        a_e_str = f"{a_e:.4f}" if a_e is not None else "N/A"
        a_g_str = f"{a_g:.4f}" if a_g is not None else "N/A"
        lines.append(
            f"| {q['seed']} | {q['qubo_size']} | "
            f"{a_e_str} | "
            f"{a_g_str} | "
            f"{b_e:.4f} | {b_g:.4f} | "
            f"{c_e:.4f} | {c_g:.4f} |"
        )

    lines.extend(
        [
            "",
            "> **说明**: gap 为各求解器相对最优解的归一化偏离，计算公式 `(energy - best) / |best|`。",
            "",
            "### QUBO 规模诊断",
            "",
        ]
    )
    # QUBO 规模诊断（解释为何 QUBO 比特数较大）
    first_qubo = report["qubo"][0] if report["qubo"] else {}
    diag = first_qubo.get("qubo_diagnostic", {})
    if diag:
        lines.append(f"- 权重参数数: {diag.get('num_weights', 'N/A')}")
        lines.append(f"- 每权重编码比特: {diag.get('n_bits_per_weight', 'N/A')}")
        lines.append(f"- QUBO 规模 = {diag.get('qubo_size_explained', 'N/A')}")
        lines.append(
            "- 说明: QUBO 规模 = 网络权重数 x 每权重编码比特数。"
            "PPO ActorCriticPolicy 包含 mlp_extractor + action_net + value_net，"
            "参数量较大导致 QUBO 规模远超单层网络。"
        )
    solver_a = first_qubo.get("solver_a", "unknown")
    lines.append(f"- 策略 A 实际求解器: **{solver_a}**")
    if solver_a == "neal":
        lines.append("  - neal (D-Wave SimulatedAnnealingSampler) 已正确调用，未回退到 numpy SA")
    else:
        lines.append(f"  - 注意: 实际使用 {solver_a}，非 neal（可能 neal 未安装或回退）")

    # 退火接受率诊断（如果训练阶段有记录）
    lines.extend(
        [
            "",
            "### 退火接受率诊断",
            "",
            "| 策略 | 求解器 | 接受次数 | 拒绝次数 | 接受率 |",
            "|------|--------|---------|---------|--------|",
        ]
    )
    for s in ("A", "B", "C"):
        # 从 per_seed 数据中提取最后一个 seed 的退火统计
        per_seed = report["rewards"][s].get("per_seed", [])
        if per_seed and isinstance(per_seed[-1], dict) and "anneal_stats" in per_seed[-1]:
            stats = per_seed[-1]["anneal_stats"]
            lines.append(
                f"| {STRATEGY_LABELS[s]} | {stats.get('solver', 'N/A')} | "
                f"{stats.get('accepted', 'N/A')} | {stats.get('rejected', 'N/A')} | "
                f"{stats.get('accept_rate', 0):.1%} |"
            )
        else:
            lines.append(f"| {STRATEGY_LABELS[s]} | N/A | N/A | N/A | N/A |")
    lines.append(
        "- 若接受率为 0%，说明退火结果均被接受准则拒绝（loss 未下降），"
        "训练轨迹退化为纯 PPO，解释为何不同策略的 Reward 可能相同。"
    )

    lines.extend(
        [
            "",
            "## 3. 各策略 Reward 对比",
            "",
        ]
    )

    # Reward 对比表
    lines.extend(
        [
            "| 策略 | 最终 Reward (mean+-std) | 最佳 mean |",
            "|------|------------------------|-----------|",
        ]
    )
    for s in STRATEGIES:
        finals = report["rewards"][s]["final_rewards"]
        mean_final = float(np.mean(finals))
        std_final = float(np.std(finals))
        best_mean = float(np.max(report["rewards"][s]["mean"]))
        lines.append(
            f"| {STRATEGY_LABELS[s]} | {mean_final:.1f} +- {std_final:.1f} | {best_mean:.1f} |"
        )

    lines.extend(
        [
            "",
            "## 4. Welch t 检验结果",
            "",
            "| 对比 | t 统计量 | p 值 | 结论 (alpha=0.05) |",
            "|------|----------|------|-------------------|",
        ]
    )
    for pair, res in report["welch_ttest"].items():
        sig = "显著" if res["significant_at_0_05"] else "不显著"
        lines.append(f"| {pair} | {res['statistic']:.4f} | {res['p_value']:.4f} | {sig} |")

    lines.extend(
        [
            "",
            "## 5. 结论",
            "",
            "### 结论 1：QUBO 求解质量差异分析",
            "",
        ]
    )

    # 统计 QUBO gap 平均值
    avg_gaps = {s: [] for s in ("A", "B", "C")}
    for q in report["qubo"]:
        for s in ("A", "B", "C"):
            g = q[s]["gap"]
            if g is not None:
                avg_gaps[s].append(g)
    gap_means = {s: float(np.mean(v)) if v else float("nan") for s, v in avg_gaps.items()}

    best_solver = min((s for s in ("A", "B", "C")), key=lambda s: gap_means.get(s, float("inf")))
    lines.append(
        f"- 平均 gap 最优的是 **{best_solver} ({STRATEGY_LABELS[best_solver]})**，"
        f"平均 gap = {gap_means[best_solver]:.4f}。"
    )
    # 比较退火求解器 vs 随机采样
    sa_solvers = [
        s for s in ("A", "B") if gap_means.get(s) is not None and not math.isnan(gap_means[s])
    ]
    if sa_solvers and gap_means.get("C") is not None and not math.isnan(gap_means["C"]):
        avg_sa_gap = sum(gap_means[s] for s in sa_solvers) / len(sa_solvers)
        if gap_means["C"] > avg_sa_gap:
            lines.append(
                "- 随机采样 (C) 的 gap 高于退火求解器 (A/B)，说明结构化 QUBO 求解具有明显优势。"
            )
        else:
            lines.append(
                "- 随机采样 (C) 的 gap 不高于退火求解器 (A/B)，"
                "说明 QUBO 求解器在该问题上未体现明显优势。"
            )
    else:
        lines.append("- 退火求解器 (A/B) 与随机 (C) 的 gap 数据不完整。")

    lines.extend(
        [
            "",
            "### 结论 2：退火对训练效果的影响",
            "",
        ]
    )
    a_final = float(np.mean(report["rewards"]["A"]["final_rewards"]))
    d_final = float(np.mean(report["rewards"]["D"]["final_rewards"]))
    diff = a_final - d_final
    rel = diff / (abs(d_final) + 1e-8) * 100
    lines.append(f"- 策略 A 最终 reward = {a_final:.1f}，策略 D = {d_final:.1f}。")
    lines.append(f"- 差异 = {diff:+.1f} (相对 {rel:+.1f}%)。")
    welch_ad = report["welch_ttest"]["A_vs_D"]
    if welch_ad["significant_at_0_05"]:
        lines.append("- Welch t 检验显示 A 与 D 间**差异显著** (p < 0.05)。")
    else:
        lines.append("- Welch t 检验显示 A 与 D 间**差异不显著** (p >= 0.05)。")

    lines.append("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def run_experiment(seeds: list[int], total_timesteps: int) -> dict[str, Any]:
    """运行完整实验：遍历每个 seed 执行所有策略

    Args:
        seeds:            随机种子列表。
        total_timesteps:  训练总步数。

    Returns:
        包含完整实验结果的字典。
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_results: dict[str, list[dict[str, Any]]] = {s: [] for s in STRATEGIES}
    qubo_results: list[dict[str, Any]] = []

    for seed in seeds:
        print(f"\n{'=' * 60}")
        print(f"Seed = {seed}")
        print(f"{'=' * 60}")

        env = QuantumSchedulingEnv(max_steps=MAX_STEPS, seed=seed)

        # ---- 策略 D（纯 PPO）----
        print("  [D] 直接梯度下降 (纯 PPO)...")
        agent_d = PPOAgent(
            env,
            use_annealing=False,
            verbose=0,
            seed=seed,
            n_steps=2048,
            batch_size=64,
            log_dir=os.path.join(PROJECT_ROOT, "logs", f"solver_cmp_D_seed{seed}"),
        )
        agent_d.model = _build_small_ppo(agent_d, env)
        t0 = time.time()
        agent_d.train(
            total_timesteps=total_timesteps,
            eval_freq=EVAL_FREQ,
            n_eval_episodes=N_EVAL_EPISODES,
        )
        r_d = _read_eval(agent_d.log_dir)
        r_d["train_time_s"] = time.time() - t0
        all_results["D"].append(r_d)
        final_r = r_d["rewards"][-1] if r_d["rewards"] else float("nan")
        print(f"        最终 reward={final_r:.1f}  耗时={r_d['train_time_s']:.0f}s")

        # 用 D 的策略网络构建 QUBO 问题
        policy_net_d = QuantumAnnealingOptimizer._get_full_policy(  # type: ignore[attr-defined]  依赖内部实现，后续重构时需同步更新
            agent_d.model
        )
        optimizer = QuantumAnnealingOptimizer(num_qubits=ANNEAL_QUBITS)
        qubo_info = evaluate_qubo_solvers(policy_net_d, optimizer)
        qubo_info["seed"] = seed
        qubo_results.append(qubo_info)

        # ---- 策略 A ----
        print("  [A] QUBO + neal...")
        r_a = _train_with_strategy(env, seed, "A", total_timesteps)
        all_results["A"].append(r_a)
        final_r = r_a["rewards"][-1] if r_a["rewards"] else float("nan")
        print(f"        最终 reward={final_r:.1f}  耗时={r_a['train_time_s']:.0f}s")

        # ---- 策略 B ----
        print("  [B] QUBO + numpy SA...")
        r_b = _train_with_strategy(env, seed, "B", total_timesteps)
        all_results["B"].append(r_b)
        final_r = r_b["rewards"][-1] if r_b["rewards"] else float("nan")
        print(f"        最终 reward={final_r:.1f}  耗时={r_b['train_time_s']:.0f}s")

        # ---- 策略 C ----
        print("  [C] QUBO + random...")
        r_c = _train_with_strategy(env, seed, "C", total_timesteps)
        all_results["C"].append(r_c)
        final_r = r_c["rewards"][-1] if r_c["rewards"] else float("nan")
        print(f"        最终 reward={final_r:.1f}  耗时={r_c['train_time_s']:.0f}s")

    # ---- 汇总 reward 数据 ----
    ref_ts = all_results["D"][0]["timesteps"] if all_results["D"][0]["timesteps"] else []
    n_evals = len(ref_ts)
    n_seeds = len(seeds)

    reward_matrix: dict[str, np.ndarray] = {}
    for s in STRATEGIES:
        if n_evals > 0:
            mat = np.zeros((n_seeds, n_evals))
            for i, r in enumerate(all_results[s]):
                if r["rewards"]:
                    mat[i] = r["rewards"]
        else:
            mat = np.zeros((n_seeds, 0))
        reward_matrix[s] = mat

    reward_mean = {s: mat.mean(axis=0) for s, mat in reward_matrix.items()}
    reward_std = {s: mat.std(axis=0) for s, mat in reward_matrix.items()}

    # ---- Welch t 检验（比较最终 reward）----
    final_rewards = {
        s: [r["rewards"][-1] for r in all_results[s] if r["rewards"]] for s in STRATEGIES
    }
    welch_a_c = stats.ttest_ind(
        np.array(final_rewards["A"]),
        np.array(final_rewards["C"]),
        equal_var=False,
    )
    welch_a_d = stats.ttest_ind(
        np.array(final_rewards["A"]),
        np.array(final_rewards["D"]),
        equal_var=False,
    )

    report: dict[str, Any] = {
        "timestamp": timestamp,
        "config": {
            "seeds": seeds,
            "total_timesteps": total_timesteps,
            "eval_freq": EVAL_FREQ,
            "n_eval_episodes": N_EVAL_EPISODES,
            "anneal_interval": ANNEAL_INTERVAL,
            "anneal_qubits": ANNEAL_QUBITS,
            "hidden_layers": [16, 16],
        },
        "rewards": {
            s: {
                "per_seed": all_results[s],
                "mean": reward_mean[s].tolist(),
                "std": reward_std[s].tolist(),
                "final_rewards": final_rewards[s],
            }
            for s in STRATEGIES
        },
        "qubo": qubo_results,
        "welch_ttest": {
            "A_vs_C": {
                "statistic": float(welch_a_c.statistic),
                "p_value": float(welch_a_c.pvalue),
                "significant_at_0_05": bool(welch_a_c.pvalue < 0.05),
            },
            "A_vs_D": {
                "statistic": float(welch_a_d.statistic),
                "p_value": float(welch_a_d.pvalue),
                "significant_at_0_05": bool(welch_a_d.pvalue < 0.05),
            },
        },
        "timesteps": ref_ts,
    }

    json_path = os.path.join(RESULTS_DIR, f"annealing_solver_comparison_{timestamp}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # ---- 绘图 ----
    _plot_results(report, timestamp)

    # ---- 生成 Markdown 报告 ----
    md_path = os.path.join(REPORTS_DIR, "annealing_solver_comparison.md")
    _generate_markdown(report, md_path)

    print(f"\n{'=' * 60}")
    print("QUBO 求解器对比实验完成")
    print(f"{'=' * 60}")
    print(f"JSON: {json_path}")
    print(f"Report: {md_path}")
    return report


def main() -> None:
    """主函数入口"""
    # 启用量子加速（原为模块级设置，移入函数避免副作用）
    
# Check if neal is actually available (not falling back to numpy SA)
try:
    import neal
    NEAL_AVAILABLE = True
    logging.info("neal package is available and will be used for Strategy A")
except ImportError:
    NEAL_AVAILABLE = False
    logging.warning("neal package not available - Strategy A will fall back to numpy SA")

os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"
    _annealing_mod.QUANTUM_ACCELERATION_ENABLED = True

    parser = argparse.ArgumentParser(description="QUBO 求解器对比实验 (Issue #111)")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=DEFAULT_SEEDS,
        help=f"随机种子列表，默认 {DEFAULT_SEEDS}",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"训练总步数，默认 {DEFAULT_TIMESTEPS}",
    )
    args = parser.parse_args()

    run_experiment(seeds=args.seeds, total_timesteps=args.timesteps)


if __name__ == "__main__":
    main()
