#!/usr/bin/env python
"""
退火编码精度对比实验（Issue #240）

验证"退火效果不显著是编码精度问题还是方法论问题"。

实验设计：
    - 4 种 QUBO 编码精度：anneal_qubits ∈ {4, 6, 8, 12}
      对应 n_bits_per_weight = num_qubits // 4 ∈ {1, 1, 2, 3}
      （1 bit = 仅符号位，无数值精度；2 bit = 1 符号 + 1 数值；3 bit = 1 符号 + 2 数值）
    - 每种精度 5 个独立 seed
    - 与无退火基线对比
    - 记录：最终 reward、退火介入率、p 值、Cohen's d

输出：
    - results/annealing_precision_comparison_<timestamp>.json（原始数据）
    - results/reports/annealing_precision_comparison.md（分析报告）

用法：
    # 完整实验（约 15-20 分钟）
    python scripts/evaluation/annealing_precision_comparison.py

    # 快速验证（少量步数，用于 CI 烟测）
    python scripts/evaluation/annealing_precision_comparison.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("QUANTUM_ACCELERATION_ENABLED", "1")

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

for p in [_PROJECT_ROOT, _SCRIPT_DIR]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import numpy as np

from src.quantum.annealing import QuantumAnnealingOptimizer
from src.scheduler.agent import PPOAgent
from src.scheduler.env import QuantumSchedulingEnv
from src.utils.stats_significance import compare_strategies

# ============================================================
# 实验配置
# ============================================================
SEEDS: list[int] = [42, 123, 456, 789, 1024]
ANNEAL_QUBITS_LIST: list[int] = [4, 6, 8, 12]
ANNEAL_INTERVAL = 5000

# 默认配置（完整实验）
TOTAL_TIMESTEPS = 20000
EVAL_FREQ = 5000
N_EVAL_EPISODES = 3
MAX_STEPS = 100

# 快速配置（CI 烟测）
QUICK_TIMESTEPS = 5000
QUICK_EVAL_FREQ = 2500
QUICK_SEEDS = [42, 123]

RESULTS_DIR = _PROJECT_ROOT / "results"
REPORTS_DIR = RESULTS_DIR / "reports"


# ============================================================
# 单次训练
# ============================================================
def train_one(
    seed: int,
    anneal_qubits: int | None,
    total_timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
    anneal_interval: int,
) -> dict[str, Any]:
    """训练单组 PPO，返回 eval rewards、退火介入次数、训练时间。

    Args:
        seed: 随机种子
        anneal_qubits: 退火量子比特数；None 表示无退火基线
        total_timesteps: 总训练步数
        eval_freq: 评估频率
        n_eval_episodes: 评估回合数
        anneal_interval: 退火触发间隔

    Returns:
        结果字典：timesteps / rewards / final_reward / train_time_s /
        anneal_interventions / n_bits_per_weight
    """
    use_annealing = anneal_qubits is not None
    label = f"anneal_q{anneal_qubits}" if use_annealing else "no_anneal"

    env = QuantumSchedulingEnv(max_steps=MAX_STEPS, seed=seed)

    kwargs: dict[str, Any] = {
        "verbose": 0,
        "seed": seed,
        "n_steps": 2048,
        "batch_size": 64,
        "log_dir": str(_PROJECT_ROOT / "logs" / f"precision_{label}_seed{seed}"),
    }
    if use_annealing:
        kwargs.update(
            {
                "use_annealing": True,
                "anneal_interval": anneal_interval,
                "anneal_qubits": anneal_qubits,
            }
        )

    agent = PPOAgent(env, **kwargs)

    # 记录退火介入次数（仅在 use_annealing 时有效）
    # AnnealingCallback 在 agent.train() 内部注册，训练后从 model 回调列表读取
    anneal_interventions = 0

    t0 = time.time()
    agent.train(
        total_timesteps=total_timesteps,
        eval_freq=eval_freq,
        n_eval_episodes=n_eval_episodes,
    )
    train_time = time.time() - t0

    # 读取退火介入次数（train 后 callback 才被注册到 model）
    if use_annealing:
        callbacks = []
        # PPOAgent 内部封装的 SB3 model，callbacks 通常在 model._callback
        sb3_model = getattr(agent, "model", None)
        if sb3_model is not None:
            cb = getattr(sb3_model, "_callback", None)
            if cb is not None:
                callbacks.append(cb)
        # 也检查 agent 自身存储的 callbacks
        for attr in ("_callbacks", "callbacks"):
            cbs = getattr(agent, attr, None)
            if cbs:
                callbacks.extend(cbs)
        for cb in callbacks:
            if cb.__class__.__name__ == "AnnealingCallback":
                anneal_interventions = getattr(cb, "optimized_count", 0)
                break

    # 读取 eval 结果
    eval_log = os.path.join(agent.log_dir, "eval_results", "evaluations.npz")
    try:
        data = np.load(eval_log)
        ts = data["timesteps"].tolist()
        rs = data["results"].tolist()
        if rs and isinstance(rs[0], (list, np.ndarray)):
            rs = [float(np.mean(r)) for r in rs]
        else:
            rs = [float(r) for r in rs]
        final_reward = rs[-1] if rs else 0.0
    except Exception as e:
        print(f"  [WARN] seed={seed} {label}: eval 读取失败 ({e})")
        ts, rs, final_reward = [], [], 0.0

    # 计算 n_bits_per_weight
    n_bits_per_weight = 0
    if use_annealing:
        n_bits_per_weight = max(1, anneal_qubits // 4)

    return {
        "label": label,
        "anneal_qubits": anneal_qubits,
        "n_bits_per_weight": n_bits_per_weight,
        "seed": seed,
        "timesteps": ts,
        "rewards": rs,
        "final_reward": final_reward,
        "train_time_s": round(train_time, 2),
        "anneal_interventions": anneal_interventions,
    }


# ============================================================
# 主流程
# ============================================================
def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    """运行完整实验，返回汇总结果。"""
    seeds = QUICK_SEEDS if args.quick else SEEDS
    total_timesteps = QUICK_TIMESTEPS if args.quick else TOTAL_TIMESTEPS
    eval_freq = QUICK_EVAL_FREQ if args.quick else EVAL_FREQ

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\n{'=' * 60}")
    print("退火编码精度对比实验 (Issue #240)")
    print(f"{'=' * 60}")
    print(f"Seeds: {seeds}")
    print(f"Anneal qubits: {ANNEAL_QUBITS_LIST}")
    print(f"Total timesteps: {total_timesteps}")
    print(f"Eval freq: {eval_freq}")
    print(f"Anneal interval: {ANNEAL_INTERVAL}")
    print(f"{'=' * 60}\n")

    all_results: dict[str, list[dict[str, Any]]] = {}

    # 1. 无退火基线
    print(f"--- [1/{len(ANNEAL_QUBITS_LIST) + 1}] 无退火基线 ---")
    all_results["no_anneal"] = []
    for seed in seeds:
        print(f"  Seed {seed} ...", end=" ", flush=True)
        r = train_one(
            seed=seed,
            anneal_qubits=None,
            total_timesteps=total_timesteps,
            eval_freq=eval_freq,
            n_eval_episodes=N_EVAL_EPISODES,
            anneal_interval=ANNEAL_INTERVAL,
        )
        all_results["no_anneal"].append(r)
        print(f"final={r['final_reward']:.1f}  time={r['train_time_s']:.0f}s")

    # 2. 四种精度
    for idx, qubits in enumerate(ANNEAL_QUBITS_LIST, start=2):
        label = f"anneal_q{qubits}"
        print(
            f"\n--- [{idx}/{len(ANNEAL_QUBITS_LIST) + 1}] anneal_qubits={qubits} "
            f"(n_bits_per_weight={max(1, qubits // 4)}) ---"
        )
        all_results[label] = []
        for seed in seeds:
            print(f"  Seed {seed} ...", end=" ", flush=True)
            r = train_one(
                seed=seed,
                anneal_qubits=qubits,
                total_timesteps=total_timesteps,
                eval_freq=eval_freq,
                n_eval_episodes=N_EVAL_EPISODES,
                anneal_interval=ANNEAL_INTERVAL,
            )
            all_results[label].append(r)
            print(
                f"final={r['final_reward']:.1f}  interventions={r['anneal_interventions']}  "
                f"time={r['train_time_s']:.0f}s"
            )

    # 3. 汇总统计
    print(f"\n{'=' * 60}")
    print("实验完成，汇总统计...")
    print(f"{'=' * 60}\n")

    summary = build_summary(all_results, seeds, total_timesteps, eval_freq, args.quick)

    # 4. 统计检验
    final_rewards: dict[str, list[float]] = {
        label: [r["final_reward"] for r in results] for label, results in all_results.items()
    }
    stats_results = compare_strategies(final_rewards, alpha=0.05)
    summary["statistical_tests"] = {
        key: {
            "test": val.get("test", ""),
            "statistic": val.get("statistic", float("nan")),
            "p_value": val.get("p_value", float("nan")),
            "significant": val.get("significant", False),
            "effect_size": val.get("effect_size", float("nan")),
            "effect_size_type": val.get("effect_size_type", ""),
            "mean_diff": val.get("mean_diff", float("nan")),
            "ci_lower": val.get("ci_lower", float("nan")),
            "ci_upper": val.get("ci_upper", float("nan")),
            "bonferroni_alpha": val.get("bonferroni_alpha", 0.05),
            "interpretation": val.get("interpretation", ""),
        }
        for key, val in stats_results.items()
    }

    # 5. 收集退火参数配置（Issue #247；若 get_annealing_config 未合并则降级）
    annealing_configs: dict[str, Any] = {}
    for qubits in ANNEAL_QUBITS_LIST:
        cfg_opt = QuantumAnnealingOptimizer(num_qubits=qubits, simulation_mode=True)
        if hasattr(cfg_opt, "get_annealing_config"):
            cfg = cfg_opt.get_annealing_config()
        else:
            # 降级：手动构建关键字段
            cfg = {
                "num_qubits": qubits,
                "annealing_time": cfg_opt.annealing_time,
                "shots": cfg_opt.shots,
                "simulation_mode": cfg_opt.simulation_mode,
                "solver_backend": "numpy_sa",
                "n_bits_per_weight": max(1, qubits // 4),
            }
        cfg["n_bits_per_weight"] = max(1, qubits // 4)
        annealing_configs[f"anneal_q{qubits}"] = cfg
    annealing_configs["experiment"] = "annealing_precision_comparison"
    annealing_configs["anneal_qubits_scanned"] = ANNEAL_QUBITS_LIST
    annealing_configs["seeds"] = seeds
    annealing_configs["total_timesteps"] = total_timesteps
    annealing_configs["eval_freq"] = eval_freq
    annealing_configs["n_eval_episodes"] = N_EVAL_EPISODES
    annealing_configs["anneal_interval"] = ANNEAL_INTERVAL
    summary["annealing_config"] = annealing_configs

    # 6. 保存 JSON
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    json_path = RESULTS_DIR / f"annealing_precision_comparison_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"[JSON] {json_path}")

    # 7. 生成 Markdown 报告
    md_path = REPORTS_DIR / "annealing_precision_comparison.md"
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    md_content = build_markdown_report(summary, timestamp)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[MD]   {md_path}")

    return summary


# ============================================================
# 汇总统计
# ============================================================
def build_summary(
    all_results: dict[str, list[dict[str, Any]]],
    seeds: list[int],
    total_timesteps: int,
    eval_freq: int,
    quick: bool,
) -> dict[str, Any]:
    """构建汇总统计字典。"""
    summary: dict[str, Any] = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "experiment": "annealing_precision_comparison",
        "issue": 240,
        "config": {
            "seeds": seeds,
            "n_seeds": len(seeds),
            "total_timesteps": total_timesteps,
            "eval_freq": eval_freq,
            "n_eval_episodes": N_EVAL_EPISODES,
            "anneal_interval": ANNEAL_INTERVAL,
            "anneal_qubits_list": ANNEAL_QUBITS_LIST,
            "quick_mode": quick,
        },
        "groups": {},
    }

    for label, results in all_results.items():
        final_rewards = [r["final_reward"] for r in results]
        interventions = [r["anneal_interventions"] for r in results]
        train_times = [r["train_time_s"] for r in results]

        n_bits_per_weight = 0
        anneal_qubits = None
        if results and results[0]["anneal_qubits"] is not None:
            anneal_qubits = results[0]["anneal_qubits"]
            n_bits_per_weight = results[0]["n_bits_per_weight"]

        # 退火介入率 = 有效介入次数 / 预期触发次数
        expected_triggers = total_timesteps // ANNEAL_INTERVAL if anneal_qubits is not None else 0
        intervention_rate = (
            float(np.mean(interventions)) / expected_triggers if expected_triggers > 0 else 0.0
        )

        summary["groups"][label] = {
            "label": label,
            "anneal_qubits": anneal_qubits,
            "n_bits_per_weight": n_bits_per_weight,
            "n_seeds": len(results),
            "final_rewards_per_seed": final_rewards,
            "final_reward_mean": float(np.mean(final_rewards)),
            "final_reward_std": float(np.std(final_rewards, ddof=1))
            if len(final_rewards) > 1
            else 0.0,
            "final_reward_min": float(np.min(final_rewards)),
            "final_reward_max": float(np.max(final_rewards)),
            "anneal_interventions_per_seed": interventions,
            "anneal_interventions_mean": float(np.mean(interventions)),
            "expected_triggers": expected_triggers,
            "intervention_rate": intervention_rate,
            "train_time_mean_s": float(np.mean(train_times)),
            "train_time_total_s": float(np.sum(train_times)),
        }

    return summary


# ============================================================
# Markdown 报告生成
# ============================================================
def build_markdown_report(summary: dict[str, Any], timestamp: str) -> str:
    """生成 Markdown 分析报告。"""
    config = summary["config"]
    groups = summary["groups"]
    stats_tests = summary.get("statistical_tests", {})

    lines: list[str] = []
    lines.append("# 退火编码精度对比实验报告（Issue #240）")
    lines.append("")
    lines.append(f"> **生成时间**: {timestamp}")
    lines.append("> **关联 Issue**: #240")
    lines.append("> **实验目的**: 验证退火效果不显著是 QUBO 编码精度问题还是方法论问题")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. 实验设计")
    lines.append("")
    lines.append("### 1.1 核心问题")
    lines.append("")
    lines.append(
        "当前默认 `anneal_qubits=16`（`n_bits_per_weight=4`，1 符号位 + 3 数值位），"
        "退火效果 +6.4%（p=0.19 不显著）。需验证：降低编码精度是否进一步削弱退火效果，"
        "从而证明「退火效果不显著源于编码精度不足而非方法论缺陷」。"
    )
    lines.append("")
    lines.append("### 1.2 实验配置")
    lines.append("")
    lines.append("| 参数 | 值 |")
    lines.append("|:--|:--|")
    lines.append(f"| Seeds | {config['seeds']} |")
    lines.append(f"| 每组种子数 | {config['n_seeds']} |")
    lines.append(f"| 总训练步数 | {config['total_timesteps']} |")
    lines.append(f"| 评估频率 | {config['eval_freq']} |")
    lines.append(f"| 评估回合数 | {config['n_eval_episodes']} |")
    lines.append(f"| 退火触发间隔 | {config['anneal_interval']} 步 |")
    lines.append(f"| 退火精度组 | {config['anneal_qubits_list']} |")
    lines.append(f"| 快速模式 | {config['quick_mode']} |")
    lines.append("")
    lines.append("### 1.3 编码精度映射")
    lines.append("")
    lines.append("`n_bits_per_weight = num_qubits // 4`（1 符号位 + 其余数值位）")
    lines.append("")
    lines.append("| anneal_qubits | n_bits_per_weight | 数值位 | 编码精度 |")
    lines.append("|:--:|:--:|:--:|:--|")
    for q in config["anneal_qubits_list"]:
        nb = max(1, q // 4)
        numeric_bits = nb - 1
        precision_desc = (
            "仅符号位（无数值精度）"
            if numeric_bits == 0
            else f"{numeric_bits} 位数值（2^{numeric_bits}={2**numeric_bits} 级）"
        )
        lines.append(f"| {q} | {nb} | {numeric_bits} | {precision_desc} |")
    lines.append("")
    lines.append(
        "> **注**: `anneal_qubits < 16` 时 `annealing.py:108-113` 会发出精度警告。"
        "本实验刻意测试低精度场景以验证其对退火效果的影响。"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. 实验结果")
    lines.append("")
    lines.append("### 2.1 最终 Reward 对比表")
    lines.append("")
    lines.append("| 组别 | anneal_qubits | n_bits/weight | 均值 | 标准差 | min | max |")
    lines.append("|:--|:--:|:--:|:--:|:--:|:--:|:--:|")
    for label, g in groups.items():
        q = g["anneal_qubits"] if g["anneal_qubits"] is not None else "—"
        nb = g["n_bits_per_weight"] if g["n_bits_per_weight"] else "—"
        lines.append(
            f"| {label} | {q} | {nb} | {g['final_reward_mean']:.2f} | "
            f"{g['final_reward_std']:.2f} | {g['final_reward_min']:.2f} | "
            f"{g['final_reward_max']:.2f} |"
        )
    lines.append("")
    lines.append("### 2.2 退火介入率")
    lines.append("")
    lines.append(
        "退火介入率 = `AnnealingCallback.optimized_count` / 预期触发次数"
        f"（每 {config['anneal_interval']} 步触发一次）"
    )
    lines.append("")
    lines.append("| 组别 | 预期触发 | 实际介入（均值） | 介入率 | 说明 |")
    lines.append("|:--|:--:|:--:|:--:|:--|")
    for label, g in groups.items():
        if g["anneal_qubits"] is None:
            lines.append(f"| {label} | — | — | — | 无退火基线 |")
        else:
            lines.append(
                f"| {label} | {g['expected_triggers']} | "
                f"{g['anneal_interventions_mean']:.1f} | "
                f"{g['intervention_rate']:.1%} | "
                f"介入率反映退火实际改善策略的次数占比 |"
            )
    lines.append("")
    lines.append(
        "> **介入率定义**: `AnnealingCallback` 在每次退火后评估网络质量，"
        "仅当质量优于历史最佳时才计入 `optimized_count`。因此介入率 < 100% 属正常现象，"
        "表示部分退火尝试未带来改善。"
    )
    lines.append("")
    lines.append("### 2.3 训练时间")
    lines.append("")
    lines.append("| 组别 | 平均训练时间 (s) | 总训练时间 (s) |")
    lines.append("|:--|:--:|:--:|")
    for label, g in groups.items():
        lines.append(f"| {label} | {g['train_time_mean_s']:.1f} | {g['train_time_total_s']:.1f} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 3. 统计显著性检验")
    lines.append("")
    lines.append("### 3.1 各精度组 vs 无退火基线")
    lines.append("")
    lines.append(
        "对每种精度组与无退火基线进行两两统计检验（自动选择 t/Welch/Mann-Whitney U），"
        "Bonferroni 校正多重比较。"
    )
    lines.append("")
    lines.append(
        "| 对比 | 检验方法 | 统计量 | p 值 | Bonferroni α | 显著? | Cohen's d | 效应等级 |"
    )
    lines.append("|:--|:--|:--:|:--:|:--:|:--:|:--:|:--|")
    for pair_key, val in stats_tests.items():
        if "no_anneal" not in pair_key:
            continue
        sig = "✅ 是" if val.get("significant") else "❌ 否"
        effect = val.get("effect_size", float("nan"))
        if isinstance(effect, (int, float)) and effect == effect:  # not NaN
            abs_effect = abs(effect)
            if abs_effect < 0.2:
                level = "可忽略"
            elif abs_effect < 0.5:
                level = "小效应"
            elif abs_effect < 0.8:
                level = "中效应"
            else:
                level = "大效应"
        else:
            level = "N/A"
        lines.append(
            f"| {pair_key} | {val.get('test', '')} | {val.get('statistic', float('nan')):.4f} | "
            f"{val.get('p_value', float('nan')):.4e} | {val.get('bonferroni_alpha', 0.05):.4f} | "
            f"{sig} | {effect:.3f} | {level} |"
        )
    lines.append("")
    lines.append("### 3.2 解读")
    lines.append("")
    # 自动生成解读
    sig_8bit = None
    for pair_key, val in stats_tests.items():
        if "anneal_q8 vs no_anneal" in pair_key:
            sig_8bit = val.get("significant", False)
            break

    if sig_8bit:
        lines.append(
            "- **8 比特组（n_bits_per_weight=2）达到统计显著**（p < Bonferroni α），"
            "说明提高编码精度能显著改善退火效果。建议生产环境使用 `anneal_qubits ≥ 8`。"
        )
    else:
        lines.append(
            "- **8 比特组仍未达到统计显著**，说明仅靠提升编码精度不足以让退火效果显著。"
            "退火不显著的根因可能是方法论层面（如 QUBO 构造、奖励信号强度、退火频率），"
            "而非单纯的编码精度问题。"
        )
    lines.append("- 详见各组 p 值与 Cohen's d，对比精度提升对效应量的影响趋势。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 4. 退火参数配置（Issue #247）")
    lines.append("")
    lines.append("为保障实验可复现性，以下列出本实验使用的完整退火参数配置。")
    lines.append("")
    lines.append("### 4.1 实验级配置")
    lines.append("")
    lines.append("| 参数 | 值 | 说明 |")
    lines.append("|:--|:--|:--|")
    lines.append("| `experiment` | annealing_precision_comparison | 实验名称 |")
    lines.append(f"| `anneal_qubits_scanned` | {config['anneal_qubits_list']} | 扫描的精度列表 |")
    lines.append(f"| `seeds` | {config['seeds']} | 独立随机种子 |")
    lines.append(f"| `total_timesteps` | {config['total_timesteps']} | 每组训练步数 |")
    lines.append(f"| `eval_freq` | {config['eval_freq']} | 评估频率 |")
    lines.append(f"| `n_eval_episodes` | {config['n_eval_episodes']} | 评估回合数 |")
    lines.append(f"| `anneal_interval` | {config['anneal_interval']} | 退火触发间隔 |")
    lines.append("")
    lines.append("### 4.2 各精度组配置")
    lines.append("")
    lines.append(
        "完整 `annealing_config` 字段已写入 JSON 输出（"
        "`results/annealing_precision_comparison_<timestamp>.json`），"
        "包含 `num_qubits`/`annealing_time`/`shots`/`simulation_mode`/SA 超参等 11 个字段。"
    )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 5. 结论与建议")
    lines.append("")
    lines.append("### 5.1 核心结论")
    lines.append("")
    lines.append(
        "1. **编码精度对退火效果的影响**: 通过对比 4 种精度（1/1/2/3 bits/weight）的"
        "最终 reward 与退火介入率，可观察精度提升是否带来单调改善。"
    )
    lines.append(
        "2. **退火不显著的根因**: 若 8/12 比特组仍不显著，则退火效果不显著"
        "并非编码精度问题，而需从方法论层面（QUBO 构造、奖励强度、退火频率）寻找根因。"
    )
    lines.append("3. **生产环境建议**: 根据显著性与效应量结果，给出 `anneal_qubits` 的推荐值。")
    lines.append("")
    lines.append("### 5.2 后续行动")
    lines.append("")
    lines.append(
        "- 若 8 比特组显著（p < 0.05）：更新 `ablation_report.md` 等报告中"
        "的退火显著性结论，将 `anneal_qubits=8` 作为新的默认推荐值。"
    )
    lines.append(
        "- 若所有组均不显著：保持现有结论（退火 +6.4%, p=0.19），"
        "在 `docs/annealing_significance-defense.md` 中补充"
        "「精度对比实验已排除编码精度问题」的论证。"
    )
    lines.append("- 扩展实验：可进一步测试 `anneal_qubits=16/24/32` 观察精度饱和效应。")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 6. 关联文档")
    lines.append("")
    lines.append("| 文档 | 关系 |")
    lines.append("|:--|:--|")
    lines.append("| `results/reports/ablation_report.md` | D5 退火消融实验（+6.4%, p=0.19） |")
    lines.append("| `docs/annealing_significance-defense.md` | 退火显著性答辩策略 |")
    lines.append("| `results/reports/annealing_lr_sweep_report.md` | 退火学习率扫描报告 |")
    lines.append("| `results/reports/hierarchical_annealing_report.md` | 分层退火对比报告 |")
    lines.append("| `src/quantum/annealing.py` | QuantumAnnealingOptimizer 实现 |")
    lines.append("| `src/scheduler/callbacks.py` | AnnealingCallback 实现 |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(
        f"*数据源: results/annealing_precision_comparison_{timestamp}.json "
        f"(quick_mode={config['quick_mode']})*"
    )

    return "\n".join(lines) + "\n"


# ============================================================
# CLI 入口
# ============================================================
def main() -> None:
    parser = argparse.ArgumentParser(description="退火编码精度对比实验（Issue #240）")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速模式：少量步数和种子，用于 CI 烟测",
    )
    args = parser.parse_args()

    run_experiment(args)


if __name__ == "__main__":
    main()
