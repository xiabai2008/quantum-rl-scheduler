#!/usr/bin/env python
"""
D3 奖励函数训练消融实验（严格版）
Ablation D3: Reward Function Training Ablation (Strict Version)

与 ablation_d3_reward.py（推理消融）的区别：
    - 推理消融：用已训练好的 PPO 模型，改变 reward 常量后推理
      → 只能回答"L4 训练的 PPO 在不同奖励下评估性能"
    - 训练消融：每个奖励配置从头训练 PPO
      → 能回答"不同奖励训练出的 PPO 性能差异"（#223 要求）

四层奖励配置（从简到全）：
    - L1 基础兼容奖励：仅任务完成基础分（兼容基线）
    - L2 +执行收益：量子/经典资源利用效率奖励
    - L3 +等待惩罚：任务等待时间惩罚
    - L4 +利用率奖励：整体资源利用率奖励（当前完整版本）

实验配置（#223 要求）：
    - 算法：PPO
    - Seeds：10 seeds × 5 episodes = N=50
    - 训练步数：50k steps
    - 环境：10 维 Obs10Wrapper
    - 对比指标：最终reward、收敛速度、稳定性
    - 统计检验：Wilcoxon signed-rank（配对，α=0.05）

用法：
    # 完整运行（约 16-17h）
    python scripts/evaluation/ablation_d3_training.py --seeds 10 --episodes 5 --timesteps 50000

    # 快速验证（仅 1 seed × 1 episode × 1k steps，约 2min）
    python scripts/evaluation/ablation_d3_training.py --seeds 1 --episodes 1 --timesteps 1000 --quick

    # 自定义输出目录
    python scripts/evaluation/ablation_d3_training.py --output-dir results/ablation_d3_training_v2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(str(_PROJECT_ROOT))

# 复用现有基础设施
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "evaluation"))
from run_issue_38_67_experiments import (
    Obs10Wrapper,
    SimulationEnv,
    SimulationTaskGenerator,
)

from src.scheduler.env import QuantumSchedulingEnv

# ============================================================================
# 四层奖励配置（L1 → L4，逐步叠加）
# ============================================================================

# 默认奖励常量（来自 src/scheduler/env.py + env_reward.py）
DEFAULT_REWARDS = {
    "classical_reward": 5.0,
    "quantum_reward": 10.0,
    "hybrid_reward": 7.0,
    "success_bonus": 3.0,
    "mismatch_penalty": -2.0,
    "wait_penalty": -0.1,
    "low_util_penalty": -1.0,
}

# 四层配置（#223 要求的严格 L1-L4 设计）
ABLATION_LAYERS = {
    "L1_basic": {
        # L1 基础兼容奖励：仅任务完成基础分
        "classical_reward": 5.0,
        "quantum_reward": 5.0,
        "hybrid_reward": 5.0,
        "success_bonus": 0.0,
        "mismatch_penalty": 0.0,
        "wait_penalty": 0.0,
        "low_util_penalty": 0.0,
    },
    "L2_execution": {
        # L2 +执行收益：量子/经典资源利用效率奖励
        "classical_reward": 5.0,
        "quantum_reward": 10.0,
        "hybrid_reward": 7.0,
        "success_bonus": 3.0,
        "mismatch_penalty": -2.0,
        "wait_penalty": 0.0,
        "low_util_penalty": 0.0,
    },
    "L3_wait_penalty": {
        # L3 +等待惩罚：任务等待时间惩罚
        "classical_reward": 5.0,
        "quantum_reward": 10.0,
        "hybrid_reward": 7.0,
        "success_bonus": 3.0,
        "mismatch_penalty": -2.0,
        "wait_penalty": -0.1,
        "low_util_penalty": 0.0,
    },
    "L4_full": {
        # L4 +利用率奖励：整体资源利用率奖励（当前完整版本）
        **DEFAULT_REWARDS,
    },
}


# ============================================================================
# 环境奖励常量 Monkey-Patch
# ============================================================================


def patch_env_rewards(env: QuantumSchedulingEnv, rewards: dict[str, float]) -> None:
    """Monkey-patch 环境的奖励常量。

    与 ablation_d3_reward.py 一致，直接修改环境实例的奖励常量。
    必须在 PPO 训练前调用，确保训练过程中使用新的奖励信号。
    """
    env._classical_reward = rewards["classical_reward"]
    env._quantum_reward = rewards["quantum_reward"]
    env._hybrid_reward = rewards["hybrid_reward"]
    env._success_bonus = rewards["success_bonus"]
    env._mismatch_penalty = rewards["mismatch_penalty"]
    env._wait_penalty = rewards["wait_penalty"]
    env._low_util_penalty = rewards["low_util_penalty"]


def make_env_with_rewards(
    rewards: dict[str, float],
    tasks_per_episode: int = 200,
    seed: int | None = None,
    obs_dim: int = 10,
) -> Any:
    """创建带指定奖励配置的环境。

    Args:
        rewards: 奖励常量字典
        tasks_per_episode: 每 episode 最大步数
        seed: 随机种子
        obs_dim: 观测维度（10=公平对比, 14=原生）
    """
    base = QuantumSchedulingEnv(
        max_steps=tasks_per_episode,
        max_qubits=287,
        seed=seed,
    )
    patch_env_rewards(base, rewards)
    if obs_dim == 10:
        return Obs10Wrapper(base)
    return base


# ============================================================================
# PPO 训练（每配置每 seed 独立训练）
# ============================================================================


def train_ppo_with_rewards(
    rewards: dict[str, float],
    seed: int,
    timesteps: int,
    tasks_per_episode: int = 200,
    obs_dim: int = 10,
    save_path: Path | None = None,
) -> tuple[Any, dict[str, Any]]:
    """用指定奖励配置训练 PPO。

    Args:
        rewards: 奖励常量
        seed: 随机种子
        timesteps: 训练步数
        tasks_per_episode: 每 episode 最大步数
        obs_dim: 观测维度
        save_path: 模型保存路径

    Returns:
        (model, train_info) 元组
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    # 创建带奖励 patch 的环境
    env = make_env_with_rewards(
        rewards=rewards,
        tasks_per_episode=tasks_per_episode,
        seed=seed,
        obs_dim=obs_dim,
    )

    # 训练进度回调（记录每 1000 步的奖励）
    train_rewards: list[float] = []

    class RewardLogger(BaseCallback):
        def _on_step(self) -> bool:
            # SB3 默认每步调用一次
            return True

        def _on_rollout_end(self) -> None:
            # 每个 rollout 结束时记录平均奖励
            if self.model.ep_info_buffer:
                avg_r = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
                train_rewards.append(avg_r)

    # PPO 超参数（与项目现有训练一致）
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )

    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=RewardLogger())
    train_time = time.time() - start_time

    train_info = {
        "seed": seed,
        "timesteps": timesteps,
        "train_time_seconds": round(train_time, 2),
        "final_train_reward": train_rewards[-1] if train_rewards else 0.0,
        "train_reward_curve": train_rewards,
        "convergence_step": _detect_convergence(train_rewards),
    }

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(save_path))
        train_info["model_path"] = str(save_path)

    env.close()
    return model, train_info


def _detect_convergence(
    reward_curve: list[float], window: int = 10, threshold: float = 0.05
) -> int:
    """检测收敛步数（最近 window 个 reward 的变异系数 < threshold）。"""
    if len(reward_curve) < window * 2:
        return len(reward_curve) * 1000  # 粗略估计
    for i in range(window, len(reward_curve)):
        window_vals = reward_curve[i - window : i]
        mean_v = np.mean(window_vals)
        if mean_v == 0:
            continue
        cv = np.std(window_vals) / abs(mean_v)
        if cv < threshold:
            return i * 1000  # 每个点约 1000 步
    return len(reward_curve) * 1000


# ============================================================================
# 训练后评估
# ============================================================================


def evaluate_model(
    model: Any,
    rewards: dict[str, float],
    seed: int,
    episodes: int = 5,
    tasks_per_episode: int = 200,
    obs_dim: int = 10,
) -> dict[str, Any]:
    """评估训练好的模型。"""
    env = make_env_with_rewards(
        rewards=rewards,
        tasks_per_episode=tasks_per_episode,
        seed=seed + 10000,  # 评估用不同 seed 避免过拟合
        obs_dim=obs_dim,
    )
    sim_env = SimulationEnv(env=env, task_generator=SimulationTaskGenerator(seed=seed + 10000))

    ep_rewards = []
    for ep in range(episodes):
        obs, info = sim_env.reset(seed=seed + ep)
        ep_reward = 0.0
        step = 0
        while step < tasks_per_episode:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = sim_env.step(int(action))
            ep_reward += reward
            step += 1
            if terminated or truncated:
                break
        ep_rewards.append(float(ep_reward))
        sim_env.record_episode_stats(info)

    summary = sim_env.get_summary()
    summary["ep_rewards"] = ep_rewards
    summary["mean_reward"] = float(np.mean(ep_rewards))
    summary["std_reward"] = float(np.std(ep_rewards, ddof=1)) if len(ep_rewards) > 1 else 0.0
    env.close()
    return summary


# ============================================================================
# 主实验流程
# ============================================================================


def run_d3_training_ablation(
    seeds: int = 10,
    episodes: int = 5,
    timesteps: int = 50000,
    tasks_per_episode: int = 200,
    obs_dim: int = 10,
    output_dir: Path | None = None,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """运行 D3 训练消融主实验。

    配置：4 层 × N seeds × M episodes
    """
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "results" / "ablation_d3_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_list = [42 + i * 137 for i in range(seeds)]
    config_names = list(ABLATION_LAYERS.keys())

    print("=" * 72)
    print("  D3 奖励函数训练消融实验（严格版）")
    print("=" * 72)
    print(f"  配置数:        {len(config_names)} ({', '.join(config_names)})")
    print(f"  Seeds:         {seeds} ({seed_list})")
    print(f"  Episodes:      {episodes}")
    print(f"  训练步数:      {timesteps}")
    print(f"  任务规模:      {tasks_per_episode} 步/episode")
    print(f"  观测维度:      {obs_dim}")
    print(f"  总训练量:      {len(config_names) * seeds * timesteps} steps")
    print(
        f"  预估时间:      {len(config_names) * seeds * timesteps / 50000:.1f}h (约 50k steps/min)"
    )
    print("=" * 72)

    all_results: dict[str, dict] = {}
    start_time = time.time()

    for config_name in config_names:
        rewards = ABLATION_LAYERS[config_name]
        print(f"\n{'=' * 72}")
        print(f"  配置: {config_name}")
        print(f"  奖励常量: {rewards}")
        print(f"{'=' * 72}")

        config_data: dict[str, Any] = {
            "rewards": rewards,
            "seeds": {},
        }

        for seed_idx, seed in enumerate(seed_list):
            print(f"\n  --- {config_name} | Seed {seed_idx + 1}/{seeds} (seed={seed}) ---")
            seed_start = time.time()

            # 训练
            model_save_path = output_dir / "models" / f"{config_name}_seed{seed}"
            model, train_info = train_ppo_with_rewards(
                rewards=rewards,
                seed=seed,
                timesteps=timesteps,
                tasks_per_episode=tasks_per_episode,
                obs_dim=obs_dim,
                save_path=model_save_path,
            )

            # 评估
            eval_info = evaluate_model(
                model=model,
                rewards=rewards,
                seed=seed,
                episodes=episodes,
                tasks_per_episode=tasks_per_episode,
                obs_dim=obs_dim,
            )

            seed_elapsed = time.time() - seed_start
            config_data["seeds"][str(seed)] = {
                "train": train_info,
                "eval": eval_info,
                "elapsed_seconds": round(seed_elapsed, 2),
            }

            print(
                f"  完成 ({seed_elapsed:.1f}s) | "
                f"train_final={train_info['final_train_reward']:.1f} | "
                f"eval_mean={eval_info['mean_reward']:.1f}±{eval_info['std_reward']:.1f}"
            )

        # 配置汇总
        all_rewards = [config_data["seeds"][str(s)]["eval"]["mean_reward"] for s in seed_list]
        config_data["summary"] = {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards, ddof=1)) if len(all_rewards) > 1 else 0.0,
            "all_rewards": all_rewards,
        }
        all_results[config_name] = config_data
        print(
            f"\n  {config_name} 汇总: mean={config_data['summary']['mean_reward']:.2f}"
            f"±{config_data['summary']['std_reward']:.2f}"
        )

    total_elapsed = time.time() - start_time
    print(f"\n所有配置完成，总耗时 {total_elapsed:.1f}s ({total_elapsed / 3600:.2f}h)")

    # ========================================================================
    # 统计显著性检验（配对 Wilcoxon，L4 vs L1/L2/L3）
    # ========================================================================
    print("\n" + "=" * 72)
    print("  统计显著性检验（配对 Wilcoxon, α=0.05）")
    print("=" * 72)

    from scipy import stats as scipy_stats

    sig_results: dict[str, Any] = {}
    l4_rewards = all_results["L4_full"]["summary"]["all_rewards"]

    for config_name in ["L1_basic", "L2_execution", "L3_wait_penalty"]:
        cfg_rewards = all_results[config_name]["summary"]["all_rewards"]
        diffs = [a - b for a, b in zip(l4_rewards, cfg_rewards, strict=True)]

        # 配对 Wilcoxon（仅非零差值）
        non_zero_diffs = [d for d in diffs if d != 0]
        if len(non_zero_diffs) >= 5:
            wilcoxon_result = scipy_stats.wilcoxon(non_zero_diffs)
            p_value = float(wilcoxon_result.pvalue)
            statistic = float(wilcoxon_result.statistic)
        else:
            p_value = float("nan")
            statistic = float("nan")

        # Cohen's d_z（配对效应量）
        if len(diffs) >= 2:
            std_diff = float(np.std(diffs, ddof=1))
            dz = float(np.mean(diffs) / std_diff) if std_diff != 0 else 0.0
        else:
            dz = 0.0

        sig_results[f"L4_vs_{config_name}"] = {
            "test": "Wilcoxon signed-rank test",
            "statistic": statistic,
            "p_value": p_value,
            "significant": bool(p_value < alpha) if not math.isnan(p_value) else False,
            "effect_size_type": "Cohen's d_z",
            "effect_size": dz,
            "mean_diff": float(np.mean(diffs)),
            "n_pairs": len(diffs),
            "n_nonzero_diffs": len(non_zero_diffs),
        }
        sig_mark = "✅" if sig_results[f"L4_vs_{config_name}"]["significant"] else "❌"
        print(
            f"  {sig_mark} L4 vs {config_name}: p={p_value:.4f}, d_z={dz:.4f}, "
            f"Δ={np.mean(diffs):+.2f} (n={len(diffs)}, 非零={len(non_zero_diffs)})"
        )

    # ========================================================================
    # 保存结果
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_json = {
        "config": {
            "experiment": "D3 Reward Function Training Ablation",
            "seeds": seed_list,
            "episodes_per_seed": episodes,
            "timesteps": timesteps,
            "tasks_per_episode": tasks_per_episode,
            "obs_dim": obs_dim,
            "alpha": alpha,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "timestamp": timestamp,
        },
        "ablation_layers": ABLATION_LAYERS,
        "results": all_results,
        "significance": sig_results,
    }

    results_path = output_dir / f"d3_training_ablation_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2, default=str)
    canonical_path = output_dir / "d3_training_ablation.json"
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[保存] 结果: {results_path}")

    # 生成报告
    _generate_report(results_json, output_dir, timestamp)
    print(f"[保存] 报告: {output_dir / 'd3_training_ablation.md'}")

    return results_json


# ============================================================================
# 报告生成
# ============================================================================


def _generate_report(results: dict, output_dir: Path, timestamp: str) -> None:
    """生成 Markdown 报告。"""
    cfg = results["config"]
    all_results = results["results"]
    sig = results["significance"]

    lines = [
        "# D3 奖励函数训练消融实验报告（严格版）",
        "",
        f"> **生成时间**: {timestamp}",
        f"> **实验规模**: {len(cfg['seeds'])} seeds × {cfg['episodes_per_seed']} episodes = "
        f"{len(cfg['seeds']) * cfg['episodes_per_seed']} 次独立运行（每配置）",
        f"> **训练步数**: {cfg['timesteps']} steps",
        f"> **任务规模**: {cfg['tasks_per_episode']} 步/episode",
        f"> **观测维度**: {cfg['obs_dim']}",
        f"> **总耗时**: {cfg['total_elapsed_seconds'] / 3600:.2f}h",
        "> **方法**: 每配置每 seed 独立训练 PPO（MlpPolicy, lr=3e-4, n_steps=2048）",
        "",
        "---",
        "",
        "## 一、实验目的",
        "",
        "回答 #223 的核心问题：**不同奖励配置训练出的 PPO 性能差异**。",
        "与 #201 的推理消融不同，本实验通过重新训练回答『L4 奖励设计是否最优』。",
        "",
        "## 二、四层奖励配置",
        "",
        "| 配置 | classical | quantum | hybrid | success | mismatch | wait | low_util |",
        "|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|",
    ]

    for name, rewards in results["ablation_layers"].items():
        lines.append(
            f"| {name} | {rewards['classical_reward']} | {rewards['quantum_reward']} | "
            f"{rewards['hybrid_reward']} | {rewards['success_bonus']} | "
            f"{rewards['mismatch_penalty']} | {rewards['wait_penalty']} | "
            f"{rewards['low_util_penalty']} |"
        )

    lines.extend(["", "## 三、实验结果", "", "### 3.1 各配置平均奖励", ""])

    lines.append("| 配置 | 平均奖励 | 标准差 | 收敛步数（中位） | 训练时间（中位, min）|")
    lines.append("|:--|:--:|:--:|:--:|:--:|")
    for name, data in all_results.items():
        summary = data["summary"]
        convergence_steps = [s["train"]["convergence_step"] for s in data["seeds"].values()]
        train_times = [s["train"]["train_time_seconds"] / 60 for s in data["seeds"].values()]
        lines.append(
            f"| {name} | {summary['mean_reward']:.2f} | {summary['std_reward']:.2f} | "
            f"{np.median(convergence_steps):.0f} | {np.median(train_times):.1f} |"
        )

    # 统计显著性
    lines.extend(["", "### 3.2 统计显著性（L4 vs 其他，配对 Wilcoxon）", ""])
    lines.append("| 对比 | p 值 | 显著 | Cohen's d_z | 平均差值 | 非零对 |")
    lines.append("|:--|:--:|:--:|:--:|:--:|:--:|")
    for pair, info in sig.items():
        sig_mark = "✅" if info["significant"] else "❌"
        lines.append(
            f"| {pair} | {info['p_value']:.4f} | {sig_mark} | {info['effect_size']:.4f} | "
            f"{info['mean_diff']:+.2f} | {info['n_nonzero_diffs']}/{info['n_pairs']} |"
        )

    # 关键发现
    lines.extend(["", "## 四、关键发现", ""])

    l4_mean = all_results["L4_full"]["summary"]["mean_reward"]
    l1_mean = all_results["L1_basic"]["summary"]["mean_reward"]
    l2_mean = all_results["L2_execution"]["summary"]["mean_reward"]
    l3_mean = all_results["L3_wait_penalty"]["summary"]["mean_reward"]

    lines.append(
        f"1. **L4 完整奖励 vs L1 基础**: L4={l4_mean:.2f} vs L1={l1_mean:.2f}, "
        f"Δ={l4_mean - l1_mean:+.2f}"
    )
    lines.append(
        f"2. **执行收益贡献**: L2-L1={l2_mean - l1_mean:+.2f} (量子奖励 10 vs 经典 5 的激励效果)"
    )
    lines.append(f"3. **等待惩罚贡献**: L3-L2={l3_mean - l2_mean:+.2f} (等待惩罚 -0.1 的约束效果)")
    lines.append(
        f"4. **利用率奖励贡献**: L4-L3={l4_mean - l3_mean:+.2f} (低利用率惩罚 -1.0 的引导效果)"
    )

    # 与 #201 对比
    lines.extend(
        [
            "",
            "## 五、与 #201 推理消融的对比",
            "",
            "| 维度 | #201 推理消融 | #223 训练消融 |",
            "|:--|:--|:--|",
            f"| Seeds | 5 | {len(cfg['seeds'])} |",
            "| 方法 | 预训练模型推理 | 每配置从头训练 |",
            "| 能回答 | L4 训练的 PPO 在不同奖励下评估性能 | 不同奖励训练出的 PPO 性能差异 |",
            "| 等待时间变化 | 不变（策略固定） | 可能变化（策略不同） |",
            "",
            "**#223 的核心价值**：能验证『奖励设计 → 策略学习 → 性能』的完整因果链。",
            "",
            "---",
            "",
            f"*自动生成于 ablation_d3_training.py | 数据源: d3_training_ablation_{timestamp}.json*",
            "",
        ]
    )

    report_path = output_dir / "d3_training_ablation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================================
# CLI 入口
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D3 奖励函数训练消融实验（严格版，#223）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=10, help="随机种子数（默认 10，#223 要求）")
    parser.add_argument("--episodes", type=int, default=5, help="每 seed 评估 episode 数（默认 5）")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="训练步数（默认 50000，#223 要求；快速验证可用 1000）",
    )
    parser.add_argument(
        "--tasks-per-episode", type=int, default=200, help="每 episode 最大步数（默认 200）"
    )
    parser.add_argument(
        "--obs-dim", type=int, default=10, choices=[10, 14], help="观测维度（默认 10）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation_d3_training",
        help="输出目录",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平（默认 0.05）")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="快速验证模式（覆盖 seeds=1, episodes=1, timesteps=1000）",
    )
    args = parser.parse_args()

    if args.quick:
        args.seeds = 1
        args.episodes = 1
        args.timesteps = 1000
        print("[快速验证模式] seeds=1, episodes=1, timesteps=1000")

    run_d3_training_ablation(
        seeds=args.seeds,
        episodes=args.episodes,
        timesteps=args.timesteps,
        tasks_per_episode=args.tasks_per_episode,
        obs_dim=args.obs_dim,
        output_dir=Path(args.output_dir),
        alpha=args.alpha,
    )


if __name__ == "__main__":
    main()
