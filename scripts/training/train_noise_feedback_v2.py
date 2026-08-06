#!/usr/bin/env python
"""
噪声反馈 v2 — 真机噪声分布重训 PPO + 50×5 对照（Issue #456 路线甲）

实验设计：
    A. PPO-standard : 默认 uniform(0.85, 0.99) 保真度噪声
    B. PPO-noise    : 使用真机噪声分布 Beta(μ=0.886, σ=0.087) 截断 [0.671, 0.994]

对照指标：平均奖励、等待时间、任务完成率、保真度加权指标
统计方法：Mann-Whitney U 检验 + Cliff's delta 效应量（N=50 seeds × 5 episodes）

使用方法：
    python scripts/training/train_noise_feedback_v2.py                  # 完整 50 seeds
    python scripts/training/train_noise_feedback_v2.py --quick          # 快速验证 5 seeds
    python scripts/training/train_noise_feedback_v2.py --train-only     # 仅训练不评估
    python scripts/training/train_noise_feedback_v2.py --eval-only      # 仅评估（需已有模型）
"""

import argparse
import json
import multiprocessing as mp

# 8.5 N=50 并行：全局限制 torch 线程（每进程 2 线程 → 4 进程 8 线程 < 16 核），
# 避免 OpenMP 线程池多进程超载（默认 16 线程/进程 × 4 = 64 线程忙等死锁）。
import torch

torch.set_num_threads(2)
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import numpy as np

from src.scheduler.agent import PPOAgent
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv

REAL_NOISE_PROFILE = "real_machine"
SEEDS_FULL = list(range(42, 42 + 50))
SEEDS_QUICK = [42, 123, 456, 789, 1024]
TRAIN_TIMESTEPS = 500000  # 可由 --timesteps 覆盖（8.5 快速验证用 150K）
EVAL_EPISODES = 5
MAX_STEPS = 500
MODEL_DIR = PROJECT_ROOT / "models" / "noise_feedback_v2"
RESULTS_DIR = PROJECT_ROOT / "results" / "noise_feedback_v2"
LOG_DIR = PROJECT_ROOT / "logs" / "noise_feedback_v2"


def _train_worker(args: tuple) -> tuple[int, str]:
    """多进程 worker：返回 (seed, model_path)。

    8.5 修复：限制 torch 线程数（8 进程 × 2 线程 = 16，匹配 16 核机器），
    避免每个进程默认开满 16 线程导致 128 线程争抢死锁/极慢。
    """
    import torch

    torch.set_num_threads(2)
    seed, noise_profile, label, timesteps = args
    path = train_model(seed, noise_profile, label, timesteps=timesteps)
    return seed, path


def train_model(
    seed: int, noise_profile: str | None, label: str, timesteps: int = TRAIN_TIMESTEPS
) -> str:
    """训练单个 PPO 模型，返回模型路径。"""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    log_subdir = LOG_DIR / label / f"seed{seed}"
    model_path = MODEL_DIR / f"ppo_{label}_seed{seed}"

    if model_path.with_suffix(".zip").exists():
        print(f"  [SKIP] 模型已存在: {model_path}")
        return str(model_path)

    env = QuantumSchedulingEnv(
        max_steps=MAX_STEPS,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=seed,
        noise_profile=noise_profile,
    )

    agent = PPOAgent(
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=0,
        seed=seed,
        log_dir=str(log_subdir),
    )

    t0 = time.time()
    agent.train(
        total_timesteps=timesteps,
        eval_freq=TRAIN_TIMESTEPS,
        n_eval_episodes=EVAL_EPISODES,
        log_dir=str(log_subdir),
    )
    elapsed = time.time() - t0

    agent.save(str(model_path))
    print(
        f"  [TRAINED] {label} seed={seed}, reward={agent.evaluate(5)['mean_reward']:.1f}, time={elapsed:.0f}s"
    )
    return str(model_path)


def evaluate_model(model_path: str, seed: int, noise_profile: str | None) -> dict:
    """加载模型并评估，返回评估指标。"""
    env = QuantumSchedulingEnv(
        max_steps=MAX_STEPS,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=seed + 10000,
        noise_profile=noise_profile,
    )

    agent = PPOAgent(env, verbose=0, seed=seed)
    agent.load(model_path)

    results = agent.evaluate(num_episodes=EVAL_EPISODES, deterministic=True)
    return {
        "seed": seed,
        "mean_reward": float(results["mean_reward"]),
        "std_reward": float(results["std_reward"]),
        "success_rate": float(results.get("success_rate", 0.0)),
        "completion_rate": float(results.get("completion_rate", 0.0)),
    }


def run_full_experiment(
    seeds: list[int],
    train_only: bool = False,
    eval_only: bool = False,
    timesteps: int = TRAIN_TIMESTEPS,
    parallel: int = 1,
) -> dict:
    """运行完整实验：训练 + 评估 + 统计检验。"""
    conditions = [
        ("standard", None, "PPO-standard (uniform noise)"),
        ("noise", REAL_NOISE_PROFILE, "PPO-noise (real machine distribution)"),
    ]

    all_train_results = {}
    all_eval_results = {}

    for label, noise_profile, desc in conditions:
        print(f"\n{'=' * 60}")
        print(f"训练条件: {desc}")
        print(f"{'=' * 60}")

        model_paths = {}
        if not eval_only:
            if parallel > 1:
                # 8.5 N=50 实验：多进程并行训练（16 核机器 workers=8 → ~1.5h 完成 100 模型）
                tasks = [(seed, noise_profile, label, timesteps) for seed in seeds]
                with mp.Pool(parallel) as pool:
                    results = pool.map(_train_worker, tasks)
                for seed, path in results:
                    model_paths[seed] = path
                    print(f"  [done] {label} seed={seed} -> {Path(path).name}", flush=True)
            else:
                for i, seed in enumerate(seeds):
                    print(f"  [{i + 1}/{len(seeds)}] Training {label} seed={seed}...", flush=True)
                    _mp_ = train_model(seed, noise_profile, label, timesteps=timesteps)
                    model_paths[seed] = _mp_

        if train_only:
            continue

        print(f"\n评估 {label} ({len(seeds)} seeds × {EVAL_EPISODES} episodes)...")
        eval_results = []
        for i, seed in enumerate(seeds):
            _mp_ = model_paths.get(seed) or str(MODEL_DIR / f"ppo_{label}_seed{seed}")
            if not Path(_mp_).with_suffix(".zip").exists():
                print(f"  [SKIP] 模型不存在: {_mp_}")
                continue
            print(f"  [{i + 1}/{len(seeds)}] Evaluating {label} seed={seed}...", flush=True)
            result = evaluate_model(_mp_, seed, noise_profile)
            eval_results.append(result)
            print(
                f"    reward={result['mean_reward']:.1f}±{result['std_reward']:.1f}, "
                f"success={result['success_rate']:.2%}, completion={result['completion_rate']:.2%}"
            )

        all_eval_results[label] = eval_results
        all_train_results[label] = {
            "n_seeds": len(eval_results),
            "noise_profile": str(noise_profile),
        }

    if train_only:
        return {"training_completed": True, "seeds": seeds}

    stats = compute_statistics(all_eval_results)
    return {
        "timestamp": datetime.now().isoformat(),
        "train_timesteps": timesteps,  # 8.5 修复：报告实际训练步数（此前写死模块常量）
        "eval_episodes": EVAL_EPISODES,
        "seeds": seeds,
        "real_noise_profile": {
            "distribution": "beta",
            "mean": 0.8863,
            "std": 0.0874,
            "low": 0.671,
            "high": 0.994,
            "source": "10-seed real machine measurements (MBS fidelity)",
        },
        "results": dict(all_eval_results),
        "statistics": stats,
    }


def compute_statistics(all_results: dict) -> dict:
    """计算统计检验结果。"""
    from scipy import stats

    std_rewards = [
        r["mean_reward"] for r in all_results.get("standard", []) if r["mean_reward"] is not None
    ]
    noise_rewards = [
        r["mean_reward"] for r in all_results.get("noise", []) if r["mean_reward"] is not None
    ]

    results = {}

    if len(std_rewards) >= 2 and len(noise_rewards) >= 2:
        u_stat, p_value = stats.mannwhitneyu(noise_rewards, std_rewards, alternative="two-sided")
        n1, n2 = len(noise_rewards), len(std_rewards)
        gt = sum(1 for x in noise_rewards for y in std_rewards if x > y)
        lt = sum(1 for x in noise_rewards for y in std_rewards if x < y)
        cliffs_d = (gt - lt) / (n1 * n2)

        effect = "negligible"
        if abs(cliffs_d) >= 0.474:
            effect = "large"
        elif abs(cliffs_d) >= 0.33:
            effect = "medium"
        elif abs(cliffs_d) >= 0.147:
            effect = "small"

        results["reward_comparison"] = {
            "standard_mean": float(np.mean(std_rewards)),
            "standard_std": float(np.std(std_rewards)),
            "noise_mean": float(np.mean(noise_rewards)),
            "noise_std": float(np.std(noise_rewards)),
            "mean_diff": float(np.mean(noise_rewards) - np.mean(std_rewards)),
            "mann_whitney_u": float(u_stat),
            "p_value": float(p_value),
            "cliffs_delta": float(cliffs_d),
            "effect_size": effect,
            "significant_005": bool(p_value < 0.05),
            "n_standard": n1,
            "n_noise": n2,
        }

    for metric in ["success_rate", "completion_rate"]:
        std_vals = [r[metric] for r in all_results.get("standard", []) if r.get(metric) is not None]
        noise_vals = [r[metric] for r in all_results.get("noise", []) if r.get(metric) is not None]
        if len(std_vals) >= 2 and len(noise_vals) >= 2:
            _, p = stats.mannwhitneyu(noise_vals, std_vals, alternative="two-sided")
            results[f"{metric}_comparison"] = {
                "standard_mean": float(np.mean(std_vals)),
                "noise_mean": float(np.mean(noise_vals)),
                "p_value": float(p),
                "significant_005": bool(p < 0.05),
            }

    return results


def generate_report(data: dict) -> str:
    """生成 Markdown 报告。"""
    lines = [
        "# 噪声反馈 v2 实验报告（Issue #456）",
        "",
        f"实验时间: {data['timestamp']}",
        f"训练步数: {data['train_timesteps']:,}",
        f"评估配置: {len(data['seeds'])} seeds × {data['eval_episodes']} episodes",
        "",
        "## 实验设计",
        "",
        "| 条件 | 保真度噪声模型 | 说明 |",
        "|------|--------------|------|",
        "| PPO-standard | Uniform(0.85, 0.99) | 默认仿真噪声 |",
        "| PPO-noise | Beta(μ=0.886, σ=0.087) ∈ [0.671, 0.994] | 真机10-seed测量分布 |",
        "",
        "真机噪声数据来源：10-seed 真机闭环实验 MBS 保真度测量",
        f"- 均值: {data['real_noise_profile']['mean']:.4f}",
        f"- 标准差: {data['real_noise_profile']['std']:.4f}",
        f"- 范围: [{data['real_noise_profile']['low']:.3f}, {data['real_noise_profile']['high']:.3f}]",
        "",
        "## 结果",
        "",
    ]

    rc = data.get("statistics", {}).get("reward_comparison", {})
    if rc:
        lines.extend(
            [
                "### 奖励对比",
                "",
                "| 指标 | PPO-standard | PPO-noise | 差值 |",
                "|------|-------------|-----------|------|",
                f"| Mean Reward | {rc['standard_mean']:.1f} ± {data['statistics'].get('reward_comparison', {}).get('standard_std', 0):.1f} | "
                f"{rc['noise_mean']:.1f} ± {data['statistics'].get('reward_comparison', {}).get('noise_std', 0):.1f} | "
                f"{rc['mean_diff']:+.1f} |",
                "",
                f"- Mann-Whitney U: {rc['mann_whitney_u']:.1f}, p={rc['p_value']:.4f}",
                f"- Cliff's δ: {rc['cliffs_delta']:.3f} ({rc['effect_size']})",
                f"- 统计显著(p<0.05): {'是 ✅' if rc['significant_005'] else '否 ❌'}",
                "",
            ]
        )

    for metric_key, metric_name in [
        ("success_rate_comparison", "成功率"),
        ("completion_rate_comparison", "完成率"),
    ]:
        mc = data.get("statistics", {}).get(metric_key, {})
        if mc:
            lines.extend(
                [
                    f"### {metric_name}",
                    "",
                    f"- PPO-standard: {mc['standard_mean']:.2%}",
                    f"- PPO-noise: {mc['noise_mean']:.2%}",
                    f"- p={mc['p_value']:.4f}, 显著: {'是' if mc['significant_005'] else '否'}",
                    "",
                ]
            )

    lines.extend(
        [
            "## 诚信声明",
            "",
            "无论结果正负均如实报告：若 PPO-noise 显著优于 PPO-standard，说明真机噪声分布",
            "参数化提升了仿真保真度和策略鲁棒性；若无显著差异或 PPO-noise 更差，则诚实",
            '声明当前噪声模型对训练无显著增益，"量子赋能AI"叙事降级为平台接入验证。',
            "",
            "原始数据见 `results/noise_feedback_v2/noise_feedback_v2_results.json`",
        ]
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="噪声反馈 v2 训练与对照实验")
    parser.add_argument("--quick", action="store_true", help="快速模式：仅 5 seeds")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=TRAIN_TIMESTEPS,
        help="每模型训练步数（8.5 快速验证建议 150000）",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="并行训练进程数（16 核机器建议 8，N=50 完整实验）",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="起始 seed（含）——支持多进程分片：起 N 个独立进程各跑一段",
    )
    parser.add_argument(
        "--seed-end",
        type=int,
        default=91,
        help="结束 seed（含）",
    )
    parser.add_argument("--train-only", action="store_true", help="仅训练模型，不做评估")
    parser.add_argument("--eval-only", action="store_true", help="仅加载已有模型评估")
    args = parser.parse_args()

    if args.quick:
        seeds = SEEDS_QUICK
    elif args.seed_start != 42 or args.seed_end != 91:
        seeds = list(range(args.seed_start, args.seed_end + 1))
    else:
        seeds = SEEDS_FULL
    print(f"{'=' * 60}")
    print("噪声反馈 v2 实验（Issue #456）")
    print(f"Seeds: {len(seeds)}, Train steps: {TRAIN_TIMESTEPS:,}, Eval episodes: {EVAL_EPISODES}")
    print(f"{'=' * 60}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    data = run_full_experiment(
        seeds,
        train_only=args.train_only,
        eval_only=args.eval_only,
        timesteps=args.timesteps,
        parallel=args.parallel,
    )

    json_path = RESULTS_DIR / "noise_feedback_v2_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {json_path}")

    if not args.train_only:
        report = generate_report(data)
        report_path = RESULTS_DIR / "noise_feedback_v2_report.md"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"报告已保存: {report_path}")


if __name__ == "__main__":
    main()
