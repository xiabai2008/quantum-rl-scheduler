#!/usr/bin/env python
"""噪声鲁棒性交叉评估（预注册式，2×2 + MBS 跨类型）。

对应 docs/defense_narrative 的"正向量子→AI"增量实验方案。
复用 models/noise_feedback_v2 下 48 对 (standard, noise) 模型，零重训。

格子:
  SS = standard 模型 × standard 环境
  SN = standard 模型 × 真机噪声环境 (noise_profile="real_machine")
  NS = noise 模型 × standard 环境
  NN = noise 模型 × 真机噪声环境
  SM = standard 模型 × MBS 乘子注入
  NM = noise 模型 × MBS 乘子注入

假设:
  H1 : median(NN - SN) > 0        （带噪部署环境：noise 模型优于 standard）
  H1': median(NM - SM) > 0        （跨噪声类型鲁棒）
  H2 : NS 不显著劣于 SS           （非劣效）
  H3 : G_standard > G_noise       （鲁棒性缺口缩小）

用法:
    python scripts/evaluation/noise_robustness_cross_eval.py --episodes 5 --canonical
    python scripts/evaluation/noise_robustness_cross_eval.py --episodes 5 --seeds-start 42 --seeds-end 60
"""
from __future__ import annotations

import argparse
import copy
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

import numpy as np
import yaml
from scipy import stats

from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv
from src.scheduler.ppo_agent import PPOAgent

sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "evaluation"))
from quantum_noise_paired_20seeds import (  # noqa: E402
    MBS_VALUES_10SEEDS,
    _make_noisy_step_factory,
)

MODEL_DIR = _PROJECT_ROOT / "models" / "noise_feedback_v2"
RESULTS_DIR = _PROJECT_ROOT / "results" / "quantum_ai"
REPORT_DIR = _PROJECT_ROOT / "results" / "reports"

def common_seeds() -> list[int]:
    """扫描两前缀 zip 文件，返回公共 seed 列表（固定，不后补）。"""
    std = set()
    nse = set()
    for f in MODEL_DIR.glob("ppo_standard_seed*.zip"):
        std.add(int(f.stem.replace("ppo_standard_seed", "")))
    for f in MODEL_DIR.glob("ppo_noise_seed*.zip"):
        nse.add(int(f.stem.replace("ppo_noise_seed", "")))
    common = sorted(std & nse)
    print(f"公共 seeds: {len(common)} 个")
    return common


def _eval_cell(
    model_path: Path,
    env_noise_profile: str | None,
    noise_mode: str,
    seed: int,
    episodes: int,
    max_steps: int = 500,
) -> list[float]:
    """评估单个 (seed, cell)，返回 K 个 episode 奖励。

    与 train_noise_feedback_v2.py evaluate_model 完全同路径：
    env seed=seed+10000 + DEFAULT_MACHINE_CONFIGS + PPOAgent.evaluate(deterministic=True)。
    仅 MBS 注入格（SM/NM）手动 rollout（PPOAgent._create_eval_env 深拷贝配置，
    会丢失 env.step 包装，故不能走 PPOAgent.evaluate）。
    """
    rng = np.random.default_rng(seed)
    env = QuantumSchedulingEnv(
        max_steps=max_steps,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=seed + 10000,
        noise_profile=env_noise_profile,
    )
    if noise_mode == "mbs":
        orig_step = env.step
        env.step = _make_noisy_step_factory(orig_step, MBS_VALUES_10SEEDS, rng)

    agent = PPOAgent(env, verbose=0, seed=seed)
    agent.load(str(model_path))

    if noise_mode != "mbs":
        results = agent.evaluate(num_episodes=episodes, deterministic=True)
        return [float(results["mean_reward"])]

    ep_rewards: list[float] = []
    for _ep in range(episodes):
        obs, _info = env.reset()
        total_reward = 0.0
        step = 0
        done = False
        while step < max_steps and not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            step += 1
            done = terminated or truncated
        ep_rewards.append(float(total_reward))
    return ep_rewards


def compute_paired_stats(a: list[float], b: list[float]) -> dict:
    """配对统计：Wilcoxon + Cohen's d_z + 95% CI。"""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    diff = b_arr - a_arr  # b - a
    n = len(diff)
    result = {
        "p": 1.0,
        "d_z": 0.0,
        "ci_low": None,
        "ci_high": None,
        "median_diff": float(np.median(diff)) if n else 0.0,
        "mean_diff": float(diff.mean()) if n else 0.0,
        "n": n,
    }
    if n < 2 or np.all(diff == diff[0]) or np.isnan(diff).any():
        return result
    try:
        stat, p = stats.wilcoxon(diff, alternative="greater")
        result["p"] = float(p)
    except ValueError:
        result["p"] = 1.0
    ddof = diff.std(ddof=1)
    result["d_z"] = float(diff.mean() / ddof) if ddof > 0 else 0.0
    se = ddof / np.sqrt(n)
    result["ci_low"] = float(diff.mean() - 1.96 * se)
    result["ci_high"] = float(diff.mean() + 1.96 * se)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="噪声鲁棒性交叉评估")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seeds-start", type=int, default=None)
    parser.add_argument("--seeds-end", type=int, default=None)
    parser.add_argument("--canonical", action="store_true", help="对账 v2 权威值 (A≈4115, D≈4176)")
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()

    seeds = common_seeds()
    if args.seeds_start is not None:
        seeds = [s for s in seeds if s >= args.seeds_start and (args.seeds_end is None or s < args.seeds_end)]
    print(f"本次评估 seeds: {len(seeds)} 个 ({seeds[0]}..{seeds[-1]})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cells = {"SS": [], "SN": [], "NS": [], "NN": [], "SM": [], "NM": []}
    cell_models = {
        "SS": ("standard", None, "none"),
        "SN": ("standard", "real_machine", "profile"),
        "NS": ("noise", None, "none"),
        "NN": ("noise", "real_machine", "profile"),
        "SM": ("standard", None, "mbs"),
        "NM": ("noise", None, "mbs"),
    }

    start = time.time()
    for s in seeds:
        for cell, (mtype, profile, mode) in cell_models.items():
            model_path = MODEL_DIR / f"ppo_{mtype}_seed{s}.zip"
            rewards = _eval_cell(model_path, profile, mode, s, args.episodes, args.max_steps)
            cells[cell].append(np.mean(rewards))
        print(f"  seed={s}: SS={cells['SS'][-1]:.0f} SN={cells['SN'][-1]:.0f} "
              f"NN={cells['NN'][-1]:.0f} NS={cells['NS'][-1]:.0f}", flush=True)

    elapsed = time.time() - start
    print(f"评估完成，耗时 {elapsed:.0f}s")

    # 统计
    h1 = compute_paired_stats(cells["SN"], cells["NN"])      # NN - SN > 0
    h1p = compute_paired_stats(cells["SM"], cells["NM"])     # NM - SM > 0
    h2 = compute_paired_stats(cells["NS"], cells["SS"])      # NS - SS (双向)
    g_std = compute_paired_stats(cells["SS"], cells["SN"])   # 标准模型噪声损失
    g_nse = compute_paired_stats(cells["NS"], cells["NN"])   # 噪声模型噪声损失

    # 对账（canonical）
    qa_note = ""
    if args.canonical:
        a_ok = abs(np.mean(cells["SS"]) - 4115.5) / 4115.5 < 0.05
        d_ok = abs(np.mean(cells["NN"]) - 4176.2) / 4176.2 < 0.05
        qa_note = f"QA: cell A(SS)={np.mean(cells['SS']):.0f}({'OK' if a_ok else 'FAIL'}), cell D(NN)={np.mean(cells['NN']):.0f}({'OK' if d_ok else 'FAIL'})"
        print(qa_note)
        if not (a_ok and d_ok):
            print("⚠️ 对账超差，建议检查评估环境一致性")

    summary = {
        "experiment": "noise_robustness_cross_eval",
        "timestamp": datetime.now().isoformat(),
        "config": {"episodes": args.episodes, "max_steps": args.max_steps, "n_seeds": len(seeds)},
        "cells": {k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)), "n": len(v)} for k, v in cells.items()},
        "hypotheses": {
            "H1_NN_vs_SN": h1,
            "H1prime_NM_vs_SM": h1p,
            "H2_NS_vs_SS": h2,
            "H3_gap_std_vs_noise": {"g_standard": g_std, "g_noise": g_nse},
        },
        "qa": qa_note,
        "elapsed_seconds": round(elapsed, 1),
    }
    out = REPORT_DIR / f"noise_robustness_cross_eval_{datetime.now().strftime('%Y%m%d')}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"结果已保存: {out}")
    print(f"\nH1 (NN-SN>0): p={h1['p']:.3f} d_z={h1['d_z']:.2f} mean_diff={h1['mean_diff']:.1f}")
    print(f"H1'(NM-SM>0): p={h1p['p']:.3f} d_z={h1p['d_z']:.2f}")
    print(f"H2 (NS-SS): p={h2['p']:.3f} d_z={h2['d_z']:.2f}")
    print(f"H3: gap_std={g_std['mean_diff']:.1f} vs gap_noise={g_nse['mean_diff']:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
