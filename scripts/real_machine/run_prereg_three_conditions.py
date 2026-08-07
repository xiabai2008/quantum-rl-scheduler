#!/usr/bin/env python
"""
预注册三条件真机闭环评估实验（Issue #221 执行版）

设计（遵循 results/reports/real_machine_preregistration.md）：
  - C1 status_only : 仅任务状态，固定奖励（基线）
  - C2 result_aware: 真机保真度感知奖励（实验组）
  - C3 shuffled    : 打乱测量结果（消融对照）
  - 每个条件 N≥18 seeds（预注册要求，80% 功效）
  - 同一 seed 列表跨条件使用（种子锁定，配对可比）
  - 主指标：Cohen's d + 95% CI（效应量优先，p 值辅助）

使用预训练 16 维交付模型评估（非训练），每 seed 跑 1 个 episode。

用法：
    python scripts/real_machine/run_prereg_three_conditions.py --dry-run        # 仿真验证
    python scripts/real_machine/run_prereg_three_conditions.py --seeds 42 123 456   # 指定 seeds
    python scripts/real_machine/run_prereg_three_conditions.py                  # 默认 20 seeds
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

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from src.scheduler.env import QuantumSchedulingEnv
from scripts.real_machine.run_real_performance_v3 import MACHINE_CONFIGS, RetryClient

SEEDS_DEFAULT = [42, 123, 456, 789, 1024, 2026, 314, 271, 828, 5566,
                 7788, 1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012]

EPISODE_HORIZON = 500
CAP_PER_SEED = 30
REAL_SUBMIT_PROB = 0.3
SHOTS = 32
TARGET_MACHINE = "tianyan176"
PPO_MODEL_PATH = _PROJECT_ROOT / "deliverable_models" / "ppo_best_model_16dim.zip"
OUTPUT_DIR = _PROJECT_ROOT / "results" / "real_machine" / "prereg_three_conditions"


def create_ppo_policy():
    from stable_baselines3 import PPO as SB3PPO
    from scripts.evaluation.run_simulation import PPOStrategy

    model = SB3PPO.load(str(PPO_MODEL_PATH))
    return PPOStrategy(model)


def run_one(seed: int, feedback_mode: str, dry_run: bool = False, client=None) -> dict:
    """单 seed 单条件评估 episode。"""
    env = QuantumSchedulingEnv(
        max_steps=EPISODE_HORIZON,
        machine_configs=MACHINE_CONFIGS,
        seed=seed,
        use_real_machine=not dry_run,
        real_submit_probability=REAL_SUBMIT_PROB,
        max_real_submissions=CAP_PER_SEED,
        real_machine_shots=SHOTS,
        real_feedback_mode=feedback_mode,
        real_machine_max_qubits=int(os.environ.get("FREE_TIER_MAX_QUBITS", "1")),
    )
    if client is not None:
        env.attach_real_clients({TARGET_MACHINE: client})

    policy = create_ppo_policy()
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    real_completed = 0
    start = time.time()
    done = False

    while not done:
        action = policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

    records = []
    if hasattr(env, "_real_feedback_log"):
        for r in env._real_feedback_log:
            records.append({
                "real_task_id": r.get("real_task_id"),
                "outcome": r.get("outcome"),
                "fidelity": r.get("fidelity"),
                "reward": r.get("reward"),
            })
            if r.get("outcome") == "completed":
                real_completed += 1

    env.close()
    return {
        "seed": seed,
        "mode": feedback_mode,
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "elapsed_seconds": round(time.time() - start, 2),
        "real_tasks_completed": real_completed,
        "real_records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="预注册三条件真机评估（Issue #221）")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    parser.add_argument("--dry-run", action="store_true", help="仿真干跑（不碰真机）")
    args = parser.parse_args()

    modes = ["status_only", "result_aware", "shuffled"]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.dry_run:
        api_key = os.environ.get("TIANYAN_API_KEY", "")
        if not api_key:
            print("错误: 未设置 TIANYAN_API_KEY")
            sys.exit(1)
        from src.api.tianyan_cqlib import CqlibTianyanClient

        raw_client = CqlibTianyanClient(login_key=api_key, machine_name=TARGET_MACHINE, auto_retry_machine=False)
        client = RetryClient(raw_client)
        print(f"客户端: {TARGET_MACHINE} | 模式: {modes} | seeds: {len(args.seeds)}")
    else:
        print(f"DRY RUN（仿真）| 模式: {modes} | seeds: {len(args.seeds)}")

    data: dict[str, list] = {m: [] for m in modes}
    details: dict[str, list] = {m: [] for m in modes}
    total = len(args.seeds) * len(modes)
    done_count = 0

    for seed in args.seeds:
        for mode in modes:
            done_count += 1
            print(f"[{done_count}/{total}] seed={seed} mode={mode} ...", flush=True)
            try:
                r = run_one(seed, mode, dry_run=args.dry_run, client=client)
                data[mode].append(r["total_reward"])
                details[mode].append(r)
                print(
                    f"    reward={r['total_reward']:.1f} steps={r['steps']} "
                    f"real_completed={r['real_tasks_completed']}"
                )
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                details[mode].append({"seed": seed, "mode": mode, "error": str(e)})

    payload = {
        "experiment": "prereg_three_conditions",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seeds": args.seeds,
            "modes": modes,
            "episode_horizon": EPISODE_HORIZON,
            "cap_per_seed": CAP_PER_SEED,
            "real_submit_probability": REAL_SUBMIT_PROB,
            "shots": SHOTS,
            "machine": TARGET_MACHINE,
            "model": PPO_MODEL_PATH.name,
            "dry_run": args.dry_run,
        },
        # 预注册分析脚本期望的格式：{"status_only": [..], "result_aware": [..], "shuffled": [..]}
        "status_only": data["status_only"],
        "result_aware": data["result_aware"],
        "shuffled": data["shuffled"],
        "details": details,
    }
    out_path = OUTPUT_DIR / f"prereg_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=str)

    print(f"\n结果已保存: {out_path}")
    for m in modes:
        vals = data[m]
        if vals:
            import numpy as np

            print(f"  {m}: N={len(vals)} mean={np.mean(vals):.2f}±{np.std(vals, ddof=1):.2f}")


if __name__ == "__main__":
    main()
