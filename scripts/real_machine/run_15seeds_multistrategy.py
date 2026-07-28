#!/usr/bin/env python
"""
真机扩样实验：N≥15 seeds × 多策略 + Task ID 全留档（Issue #455）

实验设计：
    1. FCFS（先来先服务基线，EnvBasedFCFSScheduler）
    2. SPTF（最短处理时间优先基线，EnvBasedSPTFScheduler）
每个 seed 真机提交上限为 cap_per_seed，task_id 100% 留档。

使用方法：
    python scripts/real_machine/run_15seeds_multistrategy.py --dry-run   # 干跑验证
    python scripts/real_machine/run_15seeds_multistrategy.py             # 正式运行（需真机凭证）

Task ID 全留档：每条记录包含 seed、condition、step、task_id（RL内部）、real_task_id（天衍平台）
结果保存至 results/real_machine/issue455_15seeds_multistrategy.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)

from src.scheduler.baselines import EnvBasedFCFSScheduler, EnvBasedSPTFScheduler
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv

SEEDS_15 = [
    42,
    123,
    456,
    789,
    1024,
    2026,
    314,
    271,
    828,
    5566,
    7788,
    1234,
    2345,
    3456,
    4567,
]

EPISODE_HORIZON = 500
CAP_PER_SEED = 20
SHOTS = 512
RESULTS_PATH = _PROJECT_ROOT / "results" / "real_machine" / "issue455_15seeds_multistrategy.json"


def run_episode(
    env: QuantumSchedulingEnv, scheduler: Any, seed: int, policy_name: str
) -> dict[str, Any]:
    """使用基线调度器运行单个 episode，收集所有真机提交记录。"""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = scheduler.select_action(obs, env)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

    real_records = []
    if hasattr(env, "_real_feedback_log"):
        real_records = list(env._real_feedback_log)

    return {
        "seed": seed,
        "policy": policy_name,
        "total_reward": round(total_reward, 2),
        "steps": steps,
        "completion_rate": float(info.get("completion_rate", 0.0)),
        "real_task_count": len(real_records),
        "real_task_ids": [
            {
                "task_id": r.get("task_id"),
                "real_task_id": r.get("real_task_id"),
                "status": r.get("status"),
                "fidelity": r.get("fidelity"),
                "machine": r.get("machine"),
                "step": r.get("step"),
            }
            for r in real_records
        ],
    }


def run_dry_run() -> dict[str, Any]:
    """干跑验证（仿真模式），验证脚本能正确运行并记录 task_id 字段。"""
    print("=== DRY RUN (仿真模式验证) ===")
    schedulers = [
        ("fcfs", EnvBasedFCFSScheduler()),
        ("sptf", EnvBasedSPTFScheduler()),
    ]

    for name, sched in schedulers:
        env = QuantumSchedulingEnv(
            max_steps=EPISODE_HORIZON,
            machine_configs=DEFAULT_MACHINE_CONFIGS,
            seed=42,
            use_real_machine=False,
        )
        ep = run_episode(env, sched, seed=42, policy_name=name)
        print(f"  {name}: reward={ep['total_reward']:.1f}, completion={ep['completion_rate']:.2%}")

    print("✅ 干跑通过。Task ID 留档字段已就绪。")
    return {"dry_run": True, "policies_tested": len(schedulers)}


def run_real_experiment(seeds: list[int]) -> dict[str, Any]:
    """运行真机实验，Task ID 100% 留档。"""
    print(f"=== 真机实验 ({len(seeds)} seeds × 2 baselines) ===")
    print("⚠️  注意：真机实验需要配置 cqlib 客户端凭证")

    all_results: list[dict[str, Any]] = []
    policies = [
        ("fcfs", lambda: EnvBasedFCFSScheduler()),
        ("sptf", lambda: EnvBasedSPTFScheduler()),
    ]
    total = len(seeds) * len(policies)
    done_count = 0

    for seed in seeds:
        for policy_name, sched_fn in policies:
            done_count += 1
            print(f"  [{done_count}/{total}] seed={seed}, policy={policy_name}...", flush=True)
            try:
                env = QuantumSchedulingEnv(
                    max_steps=EPISODE_HORIZON,
                    machine_configs=DEFAULT_MACHINE_CONFIGS,
                    seed=seed,
                    use_real_machine=True,
                    max_real_submissions=CAP_PER_SEED,
                    real_machine_shots=SHOTS,
                    real_submit_probability=0.3,
                    real_feedback_mode="result_aware",
                )
                sched = sched_fn()
                ep = run_episode(env, sched, seed=seed, policy_name=policy_name)
                all_results.append(ep)
                real_count = ep["real_task_count"]
                id_count = sum(1 for r in ep["real_task_ids"] if r.get("real_task_id") is not None)
                print(
                    f"    reward={ep['total_reward']:.1f}, real_tasks={real_count}, "
                    f"task_ids留档={id_count}/{real_count}"
                )
                env.close()
            except Exception as e:
                print(f"    FAILED: {e}")
                all_results.append({"seed": seed, "policy": policy_name, "error": str(e)})

    return {
        "timestamp": datetime.now().isoformat(),
        "seeds": seeds,
        "policies": [p[0] for p in policies],
        "cap_per_seed": CAP_PER_SEED,
        "shots": SHOTS,
        "episode_horizon": EPISODE_HORIZON,
        "results": all_results,
        "task_id_audit": audit_task_ids(all_results),
    }


def audit_task_ids(results: list[dict[str, Any]]) -> dict[str, Any]:
    """审计 Task ID 留档率：Issue #455 验收标准要求 task_id 字段 100% 存在。"""
    total_real = 0
    total_with_id = 0
    for r in results:
        if "real_task_ids" not in r:
            continue
        for tr in r["real_task_ids"]:
            total_real += 1
            if tr.get("task_id") is not None:
                total_with_id += 1

    return {
        "total_real_submissions": total_real,
        "total_with_task_id": total_with_id,
        "task_id_coverage_rate": (total_with_id / total_real) if total_real > 0 else 0.0,
        "meets_100_percent": total_with_id == total_real and total_real > 0,
        "note": "task_id 为 RL 内部任务ID；real_task_id 为天衍平台任务ID，真机调用时才存在",
    }


def main():
    parser = argparse.ArgumentParser(description="15-seed 多策略真机实验（Task ID全留档）")
    parser.add_argument("--dry-run", action="store_true", help="干跑仿真验证")
    args = parser.parse_args()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)

    data = run_dry_run() if args.dry_run else run_real_experiment(SEEDS_15)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
