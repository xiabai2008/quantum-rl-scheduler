#!/usr/bin/env python
"""round3-B: 176 恢复窗口等待 + 多seed真机扩充实验（2026-08-09）。

背景：8.9 修复 cqlib probability 字符串解析后，tianyan176 实际成功率从"判定 0%"
修正为 50%+（低并发时）。176 平台状态波动（running <-> calibration），本脚本：
1. 每 60s 探测平台状态，等待 176 恢复 running
2. 恢复后立即执行 N=20 seeds × 3 策略（PPO/FCFS/SJF）低频真机实验
   （v2 权威同构：每 seed 1 次真机调用，96 步 episode，shots=32，H Q1/M Q1）
3. 结果写入 results/real_machine/round3_expansion_20260809.json

用法：
    python scripts/real_machine/round3_expansion_runner.py --wait-hours 3
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(_PROJECT_ROOT)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.baselines import EnvBasedFCFSScheduler, EnvBasedSPTFScheduler
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv
from src.scheduler.ppo_agent import PPOAgent

TARGET_MACHINE = "tianyan176"
SHOTS = 32
EPISODE_HORIZON = 96  # v2 权威同构
RESULTS_PATH = _PROJECT_ROOT / "results" / "real_machine" / "round3_expansion_20260809.json"

SEEDS_20 = [
    42,
    123,
    456,
    789,
    1024,
    2025,
    3141,
    5678,
    8765,
    9999,
    7,
    11,
    13,
    17,
    19,
    23,
    29,
    31,
    37,
    41,
]
MODEL_PATH = _PROJECT_ROOT / "deliverable_models" / "ppo_best_model_16dim.zip"


def is_available(client: CqlibTianyanClient) -> bool:
    try:
        for b in client.list_backends():
            name = b.get("name") or b.get("machine_name") or ""
            if name == TARGET_MACHINE:
                return b.get("status") == "running"
    except Exception:
        pass
    return False


def run_episode_ppo(env: QuantumSchedulingEnv, seed: int) -> dict[str, Any]:
    agent = PPOAgent(env, verbose=0, seed=seed)
    agent.load(str(MODEL_PATH))
    obs, _info = env.reset(seed=seed)
    total_reward = 0.0
    step = 0
    done = False
    while step < EPISODE_HORIZON and not done:
        action = agent.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        step += 1
        done = terminated or truncated
    records = list(getattr(env, "_real_feedback_log", []) or [])
    return {
        "seed": seed,
        "strategy": "PPO",
        "total_reward": round(total_reward, 2),
        "real_submitted": len(records),
        "real_completed": sum(1 for r in records if r.get("status") == "completed"),
        "real_records": [
            {"task_id": r.get("task_id"), "status": r.get("status"), "fidelity": r.get("fidelity")}
            for r in records
        ],
    }


def run_episode_baseline(env: QuantumSchedulingEnv, seed: int, policy: str) -> dict[str, Any]:
    sched = EnvBasedFCFSScheduler() if policy == "FCFS" else EnvBasedSPTFScheduler()
    obs, _info = env.reset(seed=seed)
    total_reward = 0.0
    step = 0
    done = False
    while step < EPISODE_HORIZON and not done:
        action = sched.select_action(obs, env)
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        step += 1
        done = terminated or truncated
    records = list(getattr(env, "_real_feedback_log", []) or [])
    return {
        "seed": seed,
        "strategy": policy,
        "total_reward": round(total_reward, 2),
        "real_submitted": len(records),
        "real_completed": sum(1 for r in records if r.get("status") == "completed"),
        "real_records": [
            {"task_id": r.get("task_id"), "status": r.get("status"), "fidelity": r.get("fidelity")}
            for r in records
        ],
    }


def run_experiment(client: CqlibTianyanClient) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    policies = ["PPO", "FCFS", "SJF"]
    total = len(SEEDS_20) * len(policies)
    done = 0
    for seed in SEEDS_20:
        for policy in policies:
            done += 1
            print(f"  [{done}/{total}] seed={seed} {policy}...", flush=True)
            try:
                env = QuantumSchedulingEnv(
                    max_steps=EPISODE_HORIZON,
                    machine_configs=DEFAULT_MACHINE_CONFIGS,
                    seed=seed,
                    use_real_machine=True,
                    max_real_submissions=1,
                    real_machine_shots=SHOTS,
                    real_submit_probability=1.0,
                    real_feedback_mode="result_aware",
                )
                if policy == "PPO":
                    ep = run_episode_ppo(env, seed)
                else:
                    ep = run_episode_baseline(env, seed, policy)
                results.append(ep)
                print(
                    f"    reward={ep['total_reward']:.1f} real={ep['real_submitted']} "
                    f"completed={ep['real_completed']}",
                    flush=True,
                )
                env.close()
            except Exception as e:
                print(f"    FAILED: {str(e)[:100]}", flush=True)
                results.append({"seed": seed, "strategy": policy, "error": str(e)})
            time.sleep(2)  # 低频提交：保持窗口稳定
    return {
        "timestamp": datetime.now().isoformat(),
        "experiment": "round3_expansion_20260809",
        "machine": TARGET_MACHINE,
        "seeds": SEEDS_20,
        "policies": policies,
        "episode_horizon": EPISODE_HORIZON,
        "shots": SHOTS,
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="176 恢复窗口等待 + N=20 多seed扩充实验")
    parser.add_argument("--wait-hours", type=float, default=3.0)
    args = parser.parse_args()

    api_key = os.environ.get("TIANYAN_API_KEY", "")
    if not api_key:
        print("错误: 未设置 TIANYAN_API_KEY")
        return 1

    client = CqlibTianyanClient(
        login_key=api_key, machine_name=TARGET_MACHINE, auto_retry_machine=False
    )

    deadline = time.time() + args.wait_hours * 3600
    while time.time() < deadline:
        if is_available(client):
            print(f"[{datetime.now():%H:%M:%S}] 176 恢复 running！开始实验", flush=True)
            data = run_experiment(client)
            RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            print(f"结果已保存: {RESULTS_PATH}", flush=True)
            return 0
        print(f"[{datetime.now():%H:%M:%S}] 176 仍不可用（校准/维护），60s 后重试...", flush=True)
        time.sleep(60)

    print(
        f"等待超时（{args.wait_hours}h），176 未恢复。放弃扩充，保持 v2 N=10 权威口径。", flush=True
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())
