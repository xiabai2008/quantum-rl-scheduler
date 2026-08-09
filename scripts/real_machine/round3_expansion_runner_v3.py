#!/usr/bin/env python
"""round3-B: 176 严格窗口模式实验 v3（2026-08-09 修正版）。

v2 失败根因：env 用了默认 machine_configs（tianyan_s/…）且未 attach_real_clients，
导致真机提交链路从未激活（real=0）。本版对齐 v2 权威脚本（tianyan287_multiseed.py）：
- 自定义 machine_configs：name=tianyan176（与客户端一致）+ classic_cpu_1
- env.attach_real_clients({TARGET_MACHINE: client}) 激活真机链路
- 提交前实时探测窗口（176 状态在 running/calibration 间闪烁）
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
from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.ppo_agent import PPOAgent

TARGET_MACHINE = "tianyan176"
SHOTS = 32
EPISODE_HORIZON = 96
RESULTS_PATH = _PROJECT_ROOT / "results" / "real_machine" / "round3_expansion_20260809_v3.json"

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

MACHINE_CONFIGS = [
    {
        "name": TARGET_MACHINE,
        "machine_type": "quantum",
        "max_qubits": 176,
        "noise_level": 0.01,
        "queue_capacity": 10,
    },
    {
        "name": "classic_cpu_1",
        "machine_type": "classic",
        "max_qubits": 0,
        "noise_level": 0.0,
        "queue_capacity": 20,
    },
]


def machine_status(client: CqlibTianyanClient) -> str:
    try:
        for b in client.list_backends():
            name = b.get("name") or b.get("machine_name") or ""
            if name == TARGET_MACHINE:
                return str(b.get("status", "unknown"))
    except Exception:
        pass
    return "unknown"


def probe_submit(client: CqlibTianyanClient, tag: str) -> bool:
    try:
        tid = client.submit_quantum_task(
            qcis="H Q1\nM Q1", shots=SHOTS, task_name=f"rd3probe_{tag}"
        )
        if not tid:
            return False
        tr = client.wait_for_task(tid, timeout=150, poll_interval=5)
        return tr.status == "completed" and bool(tr.probability)
    except Exception:
        return False


def run_one_episode(env: QuantumSchedulingEnv, seed: int, policy: str) -> dict[str, Any]:
    if policy == "PPO":
        agent = PPOAgent(env, verbose=0, seed=seed)
        agent.load(str(MODEL_PATH))
        obs, _info = env.reset(seed=seed)
        total = 0.0
        step = 0
        done = False
        while step < EPISODE_HORIZON and not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            step += 1
            done = terminated or truncated
    else:
        sched = EnvBasedFCFSScheduler() if policy == "FCFS" else EnvBasedSPTFScheduler()
        obs, _info = env.reset(seed=seed)
        total = 0.0
        step = 0
        done = False
        while step < EPISODE_HORIZON and not done:
            action = sched.select_action(obs, env)
            obs, reward, terminated, truncated, _info = env.step(action)
            total += float(reward)
            step += 1
            done = terminated or truncated
    records = list(getattr(env, "_real_feedback_log", []) or [])
    return {
        "seed": seed,
        "strategy": policy,
        "total_reward": round(total, 2),
        "real_submitted": len(records),
        "real_completed": sum(1 for r in records if r.get("status") == "completed"),
        "real_records": [
            {"task_id": r.get("task_id"), "status": r.get("status"), "fidelity": r.get("fidelity")}
            for r in records
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="176 严格窗口模式 N=20 扩充实验 v3")
    parser.add_argument("--wait-hours", type=float, default=3.0)
    parser.add_argument("--max-episodes", type=int, default=60)
    args = parser.parse_args()

    api_key = os.environ.get("TIANYAN_API_KEY", "")
    if not api_key:
        print("错误: 未设置 TIANYAN_API_KEY")
        return 1

    client = CqlibTianyanClient(
        login_key=api_key, machine_name=TARGET_MACHINE, auto_retry_machine=False
    )

    deadline = time.time() + args.wait_hours * 3600
    results: list[dict[str, Any]] = []
    done = 0
    total = min(args.max_episodes, len(SEEDS_20) * 3)
    policies = ["PPO", "FCFS", "SJF"]
    pairs = [(s, p) for s in SEEDS_20 for p in policies]

    while done < total and time.time() < deadline:
        st = machine_status(client)
        if st != "running":
            print(f"[{datetime.now():%H:%M:%S}] 176 status={st}，60s 后重试...", flush=True)
            time.sleep(60)
            continue
        print(f"[{datetime.now():%H:%M:%S}] 176 running，探测真实提交...", flush=True)
        if not probe_submit(client, f"pre{done}"):
            print("  探测失败，30s 后重试", flush=True)
            time.sleep(30)
            continue
        print("  探测成功！执行正式 episode", flush=True)
        seed, policy = pairs[done]
        try:
            env = QuantumSchedulingEnv(
                max_steps=EPISODE_HORIZON,
                machine_configs=[dict(c) for c in MACHINE_CONFIGS],
                seed=seed,
                use_real_machine=True,
                max_real_submissions=1,
                real_machine_shots=SHOTS,
                real_submit_probability=1.0,
                real_feedback_mode="result_aware",
            )
            env.attach_real_clients({TARGET_MACHINE: client})
            ep = run_one_episode(env, seed, policy)
            results.append(ep)
            print(
                f"  [{done + 1}/{total}] seed={seed} {policy} reward={ep['total_reward']:.1f} "
                f"real={ep['real_submitted']} done={ep['real_completed']}",
                flush=True,
            )
            env.close()
            done += 1
            time.sleep(3)
        except Exception as e:
            print(f"  FAILED: {str(e)[:100]}", flush=True)
            results.append({"seed": seed, "strategy": policy, "error": str(e)})
            done += 1
            time.sleep(10)

    data = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "round3_expansion_20260809_v3",
        "machine": TARGET_MACHINE,
        "completed": done,
        "total": total,
        "results": results,
    }
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {RESULTS_PATH}（完成 {done}/{total}）", flush=True)
    return 0 if done == total else 2


if __name__ == "__main__":
    sys.exit(main())
