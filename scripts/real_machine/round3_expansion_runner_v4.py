#!/usr/bin/env python
"""round3-B: 176 严格窗口模式实验 v4（2026-08-09，对齐 v2 权威提交模式）。

v3 失败根因：env 异步提交 + 96 步 episode 太短，真机任务结束时仍在 pending，
fidelity=null。v2 权威（tianyan287_multiseed.py）用同步等待：
_submit_and_poll_one_task = client.wait_for_task 到终态。

本版结构：
- episode 用 real_submit_probability=0.0（不自动提交，只跑仿真调度）
- episode 结束后同步提交 1 个 H Q1/M Q1 任务并 wait_for_task（≤150s）
- 每 seed×策略 = 1 真机调用（v2 同构），记录 MBS/保真度
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
RESULTS_PATH = _PROJECT_ROOT / "results" / "real_machine" / "round3_expansion_20260809_v4.json"
QCIS_CIRCUIT = "H Q1\nM Q1"

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

# 对齐 v2 权威 MBS 公式：1 - 2*|P(0) - 0.5|
MAX_REAL_TASKS_PER_RUN = 1


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
            qcis=QCIS_CIRCUIT, shots=SHOTS, task_name=f"rd3probe_{tag}"
        )
        if not tid:
            return False
        tr = client.wait_for_task(tid, timeout=150, poll_interval=5)
        return tr.status == "completed" and bool(tr.probability)
    except Exception:
        return False


def submit_and_poll_one_task(client: CqlibTianyanClient, tag: str) -> dict[str, Any]:
    """同步提交 1 个真机任务并等待终态（v2 权威同构）。"""
    record: dict[str, Any] = {
        "task_id": None,
        "status": "failed",
        "probability": None,
        "fidelity": None,
        "measurement_balance_score": None,
        "error": None,
    }
    try:
        tid = client.submit_quantum_task(qcis=QCIS_CIRCUIT, shots=SHOTS, task_name=tag)
        if not tid:
            record["error"] = "提交返回 None（机器不可用）"
            return record
        record["task_id"] = str(tid)
        tr = client.wait_for_task(tid, timeout=150, poll_interval=5)
        prob = getattr(tr, "probability", None) or {}
        if tr.status == "completed" and prob:
            p0 = float(prob.get("0", 0.0))
            p1 = float(prob.get("1", 0.0))
            record["status"] = "completed"
            record["probability"] = prob
            record["fidelity"] = round(1 - abs(p0 - p1), 6)
            record["measurement_balance_score"] = round(1 - 2 * abs(p0 - 0.5), 6)
        elif tr.status == "completed":
            record["status"] = "completed_no_probability"
            record["error"] = "completed 但无测量结果"
        elif tr.status == "error":
            record["status"] = "failed"
            record["error"] = "error 终态"
        else:
            record["status"] = str(tr.status)
            record["error"] = f"未到终态: {tr.status}"
    except Exception as e:
        record["status"] = "failed"
        record["error"] = str(e)[:100]
    return record


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
    return {"total_reward": round(total, 2), "steps": step}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="176 严格窗口模式 N=20 扩充实验 v4（v2 同步提交对齐）"
    )
    parser.add_argument("--wait-hours", type=float, default=3.0)
    parser.add_argument("--max-episodes", type=int, default=60)
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="覆盖默认 seeds")
    parser.add_argument("--tag", type=str, default="20260809_v4", help="结果文件名 tag")
    args = parser.parse_args()

    seeds = list(args.seeds) if args.seeds else list(SEEDS_20)
    results_path = _PROJECT_ROOT / "results" / "real_machine" / f"round3_expansion_{args.tag}.json"

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
    total = min(args.max_episodes, len(seeds) * 3)
    policies = ["PPO", "FCFS", "SJF"]
    pairs = [(s, p) for s in seeds for p in policies]

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
        print("  探测成功！执行正式 episode + 同步真机任务", flush=True)
        seed, policy = pairs[done]
        try:
            env = QuantumSchedulingEnv(
                max_steps=EPISODE_HORIZON,
                machine_configs=[dict(c) for c in MACHINE_CONFIGS],
                seed=seed,
                use_real_machine=False,
                real_submit_probability=0.0,
            )
            ep = run_one_episode(env, seed, policy)
            env.close()
            real = submit_and_poll_one_task(client, f"rd3_{seed}_{policy}")
            results.append({**ep, "seed": seed, "strategy": policy, "real": real})
            print(
                f"  [{done + 1}/{total}] seed={seed} {policy} reward={ep['total_reward']:.1f} "
                f"real={real['status']} fid={real['fidelity']} mbs={real['measurement_balance_score']}",
                flush=True,
            )
            done += 1
            time.sleep(3)
        except Exception as e:
            print(f"  FAILED: {str(e)[:100]}", flush=True)
            results.append({"seed": seed, "strategy": policy, "error": str(e)})
            done += 1
            time.sleep(10)

    data = {
        "timestamp": datetime.now().isoformat(),
        "experiment": f"round3_expansion_{args.tag}",
        "machine": TARGET_MACHINE,
        "completed": done,
        "total": total,
        "results": results,
    }
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"结果已保存: {results_path}（完成 {done}/{total}）", flush=True)
    return 0 if done == total else 2


if __name__ == "__main__":
    sys.exit(main())
