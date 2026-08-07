#!/usr/bin/env python
"""
高参与率真机性能补强实验（Issue #221 预注册执行版）

背景：历史真机实验 real_submit_probability=0.0 / 1 task per run，
真机 reward 占比仅 1/96 步，结论"主要由仿真 reward 驱动"。
本脚本：机时充足前提下提高真机参与率，让真机保真度反馈实质进入 reward。

设计（遵循预注册 real_machine_preregistration.md）：
  - 3 策略：PPO(16维交付模型) / FCFS / SJF
  - N=20 seeds（预注册要求 ≥18，80% 功效）
  - real_submit_probability=0.3（每步 30% 概率提交真机任务）
  - max_real_submissions=30/seed（机时充足，cap 提至 30）
  - real_feedback_mode=result_aware（真机保真度线性映射进 reward）
  - 主指标：Cohen's d + 95% CI（效应量优先，p 值辅助）
  - task_id 100% 留档（Issue #455 口径）

用法：
    python scripts/real_machine/run_real_performance_pretrain.py --smoke   # 冒烟 1 任务
    python scripts/real_machine/run_real_performance_pretrain.py --seeds 42 123   # 指定 seeds
    python scripts/real_machine/run_real_performance_pretrain.py           # 默认 20 seeds
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

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.baselines import EnvBasedFCFSScheduler, EnvBasedSPTFScheduler
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv


class RetryClient:
    """提交失败重试包装：tianyan176 免费机高并发时任务被平台丢弃概率高，
    低频重试可显著提高成功率（手动测试 5/5 vs 高并发 1/5）。
    只包装 submit_quantum_task / get_task_status，其余方法透传。
    """

    def __init__(self, inner, retries: int = 2, delay: float = 1.5):
        self._inner = inner
        self._retries = retries
        self._delay = delay

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def submit_quantum_task(self, qcis="", circuit=None, shots=1024, task_name="Scheduler_Task"):
        for attempt in range(self._retries + 1):
            try:
                tid = self._inner.submit_quantum_task(
                    qcis=qcis, circuit=circuit, shots=shots, task_name=task_name
                )
            except Exception as e:
                tid = None
                last_err = e
            if tid is not None:
                return tid
            if attempt < self._retries:
                time.sleep(self._delay)
        if "last_err" in locals():
            raise last_err
        return None

    def get_task_status(self, task_id):
        return self._inner.get_task_status(task_id)

    def wait_for_task(self, task_id, timeout=300, poll_interval=5):
        return self._inner.wait_for_task(task_id, timeout=timeout, poll_interval=poll_interval)

# ── 实验配置（预注册口径 + 高参与率） ──

SEEDS_DEFAULT = [42, 123, 456, 789, 1024, 2026, 314, 271, 828, 5566,
                 7788, 1234, 2345, 3456, 4567, 5678, 6789, 7890, 8901, 9012]

EPISODE_HORIZON = 500        # 步数上限（真机任务多，episode 较长）
CAP_PER_SEED = 30            # 每个 seed 真机提交上限（预注册 cap=10 → 30）
REAL_SUBMIT_PROB = 0.3       # 每步真机提交概率（预注册 0.05 → 0.3）
SHOTS = 32                   # 与历史验证口径一致（Issue #58 已核实 shots=32）
FEEDBACK_MODE = "result_aware"
TARGET_MACHINE = "tianyan176"  # 287 校准中，暂用 176（free, running）
CIRCUIT = "H Q1\nM Q1"       # 单比特 H 门（免费档兼容；若机时升级可换多 qubit）

PPO_MODEL_PATH = _PROJECT_ROOT / "deliverable_models" / "ppo_best_model_16dim.zip"
OUTPUT_DIR = _PROJECT_ROOT / "results" / "real_machine" / "perf_pretrain_v3"

# 机器配置必须与目标真机名一致（attach_real_clients 按名字匹配）
MACHINE_CONFIGS = [
    {
        "name": TARGET_MACHINE,
        "machine_type": "quantum",
        "max_qubits": 105,
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


def create_policy(name: str):
    """创建调度策略对象（select_action(obs) 接口）。"""
    if name == "fcfs":
        return EnvBasedFCFSScheduler()
    if name == "sjf":
        return EnvBasedSPTFScheduler()
    if name == "ppo":
        from stable_baselines3 import PPO as SB3PPO

        model = SB3PPO.load(str(PPO_MODEL_PATH))
        from scripts.evaluation.run_simulation import PPOStrategy

        return PPOStrategy(model)
    raise ValueError(f"未知策略: {name}")


def run_one(seed: int, policy_name: str, dry_run: bool = False, client=None) -> dict:
    """运行单个 seed 的单策略 episode，真机任务全留档。"""
    env = QuantumSchedulingEnv(
        max_steps=EPISODE_HORIZON,
        machine_configs=MACHINE_CONFIGS,
        seed=seed,
        use_real_machine=not dry_run,
        real_submit_probability=REAL_SUBMIT_PROB,
        max_real_submissions=CAP_PER_SEED,
        real_machine_shots=SHOTS,
        real_feedback_mode=FEEDBACK_MODE,
        real_machine_max_qubits=int(os.environ.get("FREE_TIER_MAX_QUBITS", "1")),
    )
    if client is not None:
        env.attach_real_clients({TARGET_MACHINE: client})
    policy = create_policy(policy_name)
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    real_submitted = 0
    real_completed = 0
    start = time.time()
    done = False

    while not done:
        action = policy.select_action(obs, env) if policy_name in ("fcfs", "sjf") else policy.select_action(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated

    records = []
    if hasattr(env, "_real_feedback_log"):
        for r in env._real_feedback_log:
            records.append({
                "task_id": r.get("task_id"),
                "real_task_id": r.get("real_task_id"),
                "status": r.get("outcome"),
                "fidelity": r.get("fidelity"),
                "step": r.get("submit_step"),
                "reward": r.get("reward"),
                "mock": False,
                "degraded": False,
            })
            if r.get("outcome") == "completed":
                real_completed += 1
            if r.get("real_task_id") is not None:
                real_submitted += 1

    env.close()
    return {
        "seed": seed,
        "policy": policy_name,
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "elapsed_seconds": round(time.time() - start, 2),
        "real_tasks_submitted": real_submitted,
        "real_tasks_completed": real_completed,
        "real_records": records,
    }


def main():
    parser = argparse.ArgumentParser(description="高参与率真机性能实验（预注册 v3）")
    parser.add_argument("--seeds", type=int, nargs="+", default=SEEDS_DEFAULT)
    parser.add_argument("--smoke", action="store_true", help="冒烟模式：仅 1 seed × 1 策略验证链路")
    parser.add_argument("--dry-run", action="store_true", help="仿真干跑（不碰真机）")
    args = parser.parse_args()

    api_key = os.environ.get("TIANYAN_API_KEY", "")
    if not api_key:
        print("错误: 未设置 TIANYAN_API_KEY")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        seeds, policies = [42], ["fcfs"]
    else:
        seeds, policies = args.seeds, ["ppo", "fcfs", "sjf"]

    if not args.dry_run:
        raw_client = CqlibTianyanClient(login_key=api_key, machine_name=TARGET_MACHINE, auto_retry_machine=False)
        client = RetryClient(raw_client)
        print(f"客户端: {getattr(raw_client, 'machine_name', '?')}  | 种子数: {len(seeds)}  | 策略: {policies}")
    else:
        client = None
        print(f"DRY RUN（仿真） | 种子数: {len(seeds)}  | 策略: {policies}")

    results: list[dict] = []
    total = len(seeds) * len(policies)
    done_count = 0
    for seed in seeds:
        for pname in policies:
            done_count += 1
            print(f"[{done_count}/{total}] seed={seed} policy={pname} ...", flush=True)
            try:
                r = run_one(seed, pname, dry_run=args.dry_run, client=client)
                results.append(r)
                print(
                    f"    reward={r['total_reward']:.1f} steps={r['steps']} "
                    f"real_sub={r['real_tasks_submitted']} completed={r['real_tasks_completed']}"
                )
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {e}")
                results.append({"seed": seed, "policy": pname, "error": str(e)})

    out = {
        "experiment": "real_performance_pretrain_v3",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seeds": seeds,
            "policies": policies,
            "episode_horizon": EPISODE_HORIZON,
            "cap_per_seed": CAP_PER_SEED,
            "real_submit_probability": REAL_SUBMIT_PROB,
            "shots": SHOTS,
            "feedback_mode": FEEDBACK_MODE,
            "machine": TARGET_MACHINE,
            "circuit": CIRCUIT,
            "model": str(PPO_MODEL_PATH.name),
        },
        "results": results,
    }
    out_path = OUTPUT_DIR / f"perf_v3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
