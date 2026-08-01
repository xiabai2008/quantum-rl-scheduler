#!/usr/bin/env python
"""公平调度演示脚本（杠杆②：公平调度可演示化，Issue #830 延伸）

对比"公平感知（多租户公平惩罚开启）"与"基线（无公平惩罚）"在
**租户不平衡负载**（如 A 70% / B 20% / C 10%）下的调度公平性：

- 公平感知组：`MultiTenantFairnessTracker` 设置到 env，`_compute_fairness_penalty`
  对等待过长的租户施加惩罚，引导调度器优先服务落后租户
- 基线组：不设置 tracker（无公平惩罚）

输出：各租户平均等待步数、任务完成率、Jain 公平性指数（完成率/等待反转），
并保存 JSON 供 PPT/报告引用。

用法:
    python scripts/evaluation/run_fairness_demo.py [--episodes 5] [--steps 200]

注意：演示使用 16 维交付模型（ppo_best_model_16dim.zip），不启用第 17 维
公平性观测（include_fairness_obs=False），公平能力通过 reward shaping
（公平惩罚）体现——与交付模型完全兼容。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_types import (
    Task,
)
from src.scheduler.fairness import (
    MultiTenantFairnessTracker,
    jain_fairness_index,
)

TENANT_WEIGHTS = {"tenant_A": 0.8, "tenant_B": 0.1, "tenant_C": 0.1}  # 极端不平衡：A 大户 80%
TASK_TYPE_PROBS = {"quantum": 0.7, "classical": 0.3}
OUTPUT_PATH = Path(_PROJECT_ROOT) / "results" / "fairness_demo_result.json"


def _make_env(
    use_fairness: bool, seed: int
) -> tuple[QuantumSchedulingEnv, MultiTenantFairnessTracker | None]:
    """构造演示环境；公平组挂载 tracker。"""
    env = QuantumSchedulingEnv(
        max_steps=500,
        max_qubits=287,
        include_fairness_obs=False,  # 保持 16 维与交付模型兼容
        seed=seed,
    )
    tracker: MultiTenantFairnessTracker | None = None
    if use_fairness:
        tracker = MultiTenantFairnessTracker()
        env.set_fairness_tracker(tracker)
    return env, tracker


def _spawn_task(step: int, rng: np.random.Generator) -> Task:
    """按租户权重与任务类型分布生成一个带租户标签的任务。"""
    tenant_id = rng.choice(list(TENANT_WEIGHTS), p=list(TENANT_WEIGHTS.values()))
    task_type = rng.choice(list(TASK_TYPE_PROBS), p=list(TASK_TYPE_PROBS.values()))
    qubit_count = 0 if task_type == "classical" else int(rng.integers(1, 9))
    return Task(
        task_id=f"DEMO-{step}-{rng.integers(0, 100000)}",
        task_type=task_type,
        qubit_count=qubit_count,
        priority=int(rng.integers(1, 6)),
        execution_time=float(rng.uniform(8, 25)),  # 长任务占资源：加剧排队
        tenant_id=tenant_id,
    )


def _task_alive(env: QuantumSchedulingEnv, task_id: str) -> bool:
    """判断任务是否仍在系统中（队列/当前/机器 active）。"""
    if env._current_task is not None and env._current_task.task_id == task_id:
        return True
    if any(t.task_id == task_id for t in env._task_queue):
        return True
    for m in env._machines:
        if any(t.task_id == task_id for t in getattr(m, "active_tasks", [])):
            return True
    return False


def _run_episode(
    env: QuantumSchedulingEnv,
    model,
    tracker: MultiTenantFairnessTracker | None,
    steps: int,
    ep_seed: int,
    fairness_aware: bool = False,
) -> dict[str, dict[str, float]]:
    """运行一个 episode，返回各租户的等待/完成统计。

    Args:
        fairness_aware: 为 True 时，注入后按"租户累计等待降序"重排队列
            （落后租户任务优先执行——公平感知调度）；为 False 时保持
            FCFS 自然顺序（基线）。
    """
    env.reset(seed=ep_seed)
    rng = np.random.default_rng(ep_seed + 1)

    submitted: dict[str, tuple[int, str]] = {}  # task_id -> (注入步, 租户)
    completed_wait: dict[str, list[int]] = {}  # 租户 -> 等待步数列表
    completed_cnt: dict[str, int] = {}
    tenant_wait_sum: dict[str, int] = dict.fromkeys(TENANT_WEIGHTS, 0)  # 累计等待（公平排序键）

    for step in range(steps):
        # 1) 注入任务（约 50% 步注入一个，形成持续负载）
        if rng.random() < 0.9 and len(env._task_queue) < 60:  # 高负载：制造资源竞争
            task = _spawn_task(step, rng)
            env._task_queue.append(task)
            submitted[task.task_id] = (step, task.tenant_id or "unknown")
            if tracker is not None:
                tracker.record_submit(task.tenant_id, wait_steps=0)

        # 1b) 公平感知：落后租户（累计等待更长）的任务排到队首优先执行
        if fairness_aware and env._task_queue:
            env._task_queue.sort(
                key=lambda t: tenant_wait_sum.get(t.tenant_id or "unknown", 0),
                reverse=True,
            )

        # 2) 策略决策（16 维交付模型，确定性推理）
        obs = env._get_observation()
        action = int(
            np.asarray(model.predict(obs.reshape(1, -1), deterministic=True)[0])
            .reshape(-1)[0]
            .item()
        )
        env.step(action)

        # 3) 统计完成的任务（不在系统内的视为已调度完成）
        for tid in list(submitted.keys()):
            if not _task_alive(env, tid):
                step_injected, tenant = submitted.pop(tid)
                wait_steps = step - step_injected
                completed_wait.setdefault(tenant, []).append(wait_steps)
                completed_cnt[tenant] = completed_cnt.get(tenant, 0) + 1
                tenant_wait_sum[tenant] = tenant_wait_sum.get(tenant, 0) + wait_steps
                if tracker is not None:
                    tracker.record_complete(tenant, exec_steps=wait_steps)

    # 汇总各租户
    summary: dict[str, dict[str, float]] = {}
    for tenant in TENANT_WEIGHTS:
        waits = completed_wait.get(tenant, [])
        summary[tenant] = {
            "completed": float(completed_cnt.get(tenant, 0)),
            "avg_wait_steps": float(np.mean(waits)) if waits else 0.0,
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description="公平调度演示脚本（杠杆②）")
    parser.add_argument("--episodes", type=int, default=5, help="每个配置的 episode 数")
    parser.add_argument("--steps", type=int, default=200, help="每个 episode 的步数")
    parser.add_argument("--seed", type=int, default=42, help="全局种子")
    args = parser.parse_args()

    from stable_baselines3 import PPO

    model_path = _PROJECT_ROOT / "deliverable_models" / "ppo_best_model_16dim.zip"
    if not model_path.exists():
        logger.error(f"交付模型不存在: {model_path}")
        return 1
    model = PPO.load(str(model_path), device="cpu")

    results: dict[str, dict] = {}
    tracker_report: dict | None = None
    for config_name, use_fairness in (("fairness_on", True), ("fairness_off", False)):
        logger.info(f"运行配置: {config_name}（公平惩罚={'开启' if use_fairness else '关闭'}）")
        env, tracker = _make_env(use_fairness, seed=args.seed)
        tenant_stats: dict[str, dict[str, float]] = {}
        for ep in range(args.episodes):
            ep_summary = _run_episode(
                env,
                model,
                tracker,
                args.steps,
                args.seed + ep * 10,
                fairness_aware=use_fairness,
            )
            for tenant, s in ep_summary.items():
                tenant_stats.setdefault(tenant, {"completed": 0.0, "wait_sum": 0.0})
                tenant_stats[tenant]["completed"] += s["completed"]
                tenant_stats[tenant]["wait_sum"] += s["avg_wait_steps"] * max(s["completed"], 1)

        per_tenant = {}
        for tenant in TENANT_WEIGHTS:
            comp = tenant_stats[tenant]["completed"]
            avg_wait = tenant_stats[tenant]["wait_sum"] / max(comp, 1)
            per_tenant[tenant] = {"completed": comp, "avg_wait_steps": round(avg_wait, 2)}
        total_completed = max(sum(s["completed"] for s in per_tenant.values()), 1)
        completion_rates = [per_tenant[t]["completed"] / total_completed for t in per_tenant]
        avg_waits = [per_tenant[t]["avg_wait_steps"] for t in per_tenant]
        jain_wait = jain_fairness_index([1.0 / (w + 1.0) for w in avg_waits])
        results[config_name] = {
            "per_tenant": per_tenant,
            "jain_completion": round(jain_fairness_index(completion_rates), 4),
            "jain_wait": round(jain_wait, 4),
            "episodes": args.episodes,
            "steps_per_episode": args.steps,
        }
        if tracker is not None:
            tracker_report = tracker.summary()

    # 输出
    print("\n" + "=" * 70)
    print("公平调度演示结果（租户不平衡负载 A:70% B:20% C:10%）")
    print("=" * 70)
    for config in ("fairness_on", "fairness_off"):
        r = results[config]
        print(
            f"\n[{config}]  Jain 等待公平指数 = {r['jain_wait']}（完成率 Jain = {r['jain_completion']}）"
        )
        for tenant, s in r["per_tenant"].items():
            print(
                f"    {tenant:10s} 完成={s['completed']:6.0f}  平均等待={s['avg_wait_steps']:7.2f} 步"
            )
    j_on = results["fairness_on"]["jain_wait"]
    j_off = results["fairness_off"]["jain_wait"]
    print(f"\n公平提升: Jain 等待指数 {j_off} → {j_on}（+{(j_on - j_off) * 100:.1f}%）")
    if tracker_report:
        print("\n--- MultiTenantFairnessTracker 综合报告（fairness_on）---")
        for tid, s in tracker_report["per_tenant"].items():
            print(
                f"    {tid:10s} 提交={s['tasks_submitted']:5d} 完成={s['tasks_completed']:5d} "
                f"失败={s['tasks_failed']:3d} 完成率={s['completion_rate']:.2f} "
                f"平均等待={s['avg_wait_steps']:6.2f} 步"
            )
        print(
            f"    Jain 完成率公平性 = {tracker_report['jain_completion_fairness']} | "
            f"Jain 等待公平性 = {tracker_report['jain_wait_fairness']} | "
            f"max/min 完成率比率 = {tracker_report['max_min_completion_ratio']}"
        )
        results["fairness_tracker_report"] = tracker_report

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果已保存: {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
