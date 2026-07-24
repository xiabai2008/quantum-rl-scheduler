#!/usr/bin/env python
"""
高负载场景(λ=1.2)多租户公平调度重测实验 (Day4-7-13)

对比策略: PPO vs FCFS vs SJF
租户数量: 5 (tenant_A~E, 优先级 5~1)
Seeds: [42, 123, 456, 789, 1024]
每个 seed 运行 1 episode, 200 步/episode
到达率: arrival_lambda=1.2 (高负载, 任务到达率 > 调度吞吐量)

输出:
  - results/fair_comparison/high_load_fairness.json
  - results/reports/high_load_fairness_report.md

用法:
  python scripts/benchmarking/high_load_fairness.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 项目根目录注入
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_dynamics import generate_random_task
from src.scheduler.env_types import (
    ACTION_CLASSICAL,
    ACTION_HYBRID,
    ACTION_QUANTUM,
    MAX_WAIT_STEPS,
)
from src.scheduler.fairness import jain_fairness_index

# ---------------------------------------------------------------------------
# 实验配置
# ---------------------------------------------------------------------------
ARRIVAL_LAMBDA = 1.2  # 高负载到达率
MAX_STEPS = 200  # 每 episode 步数
NUM_TENANTS = 5
SEEDS = [42, 123, 456, 789, 1024]
STARVATION_THRESHOLD = MAX_WAIT_STEPS  # 50 步

# 租户配置（与 λ=0.5 报告一致：优先级 5→1）
TENANTS = [
    {"id": "tenant_A", "priority": 5, "weight": "high"},
    {"id": "tenant_B", "priority": 4, "weight": "high"},
    {"id": "tenant_C", "priority": 3, "weight": "medium"},
    {"id": "tenant_D", "priority": 2, "weight": "low"},
    {"id": "tenant_E", "priority": 1, "weight": "low"},
]
TENANT_IDS = [t["id"] for t in TENANTS]


# ---------------------------------------------------------------------------
# 租户指标追踪
# ---------------------------------------------------------------------------
@dataclass
class TenantMetrics:
    """单个租户的公平性指标。"""

    tenant_id: str
    priority: int
    total_reward: float = 0.0
    quantum_alloc: int = 0
    classical_alloc: int = 0
    hybrid_alloc: int = 0
    total_scheduled: int = 0
    total_wait_steps: int = 0
    starvation_count: int = 0

    @property
    def total_alloc(self) -> int:
        return self.quantum_alloc + self.classical_alloc + self.hybrid_alloc

    @property
    def avg_wait(self) -> float:
        return self.total_wait_steps / self.total_scheduled if self.total_scheduled else 0.0

    def to_dict(self) -> dict:
        return {
            "tenant_id": self.tenant_id,
            "priority": self.priority,
            "total_reward": round(self.total_reward, 2),
            "quantum_alloc": self.quantum_alloc,
            "classical_alloc": self.classical_alloc,
            "hybrid_alloc": self.hybrid_alloc,
            "total_alloc": self.total_alloc,
            "total_scheduled": self.total_scheduled,
            "avg_wait": round(self.avg_wait, 2),
            "starvation_count": self.starvation_count,
        }


# ---------------------------------------------------------------------------
# 租户感知环境包装器
# ---------------------------------------------------------------------------
class TenantAwareEnv:
    """包装 QuantumSchedulingEnv，为任务分配租户 ID 并追踪每租户指标。

    通过猴子补丁替换环境的 _generate_random_task 方法，为每个生成的任务
    轮询分配租户 ID（确保任务均匀分布到 5 个租户），并根据租户优先级
    设置任务优先级。同时在每次 step 时将奖励和资源分配归因到对应租户。
    """

    def __init__(self, arrival_lambda: float = ARRIVAL_LAMBDA, max_steps: int = MAX_STEPS) -> None:
        self.env = QuantumSchedulingEnv(
            max_steps=max_steps,
            max_qubits=287,
            arrival_lambda=arrival_lambda,
        )
        self.tenant_metrics: dict[str, TenantMetrics] = {}
        self._tenant_counter = 0
        self._episode_reward = 0.0
        # 猴子补丁：替换任务生成器以注入租户 ID
        self.env._generate_random_task = self._generate_tenant_task

    def _generate_tenant_task(self, rng: np.random.Generator, task_id: int):
        """生成任务并轮询分配租户 ID，优先级对齐租户优先级。"""
        task = generate_random_task(rng, task_id, self.env.quantum_task_ratio)
        tenant = TENANTS[self._tenant_counter % NUM_TENANTS]
        self._tenant_counter += 1
        task.tenant_id = tenant["id"]
        task.priority = tenant["priority"]
        return task

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        """重置环境与租户指标。"""
        self._tenant_counter = 0
        self._episode_reward = 0.0
        self.tenant_metrics = {t["id"]: TenantMetrics(t["id"], t["priority"]) for t in TENANTS}
        return self.env.reset(seed=seed, options=options)

    def step(self, action: int):
        """执行一步调度，归因奖励与资源分配到租户。"""
        current_task = self.env._current_task
        current_tenant = current_task.tenant_id if current_task else None
        wait_before = current_task.wait_steps if current_task else 0

        # 记录 step 前的成功计数
        q_before = self.env._quantum_success
        c_before = self.env._classical_success
        h_before = self.env._hybrid_success
        sched_before = self.env._total_scheduled

        obs, reward, terminated, truncated, info = self.env.step(action)
        self._episode_reward += reward

        # 归因到租户
        if current_tenant and current_tenant in self.tenant_metrics:
            m = self.tenant_metrics[current_tenant]
            m.total_reward += reward

            q_delta = self.env._quantum_success - q_before
            c_delta = self.env._classical_success - c_before
            h_delta = self.env._hybrid_success - h_before
            sched_delta = self.env._total_scheduled - sched_before

            if sched_delta > 0:
                m.total_scheduled += sched_delta
                m.quantum_alloc += q_delta
                m.classical_alloc += c_delta
                m.hybrid_alloc += h_delta
                m.total_wait_steps += wait_before
                if wait_before > STARVATION_THRESHOLD:
                    m.starvation_count += 1

        return obs, reward, terminated, truncated, info

    @property
    def episode_reward(self) -> float:
        return self._episode_reward

    def compute_fairness(self) -> dict:
        """计算 Jain's Fairness Index（资源/奖励/等待）。"""
        rewards = [float(self.tenant_metrics[tid].total_reward) for tid in TENANT_IDS]
        allocs = [float(self.tenant_metrics[tid].total_alloc) for tid in TENANT_IDS]
        waits = [self.tenant_metrics[tid].avg_wait for tid in TENANT_IDS]
        # 等待公平性：反转等待时间（等待越短得分越高）
        inverted_waits = [1.0 / (w + 1.0) for w in waits]
        return {
            "jain_resource": round(jain_fairness_index(allocs), 4),
            "jain_reward": round(jain_fairness_index(rewards), 4),
            "jain_wait": round(jain_fairness_index(inverted_waits), 4),
        }


# ---------------------------------------------------------------------------
# 调度策略
# ---------------------------------------------------------------------------
class PPOStrategy:
    """PPO 强化学习调度策略。"""

    name = "PPO"

    def __init__(self, model) -> None:
        self.model = model

    def select_action(self, obs: np.ndarray) -> int:
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action.item())


class FCFSStrategy:
    """FCFS：先来先服务（环境内部按优先级/等待排序取队首，资源分配选混合执行）。"""

    name = "FCFS"

    def select_action(self, obs: np.ndarray) -> int:
        return ACTION_HYBRID  # 混合执行，最高兼容性


class SJFStrategy:
    """SJF：最短作业优先（根据队列长度和资源可用性选择资源）。"""

    name = "SJF"

    def select_action(self, obs: np.ndarray) -> int:
        qubit_availability = obs[0]
        classical_load = obs[4]
        queue_length = obs[1]
        # 队列很长时用混合执行提高吞吐量
        if queue_length > 0.6:
            return ACTION_HYBRID
        # 量子资源充足时优先量子
        if qubit_availability > 0.5:
            return ACTION_QUANTUM
        # 经典资源空闲时用经典
        if classical_load < 0.5:
            return ACTION_CLASSICAL
        return ACTION_HYBRID


# ---------------------------------------------------------------------------
# 单次 episode 运行
# ---------------------------------------------------------------------------
def run_episode(wrapper: TenantAwareEnv, strategy, seed: int) -> tuple:
    """运行单个 episode，返回总奖励、租户指标和公平性指数。"""
    obs, _info = wrapper.reset(seed=seed)
    step = 0
    while step < MAX_STEPS:
        action = strategy.select_action(obs)
        obs, _reward, terminated, truncated, _info = wrapper.step(action)
        step += 1
        if terminated or truncated:
            break
    return wrapper.episode_reward, wrapper.tenant_metrics, wrapper.compute_fairness()


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------
def generate_report(all_results: dict, elapsed: float) -> str:
    """生成 Markdown 报告。"""
    exp = all_results["experiment"]
    results = all_results["results"]

    lines = [
        "# 高负载场景(λ=1.2)公平调度重测报告",
        "",
        f"> **生成时间**: {exp['timestamp']}",
        "> **实验任务**: Day4-7-13 高负载场景公平调度重测",
        "> **对应比赛要求**: 方向3'多用户资源管理与公平调度'",
        "---",
        "",
        "## 1. 实验配置",
        "",
        "| 参数 | 值 |",
        "|------|-----|",
        f"| 到达率 (arrival_lambda) | {exp['arrival_lambda']} (高负载) |",
        f"| 租户数量 | {exp['num_tenants']} |",
        f"| Seeds | {exp['seeds']} |",
        "| 每 seed 运行 | 1 episode |",
        f"| 每 episode 步数 | {exp['max_steps']} |",
        f"| 对比策略 | {', '.join(exp['strategies'])} |",
        f"| 总运行次数 | {len(exp['seeds']) * len(exp['strategies'])} |",
        f"| 总耗时 | {elapsed:.1f}s |",
        "",
        "### 租户配置",
        "",
        "| 租户 | 优先级 | 权重等级 |",
        "|------|--------|---------|",
    ]

    for t in exp["tenant_config"]:
        lines.append(f"| {t['id']} | {t['priority']} | {t['weight']} |")

    lines.extend(
        [
            "",
            "### 高负载场景说明",
            "",
            "arrival_lambda=1.2 表示每步平均到达 1.2 个新任务（泊松分布），",
            "而环境每步最多调度 1 个任务。这意味着任务到达率 > 调度吞吐量，",
            "队列将持续增长，形成资源争抢。这是验证公平调度和防饥饿机制的关键场景。",
            "",
            "---",
            "",
            "## 2. Jain's Fairness Index 对比",
            "",
            "| 策略 | Jain's Index(资源) | Jain's Index(奖励) | Jain's Index(等待) | 总奖励 |",
            "|------|:------------------:|:------------------:|:-------------------:|:------:|",
        ]
    )

    for sname in exp["strategies"]:
        r = results[sname]
        lines.append(
            f"| {sname} | "
            f"{r['jain_resource_mean']:.4f} ± {r['jain_resource_std']:.4f} | "
            f"{r['jain_reward_mean']:.4f} ± {r['jain_reward_std']:.4f} | "
            f"{r['jain_wait_mean']:.4f} ± {r['jain_wait_std']:.4f} | "
            f"{r['total_reward_mean']:.2f} ± {r['total_reward_std']:.2f} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. 各租户奖励分布",
            "",
        ]
    )

    for sname in exp["strategies"]:
        r = results[sname]
        lines.extend(
            [
                f"### {sname} 策略",
                "",
                "| 租户 | 优先级 | 平均奖励 | 奖励标准差 | 量子分配 | 经典分配 | 混合分配 | 总分配 | 平均等待 | 饥饿数 |",
                "|------|--------|---------|-----------|---------|---------|---------|-------|---------|--------|",
            ]
        )
        for tid in TENANT_IDS:
            t = r["per_tenant"][tid]
            lines.append(
                f"| {tid} | {t['priority']} | {t['avg_reward']:.2f} | {t['std_reward']:.2f} | "
                f"{t['avg_quantum_alloc']:.1f} | {t['avg_classical_alloc']:.1f} | "
                f"{t['avg_hybrid_alloc']:.1f} | {t['avg_total_alloc']:.1f} | "
                f"{t['avg_wait']:.2f} | {t['avg_starvation']:.2f} |"
            )
        lines.append("")

    lines.extend(
        [
            "---",
            "",
            "## 4. 与 λ=0.5 结果的对比分析",
            "",
            "### 4.1 对比基准",
            "",
            "| 指标 | λ=0.5 (低负载) | λ=1.2 (高负载) | 变化趋势 |",
            "|------|:-------------:|:-------------:|:--------:|",
            "",
        ]
    )

    # λ=0.5 基准数据（来自 results/reports/fair_scheduling_report.md）
    baseline_05 = {
        "PPO": {"jain_resource": 0.9875, "jain_reward": 0.9238, "total_reward": 2334.66},
        "FCFS": {"jain_resource": 0.9875, "jain_reward": 0.9879, "total_reward": 1481.021},
    }

    for sname in ["PPO", "FCFS"]:
        if sname in results and sname in baseline_05:
            b = baseline_05[sname]
            h = results[sname]
            jain_res_change = h["jain_resource_mean"] - b["jain_resource"]
            jain_rew_change = h["jain_reward_mean"] - b["jain_reward"]
            reward_change = (
                (h["total_reward_mean"] - b["total_reward"]) / abs(b["total_reward"]) * 100
            )
            lines.append(
                f"| {sname} Jain's(资源) | {b['jain_resource']:.4f} | {h['jain_resource_mean']:.4f} | "
                f"{'+' if jain_res_change >= 0 else ''}{jain_res_change:.4f} |"
            )
            lines.append(
                f"| {sname} Jain's(奖励) | {b['jain_reward']:.4f} | {h['jain_reward_mean']:.4f} | "
                f"{'+' if jain_rew_change >= 0 else ''}{jain_rew_change:.4f} |"
            )
            lines.append(
                f"| {sname} 总奖励 | {b['total_reward']:.2f} | {h['total_reward_mean']:.2f} | "
                f"{'+' if reward_change >= 0 else ''}{reward_change:.1f}% |"
            )

    lines.extend(
        [
            "",
            "### 4.2 分析",
            "",
            "**负载压力影响**：λ=1.2 时任务到达率（1.2/步）超过调度吞吐量（1/步），",
            "队列持续增长，系统处于高负载压力状态。与 λ=0.5 的宽松场景相比，",
            "高负载下资源争抢加剧，公平性指标的分化更能体现策略差异。",
            "",
            "**奖励公平性**：在高负载下，PPO 通过 14 维观测空间感知任务紧急度，",
            "优先为高优先级租户分配量子资源（更高奖励），导致奖励分配的 Jain's Index",
            "可能低于 FCFS。这是有意为之的优先级差异化，而非不公平。",
            "",
            "**资源公平性**：FCFS 由于采用统一的混合执行策略，各租户资源分配更均匀；",
            "PPO 则根据任务特性智能选择资源类型，在高负载下可能更倾向于量子资源分配给高优先级租户。",
            "",
            "**防饥饿机制**：高负载场景是检验防饥饿机制的关键。",
            "通过 Jain's Index(等待) 和饥饿数指标，可评估各策略是否能在高负载下",
            "避免低优先级租户被饿死。",
            "",
            "---",
            "",
            "## 5. 答辩要点",
            "",
            "1. **高负载验证**: λ=1.2 模拟任务到达率 > 调度吞吐量的高负载场景，",
            "   直接验证公平调度在资源争抢下的鲁棒性",
            "2. **三策略对比**: PPO vs FCFS vs SJF，覆盖 RL/经典启发式/资源感知三类策略",
            "3. **多维公平性**: Jain's Index 覆盖资源分配/奖励分布/等待时间三个维度",
            "4. **负载对比**: 与 λ=0.5 低负载结果对比，展示公平性随负载压力的变化趋势",
            "",
            "---",
            "",
            f"*本报告由 high_load_fairness.py 自动生成 ({exp['timestamp']})*",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 64)
    print("  高负载场景(λ=1.2)公平调度重测实验 — Day4-7-13")
    print("=" * 64)
    print(f"  到达率:        {ARRIVAL_LAMBDA}")
    print(f"  租户数:        {NUM_TENANTS}")
    print(f"  Seeds:         {SEEDS}")
    print(f"  步数/episode:  {MAX_STEPS}")
    print("  策略:          PPO, FCFS, SJF")
    print("=" * 64)

    # 加载 PPO 模型
    from stable_baselines3 import PPO

    ppo_path = str(_PROJECT_ROOT / "deliverable_models" / "ppo_best_model_14dim.zip")
    print(f"[PPO] 加载模型: {ppo_path}")
    ppo_env = QuantumSchedulingEnv(
        max_steps=MAX_STEPS, max_qubits=287, arrival_lambda=ARRIVAL_LAMBDA
    )
    ppo_model = PPO.load(ppo_path, env=ppo_env)
    print("[PPO] 模型加载成功")

    strategies = [
        PPOStrategy(ppo_model),
        FCFSStrategy(),
        SJFStrategy(),
    ]

    all_results: dict[str, Any] = {}
    start_time = time.time()

    for strategy in strategies:
        print(f"\n--- 运行策略: {strategy.name} ---")
        seed_results = []

        for seed in SEEDS:
            wrapper = TenantAwareEnv(arrival_lambda=ARRIVAL_LAMBDA, max_steps=MAX_STEPS)
            reward, tenant_metrics, fairness = run_episode(wrapper, strategy, seed)

            seed_results.append(
                {
                    "seed": seed,
                    "total_reward": round(reward, 2),
                    "fairness": fairness,
                    "per_tenant": {tid: tenant_metrics[tid].to_dict() for tid in TENANT_IDS},
                }
            )
            print(
                f"  seed={seed}: reward={reward:.2f} | "
                f"jain_res={fairness['jain_resource']:.4f} | "
                f"jain_rew={fairness['jain_reward']:.4f} | "
                f"jain_wait={fairness['jain_wait']:.4f}"
            )

        # 聚合统计
        rewards = [r["total_reward"] for r in seed_results]
        jain_res = [r["fairness"]["jain_resource"] for r in seed_results]
        jain_rew = [r["fairness"]["jain_reward"] for r in seed_results]
        jain_wait = [r["fairness"]["jain_wait"] for r in seed_results]

        per_tenant_avg: dict[str, Any] = {}
        for tid in TENANT_IDS:
            t_rewards = [r["per_tenant"][tid]["total_reward"] for r in seed_results]
            t_quantum = [r["per_tenant"][tid]["quantum_alloc"] for r in seed_results]
            t_classical = [r["per_tenant"][tid]["classical_alloc"] for r in seed_results]
            t_hybrid = [r["per_tenant"][tid]["hybrid_alloc"] for r in seed_results]
            t_total = [r["per_tenant"][tid]["total_alloc"] for r in seed_results]
            t_wait = [r["per_tenant"][tid]["avg_wait"] for r in seed_results]
            t_starv = [r["per_tenant"][tid]["starvation_count"] for r in seed_results]
            per_tenant_avg[tid] = {
                "priority": next(t["priority"] for t in TENANTS if t["id"] == tid),
                "avg_reward": round(float(np.mean(t_rewards)), 2),
                "std_reward": round(float(np.std(t_rewards)), 2),
                "avg_quantum_alloc": round(float(np.mean(t_quantum)), 1),
                "avg_classical_alloc": round(float(np.mean(t_classical)), 1),
                "avg_hybrid_alloc": round(float(np.mean(t_hybrid)), 1),
                "avg_total_alloc": round(float(np.mean(t_total)), 1),
                "avg_wait": round(float(np.mean(t_wait)), 2),
                "avg_starvation": round(float(np.mean(t_starv)), 2),
            }

        all_results[strategy.name] = {
            "total_reward_mean": round(float(np.mean(rewards)), 2),
            "total_reward_std": round(float(np.std(rewards)), 2),
            "jain_resource_mean": round(float(np.mean(jain_res)), 4),
            "jain_resource_std": round(float(np.std(jain_res)), 4),
            "jain_reward_mean": round(float(np.mean(jain_rew)), 4),
            "jain_reward_std": round(float(np.std(jain_rew)), 4),
            "jain_wait_mean": round(float(np.mean(jain_wait)), 4),
            "jain_wait_std": round(float(np.std(jain_wait)), 4),
            "per_tenant": per_tenant_avg,
            "per_seed": seed_results,
        }

        print(
            f"  汇总: reward={np.mean(rewards):.2f}±{np.std(rewards):.2f} | "
            f"jain_res={np.mean(jain_res):.4f} | "
            f"jain_rew={np.mean(jain_rew):.4f} | "
            f"jain_wait={np.mean(jain_wait):.4f}"
        )

    elapsed = time.time() - start_time

    # 组装输出
    output = {
        "experiment": {
            "name": "高负载场景(λ=1.2)公平调度重测",
            "task": "Day4-7-13",
            "arrival_lambda": ARRIVAL_LAMBDA,
            "num_tenants": NUM_TENANTS,
            "seeds": SEEDS,
            "max_steps": MAX_STEPS,
            "strategies": [s.name for s in strategies],
            "tenant_config": TENANTS,
            "timestamp": datetime.now().isoformat(),
            "elapsed_seconds": round(elapsed, 2),
        },
        "results": all_results,
    }

    # 保存 JSON
    json_path = str(_PROJECT_ROOT / "results" / "fair_comparison" / "high_load_fairness.json")
    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] JSON 结果: {json_path}")

    # 生成报告
    report = generate_report(output, elapsed)
    report_path = str(_PROJECT_ROOT / "results" / "reports" / "high_load_fairness_report.md")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[保存] Markdown 报告: {report_path}")

    # 打印汇总表格
    print("\n" + "=" * 64)
    print("  实验结果汇总")
    print("=" * 64)
    print(f"  {'策略':<8} {'Jain(资源)':<16} {'Jain(奖励)':<16} {'Jain(等待)':<16} {'总奖励':<20}")
    print("  " + "-" * 76)
    for sname in [s.name for s in strategies]:
        r = all_results[sname]
        print(
            f"  {sname:<8} "
            f"{r['jain_resource_mean']:.4f}±{r['jain_resource_std']:.4f}   "
            f"{r['jain_reward_mean']:.4f}±{r['jain_reward_std']:.4f}   "
            f"{r['jain_wait_mean']:.4f}±{r['jain_wait_std']:.4f}   "
            f"{r['total_reward_mean']:.2f}±{r['total_reward_std']:.2f}"
        )
    print("=" * 64)
    print(f"\n实验完成！总耗时: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
