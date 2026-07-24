"""Issue #165：纯仿真、混合真机、纯真机 PPO 三条件消融。"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import yaml
from dotenv import load_dotenv
from stable_baselines3 import PPO

plt.switch_backend("Agg")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.training.train_agent_real import (
    AuditedRealClient,
    TrainingCurveCallback,
    _machine_config,
    completion_rate_from_info,
)
from src.api.quota_tracker import QuotaTracker
from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.env import QuantumSchedulingEnv

DEFAULT_RESULTS = _PROJECT_ROOT / "results" / "real_machine" / "issue165_ablation.json"
DEFAULT_REPORT = _PROJECT_ROOT / "results" / "reports" / "real_machine_ablation.md"
DEFAULT_PLOT = _PROJECT_ROOT / "results" / "real_machine" / "real_machine_ablation.png"
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "issue165"
DEFAULT_QUOTA_STATE = _PROJECT_ROOT / "results" / "real_machine" / "issue165_quota_state.json"
DEFAULT_QUOTA_CONFIG = _PROJECT_ROOT / "results" / "real_machine" / "issue165_quota_budget.yaml"
PHYSICAL_MACHINES = frozenset({"tianyan176", "tianyan176-2"})


def write_hard_budget(
    path: Path,
    *,
    formal_submission_cap: int,
    shots: int,
    confirmed_machine_minutes: float,
) -> None:
    """写入本次实验专用预算；该文件位于忽略目录，不改变全局配额。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    total_tasks = formal_submission_cap + 1  # 加上预检冒烟
    payload = {
        "total_quota": {
            "shots": total_tasks * shots,
            "tasks": total_tasks,
            "wall_time_hours": confirmed_machine_minutes / 60.0,
        },
        "warning_threshold": 0.8,
        "critical_threshold": 0.95,
        "notification": {"type": "log", "webhook_url": None},
    }
    path.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")


def run_preflight(
    *,
    machine: str,
    shots: int,
    formal_submission_cap: int,
    confirmed_machine_minutes: float,
    confirmed_used_seconds: float,
    observed_task_upper_seconds: float,
    prior_failed_task_id: str | None,
    prior_failed_task_shots: int,
    wait_timeout: int,
    quota_config_path: Path,
    quota_state_path: Path,
) -> tuple[CqlibTianyanClient, QuotaTracker, dict[str, Any]]:
    """执行 dotenv、认证、后端/预算检查和一次 1-qubit 冒烟。"""
    if machine not in PHYSICAL_MACHINES:
        raise RuntimeError(f"{machine} 不是 #165 允许的物理真机")
    if confirmed_machine_minutes <= 0:
        raise RuntimeError("必须提供仪表盘确认的正数剩余机时")
    if confirmed_used_seconds < 0 or observed_task_upper_seconds <= 0:
        raise RuntimeError("仪表盘已用机时和单任务估算必须有效")

    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.getenv("TIANYAN_API_KEY", "")
    if not api_key:
        raise RuntimeError("TIANYAN_API_KEY 未配置")

    write_hard_budget(
        quota_config_path,
        formal_submission_cap=formal_submission_cap,
        shots=shots,
        confirmed_machine_minutes=confirmed_machine_minutes,
    )
    quota = QuotaTracker(
        config_path=str(quota_config_path),
        state_path=str(quota_state_path),
    )
    required_tasks = formal_submission_cap + 1
    required_shots = required_tasks * shots
    remaining_before = quota.remaining()
    if not quota.can_consume(tasks=required_tasks, shots=required_shots):
        raise RuntimeError(
            "#165 本地硬预算不足："
            f"需要 {required_tasks} tasks/{required_shots} shots，剩余 {remaining_before}"
        )

    client = CqlibTianyanClient(
        login_key=api_key,
        machine_name=machine,
        auto_retry_machine=False,
        quota_tracker=quota,
    )
    if not client.authenticate():
        raise RuntimeError("cqlib authenticate 失败")
    backends = client.list_backends()
    selected = next((backend for backend in backends if backend.get("name") == machine), None)
    if selected is None:
        raise RuntimeError(f"list_backends 未返回 {machine}")
    if selected.get("status") != "running":
        raise RuntimeError(f"{machine} 当前不是 running: {selected.get('status')}")

    started = time.perf_counter()
    smoke_id = client.submit_quantum_task(
        qcis="H Q0\nM Q0",
        shots=shots,
        task_name=f"issue165_smoke_{datetime.now():%Y%m%d_%H%M%S}",
    )
    if smoke_id is None:
        raise RuntimeError("1-qubit 冒烟提交被平台拒绝；停止正式实验")
    smoke = client.wait_for_task(str(smoke_id), timeout=wait_timeout, poll_interval=2)
    if smoke.get("status") != "completed":
        raise RuntimeError(f"1-qubit 冒烟未完成: {smoke.get('status')}")

    return (
        client,
        quota,
        {
            "authenticate": True,
            "machine": machine,
            "backend_type": selected.get("type"),
            "backend_status": selected.get("status"),
            "platform_quota_api_available": False,
            "quota_evidence": "user_confirmed_dashboard_remaining_machine_minutes",
            "confirmed_machine_minutes": confirmed_machine_minutes,
            "confirmed_used_seconds": confirmed_used_seconds,
            "observed_task_upper_seconds": observed_task_upper_seconds,
            "estimated_max_machine_seconds": required_tasks * observed_task_upper_seconds,
            "prior_attempts": (
                [
                    {
                        "task_id": prior_failed_task_id,
                        "status": "failed",
                        "shots": prior_failed_task_shots,
                        "circuit_profile": "1q_h_measure",
                        "counted_as_real_success": False,
                    }
                ]
                if prior_failed_task_id
                else []
            ),
            "estimated_formal_submission_cap": formal_submission_cap,
            "estimated_total_with_smoke": required_tasks,
            "shots_per_task": shots,
            "local_quota_remaining_before": remaining_before,
            "smoke": {
                "mode": "real_hardware",
                "task_id": str(smoke_id),
                "status": "completed",
                "shots": shots,
                "qubits": 1,
                "probability": smoke.get("result"),
                "elapsed_s": round(time.perf_counter() - started, 3),
            },
        },
    )


class MockRealClient:
    """确定性 Mock 真机客户端，用于评估阶段模拟真机反馈（不消耗真机机时）。

    Issue #108 修复：原 evaluate_one_episode 使用纯仿真环境评估，导致
    mixed_real 和 pure_real 条件的评估 reward 完全相同（因为真机反馈
    仅在训练时通过 _poll_pending_real_tasks 加入 reward，评估环境
    use_real_machine=False 使反馈为零）。此 Mock 客户端使评估环境
    与训练条件匹配，确保评估 reward 反映真机参与率差异。

    - submit_quantum_task : 立即返回 fake task_id 并缓存 completed 状态
    - get_task_status     : 返回缓存的 completed 状态
    - wait_for_task       : 立即返回缓存状态（评估不阻塞）
    """

    def __init__(self, machine_name: str = "tianyan176") -> None:
        self.machine_name = machine_name
        self._counter = 0
        self._cached_status: dict[str, dict[str, Any]] = {}

    def submit_quantum_task(self, **kwargs: Any) -> str | None:
        self._counter += 1
        task_id = f"mock_eval_{self._counter}"
        self._cached_status[task_id] = {
            "status": "completed",
            "result": {"0": 0.5, "1": 0.5},
            "execution_time_s": 0.001,
        }
        return task_id

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        return self._cached_status.get(
            str(task_id), {"status": "unknown"}
        )

    def wait_for_task(self, task_id: str, **kwargs: Any) -> dict[str, Any]:
        return self._cached_status.get(
            str(task_id), {"status": "completed", "result": {"0": 0.5, "1": 0.5}}
        )


def evaluate_one_episode(
    model: PPO,
    *,
    seed: int,
    tasks: int,
    condition: str = "simulation",
    machine: str = "tianyan176",
    real_probability: float = 0.0,
    shots: int = 8,
) -> dict[str, Any]:
    """在仿真环境评估最终策略；真机条件使用 Mock 客户端模拟真机反馈。

    Issue #108 修复：评估环境需与训练条件匹配。当条件为 mixed_real 或
    pure_real 时，使用 MockRealClient 模拟真机反馈（不消耗真机机时），
    确保评估 reward 反映真机参与率差异，而非因纯仿真评估导致三条件
    reward 完全相同。

    Args:
        model            : 训练完成的 PPO 模型
        seed             : 评估随机种子
        tasks            : 评估任务数
        condition        : 实验条件（simulation / mixed_real / pure_real）
        machine          : 机器名称（用于 machine_configs）
        real_probability : 真机提交概率（与训练一致）
        shots            : 真机 shots 数（与训练一致）
    """
    use_real = condition != "simulation" and real_probability > 0
    env = QuantumSchedulingEnv(
        max_steps=tasks,
        machine_configs=_machine_config(machine, is_real=use_real),
        real_submit_probability=real_probability if use_real else 0.0,
        use_real_machine=use_real,
        max_real_submissions=None,  # 评估不限制提交次数
        real_machine_shots=shots,
        seed=seed,
    )
    if use_real:
        env.attach_real_clients({machine: MockRealClient(machine_name=machine)})
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    done = False
    info: dict[str, Any] = {}
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(int(action))
        total_reward += float(reward)
        done = terminated or truncated
    env.close()
    return {
        "reward": total_reward,
        "completion_rate": completion_rate_from_info(info),
        "tasks": tasks,
    }


def train_seed(
    *,
    condition: str,
    seed: int,
    tasks: int,
    episode_horizon: int,
    machine: str,
    real_probability: float,
    real_submission_cap: int,
    shots: int,
    client: AuditedRealClient | None,
    model_dir: Path,
) -> dict[str, Any]:
    """训练单个 seed；总 timestep 即固定任务口径。"""
    use_real = client is not None and real_probability > 0
    env = QuantumSchedulingEnv(
        max_steps=episode_horizon,
        machine_configs=_machine_config(machine, is_real=use_real),
        real_submit_probability=real_probability if use_real else 0.0,
        use_real_machine=use_real,
        max_real_submissions=real_submission_cap if use_real else 0,
        real_machine_shots=shots,
        seed=seed,
    )
    if use_real and client is not None:
        env.attach_real_clients({machine: client})

    record_start = len(client.records) if client is not None else 0
    callback = TrainingCurveCallback()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=episode_horizon,
        batch_size=episode_horizon,
        gamma=0.99,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": [128, 64]},
    )
    started = time.perf_counter()
    model.learn(total_timesteps=tasks, callback=callback)
    elapsed_s = round(time.perf_counter() - started, 3)
    evaluation = evaluate_one_episode(
        model,
        seed=seed + 1000,
        tasks=tasks,
        condition=condition,
        machine=machine,
        real_probability=real_probability if use_real else 0.0,
        shots=shots,
    )

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"ppo_{condition}_seed_{seed}"
    model.save(str(model_path))
    real_stats = env.get_real_machine_stats()
    env.close()

    records: list[dict[str, Any]] = []
    if client is not None:
        records = [
            item | {"condition": condition, "seed": seed} for item in client.records[record_start:]
        ]
    accepted = sum(record.get("task_id") is not None for record in records)
    completed = sum(record.get("status") == "completed" for record in records)
    return {
        "seed": seed,
        "tasks": tasks,
        "episode_horizon": episode_horizon,
        "training_elapsed_s": elapsed_s,
        "curve": callback.curve,
        "evaluation": evaluation,
        "real_probability": real_probability if use_real else 0.0,
        "real_submission_cap": real_submission_cap if use_real else 0,
        "real_attempted": int(real_stats["submission_attempts_total"]),
        "real_accepted": accepted,
        "real_completed": completed,
        "real_failed_or_rejected": len(records) - completed,
        "degraded": bool(real_stats["degraded"]),
        "mock_calls": 0,
        "real_records": records,
        "model_path": str(model_path) + ".zip",
    }


def _aggregate_curve(runs: list[dict[str, Any]]) -> list[dict[str, float | int]]:
    common = min(len(run["curve"]) for run in runs)
    curve: list[dict[str, float | int]] = []
    for index in range(common):
        rewards = [float(run["curve"][index]["reward"]) for run in runs]
        curve.append(
            {
                "timestep": int(runs[0]["curve"][index]["timestep"]),
                "mean_reward": float(np.mean(rewards)),
                "std_reward": float(np.std(rewards)),
            }
        )
    return curve


def _convergence_timestep(curve: list[dict[str, Any]]) -> int | None:
    if not curve:
        return None
    values = np.asarray([float(point["mean_reward"]) for point in curve])
    tail = values[-min(3, len(values)) :]
    target = 0.9 * float(np.mean(tail))
    for point, value in zip(curve, values, strict=False):
        if value >= target:
            return int(point["timestep"])
    return None


def aggregate_condition(runs: list[dict[str, Any]], *, total_tasks: int) -> dict[str, Any]:
    """汇总三个 seed 的 reward、曲线和真机审计统计。"""
    rewards = [float(run["evaluation"]["reward"]) for run in runs]
    completions = [float(run["evaluation"]["completion_rate"]) for run in runs]
    attempted = sum(int(run["real_attempted"]) for run in runs)
    accepted = sum(int(run["real_accepted"]) for run in runs)
    completed = sum(int(run["real_completed"]) for run in runs)
    failed = sum(int(run["real_failed_or_rejected"]) for run in runs)
    curve = _aggregate_curve(runs)
    return {
        "runs": runs,
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "completion_rate_mean": float(np.mean(completions)),
        "training_elapsed_s": float(sum(run["training_elapsed_s"] for run in runs)),
        "real_attempted": attempted,
        "real_accepted": accepted,
        "real_completed": completed,
        "real_failed_or_rejected": failed,
        "real_participation_rate": accepted / total_tasks if total_tasks else 0.0,
        "real_degradation_rate": failed / attempted if attempted else 0.0,
        "degraded_seed_rate": sum(bool(run["degraded"]) for run in runs) / len(runs),
        "mock_calls": sum(int(run["mock_calls"]) for run in runs),
        "curve": curve,
        "convergence_timestep": _convergence_timestep(curve),
    }


def condition_is_valid(condition: str, stats: dict[str, Any]) -> bool:
    """判断条件是否具备可用于最终结论的完整数据。"""
    if condition == "simulation":
        return len(stats["runs"]) == 3
    if len(stats["runs"]) != 3:
        return False
    return all(not run["degraded"] and run["real_completed"] > 0 for run in stats["runs"])


def experiment_status(conditions: dict[str, dict[str, Any]]) -> str:
    """任一真机条件缺失或降级时，禁止标记为完整实验。"""
    complete = all(condition_is_valid(key, conditions[key]) for key in ("mixed_real", "pure_real"))
    return "completed" if complete else "partial_degraded"


def save_plot(conditions: dict[str, dict[str, Any]], path: Path) -> None:
    """保存三线收敛曲线和最终 reward ± std。"""
    pure_partial = conditions["pure_real"]["degraded_seed_rate"] > 0
    labels = {
        "simulation": "Simulation",
        "mixed_real": "Mixed real",
        "pure_real": (
            "Pure real (partial/degraded)" if pure_partial else "Pure real (eligible submissions)"
        ),
    }
    colors = {"simulation": "#4C78A8", "mixed_real": "#F58518", "pure_real": "#54A24B"}
    # 动态计算 seed 数量用于标题（#165=3, #192=10），避免硬编码
    sim_runs = conditions.get("simulation", {}).get("runs", [])
    num_seeds = len(sim_runs) if sim_runs else 3
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for key, stats in conditions.items():
        curve = stats["curve"]
        x = np.asarray([point["timestep"] for point in curve])
        mean = np.asarray([point["mean_reward"] for point in curve])
        std = np.asarray([point["std_reward"] for point in curve])
        axes[0].plot(x, mean, marker="o", label=labels[key], color=colors[key])
        axes[0].fill_between(x, mean - std, mean + std, alpha=0.15, color=colors[key])
    axes[0].set_title(f"PPO convergence across {num_seeds} seeds")
    axes[0].set_xlabel("Training tasks")
    axes[0].set_ylabel("Episode reward (20-task window)")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    keys = list(labels)
    means = [conditions[key]["reward_mean"] for key in keys]
    stds = [conditions[key]["reward_std"] for key in keys]
    axes[1].bar(
        [labels[key] for key in keys],
        means,
        yerr=stds,
        capsize=6,
        color=[colors[key] for key in keys],
    )
    axes[1].set_title("Final reward (200-task evaluation)")
    axes[1].set_ylabel("Reward mean ± std")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def generate_report(results: dict[str, Any], report_path: Path, plot_path: Path) -> None:
    """生成明确区分仿真、Mock、真实真机和降级的消融报告。"""
    conditions = results["conditions"]
    sim_conv = conditions["simulation"]["convergence_timestep"]
    rows = []
    for key, label in (
        ("simulation", "纯仿真"),
        ("mixed_real", "仿真+真机混合"),
        ("pure_real", "纯真机（符合提交资格的步骤）"),
    ):
        stats = conditions[key]
        valid = condition_is_valid(key, stats)
        conv = stats["convergence_timestep"]
        speedup = sim_conv / conv if sim_conv and conv else None
        rows.append(
            f"| {label} | {'有效' if valid else '部分/不可作完整结论'} | "
            f"{stats['reward_mean']:.2f} ± {stats['reward_std']:.2f} | "
            f"{stats['completion_rate_mean']:.2%} | {stats['training_elapsed_s']:.1f} | "
            f"{stats['real_attempted']}/{stats['real_accepted']}/{stats['real_completed']} | "
            f"{stats['real_participation_rate']:.2%} | {stats['real_degradation_rate']:.2%} | "
            f"{conv if conv is not None else 'N/A'} | "
            f"{f'{speedup:.3f}×' if speedup is not None else 'N/A'} |"
        )

    preflight = results["preflight"]
    smoke = preflight["smoke"]
    total_records = sum(
        len(run["real_records"])
        for key in ("mixed_real", "pure_real")
        for run in conditions[key]["runs"]
    )
    report = [
        "# Issue #165 真机消融实验",
        "",
        f"生成时间：{results['generated_at']}",
        f"实验状态：`{results['status']}`",
        "",
        "## 实验口径与安全门禁",
        "",
        f"- 三种条件均为 {len(results['config']['seeds'])} seeds；每 seed 固定 "
        f"{results['config']['tasks_per_seed']} 个训练任务，并在独立 200-task 仿真环境评估。",
        f"- 混合条件 real-prob={results['config']['mixed_real_probability']:.2f}；纯真机条件 "
        'real-prob=1.0。这里的\u201c纯真机\u201d指所有符合量子真机提交资格的调度步骤均尝试真机，'
        "不表示经典动作也被伪装成量子任务。",
        "- **评估方法（Issue #108 修复）**：原评估使用纯仿真环境（`use_real_machine=False`），"
        "导致 mixed_real 和 pure_real 条件的评估 reward 完全相同——因为真机反馈仅在训练时"
        "通过 `_poll_pending_real_tasks()` 加入 reward，评估环境无真机配置使反馈为零。"
        "修复后，真机条件的评估使用 `MockRealClient` 模拟真机反馈（确定性 completed 状态 + "
        "固定概率分布），不消耗真机机时，但使评估 reward 反映真机参与率差异。",
        f"- 本次断点补跑的正式 SDK 调用硬上限 {results['config']['formal_submission_cap']}；"
        f"加冒烟最坏 {preflight['estimated_total_with_smoke']} tasks，"
        f"每任务 {results['config']['shots']} shots。",
        "- SDK 无平台额度查询接口；用户确认仪表盘额度够用，本次开始前按截图与已有调用"
        f"保守折算剩余 {preflight['confirmed_machine_minutes']:.3f} 分钟、已用 "
        f"{preflight['confirmed_used_seconds']:.3f} 秒；按截图中单任务上界 "
        f"{preflight['observed_task_upper_seconds']:.3f} 秒估算，最坏约 "
        f"{preflight['estimated_max_machine_seconds']:.3f} 秒，并使用独立持久化本地硬预算。",
        f"- authenticate/list_backends 成功，物理后端 `{preflight['machine']}` 状态 "
        f"`{preflight['backend_status']}`；1-qubit 冒烟 `{smoke['task_id']}` "
        f"状态 `{smoke['status']}`，耗时 {smoke['elapsed_s']:.3f}s。",
        f"- 正式重跑前失败冒烟 {len(preflight['prior_attempts'])} 个；均仅作失败审计，"
        "不计作真机成功或正式训练调用。",
        "",
        "## 三条件结果",
        "",
        "| 条件 | 数据有效性 | reward ± std | 完成率 | 训练耗时(s) | 真机尝试/接受/完成 | 真机参与率 | 真机降级率 | 收敛任务数 | vs 仿真加速比 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
        "",
        "纯真机行保留 reward、曲线和降级率用于失败审计；若数据有效性标为“部分”，"
        "其 reward ± std 和加速比不得作为三 seed 纯真机结论。",
        "",
        "加速比仅表示本实验曲线达到末三点均值 90% 所需任务数之比；大于 1 才表示更早达到阈值，"
        "不得解读为量子硬件计算加速。三 seed 样本仍较小。",
        "",
        "## Mock、仿真、真机边界",
        "",
        "- 纯仿真条件没有 SDK 调用。",
        "- 混合/纯真机条件只把带真实 task ID 且状态 completed 的记录计作真机成功。",
        "- 提交拒绝、失败、超时以及降级后的仿真步骤均不计作真机成功；本实验训练阶段没有调用 Mock 客户端。",
        "- **评估阶段**（Issue #108 修复）：真机条件的评估使用 `MockRealClient` 模拟真机反馈，"
        "不消耗真机机时；Mock 客户端对所有提交返回确定性 completed 状态，反馈模式为 "
        "`status_only`（固定 bonus=2.0），与训练阶段的 `real_feedback_mode` 一致。",
        f"- 共保存 {total_records} 条正式 SDK 调用审计记录；完整 task ID、状态、概率和耗时见 "
        "`results/real_machine/issue165_ablation.json`，不含 API Key。",
        "",
        "## 三线图",
        "",
        f"![Issue #165 三条件收敛与 reward 对比](../real_machine/{plot_path.name})",
        "",
    ]
    preliminary = results.get("preliminary_attempt")
    if preliminary:
        report.extend(
            [
                "## 前次中断实验审计",
                "",
                "前次正式运行因后端短暂不可用触发连续三次拒绝和自动降级，因此不纳入上表的"
                "三 seed 最终统计，也不冒充完整纯真机实验。",
                f"该次共尝试 {preliminary['real_attempted']} 次、完成 "
                f"{preliminary['real_completed']} 次、失败/拒绝 "
                f"{preliminary['real_failed_or_rejected']} 次；完整 task ID 保存在 "
                f"`{preliminary['artifact']}`。",
                "",
            ]
        )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(report), encoding="utf-8")


def parse_seeds(value: str) -> list[int]:
    seeds = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise argparse.ArgumentTypeError("#165 正式实验必须提供 3 个不同 seed")
    return seeds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("42,43,44"))
    parser.add_argument("--tasks-per-seed", type=int, default=200)
    parser.add_argument("--episode-horizon", type=int, default=20)
    parser.add_argument("--mixed-real-prob", type=float, default=0.05)
    parser.add_argument("--mixed-cap-per-seed", type=int, default=10)
    parser.add_argument("--pure-cap-per-seed", type=int, default=200)
    parser.add_argument("--shots", type=int, default=8)
    parser.add_argument("--machine", choices=sorted(PHYSICAL_MACHINES), default="tianyan176")
    parser.add_argument("--confirmed-machine-minutes", type=float, required=True)
    parser.add_argument("--confirmed-used-seconds", type=float, required=True)
    parser.add_argument("--observed-task-upper-seconds", type=float, required=True)
    parser.add_argument("--prior-failed-task-id", type=str)
    parser.add_argument("--prior-failed-task-shots", type=int, default=8)
    parser.add_argument("--wait-timeout", type=int, default=120)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--quota-state", type=Path, default=DEFAULT_QUOTA_STATE)
    parser.add_argument("--quota-config", type=Path, default=DEFAULT_QUOTA_CONFIG)
    parser.add_argument("--preliminary-results", type=Path)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=_PROJECT_ROOT / "results" / "real_machine" / "issue165_checkpoint.json",
    )
    args = parser.parse_args()

    if args.tasks_per_seed != 200:
        raise ValueError("#165 正式实验固定 --tasks-per-seed=200")
    if args.episode_horizon <= 0 or args.tasks_per_seed % args.episode_horizon:
        raise ValueError("episode horizon 必须为 200 的正因数")
    if not math.isclose(args.mixed_real_prob, 0.05):
        raise ValueError("#165 混合条件固定 --mixed-real-prob=0.05")
    if not 1 <= args.shots <= 32:
        raise ValueError("shots 必须在 1–32")
    if args.mixed_cap_per_seed > args.tasks_per_seed * args.mixed_real_prob:
        raise ValueError("混合条件硬上限不得超过 5% 任务预算")
    if args.pure_cap_per_seed > args.tasks_per_seed:
        raise ValueError("纯真机单 seed 硬上限不得超过 200")

    resume_data: dict[str, Any] = {}
    if args.resume_from:
        resume_data = json.loads(args.resume_from.read_text(encoding="utf-8"))
    if "runs_by_condition" in resume_data:
        runs_by_condition = resume_data["runs_by_condition"]
    else:
        runs_by_condition = {
            key: list(stats["runs"]) for key, stats in resume_data.get("conditions", {}).items()
        }
    for key in ("simulation", "mixed_real", "pure_real"):
        runs_by_condition.setdefault(key, [])

    def reusable(condition: str, run: dict[str, Any]) -> bool:
        if condition == "simulation":
            return True
        return not run["degraded"] and run["real_completed"] > 0

    pending_real_seeds = {
        condition: [
            seed
            for seed in args.seeds
            if not any(
                run["seed"] == seed and reusable(condition, run)
                for run in runs_by_condition[condition]
            )
        ]
        for condition in ("mixed_real", "pure_real")
    }
    formal_cap = (
        len(pending_real_seeds["mixed_real"]) * args.mixed_cap_per_seed
        + len(pending_real_seeds["pure_real"]) * args.pure_cap_per_seed
    )
    client, quota, preflight = run_preflight(
        machine=args.machine,
        shots=args.shots,
        formal_submission_cap=formal_cap,
        confirmed_machine_minutes=args.confirmed_machine_minutes,
        confirmed_used_seconds=args.confirmed_used_seconds,
        observed_task_upper_seconds=args.observed_task_upper_seconds,
        prior_failed_task_id=args.prior_failed_task_id,
        prior_failed_task_shots=args.prior_failed_task_shots,
        wait_timeout=args.wait_timeout,
        quota_config_path=args.quota_config,
        quota_state_path=args.quota_state,
    )
    audited = AuditedRealClient(client, wait_timeout=args.wait_timeout, poll_interval=2)

    condition_specs = {
        "simulation": (0.0, 0, None),
        "mixed_real": (args.mixed_real_prob, args.mixed_cap_per_seed, audited),
        "pure_real": (1.0, args.pure_cap_per_seed, audited),
    }
    total_tasks = len(args.seeds) * args.tasks_per_seed
    aborted = False
    for condition, (probability, cap, condition_client) in condition_specs.items():
        previous = {run["seed"]: run for run in runs_by_condition[condition]}
        refreshed: list[dict[str, Any]] = []
        for seed in args.seeds:
            old_run = previous.get(seed)
            if old_run is not None and reusable(condition, old_run):
                refreshed.append(old_run)
                continue
            if aborted and old_run is not None:
                refreshed.append(old_run)
                continue
            if aborted:
                continue
            run = train_seed(
                condition=condition,
                seed=seed,
                tasks=args.tasks_per_seed,
                episode_horizon=args.episode_horizon,
                machine=args.machine,
                real_probability=probability,
                real_submission_cap=cap,
                shots=args.shots,
                client=condition_client,
                model_dir=args.model_dir,
            )
            refreshed.append(run)
            runs_by_condition[condition] = refreshed + [
                item for old_seed, item in previous.items() if old_seed > seed
            ]
            args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
            args.checkpoint.write_text(
                json.dumps(
                    {
                        "issue": 165,
                        "generated_at": datetime.now().astimezone().isoformat(),
                        "resume_source": str(args.resume_from) if args.resume_from else None,
                        "runs_by_condition": runs_by_condition,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            if condition != "simulation" and run["degraded"]:
                aborted = True
        runs_by_condition[condition] = refreshed

    conditions = {
        key: aggregate_condition(runs_by_condition[key], total_tasks=total_tasks)
        for key in ("simulation", "mixed_real", "pure_real")
    }

    results = {
        "issue": 165,
        "status": experiment_status(conditions),
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": {
            "seeds": args.seeds,
            "tasks_per_seed": args.tasks_per_seed,
            "episode_horizon": args.episode_horizon,
            "mixed_real_probability": args.mixed_real_prob,
            "mixed_cap_per_seed": args.mixed_cap_per_seed,
            "pure_cap_per_seed": args.pure_cap_per_seed,
            "formal_submission_cap": formal_cap,
            "shots": args.shots,
            "machine": args.machine,
        },
        "preflight": preflight,
        "quota_remaining_after": quota.remaining(),
        "conditions": conditions,
    }
    if args.preliminary_results:
        preliminary_data = json.loads(args.preliminary_results.read_text(encoding="utf-8"))
        preliminary_conditions = preliminary_data["conditions"]
        results["preliminary_attempt"] = {
            "artifact": str(args.preliminary_results.resolve().relative_to(_PROJECT_ROOT)).replace(
                "\\", "/"
            ),
            "status": (
                "partial_degraded"
                if any(
                    run["degraded"]
                    for key in ("mixed_real", "pure_real")
                    for run in preliminary_conditions[key]["runs"]
                )
                else preliminary_data.get("status", "unknown")
            ),
            "real_attempted": sum(
                preliminary_conditions[key]["real_attempted"] for key in ("mixed_real", "pure_real")
            ),
            "real_completed": sum(
                preliminary_conditions[key]["real_completed"] for key in ("mixed_real", "pure_real")
            ),
            "real_failed_or_rejected": sum(
                preliminary_conditions[key]["real_failed_or_rejected"]
                for key in ("mixed_real", "pure_real")
            ),
        }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plot(conditions, args.plot)
    generate_report(results, args.report, args.plot)
    print(f"结果: {args.results}")
    print(f"报告: {args.report}")
    print(f"图: {args.plot}")


if __name__ == "__main__":
    main()
