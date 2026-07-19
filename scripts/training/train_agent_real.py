"""Issue #164：纯仿真 PPO 与仿真+真机混合 PPO 的闭环对比实验。

正式模式严格执行 dotenv -> authenticate -> list_backends/本地额度检查 ->
1-qubit QCIS 冒烟，再开始两个 10,000 timestep、200 步口径的 PPO 实验。
真机结果会在训练继续前完成轮询，因此成功/失败反馈确实进入下一步 reward。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.api.quota_tracker import QuotaTracker
from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.env import QuantumSchedulingEnv

DEFAULT_RESULTS = _PROJECT_ROOT / "results" / "real_machine" / "issue164_closed_loop.json"
DEFAULT_REPORT = _PROJECT_ROOT / "results" / "reports" / "real_machine_closed_loop.md"
DEFAULT_PLOT = (
    _PROJECT_ROOT / "results" / "real_machine" / "real_machine_closed_loop_convergence.png"
)
DEFAULT_MODEL_DIR = _PROJECT_ROOT / "models" / "issue164"
DEFAULT_QUOTA_STATE = _PROJECT_ROOT / "results" / "real_machine" / "issue164_quota_state.json"
PHYSICAL_HARDWARE_MACHINES = frozenset({"tianyan176", "tianyan176-2"})
TIANYAN176_ALLOCATED_QUBITS = 66


class PreflightError(RuntimeError):
    """真机预检失败，并携带可安全落盘的审计信息。"""

    def __init__(self, message: str, details: dict[str, Any]) -> None:
        super().__init__(message)
        self.details = details


class TrainingCurveCallback(BaseCallback):
    """按 200 步 episode 记录训练 reward 曲线。"""

    def __init__(self) -> None:
        super().__init__(verbose=0)
        self.curve: list[dict[str, float | int]] = []
        self._episode_reward = 0.0

    def _on_step(self) -> bool:
        self._episode_reward += float(self.locals["rewards"][0])
        if bool(self.locals["dones"][0]):
            self.curve.append(
                {
                    "timestep": int(self.num_timesteps),
                    "reward": round(self._episode_reward, 6),
                }
            )
            self._episode_reward = 0.0
        return True


@dataclass
class AuditedRealClient:
    """等待真机结果并保存可审计任务 ID；不会保存 API Key。"""

    delegate: CqlibTianyanClient
    wait_timeout: int = 180
    poll_interval: int = 3
    fixed_qcis: str = "H Q0\nM Q0"
    records: list[dict[str, Any]] = field(default_factory=list)
    _cached_status: dict[str, dict[str, Any]] = field(default_factory=dict)

    def submit_quantum_task(self, **kwargs: Any) -> str | None:
        started = time.perf_counter()
        # 天衍-176 免费/稳定路径固定使用已通过冒烟验证的 1-qubit H+测量线路。
        # 环境生成的随机 RY/RZ 等参数门曾被平台接受后运行失败，不用于正式闭环。
        kwargs["qcis"] = self.fixed_qcis
        task_id = self.delegate.submit_quantum_task(**kwargs)
        record: dict[str, Any] = {
            "task_id": str(task_id) if task_id is not None else None,
            "task_name": str(kwargs.get("task_name", "")),
            "shots": int(kwargs.get("shots", 0)),
            "machine": self.delegate.machine_name,
            "circuit_profile": "1q_h_measure",
            "status": "submit_rejected" if task_id is None else "submitted",
            "elapsed_s": 0.0,
        }
        if task_id is not None:
            status = self.delegate.wait_for_task(
                str(task_id), timeout=self.wait_timeout, poll_interval=self.poll_interval
            )
            record["status"] = str(status.get("status", "unknown"))
            record["probability"] = status.get("result")
            self._cached_status[str(task_id)] = status
        record["elapsed_s"] = round(time.perf_counter() - started, 3)
        self.records.append(record)
        return str(task_id) if task_id is not None else None

    def get_task_status(self, task_id: str) -> dict[str, Any]:
        status = self._cached_status.get(str(task_id))
        if status is None:
            return self.delegate.get_task_status(str(task_id))
        if status.get("status") in {"timeout", "failed"}:
            return {**status, "status": "error"}
        return status


def _machine_config(machine: str, *, is_real: bool) -> list[dict[str, Any]]:
    return [
        {
            "name": machine,
            "total_qubits": TIANYAN176_ALLOCATED_QUBITS,
            "supported_gates": ("H", "X", "Y", "Z", "RX", "RY", "RZ", "M"),
            "is_real": is_real,
        }
    ]


def run_preflight(
    machine: str,
    shots: int,
    max_real_submissions: int,
    wait_timeout: int,
    quota_state_path: Path,
) -> tuple[CqlibTianyanClient, QuotaTracker, dict[str, Any]]:
    """执行不泄密预检和一个 1-qubit 真机冒烟任务。"""
    if machine not in PHYSICAL_HARDWARE_MACHINES:
        raise RuntimeError(
            f"{machine} 不是本实验允许的物理真机；tianyan_s/sw/tn/tnn/sa 等均为模拟器，禁止计作真机"
        )
    load_dotenv(_PROJECT_ROOT / ".env")
    api_key = os.getenv("TIANYAN_API_KEY", "")
    if not api_key:
        raise RuntimeError("TIANYAN_API_KEY 未配置")

    quota_state_path.parent.mkdir(parents=True, exist_ok=True)
    quota = QuotaTracker(
        config_path=str(_PROJECT_ROOT / "config" / "quota.yaml"),
        state_path=str(quota_state_path),
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
    selected = next((item for item in backends if item.get("name") == machine), None)
    if selected is None:
        raise RuntimeError(f"list_backends 未返回指定机器 {machine}")
    if selected.get("status") != "running":
        raise RuntimeError(f"机器 {machine} 当前状态不是 running: {selected.get('status')}")

    required_tasks = max_real_submissions + 1
    required_shots = required_tasks * shots
    remaining_before = quota.remaining()
    if not quota.can_consume(shots=required_shots, tasks=required_tasks):
        raise RuntimeError(
            "本地配额账本不足以覆盖冒烟和硬上限："
            f"需要 {required_tasks} tasks/{required_shots} shots，"
            f"剩余 {remaining_before}"
        )

    started = time.perf_counter()
    smoke_id = client.submit_quantum_task(
        qcis="H Q0\nM Q0",
        shots=shots,
        task_name=f"issue164_smoke_{datetime.now():%Y%m%d_%H%M%S}",
    )
    if smoke_id is None:
        raise PreflightError(
            "1-qubit 冒烟任务提交被拒绝：平台返回剩余机时不足",
            {
                "authenticate": True,
                "machine": machine,
                "backend_status": selected.get("status"),
                "backend_type": selected.get("type"),
                "platform_quota_api_available": False,
                "local_quota_remaining_before": remaining_before,
                "local_quota_remaining_after_smoke": quota.remaining(),
                "smoke": {
                    "mode": "real_hardware",
                    "task_id": None,
                    "qubits": 1,
                    "shots": shots,
                    "status": "submit_rejected",
                    "reason": "platform_remaining_machine_time_insufficient",
                    "elapsed_s": round(time.perf_counter() - started, 3),
                },
            },
        )
    smoke_status = client.wait_for_task(str(smoke_id), timeout=wait_timeout, poll_interval=3)
    smoke_elapsed = round(time.perf_counter() - started, 3)
    if smoke_status.get("status") != "completed":
        raise RuntimeError(f"1-qubit 冒烟任务未成功: {smoke_status.get('status')}")

    preflight = {
        "authenticate": True,
        "machine": machine,
        "backend_status": selected.get("status"),
        "backend_type": selected.get("type"),
        "platform_quota_api_available": False,
        "local_quota_remaining_before": remaining_before,
        "local_quota_remaining_after_smoke": quota.remaining(),
        "smoke": {
            "mode": "real_hardware",
            "task_id": str(smoke_id),
            "qubits": 1,
            "shots": shots,
            "status": "completed",
            "probability": smoke_status.get("result"),
            "elapsed_s": smoke_elapsed,
        },
    }
    return client, quota, preflight


def train_condition(
    label: str,
    timesteps: int,
    seed: int,
    machine: str,
    real_probability: float,
    max_real_submissions: int,
    shots: int,
    audited_client: AuditedRealClient | None,
    model_dir: Path,
) -> tuple[PPO, dict[str, Any]]:
    """训练一个 PPO 条件并返回曲线和真机审计统计。"""
    use_real = audited_client is not None and real_probability > 0.0
    env = QuantumSchedulingEnv(
        max_steps=200,
        machine_configs=_machine_config(machine, is_real=use_real),
        real_submit_probability=real_probability if use_real else 0.0,
        use_real_machine=use_real,
        real_machine_feedback_weight=1.0,
        max_real_submissions=max_real_submissions if use_real else 0,
        real_machine_shots=shots,
        seed=seed,
    )
    if use_real and audited_client is not None:
        env.attach_real_clients({machine: audited_client})

    callback = TrainingCurveCallback()
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=200,
        batch_size=50,
        gamma=0.99,
        seed=seed,
        verbose=0,
        policy_kwargs={"net_arch": [128, 64]},
    )
    started = time.perf_counter()
    model.learn(total_timesteps=timesteps, callback=callback)
    elapsed = round(time.perf_counter() - started, 3)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / f"ppo_{label}_seed_{seed}"
    model.save(str(model_path))

    real_stats = env.get_real_machine_stats()
    records = list(audited_client.records) if audited_client is not None else []
    accepted = sum(1 for item in records if item["task_id"] is not None)
    completed = sum(1 for item in records if item["status"] == "completed")
    failed = len(records) - completed
    stats = {
        "label": label,
        "mode": "simulation+real_hardware" if use_real else "simulation",
        "timesteps_requested": timesteps,
        "timesteps_actual": int(model.num_timesteps),
        "episode_horizon_tasks": 200,
        "elapsed_s": elapsed,
        "real_probability": real_probability if use_real else 0.0,
        "max_real_submissions": max_real_submissions if use_real else 0,
        "real_submission_attempts": int(real_stats["submission_attempts_total"]),
        "real_accepted": accepted,
        "real_completed": completed,
        "real_failed_or_rejected": failed,
        "real_degraded": bool(real_stats["degraded"]),
        "real_participation_rate_per_200_tasks": accepted / 200.0,
        "curve": callback.curve,
        "model_path": str(model_path) + ".zip",
        "real_records": records,
    }
    return model, stats


def evaluate_model(model: PPO, seed: int, episodes: int = 5) -> dict[str, Any]:
    """在纯仿真环境按每 episode 200 任务公平评估。"""
    env = QuantumSchedulingEnv(max_steps=200, seed=seed)
    rewards: list[float] = []
    completion_rates: list[float] = []
    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        total = 0.0
        done = False
        info: dict[str, Any] = {}
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            total += float(reward)
            done = terminated or truncated
        rewards.append(total)
        completion_rates.append(completion_rate_from_info(info))
    return {
        "episodes": episodes,
        "tasks_per_episode": 200,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_completion_rate": float(np.mean(completion_rates)),
        "episode_rewards": rewards,
    }


def completion_rate_from_info(info: dict[str, Any]) -> float:
    """根据环境实际提供的成功计数计算任务完成率。"""
    successes = sum(
        int(info.get(key, 0)) for key in ("quantum_success", "classical_success", "hybrid_success")
    )
    total_scheduled = int(info.get("total_scheduled", 0))
    return successes / total_scheduled if total_scheduled else 0.0


def convergence_timestep(curve: list[dict[str, Any]]) -> int | None:
    """返回首次达到末五集平均 reward 90% 的 timestep。"""
    if not curve:
        return None
    rewards = np.asarray([float(point["reward"]) for point in curve], dtype=float)
    target = 0.9 * float(np.mean(rewards[-5:]))
    smooth = np.convolve(
        rewards, np.ones(min(5, len(rewards))) / min(5, len(rewards)), mode="valid"
    )
    for index, value in enumerate(smooth):
        if value >= target:
            return int(curve[index + min(5, len(rewards)) - 1]["timestep"])
    return None


def save_plot(sim_stats: dict[str, Any], mixed_stats: dict[str, Any], path: Path) -> None:
    """保存两条训练收敛曲线。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    for stats, label in ((sim_stats, "Simulation PPO"), (mixed_stats, "Mixed real PPO")):
        x = [item["timestep"] for item in stats["curve"]]
        y = [item["reward"] for item in stats["curve"]]
        plt.plot(x, y, alpha=0.35)
        if len(y) >= 5:
            smooth = np.convolve(y, np.ones(5) / 5, mode="valid")
            plt.plot(x[4:], smooth, linewidth=2, label=f"{label} (5-episode mean)")
    plt.xlabel("Training timestep")
    plt.ylabel("Episode reward (200 tasks)")
    plt.title("Issue #164: PPO closed-loop convergence")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def generate_report(results: dict[str, Any], report_path: Path, plot_path: Path) -> None:
    """生成明确区分仿真、真实真机、失败/降级的报告。"""
    sim = results["conditions"]["simulation"]
    mixed = results["conditions"]["mixed_real"]
    sim_eval = sim["evaluation"]
    mixed_eval = mixed["evaluation"]
    delta = mixed_eval["mean_reward"] - sim_eval["mean_reward"]
    sim_conv = convergence_timestep(sim["training"]["curve"])
    mixed_conv = convergence_timestep(mixed["training"]["curve"])
    speedup = (sim_conv / mixed_conv) if sim_conv and mixed_conv else None
    smoke = results["preflight"]["smoke"]
    records = mixed["training"]["real_records"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    rel_plot = Path(os.path.relpath(plot_path, report_path.parent)).as_posix()
    lines = [
        "# Issue #164 真机闭环训练报告",
        "",
        f"生成时间：{results['generated_at']}",
        "",
        "## 实验口径",
        "",
        "- PPO 训练步数：10,000；episode 固定 200 个调度步骤。",
        "- 混合条件真机抽样概率："
        f"{mixed['training']['real_probability']:.2%}；硬上限："
        f"{mixed['training']['max_real_submissions']} 次 SDK 提交调用。",
        f"- 真机电路：1-qubit QCIS；每任务 {results['config']['shots']} shots。",
        "- 最终模型统一在纯仿真环境评估 5 episodes × 200 tasks，避免评估阶段额外消耗额度。",
        "- Mock、仿真和真实真机严格分开；本实验未把失败或降级计作真机成功。",
        "",
        "## 正式运行前检查",
        "",
        f"- dotenv 加载、authenticate：成功；后端 `{results['preflight']['machine']}` 状态："
        f"`{results['preflight']['backend_status']}`。",
        "- cqlib 未暴露平台权威余额接口；额度检查采用仓库 QuotaTracker 本地账本，"
        "并已在 Issue #164 @xiabai2004 说明。",
        f"- 最小冒烟：真实真机任务 `{smoke['task_id']}`，1 qubit / {smoke['shots']} shots，"
        f"状态 `{smoke['status']}`，耗时 {smoke['elapsed_s']:.3f}s。",
        "",
        "## 结果",
        "",
        "| 条件 | 模式 | 训练耗时(s) | 最终 reward | 完成率 | 真机接受/完成/失败 | 参与率(每200任务口径) | 降级 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
        f"| 纯仿真 PPO | Simulation | {sim['training']['elapsed_s']:.3f} | "
        f"{sim_eval['mean_reward']:.2f} ± {sim_eval['std_reward']:.2f} | "
        f"{sim_eval['mean_completion_rate']:.2%} | 0/0/0 | 0.00% | 否 |",
        f"| 仿真+真机 PPO | Simulation + real hardware | {mixed['training']['elapsed_s']:.3f} | "
        f"{mixed_eval['mean_reward']:.2f} ± {mixed_eval['std_reward']:.2f} | "
        f"{mixed_eval['mean_completion_rate']:.2%} | "
        f"{mixed['training']['real_accepted']}/{mixed['training']['real_completed']}/"
        f"{mixed['training']['real_failed_or_rejected']} | "
        f"{mixed['training']['real_participation_rate_per_200_tasks']:.2%} | "
        f"{'是' if mixed['training']['real_degraded'] else '否'} |",
        "",
        f"最终 reward 差值（混合 - 仿真）：{delta:+.2f}。",
        f"收敛 timestep（末五集均值 90% 阈值）：仿真 {sim_conv}，混合 {mixed_conv}；"
        + (f"收敛加速比 {speedup:.3f}×。" if speedup is not None else "无法计算加速比。"),
        "",
        "加速比小于 1 表示没有获得收敛步数加速。混合训练墙钟时间包含真机排队、"
        "提交和结果轮询开销。单 seed 与高 reward 方差意味着本结果不能用于宣称统计显著或量子加速。",
        "",
        "## 收敛曲线",
        "",
        f"![PPO 真机闭环收敛曲线]({rel_plot})",
        "",
        "## 真机任务审计",
        "",
        "| task_id | 状态 | shots | 机器 | 测量概率 | 耗时(s) |",
        "|---|---|---:|---|---|---:|",
    ]
    for record in records:
        lines.append(
            f"| {record['task_id'] or '—'} | {record['status']} | {record['shots']} | "
            f"{record['machine']} | `{record.get('probability')}` | "
            f"{record['elapsed_s']:.3f} |"
        )
    if not records:
        lines.append("| — | 无真实提交 | 0 | — | — | 0 |")
    lines.extend(
        [
            "",
            "原始结构化数据见 `results/real_machine/issue164_closed_loop.json`；其中不含 API Key。",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_blocked_report(results: dict[str, Any], report_path: Path) -> None:
    """预检失败时保存报告，避免误生成训练数字或收敛图。"""
    smoke = results["preflight"]["smoke"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        "\n".join(
            [
                "# Issue #164 真机闭环训练报告",
                "",
                f"生成时间：{results['generated_at']}",
                "",
                "## 状态：真机预检阻塞，正式训练未执行",
                "",
                "- dotenv 已加载，authenticate 成功。",
                f"- list_backends 返回 `{results['preflight']['machine']}` 状态为 "
                f"`{results['preflight']['backend_status']}`。",
                "- cqlib 未暴露平台权威余额接口；仓库本地额度账本不能代表平台余额。",
                f"- 1-qubit QCIS / {smoke['shots']} shots 冒烟提交被平台拒绝："
                "剩余机时不足；没有 task ID。",
                "- 已在 Issue #164 @xiabai2004 请求补充/确认机时额度。",
                "",
                "## 实验结果",
                "",
                "纯仿真 PPO 与仿真+真机混合 PPO 的 10,000-step 正式对比未开始，"
                "因此本报告不提供 reward、参与率、收敛曲线或真机成功数，也没有生成对比图。",
                "失败提交没有被记录为真机成功、Mock 成功或降级成功。",
                "",
                "实现已设置训练真机提交硬上限 8 次；加冒烟总计最多 9 次、576 shots。",
                "恢复额度后可用同一命令重跑并覆盖本报告为正式结果。",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #164 PPO 真机闭环训练")
    parser.add_argument("--timesteps", type=int, default=10000)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--real-prob", type=float, default=0.04)
    parser.add_argument("--max-real-submissions", type=int, default=8)
    parser.add_argument("--shots", type=int, default=32)
    parser.add_argument(
        "--machine", choices=sorted(PHYSICAL_HARDWARE_MACHINES), default="tianyan176"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--wait-timeout", type=int, default=180)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--plot", type=Path, default=DEFAULT_PLOT)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--quota-state", type=Path, default=DEFAULT_QUOTA_STATE)
    args = parser.parse_args()

    if args.timesteps != 10000:
        raise ValueError("Issue #164 正式实验要求 --timesteps=10000")
    if args.episodes != 5:
        raise ValueError("Issue #164 正式实验要求 --episodes=5")
    if not 0.03 <= args.real_prob <= 0.05:
        raise ValueError("--real-prob 必须在 0.03–0.05")
    if not 1 <= args.max_real_submissions <= 10:
        raise ValueError("--max-real-submissions 必须在 1–10，防止意外大量提交")
    if not 1 <= args.shots <= 128:
        raise ValueError("--shots 必须在 1–128")

    try:
        client, quota, preflight = run_preflight(
            machine=args.machine,
            shots=args.shots,
            max_real_submissions=args.max_real_submissions,
            wait_timeout=args.wait_timeout,
            quota_state_path=args.quota_state,
        )
    except PreflightError as exc:
        blocked_results = {
            "issue": 164,
            "status": "blocked_preflight",
            "generated_at": datetime.now().astimezone().isoformat(),
            "config": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "preflight": exc.details,
            "conditions": {},
        }
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(
            json.dumps(blocked_results, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        save_blocked_report(blocked_results, args.report)
        print(f"预检阻塞: {exc}")
        print(f"阻塞报告: {args.report}")
        raise SystemExit(2) from exc
    audited = AuditedRealClient(client, wait_timeout=args.wait_timeout)

    sim_model, sim_training = train_condition(
        "simulation",
        args.timesteps,
        args.seed,
        args.machine,
        0.0,
        0,
        args.shots,
        None,
        args.model_dir,
    )
    mixed_model, mixed_training = train_condition(
        "mixed_real",
        args.timesteps,
        args.seed,
        args.machine,
        args.real_prob,
        args.max_real_submissions,
        args.shots,
        audited,
        args.model_dir,
    )
    sim_eval = evaluate_model(sim_model, seed=args.seed + 1000, episodes=args.episodes)
    mixed_eval = evaluate_model(mixed_model, seed=args.seed + 1000, episodes=args.episodes)
    results = {
        "issue": 164,
        "generated_at": datetime.now().astimezone().isoformat(),
        "config": vars(args) | {"shots": args.shots},
        "preflight": preflight,
        "quota_remaining_after": quota.remaining(),
        "conditions": {
            "simulation": {"training": sim_training, "evaluation": sim_eval},
            "mixed_real": {"training": mixed_training, "evaluation": mixed_eval},
        },
    }
    results["config"] = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in results["config"].items()
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    save_plot(sim_training, mixed_training, args.plot)
    generate_report(results, args.report, args.plot)
    print(f"结果: {args.results}")
    print(f"报告: {args.report}")
    print(f"曲线: {args.plot}")


if __name__ == "__main__":
    main()
