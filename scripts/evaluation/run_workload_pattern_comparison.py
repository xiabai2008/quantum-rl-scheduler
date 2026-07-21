#!/usr/bin/env python
"""Issue #160：三种负载模式下的八策略多 Seed 公平对比。"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "evaluation"))

from run_issue_38_67_experiments import (
    BaseStrategy,
    Obs10Wrapper,
    build_strategies,
)

from src.scheduler.env import QuantumSchedulingEnv

ArrivalSchedule = float | Callable[[int, int], float]


@dataclass(frozen=True)
class WorkloadPattern:
    """一组可直接注入调度环境的负载参数。"""

    key: str
    label: str
    arrival_lambda: ArrivalSchedule
    arrival_description: str
    quantum_task_ratio: float


def tidal_arrival_lambda(step: int, max_steps: int) -> float:
    """返回 [0.1, 1.0] 内周期变化的潮汐到达率。"""
    period = max(max_steps, 1)
    return 0.55 + 0.45 * math.cos(2.0 * math.pi * step / period)


WORKLOAD_PATTERNS: tuple[WorkloadPattern, ...] = (
    WorkloadPattern("default", "默认", 0.5, "固定 λ=0.5", 0.7),
    WorkloadPattern("sparse", "稀疏", 0.2, "固定 λ=0.2", 0.3),
    WorkloadPattern("tidal", "潮汐", tidal_arrival_lambda, "周期 λ=1.0→0.1→1.0", 0.7),
)


def make_workload_env(pattern: WorkloadPattern, max_steps: int) -> Obs10Wrapper:
    """创建真正注入负载参数的 10 维公平评估环境。"""
    base_env = QuantumSchedulingEnv(
        max_steps=max_steps,
        max_qubits=287,
        arrival_lambda=pattern.arrival_lambda,
        quantum_task_ratio=pattern.quantum_task_ratio,
    )
    return Obs10Wrapper(base_env)


def _evaluate_strategy(
    pattern: WorkloadPattern,
    strategy: BaseStrategy,
    seed_list: Sequence[int],
    episodes_per_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """运行一个策略并汇总逐 episode 奖励与成功调度率。"""
    rewards: list[float] = []
    completed = 0
    attempts = 0

    for seed in seed_list:
        for episode in range(episodes_per_seed):
            env = make_workload_env(pattern, max_steps)
            try:
                obs, info = env.reset(seed=seed + episode)
                episode_reward = 0.0
                for _ in range(max_steps):
                    action = strategy.select_action(obs)
                    obs, reward, terminated, truncated, info = env.step(action)
                    episode_reward += float(reward)
                    if terminated or truncated:
                        break

                scheduled = int(info.get("total_scheduled", 0))
                mismatches = int(info.get("mismatch_count", 0))
                completed += scheduled
                attempts += scheduled + mismatches
                rewards.append(episode_reward)
            finally:
                env.close()

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "completion_rate": completed / attempts if attempts else 0.0,
        "completed_schedules": completed,
        "scheduling_attempts": attempts,
        "episode_rewards": rewards,
    }


def run_comparison(
    seeds: int = 10,
    episodes_per_seed: int = 5,
    max_steps: int = 200,
    ppo_model: str = "deliverable_models/ppo_best_model_14dim.zip",
    dqn_model: str = "deliverable_models/dqn_best_model_10dim.zip",
    patterns: Sequence[WorkloadPattern] = WORKLOAD_PATTERNS,
) -> dict[str, Any]:
    """执行 3×8 对比并返回可复现的完整结果。"""
    if seeds <= 0 or episodes_per_seed <= 0 or max_steps <= 0:
        raise ValueError("seeds, episodes_per_seed and max_steps must be positive")

    seed_list = [42 + index * 137 for index in range(seeds)]
    strategies = build_strategies(dqn_path=dqn_model, ppo_path=ppo_model)
    if len(strategies) != 8:
        raise RuntimeError(f"expected 8 strategies, loaded {len(strategies)}")

    started = time.perf_counter()
    pattern_results: dict[str, Any] = {}
    for pattern in patterns:
        print(
            f"\n[{pattern.label}] {pattern.arrival_description}, 量子占比={pattern.quantum_task_ratio:.0%}"
        )
        strategy_results: dict[str, Any] = {}
        for strategy in strategies:
            strategy_started = time.perf_counter()
            metrics = _evaluate_strategy(
                pattern,
                strategy,
                seed_list,
                episodes_per_seed,
                max_steps,
            )
            metrics["elapsed_seconds"] = time.perf_counter() - strategy_started
            strategy_results[strategy.name] = metrics
            print(
                f"  {strategy.name:<16} reward={metrics['mean_reward']:>9.2f} "
                f"completion={metrics['completion_rate']:.1%}"
            )

        pattern_results[pattern.key] = {
            "label": pattern.label,
            "arrival": pattern.arrival_description,
            "quantum_task_ratio": pattern.quantum_task_ratio,
            "strategies": strategy_results,
        }

    return {
        "config": {
            "seed_list": seed_list,
            "episodes_per_seed": episodes_per_seed,
            "max_steps": max_steps,
            "total_episodes_per_strategy_pattern": seeds * episodes_per_seed,
            "observation_dim": 10,
            "wrapper": "Obs10Wrapper",
            "ppo_model": ppo_model,
            "dqn_model": dqn_model,
            "completion_rate_definition": "total_scheduled / (total_scheduled + mismatch_count)",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "patterns": pattern_results,
    }


def _improvement(ppo_reward: float, fcfs_reward: float) -> float:
    """计算 PPO 相对 FCFS 的有符号百分比变化。"""
    if fcfs_reward == 0.0:
        return math.nan
    return (ppo_reward - fcfs_reward) / abs(fcfs_reward) * 100.0


def generate_report(results: dict[str, Any], report_path: Path, data_path: Path) -> None:
    """生成包含 3 模式 × 8 策略 reward 和完成率的报告。"""
    config = results["config"]
    lines = [
        "# 多负载模式 × 8 策略对比报告",
        "",
        "> **执行类型：纯仿真。** 本实验未调用 Mock API 或量子真机。负载参数直接注入",
        "> `QuantumSchedulingEnv` 的任务到达动力学；未使用旧 `SimulationTaskGenerator` 伪注入。",
        "",
        "## 实验配置",
        "",
        f"- 随机种子：`{config['seed_list']}`",
        f"- 重复次数：{len(config['seed_list'])} seeds × {config['episodes_per_seed']} episodes "
        f"= {config['total_episodes_per_strategy_pattern']} episodes/策略/模式",
        f"- 每个 episode：{config['max_steps']} 步（固定任务口径）",
        "- 观测：10 维 `Obs10Wrapper`",
        f"- PPO：`{config['ppo_model']}`",
        f"- DQN：`{config['dqn_model']}`",
        "- 完成率：成功调度数 /（成功调度数 + 资源不兼容次数）",
        "- 默认：λ=0.5，量子占比 70%；稀疏：λ=0.2，量子占比 30%；",
        "  潮汐：单 episode 内 λ=1.0→0.1→1.0 周期变化，量子占比 70%",
        "",
        "## 3 模式 × 8 策略结果",
        "",
        "| 负载模式 | 策略 | Reward（mean ± std） | 完成率 |",
        "|:--|:--|--:|--:|",
    ]

    for pattern in results["patterns"].values():
        for strategy_name, metrics in pattern["strategies"].items():
            lines.append(
                f"| {pattern['label']} | {strategy_name} | "
                f"{metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f} | "
                f"{metrics['completion_rate']:.2%} |"
            )

    lines.extend(
        [
            "",
            "## PPO 相对 FCFS",
            "",
            "| 负载模式 | PPO reward | FCFS reward | 提升 |",
            "|:--|--:|--:|--:|",
        ]
    )
    for pattern in results["patterns"].values():
        ppo = pattern["strategies"]["PPO"]["mean_reward"]
        fcfs = pattern["strategies"]["FCFS"]["mean_reward"]
        improvement = _improvement(ppo, fcfs)
        improvement_text = "N/A" if math.isnan(improvement) else f"{improvement:+.1f}%"
        lines.append(f"| {pattern['label']} | {ppo:.2f} | {fcfs:.2f} | {improvement_text} |")

    lines.extend(
        [
            "",
            "## 口径说明",
            "",
            "- 报告中的 reward 来自本次实际运行，不复用旧默认场景的汇总数字。",
            "- `arrival_lambda=None` 仍保留环境历史 λ=1.2 行为；本实验三个模式均显式注入参数。",
            "- 原始逐 episode reward 与调度计数保存在数据 JSON 中，便于复核。",
            "",
            f"数据文件：`{data_path.as_posix()}`",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Issue #160 多负载模式 × 8 策略对比")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--ppo-model", default="deliverable_models/ppo_best_model_14dim.zip")
    parser.add_argument("--dqn-model", default="deliverable_models/dqn_best_model_10dim.zip")
    parser.add_argument(
        "--output-data",
        type=Path,
        default=Path("results/workload_pattern_evaluation/workload_pattern_results.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("results/reports/workload_pattern_comparison.md"),
    )
    args = parser.parse_args()

    results = run_comparison(
        seeds=args.seeds,
        episodes_per_seed=args.episodes,
        max_steps=args.max_steps,
        ppo_model=args.ppo_model,
        dqn_model=args.dqn_model,
    )
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    args.output_data.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_report(results, args.output_report, args.output_data)
    print(f"\n数据：{args.output_data}")
    print(f"报告：{args.output_report}")


if __name__ == "__main__":
    main()
