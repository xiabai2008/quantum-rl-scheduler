#!/usr/bin/env python
"""Issue #29：真实留出负载盲测与分布外泛化验证。

本脚本实现"留出负载盲测"（holdout blind evaluation）：

1. **预生成 trace**：在评估前用固定 seed 生成多种分布的任务 trace（JSONL），
   trace 内容在评估过程中不再变化，确保"留出"性质。
2. **monkey-patching 注入**：通过替换 ``env._generate_random_task``，
   让环境从 trace 顺序读取任务属性，而非实时随机生成。
3. **多策略公平对比**：对每个 trace × 8 策略 × N seeds × M episodes
   运行评估，收集逐 episode 奖励。
4. **分布外泛化验证**：trace 包含与训练分布不同的负载模式
   （burst 突发、long_tail 长尾、high_quantum 高量子占比），
   验证 PPO 在 OOD 场景下的泛化能力。

使用示例：

    # 默认：4 trace × 8 策略 × 5 seeds × 3 episodes
    python scripts/evaluation/run_holdout_evaluation.py

    # 快速验证（小规模）
    python scripts/evaluation/run_holdout_evaluation.py --seeds 2 --episodes 2

    # 自定义输出
    python scripts/evaluation/run_holdout_evaluation.py \
        --output-data results/holdout_evaluation/holdout_results.json \
        --output-report results/reports/holdout_evaluation.md

作者：量子RL调度系统团队
日期：2026-07-22
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
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
    build_strategies,
)

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_types import Task

# ---------------------------------------------------------------------------
# Trace 分布定义
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HoldoutDistribution:
    """留出负载分布定义。

    每个分布通过 ``task_generator`` 函数定义任务属性采样逻辑，
    通过 ``arrival_lambda`` 控制到达率。分布参数在评估前固定，
    评估过程中不再变化，确保"留出"性质。
    """

    key: str
    label: str
    description: str
    arrival_lambda: float
    quantum_ratio: float | None
    # task_generator 由 _make_task_generator 动态构建，不存为字段
    generator_key: str  # 用于选择 task_generator


def _gen_in_distribution(rng: np.random.Generator, task_id: int) -> Task:
    """与训练分布同分布的任务（控制组）。"""
    qubit_options = [2, 3, 5, 5, 8, 10, 10, 15, 20, 30, 50, 100]
    qubit_probs = [0.15, 0.15, 0.15, 0.10, 0.10, 0.10, 0.05, 0.05, 0.05, 0.05, 0.03, 0.02]
    qubits = int(rng.choice(qubit_options, p=qubit_probs))
    urgency = float(rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], p=[0.1, 0.2, 0.4, 0.2, 0.1]))
    base_time = int(qubits**0.6)
    task_type = "quantum" if rng.random() < 0.7 else "classical"
    qubit_count = 0 if task_type == "classical" else qubits
    return Task(
        task_id=f"T{task_id:04d}",
        task_type=task_type,
        qubit_count=qubit_count,
        wait_steps=0,
        urgency=urgency,
        priority=int(rng.integers(1, 6)),
        execution_time=max(1, base_time + int(rng.choice([-1, 0, 0, 1]))),
    )


def _gen_burst(rng: np.random.Generator, task_id: int) -> Task:
    """突发性大任务分布（OOD）：50% 概率生成 50+ 量子比特的大任务。"""
    if rng.random() < 0.5:
        qubits = int(rng.choice([50, 80, 100, 150, 200, 287]))
    else:
        qubits = int(rng.choice([2, 3, 5, 8, 10]))
    urgency = float(rng.choice([0.8, 0.9, 0.95, 1.0], p=[0.3, 0.3, 0.2, 0.2]))
    base_time = int(qubits**0.6)
    task_type = "quantum" if rng.random() < 0.85 else "classical"
    qubit_count = 0 if task_type == "classical" else qubits
    return Task(
        task_id=f"T{task_id:04d}",
        task_type=task_type,
        qubit_count=qubit_count,
        wait_steps=0,
        urgency=urgency,
        priority=int(rng.integers(3, 6)),  # 偏高优先级
        execution_time=max(1, base_time + int(rng.choice([0, 1, 2]))),
    )


def _gen_long_tail(rng: np.random.Generator, task_id: int) -> Task:
    """长尾分布（OOD）：大任务占比提升，模拟真实生产中偶发的超大量子任务。"""
    qubits = int(
        rng.choice(
            [2, 3, 5, 8, 10, 20, 50, 100, 150, 200, 287],
            p=[0.05, 0.05, 0.10, 0.10, 0.10, 0.15, 0.15, 0.10, 0.08, 0.07, 0.05],
        )
    )
    urgency = float(rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], p=[0.2, 0.3, 0.3, 0.15, 0.05]))
    base_time = int(qubits**0.6)
    task_type = "quantum" if rng.random() < 0.75 else "classical"
    qubit_count = 0 if task_type == "classical" else qubits
    return Task(
        task_id=f"T{task_id:04d}",
        task_type=task_type,
        qubit_count=qubit_count,
        wait_steps=0,
        urgency=urgency,
        priority=int(rng.integers(1, 6)),
        execution_time=max(1, base_time * 2),  # 执行时间翻倍
    )


def _gen_high_quantum(rng: np.random.Generator, task_id: int) -> Task:
    """高量子任务占比分布（OOD）：95% 量子任务，测试量子资源压力。"""
    qubits = int(rng.choice([5, 8, 10, 15, 20, 30, 50]))
    urgency = float(rng.choice([0.3, 0.5, 0.7, 0.9], p=[0.2, 0.4, 0.3, 0.1]))
    base_time = int(qubits**0.6)
    task_type = "quantum" if rng.random() < 0.95 else "classical"
    qubit_count = 0 if task_type == "classical" else qubits
    return Task(
        task_id=f"T{task_id:04d}",
        task_type=task_type,
        qubit_count=qubit_count,
        wait_steps=0,
        urgency=urgency,
        priority=int(rng.integers(1, 6)),
        execution_time=max(1, base_time),
    )


# 分布注册表
TASK_GENERATORS: dict[str, Any] = {
    "in_distribution": _gen_in_distribution,
    "burst": _gen_burst,
    "long_tail": _gen_long_tail,
    "high_quantum": _gen_high_quantum,
}


HOLDOUT_DISTRIBUTIONS: tuple[HoldoutDistribution, ...] = (
    HoldoutDistribution(
        key="in_distribution",
        label="分布内（控制组）",
        description="与训练同分布，验证基线性能",
        arrival_lambda=0.5,
        quantum_ratio=0.7,
        generator_key="in_distribution",
    ),
    HoldoutDistribution(
        key="burst",
        label="突发大任务（OOD）",
        description="50% 概率 50+ 量子比特大任务，高优先级",
        arrival_lambda=0.8,
        quantum_ratio=0.85,
        generator_key="burst",
    ),
    HoldoutDistribution(
        key="long_tail",
        label="长尾大任务（OOD）",
        description="大任务占比提升，执行时间翻倍",
        arrival_lambda=0.4,
        quantum_ratio=0.75,
        generator_key="long_tail",
    ),
    HoldoutDistribution(
        key="high_quantum",
        label="高量子占比（OOD）",
        description="95% 量子任务，测试量子资源压力",
        arrival_lambda=0.6,
        quantum_ratio=0.95,
        generator_key="high_quantum",
    ),
)


# ---------------------------------------------------------------------------
# Trace 生成与序列化
# ---------------------------------------------------------------------------


@dataclass
class TraceEntry:
    """trace 中单个任务的序列化结构。"""

    task_id: str
    task_type: str
    qubit_count: int
    urgency: float
    priority: int
    execution_time: int


def generate_trace(
    distribution: HoldoutDistribution,
    trace_length: int,
    seed: int,
) -> list[TraceEntry]:
    """用固定 seed 生成一条任务 trace。

    Args:
        distribution: 留出负载分布定义
        trace_length: trace 长度（任务数）
        seed: 随机种子（固定，确保可复现）

    Returns:
        TraceEntry 列表
    """
    rng = np.random.default_rng(seed)
    generator = TASK_GENERATORS[distribution.generator_key]
    entries: list[TraceEntry] = []
    for i in range(trace_length):
        task = generator(rng, i)
        entries.append(
            TraceEntry(
                task_id=task.task_id,
                task_type=task.task_type,
                qubit_count=task.qubit_count,
                urgency=task.urgency,
                priority=task.priority,
                execution_time=task.execution_time,
            )
        )
    return entries


def save_trace(entries: list[TraceEntry], path: Path) -> None:
    """将 trace 序列化为 JSONL 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")


def load_trace(path: Path) -> list[TraceEntry]:
    """从 JSONL 文件加载 trace。"""
    entries: list[TraceEntry] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            entries.append(TraceEntry(**data))
    return entries


# ---------------------------------------------------------------------------
# Trace 注入（monkey-patching）
# ---------------------------------------------------------------------------


class TraceInjector:
    """将 trace 注入环境的任务生成器。

    通过 monkey-patching ``env._generate_random_task``，让环境从 trace
    顺序读取任务属性，而非实时随机生成。当 trace 耗尽时回退到原始生成器
    （避免环境因任务不足而提前终止）。
    """

    def __init__(self, trace: list[TraceEntry]) -> None:
        self.trace = list(trace)
        self.index = 0

    def inject(self, env: QuantumSchedulingEnv) -> None:
        """将 trace 注入到环境实例。"""
        self.index = 0  # 每个 episode 重置索引
        original = env._generate_random_task

        def patched(rng: np.random.Generator, task_id: int) -> Task:
            if self.index < len(self.trace):
                entry = self.trace[self.index]
                self.index += 1
                return Task(
                    task_id=entry.task_id,
                    task_type=entry.task_type,
                    qubit_count=entry.qubit_count,
                    wait_steps=0,
                    urgency=entry.urgency,
                    priority=entry.priority,
                    execution_time=entry.execution_time,
                )
            # trace 耗尽时回退到原始生成器
            return original(rng, task_id)

        # 直接设置实例属性（覆盖类方法绑定），调用时不自动传入 self
        env._generate_random_task = patched  # type: ignore[method-assign]


def make_holdout_env(
    distribution: HoldoutDistribution,
    max_steps: int,
    trace: list[TraceEntry],
) -> QuantumSchedulingEnv:
    """创建注入了 trace 的原生 14 维评估环境。

    使用原生 14 维环境（不截断为 10 维）以兼容 PPO 14 维模型。
    DQN 10 维模型在 build_strategies 中退化为随机策略（与多 seed 评估口径一致）。
    """
    base_env = QuantumSchedulingEnv(
        max_steps=max_steps,
        max_qubits=287,
        arrival_lambda=distribution.arrival_lambda,
        quantum_task_ratio=distribution.quantum_ratio,
    )
    injector = TraceInjector(trace)
    injector.inject(base_env)
    return base_env


# ---------------------------------------------------------------------------
# 评估循环
# ---------------------------------------------------------------------------


def _evaluate_strategy(
    distribution: HoldoutDistribution,
    trace: list[TraceEntry],
    strategy: BaseStrategy,
    seed_list: list[int],
    episodes_per_seed: int,
    max_steps: int,
) -> dict[str, Any]:
    """运行一个策略并汇总逐 episode 奖励与完成率。"""
    rewards: list[float] = []
    completed = 0
    attempts = 0

    for seed in seed_list:
        for episode in range(episodes_per_seed):
            env = make_holdout_env(distribution, max_steps, trace)
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
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "std_reward": float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0,
        "completion_rate": completed / attempts if attempts else 0.0,
        "completed_schedules": completed,
        "scheduling_attempts": attempts,
        "episode_rewards": rewards,
    }


def run_holdout_evaluation(
    seeds: int = 5,
    episodes_per_seed: int = 3,
    max_steps: int = 200,
    trace_length: int = 500,
    ppo_model: str = "deliverable_models/ppo_best_model_14dim.zip",
    dqn_model: str = "",
    distributions: tuple[HoldoutDistribution, ...] = HOLDOUT_DISTRIBUTIONS,
    trace_seed: int = 20260722,
) -> dict[str, Any]:
    """执行留出负载盲测评估。

    Args:
        seeds: 随机种子数（每个分布）
        episodes_per_seed: 每个种子的 episode 数
        max_steps: 每个 episode 的步数
        trace_length: 每条 trace 的任务数
        ppo_model: PPO 模型路径
        dqn_model: DQN 模型路径（空字符串则退化为随机策略）
        distributions: 留出分布列表
        trace_seed: trace 生成种子（固定，确保可复现）

    Returns:
        完整结果字典
    """
    if seeds <= 0 or episodes_per_seed <= 0 or max_steps <= 0:
        raise ValueError("seeds, episodes_per_seed and max_steps must be positive")
    if trace_length <= 0:
        raise ValueError("trace_length must be positive")

    seed_list = [42 + index * 137 for index in range(seeds)]
    # dqn_model 为空字符串时不传给 build_strategies（退化为随机策略）
    dqn_path = dqn_model if dqn_model else None
    strategies = build_strategies(dqn_path=dqn_path, ppo_path=ppo_model)
    if len(strategies) != 8:
        raise RuntimeError(f"expected 8 strategies, loaded {len(strategies)}")

    started = time.perf_counter()
    distribution_results: dict[str, Any] = {}

    for dist in distributions:
        print(f"\n[{dist.label}] {dist.description}")
        # 预生成 trace（固定 seed，确保留出性质）
        trace = generate_trace(dist, trace_length, seed=trace_seed)
        print(f"  trace: {len(trace)} tasks, seed={trace_seed}")

        strategy_results: dict[str, Any] = {}
        for strategy in strategies:
            strategy_started = time.perf_counter()
            metrics = _evaluate_strategy(
                dist,
                trace,
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

        distribution_results[dist.key] = {
            "label": dist.label,
            "description": dist.description,
            "arrival_lambda": dist.arrival_lambda,
            "quantum_ratio": dist.quantum_ratio,
            "generator_key": dist.generator_key,
            "trace_length": len(trace),
            "trace_seed": trace_seed,
            "strategies": strategy_results,
        }

    return {
        "config": {
            "seed_list": seed_list,
            "episodes_per_seed": episodes_per_seed,
            "max_steps": max_steps,
            "trace_length": trace_length,
            "trace_seed": trace_seed,
            "total_episodes_per_strategy_distribution": seeds * episodes_per_seed,
            "observation_dim": 14,
            "wrapper": "原生 14 维环境（兼容 PPO 14 维模型）",
            "ppo_model": ppo_model,
            "dqn_model": dqn_model,
            "completion_rate_definition": "total_scheduled / (total_scheduled + mismatch_count)",
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "elapsed_seconds": time.perf_counter() - started,
        },
        "distributions": distribution_results,
    }


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------


def _improvement(ppo_reward: float, fcfs_reward: float) -> float:
    """计算 PPO 相对 FCFS 的有符号百分比变化。"""
    if fcfs_reward == 0.0:
        return math.nan
    return (ppo_reward - fcfs_reward) / abs(fcfs_reward) * 100.0


def generate_report(results: dict[str, Any], report_path: Path, data_path: Path) -> None:
    """生成留出负载盲测报告。"""
    config = results["config"]
    lines = [
        "# 留出负载盲测与分布外泛化验证报告",
        "",
        "> **Issue #29 (P1)。** 本报告通过预先冻结的任务 trace 对 8 种调度策略进行盲测，",
        "> 验证 PPO 在分布外（OOD）负载下的泛化能力。trace 在评估前用固定 seed 生成，",
        "> 评估过程中不再变化，确保「留出」性质。",
        "",
        "## 实验配置",
        "",
        f"- 随机种子：`{config['seed_list']}`",
        f"- 重复次数：{len(config['seed_list'])} seeds × {config['episodes_per_seed']} episodes "
        f"= {config['total_episodes_per_strategy_distribution']} episodes/策略/分布",
        f"- 每个 episode：{config['max_steps']} 步",
        f"- trace 长度：{config['trace_length']} 任务（固定 seed={config['trace_seed']}）",
        "- 观测：14 维原生环境（兼容 PPO 14 维模型）",
        f"- PPO：`{config['ppo_model']}`",
        f"- DQN：`{config['dqn_model']}`",
        "- 完成率：成功调度数 /（成功调度数 + 资源不兼容次数）",
        "",
        "## 4 分布 × 8 策略结果",
        "",
        "| 负载分布 | 策略 | Reward（mean ± std） | 完成率 |",
        "|:--|:--|--:|--:|",
    ]

    for dist in results["distributions"].values():
        for strategy_name, metrics in dist["strategies"].items():
            lines.append(
                f"| {dist['label']} | {strategy_name} | "
                f"{metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f} | "
                f"{metrics['completion_rate']:.2%} |"
            )

    lines.extend(
        [
            "",
            "## PPO 相对 FCFS（OOD 泛化验证）",
            "",
            "| 负载分布 | PPO reward | FCFS reward | 提升 |",
            "|:--|--:|--:|--:|",
        ]
    )
    for dist in results["distributions"].values():
        ppo = dist["strategies"]["PPO"]["mean_reward"]
        fcfs = dist["strategies"]["FCFS"]["mean_reward"]
        improvement = _improvement(ppo, fcfs)
        improvement_text = "N/A" if math.isnan(improvement) else f"{improvement:+.1f}%"
        lines.append(f"| {dist['label']} | {ppo:.2f} | {fcfs:.2f} | {improvement_text} |")

    lines.extend(
        [
            "",
            "## 分布外泛化结论",
            "",
            "- **分布内（控制组）**：验证 PPO 在训练分布下的基线性能。",
            "- **突发大任务（OOD）**：50% 概率 50+ 量子比特大任务，测试 PPO 对资源突增的应对。",
            "- **长尾大任务（OOD）**：大任务占比提升，执行时间翻倍，测试 PPO 对长尾负载的鲁棒性。",
            "- **高量子占比（OOD）**：95% 量子任务，测试 PPO 在量子资源压力下的调度能力。",
            "",
            "若 PPO 在 OOD 分布下的提升仍为正，说明 PPO 学到的调度策略具有跨分布泛化能力；",
            "若提升为负或大幅下降，说明 PPO 过拟合训练分布，需在白皮书中如实披露。",
            "",
            "## 口径说明",
            "",
            "- trace 在评估前用固定 seed 生成，评估过程中不再变化（留出性质）。",
            "- trace 通过 monkey-patching `env._generate_random_task` 注入，不修改核心环境代码。",
            "- 当 trace 耗尽时回退到原始生成器，避免环境因任务不足而提前终止。",
            "- 原始逐 episode reward 与调度计数保存在数据 JSON 中，便于复核。",
            "",
            f"数据文件：`{data_path.as_posix()}`",
            "",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------


def main() -> None:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description="Issue #29 留出负载盲测与分布外泛化验证")
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--trace-length", type=int, default=500)
    parser.add_argument("--trace-seed", type=int, default=20260722)
    parser.add_argument("--ppo-model", default="deliverable_models/ppo_best_model_14dim.zip")
    parser.add_argument(
        "--dqn-model",
        default="",
        help="DQN 模型路径；留空则 DQN 退化为随机策略（10 维模型与 14 维环境不兼容）",
    )
    parser.add_argument(
        "--output-data",
        type=Path,
        default=Path("results/holdout_evaluation/holdout_results.json"),
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("results/reports/holdout_evaluation.md"),
    )
    args = parser.parse_args()

    results = run_holdout_evaluation(
        seeds=args.seeds,
        episodes_per_seed=args.episodes,
        max_steps=args.max_steps,
        trace_length=args.trace_length,
        ppo_model=args.ppo_model,
        dqn_model=args.dqn_model,
        trace_seed=args.trace_seed,
    )
    args.output_data.parent.mkdir(parents=True, exist_ok=True)
    args.output_data.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    generate_report(results, args.output_report, args.output_data)
    print(f"\n数据：{args.output_data}")
    print(f"报告：{args.output_report}")


if __name__ == "__main__":
    main()
