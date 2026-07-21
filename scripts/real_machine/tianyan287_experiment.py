"""天衍-287 大规模真机实验脚本。

在 287 比特超导量子计算机上运行完整的 RL 调度实验，
包括多策略对比、多任务规模、多比特数任务调度。

实验设计:
    - 任务规模: 32 / 64 / 128 个量子任务
    - 调度策略: PPO / FCFS / SJF
    - 量子比特数: 1 / 2 / 4 / 8 qubit 电路
    - 对比指标: 总执行时间、资源利用率、等待时间、成功率

用法:
    # Mock dry-run（不消耗真机机时）
    python scripts/real_machine/tianyan287_experiment.py --mock --task-scale 32

    # 真机执行（需要天衍-287 真机要时）
    python scripts/real_machine/tianyan287_experiment.py --task-scale 32 --strategies ppo fcfs sjf

    # 完整实验（所有规模 × 所有策略）
    python scripts/real_machine/tianyan287_experiment.py --full

    # 仅多比特任务测试
    python scripts/real_machine/tianyan287_experiment.py --multi-qubit --qubit-sizes 1 2 4 8
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

import numpy as np

# ---------------------------------------------------------------------------
# 环境变量设置（必须在 import 项目模块之前）
# ---------------------------------------------------------------------------
os.environ.setdefault("TIANYAN_API_KEY", "")
os.environ.setdefault("TIANYAN_MOCK_MODE", "false")
os.environ.setdefault("TIANYAN_MACHINE", "tianyan287")
os.environ.setdefault("QUANTUM_ACCELERATION_ENABLED", "1")

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
_EVAL_DIR = _PROJECT_ROOT / "scripts" / "evaluation"

for p in [_PROJECT_ROOT, _SCRIPT_DIR, _EVAL_DIR]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from loguru import logger

# 复用基线策略（与 strategy_comparison.py 一致）
from run_simulation import (  # type: ignore[import-not-found]
    BaseStrategy,
    FCFSStrategy,
    ShortestJobFirstStrategy,
)

# 复用 smoke_test.py 工具函数
from smoke_test import (  # type: ignore[import-not-found]
    MockSmokeClient,
    compute_fidelity,
    compute_measurement_error,
    parse_probability,
    poll_task_result,
)

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
RESULTS_DIR = _PROJECT_ROOT / "results" / "real_machine" / "tianyan287"

STRATEGIES = ["ppo", "fcfs", "sjf"]
TASK_SCALES = [32, 64, 128]
QUBIT_SIZES = [1, 2, 4, 8]

# 真机参数
REAL_INTERVAL = 50  # 每 50 步提交 1 个真机任务
REAL_SHOTS = 1024
MAX_STEPS_BASE = 500  # 每 episode 最大步数

# 预训练模型路径
PPO_MODEL_PATH = str(_PROJECT_ROOT / "deliverable_models" / "ppo_best_model_14dim.zip")

# 动作含义
ACTION_MEANINGS: dict[int, str] = {0: "classical", 1: "quantum", 2: "hybrid"}

# 旧模型训练时的观测空间维度（v1 扩展前为 10 维）
LEGACY_OBS_DIM = 10


# ---------------------------------------------------------------------------
# 兼容包装器：将 14 维观测截断为 10 维供旧模型使用
# ---------------------------------------------------------------------------


class ModelStrategy(BaseStrategy):
    """预训练模型策略：将观测送入预训练模型进行推理。

    PPO 模型（14 维）直接接收 14 维观测；DQN 模型（10 维）需截断。
    """

    def __init__(self, model: Any, name: str = "Model", obs_dim: int | None = None):
        self.model = model
        self.name = name
        self.obs_dim = obs_dim  # None 表示不截断（使用原始维度）

    def select_action(self, obs: np.ndarray) -> int:
        """调用模型预测动作。

        Args:
            obs: 观测向量

        Returns:
            动作索引 (0/1/2)
        """
        model_obs = obs[: self.obs_dim] if self.obs_dim is not None else obs
        action, _ = self.model.predict(model_obs, deterministic=True)
        return int(action.item())


# ---------------------------------------------------------------------------
# 策略创建
# ---------------------------------------------------------------------------


def create_strategy(name: str) -> BaseStrategy:
    """创建单个调度策略实例。

    Args:
        name: 策略名称 (ppo/fcfs/sjf)

    Returns:
        策略实例
    """
    if name == "fcfs":
        return FCFSStrategy()
    if name == "sjf":
        return ShortestJobFirstStrategy()
    if name == "ppo":
        try:
            from stable_baselines3 import PPO

            model = PPO.load(PPO_MODEL_PATH)
            logger.info(f"[Strategy] PPO 模型已加载: {PPO_MODEL_PATH}")
            # PPO 模型为 14 维，环境也是 14 维，无需截断
            return ModelStrategy(model, name="PPO", obs_dim=None)
        except Exception as e:
            logger.warning(f"[Strategy] PPO 模型加载失败: {e}，使用 FCFS 替代")
            return FCFSStrategy()
    raise ValueError(f"未知策略: {name}")


# ---------------------------------------------------------------------------
# 策略对比实验
# ---------------------------------------------------------------------------


def run_strategy_comparison(
    strategy_name: str,
    num_tasks: int,
    client: Any,
    machine_name: str,
    max_real_tasks: int = 20,
) -> dict[str, Any]:
    """运行单策略对比实验。

    在相同任务集上运行指定策略，按固定间隔提交真机任务。

    Args:
        strategy_name: 策略名称 (ppo/fcfs/sjf)
        num_tasks: 任务总数（决定 episode 步数上限）
        client: 真机客户端（Mock 或真实）
        machine_name: 机器名称
        max_real_tasks: 最大真机提交任务数（控制机时消耗）

    Returns:
        实验结果字典
    """
    strategy = create_strategy(strategy_name)
    logger.info(f"开始策略对比: {strategy.name} | 任务数={num_tasks}")

    result: dict[str, Any] = {
        "strategy": strategy.name,
        "num_tasks": num_tasks,
        "machine": machine_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "real_records": [],
    }

    start_time = time.time()

    try:
        # 创建环境（14 维、异质化任务、多机器）
        env = QuantumSchedulingEnv(
            machine_configs=DEFAULT_MACHINE_CONFIGS,
            max_steps=min(num_tasks * 3, MAX_STEPS_BASE),
            arrival_lambda=0.5,
            seed=42,
            real_submit_probability=0.0,
        )

        # 注入真机客户端
        env.attach_real_clients({machine_name: client})

        obs, _info = env.reset(seed=42)
        total_reward = 0.0
        step = 0
        action_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
        real_records: list[dict[str, Any]] = []
        real_task_count = 0

        # 真机提交间隔（确保不超过 max_real_tasks）
        real_interval = max(1, min(num_tasks, MAX_STEPS_BASE) // max(1, max_real_tasks))

        # 简单测试电路
        qcis_circuit = "H Q0\nM Q0"

        while step < MAX_STEPS_BASE:
            # 策略选择动作
            action = strategy.select_action(obs)
            action_counts[action] = action_counts.get(action, 0) + 1

            # 环境步进
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            step += 1

            # 真机抽样：按间隔提交
            if step % real_interval == 0 and real_task_count < max_real_tasks:
                record: dict[str, Any] = {
                    "strategy": strategy.name,
                    "step": step,
                    "rl_action": int(action),
                    "rl_action_meaning": ACTION_MEANINGS.get(action, "unknown"),
                    "reward": round(float(reward), 4),
                    "machine": machine_name,
                    "qcis": qcis_circuit,
                    "real_task_id": None,
                    "submit_status": "pending",
                    "real_probability": {},
                    "mock_probability": {"0": 0.5, "1": 0.5},
                    "probability_diff": None,
                    "fidelity": None,
                    "measurement_error": None,
                    "poll_status": "pending",
                }

                try:
                    real_tid = client.submit_quantum_task(
                        qcis=qcis_circuit,
                        shots=REAL_SHOTS,
                        task_name=f"Ty287_{strategy.name}_step{step}",
                    )
                    record["real_task_id"] = str(real_tid) if real_tid else None
                    record["submit_status"] = "submitted" if real_tid else "rejected"
                    real_task_count += 1
                    logger.info(
                        f"[Ty287] {strategy.name} step={step} "
                        f"tid={real_tid} action={ACTION_MEANINGS.get(action, '?')}"
                    )
                except Exception as e:
                    record["submit_status"] = f"error: {str(e)[:80]}"
                    logger.error(f"[Ty287] {strategy.name} step={step} 提交失败: {e}")

                real_records.append(record)

            if terminated or truncated:
                break

        elapsed = time.time() - start_time

        # 轮询真机结果
        for record in real_records:
            if record.get("real_task_id"):
                poll_result = poll_task_result(
                    client=client,
                    task_id=record["real_task_id"],
                    timeout=60,
                    poll_interval=3,
                    max_unknown=3,
                    per_poll_timeout=15,
                )
                if poll_result.get("status") == "completed":
                    probability = {}
                    raw_data = poll_result.get("raw", {})
                    if isinstance(raw_data, dict):
                        raw_prob = raw_data.get("probability")
                        if raw_prob:
                            probability = parse_probability(raw_prob)
                    if not probability and poll_result.get("result"):
                        probability = parse_probability(poll_result["result"])

                    mock_prob = record.get("mock_probability", {"0": 0.5, "1": 0.5})
                    prob_diff = compute_measurement_error(probability, mock_prob)
                    fidelity = compute_fidelity(probability, mock_prob)

                    record["real_probability"] = probability
                    record["probability_diff"] = round(prob_diff, 4)
                    record["fidelity"] = round(fidelity, 4)
                    record["measurement_error"] = round(prob_diff, 4)
                    record["poll_status"] = "completed"
                else:
                    record["poll_status"] = poll_result.get("status", "unknown")

        # 汇总指标
        fidelities = [r["fidelity"] for r in real_records if r.get("fidelity") is not None]
        result["metrics"] = {
            "total_reward": round(total_reward, 4),
            "total_steps": step,
            "elapsed_seconds": round(elapsed, 2),
            "action_distribution": {
                ACTION_MEANINGS.get(k, str(k)): v for k, v in action_counts.items()
            },
            "real_tasks_submitted": real_task_count,
            "real_tasks_completed": len(
                [r for r in real_records if r.get("poll_status") == "completed"]
            ),
            "avg_fidelity": round(sum(fidelities) / max(len(fidelities), 1), 4)
            if fidelities
            else None,
        }

        logger.info(
            f"策略 {strategy.name} 完成: reward={total_reward:.2f}, "
            f"耗时={elapsed:.1f}s, 真机任务={real_task_count}"
        )

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"策略 {strategy.name} 执行失败: {e}")

    return result


# ---------------------------------------------------------------------------
# 多比特任务测试
# ---------------------------------------------------------------------------


def build_qcis_circuit(n_qubits: int) -> str:
    """构建 n-qubit 测试电路（每比特施加 H 门后测量）。

    Args:
        n_qubits: 量子比特数

    Returns:
        QCIS 指令字符串
    """
    lines = [f"H Q{i}" for i in range(n_qubits)]
    measure_qubits = " ".join(f"Q{i}" for i in range(n_qubits))
    lines.append(f"M {measure_qubits}")
    return "\n".join(lines)


def run_multi_qubit_test(
    qubit_sizes: list[int],
    client: Any,
    machine_name: str,
) -> dict[str, Any]:
    """多比特任务调度测试。

    测试不同量子比特数（1/2/4/8）的电路在真机上的调度性能。

    Args:
        qubit_sizes: 要测试的量子比特数列表
        client: 真机客户端
        machine_name: 机器名称

    Returns:
        测试结果字典
    """
    logger.info(f"开始多比特任务测试: qubit_sizes={qubit_sizes}")

    result: dict[str, Any] = {
        "test_type": "multi_qubit",
        "qubit_sizes": qubit_sizes,
        "machine": machine_name,
        "timestamp": datetime.now().isoformat(),
        "results_per_qubit": {},
    }

    for n_qubits in qubit_sizes:
        logger.info(f"  测试 {n_qubits}-qubit 电路...")

        qubit_result: dict[str, Any] = {
            "n_qubits": n_qubits,
            "qcis": build_qcis_circuit(n_qubits),
            "tasks_submitted": 0,
            "tasks_succeeded": 0,
            "tasks_failed": 0,
            "avg_execution_time": 0.0,
            "success_rate": 0.0,
            "fidelities": [],
        }

        execution_times: list[float] = []

        for i in range(5):
            try:
                circuit = build_qcis_circuit(n_qubits)
                t0 = time.time()

                task_id = client.submit_quantum_task(
                    qcis=circuit,
                    shots=REAL_SHOTS,
                    task_name=f"Ty287_{n_qubits}q_{i}",
                )
                qubit_result["tasks_submitted"] += 1

                if task_id:
                    poll_result = poll_task_result(
                        client=client,
                        task_id=task_id,
                        timeout=60,
                        poll_interval=3,
                        max_unknown=3,
                        per_poll_timeout=15,
                    )
                    elapsed = time.time() - t0
                    execution_times.append(elapsed)

                    if poll_result.get("status") == "completed":
                        qubit_result["tasks_succeeded"] += 1
                        # 计算保真度
                        probability = {}
                        raw_data = poll_result.get("raw", {})
                        if isinstance(raw_data, dict):
                            raw_prob = raw_data.get("probability")
                            if raw_prob:
                                probability = parse_probability(raw_prob)
                        if not probability and poll_result.get("result"):
                            probability = parse_probability(poll_result["result"])

                        # 理论值：所有比特均匀分布
                        n_outcomes = 2**n_qubits
                        theoretical = {
                            format(k, f"0{n_qubits}b"): 1.0 / n_outcomes for k in range(n_outcomes)
                        }
                        fid = compute_fidelity(probability, theoretical)
                        qubit_result["fidelities"].append(round(fid, 4))
                    else:
                        qubit_result["tasks_failed"] += 1
                else:
                    qubit_result["tasks_failed"] += 1

            except Exception as e:
                qubit_result["tasks_failed"] += 1
                logger.warning(f"    {n_qubits}q 任务 {i} 失败: {e}")

        qubit_result["avg_execution_time"] = (
            round(sum(execution_times) / max(len(execution_times), 1), 2)
            if execution_times
            else 0.0
        )
        qubit_result["success_rate"] = qubit_result["tasks_succeeded"] / max(
            1, qubit_result["tasks_submitted"]
        )
        qubit_result["avg_fidelity"] = (
            round(sum(qubit_result["fidelities"]) / max(len(qubit_result["fidelities"]), 1), 4)
            if qubit_result["fidelities"]
            else None
        )

        result["results_per_qubit"][str(n_qubits)] = qubit_result
        logger.info(
            f"  {n_qubits}q 完成: 成功={qubit_result['tasks_succeeded']}/"
            f"{qubit_result['tasks_submitted']}, "
            f"耗时={qubit_result['avg_execution_time']}s"
        )

    return result


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------


def generate_report(results: dict[str, Any]) -> str:
    """生成实验报告 Markdown。

    Args:
        results: 所有实验结果

    Returns:
        Markdown 格式报告字符串
    """
    lines = [
        "# 天衍-287 真机实验报告",
        "",
        f"> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "> 实验机器: 天衍-287（287 量子比特超导量子计算机）",
        "",
        "## 1. 实验概述",
        "",
        "### 实验环境",
        "- 机器: 天衍-287（287 量子比特）",
        f"- Mock 模式: {results.get('mock', False)}",
        f"- 实验时间: {results.get('timestamp', 'N/A')}",
        "",
        "## 2. 策略对比实验",
        "",
    ]

    # 策略对比结果
    strategy_results = results.get("strategy_comparison", {})
    if strategy_results:
        lines.append("| 策略 | 任务数 | 总奖励 | 步数 | 耗时(s) | 真机任务 | 完成 | 平均保真度 |")
        lines.append("|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
        for key, res in strategy_results.items():
            m = res.get("metrics", {})
            lines.append(
                f"| {res.get('strategy', key)} | {res.get('num_tasks', '-')} | "
                f"{m.get('total_reward', 0):.2f} | {m.get('total_steps', 0)} | "
                f"{m.get('elapsed_seconds', 0):.1f} | "
                f"{m.get('real_tasks_submitted', 0)} | "
                f"{m.get('real_tasks_completed', 0)} | "
                f"{m.get('avg_fidelity', 'N/A')} |"
            )
        lines.append("")

        # 动作分布
        lines.append("### 动作分布")
        lines.append("")
        lines.append("| 策略 | classical | quantum | hybrid |")
        lines.append("|:--|:--:|:--:|:--:|")
        for key, res in strategy_results.items():
            dist = res.get("metrics", {}).get("action_distribution", {})
            lines.append(
                f"| {res.get('strategy', key)} | "
                f"{dist.get('classical', 0)} | {dist.get('quantum', 0)} | "
                f"{dist.get('hybrid', 0)} |"
            )
        lines.append("")

    # 多比特测试结果
    multi_qubit = results.get("multi_qubit", {})
    if multi_qubit:
        lines.append("## 3. 多比特任务调度测试")
        lines.append("")
        lines.append("| 量子比特数 | 提交任务 | 成功 | 失败 | 成功率 | 平均执行时间 | 平均保真度 |")
        lines.append("|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
        for q_size, res in multi_qubit.get("results_per_qubit", {}).items():
            lines.append(
                f"| {q_size} | {res.get('tasks_submitted', 0)} | "
                f"{res.get('tasks_succeeded', 0)} | {res.get('tasks_failed', 0)} | "
                f"{res.get('success_rate', 0):.0%} | "
                f"{res.get('avg_execution_time', 0):.2f}s | "
                f"{res.get('avg_fidelity', 'N/A')} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 4. 结论",
            "",
            "（待填写 — 根据实验数据补充分析）",
            "",
            "## 5. 数据文件",
            "",
            "原始数据: `results/real_machine/tianyan287/experiment_data.json`",
            "",
        ]
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    """主函数：解析参数并运行实验。"""
    parser = argparse.ArgumentParser(description="天衍-287 大规模真机实验")
    parser.add_argument("--mock", action="store_true", help="使用 Mock 模式（不消耗真机机时）")
    parser.add_argument("--full", action="store_true", help="运行完整实验（所有规模×所有策略）")
    parser.add_argument(
        "--task-scale",
        type=int,
        default=32,
        choices=[32, 64, 128],
        help="任务规模",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=["ppo", "fcfs", "sjf"],
        help="要测试的策略",
    )
    parser.add_argument(
        "--multi-qubit",
        action="store_true",
        help="运行多比特任务测试",
    )
    parser.add_argument(
        "--qubit-sizes",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8],
        help="多比特测试的量子比特数",
    )
    parser.add_argument(
        "--max-real-tasks",
        type=int,
        default=20,
        help="每策略最大真机提交任务数（控制机时消耗）",
    )
    parser.add_argument("--verbose", action="store_true", help="DEBUG 日志")

    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # 创建结果目录
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # 机器名称
    machine_name = "tianyan287"

    # 创建客户端
    if args.mock:
        print("[Mode] Mock dry-run（不消耗真机机时）")
        client: Any = MockSmokeClient(machine_name=machine_name, mock_delay=0.01)
    else:
        print("[Mode] 真机执行")
        api_key = os.environ.get("TIANYAN_API_KEY", "")
        if not api_key:
            print("[FAIL] 未设置 TIANYAN_API_KEY 环境变量")
            sys.exit(1)
        client = CqlibTianyanClient(
            login_key=api_key,
            machine_name=machine_name,
            auto_retry_machine=True,
        )
        print(f"[Setup] 真机客户端已创建: {machine_name}")

    all_results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "mock": args.mock,
        "machine": machine_name,
    }

    # 策略对比实验
    if args.full:
        logger.info("=== 完整实验模式（所有规模 × 所有策略）===")
        strategy_results = {}
        for scale in TASK_SCALES:
            for strategy in STRATEGIES:
                key = f"{strategy}_{scale}"
                strategy_results[key] = run_strategy_comparison(
                    strategy, scale, client, machine_name, args.max_real_tasks
                )
        all_results["strategy_comparison"] = strategy_results
    else:
        strategy_results = {}
        for strategy in args.strategies:
            key = f"{strategy}_{args.task_scale}"
            strategy_results[key] = run_strategy_comparison(
                strategy, args.task_scale, client, machine_name, args.max_real_tasks
            )
        all_results["strategy_comparison"] = strategy_results

    # 多比特任务测试
    if args.multi_qubit:
        all_results["multi_qubit"] = run_multi_qubit_test(args.qubit_sizes, client, machine_name)

    # 保存原始数据
    data_path = RESULTS_DIR / "experiment_data.json"
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    logger.info(f"原始数据已保存: {data_path}")

    # 生成报告
    report = generate_report(all_results)
    report_path = RESULTS_DIR / "experiment_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"实验报告已生成: {report_path}")

    # 打印摘要
    print("\n" + "=" * 60)
    print("  天衍-287 实验完成")
    print("=" * 60)
    print(f"  Mock 模式: {args.mock}")
    print(f"  数据文件: {data_path}")
    print(f"  报告文件: {report_path}")

    # 策略对比摘要
    if strategy_results:
        print(f"\n  {'策略':<12s} {'奖励':>10s} {'步数':>6s} {'真机':>4s} {'保真度':>8s}")
        print(f"  {'-' * 12} {'-' * 10} {'-' * 6} {'-' * 4} {'-' * 8}")
        for key, res in strategy_results.items():
            m = res.get("metrics", {})
            fid = m.get("avg_fidelity", "N/A")
            fid_str = f"{fid:.4f}" if isinstance(fid, float | int) else str(fid)
            print(
                f"  {res.get('strategy', key):<12s} "
                f"{m.get('total_reward', 0):>10.2f} "
                f"{m.get('total_steps', 0):>6d} "
                f"{m.get('real_tasks_submitted', 0):>4d} "
                f"{fid_str:>8s}"
            )

    print("=" * 60)


if __name__ == "__main__":
    main()
