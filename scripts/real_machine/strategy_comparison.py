"""8 策略真机对比实验（阶段 2）。

在相同任务集上运行 8 种调度策略，每策略抽样 5 个真机任务，
共 ~40 个真机任务，对比策略间真机性能差异。

8 策略:
    1. FCFS          - 先来先服务
    2. Random        - 随机分配
    3. Greedy        - 贪心调度
    4. SJF           - 最短作业优先
    5. Quantum-Only  - 仅量子资源
    6. Classical-Only - 仅经典资源
    7. PPO           - PPO 强化学习
    8. DQN           - DQN 强化学习

用法:
    # Mock dry-run
    python scripts/real_machine/strategy_comparison.py --mock

    # 真机执行
    python scripts/real_machine/strategy_comparison.py
"""

from __future__ import annotations

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
os.environ.setdefault("TIANYAN_MACHINE", "tianyan176")

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

# 复用 smoke_test.py 工具函数
from smoke_test import (  # type: ignore[import-not-found]
    MockSmokeClient,
    parse_probability,
    compute_probability_from_shots,
    compute_measurement_error,
    compute_fidelity,
    poll_task_result,
)

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv

# 导入基线策略
from run_simulation import (  # type: ignore[import-not-found]
    BaseStrategy,
    FCFSStrategy,
    RandomStrategy,
    GreedyStrategy,
    ShortestJobFirstStrategy,
    QuantumOnlyStrategy,
    ClassicalOnlyStrategy,
    PPOStrategy,
    DQNModelStrategy,
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
RESULTS_DIR = _PROJECT_ROOT / "results" / "real_machine"

# 实验参数
NUM_TASKS = 200  # 每 episode 任务数
MAX_STEPS = 500  # 每 episode 最大步数
SEED = 42  # 随机种子
REAL_INTERVAL = 100  # 每 100 步提交 1 个真机任务 → 5 个/策略
REAL_SHOTS = 1024  # 真机 shots 数
QCIS_CIRCUIT = "H Q0\nM Q0"  # H 门（阶段 0 验证高保真度）

# 预训练模型路径
PPO_MODEL_PATH = str(_PROJECT_ROOT / "deliverable_models" / "ppo_best_model_10dim.zip")
DQN_MODEL_PATH = str(_PROJECT_ROOT / "deliverable_models" / "dqn_best_model_10dim.zip")

# 动作含义
ACTION_MEANINGS: dict[int, str] = {0: "classical", 1: "quantum", 2: "hybrid"}

# 预训练模型训练时的观测空间维度（v1 扩展前为 10 维）
LEGACY_OBS_DIM = 10


# ---------------------------------------------------------------------------
# 兼容包装器：将 14 维观测截断为 10 维供旧模型使用
# ---------------------------------------------------------------------------


class CompatModelStrategy(BaseStrategy):
    """旧模型兼容策略：截断 14 维观测为 10 维后送入预训练模型。

    v1 技术提升将观测空间从 10 维扩展到 14 维（新增物理噪声和拓扑特征），
    但预训练 PPO/DQN 模型仍基于 10 维训练。此包装器截断观测向量前 10 维
    供旧模型推理使用。
    """

    def __init__(self, model: Any, name: str = "Model"):
        self.model = model
        self.name = name

    def select_action(self, obs: np.ndarray) -> int:
        """截断观测后调用模型预测。

        Args:
            obs: 14 维观测向量

        Returns:
            动作索引 (0/1/2)
        """
        # 截断为旧模型期望的维度
        compat_obs = obs[:LEGACY_OBS_DIM]
        action, _ = self.model.predict(compat_obs, deterministic=True)
        return int(action.item())


# ---------------------------------------------------------------------------
# 创建 8 策略
# ---------------------------------------------------------------------------


def create_strategies() -> list[BaseStrategy]:
    """创建 8 种调度策略列表。

    PPO 和 DQN 使用预训练模型（通过兼容包装器），其余 6 种为规则策略。

    Returns:
        策略实例列表
    """
    strategies: list[BaseStrategy] = [
        FCFSStrategy(),
        RandomStrategy(seed=SEED),
        GreedyStrategy(),
        ShortestJobFirstStrategy(),
        QuantumOnlyStrategy(),
        ClassicalOnlyStrategy(),
    ]

    # PPO 策略（加载预训练模型，使用兼容包装器处理维度不匹配）
    try:
        from stable_baselines3 import PPO

        ppo_model = PPO.load(PPO_MODEL_PATH)
        strategies.append(CompatModelStrategy(ppo_model, name="PPO"))
        logger.info(f"[Strategy] PPO 模型已加载（兼容模式）: {PPO_MODEL_PATH}")
    except Exception as e:
        logger.warning(f"[Strategy] PPO 模型加载失败: {e}，使用 Greedy 替代")
        strategies.append(GreedyStrategy())

    # DQN 策略（加载预训练模型，使用兼容包装器）
    try:
        from stable_baselines3 import DQN

        dqn_model = DQN.load(DQN_MODEL_PATH)
        strategies.append(CompatModelStrategy(dqn_model, name="DQN"))
        logger.info(f"[Strategy] DQN 模型已加载（兼容模式）: {DQN_MODEL_PATH}")
    except Exception as e:
        logger.warning(f"[Strategy] DQN 模型加载失败: {e}，使用 SJF 替代")
        strategies.append(ShortestJobFirstStrategy())

    return strategies


# ---------------------------------------------------------------------------
# 单策略运行 + 真机抽样
# ---------------------------------------------------------------------------


def run_single_strategy(
    strategy: BaseStrategy,
    client: Any,
    machine_name: str,
    seed: int = SEED,
) -> dict[str, Any]:
    """运行单个策略并在固定间隔提交真机任务。

    Args:
        strategy: 调度策略实例
        client: 真机客户端
        machine_name: 机器名称
        seed: 随机种子

    Returns:
        策略运行结果字典
    """
    # 创建环境
    env = QuantumSchedulingEnv(
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=seed,
        real_submit_probability=0.0,
    )

    obs, _info = env.reset(seed=seed)
    total_reward = 0.0
    step = 0
    action_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}
    real_records: list[dict[str, Any]] = []

    while step < MAX_STEPS:
        # 策略选择动作
        action = strategy.select_action(obs)
        action_counts[action] = action_counts.get(action, 0) + 1

        # 环境步进
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        step += 1

        # 真机抽样：每 REAL_INTERVAL 步提交 1 个任务
        if step % REAL_INTERVAL == 0:
            record: dict[str, Any] = {
                "strategy": strategy.name,
                "step": step,
                "rl_action": int(action),
                "rl_action_meaning": ACTION_MEANINGS.get(action, "unknown"),
                "reward": round(float(reward), 4),
                "machine": machine_name,
                "qcis": QCIS_CIRCUIT,
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
                    qcis=QCIS_CIRCUIT,
                    shots=REAL_SHOTS,
                    task_name=f"StratCmp_{strategy.name}_step{step}",
                )
                record["real_task_id"] = str(real_tid) if real_tid else None
                record["submit_status"] = "submitted" if real_tid else "rejected"
                logger.info(
                    f"[StratCmp] {strategy.name} step={step} "
                    f"tid={real_tid} action={ACTION_MEANINGS.get(action, '?')}"
                )
            except Exception as e:
                record["submit_status"] = f"error: {str(e)[:80]}"
                logger.error(f"[StratCmp] {strategy.name} step={step} 提交失败: {e}")

            real_records.append(record)

        if terminated or truncated:
            break

    # 汇总环境指标
    summary = env.get_summary() if hasattr(env, "get_summary") else {}

    return {
        "strategy_name": strategy.name,
        "total_reward": round(total_reward, 4),
        "total_steps": step,
        "action_distribution": {
            ACTION_MEANINGS.get(k, str(k)): v for k, v in action_counts.items()
        },
        "env_metrics": summary,
        "real_records": real_records,
    }


# ---------------------------------------------------------------------------
# 轮询所有真机结果
# ---------------------------------------------------------------------------


def poll_all_results(
    client: Any,
    all_results: list[dict[str, Any]],
    per_task_timeout: int = 60,
) -> None:
    """轮询所有策略的真机任务结果。

    Args:
        client: 真机客户端
        all_results: 所有策略的结果列表（原地修改）
        per_task_timeout: 单任务超时秒数
    """
    # 收集所有需要轮询的记录
    all_records: list[tuple[dict[str, Any], int, int]] = []
    for strat_idx, strat_result in enumerate(all_results):
        for rec_idx, record in enumerate(strat_result["real_records"]):
            if record.get("real_task_id"):
                all_records.append((record, strat_idx, rec_idx))

    total = len(all_records)
    logger.info(f"[Poll] 开始轮询 {total} 个真机任务结果")

    for i, (record, _, _) in enumerate(all_records):
        task_id = record["real_task_id"]
        print(f"  [{i + 1}/{total}] {record['strategy']} {task_id} ...", end=" ", flush=True)

        result = poll_task_result(
            client=client,
            task_id=task_id,
            timeout=per_task_timeout,
            poll_interval=3,
            max_unknown=3,
            per_poll_timeout=15,
        )

        if result.get("status") == "completed":
            probability = {}
            raw_data = result.get("raw", {})

            if isinstance(raw_data, dict):
                raw_prob = raw_data.get("probability")
                if raw_prob:
                    probability = parse_probability(raw_prob)
                if not probability:
                    result_status = raw_data.get("resultStatus")
                    if result_status:
                        probability = compute_probability_from_shots(result_status)

            if not probability and result.get("result"):
                probability = parse_probability(result["result"])

            mock_prob = record.get("mock_probability", {"0": 0.5, "1": 0.5})
            prob_diff = compute_measurement_error(probability, mock_prob)
            fidelity = compute_fidelity(probability, mock_prob)

            record["real_probability"] = probability
            record["probability_diff"] = round(prob_diff, 4)
            record["fidelity"] = round(fidelity, 4)
            record["measurement_error"] = round(prob_diff, 4)
            record["poll_status"] = "completed"
            print(f"[PASS] fid={record['fidelity']}")
        else:
            record["poll_status"] = result.get("status", "unknown")
            record["real_probability"] = {}
            print(f"[FAIL] {record['poll_status']}")


# ---------------------------------------------------------------------------
# 结果保存与打印
# ---------------------------------------------------------------------------


def save_results(
    all_results: list[dict[str, Any]],
    output_dir: Path | None = None,
) -> str:
    """保存 8 策略对比结果到 JSON。

    Args:
        all_results: 所有策略结果
        output_dir: 输出目录

    Returns:
        保存的文件路径
    """
    if output_dir is None:
        output_dir = RESULTS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = output_dir / f"strategy_comparison_{timestamp}.json"

    # 统计汇总
    total_real = 0
    total_completed = 0
    all_fidelities: list[float] = []
    all_diffs: list[float] = []

    for result in all_results:
        for rec in result["real_records"]:
            if rec.get("real_task_id"):
                total_real += 1
                if rec.get("poll_status") == "completed":
                    total_completed += 1
                    if rec.get("fidelity") is not None:
                        all_fidelities.append(rec["fidelity"])
                    if rec.get("probability_diff") is not None:
                        all_diffs.append(rec["probability_diff"])

    summary = {
        "test_type": "strategy_comparison",
        "timestamp": datetime.now().astimezone().isoformat(),
        "config": {
            "num_tasks": NUM_TASKS,
            "max_steps": MAX_STEPS,
            "seed": SEED,
            "real_interval": REAL_INTERVAL,
            "real_shots": REAL_SHOTS,
            "qcis": QCIS_CIRCUIT,
        },
        "overall": {
            "total_strategies": len(all_results),
            "total_real_tasks": total_real,
            "completed": total_completed,
            "failed": total_real - total_completed,
            "avg_fidelity": (
                round(sum(all_fidelities) / max(len(all_fidelities), 1), 4)
                if all_fidelities
                else None
            ),
            "avg_probability_diff": (
                round(sum(all_diffs) / max(len(all_diffs), 1), 4) if all_diffs else None
            ),
        },
        "strategies": all_results,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"[StratCmp] 结果已保存: {filepath}")
    return str(filepath)


def print_summary(all_results: list[dict[str, Any]]) -> None:
    """打印 8 策略对比汇总表。

    Args:
        all_results: 所有策略结果
    """
    print(f"\n{'=' * 80}")
    print("  8 策略真机对比实验 - 汇总报告")
    print(f"{'=' * 80}")

    print(
        f"\n  {'策略':<16s} {'总奖励':>10s} {'步数':>6s} "
        f"{'真机数':>6s} {'成功':>4s} {'平均保真度':>12s} {'平均差异':>10s}"
    )
    print(f"  {'-'*16} {'-'*10} {'-'*6} {'-'*6} {'-'*4} {'-'*12} {'-'*10}")

    for result in all_results:
        name = result["strategy_name"]
        reward = result["total_reward"]
        steps = result["total_steps"]
        real_records = result["real_records"]
        real_count = len([r for r in real_records if r.get("real_task_id")])
        completed = len([r for r in real_records if r.get("poll_status") == "completed"])
        fidelities = [r["fidelity"] for r in real_records if r.get("fidelity") is not None]
        diffs = [
            r["probability_diff"] for r in real_records if r.get("probability_diff") is not None
        ]

        avg_fid = f"{sum(fidelities)/len(fidelities):.4f}" if fidelities else "N/A"
        avg_diff = f"{sum(diffs)/len(diffs):.4f}" if diffs else "N/A"

        print(
            f"  {name:<16s} {reward:>10.2f} {steps:>6d} "
            f"{real_count:>6d} {completed:>4d} {avg_fid:>12s} {avg_diff:>10s}"
        )

    # 动作分布
    print(f"\n  {'策略':<16s} {'classical':>10s} {'quantum':>10s} {'hybrid':>10s}")
    print(f"  {'-'*16} {'-'*10} {'-'*10} {'-'*10}")
    for result in all_results:
        name = result["strategy_name"]
        dist = result["action_distribution"]
        c = dist.get("classical", 0)
        q = dist.get("quantum", 0)
        h = dist.get("hybrid", 0)
        print(f"  {name:<16s} {c:>10d} {q:>10d} {h:>10d}")

    print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    """8 策略对比主入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="8 策略真机对比实验")
    parser.add_argument("--mock", action="store_true", help="Mock dry-run")
    parser.add_argument("--machine", default="tianyan176", help="首选机器")
    parser.add_argument("--verbose", action="store_true", help="DEBUG 日志")
    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    # 创建客户端
    if args.mock:
        print("[Mode] Mock dry-run")
        client: Any = MockSmokeClient(machine_name=args.machine, mock_delay=0.01)
    else:
        print("[Mode] 真机执行")
        api_key = os.environ.get("TIANYAN_API_KEY", "")
        if not api_key:
            print("[FAIL] 未设置 TIANYAN_API_KEY")
            sys.exit(1)
        client = CqlibTianyanClient(
            login_key=api_key,
            machine_name=args.machine,
            auto_retry_machine=True,
        )
        print(f"[Setup] 真机客户端已创建: {args.machine}")

    # 创建策略
    strategies = create_strategies()
    print(f"\n[Setup] 已创建 {len(strategies)} 个策略: " f"{', '.join(s.name for s in strategies)}")

    # 运行所有策略
    all_results: list[dict[str, Any]] = []
    total_real = 0

    print(f"\n{'=' * 60}")
    print(f"  8 策略真机对比实验")
    print(f"  任务数: {NUM_TASKS} | 真机间隔: {REAL_INTERVAL} | shots: {REAL_SHOTS}")
    print(
        f"  预计真机任务: {len(strategies)} x {MAX_STEPS // REAL_INTERVAL} = "
        f"{len(strategies) * (MAX_STEPS // REAL_INTERVAL)}"
    )
    print(f"{'=' * 60}")

    for i, strategy in enumerate(strategies):
        print(f"\n--- [{i+1}/{len(strategies)}] {strategy.name} ---")
        t0 = time.time()
        result = run_single_strategy(
            strategy=strategy,
            client=client,
            machine_name=args.machine,
            seed=SEED,
        )
        elapsed = round(time.time() - t0, 1)
        real_count = len([r for r in result["real_records"] if r.get("real_task_id")])
        total_real += real_count
        all_results.append(result)
        print(
            f"  {strategy.name}: reward={result['total_reward']:.2f}, "
            f"steps={result['total_steps']}, real={real_count}, "
            f"耗时={elapsed}s"
        )

    # 轮询所有真机结果
    if total_real > 0:
        print(f"\n--- 轮询 {total_real} 个真机任务结果 ---")
        poll_all_results(client, all_results)

    # 保存结果
    filepath = save_results(all_results)

    # 打印汇总
    print_summary(all_results)
    print(f"\n  结果文件: {filepath}")


if __name__ == "__main__":
    main()
