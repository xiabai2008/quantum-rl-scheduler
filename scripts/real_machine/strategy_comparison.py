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

# 导入基线策略
from run_simulation import (  # type: ignore[import-not-found]
    BaseStrategy,
    ClassicalOnlyStrategy,
    FCFSStrategy,
    GreedyStrategy,
    QuantumOnlyStrategy,
    RandomStrategy,
    ShortestJobFirstStrategy,
)

# 复用 smoke_test.py 工具函数
from smoke_test import (  # type: ignore[import-not-found]
    MockSmokeClient,
    compute_fidelity,
    compute_measurement_error,
    compute_probability_from_shots,
    parse_probability,
    poll_task_result,
)

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv

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
PPO_MODEL_PATH = str(_PROJECT_ROOT / "deliverable_models" / "ppo_best_model_14dim.zip")
DQN_MODEL_PATH = str(_PROJECT_ROOT / "deliverable_models" / "dqn_best_model_10dim.zip")

# 动作含义
ACTION_MEANINGS: dict[int, str] = {0: "classical", 1: "quantum", 2: "hybrid"}

# 预训练模型训练时的观测空间维度（v1 扩展前为 10 维）
LEGACY_OBS_DIM = 10


# ---------------------------------------------------------------------------
# 兼容包装器：自动适配模型观测维度
# ---------------------------------------------------------------------------


class CompatModelStrategy(BaseStrategy):
    """模型兼容策略：根据模型观测空间维度自动截断或填充观测向量。

    Issue #133 修复：
    - 原代码无条件截断 14 维 obs 为 10 维（``obs[:LEGACY_OBS_DIM]``），
      但 PPO 权威模型是 14 维训练的，截断后 predict() 抛 ValueError。
    - 现在从 ``model.observation_space.shape`` 自动推断模型期望维度，
      仅当模型维度 < 环境维度时才截断，维度匹配时直接使用。
    """

    def __init__(self, model: Any, name: str = "Model"):
        self.model = model
        self.name = name
        # Issue #133: 从模型观测空间推断期望维度，避免无条件截断
        self._model_obs_dim: int | None = None
        try:
            shape = getattr(model.observation_space, "shape", None)
            if shape is not None and len(shape) >= 1:
                self._model_obs_dim = int(shape[0])
                logger.info(
                    f"[CompatModel:{name}] 模型期望观测维度={self._model_obs_dim}，"
                    f"环境维度=14，"
                    f"{'需截断' if self._model_obs_dim < 14 else '维度匹配无需截断'}"
                )
        except Exception as e:
            logger.warning(f"[CompatModel:{name}] 无法读取模型观测空间维度: {e}，假设为 14 维")

    def select_action(self, obs: np.ndarray) -> int:
        """根据模型维度自动截断或直接使用观测向量。

        Args:
            obs: 环境观测向量（14 维）

        Returns:
            动作索引 (0/1/2)
        """
        # Issue #133: 仅当模型维度 < 环境维度时才截断
        if self._model_obs_dim is not None and self._model_obs_dim < obs.shape[0]:
            compat_obs = obs[: self._model_obs_dim]
        else:
            compat_obs = obs
        action, _ = self.model.predict(compat_obs, deterministic=True)
        return int(action.item())


# ---------------------------------------------------------------------------
# 创建 8 策略
# ---------------------------------------------------------------------------


def create_strategies() -> tuple[list[BaseStrategy], list[str]]:
    """创建调度策略列表。

    Issue #133 修复：
    - PPO/DQN 加载失败时不再用规则策略替代，而是直接跳过（避免 8 策略对比退化为
      7 策略且缺少 RL 算法对比时产生误导性数据）
    - 返回 ``(strategies, skipped)`` 元组，``skipped`` 记录被跳过的策略名和原因

    Returns:
        (strategies, skipped): 策略实例列表, 被跳过的策略描述列表
    """
    strategies: list[BaseStrategy] = [
        FCFSStrategy(),
        RandomStrategy(seed=SEED),
        GreedyStrategy(),
        ShortestJobFirstStrategy(),
        QuantumOnlyStrategy(),
        ClassicalOnlyStrategy(),
    ]
    skipped: list[str] = []

    # PPO 策略（加载预训练模型，使用兼容包装器处理维度不匹配）
    try:
        from stable_baselines3 import PPO

        ppo_model = PPO.load(PPO_MODEL_PATH)
        strategies.append(CompatModelStrategy(ppo_model, name="PPO"))
        logger.info(f"[Strategy] PPO 模型已加载: {PPO_MODEL_PATH}")
    except Exception as e:
        # Issue #133: 加载失败时跳过而非替代，避免误导
        skip_reason = f"PPO 加载失败: {str(e)[:100]}"
        skipped.append(skip_reason)
        logger.warning(f"[Strategy] {skip_reason}，跳过 PPO 策略")

    # DQN 策略（加载预训练模型，使用兼容包装器）
    try:
        from stable_baselines3 import DQN

        dqn_model = DQN.load(DQN_MODEL_PATH)
        strategies.append(CompatModelStrategy(dqn_model, name="DQN"))
        logger.info(f"[Strategy] DQN 模型已加载: {DQN_MODEL_PATH}")
    except Exception as e:
        # Issue #133: 加载失败时跳过而非用 SJF 替代
        skip_reason = f"DQN 加载失败: {str(e)[:100]}"
        skipped.append(skip_reason)
        logger.warning(f"[Strategy] {skip_reason}，跳过 DQN 策略")

    return strategies, skipped


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
        obs, reward, terminated, truncated, _info = env.step(action)
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
    print(f"  {'-' * 16} {'-' * 10} {'-' * 6} {'-' * 6} {'-' * 4} {'-' * 12} {'-' * 10}")

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

        avg_fid = f"{sum(fidelities) / len(fidelities):.4f}" if fidelities else "N/A"
        avg_diff = f"{sum(diffs) / len(diffs):.4f}" if diffs else "N/A"

        print(
            f"  {name:<16s} {reward:>10.2f} {steps:>6d} "
            f"{real_count:>6d} {completed:>4d} {avg_fid:>12s} {avg_diff:>10s}"
        )

    # 动作分布
    print(f"\n  {'策略':<16s} {'classical':>10s} {'quantum':>10s} {'hybrid':>10s}")
    print(f"  {'-' * 16} {'-' * 10} {'-' * 10} {'-' * 10}")
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
    strategies, skipped = create_strategies()
    print(f"\n[Setup] 已创建 {len(strategies)} 个策略: {', '.join(s.name for s in strategies)}")
    if skipped:
        print(f"[Setup] 跳过的策略: {'; '.join(skipped)}")

    # 运行所有策略
    all_results: list[dict[str, Any]] = []
    total_real = 0

    print(f"\n{'=' * 60}")
    print(f"  {len(strategies)} 策略真机对比实验")
    print(f"  任务数: {NUM_TASKS} | 真机间隔: {REAL_INTERVAL} | shots: {REAL_SHOTS}")
    print(
        f"  预计真机任务: {len(strategies)} x {MAX_STEPS // REAL_INTERVAL} = "
        f"{len(strategies) * (MAX_STEPS // REAL_INTERVAL)}"
    )
    print(f"{'=' * 60}")

    # Issue #133: 增量保存文件路径，每个策略完成后立即保存，避免崩溃丢数据
    incremental_filepath = RESULTS_DIR / "strategy_comparison_incremental.json"

    for i, strategy in enumerate(strategies):
        print(f"\n--- [{i + 1}/{len(strategies)}] {strategy.name} ---")
        t0 = time.time()
        # Issue #133: 每个策略用 try-catch 保护，单个崩溃不影响其他策略
        try:
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
        except Exception as e:
            # Issue #133: 策略崩溃时记录错误并继续下一个策略
            logger.exception(f"[StratCmp] 策略 {strategy.name} 崩溃")
            print(f"  [ERROR] 策略 {strategy.name} 崩溃: {e}", flush=True)
            all_results.append(
                {
                    "strategy_name": strategy.name,
                    "error": f"{type(e).__name__}: {str(e)[:200]}",
                    "total_reward": 0.0,
                    "total_steps": 0,
                    "action_distribution": {},
                    "env_metrics": {},
                    "real_records": [],
                }
            )

        # Issue #133: 增量保存，确保已完成策略数据不丢失
        try:
            save_results(all_results, output_dir=RESULTS_DIR)
            # 同时覆盖 incremental 文件，便于追踪最新进度
            incremental_filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(incremental_filepath, "w", encoding="utf-8") as inc_f:
                json.dump(
                    {
                        "type": "strategy_comparison_incremental",
                        "timestamp": datetime.now().astimezone().isoformat(),
                        "completed_strategies": len(all_results),
                        "total_strategies": len(strategies),
                        "skipped_strategies": skipped,
                        "strategies": all_results,
                    },
                    inc_f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception as inc_e:
            logger.warning(f"[StratCmp] 增量保存失败: {inc_e}")

    # 轮询所有真机结果
    if total_real > 0:
        print(f"\n--- 轮询 {total_real} 个真机任务结果 ---")
        try:
            poll_all_results(client, all_results)
        except Exception as e:
            # Issue #133: 轮询崩溃不丢失已保存的策略结果
            logger.exception("[StratCmp] 轮询阶段崩溃")
            print(f"  [ERROR] 轮询阶段崩溃: {e}", flush=True)

    # 保存最终结果
    filepath = save_results(all_results)
    # 清理 incremental 文件
    if incremental_filepath.exists():
        incremental_filepath.unlink()

    # 打印汇总
    print_summary(all_results)
    print(f"\n  结果文件: {filepath}")


if __name__ == "__main__":
    main()
