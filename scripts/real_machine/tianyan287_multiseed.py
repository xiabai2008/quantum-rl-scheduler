#!/usr/bin/env python
"""天衍-287 多seed真机实验

机时预算：5 seeds × 3策略 × 2真机任务 = 30个真机任务，约7分钟
目标：收集多seed数据，计算效应量(Cohen's d) + 95% CI

用法：
    python scripts/real_machine/tianyan287_multiseed.py
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# 环境变量设置（必须在 import 项目模块之前）
from dotenv import load_dotenv

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

os.environ.setdefault("TIANYAN_API_KEY", "")
os.environ.setdefault("TIANYAN_MOCK_MODE", "false")
os.environ.setdefault("TIANYAN_MACHINE", "tianyan287")

from loguru import logger

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.env import QuantumSchedulingEnv

# 复用 run_simulation 的策略类（有 act(obs) 接口）
_EVAL_DIR = _PROJECT_ROOT / "scripts" / "evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))
from run_simulation import (  # type: ignore[import-not-found]
    FCFSStrategy,
    PPOStrategy,
    ShortestJobFirstStrategy,
)

# ── 实验配置 ──

SEEDS = [42, 123, 456, 789, 1024]
STRATEGIES = ["ppo", "fcfs", "sjf"]
NUM_TASKS = 32
MAX_REAL_TASKS_PER_RUN = 1
REAL_SUBMIT_INTERVAL = 5  # 每5步提交一次真机任务
SHOTS = 1024

PPO_MODEL_PATH = _PROJECT_ROOT / "deliverable_models" / "ppo_best_model_14dim.zip"

DEFAULT_MACHINE_CONFIGS = [
    {
        "name": "tianyan287",
        "machine_type": "quantum",
        "max_qubits": 287,
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

ACTION_MEANINGS = {0: "classic", 1: "quantum", 2: "hybrid"}

OUTPUT_DIR = _PROJECT_ROOT / "results" / "real_machine" / "tianyan287_multiseed"


def create_strategy(name: str):  # type: ignore[no-untyped-def]
    """创建调度策略"""
    if name == "fcfs":
        return FCFSStrategy()
    if name == "sjf":
        return ShortestJobFirstStrategy()
    if name == "ppo":
        try:
            from stable_baselines3 import PPO

            model = PPO.load(str(PPO_MODEL_PATH))
            logger.info(f"[Strategy] PPO 模型已加载: {PPO_MODEL_PATH}")
            return PPOStrategy(model)
        except Exception as e:
            logger.warning(f"PPO 模型加载失败: {e}，使用 FCFS 替代")
            return FCFSStrategy()
    raise ValueError(f"未知策略: {name}")


def run_single_seed(
    strategy_name: str,
    seed: int,
    client: CqlibTianyanClient,
    machine_name: str,
) -> dict:
    """运行单个 seed 的单策略实验

    Args:
        strategy_name: 策略名称
        seed: 随机种子
        client: 真机客户端
        machine_name: 机器名称

    Returns:
        实验结果字典
    """
    strategy = create_strategy(strategy_name)
    logger.info(f"[Seed {seed}] 策略 {strategy.name} 开始")

    result = {
        "strategy": strategy.name,
        "seed": seed,
        "machine": machine_name,
        "timestamp": datetime.now().isoformat(),
        "metrics": {},
        "real_records": [],
    }

    start_time = time.time()

    try:
        env = QuantumSchedulingEnv(
            machine_configs=DEFAULT_MACHINE_CONFIGS,
            max_steps=min(NUM_TASKS * 3, 96),
            arrival_lambda=0.5,
            seed=seed,
            real_submit_probability=0.0,
        )
        env.attach_real_clients({machine_name: client})

        obs, _info = env.reset(seed=seed)
        total_reward = 0.0
        step = 0
        real_task_count = 0
        real_records = []

        while step < env.max_steps:
            action = strategy.select_action(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            step += 1

            # 按间隔提交真机任务
            if step % REAL_SUBMIT_INTERVAL == 0 and real_task_count < MAX_REAL_TASKS_PER_RUN:
                qcis = "H Q0\nM Q0"
                try:
                    task_id = client.submit_quantum_task(
                        qcis=qcis,
                        shots=SHOTS,
                        task_name=f"seed{seed}_{strategy_name}_{real_task_count}",
                    )
                    poll_result = client.wait_for_task(task_id, timeout=30, poll_interval=3)
                    prob = poll_result.get("probability", {}) if poll_result else {}

                    if prob:
                        p0 = prob.get("0", 0.0)
                        p1 = prob.get("1", 0.0)
                        fidelity = 1.0 - abs(p0 - 0.5) - abs(p1 - 0.5)
                        fidelity = max(0.0, min(1.0, fidelity))
                    else:
                        fidelity = None

                    real_records.append(
                        {
                            "task_id": str(task_id),
                            "step": step,
                            "probability": prob,
                            "fidelity": round(fidelity, 4) if fidelity else None,
                        }
                    )
                    real_task_count += 1
                    logger.info(
                        f"[Seed {seed}] {strategy_name} 真机任务{real_task_count}: "
                        f"fid={fidelity:.4f}"
                        if fidelity
                        else f"[Seed {seed}] 真机任务{real_task_count}"
                    )
                except Exception as e:
                    logger.warning(f"[Seed {seed}] 真机提交失败: {e}")
                    real_records.append({"error": str(e)[:100]})

            if terminated or truncated:
                break

        elapsed = time.time() - start_time
        fidelities = [r["fidelity"] for r in real_records if r.get("fidelity") is not None]

        result["metrics"] = {
            "total_reward": round(total_reward, 4),
            "total_steps": step,
            "elapsed_seconds": round(elapsed, 2),
            "real_tasks_submitted": real_task_count,
            "real_tasks_completed": len([r for r in real_records if r.get("fidelity") is not None]),
            "avg_fidelity": round(sum(fidelities) / max(len(fidelities), 1), 4)
            if fidelities
            else None,
        }
        logger.info(
            f"[Seed {seed}] {strategy_name} 完成: "
            f"reward={total_reward:.2f}, fid={result['metrics']['avg_fidelity']}, "
            f"耗时={elapsed:.1f}s"
        )

    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[Seed {seed}] {strategy_name} 失败: {e}")

    return result


def main() -> None:
    """主函数"""
    logger.info("=" * 60)
    logger.info("天衍-287 多seed真机实验")
    logger.info(
        f"配置: {len(SEEDS)} seeds × {len(STRATEGIES)} 策略 × {MAX_REAL_TASKS_PER_RUN} 真机任务"
    )
    logger.info(f"预计真机任务总数: {len(SEEDS) * len(STRATEGIES) * MAX_REAL_TASKS_PER_RUN}")
    logger.info("=" * 60)

    api_key = os.environ.get("TIANYAN_API_KEY", "")
    if not api_key:
        logger.error("未设置 TIANYAN_API_KEY")
        sys.exit(1)

    # 创建真机客户端
    client = CqlibTianyanClient(
        login_key=api_key,
        machine_name="tianyan287",
        auto_retry_machine=True,
    )
    actual_machine = getattr(client, "machine_name", "tianyan287")
    logger.info(f"真机客户端已创建: {actual_machine}")

    # 运行实验
    all_results = []
    total_start = time.time()

    for seed in SEEDS:
        for strategy_name in STRATEGIES:
            result = run_single_seed(strategy_name, seed, client, actual_machine)
            all_results.append(result)

    total_elapsed = time.time() - total_start
    logger.info(f"\n全部实验完成，总耗时: {total_elapsed:.1f}s")

    # 保存数据
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_file = OUTPUT_DIR / f"multiseed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with data_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "tianyan287_multiseed",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "seeds": SEEDS,
                    "strategies": STRATEGIES,
                    "num_tasks": NUM_TASKS,
                    "max_real_tasks_per_run": MAX_REAL_TASKS_PER_RUN,
                    "shots": SHOTS,
                },
                "total_elapsed_seconds": round(total_elapsed, 2),
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    logger.info(f"数据已保存: {data_file}")

    # 汇总表
    logger.info("\n" + "=" * 60)
    logger.info("多seed实验汇总")
    logger.info("=" * 60)
    logger.info(f"{'策略':<10} {'Seed':<6} {'奖励':<12} {'保真度':<10} {'耗时':<8}")
    logger.info("-" * 50)
    for r in all_results:
        m = r.get("metrics", {})
        logger.info(
            f"{r['strategy']:<10} {r['seed']:<6} "
            f"{m.get('total_reward', 'N/A'):<12} "
            f"{m.get('avg_fidelity', 'N/A'):<10} "
            f"{m.get('elapsed_seconds', 'N/A'):<8}"
        )

    # 按策略聚合
    logger.info("\n按策略聚合（均值±标准差）:")
    for strategy_name in STRATEGIES:
        rewards = [
            r["metrics"]["total_reward"]
            for r in all_results
            if r["strategy"].lower() == strategy_name.upper()
            or (r["strategy"].lower() == strategy_name and "total_reward" in r.get("metrics", {}))
        ]
        # 更精确匹配
        rewards = [
            r["metrics"]["total_reward"]
            for r in all_results
            if r["strategy"].upper() == strategy_name.upper()
            and "total_reward" in r.get("metrics", {})
        ]
        fids = [
            r["metrics"]["avg_fidelity"]
            for r in all_results
            if r["strategy"].upper() == strategy_name.upper()
            and r.get("metrics", {}).get("avg_fidelity")
        ]
        if rewards:
            import numpy as np

            mean_r = np.mean(rewards)
            std_r = np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0
            mean_f = np.mean(fids) if fids else 0.0
            logger.info(
                f"  {strategy_name.upper()}: "
                f"奖励={mean_r:.2f}±{std_r:.2f} (N={len(rewards)}), "
                f"保真度={mean_f:.4f} (N={len(fids)})"
            )

    logger.info(f"\n数据文件: {data_file}")


if __name__ == "__main__":
    main()
