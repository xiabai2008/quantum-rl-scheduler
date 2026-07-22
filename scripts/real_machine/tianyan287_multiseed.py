#!/usr/bin/env python
"""天衍-287 多seed真机实验（Issue #45 / PR #57）

按 Issue #58 要求重构：
- 单次正式实验只能使用一个机器（tianyan287）、一个 shots（32）、一个电路配置（H Q0/M Q0）、一个超时策略
- 10 seeds × 3策略 × 1真机任务/run = 30 个正式任务
- 冒烟上限 1，总硬上限 31
- 不得自动回退到 tianyan176、Mock 或其他机器
- 统一统计方法：Welch t-test 主分析 + 配对敏感性分析
- bonferroni_significant=false 时 judgment 必须为"不支持"

机时预算：10 seeds × 3策略 × 1真机任务 = 30个真机任务 + 1 冒烟 = 31 上限
目标：收集多seed数据，计算效应量(Cohen's d) + 95% CI

用法：
    # 冒烟（1 个真机任务，验证可用性）
    python scripts/real_machine/tianyan287_multiseed.py --smoke

    # 正式 30 个真机任务
    python scripts/real_machine/tianyan287_multiseed.py --formal
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
# 按 Issue #58：默认目标机器为 tianyan287，不得自动回退
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

# ── 实验配置（Issue #58 统一口径） ──

# 10 seeds（扩展自原 5 seeds）
SEEDS = [42, 123, 456, 789, 1024, 2025, 3141, 5678, 8765, 9999]
STRATEGIES = ["ppo", "fcfs", "sjf"]
NUM_TASKS = 32
MAX_REAL_TASKS_PER_RUN = 1
REAL_SUBMIT_INTERVAL = 5  # 每5步提交一次真机任务

# Issue #58：shots 统一为 32（原 1024 已废弃）
SHOTS = 32

# Issue #58：目标机器固定为 tianyan287，不得回退
TARGET_MACHINE = "tianyan287"

# Issue #58：提交硬上限
# 正式 30 + 冒烟 1 = 31 总硬上限
HARD_LIMIT_FORMAL = 30  # 10 seeds × 3 策略 × 1 真机任务
HARD_LIMIT_SMOKE = 1
HARD_LIMIT_TOTAL = HARD_LIMIT_FORMAL + HARD_LIMIT_SMOKE  # 31

# Issue #58：固定电路配置（1-qubit, H Q0/M Q0）
QCIS_CIRCUIT = "H Q0\nM Q0"

# Issue #58：超时统一
TASK_TIMEOUT_SECONDS = 120
TASK_POLL_INTERVAL = 5

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
    shots: int = SHOTS,
) -> dict:
    """运行单个 seed 的单策略实验

    Args:
        strategy_name: 策略名称
        seed: 随机种子
        client: 真机客户端
        machine_name: 机器名称（必须为 tianyan287）
        shots: 真机测量次数（必须为 32）

    Returns:
        实验结果字典
    """
    # Issue #58：一致性校验
    if machine_name != TARGET_MACHINE:
        raise ValueError(
            f"机器一致性违规: 期望 {TARGET_MACHINE}, 实际 {machine_name}. 不得自动回退到其他机器."
        )
    if shots != SHOTS:
        raise ValueError(f"shots 一致性违规: 期望 {SHOTS}, 实际 {shots}")

    strategy = create_strategy(strategy_name)
    logger.info(f"[Seed {seed}] 策略 {strategy.name} 开始")

    submitted_at = datetime.now().isoformat()
    result = {
        "strategy": strategy.name,
        "seed": seed,
        "machine": machine_name,
        "shots": shots,
        "circuit": QCIS_CIRCUIT,
        "submitted_at": submitted_at,
        "metrics": {},
        "real_records": [],
        # Issue #58：明确标记非 Mock/仿真
        "mock": False,
        "degraded": False,
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

            # 按间隔提交真机任务（最多 1 次/run）
            if step % REAL_SUBMIT_INTERVAL == 0 and real_task_count < MAX_REAL_TASKS_PER_RUN:
                try:
                    task_id = client.submit_quantum_task(
                        qcis=QCIS_CIRCUIT,
                        shots=shots,
                        task_name=f"seed{seed}_{strategy_name}_{real_task_count}",
                    )
                    task_submitted_at = datetime.now().isoformat()
                    poll_result = client.wait_for_task(
                        task_id, timeout=TASK_TIMEOUT_SECONDS, poll_interval=TASK_POLL_INTERVAL
                    )
                    task_completed_at = datetime.now().isoformat()
                    prob = poll_result.get("probability", {}) if poll_result else {}
                    final_status = (
                        poll_result.get("status", "unknown") if poll_result else "unknown"
                    )

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
                            "shots": shots,
                            "circuit": QCIS_CIRCUIT,
                            "submitted_at": task_submitted_at,
                            "completed_at": task_completed_at,
                            "status": final_status,
                            "probability": prob,
                            "fidelity": round(fidelity, 4) if fidelity else None,
                            "elapsed_seconds": round(
                                (
                                    datetime.fromisoformat(task_completed_at)
                                    - datetime.fromisoformat(task_submitted_at)
                                ).total_seconds(),
                                2,
                            ),
                        }
                    )
                    real_task_count += 1
                    logger.info(
                        f"[Seed {seed}] {strategy_name} 真机任务{real_task_count}: "
                        f"task_id={task_id}, status={final_status}, "
                        f"fid={fidelity:.4f}"
                        if fidelity
                        else f"[Seed {seed}] 真机任务{real_task_count}: task_id={task_id}, "
                        f"status={final_status}"
                    )
                except Exception as e:
                    logger.warning(f"[Seed {seed}] 真机提交失败: {e}")
                    real_records.append(
                        {
                            "error": str(e)[:100],
                            "submitted_at": datetime.now().isoformat(),
                            "shots": shots,
                            "circuit": QCIS_CIRCUIT,
                        }
                    )

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
        result["completed_at"] = datetime.now().isoformat()
        logger.info(
            f"[Seed {seed}] {strategy_name} 完成: "
            f"reward={total_reward:.2f}, fid={result['metrics']['avg_fidelity']}, "
            f"耗时={elapsed:.1f}s"
        )

    except Exception as e:
        result["error"] = str(e)
        result["completed_at"] = datetime.now().isoformat()
        logger.error(f"[Seed {seed}] {strategy_name} 失败: {e}")

    return result


def run_smoke_test(client: CqlibTianyanClient, machine_name: str) -> dict:
    """Issue #58：冒烟测试，验证 tianyan287 可用性。

    必须满足 task_id + completed + probability 非空 才能进入正式实验。

    Returns:
        冒烟结果字典，包含 passed / task_id / status / probability
    """
    logger.info("=" * 60)
    logger.info("冒烟测试：验证 tianyan287 可用性")
    logger.info(f"  电路: {QCIS_CIRCUIT!r}")
    logger.info(f"  shots: {SHOTS}")
    logger.info(f"  超时: {TASK_TIMEOUT_SECONDS}s")
    logger.info("=" * 60)

    smoke_result = {
        "smoke_test": True,
        "machine": machine_name,
        "shots": SHOTS,
        "circuit": QCIS_CIRCUIT,
        "submitted_at": datetime.now().isoformat(),
        "task_id": None,
        "status": None,
        "probability": None,
        "passed": False,
        "error": None,
    }

    try:
        task_id = client.submit_quantum_task(
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="smoke_test",
        )
        smoke_result["task_id"] = str(task_id)
        logger.info(f"  冒烟 task_id: {task_id}")

        poll_result = client.wait_for_task(
            task_id, timeout=TASK_TIMEOUT_SECONDS, poll_interval=TASK_POLL_INTERVAL
        )
        smoke_result["completed_at"] = datetime.now().isoformat()

        if not poll_result:
            smoke_result["error"] = "poll_result 为空"
            logger.error("  ❌ 冒烟失败：poll_result 为空")
            return smoke_result

        status = poll_result.get("status", "unknown")
        prob = poll_result.get("probability", {})

        smoke_result["status"] = status
        smoke_result["probability"] = prob

        # Issue #58：必须同时满足 task_id + completed + probability 非空
        passed = (
            task_id is not None and status == "completed" and prob is not None and len(prob) > 0
        )
        smoke_result["passed"] = passed

        if passed:
            logger.info(f"  ✅ 冒烟通过: status={status}, probability={prob}")
        else:
            logger.error(
                f"  ❌ 冒烟失败: status={status}, probability={prob}. "
                f"按 Issue #58：禁止正式 30 次提交，保留失败 pilot 记录。"
            )

    except Exception as e:
        smoke_result["error"] = str(e)[:200]
        smoke_result["completed_at"] = datetime.now().isoformat()
        logger.error(f"  ❌ 冒烟异常: {e}")

    return smoke_result


def main() -> None:
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="天衍-287 多seed真机实验（Issue #45/#58）")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=SEEDS,
        help=f"随机种子列表（默认: {SEEDS}）",
    )
    parser.add_argument(
        "--machine",
        type=str,
        default=TARGET_MACHINE,
        help=f"目标真机名称（默认: {TARGET_MACHINE}，不得回退）",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=SHOTS,
        help=f"真机测量次数（默认: {SHOTS}）",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="冒烟模式（1 个真机任务，验证可用性）",
    )
    mode.add_argument(
        "--formal",
        action="store_true",
        help="正式模式（10 seeds × 3 策略 × 1 真机任务 = 30 次）",
    )
    args = parser.parse_args()

    seeds = args.seeds
    machine = args.machine
    shots = args.shots

    # Issue #58：一致性校验
    if machine != TARGET_MACHINE:
        logger.error(f"机器一致性违规: 期望 {TARGET_MACHINE}, 实际 {machine}")
        sys.exit(1)
    if shots != SHOTS:
        logger.error(f"shots 一致性违规: 期望 {SHOTS}, 实际 {shots}")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("天衍-287 多seed真机实验（Issue #45/#58 口径）")
    logger.info(f"  模式: {'冒烟' if args.smoke else '正式'}")
    logger.info(
        f"  配置: {len(seeds)} seeds × {len(STRATEGIES)} 策略 × {MAX_REAL_TASKS_PER_RUN} 真机任务"
    )
    logger.info(f"  预计真机任务总数: {len(seeds) * len(STRATEGIES) * MAX_REAL_TASKS_PER_RUN}")
    logger.info(f"  目标机器: {machine}（固定，不得回退）")
    logger.info(f"  shots: {shots}（固定为 32）")
    logger.info(f"  电路: {QCIS_CIRCUIT!r}")
    logger.info(
        f"  硬上限: 正式 {HARD_LIMIT_FORMAL}, 冒烟 {HARD_LIMIT_SMOKE}, 总 {HARD_LIMIT_TOTAL}"
    )
    logger.info("=" * 60)

    api_key = os.environ.get("TIANYAN_API_KEY", "")
    if not api_key:
        logger.error("未设置 TIANYAN_API_KEY")
        sys.exit(1)

    # 创建真机客户端（关闭自动切换，避免回退到其他机器）
    client = CqlibTianyanClient(
        login_key=api_key,
        machine_name=machine,
        auto_retry_machine=False,
    )
    actual_machine = getattr(client, "machine_name", machine)
    if actual_machine != machine:
        logger.error(
            f"❌ 客户端机器不一致: 请求 {machine}, 实际 {actual_machine}. "
            f"按 Issue #58：禁止回退，停止执行。"
        )
        sys.exit(1)
    logger.info(f"真机客户端已创建: {actual_machine}")

    # Issue #58：先冒烟，通过后才正式
    if args.smoke:
        smoke_result = run_smoke_test(client, actual_machine)
        smoke_file = OUTPUT_DIR / f"smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with smoke_file.open("w", encoding="utf-8") as f:
            json.dump(smoke_result, f, ensure_ascii=False, indent=2, default=str)
        logger.info(f"冒烟结果已保存: {smoke_file}")
        if not smoke_result["passed"]:
            logger.error(
                "❌ 冒烟未通过。按 Issue #58：禁止正式 30 次提交，"
                "保留失败 pilot 记录，PR #57 停止。"
            )
            sys.exit(1)
        logger.info("✅ 冒烟通过，可执行 --formal 正式实验")
        return

    # Issue #58：正式实验前先执行冒烟（防止浪费配额）
    logger.info("正式实验前先执行冒烟验证...")
    smoke_result = run_smoke_test(client, actual_machine)
    if not smoke_result["passed"]:
        smoke_file = (
            OUTPUT_DIR / f"smoke_pre_formal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        with smoke_file.open("w", encoding="utf-8") as f:
            json.dump(smoke_result, f, ensure_ascii=False, indent=2, default=str)
        logger.error(
            f"❌ 冒烟未通过: status={smoke_result.get('status')}, "
            f"prob={smoke_result.get('probability')}. "
            f"按 Issue #58：禁止正式 30 次提交，保留失败 pilot 记录，PR #57 停止。"
        )
        sys.exit(1)
    logger.info("✅ 冒烟通过，开始正式 30 次提交")

    # 运行正式实验
    all_results = [smoke_result]  # 包含冒烟结果作为第一条记录
    total_start = time.time()
    submitted_count = 1  # 冒烟已用 1 次配额

    for seed in seeds:
        for strategy_name in STRATEGIES:
            if submitted_count >= HARD_LIMIT_TOTAL:
                logger.error(
                    f"已达硬上限 {HARD_LIMIT_TOTAL}，停止提交。已提交 {submitted_count} 次。"
                )
                break
            result = run_single_seed(strategy_name, seed, client, actual_machine, shots)
            all_results.append(result)
            submitted_count += 1
        if submitted_count >= HARD_LIMIT_TOTAL:
            break

    total_elapsed = time.time() - total_start
    logger.info(f"\n全部实验完成，总耗时: {total_elapsed:.1f}s")

    # 保存数据
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data_file = OUTPUT_DIR / f"multiseed_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with data_file.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "tianyan287_multiseed_10seeds",
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "seeds": seeds,
                    "strategies": STRATEGIES,
                    "num_tasks": NUM_TASKS,
                    "max_real_tasks_per_run": MAX_REAL_TASKS_PER_RUN,
                    "shots": shots,
                    "machine": actual_machine,
                    "circuit": QCIS_CIRCUIT,
                    "hard_limit_total": HARD_LIMIT_TOTAL,
                    "smoke_passed": smoke_result["passed"],
                    "unified_protocol": True,  # Issue #58 统一口径
                },
                "total_elapsed_seconds": round(total_elapsed, 2),
                "total_submitted": submitted_count,
                "results": all_results,
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
    logger.info(f"数据已保存: {data_file}")

    # 汇总表
    logger.info("\n" + "=" * 60)
    logger.info("多seed实验汇总")
    logger.info("=" * 60)
    logger.info(f"{'策略':<10} {'Seed':<6} {'奖励':<12} {'保真度':<10} {'耗时':<8}")
    logger.info("-" * 50)
    for r in all_results:
        if r.get("smoke_test"):
            continue  # 跳过冒烟记录
        m = r.get("metrics", {})
        reward = m.get("total_reward")
        fid = m.get("avg_fidelity")
        elapsed = m.get("elapsed_seconds")
        logger.info(
            f"{r.get('strategy', 'N/A'):<10} {r.get('seed', 'N/A'):<6} "
            f"{(reward if reward is not None else 'N/A')!s:<12} "
            f"{(fid if fid is not None else 'N/A')!s:<10} "
            f"{(elapsed if elapsed is not None else 'N/A')!s:<8}"
        )

    # 按策略聚合
    logger.info("\n按策略聚合（均值±标准差）:")
    for strategy_name in STRATEGIES:
        rewards = [
            r["metrics"]["total_reward"]
            for r in all_results
            if not r.get("smoke_test")
            and r.get("strategy", "").upper() == strategy_name.upper()
            and "total_reward" in r.get("metrics", {})
        ]
        fids = [
            r["metrics"]["avg_fidelity"]
            for r in all_results
            if not r.get("smoke_test")
            and r.get("strategy", "").upper() == strategy_name.upper()
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
