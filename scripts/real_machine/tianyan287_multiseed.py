#!/usr/bin/env python
"""天衍-287 多seed真机实验（Issue #45 / PR #57）

按 Issue #58 要求重构：
- 单次正式实验只能使用一个机器（tianyan-287）、一个 shots（32）、一个电路配置（H Q1/M Q1）、一个超时策略
- 10 seeds × 3策略 × 1真机任务/run = 30 个正式任务
- 冒烟上限 1，总硬上限 31
- 不得自动回退到 tianyan176、Mock 或其他机器
- 统一统计方法：Welch t-test 主分析 + 配对敏感性分析
- bonferroni_significant=false 时 judgment 必须为"不支持"

已核实事实（cqlib 1.3.11）：
- 正确后端代码：tianyan-287（有连字符）；tianyan287 不存在
- 天衍-287 物理比特 Q1～Q105，没有 Q0
- H Q0/M Q0 QCIS 校验 false；H Q1/M Q1 QCIS 校验 true
- get_task_status() 概率字段名为 result，非 probability
- 历史失败 task mapQcis/computerQcis 均为 null，属编译映射失败

机时预算：10 seeds × 3策略 × 1真机任务 = 30个真机任务 + 1 冒烟 = 31 上限
目标：收集多seed数据，计算效应量(Cohen's d) + 95% CI

用法：
    # 冒烟（1 个真机任务，验证可用性）
    python scripts/real_machine/tianyan287_multiseed.py --smoke

    # 正式 30 个真机任务
    python scripts/real_machine/tianyan287_multiseed.py --formal
"""

import contextlib
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

# 注意：不得在模块导入阶段修改 os.environ（会导致 pytest 进程环境污染，
# 使 tests/test_api.py::test_explicit_mock_mode_false 失败）。
# TIANYAN_API_KEY / TIANYAN_MOCK_MODE / TIANYAN_MACHINE 的设置只放在 main() 中，
# 或由调用者显式传参给 CqlibTianyanClient（见 main() 的 login_key/machine_name）。

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

# 已核实：shots 统一为 32
SHOTS = 32

# 已核实：正确后端代码为 tianyan-287（有连字符），不得回退
TARGET_MACHINE = "tianyan-287"

# 已核实：提交硬上限
# 正式 30 + 冒烟 1 = 31 总硬上限
HARD_LIMIT_FORMAL = 30  # 10 seeds × 3 策略 × 1 真机任务
HARD_LIMIT_SMOKE = 1
HARD_LIMIT_TOTAL = HARD_LIMIT_FORMAL + HARD_LIMIT_SMOKE  # 31

# 已核实：天衍-287 物理比特 Q1～Q105，没有 Q0
# H Q0/M Q0 QCIS 校验 false；H Q1/M Q1 QCIS 校验 true
QCIS_CIRCUIT = "H Q1\nM Q1"

# 已核实：超时统一（120→180，适应排队高峰）
TASK_TIMEOUT_SECONDS = 180
TASK_POLL_INTERVAL = 5

PPO_MODEL_PATH = _PROJECT_ROOT / "deliverable_models" / "ppo_best_model_14dim.zip"

DEFAULT_MACHINE_CONFIGS = [
    {
        "name": "tianyan-287",
        "machine_type": "quantum",
        # 已核实：天衍-287 当前配置 105 个物理比特，不得因名称写 287
        "max_qubits": 105,
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
        machine_name: 机器名称（必须为 tianyan-287）
        shots: 真机测量次数（必须为 32）

    Returns:
        实验结果字典
    """
    # 已核实：一致性校验
    if machine_name != TARGET_MACHINE:
        raise ValueError(
            f"机器一致性违规: 期望 {TARGET_MACHINE}, 实际 {machine_name}. 不得自动回退到其他机器."
        )
    if shots != SHOTS:
        raise ValueError(f"shots 一致性违规: 期望 {SHOTS}, 实际 {shots}")

    strategy = create_strategy(strategy_name)
    logger.info(f"[Seed {seed}] 策略 {strategy.name} 开始")

    result = {
        "strategy": strategy.name,
        "seed": seed,
        "machine": machine_name,
        "shots": shots,
        "circuit": QCIS_CIRCUIT,
        "metrics": {},
        "real_records": [],
        # 明确标记非 Mock/仿真
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
        real_tasks_submitted = 0  # 实际获得 task_id 的真机提交数
        real_tasks_completed = 0  # status=completed 且有合法概率结果
        real_tasks_failed = 0  # 明确失败
        real_tasks_timeout = 0  # 超时
        real_tasks_query_error = 0  # SDK 查询异常

        while step < env.max_steps:
            action = strategy.select_action(obs)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            step += 1

            # 按间隔提交真机任务（最多 1 次/run）
            if step % REAL_SUBMIT_INTERVAL == 0 and real_tasks_submitted < MAX_REAL_TASKS_PER_RUN:
                record = _submit_and_poll_one_task(
                    client=client,
                    qcis=QCIS_CIRCUIT,
                    shots=shots,
                    task_name=f"seed{seed}_{strategy_name}_{real_tasks_submitted}",
                    machine_name=machine_name,
                )
                record["step"] = step
                result["real_records"].append(record)

                # 按终态分类计数
                if record.get("task_id"):
                    real_tasks_submitted += 1
                if record["status"] == "completed" and record.get("probability"):
                    real_tasks_completed += 1
                elif record["status"] == "failed":
                    real_tasks_failed += 1
                elif record["status"] == "timeout":
                    real_tasks_timeout += 1
                elif record["status"] == "query_error":
                    real_tasks_query_error += 1

            if terminated or truncated:
                break

        elapsed = time.time() - start_time
        scores = [
            r["measurement_balance_score"]
            for r in result["real_records"]
            if r.get("measurement_balance_score") is not None
        ]

        result["metrics"] = {
            "total_reward": round(total_reward, 4),
            "total_steps": step,
            "elapsed_seconds": round(elapsed, 2),
            "real_tasks_submitted": real_tasks_submitted,
            "real_tasks_completed": real_tasks_completed,
            "real_tasks_failed": real_tasks_failed,
            "real_tasks_timeout": real_tasks_timeout,
            "real_tasks_query_error": real_tasks_query_error,
            "avg_measurement_balance_score": round(sum(scores) / max(len(scores), 1), 4)
            if scores
            else None,
        }
        result["completed_at"] = datetime.now().isoformat()
        logger.info(
            f"[Seed {seed}] {strategy_name} 完成: "
            f"reward={total_reward:.2f}, "
            f"submitted={real_tasks_submitted}, completed={real_tasks_completed}, "
            f"耗时={elapsed:.1f}s"
        )

    except Exception as e:
        result["error"] = str(e)
        result["completed_at"] = datetime.now().isoformat()
        logger.error(f"[Seed {seed}] {strategy_name} 失败: {e}")

    return result


def _submit_and_poll_one_task(
    client: CqlibTianyanClient,
    qcis: str,
    shots: int,
    task_name: str,
    machine_name: str,
) -> dict:
    """提交并轮询单个真机任务，保留 task_id 即使轮询失败。

    已核实修复：
    - 提交前调用 qcis_check_regular 校验，false 立即终止
    - task_id 为 None 时立即判定失败，不调用 wait_for_task(None)
    - probability 从 poll_result["result"] 读取（非 probability 字段）
    - 获得非空 task_id 后立即记录 submitted
    - 轮询失败也保留 task_id
    - status 分类: completed/failed/timeout/query_error
    - measurement_balance_score 衡量 H 态测量分布接近 50/50 的程度（非完整保真度）
    """
    record: dict = {
        "task_id": None,
        "step": None,
        "shots": shots,
        "circuit": qcis,
        "machine": machine_name,
        "submitted_at": None,
        "completed_at": None,
        "status": None,
        "probability": None,
        "measurement_balance_score": None,
        "elapsed_seconds": None,
        "error": None,
        "mock": False,
        "degraded": False,
    }

    # 已核实：提交前 QCIS 预校验
    try:
        platform = getattr(client, "platform", None)
        if platform is not None and hasattr(platform, "qcis_check_regular"):
            qcis_valid = platform.qcis_check_regular(qcis)
            if not qcis_valid:
                record["status"] = "failed"
                record["error"] = "QCIS 预校验失败（qcis_check_regular 返回 false）"
                logger.error(f"  ❌ QCIS 预校验失败: {qcis!r}，零提交")
                return record
    except Exception as e:
        record["status"] = "query_error"
        record["error"] = f"QCIS 校验异常: {str(e)[:100]}"
        logger.warning(f"  ⚠️ QCIS 校验异常: {e}，跳过提交")
        return record

    # 提交任务
    try:
        task_id = client.submit_quantum_task(
            qcis=qcis,
            shots=shots,
            task_name=task_name,
        )
    except Exception as e:
        record["status"] = "failed"
        record["error"] = f"提交异常: {str(e)[:150]}"
        record["submitted_at"] = datetime.now().isoformat()
        logger.warning(f"  ❌ 提交异常: {e}")
        return record

    # 已核实：task_id 为 None 时立即失败，不得调用 wait_for_task(None)
    if task_id is None or (isinstance(task_id, str) and not task_id.strip()):
        record["status"] = "failed"
        record["error"] = "submit_quantum_task 返回 None（全部机器不可用）"
        record["submitted_at"] = datetime.now().isoformat()
        logger.error("  ❌ 提交失败：task_id 为 None，不轮询")
        return record

    # 已核实：获得非空 task_id 后立即记录 submitted
    record["task_id"] = str(task_id)
    record["submitted_at"] = datetime.now().isoformat()
    logger.info(f"  ✅ task_id 已获得: {task_id}（已记录 submitted）")

    # 轮询等待结果
    try:
        poll_result = client.wait_for_task(
            task_id, timeout=TASK_TIMEOUT_SECONDS, poll_interval=TASK_POLL_INTERVAL
        )
    except Exception as e:
        # 轮询异常：保留 task_id，标记 query_error
        record["status"] = "query_error"
        record["error"] = f"轮询异常: {str(e)[:150]}"
        record["completed_at"] = datetime.now().isoformat()
        logger.warning(f"  ⚠️ 轮询异常（task_id={task_id} 保留）: {e}")
        return record

    record["completed_at"] = datetime.now().isoformat()

    if not poll_result:
        record["status"] = "timeout"
        record["error"] = "wait_for_task 返回空"
        logger.warning(f"  ⚠️ task_id={task_id} 轮询返回空")
        return record

    final_status = poll_result.get("status", "unknown")
    record["status"] = final_status

    # 已核实：probability 从 result 字段读取（get_task_status 返回的字典字段名是 result）
    prob = poll_result.get("result")
    if prob is None:
        # 兼容：部分实现可能在 probability 字段
        prob = poll_result.get("probability")
    # 已核实修复：cqlib SDK 可能返回 JSON 字符串而非 dict
    # 需要解析后才能用于 isinstance(prob, dict) 判断和 score 计算
    if isinstance(prob, str) and prob:
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            prob = json.loads(prob)
    record["probability"] = prob

    if final_status == "completed" and prob and isinstance(prob, dict):
        p0 = prob.get("0", 0.0)
        p1 = prob.get("1", 0.0)
        # measurement_balance_score：H 态测量分布接近 50/50 的分数
        # 不是完整量子态保真度，仅衡量测量分布平衡度
        score = 1.0 - abs(p0 - 0.5) - abs(p1 - 0.5)
        score = max(0.0, min(1.0, score))
        record["measurement_balance_score"] = round(score, 4)
    elif final_status == "timeout":
        record["error"] = "任务超时"
    elif final_status == "error":
        record["error"] = poll_result.get("error", "任务执行错误")
    elif final_status != "completed":
        record["error"] = f"非终态完成: status={final_status}"

    # 计算 elapsed_seconds
    if record["submitted_at"] and record["completed_at"]:
        with contextlib.suppress(ValueError, TypeError):
            record["elapsed_seconds"] = round(
                (
                    datetime.fromisoformat(record["completed_at"])
                    - datetime.fromisoformat(record["submitted_at"])
                ).total_seconds(),
                2,
            )

    log_msg = (
        f"  task_id={task_id}, status={final_status}, "
        f"prob={prob}, score={record['measurement_balance_score']}"
    )
    if record["measurement_balance_score"] is not None:
        logger.info(f"  ✅ 真机完成: {log_msg}")
    else:
        logger.warning(f"  ⚠️ 真机未完成: {log_msg}")

    return record


def run_smoke_test(client: CqlibTianyanClient, machine_name: str) -> dict:
    """已核实：冒烟测试，验证 tianyan-287 可用性。

    必须同时满足以下条件才能进入正式实验：
    - 后端精确等于 tianyan-287
    - QCIS 预校验通过
    - task_id 非空
    - status=completed
    - probability 为非空字典
    - mock=false
    - degraded=false

    Returns:
        冒烟结果字典，包含 passed / task_id / status / probability
    """
    logger.info("=" * 60)
    logger.info("冒烟测试：验证 tianyan-287 可用性")
    logger.info(f"  电路: {QCIS_CIRCUIT!r}")
    logger.info(f"  shots: {SHOTS}")
    logger.info(f"  超时: {TASK_TIMEOUT_SECONDS}s")
    logger.info(f"  机器: {machine_name}（必须为 {TARGET_MACHINE}）")
    logger.info("=" * 60)

    smoke_result = {
        "smoke_test": True,
        "machine": machine_name,
        "shots": SHOTS,
        "circuit": QCIS_CIRCUIT,
        "task_id": None,
        "submitted_at": None,
        "completed_at": None,
        "status": None,
        "probability": None,
        "measurement_balance_score": None,
        "mock": False,
        "degraded": False,
        "passed": False,
        "error": None,
    }

    # 已核实：后端一致性校验
    if machine_name != TARGET_MACHINE:
        smoke_result["error"] = f"后端不一致: 期望 {TARGET_MACHINE}, 实际 {machine_name}"
        smoke_result["status"] = "failed"
        logger.error(f"  ❌ {smoke_result['error']}")
        return smoke_result

    # 复用统一的提交+轮询逻辑
    record = _submit_and_poll_one_task(
        client=client,
        qcis=QCIS_CIRCUIT,
        shots=SHOTS,
        task_name="smoke_test",
        machine_name=machine_name,
    )

    # 合并 record 到 smoke_result
    for key in (
        "task_id",
        "submitted_at",
        "completed_at",
        "status",
        "probability",
        "measurement_balance_score",
        "error",
    ):
        if record.get(key) is not None:
            smoke_result[key] = record[key]

    # 已核实：冒烟通过条件必须同时满足
    passed = (
        machine_name == TARGET_MACHINE
        and record.get("task_id") is not None
        and record.get("status") == "completed"
        and isinstance(record.get("probability"), dict)
        and len(record.get("probability", {})) > 0
        and record.get("mock") is False
        and record.get("degraded") is False
    )
    smoke_result["passed"] = passed

    if passed:
        logger.info(
            f"  ✅ 冒烟通过: task_id={record.get('task_id')}, "
            f"status={record.get('status')}, probability={record.get('probability')}"
        )
    else:
        logger.error(
            f"  ❌ 冒烟失败: task_id={record.get('task_id')}, "
            f"status={record.get('status')}, probability={record.get('probability')}. "
            f"禁止正式 30 次提交，保留失败 pilot 记录。"
        )

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
                "experiment": "tianyan-287_multiseed_10seeds",
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
    logger.info(f"{'策略':<10} {'Seed':<6} {'奖励':<12} {'测量平衡分':<12} {'耗时':<8}")
    logger.info("-" * 50)
    for r in all_results:
        if r.get("smoke_test"):
            continue  # 跳过冒烟记录
        m = r.get("metrics", {})
        reward = m.get("total_reward")
        score = m.get("avg_measurement_balance_score")
        elapsed = m.get("elapsed_seconds")
        logger.info(
            f"{r.get('strategy', 'N/A'):<10} {r.get('seed', 'N/A'):<6} "
            f"{(reward if reward is not None else 'N/A')!s:<12} "
            f"{(score if score is not None else 'N/A')!s:<12} "
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
        scores = [
            r["metrics"]["avg_measurement_balance_score"]
            for r in all_results
            if not r.get("smoke_test")
            and r.get("strategy", "").upper() == strategy_name.upper()
            and r.get("metrics", {}).get("avg_measurement_balance_score")
        ]
        if rewards:
            import numpy as np

            mean_r = np.mean(rewards)
            std_r = np.std(rewards, ddof=1) if len(rewards) > 1 else 0.0
            mean_s = np.mean(scores) if scores else 0.0
            logger.info(
                f"  {strategy_name.upper()}: "
                f"奖励={mean_r:.2f}±{std_r:.2f} (N={len(rewards)}), "
                f"测量平衡分={mean_s:.4f} (N={len(scores)})"
            )

    logger.info(f"\n数据文件: {data_file}")


if __name__ == "__main__":
    main()
