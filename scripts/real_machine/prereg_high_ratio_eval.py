#!/usr/bin/env python
"""
真机高占比调度闭环评估脚本（实验①，预注册：docs/prereg_real_machine_high_ratio_20260814.md）

目的：
    把 real_submit_probability 提到 0.5/0.8 主导档位，验证"AI 调度决策在真机奖励
    维度上优于基线"（H1）与"真机参与度占比"（H2 代理指标）。
    解决 v2 权威实验"真机 reward 仅占 1/96 步"的评委质疑。

设计（与 v2 权威协议对齐，除 prob/steps 外零差异）：
    - 机器: tianyan-287（带连字符，强校验不得回退）
    - shots: 32（强校验，与 v2 一致）
    - 电路: "H Q1\nM Q1"（QCIS 预校验）
    - 反馈模式: result_aware（按真机测量分布计算 reward，与 v2 一致）
    - 提交机制: env 原生概率提交 + 每步轮询（run_15seeds_multistrategy.py 同机制，
      非 v2 的显式同步等待——高占比下同步会阻塞）
    - 策略: PPO(ppo_best_model_16dim.zip) / FCFS / SJF（run_simulation 策略类）
    - 配对: 同 seed 下各策略各跑 1 episode（调用方按 seed×strategy 串行/并行编排）

纪律：
    - 本脚本是实验专用评估脚本，不修改 src/ 任何行为或默认参数
    - 机时保护：--max-real-tasks 默认 = ceil(steps×prob)+1，可显式收紧
    - 结果无论正负照常入库（预注册诚实披露条款）

用法：
    # 干跑（仿真，验证流程）
    python scripts/real_machine/prereg_high_ratio_eval.py --mock --seeds 42 --prob 0.5 --steps 20

    # 冒烟（1 个真机任务验证可用性）
    python scripts/real_machine/prereg_high_ratio_eval.py --smoke

    # 正式：单档位单策略（推荐按 seed×策略 分批跑，便于中断续跑）
    python scripts/real_machine/prereg_high_ratio_eval.py \
        --seeds 42 123 456 789 1024 2025 3141 5678 8765 9999 \
        --prob 0.5 --strategy ppo --steps 100 --shots 32

    # 正式：多策略（--strategy 可多次传）
    python scripts/real_machine/prereg_high_ratio_eval.py \
        --seeds 42 123 456 789 1024 --prob 0.8 \
        --strategy ppo --strategy fcfs --steps 100
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Windows GBK 终端安全输出
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
_EVAL_DIR = _PROJECT_ROOT / "scripts" / "evaluation"
if str(_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(_EVAL_DIR))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from loguru import logger

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.scheduler.env import QuantumSchedulingEnv

# ── 协议常量（与 v2 权威 tianyan287_multiseed.py 对齐） ──
TARGET_MACHINE = "tianyan-287"
SHOTS_FIXED = 32
QCIS_CIRCUIT = "H Q1\nM Q1"
TASK_TIMEOUT_SECONDS = 180
TASK_POLL_INTERVAL = 5
ARRIVAL_LAMBDA = 0.5

# tianyan-287 + 经典机（v2 同款配置）
MACHINE_CONFIGS = [
    {
        "name": "tianyan-287",
        "machine_type": "quantum",
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

OUTPUT_DIR = _PROJECT_ROOT / "results" / "real_machine"


def create_strategy(name: str):  # type: ignore[no-untyped-def]
    """创建调度策略（run_simulation 策略类，与 v2 一致）。"""
    from run_simulation import FCFSStrategy, PPOStrategy, ShortestJobFirstStrategy

    if name == "fcfs":
        return FCFSStrategy()
    if name == "sjf":
        return ShortestJobFirstStrategy()
    if name == "ppo":
        from stable_baselines3 import PPO

        model = PPO.load(str(_PROJECT_ROOT / "deliverable_models" / "ppo_best_model_16dim.zip"))
        logger.info("[Strategy] PPO 模型已加载: deliverable_models/ppo_best_model_16dim.zip")
        return PPOStrategy(model)
    raise ValueError(f"未知策略: {name}")


def run_smoke(client: CqlibTianyanClient, machine_name: str) -> dict[str, Any]:
    """冒烟：1 个真机任务验证可用性（v2 同协议）。"""
    print("[Smoke] 提交 1 个真机任务 (H Q1/M Q1, shots=32)...")
    try:
        platform = getattr(client, "platform", None)
        if (
            platform is not None
            and hasattr(platform, "qcis_check_regular")
            and not platform.qcis_check_regular(QCIS_CIRCUIT)
        ):
            return {"passed": False, "error": "QCIS 预校验失败"}
        task_id = client.submit_quantum_task(
            qcis=QCIS_CIRCUIT, shots=SHOTS_FIXED, task_name="prereg_high_ratio_smoke"
        )
        if task_id is None:
            return {"passed": False, "error": "submit 返回 None（机器不可用）"}
        # 2026-08-14 修复：CqlibTianyanClient.wait_for_task 签名是
        # (task_id, timeout=300, poll_interval=5)，此前误用 max_wait_time/sleep_time
        # 导致 TypeError（任务已提交但等待失败）。另外平台时序：任务完成前查询返回
        # query_error，wait_for_task 连续 3 次即终止（Issue #407）——冒烟若因时序
        # 失败，用 patch_query_error_results.py 按 task_id 复查补录判定。
        result = client.wait_for_task(
            task_id, timeout=TASK_TIMEOUT_SECONDS, poll_interval=TASK_POLL_INTERVAL
        )
        ok = result.status == "completed" and bool(result.probability)
        return {
            "passed": ok,
            "task_id": str(task_id),
            "status": result.status,
            "probability": result.probability if ok else None,
            "mock": False,
        }
    except Exception as e:
        return {"passed": False, "error": str(e)[:200]}


def run_episode(
    env: QuantumSchedulingEnv, strategy: Any, seed: int, policy_name: str
) -> dict[str, Any]:
    """单 seed 单策略 episode：env 原生概率提交真机任务 + 每步轮询。"""
    obs, _info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        action = strategy.select_action(obs)
        obs, reward, terminated, truncated, _info = env.step(action)
        total_reward += float(reward)
        steps += 1
        done = terminated or truncated
        if steps >= env.max_steps:
            break

    # 真机提交记录（env._real_feedback_log：含 task_id/real_task_id/status/fidelity/machine）
    real_records: list[dict[str, Any]] = []
    if hasattr(env, "_real_feedback_log"):
        real_records = list(env._real_feedback_log)

    submitted = len([r for r in real_records if r.get("real_task_id")])
    completed = len([r for r in real_records if r.get("status") == "completed"])
    failed = len(
        [r for r in real_records if r.get("status") in ("failed", "timeout", "query_error")]
    )

    return {
        "seed": seed,
        "strategy": policy_name,
        "total_reward": round(total_reward, 4),
        "steps": steps,
        "real_tasks_submitted": submitted,
        "real_tasks_completed": completed,
        "real_tasks_failed": failed,
        # H2 代理指标：真机任务参与度（提交数/步数）——真机 reward 占比的直接度量
        # 在 env 层无累计字段，用参与度代理并在报告中注明（预注册口径）
        "real_task_share": round(submitted / steps, 4) if steps > 0 else 0.0,
        "real_records": [
            {
                "task_id": r.get("task_id"),
                "real_task_id": r.get("real_task_id"),
                "status": r.get("status"),
                "fidelity": r.get("fidelity"),
                "measurement_balance_score": r.get("measurement_balance_score"),
                "machine": r.get("machine"),
                "step": r.get("step"),
            }
            for r in real_records
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="真机高占比调度闭环评估（实验①，预注册）")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42], help="种子列表")
    parser.add_argument(
        "--prob", type=float, default=0.5, help="real_submit_probability 档位（0.15/0.5/0.8）"
    )
    parser.add_argument(
        "--strategy",
        action="append",
        choices=["ppo", "fcfs", "sjf"],
        default=None,
        help="策略（可多次传；默认 ppo）",
    )
    parser.add_argument("--steps", type=int, default=100, help="episode 最大步数（默认 100）")
    parser.add_argument(
        "--shots", type=int, default=SHOTS_FIXED, help=f"真机测量次数（默认 {SHOTS_FIXED}，强校验）"
    )
    parser.add_argument(
        "--machine",
        type=str,
        default=TARGET_MACHINE,
        help=f"目标真机（默认 {TARGET_MACHINE}，不得回退）",
    )
    parser.add_argument(
        "--max-real-tasks",
        type=int,
        default=None,
        help="每 run 真机任务上限（默认 ceil(steps×prob)+1 机时保护）",
    )
    parser.add_argument("--smoke", action="store_true", help="冒烟模式（1 个真机任务）")
    parser.add_argument("--mock", action="store_true", help="干跑（仿真模式，不消耗机时）")
    parser.add_argument(
        "--output", type=Path, default=None, help="输出 JSON 路径（默认自动时间戳）"
    )
    args = parser.parse_args()
    # append action + 非 None default 会把 default 复制进结果（argparse 语义陷阱），
    # 用 None 默认并在解析后归一
    args.strategy = args.strategy or ["ppo"]

    # ── 协议强校验（v2 纪律） ──
    if args.machine != TARGET_MACHINE:
        print(f"❌ 机器一致性违规: 期望 {TARGET_MACHINE}, 实际 {args.machine}（不得回退）")
        return 1
    if args.shots != SHOTS_FIXED:
        print(f"❌ shots 一致性违规: 期望 {SHOTS_FIXED}, 实际 {args.shots}")
        return 1
    if not (0.0 < args.prob <= 1.0):
        print(f"❌ prob 非法: {args.prob}")
        return 1

    max_real_tasks = args.max_real_tasks or (math.ceil(args.steps * args.prob) + 1)

    # ── 客户端 ──
    if args.mock:
        print("[Mode] 干跑（仿真）")
        client = None
    elif args.smoke:
        api_key = os.environ.get("TIANYAN_API_KEY", "")
        if not api_key:
            print("❌ 未设置 TIANYAN_API_KEY")
            return 1
        client = CqlibTianyanClient(
            login_key=api_key, machine_name=args.machine, auto_retry_machine=False
        )
        result = run_smoke(client, args.machine)
        print(f"冒烟结果: {json.dumps(result, ensure_ascii=False, indent=2)}")
        out = (
            OUTPUT_DIR / f"prereg_high_ratio_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        out.write_text(
            json.dumps(
                {"experiment": "prereg_high_ratio_smoke", "result": result},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"已保存: {out}")
        return 0 if result["passed"] else 1
    else:
        api_key = os.environ.get("TIANYAN_API_KEY", "")
        if not api_key:
            print("❌ 未设置 TIANYAN_API_KEY")
            return 1
        client = CqlibTianyanClient(
            login_key=api_key, machine_name=args.machine, auto_retry_machine=False
        )
        if getattr(client, "machine_name", args.machine) != args.machine:
            print("❌ 客户端机器不一致，停止执行（禁止回退）")
            return 1
        print(f"[Setup] 真机客户端已创建: {args.machine}")

    print("=== 实验① 真机高占比闭环 ===")
    print(
        f"  档位 prob={args.prob} | seeds={len(args.seeds)} | 策略={args.strategy} | "
        f"steps={args.steps} | shots={args.shots} | max_real_tasks/run={max_real_tasks}"
    )
    total_budget = len(args.seeds) * len(args.strategy) * max_real_tasks
    print(f"  ⚠ 机时预算（上限）: ~{total_budget} 个真机任务")

    # ── 执行 ──
    all_results: list[dict[str, Any]] = []
    for seed in args.seeds:
        for strategy_name in args.strategy:
            print(f"  [{seed=} {strategy_name=}] 开始...", flush=True)
            try:
                env = QuantumSchedulingEnv(
                    machine_configs=MACHINE_CONFIGS,
                    max_steps=args.steps,
                    arrival_lambda=ARRIVAL_LAMBDA,
                    seed=seed,
                    real_submit_probability=args.prob,
                    max_real_submissions=max_real_tasks,
                    real_machine_shots=args.shots,
                    real_feedback_mode="result_aware",
                    use_real_machine=not args.mock,
                )
                if client is not None:
                    env.attach_real_clients({args.machine: client})
                strategy = create_strategy(strategy_name)
                ep = run_episode(env, strategy, seed=seed, policy_name=strategy_name)
                all_results.append(ep)
                print(
                    f"    reward={ep['total_reward']:.1f}, steps={ep['steps']}, "
                    f"real={ep['real_tasks_submitted']}(完成{ep['real_tasks_completed']}), "
                    f"share={ep['real_task_share']:.2f}",
                    flush=True,
                )
                env.close()
            except Exception as e:
                print(f"    FAILED: {e}", flush=True)
                all_results.append({"seed": seed, "strategy": strategy_name, "error": str(e)[:200]})

    # ── 保存 ──
    data = {
        "experiment": "prereg_real_high_ratio",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "seeds": args.seeds,
            "strategies": args.strategy,
            "real_submit_probability": args.prob,
            "steps": args.steps,
            "shots": args.shots,
            "machine": args.machine,
            "circuit": QCIS_CIRCUIT,
            "real_feedback_mode": "result_aware",
            "max_real_tasks_per_run": max_real_tasks,
            "mock": args.mock,
            "unified_protocol": True,
            "preregistered": True,
        },
        "results": all_results,
    }
    output_path = args.output or (
        OUTPUT_DIR / f"prereg_high_ratio_{args.strategy[0]}_p{args.prob}_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    print(f"\n结果已保存: {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
