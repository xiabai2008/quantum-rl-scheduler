#!/usr/bin/env python
"""多比特真机验证实验（Issue #249）

目标：验证多比特电路在天衍-287 真机上的执行能力和保真度递减趋势。

实验设计：
- 3 组比特数（1/2/3）× 3 seeds × 1 真机任务/seed = 9 个正式任务
- 冒烟测试 1 个任务（1 比特 H 门）
- 总硬上限 10 个真机任务
- 统一 shots=32，machine=tianyan-287
- 使用 Q1/Q2/Q3（天衍-287 物理比特 Q1~Q105，无 Q0）

电路（H 门均匀分布）：
- 1 比特: H Q1 / M Q1           → 理论 {"0": 0.5, "1": 0.5}
- 2 比特: H Q1 H Q2 / M Q1 Q2   → 理论 {"00": 0.25, "01": 0.25, "10": 0.25, "11": 0.25}
- 3 比特: H Q1 H Q2 H Q3 / M Q1 Q2 Q3 → 理论 8 个 bitstring 各 0.125

保真度公式（经典保真度）：
    F(p, q) = (Σᵢ √(pᵢ · qᵢ))²

用法：
    # 冒烟测试
    python scripts/real_machine/multi_qubit_validation.py --smoke

    # 正式实验
    python scripts/real_machine/multi_qubit_validation.py --formal
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from loguru import logger

from src.api.tianyan_cqlib import CqlibTianyanClient

# ── 实验配置（Issue #249） ──

SEEDS = [42, 123, 456]
QUBIT_GROUPS = [1, 2, 3]
SHOTS = 32
TARGET_MACHINE = "tianyan-287"
TASK_TIMEOUT_SECONDS = 180
TASK_POLL_INTERVAL = 5

HARD_LIMIT_FORMAL = len(SEEDS) * len(QUBIT_GROUPS)  # 9
HARD_LIMIT_SMOKE = 1
HARD_LIMIT_TOTAL = HARD_LIMIT_FORMAL + HARD_LIMIT_SMOKE  # 10

OUTPUT_DIR = _PROJECT_ROOT / "results" / "real_machine" / "multi_qubit_validation"


def build_h_circuit(n_qubits: int) -> str:
    """构建 n 比特 H 门均匀分布电路（使用 Q1~Qn）。

    天衍-287 物理比特 Q1~Q105，无 Q0。
    每条门指令单独一行（QCIS 规范）。
    """
    if n_qubits < 1 or n_qubits > 3:
        raise ValueError(f"仅支持 1/2/3 比特，实际: {n_qubits}")
    lines = [f"H Q{i + 1}" for i in range(n_qubits)]
    measures = " ".join(f"Q{i + 1}" for i in range(n_qubits))
    lines.append(f"M {measures}")
    return "\n".join(lines)


def theoretical_distribution(n_qubits: int) -> dict[str, float]:
    """n 比特 H 门均匀分布的理论概率。"""
    n_outcomes = 2**n_qubits
    prob = 1.0 / n_outcomes
    dist = {}
    for i in range(n_outcomes):
        bitstring = format(i, f"0{n_qubits}b")
        dist[bitstring] = prob
    return dist


def compute_fidelity(measured: dict[str, float], theoretical: dict[str, float]) -> float:
    """经典保真度 F(p, q) = (Σᵢ √(pᵢ · qᵢ))²

    与 src/scheduler/env_real_machine.py:258 的 compute_result_fidelity 一致。
    """
    all_keys = set(measured.keys()) | set(theoretical.keys())
    fidelity_sum = 0.0
    for key in all_keys:
        p = measured.get(key, 0.0)
        q = theoretical.get(key, 0.0)
        fidelity_sum += (p * q) ** 0.5
    fidelity = fidelity_sum**2
    return float(max(0.0, min(1.0, fidelity)))


def submit_and_poll(
    client: CqlibTianyanClient,
    qcis: str,
    shots: int,
    task_name: str,
    machine_name: str,
) -> dict:
    """提交并轮询单个真机任务，返回完整记录。"""
    record: dict = {
        "task_id": None,
        "shots": shots,
        "circuit": qcis,
        "machine": machine_name,
        "submitted_at": None,
        "completed_at": None,
        "status": None,
        "probability": None,
        "fidelity": None,
        "elapsed_seconds": None,
        "error": None,
        "mock": False,
        "degraded": False,
    }

    # QCIS 预校验
    try:
        platform = getattr(client, "platform", None)
        if platform is not None and hasattr(platform, "qcis_check_regular"):
            qcis_valid = platform.qcis_check_regular(qcis)
            if not qcis_valid:
                record["status"] = "failed"
                record["error"] = "QCIS 预校验失败"
                logger.error(f"  ❌ QCIS 预校验失败: {qcis!r}")
                return record
    except Exception as e:
        record["status"] = "query_error"
        record["error"] = f"QCIS 校验异常: {str(e)[:100]}"
        logger.warning(f"  ⚠️ QCIS 校验异常: {e}")
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

    if task_id is None or (isinstance(task_id, str) and not task_id.strip()):
        record["status"] = "failed"
        record["error"] = "submit_quantum_task 返回 None"
        record["submitted_at"] = datetime.now().isoformat()
        logger.error("  ❌ 提交失败：task_id 为 None")
        return record

    record["task_id"] = str(task_id)
    record["submitted_at"] = datetime.now().isoformat()
    logger.info(f"  ✅ task_id: {task_id}")

    # 轮询
    try:
        poll_result = client.wait_for_task(
            task_id, timeout=TASK_TIMEOUT_SECONDS, poll_interval=TASK_POLL_INTERVAL
        )
    except Exception as e:
        record["status"] = "query_error"
        record["error"] = f"轮询异常: {str(e)[:150]}"
        record["completed_at"] = datetime.now().isoformat()
        logger.warning(f"  ⚠️ 轮询异常（task_id={task_id} 保留）: {e}")
        return record

    record["completed_at"] = datetime.now().isoformat()

    if not poll_result:
        record["status"] = "timeout"
        record["error"] = "wait_for_task 返回空"
        return record

    final_status = poll_result.get("status", "unknown")
    record["status"] = final_status

    # probability 从 result 字段读取
    prob = poll_result.get("result")
    if prob is None:
        prob = poll_result.get("probability")
    if isinstance(prob, str) and prob:
        with contextlib.suppress(json.JSONDecodeError, ValueError):
            prob = json.loads(prob)
    record["probability"] = prob

    if final_status != "completed":
        record["error"] = poll_result.get("error", f"status={final_status}")
        return record

    if not isinstance(prob, dict) or not prob:
        record["error"] = "probability 为空或非字典"
        return record

    # 计算耗时
    submitted_at = record.get("submitted_at")
    completed_at = record.get("completed_at")
    if submitted_at and completed_at:
        try:
            dt_sub = datetime.fromisoformat(submitted_at)
            dt_comp = datetime.fromisoformat(completed_at)
            record["elapsed_seconds"] = round((dt_comp - dt_sub).total_seconds(), 2)
        except (ValueError, TypeError):
            pass

    logger.info(f"  ✅ 任务完成: status={final_status}, prob={prob}")
    return record


def run_smoke_test(client: CqlibTianyanClient) -> dict:
    """冒烟测试：1 比特 H 门。"""
    logger.info("=" * 60)
    logger.info("冒烟测试：1 比特 H 门电路")
    logger.info("=" * 60)

    qcis = build_h_circuit(1)
    record = submit_and_poll(
        client=client,
        qcis=qcis,
        shots=SHOTS,
        task_name="smoke_1qubit",
        machine_name=TARGET_MACHINE,
    )

    # 计算保真度
    if record.get("probability") and isinstance(record["probability"], dict):
        record["fidelity"] = compute_fidelity(record["probability"], theoretical_distribution(1))

    passed = (
        record.get("status") == "completed"
        and record.get("probability") is not None
        and record.get("fidelity") is not None
        and record.get("mock") is False
    )

    logger.info(
        f"冒烟结果: passed={passed}, status={record.get('status')}, "
        f"fidelity={record.get('fidelity')}"
    )
    return {"smoke_test": True, "passed": passed, **record}


def run_formal_experiment(client: CqlibTianyanClient) -> list[dict]:
    """正式实验：3 组比特数 × 3 seeds。"""
    results: list[dict] = []
    total_submitted = 0

    for n_qubits in QUBIT_GROUPS:
        qcis = build_h_circuit(n_qubits)
        theoretical = theoretical_distribution(n_qubits)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"实验组: {n_qubits} 比特")
        logger.info(f"电路: {qcis!r}")
        logger.info(f"理论分布: {theoretical}")
        logger.info(f"{'=' * 60}")

        for seed in SEEDS:
            if total_submitted >= HARD_LIMIT_FORMAL:
                logger.warning(f"已达正式实验硬上限 {HARD_LIMIT_FORMAL}，停止")
                break

            logger.info(f"\n[{n_qubits}q] seed={seed} 开始")
            record = submit_and_poll(
                client=client,
                qcis=qcis,
                shots=SHOTS,
                task_name=f"multiq_{n_qubits}q_seed{seed}",
                machine_name=TARGET_MACHINE,
            )

            # 计算保真度
            if record.get("probability") and isinstance(record["probability"], dict):
                record["fidelity"] = compute_fidelity(record["probability"], theoretical)

            record["n_qubits"] = n_qubits
            record["seed"] = seed
            record["theoretical"] = theoretical

            results.append(record)
            if record.get("task_id"):
                total_submitted += 1

            logger.info(
                f"[{n_qubits}q] seed={seed} 完成: "
                f"status={record.get('status')}, "
                f"fidelity={record.get('fidelity')}"
            )

    return results


def analyze_results(results: list[dict]) -> dict:
    """分析实验结果，按比特数分组统计。"""
    analysis: dict = {"by_qubit": {}, "summary": {}}

    for n_q in QUBIT_GROUPS:
        group = [r for r in results if r.get("n_qubits") == n_q]
        completed = [
            r for r in group if r.get("status") == "completed" and r.get("fidelity") is not None
        ]
        fidelities = [r["fidelity"] for r in completed]

        stats = {
            "n_qubits": n_q,
            "total_tasks": len(group),
            "completed_tasks": len(completed),
            "success_rate": len(completed) / len(group) if group else 0.0,
            "fidelities": fidelities,
            "mean_fidelity": sum(fidelities) / len(fidelities) if fidelities else None,
            "std_fidelity": (
                math.sqrt(
                    sum((f - sum(fidelities) / len(fidelities)) ** 2 for f in fidelities)
                    / len(fidelities)
                )
                if len(fidelities) > 1
                else 0.0
            ),
            "min_fidelity": min(fidelities) if fidelities else None,
            "max_fidelity": max(fidelities) if fidelities else None,
        }
        analysis["by_qubit"][str(n_q)] = stats

    # 汇总
    all_fidelities = [r["fidelity"] for r in results if r.get("fidelity") is not None]
    analysis["summary"] = {
        "total_tasks": len(results),
        "completed_tasks": len(all_fidelities),
        "overall_success_rate": len(all_fidelities) / len(results) if results else 0.0,
        "qubit_groups": QUBIT_GROUPS,
        "seeds": SEEDS,
        "shots": SHOTS,
        "machine": TARGET_MACHINE,
    }

    return analysis


def save_results(smoke: dict, results: list[dict], analysis: dict, elapsed: float) -> Path:
    """保存完整实验数据。"""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_file = OUTPUT_DIR / f"multi_qubit_validation_{timestamp}.json"

    data = {
        "experiment": "multi_qubit_validation_issue_249",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "qubit_groups": QUBIT_GROUPS,
            "seeds": SEEDS,
            "shots": SHOTS,
            "machine": TARGET_MACHINE,
            "hard_limit_total": HARD_LIMIT_TOTAL,
            "smoke_passed": smoke.get("passed", False),
        },
        "total_elapsed_seconds": round(elapsed, 2),
        "total_submitted": 1 + len([r for r in results if r.get("task_id")]),
        "smoke": smoke,
        "results": results,
        "analysis": analysis,
    }

    with open(data_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"\n数据已保存: {data_file}")
    return data_file


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="多比特真机验证实验（Issue #249）")
    parser.add_argument("--smoke", action="store_true", help="仅运行冒烟测试")
    parser.add_argument("--formal", action="store_true", help="运行正式实验")
    parser.add_argument("--api-key", default=None, help="天衍云 API Key（覆盖环境变量）")
    args = parser.parse_args()

    if not args.smoke and not args.formal:
        args.formal = True  # 默认运行正式实验（含冒烟）

    # 获取 API Key
    api_key = args.api_key or os.getenv("TIANYAN_API_KEY") or os.getenv("TIANYAN_API_TOKEN")
    if not api_key:
        logger.error("未找到 API Key（--api-key 或 TIANYAN_API_KEY 环境变量）")
        return 1

    logger.info("实验配置:")
    logger.info(f"  机器: {TARGET_MACHINE}")
    logger.info(f"  比特组: {QUBIT_GROUPS}")
    logger.info(f"  Seeds: {SEEDS}")
    logger.info(f"  Shots: {SHOTS}")
    logger.info(f"  硬上限: {HARD_LIMIT_TOTAL}")

    # 创建客户端
    client = CqlibTianyanClient(
        login_key=api_key,
        machine_name=TARGET_MACHINE,
        auto_retry_machine=False,
    )

    start_time = time.time()

    # 冒烟测试
    smoke = run_smoke_test(client)
    if not smoke.get("passed"):
        logger.error("❌ 冒烟测试失败，终止正式实验")
        save_results(smoke, [], analyze_results([]), time.time() - start_time)
        return 1

    if args.smoke and not args.formal:
        logger.info("✅ 冒烟测试通过（--smoke 模式，跳过正式实验）")
        save_results(smoke, [], analyze_results([]), time.time() - start_time)
        return 0

    # 正式实验
    results = run_formal_experiment(client)
    elapsed = time.time() - start_time

    # 分析
    analysis = analyze_results(results)

    # 打印汇总
    logger.info(f"\n{'=' * 60}")
    logger.info("实验汇总")
    logger.info(f"{'=' * 60}")
    for n_q in QUBIT_GROUPS:
        stats = analysis["by_qubit"].get(str(n_q), {})
        logger.info(
            f"{n_q} 比特: "
            f"完成 {stats.get('completed_tasks', 0)}/{stats.get('total_tasks', 0)}, "
            f"平均保真度={stats.get('mean_fidelity')}, "
            f"标准差={stats.get('std_fidelity')}"
        )

    # 保存
    save_results(smoke, results, analysis, elapsed)
    logger.info(f"\n总耗时: {elapsed:.1f}s")
    logger.info(f"总提交: {1 + len([r for r in results if r.get('task_id')])}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
