"""
天衍云真机校准脚本 — 用真实数据调校 Mock 环境参数

运行方式: python scripts/calibrate_mock.py

流程:
1. 批量提交不同规模量子电路到真机
2. 记录真实执行时间、排队时间、成功率
3. 对比 Mock 环境预测值
4. 生成校准参数写入 config.yaml
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

import contextlib

import numpy as np
import yaml

from src.api.tianyan_cqlib import CqlibTianyanClient

# ── 测试电路模板 ──
TEST_CIRCUITS = {
    "H_1q": (1, ["H Q0", "M Q0"]),
    "H2_1q": (1, ["H Q0", "H Q0", "M Q0"]),
    "RX_1q": (1, ["RX Q0 0.5", "M Q0"]),
    "Bell_2q": (2, ["H Q0", "CZ Q0 Q1", "H Q1", "M Q0", "M Q1"]),
    "GHZ_3q": (3, ["H Q0", "CZ Q0 Q1", "H Q1", "CZ Q1 Q2", "H Q2", "M Q0", "M Q1", "M Q2"]),
    "deep_2q": (
        2,
        [
            "H Q0",
            "RX Q1 0.3",
            "CZ Q0 Q1",
            "H Q1",
            "RX Q0 0.7",
            "CZ Q0 Q1",
            "H Q1",
            "RX Q0 0.2",
            "M Q0",
            "M Q1",
        ],
    ),
    "shallow_1q": (1, ["M Q0"]),
}


def parse_args():
    p = argparse.ArgumentParser(description="真机校准 Mock 环境")
    p.add_argument("--machine", default="tianyan_s", help="量子计算机名称")
    p.add_argument("--shots", type=int, default=512, help="每个电路测量次数")
    p.add_argument("--runs", type=int, default=1, help="每个电路重复次数")
    p.add_argument("--output", default="./results/calibration.json", help="输出文件")
    return p.parse_args()


def build_qcis(instructions):
    """将指令列表转为 QCIS 字符串"""
    return "\n".join(instructions)


def submit_and_wait(client: CqlibTianyanClient, qcis: str, name: str, shots: int):
    """提交任务并等待完成"""
    start = time.time()
    task_id = client.submit_quantum_task(qcis=qcis, shots=shots, task_name=name)
    submit_time = time.time() - start

    result = client.wait_for_task(task_id, timeout=300)

    if result["status"] == "completed":
        prob_str = result.get("result", "")
        prob = {}
        with contextlib.suppress(json.JSONDecodeError, TypeError):
            prob = json.loads(prob_str)
        return {
            "task_id": task_id,
            "success": True,
            "submit_time_s": round(submit_time, 2),
            "total_time_s": round(time.time() - start, 2),
            "probability": prob,
            "shots": shots,
        }
    else:
        return {
            "task_id": task_id,
            "success": False,
            "submit_time_s": round(submit_time, 2),
            "error": result.get("error", "timeout"),
        }


def run_calibration(args):
    """执行校准流程"""
    api_key = os.getenv("TIANYAN_API_KEY", "")
    if not api_key:
        print("❌ 未设置 TIANYAN_API_KEY")
        return

    client = CqlibTianyanClient(login_key=api_key, machine_name=args.machine)

    print(f"{'=' *60}")
    print("  天衍云真机校准 — Mock 环境调校")
    print(f"{'=' *60}")
    print(f"  机器: {args.machine}")
    print(f"  Shots: {args.shots}")
    print(f"  电路类型: {len(TEST_CIRCUITS)} 种")
    print(f"{'=' *60}\n")

    results = {
        "machine": args.machine,
        "timestamp": datetime.now().isoformat(),
        "config": {"shots": args.shots, "runs": args.runs},
        "tests": [],
        "summary": {},
    }

    # ── 1. 批量提交测试 ──
    for circuit_name, (qubits, instructions) in TEST_CIRCUITS.items():
        qcis = build_qcis(instructions)
        print(f"\n[{circuit_name}] {qubits} qubit, {len(instructions)} 指令")

        for run in range(args.runs):
            name = f"Calib_{circuit_name}_R{run}"
            print(f"  Run {run +1}/{args.runs}...", end=" ", flush=True)
            record = submit_and_wait(client, qcis, name, args.shots)
            record["circuit"] = circuit_name
            record["qubits"] = qubits
            record["num_instructions"] = len(instructions)
            results["tests"].append(record)
            print(f"{'✅' if record['success'] else '❌'} {record.get('total_time_s', '?')}s")

    # ── 2. 分析结果 ──
    successes = [r for r in results["tests"] if r["success"]]
    if not successes:
        print("\n❌ 所有任务失败，无法校准")
        return

    times = [r["total_time_s"] for r in successes]
    submit_times = [r["submit_time_s"] for r in successes]

    summary = {
        "total_tests": len(results["tests"]),
        "success_count": len(successes),
        "success_rate": len(successes) / len(results["tests"]),
        "avg_total_time_s": round(np.mean(times), 2),
        "max_total_time_s": round(np.max(times), 2),
        "min_total_time_s": round(np.min(times), 2),
        "avg_submit_time_s": round(np.mean(submit_times), 2),
    }

    # 按电路复杂度分组
    by_circuit = {}
    for r in successes:
        name = r["circuit"]
        if name not in by_circuit:
            by_circuit[name] = []
        by_circuit[name].append(r["total_time_s"])

    summary["by_circuit"] = {name: round(np.mean(ts), 2) for name, ts in by_circuit.items()}

    results["summary"] = summary

    # ── 3. 保存结果 ──
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ── 4. 打印报告 ──
    print(f"\n{'=' *60}")
    print("  校准报告")
    print(f"{'=' *60}")
    print(f"  成功率：{summary['success_rate']:.0%}")
    print(f"  平均总耗时：{summary['avg_total_time_s']}s")
    print(f"  平均提交耗时：{summary['avg_submit_time_s']}s")
    print(f"  最快/最慢：{summary['min_total_time_s']}s / {summary['max_total_time_s']}s")
    print("\n  各电路类型耗时：")
    for name, t in summary["by_circuit"].items():
        print(f"    {name:15s}: {t:.2f}s")
    print(f"\n  JSON 已保存: {args.output}")

    # ── 5. 更新 config.yaml 中的 Mock 参数 ──
    config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        config["tianyan"]["mock_delay"] = round(summary["avg_total_time_s"] * 0.3, 2)
        config["tianyan"]["mock_failure_rate"] = round(1 - summary["success_rate"], 3)

        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

        print("\n  ✅ config.yaml 已更新:")
        print(f"     mock_delay = {config['tianyan']['mock_delay']}s")
        print(f"     mock_failure_rate = {config['tianyan']['mock_failure_rate']}")
    except Exception as e:
        print(f"\n  ⚠️ 无法更新 config.yaml: {e}")

    return results


if __name__ == "__main__":
    args = parse_args()
    run_calibration(args)
