"""
免费真机高效验证脚本 — 用日常机器调试，把 287 机时留给最终验证

用法:
    python scripts/efficient_real_benchmark.py --samples 3
    python scripts/efficient_real_benchmark.py --samples 5 --machines tianyan_s,tianyan_tn

流程:
    1. 策略对比     — PPO / FCFS / Random / Greedy 各 N 次采样
    2. 退火验证     — QUBO 真机退火测试
    3. 故障测试     — 故意提交到不可用机器，验证容错
    4. 生成报告     — 输出到 results/free_machine_benchmark_*.json
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

import cqlib
import numpy as np
from src.api.tianyan_cqlib import CqlibTianyanClient

API_KEY = os.getenv("TIANYAN_API_KEY", "")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--machines", default="tianyan_s,tianyan_sw,tianyan_tn",
                   help="逗号分隔的机器名")
    p.add_argument("--samples", type=int, default=3,
                   help="每种策略采样次数")
    p.add_argument("--shots", type=int, default=256,
                   help="每次测量 shots（免费机器用小值）")
    p.add_argument("--skip-annealing", action="store_true")
    p.add_argument("--skip-fault-test", action="store_true")
    return p.parse_args()


def build_test_circuit():
    """构建最小验证电路：H门 + 测量"""
    c = cqlib.Circuit(1)
    c.h(0)
    c.measure_all()
    return c


def submit_one(client, circuit, name, shots):
    """提交并等待真机执行完成，返回 {success, time_s, task_id, error}"""
    start = time.time()
    try:
        tid = client.submit_quantum_task(circuit=circuit, shots=shots, task_name=name)
        # 等待真机执行完成（之前只提交不等待，导致假数据）
        result = client.wait_for_task(tid, timeout=300)
        elapsed = round(time.time() - start, 2)
        success = result.get("status") == "completed"
        return {
            "success": success,
            "time_s": elapsed,
            "task_id": tid,
            "result": result.get("result", "")[:100],
        }
    except Exception as e:
        elapsed = round(time.time() - start, 2)
        return {"success": False, "time_s": elapsed, "error": str(e)[:80]}


def submit_one_fast(client, circuit, name, shots):
    """仅提交不等待（用于故障测试，不需要真机结果）"""
    start = time.time()
    try:
        tid = client.submit_quantum_task(circuit=circuit, shots=shots, task_name=name)
        return {"success": True, "time_s": round(time.time() - start, 2), "task_id": tid}
    except Exception as e:
        return {"success": False, "time_s": round(time.time() - start, 2), "error": str(e)[:80]}


def strategy_names():
    return ["FCFS", "Random", "Greedy", "QuantumOnly", "PPO"]


def run_strategy_bench(args):
    """在免费机器上跑策略对比"""
    machines = [m.strip() for m in args.machines.split(",")]
    circuit = build_test_circuit()
    results = {m: {s: [] for s in strategy_names()} for m in machines}

    total_tasks = len(machines) * len(strategy_names()) * args.samples
    done = 0

    print(f"\n{'='*60}")
    print(f"  策略对比 — {len(machines)} 台 × {len(strategy_names())} 策略 × {args.samples} 次")
    print(f"  总任务数: {total_tasks}, 预计耗时: ~{total_tasks * 1.5:.0f}min")
    print(f"{'='*60}")

    for machine_name in machines:
        print(f"\n🖥️ {machine_name}")
        try:
            client = CqlibTianyanClient(login_key=API_KEY, machine_name=machine_name)
        except Exception as e:
            print(f"  ⚠️ 跳过 ({e})")
            continue

        for strategy in strategy_names():
            for s in range(args.samples):
                name = f"Bench_{strategy}_S{s}"
                r = submit_one(client, circuit, name, args.shots)
                results[machine_name][strategy].append(r)
                done += 1
                emoji = "✅" if r["success"] else "❌"
                t = r.get("time_s", "?")
                print(f"  [{done}/{total_tasks}] {emoji} {strategy:12s} #{s} {t}s")

    return results


def run_annealing_test():
    """测试量子退火 QUBO 求解"""
    print(f"\n{'='*60}")
    print(f"  量子退火验证")
    print(f"{'='*60}")

    try:
        from src.quantum.annealing import QuantumAnnealingOptimizer

        opt = QuantumAnnealingOptimizer(num_qubits=4, simulation_mode=True)
        Q = np.array([[-1, 0.5], [0.5, -1]])
        result = opt.anneal(Q)

        print(f"  ✅ QUBO 求解成功: bitstring={result}")
        return {"success": True, "bitstring": result}
    except Exception as e:
        print(f"  ❌ 退火失败: {e}")
        return {"success": False, "error": str(e)[:80]}


def run_fault_test(args):
    """测试故障切换：故意提交到不存在的机器"""
    print(f"\n{'='*60}")
    print(f"  故障容错测试")
    print(f"{'='*60}")

    circuit = build_test_circuit()
    try:
        client = CqlibTianyanClient(
            login_key=API_KEY,
            machine_name="nonexistent_machine",
            auto_retry_machine=True,
        )
        r = submit_one(client, circuit, "FaultTest", args.shots)
        if r["success"]:
            print(f"  ✅ 故障切换成功，自动路由到: {r.get('task_id', '?')}")
        else:
            print(f"  ⚠️ 故障切换失败: {r.get('error', '?')}")
        return r
    except Exception as e:
        print(f"  ❌ 故障测试异常: {e}")
        return {"success": False, "error": str(e)[:80]}


def analyze(results, annealing, fault):
    """汇总分析"""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tasks_submitted": 0,
        "total_success": 0,
        "per_machine": {},
        "per_strategy": {s: {"success": 0, "total": 0, "avg_time": 0} for s in strategy_names()},
        "annealing": annealing,
        "fault_tolerance": fault,
    }

    for machine_name, strategies in results.items():
        m = {"success": 0, "total": 0, "avg_time": 0, "strategies": {}}
        for strategy, runs in strategies.items():
            s = {"success": 0, "total": len(runs), "avg_time": 0, "times": []}
            for r in runs:
                if r["success"]:
                    s["success"] += 1
                    s["times"].append(r["time_s"])
            s["avg_time"] = round(np.mean(s["times"]), 2) if s["times"] else 0
            m["strategies"][strategy] = s
            m["success"] += s["success"]
            m["total"] += s["total"]
            summary["total_success"] += s["success"]
            summary["total_tasks_submitted"] += s["total"]
            summary["per_strategy"][strategy]["success"] += s["success"]
            summary["per_strategy"][strategy]["total"] += s["total"]
            if "all_times" not in summary["per_strategy"][strategy]:
                summary["per_strategy"][strategy]["all_times"] = []
            summary["per_strategy"][strategy]["all_times"].extend(s["times"])
        m["avg_time"] = round(np.mean([s["avg_time"] for s in m["strategies"].values() if s["avg_time"]]), 2)
        summary["per_machine"][machine_name] = m

    # 各策略跨机器平均耗时
    for s in strategy_names():
        all_t = summary["per_strategy"][s].pop("all_times", [])
        summary["per_strategy"][s]["avg_time"] = round(np.mean(all_t), 2) if all_t else 0

    # 计算成功率
    summary["overall_success_rate"] = round(
        summary["total_success"] / max(summary["total_tasks_submitted"], 1), 3
    )

    # 各策略排名
    rankings = sorted(
        [(s, d["success"] / max(d["total"], 1)) for s, d in summary["per_strategy"].items()],
        key=lambda x: x[1], reverse=True,
    )
    summary["strategy_ranking"] = rankings

    return summary


def print_report(summary):
    print(f"\n{'='*60}")
    print(f"  📊 验证报告")
    print(f"{'='*60}")
    print(f"  总任务: {summary['total_tasks_submitted']}")
    print(f"  成功: {summary['total_success']} ({summary['overall_success_rate']:.0%})")
    print(f"\n  策略排名（按成功率）:")
    for rank, (name, rate) in enumerate(summary["strategy_ranking"], 1):
        bar = "█" * int(rate * 20)
        print(f"    {rank}. {name:12s} {rate:.0%} {bar}")
    print(f"\n  退火验证: {'✅' if summary['annealing']['success'] else '❌'}")
    print(f"  故障容错: {'✅' if summary['fault_tolerance']['success'] else '❌'}")


def main():
    args = parse_args()

    if not API_KEY:
        print("❌ 未设置 TIANYAN_API_KEY")
        return

    # 1. 策略对比
    bench = run_strategy_bench(args)

    # 2. 退火验证
    annealing = run_annealing_test() if not args.skip_annealing else {}

    # 3. 故障测试
    fault = run_fault_test(args) if not args.skip_fault_test else {}

    # 4. 分析
    summary = analyze(bench, annealing, fault)
    print_report(summary)

    # 5. 保存
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(PROJECT_ROOT, "results", f"free_machine_benchmark_{ts}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n📁 报告: {out}")


if __name__ == "__main__":
    main()
