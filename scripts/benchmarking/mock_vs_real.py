"""
真机 vs 仿真对比报告生成器

用真机实测数据对比 Mock 仿真预测，评估仿真保真度。
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()

import numpy as np

API_KEY = os.getenv("TIANYAN_API_KEY", "")
REPORT_PATH = os.path.join(PROJECT_ROOT, "results", "realmock_comparison_report.md")

# 真机实测数据（已收集）
REAL_DATA = {
    "H_1q": {"qubits": 1, "gates": 2, "real_time_s": 93.43, "success": True},
    "H2_1q": {"qubits": 1, "gates": 3, "real_time_s": 57.25, "success": True},
}


def run_single_real(client, name, qcis, shots):
    """提交一个真机任务并记录耗时"""
    start = time.time()
    try:
        task_id = client.submit_quantum_task(qcis=qcis, shots=shots, task_name=name)
        result = client.wait_for_task(task_id, timeout=300)
        elapsed = time.time() - start
        success = result["status"] == "completed"
        return {"time_s": round(elapsed, 2), "success": success, "task_id": task_id}
    except Exception as e:
        return {"time_s": round(time.time() - start, 2), "success": False, "error": str(e)[:80]}


def run_mock_simulation(circuits, shots=512):
    """用 Mock 环境模拟执行时间"""
    from src.api.mock_client import MockTianyanClient

    mock = MockTianyanClient()
    results = {}
    for name, info in circuits.items():
        start = time.time()
        qcis = info["qcis"]
        task_id = mock.submit_quantum_task(circuit_qasm=qcis, shots=shots)
        mock.wait_for_task(task_id)
        elapsed = time.time() - start
        results[name] = {
            "mock_time_s": round(elapsed, 2),
            "qubits": info["qubits"],
            "gates": info["gates"],
        }
    return results


def build_test_circuits():
    """构建测试电路"""

    def c(n, ins):
        return {"qubits": n, "gates": len(ins), "qcis": ins}

    return {
        "H_1q": c(1, "H Q0\nM Q0"),
        "H2_1q": c(1, "H Q0\nH Q0\nM Q0"),
        "Bell_2q": c(2, "H Q0\nCZ Q0 Q1\nH Q1\nM Q0\nM Q1"),
        "deep_1q": c(1, "H Q0\nH Q0\nH Q0\nH Q0\nH Q0\nM Q0"),
        "qft_3q": c(3, "H Q0\nCZ Q0 Q1\nH Q1\nCZ Q1 Q2\nH Q2\nM Q0\nM Q1\nM Q2"),
    }


def generate_report(real_results, mock_results):
    """生成对比报告"""
    lines = [
        "# 天衍云真机 vs Mock 仿真对比报告",
        f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**真机**: tianyan_s (超导量子计算机)",
        "**Mock**: 单任务延迟已校准至 75s（基于真机实测）",
        "",
        "## 一、执行时间对比",
        "",
        "| 电路 | 量子比特 | 门数 | 真机耗时 | Mock耗时 | 偏差 |",
        "|------|---------|------|---------|---------|------|",
    ]

    diffs = []
    for name in mock_results:
        m = mock_results[name]
        r = real_results.get(name, {})
        real_t = r.get("time_s", 0) if r.get("success") else None
        mock_t = m["mock_time_s"]

        if real_t:
            diff_pct = round((mock_t - real_t) / real_t * 100, 1)
            diff_str = f"{diff_pct:+.1f}%"
            diffs.append(abs(diff_pct))
            status = r.get("success", False)
            real_str = f"{real_t:.1f}s" if status else "❌ 失败"
        else:
            diff_str = "—"
            real_str = "—"

        lines.append(
            f"| {name:10s} | {m['qubits']} | {m['gates']} | "
            f"{real_str:8s} | {mock_t:.1f}s | {diff_str:6s} |"
        )

    lines.append("")
    lines.append("## 二、偏差分析")
    lines.append("")

    if diffs:
        avg_diff = np.mean(diffs)
        lines.append(f"- 平均绝对偏差: **{avg_diff:.1f}%**")
        if avg_diff < 30:
            quality = "🟢 良好 — Mock 环境能较准确模拟真机延迟"
        elif avg_diff < 60:
            quality = "🟡 一般 — Mock 偏差较大，建议增加校准样本"
        else:
            quality = "🔴 差 — Mock 环境需要重新校准"
        lines.append(f"- 仿真质量: {quality}")
    else:
        lines.append("- 真机数据不足，暂无法评估")

    lines.append("")
    lines.append("## 三、结论")
    lines.append("")
    lines.append("- Mock 延迟已基于真机实测校准为 75s/任务")
    lines.append("- 真机执行时间约 55-95s，存在波动（受排队和量子比特状态影响）")
    lines.append("- 后续补充更多真机数据可进一步提升 Mock 保真度")
    lines.append("- tianyan_s 不支持参数化门（RX），量子电路需限制在 H/CZ 门集合")
    lines.append("")
    lines.append("---")
    lines.append("*报告自动生成 | 数据来源: cqlib → tianyan_s*")

    return "\n".join(lines)


def main():
    print(f"{'=' *60}")
    print("  真机 vs Mock 仿真对比报告")
    print(f"{'=' *60}")

    circuits = build_test_circuits()

    # 1. 真机测试（已有数据 + 补充 Bell_2q / deep_1q）
    client = CqlibTianyanClient(login_key=API_KEY, machine_name="tianyan_s")

    real_results = {}
    for name in ["H_1q", "H2_1q"]:
        real_results[name] = {
            "time_s": REAL_DATA[name]["real_time_s"],
            "success": REAL_DATA[name]["success"],
        }

    # 补充跑两个新电路
    for name in ["Bell_2q", "deep_1q"]:
        info = circuits[name]
        print(f"\n🖥️ 真机: {name} ({info['qubits']}q)...", end=" ", flush=True)
        r = run_single_real(client, name, info["qcis"], 512)
        real_results[name] = r
        print(f"{'✅' if r['success'] else '❌'} {r['time_s']}s")

    # 2. Mock 仿真
    print("\n📐 Mock 仿真（延迟=75s）...")
    mock_results = run_mock_simulation(circuits)

    # 3. 生成报告
    report = generate_report(real_results, mock_results)

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'=' *60}")
    print(report.split("\n")[-6][:60] if len(report.split("\n")) > 6 else "")
    print(f"\n✅ 报告已保存: {REPORT_PATH}")


if __name__ == "__main__":
    from src.api.tianyan_cqlib import CqlibTianyanClient

    main()
