"""
任务解析器 QASM 性能 benchmark。

运行方式：
    python scripts/benchmark_parser.py

该脚本用于 Issue #27：测量 LegacyTaskParser 解析 1q/5q/10q
三种 QASM 电路的速度，每种场景默认执行 1000 次。
"""

from __future__ import annotations

import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.scheduler.parser import LegacyTaskParser

ITERATIONS = 1000
WARMUP_ITERATIONS = 50


@dataclass(frozen=True)
class BenchmarkCase:
    """单个 benchmark 场景。"""

    name: str
    qubits: int
    qasm: str


@dataclass(frozen=True)
class BenchmarkResult:
    """单个 benchmark 场景的统计结果。"""

    name: str
    qubits: int
    iterations: int
    total_ms: float
    mean_us: float
    median_us: float
    min_us: float
    max_us: float
    parses_per_second: float
    gate_count: int
    measurement_count: int


def build_qasm(qubits: int) -> str:
    """
    构造固定规模的 QASM 测试样例。

    1q 场景只包含单比特门和测量；5q/10q 场景额外加入相邻比特的
    CX 门，模拟稍复杂的量子线路。
    """
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{qubits}];",
        f"creg c[{qubits}];",
    ]

    for idx in range(qubits):
        lines.append(f"h q[{idx}];")
        lines.append(f"x q[{idx}];")

    for idx in range(max(0, qubits - 1)):
        lines.append(f"cx q[{idx}],q[{idx + 1}];")

    for idx in range(qubits):
        lines.append(f"measure q[{idx}] -> c[{idx}];")

    return "\n".join(lines)


def run_case(
    parser: LegacyTaskParser,
    case: BenchmarkCase,
    iterations: int = ITERATIONS,
) -> BenchmarkResult:
    """运行单个 benchmark 场景并返回统计结果。"""
    for _ in range(WARMUP_ITERATIONS):
        parser.parse(case.qasm, format="qasm")

    durations_us: list[float] = []
    last_features = None
    started = time.perf_counter()

    for _ in range(iterations):
        item_started = time.perf_counter()
        last_features = parser.parse(case.qasm, format="qasm")
        durations_us.append((time.perf_counter() - item_started) * 1_000_000)

    total_ms = (time.perf_counter() - started) * 1_000

    if last_features is None:
        raise RuntimeError(f"{case.name} 解析失败：parser 返回 None")
    if last_features.qubit_count != case.qubits:
        raise RuntimeError(
            f"{case.name} 解析结果错误：期望 {case.qubits} qubits，"
            f"实际 {last_features.qubit_count}"
        )

    return BenchmarkResult(
        name=case.name,
        qubits=case.qubits,
        iterations=iterations,
        total_ms=total_ms,
        mean_us=statistics.fmean(durations_us),
        median_us=statistics.median(durations_us),
        min_us=min(durations_us),
        max_us=max(durations_us),
        parses_per_second=iterations / (total_ms / 1000),
        gate_count=last_features.gate_count,
        measurement_count=last_features.measurement_count,
    )


def print_results(results: list[BenchmarkResult]) -> None:
    """输出 Markdown 风格 benchmark 表格。"""
    print("Task parser QASM benchmark")
    print(f"Iterations per case: {ITERATIONS}")
    print()
    print(
        "| case | qubits | gates | measures | total_ms | "
        "mean_us | median_us | min_us | max_us | parses_per_sec |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for result in results:
        print(
            f"| {result.name} | {result.qubits} | {result.gate_count} | "
            f"{result.measurement_count} | {result.total_ms:.2f} | "
            f"{result.mean_us:.2f} | {result.median_us:.2f} | "
            f"{result.min_us:.2f} | {result.max_us:.2f} | "
            f"{result.parses_per_second:.0f} |"
        )


def main() -> None:
    """脚本入口。"""
    parser = LegacyTaskParser()
    cases = [
        BenchmarkCase(name="1q QASM", qubits=1, qasm=build_qasm(1)),
        BenchmarkCase(name="5q QASM", qubits=5, qasm=build_qasm(5)),
        BenchmarkCase(name="10q QASM", qubits=10, qasm=build_qasm(10)),
    ]

    results = [run_case(parser, case) for case in cases]
    print_results(results)


if __name__ == "__main__":
    main()
