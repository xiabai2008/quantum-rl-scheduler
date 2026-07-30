#!/usr/bin/env python
"""
天衍-287全量测试包装器

通过运行时monkey-patch将标准测试脚本的Q0电路适配为Q1电路（天衍-287无Q0比特），
使所有5项标准测试套件可在天衍-287上运行。

不修改任何源代码文件，所有patch仅在运行时生效。

Usage:
    .venv\\Scripts\\python.exe scripts\real_machine\tianyan287_full_test.py

Patches applied:
    1. submit_quantum_task: Q0->Q1 qubit shift + shots cap to 32
    2. get_task_status: max_wait_time=15s + JSON string probability parsing
    3. wait_for_task: query_fail_count threshold=10 (was 3)
"""

import contextlib
import json
import os
import re
import runpy
import sys
import time
import traceback
from pathlib import Path

# Ensure project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from loguru import logger

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.api.types import TaskResult

# ============================================================
# Runtime Patches (no source code modified)
# ============================================================


def shift_qubits(qcis_str: str) -> str:
    """Shift all qubit numbers by +1 (Q0->Q1, Q1->Q2, etc.) for Tianyan-287.

    Tianyan-287 has no Q0 bit (physical bits Q1~Q105).
    All standard test scripts use Q0 circuits, which fail validation on 287.
    This function shifts every QN -> Q(N+1) in the QCIS string.
    """
    return re.sub(r"Q(\d+)", lambda m: f"Q{int(m.group(1)) + 1}", qcis_str)


# Save original methods
_orig_submit = CqlibTianyanClient.submit_quantum_task
_orig_get_status = CqlibTianyanClient.get_task_status


def _patched_submit(self, qcis="", circuit=None, shots=1024, task_name="Scheduler_Task"):
    """Patched submit_quantum_task: shift Q0->Q1 and cap shots to 32 for 287."""
    # Generate QCIS string (same logic as original)
    if qcis:
        qcis_str = qcis
    elif circuit is not None:
        qcis_str = circuit.qcis if hasattr(circuit, "qcis") else str(circuit)
    else:
        raise ValueError("必须提供 qcis 或 circuit")

    # Shift qubit numbers for 287 compatibility
    original_qcis = qcis_str
    qcis_str = shift_qubits(qcis_str)
    if qcis_str != original_qcis:
        logger.debug(f"[287-PATCH] QCIS shifted: {original_qcis[:60]} -> {qcis_str[:60]}")

    # Cap shots to 32 for 287 (machine time conservation)
    if shots > 32:
        logger.debug(f"[287-PATCH] Shots capped: {shots} -> 32")
        shots = 32

    # Call original with patched parameters
    return _orig_submit(self, qcis=qcis_str, circuit=None, shots=shots, task_name=task_name)


def _patched_get_status(self, task_id: str) -> TaskResult:
    """Patched get_task_status: max_wait_time=15, JSON string probability parsing."""
    try:
        from cqlib.exceptions import CqlibRequestError

        result = self.platform.query_experiment(
            task_id,
            max_wait_time=15,
            sleep_time=3,
        )
        if isinstance(result, list) and len(result) > 0:
            data = result[0]
            if isinstance(data, dict):
                has_result = "resultStatus" in data or "probability" in data
                probability = data.get("probability")

                # JSON string probability parsing (cqlib SDK returns str, not dict)
                if isinstance(probability, str) and probability:
                    with contextlib.suppress(json.JSONDecodeError, ValueError):
                        probability = json.loads(probability)

                # Fallback: check resultStatus if probability is empty
                if not probability and "resultStatus" in data:
                    rs = data.get("resultStatus")
                    if isinstance(rs, str) and rs:
                        try:
                            rs_parsed = json.loads(rs)
                            if isinstance(rs_parsed, dict):
                                probability = rs_parsed
                        except (json.JSONDecodeError, ValueError):
                            pass
                    elif isinstance(rs, dict):
                        probability = rs

                counts = data.get("counts")
                if isinstance(counts, str) and counts:
                    with contextlib.suppress(json.JSONDecodeError, ValueError):
                        counts = json.loads(counts)

                return TaskResult(
                    task_id=task_id,
                    status="completed" if has_result else "running",
                    probability=probability if isinstance(probability, dict) else {},
                    counts=counts if isinstance(counts, dict) else None,
                    shots=int(data.get("shots", 0) or 0),
                    backend=str(data.get("machine", self.machine_name)),
                    raw=data,
                )
        return TaskResult(
            task_id=task_id,
            status="unknown",
            probability={},
            counts=None,
            shots=0,
            backend=self.machine_name,
            raw=result,
        )
    except CqlibRequestError as e:
        message = str(e)
        terminal_failure_markers = ("运行失败", "run failure", "tasks have failed")
        if any(marker in message.lower() for marker in terminal_failure_markers):
            return TaskResult(
                task_id=task_id,
                status="error",
                probability={},
                counts=None,
                shots=0,
                backend=self.machine_name,
                error=message,
            )
        logger.debug(f"[Cqlib] 查询任务 {task_id} CqlibRequestError: {message[:80]}")
        return TaskResult(
            task_id=task_id,
            status="query_error",
            probability={},
            counts=None,
            shots=0,
            backend=self.machine_name,
            error=message[:200],
            raw={},
        )
    except Exception as e:
        logger.debug(f"[Cqlib] 查询任务 {task_id} 状态失败: {e}")
        return TaskResult(
            task_id=task_id,
            status="error",
            probability={},
            counts=None,
            shots=0,
            backend=self.machine_name,
            error=str(e),
        )


def _patched_wait(self, task_id: str, timeout: int = 300, poll_interval: int = 5) -> TaskResult:
    """Patched wait_for_task: query_fail_count threshold=10 (was 3)."""
    start = time.time()
    query_fail_count = 0
    while time.time() - start < timeout:
        status = self.get_task_status(task_id)
        if status["status"] == "completed":
            return status
        if status["status"] == "error":
            return status
        if status["status"] == "query_error":
            query_fail_count += 1
            if query_fail_count >= 10:
                return TaskResult(
                    task_id=task_id,
                    status="error",
                    probability={},
                    counts=None,
                    shots=0,
                    backend=self.machine_name,
                )
        time.sleep(poll_interval)
    return TaskResult(
        task_id=task_id,
        status="timeout",
        probability={},
        counts=None,
        shots=0,
        backend=self.machine_name,
    )


# Apply all patches
CqlibTianyanClient.submit_quantum_task = _patched_submit
CqlibTianyanClient.get_task_status = _patched_get_status
CqlibTianyanClient.wait_for_task = _patched_wait

print("[287-PATCH] All patches applied:")
print("  - submit_quantum_task: Q0->Q1 qubit shift + shots cap=32")
print("  - get_task_status: max_wait_time=15 + JSON string probability parsing")
print("  - wait_for_task: query_fail_count threshold=10")


# ============================================================
# Test Runner
# ============================================================

TESTS = [
    (
        "冒烟测试",
        "scripts/real_machine/smoke_test.py",
        ["--machine", "tianyan-287", "--shots", "32"],
    ),
    ("退火验证", "scripts/real_machine/annealing_validation.py", ["--machine", "tianyan-287"]),
    ("RL验证", "scripts/real_machine/rl_validation.py", ["--machine", "tianyan-287"]),
    ("策略对比", "scripts/real_machine/strategy_comparison.py", ["--machine", "tianyan-287"]),
    (
        "闭环训练",
        "scripts/real_machine/ppo_closed_loop_async.py",
        ["--machine", "tianyan-287", "--timesteps", "2000"],
    ),
]


def run_test(name: str, script_path: str, args: list) -> bool:
    """Run a single test script using runpy (same process, patches persist)."""
    print(f"\n{'=' * 60}")
    print(f"[287-FULL] {name}")
    print(f"[287-FULL] Script: {script_path} {' '.join(args)}")
    print(f"{'=' * 60}\n")

    old_argv = sys.argv
    sys.argv = [script_path, *args]

    try:
        runpy.run_path(script_path, run_name="__main__")
        print(f"\n[287-FULL] {name}: COMPLETED")
        return True
    except SystemExit as e:
        print(f"\n[287-FULL] {name}: EXIT({e.code})")
        return e.code == 0
    except Exception as e:
        print(f"\n[287-FULL] {name}: FAILED - {e}")
        traceback.print_exc()
        return False
    finally:
        sys.argv = old_argv


def main():
    os.environ["TIANYAN_MOCK_MODE"] = "false"
    print(f"\n[287-FULL] 天衍-287全量测试 - {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[287-FULL] 工作目录: {PROJECT_ROOT}")
    print(f"[287-FULL] 测试项: {len(TESTS)}项标准测试（与模式A相同）")
    print("[287-FULL] 电路适配: Q0->Q1 (运行时patch)")
    print("[287-FULL] Shots: 32 (287机时保护)")

    results = {}
    for name, script, script_args in TESTS:
        success = run_test(name, script, script_args)
        results[name] = "PASS" if success else "FAIL"
        # 前一项失败不阻塞后一项（与模式A一致）

    # Summary
    print(f"\n{'=' * 60}")
    print("[287-FULL] 测试汇总")
    print(f"{'=' * 60}")
    for n, s in results.items():
        print(f"  {n}: {s}")
    total_pass = sum(1 for s in results.values() if s == "PASS")
    print(f"\n  总计: {total_pass}/{len(results)} 通过")


if __name__ == "__main__":
    main()
