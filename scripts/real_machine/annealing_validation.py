"""退火算法真机验证（阶段 3）。

生成 10 变量 QUBO 问题，对比蛮力求解 / 模拟退火 / D-Wave neal 退火的
求解质量与时间。同时提交 5 个真机量子电路任务验证量子后端可用性。

天衍-176 为门控量子计算机（非量子退火器），不支持直接 QUBO 退火提交。
因此真机部分通过提交量子电路任务验证后端连通性，退火求解在本地完成。

实验设计:
    - QUBO 规模: 10 变量 (10x10 矩阵, ~800 字节)
    - 求解方法: 蛮力 (2^10=1024) / numpy 模拟退火 / D-Wave neal (如可用)
    - 真机任务: 5 个 H 门电路 (shots=1024)
    - num_reads=100, annealing_time=1.0

用法:
    # Mock dry-run
    python scripts/real_machine/annealing_validation.py --mock

    # 真机执行
    python scripts/real_machine/annealing_validation.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 环境变量设置（必须在 import 项目模块之前）
# ---------------------------------------------------------------------------
os.environ.setdefault("TIANYAN_API_KEY", "")
os.environ.setdefault("TIANYAN_MOCK_MODE", "false")
os.environ.setdefault("TIANYAN_MACHINE", "tianyan176")
os.environ.setdefault("QUANTUM_ACCELERATION_ENABLED", "1")

# ---------------------------------------------------------------------------
# 路径设置
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent

for p in [_PROJECT_ROOT, _SCRIPT_DIR]:
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from loguru import logger

# 复用 smoke_test.py 工具函数
from smoke_test import (  # type: ignore[import-not-found]
    MockSmokeClient,
    compute_fidelity,
    compute_measurement_error,
    compute_probability_from_shots,
    parse_probability,
    poll_task_result,
)

from src.api.tianyan_cqlib import CqlibTianyanClient

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
RESULTS_DIR = _PROJECT_ROOT / "results" / "real_machine"

# QUBO 参数
QUBO_SIZE = 10  # 10 变量 QUBO 问题
NUM_READS = 100  # 退火采样次数
ANNEALING_TIME = 1.0  # 退火时间 (微秒)
SEED = 42  # 随机种子

# 真机参数
REAL_SHOTS = 1024  # 真机 shots 数
REAL_TASK_COUNT = 5  # 真机任务数
QCIS_CIRCUIT = "H Q0\nM Q0"  # H 门电路


# ---------------------------------------------------------------------------
# QUBO 问题生成
# ---------------------------------------------------------------------------


def generate_qubo_problem(size: int, seed: int = 42) -> np.ndarray:
    """生成可复现的 QUBO 问题矩阵。

    生成一个对称的 QUBO 矩阵，对角元为随机负值（鼓励变量取 1），
    非对角元为随机正值（鼓励变量互斥），确保问题有非平凡解。

    Args:
        size: QUBO 矩阵维度
        seed: 随机种子

    Returns:
        QUBO 矩阵 (size x size)
    """
    rng = np.random.default_rng(seed)

    # 对角元：负值（鼓励 x_i = 1）
    diagonal = -rng.uniform(0.5, 2.0, size=size)

    # 非对角元：正值（耦合项，鼓励 x_i ≠ x_j）
    off_diag = rng.uniform(0.0, 1.0, size=(size, size))
    off_diag = np.triu(off_diag, k=1)  # 上三角

    # 组合成对称矩阵
    Q = np.diag(diagonal) + off_diag + off_diag.T  # noqa: N806

    logger.info(
        f"[QUBO] 生成 {size}x{size} QUBO 矩阵, "
        f"对角元范围=[{diagonal.min():.3f}, {diagonal.max():.3f}], "
        f"非零耦合项={int(np.count_nonzero(off_diag))}"
    )
    return Q


# ---------------------------------------------------------------------------
# 蛮力求解（精确最优）
# ---------------------------------------------------------------------------


def brute_force_solve(qubo_matrix: np.ndarray) -> dict[str, Any]:
    """蛮力枚举所有 2^n 个解，找到精确最优。

    适用于 n <= 20 的 QUBO 问题。

    Args:
        qubo_matrix: QUBO 矩阵 (n x n)

    Returns:
        求解结果字典: {bitstring, energy, solve_time_sec, total_evaluated}
    """
    n = qubo_matrix.shape[0]
    total = 2**n
    t0 = time.perf_counter()

    best_energy = float("inf")
    best_solution = np.zeros(n)

    for i in range(total):
        x = np.array([(i >> bit) & 1 for bit in range(n)], dtype=np.float64)
        energy = float(x @ qubo_matrix @ x)
        if energy < best_energy:
            best_energy = energy
            best_solution = x.copy()

    solve_time = time.perf_counter() - t0
    bitstring = "".join(str(int(b)) for b in best_solution)

    logger.info(
        f"[BruteForce] 最优能量={best_energy:.6f}, "
        f"比特串={bitstring}, 耗时={solve_time:.4f}s, 枚举数={total}"
    )

    return {
        "method": "brute_force",
        "bitstring": bitstring,
        "energy": round(best_energy, 6),
        "solve_time_sec": round(solve_time, 4),
        "total_evaluated": total,
    }


# ---------------------------------------------------------------------------
# 模拟退火求解
# ---------------------------------------------------------------------------


def simulated_annealing_solve(
    qubo_matrix: np.ndarray,
    num_reads: int = 100,
    seed: int = 42,
) -> dict[str, Any]:
    """使用 numpy 模拟退火求解 QUBO 问题。

    运行 num_reads 次独立退火，取能量最低的解。

    Args:
        qubo_matrix: QUBO 矩阵
        num_reads: 退火采样次数
        seed: 随机种子

    Returns:
        求解结果字典: {bitstring, energy, solve_time_sec, num_reads, energy_history}
    """
    n = qubo_matrix.shape[0]
    rng = np.random.default_rng(seed)

    best_energy = float("inf")
    best_solution = np.zeros(n)
    all_energies: list[float] = []
    energy_history: list[float] = []

    t0 = time.perf_counter()

    for read_idx in range(num_reads):
        # 随机初始化
        current = rng.integers(0, 2, n).astype(np.float64)
        current_energy = float(current @ qubo_matrix @ current)

        # 模拟退火参数
        temperature = 2.0
        cooling_rate = 0.995
        num_sweeps = 200

        for _sweep in range(num_sweeps):
            for _ in range(n):
                flip_idx = int(rng.integers(0, n))
                # 直接计算翻转后的能量差（避免近似公式误差）
                old_val = current[flip_idx]
                current[flip_idx] = 1.0 - old_val
                new_energy = float(current @ qubo_matrix @ current)
                delta_energy = new_energy - current_energy

                if delta_energy < 0 or rng.random() < np.exp(
                    -delta_energy / max(temperature, 1e-12)
                ):
                    # 接受翻转
                    current_energy = new_energy
                else:
                    # 拒绝翻转，恢复
                    current[flip_idx] = old_val

            temperature *= cooling_rate

        all_energies.append(current_energy)
        if current_energy < best_energy:
            best_energy = current_energy
            best_solution = current.copy()

        # 记录前 20 次的能量（用于收敛曲线）
        if read_idx < 20:
            energy_history.append(round(best_energy, 6))

    solve_time = time.perf_counter() - t0
    bitstring = "".join(str(int(b)) for b in best_solution)

    logger.info(
        f"[SimAnnealing] 最优能量={best_energy:.6f}, "
        f"比特串={bitstring}, 耗时={solve_time:.4f}s, "
        f"采样数={num_reads}, 平均能量={np.mean(all_energies):.4f}"
    )

    return {
        "method": "simulated_annealing_numpy",
        "bitstring": bitstring,
        "energy": round(best_energy, 6),
        "solve_time_sec": round(solve_time, 4),
        "num_reads": num_reads,
        "avg_energy": round(float(np.mean(all_energies)), 6),
        "min_energy": round(float(np.min(all_energies)), 6),
        "max_energy": round(float(np.max(all_energies)), 6),
        "energy_history": energy_history,
    }


# ---------------------------------------------------------------------------
# D-Wave neal 求解（如可用）
# ---------------------------------------------------------------------------


def dwave_neal_solve(
    qubo_matrix: np.ndarray,
    num_reads: int = 100,
    annealing_time: float = 1.0,
) -> dict[str, Any] | None:
    """使用 D-Wave neal 模拟退火求解器求解 QUBO。

    需要安装 dwave-neal 包。如不可用则返回 None。

    Args:
        qubo_matrix: QUBO 矩阵
        num_reads: 采样次数
        annealing_time: 退火时间

    Returns:
        求解结果字典，或 None（SDK 不可用时）
    """
    try:
        import dimod  # type: ignore[import-untyped]
        import neal  # type: ignore[import-untyped]
    except ImportError:
        logger.info("[DWave] neal SDK 不可用，跳过 D-Wave neal 求解")
        return None

    n = qubo_matrix.shape[0]

    # 转换为 dimod QUBO 字典格式
    qubo_dict: dict[tuple[int, int], float] = {}
    for i in range(n):
        for j in range(i, n):
            val = float(qubo_matrix[i, j])
            if abs(val) > 1e-12:
                qubo_dict[(i, j)] = val

    t0 = time.perf_counter()

    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_qubo(
        qubo_dict,
        num_reads=num_reads,
        annealing_time=annealing_time,
    )

    solve_time = time.perf_counter() - t0

    best_sample = sampleset.first.sample
    best_energy = float(sampleset.first.energy)
    bitstring = "".join(str(best_sample[i]) for i in range(n))

    # 收集所有样本能量统计
    all_energies = [float(e) for e in sampleset.record.energy]

    logger.info(
        f"[DWave-neal] 最优能量={best_energy:.6f}, "
        f"比特串={bitstring}, 耗时={solve_time:.4f}s, "
        f"采样数={num_reads}, 平均能量={np.mean(all_energies):.4f}"
    )

    return {
        "method": "dwave_neal",
        "bitstring": bitstring,
        "energy": round(best_energy, 6),
        "solve_time_sec": round(solve_time, 4),
        "num_reads": num_reads,
        "annealing_time": annealing_time,
        "avg_energy": round(float(np.mean(all_energies)), 6),
        "min_energy": round(float(np.min(all_energies)), 6),
        "max_energy": round(float(np.max(all_energies)), 6),
    }


# ---------------------------------------------------------------------------
# 真机量子电路任务提交
# ---------------------------------------------------------------------------


def submit_real_machine_tasks(
    client: Any,
    machine_name: str,
    count: int = REAL_TASK_COUNT,
    shots: int = REAL_SHOTS,
) -> list[dict[str, Any]]:
    """提交真机量子电路任务验证后端可用性。

    天衍-176 为门控量子计算机，不支持 QUBO 退火提交。
    此函数提交 H 门量子电路任务，验证量子后端连通性和性能。

    Args:
        client: 真机客户端 (CqlibTianyanClient 或 MockSmokeClient)
        machine_name: 机器名称
        count: 提交任务数
        shots: 每任务 shots 数

    Returns:
        任务记录列表
    """
    records: list[dict[str, Any]] = []

    for i in range(count):
        record: dict[str, Any] = {
            "task_index": i + 1,
            "machine": machine_name,
            "qcis": QCIS_CIRCUIT,
            "shots": shots,
            "task_name": f"AnnealVerify_{i+1}",
            "real_task_id": None,
            "submit_status": "pending",
            "submit_time": datetime.now().astimezone().isoformat(),
            "real_probability": {},
            "mock_probability": {"0": 0.5, "1": 0.5},
            "probability_diff": None,
            "fidelity": None,
            "poll_status": "pending",
        }

        try:
            t0 = time.perf_counter()
            real_tid = client.submit_quantum_task(
                qcis=QCIS_CIRCUIT,
                shots=shots,
                task_name=f"AnnealVerify_{i+1}",
            )
            submit_latency = time.perf_counter() - t0

            record["real_task_id"] = str(real_tid) if real_tid else None
            record["submit_status"] = "submitted" if real_tid else "rejected"
            record["submit_latency_s"] = round(submit_latency, 3)

            logger.info(
                f"[AnnealVerify] 任务 {i+1}/{count} 已提交: "
                f"tid={real_tid}, 延迟={submit_latency:.3f}s"
            )
        except Exception as e:
            record["submit_status"] = f"error: {str(e)[:80]}"
            logger.error(f"[AnnealVerify] 任务 {i+1} 提交失败: {e}")

        records.append(record)

    return records


# ---------------------------------------------------------------------------
# 轮询真机结果
# ---------------------------------------------------------------------------


def poll_real_results(
    client: Any,
    records: list[dict[str, Any]],
    per_task_timeout: int = 60,
) -> None:
    """轮询所有真机任务结果。

    Args:
        client: 真机客户端
        records: 任务记录列表（原地修改）
        per_task_timeout: 单任务超时秒数
    """
    total = len(records)
    logger.info(f"[Poll] 开始轮询 {total} 个真机任务结果")

    for i, record in enumerate(records):
        task_id = record.get("real_task_id")
        if not task_id:
            print(f"  [{i+1}/{total}] {record['task_name']} ... [SKIP] 无 task_id")
            continue

        print(f"  [{i+1}/{total}] {record['task_name']} {task_id} ...", end=" ", flush=True)

        result = poll_task_result(
            client=client,
            task_id=task_id,
            timeout=per_task_timeout,
            poll_interval=3,
            max_unknown=3,
            per_poll_timeout=15,
        )

        if result.get("status") == "completed":
            probability = {}
            raw_data = result.get("raw", {})

            if isinstance(raw_data, dict):
                raw_prob = raw_data.get("probability")
                if raw_prob:
                    probability = parse_probability(raw_prob)
                if not probability:
                    result_status = raw_data.get("resultStatus")
                    if result_status:
                        probability = compute_probability_from_shots(result_status)

            if not probability and result.get("result"):
                probability = parse_probability(result["result"])

            mock_prob = record.get("mock_probability", {"0": 0.5, "1": 0.5})
            prob_diff = compute_measurement_error(probability, mock_prob)
            fidelity = compute_fidelity(probability, mock_prob)

            record["real_probability"] = probability
            record["probability_diff"] = round(prob_diff, 4)
            record["fidelity"] = round(fidelity, 4)
            record["poll_status"] = "completed"
            record["complete_time"] = datetime.now().astimezone().isoformat()
            print(f"[PASS] fid={record['fidelity']}")
        else:
            record["poll_status"] = result.get("status", "unknown")
            record["real_probability"] = {}
            print(f"[FAIL] {record['poll_status']}")


# ---------------------------------------------------------------------------
# 结果保存
# ---------------------------------------------------------------------------


def save_results(
    qubo_matrix: np.ndarray,
    brute_result: dict[str, Any],
    sim_result: dict[str, Any],
    dwave_result: dict[str, Any] | None,
    real_records: list[dict[str, Any]],
) -> str:
    """保存退火验证结果到 JSON。

    Args:
        qubo_matrix: QUBO 矩阵
        brute_result: 蛮力求解结果
        sim_result: 模拟退火结果
        dwave_result: D-Wave neal 结果 (可为 None)
        real_records: 真机任务记录

    Returns:
        保存的文件路径
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = RESULTS_DIR / f"annealing_{timestamp}.json"

    # 计算真机汇总
    total_real = len([r for r in real_records if r.get("real_task_id")])
    completed_real = len([r for r in real_records if r.get("poll_status") == "completed"])
    fidelities = [r["fidelity"] for r in real_records if r.get("fidelity") is not None]

    # 计算与蛮力最优的差距
    brute_energy = brute_result["energy"]
    sim_gap = sim_result["energy"] - brute_energy
    dwave_gap = dwave_result["energy"] - brute_energy if dwave_result else None

    summary = {
        "test_type": "annealing_validation",
        "timestamp": datetime.now().astimezone().isoformat(),
        "config": {
            "qubo_size": QUBO_SIZE,
            "num_reads": NUM_READS,
            "annealing_time": ANNEALING_TIME,
            "seed": SEED,
            "real_shots": REAL_SHOTS,
            "real_task_count": REAL_TASK_COUNT,
            "qcis": QCIS_CIRCUIT,
        },
        "qubo_matrix": qubo_matrix.tolist(),
        "results": {
            "brute_force": brute_result,
            "simulated_annealing": sim_result,
            "dwave_neal": dwave_result,
        },
        "comparison": {
            "brute_force_energy": brute_energy,
            "sim_annealing_energy": sim_result["energy"],
            "dwave_neal_energy": dwave_result["energy"] if dwave_result else None,
            "sim_gap_to_optimal": round(sim_gap, 6),
            "dwave_gap_to_optimal": round(dwave_gap, 6) if dwave_gap is not None else None,
            "sim_optimal_found": bool(sim_gap < 1e-6),
            "dwave_optimal_found": bool(dwave_gap is not None and dwave_gap < 1e-6),
        },
        "real_machine": {
            "total_submitted": total_real,
            "completed": completed_real,
            "failed": total_real - completed_real,
            "avg_fidelity": (
                round(sum(fidelities) / max(len(fidelities), 1), 4) if fidelities else None
            ),
            "tasks": real_records,
        },
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info(f"[Anneal] 结果已保存: {filepath}")
    return str(filepath)


# ---------------------------------------------------------------------------
# 汇总打印
# ---------------------------------------------------------------------------


def print_summary(
    brute_result: dict[str, Any],
    sim_result: dict[str, Any],
    dwave_result: dict[str, Any] | None,
    real_records: list[dict[str, Any]],
) -> None:
    """打印退火验证汇总表。

    Args:
        brute_result: 蛮力求解结果
        sim_result: 模拟退火结果
        dwave_result: D-Wave neal 结果
        real_records: 真机任务记录
    """
    print(f"\n{'=' * 80}")
    print("  退火算法真机验证 - 汇总报告")
    print(f"{'=' * 80}")

    # QUBO 求解对比
    print(
        f"\n  {'方法':<28s} {'最优能量':>12s} {'耗时(s)':>10s} "
        f"{'与最优差距':>12s} {'找到最优':>10s}"
    )
    print(f"  {'-'*28} {'-'*12} {'-'*10} {'-'*12} {'-'*10}")

    brute_e = brute_result["energy"]
    print(
        f"  {'Brute Force (exact)':<28s} {brute_e:>12.6f} "
        f"{brute_result['solve_time_sec']:>10.4f} {'0.000000':>12s} {'Y':>10s}"
    )

    sim_e = sim_result["energy"]
    sim_gap = sim_e - brute_e
    print(
        f"  {'Sim Annealing (numpy)':<28s} {sim_e:>12.6f} "
        f"{sim_result['solve_time_sec']:>10.4f} {sim_gap:>12.6f} "
        f"{'Y' if sim_gap < 1e-6 else 'N':>10s}"
    )

    if dwave_result:
        dw_e = dwave_result["energy"]
        dw_gap = dw_e - brute_e
        print(
            f"  {'D-Wave neal':<28s} {dw_e:>12.6f} "
            f"{dwave_result['solve_time_sec']:>10.4f} {dw_gap:>12.6f} "
            f"{'Y' if dw_gap < 1e-6 else 'N':>10s}"
        )
    else:
        print(f"  {'D-Wave neal':<28s} {'N/A':>12s} {'N/A':>10s} " f"{'N/A':>12s} {'N/A':>10s}")

    # 真机验证
    total_real = len([r for r in real_records if r.get("real_task_id")])
    completed = len([r for r in real_records if r.get("poll_status") == "completed"])
    fidelities = [r["fidelity"] for r in real_records if r.get("fidelity") is not None]
    avg_fid = sum(fidelities) / len(fidelities) if fidelities else 0.0

    print(f"\n  真机量子电路验证:")
    print(f"    提交任务: {total_real}")
    print(f"    成功完成: {completed}")
    print(f"    平均保真度: {avg_fid:.4f}" if fidelities else "    平均保真度: N/A")

    print(f"\n  注: 天衍-176 为门控量子计算机，不支持直接 QUBO 退火提交。")
    print(f"  真机任务用于验证量子后端连通性，退火求解在本地完成。")

    print(f"{'=' * 80}")


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    """退火算法真机验证主入口。"""
    import argparse

    parser = argparse.ArgumentParser(description="退火算法真机验证")
    parser.add_argument("--mock", action="store_true", help="Mock dry-run")
    parser.add_argument("--machine", default="tianyan176", help="首选机器")
    parser.add_argument("--verbose", action="store_true", help="DEBUG 日志")
    args = parser.parse_args()

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    print(f"\n{'=' * 60}")
    print("  退火算法真机验证 (阶段 3)")
    print(
        f"  QUBO 规模: {QUBO_SIZE}x{QUBO_SIZE} | num_reads: {NUM_READS} | "
        f"annealing_time: {ANNEALING_TIME}"
    )
    print(f"  真机任务: {REAL_TASK_COUNT} x H门 (shots={REAL_SHOTS})")
    print(f"{'=' * 60}")

    # ── 步骤 1: 生成 QUBO 问题 ──
    print("\n--- [1/5] 生成 QUBO 问题 ---")
    qubo_matrix = generate_qubo_problem(QUBO_SIZE, seed=SEED)

    # ── 步骤 2: 蛮力求解 ──
    print("\n--- [2/5] 蛮力求解 (精确最优) ---")
    brute_result = brute_force_solve(qubo_matrix)

    # ── 步骤 3: 模拟退火求解 ──
    print("\n--- [3/5] 模拟退火求解 (numpy) ---")
    sim_result = simulated_annealing_solve(qubo_matrix, num_reads=NUM_READS, seed=SEED)

    # ── 步骤 4: D-Wave neal 求解（如可用）──
    print("\n--- [4/5] D-Wave neal 求解 ---")
    dwave_result = dwave_neal_solve(qubo_matrix, num_reads=NUM_READS, annealing_time=ANNEALING_TIME)
    if dwave_result is None:
        print("  D-Wave neal SDK 不可用，跳过")

    # ── 步骤 5: 真机量子电路任务 ──
    print(f"\n--- [5/5] 真机量子电路验证 ({REAL_TASK_COUNT} 个任务) ---")
    if args.mock:
        print("[Mode] Mock dry-run")
        client: Any = MockSmokeClient(machine_name=args.machine, mock_delay=0.01)
    else:
        print("[Mode] 真机执行")
        api_key = os.environ.get("TIANYAN_API_KEY", "")
        if not api_key:
            print("[FAIL] 未设置 TIANYAN_API_KEY")
            sys.exit(1)
        client = CqlibTianyanClient(
            login_key=api_key,
            machine_name=args.machine,
            auto_retry_machine=True,
        )
        print(f"[Setup] 真机客户端已创建: {args.machine}")

    # 提交真机任务
    real_records = submit_real_machine_tasks(
        client=client,
        machine_name=args.machine,
        count=REAL_TASK_COUNT,
        shots=REAL_SHOTS,
    )

    # 轮询真机结果
    total_real = len([r for r in real_records if r.get("real_task_id")])
    if total_real > 0:
        print(f"\n  轮询 {total_real} 个真机任务结果:")
        poll_real_results(client, real_records)

    # 保存结果
    filepath = save_results(
        qubo_matrix=qubo_matrix,
        brute_result=brute_result,
        sim_result=sim_result,
        dwave_result=dwave_result,
        real_records=real_records,
    )

    # 打印汇总
    print_summary(brute_result, sim_result, dwave_result, real_records)
    print(f"\n  结果文件: {filepath}")


if __name__ == "__main__":
    main()
