#!/usr/bin/env python
"""
编译层真机验证 —— Smoke Test（实验②预注册门槛）

验证"PPO/SABRE 编译布局 → 物理比特映射 → 真机执行 → 保真度对比"全链路可行性。
通过门槛后才启动正式实验；不通过则实验②终止并如实记录（预注册条款）。

三个预检（依次执行，任一失败即停止）：
    A. 天衍-287 多比特电路可执行性：4 比特 CZ 链 + CNOT 用例（QCIS 语法验证）
    B. 物理比特邻接/布局忠实度：CZ Q1 Q2（近邻）vs CZ Q1 Q3（可能非邻接），
       观察平台行为（完成/失败/自动路由）→ 决定"显式布局"方法是否可行
    C. 方向性预检：3 个深电路 × SABRE/PPO 布局 × 1 次真机执行（6 任务），
       比较 fidelity（测量分布 vs qiskit 无噪声理论分布）

技术路径（调研结论）：
    - 电路池：受控门集（QAOA 风格 h/rx/ry/rz/cx）深电路 14-16 qubits——原
      compilation_deep_scale 的 random_circuit 池为 qiskit 全门集（含 ccx/cswap/c3sx
      等 3-4 qubit 门，50+ 种门），无法转译 QCIS，真机验证必须用受控门集
    - SABRE：qiskit SabreLayout(4x4 CouplingMap, seed)+SabreSwap → transpiled 电路
    - PPO：QuantumCompilationEnv + ppo_compilation_agent.zip 决策 → env._mapping（逻辑→物理），
      再用标准最短路径 SWAP 重放重建物理电路（env 只做映射不做 SWAP 序列，需重建）
    - 物理映射：4x4 抽象网格 → 真机 Q1..Q16（偏移映射；邻接结构由预检 B 验证）
    - 指标：fidelity = 真机测量分布 vs qiskit statevector 理论分布（smoke_test.compute_fidelity）
    - 注意：MBS（测量平衡分）只适用于单比特 50/50 态，多比特电路不可用，改用 fidelity

用法：
    # 预检 A（2 个任务）
    python scripts/real_machine/compilation_real_smoke.py --check-a
    # 预检 B（2 个任务）
    python scripts/real_machine/compilation_real_smoke.py --check-b
    # 预检 C（6 个任务）
    python scripts/real_machine/compilation_real_smoke.py --check-c
    # 全流程（A→B→C 串行，任一失败 exit 1）
    python scripts/real_machine/compilation_real_smoke.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
for p in [
    _PROJECT_ROOT,
    str(_PROJECT_ROOT / "scripts" / "evaluation"),
    str(_PROJECT_ROOT / "scripts" / "real_machine"),
]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")


from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap
from smoke_test import compute_fidelity, parse_probability

from src.api.tianyan_cqlib import CqlibTianyanClient
from src.quantum.compilation_env import (
    GRID_COLS,
    GRID_ROWS,
    QuantumCompilationEnv,
    _build_2d_grid_coupling,
)

# ── 协议常量 ──
TARGET_MACHINE = "tianyan-287"
SHOTS = 1024  # 多比特分布分辨需要（v2 的 32 仅适用于单比特 MBS）
TASK_TIMEOUT_SECONDS = 300
TASK_POLL_INTERVAL = 5
DEEP_POOL_SEED = 20260810  # 与 results/compilation_tail_per_circuit.json 同源
N_DEEP_SMOKE = 3  # 预检 C 电路数（SABRE 高成本 top3）
OUTPUT_DIR = _PROJECT_ROOT / "results" / "real_machine"

GRID_COUPLING = _build_2d_grid_coupling(GRID_ROWS, GRID_COLS)
COUPLING_4x4 = CouplingMap([[i, j] for i, ns in GRID_COUPLING.items() for j in ns if i < j])


def qiskit_to_qcis(qc: QuantumCircuit, qubit_offset: int = 0) -> str:
    """qiskit 电路 → QCIS 指令字符串（物理比特编号 = 电路索引 + offset，Q1 起）。"""
    lines: list[str] = []
    measured = False
    dropped: list[str] = []
    gate_map = {
        "h": "H",
        "x": "X",
        "y": "Y",
        "z": "Z",
        "s": "S",
        "sdg": "SDG",
        "t": "T",
        "tdg": "TDG",
        "rx": "RX",
        "ry": "RY",
        "rz": "RZ",
        "cx": "CNOT",
        "cz": "CZ",
        "swap": "SWAP",
    }
    for instr in qc.data:
        name = instr.operation.name
        qubits = [qc.find_bit(q).index + qubit_offset + 1 for q in instr.qubits]
        if name == "measure":
            lines.append("M " + " ".join(f"Q{q}" for q in qubits))
            measured = True
            continue
        if name == "barrier":
            continue  # 无物理语义，跳过
        if measured:
            # SABRE 高成本电路 transpile 后 SWAP 可能落在 measure 之后——
            # 测量后的门无物理意义，丢弃（warning 级，不阻断）
            dropped.append(name)
            continue
        gname = gate_map.get(name)
        if gname is None:
            raise ValueError(f"不支持的门: {name}（需映射到 QCIS 门集）")
        params = instr.operation.params
        if gname in ("RX", "RY", "RZ"):
            if len(params) != 1:
                raise ValueError(f"参数门 {name} 需 1 参数")
            lines.append(f"{gname} Q{qubits[0]} {float(params[0]):.8f}")
        elif name == "swap":
            # SWAP = 3×CNOT 展开（平台可能不支持原生 SWAP）
            lines.append(f"CNOT Q{qubits[0]} Q{qubits[1]}")
            lines.append(f"CNOT Q{qubits[1]} Q{qubits[0]}")
            lines.append(f"CNOT Q{qubits[0]} Q{qubits[1]}")
        else:
            qs = " ".join(f"Q{q}" for q in qubits)
            lines.append(f"{gname} {qs}")
    if dropped:
        print(f"  [warn] qiskit_to_qcis: 丢弃测量后无意义门 {len(dropped)} 个（{set(dropped)}）")
    return "\n".join(lines)


def rebuild_physical_circuit(original: QuantumCircuit, mapping: dict[int, int]) -> QuantumCircuit:
    """用 PPO 最终映射作为初始布局，SabreSwap 标准插入 SWAP，重建 4x4 物理电路。

    方法学（2026-08-14 修正）：QuantumCompilationEnv 只输出映射决策（_mapping），
    不生成最终电路；手写最短路径 SWAP 重放会引入与 fair_v2 仿真口径不一致的
    额外 SWAP。改用 qiskit SetLayout 固定 PPO 映射为初始布局 + SabreSwap 标准
    插入——SABRE 与 PPO 流程的唯一差异 = 初始布局（对比更干净、可复现）。
    """
    from qiskit.transpiler import Layout
    from qiskit.transpiler.passes import SabreSwap as _SabreSwap
    from qiskit.transpiler.passes import SetLayout

    # mapping: 逻辑索引 -> 物理索引；Layout 需要 Qubit -> int
    layout = Layout({original.qubits[logical]: physical for logical, physical in mapping.items()})
    pm = PassManager([SetLayout(layout), _SabreSwap(COUPLING_4x4, trials=8, seed=42)])
    return pm.run(original)


def theoretical_distribution(qc: QuantumCircuit) -> dict[str, float]:
    """qiskit statevector 无噪声理论测量分布（按物理比特全测量）。

    显式过滤 measure/barrier 重建无测量电路（SABRE 高成本电路的 transpile 结果中
    SWAP 可能落在 measure 之后，remove_final_measurements 无法清理）。
    probabilities()[i] 的 bit j（LSB 起）对应 qubit j；format 后最右字符 = qubit 0。
    注意：真机返回的 bitstring 约定可能相反——预检 C 后若 fidelity 异常低，
    需做比特序校准（见 smoke 方案文档 §5）。
    """
    no_meas = QuantumCircuit(qc.num_qubits)
    for ins in qc.data:
        name = ins.operation.name
        if name in ("measure", "barrier"):
            continue
        qargs = [no_meas.qubits[qc.find_bit(q).index] for q in ins.qubits]
        no_meas.append(ins.operation, qargs)
    sv = Statevector(no_meas)
    probs = sv.probabilities()
    n = len(no_meas.qubits)
    return {format(i, f"0{n}b"): float(p) for i, p in enumerate(probs) if p > 0}


def submit_and_poll(client: CqlibTianyanClient, qcis: str, task_name: str) -> dict[str, Any]:
    """提交 QCIS 电路并轮询（复用 v2 重试纪律）。"""
    from scripts.real_machine import with_retry

    record: dict[str, Any] = {
        "task_id": None,
        "status": None,
        "probability": None,
        "error": None,
        "submitted_at": None,
        "completed_at": None,
    }
    try:
        platform = getattr(client, "platform", None)
        if (
            platform is not None
            and hasattr(platform, "qcis_check_regular")
            and not platform.qcis_check_regular(qcis)
        ):
            record["status"] = "failed"
            record["error"] = "QCIS 预校验失败"
            return record
    except Exception as e:
        record["status"] = "query_error"
        record["error"] = f"QCIS 校验异常: {str(e)[:100]}"
        return record
    try:
        task_id = with_retry(
            client.submit_quantum_task,
            qcis=qcis,
            shots=SHOTS,
            task_name=task_name,
            max_retries=3,
            base_delay=2.0,
        )
    except Exception as e:
        record["status"] = "failed"
        record["error"] = f"提交异常: {str(e)[:150]}"
        return record
    if task_id is None:
        record["status"] = "failed"
        record["error"] = "submit 返回 None"
        return record
    record["task_id"] = str(task_id)
    record["submitted_at"] = datetime.now().isoformat()
    try:
        result = client.wait_for_task(
            task_id, max_wait_time=TASK_TIMEOUT_SECONDS, sleep_time=TASK_POLL_INTERVAL
        )
        record["status"] = result.status
        if result.status == "completed":
            prob = parse_probability(result.probability) if result.probability else {}
            record["probability"] = prob
    except Exception as e:
        record["status"] = "timeout"
        record["error"] = str(e)[:150]
    record["completed_at"] = datetime.now().isoformat()
    return record


def make_client() -> CqlibTianyanClient:
    api_key = os.environ.get("TIANYAN_API_KEY", "")
    if not api_key:
        raise SystemExit("❌ 未设置 TIANYAN_API_KEY")
    client = CqlibTianyanClient(
        login_key=api_key, machine_name=TARGET_MACHINE, auto_retry_machine=False
    )
    if getattr(client, "machine_name", TARGET_MACHINE) != TARGET_MACHINE:
        raise SystemExit("❌ 客户端机器不一致（禁止回退）")
    return client


def check_a(client: CqlibTianyanClient) -> list[dict[str, Any]]:
    """预检 A：多比特电路可执行性（CZ 链 + CNOT）。"""
    print("=== 预检 A：天衍-287 多比特电路可执行性 ===")
    circuits = {
        "A1_cz_chain_4q": "H Q1\nH Q2\nH Q3\nH Q4\nCZ Q1 Q2\nCZ Q2 Q3\nCZ Q3 Q4\nM Q1 Q2 Q3 Q4",
        "A2_cnot_pair": "H Q1\nH Q2\nCNOT Q1 Q2\nM Q1 Q2",
    }
    results = []
    for name, qcis in circuits.items():
        r = submit_and_poll(client, qcis, f"cmp_smoke_{name}")
        r["name"] = name
        r["qcis"] = qcis
        results.append(r)
        print(f"  {name}: status={r['status']} task_id={r['task_id']}")
    return results


def check_b(client: CqlibTianyanClient) -> list[dict[str, Any]]:
    """预检 B：物理比特邻接/布局忠实度探测。"""
    print("=== 预检 B：物理比特邻接与布局忠实度 ===")
    circuits = {
        "B1_near_Q1_Q2": "H Q1\nH Q2\nCZ Q1 Q2\nM Q1 Q2",
        "B2_far_Q1_Q3": "H Q1\nH Q3\nCZ Q1 Q3\nM Q1 Q3",
    }
    results = []
    for name, qcis in circuits.items():
        r = submit_and_poll(client, qcis, f"cmp_smoke_{name}")
        r["name"] = name
        results.append(r)
        print(f"  {name}: status={r['status']} task_id={r['task_id']}")
    return results


def generate_qaoa_style_circuits(
    n: int, seed: int, qubits_range=(14, 16), gates_range=(20, 30)
) -> list[QuantumCircuit]:
    """生成纯平台门集深电路（QAOA 风格：h/rx/ry/rz/cx，全 1-2 qubit）。

    调研发现（2026-08-14）：compilation_deep_scale/fair_v2 的 random_circuit 池
    使用 qiskit 全门集（50+ 种门，100% 电路含 ccx/cswap/c3sx 等 3-4 qubit 门），
    无法转译 QCIS（天衍仅支持 H/X/Y/Z/RX/RY/RZ/CNOT/CZ/M）——真机验证必须用
    受控门集电路。本生成器 seed 派生可复现，正式实验②沿用同一生成器。
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    one_q = ("h", "rx", "ry", "rz")
    circuits: list[QuantumCircuit] = []
    for i in range(n):
        nq = int(rng.integers(qubits_range[0], qubits_range[1] + 1))
        ng = int(rng.integers(gates_range[0], gates_range[1] + 1))
        qc = QuantumCircuit(nq, nq)
        sub = np.random.default_rng(seed * 1000 + i)
        for _ in range(ng):
            if sub.random() < 0.4:  # 2q 门占比 40%（深电路特征）
                a, b = sub.choice(nq, size=2, replace=False)
                qc.cx(int(a), int(b))
            else:
                q = int(sub.integers(0, nq))
                g = one_q[int(sub.integers(0, len(one_q)))]
                if g == "h":
                    qc.h(q)
                elif g == "rx":
                    qc.rx(float(sub.uniform(0, 2 * np.pi)), q)
                elif g == "ry":
                    qc.ry(float(sub.uniform(0, 2 * np.pi)), q)
                else:
                    qc.rz(float(sub.uniform(0, 2 * np.pi)), q)
        for q in range(nq):
            qc.measure(q, q)
        circuits.append(qc)
    return circuits


def check_c(client: CqlibTianyanClient) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """预检 C：3 深电路 × SABRE/PPO 布局真机保真度方向性预检。"""
    print("=== 预检 C：深电路 PPO vs SABRE 真机保真度方向性 ===")
    from stable_baselines3 import PPO

    # 受控门集电路池（QAOA 风格，可转 QCIS），SABRE 评分选高成本 top3
    circuits = generate_qaoa_style_circuits(40, DEEP_POOL_SEED)
    scored: list[tuple[int, QuantumCircuit]] = []
    for qc in circuits:
        pm = PassManager(
            [
                SabreLayout(COUPLING_4x4, swap_trials=8, layout_trials=8, seed=42),
                SabreSwap(COUPLING_4x4, trials=8, seed=42),
            ]
        )
        compiled = pm.run(qc)
        scored.append((compiled.count_ops().get("swap", 0), qc))
    scored.sort(key=lambda t: -t[0])
    top3 = [qc for _, qc in scored[:N_DEEP_SMOKE]]
    print(
        f"  电路池: 40 个（QAOA 风格，可转 QCIS）| 选 SABRE 高成本 top{N_DEEP_SMOKE}（SWAP="
        f"{[s for s, _ in scored[:N_DEEP_SMOKE]]}）"
    )

    model = PPO.load(str(_PROJECT_ROOT / "deliverable_models" / "ppo_compilation_agent.zip"))
    records: list[dict[str, Any]] = []
    setups: list[dict[str, Any]] = []

    for i, qc in enumerate(top3):
        # SABRE 编译
        pm = PassManager(
            [
                SabreLayout(COUPLING_4x4, swap_trials=8, layout_trials=8, seed=42),
                SabreSwap(COUPLING_4x4, trials=8, seed=42),
            ]
        )
        sabre_qc = pm.run(qc)
        # PPO 编译（env 决策映射 + 最短路径 SWAP 重建）
        env = QuantumCompilationEnv(qc, max_steps=200)
        obs, _ = env.reset(seed=42)
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
        ppo_qc = rebuild_physical_circuit(qc, dict(env._mapping))

        for label, compiled_qc in (("sabre", sabre_qc), ("ppo", ppo_qc)):
            qcis = qiskit_to_qcis(compiled_qc)
            theoretical = theoretical_distribution(compiled_qc)
            r = submit_and_poll(client, qcis, f"cmp_smoke_c{i}_{label}")
            r["circuit_index"] = i
            r["layout"] = label
            r["n_qubits"] = len(qc.qubits)
            r["n_gates"] = len(qc.data) - len(qc.qubits)  # 去除 measure 行数近似
            if r["status"] == "completed" and r.get("probability"):
                r["fidelity"] = round(compute_fidelity(r["probability"], theoretical), 4)
            else:
                r["fidelity"] = None
            records.append(r)
            setups.append(
                {
                    "circuit_index": i,
                    "layout": label,
                    "qcis": qcis,
                    "n_2q_gates": sum(1 for ins in compiled_qc.data if len(ins.qubits) == 2),
                    "depth": compiled_qc.depth(),
                }
            )
            print(
                f"  c{i} {label}: status={r['status']} fid={r.get('fidelity')} task={r['task_id']}"
            )
    return records, setups


def main() -> int:
    parser = argparse.ArgumentParser(description="编译层真机验证 smoke test（实验②门槛）")
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--check-a", action="store_true", help="预检 A（多比特可执行性）")
    g.add_argument("--check-b", action="store_true", help="预检 B（邻接/忠实度）")
    g.add_argument("--check-c", action="store_true", help="预检 C（方向性 6 任务）")
    g.add_argument("--all", action="store_true", help="A→B→C 全流程")
    args = parser.parse_args()

    client = make_client()
    all_out: dict[str, Any] = {
        "experiment": "compilation_real_smoke",
        "timestamp": datetime.now().isoformat(),
        "machine": TARGET_MACHINE,
        "shots": SHOTS,
    }
    passed = True
    if args.check_a or args.all:
        ra = check_a(client)
        all_out["check_a"] = ra
        a_ok = all(r["status"] == "completed" and r.get("probability") for r in ra)
        passed &= a_ok
        print(f"预检 A: {'✅ 通过' if a_ok else '❌ 失败'}")
    if args.check_b or args.all:
        rb = check_b(client)
        all_out["check_b"] = rb
        b_ok = all(r["status"] == "completed" for r in rb)
        passed &= b_ok
        print(f"预检 B: {'✅ 完成（忠实度判定见 JSON/报告）' if b_ok else '❌ 失败'}")
    if args.check_c or args.all:
        rc, setups = check_c(client)
        all_out["check_c"] = rc
        all_out["check_c_setups"] = setups
        c_ok = all(r["status"] == "completed" for r in rc)
        passed &= c_ok
        print(f"预检 C: {'✅ 全部完成（方向性见 JSON/报告）' if c_ok else '❌ 部分失败'}")
        # 方向性汇总
        fids = {(r["circuit_index"], r["layout"]): r.get("fidelity") for r in rc}
        for idx in sorted({k[0] for k in fids}):
            print(
                f"  电路{idx}: SABRE fid={fids.get((idx, 'sabre'))}  PPO fid={fids.get((idx, 'ppo'))}"
            )

    out = OUTPUT_DIR / f"compilation_real_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n结果已保存: {out}")
    print(
        f"smoke test 总体: {'✅ 通过（可启动正式实验②）' if passed else '❌ 未通过（按预注册条款终止/重试）'}"
    )
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
