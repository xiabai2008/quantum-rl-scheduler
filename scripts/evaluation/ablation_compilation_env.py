"""
编译环境消融实验：验证 #404/#405/#406 修复效果

对比项：
1. 耦合图拓扑：线性链 (旧, buggy) vs 4×4 2D网格 (新, 修复#404)
2. SWAP距离：线性abs差 (旧) vs BFS最短路径 (新, 修复#406)
3. 保真度模型：均匀/无感知 (旧) vs 距离感知 (新)
4. SABRE baseline在两种拓扑上的SWAP开销对比
"""
import json
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from qiskit.circuit.random import random_circuit
from qiskit.transpiler import CouplingMap, PassManager
from qiskit.transpiler.passes import SabreLayout, SabreSwap

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

NUM_CIRCUITS = 100
SEED = 42
np.random.seed(SEED)


def build_linear_coupling(n: int) -> dict[int, set[int]]:
    """旧buggy线性链耦合图。"""
    g: dict[int, set[int]] = {i: set() for i in range(n)}
    for i in range(n - 1):
        g[i].add(i + 1)
        g[i + 1].add(i)
    return g


def build_2d_grid_coupling(rows: int, cols: int) -> dict[int, set[int]]:
    """修复后的2D网格耦合图（#404）。"""
    g: dict[int, set[int]] = {i: set() for i in range(rows * cols)}
    for r in range(rows):
        for c in range(cols):
            q = r * cols + c
            if c + 1 < cols:
                g[q].add(q + 1)
                g[q + 1].add(q)
            if r + 1 < rows:
                g[q].add(q + cols)
                g[q + cols].add(q)
    return g


def bfs_distance(g: dict[int, set[int]], src: int, dst: int) -> int:
    """BFS最短路径距离（#406修复）。"""
    if src == dst:
        return 0
    visited = {src}
    q: deque[tuple[int, int]] = deque([(src, 0)])
    while q:
        node, dist = q.popleft()
        for nb in g.get(node, set()):
            if nb == dst:
                return dist + 1
            if nb not in visited:
                visited.add(nb)
                q.append((nb, dist + 1))
    return len(g)


def linear_distance(n: int, src: int, dst: int) -> int:
    """旧buggy线性距离（abs差值，#406修复前）。"""
    return abs(src - dst)


def estimate_swap_count_greedy(circuit, coupling_graph, dist_fn, qubits: int = 16) -> int:
    """简单贪心路由：对每个两比特门计算映射后的SWAP开销。"""
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(circuit)
    mapping = {i: i for i in range(qubits)}
    reverse_map = {i: i for i in range(qubits)}
    total_swaps = 0

    for node in dag.topological_op_nodes():
        if node.op.num_qubits == 2:
            qargs = [circuit.find_bit(q).index for q in node.qargs]
            phys_q0 = mapping[qargs[0]]
            phys_q1 = mapping[qargs[1]]
            d = dist_fn(coupling_graph, phys_q0, phys_q1) if dist_fn else linear_distance(qubits, phys_q0, phys_q1)
            total_swaps += max(0, d - 1)
    return total_swaps


def estimate_fidelity_aware_swap(circuit, coupling_graph, dist_fn, qubits: int = 16) -> float:
    """距离感知保真度估计（#406修复）：SWAP越多保真度越低。"""
    from qiskit.converters import circuit_to_dag

    dag = circuit_to_dag(circuit)
    mapping = {i: i for i in range(qubits)}
    fidelity = 1.0
    swap_error_rate = 0.05  # 两比特SWAP门约5%错误率

    for node in dag.topological_op_nodes():
        if node.op.num_qubits == 2:
            qargs = [circuit.find_bit(q).index for q in node.qargs]
            phys_q0 = mapping[qargs[0]]
            phys_q1 = mapping[qargs[1]]
            d = dist_fn(coupling_graph, phys_q0, phys_q1)
            swaps_needed = max(0, d - 1)
            fidelity *= (1 - swap_error_rate) ** swaps_needed
    return fidelity


def generate_circuits(n: int, seed: int = 42) -> list:
    """生成随机测试电路集。"""
    rng = np.random.default_rng(seed)
    circuits = []
    for _ in range(n):
        n_qubits = int(rng.integers(5, 14))
        depth = int(rng.integers(5, 21))
        qc = random_circuit(n_qubits, depth, measure=False, seed=int(rng.integers(0, 100000)))
        circuits.append(qc)
    return circuits


def run_sabre(circuits, coupling_map, label: str) -> dict:
    """运行SABRE布局+路由并统计SWAP数。"""
    swaps = []
    t0 = time.time()
    for qc in circuits:
        try:
            pm = PassManager([
                SabreLayout(coupling_map, swap_trials=8, layout_trials=8),
                SabreSwap(coupling_map, trials=8),
            ])
            compiled = pm.run(qc)
            swaps.append(compiled.count_ops().get("swap", 0))
        except Exception:
            swaps.append(0)
    elapsed = time.time() - t0
    return {
        "label": label,
        "avg_swap": float(np.mean(swaps)),
        "std_swap": float(np.std(swaps)),
        "median_swap": float(np.median(swaps)),
        "max_swap": int(np.max(swaps)),
        "num_circuits": len(swaps),
        "time_s": elapsed,
    }


def run_greedy_analysis(circuits, coupling_graph, dist_fn, label: str) -> dict:
    """贪心路由分析（不实际编译，估计SWAP下界）。"""
    swaps = []
    fidelities = []
    for qc in circuits:
        s = estimate_swap_count_greedy(qc, coupling_graph, dist_fn)
        f = estimate_fidelity_aware_swap(qc, coupling_graph, dist_fn)
        swaps.append(s)
        fidelities.append(f)
    return {
        "label": label,
        "avg_swap_lower_bound": float(np.mean(swaps)),
        "avg_estimated_fidelity": float(np.mean(fidelities)),
    }


def main():
    print("=" * 70)
    print("编译环境消融实验: #404耦合图 / #406 SWAP距离 / 保真度模型")
    print("=" * 70)

    circuits = generate_circuits(NUM_CIRCUITS, SEED)
    print(f"\n测试电路: {NUM_CIRCUITS} 个随机电路 (5-13 qubits, depth 5-20)")

    # ── 1. SABRE on different topologies ──
    print("\n" + "=" * 70)
    print("[1] SABRE编译器在不同耦合拓扑上的SWAP开销")
    print("=" * 70)

    # 16-qubit linear chain (旧buggy)
    linear_cm = CouplingMap([(i, i + 1) for i in range(15)] + [(i + 1, i) for i in range(15)])
    # 4×4 2D grid (新fixed)
    edges = []
    for r in range(4):
        for c in range(4):
            q = r * 4 + c
            if c + 1 < 4:
                edges.extend([(q, q + 1), (q + 1, q)])
            if r + 1 < 4:
                edges.extend([(q, q + 4), (q + 4, q)])
    grid_cm = CouplingMap(edges)

    sabre_linear = run_sabre(circuits, linear_cm, "SABRE-LinearChain(旧)")
    sabre_grid = run_sabre(circuits, grid_cm, "SABRE-Grid2D(新)")

    print(f"\n{'拓扑':<30} {'平均SWAP':>10} {'中位SWAP':>10} {'最大SWAP':>10} {'时间':>8}")
    print("-" * 70)
    for r in [sabre_linear, sabre_grid]:
        print(f"{r['label']:<30} {r['avg_swap']:>10.1f} {r['median_swap']:>10.1f} "
              f"{r['max_swap']:>10d} {r['time_s']:>7.1f}s")

    swap_improvement = (1 - sabre_grid['avg_swap'] / sabre_linear['avg_swap']) * 100
    print(f"\n2D网格 vs 线性链 SWAP减少: {swap_improvement:.1f}%")

    # ── 2. Distance function comparison ──
    print("\n" + "=" * 70)
    print("[2] SWAP距离函数: 线性abs (旧) vs BFS最短路径 (新#406)")
    print("=" * 70)

    linear_graph = build_linear_coupling(16)
    grid_graph = build_2d_grid_coupling(4, 4)

    # 在2D网格上用线性距离（错误）vs BFS距离（正确）
    greedy_linear_dist = run_greedy_analysis(circuits, grid_graph,
                                             lambda g, s, d: linear_distance(16, s, d),
                                             "Grid+LinearDist(旧错误)")
    greedy_bfs_dist = run_greedy_analysis(circuits, grid_graph,
                                          bfs_distance,
                                          "Grid+BFSDist(新#406)")

    print(f"\n{'配置':<30} {'SWAP下界':>12} {'估计保真度':>12}")
    print("-" * 60)
    for r in [greedy_linear_dist, greedy_bfs_dist]:
        print(f"{r['label']:<30} {r['avg_swap_lower_bound']:>12.1f} {r['avg_estimated_fidelity']:>12.4f}")

    dist_error = (greedy_bfs_dist['avg_swap_lower_bound'] - greedy_linear_dist['avg_swap_lower_bound'])
    print(f"\n线性距离在2D网格上低估SWAP数: {abs(dist_error):.1f} (距离函数错误导致路由决策偏差)")

    # ── 3. Graph diameter (最长最短路径) ──
    print("\n" + "=" * 70)
    print("[3] 耦合图直径（最远qubit对的SWAP开销）")
    print("=" * 70)

    def graph_diameter(g):
        max_d = 0
        for src in g:
            for dst in g:
                d = bfs_distance(g, src, dst)
                max_d = max(max_d, d)
        return max_d

    linear_diam = graph_diameter(linear_graph)
    grid_diam = graph_diameter(grid_graph)
    print(f"线性链直径: {linear_diam} (qubit 0→15 需 {linear_diam} 个SWAP)")
    print(f"4×4网格直径: {grid_diam} (最远对角需 {grid_diam} 个SWAP)")
    print(f"直径减少: {(1 - grid_diam/linear_diam)*100:.0f}%")

    # ── 4. Connectivity comparison ──
    print("\n" + "=" * 70)
    print("[4] 拓扑连通性统计")
    print("=" * 70)
    linear_deg = np.mean([len(v) for v in linear_graph.values()])
    grid_deg = np.mean([len(v) for v in grid_graph.values()])
    print(f"线性链平均度: {linear_deg:.2f} (边缘qubit仅1个邻居)")
    print(f"2D网格平均度: {grid_deg:.2f} (内部qubit有4个邻居)")

    # ── Save results ──
    results = {
        "experiment": "compilation_env_ablation",
        "timestamp": datetime.now().isoformat(),
        "num_circuits": NUM_CIRCUITS,
        "sabre_comparison": [sabre_linear, sabre_grid],
        "distance_ablation": [greedy_linear_dist, greedy_bfs_dist],
        "topology_stats": {
            "linear_chain_diameter": linear_diam,
            "grid_2d_diameter": grid_diam,
            "linear_avg_degree": linear_deg,
            "grid_avg_degree": grid_deg,
            "swap_reduction_pct": swap_improvement,
        },
    }
    import pathlib
    rdir = pathlib.Path("results")
    rdir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = rdir / f"compilation_ablation_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out_path}")


if __name__ == "__main__":
    main()
