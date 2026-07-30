"""
量子电路模板生成器
Quantum Circuit Template Generator

提供标准量子电路模板的 QCIS 格式生成，用于真机验证和基准测试：
- Bell 态电路（2比特纠缠）
- GHZ-n 态电路（多比特纠缠）
- VQE 变分电路模板（默认4比特，vqe4）
- QAOA 电路模板（默认5比特，qaoa5）

QCIS 指令格式（天衍云超导原生）：
- 单比特门: GATE Qn  (如 H Q0, RY Q1 1.57)
- 双比特门: GATE Qc Qt (如 CZ Q0 Q1)
- 测量: M Qn
"""

import numpy as np
from numpy.typing import NDArray


def generate_bell_state(qubit_pairs: list[tuple[int, int]] | None = None) -> str:
    """生成 Bell 态电路（2比特最大纠缠态，用于真机验证）。

    Bell 态: |Φ+⟩ = (|00⟩ + |11⟩)/√2
    电路结构: H Qc → CZ Qc Qt → M Qc → M Qt

    说明：H 作用在 control qubit 后，CZ 门即可生成 Bell 纠缠态；
    不需要在 target qubit 上再加 H 门（否则会破坏纠缠态）。

    Args:
        qubit_pairs: 纠缠比特对列表 [(control, target), ...]，
                     None 时默认生成单对 (Q0, Q1)

    Returns:
        QCIS 格式字符串
    """
    if qubit_pairs is None:
        qubit_pairs = [(0, 1)]

    instructions: list[str] = []
    all_qubits: set[int] = set()

    for qc, qt in qubit_pairs:
        instructions.append(f"H Q{qc}")
        instructions.append(f"CZ Q{qc} Q{qt}")
        all_qubits.add(qc)
        all_qubits.add(qt)

    for q in sorted(all_qubits):
        instructions.append(f"M Q{q}")

    return "\n".join(instructions)


def generate_ghz_state(n_qubits: int = 3) -> str:
    """生成 GHZ-n 态电路（多比特 Greenberger-Horne-Zeilinger 纠缠态）。

    GHZ-n: |GHZ⟩ = (|00...0⟩ + |11...1⟩)/√2
    电路结构: H Q0 → [H Q(i+1) → CZ Q(i) Q(i+1) → H Q(i+1)] × (n-1) → M all

    说明：QCIS 仅提供 CZ 双比特门，无法直接生成 GHZ 纠缠态。这里利用
    H-CZ-H 等效 CNOT 门（CNOT(Qc, Qt) = H Qt → CZ Qc Qt → H Qt），
    对每个相邻比特对施加等效 CNOT，从而把 H Q0 产生的叠加态扩展到全部比特。

    Args:
        n_qubits: 比特数，默认 3（GHZ-3）

    Returns:
        QCIS 格式字符串

    Raises:
        ValueError: n_qubits < 2 时
    """
    if n_qubits < 2:
        raise ValueError(f"GHZ 态至少需要 2 个比特，收到 n_qubits={n_qubits}")

    instructions: list[str] = ["H Q0"]

    # 使用 H-CZ-H 等效 CNOT 门生成 GHZ 纠缠态
    for i in range(n_qubits - 1):
        instructions.append(f"H Q{i + 1}")
        instructions.append(f"CZ Q{i} Q{i + 1}")
        instructions.append(f"H Q{i + 1}")

    for i in range(n_qubits):
        instructions.append(f"M Q{i}")

    return "\n".join(instructions)


def generate_vqe_circuit(
    n_qubits: int = 4,
    depth: int = 2,
    params: NDArray[np.float64] | None = None,
    two_qubit_gates: bool = True,
) -> str:
    """生成 VQE（Variational Quantum Eigensolver）变分电路模板。

    电路结构（硬件高效 ansatz）：
        每层:
            1. RY 旋转层（单比特，参数化）
            2. RZ 旋转层（单比特，参数化）
            3. CZ 纠缠层（最近邻链式纠缠，two_qubit_gates=True 时启用）
        末尾: 测量所有比特

    参数形状: params.shape = (depth, n_qubits, 2)
        params[d][q][0] = RY 角度（弧度）
        params[d][q][1] = RZ 角度（弧度）

    Args:
        n_qubits: 比特数，默认 4（vqe4，Issue #574）
        depth: 变分层数，默认 2
        params: 变分参数数组，None 时初始化为 0
        two_qubit_gates: 是否启用双比特纠缠门（CZ），默认 True

    Returns:
        QCIS 格式字符串

    Raises:
        ValueError: n_qubits < 1, depth < 1, 或 params 形状不匹配时
    """
    if n_qubits < 1:
        raise ValueError(f"VQE 电路至少需要 1 个比特，收到 n_qubits={n_qubits}")
    if depth < 1:
        raise ValueError(f"VQE 电路深度至少为 1，收到 depth={depth}")

    expected_shape = (depth, n_qubits, 2)
    if params is None:
        params = np.zeros(expected_shape, dtype=np.float64)
    elif params.shape != expected_shape:
        raise ValueError(f"params 形状应为 {expected_shape}，收到 {params.shape}")

    instructions: list[str] = []

    for d in range(depth):
        for q in range(n_qubits):
            theta_ry = float(params[d, q, 0])
            theta_rz = float(params[d, q, 1])
            instructions.append(f"RY Q{q} {theta_ry:.10f}")
            instructions.append(f"RZ Q{q} {theta_rz:.10f}")

        if two_qubit_gates and n_qubits >= 2:
            for q in range(n_qubits - 1):
                instructions.append(f"CZ Q{q} Q{q + 1}")

    for q in range(n_qubits):
        instructions.append(f"M Q{q}")

    return "\n".join(instructions)


def generate_qaoa_circuit(
    n_qubits: int = 5,
    p_layers: int = 1,
    gamma: NDArray[np.float64] | None = None,
    beta: NDArray[np.float64] | None = None,
    two_qubit_gates: bool = True,
) -> str:
    """生成 QAOA（Quantum Approximate Optimization Algorithm）电路模板。

    电路结构（硬件高效 ansatz，默认使用 CZ 双比特门）：
        初始化: H^⊗n（所有比特叠加态）
        每层 p:
            1. 问题哈密顿量演化（cost layer）:
               - RZ Qq 2*gamma[p]（单比特相位旋转）
               - CZ 最近邻链式纠缠（two_qubit_gates=True 时启用）
            2. 混合哈密顿量演化（mixer layer）: RX Qq 2*beta[p]
        末尾: 测量所有比特

    参数形状:
        gamma.shape = (p_layers,)
        beta.shape = (p_layers,)

    Args:
        n_qubits: 比特数，默认 5（qaoa5，Issue #574）
        p_layers: QAOA 层数 p，默认 1
        gamma: 问题哈密顿量演化参数，None 时初始化为 0
        beta: 混合哈密顿量演化参数，None 时初始化为 0
        two_qubit_gates: 是否启用双比特 CZ 纠缠门，默认 True

    Returns:
        QCIS 格式字符串

    Raises:
        ValueError: n_qubits < 1, p_layers < 1, 或参数形状不匹配时
    """
    if n_qubits < 1:
        raise ValueError(f"QAOA 电路至少需要 1 个比特，收到 n_qubits={n_qubits}")
    if p_layers < 1:
        raise ValueError(f"QAOA 层数 p 至少为 1，收到 p_layers={p_layers}")

    if gamma is None:
        gamma = np.zeros(p_layers, dtype=np.float64)
    elif gamma.shape != (p_layers,):
        raise ValueError(f"gamma 形状应为 ({p_layers},)，收到 {gamma.shape}")

    if beta is None:
        beta = np.zeros(p_layers, dtype=np.float64)
    elif beta.shape != (p_layers,):
        raise ValueError(f"beta 形状应为 ({p_layers},)，收到 {beta.shape}")

    instructions: list[str] = []

    for q in range(n_qubits):
        instructions.append(f"H Q{q}")

    for p in range(p_layers):
        g = float(gamma[p])
        b = float(beta[p])

        for q in range(n_qubits):
            instructions.append(f"RZ Q{q} {2 * g:.10f}")

        if two_qubit_gates and n_qubits >= 2:
            for q in range(n_qubits - 1):
                instructions.append(f"CZ Q{q} Q{q + 1}")

        for q in range(n_qubits):
            instructions.append(f"RX Q{q} {2 * b:.10f}")

    for q in range(n_qubits):
        instructions.append(f"M Q{q}")

    return "\n".join(instructions)


__all__ = [
    "generate_bell_state",
    "generate_ghz_state",
    "generate_qaoa_circuit",
    "generate_vqe_circuit",
]
