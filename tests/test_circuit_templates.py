"""circuit_templates 单元测试 — 量子电路模板正确性验证。

覆盖 Issue #574 的四个模板生成函数：
    - generate_bell_state : Bell 态电路（验证修复后的电路结构）
    - generate_ghz_state  : GHZ-n 态电路（H-CZ-H 等效 CNOT）
    - generate_vqe_circuit: VQE 变分电路
    - generate_qaoa_circuit: QAOA 电路
"""

import numpy as np
import pytest

from src.quantum.circuit_templates import (
    generate_bell_state,
    generate_ghz_state,
    generate_qaoa_circuit,
    generate_vqe_circuit,
)


def test_bell_state_structure():
    """Bell 态电路使用 H-CZ-H 等效 CNOT 生成纠缠态。

    电路结构: H Q0 → H Q1 → CZ Q0 Q1 → H Q1 → M Q0 → M Q1
    QCIS 无原生 CNOT，用 H-CZ-H 分解实现纠缠（Issue #644 修复）。
    """
    circuit = generate_bell_state()
    lines = circuit.split("\n")
    assert "H Q0" in lines
    assert "CZ Q0 Q1" in lines
    assert "M Q0" in lines
    assert "M Q1" in lines
    # H Q1 应在 CZ 前后各出现一次（H-CZ-H 等效 CNOT）
    h1_positions = [i for i, ln in enumerate(lines) if ln == "H Q1"]
    assert len(h1_positions) == 2, f"H Q1 应出现 2 次，实际 {len(h1_positions)} 次"
    cz_pos = lines.index("CZ Q0 Q1")
    assert h1_positions[0] < cz_pos < h1_positions[1], "H Q1 应在 CZ 前后各一次"
    # H Q0 应在 CZ Q0 Q1 之前
    assert lines.index("H Q0") < cz_pos


def test_bell_state_multiple_pairs():
    """测试多对比特生成：每对都应有 H Qc, H Qt, CZ Qc Qt, H Qt（H-CZ-H 等效 CNOT）。"""
    pairs = [(0, 1), (2, 3), (4, 5)]
    circuit = generate_bell_state(pairs)
    lines = circuit.split("\n")
    for qc, qt in pairs:
        assert f"H Q{qc}" in lines
        assert f"CZ Q{qc} Q{qt}" in lines
        # target 上应有 2 个 H（CZ 前后各一个）
        ht_positions = [i for i, ln in enumerate(lines) if ln == f"H Q{qt}"]
        assert len(ht_positions) == 2, f"H Q{qt} 应出现 2 次"
        cz_pos = lines.index(f"CZ Q{qc} Q{qt}")
        assert ht_positions[0] < cz_pos < ht_positions[1]
    # 测量所有涉及的比特
    for q in [0, 1, 2, 3, 4, 5]:
        assert f"M Q{q}" in lines


def test_ghz_state_structure():
    """GHZ 态包含 H Q0，以及每个 CZ 前后各有一个 H 作用在 target 上（H-CZ-H 等效 CNOT）。"""
    circuit = generate_ghz_state(3)
    lines = circuit.split("\n")
    assert "H Q0" in lines
    # 每个 CZ Q{i} Q{i+1} 前后都应有 H Q{i+1}
    for i in range(2):  # n=3 → i=0, 1
        cz = f"CZ Q{i} Q{i + 1}"
        h_target = f"H Q{i + 1}"
        assert cz in lines
        assert h_target in lines
        cz_pos = lines.index(cz)
        h_positions = [idx for idx, ln in enumerate(lines) if ln == h_target]
        assert any(p < cz_pos for p in h_positions), f"{h_target} 应出现在 {cz} 之前"
        assert any(p > cz_pos for p in h_positions), f"{h_target} 应出现在 {cz} 之后"
    # 测量所有比特
    for q in [0, 1, 2]:
        assert f"M Q{q}" in lines


def test_ghz_state_invalid():
    """n_qubits < 2 抛出 ValueError。"""
    with pytest.raises(ValueError):
        generate_ghz_state(1)
    with pytest.raises(ValueError):
        generate_ghz_state(0)


def test_vqe_circuit_structure():
    """VQE 电路包含 RY, RZ, CZ 门。"""
    circuit = generate_vqe_circuit(n_qubits=4, depth=1)
    lines = circuit.split("\n")
    assert any(ln.startswith("RY Q") for ln in lines)
    assert any(ln.startswith("RZ Q") for ln in lines)
    assert any(ln.startswith("CZ Q") for ln in lines)
    assert any(ln.startswith("M Q") for ln in lines)


def test_vqe_circuit_params_shape():
    """params 形状不匹配时抛出 ValueError；正确形状正常工作。"""
    with pytest.raises(ValueError):
        generate_vqe_circuit(n_qubits=4, depth=2, params=np.zeros((1, 4, 2)))
    with pytest.raises(ValueError):
        generate_vqe_circuit(n_qubits=4, depth=2, params=np.zeros((2, 3, 2)))
    # 正确形状应正常工作
    circuit = generate_vqe_circuit(n_qubits=4, depth=2, params=np.zeros((2, 4, 2)))
    assert "RY Q0" in circuit


def test_qaoa_circuit_structure():
    """QAOA 电路包含 H, RZ, CZ, RX 门。"""
    circuit = generate_qaoa_circuit(n_qubits=5, p_layers=1)
    lines = circuit.split("\n")
    assert any(ln.startswith("H Q") for ln in lines)
    assert any(ln.startswith("RZ Q") for ln in lines)
    assert any(ln.startswith("CZ Q") for ln in lines)
    assert any(ln.startswith("RX Q") for ln in lines)
    assert any(ln.startswith("M Q") for ln in lines)


def test_qaoa_circuit_params():
    """gamma/beta 参数形状不匹配时抛出 ValueError；正确形状正常工作。"""
    with pytest.raises(ValueError):
        generate_qaoa_circuit(n_qubits=5, p_layers=2, gamma=np.zeros(1))
    with pytest.raises(ValueError):
        generate_qaoa_circuit(n_qubits=5, p_layers=2, beta=np.zeros(3))
    # 正确形状应正常工作
    circuit = generate_qaoa_circuit(n_qubits=5, p_layers=2, gamma=np.zeros(2), beta=np.zeros(2))
    assert "H Q0" in circuit


def test_all_templates_return_qcis():
    """所有函数返回字符串（QCIS 格式）。"""
    assert isinstance(generate_bell_state(), str)
    assert isinstance(generate_ghz_state(3), str)
    assert isinstance(generate_vqe_circuit(), str)
    assert isinstance(generate_qaoa_circuit(), str)


def test_imports():
    """验证从 src.quantum.circuit_templates 可以导入4个函数。"""
    import src.quantum.circuit_templates as ct

    assert callable(ct.generate_bell_state)
    assert callable(ct.generate_ghz_state)
    assert callable(ct.generate_vqe_circuit)
    assert callable(ct.generate_qaoa_circuit)
