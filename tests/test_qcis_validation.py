"""QCIS 电路内容验证单元测试（Issue #515）。

覆盖 _validate_qcis 函数的合法/非法输入：
- 长度上限验证
- 门数量上限验证
- 比特数上限验证
- 非法指令检测
- 合法电路通过验证
"""

import pytest

from src.api.tianyan_cqlib import (
    MAX_GATE_COUNT,
    MAX_QCIS_LENGTH,
    MAX_QUBITS_REFERENCED,
    _validate_qcis,
)


class TestValidateQcisValid:
    """合法 QCIS 电路应通过验证。"""

    def test_simple_h_gate(self) -> None:
        _validate_qcis("H Q0\nM Q0")

    def test_multi_qubit_circuit(self) -> None:
        _validate_qcis("H Q0\nCNOT Q0 Q1\nM Q0\nM Q1")

    def test_empty_string_passes(self) -> None:
        _validate_qcis("")

    def test_whitespace_only_passes(self) -> None:
        _validate_qcis("   \n  \n  ")

    def test_case_insensitive_instruction(self) -> None:
        _validate_qcis("h Q0\nm Q0")

    def test_comma_separated_qubits(self) -> None:
        _validate_qcis("CNOT Q0, Q1\nM Q0\nM Q1")


class TestValidateQcisLength:
    """QCIS 长度上限验证。"""

    def test_exceeds_max_length_raises(self) -> None:
        long_circuit = "H Q0\n" * (MAX_QCIS_LENGTH // 5 + 1)
        with pytest.raises(ValueError, match="最大长度"):
            _validate_qcis(long_circuit)

    def test_at_max_length_passes(self) -> None:
        # 用单行长指令接近长度上限，避免触发门数限制
        prefix = "RZ Q0 "
        suffix = "\nM Q0"
        circuit = prefix + "0" * (MAX_QCIS_LENGTH - len(prefix) - len(suffix)) + suffix
        _validate_qcis(circuit)


class TestValidateQcisGateCount:
    """门数量上限验证。"""

    def test_exceeds_max_gates_raises(self) -> None:
        circuit = "\n".join(["H Q0"] * (MAX_GATE_COUNT + 1))
        with pytest.raises(ValueError, match="门数量超过上限"):
            _validate_qcis(circuit)

    def test_at_max_gates_passes(self) -> None:
        circuit = "\n".join(["H Q0"] * MAX_GATE_COUNT)
        _validate_qcis(circuit)


class TestValidateQcisQubitCount:
    """引用比特数上限验证。"""

    def test_exceeds_max_qubits_raises(self) -> None:
        lines = [f"H Q{i}" for i in range(MAX_QUBITS_REFERENCED + 1)]
        with pytest.raises(ValueError, match=r"比特数.*超过上限"):
            _validate_qcis("\n".join(lines))

    def test_at_max_qubits_passes(self) -> None:
        lines = [f"H Q{i}" for i in range(MAX_QUBITS_REFERENCED)]
        _validate_qcis("\n".join(lines))


class TestValidateQcisInvalidInstruction:
    """非法指令检测。"""

    def test_unknown_instruction_raises(self) -> None:
        with pytest.raises(ValueError, match="非法指令"):
            _validate_qcis("INVALID_GATE Q0\nM Q0")

    def test_empty_instruction_raises(self) -> None:
        with pytest.raises(ValueError, match="非法指令"):
            _validate_qcis("H Q0\nUNKNOWN Q1\nM Q0")

    def test_valid_instructions_all_pass(self) -> None:
        valid_ops = ["H", "X", "Y", "Z", "S", "T", "I"]
        for op in valid_ops:
            _validate_qcis(f"{op} Q0\nM Q0")

    def test_rotation_gates_pass(self) -> None:
        _validate_qcis("RX Q0 3.14\nRY Q1 1.57\nRZ Q2 0.5\nM Q0\nM Q1\nM Q2")

    def test_two_qubit_gates_pass(self) -> None:
        _validate_qcis("CZ Q0 Q1\nCNOT Q0 Q1\nISWAP Q0 Q1\nM Q0\nM Q1")

    def test_barrier_passes(self) -> None:
        _validate_qcis("H Q0\nB\nM Q0")
