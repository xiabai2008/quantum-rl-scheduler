#!/usr/bin/env python
"""NoiseModelExtractor 噪声模型提取测试（Issue #579）。

测试噪声表征电路生成和噪声参数提取：
    - generate_readout_calibration_circuit / generate_rb_circuit / generate_t1_delay_circuit
    - NoiseModelExtractor.extract_readout_error / extract_gate_error / extract_decoherence
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import math

import pytest

from src.scheduler.env_real_machine import (
    NoiseModelExtractor,
    generate_rb_circuit,
    generate_readout_calibration_circuit,
    generate_t1_delay_circuit,
)

# =============================================================================
# 噪声表征电路生成测试
# =============================================================================


class TestReadoutCalibrationCircuit:
    """读出校准电路生成测试。"""

    def test_single_qubit(self):
        """单比特校准电路包含 H 门和测量。"""
        qcis = generate_readout_calibration_circuit(num_qubits=1)
        lines = qcis.strip().split("\n")
        assert "H Q0" in lines
        assert lines[-1] == "M Q0"

    def test_multi_qubit(self):
        """多比特校准电路为每个比特生成 H 门。"""
        qcis = generate_readout_calibration_circuit(num_qubits=3)
        lines = qcis.strip().split("\n")
        assert "H Q0" in lines
        assert "H Q1" in lines
        assert "H Q2" in lines
        assert "M Q0 Q1 Q2" in lines

    def test_invalid_qubits_raises(self):
        """num_qubits < 1 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="num_qubits"):
            generate_readout_calibration_circuit(num_qubits=0)


class TestRbCircuit:
    """随机化基准电路生成测试。"""

    def test_basic_generation(self):
        """RB 电路包含随机门和测量。"""
        qcis = generate_rb_circuit(num_cliffords=5, num_qubits=1, seed=42)
        lines = qcis.strip().split("\n")
        # 5 个随机门 + 1 个逆操作 + 1 个测量 = 7 行
        assert len(lines) == 7
        assert lines[-1].startswith("M")

    def test_deterministic_with_seed(self):
        """相同 seed 生成相同电路。"""
        q1 = generate_rb_circuit(num_cliffords=10, seed=42)
        q2 = generate_rb_circuit(num_cliffords=10, seed=42)
        assert q1 == q2

    def test_different_seed_different_circuit(self):
        """不同 seed 生成不同电路。"""
        q1 = generate_rb_circuit(num_cliffords=20, seed=42)
        q2 = generate_rb_circuit(num_cliffords=20, seed=123)
        assert q1 != q2

    def test_multi_qubit_rb(self):
        """多比特 RB 电路包含所有比特的门。"""
        qcis = generate_rb_circuit(num_cliffords=3, num_qubits=2, seed=42)
        lines = qcis.strip().split("\n")
        # 3 * 2 = 6 随机门 + 2 逆操作 + 1 测量 = 9 行
        assert len(lines) == 9
        assert "Q1" in qcis

    def test_invalid_cliffords_raises(self):
        """num_cliffords < 1 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="num_cliffords"):
            generate_rb_circuit(num_cliffords=0)

    def test_invalid_qubits_raises(self):
        """num_qubits < 1 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="num_qubits"):
            generate_rb_circuit(num_qubits=0)


class TestT1DelayCircuit:
    """T1 延迟电路生成测试。"""

    def test_basic_generation(self):
        """T1 电路包含 X 门、延迟（Z 门）和测量。"""
        qcis = generate_t1_delay_circuit(delay_steps=5, qubit=0)
        lines = qcis.strip().split("\n")
        assert lines[0] == "X Q0"
        assert lines[-1] == "M Q0"
        # 1 X + 5 Z + 1 M = 7 行
        assert len(lines) == 7

    def test_zero_delay(self):
        """delay_steps=0 时只有 X 门和测量。"""
        qcis = generate_t1_delay_circuit(delay_steps=0, qubit=0)
        lines = qcis.strip().split("\n")
        assert len(lines) == 2
        assert lines[0] == "X Q0"
        assert lines[1] == "M Q0"

    def test_custom_qubit(self):
        """自定义比特索引。"""
        qcis = generate_t1_delay_circuit(delay_steps=2, qubit=3)
        assert "X Q3" in qcis
        assert "M Q3" in qcis

    def test_invalid_delay_raises(self):
        """delay_steps < 0 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="delay_steps"):
            generate_t1_delay_circuit(delay_steps=-1)

    def test_invalid_qubit_raises(self):
        """qubit < 0 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="qubit"):
            generate_t1_delay_circuit(qubit=-1)


# =============================================================================
# NoiseModelExtractor 测试
# =============================================================================


class TestExtractReadoutError:
    """读出误差提取测试。"""

    def test_perfect_readout(self):
        """完美读出（50/50）误差为 0。"""
        error = NoiseModelExtractor.extract_readout_error({"0": 0.5, "1": 0.5})
        assert error == 0.0

    def test_biased_readout(self):
        """偏置读出产生非零误差。"""
        error = NoiseModelExtractor.extract_readout_error({"0": 0.6, "1": 0.4})
        assert error == pytest.approx(0.2)

    def test_extreme_bias(self):
        """极端偏置（全 0）误差为 1。"""
        error = NoiseModelExtractor.extract_readout_error({"0": 1.0, "1": 0.0})
        assert error == 1.0

    def test_missing_key_defaults_to_half(self):
        """缺少 "0" 键时默认 0.5（误差 0）。"""
        error = NoiseModelExtractor.extract_readout_error({"1": 1.0})
        assert error == 0.0


class TestExtractGateError:
    """门误差提取测试。"""

    def test_insufficient_data(self):
        """数据不足时返回 0。"""
        assert NoiseModelExtractor.extract_gate_error([]) == 0.0
        assert NoiseModelExtractor.extract_gate_error([{"m": 1, "fidelity": 0.99}]) == 0.0

    def test_perfect_gates(self):
        """完美门（保真度恒为 1.0）应返回接近 0 的误差。"""
        rb_results = [
            {"m": 1, "fidelity": 1.0},
            {"m": 5, "fidelity": 1.0},
            {"m": 10, "fidelity": 1.0},
        ]
        # fidelity=1.0, diff=0.5 > 0, log(0.5) = -0.693...
        # 但所有 diff 相同 → slope=0 → alpha=1 → error=0
        error = NoiseModelExtractor.extract_gate_error(rb_results)
        assert error == pytest.approx(0.0, abs=1e-10)

    def test_decay_pattern(self):
        """衰减模式应返回正的误差率。"""
        # 模拟指数衰减：F(m) = 0.5 * alpha^(2m) + 0.5
        alpha = 0.95
        rb_results = [
            {"m": m, "fidelity": 0.5 * alpha ** (2 * m) + 0.5}
            for m in [1, 2, 5, 10, 20]
        ]
        error = NoiseModelExtractor.extract_gate_error(rb_results)
        # 期望误差 = (1 - alpha) / 2 = 0.025
        assert error > 0
        assert error == pytest.approx(0.025, abs=0.01)

    def test_all_below_asymptote(self):
        """所有保真度低于渐近值时返回 0。"""
        rb_results = [
            {"m": 1, "fidelity": 0.3},
            {"m": 5, "fidelity": 0.2},
        ]
        error = NoiseModelExtractor.extract_gate_error(rb_results)
        assert error == 0.0


class TestExtractDecoherence:
    """T1 弛豫时间提取测试。"""

    def test_insufficient_data(self):
        """数据不足时返回 t1=-1。"""
        result = NoiseModelExtractor.extract_decoherence([])
        assert result["t1"] == -1.0

        result = NoiseModelExtractor.extract_decoherence([{"t": 10, "p1": 0.9}])
        assert result["t1"] == -1.0

    def test_exponential_decay_fit(self):
        """指数衰减数据应正确拟合 T1。"""
        # P(1) = exp(-t / T1), T1 = 50
        t1_true = 50.0
        delay_results = [
            {"t": t, "p1": math.exp(-t / t1_true)}
            for t in [10, 20, 30, 50, 80, 100]
        ]
        result = NoiseModelExtractor.extract_decoherence(delay_results)
        assert result["t1"] > 0
        assert result["t1"] == pytest.approx(t1_true, rel=0.1)
        assert result["fit_quality"] > 0.95

    def test_amplitude_recovered(self):
        """振幅 A 应接近 1.0（当 P(1)=exp(-t/T1) 时）。"""
        t1_true = 100.0
        delay_results = [
            {"t": t, "p1": math.exp(-t / t1_true)}
            for t in [10, 30, 50, 80, 100]
        ]
        result = NoiseModelExtractor.extract_decoherence(delay_results)
        assert result["amplitude"] == pytest.approx(1.0, rel=0.1)

    def test_non_decay_returns_failure(self):
        """非衰减数据（递增）应返回 t1=-1。"""
        delay_results = [
            {"t": 10, "p1": 0.1},
            {"t": 50, "p1": 0.5},
            {"t": 100, "p1": 0.9},
        ]
        result = NoiseModelExtractor.extract_decoherence(delay_results)
        assert result["t1"] == -1.0

    def test_result_keys(self):
        """返回字典包含所有预期键。"""
        delay_results = [
            {"t": 10, "p1": 0.9},
            {"t": 50, "p1": 0.6},
            {"t": 100, "p1": 0.3},
        ]
        result = NoiseModelExtractor.extract_decoherence(delay_results)
        assert "t1" in result
        assert "amplitude" in result
        assert "offset" in result
        assert "fit_quality" in result


class TestExtractAll:
    """批量提取测试。"""

    def test_all_parameters(self):
        """同时提供所有数据时应提取全部参数。"""
        extractor = NoiseModelExtractor()
        result = extractor.extract_all(
            measurement_results={"0": 0.45, "1": 0.55},
            rb_results=[
                {"m": 1, "fidelity": 0.99},
                {"m": 5, "fidelity": 0.95},
                {"m": 10, "fidelity": 0.90},
            ],
            delay_results=[
                {"t": 10, "p1": 0.9},
                {"t": 50, "p1": 0.6},
                {"t": 100, "p1": 0.3},
            ],
        )
        assert "readout_error" in result
        assert "gate_error" in result
        assert "decoherence" in result
        assert result["readout_error"] > 0

    def test_partial_data(self):
        """仅提供部分数据时只返回对应参数。"""
        extractor = NoiseModelExtractor()
        result = extractor.extract_all(measurement_results={"0": 0.4, "1": 0.6})
        assert "readout_error" in result
        assert "gate_error" not in result
        assert "decoherence" not in result

    def test_no_data(self):
        """不提供任何数据时返回空字典。"""
        extractor = NoiseModelExtractor()
        result = extractor.extract_all()
        assert result == {}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
