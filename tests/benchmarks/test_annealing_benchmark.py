"""性能基准测试 — 防止关键路径性能回归

覆盖关键性能路径：
    - 量子退火 QUBO 求解（10x10 / 50x50）
    - network_to_qubo 矩阵构造（小网络 nn.Linear(8,4)）
    - 调度环境 step / reset
    - LegacyTaskParser QASM 解析
    - bitstring_to_weights 解码
"""

import os
import sys

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.quantum.annealing import QuantumAnnealingOptimizer
from src.scheduler.env import OBS_DIM, QuantumSchedulingEnv
from src.scheduler.parser import LegacyTaskParser


def generate_qasm(num_qubits: int, num_gates: int) -> str:
    """生成合法的 QASM 量子电路描述字符串（用于基准与属性测试）。

    Args:
        num_qubits: 量子比特数（qreg 大小）
        num_gates:  单比特门数量（在 h/x/y/z 间循环）

    Returns:
        合法的 QASM 2.0 字符串
    """
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{num_qubits}];",
        f"creg c[{num_qubits}];",
    ]
    gates = ("h", "x", "y", "z")
    for i in range(num_gates):
        gate = gates[i % len(gates)]
        q = i % max(num_qubits, 1)
        lines.append(f"{gate} q[{q}];")
    if num_qubits > 0:
        lines.append("measure q[0] -> c[0];")
    return "\n".join(lines)


@pytest.mark.benchmark
class TestAnnealingBenchmark:
    """量子退火性能基准"""

    def test_qubo_solve_small(self, benchmark):
        """QUBO 求解基准：10x10 矩阵 < 1 秒"""
        opt = QuantumAnnealingOptimizer(
            num_qubits=16, shots=50, annealing_time=5, simulation_mode=True
        )
        opt._sim_num_sweeps = 200  # 降低 numpy 仿真扫描次数以保持基准快速
        rng = np.random.default_rng(42)
        qubo = rng.standard_normal((10, 10))
        qubo = (qubo + qubo.T) / 2  # 对称化

        def solve():
            return opt.anneal(qubo)

        result = benchmark(solve)
        assert len(result) == 10
        assert all(ch in "01" for ch in result)

    def test_qubo_solve_medium(self, benchmark):
        """QUBO 求解基准：50x50 矩阵 < 3 秒"""
        opt = QuantumAnnealingOptimizer(
            num_qubits=16, shots=50, annealing_time=5, simulation_mode=True
        )
        opt._sim_num_sweeps = 200
        rng = np.random.default_rng(42)
        qubo = rng.standard_normal((50, 50))
        qubo = (qubo + qubo.T) / 2

        def solve():
            return opt.anneal(qubo)

        result = benchmark(solve)
        assert len(result) == 50
        assert all(ch in "01" for ch in result)

    def test_network_to_qubo(self, benchmark):
        """network_to_qubo 性能基准（小网络 nn.Linear(8,4)）"""
        opt = QuantumAnnealingOptimizer(num_qubits=16, shots=10, simulation_mode=True)
        layer = nn.Linear(8, 4)
        weights = [layer.weight.detach().numpy(), layer.bias.detach().numpy()]

        def to_qubo():
            return opt.network_to_qubo(weights)

        qubo = benchmark(to_qubo)
        n_bits_per_weight = 16 // 4  # 4
        total_params = 8 * 4 + 4  # 36
        expected = total_params * n_bits_per_weight
        assert qubo.shape == (expected, expected)

    def test_bitstring_decode(self, benchmark):
        """bitstring_to_weights 解码性能基准"""
        opt = QuantumAnnealingOptimizer(num_qubits=16, shots=10, simulation_mode=True)
        rng = np.random.default_rng(42)
        w1 = rng.standard_normal((8, 4)).astype(np.float64)
        w2 = rng.standard_normal(4).astype(np.float64)
        weights = [w1, w2]
        shapes = [w1.shape, w2.shape]
        n_bits_per_weight = 16 // 4  # 4
        total_params = int(np.prod(w1.shape)) + int(np.prod(w2.shape))
        bits = rng.integers(0, 2, total_params * n_bits_per_weight)
        bitstring = "".join(str(int(b)) for b in bits)

        def decode():
            return opt.bitstring_to_weights(bitstring, shapes, current_weights=weights)

        result = benchmark(decode)
        assert len(result) == len(shapes)
        for decoded, shape in zip(result, shapes):
            assert decoded.shape == shape


@pytest.mark.benchmark
class TestEnvBenchmark:
    """调度环境性能基准"""

    def test_env_step_performance(self, benchmark):
        """QuantumSchedulingEnv.step() 性能基准（max_steps=100）"""
        env = QuantumSchedulingEnv(max_steps=100, seed=42)
        env.reset(seed=42)

        def step_once():
            action = int(env.action_space.sample())
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                env.reset(seed=42)
            return obs

        result = benchmark(step_once)
        assert result.shape == (OBS_DIM,)

    def test_env_reset_performance(self, benchmark):
        """QuantumSchedulingEnv.reset() 性能基准"""
        env = QuantumSchedulingEnv(max_steps=100, seed=42)

        def reset_once():
            obs, info = env.reset(seed=42)
            return obs

        result = benchmark(reset_once)
        assert result.shape == (OBS_DIM,)
        assert np.all(result >= 0.0) and np.all(result <= 1.0)


@pytest.mark.benchmark
class TestParserBenchmark:
    """任务解析器性能基准"""

    def test_parser_performance(self, benchmark):
        """LegacyTaskParser QASM 解析性能基准"""
        parser = LegacyTaskParser()
        qasm = generate_qasm(10, 20)

        def parse_once():
            return parser.parse(qasm, format="qasm")

        result = benchmark(parse_once)
        assert result is not None
        assert result.qubit_count == 10
        assert result.gate_count == 20
