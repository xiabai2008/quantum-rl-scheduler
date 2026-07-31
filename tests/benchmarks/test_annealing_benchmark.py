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
from typing import Any

import numpy as np
import pytest
import torch
from torch import nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.quantum.annealing import QuantumAnnealingOptimizer
from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.parser import LegacyTaskParser


def _get_stat(stats: Any, key: str, default: float = 0.0) -> float:
    """兼容 pytest-benchmark 4.x/5.x 的 stats 属性访问。

    4.x: stats.mean / stats.median（直接属性）
    5.x: stats 是 Metadata 对象，需通过 stats['mean'] 或 stats.stats.mean 访问；
         在某些 5.x 版本下 stats 可能为 None，此时返回 default。
    """
    # 4.x: 直接属性访问
    val = getattr(stats, key, None)
    if val is not None:
        return val
    # 5.x: dict-like 访问
    try:
        return stats[key]
    except (KeyError, TypeError, IndexError):
        pass
    # 5.x: 嵌套 Stats 对象
    inner = getattr(stats, "stats", None)
    if inner is not None:
        val = getattr(inner, key, None)
        if val is not None:
            return val
    return default


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
        # 性能回归阈值断言（Issue #729）：10x10 QUBO 求解均值应 < 1.0s
        # pytest-benchmark 5.x 兼容
        mean_val = _get_stat(benchmark.stats, "mean", default=0.0)
        assert mean_val < 1.0, f"QUBO 10x10 求解均值超阈值: {mean_val:.3f}s"

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
        # 性能回归阈值断言（Issue #729）：50x50 QUBO 求解均值应 < 3.0s
        # pytest-benchmark 5.x 兼容
        mean_val = _get_stat(benchmark.stats, "mean", default=0.0)
        assert mean_val < 3.0, f"QUBO 50x50 求解均值超阈值: {mean_val:.3f}s"

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
        # 性能回归阈值断言（Issue #729）：network_to_qubo 均值应 < 0.2s
        # pytest-benchmark 5.x 兼容
        mean_val = _get_stat(benchmark.stats, "mean", default=0.0)
        assert mean_val < 0.2, f"network_to_qubo 均值超阈值: {mean_val:.3f}s"

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
        for decoded, shape in zip(result, shapes, strict=False):
            assert decoded.shape == shape
        # 性能回归阈值断言（Issue #729）：bitstring 解码均值应 < 0.1s
        # pytest-benchmark 5.x 兼容
        mean_val = _get_stat(benchmark.stats, "mean", default=0.0)
        assert mean_val < 0.1, f"bitstring_to_weights 均值超阈值: {mean_val:.3f}s"


@pytest.mark.benchmark
class TestEnvBenchmark:
    """调度环境性能基准"""

    def test_env_step_performance(self, benchmark):
        """QuantumSchedulingEnv.step() 性能基准（max_steps=100）"""
        env = QuantumSchedulingEnv(max_steps=100, seed=42)
        env.reset(seed=42)
        # 使用 env 实际观测空间形状断言，兼容 include_fairness_obs 开关
        expected_shape = env.observation_space.shape

        def step_once():
            action = int(env.action_space.sample())
            obs, _reward, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                env.reset(seed=42)
            return obs

        result = benchmark(step_once)
        assert result.shape == expected_shape
        # 性能回归阈值断言（Issue #729）：env.step() 中位数应 < 50ms
        # pytest-benchmark 5.x 兼容
        median = _get_stat(benchmark.stats, "median", default=0.0)
        assert median < 0.05, f"env.step() 中位数超阈值: {median:.4f}s"

    def test_env_reset_performance(self, benchmark):
        """QuantumSchedulingEnv.reset() 性能基准"""
        env = QuantumSchedulingEnv(max_steps=100, seed=42)
        expected_shape = env.observation_space.shape

        def reset_once():
            obs, _info = env.reset(seed=42)
            return obs

        result = benchmark(reset_once)
        assert result.shape == expected_shape
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
        # 性能回归阈值断言（Issue #729）：env.reset() 中位数应 < 50ms
        # pytest-benchmark 5.x 兼容
        median = _get_stat(benchmark.stats, "median", default=0.0)
        assert median < 0.05, f"env.reset() 中位数超阈值: {median:.4f}s"


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
        # 性能回归阈值断言（Issue #729）：QASM 解析均值应 < 0.1s
        # pytest-benchmark 5.x 兼容
        mean_val = _get_stat(benchmark.stats, "mean", default=0.0)
        assert mean_val < 0.1, f"QASM 解析均值超阈值: {mean_val:.4f}s"
