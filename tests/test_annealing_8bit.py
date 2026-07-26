"""Issue #239：8-bit 权重编码与 QUBO 内存边界。"""

import numpy as np
import pytest

from src.quantum.annealing import QuantumAnnealingOptimizer


def test_8bit_encoding_roundtrip() -> None:
    optimizer = QuantumAnnealingOptimizer(n_bits_per_weight=8)
    deltas = [np.array([-0.097, -0.05, 0.0, 0.025, 0.079], dtype=np.float64)]

    bitstring = optimizer.weight_deltas_to_bitstring(deltas)
    decoded = optimizer.bitstring_to_weights(bitstring, [(5,)])[0]

    assert len(bitstring) == 5 * 8
    assert np.allclose(decoded, deltas[0], atol=0.1 / 128 / 2)


def test_8bit_qubo_for_260_parameters_stays_below_64_mib() -> None:
    optimizer = QuantumAnnealingOptimizer(
        n_bits_per_weight=8,
        max_qubo_memory_mb=64,
    )

    qubo = optimizer.network_to_qubo([np.zeros(260, dtype=np.float64)])

    assert qubo.shape == (2080, 2080)
    assert qubo.nbytes == 2080**2 * 8
    assert optimizer.last_qubo_memory_bytes == qubo.nbytes
    assert qubo.nbytes < 64 * 1024 * 1024


def test_qubo_memory_limit_prevents_oversized_allocation() -> None:
    optimizer = QuantumAnnealingOptimizer(
        n_bits_per_weight=8,
        max_qubo_memory_mb=1,
    )

    with pytest.raises(MemoryError, match="超过配置上限"):
        optimizer.network_to_qubo([np.zeros(260, dtype=np.float64)])
