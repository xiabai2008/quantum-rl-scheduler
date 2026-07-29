"""
量子计算模块
Quantum Computing Module

包含：
- QuantumAnnealingOptimizer: 量子启发式退火加速 RL 策略搜索（QUBO 映射 + 模拟退火）
  .. deprecated:: 2026-07-27  探索性功能，默认关闭，不再投入开发
- QuantumAnnealingAccelerator: 量子退火加速器（旧版兼容别名）
- 电路模板生成器（Issue #574）：
  - generate_bell_state: Bell 态电路（2比特纠缠）
  - generate_ghz_state: GHZ-n 态电路（多比特纠缠）
  - generate_vqe_circuit: VQE 变分电路模板（默认4比特）
  - generate_qaoa_circuit: QAOA 电路模板（默认5比特）
"""

from src.quantum.annealing import (
    QUANTUM_ACCELERATION_ENABLED,
    QuantumAnnealingOptimizer,
)
from src.quantum.circuit_templates import (
    generate_bell_state,
    generate_ghz_state,
    generate_qaoa_circuit,
    generate_vqe_circuit,
)

# 旧版兼容别名
QuantumAnnealingAccelerator = QuantumAnnealingOptimizer

__all__ = [
    "QUANTUM_ACCELERATION_ENABLED",
    "QuantumAnnealingAccelerator",
    "QuantumAnnealingOptimizer",
    "generate_bell_state",
    "generate_ghz_state",
    "generate_qaoa_circuit",
    "generate_vqe_circuit",
]
