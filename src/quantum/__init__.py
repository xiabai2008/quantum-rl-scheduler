"""
量子计算模块
Quantum Computing Module

包含：
- QuantumAnnealingOptimizer: 量子启发式退火加速 RL 策略搜索（QUBO 映射 + 模拟退火）
  .. deprecated:: 2026-07-27  探索性功能，默认关闭，不再投入开发
- QuantumAnnealingAccelerator: 量子退火加速器（旧版兼容别名）
"""

from src.quantum.annealing import (
    QUANTUM_ACCELERATION_ENABLED,
    QuantumAnnealingOptimizer,
)

# 旧版兼容别名
QuantumAnnealingAccelerator = QuantumAnnealingOptimizer

__all__ = [
    "QUANTUM_ACCELERATION_ENABLED",
    "QuantumAnnealingAccelerator",
    "QuantumAnnealingOptimizer",
]
