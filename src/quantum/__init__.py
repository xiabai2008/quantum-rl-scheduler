"""
量子计算模块
Quantum Computing Module

包含：
- QuantumAnnealingOptimizer: 量子退火加速 RL 策略搜索（QUBO 映射 + 退火求解）
- QuantumAnnealingAccelerator: 量子退火加速器（旧版兼容）
- QuantumCircuitGenerator: 量子电路生成器
- FidelityEstimator: 量子保真度估计器
"""

from src.quantum.annealing import QUANTUM_ACCELERATION_ENABLED, QuantumAnnealingOptimizer

# circuit / fidelity 模块尚在开发中，按需导入
try:
    from src.quantum.circuit import QuantumCircuitGenerator
except ImportError:
    QuantumCircuitGenerator = None

try:
    from src.quantum.fidelity import FidelityEstimator
except ImportError:
    FidelityEstimator = None

# 旧版兼容别名
QuantumAnnealingAccelerator = QuantumAnnealingOptimizer

__all__ = [
    "QUANTUM_ACCELERATION_ENABLED",
    "FidelityEstimator",
    "QuantumAnnealingAccelerator",
    "QuantumAnnealingOptimizer",
    "QuantumCircuitGenerator",
]
