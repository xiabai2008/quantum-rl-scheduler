"""
真机交互模块（Issue #579）
Real Machine Interaction Module

提供真机噪声参数提取、校准数据获取、真机实验辅助等功能。

模块导出：
    - ``NoiseModelExtractor``：从真机提取读出错误、门错误、T1 退相干参数
"""

from src.real_machine.noise_extractor import NoiseModelExtractor

__all__ = [
    "NoiseModelExtractor",
]
