"""
量子RL调度系统 - 主包初始化
Quantum RL-driven Scheduling System for Tianyan Cloud Platform
"""

__version__ = "0.1.0"
__author__ = "揭榜挂帅参赛团队"
__project_name__ = "量子RL驱动的天衍云平台智能调度系统"

# 导入主要模块
from src.scheduler import SchedulerAgent, SchedulingEnv
from src.api import TianyanClient
from src.quantum import QuantumAnnealingAccelerator

__all__ = [
    "SchedulerAgent",
    "SchedulingEnv", 
    "TianyanClient",
    "QuantumAnnealingAccelerator",
]
