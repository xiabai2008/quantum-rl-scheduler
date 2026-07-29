"""
Web可视化模块初始化
Web Visualization Module Initialization

调度系统监控界面
"""

from src.visualization.app import app, start_web_server

__all__ = [
    "app",
    "start_web_server",
]
