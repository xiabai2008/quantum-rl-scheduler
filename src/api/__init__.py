"""
src.api 包初始化模块
支持三种后端：Mock / REST / cqlib（真机优先）
"""

import os
from typing import Any

from src.api.mock_client import MockTianyanClient, create_tianyan_client
from src.api.tianyan_client import CircuitState, TianyanAPIError, TianyanClient
from src.api.tianyan_cqlib import (
    CqlibTianyanClient,
    MultiMachineCqlibCoordinator,
    create_multi_machine_clients,
)

__all__ = [
    "CircuitState",
    "CqlibTianyanClient",
    "MockTianyanClient",
    "MultiMachineCqlibCoordinator",
    "TianyanAPIError",
    "TianyanClient",
    "create_multi_machine_clients",
    "create_tianyan_client",
    "get_client",
    "get_cqlib_client",
]


def get_cqlib_client(machine_name: str = "tianyan_s") -> CqlibTianyanClient:
    """获取 cqlib 真机客户端

    从环境变量 TIANYAN_API_KEY 读取密钥，直接连接天衍云超导真机。

    Args:
        machine_name: 量子计算机名称（默认 tianyan_s）

    Returns:
        CqlibTianyanClient 实例
    """
    api_key = os.getenv("TIANYAN_API_KEY", "")
    if not api_key:
        raise ValueError("未设置 TIANYAN_API_KEY 环境变量")
    return CqlibTianyanClient(login_key=api_key, machine_name=machine_name)


def get_client(mock_mode: bool | None = None) -> Any:
    """获取天衍云客户端（自动选择真实或 Mock 模式）

    优先读取顺序：
    1. 显式传参 mock_mode
    2. 环境变量 TIANYAN_MOCK_MODE
    3. 默认使用 Mock 模式

    Args:
        mock_mode: 是否使用 Mock 模式（None 表示自动检测）

    Returns:
        客户端实例
    """
    return create_tianyan_client(mock_mode=mock_mode)
