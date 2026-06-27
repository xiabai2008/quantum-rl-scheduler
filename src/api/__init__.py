"""
src.api 包初始化模块
自动导出真实/Mock 客户端，根据配置自动切换
"""

from src.api.tianyan_client import TianyanClient, TianyanAPIError
from src.api.mock_client import MockTianyanClient, create_tianyan_client

__all__ = [
    "TianyanClient",
    "TianyanAPIError",
    "MockTianyanClient",
    "create_tianyan_client",
]


def get_client(mock_mode: bool = None):
    """获取天衍云客户端（自动选择真实或 Mock 模式）

    优先读取顺序：
    1. 显式传参 ``mock_mode``
    2. 环境变量 ``TIANYAN_MOCK_MODE``
    3. 配置文件 ``config/config.yaml`` 中的 ``tianyan.mock_mode``

    Args:
        mock_mode: 是否使用 Mock 模式（None 表示自动检测）

    Returns:
        真实客户端或 Mock 客户端实例

    Examples:
        >>> # 自动检测配置（推荐）
        >>> from src.api import get_client
        >>> client = get_client()
        >>>
        >>> # 强制使用 Mock 模式
        >>> client = get_client(mock_mode=True)
    """
    return create_tianyan_client(mock_mode=mock_mode)
