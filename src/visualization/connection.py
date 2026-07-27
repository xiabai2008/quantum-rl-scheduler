"""
WebSocket 连接管理器

管理所有 WebSocket 客户端连接，提供连接接受、断开、广播能力。
广播失败的单个连接会被自动移除，不影响其他连接。

线程安全：使用 threading.RLock 保护 active_connections 的并发访问（Issue #216）。
"""

import threading
from typing import Any

from fastapi import WebSocket
from loguru import logger


class ConnectionManager:
    """管理所有 WebSocket 客户端连接"""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []
        self._lock = threading.RLock()

    async def connect(self, websocket: WebSocket) -> None:
        """接受 WebSocket 连接并加入活跃连接列表（线程安全）。

        防护重复连接：若 websocket 已在列表中，不再重复添加（Issue #216）。
        """
        await websocket.accept()
        with self._lock:
            if websocket not in self.active_connections:
                self.active_connections.append(websocket)
            else:
                logger.debug("[Web] WebSocket 已在活跃连接列表中，跳过重复添加")

    def disconnect(self, websocket: WebSocket) -> None:
        """断开 WebSocket 连接（线程安全）"""
        with self._lock:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    def get_connection_count(self) -> int:
        """获取当前活跃连接数（线程安全）"""
        with self._lock:
            return len(self.active_connections)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """向所有连接的客户端广播消息（线程安全）"""
        with self._lock:
            connections = list(self.active_connections)

        disconnected: list[WebSocket] = []
        for connection in connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                # 防御性错误边界：单个连接发送失败不应中断广播，任何异常均移除该连接
                logger.debug(f"[Web] WebSocket 广播失败，将移除该连接: {e}")
                disconnected.append(connection)

        if disconnected:
            with self._lock:
                for conn in disconnected:
                    if conn in self.active_connections:
                        self.active_connections.remove(conn)
