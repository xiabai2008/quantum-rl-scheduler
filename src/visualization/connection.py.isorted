"""
WebSocket 连接管理器

管理所有 WebSocket 客户端连接，提供连接接受、断开、广播能力。
广播失败的单个连接会被自动移除，不影响其他连接。
"""

from fastapi import WebSocket
from loguru import logger


class ConnectionManager:
    """管理所有 WebSocket 客户端连接"""

    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        """接受 WebSocket 连接并加入活跃连接列表"""
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        """断开 WebSocket 连接"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        """向所有连接的客户端广播消息"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                # 防御性错误边界：单个连接发送失败不应中断广播，任何异常均移除该连接
                logger.debug(f"[Web] WebSocket 广播失败，将移除该连接: {e}")
                disconnected.append(connection)
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)
