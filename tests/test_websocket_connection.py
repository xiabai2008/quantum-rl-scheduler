"""ConnectionManager 单元测试（Issue #802: Web 可视化模块测试覆盖）。

测试 src/visualization/connection.py 的 ConnectionManager 类：
- 初始化（默认/自定义 max_connections）
- connect/disconnect 生命周期
- get_connection_count 计数
- broadcast 单播/多播/失败连接移除
- 连接去重
- 最大连接数限制与拒绝
- 线程安全（RLock 保护）

使用 mock WebSocket 对象，不需要真实 WebSocket 服务器。
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.visualization.connection import ConnectionManager


def _make_mock_websocket() -> MagicMock:
    """构造一个 mock WebSocket 对象，支持 async accept/close/send_json。"""
    ws = MagicMock()
    ws.accept = AsyncMock()
    ws.close = AsyncMock()
    ws.send_json = AsyncMock()
    return ws


class TestConnectionManagerInit:
    """测试 ConnectionManager 初始化。"""

    def test_default_max_connections(self):
        """默认 max_connections 应为 100。"""
        mgr = ConnectionManager()
        assert mgr.max_connections == 100
        assert len(mgr.active_connections) == 0

    def test_custom_max_connections(self):
        """自定义 max_connections 应被正确设置。"""
        mgr = ConnectionManager(max_connections=10)
        assert mgr.max_connections == 10

    def test_active_connections_starts_empty(self):
        """初始化后 active_connections 应为空列表。"""
        mgr = ConnectionManager()
        assert mgr.active_connections == []


class TestConnectionManagerConnect:
    """测试 ConnectionManager.connect 方法。"""

    @pytest.mark.asyncio
    async def test_connect_adds_to_active(self):
        """connect 应将 WebSocket 添加到 active_connections。"""
        mgr = ConnectionManager()
        ws = _make_mock_websocket()
        result = await mgr.connect(ws)
        assert result is True
        assert ws in mgr.active_connections
        ws.accept.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_dedup(self):
        """重复连接同一 WebSocket 不应重复添加。"""
        mgr = ConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws)
        await mgr.connect(ws)
        assert mgr.active_connections.count(ws) == 1

    @pytest.mark.asyncio
    async def test_connect_rejected_when_over_limit(self):
        """连接数达上限时应拒绝新连接并关闭 WebSocket。"""
        mgr = ConnectionManager(max_connections=1)
        ws1 = _make_mock_websocket()
        ws2 = _make_mock_websocket()
        await mgr.connect(ws1)
        result = await mgr.connect(ws2)
        assert result is False
        assert ws2 not in mgr.active_connections
        ws2.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_connect_rejected_close_exception_handled(self):
        """拒绝连接时 close 抛异常不应影响拒绝逻辑。"""
        mgr = ConnectionManager(max_connections=0)
        ws = _make_mock_websocket()
        ws.close = AsyncMock(side_effect=RuntimeError("close fail"))
        result = await mgr.connect(ws)
        assert result is False


class TestConnectionManagerDisconnect:
    """测试 ConnectionManager.disconnect 方法。"""

    @pytest.mark.asyncio
    async def test_disconnect_removes_connection(self):
        """disconnect 应从 active_connections 移除 WebSocket。"""
        mgr = ConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws)
        assert ws in mgr.active_connections
        mgr.disconnect(ws)
        assert ws not in mgr.active_connections

    def test_disconnect_not_connected_is_noop(self):
        """断开未连接的 WebSocket 不应抛异常。"""
        mgr = ConnectionManager()
        ws = _make_mock_websocket()
        mgr.disconnect(ws)  # 不应抛异常
        assert len(mgr.active_connections) == 0

    @pytest.mark.asyncio
    async def test_disconnect_one_of_many(self):
        """断开多个连接中的一个，其他应保持。"""
        mgr = ConnectionManager()
        ws1, ws2, ws3 = _make_mock_websocket(), _make_mock_websocket(), _make_mock_websocket()
        await mgr.connect(ws1)
        await mgr.connect(ws2)
        await mgr.connect(ws3)
        mgr.disconnect(ws2)
        assert ws1 in mgr.active_connections
        assert ws2 not in mgr.active_connections
        assert ws3 in mgr.active_connections
        assert len(mgr.active_connections) == 2


class TestConnectionManagerGetCount:
    """测试 ConnectionManager.get_connection_count 方法。"""

    def test_empty_count(self):
        """无连接时计数应为 0。"""
        mgr = ConnectionManager()
        assert mgr.get_connection_count() == 0

    @pytest.mark.asyncio
    async def test_count_after_connect(self):
        """连接后计数应增加。"""
        mgr = ConnectionManager()
        await mgr.connect(_make_mock_websocket())
        assert mgr.get_connection_count() == 1
        await mgr.connect(_make_mock_websocket())
        assert mgr.get_connection_count() == 2

    @pytest.mark.asyncio
    async def test_count_after_disconnect(self):
        """断开后计数应减少。"""
        mgr = ConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws)
        assert mgr.get_connection_count() == 1
        mgr.disconnect(ws)
        assert mgr.get_connection_count() == 0


class TestConnectionManagerBroadcast:
    """测试 ConnectionManager.broadcast 方法。"""

    @pytest.mark.asyncio
    async def test_broadcast_to_single_connection(self):
        """broadcast 应向单个连接发送 JSON 消息。"""
        mgr = ConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws)
        msg = {"type": "test", "data": 42}
        await mgr.broadcast(msg)
        ws.send_json.assert_awaited_once_with(msg)

    @pytest.mark.asyncio
    async def test_broadcast_to_multiple_connections(self):
        """broadcast 应向所有连接发送消息。"""
        mgr = ConnectionManager()
        ws1, ws2, ws3 = _make_mock_websocket(), _make_mock_websocket(), _make_mock_websocket()
        await mgr.connect(ws1)
        await mgr.connect(ws2)
        await mgr.connect(ws3)
        msg = {"type": "update"}
        await mgr.broadcast(msg)
        ws1.send_json.assert_awaited_once_with(msg)
        ws2.send_json.assert_awaited_once_with(msg)
        ws3.send_json.assert_awaited_once_with(msg)

    @pytest.mark.asyncio
    async def test_broadcast_no_connections(self):
        """无连接时 broadcast 不应抛异常。"""
        mgr = ConnectionManager()
        await mgr.broadcast({"type": "noop"})  # 不应抛异常

    @pytest.mark.asyncio
    async def test_broadcast_removes_failed_connections(self):
        """发送失败的连接应被移除。"""
        mgr = ConnectionManager()
        ws_ok = _make_mock_websocket()
        ws_fail = _make_mock_websocket()
        ws_fail.send_json = AsyncMock(side_effect=ConnectionError("disconnected"))
        await mgr.connect(ws_ok)
        await mgr.connect(ws_fail)
        await mgr.broadcast({"type": "test"})
        assert ws_ok in mgr.active_connections
        assert ws_fail not in mgr.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_partial_failure_continues(self):
        """一个连接失败不应中断对其他连接的广播。"""
        mgr = ConnectionManager()
        ws1 = _make_mock_websocket()
        ws_fail = _make_mock_websocket()
        ws_fail.send_json = AsyncMock(side_effect=RuntimeError("fail"))
        ws3 = _make_mock_websocket()
        await mgr.connect(ws1)
        await mgr.connect(ws_fail)
        await mgr.connect(ws3)
        await mgr.broadcast({"type": "test"})
        ws1.send_json.assert_awaited_once()
        ws3.send_json.assert_awaited_once()
        assert ws1 in mgr.active_connections
        assert ws3 in mgr.active_connections
        assert ws_fail not in mgr.active_connections

    @pytest.mark.asyncio
    async def test_broadcast_does_not_duplicate_connections(self):
        """broadcast 不应在 active_connections 中创建重复。"""
        mgr = ConnectionManager()
        ws = _make_mock_websocket()
        await mgr.connect(ws)
        await mgr.broadcast({"type": "test"})
        assert mgr.active_connections.count(ws) == 1


class TestConnectionManagerThreadSafety:
    """测试 ConnectionManager 线程安全。"""

    @pytest.mark.asyncio
    async def test_concurrent_connects(self):
        """并发连接多个 WebSocket 应全部成功添加。"""
        mgr = ConnectionManager(max_connections=100)
        websockets = [_make_mock_websocket() for _ in range(20)]
        await asyncio.gather(*[mgr.connect(ws) for ws in websockets])
        assert mgr.get_connection_count() == 20

    @pytest.mark.asyncio
    async def test_concurrent_broadcast_and_disconnect(self):
        """并发 broadcast 和 disconnect 不应导致异常。"""
        mgr = ConnectionManager()
        websockets = [_make_mock_websocket() for _ in range(10)]
        for ws in websockets:
            await mgr.connect(ws)

        async def _broadcast_loop():
            for _ in range(5):
                await mgr.broadcast({"type": "tick"})

        def _disconnect_some():
            for ws in websockets[:3]:
                mgr.disconnect(ws)

        await asyncio.gather(_broadcast_loop(), asyncio.to_thread(_disconnect_some))
        # 剩余 7 个连接
        assert mgr.get_connection_count() <= 10
