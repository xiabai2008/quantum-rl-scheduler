"""可视化 WebSocket 与 ConnectionManager 测试（拆分自 test_visualization.py，Issue #730）。"""

import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from src.visualization import state as vis_state
from src.visualization.app import (
    ConnectionManager,
    SystemStatusUpdate,
    TaskSubmit,
    app,
    simulate_scheduler,
    start_web_server,
    verify_api_key,
)
from src.visualization.security import MAX_CIRCUIT_QUBITS, rate_limiter

app_module = sys.modules["src.visualization.app"]


# ============================================================
# WebSocket 端点
# ============================================================


def test_websocket_endpoint_init_ping_and_invalid_json():
    """测试 WebSocket /ws 端点：init 消息、ping 心跳、非法 JSON 处理。

    使用 fastapi.testclient.TestClient（基于 httpx）测试 WebSocket，
    并将 simulate_scheduler mock 为空操作以避免后台任务干扰。
    """

    async def _noop_simulate():
        """空操作后台任务，供 lifespan 创建后立即完成。"""
        return None

    with (
        patch.object(app_module, "simulate_scheduler", _noop_simulate),
        TestClient(app) as client,
        client.websocket_connect("/ws") as ws,
    ):
        init_msg = ws.receive_json()
        assert init_msg["type"] == "init"
        assert "status" in init_msg
        assert "tasks" in init_msg
        # ping 心跳
        ws.send_text(json.dumps({"action": "ping"}))
        pong = ws.receive_json()
        assert pong["type"] == "pong"
        # 非法 JSON 应返回 error 而非断开
        ws.send_text("not-a-json")
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "Invalid JSON" in err["message"]


# ============================================================
# ConnectionManager 连接管理器
# ============================================================


@pytest.mark.asyncio
async def test_connection_manager_connect_disconnect():
    """connect 应接受连接并加入列表，disconnect 应移除。"""
    mgr = ConnectionManager()
    ws = AsyncMock()
    await mgr.connect(ws)
    assert ws in mgr.active_connections
    ws.accept.assert_called_once()
    mgr.disconnect(ws)
    assert ws not in mgr.active_connections
    # 重复 disconnect 不应抛错
    mgr.disconnect(ws)


@pytest.mark.asyncio
async def test_connection_manager_broadcast():
    """broadcast 应向所有连接的客户端发送消息。"""
    mgr = ConnectionManager()
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    await mgr.connect(ws1)
    await mgr.connect(ws2)
    await mgr.broadcast({"type": "test"})
    ws1.send_json.assert_called_once_with({"type": "test"})
    ws2.send_json.assert_called_once_with({"type": "test"})


@pytest.mark.asyncio
async def test_connection_manager_broadcast_removes_failed():
    """broadcast 应移除发送失败的连接，保留成功的连接。"""
    mgr = ConnectionManager()
    ws_failed = AsyncMock()
    ws_failed.send_json.side_effect = Exception("closed")
    ws_ok = AsyncMock()
    await mgr.connect(ws_failed)
    await mgr.connect(ws_ok)
    await mgr.broadcast({"type": "test"})
    assert ws_failed not in mgr.active_connections
    assert ws_ok in mgr.active_connections


@pytest.mark.asyncio
async def test_connection_manager_connect_dedup():
    """connect 应防护重复添加同一连接（Issue #216）。"""
    mgr = ConnectionManager()
    ws = AsyncMock()
    await mgr.connect(ws)
    # 重复 connect 不应重复添加
    await mgr.connect(ws)
    assert mgr.active_connections.count(ws) == 1


@pytest.mark.asyncio
async def test_websocket_endpoint_runtime_error_cleanup():
    """WebSocket 端点在 RuntimeError 时应通过 finally 清理连接（Issue #216）。"""
    from src.visualization import state as viz_state
    from src.visualization.websocket_handler import websocket_endpoint

    ws = AsyncMock()
    ws.receive_text.side_effect = RuntimeError("connection reset")

    with patch.object(viz_state, "manager") as mock_mgr:
        mock_mgr.connect = AsyncMock()
        mock_mgr.disconnect = MagicMock()
        # 执行端点函数，应捕获 RuntimeError 并清理
        await websocket_endpoint(ws)
        # finally 块应调用 disconnect
        mock_mgr.disconnect.assert_called_once_with(ws)


@pytest.mark.asyncio
async def test_websocket_endpoint_connection_closed_cleanup():
    """WebSocket 端点在 ConnectionError 时应通过 finally 清理连接（Issue #216）。"""
    from src.visualization import state as viz_state
    from src.visualization.websocket_handler import websocket_endpoint

    ws = AsyncMock()
    ws.receive_text.side_effect = ConnectionError("connection closed")

    with patch.object(viz_state, "manager") as mock_mgr:
        mock_mgr.connect = AsyncMock()
        mock_mgr.disconnect = MagicMock()
        await websocket_endpoint(ws)
        mock_mgr.disconnect.assert_called_once_with(ws)


async def _noop_simulate_scheduler() -> None:
    """空操作后台任务，供 TestClient lifespan 使用，避免后台任务干扰测试。"""
    return None


class TestWebSocket:
    """WebSocket /ws 端点测试（连接、广播、断开）。"""

    def test_init_message_structure(self):
        """连接后应收到 init 消息，包含 status/tasks/ppo_stats。"""
        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            msg = ws.receive_json()
            assert msg["type"] == "init"
            assert "status" in msg
            assert "tasks" in msg
            assert "ppo_stats" in msg

    def test_ping_pong(self):
        """发送 ping 心跳应收到 pong 响应。"""
        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            ws.receive_json()  # 消费 init 消息
            ws.send_text(json.dumps({"action": "ping"}))
            pong = ws.receive_json()
            assert pong["type"] == "pong"

    def test_invalid_json_returns_error(self):
        """发送非法 JSON 应返回 error 消息而非断开连接。"""
        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            ws.receive_json()
            ws.send_text("not-a-json")
            err = ws.receive_json()
            assert err["type"] == "error"
            assert "Invalid JSON" in err["message"]

    def test_disconnect_reduces_connection_count(self):
        """断开连接后 active_connections 数量应回落。"""
        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
        ):
            baseline = len(app_module.manager.active_connections)
            with client.websocket_connect("/ws") as ws:
                ws.receive_json()
                during = len(app_module.manager.active_connections)
                assert during >= baseline + 1
            # 退出 with 后连接应被清理
            assert len(app_module.manager.active_connections) < during

    @pytest.mark.asyncio
    async def test_post_task_triggers_broadcast(self, async_client, monkeypatch):
        """POST /api/tasks 应调用 manager.broadcast 广播 task_added 消息。"""
        broadcast_mock = AsyncMock()
        monkeypatch.setattr(app_module.manager, "broadcast", broadcast_mock)
        await async_client.post(
            "/api/tasks",
            json={
                "user_id": "u",
                "task_type": "quantum",
                "priority": 3,
                "qubit_count": 4,
                "circuit_depth": 10,
                "estimated_time": 5.0,
            },
        )
        broadcast_mock.assert_called_once()
        call_args = broadcast_mock.call_args[0][0]
        assert call_args["type"] == "task_added"
        assert "task" in call_args
        assert "status" in call_args


class TestWebSocketAdvanced:
    """WebSocket 高级功能测试（Issue #207，覆盖 websocket_handler.py 44-55 行）。"""

    @pytest.fixture(autouse=True)
    def restore_extended_state(self):
        """保存并恢复 _resource_history / _decision_log。"""
        saved_history = list(vis_state._resource_history)
        saved_log = list(vis_state._decision_log)
        yield
        vis_state._resource_history.clear()
        vis_state._resource_history.extend(saved_history)
        vis_state._decision_log.clear()
        vis_state._decision_log.extend(saved_log)

    def test_init_with_ppo_stats(self, tmp_path):
        """WebSocket init 消息应包含 PPO 排名数据（当结果文件存在时）。"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        sim_data = {"PPO": {"avg_reward": 2804}, "FCFS": {"avg_reward": 1456}}
        (results_dir / "simulation_results_test.json").write_text(
            json.dumps(sim_data), encoding="utf-8"
        )

        with (
            patch.object(app_module, "_PROJECT_ROOT", str(tmp_path)),
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            msg = ws.receive_json()
            assert msg["type"] == "init"
            assert "ppo_stats" in msg
            assert msg["ppo_stats"]["ppo_rank"] == 1
            assert msg["ppo_stats"]["total"] == 2

    def test_init_ppo_stats_no_files(self, tmp_path):
        """无结果文件时 ppo_stats 应为空字典。"""
        (tmp_path / "results").mkdir()

        with (
            patch.object(app_module, "_PROJECT_ROOT", str(tmp_path)),
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            msg = ws.receive_json()
            assert msg["type"] == "init"
            assert msg["ppo_stats"] == {}

    def test_init_ppo_stats_invalid_json(self, tmp_path):
        """结果文件非法 JSON 时 ppo_stats 应为空字典（优雅降级）。"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "simulation_results_test.json").write_text("not-json", encoding="utf-8")

        with (
            patch.object(app_module, "_PROJECT_ROOT", str(tmp_path)),
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            msg = ws.receive_json()
            assert msg["type"] == "init"
            assert msg["ppo_stats"] == {}

    def test_get_decisions_action(self):
        """发送 get_decisions 动作应返回决策日志。"""
        vis_state._decision_log.clear()
        vis_state._decision_log.extend([{"step": 1, "action": 0}])

        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            ws.receive_json()  # consume init
            ws.send_text(json.dumps({"action": "get_decisions"}))
            resp = ws.receive_json()
            assert resp["type"] == "decision_log"
            assert "decisions" in resp
            assert len(resp["decisions"]) == 1

    def test_get_resource_history_action(self):
        """发送 get_resource_history 动作应返回资源历史。"""
        vis_state._resource_history.clear()
        vis_state._resource_history.extend([{"step": 1, "qubit_utilization": 0.5}])

        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            ws.receive_json()  # consume init
            ws.send_text(json.dumps({"action": "get_resource_history"}))
            resp = ws.receive_json()
            assert resp["type"] == "resource_history"
            assert "history" in resp
            assert len(resp["history"]) == 1


class TestWebSocketOriginCheck:
    """WebSocket Origin 校验测试（Issue #514）。"""

    def test_is_origin_allowed_localhost(self):
        """localhost Origin 应允许。"""
        from src.visualization.security import is_origin_allowed

        assert is_origin_allowed("http://localhost:8000") is True
        assert is_origin_allowed("http://127.0.0.1") is True

    def test_is_origin_allowed_disallowed(self):
        """非白名单 Origin 应拒绝。"""
        from src.visualization.security import is_origin_allowed

        assert is_origin_allowed("http://evil.example.com") is False
        assert is_origin_allowed("https://attacker.net") is False

    def test_is_origin_allowed_empty_allows(self):
        """空 Origin（非浏览器客户端）应允许。"""
        from src.visualization.security import is_origin_allowed

        assert is_origin_allowed("") is True

    def test_get_allowed_ws_origins_from_env(self, monkeypatch):
        """VIZ_WS_ALLOWED_ORIGINS 环境变量应覆盖默认列表。"""
        from src.visualization.security import get_allowed_ws_origins

        monkeypatch.setenv(
            "VIZ_WS_ALLOWED_ORIGINS",
            "https://prod.example.com,https://staging.example.com",
        )
        origins = get_allowed_ws_origins()
        assert "https://prod.example.com" in origins
        assert "https://staging.example.com" in origins
        assert "http://localhost" not in origins

    def test_connection_manager_rejects_over_limit(self):
        """连接数超限时 connect 应拒绝新连接。"""
        from src.visualization.connection import ConnectionManager

        manager = ConnectionManager(max_connections=1)
        ws1 = MagicMock()
        ws1.accept = AsyncMock()
        ws1.close = AsyncMock()
        ws2 = MagicMock()
        ws2.accept = AsyncMock()
        ws2.close = AsyncMock()

        import asyncio

        loop = asyncio.new_event_loop()
        try:
            # 第一个连接成功
            assert loop.run_until_complete(manager.connect(ws1)) is True
            # 第二个连接应被拒绝
            assert loop.run_until_complete(manager.connect(ws2)) is False
            ws2.close.assert_awaited()
        finally:
            loop.close()
