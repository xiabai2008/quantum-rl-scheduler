"""
量子RL调度系统 - Web 可视化监控界面单元测试
Unit Tests for src/visualization/app.py

测试覆盖：
- FastAPI 路由（GET/POST/WebSocket）使用 httpx.AsyncClient + ASGITransport
- ConnectionManager 连接管理器（connect / disconnect / broadcast）
- 辅助函数：_load_vue3_template / _get_real_cqlib_client / _get_real_machines_status
              _load_real_submissions / _get_ppo_model / start_web_server
- 后台任务 simulate_scheduler 单次迭代（mock asyncio.sleep 退出循环）
- 状态隔离：autouse 夹具快照并恢复全局 system_status / task_queue / 连接管理器

所有真机/PPO 模型相关调用均通过 mock 替代，无需真实 TIANYAN_API_KEY 或训练好的模型。
"""

import asyncio
import copy
import json
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.app import (
    ConnectionManager,
    SystemStatusUpdate,
    TaskSubmit,
    app,
    simulate_scheduler,
    start_web_server,
    verify_api_key,
)

# 注意：src/visualization/__init__.py 执行了 `from src.visualization.app import app`，
# 这会覆盖 src.visualization 包的 app 属性为 FastAPI 实例，从而遮蔽 app 子模块。
# 因此 `import src.visualization.app as app_module` 会把 app_module 绑定为 FastAPI 实例，
# 而非模块对象。这里通过 sys.modules 直接获取真正的子模块对象，绕过属性遮蔽问题。
app_module = sys.modules["src.visualization.app"]

# ============================================================
# 公共夹具
# ============================================================


@pytest.fixture(autouse=True)
def reset_state():
    """快照并恢复全局 system_status / task_queue / 连接管理器，保证测试间隔离。"""
    saved_status = copy.deepcopy(app_module.system_status)
    saved_queue = copy.deepcopy(app_module.task_queue)
    saved_connections = list(app_module.manager.active_connections)
    saved_strategy = app_module.system_status.get("current_strategy")
    yield
    app_module.system_status.clear()
    app_module.system_status.update(copy.deepcopy(saved_status))
    app_module.task_queue.clear()
    app_module.task_queue.extend(copy.deepcopy(saved_queue))
    app_module.manager.active_connections = list(saved_connections)
    # current_strategy 可能被 POST /api/strategy 修改，强制还原
    app_module.system_status["current_strategy"] = saved_strategy


@pytest_asyncio.fixture
async def async_client():
    """提供基于 ASGITransport 的 httpx 异步客户端。

    ASGITransport 不会触发 FastAPI lifespan，因此后台任务 simulate_scheduler
    不会运行，保证测试期间全局状态不被后台任务修改。
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ============================================================
# 页面与基础 API 路由
# ============================================================


@pytest.mark.asyncio
async def test_root_returns_html(async_client):
    """GET / 应返回监控面板 HTML 页面。"""
    resp = await async_client.get("/")
    assert resp.status_code == 200
    assert "<html" in resp.text.lower()


@pytest.mark.asyncio
async def test_get_status(async_client):
    """GET /api/status 应返回系统状态字典。"""
    resp = await async_client.get("/api/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "qubit_utilization" in data
    assert "queue_length" in data
    assert "strategy_options" in data
    assert isinstance(data["strategy_options"], list)


@pytest.mark.asyncio
async def test_get_tasks_all(async_client):
    """GET /api/tasks 不带参数应返回全部任务列表。"""
    resp = await async_client.get("/api/tasks")
    assert resp.status_code == 200
    tasks = resp.json()
    assert isinstance(tasks, list)
    assert len(tasks) >= 1


@pytest.mark.asyncio
async def test_get_tasks_filter_by_status(async_client):
    """GET /api/tasks?status=pending 应只返回 pending 任务。"""
    resp = await async_client.get("/api/tasks", params={"status": "pending"})
    assert resp.status_code == 200
    tasks = resp.json()
    assert all(t["status"] == "pending" for t in tasks)


@pytest.mark.asyncio
async def test_get_tasks_filter_empty_result(async_client):
    """GET /api/tasks?status=completed 初始应返回空列表（无已完成任务）。"""
    resp = await async_client.get("/api/tasks", params={"status": "completed"})
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_submit_task(async_client):
    """POST /api/tasks 应提交新任务并返回 task_id，同时更新队列长度。"""
    payload = {
        "user_id": "test_user",
        "task_type": "quantum",
        "priority": 5,
        "qubit_count": 4,
        "circuit_depth": 50,
        "estimated_time": 10.0,
    }
    resp = await async_client.post("/api/tasks", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "task_id" in data
    assert data["task_id"].startswith("QTASK-")
    # 队列长度应至少为 1
    status_resp = await async_client.get("/api/status")
    assert status_resp.json()["queue_length"] >= 1


@pytest.mark.asyncio
async def test_get_metrics(async_client):
    """GET /api/metrics 应返回 Prometheus 格式指标文本。"""
    resp = await async_client.get("/api/metrics")
    assert resp.status_code == 200
    text = resp.text
    assert "quantum_scheduler_qubit_utilization" in text
    assert "quantum_scheduler_queue_length" in text
    assert "quantum_scheduler_completed_tasks" in text
    assert "quantum_scheduler_avg_wait_time" in text


def test_metrics_endpoint():
    """GET /metrics 应返回 Prometheus 文本格式指标，content-type 含 text/plain。

    使用 FastAPI TestClient 测试标准 Prometheus 采集端点。
    """

    async def _noop_simulate():
        """空操作后台任务，供 lifespan 创建后立即完成。"""
        return None

    with (
        patch.object(app_module, "simulate_scheduler", _noop_simulate),
        TestClient(app) as client,
    ):
        resp = client.get("/metrics")
        assert resp.status_code == 200
        content_type = resp.headers.get("content-type", "")
        assert "text/plain" in content_type
        # python_info 是 prometheus_client 默认暴露的进程指标
        body = resp.text
        assert "python_info" in body or "scheduler_" in body


@pytest.mark.asyncio
async def test_switch_strategy_valid(async_client):
    """POST /api/strategy?strategy=FCFS 应切换成功并更新当前策略。"""
    resp = await async_client.post("/api/strategy", params={"strategy": "FCFS"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert "FCFS" in data["message"]
    status = await async_client.get("/api/status")
    assert status.json()["current_strategy"] == "FCFS"


@pytest.mark.asyncio
async def test_switch_strategy_invalid(async_client):
    """POST /api/strategy?strategy=Unknown 应返回 success=False。"""
    resp = await async_client.post("/api/strategy", params={"strategy": "Unknown-Strategy"})
    assert resp.status_code == 200
    assert resp.json()["success"] is False


@pytest.mark.asyncio
async def test_update_status(async_client):
    """POST /api/update 应更新系统状态字段。"""
    payload = {
        "qubit_utilization": 0.88,
        "queue_length": 12,
        "completed_tasks": 100,
        "average_wait_time": 5.5,
    }
    resp = await async_client.post("/api/update", json=payload)
    assert resp.status_code == 200
    status = resp.json()["status"]
    assert status["qubit_utilization"] == 0.88
    assert status["queue_length"] == 12
    assert status["completed_tasks"] == 100
    assert status["average_wait_time"] == 5.5


# ============================================================
# 认证层与输入验证
# ============================================================


@pytest.mark.asyncio
async def test_api_key_auth_disabled(async_client, monkeypatch):
    """未配置 VISUALIZATION_API_KEY 时认证禁用，无 X-API-Key 也能访问 POST 端点。"""
    monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
    resp = await async_client.post("/api/strategy", params={"strategy": "FCFS"})
    assert resp.status_code == 200
    assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_api_key_auth_enabled(async_client, monkeypatch):
    """配置 VISUALIZATION_API_KEY 后，携带正确 X-API-Key 应访问成功。"""
    monkeypatch.setenv("VISUALIZATION_API_KEY", "secret-key-123")
    resp = await async_client.post(
        "/api/strategy",
        params={"strategy": "FCFS"},
        headers={"X-API-Key": "secret-key-123"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_api_key_auth_wrong_key(async_client, monkeypatch):
    """配置 VISUALIZATION_API_KEY 后，错误 X-API-Key 应返回 401。"""
    monkeypatch.setenv("VISUALIZATION_API_KEY", "secret-key-123")
    resp = await async_client.post(
        "/api/strategy",
        params={"strategy": "FCFS"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401
    assert "API" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_api_key_auth_missing_header(async_client, monkeypatch):
    """配置 VISUALIZATION_API_KEY 后，缺少 X-API-Key 头应返回 401。"""
    monkeypatch.setenv("VISUALIZATION_API_KEY", "secret-key-123")
    resp = await async_client.post("/api/strategy", params={"strategy": "FCFS"})
    assert resp.status_code == 401
    assert "API" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_api_key_auth_protects_all_post_endpoints(async_client, monkeypatch):
    """配置密钥后，所有 POST 端点（tasks/strategy/update）都应受认证保护。"""
    monkeypatch.setenv("VISUALIZATION_API_KEY", "secret-key-123")
    # POST /api/tasks 无头应 401
    resp_tasks = await async_client.post(
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
    assert resp_tasks.status_code == 401
    # POST /api/update 无头应 401
    resp_update = await async_client.post(
        "/api/update",
        json={
            "qubit_utilization": 0.5,
            "queue_length": 1,
            "completed_tasks": 1,
            "average_wait_time": 1.0,
        },
    )
    assert resp_update.status_code == 401


@pytest.mark.asyncio
async def test_api_key_auth_does_not_affect_get(async_client, monkeypatch):
    """配置密钥后，GET 端点（status/tasks/metrics）不应受认证影响。"""
    monkeypatch.setenv("VISUALIZATION_API_KEY", "secret-key-123")
    # GET /api/status 无头应 200
    assert (await async_client.get("/api/status")).status_code == 200
    # GET /api/tasks 无头应 200
    assert (await async_client.get("/api/tasks")).status_code == 200
    # GET /api/metrics 无头应 200
    assert (await async_client.get("/api/metrics")).status_code == 200


@pytest.mark.asyncio
async def test_input_validation_empty_task(async_client, monkeypatch):
    """POST /api/tasks 空 user_id 应被 Pydantic 拒绝（422）。"""
    monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
    payload = {
        "user_id": "",
        "task_type": "quantum",
        "priority": 3,
        "qubit_count": 4,
        "circuit_depth": 10,
        "estimated_time": 5.0,
    }
    resp = await async_client.post("/api/tasks", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_input_validation_empty_task_type(async_client, monkeypatch):
    """POST /api/tasks 空 task_type 应被 Pydantic 拒绝（422）。"""
    monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
    payload = {
        "user_id": "user_001",
        "task_type": "",
        "priority": 3,
        "qubit_count": 4,
        "circuit_depth": 10,
        "estimated_time": 5.0,
    }
    resp = await async_client.post("/api/tasks", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_input_validation_qubit_count_exceeds_limit(async_client, monkeypatch):
    """POST /api/tasks qubit_count 超过 287 上限应被拒绝（422）。"""
    monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
    payload = {
        "user_id": "user_001",
        "task_type": "quantum",
        "priority": 3,
        "qubit_count": 999,
        "circuit_depth": 10,
        "estimated_time": 5.0,
    }
    resp = await async_client.post("/api/tasks", json=payload)
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_input_validation_oversized_user_id(async_client, monkeypatch):
    """POST /api/tasks 超长 user_id（>128 字符）应被拒绝（422）。"""
    monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
    payload = {
        "user_id": "a" * 200,
        "task_type": "quantum",
        "priority": 3,
        "qubit_count": 4,
        "circuit_depth": 10,
        "estimated_time": 5.0,
    }
    resp = await async_client.post("/api/tasks", json=payload)
    assert resp.status_code == 422


# ============================================================
# 真机状态与提交记录路由
# ============================================================


@pytest.mark.asyncio
async def test_get_real_machines_no_client(async_client, monkeypatch):
    """GET /api/real-machines 无真机客户端时应返回空列表且 source=unavailable。"""
    monkeypatch.setattr(app_module, "_get_real_machines_status", lambda: [])
    resp = await async_client.get("/api/real-machines")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 0
    assert data["source"] == "unavailable"


@pytest.mark.asyncio
async def test_get_real_machines_with_client(async_client, monkeypatch):
    """GET /api/real-machines 有真机客户端时应返回机器列表且 source=cqlib。"""
    machines = [{"id": "1", "type": "superconducting", "status": "running", "name": "tianyan_s"}]
    monkeypatch.setattr(app_module, "_get_real_machines_status", lambda: machines)
    resp = await async_client.get("/api/real-machines")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 1
    assert data["source"] == "cqlib"
    assert data["machines"] == machines


@pytest.mark.asyncio
async def test_get_real_submissions(async_client, monkeypatch, tmp_path):
    """GET /api/real-submissions 应读取 results/real_times.json 并返回提交记录。"""
    records = [{"step": 1, "task_id": "t1"}, {"step": 2, "task_id": "t2"}]
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "real_times.json").write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    resp = await async_client.get("/api/real-submissions")
    assert resp.status_code == 200
    data = resp.json()
    assert data["count"] == 2
    # 倒序展示：第一条应为 step=2
    assert data["submissions"][0]["step"] == 2


# ============================================================
# PPO 数据接口路由
# ============================================================


def _write_sim_results(tmp_path: Path, data: dict) -> None:
    """在 tmp_path/results 下写入一个仿真结果 JSON 文件。"""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "simulation_results_test.json").write_text(json.dumps(data), encoding="utf-8")


@pytest.mark.asyncio
async def test_ppo_comparison_success(async_client, monkeypatch, tmp_path):
    """GET /api/ppo/comparison 成功路径：注入含 PPO 的仿真数据。"""
    _write_sim_results(
        tmp_path,
        {
            "PPO": {
                "avg_reward": 2804,
                "avg_wait_time": 10,
                "completion_rate": 1.0,
                "qubit_utilization": 0.45,
                "classical_utilization": 0.4,
            },
            "FCFS": {
                "avg_reward": 1456,
                "avg_wait_time": 12,
                "completion_rate": 1.0,
                "qubit_utilization": 0.46,
                "classical_utilization": 0.4,
            },
            "Random": {
                "avg_reward": 1267,
                "avg_wait_time": 15,
                "completion_rate": 1.0,
                "qubit_utilization": 0.41,
                "classical_utilization": 0.4,
            },
        },
    )
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    resp = await async_client.get("/api/ppo/comparison")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_strategies"] == 3
    assert data["ppo_rank"] == 1
    assert data["strategies"][0]["name"] == "PPO"
    assert data["strategies"][0]["rank"] == 1


@pytest.mark.asyncio
async def test_ppo_comparison_no_files(async_client, monkeypatch, tmp_path):
    """GET /api/ppo/comparison 仿真结果目录为空时应返回 error。"""
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    resp = await async_client.get("/api/ppo/comparison")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
    assert data["strategies"] == []


@pytest.mark.asyncio
async def test_ppo_comparison_invalid_json(async_client, monkeypatch, tmp_path):
    """GET /api/ppo/comparison 读取非法 JSON 时应返回 error。"""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "simulation_results_test.json").write_text("not-json", encoding="utf-8")
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    resp = await async_client.get("/api/ppo/comparison")
    assert resp.status_code == 200
    assert "error" in resp.json()


@pytest.mark.asyncio
async def test_ppo_stats_success(async_client, monkeypatch, tmp_path):
    """GET /api/ppo/stats 成功路径：注入含 PPO 的数据。"""
    _write_sim_results(
        tmp_path,
        {
            "PPO": {
                "avg_reward": 2804,
                "avg_wait_time": 10,
                "completion_rate": 1.0,
                "qubit_utilization": 0.45,
                "classical_utilization": 0.4,
            },
            "Random": {
                "avg_reward": 1267,
                "avg_wait_time": 15,
                "completion_rate": 1.0,
                "qubit_utilization": 0.41,
                "classical_utilization": 0.4,
            },
        },
    )
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    resp = await async_client.get("/api/ppo/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ppo_rank"] == 1
    assert data["total"] == 2
    assert data["best_strategy"] == "PPO"
    assert data["ppo"]["reward"] == 2804


@pytest.mark.asyncio
async def test_ppo_stats_no_files(async_client, monkeypatch, tmp_path):
    """GET /api/ppo/stats 仿真结果目录为空时应返回 error。"""
    (tmp_path / "results").mkdir()
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    resp = await async_client.get("/api/ppo/stats")
    assert resp.status_code == 200
    assert "error" in resp.json()


@pytest.mark.asyncio
async def test_ppo_stats_no_ppo_data(async_client, monkeypatch, tmp_path):
    """GET /api/ppo/stats 数据中无 PPO 键时应返回 '未找到 PPO 数据'。"""
    _write_sim_results(
        tmp_path,
        {
            "FCFS": {
                "avg_reward": 100,
                "avg_wait_time": 12,
                "completion_rate": 1.0,
                "qubit_utilization": 0.4,
                "classical_utilization": 0.4,
            }
        },
    )
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    resp = await async_client.get("/api/ppo/stats")
    assert resp.status_code == 200
    assert resp.json()["error"] == "未找到 PPO 数据"


@pytest.mark.asyncio
async def test_ppo_predict_no_model(async_client, monkeypatch):
    """GET /api/ppo/predict 模型未加载时应返回 error 且 action=None。"""
    monkeypatch.setattr(app_module, "_get_ppo_model", lambda: None)
    resp = await async_client.get("/api/ppo/predict")
    assert resp.status_code == 200
    data = resp.json()
    assert "error" in data
    assert data["action"] is None


@pytest.mark.asyncio
async def test_ppo_predict_success(async_client, monkeypatch):
    """GET /api/ppo/predict 成功路径：mock PPO 模型与环境推理。"""
    mock_model = MagicMock()
    mock_model.predict.return_value = (1, None)
    mock_obs = MagicMock()
    mock_obs.tolist.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    mock_env = MagicMock()
    mock_env.reset.return_value = (mock_obs, {})
    monkeypatch.setattr(app_module, "_get_ppo_model", lambda: mock_model)
    monkeypatch.setattr(app_module, "_ppo_env", mock_env)
    resp = await async_client.get("/api/ppo/predict")
    assert resp.status_code == 200
    data = resp.json()
    assert data["action"] == 1
    assert data["action_name"] == "量子资源"
    assert data["model_type"] == "PPO"
    assert len(data["observation"]) == 5


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


# ============================================================
# 辅助函数：_load_vue3_template
# ============================================================


def test_load_vue3_template_loads_file(monkeypatch):
    """_load_vue3_template 应加载前端 HTML 文件并缓存。"""
    monkeypatch.setattr(app_module, "_VUE3_HTML_TEMPLATE", None)
    result = app_module._load_vue3_template()
    assert "<html" in result.lower()
    # 第二次调用应使用缓存
    assert app_module._load_vue3_template() == result


def test_load_vue3_template_fallback(monkeypatch):
    """前端文件不存在时应回退到内置 HTML_TEMPLATE。"""
    monkeypatch.setattr(app_module, "_VUE3_HTML_TEMPLATE", None)
    monkeypatch.setattr(app_module, "FRONTEND_HTML_PATH", "/nonexistent/path/index.html")
    result = app_module._load_vue3_template()
    assert result == app_module.HTML_TEMPLATE


# ============================================================
# 辅助函数：_get_real_cqlib_client
# ============================================================


def test_get_real_cqlib_client_no_api_key(monkeypatch):
    """无 TIANYAN_API_KEY 时应返回 None 并标记已检查。"""
    monkeypatch.setattr(app_module, "_real_cqlib_client", None)
    monkeypatch.setattr(app_module, "_real_cqlib_checked", False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.delenv("TIANYAN_API_KEY", raising=False)
    assert app_module._get_real_cqlib_client() is None
    assert app_module._real_cqlib_checked is True


def test_get_real_cqlib_client_with_api_key(monkeypatch):
    """配置 TIANYAN_API_KEY 后应创建 cqlib 客户端并缓存。"""
    monkeypatch.setattr(app_module, "_real_cqlib_client", None)
    monkeypatch.setattr(app_module, "_real_cqlib_checked", False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setenv("TIANYAN_API_KEY", "fake-key-xyz")
    fake_client = MagicMock()
    fake_cls = MagicMock(return_value=fake_client)
    monkeypatch.setattr("src.api.tianyan_cqlib.CqlibTianyanClient", fake_cls)
    result = app_module._get_real_cqlib_client()
    assert result is fake_client
    fake_cls.assert_called_once()
    # 第二次调用应使用缓存，不再创建
    assert app_module._get_real_cqlib_client() is fake_client
    fake_cls.assert_called_once()


def test_get_real_cqlib_client_exception_returns_none(monkeypatch):
    """cqlib 客户端创建异常时应返回 None 并标记已检查。"""
    monkeypatch.setattr(app_module, "_real_cqlib_client", None)
    monkeypatch.setattr(app_module, "_real_cqlib_checked", False)
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    monkeypatch.setenv("TIANYAN_API_KEY", "fake-key")

    def _raise(**kwargs):
        raise Exception("conn fail")

    monkeypatch.setattr("src.api.tianyan_cqlib.CqlibTianyanClient", _raise)
    assert app_module._get_real_cqlib_client() is None
    assert app_module._real_cqlib_checked is True


# ============================================================
# 辅助函数：_get_real_machines_status
# ============================================================


def test_get_real_machines_status_no_client(monkeypatch):
    """无真机客户端时应返回空列表。"""
    monkeypatch.setattr(app_module, "_get_real_cqlib_client", lambda: None)
    assert app_module._get_real_machines_status() == []


def test_get_real_machines_status_with_client(monkeypatch):
    """有客户端且 list_backends 成功时应返回机器列表。"""
    fake_client = MagicMock()
    fake_client.list_backends.return_value = [
        {"id": "1", "type": "sc", "status": "running", "name": "tianyan_s"}
    ]
    monkeypatch.setattr(app_module, "_get_real_cqlib_client", lambda: fake_client)
    assert app_module._get_real_machines_status() == [
        {"id": "1", "type": "sc", "status": "running", "name": "tianyan_s"}
    ]


def test_get_real_machines_status_exception_returns_empty(monkeypatch):
    """list_backends 抛异常时应返回空列表。"""
    fake_client = MagicMock()
    fake_client.list_backends.side_effect = Exception("net down")
    monkeypatch.setattr(app_module, "_get_real_cqlib_client", lambda: fake_client)
    assert app_module._get_real_machines_status() == []


# ============================================================
# 辅助函数：_load_real_submissions
# ============================================================


def test_load_real_submissions_file_missing(monkeypatch, tmp_path):
    """real_times.json 不存在时应返回空列表。"""
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    assert app_module._load_real_submissions() == []


def test_load_real_submissions_valid(monkeypatch, tmp_path):
    """合法 JSON 列表应按倒序返回最近 50 条。"""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    records = [{"step": i, "task_id": f"t{i}"} for i in range(1, 4)]
    (results_dir / "real_times.json").write_text(json.dumps(records), encoding="utf-8")
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    result = app_module._load_real_submissions()
    assert len(result) == 3
    assert result[0]["step"] == 3  # 倒序


def test_load_real_submissions_invalid_json_returns_empty(monkeypatch, tmp_path):
    """非法 JSON 应返回空列表。"""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "real_times.json").write_text("not-json", encoding="utf-8")
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    assert app_module._load_real_submissions() == []


def test_load_real_submissions_non_list_returns_empty(monkeypatch, tmp_path):
    """JSON 内容非列表时应返回空列表。"""
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "real_times.json").write_text(json.dumps({"not": "a-list"}), encoding="utf-8")
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    assert app_module._load_real_submissions() == []


# ============================================================
# 辅助函数：_get_ppo_model
# ============================================================


def test_get_ppo_model_no_file_returns_none(monkeypatch, tmp_path):
    """无模型文件时应返回 None。"""
    monkeypatch.setattr(app_module, "_ppo_model", None)
    monkeypatch.setattr(app_module, "_ppo_env", None)
    monkeypatch.setattr("src.scheduler.env.QuantumSchedulingEnv", lambda **kw: MagicMock())
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    assert app_module._get_ppo_model() is None


def test_get_ppo_model_loads_model(monkeypatch, tmp_path):
    """存在模型文件时应调用 PPO.load 加载并缓存。"""
    monkeypatch.setattr(app_module, "_ppo_model", None)
    monkeypatch.setattr(app_module, "_ppo_env", None)
    fake_env = MagicMock()
    monkeypatch.setattr("src.scheduler.env.QuantumSchedulingEnv", lambda **kw: fake_env)
    fake_model = MagicMock()
    monkeypatch.setattr("stable_baselines3.PPO.load", lambda *a, **k: fake_model)
    # 在 tmp_path/models/ppo_seed_42_v4/ 下创建假的 best_model.zip
    models_dir = tmp_path / "models" / "ppo_seed_42_v4"
    models_dir.mkdir(parents=True)
    (models_dir / "best_model.zip").write_bytes(b"fake")
    monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
    result = app_module._get_ppo_model()
    assert result is fake_model


def test_get_ppo_model_exception_returns_none(monkeypatch):
    """环境构造抛异常时应捕获并返回 None。"""
    monkeypatch.setattr(app_module, "_ppo_model", None)
    monkeypatch.setattr(app_module, "_ppo_env", None)

    def _raise(**kwargs):
        raise RuntimeError("env init fail")

    monkeypatch.setattr("src.scheduler.env.QuantumSchedulingEnv", _raise)
    assert app_module._get_ppo_model() is None


# ============================================================
# 后台任务：simulate_scheduler
# ============================================================


@pytest.mark.asyncio
async def test_simulate_scheduler_one_iteration(monkeypatch):
    """测试 simulate_scheduler 单次迭代：mock asyncio.sleep 第二次抛 CancelledError 退出循环。"""
    sleep_calls = 0

    async def fake_sleep(_seconds):
        nonlocal sleep_calls
        sleep_calls += 1
        if sleep_calls >= 2:
            raise asyncio.CancelledError()

    monkeypatch.setattr(app_module.asyncio, "sleep", fake_sleep)
    monkeypatch.setattr(app_module, "_get_ppo_model", lambda: None)
    monkeypatch.setattr(app_module, "_get_real_machines_status", lambda: [])
    monkeypatch.setattr(app_module, "_load_real_submissions", lambda: [])
    # 控制 random 行为：触发任务状态迁移分支
    monkeypatch.setattr("random.uniform", lambda a, b: 0.0)
    monkeypatch.setattr("random.random", lambda: 0.0)
    monkeypatch.setattr("random.choice", lambda seq: seq[0])

    initial_step = app_module.system_status["current_step"]
    with pytest.raises(asyncio.CancelledError):
        await simulate_scheduler()

    # 第一次迭代后 current_step 应递增
    assert app_module.system_status["current_step"] == initial_step + 1


# ============================================================
# 入口函数：start_web_server
# ============================================================


def test_start_web_server_invokes_uvicorn(monkeypatch):
    """start_web_server 应调用 uvicorn.run（mock 避免实际启动）。"""
    captured = {}

    def fake_run(app_obj, host, port):
        captured["app"] = app_obj
        captured["host"] = host
        captured["port"] = port

    monkeypatch.setattr("uvicorn.run", fake_run)
    start_web_server(host="127.0.0.1", port=9999)
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 9999
    assert captured["app"] is app_module.app


# ============================================================
# Issue #75 扩展覆盖：API 端点 / Pydantic 验证 / 认证 / WebSocket / 错误处理
# ============================================================


async def _noop_simulate_scheduler() -> None:
    """空操作后台任务，供 TestClient lifespan 使用，避免后台任务干扰测试。"""
    return None


class TestApiStatusEndpoint:
    """GET /api/status 端点字段完整性与类型测试。"""

    @pytest.mark.asyncio
    async def test_returns_all_required_fields(self, async_client):
        """/api/status 应包含所有必需的状态字段。"""
        resp = await async_client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        required = {
            "qubit_utilization",
            "queue_length",
            "average_wait_time",
            "completed_tasks",
            "current_step",
            "current_strategy",
            "strategy_options",
            "real_machines",
            "real_submissions",
            "last_update",
        }
        assert required.issubset(data.keys())

    @pytest.mark.asyncio
    async def test_field_types(self, async_client):
        """/api/status 各字段类型应符合约定。"""
        data = (await async_client.get("/api/status")).json()
        assert isinstance(data["qubit_utilization"], (int, float))
        assert isinstance(data["queue_length"], int)
        assert isinstance(data["average_wait_time"], (int, float))
        assert isinstance(data["completed_tasks"], int)
        assert isinstance(data["current_step"], int)
        assert isinstance(data["current_strategy"], str)
        assert isinstance(data["strategy_options"], list)
        assert isinstance(data["real_machines"], list)
        assert isinstance(data["real_submissions"], list)
        assert isinstance(data["last_update"], str)

    @pytest.mark.asyncio
    async def test_qubit_utilization_in_range(self, async_client):
        """量子比特利用率应在 [0, 1] 区间。"""
        data = (await async_client.get("/api/status")).json()
        assert 0.0 <= data["qubit_utilization"] <= 1.0

    @pytest.mark.asyncio
    async def test_strategy_options_contains_known_strategies(self, async_client):
        """可选策略列表应包含已知策略。"""
        data = (await async_client.get("/api/status")).json()
        for s in ["DQN-Reward", "PPO-Balanced", "FCFS"]:
            assert s in data["strategy_options"]


class TestRealMachinesEndpoint:
    """GET /api/real-machines 端点测试（有/无真机客户端）。"""

    @pytest.mark.asyncio
    async def test_no_client_returns_unavailable(self, async_client, monkeypatch):
        """无真机客户端时应返回空列表、count=0、source=unavailable。"""
        monkeypatch.setattr(app_module, "_get_real_machines_status", lambda: [])
        resp = await async_client.get("/api/real-machines")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["machines"] == []
        assert data["source"] == "unavailable"

    @pytest.mark.asyncio
    async def test_with_client_returns_cqlib(self, async_client, monkeypatch):
        """有真机客户端时应返回机器列表且 source=cqlib。"""
        machines = [
            {"id": "tianyan_s", "type": "superconducting", "status": "running", "name": "天衍-S"},
            {
                "id": "tianyan_287",
                "type": "superconducting",
                "status": "calibrating",
                "name": "天衍-287",
            },
        ]
        monkeypatch.setattr(app_module, "_get_real_machines_status", lambda: machines)
        resp = await async_client.get("/api/real-machines")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert data["source"] == "cqlib"
        assert data["machines"] == machines

    @pytest.mark.asyncio
    async def test_response_structure_keys(self, async_client, monkeypatch):
        """返回结构应包含 machines/count/source 三个键。"""
        monkeypatch.setattr(app_module, "_get_real_machines_status", lambda: [])
        data = (await async_client.get("/api/real-machines")).json()
        assert set(data.keys()) == {"machines", "count", "source"}


class TestRealSubmissionsEndpoint:
    """GET /api/real-submissions 端点测试。"""

    @pytest.mark.asyncio
    async def test_returns_submissions_and_count(self, async_client, monkeypatch, tmp_path):
        """应返回提交记录列表及 count 字段。"""
        records = [{"step": i, "task_id": f"t{i}"} for i in range(1, 4)]
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "real_times.json").write_text(json.dumps(records), encoding="utf-8")
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
        resp = await async_client.get("/api/real-submissions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3
        assert isinstance(data["submissions"], list)

    @pytest.mark.asyncio
    async def test_no_file_returns_empty(self, async_client, monkeypatch, tmp_path):
        """real_times.json 不存在时应返回 count=0。"""
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
        resp = await async_client.get("/api/real-submissions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["submissions"] == []

    @pytest.mark.asyncio
    async def test_submissions_in_reverse_order(self, async_client, monkeypatch, tmp_path):
        """提交记录应按倒序返回（最新在前）。"""
        records = [{"step": 1}, {"step": 2}, {"step": 3}]
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "real_times.json").write_text(json.dumps(records), encoding="utf-8")
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
        data = (await async_client.get("/api/real-submissions")).json()
        assert [r["step"] for r in data["submissions"]] == [3, 2, 1]


class TestTasksEndpoint:
    """GET /api/tasks 与 POST /api/tasks 端点测试。"""

    @pytest.mark.asyncio
    async def test_get_all_returns_list(self, async_client):
        """GET /api/tasks 无参数应返回列表。"""
        resp = await async_client.get("/api/tasks")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    @pytest.mark.asyncio
    async def test_get_with_pending_filter(self, async_client):
        """GET /api/tasks?status=pending 应只返回 pending 任务。"""
        resp = await async_client.get("/api/tasks", params={"status": "pending"})
        assert resp.status_code == 200
        assert all(t["status"] == "pending" for t in resp.json())

    @pytest.mark.asyncio
    async def test_get_with_completed_filter_returns_empty(self, async_client):
        """GET /api/tasks?status=completed 初始应返回空列表。"""
        resp = await async_client.get("/api/tasks", params={"status": "completed"})
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_post_success_returns_task_id(self, async_client, monkeypatch):
        """POST /api/tasks 成功应返回 task_id 且以 QTASK- 开头。"""
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
        payload = {
            "user_id": "test_user",
            "task_type": "quantum",
            "priority": 4,
            "qubit_count": 8,
            "circuit_depth": 100,
            "estimated_time": 30.0,
        }
        resp = await async_client.post("/api/tasks", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"].startswith("QTASK-")
        assert "成功" in data["message"]

    @pytest.mark.asyncio
    async def test_post_invalid_priority_too_high(self, async_client, monkeypatch):
        """priority=6 超过上限应返回 422。"""
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
        payload = {
            "user_id": "u",
            "task_type": "quantum",
            "priority": 6,
            "qubit_count": 4,
            "circuit_depth": 10,
            "estimated_time": 5.0,
        }
        assert (await async_client.post("/api/tasks", json=payload)).status_code == 422

    @pytest.mark.asyncio
    async def test_post_invalid_priority_too_low(self, async_client, monkeypatch):
        """priority=0 低于下限应返回 422。"""
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
        payload = {
            "user_id": "u",
            "task_type": "quantum",
            "priority": 0,
            "qubit_count": 4,
            "circuit_depth": 10,
            "estimated_time": 5.0,
        }
        assert (await async_client.post("/api/tasks", json=payload)).status_code == 422

    @pytest.mark.asyncio
    async def test_post_invalid_qubit_count(self, async_client, monkeypatch):
        """qubit_count=0 低于下限应返回 422。"""
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
        payload = {
            "user_id": "u",
            "task_type": "quantum",
            "priority": 3,
            "qubit_count": 0,
            "circuit_depth": 10,
            "estimated_time": 5.0,
        }
        assert (await async_client.post("/api/tasks", json=payload)).status_code == 422

    @pytest.mark.asyncio
    async def test_post_increases_task_count(self, async_client, monkeypatch):
        """提交任务后任务总数应增加 1。"""
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
        before = len((await async_client.get("/api/tasks")).json())
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
        after = len((await async_client.get("/api/tasks")).json())
        assert after == before + 1


class TestMetricsEndpoints:
    """/api/metrics 与 /metrics 端点测试。"""

    @pytest.mark.asyncio
    async def test_api_metrics_text_format(self, async_client):
        """GET /api/metrics 应返回 Prometheus 文本格式，含 HELP/TYPE 注释行。"""
        resp = await async_client.get("/api/metrics")
        assert resp.status_code == 200
        text = resp.text
        assert "# HELP quantum_scheduler_qubit_utilization" in text
        assert "# TYPE quantum_scheduler_qubit_utilization gauge" in text
        assert "# TYPE quantum_scheduler_queue_length gauge" in text
        assert "# TYPE quantum_scheduler_completed_tasks counter" in text
        assert "# TYPE quantum_scheduler_current_step counter" in text

    @pytest.mark.asyncio
    async def test_api_metrics_contains_values(self, async_client):
        """GET /api/metrics 应包含具体指标值行。"""
        text = (await async_client.get("/api/metrics")).text
        assert "quantum_scheduler_qubit_utilization " in text
        assert "quantum_scheduler_queue_length " in text
        assert "quantum_scheduler_completed_tasks " in text

    def test_prometheus_metrics_content_type(self):
        """GET /metrics 应返回 text/plain; version=... 格式。"""
        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
        ):
            resp = client.get("/metrics")
            assert resp.status_code == 200
            content_type = resp.headers.get("content-type", "")
            assert "text/plain" in content_type
            assert "version=" in content_type

    def test_prometheus_metrics_body_contains_process_info(self):
        """GET /metrics body 应包含 prometheus_client 默认进程指标。"""
        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
        ):
            body = client.get("/metrics").text
            # prometheus_client 默认暴露 python_ 或 process_ 指标
            assert "python_info" in body or "process_" in body or "scheduler_" in body


class TestHealthEndpoints:
    """/health 与 /ready 健康检查端点测试（Issue #214）。"""

    @pytest.mark.asyncio
    async def test_health_returns_alive(self, async_client):
        """/health 应返回 status=alive，且不依赖任何外部资源。"""
        resp = await async_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"

    @pytest.mark.asyncio
    async def test_ready_returns_checks_dict(self, async_client):
        """/ready 应返回包含 checks 字段的就绪状态字典。"""
        resp = await async_client.get("/ready")
        assert resp.status_code == 200
        data = resp.json()
        # 必需字段
        assert "ready" in data
        assert "checks" in data
        assert "timestamp" in data
        # checks 字典应包含核心组件检查
        checks = data["checks"]
        assert "app" in checks
        assert "metrics" in checks
        # app 检查应为 ok=True（FastAPI 实例总是存在）
        assert checks["app"]["ok"] is True

    @pytest.mark.asyncio
    async def test_ready_required_components_ok(self, async_client):
        """所有 required=True 的组件就绪时，ready 应为 true。"""
        resp = await async_client.get("/ready")
        data = resp.json()
        # app 与 metrics 是 required=True（默认）
        # 测试环境下 app 与 metrics 总是 ok，故 ready 应为 True
        assert data["checks"]["app"]["ok"] is True
        assert data["checks"]["metrics"]["ok"] is True
        assert data["ready"] is True
        assert data["required_ok"] is True

    @pytest.mark.asyncio
    async def test_ready_optional_components_marked_not_required(self, async_client):
        """PPO 模型与配额追踪器是可选依赖，required 应为 False。"""
        resp = await async_client.get("/ready")
        checks = resp.json()["checks"]
        # PPO 模型与配额追踪器应为 required=False
        assert checks["ppo_model"].get("required") is False
        assert checks["quota_tracker"].get("required") is False

    @pytest.mark.asyncio
    async def test_ready_includes_timestamp_iso_format(self, async_client):
        """/ready 返回的 timestamp 应为 ISO 8601 格式。"""
        resp = await async_client.get("/ready")
        ts = resp.json()["timestamp"]
        # 应可被 fromisoformat 解析
        from datetime import datetime

        parsed = datetime.fromisoformat(ts)
        assert isinstance(parsed, datetime)


class TestStrategyEndpoint:
    """POST /api/strategy 端点测试。"""

    @pytest.mark.asyncio
    async def test_known_strategy_switches(self, async_client, monkeypatch):
        """已知策略应切换成功且更新 current_strategy。"""
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
        resp = await async_client.post("/api/strategy", params={"strategy": "DQN-Reward"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "DQN-Reward" in data["message"]
        status = await async_client.get("/api/status")
        assert status.json()["current_strategy"] == "DQN-Reward"

    @pytest.mark.asyncio
    async def test_unknown_strategy_fails(self, async_client, monkeypatch):
        """未知策略应返回 success=False 且不修改 current_strategy。"""
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
        before = (await async_client.get("/api/status")).json()["current_strategy"]
        resp = await async_client.post("/api/strategy", params={"strategy": "NonExistent"})
        assert resp.status_code == 200
        assert resp.json()["success"] is False
        after = (await async_client.get("/api/status")).json()["current_strategy"]
        assert after == before

    @pytest.mark.asyncio
    async def test_auth_missing_returns_401(self, async_client, monkeypatch):
        """配置密钥后缺少 X-API-Key 应返回 401。"""
        monkeypatch.setenv("VISUALIZATION_API_KEY", "secret-key-123")
        resp = await async_client.post("/api/strategy", params={"strategy": "FCFS"})
        assert resp.status_code == 401


class TestUpdateEndpoint:
    """POST /api/update 端点测试。"""

    @pytest.mark.asyncio
    async def test_update_success(self, async_client, monkeypatch):
        """合法 payload 应更新系统状态字段。"""
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
        payload = {
            "qubit_utilization": 0.77,
            "queue_length": 9,
            "completed_tasks": 50,
            "average_wait_time": 7.7,
        }
        resp = await async_client.post("/api/update", json=payload)
        assert resp.status_code == 200
        status = resp.json()["status"]
        assert status["qubit_utilization"] == 0.77
        assert status["queue_length"] == 9
        assert status["completed_tasks"] == 50
        assert status["average_wait_time"] == 7.7

    @pytest.mark.asyncio
    async def test_qubit_utilization_out_of_bounds(self, async_client, monkeypatch):
        """qubit_utilization>1.0 应返回 422。"""
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
        payload = {
            "qubit_utilization": 1.5,
            "queue_length": 1,
            "completed_tasks": 1,
            "average_wait_time": 1.0,
        }
        assert (await async_client.post("/api/update", json=payload)).status_code == 422

    @pytest.mark.asyncio
    async def test_negative_queue_length_out_of_bounds(self, async_client, monkeypatch):
        """queue_length<0 应返回 422。"""
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
        payload = {
            "qubit_utilization": 0.5,
            "queue_length": -1,
            "completed_tasks": 1,
            "average_wait_time": 1.0,
        }
        assert (await async_client.post("/api/update", json=payload)).status_code == 422

    @pytest.mark.asyncio
    async def test_average_wait_time_out_of_bounds(self, async_client, monkeypatch):
        """average_wait_time>86400 应返回 422。"""
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
        payload = {
            "qubit_utilization": 0.5,
            "queue_length": 1,
            "completed_tasks": 1,
            "average_wait_time": 90000.0,
        }
        assert (await async_client.post("/api/update", json=payload)).status_code == 422

    @pytest.mark.asyncio
    async def test_auth_missing_returns_401(self, async_client, monkeypatch):
        """配置密钥后缺少 X-API-Key 应返回 401。"""
        monkeypatch.setenv("VISUALIZATION_API_KEY", "secret-key-123")
        resp = await async_client.post(
            "/api/update",
            json={
                "qubit_utilization": 0.5,
                "queue_length": 1,
                "completed_tasks": 1,
                "average_wait_time": 1.0,
            },
        )
        assert resp.status_code == 401


class TestPydanticValidation:
    """TaskSubmit / SystemStatusUpdate Pydantic 字段边界值测试。"""

    def test_task_submit_qubit_count_exceeds_287(self):
        """qubit_count=288 超过 287 上限应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            TaskSubmit(qubit_count=288)

    def test_task_submit_qubit_count_at_max(self):
        """qubit_count=287 应通过（边界值）。"""
        t = TaskSubmit(qubit_count=287)
        assert t.qubit_count == 287

    def test_task_submit_qubit_count_below_min(self):
        """qubit_count=0 低于下限应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            TaskSubmit(qubit_count=0)

    def test_task_submit_priority_exceeds_max(self):
        """priority=6 超过 5 上限应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            TaskSubmit(priority=6)

    def test_task_submit_priority_below_min(self):
        """priority=0 低于 1 下限应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            TaskSubmit(priority=0)

    def test_task_submit_estimated_time_below_min(self):
        """estimated_time=0.05 低于 0.1 下限应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            TaskSubmit(estimated_time=0.05)

    def test_task_submit_estimated_time_above_max(self):
        """estimated_time=86401 超过 86400 上限应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            TaskSubmit(estimated_time=86401.0)

    def test_task_submit_circuit_depth_below_min(self):
        """circuit_depth=0 低于 1 下限应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            TaskSubmit(circuit_depth=0)

    def test_task_submit_circuit_depth_above_max(self):
        """circuit_depth=10001 超过 10000 上限应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            TaskSubmit(circuit_depth=10001)

    def test_task_submit_user_id_empty(self):
        """user_id 为空应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            TaskSubmit(user_id="")

    def test_task_submit_user_id_too_long(self):
        """user_id 超过 128 字符应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            TaskSubmit(user_id="a" * 200)

    def test_task_submit_task_type_empty(self):
        """task_type 为空应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            TaskSubmit(task_type="")

    def test_task_submit_defaults_valid(self):
        """TaskSubmit 默认值应全部合法。"""
        t = TaskSubmit()
        assert t.user_id == "user_001"
        assert t.task_type == "quantum"
        assert t.priority == 3
        assert t.qubit_count == 10
        assert t.circuit_depth == 100
        assert t.estimated_time == 60.0

    def test_system_status_update_qubit_utilization_above_max(self):
        """qubit_utilization>1.0 应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            SystemStatusUpdate(qubit_utilization=1.1)

    def test_system_status_update_negative(self):
        """qubit_utilization<0 应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            SystemStatusUpdate(qubit_utilization=-0.1)

    def test_system_status_update_queue_length_negative(self):
        """queue_length<0 应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            SystemStatusUpdate(queue_length=-1)

    def test_system_status_update_average_wait_time_above_max(self):
        """average_wait_time>86400 应抛 ValidationError。"""
        with pytest.raises(ValidationError):
            SystemStatusUpdate(average_wait_time=100000.0)

    def test_system_status_update_defaults_valid(self):
        """SystemStatusUpdate 默认值应全部合法。"""
        u = SystemStatusUpdate()
        assert u.qubit_utilization == 0.0
        assert u.queue_length == 0
        assert u.completed_tasks == 0
        assert u.average_wait_time == 0.0


class TestAuthLayer:
    """verify_api_key 认证层测试（未配置/缺失/不匹配/匹配）。"""

    @pytest.mark.asyncio
    async def test_no_key_configured_allows(self, monkeypatch):
        """未配置 VISUALIZATION_API_KEY 时应放行（返回 None）。"""
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
        assert await verify_api_key(x_api_key=None) is None

    @pytest.mark.asyncio
    async def test_missing_header_rejected(self, monkeypatch):
        """配置密钥后缺失 X-API-Key 应抛 HTTPException 401。"""
        monkeypatch.setenv("VISUALIZATION_API_KEY", "secret-key-123")
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(x_api_key=None)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_key_rejected(self, monkeypatch):
        """配置密钥后不匹配的 X-API-Key 应抛 HTTPException 401。"""
        monkeypatch.setenv("VISUALIZATION_API_KEY", "secret-key-123")
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(x_api_key="wrong")
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_correct_key_allows(self, monkeypatch):
        """配置密钥后匹配的 X-API-Key 应放行。"""
        monkeypatch.setenv("VISUALIZATION_API_KEY", "secret-key-123")
        assert await verify_api_key(x_api_key="secret-key-123") is None

    @pytest.mark.asyncio
    async def test_empty_env_value_disables_auth(self, monkeypatch):
        """VISUALIZATION_API_KEY 为空字符串时应禁用认证。"""
        monkeypatch.setenv("VISUALIZATION_API_KEY", "")
        assert await verify_api_key(x_api_key=None) is None


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
        monkeypatch.delenv("VISUALIZATION_API_KEY", raising=False)
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


class TestErrorHandling:
    """端点错误处理路径测试。"""

    @pytest.mark.asyncio
    async def test_real_submissions_invalid_json_returns_empty(
        self, async_client, monkeypatch, tmp_path
    ):
        """real_times.json 非法 JSON 时 /api/real-submissions 应返回 count=0。"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "real_times.json").write_text("not-json", encoding="utf-8")
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
        data = (await async_client.get("/api/real-submissions")).json()
        assert data["count"] == 0
        assert data["submissions"] == []

    @pytest.mark.asyncio
    async def test_ppo_comparison_invalid_json_returns_error(
        self, async_client, monkeypatch, tmp_path
    ):
        """仿真结果文件非法 JSON 时 /api/ppo/comparison 应返回 error。"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "simulation_results_test.json").write_text("not-json", encoding="utf-8")
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
        data = (await async_client.get("/api/ppo/comparison")).json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_ppo_stats_invalid_json_returns_error(self, async_client, monkeypatch, tmp_path):
        """仿真结果文件非法 JSON 时 /api/ppo/stats 应返回 error。"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "simulation_results_test.json").write_text("not-json", encoding="utf-8")
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
        data = (await async_client.get("/api/ppo/stats")).json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_ppo_predict_exception_returns_error(self, async_client, monkeypatch):
        """PPO 推理抛异常时应返回 error 且 action=None。"""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("infer fail")
        mock_env = MagicMock()
        mock_env.reset.return_value = ([0.0], {})
        monkeypatch.setattr(app_module, "_get_ppo_model", lambda: mock_model)
        monkeypatch.setattr(app_module, "_ppo_env", mock_env)
        data = (await async_client.get("/api/ppo/predict")).json()
        assert "error" in data
        assert data["action"] is None

    @pytest.mark.asyncio
    async def test_ppo_comparison_no_files_returns_error(self, async_client, monkeypatch, tmp_path):
        """无仿真结果文件时 /api/ppo/comparison 应返回 error。"""
        (tmp_path / "results").mkdir()
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))
        data = (await async_client.get("/api/ppo/comparison")).json()
        assert "error" in data
        assert data["strategies"] == []

    @pytest.mark.asyncio
    async def test_ppo_predict_no_env_returns_error(self, async_client, monkeypatch):
        """模型已加载但环境未初始化时应返回 error。"""
        mock_model = MagicMock()
        monkeypatch.setattr(app_module, "_get_ppo_model", lambda: mock_model)
        monkeypatch.setattr(app_module, "_ppo_env", None)
        data = (await async_client.get("/api/ppo/predict")).json()
        assert "error" in data


class TestWebSocketHandlerPPOStats:
    """WebSocket /ws 端点 PPO 统计读取测试。

    覆盖 src/visualization/websocket_handler.py 第 40-51 行：成功读取
    simulation_results JSON 并提取 PPO 排名，以及三种异常路径
    （JSONDecodeError / OSError / 无结果文件）。
    """

    def test_websocket_init_with_ppo_rank(self, tmp_path, monkeypatch):
        """有仿真结果文件且含 PPO 策略时应返回正确排名。"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        sim_data = {
            "PPO": {"avg_reward": 2746.94},
            "FCFS": {"avg_reward": 1458.77},
            "Random": {"avg_reward": 1000.0},
        }
        (results_dir / "simulation_results_001.json").write_text(
            json.dumps(sim_data), encoding="utf-8"
        )
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))

        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            msg = ws.receive_json()
            assert msg["type"] == "init"
            assert msg["ppo_stats"]["ppo_rank"] == 1
            assert msg["ppo_stats"]["total"] == 3

    def test_websocket_init_ppo_not_in_results(self, tmp_path, monkeypatch):
        """结果文件中不含 PPO 策略时 ppo_rank 应为 None。"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        sim_data = {
            "FCFS": {"avg_reward": 1458.77},
            "Random": {"avg_reward": 1000.0},
        }
        (results_dir / "simulation_results_001.json").write_text(
            json.dumps(sim_data), encoding="utf-8"
        )
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))

        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            msg = ws.receive_json()
            assert msg["type"] == "init"
            assert msg["ppo_stats"]["ppo_rank"] is None
            assert msg["ppo_stats"]["total"] == 2

    def test_websocket_init_no_simulation_files(self, tmp_path, monkeypatch):
        """results 目录存在但无仿真结果文件时 ppo_stats 应为空。"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))

        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            msg = ws.receive_json()
            assert msg["type"] == "init"
            assert msg["ppo_stats"] == {}

    def test_websocket_init_json_decode_error(self, tmp_path, monkeypatch):
        """仿真结果文件 JSON 格式错误时不应崩溃，ppo_stats 应为空。"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "simulation_results_001.json").write_text("{invalid json", encoding="utf-8")
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))

        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            msg = ws.receive_json()
            assert msg["type"] == "init"
            assert msg["ppo_stats"] == {}

    def test_websocket_init_oserror_no_results_dir(self, tmp_path, monkeypatch):
        """results 目录不存在时不应崩溃，ppo_stats 应为空。"""
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))

        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            msg = ws.receive_json()
            assert msg["type"] == "init"
            assert msg["ppo_stats"] == {}

    def test_websocket_init_oserror_on_open(self, tmp_path, monkeypatch):
        """仿真结果文件无法读取（IsADirectoryError）时不应崩溃。"""
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        # 创建同名目录使 open() 抛出 IsADirectoryError（OSError 子类）
        (results_dir / "simulation_results_001.json").mkdir()
        monkeypatch.setattr(app_module, "_PROJECT_ROOT", str(tmp_path))

        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            TestClient(app) as client,
            client.websocket_connect("/ws") as ws,
        ):
            msg = ws.receive_json()
            assert msg["type"] == "init"
            assert msg["ppo_stats"] == {}
