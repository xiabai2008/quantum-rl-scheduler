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
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.app import (
    ConnectionManager,
    app,
    simulate_scheduler,
    start_web_server,
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
        patch("src.visualization.app.simulate_scheduler", _noop_simulate),
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
