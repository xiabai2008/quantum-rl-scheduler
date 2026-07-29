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

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

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
from src.visualization.security import rate_limiter

# 注意：src/visualization/__init__.py 执行了 `from src.visualization.app import app`，
# 这会覆盖 src.visualization 包的 app 属性为 FastAPI 实例，从而遮蔽 app 子模块。
# 因此 `import src.visualization.app as app_module` 会把 app_module 绑定为 FastAPI 实例，
# 而非模块对象。这里通过 sys.modules 直接获取真正的子模块对象，绕过属性遮蔽问题。
app_module = sys.modules["src.visualization.app"]

# ============================================================
# 公共夹具
# ============================================================


# 测试统一认证密钥：默认开启认证，所有写请求经客户端默认头携带，
# 以验证“配置密钥时写操作受保护”的正确安全行为。需测试“无密钥拒绝写”
# 的用例显式 monkeypatch.delenv("VIZ_API_KEY") 即可。
TEST_VIZ_KEY = "test-viz-key"


@pytest.fixture(autouse=True)
def reset_state():
    """快照并恢复全局 system_status / task_queue / 连接管理器，保证测试间隔离。

    同时重置速率限制器状态（Issue #517），避免 POST 端点测试因共享 IP
    （testserver/127.0.0.1）累积请求而触发 429 限流，影响后续测试。
    """
    saved_status = copy.deepcopy(app_module.system_status)
    saved_queue = copy.deepcopy(app_module.task_queue)
    saved_connections = list(app_module.manager.active_connections)
    saved_strategy = app_module.system_status.get("current_strategy")
    # 重置速率限制器，确保每个测试从干净状态开始（Issue #517）
    rate_limiter.reset()
    yield
    app_module.system_status.clear()
    app_module.system_status.update(copy.deepcopy(saved_status))
    app_module.task_queue.clear()
    app_module.task_queue.extend(copy.deepcopy(saved_queue))
    app_module.manager.active_connections = list(saved_connections)
    # current_strategy 可能被 POST /api/strategy 修改，强制还原
    app_module.system_status["current_strategy"] = saved_strategy
    # 测试结束后再次重置速率限制器，避免失败用例残留状态影响后续测试
    rate_limiter.reset()


@pytest.fixture(autouse=True)
def default_viz_auth(monkeypatch):
    """测试默认配置 VIZ_API_KEY，使写操作需经认证（符合修复后的安全模型）。"""
    monkeypatch.setenv("VIZ_API_KEY", TEST_VIZ_KEY)


@pytest_asyncio.fixture
async def async_client():
    """提供基于 ASGITransport 的 httpx 异步客户端。

    ASGITransport 不会触发 FastAPI lifespan，因此后台任务 simulate_scheduler
    不会运行，保证测试期间全局状态不被后台任务修改。默认注入 X-API-Key 头，
    使写请求通过认证；如需测试“无密钥/错误密钥”场景，请显式 delenv 或传 header。
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"X-API-Key": TEST_VIZ_KEY},
    ) as client:
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

    Issue #513: /metrics 需要严格认证，测试时配置 VIZ_API_KEY 并提供 X-API-Key。
    """

    async def _noop_simulate():
        """空操作后台任务，供 lifespan 创建后立即完成。"""
        return None

    with (
        patch.object(app_module, "simulate_scheduler", _noop_simulate),
        patch.dict(os.environ, {"VIZ_API_KEY": "test-metrics-key"}),
        TestClient(app) as client,
    ):
        resp = client.get("/metrics", headers={"X-API-Key": "test-metrics-key"})
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
async def test_api_key_not_configured_blocks_write(async_client, monkeypatch):
    """未配置 VIZ_API_KEY 时，写操作（POST）必须返回 401；GET 监控端点仍放行。"""
    monkeypatch.delenv("VIZ_API_KEY", raising=False)
    # 写操作应被拒绝（零认证放行的安全漏洞已修复）
    resp_post = await async_client.post(
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
    assert resp_post.status_code == 401
    # GET 监控端点不受影响，仍返回 200
    resp_get = await async_client.get("/api/status")
    assert resp_get.status_code == 200


@pytest.mark.asyncio
async def test_write_request_401_without_key(async_client, monkeypatch):
    """未配置 VIZ_API_KEY 时，POST /api/tasks 必须返回 401（写操作零认证已修复）。"""
    monkeypatch.delenv("VIZ_API_KEY", raising=False)
    resp = await async_client.post(
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
    assert resp.status_code == 401


@pytest.mark.asyncio
async def test_correct_key_allows_write(async_client, monkeypatch):
    """配置正确 VIZ_API_KEY 且携带 X-API-Key 头时，POST /api/tasks 应成功（200）。"""
    monkeypatch.setenv("VIZ_API_KEY", "secret")
    resp = await async_client.post(
        "/api/tasks",
        json={
            "user_id": "u",
            "task_type": "quantum",
            "priority": 3,
            "qubit_count": 4,
            "circuit_depth": 10,
            "estimated_time": 5.0,
        },
        headers={"X-API-Key": "secret"},
    )
    assert resp.status_code == 200
    assert "task_id" in resp.json()


@pytest.mark.asyncio
async def test_hmac_compare_digest_called(async_client, monkeypatch):
    """写请求认证必须使用 hmac.compare_digest（恒定时间比较），替代不安全的 !=。"""
    from unittest.mock import Mock

    import src.visualization.routes as routes

    monkeypatch.setenv("VIZ_API_KEY", "secret")
    # 正确密钥：恒定时间比较应被调用一次，参数为 (提交值, 期望值)
    mock_ok = Mock(return_value=True)
    monkeypatch.setattr(routes.hmac, "compare_digest", mock_ok)
    resp_ok = await async_client.post(
        "/api/tasks",
        json={
            "user_id": "u",
            "task_type": "quantum",
            "priority": 3,
            "qubit_count": 4,
            "circuit_depth": 10,
            "estimated_time": 5.0,
        },
        headers={"X-API-Key": "secret"},
    )
    assert resp_ok.status_code == 200
    mock_ok.assert_called_once_with("secret", "secret")

    # 错误密钥：恒定时间比较仍应被调用，并最终返回 401
    mock_bad = Mock(return_value=False)
    monkeypatch.setattr(routes.hmac, "compare_digest", mock_bad)
    resp_bad = await async_client.post(
        "/api/tasks",
        json={
            "user_id": "u",
            "task_type": "quantum",
            "priority": 3,
            "qubit_count": 4,
            "circuit_depth": 10,
            "estimated_time": 5.0,
        },
        headers={"X-API-Key": "wrong"},
    )
    assert resp_bad.status_code == 401
    mock_bad.assert_called_once_with("wrong", "secret")


@pytest.mark.asyncio
async def test_api_key_auth_enabled(async_client, monkeypatch):
    """配置 VIZ_API_KEY 后，携带正确 X-API-Key 应访问成功。"""
    monkeypatch.setenv("VIZ_API_KEY", "secret-key-123")
    resp = await async_client.post(
        "/api/strategy",
        params={"strategy": "FCFS"},
        headers={"X-API-Key": "secret-key-123"},
    )
    assert resp.status_code == 200
    assert resp.json()["success"] is True


@pytest.mark.asyncio
async def test_api_key_auth_wrong_key(async_client, monkeypatch):
    """配置 VIZ_API_KEY 后，错误 X-API-Key 应返回 401。"""
    monkeypatch.setenv("VIZ_API_KEY", "secret-key-123")
    resp = await async_client.post(
        "/api/strategy",
        params={"strategy": "FCFS"},
        headers={"X-API-Key": "wrong-key"},
    )
    assert resp.status_code == 401
    assert "API" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_api_key_auth_missing_header(async_client, monkeypatch):
    """配置 VIZ_API_KEY 后，缺少 X-API-Key 头应返回 401。"""
    monkeypatch.setenv("VIZ_API_KEY", "secret-key-123")
    resp = await async_client.post("/api/strategy", params={"strategy": "FCFS"})
    assert resp.status_code == 401
    assert "API" in resp.json()["detail"]


@pytest.mark.asyncio
async def test_api_key_auth_protects_all_post_endpoints(async_client, monkeypatch):
    """配置密钥后，所有 POST 端点（tasks/strategy/update）都应受认证保护。"""
    monkeypatch.setenv("VIZ_API_KEY", "secret-key-123")
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
    """配置密钥后，GET 端点（status/tasks/api/metrics）不受认证影响，但 /metrics 需认证。"""
    monkeypatch.setenv("VIZ_API_KEY", "secret-key-123")
    # GET /api/status 无头应 200
    assert (await async_client.get("/api/status")).status_code == 200
    # GET /api/tasks 无头应 200
    assert (await async_client.get("/api/tasks")).status_code == 200
    # GET /api/metrics 无头应 200（verify_api_key 豁免 GET）
    assert (await async_client.get("/api/metrics")).status_code == 200
    # Issue #513: GET /metrics 无头应 401（require_api_key 不豁免 GET）
    assert (await async_client.get("/metrics")).status_code == 401
    # GET /metrics 带正确密钥应 200
    resp = await async_client.get(
        "/metrics", headers={"X-API-Key": "secret-key-123"}
    )
    assert resp.status_code == 200


@pytest.mark.asyncio
async def test_input_validation_empty_task(async_client):
    """POST /api/tasks 空 user_id 应被 Pydantic 拒绝（422）。"""
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
async def test_input_validation_empty_task_type(async_client):
    """POST /api/tasks 空 task_type 应被 Pydantic 拒绝（422）。"""
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
async def test_input_validation_qubit_count_exceeds_limit(async_client):
    """POST /api/tasks qubit_count 超过 287 上限应被拒绝（422）。"""
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
async def test_input_validation_oversized_user_id(async_client):
    """POST /api/tasks 超长 user_id（>128 字符）应被拒绝（422）。"""
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
    monkeypatch.setattr(app_module, "FRONTEND_DIST_PATH", "/nonexistent/dist")
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
    # 在 tmp_path/deliverable_models/ 下创建假的 ppo_best_model_16dim.zip
    # （与 app.py 中 _get_ppo_model 的优先路径一致）
    deliverable_dir = tmp_path / "deliverable_models"
    deliverable_dir.mkdir(parents=True)
    (deliverable_dir / "ppo_best_model_16dim.zip").write_bytes(b"fake")
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
        for s in ["PPO", "DQN", "FCFS"]:
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
    async def test_post_success_returns_task_id(self, async_client):
        """POST /api/tasks 成功应返回 task_id 且以 QTASK- 开头。"""
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
    async def test_post_invalid_priority_too_high(self, async_client):
        """priority=6 超过上限应返回 422。"""
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
    async def test_post_invalid_priority_too_low(self, async_client):
        """priority=0 低于下限应返回 422。"""
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
    async def test_post_invalid_qubit_count(self, async_client):
        """qubit_count=0 低于下限应返回 422。"""
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
    async def test_post_increases_task_count(self, async_client):
        """提交任务后任务总数应增加 1。"""
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
            patch.dict(os.environ, {"VIZ_API_KEY": "test-key"}),
            TestClient(app) as client,
        ):
            resp = client.get("/metrics", headers={"X-API-Key": "test-key"})
            assert resp.status_code == 200
            content_type = resp.headers.get("content-type", "")
            assert "text/plain" in content_type
            assert "version=" in content_type

    def test_prometheus_metrics_body_contains_process_info(self):
        """GET /metrics body 应包含 prometheus_client 默认进程指标。"""
        with (
            patch.object(app_module, "simulate_scheduler", _noop_simulate_scheduler),
            patch.dict(os.environ, {"VIZ_API_KEY": "test-key"}),
            TestClient(app) as client,
        ):
            body = client.get(
                "/metrics", headers={"X-API-Key": "test-key"}
            ).text
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
    async def test_known_strategy_switches(self, async_client):
        """已知策略应切换成功且更新 current_strategy。"""
        resp = await async_client.post("/api/strategy", params={"strategy": "PPO"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "PPO" in data["message"]
        status = await async_client.get("/api/status")
        assert status.json()["current_strategy"] == "PPO"

    @pytest.mark.asyncio
    async def test_unknown_strategy_fails(self, async_client):
        """未知策略应返回 success=False 且不修改 current_strategy。"""
        before = (await async_client.get("/api/status")).json()["current_strategy"]
        resp = await async_client.post("/api/strategy", params={"strategy": "NonExistent"})
        assert resp.status_code == 200
        assert resp.json()["success"] is False
        after = (await async_client.get("/api/status")).json()["current_strategy"]
        assert after == before

    @pytest.mark.asyncio
    async def test_auth_missing_returns_401(self, async_client, monkeypatch):
        """配置密钥后缺少 X-API-Key 应返回 401。"""
        monkeypatch.setenv("VIZ_API_KEY", "secret-key-123")
        resp = await async_client.post("/api/strategy", params={"strategy": "FCFS"})
        assert resp.status_code == 401


class TestUpdateEndpoint:
    """POST /api/update 端点测试。"""

    @pytest.mark.asyncio
    async def test_update_success(self, async_client):
        """合法 payload 应更新系统状态字段。"""
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
    async def test_qubit_utilization_out_of_bounds(self, async_client):
        """qubit_utilization>1.0 应返回 422。"""
        payload = {
            "qubit_utilization": 1.5,
            "queue_length": 1,
            "completed_tasks": 1,
            "average_wait_time": 1.0,
        }
        assert (await async_client.post("/api/update", json=payload)).status_code == 422

    @pytest.mark.asyncio
    async def test_negative_queue_length_out_of_bounds(self, async_client):
        """queue_length<0 应返回 422。"""
        payload = {
            "qubit_utilization": 0.5,
            "queue_length": -1,
            "completed_tasks": 1,
            "average_wait_time": 1.0,
        }
        assert (await async_client.post("/api/update", json=payload)).status_code == 422

    @pytest.mark.asyncio
    async def test_average_wait_time_out_of_bounds(self, async_client):
        """average_wait_time>86400 应返回 422。"""
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
        monkeypatch.setenv("VIZ_API_KEY", "secret-key-123")
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

    @staticmethod
    def _make_request(method: str = "POST") -> MagicMock:
        """构造 mock Request 对象，指定 HTTP 方法。"""
        req = MagicMock()
        req.method = method
        return req

    @pytest.mark.asyncio
    async def test_no_key_configured_allows(self, monkeypatch):
        """未配置 VIZ_API_KEY 时：GET 放行（返回 None），写操作（POST）必须抛 401。"""
        monkeypatch.delenv("VIZ_API_KEY", raising=False)
        # GET（只读监控端点）应放行
        assert await verify_api_key(self._make_request("GET"), x_api_key=None) is None
        # 写操作在零密钥下必须被拒绝，避免零认证放行漏洞
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(self._make_request("POST"), x_api_key=None)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_missing_header_rejected(self, monkeypatch):
        """配置密钥后缺失 X-API-Key 应抛 HTTPException 401。"""
        monkeypatch.setenv("VIZ_API_KEY", "secret-key-123")
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(self._make_request("POST"), x_api_key=None)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_wrong_key_rejected(self, monkeypatch):
        """配置密钥后不匹配的 X-API-Key 应抛 HTTPException 401。"""
        monkeypatch.setenv("VIZ_API_KEY", "secret-key-123")
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(self._make_request("POST"), x_api_key="wrong")
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_correct_key_allows(self, monkeypatch):
        """配置密钥后匹配的 X-API-Key 应放行。"""
        monkeypatch.setenv("VIZ_API_KEY", "secret-key-123")
        assert await verify_api_key(self._make_request("POST"), x_api_key="secret-key-123") is None

    @pytest.mark.asyncio
    async def test_empty_env_value_rejects_write(self, monkeypatch):
        """VIZ_API_KEY 为空字符串时等同未配置：GET 放行，写操作仍被拒绝。"""
        monkeypatch.setenv("VIZ_API_KEY", "")
        assert await verify_api_key(self._make_request("GET"), x_api_key=None) is None
        with pytest.raises(HTTPException) as exc:
            await verify_api_key(self._make_request("POST"), x_api_key=None)
        assert exc.value.status_code == 401

    @pytest.mark.asyncio
    async def test_get_request_bypasses_auth(self, monkeypatch):
        """GET 请求应跳过认证（只读端点不受密钥影响）。"""
        monkeypatch.setenv("VIZ_API_KEY", "secret-key-123")
        assert await verify_api_key(self._make_request("GET"), x_api_key=None) is None


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


class TestExplainabilityEndpoints:
    """GET /api/explainability 与 /api/explainability/summary 端点测试（Issue #73）。"""

    @pytest.mark.asyncio
    async def test_explainability_returns_feature_contributions(self, async_client, monkeypatch):
        """存在含 feature_contributions 的决策日志时，应返回对应记录。"""
        monkeypatch.delenv("VIZ_API_KEY", raising=False)
        app_module._decision_log.clear()
        app_module._decision_log.extend(
            [
                {
                    "step": 1,
                    "action": 1,
                    "action_label": "量子",
                    "feature_contributions": {"队列长度": 0.5, "平均优先级": 0.3},
                    "explanation_text": "第1步选择动作1",
                },
                {
                    "step": 2,
                    "action": 0,
                    "action_label": "经典",
                    "feature_contributions": {"队列长度": 0.2, "量子比特利用率": 0.8},
                    "explanation_text": "第2步选择动作0",
                },
            ]
        )
        resp = await async_client.get("/api/explainability")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert len(data["decisions"]) == 2
        assert data["decisions"][0]["step"] == 1
        assert "feature_contributions" in data["decisions"][0]
        assert "explanation_text" in data["decisions"][0]

    @pytest.mark.asyncio
    async def test_explainability_empty_log(self, async_client, monkeypatch):
        """空决策日志时应返回 count=0 的空列表。"""
        monkeypatch.delenv("VIZ_API_KEY", raising=False)
        app_module._decision_log.clear()
        resp = await async_client.get("/api/explainability")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["decisions"] == []

    @pytest.mark.asyncio
    async def test_explainability_limit_param(self, async_client, monkeypatch):
        """limit 参数应正确限制返回数量。"""
        monkeypatch.delenv("VIZ_API_KEY", raising=False)
        app_module._decision_log.clear()
        for i in range(10):
            app_module._decision_log.append(
                {
                    "step": i,
                    "action": 0,
                    "feature_contributions": {"队列长度": 0.1},
                    "explanation_text": f"第{i}步",
                }
            )
        resp = await async_client.get("/api/explainability?limit=3")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 3

    @pytest.mark.asyncio
    async def test_explainability_summary_returns_ranking(self, async_client, monkeypatch):
        """应返回全局特征重要性降序排名。"""
        monkeypatch.delenv("VIZ_API_KEY", raising=False)
        app_module._decision_log.clear()
        app_module._decision_log.extend(
            [
                {
                    "step": 1,
                    "action": 1,
                    "feature_contributions": {"队列长度": 0.6, "平均优先级": 0.4},
                },
                {
                    "step": 2,
                    "action": 0,
                    "feature_contributions": {"队列长度": 0.4, "平均优先级": 0.6},
                },
            ]
        )
        resp = await async_client.get("/api/explainability/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_decisions"] == 2
        assert len(data["feature_importance"]) == 2
        features = [item["feature"] for item in data["feature_importance"]]
        assert "队列长度" in features
        assert "平均优先级" in features
        # 验证 importance 字段存在且为数值
        for item in data["feature_importance"]:
            assert isinstance(item["importance"], (int, float))

    @pytest.mark.asyncio
    async def test_explainability_summary_empty_log(self, async_client, monkeypatch):
        """空决策日志时 summary 应返回空列表和 0。"""
        monkeypatch.delenv("VIZ_API_KEY", raising=False)
        app_module._decision_log.clear()
        resp = await async_client.get("/api/explainability/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_decisions"] == 0
        assert data["feature_importance"] == []


# ============================================================
# Issue #207 扩展覆盖：state.py 访问器 / routes.py 端点 / websocket_handler
# ============================================================


class TestStateAccessors:
    """state.py 线程安全访问器函数测试（Issue #207，覆盖 154-318 行）。"""

    @pytest.fixture(autouse=True)
    def restore_extended_state(self):
        """保存并恢复 _resource_history / _decision_log / _battle_state。"""
        saved_history = list(vis_state._resource_history)
        saved_log = list(vis_state._decision_log)
        yield
        vis_state._resource_history.clear()
        vis_state._resource_history.extend(saved_history)
        vis_state._decision_log.clear()
        vis_state._decision_log.extend(saved_log)

    def test_get_system_status_returns_copy(self):
        """get_system_status 应返回浅拷贝，修改拷贝不影响原状态。"""
        original = vis_state.get_system_status()
        original["qubit_utilization"] = 0.99
        assert vis_state.system_status["qubit_utilization"] != 0.99

    def test_get_system_status_ref_returns_reference(self):
        """get_system_status_ref 应返回原字典引用。"""
        assert vis_state.get_system_status_ref() is vis_state.system_status

    def test_update_system_status_merges_and_updates_timestamp(self):
        """update_system_status 应合并更新并刷新 last_update。"""
        old_update = vis_state.system_status["last_update"]
        vis_state.update_system_status({"completed_tasks": 999})
        assert vis_state.system_status["completed_tasks"] == 999
        assert vis_state.system_status["last_update"] != old_update

    def test_get_task_queue_returns_copy(self):
        """get_task_queue 应返回浅拷贝列表。"""
        q = vis_state.get_task_queue()
        original_len = len(vis_state.task_queue)
        q.append({"fake": "task"})
        assert len(vis_state.task_queue) == original_len

    def test_get_task_queue_ref_returns_reference(self):
        """get_task_queue_ref 应返回原列表引用。"""
        assert vis_state.get_task_queue_ref() is vis_state.task_queue

    def test_append_task_updates_queue_length(self):
        """append_task 应追加任务并更新 queue_length 为 pending 任务数。"""
        pending_before = len([t for t in vis_state.task_queue if t.get("status") == "pending"])
        vis_state.append_task({"task_id": "TEST-001", "status": "pending"})
        pending_after = len([t for t in vis_state.task_queue if t.get("status") == "pending"])
        assert pending_after == pending_before + 1
        assert vis_state.system_status["queue_length"] == pending_after

    def test_append_task_non_pending_does_not_increase_queue_length(self):
        """append_task 追加非 pending 任务不应增加 queue_length。"""
        pending_before = len([t for t in vis_state.task_queue if t.get("status") == "pending"])
        vis_state.append_task({"task_id": "TEST-002", "status": "running"})
        pending_after = len([t for t in vis_state.task_queue if t.get("status") == "pending"])
        assert pending_after == pending_before
        assert vis_state.system_status["queue_length"] == pending_after

    def test_get_pending_task_count(self):
        """get_pending_task_count 应返回 pending 状态的任务数。"""
        count = vis_state.get_pending_task_count()
        expected = len([t for t in vis_state.task_queue if t.get("status") == "pending"])
        assert count == expected

    def test_append_resource_history_trims_to_max(self):
        """append_resource_history 应自动裁剪到 MAX_RESOURCE_HISTORY。"""
        vis_state._resource_history.clear()
        for i in range(vis_state.MAX_RESOURCE_HISTORY + 10):
            vis_state.append_resource_history({"step": i})
        assert len(vis_state._resource_history) == vis_state.MAX_RESOURCE_HISTORY

    def test_get_resource_history_with_limit(self):
        """get_resource_history(limit) 应返回最近 limit 条数据。"""
        vis_state._resource_history.clear()
        for i in range(10):
            vis_state.append_resource_history({"step": i})
        recent = vis_state.get_resource_history(limit=3)
        assert len(recent) == 3
        assert recent[-1]["step"] == 9

    def test_get_resource_history_default_all(self):
        """get_resource_history() 默认返回全部数据。"""
        vis_state._resource_history.clear()
        for i in range(5):
            vis_state.append_resource_history({"step": i})
        all_data = vis_state.get_resource_history()
        assert len(all_data) == 5

    def test_append_decision_log_trims_to_max(self):
        """append_decision_log 应自动裁剪到 MAX_DECISION_LOG。"""
        vis_state._decision_log.clear()
        for i in range(vis_state.MAX_DECISION_LOG + 10):
            vis_state.append_decision_log({"step": i})
        assert len(vis_state._decision_log) == vis_state.MAX_DECISION_LOG

    def test_get_decision_log_with_limit(self):
        """get_decision_log(limit) 应返回最近 limit 条记录。"""
        vis_state._decision_log.clear()
        for i in range(10):
            vis_state.append_decision_log({"step": i})
        recent = vis_state.get_decision_log(limit=3)
        assert len(recent) == 3
        assert recent[-1]["step"] == 9

    def test_get_decision_log_default_all(self):
        """get_decision_log() 默认返回全部记录。"""
        vis_state._decision_log.clear()
        for i in range(5):
            vis_state.append_decision_log({"step": i})
        all_data = vis_state.get_decision_log()
        assert len(all_data) == 5

    def test_get_battle_state_returns_copy(self):
        """get_battle_state 应返回浅拷贝。"""
        battle = vis_state.get_battle_state()
        battle["running"] = True
        assert vis_state._battle_state["running"] is False

    def test_get_battle_state_ref_returns_reference(self):
        """get_battle_state_ref 应返回原字典引用。"""
        assert vis_state.get_battle_state_ref() is vis_state._battle_state

    def test_reset_battle_state_clears_all_fields(self):
        """reset_battle_state 应重置所有字段。"""
        vis_state._battle_state["running"] = True
        vis_state._battle_state["step"] = 10
        vis_state._battle_state["ppo_reward"] = 100.0
        vis_state._battle_state["fcfs_reward"] = 50.0
        vis_state._battle_state["ppo_history"] = [{"step": 1}]
        vis_state._battle_state["fcfs_history"] = [{"step": 1}]
        vis_state.reset_battle_state()
        assert vis_state._battle_state["running"] is False
        assert vis_state._battle_state["step"] == 0
        assert vis_state._battle_state["ppo_reward"] == 0.0
        assert vis_state._battle_state["fcfs_reward"] == 0.0
        assert vis_state._battle_state["ppo_history"] == []
        assert vis_state._battle_state["fcfs_history"] == []
        assert vis_state._battle_state["ppo_env"] is None
        assert vis_state._battle_state["fcfs_env"] is None
        assert vis_state._battle_state["ppo_obs"] is None
        assert vis_state._battle_state["fcfs_obs"] is None

    def test_get_connection_manager_returns_singleton(self):
        """get_connection_manager 应返回全局 manager 单例。"""
        assert vis_state.get_connection_manager() is vis_state.manager


class TestQuotaEndpoint:
    """/api/quota 端点测试（覆盖 437-444 行）。"""

    @pytest.mark.asyncio
    async def test_quota_no_tracker(self, async_client, monkeypatch):
        """无配额追踪器时应返回 available=False。"""
        monkeypatch.setattr(app_module, "_get_quota_tracker", lambda: None)
        resp = await async_client.get("/api/quota")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False
        assert "message" in data

    @pytest.mark.asyncio
    async def test_quota_with_tracker(self, async_client, monkeypatch):
        """有配额追踪器时应返回 available=True 及状态信息。"""
        mock_tracker = MagicMock()
        mock_tracker.status.return_value = {"total": 100, "used": 30, "remaining": 70}
        monkeypatch.setattr(app_module, "_get_quota_tracker", lambda: mock_tracker)
        resp = await async_client.get("/api/quota")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is True
        assert data["total"] == 100

    @pytest.mark.asyncio
    async def test_quota_tracker_exception(self, async_client, monkeypatch):
        """配额追踪器抛异常时应返回 available=False。"""
        mock_tracker = MagicMock()
        mock_tracker.status.side_effect = Exception("fail")
        monkeypatch.setattr(app_module, "_get_quota_tracker", lambda: mock_tracker)
        resp = await async_client.get("/api/quota")
        assert resp.status_code == 200
        data = resp.json()
        assert data["available"] is False


class TestResourceHistoryEndpoint:
    """/api/resource-history 端点测试（覆盖 452-463 行）。"""

    @pytest.fixture(autouse=True)
    def restore_history(self):
        """保存并恢复 _resource_history。"""
        saved = list(vis_state._resource_history)
        yield
        vis_state._resource_history.clear()
        vis_state._resource_history.extend(saved)

    @pytest.mark.asyncio
    async def test_returns_history_list(self, async_client):
        """应返回资源历史列表。"""
        vis_state._resource_history.clear()
        vis_state._resource_history.extend([{"step": 1}, {"step": 2}])
        resp = await async_client.get("/api/resource-history")
        assert resp.status_code == 200
        data = resp.json()
        assert "history" in data
        assert len(data["history"]) == 2

    @pytest.mark.asyncio
    async def test_empty_history(self, async_client):
        """空历史时应返回空列表。"""
        vis_state._resource_history.clear()
        resp = await async_client.get("/api/resource-history")
        assert resp.status_code == 200
        assert resp.json()["history"] == []


class TestDecisionLogEndpoint:
    """/api/decision-log 端点测试（覆盖 466-476 行）。"""

    @pytest.fixture(autouse=True)
    def restore_log(self):
        """保存并恢复 _decision_log。"""
        saved = list(vis_state._decision_log)
        yield
        vis_state._decision_log.clear()
        vis_state._decision_log.extend(saved)

    @pytest.mark.asyncio
    async def test_returns_decisions(self, async_client):
        """应返回决策日志列表。"""
        vis_state._decision_log.clear()
        vis_state._decision_log.extend([{"step": 1, "action": 0}, {"step": 2, "action": 1}])
        resp = await async_client.get("/api/decision-log")
        assert resp.status_code == 200
        data = resp.json()
        assert "decisions" in data
        assert len(data["decisions"]) == 2

    @pytest.mark.asyncio
    async def test_empty_decisions(self, async_client):
        """空日志时应返回空列表。"""
        vis_state._decision_log.clear()
        resp = await async_client.get("/api/decision-log")
        assert resp.status_code == 200
        assert resp.json()["decisions"] == []


class TestMachinesComparisonEndpoint:
    """/api/machines-comparison 端点测试（覆盖 484-498 行）。"""

    @pytest.mark.asyncio
    async def test_no_machines(self, async_client):
        """无真机时应返回空列表。"""
        vis_state.system_status["real_machines"] = []
        resp = await async_client.get("/api/machines-comparison")
        assert resp.status_code == 200
        data = resp.json()
        assert data["machines"] == []

    @pytest.mark.asyncio
    async def test_with_machines(self, async_client):
        """有真机时应返回机器对比数据。"""
        vis_state.system_status["real_machines"] = [
            {
                "name": "tianyan_s",
                "total_qubits": 20,
                "available_ratio": 0.8,
                "fidelity": 0.99,
                "queue_depth": 3,
                "status": "running",
                "single_gate_fidelity": 0.999,
                "two_gate_fidelity": 0.98,
            }
        ]
        resp = await async_client.get("/api/machines-comparison")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["machines"]) == 1
        m = data["machines"][0]
        assert m["name"] == "tianyan_s"
        assert m["total_qubits"] == 20
        assert m["fidelity"] == 0.99
        assert m["single_gate_fidelity"] == 0.999
        assert m["two_gate_fidelity"] == 0.98

    @pytest.mark.asyncio
    async def test_machines_with_missing_fields(self, async_client):
        """机器字段缺失时应使用默认值。"""
        vis_state.system_status["real_machines"] = [{}]
        resp = await async_client.get("/api/machines-comparison")
        assert resp.status_code == 200
        m = resp.json()["machines"][0]
        assert m["name"] == "unknown"
        assert m["total_qubits"] == 0
        assert m["status"] == "unknown"


class TestTenantsEndpoint:
    """/api/tenants 端点测试（覆盖 510-517 行）。"""

    @pytest.mark.asyncio
    async def test_returns_tenants_or_empty(self, async_client):
        """应返回租户列表或空列表（配置不存在时）。"""
        resp = await async_client.get("/api/tenants")
        assert resp.status_code == 200
        data = resp.json()
        assert "tenants" in data
        assert isinstance(data["tenants"], list)

    @pytest.mark.asyncio
    async def test_tenants_exception_returns_empty(self, async_client, monkeypatch):
        """TenantQuotaManager.from_config 抛异常时应返回空列表。"""
        monkeypatch.setattr(
            "src.scheduler.tenant.TenantQuotaManager.from_config",
            MagicMock(side_effect=Exception("config fail")),
        )
        resp = await async_client.get("/api/tenants")
        assert resp.status_code == 200
        assert resp.json()["tenants"] == []


class TestExplainabilityLatestEndpoint:
    """/api/explainability/latest 端点测试（覆盖 590-607 行）。"""

    @pytest.fixture(autouse=True)
    def restore_log(self):
        """保存并恢复 _decision_log。"""
        saved = list(vis_state._decision_log)
        yield
        vis_state._decision_log.clear()
        vis_state._decision_log.extend(saved)

    @pytest.mark.asyncio
    async def test_with_data(self, async_client):
        """有决策日志时应返回最新一条含 feature_contributions 的记录。"""
        vis_state._decision_log.clear()
        vis_state._decision_log.extend(
            [
                {"step": 1, "action": 0, "feature_contributions": {"a": 0.1}},
                {"step": 2, "action": 1, "feature_contributions": {"b": 0.2}},
            ]
        )
        resp = await async_client.get("/api/explainability/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["empty"] is False
        assert data["latest"]["step"] == 2

    @pytest.mark.asyncio
    async def test_empty(self, async_client):
        """空日志时应返回 empty=True。"""
        vis_state._decision_log.clear()
        resp = await async_client.get("/api/explainability/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["empty"] is True
        assert data["latest"] is None

    @pytest.mark.asyncio
    async def test_skips_entries_without_feature_contributions(self, async_client):
        """无 feature_contributions 的记录应被跳过。"""
        vis_state._decision_log.clear()
        vis_state._decision_log.extend(
            [
                {"step": 1, "action": 0},
                {"step": 2, "action": 1, "feature_contributions": {"a": 0.1}},
            ]
        )
        resp = await async_client.get("/api/explainability/latest")
        assert resp.status_code == 200
        data = resp.json()
        assert data["empty"] is False
        assert data["latest"]["step"] == 2


class TestBattleEndpoints:
    """/api/battle/* 对战端点测试（覆盖 619-730 行）。"""

    @pytest.fixture(autouse=True)
    def restore_battle(self):
        """每个测试后重置对战状态。"""
        yield
        vis_state.reset_battle_state()

    @pytest.mark.asyncio
    async def test_battle_start_success(self, async_client, monkeypatch):
        """battle/start 应成功启动对战。"""
        mock_env = MagicMock()
        mock_obs = MagicMock()
        mock_obs.tolist.return_value = [0.1, 0.2, 0.3, 0.4, 0.5]
        mock_env.reset.return_value = (mock_obs, {})
        monkeypatch.setattr("src.scheduler.env.QuantumSchedulingEnv", lambda **kw: mock_env)
        resp = await async_client.post("/api/battle/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["step"] == 0
        assert "ppo_obs" in data
        assert "fcfs_obs" in data

    @pytest.mark.asyncio
    async def test_battle_start_failure(self, async_client, monkeypatch):
        """battle/start 环境创建失败时应返回 success=False。"""

        def _raise(**kw):
            raise RuntimeError("env fail")

        monkeypatch.setattr("src.scheduler.env.QuantumSchedulingEnv", _raise)
        resp = await async_client.post("/api/battle/start")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is False
        assert "error" in data

    @pytest.mark.asyncio
    async def test_battle_step_not_running(self, async_client):
        """battle/step 未启动对战时应返回 error。"""
        vis_state.reset_battle_state()
        resp = await async_client.post("/api/battle/step")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data
        assert "未启动" in data["error"]

    @pytest.mark.asyncio
    async def test_battle_step_with_model(self, async_client, monkeypatch):
        """battle/step 有 PPO 模型时应使用模型预测动作。"""
        mock_env = MagicMock()
        mock_env.step.return_value = ([0.5, 0.3, 0.2], 1.5, False, False, {})

        battle = vis_state.get_battle_state_ref()
        battle["running"] = True
        battle["ppo_env"] = mock_env
        battle["fcfs_env"] = mock_env
        battle["ppo_obs"] = [0.5, 0.3, 0.2]
        battle["fcfs_obs"] = [0.5, 0.3, 0.2]

        mock_model = MagicMock()
        mock_model.predict.return_value = (1, None)
        monkeypatch.setattr(app_module, "_get_ppo_model", lambda: mock_model)

        resp = await async_client.post("/api/battle/step")
        assert resp.status_code == 200
        data = resp.json()
        assert data["step"] == 1
        assert data["ppo"]["action"] == 1
        assert data["fcfs"]["action"] == 0
        assert data["ppo_total"] == 1.5
        assert "gap" in data

    @pytest.mark.asyncio
    async def test_battle_step_no_model(self, async_client, monkeypatch):
        """battle/step 无 PPO 模型时应使用默认动作0。"""
        mock_env = MagicMock()
        mock_env.step.return_value = ([0.5, 0.3, 0.2], 1.0, False, False, {})

        battle = vis_state.get_battle_state_ref()
        battle["running"] = True
        battle["ppo_env"] = mock_env
        battle["fcfs_env"] = mock_env
        battle["ppo_obs"] = [0.5, 0.3, 0.2]
        battle["fcfs_obs"] = [0.5, 0.3, 0.2]

        monkeypatch.setattr(app_module, "_get_ppo_model", lambda: None)

        resp = await async_client.post("/api/battle/step")
        assert resp.status_code == 200
        data = resp.json()
        assert data["ppo"]["action"] == 0
        assert data["ppo"]["reward"] == 0.0

    @pytest.mark.asyncio
    async def test_battle_step_with_done(self, async_client, monkeypatch):
        """battle/step 环境结束时(terminated=True)应重置 obs。"""
        mock_env = MagicMock()
        mock_env.step.return_value = ([0.5], 1.0, True, False, {})
        mock_env.reset.return_value = ([0.1], {})

        battle = vis_state.get_battle_state_ref()
        battle["running"] = True
        battle["ppo_env"] = mock_env
        battle["fcfs_env"] = mock_env
        battle["ppo_obs"] = [0.5]
        battle["fcfs_obs"] = [0.5]

        monkeypatch.setattr(app_module, "_get_ppo_model", lambda: None)

        resp = await async_client.post("/api/battle/step")
        assert resp.status_code == 200
        mock_env.reset.assert_called()

    @pytest.mark.asyncio
    async def test_battle_step_exception(self, async_client, monkeypatch):
        """battle/step 抛异常时应返回 error。"""
        mock_env = MagicMock()
        mock_env.step.side_effect = RuntimeError("step fail")

        battle = vis_state.get_battle_state_ref()
        battle["running"] = True
        battle["ppo_env"] = mock_env
        battle["fcfs_env"] = mock_env
        battle["ppo_obs"] = [0.5]
        battle["fcfs_obs"] = [0.5]

        monkeypatch.setattr(app_module, "_get_ppo_model", lambda: None)

        resp = await async_client.post("/api/battle/step")
        assert resp.status_code == 200
        data = resp.json()
        assert "error" in data

    @pytest.mark.asyncio
    async def test_battle_status(self, async_client):
        """battle/status 应返回当前对战状态。"""
        battle = vis_state.get_battle_state_ref()
        battle["running"] = True
        battle["step"] = 5
        battle["ppo_reward"] = 100.0
        battle["fcfs_reward"] = 50.0
        battle["ppo_history"] = [
            {"step": 1, "reward": 10, "cumulative": 10, "action": 1, "util": 0.5}
        ]
        battle["fcfs_history"] = [
            {"step": 1, "reward": 5, "cumulative": 5, "action": 0, "util": 0.5}
        ]

        resp = await async_client.get("/api/battle/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is True
        assert data["step"] == 5
        assert data["ppo_total"] == 100.0
        assert data["fcfs_total"] == 50.0
        assert data["gap"] == 50.0
        assert len(data["ppo_history"]) == 1
        assert len(data["fcfs_history"]) == 1

    @pytest.mark.asyncio
    async def test_battle_reset(self, async_client):
        """battle/reset 应重置对战状态。"""
        battle = vis_state.get_battle_state_ref()
        battle["running"] = True
        battle["step"] = 10
        battle["ppo_reward"] = 200.0

        resp = await async_client.post("/api/battle/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert vis_state._battle_state["running"] is False
        assert vis_state._battle_state["step"] == 0
        assert vis_state._battle_state["ppo_reward"] == 0.0

    @pytest.mark.asyncio
    async def test_battle_step_concurrent_thread_safety(self, async_client, monkeypatch):
        """Issue #388: 多线程并发调用 battle_step 不应导致数据损坏。

        使用 state_lock 后，所有状态读写串行化，step 数应等于调用次数。
        """
        import asyncio
        import threading

        # 准备 mock env：每次 step 返回固定 reward
        call_count = {"n": 0}

        def _make_step_return(*args, **kwargs):
            call_count["n"] += 1
            return ([0.5, 0.3, 0.2], 1.0, False, False, {})

        mock_env = MagicMock()
        mock_env.step.side_effect = _make_step_return

        battle = vis_state.get_battle_state_ref()
        battle["running"] = True
        battle["ppo_env"] = mock_env
        battle["fcfs_env"] = mock_env
        battle["ppo_obs"] = [0.5, 0.3, 0.2]
        battle["fcfs_obs"] = [0.5, 0.3, 0.2]

        monkeypatch.setattr(app_module, "_get_ppo_model", lambda: None)

        # 并发发起 5 次 step 请求
        tasks = [asyncio.create_task(async_client.post("/api/battle/step")) for _ in range(5)]
        responses = await asyncio.gather(*tasks)

        # 验证所有请求成功
        for resp in responses:
            assert resp.status_code == 200

        # 验证 step 数等于调用次数（无丢失更新）
        battle_after = vis_state.get_battle_state_ref()
        assert battle_after["step"] == 5, (
            f"并发调用后 step={battle_after['step']}，期望 5（可能存在竞态条件）"
        )
        # 验证 history 长度正确
        assert len(battle_after["ppo_history"]) == 5
        assert len(battle_after["fcfs_history"]) == 5


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


# ============================================================
# Issue #381: fallback_template.py 存储型 XSS 漏洞测试
# ============================================================


class TestFallbackTemplateXSS:
    """验证 fallback_template.py 中所有 innerHTML 拼接处已对 API 数据进行 HTML 转义。"""

    def test_escapehtml_function_exists(self) -> None:
        """fallback_template 应包含 escapeHtml 函数。"""
        from src.visualization import fallback_template

        assert "function escapeHtml" in fallback_template.HTML_TEMPLATE
        assert "function escapeHtml" in fallback_template.HTML_TEMPLATE

    def test_render_tasks_uses_escapehtml(self) -> None:
        """renderTasks 中所有 API 字段应使用 escapeHtml。"""
        from src.visualization import fallback_template

        # 提取 renderTasks 函数体
        start = fallback_template.HTML_TEMPLATE.find("function renderTasks")
        end = fallback_template.HTML_TEMPLATE.find("function renderStrategies")
        body = fallback_template.HTML_TEMPLATE[start:end]
        assert body, "renderTasks 函数未找到"

        # 验证所有 innerHTML 拼接的字段都被 escapeHtml 包裹
        # task_id, task_type, qubit_count, priority, status 均需转义
        assert "escapeHtml((t.task_id" in body
        assert "escapeHtml(t.task_type" in body
        assert "escapeHtml(t.qubit_count" in body
        assert "escapeHtml(t.priority" in body
        assert "escapeHtml(statusText(t.status" in body

    def test_render_decisions_uses_escapehtml(self) -> None:
        """renderDecisions 中所有 API 字段应使用 escapeHtml。"""
        from src.visualization import fallback_template

        start = fallback_template.HTML_TEMPLATE.find("function renderDecisions")
        end = fallback_template.HTML_TEMPLATE.find("function updateDecisionPie")
        body = fallback_template.HTML_TEMPLATE[start:end]
        assert body, "renderDecisions 函数未找到"

        # task_id, source, actLabel 均需转义
        assert "escapeHtml(d.step" in body
        assert "escapeHtml(actLabel)" in body
        assert "escapeHtml((d.task_id" in body
        assert "escapeHtml(d.source)" in body

    def test_load_real_submissions_uses_escapehtml(self) -> None:
        """loadRealSubmissions 中所有 API 字段应使用 escapeHtml。"""
        from src.visualization import fallback_template

        start = fallback_template.HTML_TEMPLATE.find("function loadRealSubmissions")
        end = fallback_template.HTML_TEMPLATE.find("function loadRealMachines")
        body = fallback_template.HTML_TEMPLATE[start:end]
        assert body, "loadRealSubmissions 函数未找到"

        # taskId, machine, time 均需转义
        assert "escapeHtml(taskId" in body
        assert "escapeHtml(machine)" in body
        assert "escapeHtml(time)" in body

    def test_escapehtml_escapes_script_tag(self) -> None:
        """escapeHtml 应正确转义 <script> 标签。"""
        from src.visualization import fallback_template

        # 验证 escapeHtml 实现包含必要的字符替换
        template = fallback_template.HTML_TEMPLATE
        assert "&amp;" in template
        assert "&lt;" in template
        assert "&gt;" in template
        assert "&quot;" in template
        assert "&#39;" in template
        assert "&#x2F;" in template


# ============================================================
# 安全功能测试（Issue #514 / #515 / #516 / #517）
# ============================================================


class TestRateLimiter:
    """速率限制器单元测试（Issue #517）。"""

    def test_check_allows_under_limit(self):
        """未超限时 check 应返回 (True, limit, remaining)。"""
        from src.visualization.security import RateLimiter

        limiter = RateLimiter(max_requests=5, window_seconds=60)
        allowed, limit, remaining = limiter.check("192.168.1.1")
        assert allowed is True
        assert limit == 5
        assert remaining == 4

    def test_check_blocks_over_limit(self):
        """超出限制后 check 应返回 (False, limit, 0)。"""
        from src.visualization.security import RateLimiter

        limiter = RateLimiter(max_requests=2, window_seconds=60)
        limiter.check("10.0.0.1")
        limiter.check("10.0.0.1")
        allowed, limit, remaining = limiter.check("10.0.0.1")
        assert allowed is False
        assert limit == 2
        assert remaining == 0

    def test_check_isolates_different_keys(self):
        """不同 key 的请求计数应独立。"""
        from src.visualization.security import RateLimiter

        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.check("ip-a")
        # 不同 IP 不受影响
        allowed, _, _ = limiter.check("ip-b")
        assert allowed is True

    def test_reset_clears_all_state(self):
        """reset 应清空所有速率限制状态。"""
        from src.visualization.security import RateLimiter

        limiter = RateLimiter(max_requests=1, window_seconds=60)
        limiter.check("ip-x")
        # 超限
        assert limiter.check("ip-x")[0] is False
        # 重置后应重新允许
        limiter.reset()
        assert limiter.check("ip-x")[0] is True

    @pytest.mark.asyncio
    async def test_rate_limit_headers_present(self, async_client):
        """POST 请求响应应包含速率限制头。"""
        resp = await async_client.post("/api/strategy", params={"strategy": "PPO"})
        assert "x-ratelimit-limit" in {k.lower() for k in resp.headers}
        assert "x-ratelimit-remaining" in {k.lower() for k in resp.headers}

    @pytest.mark.asyncio
    async def test_rate_limit_returns_429_when_exceeded(self, async_client, monkeypatch):
        """超出速率限制后应返回 429。"""
        # 使用低限流阈值的小型限流器替换全局实例
        from src.visualization.security import RateLimiter

        small_limiter = RateLimiter(max_requests=1, window_seconds=60)
        monkeypatch.setattr("src.visualization.routes.rate_limiter", small_limiter)
        # 第一次请求成功
        resp1 = await async_client.post("/api/strategy", params={"strategy": "PPO"})
        assert resp1.status_code == 200
        # 第二次请求应被限流
        resp2 = await async_client.post("/api/strategy", params={"strategy": "FCFS"})
        assert resp2.status_code == 429
        assert resp2.headers.get("retry-after") == "60"


class TestCircuitValidation:
    """量子电路校验测试（Issue #515）。"""

    def test_valid_qcis_circuit_passes(self):
        """合法 QCIS 电路应通过校验。"""
        from src.visualization.security import validate_quantum_circuit

        circuit = "Q0 Q1\nQ2 Q3\n"
        result = validate_quantum_circuit(circuit, "qcis")
        assert result["gate_count"] == 2
        assert result["qubit_count"] == 4

    def test_valid_openqasm_circuit_passes(self):
        """合法 OpenQASM 电路应通过校验。"""
        from src.visualization.security import validate_quantum_circuit

        circuit = "OPENQASM 2.0;\nqreg q[4];\nH q[0];\nCX q[0], q[1];\n"
        result = validate_quantum_circuit(circuit, "openqasm")
        assert result["qubit_count"] == 4
        assert result["gate_count"] >= 3

    def test_empty_circuit_rejected(self):
        """空电路应抛出 400。"""
        from src.visualization.security import validate_quantum_circuit

        with pytest.raises(HTTPException) as exc_info:
            validate_quantum_circuit("", "qcis")
        assert exc_info.value.status_code == 400

    def test_whitespace_only_circuit_rejected(self):
        """仅空白字符的电路应抛出 400。"""
        from src.visualization.security import validate_quantum_circuit

        with pytest.raises(HTTPException) as exc_info:
            validate_quantum_circuit("   \n  \t  ", "qcis")
        assert exc_info.value.status_code == 400

    def test_invalid_chars_rejected(self):
        """包含非 ASCII 可打印字符的电路应抛出 400。"""
        from src.visualization.security import validate_quantum_circuit

        with pytest.raises(HTTPException) as exc_info:
            validate_quantum_circuit("Q0 Q1\n中文\n", "qcis")
        assert exc_info.value.status_code == 400
        assert "非法字符" in exc_info.value.detail

    def test_unsupported_format_rejected(self):
        """不支持的格式应抛出 400。"""
        from src.visualization.security import validate_quantum_circuit

        with pytest.raises(HTTPException) as exc_info:
            validate_quantum_circuit("Q0", "json")
        assert exc_info.value.status_code == 400
        assert "不支持" in exc_info.value.detail

    def test_too_many_qubits_rejected(self):
        """量子比特数超过上限应抛出 400。"""
        from src.visualization import security

        # 构造超过 105 比特的 QCIS 电路
        circuit = "\n".join(f"Q{i} Q{i + 1}" for i in range(0, 106, 2))
        with pytest.raises(HTTPException) as exc_info:
            security.validate_quantum_circuit(circuit, "qcis")
        assert exc_info.value.status_code == 400
        assert "量子比特数" in exc_info.value.detail

    @pytest.mark.asyncio
    async def test_circuit_submit_endpoint_valid(self, async_client):
        """POST /api/circuit/submit 合法电路应返回 task_id。"""
        payload = {
            "circuit": "Q0 Q1\nQ1 Q2\n",
            "format": "qcis",
            "shots": 1024,
            "task_name": "test_circuit",
        }
        resp = await async_client.post("/api/circuit/submit", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["task_id"].startswith("QCIR-")
        assert data["gate_count"] == 2
        assert data["qubit_count"] == 3

    @pytest.mark.asyncio
    async def test_circuit_submit_endpoint_empty_rejected(self, async_client):
        """POST /api/circuit/submit 仅空白字符的电路应返回 400。

        注意：空字符串会被 Pydantic 的 min_length=1 拦截（422），
        因此使用仅空白字符的内容来测试路由层 validate_quantum_circuit 的非空校验。
        """
        payload = {
            "circuit": "   \n  \t  ",
            "format": "qcis",
        }
        resp = await async_client.post("/api/circuit/submit", json=payload)
        assert resp.status_code == 400


class TestErrorSanitization:
    """错误消息净化测试（Issue #516）。"""

    def test_sanitize_removes_file_paths(self):
        """sanitize_error_message 应移除文件路径。"""
        from src.visualization.security import sanitize_error_message

        msg = "Error in /home/user/project/src/main.py at line 42"
        sanitized = sanitize_error_message(msg)
        assert "/home/user/project/src/main.py" not in sanitized
        assert "main.py" not in sanitized

    def test_sanitize_removes_windows_paths(self):
        """sanitize_error_message 应移除 Windows 文件路径。"""
        from src.visualization.security import sanitize_error_message

        msg = "Error in C:\\Users\\admin\\app\\src\\config.py"
        sanitized = sanitize_error_message(msg)
        assert "C:\\Users" not in sanitized

    def test_sanitize_removes_variable_names(self):
        """sanitize_error_message 应移除单引号包裹的变量名。"""
        from src.visualization.security import sanitize_error_message

        msg = "KeyError: 'some_internal_variable'"
        sanitized = sanitize_error_message(msg)
        assert "some_internal_variable" not in sanitized

    def test_sanitize_removes_exception_types(self):
        """sanitize_error_message 应移除异常类型名。"""
        from src.visualization.security import sanitize_error_message

        msg = "RuntimeError: something went wrong"
        sanitized = sanitize_error_message(msg)
        assert "RuntimeError" not in sanitized

    @pytest.mark.asyncio
    async def test_global_exception_handler_returns_generic_message(self):
        """全局异常处理器应返回通用错误消息和 correlation_id，不泄露内部信息。

        直接调用处理器函数进行测试，避免 ASGITransport 在
        raise_app_exceptions=True 下重新抛出异常影响断言。
        """
        from starlette.requests import Request

        from src.visualization.app import global_exception_handler

        # 构造一个模拟 Request 对象
        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/status",
            "headers": [],
            "query_string": b"",
            "client": ("127.0.0.1", 8000),
            "scheme": "http",
            "server": ("localhost", 8000),
        }
        request = Request(scope)
        exc = RuntimeError("secret internal path /src/foo.py leaked")

        response = await global_exception_handler(request, exc)
        assert response.status_code == 500
        # 解析 JSONResponse 的 body
        import json as _json

        data = _json.loads(response.body)
        assert data["detail"] == "Internal server error"
        assert "correlation_id" in data
        # 不应泄露内部路径或异常信息
        body_text = response.body.decode("utf-8")
        assert "secret" not in body_text
        assert "foo.py" not in body_text


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
