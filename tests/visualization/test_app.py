"""可视化 app.py 辅助函数测试（拆分自 test_visualization.py，Issue #730）。"""

import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import FastAPI, HTTPException
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
# Issue #885：优雅关闭 + CORS 配置 + 请求体大小限制
# ============================================================


# ---- 配置读取：_get_cors_origins ----


def test_get_cors_origins_default(monkeypatch):
    """未设置 CORS_ORIGINS 时应返回默认来源列表。"""
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    result = app_module._get_cors_origins()
    assert "http://localhost:3000" in result
    assert "http://localhost:8000" in result


def test_get_cors_origins_from_env(monkeypatch):
    """CORS_ORIGINS 环境变量应解析为来源列表（去除首尾空白）。"""
    monkeypatch.setenv("CORS_ORIGINS", "https://a.example.com, https://b.example.com")
    result = app_module._get_cors_origins()
    assert result == ["https://a.example.com", "https://b.example.com"]


def test_get_cors_origins_strips_blanks(monkeypatch):
    """空白来源项应被过滤。"""
    monkeypatch.setenv("CORS_ORIGINS", "http://x.com,, ,http://y.com")
    result = app_module._get_cors_origins()
    assert result == ["http://x.com", "http://y.com"]


# ---- 配置读取：_get_max_request_body_size ----


def test_get_max_request_body_size_default(monkeypatch):
    """未设置 MAX_REQUEST_BODY_SIZE 时应返回默认 10 MB。"""
    monkeypatch.delenv("MAX_REQUEST_BODY_SIZE", raising=False)
    assert app_module._get_max_request_body_size() == 10 * 1024 * 1024


def test_get_max_request_body_size_from_env(monkeypatch):
    """合法整数值应被使用。"""
    monkeypatch.setenv("MAX_REQUEST_BODY_SIZE", "2048")
    assert app_module._get_max_request_body_size() == 2048


def test_get_max_request_body_size_invalid(monkeypatch):
    """非法值应回退到默认 10 MB。"""
    monkeypatch.setenv("MAX_REQUEST_BODY_SIZE", "not-a-number")
    assert app_module._get_max_request_body_size() == 10 * 1024 * 1024


def test_get_max_request_body_size_non_positive(monkeypatch):
    """非正数应回退到默认 10 MB。"""
    monkeypatch.setenv("MAX_REQUEST_BODY_SIZE", "0")
    assert app_module._get_max_request_body_size() == 10 * 1024 * 1024


# ---- ConnectionManager.close_all / drain_connections ----


@pytest.mark.asyncio
async def test_connection_manager_drain_connections_clears_list():
    """drain_connections 应取出全部连接并清空活跃列表。"""
    cm = ConnectionManager()
    ws1 = MagicMock()
    ws2 = MagicMock()
    cm.active_connections = [ws1, ws2]
    drained = cm.drain_connections()
    assert drained == [ws1, ws2]
    assert cm.active_connections == []


@pytest.mark.asyncio
async def test_connection_manager_close_all_closes_connections():
    """close_all 应关闭所有活跃连接并清空列表。"""
    cm = ConnectionManager()
    ws1 = MagicMock()
    ws1.close = AsyncMock()
    ws2 = MagicMock()
    ws2.close = AsyncMock()
    cm.active_connections = [ws1, ws2]
    await cm.close_all()
    ws1.close.assert_awaited_once()
    ws2.close.assert_awaited_once()
    assert cm.active_connections == []


@pytest.mark.asyncio
async def test_connection_manager_close_all_swallows_errors():
    """单个连接 close 失败不应影响其余连接关闭。"""
    cm = ConnectionManager()
    ws1 = MagicMock()
    ws1.close = AsyncMock(side_effect=RuntimeError("boom"))
    ws2 = MagicMock()
    ws2.close = AsyncMock()
    cm.active_connections = [ws1, ws2]
    await cm.close_all()
    ws2.close.assert_awaited_once()
    assert cm.active_connections == []


# ---- 请求体大小限制中间件 ----


def _build_small_app(max_body_size: int) -> FastAPI:
    """构建带请求体大小限制中间件的极简 FastAPI 应用。"""
    small_app = FastAPI()

    @small_app.post("/echo")
    async def _echo() -> dict[str, str]:
        return {"ok": "true"}

    small_app.add_middleware(
        app_module.RequestSizeLimitMiddleware,
        max_body_size=max_body_size,
    )
    return small_app


def test_request_size_limit_rejects_oversized():
    """Content-Length 超过限制时应返回 413。"""
    client = TestClient(_build_small_app(max_body_size=100))
    resp = client.post("/echo", content="x" * 200)
    assert resp.status_code == 413
    assert "超过限制" in resp.json()["detail"]


def test_request_size_limit_allows_within_limit():
    """Content-Length 在限制内时应正常处理（200）。"""
    client = TestClient(_build_small_app(max_body_size=1000))
    resp = client.post("/echo", content="x" * 50)
    assert resp.status_code == 200
    assert resp.json() == {"ok": "true"}


def test_request_size_limit_allows_without_content_length():
    """无 Content-Length 头的请求应放行（如 GET）。"""
    small_app = _build_small_app(max_body_size=100)

    @small_app.get("/ping")
    async def _ping() -> dict[str, str]:
        return {"pong": "true"}

    client = TestClient(small_app)
    resp = client.get("/ping")
    assert resp.status_code == 200


# ---- CORS 集成（基于真实 app 实例）----


@pytest.fixture
def mock_scheduler(monkeypatch):
    """Mock simulate_scheduler 避免 TestClient 启动时实际运行仿真循环。"""

    async def _fake_simulate() -> None:
        # 阻塞直到被取消（模拟后台任务）
        await asyncio.Event().wait()

    monkeypatch.setattr(app_module, "simulate_scheduler", _fake_simulate)


def test_cors_header_for_allowed_origin(mock_scheduler):
    """允许的 Origin 应在响应头返回 Access-Control-Allow-Origin。"""
    with TestClient(app) as client:
        resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"


def test_cors_no_header_for_disallowed_origin(mock_scheduler):
    """不允许的 Origin 不应返回 Access-Control-Allow-Origin。"""
    with TestClient(app) as client:
        resp = client.get("/health", headers={"Origin": "http://evil.example.com"})
    assert resp.status_code == 200
    assert "access-control-allow-origin" not in resp.headers


def test_cors_preflight_options(mock_scheduler):
    """预检 OPTIONS 请求应返回 CORS 预检响应。"""
    with TestClient(app) as client:
        resp = client.options(
            "/api/tasks",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
    assert resp.status_code == 200
    assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"
    allow_methods = resp.headers.get("access-control-allow-methods", "")
    assert "POST" in allow_methods


# ---- 优雅关闭：lifespan ----


def test_lifespan_graceful_shutdown_calls_close_all(mock_scheduler, monkeypatch):
    """应用关闭时应调用 manager.close_all 优雅关闭 WebSocket 连接。"""
    close_called = {"called": False}
    original_close = app_module.manager.close_all

    async def _tracking_close() -> None:
        close_called["called"] = True
        await original_close()

    monkeypatch.setattr(app_module.manager, "close_all", _tracking_close)

    with TestClient(app):
        pass  # 进入触发 startup，退出触发 shutdown

    assert close_called["called"] is True


def test_lifespan_cancels_background_task(mock_scheduler, monkeypatch):
    """优雅关闭应取消后台仿真任务。"""
    cancel_seen = {"called": False}

    async def _trackable_simulate() -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancel_seen["called"] = True
            raise

    # 覆盖 mock_scheduler 注入的桩，使用可追踪取消事件的版本
    monkeypatch.setattr(app_module, "simulate_scheduler", _trackable_simulate)

    with TestClient(app):
        pass  # 进入触发 startup，退出触发 shutdown

    assert cancel_seen["called"] is True
