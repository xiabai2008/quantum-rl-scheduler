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
