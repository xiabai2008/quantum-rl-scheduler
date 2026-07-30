"""可视化 HTTP API 路由与端点测试（拆分自 test_visualization.py，Issue #730）。"""

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
            body = client.get("/metrics", headers={"X-API-Key": "test-key"}).text
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
