"""可视化安全/认证/输入验证测试（拆分自 test_visualization.py，Issue #730）。"""

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
    resp = await async_client.get("/metrics", headers={"X-API-Key": "secret-key-123"})
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
    """POST /api/tasks qubit_count 超过 MAX_CIRCUIT_QUBITS 上限应被拒绝（422，Issue #732）。"""
    payload = {
        "user_id": "user_001",
        "task_type": "quantum",
        "priority": 3,
        "qubit_count": MAX_CIRCUIT_QUBITS + 1,
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


class TestPydanticValidation:
    """TaskSubmit / SystemStatusUpdate Pydantic 字段边界值测试。"""

    def test_task_submit_qubit_count_exceeds_max(self):
        """qubit_count 超过 MAX_CIRCUIT_QUBITS 上限应抛 ValidationError（Issue #732）。"""
        with pytest.raises(ValidationError):
            TaskSubmit(qubit_count=MAX_CIRCUIT_QUBITS + 1)

    def test_task_submit_qubit_count_at_max(self):
        """qubit_count=MAX_CIRCUIT_QUBITS 应通过（边界值，Issue #732）。"""
        t = TaskSubmit(qubit_count=MAX_CIRCUIT_QUBITS)
        assert t.qubit_count == MAX_CIRCUIT_QUBITS

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
