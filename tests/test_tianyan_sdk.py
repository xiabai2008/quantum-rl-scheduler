"""天衍 cqlib 封装的无真机单元测试。

测试通过 ``sys.modules`` 注入最小 SDK 替身，不依赖专有 cqlib 包、网络或真实机时。
文件名和测试节点刻意不包含 SDK 包名，以确保 CI 不会按可选依赖规则跳过。
"""

from __future__ import annotations

import sys
import types
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from src.api.tianyan_cqlib import (
    CqlibTianyanClient,
    MultiMachineCqlibCoordinator,
    create_multi_machine_clients,
)


@pytest.fixture
def fake_sdk(monkeypatch: pytest.MonkeyPatch) -> tuple[types.ModuleType, type[Exception]]:
    """注册可配置的 cqlib 模块替身。"""
    sdk = types.ModuleType("cqlib")
    exceptions = types.ModuleType("cqlib.exceptions")

    class RequestError(Exception):
        """模拟 cqlib 查询超时异常。"""

    exceptions_api = cast(Any, exceptions)
    sdk_api = cast(Any, sdk)
    exceptions_api.CqlibRequestError = RequestError
    sdk_api.exceptions = exceptions
    sdk_api.TianYanPlatform = MagicMock(name="TianYanPlatform")
    monkeypatch.setitem(sys.modules, "cqlib", sdk)
    monkeypatch.setitem(sys.modules, "cqlib.exceptions", exceptions)
    return sdk, RequestError


@pytest.fixture
def client(fake_sdk: tuple[types.ModuleType, type[Exception]]) -> CqlibTianyanClient:
    """创建注入假平台的客户端。"""
    sdk, _ = fake_sdk
    result = CqlibTianyanClient("fake-key", "tianyan_s", auto_retry_machine=True)
    platform = MagicMock(name="platform")
    platform.query_quantum_computer_list.return_value = [
        ("id-s", "superconducting", "running", "tianyan_s"),
        ("id-sw", "superconducting", "maintenance", "tianyan_sw"),
    ]
    cast(Any, result)._platform = platform
    result.cqlib = sdk
    return result


def test_platform_authentication_and_backend_mapping(
    fake_sdk: tuple[types.ModuleType, type[Exception]],
) -> None:
    """平台应懒加载、缓存，并规范化机器信息。"""
    sdk, _ = fake_sdk
    sdk_api = cast(Any, sdk)
    platform = MagicMock()
    platform.query_quantum_computer_list.return_value = [
        ("id", "superconducting", "running", "tianyan_s")
    ]
    sdk_api.TianYanPlatform.return_value = platform
    subject = CqlibTianyanClient("key", "tianyan_s")

    assert subject.authenticate() is True
    assert subject.platform is platform
    assert subject.list_backends() == [
        {"id": "id", "type": "superconducting", "status": "running", "name": "tianyan_s"}
    ]
    assert subject.get_backend_info()["id"] == "id"
    assert subject.get_backend_info("missing") == {}
    sdk_api.TianYanPlatform.assert_called_once_with(login_key="key", machine_name="tianyan_s")


def test_authentication_and_backend_errors_return_safe_defaults(
    fake_sdk: tuple[types.ModuleType, type[Exception]],
) -> None:
    """平台构造或机器查询失败时应返回可降级结果。"""
    sdk, _ = fake_sdk
    cast(Any, sdk).TianYanPlatform.side_effect = RuntimeError("bad key")
    subject = CqlibTianyanClient("bad", "tianyan_s")
    assert subject.authenticate() is False

    platform = MagicMock()
    platform.query_quantum_computer_list.side_effect = RuntimeError("network down")
    cast(Any, subject)._platform = platform
    assert subject.list_backends() == []


def test_submit_validates_input_and_honours_quota(client: CqlibTianyanClient) -> None:
    """空输入应拒绝，配额不足时不应访问平台。"""
    with pytest.raises(ValueError, match="qcis"):
        client.submit_quantum_task()

    quota = MagicMock()
    quota.can_consume.return_value = False
    client._quota_tracker = quota
    assert client.submit_quantum_task(qcis="H Q0", shots=8) is None
    cast(Any, client)._platform.submit_experiment.assert_not_called()


@pytest.mark.parametrize(
    ("platform_result", "expected"),
    [(["task-1"], "task-1"), (1234, "1234"), ([], "[]")],
)
def test_submit_parses_result_and_consumes_quota(
    client: CqlibTianyanClient, platform_result: object, expected: str
) -> None:
    """列表、标量和空列表响应都应稳定转换为任务 ID。"""
    quota = MagicMock()
    quota.can_consume.return_value = True
    client._quota_tracker = quota
    cast(Any, client)._platform.submit_experiment.return_value = platform_result

    assert client.submit_quantum_task(qcis="H Q0", shots=16, task_name="unit") == expected
    quota.consume.assert_called_once_with(shots=16, tasks=1)


def test_submit_accepts_circuit_and_handles_unavailable_machine(
    client: CqlibTianyanClient,
) -> None:
    """电路对象应转为 QCIS，离线机器应按重试配置分流。"""
    circuit = types.SimpleNamespace(qcis="X Q0")
    cast(Any, client)._platform.submit_experiment.return_value = ["task-circuit"]
    assert client.submit_quantum_task(circuit=circuit) == "task-circuit"

    with (
        patch.object(client, "_is_machine_available", return_value=False),
        patch.object(client, "_retry_other_machine", return_value="task-alt") as retry,
    ):
        assert client.submit_quantum_task(qcis="H Q0") == "task-alt"
    retry.assert_called_once_with("H Q0", 1024, "Scheduler_Task")

    client.auto_retry_machine = False
    with patch.object(client, "_is_machine_available", return_value=False):
        assert client.submit_quantum_task(qcis="H Q0") is None


@pytest.mark.parametrize(
    ("message", "retry", "expected"),
    [
        ("capacity exceeded", True, None),
        ("machine offline", True, "task-alt"),
        ("temporary network error", True, "task-alt"),
        ("temporary network error", False, None),
    ],
)
def test_submit_exception_policy(
    client: CqlibTianyanClient, message: str, retry: bool, expected: str | None
) -> None:
    """容量、离线、普通异常应遵循各自的切换策略。"""
    client.auto_retry_machine = retry
    cast(Any, client)._platform.submit_experiment.side_effect = RuntimeError(message)
    with (
        patch.object(client, "_is_machine_available", return_value=True),
        patch.object(client, "_retry_other_machine", return_value="task-alt") as retry_call,
    ):
        assert client.submit_quantum_task(qcis="H Q0") == expected
    if retry and "capacity" not in message:
        retry_call.assert_called_once()
    else:
        retry_call.assert_not_called()


def test_error_classifiers_cover_positive_and_negative_cases() -> None:
    """错误分类应大小写无关，且不误判普通错误。"""
    assert CqlibTianyanClient._is_unavailable_error("MAINTENANCE")
    assert not CqlibTianyanClient._is_unavailable_error("bad response")
    assert CqlibTianyanClient._is_capacity_error("QUBIT capacity")
    assert not CqlibTianyanClient._is_capacity_error("bad response")
    assert CqlibTianyanClient._is_permission_error("FORBIDDEN")
    assert not CqlibTianyanClient._is_permission_error("bad response")


def test_machine_availability_is_optimistic_on_lookup_failure(
    client: CqlibTianyanClient,
) -> None:
    """已知维护机器应拒绝，未知或查询失败时应乐观放行。"""
    assert client._is_machine_available("tianyan_s") is True
    assert client._is_machine_available("tianyan_sw") is False
    assert client._is_machine_available("unknown") is True
    with patch.object(client, "list_backends", side_effect=RuntimeError("network")):
        assert client._is_machine_available("tianyan_s") is True


def test_retry_skips_unavailable_then_accepts_scalar_result(
    client: CqlibTianyanClient,
) -> None:
    """备用机应跳过离线节点，并支持 SDK 返回标量任务 ID。"""
    client.REAL_MACHINES = ["tianyan_s", "offline", "healthy"]
    alt = MagicMock()
    alt.submit_experiment.return_value = 7788
    client.cqlib.TianYanPlatform.return_value = alt
    quota = MagicMock()
    client._quota_tracker = quota
    with patch.object(client, "_is_machine_available", side_effect=lambda name: name == "healthy"):
        assert client._retry_other_machine("H Q0", 32, "retry") == "7788"
    quota.consume.assert_called_once_with(shots=32, tasks=1)


def test_retry_continues_after_permission_error(client: CqlibTianyanClient) -> None:
    """专属资源无权限时应继续尝试下一台机器。"""
    client.REAL_MACHINES = ["tianyan_s", "private", "public"]
    denied = MagicMock()
    denied.submit_experiment.side_effect = RuntimeError("permission denied")
    accepted = MagicMock()
    accepted.submit_experiment.return_value = ["task-public"]
    client.cqlib.TianYanPlatform.side_effect = [denied, accepted]
    with patch.object(client, "_is_machine_available", return_value=True):
        assert client._retry_other_machine("H Q0", 32, "retry") == "task-public"


def test_retry_stops_on_capacity_and_returns_none_after_failures(
    client: CqlibTianyanClient,
) -> None:
    """容量错误应立即停止；普通错误耗尽候选后应返回 None。"""
    client.REAL_MACHINES = ["tianyan_s", "backup"]
    alt = MagicMock()
    client.cqlib.TianYanPlatform.return_value = alt
    with patch.object(client, "_is_machine_available", return_value=True):
        alt.submit_experiment.side_effect = RuntimeError("qubit capacity exceeded")
        assert client._retry_other_machine("H Q0", 32, "retry") is None
        alt.submit_experiment.side_effect = RuntimeError("network")
        assert client._retry_other_machine("H Q0", 32, "retry") is None


def test_task_status_result_shapes_and_request_timeout(
    client: CqlibTianyanClient,
    fake_sdk: tuple[types.ModuleType, type[Exception]],
) -> None:
    """完成、运行、未知和 SDK 请求超时应映射到稳定状态。"""
    _, request_error = fake_sdk
    cast(Any, client)._platform.query_experiment.return_value = [
        {"probability": {"0": 1.0}, "resultStatus": "done"}
    ]
    assert client.get_task_status("done")["status"] == "completed"
    assert client.get_task_result("done")["result"] == {"0": 1.0}

    cast(Any, client)._platform.query_experiment.return_value = [{"queued": True}]
    assert client.get_task_status("queued")["status"] == "running"
    cast(Any, client)._platform.query_experiment.return_value = {"unexpected": True}
    assert client.get_task_status("unknown")["status"] == "unknown"
    cast(Any, client)._platform.query_experiment.side_effect = request_error("still running")
    assert client.get_task_status("timeout")["status"] == "running"


def test_task_status_generic_error(client: CqlibTianyanClient) -> None:
    """非超时查询异常应返回 error，不能中断调度循环。"""
    cast(Any, client)._platform.query_experiment.side_effect = RuntimeError("bad response")
    status = client.get_task_status("bad")
    assert status["status"] == "error"
    assert status["error"] == "bad response"


@pytest.mark.parametrize("terminal", ["completed", "error"])
def test_wait_returns_terminal_status(client: CqlibTianyanClient, terminal: str) -> None:
    """轮询遇到终态应立即返回。"""
    with patch.object(client, "get_task_status", return_value={"status": terminal}):
        assert client.wait_for_task("task", timeout=1, poll_interval=0)["status"] == terminal


def test_wait_timeout_queue_and_alias(client: CqlibTianyanClient) -> None:
    """轮询超时、队列汇总和非阻塞提交别名应保持稳定。"""
    with patch.object(client, "get_task_status", return_value={"status": "running"}):
        assert client.wait_for_task("task", timeout=0, poll_interval=0)["status"] == "timeout"

    queue = client.get_queue_status()
    assert queue == {"total_machines": 2, "running": 1, "available": ["tianyan_s"]}
    with patch.object(client, "submit_quantum_task", return_value="alias-task") as submit:
        assert client.submit_and_get_task_id("H Q0", shots=4, task_name="alias") == "alias-task"
    submit.assert_called_once_with(qcis="H Q0", shots=4, task_name="alias")


def test_is_available_handles_auth_machine_and_unexpected_errors(
    client: CqlibTianyanClient,
) -> None:
    """可用性检查应覆盖认证失败、机器离线和内部异常。"""
    with (
        patch.object(client, "authenticate", return_value=True),
        patch.object(client, "_is_machine_available", return_value=True),
    ):
        assert client.is_available() is True
    with patch.object(client, "authenticate", return_value=False):
        assert client.is_available() is False
    with patch.object(client, "authenticate", side_effect=RuntimeError("boom")):
        assert client.is_available() is False


def test_coordinator_lazy_clients_factory_and_failure_stats(
    fake_sdk: tuple[types.ModuleType, type[Exception]],
) -> None:
    """多机器构造应懒加载并缓存，非法机器应计入失败统计。"""
    coordinator = MultiMachineCqlibCoordinator("key", ["m1", "m2"])
    first = coordinator._get_client("m1")
    assert coordinator._get_client("m1") is first
    assert set(coordinator.as_client_map()) == {"m1", "m2"}

    assert coordinator.submit_to_machine("missing", "H Q0") is None
    assert coordinator.get_submit_stats()["m1"] == {"submit": 0, "fail": 0}
    assert coordinator._fail_count["missing"] == 1

    clients = create_multi_machine_clients("key", ["m1", "m2"])
    assert set(clients) == {"m1", "m2"}
    assert all(not item.auto_retry_machine for item in clients.values())


def test_coordinator_status_and_quota_paths(
    fake_sdk: tuple[types.ModuleType, type[Exception]],
) -> None:
    """协调器应聚合状态、记录成功次数和配额，并隔离单机异常。"""
    quota = MagicMock()
    coordinator = MultiMachineCqlibCoordinator("key", ["ok", "bad"], quota_tracker=quota)
    ok_client = MagicMock()
    ok_client.submit_quantum_task.return_value = "task-ok"
    ok_client.get_queue_status.return_value = {"running": 1}
    bad_client = MagicMock()
    bad_client.get_queue_status.side_effect = RuntimeError("offline")

    with patch.object(
        coordinator,
        "_get_client",
        side_effect=lambda name: ok_client if name == "ok" else bad_client,
    ):
        assert coordinator.submit_to_machine("ok", "H Q0", shots=8) == "task-ok"
        statuses = coordinator.get_all_status()

    quota.consume.assert_called_once_with(shots=8, tasks=1)
    assert coordinator.get_submit_stats()["ok"]["submit"] == 1
    assert statuses["ok"] == {"running": 1}
    assert "offline" in statuses["bad"]["error"]
