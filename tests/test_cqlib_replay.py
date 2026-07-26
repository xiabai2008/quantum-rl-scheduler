"""cqlib 录制/回放测试框架 — CqlibReplayClient 回归测试（Issue #175）。

测试 CqlibReplayClient 从 JSON fixtures 回放天衍云 API 响应的能力，
无需安装 cqlib SDK，CI 环境可直接运行。

测试覆盖：
- 认证回放（authenticate 总是返回 True）
- 机器列表回放（list_backends 从 machine_list.json 加载）
- 任务提交回放（submit_quantum_task 从 task_submit.json 加载）
- 任务状态查询回放（get_task_status 轮询计数状态机）
- 等待任务完成回放（wait_for_task 从 running 流转到 completed）
- 容量错误处理（error_mode="capacity" 返回 None）
- 机器不可用处理（校准中机器 + auto_retry=False 返回 None）
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.api.cqlib_recorder import CqlibReplayClient

# 模块级标记：所有测试均为 cqlib 回放测试，有 fixtures 时即使无 SDK 也运行
pytestmark = pytest.mark.cqlib_replay

# 默认 fixtures 目录（相对于项目根目录）
FIXTURES_DIR = str(Path(__file__).parent / "fixtures" / "cqlib_responses")


def test_authenticate(cqlib_replay_client: CqlibReplayClient) -> None:
    """认证回放应始终返回 True。"""
    assert cqlib_replay_client.authenticate() is True


def test_list_backends(cqlib_replay_client: CqlibReplayClient) -> None:
    """机器列表回放应从 machine_list.json 加载正确的后端列表。"""
    backends = cqlib_replay_client.list_backends()

    assert isinstance(backends, list)
    assert len(backends) > 0

    # 验证每个后端包含必需字段
    for backend in backends:
        assert "id" in backend
        assert "type" in backend
        assert "status" in backend
        assert "name" in backend

    # 验证已知机器存在
    names = [b["name"] for b in backends]
    assert "tianyan_s" in names
    assert "tianyan176" in names

    # 验证状态分布（fixture 中 tianyan_sw 为 calibration）
    tianyan_sw = next(b for b in backends if b["name"] == "tianyan_sw")
    assert tianyan_sw["status"] == "calibration"

    # 验证 list_backends 返回的是副本（修改不影响内部状态）
    backends[0]["status"] = "tampered"
    reloaded = cqlib_replay_client.list_backends()
    assert reloaded[0]["status"] != "tampered"


def test_submit_task(cqlib_replay_client: CqlibReplayClient) -> None:
    """任务提交回放应从 task_submit.json 加载 task_id。"""
    task_id = cqlib_replay_client.submit_quantum_task(
        qcis="H Q0\nM Q0", shots=1024, task_name="Test_Replay_Task"
    )

    assert task_id is not None
    assert isinstance(task_id, str)
    assert len(task_id) > 0

    # 验证提交后可以查询状态（task_id 已注册到轮询计数器）
    status = cqlib_replay_client.get_task_status(task_id)
    assert status["task_id"] == task_id


def test_task_status(cqlib_replay_client: CqlibReplayClient) -> None:
    """任务状态查询回放应实现轮询计数状态机：首次 running，后续 completed。"""
    task_id = cqlib_replay_client.submit_quantum_task(qcis="H Q0\nM Q0", shots=512)
    assert task_id is not None

    # 首次查询应返回 running
    status_1 = cqlib_replay_client.get_task_status(task_id)
    assert status_1["status"] == "running"
    assert status_1["task_id"] == task_id
    assert "raw" in status_1

    # 第二次查询应返回 completed
    status_2 = cqlib_replay_client.get_task_status(task_id)
    assert status_2["status"] == "completed"
    assert status_2["task_id"] == task_id
    assert status_2["result"] is not None
    assert "0" in status_2["result"]
    assert "1" in status_2["result"]

    # 第三次查询仍应返回 completed
    status_3 = cqlib_replay_client.get_task_status(task_id)
    assert status_3["status"] == "completed"


def test_wait_for_task(cqlib_replay_client: CqlibReplayClient) -> None:
    """等待任务完成回放应从 running 流转到 completed 并返回最终结果。"""
    task_id = cqlib_replay_client.submit_quantum_task(qcis="H Q0\nM Q0", shots=1024)
    assert task_id is not None

    # 使用 poll_interval=0 避免实际休眠
    result = cqlib_replay_client.wait_for_task(task_id, timeout=10, poll_interval=0)

    assert result["status"] == "completed"
    assert result["task_id"] == task_id
    assert result["result"] is not None

    # 验证概率结果（H 门测量应接近 50/50）
    prob = result["result"]
    assert "0" in prob
    assert "1" in prob
    total = prob["0"] + prob["1"]
    assert abs(total - 1.0) < 0.01  # 概率之和应接近 1


def test_submit_capacity_error() -> None:
    """容量错误处理：error_mode='capacity' 时提交应返回 None。"""
    client = CqlibReplayClient(FIXTURES_DIR, error_mode="capacity")

    task_id = client.submit_quantum_task(qcis="H Q0\nM Q0", shots=1024)

    assert task_id is None


def test_submit_machine_unavailable() -> None:
    """机器不可用处理：校准中机器 + auto_retry=False 时提交应返回 None。"""
    # tianyan_sw 在 fixture 中状态为 calibration
    client = CqlibReplayClient(FIXTURES_DIR, machine_name="tianyan_sw", auto_retry_machine=False)

    # 验证机器确实不可用
    assert client._is_machine_available("tianyan_sw") is False

    task_id = client.submit_quantum_task(qcis="H Q0\nM Q0", shots=1024)

    assert task_id is None


def test_submit_machine_unavailable_with_retry() -> None:
    """机器不可用 + auto_retry=True 时应切换到备用机器并返回 task_id。"""
    # tianyan_sw 在 fixture 中状态为 calibration，但 tianyan_s 是 running
    client = CqlibReplayClient(FIXTURES_DIR, machine_name="tianyan_sw", auto_retry_machine=True)

    task_id = client.submit_quantum_task(qcis="H Q0\nM Q0", shots=1024)

    assert task_id is not None
    assert isinstance(task_id, str)


def test_get_task_result(cqlib_replay_client: CqlibReplayClient) -> None:
    """任务结果查询回放应从 task_result.json 加载完成结果。"""
    result = cqlib_replay_client.get_task_result("any-task-id")

    assert result["status"] == "completed"
    assert result["task_id"] == "any-task-id"
    assert result["result"] is not None
    assert "raw" in result


def test_get_queue_status(cqlib_replay_client: CqlibReplayClient) -> None:
    """队列状态应基于 fixtures 机器列表正确统计。"""
    queue = cqlib_replay_client.get_queue_status()

    assert queue["total_machines"] > 0
    assert queue["running"] > 0
    assert queue["running"] <= queue["total_machines"]
    assert isinstance(queue["available"], list)
    assert "tianyan_s" in queue["available"]
    # tianyan_sw 为 calibration，不应在 available 中
    assert "tianyan_sw" not in queue["available"]


def test_is_available(cqlib_replay_client: CqlibReplayClient) -> None:
    """默认机器 tianyan_s 为 running 时 is_available 应返回 True。"""
    assert cqlib_replay_client.is_available() is True

    # tianyan_sw 为 calibration，应返回 False
    client = CqlibReplayClient(FIXTURES_DIR, machine_name="tianyan_sw")
    assert client.is_available() is False
