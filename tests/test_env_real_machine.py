#!/usr/bin/env python
"""env_real_machine.py 真机闭环模块的单元测试"""

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from src.scheduler.env_real_machine import (
    _update_task_duration,
    generate_qcis_circuit,
    poll_pending_real_tasks,
    record_real_failure,
    submit_to_real_machine,
)
from src.scheduler.env_types import (
    REAL_MACHINE_DEGRADE_FAIL_THRESHOLD,
    REAL_MACHINE_FAIL_PENALTY,
    REAL_MACHINE_MAX_POLL_STEPS,
    REAL_MACHINE_SUCCESS_BONUS,
    QuantumMachine,
    Task,
)


def _make_env(client=None, *, feedback_weight=0.5):
    """构造不依赖真实 SDK 的最小环境替身。"""
    clients = {"tianyan_s": client} if client is not None else {}
    return SimpleNamespace(
        _real_machine_degraded=False,
        _real_clients=clients,
        _machine_real_submits={},
        _pending_real_tasks=[],
        _current_step=7,
        use_real_machine=True,
        _render_log=[],
        _real_fail_count=0,
        _real_consecutive_failures=0,
        real_machine_feedback_weight=feedback_weight,
        _real_success_count=0,
        _current_task=None,
        _task_queue=[],
    )


def _machine():
    """返回测试使用的真机描述。"""
    return QuantumMachine(name="tianyan_s", total_qubits=72, is_real=True)


def _quantum_task(**overrides):
    """返回测试使用的量子任务。"""
    values = {
        "task_id": "task-1",
        "task_type": "quantum",
        "qubit_count": 3,
        "priority": 3,
    }
    values.update(overrides)
    return Task(**values)


class TestGenerateQcisCircuit:
    """QCIS 电路生成测试"""

    def test_small_quantum_task(self):
        """小规模量子任务生成合理电路"""
        task = Task(task_id="0", task_type="quantum", qubit_count=3, priority=3)
        qcis = generate_qcis_circuit(task)
        assert isinstance(qcis, str)
        assert len(qcis) > 0
        # 至少包含测量
        assert "M" in qcis
        # 至少包含单比特门
        assert any(g in qcis for g in ["H", "X", "Y", "Z", "RX", "RY", "RZ"])
        # 比特数不超过任务需求
        lines = qcis.strip().split("\n")
        for line in lines:
            if "Q" in line:
                for part in line.split():
                    if part.startswith("Q"):
                        q_idx = int(part.split(",")[0][1:])
                        assert q_idx < 3, f"比特索引 {q_idx} 超出范围"

    def test_priority_affects_depth(self):
        """高优先级任务电路更深"""
        task_low = Task(task_id="0", task_type="quantum", qubit_count=5, priority=1)
        task_high = Task(task_id="1", task_type="quantum", qubit_count=5, priority=5)
        qcis_low = generate_qcis_circuit(task_low, seed=42)
        qcis_high = generate_qcis_circuit(task_high, seed=42)
        # 高优先级应包含更多门（深度因子更大）
        assert len(qcis_high) >= len(
            qcis_low
        ), f"高优先级电路应更深: {len(qcis_high)} vs {len(qcis_low)}"

    def test_deterministic_with_seed(self):
        """相同 seed 生成相同电路"""
        task = Task(task_id="0", task_type="quantum", qubit_count=5, priority=3)
        q1 = generate_qcis_circuit(task, seed=42)
        q2 = generate_qcis_circuit(task, seed=42)
        assert q1 == q2

    def test_different_seed_produces_different_circuit(self):
        """不同 seed 可能生成不同电路"""
        # 注意：小规模电路可能恰好相同，所以只是"可能"不同
        # 用多比特任务确保大概率不同
        task_big = Task(task_id="0", task_type="quantum", qubit_count=20, priority=5)
        q3 = generate_qcis_circuit(task_big, seed=42)
        q4 = generate_qcis_circuit(task_big, seed=123)
        assert q3 != q4, "大电路不同 seed 应生成不同电路"

    def test_classical_task_generates_minimal_circuit(self):
        """经典任务（qubit_count=0）生成最小电路"""
        task = Task(task_id="0", task_type="classical", qubit_count=0, priority=1)
        qcis = generate_qcis_circuit(task)
        assert "M Q0" in qcis
        assert qcis.count("\n") >= 1  # 至少 1 个单比特门 + 1 个测量

    def test_max_qubits_limit(self):
        """超过 max_qubits 限制时被截断"""
        task = Task(task_id="0", task_type="quantum", qubit_count=500, priority=3)
        qcis = generate_qcis_circuit(task, max_qubits=10)
        # 不应包含 Q10 以上的比特
        for line in qcis.strip().split("\n"):
            for part in line.split():
                for token in part.split(","):
                    if token.startswith("Q"):
                        q_idx = int(token[1:])
                        assert q_idx < 10, f"比特索引 {q_idx} 超出 max_qubits=10"

    def test_qcis_format_valid(self):
        """生成的电路符合 QCIS 基本格式"""
        task = Task(task_id="0", task_type="quantum", qubit_count=8, priority=3)
        qcis = generate_qcis_circuit(task, seed=42)
        lines = qcis.strip().split("\n")
        for line in lines:
            parts = line.split()
            assert len(parts) >= 2, f"每行至少要有门和比特: {line}"
            # 门名
            assert parts[0] in [
                "H",
                "X",
                "Y",
                "Z",
                "RX",
                "RY",
                "RZ",
                "CNOT",
                "CZ",
                "M",
            ], f"未知门: {parts[0]}"
            # 比特引用
            for p in parts[1:]:
                assert p.startswith("Q"), f"应为比特引用: {p}"

    def test_parameterized_rotation_gate(self):
        """旋转门应包含受控范围内的角度参数。"""
        task = _quantum_task(qubit_count=1)
        with patch("src.scheduler.env_real_machine._SINGLE_QUBIT_GATES", ["RX"]):
            qcis = generate_qcis_circuit(task, seed=9)
        gate, argument = qcis.splitlines()[0].split()
        qubit, angle = argument.split(",")
        assert gate == "RX"
        assert qubit == "Q0"
        assert 0.0 <= float(angle) <= 2 * 3.14159

    def test_two_qubit_layers_are_optional(self):
        """显式启用纠缠门时应覆盖相邻与交错比特对。"""
        task = _quantum_task(qubit_count=4, priority=3)
        qcis = generate_qcis_circuit(task, seed=7, two_qubit_gates=True)
        entangling = [line for line in qcis.splitlines() if line.startswith(("CNOT", "CZ"))]
        assert entangling
        assert any("Q0 Q1" in line for line in entangling)
        assert any("Q1 Q2" in line for line in entangling)


class TestTaskQcisField:
    """Task 数据类 qcis 字段测试"""

    def test_task_has_qcis_field(self):
        """Task 数据类默认包含 qcis 字段"""
        task = Task(task_id="0", task_type="quantum")
        assert hasattr(task, "qcis")
        assert task.qcis is None  # 默认未生成

    def test_task_accepts_qcis(self):
        """Task 可以接受自定义 qcis 电路"""
        custom_qcis = "H Q0\nCNOT Q0 Q1\nM Q0\nM Q1"
        task = Task(task_id="0", task_type="quantum", qcis=custom_qcis)
        assert task.qcis == custom_qcis

    def test_task_without_qcis_still_works(self):
        """没有 qcis 的 Task 仍然可以正常使用"""
        task = Task(task_id="0", task_type="classical", qubit_count=0)
        assert task.qcis is None
        # submit_to_real_machine 应自动生成


class TestSubmitToRealMachine:
    """真机提交入口使用客户端替身，不访问云平台。"""

    def test_degraded_environment_skips_submission(self):
        client = MagicMock()
        env = _make_env(client)
        env._real_machine_degraded = True

        submit_to_real_machine(env, _machine(), _quantum_task())

        client.submit_quantum_task.assert_not_called()
        assert env._pending_real_tasks == []

    def test_missing_client_skips_submission(self):
        env = _make_env()

        submit_to_real_machine(env, _machine(), _quantum_task())

        assert env._machine_real_submits == {}
        assert env._pending_real_tasks == []

    def test_custom_qcis_is_submitted_and_registered(self):
        client = MagicMock()
        client.submit_quantum_task.return_value = 12345
        env = _make_env(client)
        task = _quantum_task(qcis="H Q0\nM Q0")

        submit_to_real_machine(env, _machine(), task)

        client.submit_quantum_task.assert_called_once_with(
            qcis="H Q0\nM Q0",
            shots=512,
            task_name="RL_task-1",
        )
        assert env._machine_real_submits == {"tianyan_s": 1}
        assert env._pending_real_tasks == [
            {
                "task_id": "12345",
                "machine_name": "tianyan_s",
                "submit_step": 7,
                "poll_count": 0,
                "task_id_str": "task-1",
            }
        ]

    def test_missing_qcis_uses_free_tier_circuit(self):
        client = MagicMock()
        client.submit_quantum_task.return_value = "real-1"
        env = _make_env(client)

        with patch(
            "src.scheduler.env_real_machine.generate_qcis_circuit",
            return_value="X Q0\nM Q0",
        ) as generate:
            submit_to_real_machine(env, _machine(), _quantum_task())

        generate.assert_called_once()
        assert generate.call_args.kwargs["max_qubits"] == 1
        assert client.submit_quantum_task.call_args.kwargs["qcis"] == "X Q0\nM Q0"

    def test_none_task_id_records_failure(self):
        client = MagicMock()
        client.submit_quantum_task.return_value = None
        env = _make_env(client)

        submit_to_real_machine(env, _machine(), _quantum_task())

        assert env._machine_real_submits == {"tianyan_s": 1}
        assert env._pending_real_tasks == []
        assert env._real_fail_count == 1
        assert env._real_consecutive_failures == 1

    def test_submit_exception_is_contained_and_logged(self):
        client = MagicMock()
        client.submit_quantum_task.side_effect = OSError("network unavailable")
        env = _make_env(client)

        submit_to_real_machine(env, _machine(), _quantum_task())

        assert env._pending_real_tasks == []
        assert env._real_fail_count == 1
        assert "network unavailable" in env._render_log[-1]


class TestRealMachineFailurePolicy:
    """连续失败达到阈值后应只触发一次自动降级。"""

    def test_degrades_exactly_at_threshold(self):
        env = _make_env()

        for attempt in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD):
            record_real_failure(env, "tianyan_s", f"failure-{attempt}")

        assert env._real_fail_count == REAL_MACHINE_DEGRADE_FAIL_THRESHOLD
        assert env._real_machine_degraded is True
        assert len(env._render_log) == 1

        record_real_failure(env, "tianyan_s", "after-degrade")
        assert env._real_fail_count == REAL_MACHINE_DEGRADE_FAIL_THRESHOLD + 1
        assert len(env._render_log) == 1


class TestPollPendingRealTasks:
    """轮询逻辑覆盖成功、失败、超时与暂态异常。"""

    @staticmethod
    def _pending(*, poll_count=0):
        return {
            "task_id": "real-1",
            "machine_name": "tianyan_s",
            "submit_step": 7,
            "poll_count": poll_count,
            "task_id_str": "task-1",
        }

    def test_empty_pending_list_returns_zero(self):
        assert poll_pending_real_tasks(_make_env()) == 0.0

    def test_missing_client_fails_and_removes_task(self):
        env = _make_env(feedback_weight=0.25)
        env._pending_real_tasks = [self._pending()]

        feedback = poll_pending_real_tasks(env)

        assert feedback == pytest.approx(REAL_MACHINE_FAIL_PENALTY * 0.25)
        assert env._pending_real_tasks == []
        assert env._real_fail_count == 1

    def test_query_exception_keeps_task_pending(self):
        client = MagicMock()
        client.get_task_status.side_effect = TimeoutError("temporary timeout")
        env = _make_env(client)
        env._pending_real_tasks = [self._pending()]

        feedback = poll_pending_real_tasks(env)

        assert feedback == 0.0
        assert env._pending_real_tasks[0]["poll_count"] == 1
        assert env._real_fail_count == 0

    def test_completed_task_rewards_and_updates_current_task(self):
        client = MagicMock()
        client.get_task_status.return_value = {
            "status": "completed",
            "execution_time_s": 1.25,
        }
        env = _make_env(client, feedback_weight=1.5)
        env._current_task = _quantum_task(execution_time=8)
        env._real_consecutive_failures = 2
        env._pending_real_tasks = [self._pending()]

        feedback = poll_pending_real_tasks(env)

        assert feedback == pytest.approx(REAL_MACHINE_SUCCESS_BONUS * 1.5)
        assert env._real_success_count == 1
        assert env._real_consecutive_failures == 0
        assert env._current_task.execution_time == 0
        assert env._pending_real_tasks == []

    def test_completed_task_without_duration_does_not_mutate_queue(self):
        client = MagicMock()
        client.get_task_status.return_value = {"status": "completed"}
        env = _make_env(client)
        queued = _quantum_task(execution_time=9)
        env._task_queue = [queued]
        env._pending_real_tasks = [self._pending()]

        poll_pending_real_tasks(env)

        assert queued.execution_time == 9

    def test_error_status_penalizes_and_removes_task(self):
        client = MagicMock()
        client.get_task_status.return_value = {"status": "error"}
        env = _make_env(client, feedback_weight=2.0)
        env._pending_real_tasks = [self._pending()]

        feedback = poll_pending_real_tasks(env)

        assert feedback == pytest.approx(REAL_MACHINE_FAIL_PENALTY * 2.0)
        assert env._pending_real_tasks == []
        assert env._real_fail_count == 1

    def test_poll_timeout_penalizes_at_boundary(self):
        client = MagicMock()
        client.get_task_status.return_value = {"status": "running"}
        env = _make_env(client)
        env._pending_real_tasks = [self._pending(poll_count=REAL_MACHINE_MAX_POLL_STEPS - 1)]

        feedback = poll_pending_real_tasks(env)

        assert feedback == pytest.approx(REAL_MACHINE_FAIL_PENALTY * 0.5)
        assert env._pending_real_tasks == []
        assert env._real_fail_count == 1

    @pytest.mark.parametrize("status", ["running", "queued", None])
    def test_nonterminal_status_stays_pending(self, status):
        client = MagicMock()
        client.get_task_status.return_value = {"status": status}
        env = _make_env(client)
        env._pending_real_tasks = [self._pending()]

        feedback = poll_pending_real_tasks(env)

        assert feedback == 0.0
        assert env._pending_real_tasks[0]["poll_count"] == 1


class TestUpdateTaskDuration:
    """真机执行时间应回写当前任务或等待队列。"""

    def test_updates_matching_queue_task(self):
        env = _make_env()
        other = _quantum_task(task_id="other", execution_time=4)
        target = _quantum_task(execution_time=6)
        env._task_queue = [other, target]

        _update_task_duration(env, "task-1", 0.75)

        assert target.execution_time == 0
        assert other.execution_time == 4

    def test_none_duration_is_ignored(self):
        env = _make_env()
        target = _quantum_task(execution_time=6)
        env._task_queue = [target]

        _update_task_duration(env, "task-1", None)

        assert target.execution_time == 6

    def test_missing_task_is_safe(self):
        env = _make_env()
        env._task_queue = [_quantum_task(task_id="other")]

        _update_task_duration(env, "missing", 1.0)

        assert env._task_queue[0].execution_time == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
