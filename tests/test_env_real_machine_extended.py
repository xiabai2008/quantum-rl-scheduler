"""
env_real_machine.py 真机闭环模块的补充单元测试（覆盖率提升）

补充覆盖以下分支：
- generate_qcis_circuit: two_qubit_gates=True 纠缠层生成
- submit_to_real_machine: client=None 跳过、real_task_id=None 被拒绝、降级保护、预算上限
- record_real_failure: 连续失败触发降级
- poll_pending_real_tasks: 空列表、客户端丢失、查询异常、completed/error/timeout/running 各状态
- _update_task_duration: 当前任务回写、队列任务回写、找不到任务
"""

import os
import sys
from typing import Any
from unittest.mock import MagicMock

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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


# ============================================================
# 辅助：构造轻量 mock 环境（避免实例化完整 QuantumSchedulingEnv）
# ============================================================
class _MockEnv:
    """轻量 mock 环境，仅提供真机闭环函数所需属性。"""

    def __init__(
        self,
        clients: dict[str, Any] | None = None,
        degraded: bool = False,
        max_real_submissions: int | None = None,
        feedback_weight: float = 1.0,
        shots: int = 64,
        use_real_machine: bool = True,
    ) -> None:
        self._real_clients: dict[str, Any] = clients or {}
        self._real_machine_degraded: bool = degraded
        self.max_real_submissions: int | None = max_real_submissions
        self.real_machine_feedback_weight: float = feedback_weight
        self.real_machine_shots: int = shots
        self.use_real_machine: bool = use_real_machine

        self._real_submission_attempts_total: int = 0
        self._machine_real_submits: dict[str, int] = {}
        self._pending_real_tasks: list[dict[str, Any]] = []
        self._current_step: int = 0
        self._render_log: list[str] = []

        self._real_fail_count: int = 0
        self._real_consecutive_failures: int = 0
        self._real_success_count: int = 0

        self._current_task: Task | None = None
        self._task_queue: list[Task] = []


def _make_machine(name: str = "tianyan176", total_qubits: int = 66) -> QuantumMachine:
    return QuantumMachine(name=name, total_qubits=total_qubits, is_real=True)


# ============================================================
# generate_qcis_circuit: two_qubit_gates=True 分支
# ============================================================
class TestGenerateQcisTwoQubitGates:
    """覆盖 two_qubit_gates=True 纠缠层生成分支（lines 110-117）。"""

    def test_two_qubit_gates_adds_entanglement(self):
        """two_qubit_gates=True 应生成 CNOT/CZ 纠缠门。"""
        task = Task(task_id="ent", task_type="quantum", qubit_count=4, priority=3)
        qcis = generate_qcis_circuit(task, seed=42, two_qubit_gates=True)
        # 应包含两比特门
        has_two_qubit = "CNOT" in qcis or "CZ" in qcis
        assert has_two_qubit, f"应包含 CNOT/CZ 门: {qcis}"

    def test_two_qubit_gates_depth_scales_with_priority(self):
        """高优先级 → 更多纠缠层（depth_factor = priority - 1）。"""
        task_low = Task(task_id="low", task_type="quantum", qubit_count=6, priority=1)
        task_high = Task(task_id="high", task_type="quantum", qubit_count=6, priority=5)
        q_low = generate_qcis_circuit(task_low, seed=42, two_qubit_gates=True)
        q_high = generate_qcis_circuit(task_high, seed=42, two_qubit_gates=True)
        # priority=1 → depth_factor=0（无纠缠层）；priority=5 → depth_factor=4
        assert len(q_high) > len(q_low)

    def test_two_qubit_gates_single_qubit_no_entangle(self):
        """单比特任务（n_qubits=1）无纠缠对，不生成两比特门。"""
        task = Task(task_id="single", task_type="quantum", qubit_count=1, priority=5)
        qcis = generate_qcis_circuit(task, seed=42, two_qubit_gates=True)
        assert "CNOT" not in qcis
        assert "CZ" not in qcis

    def test_two_qubit_gates_disabled_by_default(self):
        """默认 two_qubit_gates=False 不生成纠缠门。"""
        task = Task(task_id="noent", task_type="quantum", qubit_count=4, priority=5)
        qcis = generate_qcis_circuit(task, seed=42)
        assert "CNOT" not in qcis
        assert "CZ" not in qcis


# ============================================================
# submit_to_real_machine: client=None / rejected / degraded / budget
# ============================================================
class TestSubmitToRealMachineBranches:
    """覆盖 submit_to_real_machine 的各分支。"""

    def test_client_none_skips_submission(self):
        """client=None 时应跳过提交（line 163）。"""
        env = _MockEnv(clients={})  # 无 client
        machine = _make_machine()
        task = Task(task_id="t1", task_type="quantum", qubit_count=1)
        submit_to_real_machine(env, machine, task)
        # 不应增加提交计数
        assert env._real_submission_attempts_total == 0
        assert len(env._pending_real_tasks) == 0

    def test_degraded_skips_submission(self):
        """降级标志为 True 时应跳过提交（line 152-153）。"""
        client = MagicMock()
        client.submit_quantum_task.return_value = "real-1"
        env = _MockEnv(clients={"tianyan176": client}, degraded=True)
        machine = _make_machine()
        task = Task(task_id="t1", task_type="quantum", qubit_count=1)
        submit_to_real_machine(env, machine, task)
        client.submit_quantum_task.assert_not_called()
        assert env._real_submission_attempts_total == 0

    def test_budget_exhausted_skips_submission(self):
        """预算上限已满时应跳过提交（lines 155-159）。"""
        client = MagicMock()
        client.submit_quantum_task.return_value = "real-1"
        env = _MockEnv(clients={"tianyan176": client}, max_real_submissions=2)
        env._real_submission_attempts_total = 2  # 已达上限
        machine = _make_machine()
        task = Task(task_id="t1", task_type="quantum", qubit_count=1)
        submit_to_real_machine(env, machine, task)
        client.submit_quantum_task.assert_not_called()

    def test_submission_rejected_records_failure(self):
        """submit_quantum_task 返回 None 视为被拒绝，记录失败（line 205）。"""
        client = MagicMock()
        client.submit_quantum_task.return_value = None
        env = _MockEnv(clients={"tianyan176": client})
        machine = _make_machine()
        task = Task(task_id="rejected", task_type="quantum", qubit_count=1)
        submit_to_real_machine(env, machine, task)
        # 应记录失败
        assert env._real_fail_count == 1
        assert env._real_consecutive_failures == 1
        # 不应加入 pending
        assert len(env._pending_real_tasks) == 0

    def test_submission_exception_records_failure(self):
        """submit_quantum_task 抛异常时记录失败并写入 render_log（lines 206-210）。"""
        client = MagicMock()
        client.submit_quantum_task.side_effect = ConnectionError("网络断开")
        env = _MockEnv(clients={"tianyan176": client})
        machine = _make_machine()
        task = Task(task_id="err", task_type="quantum", qubit_count=1)
        submit_to_real_machine(env, machine, task)
        assert env._real_fail_count == 1
        assert any("提交失败" in msg for msg in env._render_log)

    def test_successful_submission_registers_pending(self):
        """成功提交应登记到 pending 列表。"""
        client = MagicMock()
        client.submit_quantum_task.return_value = "real-abc"
        env = _MockEnv(clients={"tianyan176": client})
        machine = _make_machine()
        task = Task(task_id="ok", task_type="quantum", qubit_count=1)
        submit_to_real_machine(env, machine, task)
        assert len(env._pending_real_tasks) == 1
        assert env._pending_real_tasks[0]["task_id"] == "real-abc"
        assert env._machine_real_submits["tianyan176"] == 1

    def test_uses_task_qcis_when_available(self):
        """task 有 qcis 字段时直接使用，不重新生成。"""
        custom_qcis = "H Q0\nM Q0"
        client = MagicMock()
        client.submit_quantum_task.return_value = "real-1"
        env = _MockEnv(clients={"tianyan176": client})
        machine = _make_machine()
        task = Task(task_id="custom", task_type="quantum", qubit_count=1, qcis=custom_qcis)
        submit_to_real_machine(env, machine, task)
        submitted_qcis = client.submit_quantum_task.call_args.kwargs["qcis"]
        assert submitted_qcis == custom_qcis


# ============================================================
# record_real_failure: 降级触发
# ============================================================
class TestRecordRealFailure:
    """覆盖 record_real_failure 降级逻辑。"""

    def test_failure_increments_counters(self):
        """每次失败应递增失败计数。"""
        env = _MockEnv()
        record_real_failure(env, "tianyan176", "测试失败")
        assert env._real_fail_count == 1
        assert env._real_consecutive_failures == 1
        assert not env._real_machine_degraded

    def test_degrade_after_threshold(self):
        """连续失败达阈值后触发降级。"""
        env = _MockEnv()
        for i in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD):
            record_real_failure(env, "tianyan176", f"失败{i}")
        assert env._real_machine_degraded is True
        assert any("降级" in msg for msg in env._render_log)

    def test_degrade_only_once(self):
        """已降级后不重复触发降级日志。"""
        env = _MockEnv()
        for i in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD + 2):
            record_real_failure(env, "tianyan176", f"失败{i}")
        assert env._real_machine_degraded is True
        # 降级日志只出现一次
        degrade_logs = [m for m in env._render_log if "降级" in m]
        assert len(degrade_logs) == 1


# ============================================================
# poll_pending_real_tasks: 各状态分支
# ============================================================
class TestPollPendingRealTasks:
    """覆盖 poll_pending_real_tasks 的所有状态分支。"""

    def test_empty_pending_returns_zero(self):
        """空 pending 列表返回 0.0（line 263-264）。"""
        env = _MockEnv()
        assert poll_pending_real_tasks(env) == 0.0

    def test_client_lost_records_failure(self):
        """轮询时客户端丢失视为失败（lines 278-280）。"""
        env = _MockEnv(clients={})  # 无 client
        env._pending_real_tasks = [
            {
                "task_id": "real-1",
                "machine_name": "tianyan176",
                "submit_step": 0,
                "poll_count": 0,
                "task_id_str": "t1",
            }
        ]
        feedback = poll_pending_real_tasks(env)
        assert feedback == REAL_MACHINE_FAIL_PENALTY * env.real_machine_feedback_weight
        assert env._real_fail_count == 1
        # 失败的任务不应保留在 pending
        assert len(env._pending_real_tasks) == 0

    def test_query_exception_keeps_pending(self):
        """get_task_status 抛异常时保留在 pending（lines 284-288）。"""
        client = MagicMock()
        client.get_task_status.side_effect = RuntimeError("查询超时")
        env = _MockEnv(clients={"tianyan176": client})
        env._pending_real_tasks = [
            {
                "task_id": "real-1",
                "machine_name": "tianyan176",
                "submit_step": 0,
                "poll_count": 0,
                "task_id_str": "t1",
            }
        ]
        feedback = poll_pending_real_tasks(env)
        assert feedback == 0.0  # 无新结果
        # 任务应保留在 pending
        assert len(env._pending_real_tasks) == 1
        assert env._pending_real_tasks[0]["poll_count"] == 1

    def test_completed_status_gives_bonus(self):
        """completed 状态返回成功奖励（lines 292-305）。"""
        client = MagicMock()
        client.get_task_status.return_value = {
            "status": "completed",
            "execution_time_s": 1.5,
        }
        env = _MockEnv(clients={"tianyan176": client})
        env._pending_real_tasks = [
            {
                "task_id": "real-1",
                "machine_name": "tianyan176",
                "submit_step": 0,
                "poll_count": 0,
                "task_id_str": "t1",
            }
        ]
        feedback = poll_pending_real_tasks(env)
        assert feedback == REAL_MACHINE_SUCCESS_BONUS * env.real_machine_feedback_weight
        assert env._real_success_count == 1
        assert env._real_consecutive_failures == 0
        assert len(env._pending_real_tasks) == 0

    def test_error_status_gives_penalty(self):
        """error 状态返回惩罚并记录失败（lines 306-309）。"""
        client = MagicMock()
        client.get_task_status.return_value = {"status": "error"}
        env = _MockEnv(clients={"tianyan176": client})
        env._pending_real_tasks = [
            {
                "task_id": "real-1",
                "machine_name": "tianyan176",
                "submit_step": 0,
                "poll_count": 0,
                "task_id_str": "t1",
            }
        ]
        feedback = poll_pending_real_tasks(env)
        assert feedback == REAL_MACHINE_FAIL_PENALTY * env.real_machine_feedback_weight
        assert env._real_fail_count == 1
        assert len(env._pending_real_tasks) == 0

    def test_timeout_gives_penalty(self):
        """轮询次数超上限视为超时失败（lines 310-316）。"""
        client = MagicMock()
        client.get_task_status.return_value = {"status": "running"}
        env = _MockEnv(clients={"tianyan176": client})
        env._pending_real_tasks = [
            {
                "task_id": "real-1",
                "machine_name": "tianyan176",
                "submit_step": 0,
                "poll_count": REAL_MACHINE_MAX_POLL_STEPS - 1,  # 本次 +1 后达到上限
                "task_id_str": "t1",
            }
        ]
        feedback = poll_pending_real_tasks(env)
        assert feedback == REAL_MACHINE_FAIL_PENALTY * env.real_machine_feedback_weight
        assert env._real_fail_count == 1
        assert len(env._pending_real_tasks) == 0

    def test_running_status_keeps_pending(self):
        """running 状态保留在 pending 列表（lines 317-319）。"""
        client = MagicMock()
        client.get_task_status.return_value = {"status": "running"}
        env = _MockEnv(clients={"tianyan176": client})
        env._pending_real_tasks = [
            {
                "task_id": "real-1",
                "machine_name": "tianyan176",
                "submit_step": 0,
                "poll_count": 0,
                "task_id_str": "t1",
            }
        ]
        feedback = poll_pending_real_tasks(env)
        assert feedback == 0.0
        assert len(env._pending_real_tasks) == 1

    def test_unknown_status_keeps_pending(self):
        """unknown 状态保留在 pending 列表。"""
        client = MagicMock()
        client.get_task_status.return_value = {"status": "unknown"}
        env = _MockEnv(clients={"tianyan176": client})
        env._pending_real_tasks = [
            {
                "task_id": "real-1",
                "machine_name": "tianyan176",
                "submit_step": 0,
                "poll_count": 0,
                "task_id_str": "t1",
            }
        ]
        feedback = poll_pending_real_tasks(env)
        assert feedback == 0.0
        assert len(env._pending_real_tasks) == 1

    def test_mixed_statuses_accumulate_feedback(self):
        """多个 pending 任务混合状态时累加反馈。"""
        client = MagicMock()
        # 第一次调用 completed，第二次 error
        client.get_task_status.side_effect = [
            {"status": "completed", "execution_time_s": 1.0},
            {"status": "error"},
        ]
        env = _MockEnv(clients={"tianyan176": client})
        env._pending_real_tasks = [
            {
                "task_id": "real-1",
                "machine_name": "tianyan176",
                "submit_step": 0,
                "poll_count": 0,
                "task_id_str": "t1",
            },
            {
                "task_id": "real-2",
                "machine_name": "tianyan176",
                "submit_step": 0,
                "poll_count": 0,
                "task_id_str": "t2",
            },
        ]
        feedback = poll_pending_real_tasks(env)
        expected = (
            REAL_MACHINE_SUCCESS_BONUS + REAL_MACHINE_FAIL_PENALTY
        ) * env.real_machine_feedback_weight
        assert feedback == expected


# ============================================================
# _update_task_duration: 回写分支
# ============================================================
class TestUpdateTaskDuration:
    """覆盖 _update_task_duration 的各分支（lines 349-371）。"""

    def test_none_execution_time_returns_early(self):
        """actual_execution_s=None 时直接返回（line 349-350）。"""
        env = _MockEnv()
        task = Task(task_id="t1", task_type="quantum")
        env._current_task = task
        _update_task_duration(env, "t1", None)
        # execution_time 不应被修改
        assert task.execution_time == 3  # 默认值

    def test_updates_current_task(self):
        """回写当前正在执行的任务（lines 353-358）。"""
        env = _MockEnv()
        task = Task(task_id="t1", task_type="quantum", execution_time=5)
        env._current_task = task
        _update_task_duration(env, "t1", 2.0)
        assert task.execution_time == 0

    def test_updates_queue_task(self):
        """回写队列中的任务（lines 361-368）。"""
        env = _MockEnv()
        task = Task(task_id="t2", task_type="quantum", execution_time=5)
        env._task_queue = [Task(task_id="t1", task_type="quantum"), task]
        _update_task_duration(env, "t2", 2.0)
        assert task.execution_time == 0

    def test_task_not_found_silent(self):
        """找不到任务时静默忽略（lines 370-371）。"""
        env = _MockEnv()
        env._current_task = Task(task_id="t1", task_type="quantum")
        env._task_queue = [Task(task_id="t2", task_type="quantum")]
        # 不应抛异常
        _update_task_duration(env, "nonexistent", 2.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
