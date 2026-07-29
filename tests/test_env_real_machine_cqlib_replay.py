"""env_real_machine 接 CqlibReplayClient 的端到端集成测试（Issue #360）。

与 ``test_env_real_machine.py`` / ``test_env_real_machine_extended.py`` 的区别：
    - 旧测试使用 ``MagicMock`` 模拟客户端，仅覆盖分支逻辑
    - 本测试使用真实的 ``CqlibReplayClient`` 从 fixtures 加载响应，
      验证 ``submit_to_real_machine`` → ``poll_pending_real_tasks`` →
      ``_compute_real_feedback`` → ``_record_real_result`` →
      ``_record_causal_feedback`` 的完整闭环

测试覆盖：
    - 正常闭环：提交 → 轮询 running → 轮询 completed → bonus 计入 reward
    - status_only / result_aware / shuffled 三种反馈模式
    - 真机统计计数器递增与 ``get_real_machine_stats`` 一致性
    - ``_real_result_records`` 与 ``_real_feedback_log`` 因果链可追溯性
    - 降级触发：连续 3 次 error 后 ``_real_machine_degraded=True``
    - 预算上限：``max_real_submissions=1`` 时第二次提交被拒绝
    - 端到端 step()：通过 ``route_to_machine`` 触发真机提交
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

# 模块级标记：cqlib 回放测试，有 fixtures 时无需 SDK 即可运行
pytestmark = pytest.mark.cqlib_replay

from src.api.cqlib_recorder import CqlibReplayClient
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv
from src.scheduler.env_real_machine import (
    poll_pending_real_tasks,
    submit_to_real_machine,
)
from src.scheduler.env_types import (
    REAL_FEEDBACK_RESULT_AWARE,
    REAL_FEEDBACK_SHUFFLED,
    REAL_FEEDBACK_STATUS_ONLY,
    REAL_MACHINE_DEGRADE_FAIL_THRESHOLD,
    REAL_MACHINE_FAIL_PENALTY,
    REAL_MACHINE_SUCCESS_BONUS,
    REAL_RESULT_REWARD_MAX,
    REAL_RESULT_REWARD_MIN,
    QuantumMachine,
    Task,
)

FIXTURES_DIR = str(Path(__file__).parent / "fixtures" / "cqlib_responses")

# 默认机器名（fixtures 中 status=running）
_DEFAULT_MACHINE = "tianyan_s"


# ============================================================
# 辅助：构造接好 CqlibReplayClient 的 QuantumSchedulingEnv
# ============================================================


def _make_env(
    *,
    feedback_mode: str = REAL_FEEDBACK_STATUS_ONLY,
    feedback_weight: float = 1.0,
    max_real_submissions: int | None = None,
    real_submit_probability: float = 0.0,
    real_submit_interval: int = 1,
    shots: int = 1024,
    machine_name: str = _DEFAULT_MACHINE,
) -> tuple[QuantumSchedulingEnv, CqlibReplayClient]:
    """构造一个已 attach CqlibReplayClient 的调度环境。

    Args:
        feedback_mode: 真机反馈模式
        feedback_weight: 反馈权重
        max_real_submissions: 真机提交预算上限
        real_submit_probability: 概率触发概率
        real_submit_interval: 间隔触发步数
        shots: 真机 shots
        machine_name: 绑定的机器名

    Returns:
        (env, client) 元组
    """
    # 构造机器配置：仅包含指定的真机，避免其他机器参与调度干扰断言
    machine_configs = [
        {
            "name": machine_name,
            "total_qubits": 66,
            "supported_gates": ("H", "CZ", "M"),
            "is_real": True,
        }
    ]
    env = QuantumSchedulingEnv(
        machine_configs=machine_configs,
        seed=42,
        use_real_machine=True,
        real_machine_feedback_weight=feedback_weight,
        max_real_submissions=max_real_submissions,
        real_machine_shots=shots,
        real_feedback_mode=feedback_mode,
        real_submit_probability=real_submit_probability,
        real_submit_interval=real_submit_interval,
    )
    client = CqlibReplayClient(FIXTURES_DIR, machine_name=machine_name)
    env.attach_real_clients({machine_name: client})
    env.reset(seed=42)
    return env, client


def _make_quantum_task(task_id: str = "qt1", qubit_count: int = 1, priority: int = 3) -> Task:
    """构造一个量子任务（单比特 H 门电路，匹配 fixtures 的 H Q0\\nM Q0）。"""
    return Task(
        task_id=task_id,
        task_type="quantum",
        qubit_count=qubit_count,
        priority=priority,
        qcis="H Q0\nM Q0",
    )


# ============================================================
# 1. 正常闭环：提交 → running → completed
# ============================================================


class TestNormalClosedLoop:
    """验证 CqlibReplayClient 驱动的正常真机闭环。"""

    def test_submit_registers_pending(self) -> None:
        """提交后 pending 列表应包含任务记录。"""
        env, _client = _make_env()
        machine = env._machines[0]
        task = _make_quantum_task()

        submit_to_real_machine(env, machine, task)

        assert len(env._pending_real_tasks) == 1
        pending = env._pending_real_tasks[0]
        assert pending["task_id_str"] == "qt1"
        assert pending["machine_name"] == _DEFAULT_MACHINE
        assert pending["qcis_circuit"] == "H Q0\nM Q0"
        # 提交计数递增
        assert env._real_submission_attempts_total == 1
        assert env._machine_real_submits[_DEFAULT_MACHINE] == 1

    def test_first_poll_returns_zero_feedback(self) -> None:
        """首次轮询 CqlibReplayClient 返回 running，无 reward 反馈。"""
        env, _client = _make_env()
        machine = env._machines[0]
        task = _make_quantum_task()
        submit_to_real_machine(env, machine, task)

        feedback = poll_pending_real_tasks(env)

        # running 状态无 reward，任务仍保留在 pending
        assert feedback == 0.0
        assert len(env._pending_real_tasks) == 1
        assert env._pending_real_tasks[0]["poll_count"] == 1

    def test_second_poll_completes_and_awards_bonus(self) -> None:
        """第二次轮询返回 completed，应发放 status_only 固定 bonus。"""
        env, _client = _make_env(feedback_mode=REAL_FEEDBACK_STATUS_ONLY)
        machine = env._machines[0]
        task = _make_quantum_task()
        submit_to_real_machine(env, machine, task)

        poll_pending_real_tasks(env)  # 第 1 次：running
        feedback = poll_pending_real_tasks(env)  # 第 2 次：completed

        assert feedback == REAL_MACHINE_SUCCESS_BONUS
        assert env._real_success_count == 1
        assert env._real_consecutive_failures == 0
        assert len(env._pending_real_tasks) == 0  # 完成后从 pending 移除

    def test_full_closed_loop_lifecycle(self) -> None:
        """完整生命周期：提交 → running → completed，验证统计计数。"""
        env, _client = _make_env()
        machine = env._machines[0]
        task = _make_quantum_task()

        # 提交
        submit_to_real_machine(env, machine, task)
        assert env._real_submission_attempts_total == 1

        # 轮询 running
        fb1 = poll_pending_real_tasks(env)
        assert fb1 == 0.0
        assert len(env._pending_real_tasks) == 1

        # 轮询 completed
        fb2 = poll_pending_real_tasks(env)
        assert fb2 == REAL_MACHINE_SUCCESS_BONUS

        # 验证统计
        stats = env.get_real_machine_stats()
        assert stats["success_count"] == 1
        assert stats["fail_count"] == 0
        assert stats["pending_count"] == 0
        assert stats["submission_attempts_total"] == 1


# ============================================================
# 2. 三种反馈模式
# ============================================================


class TestFeedbackModes:
    """验证 status_only / result_aware / shuffled 三种反馈模式的 reward 计算。"""

    def test_status_only_returns_fixed_bonus(self) -> None:
        """status_only 模式：固定 bonus，不计算保真度。"""
        env, _client = _make_env(feedback_mode=REAL_FEEDBACK_STATUS_ONLY)
        machine = env._machines[0]
        submit_to_real_machine(env, machine, _make_quantum_task())

        poll_pending_real_tasks(env)  # running
        feedback = poll_pending_real_tasks(env)  # completed

        assert feedback == REAL_MACHINE_SUCCESS_BONUS
        # _real_result_records 中 fidelity 应为 -1（未计算）
        records = env._real_result_records
        assert len(records) == 1
        assert records[0]["fidelity"] is None
        assert records[0]["fallback_mode"] is True

    def test_result_aware_parses_measurement(self) -> None:
        """result_aware 模式：正确解析 fixture 中直接包含概率分布的 result 字段。

        fixtures 的 ``result`` 字段直接是概率分布 (``{"0": 0.509, "1": 0.491}``)，
        parse_measurement_result 路径4 应支持这种格式，正确计算保真度和奖励。
        H 门理论分布 {0:0.5, 1:0.5}，测量分布接近理论值，保真度应接近 1.0。
        """
        env, _client = _make_env(feedback_mode=REAL_FEEDBACK_RESULT_AWARE)
        machine = env._machines[0]
        submit_to_real_machine(env, machine, _make_quantum_task())

        poll_pending_real_tasks(env)  # running
        feedback = poll_pending_real_tasks(env)  # completed

        # H 门测量分布接近理论值，奖励应为高保真度奖励
        assert feedback > REAL_RESULT_REWARD_MIN
        assert feedback <= REAL_RESULT_REWARD_MAX
        records = env._real_result_records
        assert len(records) == 1
        assert records[0]["fidelity"] is not None
        assert records[0]["fidelity"] > 0.99
        assert records[0]["result_valid"] is True
        assert records[0]["fallback_mode"] is False
        log = env._real_feedback_log[0]
        assert "measurement_parse_failed" not in log["formula"]

    def test_result_aware_with_inline_probability(self) -> None:
        """result_aware 模式 + 顶层 probability 字段：正确计算保真度。

        构造一个带顶层 ``probability`` 字段的完成状态，验证 result_aware
        模式能正确解析并计算 H 门保真度（应接近 1.0）。
        """
        env, _client = _make_env(feedback_mode=REAL_FEEDBACK_RESULT_AWARE)
        machine = env._machines[0]
        submit_to_real_machine(env, machine, _make_quantum_task())

        poll_pending_real_tasks(env)  # running

        # 人工注入一个带顶层 probability 的完成状态（模拟理想 fixture）
        original_status = env._pending_real_tasks[0]
        completed_with_prob = {
            "status": "completed",
            "task_id": original_status["task_id"],
            "probability": {"0": 0.509, "1": 0.491},
            "execution_time_s": 2.3,
        }
        original_method = env._real_clients[_DEFAULT_MACHINE].get_task_status
        env._real_clients[_DEFAULT_MACHINE].get_task_status = lambda _tid: completed_with_prob
        try:
            feedback = poll_pending_real_tasks(env)
        finally:
            env._real_clients[_DEFAULT_MACHINE].get_task_status = original_method

        # H 门理论 {0:0.5, 1:0.5}，测量 {0:0.509, 1:0.491}，保真度 ≈ 0.9999
        assert feedback > REAL_RESULT_REWARD_MIN
        assert feedback <= REAL_RESULT_REWARD_MAX

        records = env._real_result_records
        assert len(records) == 1
        assert records[0]["fidelity"] is not None
        assert records[0]["fidelity"] > 0.99  # H 门保真度应接近 1
        assert records[0]["result_valid"] is True
        assert records[0]["fallback_mode"] is False

    def test_shuffled_mode_breaks_alignment(self) -> None:
        """shuffled 模式：打乱测量结果，破坏语义关联。

        单比特 H 门只有 2 个结果，打乱后可能恰好与原分布相同（50% 概率），
        因此不强制要求 fidelity 不同，但应正常返回 reward。
        """
        env, _client = _make_env(feedback_mode=REAL_FEEDBACK_SHUFFLED)
        machine = env._machines[0]
        submit_to_real_machine(env, machine, _make_quantum_task())

        poll_pending_real_tasks(env)  # running
        feedback = poll_pending_real_tasks(env)  # completed

        # shuffled 仍应计算 reward（即使打乱后恰好相同）
        assert REAL_RESULT_REWARD_MIN <= feedback <= REAL_RESULT_REWARD_MAX

        # 公式应标注 [SHUFFLED]
        log = env._real_feedback_log
        assert len(log) == 1
        assert "[SHUFFLED]" in log[0]["formula"]


# ============================================================
# 3. 因果链可追溯性
# ============================================================


class TestCausalTraceability:
    """验证 _real_result_records 与 _real_feedback_log 的完整性。"""

    def test_result_record_contains_required_fields(self) -> None:
        """完成结果记录应包含 Issue #235 要求的所有字段。"""
        env, _client = _make_env(feedback_mode=REAL_FEEDBACK_RESULT_AWARE)
        submit_to_real_machine(env, env._machines[0], _make_quantum_task())

        poll_pending_real_tasks(env)  # running
        poll_pending_real_tasks(env)  # completed

        record = env._real_result_records[0]
        required_fields = {
            "task_id",
            "real_task_id",
            "machine",
            "submit_step",
            "complete_step",
            "shots",
            "backend",
            "feedback_mode",
            "probability",
            "fidelity",
            "reward_delta",
            "formula",
            "result_valid",
            "fallback_mode",
        }
        assert required_fields.issubset(record.keys())
        assert record["task_id"] == "qt1"
        assert record["machine"] == _DEFAULT_MACHINE
        assert record["shots"] == 1024
        assert record["feedback_mode"] == REAL_FEEDBACK_RESULT_AWARE

    def test_feedback_log_contains_rl_context(self) -> None:
        """因果记录应包含 RL 动作上下文（rl_action/rl_action_prob/observation_snapshot）。"""
        env, _client = _make_env()
        obs_snapshot = {"queue_length": 5, "qubit_avail": 0.8}
        submit_to_real_machine(
            env,
            env._machines[0],
            _make_quantum_task(),
            rl_action=1,  # ACTION_QUANTUM
            rl_action_prob=0.7,
            observation_snapshot=obs_snapshot,
        )

        poll_pending_real_tasks(env)  # running
        poll_pending_real_tasks(env)  # completed

        log = env._real_feedback_log[0]
        assert log["rl_action"] == 1
        assert log["rl_action_prob"] == 0.7
        assert log["observation_snapshot"] == obs_snapshot
        assert log["outcome"] == "completed"
        assert log["qcis_circuit"] == "H Q0\nM Q0"
        assert "machine_score" in log

    def test_failed_submission_logs_failure_outcome(self) -> None:
        """失败结果应记录 outcome='failed'。"""
        env, _client = _make_env()
        submit_to_real_machine(env, env._machines[0], _make_quantum_task())

        # 通过人工修改 pending 的 machine_name 让客户端丢失，触发失败路径
        env._pending_real_tasks[0]["machine_name"] = "non_existent_machine"

        feedback = poll_pending_real_tasks(env)

        assert feedback == REAL_MACHINE_FAIL_PENALTY
        log = env._real_feedback_log[0]
        assert log["outcome"] == "failed"
        assert log["reward"] == REAL_MACHINE_FAIL_PENALTY


# ============================================================
# 4. 降级机制
# ============================================================


class TestDegradeMechanism:
    """验证连续失败触发降级。"""

    def test_consecutive_errors_trigger_degrade(self) -> None:
        """连续 N 次 error 提交后触发降级（N=REAL_MACHINE_DEGRADE_FAIL_THRESHOLD）。

        使用 ``error_mode='capacity'`` 让 CqlibReplayClient 直接返回 None，
        触发 submit_to_real_machine 的 record_real_failure 路径。
        """
        # 使用 capacity 错误模式：每次提交都被拒绝
        machine_configs = [
            {
                "name": _DEFAULT_MACHINE,
                "total_qubits": 66,
                "supported_gates": ("H", "CZ", "M"),
                "is_real": True,
            }
        ]
        env = QuantumSchedulingEnv(
            machine_configs=machine_configs,
            seed=42,
            use_real_machine=True,
        )
        client = CqlibReplayClient(
            FIXTURES_DIR, machine_name=_DEFAULT_MACHINE, error_mode="capacity"
        )
        env.attach_real_clients({_DEFAULT_MACHINE: client})
        env.reset(seed=42)

        machine = env._machines[0]
        task = _make_quantum_task()

        # 连续提交 N 次，每次都被拒绝
        for i in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD):
            submit_to_real_machine(env, machine, task)
            assert env._real_fail_count == i + 1

        # 第 N 次后应触发降级
        assert env._real_machine_degraded is True
        assert env.is_real_machine_degraded() is True

    def test_degraded_env_skips_submission(self) -> None:
        """降级后的环境不再消耗机时。"""
        env, _client = _make_env()
        env._real_machine_degraded = True  # 人工标记降级

        before = env._real_submission_attempts_total
        submit_to_real_machine(env, env._machines[0], _make_quantum_task())

        assert env._real_submission_attempts_total == before
        assert len(env._pending_real_tasks) == 0


# ============================================================
# 5. 预算上限
# ============================================================


class TestBudgetLimit:
    """验证 max_real_submissions 预算上限保护。"""

    def test_budget_exhausted_skips_submission(self) -> None:
        """预算耗尽后不再提交。"""
        env, _client = _make_env(max_real_submissions=1)

        # 第一次提交成功
        submit_to_real_machine(env, env._machines[0], _make_quantum_task())
        assert env._real_submission_attempts_total == 1
        assert len(env._pending_real_tasks) == 1

        # 第二次提交应被跳过
        submit_to_real_machine(env, env._machines[0], _make_quantum_task(task_id="qt2"))
        assert env._real_submission_attempts_total == 1  # 未递增
        assert len(env._pending_real_tasks) == 1  # 未新增


# ============================================================
# 6. 端到端 step() 集成
# ============================================================


class TestStepIntegration:
    """通过 env.step() 触发真机提交的端到端集成测试。"""

    def test_route_to_machine_triggers_real_submission(self) -> None:
        """route_to_machine 在间隔触发步应调用 submit_to_real_machine。

        配置 real_submit_interval=1，使每步都触发间隔提交。
        但 step() 内部需要 _current_task 非空才会路由，
        我们直接调用 _route_to_machine 模拟路由。
        """
        env, _client = _make_env(real_submit_interval=1, real_submit_probability=0.0)
        machine = env._machines[0]
        task = _make_quantum_task()
        env._current_task = task
        env._current_step = 1  # % 1 == 0，间隔触发

        import numpy as np

        rng = np.random.default_rng(42)
        from src.scheduler.env_machines import route_to_machine

        route_to_machine(env, machine, task, rng, rl_action=1)

        # 应触发真机提交
        assert env._real_submission_attempts_total == 1
        assert len(env._pending_real_tasks) == 1

    def test_step_polls_pending_and_adds_reward(self) -> None:
        """step() 在 use_real_machine=True 时应轮询 pending 并将 reward 计入总奖励。

        准备一个已提交的 pending 任务，然后调用 step()，
        step() 应自动调用 _poll_pending_real_tasks。
        """
        env, _client = _make_env(real_submit_interval=999)  # 避免新提交干扰
        submit_to_real_machine(env, env._machines[0], _make_quantum_task())

        # 第一次 step：polling 返回 running，reward=0
        # 但 step() 需要 _current_task，设为 None 让其走"无任务"分支
        env._current_task = None
        env._current_step = 0  # 让 step 内部 +1 后为 1
        env.step(0)

        # running 状态无真机反馈
        assert len(env._pending_real_tasks) == 1

        # 第二次 step：polling 返回 completed，应发放 bonus
        env._current_task = None
        env.step(0)

        # 完成后应计入 reward（bonus * feedback_weight）
        # 注意：step() 总 reward 还包括队列惩罚等，所以只断言差值方向
        assert env._real_success_count == 1
        assert len(env._pending_real_tasks) == 0


# ============================================================
# 7. 多任务并发轮询
# ============================================================


class TestMultiplePendingTasks:
    """验证多个 pending 任务并发轮询的场景。"""

    def test_multiple_submissions_share_polling(self) -> None:
        """多个 pending 任务通过轮转队列逐步完成。

        注意：CqlibReplayClient 的 ``_task_submit`` fixture 只有一个 task_id，
        且 ``_task_polls`` 按 task_id 计数。两次提交返回相同 task_id 时，
        第二次提交会重置计数器。在轮转轮询模式下（max_poll_per_step=1），
        每次 poll 只轮询队列头部的一个任务，未完成的任务移到队尾等待后续轮询。

        轮询时序：
            - 第1次poll：轮询qt1 → 第1次查询 → running（移到队尾）
            - 第2次poll：轮询qt2 → 第2次查询 → completed（发放bonus，移除）
            - 第3次poll：轮询qt1 → 第3次查询 → completed（发放bonus，移除）
        """
        env, _client = _make_env()
        machine = env._machines[0]

        submit_to_real_machine(env, machine, _make_quantum_task(task_id="qt1"))
        submit_to_real_machine(env, machine, _make_quantum_task(task_id="qt2"))

        assert len(env._pending_real_tasks) == 2

        # 第1次poll：轮询qt1 → running，无反馈，qt1移到队尾
        feedback = poll_pending_real_tasks(env)
        assert feedback == 0.0
        assert env._real_success_count == 0
        assert len(env._pending_real_tasks) == 2

        # 第2次poll：轮询qt2 → completed，发放bonus，qt2移除
        feedback = poll_pending_real_tasks(env)
        assert feedback == REAL_MACHINE_SUCCESS_BONUS
        assert env._real_success_count == 1
        assert len(env._pending_real_tasks) == 1
        assert env._pending_real_tasks[0]["task_id_str"] == "qt1"

        # 第3次poll：轮询qt1 → completed，发放bonus，qt1移除
        feedback2 = poll_pending_real_tasks(env)
        assert feedback2 == REAL_MACHINE_SUCCESS_BONUS
        assert env._real_success_count == 2
        assert len(env._pending_real_tasks) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
