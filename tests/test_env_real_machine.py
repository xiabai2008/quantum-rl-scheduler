"""
量子RL调度系统 - 真机闭环模块单元测试
Unit Tests for Real-Machine Closed-Loop Module

测试覆盖：
- submit_to_real_machine : 向真机提交任务（降级跳过/无client/正常/返回None/异常）
- poll_pending_real_tasks : 轮询已提交真机任务（空/完成/错误/超时/运行中/client丢失/查询异常）
- record_real_failure     : 记录失败并触发降级（累加计数/触发降级/不重复降级）

目标：将 env_real_machine.py 覆盖率从 76% 提升到 80%+。
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.env_real_machine import (
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


def _build_mock_env(use_real_machine: bool = True) -> MagicMock:
    """
    构造一个具备真机闭环所需全部内部状态的 mock 环境。

    使用 MagicMock 而非真实 QuantumSchedulingEnv，可以精确控制每个
    分支的前置状态（如 degraded、pending 列表、client 行为），从而
    独立验证 env_real_machine.py 中每条路径。

    Args:
        use_real_machine: 是否启用真机模式（影响日志输出分支）

    Returns:
        配置好全部真机属性的 MagicMock 环境对象
    """
    env = MagicMock()
    # 真机客户端映射：machine_name -> client
    env._real_clients = {}
    # 已提交但未拿到结果的真机任务列表
    env._pending_real_tasks = []
    # 降级标志与失败计数
    env._real_machine_degraded = False
    env._real_consecutive_failures = 0
    env._real_fail_count = 0
    env._real_success_count = 0
    # 各机器真机提交计数
    env._machine_real_submits = {}
    # 当前步数（用于登记 submit_step）
    env._current_step = 5
    # 渲染日志缓冲区
    env._render_log = []
    # 真机模式开关与反馈权重
    env.use_real_machine = use_real_machine
    env.real_machine_feedback_weight = 1.0
    return env


def _build_machine(name: str = "tianyan_s") -> QuantumMachine:
    """
    构造一个测试用 QuantumMachine。

    Args:
        name: 机器名称

    Returns:
        QuantumMachine 实例
    """
    return QuantumMachine(name=name, total_qubits=287, is_real=True)


def _build_task(task_id: str = "T001") -> Task:
    """
    构造一个测试用 Task。

    Args:
        task_id: 任务标识符

    Returns:
        Task 实例
    """
    return Task(
        task_id=task_id,
        task_type="quantum",
        qubit_count=2,
        wait_steps=1,
        urgency=0.5,
        priority=3,
        execution_time=3,
    )


class TestSubmitToRealMachine(unittest.TestCase):
    """测试 submit_to_real_machine 函数。"""

    def test_submit_degraded_skips(self):
        """降级标志为 True 时应直接返回，不进行任何提交。"""
        env = _build_mock_env()
        env._real_machine_degraded = True
        machine = _build_machine()
        task = _build_task()
        client = MagicMock()
        env._real_clients[machine.name] = client

        submit_to_real_machine(env, machine, task)

        # 提交方法不应被调用，pending 列表保持为空
        client.submit_quantum_task.assert_not_called()
        self.assertEqual(env._pending_real_tasks, [])
        self.assertEqual(env._machine_real_submits, {})

    def test_submit_no_client_skips(self):
        """无对应 client 时应直接返回（覆盖 line 60）。"""
        env = _build_mock_env()
        machine = _build_machine()
        task = _build_task()
        # 不绑定任何 client

        submit_to_real_machine(env, machine, task)

        # pending 列表保持为空，提交计数未增加
        self.assertEqual(env._pending_real_tasks, [])
        self.assertEqual(env._machine_real_submits, {})

    def test_submit_success_adds_to_pending(self):
        """正常提交成功时应登记到 pending 列表并增加提交计数。"""
        env = _build_mock_env()
        machine = _build_machine()
        task = _build_task()
        client = MagicMock()
        client.submit_quantum_task.return_value = "REAL_001"
        env._real_clients[machine.name] = client

        submit_to_real_machine(env, machine, task)

        # 提交计数 +1
        self.assertEqual(env._machine_real_submits[machine.name], 1)
        # pending 列表新增一条记录
        self.assertEqual(len(env._pending_real_tasks), 1)
        record = env._pending_real_tasks[0]
        self.assertEqual(record["task_id"], "REAL_001")
        self.assertEqual(record["machine_name"], machine.name)
        self.assertEqual(record["submit_step"], env._current_step)
        self.assertEqual(record["poll_count"], 0)
        self.assertEqual(record["task_id_str"], task.task_id)

    def test_submit_success_with_custom_qcis(self):
        """任务自带 qcis 属性时应使用任务的 qcis 而非占位电路。"""
        env = _build_mock_env()
        machine = _build_machine()
        task = _build_task()
        task.qcis = "X Q0\nM Q0"  # 自定义 QCIS 电路
        client = MagicMock()
        client.submit_quantum_task.return_value = "REAL_002"
        env._real_clients[machine.name] = client

        submit_to_real_machine(env, machine, task)

        # 验证提交时使用的是任务自带的 qcis
        client.submit_quantum_task.assert_called_once()
        call_kwargs = client.submit_quantum_task.call_args.kwargs
        self.assertEqual(call_kwargs["qcis"], "X Q0\nM Q0")
        self.assertEqual(call_kwargs["shots"], 512)
        self.assertEqual(len(env._pending_real_tasks), 1)

    def test_submit_success_when_use_real_machine_false(self):
        """use_real_machine=False 时提交仍应成功登记（仅跳过 debug 日志分支）。"""
        env = _build_mock_env(use_real_machine=False)
        machine = _build_machine()
        task = _build_task()
        client = MagicMock()
        client.submit_quantum_task.return_value = "REAL_003"
        env._real_clients[machine.name] = client

        submit_to_real_machine(env, machine, task)

        # 即使不启用真机模式，提交逻辑仍应正常执行
        self.assertEqual(len(env._pending_real_tasks), 1)
        self.assertEqual(env._machine_real_submits[machine.name], 1)

    def test_submit_returns_none_records_failure(self):
        """提交返回 None（被拒绝）时应调用 record_real_failure（覆盖 line 92）。"""
        env = _build_mock_env()
        machine = _build_machine()
        task = _build_task()
        client = MagicMock()
        client.submit_quantum_task.return_value = None  # 提交被拒绝
        env._real_clients[machine.name] = client

        submit_to_real_machine(env, machine, task)

        # 提交计数仍应 +1（提交动作已发生）
        self.assertEqual(env._machine_real_submits[machine.name], 1)
        # pending 列表为空（被拒绝不登记）
        self.assertEqual(env._pending_real_tasks, [])
        # 失败计数 +1
        self.assertEqual(env._real_fail_count, 1)
        self.assertEqual(env._real_consecutive_failures, 1)

    def test_submit_exception_records_failure(self):
        """提交抛出异常时应记录日志并调用 record_real_failure（覆盖 line 93-97）。"""
        env = _build_mock_env()
        machine = _build_machine()
        task = _build_task()
        client = MagicMock()
        client.submit_quantum_task.side_effect = RuntimeError("网络超时")
        env._real_clients[machine.name] = client

        submit_to_real_machine(env, machine, task)

        # 异常被捕获，pending 列表为空
        self.assertEqual(env._pending_real_tasks, [])
        # 异常信息应写入渲染日志
        self.assertTrue(
            any("提交失败" in log for log in env._render_log),
            "渲染日志应包含提交失败信息",
        )
        # 失败计数 +1
        self.assertEqual(env._real_fail_count, 1)
        self.assertEqual(env._real_consecutive_failures, 1)

    def test_submit_exception_truncates_long_message(self):
        """异常消息过长时应被截断到 60 字符以内再写入日志。"""
        env = _build_mock_env()
        machine = _build_machine()
        task = _build_task()
        client = MagicMock()
        long_msg = "X" * 200  # 超长异常消息
        client.submit_quantum_task.side_effect = ValueError(long_msg)
        env._real_clients[machine.name] = client

        submit_to_real_machine(env, machine, task)

        # 验证写入日志的消息被截断
        failure_log = [log for log in env._render_log if "提交失败" in log]
        self.assertEqual(len(failure_log), 1)
        # 日志中包含的消息部分应被截断（"提交失败: " 后的内容 <= 60 字符）
        self.assertIn("提交失败", failure_log[0])


class TestPollPendingRealTasks(unittest.TestCase):
    """测试 poll_pending_real_tasks 函数。"""

    def test_poll_empty_returns_zero(self):
        """无待处理任务时应返回 0.0（覆盖 line 150-151）。"""
        env = _build_mock_env()
        env._pending_real_tasks = []

        feedback = poll_pending_real_tasks(env)

        self.assertEqual(feedback, 0.0)

    def test_poll_completed_success(self):
        """任务状态为 completed 时应返回成功奖励并重置连续失败计数。"""
        env = _build_mock_env()
        env._real_consecutive_failures = 2  # 预置一些失败计数
        client = MagicMock()
        client.get_task_status.return_value = {"status": "completed"}
        env._real_clients["tianyan_s"] = client
        env._pending_real_tasks = [
            {
                "task_id": "REAL_001",
                "machine_name": "tianyan_s",
                "submit_step": 1,
                "poll_count": 0,
                "task_id_str": "T001",
            }
        ]

        feedback = poll_pending_real_tasks(env)

        # 返回成功奖励
        self.assertEqual(feedback, REAL_MACHINE_SUCCESS_BONUS * 1.0)
        # 成功计数 +1
        self.assertEqual(env._real_success_count, 1)
        # 连续失败计数被重置为 0
        self.assertEqual(env._real_consecutive_failures, 0)
        # 已完成的任务从 pending 列表移除
        self.assertEqual(env._pending_real_tasks, [])

    def test_poll_error_status_records_failure(self):
        """任务状态为 error 时应返回惩罚并调用 record_real_failure（覆盖 line 190-195）。"""
        env = _build_mock_env()
        client = MagicMock()
        client.get_task_status.return_value = {"status": "error"}
        env._real_clients["tianyan_s"] = client
        env._pending_real_tasks = [
            {
                "task_id": "REAL_001",
                "machine_name": "tianyan_s",
                "submit_step": 1,
                "poll_count": 0,
                "task_id_str": "T001",
            }
        ]

        feedback = poll_pending_real_tasks(env)

        # 返回失败惩罚
        self.assertEqual(feedback, REAL_MACHINE_FAIL_PENALTY * 1.0)
        # 失败计数 +1
        self.assertEqual(env._real_fail_count, 1)
        self.assertEqual(env._real_consecutive_failures, 1)
        # 失败任务从 pending 列表移除
        self.assertEqual(env._pending_real_tasks, [])

    def test_poll_timeout_records_failure(self):
        """轮询次数超过最大值时应视为超时失败（覆盖 line 196-205）。"""
        env = _build_mock_env()
        client = MagicMock()
        client.get_task_status.return_value = {"status": "running"}
        env._real_clients["tianyan_s"] = client
        # 预置 poll_count 为最大值减 1，本次轮询 +1 后达到阈值
        env._pending_real_tasks = [
            {
                "task_id": "REAL_001",
                "machine_name": "tianyan_s",
                "submit_step": 1,
                "poll_count": REAL_MACHINE_MAX_POLL_STEPS - 1,
                "task_id_str": "T001",
            }
        ]

        feedback = poll_pending_real_tasks(env)

        # 返回失败惩罚（超时视为失败）
        self.assertEqual(feedback, REAL_MACHINE_FAIL_PENALTY * 1.0)
        # 失败计数 +1
        self.assertEqual(env._real_fail_count, 1)
        # 超时任务从 pending 列表移除
        self.assertEqual(env._pending_real_tasks, [])

    def test_poll_running_keeps_pending(self):
        """任务仍在运行时应保留在 pending 列表中，返回 0 反馈。"""
        env = _build_mock_env()
        client = MagicMock()
        client.get_task_status.return_value = {"status": "running"}
        env._real_clients["tianyan_s"] = client
        env._pending_real_tasks = [
            {
                "task_id": "REAL_001",
                "machine_name": "tianyan_s",
                "submit_step": 1,
                "poll_count": 0,
                "task_id_str": "T001",
            }
        ]

        feedback = poll_pending_real_tasks(env)

        # 仍在运行，无反馈
        self.assertEqual(feedback, 0.0)
        # 任务保留在 pending 列表，poll_count +1
        self.assertEqual(len(env._pending_real_tasks), 1)
        self.assertEqual(env._pending_real_tasks[0]["poll_count"], 1)

    def test_poll_unknown_status_keeps_pending(self):
        """未知状态应保留在 pending 列表中。"""
        env = _build_mock_env()
        client = MagicMock()
        client.get_task_status.return_value = {"status": "queued"}
        env._real_clients["tianyan_s"] = client
        env._pending_real_tasks = [
            {
                "task_id": "REAL_001",
                "machine_name": "tianyan_s",
                "submit_step": 1,
                "poll_count": 0,
                "task_id_str": "T001",
            }
        ]

        feedback = poll_pending_real_tasks(env)

        # 未知状态无反馈
        self.assertEqual(feedback, 0.0)
        # 任务保留
        self.assertEqual(len(env._pending_real_tasks), 1)

    def test_poll_client_none_records_failure(self):
        """客户端丢失时应返回惩罚并调用 record_real_failure（覆盖 line 163-167）。"""
        env = _build_mock_env()
        # 不绑定 client，模拟客户端丢失
        env._pending_real_tasks = [
            {
                "task_id": "REAL_001",
                "machine_name": "tianyan_s",
                "submit_step": 1,
                "poll_count": 0,
                "task_id_str": "T001",
            }
        ]

        feedback = poll_pending_real_tasks(env)

        # 返回失败惩罚
        self.assertEqual(feedback, REAL_MACHINE_FAIL_PENALTY * 1.0)
        # 失败计数 +1
        self.assertEqual(env._real_fail_count, 1)
        self.assertEqual(env._real_consecutive_failures, 1)
        # 任务从 pending 列表移除
        self.assertEqual(env._pending_real_tasks, [])

    def test_poll_status_exception_keeps_pending(self):
        """get_task_status 抛出异常时应保留任务在 pending 列表（覆盖 line 169-175）。"""
        env = _build_mock_env()
        client = MagicMock()
        client.get_task_status.side_effect = ConnectionError("查询超时")
        env._real_clients["tianyan_s"] = client
        env._pending_real_tasks = [
            {
                "task_id": "REAL_001",
                "machine_name": "tianyan_s",
                "submit_step": 1,
                "poll_count": 0,
                "task_id_str": "T001",
            }
        ]

        feedback = poll_pending_real_tasks(env)

        # 异常视为本步未拿到结果，无反馈
        self.assertEqual(feedback, 0.0)
        # 任务保留在 pending 列表，poll_count +1
        self.assertEqual(len(env._pending_real_tasks), 1)
        self.assertEqual(env._pending_real_tasks[0]["poll_count"], 1)

    def test_poll_mixed_statuses(self):
        """多个任务混合状态时应分别处理并累加反馈。"""
        env = _build_mock_env()
        client = MagicMock()
        # 第一次返回 completed，第二次返回 running
        client.get_task_status.side_effect = [
            {"status": "completed"},
            {"status": "running"},
        ]
        env._real_clients["tianyan_s"] = client
        env._pending_real_tasks = [
            {
                "task_id": "REAL_001",
                "machine_name": "tianyan_s",
                "submit_step": 1,
                "poll_count": 0,
                "task_id_str": "T001",
            },
            {
                "task_id": "REAL_002",
                "machine_name": "tianyan_s",
                "submit_step": 2,
                "poll_count": 0,
                "task_id_str": "T002",
            },
        ]

        feedback = poll_pending_real_tasks(env)

        # 一个成功（+2.0），一个运行中（0.0）
        self.assertEqual(feedback, REAL_MACHINE_SUCCESS_BONUS * 1.0)
        # 成功的移除，运行中的保留
        self.assertEqual(len(env._pending_real_tasks), 1)
        self.assertEqual(env._pending_real_tasks[0]["task_id"], "REAL_002")
        self.assertEqual(env._real_success_count, 1)

    def test_poll_uses_feedback_weight(self):
        """反馈应乘以 real_machine_feedback_weight。"""
        env = _build_mock_env()
        env.real_machine_feedback_weight = 0.5
        client = MagicMock()
        client.get_task_status.return_value = {"status": "completed"}
        env._real_clients["tianyan_s"] = client
        env._pending_real_tasks = [
            {
                "task_id": "REAL_001",
                "machine_name": "tianyan_s",
                "submit_step": 1,
                "poll_count": 0,
                "task_id_str": "T001",
            }
        ]

        feedback = poll_pending_real_tasks(env)

        # 反馈 = 成功奖励 * 权重 0.5
        self.assertEqual(feedback, REAL_MACHINE_SUCCESS_BONUS * 0.5)


class TestRecordRealFailure(unittest.TestCase):
    """测试 record_real_failure 函数。"""

    def test_record_failure_increments_count(self):
        """记录失败应累加失败计数与连续失败计数。"""
        env = _build_mock_env()

        record_real_failure(env, "tianyan_s", "测试失败")

        self.assertEqual(env._real_fail_count, 1)
        self.assertEqual(env._real_consecutive_failures, 1)
        # 未达阈值，不应降级
        self.assertFalse(env._real_machine_degraded)

    def test_record_failure_multiple_increments(self):
        """多次记录失败应持续累加计数。"""
        env = _build_mock_env()

        record_real_failure(env, "tianyan_s", "失败1")
        record_real_failure(env, "tianyan_s", "失败2")

        self.assertEqual(env._real_fail_count, 2)
        self.assertEqual(env._real_consecutive_failures, 2)
        # 阈值为 3，2 次还未触发降级
        self.assertFalse(env._real_machine_degraded)

    def test_record_failure_triggers_degrade_at_threshold(self):
        """连续失败达到阈值时应触发降级（覆盖 line 122-129）。"""
        env = _build_mock_env()

        # 连续失败达到阈值
        for i in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD):
            record_real_failure(env, "tianyan_s", f"失败{i + 1}")

        self.assertEqual(env._real_consecutive_failures, REAL_MACHINE_DEGRADE_FAIL_THRESHOLD)
        self.assertEqual(env._real_fail_count, REAL_MACHINE_DEGRADE_FAIL_THRESHOLD)
        # 触发降级
        self.assertTrue(env._real_machine_degraded)
        # 降级日志应写入渲染日志
        self.assertTrue(
            any("降级" in log for log in env._render_log),
            "渲染日志应包含降级信息",
        )

    def test_record_failure_no_double_degrade(self):
        """已降级时再次失败不应重复触发降级日志（覆盖 line 120 的 not 条件）。"""
        env = _build_mock_env()

        # 先触发降级
        for i in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD):
            record_real_failure(env, "tianyan_s", f"失败{i + 1}")

        degrade_log_count = sum("降级" in log for log in env._render_log)

        # 再次记录失败，不应重复触发降级
        record_real_failure(env, "tianyan_s", "失败4")

        # 失败计数继续累加
        self.assertEqual(env._real_fail_count, REAL_MACHINE_DEGRADE_FAIL_THRESHOLD + 1)
        # 降级日志数量不变（不重复写入）
        self.assertEqual(sum("降级" in log for log in env._render_log), degrade_log_count)
        # 仍处于降级状态
        self.assertTrue(env._real_machine_degraded)

    def test_record_failure_below_threshold_no_degrade(self):
        """失败次数低于阈值时不应触发降级。"""
        env = _build_mock_env()

        # 只记录阈值-1次失败
        for i in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD - 1):
            record_real_failure(env, "tianyan_s", f"失败{i + 1}")

        self.assertFalse(env._real_machine_degraded)
        # 不应有降级日志
        self.assertFalse(any("降级" in log for log in env._render_log))


class TestIntegrationWithSubmitAndFailure(unittest.TestCase):
    """测试提交与失败记录的集成场景。"""

    def test_submit_none_then_degrade_after_threshold(self):
        """连续提交被拒绝达到阈值后应触发降级，后续提交被跳过。"""
        env = _build_mock_env()
        machine = _build_machine()
        client = MagicMock()
        client.submit_quantum_task.return_value = None  # 始终被拒绝
        env._real_clients[machine.name] = client

        # 连续提交被拒绝达到阈值
        for i in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD):
            task = _build_task(task_id=f"T{i:03d}")
            submit_to_real_machine(env, machine, task)

        # 触发降级
        self.assertTrue(env._real_machine_degraded)
        self.assertEqual(env._real_fail_count, REAL_MACHINE_DEGRADE_FAIL_THRESHOLD)

        # 降级后再次提交应被跳过
        client.submit_quantum_task.reset_mock()
        task_final = _build_task(task_id="T999")
        submit_to_real_machine(env, machine, task_final)
        client.submit_quantum_task.assert_not_called()

    def test_submit_exception_then_degrade_after_threshold(self):
        """连续提交异常达到阈值后应触发降级。"""
        env = _build_mock_env()
        machine = _build_machine()
        client = MagicMock()
        client.submit_quantum_task.side_effect = RuntimeError("服务端错误")
        env._real_clients[machine.name] = client

        # 连续提交异常达到阈值
        for i in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD):
            task = _build_task(task_id=f"T{i:03d}")
            submit_to_real_machine(env, machine, task)

        # 触发降级
        self.assertTrue(env._real_machine_degraded)
        # 每次异常都写入渲染日志
        self.assertEqual(
            sum("提交失败" in log for log in env._render_log),
            REAL_MACHINE_DEGRADE_FAIL_THRESHOLD,
        )


if __name__ == "__main__":
    unittest.main()
