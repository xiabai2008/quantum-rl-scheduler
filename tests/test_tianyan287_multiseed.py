"""PR #57 / Issue #58 天衍-287 多seed真机实验脚本单元测试。

测试 scripts/real_machine/tianyan287_multiseed.py 的核心修复点：
- 正确后端名为 tianyan-287（有连字符）
- tianyan287（无连字符）被拒绝
- QCIS 预校验 false 时零提交
- task_id=None 时不轮询
- probability 从 result 字段读取
- 失败/超时/query_error 不计为 completed
- 已取得 task_id 后轮询异常仍保留 task_id
- 禁止自动回退至 176/Mock/其他机器
- 提交硬上限不会超限

所有测试均为无真机单元测试，通过 mock CqlibTianyanClient 实现。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

_SCRIPT_DIR = PROJECT_ROOT / "scripts" / "real_machine"
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from tianyan287_multiseed import (
    DEFAULT_MACHINE_CONFIGS,
    HARD_LIMIT_FORMAL,
    HARD_LIMIT_SMOKE,
    HARD_LIMIT_TOTAL,
    MAX_REAL_TASKS_PER_RUN,
    QCIS_CIRCUIT,
    SEEDS,
    SHOTS,
    STRATEGIES,
    TARGET_MACHINE,
    _submit_and_poll_one_task,
    run_single_seed,
    run_smoke_test,
)

# ── 常量测试 ──


class TestConstants:
    """测试 PR #57 核心常量。"""

    def test_target_machine_has_hyphen(self) -> None:
        """Issue #58：目标机器必须为 tianyan-287（有连字符）。"""
        assert TARGET_MACHINE == "tianyan-287"
        assert "-" in TARGET_MACHINE

    def test_tianyan287_no_hyphen_not_used(self) -> None:
        """Issue #58：tianyan287（无连字符）不是有效后端代码。"""
        assert TARGET_MACHINE != "tianyan287"

    def test_qcis_circuit_uses_q1_not_q0(self) -> None:
        """Issue #58：电路必须使用 Q1（天衍-287 无 Q0）。"""
        assert "Q1" in QCIS_CIRCUIT
        assert "Q0" not in QCIS_CIRCUIT
        assert QCIS_CIRCUIT == "H Q1\nM Q1"

    def test_shots_fixed_32(self) -> None:
        """Issue #58：shots 固定为 32。"""
        assert SHOTS == 32

    def test_max_qubits_is_105_not_287(self) -> None:
        """Issue #58：max_qubits 按实时配置写 105，不得因名称写 287。"""
        tianyan_config = next(c for c in DEFAULT_MACHINE_CONFIGS if c["name"] == "tianyan-287")
        assert tianyan_config["max_qubits"] == 105
        assert tianyan_config["max_qubits"] != 287

    def test_machine_config_name_has_hyphen(self) -> None:
        """Issue #58：DEFAULT_MACHINE_CONFIGS 中机器名必须有连字符。"""
        names = [c["name"] for c in DEFAULT_MACHINE_CONFIGS]
        assert "tianyan-287" in names
        assert "tianyan287" not in names

    def test_hard_limits(self) -> None:
        """硬上限：正式 30 + 冒烟 1 = 总 31。"""
        assert HARD_LIMIT_FORMAL == 30
        assert HARD_LIMIT_SMOKE == 1
        assert HARD_LIMIT_TOTAL == 31

    def test_seeds_count_10(self) -> None:
        """10 seeds 扩展。"""
        assert len(SEEDS) == 10

    def test_strategies_count_3(self) -> None:
        """3 策略：ppo, fcfs, sjf。"""
        assert len(STRATEGIES) == 3
        assert set(STRATEGIES) == {"ppo", "fcfs", "sjf"}

    def test_max_real_tasks_per_run_is_1(self) -> None:
        """每次 run 只提交 1 个真机任务。"""
        assert MAX_REAL_TASKS_PER_RUN == 1


# ── _submit_and_poll_one_task 测试 ──


def _make_mock_client(
    qcis_check_result: bool = True,
    submit_return: str | None = "task-123",
    wait_return: dict[str, Any] | None = None,
    wait_side_effect: Exception | None = None,
) -> MagicMock:
    """构造 mock CqlibTianyanClient。"""
    client = MagicMock(spec=["submit_quantum_task", "wait_for_task", "platform"])
    client.submit_quantum_task.return_value = submit_return

    platform = MagicMock()
    platform.qcis_check_regular.return_value = qcis_check_result
    client.platform = platform

    if wait_side_effect is not None:
        client.wait_for_task.side_effect = wait_side_effect
    elif wait_return is not None:
        client.wait_for_task.return_value = wait_return
    else:
        client.wait_for_task.return_value = {
            "task_id": "task-123",
            "status": "completed",
            "result": {"0": 0.48, "1": 0.52},
        }

    return client


class TestSubmitAndPollOneTask:
    """测试 _submit_and_poll_one_task 辅助函数。"""

    def test_qcis_precheck_false_zero_submission(self) -> None:
        """QCIS 预校验 false 时零提交，不消耗机时。"""
        client = _make_mock_client(qcis_check_result=False)

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_qcis_fail",
            machine_name=TARGET_MACHINE,
        )

        assert record["status"] == "failed"
        assert "QCIS 预校验失败" in record["error"]
        assert record["task_id"] is None
        # 关键：submit_quantum_task 不应被调用
        client.submit_quantum_task.assert_not_called()
        client.wait_for_task.assert_not_called()

    def test_qcis_precheck_exception_zero_submission(self) -> None:
        """QCIS 预校验抛异常时标记 query_error，零提交。"""
        client = _make_mock_client()
        client.platform.qcis_check_regular.side_effect = RuntimeError("SDK error")

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_qcis_exc",
            machine_name=TARGET_MACHINE,
        )

        assert record["status"] == "query_error"
        assert "QCIS 校验异常" in record["error"]
        client.submit_quantum_task.assert_not_called()

    def test_task_id_none_immediate_fail_no_polling(self) -> None:
        """task_id 为 None 时立即失败，不调用 wait_for_task(None)。"""
        client = _make_mock_client(submit_return=None)

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_none_id",
            machine_name=TARGET_MACHINE,
        )

        assert record["status"] == "failed"
        assert "返回 None" in record["error"]
        assert record["task_id"] is None
        # 关键：wait_for_task 绝不能以 None 调用
        client.wait_for_task.assert_not_called()

    def test_task_id_empty_string_immediate_fail(self) -> None:
        """task_id 为空字符串时也立即失败。"""
        client = _make_mock_client(submit_return="")

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_empty_id",
            machine_name=TARGET_MACHINE,
        )

        assert record["status"] == "failed"
        client.wait_for_task.assert_not_called()

    def test_probability_read_from_result_field(self) -> None:
        """probability 必须从 poll_result['result'] 读取。"""
        client = _make_mock_client(
            submit_return="task-456",
            wait_return={
                "task_id": "task-456",
                "status": "completed",
                "result": {"0": 0.3, "1": 0.7},
            },
        )

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_prob_result",
            machine_name=TARGET_MACHINE,
        )

        assert record["status"] == "completed"
        assert record["probability"] == {"0": 0.3, "1": 0.7}
        assert record["task_id"] == "task-456"
        assert record["measurement_balance_score"] is not None

    def test_probability_fallback_to_probability_field(self) -> None:
        """兼容：result 字段为空时回退到 probability 字段。"""
        client = _make_mock_client(
            wait_return={
                "task_id": "task-789",
                "status": "completed",
                "probability": {"0": 0.4, "1": 0.6},
            }
        )

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_prob_fallback",
            machine_name=TARGET_MACHINE,
        )

        assert record["status"] == "completed"
        assert record["probability"] == {"0": 0.4, "1": 0.6}

    def test_probability_json_string_parsed_correctly(self) -> None:
        """cqlib SDK 可能返回 JSON 字符串，必须解析后才能用于 score 计算。

        真机冒烟 task_id=2079822848580653058 发现此问题：
        probability 字段为 '{"0":0.3946,"1":0.6054}'（字符串），
        导致 isinstance(prob, dict) 返回 False，score 未计算。
        """
        client = _make_mock_client(
            submit_return="task-json-001",
            wait_return={
                "task_id": "task-json-001",
                "status": "completed",
                "result": '{"0": 0.3946, "1": 0.6054}',
            },
        )

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_json_str",
            machine_name=TARGET_MACHINE,
        )

        assert record["status"] == "completed"
        # probability 应被解析为 dict
        assert isinstance(record["probability"], dict)
        assert record["probability"] == {"0": 0.3946, "1": 0.6054}
        # score 应被正确计算
        assert record["measurement_balance_score"] is not None
        # 0.3946/0.6054 → score = 1.0 - |0.3946-0.5| - |0.6054-0.5| = 0.7892
        assert record["measurement_balance_score"] == pytest.approx(0.7892, abs=0.001)

    def test_probability_invalid_json_string_preserved(self) -> None:
        """无效 JSON 字符串应保留原始值，score 不计算。"""
        client = _make_mock_client(
            submit_return="task-bad-json",
            wait_return={
                "task_id": "task-bad-json",
                "status": "completed",
                "result": "not a json string",
            },
        )

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_bad_json",
            machine_name=TARGET_MACHINE,
        )

        assert record["status"] == "completed"
        assert record["probability"] == "not a json string"
        assert record["measurement_balance_score"] is None

    def test_failed_status_not_completed(self) -> None:
        """status=failed 不计为 completed。"""
        client = _make_mock_client(
            wait_return={
                "task_id": "task-fail",
                "status": "failed",
                "error": "machine error",
            }
        )

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_failed",
            machine_name=TARGET_MACHINE,
        )

        assert record["status"] == "failed"
        assert record["probability"] is None
        assert record["measurement_balance_score"] is None

    def test_timeout_status_not_completed(self) -> None:
        """status=timeout 不计为 completed。"""
        client = _make_mock_client(wait_return={"task_id": "task-timeout", "status": "timeout"})

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_timeout",
            machine_name=TARGET_MACHINE,
        )

        assert record["status"] == "timeout"
        assert record["probability"] is None

    def test_query_error_not_completed(self) -> None:
        """status=query_error 不计为 completed。"""
        client = _make_mock_client(
            wait_return={
                "task_id": "task-qerr",
                "status": "query_error",
                "error": "SDK connection lost",
            }
        )

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_qerr",
            machine_name=TARGET_MACHINE,
        )

        assert record["status"] == "query_error"
        assert record["probability"] is None
        assert record["measurement_balance_score"] is None

    def test_task_id_preserved_after_poll_exception(self) -> None:
        """已取得 task_id 后轮询异常仍保留 task_id。"""
        client = _make_mock_client(
            submit_return="task-preserve-001",
            wait_side_effect=RuntimeError("network timeout"),
        )

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_preserve",
            machine_name=TARGET_MACHINE,
        )

        # 关键：task_id 必须保留
        assert record["task_id"] == "task-preserve-001"
        assert record["status"] == "query_error"
        assert record["submitted_at"] is not None
        assert "network timeout" in record["error"]

    def test_submit_exception_records_failed(self) -> None:
        """submit_quantum_task 抛异常时标记 failed。"""
        client = _make_mock_client()
        client.submit_quantum_task.side_effect = RuntimeError("quota exceeded")

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_submit_exc",
            machine_name=TARGET_MACHINE,
        )

        assert record["status"] == "failed"
        assert "提交异常" in record["error"]
        assert record["task_id"] is None

    def test_measurement_balance_score_calculation(self) -> None:
        """measurement_balance_score 应正确计算（H 态 50/50 分布）。"""
        # 完美 50/50 分布 → score=1.0
        client = _make_mock_client(
            wait_return={
                "task_id": "task-perfect",
                "status": "completed",
                "result": {"0": 0.5, "1": 0.5},
            }
        )

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_score",
            machine_name=TARGET_MACHINE,
        )

        assert record["measurement_balance_score"] == pytest.approx(1.0)

    def test_measurement_balance_score_imbalanced(self) -> None:
        """极端不平衡分布 → score 较低。"""
        client = _make_mock_client(
            wait_return={
                "task_id": "task-imbalance",
                "status": "completed",
                "result": {"0": 0.9, "1": 0.1},
            }
        )

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_imbalance",
            machine_name=TARGET_MACHINE,
        )

        # 0.9/0.1 → score = 1.0 - |0.9-0.5| - |0.1-0.5| = 1.0 - 0.4 - 0.4 = 0.2
        assert record["measurement_balance_score"] == pytest.approx(0.2)

    def test_mock_and_degraded_always_false(self) -> None:
        """record 中 mock 和 degraded 必须为 False。"""
        client = _make_mock_client()

        record = _submit_and_poll_one_task(
            client=client,
            qcis=QCIS_CIRCUIT,
            shots=SHOTS,
            task_name="test_mock_flag",
            machine_name=TARGET_MACHINE,
        )

        assert record["mock"] is False
        assert record["degraded"] is False


# ── run_smoke_test 测试 ──


class TestRunSmokeTest:
    """测试 run_smoke_test 函数。"""

    def test_smoke_pass_all_conditions(self) -> None:
        """冒烟通过需同时满足所有条件。"""
        client = _make_mock_client(
            submit_return="smoke-001",
            wait_return={
                "task_id": "smoke-001",
                "status": "completed",
                "result": {"0": 0.48, "1": 0.52},
            },
        )

        result = run_smoke_test(client, TARGET_MACHINE)

        assert result["passed"] is True
        assert result["task_id"] == "smoke-001"
        assert result["status"] == "completed"
        assert isinstance(result["probability"], dict)
        assert result["mock"] is False
        assert result["degraded"] is False
        assert result["machine"] == TARGET_MACHINE

    def test_smoke_fail_wrong_machine(self) -> None:
        """后端不一致时冒烟失败。"""
        client = _make_mock_client()

        result = run_smoke_test(client, "tianyan176")

        assert result["passed"] is False
        assert "后端不一致" in result["error"]
        assert result["status"] == "failed"

    def test_smoke_fail_tianyan287_no_hyphen(self) -> None:
        """tianyan287（无连字符）后端时冒烟失败。"""
        client = _make_mock_client()

        result = run_smoke_test(client, "tianyan287")

        assert result["passed"] is False
        assert "后端不一致" in result["error"]

    def test_smoke_fail_qcis_precheck(self) -> None:
        """QCIS 预校验失败时冒烟失败。"""
        client = _make_mock_client(qcis_check_result=False)

        result = run_smoke_test(client, TARGET_MACHINE)

        assert result["passed"] is False
        assert result["task_id"] is None

    def test_smoke_fail_task_id_none(self) -> None:
        """task_id 为 None 时冒烟失败。"""
        client = _make_mock_client(submit_return=None)

        result = run_smoke_test(client, TARGET_MACHINE)

        assert result["passed"] is False
        assert result["task_id"] is None

    def test_smoke_fail_status_not_completed(self) -> None:
        """status 非 completed 时冒烟失败。"""
        client = _make_mock_client(wait_return={"task_id": "smoke-fail", "status": "failed"})

        result = run_smoke_test(client, TARGET_MACHINE)

        assert result["passed"] is False

    def test_smoke_fail_empty_probability(self) -> None:
        """probability 为空字典时冒烟失败。"""
        client = _make_mock_client(
            wait_return={
                "task_id": "smoke-empty",
                "status": "completed",
                "result": {},
            }
        )

        result = run_smoke_test(client, TARGET_MACHINE)

        assert result["passed"] is False

    def test_smoke_fail_query_error(self) -> None:
        """query_error 时冒烟失败。"""
        client = _make_mock_client(
            wait_return={
                "task_id": "smoke-qerr",
                "status": "query_error",
                "error": "SDK error",
            }
        )

        result = run_smoke_test(client, TARGET_MACHINE)

        assert result["passed"] is False


# ── run_single_seed 测试 ──


class TestRunSingleSeed:
    """测试 run_single_seed 函数。"""

    def test_wrong_machine_raises_error(self) -> None:
        """传入错误机器名应抛 ValueError。"""
        client = _make_mock_client()

        with pytest.raises(ValueError, match="机器一致性违规"):
            run_single_seed("fcfs", 42, client, "tianyan176", shots=SHOTS)

    def test_wrong_shots_raises_error(self) -> None:
        """传入错误 shots 应抛 ValueError。"""
        client = _make_mock_client()

        with pytest.raises(ValueError, match="shots 一致性违规"):
            run_single_seed("fcfs", 42, client, TARGET_MACHINE, shots=1024)

    def test_tianyan287_no_hyphen_rejected(self) -> None:
        """tianyan287（无连字符）机器名应被拒绝。"""
        client = _make_mock_client()

        with pytest.raises(ValueError, match="机器一致性违规"):
            run_single_seed("fcfs", 42, client, "tianyan287", shots=SHOTS)

    def test_result_marks_mock_false_degraded_false(self) -> None:
        """结果必须标记 mock=false, degraded=false。"""
        client = _make_mock_client(
            wait_return={
                "task_id": "seed-task-001",
                "status": "completed",
                "result": {"0": 0.5, "1": 0.5},
            }
        )

        result = run_single_seed("fcfs", 42, client, TARGET_MACHINE, shots=SHOTS)

        assert result["mock"] is False
        assert result["degraded"] is False
        assert result["machine"] == TARGET_MACHINE
        assert result["shots"] == SHOTS
        assert result["circuit"] == QCIS_CIRCUIT

    def test_counting_classification(self) -> None:
        """提交计数按终态正确分类。"""
        # 提交成功且完成
        client = _make_mock_client(
            wait_return={
                "task_id": "count-task",
                "status": "completed",
                "result": {"0": 0.5, "1": 0.5},
            }
        )

        result = run_single_seed("fcfs", 42, client, TARGET_MACHINE, shots=SHOTS)
        metrics = result["metrics"]

        assert metrics["real_tasks_submitted"] == 1
        assert metrics["real_tasks_completed"] == 1
        assert metrics["real_tasks_failed"] == 0
        assert metrics["real_tasks_timeout"] == 0
        assert metrics["real_tasks_query_error"] == 0

    def test_counting_failed_task(self) -> None:
        """失败任务计入 failed，不计 completed。"""
        client = _make_mock_client(
            submit_return="fail-task-001",
            wait_return={"task_id": "fail-task-001", "status": "failed"},
        )

        result = run_single_seed("fcfs", 42, client, TARGET_MACHINE, shots=SHOTS)
        metrics = result["metrics"]

        assert metrics["real_tasks_submitted"] == 1
        assert metrics["real_tasks_completed"] == 0
        assert metrics["real_tasks_failed"] == 1

    def test_counting_timeout_task(self) -> None:
        """超时任务计入 timeout。"""
        client = _make_mock_client(
            submit_return="timeout-task-001",
            wait_return={"task_id": "timeout-task-001", "status": "timeout"},
        )

        result = run_single_seed("fcfs", 42, client, TARGET_MACHINE, shots=SHOTS)
        metrics = result["metrics"]

        assert metrics["real_tasks_submitted"] == 1
        assert metrics["real_tasks_completed"] == 0
        assert metrics["real_tasks_timeout"] == 1

    def test_counting_query_error_task(self) -> None:
        """query_error 任务计入 query_error，不计 completed。"""
        client = _make_mock_client(
            submit_return="qerr-task-001",
            wait_return={
                "task_id": "qerr-task-001",
                "status": "query_error",
                "error": "SDK error",
            },
        )

        result = run_single_seed("fcfs", 42, client, TARGET_MACHINE, shots=SHOTS)
        metrics = result["metrics"]

        assert metrics["real_tasks_submitted"] == 1
        assert metrics["real_tasks_completed"] == 0
        assert metrics["real_tasks_query_error"] == 1

    def test_task_id_preserved_in_records_on_query_error(self) -> None:
        """轮询异常时 task_id 仍保留在 real_records 中。"""
        client = _make_mock_client(
            submit_return="preserve-002",
            wait_side_effect=RuntimeError("network dropped"),
        )

        result = run_single_seed("fcfs", 42, client, TARGET_MACHINE, shots=SHOTS)

        # real_records 应包含至少 1 条记录
        assert len(result["real_records"]) >= 1
        record = result["real_records"][0]
        assert record["task_id"] == "preserve-002"
        assert record["status"] == "query_error"


# ── 禁止自动回退测试 ──


class TestNoAutoFallback:
    """测试禁止自动回退到 176/Mock/其他机器。"""

    def test_wrong_machine_rejected_in_smoke(self) -> None:
        """冒烟测试中错误机器被拒绝。"""
        client = _make_mock_client()
        result = run_smoke_test(client, "tianyan176")
        assert result["passed"] is False

    def test_mock_machine_rejected_in_smoke(self) -> None:
        """冒烟测试中 Mock 机器被拒绝。"""
        client = _make_mock_client()
        result = run_smoke_test(client, "mock")
        assert result["passed"] is False

    def test_auto_retry_not_used_in_single_seed(self) -> None:
        """run_single_seed 中不会自动切换机器。

        通过验证 machine_name 始终为 TARGET_MACHINE 来确认。
        """
        client = _make_mock_client(
            wait_return={
                "task_id": "no-fallback-001",
                "status": "completed",
                "result": {"0": 0.5, "1": 0.5},
            }
        )

        result = run_single_seed("fcfs", 42, client, TARGET_MACHINE, shots=SHOTS)

        # 结果中 machine 字段必须始终为 TARGET_MACHINE
        assert result["machine"] == TARGET_MACHINE
        for record in result.get("real_records", []):
            assert record["machine"] == TARGET_MACHINE


# ── 硬上限测试 ──


class TestHardLimit:
    """测试提交硬上限不会超限。"""

    def test_hard_limit_total_is_31(self) -> None:
        """总硬上限为 31（30 正式 + 1 冒烟）。"""
        assert HARD_LIMIT_TOTAL == 31

    def test_hard_limit_formal_is_30(self) -> None:
        """正式实验硬上限为 30。"""
        assert HARD_LIMIT_FORMAL == 30

    def test_hard_limit_smoke_is_1(self) -> None:
        """冒烟硬上限为 1。"""
        assert HARD_LIMIT_SMOKE == 1

    def test_max_real_tasks_per_run_does_not_exceed_limit(self) -> None:
        """每次 run 最多 1 个真机任务，不会超限。"""
        assert MAX_REAL_TASKS_PER_RUN == 1
        # 10 seeds × 3 策略 × 1 = 30 = HARD_LIMIT_FORMAL
        total = len(SEEDS) * len(STRATEGIES) * MAX_REAL_TASKS_PER_RUN
        assert total == HARD_LIMIT_FORMAL
        assert total + HARD_LIMIT_SMOKE == HARD_LIMIT_TOTAL
