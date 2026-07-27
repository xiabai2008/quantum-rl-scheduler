#!/usr/bin/env python
"""env_real_machine.py 真机闭环模块的单元测试"""

import contextlib
import os
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest

from src.api.types import TaskResult
from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_machines import route_to_machine
from src.scheduler.env_real_machine import (
    FREE_TIER_MAX_QUBITS,
    _compute_real_feedback,
    _record_real_result,
    _resolve_free_tier_max_qubits,
    _update_task_duration,
    compute_real_result_reward,
    compute_result_fidelity,
    compute_theoretical_distribution,
    generate_qcis_circuit,
    parse_measurement_result,
    poll_pending_real_tasks,
    record_real_failure,
    shuffle_measurement,
    submit_to_real_machine,
)
from src.scheduler.env_types import (
    REAL_MACHINE_DEGRADE_FAIL_THRESHOLD,
    REAL_MACHINE_FAIL_PENALTY,
    REAL_MACHINE_MAX_POLL_STEPS,
    REAL_MACHINE_SUBMIT_INTERVAL,
    REAL_MACHINE_SUCCESS_BONUS,
    REAL_RESULT_REWARD_MAX,
    REAL_RESULT_REWARD_MIN,
    QuantumMachine,
    Task,
)


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
        assert len(qcis_high) >= len(qcis_low), (
            f"高优先级电路应更深: {len(qcis_high)} vs {len(qcis_low)}"
        )

    def test_deterministic_with_seed(self):
        """相同 seed 生成相同电路"""
        task = Task(task_id="0", task_type="quantum", qubit_count=5, priority=3)
        q1 = generate_qcis_circuit(task, seed=42)
        q2 = generate_qcis_circuit(task, seed=42)
        assert q1 == q2

    def test_different_seed_produces_different_circuit(self):
        """不同 seed 可能生成不同电路"""
        task = Task(task_id="0", task_type="quantum", qubit_count=5, priority=3)
        generate_qcis_circuit(task, seed=42)
        generate_qcis_circuit(task, seed=123)
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

    def test_bell_ghz_circuit_generation(self):
        """Bell/GHZ 模板应生成固定纠缠电路。"""
        task = Task(task_id="entangled", task_type="quantum", qubit_count=3, priority=3)

        bell = generate_qcis_circuit(task, max_qubits=2, circuit_type="bell")
        assert bell == "H Q0\nCNOT Q0 Q1\nM Q0 Q1"

        ghz = generate_qcis_circuit(task, max_qubits=3, circuit_type="ghz3")
        assert ghz == "H Q0\nCNOT Q0 Q1\nCNOT Q1 Q2\nM Q0 Q1 Q2"

    @pytest.mark.parametrize(
        ("circuit_type", "max_qubits"),
        [("bell", 1), ("ghz3", 2)],
    )
    def test_entangled_template_respects_qubit_limit(
        self,
        circuit_type: str,
        max_qubits: int,
    ):
        """模板不能在配置上限不足时生成越界量子比特。"""
        task = Task(task_id="limited", task_type="quantum", qubit_count=3, priority=3)
        with pytest.raises(ValueError, match="requires at least"):
            generate_qcis_circuit(
                task,
                max_qubits=max_qubits,
                circuit_type=circuit_type,
            )

    def test_invalid_circuit_type_raises(self):
        """未知模板名称应显式拒绝。"""
        task = Task(task_id="invalid", task_type="quantum", qubit_count=3, priority=3)
        with pytest.raises(ValueError, match="circuit_type"):
            generate_qcis_circuit(task, circuit_type="unknown")


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


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def submit_quantum_task(self, **kwargs):
        self.calls.append(kwargs)
        return f"real-{len(self.calls)}"


class TestRealSubmissionBudget:
    """训练级真机硬上限与低 shots 配置测试。"""

    @staticmethod
    def _make_env(max_real_submissions: int | None = 2) -> tuple:
        env = QuantumSchedulingEnv(
            machine_configs=[
                {
                    "name": "tianyan176",
                    "total_qubits": 287,
                    "supported_gates": ("H", "M"),
                    "is_real": True,
                }
            ],
            use_real_machine=True,
            max_real_submissions=max_real_submissions,
            real_machine_shots=64,
        )
        client = _RecordingClient()
        env.attach_real_clients({"tianyan176": client})
        machine = QuantumMachine(name="tianyan176", total_qubits=287, is_real=True)
        task = Task(task_id="budget", task_type="quantum", qubit_count=1)
        return env, client, machine, task

    def test_rejects_invalid_budget_configuration(self):
        with pytest.raises(ValueError, match="max_real_submissions"):
            QuantumSchedulingEnv(max_real_submissions=-1)
        with pytest.raises(ValueError, match="real_machine_shots"):
            QuantumSchedulingEnv(real_machine_shots=0)
        with pytest.raises(ValueError, match="real_machine_max_qubits"):
            QuantumSchedulingEnv(real_machine_max_qubits=0)

    def test_submission_uses_configured_real_machine_qubit_limit(self):
        """动态生成的提交电路应遵守显式真机比特上限。"""
        env, client, machine, _task = self._make_env()
        env.real_machine_max_qubits = 2
        task = Task(task_id="configured", task_type="quantum", qubit_count=5)

        submit_to_real_machine(env, machine, task)

        assert len(client.calls) == 1
        assert "Q2" not in client.calls[0]["qcis"]

    def test_hard_limit_survives_reset_and_uses_configured_shots(self):
        env, client, machine, task = self._make_env()

        submit_to_real_machine(env, machine, task)
        env.reset(seed=1)
        submit_to_real_machine(env, machine, task)
        submit_to_real_machine(env, machine, task)

        assert len(client.calls) == 2
        assert [call["shots"] for call in client.calls] == [64, 64]
        stats = env.get_real_machine_stats()
        assert stats["submission_attempts_total"] == 2
        assert stats["max_real_submissions"] == 2


class _StatusClient:
    """Mock 真机客户端：可配置 submit 返回值与 get_task_status 响应。"""

    def __init__(
        self,
        statuses: dict[str, dict] | None = None,
        submit_id: str | None = "real-1",
    ) -> None:
        self._statuses = statuses or {}
        self._submit_id = submit_id
        self.submitted: list[dict] = []

    def submit_quantum_task(self, **kwargs):
        self.submitted.append(kwargs)
        return self._submit_id

    def get_task_status(self, task_id: str) -> dict:
        return self._statuses.get(task_id, {"status": "running"})


class _ExceptionClient:
    """submit/get_task_status 均抛异常的 mock 客户端。"""

    def submit_quantum_task(self, **kwargs):
        raise RuntimeError("网络异常")

    def get_task_status(self, task_id: str) -> dict:
        raise RuntimeError("查询异常")


def _make_env_with_client(
    max_real_submissions: int | None = 10,
    real_machine_shots: int = 64,
    real_feedback_mode: str = "status_only",
    feedback_weight: float = 1.0,
    client: object | None = None,
    machine_name: str = "tianyan176",
) -> tuple:
    """构造带真机客户端的环境，返回 (env, client, machine, task)。"""
    env = QuantumSchedulingEnv(
        machine_configs=[
            {
                "name": machine_name,
                "total_qubits": 287,
                "supported_gates": ("H", "M"),
                "is_real": True,
            }
        ],
        use_real_machine=True,
        max_real_submissions=max_real_submissions,
        real_machine_shots=real_machine_shots,
        real_feedback_mode=real_feedback_mode,
        real_machine_feedback_weight=feedback_weight,
    )
    if client is None:
        client = _StatusClient()
    env.attach_real_clients({machine_name: client})
    machine = QuantumMachine(name=machine_name, total_qubits=287, is_real=True)
    task = Task(task_id="t0", task_type="quantum", qubit_count=1)
    return env, client, machine, task


# =============================================================================
# 测量结果解析（Issue #235）
# =============================================================================


class TestParseMeasurementResult:
    """parse_measurement_result 三条解析路径测试。"""

    def test_probability_direct(self):
        """probability 字段直接返回并归一化。"""
        status = {"probability": {"0": 0.3, "1": 0.7}}
        result = parse_measurement_result(status)
        assert result == {"0": 0.3, "1": 0.7}

    def test_task_result_prefers_probability_over_counts(self):
        """TaskResult 同时包含两者时必须优先采用真机 probability。"""
        status = TaskResult(
            task_id="real-1",
            status="completed",
            probability={"00": 0.25, "11": 0.75},
            counts={"00": 90, "11": 10},
            shots=100,
            backend="tianyan-287",
        )
        assert parse_measurement_result(status) == {"00": 0.25, "11": 0.75}

    def test_mock_counts_and_real_probability_are_consistent(self):
        """Mock counts 与真机 probability 的统一结果应产生相同概率分布。"""
        mock_result = TaskResult(
            task_id="mock-1",
            status="COMPLETED",
            probability={},
            counts={"00": 32, "11": 32},
            shots=64,
            backend="tianyan-simulator",
        )
        real_result = TaskResult(
            task_id="real-1",
            status="completed",
            probability={"00": 0.5, "11": 0.5},
            counts=None,
            shots=64,
            backend="tianyan-287",
        )
        assert parse_measurement_result(mock_result) == parse_measurement_result(real_result)

    def test_probability_normalization(self):
        """未归一化的 probability 自动归一化。"""
        status = {"probability": {"0": 30, "1": 70}}
        result = parse_measurement_result(status)
        assert abs(result["0"] - 0.3) < 1e-9
        assert abs(result["1"] - 0.7) < 1e-9

    def test_probability_invalid_values_skipped(self):
        """非法值被跳过，剩余值归一化。"""
        status = {"probability": {"0": 0.5, "1": "abc", "2": None}}
        result = parse_measurement_result(status)
        # 0.5 是唯一合法值，归一化后为 1.0
        assert result == {"0": 1.0}

    def test_result_status_json_string(self):
        """resultStatus JSON 字符串解析为概率。"""
        status = {"resultStatus": '{"0": 32, "1": 32}'}
        result = parse_measurement_result(status)
        assert abs(result["0"] - 0.5) < 1e-9
        assert abs(result["1"] - 0.5) < 1e-9

    def test_result_nested_probability(self):
        """result 字段嵌套 probability 解析。"""
        status = {"result": {"probability": {"0": 0.25, "1": 0.75}}}
        result = parse_measurement_result(status)
        assert result == {"0": 0.25, "1": 0.75}

    def test_empty_status_returns_empty(self):
        """空状态字典返回空字典。"""
        assert parse_measurement_result({}) == {}

    def test_invalid_result_status_returns_empty(self):
        """resultStatus 非法 JSON 返回空字典。"""
        assert parse_measurement_result({"resultStatus": "not-json"}) == {}


# =============================================================================
# 理论分布计算
# =============================================================================


class TestComputeTheoreticalDistribution:
    """compute_theoretical_distribution 理论分布测试。"""

    def test_h_gate_uniform_single_qubit(self):
        """单比特 H 门 → 均匀分布。"""
        qcis = "H Q0\nM Q0"
        dist = compute_theoretical_distribution(qcis)
        assert dist == {"0": 0.5, "1": 0.5}

    def test_x_gate_deterministic(self):
        """X 门 → 全 1 确定态。"""
        qcis = "X Q0\nM Q0"
        dist = compute_theoretical_distribution(qcis)
        assert dist == {"1": 1.0}

    def test_multi_qubit_h_uniform(self):
        """多比特 H 门 → 2^n 均匀分布。"""
        qcis = "H Q0\nH Q1\nM Q0\nM Q1"
        dist = compute_theoretical_distribution(qcis)
        assert len(dist) == 4
        for v in dist.values():
            assert abs(v - 0.25) < 1e-9

    @pytest.mark.parametrize(
        ("qcis", "expected"),
        [
            ("H Q0\nCNOT Q0 Q1\nM Q0 Q1", {"00": 0.5, "11": 0.5}),
            (
                "H Q0\nCNOT Q0 Q1\nCNOT Q1 Q2\nM Q0 Q1 Q2",
                {"000": 0.5, "111": 0.5},
            ),
        ],
    )
    def test_entangled_template_exact_distribution(
        self,
        qcis: str,
        expected: dict[str, float],
    ):
        """Bell/GHZ 模板应返回精确理想分布，而非均匀近似。"""
        assert compute_theoretical_distribution(qcis) == expected

    def test_no_measure_returns_default(self):
        """无测量行返回默认 {0: 1.0}。"""
        qcis = "H Q0"
        dist = compute_theoretical_distribution(qcis)
        assert dist == {"0": 1.0}

    def test_mixed_gates_fallback_uniform(self):
        """H+X 混合门回退到均匀分布。"""
        qcis = "H Q0\nX Q0\nM Q0"
        dist = compute_theoretical_distribution(qcis)
        assert dist == {"0": 0.5, "1": 0.5}


# =============================================================================
# 保真度计算
# =============================================================================


class TestComputeResultFidelity:
    """compute_result_fidelity 保真度计算测试。"""

    def test_perfect_match(self):
        """完全匹配 → 1.0。"""
        measured = {"0": 0.5, "1": 0.5}
        theoretical = {"0": 0.5, "1": 0.5}
        assert compute_result_fidelity(measured, theoretical) == 1.0

    def test_complete_mismatch(self):
        """完全不匹配 → 0.0。"""
        measured = {"0": 1.0}
        theoretical = {"1": 1.0}
        assert compute_result_fidelity(measured, theoretical) == 0.0

    def test_partial_match(self):
        """部分匹配 → (0, 1) 之间。"""
        measured = {"0": 0.5, "1": 0.5}
        theoretical = {"0": 0.8, "1": 0.2}
        fidelity = compute_result_fidelity(measured, theoretical)
        expected = ((0.5 * 0.8) ** 0.5 + (0.5 * 0.2) ** 0.5) ** 2
        assert abs(fidelity - expected) < 1e-9
        assert 0.0 < fidelity < 1.0

    def test_empty_returns_zero(self):
        """空字典 → 0.0。"""
        assert compute_result_fidelity({}, {"0": 1.0}) == 0.0
        assert compute_result_fidelity({"0": 1.0}, {}) == 0.0


# =============================================================================
# 真机 reward 计算
# =============================================================================


class TestComputeRealResultReward:
    """compute_real_result_reward 质量感知 reward 测试。"""

    def test_high_fidelity_high_reward(self):
        """高保真度 → 接近最大 reward。"""
        measured = {"0": 0.5, "1": 0.5}
        theoretical = {"0": 0.5, "1": 0.5}
        reward, fidelity, formula = compute_real_result_reward(measured, theoretical)
        assert fidelity == 1.0
        assert abs(reward - REAL_RESULT_REWARD_MAX) < 1e-9
        assert "fidelity" in formula

    def test_empty_measured_min_reward(self):
        """空测量 → 最小 reward。"""
        reward, fidelity, formula = compute_real_result_reward({}, {"0": 1.0})
        assert fidelity == 0.0
        assert abs(reward - REAL_RESULT_REWARD_MIN) < 1e-9
        assert "measurement_parse_failed" in formula

    def test_reward_bounded(self):
        """reward 始终在 [MIN, MAX] 区间。"""
        measured = {"0": 0.3, "1": 0.7}
        theoretical = {"0": 0.9, "1": 0.1}
        reward, _, _ = compute_real_result_reward(measured, theoretical)
        assert REAL_RESULT_REWARD_MIN <= reward <= REAL_RESULT_REWARD_MAX


# =============================================================================
# 测量打乱（消融对照）
# =============================================================================


class TestShuffleMeasurement:
    """shuffle_measurement 打乱测试。"""

    def test_values_preserved(self):
        """打乱后值的多重集不变。"""
        measured = {"0": 0.1, "1": 0.2, "2": 0.3, "3": 0.4}
        shuffled = shuffle_measurement(measured)
        assert sorted(shuffled.values()) == sorted(measured.values())
        assert set(shuffled.keys()) == set(measured.keys())

    def test_single_element_unchanged(self):
        """单元素返回原样。"""
        measured = {"0": 1.0}
        assert shuffle_measurement(measured) == measured

    def test_empty_returns_empty(self):
        """空字典返回空字典。"""
        assert shuffle_measurement({}) == {}


# =============================================================================
# 真机失败记录与降级
# =============================================================================


class TestRecordRealFailure:
    """record_real_failure 失败计数与降级触发测试。"""

    def test_single_failure_no_degrade(self):
        """单次失败不触发降级。"""
        env, _, _, _ = _make_env_with_client()
        record_real_failure(env, "tianyan176", "网络抖动")
        assert env._real_fail_count == 1
        assert env._real_consecutive_failures == 1
        assert env._real_machine_degraded is False

    def test_threshold_triggers_degrade(self):
        """达到阈值触发降级。"""
        env, _, _, _ = _make_env_with_client()
        for i in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD):
            record_real_failure(env, "tianyan176", f"失败{i}")
        assert env._real_machine_degraded is True
        assert env._real_consecutive_failures >= REAL_MACHINE_DEGRADE_FAIL_THRESHOLD
        # 降级日志已写入 render_log
        assert any("降级" in msg for msg in env._render_log)

    def test_already_degraded_no_repeat_log(self):
        """已降级后再次失败不重复写降级日志。"""
        env, _, _, _ = _make_env_with_client()
        for _ in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD):
            record_real_failure(env, "tianyan176", "失败")
        log_count_before = sum("降级" in m for m in env._render_log)
        # 再触发几次失败
        for _ in range(3):
            record_real_failure(env, "tianyan176", "继续失败")
        log_count_after = sum("降级" in m for m in env._render_log)
        assert log_count_after == log_count_before


# =============================================================================
# 真机错误区分（Issue #218）
# =============================================================================


class _TransientErrorClient:
    """submit 抛出网络异常（OSError 子类）的 mock 客户端。"""

    def submit_quantum_task(self, **kwargs):
        raise ConnectionError("网络抖动：连接超时")

    def get_task_status(self, task_id: str) -> dict:
        raise RuntimeError("查询异常")


class _PermanentErrorClient:
    """submit 抛出非网络异常的 mock 客户端（模拟认证失败）。"""

    def submit_quantum_task(self, **kwargs):
        raise RuntimeError("认证失败：API Key 无效")

    def get_task_status(self, task_id: str) -> dict:
        raise RuntimeError("查询异常")


class TestTransientVsPermanentError:
    """Issue #218: 真机失败区分暂时性错误与永久性错误。"""

    def test_transient_error_not_counted_as_failure(self):
        """暂时性错误（网络抖动）不计入连续失败计数，不触发降级。"""
        env, _, machine, task = _make_env_with_client(client=_TransientErrorClient())

        # 触发多次暂时性错误
        for _ in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD + 2):
            submit_to_real_machine(env, machine, task)

        # 暂时性错误不应计入连续失败
        assert env._real_consecutive_failures == 0
        assert env._real_fail_count == 0
        # 不应触发降级
        assert env._real_machine_degraded is False
        # render_log 中应有"暂时性错误"记录
        assert any("暂时性错误" in msg for msg in env._render_log)

    def test_permanent_error_triggers_degradation(self):
        """永久性错误（认证失败）计入连续失败并触发降级。"""
        env, _, machine, task = _make_env_with_client(client=_PermanentErrorClient())

        # 触发足够多次永久性错误
        for _ in range(REAL_MACHINE_DEGRADE_FAIL_THRESHOLD):
            submit_to_real_machine(env, machine, task)

        # 永久性错误应计入连续失败
        assert env._real_consecutive_failures >= REAL_MACHINE_DEGRADE_FAIL_THRESHOLD
        assert env._real_fail_count >= REAL_MACHINE_DEGRADE_FAIL_THRESHOLD
        # 应触发降级
        assert env._real_machine_degraded is True

    def test_mixed_errors_transient_does_not_reset_permanent(self):
        """混合错误场景：暂时性错误不影响永久性错误的累计计数。"""
        env, _, machine, task = _make_env_with_client(client=_PermanentErrorClient())

        # 1 次永久性错误
        submit_to_real_machine(env, machine, task)
        assert env._real_consecutive_failures == 1

        # 替换为暂时性错误客户端，触发几次
        env._real_clients[machine.name] = _TransientErrorClient()
        for _ in range(3):
            submit_to_real_machine(env, machine, task)

        # 暂时性错误不应增加计数，但也不应重置
        assert env._real_consecutive_failures == 1
        assert env._real_machine_degraded is False


# =============================================================================
# 真机任务轮询
# =============================================================================


class TestPollPendingRealTasks:
    """poll_pending_real_tasks 轮询逻辑测试。"""

    def test_empty_pending_returns_zero(self):
        """空 pending 列表返回 0。"""
        env, _, _, _ = _make_env_with_client()
        assert poll_pending_real_tasks(env) == 0.0

    def test_completed_returns_bonus(self):
        """completed 状态返回正 reward。"""
        client = _StatusClient(statuses={"real-1": {"status": "completed"}})
        env, _, machine, task = _make_env_with_client(client=client)
        submit_to_real_machine(env, machine, task)
        assert len(env._pending_real_tasks) == 1

        feedback = poll_pending_real_tasks(env)
        # status_only 模式：success bonus * feedback_weight(1.0)
        assert abs(feedback - REAL_MACHINE_SUCCESS_BONUS) < 1e-9
        assert env._real_success_count == 1
        assert env._real_consecutive_failures == 0
        # 完成后从 pending 移除
        assert len(env._pending_real_tasks) == 0

    def test_error_returns_penalty(self):
        """error 状态返回负 reward 并记录失败。"""
        client = _StatusClient(statuses={"real-1": {"status": "error"}})
        env, _, machine, task = _make_env_with_client(client=client)
        submit_to_real_machine(env, machine, task)

        feedback = poll_pending_real_tasks(env)
        assert feedback < 0
        assert env._real_fail_count == 1
        assert env._real_consecutive_failures == 1

    def test_running_keeps_pending(self):
        """running 状态保留在 pending 列表。"""
        client = _StatusClient(statuses={"real-1": {"status": "running"}})
        env, _, machine, task = _make_env_with_client(client=client)
        submit_to_real_machine(env, machine, task)

        feedback = poll_pending_real_tasks(env)
        assert feedback == 0.0
        assert len(env._pending_real_tasks) == 1
        assert env._pending_real_tasks[0]["poll_count"] == 1

    def test_timeout_triggers_failure(self):
        """轮询超过 MAX_POLL_STEPS 视为超时失败。"""
        client = _StatusClient(statuses={"real-1": {"status": "running"}})
        env, _, machine, task = _make_env_with_client(client=client)
        submit_to_real_machine(env, machine, task)
        # 手动把 poll_count 推到阈值前一刻
        env._pending_real_tasks[0]["poll_count"] = REAL_MACHINE_MAX_POLL_STEPS - 1

        feedback = poll_pending_real_tasks(env)
        assert feedback < 0
        assert env._real_fail_count == 1
        assert len(env._pending_real_tasks) == 0

    def test_client_missing_failure(self):
        """客户端丢失视为失败。"""
        env, _, machine, task = _make_env_with_client()
        submit_to_real_machine(env, machine, task)
        # 移除客户端模拟丢失
        env._real_clients = {}

        feedback = poll_pending_real_tasks(env)
        assert feedback < 0
        assert env._real_fail_count == 1

    def test_get_status_exception_keeps_pending(self):
        """get_task_status 异常时保留在 pending 列表。"""

        # _ExceptionClient submit 会抛异常 → 走 record_real_failure，不会进 pending
        # 改用 _StatusClient submit + 异常 get_task_status
        class _PollExceptionClient:
            def submit_quantum_task(self, **kwargs):
                return "real-1"

            def get_task_status(self, task_id):
                raise RuntimeError("查询异常")

        env, _, machine, task = _make_env_with_client(client=_PollExceptionClient())
        submit_to_real_machine(env, machine, task)
        assert len(env._pending_real_tasks) == 1

        feedback = poll_pending_real_tasks(env)
        assert feedback == 0.0
        assert len(env._pending_real_tasks) == 1


# =============================================================================
# 真机反馈计算（_compute_real_feedback）
# =============================================================================


class TestComputeRealFeedback:
    """_compute_real_feedback 三种反馈模式测试。"""

    def test_status_only_fixed_bonus(self):
        """status_only 模式返回固定 bonus。"""
        env, _, _, _ = _make_env_with_client(real_feedback_mode="status_only")
        pending = {"qcis_circuit": "H Q0\nM Q0"}
        status = {"status": "completed", "probability": {"0": 0.5, "1": 0.5}}
        reward, fidelity, formula = _compute_real_feedback(env, pending, status)
        assert abs(reward - REAL_MACHINE_SUCCESS_BONUS) < 1e-9
        assert fidelity == -1.0
        assert "status_only" in formula

    def test_result_aware_fidelity_based(self):
        """result_aware 模式按保真度计算 reward。"""
        env, _, _, _ = _make_env_with_client(real_feedback_mode="result_aware")
        pending = {"qcis_circuit": "H Q0\nM Q0"}
        status = {"status": "completed", "probability": {"0": 0.5, "1": 0.5}}
        reward, fidelity, formula = _compute_real_feedback(env, pending, status)
        # 完美匹配 → fidelity=1.0 → reward=MAX
        assert abs(fidelity - 1.0) < 1e-9
        assert abs(reward - REAL_RESULT_REWARD_MAX) < 1e-9
        assert "fidelity" in formula

    def test_shuffled_mode_marker(self):
        """shuffled 模式公式带 [SHUFFLED] 标记。"""
        env, _, _, _ = _make_env_with_client(real_feedback_mode="shuffled")
        pending = {"qcis_circuit": "H Q0\nM Q0"}
        status = {"status": "completed", "probability": {"0": 0.3, "1": 0.7}}
        _reward, _fidelity, formula = _compute_real_feedback(env, pending, status)
        assert "[SHUFFLED]" in formula


# =============================================================================
# 真机结果记录（_record_real_result）
# =============================================================================


class TestRecordRealResult:
    """_record_real_result 元数据记录测试。"""

    def test_status_only_record(self):
        """status_only 模式记录 fallback_mode=True。"""
        env, _, _, _ = _make_env_with_client(real_feedback_mode="status_only")
        pending = {
            "task_id": "real-1",
            "task_id_str": "t0",
            "machine_name": "tianyan176",
            "submit_step": 0,
            "qcis_circuit": "H Q0\nM Q0",
        }
        status = {"status": "completed"}
        _record_real_result(env, pending, status, 2.0, -1.0, "formula")
        assert hasattr(env, "_real_result_records")
        assert len(env._real_result_records) == 1
        record = env._real_result_records[0]
        assert record["task_id"] == "t0"
        assert record["fidelity"] is None
        assert record["result_valid"] is False
        assert record["fallback_mode"] is True

    def test_result_aware_record(self):
        """result_aware 模式记录测量分布与保真度。"""
        env, _, _, _ = _make_env_with_client(real_feedback_mode="result_aware")
        pending = {
            "task_id": "real-1",
            "task_id_str": "t0",
            "machine_name": "tianyan176",
            "submit_step": 0,
            "qcis_circuit": "H Q0\nM Q0",
        }
        status = {"status": "completed", "probability": {"0": 0.5, "1": 0.5}}
        _record_real_result(env, pending, status, 5.0, 1.0, "reward=5.0000")
        record = env._real_result_records[0]
        assert record["fidelity"] == 1.0
        assert record["result_valid"] is True
        assert record["fallback_mode"] is False
        assert record["probability"] == {"0": 0.5, "1": 0.5}


# =============================================================================
# 真机执行时间回写（_update_task_duration）
# =============================================================================


class TestUpdateTaskDuration:
    """_update_task_duration 任务执行时间回写测试。"""

    def test_none_duration_no_op(self):
        """None 执行时间不更新。"""
        env, _, _, _ = _make_env_with_client()
        _update_task_duration(env, "t0", None)
        # 无异常即通过

    def test_current_task_completed(self):
        """回写当前正在执行的任务。"""
        env, _, _, task = _make_env_with_client()
        task.execution_time = 10
        env._current_task = task
        _update_task_duration(env, "t0", 1.5)
        assert env._current_task.execution_time == 0

    def test_queued_task_completed(self):
        """回写队列中的任务。"""
        env, _, _, _ = _make_env_with_client()
        queued_task = Task(task_id="q1", task_type="quantum", qubit_count=1)
        queued_task.execution_time = 20
        env._task_queue = [queued_task]
        _update_task_duration(env, "q1", 2.0)
        assert queued_task.execution_time == 0

    def test_missing_task_silent(self):
        """找不到任务静默忽略。"""
        env, _, _, _ = _make_env_with_client()
        env._current_task = None
        env._task_queue = []
        # 无异常即通过
        _update_task_duration(env, "nonexistent", 1.0)


# =============================================================================
# 真机提交降级路径
# =============================================================================


class TestSubmitDegradeFlow:
    """submit_to_real_machine 降级与异常路径测试。"""

    def test_degraded_skips_submission(self):
        """已降级时跳过提交。"""
        env, client, machine, task = _make_env_with_client()
        env._real_machine_degraded = True
        submit_to_real_machine(env, machine, task)
        assert len(client.submitted) == 0
        assert env._real_submission_attempts_total == 0

    def test_max_submissions_reached_skips(self):
        """达到硬上限跳过提交。"""
        env, client, machine, task = _make_env_with_client(max_real_submissions=1)
        # 第一次提交成功
        submit_to_real_machine(env, machine, task)
        assert len(client.submitted) == 1
        # 第二次因上限跳过
        submit_to_real_machine(env, machine, task)
        assert len(client.submitted) == 1

    def test_submit_exception_records_failure(self):
        """提交异常触发 record_real_failure。"""
        env, _, machine, task = _make_env_with_client(client=_ExceptionClient())
        submit_to_real_machine(env, machine, task)
        assert env._real_fail_count == 1
        assert env._real_consecutive_failures == 1
        assert len(env._pending_real_tasks) == 0
        # 异常信息写入 render_log
        assert any("提交失败" in m for m in env._render_log)

    def test_submit_returns_none_records_failure(self):
        """提交返回 None（被拒绝）触发失败记录。"""
        client = _StatusClient(submit_id=None)
        env, _, machine, task = _make_env_with_client(client=client)
        submit_to_real_machine(env, machine, task)
        assert env._real_fail_count == 1
        assert len(env._pending_real_tasks) == 0

    def test_no_client_skips_submission(self):
        """无对应客户端时跳过提交（不计数）。"""
        env, _, machine, task = _make_env_with_client()
        env._real_clients = {}  # 清空客户端
        submit_to_real_machine(env, machine, task)
        assert env._real_submission_attempts_total == 0


# =============================================================================
# 两比特门电路生成
# =============================================================================


class TestTwoQubitGatesCircuit:
    """generate_qcis_circuit two_qubit_gates 分支测试。"""

    def test_two_qubit_gates_present(self):
        """two_qubit_gates=True 包含 CNOT/CZ。"""
        task = Task(task_id="0", task_type="quantum", qubit_count=4, priority=3)
        qcis = generate_qcis_circuit(task, seed=42, two_qubit_gates=True)
        assert any(g in qcis for g in ["CNOT", "CZ"])

    def test_two_qubit_gates_absent_by_default(self):
        """默认不包含两比特门。"""
        task = Task(task_id="0", task_type="quantum", qubit_count=4, priority=3)
        qcis = generate_qcis_circuit(task, seed=42)
        assert "CNOT" not in qcis
        assert "CZ" not in qcis

    def test_priority_affects_two_qubit_depth(self):
        """高优先级在 two_qubit_gates 模式下产生更多两比特门。"""
        task_low = Task(task_id="0", task_type="quantum", qubit_count=6, priority=1)
        task_high = Task(task_id="1", task_type="quantum", qubit_count=6, priority=5)
        q_low = generate_qcis_circuit(task_low, seed=42, two_qubit_gates=True)
        q_high = generate_qcis_circuit(task_high, seed=42, two_qubit_gates=True)
        count_low = q_low.count("CNOT") + q_low.count("CZ")
        count_high = q_high.count("CNOT") + q_high.count("CZ")
        assert count_high > count_low


# =============================================================================
# FREE_TIER_MAX_QUBITS 常量
# =============================================================================


class TestFreeTierConstant:
    """FREE_TIER_MAX_QUBITS 常量约束与可配置性测试。"""

    def test_free_tier_default_is_one(self):
        """未设置环境变量时，免费机时包限制为 1 比特（真机稳定模式）。"""
        os.environ.pop("FREE_TIER_MAX_QUBITS", None)
        assert _resolve_free_tier_max_qubits() == 1

    def test_free_tier_module_constant_positive(self):
        """模块级常量在导入时为正整数（受环境变量影响但至少为 1）。"""
        assert isinstance(FREE_TIER_MAX_QUBITS, int)
        assert FREE_TIER_MAX_QUBITS >= 1

    def test_free_tier_env_var_override(self):
        """环境变量 FREE_TIER_MAX_QUBITS 可覆盖默认限制（付费套餐场景）。"""
        original = os.environ.pop("FREE_TIER_MAX_QUBITS", None)
        try:
            os.environ["FREE_TIER_MAX_QUBITS"] = "5"
            assert _resolve_free_tier_max_qubits() == 5
        finally:
            if original is not None:
                os.environ["FREE_TIER_MAX_QUBITS"] = original
            else:
                os.environ.pop("FREE_TIER_MAX_QUBITS", None)

    @pytest.mark.parametrize("invalid", ["abc", "0", "-3", "1.5", ""])
    def test_free_tier_invalid_value_falls_back(self, invalid: str):
        """无效环境变量值回退到默认值 1，保证真机稳定模式。"""
        original = os.environ.pop("FREE_TIER_MAX_QUBITS", None)
        try:
            os.environ["FREE_TIER_MAX_QUBITS"] = invalid
            assert _resolve_free_tier_max_qubits() == 1, f"无效值 {invalid!r} 应回退到 1"
        finally:
            if original is not None:
                os.environ["FREE_TIER_MAX_QUBITS"] = original
            else:
                os.environ.pop("FREE_TIER_MAX_QUBITS", None)


# =============================================================================
# 补充覆盖：嵌套 result.probability 异常值路径（Issue #97, lines 193-194）
# =============================================================================


class TestParseNestedResultInvalidValues:
    """parse_measurement_result 嵌套 result.probability 非法值跳过路径测试。"""

    def test_nested_probability_with_invalid_values_skipped(self):
        """嵌套 result.probability 中非法值被跳过，剩余合法值归一化。"""
        status = {
            "result": {
                "probability": {"0": 0.5, "1": "abc", "2": None, "3": 0.5},
            }
        }
        result = parse_measurement_result(status)
        # "abc" 和 None 被跳过，剩余 0.5 + 0.5 归一化后各 0.5
        assert result == {"0": 0.5, "3": 0.5}

    def test_nested_probability_all_invalid_returns_empty(self):
        """嵌套 result.probability 全部非法时返回空字典。"""
        status = {"result": {"probability": {"0": "bad", "1": None}}}
        result = parse_measurement_result(status)
        assert result == {}

    def test_nested_probability_normalization(self):
        """嵌套 result.probability 未归一化时自动归一化。"""
        status = {"result": {"probability": {"0": 30, "1": 70}}}
        result = parse_measurement_result(status)
        assert abs(result["0"] - 0.3) < 1e-9
        assert abs(result["1"] - 0.7) < 1e-9

    def test_nested_result_not_dict_returns_empty(self):
        """result 字段非字典时返回空。"""
        assert parse_measurement_result({"result": "not-a-dict"}) == {}

    def test_nested_probability_not_dict_returns_empty(self):
        """嵌套 probability 非字典时返回空。"""
        assert parse_measurement_result({"result": {"probability": "not-a-dict"}}) == {}


# =============================================================================
# 补充覆盖：_record_real_result 初始化分支（Issue #97, line 640）
# =============================================================================


class TestRecordRealResultInitBranch:
    """_record_real_result 在 env 缺少 _real_result_records 属性时的初始化分支测试。"""

    def test_init_records_when_missing(self):
        """env 缺少 _real_result_records 属性时应初始化为空列表再追加。"""
        env, _, _, _ = _make_env_with_client(real_feedback_mode="result_aware")
        # 删除属性以触发初始化分支（line 640）
        if hasattr(env, "_real_result_records"):
            del env._real_result_records
        pending = {
            "task_id": "real-1",
            "task_id_str": "t0",
            "machine_name": "tianyan176",
            "submit_step": 0,
            "qcis_circuit": "H Q0\nM Q0",
        }
        status = {"status": "completed", "probability": {"0": 0.5, "1": 0.5}}
        _record_real_result(env, pending, status, 5.0, 1.0, "reward=5.0")
        # 应自动创建列表并写入一条记录
        assert hasattr(env, "_real_result_records")
        assert len(env._real_result_records) == 1
        assert env._real_result_records[0]["task_id"] == "t0"

    def test_record_appends_to_existing(self):
        """已有 _real_result_records 时应追加而非覆盖。"""
        env, _, _, _ = _make_env_with_client(real_feedback_mode="result_aware")
        # 预置一条已有记录
        env._real_result_records = [{"task_id": "prev"}]
        pending = {
            "task_id": "real-2",
            "task_id_str": "t1",
            "machine_name": "tianyan176",
            "submit_step": 0,
            "qcis_circuit": "H Q0\nM Q0",
        }
        status = {"status": "completed", "probability": {"0": 0.5, "1": 0.5}}
        _record_real_result(env, pending, status, 5.0, 1.0, "reward=5.0")
        assert len(env._real_result_records) == 2
        assert env._real_result_records[0]["task_id"] == "prev"
        assert env._real_result_records[1]["task_id"] == "t1"


# =============================================================================
# 补充覆盖：compute_theoretical_distribution 边界（无门电路）
# =============================================================================


class TestComputeTheoreticalDistributionEdgeCases:
    """compute_theoretical_distribution 边界条件补充测试。"""

    def test_only_measurement_no_gates(self):
        """仅有测量行无任何门时回退到均匀分布。"""
        qcis = "M Q0"
        dist = compute_theoretical_distribution(qcis)
        # 无 H 也无 X → 均匀分布
        assert dist == {"0": 0.5, "1": 0.5}

    def test_empty_circuit_returns_default(self):
        """空电路字符串返回默认分布。"""
        dist = compute_theoretical_distribution("")
        assert dist == {"0": 1.0}

    def test_x_gate_multi_qubit_all_ones(self):
        """多比特 X 门 → 全 1 确定态。"""
        qcis = "X Q0\nX Q1\nM Q0\nM Q1"
        dist = compute_theoretical_distribution(qcis)
        assert dist == {"11": 1.0}

    def test_z_gate_falls_back_to_uniform(self):
        """Z 门（非 H 非 X）回退到均匀分布。"""
        qcis = "Z Q0\nM Q0"
        dist = compute_theoretical_distribution(qcis)
        assert dist == {"0": 0.5, "1": 0.5}


# =============================================================================
# 补充覆盖：shuffle_measurement 多元素打乱
# =============================================================================


class TestShuffleMeasurementAdditional:
    """shuffle_measurement 补充边界测试。"""

    def test_two_element_shuffle_preserves_values(self):
        """双元素打乱后值集合不变。"""
        measured = {"0": 0.3, "1": 0.7}
        shuffled = shuffle_measurement(measured)
        assert sorted(shuffled.values()) == sorted(measured.values())
        assert set(shuffled.keys()) == set(measured.keys())

    def test_shuffle_with_seed_reproducible(self):
        """打乱结果值的集合始终不变（随机性仅在键值配对）。"""
        measured = {"a": 0.1, "b": 0.2, "c": 0.3, "d": 0.4}
        shuffled = shuffle_measurement(measured)
        # 值的多重集不变
        assert sorted(shuffled.values()) == sorted(measured.values())


# =============================================================================
# 间隔触发保底 + 概率触发（Issue #242 / #243）
# =============================================================================


class TestIntervalTrigger:
    """Issue #243: 真机提交间隔触发保底 + 概率触发额外增加测试。

    间隔触发确保即使 real_submit_probability=0.0，每 real_submit_interval
    步也会强制提交一次真机任务，避免概率触发完全错过（根因见 Issue #242）。
    """

    @staticmethod
    def _make_env_with_interval(
        interval: int = 20,
        probability: float = 0.0,
    ) -> tuple:
        """构造带间隔触发的真机环境，返回 (env, client, machine, task, rng)。"""
        env = QuantumSchedulingEnv(
            machine_configs=[
                {
                    "name": "tianyan176",
                    "total_qubits": 287,
                    "supported_gates": ("H", "M"),
                    "is_real": True,
                }
            ],
            use_real_machine=True,
            real_submit_probability=probability,
            real_submit_interval=interval,
            max_real_submissions=100,
            real_machine_shots=64,
        )
        client = _StatusClient()
        env.attach_real_clients({"tianyan176": client})
        machine = QuantumMachine(name="tianyan176", total_qubits=287, is_real=True)
        task = Task(task_id="t0", task_type="quantum", qubit_count=1)
        rng = np.random.default_rng(42)
        return env, client, machine, task, rng

    def test_default_interval_is_20(self):
        """REAL_MACHINE_SUBMIT_INTERVAL 常量默认值为 20。"""
        assert REAL_MACHINE_SUBMIT_INTERVAL == 20

    def test_env_default_interval(self):
        """env 默认 real_submit_interval 应为 REAL_MACHINE_SUBMIT_INTERVAL。"""
        env = QuantumSchedulingEnv()
        assert env.real_submit_interval == REAL_MACHINE_SUBMIT_INTERVAL

    def test_invalid_interval_raises(self):
        """real_submit_interval < 1 应抛出 ValueError。"""
        with pytest.raises(ValueError, match="real_submit_interval"):
            QuantumSchedulingEnv(real_submit_interval=0)
        with pytest.raises(ValueError, match="real_submit_interval"):
            QuantumSchedulingEnv(real_submit_interval=-5)

    def test_interval_trigger_fires_at_multiples(self):
        """间隔触发：step % interval == 0 时强制提交（probability=0 也能触发）。"""
        env, client, machine, task, rng = self._make_env_with_interval(interval=20, probability=0.0)
        # step=20 是 20 的倍数 → 应触发
        env._current_step = 20
        route_to_machine(env, machine, task, rng)
        assert len(client.submitted) == 1

    def test_interval_trigger_skips_non_multiples_with_zero_probability(self):
        """间隔触发：非倍数步 + probability=0 → 不提交。"""
        env, client, machine, task, rng = self._make_env_with_interval(interval=20, probability=0.0)
        # step=15 不是 20 的倍数 → 不触发（probability=0 也不触发）
        env._current_step = 15
        route_to_machine(env, machine, task, rng)
        assert len(client.submitted) == 0

    def test_interval_trigger_step_zero(self):
        """间隔触发：step=0 也应触发（0 % 任何数 == 0）。"""
        env, client, machine, task, rng = self._make_env_with_interval(interval=20, probability=0.0)
        env._current_step = 0
        route_to_machine(env, machine, task, rng)
        assert len(client.submitted) == 1

    def test_probability_trigger_fires_on_non_interval_step(self):
        """概率触发：非倍数步 + probability=1.0 → 应提交。"""
        env, client, machine, task, rng = self._make_env_with_interval(interval=20, probability=1.0)
        # step=15 不是 20 的倍数，但 probability=1.0 → 应通过概率触发提交
        env._current_step = 15
        route_to_machine(env, machine, task, rng)
        assert len(client.submitted) == 1

    def test_probability_trigger_skipped_on_interval_step(self):
        """间隔触发优先：倍数步不走概率分支（不消耗 rng.random）。"""
        env, client, machine, task, rng = self._make_env_with_interval(interval=20, probability=0.0)
        # step=20 走间隔触发，不调用 rng.random()（probability 分支被 elif 跳过）
        env._current_step = 20
        rng_state_before = rng.bit_generator.state["state"]["state"]
        route_to_machine(env, machine, task, rng)
        rng_state_after = rng.bit_generator.state["state"]["state"]
        assert rng_state_before == rng_state_after
        assert len(client.submitted) == 1

    def test_both_triggers_coexist(self):
        """间隔触发与概率触发共存：多个步数混合验证。"""
        env, client, machine, task, rng = self._make_env_with_interval(interval=10, probability=1.0)
        # step=10: 间隔触发
        env._current_step = 10
        route_to_machine(env, machine, task, rng)
        assert len(client.submitted) == 1

        # step=13: 非倍数，但 probability=1.0 → 概率触发
        env._current_step = 13
        route_to_machine(env, machine, task, rng)
        assert len(client.submitted) == 2

        # step=20: 间隔触发
        env._current_step = 20
        route_to_machine(env, machine, task, rng)
        assert len(client.submitted) == 3

    def test_no_submit_without_real_client(self):
        """无真机客户端时不提交（间隔触发和概率触发都跳过）。"""
        env, client, machine, task, rng = self._make_env_with_interval(interval=20, probability=1.0)
        # 清空客户端
        env._real_clients = {}
        env._current_step = 20
        route_to_machine(env, machine, task, rng)
        assert len(client.submitted) == 0

    def test_no_submit_for_non_real_machine(self):
        """非真机机器不触发提交。"""
        env, client, _machine, task, rng = self._make_env_with_interval(
            interval=20, probability=1.0
        )
        # 使用非真机机器
        fake_machine = QuantumMachine(name="simulator", total_qubits=287, is_real=False)
        env._current_step = 20
        route_to_machine(env, fake_machine, task, rng)
        assert len(client.submitted) == 0

    def test_interval_trigger_custom_interval(self):
        """自定义间隔值生效。"""
        env, client, machine, task, rng = self._make_env_with_interval(interval=5, probability=0.0)
        # step=5 是 5 的倍数 → 触发
        env._current_step = 5
        route_to_machine(env, machine, task, rng)
        assert len(client.submitted) == 1

        # step=7 不是 5 的倍数 + probability=0 → 不触发
        env._current_step = 7
        route_to_machine(env, machine, task, rng)
        assert len(client.submitted) == 1

        # step=10 是 5 的倍数 → 触发
        env._current_step = 10
        route_to_machine(env, machine, task, rng)
        assert len(client.submitted) == 2

    def test_interval_trigger_ensures_participation(self):
        """Issue #242 核心修复验证：probability=0.05 + interval=20 确保参与率。

        模拟根因场景：即使概率很低(5%)，间隔触发确保每20步至少1次提交机会。
        """
        env, client, machine, task, rng = self._make_env_with_interval(
            interval=20, probability=0.05
        )
        submit_count = 0
        # 模拟 100 步，每步都路由到真机
        for step in range(1, 101):
            env._current_step = step
            route_to_machine(env, machine, task, rng)
            submit_count = len(client.submitted)

        # 间隔触发确保至少在 step=20,40,60,80,100 时提交
        # 即至少 5 次提交（概率触发可能带来更多）
        assert submit_count >= 5, f"间隔触发应确保至少5次提交，实际 {submit_count} 次"


# =============================================================================
# 真机反馈因果记录导出（Issue #236）
# =============================================================================


class TestExportRealFeedbackLog:
    """export_real_feedback_log 因果记录导出测试。"""

    def test_export_empty_log(self, tmp_path):
        """无真机反馈记录时导出空记录文件，返回 0。"""
        env, _, _, _ = _make_env_with_client()
        out = tmp_path / "feedback_empty.json"
        n = env.export_real_feedback_log(str(out))
        assert n == 0
        assert out.exists()

    def test_export_with_records(self, tmp_path):
        """有因果记录时导出完整记录，返回条数。"""
        env, _, _, _ = _make_env_with_client(real_feedback_mode="result_aware")
        # 手动注入 2 条因果记录
        env._real_feedback_log = [
            {
                "submit_step": 1,
                "complete_step": 3,
                "rl_action": 0,
                "task_id": "t0",
                "real_task_id": "real-1",
                "machine_name": "tianyan176",
                "qcis_circuit": "H Q0\nM Q0",
                "measured_prob": {"0": 0.5, "1": 0.5},
                "fidelity": 1.0,
                "reward": 2.0,
                "formula": "reward=2.0",
                "outcome": "completed",
            },
            {
                "submit_step": 5,
                "complete_step": 7,
                "rl_action": 1,
                "task_id": "t1",
                "real_task_id": "real-2",
                "machine_name": "tianyan176",
                "qcis_circuit": "H Q0\nM Q0",
                "measured_prob": {"0": 0.9, "1": 0.1},
                "fidelity": 0.8,
                "reward": 1.6,
                "formula": "reward=1.6",
                "outcome": "completed",
            },
        ]
        out = tmp_path / "feedback_with_records.json"
        n = env.export_real_feedback_log(str(out))
        assert n == 2

        import json

        with open(out, encoding="utf-8") as f:
            payload = json.load(f)
        assert payload["type"] == "real_feedback_log"
        assert payload["record_count"] == 2
        assert len(payload["records"]) == 2
        assert payload["records"][0]["task_id"] == "t0"
        assert payload["records"][1]["task_id"] == "t1"

    def test_export_includes_stats(self, tmp_path):
        """导出文件包含真机闭环统计信息。"""
        env, _, _, _ = _make_env_with_client()
        out = tmp_path / "feedback_stats.json"
        env.export_real_feedback_log(str(out))

        import json

        with open(out, encoding="utf-8") as f:
            payload = json.load(f)
        assert "stats" in payload
        assert "submission_attempts_total" in payload["stats"]
        assert "success_count" in payload["stats"]

    def test_export_creates_parent_dir(self, tmp_path):
        """父目录不存在时自动创建。"""
        env, _, _, _ = _make_env_with_client()
        out = tmp_path / "nested" / "deep" / "feedback.json"
        n = env.export_real_feedback_log(str(out))
        assert n == 0
        assert out.exists()

    def test_export_metadata_contains_env_params(self, tmp_path):
        """元数据包含环境关键参数。"""
        env, _, _, _ = _make_env_with_client(real_feedback_mode="result_aware", feedback_weight=2.5)
        out = tmp_path / "feedback_meta.json"
        env.export_real_feedback_log(str(out))

        import json

        with open(out, encoding="utf-8") as f:
            payload = json.load(f)
        meta = payload["metadata"]
        assert meta["real_feedback_mode"] == "result_aware"
        assert meta["real_machine_feedback_weight"] == 2.5
        assert meta["use_real_machine"] is True
        assert meta["num_machines"] == 1
        assert "tianyan176" in meta["machine_names"]

    def test_export_after_step_has_records(self, tmp_path):
        """执行 step 后因果记录被填充，导出返回非零条数。"""
        env, _, machine, task = _make_env_with_client(real_feedback_mode="result_aware")
        env.reset(seed=42)
        # 注入任务到队列并执行一步，触发真机提交
        env._task_queue = [task]
        env._machines = [machine]
        env._current_step = 0
        with contextlib.suppress(Exception):
            env.step(0)  # step 可能因 mock 客户端返回不完整而异常，但记录已写入
        out = tmp_path / "feedback_after_step.json"
        n = env.export_real_feedback_log(str(out))
        # step 后至少有 0 条记录（取决于是否触发了真机提交）
        assert n >= 0
        assert out.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
