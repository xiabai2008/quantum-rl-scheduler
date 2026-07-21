"""真机测量结果反馈闭环测试（Issue #235）。

测试覆盖：
1. parse_measurement_result() — 概率分布解析（probability/resultStatus/result 三路径）
2. compute_theoretical_distribution() — 理论分布计算（H 门/X 门/复杂电路）
3. compute_result_fidelity() — 经典保真度计算
4. compute_real_result_reward() — 质量感知 reward 计算
5. shuffle_measurement() — 打乱测量结果（消融对照）
6. 集成测试 — result_aware 模式下 poll_pending_real_tasks 使用测量结果
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_real_machine import (
    compute_real_result_reward,
    compute_result_fidelity,
    compute_theoretical_distribution,
    parse_measurement_result,
    shuffle_measurement,
)
from src.scheduler.env_types import (
    REAL_FEEDBACK_MODES,
    REAL_FEEDBACK_RESULT_AWARE,
    REAL_FEEDBACK_SHUFFLED,
    REAL_FEEDBACK_STATUS_ONLY,
    REAL_MACHINE_SUCCESS_BONUS,
    REAL_RESULT_REWARD_MAX,
    REAL_RESULT_REWARD_MIN,
)

# ============================================================================
# 1. parse_measurement_result 测试
# ============================================================================


class TestParseMeasurementResult:
    """测试从真机状态字典中解析测量概率分布。"""

    def test_parse_probability_field(self) -> None:
        """直接从 probability 字段解析。"""
        status = {"status": "completed", "probability": {"0": 0.5, "1": 0.5}}
        result = parse_measurement_result(status)
        assert result == {"0": 0.5, "1": 0.5}

    def test_parse_probability_normalized(self) -> None:
        """非归一化的 probability 应被归一化。"""
        status = {"probability": {"0": 100, "1": 100}}
        result = parse_measurement_result(status)
        assert pytest.approx(result["0"]) == 0.5
        assert pytest.approx(result["1"]) == 0.5

    def test_parse_result_status_shots(self) -> None:
        """从 resultStatus 原始 shots 计数解析。"""
        status = {
            "status": "completed",
            "resultStatus": json.dumps({"0": 512, "1": 512}),
        }
        result = parse_measurement_result(status)
        assert pytest.approx(result["0"]) == 0.5
        assert pytest.approx(result["1"]) == 0.5

    def test_parse_result_nested(self) -> None:
        """从嵌套 result.probability 解析。"""
        status = {"result": {"probability": {"0": 0.6, "1": 0.4}}}
        result = parse_measurement_result(status)
        assert pytest.approx(result["0"]) == 0.6
        assert pytest.approx(result["1"]) == 0.4

    def test_parse_empty_status(self) -> None:
        """空状态字典返回空字典。"""
        assert parse_measurement_result({}) == {}

    def test_parse_invalid_probability(self) -> None:
        """无效的 probability 值应被跳过。"""
        status = {"probability": {"0": "abc", "1": 0.5}}
        result = parse_measurement_result(status)
        assert "1" in result
        assert "0" not in result

    def test_parse_invalid_result_status(self) -> None:
        """无效的 resultStatus JSON 应返回空字典。"""
        status = {"resultStatus": "not_json"}
        assert parse_measurement_result(status) == {}


# ============================================================================
# 2. compute_theoretical_distribution 测试
# ============================================================================


class TestComputeTheoreticalDistribution:
    """测试理论概率分布计算。"""

    def test_h_gate_uniform(self) -> None:
        """H 门应产生均匀分布。"""
        qcis = "H Q0\nM Q0"
        dist = compute_theoretical_distribution(qcis)
        assert pytest.approx(dist["0"]) == 0.5
        assert pytest.approx(dist["1"]) == 0.5

    def test_x_gate_deterministic(self) -> None:
        """X 门应产生确定态 |1⟩。"""
        qcis = "X Q0\nM Q0"
        dist = compute_theoretical_distribution(qcis)
        assert dist == {"1": 1.0}

    def test_multi_qubit_h_gate(self) -> None:
        """多比特 H 门应产生 2^n 均匀分布。"""
        qcis = "H Q0\nH Q1\nM Q0 Q1"
        dist = compute_theoretical_distribution(qcis)
        assert len(dist) == 4
        for val in dist.values():
            assert pytest.approx(val) == 0.25

    def test_no_measure_returns_default(self) -> None:
        """无测量指令应返回默认分布。"""
        qcis = "H Q0"
        dist = compute_theoretical_distribution(qcis)
        assert dist == {"0": 1.0}


# ============================================================================
# 3. compute_result_fidelity 测试
# ============================================================================


class TestComputeResultFidelity:
    """测试经典保真度计算。"""

    def test_perfect_match(self) -> None:
        """完美匹配的分布保真度应为 1.0。"""
        measured = {"0": 0.5, "1": 0.5}
        theoretical = {"0": 0.5, "1": 0.5}
        fid = compute_result_fidelity(measured, theoretical)
        assert pytest.approx(fid, abs=1e-6) == 1.0

    def test_complete_mismatch(self) -> None:
        """完全不同的分布保真度应为 0.0。"""
        measured = {"0": 1.0}
        theoretical = {"1": 1.0}
        fid = compute_result_fidelity(measured, theoretical)
        assert pytest.approx(fid, abs=1e-6) == 0.0

    def test_partial_match(self) -> None:
        """部分匹配的分布保真度应在 (0, 1) 之间。"""
        measured = {"0": 0.7, "1": 0.3}
        theoretical = {"0": 0.5, "1": 0.5}
        fid = compute_result_fidelity(measured, theoretical)
        assert 0.0 < fid < 1.0
        # 经典保真度 F = (sqrt(0.7*0.5) + sqrt(0.3*0.5))^2
        expected = (0.7 * 0.5) ** 0.5 + (0.3 * 0.5) ** 0.5
        expected = expected**2
        assert pytest.approx(fid, abs=1e-4) == expected

    def test_empty_measured(self) -> None:
        """空测量分布应返回 0.0。"""
        assert compute_result_fidelity({}, {"0": 1.0}) == 0.0

    def test_empty_theoretical(self) -> None:
        """空理论分布应返回 0.0。"""
        assert compute_result_fidelity({"0": 1.0}, {}) == 0.0

    def test_clamped_to_unit(self) -> None:
        """保真度应被 clamp 到 [0, 1]。"""
        measured = {"0": 0.5, "1": 0.5}
        theoretical = {"0": 0.5, "1": 0.5}
        fid = compute_result_fidelity(measured, theoretical)
        assert 0.0 <= fid <= 1.0


# ============================================================================
# 4. compute_real_result_reward 测试
# ============================================================================


class TestComputeRealResultReward:
    """测试质量感知 reward 计算。"""

    def test_perfect_fidelity_max_reward(self) -> None:
        """保真度=1 时 reward 应为最大值。"""
        measured = {"0": 0.5, "1": 0.5}
        theoretical = {"0": 0.5, "1": 0.5}
        reward, fid, _formula = compute_real_result_reward(measured, theoretical)
        assert pytest.approx(fid, abs=1e-6) == 1.0
        assert pytest.approx(reward, abs=1e-4) == REAL_RESULT_REWARD_MAX

    def test_zero_fidelity_min_reward(self) -> None:
        """保真度=0 时 reward 应为最小值。"""
        measured = {"0": 1.0}
        theoretical = {"1": 1.0}
        reward, fid, _formula = compute_real_result_reward(measured, theoretical)
        assert pytest.approx(fid, abs=1e-6) == 0.0
        assert pytest.approx(reward, abs=1e-4) == REAL_RESULT_REWARD_MIN

    def test_empty_measured_min_reward(self) -> None:
        """测量结果解析失败时给最小奖励。"""
        measured: dict[str, float] = {}
        theoretical = {"0": 0.5, "1": 0.5}
        reward, fid, formula = compute_real_result_reward(measured, theoretical)
        assert fid == 0.0
        assert reward == REAL_RESULT_REWARD_MIN
        assert "parse_failed" in formula

    def test_formula_traceable(self) -> None:
        """计算公式应包含 reward 值和保真度。"""
        measured = {"0": 0.5, "1": 0.5}
        theoretical = {"0": 0.5, "1": 0.5}
        _reward, _fid, formula = compute_real_result_reward(measured, theoretical)
        assert "reward=" in formula
        assert "fidelity" in formula

    def test_linear_mapping(self) -> None:
        """reward 应与保真度线性相关。"""
        theoretical = {"0": 0.5, "1": 0.5}
        # 保真度约 0.97 的分布
        measured = {"0": 0.55, "1": 0.45}
        reward, fid, _ = compute_real_result_reward(measured, theoretical)
        quality_range = REAL_RESULT_REWARD_MAX - REAL_RESULT_REWARD_MIN
        expected = REAL_RESULT_REWARD_MIN + fid * quality_range
        assert pytest.approx(reward, abs=1e-4) == expected


# ============================================================================
# 5. shuffle_measurement 测试
# ============================================================================


class TestShuffleMeasurement:
    """测试测量结果打乱（消融对照组）。"""

    def test_shuffle_changes_distribution(self) -> None:
        """打乱后的分布应与原始不同（概率足够大时）。"""
        measured = {"0": 0.8, "1": 0.2, "2": 0.0, "3": 0.0}
        shuffled = shuffle_measurement(measured)
        assert shuffled != measured

    def test_shuffle_preserves_values(self) -> None:
        """打乱后概率值集合不变，只是键重分配。"""
        measured = {"0": 0.7, "1": 0.2, "2": 0.1}
        shuffled = shuffle_measurement(measured)
        assert set(shuffled.values()) == set(measured.values())

    def test_shuffle_single_outcome_unchanged(self) -> None:
        """单结果分布打乱后不变。"""
        measured = {"0": 1.0}
        shuffled = shuffle_measurement(measured)
        assert shuffled == measured

    def test_shuffle_empty(self) -> None:
        """空分布打乱后仍为空。"""
        assert shuffle_measurement({}) == {}


# ============================================================================
# 6. 集成测试 — result_aware 模式
# ============================================================================


class TestRealFeedbackModes:
    """测试三种真机反馈模式的集成行为。"""

    @pytest.fixture
    def mock_env(self) -> QuantumSchedulingEnv:
        """创建测试环境（不使用真机）。"""
        env = QuantumSchedulingEnv(
            use_real_machine=False,
            real_submit_probability=0.0,
        )
        return env

    def test_status_only_mode_default(self, mock_env: QuantumSchedulingEnv) -> None:
        """默认模式应为 status_only。"""
        assert mock_env.real_feedback_mode == REAL_FEEDBACK_STATUS_ONLY

    def test_result_aware_mode(self) -> None:
        """result_aware 模式应正确设置。"""
        env = QuantumSchedulingEnv(
            use_real_machine=False,
            real_feedback_mode=REAL_FEEDBACK_RESULT_AWARE,
        )
        assert env.real_feedback_mode == REAL_FEEDBACK_RESULT_AWARE

    def test_shuffled_mode(self) -> None:
        """shuffled 模式应正确设置。"""
        env = QuantumSchedulingEnv(
            use_real_machine=False,
            real_feedback_mode=REAL_FEEDBACK_SHUFFLED,
        )
        assert env.real_feedback_mode == REAL_FEEDBACK_SHUFFLED

    def test_invalid_mode_raises(self) -> None:
        """无效模式应抛出 ValueError。"""
        with pytest.raises(ValueError, match="real_feedback_mode"):
            QuantumSchedulingEnv(
                use_real_machine=False,
                real_feedback_mode="invalid_mode",
            )

    def test_result_records_initialized(self, mock_env: QuantumSchedulingEnv) -> None:
        """_real_result_records 应在 __init__ 中初始化为空列表。"""
        assert hasattr(mock_env, "_real_result_records")
        assert mock_env._real_result_records == []

    def test_result_records_cleared_on_reset(self, mock_env: QuantumSchedulingEnv) -> None:
        """reset 应清空 _real_result_records。"""
        mock_env._real_result_records = [{"test": True}]
        mock_env.reset(seed=42)
        assert mock_env._real_result_records == []


# ============================================================================
# 7. poll_pending_real_tasks 集成测试（Mock 客户端）
# ============================================================================


class TestPollPendingRealTasksFeedback:
    """测试 poll_pending_real_tasks 在不同反馈模式下的行为。"""

    @pytest.fixture
    def mock_env_with_pending(
        self,
    ) -> tuple[QuantumSchedulingEnv, MagicMock, dict[str, Any]]:
        """创建带 pending 真机任务的环境和 Mock 客户端。"""
        from src.scheduler.env_types import DEFAULT_MACHINE_CONFIGS

        env = QuantumSchedulingEnv(
            machine_configs=DEFAULT_MACHINE_CONFIGS,
            use_real_machine=True,
            real_submit_probability=0.0,
            real_feedback_mode=REAL_FEEDBACK_RESULT_AWARE,
        )
        env.reset(seed=42)

        # 创建 Mock 客户端
        mock_client = MagicMock()
        mock_client.get_task_status.return_value = {
            "status": "completed",
            "probability": {"0": 0.52, "1": 0.48},  # 接近理论 50/50
        }

        # 手动添加 pending 任务
        env._pending_real_tasks = [
            {
                "task_id": "mock_real_task_001",
                "machine_name": "tianyan_s",
                "submit_step": 0,
                "poll_count": 0,
                "task_id_str": "test_task_001",
                "qcis_circuit": "H Q0\nM Q0",
            }
        ]

        return env, mock_client, env._pending_real_tasks[0]

    def test_result_aware_uses_measurement(
        self,
        mock_env_with_pending: tuple[QuantumSchedulingEnv, MagicMock, dict[str, Any]],
    ) -> None:
        """result_aware 模式应解析测量结果并计算保真度。"""
        env, mock_client, _ = mock_env_with_pending

        # 执行轮询
        with patch.object(env, "_real_clients", {"tianyan_s": mock_client}):
            reward = env._poll_pending_real_tasks()

        # 应获得正奖励
        assert reward > 0
        # 应记录结果
        assert len(env._real_result_records) == 1
        record = env._real_result_records[0]
        assert record["fidelity"] is not None
        assert record["fidelity"] > 0.9  # 0.52/0.48 接近 0.5/0.5
        assert record["feedback_mode"] == REAL_FEEDBACK_RESULT_AWARE
        assert record["result_valid"] is True
        assert "reward=" in record["formula"]

    def test_status_only_uses_fixed_bonus(
        self,
    ) -> None:
        """status_only 模式应使用固定 bonus。"""
        from src.scheduler.env_types import DEFAULT_MACHINE_CONFIGS

        env = QuantumSchedulingEnv(
            machine_configs=DEFAULT_MACHINE_CONFIGS,
            use_real_machine=True,
            real_submit_probability=0.0,
            real_feedback_mode=REAL_FEEDBACK_STATUS_ONLY,
        )
        env.reset(seed=42)

        mock_client = MagicMock()
        mock_client.get_task_status.return_value = {
            "status": "completed",
            "probability": {"0": 0.52, "1": 0.48},
        }

        env._pending_real_tasks = [
            {
                "task_id": "mock_real_task_001",
                "machine_name": "tianyan_s",
                "submit_step": 0,
                "poll_count": 0,
                "task_id_str": "test_task_001",
                "qcis_circuit": "H Q0\nM Q0",
            }
        ]

        with patch.object(env, "_real_clients", {"tianyan_s": mock_client}):
            reward = env._poll_pending_real_tasks()

        # 固定 bonus
        expected = REAL_MACHINE_SUCCESS_BONUS * env.real_machine_feedback_weight
        assert pytest.approx(reward) == expected
        # 记录中 fidelity 应为 None
        assert len(env._real_result_records) == 1
        record = env._real_result_records[0]
        assert record["fidelity"] is None
        assert record["feedback_mode"] == REAL_FEEDBACK_STATUS_ONLY
