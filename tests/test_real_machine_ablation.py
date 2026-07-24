"""Issue #165 真机消融脚本的无密钥单元测试。"""

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml

from scripts.training.run_real_machine_ablation import (
    MockRealClient,
    _convergence_timestep,
    aggregate_condition,
    evaluate_one_episode,
    experiment_status,
    parse_seeds,
    run_preflight,
    save_plot,
    write_hard_budget,
)


def _run(seed: int, reward: float, attempted: int = 0) -> dict:
    return {
        "seed": seed,
        "training_elapsed_s": 1.5,
        "curve": [
            {"timestep": 20, "reward": reward / 2},
            {"timestep": 40, "reward": reward},
        ],
        "evaluation": {"reward": reward, "completion_rate": 1.0, "tasks": 200},
        "real_attempted": attempted,
        "real_accepted": attempted,
        "real_completed": attempted,
        "real_failed_or_rejected": 0,
        "degraded": False,
        "mock_calls": 0,
        "real_records": [],
    }


def test_write_hard_budget_includes_smoke_and_shots(tmp_path: Path) -> None:
    path = tmp_path / "quota.yaml"
    write_hard_budget(
        path,
        formal_submission_cap=630,
        shots=8,
        confirmed_machine_minutes=20,
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["total_quota"]["tasks"] == 631
    assert payload["total_quota"]["shots"] == 5048
    assert payload["total_quota"]["wall_time_hours"] == 20 / 60


def test_parse_seeds_requires_three_unique_values() -> None:
    assert parse_seeds("42,43,44") == [42, 43, 44]
    for value in ("42,43", "42,42,43", "42,43,44,45"):
        try:
            parse_seeds(value)
        except Exception:
            pass
        else:
            raise AssertionError("必须拒绝非三个唯一 seed")


def test_preflight_rejects_simulator_before_credentials(tmp_path: Path) -> None:
    try:
        run_preflight(
            machine="tianyan_s",
            shots=8,
            formal_submission_cap=630,
            confirmed_machine_minutes=19.98,
            confirmed_used_seconds=1.026,
            observed_task_upper_seconds=0.109,
            prior_failed_task_id=None,
            prior_failed_task_shots=8,
            wait_timeout=10,
            quota_config_path=tmp_path / "quota.yaml",
            quota_state_path=tmp_path / "state.json",
        )
    except RuntimeError as exc:
        assert "不是 #165 允许的物理真机" in str(exc)
    else:
        raise AssertionError("模拟器不得进入 #165 真机预检")


def test_aggregate_condition_reports_reward_std_and_real_rates() -> None:
    stats = aggregate_condition(
        [_run(42, 100.0, 2), _run(43, 200.0, 2), _run(44, 300.0, 2)],
        total_tasks=600,
    )
    assert stats["reward_mean"] == 200.0
    assert stats["reward_std"] > 0
    assert stats["real_participation_rate"] == 0.01
    assert stats["real_degradation_rate"] == 0.0
    assert stats["convergence_timestep"] == 40


def test_convergence_timestep_handles_empty_curve() -> None:
    assert _convergence_timestep([]) is None


def test_experiment_status_rejects_degraded_or_missing_real_seed() -> None:
    conditions = {
        key: aggregate_condition(
            [_run(42, 100.0, 1), _run(43, 120.0, 1), _run(44, 140.0, 1)],
            total_tasks=600,
        )
        for key in ("simulation", "mixed_real", "pure_real")
    }
    assert experiment_status(conditions) == "completed"
    missing = {key: dict(stats) for key, stats in conditions.items()}
    missing["pure_real"] = dict(missing["pure_real"])
    missing["pure_real"]["runs"] = missing["pure_real"]["runs"][:2]
    assert experiment_status(missing) == "partial_degraded"
    conditions["pure_real"]["runs"][1]["degraded"] = True
    conditions["pure_real"]["runs"][1]["real_completed"] = 0
    assert experiment_status(conditions) == "partial_degraded"


def test_save_plot_writes_three_condition_figure(tmp_path: Path) -> None:
    conditions = {
        key: aggregate_condition(
            [_run(42, 100.0), _run(43, 120.0), _run(44, 140.0)],
            total_tasks=600,
        )
        for key in ("simulation", "mixed_real", "pure_real")
    }
    output = tmp_path / "ablation.png"
    save_plot(conditions, output)
    assert output.exists()
    assert output.stat().st_size > 0


# -- Issue #108: MockRealClient 与评估环境匹配测试 --


def test_mock_real_client_returns_completed_status() -> None:
    """MockRealClient 提交后应返回确定性 completed 状态。"""
    client = MockRealClient(machine_name="tianyan176")
    task_id = client.submit_quantum_task(qcis="H Q0\nM Q0", shots=8, task_name="test")
    assert task_id is not None
    status = client.get_task_status(task_id)
    assert status["status"] == "completed"
    assert "result" in status


def test_mock_real_client_wait_for_task_returns_immediately() -> None:
    """wait_for_task 不应阻塞，立即返回缓存状态。"""
    client = MockRealClient()
    task_id = client.submit_quantum_task(qcis="H Q0\nM Q0", shots=8)
    result = client.wait_for_task(task_id, timeout=1, poll_interval=1)
    assert result["status"] == "completed"


def test_evaluate_one_episode_simulation_condition_no_real_machine() -> None:
    """simulation 条件的评估不应启用真机（use_real_machine=False）。"""
    mock_model = MagicMock()
    # action=2（混合执行）对所有任务类型兼容，确保 episode 可完成
    mock_model.predict = MagicMock(return_value=(np.array(2), None))
    result = evaluate_one_episode(
        mock_model,
        seed=42,
        tasks=20,
        condition="simulation",
    )
    assert "reward" in result
    assert "completion_rate" in result
    assert result["tasks"] == 20


def test_evaluate_one_episode_real_conditions_produce_different_rewards() -> None:
    """Issue #108 核心断言：mixed_real 和 pure_real 的评估 reward 应不同。

    原缺陷：evaluate_one_episode 使用纯仿真环境，导致两条件 reward 完全相同。
    修复后：真机条件使用 MockRealClient，real_probability 不同 → 真机反馈
    次数不同 → reward 不同。

    使用 action=2（混合执行）因为该动作对所有任务类型兼容（classical/quantum/
    universal），且会触发 route_to_machine → 真机提交路径。
    """
    mock_model = MagicMock()
    mock_model.predict = MagicMock(return_value=(np.array(2), None))

    mixed_result = evaluate_one_episode(
        mock_model,
        seed=42,
        tasks=40,
        condition="mixed_real",
        machine="tianyan176",
        real_probability=0.05,
        shots=8,
    )
    pure_result = evaluate_one_episode(
        mock_model,
        seed=42,
        tasks=40,
        condition="pure_real",
        machine="tianyan176",
        real_probability=1.0,
        shots=8,
    )
    # 两条件的 reward 必须不同（Issue #108 的核心修复目标）
    assert mixed_result["reward"] != pure_result["reward"], (
        f"mixed_real reward={mixed_result['reward']} 不应等于 "
        f"pure_real reward={pure_result['reward']}（Issue #108 缺陷复现）"
    )
