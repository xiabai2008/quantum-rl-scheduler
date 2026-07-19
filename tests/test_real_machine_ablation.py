"""Issue #165 真机消融脚本的无密钥单元测试。"""

from pathlib import Path

import yaml

from scripts.training.run_real_machine_ablation import (
    _convergence_timestep,
    aggregate_condition,
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
