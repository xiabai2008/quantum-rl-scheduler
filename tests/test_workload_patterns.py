"""Issue #160 多负载参数注入测试。"""

from unittest.mock import Mock

import numpy as np
import pytest

from scripts.evaluation.run_workload_pattern_comparison import tidal_arrival_lambda
from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_dynamics import generate_random_task


class RecordingGenerator:
    """记录 poisson 参数，其余随机方法委托给 NumPy Generator。"""

    def __init__(self, seed: int = 42) -> None:
        self._generator = np.random.default_rng(seed)
        self.poisson_lambda: float | None = None

    def poisson(self, lam: float) -> int:
        self.poisson_lambda = float(lam)
        return 0

    def __getattr__(self, name: str):
        return getattr(self._generator, name)


def test_default_environment_behavior_is_preserved() -> None:
    env = QuantumSchedulingEnv()
    assert env._get_arrival_lambda() == pytest.approx(1.2)
    assert env.quantum_task_ratio is None


@pytest.mark.parametrize("value", [-0.01, -1.0])
def test_negative_arrival_lambda_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        QuantumSchedulingEnv(arrival_lambda=value)


@pytest.mark.parametrize("value", [-0.01, 1.01])
def test_invalid_quantum_ratio_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match="between 0 and 1"):
        QuantumSchedulingEnv(quantum_task_ratio=value)


def test_arrival_lambda_reaches_environment_dynamics() -> None:
    env = QuantumSchedulingEnv(arrival_lambda=0.2)
    env.reset(seed=42)
    rng = RecordingGenerator()
    env._advance_time(rng)  # type: ignore[arg-type]
    assert rng.poisson_lambda == pytest.approx(0.2)


def test_callback_arrival_lambda_uses_current_step_and_horizon() -> None:
    schedule = Mock(return_value=0.35)
    env = QuantumSchedulingEnv(max_steps=200, arrival_lambda=schedule)
    env._current_step = 17
    assert env._get_arrival_lambda() == pytest.approx(0.35)
    schedule.assert_called_once_with(17, 200)


def test_negative_callback_result_is_rejected() -> None:
    env = QuantumSchedulingEnv(arrival_lambda=lambda _step, _max_steps: -0.1)
    with pytest.raises(ValueError, match="negative"):
        env._get_arrival_lambda()


def test_quantum_ratio_controls_generated_task_type() -> None:
    rng = np.random.default_rng(42)
    quantum_tasks = [generate_random_task(rng, index, 1.0) for index in range(20)]
    classical_tasks = [generate_random_task(rng, index, 0.0) for index in range(20)]
    assert all(task.task_type == "quantum" for task in quantum_tasks)
    assert all(task.task_type == "classical" for task in classical_tasks)
    assert all(task.qubit_count == 0 for task in classical_tasks)


def test_tidal_schedule_has_expected_range_and_period() -> None:
    assert tidal_arrival_lambda(0, 200) == pytest.approx(1.0)
    assert tidal_arrival_lambda(100, 200) == pytest.approx(0.1)
    assert tidal_arrival_lambda(200, 200) == pytest.approx(1.0)
    values = [tidal_arrival_lambda(step, 200) for step in range(201)]
    assert min(values) >= 0.1
    assert max(values) <= 1.0
