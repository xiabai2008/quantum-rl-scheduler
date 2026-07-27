import numpy as np
import pytest

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_machines import route_to_machine
from src.scheduler.env_types import ACTION_QUANTUM, ACTION_QUANTUM_QEM, QuantumMachine, Task


def test_qem_effects_direct():
    # Setup a dummy environment and machine for direct testing of route_to_machine
    class MockEnv:
        def __init__(self):
            self._task_queue = []
            self._current_step = 0
            self._quantum = QuantumMachine(
                name="test_q_machine",
                total_qubits=10,
                fidelity=0.9,
                supported_gates=("H", "CZ", "M"),
                is_real=False,
            )
            self._machines = [
                self._quantum
            ]  # For route_to_machine to access machine from env._machines
            self.real_submit_interval = 100
            self.real_submit_probability = 0.0
            self._real_clients = {}  # No real clients for this test
            self._tenant_manager = None  # Add tenant manager for route_to_machine
            self._machine_schedule_count = {
                m.name: 0 for m in self._machines
            }  # For tracking machine schedules

        def _submit_to_real_machine(self, machine, task, *args, **kwargs):
            pass  # Mock real machine submission

    mock_env = MockEnv()
    rng = np.random.default_rng(42)

    # Create a task
    task = Task(
        task_id="test_qem_task",
        task_type="quantum",
        qubit_count=1,
        priority=1,
        execution_time=10.0,
        required_gates=["H"],
    )

    # Clone task and machine states for comparison
    original_exec_time = float(task.execution_time)
    original_fidelity = float(mock_env._quantum.fidelity)

    # Test with QEM action
    # Need to pass a machine instance that is part of the env._machines
    machine_instance_for_qem = mock_env._quantum
    route_to_machine(
        mock_env, machine_instance_for_qem, task, rng, rl_action=ACTION_QUANTUM_QEM, is_qem=True
    )

    # Assertions for QEM effects
    # 1. Task execution time should be multiplied by 3.0
    assert abs(task.execution_time - original_exec_time * 3.0) < 1e-5, (
        f"Expected {original_exec_time * 3.0}, Got {task.execution_time}"
    )

    # 2. Machine fidelity should be improved (error rate halved)
    expected_fidelity = 1.0 - ((1.0 - original_fidelity) * 0.5)
    assert abs(machine_instance_for_qem.fidelity - expected_fidelity) < 1e-5, (
        f"Expected {expected_fidelity}, Got {machine_instance_for_qem.fidelity}"
    )

    print("QEM effects verified successfully via direct call to route_to_machine.")


if __name__ == "__main__":
    test_qem_effects_direct()
