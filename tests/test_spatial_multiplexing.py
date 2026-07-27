import numpy as np
import pytest
from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_types import ACTION_QUANTUM, Task, QuantumMachine
from src.scheduler.env_machines import route_to_machine

def test_spatial_multiplexing():
    # Setup mock env
    class MockEnv:
        def __init__(self):
            # A large machine with 10 qubits
            self._quantum = QuantumMachine(name="tianyan_l", total_qubits=10, fidelity=0.98, available=True, supported_gates=("H",))
            self._machines = [self._quantum]
            self._machine_schedule_count = {self._quantum.name: 0}
            self._tenant_manager = None
            self.real_submit_interval = 100
            self.real_submit_probability = 0.0

        def _submit_to_real_machine(self, *args, **kwargs):
            pass

    env = MockEnv()
    rng = np.random.default_rng(42)

    # Submit first small task (3 qubits)
    task1 = Task(task_id="t1", task_type="quantum", qubit_count=3, execution_time=10.0, required_gates=["H"])
    route_to_machine(env, env._quantum, task1, rng, rl_action=ACTION_QUANTUM)

    # Verify first task is active and used_qubits is updated
    assert len(env._quantum.active_tasks) == 1
    assert env._quantum.active_tasks[0].task_id == "t1"
    assert env._quantum.used_qubits == 3

    # Submit second small task (5 qubits)
    task2 = Task(task_id="t2", task_type="quantum", qubit_count=5, execution_time=10.0, required_gates=["H"])
    route_to_machine(env, env._quantum, task2, rng, rl_action=ACTION_QUANTUM)

    # Verify both tasks are running concurrently
    assert len(env._quantum.active_tasks) == 2
    assert env._quantum.used_qubits == 8

    # Attempt to submit a third task that exceeds capacity (3 qubits needed, only 2 left)
    # Note: capacity checking happens BEFORE route_to_machine in env_machines.py
    # So we'll test the loop logic in env_machines directly or simulate it
    usable_qubits = env._quantum.total_qubits - env._quantum.used_qubits
    assert usable_qubits == 2

    task3 = Task(task_id="t3", task_type="quantum", qubit_count=3, execution_time=10.0, required_gates=["H"])
    can_fit = usable_qubits >= task3.qubit_count
    assert not can_fit, "Task 3 should not fit on the machine"

    print("Spatial multiplexing and capacity tracking verified successfully.")

def test_crosstalk_penalty():
    from src.scheduler.env_reward import compute_execution_reward
    
    # Base case: no crosstalk
    rng1 = np.random.default_rng(42)
    task = Task(task_id="t1", task_type="quantum", qubit_count=3, execution_time=10.0, required_gates=["H"])
    reward_single = compute_execution_reward(task, ACTION_QUANTUM, rng1, quantum_fidelity=0.98, quantum_available_ratio=1.0, crosstalk_penalty=0.0)

    # Case with crosstalk penalty
    rng2 = np.random.default_rng(42)
    reward_concurrent = compute_execution_reward(task, ACTION_QUANTUM, rng2, quantum_fidelity=0.98, quantum_available_ratio=1.0, crosstalk_penalty=0.2)

    print(f"reward_single: {reward_single}, reward_concurrent: {reward_concurrent}")
    assert reward_concurrent < reward_single
    assert abs(reward_single - reward_concurrent - 0.2) < 1e-5

    print("Crosstalk penalty verified successfully.")

if __name__ == "__main__":
    test_spatial_multiplexing()
    test_crosstalk_penalty()