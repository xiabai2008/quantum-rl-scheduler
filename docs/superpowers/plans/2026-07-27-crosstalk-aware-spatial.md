# Crosstalk-Aware Spatial Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable multiple quantum tasks to run concurrently on the same quantum chip (spatial multiplexing), while penalizing the RL agent for placing tasks too close to each other (crosstalk).

**Architecture:** We will modify the `QuantumMachine` state to track active tasks and their mapped qubits. The observation space will expand to include a `CROSSTALK_RISK` metric. If the sum of required qubits of queued tasks is $\le$ machine capacity, they can run concurrently. A crosstalk penalty is applied to the reward based on the density of concurrent tasks.

**Tech Stack:** Python, Gymnasium

---

### Task 1: Update Observation Space and Machine State

**Files:**
- Modify: `src/scheduler/env_types.py`
- Modify: `src/scheduler/env_observation.py`

- [ ] **Step 1: Add CROSSTALK_RISK to constants**
In `src/scheduler/env_types.py`:
```python
OBS_CROSSTALK_RISK = 14
OBS_DIM = 15  # Update from 14 to 15
```

- [ ] **Step 2: Add concurrency tracking to QuantumMachine**
In `src/scheduler/env_types.py`, add `active_tasks: list = field(default_factory=list)` and `used_qubits: int = 0` to `QuantumMachine`.

- [ ] **Step 3: Update Observation Builder**
In `src/scheduler/env_observation.py`, calculate `OBS_CROSSTALK_RISK` based on the ratio of `used_qubits / max_qubits` across machines, and append it to the observation array.

- [ ] **Step 4: Commit**
```bash
git add src/scheduler/env_types.py src/scheduler/env_observation.py
git commit -m "feat: add crosstalk risk to observation space"
```

### Task 2: Implement Spatial Multiplexing Logic

**Files:**
- Modify: `src/scheduler/env_machines.py`
- Modify: `src/scheduler/env_dynamics.py`

- [ ] **Step 1: Modify capacity checks**
In `src/scheduler/env_machines.py`, change `machine_supports_task` to check if `machine.qubits - machine.used_qubits >= task.required_qubits`.

- [ ] **Step 2: Allocate and Free Qubits**
When routing a task to a machine, increment `used_qubits` and append to `active_tasks`. 
In `env_dynamics.py`'s `advance_time` loop, when a task finishes, decrement `used_qubits` and remove it from `active_tasks`.

- [ ] **Step 3: Commit**
```bash
git add src/scheduler/env_machines.py src/scheduler/env_dynamics.py
git commit -m "feat: implement spatial multiplexing capacity logic"
```

### Task 3: Apply Crosstalk Penalty to Rewards

**Files:**
- Modify: `src/scheduler/env_reward.py`

- [ ] **Step 1: Add crosstalk penalty function**
In `src/scheduler/env_reward.py`, modify `compute_execution_reward` to include a penalty.
If a machine has `len(active_tasks) > 1`, calculate a crosstalk factor (e.g., `crosstalk_penalty = 0.1 * (len(active_tasks) - 1)`).
Subtract this penalty from the base fidelity reward.

- [ ] **Step 2: Commit**
```bash
git add src/scheduler/env_reward.py
git commit -m "feat: apply crosstalk penalty to concurrent quantum tasks"
```

### Task 4: Verify Concurrent Execution

**Files:**
- Create: `tests/test_spatial_multiplexing.py`

- [ ] **Step 1: Write concurrent execution test**
Create a test that submits two small quantum tasks to a large quantum machine and verifies they are both marked as running (in `active_tasks`) simultaneously, and that the crosstalk penalty is applied.

- [ ] **Step 2: Commit**
```bash
git add tests/test_spatial_multiplexing.py
git commit -m "test: verify spatial multiplexing and crosstalk penalty"
```
