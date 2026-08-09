# Dynamic QEM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow PPO to actively select whether to use Quantum Error Mitigation (QEM) by expanding the action space, trading off execution time for higher fidelity.

**Architecture:** We will introduce a new action `ACTION_QUANTUM_QEM`. When this action is selected, the task's execution time is artificially increased (simulating ZNE/REM overhead), but the base fidelity is improved. The PPO agent will learn when to apply this based on task urgency and hardware noise.

**Tech Stack:** Python, Gymnasium, Stable Baselines 3

---

### Task 1: Expand Action Space in Types

**Files:**
- Modify: `src/scheduler/env_types.py`

- [ ] **Step 1: Add new action constant**
Modify `src/scheduler/env_types.py` to add `ACTION_QUANTUM_QEM`.

```python
# In src/scheduler/env_types.py
ACTION_CLASSICAL = 0  # 分配到经典计算资源
ACTION_QUANTUM = 1  # 分配到量子计算资源
ACTION_HYBRID = 2  # 混合执行
ACTION_QUANTUM_QEM = 3  # 使用误差缓释（QEM）的量子执行
```

- [ ] **Step 2: Commit**
```bash
git add src/scheduler/env_types.py
git commit -m "feat: add ACTION_QUANTUM_QEM to action space"
```

### Task 2: Update Environment Action Space and Logic

**Files:**
- Modify: `src/scheduler/env.py`
- Modify: `src/scheduler/env_dynamics.py`

- [ ] **Step 1: Update Gym Action Space**
In `src/scheduler/env.py`, update `self.action_space = spaces.Discrete(3)` to `spaces.Discrete(4)`.

```python
# In src/scheduler/env.py inside __init__
self.action_space = spaces.Discrete(4)  # Expanded for QEM
```

- [ ] **Step 2: Handle QEM action in environment step**
In `src/scheduler/env.py` (or where the `step` method dispatches actions), map `ACTION_QUANTUM_QEM` to quantum execution but flag it for QEM.

```python
# In src/scheduler/env.py inside step()
is_qem = (action == ACTION_QUANTUM_QEM)
if action in (ACTION_QUANTUM, ACTION_QUANTUM_QEM):
    # route to quantum machine and pass is_qem flag
```
*(Note for implementer: Adapt exactly to how `env.py` routes the task. If it delegates to `env_machines.py`, pass a boolean flag `is_qem=True` to the routing function.)*

- [ ] **Step 3: Commit**
```bash
git add src/scheduler/env.py
git commit -m "feat: update gym action space for QEM"
```

### Task 3: Adjust Dynamics for QEM Penalty and Bonus

**Files:**
- Modify: `src/scheduler/env_machines.py` or `src/scheduler/env_reward.py`

- [ ] **Step 1: Apply QEM effects to Execution**
When calculating the fidelity and execution time for the task, if `is_qem` is True:
1. Multiply the `execution_time` by 3.0 (QEM sampling overhead).
2. Reduce the error rate (1 - fidelity) by half, so `fidelity = 1 - ((1 - original_fidelity) * 0.5)`.

- [ ] **Step 2: Apply to Reward**
Ensure the reward function in `src/scheduler/env_reward.py` captures the updated execution time (higher wait penalty) and updated fidelity (higher execution reward). Since it relies on the machine's fidelity and the task's execution time, updating these metrics during routing will automatically propagate to the reward.

- [ ] **Step 3: Commit**
```bash
git add src/scheduler/env_machines.py src/scheduler/env_reward.py
git commit -m "feat: implement QEM overhead and fidelity bonus"
```

### Task 4: Verify and Test

**Files:**
- Create: `tests/test_qem_action.py`

- [ ] **Step 1: Write verification test**
Create `tests/test_qem_action.py` to ensure taking `ACTION_QUANTUM_QEM` results in higher execution time and better fidelity compared to `ACTION_QUANTUM`.

```python
import numpy as np
from src.scheduler.env import QuantumSchedulerEnv
from src.scheduler.env_types import ACTION_QUANTUM, ACTION_QUANTUM_QEM


def test_qem_effects():
    env = QuantumSchedulerEnv()
    env.reset()
    # Mock a task and force QEM action
    # verify that the resulting task metrics show increased time and improved fidelity
```

- [ ] **Step 2: Commit**
```bash
git add tests/test_qem_action.py
git commit -m "test: add tests for QEM action dynamics"
```
