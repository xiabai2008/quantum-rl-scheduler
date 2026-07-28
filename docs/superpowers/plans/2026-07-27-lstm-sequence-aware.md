# LSTM Sequence-Aware Scheduling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide the PPO agent (which already supports RecurrentPPO/LSTM) with historical task arrival rates, allowing it to predict traffic bursts and reserve high-fidelity qubits for upcoming high-priority tasks.

**Architecture:** We will maintain a sliding window of task arrivals in the environment. We'll expand the observation space to include the `ARRIVAL_RATE_MA` (Moving Average). Because `ppo_agent.py` already implements `use_lstm=True`, feeding it this sequence-aware feature will allow the LSTM hidden states to naturally learn burst prediction.

**Tech Stack:** Python, Gymnasium

---

### Task 1: Add Arrival Rate to Observation Space

**Files:**
- Modify: `src/scheduler/env_types.py`
- Modify: `src/scheduler/env_observation.py`

- [ ] **Step 1: Add ARRIVAL_RATE_MA constant**
In `src/scheduler/env_types.py`:
```python
OBS_ARRIVAL_RATE_MA = 15 # or 14 if done independently
OBS_DIM = 16 # Adjust based on previous plans
```

- [ ] **Step 2: Add arrival history to Env state**
In `src/scheduler/env.py`, add `self.arrival_history = []` and `self.current_time_window = 0` to track how many tasks arrived in recent time steps.

- [ ] **Step 3: Update Observation Builder**
In `src/scheduler/env_observation.py`, calculate the moving average of arrivals over the last $N$ steps and append it to the observation array.

- [ ] **Step 4: Commit**
```bash
git add src/scheduler/env_types.py src/scheduler/env.py src/scheduler/env_observation.py
git commit -m "feat: add arrival rate moving average to observation"
```

### Task 2: Track Task Arrivals in Dynamics

**Files:**
- Modify: `src/scheduler/env_dynamics.py`

- [ ] **Step 1: Update arrival tracking**
In `generate_random_task` or the main `step` function where tasks are added to the queue, increment the arrival count for the current time step. 
When `advance_time` is called, push the current count to `arrival_history` (keep max length, e.g., 10) and reset the counter.

- [ ] **Step 2: Commit**
```bash
git add src/scheduler/env_dynamics.py
git commit -m "feat: track task arrival history over time windows"
```

### Task 3: Enable and Verify LSTM Training

**Files:**
- Modify: `src/scheduler/ppo_agent.py`
- Create: `tests/test_lstm_sequence.py`

- [ ] **Step 1: Ensure LSTM is exposed in Training scripts**
Ensure that `training.py` or the main training scripts pass `use_lstm=True` to `PPOAgent` by default or via CLI args.

- [ ] **Step 2: Write Verification Test**
Create `tests/test_lstm_sequence.py` to simulate a bursty task arrival pattern (e.g., 0 tasks for 5 steps, then 10 tasks in 1 step). Verify that the `ARRIVAL_RATE_MA` observation correctly reflects this burst.
Verify that `PPOAgent(use_lstm=True)` can initialize and take a step with the new observation dimension.

- [ ] **Step 3: Commit**
```bash
git add src/scheduler/ppo_agent.py tests/test_lstm_sequence.py
git commit -m "test: verify arrival rate MA and LSTM integration"
```
