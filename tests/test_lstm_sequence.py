import numpy as np
import pytest

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_types import OBS_ARRIVAL_RATE_MA
from src.scheduler.ppo_agent import PPOAgent


def test_arrival_rate_ma():
    env = QuantumSchedulingEnv(arrival_lambda=0.0)  # start with no tasks arriving naturally
    env.reset(seed=42)

    # Inject tasks manually to simulate burst
    # Window 1: 0 tasks
    env.current_time_window_arrivals = 0
    env.arrival_history = [0, 0, 0, 0]
    obs, _, _, _, _ = env.step(0)
    assert obs[OBS_ARRIVAL_RATE_MA] == 0.0

    # Window 2: Burst of 10 tasks
    env.arrival_history = [0, 0, 0, 0, 10]
    obs, _, _, _, _ = env.step(0)

    # After step(), a new 0 is appended because lambda=0
    # So history is [0, 0, 0, 0, 10, 0] -> sum=10, len=6
    expected_ma = (10 / 6) / 10.0

    print(f"MA: {obs[OBS_ARRIVAL_RATE_MA]}, expected: {expected_ma}")
    assert abs(obs[OBS_ARRIVAL_RATE_MA] - expected_ma) < 1e-5

    print("Arrival Rate MA verified successfully.")


def test_lstm_agent_integration():
    env = QuantumSchedulingEnv()

    # Initialize PPOAgent with LSTM enabled
    agent = PPOAgent(env, use_lstm=True, n_steps=128, batch_size=64)
    agent.model = agent._build_model()

    from sb3_contrib import RecurrentPPO

    assert isinstance(agent.model, RecurrentPPO), "Agent should use RecurrentPPO when use_lstm=True"

    # Ensure it can take a step
    obs, _ = env.reset(seed=42)
    action, _states = agent.model.predict(obs, deterministic=True)
    assert action is not None

    print("LSTM Agent integration verified successfully.")


if __name__ == "__main__":
    test_arrival_rate_ma()
    test_lstm_agent_integration()
