import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scheduler.ppo_agent import PPOAgent
from src.scheduler.env import QuantumSchedulingEnv, DEFAULT_MACHINE_CONFIGS

def main():
    print("Initializing 16-dim environment...")
    env = QuantumSchedulingEnv(
        max_steps=500,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=42,
    )
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    agent = PPOAgent(
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        use_lstm=False,
        verbose=1,
        seed=42,
    )
    
    print("Training PPO model for 100,000 timesteps...")
    t0 = time.time()
    agent.train(
        total_timesteps=100000,
        eval_freq=50000,
        n_eval_episodes=10,
    )
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")
    
    save_path = os.path.join(PROJECT_ROOT, "deliverable_models", "ppo_best_model_16dim")
    agent.save(save_path)
    print(f"Model saved to {save_path}.zip")

if __name__ == "__main__":
    main()
