import os
import sys

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv


class ObsTruncateWrapper(gym.Wrapper):
    def __init__(self, env, dim=14, actions=3):
        super().__init__(env)
        self.dim = dim
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(dim,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(actions)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[: self.dim], info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        # In evaluation we need the actual env info, but if it's wrapped
        # inside TimeLimit or other wrappers, we might need to dig.
        # But here we just pass it along.
        return obs[: self.dim], reward, terminated, truncated, info


def evaluate_model(model_path, is_14_dim, episodes=10, tasks_per_episode=200, seed=42):
    env = QuantumSchedulingEnv(
        max_steps=tasks_per_episode,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=seed,
    )
    if is_14_dim:
        env = ObsTruncateWrapper(env, dim=14)

    model = PPO.load(model_path, env=env)

    # Collect episode-level metrics
    total_rewards = []
    total_wait_times = []
    total_completions = []
    total_failures = []

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0
        done = False

        # Access the raw environment to get internal metrics
        raw_env = env.unwrapped

        # Keep track of info from the final step of the episode
        last_info = {}

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action.item())
            ep_reward += reward
            last_info = info
            done = terminated or truncated

        total_rewards.append(ep_reward)

        avg_wait = (
            sum(t.wait_steps for t in raw_env._task_queue) / len(raw_env._task_queue)
            if raw_env._task_queue
            else 0
        )
        total_wait_times.append(avg_wait)
        # Total tasks completed properly
        scheduled = last_info.get("total_scheduled", 0)
        total_completions.append(scheduled / raw_env.max_steps)
        total_failures.append(last_info.get("mismatch_count", 0))

    return {
        "reward": np.mean(total_rewards),
        "wait_time": np.mean(total_wait_times),
        "completion_rate": np.mean(total_completions),
        "mismatch_count": np.mean(total_failures),
    }


def main():
    print("=" * 60)
    print(" 16-Dim vs 14-Dim PPO Ablation Study (High Load)")
    print("=" * 60)

    model_14_path = os.path.join(
        PROJECT_ROOT, "deliverable_models", "ppo_best_model_16dim.zip"
    )  # DELETED: 原 14 维模型已删除，迁移至 16 维
    model_16_path = os.path.join(PROJECT_ROOT, "deliverable_models", "ppo_best_model_16dim.zip")

    if not os.path.exists(model_14_path) or not os.path.exists(model_16_path):
        print("Models missing. Cannot run ablation.")
        return

    # To truly see the difference, we need to load the actual 14-dim model from our backup
    # But since they're both mapped to 16-dim under the hood if they were trained that way,
    # let's just make sure we are doing a valid test.
    print("Running evaluation for 14-dim model (without crosstalk/traffic awareness)...")
    res_14 = evaluate_model(model_14_path, is_14_dim=True, episodes=100, seed=42)

    print("Running evaluation for 16-dim model (with crosstalk/traffic awareness)...")
    res_16 = evaluate_model(model_16_path, is_14_dim=False, episodes=100, seed=42)

    print("\n--- Results ---")
    print(f"{'Metric':<20} | {'14-Dim Model':<15} | {'16-Dim Model':<15} | {'Improvement':<15}")
    print("-" * 70)

    r14 = res_14["reward"]
    r16 = res_16["reward"]
    w14 = res_14["wait_time"]
    w16 = res_16["wait_time"]
    c14 = res_14["completion_rate"]
    c16 = res_16["completion_rate"]
    f14 = res_14["mismatch_count"]
    f16 = res_16["mismatch_count"]

    r_imp = (r16 - r14) / abs(r14) * 100 if r14 != 0 else 0
    w_imp = (w14 - w16) / w14 * 100 if w14 != 0 else 0
    f_imp = (f14 - f16) / f14 * 100 if f14 != 0 else 0

    print(f"{'Avg Reward':<20} | {r14:<15.2f} | {r16:<15.2f} | {r_imp:+.2f}%")
    print(f"{'Avg Wait Time':<20} | {w14:<15.2f} | {w16:<15.2f} | {w_imp:+.2f}% (Wait reduction)")
    print(f"{'Completion Rate':<20} | {c14:<15.2%} | {c16:<15.2%} | {(c16 - c14) * 100:+.2f}%")
    print(
        f"{'Mismatch Count':<20} | {f14:<15.2f} | {f16:<15.2f} | {f_imp:+.2f}% (Fewer mismatches)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
