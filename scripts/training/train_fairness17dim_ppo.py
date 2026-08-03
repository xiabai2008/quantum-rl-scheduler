"""
17 维公平感知 PPO 模型训练脚本（公平调度"默认关闭"软肋的根治方案）

与 train_16dim_ppo.py 完全同协议（seed=42, 100K steps, PPO-MLP），仅差异：
- include_fairness_obs=True → 观测空间扩展为 17 维（追加 Jain 完成率公平性指数）
- set_fairness_tracker(MultiTenantFairnessTracker) → 公平性惩罚 reward shaping 生效
  （env.py:946 公平性惩罚在 tracker 为 None 时返回 0，必须设置才有学习信号）

产出: deliverable_models/ppo_fairness17dim.zip
用法: python scripts/training/train_fairness17dim_ppo.py
"""

import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv
from src.scheduler.fairness import MultiTenantFairnessTracker
from src.scheduler.ppo_agent import PPOAgent

TENANTS = ["tenant_a", "tenant_b", "tenant_c"]


def main():
    print("Initializing 17-dim fairness-aware environment...")
    env = QuantumSchedulingEnv(
        max_steps=500,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=42,
        include_fairness_obs=True,  # Issue #588: 扩展到 17 维
    )
    env.set_fairness_tracker(MultiTenantFairnessTracker(tenant_ids=TENANTS))

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    assert env.observation_space.shape[0] == 17, (
        f"期望 17 维公平观测，实际 {env.observation_space.shape[0]}"
    )

    agent = PPOAgent(
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        use_lstm=False,
        verbose=1,
        seed=42,
    )

    print("Training 17-dim fairness-aware PPO for 100,000 timesteps...")
    t0 = time.time()
    agent.train(
        total_timesteps=100000,
        eval_freq=50000,
        n_eval_episodes=10,
    )
    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")

    save_path = os.path.join(PROJECT_ROOT, "deliverable_models", "ppo_fairness17dim")
    agent.save(save_path)
    print(f"Model saved to {save_path}.zip")


if __name__ == "__main__":
    main()
