#!/usr/bin/env python3
"""
快速训练验证脚本
在 Mock 模式下跑 5000 步 DQN 训练，验证整条 pipeline 正常
"""

import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scheduler.agent import SchedulerAgent
from src.scheduler.env import QuantumSchedulingEnv

# 清理旧模型
MODEL_DIR = "models/quick_train"
if os.path.exists(MODEL_DIR):
    shutil.rmtree(MODEL_DIR)

os.makedirs(MODEL_DIR, exist_ok=True)

env = QuantumSchedulingEnv(max_qubits=20)
agent = SchedulerAgent(
    env,
    learning_rate=1e-4,
    buffer_size=5000,
    batch_size=32,
    gamma=0.99,
    epsilon_start=1.0,
    epsilon_end=0.05,
    epsilon_decay=0.998,
    verbose=1,
    seed=42,
)

print("\n" + "=" * 60)
print("开始快速训练（5000 步）...")
print("=" * 60 + "\n")

model = agent.train(
    total_timesteps=5000,
    eval_freq=500,
    log_dir="./logs/quick_train",
)

# 保存模型
save_path = os.path.join(MODEL_DIR, "quick_train_model")
agent.save(save_path)

# 评估
eval_result = agent.evaluate(num_episodes=5, deterministic=True)
print("\n评估结果 (5 episodes):")
print(f"  平均奖励: {eval_result['mean_reward']:.2f} +/- {eval_result['std_reward']:.2f}")
print(f"  成功率:   {eval_result['success_rate'] * 100:.1f}%")

# 验证推理
obs, _ = env.reset()
action = agent.predict(obs, deterministic=True)
print(f"\n推理测试: state={obs[:3]}... -> action={action}")
print("\n" + "=" * 60)
print("训练完成!")
print("=" * 60)
