#!/usr/bin/env python
"""诊断 reward 函数问题"""

import sys

sys.path.insert(0, ".")

import numpy as np

from src.scheduler.env import QuantumSchedulingEnv


def diagnose_reward():
    """打印reward明细"""
    print("=" * 60)
    print("诊断 reward 函数")
    print("=" * 60)

    env = QuantumSchedulingEnv(max_qubits=20, max_steps=100)
    obs, _ = env.reset()

    total_reward = 0.0
    rewards = []

    for i in range(10):
        action = np.random.randint(0, 3)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward
        rewards.append(reward)
        print(f"Step {i + 1}: action={action}, reward={reward:.2f}, cumulative={total_reward:.2f}")
        if done:
            break

    print("\n" + "=" * 60)
    print(f"Reward 范围: min={min(rewards):.2f}, max={max(rewards):.2f}")
    print(f"10步累计: {total_reward:.2f}")
    print(f"平均每步: {total_reward / len(rewards):.2f}")
    print("=" * 60)

    # 模拟100步
    env2 = QuantumSchedulingEnv(max_qubits=20, max_steps=100)
    obs, _ = env2.reset()
    total = 0.0
    for i in range(100):
        action = np.random.randint(0, 3)
        _obs, reward, done, _truncated, _info = env2.step(action)
        total += reward
        if done:
            break

    print(f"\n100步随机策略累计reward: {total:.2f}")
    print(f"如果DQN训练10万步，预计累计: {total * 1000:.2f}")

    return total


if __name__ == "__main__":
    diagnose_reward()
