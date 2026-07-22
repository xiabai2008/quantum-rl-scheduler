#!/usr/bin/env python3
"""
Issue #46: 14维DQN模型重训 — 修复退化问题 (v4 - 标准Double DQN + 奖励裁剪)

根因分析（v3发现）：
  Dueling DQN的advantage stream对quantum动作(action=1)严重过估计，
  导致greedy策略100%选择quantum。量子资源耗尽后任务堆积，
  等待惩罚累积到-7000（远低于Random的1947）。

修复方案（v4）：
  1. 移除Dueling架构，用标准QNetwork（避免advantage过估计）
  2. 保留Double DQN（在线网络选动作，目标网络估值）
  3. 奖励裁剪到[-1,1]（Atari DQN标准做法，防止单步奖励主导Q值）
  4. 最终探索率0.1（保持replay buffer多样性）
  5. 小网络[64,32]（降低过拟合风险）

用法:
    python scripts/training/train_dqn_14dim.py
    python scripts/training/train_dqn_14dim.py --timesteps 100000 --seed 42
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch as th
import torch.nn.functional as F

# 确保项目根目录在路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from src.scheduler.env import QuantumSchedulingEnv


# ===========================================================================
# Double DQN: 解决Q值过估计
# ===========================================================================

class DoubleDQN(DQN):
    """
    Double DQN: 在线网络选动作，目标网络估值

    标准 DQN target:  r + γ * max_a Q_target(s', a)
    Double DQN target: r + γ * Q_target(s', argmax_a Q_online(s', a))
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                # Double DQN: 在线网络选动作，目标网络估值
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_values.gather(1, next_actions)
                next_q_values = next_q_values.reshape(-1, 1)
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values

            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            self.policy.optimizer.zero_grad()
            loss.backward()
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        self._n_updates += gradient_steps
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))


# ===========================================================================
# 奖励裁剪Wrapper: 裁剪到[-1,1]，防止单步奖励主导Q值
# ===========================================================================

class RewardClipWrapper(gym.Wrapper):
    """将奖励裁剪到[-1, 1]范围（Atari DQN标准做法）"""

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(np.clip(reward, -1.0, 1.0)), terminated, truncated, info


# ===========================================================================
# 训练主逻辑
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Issue #46: 14维DQN重训 (v4)")
    parser.add_argument("--timesteps", type=int, default=100000, help="训练总步数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--save-path",
        type=str,
        default=str(PROJECT_ROOT / "deliverable_models" / "dqn_best_model_14dim"),
        help="模型保存路径（不含扩展名）",
    )
    parser.add_argument("--max-steps", type=int, default=500, help="每episode最大步数")
    parser.add_argument("--max-qubits", type=int, default=287, help="最大量子比特数")
    parser.add_argument("--eval-episodes", type=int, default=20, help="评估episode数")
    return parser.parse_args()


def build_double_dqn(env, seed):
    """构建标准 Double DQN 模型（无Dueling架构）"""
    policy_kwargs = {"net_arch": [128, 64]}

    model = DoubleDQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0001,
        buffer_size=100000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=2000,
        train_freq=(1, "step"),
        learning_starts=1000,
        tau=1.0,
        policy_kwargs=policy_kwargs,
        verbose=1,
        seed=seed,
        tensorboard_log=str(PROJECT_ROOT / "logs" / "dqn_14dim"),
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        exploration_fraction=0.2,
        max_grad_norm=10.0,
    )

    return model


def evaluate_random(env, num_episodes=20, seed=999):
    rng = np.random.RandomState(seed)
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        total_reward = 0.0
        done = False
        while not done:
            action = int(rng.randint(0, env.action_space.n))
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return float(np.mean(rewards)), float(np.std(rewards))


def evaluate_dqn(model, env, num_episodes=20):
    rewards = []
    action_counts = {0: 0, 1: 0, 2: 0}
    for _ in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            a = int(action)
            action_counts[a] = action_counts.get(a, 0) + 1
            obs, reward, terminated, truncated, _ = env.step(a)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return float(np.mean(rewards)), float(np.std(rewards)), action_counts


def evaluate_fcfs(env, num_episodes=20, seed=999):
    rng = np.random.RandomState(seed)
    rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset(seed=int(rng.randint(0, 2**31)))
        total_reward = 0.0
        done = False
        while not done:
            action = 2
            obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
    return float(np.mean(rewards)), float(np.std(rewards))


def main():
    args = parse_args()

    print("=" * 64)
    print("  Issue #46: 14维DQN模型重训 (v4 - 标准Double DQN + 奖励裁剪)")
    print("=" * 64)
    print(f"  算法:           Double DQN (标准QNetwork, 无Dueling)")
    print(f"  训练步数:       {args.timesteps}")
    print(f"  随机种子:       {args.seed}")
    print(f"  观测维度:       14")
    print(f"  学习率:         0.0001")
    print(f"  缓冲区大小:     100000")
    print(f"  网络架构:       [128, 64]")
    print(f"  目标更新间隔:   2000")
    print(f"  探索:           1.0→0.1 (前20%步数)")
    print(f"  奖励裁剪:       [-1, 1]")
    print(f"  梯度裁剪:       10.0")
    print(f"  保存路径:       {args.save_path}.zip")
    print(f"  时间:           {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 64)

    # ── 1. 创建14维环境 ──────────────────────────────────────
    base_env = QuantumSchedulingEnv(
        max_steps=args.max_steps,
        max_qubits=args.max_qubits,
        seed=args.seed,
    )
    obs_dim = base_env.observation_space.shape[0]
    print(f"[ENV] 观测维度: {obs_dim}, 动作空间: {base_env.action_space.n}")

    # ── 2. 评估基线 ─────────────────────────────────────────
    print("\n[BASELINE] 评估基线策略...")
    random_mean, random_std = evaluate_random(base_env, num_episodes=args.eval_episodes)
    print(f"  Random: {random_mean:.2f} ± {random_std:.2f}")

    fcfs_mean, fcfs_std = evaluate_fcfs(base_env, num_episodes=args.eval_episodes)
    print(f"  FCFS:   {fcfs_mean:.2f} ± {fcfs_std:.2f}")

    # ── 3. 创建裁剪奖励环境用于训练 ─────────────────────────
    train_env = Monitor(RewardClipWrapper(QuantumSchedulingEnv(
        max_steps=args.max_steps,
        max_qubits=args.max_qubits,
        seed=args.seed,
    )))
    model = build_double_dqn(train_env, args.seed)
    print(f"\n[DQN] Double DQN (标准QNetwork) 构建完成, net_arch=[128, 64]")

    # 评估回调
    eval_env = Monitor(RewardClipWrapper(QuantumSchedulingEnv(
        max_steps=args.max_steps,
        max_qubits=args.max_qubits,
        seed=args.seed + 1000,
    )))
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=str(PROJECT_ROOT / "logs" / "dqn_14dim" / "best_model"),
        log_path=str(PROJECT_ROOT / "logs" / "dqn_14dim" / "eval_results"),
        eval_freq=max(args.timesteps // 10, 1000),
        n_eval_episodes=5,
        deterministic=True,
    )

    # ── 4. 训练 ─────────────────────────────────────────────
    print(f"\n[TRAIN] 开始训练 {args.timesteps} steps...")
    t0 = time.time()
    model.learn(
        total_timesteps=args.timesteps,
        callback=eval_callback,
        tb_log_name=f"dqn_14dim_v4_std_seed{args.seed}",
        reset_num_timesteps=True,
    )
    train_time = time.time() - t0
    print(f"[TRAIN] 训练完成，耗时 {train_time:.1f}s ({train_time / 60:.1f}min)")

    # ── 5. 保存最终模型 ─────────────────────────────────────
    save_dir = os.path.dirname(args.save_path)
    os.makedirs(save_dir, exist_ok=True)
    model.save(args.save_path)
    print(f"[SAVE] 最终模型已保存: {args.save_path}.zip")

    # ── 6. 评估最终模型和最佳模型 ───────────────────────────
    print("\n[EVAL] 评估模型 (deterministic)...")
    dqn_mean, dqn_std, action_counts = evaluate_dqn(model, base_env, num_episodes=args.eval_episodes)
    print(f"  DQN (final): {dqn_mean:.2f} ± {dqn_std:.2f}")
    print(f"  Action distribution: {action_counts}")

    # 也评估 best_model
    best_path = PROJECT_ROOT / "logs" / "dqn_14dim" / "best_model" / "best_model.zip"
    if best_path.exists():
        best_model = DoubleDQN.load(str(best_path), env=train_env)
        best_mean, best_std, best_actions = evaluate_dqn(best_model, base_env, num_episodes=args.eval_episodes)
        print(f"  DQN (best):  {best_mean:.2f} ± {best_std:.2f}")
        print(f"  Best action distribution: {best_actions}")

        # 选择更好的模型
        if best_mean > dqn_mean:
            print(f"\n  [INFO] Best model is better, using it as final model")
            best_model.save(args.save_path)
            dqn_mean, dqn_std, action_counts = best_mean, best_std, best_actions

    print(f"\n  Random:      {random_mean:.2f} ± {random_std:.2f}")
    print(f"  FCFS:        {fcfs_mean:.2f} ± {fcfs_std:.2f}")

    improvement_vs_random = ((dqn_mean - random_mean) / abs(random_mean) * 100) if random_mean != 0 else 0.0
    improvement_vs_fcfs = ((dqn_mean - fcfs_mean) / abs(fcfs_mean) * 100) if fcfs_mean != 0 else 0.0
    print(f"\n  DQN vs Random: {'+' if improvement_vs_random > 0 else ''}{improvement_vs_random:.1f}%")
    print(f"  DQN vs FCFS:   {'+' if improvement_vs_fcfs > 0 else ''}{improvement_vs_fcfs:.1f}%")

    # ── 7. 退化判定 ─────────────────────────────────────────
    print("\n" + "=" * 64)
    if dqn_mean > random_mean * 1.1:
        print("  [PASS] DQN显著优于Random，退化问题已修复!")
    elif dqn_mean > random_mean:
        print("  [WARN] DQN略优于Random，但优势不明显")
    else:
        print("  [FAIL] DQN未优于Random，仍存在退化问题")
    print("=" * 64)

    # ── 8. 保存训练报告 ─────────────────────────────────────
    report = {
        "issue": "#46",
        "version": "v4",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "algorithm": "Double DQN (standard QNetwork) + Reward Clipping [-1,1]",
        "config": {
            "timesteps": args.timesteps,
            "seed": args.seed,
            "obs_dim": obs_dim,
            "learning_rate": 0.0001,
            "buffer_size": 100000,
            "batch_size": 64,
            "net_arch": [128, 64],
            "architecture": "Double DQN (no Dueling)",
            "target_update_interval": 2000,
            "exploration_fraction": 0.2,
            "exploration_final_eps": 0.1,
            "max_grad_norm": 10.0,
            "learning_starts": 1000,
            "reward_clip": "[-1, 1]",
        },
        "results": {
            "random_mean": round(random_mean, 2),
            "random_std": round(random_std, 2),
            "fcfs_mean": round(fcfs_mean, 2),
            "fcfs_std": round(fcfs_std, 2),
            "dqn_mean": round(dqn_mean, 2),
            "dqn_std": round(dqn_std, 2),
            "improvement_vs_random_pct": round(improvement_vs_random, 2),
            "improvement_vs_fcfs_pct": round(improvement_vs_fcfs, 2),
            "train_time_seconds": round(train_time, 1),
            "action_distribution": action_counts,
        },
        "model_path": f"{args.save_path}.zip",
        "degradation_fixed": dqn_mean > random_mean * 1.1,
    }

    report_path = PROJECT_ROOT / "results" / "reports" / "dqn_14dim_retrain_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[REPORT] 训练报告已保存: {report_path}")


if __name__ == "__main__":
    main()
