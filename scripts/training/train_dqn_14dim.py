#!/usr/bin/env python
"""
14维 DQN 模型重训脚本（修复退化问题）
14-dim DQN Model Retrain Script (Fix Degradation)

Issue #46: 当前 10-dim DQN 在 14-dim 环境中退化为 Random。
本脚本训练 14-dim 标准 Double DQN（无 Dueling），
使用 reward clipping [-1,1] 防止 Q 值过高估计。

训练完成后自动评估并与 Random/FCFS 基线对比，
输出统计检验结果。

网络架构: [128, 64]（兼容 SchedulerAgent.NET_ARCH）
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.dqn import DQN

# ── 确保项目根目录在路径中 ──
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import gymnasium as gym

from src.scheduler.baselines import FCFSScheduler
from src.scheduler.env import QuantumSchedulingEnv

# ── 常量 ──
TOTAL_TIMESTEPS = 100_000
EVAL_FREQ = 5_000
N_EVAL_EPISODES = 10
N_FINAL_EPISODES = 50
NUM_SEEDS = 5  # 多 seed 评估
NET_ARCH: list[int] = [128, 64]  # 兼容 SchedulerAgent.NET_ARCH
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "deliverable_models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results", "reports")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs", "dqn_14dim_retrain")

# ── 竞赛权威数字（禁止修改） ──
AUTHORITATIVE_PPO_MEAN = 2746.94
AUTHORITATIVE_PPO_STD = 1121.19
AUTHORITATIVE_FCFS_MEAN = 1458.77
AUTHORITATIVE_FCFS_STD = 55.85
AUTHORITATIVE_IMPROVEMENT = 88.3  # %


# ============================================================================
# 奖励裁剪包装器
# ============================================================================


class RewardClipWrapper(gym.Wrapper):
    """将奖励裁剪到 [-1, 1] 区间，防止大额等待惩罚主导 Q 值训练。"""

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs, float(np.clip(reward, -1.0, 1.0)), terminated, truncated, info


# ============================================================================
# FCFS 策略（基于环境任务队列）
# ============================================================================


class FCFSEnvStrategy:
    """FCFS 调度策略：基于环境任务队列选择最早到达的任务并映射到动作。

    使用 baselines.FCFSScheduler 选择最早到达的任务，
    然后根据任务类型映射到动作空间（0=经典/1=量子/2=混合）。
    若无任务，默认选择混合执行（action=2）。
    """

    name = "FCFS"

    def __init__(self) -> None:
        """初始化 FCFS 策略。"""
        self._scheduler = FCFSScheduler()

    def select_action(self, env: QuantumSchedulingEnv) -> int:
        """根据环境任务队列选择动作。

        Args:
            env: 量子调度环境实例

        Returns:
            动作索引（0=经典, 1=量子, 2=混合）
        """
        task_queue = getattr(env, "_task_queue", [])
        if not task_queue:
            return 2  # 默认混合执行

        tasks_for_scheduler = []
        for i, task in enumerate(task_queue):
            tasks_for_scheduler.append(
                {
                    "task_id": getattr(task, "task_id", str(i)),
                    "arrival_time": float(getattr(task, "wait_steps", 0)),
                    "task_type": getattr(task, "task_type", "hybrid"),
                }
            )

        selected_idx = self._scheduler.select_action(tasks_for_scheduler, {})
        if selected_idx < 0 or selected_idx >= len(task_queue):
            return 2

        task_type = getattr(task_queue[selected_idx], "task_type", "hybrid")
        if task_type == "quantum":
            return 1
        elif task_type == "classical":
            return 0
        else:
            return 2


# ============================================================================
# Double DQN（标准实现，无 Dueling）
# ============================================================================


class DoubleDQN(DQN):
    """标准 Double DQN：online 网络选动作，target 网络估值。

    与 Dueling DQN 不同，此实现使用标准 QNetwork，
    避免 advantage stream 对量子动作的系统性高估。
    """

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        """Double DQN 训练步骤。

        Args:
            gradient_steps: 梯度更新步数
            batch_size: 批次大小
        """
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        losses = []

        for _ in range(gradient_steps):
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discounts = replay_data.discounts if replay_data.discounts is not None else self.gamma

            with th.no_grad():
                next_q_online = self.q_net(replay_data.next_observations)
                next_actions = next_q_online.argmax(dim=1, keepdim=True)
                next_q_values = self.q_net_target(replay_data.next_observations)
                next_q_values = next_q_values.gather(1, next_actions).reshape(-1, 1)
                target_q_values = (
                    replay_data.rewards + (1 - replay_data.dones) * discounts * next_q_values
                )

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


# ============================================================================
# 辅助函数
# ============================================================================


def build_double_dqn(
    env: gym.Env,
    seed: int,
    tensorboard_log: str | None = None,
) -> DoubleDQN:
    """构建标准 Double DQN 模型（无 Dueling）。

    Args:
        env: 训练环境
        seed: 随机种子
        tensorboard_log: TensorBoard 日志目录

    Returns:
        配置好的 DoubleDQN 模型
    """
    return DoubleDQN(
        policy="MlpPolicy",
        env=env,
        learning_rate=0.0001,
        buffer_size=100_000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=2_000,
        train_freq=(1, "step"),
        learning_starts=1_000,
        tau=1.0,
        policy_kwargs={"net_arch": NET_ARCH},
        verbose=1,
        seed=seed,
        tensorboard_log=tensorboard_log,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.1,
        exploration_fraction=0.2,
        max_grad_norm=10.0,
    )


def evaluate_model(
    model: DoubleDQN,
    env: gym.Env,
    num_episodes: int,
    seed: int,
) -> tuple[float, float, list[float], dict[int, int]]:
    """评估模型性能。

    Args:
        model: 训练好的模型
        env: 评估环境
        num_episodes: 评估回合数
        seed: 随机种子

    Returns:
        (mean_reward, std_reward, all_rewards, action_distribution)
    """
    rewards = []
    action_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            action_counts[int(action)] = action_counts.get(int(action), 0) + 1
            obs, reward, terminated, truncated, _info = env.step(int(action))
            ep_reward += float(reward)
            done = terminated or truncated
        rewards.append(ep_reward)

    mean = float(np.mean(rewards))
    std = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
    return mean, std, rewards, action_counts


def evaluate_random(
    env: QuantumSchedulingEnv,
    num_episodes: int,
    seed: int,
) -> tuple[float, float, list[float]]:
    """评估 Random 基线。

    Args:
        env: 评估环境
        num_episodes: 评估回合数
        seed: 随机种子

    Returns:
        (mean, std, rewards)
    """
    rewards = []
    for ep in range(num_episodes):
        _obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action = env.action_space.sample()
            _obs, reward, terminated, truncated, _info = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated
        rewards.append(ep_reward)

    mean = float(np.mean(rewards))
    std = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
    return mean, std, rewards


def evaluate_fcfs(
    env: QuantumSchedulingEnv,
    num_episodes: int,
    seed: int,
) -> tuple[float, float, list[float], dict[int, int]]:
    """评估 FCFS 基线（使用真实 FCFSScheduler）。

    Args:
        env: 评估环境
        num_episodes: 评估回合数
        seed: 随机种子

    Returns:
        (mean, std, rewards, action_distribution)
    """
    strategy = FCFSEnvStrategy()
    rewards = []
    action_counts: dict[int, int] = {0: 0, 1: 0, 2: 0}

    for ep in range(num_episodes):
        _obs, _ = env.reset(seed=seed + ep)
        ep_reward = 0.0
        done = False
        while not done:
            action = strategy.select_action(env)
            action_counts[int(action)] = action_counts.get(int(action), 0) + 1
            _obs, reward, terminated, truncated, _info = env.step(action)
            ep_reward += float(reward)
            done = terminated or truncated
        rewards.append(ep_reward)

    mean = float(np.mean(rewards))
    std = float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0
    return mean, std, rewards, action_counts


def welch_t_test(
    group_a: list[float],
    group_b: list[float],
) -> tuple[float, float, float]:
    """Welch t 检验 + Cohen's d 效应量。

    Args:
        group_a: 组A（如 DQN）
        group_b: 组B（如 Random）

    Returns:
        (t_statistic, p_value, cohens_d)
    """
    from scipy import stats

    a = np.array(group_a, dtype=np.float64)
    b = np.array(group_b, dtype=np.float64)
    t_stat, p_value = stats.ttest_ind(a, b, equal_var=False)

    n_a, n_b = len(a), len(b)
    pooled_std = np.sqrt(
        ((n_a - 1) * np.var(a, ddof=1) + (n_b - 1) * np.var(b, ddof=1)) / (n_a + n_b - 2)
    )
    cohens_d = (np.mean(a) - np.mean(b)) / pooled_std if pooled_std > 0 else 0.0

    return float(t_stat), float(p_value), float(cohens_d)


def improvement_pct(a: float, b: float) -> float:
    """计算相对提升百分比。"""
    return (a - b) / abs(b) * 100.0 if b != 0 else 0.0


# ============================================================================
# 主流程
# ============================================================================


def main() -> None:
    """主流程：训练 → 多 seed 评估 → 统计检验 → 输出报告。"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    print("=" * 60)
    print("  14维 DQN 模型重训（修复退化问题）")
    print("  Issue #46")
    print("=" * 60)

    # ── 训练 ──
    print("\n[1/4] 训练 Double DQN（100k 步）...")
    train_seed = 42
    train_env = RewardClipWrapper(QuantumSchedulingEnv(max_steps=500, seed=train_seed))
    eval_env = Monitor(QuantumSchedulingEnv(max_steps=500, seed=train_seed + 999))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=LOG_DIR,
        log_path=LOG_DIR,
        eval_freq=EVAL_FREQ,
        n_eval_episodes=N_EVAL_EPISODES,
        deterministic=True,
    )

    model = build_double_dqn(train_env, train_seed, tensorboard_log=LOG_DIR)
    t0 = time.time()
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=eval_callback,
        tb_log_name="dqn_14dim",
        reset_num_timesteps=True,
    )
    train_time = time.time() - t0
    print(f"  训练完成 ({train_time:.1f}s)")

    # ── 多 seed 评估 ──
    print(f"\n[2/4] 多 seed 评估（{NUM_SEEDS} seeds × {N_FINAL_EPISODES} episodes）...")
    base_seed = 1000
    dqn_rewards_all: list[float] = []
    random_rewards_all: list[float] = []
    fcfs_rewards_all: list[float] = []
    dqn_actions: dict[int, int] = {0: 0, 1: 0, 2: 0}
    fcfs_actions: dict[int, int] = {0: 0, 1: 0, 2: 0}

    for s in range(NUM_SEEDS):
        seed = base_seed + s
        print(f"  Seed {seed}...")

        # DQN
        dqn_env = QuantumSchedulingEnv(max_steps=500, seed=seed)
        _, _, dqn_r, dqn_act = evaluate_model(model, dqn_env, N_FINAL_EPISODES, seed)
        dqn_rewards_all.extend(dqn_r)
        for k, v in dqn_act.items():
            dqn_actions[k] = dqn_actions.get(k, 0) + v

        # Random
        rnd_env = QuantumSchedulingEnv(max_steps=500, seed=seed)
        _, _, rnd_r = evaluate_random(rnd_env, N_FINAL_EPISODES, seed)
        random_rewards_all.extend(rnd_r)

        # FCFS
        fcfs_env = QuantumSchedulingEnv(max_steps=500, seed=seed)
        _, _, fcfs_r, fcfs_act = evaluate_fcfs(fcfs_env, N_FINAL_EPISODES, seed)
        fcfs_rewards_all.extend(fcfs_r)
        for k, v in fcfs_act.items():
            fcfs_actions[k] = fcfs_actions.get(k, 0) + v

    dqn_mean = float(np.mean(dqn_rewards_all))
    dqn_std = float(np.std(dqn_rewards_all, ddof=1))
    random_mean = float(np.mean(random_rewards_all))
    random_std = float(np.std(random_rewards_all, ddof=1))
    fcfs_mean = float(np.mean(fcfs_rewards_all))
    fcfs_std = float(np.std(fcfs_rewards_all, ddof=1))

    # ── 统计检验 ──
    print("\n[3/4] 统计检验...")
    dqn_vs_random_t, dqn_vs_random_p, dqn_vs_random_d = welch_t_test(
        dqn_rewards_all, random_rewards_all
    )
    dqn_vs_fcfs_t, dqn_vs_fcfs_p, dqn_vs_fcfs_d = welch_t_test(dqn_rewards_all, fcfs_rewards_all)
    fcfs_vs_random_t, fcfs_vs_random_p, fcfs_vs_random_d = welch_t_test(
        fcfs_rewards_all, random_rewards_all
    )

    dqn_vs_random_imp = improvement_pct(dqn_mean, random_mean)
    dqn_vs_fcfs_imp = improvement_pct(dqn_mean, fcfs_mean)
    fcfs_vs_random_imp = improvement_pct(fcfs_mean, random_mean)

    # ── 结果输出 ──
    print("\n[4/4] 结果汇总...")
    print(f"\n  {'策略':<12} {'平均奖励':>10} {'标准差':>10} {'提升vs Random':>14}")
    print(f"  {'-' * 48}")
    print(f"  {'DQN':<12} {dqn_mean:>10.2f} {dqn_std:>10.2f} {dqn_vs_random_imp:>+13.1f}%")
    print(f"  {'FCFS':<12} {fcfs_mean:>10.2f} {fcfs_std:>10.2f} {fcfs_vs_random_imp:>+13.1f}%")
    print(f"  {'Random':<12} {random_mean:>10.2f} {random_std:>10.2f} {'——':>13}")

    print("\n  统计检验:")
    print(
        f"  DQN vs Random : t={dqn_vs_random_t:.2f}, p={dqn_vs_random_p:.2e}, d={dqn_vs_random_d:.2f}"
    )
    print(f"  DQN vs FCFS   : t={dqn_vs_fcfs_t:.2f}, p={dqn_vs_fcfs_p:.2e}, d={dqn_vs_fcfs_d:.2f}")
    print(
        f"  FCFS vs Random: t={fcfs_vs_random_t:.2f}, p={fcfs_vs_random_p:.2e}, d={fcfs_vs_random_d:.2f}"
    )

    print(f"\n  DQN 动作分布: {dict(dqn_actions)}")
    print(f"  FCFS 动作分布: {dict(fcfs_actions)}")

    # 判断退化是否修复
    degradation_fixed = dqn_mean > random_mean and dqn_vs_random_p < 0.05
    print(
        f"\n  {'[PASS] DQN显著优于Random，退化问题已修复!' if degradation_fixed else '[FAIL] DQN退化问题仍存在，需进一步调试'}"
    )

    # ── 保存模型 ──
    model_path = os.path.join(OUTPUT_DIR, "dqn_best_model_14dim")
    model.save(model_path)
    print(f"\n  模型已保存: {model_path}.zip")

    # ── 生成报告 ──
    report = {
        "title": "14维DQN模型重训报告",
        "issue": "#46",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "algorithm": "Double DQN (no Dueling)",
            "net_arch": NET_ARCH,
            "total_timesteps": TOTAL_TIMESTEPS,
            "learning_rate": 0.0001,
            "buffer_size": 100_000,
            "batch_size": 64,
            "gamma": 0.99,
            "target_update_interval": 2_000,
            "exploration_final_eps": 0.1,
            "exploration_fraction": 0.2,
            "reward_clipping": [-1.0, 1.0],
            "num_seeds": NUM_SEEDS,
            "episodes_per_seed": N_FINAL_EPISODES,
            "train_time_seconds": round(train_time, 1),
        },
        "results": {
            "dqn": {
                "mean": round(dqn_mean, 2),
                "std": round(dqn_std, 2),
                "action_distribution": {str(k): v for k, v in sorted(dqn_actions.items())},
            },
            "random": {
                "mean": round(random_mean, 2),
                "std": round(random_std, 2),
            },
            "fcfs": {
                "mean": round(fcfs_mean, 2),
                "std": round(fcfs_std, 2),
                "action_distribution": {str(k): v for k, v in sorted(fcfs_actions.items())},
            },
            "comparisons": {
                "dqn_vs_random": {
                    "improvement_pct": round(dqn_vs_random_imp, 1),
                    "t_stat": round(dqn_vs_random_t, 2),
                    "p_value": dqn_vs_random_p,
                    "cohens_d": round(dqn_vs_random_d, 2),
                },
                "dqn_vs_fcfs": {
                    "improvement_pct": round(dqn_vs_fcfs_imp, 1),
                    "t_stat": round(dqn_vs_fcfs_t, 2),
                    "p_value": dqn_vs_fcfs_p,
                    "cohens_d": round(dqn_vs_fcfs_d, 2),
                },
                "fcfs_vs_random": {
                    "improvement_pct": round(fcfs_vs_random_imp, 1),
                    "t_stat": round(fcfs_vs_random_t, 2),
                    "p_value": fcfs_vs_random_p,
                    "cohens_d": round(fcfs_vs_random_d, 2),
                },
            },
        },
        "degradation_fixed": degradation_fixed,
        "authoritative_reference": {
            "ppo_mean": AUTHORITATIVE_PPO_MEAN,
            "ppo_std": AUTHORITATIVE_PPO_STD,
            "fcfs_mean": AUTHORITATIVE_FCFS_MEAN,
            "fcfs_std": AUTHORITATIVE_FCFS_STD,
            "improvement": AUTHORITATIVE_IMPROVEMENT,
            "note": "本报告DQN数据为14维环境独立评估，与权威数字（10维Obs10Wrapper环境）不可直接数值对比，仅用于验证DQN > Random（退化已修复）",
        },
    }

    report_path = os.path.join(RESULTS_DIR, "dqn_14dim_retrain_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"  报告已保存: {report_path}")


if __name__ == "__main__":
    main()
