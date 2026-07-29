#!/usr/bin/env python
"""
D3 奖励函数训练消融实验（严格版，Issue #58 修复）
Ablation D3: Reward Function Training Ablation (Strict Version, Issue #58 fix)

与 ablation_d3_reward.py（推理消融）的区别：
    - 推理消融：用已训练好的 PPO 模型，改变 reward 常量后推理
      → 只能回答"L4 训练的 PPO 在不同奖励下评估性能"
    - 训练消融：每个奖励配置从头训练 PPO
      → 能回答"不同奖励训练出的 PPO 性能差异"（#223 要求）

四层奖励配置（从简到全）：
    - L1 基础兼容奖励：仅任务完成基础分（兼容基线）
    - L2 +执行收益：量子/经典资源利用效率奖励
    - L3 +等待惩罚：任务等待时间惩罚
    - L4 +利用率奖励：整体资源利用率奖励（当前完整版本）

Issue #58 修复要点：
    1. 用 RewardPatch 上下文管理器 monkey-patch 模块级常量（env_types/env/env_reward 三处）
       —— 原 patch_env_rewards() 写入 env._classical_reward 等不存在的实例属性，完全无效
    2. Callback 记录真实 self.num_timesteps，不再用 rollout 数 × 1000 估算收敛步数
    3. 不再依赖 self.model.ep_info_buffer 等 SB3 内部实现
    4. DEFAULT_REWARDS 启动时与 env_types 实际常量逐项核对，不一致立即失败
    5. quick 模式仅用于代码冒烟，不能作为正式消融结果
    6. 正式实验：4 层配置 × 10 seeds × 50000 steps × 5 episodes，paired Wilcoxon + Holm 校正
    7. 不保存模型 zip / checkpoint

用法：
    # 完整运行（约 16-17h）
    python scripts/evaluation/ablation_d3_training.py --seeds 10 --episodes 5 --timesteps 50000

    # 快速冒烟（仅 1 seed × 1 episode × 1k steps，约 2min）—— 不可作为正式结果
    python scripts/evaluation/ablation_d3_training.py --quick

    # 自定义输出目录
    python scripts/evaluation/ablation_d3_training.py --output-dir results/ablation_d3_training_v2
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(str(_PROJECT_ROOT))

# 复用现有基础设施
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "evaluation"))
from run_issue_38_67_experiments import (
    Obs10Wrapper,
    SimulationEnv,
    SimulationTaskGenerator,
)

from src.scheduler import env as env_module
from src.scheduler import env_reward as env_reward_module
from src.scheduler import env_types as env_types_module
from src.scheduler.env import QuantumSchedulingEnv

# ============================================================================
# 四层奖励配置（L1 → L4，逐步叠加）
# ============================================================================

# DEFAULT_REWARDS：与 env_types.py 实际常量逐项对应（启动时由 _verify_default_rewards 校验）
# 键名 = 脚本内部名，值 = 应与 env_types.py 一致的基线值
DEFAULT_REWARDS = {
    "classical_reward": 5.0,  # ↔ REWARD_CLASSICAL
    "quantum_reward": 10.0,  # ↔ REWARD_QUANTUM_BASE
    "hybrid_reward": 7.0,  # ↔ REWARD_HYBRID
    "success_bonus": 3.0,  # ↔ REWARD_SUCCESS_BONUS
    "mismatch_penalty": -2.0,  # ↔ REWARD_MISMATCH
    "wait_penalty": -0.1,  # ↔ REWARD_WAIT_OVER_THRESHOLD
    "low_util_penalty": -1.0,  # ↔ REWARD_LOW_QUBIT_UTIL
}

# 脚本内部键 → env_types 模块常量名映射
_REWARD_KEY_TO_CONST: dict[str, str] = {
    "classical_reward": "REWARD_CLASSICAL",
    "quantum_reward": "REWARD_QUANTUM_BASE",
    "hybrid_reward": "REWARD_HYBRID",
    "success_bonus": "REWARD_SUCCESS_BONUS",
    "mismatch_penalty": "REWARD_MISMATCH",
    "wait_penalty": "REWARD_WAIT_OVER_THRESHOLD",
    "low_util_penalty": "REWARD_LOW_QUBIT_UTIL",
}

# 每个常量在哪个模块被实际读取：
# - env_reward 模块读取：compute_execution_reward / compute_wait_penalty
# - env 模块读取：step() 中直接引用
# - env_types 模块：源模块（同步 patch 以保证一致性）
_ENV_REWARD_ATTRS: set[str] = {
    "REWARD_CLASSICAL",
    "REWARD_QUANTUM_BASE",
    "REWARD_HYBRID",
    "REWARD_SUCCESS_BONUS",
    "REWARD_WAIT_OVER_THRESHOLD",
}
_ENV_ATTRS: set[str] = {
    "REWARD_LOW_QUBIT_UTIL",
    "REWARD_MISMATCH",
}

# 四层配置（#223 要求的严格 L1-L4 设计）
ABLATION_LAYERS = {
    "L1_basic": {
        # L1 基础兼容奖励：仅任务完成基础分
        "classical_reward": 5.0,
        "quantum_reward": 5.0,
        "hybrid_reward": 5.0,
        "success_bonus": 0.0,
        "mismatch_penalty": 0.0,
        "wait_penalty": 0.0,
        "low_util_penalty": 0.0,
    },
    "L2_execution": {
        # L2 +执行收益：量子/经典资源利用效率奖励
        "classical_reward": 5.0,
        "quantum_reward": 10.0,
        "hybrid_reward": 7.0,
        "success_bonus": 3.0,
        "mismatch_penalty": -2.0,
        "wait_penalty": 0.0,
        "low_util_penalty": 0.0,
    },
    "L3_wait_penalty": {
        # L3 +等待惩罚：任务等待时间惩罚
        "classical_reward": 5.0,
        "quantum_reward": 10.0,
        "hybrid_reward": 7.0,
        "success_bonus": 3.0,
        "mismatch_penalty": -2.0,
        "wait_penalty": -0.1,
        "low_util_penalty": 0.0,
    },
    "L4_full": {
        # L4 +利用率奖励：整体资源利用率奖励（当前完整版本）
        **DEFAULT_REWARDS,
    },
}


# ============================================================================
# DEFAULT_REWARDS 与 env_types 实际常量的一致性校验
# ============================================================================


def _verify_default_rewards() -> None:
    """启动时校验 DEFAULT_REWARDS 与 env_types.py 实际常量值一致。

    Issue #58 要求：L4_full 的 DEFAULT_REWARDS 必须与实际环境默认值一致，
    不一致立即失败，不能继续训练。
    """
    mismatches: list[str] = []
    for script_key, const_name in _REWARD_KEY_TO_CONST.items():
        if not hasattr(env_types_module, const_name):
            mismatches.append(f"env_types.{const_name} 不存在（脚本键={script_key}）")
            continue
        actual = float(getattr(env_types_module, const_name))
        expected = float(DEFAULT_REWARDS[script_key])
        if not math.isclose(actual, expected, abs_tol=1e-9):
            mismatches.append(
                f"{script_key} ({const_name}): 脚本默认={expected}, env_types 实际={actual}"
            )
    if mismatches:
        raise RuntimeError(
            "DEFAULT_REWARDS 与 env_types.py 实际常量不一致，禁止启动训练：\n  - "
            + "\n  - ".join(mismatches)
        )


# ============================================================================
# RewardPatch 上下文管理器：monkey-patch 模块级常量
# ============================================================================


class RewardPatch:
    """上下文管理器：临时覆盖模块级奖励常量，退出时恢复。

    奖励常量在 env_types.py 中定义，被 env.py 和 env_reward.py 通过
    ``from env_types import REWARD_*`` 导入为各自模块的命名空间绑定。
    必须同时 patch env_types / env / env_reward 三个模块的命名空间才能
    保证运行时读取到新值。

    Issue #58 修复：原 patch_env_rewards(env, rewards) 写入 env._classical_reward
    等不存在的实例属性，env.step() 从未读取这些属性 → patch 完全无效。
    """

    _ALL_ATTRS: ClassVar[set[str]] = _ENV_REWARD_ATTRS | _ENV_ATTRS

    def __init__(self, overrides: dict[str, float]) -> None:
        self._overrides = overrides
        self._saved: dict[str, dict[str, float]] = {
            "env_types": {},
            "env_reward": {},
            "env": {},
        }

    def __enter__(self) -> RewardPatch:
        for const_name, val in self._overrides.items():
            if const_name not in self._ALL_ATTRS:
                raise KeyError(f"未知奖励常量名: {const_name}")

            # 逐模块预检：常量必须已存在于目标模块，否则抛 AttributeError（Issue #58 要求）
            # 1. env_types（源模块）
            if not hasattr(env_types_module, const_name):
                raise AttributeError(f"env_types.{const_name} 不存在，无法 monkey-patch")
            self._saved["env_types"][const_name] = getattr(env_types_module, const_name)
            setattr(env_types_module, const_name, val)

            # 2. env_reward（若该模块导入了此常量）
            if hasattr(env_reward_module, const_name):
                self._saved["env_reward"][const_name] = getattr(env_reward_module, const_name)
                setattr(env_reward_module, const_name, val)

            # 3. env（若该模块导入了此常量）
            if hasattr(env_module, const_name):
                self._saved["env"][const_name] = getattr(env_module, const_name)
                setattr(env_module, const_name, val)

        return self

    def __exit__(self, *args: Any) -> None:
        # 逆序恢复，确保原值被还原
        for const_name, val in self._saved["env"].items():
            setattr(env_module, const_name, val)
        for const_name, val in self._saved["env_reward"].items():
            setattr(env_reward_module, const_name, val)
        for const_name, val in self._saved["env_types"].items():
            setattr(env_types_module, const_name, val)


def _rewards_to_overrides(rewards: dict[str, float]) -> dict[str, float]:
    """将脚本内部奖励键名转换为 env_types 常量名。"""
    return {_REWARD_KEY_TO_CONST[k]: float(v) for k, v in rewards.items()}


def make_env_with_rewards(
    rewards: dict[str, float],
    tasks_per_episode: int = 200,
    seed: int | None = None,
    obs_dim: int = 10,
) -> Any:
    """创建带指定奖励配置的环境。

    必须在 RewardPatch 上下文内调用，且训练/评估也需在同一上下文内进行。
    """
    base = QuantumSchedulingEnv(
        max_steps=tasks_per_episode,
        max_qubits=287,
        seed=seed,
    )
    if obs_dim == 10:
        return Obs10Wrapper(base)
    return base


# ============================================================================
# PPO 训练（每配置每 seed 独立训练）
# ============================================================================


def train_ppo_with_rewards(
    rewards: dict[str, float],
    seed: int,
    timesteps: int,
    tasks_per_episode: int = 200,
    obs_dim: int = 10,
) -> tuple[Any, dict[str, Any]]:
    """用指定奖励配置训练 PPO。

    必须在 RewardPatch 上下文内调用，确保训练过程中 env.step() 读取到新奖励常量。
    不再保存模型 zip（Issue #58：不提交模型 checkpoint）。
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback

    # 创建带奖励 patch 的环境
    env = make_env_with_rewards(
        rewards=rewards,
        tasks_per_episode=tasks_per_episode,
        seed=seed,
        obs_dim=obs_dim,
    )

    # 训练进度回调：记录真实 num_timesteps + episode reward（来自 Monitor 的 infos）
    # Issue #58 修复：
    #   - 旧实现用 self.model.ep_info_buffer（SB3 内部实现，版本敏感）
    #   - 旧实现用 len(reward_curve) * 1000 估算收敛步数（不准确）
    # 新实现：
    #   - 直接读取 self.num_timesteps（BaseCallback 公开属性）
    #   - 从 self.locals["infos"] 提取 Monitor 写入的 "episode" key（公开 API）
    timestep_reward_pairs: list[tuple[int, float]] = []

    class RewardLoggerCallback(BaseCallback):
        def _on_rollout_end(self) -> None:
            # 真实训练步数（BaseCallback 公开属性，非估算）
            current_step = int(self.num_timesteps)
            # 从 locals["infos"] 提取本 rollout 内完成的 episode 奖励
            # （Monitor wrapper 在 episode 结束时写入 info["episode"]["r"]）
            infos = self.locals.get("infos", [])
            for info in infos:
                if not isinstance(info, dict):
                    continue
                ep_info = info.get("episode")
                if isinstance(ep_info, dict) and "r" in ep_info:
                    timestep_reward_pairs.append((current_step, float(ep_info["r"])))

        def _on_step(self) -> bool:
            return True

    # PPO 超参数（与项目现有训练一致）
    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        verbose=0,
        seed=seed,
    )

    start_time = time.time()
    model.learn(total_timesteps=timesteps, callback=RewardLoggerCallback())
    train_time = time.time() - start_time

    # 真实收敛步数：基于 (timestep, reward) 对检测，不再用 rollout 数 × 1000
    convergence_step = _detect_convergence_from_pairs(timestep_reward_pairs)
    final_train_reward = timestep_reward_pairs[-1][1] if timestep_reward_pairs else 0.0

    train_info = {
        "seed": seed,
        "timesteps_requested": timesteps,
        "timesteps_actual": int(model.num_timesteps),  # 真实训练步数
        "train_time_seconds": round(train_time, 2),
        "final_train_reward": final_train_reward,
        "timestep_reward_pairs": [
            {"timestep": int(ts), "reward": float(r)} for ts, r in timestep_reward_pairs
        ],
        "convergence_step": convergence_step,
        "num_rollouts": len(timestep_reward_pairs),
    }

    env.close()
    return model, train_info


def _detect_convergence_from_pairs(
    pairs: list[tuple[int, float]],
    window: int = 5,
    threshold: float = 0.05,
) -> int:
    """基于真实 (timestep, reward) 对检测收敛步数。

    Issue #58 修复：旧实现用 len(reward_curve) * 1000 估算，新实现直接返回
    满足收敛判据时的真实 timestep。

    Args:
        pairs: (timestep, reward) 列表，按时间顺序
        window: 滑动窗口大小
        threshold: 收敛判据（窗口内变异系数 CV < threshold）

    Returns:
        收敛时的真实 timestep；若无收敛点则返回最后一个 timestep；若无数据返回 0
    """
    if len(pairs) < window * 2:
        return int(pairs[-1][0]) if pairs else 0

    rewards = [r for _, r in pairs]
    timesteps = [ts for ts, _ in pairs]

    for i in range(window, len(pairs)):
        window_vals = rewards[i - window : i]
        mean_v = float(np.mean(window_vals))
        if mean_v == 0:
            continue
        cv = float(np.std(window_vals)) / abs(mean_v)
        if cv < threshold:
            return int(timesteps[i])

    return int(timesteps[-1]) if timesteps else 0


# ============================================================================
# 训练后评估
# ============================================================================


def evaluate_model(
    model: Any,
    rewards: dict[str, float],
    seed: int,
    episodes: int = 5,
    tasks_per_episode: int = 200,
    obs_dim: int = 10,
) -> dict[str, Any]:
    """评估训练好的模型。必须在 RewardPatch 上下文内调用。"""
    env = make_env_with_rewards(
        rewards=rewards,
        tasks_per_episode=tasks_per_episode,
        seed=seed + 10000,  # 评估用不同 seed 避免过拟合
        obs_dim=obs_dim,
    )
    sim_env = SimulationEnv(env=env, task_generator=SimulationTaskGenerator(seed=seed + 10000))

    ep_rewards = []
    for ep in range(episodes):
        obs, info = sim_env.reset(seed=seed + ep)
        ep_reward = 0.0
        step = 0
        while step < tasks_per_episode:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = sim_env.step(int(action))
            ep_reward += reward
            step += 1
            if terminated or truncated:
                break
        ep_rewards.append(float(ep_reward))
        sim_env.record_episode_stats(info)

    summary = sim_env.get_summary()
    summary["ep_rewards"] = ep_rewards
    summary["mean_reward"] = float(np.mean(ep_rewards))
    summary["std_reward"] = float(np.std(ep_rewards, ddof=1)) if len(ep_rewards) > 1 else 0.0
    env.close()
    return summary


# ============================================================================
# Holm-Bonferroni 校正（手动实现，无 statsmodels 依赖）
# ============================================================================


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> tuple[list[bool], list[float]]:
    """Holm-Bonferroni step-down 多重比较校正。

    比简单 Bonferroni 更强大，控制 FWER。
    """
    n = len(p_values)
    if n == 0:
        return [], []
    # 按 p 值升序排序
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    rejected = [False] * n
    adj_p = [0.0] * n

    for rank, (orig_idx, p) in enumerate(indexed):
        # Holm 校正：adj_p = p * (n - rank)
        corrected = p * (n - rank)
        # 单调性：校正后的 p 不能小于前一个
        if rank > 0:
            prev_orig_idx = indexed[rank - 1][0]
            corrected = max(corrected, adj_p[prev_orig_idx])
        # 上限 1.0
        adj_p[orig_idx] = min(1.0, corrected)

    # Step-down 决策：一旦不拒绝，后续全部不拒绝
    stop = False
    for rank, (orig_idx, _) in enumerate(indexed):
        if stop:
            rejected[orig_idx] = False
            continue
        threshold = alpha / (n - rank)
        if adj_p[orig_idx] <= threshold:
            rejected[orig_idx] = True
        else:
            rejected[orig_idx] = False
            stop = True

    return rejected, adj_p


# ============================================================================
# 主实验流程
# ============================================================================


def run_d3_training_ablation(
    seeds: int = 10,
    episodes: int = 5,
    timesteps: int = 50000,
    tasks_per_episode: int = 200,
    obs_dim: int = 10,
    output_dir: Path | None = None,
    alpha: float = 0.05,
    quick: bool = False,
) -> dict[str, Any]:
    """运行 D3 训练消融主实验。

    配置：4 层 × N seeds × M episodes
    Issue #58：quick=True 时仅作冒烟，结果不可作为正式消融结论。
    """
    # 启动时校验 DEFAULT_REWARDS 与 env_types 实际常量一致
    _verify_default_rewards()

    if output_dir is None:
        output_dir = _PROJECT_ROOT / "results" / "ablation_d3_training"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_list = [42 + i * 137 for i in range(seeds)]
    config_names = list(ABLATION_LAYERS.keys())

    print("=" * 72)
    print("  D3 奖励函数训练消融实验（严格版，Issue #58 修复）")
    print("=" * 72)
    print(f"  配置数:        {len(config_names)} ({', '.join(config_names)})")
    print(f"  Seeds:         {seeds} ({seed_list})")
    print(f"  Episodes:      {episodes}")
    print(f"  训练步数:      {timesteps}")
    print(f"  任务规模:      {tasks_per_episode} 步/episode")
    print(f"  观测维度:      {obs_dim}")
    print(f"  总训练量:      {len(config_names) * seeds * timesteps} steps")
    if quick:
        print("  [冒烟模式] 仅用于代码冒烟，结果不可作为正式消融结论")
    print("=" * 72)

    all_results: dict[str, dict] = {}
    start_time = time.time()

    for config_name in config_names:
        rewards = ABLATION_LAYERS[config_name]
        overrides = _rewards_to_overrides(rewards)
        print(f"\n{'=' * 72}")
        print(f"  配置: {config_name}")
        print(f"  奖励常量: {rewards}")
        print(f"  Overrides: {overrides}")
        print(f"{'=' * 72}")

        config_data: dict[str, Any] = {
            "rewards": rewards,
            "overrides": overrides,
            "seeds": {},
        }

        for seed_idx, seed in enumerate(seed_list):
            print(f"\n  --- {config_name} | Seed {seed_idx + 1}/{seeds} (seed={seed}) ---")
            seed_start = time.time()

            # 关键：env 创建 + 训练 + 评估必须全部在 RewardPatch 上下文内
            with RewardPatch(overrides):
                model, train_info = train_ppo_with_rewards(
                    rewards=rewards,
                    seed=seed,
                    timesteps=timesteps,
                    tasks_per_episode=tasks_per_episode,
                    obs_dim=obs_dim,
                )
                eval_info = evaluate_model(
                    model=model,
                    rewards=rewards,
                    seed=seed,
                    episodes=episodes,
                    tasks_per_episode=tasks_per_episode,
                    obs_dim=obs_dim,
                )

            seed_elapsed = time.time() - seed_start
            config_data["seeds"][str(seed)] = {
                "train": train_info,
                "eval": eval_info,
                "elapsed_seconds": round(seed_elapsed, 2),
            }

            print(
                f"  完成 ({seed_elapsed:.1f}s) | "
                f"train_actual_steps={train_info['timesteps_actual']} | "
                f"train_final={train_info['final_train_reward']:.1f} | "
                f"eval_mean={eval_info['mean_reward']:.1f}±{eval_info['std_reward']:.1f}"
            )

        # 配置汇总（seed 级聚合：每 seed 取其 mean_reward，构成 N=seeds 的样本）
        all_rewards = [config_data["seeds"][str(s)]["eval"]["mean_reward"] for s in seed_list]
        config_data["summary"] = {
            "mean_reward": float(np.mean(all_rewards)),
            "std_reward": float(np.std(all_rewards, ddof=1)) if len(all_rewards) > 1 else 0.0,
            "all_rewards": all_rewards,
            "n_seeds": len(all_rewards),
        }
        all_results[config_name] = config_data
        print(
            f"\n  {config_name} 汇总: mean={config_data['summary']['mean_reward']:.2f}"
            f"±{config_data['summary']['std_reward']:.2f} (N={len(all_rewards)} seeds)"
        )

    total_elapsed = time.time() - start_time
    print(f"\n所有配置完成，总耗时 {total_elapsed:.1f}s ({total_elapsed / 3600:.2f}h)")

    # ========================================================================
    # 统计显著性检验：配对 Wilcoxon + Holm-Bonferroni 校正（L4 vs L1/L2/L3）
    # ========================================================================
    print("\n" + "=" * 72)
    print(f"  统计显著性检验（配对 Wilcoxon + Holm-Bonferroni, α={alpha}）")
    print("=" * 72)

    from scipy import stats as scipy_stats

    sig_results: dict[str, Any] = {}
    l4_rewards = all_results["L4_full"]["summary"]["all_rewards"]

    comparisons = [
        ("L4_vs_L1_basic", "L1_basic"),
        ("L4_vs_L2_execution", "L2_execution"),
        ("L4_vs_L3_wait_penalty", "L3_wait_penalty"),
    ]

    raw_p_values: list[float] = []
    pair_data: list[tuple[str, str, list[float], list[float]]] = []

    for pair_key, cfg_name in comparisons:
        cfg_rewards = all_results[cfg_name]["summary"]["all_rewards"]
        diffs = [a - b for a, b in zip(l4_rewards, cfg_rewards, strict=True)]
        non_zero_diffs = [d for d in diffs if d != 0]

        # 配对 Wilcoxon（仅非零差值）
        if len(non_zero_diffs) >= 5:
            wilcoxon_result = scipy_stats.wilcoxon(non_zero_diffs)
            p_value = float(wilcoxon_result.pvalue)
            statistic = float(wilcoxon_result.statistic)
        else:
            p_value = float("nan")
            statistic = float("nan")

        # Cohen's d_z（配对效应量）
        if len(diffs) >= 2:
            std_diff = float(np.std(diffs, ddof=1))
            dz = float(np.mean(diffs) / std_diff) if std_diff != 0 else 0.0
        else:
            dz = 0.0

        sig_results[pair_key] = {
            "comparison": f"L4_full vs {cfg_name}",
            "test": "Wilcoxon signed-rank test (paired)",
            "statistic": statistic,
            "p_value_raw": p_value,
            "mean_diff": float(np.mean(diffs)),
            "effect_size_type": "Cohen's d_z",
            "effect_size": dz,
            "n_pairs": len(diffs),
            "n_nonzero_diffs": len(non_zero_diffs),
            "diffs": diffs,
        }
        raw_p_values.append(p_value)
        pair_data.append((pair_key, cfg_name, l4_rewards, cfg_rewards))

    # Holm-Bonferroni 校正
    valid_p_values = [p for p in raw_p_values if not math.isnan(p)]
    if valid_p_values:
        rejected, adj_p = holm_bonferroni(valid_p_values, alpha=alpha)
        # 回填到 sig_results（按 valid 顺序）
        valid_idx = 0
        for pair_key, _, _, _ in pair_data:
            raw_p = sig_results[pair_key]["p_value_raw"]
            if math.isnan(raw_p):
                sig_results[pair_key]["p_value_adjusted"] = float("nan")
                sig_results[pair_key]["holm_significant"] = False
            else:
                sig_results[pair_key]["p_value_adjusted"] = adj_p[valid_idx]
                sig_results[pair_key]["holm_significant"] = bool(rejected[valid_idx])
                valid_idx += 1
            sig_results[pair_key]["judgment"] = (
                "支持" if sig_results[pair_key].get("holm_significant", False) else "不支持"
            )
    else:
        for pair_key, _, _, _ in pair_data:
            sig_results[pair_key]["p_value_adjusted"] = float("nan")
            sig_results[pair_key]["holm_significant"] = False
            sig_results[pair_key]["judgment"] = "不支持（数据不足）"

    # 打印统计结果
    for _pair_key, info in sig_results.items():
        sig_mark = "✅" if info.get("holm_significant", False) else "❌"
        print(
            f"  {sig_mark} {info['comparison']}: "
            f"p_raw={info['p_value_raw']:.4f}, "
            f"p_adj={info.get('p_value_adjusted', float('nan')):.4f}, "
            f"d_z={info['effect_size']:.4f}, "
            f"Δ={info['mean_diff']:+.2f} "
            f"(n={info['n_pairs']}, 非零={info['n_nonzero_diffs']}) "
            f"判定={info['judgment']}"
        )

    # ========================================================================
    # 保存结果
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_json = {
        "config": {
            "experiment": "D3 Reward Function Training Ablation (Issue #58 fix)",
            "seeds": seed_list,
            "episodes_per_seed": episodes,
            "timesteps": timesteps,
            "tasks_per_episode": tasks_per_episode,
            "obs_dim": obs_dim,
            "alpha": alpha,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "timestamp": timestamp,
            "quick_mode": bool(quick),
            "multiple_comparison_correction": "Holm-Bonferroni",
            "effect_size": "Cohen's d_z (paired)",
            "n_comparisons": len(comparisons),
            "note": (
                "Issue #58 修复版：timesteps_actual 记录真实 num_timesteps，"
                "不再用 rollout × 1000 估算；统计为 seed 级配对设计 N=seeds"
            ),
        },
        "ablation_layers": ABLATION_LAYERS,
        "default_rewards_verified": True,
        "results": all_results,
        "significance": sig_results,
        "statistical_design": {
            "unit_of_analysis": "seed (5 episodes aggregated to seed mean)",
            "n_seeds_per_config": seeds,
            "paired_design": "same seed across configs",
            "main_test": "Wilcoxon signed-rank test (paired)",
            "multiple_comparison": "Holm-Bonferroni step-down",
            "effect_size": "Cohen's d_z",
        },
    }

    results_path = output_dir / f"d3_training_ablation_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2, default=str)
    canonical_path = output_dir / "d3_training_ablation.json"
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[保存] 结果: {results_path}")
    print(f"[保存] 规范路径: {canonical_path}")

    # 生成报告
    _generate_report(results_json, output_dir, timestamp)
    print(f"[保存] 报告: {output_dir / 'd3_training_ablation.md'}")

    return results_json


# ============================================================================
# 报告生成
# ============================================================================


def _generate_report(results: dict, output_dir: Path, timestamp: str) -> None:
    """生成 Markdown 报告。"""
    cfg = results["config"]
    all_results = results["results"]
    sig = results["significance"]

    lines = [
        "# D3 奖励函数训练消融实验报告（严格版，Issue #58 修复）",
        "",
        f"> **生成时间**: {timestamp}",
        f"> **实验规模**: {len(cfg['seeds'])} seeds × {cfg['episodes_per_seed']} episodes",
        f">  统计单元: seed 级（每 seed 聚合 5 episodes 均值 → N={len(cfg['seeds'])}）",
        f"> **训练步数**: {cfg['timesteps']} steps（实际记录 num_timesteps，非估算）",
        f"> **任务规模**: {cfg['tasks_per_episode']} 步/episode",
        f"> **观测维度**: {cfg['obs_dim']}",
        f"> **总耗时**: {cfg['total_elapsed_seconds'] / 3600:.2f}h",
        f"> **多重比较校正**: {cfg['multiple_comparison_correction']}",
        f"> **效应量**: {cfg['effect_size']}",
        "> **方法**: 每配置每 seed 独立训练 PPO（MlpPolicy, lr=3e-4, n_steps=2048）",
        "> **奖励 patch**: monkey-patch env_types/env/env_reward 三模块常量",
        "> **DEFAULT_REWARDS 校验**: 启动时与 env_types 实际值逐项核对一致",
    ]

    if cfg.get("quick_mode", False):
        lines.extend(
            [
                "",
                "> ⚠️ **冒烟模式**：本结果仅用于代码冒烟，不可作为正式消融结论。",
            ]
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 一、实验目的",
            "",
            "回答 #223 的核心问题：**不同奖励配置训练出的 PPO 性能差异**。",
            "与 #201 的推理消融不同，本实验通过重新训练回答『L4 奖励设计是否最优』。",
            "",
            "## 二、四层奖励配置",
            "",
            "| 配置 | classical | quantum | hybrid | success | mismatch | wait | low_util |",
            "|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|",
        ]
    )

    for name, rewards in results["ablation_layers"].items():
        lines.append(
            f"| {name} | {rewards['classical_reward']} | {rewards['quantum_reward']} | "
            f"{rewards['hybrid_reward']} | {rewards['success_bonus']} | "
            f"{rewards['mismatch_penalty']} | {rewards['wait_penalty']} | "
            f"{rewards['low_util_penalty']} |"
        )

    lines.extend(["", "## 三、实验结果", "", "### 3.1 各配置平均奖励（seed 级）", ""])

    lines.append(
        "| 配置 | 平均奖励 | 标准差 | N (seeds) | 收敛步数（中位，真实） | 训练时间（中位, min）|"
    )
    lines.append("|:--|:--:|:--:|:--:|:--:|:--:|")
    for name, data in all_results.items():
        summary = data["summary"]
        convergence_steps = [s["train"]["convergence_step"] for s in data["seeds"].values()]
        train_times = [s["train"]["train_time_seconds"] / 60 for s in data["seeds"].values()]
        lines.append(
            f"| {name} | {summary['mean_reward']:.2f} | {summary['std_reward']:.2f} | "
            f"{summary['n_seeds']} | {np.median(convergence_steps):.0f} | "
            f"{np.median(train_times):.1f} |"
        )

    # 统计显著性
    lines.extend(
        [
            "",
            "### 3.2 统计显著性（L4 vs 其他，配对 Wilcoxon + Holm-Bonferroni）",
            "",
        ]
    )
    lines.append(
        "| 对比 | p (raw) | p (Holm-adj) | Holm 显著 | Cohen's d_z | 平均差值 | 非零对 | 判定 |"
    )
    lines.append("|:--|:--:|:--:|:--:|:--:|:--:|:--:|:--:|")
    for _pair_key, info in sig.items():
        sig_mark = "✅" if info.get("holm_significant", False) else "❌"
        lines.append(
            f"| {info['comparison']} | {info['p_value_raw']:.4f} | "
            f"{info.get('p_value_adjusted', float('nan')):.4f} | {sig_mark} | "
            f"{info['effect_size']:.4f} | {info['mean_diff']:+.2f} | "
            f"{info['n_nonzero_diffs']}/{info['n_pairs']} | {info.get('judgment', 'N/A')} |"
        )

    # 关键发现
    lines.extend(["", "## 四、关键发现", ""])

    l4_mean = all_results["L4_full"]["summary"]["mean_reward"]
    l1_mean = all_results["L1_basic"]["summary"]["mean_reward"]
    l2_mean = all_results["L2_execution"]["summary"]["mean_reward"]
    l3_mean = all_results["L3_wait_penalty"]["summary"]["mean_reward"]

    lines.append(
        f"1. **L4 完整奖励 vs L1 基础**: L4={l4_mean:.2f} vs L1={l1_mean:.2f}, "
        f"Δ={l4_mean - l1_mean:+.2f}"
    )
    lines.append(
        f"2. **执行收益贡献**: L2-L1={l2_mean - l1_mean:+.2f} (量子奖励 10 vs 经典 5 的激励效果)"
    )
    lines.append(f"3. **等待惩罚贡献**: L3-L2={l3_mean - l2_mean:+.2f} (等待惩罚 -0.1 的约束效果)")
    lines.append(
        f"4. **利用率奖励贡献**: L4-L3={l4_mean - l3_mean:+.2f} (低利用率惩罚 -1.0 的引导效果)"
    )

    # 与 #201 对比
    lines.extend(
        [
            "",
            "## 五、与 #201 推理消融的对比",
            "",
            "| 维度 | #201 推理消融 | #223 训练消融 |",
            "|:--|:--|:--|",
            f"| Seeds | 5 | {len(cfg['seeds'])} |",
            "| 方法 | 预训练模型推理 | 每配置从头训练 |",
            "| 能回答 | L4 训练的 PPO 在不同奖励下评估性能 | 不同奖励训练出的 PPO 性能差异 |",
            "| 等待时间变化 | 不变（策略固定） | 可能变化（策略不同） |",
            "| 收敛步数 | N/A | 真实 num_timesteps |",
            "",
            "**#223 的核心价值**：能验证『奖励设计 → 策略学习 → 性能』的完整因果链。",
            "",
            "---",
            "",
            f"*自动生成于 ablation_d3_training.py | 数据源: d3_training_ablation_{timestamp}.json*",
            "",
        ]
    )

    report_path = output_dir / "d3_training_ablation.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================================
# CLI 入口
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="D3 奖励函数训练消融实验（严格版，#223 / Issue #58 修复）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=10, help="随机种子数（默认 10，#223 要求）")
    parser.add_argument("--episodes", type=int, default=5, help="每 seed 评估 episode 数（默认 5）")
    parser.add_argument(
        "--timesteps",
        type=int,
        default=50000,
        help="训练步数（默认 50000，#223 要求；快速验证可用 1000）",
    )
    parser.add_argument(
        "--tasks-per-episode", type=int, default=200, help="每 episode 最大步数（默认 200）"
    )
    parser.add_argument(
        "--obs-dim", type=int, default=10, choices=[10, 14], help="观测维度（默认 10）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/ablation_d3_training",
        help="输出目录",
    )
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平（默认 0.05）")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="冒烟模式（覆盖 seeds=1, episodes=1, timesteps=1000）—— 不可作为正式消融结果",
    )
    args = parser.parse_args()

    if args.quick:
        args.seeds = 1
        args.episodes = 1
        args.timesteps = 1000
        print("[冒烟模式] seeds=1, episodes=1, timesteps=1000")
        print("[警告] 冒烟结果仅用于代码冒烟，不可作为正式消融结论")

    run_d3_training_ablation(
        seeds=args.seeds,
        episodes=args.episodes,
        timesteps=args.timesteps,
        tasks_per_episode=args.tasks_per_episode,
        obs_dim=args.obs_dim,
        output_dir=Path(args.output_dir),
        alpha=args.alpha,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()
