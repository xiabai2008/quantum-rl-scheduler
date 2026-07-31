"""留出盲测评估模块。

本模块提供 ``BlindTestEvaluator``，在与训练阶段互斥的留出测试种子集上
评估模型性能，收集逐 episode 累计奖励并汇总统计量，确保评估结果不受
训练数据泄漏影响。

模型兼容性：任何提供 ``predict(obs, deterministic=...)`` 方法的对象均可用作
待评估模型，包括 Stable-Baselines3 的 PPO/DQN 模型（返回 ``(action, state)``
元组）以及项目内的 ``PPOAgent``/``DQNAgent`` 包装器（直接返回 ``int``）。
"""

from __future__ import annotations

import math
from typing import Any, Protocol

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from src.scheduler.env import QuantumSchedulingEnv


class PredictableModel(Protocol):
    """兼容 Stable-Baselines3 的可预测模型协议。

    任何提供 ``predict(obs, deterministic=...)`` 方法的对象均满足此协议，
    包括 SB3 的 PPO/DQN 模型以及项目内的 PPOAgent/DQNAgent 包装器。
    """

    def predict(self, obs: NDArray[Any], deterministic: bool = True) -> Any:
        """根据观测返回动作预测。

        Args:
            obs: 环境观测向量。
            deterministic: 是否使用确定性策略。

        Returns:
            动作预测结果。SB3 模型返回 ``(action, state)`` 元组；
            自定义模型可直接返回 ``int``。
        """
        ...


def extract_action(predict_result: Any) -> int:
    """从 ``model.predict`` 返回值中提取动作索引。

    兼容两种返回格式：
    - Stable-Baselines3：返回 ``(action, state)`` 元组，取首个元素。
    - 自定义模型：直接返回 ``int`` 或标量数组。

    Args:
        predict_result: ``model.predict`` 的返回值。

    Returns:
        动作索引（int）。
    """
    action = predict_result[0] if isinstance(predict_result, tuple) else predict_result
    return int(np.asarray(action).item())


class BlindTestEvaluator:
    """在留出测试集上评估模型性能。

    使用与训练阶段互斥的测试种子对模型进行盲测，收集逐 episode 累计奖励
    并汇总统计量（均值、标准差、最值），确保评估结果不受训练数据泄漏影响。

    Args:
        deterministic: 推理时是否使用确定性策略，默认 True。
    """

    def __init__(self, deterministic: bool = True) -> None:
        """初始化盲测评估器。

        Args:
            deterministic: 推理时是否使用确定性策略，默认 True。
        """
        self.deterministic = deterministic

    def evaluate(
        self,
        model: PredictableModel,
        env: QuantumSchedulingEnv,
        test_seeds: list[int],
        episodes_per_seed: int = 5,
        baseline_rewards: list[float] | None = None,
    ) -> dict[str, Any]:
        """在留出测试种子集上评估模型。

        对每个测试种子运行指定数量的 episode，收集累计奖励并计算统计量。
        每个 episode 通过 ``env.reset(seed=...)`` 初始化，确保种子可复现。
        同一环境实例在各 episode 间通过 ``reset`` 重置，不携带跨 episode 状态。

        当提供 ``baseline_rewards`` 时，额外返回与基线策略的统计显著性对比结果
        （调用 ``stats_significance.compare_strategies`` 计算 p 值/效应量/CI）。

        Args:
            model: 待评估模型，需提供 ``predict(obs, deterministic=...)`` 方法。
            env: 调度环境实例（每个 episode 会调用 ``reset`` 重置）。
            test_seeds: 留出测试种子列表（须与训练种子互斥）。
            episodes_per_seed: 每个种子运行的 episode 数，默认 5。
            baseline_rewards: 基线策略的奖励列表，传入时会计算与基线的
                统计显著性对比；默认 ``None`` 不执行对比。

        Returns:
            包含以下键的评估结果字典：
            - ``mean_reward``: 平均奖励
            - ``std_reward``: 奖励标准差（ddof=1）
            - ``min_reward``: 最小奖励
            - ``max_reward``: 最大奖励
            - ``sem``: 标准误
            - ``ci95_lower``: 均值 95% CI 下界
            - ``ci95_upper``: 均值 95% CI 上界
            - ``all_rewards``: 逐 episode 奖励列表
            - ``num_episodes``: 总 episode 数
            - ``baseline_comparison``: 可选，与基线策略的统计对比结果（仅当
              ``baseline_rewards`` 不为 None 时存在）

        Raises:
            ValueError: ``test_seeds`` 为空或 ``episodes_per_seed`` 非正。
        """
        if not test_seeds:
            raise ValueError("test_seeds 列表不能为空")
        if episodes_per_seed <= 0:
            raise ValueError("episodes_per_seed 必须为正整数")

        # Issue #386: 检查相邻 test_seeds 的 episode 种子是否会重叠
        # 原实现使用 seed + episode，若 test_seeds=[42, 43] 且 episodes_per_seed=5，
        # seed=42 使用 42-46，seed=43 使用 43-47，存在 4 个重叠种子
        # 修复：使用 seed * episodes_per_seed + episode 确保全局唯一
        episode_seeds_all: list[int] = []
        for seed in test_seeds:
            for episode in range(episodes_per_seed):
                episode_seeds_all.append(seed * episodes_per_seed + episode)
        if len(set(episode_seeds_all)) != len(episode_seeds_all):
            raise ValueError(
                f"episode 种子存在重叠: {episode_seeds_all}，"
                "请检查 test_seeds 间距是否 >= episodes_per_seed"
            )

        all_rewards: list[float] = []
        max_steps = env.max_steps

        for seed in test_seeds:
            for episode in range(episodes_per_seed):
                # Issue #386: 使用乘法偏移确保相邻 seed 的 episode 种子不重叠
                episode_seed = seed * episodes_per_seed + episode
                episode_reward = self._run_episode(model, env, episode_seed, max_steps)
                all_rewards.append(episode_reward)

        result = summarize_rewards(all_rewards)

        if baseline_rewards is not None:
            from src.utils.stats_significance import compare_strategies

            result["baseline_comparison"] = compare_strategies(
                {"Target": all_rewards, "Baseline": baseline_rewards}
            )

        logger.info(
            "盲测评估完成：{} episodes，mean_reward={:.2f} ± {:.2f}",
            result["num_episodes"],
            result["mean_reward"],
            result["std_reward"],
        )
        return result

    def _run_episode(
        self,
        model: PredictableModel,
        env: QuantumSchedulingEnv,
        seed: int,
        max_steps: int,
    ) -> float:
        """运行单个 episode 并返回累计奖励。

        Args:
            model: 待评估模型。
            env: 调度环境实例。
            seed: 本 episode 的随机种子。
            max_steps: 最大步数。

        Returns:
            本 episode 的累计奖励。
        """
        obs, _ = env.reset(seed=seed)
        episode_reward = 0.0
        for _ in range(max_steps):
            action = extract_action(model.predict(obs, deterministic=self.deterministic))
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            if terminated or truncated:
                break
        return episode_reward


def summarize_rewards(rewards: list[float]) -> dict[str, Any]:
    """汇总奖励列表为统计字典。

    Args:
        rewards: 逐 episode 奖励列表。

    Returns:
        包含 ``mean_reward``/``std_reward``/``min_reward``/``max_reward``/
        ``sem``/``ci95_lower``/``ci95_upper``/``all_rewards``/``num_episodes``
        的字典。空列表时各统计量均为 0.0（sem/CI 为 nan）。
    """
    if not rewards:
        return {
            "mean_reward": 0.0,
            "std_reward": 0.0,
            "min_reward": 0.0,
            "max_reward": 0.0,
            "sem": float("nan"),
            "ci95_lower": float("nan"),
            "ci95_upper": float("nan"),
            "all_rewards": [],
            "num_episodes": 0,
        }
    arr = np.array(rewards, dtype=np.float64)
    n = len(arr)
    std_reward = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    mean_reward = float(np.mean(arr))

    if n >= 2:
        sem = std_reward / math.sqrt(n)
        # P2-2: 小样本（n<30）正态近似 ±1.96·sem 会低估 CI 宽度，
        # 改用 t 分布临界值（df=n-1）更严谨；大样本时 t≈z 退化一致。
        if n < 30:
            from scipy import stats as _sp

            critical = float(_sp.t.ppf(0.975, df=n - 1))
        else:
            critical = 1.96
        margin = critical * sem
        ci95_lower = mean_reward - margin
        ci95_upper = mean_reward + margin
    else:
        sem = float("nan")
        ci95_lower = float("nan")
        ci95_upper = float("nan")

    return {
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "min_reward": float(np.min(arr)),
        "max_reward": float(np.max(arr)),
        "sem": sem,
        "ci95_lower": ci95_lower,
        "ci95_upper": ci95_upper,
        "all_rewards": rewards,
        "num_episodes": len(rewards),
    }
