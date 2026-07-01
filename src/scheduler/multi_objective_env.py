"""
多目标强化学习 (MORL) 奖励包装器
Multi-Objective RL Reward Wrapper for Quantum Scheduling

将 QuantumSchedulingEnv 的标量奖励分解为 3 个独立目标：
    1. 吞吐量目标 (throughput): 任务完成速率 = 单位时间完成的任务数
    2. 资源平衡目标 (balance):    量子/经典资源利用率的平衡度
    3. 服务质量目标 (quality):    用户等待时间

通过加权标量化 (Weighted Scalarization) 将多目标合并为标量奖励：
    reward = w[0] * throughput + w[1] * balance + w[2] * quality

每个目标的独立值通过 info["objectives"] 字典返回。

使用方式:
    from src.scheduler.env import QuantumSchedulingEnv
    from src.scheduler.multi_objective_env import MultiObjectiveRewardWrapper

    env = QuantumSchedulingEnv(max_qubits=20)
    mo_env = MultiObjectiveRewardWrapper(env, weights=[1.0, 0.5, 0.5])
    obs, info = mo_env.reset()
    obs, reward, terminated, truncated, info = mo_env.step(action)
    # info["objectives"] = {"throughput": ..., "balance": ..., "quality": ...}
"""

from typing import Any

import gymnasium as gym
import numpy as np

from src.scheduler.env import (
    MAX_WAIT_STEPS,
    QuantumSchedulingEnv,
)

# ---------------------------------------------------------------------------
# 多目标权重预设
# ---------------------------------------------------------------------------

# 预定义权重组合（对应不同调度偏好）
DEFAULT_WEIGHTS = {
    "throughput_heavy": [1.0, 0.5, 0.5],  # 偏吞吐量
    "balance_heavy": [0.5, 1.0, 0.5],  # 偏资源平衡
    "quality_heavy": [0.5, 0.5, 1.0],  # 偏服务质量
    "balanced": [1.0, 1.0, 1.0],  # 均衡
    "throughput_only": [1.0, 0.0, 0.0],  # 仅吞吐量
    "balance_only": [0.0, 1.0, 0.0],  # 仅平衡
    "quality_only": [0.0, 0.0, 1.0],  # 仅服务质量
}


class MultiObjectiveRewardWrapper(gym.Wrapper):
    """
    多目标奖励包装器。

    将原始环境的标量奖励分解为 3 个独立目标：
        - throughput: 任务完成速率（0~1 归一化）
        - balance:    量子/经典资源利用率平衡度（-1~0）
        - quality:    用户等待时间服务质量（-1~0）

    通过加权标量化合并为标量奖励，同时通过 info dict 返回每个目标的独立值。

    Attributes:
        weights: 3 元素列表，对应 [throughput, balance, quality] 的权重
        env: 被包装的 QuantumSchedulingEnv
    """

    def __init__(
        self,
        env: QuantumSchedulingEnv,
        weights: list[float] | None = None,
        weight_preset: str | None = None,
    ):
        """
        初始化多目标奖励包装器。

        Args:
            env: QuantumSchedulingEnv 实例
            weights: 3 元素列表 [w_throughput, w_balance, w_quality]
            weight_preset: 预定义权重名称，如 "throughput_heavy"
                           weights 和 weight_preset 不能同时指定
        """
        super().__init__(env)
        if weights is not None and weight_preset is not None:
            raise ValueError("不能同时指定 weights 和 weight_preset")
        if weight_preset is not None:
            if weight_preset not in DEFAULT_WEIGHTS:
                raise ValueError(
                    f"未知权重预设 '{weight_preset}'，可选: {list(DEFAULT_WEIGHTS.keys())}"
                )
            self._weights = list(DEFAULT_WEIGHTS[weight_preset])
        elif weights is not None:
            if len(weights) != 3:
                raise ValueError(f"weights 必须包含 3 个元素，实际: {len(weights)}")
            self._weights = list(weights)
        else:
            self._weights = [1.0, 0.5, 0.5]  # 默认偏吞吐量

        # 多目标累积统计（每 episode 重置）
        self._mo_stats = self._reset_mo_stats()

    # ------------------------------------------------------------------
    # 权重属性
    # ------------------------------------------------------------------

    @property
    def weights(self) -> list[float]:
        """返回当前权重 [w_throughput, w_balance, w_quality]."""
        return list(self._weights)

    @weights.setter
    def weights(self, value: list[float]) -> None:
        """
        运行时动态切换权重，无需重新训练。

        Args:
            value: 3 元素权重列表
        """
        if len(value) != 3:
            raise ValueError(f"weights 必须包含 3 个元素，实际: {len(value)}")
        self._weights = list(value)

    @property
    def weight_names(self) -> list[str]:
        """返回权重名称列表。"""
        return ["throughput", "balance", "quality"]

    # ------------------------------------------------------------------
    # reset()
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        重置环境并初始化多目标统计。

        Args:
            seed: 随机种子
            options: 额外选项

        Returns:
            observation: 状态向量
            info: 包含 objectives 初始化值的字典
        """
        obs, info = super().reset(seed=seed, options=options)
        self._mo_stats = self._reset_mo_stats()
        # 初始化时各目标为 0
        info["objectives"] = {
            "throughput": 0.0,
            "balance": 0.0,
            "quality": 0.0,
        }
        info["mo_weights"] = list(self._weights)
        return obs, info

    # ------------------------------------------------------------------
    # step()
    # ------------------------------------------------------------------

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        执行一步调度决策，计算多目标奖励。

        核心逻辑：
            1. 调用原始 env.step() 获取原始标量奖励
            2. 从当前环境状态中计算 3 个独立目标值
            3. 加权标量化：reward = w[0]*throughput + w[1]*balance + w[2]*quality
            4. 将独立目标值存入 info["objectives"]

        Args:
            action: 动作索引

        Returns:
            observation: 下一步状态
            reward: 加权标量化奖励
            terminated: 是否终止
            truncated: 是否截断
            info: 包含 objectives 和原始标量奖励的字典
        """
        # 先调用原始 step，获取原始奖励和基础 info
        # 注：原始 step 已计算了 env 内部的标量奖励，
        # 这里我们重新计算多目标，用多目标加权 reward 替代原始 reward
        orig_obs, orig_reward, terminated, truncated, info = self.env.step(action)

        # 计算 3 个独立目标值
        throughput = self._compute_throughput(info)
        balance = self._compute_balance()
        quality = self._compute_quality()

        # 累计统计
        self._mo_stats["total_throughput"] += throughput
        self._mo_stats["total_balance"] += balance
        self._mo_stats["total_quality"] += quality
        self._mo_stats["steps"] += 1

        # 加权标量化
        reward = (
            self._weights[0] * throughput + self._weights[1] * balance + self._weights[2] * quality
        )

        # 存入 info dict
        info["objectives"] = {
            "throughput": float(throughput),
            "balance": float(balance),
            "quality": float(quality),
        }
        info["mo_weights"] = list(self._weights)
        info["original_reward"] = float(orig_reward)
        info["mo_reward"] = float(reward)
        info["mo_cumulative"] = {
            "throughput": float(self._mo_stats["total_throughput"]),
            "balance": float(self._mo_stats["total_balance"]),
            "quality": float(self._mo_stats["total_quality"]),
        }

        return orig_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # 私有方法：目标计算
    # ------------------------------------------------------------------

    def _compute_throughput(self, info: dict[str, Any]) -> float:
        """
        计算吞吐量目标：本步完成的任务数。

        当本步成功调度了一个任务（兼容分配且资源可用），
        吞吐量贡献为 +1。归一化到 [0, 1] 区间。

        Args:
            info: 原始 env.step() 返回的 info 字典

        Returns:
            float: 吞吐量目标值 [0, 1]
        """
        # 判断本步是否成功调度了任务
        # 通过检查 info 中的统计变化来判断
        scheduled = 0.0
        env = self.env.unwrapped

        # 检查本步是否有调度动作且兼容
        if env._current_task is not None or env._total_scheduled > 0:  # noqa: SIM102
            # 通过比较 mismatch_count 和 success 的变化来判断
            # 简化：如果本步 reward 包含执行奖励（非惩罚），则视为成功调度
            if info.get("total_scheduled", 0) > self._mo_stats.get("_last_total_scheduled", 0):
                scheduled = 1.0

        self._mo_stats["_last_total_scheduled"] = info.get("total_scheduled", 0)
        return scheduled

    def _compute_balance(self) -> float:
        """
        计算资源平衡目标：量子/经典资源利用率的平衡度。

        公式: balance = -|quantum_available_ratio - classical_load|
        完全平衡时为 0，越不平衡负值越大。

        Returns:
            float: 平衡度目标值 [-1, 0]
        """
        env = self.env.unwrapped
        quantum_util = env._quantum.available_ratio  # 量子可用比率（越高越空闲）
        classical_util = env._classical.load  # 经典负载（越高越忙）

        # 平衡度 = -|量子空闲率 - 经典负载率|
        # 当两者相等时最平衡，值为 0
        balance = -abs(quantum_util - classical_util)
        return float(np.clip(balance, -1.0, 0.0))

    def _compute_quality(self) -> float:
        """
        计算服务质量目标：用户等待时间。

        公式: quality = -avg_wait / MAX_WAIT_STEPS
        等待时间越短，服务质量越好（值越接近 0）。

        Returns:
            float: 服务质量目标值 [-1, 0]
        """
        env = self.env.unwrapped
        if not env._task_queue:
            return 0.0

        avg_wait = sum(t.wait_steps for t in env._task_queue) / len(env._task_queue)
        quality = -avg_wait / MAX_WAIT_STEPS
        return float(np.clip(quality, -1.0, 0.0))

    # ------------------------------------------------------------------
    # 私有方法：统计重置
    # ------------------------------------------------------------------

    def _reset_mo_stats(self) -> dict[str, Any]:
        """
        重置多目标统计。

        Returns:
            dict: 初始化的统计字典
        """
        return {
            "total_throughput": 0.0,
            "total_balance": 0.0,
            "total_quality": 0.0,
            "steps": 0,
            "_last_total_scheduled": 0,
        }

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def get_episode_objectives(self) -> dict[str, float]:
        """
        获取当前 episode 累积的多目标值。

        Returns:
            dict: {"throughput": ..., "balance": ..., "quality": ...}
        """
        return {
            "throughput": self._mo_stats["total_throughput"],
            "balance": self._mo_stats["total_balance"],
            "quality": self._mo_stats["total_quality"],
        }

    def set_weight_preset(self, preset: str) -> None:
        """
        通过预设名称切换权重。

        Args:
            preset: 预设名称，如 "throughput_heavy", "balance_heavy", "quality_heavy"
        """
        if preset not in DEFAULT_WEIGHTS:
            raise ValueError(f"未知权重预设 '{preset}'，可选: {list(DEFAULT_WEIGHTS.keys())}")
        self._weights = list(DEFAULT_WEIGHTS[preset])


# ---------------------------------------------------------------------------
# 工厂函数
# ---------------------------------------------------------------------------


def make_mo_env(
    max_qubits: int = 20,
    max_steps: int = 200,
    weights: list[float] | None = None,
    weight_preset: str | None = None,
    seed: int | None = None,
    machine_configs: Any | None = None,
) -> MultiObjectiveRewardWrapper:
    """
    创建多目标调度环境的便捷工厂函数。

    Args:
        max_qubits: 最大量子比特数
        max_steps: 最大步数
        weights: 3 元素权重列表
        weight_preset: 预定义权重名称
        seed: 随机种子
        machine_configs: 多机器配置（可选）

    Returns:
        MultiObjectiveRewardWrapper: 包装好的多目标环境
    """
    env = QuantumSchedulingEnv(
        max_qubits=max_qubits,
        max_steps=max_steps,
        seed=seed,
        machine_configs=machine_configs,
    )
    return MultiObjectiveRewardWrapper(env, weights=weights, weight_preset=weight_preset)
