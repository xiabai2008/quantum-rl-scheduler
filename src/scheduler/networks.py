"""
自定义策略网络模块
Custom Policy Networks Module

包含 Dueling DQN 策略网络实现，用于量子-经典混合任务调度决策。
Dueling 架构将 Q 值拆分为状态价值函数和优势函数，提升策略学习效率。

模块内容：
    - DuelingQNetwork: Dueling DQN 策略网络（兼容 Stable-Baselines3 2.0+）
"""

import torch as th
from gymnasium import spaces
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp,
)
from stable_baselines3.dqn.policies import QNetwork
from torch import nn

# ---------------------------------------------------------------------------
# 自定义策略网络：Dueling DQN（兼容 SB3 2.0+）
# ---------------------------------------------------------------------------


class DuelingQNetwork(QNetwork):
    """
    Dueling DQN 策略网络（兼容 Stable-Baselines3 2.0+）

    相比标准 DQN，Dueling 架构将 Q(s,a) 拆分为：
        - 状态价值函数 V(s)：衡量当前状态的总体价值
        - 优势函数 A(s,a)：衡量在当前状态下选择某动作的相对优劣
    最终 Q 值：Q(s,a) = V(s) + A(s,a) - mean(A(s,a))

    网络结构：
        - 输入层：observation_shape (默认 8)
        - 共享特征层：features_dim -> 128 -> 64
        - 价值分支 V(s)：64 -> 1
        - 优势分支 A(s,a)：64 -> n_actions (默认 3)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Discrete,
        features_extractor: BaseFeaturesExtractor,
        features_dim: int,
        net_arch: list[int] | None = None,
        activation_fn: type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ) -> None:
        """
        初始化 Dueling DQN 网络。

        Args:
            observation_space: 观测空间
            action_space: 离散动作空间
            features_extractor: 特征提取器
            features_dim: 特征维度
            net_arch: 隐藏层架构，默认 [128, 64]
            activation_fn: 激活函数类型，默认 ReLU
            normalize_images: 是否归一化图像，默认 True
        """
        # 默认隐藏层 [128, 64]
        if net_arch is None:
            net_arch = [128, 64]

        # 调用 QNetwork.__init__，它会自动创建 self.q_net
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=net_arch,
            activation_fn=activation_fn,
            normalize_images=normalize_images,
        )

        # 用 Dueling 架构替换 QNetwork 创建的标准 q_net
        action_dim = int(self.action_space.n)
        shared_output_dim = self.net_arch[-1] if self.net_arch else features_dim

        # 共享特征层（提取高层表示）
        self.q_net = nn.Sequential(
            *create_mlp(features_dim, shared_output_dim, self.net_arch[:-1], self.activation_fn)
        )

        # 价值分支 V(s)：估计状态价值
        self.value_stream = nn.Sequential(
            nn.Linear(shared_output_dim, shared_output_dim // 2),
            self.activation_fn(),
            nn.Linear(shared_output_dim // 2, 1),
        )

        # 优势分支 A(s,a)：估计每个动作的相对优势
        self.advantage_stream = nn.Sequential(
            nn.Linear(shared_output_dim, shared_output_dim // 2),
            self.activation_fn(),
            nn.Linear(shared_output_dim // 2, action_dim),
        )

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        前向传播：计算 Dueling Q 值

        Args:
            obs: 观测状态张量，形状为 (batch_size, obs_dim)

        Returns:
            Q 值张量，形状为 (batch_size, action_dim)
        """
        # 提取特征（SB3 2.0+ 需要传入 features_extractor）
        features = self.extract_features(obs, self.features_extractor)
        # 通过共享层
        shared = self.q_net(features)
        # 计算状态价值和动作优势
        value = self.value_stream(shared)  # (batch, 1)
        advantage = self.advantage_stream(shared)  # (batch, action_dim)
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q_values


__all__ = ["DuelingQNetwork"]
