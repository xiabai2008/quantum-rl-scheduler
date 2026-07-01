"""
多智能体 PPO (MAPPO) 调度模块
Multi-Agent PPO for Quantum-Classical Hybrid Task Scheduling

替代单智能体 PPO 架构：为每台量子机器配置一个独立的 Actor（策略网络），
所有 Actor 共享一个集中式 Critic（价值网络），实现多机器协调调度。

核心思想（CTDE：Centralized Training, Decentralized Execution）：
    - 执行时：每个 Agent 仅根据本机局部观测独立决策（去中心化）
    - 训练时：集中式 Critic 利用所有 Agent 的局部观测 + 全局状态估计价值（中心化）

设计要点：
    - 不修改 env.py 的公共接口，通过 MultiAgentEnvWrapper 包装现有环境
    - 每个 Agent 动作空间保持 Discrete(3)：
        0 = 将当前量子任务转交经典处理
        1 = 在本机执行量子任务
        2 = 在本机执行混合（量子-经典协同）
    - 经典任务仍由中心调度器处理（聚合时若所有 Agent 投票 0 则走经典）
    - 纯 PyTorch 实现 Actor-Critic 与训练循环，不依赖第三方 MARL 框架

典型用法::

    from src.scheduler.env import QuantumSchedulingEnv, DEFAULT_MACHINE_CONFIGS
    from src.scheduler.marl import MultiAgentPPO

    env = QuantumSchedulingEnv(machine_configs=DEFAULT_MACHINE_CONFIGS)
    agent = MultiAgentPPO(env, learning_rate=3e-4, n_steps=1024)
    agent.train(total_timesteps=50000)
    agent.save("./models/mappo")
    result = agent.evaluate(num_episodes=10)
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

# 复用现有环境常量，确保观测维度与原环境一致
from src.scheduler.env import (
    MAX_QUEUE_SIZE,
    OBS_DIM,
    QuantumSchedulingEnv,
)

# ---------------------------------------------------------------------------
# 多智能体环境包装器
# ---------------------------------------------------------------------------


class MultiAgentEnvWrapper:
    """
    多智能体环境包装器：在不修改 QuantumSchedulingEnv 公共接口的前提下，
    将单智能体环境适配为多智能体环境。

    职责：
        1. 为每台量子机器构建局部观测（OBS_DIM 维全局 + 3 维本机特征）
        2. 聚合各 Agent 的动作投票为单个 env action（0/1/2）
        3. 通过临时调整机器在线状态，引导 env._select_best_machine 路由到被选中的机器

    动作聚合规则：
        - 仅在线机器的投票有效
        - 若有 Agent 投票 1（量子执行）：从投票者中按 score 选最优机器 → env action=1
        - 否则若有 Agent 投票 2（混合执行）：选最优机器 → env action=2
        - 否则（全部投票 0 或无在线机器）：env action=0（经典执行）

    路由实现：在调用 env.step 前临时将非选中机器置为不可用，
    迫使 env._select_best_machine 返回被选中的机器；step 结束后恢复原状。
    这保证不破坏 env.py 的任何逻辑，且聚合结果精确可控。
    """

    # 每台机器的局部观测增量维度：fidelity + available_ratio + queue_length
    PER_MACHINE_FEATURE_DIM = 3

    def __init__(self, env: QuantumSchedulingEnv):
        """
        初始化多智能体环境包装器。

        Args:
            env: 已配置好机器的 QuantumSchedulingEnv 实例
        """
        self.env = env
        self.num_agents = env.num_machines
        self.machine_names: list[str] = list(env.machine_names)
        # 局部观测维度 = 全局 10 + 本机 3
        self.local_obs_dim = OBS_DIM + self.PER_MACHINE_FEATURE_DIM

    # ------------------------------------------------------------------
    # 局部观测构建
    # ------------------------------------------------------------------

    def _build_local_obs(self, global_obs: np.ndarray, machine_idx: int) -> np.ndarray:
        """
        构建单个 Agent 的局部观测向量。

        局部观测 = [全局 OBS_DIM 维] + [本机 fidelity, available_ratio, queue_length_normalized]

        Args:
            global_obs: env._get_observation() 返回的全局观测
            machine_idx: 机器索引

        Returns:
            形状 (local_obs_dim,) 的 float32 向量，值域 [0, 1]
        """
        m = self.env._machines[machine_idx]
        per_machine = np.array(
            [
                float(np.clip(m.fidelity, 0.0, 1.0)),
                float(np.clip(m.available_ratio, 0.0, 1.0)),
                float(np.clip(m.quantum_queue / MAX_QUEUE_SIZE, 0.0, 1.0)),
            ],
            dtype=np.float32,
        )
        return np.concatenate([global_obs.astype(np.float32), per_machine])

    def get_local_observations(self) -> dict[str, np.ndarray]:
        """
        获取所有 Agent 的局部观测。

        Returns:
            机器名 -> 局部观测向量的字典
        """
        global_obs = self.env._get_observation()
        return {
            self.machine_names[i]: self._build_local_obs(global_obs, i)
            for i in range(self.num_agents)
        }

    def get_global_state(self) -> np.ndarray:
        """
        获取集中式 Critic 的输入：所有 Agent 局部观测的拼接。

        全局状态 = concat(local_obs_1, local_obs_2, ..., local_obs_N)
        维度 = local_obs_dim * num_agents

        Returns:
            形状 (local_obs_dim * num_agents,) 的 float32 向量
        """
        local_obs = self.get_local_observations()
        return np.concatenate([local_obs[name] for name in self.machine_names]).astype(np.float32)

    # ------------------------------------------------------------------
    # 动作聚合与路由
    # ------------------------------------------------------------------

    def _machine_score(self, machine_idx: int) -> float:
        """
        计算机器评分：保真度 * 可用比率 / (1 + 队列长度)。

        与 env._select_best_machine 的评分保持一致，用于在多个投票机器中择优。

        Args:
            machine_idx: 机器索引

        Returns:
            评分值（越高越优）
        """
        m = self.env._machines[machine_idx]
        return m.fidelity * m.available_ratio / (1.0 + m.quantum_queue)

    def aggregate_actions(self, actions: dict[str, int]) -> tuple[int, int | None]:
        """
        聚合各 Agent 的动作投票为单个 env action 和选中的机器索引。

        Args:
            actions: 机器名 -> 动作（0/1/2）的字典

        Returns:
            (env_action, chosen_machine_idx):
                env_action: 0/1/2，传给 env.step 的动作
                chosen_machine_idx: 被选中执行量子/混合任务的机器索引，经典执行时为 None
        """
        # 仅在线机器的投票有效
        quantum_voters: list[int] = []
        hybrid_voters: list[int] = []
        for idx, name in enumerate(self.machine_names):
            m = self.env._machines[idx]
            if not m.available:
                continue
            a = int(actions.get(name, 0))
            if a == 1:
                quantum_voters.append(idx)
            elif a == 2:
                hybrid_voters.append(idx)

        if quantum_voters:
            chosen = max(quantum_voters, key=self._machine_score)
            return 1, chosen
        if hybrid_voters:
            chosen = max(hybrid_voters, key=self._machine_score)
            return 2, chosen
        # 无在线机器愿意执行量子/混合，或全部投票经典
        return 0, None

    def _machine_can_handle(self, machine_idx: int, task: Any) -> bool:
        """
        检查指定机器能否承接给定任务（与 env._select_best_machine 的过滤逻辑一致）。

        过滤条件：
            1. 机器在线（available=True）
            2. 可用比特数 >= 任务需求（usable = total_qubits * available_ratio）
            3. 机器门集合兼容任务所需门

        Args:
            machine_idx: 机器索引
            task: 待执行任务（需有 qubit_count 字段）

        Returns:
            bool: 机器能否执行该任务
        """
        m = self.env._machines[machine_idx]
        if not m.available:
            return False
        usable_qubits = int(m.total_qubits * m.available_ratio)
        if usable_qubits < getattr(task, "qubit_count", 0):
            return False
        return self.env._machine_supports_task(m, task)

    def step(
        self, actions: dict[str, int]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        执行一步多智能体调度。

        聚合各 Agent 动作后调用 env.step，并引导路由到被选中的机器。
        仅当选中机器能真正承接当前任务时才强制路由（避免把任务
        路由到不兼容机器导致量子资源不可用惩罚）；若选中机器不兼容，
        则不干预路由，由 env._select_best_machine 自行挑选最佳兼容机器。

        Args:
            actions: 机器名 -> 动作（0/1/2）的字典

        Returns:
            local_obs: 机器名 -> 局部观测
            reward: 共享奖励（环境返回的标量）
            terminated: 是否自然终止
            truncated: 是否截断
            info: 环境信息字典（附加 chosen_machine 字段）
        """
        env_action, chosen_idx = self.aggregate_actions(actions)

        # 仅当多机器、选定了机器、且该机器能承接当前任务时，才强制路由
        # 这样既能尊重 Agent 的机器选择，又能在 Agent 误选不兼容机器时
        # 优雅降级到 env 的启发式选择，避免无谓的量子不可用惩罚
        original_available: dict[int, bool] | None = None
        task = self.env._current_task
        if (
            chosen_idx is not None
            and self.num_agents > 1
            and task is not None
            and self._machine_can_handle(chosen_idx, task)
        ):
            original_available = {}
            for idx, m in enumerate(self.env._machines):
                original_available[idx] = m.available
                # 仅保留被选中机器在线（若它原本就在线）
                m.available = idx == chosen_idx

        try:
            _next_obs, reward, terminated, truncated, info = self.env.step(env_action)
        finally:
            # 无论 step 是否抛异常都恢复原在线状态
            if original_available is not None:
                for idx, m in enumerate(self.env._machines):
                    m.available = original_available[idx]

        info = dict(info)
        info["chosen_machine"] = self.machine_names[chosen_idx] if chosen_idx is not None else None
        info["env_action"] = env_action

        local_obs = self.get_local_observations()
        return local_obs, float(reward), bool(terminated), bool(truncated), info

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        重置环境，返回各 Agent 的局部观测。

        Args:
            seed: 随机种子
            options: 额外选项（透传给 env.reset）

        Returns:
            local_obs: 机器名 -> 局部观测
            info: 环境信息字典
        """
        _, info = self.env.reset(seed=seed, options=options)
        return self.get_local_observations(), info


# ---------------------------------------------------------------------------
# 神经网络定义
# ---------------------------------------------------------------------------


def _build_mlp(
    input_dim: int, output_dim: int, hidden_sizes: tuple[int, ...], activation: type = nn.Tanh
) -> nn.Sequential:
    """
    构建简单的多层感知机（MLP）。

    Args:
        input_dim: 输入维度
        output_dim: 输出维度（None 表示不添加输出层，仅返回特征提取器）
        hidden_sizes: 隐藏层尺寸元组
        activation: 激活函数类

    Returns:
        nn.Sequential 模块
    """
    layers: list[nn.Module] = []
    last = input_dim
    for h in hidden_sizes:
        layers.append(nn.Linear(last, h))
        layers.append(activation())
        last = h
    if output_dim > 0:
        layers.append(nn.Linear(last, output_dim))
    return nn.Sequential(*layers)


class ActorNet(nn.Module):
    """
    单个 Agent 的策略网络（Actor）。

    输入：本机局部观测（local_obs_dim 维）
    输出：3 个离散动作的 logits（Categorical 分布）

    使用 Tanh 激活函数（PPO 常用配置，训练稳定）。
    """

    def __init__(
        self, obs_dim: int, action_dim: int = 3, hidden_sizes: tuple[int, ...] = (128, 64)
    ):
        """
        初始化 Actor 网络。

        Args:
            obs_dim: 局部观测维度
            action_dim: 动作维度，默认 3
            hidden_sizes: 隐藏层尺寸
        """
        super().__init__()
        self.feature = _build_mlp(obs_dim, 0, hidden_sizes)
        self.action_head = nn.Linear(hidden_sizes[-1], action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出动作 logits。

        Args:
            obs: 局部观测张量，形状 (batch, obs_dim)

        Returns:
            logits 张量，形状 (batch, action_dim)
        """
        features = self.feature(obs)
        return self.action_head(features)

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样动作并返回对数概率与熵。

        Args:
            obs: 局部观测张量，形状 (batch, obs_dim)
            deterministic: True 时贪心选择最大 logit 的动作

        Returns:
            (action, log_prob, entropy) 张量
        """
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        action = torch.argmax(logits, dim=-1) if deterministic else dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        给定观测和动作，重新计算对数概率与熵（用于 PPO 更新）。

        Args:
            obs: 局部观测张量
            actions: 动作张量

        Returns:
            (log_prob, entropy) 张量
        """
        logits = self.forward(obs)
        dist = torch.distributions.Categorical(logits=logits)
        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_prob, entropy


class CentralizedCritic(nn.Module):
    """
    集中式价值网络（Critic）。

    输入：全局状态（所有 Agent 局部观测的拼接）
    输出：状态价值标量 V(s)

    所有 Agent 共享同一个 Critic，利用全局信息估计价值，
    指导各 Actor 的优势函数计算。
    """

    def __init__(self, global_state_dim: int, hidden_sizes: tuple[int, ...] = (256, 128)):
        """
        初始化 Critic 网络。

        Args:
            global_state_dim: 全局状态维度（= local_obs_dim * num_agents）
            hidden_sizes: 隐藏层尺寸
        """
        super().__init__()
        self.net = _build_mlp(global_state_dim, 1, hidden_sizes)

    def forward(self, global_state: torch.Tensor) -> torch.Tensor:
        """
        前向传播，输出状态价值。

        Args:
            global_state: 全局状态张量，形状 (batch, global_state_dim)

        Returns:
            价值张量，形状 (batch,)
        """
        return self.net(global_state).squeeze(-1)


# ---------------------------------------------------------------------------
# 经验回放缓冲区
# ---------------------------------------------------------------------------


class RolloutBuffer:
    """
    MAPPO 经验回放缓冲区。

    存储一个 rollout 周期（n_steps 步）的轨迹数据：
        - 每个 Agent 独立存储：local_obs, action, log_prob
        - 所有 Agent 共享：reward, global_state, done, value

    所有数据在写入时即 detach/转 numpy，避免持有计算图导致内存泄漏。
    """

    def __init__(self, n_steps: int, num_agents: int, local_obs_dim: int, global_state_dim: int):
        """
        初始化缓冲区。

        Args:
            n_steps: 缓冲区容量（步数）
            num_agents: Agent 数量
            local_obs_dim: 局部观测维度
            global_state_dim: 全局状态维度
        """
        self.n_steps = n_steps
        self.num_agents = num_agents
        self.local_obs_dim = local_obs_dim
        self.global_state_dim = global_state_dim

        # 每 Agent 独立数据
        self.local_obs = [
            np.zeros((n_steps, local_obs_dim), dtype=np.float32) for _ in range(num_agents)
        ]
        self.actions = [np.zeros(n_steps, dtype=np.int64) for _ in range(num_agents)]
        self.log_probs = [np.zeros(n_steps, dtype=np.float32) for _ in range(num_agents)]
        # 共享数据
        self.rewards = np.zeros(n_steps, dtype=np.float32)
        self.global_states = np.zeros((n_steps, global_state_dim), dtype=np.float32)
        self.dones = np.zeros(n_steps, dtype=np.float32)
        self.values = np.zeros(n_steps, dtype=np.float32)

        self.pos = 0

    def add(
        self,
        local_obs: list[np.ndarray],
        actions: list[int],
        log_probs: list[float],
        reward: float,
        global_state: np.ndarray,
        done: bool,
        value: float,
    ) -> None:
        """
        写入一个时间步的数据。

        Args:
            local_obs: 各 Agent 的局部观测列表
            actions: 各 Agent 的动作列表
            log_probs: 各 Agent 的对数概率列表
            reward: 共享奖励
            global_state: 全局状态
            done: 是否终止
            value: Critic 估计的价值
        """
        t = self.pos
        for i in range(self.num_agents):
            self.local_obs[i][t] = local_obs[i]
            self.actions[i][t] = actions[i]
            self.log_probs[i][t] = log_probs[i]
        self.rewards[t] = reward
        self.global_states[t] = global_state
        self.dones[t] = float(done)
        self.values[t] = value
        self.pos += 1

    def reset(self) -> None:
        """重置缓冲区指针（数据会被覆盖写入，无需显式清零）。"""
        self.pos = 0

    @property
    def full(self) -> bool:
        """缓冲区是否已填满。"""
        return self.pos >= self.n_steps

    def compute_gae(
        self, last_value: float, gamma: float, gae_lambda: float
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """
        使用 GAE (Generalized Advantage Estimation) 计算优势和回报。

        由于奖励和价值都是共享的，优势对所有 Agent 相同。

        Args:
            last_value: 最后一个状态的 Bootstrap 价值
            gamma: 折扣因子
            gae_lambda: GAE lambda 参数

        Returns:
            (advantages_per_agent, returns):
                advantages_per_agent: 每个 Agent 的优势数组（内容相同，列表副本）
                returns: 共享的回报数组
        """
        n = self.pos
        advantages = np.zeros(n, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(n)):
            next_value = last_value if t == n - 1 else self.values[t + 1]
            non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values[:n]
        # 每个 Agent 一份优势副本（内容相同，训练时各自切片）
        advantages_per_agent = [advantages.copy() for _ in range(self.num_agents)]
        return advantages_per_agent, returns


# ---------------------------------------------------------------------------
# 核心类：MultiAgentPPO
# ---------------------------------------------------------------------------


class MultiAgentPPO:
    """
    多智能体 PPO (MAPPO) 调度智能体。

    为每台量子机器配置一个独立 Actor，所有 Actor 共享一个集中式 Critic。
    采用 CTDE 范式：执行时去中心化（仅看本机局部观测），训练时中心化
    （Critic 看全局状态）。

    训练流程：
        1. 收集 rollout：每个 Agent 独立采样动作，包装器聚合成 env action
        2. 用集中式 Critic 计算 GAE 优势（优势对所有 Agent 共享）
        3. 每个 Agent 独立更新 Actor（PPO clipped objective + 熵正则）
        4. 集中式更新 Critic（MSE 损失）

    Attributes:
        env: 原始 QuantumSchedulingEnv
        wrapper: 多智能体环境包装器
        actors: 各机器的 Actor 网络列表
        critic: 共享的集中式 Critic
        device: 计算设备
    """

    def __init__(
        self,
        env: QuantumSchedulingEnv,
        learning_rate: float = 3e-4,
        n_steps: int = 1024,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        actor_hidden: tuple[int, ...] = (128, 64),
        critic_hidden: tuple[int, ...] = (256, 128),
        seed: int | None = None,
        log_dir: str = "./logs/marl/",
        device: str = "auto",
        verbose: int = 1,
    ):
        """
        初始化 MAPPO 智能体。

        Args:
            env: 已配置好机器的 QuantumSchedulingEnv
            learning_rate: 学习率（Actor 和 Critic 共用）
            n_steps: 每次 rollout 收集的步数
            batch_size: 小批量大小
            n_epochs: 每次更新的 epoch 数
            gamma: 折扣因子
            gae_lambda: GAE lambda
            clip_range: PPO 裁剪范围
            ent_coef: 熵正则系数
            vf_coef: 价值损失系数
            max_grad_norm: 梯度裁剪最大范数
            actor_hidden: Actor 隐藏层尺寸
            critic_hidden: Critic 隐藏层尺寸
            seed: 随机种子
            log_dir: 日志目录
            device: 计算设备（"auto"/"cpu"/"cuda"）
            verbose: 日志详细程度
        """
        self.env = env
        self.wrapper = MultiAgentEnvWrapper(env)
        self.num_agents = self.wrapper.num_agents
        self.machine_names = self.wrapper.machine_names
        self.local_obs_dim = self.wrapper.local_obs_dim
        self.global_state_dim = self.wrapper.local_obs_dim * self.num_agents

        # 超参数
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.seed = seed
        self.log_dir = log_dir
        self.verbose = verbose

        os.makedirs(log_dir, exist_ok=True)

        # 设备选择
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # 设置随机种子
        self._set_seed(seed)

        # 构建网络
        self.actors: list[ActorNet] = [
            ActorNet(self.local_obs_dim, 3, actor_hidden).to(self.device)
            for _ in range(self.num_agents)
        ]
        self.critic = CentralizedCritic(self.global_state_dim, critic_hidden).to(self.device)

        # 优化器（每个 Actor + Critic 各一个，或合并；这里独立便于精细控制）
        self.actor_optimizers = [
            Adam(actor.parameters(), lr=learning_rate, eps=1e-5) for actor in self.actors
        ]
        self.critic_optimizer = Adam(self.critic.parameters(), lr=learning_rate, eps=1e-5)

        # rollout 缓冲区
        self.buffer = RolloutBuffer(
            n_steps, self.num_agents, self.local_obs_dim, self.global_state_dim
        )

        # 训练统计
        self.total_timesteps = 0
        self._last_obs: dict[str, np.ndarray] | None = None
        self._last_global_state: np.ndarray | None = None

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------

    def _set_seed(self, seed: int | None) -> None:
        """设置全局随机种子以保证可复现。"""
        if seed is None:
            return
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _to_tensor(self, x: np.ndarray) -> torch.Tensor:
        """将 numpy 数组转为设备上的 float32 张量。"""
        return torch.as_tensor(x, dtype=torch.float32, device=self.device)

    def _global_state_tensor(self, global_state: np.ndarray) -> torch.Tensor:
        """将全局状态转为 Critic 输入张量（带 batch 维）。"""
        return self._to_tensor(global_state).unsqueeze(0)

    # ------------------------------------------------------------------
    # 动作采样
    # ------------------------------------------------------------------

    def _sample_actions(
        self, local_obs: dict[str, np.ndarray], deterministic: bool = False
    ) -> tuple[dict[str, int], list[float], float]:
        """
        为所有 Agent 采样动作。

        Args:
            local_obs: 机器名 -> 局部观测
            deterministic: 是否贪心选择

        Returns:
            (actions, log_probs, value):
                actions: 机器名 -> 动作索引
                log_probs: 各 Agent 的对数概率列表
                value: Critic 对当前全局状态的价值估计
        """
        actions: dict[str, int] = {}
        log_probs: list[float] = []
        with torch.no_grad():
            for i, name in enumerate(self.machine_names):
                obs_t = self._to_tensor(local_obs[name]).unsqueeze(0)
                action, log_prob, _ = self.actors[i].get_action(obs_t, deterministic=deterministic)
                actions[name] = int(action.item())
                log_probs.append(float(log_prob.item()))
            # 集中式 Critic 估值
            global_state = self.wrapper.get_global_state()
            value_t = self.critic(self._global_state_tensor(global_state))
            value = float(value_t.item())
        return actions, log_probs, value

    # ------------------------------------------------------------------
    # 训练
    # ------------------------------------------------------------------

    def train(
        self,
        total_timesteps: int = 50000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 5,
        log_interval: int = 1,
        **kwargs,
    ) -> MultiAgentPPO:
        """
        训练 MAPPO 智能体。

        采用 on-policy 训练：每个 rollout 周期收集 n_steps 步数据后，
        用 GAE 优势进行 n_epochs 轮 PPO 更新。

        Args:
            total_timesteps: 总训练步数
            eval_freq: 评估频率（步数）
            n_eval_episodes: 每次评估的回合数
            log_interval: 日志打印间隔（以 rollout 周期计）
            **kwargs: 预留扩展参数

        Returns:
            self（便于链式调用）
        """
        # 初始化环境
        self._last_obs, _ = self.wrapper.reset(seed=self.seed)
        self._last_global_state = self.wrapper.get_global_state()

        best_eval_reward = -float("inf")
        n_rollouts = 0

        while self.total_timesteps < total_timesteps:
            # ---- 1. 收集 rollout ----
            self._collect_rollout()

            # ---- 2. 计算 GAE 优势 ----
            # Bootstrap 最后一个状态的价值
            with torch.no_grad():
                last_value_t = self.critic(self._global_state_tensor(self._last_global_state))
                last_value = float(last_value_t.item())
            advantages_per_agent, returns = self.buffer.compute_gae(
                last_value, self.gamma, self.gae_lambda
            )

            # ---- 3. 更新网络 ----
            update_info = self._update(advantages_per_agent, returns)

            n_rollouts += 1
            if self.verbose >= 1 and n_rollouts % log_interval == 0:
                print(
                    f"[MAPPO] rollout={n_rollouts} steps={self.total_timesteps}/"
                    f"{total_timesteps} "
                    f"mean_reward={update_info['mean_reward']:.2f} "
                    f"actor_loss={update_info['mean_actor_loss']:.4f} "
                    f"critic_loss={update_info['critic_loss']:.4f} "
                    f"entropy={update_info['mean_entropy']:.4f}"
                )

            # ---- 4. 周期性评估 ----
            if (
                eval_freq > 0
                and self.total_timesteps >= eval_freq
                and (
                    self.total_timesteps // eval_freq
                    > (self.total_timesteps - self.n_steps) // eval_freq
                )
            ):
                eval_result = self.evaluate(num_episodes=n_eval_episodes)
                if self.verbose >= 1:
                    print(
                        f"[MAPPO] 评估: mean_reward={eval_result['mean_reward']:.2f} "
                        f"± {eval_result['std_reward']:.2f}"
                    )
                if eval_result["mean_reward"] > best_eval_reward:
                    best_eval_reward = eval_result["mean_reward"]
                    self._save_internal(os.path.join(self.log_dir, "best_model"))

        return self

    def _collect_rollout(self) -> None:
        """收集一个 rollout 周期的数据（n_steps 步）。"""
        self.buffer.reset()
        ep_rewards: list[float] = []
        cur_ep_reward = 0.0

        for _ in range(self.n_steps):
            actions, log_probs, value = self._sample_actions(self._last_obs)
            local_obs, reward, terminated, truncated, _info = self.wrapper.step(actions)
            done = bool(terminated or truncated)

            # 写入缓冲区（local_obs 是写入"当前"步的观测，即动作产生前的观测）
            self.buffer.add(
                local_obs=[self._last_obs[name] for name in self.machine_names],
                actions=[actions[name] for name in self.machine_names],
                log_probs=log_probs,
                reward=reward,
                global_state=self._last_global_state,
                done=done,
                value=value,
            )

            cur_ep_reward += reward
            self.total_timesteps += 1

            if done:
                ep_rewards.append(cur_ep_reward)
                cur_ep_reward = 0.0
                self._last_obs, _ = self.wrapper.reset()
            else:
                self._last_obs = local_obs
            self._last_global_state = self.wrapper.get_global_state()

        self._last_episode_rewards = ep_rewards

    def _update(
        self,
        advantages_per_agent: list[np.ndarray],
        returns: np.ndarray,
    ) -> dict[str, float]:
        """
        执行 n_epochs 轮 PPO 更新。

        Args:
            advantages_per_agent: 各 Agent 的优势数组
            returns: 共享回报数组

        Returns:
            更新统计字典
        """
        n = self.buffer.pos
        if n == 0:
            return {
                "mean_reward": 0.0,
                "mean_actor_loss": 0.0,
                "critic_loss": 0.0,
                "mean_entropy": 0.0,
            }

        # 全局状态和回报对所有 Agent 共享
        global_states = self.buffer.global_states[:n]
        returns_arr = returns[:n]
        # 标准化优势（提升训练稳定性）
        advantages_shared = advantages_per_agent[0]
        adv_mean = advantages_shared.mean()
        adv_std = advantages_shared.std()
        adv_std = max(adv_std, 1e-8)
        norm_advantages = (advantages_shared - adv_mean) / adv_std

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            # 随机打乱索引
            indices = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                # 集中式 Critic 更新（共享）
                gs_batch = self._to_tensor(global_states[batch_idx])
                returns_batch = self._to_tensor(returns_arr[batch_idx])
                values_pred = self.critic(gs_batch)
                critic_loss = nn.functional.mse_loss(values_pred, returns_batch)

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # 每个 Agent 独立更新 Actor
                batch_adv = self._to_tensor(norm_advantages[batch_idx])
                for i in range(self.num_agents):
                    obs_batch = self._to_tensor(self.buffer.local_obs[i][batch_idx])
                    act_batch = torch.as_tensor(
                        self.buffer.actions[i][batch_idx],
                        dtype=torch.long,
                        device=self.device,
                    )
                    old_log_prob_batch = self._to_tensor(self.buffer.log_probs[i][batch_idx])

                    new_log_prob, entropy = self.actors[i].evaluate_actions(obs_batch, act_batch)
                    # PPO 比率
                    ratio = torch.exp(new_log_prob - old_log_prob_batch)
                    surr1 = ratio * batch_adv
                    surr2 = (
                        torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * batch_adv
                    )
                    actor_loss = -torch.min(surr1, surr2).mean()
                    # 熵正则（鼓励探索）
                    entropy_mean = entropy.mean()
                    loss = actor_loss - self.ent_coef * entropy_mean

                    self.actor_optimizers[i].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.actors[i].parameters(), self.max_grad_norm)
                    self.actor_optimizers[i].step()

                    total_actor_loss += float(actor_loss.item())
                    total_entropy += float(entropy_mean.item())

                total_critic_loss += float(critic_loss.item())
                n_updates += 1

        mean_reward = float(self.buffer.rewards[:n].mean()) if n > 0 else 0.0
        return {
            "mean_reward": mean_reward,
            "mean_actor_loss": total_actor_loss / max(n_updates * self.num_agents, 1),
            "critic_loss": total_critic_loss / max(n_updates, 1),
            "mean_entropy": total_entropy / max(n_updates * self.num_agents, 1),
        }

    # ------------------------------------------------------------------
    # 推理与评估
    # ------------------------------------------------------------------

    def predict(self, deterministic: bool = True) -> int:
        """
        对当前环境状态生成聚合后的 env action。

        供 evaluate / 在线推理使用。要求调用前已 reset 环境。

        Args:
            deterministic: 是否贪心选择

        Returns:
            聚合后的 env action（0/1/2）
        """
        local_obs = self.wrapper.get_local_observations()
        actions, _, _ = self._sample_actions(local_obs, deterministic=deterministic)
        env_action, _ = self.wrapper.aggregate_actions(actions)
        return env_action

    def evaluate(self, num_episodes: int = 10, deterministic: bool = True) -> dict[str, float]:
        """
        评估 MAPPO 智能体性能。

        Args:
            num_episodes: 评估回合数
            deterministic: 是否确定性策略

        Returns:
            评估结果字典（mean_reward / std_reward / success_rate / num_episodes）
        """
        episode_rewards: list[float] = []
        episode_success: list[float] = []

        for _ in range(num_episodes):
            self.wrapper.reset()
            total_reward = 0.0
            done = False
            steps = 0
            while not done and steps < self.env.max_steps:
                env_action = self.predict(deterministic=deterministic)
                _, reward, terminated, truncated, info = self.env.step(env_action)
                total_reward += reward
                done = bool(terminated or truncated)
                steps += 1
            episode_rewards.append(total_reward)
            episode_success.append(float(info.get("completion_rate", 0.0)))

        return {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "success_rate": float(np.mean(episode_success)),
            "num_episodes": num_episodes,
        }

    # ------------------------------------------------------------------
    # 模型保存与加载
    # ------------------------------------------------------------------

    def _save_internal(self, path: str) -> None:
        """内部保存（不打印日志），用于 best_model 检查点。"""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        state = {
            "actors": [actor.state_dict() for actor in self.actors],
            "critic": self.critic.state_dict(),
            "config": {
                "num_agents": self.num_agents,
                "local_obs_dim": self.local_obs_dim,
                "global_state_dim": self.global_state_dim,
                "machine_names": self.machine_names,
            },
        }
        torch.save(state, path + ".pt")

    def save(self, path: str) -> None:
        """
        保存 MAPPO 模型到指定路径。

        Args:
            path: 保存路径（不含扩展名，将自动添加 .pt）
        """
        self._save_internal(path)
        if self.verbose >= 1:
            print(f"[MAPPO] 模型已保存至: {path}.pt")

    def load(self, path: str) -> None:
        """
        从文件加载 MAPPO 模型。

        Args:
            path: 模型文件路径（.pt 文件）
        """
        if not path.endswith(".pt"):
            path = path + ".pt"
        state = torch.load(path, map_location=self.device, weights_only=False)
        cfg = state["config"]
        # 校验配置一致性
        if cfg["num_agents"] != self.num_agents:
            raise ValueError(
                f"模型 num_agents={cfg['num_agents']} 与当前环境 "
                f"num_agents={self.num_agents} 不匹配"
            )
        for i, actor in enumerate(self.actors):
            actor.load_state_dict(state["actors"][i])
        self.critic.load_state_dict(state["critic"])
        if self.verbose >= 1:
            print(f"[MAPPO] 模型已从 {path} 加载")

    # ------------------------------------------------------------------
    # 配置信息
    # ------------------------------------------------------------------

    def get_config(self) -> dict[str, Any]:
        """
        获取智能体配置信息。

        Returns:
            配置字典
        """
        return {
            "architecture": "MAPPO",
            "num_agents": self.num_agents,
            "machine_names": self.machine_names,
            "local_obs_dim": self.local_obs_dim,
            "global_state_dim": self.global_state_dim,
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "device": str(self.device),
        }

    def __repr__(self) -> str:
        """智能体的字符串表示。"""
        cfg = self.get_config()
        return (
            f"MultiAgentPPO(\n"
            f"  架构={cfg['architecture']},\n"
            f"  Agent数={cfg['num_agents']},\n"
            f"  机器={cfg['machine_names']},\n"
            f"  局部观测维度={cfg['local_obs_dim']},\n"
            f"  全局状态维度={cfg['global_state_dim']},\n"
            f"  学习率={cfg['learning_rate']},\n"
            f"  n_steps={cfg['n_steps']},\n"
            f"  batch_size={cfg['batch_size']}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# 模块测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from src.scheduler.env import DEFAULT_MACHINE_CONFIGS

    env = QuantumSchedulingEnv(max_steps=200, machine_configs=DEFAULT_MACHINE_CONFIGS)
    agent = MultiAgentPPO(env, n_steps=256, batch_size=64, n_epochs=4, verbose=1)
    print("=" * 60)
    print("MAPPO 智能体初始化完成")
    print("=" * 60)
    print(agent)
    print()
    print("配置详情:")
    for k, v in agent.get_config().items():
        print(f"  {k}: {v}")

    print("\n--- 快速训练（2000 步）---")
    agent.train(total_timesteps=2000, eval_freq=1000, n_eval_episodes=3)
    print("\n训练完成。")
