"""
RL 调度智能体模块
Reinforcement Learning Agent for Quantum-Classical Hybrid Task Scheduling

基于 Stable-Baselines3 的 DQN (Deep Q-Network) 智能体，用于量子-经典混合计算
任务调度决策。支持 Dueling DQN 架构、Epsilon-Greedy 探索策略以及 TensorBoard
训练可视化。

状态空间（10维，对应 env.py 的 QuantumSchedulingEnv）：
    0 - qubit_availability  : 当前可用量子比特比率（0-1）
    1 - queue_length         : 当前任务队列长度（归一化 0-1）
    2 - avg_wait_time        : 队列中任务平均等待时间（归一化）
    3 - fidelity             : 当前量子比特平均保真度（0-1）
    4 - classical_load       : 经典计算资源负载（0-1）
    5 - quantum_queue_ratio  : 量子专用队列占比（0-1）
    6 - time_of_day          : 一天中的时间段（0-1，模拟昼夜负载差异）
    7 - urgency_level        : 当前任务的紧急程度（0-1）
    8 - task_type_quantum    : 当前任务是否为 quantum 类型（0-1）
    9 - task_type_classical  : 当前任务是否为 classical 类型（0-1）

动作空间（Discrete(3)）：
    0 - 分配到经典计算资源
    1 - 分配到量子计算资源
    2 - 混合执行（量子-经典协同）
"""

import json
import os
import random
import time
from typing import Any

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from loguru import logger
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DQN, PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import load_from_zip_file
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp,
)
from stable_baselines3.common.utils import get_device
from stable_baselines3.dqn.policies import QNetwork
from torch import nn

from src.quantum.annealing import QuantumAnnealingOptimizer

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


# ---------------------------------------------------------------------------
# 自定义回调：记录探索率衰减
# ---------------------------------------------------------------------------


class EpsilonExplorationCallback(BaseCallback):
    """
    Epsilon-Greedy 探索率回调

    在训练过程中监控并衰减探索率 epsilon：
        - 初始 epsilon = 1.0（完全随机探索）
        - 最终 epsilon = 0.05（保持少量探索）
        - 每次回调触发时：epsilon *= 0.995

    同时将 epsilon 值记录到 TensorBoard 供可视化分析。
    """

    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        decay_freq: int = 1,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.decay_freq = decay_freq

    def _on_step(self) -> bool:
        """每步触发：衰减 epsilon 并记录到 TensorBoard"""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
        # 记录到 TensorBoard
        self.logger.record("exploration/epsilon", self.epsilon)
        return True


# ---------------------------------------------------------------------------
# 核心类：SchedulerAgent
# ---------------------------------------------------------------------------


class SchedulerAgent:
    """
    基于 DQN 的量子-经典混合调度智能体

    使用 Stable-Baselines3 的 DQN 算法，配合 Dueling DQN 架构实现
    高效的量子计算任务调度决策。智能体通过与环境交互学习最优调度策略，
    实现量子资源与经典资源的智能分配。

    训练配置：
        - batch_size          : 64
        - gamma               : 0.99
        - learning_rate      : 0.001
        - buffer_size         : 10000
        - target_update_interval : 500
        - train_freq          : (1, "step")（每步训练一次）
        - epsilon_start       : 1.0
        - epsilon_end         : 0.05
        - epsilon_decay       : 0.995

    使用方法：
        >>> from src.scheduler.env import SchedulingEnv
        >>> from src.scheduler.agent import SchedulerAgent
        >>> env = SchedulingEnv()
        >>> agent = SchedulerAgent(env)
        >>> model = agent.train(total_timesteps=100000)
        >>> action = agent.predict(state, deterministic=True)
        >>> avg_reward, success_rate = agent.evaluate(num_episodes=10)
        >>> agent.save("./models/scheduler_agent")
    """

    # ========================= 默认训练超参数 =========================
    DEFAULT_LEARNING_RATE: float = 0.001  # 学习率
    DEFAULT_BUFFER_SIZE: int = 10000  # 经验回放缓冲区大小
    DEFAULT_BATCH_SIZE: int = 64  # 训练批量大小
    DEFAULT_GAMMA: float = 0.99  # 折扣因子（长期回报权重）
    DEFAULT_TARGET_UPDATE_INTERVAL: int = 500  # 目标网络更新间隔（步数）
    DEFAULT_TRAIN_FREQ: tuple[int, str] = (1, "step")  # 训练频率：每步一次
    DEFAULT_EPSILON_START: float = 1.0  # 初始探索率（完全探索）
    DEFAULT_EPSILON_END: float = 0.05  # 最终探索率（保持少量探索）
    DEFAULT_EPSILON_DECAY: float = 0.995  # 探索率衰减系数
    DEFAULT_LEARNING_STARTS: int = 100  # 开始训练前的随机探索步数
    DEFAULT_TAU: float = 1.0  # 目标网络软更新系数（1.0 = 硬更新）
    DEFAULT_LOG_DIR: str = "./logs/"  # TensorBoard 日志目录
    DEFAULT_VERBOSE: int = 1  # 训练日志详细程度

    # 策略网络隐藏层架构
    NET_ARCH: list = [128, 64]  # noqa: RUF012

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        gamma: float = DEFAULT_GAMMA,
        target_update_interval: int = DEFAULT_TARGET_UPDATE_INTERVAL,
        train_freq: tuple[int, str] = DEFAULT_TRAIN_FREQ,
        epsilon_start: float = DEFAULT_EPSILON_START,
        epsilon_end: float = DEFAULT_EPSILON_END,
        epsilon_decay: float = DEFAULT_EPSILON_DECAY,
        learning_starts: int = DEFAULT_LEARNING_STARTS,
        tau: float = DEFAULT_TAU,
        log_dir: str = DEFAULT_LOG_DIR,
        verbose: int = DEFAULT_VERBOSE,
        seed: int | None = None,
    ):
        """
        初始化调度智能体

        接收一个 Gymnasium 环境，自动推断状态/动作空间维度，
        创建 Dueling DQN 模型和 Epsilon-Greedy 探索回调。

        Args:
            env: Gymnasium 调度环境（如 SchedulingEnv）
            learning_rate: 学习率，默认 0.001
            buffer_size: 经验回放缓冲区容量，默认 10000
            batch_size: 每次训练的批量大小，默认 64
            gamma: 折扣因子，控制未来奖励的衰减程度，默认 0.99
            target_update_interval: 目标网络参数同步间隔（步数），默认 500
            train_freq: 训练频率，默认 (1, "step") 表示每步训练一次
            epsilon_start: Epsilon-Greedy 初始探索率，默认 1.0
            epsilon_end: Epsilon-Greedy 最终探索率，默认 0.05
            epsilon_decay: Epsilon-Greedy 衰减系数，默认 0.995
            learning_starts: 开始训练前的随机探索步数，默认 100
            tau: 目标网络软更新系数，1.0 为硬更新，默认 1.0
            log_dir: TensorBoard 日志保存目录，默认 "./logs/"
            verbose: 训练日志输出详细程度（0=静默，1=进度条），默认 1
            seed: 随机种子，用于可复现实验
        """
        self.env = env
        self.learning_rate = learning_rate
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_interval = target_update_interval
        self.train_freq = train_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_starts = learning_starts
        self.tau = tau
        self.log_dir = log_dir
        self.verbose = verbose
        self.seed = seed

        # 自动从环境推断空间维度
        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)

        # 初始化模型（延迟到 train() 或 predict() 时创建）
        self.model: DQN | None = None

    def _build_model(self) -> DQN:
        """
        构建 Dueling DQN 模型

        使用自定义的 DuelingQNetwork 作为策略网络，
        配置经验回放、目标网络更新等超参数。

        Returns:
            配置好的 DQN 模型实例
        """
        # 构建 Dueling DQN 策略参数字典
        policy_kwargs = {
            "net_arch": self.NET_ARCH,
        }

        model = DQN(
            policy="MlpPolicy",  # 使用 MLP 策略（将通过 policy_kwargs 替换为 Dueling）
            env=self.env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            gamma=self.gamma,
            target_update_interval=self.target_update_interval,
            train_freq=self.train_freq,
            learning_starts=self.learning_starts,
            tau=self.tau,
            policy_kwargs=policy_kwargs,
            verbose=self.verbose,
            seed=self.seed,
            tensorboard_log=self.log_dir,
            # 探索参数
            exploration_initial_eps=self.epsilon_start,
            exploration_final_eps=self.epsilon_end,
            exploration_fraction=0.5,  # 探索占总训练步数的比例
        )

        # 替换为 Dueling DQN 网络
        self._replace_with_dueling(model)

        return model

    def _replace_with_dueling(self, model: DQN) -> None:
        """
        将标准 Q 网络替换为 Dueling Q 网络（SB3 2.0+ 兼容）

        从现有策略网络中提取 features_extractor 和 features_dim，
        然后创建 DuelingQNetwork 替换 q_net 和 q_net_target。

        Args:
            model: SB3 DQN 模型实例
        """
        device = get_device(model.device)

        # 复用现有策略的 features_extractor 和维度信息
        old_q_net = model.policy.q_net
        features_extractor = old_q_net.features_extractor
        features_dim = old_q_net.features_dim

        # 创建 Dueling Q 网络
        dueling_net = DuelingQNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=self.NET_ARCH,
        ).to(device)

        # 替换策略网络中的 q_net 和 q_net_target
        model.policy.q_net = dueling_net
        model.policy.q_net_target = DuelingQNetwork(
            observation_space=self.observation_space,
            action_space=self.action_space,
            features_extractor=features_extractor,
            features_dim=features_dim,
            net_arch=self.NET_ARCH,
        ).to(device)

    def train(
        self,
        total_timesteps: int = 100000,
        eval_freq: int = 1000,
        n_eval_episodes: int = 5,
        log_dir: str | None = None,
        **kwargs,
    ) -> DQN:
        """
        训练 DQN 调度智能体

        使用 Dueling DQN + Epsilon-Greedy 探索策略训练调度智能体。
        训练过程中会自动进行周期性评估，并将训练曲线记录到 TensorBoard。

        训练流程：
            1. 构建 Dueling DQN 模型
            2. 配置 Epsilon-Greedy 回调（监控探索率衰减）
            3. 配置评估回调（周期性评估模型性能）
            4. 调用 SB3 的 learn() 方法开始训练
            5. 训练完成后返回模型

        Args:
            total_timesteps: 总训练步数，默认 100000
            eval_freq: 评估频率（每隔多少步评估一次），默认 1000
            n_eval_episodes: 每次评估运行的回合数，默认 5
            log_dir: 本次训练的日志子目录，默认 None（自动生成）
            **kwargs: 传递给 DQN.learn() 的额外参数

        Returns:
            训练好的 DQN 模型实例
        """
        # 构建模型（如尚未构建）
        if self.model is None:
            self.model = self._build_model()

        # 创建评估环境
        eval_env = Monitor(self.env)

        # 构建 Epsilon 探索回调
        epsilon_callback = EpsilonExplorationCallback(
            epsilon_start=self.epsilon_start,
            epsilon_end=self.epsilon_end,
            epsilon_decay=self.epsilon_decay,
        )

        # 构建评估回调
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=os.path.join(self.log_dir, "best_model"),
            log_path=os.path.join(self.log_dir, "eval_results"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=False,
        )

        # 合并所有回调
        callback = CallbackList([epsilon_callback, eval_callback])

        # 设置 TensorBoard 日志名
        tb_log_name = log_dir if log_dir else "dqn_scheduling"

        # 开始训练
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=tb_log_name,
            reset_num_timesteps=True,
            **kwargs,
        )

        return self.model

    def predict(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> int:
        """
        使用训练好的模型进行调度决策

        给定当前环境状态，输出最优调度动作（任务分配决策）。

        Args:
            state: 当前环境状态向量，形状为 (obs_dim,) 或 (1, obs_dim)
            deterministic: 是否使用确定性策略（贪心选择），
                         True 用于推理/部署，False 用于继续探索

        Returns:
            动作索引（0=经典资源，1=量子资源，2=混合执行）
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练！请先调用 train() 方法或使用 load() 加载已训练模型。")

        # 确保状态是二维张量 (1, obs_dim)
        if state.ndim == 1:
            state = state.reshape(1, -1)

        # 使用 SB3 模型预测
        action, _ = self.model.predict(
            state,
            deterministic=deterministic,
        )

        return int(action.item())

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """
        评估训练好的智能体性能

        运行指定数量的评估回合，统计平均奖励和任务调度成功率。

        Args:
            num_episodes: 评估回合数，默认 10
            deterministic: 是否使用确定性策略，默认 True

        Returns:
            评估结果字典，包含：
                - mean_reward: 平均累积奖励
                - std_reward: 奖励标准差
                - success_rate: 成功率（平均每回合完成率）
                - num_episodes: 评估回合数
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练！请先调用 train() 方法或使用 load() 加载已训练模型。")

        episode_rewards = []
        episode_success_rates = []

        for _ep in range(num_episodes):
            obs, info = self.env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated

            episode_rewards.append(total_reward)

            # 从 info 中提取完成率作为成功率
            completion_rate = info.get("completion_rate", 0.0)
            episode_success_rates.append(completion_rate)

        result = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "success_rate": float(np.mean(episode_success_rates)),
            "num_episodes": num_episodes,
        }

        return result

    def save(self, path: str) -> None:
        """
        保存训练好的模型到指定路径

        将模型参数、优化器状态和探索率保存到文件，支持后续加载恢复。

        Args:
            path: 模型保存路径（不含扩展名，将自动添加 .zip）
        """
        if self.model is None:
            raise RuntimeError("没有可保存的模型！请先训练或加载模型。")

        # SB3 的 save 方法会自动添加 .zip 扩展名
        self.model.save(path)
        logger.info(f"[SchedulerAgent] 模型已保存至: {path}.zip")

    def load(self, path: str) -> None:
        """
        从文件加载已训练的模型

        加载之前保存的模型参数、优化器状态和探索率。
        加载后可直接用于 predict() 或 evaluate()。

        Args:
            path: 模型文件路径（SB3 会自动处理 .zip 扩展名）
        """
        _data, params, _ = load_from_zip_file(path, device="cpu")

        self.model = DQN(
            policy="MlpPolicy",
            env=self.env,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            batch_size=self.batch_size,
            gamma=self.gamma,
            verbose=self.verbose,
            policy_kwargs={"net_arch": self.NET_ARCH},
        )

        if "policy" in params:
            policy_state = params["policy"]
            dueling_keys = [
                k for k in policy_state if "value_stream" in k or "advantage_stream" in k
            ]
            if dueling_keys:
                self._replace_with_dueling(self.model)
                q_net_state = {}
                q_net_target_state = {}
                for k, v in policy_state.items():
                    if k.startswith("q_net."):
                        q_net_state[k.replace("q_net.", "")] = v
                    elif k.startswith("q_net_target."):
                        q_net_target_state[k.replace("q_net_target.", "")] = v
                self.model.policy.q_net.load_state_dict(q_net_state, strict=False)
                self.model.policy.q_net_target.load_state_dict(q_net_target_state, strict=False)
            else:
                self.model.set_parameters(params, exact_match=True, device=self.model.device)
        else:
            self.model.set_parameters(params, exact_match=True, device=self.model.device)
        logger.info(f"[SchedulerAgent] 模型已从 {path} 加载")

    def get_config(self) -> dict[str, Any]:
        """
        获取智能体配置信息

        Returns:
            包含所有超参数和空间维度的字典
        """
        return {
            "observation_dim": self.observation_space.shape[0],
            "action_dim": self.action_space.n,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "gamma": self.gamma,
            "target_update_interval": self.target_update_interval,
            "train_freq": self.train_freq,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay": self.epsilon_decay,
            "learning_starts": self.learning_starts,
            "tau": self.tau,
            "log_dir": self.log_dir,
            "net_arch": self.NET_ARCH,
            "architecture": "Dueling DQN",
        }

    def __repr__(self) -> str:
        """智能体的字符串表示"""
        config = self.get_config()
        return (
            f"SchedulerAgent(\n"
            f"  架构={config['architecture']},\n"
            f"  状态维度={config['observation_dim']},\n"
            f"  动作维度={config['action_dim']},\n"
            f"  隐藏层={config['net_arch']},\n"
            f"  学习率={config['learning_rate']},\n"
            f"  gamma={config['gamma']},\n"
            f"  探索率={config['epsilon_start']}->{config['epsilon_end']},\n"
            f"  衰减率={config['epsilon_decay']}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# 量子退火回调
# ---------------------------------------------------------------------------


class AnnealingCallback(BaseCallback):
    """
    每 N 步用量子退火优化 PPO 网络权重的回调。

    量子退火可以加速 PPO 的策略优化，通过在退火过程中探索
    更优的权重组合来提升策略性能。

    Attributes:
        optimizer: 量子退火优化器
        interval: 退火间隔（步数）
        best_reward: 最佳奖励值
        optimized_count: 累计优化次数
        head_only: 是否仅优化网络输出头权重（避免全量参数 OOM）
    """

    def __init__(self, optimizer, interval=1000, verbose=0, head_only=True):
        super().__init__(verbose)
        self.optimizer = optimizer
        self.interval = interval
        self.best_reward = -float("inf")
        self.optimized_count = 0
        self.head_only = head_only

    def _on_step(self) -> bool:
        """每步检查是否需要触发退火优化。"""
        if self.n_calls % self.interval == 0 and self.n_calls > 0:
            try:
                optimized_agent = self.optimizer.optimize_policy(
                    self.model,
                    head_only=self.head_only,
                )

                quality = 0.0
                if hasattr(self.optimizer, "_evaluate_network_quality"):
                    policy_net = self.optimizer._get_policy_net(optimized_agent)
                    if policy_net is not None:
                        loss = self.optimizer._evaluate_network_quality(policy_net)
                        quality = -loss

                if quality > self.best_reward:
                    self.best_reward = quality
                    self.optimized_count += 1

                    if self.verbose:
                        logger.info(
                            f"[退火] 步数{self.n_calls}: 优化完成 (质量={quality:.4f}, "
                            f"累计优化{self.optimized_count}次)"
                        )
            except Exception as e:
                # 量子退火优化可能抛出多种异常（dimod/neal/torch），无法精确收窄
                if self.verbose:
                    logger.warning(f"[退火] 步数{self.n_calls}: 退火跳过 ({e})")
        return True


# ---------------------------------------------------------------------------
# 真机抽样回调：训练过程中按概率向天衍云真机提交任务
# ---------------------------------------------------------------------------


class RealMachineCallback(BaseCallback):
    """每 N 步抽样 1 个任务提交真机，记录真实耗时。

    在 PPO 训练过程中，每隔 ``interval`` 步以概率 ``prob`` 从当前任务队列
    中随机抽取一个任务，构建最小 QCIS 电路并提交到天衍云真机，记录提交
    耗时与 task_id。训练结束时自动保存记录到 ``save_path``（默认
    ``results/real_times.json``）。

    若环境未绑定真机客户端（``env._real_clients`` 为空且未显式传入
    ``client``），回调自动降级为 no-op，仅打印一次告警，不影响训练流程。

    典型用法（PPO 训练时启用真机抽样）::

        env = QuantumSchedulingEnv(machine_configs=DEFAULT_MACHINE_CONFIGS)
        env.attach_real_clients(real_clients)   # 绑定 cqlib 客户端
        agent = PPOAgent(env)
        agent.train(
            total_timesteps=5000,
            real_callback_interval=1000,  # 每 1000 步抽样一次
            real_callback_prob=0.5,       # 抽样时 50% 概率提交真机
        )
        # 训练结束后 results/real_times.json 已生成

    Attributes:
        env        : 训练环境（需已 attach_real_clients）
        interval   : 抽样间隔（步数）
        prob       : 每次触发的提交概率（控制机时消耗，建议 0.01-0.05）
        client     : 显式指定的真机客户端；None 时自动取 env._real_clients 第一项
        save_path  : 真机提交记录 JSON 保存路径
        real_times : 真机提交记录列表 [{step, task_id, machine, latency_s, status, real_task_id}]
    """

    def __init__(
        self,
        env,
        interval: int = 1000,
        prob: float = 0.05,
        client=None,
        save_path: str = "results/real_times.json",
        shots: int = 512,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.env = env
        self.interval = int(interval)
        self.prob = float(prob)
        self.client = client
        self.save_path = save_path
        self.shots = int(shots)
        self.real_times: list[dict[str, Any]] = []
        self._warned_no_client = False

    def _on_step(self) -> bool:
        """每步触发：达到 interval 时按 prob 概率提交真机任务。"""
        # 仅在 interval 倍数步触发；跳过第 0 步（环境尚未 reset 完成）
        if self.n_calls == 0 or self.n_calls % self.interval != 0:
            return True
        if self.prob <= 0.0:
            return True
        # 概率门控：未命中则跳过本次
        if random.random() >= self.prob:
            return True

        # 解析可用的真机客户端（显式传入优先；否则从 env._real_clients 取第一项）
        client = self.client
        machine_name = getattr(client, "machine_name", "unknown") if client else "unknown"
        if client is None:
            real_clients = getattr(self.env, "_real_clients", {}) or {}
            if not real_clients:
                if not self._warned_no_client:
                    logger.warning(
                        f"[RealCallback] env 未绑定真机客户端，真机抽样已禁用 "
                        f"(step={self.n_calls})"
                    )
                    self._warned_no_client = True
                return True
            machine_name = next(iter(real_clients.keys()))
            client = real_clients[machine_name]

        # 从环境取一个待处理任务（队列空时退化为当前任务 / None）
        task = None
        if hasattr(self.env, "get_random_pending_task"):
            try:
                task = self.env.get_random_pending_task()
            except Exception as e:
                # 防御性捕获：env 内部状态访问可能抛出多种异常，降级为无任务
                logger.debug(f"[RealCallback] 获取待处理任务失败: {e}")
                task = None

        # 构造 QCIS（Task 无 qcis 字段时用最小占位电路保证可执行）
        qcis = "H Q0\nM Q0"
        task_id_str = "synthetic"
        if task is not None:
            task_id_str = str(getattr(task, "task_id", "synthetic"))
            qcis = getattr(task, "qcis", None) or "H Q0\nM Q0"

        # 提交并计时（异常安全，失败仅记录，不中断训练）
        t0 = time.time()
        record: dict[str, Any] = {
            "step": int(self.n_calls),
            "task_id": task_id_str,
            "machine": machine_name,
            "latency_s": 0.0,
            "status": "failed",
            "real_task_id": None,
        }
        try:
            real_tid = client.submit_quantum_task(
                qcis=qcis,
                shots=self.shots,
                task_name=f"RLCallback_{task_id_str}_step{self.n_calls}",
            )
            record["latency_s"] = round(time.time() - t0, 3)
            record["real_task_id"] = str(real_tid) if real_tid else None
            record["status"] = "submitted" if real_tid else "rejected"
            if self.verbose:
                logger.info(
                    f"[RealCallback] step={self.n_calls} machine={machine_name} "
                    f"tid={real_tid} latency={record['latency_s']}s "
                    f"task={task_id_str}"
                )
        except Exception as e:
            # 真机 API 提交可能因网络/认证/服务端等多种原因失败，无法精确收窄
            record["latency_s"] = round(time.time() - t0, 3)
            record["status"] = f"error: {str(e)[:80]}"
            if self.verbose:
                logger.error(f"[RealCallback] step={self.n_calls} 提交失败: {e}")

        self.real_times.append(record)
        return True

    def _on_training_end(self) -> None:
        """训练结束时保存真机提交记录到 JSON 文件。"""
        if not self.save_path:
            return
        save_dir = os.path.dirname(self.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        try:
            with open(self.save_path, "w", encoding="utf-8") as f:
                json.dump(self.real_times, f, ensure_ascii=False, indent=2)
            if self.verbose:
                logger.info(
                    f"[RealCallback] 真机提交记录已保存: {self.save_path} "
                    f"(共 {len(self.real_times)} 条)"
                )
        except OSError as e:
            logger.error(f"[RealCallback] 保存记录失败: {e}")

# ---------------------------------------------------------------------------
# PPO 智能体
# ---------------------------------------------------------------------------


class PPOAgent:
    """
    PPO (Proximal Policy Optimization) 调度智能体

    使用 PPO 算法进行量子-经典混合计算任务调度。
    PPO 在连续动作空间和高维状态空间中通常表现更稳定。

    Attributes:
        env: 训练环境
        model: 训练好的 PPO 模型
        learning_rate: 学习率
        n_steps: 每次更新的步数
        batch_size: 批次大小
        n_epochs: 每次更新的 epoch 数
        gamma: 折扣因子
        verbose: 日志详细程度
    """

    def __init__(self, env, **kwargs):
        """
        初始化 PPO 智能体。

        Args:
            env: Gymnasium 环境实例
            **kwargs: PPO 超参数
        """
        self.env = env
        self.model = None

        self.learning_rate = kwargs.get("learning_rate", 3e-4)
        self.n_steps = kwargs.get("n_steps", 2048)
        self.batch_size = kwargs.get("batch_size", 64)
        self.n_epochs = kwargs.get("n_epochs", 10)
        self.gamma = kwargs.get("gamma", 0.99)
        self.gae_lambda = kwargs.get("gae_lambda", 0.95)
        self.clip_range = kwargs.get("clip_range", 0.2)
        self.ent_coef = kwargs.get("ent_coef", 0.01)
        self.vf_coef = kwargs.get("vf_coef", 0.5)
        self.max_grad_norm = kwargs.get("max_grad_norm", 0.5)
        self.verbose = kwargs.get("verbose", 1)
        self.seed = kwargs.get("seed")
        self.log_dir = kwargs.get("log_dir", "./logs/")

        # LSTM 策略支持（阶段3）
        self.use_lstm = kwargs.get("use_lstm", False)
        self.n_lstm_layers = kwargs.get("n_lstm_layers", 1)
        self.lstm_hidden_size = kwargs.get("lstm_hidden_size", 64)

        os.makedirs(self.log_dir, exist_ok=True)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # 量子退火器初始化（可选功能）
        self.use_annealing = kwargs.get("use_annealing", False)
        self.annealing_optimizer = None
        self.anneal_interval = kwargs.get("anneal_interval", 1000)
        if self.use_annealing:
            # simulation_mode=False 且提供 cqlib_client 时尝试真机退火；
            # cqlib 为门控量子 SDK 无退火接口，会在 anneal() 中自动降级为仿真
            self.annealing_optimizer = QuantumAnnealingOptimizer(
                num_qubits=kwargs.get("anneal_qubits", 10),
                annealing_time=kwargs.get("annealing_time", 20.0),
                shots=kwargs.get("anneal_shots", 1000),
                simulation_mode=kwargs.get("anneal_simulation_mode", True),
                cqlib_client=kwargs.get("anneal_cqlib_client"),
            )
            sim_tag = "仿真" if self.annealing_optimizer.simulation_mode else "真机"
            logger.info(
                f"[PPOAgent] 量子退火器已启用（{sim_tag}模式），"
                f"退火间隔={self.anneal_interval}步"
            )

    def _build_model(self) -> PPO:
        """
        构建 PPO 模型。

        Returns:
            构建好的 PPO 模型实例
        """
        # 根据 use_lstm 选择策略类型
        if self.use_lstm:
            # 使用 RecurrentPPO 支持 LSTM 策略
            policy_kwargs = {
                "n_lstm_layers": self.n_lstm_layers,
                "lstm_hidden_size": self.lstm_hidden_size,
                "net_arch": [128, 64],
            }
            logger.info(
                f"[PPOAgent] 使用 LSTM 策略: layers={self.n_lstm_layers}, "
                f"hidden_size={self.lstm_hidden_size}"
            )
            model = RecurrentPPO(
                "MlpLstmPolicy",
                self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                verbose=self.verbose,
                seed=self.seed,
                tensorboard_log=self.log_dir,
                policy_kwargs=policy_kwargs,
            )
        else:
            policy_kwargs = {"net_arch": [128, 64]}
            model = PPO(
                "MlpPolicy",
                self.env,
                learning_rate=self.learning_rate,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                n_epochs=self.n_epochs,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                clip_range=self.clip_range,
                ent_coef=self.ent_coef,
                vf_coef=self.vf_coef,
                max_grad_norm=self.max_grad_norm,
                verbose=self.verbose,
                seed=self.seed,
                tensorboard_log=self.log_dir,
                policy_kwargs=policy_kwargs,
            )
        return model

    def train(
        self,
        total_timesteps: int = 50000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10,
        log_dir: str | None = None,
        **kwargs,
    ) -> PPO:
        """
        训练 PPO 调度智能体。

        Args:
            total_timesteps: 总训练步数
            eval_freq: 评估频率
            n_eval_episodes: 每次评估的回合数
            log_dir: 日志目录
            **kwargs: 额外参数，支持以下真机抽样回调参数：
                - real_callback_interval: 真机抽样间隔（步数），>0 时启用，默认 0（禁用）
                - real_callback_prob    : 每次触发的提交概率，默认 0.05
                - real_callback_client  : 显式指定的真机客户端，None 时取 env._real_clients
                - real_callback_save_path: 真机提交记录 JSON 保存路径，默认 results/real_times.json
                - real_callback_shots   : 真机任务 shots，默认 512

        Returns:
            训练好的 PPO 模型
        """
        if self.model is None:
            self.model = self._build_model()

        # 弹出真机抽样回调参数（不传给 learn）
        real_cb_interval = int(kwargs.pop("real_callback_interval", 0))
        real_cb_prob = float(kwargs.pop("real_callback_prob", 0.05))
        real_cb_client = kwargs.pop("real_callback_client", None)
        real_cb_save_path = kwargs.pop("real_callback_save_path", "results/real_times.json")
        real_cb_shots = int(kwargs.pop("real_callback_shots", 512))

        eval_env = Monitor(self.env)
        eval_callback = EvalCallback(
            eval_env=eval_env,
            best_model_save_path=os.path.join(self.log_dir, "best_model"),
            log_path=os.path.join(self.log_dir, "eval_results"),
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            deterministic=True,
        )

        # 构建回调列表
        callbacks = [eval_callback]

        # 如果启用了量子退火，添加退火回调
        if self.use_annealing and self.annealing_optimizer:
            annealing_callback = AnnealingCallback(
                optimizer=self.annealing_optimizer,
                interval=self.anneal_interval,
                verbose=1,
            )
            callbacks.append(annealing_callback)

        # 真机抽样回调（interval>0 时启用，需 env 已 attach_real_clients 或显式传 client）
        if real_cb_interval > 0:
            real_callback = RealMachineCallback(
                env=self.env,
                interval=real_cb_interval,
                prob=real_cb_prob,
                client=real_cb_client,
                save_path=real_cb_save_path,
                shots=real_cb_shots,
                verbose=1,
            )
            callbacks.append(real_callback)
            logger.info(
                f"[PPOAgent] 真机抽样已启用: interval={real_cb_interval} "
                f"prob={real_cb_prob} save={real_cb_save_path}"
            )

        # 合并回调
        callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]

        tb_log_name = log_dir if log_dir else "ppo_scheduling"

        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=tb_log_name,
            reset_num_timesteps=True,
            **kwargs,
        )

        return self.model

    def predict(
        self,
        state: np.ndarray,
        deterministic: bool = True,
    ) -> int:
        """
        使用训练好的模型进行调度决策。

        Args:
            state: 当前环境状态向量
            deterministic: 是否使用确定性策略

        Returns:
            动作索引
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练！请先调用 train() 方法或使用 load() 加载已训练模型。")

        if state.ndim == 1:
            state = state.reshape(1, -1)

        action, _ = self.model.predict(state, deterministic=deterministic)
        return int(action.item())

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> dict[str, float]:
        """
        评估训练好的智能体性能。

        Args:
            num_episodes: 评估回合数
            deterministic: 是否使用确定性策略

        Returns:
            评估结果字典
        """
        if self.model is None:
            raise RuntimeError("模型尚未训练！请先调用 train() 方法或使用 load() 加载已训练模型。")

        episode_rewards = []
        episode_success_rates = []

        for _ep in range(num_episodes):
            obs, info = self.env.reset()
            total_reward = 0.0
            done = False

            while not done:
                action = self.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                done = terminated or truncated

            episode_rewards.append(total_reward)
            completion_rate = info.get("completion_rate", 0.0)
            episode_success_rates.append(completion_rate)

        result = {
            "mean_reward": float(np.mean(episode_rewards)),
            "std_reward": float(np.std(episode_rewards)),
            "success_rate": float(np.mean(episode_success_rates)),
            "num_episodes": num_episodes,
        }

        return result

    def save(self, path: str) -> None:
        """
        保存训练好的模型到指定路径。

        Args:
            path: 模型保存路径
        """
        if self.model is None:
            raise RuntimeError("没有可保存的模型！请先训练或加载模型。")

        self.model.save(path)
        logger.info(f"[PPOAgent] 模型已保存至: {path}.zip")

    def load(self, path: str) -> None:
        """
        从文件加载已训练的模型。

        Args:
            path: 模型文件路径
        """
        self.model = PPO.load(path, env=self.env)
        logger.info(f"[PPOAgent] 模型已从 {path} 加载")

    def get_config(self) -> dict[str, Any]:
        """
        获取智能体配置信息。

        Returns:
            配置字典
        """
        return {
            "observation_dim": self.observation_space.shape[0],
            "action_dim": self.action_space.n,
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
            "architecture": "PPO",
        }

    def __repr__(self) -> str:
        """智能体的字符串表示"""
        config = self.get_config()
        return (
            f"PPOAgent(\n"
            f"  架构={config['architecture']},\n"
            f"  状态维度={config['observation_dim']},\n"
            f"  动作维度={config['action_dim']},\n"
            f"  学习率={config['learning_rate']},\n"
            f"  gamma={config['gamma']},\n"
            f"  n_steps={config['n_steps']},\n"
            f"  batch_size={config['batch_size']}\n"
            f")"
        )


# ---------------------------------------------------------------------------
# 模块测试入口
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    _PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from src.scheduler.env import QuantumSchedulingEnv

    # 创建环境
    env = QuantumSchedulingEnv()

    # 创建智能体
    agent = SchedulerAgent(
        env=env,
        learning_rate=0.001,
        buffer_size=10000,
        verbose=1,
    )

    # 打印智能体配置
    print("=" * 50)
    print("调度智能体初始化完成")
    print("=" * 50)
    print(agent)
    print()
    print("配置详情:")
    for key, value in agent.get_config().items():
        print(f"  {key}: {value}")

    # 简单测试：随机运行几步验证流程
    print("\n--- 快速验证（随机策略运行 5 步）---")
    obs, info = env.reset()
    print(f"初始状态形状: {obs.shape}")
    print(f"初始状态: {obs[:8]}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i + 1}: action={action}, reward={reward:.2f}")
        if terminated:
            break

    print("\n验证通过！可调用 agent.train() 开始正式训练。")
