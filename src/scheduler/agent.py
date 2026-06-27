"""
RL 调度智能体模块
Reinforcement Learning Agent for Quantum-Classical Hybrid Task Scheduling

基于 Stable-Baselines3 的 DQN (Deep Q-Network) 智能体，用于量子-经典混合计算
任务调度决策。支持 Dueling DQN 架构、Epsilon-Greedy 探索策略以及 TensorBoard
训练可视化。

状态空间（8维，对应 env.py 的 SchedulingEnv）：
    0 - qubit_availability  : 当前可用量子比特比率（0-1）
    1 - queue_length         : 当前任务队列长度（归一化 0-1）
    2 - avg_wait_time        : 队列中任务平均等待时间（归一化）
    3 - fidelity             : 当前量子比特平均保真度（0-1）
    4 - classical_load       : 经典计算资源负载（0-1）
    5 - quantum_queue_ratio  : 量子专用队列占比（0-1）
    6 - time_of_day          : 一天中的时间段（0-1，模拟昼夜负载差异）
    7 - urgency_level        : 当前任务的紧急程度（0-1）

动作空间（Discrete(3)）：
    0 - 分配到经典计算资源
    1 - 分配到量子计算资源
    2 - 混合执行（量子-经典协同）
"""

import os
import numpy as np
from typing import Dict, Tuple, Optional, Any

from stable_baselines3 import DQN
from stable_baselines3.dqn import MlpPolicy
from stable_baselines3.dqn.policies import QNetwork
from stable_baselines3.common.callbacks import (
    BaseCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    create_mlp,
)
from stable_baselines3 import PPO
from src.quantum.annealing import QuantumAnnealingOptimizer
import gymnasium as gym
from gymnasium import spaces
import torch
from torch import nn

import torch as th


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
        self.q_net = nn.Sequential(*create_mlp(
            features_dim, shared_output_dim, self.net_arch[:-1], self.activation_fn
        ))

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
        value = self.value_stream(shared)          # (batch, 1)
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
    DEFAULT_LEARNING_RATE: float = 0.001          # 学习率
    DEFAULT_BUFFER_SIZE: int = 10000              # 经验回放缓冲区大小
    DEFAULT_BATCH_SIZE: int = 64                  # 训练批量大小
    DEFAULT_GAMMA: float = 0.99                  # 折扣因子（长期回报权重）
    DEFAULT_TARGET_UPDATE_INTERVAL: int = 500    # 目标网络更新间隔（步数）
    DEFAULT_TRAIN_FREQ: Tuple[int, str] = (1, "step")  # 训练频率：每步一次
    DEFAULT_EPSILON_START: float = 1.0            # 初始探索率（完全探索）
    DEFAULT_EPSILON_END: float = 0.05             # 最终探索率（保持少量探索）
    DEFAULT_EPSILON_DECAY: float = 0.995          # 探索率衰减系数
    DEFAULT_LEARNING_STARTS: int = 100            # 开始训练前的随机探索步数
    DEFAULT_TAU: float = 1.0                      # 目标网络软更新系数（1.0 = 硬更新）
    DEFAULT_LOG_DIR: str = "./logs/"              # TensorBoard 日志目录
    DEFAULT_VERBOSE: int = 1                     # 训练日志详细程度

    # 策略网络隐藏层架构
    NET_ARCH: list = [128, 64]

    def __init__(
        self,
        env: gym.Env,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        batch_size: int = DEFAULT_BATCH_SIZE,
        gamma: float = DEFAULT_GAMMA,
        target_update_interval: int = DEFAULT_TARGET_UPDATE_INTERVAL,
        train_freq: Tuple[int, str] = DEFAULT_TRAIN_FREQ,
        epsilon_start: float = DEFAULT_EPSILON_START,
        epsilon_end: float = DEFAULT_EPSILON_END,
        epsilon_decay: float = DEFAULT_EPSILON_DECAY,
        learning_starts: int = DEFAULT_LEARNING_STARTS,
        tau: float = DEFAULT_TAU,
        log_dir: str = DEFAULT_LOG_DIR,
        verbose: int = DEFAULT_VERBOSE,
        seed: Optional[int] = None,
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
        self.model: Optional[DQN] = None

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
            policy="MlpPolicy",          # 使用 MLP 策略（将通过 policy_kwargs 替换为 Dueling）
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
            exploration_fraction=0.5,    # 探索占总训练步数的比例
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
        log_dir: Optional[str] = None,
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
            best_model_save_path=os.path.join(
                self.log_dir, "best_model"
            ),
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
            raise RuntimeError(
                "模型尚未训练！请先调用 train() 方法或使用 load() 加载已训练模型。"
            )

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
    ) -> Dict[str, float]:
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
            raise RuntimeError(
                "模型尚未训练！请先调用 train() 方法或使用 load() 加载已训练模型。"
            )

        episode_rewards = []
        episode_success_rates = []

        for ep in range(num_episodes):
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
        print(f"[SchedulerAgent] 模型已保存至: {path}.zip")

    def load(self, path: str) -> None:
        """
        从文件加载已训练的模型

        加载之前保存的模型参数、优化器状态和探索率。
        加载后可直接用于 predict() 或 evaluate()。

        Args:
            path: 模型文件路径（SB3 会自动处理 .zip 扩展名）
        """
        from stable_baselines3.common.save_util import load_from_zip_file
        data, params, _ = load_from_zip_file(path, device="cpu")
        
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
            dueling_keys = [k for k in policy_state.keys() if "value_stream" in k or "advantage_stream" in k]
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
        print(f"[SchedulerAgent] 模型已从 {path} 加载")

    def get_config(self) -> Dict[str, Any]:
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
    """

    def __init__(self, optimizer, interval=1000, verbose=0):
        super().__init__(verbose)
        self.optimizer = optimizer
        self.interval = interval
        self.best_reward = -float("inf")
        self.optimized_count = 0

    def _on_step(self) -> bool:
        """每步检查是否需要触发退火优化。"""
        if self.n_calls % self.interval == 0 and self.n_calls > 0:
            try:
                # 获取当前网络权重 → QUBO → 退火 → 优化后权重
                optimized = self.optimizer.optimize_policy(self.model)

                if optimized and optimized.get("quality", 0) > self.best_reward:
                    self.best_reward = optimized["quality"]
                    self.optimized_count += 1

                    if self.verbose:
                        print(f"[退火] 步数{self.n_calls}: 优化完成 (质量={optimized['quality']:.4f}, "
                              f"累计优化{self.optimized_count}次)")
            except Exception as e:
                if self.verbose:
                    print(f"[退火] 步数{self.n_calls}: 退火跳过 ({e})")
        return True


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
        self.seed = kwargs.get("seed", None)
        self.log_dir = kwargs.get("log_dir", "./logs/")

        os.makedirs(self.log_dir, exist_ok=True)

        self.observation_space = env.observation_space
        self.action_space = env.action_space

        # 量子退火器初始化（可选功能）
        self.use_annealing = kwargs.get("use_annealing", False)
        self.annealing_optimizer = None
        self.anneal_interval = kwargs.get("anneal_interval", 1000)
        if self.use_annealing:
            self.annealing_optimizer = QuantumAnnealingOptimizer(
                num_qubits=kwargs.get("anneal_qubits", 10),
                annealing_time=kwargs.get("annealing_time", 20.0),
                shots=kwargs.get("anneal_shots", 1000),
            )
            print(f"[PPOAgent] 量子退火器已启用，退火间隔={self.anneal_interval}步")

    def _build_model(self) -> PPO:
        """
        构建 PPO 模型。

        Returns:
            构建好的 PPO 模型实例
        """
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
            policy_kwargs={"net_arch": [128, 64]},
        )
        return model

    def train(
        self,
        total_timesteps: int = 50000,
        eval_freq: int = 5000,
        n_eval_episodes: int = 10,
        log_dir: Optional[str] = None,
        **kwargs,
    ) -> PPO:
        """
        训练 PPO 调度智能体。

        Args:
            total_timesteps: 总训练步数
            eval_freq: 评估频率
            n_eval_episodes: 每次评估的回合数
            log_dir: 日志目录
            **kwargs: 额外参数

        Returns:
            训练好的 PPO 模型
        """
        if self.model is None:
            self.model = self._build_model()

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
            raise RuntimeError(
                "模型尚未训练！请先调用 train() 方法或使用 load() 加载已训练模型。"
            )

        if state.ndim == 1:
            state = state.reshape(1, -1)

        action, _ = self.model.predict(state, deterministic=deterministic)
        return int(action.item())

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
    ) -> Dict[str, float]:
        """
        评估训练好的智能体性能。

        Args:
            num_episodes: 评估回合数
            deterministic: 是否使用确定性策略

        Returns:
            评估结果字典
        """
        if self.model is None:
            raise RuntimeError(
                "模型尚未训练！请先调用 train() 方法或使用 load() 加载已训练模型。"
            )

        episode_rewards = []
        episode_success_rates = []

        for ep in range(num_episodes):
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
        print(f"[PPOAgent] 模型已保存至: {path}.zip")

    def load(self, path: str) -> None:
        """
        从文件加载已训练的模型。

        Args:
            path: 模型文件路径
        """
        self.model = PPO.load(path, env=self.env)
        print(f"[PPOAgent] 模型已从 {path} 加载")

    def get_config(self) -> Dict[str, Any]:
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
    import os

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
