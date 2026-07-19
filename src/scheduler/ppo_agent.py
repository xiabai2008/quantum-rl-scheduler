"""
PPO 调度智能体模块
PPO (Proximal Policy Optimization) Agent for Quantum-Classical Scheduling

从 agent.py 拆分而来，包含 PPOAgent 类。
为保持向后兼容，agent.py 通过 __all__ 重新导出 PPOAgent。
"""

import os
from typing import Any

import numpy as np
from loguru import logger
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    CallbackList,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor

from src.quantum.annealing import QuantumAnnealingOptimizer
from src.scheduler.callbacks import (
    AnnealingCallback,
    RealMachineCallback,
)


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

    def __init__(self, env: Any, **kwargs: Any) -> None:
        """
        初始化 PPO 智能体。

        Args:
            env: Gymnasium 环境实例
            **kwargs: PPO 超参数
        """
        self.env = env
        self.model: PPO | None = None

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
                f"[PPOAgent] 量子退火器已启用（{sim_tag}模式），退火间隔={self.anneal_interval}步"
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
        resume_from: str | None = None,
        extra_callbacks: list[Any] | None = None,
        **kwargs: Any,
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
        is_resume = bool(resume_from) and os.path.exists(resume_from)
        if self.model is None:
            if is_resume:
                logger.info(f"[PPOAgent] 从检查点续训: {resume_from}")
                self.load(resume_from)
            else:
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
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        callback = CallbackList(callbacks) if len(callbacks) > 1 else callbacks[0]

        tb_log_name = log_dir if log_dir else "ppo_scheduling"

        assert self.model is not None  # 由上方 _build_model / load 保证已初始化
        # 续训时保留历史步数计数，避免指标与时间轴错乱
        reset_num_timesteps = not is_resume
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
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
        model_cls = RecurrentPPO if self.use_lstm else PPO
        self.model = model_cls.load(path, env=self.env)
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
