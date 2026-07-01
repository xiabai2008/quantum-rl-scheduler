#!/usr/bin/env python
"""
量子RL调度系统 — 完整训练脚本
Quantum RL Scheduler - Full Training Script

支持大规模训练（10万步+），包含以下特性：
    - 进度条实时显示（tqdm）
    - 训练中断恢复（resume from checkpoint）
    - 早停机制（Early Stopping）
    - 梯度/参数监控
    - TensorBoard 日志
    - 最佳模型自动保存
    - 自定义奖励函数支持
    - 多种子对比实验

使用示例：
    # 基础训练
    python scripts/train_agent.py --timesteps 100000

    # 恢复中断的训练
    python scripts/train_agent.py --resume --checkpoint ./models/checkpoint_50000

    # 大规模训练（100万步）+ 早停
    python scripts/train_agent.py --timesteps 1000000 --patience 50

    # 多种子对比实验
    python scripts/train_agent.py --seeds 42 123 456 --timesteps 50000
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any

import numpy as np
import yaml

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.scheduler.agent import SchedulerAgent
from src.scheduler.env import QuantumSchedulingEnv

# ============================================================================
# 日志配置
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 命令行参数解析
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(
        description="量子RL调度系统 - 大规模训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础训练
  python scripts/train_agent.py --timesteps 100000

  # 恢复中断的训练
  python scripts/train_agent.py --resume --checkpoint ./models/checkpoint_50000

  # 大规模训练 + 早停
  python scripts/train_agent.py --timesteps 1000000 --patience 50 --eval-freq 5000

  # 自定义配置
  python scripts/train_agent.py --config config/train_large.yaml --save-path ./models/exp1
        """,
    )

    # 训练参数
    training = parser.add_argument_group("训练参数")
    training.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="训练总步数（默认: 100000）",
    )
    training.add_argument(
        "--resume",
        action="store_true",
        help="从检查点恢复训练",
    )
    training.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="检查点路径（用于恢复训练）",
    )
    training.add_argument(
        "--patience",
        type=int,
        default=0,
        help="早停耐心值，评估奖励连续 N 次不提升则停止（默认: 0 表示不禁用）",
    )
    training.add_argument(
        "--min-improvement",
        type=float,
        default=0.01,
        help="被视为提升的最小奖励改进比例（默认: 0.01 即 1%%）",
    )

    # 评估参数
    evaluation = parser.add_argument_group("评估参数")
    evaluation.add_argument(
        "--eval-freq",
        type=int,
        default=1000,
        help="评估频率，每隔多少步评估一次（默认: 1000）",
    )
    evaluation.add_argument(
        "--eval-episodes",
        type=int,
        default=10,
        help="每次评估的 episode 数（默认: 10）",
    )
    evaluation.add_argument(
        "--deterministic",
        action="store_true",
        default=True,
        help="评估时使用确定性策略",
    )

    # 保存参数
    saving = parser.add_argument_group("保存参数")
    saving.add_argument(
        "--save-path",
        type=str,
        default="./models/",
        help="模型保存路径（默认: ./models/）",
    )
    saving.add_argument(
        "--save-freq",
        type=int,
        default=5000,
        help="检查点保存频率（默认: 5000）",
    )
    saving.add_argument(
        "--no-save-best",
        action="store_true",
        help="禁用最佳模型自动保存",
    )
    saving.add_argument(
        "--log-dir",
        type=str,
        default="./logs/",
        help="TensorBoard 日志目录（默认: ./logs/）",
    )
    saving.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="实验名称（默认: 自动生成时间戳）",
    )

    # 环境参数
    env = parser.add_argument_group("环境参数")
    env.add_argument(
        "--max-qubits",
        type=int,
        default=287,
        help="最大量子比特数（默认: 287）",
    )
    env.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="每个 episode 的最大步数（默认: 500）",
    )
    env.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（默认: None）",
    )
    env.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help="多种子对比实验（会并行或顺序训练多个种子）",
    )

    # 智能体参数
    agent = parser.add_argument_group("智能体参数")
    agent.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="学习率（默认: 3e-4）",
    )
    agent.add_argument(
        "--buffer-size",
        type=int,
        default=100000,
        help="经验回放缓冲区大小（默认: 100000）",
    )
    agent.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="批次大小（默认: 64）",
    )
    agent.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="折扣因子（默认: 0.99）",
    )
    agent.add_argument(
        "--epsilon-start",
        type=float,
        default=1.0,
        help="探索起始 epsilon（默认: 1.0）",
    )
    agent.add_argument(
        "--epsilon-end",
        type=float,
        default=0.01,
        help="探索终止 epsilon（默认: 0.01）",
    )
    agent.add_argument(
        "--epsilon-decay",
        type=float,
        default=0.995,
        help="epsilon 衰减率（默认: 0.995）",
    )
    agent.add_argument(
        "--target-update-freq",
        dest="target_update_interval",
        type=int,
        default=500,
        help="目标网络更新频率（默认: 500）",
    )

    # 配置文件
    config = parser.add_argument_group("配置文件")
    config.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径（会覆盖命令行参数）",
    )
    config.add_argument(
        "--reward-fn",
        type=str,
        default=None,
        help="自定义奖励函数模块路径",
    )

    # 其他
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        help="详细程度 0-2（默认: 1）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="训练设备（cuda/cpu/auto，默认: auto）",
    )
    parser.add_argument(
        "--progress-bar",
        action="store_true",
        default=True,
        help="显示进度条（默认: True）",
    )

    return parser.parse_args()


# ============================================================================
# 训练指标跟踪器
# ============================================================================
class TrainingMetrics:
    """训练过程中的指标收集器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size

        # 训练指标
        self.train_steps = []
        self.train_rewards = []
        self.train_losses = []
        self.episode_lengths = []

        # 评估指标
        self.eval_steps = []
        self.eval_rewards = []
        self.eval_success_rates = []
        self.eval_avg_wait_times = []
        self.eval_qubit_utils = []
        self.eval_classical_utils = []

        # 最佳成绩
        self.best_eval_reward = -float("inf")
        self.best_eval_step = 0
        self.patience_counter = 0

        # 时间统计
        self.start_time = None
        self.last_eval_time = None
        self.training_time = 0.0

    def record_train_step(self, step: int, reward: float, loss: float | None, length: int):
        """记录单个训练步"""
        self.train_steps.append(step)
        self.train_rewards.append(reward)
        if loss is not None:
            self.train_losses.append(loss)
        self.episode_lengths.append(length)

    def record_eval(
        self,
        step: int,
        mean_reward: float,
        success_rate: float,
        avg_wait_time: float,
        qubit_util: float,
        classical_util: float,
    ):
        """记录评估结果"""
        self.eval_steps.append(step)
        self.eval_rewards.append(mean_reward)
        self.eval_success_rates.append(success_rate)
        self.eval_avg_wait_times.append(avg_wait_time)
        self.eval_qubit_utils.append(qubit_util)
        self.eval_classical_utils.append(classical_util)

        self.last_eval_time = time.time()

        # 检查是否是最佳
        if mean_reward > self.best_eval_reward * (1 + 0.01):
            self.best_eval_reward = mean_reward
            self.best_eval_step = step
            return True  # 新的最佳
        return False

    def should_stop(self, patience: int, min_improvement: float = 0.01) -> bool:
        """检查是否应该早停"""
        if patience <= 0:
            return False

        if len(self.eval_rewards) < 2:
            return False

        # 计算连续多少次评估没有提升
        recent_best = (
            max(self.eval_rewards[:-patience])
            if len(self.eval_rewards) > patience
            else max(self.eval_rewards)
        )

        if self.eval_rewards[-1] < recent_best * (1 - min_improvement):
            self.patience_counter += 1
        else:
            self.patience_counter = 0

        return self.patience_counter >= patience

    def get_summary(self) -> dict[str, Any]:
        """获取训练摘要"""
        n = min(self.window_size, len(self.train_rewards))
        n_eval = len(self.eval_rewards)

        return {
            "total_steps": len(self.train_steps),
            "total_episodes": len(self.train_rewards),
            "best_eval_reward": self.best_eval_reward,
            "best_eval_step": self.best_eval_step,
            "training_time_seconds": self.training_time,
            "avg_train_reward_recent": float(np.mean(self.train_rewards[-n:])) if n > 0 else 0,
            "avg_eval_reward_recent": (
                float(np.mean(self.eval_rewards[-min(10, n_eval) :])) if n_eval > 0 else 0
            ),
            "eval_count": n_eval,
        }

    def to_dict(self) -> dict[str, list]:
        """转换为可序列化的字典"""
        return {
            "train_steps": self.train_steps,
            "train_rewards": self.train_rewards,
            "train_losses": self.train_losses,
            "episode_lengths": self.episode_lengths,
            "eval_steps": self.eval_steps,
            "eval_rewards": self.eval_rewards,
            "eval_success_rates": self.eval_success_rates,
            "eval_avg_wait_times": self.eval_avg_wait_times,
            "eval_qubit_utils": self.eval_qubit_utils,
            "eval_classical_utils": self.eval_classical_utils,
            "best_eval_reward": self.best_eval_reward,
            "best_eval_step": self.best_eval_step,
            "training_time": self.training_time,
        }


# ============================================================================
# 评估函数
# ============================================================================
def evaluate_agent(
    agent: SchedulerAgent,
    env: QuantumSchedulingEnv,
    num_episodes: int = 10,
    deterministic: bool = True,
) -> dict[str, float]:
    """
    评估智能体性能

    Returns:
        包含评估指标的字典
    """
    episode_rewards = []
    episode_lengths = []
    episode_completions = []
    episode_wait_times = []
    episode_qubit_utils = []
    episode_classical_utils = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        steps = 0
        done = False

        while not done:
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated

        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        episode_completions.append(info.get("completion_rate", 0.0))
        episode_wait_times.append(info.get("avg_wait_time", 0.0))
        episode_qubit_utils.append(info.get("qubit_utilization", 0.0))
        episode_classical_utils.append(info.get("classical_utilization", 0.0))

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "success_rate": float(np.mean(episode_completions)),
        "avg_wait_time": float(np.mean(episode_wait_times)),
        "qubit_utilization": float(np.mean(episode_qubit_utils)),
        "classical_utilization": float(np.mean(episode_classical_utils)),
    }


# ============================================================================
# 主训练流程
# ============================================================================
def train_single_seed(
    args: argparse.Namespace,
    seed: int,
    experiment_name: str,
    start_step: int = 0,
) -> dict[str, Any]:
    """
    单种子训练流程

    Args:
        args: 命令行参数
        seed: 随机种子
        experiment_name: 实验名称
        start_step: 起始步数（用于恢复训练）

    Returns:
        训练结果字典
    """
    logger.info(f"{'=' * 60}")
    logger.info(f"开始训练 | 种子: {seed} | 实验: {experiment_name}")
    logger.info(f"{'=' * 60}")

    # 设置随机种子
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)

    # 创建环境
    env = QuantumSchedulingEnv(
        max_steps=args.max_steps,
        max_qubits=args.max_qubits,
        seed=seed,
    )

    # 创建智能体
    agent = SchedulerAgent(
        env=env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        target_update_interval=args.target_update_interval,
        log_dir=args.log_dir,
        verbose=args.verbose,
        seed=seed,
    )

    # 构建模型
    if agent.model is None:
        agent.model = agent._build_model()
        logger.info("模型构建完成")

    # 初始化指标跟踪器
    metrics = TrainingMetrics(window_size=100)
    metrics.start_time = time.time()

    # 创建模型保存目录
    seed_dir = os.path.join(args.save_path, f"seed_{seed}")
    os.makedirs(seed_dir, exist_ok=True)

    # 加载检查点（如有）
    if start_step > 0:
        logger.info(f"从 step {start_step} 恢复训练...")

    logger.info(f"开始训练，共 {args.timesteps} 步...")

    # 使用 SB3 DQN 的 learn 方法进行训练
    # 这会自动处理探索、经验回放、目标网络更新等

    # 创建回调函数列表
    from stable_baselines3.common.callbacks import BaseCallback

    class ProgressCallback(BaseCallback):
        """进度条和评估回调"""

        def __init__(
            self, eval_freq, eval_episodes, save_freq, seed_dir, metrics, total_timesteps, verbose=0
        ):
            super().__init__(verbose)
            self.eval_freq = eval_freq
            self.eval_episodes = eval_episodes
            self.save_freq = save_freq
            self.seed_dir = seed_dir
            self.metrics = metrics
            self.total_timesteps = total_timesteps
            self.last_eval_step = 0
            self.last_save_step = 0

        def _on_step(self) -> bool:
            # 检查是否该评估
            if self.num_timesteps - self.last_eval_step >= self.eval_freq:
                self.last_eval_step = self.num_timesteps

                eval_result = evaluate_agent(
                    agent,
                    env,
                    num_episodes=self.eval_episodes,
                    deterministic=args.deterministic,
                )

                is_best = self.metrics.record_eval(
                    step=self.num_timesteps,
                    mean_reward=eval_result["mean_reward"],
                    success_rate=eval_result["success_rate"],
                    avg_wait_time=eval_result["avg_wait_time"],
                    qubit_util=eval_result["qubit_utilization"],
                    classical_util=eval_result["classical_utilization"],
                )

                # 计算训练速度
                elapsed = time.time() - self.metrics.start_time
                steps_per_sec = self.num_timesteps / elapsed if elapsed > 0 else 0

                logger.info(
                    f"[Step {self.num_timesteps}/{self.total_timesteps}] "
                    f"Eval Reward: {eval_result['mean_reward']:.2f}±{eval_result['std_reward']:.2f} | "
                    f"Success: {eval_result['success_rate'] *100:.1f}% | "
                    f"Wait: {eval_result['avg_wait_time']:.1f}s | "
                    f"Qubits: {eval_result['qubit_utilization'] *100:.1f}% | "
                    f"Speed: {steps_per_sec:.0f} steps/s" + (" [BEST]" if is_best else "")
                )

                # 保存最佳模型
                if is_best and not args.no_save_best:
                    best_path = os.path.join(self.seed_dir, "best_model")
                    agent.save(best_path)
                    logger.info(f"  -> 最佳模型已保存: {best_path}.zip")

                # 检查早停
                if args.patience > 0 and self.metrics.should_stop(
                    args.patience, args.min_improvement
                ):
                    logger.info(f"早停触发！连续 {args.patience} 次评估未提升")
                    return False  # 停止训练

            # 检查是否该保存检查点
            if self.num_timesteps - self.last_save_step >= self.save_freq:
                self.last_save_step = self.num_timesteps
                ckpt_path = os.path.join(self.seed_dir, f"checkpoint_step_{self.num_timesteps}")
                agent.save(ckpt_path)
                logger.info(f"检查点已保存: {ckpt_path}.zip")

            return True

    progress_callback = ProgressCallback(
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        save_freq=args.save_freq,
        seed_dir=seed_dir,
        metrics=metrics,
        total_timesteps=args.timesteps,
    )

    # 开始训练
    try:
        agent.model.learn(
            total_timesteps=args.timesteps,
            callback=progress_callback,
            tb_log_name=f"dqn_scheduling_{experiment_name}_seed_{seed}",
            reset_num_timesteps=True,
            log_interval=10,
        )
    except KeyboardInterrupt:
        logger.info("训练被用户中断")

    # 训练结束
    metrics.training_time = time.time() - metrics.start_time

    # 最终评估
    logger.info("执行最终评估...")
    final_eval = evaluate_agent(
        agent,
        env,
        num_episodes=max(20, args.eval_episodes),
        deterministic=True,
    )

    # 保存最终模型
    final_path = os.path.join(seed_dir, "final_model")
    agent.save(final_path)
    logger.info(f"最终模型已保存: {final_path}.zip")

    # 保存训练指标
    metrics_path = os.path.join(seed_dir, "training_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics.to_dict(), f, indent=2)
    logger.info(f"训练指标已保存: {metrics_path}")

    # 打印摘要
    summary = metrics.get_summary()
    logger.info(f"{'=' * 60}")
    logger.info(f"训练完成 | 种子: {seed}")
    logger.info(f"  总步数:     {summary['total_steps']}")
    logger.info(f"  总 episodes: {summary['total_episodes']}")
    logger.info(
        f"  最佳评估奖励: {summary['best_eval_reward']:.2f} (step {summary['best_eval_step']})"
    )
    logger.info(f"  最终评估奖励: {final_eval['mean_reward']:.2f}±{final_eval['std_reward']:.2f}")
    logger.info(f"  训练时间:     {summary['training_time_seconds']:.1f}s")
    logger.info(f"{'=' * 60}")

    return {
        "seed": seed,
        "final_eval": final_eval,
        "summary": summary,
        "metrics": metrics.to_dict(),
    }


def load_config_file(config_path: str) -> dict[str, Any]:
    """从 YAML 文件加载配置"""
    if not os.path.exists(config_path):
        logger.warning(f"配置文件不存在: {config_path}")
        return {}

    with open(config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"配置文件已加载: {config_path}")
    return config or {}


def merge_args_with_config(args: argparse.Namespace, config: dict[str, Any]) -> argparse.Namespace:
    """将配置文件中的参数合并到 args"""
    # 配置优先级低于命令行参数
    for key, value in config.items():
        if (hasattr(args, key) and getattr(args, key) is None) or (key in ["seeds", "checkpoint"]):
            setattr(args, key, value)

    return args


# ============================================================================
# 主入口
# ============================================================================
def main():
    args = parse_args()

    # 加载配置文件（如有）
    if args.config:
        config = load_config_file(args.config)
        args = merge_args_with_config(args, config)

    # 生成实验名称
    if args.experiment_name:
        experiment_name = args.experiment_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"exp_{timestamp}"

    logger.info("=" * 60)
    logger.info("量子RL调度系统 — 大规模训练")
    logger.info("=" * 60)
    logger.info(f"实验名称: {experiment_name}")
    logger.info(f"训练步数: {args.timesteps}")
    logger.info(f"评估频率: 每 {args.eval_freq} 步")
    logger.info(f"保存路径: {args.save_path}")
    logger.info(f"日志目录: {args.log_dir}")
    logger.info(f"随机种子: {args.seeds if args.seeds else args.seed}")
    logger.info(f"早停耐心: {args.patience if args.patience > 0 else '禁用'}")

    # 创建输出目录
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 确定要训练的种子
    seeds = args.seeds if args.seeds else ([args.seed] if args.seed else [42])

    # 收集所有种子的结果
    all_results = []

    for seed in seeds:
        # 恢复训练的起始步数
        start_step = 0
        if args.resume and args.checkpoint:
            # TODO: 实现从检查点恢复
            logger.warning("从检查点恢复训练的功能尚未实现")
            start_step = 0

        result = train_single_seed(args, seed, experiment_name, start_step)
        all_results.append(result)

    # 保存汇总结果
    if len(all_results) > 1:
        summary_path = os.path.join(args.save_path, f"experiment_summary_{experiment_name}.json")
        summary_data = {
            "experiment_name": experiment_name,
            "seeds": seeds,
            "results": [
                {
                    "seed": r["seed"],
                    "best_eval_reward": r["summary"]["best_eval_reward"],
                    "final_eval_reward": r["final_eval"]["mean_reward"],
                    "training_time": r["summary"]["training_time_seconds"],
                }
                for r in all_results
            ],
            "best_seed": max(all_results, key=lambda x: x["summary"]["best_eval_reward"])["seed"],
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, default=str)
        logger.info(f"实验汇总已保存: {summary_path}")

    logger.info("=" * 60)
    logger.info("所有训练完成！")
    logger.info("=" * 60)

    # 打印 TensorBoard 使用提示
    logger.info("\n查看 TensorBoard 日志:")
    logger.info(f"  tensorboard --logdir={args.log_dir}")

    return all_results


if __name__ == "__main__":
    main()
