"""
RL智能体训练脚本
Train RL Agent Script

训练量子RL调度系统的DQN智能体，支持命令行参数配置、
定期评估、TensorBoard日志、最佳模型保存等功能。

使用示例：
    python scripts/train_agent.py --timesteps 50000 --save-path ./models/dqn_scheduler
    python scripts/train_agent.py --env config/config.yaml --eval-freq 500
"""

import os
import sys
import argparse
import numpy as np
from datetime import datetime

# 确保项目根目录在路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="量子RL调度系统 - 训练RL智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n"
               "  python scripts/train_agent.py --timesteps 50000 --save-path ./models/dqn_scheduler\n"
               "  python scripts/train_agent.py --env config/config.yaml --eval-freq 500\n",
    )
    parser.add_argument(
        "--env", type=str, default="config/config.yaml",
        help="环境配置文件路径（默认: config/config.yaml）",
    )
    parser.add_argument(
        "--timesteps", type=int, default=100000,
        help="训练总步数（默认: 100000）",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=1000,
        help="评估频率，每隔多少步评估一次（默认: 1000）",
    )
    parser.add_argument(
        "--eval-episodes", type=int, default=10,
        help="每次评估运行的回合数（默认: 10）",
    )
    parser.add_argument(
        "--save-path", type=str, default="./models/",
        help="模型保存路径（默认: ./models/）",
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs/",
        help="TensorBoard 日志目录（默认: ./logs/）",
    )
    parser.add_argument(
        "--save-freq", type=int, default=5000,
        help="定期保存频率，每隔多少步保存一次模型（默认: 5000）",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="随机种子，用于可复现实验（默认: None）",
    )
    return parser.parse_args()


def load_config(config_path: str):
    """加载YAML配置文件"""
    import yaml
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        print(f"[配置] 配置文件加载成功: {config_path}")
        return config
    except FileNotFoundError:
        print(f"[警告] 配置文件不存在: {config_path}，将使用默认配置")
        return {}
    except Exception as e:
        print(f"[错误] 配置文件加载失败: {e}，将使用默认配置")
        return {}


def build_env_from_config(config: dict):
    """根据配置创建调度环境"""
    from src.scheduler.env import SchedulingEnv

    quantum_cfg = config.get("quantum", {})
    system_cfg = config.get("system", {})

    env = SchedulingEnv(
        max_qubits=quantum_cfg.get("max_qubits", 287),
        max_queue_size=system_cfg.get("max_queue_size", 100),
        max_wait_time=float(system_cfg.get("max_wait_time", 3600)),
        simulation_mode=True,
    )
    return env


def build_agent_from_config(env, config: dict, log_dir: str, seed=None):
    """根据配置创建调度智能体"""
    from src.scheduler.agent import SchedulerAgent

    sched_cfg = config.get("scheduler", {})

    agent = SchedulerAgent(
        env=env,
        learning_rate=sched_cfg.get("learning_rate", 3e-4),
        buffer_size=sched_cfg.get("replay_buffer_size", 10000),
        batch_size=sched_cfg.get("batch_size", 64),
        gamma=sched_cfg.get("gamma", 0.99),
        epsilon_start=sched_cfg.get("epsilon_start", 1.0),
        epsilon_end=sched_cfg.get("epsilon_end", 0.01),
        epsilon_decay=sched_cfg.get("epsilon_decay", 0.995),
        log_dir=log_dir,
        verbose=1,
        seed=seed,
    )
    return agent


class TrainingMetricsTracker:
    """训练指标跟踪器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.episode_rewards = []
        self.completion_rates = []
        self.avg_wait_times = []
        self.resource_utilizations = []
        self.best_eval_reward = -float("inf")

    def record_eval(self, eval_result: dict):
        """记录一次评估结果"""
        self.episode_rewards.append(eval_result.get("mean_reward", 0.0))
        self.completion_rates.append(eval_result.get("success_rate", 0.0))
        self.avg_wait_times.append(eval_result.get("avg_wait_time", 0.0))
        self.resource_utilizations.append(eval_result.get("resource_utilization", 0.0))

    def print_summary(self, step: int, total_timesteps: int, epsilon: float = None):
        """打印当前训练指标摘要"""
        n = min(self.window_size, len(self.episode_rewards))
        if n == 0:
            return

        avg_reward = np.mean(self.episode_rewards[-n:])
        avg_completion = np.mean(self.completion_rates[-n:]) * 100
        avg_wait = np.mean(self.avg_wait_times[-n:])
        avg_util = np.mean(self.resource_utilizations[-n:]) * 100

        progress = step / total_timesteps * 100
        print(
            f"[训练] 步数: {step}/{total_timesteps} ({progress:.1f}%) | "
            f"平均奖励(近{n}): {avg_reward:.2f} | "
            f"任务完成率: {avg_completion:.1f}% | "
            f"平均等待时间: {avg_wait:.1f}s | "
            f"量子资源利用率: {avg_util:.1f}%"
            + (f" | Epsilon: {epsilon:.4f}" if epsilon is not None else "")
        )

        return avg_reward


def evaluate_agent(agent, num_episodes: int = 10, deterministic: bool = True):
    """
    评估智能体性能，返回详细指标

    Returns:
        dict: 包含 mean_reward, success_rate, avg_wait_time, resource_utilization
    """
    env = agent.env
    episode_rewards = []
    episode_completion_rates = []
    episode_wait_times = []
    episode_resource_utils = []

    for _ in range(num_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        done = False
        step_count = 0
        total_wait_time = 0.0
        total_utilization = 0.0

        while not done:
            action = agent.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step_count += 1
            done = terminated or truncated

            # 收集环境指标
            total_wait_time += info.get("avg_wait_time", 0.0)
            total_utilization += info.get("resource_utilization", 0.0)

        episode_rewards.append(total_reward)
        episode_completion_rates.append(info.get("completion_rate", 0.0))
        episode_wait_times.append(total_wait_time / max(step_count, 1))
        episode_resource_utils.append(total_utilization / max(step_count, 1))

    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "success_rate": float(np.mean(episode_completion_rates)),
        "avg_wait_time": float(np.mean(episode_wait_times)),
        "resource_utilization": float(np.mean(episode_resource_utils)),
        "num_episodes": num_episodes,
    }


def train(args):
    """主训练流程"""
    print("=" * 70)
    print("量子RL驱动的天衍云平台智能调度系统 - 智能体训练")
    print("=" * 70)

    # 1. 加载配置
    config = load_config(args.env)

    # 2. 创建环境
    print("[初始化] 正在创建调度环境...")
    env = build_env_from_config(config)
    print(f"[初始化] 环境创建成功 - 状态空间: {env.observation_space.shape}, 动作空间: {env.action_space.n}")

    # 3. 创建智能体
    print("[初始化] 正在创建DQN智能体...")
    agent = build_agent_from_config(env, config, args.log_dir, seed=args.seed)
    print(f"[初始化] 智能体创建成功\n{agent}")

    # 4. 训练参数
    total_timesteps = args.timesteps
    eval_freq = args.eval_freq
    eval_episodes = args.eval_episodes
    save_freq = args.save_freq
    save_path = args.save_path

    os.makedirs(save_path, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # 5. 构建模型
    print("\n[训练] 正在构建Dueling DQN模型...")
    agent.train.__doc__  # 触发模型构建
    if agent.model is None:
        agent._build_model()

    # 6. 设置训练回调
    from stable_baselines3.common.callbacks import BaseCallback, CallbackList

    tracker = TrainingMetricsTracker(window_size=100)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

    class EvalAndSaveCallback(BaseCallback):
        """自定义回调：定期评估 + 定期保存 + 打印指标"""

        def __init__(self, eval_freq, save_freq, save_path, tracker, eval_episodes,
                     scheduler_agent, verbose=0):
            super().__init__(verbose)
            self.eval_freq = eval_freq
            self.save_freq = save_freq
            self.save_path = save_path
            self.tracker = tracker
            self.eval_episodes = eval_episodes
            self.scheduler_agent = scheduler_agent  # SchedulerAgent 实例，用于评估

        def _on_step(self) -> bool:
            num_timesteps = self.num_timesteps

            # 定期评估
            if num_timesteps % self.eval_freq == 0 and num_timesteps > 0:
                eval_result = evaluate_agent(
                    self.scheduler_agent,  # 使用 SchedulerAgent 的 predict 方法
                    num_episodes=self.eval_episodes,
                    deterministic=True,
                )
                self.tracker.record_eval(eval_result)

                # 获取当前 epsilon
                epsilon = getattr(self.model, "exploration_rate",
                                  self.tracker.best_eval_reward and 0.0)

                avg_reward = self.tracker.print_summary(
                    step=num_timesteps,
                    total_timesteps=total_timesteps,
                    epsilon=epsilon if hasattr(self.model, 'exploration_rate') else None,
                )

                # 保存最佳模型
                if avg_reward is not None and avg_reward > self.tracker.best_eval_reward:
                    self.tracker.best_eval_reward = avg_reward
                    best_path = os.path.join(self.save_path, "best_model")
                    self.model.save(best_path)
                    print(f"[保存] 最佳模型已保存 (奖励: {avg_reward:.2f}): {best_path}.zip")

                # 记录到 TensorBoard
                self.logger.record("eval/mean_reward", eval_result["mean_reward"])
                self.logger.record("eval/success_rate", eval_result["success_rate"])
                self.logger.record("eval/avg_wait_time", eval_result["avg_wait_time"])
                self.logger.record("eval/resource_utilization", eval_result["resource_utilization"])

            # 定期保存检查点
            if num_timesteps % self.save_freq == 0 and num_timesteps > 0:
                ckpt_path = os.path.join(
                    self.save_path,
                    f"checkpoint_step_{num_timesteps}_{timestamp_str}",
                )
                self.model.save(ckpt_path)
                print(f"[保存] 检查点已保存: {ckpt_path}.zip")

            return True

    # Epsilon 探索回调
    epsilon_callback = agent.EpsilonExplorationCallback(
        epsilon_start=agent.epsilon_start,
        epsilon_end=agent.epsilon_end,
        epsilon_decay=agent.epsilon_decay,
    )

    # 评估与保存回调
    eval_save_callback = EvalAndSaveCallback(
        eval_freq=eval_freq,
        save_freq=save_freq,
        save_path=save_path,
        tracker=tracker,
        eval_episodes=eval_episodes,
        scheduler_agent=agent,
    )

    # 使用 SchedulerAgent 内部逻辑启动训练，但用自定义回调覆盖
    from stable_baselines3.common.callbacks import CheckpointCallback

    callback_list = CallbackList([epsilon_callback, eval_save_callback])

    # 7. 开始训练
    print(f"\n[训练] 开始训练，总步数: {total_timesteps}")
    print(f"[训练] 评估频率: 每 {eval_freq} 步 | 保存频率: 每 {save_freq} 步")
    print(f"[训练] 模型保存路径: {save_path}")
    print(f"[训练] TensorBoard日志: {args.log_dir}")
    print("-" * 70)

    agent.model.learn(
        total_timesteps=total_timesteps,
        callback=callback_list,
        tb_log_name=f"dqn_scheduling_{timestamp_str}",
        reset_num_timesteps=True,
    )

    # 8. 保存最终模型
    final_path = os.path.join(save_path, f"final_model_{timestamp_str}")
    agent.model.save(final_path)
    print(f"\n[保存] 最终模型已保存: {final_path}.zip")

    # 9. 最终评估
    print("\n" + "=" * 70)
    print("训练完成！最终评估结果：")
    print("=" * 70)

    final_eval = evaluate_agent(agent, num_episodes=20, deterministic=True)
    print(f"  平均奖励 (20 episodes): {final_eval['mean_reward']:.2f} +/- {final_eval['std_reward']:.2f}")
    print(f"  任务完成率:             {final_eval['success_rate'] * 100:.1f}%")
    print(f"  平均等待时间:           {final_eval['avg_wait_time']:.1f}s")
    print(f"  量子资源利用率:         {final_eval['resource_utilization'] * 100:.1f}%")
    print(f"  历史最佳评估奖励:       {tracker.best_eval_reward:.2f}")

    # 10. 保存训练摘要
    summary = {
        "timestamp": timestamp_str,
        "total_timesteps": total_timesteps,
        "eval_freq": eval_freq,
        "save_freq": save_freq,
        "config": config,
        "final_eval": final_eval,
        "best_eval_reward": tracker.best_eval_reward,
        "all_eval_rewards": tracker.episode_rewards,
        "all_completion_rates": tracker.completion_rates,
        "all_avg_wait_times": tracker.avg_wait_times,
        "all_resource_utilizations": tracker.resource_utilizations,
    }

    import json
    summary_path = os.path.join(save_path, f"training_summary_{timestamp_str}.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[保存] 训练摘要已保存: {summary_path}")

    print("=" * 70)
    print("提示: 使用以下命令查看TensorBoard日志:")
    print(f"  tensorboard --logdir={args.log_dir}")
    print("=" * 70)


def main():
    """主入口"""
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
