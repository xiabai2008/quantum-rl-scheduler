"""
异步量子退火闭环训练脚本

演示如何将 PPO 训练与 AsyncAnnealingLoop / AsyncAnnealingCallback 结合，
实现 "训练 → 异步退火优化 → 回写权重 → 继续训练" 的全自动闭环。
"""

import argparse
import os
import sys

from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.monitor import Monitor

# 允许从项目根目录导入 src 包
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.quantum.annealing import QuantumAnnealingOptimizer
from src.quantum.annealing_loop import AsyncAnnealingLoop
from src.scheduler.agent import PPOAgent
from src.scheduler.async_annealing_callback import AsyncAnnealingCallback
from src.scheduler.env import QuantumSchedulingEnv


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="使用异步量子退火闭环训练 PPO 调度智能体")
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=20000,
        help="总训练步数，默认 20000",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=5000,
        help="评估频率（步数），默认 5000",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="每次评估的回合数，默认 5",
    )
    parser.add_argument(
        "--anneal-interval",
        type=int,
        default=5000,
        help="初始退火触发间隔（步数），默认 5000",
    )
    parser.add_argument(
        "--anneal-simulation",
        action="store_true",
        default=True,
        help="使用模拟退火（默认 True）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子，默认 42",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs/annealing_loop",
        help="TensorBoard 日志目录",
    )
    parser.add_argument(
        "--model-save-path",
        type=str,
        default="./models/ppo_async_annealing",
        help="模型保存路径",
    )
    return parser.parse_args()


def main() -> None:
    """主训练流程。"""
    args = parse_args()

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.model_save_path) or ".", exist_ok=True)

    # 创建训练环境与验证环境
    train_env = QuantumSchedulingEnv(seed=args.seed)
    eval_env = Monitor(QuantumSchedulingEnv(seed=args.seed + 1))
    validation_env = QuantumSchedulingEnv(seed=args.seed + 2)

    # 创建 PPO 智能体（不直接调用 agent.train，以便自定义回调）
    agent = PPOAgent(
        train_env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        seed=args.seed,
        log_dir=args.log_dir,
    )

    # 手动构建模型，方便挂载自定义回调
    agent.model = agent._build_model()

    # 创建量子退火优化器
    optimizer = QuantumAnnealingOptimizer(
        num_qubits=16,
        annealing_time=20.0,
        shots=500,
        simulation_mode=args.anneal_simulation,
    )

    # 创建异步退火闭环
    loop = AsyncAnnealingLoop(
        optimizer=optimizer,
        validation_env=validation_env,
        eval_episodes=3,
        eval_deterministic=True,
        initial_interval=args.anneal_interval,
        min_interval=1000,
        max_interval=20000,
        improvement_threshold=0.0,
        log_path="results/annealing_loop_log.json",
    )

    # 评估回调：周期性保存最优模型
    eval_callback = EvalCallback(
        eval_env=eval_env,
        best_model_save_path=os.path.join(args.log_dir, "best_model"),
        log_path=os.path.join(args.log_dir, "eval_results"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
    )

    # 异步退火回调
    async_annealing_callback = AsyncAnnealingCallback(loop, verbose=1)

    callback = CallbackList([eval_callback, async_annealing_callback])

    print("=" * 60)
    print("开始异步量子退火闭环训练")
    print(f"  总步数: {args.total_timesteps}")
    print(f"  初始退火间隔: {args.anneal_interval}")
    print(f"  退火模式: {'仿真' if args.anneal_simulation else '真机'}")
    print(f"  日志目录: {args.log_dir}")
    print("=" * 60)

    agent.model.learn(
        total_timesteps=args.total_timesteps,
        callback=callback,
        tb_log_name="ppo_async_annealing",
        reset_num_timesteps=True,
    )

    # 保存最终模型
    agent.save(args.model_save_path)
    print(f"训练完成，模型已保存至 {args.model_save_path}.zip")
    print(f"退火效果日志: {loop.log_path}")


if __name__ == "__main__":
    main()
