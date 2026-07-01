"""
量子退火 + PPO 联调测试

验证量子退火优化器能够正确接入 PPO 训练循环。
"""

import os
import sys

# 必须在导入 annealing 模块之前设置，因为 QUANTUM_ACCELERATION_ENABLED
# 是在模块顶层读取的全局常量
os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.scheduler.agent import PPOAgent
from src.scheduler.env import QuantumSchedulingEnv


def test_annealing_ppo():
    """测试量子退火与 PPO 的联调"""
    print("=" * 60)
    print("量子退火 + PPO 联调测试")
    print("=" * 60)

    # 创建环境
    env = QuantumSchedulingEnv(max_steps=100, seed=42)
    print(f"[环境] 状态维度: {env.observation_space.shape}")
    print(f"[环境] 动作空间: {env.action_space.n}")

    # 创建带退火的 PPO
    print("\n[创建] PPOAgent + 量子退火...")
    agent = PPOAgent(
        env,
        use_annealing=True,
        anneal_interval=500,
        anneal_qubits=16,
        simulation_mode=True,
        verbose=1,
        n_steps=256,
        batch_size=64,
    )
    print(f"[PPOAgent] 配置: {agent.get_config()}")
    print(f"[PPOAgent] 退火器: {agent.annealing_optimizer}")
    print(f"[PPOAgent] 退火间隔: {agent.anneal_interval}")

    # 训练
    print("\n[训练] 开始训练 (3000步)...")
    model = agent.train(
        total_timesteps=3000,
        eval_freq=1000,
        n_eval_episodes=3,
    )
    print("[训练] 训练完成!")

    # 验证
    print("\n[验证] 测试预测...")
    obs = env.reset()[0]
    action, _ = model.predict(obs, deterministic=True)
    print(f"[验证] 预测动作: {action}")

    # 运行几个 episode 验证
    print("\n[验证] 运行 3 个 episode...")
    total_rewards = []
    for ep in range(3):
        obs = env.reset()[0]
        total_reward = 0
        done = False
        steps = 0
        while not done and steps < 50:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            done = terminated or truncated
            steps += 1
        total_rewards.append(total_reward)
        print(f"  Episode {ep + 1}: reward={total_reward:.2f}, steps={steps}")

    avg_reward = sum(total_rewards) / len(total_rewards)
    print(f"\n[验证] 平均奖励: {avg_reward:.2f}")

    print("\n" + "=" * 60)
    print("[PASS] 退火 + PPO 联调测试通过!")
    print("=" * 60)

    return True


if __name__ == "__main__":
    try:
        test_annealing_ppo()
    except Exception as e:
        print(f"\n[FAIL] 测试失败: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
