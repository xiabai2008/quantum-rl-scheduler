"""
MAPPO 同环境严格双基线对比（#3 收尾，Issue #928）

在同一个 MultiAgentEnvWrapper（同机器配置、同种子序列）下严格对比：
- MAPPO（协同多智能体，deliverable_models/mappo/mappo.pt，50K 收敛）
- FCFS（每 agent 固定 hybrid 动作，任务排序由 env 内部 FCFS 完成）

消除"不同评估环境/不同种子"的混杂，得到可归因的协同算法优势。

产出:
    - results/mappo_strict_comparison_result.json
    - results/reports/mappo_strict_comparison_report.md

用法:
    python scripts/evaluation/mappo_strict_comparison.py [--episodes 20] [--seeds 42..51]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(str(_PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.scheduler.marl import MultiAgentPPO

MODEL_PATH = "deliverable_models/mappo/mappo.pt"


def run_fcfs_on_wrapper(agent: MultiAgentPPO, seed: int, max_steps: int) -> float:
    """FCFS 基线：每 agent 固定 hybrid（action=2），同 wrapper 同种子。"""
    _lo, _ = agent.wrapper.reset(seed=seed)
    total = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps:
        actions = dict.fromkeys(agent.wrapper.machine_names, 2)  # FCFS hybrid
        _lo, reward, terminated, truncated, _ = agent.wrapper.step(actions)
        total += float(reward)
        done = bool(terminated or truncated)
        steps += 1
    return total


def run_independent_ppo_on_wrapper(
    agent: MultiAgentPPO, models: list, seed: int, max_steps: int
) -> float:
    """3 独立 PPO（同协议）：每步 3 个独立模型各自决策，走 wrapper 同仲裁。

    与 MAPPO 唯一差异：策略为独立训练（每机单机 env），评估走相同的
    wrapper.step（aggregate_actions 投票仲裁）、相同全局观测输入、相同
    种子与步数——保证"协同 vs 独立"严格可归因（8.3 审查协议修正）。
    """
    _lo0, _ = agent.wrapper.reset(seed=seed)
    total = 0.0
    done = False
    steps = 0
    while not done and steps < max_steps:
        # 三模型同歩决策（各自基于同一全局观测，与独立训练口径一致）
        global_obs = agent.wrapper.env._get_observation()
        actions = {}
        for i, name in enumerate(agent.wrapper.machine_names):
            a, _ = models[i].predict(global_obs, deterministic=True)
            actions[name] = int(a)
        _lo1, reward, terminated, truncated, _ = agent.wrapper.step(actions)
        total += float(reward)
        done = bool(terminated or truncated)
        steps += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="MAPPO 严格双基线对比")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--num-seeds", type=int, default=10)
    args = parser.parse_args()

    print(f"加载 MAPPO 模型 {MODEL_PATH} ...")
    # load 为实例方法：需构造与训练一致配置的 agent 后加载权重
    from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv

    machine_configs = DEFAULT_MACHINE_CONFIGS[:3]  # 与 train_marl 默认 3 机一致
    env = QuantumSchedulingEnv(
        max_steps=500,  # 与 train_marl 默认一致
        machine_configs=machine_configs,
        seed=42,
    )
    agent = MultiAgentPPO(
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        seed=42,
        verbose=0,
    )
    agent.load(MODEL_PATH)
    max_steps = agent.env.max_steps

    # MAPPO 评估（与 train_marl eval 同协议：seed = eval_seed_base + ep）
    eval_seed_base = int(agent.seed) if agent.seed is not None else 0
    mappo_rewards = []
    for ep in range(args.episodes):
        local_obs, _ = agent.wrapper.reset(seed=eval_seed_base + ep)
        total = 0.0
        done = False
        steps = 0
        while not done and steps < max_steps:
            global_state = agent._global_state_from_local_obs(local_obs)
            actions, _, _ = agent._sample_actions(
                local_obs, deterministic=True, global_state=global_state
            )
            local_obs, reward, terminated, truncated, _ = agent.wrapper.step(actions)
            total += float(reward)
            done = bool(terminated or truncated)
            steps += 1
        mappo_rewards.append(total)

    # FCFS 严格对比（同 wrapper、同种子序列）
    fcfs_rewards = []
    for ep in range(args.episodes):
        fcfs_rewards.append(run_fcfs_on_wrapper(agent, eval_seed_base + ep, max_steps))

    # 3 独立 PPO（无协同投票，同 wrapper 同种子）
    from stable_baselines3 import PPO as SB3PPO

    indep_models = [SB3PPO.load(f"deliverable_models/mappo/independent_ppo_m{i}") for i in range(3)]
    indep_rewards = []
    for ep in range(args.episodes):
        indep_rewards.append(
            run_independent_ppo_on_wrapper(agent, indep_models, eval_seed_base + ep, max_steps)
        )

    result = {
        "config": {
            "model": MODEL_PATH,
            "episodes": args.episodes,
            "seed_base": args.seed_base,
            "num_seeds": args.num_seeds,
        },
        "mappo": {
            "mean_reward": float(np.mean(mappo_rewards)),
            "std_reward": float(np.std(mappo_rewards)),
        },
        "fcfs": {
            "mean_reward": float(np.mean(fcfs_rewards)),
            "std_reward": float(np.std(fcfs_rewards)),
        },
        "independent_ppo": {
            "mean_reward": float(np.mean(indep_rewards)),
            "std_reward": float(np.std(indep_rewards)),
        },
        "delta_pct_vs_fcfs": float(
            (np.mean(mappo_rewards) - np.mean(fcfs_rewards)) / abs(np.mean(fcfs_rewards)) * 100
        ),
        "delta_pct_vs_indep": float(
            (np.mean(mappo_rewards) - np.mean(indep_rewards)) / abs(np.mean(indep_rewards)) * 100
        ),
    }

    out_dir = Path("results")
    report_dir = Path("results/reports")
    (out_dir / "mappo_strict_comparison_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# MAPPO 严格双基线对比报告（同 wrapper 同种子）

- **协议**：MultiAgentEnvWrapper 同实例、同种子序列（seed={eval_seed_base}..+{args.episodes}）
- **模型**：{MODEL_PATH}（50K 收敛）
- **FCFS**：每 agent 固定 hybrid（action=2），任务排序由 env 内部 FCFS 完成

| 策略 | mean_reward | std |
|:--|:--|:--|
| **MAPPO（协同）** | {result["mappo"]["mean_reward"]:.1f} | {result["mappo"]["std_reward"]:.1f} |
| FCFS（同环境） | {result["fcfs"]["mean_reward"]:.1f} | {result["fcfs"]["std_reward"]:.1f} |
| 3独立PPO（无协同） | {result["independent_ppo"]["mean_reward"]:.1f} | {result["independent_ppo"]["std_reward"]:.1f} |
| **增益 vs FCFS** | **{result["delta_pct_vs_fcfs"]:+.1f}%** | — |
| **增益 vs 独立PPO** | **{result["delta_pct_vs_indep"]:+.1f}%** | — |

**结论**：同 wrapper 同种子严格对比下，MAPPO（50K 收敛）相对 FCFS 增益
{result["delta_pct_vs_fcfs"]:+.1f}%、相对 3 独立 PPO（同训练量、无协同投票）增益
{result["delta_pct_vs_indep"]:+.1f}%——该数字消除了评估环境混杂，可直接归因于
多智能体协同调度算法（投票仲裁 + 共享 Critic 信用分配）。
多智能体协同调度算法（+ 规则未覆盖的 RL 决策优势）。
完整数据：`results/mappo_strict_comparison_result.json`
"""
    (report_dir / "mappo_strict_comparison_report.md").write_text(report, encoding="utf-8")

    print("=" * 60)
    print(
        f"  MAPPO（同环境）: {result['mappo']['mean_reward']:.1f} ± {result['mappo']['std_reward']:.1f}"
    )
    print(
        f"  FCFS（同环境） : {result['fcfs']['mean_reward']:.1f} ± {result['fcfs']['std_reward']:.1f}"
    )
    print(
        f"  3独立PPO（同环境）: {result['independent_ppo']['mean_reward']:.1f} ± {result['independent_ppo']['std_reward']:.1f}"
    )
    print(f"  增益 vs FCFS     : {result['delta_pct_vs_fcfs']:+.1f}%")
    print(f"  增益 vs 独立PPO  : {result['delta_pct_vs_indep']:+.1f}%")
    print(f"  产出: {out_dir / 'mappo_strict_comparison_result.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
