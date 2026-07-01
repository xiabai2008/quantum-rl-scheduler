"""
多机器调度演示脚本 — Multi-Machine Scheduling Demo

在天衍云多台超导量子计算机之间做智能调度，验证 RL 智能体（PPO）在多机器
场景下的负载均衡与质量择优能力。

两种运行模式：
    1. 纯仿真模式（默认）：不消耗真机机时，对比单机 vs 多机调度效果
    2. 真机验证模式（--real）：以小概率将量子任务真正提交到天衍云真机，
       验证调度决策的可执行性

用法示例：
    # 纯仿真对比（默认，安全）
    python scripts/demo_multi_machine.py --episodes 20

    # 加载训练好的 PPO 模型做多机器调度
    python scripts/demo_multi_machine.py --ppo-model models/ppo_seed_42_v4/best_model.zip

    # 真机验证模式（5% 量子任务上真机）
    python scripts/demo_multi_machine.py --real --real-prob 0.05 --episodes 5
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# 确保从项目根目录运行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.scheduler.env import (
    DEFAULT_MACHINE_CONFIGS,
    QuantumSchedulingEnv,
)

# ---------------------------------------------------------------------------
# 策略函数
# ---------------------------------------------------------------------------

def ppo_policy_factory(model_path: str, env):
    """加载 PPO 模型构造策略函数；失败返回 None。"""
    try:
        from src.scheduler.agent import PPOAgent
        agent = PPOAgent(env)
        agent.load(model_path)
        print(f"[Policy] PPO 模型已加载: {model_path}")
        return lambda obs: agent.predict(obs, deterministic=True)
    except Exception as e:
        print(f"[Policy] PPO 加载失败 ({e})，退化为启发式策略")
        return None


def heuristic_policy(obs: np.ndarray, info: dict) -> int:
    """启发式调度策略（PPO 不可用时的兜底）。

    规则：量子任务→量子资源(1)，经典任务→经典资源(0)，universal→混合(2)。
    """
    task = info.get("current_task")
    if task is None:
        return 1
    ttype = task["task_type"]
    if ttype == "quantum":
        return 1
    elif ttype == "classical":
        return 0
    else:
        return 2


# ---------------------------------------------------------------------------
# 单 episode 运行
# ---------------------------------------------------------------------------

def run_episode(
    env: QuantumSchedulingEnv,
    policy,
    seed: int,
    use_ppo: bool,
) -> dict:
    """运行一个 episode 并返回统计结果。"""
    obs, info = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0
    done = False

    while not done:
        if use_ppo:
            action = policy(obs)
        else:
            action = heuristic_policy(obs, info)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or truncated

    return {
        "reward": total_reward,
        "steps": steps,
        "quantum_success": info["quantum_success"],
        "hybrid_success": info["hybrid_success"],
        "classical_success": info["classical_success"],
        "mismatch": info["mismatch_count"],
        "machine_schedule_count": dict(info["machine_schedule_count"]),
        "machine_real_submits": dict(info["machine_real_submits"]),
        "machines": info["machines"],
    }


def run_multi_episode(
    env_configs: list[dict],
    policy_factory,
    episodes: int,
    base_seed: int,
    label: str,
    real_clients: dict | None = None,
) -> dict:
    """运行多 episode 并聚合统计。

    Args:
        env_configs   : 环境配置列表（取第一项作为模板）
        policy_factory: PPO 策略工厂（None 用启发式）
        episodes      : 运行 episode 数
        base_seed     : 基础种子
        label         : 场景标签
        real_clients  : 真机客户端映射（可选，注入到多机器 env）
    """
    print(f"\n{'='*60}")
    print(f"  场景: {label}")
    print(f"  episodes={episodes} 机器数={len(env_configs[0].get('machine_configs') or [])}")
    print(f"{'='*60}")

    # 用第一个 env 构造策略（PPO 需要绑定 env）
    first_env = QuantumSchedulingEnv(**env_configs[0])
    if real_clients and env_configs[0].get("machine_configs") is not None:
        first_env.attach_real_clients(real_clients)
    ppo_policy = policy_factory(first_env) if policy_factory else None
    use_ppo = ppo_policy is not None
    print(f"[Policy] 使用策略: {'PPO' if use_ppo else '启发式'}")

    results = []
    for ep in range(episodes):
        env = QuantumSchedulingEnv(**env_configs[0])
        if real_clients and env_configs[0].get("machine_configs") is not None:
            env.attach_real_clients(real_clients)
        res = run_episode(env, ppo_policy, seed=base_seed + ep, use_ppo=use_ppo)
        results.append(res)
        if (ep + 1) % max(1, episodes // 5) == 0:
            print(f"  episode {ep+1}/{episodes} reward={res['reward']:.1f}")

    # 聚合
    rewards = [r["reward"] for r in results]
    agg_machine = {}
    for r in results:
        for name, cnt in r["machine_schedule_count"].items():
            agg_machine[name] = agg_machine.get(name, 0) + cnt
    agg_real = {}
    for r in results:
        for name, cnt in r["machine_real_submits"].items():
            agg_real[name] = agg_real.get(name, 0) + cnt

    summary = {
        "label": label,
        "num_machines": len(env_configs[0].get("machine_configs") or []),
        "episodes": episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_quantum_success": float(np.mean([r["quantum_success"] for r in results])),
        "mean_hybrid_success": float(np.mean([r["hybrid_success"] for r in results])),
        "mean_classical_success": float(np.mean([r["classical_success"] for r in results])),
        "mean_mismatch": float(np.mean([r["mismatch"] for r in results])),
        "machine_schedule_total": agg_machine,
        "machine_real_submits_total": agg_real,
        "policy": "PPO" if use_ppo else "heuristic",
    }
    print(f"\n  结果: mean_reward={summary['mean_reward']:.2f}"
          f" ± {summary['std_reward']:.2f}")
    print(f"  量子成功={summary['mean_quantum_success']:.1f}"
          f" 混合={summary['mean_hybrid_success']:.1f}"
          f" 经典={summary['mean_classical_success']:.1f}"
          f" 不兼容={summary['mean_mismatch']:.1f}")
    if agg_machine:
        print(f"  机器调度分布: {agg_machine}")
    if agg_real and sum(agg_real.values()) > 0:
        print(f"  真机提交: {agg_real}")
    return summary


# ---------------------------------------------------------------------------
# 报告生成
# ---------------------------------------------------------------------------

def generate_report(
    single_summary: dict,
    multi_summary: dict,
    real_mode: bool,
    real_prob: float,
) -> str:
    """生成 Markdown 对比报告。"""
    lines = [
        "# 多机器调度演示报告",
        f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**策略**: {multi_summary['policy']}",
        f"**真机验证模式**: {'是 (prob=' + str(real_prob) + ')' if real_mode else '否（纯仿真）'}",
        "",
        "## 一、单机 vs 多机调度对比",
        "",
        "| 指标 | 单机模式 | 多机模式 | 变化 |",
        "|------|---------|---------|------|",
    ]

    def _row(name, s, m, fmt="{:.2f}", better_higher=True):
        delta = m - s
        if delta == 0:
            chg = "—"
        else:
            sign = "+" if delta > 0 else ""
            chg = f"{sign}{fmt.format(delta)}"
        lines.append(
            f"| {name} | {fmt.format(s)} | {fmt.format(m)} | {chg} |"
        )

    _row("平均奖励", single_summary["mean_reward"], multi_summary["mean_reward"])
    _row("奖励标准差", single_summary["std_reward"], multi_summary["std_reward"],
         fmt="{:.2f}", better_higher=False)
    _row("平均量子成功数", single_summary["mean_quantum_success"],
         multi_summary["mean_quantum_success"], fmt="{:.1f}")
    _row("平均混合成功数", single_summary["mean_hybrid_success"],
         multi_summary["mean_hybrid_success"], fmt="{:.1f}")
    _row("平均经典成功数", single_summary["mean_classical_success"],
         multi_summary["mean_classical_success"], fmt="{:.1f}")
    _row("平均不兼容数", single_summary["mean_mismatch"],
         multi_summary["mean_mismatch"], fmt="{:.1f}", better_higher=False)

    # 多机器负载均衡
    lines.append("")
    lines.append("## 二、多机器负载均衡")
    lines.append("")
    lines.append("| 机器 | 总调度数 | 真机提交数 |")
    lines.append("|------|---------|-----------|")
    total_sched = sum(multi_summary["machine_schedule_total"].values()) or 1
    for name, cnt in multi_summary["machine_schedule_total"].items():
        real = multi_summary["machine_real_submits_total"].get(name, 0)
        pct = cnt / total_sched * 100
        lines.append(f"| {name} | {cnt} ({pct:.1f}%) | {real} |")

    # 负载均衡度（变异系数 CV，越低越均衡）
    counts = list(multi_summary["machine_schedule_total"].values())
    if len(counts) > 1 and np.mean(counts) > 0:
        cv = float(np.std(counts) / np.mean(counts))
        lines.append("")
        lines.append(f"- 负载变异系数 (CV): **{cv:.3f}**（越低越均衡，0=完美均衡）")
        if cv < 0.15:
            quality = "🟢 优秀 — 多机器负载高度均衡"
        elif cv < 0.3:
            quality = "🟡 良好 — 负载较为均衡"
        else:
            quality = "🔴 待优化 — 负载分布不均"
        lines.append(f"- 均衡评估: {quality}")

    # 结论
    lines.append("")
    lines.append("## 三、结论")
    lines.append("")
    reward_delta = multi_summary["mean_reward"] - single_summary["mean_reward"]
    if reward_delta > 0:
        lines.append(f"- 多机器调度平均奖励 **提升 {reward_delta:.2f}**，"
                     f"证实多机器纳管能提升整体调度吞吐")
    else:
        lines.append(f"- 多机器调度平均奖励变化 {reward_delta:+.2f}，"
                     f"主要价值在负载分摊与容错而非单 episode 奖励")
    lines.append(f"- 调度策略: {multi_summary['policy']}")
    lines.append(f"- 纳管机器: {multi_summary['num_machines']} 台")
    if real_mode and sum(multi_summary["machine_real_submits_total"].values()) > 0:
        lines.append(f"- 真机验证: 已提交 {sum(multi_summary['machine_real_submits_total'].values())} 个任务到天衍云真机")
    lines.append("")
    lines.append("---")
    lines.append("*报告自动生成 | 数据来源: src/scheduler/env.py 多机器调度扩展*")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="多机器调度演示")
    parser.add_argument("--episodes", type=int, default=20,
                        help="每个场景运行的 episode 数（默认 20）")
    parser.add_argument("--max-steps", type=int, default=300,
                        help="每个 episode 最大步数（默认 300）")
    parser.add_argument("--ppo-model", type=str,
                        default="models/ppo_seed_42_v4/best_model.zip",
                        help="PPO 模型路径（不存在则用启发式策略）")
    parser.add_argument("--real", action="store_true",
                        help="启用真机验证模式（需配置 TIANYAN_API_KEY）")
    parser.add_argument("--real-prob", type=float, default=0.05,
                        help="真机提交抽样概率（默认 0.05）")
    parser.add_argument("--seed", type=int, default=42, help="基础随机种子")
    parser.add_argument("--output", type=str,
                        default="results/multi_machine_demo_report.md",
                        help="报告输出路径")
    args = parser.parse_args()

    print("=" * 60)
    print("  多机器调度演示 — Multi-Machine Scheduling Demo")
    print("=" * 60)
    print(f"  episodes={args.episodes}  max_steps={args.max_steps}")
    print(f"  PPO 模型: {args.ppo_model}")
    print(f"  真机模式: {'是 (prob=' + str(args.real_prob) + ')' if args.real else '否'}")

    # 真机模式：加载 API Key 并构造客户端映射
    real_clients = {}
    if args.real:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.getenv("TIANYAN_API_KEY", "")
        if not api_key:
            print("[警告] 未设置 TIANYAN_API_KEY，真机模式退化为纯仿真")
            args.real = False
        else:
            from src.api.tianyan_cqlib import create_multi_machine_clients
            machine_names = [c["name"] for c in DEFAULT_MACHINE_CONFIGS]
            try:
                real_clients = create_multi_machine_clients(api_key, machine_names)
                print(f"[真机] 已创建 {len(real_clients)} 个真机客户端: {machine_names}")
            except Exception as e:
                print(f"[警告] 真机客户端创建失败 ({e})，退化为纯仿真")
                args.real = False

    # PPO 策略工厂
    ppo_path = args.ppo_model
    if not os.path.exists(ppo_path):
        print(f"[提示] PPO 模型不存在: {ppo_path}，将使用启发式策略")
        ppo_path = None
    policy_factory = (
        (lambda env: ppo_policy_factory(ppo_path, env)) if ppo_path else None
    )

    # ---- 场景 1：单机基线 ----
    single_configs = [{
        "max_steps": args.max_steps,
        "machine_configs": None,  # None = 单机兼容模式
        "real_submit_probability": 0.0,
    }]
    single_summary = run_multi_episode(
        single_configs, policy_factory, args.episodes, args.seed, "单机基线 (tianyan_s)"
    )

    # ---- 场景 2：多机器调度 ----
    multi_configs = [{
        "max_steps": args.max_steps,
        "machine_configs": DEFAULT_MACHINE_CONFIGS,
        "real_submit_probability": args.real_prob if args.real else 0.0,
    }]
    multi_summary = run_multi_episode(
        multi_configs, policy_factory, args.episodes, args.seed + 100,
        f"多机器调度 ({len(DEFAULT_MACHINE_CONFIGS)} 台)",
        real_clients=real_clients if args.real else None,
    )

    # ---- 生成报告 ----
    report = generate_report(single_summary, multi_summary, args.real, args.real_prob)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n{'='*60}")
    print(f"✅ 报告已保存: {args.output}")
    print(f"{'='*60}")
    print("\n报告预览:")
    print(report[:800])


if __name__ == "__main__":
    main()
