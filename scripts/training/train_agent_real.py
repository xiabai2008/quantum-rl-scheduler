"""
真机闭环训练脚本 — PPO + cqlib Real-Machine Closed-Loop Training

将 cqlib 真机客户端注入 PPO 调度循环，实现“真机任务提交 → 结果轮询 →
奖励反馈”的完整闭环（Issue #64）。

核心特性：
    1. 真机客户端注入：从 TIANYAN_API_KEY 创建多机器 cqlib 客户端，绑定到环境
    2. 非阻塞提交：env.step() 中以 real_submit_probability 概率提交真机
    3. 结果轮询反馈：后续 step() 轮询已提交任务的结果，成功/失败叠加到 reward
    4. 自动降级：真机连续失败超过阈值时自动 fallback 到 Mock，保证训练不中断
    5. 对比报告：纯仿真 vs 真机混合训练的性能对比（收敛速度、最终 reward）

用法示例：

    # 纯仿真训练（无 API Key 时自动退化为该模式）
    python scripts/training/train_agent_real.py --timesteps 5000 --episodes 5

    # 真机混合训练（5% 量子任务上真机）
    python scripts/training/train_agent_real.py --timesteps 10000 --real-prob 0.05

    # 仅生成对比报告（已训练过两个模型）
    python scripts/training/train_agent_real.py --report-only \
        --sim-model models/ppo_sim/best_model.zip \
        --real-model models/ppo_real/best_model.zip
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# 确保从项目根目录运行
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.scheduler.agent import PPOAgent
from src.scheduler.env import (
    DEFAULT_MACHINE_CONFIGS,
    QuantumSchedulingEnv,
)

# ---------------------------------------------------------------------------
# 真机客户端构造
# ---------------------------------------------------------------------------


def build_real_clients(api_key: str) -> dict | None:
    """构造多机器真机客户端映射。

    Args:
        api_key: 天衍云 API Key

    Returns:
        {machine_name: CqlibTianyanClient} 映射；构造失败返回 None
    """
    try:
        from src.api.tianyan_cqlib import create_multi_machine_clients

        machine_names = [c["name"] for c in DEFAULT_MACHINE_CONFIGS]
        clients = create_multi_machine_clients(api_key, machine_names)
        print(f"[真机] 已创建 {len(clients)} 个真机客户端: {machine_names}")
        return clients
    except Exception as e:
        print(f"[警告] 真机客户端创建失败 ({e})，将退化为纯仿真训练")
        return None


# ---------------------------------------------------------------------------
# 训练单个 PPO 模型
# ---------------------------------------------------------------------------


def train_one_model(
    label: str,
    total_timesteps: int,
    real_clients: dict | None,
    real_prob: float,
    real_callback_interval: int,
    seed: int = 42,
    save_dir: str = "models/",
) -> dict:
    """训练一个 PPO 模型并返回训练统计。

    Args:
        label                 : 模型标签（sim / real）
        total_timesteps       : 总训练步数
        real_clients          : 真机客户端映射（None 表示纯仿真）
        real_prob             : 真机提交抽样概率
        real_callback_interval: 真机回调触发间隔（步数，0=禁用）
        seed                  : 随机种子
        save_dir              : 模型保存目录

    Returns:
        训练统计字典，包含 label/timesteps/real_submits/degraded/elapsed_s/
        model_path 等字段
    """
    print(f"\n{'=' *60}")
    print(f"  训练场景: {label}")
    print(f"  timesteps={total_timesteps} real_prob={real_prob} seed={seed}")
    print(f"{'=' *60}")

    # 构造环境：真机模式需要 use_real_machine=True + 多机器配置 + 真机客户端
    use_real = real_clients is not None and real_prob > 0.0
    env = QuantumSchedulingEnv(
        max_steps=300,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        real_submit_probability=real_prob if use_real else 0.0,
        use_real_machine=use_real,
        real_machine_feedback_weight=1.0,
        seed=seed,
    )
    if use_real and real_clients is not None:
        env.attach_real_clients(real_clients)
        print("[真机闭环] 已启用真机闭环模式（use_real_machine=True）")
    else:
        print("[纯仿真] 真机未启用，使用纯仿真训练")

    # 构造 PPO 智能体
    agent = PPOAgent(
        env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=64,
        gamma=0.99,
        verbose=1,
        seed=seed,
        log_dir=f"./logs/ppo_{label}/",
    )

    # 训练（真机回调仅在 use_real 时启用）
    t0 = time.time()
    real_cb_interval = real_callback_interval if use_real else 0
    agent.train(
        total_timesteps=total_timesteps,
        eval_freq=max(1000, total_timesteps // 5),
        n_eval_episodes=3,
        real_callback_interval=real_cb_interval,
        real_callback_prob=0.5,  # 回调触发时 50% 概率提交
        real_callback_save_path=f"results/real_times_{label}.json",
        real_callback_shots=512,
    )
    elapsed = time.time() - t0

    # 保存模型
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"ppo_{label}_seed_{seed}")
    agent.save(model_path)

    # 收集真机统计
    real_stats = env.get_real_machine_stats() if hasattr(env, "get_real_machine_stats") else {}
    real_submits_total = sum(env._machine_real_submits.values()) if use_real else 0

    stats = {
        "label": label,
        "timesteps": total_timesteps,
        "elapsed_s": round(elapsed, 2),
        "use_real_machine": use_real,
        "real_submit_probability": real_prob if use_real else 0.0,
        "real_submits_total": int(real_submits_total),
        "real_success_count": int(real_stats.get("success_count", 0)),
        "real_fail_count": int(real_stats.get("fail_count", 0)),
        "real_degraded": bool(real_stats.get("degraded", False)),
        "model_path": model_path + ".zip",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    print(f"\n[{label}] 训练完成: 耗时={elapsed:.1f}s 真机提交={real_submits_total}")
    print(f"  真机成功={stats['real_success_count']} 失败={stats['real_fail_count']}"
          f" 降级={stats['real_degraded']}")
    return stats


# ---------------------------------------------------------------------------
# 评估模型（多 episode 平均 reward）
# ---------------------------------------------------------------------------


def evaluate_model(
    model_path: str,
    episodes: int,
    real_clients: dict | None = None,
    real_prob: float = 0.0,
    seed: int = 42,
) -> dict:
    """加载模型并评估，返回平均 reward 和调度统计。

    Args:
        model_path : PPO 模型路径
        episodes   : 评估 episode 数
        real_clients: 真机客户端（可选，用于真机验证评估）
        real_prob  : 真机提交概率（评估时通常设为 0）
        seed       : 基础随机种子

    Returns:
        评估结果字典
    """
    use_real = real_clients is not None and real_prob > 0.0
    env = QuantumSchedulingEnv(
        max_steps=300,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        real_submit_probability=real_prob,
        use_real_machine=use_real,
        seed=seed,
    )
    if use_real and real_clients is not None:
        env.attach_real_clients(real_clients)

    agent = PPOAgent(env, verbose=0)
    try:
        agent.load(model_path)
    except Exception as e:
        print(f"[评估] 模型加载失败 {model_path}: {e}")
        return {"mean_reward": 0.0, "std_reward": 0.0, "episodes": episodes, "error": str(e)}

    rewards = []
    for ep in range(episodes):
        obs, _info = env.reset(seed=seed + ep)
        total_reward = 0.0
        done = False
        while not done:
            action = agent.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        rewards.append(total_reward)
        if (ep + 1) % max(1, episodes // 5) == 0:
            print(f"  [{os.path.basename(model_path)}] episode {ep+1}/{episodes}"
                  f" reward={total_reward:.1f}")

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "episodes": episodes,
    }


# ---------------------------------------------------------------------------
# 对比报告生成
# ---------------------------------------------------------------------------


def generate_comparison_report(
    sim_stats: dict,
    real_stats: dict,
    sim_eval: dict,
    real_eval: dict,
    output_path: str,
) -> str:
    """生成纯仿真 vs 真机混合训练的 Markdown 对比报告。

    Args:
        sim_stats  : 纯仿真训练统计
        real_stats : 真机混合训练统计
        sim_eval   : 纯仿真模型评估结果
        real_eval  : 真机混合模型评估结果
        output_path: 报告输出路径

    Returns:
        报告 Markdown 文本
    """
    lines = [
        "# PPO 真机闭环训练对比报告",
        f"\n**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "**Issue**: #64 PPO 真机闭环对接",
        "",
        "## 一、训练配置对比",
        "",
        "| 配置项 | 纯仿真训练 | 真机混合训练 |",
        "|--------|------------|------------|",
        f"| 训练步数 | {sim_stats['timesteps']} | {real_stats['timesteps']} |",
        "| 真机模式 | 否 | 是 |",
        f"| 真机抽样概率 | 0.0 | {real_stats['real_submit_probability']} |",
        f"| 训练耗时(s) | {sim_stats['elapsed_s']} | {real_stats['elapsed_s']} |",
        "",
        "## 二、真机闭环统计",
        "",
        "| 指标 | 数值 |",
        "|------|------|",
        f"| 真机提交总数 | {real_stats['real_submits_total']} |",
        f"| 真机成功数 | {real_stats['real_success_count']} |",
        f"| 真机失败数 | {real_stats['real_fail_count']} |",
        f"| 真机降级 | {'是' if real_stats['real_degraded'] else '否'} |",
        "",
        "## 三、模型性能对比",
        "",
        "| 指标 | 纯仿真模型 | 真机混合模型 | 变化 |",
        "|------|-----------|------------|------|",
    ]

    sim_r = sim_eval.get("mean_reward", 0.0)
    real_r = real_eval.get("mean_reward", 0.0)
    delta = real_r - sim_r
    sign = "+" if delta >= 0 else ""
    lines.append(f"| 平均奖励 | {sim_r:.2f} | {real_r:.2f} | {sign}{delta:.2f} |")
    lines.append(
        f"| 奖励标准差 | {sim_eval.get('std_reward', 0.0):.2f} | "
        f"{real_eval.get('std_reward', 0.0):.2f} | — |"
    )
    lines.append(f"| 评估 episode 数 | {sim_eval.get('episodes', 0)} | "
                 f"{real_eval.get('episodes', 0)} | — |")

    # 收敛速度对比（基于训练耗时）
    if sim_stats["elapsed_s"] > 0 and real_stats["elapsed_s"] > 0:
        time_ratio = real_stats["elapsed_s"] / sim_stats["elapsed_s"]
        lines.append(
            f"| 训练耗时比 | 1.00x | {time_ratio:.2f}x | "
            f"{'真机更慢' if time_ratio > 1 else '真机更快'} |"
        )

    # 结论
    lines.append("")
    lines.append("## 四、结论")
    lines.append("")
    if real_stats["real_submits_total"] > 0:
        success_rate = (
            real_stats["real_success_count"] / max(1, real_stats["real_submits_total"])
        )
        lines.append(
            f"- 真机闭环已启用：共提交 {real_stats['real_submits_total']} 个任务，"
            f"成功率 {success_rate:.1%}"
        )
        if real_stats["real_degraded"]:
            lines.append(
                "- ⚠️ 真机已降级到 Mock（连续失败超过阈值），建议检查 API Key 或机器状态"
            )
        else:
            lines.append("- ✅ 真机未降级，闭环运行稳定")
    else:
        lines.append("- 真机未启用或未成功提交任何任务（纯仿真模式）")

    if delta > 0:
        lines.append(f"- 真机混合训练模型奖励提升 {delta:.2f}，"
                     f"真机反馈对策略学习有正向作用")
    elif delta < 0:
        lines.append(f"- 真机混合训练模型奖励下降 {abs(delta):.2f}，"
                     f"可能是真机延迟导致训练样本减少或真机失败惩罚影响")
    else:
        lines.append("- 两种训练模式奖励无显著差异")

    lines.append("")
    lines.append("---")
    lines.append("*报告由 scripts/training/train_agent_real.py 自动生成*")

    report = "\n".join(lines)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    return report


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def main() -> None:
    """真机闭环训练主入口。"""
    parser = argparse.ArgumentParser(description="PPO 真机闭环训练（Issue #64）")
    parser.add_argument(
        "--timesteps", type=int, default=5000, help="总训练步数（默认 5000）"
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="评估 episode 数（默认 5）"
    )
    parser.add_argument(
        "--real-prob", type=float, default=0.05, help="真机提交抽样概率（默认 0.05）"
    )
    parser.add_argument(
        "--real-callback-interval",
        type=int,
        default=1000,
        help="真机回调触发间隔步数（默认 1000）",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--save-dir", type=str, default="models/", help="模型保存目录"
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="仅生成对比报告（需提供已训练的两个模型路径）",
    )
    parser.add_argument(
        "--sim-model", type=str, default="", help="纯仿真模型路径（--report-only 时使用）"
    )
    parser.add_argument(
        "--real-model", type=str, default="", help="真机混合模型路径（--report-only 时使用）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/reports/real_machine_training_comparison.md",
        help="对比报告输出路径",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  PPO 真机闭环训练 — Issue #64")
    print("=" * 60)
    print(f"  timesteps={args.timesteps} episodes={args.episodes}")
    print(f"  real_prob={args.real_prob} seed={args.seed}")

    # 加载 API Key
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("TIANYAN_API_KEY", "")
    real_clients = build_real_clients(api_key) if api_key else None
    if real_clients is None:
        print("[提示] 未配置 TIANYAN_API_KEY 或客户端创建失败，将退化为纯仿真训练")
        args.real_prob = 0.0

    # ---- 仅生成报告模式 ----
    if args.report_only:
        if not args.sim_model or not args.real_model:
            print("[错误] --report-only 模式需要同时提供 --sim-model 和 --real-model")
            sys.exit(1)
        sim_stats = {"timesteps": 0, "elapsed_s": 0.0, "real_submits_total": 0,
                     "real_success_count": 0, "real_fail_count": 0,
                     "real_degraded": False, "real_submit_probability": 0.0}
        real_stats = {"timesteps": 0, "elapsed_s": 0.0, "real_submits_total": 0,
                      "real_success_count": 0, "real_fail_count": 0,
                      "real_degraded": False, "real_submit_probability": args.real_prob}
        print("\n[评估] 纯仿真模型...")
        sim_eval = evaluate_model(args.sim_model, args.episodes, real_clients=None,
                                  real_prob=0.0, seed=args.seed)
        print("\n[评估] 真机混合模型...")
        real_eval = evaluate_model(args.real_model, args.episodes, real_clients=None,
                                   real_prob=0.0, seed=args.seed)
        report = generate_comparison_report(
            sim_stats, real_stats, sim_eval, real_eval, args.output
        )
        print(f"\n✅ 报告已保存: {args.output}")
        print("\n报告预览:")
        print(report[:800])
        return

    # ---- 完整训练流程：纯仿真 vs 真机混合 ----
    # 1. 纯仿真训练（基线）
    sim_stats = train_one_model(
        label="sim",
        total_timesteps=args.timesteps,
        real_clients=None,
        real_prob=0.0,
        real_callback_interval=0,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    # 2. 真机混合训练
    real_stats = train_one_model(
        label="real",
        total_timesteps=args.timesteps,
        real_clients=real_clients,
        real_prob=args.real_prob,
        real_callback_interval=args.real_callback_interval,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    # 3. 评估两个模型
    print(f"\n{'=' *60}")
    print("  评估阶段: 纯仿真 vs 真机混合")
    print(f"{'=' *60}")
    print("\n[评估] 纯仿真模型...")
    sim_eval = evaluate_model(
        sim_stats["model_path"], args.episodes, real_clients=None,
        real_prob=0.0, seed=args.seed + 1000,
    )
    print(f"  纯仿真平均 reward: {sim_eval['mean_reward']:.2f}"
          f" ± {sim_eval['std_reward']:.2f}")

    print("\n[评估] 真机混合模型...")
    real_eval = evaluate_model(
        real_stats["model_path"], args.episodes, real_clients=None,
        real_prob=0.0, seed=args.seed + 1000,
    )
    print(f"  真机混合平均 reward: {real_eval['mean_reward']:.2f}"
          f" ± {real_eval['std_reward']:.2f}")

    # 4. 生成对比报告
    report = generate_comparison_report(
        sim_stats, real_stats, sim_eval, real_eval, args.output
    )

    # 5. 保存训练统计 JSON
    stats_json_path = args.output.replace(".md", "_stats.json")
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"sim": sim_stats, "real": real_stats,
             "sim_eval": sim_eval, "real_eval": real_eval},
            f, ensure_ascii=False, indent=2,
        )

    print(f"\n{'=' *60}")
    print(f"✅ 对比报告已保存: {args.output}")
    print(f"✅ 训练统计已保存: {stats_json_path}")
    print(f"{'=' *60}")
    print("\n报告预览:")
    print(report[:1000])


if __name__ == "__main__":
    main()
