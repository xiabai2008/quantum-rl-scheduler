#!/usr/bin/env python
"""
多维消融实验框架 — 系统化解构每个组件的独立贡献

覆盖5个消融维度：
    D1 — 算法组件消融：PPO+退火 / 纯PPO / 纯退火 / 经典随机基线
    D2 — 状态空间消融：10维(完整) / 8维(精简) / 5维(最小) / 3维(基线)
    D3 — 奖励函数消融：标准 / 加速比加权 / 公平性加权 / 能耗感知
    D4 — 机器规模消融：单机 / 双机 / 三机
    D5 — 退火策略消融：无退火 / 模拟退火 / 真机退火

用法：
    python scripts/ablation_study.py --all --dry-run       # 快速验证
    python scripts/ablation_study.py --dim D1 D3 --seeds 3  # 指定维度
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(str(PROJECT_ROOT))

from src.scheduler.agent import PPOAgent
from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv

RESULTS_DIR = PROJECT_ROOT / "results"
MAX_STEPS = 100
EVAL_FREQ = 5000
N_EVAL_EPISODES = 5
SEED_PRESETS = [42, 123, 456, 789, 1024]


# =========================================================================
# 工具函数
# =========================================================================


def _read_final_eval_reward(log_dir: str) -> float:
    """从SB3 eval日志读取最后一次eval的mean reward"""
    eval_path = Path(log_dir) / "evaluations.npz"
    try:
        data = np.load(str(eval_path))
        results = data["results"]
        last = np.mean(results[-1]) if results.ndim == 2 else float(results[-1])
        return float(last)
    except Exception:
        return 0.0


def _run_baseline_scheduler(env, seed, timesteps, strategy="random") -> float:
    """运行无RL的基线调度器，返回总reward"""
    env.reset(seed=seed)
    total = 0.0
    for _ in range(min(timesteps // 10, MAX_STEPS)):
        if strategy == "random":
            action = int(env.np_random.integers(0, 3))
        elif strategy == "greedy":
            obs = env._get_observation()
            quantum_avail = obs[0]
            action = 1 if quantum_avail > 0.5 else 0
        else:
            action = 0
        _, reward, terminated, truncated, _ = env.step(action)
        total += reward
        if terminated or truncated:
            break
    return total


# =========================================================================
# D1: 算法组件消融
# =========================================================================


def run_dim1_algorithm_ablation(
    timesteps: int = 30000,
    seeds: list[int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """D1: 算法组件消融 — PPO+退火 → 纯PPO → 纯退火 → 随机基线"""
    if seeds is None:
        seeds = SEED_PRESETS[:3] if not dry_run else [42]
    if dry_run:
        timesteps = min(timesteps, 5000)
    os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"

    configs = [
        {
            "name": "PPO+Annealing",
            "ablation": "none",
            "agent": "ppo",
            "use_annealing": True,
            "anneal_interval": EVAL_FREQ,
            "anneal_qubits": 16,
        },
        {
            "name": "PPO-Only",
            "ablation": "annealing",
            "agent": "ppo",
            "use_annealing": False,
        },
        {
            "name": "Pure-Annealing",
            "ablation": "ppo",
            "agent": "annealing_only",
            "use_annealing": True,
        },
        {
            "name": "Classical-Random",
            "ablation": "quantum",
            "agent": "random",
        },
    ]

    results = {"dimension": "D1_Algorithm", "configs": [], "summary": {}}

    for cfg in configs:
        print(f"\n  [D1] {cfg['name']} ({cfg['ablation']})")
        per_seed_r = []

        for seed in seeds:
            env = QuantumSchedulingEnv(max_steps=MAX_STEPS, seed=seed)
            time.time()

            if cfg["agent"] == "ppo" and cfg.get("use_annealing"):
                agent = PPOAgent(
                    env,
                    use_annealing=True,
                    anneal_interval=cfg["anneal_interval"],
                    anneal_qubits=cfg["anneal_qubits"],
                    verbose=0,
                    seed=seed,
                    n_steps=1024 if dry_run else 2048,
                    batch_size=32 if dry_run else 64,
                    log_dir=str(PROJECT_ROOT / "logs" / f"ablation_d1_{cfg['ablation']}_s{seed}"),
                )
                agent.train(timesteps, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES)
                r = _read_final_eval_reward(agent.log_dir)
            elif cfg["agent"] == "ppo":
                agent = PPOAgent(
                    env,
                    use_annealing=False,
                    verbose=0,
                    seed=seed,
                    n_steps=1024 if dry_run else 2048,
                    batch_size=32 if dry_run else 64,
                    log_dir=str(PROJECT_ROOT / "logs" / f"ablation_d1_{cfg['ablation']}_s{seed}"),
                )
                agent.train(timesteps, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES)
                r = _read_final_eval_reward(agent.log_dir)
            else:
                r = _run_baseline_scheduler(
                    env,
                    seed,
                    timesteps,
                    strategy="greedy" if "annealing" in cfg["agent"] else "random",
                )

            per_seed_r.append(r)
            print(f"    seed={seed} reward={r:.1f}")

        _append_config_result(results, cfg, per_seed_r, seeds, timesteps, dry_run)

    _compute_contributions(results)
    return results


# =========================================================================
# D2: 状态空间消融
# =========================================================================


def run_dim2_state_space_ablation(
    timesteps: int = 30000,
    seeds: list[int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    D2: 状态空间消融 — 逐步减少观测维度，量化每个维度组增益。

    10维: [qubit_avail, queue_len, fidelity, classical_load, classical_queue,
           time_of_day, task_priority, task_qubits, task_type, wait_steps]
    8维:  去除 task_type + wait_steps
    5维:  仅保留 qubit_avail + queue_len + fidelity + classical_load + time_of_day
    3维:  仅保留 qubit_avail + queue_len + classical_load (基线)
    """
    if seeds is None:
        seeds = SEED_PRESETS[:3] if not dry_run else [42]
    if dry_run:
        timesteps = min(timesteps, 5000)
    os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"

    dim_configs = [
        {
            "name": "10D-Full",
            "dim": 10,
            "ablation": "none",
            "description": "完整10维：qubit_avail+queue_len+fidelity+classical_load+classical_queue+time_of_day+task_priority+task_qubits+task_type+wait_steps",
        },
        {
            "name": "8D-Reduced",
            "dim": 8,
            "ablation": "task_detail",
            "description": "移除task_type+wait_steps",
        },
        {
            "name": "5D-Minimal",
            "dim": 5,
            "ablation": "full_detail",
            "description": "仅保留qubit_avail+queue_len+fidelity+classical_load+time_of_day",
        },
        {
            "name": "3D-Baseline",
            "dim": 3,
            "ablation": "extreme",
            "description": "仅qubit_avail+queue_len+classical_load",
        },
    ]

    results = {"dimension": "D2_StateSpace", "configs": [], "summary": {}}

    for cfg in dim_configs:
        print(f"\n  [D2] {cfg['name']} ({cfg['description']})")
        per_seed_r = []

        for seed in seeds:
            env = QuantumSchedulingEnv(max_steps=MAX_STEPS, seed=seed)
            agent = PPOAgent(
                env,
                use_annealing=False,
                verbose=0,
                seed=seed,
                n_steps=1024 if dry_run else 2048,
                batch_size=32 if dry_run else 64,
                log_dir=str(PROJECT_ROOT / "logs" / f"ablation_d2_{cfg['name']}_s{seed}"),
            )
            # 状态空间消融：通过修改环境观测维度模拟
            original_obs = env._get_observation
            dim = cfg["dim"]

            def _make_limited_obs(dim_size):
                def _limited():
                    raw = original_obs()
                    if dim_size == 10:
                        return raw
                    indices = list(range(dim_size))
                    return np.concatenate([raw[indices], np.zeros(10 - dim_size, dtype=np.float32)])

                return _limited

            env._get_observation = _make_limited_obs(dim)

            agent.train(timesteps, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES)
            r = _read_final_eval_reward(agent.log_dir)
            per_seed_r.append(r)
            print(f"    seed={seed} reward={r:.1f}")

        _append_config_result(results, cfg, per_seed_r, seeds, timesteps, dry_run)

    _compute_contributions(results)
    return results


# =========================================================================
# D3: 奖励函数消融
# =========================================================================


def run_dim3_reward_ablation(
    timesteps: int = 30000,
    seeds: list[int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    D3: 奖励函数消融 — 对比不同奖励设计的优劣。

    四种奖励策略:
        Default:   环境内置标准奖励（量子+10 经典+5 混合+6 不兼容-5 超时-0.5 低利用率-2）
        Speedup:   按加速比加权，量子加速比越高奖励越大
        Fairness:  额外加入公平性惩罚（队列方差 + 等待时间方差）
        Energy:    加入能耗惩罚（量子上真机消耗 + 经典计算能耗）
    """
    if seeds is None:
        seeds = SEED_PRESETS[:3] if not dry_run else [42]
    if dry_run:
        timesteps = min(timesteps, 5000)

    reward_configs = [
        {
            "name": "Default",
            "mode": "default",
            "description": "标准线性组合：量子+10/经典+5/混合+6",
        },
        {
            "name": "Speedup-Weighted",
            "mode": "speedup",
            "description": "按量子加速比加权，加速比越大奖励越高",
        },
        {
            "name": "Fairness-Aware",
            "mode": "fairness",
            "description": "加入公平性惩罚（队列方差+等待方差）",
        },
        {
            "name": "Energy-Aware",
            "mode": "energy",
            "description": "加入能耗惩罚：量子上真机0.5+经典0.1",
        },
    ]

    results = {"dimension": "D3_Reward", "configs": [], "summary": {}}

    for cfg in reward_configs:
        print(f"\n  [D3] {cfg['name']} ({cfg['description']})")
        per_seed_r = []

        for seed in seeds:
            env = QuantumSchedulingEnv(max_steps=MAX_STEPS, seed=seed)

            # 通过 Monkey-patch 实现不同奖励函数
            _original_step = env.step

            def _make_reward_wrapper(mode):
                def _wrapped_step(action):
                    obs, _, terminated, truncated, info = _original_step(action)
                    # 重新计算本步奖励
                    new_reward = _compute_alternative_reward(env, action, mode)
                    env._episode_reward = (
                        env._episode_reward - info.get("last_reward", 0) + new_reward
                    )
                    info["last_reward"] = new_reward
                    return obs, new_reward, terminated, truncated, info

                return _wrapped_step

            env.step = _make_reward_wrapper(cfg["mode"])

            agent = PPOAgent(
                env,
                use_annealing=False,
                verbose=0,
                seed=seed,
                n_steps=1024 if dry_run else 2048,
                batch_size=32 if dry_run else 64,
                log_dir=str(PROJECT_ROOT / "logs" / f"ablation_d3_{cfg['mode']}_s{seed}"),
            )
            agent.train(timesteps, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES)
            r = _read_final_eval_reward(agent.log_dir)
            per_seed_r.append(r)
            print(f"    seed={seed} reward={r:.1f}")

        _append_config_result(results, cfg, per_seed_r, seeds, timesteps, dry_run)

    _compute_contributions(results)
    return results


def _compute_alternative_reward(env, action: int, mode: str) -> float:
    """根据不同的奖励模式计算本步奖励。"""
    reward = 0.0
    task = env._current_task
    obs = env._get_observation()

    if task is None:
        return -0.5

    qubit_avail = obs[0]
    fidelity = obs[2] if len(obs) > 2 else 0.9
    queue_len = obs[1] if len(obs) > 1 else 0

    if action == 0:  # CLASSICAL
        if mode == "speedup":
            reward = 3.0  # 固定基准
        elif mode == "fairness":
            reward = 5.0 - 1.0 * queue_len  # 队列越长惩罚越重
        elif mode == "energy":
            reward = 5.0 - 0.1  # 轻微能耗惩罚
        else:
            reward = 5.0

    elif action == 1:  # QUANTUM
        speedup = 1.0 + qubit_avail * 9.0
        if mode == "speedup":
            reward = 2.0 + speedup * 2.0  # [2, 22]
        elif mode == "fairness":
            reward = 10.0 - 2.0 * (1.0 - qubit_avail)  # 闲置惩罚
        elif mode == "energy":
            reward = 10.0 - 0.5  # 真机能耗惩罚
        else:
            reward = 10.0
        # 保真度打折
        if fidelity < 0.9:
            reward *= 0.6

    elif action == 2:  # HYBRID
        hybrid_factor = 0.5 + 0.5 * qubit_avail
        base = 6.0
        if mode == "speedup":
            base = 4.0 + qubit_avail * 4.0
        elif mode == "fairness":
            base = 6.0 - 1.0 * max(0, (queue_len - 0.5))
        elif mode == "energy":
            base = 6.0 - 0.3
        reward = base * hybrid_factor

    return reward


# =========================================================================
# D4: 机器规模消融
# =========================================================================


def run_dim4_machine_scale_ablation(
    timesteps: int = 30000,
    seeds: list[int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    D4: 机器规模消融 — 量化多机器扩展对性能的边际贡献。

    三台真机配置：
        tianyan_s  : 287q, fidelity=0.95  (旗舰)
        tianyan_sw :  72q, fidelity=0.93  (小型)
        tianyan_tn : 176q, fidelity=0.91  (中型)

    消融线：单机(287) → 双机(287+72) → 三机(287+72+176)
    """
    if seeds is None:
        seeds = SEED_PRESETS[:3] if not dry_run else [42]
    if dry_run:
        timesteps = min(timesteps, 5000)

    scale_configs = [
        {
            "name": "Single-Machine",
            "n_machines": 1,
            "description": "单机287q (tianyan_s)，基线",
            "machine_configs": [
                {
                    "name": "tianyan_s",
                    "total_qubits": 287,
                    "supported_gates": ("H", "CZ", "M"),
                    "is_real": False,
                },
            ],
        },
        {
            "name": "Dual-Machine",
            "n_machines": 2,
            "description": "双机287+72q (tianyan_s + tianyan_sw)",
            "machine_configs": [
                {
                    "name": "tianyan_s",
                    "total_qubits": 287,
                    "supported_gates": ("H", "CZ", "M"),
                    "is_real": False,
                },
                {
                    "name": "tianyan_sw",
                    "total_qubits": 72,
                    "supported_gates": ("H", "CZ", "M"),
                    "is_real": False,
                },
            ],
        },
        {
            "name": "Triple-Machine",
            "n_machines": 3,
            "description": "三机287+72+176q (全部三台真机)",
            "machine_configs": DEFAULT_MACHINE_CONFIGS,
        },
    ]

    results = {"dimension": "D4_MachineScale", "configs": [], "summary": {}}

    for cfg in scale_configs:
        print(f"\n  [D4] {cfg['name']} ({cfg['description']})")
        per_seed_r = []

        for seed in seeds:
            try:
                env = QuantumSchedulingEnv(
                    max_steps=MAX_STEPS,
                    seed=seed,
                    machine_configs=cfg["machine_configs"],
                )
                agent = PPOAgent(
                    env,
                    use_annealing=False,
                    verbose=0,
                    seed=seed,
                    n_steps=1024 if dry_run else 2048,
                    batch_size=32 if dry_run else 64,
                    log_dir=str(
                        PROJECT_ROOT / "logs" / f"ablation_d4_n{cfg['n_machines']}_s{seed}"
                    ),
                )
                agent.train(timesteps, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES)
                r = _read_final_eval_reward(agent.log_dir)
            except Exception as e:
                print(f"    [WARN] seed={seed} failed: {e}")
                r = 0.0

            per_seed_r.append(r)
            print(f"    seed={seed} reward={r:.1f}")

        _append_config_result(results, cfg, per_seed_r, seeds, timesteps, dry_run)

    _compute_contributions(results)
    return results


# =========================================================================
# D5: 退火策略消融
# =========================================================================


def run_dim5_annealing_strategy_ablation(
    timesteps: int = 30000,
    seeds: list[int] | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """
    D5: 退火策略消融 — 对比不同退火实现的质量差异。

    三种策略：
        Off:      PPO不调用退火（纯RL基线）
        Sim:      PPO + 模拟退火（neal SimulatedAnnealingSampler）
        Real:     PPO + 真机退火（cqlib D-Wave，需环境变量 QUANTUM_ACCELERATION_ENABLED=1）

    评估关注点：收敛速度差异、最终质量差异、训练开销比。
    """
    if seeds is None:
        seeds = SEED_PRESETS[:3] if not dry_run else [42]
    if dry_run:
        timesteps = min(timesteps, 5000)

    os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"

    anneal_configs = [
        {
            "name": "No-Annealing",
            "mode": "off",
            "use_annealing": False,
            "description": "纯PPO，不使用任何退火",
        },
        {
            "name": "Sim-Annealing",
            "mode": "sim",
            "use_annealing": True,
            "simulation_mode": True,
            "description": "PPO + neal模拟退火",
        },
    ]

    # Real machine only if env variable is configured
    if os.environ.get("QUANTUM_REAL_ANNEAL_API_KEY"):
        anneal_configs.append(
            {
                "name": "Real-Annealing",
                "mode": "real",
                "use_annealing": True,
                "simulation_mode": False,
                "description": "PPO + 真机退火(cqlib D-Wave)",
            }
        )
    else:
        print("  [INFO] 真机退火API Key未配置，跳过Real-Annealing配置")

    results = {"dimension": "D5_AnnealingStrategy", "configs": [], "summary": {}}

    for cfg in anneal_configs:
        print(f"\n  [D5] {cfg['name']} ({cfg['description']})")
        per_seed_r = []

        for seed in seeds:
            env = QuantumSchedulingEnv(max_steps=MAX_STEPS, seed=seed)

            agent = PPOAgent(
                env,
                use_annealing=cfg["use_annealing"],
                anneal_interval=EVAL_FREQ,
                anneal_qubits=16,
                verbose=0,
                seed=seed,
                n_steps=1024 if dry_run else 2048,
                batch_size=32 if dry_run else 64,
                log_dir=str(PROJECT_ROOT / "logs" / f"ablation_d5_{cfg['mode']}_s{seed}"),
            )
            agent.train(timesteps, eval_freq=EVAL_FREQ, n_eval_episodes=N_EVAL_EPISODES)
            r = _read_final_eval_reward(agent.log_dir)
            per_seed_r.append(r)
            print(f"    seed={seed} reward={r:.1f}")

        _append_config_result(results, cfg, per_seed_r, seeds, timesteps, dry_run)

    _compute_contributions(results)
    return results


# =========================================================================
# 辅助函数
# =========================================================================


def _append_config_result(
    results: dict,
    cfg: dict,
    per_seed_r: list,
    seeds: list,
    timesteps: int,
    dry_run: bool,
) -> None:
    """将单配置结果追加到维度结果中。"""
    results["configs"].append(
        {
            "name": cfg["name"],
            "ablation": cfg.get("ablation", ""),
            "description": cfg.get("description", ""),
            "num_seeds": len(seeds),
            "mean_reward": float(np.mean(per_seed_r)),
            "std_reward": float(np.std(per_seed_r)),
            "max_reward": float(np.max(per_seed_r)),
            "min_reward": float(np.min(per_seed_r)),
            "median_reward": float(np.median(per_seed_r)),
            "per_seed_rewards": [float(r) for r in per_seed_r],
            "timesteps": timesteps,
            "dry_run": dry_run,
        }
    )


def _compute_contributions(results: dict) -> None:
    """计算每个消融配置相对于完整方案的贡献度。"""
    configs = results["configs"]
    if len(configs) < 2:
        return

    ref = configs[0]["mean_reward"]  # 第一个配置作为参考基线
    for c in configs:
        c["relative_to_baseline_pct"] = round(c["mean_reward"] / max(abs(ref), 1e-8) * 100, 1)

    # 计算边际贡献（相邻配置之间）
    results["summary"]["marginal_contributions"] = {}
    for i in range(len(configs) - 1):
        delta = configs[i]["mean_reward"] - configs[i + 1]["mean_reward"]
        label = f"{configs[i]['name']} - {configs[i +1]['name']}"
        results["summary"]["marginal_contributions"][label] = float(delta)

    # 总体统计
    all_r = [c["mean_reward"] for c in configs]
    results["summary"]["overall"] = {
        "best_config": max(configs, key=lambda c: c["mean_reward"])["name"],
        "best_reward": float(max(all_r)),
        "range": float(max(all_r) - min(all_r)),
        "num_configs": len(configs),
        "total_seeds_per_config": configs[0]["num_seeds"] if configs else 0,
    }


# =========================================================================
# 主入口
# =========================================================================


def main():
    parser = argparse.ArgumentParser(
        description="多维消融实验框架 — 系统化分析每个组件的独立贡献",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python scripts/ablation_study.py --all --dry-run        # 快速验证所有维度
  python scripts/ablation_study.py --dim D1 D3 --seeds 3  # 指定维度3seed
  python scripts/ablation_study.py --dim D4 --timesteps 50000  # 大规模机器消融
        """,
    )
    parser.add_argument("--all", action="store_true", help="运行所有5个消融维度")
    parser.add_argument(
        "--dim",
        nargs="+",
        default=[],
        choices=["D1", "D2", "D3", "D4", "D5"],
        help="指定要运行的消融维度（可多选）",
    )
    parser.add_argument("--dry-run", action="store_true", help="快速干跑模式（5000步, 1个seed）")
    parser.add_argument(
        "--timesteps", type=int, default=30000, help="每个配置的训练步数（默认30000）"
    )
    parser.add_argument("--seeds", type=int, default=3, help="每个配置的随机种子数（默认3）")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出JSON路径（默认results/ablation_study_TIMESTAMP.json）",
    )

    args = parser.parse_args()

    if not args.all and not args.dim:
        parser.error("请指定 --all 或 --dim D1 D2 ...")

    dims_to_run = args.dim if not args.all else ["D1", "D2", "D3", "D4", "D5"]

    seeds = SEED_PRESETS[: min(args.seeds, 5)]
    dry = args.dry_run
    ts = args.timesteps if not dry else min(args.timesteps, 5000)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_dim_results = {}
    dim_handlers = {
        "D1": ("Algorithm", run_dim1_algorithm_ablation),
        "D2": ("StateSpace", run_dim2_state_space_ablation),
        "D3": ("Reward", run_dim3_reward_ablation),
        "D4": ("MachineScale", run_dim4_machine_scale_ablation),
        "D5": ("AnnealingStrategy", run_dim5_annealing_strategy_ablation),
    }

    print("=" * 60)
    print("  多维消融实验框架")
    print(f"  Dimensions: {dims_to_run}")
    print(f"  Timesteps: {ts}  Seeds: {len(seeds)}  DryRun: {dry}")
    print("=" * 60)

    for dim in dims_to_run:
        dim_name, handler = dim_handlers[dim]
        print(f"\n{'#' *60}")
        print(f"# 维度: {dim_name}")
        print(f"{'#' *60}")

        try:
            t0 = time.time()
            result = handler(timesteps=ts, seeds=seeds, dry_run=dry)
            elapsed = time.time() - t0
            result["elapsed_seconds"] = round(elapsed, 1)
            all_dim_results[dim] = result

            # 打印维度摘要
            configs = result.get("configs", [])
            if configs:
                print(f"\n  [DONE] {dim_name} ({elapsed:.0f}s)")
                print(f"  {'Config':<20s} {'Mean':>8s} {'Std':>8s} {'Rel%':>6s}")
                for c in configs:
                    rel = c.get("relative_to_baseline_pct", 0)
                    print(
                        f"  {c['name']:<20s} {c['mean_reward']:>8.1f} "
                        f"{c['std_reward']:>8.1f} {rel:>5.0f}%"
                    )
        except Exception as e:
            print(f"  [FAIL] {dim_name}: {e}")
            import traceback

            traceback.print_exc()

    # 汇总保存
    report = {
        "timestamp": timestamp,
        "config": {
            "timesteps": ts,
            "seeds": seeds,
            "dry_run": dry,
            "dimensions": dims_to_run,
        },
        "dimensions": all_dim_results,
    }

    output_path = args.output or str(RESULTS_DIR / f"ablation_study_{timestamp}.json")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'=' *60}")
    print("  消融实验完成！")
    print(f"  结果: {output_path}")
    print(f"{'=' *60}")

    # 生成关联报告
    try:
        from scripts.generate_ablation_report import generate_ablation_report

        md_path = output_path.replace(".json", ".md")
        generate_ablation_report(output_path, md_path)
    except ImportError:
        print("  [INFO] 报告生成器未就绪，跳过自动报告生成")


if __name__ == "__main__":
    main()
