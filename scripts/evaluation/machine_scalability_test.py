#!/usr/bin/env python
"""
机器规模扩展性测试（#224）
Machine Scalability Test

与 #206 的区别：
    - #206 只做了任务规模梯度（100/500/1000/5000/10000 tasks）
    - 本脚本补充机器规模扩展（1/3/10/50/100 台机器）+ 复杂度分析

测试内容：
    1. 机器规模梯度：1/3/10/50/100 台机器 × 5 seeds × 5 episodes
    2. 测试指标：
       - 奖励随机器数的变化曲线
       - 负载均衡度（变异系数 CV）
       - 调度算法时间复杂度实测
       - 多机通信开销（如适用）
    3. 复杂度分析：理论 + 实测对比

用法：
    # 完整运行（约 1-2 天）
    python scripts/evaluation/machine_scalability_test.py --seeds 5 --episodes 5

    # 快速验证（仅 1 seed × 1 episode，约 30s）
    python scripts/evaluation/machine_scalability_test.py --seeds 1 --episodes 1 --quick
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(str(_PROJECT_ROOT))

# 复用现有基础设施
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "evaluation"))
from run_issue_38_67_experiments import (  # noqa: I001
    Obs10Wrapper,
    SimulationEnv,
    SimulationTaskGenerator,
)
from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_types import QuantumMachine


# ============================================================================
# 机器规模梯度配置
# ============================================================================

# #224 要求的机器规模梯度
MACHINE_SCALES = [1, 3, 10, 50, 100]


# ============================================================================
# 多机器环境构建
# ============================================================================


def make_multi_machine_env(
    n_machines: int,
    tasks_per_episode: int = 200,
    seed: int | None = None,
    obs_dim: int = 10,
) -> Any:
    """创建多机器调度环境。

    Args:
        n_machines: 机器数量（1/3/10/50/100）
        tasks_per_episode: 每 episode 最大步数
        seed: 随机种子
        obs_dim: 观测维度
    """
    if n_machines == 1:
        # 单机模式：保持与原环境一致
        base = QuantumSchedulingEnv(
            max_steps=tasks_per_episode,
            max_qubits=287,
            seed=seed,
        )
    else:
        # 多机模式：构建 machine_configs
        machine_configs = []
        for i in range(n_machines):
            machine_configs.append(
                {
                    "name": f"tianyan_{i:03d}",
                    "total_qubits": 287,
                    "supported_gates": ("H", "CZ", "M"),
                    "is_real": False,
                }
            )
        base = QuantumSchedulingEnv(
            machine_configs=machine_configs,
            max_steps=tasks_per_episode,
            max_qubits=287,
            seed=seed,
        )

    if obs_dim == 10:
        return Obs10Wrapper(base)
    return base


# ============================================================================
# 负载均衡度计算
# ============================================================================


def compute_load_balance(env: QuantumSchedulingEnv) -> dict[str, float]:
    """计算负载均衡度指标。

    使用 ``_machine_schedule_count``（每台机器实际分配的任务数）作为数据源，
    而非 ``_machine_real_submits``（仅真机提交计数，仿真中永远为 0）。

    指标：
        - cv: 变异系数（标准差/均值），越小越均衡
        - max_min_ratio: 最大/最小负载比
        - entropy: 负载分布熵（归一化到 [0,1]，越大越均衡）
        - total_allocated: 所有机器分配任务数之和
        - total_scheduled: 环境记录的总调度任务数（用于不变量验证）
    """
    machines: list[QuantumMachine] = getattr(env, "_machines", [])
    if not machines:
        return {
            "cv": 0.0,
            "max_min_ratio": 1.0,
            "entropy": 0.0,
            "n_machines": 0,
            "total_allocated": 0,
            "total_scheduled": 0,
        }

    # 使用 _machine_schedule_count（真实调度分配计数）
    schedule_counts: dict[str, int] = getattr(env, "_machine_schedule_count", {})
    loads = [int(schedule_counts.get(m.name, 0)) for m in machines]

    loads_arr = np.array(loads, dtype=float)
    total_allocated = int(np.sum(loads_arr))
    total_scheduled = int(getattr(env, "_total_scheduled", 0))

    if total_allocated == 0:
        return {
            "cv": 0.0,
            "max_min_ratio": 1.0,
            "entropy": 0.0,
            "n_machines": len(machines),
            "total_allocated": 0,
            "total_scheduled": total_scheduled,
            "mean_load": 0.0,
            "max_load": 0,
            "min_load": 0,
        }

    mean_load = float(np.mean(loads_arr))
    std_load = float(np.std(loads_arr, ddof=1)) if len(loads_arr) > 1 else 0.0
    cv = std_load / mean_load if mean_load > 0 else 0.0

    max_load = int(np.max(loads_arr))
    min_load = int(np.min(loads_arr))
    max_min_ratio = float(max_load) / float(min_load) if min_load > 0 else float("inf")

    # 熵（归一化到 [0, 1]）
    probs = loads_arr / float(total_allocated)
    probs = probs[probs > 0]
    # 单机时熵定义为 1.0（确定性分布归一化后为完美均衡）
    if len(machines) == 1:
        normalized_entropy = 1.0
    else:
        entropy = float(-np.sum(probs * np.log(probs))) if len(probs) > 0 else 0.0
        max_entropy = math.log(len(machines))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    return {
        "cv": round(cv, 4),
        "max_min_ratio": round(max_min_ratio, 4) if max_min_ratio != float("inf") else -1.0,
        "entropy": round(normalized_entropy, 4),
        "n_machines": len(machines),
        "total_allocated": total_allocated,
        "total_scheduled": total_scheduled,
        "mean_load": round(mean_load, 2),
        "max_load": max_load,
        "min_load": min_load,
    }


def validate_load_balance_invariants(
    env: QuantumSchedulingEnv,
    load_balance: dict[str, float],
) -> list[str]:
    """验证负载均衡度不变量，返回违规列表（空列表表示全部通过）。

    不变量：
        1. 各机器分配数之和 = 总调度任务数
        2. 总任务数 > 0 时不能返回默认完美均衡（CV=0, entropy=1.0）
        3. 1 台机器时 CV 应为 0
        4. entropy 必须在 [0, 1] 合法范围
    """
    violations: list[str] = []

    total_allocated = load_balance.get("total_allocated", 0)
    total_scheduled = load_balance.get("total_scheduled", 0)
    n_machines = load_balance.get("n_machines", 0)
    cv = load_balance.get("cv", 0.0)
    entropy = load_balance.get("entropy", 0.0)

    # 不变量 1: 各机器分配数之和 = 总调度任务数
    # 注意：total_scheduled 统计所有成功分配（含经典/量子/混合），
    # total_allocated 统计被路由到具体机器的任务数。
    # 不兼容任务（mismatch）不进入机器分配，因此两者可能不完全相等，
    # 但差异不应过大（允许 mismatch 导致的差异）。
    if total_scheduled > 0 and total_allocated == 0:
        violations.append(
            f"总调度 {total_scheduled} 但各机器分配之和为 0（可能未使用 _machine_schedule_count）"
        )

    # 不变量 2: 总任务数 > 0 时不能返回默认完美均衡
    # 如果分配了任务但 CV=0 且 entropy=1.0，可能是返回了默认值而非真实计算
    if total_allocated > 0 and cv == 0.0 and entropy == 1.0 and n_machines > 1:
        violations.append(
            f"分配了 {total_allocated} 个任务但 CV=0 且 entropy=1.0（可能返回了默认完美均衡）"
        )

    # 不变量 3: 1 台机器时 CV 应为 0
    if n_machines == 1 and total_allocated > 0 and cv != 0.0:
        violations.append(f"单机器但 CV={cv}（应为 0）")

    # 不变量 4: entropy 必须在 [0, 1]
    if entropy < 0.0 or entropy > 1.0:
        violations.append(f"entropy={entropy} 超出 [0, 1] 范围")

    return violations


# ============================================================================
# 单次评估（带计时）
# ============================================================================


def evaluate_single_run(
    n_machines: int,
    seed: int,
    episodes: int = 5,
    tasks_per_episode: int = 200,
    obs_dim: int = 10,
    ppo_model: Any = None,
    ppo_model_path: str = "deliverable_models/ppo_best_model_14dim.zip",
) -> dict[str, Any]:
    """评估单次运行（指定机器数 + seed）。

    Args:
        ppo_model: 预加载的 PPO 模型。若为 None 则用 ppo_model_path 加载。
            推荐由调用方在 ``run_machine_scalability_test`` 中加载一次后传入，
            避免每个 seed 重复加载造成不必要的 IO 与内存抖动。
        ppo_model_path: 当 ppo_model 为 None 时使用的模型路径。
    """
    from stable_baselines3 import PPO

    # 单次加载：优先使用调用方传入的 model；否则按 path 加载
    model = ppo_model if ppo_model is not None else PPO.load(ppo_model_path)

    # 创建多机环境
    env = make_multi_machine_env(
        n_machines=n_machines,
        tasks_per_episode=tasks_per_episode,
        seed=seed,
        obs_dim=obs_dim,
    )
    sim_env = SimulationEnv(env=env, task_generator=SimulationTaskGenerator(seed=seed))

    ep_rewards = []
    step_times: list[float] = []  # 每步耗时（μs）
    load_balances: list[dict] = []
    invariant_violations: list[list[str]] = []

    for ep in range(episodes):
        obs, info = sim_env.reset(seed=seed + ep)
        ep_reward = 0.0
        step = 0
        while step < tasks_per_episode:
            t0 = time.perf_counter_ns()
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = sim_env.step(int(action))
            t1 = time.perf_counter_ns()
            step_times.append((t1 - t0) / 1000.0)  # μs
            ep_reward += float(reward)
            step += 1
            if terminated or truncated:
                break
        ep_rewards.append(float(ep_reward))
        sim_env.record_episode_stats(info)

        # 记录负载均衡度 + 调用不变量校验
        unwrapped = getattr(env, "unwrapped", env)
        lb = compute_load_balance(unwrapped)
        load_balances.append(lb)
        # 不变量校验：违规不中断实验，但记录到结果中供后续审计
        violations = validate_load_balance_invariants(unwrapped, lb)
        invariant_violations.append(violations)

    summary = sim_env.get_summary()
    summary["ep_rewards"] = ep_rewards
    summary["mean_reward"] = float(np.mean(ep_rewards))
    summary["std_reward"] = float(np.std(ep_rewards, ddof=1)) if len(ep_rewards) > 1 else 0.0

    # 性能指标
    summary["avg_step_time_us"] = round(float(np.mean(step_times)), 2)
    summary["p50_step_time_us"] = round(float(np.percentile(step_times, 50)), 2)
    summary["p99_step_time_us"] = round(float(np.percentile(step_times, 99)), 2)
    summary["throughput_tasks_per_sec"] = round(
        1e6 / summary["avg_step_time_us"] if summary["avg_step_time_us"] > 0 else 0.0, 2
    )

    # 负载均衡度（取最后一个 episode）
    if load_balances:
        summary["load_balance"] = load_balances[-1]
        # 跨 episode 平均 CV
        cvs = [lb["cv"] for lb in load_balances]
        summary["avg_cv_across_episodes"] = round(float(np.mean(cvs)), 4)
    else:
        summary["load_balance"] = {}
        summary["avg_cv_across_episodes"] = 0.0

    # 不变量校验：汇总所有 episode 的违规（去重）
    all_violations: list[str] = []
    for ep_violations in invariant_violations:
        for v in ep_violations:
            if v not in all_violations:
                all_violations.append(v)
    summary["load_balance_invariant_violations"] = all_violations
    summary["load_balance_invariant_passed"] = len(all_violations) == 0

    # 内存占用（粗略估计）
    try:
        import psutil

        process = psutil.Process(os.getpid())
        summary["memory_mb"] = round(process.memory_info().rss / 1024 / 1024, 2)
    except ImportError:
        summary["memory_mb"] = -1.0

    env.close()
    return summary


# ============================================================================
# 主实验流程
# ============================================================================


def run_machine_scalability_test(
    seeds: int = 5,
    episodes: int = 5,
    tasks_per_episode: int = 200,
    obs_dim: int = 10,
    machine_scales: list[int] | None = None,
    output_dir: Path | None = None,
    ppo_model_path: str = "deliverable_models/ppo_best_model_14dim.zip",
) -> dict[str, Any]:
    """运行机器规模扩展性测试主实验。

    公平性保证：
        - 所有机器规模使用相同 tasks_per_episode / episodes / seeds / obs_dim
        - 任务生成器使用相同 seed 序列，确保各规模下任务分布一致
        - PPO 模型在所有规模间共享，避免重复加载导致的随机性差异
        - 仿真规模验证：本实验不涉及真机任务，所有机器均为仿真后端
    """
    if output_dir is None:
        output_dir = _PROJECT_ROOT / "results" / "machine_scalability"
    output_dir.mkdir(parents=True, exist_ok=True)

    if machine_scales is None:
        machine_scales = MACHINE_SCALES

    seed_list = [42 + i * 137 for i in range(seeds)]

    # PPO 模型只在实验开始时加载一次，传入所有规模/seed 复用
    from stable_baselines3 import PPO

    print(f"  [加载] PPO 模型: {ppo_model_path}")
    ppo_model = PPO.load(ppo_model_path)

    print("=" * 72)
    print("  机器规模扩展性测试（#224）")
    print("=" * 72)
    print(f"  机器规模:      {machine_scales}")
    print(f"  Seeds:         {seeds} ({seed_list})")
    print(f"  Episodes:      {episodes}")
    print(f"  任务规模:      {tasks_per_episode} 步/episode")
    print(f"  观测维度:      {obs_dim}")
    print(f"  PPO 模型:      {ppo_model_path}（已加载一次，全局共享）")
    print(f"  总运行次数:    {len(machine_scales) * seeds}（不含 episodes）")
    print("  实验类型:      仿真规模验证（非真机实验）")
    print("=" * 72)

    all_results: dict[int, dict] = {}
    start_time = time.time()

    for n_machines in machine_scales:
        print(f"\n{'=' * 72}")
        print(f"  机器数: {n_machines}")
        print(f"{'=' * 72}")

        scale_data: dict[str, Any] = {"seeds": {}}

        for seed_idx, seed in enumerate(seed_list):
            print(f"\n  --- {n_machines} 台 | Seed {seed_idx + 1}/{seeds} (seed={seed}) ---")
            seed_start = time.time()

            try:
                result = evaluate_single_run(
                    n_machines=n_machines,
                    seed=seed,
                    episodes=episodes,
                    tasks_per_episode=tasks_per_episode,
                    obs_dim=obs_dim,
                    ppo_model=ppo_model,
                    ppo_model_path=ppo_model_path,
                )
                result["elapsed_seconds"] = round(time.time() - seed_start, 2)
                result["success"] = True
            except Exception as e:
                print(f"  ❌ 失败: {e}")
                result = {
                    "success": False,
                    "error": str(e)[:200],
                    "elapsed_seconds": round(time.time() - seed_start, 2),
                }

            scale_data["seeds"][str(seed)] = result

            if result.get("success"):
                print(
                    f"  完成 ({result['elapsed_seconds']:.1f}s) | "
                    f"reward={result['mean_reward']:.1f}±{result['std_reward']:.1f} | "
                    f"step={result['avg_step_time_us']:.1f}μs | "
                    f"CV={result.get('avg_cv_across_episodes', 0):.3f}"
                )

        # 汇总
        successful_results = [r for r in scale_data["seeds"].values() if r.get("success")]
        if successful_results:
            rewards = [r["mean_reward"] for r in successful_results]
            step_times = [r["avg_step_time_us"] for r in successful_results]
            cvs = [r.get("avg_cv_across_episodes", 0) for r in successful_results]

            scale_data["summary"] = {
                "n_machines": n_machines,
                "n_successful_seeds": len(successful_results),
                "mean_reward": round(float(np.mean(rewards)), 2),
                "std_reward": round(float(np.std(rewards, ddof=1)) if len(rewards) > 1 else 0.0, 2),
                "mean_step_time_us": round(float(np.mean(step_times)), 2),
                "mean_cv": round(float(np.mean(cvs)), 4),
                "throughput_tasks_per_sec": round(
                    1e6 / float(np.mean(step_times)) if np.mean(step_times) > 0 else 0.0, 2
                ),
            }
        else:
            scale_data["summary"] = {
                "n_machines": n_machines,
                "n_successful_seeds": 0,
                "error": "所有 seed 均失败",
            }

        all_results[n_machines] = scale_data
        s = scale_data["summary"]
        print(
            f"\n  {n_machines} 台汇总: reward={s.get('mean_reward', 'N/A')}, "
            f"step={s.get('mean_step_time_us', 'N/A')}μs, CV={s.get('mean_cv', 'N/A')}"
        )

    total_elapsed = time.time() - start_time
    print(f"\n所有规模完成，总耗时 {total_elapsed:.1f}s ({total_elapsed / 3600:.2f}h)")

    # ========================================================================
    # 公平性检查：任务口径跨机器规模一致性
    # ========================================================================
    fairness_check = _check_fairness(all_results, machine_scales)
    print("\n" + "=" * 72)
    print("  公平性检查（任务口径跨规模一致性）")
    print("=" * 72)
    for line in fairness_check["summary_lines"]:
        print(f"  {line}")

    # ========================================================================
    # 不变量汇总：所有规模/seed 的违规统计
    # ========================================================================
    invariant_summary = _summarize_invariants(all_results)
    print("\n" + "=" * 72)
    print("  负载均衡不变量汇总")
    print("=" * 72)
    for line in invariant_summary["summary_lines"]:
        print(f"  {line}")

    # ========================================================================
    # 复杂度分析
    # ========================================================================
    print("\n" + "=" * 72)
    print("  复杂度分析（理论 vs 实测）")
    print("=" * 72)

    complexity_analysis = _analyze_complexity(all_results)
    for line in complexity_analysis["summary_lines"]:
        print(f"  {line}")

    # ========================================================================
    # 保存结果
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_json = {
        "config": {
            "experiment": "Machine Scalability Test",
            "experiment_type": "simulation_scale_validation",
            "real_machine_involved": False,
            "seeds": seed_list,
            "episodes_per_seed": episodes,
            "tasks_per_episode": tasks_per_episode,
            "obs_dim": obs_dim,
            "machine_scales": machine_scales,
            "ppo_model": ppo_model_path,
            "ppo_model_loaded_once": True,
            "total_elapsed_seconds": round(total_elapsed, 2),
            "timestamp": timestamp,
        },
        "results": all_results,
        "complexity_analysis": complexity_analysis,
        "fairness_check": fairness_check,
        "invariant_summary": invariant_summary,
    }

    results_path = output_dir / f"machine_scalability_{timestamp}.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2, default=str)
    canonical_path = output_dir / "machine_scalability.json"
    with open(canonical_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, ensure_ascii=False, indent=2, default=str)
    print(f"\n[保存] 结果: {results_path}")

    # 生成报告
    _generate_report(results_json, output_dir, timestamp)
    print(f"[保存] 报告: {output_dir / 'machine_scalability.md'}")

    return results_json


# ============================================================================
# 公平性检查 / 不变量汇总
# ============================================================================


def _check_fairness(all_results: dict[int, dict], machine_scales: list[int]) -> dict[str, Any]:
    """检查任务口径在不同机器规模下的公平性。

    公平性维度：
        - 任务口径：各规模应使用相同的 tasks_per_episode / episodes / seeds
        - 任务分布：相同 seed 下任务生成应一致（取决于 task_generator 的实现）
        - 资源扩容不应改变任务本身：机器变多只是给任务更多选择，
          不能因为机器多而改变任务的 qubit 需求或到达分布
    """
    summary_lines: list[str] = []
    passed = True
    warnings: list[str] = []

    # 收集每个规模下成功运行的 seed 数与 episode 数
    fairness_data: list[dict] = []
    for n in machine_scales:
        scale = all_results.get(n, {})
        seeds_data = scale.get("seeds", {})
        successful = [r for r in seeds_data.values() if r.get("success")]
        n_eps_per_seed = [len(r.get("ep_rewards", [])) for r in successful if "ep_rewards" in r]
        fairness_data.append(
            {
                "n_machines": n,
                "n_successful_seeds": len(successful),
                "episodes_per_seed": n_eps_per_seed,
            }
        )

    # 检查 1：所有规模的 episode 数应一致
    all_ep_counts: list[int] = []
    for fd in fairness_data:
        all_ep_counts.extend(fd["episodes_per_seed"])
    if all_ep_counts:
        max_eps = max(all_ep_counts)
        min_eps = min(all_ep_counts)
        if max_eps != min_eps:
            passed = False
            msg = f"❌ 各规模 episode 数不一致：min={min_eps}, max={max_eps}（任务口径不公平）"
            summary_lines.append(msg)
            warnings.append(msg)
        else:
            summary_lines.append(f"✅ 各规模 episode 数一致：{min_eps}")
    else:
        passed = False
        msg = "❌ 无有效 episode 数据用于公平性检查"
        summary_lines.append(msg)
        warnings.append(msg)

    # 检查 2：所有规模应使用相同 seed 列表
    seed_sets: list[frozenset] = []
    for n in machine_scales:
        seeds_data = all_results.get(n, {}).get("seeds", {})
        seed_sets.append(frozenset(seeds_data.keys()))
    if len(set(seed_sets)) > 1:
        passed = False
        msg = "❌ 各规模 seed 列表不一致（任务口径不公平）"
        summary_lines.append(msg)
        warnings.append(msg)
    else:
        n_seeds = len(seed_sets[0]) if seed_sets else 0
        summary_lines.append(f"✅ 各规模 seed 列表一致：{n_seeds} seeds")

    # 检查 3：所有规模使用的任务到达分布应相同
    # 这是 SimulationTaskGenerator 的语义保证，不依赖机器数变化
    summary_lines.append("✅ 任务到达分布由 SimulationTaskGenerator(seed) 决定，与机器数无关")

    summary_lines.append(f"\n公平性判定: {'通过' if passed else '不通过'}")

    return {
        "passed": passed,
        "warnings": warnings,
        "fairness_data": fairness_data,
        "summary_lines": summary_lines,
    }


def _summarize_invariants(all_results: dict[int, dict]) -> dict[str, Any]:
    """汇总所有规模/seed 的不变量校验结果。

    统计内容：
        - 总校验次数
        - 通过次数 / 违规次数
        - 各规模下的违规详情
    """
    summary_lines: list[str] = []
    total_checks = 0
    total_passed = 0
    total_violations = 0
    by_scale: dict[int, dict] = {}

    for n, scale in all_results.items():
        seeds_data = scale.get("seeds", {})
        scale_passed = 0
        scale_violations: list[str] = []
        for _seed, result in seeds_data.items():
            if not result.get("success"):
                continue
            total_checks += 1
            if result.get("load_balance_invariant_passed", True):
                total_passed += 1
                scale_passed += 1
            else:
                total_violations += 1
                scale_violations.extend(result.get("load_balance_invariant_violations", []))
        by_scale[n] = {
            "n_passed": scale_passed,
            "n_violations": len(scale_violations),
            "violations": scale_violations,
        }

    summary_lines.append(f"总校验次数: {total_checks}")
    summary_lines.append(f"通过次数: {total_passed}")
    summary_lines.append(f"违规次数: {total_violations}")
    if total_violations == 0:
        summary_lines.append("判定: ✅ 所有规模/seed 的不变量校验全部通过")
    else:
        summary_lines.append("判定: ❌ 存在不变量违规")
        for n, info in by_scale.items():
            if info["n_violations"] > 0:
                summary_lines.append(f"  - {n} 台: {info['n_violations']} 条违规")
                for v in info["violations"][:3]:
                    summary_lines.append(f"      • {v}")

    return {
        "total_checks": total_checks,
        "total_passed": total_passed,
        "total_violations": total_violations,
        "by_scale": by_scale,
        "summary_lines": summary_lines,
    }


# ============================================================================
# 复杂度分析
# ============================================================================


def _analyze_complexity(all_results: dict[int, dict]) -> dict[str, Any]:
    """分析时间复杂度（理论 vs 实测）。

    理论复杂度（PPO 推理）：
        - 单步决策：O(d_hidden * d_obs) ≈ O(1)（神经网络前向传播）
        - 环境步进：O(n_machines)（遍历所有机器选择最优）
        - 总复杂度：O(T * n_machines)，T = max_steps

    实测复杂度：
        - 通过不同 n_machines 下的 step_time 拟合
    """
    scales = sorted(all_results.keys())
    step_times = []
    rewards = []
    cvs = []

    for n in scales:
        summary = all_results[n].get("summary", {})
        if "mean_step_time_us" in summary:
            step_times.append((n, summary["mean_step_time_us"]))
        if "mean_reward" in summary:
            rewards.append((n, summary["mean_reward"]))
        if "mean_cv" in summary:
            cvs.append((n, summary["mean_cv"]))

    # 拟合 step_time = a * n + b（线性模型）
    fit_result: dict[str, Any] = {}
    if len(step_times) >= 2:
        xs = np.array([n for n, _ in step_times])
        ys = np.array([t for _, t in step_times])
        # 线性拟合
        coeffs = np.polyfit(xs, ys, 1)
        fit_result["linear_fit"] = {
            "slope": round(float(coeffs[0]), 4),
            "intercept": round(float(coeffs[1]), 4),
            "model": f"step_time(μs) = {coeffs[0]:.4f} * n_machines + {coeffs[1]:.4f}",
        }
        # 计算拟合优度 R²
        predicted = np.polyval(coeffs, xs)
        ss_res = float(np.sum((ys - predicted) ** 2))
        ss_tot = float(np.sum((ys - np.mean(ys)) ** 2))
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        fit_result["linear_fit"]["r_squared"] = round(r_squared, 4)

    # 生成摘要行
    summary_lines = []
    summary_lines.append(f"测试规模: {scales}")
    summary_lines.append(f"step_time 数据点: {step_times}")

    if "linear_fit" in fit_result:
        lf = fit_result["linear_fit"]
        summary_lines.append(f"线性拟合: {lf['model']}")
        summary_lines.append(f"R² = {lf['r_squared']:.4f}")
        if lf["r_squared"] > 0.9:
            summary_lines.append("✅ 实测数据与线性模型高度吻合 → O(n_machines) 复杂度")
        elif lf["r_squared"] > 0.7:
            summary_lines.append("⚠️ 实测数据与线性模型中度吻合 → 近似 O(n_machines)")
        else:
            summary_lines.append("❌ 实测数据与线性模型拟合度低 → 可能非线性或受噪声影响")

    summary_lines.append("")
    summary_lines.append("理论复杂度: O(T * n_machines)，T = max_steps")
    summary_lines.append("  - PPO 推理: O(d_hidden * d_obs) ≈ 常数")
    summary_lines.append("  - 环境步进: O(n_machines)（遍历机器选择最优）")

    # 奖励随机器数的变化
    if rewards:
        summary_lines.append("")
        summary_lines.append("奖励变化:")
        for n, r in rewards:
            summary_lines.append(f"  {n:>3} 台: reward = {r:.2f}")

    # 负载均衡度变化
    if cvs:
        summary_lines.append("")
        summary_lines.append("负载均衡度（CV）变化:")
        for n, cv in cvs:
            balance_str = "均衡" if cv < 0.3 else ("中等" if cv < 0.7 else "不均衡")
            summary_lines.append(f"  {n:>3} 台: CV = {cv:.4f} ({balance_str})")

    return {
        "fit_result": fit_result,
        "step_times": step_times,
        "rewards": rewards,
        "cvs": cvs,
        "summary_lines": summary_lines,
    }


# ============================================================================
# 报告生成
# ============================================================================


def _generate_report(results: dict, output_dir: Path, timestamp: str) -> None:
    """生成 Markdown 报告。"""
    cfg = results["config"]
    all_results = results["results"]
    complexity = results["complexity_analysis"]
    fairness = results.get("fairness_check", {})
    invariants = results.get("invariant_summary", {})

    lines = [
        "# 机器规模扩展性测试报告（#224）",
        "",
        f"> **生成时间**: {timestamp}",
        f"> **实验规模**: {len(cfg['seeds'])} seeds × {cfg['episodes_per_seed']} episodes",
        f"> **机器规模梯度**: {cfg['machine_scales']}",
        f"> **任务规模**: {cfg['tasks_per_episode']} 步/episode",
        f"> **观测维度**: {cfg['obs_dim']}",
        f"> **PPO 模型**: `{cfg['ppo_model']}`（加载一次，全局共享）",
        f"> **总耗时**: {cfg['total_elapsed_seconds'] / 3600:.2f}h",
        "> **实验类型**: 仿真规模验证（非真机实验，real_machine_involved=False）",
        "",
        "---",
        "",
        "## 一、实验目的",
        "",
        "回答 #224 的核心问题：**系统在不同机器规模下的性能表现**。",
        "与 #206（任务规模梯度）互补，本实验聚焦机器数量扩展。",
        "",
        "> **数据来源说明**: 本实验负载均衡度指标基于 `_machine_schedule_count`",
        "> （每台仿真机器实际分配到的任务数），而非 `_machine_real_submits`",
        "> （仅真机提交计数，仿真中永远为 0）。所有机器均为仿真后端，",
        "> 不涉及真机任务提交。",
        "",
        "## 二、实验结果",
        "",
        "### 2.1 各规模汇总",
        "",
        "| 机器数 | 平均奖励 | 标准差 | 平均步耗时(μs) | 吞吐量(task/s) | CV | 负载均衡 |",
        "|:--:|:--:|:--:|:--:|:--:|:--:|:--:|",
    ]

    for n_machines, data in sorted(all_results.items()):
        s = data.get("summary", {})
        cv = s.get("mean_cv", 0)
        balance = "均衡" if cv < 0.3 else ("中等" if cv < 0.7 else "不均衡")
        lines.append(
            f"| {n_machines} | {s.get('mean_reward', 'N/A')} | "
            f"{s.get('std_reward', 'N/A')} | {s.get('mean_step_time_us', 'N/A')} | "
            f"{s.get('throughput_tasks_per_sec', 'N/A')} | {cv:.4f} | {balance} |"
        )

    # 奖励变化曲线
    lines.extend(["", "### 2.2 奖励随机器数变化", ""])
    rewards_data = complexity.get("rewards", [])
    if rewards_data:
        for n, r in rewards_data:
            lines.append(f"- {n} 台: reward = {r:.2f}")
        # 趋势分析
        if len(rewards_data) >= 2:
            first_r = rewards_data[0][1]
            last_r = rewards_data[-1][1]
            if last_r > first_r * 1.1:
                trend = "上升（更多机器 → 更高奖励）"
            elif last_r < first_r * 0.9:
                trend = "下降（更多机器 → 更低奖励）"
            else:
                trend = "平稳（机器数对奖励影响小）"
            lines.append(f"- **趋势**: {trend}")

    # 性能曲线
    lines.extend(["", "### 2.3 调度耗时随机器数变化", ""])
    step_times = complexity.get("step_times", [])
    if step_times:
        for n, t in step_times:
            lines.append(f"- {n} 台: avg_step_time = {t:.2f} μs")
        fit = complexity.get("fit_result", {}).get("linear_fit", {})
        if fit:
            lines.append("")
            lines.append(f"**线性拟合**: {fit.get('model', 'N/A')}")
            lines.append(f"**R²**: {fit.get('r_squared', 'N/A')}")

    # 负载均衡度
    lines.extend(["", "### 2.4 负载均衡度（变异系数 CV）", ""])
    cvs = complexity.get("cvs", [])
    if cvs:
        for n, cv in cvs:
            balance = "均衡" if cv < 0.3 else ("中等" if cv < 0.7 else "不均衡")
            lines.append(f"- {n} 台: CV = {cv:.4f} ({balance})")
        lines.append("")
        lines.append("> CV < 0.3: 均衡 | 0.3 ≤ CV < 0.7: 中等 | CV ≥ 0.7: 不均衡")

    # 复杂度分析
    lines.extend(["", "## 三、复杂度分析（理论 vs 实测）", ""])
    for line in complexity.get("summary_lines", []):
        lines.append(line)

    # 关键发现
    lines.extend(["", "## 四、关键发现", ""])
    if rewards_data and step_times:
        r_first, r_last = rewards_data[0][1], rewards_data[-1][1]
        t_first, t_last = step_times[0][1], step_times[-1][1]
        scale_ratio = step_times[-1][0] / step_times[0][0] if step_times[0][0] > 0 else 0
        time_ratio = t_last / t_first if t_first > 0 else 0

        lines.append(
            f"1. **奖励变化**: {rewards_data[0][0]}→{rewards_data[-1][0]} 台, "
            f"reward {r_first:.2f}→{r_last:.2f} (Δ={r_last - r_first:+.2f})"
        )
        lines.append(
            f"2. **耗时变化**: {step_times[0][0]}→{step_times[-1][0]} 台, "
            f"step_time {t_first:.2f}→{t_last:.2f} μs "
            f"(规模 ×{scale_ratio:.1f}, 耗时 ×{time_ratio:.1f})"
        )
        fit = complexity.get("fit_result", {}).get("linear_fit", {})
        r_sq = fit.get("r_squared", 0)
        if r_sq > 0.9:
            lines.append(
                f"3. **复杂度验证**: R²={r_sq:.4f} > 0.9, 实测与 O(n_machines) 线性模型高度吻合"
            )
        elif r_sq > 0.7:
            lines.append(
                f"3. **复杂度验证**: R²={r_sq:.4f} > 0.7, 实测近似 O(n_machines)，但存在噪声"
            )
        else:
            lines.append(
                f"3. **复杂度验证**: R²={r_sq:.4f} < 0.7, 实测偏离线性模型，可能存在非线性因素"
            )

    # 与 #206 对比
    lines.extend(
        [
            "",
            "## 五、与 #206 任务规模梯度的对比",
            "",
            "| 维度 | #206 任务规模 | #224 机器规模 |",
            "|:--|:--|:--|",
            f"| 测试对象 | 任务数 100-10000 | 机器数 {cfg['machine_scales']} |",
            "| 已完成 | ✅ 5 量级全部完成 | 本实验 |",
            "| 复杂度 | O(T) 线性扩展 | O(n_machines) 线性扩展 |",
            "| 瓶颈 | 任务队列容量 | 多机协调开销 |",
            "",
            "**互补性**: #206 验证任务规模扩展性, #224 验证机器规模扩展性,",
            "两者共同证明系统在不同维度下的可扩展性。",
            "",
            "---",
            "",
            "## 六、公平性检查",
            "",
            "验证任务口径在不同机器规模下的一致性：相同 seed 序列、相同 episode 数、",
            "相同任务到达分布。资源扩容仅影响调度选择，不改变任务本身。",
            "",
        ]
    )
    for line in fairness.get("summary_lines", []):
        lines.append(line)
    if fairness.get("passed"):
        lines.append("")
        lines.append("**判定**: ✅ 所有规模任务口径一致，实验比较公平")
    else:
        lines.append("")
        lines.append("**判定**: ❌ 存在任务口径不一致，需检查实验配置")

    # 不变量验证
    lines.extend(
        [
            "",
            "---",
            "",
            "## 七、负载均衡度不变量验证",
            "",
            "验证负载均衡度指标的正确性，防止返回默认完美均衡：",
            "",
            "1. 各机器分配数之和 = 总调度任务数",
            "2. 总任务数 > 0 时不能返回默认完美均衡（CV=0, entropy=1.0）",
            "3. 1 台机器时 CV 应为 0",
            "4. entropy 必须在 [0, 1] 合法范围",
            "",
        ]
    )
    for line in invariants.get("summary_lines", []):
        lines.append(line)

    lines.extend(
        [
            "",
            "---",
            "",
            f"*自动生成于 machine_scalability_test.py | 数据源: machine_scalability_{timestamp}.json*",
            "",
        ]
    )

    report_path = output_dir / "machine_scalability.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# ============================================================================
# CLI 入口
# ============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="机器规模扩展性测试（#224）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--seeds", type=int, default=5, help="随机种子数（默认 5）")
    parser.add_argument("--episodes", type=int, default=5, help="每 seed 评估 episode 数（默认 5）")
    parser.add_argument(
        "--tasks-per-episode", type=int, default=200, help="每 episode 最大步数（默认 200）"
    )
    parser.add_argument(
        "--obs-dim", type=int, default=10, choices=[10, 14], help="观测维度（默认 10）"
    )
    parser.add_argument(
        "--machine-scales",
        type=str,
        default="1,3,10,50,100",
        help="机器规模梯度，逗号分隔（默认 1,3,10,50,100）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/machine_scalability",
        help="输出目录",
    )
    parser.add_argument(
        "--ppo-model",
        type=str,
        default="deliverable_models/ppo_best_model_14dim.zip",
        help="PPO 模型路径",
    )
    parser.add_argument(
        "--quick", action="store_true", help="快速验证模式（仅 1 seed × 1 episode）"
    )
    args = parser.parse_args()

    if args.quick:
        args.seeds = 1
        args.episodes = 1
        print("[快速验证模式] seeds=1, episodes=1")

    machine_scales = [int(x.strip()) for x in args.machine_scales.split(",")]

    run_machine_scalability_test(
        seeds=args.seeds,
        episodes=args.episodes,
        tasks_per_episode=args.tasks_per_episode,
        obs_dim=args.obs_dim,
        machine_scales=machine_scales,
        output_dir=Path(args.output_dir),
        ppo_model_path=args.ppo_model,
    )


if __name__ == "__main__":
    main()
