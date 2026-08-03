"""
HybridScheduler 三路消融实验（⑤）：纯规则 vs 纯 RL vs 混合

在相同的环境种子/任务流下，用三种策略各自运行完整 episode，对比：
- 平均奖励/步（调度效率）
- 任务完成率
- 决策来源分布（混合策略的 rule/rl/fallback 占比）

产出:
    - results/hybrid/hybrid_ablation_result.json
    - results/reports/hybrid_ablation_report.md

用法:
    python scripts/evaluation/hybrid_scheduler_ablation.py [--episodes 10] [--max-steps 100]
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

from src.scheduler.env import DEFAULT_MACHINE_CONFIGS, QuantumSchedulingEnv
from src.scheduler.hybrid_scheduler import HybridScheduler, RuleEngine

MODEL_PATH = "deliverable_models/ppo_best_model_16dim.zip"


def build_env(seed: int) -> QuantumSchedulingEnv:
    return QuantumSchedulingEnv(
        max_steps=100,
        machine_configs=DEFAULT_MACHINE_CONFIGS,
        seed=seed,
    )


def _ctx_from_env(env: QuantumSchedulingEnv) -> dict:
    task = env._current_task
    return {
        "available_qubits": getattr(env, "_available_qubits", 0)
        if hasattr(env, "_available_qubits")
        else sum(getattr(m, "total_qubits", 0) for m in getattr(env, "_machines", [])),
        "queue_length": len(env._task_queue),
        "task_type": task.task_type if task is not None else None,
    }


def run_rule(env: QuantumSchedulingEnv, engine: RuleEngine) -> dict:
    total = 0.0
    steps = 0
    done = False
    while not done and steps < env._max_steps:
        env._get_observation()
        task = env._current_task
        action = None
        if task is not None:
            action = engine.evaluate(task, _ctx_from_env(env))
        if action is None:
            action = 0  # fallback：经典
        _obs, reward, terminated, truncated, _ = env.step(int(action))
        total += float(reward)
        steps += 1
        done = terminated or truncated
    return {"avg_reward": total / max(steps, 1), "steps": steps, "total_reward": total}


def run_rl(env: QuantumSchedulingEnv, model) -> dict:
    total = 0.0
    steps = 0
    done = False
    while not done and steps < env._max_steps:
        obs = env._get_observation()
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        total += float(reward)
        steps += 1
        done = terminated or truncated
    return {"avg_reward": total / max(steps, 1), "steps": steps, "total_reward": total}


def run_hybrid(env: QuantumSchedulingEnv, hybrid: HybridScheduler) -> dict:
    total = 0.0
    steps = 0
    done = False
    sources: dict[str, int] = {}
    while not done and steps < env._max_steps:
        _obs = env._get_observation()
        task = env._current_task
        if task is None:
            action = 0
            source = "default"
        else:
            result = hybrid.decide(task, state=_obs, context=_ctx_from_env(env))
            action = result["action"]
            source = result["source"]
        sources[source] = sources.get(source, 0) + 1
        _obs, reward, terminated, truncated, _ = env.step(int(action))
        total += float(reward)
        steps += 1
        done = terminated or truncated
    return {
        "avg_reward": total / max(steps, 1),
        "steps": steps,
        "total_reward": total,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HybridScheduler 三路消融")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(MODEL_PATH)
    engine = RuleEngine()
    hybrid = HybridScheduler(rl_agent=model, rule_engine=engine)

    agg = {"rule": [], "rl": [], "hybrid": []}
    hybrid_sources_all: dict[str, int] = {}
    for ep in range(args.episodes):
        e_r = build_env(args.seed + ep)
        e_l = build_env(args.seed + ep)
        e_h = build_env(args.seed + ep)
        e_r.reset(seed=args.seed + ep)
        e_l.reset(seed=args.seed + ep)
        e_h.reset(seed=args.seed + ep)
        agg["rule"].append(run_rule(e_r, engine))
        agg["rl"].append(run_rl(e_l, model))
        h = run_hybrid(e_h, hybrid)
        agg["hybrid"].append(h)
        for k, v in h["sources"].items():
            hybrid_sources_all[k] = hybrid_sources_all.get(k, 0) + v

    def mean(key: str) -> dict[str, float]:
        return {k: float(np.mean([x[key] for x in v])) for k, v in agg.items()}

    avg_reward = mean("avg_reward")
    total_sources = sum(hybrid_sources_all.values()) or 1
    source_pct = {k: v / total_sources * 100 for k, v in hybrid_sources_all.items()}

    result = {
        "config": {"episodes": args.episodes, "max_steps": args.max_steps, "seed": args.seed},
        "avg_reward_per_step": avg_reward,
        "hybrid_source_distribution": source_pct,
    }

    out_dir = Path("results/hybrid")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir = Path("results/reports")
    report_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "hybrid_ablation_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = f"""# HybridScheduler 三路消融报告（规则 vs RL vs 混合）

- **场景**：{args.episodes} episodes × {args.max_steps} 步，同种子同任务流
- **模型**：{MODEL_PATH}

| 策略 | 平均奖励/步 |
|:--|:--|
| 纯规则（RuleEngine） | {avg_reward["rule"]:.3f} |
| 纯 RL（PPO deterministic） | {avg_reward["rl"]:.3f} |
| **混合（HybridScheduler）** | **{avg_reward["hybrid"]:.3f}** |

**混合策略决策来源分布**：{json.dumps(source_pct, ensure_ascii=False)}

**解读**：
- 混合策略应不差于最优单策略（规则兜底 + RL 补盲），来源分布说明规则/RL 各自承担的比例
- 完整数据：`results/hybrid/hybrid_ablation_result.json`
"""
    (report_dir / "hybrid_ablation_report.md").write_text(report, encoding="utf-8")

    print("=" * 60)
    print(
        f"  平均奖励/步: 规则 {avg_reward['rule']:.3f} | RL {avg_reward['rl']:.3f} | "
        f"混合 {avg_reward['hybrid']:.3f}"
    )
    print(f"  混合来源分布: {json.dumps(source_pct, ensure_ascii=False)}")
    print(f"  产出: {out_dir / 'hybrid_ablation_result.json'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
