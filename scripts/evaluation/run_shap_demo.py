"""
SHAP/特征贡献度可视化演示脚本（线B：可解释性可视化，杠杆⑤）

打通 PPOExplainer 真实调用链：加载调度层 16 维 PPO 模型 → 跑 1 个 episode →
对每一步决策计算 16 维特征贡献度 → 输出：
    1. 单步决策解释图（top-8 特征贡献条形图，决策放大镜）
    2. 全局特征重要性排名图（聚合全部决策步）
    3. 决策记录 JSON（results/shap/shap_records.json）

产出（PPT/白皮书可直接引用的答辩素材）:
    - results/shap/decision_step_<n>.png     单步决策解释
    - results/shap/feature_importance.png     全局特征重要性排名
    - results/shap/shap_records.json         全部决策记录

用法:
    python scripts/evaluation/run_shap_demo.py [--episodes 3] [--max-steps 100]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无 GUI 环境出图
import matplotlib.pyplot as plt
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
os.chdir(str(_PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.explainability import STATE_FEATURE_NAMES, PPOExplainer

MODEL_PATH = "deliverable_models/ppo_best_model_16dim.zip"
OUT_DIR = Path("results/shap")


def build_env(seed: int = 31) -> QuantumSchedulingEnv:
    """构造调度层环境（与 run_simulation 一致的 3 机配置，16 维观测）。"""
    machine_configs = [
        {
            "name": "tianyan_l",
            "total_qubits": 10,
            "fidelity": 0.98,
            "available": True,
            "supported_gates": ("H", "CNOT", "M"),
        },
        {
            "name": "tianyan_s",
            "total_qubits": 6,
            "fidelity": 0.96,
            "available": True,
            "supported_gates": ("H", "CNOT", "M"),
        },
        {
            "name": "tianyan_sw",
            "total_qubits": 4,
            "fidelity": 0.95,
            "available": True,
            "supported_gates": ("H", "CNOT", "M"),
        },
    ]
    return QuantumSchedulingEnv(max_steps=100, machine_configs=machine_configs, seed=seed)


def run_demo(episodes: int, max_steps: int) -> list[dict]:
    """运行 episodes 个 episode，逐决策步计算特征贡献度。"""
    from stable_baselines3 import PPO

    model = PPO.load(MODEL_PATH)
    env = build_env()
    explainer = PPOExplainer(model=model, feature_names=STATE_FEATURE_NAMES, method="heuristic")

    records: list[dict] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=31 + ep)
        for step in range(max_steps):
            action, _ = model.predict(obs, deterministic=True)
            # PPOExplainer.explain(observation, action) -> dict[特征名, 贡献度]
            contributions = explainer.explain(np.asarray(obs, dtype=np.float64), int(action))
            ranked = sorted(contributions.items(), key=lambda kv: kv[1], reverse=True)
            records.append(
                {
                    "episode": ep,
                    "step": step,
                    "action": int(action),
                    "state": [float(v) for v in np.asarray(obs).flatten()],
                    "feature_contributions": contributions,
                    "top_features": [{"feature": k, "value": float(v)} for k, v in ranked[:8]],
                }
            )
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                break
    return records


def plot_single_step(record: dict, path: Path) -> None:
    """单步决策解释图：top-8 特征贡献条形图。"""
    top = record["top_features"]
    if not top:
        return
    names = [t["feature"] for t in top][::-1]
    values = [t["value"] for t in top][::-1]
    colors = ["#d62728" if v >= 0 else "#1f77b4" for v in values]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(names, values, color=colors)
    ax.set_xlabel(f"特征贡献度（对决策动作 action={record['action']} 的推动/抑制程度）")
    ax.set_title(f"单步调度决策解释 — Episode {record['episode']} Step {record['step']}")
    ax.axvline(0, color="black", linewidth=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_global_importance(records: list[dict], path: Path) -> None:
    """全局特征重要性排名图（聚合全部决策步的平均贡献）。"""
    acc: dict[str, float] = {}
    for r in records:
        for k, v in r["feature_contributions"].items():
            acc[k] = acc.get(k, 0.0) + abs(v)
    total = sum(acc.values()) or 1.0
    ranked = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)
    names = [k for k, _ in ranked][::-1]
    values = [v / total for k, v in ranked][::-1]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(names, values, color="#2c7fb8")
    ax.set_xlabel(f"归一化平均贡献度（聚合 {len(records)} 个决策步）")
    ax.set_title("PPO 调度决策全局特征重要性排名（SHAP 风格）")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="SHAP 可解释性可视化演示")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"加载模型 {MODEL_PATH} ...")
    records = run_demo(args.episodes, args.max_steps)
    print(f"完成 {len(records)} 个决策步的解释")

    # 单步图（取贡献度最显著的一步作代表）
    best = max(records, key=lambda r: max(abs(v) for v in r["feature_contributions"].values()))
    step_path = OUT_DIR / f"decision_step_e{best['episode']}_s{best['step']}.png"
    plot_single_step(best, step_path)
    print(f"单步决策解释图: {step_path}")

    imp_path = OUT_DIR / "feature_importance.png"
    plot_global_importance(records, imp_path)
    print(f"全局特征重要性图: {imp_path}")

    json_path = OUT_DIR / "shap_records.json"
    json_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"决策记录: {json_path}")
    print(f"模型: {MODEL_PATH} | 特征: {len(STATE_FEATURE_NAMES)} 维（调度层）")


if __name__ == "__main__":
    main()
