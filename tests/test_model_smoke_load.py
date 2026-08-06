"""交付模型冒烟加载测试 — 8.6 审查修复。

背景：此前 `tests/test_performance_benchmarks.py` 被 CI `--ignore` 排除，
模型加载路径在 CI 中无任何自动化覆盖（8.6 审查报告 N15 复现性墙）。
本测试在常规测试套件中验证交付模型 `deliverable_models/ppo_best_model_16dim.zip`
可稳定加载并完成一次 16 维观测推理，作为交付物可复现性的门禁。

- 模型文件缺失时 skip（不阻断 CI，允许仅代码评审场景）。
- 模型存在但加载失败时 FAIL（这正是门禁的用途：暴露交付物不可加载）。
- 显式 `device="cpu"`，避免无 GPU 的 CI runner 告警/依赖 CUDA。
"""

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PPO_MODEL_PATH = PROJECT_ROOT / "deliverable_models" / "ppo_best_model_16dim.zip"


def _find_ppo_model() -> Path | None:
    """在标准候选路径中查找交付 PPO 模型。"""
    candidates = [
        PPO_MODEL_PATH,
        PROJECT_ROOT / "models" / "ppo_seed_42" / "best_model.zip",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def test_deliverable_ppo_model_loads_and_infers() -> None:
    """交付模型可加载并完成一次 16 维观测推理（冒烟验证）。"""
    model_path = _find_ppo_model()
    if model_path is None:
        pytest.skip(f"PPO 交付模型未找到: {PPO_MODEL_PATH}")

    from stable_baselines3 import PPO

    from src.scheduler.env import QuantumSchedulingEnv

    env = QuantumSchedulingEnv(max_steps=100, seed=42)
    model = PPO.load(str(model_path), env=env, device="cpu")

    obs, _ = env.reset(seed=42)
    action, _ = model.predict(obs.reshape(1, -1), deterministic=True)

    assert int(action.item()) in (0, 1, 2), f"非法调度动作: {action}"
