"""
在 4×4 2D网格拓扑上重新训练 PPO 编译智能体
使用全电路分布（浅/中/深），每个 episode 采样新电路
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from gymnasium import spaces
from qiskit.circuit.random import random_circuit
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.quantum.compilation_env import QuantumCompilationEnv

SEED = 42
N_TIMESTEPS = 200_000
MODEL_SAVE_PATH = "deliverable_models/ppo_compilation_agent.zip"


class MultiCircuitCompilationEnv(QuantumCompilationEnv):
    """编译环境包装器，每次 reset 时采样新电路。"""

    def __init__(self, seed: int = SEED) -> None:
        self.rng = np.random.default_rng(seed)
        self.categories = [
            (5, 8, 5, 10),
            (9, 14, 10, 20),
            (14, 16, 20, 30),
        ]
        qc = self._sample_circuit()
        super().__init__(qc, max_steps=300)
        self.observation_space = spaces.Box(low=0, high=1, shape=(14,), dtype=np.float32)
        self.action_space = spaces.Discrete(16)

    def _sample_circuit(self):
        cat_idx = int(self.rng.integers(0, len(self.categories)))
        q_lo, q_hi, g_lo, g_hi = self.categories[cat_idx]
        n_q = int(self.rng.integers(q_lo, q_hi + 1))
        n_g = int(self.rng.integers(g_lo, g_hi + 1))
        return random_circuit(
            n_q, n_g, measure=False, seed=int(self.rng.integers(0, 2**31 - 1))
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.circuit = self._sample_circuit()
        self.n_logical = self.circuit.num_qubits
        self._init_state()
        return super().reset(seed=seed)


def main() -> None:
    print("=" * 60)
    print("  重新训练 PPO 编译智能体（4×4 2D网格，全电路分布）")
    print("=" * 60)

    env = DummyVecEnv([lambda: MultiCircuitCompilationEnv(seed=SEED)])
    print(f"  观测空间: {env.observation_space}")
    print(f"  动作空间: {env.action_space}")

    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        verbose=1,
        seed=SEED,
        device="cpu",
    )

    t0 = time.time()
    model.learn(total_timesteps=N_TIMESTEPS)
    elapsed = time.time() - t0
    print(f"\n训练完成: {elapsed:.0f}s ({N_TIMESTEPS/elapsed:.0f} FPS)")

    model.save(MODEL_SAVE_PATH)
    print(f"模型已保存: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
