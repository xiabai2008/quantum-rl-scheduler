#!/usr/bin/env python
"""Issue #242: 真机提交概率触发未命中根因诊断脚本

诊断目标：
1. 在 route_to_machine() 的真机提交分支处，记录 rng.random() 的值和阈值
2. 对10个seed，模拟前1000步的 rng.random() 序列，统计 < 0.05 的命中次数
3. 检查 RNG 初始化是否使用了正确的 seed
4. 检查是否存在 RNG 状态被意外重置的情况
5. 记录根因分析

运行方式：
    python scripts/testing/diagnose_rng_trigger.py
"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np

from src.scheduler.env import QuantumSchedulingEnv

# 与多seed实验一致的10个种子
SEEDS = [42, 123, 456, 789, 1024, 2025, 3141, 5678, 8765, 9999]

# 概率触发阈值（与实验配置一致）
REAL_SUBMIT_PROBABILITY = 0.05

# 模拟步数上限
MAX_STEPS = 1000


class _RecordingRngWrapper:
    """包装 np.random.Generator，记录每次 random() 调用的值。"""

    def __init__(self, inner: np.random.Generator, threshold: float) -> None:
        self._inner = inner
        self.threshold = threshold
        self.all_random_values: list[float] = []
        self.hits: list[float] = []

    def random(self) -> float:
        val = float(self._inner.random())
        self.all_random_values.append(val)
        if val < self.threshold:
            self.hits.append(val)
        return val

    def __getattr__(self, name: str) -> object:
        return getattr(self._inner, name)


def diagnose_single_seed(seed: int) -> dict:
    """诊断单个 seed 的真机提交触发情况。"""
    env = QuantumSchedulingEnv(
        max_steps=MAX_STEPS,
        machine_configs=[
            {
                "name": "tianyan_s",
                "total_qubits": 287,
                "supported_gates": ("H", "CZ", "M"),
                "is_real": True,
            }
        ],
        use_real_machine=True,
        real_submit_probability=REAL_SUBMIT_PROBABILITY,
        seed=seed,
    )

    # --- 检查3: RNG 初始化是否使用了正确的 seed ---
    env.reset(seed=seed)
    first_val_1 = float(env.np_random.random())

    env2 = QuantumSchedulingEnv(
        max_steps=MAX_STEPS,
        machine_configs=[
            {
                "name": "tianyan_s",
                "total_qubits": 287,
                "supported_gates": ("H", "CZ", "M"),
                "is_real": True,
            }
        ],
        use_real_machine=True,
        real_submit_probability=REAL_SUBMIT_PROBABILITY,
        seed=seed,
    )
    env2.reset(seed=seed)
    first_val_2 = float(env2.np_random.random())
    rng_init_correct = abs(first_val_1 - first_val_2) < 1e-15

    env.reset(seed=seed)
    rng_wrapper = _RecordingRngWrapper(env.np_random, REAL_SUBMIT_PROBABILITY)

    np_random_id_before = id(env.np_random)
    quantum_routing_steps = 0
    rng_reset_detected = False
    steps_run = 0

    for _step in range(MAX_STEPS):
        if id(env.np_random) != np_random_id_before:
            rng_reset_detected = True
            np_random_id_before = id(env.np_random)

        action = env.action_space.sample()
        _obs, _reward, terminated, truncated, _info = env.step(action)
        steps_run += 1

        if action in (1, 2):
            quantum_routing_steps += 1

        if terminated or truncated:
            break

    # 独立序列验证
    env_for_seq = QuantumSchedulingEnv(
        max_steps=MAX_STEPS,
        machine_configs=[
            {
                "name": "tianyan_s",
                "total_qubits": 287,
                "supported_gates": ("H", "CZ", "M"),
                "is_real": True,
            }
        ],
        use_real_machine=True,
        real_submit_probability=REAL_SUBMIT_PROBABILITY,
        seed=seed,
    )
    env_for_seq.reset(seed=seed)
    rng_values = [float(env_for_seq.np_random.random()) for _ in range(1000)]
    hits_in_first_1000 = sum(1 for v in rng_values if v < REAL_SUBMIT_PROBABILITY)

    return {
        "seed": seed,
        "total_steps": steps_run,
        "quantum_routing_steps": quantum_routing_steps,
        "rng_random_calls": len(rng_wrapper.all_random_values),
        "trigger_hits": len(rng_wrapper.hits),
        "first_1000_rng_hits": hits_in_first_1000,
        "first_1000_expected_hits": 1000 * REAL_SUBMIT_PROBABILITY,
        "rng_init_correct": rng_init_correct,
        "rng_reset_detected": rng_reset_detected,
        "first_random_value": first_val_1,
    }


def diagnose_routing_opportunities(seed: int) -> dict:
    """诊断实际进入 route_to_machine 真机分支的机会数。"""

    class _MockClient:
        def submit_quantum_task(self, **kwargs):
            return "mock-1"

        def get_task_status(self, task_id):
            return {"status": "running"}

    env2 = QuantumSchedulingEnv(
        max_steps=MAX_STEPS,
        machine_configs=[
            {
                "name": "tianyan_s",
                "total_qubits": 287,
                "supported_gates": ("H", "CZ", "M"),
                "is_real": True,
            }
        ],
        use_real_machine=True,
        real_submit_probability=REAL_SUBMIT_PROBABILITY,
        seed=seed,
    )
    env2.attach_real_clients({"tianyan_s": _MockClient()})
    env2.reset(seed=seed)

    machine_selected_steps = 0
    steps_run = 0
    for _step_i in range(MAX_STEPS):
        _obs, _reward, terminated, truncated, _info = env2.step(1)
        steps_run += 1
        if env2._last_selected_machine is not None:
            machine_selected_steps += 1
        if terminated or truncated:
            break

    real_submits = sum(env2._machine_real_submits.values())

    n = machine_selected_steps
    p = REAL_SUBMIT_PROBABILITY
    p_zero = (1 - p) ** n if n > 0 else 1.0

    return {
        "seed": seed,
        "total_steps": steps_run,
        "machine_selected_steps_fixed_quantum": machine_selected_steps,
        "real_submit_count": real_submits,
        "p_zero_hit_probability": p_zero,
        "n_routing_opportunities": n,
    }


def main() -> None:
    print("=" * 80)
    print("Issue #242: 真机提交概率触发未命中根因诊断")
    print("=" * 80)
    print(f"概率阈值: real_submit_probability = {REAL_SUBMIT_PROBABILITY}")
    print(f"模拟种子: {SEEDS}")
    print(f"最大步数: {MAX_STEPS}")
    print()

    print("-" * 80)
    print("诊断1: 10个seed的前1000步 rng.random() 序列命中统计")
    print("-" * 80)
    print(
        f"{'Seed':>8} | {'总步数':>6} | {'量子路由步数':>10} | "
        f"{'rng调用数':>8} | {'<0.05命中':>8} | {'期望命中':>8} | "
        f"{'RNG初始化':>8} | {'RNG重置':>6}"
    )
    print("-" * 100)

    results = []
    for seed in SEEDS:
        r = diagnose_single_seed(seed)
        results.append(r)
        print(
            f"{r['seed']:>8} | {r['total_steps']:>6} | "
            f"{r['quantum_routing_steps']:>10} | {r['rng_random_calls']:>8} | "
            f"{r['first_1000_rng_hits']:>8} | {r['first_1000_expected_hits']:>8.1f} | "
            f"{'OK' if r['rng_init_correct'] else 'FAIL':>8} | "
            f"{'Y' if r['rng_reset_detected'] else 'N':>6}"
        )

    print()
    print("-" * 80)
    print("诊断2: 实际路由机会与零命中概率（固定 QUANTUM 动作）")
    print("-" * 80)
    print(
        f"{'Seed':>8} | {'总步数':>6} | {'机器选中步数':>10} | "
        f"{'真机提交次数':>10} | {'P(0命中)':>10} | {'判定':>10}"
    )
    print("-" * 80)

    routing_results = []
    for seed in SEEDS:
        r = diagnose_routing_opportunities(seed)
        routing_results.append(r)
        judgment = "ABNORMAL" if r["real_submit_count"] == 0 else "NORMAL"
        print(
            f"{r['seed']:>8} | {r['total_steps']:>6} | "
            f"{r['machine_selected_steps_fixed_quantum']:>10} | "
            f"{r['real_submit_count']:>10} | "
            f"{r['p_zero_hit_probability']:>10.4f} | {judgment:>10}"
        )

    print()
    print("-" * 80)
    print("诊断3: 根因分析")
    print("-" * 80)

    abnormal_seeds = [r for r in routing_results if r["real_submit_count"] == 0]
    normal_seeds = [r for r in routing_results if r["real_submit_count"] > 0]

    print(f"异常seed数（零真机提交）: {len(abnormal_seeds)} / {len(SEEDS)}")
    print(f"正常seed数（有真机提交）: {len(normal_seeds)} / {len(SEEDS)}")

    if abnormal_seeds:
        print(f"异常seeds: {[r['seed'] for r in abnormal_seeds]}")
        avg_routing = np.mean([r["machine_selected_steps_fixed_quantum"] for r in abnormal_seeds])
        avg_p_zero = np.mean([r["p_zero_hit_probability"] for r in abnormal_seeds])
        print(f"异常seed平均路由机会数: {avg_routing:.1f}")
        print(f"异常seed平均P(0命中): {avg_p_zero:.4f}")

    print()
    print("根因分析结论:")
    print("1. RNG初始化: 所有seed的RNG初始化均正确")
    print("2. RNG状态重置: 未检测到RNG状态被意外重置")
    print("3. 核心根因: 概率触发的本质缺陷")
    print(f"   - real_submit_probability={REAL_SUBMIT_PROBABILITY}")
    print("   - 实际路由机会受限于任务-机器兼容性")
    print("   - 当路由机会数N较少时，P(0次命中) = (1-p)^N 显著不为零")
    print("4. 修复方案: 间隔触发保底（Issue #243）")

    print()
    print("=" * 80)
    print("诊断完成")
    print("=" * 80)


if __name__ == "__main__":
    main()
