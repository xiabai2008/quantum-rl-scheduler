"""tests/test_holdout_evaluation.py — Issue #29 留出负载盲测单元测试"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from scripts.evaluation.run_holdout_evaluation import (
    HOLDOUT_DISTRIBUTIONS,
    TASK_GENERATORS,
    HoldoutDistribution,
    TraceEntry,
    TraceInjector,
    _gen_burst,
    _gen_high_quantum,
    _gen_in_distribution,
    _gen_long_tail,
    _improvement,
    generate_report,
    generate_trace,
    load_trace,
    save_trace,
)

# ---------------------------------------------------------------------------
# HoldoutDistribution 数据类
# ---------------------------------------------------------------------------


class TestHoldoutDistribution:
    """HoldoutDistribution 数据类测试"""

    def test_distributions_count(self) -> None:
        """应有 4 个分布（1 控制组 + 3 OOD）"""
        assert len(HOLDOUT_DISTRIBUTIONS) == 4

    def test_in_distribution_is_control(self) -> None:
        dist = HOLDOUT_DISTRIBUTIONS[0]
        assert dist.key == "in_distribution"
        assert "控制组" in dist.label or "分布内" in dist.label

    def test_ood_distributions_present(self) -> None:
        keys = {d.key for d in HOLDOUT_DISTRIBUTIONS}
        assert "burst" in keys
        assert "long_tail" in keys
        assert "high_quantum" in keys

    def test_each_distribution_has_generator(self) -> None:
        for dist in HOLDOUT_DISTRIBUTIONS:
            assert dist.generator_key in TASK_GENERATORS


# ---------------------------------------------------------------------------
# 任务生成器
# ---------------------------------------------------------------------------


class TestTaskGenerators:
    """任务生成器测试"""

    def test_in_distribution_generator(self) -> None:
        rng = np.random.default_rng(42)
        task = _gen_in_distribution(rng, 0)
        assert task.task_id == "T0000"
        assert task.task_type in ("quantum", "classical")
        assert task.qubit_count >= 0
        assert 0.0 <= task.urgency <= 1.0
        assert 1 <= task.priority <= 5

    def test_burst_generator_produces_large_tasks(self) -> None:
        """burst 生成器应能产生 50+ 量子比特的大任务"""
        rng = np.random.default_rng(42)
        large_count = 0
        for i in range(100):
            task = _gen_burst(rng, i)
            if task.qubit_count >= 50:
                large_count += 1
        # 50% 概率大任务，100 次中至少应有 30 次大任务
        assert large_count >= 30

    def test_long_tail_generator_execution_time_doubled(self) -> None:
        """long_tail 生成器的执行时间应为基础时间 2 倍"""
        rng = np.random.default_rng(42)
        for i in range(20):
            task = _gen_long_tail(rng, i)
            if task.task_type == "quantum":
                # base_time = int(qubits**0.6), execution_time = max(1, base_time * 2)
                expected_base = int(task.qubit_count**0.6)
                assert task.execution_time == max(1, expected_base * 2)
                break

    def test_high_quantum_ratio(self) -> None:
        """high_quantum 生成器应 95% 量子任务"""
        rng = np.random.default_rng(42)
        quantum_count = 0
        total = 200
        for i in range(total):
            task = _gen_high_quantum(rng, i)
            if task.task_type == "quantum":
                quantum_count += 1
        # 95% 量子，200 次中至少 180 次
        assert quantum_count >= 180


# ---------------------------------------------------------------------------
# Trace 生成与序列化
# ---------------------------------------------------------------------------


class TestTraceGeneration:
    """trace 生成与序列化测试"""

    def test_generate_trace_length(self) -> None:
        dist = HOLDOUT_DISTRIBUTIONS[0]
        trace = generate_trace(dist, trace_length=50, seed=42)
        assert len(trace) == 50

    def test_generate_trace_reproducible(self) -> None:
        """同 seed 应生成相同 trace"""
        dist = HOLDOUT_DISTRIBUTIONS[0]
        trace1 = generate_trace(dist, trace_length=20, seed=42)
        trace2 = generate_trace(dist, trace_length=20, seed=42)
        assert trace1 == trace2

    def test_generate_trace_different_seeds_differ(self) -> None:
        """不同 seed 应生成不同 trace"""
        dist = HOLDOUT_DISTRIBUTIONS[0]
        trace1 = generate_trace(dist, trace_length=20, seed=42)
        trace2 = generate_trace(dist, trace_length=20, seed=99)
        assert trace1 != trace2

    def test_trace_entry_fields(self) -> None:
        trace = generate_trace(HOLDOUT_DISTRIBUTIONS[0], trace_length=1, seed=42)
        entry = trace[0]
        assert isinstance(entry, TraceEntry)
        assert entry.task_id.startswith("T")
        assert entry.task_type in ("quantum", "classical")
        assert entry.qubit_count >= 0
        assert 0.0 <= entry.urgency <= 1.0
        assert 1 <= entry.priority <= 5
        assert entry.execution_time >= 1

    def test_save_and_load_trace(self, tmp_path: Path) -> None:
        trace = generate_trace(HOLDOUT_DISTRIBUTIONS[0], trace_length=10, seed=42)
        path = tmp_path / "trace.jsonl"
        save_trace(trace, path)
        assert path.exists()
        loaded = load_trace(path)
        assert loaded == trace

    def test_trace_jsonl_format(self, tmp_path: Path) -> None:
        """trace 文件应为 JSONL 格式（每行一个 JSON）"""
        trace = generate_trace(HOLDOUT_DISTRIBUTIONS[0], trace_length=3, seed=42)
        path = tmp_path / "trace.jsonl"
        save_trace(trace, path)
        with path.open(encoding="utf-8") as f:
            lines = f.readlines()
        assert len(lines) == 3
        for line in lines:
            data = json.loads(line.strip())
            assert "task_id" in data
            assert "task_type" in data
            assert "qubit_count" in data


# ---------------------------------------------------------------------------
# TraceInjector
# ---------------------------------------------------------------------------


class TestTraceInjector:
    """TraceInjector 测试"""

    def test_injector_returns_trace_tasks(self) -> None:
        """注入后环境应从 trace 顺序读取任务"""
        from src.scheduler.env import QuantumSchedulingEnv

        trace = [
            TraceEntry("T0001", "quantum", 10, 0.5, 3, 5),
            TraceEntry("T0002", "classical", 0, 0.3, 2, 3),
            TraceEntry("T0003", "quantum", 50, 0.9, 5, 10),
        ]
        env = QuantumSchedulingEnv(max_steps=10, max_qubits=287)
        injector = TraceInjector(trace)
        injector.inject(env)

        rng = np.random.default_rng(42)
        task1 = env._generate_random_task(rng, 1)
        task2 = env._generate_random_task(rng, 2)
        task3 = env._generate_random_task(rng, 3)

        assert task1.task_id == "T0001"
        assert task1.qubit_count == 10
        assert task2.task_id == "T0002"
        assert task2.task_type == "classical"
        assert task3.task_id == "T0003"
        assert task3.qubit_count == 50
        env.close()

    def test_injector_falls_back_when_trace_exhausted(self) -> None:
        """trace 耗尽时应回退到原始生成器"""
        from src.scheduler.env import QuantumSchedulingEnv

        trace = [TraceEntry("T0001", "quantum", 10, 0.5, 3, 5)]
        env = QuantumSchedulingEnv(max_steps=10, max_qubits=287)
        injector = TraceInjector(trace)
        injector.inject(env)

        rng = np.random.default_rng(42)
        task1 = env._generate_random_task(rng, 1)
        # trace 耗尽，回退
        task2 = env._generate_random_task(rng, 2)

        assert task1.task_id == "T0001"
        # 回退的任务应有合法 task_id（由原始生成器创建）
        assert task2.task_id.startswith("T")
        env.close()

    def test_injector_index_resets_on_reinject(self) -> None:
        """重新注入时索引应重置"""
        from src.scheduler.env import QuantumSchedulingEnv

        trace = [
            TraceEntry("T0001", "quantum", 10, 0.5, 3, 5),
            TraceEntry("T0002", "classical", 0, 0.3, 2, 3),
        ]
        env = QuantumSchedulingEnv(max_steps=10, max_qubits=287)
        injector = TraceInjector(trace)
        injector.inject(env)

        rng = np.random.default_rng(42)
        env._generate_random_task(rng, 1)  # 消费 T0001

        # 重新注入
        injector.inject(env)
        task = env._generate_random_task(rng, 1)
        assert task.task_id == "T0001"  # 从头开始
        env.close()


# ---------------------------------------------------------------------------
# _improvement
# ---------------------------------------------------------------------------


class TestImprovement:
    """_improvement 函数测试"""

    def test_positive_improvement(self) -> None:
        result = _improvement(150.0, 100.0)
        assert result == pytest.approx(50.0)

    def test_negative_improvement(self) -> None:
        result = _improvement(80.0, 100.0)
        assert result == pytest.approx(-20.0)

    def test_zero_improvement(self) -> None:
        result = _improvement(100.0, 100.0)
        assert result == pytest.approx(0.0)

    def test_fcfs_zero_returns_nan(self) -> None:
        result = _improvement(100.0, 0.0)
        assert math.isnan(result)


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------


class TestGenerateReport:
    """generate_report 测试"""

    def test_report_contains_required_sections(self, tmp_path: Path) -> None:
        results = {
            "config": {
                "seed_list": [42, 179],
                "episodes_per_seed": 3,
                "max_steps": 200,
                "trace_length": 500,
                "trace_seed": 20260722,
                "total_episodes_per_strategy_distribution": 6,
                "ppo_model": "ppo.zip",
                "dqn_model": "dqn.zip",
            },
            "distributions": {
                "in_distribution": {
                    "label": "分布内",
                    "strategies": {
                        "PPO": {
                            "mean_reward": 2500.0,
                            "std_reward": 100.0,
                            "completion_rate": 0.95,
                        },
                        "FCFS": {
                            "mean_reward": 1500.0,
                            "std_reward": 50.0,
                            "completion_rate": 0.90,
                        },
                    },
                },
            },
        }
        report_path = tmp_path / "report.md"
        data_path = tmp_path / "data.json"
        generate_report(results, report_path, data_path)
        content = report_path.read_text(encoding="utf-8")
        assert "留出负载盲测" in content
        assert "实验配置" in content
        assert "PPO 相对 FCFS" in content
        assert "分布外泛化结论" in content

    def test_report_shows_improvement(self, tmp_path: Path) -> None:
        results = {
            "config": {
                "seed_list": [42],
                "episodes_per_seed": 1,
                "max_steps": 200,
                "trace_length": 100,
                "trace_seed": 20260722,
                "total_episodes_per_strategy_distribution": 1,
                "ppo_model": "ppo.zip",
                "dqn_model": "dqn.zip",
            },
            "distributions": {
                "burst": {
                    "label": "突发大任务",
                    "strategies": {
                        "PPO": {
                            "mean_reward": 200.0,
                            "std_reward": 10.0,
                            "completion_rate": 0.85,
                        },
                        "FCFS": {
                            "mean_reward": 100.0,
                            "std_reward": 5.0,
                            "completion_rate": 0.80,
                        },
                    },
                },
            },
        }
        report_path = tmp_path / "report.md"
        data_path = tmp_path / "data.json"
        generate_report(results, report_path, data_path)
        content = report_path.read_text(encoding="utf-8")
        assert "+100.0%" in content
