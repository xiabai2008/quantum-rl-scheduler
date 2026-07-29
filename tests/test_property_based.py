"""
Property-based testing for quantum-rl-scheduler (Issue #520)
使用 Hypothesis 进行基于属性的测试，验证核心不变量

覆盖模块：
- DAG 调度器（依赖约束满足）
- 混合调度器（动作合法性、置信度范围）
- 可解释性模块（特征重要性归一化、非负性）
- 环境 step()（观测维度、奖励有限性、done 类型）
- Jain 公平性指数（值域 (0, 1]）
"""

import os
import sys
import types
import unittest
from typing import Any
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from src.scheduler.dag_scheduler import DAGScheduler, DAGTask
from src.scheduler.env import OBS_DIM, QuantumSchedulingEnv
from src.scheduler.explainability import DecisionExplainer
from src.scheduler.fairness import jain_fairness_index
from src.scheduler.hybrid_scheduler import (
    ACTION_CLASSICAL,
    ACTION_HYBRID,
    ACTION_QUANTUM,
    HybridScheduler,
    RuleEngine,
)


@st.composite
def random_dag_tasks(
    draw: st.DrawFn,
    min_tasks: int = 1,
    max_tasks: int = 8,
) -> list[DAGTask]:
    """
    生成随机合法 DAG 的任务列表。
    依赖边保证无环（仅允许从早生成的任务指向晚生成的任务）。
    """
    n_tasks = draw(st.integers(min_value=min_tasks, max_value=max_tasks))
    task_ids = [f"t{i}" for i in range(n_tasks)]

    tasks: list[DAGTask] = []

    for i, tid in enumerate(task_ids):
        qubits = draw(st.integers(min_value=0, max_value=15))
        est_time = draw(
            st.floats(min_value=0.1, max_value=5.0, allow_nan=False, allow_infinity=False)
        )
        priority = draw(st.integers(min_value=1, max_value=5))
        dependencies: list[str] = []

        if i > 0:
            max_deps = min(i, 3)
            n_deps = draw(st.integers(min_value=0, max_value=max_deps))
            if n_deps > 0:
                possible_deps = task_ids[:i]
                deps = draw(
                    st.lists(
                        st.sampled_from(possible_deps),
                        min_size=n_deps,
                        max_size=n_deps,
                        unique=True,
                    )
                )
                dependencies = list(deps)

        task = DAGTask(
            task_id=tid,
            task_type=draw(st.sampled_from(["quantum", "classical", "hybrid"])),
            qubits_required=qubits,
            estimated_time=est_time,
            priority=priority,
            dependencies=dependencies,
        )
        tasks.append(task)

    return tasks


@st.composite
def random_task_for_hybrid(draw: st.DrawFn) -> Any:
    """生成随机任务对象用于混合调度器测试。"""
    return types.SimpleNamespace(
        task_id=f"task_{draw(st.integers(min_value=0, max_value=9999))}",
        task_type=draw(st.sampled_from(["quantum", "classical", "universal", "hybrid"])),
        qubit_count=draw(st.integers(min_value=0, max_value=300)),
        wait_steps=draw(st.integers(min_value=0, max_value=100)),
        urgency=draw(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)
        ),
        priority=draw(st.integers(min_value=1, max_value=5)),
        execution_time=draw(st.integers(min_value=1, max_value=20)),
    )


@st.composite
def random_non_negative_values(
    draw: st.DrawFn, min_size: int = 1, max_size: int = 30
) -> list[float]:
    """生成非负数值列表（不全为零，避免除零）。"""
    n = draw(st.integers(min_value=min_size, max_value=max_size))
    values = draw(
        st.lists(
            st.floats(min_value=0.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=n,
            max_size=n,
        )
    )
    assume(any(v > 1e-12 for v in values))
    return values


@st.composite
def random_feature_vector(draw: st.DrawFn, dim: int | None = None) -> np.ndarray:
    """生成随机状态特征向量（任意维度，值为有限浮点数）。"""
    if dim is None:
        dim = draw(st.integers(min_value=1, max_value=25))
    values = draw(
        st.lists(
            st.floats(min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False),
            min_size=dim,
            max_size=dim,
        )
    )
    return np.array(values, dtype=np.float64)


class TestDAGSchedulerProperty(unittest.TestCase):
    """DAG 调度器 property-based 测试。"""

    @given(
        tasks=random_dag_tasks(min_tasks=1, max_tasks=6),
        available_qubits=st.integers(min_value=1, max_value=50),
        available_machines=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=50, deadline=None)
    def test_schedule_respects_dependencies(
        self,
        tasks: list[DAGTask],
        available_qubits: int,
        available_machines: int,
    ) -> None:
        """
        Property: 对任意合法 DAG，schedule_with_resources 的结果必须满足依赖约束：
        每个任务的 start_time >= 所有前驱任务的 estimated_finish。
        """
        scheduler = DAGScheduler(tasks=tasks)
        schedule = scheduler.schedule_with_resources(
            available_qubits=available_qubits,
            available_machines=available_machines,
        )

        self.assertEqual(len(schedule), len(tasks))

        schedule_by_id = {item["task_id"]: item for item in schedule}

        for task in tasks:
            tid = task.task_id
            self.assertIn(tid, schedule_by_id)
            task_start = schedule_by_id[tid]["start_time"]
            for dep_id in task.dependencies:
                if dep_id in schedule_by_id:
                    dep_finish = schedule_by_id[dep_id]["estimated_finish"]
                    self.assertGreaterEqual(
                        task_start,
                        dep_finish - 1e-9,
                        f"任务 {tid} 在其依赖 {dep_id} 完成前开始了",
                    )

    @given(tasks=random_dag_tasks(min_tasks=1, max_tasks=6))
    @settings(max_examples=30, deadline=None)
    def test_schedule_result_has_required_fields(self, tasks: list[DAGTask]) -> None:
        """
        Property: 调度结果每项必须包含 task_id, start_time, machine_id, estimated_finish。
        """
        scheduler = DAGScheduler(tasks=tasks)
        schedule = scheduler.schedule_with_resources(available_qubits=20, available_machines=2)

        required_fields = {"task_id", "start_time", "machine_id", "estimated_finish"}
        for item in schedule:
            self.assertEqual(set(item.keys()), required_fields)
            self.assertIsInstance(item["task_id"], str)
            self.assertIsInstance(item["start_time"], float)
            self.assertIsInstance(item["machine_id"], int)
            self.assertIsInstance(item["estimated_finish"], float)
            self.assertGreaterEqual(item["start_time"], 0.0)
            self.assertGreaterEqual(item["estimated_finish"], item["start_time"] - 1e-9)

    @given(tasks=random_dag_tasks(min_tasks=1, max_tasks=8))
    @settings(max_examples=30, deadline=None)
    def test_topological_sort_valid(self, tasks: list[DAGTask]) -> None:
        """
        Property: 拓扑排序中每个任务都出现在其所有前驱之后。
        """
        scheduler = DAGScheduler(tasks=tasks)
        order = scheduler.topological_sort()

        self.assertEqual(len(order), len(tasks))
        self.assertEqual(set(order), {t.task_id for t in tasks})

        position = {tid: idx for idx, tid in enumerate(order)}
        for task in tasks:
            for dep_id in task.dependencies:
                if dep_id in position:
                    self.assertLess(
                        position[dep_id],
                        position[task.task_id],
                        f"依赖 {dep_id} 应排在 {task.task_id} 之前",
                    )


class TestHybridSchedulerProperty(unittest.TestCase):
    """混合调度器 property-based 测试。"""

    @given(
        task=random_task_for_hybrid(),
        available_qubits=st.integers(min_value=1, max_value=500),
        queue_length=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=40, deadline=None)
    def test_rule_engine_returns_valid_action_or_none(
        self, task: Any, available_qubits: int, queue_length: int
    ) -> None:
        """
        Property: RuleEngine.evaluate 返回值必须是 ACTION_CLASSICAL/ACTION_QUANTUM/ACTION_HYBRID 或 None。
        """
        engine = RuleEngine()
        ctx = {"available_qubits": available_qubits, "queue_length": queue_length}
        result = engine.evaluate(task, ctx)

        valid_actions = {ACTION_CLASSICAL, ACTION_QUANTUM, ACTION_HYBRID, None}
        self.assertIn(result, valid_actions)

    @given(
        task=random_task_for_hybrid(),
        available_qubits=st.integers(min_value=1, max_value=500),
        queue_length=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=40, deadline=None)
    def test_hybrid_scheduler_decide_returns_valid_structure(
        self, task: Any, available_qubits: int, queue_length: int
    ) -> None:
        """
        Property: HybridScheduler.decide 返回值必须包含 action/source/confidence/reason，
        action 为合法动作，confidence 在 [0, 1] 范围内。
        """
        scheduler = HybridScheduler(rl_agent=None, fallback_to_rule=True)
        ctx = {"available_qubits": available_qubits, "queue_length": queue_length}
        result = scheduler.decide(task, state=None, context=ctx)

        self.assertIn("action", result)
        self.assertIn("source", result)
        self.assertIn("confidence", result)
        self.assertIn("reason", result)

        valid_actions = {ACTION_CLASSICAL, ACTION_QUANTUM, ACTION_HYBRID}
        self.assertIn(result["action"], valid_actions)
        self.assertGreaterEqual(result["confidence"], 0.0 - 1e-9)
        self.assertLessEqual(result["confidence"], 1.0 + 1e-9)
        self.assertIsInstance(result["source"], str)

    @given(
        n_tasks=st.integers(min_value=1, max_value=8),
    )
    @settings(max_examples=20, deadline=None)
    def test_decide_batch_returns_correct_length(self, n_tasks: int) -> None:
        """
        Property: decide_batch 返回结果列表长度必须等于输入任务列表长度。
        """
        rl_agent = Mock()
        rl_agent.predict = Mock(return_value=ACTION_QUANTUM)
        scheduler = HybridScheduler(rl_agent=rl_agent)

        tasks = [
            types.SimpleNamespace(
                task_id=f"t{i}",
                task_type="quantum",
                qubit_count=5,
                wait_steps=0,
                urgency=0.5,
                priority=3,
                execution_time=3,
            )
            for i in range(n_tasks)
        ]
        states = [np.zeros(OBS_DIM, dtype=np.float32) for _ in range(n_tasks)]
        ctx = {"available_qubits": 100, "queue_length": 5}

        results = scheduler.decide_batch(tasks, states=states, context=ctx)
        self.assertEqual(len(results), n_tasks)
        valid_actions = {ACTION_CLASSICAL, ACTION_QUANTUM, ACTION_HYBRID}
        for r in results:
            self.assertIn("action", r)
            self.assertIn(r["action"], valid_actions)


class TestExplainabilityProperty(unittest.TestCase):
    """可解释性模块 property-based 测试。"""

    @given(
        state=random_feature_vector(dim=17),
        action=st.integers(min_value=0, max_value=2),
    )
    @settings(max_examples=50, deadline=None)
    def test_feature_contributions_sum_to_one(self, state: np.ndarray, action: int) -> None:
        """
        Property: explain() 返回的 feature_contributions 值之和应≈1（归一化）。
        """
        explainer = DecisionExplainer()
        rec = explainer.explain(state, action=action)
        total = sum(rec.feature_contributions.values())
        self.assertAlmostEqual(total, 1.0, places=4)

    @given(
        state=random_feature_vector(dim=17),
        action=st.integers(min_value=0, max_value=2),
    )
    @settings(max_examples=50, deadline=None)
    def test_feature_contributions_non_negative(self, state: np.ndarray, action: int) -> None:
        """
        Property: 默认 heuristic 模式下所有特征贡献度值必须非负。
        """
        explainer = DecisionExplainer(method="heuristic")
        rec = explainer.explain(state, action=action)
        for name, value in rec.feature_contributions.items():
            self.assertGreaterEqual(
                value,
                -1e-9,
                f"特征 {name} 的贡献度 {value} 为负值",
            )

    @given(
        dim=st.integers(min_value=2, max_value=20),
        action=st.integers(min_value=0, max_value=2),
    )
    @settings(max_examples=30, deadline=None)
    def test_arbitrary_dimension_state_handled(self, dim: int, action: int) -> None:
        """
        Property: 任意维度（>=1）的状态向量都不应崩溃，贡献度数量匹配且归一化。
        """
        state = np.random.randn(dim)
        feature_names = [f"f{i}" for i in range(dim)]
        explainer = DecisionExplainer(feature_names=feature_names)
        rec = explainer.explain(state, action=action)

        self.assertEqual(len(rec.feature_contributions), dim)
        total = sum(rec.feature_contributions.values())
        self.assertAlmostEqual(total, 1.0, places=4)


class TestEnvStepProperty(unittest.TestCase):
    """环境 step() property-based 测试。"""

    @given(
        action=st.integers(min_value=0, max_value=2),
        seed=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=40, deadline=None)
    def test_step_returns_correct_obs_dimension(self, action: int, seed: int) -> None:
        """
        Property: 对任意有效动作 (0,1,2)，step() 返回的 obs 维度应为 OBS_DIM。
        """
        env = QuantumSchedulingEnv(max_steps=50, seed=seed)
        obs, _info = env.reset(seed=seed)
        self.assertEqual(obs.shape, (OBS_DIM,))

        next_obs, _reward, _terminated, _truncated, _info = env.step(action)
        self.assertEqual(next_obs.shape, (OBS_DIM,))
        self.assertEqual(next_obs.dtype, np.float32)

    @given(
        action=st.integers(min_value=0, max_value=2),
        seed=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=40, deadline=None)
    def test_step_reward_is_finite_float(self, action: int, seed: int) -> None:
        """
        Property: reward 必须为有限浮点数（非 NaN、非 Inf）。
        """
        env = QuantumSchedulingEnv(max_steps=50, seed=seed)
        env.reset(seed=seed)
        _obs, reward, _terminated, _truncated, _info = env.step(action)

        self.assertIsInstance(float(reward), float)
        self.assertTrue(np.isfinite(reward), f"reward={reward} 不是有限值")

    @given(
        action=st.integers(min_value=0, max_value=2),
        seed=st.integers(min_value=0, max_value=500),
    )
    @settings(max_examples=40, deadline=None)
    def test_step_done_is_bool(self, action: int, seed: int) -> None:
        """
        Property: terminated 和 truncated 必须为 bool 类型。
        """
        env = QuantumSchedulingEnv(max_steps=50, seed=seed)
        env.reset(seed=seed)
        _obs, _reward, terminated, truncated, info = env.step(action)

        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIsInstance(info, dict)

    @given(seed=st.integers(min_value=0, max_value=200))
    @settings(max_examples=15, deadline=None)
    def test_episode_terminates_within_max_steps(self, seed: int) -> None:
        """
        Property: episode 必然在 max_steps 内终止（terminated 或 truncated）。
        """
        max_steps = 30
        env = QuantumSchedulingEnv(max_steps=max_steps, seed=seed)
        env.reset(seed=seed)

        done_flag = False
        for _ in range(max_steps + 10):
            _obs, _reward, terminated, truncated, _info = env.step(0)
            if terminated or truncated:
                done_flag = True
                break

        self.assertTrue(done_flag, "Episode 未能在 max_steps 内终止")


class TestJainFairnessProperty(unittest.TestCase):
    """Jain 公平性指数 property-based 测试。"""

    @given(values=random_non_negative_values(min_size=1, max_size=30))
    @settings(max_examples=100, deadline=None)
    def test_jain_index_in_range(self, values: list[float]) -> None:
        """
        Property: 对任意非负等待时间列表（不全为零），jain_fairness_index 返回值在 (0, 1] 范围内。
        """
        fi = jain_fairness_index(values)
        self.assertGreater(fi, 0.0, f"Jain index={fi} 应 > 0")
        self.assertLessEqual(fi, 1.0 + 1e-9, f"Jain index={fi} 应 <= 1")

    @given(n=st.integers(min_value=1, max_value=30))
    @settings(max_examples=50, deadline=None)
    def test_jain_index_perfect_fairness_is_one(self, n: int) -> None:
        """
        Property: 所有值相等时，Jain index 应为 1.0（完全公平）。
        """
        v = 42.0
        values = [v for _ in range(n)]
        fi = jain_fairness_index(values)
        self.assertAlmostEqual(fi, 1.0, places=9)

    @given(
        n=st.integers(min_value=2, max_value=15),
        large_value=st.floats(
            min_value=100.0, max_value=10000.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_jain_index_extreme_unfairness_approaches_1_over_n(
        self, n: int, large_value: float
    ) -> None:
        """
        Property: 一个值极大、其余接近0时，Jain index 应接近 1/n（极端不公平）。
        """
        small_value = 0.001
        values = [small_value for _ in range(n - 1)] + [large_value]
        fi = jain_fairness_index(values)
        self.assertLess(fi, 2.0 / n + 0.1)

    @given(values=random_non_negative_values(min_size=1, max_size=20))
    @settings(max_examples=50, deadline=None)
    def test_jain_index_scale_invariant(self, values: list[float]) -> None:
        """
        Property: Jain index 具有尺度不变性：所有值乘以正常数 k 后结果不变。
        """
        k = 3.7
        fi_original = jain_fairness_index(values)
        fi_scaled = jain_fairness_index([v * k for v in values])
        self.assertAlmostEqual(fi_original, fi_scaled, places=9)


if __name__ == "__main__":
    unittest.main()
