"""Property-based 测试 — 使用 Hypothesis 自动生成测试输入验证不变量

覆盖不变量：
    - 状态向量始终在 [0,1] 且形状为 (10,)
    - step() 返回值形状/类型始终正确
    - 奖励始终为有限浮点数
    - 任意种子下 episode 必然在 max_steps 内终止
    - 解析器对任意合法 QASM 不崩溃
    - normalize_vector 幂等且保序
"""

import os
import sys
import unittest

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.env import OBS_DIM, QuantumSchedulingEnv
from src.scheduler.parser import LegacyTaskParser
from src.utils.helpers import normalize_vector


def generate_qasm(num_qubits: int, num_gates: int) -> str:
    """生成合法的 QASM 量子电路描述字符串。"""
    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{num_qubits}];",
        f"creg c[{num_qubits}];",
    ]
    gates = ("h", "x", "y", "z")
    for i in range(num_gates):
        gate = gates[i % len(gates)]
        q = i % max(num_qubits, 1)
        lines.append(f"{gate} q[{q}];")
    if num_qubits > 0:
        lines.append("measure q[0] -> c[0];")
    return "\n".join(lines)


class TestEnvProperty(unittest.TestCase):
    """调度环境属性测试"""

    @given(seed=st.integers(min_value=0, max_value=10000))
    @settings(max_examples=50, deadline=None)
    def test_observation_always_in_range(self, seed):
        """Property: 任意种子下，状态向量始终在 [0,1] 范围内"""
        env = QuantumSchedulingEnv(max_steps=100, seed=seed)
        obs, _ = env.reset(seed=seed)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)

    @given(action=st.integers(min_value=0, max_value=2))
    @settings(max_examples=30, deadline=None)
    def test_step_always_returns_valid_shapes(self, action):
        """Property: 任意合法动作下，step() 返回值形状正确"""
        env = QuantumSchedulingEnv(max_steps=100, seed=42)
        env.reset(seed=42)
        obs, _reward, terminated, truncated, info = env.step(action)
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    @given(
        action=st.integers(min_value=0, max_value=2),
        seed=st.integers(min_value=0, max_value=1000),
    )
    @settings(max_examples=30, deadline=None)
    def test_step_reward_is_finite(self, action, seed):
        """Property: 任意动作下，奖励始终为有限浮点数"""
        env = QuantumSchedulingEnv(max_steps=100, seed=seed)
        env.reset(seed=seed)
        _obs, reward, _terminated, _truncated, _info = env.step(action)
        assert np.isfinite(reward)

    @given(seed=st.integers(min_value=0, max_value=1000))
    @settings(max_examples=20, deadline=None)
    def test_env_episode_always_terminates(self, seed):
        """Property: 任意种子下，episode 在 max_steps 内必然终止（terminated 或 truncated）"""
        env = QuantumSchedulingEnv(max_steps=50, seed=seed)
        env.reset(seed=seed)
        done = False
        for _ in range(60):
            _obs, _reward, terminated, truncated, _info = env.step(0)
            if terminated or truncated:
                done = True
                break
        assert done


class TestParserProperty(unittest.TestCase):
    """任务解析器属性测试"""

    @given(
        num_qubits=st.integers(min_value=1, max_value=20),
        num_gates=st.integers(min_value=0, max_value=50),
    )
    @settings(max_examples=20, deadline=None)
    def test_parser_handles_arbitrary_qasm(self, num_qubits, num_gates):
        """Property: 任意合法 QASM 字符串都能被正确解析（不崩溃）"""
        qasm = generate_qasm(num_qubits, num_gates)
        parser = LegacyTaskParser()
        result = parser.parse(qasm, format="qasm")
        assert result is not None
        assert result.task_type == "quantum"
        assert result.qubit_count == num_qubits


class TestUtilsProperty(unittest.TestCase):
    """工具函数属性测试"""

    @given(
        v=st.lists(
            st.floats(min_value=-1000.0, max_value=1000.0, allow_nan=False, allow_infinity=False),
            min_size=2,
            max_size=20,
        )
    )
    @settings(max_examples=30, deadline=None)
    def test_normalize_vector_idempotent(self, v):
        """Property: normalize_vector 幂等且保序"""
        norm1 = normalize_vector(v)
        norm2 = normalize_vector(norm1)
        assert len(norm1) == len(v)
        assert len(norm2) == len(v)
        # 幂等性：二次归一化结果与一次归一化一致
        for a, b in zip(norm1, norm2, strict=False):
            assert abs(a - b) < 1e-6
        # 保序性：当原始值差异足够大（未被归一化折叠为 0.5）时，
        # 最大值/最小值索引保持一致。注意：当最大值与次大值（或最小值与次小值）
        # 差异极小时，浮点精度可能导致归一化后并列，此时跳过索引校验。
        v_arr = np.array(v)
        if float(np.max(v_arr)) - float(np.min(v_arr)) >= 1e-10:
            sorted_desc = np.sort(v_arr)[::-1]
            sorted_asc = np.sort(v_arr)
            if sorted_desc[0] - sorted_desc[1] >= 1e-10:
                assert int(np.argmax(v)) == int(np.argmax(norm1))
            if sorted_asc[1] - sorted_asc[0] >= 1e-10:
                assert int(np.argmin(v)) == int(np.argmin(norm1))


# ============================================================
# Issue #520: 补充 DAG/混合/可解释性模块 property-based 测试
# ============================================================
from src.scheduler.dag_scheduler import DAGScheduler, DAGTask
from src.scheduler.explainability import DecisionExplainer
from src.scheduler.hybrid_scheduler import HybridScheduler


def _generate_dag_tasks(task_ids: list[str]) -> list[DAGTask]:
    """根据任务 ID 列表生成无环 DAG（依赖只引用前序任务）。"""
    tasks: list[DAGTask] = []
    for i, tid in enumerate(task_ids):
        deps = [task_ids[j] for j in range(i) if (i + j) % 3 == 0]
        tasks.append(DAGTask(task_id=tid, dependencies=deps))
    return tasks


class TestDAGSchedulerProperty(unittest.TestCase):
    """DAG 调度器属性测试。"""

    @given(
        n_tasks=st.integers(min_value=1, max_value=20),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=30, deadline=None)
    def test_topological_sort_always_valid(self, n_tasks, seed):
        """Property: 拓扑排序结果中每个任务的依赖都在其之前出现。"""
        task_ids = [f"task_{i}" for i in range(n_tasks)]
        tasks = _generate_dag_tasks(task_ids)
        scheduler = DAGScheduler(tasks=tasks)
        order = scheduler.topological_sort()

        assert len(order) == n_tasks
        seen: set[str] = set()
        task_map = {t.task_id: t for t in tasks}
        for tid in order:
            for dep in task_map[tid].dependencies:
                assert dep in seen, f"依赖 {dep} 未在 {tid} 之前出现"
            seen.add(tid)

    @given(
        n_tasks=st.integers(min_value=1, max_value=15),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=20, deadline=None)
    def test_topological_sort_contains_all_tasks(self, n_tasks, seed):
        """Property: 拓扑排序结果包含所有任务且无重复。"""
        task_ids = [f"t{i}" for i in range(n_tasks)]
        tasks = _generate_dag_tasks(task_ids)
        scheduler = DAGScheduler(tasks=tasks)
        order = scheduler.topological_sort()

        assert set(order) == set(task_ids)
        assert len(order) == len(set(order))


class TestHybridSchedulerProperty(unittest.TestCase):
    """混合调度器属性测试。"""

    @given(
        state=st.lists(
            st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=10,
            max_size=10,
        ),
    )
    @settings(max_examples=30, deadline=None)
    def test_decide_always_returns_valid_action(self, state):
        """Property: decide 始终返回 {0, 1, 2} 中的合法动作。"""
        scheduler = HybridScheduler()
        task = type("Task", (), {"task_type": "quantum", "qubits_required": 5})()
        state_arr = np.array(state, dtype=np.float32)
        result = scheduler.decide(task, state=state_arr)
        assert result["action"] in (0, 1, 2)


class TestExplainabilityProperty(unittest.TestCase):
    """可解释性模块属性测试。"""

    @given(
        state=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=14,
            max_size=14,
        ),
        action=st.integers(min_value=0, max_value=2),
    )
    @settings(max_examples=30, deadline=None)
    def test_contributions_sum_approximately_one(self, state, action):
        """Property: 特征贡献度归一化，和约为 1.0。"""
        explainer = DecisionExplainer()
        state_arr = np.array(state, dtype=np.float32)
        record = explainer.explain(state_arr, action)
        total = sum(abs(v) for v in record.feature_contributions.values())
        assert abs(total - 1.0) < 0.15, f"贡献度和 {total} 偏离 1.0"

    @given(
        state=st.lists(
            st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
            min_size=14,
            max_size=14,
        ),
        action=st.integers(min_value=0, max_value=2),
    )
    @settings(max_examples=20, deadline=None)
    def test_explain_returns_non_empty_contributions(self, state, action):
        """Property: explain 始终返回非空贡献度字典。"""
        explainer = DecisionExplainer()
        state_arr = np.array(state, dtype=np.float32)
        record = explainer.explain(state_arr, action)
        assert len(record.feature_contributions) > 0
