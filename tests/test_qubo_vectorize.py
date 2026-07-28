"""测试 build_scheduling_qubo 向量化实现的正确性。

验证：
1. QUBO 矩阵形状正确
2. QUBO 矩阵对称
3. 容量约束和分配约束的系数正确
4. 变量数 >512 时抛出 ValueError
5. 与朴素循环实现结果一致
"""

import sys
import unittest

import numpy as np

sys.path.insert(0, ".")

from src.scheduler.dag_scheduler import DAGTask, DAGScheduler


def build_qubo_naive(
    tasks: list[DAGTask],
    time_slots: int,
    max_qubits: int,
    penalty_weight: float = 1.0,
) -> np.ndarray:
    """朴素四层循环实现（用于验证向量化结果的正确性）。"""
    n_tasks = len(tasks)
    n_vars = n_tasks * time_slots
    qubits = np.array([max(0, t.qubits_required) for t in tasks], dtype=np.float64)
    C = float(max_qubits)

    Q = np.zeros((n_vars, n_vars), dtype=np.float64)

    for i in range(n_tasks):
        qi = qubits[i]
        for j in range(n_tasks):
            qj = qubits[j]
            for t in range(time_slots):
                for s in range(time_slots):
                    idx_i = i * time_slots + t
                    idx_j = j * time_slots + s

                    if t == s:
                        if i == j:
                            Q[idx_i, idx_j] += penalty_weight * (qi * qi - 2.0 * C * qi)
                        else:
                            Q[idx_i, idx_j] += penalty_weight * qi * qj

                    if i == j:
                        if t == s:
                            Q[idx_i, idx_j] += -penalty_weight
                        else:
                            Q[idx_i, idx_j] += penalty_weight

    Q = (Q + Q.T) * 0.5
    return Q


class TestBuildSchedulingQubo(unittest.TestCase):
    def setUp(self):
        self.tasks = [
            DAGTask("t1", qubits_required=10),
            DAGTask("t2", qubits_required=20),
            DAGTask("t3", qubits_required=5),
        ]
        self.scheduler = DAGScheduler(self.tasks, max_qubits=30)

    def test_shape(self):
        Q = self.scheduler.build_scheduling_qubo(self.tasks, time_slots=4)
        self.assertEqual(Q.shape, (12, 12))

    def test_symmetric(self):
        Q = self.scheduler.build_scheduling_qubo(self.tasks, time_slots=4)
        self.assertTrue(np.allclose(Q, Q.T, atol=1e-12))

    def test_value_error_on_too_many_vars(self):
        tasks = [DAGTask(f"t{i}") for i in range(200)]
        s = DAGScheduler(tasks)
        with self.assertRaises(ValueError) as ctx:
            s.build_scheduling_qubo(tasks, time_slots=3)
        self.assertIn("512", str(ctx.exception))

    def test_value_error_at_boundary(self):
        tasks = [DAGTask(f"t{i}") for i in range(128)]
        s = DAGScheduler(tasks)
        Q = s.build_scheduling_qubo(tasks, time_slots=4)
        self.assertEqual(Q.shape, (512, 512))

    def test_vs_naive(self):
        """向量化结果应与朴素循环实现完全一致。"""
        Q_vec = self.scheduler.build_scheduling_qubo(self.tasks, time_slots=3)
        Q_naive = build_qubo_naive(self.tasks, time_slots=3, max_qubits=30)
        self.assertTrue(np.allclose(Q_vec, Q_naive, atol=1e-12))

    def test_vs_naive_different_params(self):
        """测试不同参数下向量化与朴素实现的一致性。"""
        tasks = [
            DAGTask("a", qubits_required=8),
            DAGTask("b", qubits_required=15),
        ]
        s = DAGScheduler(tasks, max_qubits=20)
        Q_vec = s.build_scheduling_qubo(tasks, time_slots=5, penalty_weight=2.0)
        Q_naive = build_qubo_naive(tasks, time_slots=5, max_qubits=20, penalty_weight=2.0)
        self.assertTrue(np.allclose(Q_vec, Q_naive, atol=1e-12))

    def test_diagonal_entries(self):
        """验证对角线条目：容量约束贡献 + 分配约束贡献。"""
        tasks = [DAGTask("x", qubits_required=10)]
        s = DAGScheduler(tasks, max_qubits=20)
        Q = s.build_scheduling_qubo(tasks, time_slots=2)

        q = 10.0
        C = 20.0
        expected_diag_0 = q * q - 2.0 * C * q + (-1.0)
        self.assertAlmostEqual(Q[0, 0], expected_diag_0, places=10)

    def test_default_max_qubits(self):
        """不传 max_qubits 时应使用 self.max_qubits。"""
        s = DAGScheduler(self.tasks, max_qubits=100)
        Q = s.build_scheduling_qubo(self.tasks, time_slots=2)
        Q2 = s.build_scheduling_qubo(self.tasks, time_slots=2, max_qubits=100)
        self.assertTrue(np.allclose(Q, Q2, atol=1e-12))

    def test_empty_tasks(self):
        """空任务列表应产生 0×0 矩阵。"""
        s = DAGScheduler([], max_qubits=30)
        Q = s.build_scheduling_qubo([], time_slots=3)
        self.assertEqual(Q.shape, (0, 0))

    def test_single_task_single_slot(self):
        """单任务单时间槽。"""
        tasks = [DAGTask("only", qubits_required=5)]
        s = DAGScheduler(tasks, max_qubits=10)
        Q = s.build_scheduling_qubo(tasks, time_slots=1)
        self.assertEqual(Q.shape, (1, 1))
        self.assertAlmostEqual(Q[0, 0], 25.0 - 100.0 - 1.0, places=10)


if __name__ == "__main__":
    unittest.main(verbosity=2)