"""
QAOA 调度器基线单元测试
Unit Tests for QAOAScheduler (Issue #599)

测试覆盖：
- TestQAOASchedulerBasic        : 基本功能（空列表、单任务、多任务选择）
- TestQAOAQUBOConstruction      : QUBO 矩阵构建正确性
- TestQAOAIsingConversion       : QUBO→Ising 转换正确性
- TestQAOAProbabilityComputation: 概率分布计算正确性
- TestQAOASolutionDecoding      : 解码逻辑正确性
- TestQAOABackwardCompatibility : 向后兼容（BaselineScheduler 接口）
- TestQAOAIntegration           : 集成测试（run_baseline_comparison）
- TestQAOAEdgeCases             : 边界条件
- TestQAOAPerformance           : 性能/超时保护
"""

import unittest

import numpy as np

from src.scheduler.baselines import (
    BaselineScheduler,
    QAOAScheduler,
    get_all_baseline_schedulers,
    run_baseline_comparison,
)


# ============================================================
# 测试辅助
# ============================================================
def _make_task(
    task_id: str,
    priority: int = 3,
    estimated_time: float = 10.0,
    arrival_time: float = 0.0,
    deadline: float | None = None,
    qubit_count: int = 4,
) -> dict:
    """构造测试用任务字典。"""
    task: dict = {
        "task_id": task_id,
        "priority": priority,
        "estimated_time": estimated_time,
        "arrival_time": arrival_time,
        "qubit_count": qubit_count,
    }
    if deadline is not None:
        task["deadline"] = deadline
    return task


_EMPTY_RESOURCES: dict = {"qubits": 20, "classical_load": 0.0}


# ============================================================
# TestQAOASchedulerBasic
# ============================================================
class TestQAOASchedulerBasic(unittest.TestCase):
    """测试 QAOA 调度器基本功能。"""

    def setUp(self):
        self.scheduler = QAOAScheduler(p=1, grid_size=8, max_qubits=8, seed=42)

    def test_is_baseline_scheduler(self):
        """QAOAScheduler 应继承 BaselineScheduler。"""
        self.assertIsInstance(self.scheduler, BaselineScheduler)

    def test_name_is_qaoa(self):
        """策略名称应为 'QAOA'。"""
        self.assertEqual(self.scheduler.name, "QAOA")

    def test_empty_tasks_returns_negative_one(self):
        """空任务列表应返回 -1。"""
        result = self.scheduler.select_action([], _EMPTY_RESOURCES)
        self.assertEqual(result, -1)

    def test_single_task_returns_zero(self):
        """单个任务应返回索引 0。"""
        tasks = [_make_task("T1", priority=3)]
        result = self.scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(result, 0)

    def test_multiple_tasks_returns_valid_index(self):
        """多任务应返回有效索引。"""
        tasks = [
            _make_task("T1", priority=1, estimated_time=20.0),
            _make_task("T2", priority=5, estimated_time=5.0),
            _make_task("T3", priority=3, estimated_time=10.0),
        ]
        result = self.scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertIn(result, [0, 1, 2])

    def test_prefers_high_priority_task(self):
        """应倾向于选择高优先级任务。"""
        tasks = [
            _make_task("T1", priority=1, estimated_time=5.0),
            _make_task("T2", priority=5, estimated_time=5.0),
            _make_task("T3", priority=1, estimated_time=5.0),
        ]
        results = []
        for _ in range(5):
            results.append(self.scheduler.select_action(tasks, _EMPTY_RESOURCES))
        # 至少有一次选择了高优先级任务（索引 1）
        self.assertIn(1, results)

    def test_reset_does_not_crash(self):
        """reset() 应正常执行。"""
        self.scheduler.reset()


# ============================================================
# TestQAOAQUBOConstruction
# ============================================================
class TestQAOAQUBOConstruction(unittest.TestCase):
    """测试 QUBO 矩阵构建。"""

    def setUp(self):
        self.scheduler = QAOAScheduler(p=1, grid_size=5, max_qubits=8, seed=42)

    def test_qubo_shape(self):
        """QUBO 矩阵应为 n x n。"""
        tasks = [_make_task(f"T{i}") for i in range(4)]
        qubo = self.scheduler._build_assignment_qubo(tasks, _EMPTY_RESOURCES)
        self.assertEqual(qubo.shape, (4, 4))

    def test_qubo_symmetric(self):
        """QUBO 矩阵应对称。"""
        tasks = [_make_task(f"T{i}", priority=i + 1) for i in range(3)]
        qubo = self.scheduler._build_assignment_qubo(tasks, _EMPTY_RESOURCES)
        np.testing.assert_array_almost_equal(qubo, qubo.T)

    def test_qubo_diagonal_contains_objective(self):
        """对角线应包含目标成本和约束惩罚。"""
        tasks = [
            _make_task("T1", priority=5, estimated_time=5.0),
            _make_task("T2", priority=1, estimated_time=20.0),
        ]
        qubo = self.scheduler._build_assignment_qubo(tasks, _EMPTY_RESOURCES)
        # 高优先级任务的对角线值应更低（成本更低）
        self.assertLess(qubo[0, 0], qubo[1, 1])

    def test_qubo_off_diagonal_is_penalty(self):
        """非对角线应为约束惩罚值。"""
        tasks = [_make_task("T1"), _make_task("T2")]
        qubo = self.scheduler._build_assignment_qubo(tasks, _EMPTY_RESOURCES)
        n = len(tasks)
        expected_penalty = 10.0 * max(1, n)
        self.assertAlmostEqual(qubo[0, 1], expected_penalty)
        self.assertAlmostEqual(qubo[1, 0], expected_penalty)

    def test_qubo_resource_penalty(self):
        """资源不足时应有额外惩罚。"""
        tasks = [_make_task("T1", qubit_count=30)]
        resources = {"qubits": 10}
        qubo = self.scheduler._build_assignment_qubo(tasks, resources)
        # 对角线应包含资源不匹配惩罚
        # penalty = max(0, 30-10) * 5.0 = 100
        # 高优先级任务不匹配时对角线应更高
        tasks2 = [_make_task("T1", qubit_count=5)]
        qubo2 = self.scheduler._build_assignment_qubo(tasks2, resources)
        self.assertGreater(qubo[0, 0], qubo2[0, 0])

    def test_qubo_stored_after_select(self):
        """select_action 后应存储 QUBO。"""
        tasks = [_make_task("T1"), _make_task("T2")]
        self.scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertIsNotNone(self.scheduler._last_qubo)
        self.assertEqual(self.scheduler._last_qubo.shape, (2, 2))


# ============================================================
# TestQAOAIsingConversion
# ============================================================
class TestQAOAIsingConversion(unittest.TestCase):
    """测试 QUBO → Ising 转换。"""

    def setUp(self):
        self.scheduler = QAOAScheduler(p=1, grid_size=5, max_qubits=8, seed=42)

    def test_ising_dimensions(self):
        """Ising 参数维度应正确。"""
        qubo = np.array([[-1.0, 5.0], [5.0, -2.0]])
        h, J = self.scheduler._qubo_to_ising(qubo)
        self.assertEqual(h.shape, (2,))
        self.assertEqual(J.shape, (2, 2))

    def test_ising_h_values(self):
        """h 值应正确计算。"""
        qubo = np.array([[-4.0, 0.0], [0.0, -4.0]])
        h, _J = self.scheduler._qubo_to_ising(qubo)
        # h[i] = -Q[i,i]/2 - sum_{j!=i} Q[i,j]/4
        # h[0] = -(-4)/2 - 0/4 = 2.0
        # h[1] = -(-4)/2 - 0/4 = 2.0
        self.assertAlmostEqual(h[0], 2.0)
        self.assertAlmostEqual(h[1], 2.0)

    def test_ising_j_values(self):
        """J 值应正确计算。"""
        qubo = np.array([[0.0, 8.0], [8.0, 0.0]])
        _h, J = self.scheduler._qubo_to_ising(qubo)
        # J[i,j] = Q[i,j] / 4 = 8/4 = 2.0
        self.assertAlmostEqual(J[0, 1], 2.0)
        self.assertAlmostEqual(J[1, 0], 2.0)


# ============================================================
# TestQAOAProbabilityComputation
# ============================================================
class TestQAOAProbabilityComputation(unittest.TestCase):
    """测试 QAOA 概率分布计算。"""

    def setUp(self):
        self.scheduler = QAOAScheduler(p=1, grid_size=5, max_qubits=8, seed=42)

    def test_probabilities_sum_to_one(self):
        """概率总和应为 1。"""
        n = 3
        N = 2**n
        all_bits = np.array([[(x >> i) & 1 for i in range(n)] for x in range(N)], dtype=np.float64)
        cost_phases = np.ones(N, dtype=np.complex128)
        probs = self.scheduler._compute_probabilities_fast(cost_phases, all_bits, 0.5, n, N)
        self.assertAlmostEqual(float(np.sum(probs)), 1.0, places=6)

    def test_probabilities_non_negative(self):
        """概率应非负。"""
        n = 3
        N = 2**n
        all_bits = np.array([[(x >> i) & 1 for i in range(n)] for x in range(N)], dtype=np.float64)
        cost_phases = np.exp(-1j * 0.5 * np.random.randn(N))
        probs = self.scheduler._compute_probabilities_fast(cost_phases, all_bits, 0.3, n, N)
        self.assertTrue(np.all(probs >= -1e-15))

    def test_beta_zero_uniform(self):
        """beta=0 时（无混合），应接近均匀分布。"""
        n = 2
        N = 4
        all_bits = np.array([[(x >> i) & 1 for i in range(n)] for x in range(N)], dtype=np.float64)
        cost_phases = np.ones(N, dtype=np.complex128)
        probs = self.scheduler._compute_probabilities_fast(cost_phases, all_bits, 0.001, n, N)
        # beta≈0 时 mixer 退化为单位矩阵，|+>^n 保持均匀
        np.testing.assert_array_almost_equal(probs, np.ones(N) / N, decimal=2)


# ============================================================
# TestQAOASolutionDecoding
# ============================================================
class TestQAOASolutionDecoding(unittest.TestCase):
    """测试 QAOA 解码逻辑。"""

    def setUp(self):
        self.scheduler = QAOAScheduler(p=1, grid_size=5, max_qubits=8, seed=42)

    def test_empty_solution(self):
        """空解应返回 -1。"""
        result = self.scheduler._decode_assignment(np.array([], dtype=np.int8), 0)
        self.assertEqual(result, -1)

    def test_single_one(self):
        """单个 1 应返回对应索引。"""
        solution = np.array([0, 0, 1, 0], dtype=np.int8)
        result = self.scheduler._decode_assignment(solution, 4)
        self.assertEqual(result, 2)

    def test_all_zeros(self):
        """全 0 解应返回 -1。"""
        solution = np.array([0, 0, 0], dtype=np.int8)
        result = self.scheduler._decode_assignment(solution, 3)
        self.assertEqual(result, -1)

    def test_multiple_ones_returns_first(self):
        """多个 1 应返回第一个。"""
        solution = np.array([1, 0, 1, 0], dtype=np.int8)
        result = self.scheduler._decode_assignment(solution, 4)
        self.assertEqual(result, 0)

    def test_greedy_solution(self):
        """贪心回退应选择对角线最小值。"""
        qubo = np.array([[-5.0, 0.0], [0.0, -2.0]])
        solution = self.scheduler._greedy_solution(qubo)
        self.assertEqual(int(np.argmax(solution)), 0)


# ============================================================
# TestQAOABackwardCompatibility
# ============================================================
class TestQAOABackwardCompatibility(unittest.TestCase):
    """测试向后兼容性。"""

    def test_in_get_all_baseline_schedulers(self):
        """QAOAScheduler 应在 get_all_baseline_schedulers 中。"""
        schedulers = get_all_baseline_schedulers()
        names = [s.name for s in schedulers]
        self.assertIn("QAOA", names)

    def test_repr(self):
        """__repr__ 应包含类名。"""
        scheduler = QAOAScheduler()
        self.assertIn("QAOAScheduler", repr(scheduler))

    def test_default_p_is_1(self):
        """默认 p 应为 1。"""
        scheduler = QAOAScheduler()
        self.assertEqual(scheduler.p, 1)

    def test_default_grid_size(self):
        """默认 grid_size 应为 10。"""
        scheduler = QAOAScheduler()
        self.assertEqual(scheduler.grid_size, 10)

    def test_default_max_qubits(self):
        """默认 max_qubits 应为 10。"""
        scheduler = QAOAScheduler()
        self.assertEqual(scheduler.max_qubits, 10)


# ============================================================
# TestQAOAIntegration
# ============================================================
class TestQAOAIntegration(unittest.TestCase):
    """集成测试。"""

    def test_run_baseline_comparison_includes_qaoa(self):
        """run_baseline_comparison 应包含 QAOA 结果。"""
        tasks = [_make_task(f"T{i}", priority=i % 5 + 1) for i in range(5)]
        results = run_baseline_comparison(tasks, num_steps=10)
        self.assertIn("QAOA", results)
        self.assertIn("total_reward", results["QAOA"])
        self.assertIn("completed_tasks", results["QAOA"])

    def test_qaoa_completes_all_tasks(self):
        """QAOA 应能完成所有任务。"""
        tasks = [_make_task(f"T{i}") for i in range(3)]
        results = run_baseline_comparison(tasks, num_steps=10)
        self.assertEqual(results["QAOA"]["completed_tasks"], 3)

    def test_qaoa_reward_positive(self):
        """QAOA 奖励应为正值。"""
        tasks = [_make_task(f"T{i}", priority=3) for i in range(3)]
        results = run_baseline_comparison(tasks, num_steps=10)
        self.assertGreater(results["QAOA"]["total_reward"], 0)

    def test_qaoa_with_comparison_mode(self):
        """结果应包含 comparison_mode 字段。"""
        tasks = [_make_task("T1")]
        results = run_baseline_comparison(tasks, num_steps=5)
        self.assertEqual(results["QAOA"]["comparison_mode"], "legacy")


# ============================================================
# TestQAOAEdgeCases
# ============================================================
class TestQAOAEdgeCases(unittest.TestCase):
    """边界条件测试。"""

    def setUp(self):
        self.scheduler = QAOAScheduler(p=1, grid_size=5, max_qubits=6, seed=42)

    def test_all_same_priority(self):
        """所有任务优先级相同时应正常工作。"""
        tasks = [_make_task(f"T{i}", priority=3, estimated_time=10.0) for i in range(4)]
        result = self.scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertIn(result, [0, 1, 2, 3])

    def test_large_task_count(self):
        """任务数超过 max_qubits 时应预选。"""
        tasks = [_make_task(f"T{i}", priority=i % 5 + 1) for i in range(15)]
        result = self.scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertGreaterEqual(result, 0)
        self.assertLess(result, 15)

    def test_missing_fields(self):
        """任务缺少字段时应使用默认值。"""
        tasks = [{"task_id": "T1"}, {"task_id": "T2"}]
        result = self.scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertIn(result, [0, 1])

    def test_zero_qubit_count(self):
        """qubit_count=0 时应正常工作。"""
        tasks = [_make_task("T1", qubit_count=0), _make_task("T2", qubit_count=0)]
        result = self.scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertIn(result, [0, 1])

    def test_empty_resources(self):
        """空资源字典应正常工作。"""
        tasks = [_make_task("T1"), _make_task("T2")]
        result = self.scheduler.select_action(tasks, {})
        self.assertIn(result, [0, 1])

    def test_custom_parameters(self):
        """自定义参数应正常工作。"""
        scheduler = QAOAScheduler(p=1, grid_size=6, max_qubits=5, seed=123)
        tasks = [_make_task(f"T{i}", priority=i + 1) for i in range(3)]
        result = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertIn(result, [0, 1, 2])


# ============================================================
# TestQAOAPerformance
# ============================================================
class TestQAOAPerformance(unittest.TestCase):
    """性能测试。"""

    def test_solve_within_timeout(self):
        """求解应在合理时间内完成。"""
        import time

        scheduler = QAOAScheduler(p=1, grid_size=5, max_qubits=8, seed=42)
        tasks = [_make_task(f"T{i}", priority=i % 5 + 1) for i in range(6)]

        start = time.time()
        scheduler.select_action(tasks, _EMPTY_RESOURCES)
        elapsed = time.time() - start

        # 6 qubits, grid_size=5 应在 30 秒内完成
        self.assertLess(elapsed, 30.0)

    def test_max_qubits_limit(self):
        """max_qubits 应限制模拟规模。"""
        scheduler = QAOAScheduler(p=1, grid_size=3, max_qubits=4, seed=42)
        tasks = [_make_task(f"T{i}", priority=i + 1) for i in range(20)]
        # 应正常完成，不会因任务数过多而超时
        result = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertGreaterEqual(result, 0)

    def test_solution_stored(self):
        """求解后应存储解。"""
        scheduler = QAOAScheduler(p=1, grid_size=3, max_qubits=4, seed=42)
        tasks = [_make_task("T1"), _make_task("T2")]
        scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertIsNotNone(scheduler._last_solution)
        self.assertEqual(len(scheduler._last_solution), 2)

    def test_gamma_beta_stored(self):
        """求解后应存储最优参数。"""
        scheduler = QAOAScheduler(p=1, grid_size=3, max_qubits=4, seed=42)
        tasks = [_make_task("T1", priority=1), _make_task("T2", priority=5)]
        scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertGreater(scheduler._last_gamma, 0)
        self.assertGreater(scheduler._last_beta, 0)


if __name__ == "__main__":
    unittest.main()
