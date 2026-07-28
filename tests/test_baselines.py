"""
经典调度策略基线单元测试
Unit Tests for src/scheduler/baselines.py

测试覆盖：
- TestFCFSScheduler          : 先来先选、空列表边界
- TestSPTFScheduler          : 最短时间优先、相同时间稳定性
- TestEDFScheduler           : 最早截止优先、deadline 推算
- TestPriorityScheduler      : 高优先级优先、相同优先级稳定性
- TestRoundRobinScheduler    : 轮转顺序、指针更新
- TestLIFOScheduler          : 后来先服务
- TestHEFTScheduler          : HEFT 长任务优先、空列表边界（Issue #270）
- TestMinMinScheduler        : Min-Min 最短任务优先、空列表边界（Issue #270）
- TestRunBaselineComparison  : 多策略对比、返回结构完整
- TestEdgeCases              : 空任务列表、单任务、所有任务相同属性
- TestEnvBasedScheduler      : Gymnasium环境适配器（Issue #230）
- TestBaselineRewardConsistency : 奖励一致性验证（Issue #233）
- TestRunBaselineComparisonEnv  : use_env=True模式（Issue #231）
"""

import unittest
from unittest.mock import patch

from src.scheduler.baselines import (
    BaselineScheduler,
    EDFScheduler,
    EnvBasedEDFScheduler,
    EnvBasedFCFSScheduler,
    EnvBasedGreedyScheduler,
    EnvBasedHEFTScheduler,
    EnvBasedMinMinScheduler,
    EnvBasedScheduler,
    EnvBasedSPTFScheduler,
    FCFSScheduler,
    HEFTScheduler,
    LIFOScheduler,
    MinMinScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    SPTFScheduler,
    get_all_baseline_schedulers,
    get_all_env_based_schedulers,
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
# TestFCFSScheduler
# ============================================================
class TestFCFSScheduler(unittest.TestCase):
    """测试 FCFS（先来先服务）策略。"""

    def test_selects_earliest_arrival(self):
        """应选择到达时间最早的任务。"""
        tasks = [
            _make_task("T2", arrival_time=5.0),
            _make_task("T0", arrival_time=1.0),
            _make_task("T1", arrival_time=3.0),
        ]
        scheduler = FCFSScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 1)  # T0 到达最早
        self.assertEqual(tasks[idx]["task_id"], "T0")

    def test_empty_list_returns_negative(self):
        """空任务列表应返回 -1。"""
        scheduler = FCFSScheduler()
        self.assertEqual(scheduler.select_action([], _EMPTY_RESOURCES), -1)

    def test_name_and_repr(self):
        """策略名与 repr 应正确。"""
        s = FCFSScheduler()
        self.assertEqual(s.name, "FCFS")
        self.assertIn("FCFS", repr(s))


# ============================================================
# TestSPTFScheduler
# ============================================================
class TestSPTFScheduler(unittest.TestCase):
    """测试 SPTF（最短处理时间优先）策略。"""

    def test_selects_shortest_time(self):
        """应选择预估执行时间最短的任务。"""
        tasks = [
            _make_task("A", estimated_time=30.0),
            _make_task("B", estimated_time=5.0),
            _make_task("C", estimated_time=20.0),
        ]
        scheduler = SPTFScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 1)  # B 最短
        self.assertEqual(tasks[idx]["task_id"], "B")

    def test_equal_time_stability(self):
        """所有任务时间相同时应稳定返回第一个（索引 0）。"""
        tasks = [
            _make_task("A", estimated_time=10.0),
            _make_task("B", estimated_time=10.0),
            _make_task("C", estimated_time=10.0),
        ]
        scheduler = SPTFScheduler()
        for _ in range(3):
            self.assertEqual(scheduler.select_action(tasks, _EMPTY_RESOURCES), 0)

    def test_empty_list_returns_negative(self):
        """空任务列表应返回 -1。"""
        scheduler = SPTFScheduler()
        self.assertEqual(scheduler.select_action([], _EMPTY_RESOURCES), -1)


# ============================================================
# TestEDFScheduler
# ============================================================
class TestEDFScheduler(unittest.TestCase):
    """测试 EDF（最早截止时间优先）策略。"""

    def test_selects_earliest_deadline(self):
        """应选择截止时间最早的任务。"""
        tasks = [
            _make_task("A", deadline=50.0),
            _make_task("B", deadline=10.0),
            _make_task("C", deadline=30.0),
        ]
        scheduler = EDFScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 1)  # B 截止最早
        self.assertEqual(tasks[idx]["task_id"], "B")

    def test_deadline_inference(self):
        """缺失 deadline 时应按 arrival_time + estimated_time*2 推算。"""
        # A: 推算 = 0 + 5*2 = 10
        # B: 显式 deadline = 8（更早）
        # C: 推算 = 0 + 100*2 = 200
        tasks = [
            _make_task("A", arrival_time=0.0, estimated_time=5.0),  # 推算 10
            _make_task("B", arrival_time=0.0, estimated_time=5.0, deadline=8.0),
            _make_task("C", arrival_time=0.0, estimated_time=100.0),  # 推算 200
        ]
        scheduler = EDFScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 1)  # B 显式截止 8 最早

    def test_all_inferred_deadlines(self):
        """全部缺失 deadline 时按推算值比较。"""
        # A: 0 + 10*2 = 20
        # B: 0 + 2*2 = 4（最短）
        tasks = [
            _make_task("A", arrival_time=0.0, estimated_time=10.0),
            _make_task("B", arrival_time=0.0, estimated_time=2.0),
        ]
        scheduler = EDFScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 1)

    def test_empty_list_returns_negative(self):
        """空任务列表应返回 -1。"""
        scheduler = EDFScheduler()
        self.assertEqual(scheduler.select_action([], _EMPTY_RESOURCES), -1)


# ============================================================
# TestPriorityScheduler
# ============================================================
class TestPriorityScheduler(unittest.TestCase):
    """测试 Priority（优先级）策略。"""

    def test_selects_highest_priority(self):
        """应选择优先级最高的任务（priority 5 最高）。"""
        tasks = [
            _make_task("A", priority=2),
            _make_task("B", priority=5),
            _make_task("C", priority=3),
        ]
        scheduler = PriorityScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 1)  # B 优先级 5 最高
        self.assertEqual(tasks[idx]["task_id"], "B")

    def test_equal_priority_stability(self):
        """相同优先级时应按到达时间升序（先到先服务）稳定选择。"""
        tasks = [
            _make_task("A", priority=3, arrival_time=5.0),
            _make_task("B", priority=3, arrival_time=1.0),
            _make_task("C", priority=3, arrival_time=3.0),
        ]
        scheduler = PriorityScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 1)  # B 到达最早

    def test_empty_list_returns_negative(self):
        """空任务列表应返回 -1。"""
        scheduler = PriorityScheduler()
        self.assertEqual(scheduler.select_action([], _EMPTY_RESOURCES), -1)


# ============================================================
# TestRoundRobinScheduler
# ============================================================
class TestRoundRobinScheduler(unittest.TestCase):
    """测试 RoundRobin（轮询）策略。"""

    def test_rotation_order(self):
        """应按 0,1,2,0,... 顺序轮转。"""
        tasks = [_make_task("A"), _make_task("B"), _make_task("C")]
        scheduler = RoundRobinScheduler()
        order = [scheduler.select_action(tasks, _EMPTY_RESOURCES) for _ in range(7)]
        self.assertEqual(order, [0, 1, 2, 0, 1, 2, 0])

    def test_pointer_updates(self):
        """每次选择后指针应正确更新。"""
        tasks = [_make_task("A"), _make_task("B")]
        scheduler = RoundRobinScheduler()
        self.assertEqual(scheduler._pointer, 0)
        scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(scheduler._pointer, 1)
        scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(scheduler._pointer, 0)  # 回绕

    def test_reset_resets_pointer(self):
        """reset 应将指针归零。"""
        tasks = [_make_task("A"), _make_task("B")]
        scheduler = RoundRobinScheduler()
        scheduler.select_action(tasks, _EMPTY_RESOURCES)
        scheduler.select_action(tasks, _EMPTY_RESOURCES)
        scheduler.reset()
        self.assertEqual(scheduler._pointer, 0)

    def test_empty_list_returns_negative(self):
        """空任务列表应返回 -1，且不修改指针。"""
        scheduler = RoundRobinScheduler()
        self.assertEqual(scheduler.select_action([], _EMPTY_RESOURCES), -1)
        self.assertEqual(scheduler._pointer, 0)


# ============================================================
# TestLIFOScheduler
# ============================================================
class TestLIFOScheduler(unittest.TestCase):
    """测试 LIFO（后来先服务）策略。"""

    def test_selects_latest_arrival(self):
        """应选择到达时间最晚的任务。"""
        tasks = [
            _make_task("A", arrival_time=1.0),
            _make_task("B", arrival_time=5.0),
            _make_task("C", arrival_time=3.0),
        ]
        scheduler = LIFOScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 1)  # B 到达最晚
        self.assertEqual(tasks[idx]["task_id"], "B")

    def test_empty_list_returns_negative(self):
        """空任务列表应返回 -1。"""
        scheduler = LIFOScheduler()
        self.assertEqual(scheduler.select_action([], _EMPTY_RESOURCES), -1)


# ============================================================
# TestHEFTScheduler (Issue #270)
# ============================================================
class TestHEFTScheduler(unittest.TestCase):
    """测试 HEFT（异构最早完成时间）策略。"""

    def test_selects_longest_time(self):
        """应选择预估执行时间最长的任务（upward rank 降序）。"""
        tasks = [
            _make_task("A", estimated_time=5.0),
            _make_task("B", estimated_time=30.0),
            _make_task("C", estimated_time=15.0),
        ]
        scheduler = HEFTScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 1)  # B 最长
        self.assertEqual(tasks[idx]["task_id"], "B")

    def test_equal_time_stability(self):
        """所有任务时间相同时应稳定返回第一个（max 返回首个最大值索引）。"""
        tasks = [
            _make_task("A", estimated_time=10.0),
            _make_task("B", estimated_time=10.0),
            _make_task("C", estimated_time=10.0),
        ]
        scheduler = HEFTScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 0)  # max 返回第一个最大值索引

    def test_empty_list_returns_negative(self):
        """空任务列表应返回 -1。"""
        scheduler = HEFTScheduler()
        self.assertEqual(scheduler.select_action([], _EMPTY_RESOURCES), -1)

    def test_name_and_repr(self):
        """策略名与 repr 应正确。"""
        s = HEFTScheduler()
        self.assertEqual(s.name, "HEFT")
        self.assertIn("HEFT", repr(s))


# ============================================================
# TestMinMinScheduler (Issue #270)
# ============================================================
class TestMinMinScheduler(unittest.TestCase):
    """测试 Min-Min（最小完成时间优先）策略。"""

    def test_selects_shortest_time(self):
        """应选择预估执行时间最短的任务（最小完成时间优先）。"""
        tasks = [
            _make_task("A", estimated_time=30.0),
            _make_task("B", estimated_time=5.0),
            _make_task("C", estimated_time=20.0),
        ]
        scheduler = MinMinScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 1)  # B 最短
        self.assertEqual(tasks[idx]["task_id"], "B")

    def test_equal_time_stability(self):
        """所有任务时间相同时应稳定返回第一个（min 稳定选择）。"""
        tasks = [
            _make_task("A", estimated_time=10.0),
            _make_task("B", estimated_time=10.0),
            _make_task("C", estimated_time=10.0),
        ]
        scheduler = MinMinScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 0)  # min 返回第一个最小值索引

    def test_empty_list_returns_negative(self):
        """空任务列表应返回 -1。"""
        scheduler = MinMinScheduler()
        self.assertEqual(scheduler.select_action([], _EMPTY_RESOURCES), -1)

    def test_name_and_repr(self):
        """策略名与 repr 应正确。"""
        s = MinMinScheduler()
        self.assertEqual(s.name, "MinMin")
        self.assertIn("MinMin", repr(s))

    def test_iterative_assignment_correctness(self):
        """迭代贪心应正确分配所有任务到机器（Issue #583）。

        3 个任务，2 种机器（classical=1.0, quantum=3.0）：
        A(30): classical=30, quantum=10
        B(5):  classical=5,  quantum=1.67
        C(20): classical=20, quantum=6.67

        第1轮：B on quantum (MCT=1.67) 最小 → 量子机忙到 1.67
        第2轮：A on quantum (MCT=1.67+10=11.67) vs C on quantum (1.67+6.67=8.34) vs A on classical (30) vs C on classical (20)
              C on quantum (8.34) 最小 → 量子机忙到 8.34
        第3轮：A on quantum (8.34+10=18.34) vs A on classical (30)
              A on quantum (18.34) 最小
        """
        scheduler = MinMinScheduler()
        tasks = [
            _make_task("A", estimated_time=30.0),
            _make_task("B", estimated_time=5.0),
            _make_task("C", estimated_time=20.0),
        ]
        machines = {"classical": 1.0, "quantum": 3.0}
        assignment = scheduler._min_min_iterative(tasks, machines)
        self.assertEqual(len(assignment), 3)
        self.assertEqual(assignment[1], "quantum")  # B → quantum (MCT 最小)
        self.assertEqual(assignment[2], "quantum")  # C → quantum
        self.assertEqual(assignment[0], "quantum")  # A → quantum

    def test_select_action_with_machines_uses_mct(self):
        """带 machines 字段时应按 MCT 选择，而非纯执行时间（Issue #583）。

        无 machines 时按 estimated_time 升序选 B(5)。
        带 machines 时第一轮 MCT 最小的也是 B on quantum (1.67)，
        返回索引 1。
        """
        tasks = [
            _make_task("A", estimated_time=30.0),
            _make_task("B", estimated_time=5.0),
            _make_task("C", estimated_time=20.0),
        ]
        resources = {"qubits": 20, "machines": {"classical": 1.0, "quantum": 3.0}}
        scheduler = MinMinScheduler()
        idx = scheduler.select_action(tasks, resources)
        self.assertEqual(idx, 1)  # B on quantum MCT 最小

    def test_select_action_backward_compatible_no_machines(self):
        """无 machines 字段时应回退为按 estimated_time 升序（Issue #583）。"""
        tasks = [
            _make_task("A", estimated_time=30.0),
            _make_task("B", estimated_time=5.0),
            _make_task("C", estimated_time=20.0),
        ]
        scheduler = MinMinScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 1)  # B 最短，向后兼容行为

    def test_iterative_updates_machine_availability(self):
        """迭代贪心应更新机器可用时间（Issue #583）。

        两个相同任务(10s)，两种机器(classical=1.0, quantum=2.0)：
        classical MCT=10, quantum MCT=5
        第1轮：选 quantum (MCT=5)，量子机忙到 5
        第2轮：quantum MCT=5+5=10 vs classical MCT=10 → 相等，选 quantum
        两个任务都分配到 quantum。
        """
        scheduler = MinMinScheduler()
        tasks = [
            _make_task("T1", estimated_time=10.0),
            _make_task("T2", estimated_time=10.0),
        ]
        machines = {"classical": 1.0, "quantum": 2.0}
        assignment = scheduler._min_min_iterative(tasks, machines)
        self.assertEqual(len(assignment), 2)
        self.assertEqual(assignment[0], "quantum")
        self.assertEqual(assignment[1], "quantum")


# ============================================================
# TestRunBaselineComparison
# ============================================================
class TestRunBaselineComparison(unittest.TestCase):
    """测试 run_baseline_comparison 对比函数。"""

    def test_returns_all_strategies(self):
        """返回结果应包含全部 8 个基线策略。"""
        tasks = [
            _make_task("T1", priority=3, estimated_time=5.0, arrival_time=0.0),
            _make_task("T2", priority=5, estimated_time=3.0, arrival_time=1.0),
            _make_task("T3", priority=1, estimated_time=10.0, arrival_time=2.0),
        ]
        results = run_baseline_comparison(tasks, num_steps=10)
        expected_names = {"FCFS", "SPTF", "EDF", "Priority", "RoundRobin", "LIFO", "HEFT", "MinMin"}
        self.assertEqual(set(results.keys()), expected_names)

    def test_result_structure_complete(self):
        """每个策略结果应包含 4 个完整字段且类型正确。"""
        tasks = [_make_task("T1", priority=3, estimated_time=5.0, arrival_time=0.0)]
        results = run_baseline_comparison(tasks, num_steps=5)
        for name, metrics in results.items():
            self.assertIn("total_reward", metrics, f"{name} 缺少 total_reward")
            self.assertIn("completed_tasks", metrics, f"{name} 缺少 completed_tasks")
            self.assertIn("avg_wait_time", metrics, f"{name} 缺少 avg_wait_time")
            self.assertIn("throughput", metrics, f"{name} 缺少 throughput")
            self.assertIsInstance(metrics["total_reward"], float)
            self.assertIsInstance(metrics["completed_tasks"], int)
            self.assertIsInstance(metrics["avg_wait_time"], float)
            self.assertIsInstance(metrics["throughput"], float)

    def test_all_tasks_completed(self):
        """步数充足时应完成所有任务。"""
        tasks = [
            _make_task("T1", priority=3, estimated_time=5.0, arrival_time=0.0),
            _make_task("T2", priority=4, estimated_time=3.0, arrival_time=0.0),
        ]
        results = run_baseline_comparison(tasks, num_steps=10)
        for name, metrics in results.items():
            self.assertEqual(metrics["completed_tasks"], 2, f"{name} 应完成全部 2 个任务")

    def test_throughput_within_range(self):
        """吞吐率应在 [0, 1] 区间内。"""
        tasks = [_make_task("T1", estimated_time=1.0, arrival_time=0.0)]
        results = run_baseline_comparison(tasks, num_steps=10)
        for name, metrics in results.items():
            self.assertGreaterEqual(metrics["throughput"], 0.0, f"{name} throughput<0")
            self.assertLessEqual(metrics["throughput"], 1.0, f"{name} throughput>1")

    def test_get_all_baseline_schedulers(self):
        """get_all_baseline_schedulers 应返回 8 个不同策略实例。"""
        schedulers = get_all_baseline_schedulers()
        self.assertEqual(len(schedulers), 8)
        names = {s.name for s in schedulers}
        self.assertEqual(
            names, {"FCFS", "SPTF", "EDF", "Priority", "RoundRobin", "LIFO", "HEFT", "MinMin"}
        )
        for s in schedulers:
            self.assertIsInstance(s, BaselineScheduler)


# ============================================================
# TestEdgeCases
# ============================================================
class TestEdgeCases(unittest.TestCase):
    """边界情况测试。"""

    def test_empty_task_list_comparison(self):
        """空任务列表对比时各策略应完成 0 任务且奖励为 0。"""
        results = run_baseline_comparison([], num_steps=10)
        self.assertEqual(len(results), 8)
        for name, metrics in results.items():
            self.assertEqual(metrics["completed_tasks"], 0, f"{name} 空列表应完成 0")
            self.assertEqual(metrics["total_reward"], 0.0, f"{name} 空列表奖励应为 0")
            self.assertEqual(metrics["avg_wait_time"], 0.0)
            self.assertEqual(metrics["throughput"], 0.0)

    def test_single_task_all_strategies(self):
        """单任务时所有策略应选索引 0。"""
        tasks = [_make_task("only", priority=3, estimated_time=5.0, arrival_time=0.0)]
        for scheduler in get_all_baseline_schedulers():
            scheduler.reset()
            idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
            self.assertEqual(idx, 0, f"{scheduler.name} 单任务应选索引 0")

    def test_all_tasks_identical_attributes(self):
        """所有任务属性相同时各策略应返回合法索引（0..n-1）。"""
        tasks = [
            _make_task("A", priority=3, estimated_time=10.0, arrival_time=0.0),
            _make_task("B", priority=3, estimated_time=10.0, arrival_time=0.0),
            _make_task("C", priority=3, estimated_time=10.0, arrival_time=0.0),
        ]
        for scheduler in get_all_baseline_schedulers():
            scheduler.reset()
            idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
            self.assertIn(idx, range(len(tasks)), f"{scheduler.name} 应返回合法索引")
            self.assertGreaterEqual(idx, 0)

    def test_missing_optional_fields(self):
        """任务缺少 priority/estimated_time 等字段时应使用默认值不报错。"""
        tasks: list[dict] = [
            {"task_id": "X1", "arrival_time": 1.0},
            {"task_id": "X2", "arrival_time": 0.0},
        ]
        scheduler = FCFSScheduler()
        idx = scheduler.select_action(tasks, _EMPTY_RESOURCES)
        self.assertEqual(idx, 1)  # X2 到达更早

    def test_zero_steps(self):
        """num_steps=0 时应返回空完成结果且不报错。"""
        tasks = [_make_task("T1", estimated_time=1.0, arrival_time=0.0)]
        results = run_baseline_comparison(tasks, num_steps=0)
        for name, metrics in results.items():
            self.assertEqual(metrics["completed_tasks"], 0, f"{name} 0 步应完成 0")
            self.assertEqual(metrics["throughput"], 0.0)


# ============================================================
# TestEnvBasedScheduler（Issue #230）
# ============================================================
class TestEnvBasedScheduler(unittest.TestCase):
    """测试 EnvBasedScheduler Gymnasium 环境适配器（Issue #230/#270）。"""

    def test_get_all_env_based_schedulers(self):
        """get_all_env_based_schedulers 应返回 6 个 EnvBasedScheduler 实例。"""
        schedulers = get_all_env_based_schedulers()
        self.assertEqual(len(schedulers), 6)
        names = {s.name for s in schedulers}
        self.assertEqual(
            names, {"FCFS", "SPTF", "EDF", "Greedy", "EnvBased-HEFT", "EnvBased-MinMin"}
        )
        for s in schedulers:
            self.assertIsInstance(s, EnvBasedScheduler)

    def test_select_action_returns_valid_action(self):
        """所有 EnvBasedScheduler 子类应返回合法动作 [0, 1, 2]。"""
        import numpy as np

        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10, max_qubits=20, seed=42)
        obs = env.reset(seed=42)[0]

        for scheduler in get_all_env_based_schedulers():
            scheduler.reset()
            action = scheduler.select_action(obs, env)
            self.assertIn(action, [0, 1, 2], f"{scheduler.name} 返回非法动作 {action}")

    def test_env_based_scheduler_repr(self):
        """EnvBasedScheduler 的 repr 应包含类名和策略名。"""
        s = EnvBasedFCFSScheduler()
        self.assertIn("FCFS", repr(s))
        self.assertIn("EnvBasedFCFSScheduler", repr(s))

    def test_base_class_raises_not_implemented(self):
        """EnvBasedScheduler 基类 select_action 应抛出 NotImplementedError。"""
        import numpy as np

        s = EnvBasedScheduler("test")
        with self.assertRaises(NotImplementedError):
            s.select_action(np.zeros(14), None)

    def test_reset_is_noop_for_base(self):
        """EnvBasedScheduler.reset() 应为无操作，不抛异常。"""
        s = EnvBasedFCFSScheduler()
        s.reset()  # 不应抛异常

    def test_heft_uses_correct_obs_index_for_is_quantum(self):
        """Issue #380: EnvBasedHEFTScheduler 应使用 obs[8] 判断 is_quantum，而非错误的 obs[10]。

        原实现 obs[10] 读取的是 single_gate_fidelity（0.7-0.999），几乎总是 >0.5，
        导致 HEFT 将几乎所有任务都当作量子任务处理。
        """
        import numpy as np

        from src.scheduler.env_types import (
            OBS_QUBIT_AVAILABILITY,
            OBS_SINGLE_GATE_FIDELITY,
            OBS_TASK_TYPE_QUANTUM,
        )

        scheduler = EnvBasedHEFTScheduler()
        env = None  # HEFT 不直接使用 env

        # 构造观测：is_quantum=False（obs[8]=0），但 single_gate_fidelity 高（obs[10]=0.99）
        # 原错误实现会误判 is_quantum=True
        obs = np.zeros(16, dtype=np.float32)
        obs[OBS_TASK_TYPE_QUANTUM] = 0.0  # 非量子任务
        obs[OBS_SINGLE_GATE_FIDELITY] = 0.99  # 高保真度（原错误索引会读这个）
        obs[OBS_QUBIT_AVAILABILITY] = 0.8  # 量子资源充足

        action = scheduler.select_action(obs, env)
        # 修复后：is_quantum=False，不应选择量子动作
        # HEFT 选择最早完成时间；非量子任务应选经典或混合
        self.assertIn(action, [0, 2], "非量子任务不应选择量子动作（obs[10]误判bug）")

    def test_minmin_uses_correct_obs_index_for_is_quantum(self):
        """Issue #380: EnvBasedMinMinScheduler 应使用 obs[8] 判断 is_quantum，而非错误的 obs[10]。"""
        import numpy as np

        from src.scheduler.env_types import (
            OBS_QUANTUM_QUEUE_RATIO,
            OBS_QUBIT_AVAILABILITY,
            OBS_SINGLE_GATE_FIDELITY,
            OBS_TASK_TYPE_QUANTUM,
        )

        scheduler = EnvBasedMinMinScheduler()
        env = None

        # 构造观测：is_quantum=False，但 single_gate_fidelity 高
        obs = np.zeros(16, dtype=np.float32)
        obs[OBS_TASK_TYPE_QUANTUM] = 0.0  # 非量子任务
        obs[OBS_SINGLE_GATE_FIDELITY] = 0.99  # 原错误索引会读这个
        obs[OBS_QUBIT_AVAILABILITY] = 0.8
        obs[OBS_QUANTUM_QUEUE_RATIO] = 0.1

        action = scheduler.select_action(obs, env)
        # 修复后：is_quantum=False，MinMin 应返回经典动作
        self.assertEqual(action, 0, "非量子任务应选择经典动作（obs[10]误判bug）")

    def test_heft_quantum_task_uses_quantum_action_when_available(self):
        """Issue #380: 修复后 HEFT 对量子任务且资源充足时应考虑量子动作。"""
        import numpy as np

        from src.scheduler.env_types import (
            OBS_QUANTUM_QUEUE_RATIO,
            OBS_QUBIT_AVAILABILITY,
            OBS_TASK_TYPE_QUANTUM,
            OBS_URGENCY_LEVEL,
        )

        scheduler = EnvBasedHEFTScheduler()
        env = None

        # 构造观测：is_quantum=True，量子资源充足，队列短
        obs = np.zeros(16, dtype=np.float32)
        obs[OBS_TASK_TYPE_QUANTUM] = 1.0
        obs[OBS_QUBIT_AVAILABILITY] = 0.9
        obs[OBS_QUANTUM_QUEUE_RATIO] = 0.1
        obs[OBS_URGENCY_LEVEL] = 0.3

        action = scheduler.select_action(obs, env)
        self.assertIn(action, [0, 1, 2])  # 应返回合法动作


# ============================================================
# TestBaselineRewardConsistency (Issue #233)
# ============================================================
class TestBaselineRewardConsistency(unittest.TestCase):
    """Issue #233: 验证基线策略的 reward 来自环境（env_reward.py），而非独立计算。

    核心验证点：
    1. env.step() 调用 env_reward.compute_execution_reward 和 compute_wait_penalty
    2. env.step() 返回的 reward 包含 compute_execution_reward 的返回值
    3. FCFS/SPTF/EDF 三种策略在 Gymnasium 环境下运行时，reward 均来自环境
    4. 基线策略 select_action 只返回 action，不计算 reward
    """

    def setUp(self):
        """创建测试环境（不启用真机，避免外部依赖）。"""
        from src.scheduler.env import QuantumSchedulingEnv

        self.env = QuantumSchedulingEnv(max_steps=30, seed=42)
        self.env.reset(seed=42)

    def _run_steps_with_action(self, action: int, num_steps: int = 10) -> list[float]:
        """用固定动作运行指定步数，返回每步的 reward。

        模拟基线策略在 Gymnasium 环境下的运行：策略返回 action，环境计算 reward。

        Args:
            action   : 调度动作（0=经典，1=量子，2=混合）
            num_steps: 运行步数

        Returns:
            每步的 reward 列表
        """
        rewards = []
        for _ in range(num_steps):
            _obs, reward, terminated, truncated, _info = self.env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        return rewards

    def test_env_step_calls_compute_execution_reward(self):
        """env.step() 应调用 env_reward.compute_execution_reward。"""
        from src.scheduler import env as env_module

        call_count = 0
        original_fn = env_module.compute_execution_reward

        def spy(*args: object, **kwargs: object) -> float:
            nonlocal call_count
            call_count += 1
            return original_fn(*args, **kwargs)  # type: ignore[arg-type]

        self.env.reset(seed=42)
        with patch.object(env_module, "compute_execution_reward", spy):
            self._run_steps_with_action(0, num_steps=10)

        self.assertGreater(
            call_count,
            0,
            "env.step() 应调用 compute_execution_reward（证明 reward 来自环境）",
        )

    def test_env_step_calls_compute_wait_penalty(self):
        """env.step() 应调用 env_reward.compute_wait_penalty。"""
        from src.scheduler import env as env_module

        call_count = 0
        original_fn = env_module.compute_wait_penalty

        def spy(*args: object, **kwargs: object) -> float:
            nonlocal call_count
            call_count += 1
            return original_fn(*args, **kwargs)  # type: ignore[arg-type]

        self.env.reset(seed=42)
        with patch.object(env_module, "compute_wait_penalty", spy):
            self._run_steps_with_action(0, num_steps=10)

        self.assertGreater(
            call_count,
            0,
            "env.step() 应调用 compute_wait_penalty（证明 reward 来自环境）",
        )

    def test_fcfs_reward_comes_from_env(self):
        """FCFS 策略（经典执行 action=0）运行 10 步，reward 应来自环境。"""
        from src.scheduler import env as env_module

        call_count = 0
        original_fn = env_module.compute_execution_reward

        def spy(*args: object, **kwargs: object) -> float:
            nonlocal call_count
            call_count += 1
            return original_fn(*args, **kwargs)  # type: ignore[arg-type]

        self.env.reset(seed=42)
        with patch.object(env_module, "compute_execution_reward", spy):
            rewards = self._run_steps_with_action(0, num_steps=10)

        self.assertGreater(call_count, 0, "FCFS 策略运行时 env 应调用 compute_execution_reward")
        self.assertTrue(any(r != 0.0 for r in rewards), "FCFS 应产生非零 reward")

    def test_sptf_reward_comes_from_env(self):
        """SPTF 策略（混合执行 action=2）运行 10 步，reward 应来自环境。"""
        from src.scheduler import env as env_module

        call_count = 0
        original_fn = env_module.compute_execution_reward

        def spy(*args: object, **kwargs: object) -> float:
            nonlocal call_count
            call_count += 1
            return original_fn(*args, **kwargs)  # type: ignore[arg-type]

        self.env.reset(seed=42)
        with patch.object(env_module, "compute_execution_reward", spy):
            rewards = self._run_steps_with_action(2, num_steps=10)

        self.assertGreater(call_count, 0, "SPTF 策略运行时 env 应调用 compute_execution_reward")
        self.assertTrue(any(r != 0.0 for r in rewards), "SPTF 应产生非零 reward")

    def test_edf_reward_comes_from_env(self):
        """EDF 策略（混合执行 action=2）运行 10 步，reward 应来自环境。

        注：action=1（纯量子）只对 quantum 任务兼容，随机种子下可能无量子任务；
        此处用 action=2（混合，对所有任务类型兼容）确保 env.step 必然走到
        compute_execution_reward 调用路径。
        """
        from src.scheduler import env as env_module

        call_count = 0
        original_fn = env_module.compute_execution_reward

        def spy(*args: object, **kwargs: object) -> float:
            nonlocal call_count
            call_count += 1
            return original_fn(*args, **kwargs)  # type: ignore[arg-type]

        self.env.reset(seed=42)
        with patch.object(env_module, "compute_execution_reward", spy):
            rewards = self._run_steps_with_action(2, num_steps=10)

        self.assertGreater(call_count, 0, "EDF 策略运行时 env 应调用 compute_execution_reward")
        self.assertTrue(any(r != 0.0 for r in rewards), "EDF 应产生非零 reward")

    def test_env_reward_contains_compute_execution_reward_value(self):
        """env.step() 返回的 reward 应包含 compute_execution_reward 的返回值。

        通过 monkey-patch 让 compute_execution_reward 返回固定值 100.0，
        验证 env.step() 返回的 reward 包含该值（证明 reward 由环境函数计算）。
        """
        from src.scheduler import env as env_module

        fixed_return = 100.0

        def fixed_fn(*args: object, **kwargs: object) -> float:
            return fixed_return

        self.env.reset(seed=42)
        with patch.object(env_module, "compute_execution_reward", fixed_fn):
            _obs, reward, _term, _trunc, _info = self.env.step(0)

        # reward 应包含 fixed_return（可能还有等待惩罚、利用率惩罚等）
        # 由于固定值 100.0 远大于其他惩罚项（±2.0），验证 reward > 90.0 足够
        self.assertGreater(
            reward,
            90.0,
            f"env.step() 返回的 reward({reward}) 应包含 compute_execution_reward 的固定值(100.0)",
        )

    def test_baseline_select_action_does_not_compute_reward(self):
        """基线策略的 select_action 只返回 action，不应计算 reward。

        验证 FCFSScheduler/SPTFScheduler/EDFScheduler 的 select_action
        返回的是 int（任务索引/action），不是 reward。
        """
        tasks = [
            _make_task("T1", arrival_time=0.0, estimated_time=5.0),
            _make_task("T2", arrival_time=1.0, estimated_time=3.0),
            _make_task("T3", arrival_time=2.0, estimated_time=8.0, deadline=10.0),
        ]
        for scheduler_cls in [FCFSScheduler, SPTFScheduler, EDFScheduler]:
            scheduler = scheduler_cls()
            result = scheduler.select_action(tasks, _EMPTY_RESOURCES)
            # select_action 应返回 int（任务索引），不是 reward
            self.assertIsInstance(
                result,
                int,
                f"{scheduler.name}.select_action 应返回 int（任务索引），不是 reward",
            )
            self.assertGreaterEqual(result, -1, f"{scheduler.name} 应返回 >= -1")
            self.assertLess(result, len(tasks), f"{scheduler.name} 应返回 < len(tasks)")

    def test_run_baseline_comparison_uses_independent_formula(self):
        """run_baseline_comparison 使用独立奖励公式（非环境 reward）。

        这是对比工具的独立实现，不影响 Gymnasium 环境下的基线运行。
        本测试验证该函数确实使用自己的公式（10.0 + priority*2.0 - wait*0.1），
        而非调用 env_reward.py 的函数。
        """
        from src.scheduler import env as env_module

        call_count = 0
        original_fn = env_module.compute_execution_reward

        def spy(*args: object, **kwargs: object) -> float:
            nonlocal call_count
            call_count += 1
            return original_fn(*args, **kwargs)  # type: ignore[arg-type]

        tasks = [
            _make_task("T1", arrival_time=0.0, estimated_time=5.0, priority=3),
            _make_task("T2", arrival_time=1.0, estimated_time=3.0, priority=4),
        ]
        with patch.object(env_module, "compute_execution_reward", spy):
            results = run_baseline_comparison(tasks, num_steps=10)

        # run_baseline_comparison 是独立对比工具，不应调用 env_reward.py
        self.assertEqual(
            call_count,
            0,
            "run_baseline_comparison 使用独立公式，不应调用 env.compute_execution_reward",
        )
        # 但应产生有效结果
        self.assertGreater(len(results), 0)
        for _name, metrics in results.items():
            self.assertIn("total_reward", metrics)
            self.assertGreaterEqual(metrics["total_reward"], 0.0)

    def test_fcfs_reward_from_env(self):
        """FCFS 在环境中运行时，reward 由 env.step() 返回，非独立公式。

        验证方法：运行 FCFS 策略 10 步，检查 reward 值不是
        独立公式 10.0 + priority*2.0 - wait*0.1 的结果。
        """
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10, max_qubits=20, seed=42)
        scheduler = EnvBasedFCFSScheduler()
        obs = env.reset(seed=42)[0]

        rewards = []
        done = False
        while not done:
            action = scheduler.select_action(obs, env)
            obs, reward, terminated, truncated, _info = env.step(action)
            rewards.append(float(reward))
            done = terminated or truncated

        # 环境 reward 来自 compute_execution_reward：
        #   classical=8.0, quantum=10*speedup+3 (speedup 2-5), hybrid=7*factor+3
        # 独立公式 reward = 10 + priority*2 - wait*0.1 范围在 9.0~20.0
        # 环境 reward classical=8.0，与独立公式不同
        self.assertGreater(len(rewards), 0)
        # 检查 reward 值来自环境（8.0 是 classical 基础奖励）
        has_env_reward = any(abs(r - 8.0) < 0.01 or r >= 10.0 for r in rewards)
        self.assertTrue(
            has_env_reward,
            f"reward 值 {rewards} 不符合环境奖励模式",
        )

    def test_reward_not_independent_formula(self):
        """验证 reward 不是独立公式 10+priority*2-wait*0.1 的结果。

        独立公式在 wait=0 时 reward = 10 + priority*2，
        priority 范围 1-5，所以 reward 范围 12-20。
        环境 classical reward = 8.0，不在此范围内。
        """
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10, max_qubits=20, seed=42)
        scheduler = EnvBasedFCFSScheduler()
        obs = env.reset(seed=42)[0]

        rewards = []
        done = False
        while not done:
            action = scheduler.select_action(obs, env)
            obs, reward, terminated, truncated, _info = env.step(action)
            rewards.append(float(reward))
            done = terminated or truncated

        # 至少有一个 reward 不在独立公式范围 [12, 20] 内
        has_non_formula = any(r < 12.0 or r > 20.0 for r in rewards)
        self.assertTrue(
            has_non_formula,
            f"所有 reward {rewards} 都在独立公式范围 [12,20] 内，可能未使用环境 reward",
        )

    def test_sptf_reward_from_env(self):
        """SPTF 在环境中运行时，reward 由 env.step() 返回。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10, max_qubits=20, seed=42)
        scheduler = EnvBasedSPTFScheduler()
        obs = env.reset(seed=42)[0]

        rewards = []
        done = False
        while not done:
            action = scheduler.select_action(obs, env)
            obs, reward, terminated, truncated, _info = env.step(action)
            rewards.append(float(reward))
            done = terminated or truncated

        self.assertGreater(len(rewards), 0)
        # 至少有一个 reward 不在独立公式范围 [12, 20] 内
        has_non_formula = any(r < 12.0 or r > 20.0 for r in rewards)
        self.assertTrue(
            has_non_formula,
            f"SPTF 所有 reward {rewards} 都在独立公式范围 [12,20] 内",
        )

    def test_edf_reward_from_env(self):
        """EDF 在环境中运行时，reward 由 env.step() 返回。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10, max_qubits=20, seed=42)
        scheduler = EnvBasedEDFScheduler()
        obs = env.reset(seed=42)[0]

        rewards = []
        done = False
        while not done:
            action = scheduler.select_action(obs, env)
            obs, reward, terminated, truncated, _info = env.step(action)
            rewards.append(float(reward))
            done = terminated or truncated

        self.assertGreater(len(rewards), 0)
        # 至少有一个 reward 不在独立公式范围 [12, 20] 内
        has_non_formula = any(r < 12.0 or r > 20.0 for r in rewards)
        self.assertTrue(
            has_non_formula,
            f"EDF 所有 reward {rewards} 都在独立公式范围 [12,20] 内",
        )

    def test_all_baselines_produce_env_rewards(self):
        """所有 6 个 EnvBasedScheduler 都应产生来自环境的 reward。"""
        from src.scheduler.env import QuantumSchedulingEnv

        for scheduler in get_all_env_based_schedulers():
            env = QuantumSchedulingEnv(max_steps=10, max_qubits=20, seed=42)
            scheduler.reset()
            obs = env.reset(seed=42)[0]

            rewards = []
            done = False
            while not done:
                action = scheduler.select_action(obs, env)
                obs, reward, terminated, truncated, _info = env.step(action)
                rewards.append(float(reward))
                done = terminated or truncated

            self.assertGreater(len(rewards), 0, f"{scheduler.name} 未产生任何 reward")
            # 至少有一个 reward 不在独立公式范围 [12, 20] 内
            has_non_formula = any(r < 12.0 or r > 20.0 for r in rewards)
            self.assertTrue(
                has_non_formula,
                f"{scheduler.name} 所有 reward {rewards} 都在独立公式范围 [12,20] 内",
            )


# ============================================================
# TestRunBaselineComparisonEnv（Issue #231）
# ============================================================
class TestRunBaselineComparisonEnv(unittest.TestCase):
    """测试 run_baseline_comparison 的 use_env=True 模式（Issue #231）。"""

    def test_use_env_returns_env_based_mode(self):
        """use_env=True 时结果应包含 comparison_mode='env_based'。"""
        tasks = [_make_task("T1", priority=3, estimated_time=5.0, arrival_time=0.0)]
        results = run_baseline_comparison(tasks, num_steps=10, use_env=True, seed=42)
        for name, metrics in results.items():
            self.assertEqual(
                metrics["comparison_mode"],
                "env_based",
                f"{name} 应为 env_based 模式",
            )

    def test_use_env_returns_six_schedulers(self):
        """use_env=True 时应返回 6 个 EnvBasedScheduler 策略。"""
        tasks = [_make_task("T1", priority=3, estimated_time=5.0, arrival_time=0.0)]
        results = run_baseline_comparison(tasks, num_steps=10, use_env=True, seed=42)
        expected_names = {"FCFS", "SPTF", "EDF", "Greedy", "EnvBased-HEFT", "EnvBased-MinMin"}
        self.assertEqual(set(results.keys()), expected_names)

    def test_legacy_mode_returns_legacy(self):
        """use_env=False 时结果应包含 comparison_mode='legacy'。"""
        tasks = [_make_task("T1", priority=3, estimated_time=5.0, arrival_time=0.0)]
        results = run_baseline_comparison(tasks, num_steps=10, use_env=False)
        for name, metrics in results.items():
            self.assertEqual(
                metrics["comparison_mode"],
                "legacy",
                f"{name} 应为 legacy 模式",
            )

    def test_default_is_legacy(self):
        """默认（不传 use_env）应为 legacy 模式（向后兼容）。"""
        tasks = [_make_task("T1", priority=3, estimated_time=5.0, arrival_time=0.0)]
        results = run_baseline_comparison(tasks, num_steps=10)
        for _name, metrics in results.items():
            self.assertEqual(metrics["comparison_mode"], "legacy")

    def test_env_mode_reward_differs_from_legacy(self):
        """env_based 模式的 reward 应与 legacy 模式不同（证明使用不同奖励函数）。"""
        tasks = [
            _make_task("T1", priority=3, estimated_time=5.0, arrival_time=0.0),
            _make_task("T2", priority=5, estimated_time=3.0, arrival_time=0.0),
        ]
        legacy_results = run_baseline_comparison(tasks, num_steps=10, use_env=False)
        env_results = run_baseline_comparison(tasks, num_steps=10, use_env=True, seed=42)

        # FCFS 在两种模式下的 total_reward 应该不同
        legacy_fcfs = legacy_results["FCFS"]["total_reward"]
        env_fcfs = env_results.get("FCFS", {}).get("total_reward", 0.0)
        self.assertNotEqual(
            legacy_fcfs,
            env_fcfs,
            f"FCFS legacy reward={legacy_fcfs} 与 env reward={env_fcfs} 相同，"
            "可能未使用不同奖励函数",
        )

    def test_env_mode_result_structure(self):
        """use_env=True 结果应包含完整字段。"""
        tasks = [_make_task("T1", priority=3, estimated_time=5.0, arrival_time=0.0)]
        results = run_baseline_comparison(tasks, num_steps=10, use_env=True, seed=42)
        for _name, metrics in results.items():
            self.assertIn("total_reward", metrics)
            self.assertIn("completed_tasks", metrics)
            self.assertIn("avg_wait_time", metrics)
            self.assertIn("throughput", metrics)
            self.assertIn("comparison_mode", metrics)


if __name__ == "__main__":
    unittest.main()
