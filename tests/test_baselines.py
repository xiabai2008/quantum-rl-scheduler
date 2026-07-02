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
- TestRunBaselineComparison  : 多策略对比、返回结构完整
- TestEdgeCases              : 空任务列表、单任务、所有任务相同属性
"""

import unittest

from src.scheduler.baselines import (
    BaselineScheduler,
    EDFScheduler,
    FCFSScheduler,
    LIFOScheduler,
    PriorityScheduler,
    RoundRobinScheduler,
    SPTFScheduler,
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
# TestRunBaselineComparison
# ============================================================
class TestRunBaselineComparison(unittest.TestCase):
    """测试 run_baseline_comparison 对比函数。"""

    def test_returns_all_strategies(self):
        """返回结果应包含全部 6 个基线策略。"""
        tasks = [
            _make_task("T1", priority=3, estimated_time=5.0, arrival_time=0.0),
            _make_task("T2", priority=5, estimated_time=3.0, arrival_time=1.0),
            _make_task("T3", priority=1, estimated_time=10.0, arrival_time=2.0),
        ]
        results = run_baseline_comparison(tasks, num_steps=10)
        expected_names = {"FCFS", "SPTF", "EDF", "Priority", "RoundRobin", "LIFO"}
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
            self.assertEqual(
                metrics["completed_tasks"], 2, f"{name} 应完成全部 2 个任务"
            )

    def test_throughput_within_range(self):
        """吞吐率应在 [0, 1] 区间内。"""
        tasks = [_make_task("T1", estimated_time=1.0, arrival_time=0.0)]
        results = run_baseline_comparison(tasks, num_steps=10)
        for name, metrics in results.items():
            self.assertGreaterEqual(metrics["throughput"], 0.0, f"{name} throughput<0")
            self.assertLessEqual(metrics["throughput"], 1.0, f"{name} throughput>1")

    def test_get_all_baseline_schedulers(self):
        """get_all_baseline_schedulers 应返回 6 个不同策略实例。"""
        schedulers = get_all_baseline_schedulers()
        self.assertEqual(len(schedulers), 6)
        names = {s.name for s in schedulers}
        self.assertEqual(names, {"FCFS", "SPTF", "EDF", "Priority", "RoundRobin", "LIFO"})
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
        self.assertEqual(len(results), 6)
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


if __name__ == "__main__":
    unittest.main()
