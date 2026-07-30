"""
量子RL调度系统 - DAG 调度器优先级感知列表调度测试
Unit Tests for upward rank computation and priority-aware list scheduling.

测试覆盖（Issue #642）：
- compute_upward_ranks：空图、单任务、线性链、菱形依赖、多出口节点
- schedule_priority_aware：空图、单任务、无依赖多任务、makespan 对比、
  高 upward rank 任务优先调度、返回格式一致性
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.dag_scheduler import DAGScheduler, DAGTask


def _makespan(schedule: list[dict[str, object]]) -> float:
    """计算调度的 makespan（最大完成时间）。"""
    if not schedule:
        return 0.0
    return max(float(item["estimated_finish"]) for item in schedule)


# ============================================================
# compute_upward_ranks 测试
# ============================================================
class TestComputeUpwardRanks(unittest.TestCase):
    """测试 upward rank 计算。"""

    def test_empty_graph_returns_empty_dict(self) -> None:
        """空图应返回空字典。"""
        scheduler = DAGScheduler()
        self.assertEqual(scheduler.compute_upward_ranks(), {})

    def test_single_task_rank_equals_estimated_time(self) -> None:
        """单任务（出口任务）的 rank_u 等于其 estimated_time。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", estimated_time=5.0))
        ranks = scheduler.compute_upward_ranks()
        self.assertEqual(ranks, {"a": 5.0})

    def test_linear_chain_rank_accumulates(self) -> None:
        """线性链 A->B->C 的 rank_u 应累加后继时长。

        rank(C)=d_C, rank(B)=d_B+d_C, rank(A)=d_A+d_B+d_C。
        """
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", estimated_time=1.0))
        scheduler.add_task(DAGTask(task_id="b", estimated_time=2.0, dependencies=["a"]))
        scheduler.add_task(DAGTask(task_id="c", estimated_time=3.0, dependencies=["b"]))
        ranks = scheduler.compute_upward_ranks()
        self.assertAlmostEqual(ranks["c"], 3.0)
        self.assertAlmostEqual(ranks["b"], 2.0 + 3.0)
        self.assertAlmostEqual(ranks["a"], 1.0 + 2.0 + 3.0)

    def test_diamond_dependency_rank(self) -> None:
        """菱形依赖 A->{B,C}->D 的 rank_u 符合公式。

        预期：D.rank=d_D, B.rank=d_B+d_D, C.rank=d_C+d_D,
        A.rank=d_A+max(d_B,d_C)+d_D。
        """
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", estimated_time=1.0))
        scheduler.add_task(DAGTask(task_id="b", estimated_time=2.0, dependencies=["a"]))
        scheduler.add_task(DAGTask(task_id="c", estimated_time=5.0, dependencies=["a"]))
        scheduler.add_task(DAGTask(task_id="d", estimated_time=1.0, dependencies=["b", "c"]))
        ranks = scheduler.compute_upward_ranks()
        self.assertAlmostEqual(ranks["d"], 1.0)
        self.assertAlmostEqual(ranks["b"], 2.0 + 1.0)
        self.assertAlmostEqual(ranks["c"], 5.0 + 1.0)
        self.assertAlmostEqual(ranks["a"], 1.0 + max(2.0, 5.0) + 1.0)

    def test_multi_exit_nodes_rank(self) -> None:
        """多出口节点（无后继任务）的 rank_u 均等于各自 estimated_time。

        结构：a->{b, c}，b、c 均为出口。
        """
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", estimated_time=2.0))
        scheduler.add_task(DAGTask(task_id="b", estimated_time=3.0, dependencies=["a"]))
        scheduler.add_task(DAGTask(task_id="c", estimated_time=4.0, dependencies=["a"]))
        ranks = scheduler.compute_upward_ranks()
        self.assertAlmostEqual(ranks["b"], 3.0)
        self.assertAlmostEqual(ranks["c"], 4.0)
        # a 的 rank = d_a + max(rank_b, rank_c) = 2.0 + 4.0
        self.assertAlmostEqual(ranks["a"], 2.0 + max(3.0, 4.0))

    def test_zero_duration_tasks(self) -> None:
        """零时长任务的 rank_u 应正确处理（累加结果为 0）。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", estimated_time=0.0))
        scheduler.add_task(DAGTask(task_id="b", estimated_time=0.0, dependencies=["a"]))
        ranks = scheduler.compute_upward_ranks()
        self.assertAlmostEqual(ranks["a"], 0.0)
        self.assertAlmostEqual(ranks["b"], 0.0)


# ============================================================
# schedule_priority_aware 测试
# ============================================================
class TestSchedulePriorityAware(unittest.TestCase):
    """测试基于 upward rank 的优先级感知列表调度。"""

    def test_empty_graph_returns_empty_list(self) -> None:
        """空图调度应返回空列表。"""
        scheduler = DAGScheduler()
        self.assertEqual(scheduler.schedule_priority_aware(available_qubits=10), [])

    def test_single_task_schedule(self) -> None:
        """单任务调度应返回单个调度项。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="solo", qubits_required=3, estimated_time=5.0))
        schedule = scheduler.schedule_priority_aware(available_qubits=10)
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0]["task_id"], "solo")
        self.assertEqual(schedule[0]["start_time"], 0.0)
        self.assertEqual(schedule[0]["estimated_finish"], 5.0)
        self.assertEqual(schedule[0]["machine_id"], 0)

    def test_independent_tasks_all_scheduled(self) -> None:
        """无依赖多任务应全部被调度，且资源充足时均可从 0 时刻开始。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", qubits_required=2, estimated_time=3.0))
        scheduler.add_task(DAGTask(task_id="b", qubits_required=2, estimated_time=4.0))
        scheduler.add_task(DAGTask(task_id="c", qubits_required=2, estimated_time=1.0))
        schedule = scheduler.schedule_priority_aware(available_qubits=10, available_machines=1)
        self.assertEqual(len(schedule), 3)
        self.assertEqual({item["task_id"] for item in schedule}, {"a", "b", "c"})
        # 资源充足，三任务均可从 0 时刻并行
        for item in schedule:
            self.assertEqual(item["start_time"], 0.0)

    def test_makespan_no_worse_than_topological_schedule(self) -> None:
        """资源竞争场景下 schedule_priority_aware 的 makespan 不大于
        schedule_with_resources 的 makespan。

        构造：2 机器 × 10 比特，每个任务需 10 比特（独占机器）。
        任务添加顺序为 short1, short2, long, tail（long 为关键路径）。
        拓扑序调度会先排 short 任务，延迟 long 与 tail；
        优先级感知会先排 long（rank 最高），使 tail 尽早就绪。
        """
        scheduler = DAGScheduler()
        # 注意：添加顺序影响 Kahn 拓扑序（short 先入队）
        scheduler.add_task(DAGTask(task_id="short1", qubits_required=10, estimated_time=2.0))
        scheduler.add_task(DAGTask(task_id="short2", qubits_required=10, estimated_time=2.0))
        scheduler.add_task(DAGTask(task_id="long", qubits_required=10, estimated_time=10.0))
        scheduler.add_task(
            DAGTask(task_id="tail", qubits_required=10, estimated_time=1.0, dependencies=["long"])
        )

        classic = scheduler.schedule_with_resources(available_qubits=10, available_machines=2)
        priority = scheduler.schedule_priority_aware(available_qubits=10, available_machines=2)

        classic_makespan = _makespan(classic)
        priority_makespan = _makespan(priority)
        # 优先级感知 makespan 应严格优于或等于拓扑序调度
        self.assertLessEqual(priority_makespan, classic_makespan)
        # 本构造下优先级感知应严格更优（13 → 11）
        self.assertLess(priority_makespan, classic_makespan)

    def test_high_rank_task_scheduled_first(self) -> None:
        """高 upward rank 任务在资源竞争中应先于低 rank 任务被调度。

        构造：1 机器 × 5 比特，long（rank=11）与 short（rank=1）竞争。
        优先级感知应先调度 long，使其后继 tail 尽早就绪。
        """
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="short", qubits_required=5, estimated_time=1.0))
        scheduler.add_task(DAGTask(task_id="long", qubits_required=5, estimated_time=10.0))
        scheduler.add_task(
            DAGTask(
                task_id="tail",
                qubits_required=5,
                estimated_time=1.0,
                dependencies=["long"],
            )
        )

        ranks = scheduler.compute_upward_ranks()
        # long 的 rank 应高于 short
        self.assertGreater(ranks["long"], ranks["short"])

        schedule = scheduler.schedule_priority_aware(available_qubits=5, available_machines=1)
        by_id = {item["task_id"]: item for item in schedule}
        # long 应从 0 时刻开始（优先级最高）
        self.assertEqual(by_id["long"]["start_time"], 0.0)
        # short 应在 long 之后开始
        self.assertGreaterEqual(by_id["short"]["start_time"], by_id["long"]["estimated_finish"])

    def test_result_format_consistency(self) -> None:
        """调度结果字段与 schedule_with_resources 一致。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", qubits_required=2, estimated_time=5.0))
        scheduler.add_task(
            DAGTask(task_id="b", qubits_required=2, estimated_time=3.0, dependencies=["a"])
        )
        schedule = scheduler.schedule_priority_aware(available_qubits=10, available_machines=2)
        self.assertEqual(len(schedule), 2)
        required_fields = {"task_id", "start_time", "machine_id", "estimated_finish"}
        for item in schedule:
            self.assertEqual(set(item.keys()), required_fields)

    def test_dependency_constraint_respected(self) -> None:
        """后继任务的开始时间不早于前驱完成时间。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", qubits_required=2, estimated_time=5.0))
        scheduler.add_task(
            DAGTask(task_id="b", qubits_required=2, estimated_time=3.0, dependencies=["a"])
        )
        schedule = scheduler.schedule_priority_aware(available_qubits=10, available_machines=1)
        by_id = {item["task_id"]: item for item in schedule}
        self.assertGreaterEqual(by_id["b"]["start_time"], by_id["a"]["estimated_finish"])

    def test_result_sorted_by_start_time_machine_task(self) -> None:
        """调度结果按 start_time、machine_id、task_id 升序排列。"""
        scheduler = DAGScheduler()
        for tid, est in [("a", 3.0), ("b", 1.0), ("c", 2.0)]:
            scheduler.add_task(DAGTask(task_id=tid, qubits_required=2, estimated_time=est))
        schedule = scheduler.schedule_priority_aware(available_qubits=10, available_machines=2)
        keys = [(item["start_time"], item["machine_id"], item["task_id"]) for item in schedule]
        self.assertEqual(keys, sorted(keys))

    def test_multi_machine_priority_aware_completes_all_tasks(self) -> None:
        """多机器场景下所有任务均被调度且依赖约束满足。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", qubits_required=3, estimated_time=5.0))
        scheduler.add_task(
            DAGTask(task_id="b", qubits_required=3, estimated_time=4.0, dependencies=["a"])
        )
        scheduler.add_task(
            DAGTask(task_id="c", qubits_required=2, estimated_time=2.0, dependencies=["a"])
        )
        scheduler.add_task(
            DAGTask(
                task_id="d",
                qubits_required=2,
                estimated_time=1.0,
                dependencies=["b", "c"],
            )
        )
        schedule = scheduler.schedule_priority_aware(available_qubits=5, available_machines=2)
        self.assertEqual(len(schedule), 4)
        by_id = {item["task_id"]: item for item in schedule}
        # 依赖约束
        self.assertGreaterEqual(by_id["b"]["start_time"], by_id["a"]["estimated_finish"])
        self.assertGreaterEqual(by_id["c"]["start_time"], by_id["a"]["estimated_finish"])
        self.assertGreaterEqual(by_id["d"]["start_time"], by_id["b"]["estimated_finish"])
        self.assertGreaterEqual(by_id["d"]["start_time"], by_id["c"]["estimated_finish"])


if __name__ == "__main__":
    unittest.main()
