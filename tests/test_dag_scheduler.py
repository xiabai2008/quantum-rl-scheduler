"""
量子RL调度系统 - DAG 调度器单元测试
Unit Tests for src/scheduler/dag_scheduler.py

测试覆盖：
- DAGTask 数据类默认值与字段
- DAG 构建（add_task / add_dependency / 重复任务 / 自引用）
- DAG 校验（validate_dag / 环检测 / 缺失依赖）
- 拓扑排序（线性依赖 / 菱形依赖 / 环抛异常）
- 就绪任务查询（初始就绪 / 完成后后继就绪）
- 状态流转（mark_completed / mark_failed）
- 关键路径（CPM 最长路径）
- 资源约束调度（比特排队 / 多机器分配）
- 执行顺序
- 序列化（to_dict / from_tasks 往返）
- 边界（空图 / 单节点）
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.dag_scheduler import DAGScheduler, DAGTask


# ============================================================
# DAGTask 数据类测试
# ============================================================
class TestDAGTask(unittest.TestCase):
    """测试 DAGTask 数据类。"""

    def test_default_values(self) -> None:
        """测试默认字段值。"""
        task = DAGTask(task_id="t1")
        self.assertEqual(task.task_id, "t1")
        self.assertEqual(task.task_type, "quantum")
        self.assertEqual(task.qubits_required, 0)
        self.assertEqual(task.estimated_time, 0.0)
        self.assertEqual(task.priority, 3)
        self.assertEqual(task.dependencies, [])
        self.assertEqual(task.status, "pending")

    def test_custom_values(self) -> None:
        """测试自定义字段值。"""
        task = DAGTask(
            task_id="t2",
            task_type="classical",
            qubits_required=10,
            estimated_time=5.5,
            priority=5,
            dependencies=["t1"],
            status="running",
        )
        self.assertEqual(task.task_type, "classical")
        self.assertEqual(task.qubits_required, 10)
        self.assertEqual(task.estimated_time, 5.5)
        self.assertEqual(task.priority, 5)
        self.assertEqual(task.dependencies, ["t1"])
        self.assertEqual(task.status, "running")

    def test_dependencies_isolated(self) -> None:
        """测试 dependencies 默认列表互相隔离。"""
        t1 = DAGTask(task_id="t1")
        t2 = DAGTask(task_id="t2")
        t1.dependencies.append("x")
        self.assertEqual(t2.dependencies, [])


# ============================================================
# DAG 构建测试
# ============================================================
class TestDAGConstruction(unittest.TestCase):
    """测试 DAG 构建操作。"""

    def test_add_task(self) -> None:
        """测试添加任务。"""
        scheduler = DAGScheduler()
        task = DAGTask(task_id="t1", qubits_required=5)
        scheduler.add_task(task)
        self.assertIn("t1", scheduler.tasks)
        self.assertEqual(scheduler.tasks["t1"].qubits_required, 5)

    def test_add_duplicate_task(self) -> None:
        """测试重复添加任务抛出 ValueError。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="t1"))
        with self.assertRaises(ValueError):
            scheduler.add_task(DAGTask(task_id="t1"))

    def test_add_dependency(self) -> None:
        """测试添加依赖关系。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b"))
        scheduler.add_dependency("b", "a")
        self.assertIn("a", scheduler.tasks["b"].dependencies)

    def test_add_dependency_nonexistent_task(self) -> None:
        """测试对不存在的任务添加依赖抛出 ValueError。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        with self.assertRaises(ValueError):
            scheduler.add_dependency("b", "a")
        with self.assertRaises(ValueError):
            scheduler.add_dependency("a", "b")

    def test_add_dependency_self_reference(self) -> None:
        """测试自引用依赖抛出 ValueError。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        with self.assertRaises(ValueError):
            scheduler.add_dependency("a", "a")

    def test_add_dependency_idempotent(self) -> None:
        """测试重复添加相同依赖不重复。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b"))
        scheduler.add_dependency("b", "a")
        scheduler.add_dependency("b", "a")
        self.assertEqual(scheduler.tasks["b"].dependencies.count("a"), 1)

    def test_init_with_tasks(self) -> None:
        """测试通过构造函数传入任务列表。"""
        tasks = [
            DAGTask(task_id="a"),
            DAGTask(task_id="b", dependencies=["a"]),
        ]
        scheduler = DAGScheduler(tasks=tasks)
        self.assertEqual(len(scheduler.tasks), 2)
        self.assertIn("a", scheduler.tasks["b"].dependencies)


# ============================================================
# DAG 校验与环检测测试
# ============================================================
class TestDAGValidation(unittest.TestCase):
    """测试 DAG 合法性校验。"""

    def test_validate_valid_dag(self) -> None:
        """测试合法 DAG 校验通过。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b", dependencies=["a"]))
        self.assertTrue(scheduler.validate_dag())

    def test_validate_cycle_raises(self) -> None:
        """测试有环 DAG 抛出 ValueError。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", dependencies=["b"]))
        scheduler.add_task(DAGTask(task_id="b", dependencies=["a"]))
        with self.assertRaises(ValueError):
            scheduler.validate_dag()

    def test_validate_missing_dependency_raises(self) -> None:
        """测试缺失依赖抛出 ValueError。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", dependencies=["ghost"]))
        with self.assertRaises(ValueError):
            scheduler.validate_dag()

    def test_detect_cycle_direct(self) -> None:
        """测试直接环检测 A->B->A。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b"))
        scheduler.add_dependency("b", "a")
        scheduler.add_dependency("a", "b")
        self.assertTrue(scheduler._detect_cycle())

    def test_detect_cycle_self_loop(self) -> None:
        """测试自环检测（通过 dependencies 字段构造）。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", dependencies=["a"]))
        self.assertTrue(scheduler._detect_cycle())

    def test_detect_no_cycle(self) -> None:
        """测试无环图不误报。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b", dependencies=["a"]))
        scheduler.add_task(DAGTask(task_id="c", dependencies=["b"]))
        self.assertFalse(scheduler._detect_cycle())


# ============================================================
# 拓扑排序测试
# ============================================================
class TestTopologicalSort(unittest.TestCase):
    """测试 Kahn 算法拓扑排序。"""

    def test_linear_dependency(self) -> None:
        """测试线性依赖 A->B->C 拓扑顺序。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b", dependencies=["a"]))
        scheduler.add_task(DAGTask(task_id="c", dependencies=["b"]))
        order = scheduler.topological_sort()
        self.assertEqual(order, ["a", "b", "c"])

    def test_diamond_dependency(self) -> None:
        """测试菱形依赖 A->{B,C}->D 拓扑顺序。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b", dependencies=["a"]))
        scheduler.add_task(DAGTask(task_id="c", dependencies=["a"]))
        scheduler.add_task(DAGTask(task_id="d", dependencies=["b", "c"]))
        order = scheduler.topological_sort()
        # A 必须在 B、C 之前；B、C 必须在 D 之前
        self.assertEqual(order[0], "a")
        self.assertEqual(order[-1], "d")
        self.assertLess(order.index("b"), order.index("d"))
        self.assertLess(order.index("c"), order.index("d"))
        self.assertLess(order.index("a"), order.index("b"))
        self.assertLess(order.index("a"), order.index("c"))

    def test_topological_sort_cycle_raises(self) -> None:
        """测试有环图拓扑排序抛出 ValueError。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b"))
        scheduler.add_dependency("b", "a")
        scheduler.add_dependency("a", "b")
        with self.assertRaises(ValueError):
            scheduler.topological_sort()

    def test_topological_sort_independent_tasks(self) -> None:
        """测试独立任务（无依赖）拓扑排序。"""
        scheduler = DAGScheduler()
        for tid in ["x", "y", "z"]:
            scheduler.add_task(DAGTask(task_id=tid))
        order = scheduler.topological_sort()
        self.assertEqual(set(order), {"x", "y", "z"})
        self.assertEqual(len(order), 3)

    def test_topological_sort_empty(self) -> None:
        """测试空图拓扑排序返回空列表。"""
        scheduler = DAGScheduler()
        self.assertEqual(scheduler.topological_sort(), [])


# ============================================================
# 就绪任务测试
# ============================================================
class TestReadyTasks(unittest.TestCase):
    """测试就绪任务查询。"""

    def test_initial_ready_tasks(self) -> None:
        """测试初始就绪任务（无依赖的任务）。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b", dependencies=["a"]))
        ready = scheduler.get_ready_tasks()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "a")

    def test_ready_after_completion(self) -> None:
        """测试完成后后继任务变为就绪。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b", dependencies=["a"]))
        # 初始只有 a 就绪
        self.assertEqual([t.task_id for t in scheduler.get_ready_tasks()], ["a"])
        # 完成 a 后 b 变为就绪
        scheduler.mark_completed("a")
        ready = scheduler.get_ready_tasks()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "b")

    def test_ready_priority_order(self) -> None:
        """测试就绪任务按优先级排序。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="low", priority=1))
        scheduler.add_task(DAGTask(task_id="high", priority=5))
        scheduler.add_task(DAGTask(task_id="mid", priority=3))
        ready = scheduler.get_ready_tasks()
        self.assertEqual([t.task_id for t in ready], ["high", "mid", "low"])

    def test_ready_excludes_non_pending(self) -> None:
        """测试已完成/失败任务不在就绪列表中。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b"))
        scheduler.mark_completed("a")
        scheduler.mark_failed("b")
        ready = scheduler.get_ready_tasks()
        self.assertEqual(ready, [])

    def test_ready_diamond_progression(self) -> None:
        """测试菱形依赖的就绪推进。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b", dependencies=["a"]))
        scheduler.add_task(DAGTask(task_id="c", dependencies=["a"]))
        scheduler.add_task(DAGTask(task_id="d", dependencies=["b", "c"]))
        # 初始只有 a
        self.assertEqual({t.task_id for t in scheduler.get_ready_tasks()}, {"a"})
        scheduler.mark_completed("a")
        # a 完成后 b、c 就绪
        self.assertEqual({t.task_id for t in scheduler.get_ready_tasks()}, {"b", "c"})
        scheduler.mark_completed("b")
        # b 完成后 d 仍不就绪（c 未完成）
        self.assertEqual({t.task_id for t in scheduler.get_ready_tasks()}, {"c"})
        scheduler.mark_completed("c")
        # c 完成后 d 就绪
        self.assertEqual({t.task_id for t in scheduler.get_ready_tasks()}, {"d"})


# ============================================================
# 状态流转测试
# ============================================================
class TestStateTransition(unittest.TestCase):
    """测试任务状态流转。"""

    def test_mark_completed(self) -> None:
        """测试标记完成。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.mark_completed("a")
        self.assertEqual(scheduler.tasks["a"].status, "completed")
        self.assertIn("a", scheduler.completed)
        self.assertNotIn("a", scheduler.failed)

    def test_mark_failed(self) -> None:
        """测试标记失败。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.mark_failed("a")
        self.assertEqual(scheduler.tasks["a"].status, "failed")
        self.assertIn("a", scheduler.failed)
        self.assertNotIn("a", scheduler.completed)

    def test_mark_completed_overrides_failed(self) -> None:
        """测试完成覆盖失败状态。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.mark_failed("a")
        scheduler.mark_completed("a")
        self.assertEqual(scheduler.tasks["a"].status, "completed")
        self.assertIn("a", scheduler.completed)
        self.assertNotIn("a", scheduler.failed)

    def test_mark_failed_overrides_completed(self) -> None:
        """测试失败覆盖完成状态。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.mark_completed("a")
        scheduler.mark_failed("a")
        self.assertEqual(scheduler.tasks["a"].status, "failed")
        self.assertIn("a", scheduler.failed)
        self.assertNotIn("a", scheduler.completed)

    def test_mark_completed_nonexistent(self) -> None:
        """测试标记不存在的任务抛出 ValueError。"""
        scheduler = DAGScheduler()
        with self.assertRaises(ValueError):
            scheduler.mark_completed("ghost")

    def test_mark_failed_nonexistent(self) -> None:
        """测试标记不存在的任务失败抛出 ValueError。"""
        scheduler = DAGScheduler()
        with self.assertRaises(ValueError):
            scheduler.mark_failed("ghost")


# ============================================================
# 关键路径测试
# ============================================================
class TestCriticalPath(unittest.TestCase):
    """测试 CPM 关键路径分析。"""

    def test_critical_path_linear(self) -> None:
        """测试线性依赖关键路径为全路径。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", estimated_time=1.0))
        scheduler.add_task(DAGTask(task_id="b", estimated_time=2.0, dependencies=["a"]))
        scheduler.add_task(DAGTask(task_id="c", estimated_time=3.0, dependencies=["b"]))
        path = scheduler.critical_path()
        self.assertEqual(path, ["a", "b", "c"])

    def test_critical_path_longest(self) -> None:
        """测试多条路径时选择最长路径。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", estimated_time=1.0))
        # 路径1: a->b->d (1+2+1=4)
        scheduler.add_task(DAGTask(task_id="b", estimated_time=2.0, dependencies=["a"]))
        # 路径2: a->c->d (1+5+1=7) ← 关键路径
        scheduler.add_task(DAGTask(task_id="c", estimated_time=5.0, dependencies=["a"]))
        scheduler.add_task(
            DAGTask(
                task_id="d", estimated_time=1.0, dependencies=["b", "c"]
            )
        )
        path = scheduler.critical_path()
        self.assertEqual(path, ["a", "c", "d"])

    def test_critical_path_single_node(self) -> None:
        """测试单节点关键路径。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="only", estimated_time=3.0))
        path = scheduler.critical_path()
        self.assertEqual(path, ["only"])

    def test_critical_path_empty(self) -> None:
        """测试空图关键路径为空。"""
        scheduler = DAGScheduler()
        self.assertEqual(scheduler.critical_path(), [])

    def test_critical_path_zero_duration(self) -> None:
        """测试零时长任务的关键路径。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", estimated_time=0.0))
        scheduler.add_task(DAGTask(task_id="b", estimated_time=0.0, dependencies=["a"]))
        path = scheduler.critical_path()
        self.assertEqual(path, ["a", "b"])


# ============================================================
# 资源约束调度测试
# ============================================================
class TestScheduleWithResources(unittest.TestCase):
    """测试资源约束调度。"""

    def test_schedule_basic(self) -> None:
        """测试基本调度（资源充足时并行）。"""
        scheduler = DAGScheduler()
        scheduler.add_task(
            DAGTask(task_id="a", qubits_required=2, estimated_time=5.0)
        )
        scheduler.add_task(
            DAGTask(task_id="b", qubits_required=2, estimated_time=5.0)
        )
        schedule = scheduler.schedule_with_resources(
            available_qubits=10, available_machines=1
        )
        self.assertEqual(len(schedule), 2)
        # 资源充足，两任务可并行（同一机器）
        for item in schedule:
            self.assertEqual(item["start_time"], 0.0)
            self.assertEqual(item["machine_id"], 0)

    def test_schedule_resource_queuing(self) -> None:
        """测试资源不足时任务排队。"""
        scheduler = DAGScheduler()
        scheduler.add_task(
            DAGTask(task_id="a", qubits_required=3, estimated_time=5.0)
        )
        scheduler.add_task(
            DAGTask(task_id="b", qubits_required=3, estimated_time=5.0)
        )
        schedule = scheduler.schedule_with_resources(
            available_qubits=4, available_machines=1
        )
        # a、b 各需 3 比特，容量 4，无法并行（3+3>4），b 须等 a 完成
        schedule_by_id = {item["task_id"]: item for item in schedule}
        self.assertEqual(schedule_by_id["a"]["start_time"], 0.0)
        self.assertEqual(schedule_by_id["a"]["estimated_finish"], 5.0)
        self.assertEqual(schedule_by_id["b"]["start_time"], 5.0)
        self.assertEqual(schedule_by_id["b"]["estimated_finish"], 10.0)

    def test_schedule_multi_machine(self) -> None:
        """测试多机器分配。"""
        scheduler = DAGScheduler()
        scheduler.add_task(
            DAGTask(task_id="a", qubits_required=3, estimated_time=5.0)
        )
        scheduler.add_task(
            DAGTask(task_id="b", qubits_required=3, estimated_time=5.0)
        )
        schedule = scheduler.schedule_with_resources(
            available_qubits=4, available_machines=2
        )
        # 两台机器，两任务可分别在不同机器上并行
        machine_ids = {item["machine_id"] for item in schedule}
        self.assertEqual(machine_ids, {0, 1})
        for item in schedule:
            self.assertEqual(item["start_time"], 0.0)

    def test_schedule_with_dependencies(self) -> None:
        """测试依赖约束影响开始时间。"""
        scheduler = DAGScheduler()
        scheduler.add_task(
            DAGTask(task_id="a", qubits_required=2, estimated_time=5.0)
        )
        scheduler.add_task(
            DAGTask(
                task_id="b", qubits_required=2, estimated_time=3.0, dependencies=["a"]
            )
        )
        schedule = scheduler.schedule_with_resources(
            available_qubits=10, available_machines=1
        )
        schedule_by_id = {item["task_id"]: item for item in schedule}
        self.assertEqual(schedule_by_id["a"]["start_time"], 0.0)
        self.assertEqual(schedule_by_id["a"]["estimated_finish"], 5.0)
        # b 必须在 a 完成后开始
        self.assertEqual(schedule_by_id["b"]["start_time"], 5.0)
        self.assertEqual(schedule_by_id["b"]["estimated_finish"], 8.0)

    def test_schedule_sorted_by_start_time(self) -> None:
        """测试调度结果按开始时间排序。"""
        scheduler = DAGScheduler()
        scheduler.add_task(
            DAGTask(task_id="a", qubits_required=2, estimated_time=5.0)
        )
        scheduler.add_task(
            DAGTask(
                task_id="b", qubits_required=2, estimated_time=3.0, dependencies=["a"]
            )
        )
        schedule = scheduler.schedule_with_resources(
            available_qubits=10, available_machines=1
        )
        start_times = [item["start_time"] for item in schedule]
        self.assertEqual(start_times, sorted(start_times))

    def test_schedule_empty(self) -> None:
        """测试空图调度返回空列表。"""
        scheduler = DAGScheduler()
        self.assertEqual(
            scheduler.schedule_with_resources(available_qubits=10), []
        )

    def test_schedule_result_fields(self) -> None:
        """测试调度结果包含所有必要字段。"""
        scheduler = DAGScheduler()
        scheduler.add_task(
            DAGTask(task_id="a", qubits_required=2, estimated_time=5.0)
        )
        schedule = scheduler.schedule_with_resources(
            available_qubits=10, available_machines=1
        )
        item = schedule[0]
        self.assertIn("task_id", item)
        self.assertIn("start_time", item)
        self.assertIn("machine_id", item)
        self.assertIn("estimated_finish", item)


# ============================================================
# 执行顺序测试
# ============================================================
class TestExecutionOrder(unittest.TestCase):
    """测试资源约束执行顺序。"""

    def test_execution_order_basic(self) -> None:
        """测试基本执行顺序。"""
        scheduler = DAGScheduler(max_qubits=10)
        scheduler.add_task(DAGTask(task_id="a", estimated_time=1.0))
        scheduler.add_task(
            DAGTask(task_id="b", estimated_time=1.0, dependencies=["a"])
        )
        order = scheduler.get_execution_order()
        self.assertEqual(order, ["a", "b"])

    def test_execution_order_empty(self) -> None:
        """测试空图执行顺序。"""
        scheduler = DAGScheduler()
        self.assertEqual(scheduler.get_execution_order(), [])


# ============================================================
# 序列化测试
# ============================================================
class TestSerialization(unittest.TestCase):
    """测试 to_dict / from_tasks 序列化。"""

    def test_to_dict_structure(self) -> None:
        """测试 to_dict 返回结构。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a", qubits_required=5))
        scheduler.add_task(
            DAGTask(task_id="b", qubits_required=3, dependencies=["a"])
        )
        result = scheduler.to_dict()
        self.assertIn("nodes", result)
        self.assertIn("edges", result)
        self.assertEqual(len(result["nodes"]), 2)
        self.assertEqual(len(result["edges"]), 1)
        self.assertEqual(result["edges"][0], ["a", "b"])

    def test_to_dict_node_fields(self) -> None:
        """测试 to_dict 节点包含所有字段。"""
        scheduler = DAGScheduler()
        scheduler.add_task(
            DAGTask(
                task_id="a",
                task_type="quantum",
                qubits_required=5,
                estimated_time=3.0,
                priority=4,
                dependencies=[],
                status="pending",
            )
        )
        result = scheduler.to_dict()
        node = result["nodes"][0]
        self.assertEqual(node["task_id"], "a")
        self.assertEqual(node["task_type"], "quantum")
        self.assertEqual(node["qubits_required"], 5)
        self.assertEqual(node["estimated_time"], 3.0)
        self.assertEqual(node["priority"], 4)
        self.assertEqual(node["dependencies"], [])
        self.assertEqual(node["status"], "pending")

    def test_from_tasks_basic(self) -> None:
        """测试从字典列表构建调度器。"""
        tasks = [
            {
                "task_id": "a",
                "task_type": "quantum",
                "qubits_required": 5,
                "estimated_time": 3.0,
                "priority": 4,
                "dependencies": [],
            },
            {
                "task_id": "b",
                "task_type": "classical",
                "qubits_required": 0,
                "estimated_time": 1.0,
                "priority": 2,
                "dependencies": ["a"],
            },
        ]
        scheduler = DAGScheduler.from_tasks(tasks)
        self.assertEqual(len(scheduler.tasks), 2)
        self.assertEqual(scheduler.tasks["a"].qubits_required, 5)
        self.assertEqual(scheduler.tasks["b"].dependencies, ["a"])

    def test_from_tasks_defaults(self) -> None:
        """测试从字典构建时使用默认值。"""
        tasks = [{"task_id": "a"}]
        scheduler = DAGScheduler.from_tasks(tasks)
        task = scheduler.tasks["a"]
        self.assertEqual(task.task_type, "quantum")
        self.assertEqual(task.qubits_required, 0)
        self.assertEqual(task.estimated_time, 0.0)
        self.assertEqual(task.priority, 3)
        self.assertEqual(task.dependencies, [])
        self.assertEqual(task.status, "pending")

    def test_roundtrip(self) -> None:
        """测试 to_dict → from_tasks 往返一致性。"""
        original = DAGScheduler()
        original.add_task(
            DAGTask(
                task_id="a",
                task_type="quantum",
                qubits_required=5,
                estimated_time=3.0,
                priority=4,
            )
        )
        original.add_task(
            DAGTask(
                task_id="b",
                task_type="classical",
                qubits_required=0,
                estimated_time=1.0,
                priority=2,
                dependencies=["a"],
            )
        )
        original.add_task(
            DAGTask(
                task_id="c",
                qubits_required=2,
                estimated_time=2.0,
                dependencies=["a", "b"],
            )
        )
        data = original.to_dict()
        restored = DAGScheduler.from_tasks(data["nodes"])
        # 验证节点一致
        self.assertEqual(len(restored.tasks), len(original.tasks))
        for tid in original.tasks:
            self.assertIn(tid, restored.tasks)
            orig_task = original.tasks[tid]
            rest_task = restored.tasks[tid]
            self.assertEqual(orig_task.task_type, rest_task.task_type)
            self.assertEqual(orig_task.qubits_required, rest_task.qubits_required)
            self.assertEqual(orig_task.estimated_time, rest_task.estimated_time)
            self.assertEqual(orig_task.priority, rest_task.priority)
            self.assertEqual(orig_task.dependencies, rest_task.dependencies)
            self.assertEqual(orig_task.status, rest_task.status)
        # 验证边一致
        self.assertEqual(restored.to_dict()["edges"], data["edges"])

    def test_roundtrip_empty(self) -> None:
        """测试空图往返。"""
        scheduler = DAGScheduler()
        data = scheduler.to_dict()
        self.assertEqual(data, {"nodes": [], "edges": []})
        restored = DAGScheduler.from_tasks(data["nodes"])
        self.assertEqual(len(restored.tasks), 0)


# ============================================================
# 边界测试
# ============================================================
class TestEdgeCases(unittest.TestCase):
    """测试边界情况。"""

    def test_empty_graph(self) -> None:
        """测试空图各项操作。"""
        scheduler = DAGScheduler()
        self.assertEqual(scheduler.topological_sort(), [])
        self.assertEqual(scheduler.critical_path(), [])
        self.assertEqual(scheduler.get_ready_tasks(), [])
        self.assertTrue(scheduler.validate_dag())
        self.assertEqual(scheduler.to_dict(), {"nodes": [], "edges": []})

    def test_single_node(self) -> None:
        """测试单节点图。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="solo", estimated_time=2.0))
        self.assertEqual(scheduler.topological_sort(), ["solo"])
        self.assertEqual(scheduler.critical_path(), ["solo"])
        ready = scheduler.get_ready_tasks()
        self.assertEqual(len(ready), 1)
        self.assertEqual(ready[0].task_id, "solo")
        self.assertTrue(scheduler.validate_dag())

    def test_single_node_schedule(self) -> None:
        """测试单节点资源调度。"""
        scheduler = DAGScheduler()
        scheduler.add_task(
            DAGTask(task_id="solo", qubits_required=3, estimated_time=5.0)
        )
        schedule = scheduler.schedule_with_resources(
            available_qubits=10, available_machines=1
        )
        self.assertEqual(len(schedule), 1)
        self.assertEqual(schedule[0]["task_id"], "solo")
        self.assertEqual(schedule[0]["start_time"], 0.0)
        self.assertEqual(schedule[0]["estimated_finish"], 5.0)
        self.assertEqual(schedule[0]["machine_id"], 0)

    def test_build_adjacency(self) -> None:
        """测试邻接表构建。"""
        scheduler = DAGScheduler()
        scheduler.add_task(DAGTask(task_id="a"))
        scheduler.add_task(DAGTask(task_id="b", dependencies=["a"]))
        scheduler.add_task(DAGTask(task_id="c", dependencies=["a"]))
        adj = scheduler._build_adjacency()
        self.assertEqual(set(adj["a"]), {"b", "c"})
        self.assertEqual(adj["b"], [])
        self.assertEqual(adj["c"], [])

    def test_max_qubits_default(self) -> None:
        """测试默认 max_qubits 为 287。"""
        scheduler = DAGScheduler()
        self.assertEqual(scheduler.max_qubits, 287)

    def test_max_qubits_custom(self) -> None:
        """测试自定义 max_qubits。"""
        scheduler = DAGScheduler(max_qubits=100)
        self.assertEqual(scheduler.max_qubits, 100)


if __name__ == "__main__":
    unittest.main()
