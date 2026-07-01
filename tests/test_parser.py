"""
量子RL调度系统 - 任务解析器单元测试
Unit Tests for src/scheduler/parser.py

测试覆盖：
- Task 数据类（字段、默认值）
- TaskBuilder 构造器（链式调用、校验、from_dict、build）
- TaskParser 解析器（parse / validate / _collect_errors 全部分支 / estimate_resources / to_internal_format）
- TaskFeatures 特征向量（to_vector 各分支与维度截断/填充）
- LegacyTaskParser 旧版解析器（json / yaml / qasm / text / batch_parse / 错误路径）
- QASM 解析边界（空串、注释、畸形、超大比特数、仅经典位等）
"""

import os
import sys
import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.parser import (
    MAX_CIRCUIT_DEPTH,
    MAX_QUBITS,
    MAX_SHOTS,
    PRIORITY_MAP,
    PRIORITY_REVERSE,
    LegacyTaskParser,
)
from src.scheduler.parser import Task as ParserTask
from src.scheduler.parser import (
    TaskBuilder,
    TaskFeatures,
    TaskParser,
)

# ============================================================
# Task 数据类
# ============================================================


class TestTask(unittest.TestCase):
    """测试 Task 规范化任务数据类。"""

    def test_required_fields_construction(self):
        """仅必填字段构造应成功并填充默认值。"""
        task = ParserTask(
            task_id="t1",
            task_type="quantum",
            qubits_required=4,
            estimated_time=10.0,
            priority=2,
        )
        self.assertEqual(task.task_id, "t1")
        self.assertEqual(task.task_type, "quantum")
        self.assertEqual(task.qubits_required, 4)
        self.assertEqual(task.estimated_time, 10.0)
        self.assertEqual(task.priority, 2)
        # 默认值
        self.assertIsNone(task.algorithm)
        self.assertIsNone(task.circuit_depth)
        self.assertIsNone(task.shots)
        self.assertIsNone(task.deadline)
        self.assertEqual(task.status, "pending")
        self.assertIsInstance(task.submitted_at, datetime)

    def test_full_construction(self):
        """全字段构造应保留所有值。"""
        deadline = datetime(2026, 8, 1, 12, 0, 0)
        task = ParserTask(
            task_id="t2",
            task_type="hybrid",
            qubits_required=8,
            estimated_time=120.0,
            priority=4,
            submitted_at=datetime(2026, 7, 1),
            algorithm="VQE",
            circuit_depth=50,
            shots=1024,
            deadline=deadline,
            status="queued",
        )
        self.assertEqual(task.algorithm, "VQE")
        self.assertEqual(task.circuit_depth, 50)
        self.assertEqual(task.shots, 1024)
        self.assertEqual(task.deadline, deadline)
        self.assertEqual(task.status, "queued")
        self.assertEqual(task.submitted_at, datetime(2026, 7, 1))


# ============================================================
# TaskBuilder
# ============================================================


class TestTaskBuilder(unittest.TestCase):
    """测试 Task Builder 模式构造器。"""

    def test_init_defaults(self):
        """初始化应设置全部默认值。"""
        b = TaskBuilder()
        self.assertEqual(b._data["task_id"], "")
        self.assertEqual(b._data["task_type"], "quantum")
        self.assertEqual(b._data["qubits_required"], 0)
        self.assertEqual(b._data["estimated_time"], 0.0)
        self.assertEqual(b._data["priority"], 2)
        self.assertIsNone(b._data["algorithm"])
        self.assertIsNone(b._data["deadline"])
        self.assertEqual(b._data["status"], "pending")

    def test_chaining_returns_self(self):
        """所有 set_* 方法应返回 self 以支持链式调用。"""
        b = TaskBuilder()
        self.assertIs(b.set_id("x"), b)
        self.assertIs(b.set_type("quantum"), b)
        self.assertIs(b.set_algorithm("VQE"), b)
        self.assertIs(b.set_qubits(4), b)
        self.assertIs(b.set_circuit_depth(10), b)
        self.assertIs(b.set_shots(64), b)
        self.assertIs(b.set_estimated_time(5.0), b)
        self.assertIs(b.set_priority("high"), b)
        self.assertIs(b.set_deadline(None), b)
        self.assertIs(b.set_status("queued"), b)
        self.assertIs(b.set_submitted_at(None), b)

    def test_set_type_invalid_raises(self):
        """无效 task_type 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            TaskBuilder().set_type("invalid")

    def test_set_type_case_insensitive(self):
        """task_type 应大小写不敏感并去空白。"""
        b = TaskBuilder().set_type("  QUANTUM  ")
        self.assertEqual(b._data["task_type"], "quantum")

    def test_set_priority_string_variants(self):
        """字符串 priority 应映射到 1-4。"""
        for label, val in PRIORITY_MAP.items():
            b = TaskBuilder().set_priority(label)
            self.assertEqual(b._data["priority"], val)

    def test_set_priority_int_in_range(self):
        """1-4 整数 priority 应被接受。"""
        for p in (1, 2, 3, 4):
            b = TaskBuilder().set_priority(p)
            self.assertEqual(b._data["priority"], p)

    def test_set_priority_int_out_of_range_raises(self):
        """超出 [1,4] 的整数 priority 应抛出 ValueError。"""
        for p in (0, 5, -1):
            with self.assertRaises(ValueError):
                TaskBuilder().set_priority(p)

    def test_set_priority_invalid_string_raises(self):
        """无效字符串 priority 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            TaskBuilder().set_priority("super_urgent")

    def test_set_deadline_none(self):
        """None deadline 应存储为 None。"""
        b = TaskBuilder().set_deadline(None)
        self.assertIsNone(b._data["deadline"])

    def test_set_deadline_datetime(self):
        """datetime deadline 应原样存储。"""
        dt = datetime(2026, 9, 1)
        b = TaskBuilder().set_deadline(dt)
        self.assertEqual(b._data["deadline"], dt)

    def test_set_deadline_iso_string(self):
        """ISO 字符串 deadline 应解析为 datetime。"""
        b = TaskBuilder().set_deadline("2026-09-01T12:00:00")
        self.assertEqual(b._data["deadline"], datetime(2026, 9, 1, 12, 0, 0))

    def test_set_status_invalid_raises(self):
        """无效 status 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            TaskBuilder().set_status("unknown")

    def test_set_status_valid(self):
        """合法 status 应被存储。"""
        for s in ("pending", "queued", "running", "completed", "failed"):
            b = TaskBuilder().set_status(s)
            self.assertEqual(b._data["status"], s)

    def test_set_submitted_at_none_uses_now(self):
        """submitted_at=None 应使用当前时间。"""
        before = datetime.now()
        b = TaskBuilder().set_submitted_at(None)
        after = datetime.now()
        self.assertGreaterEqual(b._data["submitted_at"], before)
        self.assertLessEqual(b._data["submitted_at"], after)

    def test_set_submitted_at_datetime(self):
        """submitted_at 显式 datetime 应原样存储。"""
        dt = datetime(2026, 1, 1)
        b = TaskBuilder().set_submitted_at(dt)
        self.assertEqual(b._data["submitted_at"], dt)

    def test_build_empty_id_raises(self):
        """空 task_id build 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            TaskBuilder().build()

    def test_build_basic(self):
        """完整链式 build 应返回 Task 实例。"""
        task = (
            TaskBuilder()
            .set_id("b_001")
            .set_type("quantum")
            .set_algorithm("Grover")
            .set_qubits(16)
            .set_circuit_depth(100)
            .set_shots(2048)
            .set_estimated_time(200)
            .set_priority("urgent")
            .set_deadline("2026-08-01T00:00:00")
            .build()
        )
        self.assertIsInstance(task, ParserTask)
        self.assertEqual(task.task_id, "b_001")
        self.assertEqual(task.algorithm, "Grover")
        self.assertEqual(task.qubits_required, 16)
        self.assertEqual(task.circuit_depth, 100)
        self.assertEqual(task.shots, 2048)
        self.assertEqual(task.priority, 4)

    def test_from_dict_full(self):
        """from_dict 应映射全部字段。"""
        d = {
            "task_id": "f1",
            "type": "hybrid",
            "algorithm": "QAOA",
            "qubits_required": 12,
            "circuit_depth": 30,
            "shots": 512,
            "estimated_time": 60.0,
            "priority": "low",
            "deadline": "2026-07-15T00:00:00",
            "status": "queued",
        }
        task = TaskBuilder.from_dict(d).build()
        self.assertEqual(task.task_id, "f1")
        self.assertEqual(task.task_type, "hybrid")
        self.assertEqual(task.algorithm, "QAOA")
        self.assertEqual(task.qubits_required, 12)
        self.assertEqual(task.priority, 1)
        self.assertEqual(task.status, "queued")
        self.assertIsInstance(task.deadline, datetime)

    def test_from_dict_type_alias(self):
        """from_dict 应兼容 task_type 键。"""
        task = TaskBuilder.from_dict({"task_id": "x", "task_type": "classical"}).build()
        self.assertEqual(task.task_type, "classical")

    def test_from_dict_qubit_alias(self):
        """from_dict 应兼容 qubit_count 键。"""
        task = TaskBuilder.from_dict(
            {"task_id": "x", "type": "classical", "qubit_count": 7}
        ).build()
        self.assertEqual(task.qubits_required, 7)

    def test_from_dict_priority_int(self):
        """from_dict 应接受整数 priority。"""
        task = TaskBuilder.from_dict({"task_id": "x", "type": "classical", "priority": 3}).build()
        self.assertEqual(task.priority, 3)

    def test_from_dict_defaults_when_missing(self):
        """from_dict 缺省字段应使用默认值。"""
        task = TaskBuilder.from_dict({"task_id": "x"}).build()
        self.assertEqual(task.task_type, "quantum")
        self.assertEqual(task.qubits_required, 0)
        self.assertEqual(task.estimated_time, 0.0)
        self.assertEqual(task.priority, 2)
        self.assertIsNone(task.algorithm)


# ============================================================
# TaskParser
# ============================================================


class TestTaskParser(unittest.TestCase):
    """测试新版任务解析器 TaskParser。"""

    def setUp(self):
        """构造解析器与样例字典。"""
        self.parser = TaskParser()
        self.sample = {
            "task_id": "task_001",
            "type": "quantum",
            "algorithm": "VQE",
            "qubits_required": 8,
            "circuit_depth": 50,
            "shots": 1024,
            "estimated_time": 120,
            "priority": "high",
            "deadline": "2026-12-01T12:00:00",
        }

    def test_init_defaults(self):
        """__init__ 应设置天衍-287 约束常量。"""
        self.assertEqual(self.parser.max_qubits, MAX_QUBITS)
        self.assertEqual(self.parser.max_circuit_depth, MAX_CIRCUIT_DEPTH)
        self.assertEqual(self.parser.max_shots, MAX_SHOTS)

    def test_parse_basic(self):
        """基本字典解析应返回 Task。"""
        task = self.parser.parse(self.sample)
        self.assertIsInstance(task, ParserTask)
        self.assertEqual(task.task_id, "task_001")
        self.assertEqual(task.task_type, "quantum")
        self.assertEqual(task.algorithm, "VQE")
        self.assertEqual(task.qubits_required, 8)
        self.assertEqual(task.priority, 3)
        self.assertIsInstance(task.deadline, datetime)

    def test_parse_not_dict_raises(self):
        """非字典输入应抛出 TypeError。"""
        with self.assertRaises(TypeError):
            self.parser.parse("not a dict")
        with self.assertRaises(TypeError):
            self.parser.parse(["list"])

    def test_parse_missing_task_id_raises(self):
        """缺少 task_id 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.parser.parse({"type": "quantum", "qubits_required": 8})

    def test_parse_invalid_type_raises(self):
        """无效 task_type 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.parser.parse({"task_id": "x", "type": "invalid"})

    def test_parse_invalid_priority_string_raises(self):
        """无效字符串 priority 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.parser.parse(dict(self.sample, priority="super_urgent"))

    def test_parse_invalid_priority_int_raises(self):
        """超出范围的整数 priority 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.parser.parse(dict(self.sample, priority=99))

    def test_parse_qubits_exceed_limit_raises(self):
        """量子任务 qubits 超过 287 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.parser.parse(dict(self.sample, qubits_required=999))

    def test_parse_circuit_depth_exceed_raises(self):
        """circuit_depth 超过上限应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.parser.parse(dict(self.sample, circuit_depth=MAX_CIRCUIT_DEPTH + 1))

    def test_parse_shots_exceed_raises(self):
        """shots 超过上限应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.parser.parse(dict(self.sample, shots=MAX_SHOTS + 1))

    def test_parse_quantum_zero_qubits_raises(self):
        """量子任务 qubits_required=0 应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.parser.parse({"task_id": "x", "type": "quantum", "qubits_required": 0})

    def test_parse_classical_allows_large_qubits(self):
        """经典任务不受量子比特上限约束。"""
        task = self.parser.parse(
            {
                "task_id": "x",
                "type": "classical",
                "qubits_required": 999,
                "estimated_time": 1,
                "priority": 2,
            }
        )
        self.assertEqual(task.qubits_required, 999)

    def test_validate_valid_returns_true(self):
        """合法任务 validate 应返回 True。"""
        task = self.parser.parse(self.sample)
        self.assertTrue(self.parser.validate(task))

    def test_validate_not_task_raises(self):
        """validate 非 Task 输入应抛出 TypeError。"""
        with self.assertRaises(TypeError):
            self.parser.validate("not a task")

    def test_validate_invalid_returns_false(self):
        """多重非法字段任务 validate 应返回 False。"""
        bad = ParserTask(
            task_id="",
            task_type="invalid",
            qubits_required=-1,
            estimated_time=-5.0,
            priority=99,
            status="bad",
        )
        # validate 会向 stderr 打印错误信息，这里抑制以保持测试输出整洁
        with patch("sys.stderr"):
            self.assertFalse(self.parser.validate(bad))

    # ---- _collect_errors 各分支 ----

    def _errors(self, **kwargs):
        """构造 Task 并收集错误列表。"""
        defaults = {
            "task_id": "ok",
            "task_type": "quantum",
            "qubits_required": 4,
            "estimated_time": 10.0,
            "priority": 2,
        }
        defaults.update(kwargs)
        task = ParserTask(**defaults)
        return self.parser._collect_errors(task)

    def test_errors_task_id_empty(self):
        """空 task_id 应报错。"""
        self.assertTrue(self._errors(task_id=""))

    def test_errors_task_type_invalid(self):
        """非法 task_type 应报错。"""
        self.assertTrue(self._errors(task_type="bad"))

    def test_errors_qubits_negative(self):
        """负 qubits 应报错。"""
        self.assertTrue(self._errors(qubits_required=-1))

    def test_errors_qubits_exceed_quantum(self):
        """量子任务 qubits 超限应报错。"""
        self.assertTrue(self._errors(qubits_required=MAX_QUBITS + 1))

    def test_errors_qubits_exceed_hybrid(self):
        """混合任务 qubits 超限应报错。"""
        self.assertTrue(self._errors(task_type="hybrid", qubits_required=MAX_QUBITS + 1))

    def test_errors_classical_qubits_no_limit(self):
        """经典任务 qubits 不受上限约束（仅非负校验）。"""
        errs = self._errors(task_type="classical", qubits_required=9999)
        self.assertFalse(errs)

    def test_errors_circuit_depth_negative(self):
        """负 circuit_depth 应报错。"""
        self.assertTrue(self._errors(circuit_depth=-1))

    def test_errors_circuit_depth_exceed(self):
        """circuit_depth 超限应报错。"""
        self.assertTrue(self._errors(circuit_depth=MAX_CIRCUIT_DEPTH + 1))

    def test_errors_shots_negative(self):
        """负 shots 应报错。"""
        self.assertTrue(self._errors(shots=-1))

    def test_errors_shots_exceed(self):
        """shots 超限应报错。"""
        self.assertTrue(self._errors(shots=MAX_SHOTS + 1))

    def test_errors_estimated_time_negative(self):
        """负 estimated_time 应报错。"""
        self.assertTrue(self._errors(estimated_time=-1.0))

    def test_errors_priority_out_of_range(self):
        """priority 超范围应报错。"""
        self.assertTrue(self._errors(priority=0))
        self.assertTrue(self._errors(priority=5))

    def test_errors_status_invalid(self):
        """非法 status 应报错。"""
        self.assertTrue(self._errors(status="bad"))

    def test_errors_deadline_not_datetime(self):
        """deadline 非 datetime 应报错。"""
        self.assertTrue(self._errors(deadline="not-a-datetime"))

    def test_errors_quantum_zero_qubits(self):
        """量子任务 qubits=0 应报错。"""
        self.assertTrue(self._errors(task_type="quantum", qubits_required=0))

    def test_errors_quantum_algorithm_not_string(self):
        """量子任务 algorithm 非字符串应报错。"""
        self.assertTrue(self._errors(algorithm=123))

    def test_errors_valid_task_no_errors(self):
        """合法任务应无错误。"""
        self.assertEqual(self._errors(), [])

    # ---- estimate_resources ----

    def test_estimate_resources_quantum(self):
        """量子任务资源预估应返回正确字段与 0.1 经典占比。"""
        task = self.parser.parse(self.sample)
        res = self.parser.estimate_resources(task)
        self.assertGreater(res["qubit_hours"], 0)
        self.assertEqual(res["total_gate_operations"], 50 * 1024)
        self.assertEqual(res["classical_compute_ratio"], 0.1)
        self.assertIn("memory_mb", res)
        self.assertIn("estimated_queue_time", res)

    def test_estimate_resources_classical(self):
        """经典任务经典计算占比应为 1.0。"""
        task = self.parser.parse(
            {
                "task_id": "c1",
                "type": "classical",
                "qubits_required": 0,
                "estimated_time": 60,
                "priority": 2,
            }
        )
        res = self.parser.estimate_resources(task)
        self.assertEqual(res["classical_compute_ratio"], 1.0)

    def test_estimate_resources_hybrid(self):
        """混合任务经典计算占比应为 0.5。"""
        task = self.parser.parse(
            {
                "task_id": "h1",
                "type": "hybrid",
                "qubits_required": 4,
                "estimated_time": 60,
                "priority": 2,
            }
        )
        res = self.parser.estimate_resources(task)
        self.assertEqual(res["classical_compute_ratio"], 0.5)

    def test_estimate_resources_large_qubits_sparse_path(self):
        """qubits>30 应走稀疏内存估算分支。"""
        task = self.parser.parse(
            {
                "task_id": "big",
                "type": "quantum",
                "qubits_required": 50,
                "circuit_depth": 100,
                "estimated_time": 10,
                "priority": 2,
            }
        )
        res = self.parser.estimate_resources(task)
        # 稀疏分支 memory_mb = depth * qubits * 0.001 = 100*50*0.001 = 5.0
        self.assertEqual(res["memory_mb"], 5.0)

    def test_estimate_resources_priority_factor(self):
        """不同 priority 应影响排队时间（priority 越高排队越短）。"""
        base = dict(self.sample)
        results = {}
        for p in (1, 2, 3, 4):
            t = self.parser.parse(dict(base, priority=p))
            results[p] = self.parser.estimate_resources(t)["estimated_queue_time"]
        self.assertGreater(results[1], results[4])

    def test_estimate_resources_no_depth_no_shots(self):
        """circuit_depth/shots 为 None 时应使用 0/1 默认。"""
        task = self.parser.parse(
            {
                "task_id": "n1",
                "type": "quantum",
                "qubits_required": 4,
                "estimated_time": 10,
                "priority": 2,
            }
        )
        res = self.parser.estimate_resources(task)
        self.assertEqual(res["total_gate_operations"], 0)

    def test_estimate_resources_unknown_priority_default_factor(self):
        """未知 priority 应使用默认排队因子 1.0（validate 不阻断执行）。"""
        task = ParserTask(
            task_id="u1",
            task_type="classical",
            qubits_required=2,
            estimated_time=10.0,
            priority=5,
        )
        with patch("sys.stderr"):
            res = self.parser.estimate_resources(task)
        self.assertIn("estimated_queue_time", res)

    # ---- to_internal_format ----

    def test_to_internal_format_with_future_deadline(self):
        """带未来 deadline 的内部格式应含 scheduling_weight 与 resource_estimate。"""
        task = self.parser.parse(self.sample)
        internal = self.parser.to_internal_format(task)
        self.assertEqual(internal["task_id"], "task_001")
        self.assertEqual(internal["priority_label"], "high")
        self.assertIn("resource_estimate", internal)
        self.assertIsInstance(internal["scheduling_weight"], float)
        self.assertGreater(internal["scheduling_weight"], 0)
        self.assertIsInstance(internal["deadline"], str)
        self.assertIsInstance(internal["submitted_at"], str)

    def test_to_internal_format_without_deadline(self):
        """无 deadline 时 deadline 字段应为 None。"""
        task = self.parser.parse(
            {
                "task_id": "nd",
                "type": "quantum",
                "qubits_required": 4,
                "estimated_time": 10,
                "priority": 2,
            }
        )
        internal = self.parser.to_internal_format(task)
        self.assertIsNone(internal["deadline"])
        self.assertGreater(internal["scheduling_weight"], 0)

    def test_to_internal_format_past_deadline(self):
        """过去 deadline 时紧迫度应保持 1.0（remaining<=0 分支）。"""
        past = (datetime.now() - timedelta(hours=1)).isoformat()
        task = self.parser.parse(
            {
                "task_id": "pd",
                "type": "quantum",
                "qubits_required": 4,
                "estimated_time": 10,
                "priority": 2,
                "deadline": past,
            }
        )
        internal = self.parser.to_internal_format(task)
        self.assertIsInstance(internal["deadline"], str)

    def test_to_internal_format_priority_label_all(self):
        """所有 priority 都应有对应 label。"""
        for p, label in PRIORITY_REVERSE.items():
            task = self.parser.parse(
                {
                    "task_id": "x",
                    "type": "quantum",
                    "qubits_required": 4,
                    "estimated_time": 10,
                    "priority": p,
                }
            )
            internal = self.parser.to_internal_format(task)
            self.assertEqual(internal["priority_label"], label)


# ============================================================
# TaskFeatures
# ============================================================


class TestTaskFeatures(unittest.TestCase):
    """测试 TaskFeatures 特征向量与 to_vector。"""

    def test_construction_defaults(self):
        """必填字段构造应填充默认值。"""
        f = TaskFeatures(task_id="t", user_id="u", task_type="quantum")
        self.assertEqual(f.qubit_count, 0)
        self.assertEqual(f.circuit_depth, 0)
        self.assertEqual(f.gate_count, 0)
        self.assertEqual(f.algorithm, "unknown")
        self.assertEqual(f.priority, 3)
        self.assertEqual(f.user_historical_completion_rate, 1.0)
        self.assertIsInstance(f.arrival_time, datetime)

    def test_to_vector_length_default(self):
        """默认维度应为 20。"""
        f = TaskFeatures(task_id="t", user_id="u", task_type="quantum")
        v = f.to_vector()
        self.assertEqual(len(v), 20)
        self.assertTrue(all(0.0 <= x <= 1.0 for x in v))

    def test_to_vector_quantum_onehot(self):
        """quantum 类型 one-hot 应在首位。"""
        f = TaskFeatures(task_id="t", user_id="u", task_type="quantum")
        v = f.to_vector(20)
        self.assertEqual(v[0:3], [1.0, 0.0, 0.0])

    def test_to_vector_classical_onehot(self):
        """classical 类型 one-hot 应在第二位。"""
        f = TaskFeatures(task_id="t", user_id="u", task_type="classical")
        v = f.to_vector(20)
        self.assertEqual(v[0:3], [0.0, 1.0, 0.0])

    def test_to_vector_hybrid_onehot(self):
        """hybrid 类型 one-hot 应在第三位。"""
        f = TaskFeatures(task_id="t", user_id="u", task_type="hybrid")
        v = f.to_vector(20)
        self.assertEqual(v[0:3], [0.0, 0.0, 1.0])

    def test_to_vector_algorithm_known(self):
        """已知算法应点亮对应 one-hot 位。"""
        for algo in ("VQE", "QAOA", "Grover", "Shor"):
            f = TaskFeatures(task_id="t", user_id="u", task_type="quantum", algorithm=algo)
            v = f.to_vector(20)
            self.assertIn(1.0, v[7:12], f"algo {algo} not one-hot")

    def test_to_vector_algorithm_other(self):
        """algorithm='Other' 应点亮 Other 位。"""
        f = TaskFeatures(task_id="t", user_id="u", task_type="quantum", algorithm="Other")
        v = f.to_vector(20)
        self.assertEqual(v[11], 1.0)

    def test_to_vector_algorithm_unknown_no_onehot(self):
        """未知算法（非列表内）不应点亮任何算法位。"""
        f = TaskFeatures(task_id="t", user_id="u", task_type="quantum", algorithm="weird")
        v = f.to_vector(20)
        self.assertEqual(v[7:12], [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_to_vector_with_deadline(self):
        """带 deadline 应计算剩余时间特征（>0）。"""
        f = TaskFeatures(
            task_id="t",
            user_id="u",
            task_type="quantum",
            deadline=datetime.now() + timedelta(days=2),
        )
        v = f.to_vector(20)
        self.assertGreater(v[15], 0.0)

    def test_to_vector_without_deadline(self):
        """无 deadline 应在时间特征位填 1.0。"""
        f = TaskFeatures(task_id="t", user_id="u", task_type="quantum")
        v = f.to_vector(20)
        self.assertEqual(v[15], 1.0)

    def test_to_vector_padding(self):
        """feature_dim 大于向量长度应补零。"""
        f = TaskFeatures(task_id="t", user_id="u", task_type="quantum")
        v = f.to_vector(30)
        self.assertEqual(len(v), 30)
        self.assertEqual(v[20:], [0.0] * 10)

    def test_to_vector_truncation(self):
        """feature_dim 小于向量长度应截断。"""
        f = TaskFeatures(task_id="t", user_id="u", task_type="quantum")
        v = f.to_vector(5)
        self.assertEqual(len(v), 5)


# ============================================================
# LegacyTaskParser
# ============================================================


class TestLegacyTaskParser(unittest.TestCase):
    """测试旧版任务解析器 LegacyTaskParser。"""

    def setUp(self):
        self.parser = LegacyTaskParser()

    def test_init_supported_formats(self):
        """__init__ 应设置支持的格式列表。"""
        self.assertEqual(self.parser.supported_formats, ["json", "yaml", "qasm", "text"])

    def test_parse_json(self):
        """JSON 格式解析应提取特征。"""
        s = (
            '{"task_id":"t1","user_id":"u","task_type":"quantum",'
            '"qubit_count":10,"circuit_depth":50,"algorithm":"VQE",'
            '"estimated_time":120.0,"priority":4}'
        )
        f = self.parser.parse(s, format="json")
        self.assertIsNotNone(f)
        self.assertEqual(f.task_id, "t1")
        self.assertEqual(f.qubit_count, 10)
        self.assertEqual(f.algorithm, "VQE")
        self.assertEqual(f.priority, 4)

    def test_parse_json_invalid_returns_none(self):
        """非法 JSON 应返回 None。"""
        self.assertIsNone(self.parser.parse("{not json", format="json"))

    def test_parse_json_defaults(self):
        """JSON 仅必填字段应使用默认值。"""
        f = self.parser.parse('{"task_id":"x"}', format="json")
        self.assertIsNotNone(f)
        self.assertEqual(f.task_id, "x")
        self.assertEqual(f.task_type, "quantum")
        self.assertEqual(f.qubit_count, 0)
        self.assertEqual(f.algorithm, "unknown")

    def test_parse_yaml(self):
        """YAML 格式解析应提取特征。"""
        s = "task_id: y1\nuser_id: u\ntask_type: quantum\nqubit_count: 5\nalgorithm: QAOA\n"
        f = self.parser.parse(s, format="yaml")
        self.assertIsNotNone(f)
        self.assertEqual(f.task_id, "y1")
        self.assertEqual(f.qubit_count, 5)
        self.assertEqual(f.algorithm, "QAOA")

    def test_parse_yaml_invalid_returns_none(self):
        """畸形 YAML（未闭合流式序列）应返回 None。"""
        self.assertIsNone(self.parser.parse("{a: [unclosed", format="yaml"))

    def test_parse_text_quantum(self):
        """文本含 quantum 关键词应识别为量子任务。"""
        f = self.parser.parse("a quantum task with 8 qubits using VQE", format="text")
        self.assertIsNotNone(f)
        self.assertEqual(f.task_type, "quantum")
        self.assertEqual(f.qubit_count, 8)
        self.assertEqual(f.algorithm, "VQE")

    def test_parse_text_classical(self):
        """文本含 classical 关键词应识别为经典任务。"""
        f = self.parser.parse("a classical job", format="text")
        self.assertIsNotNone(f)
        self.assertEqual(f.task_type, "classical")

    def test_parse_text_hybrid_default(self):
        """文本无类型关键词应默认 hybrid。"""
        f = self.parser.parse("an ordinary job", format="text")
        self.assertIsNotNone(f)
        self.assertEqual(f.task_type, "hybrid")

    def test_parse_text_qubit_chinese(self):
        """文本含中文'比特'应提取比特数。"""
        f = self.parser.parse("需要12比特的量子任务", format="text")
        self.assertIsNotNone(f)
        self.assertEqual(f.qubit_count, 12)

    def test_parse_text_algorithm_keywords(self):
        """文本应识别各算法关键词。"""
        for algo in ("VQE", "QAOA", "Grover", "Shor", "HHL"):
            f = self.parser.parse(f"task using {algo}", format="text")
            self.assertIsNotNone(f)
            self.assertEqual(f.algorithm, algo)

    def test_parse_unsupported_format_raises(self):
        """不支持的格式应抛出 ValueError。"""
        with self.assertRaises(ValueError):
            self.parser.parse("x", format="xml")

    def test_batch_parse_filters_none(self):
        """batch_parse 应过滤解析失败的项。"""
        descs = [
            '{"task_id":"t1","task_type":"quantum"}',
            "{broken",
            '{"task_id":"t2","task_type":"classical"}',
        ]
        results = self.parser.batch_parse(descs, format="json")
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].task_id, "t1")
        self.assertEqual(results[1].task_id, "t2")

    def test_batch_parse_empty(self):
        """空列表 batch_parse 应返回空列表。"""
        self.assertEqual(self.parser.batch_parse([], format="json"), [])

    def test_parse_text_internal_error_returns_none(self):
        """_parse_text 内部异常应被捕获并返回 None（健壮性边界）。"""
        with patch("src.scheduler.parser.re.search", side_effect=RuntimeError("boom")):
            self.assertIsNone(self.parser.parse("some text", format="text"))


# ============================================================
# QASM 解析边界
# ============================================================


class TestQasmParsing(unittest.TestCase):
    """测试 QASM 解析的边界与异常输入。"""

    def setUp(self):
        self.parser = LegacyTaskParser()
        self.bell = """
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q[0] -> c[0];
        """

    def test_bell_state_parsing(self):
        """Bell 态电路应正确提取比特数、门数与测量数。"""
        f = self.parser.parse(self.bell, format="qasm")
        self.assertIsNotNone(f)
        self.assertEqual(f.qubit_count, 2)
        self.assertEqual(f.gate_count, 2)
        self.assertEqual(f.measurement_count, 1)
        self.assertEqual(f.circuit_depth, 1)  # gate_count // 2

    def test_empty_string(self):
        """空字符串应返回 0 比特的特征（不抛异常）。"""
        f = self.parser.parse("", format="qasm")
        self.assertIsNotNone(f)
        self.assertEqual(f.qubit_count, 0)
        self.assertEqual(f.gate_count, 0)

    def test_whitespace_only(self):
        """纯空白输入应返回 0 比特特征。"""
        f = self.parser.parse("   \n\t  \n", format="qasm")
        self.assertIsNotNone(f)
        self.assertEqual(f.qubit_count, 0)

    def test_only_classical_bits(self):
        """仅含 creg 的电路应返回 0 比特。"""
        f = self.parser.parse("creg c[4];", format="qasm")
        self.assertIsNotNone(f)
        self.assertEqual(f.qubit_count, 0)

    def test_malformed_qreg_brackets_returns_none(self):
        """qreg 缺少比特数应触发异常并返回 None。"""
        self.assertIsNone(self.parser.parse("qreg q[];", format="qasm"))

    def test_large_qubit_count(self):
        """超大比特数应被正常解析（不在此层校验上限）。"""
        f = self.parser.parse("qreg q[500];", format="qasm")
        self.assertIsNotNone(f)
        self.assertEqual(f.qubit_count, 500)

    def test_comments_ignored(self):
        """注释行应被忽略，不影响门计数。"""
        qasm = """
        // this is a comment
        qreg q[3];
        // another comment
        h q[0];
        x q[1];
        measure q[0] -> c[0];
        """
        f = self.parser.parse(qasm, format="qasm")
        self.assertIsNotNone(f)
        self.assertEqual(f.qubit_count, 3)
        self.assertEqual(f.gate_count, 2)

    def test_multiple_measurements(self):
        """多测量行应累加 measurement_count。"""
        qasm = """
        qreg q[2];
        creg c[2];
        h q[0];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        """
        f = self.parser.parse(qasm, format="qasm")
        self.assertEqual(f.measurement_count, 2)

    def test_circuit_depth_floor_division(self):
        """circuit_depth 应为 gate_count//2（奇数门向下取整）。"""
        qasm = """
        qreg q[1];
        h q[0];
        x q[0];
        y q[0];
        """
        f = self.parser.parse(qasm, format="qasm")
        self.assertEqual(f.gate_count, 3)
        self.assertEqual(f.circuit_depth, 1)

    def test_no_qreg_line(self):
        """无 qreg 行时 qubit_count 应为 0。"""
        qasm = "h q[0];\nx q[1];"
        f = self.parser.parse(qasm, format="qasm")
        self.assertIsNotNone(f)
        self.assertEqual(f.qubit_count, 0)
        self.assertEqual(f.gate_count, 2)

    def test_various_gates_detected(self):
        """h/x/y/z/cx/cz 门应被识别计数。"""
        qasm = """
        qreg q[2];
        h q[0];
        x q[0];
        y q[0];
        z q[0];
        cx q[0],q[1];
        cz q[0],q[1];
        """
        f = self.parser.parse(qasm, format="qasm")
        self.assertEqual(f.gate_count, 6)

    def test_qasm_task_id_prefix(self):
        """QASM 解析生成的 task_id 应以 qasm_task_ 开头。"""
        f = self.parser.parse(self.bell, format="qasm")
        self.assertTrue(f.task_id.startswith("qasm_task_"))

    def test_qasm_estimated_time_proportional(self):
        """QASM estimated_time 应与 circuit_depth 成正比（×0.001）。"""
        qasm = """
        qreg q[1];
        h q[0];
        x q[0];
        y q[0];
        z q[0];
        """
        f = self.parser.parse(qasm, format="qasm")
        # gate_count=4 → depth=2 → estimated_time = 2 * 0.001
        self.assertAlmostEqual(f.estimated_time, 0.002)


if __name__ == "__main__":
    unittest.main()
