"""
量子RL调度系统 - 真机配额追踪器单元测试
Unit Tests for src/api/quota_tracker.py

测试覆盖：
- QuotaTracker 配置加载（正常/缺失配置文件回退默认值/格式错误回退）
- consume 正常消费与持久化（state 文件落盘与重载）
- can_consume 边界（配额恰好用完、超额返回 False、刚好等于配额允许）
- usage_ratio 计算（含总配额为 0 的边界）
- status 返回结构与 warning_level 判断（normal/warning/critical）
- check_and_alert 日志告警（warning/critical/normal 三态）
- estimated_exhaustion_time 估算（有/无历史数据）
- record_daily_usage 与 get_daily_history（覆盖当日、保留 30 天）
- QuotaExhaustedError 异常层次与属性
- 配额耗尽时 tianyan_cqlib 集成（submit_quantum_task 返回 None）
- MultiMachineCqlibCoordinator 配额集成（成功后 consume）

所有持久化文件均写入 tempfile.TemporaryDirectory，避免污染真实 logs/ 目录。
"""

import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.api.quota_tracker import QuotaExhaustedError, QuotaTracker
from src.api.tianyan_cqlib import CqlibTianyanClient, MultiMachineCqlibCoordinator
from src.exceptions import ResourceExhaustedError


def _write_config(path: str, total: dict | None = None, warning: float = 0.8, critical: float = 0.95) -> None:
    """写入测试用配额配置文件。"""
    import yaml

    cfg = {
        "total_quota": total or {"shots": 1000, "tasks": 20, "wall_time_hours": 5},
        "warning_threshold": warning,
        "critical_threshold": critical,
        "notification": {"type": "log", "webhook_url": None},
    }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)


class TestQuotaTrackerConfig(unittest.TestCase):
    """测试 QuotaTracker 配置加载逻辑。"""

    def test_load_normal_config(self):
        """正常配置文件应被正确加载为总配额与阈值。"""
        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "quota.yaml")
            state_path = os.path.join(d, "state.json")
            _write_config(cfg_path, total={"shots": 500, "tasks": 10, "wall_time_hours": 2})
            tracker = QuotaTracker(config_path=cfg_path, state_path=state_path)
            self.assertEqual(tracker._total_quota["shots"], 500)
            self.assertEqual(tracker._total_quota["tasks"], 10)
            self.assertEqual(tracker._total_quota["wall_time_hours"], 2)
            self.assertEqual(tracker._warning_threshold, 0.8)
            self.assertEqual(tracker._critical_threshold, 0.95)

    def test_missing_config_falls_back_to_defaults(self):
        """配置文件缺失时应使用默认配额（shots=10000, tasks=200, wall_time_hours=50）。"""
        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "nonexistent.yaml")
            state_path = os.path.join(d, "state.json")
            tracker = QuotaTracker(config_path=cfg_path, state_path=state_path)
            self.assertEqual(tracker._total_quota["shots"], 10000)
            self.assertEqual(tracker._total_quota["tasks"], 200)
            self.assertEqual(tracker._total_quota["wall_time_hours"], 50)
            self.assertEqual(tracker._warning_threshold, 0.8)
            self.assertEqual(tracker._critical_threshold, 0.95)

    def test_corrupt_config_falls_back_to_defaults(self):
        """配置文件格式错误（非法 YAML）应回退默认值而不抛异常。"""
        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "quota.yaml")
            state_path = os.path.join(d, "state.json")
            with open(cfg_path, "w", encoding="utf-8") as f:
                f.write("total_quota: [unclosed bracket\n  - bad: : :")
            tracker = QuotaTracker(config_path=cfg_path, state_path=state_path)
            self.assertEqual(tracker._total_quota["shots"], 10000)

    def test_missing_state_file_initializes_zero(self):
        """状态文件不存在时已用量应初始化为 0。"""
        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "quota.yaml")
            state_path = os.path.join(d, "state.json")
            _write_config(cfg_path)
            tracker = QuotaTracker(config_path=cfg_path, state_path=state_path)
            for dim in ("shots", "tasks", "wall_time_hours"):
                self.assertEqual(tracker._used[dim], 0.0)
            self.assertEqual(tracker._daily_history, [])

    def test_corrupt_state_falls_back_to_zero(self):
        """状态文件损坏（非法 JSON）应回退为 0 而不抛异常。"""
        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "quota.yaml")
            state_path = os.path.join(d, "state.json")
            _write_config(cfg_path)
            with open(state_path, "w", encoding="utf-8") as f:
                f.write("{not valid json")
            tracker = QuotaTracker(config_path=cfg_path, state_path=state_path)
            self.assertEqual(tracker._used["shots"], 0.0)


class TestQuotaTrackerConsume(unittest.TestCase):
    """测试 consume / can_consume / remaining 消费与检查逻辑。"""

    def setUp(self):
        """每个测试使用独立的临时目录与默认配额（shots=1000, tasks=20）。"""
        self._tmp = tempfile.TemporaryDirectory()
        self.cfg_path = os.path.join(self._tmp.name, "quota.yaml")
        self.state_path = os.path.join(self._tmp.name, "state.json")
        _write_config(self.cfg_path)
        self.tracker = QuotaTracker(config_path=self.cfg_path, state_path=self.state_path)

    def tearDown(self):
        """清理临时目录。"""
        self._tmp.cleanup()

    def test_consume_normal_increments_used(self):
        """正常消费应递增各维度已用量。"""
        ok = self.tracker.consume(shots=100, tasks=2, wall_time_hours=1.0)
        self.assertTrue(ok)
        self.assertEqual(self.tracker._used["shots"], 100)
        self.assertEqual(self.tracker._used["tasks"], 2)
        self.assertEqual(self.tracker._used["wall_time_hours"], 1.0)

    def test_consume_persists_state_to_file(self):
        """消费后状态应持久化到 state 文件。"""
        self.tracker.consume(shots=200, tasks=1)
        self.assertTrue(os.path.exists(self.state_path))
        with open(self.state_path, encoding="utf-8") as f:
            state = json.load(f)
        self.assertEqual(state["used"]["shots"], 200)
        self.assertEqual(state["used"]["tasks"], 1)

    def test_consume_reload_restores_state(self):
        """新实例加载 state 文件应恢复已用量。"""
        self.tracker.consume(shots=300, tasks=3, wall_time_hours=0.5)
        reloaded = QuotaTracker(config_path=self.cfg_path, state_path=self.state_path)
        self.assertEqual(reloaded._used["shots"], 300)
        self.assertEqual(reloaded._used["tasks"], 3)
        self.assertEqual(reloaded._used["wall_time_hours"], 0.5)

    def test_consume_over_limit_returns_false(self):
        """超额消费应返回 False 且不修改已用量。"""
        # 总配额 shots=1000, tasks=20
        ok = self.tracker.consume(shots=1500, tasks=1)
        self.assertFalse(ok)
        # 未修改
        self.assertEqual(self.tracker._used["shots"], 0.0)
        self.assertEqual(self.tracker._used["tasks"], 0.0)

    def test_consume_partial_over_limit_rolls_back(self):
        """某维度超额时整体拒绝（不部分扣减）。"""
        # shots 在配额内但 tasks 超额
        ok = self.tracker.consume(shots=500, tasks=25)
        self.assertFalse(ok)
        self.assertEqual(self.tracker._used["shots"], 0.0)
        self.assertEqual(self.tracker._used["tasks"], 0.0)

    def test_can_consume_within_limit_returns_true(self):
        """配额内 can_consume 应返回 True 且不修改状态。"""
        self.assertTrue(self.tracker.can_consume(shots=500, tasks=10))
        self.assertEqual(self.tracker._used["shots"], 0.0)

    def test_can_consume_exactly_at_limit_returns_true(self):
        """恰好用完配额（used + request == total）应允许消费。"""
        # shots=1000, 一次消费 1000 恰好用完
        self.assertTrue(self.tracker.can_consume(shots=1000, tasks=1))
        ok = self.tracker.consume(shots=1000, tasks=1)
        self.assertTrue(ok)

    def test_can_consume_over_limit_returns_false(self):
        """超额时 can_consume 应返回 False。"""
        self.assertFalse(self.tracker.can_consume(shots=1001, tasks=1))

    def test_remaining_after_consume(self):
        """remaining 应正确返回各维度剩余量。"""
        self.tracker.consume(shots=300, tasks=5, wall_time_hours=2.0)
        rem = self.tracker.remaining()
        self.assertEqual(rem["shots"], 700)
        self.assertEqual(rem["tasks"], 15)
        self.assertAlmostEqual(rem["wall_time_hours"], 3.0)

    def test_remaining_never_negative(self):
        """remaining 在超额场景下不应为负（clamp 到 0）。"""
        # 直接篡改 used 模拟超额边界
        self.tracker._used["shots"] = 1500
        rem = self.tracker.remaining()
        self.assertEqual(rem["shots"], 0.0)


class TestQuotaTrackerUsageRatio(unittest.TestCase):
    """测试 usage_ratio 计算逻辑。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cfg_path = os.path.join(self._tmp.name, "quota.yaml")
        self.state_path = os.path.join(self._tmp.name, "state.json")
        _write_config(cfg_path)
        self.tracker = QuotaTracker(config_path=cfg_path, state_path=self.state_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_usage_ratio_zero_initially(self):
        """初始使用比例应为 0。"""
        ratio = self.tracker.usage_ratio()
        self.assertEqual(ratio["shots"], 0.0)
        self.assertEqual(ratio["tasks"], 0.0)

    def test_usage_ratio_after_consume(self):
        """消费后使用比例应正确计算。"""
        # shots=1000, 消费 250 -> 0.25
        self.tracker.consume(shots=250)
        ratio = self.tracker.usage_ratio()
        self.assertAlmostEqual(ratio["shots"], 0.25)
        self.assertAlmostEqual(ratio["tasks"], 1 / 20)

    def test_usage_ratio_zero_quota_returns_zero(self):
        """总配额为 0 的维度使用比例应返回 0（避免除零）。"""
        self.tracker._total_quota["shots"] = 0
        ratio = self.tracker.usage_ratio()
        self.assertEqual(ratio["shots"], 0.0)


class TestQuotaTrackerStatus(unittest.TestCase):
    """测试 status 返回结构与 warning_level 判断。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cfg_path = os.path.join(self._tmp.name, "quota.yaml")
        self.state_path = os.path.join(self._tmp.name, "state.json")
        _write_config(cfg_path, total={"shots": 1000, "tasks": 20, "wall_time_hours": 5})

    def tearDown(self):
        self._tmp.cleanup()

    def _make_tracker(self):
        return QuotaTracker(config_path=os.path.join(self._tmp.name, "quota.yaml"), state_path=self.state_path)

    def test_status_structure_contains_all_fields(self):
        """status 应包含 total/used/remaining/usage_ratio/warning_level 等字段。"""
        tracker = self._make_tracker()
        s = tracker.status()
        for key in ("total", "used", "remaining", "usage_ratio", "warning_level",
                    "warning_threshold", "critical_threshold", "estimated_exhaustion_time"):
            self.assertIn(key, s, f"status 缺少字段: {key}")

    def test_warning_level_normal(self):
        """使用比例低于 warning 阈值时 warning_level=normal。"""
        tracker = self._make_tracker()
        # shots 1000 * 0.8 = 800 才到 warning；消费 100 -> 0.1
        tracker.consume(shots=100)
        self.assertEqual(tracker.status()["warning_level"], "normal")

    def test_warning_level_warning(self):
        """达到 warning 阈值（0.8）时 warning_level=warning。"""
        tracker = self._make_tracker()
        tracker.consume(shots=810)  # 0.81 >= 0.8
        self.assertEqual(tracker.status()["warning_level"], "warning")

    def test_warning_level_critical(self):
        """达到 critical 阈值（0.95）时 warning_level=critical。"""
        tracker = self._make_tracker()
        tracker.consume(shots=960)  # 0.96 >= 0.95
        self.assertEqual(tracker.status()["warning_level"], "critical")

    def test_warning_level_uses_max_ratio_across_dims(self):
        """warning_level 应取所有维度中的最高比例判断。"""
        tracker = self._make_tracker()
        # tasks=20, 消费 19 -> 0.95 -> critical
        tracker.consume(tasks=19)
        self.assertEqual(tracker.status()["warning_level"], "critical")


class TestQuotaTrackerCheckAndAlert(unittest.TestCase):
    """测试 check_and_alert 日志告警逻辑。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cfg_path = os.path.join(self._tmp.name, "quota.yaml")
        self.state_path = os.path.join(self._tmp.name, "state.json")
        _write_config(cfg_path)

    def tearDown(self):
        self._tmp.cleanup()

    def _make_tracker(self):
        return QuotaTracker(config_path=os.path.join(self._tmp.name, "quota.yaml"), state_path=self.state_path)

    def test_check_and_alert_normal_returns_none(self):
        """未达阈值时 check_and_alert 应返回 None。"""
        tracker = self._make_tracker()
        self.assertIsNone(tracker.check_and_alert())

    def test_check_and_alert_warning_returns_warning(self):
        """达到 warning 阈值应返回 'warning'。"""
        tracker = self._make_tracker()
        tracker.consume(shots=810)
        self.assertEqual(tracker.check_and_alert(), "warning")

    def test_check_and_alert_critical_returns_critical(self):
        """达到 critical 阈值应返回 'critical'。"""
        tracker = self._make_tracker()
        tracker.consume(shots=960)
        self.assertEqual(tracker.check_and_alert(), "critical")

    def test_check_and_alert_logs_warning(self):
        """warning 级别应触发 logger.warning 调用。"""
        tracker = self._make_tracker()
        tracker.consume(shots=850)
        with patch("src.api.quota_tracker.logger") as mock_logger:
            result = tracker.check_and_alert()
        self.assertEqual(result, "warning")
        mock_logger.warning.assert_called_once()

    def test_check_and_alert_logs_critical(self):
        """critical 级别应触发 logger.critical 调用。"""
        tracker = self._make_tracker()
        tracker.consume(shots=970)
        with patch("src.api.quota_tracker.logger") as mock_logger:
            result = tracker.check_and_alert()
        self.assertEqual(result, "critical")
        mock_logger.critical.assert_called_once()


class TestQuotaTrackerExhaustionEstimate(unittest.TestCase):
    """测试 estimated_exhaustion_time 估算逻辑。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cfg_path = os.path.join(self._tmp.name, "quota.yaml")
        self.state_path = os.path.join(self._tmp.name, "state.json")
        _write_config(cfg_path, total={"shots": 1000, "tasks": 20, "wall_time_hours": 5})

    def tearDown(self):
        self._tmp.cleanup()

    def _make_tracker(self):
        return QuotaTracker(config_path=os.path.join(self._tmp.name, "quota.yaml"), state_path=self.state_path)

    def test_no_history_returns_none(self):
        """无历史数据时 estimated_exhaustion_time 应为 None。"""
        tracker = self._make_tracker()
        s = tracker.status()
        self.assertIsNone(s["estimated_exhaustion_time"])

    def test_with_history_returns_estimates(self):
        """有历史数据时应返回各维度估算字典。"""
        tracker = self._make_tracker()
        tracker.consume(shots=100, tasks=2)
        tracker.record_daily_usage()
        s = tracker.status()
        est = s["estimated_exhaustion_time"]
        self.assertIsNotNone(est)
        self.assertIn("shots", est)
        self.assertIn("tasks", est)
        self.assertIn("wall_time_hours", est)
        # 剩余 900 / 日均 100 = 9 天
        self.assertAlmostEqual(est["shots"]["days"], 9.0)
        self.assertIsNotNone(est["shots"]["date"])

    def test_zero_daily_avg_returns_none_days(self):
        """某维度日均消耗为 0 时该维度 days 应为 None。"""
        tracker = self._make_tracker()
        # 只消费 shots，不消费 tasks
        tracker.consume(shots=100, tasks=0)
        tracker.record_daily_usage()
        s = tracker.status()
        est = s["estimated_exhaustion_time"]
        self.assertIsNotNone(est)
        self.assertIsNone(est["tasks"]["days"])

    def test_record_daily_usage_overwrites_same_day(self):
        """同一天多次 record_daily_usage 应覆盖当日记录而非追加。"""
        tracker = self._make_tracker()
        tracker.consume(shots=100)
        tracker.record_daily_usage()
        tracker.consume(shots=50)
        tracker.record_daily_usage()
        history = tracker.get_daily_history()
        # 同一天只有一条记录，值为最新用量 150
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["shots"], 150)

    def test_get_daily_history_returns_copy(self):
        """get_daily_history 应返回副本，修改不影响内部状态。"""
        tracker = self._make_tracker()
        tracker.consume(shots=100)
        tracker.record_daily_usage()
        history = tracker.get_daily_history()
        history.append({"date": "2099-01-01", "shots": 999})
        self.assertEqual(len(tracker.get_daily_history()), 1)


class TestQuotaExhaustedError(unittest.TestCase):
    """测试 QuotaExhaustedError 异常类型。"""

    def test_is_resource_exhausted_error_subclass(self):
        """QuotaExhaustedError 应是 ResourceExhaustedError 的子类。"""
        self.assertTrue(issubclass(QuotaExhaustedError, ResourceExhaustedError))

    def test_raises_and_carries_attributes(self):
        """异常应可被 raise 并携带 dimension/used/total 属性。"""
        with self.assertRaises(QuotaExhaustedError) as ctx:
            raise QuotaExhaustedError("shots", used=1500, total=1000)
        err = ctx.exception
        self.assertEqual(err.dimension, "shots")
        self.assertEqual(err.used, 1500)
        self.assertEqual(err.total, 1000)
        # 应同时是 ResourceExhaustedError 实例（便于上层统一捕获）
        self.assertIsInstance(err, ResourceExhaustedError)
        # 错误码默认 QUOTA_EXHAUSTED
        self.assertEqual(err.code, "QUOTA_EXHAUSTED")

    def test_caught_by_base_class(self):
        """ResourceExhaustedError 应能捕获 QuotaExhaustedError。"""
        with self.assertRaises(ResourceExhaustedError):
            raise QuotaExhaustedError("tasks", used=25, total=20)


class TestCqlibClientQuotaIntegration(unittest.TestCase):
    """测试 CqlibTianyanClient 配额集成（platform 用 mock 替代）。"""

    def setUp(self):
        """创建 cqlib 客户端（注入 mock platform）与配额追踪器。"""
        self._tmp = tempfile.TemporaryDirectory()
        cfg_path = os.path.join(self._tmp.name, "quota.yaml")
        self.state_path = os.path.join(self._tmp.name, "state.json")
        _write_config(cfg_path, total={"shots": 1000, "tasks": 20, "wall_time_hours": 5})
        self.tracker = QuotaTracker(config_path=cfg_path, state_path=self.state_path)

        self.client = CqlibTianyanClient(
            login_key="fake-key",
            machine_name="tianyan_s",
            auto_retry_machine=True,
            quota_tracker=self.tracker,
        )
        self.client._platform = MagicMock()
        self.client._platform.query_quantum_computer_list.return_value = [
            ("id1", "superconducting", "running", "tianyan_s"),
        ]

    def tearDown(self):
        self._tmp.cleanup()

    def test_submit_success_consumes_quota(self):
        """提交成功后应扣减配额。"""
        self.client._platform.submit_experiment.return_value = ["tid-1"]
        tid = self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=128)
        self.assertEqual(tid, "tid-1")
        self.assertEqual(self.tracker._used["shots"], 128)
        self.assertEqual(self.tracker._used["tasks"], 1)

    def test_submit_quota_exhausted_returns_none(self):
        """配额耗尽时提交应跳过并返回 None（不抛异常）。"""
        # 先把 shots 配额耗尽
        self.tracker.consume(shots=1000, tasks=1)
        self.client._platform.submit_experiment.return_value = ["tid-2"]
        tid = self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64)
        self.assertIsNone(tid)
        # 未调用 submit_experiment
        self.client._platform.submit_experiment.assert_not_called()

    def test_submit_failure_does_not_consume(self):
        """提交失败（抛异常）不应扣减配额。"""
        self.client._platform.submit_experiment.side_effect = Exception("校准中")
        # auto_retry=True 会尝试备用机；mock _retry_other_machine 也失败返回 None
        with (
            patch.object(self.client, "_is_machine_available", return_value=True),
            patch.object(self.client, "_retry_other_machine", return_value=None),
        ):
            tid = self.client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64)
        self.assertIsNone(tid)
        # 配额未扣减
        self.assertEqual(self.tracker._used["shots"], 0.0)
        self.assertEqual(self.tracker._used["tasks"], 0.0)

    def test_no_quota_tracker_keeps_backward_compat(self):
        """未传入 quota_tracker 时行为与原有一致（不检查、不扣减）。"""
        client = CqlibTianyanClient(login_key="k", machine_name="tianyan_s")
        client._platform = MagicMock()
        client._platform.query_quantum_computer_list.return_value = [
            ("id1", "superconducting", "running", "tianyan_s"),
        ]
        client._platform.submit_experiment.return_value = ["tid-3"]
        tid = client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64)
        self.assertEqual(tid, "tid-3")
        self.assertIsNone(client._quota_tracker)


class TestMultiMachineCoordinatorQuotaIntegration(unittest.TestCase):
    """测试 MultiMachineCqlibCoordinator 配额集成。"""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        cfg_path = os.path.join(self._tmp.name, "quota.yaml")
        self.state_path = os.path.join(self._tmp.name, "state.json")
        _write_config(cfg_path)
        self.tracker = QuotaTracker(config_path=cfg_path, state_path=self.state_path)

    def tearDown(self):
        self._tmp.cleanup()

    def test_submit_success_consumes_quota(self):
        """协调器提交成功后应扣减配额。"""
        coord = MultiMachineCqlibCoordinator(
            login_key="key",
            machine_names=["tianyan_s"],
            quota_tracker=self.tracker,
        )
        mock_client = MagicMock()
        mock_client.submit_quantum_task.return_value = "tid-1"
        with patch.object(coord, "_get_client", return_value=mock_client):
            tid = coord.submit_to_machine("tianyan_s", "H Q0\nM Q0", shots=128)
        self.assertEqual(tid, "tid-1")
        self.assertEqual(self.tracker._used["shots"], 128)
        self.assertEqual(self.tracker._used["tasks"], 1)

    def test_submit_failure_does_not_consume(self):
        """协调器提交失败（返回 None）不应扣减配额。"""
        coord = MultiMachineCqlibCoordinator(
            login_key="key",
            machine_names=["tianyan_s"],
            quota_tracker=self.tracker,
        )
        mock_client = MagicMock()
        mock_client.submit_quantum_task.return_value = None
        with patch.object(coord, "_get_client", return_value=mock_client):
            tid = coord.submit_to_machine("tianyan_s", "H Q0\nM Q0", shots=128)
        self.assertIsNone(tid)
        self.assertEqual(self.tracker._used["shots"], 0.0)

    def test_no_quota_tracker_keeps_backward_compat(self):
        """未传入 quota_tracker 时协调器行为不变。"""
        coord = MultiMachineCqlibCoordinator(
            login_key="key",
            machine_names=["tianyan_s"],
        )
        self.assertIsNone(coord._quota_tracker)
        mock_client = MagicMock()
        mock_client.submit_quantum_task.return_value = "tid-2"
        with patch.object(coord, "_get_client", return_value=mock_client):
            tid = coord.submit_to_machine("tianyan_s", "H Q0\nM Q0", shots=64)
        self.assertEqual(tid, "tid-2")


class TestQuotaTrackerThreadSafety(unittest.TestCase):
    """测试 QuotaTracker 线程安全（并发 consume 不超额）。"""

    def test_concurrent_consume_respects_limit(self):
        """多线程并发 consume 时总扣减不应超过配额上限。"""
        import threading

        with tempfile.TemporaryDirectory() as d:
            cfg_path = os.path.join(d, "quota.yaml")
            state_path = os.path.join(d, "state.json")
            # tasks 配额设为 10，每个线程尝试消费 1 个 task
            _write_config(cfg_path, total={"shots": 10000, "tasks": 10, "wall_time_hours": 50})
            tracker = QuotaTracker(config_path=cfg_path, state_path=state_path)

            results: list[bool] = []
            results_lock = threading.Lock()
            barrier = threading.Barrier(20)

            def worker():
                barrier.wait()
                ok = tracker.consume(tasks=1)
                with results_lock:
                    results.append(ok)

            threads = [threading.Thread(target=worker) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            # 配额 tasks=10，20 个线程并发，只有 10 个应成功
            self.assertEqual(sum(results), 10)
            self.assertEqual(tracker._used["tasks"], 10)


if __name__ == "__main__":
    unittest.main()
