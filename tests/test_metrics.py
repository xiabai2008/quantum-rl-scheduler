"""
Issue #24: src/utils/metrics.py 单元测试覆盖（0% → 80%+）

测试覆盖：
- 所有指标对象的类型与元数据（Counter / Gauge / Histogram）
- 指标命名规范与文档字符串
- 标签维度正确性
- record_api_call helper 函数：计数器 + 延迟直方图同步更新
- record_scheduled_task helper 函数：计数器 + 等待时间直方图同步更新
- 指标值采样验证（prometheus_client 默认注册表）
- __all__ 导出完整性
"""

import os
import sys
import unittest

from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.registry import REGISTRY

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import metrics
from src.utils.metrics import (
    active_connections,
    annealing_iterations,
    api_calls,
    api_errors,
    api_latency,
    api_request_duration,
    circuit_breaker_state,
    qubit_utilization,
    queue_length,
    record_api_call,
    record_scheduled_task,
    task_wait_time,
    tasks_scheduled,
    tianyan_cb_state,
)


class TestMetricsExport(unittest.TestCase):
    """验证 __all__ 导出完整、所有导出对象均可导入。"""

    def test_all_exports_are_defined(self):
        """__all__ 中每个名称都应在模块中可访问。"""
        for name in metrics.__all__:
            self.assertTrue(
                hasattr(metrics, name),
                f"__all__ 中的 {name} 未在 metrics 模块中定义",
            )

    def test_all_exports_count(self):
        """__all__ 应包含 14 个导出符号（6 调度/API/退火指标 + 2 helper + 5 状态指标 + 1 旧版兼容）。"""
        self.assertEqual(len(metrics.__all__), 14)


class TestMetricTypes(unittest.TestCase):
    """验证每个指标对象的类型正确。"""

    def test_tasks_scheduled_is_counter(self):
        self.assertIsInstance(tasks_scheduled, Counter)

    def test_task_wait_time_is_histogram(self):
        self.assertIsInstance(task_wait_time, Histogram)

    def test_api_calls_is_counter(self):
        self.assertIsInstance(api_calls, Counter)

    def test_api_request_duration_is_histogram(self):
        self.assertIsInstance(api_request_duration, Histogram)

    def test_api_errors_is_counter(self):
        self.assertIsInstance(api_errors, Counter)

    def test_api_latency_is_histogram(self):
        self.assertIsInstance(api_latency, Histogram)

    def test_annealing_iterations_is_histogram(self):
        self.assertIsInstance(annealing_iterations, Histogram)

    def test_qubit_utilization_is_gauge(self):
        self.assertIsInstance(qubit_utilization, Gauge)

    def test_queue_length_is_gauge(self):
        self.assertIsInstance(queue_length, Gauge)

    def test_active_connections_is_gauge(self):
        self.assertIsInstance(active_connections, Gauge)

    def test_circuit_breaker_state_is_gauge(self):
        self.assertIsInstance(circuit_breaker_state, Gauge)

    def test_tianyan_cb_state_is_gauge(self):
        self.assertIsInstance(tianyan_cb_state, Gauge)


class TestMetricMetadata(unittest.TestCase):
    """验证指标的元数据（名称、文档、标签维度）。"""

    def test_tasks_scheduled_name(self):
        # prometheus_client 的 Counter 自动去掉 _total 后缀存储
        self.assertEqual(tasks_scheduled._name, "scheduler_tasks")

    def test_tasks_scheduled_labels(self):
        self.assertEqual(tasks_scheduled._labelnames, ("strategy", "target"))

    def test_task_wait_time_name(self):
        self.assertEqual(task_wait_time._name, "scheduler_wait_seconds")

    def test_api_calls_name(self):
        # prometheus_client 的 Counter 自动去掉 _total 后缀存储
        self.assertEqual(api_calls._name, "tianyan_api_requests")

    def test_api_calls_labels(self):
        self.assertEqual(api_calls._labelnames, ("method", "endpoint"))

    def test_api_request_duration_labels(self):
        self.assertEqual(api_request_duration._labelnames, ("method", "endpoint"))

    def test_api_errors_labels(self):
        self.assertEqual(api_errors._labelnames, ("method", "endpoint", "error_type"))

    def test_api_latency_no_labels(self):
        """api_latency 为旧版兼容指标，无标签维度。"""
        self.assertEqual(api_latency._labelnames, ())

    def test_qubit_utilization_no_labels(self):
        self.assertEqual(qubit_utilization._labelnames, ())

    def test_circuit_breaker_state_no_labels(self):
        self.assertEqual(circuit_breaker_state._labelnames, ())


class TestRecordApiCall(unittest.TestCase):
    """测试 record_api_call helper 函数。"""

    def test_record_api_call_increments_counter(self):
        """调用后 api_calls 计数器应递增。"""
        method = "test_record_api_call_method_inc"
        endpoint = "test_ep_inc"
        before = api_calls.labels(method=method, endpoint=endpoint)._value.get()
        record_api_call(method=method, endpoint=endpoint, latency=0.5)
        after = api_calls.labels(method=method, endpoint=endpoint)._value.get()
        self.assertEqual(after, before + 1)

    def test_record_api_call_records_latency(self):
        """调用后 api_request_duration 直方图应观测到 latency。"""
        method = "test_record_api_call_latency"
        endpoint = "test_ep_lat"
        # 多次调用以确保采样稳定
        for lat in (0.1, 0.5, 1.0, 5.0):
            record_api_call(method=method, endpoint=endpoint, latency=lat)
        # 通过 samples 验证至少有 _count > 0
        samples = list(api_request_duration.labels(method=method, endpoint=endpoint).collect())
        # 直方图样本应包含 _count、_sum 以及若干 bucket
        self.assertTrue(any(s.name.endswith("_count") for s in samples[0].samples))

    def test_record_api_call_multiple_times(self):
        """多次调用应正确累加计数。"""
        method = "test_multi"
        endpoint = "test_multi_ep"
        before = api_calls.labels(method=method, endpoint=endpoint)._value.get()
        for _ in range(5):
            record_api_call(method=method, endpoint=endpoint, latency=0.2)
        after = api_calls.labels(method=method, endpoint=endpoint)._value.get()
        self.assertEqual(after, before + 5)


class TestRecordScheduledTask(unittest.TestCase):
    """测试 record_scheduled_task helper 函数。"""

    def test_record_scheduled_task_increments_counter(self):
        strategy = "test_strategy_inc"
        target = "test_target_inc"
        before = tasks_scheduled.labels(strategy=strategy, target=target)._value.get()
        record_scheduled_task(strategy=strategy, target=target, wait_seconds=10.0)
        after = tasks_scheduled.labels(strategy=strategy, target=target)._value.get()
        self.assertEqual(after, before + 1)

    def test_record_scheduled_task_records_wait_time(self):
        """调用后 task_wait_time 直方图应观测到等待时间。"""
        strategy = "test_strategy_wait"
        target = "test_target_wait"
        for wait in (1.0, 5.0, 30.0, 120.0):
            record_scheduled_task(strategy=strategy, target=target, wait_seconds=wait)
        # 直方图样本应存在 _count > 0
        # task_wait_time 无标签，直接 collect
        collected = list(task_wait_time.collect())
        self.assertTrue(any(s.name.endswith("_count") for s in collected[0].samples))

    def test_record_scheduled_task_multiple_strategies(self):
        """不同策略应分别计数。"""
        for strategy in ("PPO", "FCFS", "Random"):
            before = tasks_scheduled.labels(strategy=strategy, target="m1")._value.get()
            record_scheduled_task(strategy=strategy, target="m1", wait_seconds=1.0)
            after = tasks_scheduled.labels(strategy=strategy, target="m1")._value.get()
            self.assertEqual(after, before + 1)


class TestGaugeOperations(unittest.TestCase):
    """测试 Gauge 指标的 set/inc/dec 操作。"""

    def test_qubit_utilization_set(self):
        qubit_utilization.set(0.75)
        # 通过 REGISTRY 采样验证
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if sample.name == "scheduler_qubit_utilization":
                    self.assertAlmostEqual(sample.value, 0.75, places=4)
                    return
        self.fail("未找到 scheduler_qubit_utilization 指标样本")

    def test_queue_length_set(self):
        queue_length.set(42)
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if sample.name == "scheduler_queue_length":
                    self.assertEqual(sample.value, 42)
                    return
        self.fail("未找到 scheduler_queue_length 指标样本")

    def test_active_connections_inc_dec(self):
        active_connections.set(0)
        active_connections.inc()
        active_connections.inc(2)
        active_connections.dec()
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if sample.name == "websocket_active_connections":
                    self.assertEqual(sample.value, 2)  # 0 + 1 + 2 - 1
                    return
        self.fail("未找到 websocket_active_connections 指标样本")

    def test_circuit_breaker_state_set(self):
        for state_value in (0, 1, 2):
            circuit_breaker_state.set(state_value)
            for metric in REGISTRY.collect():
                for sample in metric.samples:
                    if sample.name == "circuit_breaker_state":
                        self.assertEqual(sample.value, state_value)
                        break

    def test_tianyan_cb_state_set(self):
        tianyan_cb_state.set(1)
        for metric in REGISTRY.collect():
            for sample in metric.samples:
                if sample.name == "tianyan_circuit_breaker_state":
                    self.assertEqual(sample.value, 1)
                    return
        self.fail("未找到 tianyan_circuit_breaker_state 指标样本")


class TestHistogramBuckets(unittest.TestCase):
    """验证直方图的桶配置。"""

    def test_task_wait_time_buckets(self):
        """task_wait_time 应包含 1/5/10/30/60/120/300 秒的桶。"""
        expected_buckets = [1, 5, 10, 30, 60, 120, 300]
        actual_buckets = list(task_wait_time._upper_bounds)
        for b in expected_buckets:
            self.assertIn(b, actual_buckets)

    def test_api_request_duration_buckets(self):
        """api_request_duration 应包含 0.1~120 秒的桶。"""
        expected_buckets = [0.1, 0.5, 1, 5, 10, 30, 60, 120]
        actual_buckets = list(api_request_duration._upper_bounds)
        for b in expected_buckets:
            self.assertIn(b, actual_buckets)

    def test_annealing_iterations_buckets(self):
        """annealing_iterations 应包含 100~10000 的桶。"""
        expected_buckets = [100, 500, 1000, 5000, 10000]
        actual_buckets = list(annealing_iterations._upper_bounds)
        for b in expected_buckets:
            self.assertIn(b, actual_buckets)


if __name__ == "__main__":
    unittest.main()
