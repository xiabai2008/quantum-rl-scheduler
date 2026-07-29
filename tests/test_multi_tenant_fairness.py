"""
Issue #167: 多租户公平性调度 — 单元测试
"""

import math
import unittest

from src.scheduler.fairness import (
    MultiTenantFairnessTracker,
    TenantFairnessStats,
    jain_fairness_index,
    max_min_fairness,
)


class TestJainFairnessIndex(unittest.TestCase):
    """Jain Fairness Index 计算测试。"""

    def test_perfect_fairness(self):
        """完全相同值应返回 1.0。"""
        self.assertAlmostEqual(jain_fairness_index([1.0, 1.0, 1.0]), 1.0)
        self.assertAlmostEqual(jain_fairness_index([0.5, 0.5, 0.5, 0.5]), 1.0)

    def test_extreme_unfairness(self):
        """极端不公平应返回接近 1/n 的值。"""
        # 3 个值，只有一个非零 → FI = 1/3 ≈ 0.333
        fi = jain_fairness_index([1.0, 0.0, 0.0])
        self.assertAlmostEqual(fi, 1.0 / 3.0)

    def test_two_values(self):
        """两个值的 FI 计算。"""
        fi = jain_fairness_index([0.8, 0.4])
        expected = (0.8 + 0.4) ** 2 / (2 * (0.8**2 + 0.4**2))
        self.assertAlmostEqual(fi, expected)

    def test_all_zeros(self):
        """全零值应返回 0.0。"""
        self.assertEqual(jain_fairness_index([0.0, 0.0, 0.0]), 0.0)

    def test_empty_raises(self):
        """空列表应抛出异常。"""
        with self.assertRaises(ValueError):
            jain_fairness_index([])

    def test_single_value(self):
        """单值应返回 1.0。"""
        self.assertEqual(jain_fairness_index([0.7]), 1.0)

    def test_known_example(self):
        """已知测试用例：值域验证。"""
        fi = jain_fairness_index([0.9, 0.8, 0.7])
        self.assertGreater(fi, 0.9)
        self.assertLess(fi, 1.0)


class TestMaxMinFairness(unittest.TestCase):
    """Max-Min Fairness 比率测试。"""

    def test_perfect(self):
        self.assertEqual(max_min_fairness([0.5, 0.5, 0.5]), 1.0)

    def test_skewed(self):
        self.assertEqual(max_min_fairness([0.1, 0.5, 1.0]), 0.1)

    def test_empty(self):
        self.assertEqual(max_min_fairness([]), 0.0)

    def test_all_zeros(self):
        self.assertEqual(max_min_fairness([0.0, 0.0]), 0.0)


class TestTenantFairnessStats(unittest.TestCase):
    """租户统计测试。"""

    def test_defaults(self):
        stats = TenantFairnessStats()
        self.assertEqual(stats.completion_rate, 0.0)
        self.assertEqual(stats.avg_wait_steps, 0.0)

    def test_completion_rate(self):
        stats = TenantFairnessStats(
            tenant_id="test",
            tasks_submitted=10,
            tasks_completed=8,
            tasks_failed=2,
        )
        self.assertAlmostEqual(stats.completion_rate, 0.8)

    def test_avg_wait(self):
        stats = TenantFairnessStats(
            tasks_submitted=5,
            total_wait_steps=100,
        )
        self.assertEqual(stats.avg_wait_steps, 20.0)


class TestMultiTenantFairnessTracker(unittest.TestCase):
    """多租户公平性跟踪器测试。"""

    def setUp(self):
        self.tracker = MultiTenantFairnessTracker(["a", "b", "c"])

    def test_record_events(self):
        """记录提交/完成/失败事件。"""
        self.tracker.record_submit("a", wait_steps=5)
        self.tracker.record_submit("a")
        self.tracker.record_complete("a", exec_steps=3)
        self.tracker.record_complete("a", exec_steps=4)
        self.tracker.record_fail("b")
        self.tracker.record_submit("c")

        stats_a = self.tracker._stats["a"]
        self.assertEqual(stats_a.tasks_submitted, 2)
        self.assertEqual(stats_a.tasks_completed, 2)
        self.assertEqual(stats_a.total_wait_steps, 5)
        self.assertEqual(stats_a.total_exec_steps, 7)

        stats_b = self.tracker._stats["b"]
        self.assertEqual(stats_b.tasks_submitted, 0)
        self.assertEqual(stats_b.tasks_failed, 1)

    def test_perfect_fairness_tracker(self):
        """三租户同等完成时应返回 1.0。"""
        for tid in ["a", "b", "c"]:
            self.tracker.record_submit(tid)
            self.tracker.record_complete(tid)
        fi = self.tracker.jain_completion_fairness()
        self.assertAlmostEqual(fi, 1.0)

    def test_extreme_unfairness_tracker(self):
        """极不公平时应返回约 1/3。"""
        self.tracker.record_submit("a")
        self.tracker.record_complete("a")
        # b, c 有提交但无完成
        self.tracker.record_submit("b")
        self.tracker.record_submit("c")
        fi = self.tracker.jain_completion_fairness()
        self.assertAlmostEqual(fi, 1.0 / 3.0, delta=0.01)

    def test_unknown_tenant_auto_register(self):
        """未知租户应自动注册。"""
        self.tracker.record_submit("unknown_x")
        self.assertIn("unknown_x", self.tracker._stats)
        self.assertEqual(self.tracker._stats["unknown_x"].tasks_submitted, 1)

    def test_none_tenant_id(self):
        """None tenant_id 应映射到 "unknown"。"""
        self.tracker.record_submit(None)
        self.assertIn("unknown", self.tracker._stats)

    def test_summary_complete(self):
        """summary 应包含所有必要字段。"""
        self.tracker.record_submit("a")
        self.tracker.record_complete("a", exec_steps=5)
        s = self.tracker.summary()
        self.assertIn("jain_completion_fairness", s)
        self.assertIn("jain_wait_fairness", s)
        self.assertIn("max_min_completion_ratio", s)
        self.assertIn("per_tenant", s)
        self.assertIn("a", s["per_tenant"])

    def test_summary_table_format(self):
        """get_summary_table 应生成合法的 Markdown 表格。"""
        self.tracker.record_submit("a")
        self.tracker.record_complete("a")
        table = self.tracker.get_summary_table()
        self.assertIn("租户", table)
        self.assertIn("完成率", table)
        self.assertIn("a", table)
        self.assertIn("b", table)
        self.assertIn("c", table)


if __name__ == "__main__":
    unittest.main()
