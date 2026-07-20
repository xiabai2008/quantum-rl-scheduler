"""
量子RL调度系统 - 决策可解释性追踪模块单元测试
Unit Tests for src/scheduler/explainability.py

测试覆盖：
- TestDecisionRecord            : 数据类创建、序列化 round-trip
- TestDecisionExplainerExplain  : explain 返回完整字段、贡献度归一化
- TestFormatExplanation         : 中文格式化、top_k 控制、无 q_values 情况
- TestFeatureImportance         : 聚合重要性、排序
- TestDetectAnomalies           : 低 action_prob 检测、正常决策不误报
- TestSummarizeSession          : 汇总统计、动作分布
- TestDecisionLogger            : 日志写入/加载/clear、JSONL 格式
- TestEdgeCases                 : 空记录、单条记录、全零状态
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from src.scheduler.explainability import (
    STATE_FEATURE_NAMES,
    DecisionExplainer,
    DecisionLogger,
    DecisionRecord,
)


# ============================================================
# TestDecisionRecord 测试
# ============================================================
class TestDecisionRecord(unittest.TestCase):
    """测试 DecisionRecord 数据类。"""

    def test_create_record(self):
        """应正确创建 DecisionRecord 实例。"""
        state = np.arange(14, dtype=np.float64)
        contribs = {f"f{i}": float(i) / 100.0 for i in range(14)}
        rec = DecisionRecord(
            step=5,
            state=state,
            action=1,
            action_prob=0.8,
            q_values=np.array([1.0, 2.0, 3.0]),
            feature_contributions=contribs,
            timestamp="2026-07-02T10:00:00",
        )
        self.assertEqual(rec.step, 5)
        self.assertEqual(rec.action, 1)
        self.assertAlmostEqual(rec.action_prob, 0.8)
        self.assertEqual(rec.timestamp, "2026-07-02T10:00:00")
        np.testing.assert_array_equal(rec.state, state)

    def test_to_dict_serialization(self):
        """to_dict 应返回可 JSON 序列化的字典。"""
        rec = DecisionRecord(
            step=1,
            state=np.array([0.1, 0.2, 0.3]),
            action=0,
            action_prob=0.5,
            q_values=np.array([1.0, 2.0]),
            feature_contributions={"a": 0.6, "b": 0.4},
            timestamp="2026-07-02T10:00:00",
        )
        d = rec.to_dict()
        # 应可 JSON 序列化（不抛异常）
        s = json.dumps(d, ensure_ascii=False)
        d2 = json.loads(s)
        self.assertEqual(d2["step"], 1)
        self.assertEqual(d2["action"], 0)
        self.assertAlmostEqual(d2["action_prob"], 0.5)
        self.assertEqual(d2["state"], [0.1, 0.2, 0.3])
        self.assertEqual(d2["q_values"], [1.0, 2.0])
        self.assertEqual(d2["feature_contributions"], {"a": 0.6, "b": 0.4})

    def test_to_dict_with_none_q_values(self):
        """q_values 为 None 时 to_dict 应输出 null。"""
        rec = DecisionRecord(
            step=1,
            state=np.array([0.1, 0.2]),
            action=0,
            action_prob=0.5,
            q_values=None,
            feature_contributions={"a": 0.5, "b": 0.5},
            timestamp="t",
        )
        d = rec.to_dict()
        self.assertIsNone(d["q_values"])

    def test_from_dict_round_trip(self):
        """from_dict 应正确还原 to_dict 的输出。"""
        rec = DecisionRecord(
            step=3,
            state=np.array([0.4, 0.5]),
            action=2,
            action_prob=0.7,
            q_values=None,
            feature_contributions={"x": 0.5, "y": 0.5},
            timestamp="2026-07-02T11:00:00",
        )
        d = rec.to_dict()
        rec2 = DecisionRecord.from_dict(d)
        self.assertEqual(rec2.step, rec.step)
        self.assertEqual(rec2.action, rec.action)
        np.testing.assert_array_almost_equal(rec2.state, rec.state)
        self.assertIsNone(rec2.q_values)
        self.assertEqual(rec2.feature_contributions, rec.feature_contributions)
        self.assertEqual(rec2.timestamp, rec.timestamp)

    def test_from_dict_with_q_values(self):
        """from_dict 应正确还原带 q_values 的记录。"""
        rec = DecisionRecord(
            step=2,
            state=np.array([0.1, 0.2, 0.3]),
            action=1,
            action_prob=0.9,
            q_values=np.array([1.0, 2.0, 3.0]),
            feature_contributions={"a": 0.3, "b": 0.3, "c": 0.4},
            timestamp="t",
        )
        rec2 = DecisionRecord.from_dict(rec.to_dict())
        self.assertIsNotNone(rec2.q_values)
        np.testing.assert_array_almost_equal(rec2.q_values, rec.q_values)


# ============================================================
# TestDecisionExplainerExplain 测试
# ============================================================
class TestDecisionExplainerExplain(unittest.TestCase):
    """测试 DecisionExplainer.explain。"""

    def setUp(self):
        self.explainer = DecisionExplainer()
        self.state = np.array(
            [0.8, 0.3, 0.6, 0.5, 0.4, 0.2, 0.1, 0.05, 0.3, 0.7, 0.5, 0.2, 0.9, 0.1],
            dtype=np.float64,
        )

    def test_returns_decision_record(self):
        """explain 应返回 DecisionRecord 实例。"""
        rec = self.explainer.explain(self.state, action=1, step=2)
        self.assertIsInstance(rec, DecisionRecord)

    def test_record_fields_complete(self):
        """返回的记录应包含所有必要字段。"""
        rec = self.explainer.explain(
            self.state,
            action=1,
            q_values=np.array([1.0, 3.0, 2.0]),
            action_prob=0.7,
            step=4,
        )
        self.assertEqual(rec.step, 4)
        self.assertEqual(rec.action, 1)
        self.assertAlmostEqual(rec.action_prob, 0.7)
        self.assertIsNotNone(rec.q_values)
        self.assertEqual(len(rec.feature_contributions), 14)
        self.assertTrue(rec.timestamp)

    def test_contributions_normalized_with_q_values(self):
        """有 q_values 时贡献度应归一化（和≈1）。"""
        rec = self.explainer.explain(self.state, action=1, q_values=np.array([1.0, 5.0, 2.0]))
        total = sum(rec.feature_contributions.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_contributions_normalized_without_q_values(self):
        """无 q_values 时贡献度应归一化（和≈1）。"""
        rec = self.explainer.explain(self.state, action=0)
        total = sum(rec.feature_contributions.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_contributions_keys_are_feature_names(self):
        """贡献度字典的键应为标准 14 个特征名。"""
        rec = self.explainer.explain(self.state, action=0)
        for name in STATE_FEATURE_NAMES:
            self.assertIn(name, rec.feature_contributions)
        self.assertEqual(len(rec.feature_contributions), 14)

    def test_q_values_stored_correctly(self):
        """q_values 应被正确存储（一维数组）。"""
        qv = np.array([1.0, 2.0, 3.0])
        rec = self.explainer.explain(self.state, action=2, q_values=qv)
        np.testing.assert_array_almost_equal(rec.q_values, qv)

    def test_q_values_none_when_not_provided(self):
        """未提供 q_values 时字段应为 None。"""
        rec = self.explainer.explain(self.state, action=0)
        self.assertIsNone(rec.q_values)

    def test_custom_feature_names(self):
        """应支持自定义特征名。"""
        custom = ["a", "b", "c"]
        explainer = DecisionExplainer(feature_names=custom)
        rec = explainer.explain(np.array([0.1, 0.2, 0.3]), action=0)
        self.assertEqual(set(rec.feature_contributions.keys()), {"a", "b", "c"})

    def test_contributions_are_non_negative(self):
        """所有贡献度应为非负数。"""
        rec = self.explainer.explain(self.state, action=1, q_values=np.array([1.0, 3.0, 2.0]))
        for v in rec.feature_contributions.values():
            self.assertGreaterEqual(v, 0.0)

    def test_default_action_prob_is_one(self):
        """未提供 action_prob 时默认应为 1.0。"""
        rec = self.explainer.explain(self.state, action=0)
        self.assertAlmostEqual(rec.action_prob, 1.0)

    def test_default_step_is_zero(self):
        """未提供 step 时默认应为 0。"""
        rec = self.explainer.explain(self.state, action=0)
        self.assertEqual(rec.step, 0)


# ============================================================
# TestFormatExplanation 测试
# ============================================================
class TestFormatExplanation(unittest.TestCase):
    """测试 format_explanation。"""

    def setUp(self):
        self.explainer = DecisionExplainer()
        self.state = np.array(
            [0.9, 0.1, 0.8, 0.2, 0.3, 0.1, 0.1, 0.1, 0.2, 0.6, 0.4, 0.1, 0.7, 0.1],
            dtype=np.float64,
        )

    def test_chinese_format(self):
        """中文格式化应包含步数、动作、影响因素。"""
        rec = self.explainer.explain(self.state, action=1, step=7)
        text = self.explainer.format_explanation(rec, top_k=3)
        self.assertIn("第7步", text)
        self.assertIn("动作1", text)
        self.assertIn("影响因素", text)
        # 应包含等级标注（高/中/低）
        self.assertTrue("高" in text or "中" in text or "低" in text)

    def test_top_k_controls_count(self):
        """top_k 应控制显示的影响因素数量。"""
        rec = self.explainer.explain(self.state, action=0, step=1)
        text3 = self.explainer.format_explanation(rec, top_k=3)
        text5 = self.explainer.format_explanation(rec, top_k=5)
        # 每个因素包含一个"值="，按出现次数计数
        self.assertEqual(text3.count("值="), 3)
        self.assertEqual(text5.count("值="), 5)

    def test_format_without_q_values(self):
        """无 q_values 的记录也应能格式化。"""
        rec = self.explainer.explain(self.state, action=2, step=3)
        text = self.explainer.format_explanation(rec, top_k=5)
        self.assertIn("第3步", text)
        self.assertIn("动作2", text)

    def test_english_format(self):
        """lang='en' 应输出英文格式。"""
        rec = self.explainer.explain(self.state, action=1, step=2)
        text = self.explainer.format_explanation(rec, top_k=3, lang="en")
        self.assertIn("Step 2", text)
        self.assertIn("action 1", text)

    def test_top_k_zero(self):
        """top_k=0 应不显示任何影响因素。"""
        rec = self.explainer.explain(self.state, action=0, step=1)
        text = self.explainer.format_explanation(rec, top_k=0)
        self.assertEqual(text.count("值="), 0)

    def test_format_includes_feature_values(self):
        """格式化文本应包含特征对应的状态值。"""
        rec = self.explainer.explain(self.state, action=0, step=1)
        text = self.explainer.format_explanation(rec, top_k=5)
        # 文本中应出现数值（形如 值=0.xxx）
        self.assertIn("值=0.", text)


# ============================================================
# TestFeatureImportance 测试
# ============================================================
class TestFeatureImportance(unittest.TestCase):
    """测试 get_feature_importance。"""

    def test_empty_records(self):
        """空记录列表应返回空字典。"""
        explainer = DecisionExplainer()
        self.assertEqual(explainer.get_feature_importance([]), {})

    def test_aggregate_mean(self):
        """应正确计算多条记录的均值贡献度。"""
        explainer = DecisionExplainer(feature_names=["a", "b"])
        rec1 = DecisionRecord(
            step=0,
            state=np.array([1.0, 0.0]),
            action=0,
            action_prob=1.0,
            q_values=None,
            feature_contributions={"a": 1.0, "b": 0.0},
            timestamp="t1",
        )
        rec2 = DecisionRecord(
            step=1,
            state=np.array([0.0, 1.0]),
            action=1,
            action_prob=1.0,
            q_values=None,
            feature_contributions={"a": 0.0, "b": 1.0},
            timestamp="t2",
        )
        imp = explainer.get_feature_importance([rec1, rec2])
        self.assertAlmostEqual(imp["a"], 0.5)
        self.assertAlmostEqual(imp["b"], 0.5)

    def test_single_record(self):
        """单条记录的聚合重要性应等于该记录的贡献度。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c"])
        rec = DecisionRecord(
            step=0,
            state=np.array([1.0, 1.0, 1.0]),
            action=0,
            action_prob=1.0,
            q_values=None,
            feature_contributions={"a": 0.5, "b": 0.3, "c": 0.2},
            timestamp="t",
        )
        imp = explainer.get_feature_importance([rec])
        self.assertAlmostEqual(imp["a"], 0.5)
        self.assertAlmostEqual(imp["b"], 0.3)
        self.assertAlmostEqual(imp["c"], 0.2)

    def test_sorted_by_importance(self):
        """聚合结果应可用于排序（高贡献度在前）。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c"])
        rec = DecisionRecord(
            step=0,
            state=np.array([1.0, 1.0, 1.0]),
            action=0,
            action_prob=1.0,
            q_values=None,
            feature_contributions={"a": 0.6, "b": 0.3, "c": 0.1},
            timestamp="t",
        )
        imp = explainer.get_feature_importance([rec])
        sorted_items = sorted(imp.items(), key=lambda kv: kv[1], reverse=True)
        self.assertEqual(sorted_items[0][0], "a")
        self.assertEqual(sorted_items[-1][0], "c")


# ============================================================
# TestDetectAnomalies 测试
# ============================================================
class TestDetectAnomalies(unittest.TestCase):
    """测试 detect_anomalies。"""

    def test_low_action_prob_detected(self):
        """action_prob < 0.3 应被检测为异常。"""
        explainer = DecisionExplainer(feature_names=["a", "b"])
        rec = DecisionRecord(
            step=0,
            state=np.array([1.0, 0.0]),
            action=0,
            action_prob=0.2,
            q_values=None,
            feature_contributions={"a": 0.5, "b": 0.5},
            timestamp="t",
        )
        anomalies = explainer.detect_anomalies([rec])
        self.assertEqual(anomalies, [0])

    def test_normal_decision_not_flagged(self):
        """正常决策（高置信度、均匀贡献）不应被误报。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c", "d"])
        # 均匀贡献：max/mean = 1.0 < threshold 2.0
        rec = DecisionRecord(
            step=0,
            state=np.array([1.0, 1.0, 1.0, 1.0]),
            action=0,
            action_prob=0.95,
            q_values=None,
            feature_contributions={"a": 0.25, "b": 0.25, "c": 0.25, "d": 0.25},
            timestamp="t",
        )
        anomalies = explainer.detect_anomalies([rec])
        self.assertEqual(anomalies, [])

    def test_multiple_anomalies_indices(self):
        """应返回多个异常记录的正确索引。"""
        explainer = DecisionExplainer(feature_names=["a", "b"])
        recs = [
            DecisionRecord(  # 正常
                step=0,
                state=np.array([1.0, 1.0]),
                action=0,
                action_prob=0.9,
                q_values=None,
                feature_contributions={"a": 0.5, "b": 0.5},
                timestamp="t0",
            ),
            DecisionRecord(  # 异常：低置信度
                step=1,
                state=np.array([1.0, 0.0]),
                action=1,
                action_prob=0.1,
                q_values=None,
                feature_contributions={"a": 0.5, "b": 0.5},
                timestamp="t1",
            ),
            DecisionRecord(  # 正常
                step=2,
                state=np.array([0.5, 0.5]),
                action=0,
                action_prob=0.85,
                q_values=None,
                feature_contributions={"a": 0.5, "b": 0.5},
                timestamp="t2",
            ),
        ]
        anomalies = explainer.detect_anomalies(recs)
        self.assertEqual(anomalies, [1])

    def test_empty_records(self):
        """空记录列表应返回空列表。"""
        explainer = DecisionExplainer()
        self.assertEqual(explainer.detect_anomalies([]), [])

    def test_concentrated_contribution_anomaly(self):
        """贡献过度集中（max/mean > threshold）应被检测为异常。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c", "d"])
        # a 占 0.7，其余各 0.1；mean=0.25，max/mean=2.8 > 2.0
        rec = DecisionRecord(
            step=0,
            state=np.array([1.0, 0.0, 0.0, 0.0]),
            action=0,
            action_prob=0.9,
            q_values=None,
            feature_contributions={"a": 0.7, "b": 0.1, "c": 0.1, "d": 0.1},
            timestamp="t",
        )
        anomalies = explainer.detect_anomalies([rec], threshold=2.0)
        self.assertEqual(anomalies, [0])

    def test_custom_threshold(self):
        """自定义 threshold 应影响检测灵敏度。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c", "d"])
        # a 占 0.4，其余各 0.2；mean=0.25，max/mean=1.6
        rec = DecisionRecord(
            step=0,
            state=np.array([1.0, 0.0, 0.0, 0.0]),
            action=0,
            action_prob=0.9,
            q_values=None,
            feature_contributions={"a": 0.4, "b": 0.2, "c": 0.2, "d": 0.2},
            timestamp="t",
        )
        # threshold=1.5 → 1.6 > 1.5 触发
        self.assertEqual(explainer.detect_anomalies([rec], threshold=1.5), [0])
        # threshold=2.0 → 1.6 < 2.0 不触发
        self.assertEqual(explainer.detect_anomalies([rec], threshold=2.0), [])


# ============================================================
# TestSummarizeSession 测试
# ============================================================
class TestSummarizeSession(unittest.TestCase):
    """测试 summarize_session。"""

    def test_summary_statistics(self):
        """应正确汇总总步数、动作分布、异常数。"""
        explainer = DecisionExplainer(feature_names=["a", "b"])
        recs = [
            DecisionRecord(
                step=0,
                state=np.array([1.0, 0.0]),
                action=0,
                action_prob=0.9,
                q_values=None,
                feature_contributions={"a": 0.6, "b": 0.4},
                timestamp="t0",
            ),
            DecisionRecord(
                step=1,
                state=np.array([0.0, 1.0]),
                action=1,
                action_prob=0.9,
                q_values=None,
                feature_contributions={"a": 0.3, "b": 0.7},
                timestamp="t1",
            ),
            DecisionRecord(
                step=2,
                state=np.array([1.0, 1.0]),
                action=0,
                action_prob=0.2,
                q_values=None,
                feature_contributions={"a": 0.5, "b": 0.5},
                timestamp="t2",
            ),
        ]
        summary = explainer.summarize_session(recs)
        self.assertEqual(summary["total_steps"], 3)
        self.assertEqual(summary["action_distribution"], {0: 2, 1: 1})
        self.assertEqual(summary["anomaly_count"], 1)
        self.assertIn("top5_features", summary)
        self.assertLessEqual(len(summary["top5_features"]), 5)

    def test_empty_session(self):
        """空会话应返回零值统计。"""
        explainer = DecisionExplainer()
        summary = explainer.summarize_session([])
        self.assertEqual(summary["total_steps"], 0)
        self.assertEqual(summary["action_distribution"], {})
        self.assertEqual(summary["anomaly_count"], 0)
        self.assertEqual(summary["top5_features"], [])

    def test_top5_features_sorted_desc(self):
        """top5_features 应按重要性降序排列。"""
        explainer = DecisionExplainer(feature_names=["a", "b", "c"])
        rec = DecisionRecord(
            step=0,
            state=np.array([1.0, 1.0, 1.0]),
            action=0,
            action_prob=0.9,
            q_values=None,
            feature_contributions={"a": 0.1, "b": 0.7, "c": 0.2},
            timestamp="t",
        )
        summary = explainer.summarize_session([rec])
        importances = [item["importance"] for item in summary["top5_features"]]
        self.assertEqual(importances, sorted(importances, reverse=True))
        self.assertEqual(summary["top5_features"][0]["feature"], "b")

    def test_action_distribution_keys(self):
        """动作分布应以动作编号为键。"""
        explainer = DecisionExplainer(feature_names=["a", "b"])
        recs = [
            DecisionRecord(
                step=0,
                state=np.array([1.0, 0.0]),
                action=0,
                action_prob=0.9,
                q_values=None,
                feature_contributions={"a": 0.5, "b": 0.5},
                timestamp="t0",
            ),
            DecisionRecord(
                step=1,
                state=np.array([0.0, 1.0]),
                action=2,
                action_prob=0.9,
                q_values=None,
                feature_contributions={"a": 0.5, "b": 0.5},
                timestamp="t1",
            ),
        ]
        summary = explainer.summarize_session(recs)
        self.assertIn(0, summary["action_distribution"])
        self.assertIn(2, summary["action_distribution"])


# ============================================================
# TestDecisionLogger 测试
# ============================================================
class TestDecisionLogger(unittest.TestCase):
    """测试 DecisionLogger。"""

    def test_log_and_load_round_trip(self):
        """写入的记录应能完整加载。"""
        with tempfile.TemporaryDirectory() as tmp:
            logger_obj = DecisionLogger(log_dir=tmp)
            explainer = DecisionExplainer()
            rec = explainer.explain(
                np.arange(14, dtype=np.float64),
                action=1,
                q_values=np.array([1.0, 2.0, 3.0]),
                action_prob=0.8,
                step=3,
            )
            logger_obj.log(rec)
            loaded = logger_obj.load()
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].step, 3)
            self.assertEqual(loaded[0].action, 1)
            self.assertAlmostEqual(loaded[0].action_prob, 0.8)
            np.testing.assert_array_almost_equal(loaded[0].state, np.arange(14, dtype=np.float64))

    def test_jsonl_format(self):
        """日志文件应为 JSONL 格式（每行一个 JSON 对象）。"""
        with tempfile.TemporaryDirectory() as tmp:
            logger_obj = DecisionLogger(log_dir=tmp)
            explainer = DecisionExplainer()
            r1 = explainer.explain(np.zeros(14), action=0, step=0)
            r2 = explainer.explain(np.ones(14), action=1, step=1)
            logger_obj.log(r1)
            logger_obj.log(r2)
            self.assertTrue(os.path.exists(logger_obj.log_path))
            with open(logger_obj.log_path, encoding="utf-8") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 2)
            for line in lines:
                obj = json.loads(line)  # 每行应是合法 JSON
                self.assertIn("step", obj)
                self.assertIn("feature_contributions", obj)

    def test_clear(self):
        """clear 应清空日志文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            logger_obj = DecisionLogger(log_dir=tmp)
            explainer = DecisionExplainer()
            logger_obj.log(explainer.explain(np.zeros(14), action=0, step=0))
            self.assertEqual(len(logger_obj.load()), 1)
            logger_obj.clear()
            self.assertEqual(len(logger_obj.load()), 0)

    def test_load_nonexistent_file(self):
        """加载不存在的日志文件应返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            logger_obj = DecisionLogger(log_dir=tmp)
            # 删除文件以模拟不存在
            if os.path.exists(logger_obj.log_path):
                os.remove(logger_obj.log_path)
            self.assertEqual(logger_obj.load(), [])

    def test_utf8_encoding(self):
        """日志文件应以 UTF-8 编码保存中文特征名。"""
        with tempfile.TemporaryDirectory() as tmp:
            logger_obj = DecisionLogger(log_dir=tmp)
            explainer = DecisionExplainer()  # 默认中文特征名
            rec = explainer.explain(np.arange(14, dtype=np.float64), action=0, step=0)
            logger_obj.log(rec)
            with open(logger_obj.log_path, encoding="utf-8") as f:
                content = f.read()
            self.assertIn("队列长度", content)  # 中文特征名应正确保存

    def test_creates_log_dir(self):
        """初始化时应自动创建日志目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "nested", "decisions")
            DecisionLogger(log_dir=log_dir)
            self.assertTrue(os.path.isdir(log_dir))

    def test_multiple_logs_appended(self):
        """多次 log 应追加而非覆盖。"""
        with tempfile.TemporaryDirectory() as tmp:
            logger_obj = DecisionLogger(log_dir=tmp)
            explainer = DecisionExplainer()
            for i in range(5):
                logger_obj.log(explainer.explain(np.zeros(14), action=0, step=i))
            loaded = logger_obj.load()
            self.assertEqual(len(loaded), 5)
            self.assertEqual(loaded[0].step, 0)
            self.assertEqual(loaded[4].step, 4)

    def test_load_preserves_q_values(self):
        """加载后 q_values 字段应保留。"""
        with tempfile.TemporaryDirectory() as tmp:
            logger_obj = DecisionLogger(log_dir=tmp)
            explainer = DecisionExplainer()
            rec = explainer.explain(
                np.arange(14, dtype=np.float64),
                action=1,
                q_values=np.array([1.0, 2.0, 3.0]),
                step=0,
            )
            logger_obj.log(rec)
            loaded = logger_obj.load()
            self.assertIsNotNone(loaded[0].q_values)
            np.testing.assert_array_almost_equal(loaded[0].q_values, np.array([1.0, 2.0, 3.0]))


# ============================================================
# TestEdgeCases 测试
# ============================================================
class TestEdgeCases(unittest.TestCase):
    """测试边界情况。"""

    def test_all_zero_state(self):
        """全零状态不应崩溃，贡献度应均匀分布。"""
        explainer = DecisionExplainer()
        rec = explainer.explain(np.zeros(14), action=0)
        total = sum(rec.feature_contributions.values())
        self.assertAlmostEqual(total, 1.0, places=6)
        # 全零状态退化为均匀分布
        for v in rec.feature_contributions.values():
            self.assertAlmostEqual(v, 1.0 / 14, places=6)

    def test_constant_nonzero_state(self):
        """常量非零状态应可处理且归一化。"""
        explainer = DecisionExplainer()
        rec = explainer.explain(np.full(14, 0.5), action=0)
        total = sum(rec.feature_contributions.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_single_record_session(self):
        """单条记录的会话汇总应正确。"""
        explainer = DecisionExplainer()
        rec = explainer.explain(np.arange(14, dtype=np.float64), action=1, step=0)
        summary = explainer.summarize_session([rec])
        self.assertEqual(summary["total_steps"], 1)
        self.assertEqual(summary["action_distribution"], {1: 1})

    def test_empty_records_importance(self):
        """空记录的特征重要性应为空字典。"""
        explainer = DecisionExplainer()
        self.assertEqual(explainer.get_feature_importance([]), {})

    def test_empty_records_anomalies(self):
        """空记录的异常检测应返回空列表。"""
        explainer = DecisionExplainer()
        self.assertEqual(explainer.detect_anomalies([]), [])

    def test_empty_records_summary(self):
        """空记录的会话汇总应返回零值。"""
        explainer = DecisionExplainer()
        summary = explainer.summarize_session([])
        self.assertEqual(summary["total_steps"], 0)
        self.assertEqual(summary["anomaly_count"], 0)

    def test_state_shorter_than_feature_names(self):
        """状态向量短于特征名列表时应补齐特征名。"""
        explainer = DecisionExplainer()  # 14 个特征名
        rec = explainer.explain(np.array([0.1, 0.2, 0.3]), action=0)
        # 应仅包含 3 个特征（前 3 个标准特征名）
        self.assertEqual(len(rec.feature_contributions), 3)
        self.assertIn("队列长度", rec.feature_contributions)

    def test_q_values_with_negative_advantage(self):
        """q_values 差分为负（选中动作低于均值）时也应正常计算。"""
        explainer = DecisionExplainer()
        state = np.arange(14, dtype=np.float64)
        # action=0 的 q 值最低
        rec = explainer.explain(state, action=0, q_values=np.array([1.0, 5.0, 3.0]))
        total = sum(rec.feature_contributions.values())
        self.assertAlmostEqual(total, 1.0, places=6)


# ============================================================
# TestCoverageFiller 补充覆盖测试
# 覆盖 explainability.py 中剩余未覆盖分支
# ============================================================
class TestCoverageFiller(unittest.TestCase):
    """补充覆盖 explainability.py 中剩余分支。"""

    def test_explain_empty_state(self):
        """空状态向量（n=0）应返回空贡献度（line 238）。"""
        explainer = DecisionExplainer(feature_names=["a", "b"])
        rec = explainer.explain(np.array([]), action=0, step=0)
        # n=0 时 contributions 为空数组
        self.assertEqual(len(rec.feature_contributions), 0)

    def test_feature_names_fewer_than_state(self):
        """特征名少于状态维度时应自动补齐（line 243）。"""
        explainer = DecisionExplainer(feature_names=["only_one"])
        rec = explainer.explain(np.array([1.0, 2.0, 3.0]), action=0, step=0)
        # 应补齐到 3 个特征名
        self.assertEqual(len(rec.feature_contributions), 3)
        self.assertIn("only_one", rec.feature_contributions)
        # 补齐的特征名格式为 "特征{i}"
        self.assertIn("特征1", rec.feature_contributions)
        self.assertIn("特征2", rec.feature_contributions)
        # 贡献度应归一化
        total = sum(rec.feature_contributions.values())
        self.assertAlmostEqual(total, 1.0, places=6)

    def test_format_explanation_english(self):
        """英文格式化应返回英文文本。"""
        explainer = DecisionExplainer(feature_names=["a", "b"])
        rec = explainer.explain(np.array([1.0, 2.0]), action=1, step=5)
        text_en = explainer.format_explanation(rec, top_k=2, lang="en")
        self.assertIn("Step 5", text_en)
        self.assertIn("action 1", text_en)
        self.assertIn("Key factors", text_en)

    def test_format_explanation_top_k_zero(self):
        """top_k=0 时应返回空因素列表。"""
        explainer = DecisionExplainer(feature_names=["a", "b"])
        rec = explainer.explain(np.array([1.0, 2.0]), action=0, step=0)
        text = explainer.format_explanation(rec, top_k=0)
        # 因素部分应为空
        self.assertIn("主要影响因素：", text)

    def test_state_value_by_name_not_found(self):
        """_state_value_by_name 找不到特征名时返回 0.0（lines 339-340）。"""
        explainer = DecisionExplainer(feature_names=["a", "b"])
        rec = explainer.explain(np.array([1.0, 2.0]), action=0, step=0)
        # 调用一个不存在于 feature_contributions 中的特征名
        val = explainer._state_value_by_name(rec, "nonexistent_feature")
        self.assertEqual(val, 0.0)

    def test_state_value_by_name_idx_out_of_range(self):
        """_state_value_by_name 索引超出状态范围时返回 0.0（line 344）。

        构造一个 feature_contributions 键数多于 state 元素的记录，
        使 idx >= len(state_arr) 触发越界返回。
        """
        rec = DecisionRecord(
            step=0,
            state=np.array([1.0]),  # 仅 1 个元素
            action=0,
            action_prob=1.0,
            q_values=None,
            feature_contributions={"a": 0.5, "b": 0.5},  # 2 个键
            timestamp="t",
        )
        explainer = DecisionExplainer(feature_names=["a", "b"])
        # "b" 的 idx=1 但 state 仅 1 个元素 → 越界 → 返回 0.0
        val = explainer._state_value_by_name(rec, "b")
        self.assertEqual(val, 0.0)
        # "a" 的 idx=0 在范围内 → 返回 state[0]
        val_a = explainer._state_value_by_name(rec, "a")
        self.assertAlmostEqual(val_a, 1.0)

    def test_load_skips_empty_lines(self):
        """load() 应跳过空行（line 499）。"""
        with tempfile.TemporaryDirectory() as tmp:
            logger_obj = DecisionLogger(log_dir=tmp)
            explainer = DecisionExplainer()
            rec = explainer.explain(np.zeros(14), action=0, step=0)
            logger_obj.log(rec)
            # 追加空行和空白行
            with open(logger_obj.log_path, "a", encoding="utf-8") as f:
                f.write("\n\n   \n")
            loaded = logger_obj.load()
            # 应只加载 1 条记录，跳过空行
            self.assertEqual(len(loaded), 1)

    def test_contribution_level_high(self):
        """_contribution_level 在 contrib >= 2*uniform 时返回 '高'。"""
        level = DecisionExplainer._contribution_level(0.6, 0.2)
        self.assertEqual(level, "高")

    def test_contribution_level_medium(self):
        """_contribution_level 在 uniform <= contrib < 2*uniform 时返回 '中'。"""
        level = DecisionExplainer._contribution_level(0.25, 0.2)
        self.assertEqual(level, "中")

    def test_contribution_level_low(self):
        """_contribution_level 在 contrib < uniform 时返回 '低'。"""
        level = DecisionExplainer._contribution_level(0.1, 0.2)
        self.assertEqual(level, "低")


if __name__ == "__main__":
    unittest.main()
