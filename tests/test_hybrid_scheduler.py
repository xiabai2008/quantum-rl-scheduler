"""
量子RL调度系统 - 混合调度模式单元测试
Unit Tests for src/scheduler/hybrid_scheduler.py

测试覆盖：
- RuleEngine 默认规则命中（classical/极紧急/量子资源充足/空队列）
- RuleEngine 无规则匹配返回 None
- RuleEngine 动态添加规则
- HybridScheduler 规则优先（规则命中时不调用 RL）
- HybridScheduler RL 兜底（规则未命中 + RL 可用 → source=rl）
- HybridScheduler RL 未训练回退（predict 抛 RuntimeError → source=fallback）
- HybridScheduler 无 RL 且规则未命中 → source=default
- decide_batch 批量决策
- get_stats / reset_stats 统计
- confidence 值验证
- set_confidence_threshold 调整
"""

import os
import sys
import types
import unittest
from unittest.mock import Mock

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.hybrid_scheduler import (
    ACTION_CLASSICAL,
    ACTION_HYBRID,
    ACTION_QUANTUM,
    HybridScheduler,
    RuleEngine,
)


def _make_task(
    task_type: str = "universal",
    qubit_count: int = 0,
    urgency: float = 0.5,
    priority: int = 3,
    task_id: str = "t1",
) -> types.SimpleNamespace:
    """构造测试用任务对象（模拟 env.Task 的字段结构）。"""
    return types.SimpleNamespace(
        task_id=task_id,
        task_type=task_type,
        qubit_count=qubit_count,
        wait_steps=0,
        urgency=urgency,
        priority=priority,
        execution_time=3,
    )


# ============================================================
# RuleEngine 测试
# ============================================================
class TestRuleEngine(unittest.TestCase):
    """测试规则引擎 RuleEngine。"""

    def setUp(self) -> None:
        """每个测试前创建全新的规则引擎。"""
        self.engine = RuleEngine()

    def test_default_rule_classical(self):
        """rule1: classical 任务 → ACTION_CLASSICAL。"""
        task = _make_task(task_type="classical")
        ctx = {"available_qubits": 100, "queue_length": 5}
        self.assertEqual(self.engine.evaluate(task, ctx), ACTION_CLASSICAL)

    def test_default_rule_urgent(self):
        """rule2: 极紧急任务（priority>=5 且 urgency>=0.9）→ ACTION_CLASSICAL。"""
        task = _make_task(task_type="universal", priority=5, urgency=0.95)
        ctx = {"available_qubits": 100, "queue_length": 5}
        self.assertEqual(self.engine.evaluate(task, ctx), ACTION_CLASSICAL)

    def test_default_rule_urgent_boundary_not_hit(self):
        """rule2 边界：priority=5 但 urgency<0.9 不命中。"""
        task = _make_task(task_type="universal", priority=5, urgency=0.89)
        ctx = {"available_qubits": 100, "queue_length": 5}
        # rule2 不命中；universal 不命中 rule1/rule3；queue_length!=0 不命中 rule4
        self.assertIsNone(self.engine.evaluate(task, ctx))

    def test_default_rule_quantum_sufficient(self):
        """rule3: quantum 任务 + 资源充足 → ACTION_QUANTUM。"""
        task = _make_task(task_type="quantum", qubit_count=50)
        ctx = {"available_qubits": 100, "queue_length": 5}
        self.assertEqual(self.engine.evaluate(task, ctx), ACTION_QUANTUM)

    def test_default_rule_quantum_insufficient(self):
        """rule3: quantum 任务但资源不足不命中。"""
        task = _make_task(task_type="quantum", qubit_count=200)
        ctx = {"available_qubits": 100, "queue_length": 5}
        # rule3 不命中（资源不足），其他规则也不命中
        self.assertIsNone(self.engine.evaluate(task, ctx))

    def test_default_rule_empty_queue_quantum(self):
        """rule4: 空队列 + quantum 任务 → ACTION_QUANTUM。"""
        # qubit_count=200 资源不足，rule3 不命中；queue_length=0 命中 rule4
        task = _make_task(task_type="quantum", qubit_count=200)
        ctx = {"available_qubits": 100, "queue_length": 0}
        self.assertEqual(self.engine.evaluate(task, ctx), ACTION_QUANTUM)

    def test_default_rule_empty_queue_non_quantum(self):
        """rule4: 空队列 + 非 quantum 任务 → ACTION_CLASSICAL。"""
        task = _make_task(task_type="universal")
        ctx = {"available_qubits": 100, "queue_length": 0}
        self.assertEqual(self.engine.evaluate(task, ctx), ACTION_CLASSICAL)

    def test_no_rule_match_returns_none(self):
        """无规则匹配时返回 None。"""
        task = _make_task(task_type="universal", priority=3, urgency=0.5)
        ctx = {"available_qubits": 100, "queue_length": 5}
        self.assertIsNone(self.engine.evaluate(task, ctx))

    def test_list_rules_returns_four_defaults(self):
        """list_rules 返回 4 条默认规则名。"""
        rules = self.engine.list_rules()
        self.assertEqual(len(rules), 4)
        self.assertIn("rule1_classical", rules)
        self.assertIn("rule2_urgent", rules)
        self.assertIn("rule3_quantum_sufficient", rules)
        self.assertIn("rule4_empty_queue", rules)

    def test_add_rule_dynamic(self):
        """动态添加规则后可命中。"""
        task = _make_task(task_type="universal", priority=3, urgency=0.5)
        ctx = {"available_qubits": 100, "queue_length": 5}
        # 默认规则不命中
        self.assertIsNone(self.engine.evaluate(task, ctx))

        # 添加自定义规则：universal → ACTION_HYBRID
        self.engine.add_rule(
            "custom_universal_hybrid",
            lambda t, c: getattr(t, "task_type", "") == "universal",
            ACTION_HYBRID,
        )
        self.assertEqual(self.engine.evaluate(task, ctx), ACTION_HYBRID)
        self.assertIn("custom_universal_hybrid", self.engine.list_rules())

    def test_rule_priority_first_match_wins(self):
        """规则按添加顺序评估，首个命中即返回。"""
        # classical + 极紧急 + 空队列 → rule1 优先命中
        task = _make_task(task_type="classical", priority=5, urgency=0.95)
        ctx = {"available_qubits": 100, "queue_length": 0}
        self.assertEqual(self.engine.evaluate(task, ctx), ACTION_CLASSICAL)

    def test_config_param_accepted(self):
        """config 参数被接受且不影响默认规则加载。"""
        engine = RuleEngine(config={"custom": "value"})
        self.assertEqual(len(engine.list_rules()), 4)


# ============================================================
# HybridScheduler 测试
# ============================================================
class TestHybridScheduler(unittest.TestCase):
    """测试混合调度器 HybridScheduler。"""

    def test_rule_priority_over_rl(self):
        """规则命中时不调用 RL。"""
        rl_agent = Mock()
        rl_agent.predict = Mock(return_value=ACTION_QUANTUM)
        scheduler = HybridScheduler(rl_agent=rl_agent)

        task = _make_task(task_type="classical")
        ctx = {"available_qubits": 100, "queue_length": 5}
        result = scheduler.decide(task, state=np.zeros(14), context=ctx)

        self.assertEqual(result["action"], ACTION_CLASSICAL)
        self.assertEqual(result["source"], "rule")
        self.assertEqual(result["confidence"], 1.0)
        rl_agent.predict.assert_not_called()

    def test_rl_fallback_when_rule_misses(self):
        """规则未命中 + RL 可用 → source=rl。"""
        rl_agent = Mock()
        rl_agent.predict = Mock(return_value=ACTION_QUANTUM)
        scheduler = HybridScheduler(rl_agent=rl_agent)

        task = _make_task(task_type="universal", priority=3, urgency=0.5)
        ctx = {"available_qubits": 100, "queue_length": 5}
        result = scheduler.decide(task, state=np.zeros(14), context=ctx)

        self.assertEqual(result["action"], ACTION_QUANTUM)
        self.assertEqual(result["source"], "rl")
        self.assertEqual(result["confidence"], 0.8)
        rl_agent.predict.assert_called_once()

    def test_rl_untrained_falls_back(self):
        """RL predict 抛 RuntimeError → source=fallback。"""
        rl_agent = Mock()
        rl_agent.predict = Mock(side_effect=RuntimeError("模型未训练"))
        scheduler = HybridScheduler(rl_agent=rl_agent)

        task = _make_task(task_type="universal", priority=3, urgency=0.5)
        ctx = {"available_qubits": 100, "queue_length": 5}
        result = scheduler.decide(task, state=np.zeros(14), context=ctx)

        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["confidence"], 0.3)
        # universal → fallback 返回 ACTION_HYBRID
        self.assertEqual(result["action"], ACTION_HYBRID)

    def test_rl_untrained_falls_back_quantum(self):
        """RL 未训练 + quantum 任务 → fallback 返回 ACTION_QUANTUM。"""
        rl_agent = Mock()
        rl_agent.predict = Mock(side_effect=RuntimeError("模型未训练"))
        scheduler = HybridScheduler(rl_agent=rl_agent)

        # quantum 任务但资源不足，rule3 不命中；非空队列，rule4 不命中
        task = _make_task(task_type="quantum", qubit_count=200, priority=3, urgency=0.5)
        ctx = {"available_qubits": 100, "queue_length": 5}
        result = scheduler.decide(task, state=np.zeros(14), context=ctx)

        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["action"], ACTION_QUANTUM)

    def test_no_rl_rule_misses_default(self):
        """无 RL 且规则未命中 + fallback_to_rule=False → source=default。"""
        scheduler = HybridScheduler(rl_agent=None, fallback_to_rule=False)

        task = _make_task(task_type="universal", priority=3, urgency=0.5)
        ctx = {"available_qubits": 100, "queue_length": 5}
        result = scheduler.decide(task, state=None, context=ctx)

        self.assertEqual(result["source"], "default")
        self.assertEqual(result["action"], ACTION_HYBRID)
        self.assertEqual(result["confidence"], 0.0)

    def test_no_rl_fallback_to_rule_enabled(self):
        """无 RL + fallback_to_rule=True → 走兜底规则。"""
        scheduler = HybridScheduler(rl_agent=None, fallback_to_rule=True)

        task = _make_task(task_type="universal", priority=3, urgency=0.5)
        ctx = {"available_qubits": 100, "queue_length": 5}
        result = scheduler.decide(task, state=None, context=ctx)

        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["action"], ACTION_HYBRID)
        self.assertEqual(result["confidence"], 0.3)

    def test_rl_present_but_state_none_falls_back(self):
        """RL 存在但 state=None → 走兜底。"""
        rl_agent = Mock()
        rl_agent.predict = Mock(return_value=ACTION_QUANTUM)
        scheduler = HybridScheduler(rl_agent=rl_agent)

        task = _make_task(task_type="universal", priority=3, urgency=0.5)
        ctx = {"available_qubits": 100, "queue_length": 5}
        result = scheduler.decide(task, state=None, context=ctx)

        self.assertEqual(result["source"], "fallback")
        rl_agent.predict.assert_not_called()

    def test_decide_batch(self):
        """批量决策返回与任务等长的结果列表。"""
        rl_agent = Mock()
        rl_agent.predict = Mock(return_value=ACTION_QUANTUM)
        scheduler = HybridScheduler(rl_agent=rl_agent)

        tasks = [
            _make_task(task_type="classical", task_id="t1"),  # rule → CLASSICAL
            _make_task(task_type="universal", task_id="t2"),  # RL → QUANTUM
            _make_task(task_type="quantum", qubit_count=50, task_id="t3"),  # rule → QUANTUM
        ]
        states = [np.zeros(14), np.zeros(14), np.zeros(14)]
        ctx = {"available_qubits": 100, "queue_length": 5}
        results = scheduler.decide_batch(tasks, states=states, context=ctx)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0]["action"], ACTION_CLASSICAL)
        self.assertEqual(results[0]["source"], "rule")
        self.assertEqual(results[1]["action"], ACTION_QUANTUM)
        self.assertEqual(results[1]["source"], "rl")
        self.assertEqual(results[2]["action"], ACTION_QUANTUM)
        self.assertEqual(results[2]["source"], "rule")

    def test_decide_batch_without_states(self):
        """批量决策无 states 时走规则或兜底。"""
        scheduler = HybridScheduler(rl_agent=None, fallback_to_rule=True)
        tasks = [
            _make_task(task_type="classical", task_id="t1"),
            _make_task(task_type="universal", task_id="t2"),
        ]
        ctx = {"available_qubits": 100, "queue_length": 5}
        results = scheduler.decide_batch(tasks, states=None, context=ctx)

        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["source"], "rule")
        self.assertEqual(results[1]["source"], "fallback")

    def test_get_stats(self):
        """get_stats 正确统计各来源决策次数。"""
        rl_agent = Mock()
        rl_agent.predict = Mock(return_value=ACTION_QUANTUM)
        scheduler = HybridScheduler(rl_agent=rl_agent)

        ctx = {"available_qubits": 100, "queue_length": 5}
        state = np.zeros(14)

        # 1 次规则决策
        scheduler.decide(_make_task(task_type="classical"), state=state, context=ctx)
        # 1 次 RL 决策
        scheduler.decide(
            _make_task(task_type="universal", priority=3, urgency=0.5),
            state=state,
            context=ctx,
        )

        stats = scheduler.get_stats()
        self.assertEqual(stats["rule_decisions"], 1)
        self.assertEqual(stats["rl_decisions"], 1)
        self.assertEqual(stats["fallback_decisions"], 0)
        self.assertEqual(stats["total"], 2)

    def test_reset_stats(self):
        """reset_stats 重置所有计数器。"""
        scheduler = HybridScheduler()
        ctx = {"available_qubits": 100, "queue_length": 5}
        scheduler.decide(_make_task(task_type="classical"), state=None, context=ctx)

        self.assertGreater(scheduler.get_stats()["total"], 0)

        scheduler.reset_stats()
        stats = scheduler.get_stats()
        self.assertEqual(stats["rule_decisions"], 0)
        self.assertEqual(stats["rl_decisions"], 0)
        self.assertEqual(stats["fallback_decisions"], 0)
        self.assertEqual(stats["total"], 0)

    def test_confidence_values(self):
        """各决策来源的 confidence 值正确。"""
        ctx = {"available_qubits": 100, "queue_length": 5}

        # 规则决策 = 1.0
        scheduler_rule = HybridScheduler()
        result_rule = scheduler_rule.decide(
            _make_task(task_type="classical"), state=None, context=ctx
        )
        self.assertEqual(result_rule["confidence"], 1.0)

        # RL 决策 = 0.8
        rl_agent = Mock()
        rl_agent.predict = Mock(return_value=ACTION_QUANTUM)
        scheduler_rl = HybridScheduler(rl_agent=rl_agent)
        result_rl = scheduler_rl.decide(
            _make_task(task_type="universal"), state=np.zeros(14), context=ctx
        )
        self.assertEqual(result_rl["confidence"], 0.8)

        # fallback = 0.3
        rl_agent_fail = Mock()
        rl_agent_fail.predict = Mock(side_effect=RuntimeError("未训练"))
        scheduler_fb = HybridScheduler(rl_agent=rl_agent_fail)
        result_fb = scheduler_fb.decide(
            _make_task(task_type="universal"), state=np.zeros(14), context=ctx
        )
        self.assertEqual(result_fb["confidence"], 0.3)

        # default = 0.0
        scheduler_def = HybridScheduler(rl_agent=None, fallback_to_rule=False)
        result_def = scheduler_def.decide(
            _make_task(task_type="universal"), state=None, context=ctx
        )
        self.assertEqual(result_def["confidence"], 0.0)

    def test_set_confidence_threshold_blocks_rl(self):
        """阈值高于 RL 置信度时 RL 不被采用，走 fallback。"""
        rl_agent = Mock()
        rl_agent.predict = Mock(return_value=ACTION_QUANTUM)
        # 阈值 0.9 > RL 置信度 0.8 → RL 不被采用
        scheduler = HybridScheduler(rl_agent=rl_agent, confidence_threshold=0.9)

        task = _make_task(task_type="universal", priority=3, urgency=0.5)
        ctx = {"available_qubits": 100, "queue_length": 5}
        result = scheduler.decide(task, state=np.zeros(14), context=ctx)

        self.assertEqual(result["source"], "fallback")
        rl_agent.predict.assert_not_called()

    def test_set_confidence_threshold_allows_rl(self):
        """调整阈值后 RL 被采用。"""
        rl_agent = Mock()
        rl_agent.predict = Mock(return_value=ACTION_QUANTUM)
        scheduler = HybridScheduler(rl_agent=rl_agent, confidence_threshold=0.9)

        task = _make_task(task_type="universal", priority=3, urgency=0.5)
        ctx = {"available_qubits": 100, "queue_length": 5}

        # 阈值 0.9 → fallback
        result1 = scheduler.decide(task, state=np.zeros(14), context=ctx)
        self.assertEqual(result1["source"], "fallback")

        # 调整阈值为 0.5 < 0.8 → RL 被采用
        scheduler.set_confidence_threshold(0.5)
        result2 = scheduler.decide(task, state=np.zeros(14), context=ctx)
        self.assertEqual(result2["source"], "rl")
        self.assertEqual(result2["confidence"], 0.8)

    def test_custom_rule_engine(self):
        """可传入自定义 RuleEngine。"""
        custom_engine = RuleEngine()
        custom_engine.add_rule(
            "custom_universal_hybrid",
            lambda t, c: getattr(t, "task_type", "") == "universal",
            ACTION_HYBRID,
        )
        scheduler = HybridScheduler(rule_engine=custom_engine)

        task = _make_task(task_type="universal")
        ctx = {"available_qubits": 100, "queue_length": 5}
        result = scheduler.decide(task, state=None, context=ctx)

        self.assertEqual(result["action"], ACTION_HYBRID)
        self.assertEqual(result["source"], "rule")

    def test_default_rule_engine_created(self):
        """不传 rule_engine 时自动创建默认引擎。"""
        scheduler = HybridScheduler()
        self.assertIsInstance(scheduler._rule_engine, RuleEngine)
        self.assertEqual(len(scheduler._rule_engine.list_rules()), 4)

    def test_decide_returns_required_keys(self):
        """decide 返回的字典包含所有必需字段。"""
        scheduler = HybridScheduler()
        ctx = {"available_qubits": 100, "queue_length": 5}
        result = scheduler.decide(_make_task(task_type="classical"), state=None, context=ctx)

        self.assertIn("action", result)
        self.assertIn("source", result)
        self.assertIn("confidence", result)
        self.assertIn("reason", result)


# ============================================================
# TestCoverageFiller 补充覆盖测试
# 覆盖 hybrid_scheduler.py 中剩余未覆盖分支
# ============================================================
class TestHybridCoverageFiller(unittest.TestCase):
    """补充覆盖 hybrid_scheduler.py 中剩余分支。"""

    def test_rl_generic_exception_falls_back(self):
        """RL predict 抛非 RuntimeError 异常时走兜底（lines 289-291）。"""
        rl_agent = Mock()
        rl_agent.predict = Mock(side_effect=ValueError("维度不匹配"))
        scheduler = HybridScheduler(rl_agent=rl_agent)

        task = _make_task(task_type="universal", urgency=0.3, qubit_count=5)
        ctx = {"available_qubits": 100, "queue_length": 5}
        state = np.zeros(14, dtype=np.float32)
        result = scheduler.decide(task, state=state, context=ctx)

        # 应走兜底（source=fallback），而非崩溃
        self.assertEqual(result["source"], "fallback")

    def test_rl_type_error_falls_back(self):
        """RL predict 抛 TypeError 时走兜底（lines 289-291）。"""
        rl_agent = Mock()
        rl_agent.predict = Mock(side_effect=TypeError("参数类型错误"))
        scheduler = HybridScheduler(rl_agent=rl_agent)

        task = _make_task(task_type="universal", urgency=0.3, qubit_count=5)
        ctx = {"available_qubits": 100, "queue_length": 5}
        state = np.zeros(14, dtype=np.float32)
        result = scheduler.decide(task, state=state, context=ctx)
        self.assertEqual(result["source"], "fallback")

    def test_fallback_rule_classical(self):
        """_fallback_rule 对 classical 任务返回 ACTION_CLASSICAL（line 328）。"""
        scheduler = HybridScheduler()
        task = _make_task(task_type="classical")
        action = scheduler._fallback_rule(task, {})
        self.assertEqual(action, ACTION_CLASSICAL)

    def test_fallback_rule_quantum(self):
        """_fallback_rule 对 quantum 任务返回 ACTION_QUANTUM。"""
        scheduler = HybridScheduler()
        task = _make_task(task_type="quantum")
        action = scheduler._fallback_rule(task, {})
        self.assertEqual(action, ACTION_QUANTUM)

    def test_fallback_rule_universal(self):
        """_fallback_rule 对 universal 任务返回 ACTION_HYBRID。"""
        scheduler = HybridScheduler()
        task = _make_task(task_type="universal")
        action = scheduler._fallback_rule(task, {})
        self.assertEqual(action, ACTION_HYBRID)

    def test_fallback_rule_unknown_type(self):
        """_fallback_rule 对未知任务类型返回 ACTION_HYBRID。"""
        scheduler = HybridScheduler()
        task = _make_task(task_type="unknown_type")
        action = scheduler._fallback_rule(task, {})
        self.assertEqual(action, ACTION_HYBRID)

    def test_fallback_triggered_for_classical_when_rl_fails(self):
        """RL 失败且规则未命中时，classical 任务走 fallback → ACTION_CLASSICAL。"""
        # 清空规则引擎的默认规则，确保规则不命中
        rule_engine = RuleEngine()
        rule_engine._rules = []

        rl_agent = Mock()
        rl_agent.predict = Mock(side_effect=RuntimeError("未训练"))
        scheduler = HybridScheduler(rl_agent=rl_agent, rule_engine=rule_engine)

        task = _make_task(task_type="classical", urgency=0.3, qubit_count=5)
        ctx = {"available_qubits": 100, "queue_length": 5}
        state = np.zeros(14, dtype=np.float32)
        result = scheduler.decide(task, state=state, context=ctx)
        self.assertEqual(result["source"], "fallback")
        self.assertEqual(result["action"], ACTION_CLASSICAL)


if __name__ == "__main__":
    unittest.main()
