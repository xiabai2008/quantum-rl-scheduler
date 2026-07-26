"""
混合调度模式模块
Hybrid Scheduling Mode: Rule Engine + RL Fallback

规则引擎优先匹配明确场景（classical 任务、极紧急任务、量子资源充足的量子任务、
空队列），未命中时调用 RL 智能体（DQN/PPO）进行细粒度决策；RL 不可用或未训练
时回退到默认规则兜底，确保调度系统在 RL 模型未上线或推理失败时仍能稳定运行。

决策优先级：
    1. 规则引擎（确定性场景，confidence=1.0）
    2. RL 智能体（细粒度决策，confidence=0.8）
    3. 默认规则兜底（RL 不可用时，confidence=0.3）
    4. 默认动作（全部失败，confidence=0.0）

动作空间（与 env.py 一致）：
    - 0 : 分配到经典计算资源 (ACTION_CLASSICAL)
    - 1 : 分配到量子计算资源 (ACTION_QUANTUM)
    - 2 : 混合执行 (ACTION_HYBRID)
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "ACTION_CLASSICAL",
    "ACTION_HYBRID",
    "ACTION_QUANTUM",
    "HybridScheduler",
    "RuleEngine",
]

# ---------------------------------------------------------------------------
# 动作常量（本模块内定义，避免 import env 的私有常量）
# ---------------------------------------------------------------------------
ACTION_CLASSICAL = 0  # 分配到经典计算资源
ACTION_QUANTUM = 1  # 分配到量子计算资源
ACTION_HYBRID = 2  # 混合执行

# 决策置信度常量（简化模型）
_RULE_CONFIDENCE = 1.0  # 规则决策
_RL_CONFIDENCE = 0.8  # RL 决策
_FALLBACK_CONFIDENCE = 0.3  # 兜底决策
_DEFAULT_CONFIDENCE = 0.0  # 默认决策


# ---------------------------------------------------------------------------
# 默认规则条件函数
# ---------------------------------------------------------------------------


def _rule_classical_condition(task: Any, context: dict[str, Any]) -> bool:
    """rule1: classical 任务直接走经典资源。"""
    return getattr(task, "task_type", "") == "classical"


def _rule_urgent_condition(task: Any, context: dict[str, Any]) -> bool:
    """rule2: 极紧急任务（priority>=5 且 urgency>=0.9）走经典（更快）。"""
    priority = getattr(task, "priority", 0)
    urgency = getattr(task, "urgency", 0.0)
    return priority >= 5 and urgency >= 0.9


def _rule_quantum_sufficient_condition(task: Any, context: dict[str, Any]) -> bool:
    """rule3: quantum 任务且量子资源充足（qubit_count <= available_qubits）→ 走量子。"""
    if getattr(task, "task_type", "") != "quantum":
        return False
    available_qubits = context.get("available_qubits", 0)
    qubit_count = getattr(task, "qubit_count", 0)
    return bool(qubit_count <= available_qubits)


def _rule_empty_queue_condition(task: Any, context: dict[str, Any]) -> bool:
    """rule4: 队列为空时直接分配。"""
    return bool(context.get("queue_length", 0) == 0)


def _rule_empty_queue_action(task: Any, context: dict[str, Any]) -> int:
    """rule4 的动作：quantum 任务→量子，否则经典。"""
    if getattr(task, "task_type", "") == "quantum":
        return ACTION_QUANTUM
    return ACTION_CLASSICAL


# ---------------------------------------------------------------------------
# 内部规则数据结构
# ---------------------------------------------------------------------------


@dataclass
class _Rule:
    """
    内部规则存储结构。

    Attributes:
        name      : 规则名称（用于 list_rules 和调试）
        condition : 规则条件函数 (task, context) -> bool
        action    : 命中时返回的动作，可为 int 或 (task, context) -> int 的可调用对象
    """

    name: str
    condition: Callable[[Any, dict[str, Any]], bool]
    action: int | Callable[[Any, dict[str, Any]], int]


# ---------------------------------------------------------------------------
# RuleEngine 规则引擎
# ---------------------------------------------------------------------------


class RuleEngine:
    """
    规则引擎：按优先级顺序评估规则，返回首个命中规则的动作。

    内置 4 条默认规则：
        - rule1_classical          : classical 任务 → ACTION_CLASSICAL
        - rule2_urgent             : 极紧急任务 → ACTION_CLASSICAL
        - rule3_quantum_sufficient : quantum 任务 + 资源充足 → ACTION_QUANTUM
        - rule4_empty_queue        : 空队列 → 直接分配（quantum→量子，否则经典）

    可通过 add_rule 动态扩展自定义规则。

    Args:
        config: 规则配置字典（预留扩展，当前未使用）
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        """初始化规则引擎并加载默认规则。"""
        self._config: dict[str, Any] = config or {}
        self._rules: list[_Rule] = []
        self._load_default_rules()

    def _load_default_rules(self) -> None:
        """加载默认规则集（按优先级顺序）。"""
        self._rules.append(_Rule("rule1_classical", _rule_classical_condition, ACTION_CLASSICAL))
        self._rules.append(_Rule("rule2_urgent", _rule_urgent_condition, ACTION_CLASSICAL))
        self._rules.append(
            _Rule(
                "rule3_quantum_sufficient",
                _rule_quantum_sufficient_condition,
                ACTION_QUANTUM,
            )
        )
        self._rules.append(
            _Rule("rule4_empty_queue", _rule_empty_queue_condition, _rule_empty_queue_action)
        )

    def evaluate(self, task: Any, context: dict[str, Any]) -> int | None:
        """
        按规则优先级评估任务，返回首个命中规则的动作。

        Args:
            task    : 任务对象（需有 task_type/priority/urgency/qubit_count 等属性）
            context : 调度上下文，包含 available_qubits/queue_length/available_ratio 等

        Returns:
            命中规则的动作（int），无规则命中时返回 None
        """
        for rule in self._rules:
            if rule.condition(task, context):
                action = rule.action
                if isinstance(action, int):
                    return action
                return action(task, context)
        return None

    def add_rule(
        self,
        name: str,
        condition: Callable[[Any, dict[str, Any]], bool],
        action: int,
    ) -> None:
        """
        动态添加自定义规则（追加到规则列表末尾，优先级最低）。

        Args:
            name      : 规则名称
            condition : 规则条件函数 (task, context) -> bool
            action    : 命中时返回的动作
        """
        self._rules.append(_Rule(name, condition, action))

    def list_rules(self) -> list[str]:
        """
        列出所有规则名称（按优先级顺序）。

        Returns:
            规则名称列表
        """
        return [rule.name for rule in self._rules]


# ---------------------------------------------------------------------------
# HybridScheduler 混合调度器
# ---------------------------------------------------------------------------


class HybridScheduler:
    """
    混合调度器：规则引擎优先 + RL 兜底。

    决策流程：
        1. 规则引擎评估 → 命中则返回（source=rule, confidence=1.0）
        2. RL 智能体推理 → 成功且置信度达标则返回（source=rl, confidence=0.8）
        3. 默认规则兜底 → RL 不可用时返回（source=fallback, confidence=0.3）
        4. 默认动作 → 全部失败时返回 ACTION_HYBRID（source=default, confidence=0.0）

    Args:
        rl_agent             : RL 智能体（需有 predict(state, deterministic)->int 方法）
        rule_engine          : 规则引擎（默认创建 RuleEngine()）
        confidence_threshold : RL 置信度阈值，低于此值回退规则
        fallback_to_rule     : RL 不可用时是否回退到默认规则
    """

    def __init__(
        self,
        rl_agent: Any | None = None,
        rule_engine: RuleEngine | None = None,
        confidence_threshold: float = 0.6,
        fallback_to_rule: bool = True,
    ) -> None:
        """初始化混合调度器。"""
        self._rl_agent: Any | None = rl_agent
        self._rule_engine: RuleEngine = rule_engine if rule_engine is not None else RuleEngine()
        self._confidence_threshold: float = confidence_threshold
        self._fallback_to_rule: bool = fallback_to_rule

        # 决策统计
        self._rule_decisions: int = 0
        self._rl_decisions: int = 0
        self._fallback_decisions: int = 0
        self._default_decisions: int = 0

    def decide(
        self,
        task: Any,
        state: np.ndarray | None = None,
        context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        核心决策方法：规则优先 → RL → 兜底 → 默认。

        Args:
            task    : 待调度任务对象
            state   : RL 状态向量（规则未命中时传给 RL 智能体）
            context : 调度上下文（available_qubits/queue_length 等）

        Returns:
            决策结果字典：
                - action     : 动作（0/1/2）
                - source     : 决策来源（rule/rl/fallback/default）
                - confidence : 置信度（0.0-1.0）
                - reason     : 决策原因说明
        """
        ctx: dict[str, Any] = context if context is not None else {}

        # 1. 规则引擎优先
        action = self._rule_engine.evaluate(task, ctx)
        if action is not None:
            self._rule_decisions += 1
            return {
                "action": action,
                "source": "rule",
                "confidence": _RULE_CONFIDENCE,
                "reason": f"规则引擎命中，动作={action}",
            }

        # 2. RL 智能体决策（先校验置信度阈值，避免不必要的推理）
        if (
            self._rl_agent is not None
            and state is not None
            and self._confidence_threshold <= _RL_CONFIDENCE
        ):
            try:
                raw_action = self._rl_agent.predict(state, deterministic=True)
                resolved_action = int(raw_action)
                self._rl_decisions += 1
                return {
                    "action": resolved_action,
                    "source": "rl",
                    "confidence": _RL_CONFIDENCE,
                    "reason": "RL 智能体决策",
                }
            except RuntimeError:
                # RL 未训练，走兜底
                pass
            except Exception:
                # 其他异常也走兜底
                pass

        # 3. 默认规则兜底
        if self._fallback_to_rule:
            fb_action = self._fallback_rule(task, ctx)
            self._fallback_decisions += 1
            return {
                "action": fb_action,
                "source": "fallback",
                "confidence": _FALLBACK_CONFIDENCE,
                "reason": "RL 不可用，回退到默认规则",
            }

        # 4. 全部失败 → 默认动作
        self._default_decisions += 1
        return {
            "action": ACTION_HYBRID,
            "source": "default",
            "confidence": _DEFAULT_CONFIDENCE,
            "reason": "无可用决策路径，返回默认动作",
        }

    def _fallback_rule(self, task: Any, context: dict[str, Any]) -> int:
        """
        RL 不可用时的简单兜底规则：按任务类型分配。

        Args:
            task    : 任务对象
            context : 调度上下文

        Returns:
            兜底动作（quantum→ACTION_QUANTUM，classical→ACTION_CLASSICAL，其他→ACTION_HYBRID）
        """
        task_type = getattr(task, "task_type", "universal")
        if task_type == "quantum":
            return ACTION_QUANTUM
        if task_type == "classical":
            return ACTION_CLASSICAL
        return ACTION_HYBRID

    def decide_batch(
        self,
        tasks: list[Any],
        states: list[np.ndarray] | None = None,
        context: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """
        批量决策：对任务列表逐个调用 decide。

        Args:
            tasks   : 任务列表
            states  : 对应的状态向量列表（与 tasks 等长，可为 None）
            context : 共享的调度上下文

        Returns:
            决策结果字典列表（与 tasks 等长）
        """
        results: list[dict[str, Any]] = []
        for i, task in enumerate(tasks):
            state: np.ndarray | None = None
            if states is not None and i < len(states):
                state = states[i]
            result = self.decide(task, state=state, context=context)
            results.append(result)
        return results

    def get_stats(self) -> dict[str, int]:
        """
        返回决策统计。

        Returns:
            统计字典：
                - rule_decisions     : 规则决策次数
                - rl_decisions       : RL 决策次数
                - fallback_decisions : 兜底决策次数
                - total              : 总决策次数
        """
        return {
            "rule_decisions": self._rule_decisions,
            "rl_decisions": self._rl_decisions,
            "fallback_decisions": self._fallback_decisions,
            "total": (
                self._rule_decisions
                + self._rl_decisions
                + self._fallback_decisions
                + self._default_decisions
            ),
        }

    def reset_stats(self) -> None:
        """重置所有决策统计计数器。"""
        self._rule_decisions = 0
        self._rl_decisions = 0
        self._fallback_decisions = 0
        self._default_decisions = 0

    def set_confidence_threshold(self, threshold: float) -> None:
        """
        调整 RL 置信度阈值。

        Args:
            threshold: 新的阈值（0.0-1.0）
        """
        self._confidence_threshold = threshold
