"""
调度决策可解释性追踪模块
Decision Explainability Tracking Module

为 RL 调度决策提供可解释性支持，将黑箱决策转化为可读的特征贡献度分析，
便于向评委解释"为什么选择了这个任务"。

核心能力：
- DecisionRecord    : 决策记录数据类（状态/动作/置信度/特征贡献度）
- DecisionExplainer : 决策解释器（贡献度计算、文本格式化、异常检测、会话汇总）
- DecisionLogger    : 决策日志记录器（JSONL 持久化，UTF-8 编码）

贡献度算法（简化方法，不依赖 shap/lime 等外部库）：
- 有 q_values 时：contribution[i] = |state[i] * advantage| 归一化
                  advantage = q_values[action] - mean(q_values)
                  （选中动作相对平均的优势越大，整体贡献度越集中）
- 无 q_values 时：contribution[i] = |z_score[i]| 归一化
                  z_score[i] = (state[i] - mean(state)) / std(state)
                  （状态偏离均值越远，对该决策的解释力越强）

使用示例：
    from src.scheduler.explainability import DecisionExplainer, DecisionLogger
    import numpy as np

    explainer = DecisionExplainer()
    record = explainer.explain(
        state=np.random.rand(14), action=1, q_values=np.array([1.0, 3.0, 2.0]),
        action_prob=0.85, step=10,
    )
    print(explainer.format_explanation(record, top_k=5))

    logger = DecisionLogger(log_dir="logs/decisions")
    logger.log(record)
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 状态空间 14 维特征名（与 env.py 的 14 维状态向量对应）
STATE_FEATURE_NAMES: list[str] = [
    "队列长度",
    "平均优先级",
    "最大等待时间",
    "量子比特利用率",
    "电路深度均值",
    "任务类型_量子占比",
    "任务类型_经典占比",
    "任务类型_混合占比",
    "优先级方差",
    "预计执行时间均值",
    "队列紧迫度",
    "资源碎片化",
    "历史完成率",
    "当前步数",
]

# 异常决策检测：低置信度阈值（action_prob 低于此值视为异常）
_LOW_CONFIDENCE_THRESHOLD: float = 0.3


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class DecisionRecord:
    """
    单步调度决策记录。

    记录 RL 智能体在某一步的决策上下文，包括状态、动作、置信度以及
    各特征对决策的贡献度，用于事后解释与审计。

    Attributes:
        step                 : 决策步序号
        state                : 决策时的状态向量（14维）
        action               : 选择的动作编号
        action_prob          : 动作概率/置信度（0-1）
        q_values             : 各动作的 Q 值（DQN 可用，PPO 可为 None）
        feature_contributions: 各特征对决策的贡献度（归一化，和为 1）
        timestamp            : 记录生成时间（ISO 格式字符串）
    """

    step: int
    state: np.ndarray
    action: int
    action_prob: float
    q_values: np.ndarray | None
    feature_contributions: dict[str, float]
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        """
        将记录序列化为可 JSON 化的字典。

        将 numpy 数组转换为列表，便于写入 JSONL 文件。

        Returns:
            包含全部字段的字典，所有值均为 JSON 可序列化类型
        """
        return {
            "step": int(self.step),
            "state": [float(x) for x in np.asarray(self.state).tolist()],
            "action": int(self.action),
            "action_prob": float(self.action_prob),
            "q_values": (
                [float(x) for x in np.asarray(self.q_values).tolist()]
                if self.q_values is not None
                else None
            ),
            "feature_contributions": dict(self.feature_contributions),
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DecisionRecord":
        """
        从字典反序列化为 DecisionRecord。

        与 to_dict 互逆，用于从 JSONL 日志加载历史记录。

        Args:
            data: 由 to_dict 生成的字典

        Returns:
            还原后的 DecisionRecord 实例
        """
        q_raw = data.get("q_values")
        fc_raw = data.get("feature_contributions", {})
        return cls(
            step=int(data["step"]),
            state=np.asarray(data["state"], dtype=np.float64),
            action=int(data["action"]),
            action_prob=float(data["action_prob"]),
            q_values=(np.asarray(q_raw, dtype=np.float64) if q_raw is not None else None),
            feature_contributions={str(k): float(v) for k, v in fc_raw.items()},
            timestamp=str(data["timestamp"]),
        )


# ---------------------------------------------------------------------------
# 决策解释器
# ---------------------------------------------------------------------------


class DecisionExplainer:
    """
    调度决策解释器。

    基于状态向量与（可选）Q 值，计算各特征对决策的贡献度，
    并提供文本格式化、特征重要性聚合、异常检测、会话汇总等能力。

    贡献度算法（简化方法）：
        - 有 q_values : contribution[i] = |state[i] * advantage| 归一化
                        advantage = q_values[action] - mean(q_values)
        - 无 q_values : contribution[i] = |z_score[i]| 归一化
                        z_score[i] = (state[i] - mean) / std

    Attributes:
        feature_names: 状态空间各维度的特征名列表
    """

    def __init__(self, feature_names: list[str] | None = None) -> None:
        """
        初始化决策解释器。

        Args:
            feature_names: 状态空间特征名列表，为 None 时使用默认 14 维特征名
        """
        self.feature_names: list[str] = (
            list(feature_names) if feature_names is not None else list(STATE_FEATURE_NAMES)
        )

    def explain(
        self,
        state: np.ndarray,
        action: int,
        q_values: np.ndarray | None = None,
        action_prob: float = 1.0,
        step: int = 0,
    ) -> DecisionRecord:
        """
        计算单步决策的特征贡献度并生成决策记录。

        Args:
            state       : 状态向量（长度应与 feature_names 一致）
            action      : 选择的动作编号
            q_values    : 各动作 Q 值（可选，提供则使用 q_values 差分计算权重）
            action_prob : 动作概率/置信度，默认 1.0
            step        : 决策步序号，默认 0

        Returns:
            DecisionRecord 包含状态、动作、贡献度等完整信息
        """
        state_arr = np.asarray(state, dtype=np.float64).flatten()
        n = len(state_arr)

        # 计算 Q 值数组（标准化存储）
        q_arr: np.ndarray | None = None
        if q_values is not None:
            q_arr = np.asarray(q_values, dtype=np.float64).flatten()

        # 计算原始贡献度
        if q_arr is not None and len(q_arr) > 0:
            # 有 q_values：用选中动作相对平均的优势作为权重
            advantage = float(q_arr[action] - q_arr.mean())
            weight = abs(advantage)
            raw = np.abs(state_arr) * weight
        else:
            # 无 q_values：用 z-score 近似（偏离均值越远贡献越大）
            mean = float(state_arr.mean()) if n > 0 else 0.0
            std = float(state_arr.std()) if n > 0 else 0.0
            if std > 1e-12:
                raw = np.abs((state_arr - mean) / std)
            else:
                # 状态无方差（如全零/常量），退化为绝对值
                raw = np.abs(state_arr)
                if float(raw.sum()) <= 1e-12:
                    # 全零状态：均匀分布
                    raw = np.ones(n, dtype=np.float64)

        # 归一化（和为 1）
        total = float(raw.sum())
        if total > 1e-12:
            contributions = raw / total
        elif n > 0:
            contributions = np.full(n, 1.0 / n, dtype=np.float64)
        else:
            contributions = np.zeros(0, dtype=np.float64)

        # 对齐特征名（长度不一致时补齐或截断）
        names = list(self.feature_names)
        if len(names) < n:
            names = names + [f"特征{i}" for i in range(len(names), n)]
        elif len(names) > n:
            names = names[:n]

        feature_contributions: dict[str, float] = {
            name: float(c) for name, c in zip(names, contributions, strict=True)
        }

        return DecisionRecord(
            step=int(step),
            state=state_arr,
            action=int(action),
            action_prob=float(action_prob),
            q_values=q_arr,
            feature_contributions=feature_contributions,
            timestamp=datetime.now().isoformat(timespec="seconds"),
        )

    def format_explanation(
        self,
        record: DecisionRecord,
        top_k: int = 5,
        lang: str = "zh",
    ) -> str:
        """
        将决策记录格式化为可读文本。

        Args:
            record : 决策记录
            top_k  : 显示前 k 个影响因素，默认 5
            lang   : 语言代码，"zh" 中文 / "en" 英文，默认 "zh"

        Returns:
            格式化文本，例如：
            "第N步选择动作A，主要影响因素：1.队列长度(高,值=0.850) 2.最大等待时间(中,值=0.620) ..."
        """
        # 按贡献度降序排序后取前 top_k
        sorted_items = sorted(
            record.feature_contributions.items(), key=lambda kv: kv[1], reverse=True
        )
        top_items = sorted_items[: max(0, top_k)]

        # 均匀分布参考值，用于判定贡献等级
        n = len(record.feature_contributions)
        uniform = (1.0 / n) if n > 0 else 0.0

        if lang == "en":
            parts: list[str] = []
            for idx, (name, contrib) in enumerate(top_items, start=1):
                level = self._contribution_level(contrib, uniform)
                state_value = self._state_value_by_name(record, name)
                parts.append(f"{idx}.{name}({level},val={state_value:.3f})")
            factors = " ".join(parts)
            return f"Step {record.step} chose action {record.action}. Key factors: {factors}"

        parts_zh: list[str] = []
        for idx, (name, contrib) in enumerate(top_items, start=1):
            level = self._contribution_level(contrib, uniform)
            state_value = self._state_value_by_name(record, name)
            parts_zh.append(f"{idx}.{name}({level},值={state_value:.3f})")
        factors = " ".join(parts_zh)
        return f"第{record.step}步选择动作{record.action}，主要影响因素：{factors}"

    @staticmethod
    def _contribution_level(contrib: float, uniform: float) -> str:
        """
        根据贡献度与均匀参考值判定等级。

        Args:
            contrib : 单特征贡献度
            uniform : 均匀分布时的贡献度参考值（1/n）

        Returns:
            等级字符串："高" / "中" / "低"
        """
        if contrib >= 2.0 * uniform:
            return "高"
        if contrib >= uniform:
            return "中"
        return "低"

    @staticmethod
    def _state_value_by_name(record: DecisionRecord, name: str) -> float:
        """
        根据特征名获取对应的状态值（按特征顺序对齐）。

        Args:
            record: 决策记录
            name  : 特征名

        Returns:
            该特征在状态向量中的取值，无法定位时返回 0.0
        """
        names = list(record.feature_contributions.keys())
        try:
            idx = names.index(name)
        except ValueError:
            return 0.0
        state_arr = np.asarray(record.state, dtype=np.float64).flatten()
        if 0 <= idx < len(state_arr):
            return float(state_arr[idx])
        return 0.0

    def get_feature_importance(self, records: list[DecisionRecord]) -> dict[str, float]:
        """
        从多条决策记录聚合特征重要性（均值）。

        Args:
            records: 决策记录列表

        Returns:
            特征名 -> 平均贡献度 的字典，空记录返回空字典
        """
        if not records:
            return {}

        accumulator: dict[str, float] = {}
        for record in records:
            for name, contrib in record.feature_contributions.items():
                accumulator[name] = accumulator.get(name, 0.0) + contrib

        count = len(records)
        return {name: total / count for name, total in accumulator.items()}

    def detect_anomalies(
        self,
        records: list[DecisionRecord],
        threshold: float = 2.0,
    ) -> list[int]:
        """
        检测异常决策。

        判定规则（满足其一即视为异常）：
            - action_prob < 0.3（低置信度决策）
            - 最大特征贡献度 > threshold * 平均贡献度（贡献过度集中）

        Args:
            records  : 决策记录列表
            threshold: 贡献集中度的倍数阈值，默认 2.0

        Returns:
            异常记录的索引列表（按出现顺序升序）
        """
        anomalies: list[int] = []
        for idx, record in enumerate(records):
            is_anomaly = False

            # 规则 1：低置信度
            if record.action_prob < _LOW_CONFIDENCE_THRESHOLD:
                is_anomaly = True

            # 规则 2：贡献度分布异常（过度集中在单一特征）
            if not is_anomaly and record.feature_contributions:
                contribs = list(record.feature_contributions.values())
                mean_c = sum(contribs) / len(contribs)
                max_c = max(contribs)
                if mean_c > 1e-12 and (max_c / mean_c) > threshold:
                    is_anomaly = True

            if is_anomaly:
                anomalies.append(idx)

        return anomalies

    def summarize_session(self, records: list[DecisionRecord]) -> dict[str, Any]:
        """
        汇总决策会话的统计信息。

        Args:
            records: 决策记录列表

        Returns:
            包含以下字段的字典：
                - total_steps         : 总步数
                - action_distribution : 动作分布 {动作编号: 出现次数}
                - top5_features       : 贡献度前 5 的特征及均值贡献度
                - anomaly_count       : 异常决策数
        """
        total_steps = len(records)

        # 动作分布
        action_dist: dict[int, int] = {}
        for record in records:
            action_dist[record.action] = action_dist.get(record.action, 0) + 1

        # 特征重要性（均值），按贡献度降序取前 5
        importance = self.get_feature_importance(records)
        sorted_imp = sorted(importance.items(), key=lambda kv: kv[1], reverse=True)
        top5 = [{"feature": name, "importance": val} for name, val in sorted_imp[:5]]

        # 异常决策数
        anomaly_count = len(self.detect_anomalies(records))

        return {
            "total_steps": total_steps,
            "action_distribution": action_dist,
            "top5_features": top5,
            "anomaly_count": anomaly_count,
        }


# ---------------------------------------------------------------------------
# 决策日志记录器
# ---------------------------------------------------------------------------


class DecisionLogger:
    """
    决策日志记录器（JSONL 持久化）。

    将 DecisionRecord 以 JSON Lines 格式追加写入日志文件，
    支持加载历史记录与清空日志。文件统一使用 UTF-8 编码以正确保存中文特征名。

    Attributes:
        log_dir : 日志目录
        log_path: 日志文件路径（log_dir/decisions.jsonl）
    """

    def __init__(self, log_dir: str = "logs/decisions") -> None:
        """
        初始化决策日志记录器。

        若日志目录不存在会自动创建。

        Args:
            log_dir: 日志目录路径，默认 "logs/decisions"
        """
        self.log_dir: str = log_dir
        self.log_path: str = os.path.join(log_dir, "decisions.jsonl")
        os.makedirs(log_dir, exist_ok=True)

    def log(self, record: DecisionRecord) -> None:
        """
        将一条决策记录追加写入 JSONL 日志文件。

        Args:
            record: 决策记录
        """
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")

    def load(self) -> list[DecisionRecord]:
        """
        加载日志文件中的所有决策记录。

        Returns:
            DecisionRecord 列表（按写入顺序）。文件不存在时返回空列表。
        """
        if not os.path.exists(self.log_path):
            return []

        records: list[DecisionRecord] = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                records.append(DecisionRecord.from_dict(data))
        return records

    def clear(self) -> None:
        """清空日志文件内容（保留文件本身）。"""
        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("")
