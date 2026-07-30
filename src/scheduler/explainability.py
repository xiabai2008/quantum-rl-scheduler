"""
调度决策可解释性追踪模块
Decision Explainability Tracking Module

为 RL 调度决策提供可解释性支持，将黑箱决策转化为可读的特征贡献度分析，
便于向评委解释"为什么选择了这个任务"。

核心能力：
- DecisionRecord    : 决策记录数据类（状态/动作/置信度/特征贡献度）
- DecisionExplainer : 决策解释器（贡献度计算、文本格式化、异常检测、会话汇总）
- PPOExplainer      : PPO 模型专用 SHAP 解释器（Issue #596，保留正/负方向）
- DecisionLogger    : 决策日志记录器（JSONL 持久化，UTF-8 编码）

贡献度算法（两种模式）：
- heuristic（默认）：取绝对值，贡献度恒非负
    - 有 q_values 时：contribution[i] = |state[i] * advantage| 归一化
                      advantage = q_values[action] - mean(q_values)
    - 无 q_values 时：contribution[i] = |z_score[i]| 归一化
                      z_score[i] = (state[i] - mean(state)) / std(state)
- shap（Issue #596）：保留正/负方向，可区分特征对决策的推动/抑制
    - shap 库可用且提供 predict_fn/PPO 模型时：使用 SHAP Explainer 精确计算
    - 否则回退到方向感知启发式（不取绝对值），仍能区分正/负方向

使用示例：
    from src.scheduler.explainability import DecisionExplainer, DecisionLogger, PPOExplainer
    import numpy as np

    # 方式 1：通用决策解释器
    explainer = DecisionExplainer()
    record = explainer.explain(
        state=np.random.rand(17), action=1, q_values=np.array([1.0, 3.0, 2.0]),
        action_prob=0.85, step=10,
    )
    print(explainer.format_explanation(record, top_k=5))

    # 方式 2：PPO 模型专用 SHAP 解释器（Issue #596）
    # ppo_explainer = PPOExplainer(ppo_agent.model)
    # shap_values = ppo_explainer.explain(observation)
    # importance = ppo_explainer.get_feature_importance(observations_batch)

    logger = DecisionLogger(log_dir="logs/decisions")
    logger.log(record)
"""

import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# 常量定义
# ---------------------------------------------------------------------------

# 状态空间 16 维特征名（与 env_types.py 的 OBS_* 常量严格对应，OBS_DIM=16）
STATE_FEATURE_NAMES: list[str] = [
    "量子比特可用率",  # OBS_QUBIT_AVAILABILITY = 0
    "队列长度",  # OBS_QUEUE_LENGTH = 1
    "平均等待时间",  # OBS_AVG_WAIT_TIME = 2
    "量子比特保真度",  # OBS_FIDELITY = 3
    "经典资源负载",  # OBS_CLASSICAL_LOAD = 4
    "量子队列占比",  # OBS_QUANTUM_QUEUE_RATIO = 5
    "时间段",  # OBS_TIME_OF_DAY = 6
    "任务紧急程度",  # OBS_URGENCY_LEVEL = 7
    "量子任务标记",  # OBS_TASK_TYPE_QUANTUM = 8
    "经典任务标记",  # OBS_TASK_TYPE_CLASSICAL = 9
    "单比特门保真度",  # OBS_SINGLE_GATE_FIDELITY = 10
    "两比特门保真度",  # OBS_TWO_GATE_FIDELITY = 11
    "耦合图密度",  # OBS_COUPLING_DENSITY = 12
    "平均连通度",  # OBS_AVG_CONNECTIVITY = 13
    "串扰风险",  # OBS_CROSSTALK_RISK = 14
    "到达率MA",  # OBS_ARRIVAL_RATE_MA = 15
]

# Issue #588: 公平性指数特征名（第17维，仅在 include_fairness_obs=True 时使用）
FAIRNESS_FEATURE_NAME: str = "公平性指数"  # OBS_FAIRNESS_INDEX = 16

# 包含公平性观测的 17 维特征名列表
STATE_FEATURE_NAMES_WITH_FAIRNESS: list[str] = [*STATE_FEATURE_NAMES, FAIRNESS_FEATURE_NAME]

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
        state                : 决策时的状态向量（17维）
        action               : 选择的动作编号
        action_prob          : 动作概率/置信度（0-1）
        q_values             : 各动作的 Q 值（DQN 可用，PPO 可为 None）
        feature_contributions: 各特征对决策的贡献度（归一化，和为 1）
        timestamp            : 记录生成时间（ISO 格式字符串）
    """

    step: int
    state: NDArray[Any]
    action: int
    action_prob: float
    q_values: NDArray[Any] | None
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

    贡献度算法（两种模式）：
        - heuristic（默认）：取绝对值，贡献度恒非负
            - 有 q_values : contribution[i] = |state[i] * advantage| 归一化
            - 无 q_values : contribution[i] = |z_score[i]| 归一化
        - shap（Issue #596）：保留正/负方向
            - shap 库 + predict_fn 可用时：SHAP Explainer 精确计算
            - 否则回退到方向感知启发式（不取绝对值）

    Attributes:
        feature_names: 状态空间各维度的特征名列表
        method        : 贡献度计算方法，"heuristic" 或 "shap"
    """

    def __init__(
        self,
        feature_names: list[str] | None = None,
        method: str = "heuristic",
    ) -> None:
        """
        初始化决策解释器。

        Args:
            feature_names: 状态空间特征名列表，为 None 时使用默认 17 维特征名
            method        : 贡献度计算方法，"heuristic"（默认）或 "shap"
        """
        self.feature_names: list[str] = (
            list(feature_names) if feature_names is not None else list(STATE_FEATURE_NAMES)
        )
        self.method: str = method
        self._shap_available: bool = self._check_shap_available()

    @staticmethod
    def _check_shap_available() -> bool:
        """检查 shap 库是否已安装。"""
        try:
            import shap  # noqa: F401

            return True
        except ImportError:
            return False

    def explain(
        self,
        state: NDArray[Any],
        action: int,
        q_values: NDArray[Any] | None = None,
        action_prob: float = 1.0,
        step: int = 0,
        predict_fn: Any | None = None,
    ) -> DecisionRecord:
        """
        计算单步决策的特征贡献度并生成决策记录。

        Args:
            state       : 状态向量（长度应与 feature_names 一致）
            action      : 选择的动作编号
            q_values    : 各动作 Q 值（可选，提供则使用 q_values 差分计算权重）
            action_prob : 动作概率/置信度，默认 1.0
            step        : 决策步序号，默认 0
            predict_fn  : 预测函数（method="shap" 时使用），接受状态数组返回 Q 值/概率

        Returns:
            DecisionRecord 包含状态、动作、贡献度等完整信息
        """
        state_arr = np.asarray(state, dtype=np.float64).flatten()
        n = len(state_arr)

        # 计算 Q 值数组（标准化存储）
        q_arr: NDArray[Any] | None = None
        if q_values is not None:
            q_arr = np.asarray(q_values, dtype=np.float64).flatten()

        # 根据 method 选择贡献度计算方式
        if self.method == "shap":
            contributions = self._compute_shap_values(state_arr, action, q_arr, predict_fn)
        else:
            contributions = self._compute_heuristic_contributions(state_arr, action, q_arr)

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

    def _compute_heuristic_contributions(
        self,
        state_arr: NDArray[Any],
        action: int,
        q_arr: NDArray[Any] | None,
    ) -> NDArray[Any]:
        """
        启发式贡献度计算（取绝对值，贡献度恒非负）。

        Args:
            state_arr: 状态向量
            action   : 选择的动作
            q_arr    : Q 值数组（可选）

        Returns:
            归一化贡献度数组（非负，和为 1）
        """
        n = len(state_arr)

        if q_arr is not None and len(q_arr) > 0:
            advantage = float(q_arr[action] - q_arr.mean())
            weight = abs(advantage)
            raw = np.abs(state_arr) * weight
        else:
            mean = float(state_arr.mean()) if n > 0 else 0.0
            std = float(state_arr.std()) if n > 0 else 0.0
            if std > 1e-12:
                raw = np.abs((state_arr - mean) / std)
            else:
                raw = np.abs(state_arr)
                if float(raw.sum()) <= 1e-12:
                    raw = np.ones(n, dtype=np.float64)

        total = float(raw.sum())
        if total > 1e-12:
            return np.asarray(raw / total, dtype=np.float64)
        elif n > 0:
            return np.full(n, 1.0 / n, dtype=np.float64)
        else:
            return np.zeros(0, dtype=np.float64)

    def _compute_shap_values(
        self,
        state_arr: NDArray[Any],
        action: int,
        q_arr: NDArray[Any] | None,
        predict_fn: Any | None = None,
    ) -> NDArray[Any]:
        """
        SHAP 贡献度计算（保留正/负方向，Issue #596）。

        当 shap 库可用且提供 predict_fn 时，使用 SHAP Explainer 精确计算。
        否则回退到方向感知启发式（不取绝对值），仍能区分正/负方向。

        Args:
            state_arr : 状态向量
            action    : 选择的动作
            q_arr     : Q 值数组（可选）
            predict_fn: 预测函数，接受状态数组返回 Q 值/概率

        Returns:
            贡献度数组（可正可负），绝对值之和为 1
        """
        n = len(state_arr)

        # 尝试使用 SHAP 库精确计算
        if self._shap_available and predict_fn is not None:
            try:
                import shap

                background = np.zeros((1, n))
                explainer = shap.Explainer(predict_fn, background)
                shap_values = explainer(np.array([state_arr]))
                vals = np.asarray(shap_values.values)

                if vals.ndim == 3:
                    # 多输出：取选中动作的 SHAP 值
                    raw = vals[0, :, action]
                elif vals.ndim == 2:
                    raw = vals[0]
                else:
                    raw = np.zeros(n, dtype=np.float64)
            except Exception:
                # SHAP 计算失败时回退到方向感知启发式
                raw = self._directional_raw(state_arr, action, q_arr)
        else:
            # shap 库不可用或无 predict_fn：方向感知启发式
            raw = self._directional_raw(state_arr, action, q_arr)

        # 归一化：绝对值之和为 1，保留正/负方向
        abs_sum = float(np.abs(raw).sum())
        if abs_sum > 1e-12:
            return raw / abs_sum
        elif n > 0:
            return np.full(n, 1.0 / n, dtype=np.float64)
        else:
            return np.zeros(0, dtype=np.float64)

    @staticmethod
    def _directional_raw(
        state_arr: NDArray[Any],
        action: int,
        q_arr: NDArray[Any] | None,
    ) -> NDArray[Any]:
        """
        方向感知原始贡献度（不取绝对值，保留正/负方向）。

        Args:
            state_arr: 状态向量
            action   : 选择的动作
            q_arr    : Q 值数组（可选）

        Returns:
            原始贡献度数组（可正可负，未归一化）
        """
        n = len(state_arr)

        if q_arr is not None and len(q_arr) > 0:
            advantage = float(q_arr[action] - q_arr.mean())
            return state_arr * advantage
        else:
            mean = float(state_arr.mean()) if n > 0 else 0.0
            std = float(state_arr.std()) if n > 0 else 0.0
            if std > 1e-12:
                return (state_arr - mean) / std
            else:
                raw = state_arr.copy()
                if float(np.abs(raw).sum()) <= 1e-12:
                    return np.ones(n, dtype=np.float64)
                return raw

    def format_explanation(
        self,
        record: DecisionRecord,
        top_k: int = 5,
        lang: str = "zh",
    ) -> str:
        """
        将决策记录格式化为可读文本。

        SHAP 模式下始终标注正/负方向并按绝对值降序排序。
        heuristic 模式下贡献度恒非负，不标注方向，按值降序排序。

        Args:
            record : 决策记录
            top_k  : 显示前 k 个影响因素，默认 5
            lang   : 语言代码，"zh" 中文 / "en" 英文，默认 "zh"

        Returns:
            格式化文本，例如：
            "第N步选择动作A，主要影响因素：1.队列长度(高,正向,值=0.850) 2.最大等待时间(中,负向,值=0.620) ..."
        """
        # SHAP 模式始终显示方向标注并按绝对值排序
        show_direction = self.method == "shap"

        if show_direction:
            # SHAP 模式：按绝对值降序排序
            sorted_items = sorted(
                record.feature_contributions.items(),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
        else:
            # heuristic 模式：按值降序排序
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
                level = self._contribution_level(abs(contrib), uniform)
                state_value = self._state_value_by_name(record, name)
                if show_direction:
                    direction = "+" if contrib >= 0 else "-"
                    parts.append(f"{idx}.{name}({level},{direction},val={state_value:.3f})")
                else:
                    parts.append(f"{idx}.{name}({level},val={state_value:.3f})")
            factors = " ".join(parts)
            return f"Step {record.step} chose action {record.action}. Key factors: {factors}"

        parts_zh: list[str] = []
        for idx, (name, contrib) in enumerate(top_items, start=1):
            level = self._contribution_level(abs(contrib), uniform)
            state_value = self._state_value_by_name(record, name)
            if show_direction:
                direction = "正向" if contrib >= 0 else "负向"
                parts_zh.append(f"{idx}.{name}({level},{direction},值={state_value:.3f})")
            else:
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


# ---------------------------------------------------------------------------
# PPO 模型专用 SHAP 解释器（Issue #596）
# ---------------------------------------------------------------------------


class PPOExplainer:
    """
    PPO 模型专用 SHAP 可解释性解释器（Issue #596）。

    与 DecisionExplainer 的区别：
        - DecisionExplainer 是通用解释器，基于 state + q_values/action_prob 工作
        - PPOExplainer 直接封装训练好的 PPO 模型，使用 SHAP 库计算精确的特征贡献度
        - 保留 SHAP 值的正/负方向，可区分特征对决策的「推动」或「抑制」作用

    依赖说明：
        - shap 为可选依赖，未安装时自动回退到方向感知启发式方法
        - SHAP 相关导入均延迟到方法内部，避免强制依赖

    Attributes:
        model        : 训练好的 stable-baselines3 PPO/RecurrentPPO 模型
        feature_names: 状态空间特征名列表
        method       : 解释方法，"shap"（默认）或 "heuristic"
        n_features   : 状态空间维度
        n_actions    : 动作空间维度
        _shap_available: shap 库是否可用
        _explainer   : 懒加载的 SHAP Explainer 实例
    """

    def __init__(
        self,
        model: Any,
        feature_names: list[str] | None = None,
        method: str = "shap",
        background_samples: int = 100,
    ) -> None:
        """
        初始化 PPO SHAP 解释器。

        Args:
            model             : 训练好的 stable-baselines3 PPO/RecurrentPPO 模型
            feature_names     : 状态特征名列表，None 时使用默认 17 维特征名
            method            : 解释方法，"shap"（默认）或 "heuristic"
            background_samples: KernelExplainer 背景样本数，默认 100
        """
        self.model = model
        self.feature_names: list[str] = (
            list(feature_names) if feature_names is not None else list(STATE_FEATURE_NAMES)
        )
        self.method: str = method
        self.background_samples: int = background_samples
        self._shap_available: bool = self._check_shap_available()
        self._explainer: Any = None
        self._background_data: NDArray[Any] | None = None

        obs_space = getattr(model, "observation_space", None)
        action_space = getattr(model, "action_space", None)
        self.n_features: int = (
            int(obs_space.shape[0]) if obs_space is not None and obs_space.shape else 17
        )
        self.n_actions: int = int(action_space.n) if action_space is not None else 4

        if len(self.feature_names) < self.n_features:
            self.feature_names = self.feature_names + [
                f"特征{i}" for i in range(len(self.feature_names), self.n_features)
            ]
        elif len(self.feature_names) > self.n_features:
            self.feature_names = self.feature_names[: self.n_features]

    @staticmethod
    def _check_shap_available() -> bool:
        """检查 shap 库是否已安装（不强制依赖）。"""
        try:
            import shap  # noqa: F401

            return True
        except ImportError:
            return False

    def _get_shap_import_error_msg(self) -> str:
        """返回 shap 库未安装时的友好错误提示。"""
        return (
            "SHAP 解释需要 shap 库支持，但当前环境未安装 shap。\n"
            "请使用以下命令安装可选依赖：\n"
            "  pip install shap\n"
            "或使用 method='heuristic' 回退到方向感知启发式方法。"
        )

    def _predict_proba(self, observations: NDArray[Any]) -> NDArray[Any]:
        """
        模型预测函数：返回各动作的概率分布（SHAP 解释用）。

        将 stable-baselines3 PPO 模型包装为 SHAP 所需的 f(X) -> y 接口，
        输入批量状态，输出批量动作概率矩阵。

        Args:
            observations: 批量状态数组，形状 (batch_size, n_features)

        Returns:
            动作概率数组，形状 (batch_size, n_actions)
        """
        import torch  # stable-baselines3 依赖 torch

        obs_t = torch.as_tensor(observations, dtype=torch.float32)
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            probs = dist.distribution.probs.cpu().numpy()
        return np.asarray(probs, dtype=np.float64)

    def _init_shap_explainer(self) -> Any:
        """
        懒加载初始化 SHAP KernelExplainer。

        使用零向量作为背景数据（可配置 background_samples 数量），
        KernelExplainer 适用于任何黑盒模型，兼容性最好。

        Returns:
            初始化好的 shap.KernelExplainer 实例

        Raises:
            ImportError: shap 库未安装
        """
        if self._explainer is not None:
            return self._explainer

        if not self._shap_available:
            raise ImportError(self._get_shap_import_error_msg())

        import shap

        rng = np.random.default_rng(seed=42)
        self._background_data = rng.normal(
            loc=0.0, scale=0.1, size=(self.background_samples, self.n_features)
        )
        self._explainer = shap.KernelExplainer(
            model=self._predict_proba,
            data=self._background_data,
        )
        return self._explainer

    def explain(
        self,
        observation: NDArray[Any],
        action: int | None = None,
    ) -> dict[str, float]:
        """
        解释单个观察的决策，返回各特征的 SHAP 值（含正负方向）。

        Args:
            observation: 状态向量（1D 数组，长度 n_features）
            action     : 指定解释的动作索引，None 时自动选择模型预测的动作

        Returns:
            特征名 -> SHAP 值 的字典，值可正可负：
            - 正值：该特征推动模型选择该动作
            - 负值：该特征抑制模型选择该动作

        Raises:
            RuntimeError: method='shap' 但 shap 库不可用（已回退到启发式时不抛出）
        """
        obs_arr = np.asarray(observation, dtype=np.float64).flatten()
        if len(obs_arr) != self.n_features:
            obs_arr = np.resize(obs_arr, self.n_features)

        if self.method == "shap" and self._shap_available:
            return self._explain_shap(obs_arr, action)
        else:
            if self.method == "shap" and not self._shap_available:
                import warnings

                warnings.warn(
                    self._get_shap_import_error_msg() + "\n已自动回退到 heuristic 方法。",
                    stacklevel=2,
                )
            return self._explain_heuristic(obs_arr, action)

    def _explain_shap(self, obs_arr: NDArray[Any], action: int | None) -> dict[str, float]:
        """使用 SHAP KernelExplainer 计算特征贡献度。"""
        raw: NDArray[Any]
        try:
            explainer = self._init_shap_explainer()
            shap_values = explainer.shap_values(obs_arr.reshape(1, -1), nsamples=100)

            if action is None:
                probs = self._predict_proba(obs_arr.reshape(1, -1))
                action = int(np.argmax(probs[0]))

            action_idx = int(action) % self.n_actions

            if isinstance(shap_values, list):
                raw = np.asarray(shap_values[action_idx], dtype=np.float64).flatten()
            elif isinstance(shap_values, np.ndarray):
                sv = np.asarray(shap_values, dtype=np.float64)
                if sv.ndim == 3:
                    raw = np.asarray(sv[0, :, action_idx], dtype=np.float64)
                elif sv.ndim == 2:
                    raw = np.asarray(sv[0], dtype=np.float64)
                else:
                    raw = np.asarray(sv.flatten(), dtype=np.float64)
            else:
                raw = np.zeros(self.n_features, dtype=np.float64)
        except Exception:
            raw = self._directional_raw_values(obs_arr, action)

        return {name: float(val) for name, val in zip(self.feature_names, raw, strict=True)}

    def _explain_heuristic(self, obs_arr: NDArray[Any], action: int | None) -> dict[str, float]:
        """方向感知启发式方法（回退方案，保留正/负方向）。"""
        raw = self._directional_raw_values(obs_arr, action)
        return {name: float(val) for name, val in zip(self.feature_names, raw, strict=True)}

    def _directional_raw_values(self, obs_arr: NDArray[Any], action: int | None) -> NDArray[Any]:
        """
        方向感知原始贡献度计算（不取绝对值，保留正/负方向）。

        当 shap 不可用时作为回退方案，基于 z-score 符号保留方向信息。

        Args:
            obs_arr: 状态向量
            action : 动作索引（启发式下主要用于决定advantage符号）

        Returns:
            原始贡献度数组（可正可负，未归一化）
        """
        n = len(obs_arr)
        if n == 0:
            return np.zeros(0, dtype=np.float64)

        mean = float(obs_arr.mean())
        std = float(obs_arr.std())
        if std > 1e-12:
            z_scores = (obs_arr - mean) / std
        else:
            z_scores = obs_arr - mean
            if float(np.abs(z_scores).sum()) <= 1e-12:
                z_scores = np.ones(n, dtype=np.float64)

        if action is not None and self.model is not None:
            try:
                probs = self._predict_proba(obs_arr.reshape(1, -1))
                advantage = float(probs[0, action] - probs.mean())
                return z_scores * advantage
            except Exception:
                pass

        return z_scores

    def get_feature_importance(
        self,
        observations_batch: NDArray[Any] | list[NDArray[Any]],
    ) -> dict[str, float]:
        """
        批量计算特征重要性：平均 |SHAP| 值。

        对一批状态分别计算 SHAP 值，然后取各特征绝对 SHAP 值的均值
        作为全局特征重要性指标。

        Args:
            observations_batch: 批量状态数组，形状 (N, n_features) 或状态列表

        Returns:
            特征名 -> 平均|SHAP|值 的字典（值均为非负），
            按重要性降序排列
        """
        batch = np.asarray(observations_batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch.reshape(1, -1)

        n_samples = batch.shape[0]
        accumulator = np.zeros(self.n_features, dtype=np.float64)

        for i in range(n_samples):
            obs = batch[i]
            shap_dict = self.explain(obs)
            for j, name in enumerate(self.feature_names):
                accumulator[j] += abs(shap_dict[name])

        mean_abs = accumulator / n_samples if n_samples > 0 else accumulator

        importance_dict = {
            name: float(val) for name, val in zip(self.feature_names, mean_abs, strict=True)
        }
        sorted_items = sorted(importance_dict.items(), key=lambda kv: kv[1], reverse=True)
        return dict(sorted_items)
