"""
天衍云真机配额追踪与预警模块
Quota Tracker for Tianyan Cloud Real-Machine Usage

持久化追踪真机配额消耗（shots/tasks/wall_time_hours），支持：
- 多维度配额检查（剩余/使用比例/告警级别）
- 阈值告警（warning/critical，使用 loguru 日志）
- 每日消耗历史记录，用于估算配额耗尽时间
- 状态持久化（JSON 文件），重启后自动恢复
- 线程安全（threading.Lock）

注意：与 src/api/tianyan_client.py 中的 QuotaTracker（按窗口计数的轻量 API
配额追踪器）是不同概念；本模块面向真机配额，做持久化追踪与告警预警。

使用示例::

    tracker = QuotaTracker()
    if tracker.can_consume(shots=1024, tasks=1):
        tracker.consume(shots=1024, tasks=1)
    tracker.check_and_alert()
    summary = tracker.status()
"""

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

import yaml
from loguru import logger

from src.exceptions import ResourceExhaustedError

__all__ = ["QuotaExhaustedError", "QuotaTracker"]


# 默认配额（配置文件缺失时使用）
_DEFAULT_TOTAL_QUOTA: dict[str, float] = {
    "shots": 10000,
    "tasks": 200,
    "wall_time_hours": 50,
}
_DEFAULT_WARNING_THRESHOLD: float = 0.8
_DEFAULT_CRITICAL_THRESHOLD: float = 0.95

# 配额维度名称（用于遍历）
_QUOTA_DIMENSIONS: tuple[str, ...] = ("shots", "tasks", "wall_time_hours")


class QuotaExhaustedError(ResourceExhaustedError):
    """真机配额耗尽异常

    当 consume/can_consume 检测到任一维度配额超出上限时抛出。
    继承自 ResourceExhaustedError，便于上层统一资源异常处理。

    Attributes:
        dimension: 触发耗尽的维度名（shots/tasks/wall_time_hours）
        used: 已用量
        total: 总配额
    """

    def __init__(
        self,
        dimension: str,
        used: float,
        total: float,
        *,
        code: str = "QUOTA_EXHAUSTED",
        retryable: bool = False,
    ) -> None:
        """初始化配额耗尽异常。

        Args:
            dimension: 触发耗尽的维度名
            used: 已用量
            total: 总配额
            code: 错误码（关键字参数）
            retryable: 是否可重试（关键字参数）
        """
        self.dimension = dimension
        self.used = used
        self.total = total
        message = f"真机配额耗尽: {dimension}={used}/{total}"
        super().__init__(message, code=code, retryable=retryable)


class QuotaTracker:
    """真机配额追踪器

    从 config/quota.yaml 读取总配额配置，从 logs/quota_state.json 读取持久化状态，
    支持多维度配额检查、阈值告警、耗尽时间估算。

    线程安全：所有公开方法通过 threading.Lock 串行化，适合多线程调度循环调用。

    Args:
        config_path: 配额配置文件路径（默认 config/quota.yaml）
        state_path: 状态持久化文件路径（默认 logs/quota_state.json）
    """

    def __init__(
        self,
        config_path: str = "config/quota.yaml",
        state_path: str = "logs/quota_state.json",
    ) -> None:
        """初始化配额追踪器，加载配置与持久化状态。"""
        self._config_path = config_path
        self._state_path = state_path
        self._lock = threading.Lock()

        # 加载配置（缺失时使用默认值）
        self._total_quota: dict[str, float] = dict(_DEFAULT_TOTAL_QUOTA)
        self._warning_threshold: float = _DEFAULT_WARNING_THRESHOLD
        self._critical_threshold: float = _DEFAULT_CRITICAL_THRESHOLD
        self._notification: dict[str, Any] = {"type": "log", "webhook_url": None}
        self._load_config()

        # 加载持久化状态（文件不存在则初始化为 0）
        self._used: dict[str, float] = dict.fromkeys(_QUOTA_DIMENSIONS, 0.0)
        self._daily_history: list[dict[str, Any]] = []
        self._load_state()

        logger.info(
            f"[QuotaTracker] 初始化完成: total={self._total_quota}, used={self._used}, state={self._state_path}"
        )

    # ------------------------------------------------------------------
    # 配置与状态加载/持久化
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        """从 config_path 加载配额配置，文件缺失或格式错误时使用默认值。"""
        if not os.path.exists(self._config_path):
            logger.warning(
                f"[QuotaTracker] 配置文件不存在: {self._config_path}，使用默认配额"
            )
            return
        try:
            with open(self._config_path, encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}
        except (OSError, yaml.YAMLError) as e:
            logger.error(f"[QuotaTracker] 配置文件读取失败: {e}，使用默认配额")
            return

        total = cfg.get("total_quota", {})
        if isinstance(total, dict):
            for dim in _QUOTA_DIMENSIONS:
                val = total.get(dim)
                if isinstance(val, (int, float)) and val >= 0:
                    self._total_quota[dim] = float(val)

        wt = cfg.get("warning_threshold")
        if isinstance(wt, (int, float)) and 0 < wt < 1:
            self._warning_threshold = float(wt)
        ct = cfg.get("critical_threshold")
        if isinstance(ct, (int, float)) and 0 < ct <= 1:
            self._critical_threshold = float(ct)

        notification = cfg.get("notification")
        if isinstance(notification, dict):
            self._notification = {
                "type": notification.get("type", "log"),
                "webhook_url": notification.get("webhook_url"),
            }

    def _load_state(self) -> None:
        """从 state_path 加载持久化状态，文件不存在或损坏时初始化为 0。"""
        if not os.path.exists(self._state_path):
            logger.debug(f"[QuotaTracker] 状态文件不存在: {self._state_path}，初始化为 0")
            return
        try:
            with open(self._state_path, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"[QuotaTracker] 状态文件读取失败: {e}，初始化为 0")
            return

        used = state.get("used", {})
        if isinstance(used, dict):
            for dim in _QUOTA_DIMENSIONS:
                val = used.get(dim)
                if isinstance(val, (int, float)) and val >= 0:
                    self._used[dim] = float(val)

        history = state.get("daily_history", [])
        if isinstance(history, list):
            self._daily_history = [h for h in history if isinstance(h, dict)]

    def _persist_state(self) -> None:
        """将当前状态写入 state_path（需在锁内调用）。自动创建父目录。"""
        state = {
            "used": self._used,
            "daily_history": self._daily_history,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            parent = os.path.dirname(self._state_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(self._state_path, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
        except OSError as e:
            logger.error(f"[QuotaTracker] 状态持久化失败: {e}")

    # ------------------------------------------------------------------
    # 配额检查与消费
    # ------------------------------------------------------------------
    def can_consume(
        self,
        shots: int = 0,
        tasks: int = 1,
        wall_time_hours: float = 0.0,
    ) -> bool:
        """检查是否还能消费指定额度（不实际扣减）。

        Args:
            shots: 本次拟消费的 shots 数
            tasks: 本次拟消费的任务数
            wall_time_hours: 本次拟消费的墙上时间（小时）

        Returns:
            bool: 全部维度均在配额内返回 True，任一维度超额返回 False
        """
        request = {"shots": float(shots), "tasks": float(tasks), "wall_time_hours": float(wall_time_hours)}
        with self._lock:
            for dim in _QUOTA_DIMENSIONS:
                if self._used[dim] + request[dim] > self._total_quota[dim]:
                    return False
            return True

    def consume(
        self,
        shots: int = 0,
        tasks: int = 1,
        wall_time_hours: float = 0.0,
    ) -> bool:
        """消费配额，前置检查并持久化。

        Args:
            shots: 本次消费的 shots 数
            tasks: 本次消费的任务数
            wall_time_hours: 本次消费的墙上时间（小时）

        Returns:
            bool: 允许消费并已记录返回 True；任一维度超额返回 False（不抛异常）
        """
        request = {"shots": float(shots), "tasks": float(tasks), "wall_time_hours": float(wall_time_hours)}
        with self._lock:
            for dim in _QUOTA_DIMENSIONS:
                if self._used[dim] + request[dim] > self._total_quota[dim]:
                    logger.warning(
                        f"[QuotaTracker] 配额不足，拒绝消费: {dim} used={self._used[dim]}/{self._total_quota[dim]} request={request[dim]}"
                    )
                    return False
            for dim in _QUOTA_DIMENSIONS:
                self._used[dim] += request[dim]
            self._persist_state()
            logger.debug(
                f"[QuotaTracker] 消费成功: {request}，当前 used={self._used}"
            )
            return True

    def remaining(self) -> dict[str, float]:
        """返回各维度剩余配额。

        Returns:
            {shots, tasks, wall_time_hours} 剩余量（不会小于 0）
        """
        with self._lock:
            return {
                dim: max(0.0, self._total_quota[dim] - self._used[dim])
                for dim in _QUOTA_DIMENSIONS
            }

    def usage_ratio(self) -> dict[str, float]:
        """返回各维度使用比例（0-1）。

        Returns:
            {shots, tasks, wall_time_hours} 使用比例；总配额为 0 时返回 0.0
        """
        with self._lock:
            ratio: dict[str, float] = {}
            for dim in _QUOTA_DIMENSIONS:
                total = self._total_quota[dim]
                ratio[dim] = (self._used[dim] / total) if total > 0 else 0.0
            return ratio

    # ------------------------------------------------------------------
    # 状态摘要与告警
    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """返回完整状态摘要供 Web 面板展示。

        Returns:
            包含 total/used/remaining/usage_ratio/warning_level/
            estimated_exhaustion_time 的字典
        """
        with self._lock:
            total = dict(self._total_quota)
            used = dict(self._used)
            remaining = {
                dim: max(0.0, total[dim] - used[dim]) for dim in _QUOTA_DIMENSIONS
            }
            ratio = {
                dim: (used[dim] / total[dim]) if total[dim] > 0 else 0.0
                for dim in _QUOTA_DIMENSIONS
            }
            warning_level = self._compute_warning_level(ratio)
            est_exhaustion = self._estimate_exhaustion_time(ratio)

        return {
            "total": total,
            "used": used,
            "remaining": remaining,
            "usage_ratio": ratio,
            "warning_threshold": self._warning_threshold,
            "critical_threshold": self._critical_threshold,
            "warning_level": warning_level,
            "estimated_exhaustion_time": est_exhaustion,
            "daily_history_count": len(self._daily_history),
        }

    def _compute_warning_level(self, ratio: dict[str, float]) -> str:
        """根据各维度使用比例计算告警级别（需在锁内调用）。

        任一维度达到 critical 即为 critical；否则任一达到 warning 即为 warning；
        否则为 normal。

        Args:
            ratio: 各维度使用比例

        Returns:
            "normal" / "warning" / "critical"
        """
        max_ratio = max(ratio.values()) if ratio else 0.0
        if max_ratio >= self._critical_threshold:
            return "critical"
        if max_ratio >= self._warning_threshold:
            return "warning"
        return "normal"

    def _estimate_exhaustion_time(self, ratio: dict[str, float]) -> dict[str, Any] | None:
        """基于最近日均消耗估算各维度耗尽时间（需在锁内调用）。

        取最近 7 天（不足则取全部）的日均消耗，按当前剩余量估算还需多少天耗尽。
        无历史数据或日均消耗为 0 时对应维度返回 None。

        Args:
            ratio: 各维度使用比例

        Returns:
            {dim: {"days": float | None, "date": str | None}} 或 None（无任何历史数据）
        """
        if not self._daily_history:
            return None

        # 取最近 7 天历史
        recent = self._daily_history[-7:]
        days_count = len(recent)
        if days_count == 0:
            return None

        sums: dict[str, float] = dict.fromkeys(_QUOTA_DIMENSIONS, 0.0)
        for entry in recent:
            for dim in _QUOTA_DIMENSIONS:
                val = entry.get(dim)
                if isinstance(val, (int, float)):
                    sums[dim] += float(val)

        estimates: dict[str, Any] = {}
        now = datetime.now(timezone.utc)
        for dim in _QUOTA_DIMENSIONS:
            daily_avg = sums[dim] / days_count
            remaining = max(0.0, self._total_quota[dim] - self._used[dim])
            if daily_avg <= 0:
                estimates[dim] = {"days": None, "date": None}
            else:
                days_left = remaining / daily_avg
                # 估算耗尽日期（按天粒度）
                est_date = datetime.fromtimestamp(
                    now.timestamp() + days_left * 86400, tz=timezone.utc
                ).strftime("%Y-%m-%d")
                estimates[dim] = {"days": round(days_left, 2), "date": est_date}
        return estimates

    def check_and_alert(self) -> str | None:
        """检查阈值并发出告警（使用 loguru.logger）。

        Returns:
            告警级别字符串（"warning"/"critical"）或 None（未触发告警）
        """
        with self._lock:
            ratio = {
                dim: (self._used[dim] / self._total_quota[dim])
                if self._total_quota[dim] > 0
                else 0.0
                for dim in _QUOTA_DIMENSIONS
            }
            warning_level = self._compute_warning_level(ratio)
            est_exhaustion = self._estimate_exhaustion_time(ratio)

        if warning_level == "critical":
            logger.critical(
                f"[QuotaTracker] 配额危急! ratio={ratio}, 估算耗尽={est_exhaustion}"
            )
            self._dispatch_notification("critical", ratio, est_exhaustion)
            return "critical"
        if warning_level == "warning":
            logger.warning(
                f"[QuotaTracker] 配额警告: ratio={ratio}, 估算耗尽={est_exhaustion}"
            )
            self._dispatch_notification("warning", ratio, est_exhaustion)
            return "warning"
        return None

    def _dispatch_notification(
        self,
        level: str,
        ratio: dict[str, float],
        est_exhaustion: dict[str, Any] | None,
    ) -> None:
        """分发告警通知（log / webhook）。webhook 失败仅记录日志不抛异常。

        Args:
            level: 告警级别
            ratio: 各维度使用比例
            est_exhaustion: 估算耗尽时间
        """
        ntype = self._notification.get("type", "log")
        webhook_url = self._notification.get("webhook_url")
        if ntype == "webhook" and webhook_url:
            # 延迟导入 requests，避免未安装时影响 log 模式
            try:
                import requests

                payload = {
                    "level": level,
                    "ratio": ratio,
                    "estimated_exhaustion": est_exhaustion,
                }
                requests.post(webhook_url, json=payload, timeout=5)
            except Exception as e:
                # webhook 失败不阻塞主流程，仅记录日志
                logger.error(f"[QuotaTracker] webhook 通知失败: {e}")

    # ------------------------------------------------------------------
    # 每日历史记录
    # ------------------------------------------------------------------
    def record_daily_usage(self) -> None:
        """记录当日用量到历史（用于估算耗尽时间）。

        若当日已有记录则覆盖更新，否则追加新条目。
        """
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self._lock:
            snapshot = {dim: self._used[dim] for dim in _QUOTA_DIMENSIONS}
            # 覆盖当日记录
            updated = False
            for entry in self._daily_history:
                if entry.get("date") == today:
                    for dim in _QUOTA_DIMENSIONS:
                        entry[dim] = snapshot[dim]
                    updated = True
                    break
            if not updated:
                entry = {"date": today}
                entry.update(snapshot)
                self._daily_history.append(entry)
            # 仅保留最近 30 天历史，避免无限增长
            if len(self._daily_history) > 30:
                self._daily_history = self._daily_history[-30:]
            self._persist_state()
            logger.debug(f"[QuotaTracker] 已记录当日用量: {snapshot}")

    def get_daily_history(self) -> list[dict[str, Any]]:
        """返回每日消耗历史（按日期升序）。

        Returns:
            每日用量记录列表，每条含 date/shots/tasks/wall_time_hours
        """
        with self._lock:
            return list(self._daily_history)
