"""
多租户公平性调度指标模块
Multi-Tenant Fairness Metrics

提供公平性评估的核心指标：
- Jain Fairness Index: 经典公平性度量，值域 (0, 1]，1 表示完全公平
- MultiTenantFairnessTracker: 跟踪每个租户的完成率、等待时间等公平性指标

参考文献：
    R. Jain, "The Art of Computer Systems Performance Analysis", 1991
    FI = (Σx_i)² / (n × Σx_i²)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


# ============================================================================
# Jain Fairness Index
# ============================================================================

def jain_fairness_index(values: list[float]) -> float:
    """计算 Jain 公平性指数。

    FI = (Σx_i)² / (n × Σx_i²)

    返回值范围 (0, 1]：
    - 1.0 表示完全公平（所有值相等）
    - 接近 1/n 表示极端不公平
    - 0.0 仅在所有值均为 0 时返回

    Args:
        values: 需要评估公平性的数值列表（如各租户完成率）

    Returns:
        Jain Fairness Index (0, 1]

    Raises:
        ValueError: 输入列表为空时
    """
    if not values:
        raise ValueError("values 列表不能为空")

    n = len(values)
    sum_val = sum(values)
    sum_sq = sum(v * v for v in values)

    if sum_sq == 0.0:
        return 0.0

    fi = (sum_val * sum_val) / (n * sum_sq)
    return float(fi)


def max_min_fairness(values: list[float]) -> float:
    """计算最大-最小公平性比率。

    ratio = min(values) / max(values)
    值域 [0, 1]，1 表示完全公平。

    Args:
        values: 需要评估的数值列表

    Returns:
        最小/最大比率，max(values) == 0 时返回 0.0
    """
    if not values:
        return 0.0
    mx = max(values)
    if mx == 0.0:
        return 0.0
    return min(values) / mx


# ============================================================================
# 多租户公平性跟踪器
# ============================================================================

@dataclass
class TenantFairnessStats:
    """单个租户的公平性统计。

    Attributes:
        tenant_id        : 租户 ID
        tasks_submitted  : 提交的任务总数
        tasks_completed  : 完成的任务数
        tasks_failed     : 失败/超时的任务数
        total_wait_steps : 累计等待步数
        total_exec_steps : 累计执行步数
        completion_rate  : 完成率（tasks_completed / tasks_submitted）
        avg_wait_steps   : 平均等待步数
    """
    tenant_id: str = ""
    tasks_submitted: int = 0
    tasks_completed: int = 0
    tasks_failed: int = 0
    total_wait_steps: int = 0
    total_exec_steps: int = 0

    @property
    def completion_rate(self) -> float:
        """任务完成率。"""
        if self.tasks_submitted == 0:
            return 0.0
        return self.tasks_completed / self.tasks_submitted

    @property
    def avg_wait_steps(self) -> float:
        """平均等待步数。"""
        if self.tasks_submitted == 0:
            return 0.0
        return self.total_wait_steps / self.tasks_submitted

    @property
    def avg_exec_steps(self) -> float:
        """平均执行步数。"""
        if self.tasks_completed == 0:
            return 0.0
        return self.total_exec_steps / self.tasks_completed


class MultiTenantFairnessTracker:
    """多租户公平性跟踪器。

    在调度环境中跟踪每个租户的任务完成情况，
    提供 Jain Fairness Index、max-min ratio 等公平性指标。

    Args:
        tenant_ids: 需要跟踪的租户 ID 列表
    """

    def __init__(self, tenant_ids: list[str] | None = None) -> None:
        self._stats: dict[str, TenantFairnessStats] = {}
        if tenant_ids:
            for tid in tenant_ids:
                self._stats[tid] = TenantFairnessStats(tenant_id=tid)

    # ------------------------------------------------------------------
    # 事件记录
    # ------------------------------------------------------------------

    def record_submit(self, tenant_id: str | None, wait_steps: int = 0) -> None:
        """记录任务提交。

        Args:
            tenant_id: 租户 ID（None 时自动赋 "unknown"）
            wait_steps: 提交前已等待的步数
        """
        tid = tenant_id or "unknown"
        if tid not in self._stats:
            self._stats[tid] = TenantFairnessStats(tenant_id=tid)
        self._stats[tid].tasks_submitted += 1
        self._stats[tid].total_wait_steps += wait_steps

    def record_complete(self, tenant_id: str | None, exec_steps: int = 0) -> None:
        """记录任务完成。

        Args:
            tenant_id: 租户 ID
            exec_steps: 任务执行步数
        """
        tid = tenant_id or "unknown"
        if tid not in self._stats:
            self._stats[tid] = TenantFairnessStats(tenant_id=tid)
        self._stats[tid].tasks_completed += 1
        self._stats[tid].total_exec_steps += exec_steps

    def record_fail(self, tenant_id: str | None) -> None:
        """记录任务失败/超时。

        Args:
            tenant_id: 租户 ID
        """
        tid = tenant_id or "unknown"
        if tid not in self._stats:
            self._stats[tid] = TenantFairnessStats(tenant_id=tid)
        self._stats[tid].tasks_failed += 1

    # ------------------------------------------------------------------
    # 指标计算
    # ------------------------------------------------------------------

    def get_completion_rates(self) -> list[float]:
        """获取所有租户的完成率列表。"""
        return [s.completion_rate for s in self._stats.values()]

    def get_avg_wait_times(self) -> list[float]:
        """获取所有租户的平均等待步数列表。"""
        return [s.avg_wait_steps for s in self._stats.values()]

    def jain_completion_fairness(self) -> float:
        """以完成率为基础的 Jain 公平性指数。"""
        rates = self.get_completion_rates()
        if not rates:
            return 0.0
        return jain_fairness_index(rates)

    def jain_wait_fairness(self) -> float:
        """以平均等待时间为基（反转后）的 Jain 公平性指数。
        
        由于更公平意味着等待时间更均匀（而非数值更相等），
        我们反转等待时间：x_i = 1/(w_i + 1)，使得更低的等待获得更高的分值。
        """
        waits = self.get_avg_wait_times()
        if not waits:
            return 0.0
        # 反转：等待越短，得分越高
        inverted = [1.0 / (w + 1.0) for w in waits]
        return jain_fairness_index(inverted)

    def max_min_completion_ratio(self) -> float:
        """完成率的最大-最小比率。"""
        return max_min_fairness(self.get_completion_rates())

    # ------------------------------------------------------------------
    # 综合报告
    # ------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """生成多租户公平性综合报告。

        Returns:
            包含所有指标的字典
        """
        per_tenant = {}
        for tid, stats in self._stats.items():
            per_tenant[tid] = {
                "tenant_id": stats.tenant_id,
                "tasks_submitted": stats.tasks_submitted,
                "tasks_completed": stats.tasks_completed,
                "tasks_failed": stats.tasks_failed,
                "completion_rate": round(stats.completion_rate, 4),
                "avg_wait_steps": round(stats.avg_wait_steps, 2),
                "avg_exec_steps": round(stats.avg_exec_steps, 2),
            }

        return {
            "per_tenant": per_tenant,
            "num_tenants": len(self._stats),
            "total_tasks_submitted": sum(s.tasks_submitted for s in self._stats.values()),
            "total_tasks_completed": sum(s.tasks_completed for s in self._stats.values()),
            "jain_completion_fairness": round(self.jain_completion_fairness(), 4),
            "jain_wait_fairness": round(self.jain_wait_fairness(), 4),
            "max_min_completion_ratio": round(self.max_min_completion_ratio(), 4),
            "completion_rates": [round(r, 4) for r in self.get_completion_rates()],
        }

    def get_summary_table(self) -> str:
        """生成供报告使用的 Markdown 表格。"""
        lines = [
            "| 租户 | 提交 | 完成 | 失败 | 完成率 | 平均等待 | 平均执行 |",
            "|------|------|------|------|--------|----------|----------|",
        ]
        for stats in self._stats.values():
            lines.append(
                f"| {stats.tenant_id} "
                f"| {stats.tasks_submitted} "
                f"| {stats.tasks_completed} "
                f"| {stats.tasks_failed} "
                f"| {stats.completion_rate:.2%} "
                f"| {stats.avg_wait_steps:.1f} "
                f"| {stats.avg_exec_steps:.1f} |"
            )
        return "\n".join(lines)
