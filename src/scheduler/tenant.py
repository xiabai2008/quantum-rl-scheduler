"""
多租户资源配额隔离模块
Multi-Tenant Resource Quota Isolation

为量子调度环境提供租户维度的资源配额管理：
- 每个租户有独立的量子比特上限、最大并发任务数、每日任务限额
- 调度时检查租户配额，超出时拒绝或排队
- 支持从 config/tenants.yaml 加载配额配置

使用示例::

    manager = TenantQuotaManager.from_config("config/tenants.yaml")
    if manager.check_and_consume("tenant_a", qubits=8, tasks=1):
        # 允许调度
        ...
    else:
        # 租户配额不足，排队或降级
        ...
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

import yaml
from loguru import logger


@dataclass
class TenantQuota:
    """单个租户的配额配置与运行时状态。

    Attributes:
        tenant_id            : 租户唯一标识
        max_qubits           : 单次任务最大量子比特数
        max_concurrent_tasks : 最大并发任务数
        daily_limit          : 每日任务提交上限（0 表示不限）
        priority             : 租户优先级（1-5，影响排队权重）
        used_qubits          : 当前占用的量子比特数（运行时）
        active_tasks         : 当前活跃任务数（运行时）
        daily_used           : 当日已提交任务数（运行时）
        daily_date           : 当日日期字符串（用于跨日重置）
    """

    tenant_id: str = "default"
    max_qubits: int = 287
    max_concurrent_tasks: int = 10
    daily_limit: int = 0
    priority: int = 3
    # 运行时状态
    used_qubits: int = 0
    active_tasks: int = 0
    daily_used: int = 0
    daily_date: str = field(default_factory=lambda: date.today().isoformat())


class TenantQuotaManager:
    """多租户配额管理器。

    管理多个租户的资源配额，在调度时进行配额检查与扣减。
    所有方法均为线程安全（使用内置 GIL 保证的简单操作，无需额外锁）。

    Args:
        tenants: 租户配额字典 {tenant_id: TenantQuota}
        default_tenant_id: 未声明租户的任务归属的默认租户
    """

    def __init__(
        self,
        tenants: dict[str, TenantQuota] | None = None,
        default_tenant_id: str = "default",
    ) -> None:
        """初始化租户配额管理器。

        Args:
            tenants          : 租户配额字典，None 时创建仅含默认租户的配置
            default_tenant_id: 未声明租户的任务归属的默认租户 ID
        """
        if tenants is None:
            tenants = {
                default_tenant_id: TenantQuota(
                    tenant_id=default_tenant_id,
                    max_qubits=287,
                    max_concurrent_tasks=10,
                )
            }
        self._tenants: dict[str, TenantQuota] = dict(tenants)
        self._default_tenant_id = default_tenant_id

    # ------------------------------------------------------------------
    # 配置加载
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config_path: str = "config/tenants.yaml") -> TenantQuotaManager:
        """从 YAML 配置文件加载租户配额。

        配置文件格式见 config/tenants.yaml。文件缺失或解析失败时
        回退到仅含默认租户的配置。

        Args:
            config_path: 配置文件路径

        Returns:
            TenantQuotaManager 实例
        """
        try:
            with open(config_path, encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            logger.warning(f"[Tenant] 配置文件 {config_path} 不存在，使用默认租户配置")
            return cls()
        except (yaml.YAMLError, OSError) as e:
            logger.warning(f"[Tenant] 配置文件解析失败: {e}，使用默认租户配置")
            return cls()

        if not isinstance(config, dict):
            logger.warning("[Tenant] 配置文件格式非字典，使用默认租户配置")
            return cls()

        default_id = config.get("default_tenant", "default")
        tenants: dict[str, TenantQuota] = {}
        tenant_list = config.get("tenants", [])

        for t_cfg in tenant_list:
            if not isinstance(t_cfg, dict):
                continue
            tid = str(t_cfg.get("tenant_id", "default"))
            tenants[tid] = TenantQuota(
                tenant_id=tid,
                max_qubits=int(t_cfg.get("max_qubits", 287)),
                max_concurrent_tasks=int(t_cfg.get("max_concurrent_tasks", 10)),
                daily_limit=int(t_cfg.get("daily_limit", 0)),
                priority=int(t_cfg.get("priority", 3)),
            )

        # 确保默认租户存在
        if default_id not in tenants:
            tenants[default_id] = TenantQuota(tenant_id=default_id)

        logger.info(f"[Tenant] 已加载 {len(tenants)} 个租户配置: {list(tenants.keys())}")
        return cls(tenants=tenants, default_tenant_id=default_id)

    # ------------------------------------------------------------------
    # 配额检查与扣减
    # ------------------------------------------------------------------

    def _get_tenant(self, tenant_id: str | None) -> TenantQuota:
        """获取租户配额对象，未找到时回退到默认租户。

        Args:
            tenant_id: 租户 ID，None 时使用默认租户

        Returns:
            对应的 TenantQuota 对象
        """
        tid = tenant_id or self._default_tenant_id
        if tid in self._tenants:
            return self._tenants[tid]
        # 回退到默认租户
        if self._default_tenant_id in self._tenants:
            return self._tenants[self._default_tenant_id]
        # 最后回退：返回任意一个租户
        return next(iter(self._tenants.values()))

    def _check_daily_reset(self, quota: TenantQuota) -> None:
        """检查是否跨日，跨日时重置每日计数。

        Args:
            quota: 租户配额对象
        """
        today = date.today().isoformat()
        if quota.daily_date != today:
            quota.daily_date = today
            quota.daily_used = 0

    def can_schedule(
        self,
        tenant_id: str | None,
        qubits: int = 0,
        tasks: int = 1,
    ) -> bool:
        """检查租户是否还有足够配额调度任务（不实际扣减）。

        Args:
            tenant_id: 租户 ID
            qubits   : 本次任务需要的量子比特数
            tasks    : 本次任务数（通常为 1）

        Returns:
            True 表示配额充足可以调度
        """
        quota = self._get_tenant(tenant_id)
        self._check_daily_reset(quota)

        # 检查量子比特上限
        if qubits > quota.max_qubits:
            logger.debug(
                f"[Tenant] {quota.tenant_id} 请求 {qubits} 比特超过上限 {quota.max_qubits}"
            )
            return False

        # 检查并发任务数
        if quota.active_tasks + tasks > quota.max_concurrent_tasks:
            logger.debug(
                f"[Tenant] {quota.tenant_id} 并发任务 {quota.active_tasks + tasks} "
                f"超过上限 {quota.max_concurrent_tasks}"
            )
            return False

        # 检查每日限额（0 表示不限）
        if quota.daily_limit > 0 and quota.daily_used + tasks > quota.daily_limit:
            logger.debug(
                f"[Tenant] {quota.tenant_id} 每日任务 {quota.daily_used + tasks} "
                f"超过限额 {quota.daily_limit}"
            )
            return False

        return True

    def consume(
        self,
        tenant_id: str | None,
        qubits: int = 0,
        tasks: int = 1,
    ) -> bool:
        """检查并扣减租户配额。

        Args:
            tenant_id: 租户 ID
            qubits   : 本次任务占用的量子比特数
            tasks    : 本次任务数

        Returns:
            True 表示配额扣减成功，False 表示配额不足
        """
        if not self.can_schedule(tenant_id, qubits, tasks):
            return False
        quota = self._get_tenant(tenant_id)
        quota.used_qubits += qubits
        quota.active_tasks += tasks
        quota.daily_used += tasks
        return True

    def release(
        self,
        tenant_id: str | None,
        qubits: int = 0,
        tasks: int = 1,
    ) -> None:
        """释放租户资源（任务完成后调用）。

        Args:
            tenant_id: 租户 ID
            qubits   : 释放的量子比特数
            tasks    : 释放的任务数
        """
        quota = self._get_tenant(tenant_id)
        quota.used_qubits = max(0, quota.used_qubits - qubits)
        quota.active_tasks = max(0, quota.active_tasks - tasks)

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------

    def get_tenant_info(self, tenant_id: str | None = None) -> dict[str, Any]:
        """获取租户配额状态信息。

        Args:
            tenant_id: 租户 ID，None 时返回默认租户

        Returns:
            租户状态字典
        """
        quota = self._get_tenant(tenant_id)
        self._check_daily_reset(quota)
        return {
            "tenant_id": quota.tenant_id,
            "max_qubits": quota.max_qubits,
            "max_concurrent_tasks": quota.max_concurrent_tasks,
            "daily_limit": quota.daily_limit,
            "priority": quota.priority,
            "used_qubits": quota.used_qubits,
            "active_tasks": quota.active_tasks,
            "daily_used": quota.daily_used,
            "daily_remaining": (
                max(0, quota.daily_limit - quota.daily_used) if quota.daily_limit > 0 else -1
            ),
        }

    def get_all_tenants_info(self) -> list[dict[str, Any]]:
        """获取所有租户的状态信息。

        Returns:
            租户状态列表
        """
        return [self.get_tenant_info(tid) for tid in self._tenants]

    @property
    def tenant_ids(self) -> list[str]:
        """所有租户 ID 列表。"""
        return list(self._tenants.keys())

    @property
    def default_tenant_id(self) -> str:
        """默认租户 ID。"""
        return self._default_tenant_id

    def add_tenant(self, quota: TenantQuota) -> None:
        """动态添加租户。

        Args:
            quota: 租户配额配置
        """
        self._tenants[quota.tenant_id] = quota
        logger.info(f"[Tenant] 已添加租户: {quota.tenant_id}")

    def remove_tenant(self, tenant_id: str) -> bool:
        """移除租户（默认租户不可移除）。

        Args:
            tenant_id: 要移除的租户 ID

        Returns:
            True 表示移除成功
        """
        if tenant_id == self._default_tenant_id:
            logger.warning(f"[Tenant] 默认租户 {tenant_id} 不可移除")
            return False
        if tenant_id in self._tenants:
            del self._tenants[tenant_id]
            logger.info(f"[Tenant] 已移除租户: {tenant_id}")
            return True
        return False
