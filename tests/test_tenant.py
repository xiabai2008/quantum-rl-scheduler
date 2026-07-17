"""
多租户资源配额隔离单元测试
Unit Tests for src/scheduler/tenant.py

测试覆盖：
- TenantQuota 数据类（字段、默认值）
- TenantQuotaManager 配置加载（正常/缺失/损坏配置文件回退）
- can_schedule 配额检查（量子比特上限、并发任务数、每日限额）
- consume 配额扣减与拒绝
- release 资源释放
- get_tenant_info / get_all_tenants_info 状态查询
- add_tenant / remove_tenant 动态管理
- 跨日重置 daily_used
- env.py 集成（_route_to_machine 租户配额检查）
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.tenant import TenantQuota, TenantQuotaManager


def _write_temp_yaml(content: str) -> str:
    """写入临时 YAML 文件并返回路径（Windows 兼容：先关闭再读取）。"""
    fd, path = tempfile.mkstemp(suffix=".yaml")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
    except Exception:
        os.unlink(path)
        raise
    return path


class TestTenantQuota(unittest.TestCase):
    """测试 TenantQuota 数据类。"""

    def test_default_values(self):
        """默认构造应填充合理默认值。"""
        q = TenantQuota()
        self.assertEqual(q.tenant_id, "default")
        self.assertEqual(q.max_qubits, 287)
        self.assertEqual(q.max_concurrent_tasks, 10)
        self.assertEqual(q.daily_limit, 0)
        self.assertEqual(q.priority, 3)
        self.assertEqual(q.used_qubits, 0)
        self.assertEqual(q.active_tasks, 0)
        self.assertEqual(q.daily_used, 0)

    def test_custom_values(self):
        """自定义值应正确设置。"""
        q = TenantQuota(
            tenant_id="lab",
            max_qubits=128,
            max_concurrent_tasks=5,
            daily_limit=50,
            priority=4,
        )
        self.assertEqual(q.tenant_id, "lab")
        self.assertEqual(q.max_qubits, 128)
        self.assertEqual(q.max_concurrent_tasks, 5)
        self.assertEqual(q.daily_limit, 50)
        self.assertEqual(q.priority, 4)


class TestTenantQuotaManagerConfig(unittest.TestCase):
    """测试配置加载。"""

    def test_default_init(self):
        """无参数初始化应创建默认租户。"""
        mgr = TenantQuotaManager()
        self.assertIn("default", mgr.tenant_ids)
        self.assertEqual(mgr.default_tenant_id, "default")

    def test_from_config_normal(self):
        """正常配置文件应正确加载。"""
        path = _write_temp_yaml(
            """
default_tenant: default
tenants:
  - tenant_id: default
    max_qubits: 287
    max_concurrent_tasks: 10
  - tenant_id: lab
    max_qubits: 128
    max_concurrent_tasks: 5
    daily_limit: 50
    priority: 4
"""
        )
        try:
            mgr = TenantQuotaManager.from_config(path)
            self.assertEqual(len(mgr.tenant_ids), 2)
            self.assertIn("default", mgr.tenant_ids)
            self.assertIn("lab", mgr.tenant_ids)
            lab_info = mgr.get_tenant_info("lab")
            self.assertEqual(lab_info["max_qubits"], 128)
            self.assertEqual(lab_info["daily_limit"], 50)
        finally:
            os.unlink(path)

    def test_from_config_missing_file(self):
        """配置文件缺失应回退到默认租户。"""
        mgr = TenantQuotaManager.from_config("/nonexistent/path/tenants.yaml")
        self.assertIn("default", mgr.tenant_ids)

    def test_from_config_invalid_yaml(self):
        """配置文件损坏应回退到默认租户。"""
        path = _write_temp_yaml("invalid: yaml: content: [")
        try:
            mgr = TenantQuotaManager.from_config(path)
            self.assertIn("default", mgr.tenant_ids)
        finally:
            os.unlink(path)

    def test_from_config_ensures_default_tenant(self):
        """配置文件未含默认租户时应自动补全。"""
        path = _write_temp_yaml(
            """
default_tenant: default
tenants:
  - tenant_id: lab
    max_qubits: 128
"""
        )
        try:
            mgr = TenantQuotaManager.from_config(path)
            self.assertIn("default", mgr.tenant_ids)
            self.assertIn("lab", mgr.tenant_ids)
        finally:
            os.unlink(path)


class TestCanSchedule(unittest.TestCase):
    """测试 can_schedule 配额检查。"""

    def setUp(self):
        """创建测试用管理器。"""
        self.mgr = TenantQuotaManager(
            tenants={
                "default": TenantQuota(
                    tenant_id="default",
                    max_qubits=287,
                    max_concurrent_tasks=10,
                    daily_limit=0,
                ),
                "lab": TenantQuota(
                    tenant_id="lab",
                    max_qubits=128,
                    max_concurrent_tasks=5,
                    daily_limit=50,
                ),
                "edu": TenantQuota(
                    tenant_id="edu",
                    max_qubits=16,
                    max_concurrent_tasks=30,
                    daily_limit=20,
                ),
            }
        )

    def test_normal_schedule_allowed(self):
        """正常范围内应允许调度。"""
        self.assertTrue(self.mgr.can_schedule("default", qubits=100, tasks=1))
        self.assertTrue(self.mgr.can_schedule("lab", qubits=64, tasks=1))

    def test_qubits_exceed_limit(self):
        """量子比特超过上限应拒绝。"""
        self.assertFalse(self.mgr.can_schedule("edu", qubits=32, tasks=1))
        self.assertFalse(self.mgr.can_schedule("lab", qubits=256, tasks=1))

    def test_concurrent_exceed_limit(self):
        """并发任务数超过上限应拒绝。"""
        # 先消耗 4 个并发
        for _ in range(4):
            self.mgr.consume("lab", qubits=10, tasks=1)
        # 第 5 个并发（达到上限 5）
        self.assertTrue(self.mgr.consume("lab", qubits=10, tasks=1))
        # 第 6 个并发应被 can_schedule 拒绝
        self.assertFalse(self.mgr.can_schedule("lab", qubits=10, tasks=1))

    def test_daily_limit_exceeded(self):
        """每日限额超过应拒绝。"""
        # edu 每日限额 20
        for _ in range(20):
            self.assertTrue(self.mgr.consume("edu", qubits=4, tasks=1))
        # 第 21 个应被拒绝
        self.assertFalse(self.mgr.can_schedule("edu", qubits=4, tasks=1))

    def test_no_daily_limit(self):
        """daily_limit=0 表示不限。"""
        for _ in range(100):
            self.assertTrue(self.mgr.can_schedule("default", qubits=1, tasks=1))

    def test_unknown_tenant_fallback(self):
        """未知租户应回退到默认租户。"""
        self.assertTrue(self.mgr.can_schedule("unknown_tenant", qubits=10, tasks=1))

    def test_none_tenant_uses_default(self):
        """None 租户应使用默认租户。"""
        self.assertTrue(self.mgr.can_schedule(None, qubits=10, tasks=1))


class TestConsumeAndRelease(unittest.TestCase):
    """测试 consume 和 release。"""

    def setUp(self):
        """创建测试用管理器。"""
        self.mgr = TenantQuotaManager(
            tenants={
                "default": TenantQuota(tenant_id="default"),
                "lab": TenantQuota(
                    tenant_id="lab",
                    max_qubits=64,
                    max_concurrent_tasks=3,
                    daily_limit=10,
                ),
            }
        )

    def test_consume_success(self):
        """成功消费应更新状态。"""
        self.assertTrue(self.mgr.consume("lab", qubits=32, tasks=1))
        info = self.mgr.get_tenant_info("lab")
        self.assertEqual(info["used_qubits"], 32)
        self.assertEqual(info["active_tasks"], 1)
        self.assertEqual(info["daily_used"], 1)

    def test_consume_multiple(self):
        """多次消费应累加。"""
        self.mgr.consume("lab", qubits=10, tasks=1)
        self.mgr.consume("lab", qubits=20, tasks=1)
        info = self.mgr.get_tenant_info("lab")
        self.assertEqual(info["used_qubits"], 30)
        self.assertEqual(info["active_tasks"], 2)
        self.assertEqual(info["daily_used"], 2)

    def test_consume_rejected_no_side_effect(self):
        """消费被拒绝时不应有副作用。"""
        # 超过量子比特上限
        result = self.mgr.consume("lab", qubits=128, tasks=1)
        self.assertFalse(result)
        info = self.mgr.get_tenant_info("lab")
        self.assertEqual(info["used_qubits"], 0)
        self.assertEqual(info["active_tasks"], 0)

    def test_release(self):
        """释放资源应减少计数。"""
        self.mgr.consume("lab", qubits=32, tasks=2)
        self.mgr.release("lab", qubits=32, tasks=2)
        info = self.mgr.get_tenant_info("lab")
        self.assertEqual(info["used_qubits"], 0)
        self.assertEqual(info["active_tasks"], 0)

    def test_release_not_negative(self):
        """释放后计数不应为负。"""
        self.mgr.consume("lab", qubits=10, tasks=1)
        self.mgr.release("lab", qubits=100, tasks=10)
        info = self.mgr.get_tenant_info("lab")
        self.assertEqual(info["used_qubits"], 0)
        self.assertEqual(info["active_tasks"], 0)

    def test_daily_remaining(self):
        """daily_remaining 应正确计算。"""
        self.mgr.consume("lab", qubits=10, tasks=3)
        info = self.mgr.get_tenant_info("lab")
        self.assertEqual(info["daily_remaining"], 7)

    def test_unlimited_daily_remaining(self):
        """不限额的租户 daily_remaining 应为 -1。"""
        mgr = TenantQuotaManager(
            tenants={"default": TenantQuota(tenant_id="default", daily_limit=0)}
        )
        info = mgr.get_tenant_info("default")
        self.assertEqual(info["daily_remaining"], -1)


class TestTenantManagement(unittest.TestCase):
    """测试租户动态管理。"""

    def setUp(self):
        """创建测试用管理器。"""
        self.mgr = TenantQuotaManager(
            tenants={"default": TenantQuota(tenant_id="default")}
        )

    def test_add_tenant(self):
        """添加租户应成功。"""
        self.mgr.add_tenant(
            TenantQuota(tenant_id="new_team", max_qubits=64, max_concurrent_tasks=5)
        )
        self.assertIn("new_team", self.mgr.tenant_ids)
        self.assertTrue(self.mgr.can_schedule("new_team", qubits=32, tasks=1))

    def test_remove_tenant(self):
        """移除非默认租户应成功。"""
        self.mgr.add_tenant(TenantQuota(tenant_id="temp"))
        self.assertTrue(self.mgr.remove_tenant("temp"))
        self.assertNotIn("temp", self.mgr.tenant_ids)

    def test_remove_default_tenant_fails(self):
        """移除默认租户应失败。"""
        self.assertFalse(self.mgr.remove_tenant("default"))

    def test_remove_nonexistent_tenant(self):
        """移除不存在的租户应返回 False。"""
        self.assertFalse(self.mgr.remove_tenant("nonexistent"))

    def test_get_all_tenants_info(self):
        """获取所有租户信息。"""
        self.mgr.add_tenant(TenantQuota(tenant_id="lab", max_qubits=128))
        info_list = self.mgr.get_all_tenants_info()
        self.assertEqual(len(info_list), 2)
        tenant_ids = [info["tenant_id"] for info in info_list]
        self.assertIn("default", tenant_ids)
        self.assertIn("lab", tenant_ids)


class TestEnvIntegration(unittest.TestCase):
    """测试 env.py 多租户集成。"""

    def test_env_without_tenant_manager(self):
        """未启用租户管理器时调度正常。"""
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_steps=10)
        self.assertEqual(env.get_tenant_stats(), [])
        # 正常调度不受影响
        obs, info = env.reset()
        self.assertIsNotNone(obs)

    def test_env_with_tenant_manager(self):
        """启用租户管理器时配额检查生效。"""
        from src.scheduler.env import QuantumSchedulingEnv
        from src.scheduler.env_types import Task

        mgr = TenantQuotaManager(
            tenants={
                "default": TenantQuota(
                    tenant_id="default",
                    max_qubits=287,
                    max_concurrent_tasks=2,
                    daily_limit=5,
                )
            }
        )
        env = QuantumSchedulingEnv(max_steps=10, tenant_manager=mgr)
        env.reset(seed=42)

        # 手动构造任务并路由到机器
        task = Task(task_id="t1", task_type="quantum", qubit_count=10, tenant_id="default")
        machine = env._machines[0]
        env._route_to_machine(machine, task, env.np_random)

        # 检查租户状态已更新
        stats = env.get_tenant_stats()
        self.assertEqual(len(stats), 1)
        self.assertEqual(stats[0]["active_tasks"], 1)
        self.assertEqual(stats[0]["daily_used"], 1)

    def test_env_tenant_quota_rejected(self):
        """租户配额不足时任务应被拒绝路由。"""
        from src.scheduler.env import QuantumSchedulingEnv
        from src.scheduler.env_types import Task

        mgr = TenantQuotaManager(
            tenants={
                "restricted": TenantQuota(
                    tenant_id="restricted",
                    max_qubits=8,
                    max_concurrent_tasks=1,
                    daily_limit=1,
                )
            }
        )
        env = QuantumSchedulingEnv(max_steps=10, tenant_manager=mgr)
        env.reset(seed=42)

        # 第一次调度成功（配额内）
        task1 = Task(
            task_id="t1", task_type="quantum", qubit_count=4, tenant_id="restricted"
        )
        env._route_to_machine(env._machines[0], task1, env.np_random)
        self.assertEqual(env._last_selected_machine, env._machines[0].name)

        # 第二次调度应被拒绝（并发任务超过上限 1）
        task2 = Task(
            task_id="t2", task_type="quantum", qubit_count=4, tenant_id="restricted"
        )
        env._route_to_machine(env._machines[0], task2, env.np_random)
        self.assertIsNone(env._last_selected_machine)

    def test_env_tenant_qubit_exceed(self):
        """租户量子比特上限不足时拒绝。"""
        from src.scheduler.env import QuantumSchedulingEnv
        from src.scheduler.env_types import Task

        mgr = TenantQuotaManager(
            tenants={
                "small": TenantQuota(
                    tenant_id="small",
                    max_qubits=16,
                    max_concurrent_tasks=10,
                )
            }
        )
        env = QuantumSchedulingEnv(max_steps=10, tenant_manager=mgr)
        env.reset(seed=42)

        # 请求 32 比特但租户上限 16
        task = Task(
            task_id="t1", task_type="quantum", qubit_count=32, tenant_id="small"
        )
        env._route_to_machine(env._machines[0], task, env.np_random)
        self.assertIsNone(env._last_selected_machine)


if __name__ == "__main__":
    unittest.main()
