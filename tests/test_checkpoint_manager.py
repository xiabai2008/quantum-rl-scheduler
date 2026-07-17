"""
量子RL调度系统 - 检查点版本管理单元测试
Unit Tests for src/scheduler/checkpoint_manager.py

Issue #83：测试 CheckpointMeta 数据类与 CheckpointManager 管理器的全部方法，
覆盖注册/列出/排序/最优/对比/删除/标签/持久化/清理/边界情况。

测试组织：
- TestCheckpointMeta      数据类创建与序列化
- TestRegister            注册检查点、自动版本号、元数据文件创建
- TestListCheckpoints     列出与排序（created_at/mean_reward/timesteps）
- TestGetBest             最优检查点（mean_reward/timesteps）
- TestCompare             版本对比与 improvement_pct 计算
- TestDelete              删除检查点（文件 + 元数据）
- TestTagUntag            添加/移除标签
- TestLoadSaveMeta        元数据持久化与往返一致性
- TestCleanupOrphans      清理孤立条目
- TestEdgeCases           空管理器、不存在的版本、重复版本等边界
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.checkpoint_manager import CheckpointManager, CheckpointMeta


# ============================================================
# 辅助函数
# ============================================================
def make_manager(tmp: str) -> CheckpointManager:
    """在临时目录下创建 CheckpointManager。"""
    checkpoint_dir = os.path.join(tmp, "models")
    meta_file = os.path.join(checkpoint_dir, "checkpoints.json")
    return CheckpointManager(checkpoint_dir=checkpoint_dir, meta_file=meta_file)


def make_file(tmp: str, name: str = "cp.zip") -> str:
    """在临时目录下创建一个占位检查点文件并返回路径。"""
    path = os.path.join(tmp, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("dummy")
    return path


def make_meta(
    version: str,
    path: str = "",
    algorithm: str = "ppo",
    timesteps: int = 10000,
    mean_reward: float = 100.0,
    std_reward: float = 10.0,
    created_at: str = "2026-01-01T00:00:00",
    tags: list[str] | None = None,
    notes: str = "",
) -> CheckpointMeta:
    """构造一个 CheckpointMeta 实例（便于测试中快速生成）。"""
    return CheckpointMeta(
        version=version,
        path=path,
        algorithm=algorithm,
        timesteps=timesteps,
        mean_reward=mean_reward,
        std_reward=std_reward,
        created_at=created_at,
        tags=tags if tags is not None else [],
        notes=notes,
    )


# ============================================================
# TestCheckpointMeta 数据类
# ============================================================
class TestCheckpointMeta(unittest.TestCase):
    """测试 CheckpointMeta 数据类的创建与序列化。"""

    def test_meta_creation_basic(self):
        """应正确创建包含全部必填字段的元数据。"""
        meta = CheckpointMeta(
            version="v1.0.0",
            path="/tmp/cp.zip",
            algorithm="ppo",
            timesteps=50000,
            mean_reward=200.0,
        )
        self.assertEqual(meta.version, "v1.0.0")
        self.assertEqual(meta.path, "/tmp/cp.zip")
        self.assertEqual(meta.algorithm, "ppo")
        self.assertEqual(meta.timesteps, 50000)
        self.assertEqual(meta.mean_reward, 200.0)
        self.assertEqual(meta.std_reward, 0.0)
        self.assertEqual(meta.tags, [])
        self.assertEqual(meta.notes, "")

    def test_meta_auto_created_at(self):
        """未提供 created_at 时应自动填充非空 ISO 时间字符串。"""
        meta = CheckpointMeta(
            version="v1",
            path="cp.zip",
            algorithm="dqn",
            timesteps=1000,
            mean_reward=10.0,
        )
        self.assertTrue(meta.created_at)
        # ISO 格式应包含日期分隔符
        self.assertIn("-", meta.created_at)

    def test_meta_preserves_custom_created_at(self):
        """显式提供的 created_at 应被保留，不被覆盖。"""
        custom = "2026-07-01T12:00:00"
        meta = CheckpointMeta(
            version="v1",
            path="cp.zip",
            algorithm="ppo",
            timesteps=1000,
            mean_reward=10.0,
            created_at=custom,
        )
        self.assertEqual(meta.created_at, custom)

    def test_meta_to_dict_from_dict_roundtrip(self):
        """to_dict / from_dict 应保持数据往返一致。"""
        meta = make_meta(
            "v1.0.0",
            path="/tmp/cp.zip",
            tags=["baseline", "exp1"],
            notes="首次实验",
        )
        data = meta.to_dict()
        restored = CheckpointMeta.from_dict(data)
        self.assertEqual(restored.version, meta.version)
        self.assertEqual(restored.path, meta.path)
        self.assertEqual(restored.algorithm, meta.algorithm)
        self.assertEqual(restored.timesteps, meta.timesteps)
        self.assertEqual(restored.mean_reward, meta.mean_reward)
        self.assertEqual(restored.std_reward, meta.std_reward)
        self.assertEqual(restored.created_at, meta.created_at)
        self.assertEqual(restored.tags, meta.tags)
        self.assertEqual(restored.notes, meta.notes)

    def test_meta_from_dict_ignores_unknown_fields(self):
        """from_dict 应忽略未知字段，不抛出异常。"""
        data = {
            "version": "v1",
            "path": "cp.zip",
            "algorithm": "ppo",
            "timesteps": 1000,
            "mean_reward": 10.0,
            "unknown_field": "should_be_ignored",
        }
        meta = CheckpointMeta.from_dict(data)
        self.assertEqual(meta.version, "v1")

    def test_meta_from_dict_none_tags_to_empty(self):
        """from_dict 在 tags 为 None 时应回退为空列表。"""
        data = {
            "version": "v1",
            "path": "cp.zip",
            "algorithm": "ppo",
            "timesteps": 1000,
            "mean_reward": 10.0,
            "tags": None,
        }
        meta = CheckpointMeta.from_dict(data)
        self.assertEqual(meta.tags, [])


# ============================================================
# TestRegister 注册
# ============================================================
class TestRegister(unittest.TestCase):
    """测试 CheckpointManager.register。"""

    def test_register_basic_returns_meta(self):
        """注册应返回包含传入参数的 CheckpointMeta。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            meta = mgr.register(
                path="/tmp/cp.zip",
                algorithm="ppo",
                timesteps=50000,
                mean_reward=200.0,
                std_reward=15.0,
            )
            self.assertIsInstance(meta, CheckpointMeta)
            self.assertEqual(meta.algorithm, "ppo")
            self.assertEqual(meta.timesteps, 50000)
            self.assertEqual(meta.mean_reward, 200.0)
            self.assertEqual(meta.std_reward, 15.0)
            self.assertTrue(meta.version)

    def test_register_auto_version_generated(self):
        """未提供 version 时应自动生成以 'v' 开头的版本号。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            meta = mgr.register(path="cp.zip", algorithm="ppo", timesteps=1000, mean_reward=10.0)
            self.assertTrue(meta.version.startswith("v"))
            self.assertGreater(len(meta.version), 1)

    def test_register_custom_version(self):
        """显式提供的版本号应被采用。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            meta = mgr.register(
                path="cp.zip",
                algorithm="dqn",
                timesteps=1000,
                mean_reward=10.0,
                version="v1.0.0",
            )
            self.assertEqual(meta.version, "v1.0.0")

    def test_register_creates_meta_file(self):
        """注册后应在 meta_file 路径创建 JSON 文件。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            self.assertFalse(os.path.exists(mgr.meta_file))
            mgr.register(path="cp.zip", algorithm="ppo", timesteps=1000, mean_reward=10.0)
            self.assertTrue(os.path.exists(mgr.meta_file))
            with open(mgr.meta_file, encoding="utf-8") as f:
                data = json.load(f)
            self.assertIsInstance(data, list)
            self.assertEqual(len(data), 1)

    def test_register_duplicate_version_raises(self):
        """显式提供已存在的版本号应抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.register(
                path="cp.zip",
                algorithm="ppo",
                timesteps=1000,
                mean_reward=10.0,
                version="v1.0.0",
            )
            with self.assertRaises(ValueError):
                mgr.register(
                    path="cp2.zip",
                    algorithm="dqn",
                    timesteps=2000,
                    mean_reward=20.0,
                    version="v1.0.0",
                )

    def test_register_with_tags_and_notes(self):
        """注册时应正确保存 tags 与 notes。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            meta = mgr.register(
                path="cp.zip",
                algorithm="ppo",
                timesteps=1000,
                mean_reward=10.0,
                tags=["baseline", "exp1"],
                notes="首次实验",
            )
            self.assertEqual(meta.tags, ["baseline", "exp1"])
            self.assertEqual(meta.notes, "首次实验")
            # 持久化后重新加载应一致
            loaded = mgr.load_meta()
            self.assertEqual(loaded[0].tags, ["baseline", "exp1"])
            self.assertEqual(loaded[0].notes, "首次实验")

    def test_register_tags_none_to_empty(self):
        """tags=None 时应保存为空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            meta = mgr.register(
                path="cp.zip",
                algorithm="ppo",
                timesteps=1000,
                mean_reward=10.0,
                tags=None,
            )
            self.assertEqual(meta.tags, [])

    def test_register_multiple_does_not_collide(self):
        """多次自动注册应生成互不相同的版本号。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            v1 = mgr.register(
                path="cp1.zip", algorithm="ppo", timesteps=1000, mean_reward=10.0
            ).version
            v2 = mgr.register(
                path="cp2.zip", algorithm="ppo", timesteps=2000, mean_reward=20.0
            ).version
            self.assertNotEqual(v1, v2)
            self.assertEqual(len(mgr.load_meta()), 2)


# ============================================================
# TestListCheckpoints 列出与排序
# ============================================================
class TestListCheckpoints(unittest.TestCase):
    """测试 CheckpointManager.list_checkpoints。"""

    def test_list_empty(self):
        """空管理器应返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            self.assertEqual(mgr.list_checkpoints(), [])

    def test_list_sort_by_created_at(self):
        """应按 created_at 字符串字典序排序（ISO 格式 == 时间顺序）。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta(
                [
                    make_meta("v3", created_at="2026-03-01T00:00:00"),
                    make_meta("v1", created_at="2026-01-01T00:00:00"),
                    make_meta("v2", created_at="2026-02-01T00:00:00"),
                ]
            )
            desc = mgr.list_checkpoints(sort_by="created_at", descending=True)
            self.assertEqual([cp.version for cp in desc], ["v3", "v2", "v1"])
            asc = mgr.list_checkpoints(sort_by="created_at", descending=False)
            self.assertEqual([cp.version for cp in asc], ["v1", "v2", "v3"])

    def test_list_sort_by_mean_reward(self):
        """应按 mean_reward 数值排序。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta(
                [
                    make_meta("v1", mean_reward=100.0),
                    make_meta("v2", mean_reward=300.0),
                    make_meta("v3", mean_reward=200.0),
                ]
            )
            desc = mgr.list_checkpoints(sort_by="mean_reward", descending=True)
            self.assertEqual([cp.version for cp in desc], ["v2", "v3", "v1"])

    def test_list_sort_by_timesteps(self):
        """应按 timesteps 数值排序。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta(
                [
                    make_meta("v1", timesteps=10000),
                    make_meta("v2", timesteps=50000),
                    make_meta("v3", timesteps=30000),
                ]
            )
            desc = mgr.list_checkpoints(sort_by="timesteps", descending=True)
            self.assertEqual([cp.version for cp in desc], ["v2", "v3", "v1"])

    def test_list_descending_vs_ascending(self):
        """descending=False 应返回升序。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta(
                [
                    make_meta("v1", mean_reward=100.0),
                    make_meta("v2", mean_reward=300.0),
                ]
            )
            asc = mgr.list_checkpoints(sort_by="mean_reward", descending=False)
            self.assertEqual([cp.version for cp in asc], ["v1", "v2"])

    def test_list_invalid_sort_by_raises(self):
        """不支持的排序字段应抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            with self.assertRaises(ValueError):
                mgr.list_checkpoints(sort_by="invalid_field")


# ============================================================
# TestGetBest 最优检查点
# ============================================================
class TestGetBest(unittest.TestCase):
    """测试 CheckpointManager.get_best。"""

    def test_get_best_mean_reward(self):
        """metric=mean_reward 应返回平均奖励最大的检查点。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta(
                [
                    make_meta("v1", mean_reward=100.0),
                    make_meta("v2", mean_reward=300.0),
                    make_meta("v3", mean_reward=200.0),
                ]
            )
            best = mgr.get_best(metric="mean_reward")
            assert best is not None
            self.assertEqual(best.version, "v2")

    def test_get_best_timesteps(self):
        """metric=timesteps 应返回训练步数最大的检查点。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta(
                [
                    make_meta("v1", timesteps=10000),
                    make_meta("v2", timesteps=50000),
                    make_meta("v3", timesteps=30000),
                ]
            )
            best = mgr.get_best(metric="timesteps")
            assert best is not None
            self.assertEqual(best.version, "v2")

    def test_get_best_empty_returns_none(self):
        """空管理器应返回 None。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            self.assertIsNone(mgr.get_best())

    def test_get_best_invalid_metric_raises(self):
        """不支持的指标应抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            with self.assertRaises(ValueError):
                mgr.get_best(metric="invalid_metric")


# ============================================================
# TestCompare 版本对比
# ============================================================
class TestCompare(unittest.TestCase):
    """测试 CheckpointManager.compare。"""

    def test_compare_basic(self):
        """应返回包含 reward_diff 与 timestep_diff 的字典。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta(
                [
                    make_meta("v1", mean_reward=200.0, timesteps=50000),
                    make_meta("v2", mean_reward=150.0, timesteps=30000),
                ]
            )
            result = mgr.compare("v1", "v2")
            self.assertEqual(result["version_a"], "v1")
            self.assertEqual(result["version_b"], "v2")
            self.assertEqual(result["reward_diff"], 50.0)
            self.assertEqual(result["timestep_diff"], 20000)

    def test_compare_improvement_pct_positive(self):
        """A 优于 B 时 improvement_pct 应为正值。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta(
                [
                    make_meta("v1", mean_reward=200.0),
                    make_meta("v2", mean_reward=100.0),
                ]
            )
            result = mgr.compare("v1", "v2")
            # (200-100)/|100| * 100 = 100.0
            self.assertEqual(result["improvement_pct"], 100.0)

    def test_compare_improvement_pct_negative(self):
        """A 差于 B 时 improvement_pct 应为负值。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta(
                [
                    make_meta("v1", mean_reward=80.0),
                    make_meta("v2", mean_reward=100.0),
                ]
            )
            result = mgr.compare("v1", "v2")
            # (80-100)/|100| * 100 = -20.0
            self.assertEqual(result["improvement_pct"], -20.0)

    def test_compare_missing_version_a_raises(self):
        """版本 A 不存在应抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta([make_meta("v1", mean_reward=100.0)])
            with self.assertRaises(ValueError):
                mgr.compare("v_missing", "v1")

    def test_compare_missing_version_b_raises(self):
        """版本 B 不存在应抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta([make_meta("v1", mean_reward=100.0)])
            with self.assertRaises(ValueError):
                mgr.compare("v1", "v_missing")

    def test_compare_zero_baseline(self):
        """基线 mean_reward=0 且 A>0 时 improvement_pct 应为 inf。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta(
                [
                    make_meta("v1", mean_reward=50.0),
                    make_meta("v2", mean_reward=0.0),
                ]
            )
            result = mgr.compare("v1", "v2")
            self.assertEqual(result["improvement_pct"], float("inf"))

    def test_compare_same_version(self):
        """对比同一版本时差值应为 0。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta([make_meta("v1", mean_reward=100.0, timesteps=10000)])
            result = mgr.compare("v1", "v1")
            self.assertEqual(result["reward_diff"], 0.0)
            self.assertEqual(result["timestep_diff"], 0)
            self.assertEqual(result["improvement_pct"], 0.0)


# ============================================================
# TestDelete 删除
# ============================================================
class TestDelete(unittest.TestCase):
    """测试 CheckpointManager.delete。"""

    def test_delete_removes_file_and_meta(self):
        """删除应同时移除检查点文件与元数据条目。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            cp_path = make_file(tmp, "cp1.zip")
            mgr.save_meta([make_meta("v1", path=cp_path)])
            self.assertTrue(os.path.exists(cp_path))

            result = mgr.delete("v1")
            self.assertTrue(result)
            self.assertFalse(os.path.exists(cp_path))
            self.assertEqual(mgr.load_meta(), [])

    def test_delete_nonexistent_returns_false(self):
        """删除不存在的版本应返回 False。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta([make_meta("v1", path="cp.zip")])
            result = mgr.delete("v_missing")
            self.assertFalse(result)
            self.assertEqual(len(mgr.load_meta()), 1)

    def test_delete_missing_file_still_removes_meta(self):
        """文件已不存在时，删除仍应移除元数据条目。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta([make_meta("v1", path="/nonexistent/cp.zip")])
            result = mgr.delete("v1")
            self.assertTrue(result)
            self.assertEqual(mgr.load_meta(), [])


# ============================================================
# TestTagUntag 标签
# ============================================================
class TestTagUntag(unittest.TestCase):
    """测试 CheckpointManager.tag / untag。"""

    def test_tag_adds_tag(self):
        """tag 应向指定版本添加标签。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta([make_meta("v1", tags=[])])
            mgr.tag("v1", "baseline")
            loaded = mgr.load_meta()
            self.assertEqual(loaded[0].tags, ["baseline"])

    def test_tag_idempotent(self):
        """重复添加同一标签应为幂等操作。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta([make_meta("v1", tags=["baseline"])])
            mgr.tag("v1", "baseline")
            mgr.tag("v1", "baseline")
            loaded = mgr.load_meta()
            self.assertEqual(loaded[0].tags, ["baseline"])

    def test_untag_removes_tag(self):
        """untag 应移除指定标签。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta([make_meta("v1", tags=["baseline", "exp1"])])
            mgr.untag("v1", "baseline")
            loaded = mgr.load_meta()
            self.assertEqual(loaded[0].tags, ["exp1"])

    def test_untag_nonexistent_tag_noop(self):
        """移除不存在的标签应为幂等操作，不报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta([make_meta("v1", tags=["baseline"])])
            mgr.untag("v1", "nonexistent")
            loaded = mgr.load_meta()
            self.assertEqual(loaded[0].tags, ["baseline"])

    def test_tag_nonexistent_version_raises(self):
        """向不存在的版本添加标签应抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta([make_meta("v1", tags=[])])
            with self.assertRaises(ValueError):
                mgr.tag("v_missing", "baseline")

    def test_untag_nonexistent_version_raises(self):
        """从不存在的版本移除标签应抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta([make_meta("v1", tags=[])])
            with self.assertRaises(ValueError):
                mgr.untag("v_missing", "baseline")


# ============================================================
# TestLoadSaveMeta 元数据持久化
# ============================================================
class TestLoadSaveMeta(unittest.TestCase):
    """测试 CheckpointManager.load_meta / save_meta。"""

    def test_load_meta_empty_when_no_file(self):
        """元数据文件不存在时应返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            self.assertEqual(mgr.load_meta(), [])

    def test_save_load_roundtrip(self):
        """save_meta → load_meta 应保持数据往返一致。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            original = [
                make_meta("v1", path="/tmp/a.zip", tags=["x"], notes="A"),
                make_meta("v2", path="/tmp/b.zip", mean_reward=250.0, tags=["y", "z"]),
            ]
            mgr.save_meta(original)
            loaded = mgr.load_meta()
            self.assertEqual(len(loaded), 2)
            self.assertEqual(loaded[0].version, "v1")
            self.assertEqual(loaded[0].tags, ["x"])
            self.assertEqual(loaded[0].notes, "A")
            self.assertEqual(loaded[1].version, "v2")
            self.assertEqual(loaded[1].mean_reward, 250.0)
            self.assertEqual(loaded[1].tags, ["y", "z"])

    def test_load_meta_malformed_json_returns_empty(self):
        """JSON 解析失败时应返回空列表，不抛出异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            os.makedirs(os.path.dirname(mgr.meta_file), exist_ok=True)
            with open(mgr.meta_file, "w", encoding="utf-8") as f:
                f.write("{not valid json}")
            self.assertEqual(mgr.load_meta(), [])

    def test_load_meta_non_list_returns_empty(self):
        """顶层结构非列表时应返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            os.makedirs(os.path.dirname(mgr.meta_file), exist_ok=True)
            with open(mgr.meta_file, "w", encoding="utf-8") as f:
                json.dump({"not": "a list"}, f)
            self.assertEqual(mgr.load_meta(), [])

    def test_save_meta_creates_parent_dir(self):
        """save_meta 应自动创建元数据文件的父目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c")
            meta_file = os.path.join(nested, "checkpoints.json")
            mgr = CheckpointManager(checkpoint_dir=nested, meta_file=meta_file)
            mgr.save_meta([make_meta("v1")])
            self.assertTrue(os.path.exists(meta_file))


# ============================================================
# TestCleanupOrphans 清理孤立条目
# ============================================================
class TestCleanupOrphans(unittest.TestCase):
    """测试 CheckpointManager.cleanup_orphans。"""

    def test_cleanup_removes_orphans(self):
        """应清理文件不存在的条目并返回其版本号。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            existing = make_file(tmp, "cp.zip")
            mgr.save_meta(
                [
                    make_meta("v1", path=existing),
                    make_meta("v2", path="/nonexistent/cp.zip"),
                    make_meta("v3", path="/another/missing.zip"),
                ]
            )
            removed = mgr.cleanup_orphans()
            self.assertEqual(sorted(removed), ["v2", "v3"])
            remaining = mgr.load_meta()
            self.assertEqual([cp.version for cp in remaining], ["v1"])

    def test_cleanup_keeps_existing_files(self):
        """文件存在的条目不应被清理。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            cp1 = make_file(tmp, "cp1.zip")
            cp2 = make_file(tmp, "cp2.zip")
            mgr.save_meta(
                [
                    make_meta("v1", path=cp1),
                    make_meta("v2", path=cp2),
                ]
            )
            removed = mgr.cleanup_orphans()
            self.assertEqual(removed, [])
            self.assertEqual(len(mgr.load_meta()), 2)

    def test_cleanup_no_orphans_returns_empty(self):
        """无孤立条目时应返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            cp = make_file(tmp, "cp.zip")
            mgr.save_meta([make_meta("v1", path=cp)])
            self.assertEqual(mgr.cleanup_orphans(), [])


# ============================================================
# TestEdgeCases 边界情况
# ============================================================
class TestEdgeCases(unittest.TestCase):
    """测试空管理器、不存在的版本等边界情况。"""

    def test_empty_manager_list(self):
        """空管理器 list_checkpoints 应返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            self.assertEqual(mgr.list_checkpoints(), [])
            self.assertEqual(mgr.list_checkpoints(sort_by="mean_reward"), [])

    def test_empty_manager_get_best(self):
        """空管理器 get_best 应返回 None。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            self.assertIsNone(mgr.get_best())
            self.assertIsNone(mgr.get_best(metric="timesteps"))

    def test_empty_manager_cleanup_returns_empty(self):
        """空管理器 cleanup_orphans 应返回空列表。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            self.assertEqual(mgr.cleanup_orphans(), [])

    def test_nonexistent_version_compare_raises(self):
        """对比不存在的版本应抛出 ValueError。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            with self.assertRaises(ValueError):
                mgr.compare("v_missing_a", "v_missing_b")

    def test_register_default_args(self):
        """register 使用默认参数应正常工作。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            meta = mgr.register(path="cp.zip", algorithm="ppo", timesteps=1000, mean_reward=10.0)
            self.assertEqual(meta.std_reward, 0.0)
            self.assertEqual(meta.tags, [])
            self.assertEqual(meta.notes, "")

    def test_delete_on_empty_manager_returns_false(self):
        """空管理器 delete 应返回 False。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            self.assertFalse(mgr.delete("v_missing"))

    def test_tag_preserves_other_fields(self):
        """添加标签不应影响其他字段。"""
        with tempfile.TemporaryDirectory() as tmp:
            mgr = make_manager(tmp)
            mgr.save_meta(
                [
                    make_meta(
                        "v1",
                        path="/tmp/cp.zip",
                        algorithm="dqn",
                        timesteps=20000,
                        mean_reward=180.0,
                        tags=["a"],
                        notes="keep",
                    )
                ]
            )
            mgr.tag("v1", "b")
            loaded = mgr.load_meta()
            self.assertEqual(loaded[0].tags, ["a", "b"])
            self.assertEqual(loaded[0].algorithm, "dqn")
            self.assertEqual(loaded[0].timesteps, 20000)
            self.assertEqual(loaded[0].mean_reward, 180.0)
            self.assertEqual(loaded[0].notes, "keep")

    def test_manager_init_creates_checkpoint_dir(self):
        """__init__ 应自动创建 checkpoint_dir。"""
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint_dir = os.path.join(tmp, "new_models_dir")
            meta_file = os.path.join(checkpoint_dir, "checkpoints.json")
            self.assertFalse(os.path.exists(checkpoint_dir))
            CheckpointManager(checkpoint_dir=checkpoint_dir, meta_file=meta_file)
            self.assertTrue(os.path.isdir(checkpoint_dir))


if __name__ == "__main__":
    unittest.main()
