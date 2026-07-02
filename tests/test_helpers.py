"""
量子RL调度系统 - 工具函数单元测试
Unit Tests for src/utils/helpers.py

测试覆盖：
- setup_logging 日志配置（创建目录、返回 logger、已存在目录、自定义级别）
- load_config / save_config 配置加载与保存（YAML round-trip、缺失/非法文件、中文）
- normalize_vector 向量归一化（基本、自定义范围、空、常量、负值、单元素）
- one_hot_encode 独热编码（各位置、未知类别、空列表）
- format_time 时间格式化（秒/分/时、边界、零、负数、大值）
- save_json / load_json JSON 序列化（round-trip、嵌套、父目录、缺失文件、中文、列表）
- MetricsCalculator 评估指标（奖励各分支、改进百分比各分支）
- calculate_completion_rate / average_wait_time / resource_utilization / get_current_timestamp
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.helpers import (
    MetricsCalculator,
    calculate_average_wait_time,
    calculate_completion_rate,
    calculate_resource_utilization,
    format_time,
    get_current_timestamp,
    load_config,
    load_json,
    normalize_vector,
    one_hot_encode,
    save_config,
    save_json,
    setup_logging,
)


# ============================================================
# setup_logging 测试
# ============================================================
class TestSetupLogging(unittest.TestCase):
    """测试 setup_logging 日志配置。"""

    def test_creates_log_dir_and_returns_logger(self):
        """应在指定目录创建日志并返回 logger。"""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "logs")
            lg = setup_logging(log_dir=log_dir, log_file="test.log")
            try:
                self.assertIsNotNone(lg)
                self.assertTrue(os.path.isdir(log_dir))
            finally:
                # 关闭 loguru 文件句柄，避免 Windows 临时目录清理锁文件
                lg.remove()

    def test_handles_existing_dir(self):
        """已存在的日志目录不应报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            log_dir = os.path.join(tmp, "logs")
            os.makedirs(log_dir)
            lg = setup_logging(log_dir=log_dir, log_file="a.log")
            try:
                self.assertIsNotNone(lg)
            finally:
                lg.remove()

    def test_custom_log_level(self):
        """自定义日志级别应被接受且不抛异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            lg = setup_logging(log_dir=tmp, log_level="DEBUG")
            try:
                self.assertIsNotNone(lg)
            finally:
                lg.remove()

    def test_creates_log_file_on_write(self):
        """配置后日志目录应可被创建（间接验证 makedirs 调用）。"""
        with tempfile.TemporaryDirectory() as tmp:
            nested = os.path.join(tmp, "a", "b", "c")
            lg = setup_logging(log_dir=nested, log_file="x.log")
            try:
                self.assertIsNotNone(lg)
                self.assertTrue(os.path.isdir(nested))
            finally:
                lg.remove()


# ============================================================
# load_config 测试
# ============================================================
class TestLoadConfig(unittest.TestCase):
    """测试 load_config 配置加载。"""

    def test_loads_valid_yaml(self):
        """应正确加载 YAML 配置文件为字典。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cfg.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("name: test\nvalue: 42\nnested:\n  a: 1\n")
            cfg = load_config(path)
            self.assertIsInstance(cfg, dict)
            self.assertEqual(cfg["name"], "test")
            self.assertEqual(cfg["value"], 42)
            self.assertEqual(cfg["nested"]["a"], 1)

    def test_missing_file_returns_empty_dict(self):
        """文件不存在时应返回空字典而非抛异常。"""
        cfg = load_config("/nonexistent/path/config.yaml")
        self.assertEqual(cfg, {})

    def test_invalid_yaml_returns_empty_dict(self):
        """无效 YAML 应返回空字典而非抛异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bad.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("[1, 2, 3\n")  # 未闭合的 flow 序列
            cfg = load_config(path)
            self.assertEqual(cfg, {})

    def test_loads_list_root(self):
        """YAML 根为列表时应返回列表（非字典，记录实际行为）。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("- a\n- b\n")
            cfg = load_config(path)
            self.assertEqual(cfg, ["a", "b"])

    def test_expands_env_vars_in_strings(self):
        """配置中的 ${VAR} 引用应被展开为环境变量值。"""
        os.environ["TEST_CONFIG_SECRET"] = "my-secret-value"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "env.yaml")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("key: ${TEST_CONFIG_SECRET}\nplain: hello\n")
                cfg = load_config(path)
                self.assertEqual(cfg["key"], "my-secret-value")
                self.assertEqual(cfg["plain"], "hello")
        finally:
            del os.environ["TEST_CONFIG_SECRET"]

    def test_expands_env_vars_in_nested_dicts(self):
        """嵌套字典中的 ${VAR} 引用应被递归展开。"""
        os.environ["TEST_NESTED"] = "expanded"
        try:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "nested.yaml")
                with open(path, "w", encoding="utf-8") as f:
                    f.write("outer:\n  inner: ${TEST_NESTED}\n  other: plain\n")
                cfg = load_config(path)
                self.assertEqual(cfg["outer"]["inner"], "expanded")
                self.assertEqual(cfg["outer"]["other"], "plain")
        finally:
            del os.environ["TEST_NESTED"]

    def test_unset_env_var_keeps_placeholder(self):
        """未设置的环境变量 ${} 引用保留原样。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "unset.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("key: ${UNDEFINED_VAR_XYZ}\n")
            cfg = load_config(path)
            self.assertEqual(cfg["key"], "${UNDEFINED_VAR_XYZ}")

    def test_no_env_vars_works_normally(self):
        """没有环境变量引用的配置应正常加载。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "plain.yaml")
            with open(path, "w", encoding="utf-8") as f:
                f.write("name: test\nvalue: 42\n")
            cfg = load_config(path)
            self.assertEqual(cfg["name"], "test")
            self.assertEqual(cfg["value"], 42)


# ============================================================
# save_config 测试
# ============================================================
class TestSaveConfig(unittest.TestCase):
    """测试 save_config 配置保存。"""

    def test_save_and_load_round_trip(self):
        """保存后加载应得到等价配置（round-trip）。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "cfg.yaml")
            original = {"name": "demo", "count": 3, "items": ["a", "b"]}
            save_config(original, path)
            self.assertTrue(os.path.exists(path))
            loaded = load_config(path)
            self.assertEqual(loaded, original)

    def test_save_creates_parent_dir(self):
        """保存时应自动创建父目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "deep", "nest", "cfg.yaml")
            save_config({"x": 1}, path)
            self.assertTrue(os.path.exists(path))

    def test_save_unicode_content(self):
        """应正确保存含中文的配置并 round-trip。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cn.yaml")
            save_config({"描述": "量子调度"}, path)
            loaded = load_config(path)
            self.assertEqual(loaded["描述"], "量子调度")

    def test_save_nested_dict(self):
        """嵌套字典应被正确保存与加载。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested.yaml")
            original = {"a": {"b": {"c": 1}}, "list": [1, 2, 3]}
            save_config(original, path)
            self.assertEqual(load_config(path), original)


# ============================================================
# normalize_vector 测试
# ============================================================
class TestNormalizeVector(unittest.TestCase):
    """测试 normalize_vector 向量归一化。"""

    def test_basic_normalization(self):
        """基本归一化应映射到 [0, 1]。"""
        result = normalize_vector([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertEqual(len(result), 5)
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[-1], 1.0)
        self.assertAlmostEqual(result[2], 0.5)

    def test_custom_range(self):
        """自定义 [min, max] 范围应正确缩放。"""
        result = normalize_vector([0.0, 5.0, 10.0], min_val=-1.0, max_val=1.0)
        self.assertAlmostEqual(result[0], -1.0)
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 1.0)

    def test_empty_vector(self):
        """空向量应返回空列表。"""
        self.assertEqual(normalize_vector([]), [])

    def test_constant_vector_returns_midpoint(self):
        """所有值相同（min==max）应返回全 0.5。"""
        result = normalize_vector([5.0, 5.0, 5.0])
        self.assertEqual(result, [0.5, 0.5, 0.5])

    def test_negative_values(self):
        """含负值的向量应正确归一化到 [0, 1]。"""
        result = normalize_vector([-2.0, 0.0, 2.0])
        self.assertAlmostEqual(result[0], 0.0)
        self.assertAlmostEqual(result[1], 0.5)
        self.assertAlmostEqual(result[2], 1.0)

    def test_single_element_constant(self):
        """单元素向量（min==max）应返回 [0.5]。"""
        result = normalize_vector([7.0])
        self.assertEqual(result, [0.5])

    def test_custom_range_constant_returns_midpoint(self):
        """常量向量在自定义范围下应返回范围中点。"""
        result = normalize_vector([3.0, 3.0], min_val=0.0, max_val=10.0)
        # min==max → 返回 [0.5, 0.5]（与范围无关，固定中点）
        self.assertEqual(result, [0.5, 0.5])


# ============================================================
# one_hot_encode 测试
# ============================================================
class TestOneHotEncode(unittest.TestCase):
    """测试 one_hot_encode 独热编码。"""

    def test_correct_encoding_first(self):
        """首个类别应编码为 [1,0,0]。"""
        self.assertEqual(
            one_hot_encode("quantum", ["quantum", "classical", "hybrid"]),
            [1, 0, 0],
        )

    def test_correct_encoding_middle(self):
        """中间类别应正确编码。"""
        self.assertEqual(
            one_hot_encode("classical", ["quantum", "classical", "hybrid"]),
            [0, 1, 0],
        )

    def test_correct_encoding_last(self):
        """最后一个类别应编码为 [0,0,1]。"""
        self.assertEqual(
            one_hot_encode("hybrid", ["quantum", "classical", "hybrid"]),
            [0, 0, 1],
        )

    def test_unknown_category_returns_zeros(self):
        """未知类别应返回全零向量。"""
        self.assertEqual(
            one_hot_encode("unknown", ["quantum", "classical"]),
            [0, 0],
        )

    def test_empty_categories(self):
        """空类别列表应返回空编码。"""
        self.assertEqual(one_hot_encode("x", []), [])

    def test_length_matches_categories(self):
        """编码长度应等于类别列表长度。"""
        cats = ["a", "b", "c", "d"]
        enc = one_hot_encode("c", cats)
        self.assertEqual(len(enc), len(cats))


# ============================================================
# format_time 测试
# ============================================================
class TestFormatTime(unittest.TestCase):
    """测试 format_time 时间格式化。"""

    def test_seconds(self):
        """小于 60 秒应格式化为秒。"""
        self.assertEqual(format_time(30.0), "30.0秒")

    def test_zero(self):
        """0 秒应格式化为秒。"""
        self.assertEqual(format_time(0.0), "0.0秒")

    def test_minutes(self):
        """60-3600 秒应格式化为分钟。"""
        self.assertEqual(format_time(90.0), "1.5分钟")

    def test_hours(self):
        """大于 3600 秒应格式化为小时。"""
        self.assertEqual(format_time(7200.0), "2.0小时")

    def test_boundary_60(self):
        """60 秒边界应格式化为分钟。"""
        self.assertEqual(format_time(60.0), "1.0分钟")

    def test_boundary_3600(self):
        """3600 秒边界应格式化为小时。"""
        self.assertEqual(format_time(3600.0), "1.0小时")

    def test_negative_seconds(self):
        """负数秒应落入秒分支（记录实际行为）。"""
        self.assertEqual(format_time(-5.0), "-5.0秒")

    def test_large_value_hours(self):
        """大数值应格式化为小时。"""
        self.assertEqual(format_time(86400.0), "24.0小时")

    def test_returns_str(self):
        """返回值应为字符串类型。"""
        self.assertIsInstance(format_time(1.0), str)


# ============================================================
# save_json / load_json 测试
# ============================================================
class TestSaveLoadJson(unittest.TestCase):
    """测试 save_json / load_json JSON 序列化。"""

    def test_save_load_round_trip(self):
        """保存后加载应得到等价数据。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "data.json")
            data = {"a": 1, "b": [1, 2, 3], "c": "hello"}
            save_json(data, path)
            loaded = load_json(path)
            self.assertEqual(loaded, data)

    def test_save_nested_dict(self):
        """嵌套字典应正确序列化并 round-trip。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nested.json")
            data = {"outer": {"inner": {"deep": [1, 2, {"x": True}]}}}
            save_json(data, path)
            self.assertEqual(load_json(path), data)

    def test_save_creates_parent_dir(self):
        """保存时应自动创建父目录。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "dir", "f.json")
            save_json({"k": "v"}, path)
            self.assertTrue(os.path.exists(path))

    def test_load_missing_file_returns_none(self):
        """加载不存在的文件应返回 None。"""
        self.assertIsNone(load_json("/nonexistent/file.json"))

    def test_save_unicode(self):
        """含中文的 JSON 应正确保存与加载。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "cn.json")
            save_json({"任务": "量子调度"}, path)
            self.assertEqual(load_json(path)["任务"], "量子调度")

    def test_save_list_data(self):
        """列表数据应正确序列化并 round-trip。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "list.json")
            data = [1, 2, {"x": 3}]
            save_json(data, path)
            self.assertEqual(load_json(path), data)

    def test_save_boolean_and_null(self):
        """布尔值与 None 应正确序列化。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "bool.json")
            save_json({"flag": True, "none_val": None}, path)
            loaded = load_json(path)
            self.assertTrue(loaded["flag"])
            self.assertIsNone(loaded["none_val"])


# ============================================================
# MetricsCalculator 测试
# ============================================================
class TestMetricsCalculator(unittest.TestCase):
    """测试 MetricsCalculator 评估指标。"""

    def test_reward_perfect_inputs(self):
        """完美输入（完成率1、等待0、利用率1）应得满分 1.0。"""
        reward = MetricsCalculator.calculate_reward(
            completion_rate=1.0, avg_wait_time=0.0, resource_utilization=1.0
        )
        self.assertAlmostEqual(reward, 1.0)

    def test_reward_known_value(self):
        """已知输入应得到正确加权奖励。"""
        # normalized_wait = 1 - 120/3600 = 0.9667
        # reward = 0.4*0.85 + 0.3*0.9667 + 0.3*0.75 = 0.855
        reward = MetricsCalculator.calculate_reward(
            completion_rate=0.85, avg_wait_time=120.0, resource_utilization=0.75
        )
        self.assertAlmostEqual(reward, 0.855, places=3)

    def test_reward_wait_capped_at_max(self):
        """等待时间超过 max_wait_time 时归一化应截断为 0。"""
        reward = MetricsCalculator.calculate_reward(
            completion_rate=1.0,
            avg_wait_time=7200.0,
            resource_utilization=1.0,
            max_wait_time=3600.0,
        )
        # normalized_wait = 1 - min(7200/3600,1) = 0 → reward = 0.4 + 0 + 0.3 = 0.7
        self.assertAlmostEqual(reward, 0.7)

    def test_reward_zero_inputs(self):
        """全零输入（完成率0、等待满、利用率0）应得 0 奖励。"""
        reward = MetricsCalculator.calculate_reward(0.0, 3600.0, 0.0)
        # normalized_wait = 1 - 1 = 0 → reward = 0
        self.assertAlmostEqual(reward, 0.0)

    def test_reward_custom_max_wait(self):
        """自定义 max_wait_time 应影响归一化。"""
        # avg_wait=50, max=100 → normalized_wait = 1 - 0.5 = 0.5
        # reward = 0.4*1 + 0.3*0.5 + 0.3*1 = 0.4+0.15+0.3 = 0.85
        reward = MetricsCalculator.calculate_reward(
            completion_rate=1.0,
            avg_wait_time=50.0,
            resource_utilization=1.0,
            max_wait_time=100.0,
        )
        self.assertAlmostEqual(reward, 0.85)

    def test_improvement_positive(self):
        """正改进应返回正百分比。"""
        self.assertAlmostEqual(MetricsCalculator.calculate_improvement(120.0, 100.0), 20.0)

    def test_improvement_negative(self):
        """负改进应返回负百分比。"""
        self.assertAlmostEqual(MetricsCalculator.calculate_improvement(80.0, 100.0), -20.0)

    def test_improvement_baseline_zero_new_zero(self):
        """基线与新值均为 0 应返回 0。"""
        self.assertAlmostEqual(MetricsCalculator.calculate_improvement(0.0, 0.0), 0.0)

    def test_improvement_baseline_zero_new_nonzero(self):
        """基线为 0、新值非 0 应返回 100。"""
        self.assertAlmostEqual(MetricsCalculator.calculate_improvement(5.0, 0.0), 100.0)

    def test_improvement_negative_baseline(self):
        """负基线应正确计算改进百分比。"""
        # (10 - (-10)) / |-10| * 100 = 200
        self.assertAlmostEqual(MetricsCalculator.calculate_improvement(10.0, -10.0), 200.0)

    def test_improvement_no_change(self):
        """新值等于基线应返回 0。"""
        self.assertAlmostEqual(MetricsCalculator.calculate_improvement(50.0, 50.0), 0.0)


# ============================================================
# 其他工具函数测试
# ============================================================
class TestAdditionalUtils(unittest.TestCase):
    """测试其他工具函数（完成率/平均等待/资源利用率/时间戳）。"""

    def test_completion_rate_normal(self):
        """正常完成率计算。"""
        self.assertAlmostEqual(calculate_completion_rate(8, 10), 0.8)

    def test_completion_rate_full(self):
        """全部完成应返回 1.0。"""
        self.assertAlmostEqual(calculate_completion_rate(10, 10), 1.0)

    def test_completion_rate_zero_total(self):
        """总数为 0 应返回 0。"""
        self.assertEqual(calculate_completion_rate(0, 0), 0.0)

    def test_average_wait_time_normal(self):
        """正常平均等待时间。"""
        self.assertAlmostEqual(calculate_average_wait_time([10.0, 20.0, 30.0]), 20.0)

    def test_average_wait_time_empty(self):
        """空列表应返回 0。"""
        self.assertEqual(calculate_average_wait_time([]), 0.0)

    def test_average_wait_time_single(self):
        """单元素应返回该元素值。"""
        self.assertAlmostEqual(calculate_average_wait_time([42.5]), 42.5)

    def test_resource_utilization_normal(self):
        """正常资源利用率。"""
        self.assertAlmostEqual(calculate_resource_utilization(7.0, 10.0), 0.7)

    def test_resource_utilization_zero_total(self):
        """总资源为 0 应返回 0。"""
        self.assertEqual(calculate_resource_utilization(5.0, 0.0), 0.0)

    def test_resource_utilization_full(self):
        """满载应返回 1.0。"""
        self.assertAlmostEqual(calculate_resource_utilization(10.0, 10.0), 1.0)

    def test_get_current_timestamp_format(self):
        """时间戳应为 YYYY-MM-DD HH:MM:SS 格式。"""
        ts = get_current_timestamp()
        self.assertIsInstance(ts, str)
        self.assertEqual(len(ts), 19)
        self.assertEqual(ts[4], "-")
        self.assertEqual(ts[7], "-")
        self.assertEqual(ts[10], " ")
        self.assertEqual(ts[13], ":")
        self.assertEqual(ts[16], ":")

    def test_get_current_timestamp_is_valid_datetime(self):
        """时间戳应可被解析为有效日期时间。"""
        ts = get_current_timestamp()
        parsed = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        self.assertIsInstance(parsed, datetime)


if __name__ == "__main__":
    unittest.main()
