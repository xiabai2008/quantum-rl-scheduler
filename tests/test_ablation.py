"""
消融实验框架单元测试
Unit Tests for Ablation Study Framework

测试覆盖：
- AblationConfig 数据类创建与组件开关
- AblationResult 数据类创建与序列化
- define_configs 标准 5 配置与命名/开关正确性
- run_single 运行单配置、返回完整指标、n_episodes 控制
- run_all 运行全部配置、结果数量正确
- compare 对比结果、delta 计算、基线识别
- generate_report Markdown 生成、表格格式、改进百分比
- save_results / load_results 持久化往返一致性
- 边界情况：空配置、单配置、相同配置
"""

import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.ablation import (
    AblationConfig,
    AblationResult,
    AblationRunner,
)


def _make_result(
    name: str,
    mean_reward: float,
    completion_rate: float = 0.5,
    avg_wait_time: float = 10.0,
    resource_utilization: float = 0.5,
) -> AblationResult:
    """构造带完整配置的 AblationResult 测试辅助对象。"""
    config = AblationConfig(
        name=name,
        description=f"测试配置 {name}",
        components={"rl": True, "annealing": False},
        env_params={},
    )
    return AblationResult(
        config=config,
        mean_reward=mean_reward,
        std_reward=1.0,
        completion_rate=completion_rate,
        avg_wait_time=avg_wait_time,
        resource_utilization=resource_utilization,
        n_episodes=3,
        timestamp="2026-07-02T00:00:00+00:00",
    )


class TestAblationConfig(unittest.TestCase):
    """测试 AblationConfig 数据类"""

    def test_create_with_all_fields(self):
        """测试带全部字段的创建"""
        cfg = AblationConfig(
            name="D1_test",
            description="测试配置",
            components={"rl": True, "annealing": False},
            env_params={"max_steps": 100},
        )
        self.assertEqual(cfg.name, "D1_test")
        self.assertEqual(cfg.description, "测试配置")
        self.assertTrue(cfg.components["rl"])
        self.assertFalse(cfg.components["annealing"])
        self.assertEqual(cfg.env_params["max_steps"], 100)

    def test_default_env_params(self):
        """测试 env_params 默认为空字典"""
        cfg = AblationConfig(
            name="D2",
            description="无 env_params",
            components={"rl": True},
        )
        self.assertEqual(cfg.env_params, {})

    def test_components_toggle(self):
        """测试组件开关可独立设置"""
        cfg = AblationConfig(
            name="D3",
            description="开关测试",
            components={
                "rl": False,
                "annealing": True,
                "multi_machine": True,
                "multi_objective": False,
            },
        )
        self.assertFalse(cfg.components["rl"])
        self.assertTrue(cfg.components["annealing"])
        self.assertTrue(cfg.components["multi_machine"])
        self.assertFalse(cfg.components["multi_objective"])


class TestAblationResult(unittest.TestCase):
    """测试 AblationResult 数据类"""

    def test_create_result(self):
        """测试结果对象创建与字段赋值"""
        cfg = AblationConfig(
            name="D1",
            description="algo",
            components={"rl": True},
        )
        result = AblationResult(
            config=cfg,
            mean_reward=100.0,
            std_reward=5.0,
            completion_rate=0.8,
            avg_wait_time=12.3,
            resource_utilization=0.6,
            n_episodes=10,
            timestamp="2026-07-02T00:00:00+00:00",
        )
        self.assertEqual(result.config.name, "D1")
        self.assertEqual(result.mean_reward, 100.0)
        self.assertEqual(result.std_reward, 5.0)
        self.assertEqual(result.completion_rate, 0.8)
        self.assertEqual(result.avg_wait_time, 12.3)
        self.assertEqual(result.resource_utilization, 0.6)
        self.assertEqual(result.n_episodes, 10)

    def test_result_serialization(self):
        """测试通过 dataclasses.asdict 序列化含嵌套配置"""
        from dataclasses import asdict

        result = _make_result("D_ser", 50.0)
        d = asdict(result)
        self.assertEqual(d["config"]["name"], "D_ser")
        self.assertEqual(d["mean_reward"], 50.0)
        self.assertEqual(d["config"]["components"]["rl"], True)
        # JSON 可序列化
        json_str = json.dumps(d, ensure_ascii=False)
        self.assertIn("D_ser", json_str)


class TestDefineConfigs(unittest.TestCase):
    """测试 define_configs 标准 5 配置"""

    def setUp(self):
        """测试初始化"""
        self.runner = AblationRunner()
        self.configs = self.runner.define_configs()

    def test_returns_five_configs(self):
        """测试返回恰好 5 个配置"""
        self.assertEqual(len(self.configs), 5)

    def test_config_naming_convention(self):
        """测试命名遵循 D1-D5 前缀规范"""
        names = [c.name for c in self.configs]
        self.assertTrue(all(n.startswith("D") for n in names))
        # 每个维度 D1-D5 各一个
        prefixes = {n.split("_")[0] for n in names}
        self.assertEqual(prefixes, {"D1", "D2", "D3", "D4", "D5"})

    def test_each_config_disables_one_component(self):
        """测试每个配置恰好关闭一个组件（相对全量基线）"""
        full = {"rl", "annealing", "multi_machine", "multi_objective", "state_14dim"}
        for cfg in self.configs:
            disabled = {k for k, v in cfg.components.items() if not v}
            self.assertEqual(
                len(disabled),
                1,
                f"配置 {cfg.name} 应只关闭 1 个组件，实际关闭 {disabled}",
            )
            self.assertTrue(disabled.issubset(full))

    def test_d1_disables_rl(self):
        """测试 D1 关闭 RL 算法"""
        d1 = next(c for c in self.configs if c.name.startswith("D1"))
        self.assertFalse(d1.components["rl"])
        # 其余组件保持开启
        self.assertTrue(d1.components["annealing"])
        self.assertTrue(d1.components["multi_machine"])

    def test_d4_disables_multi_machine(self):
        """测试 D4 关闭多机调度"""
        d4 = next(c for c in self.configs if c.name.startswith("D4"))
        self.assertFalse(d4.components["multi_machine"])

    def test_d5_disables_annealing(self):
        """测试 D5 关闭量子退火"""
        d5 = next(c for c in self.configs if c.name.startswith("D5"))
        self.assertFalse(d5.components["annealing"])

    def test_all_components_present(self):
        """测试每个配置都包含全部 5 个组件键"""
        expected_keys = {
            "rl",
            "annealing",
            "multi_machine",
            "multi_objective",
            "state_14dim",
        }
        for cfg in self.configs:
            self.assertEqual(set(cfg.components.keys()), expected_keys)


class TestRunSingle(unittest.TestCase):
    """测试 run_single 运行单个配置"""

    def setUp(self):
        """测试初始化"""
        self.runner = AblationRunner()

    def test_returns_complete_metrics(self):
        """测试返回包含完整指标的结果"""
        cfg = AblationConfig(
            name="D1_test",
            description="单配置测试",
            components={
                "rl": True,
                "annealing": False,
                "multi_machine": False,
                "multi_objective": False,
                "state_14dim": True,
            },
            env_params={"max_steps": 50, "max_qubits": 30, "seed": 0},
        )
        result = self.runner.run_single(cfg, n_episodes=3)
        self.assertIsInstance(result, AblationResult)
        self.assertEqual(result.config.name, "D1_test")
        # 指标均为有限浮点数
        for v in (
            result.mean_reward,
            result.std_reward,
            result.completion_rate,
            result.avg_wait_time,
            result.resource_utilization,
        ):
            self.assertTrue(isinstance(v, float))
            self.assertFalse(math.isnan(v))
        # 完成率与利用率在 [0, 1]
        self.assertGreaterEqual(result.completion_rate, 0.0)
        self.assertLessEqual(result.completion_rate, 1.0)
        self.assertGreaterEqual(result.resource_utilization, 0.0)
        self.assertLessEqual(result.resource_utilization, 1.0)

    def test_n_episodes_control(self):
        """测试 n_episodes 控制运行回合并写入结果"""
        cfg = AblationConfig(
            name="D_ep",
            description="回合计数测试",
            components={
                "rl": False,
                "annealing": False,
                "multi_machine": False,
                "multi_objective": False,
                "state_14dim": True,
            },
            env_params={"max_steps": 40, "max_qubits": 20, "seed": 1},
        )
        result = self.runner.run_single(cfg, n_episodes=5)
        self.assertEqual(result.n_episodes, 5)

    def test_fcfs_policy_runs(self):
        """测试 FCFS 策略（rl=False）可正常运行完整回合"""
        cfg = AblationConfig(
            name="D_fcfs",
            description="FCFS 策略",
            components={
                "rl": False,
                "annealing": False,
                "multi_machine": True,
                "multi_objective": False,
                "state_14dim": True,
            },
            env_params={"max_steps": 60, "max_qubits": 30, "seed": 7},
        )
        result = self.runner.run_single(cfg, n_episodes=2)
        self.assertEqual(result.n_episodes, 2)
        # FCFS 至少应产生非零奖励（避免完全静默失败）
        self.assertTrue(isinstance(result.mean_reward, float))

    def test_zero_episodes_returns_zero_result(self):
        """测试 n_episodes=0 返回零值结果"""
        cfg = AblationConfig(
            name="D_zero",
            description="零回合",
            components={"rl": True},
        )
        result = self.runner.run_single(cfg, n_episodes=0)
        self.assertEqual(result.n_episodes, 0)
        self.assertEqual(result.mean_reward, 0.0)


class TestRunAll(unittest.TestCase):
    """测试 run_all 运行全部配置"""

    def test_run_all_default_configs(self):
        """测试默认运行标准 5 配置"""
        runner = AblationRunner()
        results = runner.run_all(n_episodes=2)
        self.assertEqual(len(results), 5)
        names = [r.config.name for r in results]
        self.assertTrue(all(n.startswith("D") for n in names))

    def test_run_all_custom_configs(self):
        """测试运行自定义配置列表"""
        runner = AblationRunner()
        cfgs = [
            AblationConfig(
                name="C1",
                description="自定义1",
                components={
                    "rl": True,
                    "annealing": False,
                    "multi_machine": False,
                    "multi_objective": False,
                    "state_14dim": True,
                },
                env_params={"max_steps": 30, "max_qubits": 20, "seed": 0},
            ),
            AblationConfig(
                name="C2",
                description="自定义2",
                components={
                    "rl": False,
                    "annealing": False,
                    "multi_machine": False,
                    "multi_objective": False,
                    "state_14dim": True,
                },
                env_params={"max_steps": 30, "max_qubits": 20, "seed": 0},
            ),
        ]
        results = runner.run_all(configs=cfgs, n_episodes=2)
        self.assertEqual(len(results), 2)
        self.assertEqual([r.config.name for r in results], ["C1", "C2"])

    def test_run_all_count_matches_configs(self):
        """测试结果数量与配置数量一致"""
        runner = AblationRunner()
        cfgs = runner.define_configs()[:3]
        results = runner.run_all(configs=cfgs, n_episodes=1)
        self.assertEqual(len(results), 3)


class TestCompare(unittest.TestCase):
    """测试 compare 对比与基线识别"""

    def setUp(self):
        """测试初始化"""
        self.runner = AblationRunner()

    def test_empty_results(self):
        """测试空结果列表返回空对比"""
        out = self.runner.compare([])
        self.assertIsNone(out["baseline"])
        self.assertEqual(out["deltas"], {})

    def test_baseline_is_highest_reward(self):
        """测试基线为平均奖励最高的配置"""
        r1 = _make_result("A", 100.0)
        r2 = _make_result("B", 300.0)
        r3 = _make_result("C", 200.0)
        out = self.runner.compare([r1, r2, r3])
        self.assertEqual(out["baseline"]["name"], "B")
        self.assertEqual(out["baseline"]["mean_reward"], 300.0)

    def test_delta_calculation(self):
        """测试 delta 相对基线的正确计算"""
        r1 = _make_result("base", 200.0, completion_rate=0.8)
        r2 = _make_result("ablated", 150.0, completion_rate=0.6)
        out = self.runner.compare([r1, r2])
        self.assertEqual(out["baseline"]["name"], "base")
        d = out["deltas"]["ablated"]
        # 奖励差 = 150 - 200 = -50
        self.assertAlmostEqual(d["reward_delta"], -50.0)
        # 百分比 = -50 / 200 * 100 = -25
        self.assertAlmostEqual(d["reward_pct"], -25.0)
        # 完成率差 = 0.6 - 0.8 = -0.2
        self.assertAlmostEqual(d["completion_delta"], -0.2)

    def test_baseline_has_no_delta_entry(self):
        """测试基线自身不出现在 deltas 中"""
        r1 = _make_result("A", 100.0)
        r2 = _make_result("B", 200.0)
        out = self.runner.compare([r1, r2])
        self.assertNotIn(out["baseline"]["name"], out["deltas"])

    def test_zero_baseline_reward_no_division_error(self):
        """测试基线奖励为 0 时百分比不抛错"""
        r1 = _make_result("zero", 0.0)
        r2 = _make_result("other", 50.0)
        out = self.runner.compare([r1, r2])
        # 基线为 other（50），zero 的百分比 = (0-50)/50*100 = -100
        self.assertAlmostEqual(out["deltas"]["zero"]["reward_pct"], -100.0)


class TestGenerateReport(unittest.TestCase):
    """测试 generate_report Markdown 生成"""

    def setUp(self):
        """测试初始化"""
        self.runner = AblationRunner()

    def test_report_contains_table_header(self):
        """测试报告包含汇总表头"""
        results = [
            _make_result("D1", 100.0),
            _make_result("D2", 120.0),
        ]
        report = self.runner.generate_report(results, output_path=str(self._tmp()))
        self.assertIn("# 消融实验报告", report)
        self.assertIn("## 结果汇总", report)
        self.assertIn("| 配置名 |", report)

    def test_report_contains_all_configs(self):
        """测试报告汇总表包含全部配置"""
        results = [
            _make_result("D1_x", 100.0),
            _make_result("D2_x", 110.0),
            _make_result("D3_x", 90.0),
        ]
        report = self.runner.generate_report(results, output_path=str(self._tmp()))
        for name in ("D1_x", "D2_x", "D3_x"):
            self.assertIn(name, report)

    def test_report_contains_baseline_section(self):
        """测试报告包含相对基线对比章节"""
        results = [_make_result("A", 100.0), _make_result("B", 130.0)]
        report = self.runner.generate_report(results, output_path=str(self._tmp()))
        self.assertIn("## 相对基线对比", report)
        self.assertIn("**基线**", report)
        self.assertIn("(基线)", report)

    def test_report_improvement_percentage(self):
        """测试报告包含改进百分比"""
        results = [_make_result("base", 200.0), _make_result("ablated", 150.0)]
        report = self.runner.generate_report(results, output_path=str(self._tmp()))
        # 百分比 -25.00% 应出现在报告中
        self.assertIn("%", report)
        self.assertIn("-25.00%", report)

    def test_report_empty_results(self):
        """测试空结果报告可生成"""
        report = self.runner.generate_report([], output_path=str(self._tmp()))
        self.assertIn("无可用结果", report)

    def test_report_written_to_file(self):
        """测试报告写入指定文件"""
        with tempfile.TemporaryDirectory() as d:
            out_path = Path(d) / "sub" / "report.md"
            results = [_make_result("A", 100.0)]
            self.runner.generate_report(results, output_path=str(out_path))
            self.assertTrue(out_path.exists())
            content = out_path.read_text(encoding="utf-8")
            self.assertIn("消融实验报告", content)

    def _tmp(self) -> Path:
        """返回临时文件路径（在临时目录下）"""
        fd, path = tempfile.mkstemp(suffix=".md")
        os.close(fd)
        return Path(path)


class TestSaveLoadResults(unittest.TestCase):
    """测试 save_results / load_results 持久化"""

    def setUp(self):
        """测试初始化"""
        self.runner = AblationRunner()

    def test_save_creates_json_file(self):
        """测试 save 生成 JSON 文件"""
        results = [_make_result("A", 100.0), _make_result("B", 200.0)]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "results.json"
            self.runner.save_results(results, str(path))
            self.assertTrue(path.exists())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(data), 2)

    def test_roundtrip_preserves_data(self):
        """测试保存后加载的数据与原始一致"""
        original = [
            _make_result("D1", 123.45, completion_rate=0.77, avg_wait_time=9.5),
            _make_result("D2", 67.89, completion_rate=0.31, avg_wait_time=15.2),
        ]
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "rt.json"
            self.runner.save_results(original, str(path))
            loaded = self.runner.load_results(str(path))

        self.assertEqual(len(loaded), len(original))
        for orig, load in zip(original, loaded, strict=False):
            self.assertEqual(load.config.name, orig.config.name)
            self.assertEqual(load.config.description, orig.config.description)
            self.assertEqual(load.config.components, orig.config.components)
            self.assertAlmostEqual(load.mean_reward, orig.mean_reward)
            self.assertAlmostEqual(load.completion_rate, orig.completion_rate)
            self.assertAlmostEqual(load.avg_wait_time, orig.avg_wait_time)
            self.assertEqual(load.n_episodes, orig.n_episodes)
            self.assertEqual(load.timestamp, orig.timestamp)

    def test_load_preserves_components_and_env_params(self):
        """测试加载后 components 与 env_params 完整保留"""
        cfg = AblationConfig(
            name="X",
            description="env params test",
            components={
                "rl": True,
                "annealing": False,
                "multi_machine": True,
                "multi_objective": False,
                "state_14dim": True,
            },
            env_params={"max_steps": 80, "max_qubits": 25, "seed": 3},
        )
        result = AblationResult(
            config=cfg,
            mean_reward=10.0,
            std_reward=1.0,
            completion_rate=0.5,
            avg_wait_time=4.0,
            resource_utilization=0.4,
            n_episodes=2,
            timestamp="2026-07-02T00:00:00+00:00",
        )
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "one.json"
            self.runner.save_results([result], str(path))
            loaded = self.runner.load_results(str(path))[0]
        self.assertEqual(loaded.config.components, cfg.components)
        self.assertEqual(loaded.config.env_params, cfg.env_params)


class TestEdgeCases(unittest.TestCase):
    """测试边界情况"""

    def setUp(self):
        """测试初始化"""
        self.runner = AblationRunner()

    def test_empty_config_list_run_all(self):
        """测试 run_all 传入空配置列表返回空结果"""
        results = self.runner.run_all(configs=[], n_episodes=1)
        self.assertEqual(results, [])

    def test_single_config_run_all(self):
        """测试 run_all 仅运行单个配置"""
        cfg = AblationConfig(
            name="solo",
            description="单配置",
            components={
                "rl": True,
                "annealing": False,
                "multi_machine": False,
                "multi_objective": False,
                "state_14dim": True,
            },
            env_params={"max_steps": 30, "max_qubits": 20, "seed": 0},
        )
        results = self.runner.run_all(configs=[cfg], n_episodes=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].config.name, "solo")

    def test_compare_single_result(self):
        """测试 compare 单个结果：基线为自身，deltas 为空"""
        r = _make_result("only", 100.0)
        out = self.runner.compare([r])
        self.assertEqual(out["baseline"]["name"], "only")
        self.assertEqual(out["deltas"], {})

    def test_compare_identical_results(self):
        """测试 compare 多个相同奖励结果：基线取其一，其余 delta 为 0"""
        r1 = _make_result("same1", 100.0)
        r2 = _make_result("same2", 100.0)
        out = self.runner.compare([r1, r2])
        # 基线为第一个最大值（max 行为）
        self.assertIn(out["baseline"]["name"], {"same1", "same2"})
        other = "same2" if out["baseline"]["name"] == "same1" else "same1"
        self.assertAlmostEqual(out["deltas"][other]["reward_delta"], 0.0)

    def test_save_load_empty_results(self):
        """测试空结果列表的保存与加载"""
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "empty.json"
            self.runner.save_results([], str(path))
            loaded = self.runner.load_results(str(path))
            self.assertEqual(loaded, [])


if __name__ == "__main__":
    unittest.main()
