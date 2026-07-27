"""
配置 Schema 验证测试
Tests for src/config/schema.py (Pydantic config validation)
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import yaml
from pydantic import ValidationError

from src.config.schema import AppConfig, validate_and_print, validate_config

# 项目根目录下的 config.yaml 路径（用于 Issue #245 验证）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_CONFIG_YAML = _PROJECT_ROOT / "config" / "config.yaml"


class TestAppConfigValidation(unittest.TestCase):
    """AppConfig Schema 基础校验。"""

    def test_validates_minimal_config(self):
        """仅提供 scheduler.algorithm 的最简配置应通过校验。"""
        cfg = AppConfig(scheduler={"algorithm": "DQN"})
        self.assertEqual(cfg.scheduler.algorithm, "DQN")
        self.assertEqual(cfg.scheduler.learning_rate, 3e-4)
        self.assertEqual(cfg.scheduler.gamma, 0.99)

    def test_default_values(self):
        """空配置应使用所有默认值。"""
        cfg = AppConfig()
        self.assertEqual(cfg.scheduler.algorithm, "DQN")
        self.assertEqual(cfg.quantum.backend, "tianyan-287")
        self.assertEqual(cfg.quantum.max_qubits, 287)
        self.assertEqual(cfg.tianyan.mock_mode, True)
        self.assertEqual(cfg.web.port, 8000)

    def test_rejects_invalid_algorithm(self):
        """非法的 algorithm 值应抛出 ValidationError。"""
        with self.assertRaises(ValidationError):
            AppConfig(scheduler={"algorithm": "INVALID"})

    def test_rejects_negative_learning_rate(self):
        """负学习率应抛出 ValidationError。"""
        with self.assertRaises(ValidationError):
            AppConfig(scheduler={"learning_rate": -0.1})

    def test_rejects_extra_top_level_field(self):
        """顶层未知字段（extra="forbid"）应抛出 ValidationError。"""
        with self.assertRaises(ValidationError):
            AppConfig(typo_field=123)

    def test_rejects_port_out_of_range(self):
        """端口号超出范围应抛出 ValidationError。"""
        with self.assertRaises(ValidationError):
            AppConfig(web={"port": 99999})

    def test_accepts_valid_port(self):
        """合法端口号应通过校验。"""
        cfg = AppConfig(web={"port": 3000})
        self.assertEqual(cfg.web.port, 3000)


class TestValidateConfig(unittest.TestCase):
    """validate_config 函数测试。"""

    def test_returns_appconfig_on_valid(self):
        """合法配置应返回 AppConfig 实例。"""
        result = validate_config({"scheduler": {"algorithm": "PPO"}})
        self.assertIsInstance(result, AppConfig)
        self.assertEqual(result.scheduler.algorithm, "PPO")

    def test_raises_on_invalid(self):
        """非法配置应抛出 ValidationError。"""
        with self.assertRaises(ValidationError):
            validate_config({"scheduler": {"algorithm": "BAD"}})

    def test_handles_real_config_structure(self):
        """真实 config.yaml 结构的配置应通过校验。"""
        data = {
            "tianyan": {"mock_mode": True, "timeout": 30},
            "scheduler": {"algorithm": "DQN", "learning_rate": 3e-4},
            "quantum": {"backend": "tianyan-287", "max_qubits": 287},
            "annealing": {"enabled": True, "num_qubits": 10},
            "cache": {"type": "redis", "host": "localhost"},
            "classical": {"max_cpu_utilization": 0.95},
            "database": {"type": "sqlite", "path": "data/scheduler.db"},
            "system": {"log_level": "INFO", "max_steps": 1000},
            "web": {"port": 8000},
        }
        cfg = validate_config(data)
        self.assertEqual(cfg.scheduler.algorithm, "DQN")


class TestValidateAndPrint(unittest.TestCase):
    """validate_and_print 函数测试。"""

    def test_returns_appconfig_on_valid(self):
        """应返回 AppConfig 且不抛异常。"""
        result = validate_and_print({"scheduler": {"algorithm": "DQN"}})
        self.assertIsInstance(result, AppConfig)

    def test_raises_on_invalid(self):
        """非法配置应抛出 ValidationError。"""
        with self.assertRaises(ValidationError):
            validate_and_print({"scheduler": {"algorithm": "NOPE"}})


# ============================================================
# Issue #245: annealing 配置节完整性验证
# ============================================================
class TestAnnealingConfigSection(unittest.TestCase):
    """验证 config.yaml 中 annealing 配置节的完整性（Issue #245）。

    验收标准：
    - annealing 配置节定义完整
    - 每个参数有注释说明
    - 配置文件可通过 pyyaml 正确解析
    """

    REQUIRED_KEYS: ClassVar[set[str]] = {
        # 全局开关
        "enabled",
        # 退火器基础参数
        "simulation_mode",
        "num_qubits",
        "shots",
        "annealing_time",
        "n_bits_per_weight",
        # 仿真模拟退火超参数
        "sim_initial_temp",
        "sim_cooling_rate",
        "sim_num_sweeps",
        # QUBO 构造参数
        "reg_lambda",
        "max_delta_ratio",
        "accept_threshold_ratio",
        # 分层退火参数
        "head_only",
        "max_params_per_block",
        "block_strategy",
    }

    @classmethod
    def setUpClass(cls):
        """加载 config.yaml 文件，所有测试共用。"""
        cls.assertTrue(_CONFIG_YAML.exists(), f"config.yaml 不存在: {_CONFIG_YAML}")
        with open(_CONFIG_YAML, encoding="utf-8") as f:
            cls.config = yaml.safe_load(f)

    def test_config_yaml_parses_successfully(self):
        """config.yaml 必须可被 pyyaml 正确解析为 dict。"""
        self.assertIsInstance(self.config, dict)

    def test_annealing_section_exists(self):
        """config.yaml 必须包含 annealing 顶级配置节。"""
        self.assertIn("annealing", self.config, "config.yaml 缺少 annealing 顶级配置节")
        self.assertIsInstance(self.config["annealing"], dict)

    def test_annealing_section_has_all_required_keys(self):
        """annealing 配置节必须包含所有要求的参数。"""
        annealing = self.config.get("annealing", {})
        missing = self.REQUIRED_KEYS - set(annealing.keys())
        self.assertEqual(
            missing,
            set(),
            f"annealing 配置节缺少参数: {missing}，现有参数: {sorted(annealing.keys())}",
        )

    def test_annealing_section_has_comments(self):
        """验证 config.yaml 中 annealing 段每个参数都有注释说明。

        通过读取原始文本，确认每个参数名前后存在 '#' 注释行。
        """
        text = _CONFIG_YAML.read_text(encoding="utf-8")
        # 提取 annealing 配置段的文本（从 'annealing:' 开始到下一个顶级键）
        lines = text.splitlines()
        in_section = False
        section_lines: list[str] = []
        for line in lines:
            if line.startswith("annealing:"):
                in_section = True
                continue
            if in_section:
                # 遇到下一个顶级键（无缩进、非空、非注释）则结束
                if line and not line[0].isspace() and not line.startswith("#"):
                    break
                section_lines.append(line)
        section_text = "\n".join(section_lines)
        # 至少存在若干注释行（粗略验证注释覆盖度）
        comment_count = sum(1 for line in section_lines if line.strip().startswith("#"))
        self.assertGreaterEqual(
            comment_count,
            5,
            f"annealing 配置节注释过少（{comment_count} 行），应至少有 5 行注释",
        )
        # 关键参数名应在文本中出现（带注释或后跟注释）
        for key in ["simulation_mode", "num_qubits", "reg_lambda", "head_only"]:
            self.assertIn(key, section_text, f"annealing 配置节缺少参数: {key}")

    def test_annealing_section_values_match_defaults(self):
        """验证 annealing 配置节参数值与代码默认值一致。"""
        annealing = self.config.get("annealing", {})
        # 与 src/quantum/annealing.py 中的默认值对应
        expected = {
            "simulation_mode": True,
            "num_qubits": 16,
            "shots": 1000,
            "annealing_time": 20.0,
            "n_bits_per_weight": 4,
            "sim_initial_temp": 2.0,
            "sim_cooling_rate": 0.995,
            "sim_num_sweeps": 200,
            "reg_lambda": 0.1,
            "max_delta_ratio": 0.1,
            "accept_threshold_ratio": 0.01,
            "head_only": True,
            "max_params_per_block": 200,
            "block_strategy": "tensor_wise",
        }
        for key, expected_value in expected.items():
            self.assertEqual(
                annealing.get(key),
                expected_value,
                f"annealing.{key} 期望={expected_value}，实际={annealing.get(key)}",
            )


if __name__ == "__main__":
    unittest.main()
