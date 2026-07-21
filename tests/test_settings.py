"""
统一配置加载测试
Tests for src/config/settings.py (unified config loader)

测试覆盖：
- 默认值加载（缺失文件回退）
- 环境变量覆盖
- .env 文件覆盖
- 三层优先级（env > .env > yaml）
- 类型转换（int / float / bool）
- ${VAR} 展开
- 字段完整性
- .env 文件解析
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config.settings import (
    Settings,
    _parse_env_file,
    _to_bool,
    load_settings,
)

# 测试中需要管理的环境变量名集合
_ENV_VARS = [
    "TIANYAN_API_KEY",
    "TIANYAN_API_TOKEN",
    "TIANYAN_API_TIMEOUT",
    "TIANYAN_API_MAX_RETRIES",
    "QUANTUM_MAX_QUBITS",
    "SCHEDULER_MAX_STEPS",
    "SCHEDULER_ALGORITHM",
    "ANNEALING_ENABLED",
    "QUANTUM_SHOTS",
    "LOG_LEVEL",
    "LOG_FORMAT",
    "LOG_DIR",
    "VIZ_API_KEY",
    "WEB_PORT",
    "MY_TEST_SECRET",
    "MY_TEST_BASE",
]


def _write_yaml(path: str, content: str) -> None:
    """写入 YAML 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _write_env(path: str, content: str) -> None:
    """写入 .env 文件。"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


class TestEnvVarsMixin:
    """环境变量保存/恢复混入。

    在 setUp 保存指定环境变量快照，tearDown 还原，避免测试间污染。
    """

    def _snapshot_env(self) -> None:
        self._saved_env: dict[str, str | None] = {k: os.environ.get(k) for k in _ENV_VARS}
        # 清空所有受管环境变量，确保测试起点干净
        for k in _ENV_VARS:
            os.environ.pop(k, None)

    def _restore_env(self) -> None:
        for k, v in self._saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestDefaults(unittest.TestCase, TestEnvVarsMixin):
    """默认值加载测试。"""

    def setUp(self) -> None:
        self._snapshot_env()

    def tearDown(self) -> None:
        self._restore_env()

    def test_load_with_missing_files_returns_defaults(self):
        """配置文件和 .env 都不存在时应返回全部默认值。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "no.yaml")
            env = os.path.join(tmp, "no.env")
            s = load_settings(config_path=cfg, env_path=env)
            self.assertEqual(s.api_key, "")
            self.assertEqual(s.api_token, "")
            self.assertEqual(s.api_timeout, 30.0)
            self.assertEqual(s.api_retries, 3)
            self.assertEqual(s.max_qubits, 287)
            self.assertEqual(s.max_steps, 1000)
            self.assertEqual(s.algorithm, "DQN")
            self.assertTrue(s.annealing_enabled)
            self.assertEqual(s.quantum_shots, 1024)
            self.assertEqual(s.log_level, "INFO")
            self.assertEqual(s.log_format, "text")
            self.assertEqual(s.log_dir, "logs")
            self.assertEqual(s.viz_api_key, "")
            self.assertEqual(s.viz_port, 8000)

    def test_from_env_returns_settings_instance(self):
        """from_env 类方法应返回 Settings 实例。"""
        s = Settings.from_env()
        self.assertIsInstance(s, Settings)


class TestEnvVarOverride(unittest.TestCase, TestEnvVarsMixin):
    """环境变量覆盖测试。"""

    def setUp(self) -> None:
        self._snapshot_env()

    def tearDown(self) -> None:
        self._restore_env()

    def test_env_var_overrides_default(self):
        """环境变量应覆盖字段默认值。"""
        os.environ["TIANYAN_API_KEY"] = "env_key_123"
        with tempfile.TemporaryDirectory() as tmp:
            s = load_settings(
                config_path=os.path.join(tmp, "no.yaml"),
                env_path=os.path.join(tmp, "no.env"),
            )
            self.assertEqual(s.api_key, "env_key_123")

    def test_env_var_overrides_yaml(self):
        """环境变量优先级高于 config.yaml。"""
        os.environ["QUANTUM_MAX_QUBITS"] = "64"
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "c.yaml")
            _write_yaml(cfg, "quantum:\n  max_qubits: 128\n")
            s = load_settings(
                config_path=cfg,
                env_path=os.path.join(tmp, "no.env"),
            )
            # 环境变量 64 应胜过 yaml 的 128
            self.assertEqual(s.max_qubits, 64)

    def test_env_var_int_conversion(self):
        """字符串环境变量应转换为 int。"""
        os.environ["SCHEDULER_MAX_STEPS"] = "5000"
        with tempfile.TemporaryDirectory() as tmp:
            s = load_settings(
                config_path=os.path.join(tmp, "no.yaml"),
                env_path=os.path.join(tmp, "no.env"),
            )
            self.assertEqual(s.max_steps, 5000)
            self.assertIsInstance(s.max_steps, int)

    def test_env_var_float_conversion(self):
        """字符串环境变量应转换为 float。"""
        os.environ["TIANYAN_API_TIMEOUT"] = "12.5"
        with tempfile.TemporaryDirectory() as tmp:
            s = load_settings(
                config_path=os.path.join(tmp, "no.yaml"),
                env_path=os.path.join(tmp, "no.env"),
            )
            self.assertEqual(s.api_timeout, 12.5)
            self.assertIsInstance(s.api_timeout, float)

    def test_env_var_bool_true_values(self):
        """bool 类型应支持 true/1/yes/on（大小写不敏感）。"""
        truthy_values = ["true", "TRUE", "1", "yes", "on", "True"]
        for val in truthy_values:
            with self.subTest(value=val):
                os.environ["ANNEALING_ENABLED"] = val
                with tempfile.TemporaryDirectory() as tmp:
                    s = load_settings(
                        config_path=os.path.join(tmp, "no.yaml"),
                        env_path=os.path.join(tmp, "no.env"),
                    )
                    self.assertTrue(s.annealing_enabled, f"应解析为 True: {val!r}")

    def test_env_var_bool_false_values(self):
        """bool 类型应支持 false/0/no/off/空串。"""
        falsy_values = ["false", "FALSE", "0", "no", "off", ""]
        for val in falsy_values:
            with self.subTest(value=val):
                os.environ["ANNEALING_ENABLED"] = val
                with tempfile.TemporaryDirectory() as tmp:
                    s = load_settings(
                        config_path=os.path.join(tmp, "no.yaml"),
                        env_path=os.path.join(tmp, "no.env"),
                    )
                    self.assertFalse(s.annealing_enabled, f"应解析为 False: {val!r}")

    def test_invalid_int_falls_back_to_default(self):
        """无法转换为 int 的环境变量应回退到默认值。"""
        os.environ["SCHEDULER_MAX_STEPS"] = "not_a_number"
        with tempfile.TemporaryDirectory() as tmp:
            s = load_settings(
                config_path=os.path.join(tmp, "no.yaml"),
                env_path=os.path.join(tmp, "no.env"),
            )
            self.assertEqual(s.max_steps, 1000)


class TestEnvFileOverride(unittest.TestCase, TestEnvVarsMixin):
    """.env 文件覆盖测试。"""

    def setUp(self) -> None:
        self._snapshot_env()

    def tearDown(self) -> None:
        self._restore_env()

    def test_env_file_overrides_default(self):
        """.env 文件应覆盖字段默认值。"""
        with tempfile.TemporaryDirectory() as tmp:
            env = os.path.join(tmp, ".env")
            _write_env(env, "TIANYAN_API_KEY=from_env_file\n")
            s = load_settings(
                config_path=os.path.join(tmp, "no.yaml"),
                env_path=env,
            )
            self.assertEqual(s.api_key, "from_env_file")

    def test_env_file_overrides_yaml(self):
        """.env 文件优先级高于 config.yaml。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "c.yaml")
            _write_yaml(cfg, "scheduler:\n  algorithm: PPO\n")
            env = os.path.join(tmp, ".env")
            _write_env(env, "SCHEDULER_ALGORITHM=MAPPO\n")
            s = load_settings(config_path=cfg, env_path=env)
            self.assertEqual(s.algorithm, "MAPPO")

    def test_env_file_skips_comments_and_blanks(self):
        """.env 文件应跳过注释行与空行。"""
        with tempfile.TemporaryDirectory() as tmp:
            env = os.path.join(tmp, ".env")
            _write_env(
                env,
                "# 这是注释\n\nTIANYAN_API_KEY=k1\n\n# 另一段注释\nLOG_LEVEL=DEBUG\n",
            )
            parsed = _parse_env_file(env)
            self.assertEqual(parsed.get("TIANYAN_API_KEY"), "k1")
            self.assertEqual(parsed.get("LOG_LEVEL"), "DEBUG")
            self.assertNotIn("# 这是注释", parsed)

    def test_env_file_strips_quotes(self):
        """.env 文件值两端的引号应被剥离。"""
        with tempfile.TemporaryDirectory() as tmp:
            env = os.path.join(tmp, ".env")
            _write_env(
                env,
                "TIANYAN_API_KEY=\"quoted_key\"\nTIANYAN_API_TOKEN='single_quoted'\n",
            )
            parsed = _parse_env_file(env)
            self.assertEqual(parsed.get("TIANYAN_API_KEY"), "quoted_key")
            self.assertEqual(parsed.get("TIANYAN_API_TOKEN"), "single_quoted")

    def test_missing_env_file_returns_empty(self):
        """.env 文件不存在时 _parse_env_file 返回空字典。"""
        result = _parse_env_file("/nonexistent/path/.env")
        self.assertEqual(result, {})


class TestThreeLayerPriority(unittest.TestCase, TestEnvVarsMixin):
    """三层优先级测试：环境变量 > .env 文件 > config.yaml。"""

    def setUp(self) -> None:
        self._snapshot_env()

    def tearDown(self) -> None:
        self._restore_env()

    def test_priority_env_beats_envfile_beats_yaml(self):
        """同一字段在三层都设置时，环境变量应胜出。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "c.yaml")
            _write_yaml(cfg, "web:\n  port: 1000\n")
            env = os.path.join(tmp, ".env")
            _write_env(env, "WEB_PORT=2000\n")
            os.environ["WEB_PORT"] = "3000"
            s = load_settings(config_path=cfg, env_path=env)
            self.assertEqual(s.viz_port, 3000)

    def test_priority_envfile_beats_yaml_when_no_env(self):
        """无环境变量时，.env 文件应胜过 config.yaml。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "c.yaml")
            _write_yaml(cfg, "web:\n  port: 1000\n")
            env = os.path.join(tmp, ".env")
            _write_env(env, "WEB_PORT=2000\n")
            s = load_settings(config_path=cfg, env_path=env)
            self.assertEqual(s.viz_port, 2000)

    def test_priority_yaml_used_when_no_env_no_envfile(self):
        """无环境变量与 .env 时，应使用 config.yaml 值。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "c.yaml")
            _write_yaml(cfg, "web:\n  port: 1000\n")
            s = load_settings(
                config_path=cfg,
                env_path=os.path.join(tmp, "no.env"),
            )
            self.assertEqual(s.viz_port, 1000)


class TestVarExpansion(unittest.TestCase, TestEnvVarsMixin):
    """${VAR} 变量引用展开测试。"""

    def setUp(self) -> None:
        self._snapshot_env()

    def tearDown(self) -> None:
        self._restore_env()

    def test_yaml_var_expansion(self):
        """config.yaml 中的 ${VAR} 应使用 os.environ 展开。"""
        os.environ["MY_TEST_SECRET"] = "expanded_secret"
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "c.yaml")
            _write_yaml(cfg, "tianyan:\n  api_key: ${MY_TEST_SECRET}\n")
            s = load_settings(
                config_path=cfg,
                env_path=os.path.join(tmp, "no.env"),
            )
            self.assertEqual(s.api_key, "expanded_secret")

    def test_env_file_var_expansion(self):
        """.env 文件中的 ${VAR} 应使用 os.environ 展开。"""
        os.environ["MY_TEST_BASE"] = "prod"
        with tempfile.TemporaryDirectory() as tmp:
            env = os.path.join(tmp, ".env")
            _write_env(env, "TIANYAN_API_KEY=${MY_TEST_BASE}_key\n")
            s = load_settings(
                config_path=os.path.join(tmp, "no.yaml"),
                env_path=env,
            )
            self.assertEqual(s.api_key, "prod_key")

    def test_yaml_var_unresolved_left_as_is(self):
        """config.yaml 中未定义的 ${VAR} 应保留原样（os.path.expandvars 行为）。"""
        # 确保变量未定义
        os.environ.pop("UNDEFINED_VAR_XYZ", None)
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "c.yaml")
            _write_yaml(cfg, "tianyan:\n  api_key: ${UNDEFINED_VAR_XYZ}\n")
            s = load_settings(
                config_path=cfg,
                env_path=os.path.join(tmp, "no.env"),
            )
            # os.path.expandvars 对未定义变量保留 ${VAR} 原样
            self.assertEqual(s.api_key, "${UNDEFINED_VAR_XYZ}")


class TestFieldIntegrity(unittest.TestCase, TestEnvVarsMixin):
    """字段完整性测试。"""

    def setUp(self) -> None:
        self._snapshot_env()

    def tearDown(self) -> None:
        self._restore_env()

    def test_all_expected_fields_present(self):
        """Settings 应包含全部预期字段。"""
        expected = {
            "api_key",
            "api_token",
            "api_timeout",
            "api_retries",
            "max_qubits",
            "max_steps",
            "algorithm",
            "annealing_enabled",
            "quantum_shots",
            "log_level",
            "log_format",
            "log_dir",
            "viz_api_key",
            "viz_port",
        }
        actual = {f.name for f in __import__("dataclasses").fields(Settings)}
        self.assertEqual(actual, expected)

    def test_field_types(self):
        """字段类型应符合规范。"""
        from dataclasses import fields as dfields

        type_map = {f.name: f.type for f in dfields(Settings)}
        self.assertEqual(type_map["api_key"], "str")
        self.assertEqual(type_map["api_token"], "str")
        self.assertEqual(type_map["api_timeout"], "float")
        self.assertEqual(type_map["api_retries"], "int")
        self.assertEqual(type_map["max_qubits"], "int")
        self.assertEqual(type_map["max_steps"], "int")
        self.assertEqual(type_map["algorithm"], "str")
        self.assertEqual(type_map["annealing_enabled"], "bool")
        self.assertEqual(type_map["quantum_shots"], "int")
        self.assertEqual(type_map["log_level"], "str")
        self.assertEqual(type_map["log_format"], "str")
        self.assertEqual(type_map["log_dir"], "str")
        self.assertEqual(type_map["viz_api_key"], "str")
        self.assertEqual(type_map["viz_port"], "int")

    def test_to_dict_returns_all_fields(self):
        """to_dict 应返回全部字段。"""
        s = Settings()
        d = s.to_dict()
        self.assertEqual(len(d), 14)
        self.assertIn("api_key", d)
        self.assertIn("viz_port", d)
        self.assertIn("annealing_enabled", d)

    def test_load_full_config_from_yaml(self):
        """完整 config.yaml 加载应正确映射所有字段。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "c.yaml")
            _write_yaml(
                cfg,
                "tianyan:\n"
                "  api_key: yaml_key\n"
                "  timeout: 45\n"
                "scheduler:\n"
                "  algorithm: PPO\n"
                "quantum:\n"
                "  max_qubits: 50\n"
                "  shots: 2048\n"
                "annealing:\n"
                "  enabled: false\n"
                "system:\n"
                "  log_level: WARNING\n"
                "  max_steps: 2000\n"
                "web:\n"
                "  port: 9000\n",
            )
            s = load_settings(
                config_path=cfg,
                env_path=os.path.join(tmp, "no.env"),
            )
            self.assertEqual(s.api_key, "yaml_key")
            self.assertEqual(s.api_timeout, 45.0)
            self.assertEqual(s.algorithm, "PPO")
            self.assertEqual(s.max_qubits, 50)
            self.assertEqual(s.quantum_shots, 2048)
            self.assertFalse(s.annealing_enabled)
            self.assertEqual(s.log_level, "WARNING")
            self.assertEqual(s.max_steps, 2000)
            self.assertEqual(s.viz_port, 9000)


class TestToBoolHelper(unittest.TestCase):
    """_to_bool 辅助函数单元测试。"""

    def test_bool_input_passthrough(self):
        self.assertTrue(_to_bool(True))
        self.assertFalse(_to_bool(False))

    def test_int_input(self):
        self.assertTrue(_to_bool(1))
        self.assertFalse(_to_bool(0))

    def test_string_truthy(self):
        for v in ["true", "TRUE", "True", "1", "yes", "on"]:
            self.assertTrue(_to_bool(v), f"应解析为 True: {v!r}")

    def test_string_falsy(self):
        for v in ["false", "FALSE", "False", "0", "no", "off", ""]:
            self.assertFalse(_to_bool(v), f"应解析为 False: {v!r}")

    def test_unknown_value_fallback(self):
        """未知字符串按 bool(value) 兜底（非空为 True）。"""
        self.assertTrue(_to_bool("something"))


class TestMissingConfigFallback(unittest.TestCase, TestEnvVarsMixin):
    """缺失配置文件回退测试。"""

    def setUp(self) -> None:
        self._snapshot_env()

    def tearDown(self) -> None:
        self._restore_env()

    def test_invalid_yaml_falls_back_to_defaults(self):
        """非法 YAML 文件应回退到默认值（不抛异常）。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "bad.yaml")
            _write_yaml(cfg, ": : : not valid yaml : : :\n")
            s = load_settings(
                config_path=cfg,
                env_path=os.path.join(tmp, "no.env"),
            )
            # 应得到默认值
            self.assertEqual(s.max_qubits, 287)
            self.assertEqual(s.algorithm, "DQN")

    def test_empty_yaml_uses_defaults(self):
        """空 YAML 文件应使用默认值。"""
        with tempfile.TemporaryDirectory() as tmp:
            cfg = os.path.join(tmp, "empty.yaml")
            _write_yaml(cfg, "")
            s = load_settings(
                config_path=cfg,
                env_path=os.path.join(tmp, "no.env"),
            )
            self.assertEqual(s.max_qubits, 287)


class TestAppEnvSelection(unittest.TestCase, TestEnvVarsMixin):
    """APP_ENV 环境变量自动选择配置文件测试（Issue #214）。"""

    def setUp(self) -> None:
        self._snapshot_env()
        # APP_ENV 测试需要额外管理 APP_ENV 变量
        self._saved_app_env = os.environ.get("APP_ENV")
        os.environ.pop("APP_ENV", None)

    def tearDown(self) -> None:
        self._restore_env()
        if self._saved_app_env is None:
            os.environ.pop("APP_ENV", None)
        else:
            os.environ["APP_ENV"] = self._saved_app_env

    def test_app_env_dev_loads_dev_config(self):
        """APP_ENV=dev 应加载 config.dev.yaml。"""
        os.environ["APP_ENV"] = "dev"
        with tempfile.TemporaryDirectory() as tmp:
            # 在 tmp/config/config.dev.yaml 下创建开发配置
            config_dir = os.path.join(tmp, "config")
            os.makedirs(config_dir, exist_ok=True)
            dev_cfg = os.path.join(config_dir, "config.dev.yaml")
            _write_yaml(
                dev_cfg,
                "system:\n  log_level: DEBUG\n  max_steps: 500\nscheduler:\n  algorithm: PPO\n",
            )
            # 切换到 tmp 目录调用 load_settings
            saved_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                s = load_settings(env_path=os.path.join(tmp, "no.env"))
                self.assertEqual(s.log_level, "DEBUG")
                self.assertEqual(s.max_steps, 500)
                self.assertEqual(s.algorithm, "PPO")
            finally:
                os.chdir(saved_cwd)

    def test_app_env_prod_loads_prod_config(self):
        """APP_ENV=prod 应加载 config.prod.yaml。"""
        os.environ["APP_ENV"] = "prod"
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = os.path.join(tmp, "config")
            os.makedirs(config_dir, exist_ok=True)
            prod_cfg = os.path.join(config_dir, "config.prod.yaml")
            _write_yaml(
                prod_cfg,
                "system:\n  log_level: INFO\n  max_steps: 2000\nscheduler:\n  algorithm: PPO\n",
            )
            saved_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                s = load_settings(env_path=os.path.join(tmp, "no.env"))
                self.assertEqual(s.log_level, "INFO")
                self.assertEqual(s.max_steps, 2000)
            finally:
                os.chdir(saved_cwd)

    def test_app_env_unset_falls_back_to_default_config(self):
        """APP_ENV 未设置时应回退到 config/config.yaml。"""
        # APP_ENV 已在 setUp 中清空
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = os.path.join(tmp, "config")
            os.makedirs(config_dir, exist_ok=True)
            default_cfg = os.path.join(config_dir, "config.yaml")
            _write_yaml(
                default_cfg,
                "system:\n  log_level: WARNING\n  max_steps: 999\n",
            )
            saved_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                s = load_settings(env_path=os.path.join(tmp, "no.env"))
                self.assertEqual(s.log_level, "WARNING")
                self.assertEqual(s.max_steps, 999)
            finally:
                os.chdir(saved_cwd)

    def test_app_env_case_insensitive(self):
        """APP_ENV 大写也应能正确加载（DEV/PROD）。"""
        os.environ["APP_ENV"] = "DEV"
        with tempfile.TemporaryDirectory() as tmp:
            config_dir = os.path.join(tmp, "config")
            os.makedirs(config_dir, exist_ok=True)
            dev_cfg = os.path.join(config_dir, "config.dev.yaml")
            _write_yaml(
                dev_cfg,
                "system:\n  log_level: DEBUG\n",
            )
            saved_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                s = load_settings(env_path=os.path.join(tmp, "no.env"))
                self.assertEqual(s.log_level, "DEBUG")
            finally:
                os.chdir(saved_cwd)

    def test_explicit_config_path_overrides_app_env(self):
        """显式传 config_path 时应忽略 APP_ENV。"""
        os.environ["APP_ENV"] = "dev"
        with tempfile.TemporaryDirectory() as tmp:
            # 在 tmp 下创建 dev.yaml（不该被加载）
            config_dir = os.path.join(tmp, "config")
            os.makedirs(config_dir, exist_ok=True)
            dev_cfg = os.path.join(config_dir, "config.dev.yaml")
            _write_yaml(dev_cfg, "system:\n  log_level: DEBUG\n")
            # 显式指定的配置文件
            explicit_cfg = os.path.join(tmp, "explicit.yaml")
            _write_yaml(explicit_cfg, "system:\n  log_level: ERROR\n")
            s = load_settings(
                config_path=explicit_cfg,
                env_path=os.path.join(tmp, "no.env"),
            )
            # 应使用 explicit_cfg，而非 dev.yaml
            self.assertEqual(s.log_level, "ERROR")

    def test_app_env_with_missing_config_falls_back_to_defaults(self):
        """APP_ENV 设置但对应配置文件不存在时应回退到默认值。"""
        os.environ["APP_ENV"] = "staging"
        with tempfile.TemporaryDirectory() as tmp:
            saved_cwd = os.getcwd()
            try:
                os.chdir(tmp)
                # tmp/config/config.staging.yaml 不存在
                s = load_settings(env_path=os.path.join(tmp, "no.env"))
                # 应回退到默认值
                self.assertEqual(s.max_qubits, 287)
                self.assertEqual(s.algorithm, "DQN")
            finally:
                os.chdir(saved_cwd)


if __name__ == "__main__":
    unittest.main()
