"""
统一配置加载模块
Unified Configuration Loader

实现 config.yaml + .env + 环境变量三层合一的配置管理：
- 优先级：环境变量（os.environ）> .env 文件 > config.yaml 默认值
- 支持 ${VAR} 变量引用（复用 helpers._expand_env_vars）
- 支持类型转换（int / float / bool / str）
- 不引入新依赖：使用 dataclass + 手动解析 .env

用法::

    from src.config.settings import Settings, load_settings

    settings = load_settings()
    print(settings.api_key, settings.max_qubits)

    # 或便捷创建
    settings = Settings.from_env()
"""

from __future__ import annotations

import os
from dataclasses import dataclass, fields
from typing import Any, cast

import yaml

from src.utils.helpers import _expand_env_vars

# =============================================================================
# 字段元数据：每个 Settings 字段对应的 config.yaml 路径与环境变量名
# =============================================================================
# 格式：{field_name: (config_yaml_path, env_var_name)}
# config_yaml_path 使用点号分隔，如 "tianyan.api_key"；None 表示无对应 yaml 键
# env_var_name 为大写环境变量名；None 表示无对应环境变量
_FIELD_MAP: dict[str, tuple[str | None, str | None]] = {
    # ── API 配置 ──
    "api_key": ("tianyan.api_key", "TIANYAN_API_KEY"),
    "api_token": (None, "TIANYAN_API_TOKEN"),
    "api_timeout": ("tianyan.timeout", "TIANYAN_API_TIMEOUT"),
    "api_retries": (None, "TIANYAN_API_MAX_RETRIES"),
    # ── 调度器配置 ──
    "max_qubits": ("quantum.max_qubits", "QUANTUM_MAX_QUBITS"),
    "max_steps": ("system.max_steps", "SCHEDULER_MAX_STEPS"),
    "algorithm": ("scheduler.algorithm", "SCHEDULER_ALGORITHM"),
    # ── 量子配置 ──
    "annealing_enabled": ("annealing.enabled", "ANNEALING_ENABLED"),
    "quantum_shots": ("quantum.shots", "QUANTUM_SHOTS"),
    # ── 日志配置 ──
    "log_level": ("system.log_level", "LOG_LEVEL"),
    "log_format": (None, "LOG_FORMAT"),
    "log_dir": (None, "LOG_DIR"),
    # ── 可视化配置 ──
    "viz_api_key": (None, "VIZ_API_KEY"),
    "viz_port": ("web.port", "WEB_PORT"),
}

# 类型名→内建类型映射（用于解析 dataclass 字段的字符串类型注解）
_TYPE_MAP: dict[str, type] = {
    "str": str,
    "int": int,
    "float": float,
    "bool": bool,
}


# =============================================================================
# 类型转换工具
# =============================================================================


def _to_bool(value: Any) -> bool:
    """将字符串/整数转换为布尔值。

    支持的 truthy 取值：``"true"``、``"1"``、``"yes"``、``"on"``（大小写不敏感）
    支持的 falsy 取值：``"false"``、``"0"``、``"no"``、``"off"``、空串（大小写不敏感）

    Args:
        value: 原始值（str / int / bool）

    Returns:
        转换后的布尔值；无法识别的值按 ``bool(value)`` 兜底
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in ("true", "1", "yes", "on"):
            return True
        if normalized in ("false", "0", "no", "off", ""):
            return False
    return bool(value)


def _convert(value: Any, target_type: type) -> Any:
    """根据目标类型转换原始值。

    Args:
        value: 原始值（通常为字符串，也可能是 yaml 解析出的原生类型）
        target_type: 目标 Python 类型（int / float / bool / str）

    Returns:
        转换后的值；目标类型未知时原样返回

    Raises:
        ValueError: int / float 转换失败
        TypeError: 转换失败
    """
    # 已是目标类型直接返回，避免重复转换
    if target_type is bool:
        return _to_bool(value)
    if target_type is int:
        if isinstance(value, bool):
            return int(value)
        return int(value)
    if target_type is float:
        return float(value)
    return str(value)


def _resolve_type(type_hint: Any) -> type:
    """将类型注解解析为实际类型对象。

    在 ``from __future__ import annotations`` 下，dataclass 字段的 ``type``
    为字符串形式；本函数将其映射回内建类型。

    Args:
        type_hint: 字段类型注解（可能是字符串或类型对象）

    Returns:
        对应的 Python 类型对象；无法识别时返回 ``str``
    """
    if isinstance(type_hint, type):
        return type_hint
    if isinstance(type_hint, str):
        base = type_hint.strip()
        # 简单场景：直接查表
        if base in _TYPE_MAP:
            return _TYPE_MAP[base]
        # 处理 Optional[X] / X | None 等复合注解
        for name, tp in _TYPE_MAP.items():
            if name in base:
                return tp
    return str


# =============================================================================
# .env 文件解析（手动实现，不依赖 python-dotenv）
# =============================================================================


def _parse_env_file(path: str) -> dict[str, str]:
    """手动解析 .env 文件，返回键值对字典。

    支持特性：
    - ``KEY=VALUE`` 格式
    - 以 ``#`` 开头的注释行
    - 空行跳过
    - 值两端的引号（单/双）剥离
    - 等号两侧空格剥离
    - **不修改** ``os.environ``

    Args:
        path: .env 文件路径

    Returns:
        键值对字典；文件不存在或读取失败时返回空字典
    """
    result: dict[str, str] = {}
    if not os.path.isfile(path):
        return result
    try:
        with open(path, encoding="utf-8") as env_file:
            for raw_line in env_file:
                line = raw_line.strip()
                # 跳过空行与注释
                if not line or line.startswith("#"):
                    continue
                # 必须包含等号
                if "=" not in line:
                    continue
                key, _, value = line.partition("=")
                key = key.strip()
                if not key:
                    continue
                # 剥离值两端的引号
                value = value.strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                    value = value[1:-1]
                result[key] = value
    except OSError:
        # 读取失败返回空字典（不破坏调用方流程）
        return result
    return result


# =============================================================================
# Settings 数据类
# =============================================================================


@dataclass
class Settings:
    """统一配置数据类。

    字段覆盖五个维度：API、调度器、量子、日志、可视化。
    三层优先级：环境变量 > .env 文件 > config.yaml 默认值。

    所有字段均有默认值，确保 ``Settings()`` 总是可用。
    """

    # ── API 配置 ──
    api_key: str = ""
    api_token: str = ""
    api_timeout: float = 30.0
    api_retries: int = 3
    # ── 调度器配置 ──
    max_qubits: int = 287
    max_steps: int = 1000
    algorithm: str = "DQN"
    # ── 量子配置 ──
    annealing_enabled: bool = True
    quantum_shots: int = 1024
    # ── 日志配置 ──
    log_level: str = "INFO"
    log_format: str = "text"
    log_dir: str = "logs"
    # ── 可视化配置 ──
    viz_api_key: str = ""
    viz_port: int = 8000

    @classmethod
    def from_env(cls) -> Settings:
        """便捷创建：使用默认路径加载配置。

        等价于 ``load_settings(config_path="config/config.yaml", env_path=".env")``。

        Returns:
            Settings 实例
        """
        return load_settings()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典（便于日志输出与序列化）。

        Returns:
            字段名→值的字典
        """
        return {f.name: getattr(self, f.name) for f in fields(self)}


# =============================================================================
# 辅助：按点号路径从嵌套字典取值
# =============================================================================


def _get_nested(data: dict[str, Any], path: str) -> Any:
    """按点号路径从嵌套字典取值。

    Args:
        data: 嵌套字典
        path: 点号分隔路径，如 ``"tianyan.api_key"``

    Returns:
        找到的值；任一层缺失返回 ``None``
    """
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
        if current is None:
            return None
    return current


# =============================================================================
# 主加载函数
# =============================================================================


def load_settings(
    config_path: str = "config/config.yaml",
    env_path: str = ".env",
) -> Settings:
    """加载统一配置，按三层优先级合并。

    优先级（高→低）::

        1. 环境变量（os.environ）
        2. .env 文件
        3. config.yaml 默认值

    Args:
        config_path: config.yaml 路径；文件不存在时回退到字段默认值
        env_path: .env 文件路径；文件不存在时跳过

    Returns:
        Settings 实例
    """
    # ── 第 1 层：config.yaml 基础值 ──
    yaml_data: dict[str, Any] = {}
    if os.path.isfile(config_path):
        try:
            with open(config_path, encoding="utf-8") as yaml_file:
                loaded = yaml.safe_load(yaml_file)
            if isinstance(loaded, dict):
                # 展开 ${VAR} 引用（使用 os.environ）
                yaml_data = cast(dict[str, Any], _expand_env_vars(loaded))
        except (yaml.YAMLError, OSError):
            yaml_data = {}

    # ── 第 2 层：.env 文件 ──
    env_file_data = _parse_env_file(env_path)
    # 展开 .env 值中的 ${VAR} 引用（使用 os.environ）
    env_file_data = {
        k: cast(str, _expand_env_vars(v)) for k, v in env_file_data.items()
    }

    # ── 合并：按字段逐个解析 ──
    defaults = Settings()
    field_values: dict[str, Any] = {}
    for f in fields(Settings):
        fname = f.name
        default_val = getattr(defaults, fname)
        yaml_path, env_var = _FIELD_MAP.get(fname, (None, None))

        # 1) 起始值：dataclass 默认值
        value: Any = default_val

        # 2) config.yaml 覆盖
        if yaml_path is not None:
            yaml_value = _get_nested(yaml_data, yaml_path)
            if yaml_value is not None:
                value = yaml_value

        # 3) .env 文件覆盖
        if env_var is not None and env_var in env_file_data:
            value = env_file_data[env_var]

        # 4) 环境变量覆盖（最高优先级）
        if env_var is not None and env_var in os.environ:
            value = os.environ[env_var]

        # 类型转换（失败时回退到默认值，保证健壮性）
        target_type = _resolve_type(f.type)
        try:
            field_values[fname] = _convert(value, target_type)
        except (ValueError, TypeError):
            field_values[fname] = default_val

    return Settings(**field_values)
