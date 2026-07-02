"""
工具函数模块
Utility Functions Module

提供通用的工具函数，包括：
- 日志配置
- 数据预处理
- 性能评估
- 配置加载
"""

import json
import os
import re
import sys
from datetime import datetime
from typing import Any, cast

import numpy as np
import yaml
from loguru import logger

# ---------------------------------------------------------------------------
# 环境变量展开
# ---------------------------------------------------------------------------
_UNRESOLVED_VAR_PATTERN = re.compile(r"\$\{[^}]+\}")


def _expand_env_vars(value: Any) -> Any:
    """
    递归展开字典/列表/字符串中的 ${VAR} 环境变量引用。

    对字符串：调用 os.path.expandvars() 展开 ${VAR}。
    对字典/列表：递归处理每个值。
    其他类型直接返回。

    Args:
        value: 待展开的值（dict / list / str / Any）

    Returns:
        展开后的值（类型与输入对应）
    """
    if isinstance(value, str):
        expanded = os.path.expandvars(value)
        return expanded
    elif isinstance(value, dict):
        return {k: _expand_env_vars(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    else:
        return value


def _warn_unresolved(data: dict[str, Any], source_path: str = "") -> None:
    """
    检查配置字典中是否存在未展开的 ${VAR} 引用并发出警告。

    Args:
        data: 配置字典
        source_path: 配置来源标识（用于日志）
    """
    unresolved: list[str] = []

    def _walk(prefix: str, value: Any) -> None:
        if isinstance(value, str) and _UNRESOLVED_VAR_PATTERN.search(value):
            unresolved.append(f"{prefix} = {value!r}")
        elif isinstance(value, dict):
            for k, v in value.items():
                _walk(f"{prefix}.{k}" if prefix else str(k), v)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                _walk(f"{prefix}[{i}]", v)

    for key, val in data.items():
        _walk(key, val)

    if unresolved:
        logger.warning(f"配置文件 {source_path!r} 中存在 {len(unresolved)} 个未展开的环境变量引用:")
        for entry in unresolved:
            logger.warning(f"  {entry}")


# 日志配置
def _json_serializer(record: dict[str, Any]) -> str:
    """将 loguru 日志记录序列化为 JSON 格式。

    输出字段：timestamp（ISO 格式）、level、module、function、line、message，
    以及 record["extra"] 中的所有自定义字段（不覆盖内置字段）。

    Args:
        record: loguru 日志记录字典

    Returns:
        JSON 格式的字符串（UTF-8，保留中文，非可序列化值用 str 兜底）
    """
    subset: dict[str, Any] = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "module": record["module"],
        "function": record["function"],
        "line": record["line"],
        "message": record["message"],
    }
    # 合并 extra 上下文（不覆盖内置字段）
    extra = record.get("extra")
    if extra:
        for key, value in extra.items():
            if key not in subset:
                subset[key] = value
    return json.dumps(subset, ensure_ascii=False, default=str)


def _json_console_sink(message: Any) -> None:
    """loguru JSON 控制台 sink，将日志记录序列化为 JSON 并输出到 stdout。"""
    sys.stdout.write(_json_serializer(message.record) + "\n")


class _JsonFileSink:
    """JSON 文件 sink，将日志以 JSON Lines 格式写入文件。

    适用于 ELK/Loki/Graylog 等日志采集系统解析。
    注意：不支持 loguru 内建的 rotation/retention，如需轮转请依赖外部工具（如 logrotate）。
    """

    def __init__(self, filepath: str) -> None:
        # 文件句柄需在 sink 生命周期内保持打开，无法使用 with 语句
        self._file = open(filepath, "a", encoding="utf-8")  # noqa: SIM115

    def write(self, message: Any) -> None:
        self._file.write(_json_serializer(message.record) + "\n")
        self._file.flush()

    def stop(self) -> None:
        self._file.close()


def setup_logging(
    log_dir: str = "logs",
    log_level: str = "INFO",
    log_file: str = "scheduler.log",
) -> Any:
    """配置日志系统，支持文本和 JSON 两种格式。

    通过环境变量 LOG_FORMAT 控制输出格式：
    - "json": 结构化 JSON 输出（适合 ELK/Loki/Graylog 采集）
    - "text"（默认）: 人类可读的文本格式

    Args:
        log_dir: 日志目录
        log_level: 日志级别
        log_file: 日志文件名

    Returns:
        配置后的 loguru logger
    """
    os.makedirs(log_dir, exist_ok=True)

    logger.remove()  # 移除默认处理器

    log_format = os.getenv("LOG_FORMAT", "text").lower()
    use_json = log_format == "json"

    if use_json:
        # JSON 格式：结构化输出，便于 ELK/Loki/Graylog 采集
        logger.add(sink=_json_console_sink, level=log_level)
        logger.add(
            sink=_JsonFileSink(os.path.join(log_dir, log_file)),
            level=log_level,
        )
    else:
        # 文本格式：人类可读（使用 sys.stderr 避免 print）
        logger.add(
            sink=sys.stderr,
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}",
        )
        logger.add(
            sink=os.path.join(log_dir, log_file),
            rotation="100 MB",
            retention="30 days",
            level=log_level,
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {file}:{function}:{line} | {message}",
        )

    return logger


# 配置加载
def load_config(
    config_path: str = "config/config.yaml",
    validate: bool = False,
) -> dict[str, Any]:
    """
    加载配置文件，自动展开 ${VAR} 环境变量引用。

    Args:
        config_path: 配置文件路径
        validate: 是否启用 Pydantic Schema 校验（默认 False）

    Returns:
        配置字典（环境变量已展开）
    """
    try:
        with open(config_path, encoding="utf-8") as f:
            config = yaml.safe_load(f)

        # 递归展开环境变量引用
        expanded = _expand_env_vars(config)

        # 对字典根元素检查是否残留未展开的 ${}
        if isinstance(expanded, dict):
            _warn_unresolved(expanded, source_path=config_path)

        # 可选：Pydantic Schema 校验
        if validate and isinstance(expanded, dict):
            try:
                from src.config.schema import validate_and_print

                validate_and_print(expanded)
            except ImportError:
                logger.debug("跳过 Pydantic 校验（schema 模块不可用）")

        logger.info(f"配置文件加载成功：{config_path}")
        return cast(dict[str, Any], expanded)
    except (yaml.YAMLError, OSError, ValueError) as e:
        # YAML 解析错误 / 文件 I/O 错误 / Pydantic 校验错误（ValidationError 为 ValueError 子类）
        logger.error(f"配置文件加载失败：{e}")
        return {}


def save_config(config: dict[str, Any], config_path: str = "config/config.yaml") -> None:
    """
    保存配置文件

    Args:
        config: 配置字典
        config_path: 配置文件路径
    """
    try:
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, "w", encoding="utf-8") as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
        logger.info(f"配置文件保存成功：{config_path}")
    except (OSError, yaml.YAMLError) as e:
        # 文件 I/O 错误 / YAML 序列化错误
        logger.error(f"配置文件保存失败：{e}")


# 数据预处理
def normalize_vector(
    vector: list[float], min_val: float = 0.0, max_val: float = 1.0
) -> list[float]:
    """
    归一化向量

    Args:
        vector: 输入向量
        min_val: 最小值
        max_val: 最大值

    Returns:
        归一化后的向量
    """
    if len(vector) == 0:
        return []

    vec_array = np.array(vector)
    min_v = np.min(vec_array)
    max_v = np.max(vec_array)

    if max_v - min_v < 1e-10:
        return [0.5] * len(vector)

    normalized = (vec_array - min_v) / (max_v - min_v)
    normalized = normalized * (max_val - min_val) + min_val

    return cast(list[float], normalized.tolist())


def one_hot_encode(category: str, categories: list[str]) -> list[int]:
    """
    独热编码

    Args:
        category: 类别
        categories: 所有类别列表

    Returns:
        独热编码向量
    """
    encoding = [0] * len(categories)
    if category in categories:
        idx = categories.index(category)
        encoding[idx] = 1
    return encoding


# 性能评估
def calculate_completion_rate(completed: int, total: int) -> float:
    """计算完成率"""
    if total == 0:
        return 0.0
    return completed / total


def calculate_average_wait_time(wait_times: list[float]) -> float:
    """计算平均等待时间"""
    if len(wait_times) == 0:
        return 0.0
    return float(np.mean(wait_times))


def calculate_resource_utilization(
    used: float,
    total: float,
) -> float:
    """计算资源利用率"""
    if total == 0:
        return 0.0
    return used / total


# 时间工具
def format_time(seconds: float) -> str:
    """
    格式化时间

    Args:
        seconds: 秒数

    Returns:
        格式化后的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"


def get_current_timestamp() -> str:
    """获取当前时间戳"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# 数据保存/加载
def save_json(data: Any, filepath: str) -> None:
    """
    保存为JSON文件

    Args:
        data: 数据
        filepath: 文件路径
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"JSON文件保存成功：{filepath}")


def load_json(filepath: str) -> Any:
    """
    加载JSON文件

    Args:
        filepath: 文件路径

    Returns:
        加载的数据
    """
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"JSON文件加载成功：{filepath}")
        return data
    except (json.JSONDecodeError, OSError) as e:
        # JSON 解析错误 / 文件 I/O 错误
        logger.error(f"JSON文件加载失败：{e}")
        return None


# 评估指标
class MetricsCalculator:
    """评估指标计算器"""

    @staticmethod
    def calculate_reward(
        completion_rate: float,
        avg_wait_time: float,
        resource_utilization: float,
        max_wait_time: float = 3600.0,
    ) -> float:
        """
        计算综合奖励

        Args:
            completion_rate: 完成率
            avg_wait_time: 平均等待时间
            resource_utilization: 资源利用率
            max_wait_time: 最大等待时间（用于归一化）

        Returns:
            综合奖励值
        """
        # 归一化等待时间（越小越好）
        normalized_wait = 1.0 - min(avg_wait_time / max_wait_time, 1.0)

        # 加权综合
        reward = 0.4 * completion_rate + 0.3 * normalized_wait + 0.3 * resource_utilization

        return reward

    @staticmethod
    def calculate_improvement(
        new_value: float,
        baseline_value: float,
    ) -> float:
        """
        计算改进百分比

        Args:
            new_value: 新值
            baseline_value: 基线值

        Returns:
            改进百分比（%）
        """
        if baseline_value == 0:
            return 0.0 if new_value == 0 else 100.0

        improvement = (new_value - baseline_value) / abs(baseline_value) * 100
        return improvement


if __name__ == "__main__":
    # 测试代码
    logger.info("工具函数模块测试")

    # 测试归一化
    vector = [1.0, 2.0, 3.0, 4.0, 5.0]
    normalized = normalize_vector(vector)
    print(f"归一化结果：{normalized}")

    # 测试独热编码
    categories = ["quantum", "classical", "hybrid"]
    encoding = one_hot_encode("quantum", categories)
    print(f"独热编码：{encoding}")

    # 测试评估指标
    reward = MetricsCalculator.calculate_reward(
        completion_rate=0.85,
        avg_wait_time=120.0,
        resource_utilization=0.75,
    )
    print(f"综合奖励：{reward:.3f}")
