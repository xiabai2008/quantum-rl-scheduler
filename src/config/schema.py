"""
配置 Schema 验证模块（Pydantic）
Configuration Schema Validation with Pydantic

提供完整的 config.yaml 结构校验，启动时即捕获配置错误。
所有字段均有默认值——仅需校验时传入已加载的配置字典即可。

用法:
    from src.config.schema import AppConfig, validate_and_print

    # 校验并打印摘要
    cfg = validate_and_print(config_dict)

    # 或直接实例化
    app = AppConfig(**config_dict)
"""

from __future__ import annotations

from typing import Any, Literal

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator

# =============================================================================
# 子配置模型
# =============================================================================


class TianyanConfig(BaseModel):
    """天衍云平台 API 配置。"""

    api_key: str = Field(default="", description="天衍云 API Key")
    api_secret: str = Field(default="", description="天衍云 API Secret")
    base_url: str = Field(
        default="https://api.tianyanyun.cn/v1",
        description="天衍云 API 基础地址",
    )
    mock_delay: float = Field(default=90.0, ge=0, description="Mock 模式模拟延迟（秒）")
    mock_delay_note: str = Field(default="", description="Mock 延迟备注说明")
    mock_failure_rate: float = Field(default=0.0, ge=0, le=1, description="Mock 模式失败率")
    mock_machine_delays: dict[str, float] = Field(
        default_factory=dict, description="Mock 各机器延迟"
    )
    mock_mode: bool = Field(default=True, description="Mock 模式开关")
    timeout: int = Field(default=30, ge=1, le=300, description="API 超时（秒）")


class SchedulerConfig(BaseModel):
    """调度引擎配置。"""

    algorithm: Literal["DQN", "PPO", "MAPPO"] = Field(
        default="PPO", description="RL 算法"
    )  # Issue #687: v9 主线为 PPO
    batch_size: int = Field(default=64, ge=1, description="训练批量大小")
    epsilon_decay: float = Field(default=0.995, gt=0, le=1, description="Epsilon 衰减系数")
    epsilon_end: float = Field(default=0.01, ge=0, le=1, description="Epsilon 终值")
    epsilon_start: float = Field(default=1.0, ge=0, le=1, description="Epsilon 初值")
    gamma: float = Field(default=0.99, ge=0, le=1, description="折扣因子")
    learning_rate: float = Field(default=3e-4, gt=0, lt=1, description="学习率")
    replay_buffer_size: int = Field(default=10000, ge=1, description="回放缓冲区大小")


class QuantumConfig(BaseModel):
    """量子计算配置。"""

    backend: str = Field(default="tianyan-287", description="量子后端名称")
    max_qubits: int = Field(default=287, ge=1, description="最大量子比特数")
    shots: int = Field(default=1024, ge=1, description="测量次数")
    simulator: str = Field(default="qiskit-aer", description="模拟器")
    # Issue #243: 真机提交间隔步数（每N步强制提交一次，保底机制）
    real_submit_interval: int = Field(
        default=20, ge=1, description="真机提交间隔步数（保底机制，与概率触发共存）"
    )


class AnnealingConfig(BaseModel):
    """量子退火配置。

    对应代码：src/quantum/annealing.py 的 QuantumAnnealingOptimizer。
    所有参数均可通过环境变量覆盖（见 .env.example 的 ANNEALING_* 段）。
    """

    # ── 全局开关 ──
    # 退火已降级为探索性功能（2026-07-27），默认关闭
    enabled: bool = Field(default=False, description="退火开关（探索性功能，默认关闭）")

    # ── 退火器基础参数（QuantumAnnealingOptimizer.__init__） ──
    simulation_mode: bool = Field(default=True, description="仿真模式（true=纯仿真 numpy/neal）")
    num_qubits: int = Field(default=16, ge=1, le=256, description="退火量子比特数，建议 ≥16")
    shots: int = Field(default=1000, ge=1, le=10000, description="退火采样次数")
    annealing_time: float = Field(
        default=20.0, gt=0, le=1000.0, description="退火时间（微秒），仅真机模式生效"
    )
    n_bits_per_weight: int = Field(
        default=4,
        ge=2,
        description="每个权重的编码位数（1 个符号位 + 数值位）",
    )

    # ── 仿真模拟退火超参数（仅 simulation_mode=true 时生效） ──
    sim_initial_temp: float = Field(default=2.0, ge=0.1, le=10.0, description="初始温度")
    sim_cooling_rate: float = Field(default=0.995, gt=0.0, lt=1.0, description="降温系数")
    sim_num_sweeps: int = Field(default=200, ge=10, le=10000, description="扫描次数")
    sim_patience: int = Field(
        default=20, ge=1, le=500, description="早停耐心值——连续 N 次扫描无改进则终止"
    )

    # ── QUBO 构造参数（network_to_qubo） ──
    reg_lambda: float = Field(default=0.1, gt=0, le=1.0, description="L2 正则化系数")
    max_delta_ratio: float = Field(default=0.1, gt=0.0, le=1.0, description="权重更新幅度比例")
    accept_threshold_ratio: float = Field(default=0.01, ge=0.0, le=0.1, description="接受阈值比例")

    # ── 分层退火参数（optimize_policy_hierarchical，对应 Issue #148） ──
    head_only: bool = Field(default=True, description="仅优化尾部参数张量（true=head_only 模式）")
    max_params_per_block: int = Field(
        default=200, ge=10, le=10000, description="分层退火每块最大参数数"
    )
    block_strategy: str = Field(
        default="tensor_wise", description="分块策略: tensor_wise / size_limited"
    )


class CacheConfig(BaseModel):
    """缓存配置。"""

    db: int = Field(default=0, ge=0, description="Redis DB 编号")
    host: str = Field(default="localhost", description="缓存主机")
    port: int = Field(default=6379, ge=1, le=65535, description="缓存端口")
    type: str = Field(default="redis", description="缓存类型")


class ClassicalConfig(BaseModel):
    """经典计算资源配置。"""

    max_cpu_utilization: float = Field(default=0.95, ge=0, le=1, description="最大 CPU 利用率")
    memory_limit: int = Field(default=16384, ge=1, description="内存限制（MB）")


class DatabaseConfig(BaseModel):
    """数据库配置。"""

    path: str = Field(default="data/scheduler.db", description="数据库文件路径")
    type: str = Field(default="sqlite", description="数据库类型")


class SystemConfig(BaseModel):
    """系统配置。"""

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        default="INFO", description="日志级别"
    )
    max_queue_size: int = Field(default=100, ge=1, description="最大队列长度")
    max_steps: int = Field(default=1000, ge=1, description="最大步数")
    max_wait_time: int = Field(default=3600, ge=0, description="最大等待时间（秒）")


class WebConfig(BaseModel):
    """Web 服务配置。"""

    debug: bool = Field(default=False, description="调试模式")
    host: str = Field(default="0.0.0.0", description="绑定地址")  # nosec B104: demo/dev
    port: int = Field(default=8000, ge=1, le=65535, description="绑定端口")


# =============================================================================
# 顶层配置模型
# =============================================================================


class AppConfig(BaseModel):
    """应用顶层配置，包含所有子系统配置。"""

    model_config = ConfigDict(extra="forbid")

    tianyan: TianyanConfig = Field(default_factory=TianyanConfig)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    quantum: QuantumConfig = Field(default_factory=QuantumConfig)
    annealing: AnnealingConfig = Field(default_factory=AnnealingConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)
    classical: ClassicalConfig = Field(default_factory=ClassicalConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    web: WebConfig = Field(default_factory=WebConfig)

    @field_validator("tianyan", mode="after")
    @classmethod
    def _warn_mock_api_keys(cls, v: TianyanConfig) -> TianyanConfig:
        """Mock 模式下空 API key 不报错，仅给出提示。"""
        return v


# =============================================================================
# 校验辅助函数
# =============================================================================


def validate_config(data: dict[str, Any]) -> AppConfig:
    """
    校验配置字典，返回 AppConfig 实例。

    Args:
        data: 原始配置字典（已展开环境变量的 load_config 返回值）

    Returns:
        通过校验的 AppConfig 实例

    Raises:
        pydantic.ValidationError: 配置不符合 Schema 时抛出，包含详细错误信息
    """
    app = AppConfig(**data)
    return app


def validate_and_print(data: dict[str, Any]) -> AppConfig:
    """
    校验配置并打印关键信息摘要（用于启动日志）。

    Args:
        data: 原始配置字典

    Returns:
        通过校验的 AppConfig 实例
    """
    try:
        app = AppConfig(**data)
    except (ValueError, TypeError) as e:
        # Pydantic ValidationError 为 ValueError 子类；TypeError 捕获字段类型不匹配
        logger.error(f"配置校验失败 — {e}")
        raise

    logger.debug(
        "配置校验通过 ｜ 算法=%s 退火=%s Mock=%s Backend=%s",
        app.scheduler.algorithm,
        "启用" if app.annealing.enabled else "禁用",
        "启用" if app.tianyan.mock_mode else "真机",
        app.quantum.backend,
    )
    return app
