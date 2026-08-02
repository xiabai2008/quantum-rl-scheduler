"""PPO 训练过程可视化日志模块。

提供训练指标记录与 TensorBoard 可视化集成：
- TrainingMetricsLogger: 标量/直方图/文本/超参数/Episode 级指标记录器
- TensorboardCallback: 兼容 Stable-Baselines3 的训练回调
- create_training_logger: 工厂函数，一次性创建 logger + callback

降级策略：tensorboard 不可用时自动切换到 JSONL 文件记录。
"""

import json
import os
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from loguru import logger
from stable_baselines3.common.callbacks import BaseCallback

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

try:
    from torch.utils.tensorboard import SummaryWriter

    _TENSORBOARD_AVAILABLE = True
except ImportError:  # pragma: no cover
    _TENSORBOARD_AVAILABLE = False

__all__ = [
    "TensorboardCallback",
    "TrainingMetricsLogger",
    "create_training_logger",
]


class TrainingMetricsLogger:
    """训练指标记录器，支持 TensorBoard 与 JSONL 文件双后端。

    TensorBoard 不可用时自动降级到 JSONL 文件。

    Attributes:
        log_dir         : 日志根目录
        experiment_name : 实验名称
        use_tensorboard : 是否启用 TensorBoard 后端
    """

    def __init__(
        self,
        log_dir: str = "logs/tensorboard",
        experiment_name: str = "ppo_training",
    ) -> None:
        """初始化训练指标记录器。

        Args:
            log_dir        : 日志根目录
            experiment_name: 实验名称
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self._records: dict[str, list[dict[str, Any]]] = {
            "scalars": [],
            "histograms": [],
            "texts": [],
            "hyperparams": [],
            "episodes": [],
        }
        self._closed = False
        self._jsonl_path = os.path.join(log_dir, f"{experiment_name}.jsonl")
        # Issue #888: JSONL 文件句柄缓存，避免高频记录时反复 open/close
        self._jsonl_file: Any = None

        if _TENSORBOARD_AVAILABLE:
            self.use_tensorboard = True
            tb_dir = os.path.join(log_dir, experiment_name)
            os.makedirs(tb_dir, exist_ok=True)
            self._writer: Any = SummaryWriter(log_dir=tb_dir)
            logger.info(f"[TrainingMetricsLogger] TensorBoard 后端已启用，日志目录: {tb_dir}")
        else:
            self.use_tensorboard = False
            self._writer = None
            os.makedirs(log_dir, exist_ok=True)
            logger.warning(
                f"[TrainingMetricsLogger] tensorboard 未安装，降级到 JSONL 文件: {self._jsonl_path}"
            )

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """记录标量指标。

        Args:
            tag  : 指标名称，如 ``train/reward``
            value: 指标值
            step : 训练步数
        """
        record = {"tag": tag, "value": float(value), "step": int(step)}
        self._records["scalars"].append(record)
        self._trim_records()
        if self._writer is not None:
            self._writer.add_scalar(tag, float(value), int(step))
        else:
            self._append_jsonl({"type": "scalar", **record})

    def log_histogram(self, tag: str, values: np.ndarray, step: int) -> None:
        """记录直方图（如奖励分布、Q 值分布）。

        Args:
            tag    : 直方图名称
            values : 数值数组
            step   : 训练步数
        """
        values_arr = np.asarray(values, dtype=np.float64)
        has_data = values_arr.size > 0
        record = {
            "tag": tag,
            "step": int(step),
            "count": int(values_arr.size),
            "mean": float(values_arr.mean()) if has_data else 0.0,
            "std": float(values_arr.std()) if has_data else 0.0,
            "min": float(values_arr.min()) if has_data else 0.0,
            "max": float(values_arr.max()) if has_data else 0.0,
        }
        self._records["histograms"].append(record)
        self._trim_records()
        if self._writer is not None:
            if has_data:
                self._writer.add_histogram(tag, values_arr, int(step))
        else:
            self._append_jsonl(
                {
                    "type": "histogram",
                    "tag": tag,
                    "step": int(step),
                    "count": int(values_arr.size),
                    "mean": record["mean"],
                    "std": record["std"],
                    "min": record["min"],
                    "max": record["max"],
                    "values": values_arr.tolist(),
                }
            )

    def log_text(self, tag: str, text: str, step: int) -> None:
        """记录文本（如配置信息、超参数描述）。

        Args:
            tag  : 文本名称
            text : 文本内容
            step : 训练步数
        """
        record = {"tag": tag, "text": text, "step": int(step)}
        self._records["texts"].append(record)
        self._trim_records()
        if self._writer is not None:
            self._writer.add_text(tag, text, int(step))
        else:
            self._append_jsonl({"type": "text", **record})

    def log_hyperparams(
        self,
        params: dict[str, Any],
        metrics: dict[str, float] | None = None,
    ) -> None:
        """记录超参数及对应指标。

        Args:
            params : 超参数字典
            metrics: 可选的指标字典（如最终 reward）
        """
        metrics = metrics or {}
        record = {"params": dict(params), "metrics": dict(metrics)}
        self._records["hyperparams"].append(record)
        self._trim_records()
        if self._writer is not None:
            flat_params = {
                k: (v if isinstance(v, int | float | str | bool) else str(v))
                for k, v in params.items()
            }
            safe_metrics = {k: float(v) for k, v in metrics.items()}
            try:
                self._writer.add_hparams(flat_params, safe_metrics)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    f"[TrainingMetricsLogger] add_hparams 失败: {type(exc).__name__}: {exc}"
                )
        else:
            self._append_jsonl({"type": "hyperparams", **record})

    def log_episode(
        self,
        episode: int,
        reward: float,
        length: int,
        info: dict | None = None,
    ) -> None:
        """记录 Episode 级指标。

        Args:
            episode: Episode 序号
            reward : Episode 累计奖励
            length : Episode 步数
            info   : 额外信息字典
        """
        record = {
            "episode": int(episode),
            "reward": float(reward),
            "length": int(length),
            "info": dict(info) if info else {},
        }
        self._records["episodes"].append(record)
        self._trim_records()
        if self._writer is not None:
            self._writer.add_scalar("episode/reward", float(reward), int(episode))
            self._writer.add_scalar("episode/length", int(length), int(episode))
        else:
            self._append_jsonl({"type": "episode", **record})

    # Issue #888: 各类记录的内存上限（超过时保留最近的一半）
    MAX_RECORDS: ClassVar[dict[str, int]] = {
        "scalars": 10000,
        "histograms": 1000,
        "texts": 500,
        "hyperparams": 500,
        "episodes": 5000,
    }

    def _trim_records(self) -> None:
        """统一裁剪各类内存记录，避免长训练 OOM（Issue #888/#670）。"""
        for key, limit in self.MAX_RECORDS.items():
            records = self._records.get(key)
            if records is None:
                continue
            if len(records) > limit:
                self._records[key] = records[-limit // 2 :]

    def flush(self) -> None:
        """刷新底层写入缓冲。"""
        if self._writer is not None:
            self._writer.flush()
        if self._jsonl_file is not None:
            self._jsonl_file.flush()

    def close(self) -> None:
        """关闭日志后端，释放资源。重复调用安全。"""
        if self._closed:
            return
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
        if self._jsonl_file is not None:
            self._jsonl_file.flush()
            self._jsonl_file.close()
            self._jsonl_file = None
        self._closed = True

    def get_summary(self) -> dict[str, list[dict]]:
        """返回所有已记录的指标摘要。

        JSONL 降级模式下从文件读取；TensorBoard 模式下返回内存缓存。

        Returns:
            按类型分组的指标记录字典
        """
        if not self.use_tensorboard and os.path.exists(self._jsonl_path):
            # 先 flush 缓存句柄，确保已写入内容可被读取
            if self._jsonl_file is not None:
                self._jsonl_file.flush()
            return self._read_summary_from_jsonl()
        return {
            "scalars": list(self._records["scalars"]),
            "histograms": list(self._records["histograms"]),
            "texts": list(self._records["texts"]),
            "hyperparams": list(self._records["hyperparams"]),
            "episodes": list(self._records["episodes"]),
        }

    def _append_jsonl(self, record: dict[str, Any]) -> None:
        """向 JSONL 降级文件追加一条记录（缓存文件句柄，避免反复 open/close）。"""
        if self._jsonl_file is None:
            os.makedirs(self.log_dir, exist_ok=True)
            # Issue #888: 有意缓存句柄避免高频 open/close（SIM115 豁免）
            self._jsonl_file = open(self._jsonl_path, "a", encoding="utf-8")  # noqa: SIM115
        self._jsonl_file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _read_summary_from_jsonl(self) -> dict[str, list[dict]]:
        """从 JSONL 降级文件解析所有记录并按类型分组。"""
        summary: dict[str, list[dict]] = {
            "scalars": [],
            "histograms": [],
            "texts": [],
            "hyperparams": [],
            "episodes": [],
        }
        type_map = {
            "scalar": "scalars",
            "histogram": "histograms",
            "text": "texts",
            "hyperparams": "hyperparams",
            "episode": "episodes",
        }
        with open(self._jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_type = record.pop("type", None)
                key = type_map.get(rec_type)
                if key:
                    summary[key].append(record)
        return summary


class TensorboardCallback(BaseCallback):
    """PPO 训练过程回调。

    兼容 Stable-Baselines3 回调协议，定期将训练过程指标写入 TrainingMetricsLogger。

    Attributes:
        metrics_logger: 训练指标记录器
        log_freq      : 标量指标记录频率（步数）
    """

    def __init__(
        self,
        logger: TrainingMetricsLogger,
        log_freq: int = 100,
        verbose: int = 0,
    ) -> None:
        """初始化训练回调。

        Args:
            logger  : TrainingMetricsLogger 实例
            log_freq: 标量指标记录频率，每 log_freq 步记录一次
            verbose : 日志详细程度
        """
        super().__init__(verbose)
        self.metrics_logger = logger
        self.log_freq = max(1, int(log_freq))
        self._last_episode_count = 0
        # Issue #888: 首次 Prometheus 指标采集失败后记录 warning 并置位，
        # 避免每个 step 重复输出相同错误日志
        self._prometheus_warned = False

    def _on_step(self) -> bool:
        """每步触发：每 log_freq 步记录训练标量，并检测 Episode 结束。

        Returns:
            True 表示继续训练
        """
        if self.n_calls % self.log_freq == 0:
            self.metrics_logger.log_scalar(
                "train/n_steps",
                float(self.num_timesteps),
                self.n_calls,
            )
        self._check_episode_end()
        # 提取 RL 训练指标到 Prometheus（Issue #411）
        try:
            from src.utils.metrics import (
                GRADIENT_NORM,
                POLICY_ENTROPY,
                POLICY_LOSS,
                TRAINING_STEPS,
                VALUE_LOSS,
            )

            info = self.model.logger.name_to_value
            if "entropy_loss" in info:
                POLICY_ENTROPY.set(-info["entropy_loss"])
            if "policy_gradient_loss" in info:
                POLICY_LOSS.set(info["policy_gradient_loss"])
            if "value_loss" in info:
                VALUE_LOSS.set(info["value_loss"])
            if "grad_norm" in info:
                GRADIENT_NORM.set(info["grad_norm"])
            TRAINING_STEPS.inc()
        except Exception as exc:  # noqa: BLE001
            # Issue #888: 首次失败记录 warning 并置位，避免静默吞异常且
            # 不重复输出相同错误日志
            if not self._prometheus_warned:
                self._prometheus_warned = True
                logger.warning(
                    f"[TrainingMetricsLogger] Prometheus 指标采集失败: {type(exc).__name__}: {exc}"
                )
        return True

    def _check_episode_end(self) -> None:
        """检测已完成的 Episode 并触发 on_episode_end 记录。"""
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos, strict=False):
            if not done:
                continue
            ep_info = info.get("episode") if isinstance(info, dict) else None
            if ep_info and isinstance(ep_info, dict):
                reward = float(ep_info.get("r", 0.0))
                length = int(ep_info.get("l", 0))
                self.on_episode_end(self._last_episode_count, reward, length, info)
                self._last_episode_count += 1

    def on_episode_end(
        self,
        episode: int,
        reward: float,
        length: int,
        info: dict | None = None,
    ) -> None:
        """Episode 结束时记录指标。

        Args:
            episode: Episode 序号
            reward : Episode 累计奖励
            length : Episode 步数
            info   : 额外信息
        """
        self.metrics_logger.log_episode(episode, reward, length, info)

    def _on_training_end(self) -> None:
        """训练结束时刷新日志。"""
        self.metrics_logger.flush()


def create_training_logger(
    log_dir: str = "logs/tensorboard",
    experiment_name: str = "ppo_training",
) -> tuple[TrainingMetricsLogger, TensorboardCallback]:
    """工厂函数：创建训练指标记录器与回调。

    Args:
        log_dir        : 日志根目录
        experiment_name: 实验名称

    Returns:
        ``(TrainingMetricsLogger, TensorboardCallback)`` 元组
    """
    metrics_logger = TrainingMetricsLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
    )
    callback = TensorboardCallback(metrics_logger)
    return metrics_logger, callback
