"""
RL 模型导出模块
Model Export Module for RL Trained Policies

将训练好的 Stable-Baselines3 模型（PPO/DQN）导出为 ONNX / TorchScript 格式，
支持轻量推理部署。TorchScript 为主要导出格式（不依赖额外包），ONNX 需要安装
onnx 包（不可用时优雅降级，不中断流程）。

使用示例：
    >>> from src.scheduler.export import ModelExporter, export_model
    >>> exporter = ModelExporter("./models/ppo_scheduler.zip")
    >>> ts_path = exporter.export_torchscript()
    >>> paths = exporter.export_all()
    >>> # 或使用便捷函数
    >>> paths = export_model("./models/ppo_scheduler.zip")
"""

from __future__ import annotations

import os
from typing import Any, cast

import numpy as np
import torch as th
from torch import nn

try:
    from loguru import logger
except ImportError:  # pragma: no cover
    import logging

    logger = logging.getLogger(__name__)  # type: ignore[assignment]

__all__ = ["ModelExporter", "export_model"]

# 默认输入形状（对应 14 维调度状态空间）
_DEFAULT_INPUT_SHAPE: tuple[int, ...] = (14,)
# 验证阈值：max_diff < 1e-4 视为导出有效
_VALIDATION_THRESHOLD: float = 1e-4


class _PolicyWrapper(nn.Module):
    """
    SB3 策略包装器：将 SB3 policy 包装为可导出的标准 nn.Module。

    PPO（ActorCriticPolicy）：返回动作网络输出（动作 logits / 均值）
    DQN（QNetwork）：返回 Q 值

    包装后的模块接收观测张量，输出与决策直接相关的张量，
    便于 ONNX / TorchScript 导出与跨平台推理。
    """

    def __init__(self, policy: Any, algorithm: str) -> None:
        """
        初始化策略包装器。

        Args:
            policy: SB3 策略对象（ActorCriticPolicy 或 QNetwork）
            algorithm: 算法名称（"ppo" / "dqn"），决定前向传播路径
        """
        super().__init__()
        self.policy = policy
        self.algorithm = algorithm.lower()

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        前向传播：从观测计算决策相关输出。

        Args:
            obs: 观测张量，形状 (batch_size, obs_dim)

        Returns:
            PPO 返回动作网络输出（Discrete 为 logits，形状 (batch, action_dim)）；
            DQN 返回 Q 值张量，形状 (batch_size, action_dim)
        """
        if self.algorithm == "dqn":
            # DQNPolicy.forward 返回动作（argmax），需调用 q_net 获取 Q 值。
            # q_net 是 QNetwork（或 DuelingQNetwork），其 forward 返回 Q 值。
            # policy 为 Any 类型（SB3 策略），调用结果需 cast 为 Tensor
            return cast(th.Tensor, self.policy.q_net(obs))
        # PPO ActorCriticPolicy：提取特征 → 共享 MLP（actor 分支）→ 动作网络
        features = self.policy.extract_features(obs, self.policy.features_extractor)
        latent_pi = self.policy.mlp_extractor.forward_actor(features)
        return cast(th.Tensor, self.policy.action_net(latent_pi))


class ModelExporter:
    """
    RL 模型导出器：将 SB3 模型导出为 ONNX / TorchScript 格式。

    支持 PPO 与 DQN 模型，自动检测算法类型，提供导出后验证。
    TorchScript 为主要导出格式（无额外依赖），ONNX 在缺少 onnx 包时优雅降级。

    使用示例：
        >>> exporter = ModelExporter("./models/ppo.zip", output_dir="./exported")
        >>> ts_path = exporter.export_torchscript()
        >>> result = exporter.validate_export(torchscript_path=ts_path)
        >>> all_paths = exporter.export_all()
    """

    def __init__(self, model_path: str, output_dir: str = "models/exported") -> None:
        """
        初始化模型导出器。

        Args:
            model_path: SB3 模型文件路径（.zip 格式）
            output_dir: 导出文件保存目录，默认 "models/exported"
        """
        self.model_path = model_path
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 延迟加载的属性
        self.model: Any = None
        self.algorithm: str = ""
        self._wrapper: _PolicyWrapper | None = None

    # ------------------------------------------------------------------
    # 内部辅助方法
    # ------------------------------------------------------------------

    def _load_model(self) -> Any:
        """
        加载 SB3 模型，自动检测 PPO / DQN。

        依次尝试以 PPO、DQN 格式加载，首个成功者生效。

        Returns:
            加载的 SB3 模型对象

        Raises:
            FileNotFoundError: 模型文件不存在
            ImportError: stable_baselines3 未安装
            ValueError: 无法以 PPO 或 DQN 格式加载
        """
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")

        if self.model is not None:
            return self.model

        try:
            from stable_baselines3 import DQN, PPO
            from stable_baselines3.common.base_class import BaseAlgorithm
        except ImportError as e:
            raise ImportError(
                "stable_baselines3 未安装，无法加载模型。 请执行: pip install stable-baselines3"
            ) from e

        errors: list[str] = []
        for algo_cls, name in [(PPO, "ppo"), (DQN, "dqn")]:
            try:
                # algo_cls 经 mypy 推断为 ABCMeta（抽象基类的元类），无 .load 属性；
                # 实际 PPO/DQN 均为 BaseAlgorithm 子类，cast 为 type[BaseAlgorithm] 以访问 .load
                model = cast(type[BaseAlgorithm], algo_cls).load(self.model_path, device="cpu")
                self.model = model
                self.algorithm = name
                logger.info(f"成功以 {name.upper()} 格式加载模型: {self.model_path}")
                return model
            except Exception as e:
                errors.append(f"{name.upper()}: {e}")
                logger.debug(f"以 {name.upper()} 加载失败: {e}")

        raise ValueError(f"无法加载模型 {self.model_path}，尝试 PPO/DQN 均失败: {errors}")

    def _get_wrapper(self) -> _PolicyWrapper:
        """
        获取策略包装器（惰性创建），切换至 eval 模式并移至 CPU。

        Returns:
            _PolicyWrapper 实例
        """
        if self._wrapper is None:
            model = self._load_model()
            # 将策略切换到 eval 模式并移至 CPU，确保导出确定性
            model.policy.set_training_mode(False)
            model.policy = model.policy.to("cpu")
            self._wrapper = _PolicyWrapper(model.policy, self.algorithm)
            self._wrapper.eval()
        return self._wrapper

    def _get_input_shape(self, input_shape: tuple[int, ...] | None = None) -> tuple[int, ...]:
        """
        推断输入形状。优先使用参数，其次从模型观测空间推断，最后回退默认值。

        Args:
            input_shape: 显式指定的输入形状，若为 None 则自动推断

        Returns:
            输入形状元组
        """
        if input_shape is not None:
            return input_shape
        try:
            model = self._load_model()
            obs_space = model.observation_space
            if hasattr(obs_space, "shape") and obs_space.shape:
                return tuple(obs_space.shape)
        except Exception:
            pass
        return _DEFAULT_INPUT_SHAPE

    def _resolve_shape(self, input_shape: tuple[int, ...]) -> tuple[int, ...]:
        """
        解析输入形状：若传入默认值 (14,) 且模型观测空间不同，则自动推断。

        这样 export_torchscript() / export_onnx() / export_all() 在使用默认
        参数时能自动适配模型的实际观测维度，避免形状不匹配。

        Args:
            input_shape: 调用方传入的输入形状

        Returns:
            解析后的输入形状（可能与传入值不同，若触发了自动推断）
        """
        if input_shape != _DEFAULT_INPUT_SHAPE:
            return input_shape
        try:
            model = self._load_model()
            obs_space = model.observation_space
            if hasattr(obs_space, "shape") and obs_space.shape:
                inferred = tuple(obs_space.shape)
                if inferred != _DEFAULT_INPUT_SHAPE:
                    logger.debug(
                        f"从模型观测空间推断输入形状: {inferred}（覆盖默认 {input_shape}）"
                    )
                    return inferred
        except Exception:
            pass
        return input_shape

    def _make_dummy_input(self, input_shape: tuple[int, ...] | None = None) -> th.Tensor:
        """
        创建用于导出的 dummy 输入张量。

        Args:
            input_shape: 输入形状，若为 None 则自动推断

        Returns:
            形状为 (1, *input_shape) 的 float32 随机张量
        """
        shape = self._get_input_shape(input_shape)
        return th.randn(1, *shape, dtype=th.float32)

    def _get_original_output(self, obs_tensor: th.Tensor) -> th.Tensor:
        """
        使用原始 SB3 策略包装器计算输出（用于验证对比）。

        Args:
            obs_tensor: 观测张量

        Returns:
            原始模型输出张量（已 detach 并移至 CPU）
        """
        wrapper = self._get_wrapper()
        with th.no_grad():
            # wrapper(...) 调用 nn.Module.__call__，返回 Any，需 cast 为 Tensor
            return cast(th.Tensor, wrapper(obs_tensor).detach().cpu())

    # ------------------------------------------------------------------
    # 公开导出方法
    # ------------------------------------------------------------------

    def export_onnx(
        self,
        input_shape: tuple[int, ...] = (14,),
        opset_version: int = 14,
    ) -> str:
        """
        将模型导出为 ONNX 格式。

        需要 onnx 包支持；未安装时抛出 ImportError，调用方可 try/except 优雅降级。
        支持 dynamic_axes（batch 维度动态），便于批量推理。

        Args:
            input_shape: 输入形状，默认 (14,)
            opset_version: ONNX opset 版本，默认 14

        Returns:
            导出的 ONNX 文件路径

        Raises:
            ImportError: onnx 包未安装
            RuntimeError: 导出过程失败
        """
        input_shape = self._resolve_shape(input_shape)
        logger.info(f"开始导出 ONNX，input_shape={input_shape}, opset={opset_version}")

        try:
            import onnx  # noqa: F401
        except ImportError as e:
            logger.warning("onnx 包未安装，ONNX 导出不可用。请执行: pip install onnx")
            raise ImportError("onnx 包未安装，无法导出 ONNX。请执行: pip install onnx") from e

        wrapper = self._get_wrapper()
        dummy_input = self._make_dummy_input(input_shape)

        base_name = os.path.splitext(os.path.basename(self.model_path))[0]
        onnx_path = os.path.join(self.output_dir, f"{base_name}.onnx")

        try:
            th.onnx.export(
                wrapper,
                (dummy_input,),  # args 参数期望 tuple[Tensor, ...]，包装为单元素元组
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={
                    "input": {0: "batch_size"},
                    "output": {0: "batch_size"},
                },
            )
            logger.info(f"ONNX 导出成功: {onnx_path}")
            return onnx_path
        except Exception as e:
            logger.error(f"ONNX 导出失败: {e}")
            raise RuntimeError(f"ONNX 导出失败: {e}") from e

    def export_torchscript(self, input_shape: tuple[int, ...] = (14,)) -> str:
        """
        将模型导出为 TorchScript 格式。

        使用 torch.jit.trace 跟踪策略网络，生成可在无 Python 环境下加载的 .pt 文件。
        TorchScript 为主要导出格式，不依赖额外包。

        Args:
            input_shape: 输入形状，默认 (14,)

        Returns:
            导出的 TorchScript 文件路径 (.pt)

        Raises:
            RuntimeError: 导出过程失败
        """
        input_shape = self._resolve_shape(input_shape)
        logger.info(f"开始导出 TorchScript，input_shape={input_shape}")

        wrapper = self._get_wrapper()
        wrapper.eval()
        dummy_input = self._make_dummy_input(input_shape)

        base_name = os.path.splitext(os.path.basename(self.model_path))[0]
        ts_path = os.path.join(self.output_dir, f"{base_name}.pt")

        try:
            with th.no_grad():
                traced = th.jit.trace(wrapper, dummy_input)
            traced.save(ts_path)
            logger.info(f"TorchScript 导出成功: {ts_path}")
            return ts_path
        except Exception as e:
            logger.error(f"TorchScript 导出失败: {e}")
            raise RuntimeError(f"TorchScript 导出失败: {e}") from e

    def validate_export(
        self,
        onnx_path: str | None = None,
        torchscript_path: str | None = None,
        test_input: np.ndarray | None = None,
    ) -> dict:
        """
        验证导出的模型与原始 SB3 模型输出一致性。

        加载导出的 ONNX / TorchScript 模型，在相同输入下与原始策略网络输出对比，
        计算 max_diff / mean_diff，valid = max_diff < 1e-4。

        Args:
            onnx_path: ONNX 文件路径，None 则跳过 ONNX 验证
            torchscript_path: TorchScript 文件路径，None 则跳过验证
            test_input: 测试输入，None 则随机生成

        Returns:
            验证结果字典，包含 max_diff, mean_diff, valid 及各格式详情（details）
        """
        input_shape = self._get_input_shape()
        if test_input is None:
            test_input = np.random.randn(1, *input_shape).astype(np.float32)
        else:
            test_input = np.asarray(test_input, dtype=np.float32)

        obs_tensor = th.as_tensor(test_input).float()
        original_output = self._get_original_output(obs_tensor).numpy()

        diffs: list[np.ndarray] = []
        details: dict[str, Any] = {}

        # 验证 TorchScript
        if torchscript_path is not None:
            if not os.path.exists(torchscript_path):
                logger.warning(f"TorchScript 文件不存在: {torchscript_path}")
                details["torchscript"] = {"valid": False, "error": "文件不存在"}
            else:
                try:
                    loaded = th.jit.load(torchscript_path)
                    loaded.eval()
                    with th.no_grad():
                        ts_output = loaded(obs_tensor).numpy()
                    diff = np.abs(ts_output - original_output)
                    ts_max = float(diff.max())
                    ts_mean = float(diff.mean())
                    ts_valid = ts_max < _VALIDATION_THRESHOLD
                    details["torchscript"] = {
                        "max_diff": ts_max,
                        "mean_diff": ts_mean,
                        "valid": ts_valid,
                    }
                    diffs.append(diff)
                    logger.info(
                        f"TorchScript 验证: max_diff={ts_max:.2e}, "
                        f"mean_diff={ts_mean:.2e}, valid={ts_valid}"
                    )
                except Exception as e:
                    logger.error(f"TorchScript 验证失败: {e}")
                    details["torchscript"] = {"valid": False, "error": str(e)}

        # 验证 ONNX（需 onnxruntime）
        if onnx_path is not None:
            if not os.path.exists(onnx_path):
                logger.warning(f"ONNX 文件不存在: {onnx_path}")
                details["onnx"] = {"valid": False, "error": "文件不存在"}
            else:
                try:
                    import onnxruntime as ort

                    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
                    input_name = sess.get_inputs()[0].name
                    onnx_output = sess.run(None, {input_name: test_input})[0]
                    diff = np.abs(onnx_output - original_output)
                    onnx_max = float(diff.max())
                    onnx_mean = float(diff.mean())
                    onnx_valid = onnx_max < _VALIDATION_THRESHOLD
                    details["onnx"] = {
                        "max_diff": onnx_max,
                        "mean_diff": onnx_mean,
                        "valid": onnx_valid,
                    }
                    diffs.append(diff)
                    logger.info(
                        f"ONNX 验证: max_diff={onnx_max:.2e}, "
                        f"mean_diff={onnx_mean:.2e}, valid={onnx_valid}"
                    )
                except ImportError:
                    logger.warning("onnxruntime 未安装，跳过 ONNX 验证")
                    details["onnx"] = {
                        "valid": False,
                        "error": "onnxruntime 未安装",
                    }
                except Exception as e:
                    logger.error(f"ONNX 验证失败: {e}")
                    details["onnx"] = {"valid": False, "error": str(e)}

        # 汇总 max_diff / mean_diff
        if diffs:
            all_diffs = np.concatenate([d.flatten() for d in diffs])
            max_diff = float(all_diffs.max())
            mean_diff = float(all_diffs.mean())
        else:
            max_diff = 0.0
            mean_diff = 0.0

        # overall valid：所有已验证格式均有效且 max_diff < 阈值
        all_formats_valid = (
            all(d.get("valid", False) for d in details.values()) if details else False
        )
        overall_valid = (max_diff < _VALIDATION_THRESHOLD) and all_formats_valid

        return {
            "max_diff": max_diff,
            "mean_diff": mean_diff,
            "valid": overall_valid,
            "details": details,
        }

    def export_all(self, input_shape: tuple[int, ...] = (14,)) -> dict:
        """
        同时导出 ONNX + TorchScript 并验证。

        ONNX 导出失败时优雅降级（记录警告，不中断流程），
        TorchScript 为主要格式必须成功。

        Args:
            input_shape: 输入形状，默认 (14,)

        Returns:
            结果字典：{onnx_path, torchscript_path, validation}
        """
        result: dict[str, Any] = {
            "onnx_path": None,
            "torchscript_path": None,
            "validation": None,
        }

        # 解析输入形状（默认值时自动从模型推断）
        input_shape = self._resolve_shape(input_shape)

        # 导出 TorchScript（主要格式，必须成功）
        ts_path = self.export_torchscript(input_shape)
        result["torchscript_path"] = ts_path

        # 导出 ONNX（优雅降级）
        onnx_path: str | None = None
        try:
            onnx_path = self.export_onnx(input_shape)
            result["onnx_path"] = onnx_path
        except ImportError as e:
            logger.warning(f"ONNX 导出跳过（依赖缺失）: {e}")
        except Exception as e:
            logger.warning(f"ONNX 导出失败（已降级）: {e}")

        # 验证
        validation = self.validate_export(
            onnx_path=onnx_path,
            torchscript_path=ts_path,
        )
        result["validation"] = validation

        return result


def export_model(
    model_path: str,
    output_dir: str = "models/exported",
    formats: list[str] | None = None,
) -> dict:
    """
    便捷函数：导出模型到指定格式。

    Args:
        model_path: SB3 模型文件路径（.zip）
        output_dir: 输出目录，默认 "models/exported"
        formats: 要导出的格式列表，None 表示全部 ["onnx", "torchscript"]

    Returns:
        导出结果字典（同 ModelExporter.export_all 返回值）

    Raises:
        ValueError: formats 包含不支持的格式
    """
    if formats is None:
        formats = ["onnx", "torchscript"]

    supported = {"onnx", "torchscript"}
    invalid = set(formats) - supported
    if invalid:
        raise ValueError(f"不支持的导出格式: {invalid}，支持: {supported}")

    exporter = ModelExporter(model_path, output_dir=output_dir)

    # 同时导出两种格式时直接走 export_all
    if set(formats) == supported:
        return exporter.export_all()

    result: dict[str, Any] = {
        "onnx_path": None,
        "torchscript_path": None,
        "validation": None,
    }

    ts_path: str | None = None
    onnx_path: str | None = None

    if "torchscript" in formats:
        ts_path = exporter.export_torchscript()
        result["torchscript_path"] = ts_path

    if "onnx" in formats:
        try:
            onnx_path = exporter.export_onnx()
            result["onnx_path"] = onnx_path
        except ImportError as e:
            logger.warning(f"ONNX 导出跳过: {e}")
        except Exception as e:
            logger.warning(f"ONNX 导出失败: {e}")

    result["validation"] = exporter.validate_export(
        onnx_path=onnx_path,
        torchscript_path=ts_path,
    )

    return result
