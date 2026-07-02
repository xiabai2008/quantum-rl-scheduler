"""
RL 模型导出补充测试（覆盖率提升）
Extended Unit Tests for Model Export

补充 tests/test_model_export.py 未覆盖的路径：
- _load_model 在 stable_baselines3 未安装时抛 ImportError（143-144 行）
- _get_input_shape / _resolve_shape 异常回退路径（195-197, 224-226 行）
- export_onnx 成功 / 失败路径（290-315 行，mock onnx 包与 th.onnx.export）
- validate_export 的 TorchScript 异常路径与 ONNX 验证各分支（410-448 行）
- export_all / export_model 中 ONNX 成功与非 ImportError 降级路径（502, 505-506, 567, 570-571 行）
"""

import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.export import _DEFAULT_INPUT_SHAPE, ModelExporter, export_model

# ---------------------------------------------------------------------------
# 依赖可用性检测
# ---------------------------------------------------------------------------

try:
    import stable_baselines3

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    import onnxruntime

    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False


# ---------------------------------------------------------------------------
# 测试用极简环境：避免依赖真实 QuantumSchedulingEnv，加速测试
# ---------------------------------------------------------------------------


class _DummyEnv(gym.Env):
    """测试用极简 Gymnasium 环境。

    提供 4 维观测空间和 Discrete(3) 动作空间，每个 episode 5 步，
    用于快速验证模型导出逻辑，不依赖真实调度环境。
    """

    observation_space = spaces.Box(low=0.0, high=1.0, shape=(4,), dtype=np.float32)
    action_space = spaces.Discrete(3)

    def __init__(self) -> None:
        """初始化测试环境。"""
        super().__init__()
        self._step_count = 0

    def reset(self, *, seed=None, options=None):
        """重置环境到初始状态。"""
        super().reset(seed=seed)
        self._step_count = 0
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        """执行一步环境交互。"""
        self._step_count += 1
        obs = np.zeros(4, dtype=np.float32)
        reward = 0.0
        terminated = self._step_count >= 5
        truncated = False
        return obs, reward, terminated, truncated, {}


# ---------------------------------------------------------------------------
# 测试基类：提供创建临时极小模型的公共逻辑
# ---------------------------------------------------------------------------


class _ExportExtendedTestBase(unittest.TestCase):
    """模型导出补充测试基类，提供临时目录与极小模型创建逻辑。"""

    def setUp(self) -> None:
        """每个测试前创建独立临时目录和测试环境。"""
        if not HAS_SB3:
            self.skipTest("stable_baselines3 未安装，跳过导出测试")
        self.tmpdir = tempfile.mkdtemp(prefix="export_ext_test_")
        self.env = _DummyEnv()
        self.input_shape: tuple[int, ...] = (4,)
        self.action_dim: int = 3

    def tearDown(self) -> None:
        """测试结束后清理临时目录。"""
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_ppo_model(self, name: str = "ppo_tiny", timesteps: int = 100) -> str:
        """创建并训练极小 PPO 模型，返回 .zip 文件路径。

        Args:
            name: 模型文件名（不含扩展名）
            timesteps: 训练步数

        Returns:
            模型 .zip 文件完整路径
        """
        from stable_baselines3 import PPO

        model = PPO(
            "MlpPolicy",
            self.env,
            verbose=0,
            n_steps=64,
            batch_size=32,
            policy_kwargs={"net_arch": [32]},
        )
        model.learn(total_timesteps=timesteps, reset_num_timesteps=True)
        path = os.path.join(self.tmpdir, name)
        model.save(path)
        return path + ".zip"


# ---------------------------------------------------------------------------
# 辅助函数：模拟 th.onnx.export 创建实际文件
# ---------------------------------------------------------------------------


def _fake_onnx_export(*args, **kwargs):
    """模拟 th.onnx.export，创建一个假的 onnx 文件。

    Args:
        *args: 位置参数，args[2] 为输出文件路径
        **kwargs: 关键字参数（忽略）
    """
    onnx_path = args[2]
    with open(onnx_path, "wb") as f:
        f.write(b"fake onnx content")


# ---------------------------------------------------------------------------
# TestLoadModelImportError: 覆盖 143-144 行
# ---------------------------------------------------------------------------


class TestLoadModelImportError(_ExportExtendedTestBase):
    """测试 _load_model 在 stable_baselines3 不可用时的异常路径。"""

    def test_load_model_raises_import_error_without_sb3(self) -> None:
        """stable_baselines3 未安装时 _load_model 应抛出 ImportError。"""
        # 创建一个存在的假文件，让 os.path.exists 检查通过
        fake_path = os.path.join(self.tmpdir, "fake.zip")
        with open(fake_path, "wb") as f:
            f.write(b"fake content")

        exporter = ModelExporter(fake_path, output_dir=self.tmpdir)
        # 让 stable_baselines3 导入失败（sys.modules 中设为 None 可阻断导入）
        with mock.patch.dict(sys.modules, {"stable_baselines3": None}):
            with self.assertRaises(ImportError) as cm:
                exporter._load_model()
            self.assertIn("stable_baselines3", str(cm.exception))


# ---------------------------------------------------------------------------
# TestShapeInferenceFallback: 覆盖 195-197, 224-226 行
# ---------------------------------------------------------------------------


class TestShapeInferenceFallback(_ExportExtendedTestBase):
    """测试输入形状推断的异常回退路径。"""

    def test_get_input_shape_falls_back_on_exception(self) -> None:
        """_get_input_shape 在模型加载异常时应回退到默认值 (14,)。"""
        exporter = ModelExporter(
            os.path.join(self.tmpdir, "fake.zip"),
            output_dir=self.tmpdir,
        )
        # mock _load_model 抛异常，触发 except 分支回退默认值
        with mock.patch.object(
            exporter, "_load_model", side_effect=RuntimeError("load failed")
        ):
            shape = exporter._get_input_shape()
        self.assertEqual(shape, _DEFAULT_INPUT_SHAPE)

    def test_resolve_shape_falls_back_on_exception(self) -> None:
        """_resolve_shape 在模型加载异常时应返回传入的默认形状。"""
        exporter = ModelExporter(
            os.path.join(self.tmpdir, "fake.zip"),
            output_dir=self.tmpdir,
        )
        with mock.patch.object(
            exporter, "_load_model", side_effect=RuntimeError("load failed")
        ):
            shape = exporter._resolve_shape(_DEFAULT_INPUT_SHAPE)
        self.assertEqual(shape, _DEFAULT_INPUT_SHAPE)


# ---------------------------------------------------------------------------
# TestExportOnnxWithMock: 覆盖 290-315 行（mock onnx 包）
# ---------------------------------------------------------------------------


class TestExportOnnxWithMock(_ExportExtendedTestBase):
    """测试 ONNX 导出成功与失败路径（通过 mock 模拟 onnx 包可用）。"""

    def test_export_onnx_success_with_mocked_onnx(self) -> None:
        """onnx 包可用时 export_onnx 应成功导出文件。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)

        # 注入 mock onnx 模块使 import onnx 成功
        mock_onnx = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"onnx": mock_onnx}), mock.patch(
            "torch.onnx.export", side_effect=_fake_onnx_export
        ) as mock_export:
            onnx_path = exporter.export_onnx(input_shape=self.input_shape)

        self.assertTrue(os.path.exists(onnx_path))
        self.assertTrue(onnx_path.endswith(".onnx"))
        mock_export.assert_called_once()

    def test_export_onnx_raises_runtime_error_on_export_failure(self) -> None:
        """th.onnx.export 抛异常时 export_onnx 应抛出 RuntimeError。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)

        mock_onnx = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"onnx": mock_onnx}), mock.patch(
            "torch.onnx.export", side_effect=RuntimeError("export failed")
        ):
            with self.assertRaises(RuntimeError) as cm:
                exporter.export_onnx(input_shape=self.input_shape)
            self.assertIn("ONNX 导出失败", str(cm.exception))


# ---------------------------------------------------------------------------
# TestValidateExportEdgeCases: 覆盖 410-412, 416-448 行
# ---------------------------------------------------------------------------


class TestValidateExportEdgeCases(_ExportExtendedTestBase):
    """测试 validate_export 的异常路径与 ONNX 验证各分支。"""

    def test_validate_torchscript_corrupted_file(self) -> None:
        """TorchScript 文件损坏时验证应标记 invalid 并记录错误。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        # 创建一个损坏的 .pt 文件（非 TorchScript 格式）
        corrupted_path = os.path.join(self.tmpdir, "corrupted.pt")
        with open(corrupted_path, "wb") as f:
            f.write(b"not a real torchscript file")

        result = exporter.validate_export(torchscript_path=corrupted_path)

        self.assertFalse(result["valid"])
        self.assertIn("torchscript", result["details"])
        self.assertFalse(result["details"]["torchscript"]["valid"])
        self.assertIn("error", result["details"]["torchscript"])

    def test_validate_onnx_nonexistent_path(self) -> None:
        """ONNX 文件不存在时验证应标记 invalid 并记录 '文件不存在'。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        fake_onnx = os.path.join(self.tmpdir, "nonexistent.onnx")

        result = exporter.validate_export(onnx_path=fake_onnx)

        self.assertFalse(result["valid"])
        self.assertIn("onnx", result["details"])
        self.assertFalse(result["details"]["onnx"]["valid"])
        self.assertEqual(result["details"]["onnx"]["error"], "文件不存在")

    @unittest.skipIf(
        HAS_ONNXRUNTIME, "onnxruntime 已安装，跳过降级测试"
    )
    def test_validate_onnx_runtime_not_installed(self) -> None:
        """onnxruntime 未安装时 ONNX 验证应标记 invalid 并记录错误。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        # 创建一个假的 onnx 文件（只需存在）
        onnx_path = os.path.join(self.tmpdir, "fake.onnx")
        with open(onnx_path, "wb") as f:
            f.write(b"fake onnx")

        result = exporter.validate_export(onnx_path=onnx_path)

        self.assertIn("onnx", result["details"])
        self.assertFalse(result["details"]["onnx"]["valid"])
        self.assertEqual(result["details"]["onnx"]["error"], "onnxruntime 未安装")

    def test_validate_onnx_success_with_mocked_runtime(self) -> None:
        """ONNX 验证在 onnxruntime 可用时应计算 diff 并返回 valid。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        # 创建一个假的 onnx 文件（只需存在）
        onnx_path = os.path.join(self.tmpdir, "fake.onnx")
        with open(onnx_path, "wb") as f:
            f.write(b"fake onnx")

        # 准备测试输入，并获取原始输出，让 mock onnxruntime 返回相同值 → diff=0
        test_input = np.zeros((1, *self.input_shape), dtype=np.float32)
        obs_tensor = th.as_tensor(test_input).float()
        original_output = exporter._get_original_output(obs_tensor).numpy()

        # mock onnxruntime 返回与原始一致的输出
        mock_ort = mock.MagicMock()
        mock_sess = mock.MagicMock()
        mock_input = mock.MagicMock()
        mock_input.name = "input"
        mock_sess.get_inputs.return_value = [mock_input]
        mock_sess.run.return_value = [original_output]
        mock_ort.InferenceSession.return_value = mock_sess

        with mock.patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            result = exporter.validate_export(
                onnx_path=onnx_path, test_input=test_input
            )

        self.assertIn("onnx", result["details"])
        self.assertTrue(result["details"]["onnx"]["valid"])
        self.assertLess(result["details"]["onnx"]["max_diff"], 1e-4)

    def test_validate_onnx_exception_with_mocked_runtime(self) -> None:
        """ONNX 验证过程抛异常时应标记 invalid 并记录错误。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        onnx_path = os.path.join(self.tmpdir, "fake.onnx")
        with open(onnx_path, "wb") as f:
            f.write(b"fake onnx")

        # mock onnxruntime 在创建 InferenceSession 时抛异常
        mock_ort = mock.MagicMock()
        mock_ort.InferenceSession.side_effect = RuntimeError("session creation failed")

        with mock.patch.dict(sys.modules, {"onnxruntime": mock_ort}):
            result = exporter.validate_export(onnx_path=onnx_path)

        self.assertFalse(result["valid"])
        self.assertIn("onnx", result["details"])
        self.assertFalse(result["details"]["onnx"]["valid"])
        self.assertIn("error", result["details"]["onnx"])


# ---------------------------------------------------------------------------
# TestExportAllOnnxPaths: 覆盖 502, 505-506 行
# ---------------------------------------------------------------------------


class TestExportAllOnnxPaths(_ExportExtendedTestBase):
    """测试 export_all 中 ONNX 成功与非 ImportError 降级路径。"""

    def test_export_all_onnx_success(self) -> None:
        """ONNX 可用时 export_all 应成功导出 ONNX 文件。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)

        mock_onnx = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"onnx": mock_onnx}):
            with mock.patch("torch.onnx.export", side_effect=_fake_onnx_export):
                result = exporter.export_all(input_shape=self.input_shape)

        self.assertIsNotNone(result["onnx_path"])
        self.assertTrue(os.path.exists(result["onnx_path"]))
        self.assertIsNotNone(result["torchscript_path"])
        self.assertIsNotNone(result["validation"])

    def test_export_all_onnx_runtime_error_degrades(self) -> None:
        """ONNX 导出抛 RuntimeError 时 export_all 应优雅降级。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)

        mock_onnx = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"onnx": mock_onnx}), mock.patch(
            "torch.onnx.export", side_effect=RuntimeError("export failed")
        ):
            result = exporter.export_all(input_shape=self.input_shape)

        # ONNX 降级为 None，TorchScript 仍成功
        self.assertIsNone(result["onnx_path"])
        self.assertIsNotNone(result["torchscript_path"])
        self.assertIsNotNone(result["validation"])


# ---------------------------------------------------------------------------
# TestExportModelOnnxPaths: 覆盖 567, 570-571 行
# ---------------------------------------------------------------------------


class TestExportModelOnnxPaths(_ExportExtendedTestBase):
    """测试 export_model 中 ONNX 成功与非 ImportError 降级路径。"""

    def test_export_model_onnx_only_success(self) -> None:
        """ONNX 可用时 export_model 仅导出 onnx 应成功。"""
        model_path = self._create_ppo_model()

        mock_onnx = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"onnx": mock_onnx}):
            with mock.patch("torch.onnx.export", side_effect=_fake_onnx_export):
                result = export_model(
                    model_path, output_dir=self.tmpdir, formats=["onnx"]
                )

        self.assertIsNotNone(result["onnx_path"])
        self.assertTrue(os.path.exists(result["onnx_path"]))
        self.assertIsNone(result["torchscript_path"])

    def test_export_model_onnx_runtime_error_degrades(self) -> None:
        """ONNX 导出抛 RuntimeError 时 export_model 应优雅降级。"""
        model_path = self._create_ppo_model()

        mock_onnx = mock.MagicMock()
        with mock.patch.dict(sys.modules, {"onnx": mock_onnx}), mock.patch(
            "torch.onnx.export", side_effect=RuntimeError("export failed")
        ):
            result = export_model(
                model_path, output_dir=self.tmpdir, formats=["onnx"]
            )

        self.assertIsNone(result["onnx_path"])
        self.assertIsNone(result["torchscript_path"])
        self.assertIsNotNone(result["validation"])


if __name__ == "__main__":
    unittest.main()
