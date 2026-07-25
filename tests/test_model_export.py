"""
RL 模型导出测试（Issue #85）
Unit Tests for Model Export (ONNX / TorchScript)

测试覆盖：
- TestModelExporterBasic: 创建导出器、输出目录自动创建
- TestExportTorchScript: 导出 TorchScript 文件存在、可加载、输出形状正确
- TestExportOnnx: 导出 ONNX（如 onnx 包可用）；不可用时 skip + 优雅降级
- TestValidateExport: 验证导出模型与原始模型输出一致（max_diff < 1e-4）
- TestExportAll: 同时导出 ONNX + TorchScript + 验证
- TestExportModelFunction: 便捷函数 export_model 调用
- TestExportEdgeCases: 不存在的模型路径、无效输入形状、不支持的格式
"""

import os
import shutil
import sys
import tempfile
import unittest

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.scheduler.export import ModelExporter, export_model

# ---------------------------------------------------------------------------
# 依赖可用性检测
# ---------------------------------------------------------------------------

try:
    import stable_baselines3

    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

try:
    import onnx

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

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


class _ExportTestBase(unittest.TestCase):
    """模型导出测试基类，提供临时目录与极小模型创建逻辑。"""

    def setUp(self) -> None:
        """每个测试前创建独立临时目录和测试环境。"""
        if not HAS_SB3:
            self.skipTest("stable_baselines3 未安装，跳过导出测试")
        self.tmpdir = tempfile.mkdtemp(prefix="export_test_")
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

    def _create_dqn_model(self, name: str = "dqn_tiny", timesteps: int = 100) -> str:
        """创建并训练极小 DQN 模型，返回 .zip 文件路径。

        Args:
            name: 模型文件名（不含扩展名）
            timesteps: 训练步数

        Returns:
            模型 .zip 文件完整路径
        """
        from stable_baselines3 import DQN

        model = DQN(
            "MlpPolicy",
            self.env,
            verbose=0,
            learning_starts=50,
            buffer_size=500,
            batch_size=32,
            policy_kwargs={"net_arch": [32]},
        )
        model.learn(total_timesteps=timesteps, reset_num_timesteps=True)
        path = os.path.join(self.tmpdir, name)
        model.save(path)
        return path + ".zip"


# ---------------------------------------------------------------------------
# TestModelExporterBasic
# ---------------------------------------------------------------------------


class TestModelExporterBasic(_ExportTestBase):
    """测试 ModelExporter 基本功能：创建、输出目录创建。"""

    def test_init_creates_output_dir(self) -> None:
        """__init__ 应自动创建输出目录。"""
        output_dir = os.path.join(self.tmpdir, "exported")
        exporter = ModelExporter(
            model_path=os.path.join(self.tmpdir, "fake.zip"),
            output_dir=output_dir,
        )
        self.assertTrue(os.path.isdir(output_dir))
        self.assertEqual(exporter.model_path, os.path.join(self.tmpdir, "fake.zip"))
        self.assertEqual(exporter.output_dir, output_dir)

    def test_default_output_dir(self) -> None:
        """未指定 output_dir 时使用默认值（基于项目根目录的绝对路径）。"""
        exporter = ModelExporter(model_path=os.path.join(self.tmpdir, "fake.zip"))
        # 默认 output_dir 解析为项目根目录下的绝对路径
        self.assertTrue(exporter.output_dir.endswith("models/exported"))
        self.assertTrue(os.path.isabs(exporter.output_dir))
        self.assertTrue(os.path.isdir(exporter.output_dir))
        # 清理默认目录
        shutil.rmtree(exporter.output_dir, ignore_errors=True)

    def test_init_does_not_load_model(self) -> None:
        """__init__ 不应立即加载模型（延迟加载）。"""
        exporter = ModelExporter(
            model_path=os.path.join(self.tmpdir, "not_exist.zip"),
            output_dir=self.tmpdir,
        )
        self.assertIsNone(exporter.model)
        self.assertEqual(exporter.algorithm, "")

    def test_input_shape_inference_from_model(self) -> None:
        """_get_input_shape 应从模型观测空间推断输入形状。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        shape = exporter._get_input_shape()
        self.assertEqual(shape, (4,))

    def test_input_shape_explicit_param_overrides(self) -> None:
        """显式传入 input_shape 时优先使用参数值。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        shape = exporter._get_input_shape(input_shape=(8,))
        self.assertEqual(shape, (8,))


# ---------------------------------------------------------------------------
# TestExportTorchScript
# ---------------------------------------------------------------------------


class TestExportTorchScript(_ExportTestBase):
    """测试 TorchScript 导出：文件存在、可加载、输出形状正确。"""

    def test_export_torchscript_creates_file(self) -> None:
        """导出后 .pt 文件应存在。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        ts_path = exporter.export_torchscript(input_shape=self.input_shape)

        self.assertTrue(os.path.exists(ts_path))
        self.assertTrue(ts_path.endswith(".pt"))

    def test_exported_torchscript_loadable(self) -> None:
        """导出的 TorchScript 文件应可被 torch.jit.load 加载。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        ts_path = exporter.export_torchscript(input_shape=self.input_shape)

        loaded = th.jit.load(ts_path)
        self.assertIsNotNone(loaded)

    def test_exported_torchscript_output_shape(self) -> None:
        """导出的 TorchScript 输出形状应为 (batch, action_dim)。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        ts_path = exporter.export_torchscript(input_shape=self.input_shape)

        loaded = th.jit.load(ts_path)
        loaded.eval()
        test_input = th.randn(1, *self.input_shape, dtype=th.float32)
        with th.no_grad():
            output = loaded(test_input)
        # PPO Discrete 动作空间：action_net 输出 (batch, action_dim)
        self.assertEqual(output.shape, (1, self.action_dim))

    def test_export_torchscript_dqn(self) -> None:
        """DQN 模型导出 TorchScript 应成功且输出 Q 值。"""
        model_path = self._create_dqn_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        ts_path = exporter.export_torchscript(input_shape=self.input_shape)

        self.assertTrue(os.path.exists(ts_path))
        loaded = th.jit.load(ts_path)
        loaded.eval()
        test_input = th.randn(1, *self.input_shape, dtype=th.float32)
        with th.no_grad():
            output = loaded(test_input)
        # DQN 输出 Q 值，形状 (batch, action_dim)
        self.assertEqual(output.shape, (1, self.action_dim))

    def test_export_torchscript_returns_correct_path(self) -> None:
        """返回路径应基于模型文件名，位于 output_dir 内。"""
        model_path = self._create_ppo_model(name="my_ppo")
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        ts_path = exporter.export_torchscript(input_shape=self.input_shape)

        self.assertEqual(os.path.basename(ts_path), "my_ppo.pt")
        self.assertEqual(os.path.dirname(ts_path), self.tmpdir)


# ---------------------------------------------------------------------------
# TestExportOnnx
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_ONNX, "onnx 包未安装，跳过 ONNX 导出测试")
class TestExportOnnx(_ExportTestBase):
    """测试 ONNX 导出（仅当 onnx 包可用时执行）。"""

    def test_export_onnx_creates_file(self) -> None:
        """导出后 .onnx 文件应存在。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        onnx_path = exporter.export_onnx(input_shape=self.input_shape)

        self.assertTrue(os.path.exists(onnx_path))
        self.assertTrue(onnx_path.endswith(".onnx"))

    def test_export_onnx_returns_correct_path(self) -> None:
        """返回路径应基于模型文件名，位于 output_dir 内。"""
        model_path = self._create_ppo_model(name="my_ppo")
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        onnx_path = exporter.export_onnx(input_shape=self.input_shape)

        self.assertEqual(os.path.basename(onnx_path), "my_ppo.onnx")

    @unittest.skipUnless(HAS_ONNXRUNTIME, "onnxruntime 未安装，跳过 ONNX 推理验证")
    def test_exported_onnx_inferable(self) -> None:
        """导出的 ONNX 模型应可通过 onnxruntime 推理。"""
        import onnxruntime as ort

        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        onnx_path = exporter.export_onnx(input_shape=self.input_shape)

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        test_input = np.random.randn(1, *self.input_shape).astype(np.float32)
        output = sess.run(None, {input_name: test_input})[0]
        self.assertEqual(output.shape, (1, self.action_dim))


class TestExportOnnxGracefulDegradation(_ExportTestBase):
    """测试 ONNX 不可用时的优雅降级。"""

    @unittest.skipIf(HAS_ONNX, "onnx 包已安装，跳过降级测试")
    def test_export_onnx_raises_import_error_without_onnx(self) -> None:
        """onnx 包未安装时 export_onnx 应抛出 ImportError。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        with self.assertRaises(ImportError):
            exporter.export_onnx(input_shape=self.input_shape)

    def test_export_all_degrades_gracefully_without_onnx(self) -> None:
        """export_all 在 ONNX 不可用时应优雅降级，TorchScript 仍成功。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        result = exporter.export_all(input_shape=self.input_shape)

        # TorchScript 必须成功
        self.assertIsNotNone(result["torchscript_path"])
        self.assertTrue(os.path.exists(result["torchscript_path"]))
        # ONNX 在无 onnx 包时为 None（降级）
        if not HAS_ONNX:
            self.assertIsNone(result["onnx_path"])
        # 验证结果仍存在
        self.assertIsNotNone(result["validation"])


# ---------------------------------------------------------------------------
# TestValidateExport
# ---------------------------------------------------------------------------


class TestValidateExport(_ExportTestBase):
    """测试导出模型与原始模型输出一致性验证。"""

    def test_validate_torchscript_consistent(self) -> None:
        """TorchScript 导出模型输出应与原始模型一致（max_diff < 1e-4）。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        ts_path = exporter.export_torchscript(input_shape=self.input_shape)

        result = exporter.validate_export(torchscript_path=ts_path)

        self.assertIn("max_diff", result)
        self.assertIn("mean_diff", result)
        self.assertIn("valid", result)
        self.assertLess(result["max_diff"], 1e-4)
        self.assertTrue(result["valid"])

    def test_validate_with_explicit_test_input(self) -> None:
        """传入显式 test_input 时验证应正常工作。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        ts_path = exporter.export_torchscript(input_shape=self.input_shape)

        test_input = np.random.randn(1, *self.input_shape).astype(np.float32)
        result = exporter.validate_export(torchscript_path=ts_path, test_input=test_input)

        self.assertTrue(result["valid"])
        self.assertLess(result["max_diff"], 1e-4)

    def test_validate_returns_details(self) -> None:
        """验证结果应包含 details 字段，记录各格式详情。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        ts_path = exporter.export_torchscript(input_shape=self.input_shape)

        result = exporter.validate_export(torchscript_path=ts_path)

        self.assertIn("details", result)
        self.assertIn("torchscript", result["details"])
        ts_detail = result["details"]["torchscript"]
        self.assertIn("max_diff", ts_detail)
        self.assertIn("mean_diff", ts_detail)
        self.assertTrue(ts_detail["valid"])

    def test_validate_dqn_torchscript_consistent(self) -> None:
        """DQN 模型 TorchScript 导出应与原始模型一致。"""
        model_path = self._create_dqn_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        ts_path = exporter.export_torchscript(input_shape=self.input_shape)

        result = exporter.validate_export(torchscript_path=ts_path)

        self.assertLess(result["max_diff"], 1e-4)
        self.assertTrue(result["valid"])

    def test_validate_nonexistent_torchscript_path(self) -> None:
        """TorchScript 文件不存在时验证应标记为 invalid。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        fake_path = os.path.join(self.tmpdir, "nonexistent.pt")

        result = exporter.validate_export(torchscript_path=fake_path)

        self.assertFalse(result["valid"])
        self.assertIn("torchscript", result["details"])

    def test_validate_no_paths_returns_invalid(self) -> None:
        """未提供任何路径时验证结果应为 invalid。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)

        result = exporter.validate_export()

        self.assertFalse(result["valid"])

    @unittest.skipUnless(
        HAS_ONNX and HAS_ONNXRUNTIME, "onnx/onnxruntime 未安装，跳过 ONNX 验证测试"
    )
    def test_validate_onnx_consistent(self) -> None:
        """ONNX 导出模型输出应与原始模型一致。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        onnx_path = exporter.export_onnx(input_shape=self.input_shape)

        result = exporter.validate_export(onnx_path=onnx_path)

        self.assertLess(result["max_diff"], 1e-4)
        self.assertTrue(result["valid"])


# ---------------------------------------------------------------------------
# TestExportAll
# ---------------------------------------------------------------------------


class TestExportAll(_ExportTestBase):
    """测试同时导出 ONNX + TorchScript + 验证。"""

    def test_export_all_returns_complete_dict(self) -> None:
        """export_all 应返回包含所有键的结果字典。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        result = exporter.export_all(input_shape=self.input_shape)

        self.assertIn("onnx_path", result)
        self.assertIn("torchscript_path", result)
        self.assertIn("validation", result)

    def test_export_all_torchscript_succeeds(self) -> None:
        """export_all 中 TorchScript 必须成功导出。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        result = exporter.export_all(input_shape=self.input_shape)

        self.assertIsNotNone(result["torchscript_path"])
        self.assertTrue(os.path.exists(result["torchscript_path"]))

    def test_export_all_validation_valid(self) -> None:
        """export_all 的验证结果应为 valid（至少 TorchScript 验证通过）。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        result = exporter.export_all(input_shape=self.input_shape)

        validation = result["validation"]
        self.assertTrue(validation["valid"])
        self.assertLess(validation["max_diff"], 1e-4)

    def test_export_all_dqn(self) -> None:
        """DQN 模型 export_all 应成功。"""
        model_path = self._create_dqn_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        result = exporter.export_all(input_shape=self.input_shape)

        self.assertIsNotNone(result["torchscript_path"])
        self.assertTrue(result["validation"]["valid"])


# ---------------------------------------------------------------------------
# TestExportModelFunction
# ---------------------------------------------------------------------------


class TestExportModelFunction(_ExportTestBase):
    """测试便捷函数 export_model。"""

    def test_export_model_default_all_formats(self) -> None:
        """默认导出所有格式（ONNX 不可用时降级）。"""
        model_path = self._create_ppo_model()
        result = export_model(model_path, output_dir=self.tmpdir)

        self.assertIn("onnx_path", result)
        self.assertIn("torchscript_path", result)
        self.assertIn("validation", result)
        self.assertIsNotNone(result["torchscript_path"])
        self.assertTrue(os.path.exists(result["torchscript_path"]))

    def test_export_model_torchscript_only(self) -> None:
        """仅导出 torchscript 格式。"""
        model_path = self._create_ppo_model()
        result = export_model(model_path, output_dir=self.tmpdir, formats=["torchscript"])

        self.assertIsNotNone(result["torchscript_path"])
        self.assertIsNone(result["onnx_path"])
        self.assertTrue(result["validation"]["valid"])

    def test_export_model_onnx_only(self) -> None:
        """仅导出 onnx 格式（不可用时降级）。"""
        model_path = self._create_ppo_model()
        result = export_model(model_path, output_dir=self.tmpdir, formats=["onnx"])

        self.assertIsNone(result["torchscript_path"])
        if HAS_ONNX:
            self.assertIsNotNone(result["onnx_path"])
        else:
            self.assertIsNone(result["onnx_path"])

    def test_export_model_creates_output_dir(self) -> None:
        """output_dir 不存在时应自动创建。"""
        model_path = self._create_ppo_model()
        new_dir = os.path.join(self.tmpdir, "new_export_dir")
        result = export_model(model_path, output_dir=new_dir, formats=["torchscript"])

        self.assertTrue(os.path.isdir(new_dir))
        self.assertTrue(os.path.exists(result["torchscript_path"]))


# ---------------------------------------------------------------------------
# TestExportEdgeCases
# ---------------------------------------------------------------------------


class TestExportEdgeCases(_ExportTestBase):
    """测试边界情况：不存在的模型路径、无效输入形状、不支持的格式。"""

    def test_nonexistent_model_path_raises(self) -> None:
        """不存在的模型路径应抛出 FileNotFoundError。"""
        fake_path = os.path.join(self.tmpdir, "not_exist.zip")
        exporter = ModelExporter(fake_path, output_dir=self.tmpdir)
        with self.assertRaises(FileNotFoundError):
            exporter.export_torchscript(input_shape=self.input_shape)

    def test_nonexistent_model_path_load_raises(self) -> None:
        """_load_model 对不存在的路径应抛出 FileNotFoundError。"""
        fake_path = os.path.join(self.tmpdir, "not_exist.zip")
        exporter = ModelExporter(fake_path, output_dir=self.tmpdir)
        with self.assertRaises(FileNotFoundError):
            exporter._load_model()

    def test_invalid_input_shape_raises_runtime_error(self) -> None:
        """传入与模型不匹配的输入形状导出时应抛出 RuntimeError。"""
        model_path = self._create_ppo_model()
        exporter = ModelExporter(model_path, output_dir=self.tmpdir)
        # 模型期望 4 维，传入 99 维应导致前向传播失败
        with self.assertRaises((RuntimeError, Exception)):
            exporter.export_torchscript(input_shape=(99,))

    def test_corrupted_model_file_raises(self) -> None:
        """损坏的模型文件应抛出异常。"""
        fake_zip = os.path.join(self.tmpdir, "corrupt.zip")
        with open(fake_zip, "wb") as f:
            f.write(b"not a real zip file content")

        exporter = ModelExporter(fake_zip, output_dir=self.tmpdir)
        with self.assertRaises((ValueError, Exception)):
            exporter.export_torchscript(input_shape=self.input_shape)

    def test_unsupported_format_raises_value_error(self) -> None:
        """export_model 传入不支持的格式应抛出 ValueError。"""
        model_path = self._create_ppo_model()
        with self.assertRaises(ValueError):
            export_model(model_path, output_dir=self.tmpdir, formats=["invalid_format"])

    def test_empty_formats_list(self) -> None:
        """空 formats 列表应正常处理（不导出任何格式，验证结果为 invalid）。"""
        model_path = self._create_ppo_model()
        result = export_model(model_path, output_dir=self.tmpdir, formats=[])

        self.assertIsNone(result["onnx_path"])
        self.assertIsNone(result["torchscript_path"])
        # 没有导出任何文件，验证应为 invalid
        self.assertFalse(result["validation"]["valid"])

    def test_export_torchscript_default_input_shape_param(self) -> None:
        """export_torchscript 默认 input_shape 参数为 (14,)。"""
        # 验证默认参数值（不实际调用，因为模型是 4 维）
        import inspect

        sig = inspect.signature(ModelExporter.export_torchscript)
        default = sig.parameters["input_shape"].default
        self.assertEqual(default, (14,))

    def test_export_onnx_default_opset_version(self) -> None:
        """export_onnx 默认 opset_version 参数为 14。"""
        import inspect

        sig = inspect.signature(ModelExporter.export_onnx)
        default = sig.parameters["opset_version"].default
        self.assertEqual(default, 14)


if __name__ == "__main__":
    unittest.main()
