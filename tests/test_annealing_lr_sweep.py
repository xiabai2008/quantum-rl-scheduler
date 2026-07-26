"""
学习率扫描实验脚本 smoke test（Issue #194）

验证 scripts/evaluation/annealing_lr_sweep.py 的核心组件可正常工作，
不运行完整 50k 步训练（太慢），仅验证导入、包装器和报告生成。
"""

import json
import os
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# 将项目根目录加入 sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


class TestLRSweepImport:
    """验证脚本模块可正常导入。"""

    def test_module_importable(self):
        """验证 annealing_lr_sweep 模块可被导入。"""
        # 设置环境变量以启用量子加速
        os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"
        from scripts.evaluation import annealing_lr_sweep

        assert hasattr(annealing_lr_sweep, "main")
        assert hasattr(annealing_lr_sweep, "LROverrideOptimizer")
        assert hasattr(annealing_lr_sweep, "_run_single")
        assert hasattr(annealing_lr_sweep, "_generate_report")

    def test_main_is_click_command(self):
        """验证 main 是 Click 命令。"""
        from scripts.evaluation import annealing_lr_sweep

        # Click 命令对象具有 callback 属性
        assert hasattr(annealing_lr_sweep.main, "callback")
        # Click 命令对象具有 params 属性（命令行参数列表）
        assert hasattr(annealing_lr_sweep.main, "params")


class TestLROverrideOptimizer:
    """验证学习率注入包装器。"""

    def test_wrapper_injects_learning_rate(self):
        """验证包装器在调用 optimize_policy 时注入 learning_rate。"""
        from scripts.evaluation.annealing_lr_sweep import LROverrideOptimizer

        # 创建 mock 基础优化器
        base = MagicMock()
        base.simulation_mode = True
        base.optimize_policy.return_value = "optimized_agent"

        wrapper = LROverrideOptimizer(base, learning_rate=0.3)
        assert wrapper.simulation_mode is True

        agent = MagicMock()
        result = wrapper.optimize_policy(agent, head_only=True)

        assert result == "optimized_agent"
        base.optimize_policy.assert_called_once()
        call_kwargs = base.optimize_policy.call_args
        assert call_kwargs.kwargs.get("learning_rate") == 0.3
        assert call_kwargs.kwargs.get("head_only") is True

    def test_wrapper_does_not_override_explicit_lr(self):
        """验证当调用方显式传入 learning_rate 时，包装器不覆盖。"""
        from scripts.evaluation.annealing_lr_sweep import LROverrideOptimizer

        base = MagicMock()
        base.simulation_mode = True
        base.optimize_policy.return_value = "optimized"

        wrapper = LROverrideOptimizer(base, learning_rate=0.3)
        wrapper.optimize_policy(MagicMock(), learning_rate=0.5)

        call_kwargs = base.optimize_policy.call_args
        # 显式传入的 0.5 不应被覆盖
        assert call_kwargs.kwargs.get("learning_rate") == 0.5

    def test_last_anneal_stats_property(self):
        """验证 last_anneal_stats 属性透传基础优化器的统计。"""
        from scripts.evaluation.annealing_lr_sweep import LROverrideOptimizer

        base = MagicMock()
        base.simulation_mode = True
        base._last_anneal_stats = {"ineffective_count": 5, "weight_l2_diff": 0.123}

        wrapper = LROverrideOptimizer(base, learning_rate=0.1)
        stats = wrapper.last_anneal_stats
        assert stats["ineffective_count"] == 5
        assert stats["weight_l2_diff"] == 0.123

    def test_last_anneal_stats_empty_when_no_attr(self):
        """验证基础优化器无 _last_anneal_stats 时返回空字典。"""
        from scripts.evaluation.annealing_lr_sweep import LROverrideOptimizer

        base = MagicMock(spec=["optimize_policy", "simulation_mode"])
        base.simulation_mode = True

        wrapper = LROverrideOptimizer(base, learning_rate=0.1)
        stats = wrapper.last_anneal_stats
        assert stats == {}


class TestGenerateReport:
    """验证报告生成函数。"""

    def test_generate_report_creates_file(self, tmp_path):
        """验证 _generate_report 生成 Markdown 报告文件。"""
        from scripts.evaluation.annealing_lr_sweep import _generate_report

        results: list[dict[str, Any]] = [
            {
                "learning_rate": 0.01,
                "seed": 42,
                "total_triggers": 10,
                "effective_triggers": 1,
                "impact_rate": 0.1,
                "ineffective_count": 9,
                "weight_l2_diff": 0.001,
                "final_reward": 1500.0,
                "train_time_s": 120.0,
            },
            {
                "learning_rate": 0.5,
                "seed": 42,
                "total_triggers": 10,
                "effective_triggers": 8,
                "impact_rate": 0.8,
                "ineffective_count": 2,
                "weight_l2_diff": 0.5,
                "final_reward": 1800.0,
                "train_time_s": 130.0,
            },
        ]
        output_path = str(tmp_path / "test_report.md")
        _generate_report(results, [0.01, 0.5], output_path)

        assert os.path.exists(output_path)
        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        assert "学习率扫描实验报告" in content
        assert "0.01" in content
        assert "0.5" in content
        assert "介入率" in content

    def test_generate_report_with_empty_results(self, tmp_path):
        """验证空结果列表也能生成报告（不崩溃）。"""
        from scripts.evaluation.annealing_lr_sweep import _generate_report

        output_path = str(tmp_path / "empty_report.md")
        _generate_report([], [0.01], output_path)

        assert os.path.exists(output_path)
        with open(output_path, encoding="utf-8") as f:
            content = f.read()
        assert "学习率扫描实验报告" in content


class TestRunSingleSmoke:
    """验证 _run_single 的基本流程（使用极小步数）。"""

    @pytest.mark.timeout(120, method="thread")
    def test_run_single_minimal_timesteps(self, tmp_path):
        """验证 _run_single 用极少步数能正常完成并返回结果字典。"""
        from scripts.evaluation import annealing_lr_sweep

        result = annealing_lr_sweep._run_single(
            learning_rate=0.1,
            seed=42,
            timesteps=256,
            anneal_interval=128,
            log_dir=str(tmp_path / "smoke_log"),
        )

        assert isinstance(result, dict)
        assert "learning_rate" in result
        assert "seed" in result
        assert "total_triggers" in result
        assert "effective_triggers" in result
        assert "impact_rate" in result
        assert "ineffective_count" in result
        assert "weight_l2_diff" in result
        assert "final_reward" in result
        assert result["learning_rate"] == 0.1
        assert result["seed"] == 42
        assert isinstance(result["impact_rate"], float)
        assert isinstance(result["ineffective_count"], int)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
