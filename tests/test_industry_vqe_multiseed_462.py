"""industry_vqe_multiseed.py 单元测试（Issue #462）。

测试覆盖：
    - 分子配置完整性
    - 执行时间计算
    - 机制分析逻辑（模式 A/B 自动选择）
    - 报告生成（含机制解释）
    - 统计计算正确性
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import pytest

from scripts.evaluation.industry_vqe_multiseed import (
    DEFAULT_SEEDS,
    MOLECULE_GATE_COUNTS,
    MOLECULES,
    _analyze_mechanism,
    _compute_exec_time,
    _compute_stats,
    _load_ppo_model,
    generate_report,
)

# ============================================================
# 1. 分子配置完整性
# ============================================================


class TestMoleculeConfig:
    """验证分子配置与原 industry_vqe.py 一致。"""

    def test_molecule_count_is_10(self) -> None:
        """应有 10 种分子（H2 到 F2）。"""
        assert len(MOLECULES) == 10

    def test_gate_counts_match_molecules(self) -> None:
        """MOLECULE_GATE_COUNTS 应覆盖所有分子。"""
        assert set(MOLECULE_GATE_COUNTS.keys()) == set(MOLECULES.keys())

    def test_gate_counts_positive(self) -> None:
        """所有门数应为正整数。"""
        for name, count in MOLECULE_GATE_COUNTS.items():
            assert count > 0, f"{name} gate_count={count} 应为正"

    def test_molecule_fields_complete(self) -> None:
        """每个分子配置应包含必需字段。"""
        required = {"qubits", "reps", "shots", "priority"}
        for name, cfg in MOLECULES.items():
            assert required.issubset(cfg.keys()), f"{name} 缺字段: {cfg}"

    def test_qubits_within_machine_capacity(self) -> None:
        """所有分子比特数应 ≤ 14（天衍-287 单任务上限）。"""
        for name, cfg in MOLECULES.items():
            assert cfg["qubits"] <= 14, f"{name} qubits={cfg['qubits']} 超 14"


# ============================================================
# 2. 执行时间计算
# ============================================================


class TestExecTimeCalc:
    """验证执行时间估算公式。"""

    def test_returns_positive(self) -> None:
        """执行时间应 > 0。"""
        assert _compute_exec_time(100, 1024) > 0

    def test_scales_with_gate_count(self) -> None:
        """执行时间应随门数增加而增加。"""
        t1 = _compute_exec_time(100, 1024)
        t2 = _compute_exec_time(200, 1024)
        assert t2 > t1

    def test_scales_with_shots(self) -> None:
        """执行时间应随 shots 增加而增加。"""
        t1 = _compute_exec_time(100, 1024)
        t2 = _compute_exec_time(100, 4096)
        assert t2 > t1

    def test_zero_gates_returns_min(self) -> None:
        """0 门时返回最小值 0.001。"""
        assert _compute_exec_time(0, 1024) == 0.001


# ============================================================
# 3. 机制分析
# ============================================================


class TestMechanismAnalysis:
    """验证机制分析逻辑（模式 A/B 自动选择）。"""

    def test_mode_a_low_qubit_high_reward(self) -> None:
        """模式 A：PPO 量子利用率低于 FCFS，奖励显著高于 → 高保真偏好 + 排队避免。"""
        rewards = {
            "PPO": [3000.0, 3100.0, 2900.0],
            "FCFS": [1400.0, 1500.0, 1450.0],
            "SJF": [1300.0, 1400.0, 1350.0],
        }
        qubit_utils = {
            "PPO": [0.20, 0.25, 0.22],
            "FCFS": [0.45, 0.46, 0.44],
            "SJF": [0.40, 0.42, 0.41],
        }
        classical_utils = {
            "PPO": [0.55, 0.56, 0.54],
            "FCFS": [0.50, 0.51, 0.49],
            "SJF": [0.48, 0.49, 0.47],
        }
        wait_times = {
            "PPO": [50.0, 55.0, 52.0],
            "FCFS": [40.0, 42.0, 41.0],
            "SJF": [44.0, 45.0, 43.0],
        }

        result = _analyze_mechanism(rewards, qubit_utils, classical_utils, wait_times)

        assert result["mechanism_established"] is True
        assert result["mechanism_mode"].startswith("A")
        assert "高保真偏好" in result["interpretation"]
        assert result["improvements"]["reward_pct"] > 80

    def test_mode_b_high_qubit_high_reward(self) -> None:
        """模式 B：PPO 量子利用率高于 FCFS，奖励显著高于 → 智能资源分配。"""
        rewards = {
            "PPO": [2800.0, 2900.0, 2700.0],
            "FCFS": [1450.0, 1500.0, 1400.0],
            "SJF": [1350.0, 1400.0, 1300.0],
        }
        qubit_utils = {
            "PPO": [0.48, 0.50, 0.46],
            "FCFS": [0.43, 0.44, 0.42],
            "SJF": [0.40, 0.41, 0.39],
        }
        classical_utils = {
            "PPO": [0.55, 0.56, 0.54],
            "FCFS": [0.52, 0.53, 0.51],
            "SJF": [0.48, 0.49, 0.47],
        }
        wait_times = {
            "PPO": [55.0, 60.0, 52.0],
            "FCFS": [40.0, 42.0, 41.0],
            "SJF": [44.0, 45.0, 43.0],
        }

        result = _analyze_mechanism(rewards, qubit_utils, classical_utils, wait_times)

        assert result["mechanism_established"] is True
        assert result["mechanism_mode"].startswith("B")
        assert "智能资源分配" in result["interpretation"]
        assert "异常值" in result["interpretation"]

    def test_mechanism_not_established_low_reward(self) -> None:
        """PPO 奖励未显著高于 FCFS → 机制不成立。"""
        rewards = {
            "PPO": [1500.0, 1600.0, 1400.0],
            "FCFS": [1450.0, 1500.0, 1400.0],
            "SJF": [1350.0, 1400.0, 1300.0],
        }
        qubit_utils = {
            "PPO": [0.48, 0.50, 0.46],
            "FCFS": [0.43, 0.44, 0.42],
            "SJF": [0.40, 0.41, 0.39],
        }
        classical_utils = {
            "PPO": [0.55, 0.56, 0.54],
            "FCFS": [0.52, 0.53, 0.51],
            "SJF": [0.48, 0.49, 0.47],
        }
        wait_times = {
            "PPO": [55.0, 60.0, 52.0],
            "FCFS": [40.0, 42.0, 41.0],
            "SJF": [44.0, 45.0, 43.0],
        }

        result = _analyze_mechanism(rewards, qubit_utils, classical_utils, wait_times)

        assert result["mechanism_established"] is False
        assert "降级" in result["interpretation"]

    def test_evidence_list_has_4_items(self) -> None:
        """证据列表应有 4 条。"""
        rewards = {
            "PPO": [3000.0, 3100.0],
            "FCFS": [1400.0, 1500.0],
            "SJF": [1300.0, 1400.0],
        }
        qubit_utils = {"PPO": [0.2], "FCFS": [0.4], "SJF": [0.3]}
        classical_utils = {"PPO": [0.5], "FCFS": [0.5], "SJF": [0.4]}
        wait_times = {"PPO": [50.0], "FCFS": [40.0], "SJF": [44.0]}

        result = _analyze_mechanism(rewards, qubit_utils, classical_utils, wait_times)
        assert len(result["evidence"]) == 4


# ============================================================
# 4. 统计计算
# ============================================================


class TestStatsComputation:
    """验证统计摘要计算。"""

    def test_descriptive_stats(self) -> None:
        """描述性统计应正确计算均值/std/中位数。"""
        rewards = {
            "PPO": [100.0, 200.0, 300.0],
            "FCFS": [50.0, 60.0, 70.0],
            "SJF": [40.0, 50.0, 60.0],
        }
        stats = _compute_stats(rewards, alpha=0.05)

        assert stats["PPO"]["n"] == 3
        assert stats["PPO"]["mean"] == pytest.approx(200.0)
        assert stats["PPO"]["std"] == pytest.approx(100.0, rel=0.01)
        assert stats["PPO"]["median"] == 200.0
        assert stats["PPO"]["min"] == 100.0
        assert stats["PPO"]["max"] == 300.0

    def test_comparisons_returned(self) -> None:
        """两两比较应返回 3 对（PPO vs FCFS, PPO vs SJF, FCFS vs SJF）。"""
        rewards = {
            "PPO": [100.0, 200.0, 300.0, 400.0],
            "FCFS": [50.0, 60.0, 70.0, 80.0],
            "SJF": [40.0, 50.0, 60.0, 70.0],
        }
        stats = _compute_stats(rewards, alpha=0.05)

        assert len(stats["comparisons"]) == 3
        assert "PPO vs FCFS" in stats["comparisons"]
        assert "PPO vs SJF" in stats["comparisons"]
        assert "FCFS vs SJF" in stats["comparisons"]

    def test_extra_metrics_for_ppo_vs_fcfs(self) -> None:
        """应额外计算 Cohen's d 和 rank-biserial。"""
        rewards = {
            "PPO": [100.0, 200.0, 300.0, 400.0],
            "FCFS": [50.0, 60.0, 70.0, 80.0],
            "SJF": [40.0, 50.0, 60.0, 70.0],
        }
        stats = _compute_stats(rewards, alpha=0.05)

        extra = stats["ppo_vs_fcfs_extra"]
        assert "cohen_d" in extra
        assert "rank_biserial" in extra
        assert "improvement_pct" in extra
        assert extra["improvement_pct"] > 0


# ============================================================
# 5. 报告生成
# ============================================================


class TestReportGeneration:
    """验证报告生成。"""

    def _make_fake_result(self) -> dict:
        """构造一个用于测试的完整结果字典。"""
        return {
            "config": {
                "seeds": [42, 123],
                "episodes": 2,
                "tasks_per_episode": 200,
                "alpha": 0.05,
                "molecules": dict(MOLECULES),
                "gate_counts": dict(MOLECULE_GATE_COUNTS),
                "total_runs": 12,
                "elapsed_seconds": 1.5,
                "timestamp": "2026-07-27T00:00:00",
            },
            "raw_data": {
                "rewards": {
                    "PPO": [2800.0, 2900.0, 2700.0, 2600.0],
                    "FCFS": [1450.0, 1500.0, 1400.0, 1350.0],
                    "SJF": [1350.0, 1400.0, 1300.0, 1250.0],
                },
                "qubit_utils": {
                    "PPO": [0.48, 0.50, 0.46, 0.44],
                    "FCFS": [0.43, 0.44, 0.42, 0.41],
                    "SJF": [0.40, 0.41, 0.39, 0.38],
                },
                "classical_utils": {
                    "PPO": [0.55, 0.56, 0.54, 0.53],
                    "FCFS": [0.52, 0.53, 0.51, 0.50],
                    "SJF": [0.48, 0.49, 0.47, 0.46],
                },
                "wait_times": {
                    "PPO": [55.0, 60.0, 52.0, 50.0],
                    "FCFS": [40.0, 42.0, 41.0, 39.0],
                    "SJF": [44.0, 45.0, 43.0, 42.0],
                },
            },
            "stats": _compute_stats(
                {
                    "PPO": [2800.0, 2900.0, 2700.0, 2600.0],
                    "FCFS": [1450.0, 1500.0, 1400.0, 1350.0],
                    "SJF": [1350.0, 1400.0, 1300.0, 1250.0],
                },
                alpha=0.05,
            ),
            "mechanism": _analyze_mechanism(
                {
                    "PPO": [2800.0, 2900.0, 2700.0, 2600.0],
                    "FCFS": [1450.0, 1500.0, 1400.0, 1350.0],
                    "SJF": [1350.0, 1400.0, 1300.0, 1250.0],
                },
                {
                    "PPO": [0.48, 0.50, 0.46, 0.44],
                    "FCFS": [0.43, 0.44, 0.42, 0.41],
                    "SJF": [0.40, 0.41, 0.39, 0.38],
                },
                {
                    "PPO": [0.55, 0.56, 0.54, 0.53],
                    "FCFS": [0.52, 0.53, 0.51, 0.50],
                    "SJF": [0.48, 0.49, 0.47, 0.46],
                },
                {
                    "PPO": [55.0, 60.0, 52.0, 50.0],
                    "FCFS": [40.0, 42.0, 41.0, 39.0],
                    "SJF": [44.0, 45.0, 43.0, 42.0],
                },
            ),
        }

    def test_generates_markdown_and_json(self, tmp_path: Path) -> None:
        """应生成 .md 和 .json 两个文件。"""
        result = self._make_fake_result()
        report_path, json_path = generate_report(result, output_dir=tmp_path)

        assert os.path.exists(report_path)
        assert os.path.exists(json_path)
        assert report_path.endswith(".md")
        assert json_path.endswith(".json")

    def test_report_contains_required_sections(self, tmp_path: Path) -> None:
        """报告应包含所有必需章节。"""
        result = self._make_fake_result()
        report_path, _ = generate_report(result, output_dir=tmp_path)

        content = Path(report_path).read_text(encoding="utf-8")
        required_sections = [
            "# VQE 行业场景多 seed 评估报告",
            "## 1. 背景与动机",
            "## 2. 分子清单",
            "## 3. 实验配置",
            "## 4. 实验结果",
            "### 4.1 描述性统计",
            "### 4.2 统计显著性检验",
            "## 5. 机制分析",
            "### 5.1 现象描述",
            "### 5.2 策略机制",
            "### 5.3 证据验证",
            "### 5.4 结论",
            "## 6. 结论",
            "Issue #462",
        ]
        for section in required_sections:
            assert section in content, f"缺少章节: {section}"

    def test_report_contains_n_value(self, tmp_path: Path) -> None:
        """报告应标注 N 值。"""
        result = self._make_fake_result()
        report_path, _ = generate_report(result, output_dir=tmp_path)

        content = Path(report_path).read_text(encoding="utf-8")
        assert "N=4" in content or "N = 4" in content

    def test_report_contains_mechanism_mode(self, tmp_path: Path) -> None:
        """报告应包含机制模式标识。"""
        result = self._make_fake_result()
        report_path, _ = generate_report(result, output_dir=tmp_path)

        content = Path(report_path).read_text(encoding="utf-8")
        assert "机制模式" in content
        # 报告中模式格式为 "A: ..." 或 "B: ..."
        assert "A:" in content or "B:" in content

    def test_json_contains_complete_data(self, tmp_path: Path) -> None:
        """JSON 应包含完整的原始数据。"""
        import json

        result = self._make_fake_result()
        _, json_path = generate_report(result, output_dir=tmp_path)

        data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        assert "config" in data
        assert "raw_data" in data
        assert "stats" in data
        assert "mechanism" in data
        assert "rewards" in data["raw_data"]
        assert len(data["raw_data"]["rewards"]["PPO"]) == 4


# ============================================================
# 6. 默认种子配置
# ============================================================


class TestDefaultSeeds:
    """验证默认种子配置。"""

    def test_default_seeds_count_is_10(self) -> None:
        """默认种子列表应有 10 个。"""
        assert len(DEFAULT_SEEDS) == 10

    def test_default_seeds_unique(self) -> None:
        """默认种子应互不相同。"""
        assert len(set(DEFAULT_SEEDS)) == 10

    def test_default_seeds_positive(self) -> None:
        """所有种子应为正整数。"""
        for s in DEFAULT_SEEDS:
            assert isinstance(s, int)
            assert s > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
