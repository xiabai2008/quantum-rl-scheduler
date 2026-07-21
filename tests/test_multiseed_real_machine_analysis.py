"""多seed真机实验统计分析脚本测试

测试 scripts/evaluation/multiseed_real_machine_analysis.py 的核心功能：
- 数据分析（描述性统计、正态性检验、两两比较）
- 效应量判定逻辑
- 报告生成
- Cohen's d 等级划分
- Welch t 检验 + CI 计算
"""

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.multiseed_real_machine_analysis import (
    ALPHA_BONFERRONI,
    COMPARISONS,
    N_COMPARISONS,
    STRATEGIES,
    _effect_level,
    _welch_mean_diff_ci,
    analyze,
    generate_report,
)

# ── 测试数据 ──

SAMPLE_DATA = {
    "experiment": "tianyan287_multiseed",
    "timestamp": "2026-07-21T21:55:16",
    "config": {
        "seeds": [42, 123, 456, 789, 1024],
        "strategies": ["ppo", "fcfs", "sjf"],
        "num_tasks": 32,
        "max_real_tasks_per_run": 1,
        "shots": 1024,
    },
    "total_elapsed_seconds": 536.16,
    "results": [
        {"strategy": "PPO", "seed": 42, "metrics": {"total_reward": 1560.86}},
        {"strategy": "FCFS", "seed": 42, "metrics": {"total_reward": 410.23}},
        {"strategy": "SJF", "seed": 42, "metrics": {"total_reward": 704.32}},
        {"strategy": "PPO", "seed": 123, "metrics": {"total_reward": 1224.13}},
        {"strategy": "FCFS", "seed": 123, "metrics": {"total_reward": 288.77}},
        {"strategy": "SJF", "seed": 123, "metrics": {"total_reward": 502.91}},
        {"strategy": "PPO", "seed": 456, "metrics": {"total_reward": 1615.17}},
        {"strategy": "FCFS", "seed": 456, "metrics": {"total_reward": 317.66}},
        {"strategy": "SJF", "seed": 456, "metrics": {"total_reward": 854.43}},
        {"strategy": "PPO", "seed": 789, "metrics": {"total_reward": 2097.05}},
        {"strategy": "FCFS", "seed": 789, "metrics": {"total_reward": 404.57}},
        {"strategy": "SJF", "seed": 789, "metrics": {"total_reward": 390.39}},
        {"strategy": "PPO", "seed": 1024, "metrics": {"total_reward": 1828.90}},
        {"strategy": "FCFS", "seed": 1024, "metrics": {"total_reward": 344.85}},
        {"strategy": "SJF", "seed": 1024, "metrics": {"total_reward": 383.93}},
    ],
}


@pytest.fixture
def tmp_data_file(tmp_path: Path) -> Path:
    """创建临时数据文件。"""
    data_file = tmp_path / "test_multiseed_data.json"
    with data_file.open("w", encoding="utf-8") as f:
        json.dump(SAMPLE_DATA, f)
    return data_file


# ── 常量测试 ──


class TestConstants:
    """测试常量定义。"""

    def test_strategies_order(self) -> None:
        """策略顺序应为 PPO, SJF, FCFS。"""
        assert STRATEGIES == ["PPO", "SJF", "FCFS"]

    def test_comparisons_count(self) -> None:
        """比较对数量应为 3。"""
        assert len(COMPARISONS) == 3

    def test_bonferroni_alpha(self) -> None:
        """Bonferroni 校正 α 应为 0.05/3 ≈ 0.0167。"""
        assert N_COMPARISONS == 3
        assert pytest.approx(0.01667, rel=0.01) == ALPHA_BONFERRONI


# ── 效应量等级测试 ──


class TestEffectLevel:
    """测试 Cohen's d 效应量等级划分。"""

    def test_negligible(self) -> None:
        """d < 0.2 为可忽略。"""
        assert _effect_level(0.1) == "可忽略"

    def test_small(self) -> None:
        """0.2 ≤ d < 0.5 为小效应。"""
        assert _effect_level(0.3) == "小效应"

    def test_medium(self) -> None:
        """0.5 ≤ d < 0.8 为中效应。"""
        assert _effect_level(0.6) == "中效应"

    def test_large(self) -> None:
        """d ≥ 0.8 为大效应。"""
        assert _effect_level(0.8) == "大效应"
        assert _effect_level(1.0) == "大效应"
        assert _effect_level(5.0) == "大效应"


# ── Welch t 检验 + CI 测试 ──


class TestWelchMeanDiffCI:
    """测试 Welch t 检验 + 均值差 CI 计算。"""

    def test_identical_groups(self) -> None:
        """相同组均值差应为 0。"""
        a = [100.0, 200.0, 300.0]
        b = [100.0, 200.0, 300.0]
        mean_diff, _t_stat, _p_value, _ci_lo, _ci_hi = _welch_mean_diff_ci(a, b)
        assert mean_diff == pytest.approx(0.0)

    def test_different_groups(self) -> None:
        """不同组均值差应正确。"""
        a = [1000.0, 1100.0, 1200.0]
        b = [100.0, 200.0, 300.0]
        mean_diff, t_stat, p_value, ci_lo, ci_hi = _welch_mean_diff_ci(a, b)
        assert mean_diff == pytest.approx(900.0)
        assert t_stat > 0
        assert p_value < 0.05
        assert ci_lo > 0
        assert ci_hi > ci_lo

    def test_ci_contains_mean_diff(self) -> None:
        """均值差应在 CI 范围内。"""
        a = [10.0, 20.0, 30.0, 40.0, 50.0]
        b = [5.0, 15.0, 25.0, 35.0, 45.0]
        mean_diff, _, _, ci_lo, ci_hi = _welch_mean_diff_ci(a, b)
        assert ci_lo < mean_diff < ci_hi


# ── 分析函数测试 ──


class TestAnalyze:
    """测试 analyze 函数。"""

    def test_descriptive_stats(self) -> None:
        """描述性统计应正确计算。"""
        analysis = analyze(SAMPLE_DATA)
        desc = analysis["descriptive"]
        assert desc["PPO"]["n"] == 5
        assert desc["PPO"]["mean"] == pytest.approx(1665.22, rel=0.01)
        assert desc["FCFS"]["mean"] == pytest.approx(353.22, rel=0.01)
        assert desc["SJF"]["mean"] == pytest.approx(567.20, rel=0.01)

    def test_normality(self) -> None:
        """正态性检验应返回结果。"""
        analysis = analyze(SAMPLE_DATA)
        norm = analysis["normality"]
        for name in STRATEGIES:
            assert name in norm
            assert "test" in norm[name]
            assert "p_value" in norm[name]
            assert "is_normal" in norm[name]

    def test_pairwise_comparisons(self) -> None:
        """两两比较应包含所有比较对。"""
        analysis = analyze(SAMPLE_DATA)
        pairwise = analysis["pairwise"]
        assert "PPO_vs_FCFS" in pairwise
        assert "PPO_vs_SJF" in pairwise
        assert "SJF_vs_FCFS" in pairwise

    def test_ppo_dominates(self) -> None:
        """PPO 应在所有比较中胜出。"""
        analysis = analyze(SAMPLE_DATA)
        ppo_fcfs = analysis["pairwise"]["PPO_vs_FCFS"]
        ppo_sjf = analysis["pairwise"]["PPO_vs_SJF"]
        assert ppo_fcfs["cohens_d"] > 0.8  # 大效应
        assert ppo_sjf["cohens_d"] > 0.8
        assert ppo_fcfs["judgment"] == "支持"
        assert ppo_sjf["judgment"] == "支持"

    def test_full_stats_present(self) -> None:
        """完整统计比较结果应存在。"""
        analysis = analyze(SAMPLE_DATA)
        assert "full_stats" in analysis
        assert len(analysis["full_stats"]) > 0


# ── 报告生成测试 ──


class TestGenerateReport:
    """测试报告生成。"""

    def test_report_contains_sections(self) -> None:
        """报告应包含所有章节。"""
        analysis = analyze(SAMPLE_DATA)
        report = generate_report(analysis, Path("test_data.json"))
        assert "# 天衍-287 多seed真机实验统计分析报告" in report
        assert "## 1. 描述性统计" in report
        assert "## 2. 正态性检验" in report
        assert "## 3. 两两比较" in report
        assert "## 4. 汇总表" in report
        assert "## 5. compare_strategies 完整输出" in report
        assert "## 6. 结论" in report

    def test_report_contains_key_numbers(self) -> None:
        """报告应包含关键数字。"""
        analysis = analyze(SAMPLE_DATA)
        report = generate_report(analysis, Path("test_data.json"))
        assert "Cohen's d" in report
        assert "Bonferroni" in report
        assert "95% CI" in report

    def test_report_contains_conclusions(self) -> None:
        """报告结论应包含 PPO 胜出判定。"""
        analysis = analyze(SAMPLE_DATA)
        report = generate_report(analysis, Path("test_data.json"))
        assert "支持" in report
        assert "PPO" in report
