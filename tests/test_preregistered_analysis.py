"""预注册真机闭环分析脚本测试

测试 scripts/evaluation/preregistered_real_machine_analysis.py 的核心功能：
- 数据加载
- 数据清洗（4σ 异常值剔除）
- 两两比较（效应量 + CI + p 值）
- 假设判定逻辑
- 报告生成
"""

import json
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))

from scripts.evaluation.preregistered_real_machine_analysis import (
    BONFERRONI_ALPHA,
    CONDITION_RESULT_AWARE,
    CONDITION_SHUFFLED,
    CONDITION_STATUS_ONLY,
    HYPOTHESES,
    MAX_SAMPLE_SIZE,
    MIN_SAMPLE_SIZE,
    OUTLIER_SIGMA,
    CleaningRecord,
    ComparisonResult,
    _effect_level_cohen_d,
    _judge_hypothesis,
    clean_data,
    generate_report,
    load_data,
    run_comparisons,
)

# ── 测试数据 ──

SAMPLE_DATA = {
    CONDITION_STATUS_ONLY: [1450.0 + i * 0.5 for i in range(18)],
    CONDITION_RESULT_AWARE: [1550.0 + i * 0.5 for i in range(18)],
    CONDITION_SHUFFLED: [1450.3 + i * 0.5 for i in range(18)],
}

SAMPLE_DATA_WITH_OUTLIER = {
    CONDITION_STATUS_ONLY: [1450.0 + i * 0.5 for i in range(17)] + [5000.0],
    CONDITION_RESULT_AWARE: [1550.0 + i * 0.5 for i in range(18)],
    CONDITION_SHUFFLED: [1450.3 + i * 0.5 for i in range(18)],
}


@pytest.fixture
def tmp_data_file(tmp_path: Path) -> Path:
    """创建临时数据文件"""
    data_file = tmp_path / "test_data.json"
    with data_file.open("w", encoding="utf-8") as f:
        json.dump(SAMPLE_DATA, f)
    return data_file


@pytest.fixture
def tmp_outlier_file(tmp_path: Path) -> Path:
    """创建含异常值的临时数据文件"""
    data_file = tmp_path / "test_outlier_data.json"
    with data_file.open("w", encoding="utf-8") as f:
        json.dump(SAMPLE_DATA_WITH_OUTLIER, f)
    return data_file


# ── 常量测试 ──


class TestPreregistrationConstants:
    """测试预注册常量"""

    def test_min_sample_size(self) -> None:
        """最低样本量应为 18"""
        assert MIN_SAMPLE_SIZE == 18

    def test_max_sample_size(self) -> None:
        """最高样本量应为 30"""
        assert MAX_SAMPLE_SIZE == 30

    def test_outlier_sigma(self) -> None:
        """异常值剔除标准应为 4σ"""
        assert OUTLIER_SIGMA == 4.0

    def test_bonferroni_alpha(self) -> None:
        """Bonferroni 校正 α 应为 0.05/3 ≈ 0.0167"""
        assert pytest.approx(0.05 / 3, rel=1e-4) == BONFERRONI_ALPHA

    def test_hypotheses_count(self) -> None:
        """应有 3 个假设"""
        assert len(HYPOTHESES) == 3

    def test_conditions(self) -> None:
        """应有 3 个条件"""
        conditions = {CONDITION_STATUS_ONLY, CONDITION_RESULT_AWARE, CONDITION_SHUFFLED}
        assert len(conditions) == 3


# ── 数据加载测试 ──


class TestLoadData:
    """测试数据加载"""

    def test_load_valid_data(self, tmp_data_file: Path) -> None:
        """正常加载三条件数据"""
        data = load_data(tmp_data_file)
        assert CONDITION_STATUS_ONLY in data
        assert CONDITION_RESULT_AWARE in data
        assert CONDITION_SHUFFLED in data
        assert len(data[CONDITION_STATUS_ONLY]) == 18

    def test_load_missing_condition(self, tmp_path: Path) -> None:
        """缺少条件应报错"""
        bad_file = tmp_path / "bad_data.json"
        with bad_file.open("w", encoding="utf-8") as f:
            json.dump({CONDITION_STATUS_ONLY: [1.0]}, f)
        with pytest.raises(Exception, match="缺少预注册条件"):
            load_data(bad_file)

    def test_load_non_dict(self, tmp_path: Path) -> None:
        """非字典格式应报错"""
        bad_file = tmp_path / "bad_data.json"
        with bad_file.open("w", encoding="utf-8") as f:
            json.dump([1, 2, 3], f)
        with pytest.raises(Exception, match="顶层必须是对象"):
            load_data(bad_file)


# ── 数据清洗测试 ──


class TestCleanData:
    """测试数据清洗"""

    def test_clean_normal_data(self) -> None:
        """正常数据不应被剔除"""
        cleaned, records = clean_data(SAMPLE_DATA)
        for condition in [CONDITION_STATUS_ONLY, CONDITION_RESULT_AWARE, CONDITION_SHUFFLED]:
            assert len(cleaned[condition]) == 18
            rec = next(r for r in records if r.condition == condition)
            assert rec.removed_count == 0
            assert rec.final_count == 18

    def test_clean_outlier_removal(self) -> None:
        """4σ 异常值应被剔除"""
        _, records = clean_data(SAMPLE_DATA_WITH_OUTLIER)
        rec = next(r for r in records if r.condition == CONDITION_STATUS_ONLY)
        assert rec.removed_count == 1
        assert 5000.0 in rec.removed_outliers
        assert rec.final_count == 17

    def test_clean_preserves_other_conditions(self) -> None:
        """剔除异常值不影响其他条件"""
        cleaned, _ = clean_data(SAMPLE_DATA_WITH_OUTLIER)
        assert len(cleaned[CONDITION_RESULT_AWARE]) == 18
        assert len(cleaned[CONDITION_SHUFFLED]) == 18

    def test_clean_empty_data(self) -> None:
        """空数据应正确处理"""
        data = {
            CONDITION_STATUS_ONLY: [],
            CONDITION_RESULT_AWARE: [],
            CONDITION_SHUFFLED: [],
        }
        _, records = clean_data(data)
        for rec in records:
            assert rec.original_count == 0
            assert rec.final_count == 0


# ── 效应量等级测试 ──


class TestEffectLevel:
    """测试 Cohen's d 效应量等级判定"""

    def test_negligible(self) -> None:
        assert _effect_level_cohen_d(0.1) == "可忽略"

    def test_small(self) -> None:
        assert _effect_level_cohen_d(0.3) == "小效应"

    def test_medium(self) -> None:
        assert _effect_level_cohen_d(0.6) == "中效应"

    def test_large(self) -> None:
        assert _effect_level_cohen_d(1.0) == "大效应"

    def test_nan(self) -> None:
        assert _effect_level_cohen_d(float("nan")) == "无法计算"

    def test_negative(self) -> None:
        assert _effect_level_cohen_d(-0.6) == "中效应"


# ── 假设判定测试 ──


class TestJudgeHypothesis:
    """测试假设判定逻辑"""

    def test_h1_supported(self) -> None:
        """H1: d≥0.5 且 CI 下界>0 → 支持"""
        assert _judge_hypothesis("H1", 0.8, 10.0, 20.0) == "支持"

    def test_h1_not_supported_small_d(self) -> None:
        """H1: d<0.2 → 不支持"""
        assert _judge_hypothesis("H1", 0.1, 10.0, 20.0) == "不支持"

    def test_h1_not_supported_ci_crosses_zero(self) -> None:
        """H1: CI 跨0 → 不支持"""
        assert _judge_hypothesis("H1", 0.8, -5.0, 10.0) == "不支持"

    def test_h1_uncertain_middle_range(self) -> None:
        """H1: 0.2≤d<0.5 且 CI 不跨0 → 不确定"""
        assert _judge_hypothesis("H1", 0.3, 5.0, 15.0) == "不确定，需更多数据"

    def test_h1_nan_ci_with_large_d(self) -> None:
        """H1: CI 为 NaN 但 d≥0.5 → 不确定（CI 无法计算）"""
        result = _judge_hypothesis("H1", 0.8, float("nan"), float("nan"))
        assert "不确定" in result

    def test_h2_supported(self) -> None:
        """H2: d≥0.5 且 CI 下界>0 → 支持"""
        assert _judge_hypothesis("H2", 0.8, 10.0, 20.0) == "支持"

    def test_h3_supported_negligible(self) -> None:
        """H3: |d|<0.2 → 支持"""
        assert _judge_hypothesis("H3", 0.1, -5.0, 5.0) == "支持"

    def test_h3_not_supported_large_d(self) -> None:
        """H3: |d|≥0.5 → 不支持"""
        assert _judge_hypothesis("H3", 0.6, 10.0, 20.0) == "不支持"

    def test_h3_uncertain_middle_range(self) -> None:
        """H3: 0.2≤|d|<0.5 → 不确定"""
        assert _judge_hypothesis("H3", 0.3, 5.0, 15.0) == "不确定，需更多数据"

    def test_nan_d(self) -> None:
        """d 为 NaN → 无法判定"""
        result = _judge_hypothesis("H1", float("nan"), 10.0, 20.0)
        assert "无法判定" in result


# ── 两两比较测试 ──


class TestRunComparisons:
    """测试两两比较"""

    def test_all_three_comparisons(self) -> None:
        """应生成 3 个比较结果"""
        results = run_comparisons(SAMPLE_DATA)
        assert len(results) == 3

    def test_comparison_order(self) -> None:
        """比较顺序应与 HYPOTHESES 一致"""
        results = run_comparisons(SAMPLE_DATA)
        assert results[0].hypothesis == "H1"
        assert results[1].hypothesis == "H2"
        assert results[2].hypothesis == "H3"

    def test_large_effect_detected(self) -> None:
        """大效应量应被正确检测"""
        results = run_comparisons(SAMPLE_DATA)
        h1 = next(r for r in results if r.hypothesis == "H1")
        assert h1.cohen_d > 0.5
        assert h1.effect_level == "大效应"

    def test_ci_not_nan(self) -> None:
        """CI 不应为 NaN（样本充足时）"""
        results = run_comparisons(SAMPLE_DATA)
        for r in results:
            assert not math.isnan(r.ci_lower)
            assert not math.isnan(r.ci_upper)

    def test_p_value_not_nan(self) -> None:
        """p 值不应为 NaN（样本充足时）"""
        results = run_comparisons(SAMPLE_DATA)
        for r in results:
            assert not math.isnan(r.p_value)

    def test_insufficient_samples(self) -> None:
        """样本不足应正确处理"""
        data = {
            CONDITION_STATUS_ONLY: [1.0],
            CONDITION_RESULT_AWARE: [2.0],
            CONDITION_SHUFFLED: [3.0],
        }
        results = run_comparisons(data)
        for r in results:
            assert math.isnan(r.cohen_d)
            assert r.test_name == "样本不足"

    def test_h3_negligible_effect(self) -> None:
        """H3: status_only 和 shuffled 相近 → |d| 应很小"""
        results = run_comparisons(SAMPLE_DATA)
        h3 = next(r for r in results if r.hypothesis == "H3")
        assert abs(h3.cohen_d) < 0.2


# ── 报告生成测试 ──


class TestGenerateReport:
    """测试报告生成"""

    def test_report_contains_required_sections(self) -> None:
        """报告应包含所有必需章节"""
        cleaned, records = clean_data(SAMPLE_DATA)
        comparisons = run_comparisons(cleaned)
        report = generate_report(
            data=SAMPLE_DATA,
            cleaned_data=cleaned,
            cleaning_records=records,
            comparisons=comparisons,
            input_path=Path("test.json"),
        )
        assert "实验概述" in report
        assert "数据清洗报告" in report
        assert "描述性统计" in report
        assert "正态性检验" in report
        assert "效应量报告" in report
        assert "假设检验报告" in report
        assert "假设判定" in report
        assert "功效分析" in report
        assert "探索性分析" in report
        assert "偏离声明" in report

    def test_report_contains_hypothesis_judgments(self) -> None:
        """报告应包含假设判定结果"""
        cleaned, records = clean_data(SAMPLE_DATA)
        comparisons = run_comparisons(cleaned)
        report = generate_report(
            data=SAMPLE_DATA,
            cleaned_data=cleaned,
            cleaning_records=records,
            comparisons=comparisons,
            input_path=Path("test.json"),
        )
        assert "H1" in report
        assert "H2" in report
        assert "H3" in report
        assert "支持" in report or "不支持" in report or "不确定" in report

    def test_report_contains_cohen_d(self) -> None:
        """报告应包含 Cohen's d 值"""
        cleaned, records = clean_data(SAMPLE_DATA)
        comparisons = run_comparisons(cleaned)
        report = generate_report(
            data=SAMPLE_DATA,
            cleaned_data=cleaned,
            cleaning_records=records,
            comparisons=comparisons,
            input_path=Path("test.json"),
        )
        assert "Cohen" in report

    def test_report_contains_ci(self) -> None:
        """报告应包含 95% CI"""
        cleaned, records = clean_data(SAMPLE_DATA)
        comparisons = run_comparisons(cleaned)
        report = generate_report(
            data=SAMPLE_DATA,
            cleaned_data=cleaned,
            cleaning_records=records,
            comparisons=comparisons,
            input_path=Path("test.json"),
        )
        assert "CI" in report

    def test_report_contains_bonferroni(self) -> None:
        """报告应包含 Bonferroni 校正"""
        cleaned, records = clean_data(SAMPLE_DATA)
        comparisons = run_comparisons(cleaned)
        report = generate_report(
            data=SAMPLE_DATA,
            cleaned_data=cleaned,
            cleaning_records=records,
            comparisons=comparisons,
            input_path=Path("test.json"),
        )
        assert "Bonferroni" in report
