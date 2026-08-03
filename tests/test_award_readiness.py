"""tests/test_award_readiness.py — 冲奖就绪度检查核心逻辑单元测试"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.ci.award_readiness_check import (
    AwardRoadmapParser,
    DimensionScore,
    EvidenceChecker,
    EvidenceItem,
    ReadinessReportGenerator,
    ReadinessResult,
    RoadmapTask,
)

# ---------------------------------------------------------------------------
# Mock 数据：模拟 award_roadmap.md 内容
# ---------------------------------------------------------------------------

MOCK_ROADMAP = """# 冲奖路线图

## 1. 评审维度现状评估

### 1.1 五维评分（基于比赛方案）

| 评审维度 | 当前得分 | 目标得分 | 差距 | 核心短板 |
|---------|---------|---------|------|---------|
| 主题契合度 | 85/100 | 90+ | -5 | 双向赋能闭环证据不足 |
| 技术创新性 | 82/100 | 90+ | -8 | 退火加速统计不显著 |
| 方案可行性 | 78/100 | 88+ | -10 | 真机性能验证未完成 |
| 落地与价值 | 68/100 | 85+ | -17 | 技术导向转价值导向不足 |
| 验证严谨性 | 75/100 | 88+ | -13 | 真机样本量不足 |
| **综合** | **~78/100** | **88+** | **-10** | — |

### 1.2 评分依据

一些说明文字。

---

## 2. P0 冲奖必做

### 2.1 落地与价值提升（+15 分目标）

| 任务 | 负责人 | 状态 | 预期提升 |
|------|--------|------|---------|
| 落地价值深度量化 | NN2914 | ✅ 完成（docs/value_quantification.md） | +5 |
| 平台落地商业分析 | NN2914 | ✅ 完成 | +3 |
| PPT/白皮书 v3→v4 | 瑞哥 | 📋 待手动更新 | +2 |

### 2.2 技术创新补强

| 任务 | 负责人 | 状态 | 预期提升 |
|------|--------|------|---------|
| 16 维 DQN 重训 | qpqpalalzmzm112 | 🔄 PR #68 Open | +3 |
| 量子占比敏感性 | qpqpalalzmzm112 | 📋 Open #67 | +2 |

### 2.3 可行性补强

| 任务 | 负责人 | 状态 | 预期提升 |
|------|--------|------|---------|
| 真机闭环统计显著性 | DUMNOX | ✅ 完成（10 seeds） | +3 |

---

## 3. P1 冲奖加分

### 3.1 验证严谨性提升

| 任务 | 负责人 | 状态 | 价值 |
|------|--------|------|------|
| 50 seed 统计扩展 | Jackhock-1 | 📋 Open #41 | 置信区间权威性 |
| 多用户公平性 | heka-ky | ✅ 完成（#587） | 公平调度 |

### 3.2 工程深度

| 任务 | 负责人 | 状态 | 价值 |
|------|--------|------|------|
| DAG 工作流调度 | qpqpalalzmzm112 | 📋 Open #32 | 工作流支持 |

---

## 4. P2 远期探索

| 任务 | 负责人 | 状态 | 价值 |
|------|--------|------|------|
| 论文级成果包装 | — | 📋 Open #10 | 学术影响力 |
| CI 流水线优化 | heka-ky | 📋 Open #50 | 工程效率 |

---

## 5. 关键数字锁定

一些内容。
"""


# ---------------------------------------------------------------------------
# AwardRoadmapParser 解析器测试
# ---------------------------------------------------------------------------


class TestAwardRoadmapParserDimensionScores:
    """AwardRoadmapParser 五维评分表解析测试"""

    def test_parses_five_dimensions(self) -> None:
        """应正确解析出 5 个评审维度（排除综合行）。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        scores = parser.parse_dimension_scores()
        assert len(scores) == 5
        assert "主题契合度" in scores
        assert "技术创新性" in scores
        assert "方案可行性" in scores
        assert "落地与价值" in scores
        assert "验证严谨性" in scores

    def test_excludes_summary_row(self) -> None:
        """应排除「综合」行。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        scores = parser.parse_dimension_scores()
        assert all("综合" not in name for name in scores)

    def test_current_score_parsed_correctly(self) -> None:
        """当前得分应从 X/100 格式中正确提取。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        scores = parser.parse_dimension_scores()
        assert scores["主题契合度"].current_score == 85.0
        assert scores["技术创新性"].current_score == 82.0
        assert scores["验证严谨性"].current_score == 75.0

    def test_target_score_parsed_correctly(self) -> None:
        """目标得分应从 X+ 格式中正确提取。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        scores = parser.parse_dimension_scores()
        assert scores["主题契合度"].target_score == 90.0
        assert scores["落地与价值"].target_score == 85.0

    def test_gap_parsed_correctly(self) -> None:
        """差距应正确解析为负数。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        scores = parser.parse_dimension_scores()
        assert scores["主题契合度"].gap == -5.0
        assert scores["落地与价值"].gap == -17.0

    def test_core_weakness_preserved(self) -> None:
        """核心短板文本应保留。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        scores = parser.parse_dimension_scores()
        assert "双向赋能" in scores["主题契合度"].core_weakness

    def test_strips_bold_markers(self) -> None:
        """应移除单元格中的 ** 加粗标记。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        scores = parser.parse_dimension_scores()
        for name in scores:
            assert "**" not in name
            assert "**" not in scores[name].core_weakness


class TestAwardRoadmapParserTasks:
    """AwardRoadmapParser P0/P1/P2 任务清单解析测试"""

    def test_parses_all_priorities(self) -> None:
        """应解析出 P0、P1、P2 三个优先级的任务。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        tasks = parser.parse_tasks()
        priorities = {t.priority for t in tasks}
        assert "P0" in priorities
        assert "P1" in priorities
        assert "P2" in priorities

    def test_p0_task_count(self) -> None:
        """P0 任务数量应正确。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        tasks = parser.parse_tasks()
        p0_tasks = [t for t in tasks if t.priority == "P0"]
        assert len(p0_tasks) == 6

    def test_p1_task_count(self) -> None:
        """P1 任务数量应正确。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        tasks = parser.parse_tasks()
        p1_tasks = [t for t in tasks if t.priority == "P1"]
        assert len(p1_tasks) == 3

    def test_p2_task_count(self) -> None:
        """P2 任务数量应正确。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        tasks = parser.parse_tasks()
        p2_tasks = [t for t in tasks if t.priority == "P2"]
        assert len(p2_tasks) == 2

    def test_task_category_assigned(self) -> None:
        """任务应正确分配所属分类。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        tasks = parser.parse_tasks()
        p0_value_tasks = [t for t in tasks if t.priority == "P0" and "价值" in t.category]
        assert len(p0_value_tasks) == 3

    def test_completed_tasks_detected(self) -> None:
        """✅ 标记的任务应被识别为已完成。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        tasks = parser.parse_tasks()
        completed = [t for t in tasks if t.is_completed]
        assert len(completed) >= 4  # P0 有 4 个完成的 + P1 有 1 个
        assert any("落地价值深度量化" in t.name for t in completed)

    def test_in_progress_tasks_detected(self) -> None:
        """🔄/Open 标记的任务应被识别为进行中。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        tasks = parser.parse_tasks()
        in_progress = [t for t in tasks if t.is_in_progress]
        assert len(in_progress) >= 1
        assert any("DQN 重训" in t.name for t in in_progress)

    def test_pending_tasks_detected(self) -> None:
        """📋 标记的任务应被识别为待开始。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        tasks = parser.parse_tasks()
        pending = [
            t for t in tasks if not t.is_completed and not t.is_in_progress and t.status == "待开始"
        ]
        assert len(pending) >= 3

    def test_assignee_preserved(self) -> None:
        """负责人信息应保留。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        tasks = parser.parse_tasks()
        task = next(t for t in tasks if "落地价值深度量化" in t.name)
        assert task.assignee == "NN2914"

    def test_expected_improvement_preserved(self) -> None:
        """预期提升信息应保留。"""
        parser = AwardRoadmapParser(MOCK_ROADMAP)
        tasks = parser.parse_tasks()
        task = next(t for t in tasks if "落地价值深度量化" in t.name)
        assert "+5" in task.expected_improvement


class TestAwardRoadmapParserHelpers:
    """AwardRoadmapParser 辅助方法测试"""

    def test_parse_score_slash_format(self) -> None:
        """_parse_score 应正确解析 X/100 格式。"""
        assert AwardRoadmapParser._parse_score("85/100") == 85.0
        assert AwardRoadmapParser._parse_score("78.5/100") == 78.5

    def test_parse_score_plus_format(self) -> None:
        """_parse_score 应正确解析 X+ 格式。"""
        assert AwardRoadmapParser._parse_score("90+") == 90.0
        assert AwardRoadmapParser._parse_score("88+") == 88.0

    def test_parse_score_negative(self) -> None:
        """_parse_score 应正确解析负数。"""
        assert AwardRoadmapParser._parse_score("-5") == -5.0
        assert AwardRoadmapParser._parse_score("-17") == -17.0

    def test_parse_score_plain_number(self) -> None:
        """_parse_score 应正确解析纯数字。"""
        assert AwardRoadmapParser._parse_score("85") == 85.0
        assert AwardRoadmapParser._parse_score("78.5") == 78.5

    def test_parse_score_with_tilde(self) -> None:
        """_parse_score 应忽略 ~ 符号。"""
        assert AwardRoadmapParser._parse_score("~78") == 78.0

    def test_parse_score_with_bold_markers(self) -> None:
        """_parse_score 应移除 ** 加粗标记。"""
        assert AwardRoadmapParser._parse_score("**75/100**") == 75.0

    def test_parse_score_empty_string(self) -> None:
        """_parse_score 对空字符串应返回 0。"""
        assert AwardRoadmapParser._parse_score("") == 0.0

    def test_parse_status_checkmark_icon(self) -> None:
        """_parse_status 应识别 ✅ 为完成。"""
        assert AwardRoadmapParser._parse_status("✅ 完成") == "完成"

    def test_parse_status_clipboard_icon(self) -> None:
        """_parse_status 应识别 📋 为待开始。"""
        assert AwardRoadmapParser._parse_status("📋 Open #850") == "待开始"

    def test_parse_status_recycle_icon(self) -> None:
        """_parse_status 应识别 🔄 为进行中。"""
        assert AwardRoadmapParser._parse_status("🔄 PR #68 Open") == "进行中"

    def test_parse_status_calendar_icon(self) -> None:
        """_parse_status 应识别 📅 为计划中。"""
        assert AwardRoadmapParser._parse_status("📅 2026-08-15") == "计划中"

    def test_parse_status_text_keywords(self) -> None:
        """_parse_status 应通过文本关键词识别状态。"""
        assert AwardRoadmapParser._parse_status("已完成任务") == "完成"
        assert AwardRoadmapParser._parse_status("Open issue") == "进行中"
        assert AwardRoadmapParser._parse_status("待处理") == "待开始"

    def test_parse_status_plain_text_returned_as_is(self) -> None:
        """_parse_status 对未知文本应原样返回。"""
        assert AwardRoadmapParser._parse_status("unknown") == "unknown"

    def test_extract_issue_number_simple(self) -> None:
        """_extract_issue_number 应提取简单的 #数字。"""
        assert AwardRoadmapParser._extract_issue_number("Open #850") == 850
        assert AwardRoadmapParser._extract_issue_number("Fix #42") == 42

    def test_extract_issue_number_skips_pr(self) -> None:
        """_extract_issue_number 应跳过 PR 编号。"""
        result = AwardRoadmapParser._extract_issue_number("PR #68 Open")
        assert result is None

    def test_extract_issue_number_with_both_pr_and_issue(self) -> None:
        """同时有 PR 和 Issue 时应提取 Issue 编号。"""
        result = AwardRoadmapParser._extract_issue_number("🔄 PR #68 Open (related #850)")
        assert result == 850

    def test_extract_issue_number_none_when_absent(self) -> None:
        """没有编号时应返回 None。"""
        assert AwardRoadmapParser._extract_issue_number("完成任务") is None

    def test_extract_pr_number(self) -> None:
        """_extract_pr_number 应提取 PR 编号。"""
        assert AwardRoadmapParser._extract_pr_number("PR #68 Open") == 68
        assert AwardRoadmapParser._extract_pr_number("pr #123") == 123

    def test_extract_pr_number_none_when_absent(self) -> None:
        """没有 PR 时应返回 None。"""
        assert AwardRoadmapParser._extract_pr_number("Open #850") is None


# ---------------------------------------------------------------------------
# EvidenceChecker 证据检查器测试
# ---------------------------------------------------------------------------


class TestEvidenceCheckerFile:
    """EvidenceChecker 文件存在性检查测试"""

    def test_existing_file_returns_passed(self, tmp_path: Path) -> None:
        """存在的文件应返回 exists=True。"""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "test.md").write_text("test", encoding="utf-8")

        checker = EvidenceChecker(tmp_path)
        item = checker.check_file("测试文档", "测试维度", "docs/test.md")

        assert item.exists is True
        assert item.is_glob is False
        assert item.name == "测试文档"
        assert item.dimension == "测试维度"
        assert item.matched_files == ["docs/test.md"]

    def test_missing_file_returns_failed(self, tmp_path: Path) -> None:
        """不存在的文件应返回 exists=False。"""
        checker = EvidenceChecker(tmp_path)
        item = checker.check_file("缺失文档", "测试维度", "docs/missing.md")

        assert item.exists is False
        assert item.is_glob is False
        assert item.matched_files == []

    def test_preserves_name_and_dimension(self, tmp_path: Path) -> None:
        """应保留证据项名称和所属维度。"""
        checker = EvidenceChecker(tmp_path)
        item = checker.check_file("证据名", "维度名", "nonexistent.md")

        assert item.name == "证据名"
        assert item.dimension == "维度名"
        assert item.path == "nonexistent.md"


class TestEvidenceCheckerGlob:
    """EvidenceChecker glob 模式匹配测试"""

    def test_glob_with_matches(self, tmp_path: Path) -> None:
        """glob 匹配到文件时应返回 exists=True 和匹配列表。"""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "a.md").write_text("a", encoding="utf-8")
        (tmp_path / "docs" / "b.md").write_text("b", encoding="utf-8")

        checker = EvidenceChecker(tmp_path)
        item = checker.check_glob("Markdown文件", "测试维度", "docs/*.md")

        assert item.exists is True
        assert item.is_glob is True
        assert len(item.matched_files) == 2
        assert "docs/a.md" in item.matched_files
        assert "docs/b.md" in item.matched_files

    def test_glob_with_no_matches(self, tmp_path: Path) -> None:
        """glob 未匹配到文件时应返回 exists=False。"""
        checker = EvidenceChecker(tmp_path)
        item = checker.check_glob("缺失文件", "测试维度", "docs/*.xyz")

        assert item.exists is False
        assert item.is_glob is True
        assert item.matched_files == []

    def test_glob_recursive(self, tmp_path: Path) -> None:
        """递归 glob 模式应能匹配子目录文件。"""
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "sub").mkdir(parents=True)
        (tmp_path / "tests" / "sub" / "test_example.py").write_text("pass", encoding="utf-8")

        checker = EvidenceChecker(tmp_path)
        item = checker.check_glob("测试文件", "测试维度", "tests/**/test_*.py")

        assert item.exists is True
        assert any("test_example.py" in f for f in item.matched_files)


class TestEvidenceCheckerTestFiles:
    """EvidenceChecker 测试文件匹配检查测试"""

    def test_finds_matching_test_files(self, tmp_path: Path) -> None:
        """应能在 tests/ 目录下找到匹配关键词的测试文件。"""
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_award_readiness.py").write_text("pass", encoding="utf-8")

        checker = EvidenceChecker(tmp_path)
        item = checker.check_test_files("奖励测试", "验证严谨性", "award")

        assert item.exists is True
        assert item.is_glob is True
        assert any("test_award_readiness.py" in f for f in item.matched_files)

    def test_no_match_when_no_test_files(self, tmp_path: Path) -> None:
        """没有匹配的测试文件时应返回 exists=False。"""
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_other.py").write_text("pass", encoding="utf-8")

        checker = EvidenceChecker(tmp_path)
        item = checker.check_test_files("不存在的测试", "验证", "nonexistent")

        assert item.exists is False


class TestEvidenceCheckerCheckAll:
    """EvidenceChecker 批量检查测试"""

    def test_check_all_processes_all_items(self, tmp_path: Path) -> None:
        """check_all 应处理配置中的所有项。"""
        (tmp_path / "docs").mkdir()
        (tmp_path / "docs" / "exist.md").write_text("x", encoding="utf-8")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_unit.py").write_text("x", encoding="utf-8")

        config = [
            {"name": "存在的文件", "dimension": "A", "path": "docs/exist.md", "type": "file"},
            {"name": "缺失的文件", "dimension": "B", "path": "docs/missing.md", "type": "file"},
            {"name": "测试文件", "dimension": "C", "path": "unit", "type": "test"},
            {"name": "glob匹配", "dimension": "D", "path": "docs/*.md", "type": "glob"},
        ]

        checker = EvidenceChecker(tmp_path)
        results = checker.check_all(config)

        assert len(results) == 4
        assert results[0].exists is True
        assert results[1].exists is False
        assert results[2].exists is True
        assert results[3].exists is True

    def test_check_all_empty_config(self, tmp_path: Path) -> None:
        """空配置应返回空列表。"""
        checker = EvidenceChecker(tmp_path)
        results = checker.check_all([])
        assert results == []

    def test_check_all_default_type_is_file(self, tmp_path: Path) -> None:
        """未指定 type 时默认按 file 处理。"""
        (tmp_path / "a.txt").write_text("x", encoding="utf-8")
        config = [{"name": "默认类型", "dimension": "A", "path": "a.txt"}]

        checker = EvidenceChecker(tmp_path)
        results = checker.check_all(config)

        assert len(results) == 1
        assert results[0].exists is True
        assert results[0].is_glob is False


# ---------------------------------------------------------------------------
# ReadinessResult 数据类测试
# ---------------------------------------------------------------------------


class TestReadinessResultOverallScore:
    """ReadinessResult overall_score 属性测试"""

    def test_average_of_dimension_scores(self) -> None:
        """综合得分应为各维度得分的平均值。"""
        result = ReadinessResult(
            dimension_scores={
                "维度A": DimensionScore("维度A", 80.0, 90.0, -10, "短板A"),
                "维度B": DimensionScore("维度B", 90.0, 95.0, -5, "短板B"),
            }
        )
        assert result.overall_score == pytest.approx(85.0)

    def test_single_dimension(self) -> None:
        """只有一个维度时，综合得分等于该维度得分。"""
        result = ReadinessResult(
            dimension_scores={
                "唯一维度": DimensionScore("唯一维度", 75.0, 90.0, -15, "短板"),
            }
        )
        assert result.overall_score == 75.0

    def test_empty_dimensions_returns_zero(self) -> None:
        """空维度列表应返回 0。"""
        result = ReadinessResult()
        assert result.overall_score == 0.0


class TestReadinessResultEvidenceCompleteness:
    """ReadinessResult evidence_completeness 属性测试"""

    def test_all_passed(self) -> None:
        """所有证据都存在时完整性应为 1.0。"""
        result = ReadinessResult(
            evidence_items=[
                EvidenceItem("a", "d1", "path/a.md", True),
                EvidenceItem("b", "d2", "path/b.md", True),
            ]
        )
        assert result.evidence_completeness == 1.0

    def test_none_passed(self) -> None:
        """所有证据都缺失时完整性应为 0.0。"""
        result = ReadinessResult(
            evidence_items=[
                EvidenceItem("a", "d1", "path/a.md", False),
                EvidenceItem("b", "d2", "path/b.md", False),
            ]
        )
        assert result.evidence_completeness == 0.0

    def test_partial_passed(self) -> None:
        """部分通过时应返回正确比例。"""
        result = ReadinessResult(
            evidence_items=[
                EvidenceItem("a", "d1", "path/a.md", True),
                EvidenceItem("b", "d2", "path/b.md", False),
                EvidenceItem("c", "d3", "path/c.md", True),
                EvidenceItem("d", "d4", "path/d.md", False),
            ]
        )
        assert result.evidence_completeness == 0.5

    def test_empty_evidence_returns_zero(self) -> None:
        """空证据列表应返回 0。"""
        result = ReadinessResult()
        assert result.evidence_completeness == 0.0


class TestReadinessResultTaskProgress:
    """ReadinessResult task_progress 属性测试"""

    def test_groups_by_priority(self) -> None:
        """应按优先级分组统计。"""
        result = ReadinessResult(
            tasks=[
                RoadmapTask("T1", "P0", "分类A", "人1", "完成"),
                RoadmapTask("T2", "P0", "分类A", "人2", "进行中"),
                RoadmapTask("T3", "P0", "分类B", "人3", "待开始"),
                RoadmapTask("T4", "P1", "分类C", "人4", "完成"),
                RoadmapTask("T5", "P2", "分类D", "人5", "待开始"),
            ]
        )
        progress = result.task_progress

        assert progress["P0"]["total"] == 3
        assert progress["P0"]["completed"] == 1
        assert progress["P0"]["in_progress"] == 1

        assert progress["P1"]["total"] == 1
        assert progress["P1"]["completed"] == 1
        assert progress["P1"]["in_progress"] == 0

        assert progress["P2"]["total"] == 1
        assert progress["P2"]["completed"] == 0

    def test_empty_tasks_returns_empty_dict(self) -> None:
        """空任务列表应返回空字典。"""
        result = ReadinessResult()
        assert result.task_progress == {}


class TestReadinessResultMissingEvidence:
    """ReadinessResult missing_evidence 属性测试"""

    def test_returns_only_missing_items(self) -> None:
        """应只返回不存在的证据项。"""
        items = [
            EvidenceItem("a", "d1", "path/a.md", True),
            EvidenceItem("b", "d2", "path/b.md", False),
            EvidenceItem("c", "d3", "path/c.md", True),
            EvidenceItem("d", "d4", "path/d.md", False),
        ]
        result = ReadinessResult(evidence_items=items)

        missing = result.missing_evidence
        assert len(missing) == 2
        assert all(not item.exists for item in missing)
        assert {item.name for item in missing} == {"b", "d"}

    def test_all_passed_returns_empty(self) -> None:
        """全部通过时应返回空列表。"""
        items = [
            EvidenceItem("a", "d1", "path/a.md", True),
            EvidenceItem("b", "d2", "path/b.md", True),
        ]
        result = ReadinessResult(evidence_items=items)
        assert result.missing_evidence == []

    def test_empty_evidence_returns_empty(self) -> None:
        """空证据列表应返回空列表。"""
        result = ReadinessResult()
        assert result.missing_evidence == []


# ---------------------------------------------------------------------------
# ReadinessReportGenerator 报告生成器测试
# ---------------------------------------------------------------------------


def _build_sample_result() -> ReadinessResult:
    """构建一个示例 ReadinessResult 用于报告生成测试。"""
    return ReadinessResult(
        dimension_scores={
            "主题契合度": DimensionScore("主题契合度", 85.0, 90.0, -5, "双向赋能证据不足"),
            "技术创新性": DimensionScore("技术创新性", 82.0, 90.0, -8, "退火统计不显著"),
        },
        tasks=[
            RoadmapTask("任务A", "P0", "分类1", "张三", "完成", "+5"),
            RoadmapTask("任务B", "P0", "分类2", "李四", "进行中", "+3"),
            RoadmapTask("任务C", "P1", "分类3", "王五", "待开始", "+2"),
        ],
        evidence_items=[
            EvidenceItem("证据1", "主题契合度", "docs/a.md", True),
            EvidenceItem("证据2", "技术创新性", "docs/b.md", False),
        ],
        generated_at="2026-08-03T12:00:00",
    )


class TestReadinessReportGeneratorMarkdown:
    """ReadinessReportGenerator to_markdown 测试"""

    def test_contains_title(self) -> None:
        """报告应包含标题。"""
        result = _build_sample_result()
        md = ReadinessReportGenerator(result).to_markdown()
        assert "# 冲奖就绪度评分仪表盘" in md

    def test_contains_generation_time(self) -> None:
        """报告应包含生成时间。"""
        result = _build_sample_result()
        md = ReadinessReportGenerator(result).to_markdown()
        assert "2026-08-03T12:00:00" in md

    def test_contains_overall_score(self) -> None:
        """报告应包含综合得分。"""
        result = _build_sample_result()
        md = ReadinessReportGenerator(result).to_markdown()
        assert "综合得分" in md
        # (85 + 82) / 2 = 83.5
        assert "83.5" in md

    def test_contains_evidence_completeness(self) -> None:
        """报告应包含证据完整性。"""
        result = _build_sample_result()
        md = ReadinessReportGenerator(result).to_markdown()
        assert "证据完整性" in md
        # 1/2 = 50%
        assert "50%" in md

    def test_contains_dimension_table(self) -> None:
        """报告应包含五维评分表。"""
        result = _build_sample_result()
        md = ReadinessReportGenerator(result).to_markdown()
        assert "## 五维评分" in md
        assert "主题契合度" in md
        assert "技术创新性" in md

    def test_contains_task_progress(self) -> None:
        """报告应包含任务进度。"""
        result = _build_sample_result()
        md = ReadinessReportGenerator(result).to_markdown()
        assert "## 任务进度" in md
        assert "P0" in md
        assert "P1" in md

    def test_contains_evidence_section(self) -> None:
        """报告应包含证据链完整性章节。"""
        result = _build_sample_result()
        md = ReadinessReportGenerator(result).to_markdown()
        assert "## 证据链完整性" in md

    def test_missing_evidence_listed(self) -> None:
        """缺失的证据应在报告中列出。"""
        result = _build_sample_result()
        md = ReadinessReportGenerator(result).to_markdown()
        assert "### 缺失证据" in md
        assert "证据2" in md

    def test_empty_result_generates_valid_markdown(self) -> None:
        """空结果也应生成有效的 Markdown。"""
        result = ReadinessResult(generated_at="2026-08-03T00:00:00")
        md = ReadinessReportGenerator(result).to_markdown()
        assert "# 冲奖就绪度评分仪表盘" in md
        assert "综合得分" in md
        assert "0.0" in md


class TestReadinessReportGeneratorJson:
    """ReadinessReportGenerator to_json 测试"""

    def test_returns_valid_json(self) -> None:
        """输出应是有效的 JSON。"""
        result = _build_sample_result()
        json_str = ReadinessReportGenerator(result).to_json()
        data = json.loads(json_str)
        assert isinstance(data, dict)

    def test_contains_top_level_fields(self) -> None:
        """JSON 应包含顶层字段。"""
        result = _build_sample_result()
        data = json.loads(ReadinessReportGenerator(result).to_json())

        assert "generated_at" in data
        assert "overall_score" in data
        assert "evidence_completeness" in data
        assert "dimension_scores" in data
        assert "task_progress" in data
        assert "tasks" in data
        assert "evidence" in data

    def test_overall_score_correct(self) -> None:
        """JSON 中综合得分应正确。"""
        result = _build_sample_result()
        data = json.loads(ReadinessReportGenerator(result).to_json())
        assert data["overall_score"] == pytest.approx(83.5)

    def test_dimension_scores_structure(self) -> None:
        """维度评分应具有正确的结构。"""
        result = _build_sample_result()
        data = json.loads(ReadinessReportGenerator(result).to_json())

        assert "主题契合度" in data["dimension_scores"]
        dim = data["dimension_scores"]["主题契合度"]
        assert dim["current_score"] == 85.0
        assert dim["target_score"] == 90.0
        assert dim["gap"] == -5.0
        assert "core_weakness" in dim
        assert "progress_ratio" in dim

    def test_task_progress_structure(self) -> None:
        """任务进度应具有正确的结构。"""
        result = _build_sample_result()
        data = json.loads(ReadinessReportGenerator(result).to_json())

        assert "P0" in data["task_progress"]
        assert data["task_progress"]["P0"]["total"] == 2
        assert data["task_progress"]["P0"]["completed"] == 1
        assert data["task_progress"]["P0"]["in_progress"] == 1

    def test_evidence_structure(self) -> None:
        """证据信息应具有正确的结构。"""
        result = _build_sample_result()
        data = json.loads(ReadinessReportGenerator(result).to_json())

        assert data["evidence"]["total"] == 2
        assert data["evidence"]["passed"] == 1
        assert len(data["evidence"]["missing"]) == 1
        assert data["evidence"]["missing"][0]["name"] == "证据2"
        assert len(data["evidence"]["items"]) == 2

    def test_tasks_have_all_fields(self) -> None:
        """任务列表中的每个任务应包含所有字段。"""
        result = _build_sample_result()
        data = json.loads(ReadinessReportGenerator(result).to_json())

        for task in data["tasks"]:
            assert "name" in task
            assert "priority" in task
            assert "category" in task
            assert "assignee" in task
            assert "status" in task
            assert "is_completed" in task
            assert "is_in_progress" in task

    def test_empty_result_valid_json(self) -> None:
        """空结果也应生成有效的 JSON。"""
        result = ReadinessResult(generated_at="2026-08-03T00:00:00")
        data = json.loads(ReadinessReportGenerator(result).to_json())
        assert data["overall_score"] == 0.0
        assert data["evidence_completeness"] == 0.0
        assert data["dimension_scores"] == {}
        assert data["tasks"] == []
