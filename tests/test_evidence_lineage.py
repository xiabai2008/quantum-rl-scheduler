"""tests/test_evidence_lineage.py — Issue #8 证据谱系脚本单元测试"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.ci.evidence_lineage import (
    CANONICAL_METRICS,
    LEAK_PATTERNS,
    EvidenceReport,
    LeakOccurrence,
    MetricOccurrence,
    find_leak_occurrences,
    find_metric_occurrences,
    get_git_info,
    git_blame_line,
    iter_text_files,
    render_markdown,
    scan_repository,
)

# ---------------------------------------------------------------------------
# 测试数据
# ---------------------------------------------------------------------------
SAMPLE_METRIC_TEXT = """# 策略对比报告

PPO 平均奖励：2746.94，标准差 1121.19
FCFS 平均奖励：1458.77，标准差 55.85
PPO 相对 FCFS 提升 +88.3%
统计显著性 p=3.04e-11，Cohen's d=-1.70
样本量 N=250
"""

SAMPLE_LEAK_TEXT = """# 文档

代码位置：file:///c:/Users/HZR/Desktop/quantum-rl-scheduler/src/env.py
另一处：d:\\桌面\\quantum-rl-scheduler\\src\\api.py
Linux: /home/username/project/file.py
"""

SAMPLE_CLEAN_TEXT = """# 干净文档

PPO 平均奖励 2746.94，相对 FCFS 提升 +88.3%。
没有任何本机路径泄漏。
"""


# ---------------------------------------------------------------------------
# MetricOccurrence / LeakOccurrence 数据类
# ---------------------------------------------------------------------------
class TestMetricOccurrence:
    """MetricOccurrence 数据类测试"""

    def test_default_values(self) -> None:
        occ = MetricOccurrence(
            metric_name="PPO 平均奖励",
            file_path="docs/test.md",
            line_number=1,
            line_content="PPO: 2746.94",
        )
        assert occ.commit == ""
        assert occ.author == ""
        assert occ.commit_date == ""


class TestLeakOccurrence:
    """LeakOccurrence 数据类测试"""

    def test_fields(self) -> None:
        occ = LeakOccurrence(
            leak_type="Windows 用户目录",
            file_path="docs/test.md",
            line_number=5,
            line_content="path: C:\\Users\\xxx",
        )
        assert occ.leak_type == "Windows 用户目录"
        assert occ.line_number == 5


# ---------------------------------------------------------------------------
# EvidenceReport 数据类
# ---------------------------------------------------------------------------
class TestEvidenceReport:
    """EvidenceReport 数据类测试"""

    def test_empty_report_has_no_leaks(self) -> None:
        report = EvidenceReport(
            repo_root="/tmp",
            generated_at="2026-07-22",
            current_commit="abc123",
            current_branch="main",
        )
        assert not report.has_leaks
        assert report.metric_summary == {}

    def test_metric_summary_groups_by_name(self) -> None:
        report = EvidenceReport(
            repo_root="/tmp",
            generated_at="2026-07-22",
            current_commit="abc123",
            current_branch="main",
            metric_occurrences=[
                MetricOccurrence("PPO 平均奖励", "a.md", 1, "line"),
                MetricOccurrence("PPO 平均奖励", "b.md", 2, "line"),
                MetricOccurrence("FCFS 平均奖励", "c.md", 3, "line"),
            ],
        )
        assert report.metric_summary == {"PPO 平均奖励": 2, "FCFS 平均奖励": 1}

    def test_has_leaks_with_occurrences(self) -> None:
        report = EvidenceReport(
            repo_root="/tmp",
            generated_at="2026-07-22",
            current_commit="abc123",
            current_branch="main",
            leak_occurrences=[
                LeakOccurrence("Windows 用户目录", "a.md", 1, "leak"),
            ],
        )
        assert report.has_leaks


# ---------------------------------------------------------------------------
# find_metric_occurrences
# ---------------------------------------------------------------------------
class TestFindMetricOccurrences:
    """find_metric_occurrences 测试"""

    def test_finds_all_canonical_metrics(self) -> None:
        occurrences = find_metric_occurrences(SAMPLE_METRIC_TEXT, "docs/test.md")
        names = [occ.metric_name for occ in occurrences]
        # 至少命中 5 个不同指标
        assert "PPO 平均奖励" in names
        assert "FCFS 平均奖励" in names
        assert "PPO 相对 FCFS 提升" in names
        assert "Welch t 检验 p 值" in names
        assert "N=250 样本量" in names

    def test_line_numbers_correct(self) -> None:
        occurrences = find_metric_occurrences(SAMPLE_METRIC_TEXT, "docs/test.md")
        # PPO 2746.94 在第 3 行
        ppo_occs = [occ for occ in occurrences if occ.metric_name == "PPO 平均奖励"]
        assert ppo_occs[0].line_number == 3

    def test_no_false_positive_on_clean_text(self) -> None:
        # 清晰文本中不应误报禁用数字（如 100、50 等）
        text = "页数 100，文件大小 50MB，时间 2026-07-22\n"
        occurrences = find_metric_occurrences(text, "docs/test.md")
        assert occurrences == []

    def test_file_path_recorded(self) -> None:
        occurrences = find_metric_occurrences(SAMPLE_METRIC_TEXT, "custom/path.md")
        for occ in occurrences:
            assert occ.file_path == "custom/path.md"

    def test_line_content_stripped(self) -> None:
        occurrences = find_metric_occurrences("  PPO: 2746.94  \n", "a.md")
        assert occurrences[0].line_content == "PPO: 2746.94"


# ---------------------------------------------------------------------------
# find_leak_occurrences
# ---------------------------------------------------------------------------
class TestFindLeakOccurrences:
    """find_leak_occurrences 测试"""

    def test_finds_windows_user_path(self) -> None:
        text = "路径: C:\\Users\\xxx\\file.py\n"
        occurrences = find_leak_occurrences(text, "docs/test.md")
        assert len(occurrences) >= 1
        assert any(o.leak_type == "Windows 用户目录" for o in occurrences)

    def test_finds_windows_chinese_desktop(self) -> None:
        text = "路径: d:\\桌面\\project\\file.py\n"
        occurrences = find_leak_occurrences(text, "docs/test.md")
        assert any("Windows" in o.leak_type for o in occurrences)

    def test_finds_unix_home_path(self) -> None:
        text = "path: /home/username/project/file.py\n"
        occurrences = find_leak_occurrences(text, "docs/test.md")
        assert any("Unix" in o.leak_type for o in occurrences)

    def test_finds_file_protocol(self) -> None:
        text = "link: file:///c:/Users/x/file.md\n"
        occurrences = find_leak_occurrences(text, "docs/test.md")
        # file:// + Windows 用户目录两类
        assert len(occurrences) >= 1

    def test_clean_text_no_leaks(self) -> None:
        text = "相对路径：./src/env.py，无任何本机路径\n"
        assert find_leak_occurrences(text, "docs/test.md") == []

    def test_exempt_devcontainer_path(self) -> None:
        """devcontainer 标准路径 /home/vscode/ 应豁免"""
        text = "path: /home/vscode/.vscode-server/extensions\n"
        assert find_leak_occurrences(text, ".devcontainer/test.json") == []

    def test_exempt_github_actions_path(self) -> None:
        """GitHub Actions 标准路径 /home/runner/ 应豁免"""
        text = "workspace: /home/runner/work/repo\n"
        assert find_leak_occurrences(text, ".github/workflows/test.yml") == []


# ---------------------------------------------------------------------------
# iter_text_files
# ---------------------------------------------------------------------------
class TestIterTextFiles:
    """iter_text_files 测试"""

    def test_iterates_only_text_files(self, tmp_path: Path) -> None:
        (tmp_path / "a.md").write_text("test", encoding="utf-8")
        (tmp_path / "b.py").write_text("test", encoding="utf-8")
        (tmp_path / "c.bin").write_text("test", encoding="utf-8")
        files = list(iter_text_files(tmp_path))
        suffixes = {f.suffix for f in files}
        assert ".md" in suffixes
        assert ".py" in suffixes
        assert ".bin" not in suffixes

    def test_skips_venv_directories(self, tmp_path: Path) -> None:
        (tmp_path / ".venv").mkdir()
        (tmp_path / ".venv" / "x.py").write_text("test", encoding="utf-8")
        (tmp_path / "main.py").write_text("test", encoding="utf-8")
        files = list(iter_text_files(tmp_path))
        rel_paths = [f.relative_to(tmp_path).as_posix() for f in files]
        assert "main.py" in rel_paths
        assert not any(p.startswith(".venv") for p in rel_paths)


# ---------------------------------------------------------------------------
# get_git_info / git_blame_line（使用当前仓库）
# ---------------------------------------------------------------------------
class TestGitInfo:
    """get_git_info / git_blame_line 测试（依赖真实 git 仓库）"""

    def test_get_git_info_returns_strings(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        commit, branch = get_git_info(repo_root)
        assert isinstance(commit, str)
        assert isinstance(branch, str)
        # 当前在 feature/issue-8-evidence-lineage 分支
        assert branch != "unknown"

    def test_git_blame_line_returns_strings(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        # 对 README.md 第 1 行执行 blame（应该能拿到 commit）
        commit, author, date = git_blame_line(repo_root, "README.md", 1)
        assert isinstance(commit, str)
        assert isinstance(author, str)
        assert isinstance(date, str)

    def test_git_blame_line_handles_invalid_file(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        commit, author, date = git_blame_line(repo_root, "nonexistent.md", 1)
        assert commit == ""
        assert author == ""
        assert date == ""


# ---------------------------------------------------------------------------
# render_markdown
# ---------------------------------------------------------------------------
class TestRenderMarkdown:
    """render_markdown 测试"""

    def test_render_contains_required_sections(self) -> None:
        report = EvidenceReport(
            repo_root="/tmp",
            generated_at="2026-07-22",
            current_commit="abc123",
            current_branch="main",
        )
        md = render_markdown(report)
        assert "# 实验证据谱系与一键复核包" in md
        assert "## 1. 权威数字汇总" in md
        assert "## 2. 权威数字溯源表" in md
        assert "## 3. 本机路径泄漏清单" in md
        assert "## 4. 一键复核命令清单" in md
        assert "## 5. 数字一致性声明" in md

    def test_render_includes_commit_and_branch(self) -> None:
        report = EvidenceReport(
            repo_root="/tmp",
            generated_at="2026-07-22",
            current_commit="abc123def",
            current_branch="feature/test",
        )
        md = render_markdown(report)
        assert "abc123def" in md
        assert "feature/test" in md

    def test_render_with_leaks_shows_warning(self) -> None:
        report = EvidenceReport(
            repo_root="/tmp",
            generated_at="2026-07-22",
            current_commit="abc",
            current_branch="main",
            leak_occurrences=[
                LeakOccurrence("Windows 用户目录", "docs/x.md", 5, "C:\\Users\\x"),
            ],
        )
        md = render_markdown(report)
        assert "1" in md  # 泄漏数
        assert "Windows 用户目录" in md

    def test_render_no_leaks_shows_success(self) -> None:
        report = EvidenceReport(
            repo_root="/tmp",
            generated_at="2026-07-22",
            current_commit="abc",
            current_branch="main",
        )
        md = render_markdown(report)
        assert "未发现本机路径泄漏" in md

    def test_render_includes_reproduction_commands(self) -> None:
        report = EvidenceReport(
            repo_root="/tmp",
            generated_at="2026-07-22",
            current_commit="abc",
            current_branch="main",
        )
        md = render_markdown(report)
        assert "audit_authoritative_metrics.py" in md
        assert "validate_submission.py" in md
        assert "run_multiseed_evaluation.py" in md


# ---------------------------------------------------------------------------
# scan_repository（集成测试，使用当前仓库）
# ---------------------------------------------------------------------------
class TestScanRepository:
    """scan_repository 集成测试"""

    def test_scan_current_repo_finds_known_metrics(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        # 使用 --no-blame 加速
        report = scan_repository(repo_root, with_blame=False)
        # 仓库中应该能找到权威数字（AGENTS.md / strategy_comparison.md 等都包含）
        assert len(report.metric_occurrences) > 0
        metric_names = {occ.metric_name for occ in report.metric_occurrences}
        assert "PPO 平均奖励" in metric_names

    def test_scan_fills_blame_info_when_requested(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        report = scan_repository(repo_root, with_blame=True)
        # 至少有一个位置的 commit 被填充（README.md 等已 tracked 的文件）
        assert any(occ.commit for occ in report.metric_occurrences)

    def test_scan_finds_no_leak_in_cleaned_code_wiki(self) -> None:
        """docs/Code_Wiki.md 已清理本机路径，扫描结果应为 0 处泄漏。

        Issue #8 已批量替换 file:///c:/Users/HZR/Desktop/... 为相对路径 ../，
        此测试作为回归保护，防止本机路径再次进入 Code_Wiki.md。
        """
        repo_root = Path(__file__).resolve().parents[1]
        report = scan_repository(repo_root, with_blame=False)
        code_wiki_leaks = [
            occ for occ in report.leak_occurrences if occ.file_path == "docs/Code_Wiki.md"
        ]
        assert code_wiki_leaks == []


# ---------------------------------------------------------------------------
# CANONICAL_METRICS / LEAK_PATTERNS 常量
# ---------------------------------------------------------------------------
class TestConstants:
    """常量完整性测试"""

    def test_canonical_metrics_non_empty(self) -> None:
        assert len(CANONICAL_METRICS) >= 5
        for name, pattern in CANONICAL_METRICS:
            assert isinstance(name, str)
            assert pattern  # 编译后的 pattern 真值

    def test_leak_patterns_cover_major_os(self) -> None:
        leak_types = [name for name, _ in LEAK_PATTERNS]
        # 至少覆盖 Windows 和 Unix
        assert any("Windows" in t for t in leak_types)
        assert any("Unix" in t for t in leak_types)
