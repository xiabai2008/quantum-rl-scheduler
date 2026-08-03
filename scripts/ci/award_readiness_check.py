#!/usr/bin/env python3
"""Award readiness check - core classes for testing."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, ClassVar

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


@dataclass
class DimensionScore:
    """Single dimension scoring info."""

    name: str
    current_score: float
    target_score: float
    gap: float
    core_weakness: str

    @property
    def progress_ratio(self) -> float:
        """Ratio of current to target score (0~1)."""
        if self.target_score <= 0:
            return 0.0
        return min(max(self.current_score / self.target_score, 0.0), 1.0)


@dataclass
class RoadmapTask:
    """Single roadmap task."""

    name: str
    priority: str
    category: str
    assignee: str
    status: str
    expected_improvement: str = ""
    issue_number: int | None = None
    pr_number: int | None = None

    @property
    def is_completed(self) -> bool:
        """Whether task is completed."""
        completed_markers = ["完成", "done", "已完成", "merged"]
        return any(m in self.status.lower() for m in completed_markers)

    @property
    def is_in_progress(self) -> bool:
        """Whether task is in progress."""
        in_progress_markers = ["进行中", "open", "pr", "\U0001f504"]
        return any(m in self.status.lower() for m in in_progress_markers)


@dataclass
class EvidenceItem:
    """Evidence check result."""

    name: str
    dimension: str
    path: str
    exists: bool
    is_glob: bool = False
    matched_files: list[str] = field(default_factory=list)


@dataclass
class ReadinessResult:
    """Overall readiness result."""

    dimension_scores: dict[str, DimensionScore] = field(default_factory=dict)
    tasks: list[RoadmapTask] = field(default_factory=list)
    evidence_items: list[EvidenceItem] = field(default_factory=list)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @property
    def overall_score(self) -> float:
        """Overall average score across all dimensions."""
        if not self.dimension_scores:
            return 0.0
        scores = [ds.current_score for ds in self.dimension_scores.values()]
        return sum(scores) / len(scores)

    @property
    def evidence_completeness(self) -> float:
        """Evidence completeness ratio (0~1)."""
        if not self.evidence_items:
            return 0.0
        passed = sum(1 for item in self.evidence_items if item.exists)
        return passed / len(self.evidence_items)

    @property
    def task_progress(self) -> dict[str, dict[str, int]]:
        """Task progress stats by priority."""
        progress: dict[str, dict[str, int]] = {}
        for task in self.tasks:
            p = task.priority
            if p not in progress:
                progress[p] = {"total": 0, "completed": 0, "in_progress": 0}
            progress[p]["total"] += 1
            if task.is_completed:
                progress[p]["completed"] += 1
            elif task.is_in_progress:
                progress[p]["in_progress"] += 1
        return progress

    @property
    def missing_evidence(self) -> list[EvidenceItem]:
        """List of missing evidence items."""
        return [item for item in self.evidence_items if not item.exists]


class AwardRoadmapParser:
    """Markdown parser for award roadmap."""

    _STATUS_ICONS: ClassVar[dict[str, str]] = {
        "\u2705": "完成",
        "\U0001f4cb": "待开始",
        "\U0001f504": "进行中",
        "\U0001f4c5": "计划中",
        "\u274c": "已取消",
    }

    def __init__(self, markdown_content: str) -> None:
        self.content = markdown_content
        self.lines = markdown_content.splitlines()

    def parse_dimension_scores(self) -> dict[str, DimensionScore]:
        """Parse 5-dimension score table."""
        scores: dict[str, DimensionScore] = {}
        in_table = False
        header_found = False

        for line in self.lines:
            stripped = line.strip()

            if "五维评分" in stripped and stripped.startswith("###"):
                in_table = True
                continue

            if not in_table:
                continue

            if not stripped:
                continue

            if "评审维度" in stripped and "当前得分" in stripped:
                header_found = True
                continue

            if re.match(r"^\|[\s\-:|]+\|$", stripped):
                continue

            if header_found and stripped.startswith("|"):
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if len(cells) >= 5:
                    name = cells[0].replace("**", "").strip()
                    current = self._parse_score(cells[1])
                    target = self._parse_score(cells[2])
                    gap = self._parse_score(cells[3])
                    weakness = cells[4].strip()

                    if "综合" in name:
                        continue

                    scores[name] = DimensionScore(
                        name=name,
                        current_score=current,
                        target_score=target,
                        gap=gap,
                        core_weakness=weakness,
                    )

            if header_found and stripped.startswith("##") and "五维评分" not in stripped:
                break

        return scores

    def parse_tasks(self) -> list[RoadmapTask]:
        """Parse P0/P1/P2 task lists."""
        tasks: list[RoadmapTask] = []
        current_priority = ""
        current_category = ""
        in_task_table = False

        for i, line in enumerate(self.lines):
            stripped = line.strip()

            priority_match = re.match(r"^##\s+\d+\.\s+(P[012])", stripped)
            if priority_match:
                current_priority = priority_match.group(1)
                in_task_table = False
                continue

            if stripped.startswith("###") and current_priority:
                current_category = re.sub(r"^###\s+\d+\.\d+\s*", "", stripped).strip()
                current_category = re.sub(r"[（(].*?[)）]", "", current_category).strip()
                in_task_table = False
                continue

            if (
                current_priority
                and stripped.startswith("|")
                and "任务" in stripped
                and "状态" in stripped
            ):
                in_task_table = True
                continue

            if in_task_table and re.match(r"^\|[\s\-:|]+\|$", stripped):
                continue

            if in_task_table and stripped.startswith("|") and current_priority:
                cells = [c.strip() for c in stripped.strip("|").split("|")]
                if len(cells) >= 3:
                    task_name = cells[0].strip()
                    assignee = cells[1].strip() if len(cells) > 1 else ""
                    status_raw = cells[2].strip() if len(cells) > 2 else ""
                    improvement = cells[3].strip() if len(cells) > 3 else ""

                    status = self._parse_status(status_raw)
                    issue_num = self._extract_issue_number(status_raw + " " + task_name)
                    pr_num = self._extract_pr_number(status_raw + " " + task_name)

                    tasks.append(
                        RoadmapTask(
                            name=task_name,
                            priority=current_priority,
                            category=current_category,
                            assignee=assignee,
                            status=status,
                            expected_improvement=improvement,
                            issue_number=issue_num,
                            pr_number=pr_num,
                        )
                    )
                continue

            if (
                in_task_table
                and not stripped
                and i + 1 < len(self.lines)
                and not self.lines[i + 1].strip().startswith("|")
            ):
                in_task_table = False

        return tasks

    @staticmethod
    def _parse_score(text: str) -> float:
        """Extract numeric score from text."""
        text = text.strip().replace("**", "").replace("~", "")

        match = re.match(r"^([+\-]?\d+\.?\d*)/", text)
        if match:
            return float(match.group(1))

        match = re.match(r"^([+\-]?\d+\.?\d*)\+$", text)
        if match:
            return float(match.group(1))

        match = re.match(r"^[+\-]?\d+\.?\d*", text)
        if match:
            return float(match.group(0))

        return 0.0

    @staticmethod
    def _parse_status(text: str) -> str:
        """Parse status from status text."""
        text = text.strip()

        for icon, status in AwardRoadmapParser._STATUS_ICONS.items():
            if icon in text:
                return status

        text_lower = text.lower()
        if "完成" in text or "done" in text_lower or "merged" in text_lower:
            return "完成"
        if "进行中" in text or "open" in text_lower or "pr" in text_lower:
            return "进行中"
        if "待" in text or "todo" in text_lower or "planned" in text_lower:
            return "待开始"
        if "计划" in text or "scheduled" in text_lower:
            return "计划中"

        return text

    @staticmethod
    def _extract_issue_number(text: str) -> int | None:
        """Extract issue number from text."""
        text_without_pr = re.sub(r"PR\s*#\d+", "", text, flags=re.IGNORECASE)
        match = re.search(r"#(\d+)", text_without_pr)
        if match:
            return int(match.group(1))
        return None

    @staticmethod
    def _extract_pr_number(text: str) -> int | None:
        """Extract PR number from text."""
        match = re.search(r"PR\s*#(\d+)", text, flags=re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None


class EvidenceChecker:
    """Evidence file existence checker."""

    def __init__(self, project_root: Path | str) -> None:
        self.project_root = Path(project_root)

    def check_file(self, name: str, dimension: str, file_path: str) -> EvidenceItem:
        """Check if a single file exists."""
        full_path = self.project_root / file_path
        exists = full_path.exists()
        return EvidenceItem(
            name=name,
            dimension=dimension,
            path=file_path,
            exists=exists,
            is_glob=False,
            matched_files=[str(file_path)] if exists else [],
        )

    def check_glob(self, name: str, dimension: str, pattern: str) -> EvidenceItem:
        """Check if glob pattern matches at least one file."""
        matched = sorted(self.project_root.glob(pattern))
        rel_paths = [p.relative_to(self.project_root).as_posix() for p in matched]
        return EvidenceItem(
            name=name,
            dimension=dimension,
            path=pattern,
            exists=len(matched) > 0,
            is_glob=True,
            matched_files=rel_paths,
        )

    def check_test_files(self, name: str, dimension: str, keyword: str) -> EvidenceItem:
        """Check for test files containing keyword."""
        pattern = f"tests/**/test_*{keyword}*.py"
        return self.check_glob(name, dimension, pattern)

    def check_all(self, evidence_config: list[dict[str, Any]]) -> list[EvidenceItem]:
        """Check all evidence items from config."""
        results: list[EvidenceItem] = []
        for item in evidence_config:
            name = item.get("name", "")
            dimension = item.get("dimension", "")
            path = item.get("path", "")
            item_type = item.get("type", "file")

            if item_type == "glob":
                results.append(self.check_glob(name, dimension, path))
            elif item_type == "test":
                results.append(self.check_test_files(name, dimension, path))
            else:
                results.append(self.check_file(name, dimension, path))

        return results


class ReadinessReportGenerator:
    """Readiness report generator."""

    def __init__(self, result: ReadinessResult) -> None:
        self.result = result

    def to_markdown(self) -> str:
        """Generate Markdown format report."""
        lines: list[str] = []
        r = self.result

        lines.append("# 冲奖就绪度评分仪表盘")
        lines.append("")
        lines.append(f"> 生成时间：{r.generated_at}")
        lines.append("")

        lines.append("## 综合概览")
        lines.append("")
        lines.append(f"- **综合得分**：{r.overall_score:.1f}/100")
        lines.append(f"- **证据完整性**：{r.evidence_completeness:.0%}")
        lines.append("")

        lines.append("## 五维评分")
        lines.append("")
        lines.append("| 评审维度 | 当前得分 | 目标得分 | 差距 | 进度 | 核心短板 |")
        lines.append("|---------|---------|---------|------|------|---------|")
        for name, ds in r.dimension_scores.items():
            progress_pct = ds.progress_ratio * 100
            lines.append(
                f"| {name} | {ds.current_score:.0f}/100 | {ds.target_score:.0f}+ | "
                f"{ds.gap:+.0f} | {progress_pct:.0f}% | {ds.core_weakness} |"
            )
        lines.append("")

        lines.append("## 任务进度")
        lines.append("")
        for priority in sorted(r.task_progress.keys()):
            prog = r.task_progress[priority]
            total = prog["total"]
            completed = prog["completed"]
            in_progress = prog["in_progress"]
            pct = completed / total * 100 if total > 0 else 0
            lines.append(
                f"### {priority}：{completed}/{total} 完成 （{pct:.0f}%，进行中 {in_progress}）"
            )
            lines.append("")
            priority_tasks = [t for t in r.tasks if t.priority == priority]
            lines.append("| 任务 | 分类 | 负责人 | 状态 |")
            lines.append("|------|------|--------|------|")
            for task in priority_tasks:
                status_icon = (
                    "\u2705"
                    if task.is_completed
                    else "\U0001f504"
                    if task.is_in_progress
                    else "\U0001f4cb"
                )
                lines.append(
                    f"| {task.name} | {task.category} | "
                    f"{task.assignee} | {status_icon} {task.status} |"
                )
            lines.append("")

        lines.append("## 证据链完整性")
        lines.append("")
        lines.append(
            f"共检查 {len(r.evidence_items)} 项证据，"
            f"已就位 {len(r.evidence_items) - len(r.missing_evidence)} 项，"
            f"缺失 {len(r.missing_evidence)} 项。"
        )
        lines.append("")

        if r.missing_evidence:
            lines.append("### 缺失证据")
            lines.append("")
            for item in r.missing_evidence:
                lines.append(f"- **[{item.dimension}]** {item.name}：`{item.path}`")
            lines.append("")

        return "\n".join(lines)

    def to_json(self) -> str:
        """Generate JSON format report."""
        r = self.result

        data: dict[str, Any] = {
            "generated_at": r.generated_at,
            "overall_score": round(r.overall_score, 2),
            "evidence_completeness": round(r.evidence_completeness, 4),
            "dimension_scores": {
                name: {
                    "current_score": ds.current_score,
                    "target_score": ds.target_score,
                    "gap": ds.gap,
                    "core_weakness": ds.core_weakness,
                    "progress_ratio": round(ds.progress_ratio, 4),
                }
                for name, ds in r.dimension_scores.items()
            },
            "task_progress": r.task_progress,
            "tasks": [
                {
                    "name": t.name,
                    "priority": t.priority,
                    "category": t.category,
                    "assignee": t.assignee,
                    "status": t.status,
                    "is_completed": t.is_completed,
                    "is_in_progress": t.is_in_progress,
                    "expected_improvement": t.expected_improvement,
                    "issue_number": t.issue_number,
                    "pr_number": t.pr_number,
                }
                for t in r.tasks
            ],
            "evidence": {
                "total": len(r.evidence_items),
                "passed": len(r.evidence_items) - len(r.missing_evidence),
                "missing": [
                    {
                        "name": item.name,
                        "dimension": item.dimension,
                        "path": item.path,
                    }
                    for item in r.missing_evidence
                ],
                "items": [
                    {
                        "name": item.name,
                        "dimension": item.dimension,
                        "path": item.path,
                        "exists": item.exists,
                        "is_glob": item.is_glob,
                        "matched_files": item.matched_files,
                    }
                    for item in r.evidence_items
                ],
            },
        }

        return json.dumps(data, ensure_ascii=False, indent=2)


def _load_roadmap(project_root: Path) -> str:
    """Load roadmap markdown content."""
    roadmap_path = project_root / "docs" / "award_roadmap.md"
    if roadmap_path.exists():
        return roadmap_path.read_text(encoding="utf-8")
    return ""


def _default_evidence_config() -> list[dict[str, Any]]:
    """Default evidence configuration."""
    return [
        {
            "name": "冲奖路线图",
            "dimension": "主题契合度",
            "path": "docs/award_roadmap.md",
            "type": "file",
        },
        {
            "name": "价值量化",
            "dimension": "落地与价值",
            "path": "docs/value_quantification.md",
            "type": "file",
        },
        {
            "name": "单元测试覆盖",
            "dimension": "验证严谨性",
            "path": "test_",
            "type": "test",
        },
    ]


def build_readiness_result(project_root: Path) -> ReadinessResult:
    """Build readiness result."""
    result = ReadinessResult()

    roadmap_content = _load_roadmap(project_root)
    if roadmap_content:
        parser = AwardRoadmapParser(roadmap_content)
        result.dimension_scores = parser.parse_dimension_scores()
        result.tasks = parser.parse_tasks()

    checker = EvidenceChecker(project_root)
    config = _default_evidence_config()
    result.evidence_items = checker.check_all(config)

    return result


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="冲奖就绪度评分仪表盘 — 五维度证据链自动化校验")
    parser.add_argument(
        "--check",
        action="store_true",
        help="检查模式：证据链完整性校验，退出码 0=通过 1=不通过",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="报告模式：输出 Markdown 格式就绪度报告",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="输出 JSON 格式结果",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        help="输出 Markdown 格式结果（默认）",
    )

    args = parser.parse_args()

    project_root = _PROJECT_ROOT
    result = build_readiness_result(project_root)

    if args.json:
        generator = ReadinessReportGenerator(result)
        print(generator.to_json())
    elif args.markdown or args.report:
        generator = ReadinessReportGenerator(result)
        print(generator.to_markdown())
    elif args.check:
        missing = result.missing_evidence
        if missing:
            print(f"证据链不完整：缺失 {len(missing)} 项")
            for item in missing:
                print(f"  - [{item.dimension}] {item.name}: {item.path}")
            return 1
        else:
            print("所有证据项均已就位")
            return 0
    else:
        generator = ReadinessReportGenerator(result)
        print(generator.to_markdown())

    return 0


if __name__ == "__main__":
    sys.exit(main())
