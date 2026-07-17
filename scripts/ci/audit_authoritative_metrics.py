#!/usr/bin/env python
"""审计仓库中的权威实验数字和状态空间表述。"""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree

TEXT_SUFFIXES = {".md", ".py", ".txt", ".rst", ".yaml", ".yml", ".json"}
OFFICE_SUFFIXES = {".docx", ".pptx"}
SKIP_DIRS = {".git", ".pytest_cache", ".mypy_cache", "__pycache__", "venv"}
SELF_EXCLUDES = {
    Path("scripts/ci/audit_authoritative_metrics.py"),
    Path("tests/test_metric_audit.py"),
}

FORBIDDEN_PATTERNS = (
    ("旧提升比例", re.compile(r"(?<![\d.])\+?95\.4%(?!\d)")),
    ("旧提升比例", re.compile(r"(?<![\d.])\+?75\.3%(?!\d)")),
    ("旧 PPO 奖励", re.compile(r"(?<![\d.])2864(?:\.\d+)?(?![\d.])")),
    ("旧实验奖励", re.compile(r"(?<![\d.])2555(?:\.\d+)?(?![\d.])")),
)

CANONICAL_RANKING = (
    "PPO",
    "SJF",
    "FCFS",
    "Random",
    "Greedy",
    "DQN",
    "Quantum-Only",
    "Classical-Only",
)


def find_forbidden(text: str) -> list[tuple[int, str, str]]:
    """返回文本中的旧数字及其行号。"""
    findings: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for label, pattern in FORBIDDEN_PATTERNS:
            if match := pattern.search(line):
                findings.append((line_number, label, match.group(0)))
    return findings


def validate_canonical_report(text: str) -> list[str]:
    """验证权威报告包含锁定数字、排名和 14→10 维说明。"""
    errors: list[str] = []
    for expected in ("2814.19", "1462.48", "+92.4%", "Obs10Wrapper", "14 维"):
        if expected not in text:
            errors.append(f"权威报告缺少：{expected}")

    positions = []
    for rank, strategy in enumerate(CANONICAL_RANKING, 1):
        row = re.search(
            rf"\|\s*{rank}\s*\|\s*(?:\*\*)?{re.escape(strategy)}(?:\*\*)?\s*\|",
            text,
        )
        positions.append(row.start() if row else -1)
    if any(position < 0 for position in positions):
        errors.append("权威报告缺少完整的八策略排名")
    elif positions != sorted(positions):
        errors.append("权威报告的八策略排名顺序不正确")
    return errors


def extract_office_text(path: Path) -> str:
    """从 DOCX/PPTX 的 XML 部件提取可见文本。"""
    chunks: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.endswith(".xml"):
                continue
            if path.suffix.lower() == ".docx" and not name.startswith("word/"):
                continue
            if path.suffix.lower() == ".pptx" and not name.startswith("ppt/"):
                continue
            root = ElementTree.fromstring(archive.read(name))
            chunks.extend(node.text for node in root.iter() if node.text)
    return "\n".join(chunks)


def iter_audited_files(root: Path):
    """遍历参与数字审计的文本和 Office 文件。"""
    for path in root.rglob("*"):
        if not path.is_file() or any(part in SKIP_DIRS for part in path.parts):
            continue
        relative = path.relative_to(root)
        if relative in SELF_EXCLUDES:
            continue
        if path.suffix.lower() in TEXT_SUFFIXES | OFFICE_SUFFIXES:
            yield path


def audit_repository(root: Path) -> list[str]:
    """执行完整仓库审计并返回人类可读的错误列表。"""
    errors: list[str] = []
    for path in iter_audited_files(root):
        try:
            if path.suffix.lower() in OFFICE_SUFFIXES:
                text = extract_office_text(path)
            else:
                text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError, zipfile.BadZipFile, ElementTree.ParseError) as exc:
            errors.append(f"无法读取 {path.relative_to(root)}：{exc}")
            continue

        for line_number, label, value in find_forbidden(text):
            errors.append(f"{path.relative_to(root)}:{line_number} 包含{label} {value}")

    report_path = root / "results/reports/strategy_comparison.md"
    if not report_path.exists():
        errors.append("缺少权威报告 results/reports/strategy_comparison.md")
    else:
        report_text = report_path.read_text(encoding="utf-8")
        errors.extend(validate_canonical_report(report_text))

    agent_text = (root / "src/scheduler/agent.py").read_text(encoding="utf-8")
    if "原生输出 14 维" not in agent_text or "Obs10Wrapper" not in agent_text:
        errors.append("agent.py 未说明原生 14 维与 Obs10Wrapper 兼容关系")
    return errors


def main() -> int:
    """运行命令行审计。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="项目根目录",
    )
    args = parser.parse_args()
    errors = audit_repository(args.root.resolve())
    if errors:
        print("权威数字审计失败：")
        for error in errors:
            print(f"- {error}")
        return 1
    print("权威数字审计通过：八策略排名、核心数字和 14→10 维口径一致。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
