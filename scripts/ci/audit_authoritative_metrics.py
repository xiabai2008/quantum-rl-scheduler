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
SKIP_DIRS = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    "__pycache__",
    "venv",
    ".venv",
    ".venv-mutmut",
    "node_modules",
    # 实验数据目录：带时间戳的历史快照，记录当时真实数据，
    # 不应被强制更新为当前权威数字，不参与口径审计
    "fair_comparison",
    "issue_experiments",
    "multiseed_evaluation",
    "real_machine",
    "gradient_stress",
    "models",
}
SELF_EXCLUDES = {
    Path("scripts/ci/audit_authoritative_metrics.py"),
    Path("tests/test_metric_audit.py"),
}

FORBIDDEN_PATTERNS = (
    ("旧提升比例", re.compile(r"(?<![\d.])\+?95\.4%(?!\d)")),
    ("旧提升比例", re.compile(r"(?<![\d.])\+?75\.3%(?!\d)")),
    ("旧 PPO 奖励", re.compile(r"(?<![\d.])2864(?:\.\d+)?(?![\d.])")),
    ("旧 PPO 奖励", re.compile(r"(?<![\d.])2723(?:\.\d+)?(?![\d.])")),
    ("旧实验奖励", re.compile(r"(?<![\d.])2555(?:\.\d+)?(?![\d.])")),
)

# 行级豁免标记：包含此标记的行视为明确标注的历史数据，跳过该行禁止模式检查。
# 豁免仅对当前行生效，不影响其他行或其他文件，不会全局跳过任何文件。
AUDIT_EXEMPT_MARKER = "<!-- audit-exempt:"

CANONICAL_RANKING = (
    "PPO",
    "SJF",
    "FCFS",
    "DQN",
    "Random",
    "Greedy",
    "Quantum-Only",
    "Classical-Only",
)


def find_forbidden(text: str) -> list[tuple[int, str, str]]:
    """返回文本中的旧数字及其行号。

    带有 ``<!-- audit-exempt: ... -->`` 标记的行被视为明确豁免的历史数据行，
    跳过该行的禁止模式检查。豁免仅对当前行生效，不影响其他行或文件，
    不会降低当前权威指标审计的严格性。
    """
    findings: list[tuple[int, str, str]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if AUDIT_EXEMPT_MARKER in line:
            continue
        for label, pattern in FORBIDDEN_PATTERNS:
            if match := pattern.search(line):
                findings.append((line_number, label, match.group(0)))
    return findings


def validate_canonical_report(text: str) -> list[str]:
    """验证权威报告包含锁定数字、排名和 14→10 维说明。

    权威数字（2026-07-19 更新，50 seed × 5 episodes = N=250 验证）：
        - PPO 平均奖励 2746.94
        - FCFS 平均奖励 1458.77
        - PPO vs FCFS 提升 +88.3%
        - Obs10Wrapper（14→10 维兼容）
        - 14 维说明
    """
    errors: list[str] = []
    for expected in ("2746.94", "1458.77", "+88.3%", "Obs10Wrapper", "14 维"):
        if expected not in text:
            errors.append(f"权威报告缺少：{expected}")

    positions = []
    for rank, strategy in enumerate(CANONICAL_RANKING, 1):
        # 允许策略名后带括号注释，如 "PPO (14维)"
        row = re.search(
            rf"\|\s*{rank}\s*\|\s*(?:\*\*)?{re.escape(strategy)}"
            rf"(?:\s*\([^)]*\))?(?:\*\*)?\s*\|",
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
    """遍历参与数字审计的文本和 Office 文件。

    使用 os.walk 而非 Path.rglob，以便在遍历时直接跳过 SKIP_DIRS，
    避免进入损坏的符号链接目录（如 Windows 上的 .venv-mutmut/lib64）。
    同时跳过所有以 .venv 开头的目录（各种虚拟环境）和 site-packages。
    """
    import os

    for dirpath, dirnames, filenames in os.walk(root):
        # 原地修改 dirnames 以跳过 SKIP_DIRS（os.walk 约定）
        # 额外跳过所有 .venv* 开头的目录和 site-packages
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIRS
            and not d.startswith(".venv")
            and d not in ("site-packages", "lib", "lib64")
        ]
        for filename in filenames:
            path = Path(dirpath) / filename
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
