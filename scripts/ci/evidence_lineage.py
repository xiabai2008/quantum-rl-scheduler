#!/usr/bin/env python3
"""
实验证据谱系与一键复核包生成器

Issue #8 (P0 评委审查)：建立不可抵赖的实验证据谱系与一键复核包。

核心能力：
1. 扫描仓库中所有权威实验数字（PPO=2746.94 / FCFS=1458.77 / +88.3% 等），
   记录每个出现位置的文件、行号。
2. 对每个出现位置调用 ``git blame`` 获取最后修改该行的 commit、作者、时间，
   形成 commit → 数字 → 报告 的可追溯链。
3. 扫描所有文本文件中的本机路径泄漏（Windows: ``C:\\Users\\`` / ``d:\\桌面\\`` 等；
   Unix: ``/home/`` / ``/Users/`` 等），避免评委复核时出现他人的私有路径。
4. 生成 Markdown 证据谱系报告，列出权威数字溯源表、本机路径泄漏清单、
   以及一键复核命令清单。

使用示例：

    # 默认：扫描仓库并打印证据谱系摘要
    python scripts/ci/evidence_lineage.py

    # 生成 Markdown 报告到指定路径
    python scripts/ci/evidence_lineage.py --report results/reports/evidence_lineage.md

    # 仅检测本机路径泄漏（用于 CI 阻断）
    python scripts/ci/evidence_lineage.py --check-leaks

作者：量子RL调度系统团队
日期：2026-07-22
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# 权威数字定义（与 audit_authoritative_metrics.py 保持一致）
# ---------------------------------------------------------------------------
# 这里只锁定"对外宣传的最终数字"，避免把每个细分指标都纳入追溯。
CANONICAL_METRICS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("PPO 平均奖励", re.compile(r"(?<![\d.])2746\.94(?![\d])")),
    ("FCFS 平均奖励", re.compile(r"(?<![\d.])1458\.77(?![\d])")),
    ("PPO 相对 FCFS 提升", re.compile(r"\+88\.3%")),
    ("Cohen's d", re.compile(r"(?<![\d.])-?1\.70(?![\d])")),
    ("Welch t 检验 p 值", re.compile(r"3\.04[eE]-11")),
    ("N=250 样本量", re.compile(r"(?<![\d])N\s*=\s*250(?![\d])")),
)

# ---------------------------------------------------------------------------
# 本机路径检测模式
# ---------------------------------------------------------------------------
# Windows: C:\Users\xxx / D:\桌面\xxx / d:\桌面\
# Unix: /home/xxx / /Users/xxx
# file:// 协议指向本机路径
LEAK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("Windows 用户目录", re.compile(r"[a-zA-Z]:\\Users\\", re.IGNORECASE)),
    ("Windows 桌面路径", re.compile(r"[a-zA-Z]:\\[^\\]*\\Desktop\\", re.IGNORECASE)),
    ("Windows 中文桌面", re.compile(r"[a-zA-Z]:\\桌面\\", re.IGNORECASE)),
    ("Unix home 目录", re.compile(r"/home/[a-zA-Z0-9_]+/")),
    ("Unix Users 目录", re.compile(r"/Users/[a-zA-Z0-9_]+/")),
    ("file:// 本机协议", re.compile(r"file:///")),
)

# 跳过扫描的目录（与 audit_authoritative_metrics.py 对齐）
SKIP_DIRS: set[str] = {
    ".git",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "__pycache__",
    "venv",
    ".venv",
    ".venv-mutmut",
    "node_modules",
    "site-packages",
    "lib",
    "lib64",
    "dist",
    "build",
    "htmlcov",
    ".trae-html-share-packages",
}

# 参与扫描的文本后缀
TEXT_SUFFIXES: set[str] = {".md", ".py", ".txt", ".rst", ".yaml", ".yml", ".json", ".sh"}

# 明确豁免本机路径检测的文件（自身就是用来检测/演示路径的）
LEAK_EXEMPT_FILES: set[Path] = {
    Path("scripts/ci/evidence_lineage.py"),
    Path("tests/test_evidence_lineage.py"),
    Path("tests/test_platform_compat.py"),  # 测试用例使用示例路径 C:/Users/test
    # 脚本自身生成的报告会回显泄漏行内容，不应再次扫描
    Path("results/reports/evidence_lineage.md"),
    # 历史日志快照，不应修改其中已固化的路径
    Path("mutmut_env_results.txt"),
}

# 完全跳过扫描的文件（既不扫权威数字，也不扫泄漏）
SELF_EXCLUDES: set[Path] = {
    Path("scripts/ci/evidence_lineage.py"),
    Path("tests/test_evidence_lineage.py"),
    Path("results/reports/evidence_lineage.md"),
    Path("results/reports/authoritative_metric_audit.md"),
}

# 已知的合理路径模式（容器内标准路径、CI 系统路径等），不视为泄漏
# 这些路径在所有开发者机器上一致，不会泄漏私有信息
LEAK_EXEMPT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"/home/vscode/"),  # devcontainer 标准用户主目录
    re.compile(r"/home/runner/"),  # GitHub Actions runner 标准路径
)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------
@dataclass
class MetricOccurrence:
    """权威数字的单次出现"""

    metric_name: str
    file_path: str  # 相对仓库根的 POSIX 路径
    line_number: int
    line_content: str
    commit: str = ""
    author: str = ""
    commit_date: str = ""


@dataclass
class LeakOccurrence:
    """本机路径泄漏的单次出现"""

    leak_type: str
    file_path: str
    line_number: int
    line_content: str


@dataclass
class EvidenceReport:
    """证据谱系报告"""

    repo_root: str
    generated_at: str
    current_commit: str
    current_branch: str
    metric_occurrences: list[MetricOccurrence] = field(default_factory=list)
    leak_occurrences: list[LeakOccurrence] = field(default_factory=list)

    @property
    def has_leaks(self) -> bool:
        """是否存在本机路径泄漏"""
        return bool(self.leak_occurrences)

    @property
    def metric_summary(self) -> dict[str, int]:
        """按指标名分组的出现次数"""
        summary: dict[str, int] = {}
        for occ in self.metric_occurrences:
            summary[occ.metric_name] = summary.get(occ.metric_name, 0) + 1
        return summary


# ---------------------------------------------------------------------------
# 核心扫描逻辑
# ---------------------------------------------------------------------------
def iter_text_files(root: Path) -> Iterable[Path]:
    """遍历参与扫描的文本文件。

    使用 os.walk 并就地修改 dirnames 跳过 SKIP_DIRS，避免进入虚拟环境等。
    """
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".venv")]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in TEXT_SUFFIXES:
                yield path


def find_metric_occurrences(text: str, file_path: str) -> list[MetricOccurrence]:
    """在单个文件文本中查找所有权威数字出现位置。

    Args:
        text: 文件全文
        file_path: 相对仓库根的 POSIX 路径（仅用于记录）

    Returns:
        出现位置列表（未填充 commit/author/date，由 blame 阶段补充）
    """
    occurrences: list[MetricOccurrence] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for metric_name, pattern in CANONICAL_METRICS:
            if pattern.search(line):
                occurrences.append(
                    MetricOccurrence(
                        metric_name=metric_name,
                        file_path=file_path,
                        line_number=line_number,
                        line_content=line.strip(),
                    )
                )
    return occurrences


def find_leak_occurrences(text: str, file_path: str) -> list[LeakOccurrence]:
    """在单个文件文本中查找所有本机路径泄漏位置。

    已知合理路径（如 devcontainer 的 ``/home/vscode/``、GitHub Actions 的
    ``/home/runner/``）通过 LEAK_EXEMPT_PATTERNS 豁免，不计为泄漏。
    """
    occurrences: list[LeakOccurrence] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        for leak_type, pattern in LEAK_PATTERNS:
            if not pattern.search(line):
                continue
            # 检查是否命中豁免模式（如 /home/vscode/）
            if any(exempt.search(line) for exempt in LEAK_EXEMPT_PATTERNS):
                continue
            occurrences.append(
                LeakOccurrence(
                    leak_type=leak_type,
                    file_path=file_path,
                    line_number=line_number,
                    line_content=line.strip(),
                )
            )
    return occurrences


def get_git_info(root: Path) -> tuple[str, str]:
    """获取当前仓库的 commit 和 branch。"""
    try:
        commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(root), stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        commit = "unknown"
    try:
        branch = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=str(root),
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        branch = "unknown"
    return commit, branch


def git_blame_line(root: Path, file_path: str, line_number: int) -> tuple[str, str, str]:
    """对指定文件的指定行执行 git blame，返回 (commit, author, date)。

    失败时返回三个空字符串，不阻断整体扫描。
    """
    try:
        result = subprocess.run(
            [
                "git",
                "blame",
                "-L",
                f"{line_number},{line_number}",
                "--porcelain",
                "--",
                file_path,
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            return "", "", ""
        lines = result.stdout.splitlines()
        if not lines:
            return "", "", ""
        header = lines[0].split()
        commit = header[0] if header else ""
        if len(commit) > 12:
            commit = commit[:12]
        author = ""
        date = ""
        for ln in lines:
            if ln.startswith("author "):
                author = ln[len("author ") :]
            elif ln.startswith("author-time "):
                ts = ln[len("author-time ") :]
                try:
                    date = datetime.fromtimestamp(int(ts)).strftime("%Y-%m-%d")
                except (ValueError, OSError):
                    date = ""
        return commit, author, date
    except Exception:
        return "", "", ""


def scan_repository(root: Path, with_blame: bool = True) -> EvidenceReport:
    """扫描整个仓库，生成证据谱系报告。

    Args:
        root: 仓库根目录
        with_blame: 是否对每个权威数字位置执行 git blame（CI 中可关闭以加速）

    Returns:
        EvidenceReport 对象
    """
    commit, branch = get_git_info(root)
    report = EvidenceReport(
        repo_root=str(root),
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        current_commit=commit,
        current_branch=branch,
    )

    for path in iter_text_files(root):
        relative_path = path.relative_to(root)
        # 跳过自身和相关测试/报告，避免自引用
        if relative_path in SELF_EXCLUDES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError):
            continue
        relative = relative_path.as_posix()

        # 权威数字
        for occ in find_metric_occurrences(text, relative):
            if with_blame:
                blame_commit, blame_author, blame_date = git_blame_line(
                    root, relative, occ.line_number
                )
                occ.commit = blame_commit
                occ.author = blame_author
                occ.commit_date = blame_date
            report.metric_occurrences.append(occ)

        # 本机路径泄漏（自身脚本豁免）
        if relative_path not in LEAK_EXEMPT_FILES:
            report.leak_occurrences.extend(find_leak_occurrences(text, relative))

    return report


# ---------------------------------------------------------------------------
# 报告渲染
# ---------------------------------------------------------------------------
def render_markdown(report: EvidenceReport) -> str:
    """将证据谱系报告渲染为 Markdown。"""
    lines: list[str] = []
    lines.append("# 实验证据谱系与一键复核包")
    lines.append("")
    lines.append(f"**生成时间**：{report.generated_at}  ")
    lines.append(f"**仓库 commit**：`{report.current_commit}`  ")
    lines.append(f"**分支**：`{report.current_branch}`  ")
    lines.append("")

    # 1. 权威数字汇总
    lines.append("## 1. 权威数字汇总")
    lines.append("")
    summary = report.metric_summary
    if summary:
        lines.append("| 指标 | 出现次数 |")
        lines.append("|------|----------|")
        for metric_name, count in sorted(summary.items()):
            lines.append(f"| {metric_name} | {count} |")
    else:
        lines.append("_未在仓库中扫描到权威数字。_")
    lines.append("")

    # 2. 权威数字溯源表
    lines.append("## 2. 权威数字溯源表（commit → 数字 → 报告）")
    lines.append("")
    if report.metric_occurrences:
        lines.append("| 指标 | 文件 | 行号 | 来源 commit | 作者 | 时间 |")
        lines.append("|------|------|------|-------------|------|------|")
        for occ in report.metric_occurrences:
            content_preview = occ.line_content[:60].replace("|", "\\|")
            if len(occ.line_content) > 60:
                content_preview += "..."
            lines.append(
                f"| {occ.metric_name} | `{occ.file_path}` | {occ.line_number} | "
                f"`{occ.commit or 'N/A'}` | {occ.author or 'N/A'} | {occ.commit_date or 'N/A'} |"
            )
        lines.append("")
        lines.append("> 行内容预览（前 60 字符）见原始报告；完整内容请按文件+行号定位。")
    else:
        lines.append("_无权威数字出现位置。_")
    lines.append("")

    # 3. 本机路径泄漏清单
    lines.append("## 3. 本机路径泄漏清单")
    lines.append("")
    if report.leak_occurrences:
        lines.append(
            f"⚠️ 发现 **{len(report.leak_occurrences)}** 处本机路径泄漏，需在代码冻结前清理："
        )
        lines.append("")
        lines.append("| 类型 | 文件 | 行号 | 行内容预览 |")
        lines.append("|------|------|------|------------|")
        for occ in report.leak_occurrences:
            preview = occ.line_content[:80].replace("|", "\\|")
            if len(occ.line_content) > 80:
                preview += "..."
            lines.append(f"| {occ.leak_type} | `{occ.file_path}` | {occ.line_number} | {preview} |")
    else:
        lines.append("✅ 未发现本机路径泄漏。")
    lines.append("")

    # 4. 一键复核命令清单
    lines.append("## 4. 一键复核命令清单")
    lines.append("")
    lines.append("评委可按以下步骤复核实验结果：")
    lines.append("")
    lines.append("```bash")
    lines.append("# 1. 克隆仓库并切换到本报告对应的 commit")
    lines.append("git clone <repo-url> && cd quantum-rl-scheduler")
    lines.append(f"git checkout {report.current_commit}")
    lines.append("")
    lines.append("# 2. 安装依赖（含退火库 dimod/dwave-neal）")
    lines.append("pip install -r requirements.txt")
    lines.append("")
    lines.append("# 3. 运行权威数字审计（验证八策略排名和核心数字一致）")
    lines.append("python scripts/ci/audit_authoritative_metrics.py")
    lines.append("")
    lines.append("# 4. 运行提交物清单校验")
    lines.append("python scripts/ci/validate_submission.py --check")
    lines.append("")
    lines.append("# 5. 运行多 seed 评估复现（50 seeds × 5 episodes = N=250）")
    lines.append("python scripts/evaluation/run_multiseed_evaluation.py --seeds 50 --episodes 5")
    lines.append("")
    lines.append("# 6. 统计显著性检验")
    lines.append(
        "python scripts/evaluation/statistical_significance.py "
        "--input results/multiseed_evaluation/rewards_multiseed.json"
    )
    lines.append("")
    lines.append("# 7. 重新生成本证据谱系报告（应与本报告一致）")
    lines.append(
        "python scripts/ci/evidence_lineage.py --report results/reports/evidence_lineage.md"
    )
    lines.append("```")
    lines.append("")

    # 5. 数字一致性声明
    lines.append("## 5. 数字一致性声明")
    lines.append("")
    lines.append(
        "本报告通过 `scripts/ci/evidence_lineage.py` 自动生成，"
        "所有权威数字的出现位置均经 `git blame` 追溯到具体 commit。"
    )
    lines.append(
        "如评委发现报告中数字与 `results/reports/strategy_comparison.md` 不一致，"
        "请以 `audit_authoritative_metrics.py` 的审计结果为准。"
    )
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------
def main() -> int:
    """命令行入口。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="项目根目录（默认：脚本所在仓库根）",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="将 Markdown 报告写入指定路径（不指定则只打印摘要）",
    )
    parser.add_argument(
        "--check-leaks",
        action="store_true",
        help="仅检测本机路径泄漏，发现泄漏时以非零退出码阻断 CI",
    )
    parser.add_argument(
        "--no-blame",
        action="store_true",
        help="跳过 git blame 阶段（加速扫描，但溯源表无 commit 信息）",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not (root / ".git").exists():
        print(f"错误：{root} 不是 git 仓库根目录", file=sys.stderr)
        return 2

    report = scan_repository(root, with_blame=not args.no_blame)

    if args.check_leaks:
        if report.has_leaks:
            print(f"❌ 发现 {len(report.leak_occurrences)} 处本机路径泄漏：")
            for occ in report.leak_occurrences:
                print(
                    f"  - {occ.leak_type}: {occ.file_path}:{occ.line_number} -> "
                    f"{occ.line_content[:80]}"
                )
            return 1
        print("✅ 未发现本机路径泄漏。")
        return 0

    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(render_markdown(report), encoding="utf-8")
        print(f"✅ 证据谱系报告已写入：{args.report}")
    else:
        # 打印摘要
        print("=== 实验证据谱系摘要 ===")
        print(f"仓库 commit: {report.current_commit}")
        print(f"分支: {report.current_branch}")
        print(f"生成时间: {report.generated_at}")
        print()
        print("权威数字出现次数：")
        for metric_name, count in sorted(report.metric_summary.items()):
            print(f"  - {metric_name}: {count}")
        print()
        if report.has_leaks:
            print(f"⚠️  本机路径泄漏：{len(report.leak_occurrences)} 处")
            for occ in report.leak_occurrences[:10]:
                print(f"  - {occ.leak_type}: {occ.file_path}:{occ.line_number}")
            if len(report.leak_occurrences) > 10:
                print(f"  ...（共 {len(report.leak_occurrences)} 处，详见 --report 输出）")
        else:
            print("✅ 未发现本机路径泄漏。")

    # 即使有泄漏也返回 0（除非 --check-leaks 模式），避免阻断开发流程
    return 0


if __name__ == "__main__":
    sys.exit(main())
