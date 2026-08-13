#!/usr/bin/env python
"""
文档同步检查脚本 (Documentation Sync Checker)

检测文档中的关键数字与项目实际状态是否一致，防止文档过时：
1. 文档中的测试用例数 vs 实际 `pytest --co -q` 收集数
2. 文档中的版本号 vs `pyproject.toml` 的 version 字段
3. AGENTS.md 中的 "open PR/issue" 表述 vs `gh pr/issue list --state open`
4. AGENTS.md 中的 "最后更新" 日期 vs 当天日期

用法:
    python scripts/ci/check_doc_sync.py
    python scripts/ci/check_doc_sync.py --skip-pytest   # 跳过耗时的pytest收集
    python scripts/ci/check_doc_sync.py --skip-gh        # 跳过gh CLI检查（无token时）
"""

from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

# 修复 Windows GBK 终端下 emoji 字符导致的 UnicodeEncodeError 崩溃
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# 需要检查测试用例数的文档（文件 → 期望出现的数字）
# 这些文件应包含统一的测试用例总数
DOCS_WITH_TEST_COUNT: list[Path] = [
    _PROJECT_ROOT / "AGENTS.md",
    _PROJECT_ROOT / "README.md",
    _PROJECT_ROOT / "docs" / "authoritative_numbers.md",
    _PROJECT_ROOT / "docs" / "code_freeze.md",
    _PROJECT_ROOT / "docs" / "requirements_traceability.md",
    # 注：答辩PPT大纲.md 已被 #891 文档清理删除，不再列入检查
]

# 已废弃的旧测试数（文档中不应再出现，历史CHANGELOG段除外）
# 8.7-v4 外部红队修复：补充 3717/3738/3711/3732 等近几轮旧口径，
# 防止测试数口径漂移（红队实测 doc_sync 声称 3720 vs CI 实际 3695 vs 全量 3741）。
# 注意：check_doc_sync 的权威口径 = 主套件收集（排除 benchmark）= 3720，
# 全量（含 benchmark）= 3741；两者都不可作为"废弃数"。
DEPRECATED_TEST_COUNTS = [
    "2824+",
    "2824",
    "3106",
    "3359",
    "3467",
    "3385",
    "3695",
    "3696",
    "3697",
    "3709",
    "3711",
    "3717",
    "3730",
    "3732",
    "3738",
]

# 需要检查版本号的文档
DOCS_WITH_VERSION: list[Path] = [
    _PROJECT_ROOT / "README.md",
    _PROJECT_ROOT / "SECURITY.md",
    _PROJECT_ROOT / "AGENTS.md",
]

AGENTS_MD = _PROJECT_ROOT / "AGENTS.md"
PYPROJECT_TOML = _PROJECT_ROOT / "pyproject.toml"


class CheckResult(NamedTuple):
    """单项检查结果。"""

    name: str
    passed: bool
    detail: str
    # warning=True 表示信息性提醒，不计入失败数（用于实时变化的指标如 open issue 数）
    warning: bool = False


def _read_text(path: Path) -> str:
    """安全读取文件文本。"""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def get_pytest_count() -> int | None:
    """运行 pytest --collect-only 获取实际测试用例数。"""
    try:
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--ignore=tests/test_performance_benchmarks.py",
                "--ignore=tests/benchmarks",
                "-m",
                "not benchmark",
                "--co",
                "-q",
            ],
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode not in (0, 5):
        return None

    # 匹配末尾 "===== 3523 tests collected in 12.17s ====="
    match = re.search(r"(\d+) tests? collected", result.stdout)
    if match:
        return int(match.group(1))
    return None


def get_pyproject_version() -> str | None:
    """从 pyproject.toml 提取 version 字段。"""
    text = _read_text(PYPROJECT_TOML)
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else None


def get_gh_open_count(kind: str) -> int | None:
    """通过 gh CLI 获取 open PR/issue 数量。kind: 'pr' 或 'issue'。"""
    cmd = ["gh", kind, "list", "--state", "open", "--limit", "200", "--json", "number"]
    try:
        result = subprocess.run(
            cmd,
            cwd=_PROJECT_ROOT,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None

    if result.returncode != 0:
        return None

    # gh --json number 输出 [{"number": 773}, ...]，用 json 解析确保准确
    text = result.stdout.strip()
    if not text or text == "[]":
        return 0
    try:
        data = json.loads(text)
        return len(data) if isinstance(data, list) else None
    except json.JSONDecodeError:
        return None


def check_test_count(expected: int) -> list[CheckResult]:
    """检查文档中的测试用例数是否与实际一致。"""
    results: list[CheckResult] = []
    expected_str = str(expected)

    for doc in DOCS_WITH_TEST_COUNT:
        if not doc.exists():
            results.append(CheckResult(f"test_count:{doc.name}", False, "文件不存在"))
            continue
        text = _read_text(doc)
        # 文档应包含期望的测试数
        if expected_str not in text:
            results.append(
                CheckResult(
                    f"test_count:{doc.name}",
                    False,
                    f"未找到测试用例数 {expected_str}",
                )
            )
        else:
            results.append(CheckResult(f"test_count:{doc.name}", True, f"包含 {expected_str}"))

    # 检查废弃旧数（CHANGELOG 历史段允许，其他文档不允许）
    # 8.7-v4 修复：AGENTS.md 的"最后更新"变更日志段是合法历史记录（记录各轮
    # 修复时点使用的旧口径数字），此前豁免逻辑只写在注释里从未实现，导致
    # changelog 里的历史测试数（如 3696/3717）被误报为违规。
    changelog_re = re.compile(r"最后更新.*?(?=\n\*{3,}|\Z)", re.DOTALL)
    for doc in DOCS_WITH_TEST_COUNT:
        text = _read_text(doc)
        # 提取 changelog 段（仅 AGENTS.md 有此结构；其他文档无 changelog 段则豁免为空）
        m = changelog_re.search(text)
        changelog_text = m.group(0) if m else ""
        body_text = text.replace(changelog_text, "")
        for old in DEPRECATED_TEST_COUNTS:
            # 精确匹配旧数（避免误报，如 "2824" 在 "28240" 中）
            pattern = rf"(?<!\d){re.escape(old)}(?!\d)(?:\+)?"
            # changelog 段内的旧数属于历史记录，豁免；正文中则违规
            if re.search(pattern, body_text):
                results.append(
                    CheckResult(
                        f"deprecated_count:{doc.name}",
                        False,
                        f"发现废弃旧测试数 '{old}'，应统一为 {expected_str}",
                    )
                )
                break

    return results


def check_version(expected: str) -> list[CheckResult]:
    """检查文档中的版本号是否与 pyproject.toml 一致。"""
    results: list[CheckResult] = []

    for doc in DOCS_WITH_VERSION:
        text = _read_text(doc)
        # 检查 Version: x.y.z 或 version = "x.y.z" 模式
        if expected not in text:
            results.append(CheckResult(f"version:{doc.name}", False, f"未找到版本号 {expected}"))
        else:
            results.append(CheckResult(f"version:{doc.name}", True, f"包含版本号 {expected}"))

    # 检查 AGENTS.md Version 行
    agents_text = _read_text(AGENTS_MD)
    if f"Version: {expected}" not in agents_text:
        results.append(
            CheckResult(
                "version:AGENTS.md header",
                False,
                f"AGENTS.md 头部未找到 'Version: {expected}'",
            )
        )

    return results


def check_agents_date(today: dt.date) -> CheckResult:
    """检查 AGENTS.md 的'最后更新'日期是否为今天。

    Issue #807 遗留：'最后更新'是手写日期，任何非当天提交都会触发不匹配，
    若作为硬失败会让 CI 在非更新日必然红灯。改为 warning（信息性提醒），
    由维护者在实质更新文档时顺手刷新日期。
    """
    text = _read_text(AGENTS_MD)
    today_str = today.strftime("%Y-%m-%d")
    # 匹配 "最后更新**：2026-07-31" 或 "最后更新：2026-07-31"
    match = re.search(r"最后更新[*]*[：:]\s*(\d{4}-\d{2}-\d{2})", text)
    if not match:
        return CheckResult("agents_update_date", False, "AGENTS.md 未找到'最后更新'日期")
    doc_date = match.group(1)
    if doc_date == today_str:
        return CheckResult("agents_update_date", True, f"最后更新日期为今天 {today_str}")
    return CheckResult(
        "agents_update_date",
        False,
        f"AGENTS.md 最后更新日期 {doc_date} != 今天 {today_str}（文档可能过时；信息性提醒，不阻断 CI）",
        warning=True,
    )


def check_open_pr_issue(open_pr: int | None, open_issue: int | None) -> list[CheckResult]:
    """检查 AGENTS.md 中 open PR/issue 表述是否与 gh CLI 一致。

    注意：open issue 数会随团队日常 issue 创建/关闭实时变化，AGENTS.md 只能记录
    快照值。因此不匹配时标记为 warning（信息性提醒），不计入 CI 失败数。
    """
    results: list[CheckResult] = []
    text = _read_text(AGENTS_MD)

    if open_pr is not None:
        head_match = re.search(r"(\d+)\s*open PR", text)
        if head_match:
            doc_pr = int(head_match.group(1))
            if doc_pr == open_pr:
                results.append(
                    CheckResult(
                        "open_pr",
                        True,
                        f"AGENTS.md open PR={doc_pr} 与 gh CLI {open_pr} 一致",
                    )
                )
            else:
                # open PR 数不匹配视为失败（PR 数变化不频繁，应保持同步）
                results.append(
                    CheckResult(
                        "open_pr",
                        False,
                        f"AGENTS.md open PR={doc_pr} != gh CLI {open_pr}",
                    )
                )
        else:
            results.append(CheckResult("open_pr", False, "AGENTS.md 未找到 'open PR' 表述"))

    if open_issue is not None:
        head_match = re.search(r"(\d+)\s*open issue", text)
        if head_match:
            doc_issue = int(head_match.group(1))
            if doc_issue == open_issue:
                results.append(
                    CheckResult(
                        "open_issue",
                        True,
                        f"AGENTS.md open issue={doc_issue} 与 gh CLI {open_issue} 一致",
                    )
                )
            else:
                # open issue 数实时变化，不匹配标记为 warning 不阻断 CI
                results.append(
                    CheckResult(
                        "open_issue",
                        False,
                        f"AGENTS.md open issue={doc_issue} != gh CLI {open_issue}（信息性提醒：issue数实时变化，请定期同步）",
                        warning=True,
                    )
                )
        else:
            results.append(
                CheckResult(
                    "open_issue",
                    False,
                    "AGENTS.md 未找到 'open issue' 表述",
                    warning=True,
                )
            )

    return results


def main() -> int:
    """主入口：运行所有检查，返回退出码（0=全通过，1=有不一致）。"""
    import argparse

    parser = argparse.ArgumentParser(description="文档同步检查")
    parser.add_argument("--skip-pytest", action="store_true", help="跳过 pytest 收集（耗时）")
    parser.add_argument("--skip-gh", action="store_true", help="跳过 gh CLI 检查（无token时）")
    args = parser.parse_args()

    today = dt.date.today()
    all_results: list[CheckResult] = []

    print("=" * 70)
    print("文档同步检查 (Documentation Sync Checker)")
    print(f"日期: {today.strftime('%Y-%m-%d')}")
    print("=" * 70)

    # 1. 版本号检查（快，先跑）
    pyproject_version = get_pyproject_version()
    if pyproject_version is None:
        all_results.append(
            CheckResult("pyproject_version", False, "无法从 pyproject.toml 读取 version")
        )
    else:
        print(f"\n[1] 版本号检查 (pyproject.toml version={pyproject_version})")
        all_results.extend(check_version(pyproject_version))

    # 2. 测试用例数检查
    if args.skip_pytest:
        print("\n[2] 测试用例数检查: 已跳过 (--skip-pytest)")
    else:
        print("\n[2] 测试用例数检查 (运行 pytest --co -q)...")
        pytest_count = get_pytest_count()
        if pytest_count is None:
            all_results.append(
                CheckResult("pytest_count", False, "无法获取 pytest 收集数（pytest 未安装或超时）")
            )
            print("    ⚠ 无法获取 pytest 收集数")
        else:
            print(f"    实际测试用例数: {pytest_count}")
            all_results.extend(check_test_count(pytest_count))

    # 3. AGENTS.md 最后更新日期检查
    print(f"\n[3] AGENTS.md 最后更新日期检查 (今天: {today.strftime('%Y-%m-%d')})")
    all_results.append(check_agents_date(today))

    # 4. open PR/issue 检查
    if args.skip_gh:
        print("\n[4] open PR/issue 检查: 已跳过 (--skip-gh)")
    else:
        print("\n[4] open PR/issue 检查 (gh CLI)...")
        open_pr = get_gh_open_count("pr")
        open_issue = get_gh_open_count("issue")
        if open_pr is not None:
            print(f"    gh open PR: {open_pr}")
        else:
            print("    ⚠ 无法获取 gh open PR（gh 未认证或未安装）")
        if open_issue is not None:
            print(f"    gh open issue: {open_issue}")
        else:
            print("    ⚠ 无法获取 gh open issue（gh 未认证或未安装）")
        all_results.extend(check_open_pr_issue(open_pr, open_issue))

    # 汇总
    print("\n" + "=" * 70)
    print("检查结果汇总")
    print("=" * 70)
    passed = sum(1 for r in all_results if r.passed)
    failed = sum(1 for r in all_results if not r.passed and not r.warning)
    warnings = sum(1 for r in all_results if not r.passed and r.warning)
    for r in all_results:
        if r.passed:
            status = "✅"
        elif r.warning:
            status = "⚠️"
        else:
            status = "❌"
        print(f"  {status} {r.name}: {r.detail}")

    print(f"\n总计: {passed} 通过, {failed} 失败, {warnings} 警告")
    if failed > 0:
        print("\n❌ 文档同步检查未通过，请修复上述不一致项。")
        return 1
    if warnings > 0:
        print("\n⚠️ 存在信息性警告（open issue 数实时变化，请定期同步 AGENTS.md）。")
    print("\n✅ 文档同步检查通过。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
