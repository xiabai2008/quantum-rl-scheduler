#!/usr/bin/env python3
"""Parse total line coverage from coverage.xml and post a PR comment via gh.

Issue #260: CI 覆盖率评论助手。
- 读取 coverage.xml 顶层 line-rate 作为总覆盖率。
- 通过 gh pr comment 在当前 PR 上留言覆盖率数值。
- 使用环境变量 GITHUB_TOKEN 鉴权、PR_NUMBER 指定目标 PR。
- 在 PR 事件之外（如 push to main）或缺少参数时安全跳过，不报错。
"""
import os
import subprocess
import sys
import xml.etree.ElementTree as ET

COVERAGE_XML = "coverage.xml"


def main() -> int:
    # 非 PR 上下文或缺少 PR 编号时，跳过评论（保持幂等、不失败）。
    pr_number = os.environ.get("PR_NUMBER")
    if not pr_number:
        print("PR_NUMBER not set; skipping coverage comment.")
        return 0

    if not os.path.exists(COVERAGE_XML):
        print(f"{COVERAGE_XML} not found; skipping coverage comment.")
        return 0

    try:
        tree = ET.parse(COVERAGE_XML)
        root = tree.getroot()
        line_rate = float(root.get("line-rate", "0") or "0")
    except (ET.ParseError, ValueError, OSError) as exc:
        print(f"Failed to parse {COVERAGE_XML}: {exc}")
        return 0

    total_pct = round(line_rate * 100, 2)

    body = (
        "## Test Coverage Report (覆盖率)\n\n"
        f"- **Total line coverage**: `{total_pct}%`\n"
        f"- `coverage.xml` 已作为可下载 artifact 生成（artifact 名：`coverage-xml`）。\n\n"
        "> 本评论由 `ci(#253,#260)` 自动发布。当前仅展示本次构建的总覆盖率；"
        "如需对比基线请在 Codecov 查看 diff。"
    )

    env = os.environ.copy()
    cmd = ["gh", "pr", "comment", pr_number, "--body", body]
    try:
        subprocess.run(cmd, env=env, check=True)
    except subprocess.CalledProcessError as exc:
        print(f"Failed to post PR comment: {exc}")
        return 1

    print(f"Posted coverage comment to PR #{pr_number}: total={total_pct}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
