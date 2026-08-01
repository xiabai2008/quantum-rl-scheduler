"""Issue #912: validate_submission.py 的 must_pass_ci 与 must_have_readme 校验测试。

覆盖场景：
- must_pass_ci: 最新 CI 运行成功 / 失败 / gh CLI 不可用 / 无运行记录 / 运行进行中
- must_have_readme: README.md 存在 / 不存在
- 既有 git_tag 标签存在性校验未被破坏（标签存在 / 标签缺失）
- 端到端 validate_all() 集成校验
"""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.ci.validate_submission import SubmissionValidator


def _make_validator(project_root: Path) -> SubmissionValidator:
    """构造一个仅用于单元测试的 SubmissionValidator（绕过 manifest 加载）。

    Args:
        project_root: 校验器使用的项目根目录

    Returns:
        已初始化 errors/warnings/project_root 的 SubmissionValidator 实例
    """
    v = SubmissionValidator.__new__(SubmissionValidator)
    v.project_root = project_root
    v.errors = []
    v.warnings = []
    v.results = []
    return v


def _git_tag_item(
    *,
    must_pass_ci: bool = False,
    must_have_readme: bool = False,
    tag: str | None = "v9.1-submission",
) -> dict[str, Any]:
    """构造一个 git_tag 类型的提交物定义。

    Args:
        must_pass_ci: 是否声明 must_pass_ci 要求
        must_have_readme: 是否声明 must_have_readme 要求
        tag: 期望的 git 标签名；为 None 时不声明 tag 要求

    Returns:
        CODE_REPO 提交物定义字典
    """
    reqs: dict[str, Any] = {"must_pass_ci": must_pass_ci, "must_have_readme": must_have_readme}
    if tag is not None:
        reqs["tag"] = tag
    return {
        "id": "CODE_REPO",
        "name": "代码仓库",
        "type": "git_tag",
        "path": ".",
        "requirements": reqs,
    }


def _run_side_effect(git_stdout: str = "", gh_runs: list[dict[str, Any]] | None = None) -> Any:
    """生成 subprocess.run 的 side_effect，按命令分发 git/gh 返回。

    Args:
        git_stdout: ``git tag -l`` 命令的模拟标准输出
        gh_runs: ``gh run list`` 命令的模拟返回运行列表；None 表示空列表

    Returns:
        可用作 patch side_effect 的可调用对象
    """

    def _side_effect(*args: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        cmd = args[0] if args else kwargs.get("args", [])
        if cmd[:2] == ["git", "tag"]:
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=git_stdout, stderr="")
        if cmd[:3] == ["gh", "run", "list"]:
            stdout = json.dumps(gh_runs) if gh_runs is not None else "[]"
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    return _side_effect


class TestMustPassCi:
    """must_pass_ci 校验测试（Issue #912）。"""

    def test_ci_success(self, tmp_path: Path) -> None:
        """最新 CI 运行 success 时不应产生错误。"""
        v = _make_validator(tmp_path)
        item = _git_tag_item(must_pass_ci=True, must_have_readme=False, tag=None)
        messages: list[str] = []
        with (
            patch(
                "scripts.ci.validate_submission.shutil.which",
                return_value="/usr/bin/gh",
            ),
            patch(
                "scripts.ci.validate_submission.subprocess.run",
                side_effect=_run_side_effect(
                    gh_runs=[{"status": "completed", "conclusion": "success"}]
                ),
            ),
        ):
            v._validate_git_tag(item, messages)
        assert v.errors == []
        assert any("CI 运行通过" in m for m in messages)

    def test_ci_failure(self, tmp_path: Path) -> None:
        """最新 CI 运行 failure 时应产生错误。"""
        v = _make_validator(tmp_path)
        item = _git_tag_item(must_pass_ci=True, must_have_readme=False, tag=None)
        messages: list[str] = []
        with (
            patch(
                "scripts.ci.validate_submission.shutil.which",
                return_value="/usr/bin/gh",
            ),
            patch(
                "scripts.ci.validate_submission.subprocess.run",
                side_effect=_run_side_effect(
                    gh_runs=[{"status": "completed", "conclusion": "failure"}]
                ),
            ),
        ):
            v._validate_git_tag(item, messages)
        assert any("CI 运行未通过" in e for e in v.errors)
        assert any("CI 运行未通过" in m for m in messages)

    def test_ci_gh_unavailable(self, tmp_path: Path) -> None:
        """gh CLI 不可用时仅告警，不报错。"""
        v = _make_validator(tmp_path)
        item = _git_tag_item(must_pass_ci=True, must_have_readme=False, tag=None)
        messages: list[str] = []
        with patch("scripts.ci.validate_submission.shutil.which", return_value=None):
            v._validate_git_tag(item, messages)
        assert v.errors == []
        assert any("gh CLI 不可用" in w for w in v.warnings)

    def test_ci_no_runs(self, tmp_path: Path) -> None:
        """无 CI 运行记录时仅告警，不报错。"""
        v = _make_validator(tmp_path)
        item = _git_tag_item(must_pass_ci=True, must_have_readme=False, tag=None)
        messages: list[str] = []
        with (
            patch(
                "scripts.ci.validate_submission.shutil.which",
                return_value="/usr/bin/gh",
            ),
            patch(
                "scripts.ci.validate_submission.subprocess.run",
                side_effect=_run_side_effect(gh_runs=[]),
            ),
        ):
            v._validate_git_tag(item, messages)
        assert v.errors == []
        assert any("未找到任何 CI 运行记录" in w for w in v.warnings)

    def test_ci_in_progress(self, tmp_path: Path) -> None:
        """最新 CI 运行进行中时仅告警，不报错。"""
        v = _make_validator(tmp_path)
        item = _git_tag_item(must_pass_ci=True, must_have_readme=False, tag=None)
        messages: list[str] = []
        with (
            patch(
                "scripts.ci.validate_submission.shutil.which",
                return_value="/usr/bin/gh",
            ),
            patch(
                "scripts.ci.validate_submission.subprocess.run",
                side_effect=_run_side_effect(
                    gh_runs=[{"status": "in_progress", "conclusion": None}]
                ),
            ),
        ):
            v._validate_git_tag(item, messages)
        assert v.errors == []
        assert any("尚未完成" in w for w in v.warnings)

    def test_ci_called_process_error_warns(self, tmp_path: Path) -> None:
        """gh run list 执行失败（CalledProcessError）时仅告警，不报错。"""
        v = _make_validator(tmp_path)
        item = _git_tag_item(must_pass_ci=True, must_have_readme=False, tag=None)
        messages: list[str] = []

        def _raise(*args: Any, **kwargs: Any) -> Any:
            raise subprocess.CalledProcessError(returncode=1, cmd=args[0] if args else [])

        with (
            patch(
                "scripts.ci.validate_submission.shutil.which",
                return_value="/usr/bin/gh",
            ),
            patch("scripts.ci.validate_submission.subprocess.run", side_effect=_raise),
        ):
            v._validate_git_tag(item, messages)
        assert v.errors == []
        assert any("gh run list 执行失败" in w for w in v.warnings)


class TestMustHaveReadme:
    """must_have_readme 校验测试（Issue #912）。"""

    def test_readme_exists(self, tmp_path: Path) -> None:
        """README.md 存在时不报错。"""
        (tmp_path / "README.md").write_text("# Quantum RL Scheduler", encoding="utf-8")
        v = _make_validator(tmp_path)
        item = _git_tag_item(must_pass_ci=False, must_have_readme=True, tag=None)
        messages: list[str] = []
        v._validate_git_tag(item, messages)
        assert v.errors == []
        assert any("README.md 存在" in m for m in messages)

    def test_readme_missing(self, tmp_path: Path) -> None:
        """README.md 不存在时报错。"""
        v = _make_validator(tmp_path)
        item = _git_tag_item(must_pass_ci=False, must_have_readme=True, tag=None)
        messages: list[str] = []
        v._validate_git_tag(item, messages)
        assert any("README.md 不存在" in e for e in v.errors)
        assert any("README.md 不存在" in m for m in messages)

    def test_readme_in_subdir(self, tmp_path: Path) -> None:
        """item.path 指向子目录时，应校验该子目录下的 README.md。"""
        sub = tmp_path / "pkg"
        sub.mkdir()
        (sub / "README.md").write_text("nested", encoding="utf-8")
        v = _make_validator(tmp_path)
        item = _git_tag_item(must_pass_ci=False, must_have_readme=True, tag=None)
        item["path"] = "pkg"
        messages: list[str] = []
        v._validate_git_tag(item, messages)
        assert v.errors == []
        assert any("README.md 存在" in m for m in messages)


class TestGitTagCheckNotBroken:
    """既有 git_tag 标签存在性校验未被破坏（Issue #912 回归保护）。"""

    def test_tag_exists(self, tmp_path: Path) -> None:
        """标签存在时不报错。"""
        v = _make_validator(tmp_path)
        item = _git_tag_item(must_pass_ci=False, must_have_readme=False, tag="v9.1-submission")
        messages: list[str] = []
        with patch(
            "scripts.ci.validate_submission.subprocess.run",
            side_effect=_run_side_effect(git_stdout="v9.1-submission\n"),
        ):
            v._validate_git_tag(item, messages)
        assert v.errors == []
        assert any("标签存在" in m for m in messages)

    def test_tag_missing(self, tmp_path: Path) -> None:
        """标签缺失时报错。"""
        v = _make_validator(tmp_path)
        item = _git_tag_item(must_pass_ci=False, must_have_readme=False, tag="v9.1-submission")
        messages: list[str] = []
        with patch(
            "scripts.ci.validate_submission.subprocess.run",
            side_effect=_run_side_effect(git_stdout=""),
        ):
            v._validate_git_tag(item, messages)
        assert any("标签不存在" in e for e in v.errors)

    def test_combined_tag_and_ci_and_readme_all_pass(self, tmp_path: Path) -> None:
        """标签存在 + CI 通过 + README 存在时全部通过，无错误。"""
        (tmp_path / "README.md").write_text("# Quantum RL Scheduler v9.1.0", encoding="utf-8")
        v = _make_validator(tmp_path)
        item = _git_tag_item(must_pass_ci=True, must_have_readme=True, tag="v9.1-submission")
        messages: list[str] = []
        with (
            patch(
                "scripts.ci.validate_submission.shutil.which",
                return_value="/usr/bin/gh",
            ),
            patch(
                "scripts.ci.validate_submission.subprocess.run",
                side_effect=_run_side_effect(
                    git_stdout="v9.1-submission\n",
                    gh_runs=[{"status": "completed", "conclusion": "success"}],
                ),
            ),
        ):
            v._validate_git_tag(item, messages)
        assert v.errors == []
        assert any("标签存在" in m for m in messages)
        assert any("CI 运行通过" in m for m in messages)
        assert any("README.md 存在" in m for m in messages)


class TestValidateAllIntegration:
    """端到端 validate_all() 集成校验（Issue #912）。"""

    @staticmethod
    def _write_manifest(tmp_path: Path) -> Path:
        """在 tmp_path 下写入一个仅含 CODE_REPO 的 manifest 并返回路径。"""
        manifest = {
            "submission": {"deadline": "2026-09-15", "version": "9.1.0"},
            "items": [
                {
                    "id": "CODE_REPO",
                    "name": "代码仓库",
                    "type": "git_tag",
                    "path": ".",
                    "requirements": {
                        "tag": "v9.1-submission",
                        "must_pass_ci": True,
                        "must_have_readme": True,
                    },
                }
            ],
        }
        manifest_path = tmp_path / "submission_manifest.yaml"
        manifest_path.write_text(yaml.dump(manifest, allow_unicode=True), encoding="utf-8")
        return manifest_path

    def test_validate_all_passes_when_all_ok(self, tmp_path: Path) -> None:
        """标签存在 + CI 通过 + README 存在时 validate_all 返回 True。"""
        (tmp_path / "README.md").write_text("# Quantum RL Scheduler 9.1.0", encoding="utf-8")
        manifest_path = self._write_manifest(tmp_path)
        with (
            patch(
                "scripts.ci.validate_submission.shutil.which",
                return_value="/usr/bin/gh",
            ),
            patch(
                "scripts.ci.validate_submission.subprocess.run",
                side_effect=_run_side_effect(
                    git_stdout="v9.1-submission\n",
                    gh_runs=[{"status": "completed", "conclusion": "success"}],
                ),
            ),
        ):
            validator = SubmissionValidator(str(manifest_path), str(tmp_path))
            assert validator.validate_all() is True

    def test_validate_all_fails_when_readme_missing(self, tmp_path: Path) -> None:
        """README 缺失时 validate_all 返回 False。"""
        manifest_path = self._write_manifest(tmp_path)
        with (
            patch(
                "scripts.ci.validate_submission.shutil.which",
                return_value="/usr/bin/gh",
            ),
            patch(
                "scripts.ci.validate_submission.subprocess.run",
                side_effect=_run_side_effect(
                    git_stdout="v9.1-submission\n",
                    gh_runs=[{"status": "completed", "conclusion": "success"}],
                ),
            ),
        ):
            validator = SubmissionValidator(str(manifest_path), str(tmp_path))
            assert validator.validate_all() is False
            assert any("README.md 不存在" in e for e in validator.errors)

    def test_validate_all_fails_when_ci_failed(self, tmp_path: Path) -> None:
        """CI 失败时 validate_all 返回 False。"""
        (tmp_path / "README.md").write_text("# Quantum RL Scheduler 9.1.0", encoding="utf-8")
        manifest_path = self._write_manifest(tmp_path)
        with (
            patch(
                "scripts.ci.validate_submission.shutil.which",
                return_value="/usr/bin/gh",
            ),
            patch(
                "scripts.ci.validate_submission.subprocess.run",
                side_effect=_run_side_effect(
                    git_stdout="v9.1-submission\n",
                    gh_runs=[{"status": "completed", "conclusion": "failure"}],
                ),
            ),
        ):
            validator = SubmissionValidator(str(manifest_path), str(tmp_path))
            assert validator.validate_all() is False
            assert any("CI 运行未通过" in e for e in validator.errors)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
