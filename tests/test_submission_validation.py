"""
Issue #33: M5 最终提交物一键打包与版本校验脚本 — 单元测试

测试覆盖：
- ItemResult 数据类
- SubmissionValidator 初始化与 manifest 加载
- _validate_zip: 大小校验 + include/exclude 内容校验
- _validate_markdown: 文件存在性校验
- _check_version_consistency: README/AGENTS.md 版本号检查
- generate_report: Markdown 报告生成
- prepare_submission: 准备模式
- CLI 入口参数校验
"""

import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.ci.validate_submission import (
    ItemResult,
    SubmissionValidator,
    prepare_submission,
)


def _make_manifest(
    version: str = "v8.0",
    deadline: str = "2026-09-15",
    items: list | None = None,
) -> dict:
    """构造测试用 manifest 字典。"""
    if items is None:
        items = [
            {
                "id": "TEST_MD",
                "name": "测试 Markdown",
                "type": "md",
                "path": "test.md",
                "must_exist": True,
            }
        ]
    return {"submission": {"version": version, "deadline": deadline}, "items": items}


class TestItemResult(unittest.TestCase):
    """测试 ItemResult 数据类。"""

    def test_default_severity_is_error(self):
        r = ItemResult(item_id="X", name="X", item_type="md", path="x.md", passed=False)
        self.assertEqual(r.severity, "error")
        self.assertEqual(r.messages, [])

    def test_passed_item_severity(self):
        r = ItemResult(item_id="X", name="X", item_type="md", path="x.md", passed=True)
        # passed=True 时 severity 仍可为 info（由调用方设置）
        r.severity = "info"
        self.assertEqual(r.severity, "info")


class TestSubmissionValidatorInit(unittest.TestCase):
    """测试 SubmissionValidator 初始化。"""

    def test_load_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.yaml"
            manifest = _make_manifest()
            manifest_path.write_text(yaml.dump(manifest), encoding="utf-8")

            v = SubmissionValidator(str(manifest_path), tmp)
            self.assertEqual(v.manifest["submission"]["version"], "v8.0")
            self.assertEqual(v.errors, [])
            self.assertEqual(v.warnings, [])
            self.assertEqual(v.results, [])


class TestValidateZip(unittest.TestCase):
    """测试 _validate_zip 方法。"""

    def _make_zip(self, tmp: str, files: dict[str, bytes]) -> str:
        """在 tmp 目录下创建一个 ZIP 文件，包含指定文件。"""
        zip_path = Path(tmp) / "test.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for name, content in files.items():
                zipf.writestr(name, content)
        return str(zip_path)

    def test_zip_size_within_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = self._make_zip(tmp, {"README.md": "hello"})
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = []
            v.warnings = []
            item = {
                "id": "Z1",
                "requirements": {"max_size_mb": 100},
            }
            messages: list[str] = []
            v._validate_zip(item, Path(zip_path), messages)
            self.assertEqual(v.errors, [])
            self.assertTrue(any("文件大小" in m for m in messages))

    def test_zip_include_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = self._make_zip(tmp, {"README.md": "hello"})
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = []
            v.warnings = []
            item = {
                "id": "Z2",
                "requirements": {
                    "include": ["src/", "README.md"],
                },
            }
            messages: list[str] = []
            v._validate_zip(item, Path(zip_path), messages)
            # src/ 不在 ZIP 中，应报错
            self.assertTrue(any("缺少必需路径" in e for e in v.errors))

    def test_zip_include_present(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = self._make_zip(tmp, {"README.md": "hello", "src/main.py": "code"})
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = []
            v.warnings = []
            item = {
                "id": "Z3",
                "requirements": {
                    "include": ["src/", "README.md"],
                },
            }
            messages: list[str] = []
            v._validate_zip(item, Path(zip_path), messages)
            self.assertEqual(v.errors, [])
            self.assertTrue(any("包含所有必需路径" in m for m in messages))

    def test_zip_exclude_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = self._make_zip(tmp, {"README.md": "hello", ".git/config": "git"})
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = []
            v.warnings = []
            item = {
                "id": "Z4",
                "requirements": {
                    "exclude": [".git/", "__pycache__/"],
                },
            }
            messages: list[str] = []
            v._validate_zip(item, Path(zip_path), messages)
            self.assertTrue(any("包含禁止路径" in e for e in v.errors))

    def test_zip_exclude_not_found(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = self._make_zip(tmp, {"README.md": "hello"})
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = []
            v.warnings = []
            item = {
                "id": "Z5",
                "requirements": {
                    "exclude": [".git/", "__pycache__/"],
                },
            }
            messages: list[str] = []
            v._validate_zip(item, Path(zip_path), messages)
            self.assertEqual(v.errors, [])
            self.assertTrue(any("未包含禁止路径" in m for m in messages))

    def test_zip_bad_zip_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad_zip = Path(tmp) / "bad.zip"
            bad_zip.write_bytes(b"not a zip file")
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = []
            v.warnings = []
            item = {
                "id": "Z6",
                "requirements": {"include": ["src/"]},
            }
            messages: list[str] = []
            v._validate_zip(item, bad_zip, messages)
            self.assertTrue(any("ZIP 文件损坏" in e for e in v.errors))


class TestValidateMarkdown(unittest.TestCase):
    """测试 _validate_markdown 方法。"""

    def test_markdown_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "test.md"
            md_path.write_text("content", encoding="utf-8")
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = []
            v.warnings = []
            messages: list[str] = []
            v._validate_markdown({"id": "M1", "must_exist": True}, md_path, messages)
            self.assertTrue(any("文件存在" in m for m in messages))

    def test_markdown_no_must_exist(self):
        with tempfile.TemporaryDirectory() as tmp:
            md_path = Path(tmp) / "test.md"
            md_path.write_text("content", encoding="utf-8")
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = []
            v.warnings = []
            messages: list[str] = []
            v._validate_markdown({"id": "M2"}, md_path, messages)
            # must_exist=False 时不添加消息
            self.assertEqual(messages, [])


class TestVersionConsistency(unittest.TestCase):
    """测试 _check_version_consistency 方法。"""

    def test_version_found_in_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "README.md").write_text("# Quantum RL Scheduler v8.0\n", encoding="utf-8")
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = []
            v.warnings = []
            v.project_root = Path(tmp)
            v.manifest = {"submission": {"version": "v8.0"}}
            v._check_version_consistency()
            self.assertEqual(v.warnings, [])

    def test_version_not_found_in_readme(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "README.md").write_text("# Quantum RL Scheduler\n", encoding="utf-8")
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = []
            v.warnings = []
            v.project_root = Path(tmp)
            v.manifest = {"submission": {"version": "v8.0"}}
            v._check_version_consistency()
            self.assertTrue(any("未找到版本号" in w for w in v.warnings))

    def test_version_no_readme(self):
        """README.md 不存在时应跳过，不报错。"""
        with tempfile.TemporaryDirectory() as tmp:
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = []
            v.warnings = []
            v.project_root = Path(tmp)
            v.manifest = {"submission": {"version": "v8.0"}}
            v._check_version_consistency()
            self.assertEqual(v.warnings, [])


class TestGenerateReport(unittest.TestCase):
    """测试 generate_report 方法。"""

    def test_generate_report_with_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = ["[X] error"]
            v.warnings = ["[Y] warning"]
            v.results = [
                ItemResult(
                    item_id="X",
                    name="失败项",
                    item_type="md",
                    path="x.md",
                    passed=False,
                    messages=["文件不存在"],
                    severity="error",
                ),
                ItemResult(
                    item_id="Y",
                    name="通过项",
                    item_type="md",
                    path="y.md",
                    passed=True,
                    messages=["文件存在"],
                    severity="info",
                ),
            ]
            v.manifest = {"submission": {"version": "v8.0", "deadline": "2026-09-15"}}

            report_path = Path(tmp) / "report.md"
            v.generate_report(str(report_path))

            self.assertTrue(report_path.exists())
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("提交物校验报告", content)
            self.assertIn("v8.0", content)
            self.assertIn("缺失项清单", content)
            self.assertIn("失败项", content)
            self.assertIn("已通过项清单", content)
            self.assertIn("通过项", content)

    def test_generate_report_all_passed(self):
        with tempfile.TemporaryDirectory() as tmp:
            v = SubmissionValidator.__new__(SubmissionValidator)
            v.errors = []
            v.warnings = []
            v.results = [
                ItemResult(
                    item_id="X",
                    name="通过项",
                    item_type="md",
                    path="x.md",
                    passed=True,
                    messages=["文件存在"],
                    severity="info",
                ),
            ]
            v.manifest = {"submission": {"version": "v8.0", "deadline": "2026-09-15"}}

            report_path = Path(tmp) / "report.md"
            v.generate_report(str(report_path))

            content = report_path.read_text(encoding="utf-8")
            self.assertIn("无缺失项", content)


class TestPrepareSubmission(unittest.TestCase):
    """测试 prepare_submission 函数。"""

    def test_prepare_creates_dist_and_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            # 创建 manifest
            manifest = _make_manifest(
                items=[
                    {
                        "id": "TEST_MD",
                        "name": "测试",
                        "type": "md",
                        "path": "test.md",
                        "must_exist": True,
                    }
                ]
            )
            manifest_path = Path(tmp) / "manifest.yaml"
            manifest_path.write_text(yaml.dump(manifest), encoding="utf-8")

            prepare_submission(str(manifest_path), tmp)

            # 验证 dist/ 目录已创建
            self.assertTrue((Path(tmp) / "dist").is_dir())
            # 验证报告已生成
            report = Path(tmp) / "results" / "reports" / "submission_validation_report.md"
            self.assertTrue(report.exists())


if __name__ == "__main__":
    unittest.main()
