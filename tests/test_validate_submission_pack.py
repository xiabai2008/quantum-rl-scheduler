"""validate_submission --pack 打包逻辑测试

覆盖 Issue: --pack 会把整仓（含 .git/ 与 dist/ 输出文件本身）打入提交压缩包，
导致体积失控与 Release CI 打包超时。
"""

from pathlib import Path
from zipfile import ZipFile

import yaml

from scripts.ci.validate_submission import _build_code_archive, package_submission

MINIMAL_MANIFEST = {
    "submission": {"deadline": "2026-09-15", "version": "9.1.0"},
    "items": [
        {
            "id": "CODE_REPO",
            "name": "代码仓库",
            "type": "git_tag",
            "path": ".",
            "requirements": {"tag": "v9.1-submission"},
        },
        {
            "id": "CODE_ARCHIVE",
            "name": "代码压缩包",
            "type": "zip",
            "path": "dist/quantum-rl-scheduler-v9.1.zip",
            "requirements": {
                "max_size_mb": 100,
                "exclude": [".git/", "__pycache__/"],
                "include": ["src/", "README.md"],
            },
        },
    ],
}


def _write_manifest(project: Path) -> Path:
    manifest = project / "submission_manifest.yaml"
    manifest.write_text(yaml.safe_dump(MINIMAL_MANIFEST, allow_unicode=True), encoding="utf-8")
    return manifest


def _make_project(tmp_path: Path) -> Path:
    project = tmp_path / "proj"
    project.mkdir(exist_ok=True)
    (project / "src").mkdir(exist_ok=True)
    (project / "src" / "scheduler.py").write_text("# code\n", encoding="utf-8")
    (project / "README.md").write_text("# readme\n", encoding="utf-8")
    (project / "results").mkdir(exist_ok=True)
    (project / "results" / "big.bin").write_bytes(b"x" * 1024)
    (project / ".git").mkdir(exist_ok=True)
    (project / ".git" / "objects").mkdir(exist_ok=True)
    (project / ".git" / "objects" / "big.bin").write_bytes(b"g" * 1024)
    return project


def test_build_code_archive_honors_include_and_exclude(tmp_path: Path) -> None:
    project = _make_project(tmp_path)
    manifest = _write_manifest(project)

    archive = _build_code_archive(str(manifest), str(project))

    assert archive is not None
    assert archive.exists()
    with ZipFile(archive) as zipf:
        names = set(zipf.namelist())
    assert "src/scheduler.py" in names
    assert "README.md" in names
    assert "results/big.bin" not in names
    assert ".git/objects/big.bin" not in names


def test_package_submission_does_not_bundle_git_repo(tmp_path: Path, monkeypatch: object) -> None:
    project = _make_project(tmp_path)
    manifest = _write_manifest(project)
    # CODE_REPO git tag 校验需要真实 git 仓库，这里用跳过项避免依赖 git
    monkeypatch.setattr(
        "scripts.ci.validate_submission.sys.exit",
        lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )

    package_submission(str(manifest), str(project), skip_items=["CODE_REPO"])

    outputs = list(project.glob("dist/submission_*.zip"))
    assert len(outputs) == 1
    with ZipFile(outputs[0]) as zipf:
        names = set(zipf.namelist())
    assert ".git/objects/big.bin" not in names
    assert "results/big.bin" not in names
    assert "dist/quantum-rl-scheduler-v9.1.zip" in names
    # 输出压缩包自身不得被递归打入自身
    assert outputs[0].name not in names
    # 归档内应包含受 include/exclude 控制的代码文件
    with ZipFile(project / "dist" / "quantum-rl-scheduler-v9.1.zip") as inner:
        inner_names = set(inner.namelist())
    assert "src/scheduler.py" in inner_names
    assert "README.md" in inner_names
    assert ".git/objects/big.bin" not in inner_names
    assert "results/big.bin" not in inner_names


def test_package_submission_rejects_when_validation_fails(
    tmp_path: Path, monkeypatch: object
) -> None:
    project = _make_project(tmp_path)
    manifest = _write_manifest(project)
    # CODE_REPO git tag 校验需要真实 git 仓库，用跳过项避免依赖 git；
    # 保留 CODE_ARCHIVE 校验：删除 README.md 后归档缺少必需路径 → 校验失败
    (project / "README.md").unlink()
    monkeypatch.setattr(
        "scripts.ci.validate_submission.sys.exit",
        lambda code: (_ for _ in ()).throw(SystemExit(code)),
    )

    try:
        package_submission(str(manifest), str(project), skip_items=["CODE_REPO"])
    except SystemExit as exc:
        assert exc.code == 1
    else:
        raise AssertionError("校验失败时应退出码 1")
