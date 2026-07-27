"""版本号一致性校验（Issue #390）。

确保包版本（pyproject.toml）与提交物清单版本
（config/submission_manifest.yaml 的 submission.version）保持一致，
且均为合法的语义化版本号（semver）。

注意：
- src/visualization/app.py 中 FastAPI 的 ``version="1.0.0"`` 是 **API 版本**
  语义，与包版本无关，不在此校验范围内。
- manifest 中的版本号可能为 ``v8.0`` 这类带 ``v`` 前缀的形式，比较前需去掉
  前缀再与 pyproject 的纯 semver 对齐。
"""

import os
import tomllib

import pytest
import yaml

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_pyproject_version() -> str:
    pyproject_path = os.path.join(_REPO_ROOT, "pyproject.toml")
    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)
    return data["project"]["version"]


def _load_manifest_version() -> str:
    manifest_path = os.path.join(_REPO_ROOT, "config", "submission_manifest.yaml")
    with open(manifest_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["submission"]["version"]


def _strip_v_prefix(version: str) -> str:
    """去掉可选的 ``v`` 前缀（如 ``v8.0.0`` -> ``8.0.0``）。"""
    return version[1:] if version.startswith("v") else version


def test_pyproject_matches_manifest_version() -> None:
    """pyproject.toml 的 version 与 manifest submission.version 必须相等。"""
    pyproject_version = _load_pyproject_version()
    manifest_version = _strip_v_prefix(_load_manifest_version())

    assert pyproject_version == manifest_version, (
        f"版本号不一致: pyproject={pyproject_version!r} "
        f"manifest={manifest_version!r}"
    )


def test_versions_are_semver() -> None:
    """版本号必须是合法的语义化版本（MAJOR.MINOR.PATCH）。"""
    import re

    semver_re = re.compile(r"^\d+\.\d+\.\d+$")
    pyproject_version = _load_pyproject_version()
    manifest_version = _strip_v_prefix(_load_manifest_version())

    assert semver_re.match(pyproject_version), (
        f"pyproject version 非 semver: {pyproject_version!r}"
    )
    assert semver_re.match(manifest_version), (
        f"manifest version 非 semver: {manifest_version!r}"
    )
