"""pytest 配置：注册自定义标记 + CI 环境下跳过外部 SDK 依赖测试。"""

import importlib
import os

import pytest


def pytest_configure(config):
    """注册 benchmark 标记，避免 --strict-markers 报错。"""
    config.addinivalue_line("markers", "benchmark: marks performance benchmark tests")


def pytest_collection_modifyitems(config, items):
    """在 CI 环境下自动跳过依赖外部 SDK 或脆弱的测试。"""
    in_ci = os.environ.get("CI", "") == "true" or os.environ.get("GITHUB_ACTIONS", "") == "true"

    if not in_ci:
        return

    # cqlib 是专有 SDK，CI 环境无法安装
    cqlib_skip = "cqlib"
    try:
        importlib.import_module("cqlib")
        cqlib_skip = None
    except ImportError:
        pass

    for item in items:
        if cqlib_skip and "cqlib" in item.nodeid.lower():
            item.add_marker(pytest.mark.skip(reason="cqlib SDK not available in CI"))
