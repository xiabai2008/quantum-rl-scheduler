"""pytest 配置：注册自定义标记 + cqlib 回放测试 fixtures 感知跳过逻辑。

Issue #175 改进：将"CI 下跳过所有 cqlib 测试"改为"有 fixtures 则运行回放测试，
无则 skip 并告警"。回放测试标记为 ``cqlib_replay``，无需 cqlib SDK 即可运行。
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from src.api.cqlib_recorder import CqlibReplayClient

# cqlib 回放 fixtures 目录
_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "cqlib_responses"


def pytest_configure(config: Any) -> None:
    """注册自定义标记，避免 --strict-markers 报错。"""
    config.addinivalue_line("markers", "benchmark: marks performance benchmark tests")
    config.addinivalue_line(
        "markers",
        "cqlib_replay: marks cqlib replay tests (run from fixtures, no SDK needed)",
    )


def _has_cqlib_sdk() -> bool:
    """检测 cqlib SDK 是否可导入。"""
    try:
        importlib.import_module("cqlib")
        return True
    except ImportError:
        return False


def _has_replay_fixtures() -> bool:
    """检测 cqlib 回放 fixtures 是否存在。"""
    return _FIXTURES_DIR.is_dir() and any(_FIXTURES_DIR.glob("*.json"))


def pytest_collection_modifyitems(config: Any, items: list[Any]) -> None:
    """cqlib 测试跳过逻辑：有 fixtures 则运行回放测试，无则 skip 并告警。

    跳过策略：
        1. cqlib SDK 可用 → 运行所有 cqlib 测试（含回放与真机测试）
        2. cqlib SDK 不可用 + 有 fixtures → 运行 cqlib_replay 标记的回放测试，
           跳过依赖真机 SDK 的测试
        3. cqlib SDK 不可用 + 无 fixtures → 跳过所有 cqlib 测试并发出告警
    """
    has_cqlib = _has_cqlib_sdk()
    has_fixtures = _has_replay_fixtures()

    # 无 SDK 且无 fixtures 时告警一次
    if not has_cqlib and not has_fixtures:
        warnings.warn(
            "cqlib SDK not available and replay fixtures not found at "
            f"{_FIXTURES_DIR}, all cqlib tests will be skipped",
            stacklevel=2,
        )

    for item in items:
        if "cqlib" not in item.nodeid.lower():
            continue

        if has_cqlib:
            # cqlib SDK 可用，运行所有 cqlib 测试
            continue

        # cqlib SDK 不可用
        is_replay = item.get_closest_marker("cqlib_replay") is not None

        if is_replay and has_fixtures:
            # 回放测试 + 有 fixtures → 运行
            continue

        if is_replay:
            # 回放测试但无 fixtures → skip
            item.add_marker(pytest.mark.skip(reason="cqlib replay fixtures not found"))
        else:
            # 真机 SDK 测试，无 SDK → skip
            item.add_marker(pytest.mark.skip(reason="cqlib SDK not available"))


@pytest.fixture
def cqlib_replay_client() -> CqlibReplayClient:
    """返回 CqlibReplayClient 实例，使用默认 fixtures 目录。

    Returns:
        已加载全部 fixtures 的 CqlibReplayClient 实例
    """
    from src.api.cqlib_recorder import CqlibReplayClient as _ReplayClient

    return _ReplayClient(str(_FIXTURES_DIR))
