#!/usr/bin/env python
"""本地测试运行器（Windows torch 崩溃规避）。

背景：在 Windows + 该 conda 环境下，直接 `python -m pytest` 会在 pytest 启动其导入
机制后再加载 torch 的 C 扩展，触发非确定性的 access violation（段错误，exit 139）。
经排查，`python -c "import torch"` 单独导入 20/20 稳定，但 pytest 进程内延迟加载
torch 必崩。根因是 torch 的原生 DLL 需要在 pytest 改动导入系统之前先完成加载。

解法：在调用 pytest 之前先 import torch，让其原生 DLL 在纯净解释器状态下加载完成，
之后 pytest 正常运行即可稳定。

用法：
    python run_tests.py                 # 跑全部测试
    python run_tests.py tests/test_marl.py -v
    python run_tests.py -k marl -q
    （所有参数原样透传给 pytest）
"""

import sys

# 关键：必须在导入 pytest 之前先加载 torch，预热其原生 DLL。
import torch  # noqa: F401,E402

import pytest  # noqa: E402


def main() -> int:
    args = sys.argv[1:]
    if not args:
        args = ["tests/"]
    return int(pytest.main(args))


if __name__ == "__main__":
    raise SystemExit(main())
