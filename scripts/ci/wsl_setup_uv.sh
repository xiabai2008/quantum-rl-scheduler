#!/bin/bash
set -e
export PATH="$HOME/.local/bin:$PATH"
# 安装 uv（若未装）
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
uv --version
# 装 Python 3.12（与 Windows 一致，SB3/torch 兼容）
uv python install 3.12 2>&1 | tail -2
uv python list 2>/dev/null | grep 3.12 | head -3
