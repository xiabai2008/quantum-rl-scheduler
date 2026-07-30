"""tests/visualization 共享夹具 — 状态隔离与异步客户端。

从原 tests/test_visualization.py 提取的公共夹具：
- reset_state: 快照并恢复全局 system_status / task_queue / 连接管理器
- default_viz_auth: 默认配置 VIZ_API_KEY
- async_client: 基于 ASGITransport 的 httpx 异步客户端
"""

import copy
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization import state as vis_state
from src.visualization.app import app
from src.visualization.security import rate_limiter

# 注意：src/visualization/__init__.py 执行了 `from src.visualization.app import app`，
# 这会覆盖 src.visualization 包的 app 属性为 FastAPI 实例，从而遮蔽 app 子模块。
# 因此 `import src.visualization.app as app_module` 会把 app_module 绑定为 FastAPI 实例，
# 而非模块对象。这里通过 sys.modules 直接获取真正的子模块对象，绕过属性遮蔽问题。
app_module = sys.modules["src.visualization.app"]

# 测试统一认证密钥：默认开启认证，所有写请求经客户端默认头携带
TEST_VIZ_KEY = "test-viz-key"


@pytest.fixture(autouse=True)
def reset_state():
    """快照并恢复全局 system_status / task_queue / 连接管理器，保证测试间隔离。

    同时重置速率限制器状态（Issue #517），避免 POST 端点测试因共享 IP
    （testserver/127.0.0.1）累积请求而触发 429 限流，影响后续测试。
    """
    saved_status = copy.deepcopy(app_module.system_status)
    saved_queue = copy.deepcopy(app_module.task_queue)
    saved_connections = list(app_module.manager.active_connections)
    saved_strategy = app_module.system_status.get("current_strategy")
    # 重置速率限制器，确保每个测试从干净状态开始（Issue #517）
    rate_limiter.reset()
    yield
    app_module.system_status.clear()
    app_module.system_status.update(copy.deepcopy(saved_status))
    app_module.task_queue.clear()
    app_module.task_queue.extend(copy.deepcopy(saved_queue))
    app_module.manager.active_connections = list(saved_connections)
    # current_strategy 可能被 POST /api/strategy 修改，强制还原
    app_module.system_status["current_strategy"] = saved_strategy
    # 测试结束后再次重置速率限制器，避免失败用例残留状态影响后续测试
    rate_limiter.reset()


@pytest.fixture(autouse=True)
def default_viz_auth(monkeypatch):
    """测试默认配置 VIZ_API_KEY，使写操作需经认证（符合修复后的安全模型）。

    同时允许空 Origin 的 WebSocket 连接（TestClient 不发送 Origin 头），
    以适配 Issue #736 对 is_origin_allowed 的安全收紧。
    """
    monkeypatch.setenv("VIZ_API_KEY", TEST_VIZ_KEY)
    monkeypatch.setenv("VIZ_WS_ALLOW_EMPTY_ORIGIN", "1")


@pytest_asyncio.fixture
async def async_client():
    """提供基于 ASGITransport 的 httpx 异步客户端。

    ASGITransport 不会触发 FastAPI lifespan，因此后台任务 simulate_scheduler
    不会运行，保证测试期间全局状态不被后台任务修改。默认注入 X-API-Key 头，
    使写请求通过认证；如需测试"无密钥/错误密钥"场景，请显式 delenv 或传 header。
    """
    transport = ASGITransport(app=app)
    async with AsyncClient(
        transport=transport,
        base_url="http://testserver",
        headers={"X-API-Key": TEST_VIZ_KEY},
    ) as client:
        yield client
