"""
Web可视化监控界面
Web Visualization Monitoring Dashboard

基于 FastAPI + 原生 HTML/JS 的量子RL调度系统监控界面
支持 WebSocket 实时推送、手动任务提交、调度策略切换等功能

模块拆分说明（v8）：
    本模块保留应用核心：辅助函数（懒加载模型/真机客户端/模板）、
    FastAPI 应用实例、生命周期与启动入口。HTTP 路由、WebSocket 端点、
    后台仿真循环、数据模型、连接管理器、回退 HTML 模板分别拆分至：
        - state.py              共享全局状态 + 线程安全访问器（Issue #179）
        - routes.py            REST API 路由（APIRouter）
        - websocket_handler.py WebSocket /ws 端点
        - simulator.py         后台仿真循环 simulate_scheduler
        - models.py            Pydantic 模型（TaskSubmit / SystemStatusUpdate）
        - connection.py        ConnectionManager 连接管理器
        - fallback_template.py 内置回退 HTML 模板

循环依赖说明（Issue #179）：
    app.py 在底部导入 routes.py / simulator.py / websocket_handler.py，
    这些模块通过 ``import src.visualization.app as _app`` 反向访问本模块
    的辅助函数（``_get_ppo_model`` 等）。这是 Python 中常见的延迟引用模式：
    app.py 先完成全部模块级定义，再在底部导入子模块，此时 ``_app.X``
    的属性访问在函数体内执行，app 模块已完成加载。
    共享状态（``system_status`` / ``task_queue`` 等）已提取到 ``state.py``，
    子模块直接从 ``state.py`` 导入状态，不再通过 ``_app`` 中转——
    仅辅助函数仍通过 ``_app`` 访问（测试通过 monkeypatch 替换）。

运行方式:
    python src/visualization/app.py
    或
    python -m src.visualization.app
"""

import asyncio
import json
import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from loguru import logger

# 确保项目根目录在 Python 路径中
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 从拆分模块导入数据模型、连接管理器与回退 HTML 模板（无循环依赖）。
# 以下符号在本模块内仅作再导出用途（供 `from src.visualization.app import X` 使用），
# 通过 __all__ 声明以避免 ruff F401 误报。
from src.visualization.connection import ConnectionManager
from src.visualization.fallback_template import HTML_TEMPLATE
from src.visualization.models import SystemStatusUpdate, TaskSubmit

# 共享全局状态从 state.py 再导出（Issue #179）。
# state.py 是共享可变状态的唯一定义点；此处再导出为模块级属性，
# 确保 `from src.visualization.app import system_status` 等既有导入路径
# 与测试 monkeypatch（`app_module.system_status.clear()` 等）仍然可用。
# 新代码请直接 `from src.visualization.state import ...` 获取状态或访问器。
from src.visualization.state import (  # noqa: F401 — 再导出，供 __all__ 声明
    STRATEGY_OPTIONS,
    _battle_state,
    _decision_log,
    _resource_history,
    manager,
    system_status,
    task_queue,
)

# 向后兼容再导出清单：测试与外部代码沿用 `from src.visualization.app import ...`。
__all__ = [
    "HTML_TEMPLATE",
    "STRATEGY_OPTIONS",
    "ConnectionManager",
    "SystemStatusUpdate",
    "TaskSubmit",
    "app",
    "lifespan",
    "manager",
    "simulate_scheduler",
    "start_web_server",
    "system_status",
    "task_queue",
    "verify_api_key",
    "websocket_endpoint",
]


# ============================================================
# 懒加载状态（保留在 app.py — 测试通过 monkeypatch 替换）
# ============================================================
# 以下懒加载状态不迁移到 state.py，因为测试通过
# `monkeypatch.setattr(app_module, "_ppo_model", None)` 等方式替换引用，
# routes.py / simulator.py 通过 `_app._ppo_model` 访问，必须保留在 app 模块。

# 懒加载 PPO 模型和环境
_ppo_model = None
_ppo_env = None

# 懒加载真机 cqlib 客户端（仅在配置了 TIANYAN_API_KEY 时创建）
_real_cqlib_client = None
_real_cqlib_checked = False

# 全局配额追踪器实例（懒加载）
_quota_tracker_instance: Any = None

# 前端目录路径
_FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
FRONTEND_HTML_PATH = os.path.join(_FRONTEND_DIR, "index.html")
# Vue3 构建产物目录（npm run build 后生成）
FRONTEND_DIST_PATH = os.path.join(_FRONTEND_DIR, "dist")

# 缓存前端 HTML 内容
_VUE3_HTML_TEMPLATE = None


# ============================================================
# 辅助函数：模板加载 / PPO 模型 / 真机客户端 / 配额追踪
# ============================================================


def _load_vue3_template() -> str:
    """加载 Vue3 前端 HTML 模板。

    优先级：
    1. dist/index.html（npm run build 产物，生产模式）
    2. frontend/index.html（开发模式源文件）
    3. HTML_TEMPLATE（内置回退模板）
    """
    global _VUE3_HTML_TEMPLATE
    if _VUE3_HTML_TEMPLATE is None:
        # 优先使用构建产物
        dist_html = os.path.join(FRONTEND_DIST_PATH, "index.html")
        if os.path.exists(dist_html):
            with open(dist_html, encoding="utf-8") as f:
                _VUE3_HTML_TEMPLATE = f.read()
            logger.info("[Web] 使用 Vue3 构建产物 (dist/index.html)")
        elif os.path.exists(FRONTEND_HTML_PATH):
            with open(FRONTEND_HTML_PATH, encoding="utf-8") as f:
                _VUE3_HTML_TEMPLATE = f.read()
            logger.info("[Web] 使用 Vue3 源文件 (frontend/index.html)")
        else:
            _VUE3_HTML_TEMPLATE = HTML_TEMPLATE  # 回退到内置 HTML
            logger.info("[Web] 使用内置回退 HTML 模板")
    return _VUE3_HTML_TEMPLATE


def _get_ppo_model() -> Any:
    """加载 PPO 模型（懒加载，避免启动时阻塞）"""
    global _ppo_model, _ppo_env
    if _ppo_model is None:
        try:
            from stable_baselines3 import PPO

            from src.scheduler.env import QuantumSchedulingEnv

            _ppo_env = QuantumSchedulingEnv(max_qubits=287, seed=42)
            # 优先使用 deliverable_models/ 下的权威模型（入库模型，所有环境都有）
            deliverable_dir = os.path.join(_PROJECT_ROOT, "deliverable_models")
            model_path = os.path.join(deliverable_dir, "ppo_best_model_14dim.zip")

            if not os.path.exists(model_path):
                # 回退：自动发现 deliverable_models/ 或 models/ 下的 PPO 模型
                for search_dir in [deliverable_dir, os.path.join(_PROJECT_ROOT, "models")]:
                    if os.path.isdir(search_dir):
                        for root, _dirs, files in os.walk(search_dir):
                            for f in files:
                                if f.endswith(".zip") and "ppo" in f.lower() and "14dim" in f:
                                    model_path = os.path.join(root, f)
                                    break
                            if os.path.exists(model_path):
                                break
                    if os.path.exists(model_path):
                        break

            if os.path.exists(model_path):
                _ppo_model = PPO.load(model_path, env=_ppo_env)
                logger.info(f"[PPO] 模型加载成功: {model_path}")
            else:
                logger.warning(f"[PPO] 模型文件不存在: {model_path}，尝试使用 DQN")
        except (OSError, ValueError, RuntimeError) as e:
            # 文件 I/O 错误 / 模型格式错误 / 运行时错误
            logger.error(f"[PPO] 模型加载失败: {e}")
            _ppo_model = None
    return _ppo_model


def _get_real_cqlib_client() -> Any:
    """懒加载天衍云 cqlib 客户端。

    从 .env 读取 TIANYAN_API_KEY，无 Key 时返回 None（降级为纯仿真展示）。
    客户端创建失败也返回 None，保证 Web 界面不会因真机不可达而崩溃。
    """
    global _real_cqlib_client, _real_cqlib_checked
    if _real_cqlib_checked:
        return _real_cqlib_client
    _real_cqlib_checked = True
    try:
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("TIANYAN_API_KEY", "")
        if not api_key:
            logger.info("[Web] 未配置 TIANYAN_API_KEY，真机状态轮询已禁用")
            return None
        from src.api.tianyan_cqlib import CqlibTianyanClient

        _real_cqlib_client = CqlibTianyanClient(
            login_key=api_key,
            machine_name="tianyan_s",
            auto_retry_machine=True,
        )
        logger.info("[Web] 真机 cqlib 客户端已就绪: tianyan_s")
    except Exception as e:
        # 防御性错误边界：客户端创建可能因依赖缺失/网络/认证/配置等多种原因失败，统一降级为离线
        logger.warning(f"[Web] 真机客户端创建失败 ({e})，真机状态降级为离线")
        _real_cqlib_client = None
    return _real_cqlib_client


def _get_real_machines_status() -> list[dict]:
    """查询天衍云真实量子计算机列表及状态。

    调用 ``CqlibTianyanClient.list_backends()``（底层
    ``platform.query_quantum_computer_list()``），返回包含
    running/calibrating/maintenance 等真实状态的机器列表。

    Returns:
        机器字典列表 [{id, type, status, name}]；查询失败或无客户端时返回 []
    """
    client = _get_real_cqlib_client()
    if client is None:
        return []
    try:
        return client.list_backends()  # type: ignore[no-any-return]
    except Exception as e:  # 防御性错误边界：cqlib 任意异常均需优雅降级为空列表
        logger.error(f"[Web] 查询真机状态失败: {e}")
        return []


def _load_real_submissions() -> list[dict]:
    """从 results/real_times.json 加载最近的真机提交记录。

    训练回调 ``RealMachineCallback`` 会把真机提交记录写入该文件。
    Web 界面读取后展示真实提交历史（步数/机器/耗时/task_id）。

    Returns:
        提交记录列表（最多保留最近 50 条）；文件不存在时返回 []
    """
    path = os.path.join(_PROJECT_ROOT, "results", "real_times.json")
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            records = json.load(f)
        if isinstance(records, list):
            # 保留最近 50 条，倒序展示
            return records[-50:][::-1]
        return []
    except (json.JSONDecodeError, OSError) as e:
        # JSON 解析错误 / 文件 I/O 错误
        logger.error(f"[Web] 加载真机提交记录失败: {e}")
        return []


def _get_quota_tracker() -> Any:
    """懒加载全局 QuotaTracker 实例。

    Returns:
        QuotaTracker 实例（初始化失败时返回 None）
    """
    global _quota_tracker_instance
    if _quota_tracker_instance is None:
        try:
            from src.api.quota_tracker import QuotaTracker

            _quota_tracker_instance = QuotaTracker(
                config_path=str(_PROJECT_ROOT / "config" / "quota.yaml"),
                state_path=str(_PROJECT_ROOT / "logs" / "quota_state.json"),
            )
        except Exception as e:
            logger.debug(f"[Web] QuotaTracker 初始化失败: {e}")
            return None
    return _quota_tracker_instance


# ============================================================
# 应用生命周期与 FastAPI 实例
# ============================================================


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """应用生命周期：启动时开启后台模拟任务"""
    task = asyncio.create_task(simulate_scheduler())
    yield
    task.cancel()


app = FastAPI(title="量子RL调度系统监控界面", version="1.0.0", lifespan=lifespan)

# 挂载 Vue3 构建产物的静态资源目录（dist/assets/）
_dist_assets = os.path.join(FRONTEND_DIST_PATH, "assets")
if os.path.isdir(_dist_assets):
    from fastapi.staticfiles import StaticFiles

    app.mount("/assets", StaticFiles(directory=_dist_assets), name="assets")
    logger.info(f"[Web] 静态资源目录已挂载: {_dist_assets}")


# ============================================================
# 注册拆分模块：REST 路由 / WebSocket 端点 / 后台仿真循环
# 这些模块通过 `import src.visualization.app as _app` 反向访问本模块
# 的全局状态与辅助函数，故须在本模块完成上述定义后再导入。
# ============================================================

from src.visualization.routes import router, verify_api_key
from src.visualization.simulator import simulate_scheduler
from src.visualization.websocket_handler import websocket_endpoint

app.include_router(router)
app.websocket("/ws")(websocket_endpoint)


# ============================================================
# 服务器启动入口
# ============================================================


def start_web_server(
    host: str = "0.0.0.0",  # nosec B104: demo binding
    port: int = 8000,
) -> None:
    """启动 Web 服务器"""
    # 初始化统一日志配置（Issue #193）
    from src.config.settings import install_intercept_handler
    from src.utils.helpers import setup_logging

    setup_logging()
    install_intercept_handler()

    import uvicorn

    logger.info("========================================")
    logger.info("  量子RL调度系统 - 监控面板")
    logger.info(f"  访问地址: http://{host}:{port}")
    logger.info("========================================")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_web_server()
