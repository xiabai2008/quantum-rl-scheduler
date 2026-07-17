"""
Web可视化监控界面
Web Visualization Monitoring Dashboard

基于 FastAPI + 原生 HTML/JS 的量子RL调度系统监控界面
支持 WebSocket 实时推送、手动任务提交、调度策略切换等功能

模块拆分说明（v8）：
    本模块保留应用核心：全局状态、辅助函数（懒加载模型/真机客户端/模板）、
    FastAPI 应用实例、生命周期与启动入口。HTTP 路由、WebSocket 端点、
    后台仿真循环、数据模型、连接管理器、回退 HTML 模板分别拆分至：
        - routes.py            REST API 路由（APIRouter）
        - websocket_handler.py WebSocket /ws 端点
        - simulator.py         后台仿真循环 simulate_scheduler
        - models.py            Pydantic 模型（TaskSubmit / SystemStatusUpdate）
        - connection.py        ConnectionManager 连接管理器
        - fallback_template.py 内置回退 HTML 模板

运行方式:
    python src/visualization/app.py
    或
    python -m src.visualization.app
"""

import asyncio
import json
import os
import sys
import uuid
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime
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

# 向后兼容再导出清单：测试与外部代码沿用 `from src.visualization.app import ...`。
__all__ = [
    "HTML_TEMPLATE",
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
# 内存存储（生产环境应替换为 Redis 等外部存储）
# ============================================================

# 当前系统状态
system_status: dict = {
    "qubit_utilization": 0.65,  # 量子比特利用率 (0~1)
    "queue_length": 5,  # 任务队列长度
    "average_wait_time": 12.3,  # 平均等待时间(秒)
    "completed_tasks": 42,  # 已完成任务数
    "current_step": 1024,  # 当前调度步数
    "current_strategy": "PPO-Balanced",  # 当前调度策略
    "strategy_options": [  # 可选策略列表
        "DQN-Reward",
        "DQN-Latency",
        "PPO-Balanced",
        "QAOA-Hybrid",
        "FCFS",
    ],
    "real_machines": [],  # 真机列表 [{name, status, type, id}]
    "real_submissions": [],  # 真机提交记录 [{step, task_id, machine, latency_s, status}]
    "last_update": datetime.now().isoformat(),
}

# 任务队列
task_queue: list[dict] = [
    {
        "task_id": "QTASK-" + uuid.uuid4().hex[:6],
        "user_id": "user_001",
        "task_type": "quantum",
        "status": "pending",
        "priority": 4,
        "qubit_count": 12,
        "circuit_depth": 150,
        "estimated_time": 45.0,
        "arrival_time": datetime.now().isoformat(),
    },
    {
        "task_id": "QTASK-" + uuid.uuid4().hex[:6],
        "user_id": "user_002",
        "task_type": "hybrid",
        "status": "pending",
        "priority": 3,
        "qubit_count": 8,
        "circuit_depth": 80,
        "estimated_time": 30.0,
        "arrival_time": datetime.now().isoformat(),
    },
    {
        "task_id": "QTASK-" + uuid.uuid4().hex[:6],
        "user_id": "user_001",
        "task_type": "classical",
        "status": "pending",
        "priority": 2,
        "qubit_count": 0,
        "circuit_depth": 0,
        "estimated_time": 20.0,
        "arrival_time": datetime.now().isoformat(),
    },
]

# WebSocket 连接管理器实例（全局单例）
manager = ConnectionManager()

# 资源利用率历史数据（内存缓存，最多保留 100 个数据点）
_resource_history: list[dict[str, Any]] = []

# 决策日志（内存缓存，最多保留 200 条）
_decision_log: list[dict[str, Any]] = []

# 懒加载 PPO 模型和环境
_ppo_model = None
_ppo_env = None

# 懒加载真机 cqlib 客户端（仅在配置了 TIANYAN_API_KEY 时创建）
_real_cqlib_client = None
_real_cqlib_checked = False

# 全局配额追踪器实例（懒加载）
_quota_tracker_instance: Any = None

# 前端 HTML 文件路径
FRONTEND_HTML_PATH = os.path.join(os.path.dirname(__file__), "frontend", "index.html")

# 缓存前端 HTML 内容
_VUE3_HTML_TEMPLATE = None


# ============================================================
# 辅助函数：模板加载 / PPO 模型 / 真机客户端 / 配额追踪
# ============================================================


def _load_vue3_template() -> str:
    """加载 Vue3 前端 HTML 模板"""
    global _VUE3_HTML_TEMPLATE
    if _VUE3_HTML_TEMPLATE is None:
        if os.path.exists(FRONTEND_HTML_PATH):
            with open(FRONTEND_HTML_PATH, encoding="utf-8") as f:
                _VUE3_HTML_TEMPLATE = f.read()
        else:
            _VUE3_HTML_TEMPLATE = HTML_TEMPLATE  # 回退到内置 HTML
    return _VUE3_HTML_TEMPLATE


def _get_ppo_model() -> Any:
    """加载 PPO 模型（懒加载，避免启动时阻塞）"""
    global _ppo_model, _ppo_env
    if _ppo_model is None:
        try:
            from stable_baselines3 import PPO

            from src.scheduler.env import QuantumSchedulingEnv

            _ppo_env = QuantumSchedulingEnv(max_qubits=20, seed=42)
            model_path = os.path.join(_PROJECT_ROOT, "models", "ppo_seed_42_v4", "best_model.zip")

            if not os.path.exists(model_path):
                # 自动发现：在 models/ 下找任意 ppo 开头的目录中的 best_model.zip
                models_dir = os.path.join(_PROJECT_ROOT, "models")
                for root, _dirs, files in os.walk(models_dir):
                    if "ppo" in os.path.basename(root).lower():
                        for f in files:
                            if f.endswith(".zip"):
                                model_path = os.path.join(root, f)
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
    import uvicorn

    logger.info("========================================")
    logger.info("  量子RL调度系统 - 监控面板")
    logger.info(f"  访问地址: http://{host}:{port}")
    logger.info("========================================")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_web_server()
