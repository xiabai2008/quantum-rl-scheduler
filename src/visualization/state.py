"""
可视化层共享状态与线程安全访问器（Issue #179）

本模块是可视化层全局可变状态的**唯一定义点**，解决 app.py 与 routes.py /
simulator.py / websocket_handler.py 之间的循环依赖问题。

架构说明：
    在 Issue #179 之前，``system_status`` / ``task_queue`` / ``manager`` 等
    全局可变状态定义在 ``app.py`` 中，routes.py / simulator.py /
    websocket_handler.py 通过 ``import src.visualization.app as _app`` 反向
   访问这些状态，形成循环依赖（app.py 在底部导入 routes.py 等）。

    现在共享状态提取到本模块（state.py），app.py 从此处再导出以保持向后兼容，
    routes.py / simulator.py / websocket_handler.py 直接从本模块导入状态，
    不再依赖 app.py 获取共享状态——仅依赖 app.py 获取辅助函数
    （``_get_ppo_model`` 等，测试通过 monkeypatch 替换）。

线程安全：
    当前 FastAPI + asyncio 为单线程事件循环，理论上线程安全风险较低。
    但未来若引入线程池（如 sync 路由 handler）或多 worker 部署，
    全局可变状态可能产生竞态条件。本模块提供 ``state_lock``（RLock）
    和线程安全访问器函数，新代码应优先使用访问器而非直接操作全局变量。

    现有代码（routes.py / simulator.py 等）因测试通过 monkeypatch 直接
    操作 app 模块属性，仍保留直接引用方式；新代码请使用本模块的访问器函数。
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from typing import Any

from loguru import logger

from src.visualization.connection import ConnectionManager
from src.visualization.security import WS_MAX_CONNECTIONS

# ============================================================
# 常量
# ============================================================

#: 可选调度策略列表（供前端下拉选择与后端策略校验）
STRATEGY_OPTIONS: list[str] = ["PPO", "DQN", "FCFS", "SJF", "Random"]

#: 资源利用率历史最大保留条数
MAX_RESOURCE_HISTORY: int = 100

#: 决策日志最大保留条数
MAX_DECISION_LOG: int = 200

#: 对战历史最大保留条数
MAX_BATTLE_HISTORY: int = 200

# ============================================================
# 线程安全锁
# ============================================================

#: 全局状态读写锁（RLock 允许同一线程重复获取）。
#: 用于线程安全访问器函数；现有直接引用代码因 asyncio 单线程模型暂不需要加锁。
state_lock = threading.RLock()


# ============================================================
# 全局可变状态（生产环境应替换为 Redis 等外部存储）
# ============================================================

# 当前系统状态
system_status: dict[str, Any] = {
    "qubit_utilization": 0.65,  # 量子比特利用率 (0~1)
    "queue_length": 5,  # 任务队列长度
    "average_wait_time": 12.3,  # 平均等待时间(秒)
    "completed_tasks": 42,  # 已完成任务数
    "current_step": 1024,  # 当前调度步数
    "current_strategy": "PPO",  # 当前调度策略
    "strategy_options": STRATEGY_OPTIONS.copy(),  # 可选策略列表
    "real_machines": [],  # 真机列表 [{name, status, type, id}]
    "real_submissions": [],  # 真机提交记录 [{step, task_id, machine, latency_s, status}]
    "last_update": datetime.now().isoformat(),
}

# 任务队列（初始示例任务）
task_queue: list[dict[str, Any]] = [
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
# Issue #514: 通过 WS_MAX_CONNECTIONS 限制最大并发连接数
manager = ConnectionManager(max_connections=WS_MAX_CONNECTIONS)

# 资源利用率历史数据（内存缓存，最多保留 MAX_RESOURCE_HISTORY 个数据点）
_resource_history: list[dict[str, Any]] = []

# 决策日志（内存缓存，最多保留 MAX_DECISION_LOG 条）
_decision_log: list[dict[str, Any]] = []

# PPO vs FCFS 对战状态（Day4-7-11）
_battle_state: dict[str, Any] = {
    "running": False,
    "step": 0,
    "ppo_reward": 0.0,
    "fcfs_reward": 0.0,
    "ppo_history": [],  # [{step, reward, cumulative, action, util}]
    "fcfs_history": [],
    "ppo_env": None,  # 运行时注入
    "fcfs_env": None,
    "ppo_obs": None,
    "fcfs_obs": None,
}


# ============================================================
# 线程安全访问器函数（新代码优先使用）
# ============================================================


def get_system_status() -> dict[str, Any]:
    """线程安全地获取系统状态字典的浅拷贝。

    Returns:
        system_status 的浅拷贝（修改拷贝不影响全局状态）
    """
    with state_lock:
        return dict(system_status)


def get_system_status_ref() -> dict[str, Any]:
    """线程安全地获取系统状态字典的引用（非拷贝）。

    用于需要原地修改状态的场景（如 simulator 后台任务更新指标）。
    调用方应确保在锁内完成修改，或接受 asyncio 单线程模型的隐式安全保证。

    Returns:
        system_status 字典引用
    """
    return system_status


def update_system_status(updates: dict[str, Any]) -> None:
    """线程安全地批量更新系统状态。

    Args:
        updates: 要合并到 system_status 的键值对
    """
    with state_lock:
        system_status.update(updates)
        system_status["last_update"] = datetime.now().isoformat()


def get_task_queue() -> list[dict[str, Any]]:
    """线程安全地获取任务队列的浅拷贝。

    Returns:
        task_queue 的浅拷贝列表
    """
    with state_lock:
        return list(task_queue)


def get_task_queue_ref() -> list[dict[str, Any]]:
    """线程安全地获取任务队列的引用（非拷贝）。

    Returns:
        task_queue 列表引用
    """
    return task_queue


def append_task(task: dict[str, Any]) -> None:
    """线程安全地向任务队列追加新任务。

    同时更新系统状态中的 queue_length。

    Args:
        task: 新任务字典
    """
    with state_lock:
        task_queue.append(task)
        system_status["queue_length"] = len([t for t in task_queue if t.get("status") == "pending"])
        system_status["last_update"] = datetime.now().isoformat()


def get_pending_task_count() -> int:
    """线程安全地获取等待中的任务数量。

    Returns:
        status == "pending" 的任务数量
    """
    with state_lock:
        return len([t for t in task_queue if t.get("status") == "pending"])


def append_resource_history(entry: dict[str, Any]) -> None:
    """线程安全地向资源历史追加数据点，自动裁剪到最大长度。

    Args:
        entry: 资源利用率数据点字典
    """
    with state_lock:
        _resource_history.append(entry)
        if len(_resource_history) > MAX_RESOURCE_HISTORY:
            _resource_history.pop(0)


def get_resource_history(limit: int = MAX_RESOURCE_HISTORY) -> list[dict[str, Any]]:
    """线程安全地获取最近的资源历史数据。

    Args:
        limit: 返回最近多少条数据点，默认全部

    Returns:
        资源历史列表（浅拷贝）
    """
    with state_lock:
        return list(_resource_history[-limit:])


def append_decision_log(entry: dict[str, Any]) -> None:
    """线程安全地向决策日志追加记录，自动裁剪到最大长度。

    Args:
        entry: 决策记录字典
    """
    with state_lock:
        _decision_log.append(entry)
        if len(_decision_log) > MAX_DECISION_LOG:
            _decision_log.pop(0)


def get_decision_log(limit: int = MAX_DECISION_LOG) -> list[dict[str, Any]]:
    """线程安全地获取最近的决策日志。

    Args:
        limit: 返回最近多少条记录，默认全部

    Returns:
        决策日志列表（浅拷贝）
    """
    with state_lock:
        return list(_decision_log[-limit:])


def get_battle_state() -> dict[str, Any]:
    """线程安全地获取对战状态的浅拷贝。

    Returns:
        _battle_state 的浅拷贝
    """
    with state_lock:
        return dict(_battle_state)


def get_battle_state_ref() -> dict[str, Any]:
    """线程安全地获取对战状态的引用（非拷贝）。

    Returns:
        _battle_state 字典引用
    """
    return _battle_state


def reset_battle_state() -> None:
    """线程安全地重置对战状态。"""
    with state_lock:
        _battle_state["running"] = False
        _battle_state["step"] = 0
        _battle_state["ppo_reward"] = 0.0
        _battle_state["fcfs_reward"] = 0.0
        _battle_state["ppo_history"] = []
        _battle_state["fcfs_history"] = []
        _battle_state["ppo_env"] = None
        _battle_state["fcfs_env"] = None
        _battle_state["ppo_obs"] = None
        _battle_state["fcfs_obs"] = None
        logger.info("[Web] 对战状态已重置")


def get_connection_manager() -> ConnectionManager:
    """获取 WebSocket 连接管理器实例。

    ConnectionManager 自身的方法（connect/disconnect/broadcast）已处理
    连接列表的并发修改，无需额外加锁。

    Returns:
        ConnectionManager 全局单例
    """
    return manager
