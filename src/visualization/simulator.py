"""
后台仿真循环
Background Simulation Loop

模拟量子RL调度系统的实时运行过程，包括：
- 使用 Gymnasium 环境进行调度仿真
- 使用 PPO/DQN 模型进行决策
- 收集资源利用率历史
- 记录调度决策日志
- 通过 WebSocket 实时推送状态更新

运行方式：由 app.py 的 lifespan 自动启动
"""

import asyncio
import random
import time
from datetime import datetime
from typing import Any

from loguru import logger

import src.visualization.app as _app

# 仿真更新间隔（秒）
SIMULATION_INTERVAL = 3.0

# 最大历史数据点
MAX_HISTORY_POINTS = 100
MAX_DECISION_LOG = 200

# 任务类型配置
TASK_TYPES = [
    {"task_type": "quantum", "qubit_count": 8, "circuit_depth": 100, "priority": 4},
    {"task_type": "quantum", "qubit_count": 16, "circuit_depth": 200, "priority": 5},
    {"task_type": "hybrid", "qubit_count": 4, "circuit_depth": 50, "priority": 3},
    {"task_type": "classical", "qubit_count": 0, "circuit_depth": 0, "priority": 2},
]

# 动作标签映射
ACTION_MAP = {0: "经典资源", 1: "量子资源", 2: "混合执行", 3: "等待", 4: "抢占"}


async def simulate_scheduler() -> None:
    """
    后台调度仿真主循环

    每 SIMULATION_INTERVAL 秒执行一次调度步骤：
    1. 使用 PPO 模型进行决策
    2. 更新系统状态（资源利用率、队列长度等）
    3. 收集资源历史数据
    4. 记录决策日志
    5. 通过 WebSocket 推送实时更新
    """
    logger.info("[Web] 调度仿真循环已启动")

    try:
        while True:
            try:
                await _step_simulation()
                await asyncio.sleep(SIMULATION_INTERVAL)
            except Exception as e:
                if isinstance(e, asyncio.CancelledError):
                    raise
                logger.error(f"[Web] 仿真循环异常: {e}")
                await asyncio.sleep(SIMULATION_INTERVAL)
    except asyncio.CancelledError:
        logger.info("[Web] 调度仿真循环已停止")
        raise


async def _step_simulation() -> None:
    """执行单次仿真步骤"""
    step_start = time.perf_counter()

    model = _app._get_ppo_model()
    env = _app._ppo_env

    action = None
    action_label = "未知"
    reward = 0.0
    observation = []

    if model is not None and env is not None:
        try:
            obs = env.reset()[0]
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            observation = obs.tolist()[:5]
            action_label = ACTION_MAP.get(int(action), f"动作{int(action)}")
        except Exception as e:
            logger.debug(f"[Web] PPO 推理失败，使用随机策略: {e}")
            action = random.randint(0, 4)
            action_label = ACTION_MAP.get(action, f"动作{action}")

    _update_system_status(action, action_label, reward)
    _record_resource_history()
    _record_decision(action, action_label, reward)

    elapsed = time.perf_counter() - step_start

    await _app.manager.broadcast(
        {
            "type": "status_update",
            "status": _app.system_status,
            "step_duration_ms": round(elapsed * 1000, 2),
        }
    )

    logger.debug(f"[Web] 仿真步骤完成: step={_app.system_status['current_step']}, "
                 f"action={action_label}, reward={reward:.2f}, "
                 f"elapsed={elapsed:.3f}s")


def _update_system_status(action: Any, action_label: str, reward: float) -> None:
    """更新系统状态"""
    status = _app.system_status

    status["current_step"] += 1
    status["last_update"] = datetime.now().isoformat()

    status["qubit_utilization"] = max(0.1, min(0.95,
        status["qubit_utilization"] + random.uniform(-0.03, 0.05)
    ))

    pending_count = len([t for t in _app.task_queue if t["status"] == "pending"])
    status["queue_length"] = pending_count

    status["completed_tasks"] += random.randint(0, 2)

    status["average_wait_time"] = max(1,
        status["average_wait_time"] + random.uniform(-2, 3)
    )

    status["current_action"] = action_label
    status["current_reward"] = round(reward, 2)


def _record_resource_history() -> None:
    """记录资源利用率历史"""
    point = {
        "step": _app.system_status["current_step"],
        "timestamp": datetime.now().isoformat(),
        "qubit_utilization": round(_app.system_status["qubit_utilization"], 4),
        "queue_length": _app.system_status["queue_length"],
        "completed_tasks": _app.system_status["completed_tasks"],
        "average_wait_time": round(_app.system_status["average_wait_time"], 2),
        "current_strategy": _app.system_status["current_strategy"],
    }
    _app._resource_history.append(point)
    if len(_app._resource_history) > MAX_HISTORY_POINTS:
        _app._resource_history.pop(0)


def _record_decision(action: Any, action_label: str, reward: float) -> None:
    """记录调度决策日志"""
    pending_tasks = [t for t in _app.task_queue if t["status"] == "pending"]
    task_id = pending_tasks[0]["task_id"] if pending_tasks else "NO_TASK"

    decision = {
        "step": _app.system_status["current_step"],
        "timestamp": datetime.now().isoformat(),
        "task_id": task_id,
        "action": int(action) if action is not None else -1,
        "action_label": action_label,
        "reward": round(reward, 4),
        "source": "PPO" if _app._ppo_model is not None else "random",
        "queue_length": _app.system_status["queue_length"],
        "qubit_utilization": round(_app.system_status["qubit_utilization"], 4),
    }
    _app._decision_log.append(decision)
    if len(_app._decision_log) > MAX_DECISION_LOG:
        _app._decision_log.pop(0)