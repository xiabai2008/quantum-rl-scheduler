"""
后台仿真任务

simulate_scheduler：模拟调度引擎行为，使用 PPO 模型进行推理决策，
定时推送状态更新，并周期性轮询天衍云真机状态与提交记录。

为兼容测试对 app 模块全局状态的 monkeypatch，本模块通过 ``_app`` 引用
访问 app 模块上的共享状态与辅助函数（system_status / task_queue /
manager / _get_ppo_model 等），确保运行时看到的总是 app 模块当前绑定。
"""

import asyncio
import random
from datetime import datetime

from loguru import logger

# 通过 _app 访问 app 模块的全局状态与辅助函数，避免循环导入：
# 此处 app 模块虽可能处于部分加载状态，但 _app.X 均在函数体内访问，
# 实际调用时 app 模块已完成加载。
import src.visualization.app as _app


async def simulate_scheduler() -> None:
    """模拟调度引擎行为 — 使用 PPO 模型进行推理决策。

    每 3 秒推送一次状态更新。其中每 20 个 tick（约 60 秒）轮询一次天衍云
    真机状态（``query_quantum_computer_list``）和真机提交记录
    （``results/real_times.json``），将真实机器名/状态（running/calibrating/
    maintenance）与真实提交历史通过 WebSocket 推送到前端监控卡片。
    """
    tick = 0
    while True:
        await asyncio.sleep(3)
        tick += 1
        _app.system_status["current_step"] += 1

        # 本轮 PPO 推理动作（-1 表示未推理）
        action: int = -1

        # 尝试使用 PPO 推理
        model = _app._get_ppo_model()
        if model is not None and model.env is not None and _app._ppo_env is not None:
            try:
                obs = model.env.reset()[0]
                action, _ = model.predict(obs, deterministic=True)
                # 根据 PPO 预测更新利用率
                target_qubit = 0.45 if action == 1 else (0.40 if action == 2 else 0.35)
                _app.system_status["qubit_utilization"] = round(
                    _app.system_status["qubit_utilization"] * 0.7 + target_qubit * 0.3, 4
                )
            except (ValueError, RuntimeError, OSError) as e:
                # PPO 推理失败，回退随机
                logger.debug(f"[Web] PPO 推理失败，回退随机: {e}")
                _app.system_status["qubit_utilization"] = round(
                    max(
                        0.1,
                        min(
                            1.0,
                            _app.system_status["qubit_utilization"] + random.uniform(-0.03, 0.03),
                        ),
                    ),
                    4,
                )
        else:
            # 无模型，随机模拟
            _app.system_status["qubit_utilization"] = round(
                max(
                    0.1,
                    min(1.0, _app.system_status["qubit_utilization"] + random.uniform(-0.03, 0.03)),
                ),
                4,
            )

        _app.system_status["queue_length"] = len(
            [t for t in _app.task_queue if t["status"] == "pending"]
        )
        _app.system_status["average_wait_time"] = round(
            max(0.5, _app.system_status["average_wait_time"] + random.uniform(-0.5, 0.5)), 1
        )
        _app.system_status["last_update"] = datetime.now().isoformat()

        # 每 20 个 tick（约 60 秒）轮询真机状态 + 真机提交记录
        # 避免高频查询天衍云 API（免费额度有限）
        if tick % 20 == 0:
            try:
                real_machines = _app._get_real_machines_status()
                if real_machines:
                    _app.system_status["real_machines"] = real_machines
            except (OSError, RuntimeError, ValueError) as e:
                # 网络/ API 错误 / 运行时错误 / 返回值格式错误
                logger.error(f"[Web] 轮询真机状态异常: {e}")
            try:
                _app.system_status["real_submissions"] = _app._load_real_submissions()
            except (OSError, ValueError, RuntimeError) as e:
                # 文件 I/O 错误 / 数据格式错误 / 运行时错误
                logger.error(f"[Web] 加载真机提交记录异常: {e}")

        # PPO-Balanced 策略：平衡量子/经典资源分配
        pending = [t for t in _app.task_queue if t["status"] == "pending"]
        if pending and random.random() < 0.35:
            task = random.choice(pending)
            task["status"] = "completed"
            _app.system_status["completed_tasks"] += 1
            _app.system_status["queue_length"] = max(0, _app.system_status["queue_length"] - 1)

        pending = [t for t in _app.task_queue if t["status"] == "pending"]
        if pending and random.random() < 0.25:
            task = random.choice(pending)
            task["status"] = "running"

        # 记录资源利用率历史（Issue #22：资源利用率历史趋势图）
        _app._resource_history.append(
            {
                "step": _app.system_status["current_step"],
                "qubit_utilization": _app.system_status["qubit_utilization"],
                "queue_length": _app.system_status["queue_length"],
                "completed_tasks": _app.system_status["completed_tasks"],
                "average_wait_time": _app.system_status["average_wait_time"],
            }
        )
        if len(_app._resource_history) > 100:
            _app._resource_history.pop(0)

        # 记录决策日志（Issue #22：决策过程回放）
        if action >= 0:
            action_label_map = {0: "经典", 1: "量子", 2: "混合"}
            _app._decision_log.append(
                {
                    "step": _app.system_status["current_step"],
                    "task_id": f"task_{_app.system_status['current_step']}",
                    "action": int(action),
                    "action_label": action_label_map.get(int(action), "未知"),
                    "reward": round(_app.system_status["qubit_utilization"] * 10, 2),
                    "source": "PPO",
                }
            )
            if len(_app._decision_log) > 200:
                _app._decision_log.pop(0)

        await _app.manager.broadcast(
            {
                "type": "status_update",
                "status": _app.system_status,
                "tasks": _app.task_queue,
                "ppo_active": _app._ppo_model is not None,
            }
        )
