"""
WebSocket 端点处理

提供 /ws WebSocket 实时推送端点：客户端连接后服务端推送初始化状态，
并监听客户端心跳/指令消息。

共享状态访问（Issue #179）：
    共享全局状态（``manager`` / ``system_status`` / ``task_queue``）从
    ``state.py`` 直接导入。路径常量（``_PROJECT_ROOT``）仍通过 ``_app``
    访问——该符号被测试通过 ``monkeypatch.setattr(app_module, "_PROJECT_ROOT", ...)``
    替换，必须保留在 app 模块上。
"""

import json
import os

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

import src.visualization.app as _app
from src.visualization import state


async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket 实时推送端点

    客户端连接后，服务端会自动推送：
    - 状态更新（status_update）
    - 新任务通知（task_added）
    - 策略变更通知（strategy_changed）
    """
    await state.manager.connect(websocket)
    try:
        # 连接后立即发送当前状态 + PPO 数据
        ppo_stats: dict = {}
        try:
            report_dir = os.path.join(_app._PROJECT_ROOT, "results")
            json_files = sorted(
                [f for f in os.listdir(report_dir) if f.startswith("simulation_results_")],
                reverse=True,
            )
            if json_files:
                with open(os.path.join(report_dir, json_files[0])) as f:
                    sim_data = json.load(f)
                sorted_items = sorted(
                    sim_data.items(), key=lambda x: x[1].get("avg_reward", -9999), reverse=True
                )
                ppo_rank = next(
                    (i + 1 for i, (k, _) in enumerate(sorted_items) if "PPO" in k.upper()), None
                )
                ppo_stats = {"ppo_rank": ppo_rank, "total": len(sorted_items)}
        except (json.JSONDecodeError, OSError, KeyError) as e:
            # JSON 解析错误 / 文件 I/O 错误 / 数据字段缺失
            logger.debug(f"[Web] WebSocket 初始化读取 PPO 数据失败: {e}")

        await websocket.send_json(
            {
                "type": "init",
                "status": state.system_status,
                "tasks": state.task_queue,
                "ppo_stats": ppo_stats,
            }
        )
        # 保持连接，监听客户端消息（心跳/指令）
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except json.JSONDecodeError:
                # 忽略非 JSON 消息，避免连接断开
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "Invalid JSON format",
                    }
                )
                continue
            # 客户端可发送 {"action": "ping"} 作为心跳
            if msg.get("action") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        state.manager.disconnect(websocket)
