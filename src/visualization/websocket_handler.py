"""
WebSocket 端点处理

提供 /ws WebSocket 实时推送端点：客户端连接后服务端推送初始化状态，
并监听客户端心跳/指令消息。

为兼容测试对 app 模块全局状态的 monkeypatch，本模块通过 ``_app`` 引用
访问 app 模块上的共享状态（manager / system_status / task_queue / _PROJECT_ROOT）。
"""

import json
import os

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

import src.visualization.app as _app


async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket 实时推送端点

    客户端连接后，服务端会自动推送：
    - 状态更新（status_update）
    - 新任务通知（task_added）
    - 策略变更通知（strategy_changed）
    """
    await _app.manager.connect(websocket)
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
                "status": _app.system_status,
                "tasks": _app.task_queue,
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
        _app.manager.disconnect(websocket)
