"""
WebSocket 端点处理

提供 /ws WebSocket 实时推送端点：客户端连接后服务端推送初始化状态，
并监听客户端心跳/指令消息。

共享状态访问（Issue #179）：
    共享全局状态（``manager`` / ``system_status`` / ``task_queue``）从
    ``state.py`` 直接导入。路径常量（``_PROJECT_ROOT``）仍通过 ``_app``
    访问——该符号被测试通过 ``monkeypatch.setattr(app_module, "_PROJECT_ROOT", ...)``
    替换，必须保留在 app 模块上。

安全防护（Issue #514）：
    - Origin 头检查：拒绝未授权来源的跨站 WebSocket 连接（CSWSH 防护）
    - 消息大小限制：单条消息最大 1MB，防止内存耗尽攻击
    - 连接数限制：由 ``ConnectionManager`` 在 ``connect()`` 中强制执行
"""

import asyncio
import json
import os
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

import src.visualization.app as _app
from src.visualization import state
from src.visualization.security import WS_MAX_MESSAGE_BYTES, is_origin_allowed


def _load_ppo_stats(report_dir: str) -> dict[str, Any]:
    """从 results 目录读取最近的仿真结果并计算 PPO 排名（同步文件 I/O）。

    供 WebSocket 端点通过 ``asyncio.to_thread`` 调用，避免阻塞事件循环（Issue #739）。

    Args:
        report_dir: results 目录绝对路径

    Returns:
        包含 ``ppo_rank`` 和 ``total`` 的字典；无可用文件时返回空字典

    Raises:
        OSError: 目录/文件读取失败
        json.JSONDecodeError: JSON 解析失败
        KeyError: 数据字段缺失
    """
    json_files = sorted(
        [f for f in os.listdir(report_dir) if f.startswith("simulation_results_")],
        reverse=True,
    )
    if not json_files:
        return {}
    with open(os.path.join(report_dir, json_files[0])) as f:
        sim_data = json.load(f)
    sorted_items = sorted(
        sim_data.items(), key=lambda x: x[1].get("avg_reward", -9999), reverse=True
    )
    ppo_rank = next((i + 1 for i, (k, _) in enumerate(sorted_items) if "PPO" in k.upper()), None)
    return {"ppo_rank": ppo_rank, "total": len(sorted_items)}


def _check_websocket_origin(websocket: WebSocket) -> bool:
    """检查 WebSocket 请求的 Origin 头是否允许连接（Issue #514）。

    防御跨站 WebSocket 劫持（CSWSH）：浏览器会在 WebSocket 握手请求中
    自动携带 Origin 头，服务端通过白名单校验拒绝恶意网站发起的连接。

    8.7-v4 修复：异常/非字符串 Origin 一律拒绝（fail-closed），
    原实现 return True 放行会被攻击者利用（伪造异常/非字符串 Origin 绕过校验）。

    Args:
        websocket: WebSocket 连接对象

    Returns:
        True 表示 Origin 允许；False 表示应拒绝连接
    """
    try:
        origin = websocket.headers.get("origin", "")
    except Exception:  # noqa: BLE001 - headers 不可读时按拒绝处理（fail-closed）
        return False
    # 非字符串 Origin（如异常 mock 对象）拒绝，避免伪造绕过
    if not isinstance(origin, str):
        return False
    return is_origin_allowed(origin)


async def websocket_endpoint(websocket: WebSocket) -> None:
    """
    WebSocket 实时推送端点

    客户端连接后，服务端会自动推送：
    - 状态更新（status_update）
    - 新任务通知（task_added）
    - 策略变更通知（strategy_changed）

    安全防护（Issue #514）：
        - Origin 头检查：拒绝未授权来源的连接
        - 消息大小限制：单条消息最大 1MB
        - 连接数限制：由 ConnectionManager 强制执行

    连接清理（Issue #216）：
        使用 try/finally 确保任何异常（包括非 WebSocketDisconnect 的异常，
        如 RuntimeError / ConnectionError / asyncio.CancelledError）都会
        调用 ``manager.disconnect(websocket)``，避免连接泄漏。
    """
    # ── Origin 头检查（CSWSH 防护，Issue #514）──
    if not _check_websocket_origin(websocket):
        logger.warning("[Web] WebSocket 连接被拒绝：Origin 不在允许列表中")
        try:
            await websocket.close(code=1008)  # Policy Violation
        except Exception as e:  # noqa: BLE001
            logger.debug(f"[Web] 拒绝连接时 close 失败: {e}")
        return

    # ── 连接数限制 + 接受连接（Issue #514）──
    connected = await state.manager.connect(websocket)
    if not connected:
        # 连接数达上限，已被 ConnectionManager 关闭，直接返回
        return

    try:
        # 连接后立即发送当前状态 + PPO 数据
        ppo_stats: dict[str, Any] = {}
        try:
            report_dir = os.path.join(_app._PROJECT_ROOT, "results")
            # 通过线程池执行同步文件 I/O，避免阻塞事件循环（Issue #739）
            ppo_stats = await asyncio.to_thread(_load_ppo_stats, report_dir)
        except (json.JSONDecodeError, OSError, KeyError) as e:
            # JSON 解析错误 / 文件 I/O 错误 / 数据字段缺失
            logger.debug(f"[Web] WebSocket 初始化读取 PPO 数据失败: {e}")

        await websocket.send_json(
            {
                "type": "init",
                "status": state.get_system_status(),
                "tasks": state.get_task_queue(),
                "ppo_stats": ppo_stats,
                "realtime_metrics_history": state.get_metrics_history(50),
                "reward_comparison": state.get_reward_comparison(),
            }
        )
        # 保持连接，监听客户端消息（心跳/指令）
        while True:
            data = await websocket.receive_text()

            # ── 消息大小限制（DoS 防护，Issue #514）──
            if len(data.encode("utf-8")) > WS_MAX_MESSAGE_BYTES:
                logger.warning(
                    f"[Web] WebSocket 消息超过大小限制 {WS_MAX_MESSAGE_BYTES} 字节，已拒绝"
                )
                await websocket.send_json(
                    {
                        "type": "error",
                        "message": "消息过大，已超过大小限制",
                    }
                )
                continue

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

            # 客户端请求决策日志（Issue #161：实时调度过程可视化）
            if msg.get("action") == "get_decisions":
                await websocket.send_json(
                    {
                        "type": "decision_log",
                        "decisions": state.get_decision_log(limit=200),
                    }
                )

            # 客户端请求资源利用率历史（Issue #161：实时调度过程可视化）
            if msg.get("action") == "get_resource_history":
                await websocket.send_json(
                    {
                        "type": "resource_history",
                        "history": state.get_resource_history(limit=100),
                    }
                )
    except WebSocketDisconnect:
        # 客户端主动断开：正常流程，无需告警
        pass
    except Exception as e:  # noqa: BLE001
        # 非 WebSocketDisconnect 异常（如 RuntimeError / ConnectionError /
        # asyncio.CancelledError）：记录日志，避免静默泄漏（Issue #216）
        logger.warning(f"[Web] WebSocket 异常关闭: {type(e).__name__}: {e}")
    finally:
        # 无论何种异常，都确保从 ConnectionManager 清理连接（Issue #216）
        # disconnect 内部已处理"连接不在列表"的情况，重复调用安全
        state.manager.disconnect(websocket)
