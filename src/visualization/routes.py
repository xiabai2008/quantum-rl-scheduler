"""
REST API 路由处理器

使用 APIRouter 定义所有 HTTP 路由，并在 app.py 中通过 ``app.include_router(router)``
注册。路由路径与原 app.py 完全一致，保持向后兼容。

为兼容测试对 app 模块全局状态的 monkeypatch，本模块通过 ``_app`` 引用访问
app 模块上的共享状态与辅助函数（system_status / task_queue / manager /
_get_ppo_model / _PROJECT_ROOT 等），确保运行时看到的总是 app 模块当前绑定。
"""

import json
import os
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.responses import HTMLResponse
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import src.visualization.app as _app
from src.visualization.models import SystemStatusUpdate, TaskSubmit

router = APIRouter()


# ============================================================
# 认证层：基于 X-API-Key 的可选 API 密钥认证
# ============================================================


async def verify_api_key(x_api_key: str | None = Header(None)) -> None:
    """验证 API 密钥。未配置 VISUALIZATION_API_KEY 时禁用认证。

    通过环境变量 ``VISUALIZATION_API_KEY`` 配置期望密钥：
    - 未配置（None 或空字符串）：认证禁用，所有请求放行（开发模式）。
    - 已配置：请求头 ``X-API-Key`` 必须与配置值完全匹配，否则返回 401。
    """
    expected_key = os.getenv("VISUALIZATION_API_KEY")
    if not expected_key:
        # 未配置密钥，认证禁用（开发环境）
        return
    if x_api_key != expected_key:
        logger.warning("[Web] API 密钥认证失败：X-API-Key 缺失或不匹配")
        raise HTTPException(status_code=401, detail="无效的 API 密钥")


# ============================================================
# 页面路由：返回监控面板 HTML
# ============================================================


@router.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    """返回监控面板 HTML 页面（Vue3 + Echarts 版本）"""
    return HTMLResponse(content=_app._load_vue3_template())


# ============================================================
# 基础 API 路由
# ============================================================


@router.get("/api/status")
async def get_status() -> dict:
    """获取当前系统状态（JSON）"""
    return _app.system_status


@router.get("/api/real-machines")
async def get_real_machines() -> dict:
    """查询天衍云真实量子计算机状态（实时轮询 cqlib）。

    返回 ``[{id, type, status, name}]``，其中 status 为
    running/calibrating/maintenance 等真实状态。
    无 TIANYAN_API_KEY 时返回空列表。
    """
    machines = _app._get_real_machines_status()
    return {
        "machines": machines,
        "count": len(machines),
        "source": "cqlib" if machines else "unavailable",
    }


@router.get("/api/real-submissions")
async def get_real_submissions() -> dict:
    """查询最近的真机提交记录（从 results/real_times.json 读取）。"""
    records = _app._load_real_submissions()
    return {
        "submissions": records,
        "count": len(records),
    }


@router.get("/api/tasks")
async def get_tasks(status: str | None = None) -> list[dict]:
    """
    获取任务列表
    - status=pending: 只返回等待中的任务
    - status=running: 只返回运行中的任务
    - status=completed: 只返回已完成的任务
    - 不传: 返回全部任务
    """
    if status:
        return [t for t in _app.task_queue if t["status"] == status]
    return _app.task_queue


@router.post("/api/tasks")
async def submit_task(task: TaskSubmit, _auth: None = Depends(verify_api_key)) -> dict:
    """提交新任务"""
    new_task = {
        "task_id": "QTASK-" + uuid.uuid4().hex[:8],
        "user_id": task.user_id,
        "task_type": task.task_type,
        "status": "pending",
        "priority": task.priority,
        "qubit_count": task.qubit_count,
        "circuit_depth": task.circuit_depth,
        "estimated_time": task.estimated_time,
        "arrival_time": datetime.now().isoformat(),
    }
    _app.task_queue.append(new_task)
    # 更新系统状态中的队列长度
    _app.system_status["queue_length"] = len(
        [t for t in _app.task_queue if t["status"] == "pending"]
    )
    _app.system_status["last_update"] = datetime.now().isoformat()
    # 广播更新
    await _app.manager.broadcast(
        {
            "type": "task_added",
            "task": new_task,
            "status": _app.system_status,
        }
    )
    return {"message": "任务提交成功", "task_id": new_task["task_id"]}


@router.get("/api/metrics")
async def get_metrics() -> str:
    """返回 Prometheus 格式的指标（可选功能）"""
    lines = [
        "# HELP quantum_scheduler_qubit_utilization 量子比特利用率 0~1",
        "# TYPE quantum_scheduler_qubit_utilization gauge",
        f"quantum_scheduler_qubit_utilization {_app.system_status['qubit_utilization']:.4f}",
        "",
        "# HELP quantum_scheduler_queue_length 任务队列长度",
        "# TYPE quantum_scheduler_queue_length gauge",
        f"quantum_scheduler_queue_length {_app.system_status['queue_length']}",
        "",
        "# HELP quantum_scheduler_completed_tasks 已完成任务总数",
        "# TYPE quantum_scheduler_completed_tasks counter",
        f"quantum_scheduler_completed_tasks {_app.system_status['completed_tasks']}",
        "",
        "# HELP quantum_scheduler_avg_wait_time 平均等待时间(秒)",
        "# TYPE quantum_scheduler_avg_wait_time gauge",
        f"quantum_scheduler_avg_wait_time {_app.system_status['average_wait_time']:.2f}",
        "",
        "# HELP quantum_scheduler_current_step 当前调度步数",
        "# TYPE quantum_scheduler_current_step counter",
        f"quantum_scheduler_current_step {_app.system_status['current_step']}",
    ]
    return "\n".join(lines)


@router.get("/metrics", tags=["监控"])
async def metrics() -> Response:
    """Prometheus 指标端点，供 Prometheus 采集器抓取。

    返回 prometheus_client 默认注册表中所有指标的 Prometheus 文本格式输出，
    采集器（Prometheus server）可通过该端点定期拉取监控数据。
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@router.post("/api/strategy")
async def switch_strategy(
    strategy: str, _auth: None = Depends(verify_api_key)
) -> dict:
    """切换调度策略"""
    if strategy not in _app.system_status["strategy_options"]:
        return {"message": f"未知策略: {strategy}", "success": False}
    old = _app.system_status["current_strategy"]
    _app.system_status["current_strategy"] = strategy
    _app.system_status["last_update"] = datetime.now().isoformat()
    await _app.manager.broadcast(
        {
            "type": "strategy_changed",
            "old_strategy": old,
            "new_strategy": strategy,
            "status": _app.system_status,
        }
    )
    return {"message": f"策略切换: {old} -> {strategy}", "success": True}


@router.post("/api/update")
async def update_status(
    update: SystemStatusUpdate, _auth: None = Depends(verify_api_key)
) -> dict:
    """更新系统状态（供调度引擎调用）"""
    _app.system_status["qubit_utilization"] = update.qubit_utilization
    _app.system_status["queue_length"] = update.queue_length
    _app.system_status["completed_tasks"] = update.completed_tasks
    _app.system_status["average_wait_time"] = update.average_wait_time
    _app.system_status["last_update"] = datetime.now().isoformat()
    await _app.manager.broadcast(
        {
            "type": "status_update",
            "status": _app.system_status,
        }
    )
    return {"message": "状态更新成功", "status": _app.system_status}


# ============================================================
# PPO 数据接口
# ============================================================


@router.get("/api/ppo/comparison")
async def get_ppo_comparison() -> dict:
    """返回 PPO 与其他策略的对比数据（从 v4 报告中读取）"""
    report_dir = os.path.join(_app._PROJECT_ROOT, "results")
    json_files = sorted(
        [
            f
            for f in os.listdir(report_dir)
            if f.startswith("simulation_results_") and f.endswith(".json")
        ],
        reverse=True,
    )
    if not json_files:
        return {"error": "未找到仿真结果文件", "strategies": [], "ppo_rank": None}

    latest_file = os.path.join(report_dir, json_files[0])
    try:
        with open(latest_file, encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # JSON 解析错误 / 文件 I/O 错误
        logger.error(f"[Web] 读取仿真结果文件失败: {e}")
        return {"error": f"无法读取: {latest_file}", "strategies": [], "ppo_rank": None}

    sorted_items = sorted(data.items(), key=lambda x: x[1].get("avg_reward", -9999), reverse=True)
    ppo_rank = next((i + 1 for i, (k, _) in enumerate(sorted_items) if "PPO" in k.upper()), None)

    strategies = []
    for rank, (name, metrics) in enumerate(sorted_items, 1):
        strategies.append(
            {
                "rank": rank,
                "name": name,
                "avg_reward": metrics.get("avg_reward", 0),
                "avg_wait_time": metrics.get("avg_wait_time", 0),
                "completion_rate": metrics.get("completion_rate", 0),
                "qubit_utilization": metrics.get("qubit_utilization", 0),
                "classical_utilization": metrics.get("classical_utilization", 0),
            }
        )

    return {
        "strategies": strategies,
        "ppo_rank": ppo_rank,
        "total_strategies": len(strategies),
        "data_source": json_files[0],
    }


@router.get("/api/ppo/predict")
async def ppo_predict() -> dict:
    """使用 PPO 模型对当前环境状态进行一次推理预测"""
    model = _app._get_ppo_model()
    if model is None:
        return {"error": "PPO 模型未加载", "action": None, "confidence": 0}

    try:
        if _app._ppo_env is None:
            return {"error": "PPO 环境未初始化", "action": None}
        obs = _app._ppo_env.reset()[0]
        action, _states = model.predict(obs, deterministic=True)

        action_map = {0: "经典资源", 1: "量子资源", 2: "混合执行"}
        return {
            "action": int(action),
            "action_name": action_map.get(int(action), "未知"),
            "observation": obs.tolist()[:5],
            "model_type": "PPO",
        }
    except (ValueError, RuntimeError, KeyError, OSError) as e:
        # 路由级错误边界：模型推理可能抛出值错误/运行时错误/键错误/IO错误
        logger.error(f"[Web] PPO 推理失败: {e}")
        return {"error": str(e), "action": None}


@router.get("/api/ppo/stats")
async def ppo_stats() -> dict:
    """返回 PPO 关键性能指标"""
    report_dir = os.path.join(_app._PROJECT_ROOT, "results")
    json_files = sorted(
        [
            f
            for f in os.listdir(report_dir)
            if f.startswith("simulation_results_") and f.endswith(".json")
        ],
        reverse=True,
    )
    if not json_files:
        return {"error": "未找到仿真结果"}

    try:
        with open(os.path.join(report_dir, json_files[0]), encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        # JSON 解析错误 / 文件 I/O 错误
        logger.error(f"[Web] 读取结果文件失败: {e}")
        return {"error": "无法读取结果文件"}

    ppo_data = None
    for k, v in data.items():
        if "PPO" in k.upper():
            ppo_data = v
            break

    if not ppo_data:
        return {"error": "未找到 PPO 数据"}

    # 计算排名
    sorted_items = sorted(data.items(), key=lambda x: x[1].get("avg_reward", -9999), reverse=True)
    ppo_rank = next(i + 1 for i, (k, _) in enumerate(sorted_items) if "PPO" in k.upper())
    best_name, best_data = sorted_items[0]

    return {
        "ppo": {
            "reward": ppo_data.get("avg_reward"),
            "wait_time": ppo_data.get("avg_wait_time"),
            "completion_rate": ppo_data.get("completion_rate"),
            "qubit_util": ppo_data.get("qubit_utilization"),
            "classical_util": ppo_data.get("classical_utilization"),
        },
        "ppo_rank": ppo_rank,
        "total": len(sorted_items),
        "best_strategy": best_name,
        "best_reward": best_data.get("avg_reward"),
        "vs_random": round(
            ppo_data.get("avg_reward", 0) - data.get("Random", {}).get("avg_reward", 0), 1
        ),
    }


# ============================================================
# 配额追踪、资源历史、决策日志、多机器对比、租户
# Issue #103 / #22 / #97
# ============================================================


@router.get("/api/quota")
async def get_quota() -> dict:
    """获取天衍云真机配额使用状态（Issue #103）。

    返回配额总量、已用、剩余、使用比例、预警等级等信息，
    供前端监控面板顶部进度条展示。
    """
    tracker = _app._get_quota_tracker()
    if tracker is None:
        return {"available": False, "message": "配额追踪未启用"}
    try:
        return {"available": True, **tracker.status()}
    except Exception as e:
        logger.warning(f"[Web] 获取配额状态失败: {e}")
        return {"available": False, "message": str(e)}


@router.get("/api/resource-history")
async def get_resource_history() -> dict:
    """获取资源利用率历史趋势数据（Issue #22）。

    返回最近 100 个数据点的资源利用率历史，供前端 Echarts 折线图渲染。
    数据来源：后台 simulate_scheduler 每 3 秒采集一次。

    Returns:
        包含 history 列表的字典，每项含 step/qubit_utilization/queue_length/
        completed_tasks/average_wait_time 字段
    """
    return {"history": _app._resource_history[-100:]}


@router.get("/api/decision-log")
async def get_decision_log() -> dict:
    """获取调度决策日志（Issue #22）。

    返回最近的决策记录列表，供前端决策过程回放组件渲染。
    每条记录包含 step/task_id/action/action_label/reward/source 字段。

    Returns:
        包含 decisions 列表的字典
    """
    return {"decisions": _app._decision_log[-200:]}


@router.get("/api/machines-comparison")
async def get_machines_comparison() -> dict:
    """获取多机器对比数据（Issue #22）。

    聚合当前所有量子机器的关键指标（总量子比特、可用比率、保真度、
    队列深度、状态、单/双比特门保真度），供前端雷达图和对比表格渲染。

    Returns:
        包含 machines 列表的字典
    """
    machines: list[dict[str, Any]] = []
    for m in _app.system_status.get("real_machines", []):
        machines.append(
            {
                "name": m.get("name", "unknown"),
                "total_qubits": m.get("total_qubits", 0),
                "available_ratio": m.get("available_ratio", 0.0),
                "fidelity": m.get("fidelity", 0.0),
                "queue_depth": m.get("queue_depth", 0),
                "status": m.get("status", "unknown"),
                "single_gate_fidelity": m.get("single_gate_fidelity", 0.0),
                "two_gate_fidelity": m.get("two_gate_fidelity", 0.0),
            }
        )
    return {"machines": machines}


@router.get("/api/tenants")
async def get_tenants() -> dict:
    """获取多租户配额状态（Issue #97）。

    返回所有租户的配额配置与运行时使用状态。

    Returns:
        包含 tenants 列表的字典
    """
    try:
        from src.scheduler.tenant import TenantQuotaManager

        mgr = TenantQuotaManager.from_config(
            str(_app._PROJECT_ROOT / "config" / "tenants.yaml")
        )
        return {"tenants": mgr.get_all_tenants_info()}
    except Exception as e:
        logger.debug(f"[Web] 租户状态查询失败: {e}")
        return {"tenants": []}
