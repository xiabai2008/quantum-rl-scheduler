"""
REST API 路由处理器

使用 APIRouter 定义所有 HTTP 路由，并在 app.py 中通过 ``app.include_router(router)``
注册。路由路径与原 app.py 完全一致，保持向后兼容。

共享状态访问（Issue #179）：
    共享全局状态（``system_status`` / ``task_queue`` / ``manager`` /
    ``_resource_history`` / ``_decision_log`` / ``_battle_state``）从
    ``state.py`` 直接导入，不再通过 ``_app`` 中转，减少循环依赖耦合。

    辅助函数（``_get_ppo_model`` / ``_get_real_machines_status`` /
    ``_load_real_submissions`` / ``_get_quota_tracker`` / ``_load_vue3_template``）
    及懒加载状态（``_ppo_env``）和路径常量（``_PROJECT_ROOT``）仍通过
    ``_app`` 访问——这些符号被测试通过 ``monkeypatch.setattr(app_module, ...)``
    替换，必须保留在 app 模块上。
"""

import asyncio
import hmac
import json
import os
import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Header, HTTPException, Request, Response
from fastapi.responses import HTMLResponse
from loguru import logger
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

import src.visualization.app as _app
from src.visualization import state
from src.visualization.models import CircuitSubmit, SystemStatusUpdate, TaskSubmit
from src.visualization.security import (
    rate_limiter,
    validate_quantum_circuit,
)

router = APIRouter()


# ============================================================
# 速率限制依赖（Issue #517）
# ============================================================


async def rate_limit_dependency(
    request: Request,
    response: Response,
) -> None:
    """POST 端点速率限制依赖（Issue #517）。

    基于客户端 IP 的滑动窗口限流：每分钟最多 60 次请求（可通过
    ``VIZ_RATE_LIMIT`` 环境变量配置）。超出限制时返回 429 Too Many Requests。

    响应头：
        - ``X-RateLimit-Limit``: 时间窗口内最大请求数
        - ``X-RateLimit-Remaining``: 剩余可用请求数

    Args:
        request: FastAPI Request 对象（用于获取客户端 IP）
        response: FastAPI Response 对象（用于设置响应头）

    Raises:
        HTTPException: 超出速率限制时抛出 429 错误
    """
    client_ip = request.client.host if request.client else "unknown"
    allowed, limit, remaining = rate_limiter.check(client_ip)
    # 对成功响应设置速率限制头
    response.headers["X-RateLimit-Limit"] = str(limit)
    response.headers["X-RateLimit-Remaining"] = str(remaining)
    if not allowed:
        # 超出限制：429 响应也需携带速率限制头
        raise HTTPException(
            status_code=429,
            detail="请求过于频繁，请稍后再试",
            headers={
                "X-RateLimit-Limit": str(limit),
                "X-RateLimit-Remaining": "0",
                "Retry-After": "60",
            },
        )


# ============================================================
# 认证层：基于 X-API-Key 的可选 API 密钥认证
# ============================================================


async def verify_api_key(request: Request, x_api_key: str | None = Header(None)) -> None:
    """验证 API 密钥。未配置 VIZ_API_KEY 时仅放行只读 GET 请求。

    通过环境变量 ``VIZ_API_KEY`` 配置期望密钥：
    - GET 请求（只读端点，如监控/指标）：始终放行，不受认证影响。
    - 未配置 ``VIZ_API_KEY`` 时：写操作（POST/PUT/DELETE）一律返回 401，
      避免零认证放行的安全风险。
    - 已配置：POST/PUT/DELETE 请求头 ``X-API-Key`` 须通过恒定时间比较
      与配置值完全匹配，否则返回 401。
    """
    expected_key = os.getenv("VIZ_API_KEY")
    # GET 请求为只读操作（监控/指标端点），始终放行，不受认证影响。
    if request.method == "GET":
        return
    # 写操作（POST/PUT/DELETE）必须配置密钥，否则一律拒绝，避免零认证放行。
    if not expected_key:
        logger.warning("[Web] 未配置 VIZ_API_KEY，写操作被拒绝")
        raise HTTPException(status_code=401, detail="未配置认证，写操作被拒绝")
    # 使用恒定时间比较，防止时序侧信道攻击；x_api_key 为 None 时回退为空串。
    if not hmac.compare_digest(x_api_key or "", expected_key):
        logger.warning("[Web] API 密钥认证失败：X-API-Key 缺失或不匹配")
        raise HTTPException(status_code=401, detail="无效的 API 密钥")


async def require_api_key(x_api_key: str | None = Header(None)) -> None:
    """严格 API 密钥认证（不豁免 GET 请求）。

    Issue #513: /metrics 端点暴露内部运行时指标，即使 GET 请求也需认证。
    与 ``verify_api_key`` 不同，此函数不豁免 GET 请求。

    - 未配置 ``VIZ_API_KEY`` 时：返回 401，拒绝所有请求。
    - 已配置：请求头 ``X-API-Key`` 须通过恒定时间比较与配置值完全匹配。
    """
    expected_key = os.getenv("VIZ_API_KEY")
    if not expected_key:
        logger.warning("[Web] 未配置 VIZ_API_KEY，/metrics 端点拒绝访问")
        raise HTTPException(status_code=401, detail="未配置认证，指标端点不可访问")
    if not hmac.compare_digest(x_api_key or "", expected_key):
        logger.warning("[Web] /metrics 端点认证失败：X-API-Key 缺失或不匹配")
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
async def get_status(_auth: None = Depends(verify_api_key)) -> dict:
    """获取当前系统状态（JSON）"""
    return state.get_system_status()


@router.get("/api/real-machines")
async def get_real_machines(_auth: None = Depends(verify_api_key)) -> dict:
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
async def get_real_submissions(_auth: None = Depends(verify_api_key)) -> dict:
    """查询最近的真机提交记录（从 results/real_times.json 读取）。"""
    records = _app._load_real_submissions()
    return {
        "submissions": records,
        "count": len(records),
    }


@router.get("/api/tasks")
async def get_tasks(status: str | None = None, _auth: None = Depends(verify_api_key)) -> list[dict]:
    """
    获取任务列表
    - status=pending: 只返回等待中的任务
    - status=running: 只返回运行中的任务
    - status=completed: 只返回已完成的任务
    - 不传: 返回全部任务
    """
    if status:
        return [t for t in state.get_task_queue() if t["status"] == status]
    return state.get_task_queue()


@router.post("/api/tasks")
async def submit_task(
    task: TaskSubmit,
    _rate: None = Depends(rate_limit_dependency),
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
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
    state.append_task(new_task)
    # 广播更新
    await state.manager.broadcast(
        {
            "type": "task_added",
            "task": new_task,
            "status": state.get_system_status(),
        }
    )
    return {"message": "任务提交成功", "task_id": new_task["task_id"]}


@router.get("/api/metrics")
async def get_metrics(_auth: None = Depends(verify_api_key)) -> str:
    """返回 Prometheus 格式的指标（可选功能）"""
    status = state.get_system_status()
    lines = [
        "# HELP quantum_scheduler_qubit_utilization 量子比特利用率 0~1",
        "# TYPE quantum_scheduler_qubit_utilization gauge",
        f"quantum_scheduler_qubit_utilization {status['qubit_utilization']:.4f}",
        "",
        "# HELP quantum_scheduler_queue_length 任务队列长度",
        "# TYPE quantum_scheduler_queue_length gauge",
        f"quantum_scheduler_queue_length {status['queue_length']}",
        "",
        "# HELP quantum_scheduler_completed_tasks 已完成任务总数",
        "# TYPE quantum_scheduler_completed_tasks counter",
        f"quantum_scheduler_completed_tasks {status['completed_tasks']}",
        "",
        "# HELP quantum_scheduler_avg_wait_time 平均等待时间(秒)",
        "# TYPE quantum_scheduler_avg_wait_time gauge",
        f"quantum_scheduler_avg_wait_time {status['average_wait_time']:.2f}",
        "",
        "# HELP quantum_scheduler_current_step 当前调度步数",
        "# TYPE quantum_scheduler_current_step counter",
        f"quantum_scheduler_current_step {status['current_step']}",
    ]
    return "\n".join(lines)


@router.get("/metrics", tags=["监控"])
async def metrics(_auth: None = Depends(require_api_key)) -> Response:
    """Prometheus 指标端点，供 Prometheus 采集器抓取。

    返回 prometheus_client 默认注册表中所有指标的 Prometheus 文本格式输出，
    采集器（Prometheus server）可通过该端点定期拉取监控数据。

    Issue #513: /metrics 暴露内部运行时指标，即使 GET 请求也需严格认证，
    使用 require_api_key 而非 verify_api_key。
    """
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ============================================================
# 健康检查端点（Issue #214）
# ============================================================


@router.get("/health", tags=["运维"])
async def health() -> dict[str, Any]:
    """存活探针（Liveness Probe）。

    只要进程在运行就返回 200，用于判断应用是否还活着。
    不依赖任何外部资源，避免因外部抖动导致进程被重启。

    Returns:
        ``{"status": "alive"}``，HTTP 200
    """
    return {"status": "alive"}


@router.get("/ready", tags=["运维"])
async def ready() -> dict[str, Any]:
    """就绪探针（Readiness Probe）。

    检查应用关键依赖是否就绪：FastAPI app、Prometheus 注册表、
    模型加载状态、配额追踪器。任一关键依赖不可用返回 503。

    Returns:
        就绪状态字典，含各组件检查结果与总体 ready 标志
    """
    checks: dict[str, dict[str, Any]] = {}

    # 1. app 实例
    checks["app"] = {"ok": _app.app is not None}

    # 2. Prometheus 指标可采集
    try:
        metrics_body = generate_latest().decode("utf-8", errors="replace")
        checks["metrics"] = {"ok": len(metrics_body) > 0}
    except Exception as e:
        # Issue #516: 不泄露内部异常详情
        logger.debug(f"[Web] /ready metrics 检查失败: {e}")
        checks["metrics"] = {"ok": False, "error": "指标采集失败"}

    # 3. PPO 模型（可选依赖，未加载不算不可用）
    try:
        model = _app._get_ppo_model()
        checks["ppo_model"] = {"ok": model is not None, "required": False}
    except Exception as e:
        # Issue #516: 不泄露内部异常详情
        logger.debug(f"[Web] /ready ppo_model 检查失败: {e}")
        checks["ppo_model"] = {"ok": False, "required": False, "error": "模型检查失败"}

    # 4. 配额追踪器（可选）
    try:
        tracker = _app._get_quota_tracker()
        checks["quota_tracker"] = {"ok": tracker is not None, "required": False}
    except Exception as e:
        # Issue #516: 不泄露内部异常详情
        logger.debug(f"[Web] /ready quota_tracker 检查失败: {e}")
        checks["quota_tracker"] = {"ok": False, "required": False, "error": "配额检查失败"}

    # 总体就绪：所有 required=True 的检查必须通过
    required_ok = all(c["ok"] for c in checks.values() if c.get("required", True))

    return {
        "ready": required_ok,
        "checks": checks,
        "required_ok": required_ok,
        "timestamp": datetime.now().isoformat(),
    }


@router.post("/api/strategy")
async def switch_strategy(
    strategy: str,
    _rate: None = Depends(rate_limit_dependency),
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
    """切换调度策略"""
    current_status = state.get_system_status()
    if strategy not in current_status["strategy_options"]:
        return {"message": f"未知策略: {strategy}", "success": False}
    old = current_status["current_strategy"]
    state.update_system_status({"current_strategy": strategy})
    await state.manager.broadcast(
        {
            "type": "strategy_changed",
            "old_strategy": old,
            "new_strategy": strategy,
            "status": state.get_system_status(),
        }
    )
    return {"message": f"策略切换: {old} -> {strategy}", "success": True}


@router.post("/api/update")
async def update_status(
    update: SystemStatusUpdate,
    _rate: None = Depends(rate_limit_dependency),
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
    """更新系统状态（供调度引擎调用）"""
    state.update_system_status(
        {
            "qubit_utilization": update.qubit_utilization,
            "queue_length": update.queue_length,
            "completed_tasks": update.completed_tasks,
            "average_wait_time": update.average_wait_time,
        }
    )
    await state.manager.broadcast(
        {
            "type": "status_update",
            "status": state.get_system_status(),
        }
    )
    return {"message": "状态更新成功", "status": state.get_system_status()}


# ============================================================
# PPO 数据接口
# ============================================================


@router.get("/api/ppo/comparison")
async def get_ppo_comparison(_auth: None = Depends(verify_api_key)) -> dict:
    """返回 PPO 与其他策略的对比数据（从 v4 报告中读取）"""
    report_dir = os.path.join(_app._PROJECT_ROOT, "results")
    # Issue #712: listdir 前检查目录存在，避免目录缺失返回 500
    if not os.path.isdir(report_dir):
        return {"error": "结果目录不存在", "strategies": [], "ppo_rank": None}
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
        # Issue #516: 不泄露内部文件路径，返回通用错误消息
        return {"error": "无法读取仿真结果文件", "strategies": [], "ppo_rank": None}

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
async def ppo_predict(_auth: None = Depends(verify_api_key)) -> dict:
    """使用 PPO 模型对当前环境状态进行一次推理预测"""
    model = _app._get_ppo_model()
    if model is None:
        return {"error": "PPO 模型未加载", "action": None, "confidence": 0}

    try:
        # Issue #673: 使用独立环境实例，避免 API 调用 reset 污染后台仿真环境状态
        from src.scheduler.env import QuantumSchedulingEnv

        eval_env = QuantumSchedulingEnv(max_qubits=287, seed=42)
        obs, _ = eval_env.reset()
        # Issue #673: 使用 asyncio.to_thread 避免同步推理阻塞事件循环
        action, _states = await asyncio.to_thread(model.predict, obs, deterministic=True)
        eval_env.close()

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
        # Issue #516: 不泄露内部异常详情，返回通用错误消息
        return {"error": "PPO 推理失败", "action": None}


@router.get("/api/ppo/stats")
async def ppo_stats(_auth: None = Depends(verify_api_key)) -> dict:
    """返回 PPO 关键性能指标"""
    report_dir = os.path.join(_app._PROJECT_ROOT, "results")
    # Issue #712: listdir 前检查目录存在，避免目录缺失返回 500
    if not os.path.isdir(report_dir):
        return {"error": "结果目录不存在", "strategies": [], "ppo_rank": None}
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
async def get_quota(_auth: None = Depends(verify_api_key)) -> dict:
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
        # Issue #516: 不泄露内部异常详情
        return {"available": False, "message": "配额查询失败"}


@router.get("/api/resource-history")
async def get_resource_history(_auth: None = Depends(verify_api_key)) -> dict:
    """获取资源利用率历史趋势数据（Issue #22）。

    返回最近 100 个数据点的资源利用率历史，供前端 Echarts 折线图渲染。
    数据来源：后台 simulate_scheduler 每 3 秒采集一次。

    Returns:
        包含 history 列表的字典，每项含 step/qubit_utilization/queue_length/
        completed_tasks/average_wait_time 字段
    """
    return {"history": state.get_resource_history(100)}


@router.get("/api/decision-log")
async def get_decision_log(_auth: None = Depends(verify_api_key)) -> dict:
    """获取调度决策日志（Issue #22）。

    返回最近的决策记录列表，供前端决策过程回放组件渲染。
    每条记录包含 step/task_id/action/action_label/reward/source 字段。

    Returns:
        包含 decisions 列表的字典
    """
    return {"decisions": state.get_decision_log(200)}


@router.get("/api/machines-comparison")
async def get_machines_comparison(_auth: None = Depends(verify_api_key)) -> dict:
    """获取多机器对比数据（Issue #22）。

    聚合当前所有量子机器的关键指标（总量子比特、可用比率、保真度、
    队列深度、状态、单/双比特门保真度），供前端雷达图和对比表格渲染。

    Returns:
        包含 machines 列表的字典
    """
    current_status = state.get_system_status()
    machines: list[dict[str, Any]] = []
    for m in current_status.get("real_machines", []):
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
async def get_tenants(_auth: None = Depends(verify_api_key)) -> dict:
    """获取多租户配额状态（Issue #97）。

    返回所有租户的配额配置与运行时使用状态。

    Returns:
        包含 tenants 列表的字典
    """
    try:
        from src.scheduler.tenant import TenantQuotaManager

        mgr = TenantQuotaManager.from_config(str(_app._PROJECT_ROOT / "config" / "tenants.yaml"))
        return {"tenants": mgr.get_all_tenants_info()}
    except Exception as e:
        logger.debug(f"[Web] 租户状态查询失败: {e}")
        return {"tenants": []}


@router.get("/api/explainability")
async def get_explainability(limit: int = 20, _auth: None = Depends(verify_api_key)) -> dict:
    """获取最近决策的特征贡献度摘要（Issue #73）。

    从决策日志中提取包含 feature_contributions 的记录，
    返回最近 limit 条决策的可解释性数据。

    Args:
        limit: 返回最近多少条决策，默认 20，最大 200

    Returns:
        包含 decisions 列表和 count 的字典
    """
    decisions = state.get_decision_log(min(limit, 200))
    result = [
        {
            "step": d.get("step"),
            "action": d.get("action"),
            "action_label": d.get("action_label"),
            "feature_contributions": d.get("feature_contributions", {}),
            "explanation_text": d.get("explanation_text", ""),
        }
        for d in decisions
        if "feature_contributions" in d
    ]
    return {"decisions": result, "count": len(result)}


@router.get("/api/explainability/summary")
async def get_explainability_summary(_auth: None = Depends(verify_api_key)) -> dict:
    """获取当前会话的全局特征重要性排名（Issue #73）。

    聚合所有包含特征贡献度的决策记录，计算各特征的平均贡献度，
    并返回降序排列的特征重要性列表。

    Returns:
        包含 feature_importance 列表和 total_decisions 的字典
    """
    records = [d for d in state.get_decision_log() if "feature_contributions" in d]
    if not records:
        return {"feature_importance": [], "total_decisions": 0}

    accumulator: dict[str, float] = {}
    for d in records:
        for name, contrib in d["feature_contributions"].items():
            accumulator[name] = accumulator.get(name, 0.0) + contrib

    count = len(records)
    feature_importance = [
        {"feature": name, "importance": round(total / count, 6)}
        for name, total in sorted(accumulator.items(), key=lambda x: x[1], reverse=True)
    ]

    return {
        "feature_importance": feature_importance,
        "total_decisions": count,
    }


# ============================================================
# 决策放大镜：最新决策详情（Day2-3-10）
# ============================================================


@router.get("/api/explainability/latest")
async def get_explainability_latest(_auth: None = Depends(verify_api_key)) -> dict:
    """获取最新一条决策的完整可解释性数据（Day2-3-10）。

    返回最近一条包含 feature_contributions 的决策记录，
    包含状态向量、动作、特征贡献度、解释文本等完整信息，
    供前端决策放大镜面板实时展示。

    Returns:
        包含 latest 决策记录的字典；无记录时返回 empty=True
    """
    for d in reversed(state.get_decision_log()):
        if "feature_contributions" in d:
            return {
                "empty": False,
                "latest": d,
            }
    return {"empty": True, "latest": None}


# ============================================================
# PPO vs FCFS 实时对战面板（Day4-7-11）
# ============================================================


@router.post("/api/battle/start")
async def battle_start(
    _rate: None = Depends(rate_limit_dependency),
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
    """启动 PPO vs FCFS 对战（Day4-7-11）。

    初始化两个独立的调度环境实例，分别使用 PPO 和 FCFS 策略。
    后续通过 /api/battle/step 逐步推进对比。

    Returns:
        包含 success 和 initial state 的字典
    """
    try:
        from src.scheduler.env import QuantumSchedulingEnv

        # 创建两个独立环境（相同 seed 确保公平对比）
        battle = state.get_battle_state_ref()
        with state.state_lock:
            battle["ppo_env"] = QuantumSchedulingEnv(max_qubits=20, seed=42)
            battle["fcfs_env"] = QuantumSchedulingEnv(max_qubits=20, seed=42)

            battle["ppo_obs"], _ = battle["ppo_env"].reset()
            battle["fcfs_obs"], _ = battle["fcfs_env"].reset()

            battle["running"] = True
            battle["step"] = 0
            battle["ppo_reward"] = 0.0
            battle["fcfs_reward"] = 0.0
            battle["ppo_history"] = []
            battle["fcfs_history"] = []

        return {
            "success": True,
            "message": "对战已启动",
            "step": 0,
            "ppo_obs": battle["ppo_obs"].tolist()[:5],
            "fcfs_obs": battle["fcfs_obs"].tolist()[:5],
        }
    except Exception as e:
        logger.error(f"[Web] 对战启动失败: {e}")
        # Issue #516: 不泄露内部异常详情
        return {"success": False, "error": "对战启动失败"}


@router.post("/api/battle/step")
async def battle_step(
    _rate: None = Depends(rate_limit_dependency),
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
    """推进对战一步（Day4-7-11）。

    PPO 使用模型预测动作，FCFS 使用固定策略（始终选择动作 0=经典资源）。
    两个环境各 step 一次，记录奖励和状态。

    Returns:
        包含本步两个策略的 reward/action/util 和累积奖励的字典
    """
    battle = state.get_battle_state_ref()
    # Issue #388: 使用 state_lock 保护 battle 状态读写，防止并发调用导致数据损坏
    with state.state_lock:
        if not battle["running"]:
            return {"error": "对战未启动，请先调用 /api/battle/start"}

        try:
            # --- PPO 策略 ---
            model = _app._get_ppo_model()
            ppo_action = 0
            ppo_step_reward = 0.0
            ppo_util = 0.0
            ppo_done = False

            if model is not None:
                ppo_action, _ = model.predict(battle["ppo_obs"], deterministic=True)
                new_obs, reward, terminated, truncated, _info = battle["ppo_env"].step(
                    int(ppo_action)
                )
                ppo_step_reward = float(reward)
                ppo_util = float(new_obs[0])  # 量子比特可用率
                battle["ppo_reward"] += ppo_step_reward
                battle["ppo_obs"] = new_obs
                ppo_done = terminated or truncated
                if ppo_done:
                    battle["ppo_obs"], _ = battle["ppo_env"].reset()

            # --- FCFS 策略（固定选择经典资源=动作0） ---
            fcfs_action = 0
            new_obs, reward, terminated, truncated, _info = battle["fcfs_env"].step(0)
            fcfs_step_reward = float(reward)
            fcfs_util = float(new_obs[0])
            battle["fcfs_reward"] += fcfs_step_reward
            battle["fcfs_obs"] = new_obs
            fcfs_done = terminated or truncated
            if fcfs_done:
                battle["fcfs_obs"], _ = battle["fcfs_env"].reset()

            # 更新步数
            battle["step"] += 1
            step = battle["step"]

            # 记录历史
            ppo_entry = {
                "step": step,
                "reward": round(ppo_step_reward, 4),
                "cumulative": round(battle["ppo_reward"], 2),
                "action": int(ppo_action),
                "util": round(ppo_util, 4),
            }
            fcfs_entry = {
                "step": step,
                "reward": round(fcfs_step_reward, 4),
                "cumulative": round(battle["fcfs_reward"], 2),
                "action": int(fcfs_action),
                "util": round(fcfs_util, 4),
            }
            battle["ppo_history"].append(ppo_entry)
            battle["fcfs_history"].append(fcfs_entry)

            # 限制历史长度
            if len(battle["ppo_history"]) > 200:
                battle["ppo_history"] = battle["ppo_history"][-200:]
                battle["fcfs_history"] = battle["fcfs_history"][-200:]

            return {
                "step": step,
                "ppo": ppo_entry,
                "fcfs": fcfs_entry,
                "ppo_total": round(battle["ppo_reward"], 2),
                "fcfs_total": round(battle["fcfs_reward"], 2),
                "gap": round(battle["ppo_reward"] - battle["fcfs_reward"], 2),
            }
        except Exception as e:
            logger.error(f"[Web] 对战步进失败: {e}")
            # Issue #516: 不泄露内部异常详情
            return {"error": "对战步进失败"}


@router.get("/api/battle/status")
async def battle_status(_auth: None = Depends(verify_api_key)) -> dict:
    """获取对战当前状态（Day4-7-11）。

    Returns:
        包含 running/step/累积奖励/历史数据的字典
    """
    battle = state.get_battle_state_ref()
    # Issue #388: 读取 battle 状态也加锁，保证读到一致快照
    with state.state_lock:
        return {
            "running": battle["running"],
            "step": battle["step"],
            "ppo_total": round(battle["ppo_reward"], 2),
            "fcfs_total": round(battle["fcfs_reward"], 2),
            "gap": round(battle["ppo_reward"] - battle["fcfs_reward"], 2),
            "ppo_history": list(battle["ppo_history"][-50:]),
            "fcfs_history": list(battle["fcfs_history"][-50:]),
        }


@router.post("/api/battle/reset")
async def battle_reset(
    _rate: None = Depends(rate_limit_dependency),
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
    """重置对战状态（Day4-7-11）。"""
    state.reset_battle_state()
    return {"success": True, "message": "对战已重置"}


# ============================================================
# 量子电路提交端点（Issue #515）
# ============================================================


@router.post("/api/circuit/submit")
async def submit_circuit(
    payload: CircuitSubmit,
    _rate: None = Depends(rate_limit_dependency),
    _auth: None = Depends(verify_api_key),
) -> dict[str, Any]:
    """提交量子电路内容进行校验与调度（Issue #515）。

    接收 QCIS/QASM 格式的量子电路字符串，执行安全校验：
        1. 电路内容非空
        2. 仅包含合法可打印 ASCII 字符
        3. 门数不超过 10000（电路深度上限）
        4. 量子比特数不超过 105（天衍-287 可用比特上限）

    校验通过后返回校验摘要；校验失败返回 400 错误。

    Args:
        payload: 电路提交请求体（含 circuit / format / shots / task_name）

    Returns:
        包含 task_id、gate_count、qubit_count 的字典

    Raises:
        HTTPException: 电路无效时返回 400；速率超限返回 429；认证失败返回 401
    """
    # 电路内容校验（Issue #515）：无效时抛出 400 HTTPException
    validation = validate_quantum_circuit(payload.circuit, payload.format)

    task_id = "QCIR-" + uuid.uuid4().hex[:8]
    logger.info(
        f"[Web] 电路提交成功 task_id={task_id} "
        f"gates={validation['gate_count']} qubits={validation['qubit_count']} "
        f"format={payload.format} shots={payload.shots}"
    )

    return {
        "task_id": task_id,
        "message": "电路校验通过，已接受提交",
        "format": payload.format,
        "shots": payload.shots,
        "gate_count": validation["gate_count"],
        "qubit_count": validation["qubit_count"],
    }


# ============================================================
# 实时指标 API（Issue #526：实时调度过程可视化增强）
# ============================================================


@router.get("/api/realtime-metrics")
async def get_realtime_metrics(
    limit: int = 100,
    _auth: None = Depends(verify_api_key),
) -> dict:
    """获取实时指标历史数据（Issue #526）。

    返回吞吐量、平均等待时间趋势、量子/经典资源利用率、PPO vs Baseline reward 对比历史。

    Args:
        limit: 返回最近多少条数据点，默认 100，最大 200

    Returns:
        包含 metrics 历史列表和当前最新指标的字典
    """
    limit = min(max(limit, 1), 200)
    current_status = state.get_system_status()
    metrics_history = state.get_metrics_history(limit)
    reward_comparison = state.get_reward_comparison()

    return {
        "current": {
            "throughput": current_status.get("throughput", 0.0),
            "qubit_utilization": current_status.get("qubit_utilization", 0.0),
            "classical_utilization": current_status.get("classical_utilization", 0.0),
            "average_wait_time": current_status.get("average_wait_time", 0.0),
            "queue_length": current_status.get("queue_length", 0),
            "completed_tasks": current_status.get("completed_tasks", 0),
        },
        "history": metrics_history,
        "reward_comparison": reward_comparison,
    }


@router.get("/api/reward-comparison")
async def get_reward_comparison(
    limit: int = 50,
    _auth: None = Depends(verify_api_key),
) -> dict:
    """获取 PPO vs Baseline (FCFS) reward 对比数据（Issue #526）。

    Args:
        limit: 返回最近多少条历史点，默认 50

    Returns:
        包含累积奖励和对比曲线的字典
    """
    return state.get_reward_comparison()
