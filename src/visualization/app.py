"""
Web可视化监控界面
Web Visualization Monitoring Dashboard

基于 FastAPI + 原生 HTML/JS 的量子RL调度系统监控界面
支持 WebSocket 实时推送、手动任务提交、调度策略切换等功能

运行方式:
    python src/visualization/app.py
    或
    python -m src.visualization.app
"""

import asyncio
import json
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# ============================================================
# 数据模型定义
# ============================================================

class TaskSubmit(BaseModel):
    """提交新任务的请求体"""
    user_id: str = Field(default="user_001", description="用户ID")
    task_type: str = Field(default="quantum", description="任务类型: quantum/classical/hybrid")
    priority: int = Field(default=3, ge=1, le=5, description="优先级 1-5")
    qubit_count: int = Field(default=10, ge=1, description="所需量子比特数")
    circuit_depth: int = Field(default=100, ge=1, description="电路深度")
    estimated_time: float = Field(default=60.0, ge=0.1, description="预计执行时间(秒)")


class SystemStatusUpdate(BaseModel):
    """系统状态更新请求体（供调度引擎调用）"""
    qubit_utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    queue_length: int = Field(default=0, ge=0)
    completed_tasks: int = Field(default=0, ge=0)
    average_wait_time: float = Field(default=0.0, ge=0.0)


# ============================================================
# 内存存储（生产环境应替换为 Redis 等外部存储）
# ============================================================

# 当前系统状态
system_status: Dict = {
    "qubit_utilization": 0.65,       # 量子比特利用率 (0~1)
    "queue_length": 5,               # 任务队列长度
    "average_wait_time": 12.3,       # 平均等待时间(秒)
    "completed_tasks": 42,           # 已完成任务数
    "current_step": 1024,            # 当前调度步数
    "current_strategy": "DQN-Reward",  # 当前调度策略
    "strategy_options": [            # 可选策略列表
        "DQN-Reward",
        "DQN-Latency",
        "PPO-Balanced",
        "QAOA-Hybrid",
        "FCFS",
    ],
    "last_update": datetime.now().isoformat(),
}

# 任务队列
task_queue: List[Dict] = [
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

# WebSocket 连接管理器
class ConnectionManager:
    """管理所有 WebSocket 客户端连接"""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """向所有连接的客户端广播消息"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)


manager = ConnectionManager()


# ============================================================
# 后台模拟任务：定时更新状态（模拟调度引擎行为）
# ============================================================

async def simulate_scheduler():
    """模拟调度引擎行为，定时更新系统状态"""
    import random
    while True:
        await asyncio.sleep(3)  # 每3秒模拟一次调度
        # 随机波动量子比特利用率
        system_status["qubit_utilization"] = round(
            max(0.1, min(1.0, system_status["qubit_utilization"] + random.uniform(-0.05, 0.05))),
            4,
        )
        system_status["current_step"] += 1
        system_status["queue_length"] = len([t for t in task_queue if t["status"] == "pending"])
        system_status["average_wait_time"] = round(
            max(0.5, system_status["average_wait_time"] + random.uniform(-1.0, 1.0)),
            1,
        )
        system_status["last_update"] = datetime.now().isoformat()
        # 随机完成一个 pending 任务
        pending = [t for t in task_queue if t["status"] == "pending"]
        if pending and random.random() < 0.3:
            task = random.choice(pending)
            task["status"] = "completed"
            system_status["completed_tasks"] += 1
            system_status["queue_length"] = max(0, system_status["queue_length"] - 1)
        # 随机启动一个 pending 任务
        pending = [t for t in task_queue if t["status"] == "pending"]
        if pending and random.random() < 0.2:
            task = random.choice(pending)
            task["status"] = "running"
        # 广播状态更新
        await manager.broadcast({
            "type": "status_update",
            "status": system_status,
            "tasks": task_queue,
        })


@asynccontextmanager
async def lifespan(app_instance: FastAPI):
    """应用生命周期：启动时开启后台模拟任务"""
    task = asyncio.create_task(simulate_scheduler())
    yield
    task.cancel()


# ============================================================
# FastAPI 应用实例
# ============================================================

app = FastAPI(title="量子RL调度系统监控界面", version="1.0.0", lifespan=lifespan)


# ============================================================
# 页面路由：返回监控面板 HTML
# ============================================================

@app.get("/", response_class=HTMLResponse)
async def root():
    """返回监控面板 HTML 页面"""
    return HTMLResponse(content=HTML_TEMPLATE)


# ============================================================
# API 路由
# ============================================================

@app.get("/api/status")
async def get_status():
    """获取当前系统状态（JSON）"""
    return system_status


@app.get("/api/tasks")
async def get_tasks(status: Optional[str] = None):
    """
    获取任务列表
    - status=pending: 只返回等待中的任务
    - status=running: 只返回运行中的任务
    - status=completed: 只返回已完成的任务
    - 不传: 返回全部任务
    """
    if status:
        return [t for t in task_queue if t["status"] == status]
    return task_queue


@app.post("/api/tasks")
async def submit_task(task: TaskSubmit):
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
    task_queue.append(new_task)
    # 更新系统状态中的队列长度
    system_status["queue_length"] = len([t for t in task_queue if t["status"] == "pending"])
    system_status["last_update"] = datetime.now().isoformat()
    # 广播更新
    await manager.broadcast({
        "type": "task_added",
        "task": new_task,
        "status": system_status,
    })
    return {"message": "任务提交成功", "task_id": new_task["task_id"]}


@app.get("/api/metrics")
async def get_metrics():
    """返回 Prometheus 格式的指标（可选功能）"""
    lines = [
        "# HELP quantum_scheduler_qubit_utilization 量子比特利用率 0~1",
        "# TYPE quantum_scheduler_qubit_utilization gauge",
        f"quantum_scheduler_qubit_utilization {system_status['qubit_utilization']:.4f}",
        "",
        "# HELP quantum_scheduler_queue_length 任务队列长度",
        "# TYPE quantum_scheduler_queue_length gauge",
        f"quantum_scheduler_queue_length {system_status['queue_length']}",
        "",
        "# HELP quantum_scheduler_completed_tasks 已完成任务总数",
        "# TYPE quantum_scheduler_completed_tasks counter",
        f"quantum_scheduler_completed_tasks {system_status['completed_tasks']}",
        "",
        "# HELP quantum_scheduler_avg_wait_time 平均等待时间(秒)",
        "# TYPE quantum_scheduler_avg_wait_time gauge",
        f"quantum_scheduler_avg_wait_time {system_status['average_wait_time']:.2f}",
        "",
        "# HELP quantum_scheduler_current_step 当前调度步数",
        "# TYPE quantum_scheduler_current_step counter",
        f"quantum_scheduler_current_step {system_status['current_step']}",
    ]
    return "\n".join(lines)


@app.post("/api/strategy")
async def switch_strategy(strategy: str):
    """切换调度策略"""
    if strategy not in system_status["strategy_options"]:
        return {"message": f"未知策略: {strategy}", "success": False}
    old = system_status["current_strategy"]
    system_status["current_strategy"] = strategy
    system_status["last_update"] = datetime.now().isoformat()
    await manager.broadcast({
        "type": "strategy_changed",
        "old_strategy": old,
        "new_strategy": strategy,
        "status": system_status,
    })
    return {"message": f"策略切换: {old} -> {strategy}", "success": True}


@app.post("/api/update")
async def update_status(update: SystemStatusUpdate):
    """更新系统状态（供调度引擎调用）"""
    system_status["qubit_utilization"] = update.qubit_utilization
    system_status["queue_length"] = update.queue_length
    system_status["completed_tasks"] = update.completed_tasks
    system_status["average_wait_time"] = update.average_wait_time
    system_status["last_update"] = datetime.now().isoformat()
    await manager.broadcast({
        "type": "status_update",
        "status": system_status,
    })
    return {"message": "状态更新成功", "status": system_status}


# ============================================================
# WebSocket 路由：实时推送状态更新
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket 实时推送端点
    客户端连接后，服务端会自动推送：
    - 状态更新（status_update）
    - 新任务通知（task_added）
    - 策略变更通知（strategy_changed）
    """
    await manager.connect(websocket)
    try:
        # 连接后立即发送当前状态
        await websocket.send_json({
            "type": "init",
            "status": system_status,
            "tasks": task_queue,
        })
        # 保持连接，监听客户端消息（心跳/指令）
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)
            # 客户端可发送 {"action": "ping"} 作为心跳
            if msg.get("action") == "ping":
                await websocket.send_json({"type": "pong"})
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============================================================
# 后台模拟任务：定时更新状态（模拟调度引擎行为）
# ============================================================

async def simulate_scheduler():
    """模拟调度引擎行为，定时更新系统状态"""
    while True:
        await asyncio.sleep(3)  # 每3秒模拟一次调度
        # 随机波动量子比特利用率
        import random
        system_status["qubit_utilization"] = round(
            max(0.1, min(1.0, system_status["qubit_utilization"] + random.uniform(-0.05, 0.05))),
            4,
        )
        system_status["current_step"] += 1
        system_status["queue_length"] = len([t for t in task_queue if t["status"] == "pending"])
        system_status["average_wait_time"] = round(
            max(0.5, system_status["average_wait_time"] + random.uniform(-1.0, 1.0)),
            1,
        )
        system_status["last_update"] = datetime.now().isoformat()
        # 随机完成一个 pending 任务
        pending = [t for t in task_queue if t["status"] == "pending"]
        if pending and random.random() < 0.3:
            task = random.choice(pending)
            task["status"] = "completed"
            system_status["completed_tasks"] += 1
            system_status["queue_length"] = max(0, system_status["queue_length"] - 1)
        # 随机启动一个 pending 任务
        pending = [t for t in task_queue if t["status"] == "pending"]
        if pending and random.random() < 0.2:
            task = random.choice(pending)
            task["status"] = "running"
        # 广播状态更新
        await manager.broadcast({
            "type": "status_update",
            "status": system_status,
            "tasks": task_queue,
        })


# ============================================================
# 前端 HTML 模板（原生 HTML/CSS/JS，不依赖前端框架）
# ============================================================

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>量子RL调度系统 - 监控面板</title>
    <style>
        /* ===== 全局样式 ===== */
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            min-height: 100vh;
        }

        /* ===== 顶部标题栏 ===== */
        .header {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            border-bottom: 1px solid #334155;
            padding: 16px 32px;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .header h1 {
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(90deg, #60a5fa, #a78bfa);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .header .ws-status {
            font-size: 13px;
            padding: 4px 12px;
            border-radius: 12px;
            background: #1e293b;
            border: 1px solid #334155;
        }
        .ws-status.connected { color: #4ade80; border-color: #22c55e; }
        .ws-status.disconnected { color: #f87171; border-color: #ef4444; }

        /* ===== 系统状态卡片区域 ===== */
        .status-cards {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 16px;
            padding: 24px 32px;
        }
        .status-card {
            background: linear-gradient(145deg, #1e293b, #1a2332);
            border: 1px solid #334155;
            border-radius: 12px;
            padding: 20px;
            transition: border-color 0.3s;
        }
        .status-card:hover { border-color: #60a5fa; }
        .status-card .card-label {
            font-size: 13px;
            color: #94a3b8;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .status-card .card-value {
            font-size: 32px;
            font-weight: 700;
            line-height: 1.2;
        }
        .status-card .card-sub {
            font-size: 12px;
            color: #64748b;
            margin-top: 6px;
        }
        /* 卡片颜色主题 */
        .card-blue .card-value { color: #60a5fa; }
        .card-purple .card-value { color: #a78bfa; }
        .card-green .card-value { color: #4ade80; }
        .card-amber .card-value { color: #fbbf24; }
        .card-cyan .card-value { color: #22d3ee; }

        /* ===== 主内容区域 ===== */
        .main-content {
            padding: 0 32px 32px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }

        /* ===== 通用面板样式 ===== */
        .panel {
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 12px;
            overflow: hidden;
        }
        .panel-header {
            padding: 14px 20px;
            border-bottom: 1px solid #334155;
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        .panel-header h2 {
            font-size: 16px;
            font-weight: 600;
        }
        .panel-header .badge {
            font-size: 12px;
            padding: 2px 10px;
            border-radius: 10px;
            background: #334155;
            color: #94a3b8;
        }
        .panel-body { padding: 16px 20px; }

        /* ===== 任务队列表格 ===== */
        .task-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        .task-table th {
            text-align: left;
            padding: 10px 12px;
            color: #94a3b8;
            font-weight: 600;
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid #334155;
        }
        .task-table td {
            padding: 10px 12px;
            border-bottom: 1px solid #1e293b;
        }
        .task-table tbody tr:hover { background: #253347; }
        .task-table tbody tr { transition: background 0.2s; }
        /* 状态标签 */
        .status-tag {
            display: inline-block;
            padding: 2px 10px;
            border-radius: 10px;
            font-size: 12px;
            font-weight: 600;
        }
        .status-tag.pending { background: rgba(251, 191, 36, 0.15); color: #fbbf24; }
        .status-tag.running { background: rgba(96, 165, 250, 0.15); color: #60a5fa; }
        .status-tag.completed { background: rgba(74, 222, 128, 0.15); color: #4ade80; }
        .status-tag.failed { background: rgba(248, 113, 113, 0.15); color: #f87171; }
        /* 优先级 */
        .priority-high { color: #f87171; }
        .priority-medium { color: #fbbf24; }
        .priority-low { color: #4ade80; }

        /* ===== 控制面板 ===== */
        .control-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }
        .control-section h3 {
            font-size: 14px;
            color: #94a3b8;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        /* 表单样式 */
        .form-group {
            margin-bottom: 12px;
        }
        .form-group label {
            display: block;
            font-size: 13px;
            color: #94a3b8;
            margin-bottom: 4px;
        }
        .form-group input,
        .form-group select {
            width: 100%;
            padding: 8px 12px;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            color: #e2e8f0;
            font-size: 14px;
            outline: none;
            transition: border-color 0.2s;
        }
        .form-group input:focus,
        .form-group select:focus {
            border-color: #60a5fa;
        }
        .form-row {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
        }
        /* 按钮 */
        .btn {
            padding: 10px 20px;
            border: none;
            border-radius: 8px;
            font-size: 14px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }
        .btn-primary {
            background: linear-gradient(135deg, #3b82f6, #6366f1);
            color: white;
        }
        .btn-primary:hover { opacity: 0.9; transform: translateY(-1px); }
        .btn-secondary {
            background: #334155;
            color: #e2e8f0;
        }
        .btn-secondary:hover { background: #475569; }
        .btn-secondary.active {
            background: linear-gradient(135deg, #3b82f6, #6366f1);
            color: white;
        }

        /* 策略选择按钮组 */
        .strategy-buttons {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        .strategy-btn {
            padding: 8px 16px;
            background: #0f172a;
            border: 1px solid #334155;
            border-radius: 8px;
            color: #94a3b8;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
        }
        .strategy-btn:hover { border-color: #60a5fa; color: #e2e8f0; }
        .strategy-btn.active {
            background: linear-gradient(135deg, #3b82f6, #6366f1);
            border-color: transparent;
            color: white;
        }

        /* ===== 通知 Toast ===== */
        .toast-container {
            position: fixed;
            top: 80px;
            right: 24px;
            z-index: 1000;
            display: flex;
            flex-direction: column;
            gap: 8px;
        }
        .toast {
            padding: 12px 20px;
            background: #1e293b;
            border: 1px solid #334155;
            border-radius: 10px;
            font-size: 14px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.4);
            animation: slideIn 0.3s ease-out;
            max-width: 320px;
        }
        .toast.success { border-left: 3px solid #4ade80; }
        .toast.info { border-left: 3px solid #60a5fa; }
        .toast.warn { border-left: 3px solid #fbbf24; }
        @keyframes slideIn {
            from { transform: translateX(100%); opacity: 0; }
            to { transform: translateX(0); opacity: 1; }
        }

        /* ===== 空状态 ===== */
        .empty-state {
            text-align: center;
            padding: 40px;
            color: #64748b;
            font-size: 14px;
        }

        /* ===== 响应式 ===== */
        @media (max-width: 768px) {
            .header { padding: 12px 16px; }
            .header h1 { font-size: 16px; }
            .status-cards { padding: 16px; gap: 12px; }
            .main-content { padding: 0 16px 16px; }
            .control-grid { grid-template-columns: 1fr; }
            .form-row { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>

    <!-- 顶部标题栏 -->
    <div class="header">
        <h1>量子RL调度系统 - 监控面板</h1>
        <span id="ws-status" class="ws-status disconnected">WebSocket 未连接</span>
    </div>

    <!-- 系统状态卡片 -->
    <div class="status-cards">
        <div class="status-card card-blue">
            <div class="card-label">量子比特利用率</div>
            <div class="card-value" id="val-qubit">0%</div>
            <div class="card-sub" id="sub-qubit">实时资源占用</div>
        </div>
        <div class="status-card card-purple">
            <div class="card-label">任务队列长度</div>
            <div class="card-value" id="val-queue">0</div>
            <div class="card-sub">等待调度执行</div>
        </div>
        <div class="status-card card-amber">
            <div class="card-label">平均等待时间</div>
            <div class="card-value" id="val-wait">0s</div>
            <div class="card-sub">最近100个任务</div>
        </div>
        <div class="status-card card-green">
            <div class="card-label">已完成任务</div>
            <div class="card-value" id="val-completed">0</div>
            <div class="card-sub">累计完成数</div>
        </div>
        <div class="status-card card-cyan">
            <div class="card-label">当前调度策略</div>
            <div class="card-value" id="val-strategy" style="font-size:20px;">-</div>
            <div class="card-sub" id="val-step">Step: 0</div>
        </div>
    </div>

    <!-- 主内容区域 -->
    <div class="main-content">

        <!-- 任务队列面板 -->
        <div class="panel">
            <div class="panel-header">
                <h2>任务队列</h2>
                <span class="badge" id="task-count">0 个任务</span>
            </div>
            <div class="panel-body" style="padding:0; overflow-x:auto;">
                <table class="task-table">
                    <thead>
                        <tr>
                            <th>任务ID</th>
                            <th>用户</th>
                            <th>类型</th>
                            <th>优先级</th>
                            <th>量子比特</th>
                            <th>预计时间</th>
                            <th>状态</th>
                            <th>到达时间</th>
                        </tr>
                    </thead>
                    <tbody id="task-tbody">
                        <!-- 由 JS 动态填充 -->
                    </tbody>
                </table>
                <div id="task-empty" class="empty-state" style="display:none;">
                    暂无任务，请在下方控制面板提交新任务
                </div>
            </div>
        </div>

        <!-- 控制面板 -->
        <div class="panel">
            <div class="panel-header">
                <h2>控制面板</h2>
            </div>
            <div class="panel-body">
                <div class="control-grid">

                    <!-- 左侧：提交新任务 -->
                    <div class="control-section">
                        <h3>提交新任务</h3>
                        <div class="form-group">
                            <label>用户ID</label>
                            <input type="text" id="input-user" value="user_001" placeholder="输入用户ID">
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label>任务类型</label>
                                <select id="input-type">
                                    <option value="quantum">量子任务 (quantum)</option>
                                    <option value="classical">经典任务 (classical)</option>
                                    <option value="hybrid">混合任务 (hybrid)</option>
                                </select>
                            </div>
                            <div class="form-group">
                                <label>优先级 (1-5)</label>
                                <select id="input-priority">
                                    <option value="1">1 - 最低</option>
                                    <option value="2">2 - 低</option>
                                    <option value="3" selected>3 - 中</option>
                                    <option value="4">4 - 高</option>
                                    <option value="5">5 - 最高</option>
                                </select>
                            </div>
                        </div>
                        <div class="form-row">
                            <div class="form-group">
                                <label>量子比特数</label>
                                <input type="number" id="input-qubits" value="10" min="1">
                            </div>
                            <div class="form-group">
                                <label>电路深度</label>
                                <input type="number" id="input-depth" value="100" min="1">
                            </div>
                        </div>
                        <div class="form-group">
                            <label>预计执行时间(秒)</label>
                            <input type="number" id="input-time" value="60" min="0.1" step="0.1">
                        </div>
                        <button class="btn btn-primary" onclick="submitTask()" style="width:100%; margin-top:4px;">
                            提交任务
                        </button>
                    </div>

                    <!-- 右侧：调度策略切换 -->
                    <div class="control-section">
                        <h3>调度策略切换</h3>
                        <p style="font-size:13px; color:#64748b; margin-bottom:16px;">
                            选择当前使用的RL调度策略，切换后将立即生效。
                        </p>
                        <div class="strategy-buttons" id="strategy-buttons">
                            <!-- 由 JS 动态填充 -->
                        </div>
                    </div>

                </div>
            </div>
        </div>

    </div>

    <!-- Toast 通知容器 -->
    <div class="toast-container" id="toast-container"></div>

    <script>
        // ============================================================
        // 全局状态
        // ============================================================
        let ws = null;                // WebSocket 实例
        let currentStatus = {};       // 当前系统状态
        let currentTasks = [];       // 当前任务列表
        let reconnectTimer = null;   // 重连定时器
        let strategyOptions = [];     // 可用策略列表

        // ============================================================
        // 工具函数
        // ============================================================

        /** 显示 Toast 通知 */
        function showToast(message, type) {
            // type: 'success' | 'info' | 'warn'
            var container = document.getElementById('toast-container');
            var toast = document.createElement('div');
            toast.className = 'toast ' + type;
            toast.textContent = message;
            container.appendChild(toast);
            // 3秒后自动移除
            setTimeout(function() {
                if (toast.parentNode) toast.parentNode.removeChild(toast);
            }, 3000);
        }

        /** 格式化时间字符串 */
        function formatTime(isoStr) {
            if (!isoStr) return '-';
            var d = new Date(isoStr);
            var hh = String(d.getHours()).padStart(2, '0');
            var mm = String(d.getMinutes()).padStart(2, '0');
            var ss = String(d.getSeconds()).padStart(2, '0');
            return hh + ':' + mm + ':' + ss;
        }

        /** 获取优先级样式 */
        function priorityClass(p) {
            if (p >= 4) return 'priority-high';
            if (p >= 3) return 'priority-medium';
            return 'priority-low';
        }

        /** 状态中文名 */
        function statusText(s) {
            var map = { pending: '等待中', running: '运行中', completed: '已完成', failed: '失败' };
            return map[s] || s;
        }

        // ============================================================
        // 页面渲染
        // ============================================================

        /** 更新顶部状态卡片 */
        function renderStatus(status) {
            document.getElementById('val-qubit').textContent =
                (status.qubit_utilization * 100).toFixed(1) + '%';
            document.getElementById('val-queue').textContent = status.queue_length;
            document.getElementById('val-wait').textContent = status.average_wait_time.toFixed(1) + 's';
            document.getElementById('val-completed').textContent = status.completed_tasks;
            document.getElementById('val-strategy').textContent = status.current_strategy || '-';
            document.getElementById('val-step').textContent = 'Step: ' + (status.current_step || 0);
        }

        /** 更新任务队列表格 */
        function renderTasks(tasks) {
            var tbody = document.getElementById('task-tbody');
            var empty = document.getElementById('task-empty');
            var countBadge = document.getElementById('task-count');

            countBadge.textContent = tasks.length + ' 个任务';

            if (tasks.length === 0) {
                tbody.innerHTML = '';
                empty.style.display = 'block';
                return;
            }
            empty.style.display = 'none';

            // 按优先级降序、到达时间升序排列
            var sorted = tasks.slice().sort(function(a, b) {
                if (a.status === 'pending' && b.status !== 'pending') return -1;
                if (a.status !== 'pending' && b.status === 'pending') return 1;
                return b.priority - a.priority;
            });

            var html = '';
            for (var i = 0; i < sorted.length; i++) {
                var t = sorted[i];
                html += '<tr>' +
                    '<td style="font-family:monospace;color:#94a3b8;">' + t.task_id + '</td>' +
                    '<td>' + t.user_id + '</td>' +
                    '<td>' + t.task_type + '</td>' +
                    '<td><span class="' + priorityClass(t.priority) + '">' + t.priority + '</span></td>' +
                    '<td>' + (t.qubit_count || '-') + '</td>' +
                    '<td>' + (t.estimated_time || '-') + 's</td>' +
                    '<td><span class="status-tag ' + t.status + '">' + statusText(t.status) + '</span></td>' +
                    '<td style="color:#64748b;">' + formatTime(t.arrival_time) + '</td>' +
                    '</tr>';
            }
            tbody.innerHTML = html;
        }

        /** 渲染策略选择按钮 */
        function renderStrategies(strategies, currentStrategy) {
            var container = document.getElementById('strategy-buttons');
            var html = '';
            for (var i = 0; i < strategies.length; i++) {
                var s = strategies[i];
                var activeClass = (s === currentStrategy) ? ' active' : '';
                html += '<button class="strategy-btn' + activeClass + '" ' +
                    'onclick="switchStrategy(\\'' + s + '\\')">' + s + '</button>';
            }
            container.innerHTML = html;
        }

        // ============================================================
        // API 调用
        // ============================================================

        /** 初始加载：拉取系统状态和任务列表 */
        async function fetchInitialState() {
            try {
                var statusResp = await fetch('/api/status');
                currentStatus = await statusResp.json();

                var tasksResp = await fetch('/api/tasks');
                currentTasks = await tasksResp.json();

                strategyOptions = currentStatus.strategy_options || [];

                renderStatus(currentStatus);
                renderTasks(currentTasks);
                renderStrategies(strategyOptions, currentStatus.current_strategy);
            } catch (e) {
                console.error('初始数据加载失败:', e);
            }
        }

        /** 提交新任务 */
        async function submitTask() {
            var payload = {
                user_id: document.getElementById('input-user').value || 'user_001',
                task_type: document.getElementById('input-type').value,
                priority: parseInt(document.getElementById('input-priority').value),
                qubit_count: parseInt(document.getElementById('input-qubits').value) || 10,
                circuit_depth: parseInt(document.getElementById('input-depth').value) || 100,
                estimated_time: parseFloat(document.getElementById('input-time').value) || 60.0,
            };
            try {
                var resp = await fetch('/api/tasks', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                var result = await resp.json();
                if (result.task_id) {
                    showToast('任务已提交: ' + result.task_id, 'success');
                } else {
                    showToast('提交结果: ' + result.message, 'info');
                }
            } catch (e) {
                showToast('提交失败: ' + e.message, 'warn');
            }
        }

        /** 切换调度策略 */
        async function switchStrategy(strategy) {
            try {
                var resp = await fetch('/api/strategy?strategy=' + encodeURIComponent(strategy), {
                    method: 'POST',
                });
                var result = await resp.json();
                if (result.success) {
                    showToast(result.message, 'success');
                } else {
                    showToast(result.message, 'warn');
                }
            } catch (e) {
                showToast('策略切换失败: ' + e.message, 'warn');
            }
        }

        // ============================================================
        // WebSocket 连接管理
        // ============================================================

        function connectWebSocket() {
            var protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            var wsUrl = protocol + '//' + window.location.host + '/ws';
            ws = new WebSocket(wsUrl);

            // 更新连接状态指示器
            var statusEl = document.getElementById('ws-status');

            ws.onopen = function() {
                statusEl.textContent = 'WebSocket 已连接';
                statusEl.className = 'ws-status connected';
                console.log('WebSocket 已连接');
                // 清除重连定时器
                if (reconnectTimer) {
                    clearTimeout(reconnectTimer);
                    reconnectTimer = null;
                }
            };

            ws.onmessage = function(event) {
                var msg = JSON.parse(event.data);

                if (msg.type === 'init') {
                    // 初始化消息：包含当前状态和任务
                    currentStatus = msg.status;
                    currentTasks = msg.tasks || [];
                    strategyOptions = currentStatus.strategy_options || [];
                    renderStatus(currentStatus);
                    renderTasks(currentTasks);
                    renderStrategies(strategyOptions, currentStatus.current_strategy);

                } else if (msg.type === 'status_update') {
                    // 状态更新
                    if (msg.status) {
                        currentStatus = msg.status;
                        renderStatus(currentStatus);
                    }
                    if (msg.tasks) {
                        currentTasks = msg.tasks;
                        renderTasks(currentTasks);
                    }

                } else if (msg.type === 'task_added') {
                    // 新任务通知
                    if (msg.status) {
                        currentStatus = msg.status;
                        renderStatus(currentStatus);
                    }
                    // 拉取最新任务列表
                    fetch('/api/tasks').then(function(r) {
                        return r.json();
                    }).then(function(tasks) {
                        currentTasks = tasks;
                        renderTasks(currentTasks);
                    });

                } else if (msg.type === 'strategy_changed') {
                    // 策略变更通知
                    if (msg.status) {
                        currentStatus = msg.status;
                        renderStatus(currentStatus);
                        renderStrategies(
                            currentStatus.strategy_options || strategyOptions,
                            currentStatus.current_strategy
                        );
                    }
                    showToast('策略已切换: ' + msg.new_strategy, 'info');

                } else if (msg.type === 'pong') {
                    // 心跳响应，无需处理
                }
            };

            ws.onclose = function() {
                statusEl.textContent = 'WebSocket 已断开';
                statusEl.className = 'ws-status disconnected';
                console.log('WebSocket 已断开，3秒后尝试重连...');
                // 自动重连
                reconnectTimer = setTimeout(function() {
                    connectWebSocket();
                }, 3000);
            };

            ws.onerror = function(err) {
                console.error('WebSocket 错误:', err);
                ws.close();
            };

            // 心跳：每30秒发送一次 ping
            setInterval(function() {
                if (ws && ws.readyState === WebSocket.OPEN) {
                    ws.send(JSON.stringify({ action: 'ping' }));
                }
            }, 30000);
        }

        // ============================================================
        // 页面初始化
        // ============================================================

        (function init() {
            // 先加载初始数据（HTTP 方式）
            fetchInitialState();
            // 然后建立 WebSocket 连接（实时更新）
            connectWebSocket();
        })();
    </script>
</body>
</html>"""


# ============================================================
# 服务器启动入口
# ============================================================

def start_web_server(host: str = "0.0.0.0", port: int = 8000):
    """启动 Web 服务器"""
    import uvicorn
    print(f"========================================")
    print(f"  量子RL调度系统 - 监控面板")
    print(f"  访问地址: http://{host}:{port}")
    print(f"========================================")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_web_server()
