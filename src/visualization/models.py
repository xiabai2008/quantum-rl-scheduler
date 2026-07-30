"""
Pydantic 数据模型定义

定义 Web 可视化接口使用的请求体模型：
- TaskSubmit：提交新任务的请求体
- SystemStatusUpdate：系统状态更新请求体（供调度引擎调用）
"""

from pydantic import BaseModel, Field

from src.visualization.security import MAX_CIRCUIT_QUBITS


class TaskSubmit(BaseModel):
    """提交新任务的请求体"""

    user_id: str = Field(default="user_001", min_length=1, max_length=128, description="用户ID")
    task_type: str = Field(
        default="quantum",
        min_length=1,
        max_length=32,
        description="任务类型: quantum/classical/hybrid",
    )
    priority: int = Field(default=3, ge=1, le=5, description="优先级 1-5")
    # Issue #732: qubit_count 上限与 validate_quantum_circuit 的 MAX_CIRCUIT_QUBITS 保持一致，
    # 统一"任务请求比特数"与"电路实际使用比特数"的语义，避免用户提交后在校验阶段被拒。
    qubit_count: int = Field(
        default=10,
        ge=1,
        le=MAX_CIRCUIT_QUBITS,
        description="所需量子比特数",
    )
    circuit_depth: int = Field(default=100, ge=1, le=10000, description="电路深度")
    estimated_time: float = Field(default=60.0, ge=0.1, le=86400.0, description="预计执行时间(秒)")


class SystemStatusUpdate(BaseModel):
    """系统状态更新请求体（供调度引擎调用）"""

    qubit_utilization: float = Field(default=0.0, ge=0.0, le=1.0)
    queue_length: int = Field(default=0, ge=0, le=100000)
    completed_tasks: int = Field(default=0, ge=0, le=10**9)
    average_wait_time: float = Field(default=0.0, ge=0.0, le=86400.0)


class CircuitSubmit(BaseModel):
    """量子电路提交请求体（Issue #515）。

    用于 POST /api/circuit/submit 端点，提交 QCIS/QASM 格式的
    量子电路内容进行校验与调度。电路内容在路由层经过
    ``validate_quantum_circuit`` 校验后才会被接受。
    """

    circuit: str = Field(
        ...,
        min_length=1,
        max_length=1024 * 1024,
        description="量子电路内容（QCIS 或 OpenQASM 格式）",
    )
    format: str = Field(
        default="qcis",
        description="电路格式: qcis 或 openqasm",
    )
    shots: int = Field(default=1024, ge=1, le=65536, description="测量次数")
    task_name: str = Field(
        default="Web_Submit",
        max_length=128,
        description="任务名称",
    )
