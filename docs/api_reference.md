# 量子RL调度系统 API 接口文档

> **文档版本**: v1.1
> **生成日期**: 2026-07-25
> **适用模块**: `src/api/` 目录下所有公开接口、`src/visualization/routes.py` Web 可视化 API
> **维护状态**: 活跃维护

---

## 目录

1. [概述](#1-概述)
2. [通用规范](#2-通用规范)
3. [TianyanClient 接口](#3-tianyanclient-接口)
4. [TianyanCqlibClient 接口](#4-tianyancqlibclient-接口)
5. [MockClient 接口](#5-mockclient-接口)
6. [Web 可视化 API](#6-web-可视化-api)
7. [CircuitBreaker 接口](#7-circuitbreaker-接口)
8. [异常处理](#8-异常处理)
9. [使用示例](#9-使用示例)
10. [附录](#10-附录)
11. [CqlibRecorder 接口（录制/回放）](#11-cqlibrecorder-接口录制回放)
12. [HardwareAdapter 接口（硬件抽象层）](#12-hardwareadapter-接口硬件抽象层)
13. [QuotaTracker 接口（配额追踪）](#13-quotatracker-接口配额追踪)

---

## 1. 概述

### 1.1 模块职责

`src/api/` 目录封装了与天衍云量子计算平台的交互逻辑，提供统一的接口抽象层，支持：

- **真机模式**：通过 cqlib SDK 连接天衍云超导量子计算机（105数据比特+182耦合比特）
- **Mock 模式**：本地模拟环境，用于开发调试和策略训练
- **熔断保护**：防止 API 故障雪崩，保障系统稳定性

### 1.2 核心文件

| 文件 | 职责 | 代码行数 |
|------|------|---------|
| `tianyan_client.py` | 天衍云 REST API 客户端（Mock 模式） | 633 行 |
| `tianyan_cqlib.py` | 天衍云 cqlib SDK 客户端（真机模式）+ 多机器协调器 | 512 行 |
| `mock_client.py` | Mock API 客户端（开发/测试） | 287 行 |
| `circuit_breaker.py` | 熔断器实现（CLOSED/OPEN/HALF_OPEN 三态） | 156 行 |

### 1.3 设计原则

- **接口一致性**：所有客户端实现相同的 `QuantumAPIClient` 抽象接口
- **故障隔离**：熔断器自动检测 API 故障，防止级联失败
- **可观测性**：内置 Prometheus 指标（请求延迟、成功率、熔断状态）
- **环境切换**：通过环境变量 `TIANYAN_MODE` 控制真机/Mock 模式

---

## 2. 通用规范

### 2.1 认证方式

**真机模式（cqlib）**：
```python
# 通过环境变量配置
export TIANYAN_API_KEY="your_api_key_here"
export TIANYAN_USER_ID="your_user_id"
```

**Mock 模式**：
```python
# 无需认证，本地模拟
export TIANYAN_MODE="mock"
```

**Web 可视化 API（可选认证）**：
```python
# 通过环境变量配置 API 密钥（未配置时认证禁用，所有请求放行）
export VISUALIZATION_API_KEY="your_web_api_key"
# 客户端请求时通过 X-API-Key 请求头传入密钥
# curl -H "X-API-Key: your_web_api_key" http://localhost:8000/api/tasks -X POST ...
```

| 配置项 | 说明 |
|--------|------|
| 环境变量 `VISUALIZATION_API_KEY` | 期望密钥值。未配置（None 或空字符串）时认证禁用，所有请求放行（开发模式） |
| 请求头 `X-API-Key` | 客户端传入的密钥，须与配置值完全匹配，否则返回 401 Unauthorized |

> 仅写操作（POST）端点通过 `verify_api_key` 依赖启用认证；读操作（GET）端点无需认证。

### 2.2 超时配置

| 操作类型 | 默认超时 | 可配置参数 |
|---------|---------|-----------|
| 任务提交 | 30 秒 | `TIANYAN_SUBMIT_TIMEOUT` |
| 结果查询 | 60 秒 | `TIANYAN_QUERY_TIMEOUT` |
| 机器状态 | 10 秒 | `TIANYAN_STATUS_TIMEOUT` |

### 2.3 重试策略

```python
# 默认重试配置
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # 秒
RETRY_BACKOFF = 2.0  # 指数退避因子
```

### 2.4 熔断器阈值

```python
# 熔断器配置
FAILURE_THRESHOLD = 5  # 连续失败次数触发熔断
RECOVERY_TIMEOUT = 60  # 熔断恢复等待时间（秒）
SUCCESS_THRESHOLD = 2  # 半开状态成功次数触发恢复
```

---

## 3. TianyanClient 接口

### 3.1 类定义

```python
class TianyanClient(QuantumAPIClient):
    """
    天衍云 REST API 客户端（Mock 模式）

    通过 HTTP 请求与天衍云平台交互，支持任务提交、结果查询、机器状态监控。
    内置熔断器和重试机制，保障 API 调用稳定性。
    """
```

### 3.2 初始化方法

```python
def __init__(
    self,
    api_key: str | None = None,
    user_id: str | None = None,
    base_url: str = "https://tianyan.ctyun.com/api/v1",
    timeout: float = 30.0,
    max_retries: int = 3,
    circuit_breaker: CircuitBreaker | None = None
) -> None:
    """
    初始化天衍云客户端

    Args:
        api_key: API 密钥（可选，默认从环境变量 TIANYAN_API_KEY 读取）
        user_id: 用户 ID（可选，默认从环境变量 TIANYAN_USER_ID 读取）
        base_url: API 基础 URL（默认：https://tianyan.ctyun.com/api/v1）
        timeout: 请求超时时间（秒，默认：30.0）
        max_retries: 最大重试次数（默认：3）
        circuit_breaker: 熔断器实例（可选，默认创建新实例）

    Raises:
        TianyanAuthError: API 密钥或用户 ID 缺失
        TianyanConnectionError: 网络连接失败

    Example:
        >>> client = TianyanClient(api_key="xxx", user_id="user123")
        >>> task_id = client.submit_task(circuit_qasm, machine_id="tianyan_s")
    """
```

### 3.3 公开方法

#### 3.3.1 submit_task

```python
def submit_task(
    self,
    circuit_qasm: str,
    machine_id: str,
    shots: int = 1000,
    priority: int = 0,
    metadata: dict[str, Any] | None = None
) -> str:
    """
    提交量子任务到天衍云平台

    Args:
        circuit_qasm: QASM 格式的量子电路（字符串）
        machine_id: 目标量子机器 ID（如 "tianyan_s", "tianyan_sw", "tianyan_tn"）
        shots: 测量次数（默认：1000）
        priority: 任务优先级（0=普通，1=高，2=紧急，默认：0）
        metadata: 附加元数据（可选，如任务标签、用户备注）

    Returns:
        task_id: 任务唯一标识符（字符串）

    Raises:
        TianyanSubmissionError: 任务提交失败
        TianyanValidationError: QASM 格式校验失败
        CircuitBreakerOpenError: 熔断器处于开启状态

    Example:
        >>> qasm = '''
        ... OPENQASM 2.0;
        ... include "qelib1.inc";
        ... qreg q[2];
        ... h q[0];
        ... cx q[0], q[1];
        ... measure q -> c;
        ... '''
        >>> task_id = client.submit_task(qasm, "tianyan_s", shots=1024)
        >>> print(f"任务已提交: {task_id}")
    """
```

#### 3.3.2 query_result

```python
def query_result(self, task_id: str) -> dict[str, Any]:
    """
    查询量子任务执行结果

    Args:
        task_id: 任务 ID（由 submit_task 返回）

    Returns:
        result: 包含以下字段的字典：
            - status: 任务状态（"submitted", "running", "completed", "failed"）
            - counts: 测量结果分布（如 {"00": 512, "11": 512}）
            - execution_time: 执行时间（秒）
            - queue_time: 排队时间（秒）
            - fidelity: 量子态保真度（0-1）
            - error_message: 错误信息（仅失败时存在）

    Raises:
        TianyanQueryError: 结果查询失败
        TianyanNotFoundError: 任务 ID 不存在
        CircuitBreakerOpenError: 熔断器处于开启状态

    Example:
        >>> result = client.query_result("task_12345")
        >>> if result["status"] == "completed":
        ...     print(f"测量结果: {result['counts']}")
        ...     print(f"执行时间: {result['execution_time']}s")
    """
```

#### 3.3.3 get_machine_status

```python
def get_machine_status(self, machine_id: str) -> dict[str, Any]:
    """
    获取量子机器实时状态

    Args:
        machine_id: 机器 ID（如 "tianyan_s"）

    Returns:
        status: 包含以下字段的字典：
            - online: 是否在线（布尔值）
            - qubits: 量子比特数（整数）
            - queue_length: 当前队列长度（整数）
            - avg_wait_time: 平均等待时间（秒）
            - last_calibration: 最后校准时间（ISO 格式字符串）
            - fidelity_1q: 单量子比特门保真度
            - fidelity_2q: 双量子比特门保真度
            - t1_time: T1 相干时间（微秒）
            - t2_time: T2 相干时间（微秒）

    Raises:
        TianyanQueryError: 状态查询失败
        TianyanNotFoundError: 机器 ID 不存在
        CircuitBreakerOpenError: 熔断器处于开启状态

    Example:
        >>> status = client.get_machine_status("tianyan_s")
        >>> if status["online"]:
        ...     print(f"机器在线，队列长度: {status['queue_length']}")
        ...     print(f"单比特门保真度: {status['fidelity_1q']:.4f}")
    """
```

#### 3.3.4 cancel_task

```python
def cancel_task(self, task_id: str) -> bool:
    """
    取消已提交的量子任务

    Args:
        task_id: 任务 ID

    Returns:
        success: 取消是否成功（布尔值）

    Raises:
        TianyanCancellationError: 取消操作失败
        TianyanNotFoundError: 任务 ID 不存在
        CircuitBreakerOpenError: 熔断器处于开启状态

    Example:
        >>> success = client.cancel_task("task_12345")
        >>> if success:
        ...     print("任务已取消")
    """
```

#### 3.3.5 list_machines

```python
def list_machines(self) -> list[dict[str, Any]]:
    """
    列出所有可用的量子机器

    Returns:
        machines: 机器列表，每个元素为字典，包含：
            - machine_id: 机器 ID
            - name: 机器名称
            - qubits: 量子比特数
            - online: 是否在线
            - queue_length: 当前队列长度

    Raises:
        TianyanQueryError: 列表查询失败
        CircuitBreakerOpenError: 熔断器处于开启状态

    Example:
        >>> machines = client.list_machines()
        >>> for m in machines:
        ...     print(f"{m['name']}: {m['qubits']} 量子比特，在线={m['online']}")
    """
```

---

## 4. TianyanCqlibClient 接口

### 4.1 类定义

```python
class TianyanCqlibClient(QuantumAPIClient):
    """
    天衍云 cqlib SDK 客户端（真机模式）

    通过 cqlib SDK 直接连接天衍云超导量子计算机，支持：
    - 真机任务提交与结果查询
    - 多机器协调调度
    - 量子启发式退火任务提交（QUBO 问题求解）
    """
```

### 4.2 初始化方法

```python
def __init__(
    self,
    api_key: str | None = None,
    user_id: str | None = None,
    machine_ids: list[str] | None = None,
    timeout: float = 60.0,
    circuit_breaker: CircuitBreaker | None = None
) -> None:
    """
    初始化 cqlib 客户端

    Args:
        api_key: API 密钥（可选，默认从环境变量读取）
        user_id: 用户 ID（可选，默认从环境变量读取）
        machine_ids: 目标机器 ID 列表（默认：["tianyan_s", "tianyan_sw", "tianyan_tn"]）
        timeout: 请求超时时间（秒，默认：60.0）
        circuit_breaker: 熔断器实例（可选）

    Raises:
        TianyanAuthError: 认证信息缺失
        TianyanConnectionError: cqlib SDK 未安装或连接失败

    Example:
        >>> client = TianyanCqlibClient(machine_ids=["tianyan_s", "tianyan_tn"])
        >>> task_id = client.submit_task(qasm, "tianyan_s")
    """
```

### 4.3 公开方法

#### 4.3.1 submit_task

```python
def submit_task(
    self,
    circuit_qasm: str,
    machine_id: str,
    shots: int = 1000,
    priority: int = 0,
    metadata: dict[str, Any] | None = None
) -> str:
    """
    通过 cqlib 提交量子任务（真机模式）

    Args:
        circuit_qasm: QASM 格式的量子电路
        machine_id: 目标机器 ID
        shots: 测量次数（默认：1000）
        priority: 任务优先级（0-2，默认：0）
        metadata: 附加元数据（可选）

    Returns:
        task_id: 任务唯一标识符

    Raises:
        TianyanSubmissionError: 任务提交失败
        TianyanValidationError: QASM 格式校验失败
        CircuitBreakerOpenError: 熔断器开启

    Example:
        >>> task_id = client.submit_task(qasm, "tianyan_s", shots=2048)
    """
```

#### 4.3.2 submit_annealing_task

```python
def submit_annealing_task(
    self,
    qubo_matrix: np.ndarray,
    shots: int = 1000,
    annealing_time: float = 20.0,
    machine_id: str = "tianyan_annealer"
) -> str:
    """
    提交退火任务（QUBO 问题求解）

    Args:
        qubo_matrix: QUBO 矩阵（numpy 二维数组，形状 N×N）
        shots: 退火采样次数（默认：1000）
        annealing_time: 退火时间（微秒，默认：20.0）
        machine_id: 退火器 ID（默认："tianyan_annealer"）

    Returns:
        task_id: 任务唯一标识符

    Raises:
        TianyanSubmissionError: 退火任务提交失败
        TianyanValidationError: QUBO 矩阵格式错误
        CircuitBreakerOpenError: 熔断器开启

    Example:
        >>> import numpy as np
        >>> Q = np.array([[1, -2], [-2, 1]])  # 2x2 QUBO 矩阵
        >>> task_id = client.submit_annealing_task(Q, shots=500)
    """
```

#### 4.3.3 query_result

```python
def query_result(self, task_id: str) -> dict[str, Any]:
    """
    查询 cqlib 任务结果

    Args:
        task_id: 任务 ID

    Returns:
        result: 结果字典（字段同 TianyanClient.query_result）

    Raises:
        TianyanQueryError: 查询失败
        TianyanNotFoundError: 任务不存在
    """
```

#### 4.3.4 get_machine_status

```python
def get_machine_status(self, machine_id: str) -> dict[str, Any]:
    """
    获取 cqlib 机器状态

    Args:
        machine_id: 机器 ID

    Returns:
        status: 状态字典（字段同 TianyanClient.get_machine_status）

    Raises:
        TianyanQueryError: 查询失败
        TianyanNotFoundError: 机器不存在
    """
```

#### 4.3.5 MultiMachineCoordinator（多机器协调器）

```python
class MultiMachineCoordinator:
    """
    多量子机器协调器

    在多台量子机器间智能分配任务，实现负载均衡和最优调度。
    """

    def __init__(
        self,
        clients: dict[str, TianyanCqlibClient],
        strategy: str = "load_balanced"
    ) -> None:
        """
        初始化多机器协调器

        Args:
            clients: 机器 ID 到客户端实例的映射
            strategy: 调度策略（"load_balanced", "round_robin", "priority"）

        Example:
            >>> clients = {
            ...     "tianyan_s": TianyanCqlibClient(machine_ids=["tianyan_s"]),
            ...     "tianyan_tn": TianyanCqlibClient(machine_ids=["tianyan_tn"])
            ... }
            >>> coordinator = MultiMachineCoordinator(clients, strategy="load_balanced")
            >>> task_id = coordinator.submit_task(qasm, shots=1024)
        """

    def submit_task(
        self,
        circuit_qasm: str,
        shots: int = 1000,
        priority: int = 0
    ) -> str:
        """
        智能提交任务到最优机器

        Args:
            circuit_qasm: QASM 量子电路
            shots: 测量次数
            priority: 优先级

        Returns:
            task_id: 任务 ID

        Example:
            >>> task_id = coordinator.submit_task(qasm, shots=2048)
            >>> print(f"任务已提交到最优机器: {task_id}")
        """

    def get_cluster_status(self) -> dict[str, Any]:
        """
        获取集群整体状态

        Returns:
            cluster_status: 包含以下字段：
                - total_machines: 总机器数
                - online_machines: 在线机器数
                - total_queue_length: 总队列长度
                - avg_fidelity: 平均保真度
                - machine_details: 各机器详细状态

        Example:
            >>> status = coordinator.get_cluster_status()
            >>> print(f"在线机器: {status['online_machines']}/{status['total_machines']}")
        """
```

---

## 5. MockClient 接口

### 5.1 类定义

```python
class MockClient(QuantumAPIClient):
    """
    Mock API 客户端（开发/测试模式）

    模拟天衍云 API 行为，用于本地开发、单元测试和策略训练。
    支持可配置的延迟、失败率、机器状态等。
    """
```

### 5.2 初始化方法

```python
def __init__(
    self,
    mock_delay: float = 90.0,
    failure_rate: float = 0.0,
    machine_delays: dict[str, float] | None = None,
    seed: int | None = None
) -> None:
    """
    初始化 Mock 客户端

    Args:
        mock_delay: 默认任务执行延迟（秒，默认：90.0）
        failure_rate: 任务失败率（0.0-1.0，默认：0.0）
        machine_delays: 各机器特定延迟（可选，如 {"tianyan_s": 124.0}）
        seed: 随机种子（可选，用于可重复测试）

    Example:
        >>> mock = MockClient(mock_delay=5.0, failure_rate=0.1, seed=42)
        >>> task_id = mock.submit_task(qasm, "tianyan_s")
    """
```

### 5.3 公开方法

MockClient 实现与 TianyanClient 相同的接口：

- `submit_task(circuit_qasm, machine_id, shots, priority, metadata) -> str`
- `query_result(task_id) -> dict[str, Any]`
- `get_machine_status(machine_id) -> dict[str, Any]`
- `cancel_task(task_id) -> bool`
- `list_machines() -> list[dict[str, Any]]`

**特殊行为**：

- 任务执行延迟可配置（模拟真机延迟）
- 可注入随机失败（测试熔断器）
- 固定随机种子保证可重复性

---

## 6. Web 可视化 API（src/visualization/routes.py）

本节覆盖 Web 可视化监控面板提供的全部 **27 个 HTTP 端点**，源文件为 `src/visualization/routes.py`（约 760 行）。所有端点通过 `APIRouter` 定义并在 `app.py` 中通过 `app.include_router(router)` 注册，路由路径与原 app.py 完全一致，保持向后兼容。

> **权威实验数字**（文档引用须与此一致）：50seed 仿真 PPO=1982.69±857.25 vs FCFS=1648.91±58.34，提升 +20.2%，Welch t 检验，p=7.56e-12，Cohen's d=-2.1353；多 seed 真机 PPO=1736.32±355.78 vs FCFS=383.00±49.13，d=5.33，p<0.001（Bonferroni 校正后显著）。

### 6.1 认证机制

写操作（POST）端点通过 `verify_api_key` 依赖进行可选 API 密钥认证（实现位于 `routes.py` L34-47）：

- **未配置 `VIZ_API_KEY`**：认证禁用，所有请求放行（开发模式）
- **已配置**：请求头 `X-API-Key` 必须与配置值完全匹配，否则返回 `401 Unauthorized`
- 读操作（GET）端点无需认证

### 6.2 页面路由

#### `GET /`

返回监控面板 HTML 页面（Vue3 + Echarts 版本）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/` |
| 认证 | 无 |
| 响应类型 | `text/html`（HTMLResponse） |

**响应**：Vue3 + Echarts 监控面板 HTML 页面。

---

### 6.3 核心监控端点

#### `GET /api/status`

获取当前系统状态（JSON）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/status` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "current_strategy": "PPO",
  "qubit_utilization": 0.65,
  "queue_length": 3,
  "completed_tasks": 128,
  "average_wait_time": 45.2,
  "current_step": 1024,
  "last_update": "2026-07-25T10:30:00",
  "strategy_options": ["PPO", "FCFS", "Random"]
}
```

#### `GET /api/real-machines`

查询天衍云真实量子计算机状态（实时轮询 cqlib）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/real-machines` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "machines": [
    {"id": "tianyan_s", "type": "superconducting", "status": "running", "name": "天衍-S"}
  ],
  "count": 1,
  "source": "cqlib"
}
```

> 无 `TIANYAN_API_KEY` 时返回空列表，`source` 为 `"unavailable"`。

#### `GET /api/real-submissions`

查询最近的真机提交记录（从 `results/real_times.json` 读取）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/real-submissions` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "submissions": [],
  "count": 0
}
```

#### `GET /api/tasks`

获取任务列表，支持按状态过滤。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/tasks` |
| 认证 | 无 |
| Query 参数 | `status`（可选）：`pending` / `running` / `completed`，不传返回全部 |
| 响应类型 | `application/json`（数组） |

**响应示例**：
```json
[
  {
    "task_id": "QTASK-a1b2c3d4",
    "user_id": "user1",
    "task_type": "optimization",
    "status": "pending",
    "priority": 1,
    "qubit_count": 5,
    "circuit_depth": 20,
    "estimated_time": 120.0,
    "arrival_time": "2026-07-25T10:00:00"
  }
]
```

#### `POST /api/tasks`

提交新任务。

| 项目 | 说明 |
|------|------|
| 方法 | POST |
| 路径 | `/api/tasks` |
| 认证 | 是（`X-API-Key`，未配置时放行） |
| 请求体 | `TaskSubmit` JSON |
| 响应类型 | `application/json` |

**请求体（TaskSubmit）**：
```json
{
  "user_id": "user1",
  "task_type": "optimization",
  "priority": 1,
  "qubit_count": 5,
  "circuit_depth": 20,
  "estimated_time": 120.0
}
```

**响应示例**：
```json
{
  "message": "任务提交成功",
  "task_id": "QTASK-a1b2c3d4"
}
```

#### `GET /api/metrics`

返回 Prometheus 格式的自定义指标。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/metrics` |
| 认证 | 无 |
| 响应类型 | `text/plain` |

**响应示例**：
```
# HELP quantum_scheduler_qubit_utilization 量子比特利用率 0~1
# TYPE quantum_scheduler_qubit_utilization gauge
quantum_scheduler_qubit_utilization 0.6500

# HELP quantum_scheduler_queue_length 任务队列长度
# TYPE quantum_scheduler_queue_length gauge
quantum_scheduler_queue_length 3
```

> 包含 qubit_utilization、queue_length、completed_tasks、avg_wait_time、current_step 五项指标。

#### `GET /metrics`

Prometheus 指标端点，供 Prometheus 采集器抓取。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/metrics` |
| 认证 | 无 |
| 响应类型 | `text/plain`（prometheus_client 默认格式） |

**响应**：`prometheus_client` 默认注册表中所有指标的 Prometheus 文本格式输出。

#### `GET /health`

存活探针（Liveness Probe）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/health` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{"status": "alive"}
```

> 只要进程在运行就返回 200，不依赖任何外部资源。

#### `GET /ready`

就绪探针（Readiness Probe）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/ready` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "ready": true,
  "checks": {
    "app": {"ok": true},
    "metrics": {"ok": true},
    "ppo_model": {"ok": true, "required": false},
    "quota_tracker": {"ok": true, "required": false}
  },
  "required_ok": true,
  "timestamp": "2026-07-25T10:30:00"
}
```

> 检查 app 实例、Prometheus 指标、PPO 模型（可选）、配额追踪器（可选）。任一 `required=True` 的检查不可用返回 503。

---

### 6.4 策略控制端点

#### `POST /api/strategy`

切换调度策略。

| 项目 | 说明 |
|------|------|
| 方法 | POST |
| 路径 | `/api/strategy` |
| 认证 | 是（`X-API-Key`，未配置时放行） |
| Query 参数 | `strategy`（必填）：策略名称，须为 `strategy_options` 中的有效值 |
| 响应类型 | `application/json` |

**响应示例（成功）**：
```json
{
  "message": "策略切换: FCFS -> PPO",
  "success": true
}
```

**响应示例（失败）**：
```json
{
  "message": "未知策略: UnknownStrategy",
  "success": false
}
```

#### `POST /api/update`

更新系统状态（供调度引擎调用）。

| 项目 | 说明 |
|------|------|
| 方法 | POST |
| 路径 | `/api/update` |
| 认证 | 是（`X-API-Key`，未配置时放行） |
| 请求体 | `SystemStatusUpdate` JSON |
| 响应类型 | `application/json` |

**请求体（SystemStatusUpdate）**：
```json
{
  "qubit_utilization": 0.72,
  "queue_length": 5,
  "completed_tasks": 130,
  "average_wait_time": 42.0
}
```

**响应示例**：
```json
{
  "message": "状态更新成功",
  "status": {"qubit_utilization": 0.72, "queue_length": 5, "...": "..."}
}
```

---

### 6.5 PPO 数据端点

#### `GET /api/ppo/comparison`

返回 PPO 与其他策略的对比数据（从 `results/` 目录最新的 `simulation_results_*.json` 读取）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/ppo/comparison` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "strategies": [
    {
      "rank": 1,
      "name": "PPO",
      "avg_reward": 1982.69,
      "avg_wait_time": 45.2,
      "completion_rate": 0.95,
      "qubit_utilization": 0.78,
      "classical_utilization": 0.65
    }
  ],
  "ppo_rank": 1,
  "total_strategies": 4,
  "data_source": "simulation_results_20260725.json"
}
```

> 未找到仿真结果文件时返回 `{"error": "未找到仿真结果文件", "strategies": [], "ppo_rank": null}`。

#### `GET /api/ppo/predict`

使用 PPO 模型对当前环境状态进行一次推理预测。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/ppo/predict` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "action": 1,
  "action_name": "量子资源",
  "observation": [0.65, 3.0, 128, 45.2, 1024],
  "model_type": "PPO"
}
```

> 动作映射：0=经典资源，1=量子资源，2=混合执行。模型未加载时返回 `{"error": "PPO 模型未加载", "action": null, "confidence": 0}`。

#### `GET /api/ppo/stats`

返回 PPO 关键性能指标。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/ppo/stats` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "ppo": {
    "reward": 1982.69,
    "wait_time": 45.2,
    "completion_rate": 0.95,
    "qubit_util": 0.78,
    "classical_util": 0.65
  },
  "ppo_rank": 1,
  "total": 4,
  "best_strategy": "PPO",
  "best_reward": 1982.69,
  "vs_random": 1200.5
}
```

---

### 6.6 资源与决策端点

#### `GET /api/quota`

获取天衍云真机配额使用状态（Issue #103）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/quota` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "available": true,
  "total": 1000,
  "used": 350,
  "remaining": 650,
  "usage_ratio": 0.35,
  "alert_level": "normal"
}
```

> 配额追踪未启用时返回 `{"available": false, "message": "配额追踪未启用"}`。

#### `GET /api/resource-history`

获取资源利用率历史趋势数据（最近 100 个数据点，Issue #22）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/resource-history` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "history": [
    {
      "step": 1,
      "qubit_utilization": 0.65,
      "queue_length": 3,
      "completed_tasks": 128,
      "average_wait_time": 45.2
    }
  ]
}
```

> 数据来源：后台 `simulate_scheduler` 每 3 秒采集一次。

#### `GET /api/decision-log`

获取调度决策日志（最近 200 条，Issue #22）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/decision-log` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "decisions": [
    {
      "step": 1,
      "task_id": "QTASK-a1b2c3d4",
      "action": 1,
      "action_label": "量子资源",
      "reward": 12.5,
      "source": "PPO"
    }
  ]
}
```

#### `GET /api/machines-comparison`

获取多机器对比数据（雷达图和对比表格，Issue #22）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/machines-comparison` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "machines": [
    {
      "name": "天衍-S",
      "total_qubits": 287,
      "available_ratio": 0.95,
      "fidelity": 0.998,
      "queue_depth": 3,
      "status": "running",
      "single_gate_fidelity": 0.999,
      "two_gate_fidelity": 0.995
    }
  ]
}
```

#### `GET /api/tenants`

获取多租户配额状态（Issue #97）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/tenants` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "tenants": [
    {
      "tenant_id": "tenant_1",
      "name": "研发组"
    }
  ]
}
```

> 租户状态查询失败时返回 `{"tenants": []}`。

---

### 6.7 决策可解释性端点（Day4-7 新增）

#### `GET /api/explainability`

获取最近决策的特征贡献度摘要（Issue #73）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/explainability` |
| 认证 | 无 |
| Query 参数 | `limit`（可选，默认 20，最大 200）：返回最近多少条决策 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "decisions": [
    {
      "step": 1,
      "action": 1,
      "action_label": "量子资源",
      "feature_contributions": {"queue_length": 0.35, "qubit_util": 0.28},
      "explanation_text": "队列较长，优先量子资源"
    }
  ],
  "count": 1
}
```

#### `GET /api/explainability/summary`

获取当前会话的全局特征重要性排名（Issue #73）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/explainability/summary` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "feature_importance": [
    {"feature": "queue_length", "importance": 0.35},
    {"feature": "qubit_util", "importance": 0.28}
  ],
  "total_decisions": 150
}
```

> 聚合所有包含特征贡献度的决策记录，计算各特征的平均贡献度，降序排列。无记录时返回 `{"feature_importance": [], "total_decisions": 0}`。

#### `GET /api/explainability/latest`

获取最新一条决策的完整可解释性数据（决策放大镜，Day2-3-10）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/explainability/latest` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "empty": false,
  "latest": {
    "step": 200,
    "action": 1,
    "action_label": "量子资源",
    "feature_contributions": {"queue_length": 0.35},
    "explanation_text": "..."
  }
}
```

> 无记录时返回 `{"empty": true, "latest": null}`。

---

### 6.8 PPO vs FCFS 实时对战面板端点（Day4-7 新增）

#### `POST /api/battle/start`

启动 PPO vs FCFS 对战（Day4-7-11）。

| 项目 | 说明 |
|------|------|
| 方法 | POST |
| 路径 | `/api/battle/start` |
| 认证 | 是（`X-API-Key`，未配置时放行） |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "success": true,
  "message": "对战已启动",
  "step": 0,
  "ppo_obs": [0.5, 0, 0, 0, 0],
  "fcfs_obs": [0.5, 0, 0, 0, 0]
}
```

> 初始化两个独立的调度环境实例（相同 seed=42 确保公平对比），分别使用 PPO 和 FCFS 策略。启动失败时返回 `{"success": false, "error": "..."}`。

#### `POST /api/battle/step`

推进对战一步（Day4-7-11）。

| 项目 | 说明 |
|------|------|
| 方法 | POST |
| 路径 | `/api/battle/step` |
| 认证 | 是（`X-API-Key`，未配置时放行） |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "step": 1,
  "ppo": {
    "step": 1,
    "reward": 12.5,
    "cumulative": 12.5,
    "action": 1,
    "util": 0.65
  },
  "fcfs": {
    "step": 1,
    "reward": 8.2,
    "cumulative": 8.2,
    "action": 0,
    "util": 0.65
  },
  "ppo_total": 12.5,
  "fcfs_total": 8.2,
  "gap": 4.3
}
```

> PPO 使用模型预测动作，FCFS 使用固定策略（始终选择动作 0=经典资源）。对战未启动时返回 `{"error": "对战未启动，请先调用 /api/battle/start"}`。

#### `GET /api/battle/status`

获取对战当前状态（Day4-7-11）。

| 项目 | 说明 |
|------|------|
| 方法 | GET |
| 路径 | `/api/battle/status` |
| 认证 | 无 |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "running": true,
  "step": 50,
  "ppo_total": 625.0,
  "fcfs_total": 410.0,
  "gap": 215.0,
  "ppo_history": [],
  "fcfs_history": []
}
```

> 返回最近 50 条历史数据。

#### `POST /api/battle/reset`

重置对战状态（Day4-7-11）。

| 项目 | 说明 |
|------|------|
| 方法 | POST |
| 路径 | `/api/battle/reset` |
| 认证 | 是（`X-API-Key`，未配置时放行） |
| 响应类型 | `application/json` |

**响应示例**：
```json
{
  "success": true,
  "message": "对战已重置"
}
```

---

## 7. CircuitBreaker 接口

### 7.1 类定义

```python
class CircuitBreaker:
    """
    熔断器实现（三态模型）

    状态转换：
    - CLOSED（关闭）: 正常状态，请求通过
    - OPEN（开启）: 熔断状态，请求被拒绝
    - HALF_OPEN（半开）: 恢复探测状态，允许少量请求试探
    """
```

### 7.2 初始化方法

```python
def __init__(
    self,
    failure_threshold: int = 5,
    recovery_timeout: float = 60.0,
    success_threshold: int = 2
) -> None:
    """
    初始化熔断器

    Args:
        failure_threshold: 触发熔断的连续失败次数（默认：5）
        recovery_timeout: 熔断恢复等待时间（秒，默认：60.0）
        success_threshold: 半开状态成功次数阈值（默认：2）

    Example:
        >>> breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
    """
```

### 7.3 公开方法

#### 7.3.1 call

```python
def call(self, func: Callable, *args, **kwargs) -> Any:
    """
    通过熔断器调用函数

    Args:
        func: 要调用的函数
        *args: 位置参数
        **kwargs: 关键字参数

    Returns:
        result: 函数返回值

    Raises:
        CircuitBreakerOpenError: 熔断器处于开启状态
        Exception: 被调用函数抛出的异常

    Example:
        >>> breaker = CircuitBreaker()
        >>> try:
        ...     result = breaker.call(client.submit_task, qasm, "tianyan_s")
        ... except CircuitBreakerOpenError:
        ...     print("熔断器开启，请求被拒绝")
    """
```

#### 7.3.2 get_state

```python
def get_state(self) -> str:
    """
    获取熔断器当前状态

    Returns:
        state: 状态字符串（"CLOSED", "OPEN", "HALF_OPEN"）

    Example:
        >>> state = breaker.get_state()
        >>> print(f"熔断器状态: {state}")
    """
```

#### 7.3.3 reset

```python
def reset(self) -> None:
    """
    手动重置熔断器到 CLOSED 状态

    Example:
        >>> breaker.reset()
        >>> print("熔断器已重置")
    """
```

---

## 8. 异常处理

### 8.1 异常层次结构

```python
QuantumSchedulerError (基类)
├── TianyanAPIError (API 错误基类)
├── CircuitOpenError (熔断器开启)
├── ConfigurationError (配置错误，不可重试)
├── TaskParseError (任务解析错误)
├── SchedulingError (调度错误)
├── QuantumAnnealingError (量子启发式退火错误)
├── ResourceExhaustedError (资源耗尽错误)
└── RateLimitError (API 限流错误，含 retry_after 属性，默认可重试)
```

### 8.2 异常属性

所有异常继承自 `QuantumSchedulerError`，使用关键字参数传递 code 和 retryable：

```python
class QuantumSchedulerError(Exception):
    def __init__(self, message: str, *, code: str = "UNKNOWN", retryable: bool = False) -> None:
        """
        Args:
            message: 错误描述
            code: 错误代码（关键字参数，默认 "UNKNOWN"）
            retryable: 是否可重试（关键字参数，默认 False）
        """
        self.code = code
        self.retryable = retryable
        super().__init__(message)


class RateLimitError(QuantumSchedulerError):
    """API 限流错误，默认可重试，携带 retry_after 属性"""
    def __init__(
        self,
        message: str,
        *,
        code: str = "RATE_LIMIT",
        retryable: bool = True,
        retry_after: float | None = None,
    ) -> None:
        self.retry_after = retry_after
        super().__init__(message, code=code, retryable=retryable)
```

### 8.3 异常处理示例

```python
from src.exceptions import (
    TianyanAPIError,
    CircuitOpenError,
    ConfigurationError,
    RateLimitError,
    ResourceExhaustedError,
)

try:
    task_id = client.submit_task(qasm, "tianyan_s")
except ConfigurationError as e:
    print(f"配置错误: {e}")
    print(f"错误码: {e.code}, 可重试: {e.retryable}")
    # 检查 API 密钥配置等
except TianyanAPIError as e:
    if e.retryable:
        print(f"API 调用失败，可重试: {e}")
        # 执行重试逻辑
    else:
        print(f"API 调用失败，不可重试: {e}")
        # 记录错误，通知用户
except RateLimitError as e:
    wait = e.retry_after if e.retry_after else 1.0
    print(f"触发限流，等待 {wait} 秒后重试")
    # 等待后重试
except CircuitOpenError:
    print("熔断器开启，API 暂时不可用")
    # 降级到 Mock 模式或排队等待
except ResourceExhaustedError:
    print("资源耗尽，无法接受新任务")
    # 排队等待或返回错误
```

---

## 9. 使用示例

### 9.1 基础使用（Mock 模式）

```python
from src.api.mock_client import MockClient

# 初始化 Mock 客户端
client = MockClient(mock_delay=5.0, seed=42)

# 提交任务
qasm = '''
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
h q[0];
cx q[0], q[1];
measure q -> c;
'''
task_id = client.submit_task(qasm, "tianyan_s", shots=1024)
print(f"任务已提交: {task_id}")

# 查询结果
result = client.query_result(task_id)
if result["status"] == "completed":
    print(f"测量结果: {result['counts']}")
    print(f"执行时间: {result['execution_time']}s")
```

### 9.2 真机使用（cqlib 模式）

```python
import os
from src.api.tianyan_cqlib import TianyanCqlibClient

# 配置环境变量
os.environ["TIANYAN_API_KEY"] = "your_api_key"
os.environ["TIANYAN_USER_ID"] = "your_user_id"

# 初始化 cqlib 客户端
client = TianyanCqlibClient(machine_ids=["tianyan_s", "tianyan_tn"])

# 提交任务
task_id = client.submit_task(qasm, "tianyan_s", shots=2048)

# 轮询结果
import time
while True:
    result = client.query_result(task_id)
    if result["status"] == "completed":
        print(f"任务完成: {result['counts']}")
        break
    elif result["status"] == "failed":
        print(f"任务失败: {result['error_message']}")
        break
    time.sleep(5)  # 等待 5 秒后重试
```

### 9.3 多机器协调

```python
from src.api.tianyan_cqlib import TianyanCqlibClient, MultiMachineCoordinator

# 创建多个客户端
clients = {
    "tianyan_s": TianyanCqlibClient(machine_ids=["tianyan_s"]),
    "tianyan_sw": TianyanCqlibClient(machine_ids=["tianyan_sw"]),
    "tianyan_tn": TianyanCqlibClient(machine_ids=["tianyan_tn"])
}

# 初始化协调器（负载均衡策略）
coordinator = MultiMachineCoordinator(clients, strategy="load_balanced")

# 智能提交任务（自动选择最优机器）
task_id = coordinator.submit_task(qasm, shots=1024)
print(f"任务已提交到最优机器: {task_id}")

# 查看集群状态
cluster_status = coordinator.get_cluster_status()
print(f"在线机器: {cluster_status['online_machines']}/{cluster_status['total_machines']}")
```

### 9.4 熔断器集成

```python
from src.api.circuit_breaker import CircuitBreaker
from src.api.tianyan_client import TianyanClient

# 创建熔断器
breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

# 创建客户端（注入熔断器）
client = TianyanClient(circuit_breaker=breaker)

# 通过熔断器调用
try:
    task_id = breaker.call(client.submit_task, qasm, "tianyan_s")
except CircuitBreakerOpenError:
    print("API 暂时不可用，降级到 Mock 模式")
    mock_client = MockClient()
    task_id = mock_client.submit_task(qasm, "tianyan_s")
```

---

## 10. 附录

### 10.1 Prometheus 指标

API 层暴露以下 Prometheus 指标：

| 指标名称 | 类型 | 说明 |
|---------|------|------|
| `tianyan_api_requests_total` | Counter | API 请求总数（按方法、状态分组） |
| `tianyan_api_request_duration_seconds` | Histogram | API 请求延迟分布 |
| `tianyan_api_success_rate` | Gauge | API 成功率（滑动窗口） |
| `circuit_breaker_state` | Gauge | 熔断器状态（0=CLOSED, 1=OPEN, 2=HALF_OPEN） |
| `circuit_breaker_failures_total` | Counter | 熔断器失败计数 |

### 10.2 环境变量参考

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `TIANYAN_MODE` | 运行模式（"real" / "mock"） | "mock" |
| `TIANYAN_API_KEY` | 天衍云 API 密钥 | - |
| `TIANYAN_USER_ID` | 天衍云用户 ID | - |
| `TIANYAN_SUBMIT_TIMEOUT` | 任务提交超时（秒） | 30.0 |
| `TIANYAN_QUERY_TIMEOUT` | 结果查询超时（秒） | 60.0 |
| `TIANYAN_STATUS_TIMEOUT` | 状态查询超时（秒） | 10.0 |

### 10.3 机器 ID 参考

| 机器 ID | 名称 | 物理比特数（数据+耦合） | 类型 |
|---------|------|-----------|------|
| `tianyan_s` | 天衍-S | 287（105+182） | 超导量子计算机 |
| `tianyan_sw` | 天衍-SW | 287（105+182） | 超导量子计算机 |
| `tianyan_tn` | 天衍-TN | 287（105+182） | 超导量子计算机 |
| `tianyan_annealer` | 天衍退火器 | - | 退火求解器（仿真模拟退火） |

### 10.4 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|---------|
| v1.0 | 2026-07-02 | 初始版本，覆盖所有公开接口 |
| v1.1 | 2026-07-25 | 新增「Web 可视化 API」章节，覆盖 routes.py 全部 27 个端点；补充 VISUALIZATION_API_KEY / X-API-Key 认证机制说明 |
| v1.2 | 2026-07-29 | 补充 CqlibRecorder（录制/回放）、HardwareAdapter（硬件抽象层）、QuotaTracker（配额追踪）三个 API 模块文档 |

---

## 11. CqlibRecorder 接口（录制/回放）

### 11.1 模块概述

`cqlib_recorder.py` 实现了 cqlib SDK 的「录制-回放」测试框架（Issue #175），包含两个客户端：

- **CqlibReplayClient**：从 JSON fixtures 加载响应，无需安装 cqlib SDK 即可运行真机交互逻辑的回归测试
- **CqlibRecordingClient**：包装真实 `CqlibTianyanClient`，在调用真机 API 的同时将响应序列化为 JSON fixtures 供后续回放

### 11.2 CqlibReplayClient 类

#### 11.2.1 类定义

```python
class CqlibReplayClient:
    """
    cqlib 回放客户端：从 JSON fixtures 加载响应，替代真实 SDK 调用。

    作为 CqlibTianyanClient 的回放替代品，无需安装 cqlib SDK 即可运行
    真机交互逻辑的回归测试。所有响应来自 fixtures_dir 下的 JSON 文件。

    任务状态采用轮询计数状态机模拟真实行为：
    - 首次查询某 task_id 返回 running 状态
    - 后续查询返回 completed 状态
    """
```

#### 11.2.2 初始化方法

```python
def __init__(
    self,
    fixtures_dir: str = "tests/fixtures/cqlib_responses",
    *,
    machine_name: str = "tianyan_s",
    auto_retry_machine: bool = True,
    error_mode: str | None = None,
) -> None:
    """
    初始化回放客户端并加载全部 fixtures。

    Args:
        fixtures_dir: JSON fixtures 目录路径（默认："tests/fixtures/cqlib_responses"）
        machine_name: 默认机器名（用于可用性判断与重试逻辑，默认："tianyan_s"）
        auto_retry_machine: 机器不可用时是否自动切换备用机（默认：True）
        error_mode: 错误注入模式（仅用于测试）：
            - "capacity"：模拟机时包容量不足
            - "unavailable"：模拟所有机器不可用
            - None：正常回放模式（默认）

    Raises:
        FileNotFoundError: fixture 文件不存在
        json.JSONDecodeError: JSON 格式错误

    Example:
        >>> replay = CqlibReplayClient("tests/fixtures/cqlib_responses")
        >>> assert replay.authenticate() is True
    """
```

#### 11.2.3 公开方法

#### 11.2.3.1 authenticate

```python
def authenticate(self) -> bool:
    """
    回放认证（总是返回 True）。

    Returns:
        始终返回 True，模拟认证成功

    Example:
        >>> result = replay.authenticate()
        >>> print(result)  # True
    """
```

#### 11.2.3.2 list_backends

```python
def list_backends(self) -> list[dict[str, Any]]:
    """
    从 machine_list.json 加载量子计算机列表。

    Returns:
        backends: 后端字典列表，每项含 id/type/status/name 字段

    Example:
        >>> backends = replay.list_backends()
        >>> for b in backends:
        ...     print(f"{b['name']}: {b['status']}")
    """
```

#### 11.2.3.3 get_backend_info

```python
def get_backend_info(self, backend_name: str | None = None) -> dict[str, Any]:
    """
    获取指定后端信息。

    Args:
        backend_name: 后端名称，为 None 时使用默认机器名

    Returns:
        info: 匹配的后端字典；未找到返回空字典

    Example:
        >>> info = replay.get_backend_info("tianyan_s")
        >>> print(info.get("status"))
    """
```

#### 11.2.3.4 submit_quantum_task

```python
def submit_quantum_task(
    self,
    qcis: str = "",
    circuit: Any = None,
    shots: int = 1024,
    task_name: str = "Scheduler_Task",
) -> str | None:
    """
    从 task_submit.json 加载任务提交响应。

    模拟真实提交逻辑：预检机器可用性、容量错误处理、备用机切换。
    当 error_mode="capacity" 时直接返回 None（模拟机时包容量不足）。

    Args:
        qcis: QCIS 指令字符串
        circuit: cqlib.Circuit 对象（与 qcis 二选一，回放中不使用）
        shots: 测量次数（默认：1024）
        task_name: 任务名称（默认："Scheduler_Task"）

    Returns:
        task_id: 任务 ID 字符串；机器不可用或容量不足时返回 None

    Example:
        >>> tid = replay.submit_quantum_task(qcis="H Q0\\nM Q0", shots=1024)
        >>> print(f"任务ID: {tid}")
    """
```

#### 11.2.3.5 get_task_status

```python
def get_task_status(self, task_id: str) -> dict[str, Any]:
    """
    根据 task_id 轮询计数返回 running 或 completed 状态。

    状态机：首次查询返回 running，后续查询返回 completed。
    模拟真实场景中任务从运行到完成的自然流转。

    Args:
        task_id: 任务 ID

    Returns:
        status: 状态字典，含 task_id/status/result/raw 字段

    Example:
        >>> status = replay.get_task_status("task_123")
        >>> print(status["status"])  # 首次 "running"，再次 "completed"
    """
```

#### 11.2.3.6 get_task_result

```python
def get_task_result(self, task_id: str) -> dict[str, Any]:
    """
    获取任务执行结果（从 task_result.json 加载）。

    Args:
        task_id: 任务 ID

    Returns:
        result: 结果字典，含 task_id/status/result/raw 字段

    Example:
        >>> result = replay.get_task_result("task_123")
        >>> print(result["status"])
    """
```

#### 11.2.3.7 wait_for_task

```python
def wait_for_task(
    self, task_id: str, timeout: int = 300, poll_interval: int = 5
) -> dict[str, Any]:
    """
    轮询等待任务完成并返回结果。

    回放中通过 get_task_status 的轮询计数状态机实现：
    首次查询返回 running，第二次返回 completed 即结束等待。

    Args:
        task_id: 任务 ID
        timeout: 超时秒数（默认：300）
        poll_interval: 轮询间隔秒数（默认：5）

    Returns:
        result: 完成状态字典；超时返回 {"task_id": ..., "status": "timeout"}

    Example:
        >>> result = replay.wait_for_task("task_123", timeout=60, poll_interval=1)
        >>> print(result["status"])
    """
```

#### 11.2.3.8 get_queue_status

```python
def get_queue_status(self) -> dict[str, Any]:
    """
    获取队列状态（基于 fixtures 机器列表统计）。

    Returns:
        status: 含 total_machines/running/available 字段的字典

    Example:
        >>> qs = replay.get_queue_status()
        >>> print(f"在线机器: {qs['running']}/{qs['total_machines']}")
    """
```

#### 11.2.3.9 is_available

```python
def is_available(self) -> bool:
    """
    检查真机是否可用（回放中认证通过且默认机器 running）。

    Returns:
        True 表示可提交，False 表示应降级

    Example:
        >>> if replay.is_available():
        ...     tid = replay.submit_quantum_task(qcis="H Q0\\nM Q0")
    """
```

#### 11.2.3.10 submit_and_get_task_id

```python
def submit_and_get_task_id(
    self,
    qcis: str,
    shots: int = 512,
    task_name: str = "Scheduler_Real_Task",
) -> str | None:
    """
    提交量子任务并立即返回 task_id（非阻塞，语义化别名）。

    Args:
        qcis: QCIS 指令字符串
        shots: 测量次数（默认：512）
        task_name: 任务名称（默认："Scheduler_Real_Task"）

    Returns:
        task_id: 任务 ID 字符串；提交失败返回 None

    Example:
        >>> tid = replay.submit_and_get_task_id("H Q0\\nM Q0", shots=1024)
    """
```

### 11.3 CqlibRecordingClient 类

#### 11.3.1 类定义

```python
class CqlibRecordingClient:
    """
    cqlib 录制客户端：包装真实 CqlibTianyanClient 并录制响应。

    所有方法调用转发给真实客户端，同时将响应序列化为 JSON fixtures
    保存到 fixtures_dir，供 CqlibReplayClient 后续回放使用。

    录制时自动处理不可 JSON 序列化的对象（如 cqlib 自定义类型），
    将其递归转为字符串表示。
    """
```

#### 11.3.2 初始化方法

```python
def __init__(self, real_client: CqlibTianyanClient, fixtures_dir: str) -> None:
    """
    初始化录制客户端。

    Args:
        real_client: 真实 CqlibTianyanClient 实例（需已配置凭证）
        fixtures_dir: JSON fixtures 输出目录（不存在则自动创建）

    Example:
        >>> real = CqlibTianyanClient(login_key="xxx", machine_name="tianyan_s")
        >>> recorder = CqlibRecordingClient(real, "tests/fixtures/cqlib_responses")
    """
```

#### 11.3.3 公开方法

#### 11.3.3.1 authenticate

```python
def authenticate(self) -> bool:
    """
    录制认证结果并保存到 authenticate.json。

    Returns:
        result: 真实客户端的认证结果（布尔值）

    Example:
        >>> result = recorder.authenticate()
    """
```

#### 11.3.3.2 list_backends

```python
def list_backends(self) -> list[dict[str, Any]]:
    """
    录制量子计算机列表响应并保存到 machine_list.json。

    Returns:
        backends: 真实客户端返回的后端字典列表

    Example:
        >>> backends = recorder.list_backends()
    """
```

#### 11.3.3.3 get_backend_info

```python
def get_backend_info(self, backend_name: str | None = None) -> dict[str, Any]:
    """
    获取后端信息（转发给真实客户端，不单独保存 fixture）。

    Args:
        backend_name: 后端名称，为 None 时使用默认机器名

    Returns:
        info: 真实客户端返回的后端信息字典
    """
```

#### 11.3.3.4 submit_quantum_task

```python
def submit_quantum_task(
    self,
    qcis: str = "",
    circuit: Any = None,
    shots: int = 1024,
    task_name: str = "Scheduler_Task",
) -> str | None:
    """
    录制任务提交响应并保存到 task_submit.json。

    Args:
        qcis: QCIS 指令字符串
        circuit: cqlib.Circuit 对象（与 qcis 二选一）
        shots: 测量次数（默认：1024）
        task_name: 任务名称（默认："Scheduler_Task"）

    Returns:
        task_id: 真实客户端返回的任务 ID；失败返回 None

    Example:
        >>> tid = recorder.submit_quantum_task(qcis="H Q0\\nM Q0", shots=1024)
    """
```

#### 11.3.3.5 get_task_status

```python
def get_task_status(self, task_id: str) -> TaskResult:
    """
    录制任务状态查询响应。

    根据 status 自动保存到 task_status_running.json 或 task_status_completed.json。

    Args:
        task_id: 任务 ID

    Returns:
        status: 真实客户端返回的状态字典

    Example:
        >>> status = recorder.get_task_status("task_123")
    """
```

#### 11.3.3.6 get_task_result

```python
def get_task_result(self, task_id: str) -> TaskResult:
    """
    录制任务结果查询响应并保存到 task_result.json。

    Args:
        task_id: 任务 ID

    Returns:
        result: 真实客户端返回的结果字典

    Example:
        >>> result = recorder.get_task_result("task_123")
    """
```

#### 11.3.3.7 wait_for_task

```python
def wait_for_task(self, task_id: str, timeout: int = 300, poll_interval: int = 5) -> TaskResult:
    """
    录制等待任务完成的最终结果。

    轮询过程中的中间状态由 get_task_status 录制，
    此方法额外录制最终完成结果到 task_status_completed.json。

    Args:
        task_id: 任务 ID
        timeout: 超时秒数（默认：300）
        poll_interval: 轮询间隔秒数（默认：5）

    Returns:
        result: 最终状态字典

    Example:
        >>> result = recorder.wait_for_task("task_123")
        >>> print(result["status"])
    """
```

#### 11.3.4 Fixture 文件清单

录制客户端生成以下 JSON fixture 文件，供回放客户端加载：

| 文件名 | 来源方法 | 说明 |
|--------|---------|------|
| `authenticate.json` | `authenticate()` | 认证结果 |
| `machine_list.json` | `list_backends()` | 量子计算机列表 |
| `task_submit.json` | `submit_quantum_task()` | 任务提交响应 |
| `task_status_running.json` | `get_task_status()` | 任务运行中状态 |
| `task_status_completed.json` | `get_task_status()` / `wait_for_task()` | 任务完成状态 |
| `task_result.json` | `get_task_result()` | 任务执行结果 |

#### 11.3.5 典型工作流示例

```python
from src.api.cqlib_recorder import CqlibRecordingClient, CqlibReplayClient
from src.api.tianyan_cqlib import CqlibTianyanClient

# ============ 录制阶段（需真机凭证 + cqlib SDK） ============
real = CqlibTianyanClient(login_key="xxx", machine_name="tianyan_s")
recorder = CqlibRecordingClient(real, "tests/fixtures/cqlib_responses")
recorder.list_backends()
tid = recorder.submit_quantum_task(qcis="H Q0\nM Q0", shots=1024)
recorder.wait_for_task(tid)

# ============ 回放阶段（无需 cqlib SDK，CI 可直接运行） ============
replay = CqlibReplayClient("tests/fixtures/cqlib_responses")
assert replay.authenticate() is True
backends = replay.list_backends()
tid = replay.submit_quantum_task(qcis="H Q0\nM Q0", shots=1024)
result = replay.wait_for_task(tid)
assert result["status"] == "completed"
```

---

## 12. HardwareAdapter 接口（硬件抽象层）

### 12.1 模块概述

`hardware_adapter.py` 定义了统一的量子硬件后端抽象层（Issue #256/#258/#259），使超导、离子阱、光量子等不同硬件路线可以通过同一接口接入调度系统。

**当前实现状态**：

| 硬件类型 | 实现类 | 状态 |
|---------|--------|------|
| 超导（superconducting） | `CqlibTianyanClient` | 已完成真机验证 |
| 离子阱（ion_trap） | `IonTrapBackend` | 桩实现，待接入真实平台 |
| 光量子（photonic） | `PhotonicBackend` | 桩实现，待接入真实平台 |

### 12.2 CircuitFormat 枚举

```python
class CircuitFormat(Enum):
    """
    量子电路格式枚举（Issue #256）。

    不同硬件平台原生支持的电路描述格式不同，
    本枚举用于在提交电路时标注格式类型。
    """
```

| 枚举值 | 字符串 | 说明 |
|--------|--------|------|
| `CircuitFormat.QCIS` | `"qcis"` | 天衍云 QCIS 指令格式（超导） |
| `CircuitFormat.OPENQASM` | `"openqasm"` | OpenQASM 2.0/3.0 格式（跨平台通用） |
| `CircuitFormat.IONQ_JSON` | `"ionq_json"` | IonQ JSON 格式（离子阱） |
| `CircuitFormat.PHOTONIC_HAMILTONIAN` | `"photonic_hamiltonian"` | 光量子哈密顿量描述格式 |
| `CircuitFormat.QISKIT_CIRCUIT` | `"qiskit_circuit"` | Qiskit Circuit 对象（内存对象，非文本格式） |

### 12.3 QuantumHardwareBackend 抽象基类

#### 12.3.1 类定义

```python
class QuantumHardwareBackend(ABC):
    """
    量子硬件后端抽象基类（Issue #256）。

    所有具体硬件后端（超导/离子阱/光量子）继承本类并实现抽象接口，
    使调度系统可以通过统一接口操作不同硬件平台。
    """
```

#### 12.3.2 抽象方法

#### 12.3.2.1 submit_circuit

```python
@abstractmethod
def submit_circuit(
    self,
    circuit: str,
    shots: int = 1024,
    task_name: str = "Scheduler_Task",
) -> str | None:
    """
    提交量子电路到硬件后端执行。

    Args:
        circuit: 电路描述字符串（格式由子类的 circuit_format 决定）
        shots: 测量次数（默认：1024）
        task_name: 任务名称（默认："Scheduler_Task"）

    Returns:
        task_id: 任务 ID 字符串；提交失败时返回 None
    """
```

#### 12.3.2.2 get_task_status

```python
@abstractmethod
def get_task_status(self, task_id: str) -> Mapping[str, Any]:
    """
    查询任务状态（非阻塞）。

    Args:
        task_id: 任务 ID

    Returns:
        status: 状态字典，包含 status 字段（"running"/"completed"/"error"/"unknown"）
    """
```

#### 12.3.3 抽象属性

| 属性 | 类型 | 说明 |
|------|------|------|
| `supported_gates` | `list[str]` | 该后端支持的量子门列表 |
| `topology` | `dict[str, Any]` | 硬件拓扑结构信息（耦合图、连接矩阵等） |
| `backend_type` | `str` | 后端类型标识（如 "superconducting"/"ion_trap"/"photonic"） |

#### 12.3.4 公开属性与方法

#### 12.3.4.1 circuit_format 属性

```python
@property
def circuit_format(self) -> CircuitFormat:
    """
    该后端原生支持的电路格式。

    Returns:
        CircuitFormat 枚举值，默认返回 CircuitFormat.QCIS，子类可覆盖
    """
```

#### 12.3.4.2 is_available

```python
def is_available(self) -> bool:
    """
    检查后端是否可用（默认实现：返回 True）。

    子类可覆盖此方法提供更精确的可用性检测（如尝试认证、检查机器在线状态）。

    Returns:
        True 表示后端可接受任务，False 表示不可用

    Example:
        >>> if backend.is_available():
        ...     tid = backend.submit_circuit("H Q0\\nM Q0", shots=1024)
    """
```

### 12.4 IonTrapBackend 类

#### 12.4.1 类定义

```python
class IonTrapBackend(QuantumHardwareBackend):
    """
    离子阱量子计算后端桩实现（Issue #258）。

    离子阱量子计算机使用囚禁离子作为量子比特，特点是：
    - 全连通拓扑（任意两个离子均可直接纠缠）
    - 较长相干时间（秒级，远超超导的微秒级）
    - 典型门集：单比特旋转门 + Mølmer-Sørensen 两比特门

    当前为桩实现，submit_circuit 返回模拟 task_id，不连接真实离子阱平台。
    TODO: 接入真实离子阱平台（如 IonQ Aria / Quantinuum H2）
    """
```

#### 12.4.2 初始化方法

```python
def __init__(
    self,
    num_ions: int = 20,
    api_key: str | None = None,
) -> None:
    """
    初始化离子阱后端桩实现。

    Args:
        num_ions: 离子数量（量子比特数，默认：20）
        api_key: 平台 API Key（桩实现不使用，预留接口）

    Example:
        >>> backend = IonTrapBackend(num_ions=50)
        >>> tid = backend.submit_circuit(circuit_json, shots=1024)
    """
```

#### 12.4.3 属性

| 属性 | 返回值 | 说明 |
|------|--------|------|
| `supported_gates` | `["RZ", "RY", "RX", "RXX", "RYY", "MS", "M"]` | 离子阱典型门集 |
| `topology` | `{"type": "all_to_all", "num_qubits": N, "connectivity": "full", ...}` | 全连通拓扑 |
| `backend_type` | `"ion_trap"` | 后端类型标识 |
| `circuit_format` | `CircuitFormat.IONQ_JSON` | 使用 IonQ JSON 格式 |

#### 12.4.4 公开方法

#### 12.4.4.1 submit_circuit

```python
def submit_circuit(
    self,
    circuit: str,
    shots: int = 1024,
    task_name: str = "IonTrap_Task",
) -> str | None:
    """
    提交电路到离子阱后端（桩实现：返回模拟 task_id）。

    Args:
        circuit: 电路描述（IonQ JSON 格式字符串）
        shots: 测量次数（默认：1024）
        task_name: 任务名称（默认："IonTrap_Task"）

    Returns:
        task_id: 模拟任务 ID（格式："iontrap_stub_{uuid12}"）

    Example:
        >>> tid = backend.submit_circuit('{"qubits": 2, "circuit": [...]}', shots=2048)
    """
```

#### 12.4.4.2 get_task_status

```python
def get_task_status(self, task_id: str) -> dict[str, Any]:
    """
    查询任务状态（桩实现：立即返回 completed）。

    Args:
        task_id: 任务 ID

    Returns:
        status: 模拟完成状态字典，含 task_id/status/result/raw

    Example:
        >>> status = backend.get_task_status("iontrap_stub_abc123")
        >>> assert status["status"] == "completed"
    """
```

#### 12.4.4.3 is_available

```python
def is_available(self) -> bool:
    """
    桩实现始终返回 True（模拟可用）。

    Returns:
        始终返回 True
    """
```

### 12.5 PhotonicBackend 类

#### 12.5.1 类定义

```python
class PhotonicBackend(QuantumHardwareBackend):
    """
    光量子计算后端桩实现（Issue #258）。

    光量子计算机使用光子作为量子比特，特点是：
    - 室温操作（无需极低温环境）
    - 高速并行处理（光速计算）
    - 典型门集：Hadamard + 分束器 + 相移器 + 光子探测
    - 主要范式：玻色采样 / 高斯玻色采样 / 离散变量光量子

    当前为桩实现，submit_circuit 返回模拟 task_id，不连接真实光量子平台。
    TODO: 接入真实光量子平台（如 Xanadu Borealis / 国盾量子）
    """
```

#### 12.5.2 初始化方法

```python
def __init__(
    self,
    num_modes: int = 16,
    api_key: str | None = None,
) -> None:
    """
    初始化光量子后端桩实现。

    Args:
        num_modes: 光学模式数（等效量子比特数，默认：16）
        api_key: 平台 API Key（桩实现不使用，预留接口）

    Example:
        >>> backend = PhotonicBackend(num_modes=32)
        >>> tid = backend.submit_circuit(hamiltonian_str, shots=1024)
    """
```

#### 12.5.3 属性

| 属性 | 返回值 | 说明 |
|------|--------|------|
| `supported_gates` | `["H", "BS", "PS", "S2", "M", "PNR"]` | 光量子典型门集（Hadamard/分束器/相移器/压缩门/测量/光子数分辨） |
| `topology` | `{"type": "linear_chain", "num_modes": N, "connectivity": "nearest_neighbor", ...}` | 线性波导阵列拓扑 |
| `backend_type` | `"photonic"` | 后端类型标识 |
| `circuit_format` | `CircuitFormat.PHOTONIC_HAMILTONIAN` | 使用哈密顿量描述格式 |

#### 12.5.4 公开方法

#### 12.5.4.1 submit_circuit

```python
def submit_circuit(
    self,
    circuit: str,
    shots: int = 1024,
    task_name: str = "Photonic_Task",
) -> str | None:
    """
    提交电路到光量子后端（桩实现：返回模拟 task_id）。

    Args:
        circuit: 电路描述（哈密顿量格式字符串）
        shots: 测量次数（默认：1024）
        task_name: 任务名称（默认："Photonic_Task"）

    Returns:
        task_id: 模拟任务 ID（格式："photonic_stub_{uuid12}"）

    Example:
        >>> tid = backend.submit_circuit(hamiltonian_str, shots=2048)
    """
```

#### 12.5.4.2 get_task_status

```python
def get_task_status(self, task_id: str) -> dict[str, Any]:
    """
    查询任务状态（桩实现：立即返回 completed）。

    Args:
        task_id: 任务 ID

    Returns:
        status: 模拟完成状态字典，含 task_id/status/result/raw

    Example:
        >>> status = backend.get_task_status("photonic_stub_abc123")
        >>> assert status["status"] == "completed"
    """
```

#### 12.5.4.3 is_available

```python
def is_available(self) -> bool:
    """
    桩实现始终返回 True（模拟可用）。

    Returns:
        始终返回 True
    """
```

### 12.6 create_hardware_backend 工厂函数

```python
def create_hardware_backend(
    config: dict[str, Any] | None = None,
) -> QuantumHardwareBackend:
    """
    根据配置创建硬件后端实例（Issue #259）。

    工厂函数根据 config["hardware_type"] 选择对应的后端实现：
    - "superconducting" → CqlibTianyanClient（超导真机）
    - "ion_trap" → IonTrapBackend（离子阱桩）
    - "photonic" → PhotonicBackend（光量子桩）

    Args:
        config: 配置字典，支持以下字段：
            - hardware_type (str): 后端类型，默认 "superconducting"
            - login_key (str): 超导后端的 API Key
            - machine_name (str): 超导后端的机器名（默认 "tianyan_s"）
            - num_ions (int): 离子阱离子数（默认 20）
            - num_modes (int): 光量子模式数（默认 16）
            - api_key (str): 离子阱/光量子的 API Key（预留）

    Returns:
        backend: 对应的 QuantumHardwareBackend 实例

    Raises:
        ValueError: 未知的 hardware_type 时抛出

    Example:
        >>> # 创建超导真机后端
        >>> backend = create_hardware_backend({
        ...     "hardware_type": "superconducting",
        ...     "login_key": "your_api_key",
        ...     "machine_name": "tianyan_s"
        ... })
        >>> # 创建离子阱后端
        >>> backend = create_hardware_backend({
        ...     "hardware_type": "ion_trap",
        ...     "num_ions": 50
        ... })
        >>> # 创建光量子后端
        >>> backend = create_hardware_backend({
        ...     "hardware_type": "photonic",
        ...     "num_modes": 32
        ... })
    """
```

### 12.7 使用示例

```python
from src.api.hardware_adapter import (
    create_hardware_backend,
    CircuitFormat,
)

# 通过工厂函数创建后端
backend = create_hardware_backend({"hardware_type": "superconducting"})

# 统一接口提交任务（不关心底层硬件类型）
if backend.is_available():
    circuit_fmt = backend.circuit_format
    print(f"后端类型: {backend.backend_type}, 电路格式: {circuit_fmt.value}")
    print(f"支持的门: {backend.supported_gates[:5]}...")
    print(f"拓扑: {backend.topology['type']}")

    task_id = backend.submit_circuit("H Q0\nM Q0", shots=1024)
    if task_id:
        status = backend.get_task_status(task_id)
        print(f"任务状态: {status['status']}")
```

---

## 13. QuotaTracker 接口（配额追踪）

### 13.1 模块概述

`quota_tracker.py` 实现了天衍云真机配额持久化追踪与预警模块，支持多维度配额检查、阈值告警、耗尽时间估算和状态持久化。

**注意**：与 `src/api/tianyan_client.py` 中的 QuotaTracker（按窗口计数的轻量 API 配额追踪器）是不同概念；本模块面向真机配额，做持久化追踪与告警预警。

**核心特性**：

- 多维度配额检查（shots/tasks/wall_time_hours）
- 阈值告警（warning/critical，使用 loguru 日志 + 可选 webhook）
- 每日消耗历史记录，用于估算配额耗尽时间
- 状态持久化（JSON 文件），重启后自动恢复
- 线程安全（threading.Lock）

### 13.2 QuotaExhaustedError 异常类

```python
class QuotaExhaustedError(ResourceExhaustedError):
    """
    真机配额耗尽异常。

    当 consume/can_consume 检测到任一维度配额超出上限时抛出。
    继承自 ResourceExhaustedError，便于上层统一资源异常处理。
    """
```

#### 13.2.1 初始化方法

```python
def __init__(
    self,
    dimension: str,
    used: float,
    total: float,
    *,
    code: str = "QUOTA_EXHAUSTED",
    retryable: bool = False,
) -> None:
    """
    初始化配额耗尽异常。

    Args:
        dimension: 触发耗尽的维度名（"shots"/"tasks"/"wall_time_hours"）
        used: 已用量
        total: 总配额
        code: 错误码（关键字参数，默认："QUOTA_EXHAUSTED"）
        retryable: 是否可重试（关键字参数，默认：False）

    Attributes:
        dimension: 触发耗尽的维度名
        used: 已用量
        total: 总配额

    Example:
        >>> try:
        ...     if not tracker.consume(shots=10000):
        ...         raise QuotaExhaustedError("shots", 10000, 10000)
        ... except QuotaExhaustedError as e:
        ...     print(f"{e.dimension} 配额耗尽: {e.used}/{e.total}")
    """
```

### 13.3 QuotaTracker 类

#### 13.3.1 类定义

```python
class QuotaTracker:
    """
    真机配额追踪器。

    从 config/quota.yaml 读取总配额配置，从 logs/quota_state.json 读取持久化状态，
    支持多维度配额检查、阈值告警、耗尽时间估算。

    线程安全：所有公开方法通过 threading.Lock 串行化，适合多线程调度循环调用。
    """
```

#### 13.3.2 初始化方法

```python
def __init__(
    self,
    config_path: str = "config/quota.yaml",
    state_path: str = "logs/quota_state.json",
) -> None:
    """
    初始化配额追踪器，加载配置与持久化状态。

    Args:
        config_path: 配额配置文件路径（默认："config/quota.yaml"）
        state_path: 状态持久化文件路径（默认："logs/quota_state.json"）

    配置文件格式（config/quota.yaml）：
        total_quota:
          shots: 10000
          tasks: 200
          wall_time_hours: 50
        warning_threshold: 0.8
        critical_threshold: 0.95
        notification:
          type: log  # 或 "webhook"
          webhook_url: null  # webhook 地址

    默认配额（配置文件缺失时使用）：
        - shots: 10000
        - tasks: 200
        - wall_time_hours: 50
        - warning_threshold: 0.8
        - critical_threshold: 0.95

    Example:
        >>> tracker = QuotaTracker()
        >>> tracker = QuotaTracker("config/my_quota.yaml", "logs/my_state.json")
    """
```

#### 13.3.3 公开方法

#### 13.3.3.1 can_consume

```python
def can_consume(
    self,
    shots: int = 0,
    tasks: int = 1,
    wall_time_hours: float = 0.0,
) -> bool:
    """
    检查是否还能消费指定额度（不实际扣减）。

    Args:
        shots: 本次拟消费的 shots 数（默认：0）
        tasks: 本次拟消费的任务数（默认：1）
        wall_time_hours: 本次拟消费的墙上时间，单位小时（默认：0.0）

    Returns:
        bool: 全部维度均在配额内返回 True，任一维度超额返回 False

    Example:
        >>> if tracker.can_consume(shots=1024, tasks=1):
        ...     tracker.consume(shots=1024, tasks=1)
        ... else:
        ...     print("配额不足，无法提交任务")
    """
```

#### 13.3.3.2 consume

```python
def consume(
    self,
    shots: int = 0,
    tasks: int = 1,
    wall_time_hours: float = 0.0,
) -> bool:
    """
    消费配额，前置检查并持久化。

    消费成功后自动将状态写入 state_path，确保重启后可恢复。

    Args:
        shots: 本次消费的 shots 数（默认：0）
        tasks: 本次消费的任务数（默认：1）
        wall_time_hours: 本次消费的墙上时间，单位小时（默认：0.0）

    Returns:
        bool: 允许消费并已记录返回 True；任一维度超额返回 False（不抛异常）

    Example:
        >>> success = tracker.consume(shots=2048, tasks=1, wall_time_hours=0.5)
        >>> if not success:
        ...     print("配额不足，消费被拒绝")
    """
```

#### 13.3.3.3 remaining

```python
def remaining(self) -> dict[str, float]:
    """
    返回各维度剩余配额。

    Returns:
        remaining: 各维度剩余量字典，包含：
            - shots: 剩余 shots 数
            - tasks: 剩余任务数
            - wall_time_hours: 剩余墙上时间（小时）
        各值不会小于 0。

    Example:
        >>> rem = tracker.remaining()
        >>> print(f"剩余 shots: {rem['shots']}, 剩余任务: {rem['tasks']}")
    """
```

#### 13.3.3.4 usage_ratio

```python
def usage_ratio(self) -> dict[str, float]:
    """
    返回各维度使用比例（0-1）。

    Returns:
        ratio: 各维度使用比例字典，值范围 0.0-1.0；
            总配额为 0 时对应维度返回 0.0

    Example:
        >>> ratio = tracker.usage_ratio()
        >>> print(f"shots 使用率: {ratio['shots']:.1%}")
    """
```

#### 13.3.3.5 status

```python
def status(self) -> dict[str, Any]:
    """
    返回完整状态摘要供 Web 面板展示。

    Returns:
        summary: 包含以下字段的字典：
            - total: 各维度总配额 {shots, tasks, wall_time_hours}
            - used: 各维度已用量 {shots, tasks, wall_time_hours}
            - remaining: 各维度剩余量 {shots, tasks, wall_time_hours}
            - usage_ratio: 各维度使用比例 {shots, tasks, wall_time_hours}
            - warning_threshold: 警告阈值（默认 0.8）
            - critical_threshold: 危急阈值（默认 0.95）
            - warning_level: 当前告警级别（"normal"/"warning"/"critical"）
            - estimated_exhaustion_time: 各维度估算耗尽时间（或 None）
            - daily_history_count: 每日历史记录条数

    Example:
        >>> s = tracker.status()
        >>> print(f"告警级别: {s['warning_level']}")
        >>> print(f"使用率: shots={s['usage_ratio']['shots']:.1%}")
        >>> if s['estimated_exhaustion_time']:
        ...     print(f"shots 预计耗尽日期: {s['estimated_exhaustion_time']['shots']['date']}")
    """
```

#### 13.3.3.6 check_and_alert

```python
def check_and_alert(self) -> str | None:
    """
    检查阈值并发出告警（使用 loguru.logger）。

    - 任一维度使用率 ≥ critical_threshold 时：logger.critical + 可选 webhook
    - 任一维度使用率 ≥ warning_threshold 时：logger.warning + 可选 webhook
    - 否则：不发出告警

    webhook 通知失败仅记录日志，不阻塞主流程。

    Returns:
        level: 告警级别字符串（"warning"/"critical"），未触发告警返回 None

    Example:
        >>> level = tracker.check_and_alert()
        >>> if level == "critical":
        ...     print("配额危急，请立即处理!")
        ... elif level == "warning":
        ...     print("配额即将耗尽，请注意")
    """
```

#### 13.3.3.7 record_daily_usage

```python
def record_daily_usage(self) -> None:
    """
    记录当日用量到历史（用于估算耗尽时间）。

    若当日已有记录则覆盖更新，否则追加新条目。
    自动保留最近 30 天历史，避免文件无限增长。
    建议通过定时任务每日调用一次。

    Example:
        >>> # 每日定时记录（如通过 APScheduler 或 cron）
        >>> tracker.record_daily_usage()
    """
```

#### 13.3.3.8 get_daily_history

```python
def get_daily_history(self) -> list[dict[str, Any]]:
    """
    返回每日消耗历史（按日期升序）。

    Returns:
        history: 每日用量记录列表，每条记录包含：
            - date: 日期字符串（"YYYY-MM-DD"，UTC）
            - shots: 当日累计 shots 消耗
            - tasks: 当日累计任务数
            - wall_time_hours: 当日累计墙上时间（小时）

    Example:
        >>> history = tracker.get_daily_history()
        >>> for entry in history:
        ...     print(f"{entry['date']}: shots={entry['shots']}, tasks={entry['tasks']}")
    """
```

#### 13.3.4 配额维度说明

| 维度 | 字段名 | 默认总量 | 说明 |
|------|--------|---------|------|
| 测量次数 | `shots` | 10,000 | 量子电路测量采样次数 |
| 任务数 | `tasks` | 200 | 提交到真机的量子任务数量 |
| 墙上时间 | `wall_time_hours` | 50 | 真机占用时间（小时） |

#### 13.3.5 告警级别说明

| 级别 | 触发条件 | 日志级别 | 说明 |
|------|---------|---------|------|
| `normal` | 所有维度使用率 < warning_threshold | - | 正常使用 |
| `warning` | 任一维度使用率 ≥ 0.8（默认） | `logger.warning` | 配额即将耗尽，需关注 |
| `critical` | 任一维度使用率 ≥ 0.95（默认） | `logger.critical` | 配额危急，需立即处理 |

#### 13.3.6 Web `/api/quota` 端点响应

QuotaTracker 的 `status()` 方法返回值直接映射到 Web 可视化 API 的 `GET /api/quota` 端点：

```json
{
  "available": true,
  "total": 10000,
  "used": 3500,
  "remaining": 6500,
  "usage_ratio": 0.35,
  "alert_level": "normal"
}
```

> 配额追踪未启用时返回 `{"available": false, "message": "配额追踪未启用"}`。

#### 13.3.7 使用示例

```python
from src.api.quota_tracker import QuotaTracker, QuotaExhaustedError

# 初始化配额追踪器
tracker = QuotaTracker()

# 提交任务前检查配额
shots_needed = 2048
if tracker.can_consume(shots=shots_needed, tasks=1):
    # 配额充足，消费配额并提交任务
    tracker.consume(shots=shots_needed, tasks=1)
    print(f"任务已提交，剩余 shots: {tracker.remaining()['shots']}")
else:
    print("配额不足，无法提交任务")

# 检查并发出告警
level = tracker.check_and_alert()
if level:
    print(f"告警级别: {level}")

# 获取完整状态
status = tracker.status()
print(f"当前告警级别: {status['warning_level']}")
print(f"使用率: {status['usage_ratio']}")

# 每日定时记录用量（建议放入定时任务）
tracker.record_daily_usage()

# 查看历史消耗
for day in tracker.get_daily_history():
    print(f"{day['date']}: shots={day['shots']}, tasks={day['tasks']}")
```

---

*本文档由文档工程师自动生成，数据来源：`src/api/` 目录源码、`src/visualization/routes.py` Web 可视化 API 源码。*
