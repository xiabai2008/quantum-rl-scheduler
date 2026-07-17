# 量子RL调度系统 API 接口文档

> **文档版本**: v1.0
> **生成日期**: 2026-07-02
> **适用模块**: `src/api/` 目录下所有公开接口
> **维护状态**: 活跃维护

---

## 目录

1. [概述](#1-概述)
2. [通用规范](#2-通用规范)
3. [TianyanClient 接口](#3-tianyanclient-接口)
4. [TianyanCqlibClient 接口](#4-tianyancqlibclient-接口)
5. [MockClient 接口](#5-mockclient-接口)
6. [CircuitBreaker 接口](#6-circuitbreaker-接口)
7. [异常处理](#7-异常处理)
8. [使用示例](#8-使用示例)
9. [附录](#9-附录)

---

## 1. 概述

### 1.1 模块职责

`src/api/` 目录封装了与天衍云量子计算平台的交互逻辑，提供统一的接口抽象层，支持：

- **真机模式**：通过 cqlib SDK 连接天衍云超导量子计算机（287 量子比特）
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
    - 量子退火任务提交（QUBO 问题求解）
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
    提交量子退火任务（QUBO 问题求解）
    
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

## 6. CircuitBreaker 接口

### 6.1 类定义

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

### 6.2 初始化方法

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

### 6.3 公开方法

#### 6.3.1 call

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

#### 6.3.2 get_state

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

#### 6.3.3 reset

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

## 7. 异常处理

### 7.1 异常层次结构

```python
QuantumSchedulerError (基类)
├── TianyanAPIError (API 错误基类)
│   ├── TianyanAuthError (认证错误)
│   ├── TianyanConnectionError (连接错误)
│   ├── TianyanSubmissionError (提交错误)
│   ├── TianyanQueryError (查询错误)
│   ├── TianyanCancellationError (取消错误)
│   ├── TianyanValidationError (校验错误)
│   └── TianyanNotFoundError (资源不存在)
└── CircuitBreakerOpenError (熔断器开启)
```

### 7.2 异常属性

所有异常继承自 `QuantumSchedulerError`，包含以下属性：

```python
class QuantumSchedulerError(Exception):
    def __init__(self, message: str, code: str | None = None, retryable: bool = False):
        """
        Args:
            message: 错误描述
            code: 错误代码（如 "TIANYAN_AUTH_FAILED"）
            retryable: 是否可重试（True 表示可安全重试）
        """
```

### 7.3 异常处理示例

```python
from src.exceptions import (
    TianyanAuthError,
    TianyanSubmissionError,
    CircuitBreakerOpenError
)

try:
    task_id = client.submit_task(qasm, "tianyan_s")
except TianyanAuthError as e:
    print(f"认证失败: {e.message}")
    # 检查 API 密钥配置
except TianyanSubmissionError as e:
    if e.retryable:
        print(f"提交失败，可重试: {e.message}")
        # 执行重试逻辑
    else:
        print(f"提交失败，不可重试: {e.message}")
        # 记录错误，通知用户
except CircuitBreakerOpenError:
    print("熔断器开启，API 暂时不可用")
    # 降级到 Mock 模式或排队等待
```

---

## 8. 使用示例

### 8.1 基础使用（Mock 模式）

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

### 8.2 真机使用（cqlib 模式）

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

### 8.3 多机器协调

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

### 8.4 熔断器集成

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

## 9. 附录

### 9.1 Prometheus 指标

API 层暴露以下 Prometheus 指标：

| 指标名称 | 类型 | 说明 |
|---------|------|------|
| `tianyan_api_requests_total` | Counter | API 请求总数（按方法、状态分组） |
| `tianyan_api_request_duration_seconds` | Histogram | API 请求延迟分布 |
| `tianyan_api_success_rate` | Gauge | API 成功率（滑动窗口） |
| `circuit_breaker_state` | Gauge | 熔断器状态（0=CLOSED, 1=OPEN, 2=HALF_OPEN） |
| `circuit_breaker_failures_total` | Counter | 熔断器失败计数 |

### 9.2 环境变量参考

| 变量名 | 说明 | 默认值 |
|--------|------|--------|
| `TIANYAN_MODE` | 运行模式（"real" / "mock"） | "mock" |
| `TIANYAN_API_KEY` | 天衍云 API 密钥 | - |
| `TIANYAN_USER_ID` | 天衍云用户 ID | - |
| `TIANYAN_SUBMIT_TIMEOUT` | 任务提交超时（秒） | 30.0 |
| `TIANYAN_QUERY_TIMEOUT` | 结果查询超时（秒） | 60.0 |
| `TIANYAN_STATUS_TIMEOUT` | 状态查询超时（秒） | 10.0 |

### 9.3 机器 ID 参考

| 机器 ID | 名称 | 量子比特数 | 类型 |
|---------|------|-----------|------|
| `tianyan_s` | 天衍-S | 287 | 超导量子计算机 |
| `tianyan_sw` | 天衍-SW | 287 | 超导量子计算机 |
| `tianyan_tn` | 天衍-TN | 287 | 超导量子计算机 |
| `tianyan_annealer` | 天衍退火器 | - | 量子退火器 |

### 9.4 版本历史

| 版本 | 日期 | 变更说明 |
|------|------|---------|
| v1.0 | 2026-07-02 | 初始版本，覆盖所有公开接口 |

---

*本文档由文档工程师自动生成，数据来源：`src/api/` 目录源码。*
