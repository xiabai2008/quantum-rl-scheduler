"""
任务解析器模块
Task Parser for Quantum Scheduling System

解析量子任务描述，提取关键特征
支持量子电路、算法描述、资源需求等多种输入格式

包含：
- Task: 规范化的任务数据类（Builder 模式构建）
- TaskParser: 任务解析器（解析/校验/资源预估/格式转换）
- TaskFeatures: 旧版特征向量（向后兼容）
- LegacyTaskParser: 旧版字符串解析器（向后兼容）
"""

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

import yaml

# ============================================================
# 常量定义
# ============================================================

PRIORITY_MAP: dict[str, int] = {
    "low": 1,
    "medium": 2,
    "high": 3,
    "urgent": 4,
}
PRIORITY_REVERSE: dict[int, str] = {v: k for k, v in PRIORITY_MAP.items()}

VALID_TASK_TYPES: set = {"quantum", "classical", "hybrid"}
VALID_STATUSES: set = {"pending", "queued", "running", "completed", "failed"}

KNOWN_ALGORITHMS: set = {
    "VQE",
    "QAOA",
    "Grover",
    "Shor",
    "HHL",
    "QSVM",
    "QFT",
    "QPE",
    "AmplitudeEncoding",
    "Variational",
}

# 天衍-287 真机约束
MAX_QUBITS: int = 287
MAX_CIRCUIT_DEPTH: int = 10000
MAX_SHOTS: int = 100000


# ============================================================
# Task dataclass — 规范化任务表示
# ============================================================


@dataclass
class Task:
    """规范化量子/经典任务数据结构"""

    task_id: str
    task_type: Literal["quantum", "classical", "hybrid"]
    qubits_required: int
    estimated_time: float
    priority: int  # 1-4, low→urgent
    submitted_at: datetime = field(default_factory=datetime.now)
    algorithm: str | None = None
    circuit_depth: int | None = None
    shots: int | None = None
    deadline: datetime | None = None
    status: Literal["pending", "queued", "running", "completed", "failed"] = "pending"


# ============================================================
# TaskBuilder — Builder 模式
# ============================================================


class TaskBuilder:
    """
    Task 的 Builder 模式构造器

    用法::

        task = (
            TaskBuilder()
            .set_id("task_001")
            .set_type("quantum")
            .set_algorithm("VQE")
            .set_qubits(8)
            .set_circuit_depth(50)
            .set_shots(1024)
            .set_estimated_time(120)
            .set_priority("high")
            .set_deadline("2026-07-01T12:00:00")
            .build()
        )
    """

    def __init__(self) -> None:
        self._data: dict[str, Any] = {
            "task_id": "",
            "task_type": "quantum",
            "qubits_required": 0,
            "estimated_time": 0.0,
            "priority": 2,  # default medium
            "submitted_at": datetime.now(),
            "algorithm": None,
            "circuit_depth": None,
            "shots": None,
            "deadline": None,
            "status": "pending",
        }

    # ---- setters（链式调用，返回 self）----

    def set_id(self, task_id: str) -> "TaskBuilder":
        self._data["task_id"] = task_id
        return self

    def set_type(self, task_type: str) -> "TaskBuilder":
        task_type = task_type.lower().strip()
        if task_type not in VALID_TASK_TYPES:
            raise ValueError(
                f"Invalid task_type '{task_type}'. " f"Must be one of {sorted(VALID_TASK_TYPES)}"
            )
        self._data["task_type"] = task_type
        return self

    def set_algorithm(self, algorithm: str | None) -> "TaskBuilder":
        self._data["algorithm"] = algorithm
        return self

    def set_qubits(self, qubits: int) -> "TaskBuilder":
        self._data["qubits_required"] = qubits
        return self

    def set_circuit_depth(self, depth: int | None) -> "TaskBuilder":
        self._data["circuit_depth"] = depth
        return self

    def set_shots(self, shots: int | None) -> "TaskBuilder":
        self._data["shots"] = shots
        return self

    def set_estimated_time(self, seconds: float) -> "TaskBuilder":
        self._data["estimated_time"] = float(seconds)
        return self

    def set_priority(self, priority: str | int) -> "TaskBuilder":
        if isinstance(priority, str):
            priority = priority.lower().strip()
            if priority not in PRIORITY_MAP:
                raise ValueError(
                    f"Invalid priority '{priority}'. " f"Must be one of {list(PRIORITY_MAP.keys())}"
                )
            self._data["priority"] = PRIORITY_MAP[priority]
        else:
            if not 1 <= priority <= 4:
                raise ValueError("Priority must be an integer between 1 and 4.")
            self._data["priority"] = priority
        return self

    def set_deadline(self, deadline: str | datetime | None) -> "TaskBuilder":
        if deadline is None:
            self._data["deadline"] = None
        elif isinstance(deadline, datetime):
            self._data["deadline"] = deadline
        else:
            self._data["deadline"] = datetime.fromisoformat(deadline)
        return self

    def set_status(self, status: str) -> "TaskBuilder":
        status = status.lower().strip()
        if status not in VALID_STATUSES:
            raise ValueError(
                f"Invalid status '{status}'. " f"Must be one of {sorted(VALID_STATUSES)}"
            )
        self._data["status"] = status
        return self

    def set_submitted_at(self, dt: datetime | None) -> "TaskBuilder":
        self._data["submitted_at"] = dt or datetime.now()
        return self

    # ---- from dict ----

    @classmethod
    def from_dict(cls, task_dict: dict[str, Any]) -> "TaskBuilder":
        """从字典构建 Builder，自动映射字段名。"""
        builder = cls()

        # task_id
        if "task_id" in task_dict:
            builder.set_id(task_dict["task_id"])

        # task_type — 兼容 "type" 键
        t = task_dict.get("type") or task_dict.get("task_type", "quantum")
        builder.set_type(str(t))

        # algorithm
        if "algorithm" in task_dict:
            builder.set_algorithm(task_dict["algorithm"])

        # qubits — 兼容 "qubits_required" / "qubit_count"
        q = task_dict.get("qubits_required", task_dict.get("qubit_count", 0))
        builder.set_qubits(int(q))

        # circuit_depth
        builder.set_circuit_depth(task_dict.get("circuit_depth"))

        # shots
        builder.set_shots(task_dict.get("shots"))

        # estimated_time
        builder.set_estimated_time(task_dict.get("estimated_time", 0.0))

        # priority — 兼容字符串 / 整数
        p = task_dict.get("priority", 2)
        builder.set_priority(p)

        # deadline
        if "deadline" in task_dict:
            builder.set_deadline(task_dict["deadline"])

        # status
        if "status" in task_dict:
            builder.set_status(task_dict["status"])

        return builder

    # ---- build ----

    def build(self) -> Task:
        """构建 Task 实例（基本字段校验）。"""
        if not self._data["task_id"]:
            raise ValueError("task_id is required and cannot be empty.")
        return Task(**self._data)


# ============================================================
# TaskParser — 核心解析器
# ============================================================


class TaskParser:
    """
    量子任务解析器

    功能：
    1. parse(task_dict)          — 将字典解析为 Task 对象
    2. validate(task)            — 验证任务参数的合法性
    3. estimate_resources(task)  — 预估资源消耗
    4. to_internal_format(task)   — 转换为内部调度格式
    """

    def __init__(self) -> None:
        self.max_qubits: int = MAX_QUBITS
        self.max_circuit_depth: int = MAX_CIRCUIT_DEPTH
        self.max_shots: int = MAX_SHOTS

    # ----------------------------------------------------------
    # 1. parse — 字典 → Task
    # ----------------------------------------------------------

    def parse(self, task_dict: dict[str, Any]) -> Task:
        """
        将字典解析为 Task 对象。

        Args:
            task_dict: 任务描述字典。

        Returns:
            解析后的 Task 对象。

        Raises:
            TypeError:  输入不是字典。
            ValueError:  缺少必填字段或字段类型不合法。
        """
        if not isinstance(task_dict, dict):
            raise TypeError(f"task_dict must be a dict, got {type(task_dict).__name__}")

        # 必填字段检查
        required_keys = {"task_id"}
        missing = required_keys - set(task_dict.keys())
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")

        task = TaskBuilder.from_dict(task_dict).build()

        # 解析后自动校验
        errors = self._collect_errors(task)
        if errors:
            raise ValueError("Task validation failed:\n  - " + "\n  - ".join(errors))

        return task

    # ----------------------------------------------------------
    # 2. validate — 校验合法性
    # ----------------------------------------------------------

    def validate(self, task: Task) -> bool:
        """
        验证任务参数的合法性。

        Args:
            task: Task 实例。

        Returns:
            True if valid, else False（错误信息会打印到 stderr）。
        """
        if not isinstance(task, Task):
            raise TypeError(f"validate() expects a Task instance, got {type(task).__name__}")
        errors = self._collect_errors(task)
        if errors:
            import sys

            for e in errors:
                print(f"[validation error] {e}", file=sys.stderr)
            return False
        return True

    def _collect_errors(self, task: Task) -> list[str]:
        """收集所有校验错误（不抛异常）。"""
        errors: list[str] = []

        # task_id
        if not task.task_id or not isinstance(task.task_id, str):
            errors.append("task_id must be a non-empty string.")

        # task_type
        if task.task_type not in VALID_TASK_TYPES:
            errors.append(
                f"task_type '{task.task_type}' is invalid. "
                f"Expected one of {sorted(VALID_TASK_TYPES)}."
            )

        # qubits_required
        if not isinstance(task.qubits_required, int) or task.qubits_required < 0:
            errors.append("qubits_required must be a non-negative integer.")
        elif task.task_type in ("quantum", "hybrid") and task.qubits_required > self.max_qubits:
            errors.append(
                f"qubits_required ({task.qubits_required}) exceeds "
                f"system limit ({self.max_qubits})."
            )

        # circuit_depth
        if task.circuit_depth is not None:
            if not isinstance(task.circuit_depth, int) or task.circuit_depth < 0:
                errors.append("circuit_depth must be a non-negative integer.")
            elif task.circuit_depth > self.max_circuit_depth:
                errors.append(
                    f"circuit_depth ({task.circuit_depth}) exceeds "
                    f"system limit ({self.max_circuit_depth})."
                )

        # shots
        if task.shots is not None:
            if not isinstance(task.shots, int) or task.shots < 0:
                errors.append("shots must be a non-negative integer.")
            elif task.shots > self.max_shots:
                errors.append(f"shots ({task.shots}) exceeds system limit ({self.max_shots}).")

        # estimated_time
        if not isinstance(task.estimated_time, (int, float)) or task.estimated_time < 0:
            errors.append("estimated_time must be a non-negative number.")

        # priority
        if not isinstance(task.priority, int) or task.priority < 1 or task.priority > 4:
            errors.append("priority must be an integer in range [1, 4].")

        # status
        if task.status not in VALID_STATUSES:
            errors.append(
                f"status '{task.status}' is invalid. " f"Expected one of {sorted(VALID_STATUSES)}."
            )

        # deadline
        if task.deadline is not None and not isinstance(task.deadline, datetime):
            errors.append("deadline must be a datetime object or None.")

        # quantum 类型约束
        if task.task_type == "quantum":
            if task.qubits_required <= 0:
                errors.append("Quantum task must have qubits_required > 0.")
            if task.algorithm and not isinstance(task.algorithm, str):
                errors.append("algorithm must be a string when provided.")

        return errors

    # ----------------------------------------------------------
    # 3. estimate_resources — 预估资源消耗
    # ----------------------------------------------------------

    def estimate_resources(self, task: Task) -> dict[str, Any]:
        """
        预估任务资源消耗。

        Args:
            task: Task 实例。

        Returns:
            资源预估字典，包含：
            - qubit_hours: 量子比特·小时
            - total_gate_operations: 总门操作数
            - memory_mb: 预估内存占用 (MB)
            - classical_compute_ratio: 经典计算占比
            - estimated_queue_time: 预估排队时间 (秒)
        """
        self.validate(task)  # 确保数据合法

        depth = task.circuit_depth or 0
        shots = task.shots or 1
        qubits = max(task.qubits_required, 1)

        # 量子比特·小时 = qubits × 执行时间
        qubit_hours = (qubits * task.estimated_time) / 3600.0

        # 总门操作数 ≈ depth × shots
        total_gate_ops = depth * shots

        # 内存：状态向量 2^n × 复数精度 ~ 16B，取 log 尺度
        if qubits <= 30:
            state_vector_bytes = (2**qubits) * 16
            memory_mb = state_vector_bytes / (1024**2)
        else:
            # 大比特数无法存储全状态向量，按稀疏/张量网络估算
            memory_mb = depth * qubits * 0.001  # 简化启发式

        # 经典计算占比
        if task.task_type == "classical":
            classical_ratio = 1.0
        elif task.task_type == "hybrid":
            classical_ratio = 0.5
        else:
            classical_ratio = 0.1

        # 排队时间预估：与 qubit_hours 和 priority 相关
        # priority 1(低) → 基准时间，priority 4(紧急) → 加速
        base_queue = qubit_hours * 60.0  # 简化线性模型
        priority_factor = {1: 2.0, 2: 1.5, 3: 1.0, 4: 0.3}
        estimated_queue = base_queue * priority_factor.get(task.priority, 1.0)

        return {
            "qubit_hours": round(qubit_hours, 4),
            "total_gate_operations": total_gate_ops,
            "memory_mb": round(memory_mb, 2),
            "classical_compute_ratio": classical_ratio,
            "estimated_queue_time": round(estimated_queue, 2),
        }

    # ----------------------------------------------------------
    # 4. to_internal_format — 转换为内部调度格式
    # ----------------------------------------------------------

    def to_internal_format(self, task: Task) -> dict[str, Any]:
        """
        将 Task 转换为内部调度系统使用的字典格式。

        内部格式额外包含：
        - resource_estimate: 资源预估
        - priority_label: 优先级文字标签
        - scheduling_weight: 调度权重分数

        Args:
            task: Task 实例。

        Returns:
            内部调度格式字典。
        """
        self.validate(task)

        resource_estimate = self.estimate_resources(task)

        # 调度权重 = priority × deadline_urgency × (1 / estimated_time)
        deadline_urgency = 1.0
        if task.deadline:
            remaining = (task.deadline - datetime.now()).total_seconds()
            if remaining > 0:
                # 越接近截止时间，紧迫度越高
                deadline_urgency = 1.0 + 3.0 / (remaining / 3600.0 + 1.0)

        time_factor = 1.0 / max(task.estimated_time, 1.0)
        scheduling_weight = task.priority * deadline_urgency * time_factor * 1000

        internal: dict[str, Any] = {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "algorithm": task.algorithm,
            "qubits_required": task.qubits_required,
            "circuit_depth": task.circuit_depth,
            "shots": task.shots,
            "estimated_time": task.estimated_time,
            "priority": task.priority,
            "priority_label": PRIORITY_REVERSE.get(task.priority, "medium"),
            "deadline": task.deadline.isoformat() if task.deadline else None,
            "submitted_at": task.submitted_at.isoformat(),
            "status": task.status,
            "resource_estimate": resource_estimate,
            "scheduling_weight": round(scheduling_weight, 4),
        }

        return internal


# ============================================================
# 旧版兼容 — TaskFeatures & LegacyTaskParser
# ============================================================


@dataclass
class TaskFeatures:
    """任务特征向量（向后兼容）"""

    task_id: str
    user_id: str
    task_type: str  # "quantum", "classical", "hybrid"

    # 量子特征
    qubit_count: int = 0
    circuit_depth: int = 0
    gate_count: int = 0
    measurement_count: int = 0

    # 算法特征
    algorithm: str = "unknown"  # "VQE", "QAOA", "Grover", etc.
    problem_size: int = 0  # 问题规模（如TSP城市数）

    # 资源需求
    estimated_time: float = 0.0  # 秒
    priority: int = 3  # 1-5
    memory_requirement: float = 0.0  # MB

    # 时间特征
    arrival_time: datetime = field(default_factory=datetime.now)
    deadline: datetime | None = None

    # 历史特征
    user_historical_usage: float = 0.0  # 用户历史资源使用量
    user_historical_completion_rate: float = 1.0

    def to_vector(self, feature_dim: int = 20) -> list[float]:
        """
        转换为特征向量

        Returns:
            归一化后的特征向量
        """
        vector = []

        # 任务类型（one-hot）
        type_vec = [0.0, 0.0, 0.0]
        if self.task_type == "quantum":
            type_vec[0] = 1.0
        elif self.task_type == "classical":
            type_vec[1] = 1.0
        else:  # hybrid
            type_vec[2] = 1.0
        vector.extend(type_vec)

        # 量子特征（归一化）
        vector.append(min(self.qubit_count / 287.0, 1.0))  # 天衍-287
        vector.append(min(self.circuit_depth / 1000.0, 1.0))
        vector.append(min(self.gate_count / 10000.0, 1.0))
        vector.append(min(self.measurement_count / 1000.0, 1.0))

        # 算法特征
        algo_vec = [0.0] * 5
        algo_list = ["VQE", "QAOA", "Grover", "Shor", "Other"]
        if self.algorithm in algo_list:
            idx = algo_list.index(self.algorithm) if self.algorithm != "Other" else 4
            algo_vec[idx] = 1.0
        vector.extend(algo_vec)

        # 资源需求（归一化）
        vector.append(min(self.estimated_time / 3600.0, 1.0))  # 最长1小时
        vector.append(self.priority / 5.0)
        vector.append(min(self.memory_requirement / 16384.0, 1.0))  # 最长16GB

        # 时间特征
        if self.deadline:
            time_remaining = (self.deadline - datetime.now()).total_seconds()
            vector.append(max(time_remaining / 86400.0, 0.0))  # 剩余天数
        else:
            vector.append(1.0)  # 无截止时间

        # 用户历史特征
        vector.append(min(self.user_historical_usage / 1000.0, 1.0))
        vector.append(self.user_historical_completion_rate)

        # 填充或截断到指定维度
        if len(vector) < feature_dim:
            vector.extend([0.0] * (feature_dim - len(vector)))
        else:
            vector = vector[:feature_dim]

        return vector


class LegacyTaskParser:
    """
    旧版任务解析器（向后兼容）

    解析多种格式的任务描述，提取结构化特征。
    保留原有接口：parse(str, format) → TaskFeatures。
    """

    def __init__(self) -> None:
        self.supported_formats = ["json", "yaml", "qasm", "text"]

    def parse(self, task_description: str, format: str = "json") -> TaskFeatures | None:
        """
        解析任务描述

        Args:
            task_description: 任务描述字符串
            format: 输入格式 ("json", "yaml", "qasm", "text")

        Returns:
            TaskFeatures对象，解析失败返回None
        """
        if format == "json":
            return self._parse_json(task_description)
        elif format == "yaml":
            return self._parse_yaml(task_description)
        elif format == "qasm":
            return self._parse_qasm(task_description)
        elif format == "text":
            return self._parse_text(task_description)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _parse_json(self, json_str: str) -> TaskFeatures | None:
        """解析JSON格式任务描述"""
        try:
            data = json.loads(json_str)

            features = TaskFeatures(
                task_id=data.get("task_id", "unknown"),
                user_id=data.get("user_id", "unknown"),
                task_type=data.get("task_type", "quantum"),
                qubit_count=data.get("qubit_count", 0),
                circuit_depth=data.get("circuit_depth", 0),
                gate_count=data.get("gate_count", 0),
                algorithm=data.get("algorithm", "unknown"),
                estimated_time=data.get("estimated_time", 0.0),
                priority=data.get("priority", 3),
                memory_requirement=data.get("memory", 0.0),
            )

            return features

        except Exception as e:
            print(f"JSON解析失败: {e}")
            return None

    def _parse_yaml(self, yaml_str: str) -> TaskFeatures | None:
        """解析YAML格式任务描述"""
        try:
            data = yaml.safe_load(yaml_str)
            return self._parse_json(json.dumps(data))
        except Exception as e:
            print(f"YAML解析失败: {e}")
            return None

    def _parse_qasm(self, qasm_str: str) -> TaskFeatures | None:
        """
        解析QASM格式量子电路描述

        QASM (Quantum Assembly Language)是量子电路的标准描述语言
        示例：
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[5];
            creg c[5];
            h q[0];
            cx q[0],q[1];
            measure q[0] -> c[0];
        """
        try:
            lines = qasm_str.strip().split("\n")

            qubit_count = 0
            gate_count = 0
            measurement_count = 0

            for line in lines:
                line = line.strip().lower()

                if line.startswith("qreg"):
                    start = line.find("[") + 1
                    end = line.find("]")
                    qubit_count = int(line[start:end])

                elif any(gate in line for gate in ["h ", "x ", "y ", "z ", "cx ", "cz "]):
                    gate_count += 1

                elif line.startswith("measure"):
                    measurement_count += 1

            circuit_depth = gate_count // 2

            features = TaskFeatures(
                task_id=f"qasm_task_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                user_id="unknown",
                task_type="quantum",
                qubit_count=qubit_count,
                circuit_depth=circuit_depth,
                gate_count=gate_count,
                measurement_count=measurement_count,
                algorithm="unknown",
                estimated_time=circuit_depth * 0.001,
                priority=3,
            )

            return features

        except Exception as e:
            print(f"QASM解析失败: {e}")
            return None

    def _parse_text(self, text: str) -> TaskFeatures | None:
        """
        解析自然语言任务描述（简化版）

        使用关键词匹配提取特征
        实际生产环境应使用NLP模型
        """
        try:
            text_lower = text.lower()

            if "量子" in text or "quantum" in text_lower:
                task_type = "quantum"
            elif "经典" in text or "classical" in text_lower:
                task_type = "classical"
            else:
                task_type = "hybrid"

            qubit_count = 0
            qubit_match = re.search(r"(\d+)\s*(比特|qubit)", text)
            if qubit_match:
                qubit_count = int(qubit_match.group(1))

            algorithm = "unknown"
            algo_keywords = ["VQE", "QAOA", "Grover", "Shor", "HHL"]
            for algo in algo_keywords:
                if algo.lower() in text_lower:
                    algorithm = algo
                    break

            features = TaskFeatures(
                task_id=f"text_task_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                user_id="unknown",
                task_type=task_type,
                qubit_count=qubit_count,
                algorithm=algorithm,
                estimated_time=60.0,
                priority=3,
            )

            return features

        except Exception as e:
            print(f"文本解析失败: {e}")
            return None

    def batch_parse(self, task_descriptions: list[str], format: str = "json") -> list[TaskFeatures]:
        """批量解析任务描述"""
        results = []
        for desc in task_descriptions:
            result = self.parse(desc, format)
            if result:
                results.append(result)
        return results


# ============================================================
# __main__ — 演示
# ============================================================

if __name__ == "__main__":
    # ---- 新版 TaskParser 演示 ----
    parser = TaskParser()

    task_dict = {
        "task_id": "task_001",
        "type": "quantum",
        "algorithm": "VQE",
        "qubits_required": 8,
        "circuit_depth": 50,
        "shots": 1024,
        "estimated_time": 120,
        "priority": "high",
        "deadline": "2026-07-01T12:00:00",
    }

    print("=" * 60)
    print("新版 TaskParser 演示")
    print("=" * 60)

    # 1. parse
    task = parser.parse(task_dict)
    print(f"\n[parse] Task: {task}")

    # 2. validate
    is_valid = parser.validate(task)
    print(f"\n[validate] valid = {is_valid}")

    # 3. estimate_resources
    resources = parser.estimate_resources(task)
    print("\n[estimate_resources]:")
    for k, v in resources.items():
        print(f"  {k}: {v}")

    # 4. to_internal_format
    internal = parser.to_internal_format(task)
    print("\n[to_internal_format]:")
    for k, v in internal.items():
        print(f"  {k}: {v}")

    # ---- Builder 模式演示 ----
    print("\n" + "=" * 60)
    print("Builder 模式演示")
    print("=" * 60)
    task2 = (
        TaskBuilder()
        .set_id("task_002")
        .set_type("hybrid")
        .set_algorithm("QAOA")
        .set_qubits(20)
        .set_circuit_depth(200)
        .set_shots(4096)
        .set_estimated_time(300)
        .set_priority("urgent")
        .build()
    )
    print(f"Builder 构建结果: {task2}")

    # ---- 旧版兼容演示 ----
    print("\n" + "=" * 60)
    print("旧版 LegacyTaskParser 兼容演示")
    print("=" * 60)
    legacy_parser = LegacyTaskParser()
    json_str = """{
        "task_id": "task_001",
        "user_id": "user_123",
        "task_type": "quantum",
        "qubit_count": 10,
        "circuit_depth": 50,
        "algorithm": "VQE",
        "estimated_time": 120.0,
        "priority": 4
    }"""
    features = legacy_parser.parse(json_str, format="json")
    if features:
        print(f"task_id: {features.task_id}")
        print(f"qubit_count: {features.qubit_count}")
        print(f"algorithm: {features.algorithm}")
        print(f"特征向量（前5维）: {features.to_vector()[:5]}")
