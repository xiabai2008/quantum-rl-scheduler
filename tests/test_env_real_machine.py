#!/usr/bin/env python
"""env_real_machine.py 真机闭环模块的单元测试"""

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import pytest

from src.scheduler.env import QuantumSchedulingEnv
from src.scheduler.env_real_machine import generate_qcis_circuit, submit_to_real_machine
from src.scheduler.env_types import QuantumMachine, Task


class TestGenerateQcisCircuit:
    """QCIS 电路生成测试"""

    def test_small_quantum_task(self):
        """小规模量子任务生成合理电路"""
        task = Task(task_id="0", task_type="quantum", qubit_count=3, priority=3)
        qcis = generate_qcis_circuit(task)
        assert isinstance(qcis, str)
        assert len(qcis) > 0
        # 至少包含测量
        assert "M" in qcis
        # 至少包含单比特门
        assert any(g in qcis for g in ["H", "X", "Y", "Z", "RX", "RY", "RZ"])
        # 比特数不超过任务需求
        lines = qcis.strip().split("\n")
        for line in lines:
            if "Q" in line:
                for part in line.split():
                    if part.startswith("Q"):
                        q_idx = int(part.split(",")[0][1:])
                        assert q_idx < 3, f"比特索引 {q_idx} 超出范围"

    def test_priority_affects_depth(self):
        """高优先级任务电路更深"""
        task_low = Task(task_id="0", task_type="quantum", qubit_count=5, priority=1)
        task_high = Task(task_id="1", task_type="quantum", qubit_count=5, priority=5)
        qcis_low = generate_qcis_circuit(task_low, seed=42)
        qcis_high = generate_qcis_circuit(task_high, seed=42)
        # 高优先级应包含更多门（深度因子更大）
        assert len(qcis_high) >= len(
            qcis_low
        ), f"高优先级电路应更深: {len(qcis_high)} vs {len(qcis_low)}"

    def test_deterministic_with_seed(self):
        """相同 seed 生成相同电路"""
        task = Task(task_id="0", task_type="quantum", qubit_count=5, priority=3)
        q1 = generate_qcis_circuit(task, seed=42)
        q2 = generate_qcis_circuit(task, seed=42)
        assert q1 == q2

    def test_different_seed_produces_different_circuit(self):
        """不同 seed 可能生成不同电路"""
        task = Task(task_id="0", task_type="quantum", qubit_count=5, priority=3)
        q1 = generate_qcis_circuit(task, seed=42)
        q2 = generate_qcis_circuit(task, seed=123)
        # 注意：小规模电路可能恰好相同，所以只是"可能"不同
        # 用多比特任务确保大概率不同
        task_big = Task(task_id="0", task_type="quantum", qubit_count=20, priority=5)
        q3 = generate_qcis_circuit(task_big, seed=42)
        q4 = generate_qcis_circuit(task_big, seed=123)
        assert q3 != q4, "大电路不同 seed 应生成不同电路"

    def test_classical_task_generates_minimal_circuit(self):
        """经典任务（qubit_count=0）生成最小电路"""
        task = Task(task_id="0", task_type="classical", qubit_count=0, priority=1)
        qcis = generate_qcis_circuit(task)
        assert "M Q0" in qcis
        assert qcis.count("\n") >= 1  # 至少 1 个单比特门 + 1 个测量

    def test_max_qubits_limit(self):
        """超过 max_qubits 限制时被截断"""
        task = Task(task_id="0", task_type="quantum", qubit_count=500, priority=3)
        qcis = generate_qcis_circuit(task, max_qubits=10)
        # 不应包含 Q10 以上的比特
        for line in qcis.strip().split("\n"):
            for part in line.split():
                for token in part.split(","):
                    if token.startswith("Q"):
                        q_idx = int(token[1:])
                        assert q_idx < 10, f"比特索引 {q_idx} 超出 max_qubits=10"

    def test_qcis_format_valid(self):
        """生成的电路符合 QCIS 基本格式"""
        task = Task(task_id="0", task_type="quantum", qubit_count=8, priority=3)
        qcis = generate_qcis_circuit(task, seed=42)
        lines = qcis.strip().split("\n")
        for line in lines:
            parts = line.split()
            assert len(parts) >= 2, f"每行至少要有门和比特: {line}"
            # 门名
            assert parts[0] in [
                "H",
                "X",
                "Y",
                "Z",
                "RX",
                "RY",
                "RZ",
                "CNOT",
                "CZ",
                "M",
            ], f"未知门: {parts[0]}"
            # 比特引用
            for p in parts[1:]:
                assert p.startswith("Q"), f"应为比特引用: {p}"


class TestTaskQcisField:
    """Task 数据类 qcis 字段测试"""

    def test_task_has_qcis_field(self):
        """Task 数据类默认包含 qcis 字段"""
        task = Task(task_id="0", task_type="quantum")
        assert hasattr(task, "qcis")
        assert task.qcis is None  # 默认未生成

    def test_task_accepts_qcis(self):
        """Task 可以接受自定义 qcis 电路"""
        custom_qcis = "H Q0\nCNOT Q0 Q1\nM Q0\nM Q1"
        task = Task(task_id="0", task_type="quantum", qcis=custom_qcis)
        assert task.qcis == custom_qcis

    def test_task_without_qcis_still_works(self):
        """没有 qcis 的 Task 仍然可以正常使用"""
        task = Task(task_id="0", task_type="classical", qubit_count=0)
        assert task.qcis is None
        # submit_to_real_machine 应自动生成


class _RecordingClient:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def submit_quantum_task(self, **kwargs):
        self.calls.append(kwargs)
        return f"real-{len(self.calls)}"


class TestRealSubmissionBudget:
    """训练级真机硬上限与低 shots 配置测试。"""

    @staticmethod
    def _make_env(max_real_submissions: int | None = 2) -> tuple:
        env = QuantumSchedulingEnv(
            machine_configs=[
                {
                    "name": "tianyan_s",
                    "total_qubits": 287,
                    "supported_gates": ("H", "M"),
                    "is_real": True,
                }
            ],
            use_real_machine=True,
            max_real_submissions=max_real_submissions,
            real_machine_shots=64,
        )
        client = _RecordingClient()
        env.attach_real_clients({"tianyan_s": client})
        machine = QuantumMachine(name="tianyan_s", total_qubits=287, is_real=True)
        task = Task(task_id="budget", task_type="quantum", qubit_count=1)
        return env, client, machine, task

    def test_rejects_invalid_budget_configuration(self):
        with pytest.raises(ValueError, match="max_real_submissions"):
            QuantumSchedulingEnv(max_real_submissions=-1)
        with pytest.raises(ValueError, match="real_machine_shots"):
            QuantumSchedulingEnv(real_machine_shots=0)

    def test_hard_limit_survives_reset_and_uses_configured_shots(self):
        env, client, machine, task = self._make_env()

        submit_to_real_machine(env, machine, task)
        env.reset(seed=1)
        submit_to_real_machine(env, machine, task)
        submit_to_real_machine(env, machine, task)

        assert len(client.calls) == 2
        assert [call["shots"] for call in client.calls] == [64, 64]
        stats = env.get_real_machine_stats()
        assert stats["submission_attempts_total"] == 2
        assert stats["max_real_submissions"] == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
