"""
量子RL调度系统 - 异步退火仿真降级失败路径测试

Issue #519: 补齐 async annealing 降级/兜底路径的测试覆盖。

测试覆盖：
- neal 不可用时优雅回退到 numpy SA
- cqlib_client 失败时回退到仿真模式
- 退火中的异常被记录而非静默吞掉
- 回退发生时 solver_type 正确设置为 "numpy_sa"
- 任何 anneal() 调用前 solver_type 为 "none"

测试风格：pytest 函数式，中文注释，使用 unittest.mock 模拟外部依赖
"""

import os
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quantum import annealing as annealing_mod
from src.quantum.annealing import QuantumAnnealingOptimizer

# 小规模 QUBO 矩阵，用于快速退火测试（2×2）
_QUBO_2X2 = np.array([[1.0, 0.5], [0.5, 1.0]], dtype=np.float64)


def test_neal_unavailable_falls_back_to_numpy_sa():
    """测试 neal 不可用时优雅回退到 numpy 模拟退火。

    通过 patch _DWAVE_AVAILABLE=False 模拟 D-Wave SDK 未安装的环境，
    验证 anneal() 不抛异常并返回合法比特串。
    """
    with patch.object(annealing_mod, "_DWAVE_AVAILABLE", False):
        opt = QuantumAnnealingOptimizer(num_qubits=4, shots=10)
        opt._sim_num_sweeps = 5  # 加速测试
        # use_dw 在 __init__ 中由 _DWAVE_AVAILABLE 决定
        assert opt.use_dw is False, "neal 不可用时 use_dw 应为 False"

        bitstring = opt.anneal(_QUBO_2X2)

        # 返回合法比特串
        assert isinstance(bitstring, str)
        assert len(bitstring) == 2
        assert set(bitstring).issubset({"0", "1"})
        # 回退到 numpy SA
        assert opt.solver_type == "numpy_sa"


def test_cqlib_client_failure_falls_back_to_simulation():
    """测试 cqlib_client 失败时回退到仿真模式。

    构造 simulation_mode=False 且 cqlib_client.submit_annealing_task 抛异常的场景，
    验证 anneal() 不传播异常，而是降级到仿真求解器。
    """
    client = MagicMock()
    client.submit_annealing_task = MagicMock(side_effect=RuntimeError("网络连接失败"))

    opt = QuantumAnnealingOptimizer(
        num_qubits=4,
        shots=10,
        simulation_mode=False,
        cqlib_client=client,
    )
    opt._sim_num_sweeps = 5  # 加速测试

    # 不应抛异常
    bitstring = opt.anneal(_QUBO_2X2)

    # 返回合法比特串
    assert isinstance(bitstring, str)
    assert len(bitstring) == 2
    assert set(bitstring).issubset({"0", "1"})
    # 回退到仿真求解器（numpy_sa 或 neal_sa，取决于 SDK 是否可用）
    assert opt.solver_type in ("numpy_sa", "neal_sa"), (
        f"回退后 solver_type 应为仿真求解器，实际: {opt.solver_type}"
    )


def test_annealing_exceptions_are_logged_not_swallowed():
    """测试退火中的异常被记录到日志而非静默吞掉。

    验证：
    1. cqlib 异常时 logger.warning 被调用，消息包含 [降级]
    2. anneal() 仍然返回结果（异常被处理，不是静默吞掉后返回 None）
    """

    class _FakeClient:
        """模拟 cqlib 客户端，submit_annealing_task 抛出异常。"""

        def submit_annealing_task(self, qubo, **kwargs):
            raise ConnectionError("cqlib 服务不可达")

    opt = QuantumAnnealingOptimizer(
        num_qubits=4,
        shots=10,
        simulation_mode=False,
        cqlib_client=_FakeClient(),
    )
    opt._sim_num_sweeps = 5

    with patch("src.quantum.annealing.logger") as mock_logger:
        bitstring = opt.anneal(_QUBO_2X2)

        # 验证 warning 日志被调用
        warning_calls = mock_logger.warning.call_args_list
        assert len(warning_calls) > 0, "异常降级时应调用 logger.warning"

        # 验证日志消息包含 [降级] 标记
        warning_msgs = [str(call.args[0]) for call in warning_calls]
        assert any("[降级]" in msg for msg in warning_msgs), (
            f"warning 日志应包含 '[降级]'，实际: {warning_msgs}"
        )

    # 验证异常未被静默吞掉——anneal 仍返回了合法结果
    assert bitstring is not None, "异常不应导致返回 None"
    assert isinstance(bitstring, str)
    assert len(bitstring) == 2


def test_solver_type_numpy_sa_when_fallback_occurs():
    """测试回退发生时 solver_type 正确设置为 "numpy_sa"。

    场景：neal 不可用（_DWAVE_AVAILABLE=False）且 cqlib 真机退火失败，
    应回退到 numpy SA，solver_type 设为 "numpy_sa"。
    """
    client = MagicMock()
    client.submit_annealing_task = MagicMock(side_effect=RuntimeError("真机退火失败"))

    with patch.object(annealing_mod, "_DWAVE_AVAILABLE", False):
        opt = QuantumAnnealingOptimizer(
            num_qubits=4,
            shots=10,
            simulation_mode=False,
            cqlib_client=client,
        )
        opt._sim_num_sweeps = 5

        opt.anneal(_QUBO_2X2)

        # neal 不可用 + cqlib 失败 → numpy_sa
        assert opt.solver_type == "numpy_sa", (
            f"回退到 numpy SA 时 solver_type 应为 'numpy_sa'，实际: {opt.solver_type}"
        )
        assert opt._last_solver == "numpy_sa"


def test_solver_type_none_before_any_anneal():
    """测试任何 anneal() 调用之前 solver_type 为 "none"。

    验证优化器刚构造完成、尚未调用 anneal() 时，solver_type 保持初始值 "none"。
    """
    opt = QuantumAnnealingOptimizer(num_qubits=4, shots=10)
    assert opt.solver_type == "none", "初始化后 solver_type 应为 'none'"
    assert opt._last_solver == "none", "初始化后 _last_solver 应为 'none'"

    # 构造但未调用 anneal，solver_type 不应改变
    opt2 = QuantumAnnealingOptimizer(
        num_qubits=8,
        shots=50,
        simulation_mode=False,
        cqlib_client=MagicMock(),
    )
    assert opt2.solver_type == "none", (
        "即使配置了 cqlib_client，未调用 anneal 前 solver_type 仍为 'none'"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
