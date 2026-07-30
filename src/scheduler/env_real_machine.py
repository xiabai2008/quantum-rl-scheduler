"""
量子任务调度环境的真机闭环模块
Real-Machine Closed-Loop for Quantum-Classical Hybrid Task Scheduling Environment

本模块封装真机闭环的核心逻辑（Issue #64），将依赖环境内部状态的
方法抽离为独立函数，便于单测与复用：
    - generate_qcis_circuit       : 根据任务参数生成 QCIS 电路
    - submit_to_real_machine      : 向真机非阻塞提交一个量子任务
    - record_real_failure         : 记录一次真机失败并在阈值时触发降级
    - poll_pending_real_tasks     : 非阻塞轮询已提交真机任务的结果
    - update_task_queue_from_real : 真机完成后回写任务队列状态

依赖关系：仅依赖 env_types.py 中的常量与数据类，不依赖 env.py。
真机函数通过 ``env`` 参数访问环境内部状态（如 _pending_real_tasks、
_real_clients 等），从而避免循环导入。
"""

import math
import os
import random
import time
from collections.abc import Mapping
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from loguru import logger

from src.scheduler.env_types import (
    REAL_FEEDBACK_SHUFFLED,
    REAL_FEEDBACK_STATUS_ONLY,
    REAL_MACHINE_DEGRADE_FAIL_THRESHOLD,
    REAL_MACHINE_FAIL_PENALTY,
    REAL_MACHINE_MAX_POLL_STEPS,
    REAL_MACHINE_SUCCESS_BONUS,
    REAL_RESULT_REWARD_MAX,
    REAL_RESULT_REWARD_MIN,
    QuantumMachine,
    RealMachineConfig,
    Task,
)

if TYPE_CHECKING:
    # 仅用于类型标注，避免运行时循环导入
    from src.api.types import TaskResult
    from src.scheduler.env import QuantumSchedulingEnv


# =============================================================================
# QCIS 电路生成
# =============================================================================

# 可用的单比特门集合（天衍-287 支持的基础门）
_SINGLE_QUBIT_GATES = ["H", "X", "Y", "Z", "RX", "RY", "RZ"]

# 可用的两比特纠缠门
_TWO_QUBIT_GATES = ["CNOT", "CZ"]

# 最大比特数（真机实际容量上限，避免生成超出硬件的电路）
_MAX_REAL_QUBITS = 287

# 免费机时包最大量子比特数（天衍云免费额度限制）
# 超过此限制的电路会触发"您的机时包最大比特数不支持本任务"错误
# .. note::
#   当前真机验证阶段使用 1 比特电路（受天衍云免费套餐限制），
#   验证的是端到端调度闭环而非大规模量子计算能力。
#   多比特电路实验是下一阶段工作（需付费套餐额度）。
#   获得付费套餐后可通过环境变量 ``FREE_TIER_MAX_QUBITS`` 调高此限制，
#   例如 ``export FREE_TIER_MAX_QUBITS=5``，无需修改代码。


def _resolve_free_tier_max_qubits() -> int:
    """从环境变量读取免费机时包最大量子比特数。

    默认值为 1（天衍云免费套餐仅支持 1-qubit 电路）。获得付费机时包后
    可通过环境变量 ``FREE_TIER_MAX_QUBITS`` 调高此限制，无需修改代码。
    无效值（非正整数或解析失败）回退到默认值 1，保证真机稳定模式。

    Returns:
        免费机时包最大量子比特数（≥1）
    """
    raw = os.environ.get("FREE_TIER_MAX_QUBITS", "1")
    try:
        value = int(raw)
        if value < 1:
            return 1
        return value
    except (ValueError, TypeError):
        return 1


FREE_TIER_MAX_QUBITS = _resolve_free_tier_max_qubits()


def generate_qcis_circuit(
    task: Task,
    max_qubits: int = _MAX_REAL_QUBITS,
    seed: int | None = None,
    two_qubit_gates: bool = False,
    circuit_type: str = "random",
) -> str:
    """根据任务参数生成适合真机执行的 QCIS 电路。

    电路结构（分层生成）：
        1. 单比特门层：每个参与比特随机选择一个基础门
        2. [可选] 纠缠层：相邻比特对之间添加 CNOT/CZ 门
        3. 测量层：所有参与比特的测量

    电路规模与任务的 qubit_count 成正比，复杂度与 priority 正相关。

    注意：天衍-176 真机上两比特门（CNOT/CZ）不稳定，Bell 态有失败率。
    默认 two_qubit_gates=False 仅生成单比特门电路，确保高成功率。

    Args:
        task            : 任务对象（含 qubit_count, priority, task_id 等）
        max_qubits      : 真机最大比特数限制（默认 287）
        seed            : 可选的随机种子（用于可复现测试）
        two_qubit_gates : 是否包含两比特纠缠门（默认 False，真机稳定模式）
        circuit_type    : 电路模板（random / bell / ghz3，默认 random）

    Returns:
        QCIS 格式的电路字符串，每行一条指令

    Examples:
        >>> t = Task(task_id="0", task_type="quantum", qubit_count=3, priority=3)
        >>> qcis = generate_qcis_circuit(t)
        >>> assert "H" in qcis or "X" in qcis
        >>> assert "M" in qcis
    """
    if max_qubits <= 0:
        raise ValueError("max_qubits must be positive")
    if circuit_type not in {"random", "bell", "ghz3"}:
        raise ValueError("circuit_type must be one of: random, bell, ghz3")

    template_qubits = {"bell": 2, "ghz3": 3}
    if circuit_type in template_qubits:
        required_qubits = template_qubits[circuit_type]
        if max_qubits < required_qubits:
            raise ValueError(
                f"{circuit_type} circuit requires at least {required_qubits} qubits, "
                f"but max_qubits={max_qubits}"
            )
        if circuit_type == "bell":
            return "H Q0\nCNOT Q0 Q1\nM Q0 Q1"
        return "H Q0\nCNOT Q0 Q1\nCNOT Q1 Q2\nM Q0 Q1 Q2"

    rng = random.Random(seed if seed is not None else hash(task.task_id))

    # 确定参与比特数：至少 1 个，不超过任务需求和真机上限
    n_qubits = max(1, min(task.qubit_count, max_qubits))

    # 复杂度因子：priority 越高，电路越深（更多门层）
    depth_factor = max(1, task.priority - 1)  # priority 1-5 → 0-4 层额外纠缠

    lines: list[str] = []

    # ── 第 1 层：单比特门 ──
    for q in range(n_qubits):
        gate = rng.choice(_SINGLE_QUBIT_GATES)
        if gate in ("RX", "RY", "RZ"):
            # 参数化旋转门：随机角度
            angle = round(rng.uniform(0, 2 * math.pi), 4)
            lines.append(f"{gate} Q{q},{angle}")
        else:
            lines.append(f"{gate} Q{q}")

    # ── 第 2 层（可重复）：纠缠层（仅当 two_qubit_gates=True）──
    if two_qubit_gates:
        for _ in range(depth_factor):
            for q in range(0, n_qubits - 1, 2):
                gate = rng.choice(_TWO_QUBIT_GATES)
                lines.append(f"{gate} Q{q} Q{q + 1}")
            # 交错对：覆盖奇数起始的比特对
            for q in range(1, n_qubits - 1, 2):
                gate = rng.choice(_TWO_QUBIT_GATES)
                lines.append(f"{gate} Q{q} Q{q + 1}")

    # ── 第 3 层：测量 ──
    for q in range(n_qubits):
        lines.append(f"M Q{q}")

    return "\n".join(lines)


# =============================================================================
# 真机测量结果解析与 reward 计算（Issue #235）
# 性能优化（Issue #525）：添加结果缓存避免重复解析
# =============================================================================

_MEASUREMENT_CACHE_TTL = 300.0
_measurement_result_cache: dict[str, tuple[float, dict[str, float]]] = {}


def _get_cached_measurement(task_id: str) -> dict[str, float] | None:
    cache_entry = _measurement_result_cache.get(task_id)
    if cache_entry is None:
        return None
    cached_time, cached_result = cache_entry
    if time.time() - cached_time > _MEASUREMENT_CACHE_TTL:
        del _measurement_result_cache[task_id]
        return None
    return cached_result


def _set_cached_measurement(task_id: str, result: dict[str, float]) -> None:
    _measurement_result_cache[task_id] = (time.time(), result)


def parse_measurement_result(
    status: "TaskResult | Mapping[str, Any]",
    task_id: str | None = None,
) -> dict[str, float]:
    """从统一任务结果或旧状态字典中解析测量概率分布。

    Mock 与 cqlib 客户端的 ``TaskResult`` 以及兼容期旧字典可能包含：
    - ``probability``: 直接的概率分布字典 {"0": 0.5, "1": 0.5}
    - ``counts``: 原始 shots 计数，在 probability 缺失时转换为概率
    - ``resultStatus``: 原始 shots 计数，需转换为概率
    - ``result``: 某些版本返回的嵌套结果

    性能优化（Issue #525）：当提供 task_id 时，结果会被缓存（TTL=5分钟），
    同一 task_id 重复调用直接返回缓存结果，避免重复解析。

    Args:
        status: get_task_status() 返回的 ``TaskResult`` 或兼容字典
        task_id: 可选的任务 ID，用于结果缓存。提供后相同 ID 直接返回缓存。

    Returns:
        归一化的概率分布字典 {"bitstring": probability}，空字典表示解析失败
    """
    if task_id is not None:
        cached = _get_cached_measurement(task_id)
        if cached is not None:
            return dict(cached)

    probability: dict[str, float] = {}

    # 路径 1: 直接的 probability 字段
    raw_prob = status.get("probability")
    if raw_prob and isinstance(raw_prob, Mapping):
        for key, val in raw_prob.items():
            try:
                probability[str(key)] = float(val)
            except (ValueError, TypeError):
                continue
        if probability:
            total = sum(probability.values())
            if total > 0:
                probability = {k: v / total for k, v in probability.items()}
            if task_id is not None:
                _set_cached_measurement(task_id, probability)
            return probability

    # 路径 2: 统一 TaskResult / Mock 的 counts 字段
    raw_counts = status.get("counts")
    if raw_counts and isinstance(raw_counts, Mapping):
        try:
            total_shots = sum(float(value) for value in raw_counts.values())
            if total_shots > 0:
                counts_result = {
                    str(key): float(value) / total_shots for key, value in raw_counts.items()
                }
                if task_id is not None:
                    _set_cached_measurement(task_id, counts_result)
                return counts_result
        except (ValueError, TypeError):
            pass

    # 路径 3: resultStatus 原始 shots 计数
    result_status = status.get("resultStatus")
    if result_status and isinstance(result_status, str):
        try:
            import json

            counts = json.loads(result_status)
            if isinstance(counts, dict):
                total_shots = sum(counts.values())
                if total_shots > 0:
                    probability = {str(k): float(v) / total_shots for k, v in counts.items()}
                    if task_id is not None:
                        _set_cached_measurement(task_id, probability)
                    return probability
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.debug(f"resultStatus JSON 解析失败: {e}")

    # 路径 4: result 字段（嵌套 probability 或直接是概率分布）
    result = status.get("result")
    if result and isinstance(result, Mapping):
        inner_prob = result.get("probability")
        if inner_prob and isinstance(inner_prob, Mapping):
            for key, val in inner_prob.items():
                try:
                    probability[str(key)] = float(val)
                except (ValueError, TypeError):
                    continue
        else:
            for key, val in result.items():
                if key in ("task_id", "status", "execution_time_s", "execute_time"):
                    continue
                try:
                    probability[str(key)] = float(val)
                except (ValueError, TypeError):
                    continue
        if probability:
            total = sum(probability.values())
            if total > 0:
                probability = {k: v / total for k, v in probability.items()}
            if task_id is not None:
                _set_cached_measurement(task_id, probability)
            return probability

    empty_result: dict[str, float] = {}
    return empty_result


@lru_cache(maxsize=1024)
def _compute_theoretical_distribution_cached(qcis: str) -> tuple[tuple[str, float], ...]:
    lines = [line.strip() for line in qcis.strip().split("\n") if line.strip()]
    gate_lines = [line for line in lines if not line.startswith("M")]
    measure_lines = [line for line in lines if line.startswith("M")]

    if not measure_lines:
        return (("0", 1.0),)

    measure_qubits: list[int] = []
    for line in measure_lines:
        parts = line.replace("M", "").strip().split()
        for p in parts:
            p = p.strip().rstrip(",")
            if p.startswith("Q"):
                try:
                    measure_qubits.append(int(p[1:]))
                except ValueError:
                    continue
    n_qubits = max(1, len(set(measure_qubits)))

    all_qubit_refs: set[int] = set()
    for line in gate_lines:
        tokens = line.replace(",", " ").split()
        for tok in tokens[1:]:
            tok = tok.strip()
            if tok.startswith("Q"):
                try:
                    all_qubit_refs.add(int(tok[1:]))
                except ValueError:
                    continue
    total_qubits = max(n_qubits, max(all_qubit_refs) + 1 if all_qubit_refs else n_qubits)

    if total_qubits > 3:
        n_states = 2**n_qubits
        return tuple((format(i, f"0{n_qubits}b"), 1.0 / n_states) for i in range(n_states))

    try:
        raw_dist = _simulate_qcis_statevector(qcis, total_qubits, measure_qubits)
        return tuple((k, round(v, 12)) for k, v in raw_dist.items())
    except Exception as e:
        logger.warning(f"状态向量模拟失败({e})，回退均匀分布")
        n_states = 2**n_qubits
        return tuple((format(i, f"0{n_qubits}b"), 1.0 / n_states) for i in range(n_states))


def compute_theoretical_distribution(qcis: str) -> dict[str, float]:
    """根据 QCIS 电路计算理论概率分布（用于保真度对比，Issue #405 修复）。

    使用 numpy 状态向量模拟精确计算 1-3 比特电路的理论概率分布，
    支持 H/X/Y/Z/RX/RY/RZ/S/T/CNOT/CZ 等常见门。4 比特及以上电路
    回退到均匀分布近似（保守估计）。

    性能优化（Issue #525）：使用 LRU 缓存相同 QCIS 电路的计算结果，
    最多缓存 1024 个不同电路，避免重复进行状态向量模拟。

    Args:
        qcis: QCIS 格式电路字符串

    Returns:
        理论概率分布字典（键为测量比特串，值为概率）
    """
    cached_tuple = _compute_theoretical_distribution_cached(qcis)
    return dict(cached_tuple)


def _simulate_qcis_statevector(
    qcis: str,
    n_qubits: int,
    measure_qubits: list[int],
) -> dict[str, float]:
    """使用 numpy 精确模拟 1-2 比特 QCIS 电路，返回测量概率分布。"""
    import numpy as np

    I2 = np.eye(2, dtype=np.complex128)  # noqa: N806
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)  # noqa: N806
    Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)  # noqa: N806
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)  # noqa: N806
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)  # noqa: N806
    S = np.array([[1, 0], [0, 1j]], dtype=np.complex128)  # noqa: N806
    SDG = np.array([[1, 0], [0, -1j]], dtype=np.complex128)  # noqa: N806
    T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=np.complex128)  # noqa: N806

    def _kron_n(ops: list[np.ndarray]) -> np.ndarray:
        result = ops[0]
        for op in ops[1:]:
            result = np.kron(result, op)
        return result

    def _single_qubit_gate(op: np.ndarray, target: int) -> np.ndarray:
        ops = [I2] * n_qubits
        ops[target] = op
        return _kron_n(ops)

    def _cnot(control: int, target: int) -> np.ndarray:
        dim = 2**n_qubits
        mat = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(dim):
            bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
            c_bit = bits[control]
            out_bits = bits[:]
            if c_bit == 1:
                out_bits[target] = 1 - out_bits[target]
            j = 0
            for q in range(n_qubits):
                j = (j << 1) | out_bits[q]
            mat[j, i] = 1.0
        return mat

    def _cz(control: int, target: int) -> np.ndarray:
        dim = 2**n_qubits
        mat = np.eye(dim, dtype=np.complex128)
        for i in range(dim):
            bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
            if bits[control] == 1 and bits[target] == 1:
                mat[i, i] = -1.0
        return mat

    dim = 2**n_qubits
    state = np.zeros(dim, dtype=np.complex128)
    state[0] = 1.0

    lines = [ln.strip() for ln in qcis.strip().split("\n") if ln.strip()]
    for line in lines:
        if line.startswith("M"):
            continue
        parts = line.replace(",", " ").split()
        if not parts:
            continue
        gate = parts[0]
        qubits: list[int] = []
        param: float | None = None
        for tok in parts[1:]:
            tok = tok.strip()
            if tok.startswith("Q"):
                try:
                    qubits.append(int(tok[1:]))
                except ValueError:
                    continue
            else:
                try:
                    param = float(tok)
                except ValueError:
                    continue

        if gate in ("H", "X", "Y", "Z") and len(qubits) == 1:
            op_map = {"H": H, "X": X, "Y": Y, "Z": Z}
            state = _single_qubit_gate(op_map[gate], qubits[0]) @ state
        elif gate == "S" and len(qubits) == 1:
            state = _single_qubit_gate(S, qubits[0]) @ state
        elif gate in ("SDG", "SDAG") and len(qubits) == 1:
            state = _single_qubit_gate(SDG, qubits[0]) @ state
        elif gate == "T" and len(qubits) == 1:
            state = _single_qubit_gate(T, qubits[0]) @ state
        elif gate in ("RX", "RY", "RZ") and len(qubits) == 1 and param is not None:
            theta = param
            if gate == "RX":
                op = np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * X
            elif gate == "RY":
                op = np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * Y
            else:
                op = np.cos(theta / 2) * I2 - 1j * np.sin(theta / 2) * Z
            state = _single_qubit_gate(op, qubits[0]) @ state
        elif gate == "CNOT" and len(qubits) == 2:
            state = _cnot(qubits[0], qubits[1]) @ state
        elif gate == "CZ" and len(qubits) == 2:
            state = _cz(qubits[0], qubits[1]) @ state

    probs = np.abs(state) ** 2
    measured_set = sorted(set(measure_qubits))
    result: dict[str, float] = {}
    for i in range(dim):
        bits = [(i >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        key = "".join(str(bits[q]) for q in measured_set)
        result[key] = result.get(key, 0.0) + float(probs[i])

    return {k: v for k, v in result.items() if v > 1e-10}


def compute_result_fidelity(
    measured: dict[str, float],
    theoretical: dict[str, float],
) -> float:
    """计算测量分布与理论分布之间的保真度（classical fidelity）。

    F(p, q) = (sum_i sqrt(p_i * q_i))^2

    保真度范围 [0, 1]，1 表示完美匹配。

    Args:
        measured: 真机测量得到的概率分布
        theoretical: 理论计算的概率分布

    Returns:
        保真度 [0, 1]，0 表示解析失败
    """
    if not measured or not theoretical:
        return 0.0

    # 对齐两个分布的键空间
    all_keys = set(measured.keys()) | set(theoretical.keys())
    fidelity_sum = 0.0
    for key in all_keys:
        p = measured.get(key, 0.0)
        q = theoretical.get(key, 0.0)
        fidelity_sum += (p * q) ** 0.5

    fidelity = fidelity_sum**2
    return float(max(0.0, min(1.0, fidelity)))


def compute_real_result_reward(
    measured: dict[str, float],
    theoretical: dict[str, float],
) -> tuple[float, float, str]:
    """根据真机测量结果计算质量感知 reward（Issue #235）。

    reward 公式：
        quality = fidelity(measured, theoretical)
        reward = REAL_RESULT_REWARD_MIN + quality * (REAL_RESULT_REWARD_MAX - REAL_RESULT_REWARD_MIN)

    线性映射：quality=0 → reward=0.0（失败不给奖励），quality=1 → reward=5.0。
    这使得真机测量结果的质量直接影响力学习，而非仅靠 completed 状态。

    Args:
        measured: 真机测量得到的概率分布
        theoretical: 理论计算的概率分布

    Returns:
        (reward, fidelity, formula_str) 三元组：
        - reward: 计算得到的奖励值
        - fidelity: 保真度 [0, 1]
        - formula_str: 可追溯的计算公式描述
    """
    if not measured:
        # 测量结果解析失败，给最小奖励（仅证明平台可用）
        fidelity = 0.0
        reward = REAL_RESULT_REWARD_MIN
        formula = f"reward={REAL_RESULT_REWARD_MIN:.1f} (measurement_parse_failed, fidelity=0)"
    else:
        fidelity = compute_result_fidelity(measured, theoretical)
        quality_range = REAL_RESULT_REWARD_MAX - REAL_RESULT_REWARD_MIN
        reward = REAL_RESULT_REWARD_MIN + fidelity * quality_range
        formula = (
            f"reward={reward:.4f} = {REAL_RESULT_REWARD_MIN:.1f} + "
            f"fidelity({fidelity:.4f}) * {quality_range:.1f}"
        )

    return float(reward), fidelity, formula


def shuffle_measurement(measured: dict[str, float]) -> dict[str, float]:
    """打乱测量结果的概率分布（消融对照组，Issue #235）。

    保留概率值但随机分配到不同的 bitstring 键上，
    破坏测量结果与任务目标之间的语义关联。
    如果打乱后的分布恰好和原始分布相同（极低概率），重新打乱。

    Args:
        measured: 原始测量概率分布

    Returns:
        打乱后的概率分布（值不变，键重新分配）
    """
    if not measured or len(measured) <= 1:
        return dict(measured)

    keys = list(measured.keys())
    values = list(measured.values())
    shuffled = dict(measured)

    # 尝试打乱，确保结果与原始不同（最多重试 10 次）
    for _ in range(10):
        random.shuffle(values)
        shuffled = dict(zip(keys, values, strict=True))
        # 检查是否确实发生了变化
        if any(shuffled[k] != measured[k] for k in keys):
            break

    return shuffled


# =============================================================================
# 真机提交与轮询
# =============================================================================


def submit_to_real_machine(
    env: "QuantumSchedulingEnv",
    machine: QuantumMachine,
    task: Task,
    rl_action: int = -1,
    rl_action_prob: float = 0.0,
    observation_snapshot: dict[str, Any] | None = None,
) -> None:
    """
    向真机提交一个量子任务（非阻塞，异常安全）。

    真机提交在仿真循环中是非阻塞的：提交后立即返回 task_id 并登记到
    ``env._pending_real_tasks``，后续 step() 通过 ``poll_pending_real_tasks``
    轮询结果，避免阻塞 RL 训练。

    降级机制（Issue #64）：当 ``env._real_machine_degraded=True`` 时跳过提交，
    真机不可用时自动 fallback 到 Mock（仅计入仿真统计）。

    RL 动作上下文（Issue #234）：pending 记录中增加 ``rl_action``、
    ``rl_action_prob``、``observation_snapshot``、``machine_score`` 字段，
    建立"RL决策→真机任务"的因果链。

    Args:
        env                 : 调度环境实例（提供真机客户端、pending 列表等内部状态）
        machine             : 目标真机
        task                : 待提交任务
        rl_action           : RL 动作类型（0=classical, 1=quantum, 2=hybrid，默认 -1 表示未知）
        rl_action_prob      : 该动作被选择的概率（默认 0.0）
        observation_snapshot: 观测向量摘要（队列长度、机器负载等关键字段，非完整向量）
    """
    # 降级保护：已知真机不可用时直接返回，不再消耗机时
    if env._real_machine_degraded:
        return

    if (
        env.max_real_submissions is not None
        and env._real_submission_attempts_total >= env.max_real_submissions
    ):
        return

    client = env._real_clients.get(machine.name)
    if client is None:
        return

    # 在真正调用 SDK 前计数；无论平台接受或拒绝，该调用都占用硬上限。
    env._real_submission_attempts_total += 1

    # 优先使用 task.qcis（由 parser 生成），否则动态生成电路
    # 注意：免费机时包有量子比特数限制（FREE_TIER_MAX_QUBITS），
    # 超限电路会触发"您的机时包最大比特数不支持本任务"错误，
    # 因此生成电路时强制限制比特数，避免容量错误触发降级
    qcis = getattr(task, "qcis", None)
    if not qcis:
        qcis = generate_qcis_circuit(
            task,
            max_qubits=min(
                machine.total_qubits,
                getattr(env, "real_machine_max_qubits", FREE_TIER_MAX_QUBITS),
            ),
        )

    try:
        real_task_id = client.submit_quantum_task(
            qcis=qcis,
            shots=env.real_machine_shots,
            task_name=f"RL_{task.task_id}",
        )
        env._machine_real_submits[machine.name] = env._machine_real_submits.get(machine.name, 0) + 1
        # 登记到 pending 列表，后续轮询结果（Issue #64）
        # real_task_id 为 None 表示提交被拒绝（如机器校准中），计入失败
        if real_task_id is not None:
            # 计算 machine_score（与 select_best_machine 评分公式一致）
            machine_score = (
                machine.fidelity * machine.available_ratio / (1.0 + machine.quantum_queue)
            )
            env._pending_real_tasks.append(
                {
                    "task_id": str(real_task_id),
                    "machine_name": machine.name,
                    "submit_step": env._current_step,
                    "poll_count": 0,
                    "task_id_str": str(task.task_id),
                    "qcis_circuit": qcis,
                    # RL 动作上下文（Issue #234）
                    "rl_action": rl_action,
                    "rl_action_prob": rl_action_prob,
                    "observation_snapshot": observation_snapshot or {},
                    "machine_score": float(machine_score),
                }
            )
            if env.use_real_machine:
                logger.debug(
                    f"[真机闭环] 任务 {task.task_id} 已提交 {machine.name} "
                    f"(real_task_id={real_task_id})，等待结果轮询"
                )
        else:
            # 提交被拒绝（非异常），计入失败并触发降级判断
            record_real_failure(env, machine.name, "提交被拒绝（返回 None）")
    except Exception as e:
        # 真机 API 提交失败：区分暂时性错误与永久性错误（Issue #218）
        # - 暂时性错误（网络超时/连接错误/服务端 5xx）：不计入连续失败，仅记录日志，
        #   避免网络抖动误触发降级
        # - 永久性错误（认证失败/参数错误）：计入连续失败，可能触发降级
        from src.exceptions import is_transient_exception

        if is_transient_exception(e):
            logger.warning(
                f"[真机] {machine.name} 提交遭遇暂时性错误，不计入连续失败: {type(e).__name__}: {e}"
            )
            env._render_log.append(f"[真机] {machine.name} 暂时性错误（已忽略）: {str(e)[:60]}")
        else:
            logger.error(f"[真机] {machine.name} 提交失败: {e}")
            env._render_log.append(f"[真机] {machine.name} 提交失败: {str(e)[:60]}")
            record_real_failure(env, machine.name, f"提交异常: {str(e)[:60]}")


def record_real_failure(
    env: "QuantumSchedulingEnv",
    machine_name: str,
    reason: str,
) -> None:
    """
    记录一次真机失败，并在达到阈值时触发降级（Issue #64）。

    连续失败次数达到 ``REAL_MACHINE_DEGRADE_FAIL_THRESHOLD`` 时，将
    ``env._real_machine_degraded`` 置为 True，后续真机提交将被跳过。

    Args:
        env          : 调度环境实例
        machine_name : 失败的机器名
        reason       : 失败原因（用于日志）
    """
    env._real_fail_count += 1
    env._real_consecutive_failures += 1
    if (
        env._real_consecutive_failures >= REAL_MACHINE_DEGRADE_FAIL_THRESHOLD
        and not env._real_machine_degraded
    ):
        env._real_machine_degraded = True
        logger.warning(
            f"[真机闭环] 连续失败 {env._real_consecutive_failures} 次，"
            f"已自动降级到 Mock 模式（最后失败: {machine_name} - {reason}）"
        )
        env._render_log.append(
            f"[真机闭环] 已降级到 Mock（连续失败 {env._real_consecutive_failures} 次）"
        )


def poll_pending_real_tasks(env: "QuantumSchedulingEnv") -> float:
    """
    非阻塞轮询已提交真机任务的结果，返回本步反馈 reward（Issue #64, #524）。

    Issue #524 优化：不再在单次 step() 中轮询所有 pending 任务（会导致同步阻塞，
    每个任务最多阻塞2秒），而是采用轮转队列策略：
        - 每步最多轮询 ``max_poll_per_step`` 个任务（默认1），将阻塞时间控制在单次网络请求延迟内
        - 轮询后仍在运行的任务移到 pending 队列末尾，等待后续步轮询
        - 任务级状态缓存：同一步内不重复轮询同一任务（防御性检查）

    对被轮询的任务调用 ``get_task_status`` 查询状态：
        - completed : 计入成功，返回 REAL_MACHINE_SUCCESS_BONUS
        - error     : 计入失败，返回 REAL_MACHINE_FAIL_PENALTY，触发降级判断
        - timeout   : 轮询次数超过 REAL_MACHINE_MAX_POLL_STEPS，视为超时失败
        - running/unknown : poll_count +1，移到队尾等待下一轮轮询

    所有反馈乘以 ``env.real_machine_feedback_weight`` 后累加返回。

    Args:
        env: 调度环境实例

    Returns:
        本步真机反馈 reward（正为成功加成，负为失败惩罚，0 表示无新结果）
    """
    if not env._pending_real_tasks:
        return 0.0

    max_poll = getattr(env, "max_poll_per_step", 1)
    n_pending = len(env._pending_real_tasks)
    n_to_poll = min(max_poll, n_pending)

    to_poll = env._pending_real_tasks[:n_to_poll]
    remaining = env._pending_real_tasks[n_to_poll:]

    total_feedback = 0.0
    still_pending: list[dict[str, Any]] = []

    for pending in to_poll:
        pending["poll_count"] = pending.get("poll_count", 0) + 1
        machine_name = pending["machine_name"]
        real_task_id = pending["task_id"]
        task_id_str = pending["task_id_str"]
        client = env._real_clients.get(machine_name)

        pending["last_poll_step"] = env._current_step

        # 客户端丢失（理论上不应发生），视为失败
        if client is None:
            total_feedback += REAL_MACHINE_FAIL_PENALTY * env.real_machine_feedback_weight
            _record_causal_feedback(env, pending, {}, REAL_MACHINE_FAIL_PENALTY, -1.0, "", "failed")
            record_real_failure(env, machine_name, "客户端丢失")
            continue

        try:
            status = client.get_task_status(real_task_id)
            pending["cached_status"] = dict(status) if status else {}
        except Exception as e:
            # 查询异常视为本步未拿到结果，移到队尾等待下次轮询
            logger.debug(f"[真机闭环] 查询 {real_task_id} 异常: {e}")
            pending["cached_status"] = {"status": "unknown"}
            still_pending.append(pending)
            continue

        status_str = str(status.get("status", "unknown"))

        if status_str == "completed":
            # 真机成功：根据反馈模式计算 reward（Issue #235）
            reward_delta, fidelity, formula, measured, _theoretical = _compute_real_feedback(
                env, pending, status
            )
            total_feedback += reward_delta * env.real_machine_feedback_weight
            env._real_success_count += 1
            env._real_consecutive_failures = 0  # 成功重置连续失败计数

            # Issue #577: 触发噪声感知奖励反馈（保真度→后续步惩罚/加成闭环）
            trigger_noise_aware_feedback(env, fidelity)

            # 记录详细结果元数据（Issue #235 可追溯性），复用已解析的 measured
            _record_real_result(env, pending, status, reward_delta, fidelity, formula, measured)

            # 写入因果记录到 _real_feedback_log（Issue #235），复用已解析的 measured
            _record_causal_feedback(
                env, pending, status, reward_delta, fidelity, formula, "completed", measured
            )

            # 真机执行时间回写队列（Issue #64 增强）
            actual_duration = status.get("execution_time_s", None)
            _update_task_duration(env, task_id_str, actual_duration)

            logger.debug(
                f"[真机闭环] 任务 {task_id_str} 真机执行成功 "
                f"(machine={machine_name}, real_task_id={real_task_id}, "
                f"fidelity={fidelity:.4f}, reward={reward_delta:.4f})"
            )
        elif status_str == "error":
            # 真机失败：负向反馈 + 降级判断
            total_feedback += REAL_MACHINE_FAIL_PENALTY * env.real_machine_feedback_weight
            _record_causal_feedback(
                env, pending, status, REAL_MACHINE_FAIL_PENALTY, -1.0, "", "failed"
            )
            record_real_failure(env, machine_name, "任务状态=error")
        elif status_str == "query_error":
            # 查询失败：连续3次后视为失败，避免无效轮询（Issue #407）
            pending["query_fail_count"] = pending.get("query_fail_count", 0) + 1
            if pending["query_fail_count"] >= 3:
                total_feedback += REAL_MACHINE_FAIL_PENALTY * env.real_machine_feedback_weight
                _record_causal_feedback(
                    env, pending, status, REAL_MACHINE_FAIL_PENALTY, -1.0, "", "query_error"
                )
                record_real_failure(env, machine_name, "连续查询失败(query_error)")
                logger.debug(
                    f"[真机闭环] 任务 {task_id_str} 连续查询失败 "
                    f"(query_fail_count={pending['query_fail_count']})"
                )
            else:
                still_pending.append(pending)
        elif pending["poll_count"] >= REAL_MACHINE_MAX_POLL_STEPS:
            # 超时：视为失败
            total_feedback += REAL_MACHINE_FAIL_PENALTY * env.real_machine_feedback_weight
            _record_causal_feedback(
                env, pending, status, REAL_MACHINE_FAIL_PENALTY, -1.0, "", "timeout"
            )
            record_real_failure(env, machine_name, "轮询超时")
            logger.debug(
                f"[真机闭环] 任务 {task_id_str} 轮询超时 (poll_count={pending['poll_count']})"
            )
        else:
            # 仍在运行，移到队尾等待后续步轮询
            still_pending.append(pending)

    # 新队列 = 未轮询的任务（保持在前） + 轮询后仍 pending 的任务（移到队尾）
    env._pending_real_tasks = remaining + still_pending
    return total_feedback


# =============================================================================
# 真机反馈计算与结果记录（Issue #235）
# =============================================================================


def _compute_real_feedback(
    env: "QuantumSchedulingEnv",
    pending: dict[str, Any],
    status: dict[str, Any],
) -> tuple[float, float, str, dict[str, float], dict[str, float]]:
    """根据真机反馈模式计算 reward（Issue #235）。

    三种模式：
    - status_only  : 固定 bonus（旧行为）
    - result_aware : 解析测量分布，按保真度计算 reward
    - shuffled     : 打乱测量结果后按保真度计算（消融对照）

    性能优化（Issue #525）：返回解析后的 measured 和 theoretical 分布，
    供调用方复用，避免重复解析和计算。

    Args:
        env     : 调度环境实例
        pending : pending 任务记录（含 qcis_circuit, task_id）
        status  : get_task_status() 返回的状态

    Returns:
        (reward, fidelity, formula_str, measured, theoretical) 五元组
    """
    mode = getattr(env, "real_feedback_mode", REAL_FEEDBACK_STATUS_ONLY)
    real_task_id = pending.get("task_id")

    if mode == REAL_FEEDBACK_STATUS_ONLY:
        # 旧行为：固定 bonus，不解析测量结果
        return (
            REAL_MACHINE_SUCCESS_BONUS,
            -1.0,
            f"reward={REAL_MACHINE_SUCCESS_BONUS:.1f} (status_only, fixed bonus)",
            {},
            {},
        )

    # result_aware 或 shuffled 模式：解析测量结果（传入 task_id 利用缓存）
    original_measured = parse_measurement_result(status, task_id=real_task_id)
    qcis = pending.get("qcis_circuit", "H Q0\nM Q0")
    theoretical = compute_theoretical_distribution(qcis)

    reward_measured = original_measured
    if mode == REAL_FEEDBACK_SHUFFLED:
        # 打乱测量结果（消融对照）：reward 计算用打乱后的，但记录用原始结果
        reward_measured = shuffle_measurement(original_measured)

    reward, fidelity, formula = compute_real_result_reward(reward_measured, theoretical)

    if mode == REAL_FEEDBACK_SHUFFLED:
        formula += " [SHUFFLED]"

    return reward, fidelity, formula, original_measured, theoretical


def _record_real_result(
    env: "QuantumSchedulingEnv",
    pending: dict[str, Any],
    status: dict[str, Any],
    reward_delta: float,
    fidelity: float,
    formula: str,
    measured: dict[str, float] | None = None,
) -> None:
    """记录真机结果的详细元数据（Issue #235 可追溯性）。

    每条记录包含 task_id、circuit_hash、backend、shots、counts/probability、
    objective_value、result_valid、fallback_mode、reward_delta 及计算公式。

    性能优化（Issue #525）：支持传入已解析的 measured 分布，避免重复解析。

    Args:
        env          : 调度环境实例
        pending      : pending 任务记录
        status       : 真机返回的状态字典
        reward_delta : 实际 reward 增量
        fidelity     : 保真度（-1 表示未计算）
        formula      : 计算公式描述
        measured     : 已解析的测量概率分布（可选，传入则复用）
    """
    if not hasattr(env, "_real_result_records"):
        env._real_result_records = []

    if measured is not None:
        parsed_measured = measured
    elif fidelity >= 0:
        real_task_id = pending.get("task_id")
        parsed_measured = parse_measurement_result(status, task_id=real_task_id)
    else:
        parsed_measured = {}
    mode = getattr(env, "real_feedback_mode", REAL_FEEDBACK_STATUS_ONLY)

    record: dict[str, Any] = {
        "task_id": pending.get("task_id_str", ""),
        "real_task_id": pending.get("task_id", ""),
        "machine": pending.get("machine_name", ""),
        "submit_step": pending.get("submit_step", 0),
        "complete_step": env._current_step,
        "shots": env.real_machine_shots,
        "backend": pending.get("machine_name", ""),
        "feedback_mode": mode,
        "probability": parsed_measured,
        "fidelity": round(fidelity, 6) if fidelity >= 0 else None,
        "reward_delta": round(reward_delta, 6),
        "formula": formula,
        "result_valid": len(parsed_measured) > 0,
        "fallback_mode": mode == REAL_FEEDBACK_STATUS_ONLY and len(parsed_measured) == 0,
    }

    env._real_result_records.append(record)


def _record_causal_feedback(
    env: "QuantumSchedulingEnv",
    pending: dict[str, Any],
    status: dict[str, Any],
    reward: float,
    fidelity: float,
    formula: str,
    outcome: str,
    measured: dict[str, float] | None = None,
) -> None:
    """写入完整因果记录到 ``env._real_feedback_log``（Issue #235）。

    记录"RL动作→真机任务→结果→reward"的完整因果链，每条记录包含：
    - 提交步数 / 完成步数
    - RL 动作类型和概率（来自 Issue #234 的 pending 记录）
    - 任务 ID / 真机任务 ID / 机器名
    - QCIS 电路 / 测量概率分布 / 保真度
    - reward / 计算公式 / 结果状态

    在成功、失败、超时场景下均写入记录（通过 ``outcome`` 字段区分）。

    性能优化（Issue #525）：支持传入已解析的 measured 分布，避免重复解析。

    Args:
        env      : 调度环境实例
        pending  : pending 任务记录（含 rl_action, rl_action_prob 等字段）
        status   : 真机返回的状态字典（失败时可为空字典）
        reward   : 实际 reward 值
        fidelity : 保真度（-1 表示未计算）
        formula  : 计算公式描述（失败时为空字符串）
        outcome  : 结果状态："completed" / "failed" / "timeout"
        measured : 已解析的测量概率分布（可选，传入则复用）
    """
    if not hasattr(env, "_real_feedback_log"):
        env._real_feedback_log = []

    if measured is not None:
        parsed_measured = measured
    elif fidelity >= 0 and status:
        real_task_id = pending.get("task_id")
        parsed_measured = parse_measurement_result(status, task_id=real_task_id)
    else:
        parsed_measured = {}

    record: dict[str, Any] = {
        "submit_step": pending.get("submit_step", 0),
        "complete_step": env._current_step,
        "rl_action": pending.get("rl_action", -1),
        "rl_action_prob": pending.get("rl_action_prob", 0.0),
        "task_id": pending.get("task_id_str", ""),
        "real_task_id": pending.get("task_id", ""),
        "machine_name": pending.get("machine_name", ""),
        "qcis_circuit": pending.get("qcis_circuit", ""),
        "machine_score": pending.get("machine_score", 0.0),
        "observation_snapshot": pending.get("observation_snapshot", {}),
        "measured_prob": parsed_measured,
        "fidelity": round(fidelity, 6) if fidelity >= 0 else None,
        "reward": round(reward, 6),
        "formula": formula,
        "outcome": outcome,
    }

    env._real_feedback_log.append(record)


# =============================================================================
# 真机执行时间回写
# =============================================================================


def _update_task_duration(
    env: "QuantumSchedulingEnv",
    task_id_str: str,
    actual_execution_s: float | None,
) -> None:
    """根据真机实际执行时间更新队列中任务的剩余执行时间。

    当任务在真机上实际完成后，需要在全局任务队列中找到该任务
    并将其 remaining_time 置 0 标记为已完成。
    已完成任务会在下一次仿真时间推进中从队列移除。

    这使得仿真队列进度与真机实际进度对齐，实现真正的闭环反馈。
    如果找不到任务（可能已经完成并移除），静默忽略。

    Args:
        env               : 调度环境实例（访问 _task_queue 和 _current_task）
        task_id_str       : 任务 ID（字符串）
        actual_execution_s: 真机实际执行时间（秒），None 表示无数据
    """
    if actual_execution_s is None:
        return

    # 1. 检查当前正在执行的任务
    if env._current_task is not None and str(env._current_task.task_id) == task_id_str:
        env._current_task.execution_time = 0
        logger.debug(
            f"[真机闭环] 回写当前任务 {task_id_str} 实际执行 {actual_execution_s:.2f}s → 标记完成"
        )
        return

    # 2. 检查全局任务队列
    for task in env._task_queue:
        if str(task.task_id) == task_id_str:
            task.execution_time = 0
            logger.debug(
                f"[真机闭环] 回写队列任务 {task_id_str} "
                f"实际执行 {actual_execution_s:.2f}s → 标记完成"
            )
            return

    # 3. 找不到任务（已经被移除），不报错
    logger.debug(f"[真机闭环] 回写任务 {task_id_str} 找不到，已完成移除")


# =============================================================================
# 噪声感知奖励整形（Issue #577）
# =============================================================================


def init_noise_aware_state(env: "QuantumSchedulingEnv") -> None:
    """初始化噪声感知奖励状态（在环境 reset 时调用）。

    状态机：
        - 新触发的值存在 _noise_aware_pending_value，标记 _has_pending=True
        - 每步开头 advance 时：
            * 如果有 pending：激活为 current_value，设置 decay_remaining=steps-1
            * 否则如果 decay_remaining>0：current_value *= decay, decay_remaining-=1
            * 否则：current_value=0

    Args:
        env: 调度环境实例
    """
    env._noise_aware_current_value = 0.0
    env._noise_aware_decay_remaining = 0
    env._noise_aware_pending_value = 0.0
    env._noise_aware_has_pending = False
    env._noise_aware_last_fidelity = 0.0
    env._noise_aware_trigger_step = -1


def _get_noise_config(env: "QuantumSchedulingEnv") -> RealMachineConfig:
    """获取噪声感知奖励配置，兼容无配置的旧环境。

    Args:
        env: 调度环境实例

    Returns:
        RealMachineConfig 配置对象
    """
    config = getattr(env, "real_machine_config", None)
    if config is not None and isinstance(config, RealMachineConfig):
        return config
    return RealMachineConfig()


def trigger_noise_aware_feedback(
    env: "QuantumSchedulingEnv",
    fidelity: float,
) -> None:
    """当真机任务完成时，根据保真度触发噪声感知奖励反馈。

    逻辑：
        - fidelity < penalty_threshold: 施加负惩罚，强度与 (threshold - fidelity) 成正比
        - fidelity > bonus_threshold: 施加正加成，强度与 (fidelity - threshold) 成正比
        - penalty_threshold <= fidelity <= bonus_threshold: 无调整
        - fidelity < 0（如 status_only 模式未计算保真度）：不触发

    新触发的值不会影响当前步（当前步奖励已计算），将从下一步开始生效。
    惩罚/加成采用指数衰减：第一步使用完整强度，后续 N-1 步每步乘以 decay_factor。

    Args:
        env     : 调度环境实例
        fidelity: 真机测量保真度 [0, 1]，-1 表示未计算
    """
    if fidelity < 0:
        return

    config = _get_noise_config(env)
    if not config.noise_aware_reward:
        return

    if not hasattr(env, "_noise_aware_has_pending"):
        init_noise_aware_state(env)

    steps = config.noise_penalty_steps
    if fidelity < config.noise_penalty_threshold:
        severity = config.noise_penalty_threshold - fidelity
        magnitude = config.noise_penalty_strength * severity
        env._noise_aware_pending_value = -magnitude
        env._noise_aware_has_pending = True
        env._noise_aware_last_fidelity = fidelity
        env._noise_aware_trigger_step = env._current_step
        logger.debug(
            f"[噪声感知] 低保真度触发惩罚: fidelity={fidelity:.4f} "
            f"< threshold={config.noise_penalty_threshold:.2f}, "
            f"penalty={-magnitude:.4f}, steps={steps}"
        )
    elif fidelity > config.noise_bonus_threshold:
        quality = fidelity - config.noise_bonus_threshold
        magnitude = config.noise_bonus_strength * quality
        env._noise_aware_pending_value = magnitude
        env._noise_aware_has_pending = True
        env._noise_aware_last_fidelity = fidelity
        env._noise_aware_trigger_step = env._current_step
        logger.debug(
            f"[噪声感知] 高保真度触发加成: fidelity={fidelity:.4f} "
            f"> threshold={config.noise_bonus_threshold:.2f}, "
            f"bonus={magnitude:.4f}, steps={steps}"
        )


def advance_noise_aware_to_next_step(env: "QuantumSchedulingEnv") -> None:
    """在每步开始时调用，将噪声感知状态推进到当前步。

    状态转换逻辑：
        1. 如果有新触发的 pending 值（真机刚返回结果）：
           - current_value = pending_value（完整强度）
           - decay_remaining = steps - 1（本步用完整值，还有 steps-1 次衰减）
           - 清空 pending
        2. 否则如果 decay_remaining > 0：
           - current_value *= decay_factor
           - decay_remaining -= 1
        3. 否则：
           - current_value = 0

    时序示例（steps=5, decay=0.7, 触发值=-X）：
        - 触发后下一步 advance: current=-X, decay_remaining=4  （第1步：完整值）
        - 再下一步: current=-X*0.7=-0.7X, decay_remaining=3      （第2步）
        - 再下一步: current=-0.49X, decay_remaining=2            （第3步）
        - 再下一步: current=-0.343X, decay_remaining=1           （第4步）
        - 再下一步: current=-0.240X, decay_remaining=0           （第5步）
        - 再下一步: current=0                                    （结束）

    Args:
        env: 调度环境实例
    """
    if not hasattr(env, "_noise_aware_has_pending"):
        init_noise_aware_state(env)

    config = _get_noise_config(env)
    if not config.noise_aware_reward:
        env._noise_aware_current_value = 0.0
        env._noise_aware_decay_remaining = 0
        env._noise_aware_has_pending = False
        return

    # 1. 优先激活新触发的 pending 值（覆盖当前衰减中的值，新结果优先）
    if env._noise_aware_has_pending:
        env._noise_aware_current_value = env._noise_aware_pending_value
        env._noise_aware_decay_remaining = config.noise_penalty_steps - 1
        env._noise_aware_pending_value = 0.0
        env._noise_aware_has_pending = False
        return

    # 2. 正在衰减中：指数衰减
    if env._noise_aware_decay_remaining > 0:
        env._noise_aware_current_value *= config.noise_decay_factor
        env._noise_aware_decay_remaining -= 1
    else:
        # 3. 衰减结束，归零
        env._noise_aware_current_value = 0.0


def get_noise_aware_adjustment(
    env: "QuantumSchedulingEnv",
    action: int,
) -> float:
    """获取当前步的噪声感知奖励调整值（已通过动作类型过滤）。

    仅对量子相关动作（ACTION_QUANTUM, ACTION_QUANTUM_QEM, ACTION_HYBRID）施加调整；
    经典动作（ACTION_CLASSICAL）不受噪声影响，返回 0。

    注意：必须在 advance_noise_aware_to_next_step() 之后调用，否则值未刷新。

    Args:
        env   : 调度环境实例
        action: 当前调度动作

    Returns:
        奖励调整值（负为惩罚，正为加成，0 为无调整）
    """
    from src.scheduler.env_types import ACTION_CLASSICAL as _ACTION_CLASSICAL

    if action == _ACTION_CLASSICAL:
        return 0.0

    if not hasattr(env, "_noise_aware_current_value"):
        init_noise_aware_state(env)

    config = _get_noise_config(env)
    if not config.noise_aware_reward:
        return 0.0

    return float(env._noise_aware_current_value)


# =============================================================================
# 噪声模型提取与注入（Issue #579）
# =============================================================================


def attach_noise_extractor(
    env: "QuantumSchedulingEnv",
    backend: Any = None,
    num_qubits: int = 20,
    seed: int | None = None,
) -> dict[str, Any] | None:
    """创建 NoiseModelExtractor 并将噪声画像注入环境（Issue #579）。

    在 ``attach_real_clients`` 后调用，从真机后端（或 Mock）提取噪声参数
    并注入到环境的量子机器中，驱动仿真环境的噪声特征。

    当 ``backend`` 为 None 或后端不可用时，自动降级为 Mock 模式，
    返回基于真实超导量子比特统计分布的仿真噪声数据。

    Args:
        env        : 调度环境实例（需已初始化 _machines）
        backend    : 量子硬件后端实例（``QuantumHardwareBackend`` 子类），
                     None 时使用 Mock 模式返回仿真数据。
        num_qubits : Mock 模式下模拟的量子比特数（默认 20）。
        seed       : 随机数种子（用于 Mock 模式可复现）。

    Returns:
        噪声画像字典，注入失败时返回 None。
    """
    from src.real_machine.noise_extractor import NoiseModelExtractor

    try:
        extractor = NoiseModelExtractor(backend=backend, num_qubits=num_qubits, seed=seed)
        noise_profile = extractor.extract_noise_profile()
        env.inject_noise_profile(noise_profile)
        return noise_profile
    except (
        AttributeError,
        ConnectionError,
        TimeoutError,
        ValueError,
        TypeError,
        RuntimeError,
    ) as e:
        logger.warning(f"[NoiseExtractor] 噪声画像注入失败，跳过: {e}")
        return None
