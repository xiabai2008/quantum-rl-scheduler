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

import random
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
    Task,
)

if TYPE_CHECKING:
    # 仅用于类型标注，避免运行时循环导入
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
FREE_TIER_MAX_QUBITS = 1  # 天衍-176 真机仅 1-qubit 电路稳定，多量子比特频繁"运行失败"


def generate_qcis_circuit(
    task: Task,
    max_qubits: int = _MAX_REAL_QUBITS,
    seed: int | None = None,
    two_qubit_gates: bool = False,
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

    Returns:
        QCIS 格式的电路字符串，每行一条指令

    Examples:
        >>> t = Task(task_id="0", task_type="quantum", qubit_count=3, priority=3)
        >>> qcis = generate_qcis_circuit(t)
        >>> assert "H" in qcis or "X" in qcis
        >>> assert "M" in qcis
    """
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
            angle = round(rng.uniform(0, 2 * 3.14159), 4)
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
# =============================================================================


def parse_measurement_result(status: dict[str, Any]) -> dict[str, float]:
    """从真机任务状态中解析测量概率分布。

    天衍云 cqlib 返回的 status 字典可能包含：
    - ``probability``: 直接的概率分布字典 {"0": 0.5, "1": 0.5}
    - ``resultStatus``: 原始 shots 计数，需转换为概率
    - ``result``: 某些版本返回的嵌套结果

    Args:
        status: get_task_status() 返回的状态字典

    Returns:
        归一化的概率分布字典 {"bitstring": probability}，空字典表示解析失败
    """
    probability: dict[str, float] = {}

    # 路径 1: 直接的 probability 字段
    raw_prob = status.get("probability")
    if raw_prob and isinstance(raw_prob, dict):
        for key, val in raw_prob.items():
            try:
                probability[str(key)] = float(val)
            except (ValueError, TypeError):
                continue
        if probability:
            total = sum(probability.values())
            if total > 0:
                probability = {k: v / total for k, v in probability.items()}
            return probability

    # 路径 2: resultStatus 原始 shots 计数
    result_status = status.get("resultStatus")
    if result_status and isinstance(result_status, str):
        try:
            import json

            counts = json.loads(result_status)
            if isinstance(counts, dict):
                total_shots = sum(counts.values())
                if total_shots > 0:
                    probability = {str(k): float(v) / total_shots for k, v in counts.items()}
                    return probability
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    # 路径 3: result 字段（嵌套 probability）
    result = status.get("result")
    if result and isinstance(result, dict):
        inner_prob = result.get("probability")
        if inner_prob and isinstance(inner_prob, dict):
            for key, val in inner_prob.items():
                try:
                    probability[str(key)] = float(val)
                except (ValueError, TypeError):
                    continue
            if probability:
                total = sum(probability.values())
                if total > 0:
                    probability = {k: v / total for k, v in probability.items()}
                return probability

    return {}


def compute_theoretical_distribution(qcis: str) -> dict[str, float]:
    """根据 QCIS 电路计算理论概率分布（用于保真度对比）。

    对于简单电路（仅 H 门 + 测量），理论分布为均匀分布。
    对于无 H 门的电路（如 X 门），理论分布为确定态。

    当前支持的电路模式：
    - 仅 H 门：均匀分布 {"0": 0.5, "1": 0.5}
    - 含 X 门：确定态 {"1": 1.0}
    - 其他/复杂电路：均匀分布（保守估计）

    Args:
        qcis: QCIS 格式电路字符串

    Returns:
        理论概率分布字典
    """
    lines = [line.strip() for line in qcis.strip().split("\n") if line.strip()]
    gates = [line for line in lines if not line.startswith("M")]

    has_h = any(g.startswith("H ") for g in gates)
    has_x = any(g.startswith("X ") for g in gates)

    # 统计测量的量子比特数
    measure_lines = [line for line in lines if line.startswith("M")]
    if not measure_lines:
        return {"0": 1.0}

    # 提取测量的比特数
    measure_qubits: list[str] = []
    for line in measure_lines:
        parts = line.replace("M", "").strip().split()
        measure_qubits.extend(parts)
    n_qubits = max(1, len(measure_qubits))

    if has_h and not has_x:
        # H 门产生均匀分布
        n_outcomes = 2**n_qubits
        prob = 1.0 / n_outcomes
        return {format(k, f"0{n_qubits}b"): prob for k in range(n_outcomes)}
    elif has_x and not has_h:
        # X 门翻转，全 1 态
        all_ones = "1" * n_qubits
        return {all_ones: 1.0}
    else:
        # 混合或复杂电路，使用均匀分布作为保守估计
        n_outcomes = 2**n_qubits
        prob = 1.0 / n_outcomes
        return {format(k, f"0{n_qubits}b"): prob for k in range(n_outcomes)}


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

    线性映射：quality=0 → reward=0.5，quality=1 → reward=5.0。
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
) -> None:
    """
    向真机提交一个量子任务（非阻塞，异常安全）。

    真机提交在仿真循环中是非阻塞的：提交后立即返回 task_id 并登记到
    ``env._pending_real_tasks``，后续 step() 通过 ``poll_pending_real_tasks``
    轮询结果，避免阻塞 RL 训练。

    降级机制（Issue #64）：当 ``env._real_machine_degraded=True`` 时跳过提交，
    真机不可用时自动 fallback 到 Mock（仅计入仿真统计）。

    Args:
        env     : 调度环境实例（提供真机客户端、pending 列表等内部状态）
        machine : 目标真机
        task    : 待提交任务
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
            max_qubits=min(machine.total_qubits, FREE_TIER_MAX_QUBITS),
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
            env._pending_real_tasks.append(
                {
                    "task_id": str(real_task_id),
                    "machine_name": machine.name,
                    "submit_step": env._current_step,
                    "poll_count": 0,
                    "task_id_str": str(task.task_id),
                    "qcis_circuit": qcis,
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
        # 真机 API 提交可能因网络/认证/服务端等多种原因失败，无法精确收窄
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
    非阻塞轮询已提交真机任务的结果，返回本步反馈 reward（Issue #64）。

    遍历 ``env._pending_real_tasks``，对每个任务调用 ``get_task_status`` 查询状态：
        - completed : 计入成功，返回 REAL_MACHINE_SUCCESS_BONUS
        - error     : 计入失败，返回 REAL_MACHINE_FAIL_PENALTY，触发降级判断
        - timeout   : 轮询次数超过 REAL_MACHINE_MAX_POLL_STEPS，视为超时失败
        - running/unknown : poll_count +1，保留在 pending 列表

    所有反馈乘以 ``env.real_machine_feedback_weight`` 后累加返回。

    Args:
        env: 调度环境实例

    Returns:
        本步真机反馈 reward（正为成功加成，负为失败惩罚，0 表示无新结果）
    """
    if not env._pending_real_tasks:
        return 0.0

    total_feedback = 0.0
    still_pending: list[dict[str, Any]] = []

    for pending in env._pending_real_tasks:
        pending["poll_count"] += 1
        machine_name = pending["machine_name"]
        real_task_id = pending["task_id"]
        task_id_str = pending["task_id_str"]
        client = env._real_clients.get(machine_name)

        # 客户端丢失（理论上不应发生），视为失败
        if client is None:
            total_feedback += REAL_MACHINE_FAIL_PENALTY * env.real_machine_feedback_weight
            record_real_failure(env, machine_name, "客户端丢失")
            continue

        try:
            status = client.get_task_status(real_task_id)
        except Exception as e:
            # 查询异常视为本步未拿到结果，保留在 pending 列表
            logger.debug(f"[真机闭环] 查询 {real_task_id} 异常: {e}")
            still_pending.append(pending)
            continue

        status_str = str(status.get("status", "unknown"))

        if status_str == "completed":
            # 真机成功：根据反馈模式计算 reward（Issue #235）
            reward_delta, fidelity, formula = _compute_real_feedback(env, pending, status)
            total_feedback += reward_delta * env.real_machine_feedback_weight
            env._real_success_count += 1
            env._real_consecutive_failures = 0  # 成功重置连续失败计数

            # 记录详细结果元数据（Issue #235 可追溯性）
            _record_real_result(env, pending, status, reward_delta, fidelity, formula)

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
            record_real_failure(env, machine_name, "任务状态=error")
        elif pending["poll_count"] >= REAL_MACHINE_MAX_POLL_STEPS:
            # 超时：视为失败
            total_feedback += REAL_MACHINE_FAIL_PENALTY * env.real_machine_feedback_weight
            record_real_failure(env, machine_name, "轮询超时")
            logger.debug(
                f"[真机闭环] 任务 {task_id_str} 轮询超时 (poll_count={pending['poll_count']})"
            )
        else:
            # 仍在运行，保留到下一步轮询
            still_pending.append(pending)

    env._pending_real_tasks = still_pending
    return total_feedback


# =============================================================================
# 真机反馈计算与结果记录（Issue #235）
# =============================================================================


def _compute_real_feedback(
    env: "QuantumSchedulingEnv",
    pending: dict[str, Any],
    status: dict[str, Any],
) -> tuple[float, float, str]:
    """根据真机反馈模式计算 reward（Issue #235）。

    三种模式：
    - status_only  : 固定 bonus（旧行为）
    - result_aware : 解析测量分布，按保真度计算 reward
    - shuffled     : 打乱测量结果后按保真度计算（消融对照）

    Args:
        env     : 调度环境实例
        pending : pending 任务记录（含 qcis_circuit）
        status  : get_task_status() 返回的状态

    Returns:
        (reward, fidelity, formula_str) 三元组
    """
    mode = getattr(env, "real_feedback_mode", REAL_FEEDBACK_STATUS_ONLY)

    if mode == REAL_FEEDBACK_STATUS_ONLY:
        # 旧行为：固定 bonus，不解析测量结果
        return (
            REAL_MACHINE_SUCCESS_BONUS,
            -1.0,  # -1 表示未计算保真度
            f"reward={REAL_MACHINE_SUCCESS_BONUS:.1f} (status_only, fixed bonus)",
        )

    # result_aware 或 shuffled 模式：解析测量结果
    measured = parse_measurement_result(status)
    qcis = pending.get("qcis_circuit", "H Q0\nM Q0")
    theoretical = compute_theoretical_distribution(qcis)

    if mode == REAL_FEEDBACK_SHUFFLED:
        # 打乱测量结果（消融对照）
        measured = shuffle_measurement(measured)

    reward, fidelity, formula = compute_real_result_reward(measured, theoretical)

    if mode == REAL_FEEDBACK_SHUFFLED:
        formula += " [SHUFFLED]"

    return reward, fidelity, formula


def _record_real_result(
    env: "QuantumSchedulingEnv",
    pending: dict[str, Any],
    status: dict[str, Any],
    reward_delta: float,
    fidelity: float,
    formula: str,
) -> None:
    """记录真机结果的详细元数据（Issue #235 可追溯性）。

    每条记录包含 task_id、circuit_hash、backend、shots、counts/probability、
    objective_value、result_valid、fallback_mode、reward_delta 及计算公式。

    Args:
        env          : 调度环境实例
        pending      : pending 任务记录
        status       : 真机返回的状态字典
        reward_delta : 实际 reward 增量
        fidelity     : 保真度（-1 表示未计算）
        formula      : 计算公式描述
    """
    if not hasattr(env, "_real_result_records"):
        env._real_result_records = []

    measured = parse_measurement_result(status) if fidelity >= 0 else {}
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
        "probability": measured,
        "fidelity": round(fidelity, 6) if fidelity >= 0 else None,
        "reward_delta": round(reward_delta, 6),
        "formula": formula,
        "result_valid": len(measured) > 0,
        "fallback_mode": mode == REAL_FEEDBACK_STATUS_ONLY and len(measured) == 0,
    }

    env._real_result_records.append(record)


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
