"""
量子任务调度环境的渲染模块
Rendering Module for Quantum-Classical Hybrid Task Scheduling Environment

本模块将调度环境的渲染与关闭逻辑抽离为独立函数，便于单元测试与复用：
    - render_env  : 渲染当前环境状态（"human" 打印日志，"ansi" 返回字符串）
    - close_env   : 关闭环境，清空内部状态

依赖关系：不依赖 env.py，通过 ``env`` 参数访问环境内部状态，避免循环导入。
"""

from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:
    # 仅用于类型标注，避免运行时循环导入
    from src.scheduler.env import QuantumSchedulingEnv


def render_env(env: "QuantumSchedulingEnv") -> Any | None:
    """
    渲染当前环境状态。

    - "human" 模式：通过 logger 打印格式化的状态面板
    - "ansi" 模式：返回 ANSI 字符串（适合日志或测试）
    - None（未设置渲染模式）：不执行任何操作

    Args:
        env: 调度环境实例

    Returns:
        render_mode == "ansi" 时返回格式化字符串，否则返回 None
    """
    if env.render_mode is None:
        return None

    # 构建状态面板
    sep = "=" * 64
    lines = [
        sep,
        f"  量子任务调度环境  |  步骤: {env._current_step}/{env.max_steps}"
        f"  |  累计奖励: {env._episode_reward:.2f}"
        f"  |  机器数: {len(env._machines)}",
        "-" * 64,
        f"  [量子资源(聚合)] 可用比率: {env._quantum.available_ratio:.2%}"
        f"  |  保真度: {env._quantum.fidelity:.4f}"
        f"  |  量子队列: {env._quantum.quantum_queue}",
        f"  [经典资源] 负载: {env._classical.load:.2%}"
        f"  |  经典队列: {env._classical.queue}",
        f"  [任务队列]  长度: {len(env._task_queue)}" f"  |  已调度: {env._total_scheduled}",
        f"  [统计] 量子成功: {env._quantum_success}"
        f"  |  经典成功: {env._classical_success}"
        f"  |  混合成功: {env._hybrid_success}"
        f"  |  不兼容: {env._mismatch_count}",
    ]

    # 多机器明细表
    if len(env._machines) > 1:
        lines.append("-" * 64)
        lines.append("  [量子机器明细]")
        for m in env._machines:
            status = "在线" if m.available else "维护"
            real_tag = "真机" if m.is_real else "仿真"
            sched_cnt = env._machine_schedule_count.get(m.name, 0)
            lines.append(
                f"    {m.name:14s} | {m.total_qubits:3d}q | "
                f"可用{m.available_ratio:5.1%} | 保真{m.fidelity:.3f} | "
                f"队列{m.quantum_queue:2d} | {status} | {real_tag} | "
                f"调度{sched_cnt}"
            )
        if env._last_selected_machine:
            lines.append(f"  [本步路由] → {env._last_selected_machine}")

    lines.append("-" * 64)

    # 附加当前任务信息
    if env._current_task is not None:
        t = env._current_task
        lines.append(
            f"  [当前任务] ID={t.task_id}  类型={t.task_type}"
            f"  紧急={t.urgency:.2f}  等待={t.wait_steps}步"
        )
    else:
        lines.append("  [当前任务] 无")

    # 附加最近日志
    if env._render_log:
        recent = env._render_log[-5:]  # 最近5条
        lines.append("-" * 64)
        lines.append("  最近日志:")
        for log in recent:
            lines.append(f"    {log}")

    lines.append(sep)
    output = "\n".join(lines) + "\n"

    if env.render_mode == "human":
        logger.info(output)
    elif env.render_mode == "ansi":
        return output

    return None


def close_env(env: "QuantumSchedulingEnv") -> None:
    """
    关闭环境，释放资源。

    当前实现中无额外资源需要释放，仅清空内部状态。

    Args:
        env: 调度环境实例
    """
    env._task_queue.clear()
    env._current_task = None
    env._render_log.clear()
