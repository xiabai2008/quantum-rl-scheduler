"""
后台仿真任务

simulate_scheduler：使用 PPO 模型进行真实的调度推理决策，
定时推送状态更新，并周期性轮询天衍云真机状态与提交记录。

v2 改进（Day2-3-9）：
- 维护持久化环境状态，不再每次 tick reset
- 通过 env.step(action) 推进真实调度状态
- 系统指标从环境真实观测值提取，非伪造
- 支持 episode 结束后自动重置

共享状态访问（Issue #179 / #723）：
    共享全局状态（``system_status`` / ``task_queue`` / ``manager`` /
    ``_resource_history`` / ``_decision_log``）定义在 ``state.py`` 中，
    由 ``app.py`` 再导出为模块级属性。本模块通过 ``state.py`` 的线程安全
    访问器（``update_system_status`` / ``append_resource_history`` /
    ``append_decision_log`` / ``get_pending_task_count`` 等）读写全局状态，
    避免多 worker 部署时的竞态风险。``_app`` 仅用于访问辅助函数
    （``_get_ppo_model`` / ``_ppo_env`` / ``manager`` 等）。

    测试通过 ``patch("src.visualization.simulator._app", mock_app)`` 替换
    辅助函数引用，并通过 ``_isolate_global_state`` 将 state 模块变量重定向到
    mock_app 的隔离对象，从而隔离全局状态。
"""

import asyncio
import random
from typing import Any

from loguru import logger
from numpy.typing import NDArray

# _app 仅提供辅助函数访问（_get_ppo_model / _ppo_env / manager 等）。
# 全局状态读写统一走 state.py 访问器（Issue #723）。
# 测试通过 patch("src.visualization.simulator._app", mock_app) 替换辅助函数引用。
import src.visualization.app as _app
from src.scheduler.env_types import MAX_QUEUE_SIZE
from src.scheduler.explainability import DecisionExplainer
from src.utils.metrics import update_runtime_gauges
from src.visualization import state as viz_state

_explainer = DecisionExplainer()

# PPO 推理持久化状态
_ppo_current_obs: NDArray[Any] | None = None  # 当前观测向量（episode 内持续更新）
_ppo_episode_reward = 0.0  # 当前 episode 累积奖励
_ppo_episode_step = 0  # 当前 episode 步数


async def simulate_scheduler() -> None:
    """模拟调度引擎行为 — 使用 PPO 模型进行真实推理决策。

    每 3 秒推送一次状态更新。使用持久化的 Gymnasium 环境实例，
    通过 ``env.step(action)`` 推进真实调度状态，系统指标从环境
    观测值中提取（非伪造）。episode 结束后自动重置。

    其中每 20 个 tick（约 60 秒）轮询一次天衍云真机状态
    （``query_quantum_computer_list``）和真机提交记录
    （``results/real_times.json``），将真实机器名/状态与真实提交
    历史通过 WebSocket 推送到前端监控卡片。

    全局状态读写均通过 ``state.py`` 访问器完成（Issue #723）：
    ``update_system_status`` 批量更新并自动设置 ``last_update``，
    ``append_resource_history`` / ``append_decision_log`` 自动裁剪。
    """
    global _ppo_current_obs, _ppo_episode_reward, _ppo_episode_step

    tick = 0
    while True:
        await asyncio.sleep(3)
        tick += 1

        # Issue #723: 通过 state.py 访问器读写全局状态。
        # status 为 system_status 的活动引用（get_system_status_ref），
        # 读取始终反映最新值；写操作统一走 update_system_status 批量更新。
        status = viz_state.get_system_status_ref()
        viz_state.update_system_status({"current_step": status["current_step"] + 1})

        # 本轮 PPO 推理动作（-1 表示未推理）
        action: int = -1
        obs = None  # 保存状态用于可解释性分析
        step_reward: float = 0.0  # 本步真实奖励

        # 尝试使用 PPO 推理
        model = _app._get_ppo_model()
        if model is not None and _app._ppo_env is not None:
            try:
                # 首次运行或 episode 结束后需要 reset
                if _ppo_current_obs is None:
                    _ppo_current_obs, _ = _app._ppo_env.reset()
                    _ppo_episode_reward = 0.0
                    _ppo_episode_step = 0
                    logger.info("[Web] PPO 环境 reset，新 episode 开始")

                # 使用当前观测进行预测（不 reset！）
                obs = _ppo_current_obs
                # Issue #673: 使用 asyncio.to_thread 避免同步推理阻塞事件循环
                action, _ = await asyncio.to_thread(model.predict, obs, deterministic=True)

                # 调用 env.step() 推进真实调度状态
                new_obs, reward, terminated, truncated, _info = _app._ppo_env.step(int(action))
                step_reward = float(reward)
                _ppo_episode_reward += step_reward
                _ppo_episode_step += 1

                # 从真实环境观测值更新系统状态（OBS_DIM=16）
                # obs[0] = 量子比特可用率, obs[1] = 队列长度(归一化), obs[2] = 平均等待时间(归一化)
                # Issue #673: 经典资源利用率从量子利用率确定性推导，不再添加随机噪声
                # Issue #723: 通过 update_system_status 批量更新
                viz_state.update_system_status(
                    {
                        # 真实量子比特利用率
                        "qubit_utilization": round(float(new_obs[0]), 4),
                        # 经典资源利用率（与量子利用率正相关，确定性推导）
                        "classical_utilization": round(
                            max(0.1, min(1.0, float(new_obs[0]) * 0.85 + 0.1)), 4
                        ),
                        # 真实平均等待时间（反归一化）
                        "average_wait_time": round(float(new_obs[2]) * 100, 1),
                        # 队列长度从真实环境观测读取（obs[1] = queue_length / MAX_QUEUE_SIZE）
                        "queue_length": round(float(new_obs[1]) * MAX_QUEUE_SIZE),
                        # Issue #673: 任务完成基于 episode 步数确定性计数，不使用随机数
                        # 每步完成一个任务（简化模型：每步处理队列头部任务）
                        "completed_tasks": status["completed_tasks"] + 1,
                    }
                )

                # 检查 episode 是否结束
                if terminated or truncated:
                    logger.info(
                        f"[Web] PPO episode 结束: "
                        f"steps={_ppo_episode_step}, "
                        f"total_reward={_ppo_episode_reward:.2f}"
                    )
                    _ppo_current_obs = None  # 下次 tick 会自动 reset
                else:
                    _ppo_current_obs = new_obs  # 更新观测，继续下一步

            except (ValueError, RuntimeError, OSError, KeyError) as e:
                # PPO 推理失败，回退随机
                logger.debug(f"[Web] PPO 推理失败，回退随机: {e}")
                _ppo_current_obs = None  # 重置状态，下次重新开始
                # Issue #723: 通过 update_system_status 批量更新
                fallback_updates = {
                    "qubit_utilization": round(
                        max(
                            0.1,
                            min(1.0, status["qubit_utilization"] + random.uniform(-0.03, 0.03)),
                        ),
                        4,
                    ),
                    "classical_utilization": round(
                        max(
                            0.1,
                            min(
                                1.0,
                                status.get("classical_utilization", 0.0)
                                + random.uniform(-0.05, 0.05),
                            ),
                        ),
                        4,
                    ),
                }
                # 模拟任务完成
                if random.random() < 0.2:
                    fallback_updates["completed_tasks"] = status[
                        "completed_tasks"
                    ] + random.randint(0, 1)
                viz_state.update_system_status(fallback_updates)
        else:
            # 无模型，随机模拟
            # Issue #723: 通过 update_system_status 批量更新
            viz_state.update_system_status(
                {
                    "qubit_utilization": round(
                        max(
                            0.1,
                            min(1.0, status["qubit_utilization"] + random.uniform(-0.03, 0.03)),
                        ),
                        4,
                    ),
                    "classical_utilization": round(
                        max(
                            0.1,
                            min(
                                1.0,
                                status.get("classical_utilization", 0.0)
                                + random.uniform(-0.05, 0.05),
                            ),
                        ),
                        4,
                    ),
                }
            )

        # 无 PPO 模型时，队列长度从 Web 任务队列读取
        if model is None:
            # Issue #723: 通过 get_pending_task_count 访问器读取待处理任务数
            viz_state.update_system_status(
                {
                    "queue_length": viz_state.get_pending_task_count(),
                    "average_wait_time": round(
                        max(0.5, status["average_wait_time"] + random.uniform(-0.5, 0.5)), 1
                    ),
                }
            )
        # Issue #723: last_update 由 update_system_status 自动设置，无需手动赋值

        # Issue #679: 同步运行时 Gauge 到 Prometheus 注册表，
        # 使 /metrics 端点输出真实数值而非恒定初始值 0
        update_runtime_gauges(status)

        # 每 20 个 tick（约 60 秒）轮询真机状态 + 真机提交记录
        # 避免高频查询天衍云 API（免费额度有限）
        if tick % 20 == 0:
            try:
                real_machines = _app._get_real_machines_status()
                if real_machines:
                    # Issue #723: 通过 update_system_status 更新
                    viz_state.update_system_status({"real_machines": real_machines})
            except (OSError, RuntimeError, ValueError) as e:
                # 网络/ API 错误 / 运行时错误 / 返回值格式错误
                logger.error(f"[Web] 轮询真机状态异常: {e}")
            try:
                # Issue #723: 通过 update_system_status 更新
                viz_state.update_system_status({"real_submissions": _app._load_real_submissions()})
            except (OSError, ValueError, RuntimeError) as e:
                # 文件 I/O 错误 / 数据格式错误 / 运行时错误
                logger.error(f"[Web] 加载真机提交记录异常: {e}")

        # 注意：PPO激活时，队列长度已在上面从new_obs[1]读取，
        # 此处不再通过随机动画修改queue_length和completed_tasks，避免污染真实指标。
        # 前端任务卡片的动画效果由前端Vue组件独立处理，不影响后端真实数据。

        # 记录资源利用率历史（Issue #22：资源利用率历史趋势图）
        # Issue #723: 通过 append_resource_history 访问器追加并自动裁剪
        viz_state.append_resource_history(
            {
                "step": status["current_step"],
                "qubit_utilization": status["qubit_utilization"],
                "classical_utilization": status["classical_utilization"],
                "queue_length": status["queue_length"],
                "completed_tasks": status["completed_tasks"],
                "average_wait_time": status["average_wait_time"],
                "ppo_episode_reward": round(_ppo_episode_reward, 2),
            }
        )

        # 计算实时指标（Issue #526：实时调度过程可视化增强）
        realtime_metrics = viz_state.calculate_realtime_metrics(
            current_step=status["current_step"],
            completed_tasks=status["completed_tasks"],
            qubit_util=status["qubit_utilization"],
            classical_util=status["classical_utilization"],
            avg_wait_time=status["average_wait_time"],
            ppo_step_reward=step_reward,
        )
        # Issue #723: 通过 update_system_status 更新 throughput
        viz_state.update_system_status({"throughput": realtime_metrics["throughput"]})

        # 记录决策日志（Issue #22：决策过程回放）
        if action >= 0:
            action_label_map = {0: "经典", 1: "量子", 2: "混合", 3: "QEM"}
            log_entry = {
                "step": status["current_step"],
                "task_id": f"task_{status['current_step']}",
                "action": int(action),
                "action_label": action_label_map.get(int(action), "未知"),
                "reward": round(step_reward, 4),
                "source": "PPO",
                "episode_reward": round(_ppo_episode_reward, 2),
                "episode_step": _ppo_episode_step,
            }
            # 计算特征贡献度（Issue #73）
            if obs is not None:
                record = _explainer.explain(
                    state=obs, action=int(action), step=status["current_step"]
                )
                log_entry["feature_contributions"] = record.feature_contributions
                log_entry["explanation_text"] = _explainer.format_explanation(record, top_k=5)
            # Issue #723: 通过 append_decision_log 访问器追加并自动裁剪
            viz_state.append_decision_log(log_entry)

        # 获取 reward 对比数据
        reward_comparison = viz_state.get_reward_comparison()

        await _app.manager.broadcast(
            {
                "type": "status_update",
                # Issue #723: 通过访问器获取状态引用与任务队列引用
                "status": viz_state.get_system_status_ref(),
                "tasks": viz_state.get_task_queue_ref(),
                "ppo_active": _app._ppo_model is not None,
                "ppo_episode_reward": round(_ppo_episode_reward, 2),
                "ppo_episode_step": _ppo_episode_step,
                "realtime_metrics": realtime_metrics,
                "reward_comparison": reward_comparison,
            }
        )
