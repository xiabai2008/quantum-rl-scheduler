"""可视化 state.py 线程安全访问器测试（拆分自 test_visualization.py，Issue #730）。"""

import asyncio
import copy
import json
import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi import HTTPException
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient
from pydantic import ValidationError

from src.visualization import state as vis_state
from src.visualization.app import (
    ConnectionManager,
    SystemStatusUpdate,
    TaskSubmit,
    app,
    simulate_scheduler,
    start_web_server,
    verify_api_key,
)
from src.visualization.security import MAX_CIRCUIT_QUBITS, rate_limiter

app_module = sys.modules["src.visualization.app"]


# ============================================================
# Issue #207 扩展覆盖：state.py 访问器 / routes.py 端点 / websocket_handler
# ============================================================


class TestStateAccessors:
    """state.py 线程安全访问器函数测试（Issue #207，覆盖 154-318 行）。"""

    @pytest.fixture(autouse=True)
    def restore_extended_state(self):
        """保存并恢复 _resource_history / _decision_log / _battle_state。"""
        saved_history = list(vis_state._resource_history)
        saved_log = list(vis_state._decision_log)
        yield
        vis_state._resource_history.clear()
        vis_state._resource_history.extend(saved_history)
        vis_state._decision_log.clear()
        vis_state._decision_log.extend(saved_log)

    def test_get_system_status_returns_copy(self):
        """get_system_status 应返回浅拷贝，修改拷贝不影响原状态。"""
        original = vis_state.get_system_status()
        original["qubit_utilization"] = 0.99
        assert vis_state.system_status["qubit_utilization"] != 0.99

    def test_get_system_status_ref_returns_reference(self):
        """get_system_status_ref 应返回原字典引用。"""
        assert vis_state.get_system_status_ref() is vis_state.system_status

    def test_update_system_status_merges_and_updates_timestamp(self):
        """update_system_status 应合并更新并刷新 last_update。"""
        old_update = vis_state.system_status["last_update"]
        vis_state.update_system_status({"completed_tasks": 999})
        assert vis_state.system_status["completed_tasks"] == 999
        assert vis_state.system_status["last_update"] != old_update

    def test_get_task_queue_returns_copy(self):
        """get_task_queue 应返回浅拷贝列表。"""
        q = vis_state.get_task_queue()
        original_len = len(vis_state.task_queue)
        q.append({"fake": "task"})
        assert len(vis_state.task_queue) == original_len

    def test_get_task_queue_ref_returns_reference(self):
        """get_task_queue_ref 应返回原列表引用。"""
        assert vis_state.get_task_queue_ref() is vis_state.task_queue

    def test_append_task_updates_queue_length(self):
        """append_task 应追加任务并更新 queue_length 为 pending 任务数。"""
        pending_before = len([t for t in vis_state.task_queue if t.get("status") == "pending"])
        vis_state.append_task({"task_id": "TEST-001", "status": "pending"})
        pending_after = len([t for t in vis_state.task_queue if t.get("status") == "pending"])
        assert pending_after == pending_before + 1
        assert vis_state.system_status["queue_length"] == pending_after

    def test_append_task_non_pending_does_not_increase_queue_length(self):
        """append_task 追加非 pending 任务不应增加 queue_length。"""
        pending_before = len([t for t in vis_state.task_queue if t.get("status") == "pending"])
        vis_state.append_task({"task_id": "TEST-002", "status": "running"})
        pending_after = len([t for t in vis_state.task_queue if t.get("status") == "pending"])
        assert pending_after == pending_before
        assert vis_state.system_status["queue_length"] == pending_after

    def test_get_pending_task_count(self):
        """get_pending_task_count 应返回 pending 状态的任务数。"""
        count = vis_state.get_pending_task_count()
        expected = len([t for t in vis_state.task_queue if t.get("status") == "pending"])
        assert count == expected

    def test_append_resource_history_trims_to_max(self):
        """append_resource_history 应自动裁剪到 MAX_RESOURCE_HISTORY。"""
        vis_state._resource_history.clear()
        for i in range(vis_state.MAX_RESOURCE_HISTORY + 10):
            vis_state.append_resource_history({"step": i})
        assert len(vis_state._resource_history) == vis_state.MAX_RESOURCE_HISTORY

    def test_get_resource_history_with_limit(self):
        """get_resource_history(limit) 应返回最近 limit 条数据。"""
        vis_state._resource_history.clear()
        for i in range(10):
            vis_state.append_resource_history({"step": i})
        recent = vis_state.get_resource_history(limit=3)
        assert len(recent) == 3
        assert recent[-1]["step"] == 9

    def test_get_resource_history_default_all(self):
        """get_resource_history() 默认返回全部数据。"""
        vis_state._resource_history.clear()
        for i in range(5):
            vis_state.append_resource_history({"step": i})
        all_data = vis_state.get_resource_history()
        assert len(all_data) == 5

    def test_append_decision_log_trims_to_max(self):
        """append_decision_log 应自动裁剪到 MAX_DECISION_LOG。"""
        vis_state._decision_log.clear()
        for i in range(vis_state.MAX_DECISION_LOG + 10):
            vis_state.append_decision_log({"step": i})
        assert len(vis_state._decision_log) == vis_state.MAX_DECISION_LOG

    def test_get_decision_log_with_limit(self):
        """get_decision_log(limit) 应返回最近 limit 条记录。"""
        vis_state._decision_log.clear()
        for i in range(10):
            vis_state.append_decision_log({"step": i})
        recent = vis_state.get_decision_log(limit=3)
        assert len(recent) == 3
        assert recent[-1]["step"] == 9

    def test_get_decision_log_default_all(self):
        """get_decision_log() 默认返回全部记录。"""
        vis_state._decision_log.clear()
        for i in range(5):
            vis_state.append_decision_log({"step": i})
        all_data = vis_state.get_decision_log()
        assert len(all_data) == 5

    def test_get_battle_state_returns_copy(self):
        """get_battle_state 应返回浅拷贝。"""
        battle = vis_state.get_battle_state()
        battle["running"] = True
        assert vis_state._battle_state["running"] is False

    def test_get_battle_state_ref_returns_reference(self):
        """get_battle_state_ref 应返回原字典引用。"""
        assert vis_state.get_battle_state_ref() is vis_state._battle_state

    def test_reset_battle_state_clears_all_fields(self):
        """reset_battle_state 应重置所有字段。"""
        vis_state._battle_state["running"] = True
        vis_state._battle_state["step"] = 10
        vis_state._battle_state["ppo_reward"] = 100.0
        vis_state._battle_state["fcfs_reward"] = 50.0
        vis_state._battle_state["ppo_history"] = [{"step": 1}]
        vis_state._battle_state["fcfs_history"] = [{"step": 1}]
        vis_state.reset_battle_state()
        assert vis_state._battle_state["running"] is False
        assert vis_state._battle_state["step"] == 0
        assert vis_state._battle_state["ppo_reward"] == 0.0
        assert vis_state._battle_state["fcfs_reward"] == 0.0
        assert vis_state._battle_state["ppo_history"] == []
        assert vis_state._battle_state["fcfs_history"] == []
        assert vis_state._battle_state["ppo_env"] is None
        assert vis_state._battle_state["fcfs_env"] is None
        assert vis_state._battle_state["ppo_obs"] is None
        assert vis_state._battle_state["fcfs_obs"] is None

    def test_get_connection_manager_returns_singleton(self):
        """get_connection_manager 应返回全局 manager 单例。"""
        assert vis_state.get_connection_manager() is vis_state.manager
