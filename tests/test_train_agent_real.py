"""Issue #164 真机闭环训练脚本的单元测试。"""

from pathlib import Path

from scripts.training.train_agent_real import (
    AuditedRealClient,
    convergence_timestep,
    save_plot,
)


class _CompletedDelegate:
    machine_name = "tianyan_s"

    def submit_quantum_task(self, **_kwargs):
        return "real-task-1"

    def wait_for_task(self, task_id, timeout, poll_interval):
        assert timeout == 10
        assert poll_interval == 2
        return {"task_id": task_id, "status": "completed", "raw": {"probability": {}}}

    def get_task_status(self, _task_id):
        raise AssertionError("已完成任务应使用缓存状态")


def test_audited_client_records_real_id_without_credentials():
    client = AuditedRealClient(_CompletedDelegate(), wait_timeout=10, poll_interval=2)

    task_id = client.submit_quantum_task(qcis="H Q0\nM Q0", shots=64, task_name="test")

    assert task_id == "real-task-1"
    assert client.get_task_status(task_id)["status"] == "completed"
    assert client.records[0]["task_id"] == task_id
    assert client.records[0]["shots"] == 64
    assert "key" not in client.records[0]


def test_convergence_timestep_uses_final_five_episode_target():
    curve = [
        {"timestep": (index + 1) * 200, "reward": reward}
        for index, reward in enumerate(range(10))
    ]
    assert convergence_timestep(curve) == 2000
    assert convergence_timestep([]) is None


def test_save_plot_writes_two_condition_figure(tmp_path: Path):
    curve = [{"timestep": (index + 1) * 200, "reward": float(index)} for index in range(6)]
    output = tmp_path / "curve.png"

    save_plot({"curve": curve}, {"curve": curve}, output)

    assert output.exists()
    assert output.stat().st_size > 0
