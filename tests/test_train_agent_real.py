"""Issue #164 真机闭环训练脚本的单元测试。"""

from pathlib import Path

from scripts.training.train_agent_real import (
    AuditedRealClient,
    completion_rate_from_info,
    convergence_timestep,
    run_preflight,
    save_plot,
)


class _CompletedDelegate:
    machine_name = "tianyan176"

    def __init__(self):
        self.kwargs = None

    def submit_quantum_task(self, **kwargs):
        self.kwargs = kwargs
        return "real-task-1"

    def wait_for_task(self, task_id, timeout, poll_interval):
        assert timeout == 10
        assert poll_interval == 2
        return {"task_id": task_id, "status": "completed", "result": {}}

    def get_task_status(self, _task_id):
        raise AssertionError("已完成任务应使用缓存状态")


def test_audited_client_records_real_id_without_credentials():
    delegate = _CompletedDelegate()
    client = AuditedRealClient(delegate, wait_timeout=10, poll_interval=2)

    task_id = client.submit_quantum_task(qcis="RY Q0,3.14\nM Q0", shots=64, task_name="test")

    assert task_id == "real-task-1"
    assert delegate.kwargs["qcis"] == "H Q0\nM Q0"
    assert client.get_task_status(task_id)["status"] == "completed"
    assert client.records[0]["task_id"] == task_id
    assert client.records[0]["shots"] == 64
    assert client.records[0]["probability"] == {}
    assert "key" not in client.records[0]


def test_preflight_rejects_simulator_before_loading_credentials(tmp_path: Path):
    try:
        run_preflight("tianyan_s", 32, 8, 10, tmp_path / "quota.json")
    except RuntimeError as exc:
        assert "不是本实验允许的物理真机" in str(exc)
    else:
        raise AssertionError("模拟器不得通过物理真机预检")


def test_completion_rate_uses_environment_success_counters():
    info = {
        "quantum_success": 40,
        "classical_success": 30,
        "hybrid_success": 10,
        "total_scheduled": 100,
    }
    assert completion_rate_from_info(info) == 0.8
    assert completion_rate_from_info({}) == 0.0


def test_convergence_timestep_uses_final_five_episode_target():
    curve = [
        {"timestep": (index + 1) * 200, "reward": reward} for index, reward in enumerate(range(10))
    ]
    assert convergence_timestep(curve) == 2000
    assert convergence_timestep([]) is None


def test_save_plot_writes_two_condition_figure(tmp_path: Path):
    curve = [{"timestep": (index + 1) * 200, "reward": float(index)} for index in range(6)]
    output = tmp_path / "curve.png"

    save_plot({"curve": curve}, {"curve": curve}, output)

    assert output.exists()
    assert output.stat().st_size > 0
