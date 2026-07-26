"""TaskResult 统一返回类型测试。"""

from src.api.types import TaskResult


def test_task_result_attributes_and_legacy_mapping() -> None:
    result = TaskResult(
        task_id="task-1",
        status="completed",
        probability={"0": 0.75, "1": 0.25},
        counts={"0": 3, "1": 1},
        shots=4,
        backend="tianyan-287",
    )

    assert result.status == "completed"
    assert result["task_id"] == "task-1"
    assert result.get("shots") == 4
    assert result.to_dict()["result"] == {"0": 0.75, "1": 0.25}


def test_task_result_to_dict_does_not_share_nested_mappings() -> None:
    result = TaskResult(
        task_id="task-2",
        status="running",
        probability={},
        counts=None,
        shots=0,
        backend="tianyan-287",
    )

    serialized = result.to_dict()
    serialized["probability"]["0"] = 1.0
    assert result.probability == {}
