"""统一的量子任务结果类型。"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class TaskResult(Mapping[str, Any]):
    """Mock 与真机客户端共享的任务状态/结果结构。

    实现 ``Mapping`` 是为了让现有的 ``result["status"]`` 和
    ``result.get(...)`` 调用在迁移期间继续工作；新代码应优先使用属性。
    """

    task_id: str
    status: str
    probability: dict[str, float]
    counts: dict[str, int] | None
    shots: int
    backend: str
    error: str | None = None
    raw: Any = None

    def to_dict(self) -> dict[str, Any]:
        """返回兼容旧客户端返回值的普通字典。"""
        result: dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status,
            "probability": dict(self.probability),
            "counts": None if self.counts is None else dict(self.counts),
            "shots": self.shots,
            "backend": self.backend,
        }
        if self.error is not None:
            result["error"] = self.error
        if self.raw is not None:
            result["raw"] = self.raw

        # cqlib 的旧返回结构使用 result 保存概率。
        result["result"] = dict(self.probability)
        return result

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())
