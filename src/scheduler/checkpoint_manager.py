"""
训练检查点版本管理与性能对比模块
Checkpoint Version Management & Performance Comparison

Issue #83：解决训练检查点散落在 ``models/`` 目录、无版本管理、难以对比不同版本
性能的问题。

本模块提供：
    - :class:`CheckpointMeta`：检查点元数据数据类（版本/路径/算法/步数/奖励/标签等）
    - :class:`CheckpointManager`：检查点版本管理器，支持
        * 注册新检查点（自动生成版本号）
        * 列出检查点（按创建时间/平均奖励/训练步数排序）
        * 获取指定指标最优的检查点
        * 对比两个版本的奖励差与改进百分比
        * 删除检查点（文件 + 元数据）
        * 添加/移除自定义标签
        * 元数据 JSON 持久化
        * 清理孤立条目（元数据引用了但文件已不存在）

元数据以 JSON 文件持久化（默认 ``models/checkpoints.json``），单进程使用，
不做线程安全保证。
"""

import json
import os
import uuid
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from typing import Any, ClassVar

from loguru import logger


@dataclass
class CheckpointMeta:
    """检查点元数据数据类

    封装单个训练检查点的描述信息，用于版本管理、性能对比与检索。

    Attributes:
        version: 版本号，如 ``"v1.0.0"`` 或基于时间戳自动生成的字符串。
        path: 检查点文件路径（通常为 ``.zip`` 文件）。
        algorithm: 算法名称，如 ``"ppo"`` / ``"dqn"``。
        timesteps: 训练步数。
        mean_reward: 评估平均奖励。
        std_reward: 奖励标准差（衡量稳定性，越小越稳定）。
        created_at: 创建时间（ISO 8601 字符串，未提供时自动填充当前时间）。
        tags: 自定义标签列表，便于分组与检索。
        notes: 备注信息。
    """

    version: str
    path: str
    algorithm: str
    timesteps: int
    mean_reward: float
    std_reward: float = 0.0
    created_at: str = ""
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def __post_init__(self) -> None:
        """初始化后处理：当 ``created_at`` 为空时自动填充当前时间。"""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """转换为字典，用于 JSON 序列化。

        Returns:
            包含全部字段的字典。
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CheckpointMeta":
        """从字典构造实例，用于 JSON 反序列化。

        仅读取已知字段，忽略多余键；当 ``tags`` 为 ``None`` 时回退为空列表，
        保证数据健壮性。

        Args:
            data: 包含检查点字段的字典。

        Returns:
            构造得到的 :class:`CheckpointMeta` 实例。
        """
        known = {f.name for f in fields(cls)}
        filtered: dict[str, Any] = {k: v for k, v in data.items() if k in known}
        if filtered.get("tags") is None:
            filtered["tags"] = []
        return cls(**filtered)


class CheckpointManager:
    """训练检查点版本管理器

    维护一份检查点元数据 JSON 文件，提供注册、检索、对比、删除、标签与清理
    能力，解决训练检查点散落、难以对比版本性能的问题。

    Args:
        checkpoint_dir: 检查点文件存放目录，默认 ``"models/"``。
        meta_file: 元数据 JSON 文件路径，默认 ``"models/checkpoints.json"``。

    Note:
        - 单进程使用，未做线程安全保证。
        - 元数据文件不存在或损坏时按空列表处理，不抛出异常。
        - ``register`` 不会校验 ``path`` 指向的文件是否存在，允许先注册后落盘。
    """

    #: ``list_checkpoints`` 支持的排序字段
    _SORT_FIELDS: ClassVar[set[str]] = {"created_at", "mean_reward", "timesteps", "std_reward"}
    #: ``get_best`` 支持的指标字段（取最大值为最优）
    _BEST_METRICS: ClassVar[set[str]] = {"mean_reward", "timesteps"}

    def __init__(
        self,
        checkpoint_dir: str = "models/",
        meta_file: str = "models/checkpoints.json",
    ) -> None:
        """初始化检查点管理器，确保检查点目录存在。

        Args:
            checkpoint_dir: 检查点文件存放目录。
            meta_file: 元数据 JSON 文件路径。
        """
        self.checkpoint_dir = checkpoint_dir
        self.meta_file = meta_file
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger.debug(f"[CheckpointManager] 初始化: dir={checkpoint_dir}, meta={meta_file}")

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    def _generate_version(self) -> str:
        """生成基于时间戳的自动版本号。

        格式为 ``v{YYYYMMDD_HHMMSS}_{6位hex}``，时间戳 + 随机后缀确保唯一性。

        Returns:
            自动生成的版本号字符串。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = uuid.uuid4().hex[:6]
        return f"v{timestamp}_{short_id}"

    def _find(self, checkpoints: list[CheckpointMeta], version: str) -> CheckpointMeta | None:
        """在给定列表中按版本号查找检查点。

        Args:
            checkpoints: 检查点列表。
            version: 目标版本号。

        Returns:
            匹配的检查点；未找到时返回 ``None``。
        """
        for cp in checkpoints:
            if cp.version == version:
                return cp
        return None

    # ------------------------------------------------------------------
    # 元数据持久化
    # ------------------------------------------------------------------
    def load_meta(self) -> list[CheckpointMeta]:
        """从元数据 JSON 文件加载检查点列表。

        文件不存在、JSON 解析失败或顶层结构非列表时，均返回空列表并记录警告，
        不抛出异常。

        Returns:
            检查点元数据列表。
        """
        if not os.path.exists(self.meta_file):
            return []
        try:
            with open(self.meta_file, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"[CheckpointManager] 读取元数据文件失败: {e}")
            return []
        if not isinstance(data, list):
            logger.warning("[CheckpointManager] 元数据文件顶层非列表，返回空列表")
            return []
        return [CheckpointMeta.from_dict(item) for item in data if isinstance(item, dict)]

    def save_meta(self, checkpoints: list[CheckpointMeta]) -> None:
        """将检查点列表持久化到元数据 JSON 文件。

        自动创建元数据文件的父目录，使用 ``ensure_ascii=False`` 保留中文备注。

        Args:
            checkpoints: 待持久化的检查点列表。
        """
        parent = os.path.dirname(self.meta_file)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(self.meta_file, "w", encoding="utf-8") as f:
            json.dump(
                [cp.to_dict() for cp in checkpoints],
                f,
                ensure_ascii=False,
                indent=2,
            )
        logger.debug(f"[CheckpointManager] 保存 {len(checkpoints)} 条元数据到 {self.meta_file}")

    # ------------------------------------------------------------------
    # 注册
    # ------------------------------------------------------------------
    def register(
        self,
        path: str,
        algorithm: str,
        timesteps: int,
        mean_reward: float,
        std_reward: float = 0.0,
        tags: list[str] | None = None,
        notes: str = "",
        version: str | None = None,
    ) -> CheckpointMeta:
        """注册新检查点到元数据文件。

        当 ``version`` 为 ``None`` 时自动生成唯一版本号；若显式提供的版本号已
        存在，则抛出 :class:`ValueError`。

        Args:
            path: 检查点文件路径。
            algorithm: 算法名称（如 ``"ppo"`` / ``"dqn"``）。
            timesteps: 训练步数。
            mean_reward: 平均奖励。
            std_reward: 奖励标准差，默认 ``0.0``。
            tags: 自定义标签列表，``None`` 时为空列表。
            notes: 备注信息，默认空字符串。
            version: 版本号；``None`` 时自动生成。

        Returns:
            新注册的 :class:`CheckpointMeta` 实例。

        Raises:
            ValueError: 显式提供的 ``version`` 已存在。
        """
        checkpoints = self.load_meta()
        existing_versions = {cp.version for cp in checkpoints}

        if version is None:
            version = self._generate_version()
            while version in existing_versions:
                version = self._generate_version()
        elif version in existing_versions:
            raise ValueError(f"版本号 {version} 已存在")

        meta = CheckpointMeta(
            version=version,
            path=path,
            algorithm=algorithm,
            timesteps=timesteps,
            mean_reward=mean_reward,
            std_reward=std_reward,
            tags=list(tags) if tags is not None else [],
            notes=notes,
        )
        checkpoints.append(meta)
        self.save_meta(checkpoints)
        logger.info(
            f"[CheckpointManager] 注册检查点 {version} "
            f"(algorithm={algorithm}, timesteps={timesteps}, "
            f"mean_reward={mean_reward:.4f})"
        )
        return meta

    # ------------------------------------------------------------------
    # 列出与检索
    # ------------------------------------------------------------------
    def list_checkpoints(
        self,
        sort_by: str = "created_at",
        descending: bool = True,
    ) -> list[CheckpointMeta]:
        """列出所有检查点，支持排序。

        Args:
            sort_by: 排序字段，支持 ``"created_at"`` / ``"mean_reward"`` /
                ``"timesteps"`` / ``"std_reward"``，默认 ``"created_at"``。
            descending: 是否降序，默认 ``True``。

        Returns:
            排序后的检查点列表。

        Raises:
            ValueError: ``sort_by`` 不在支持字段中。
        """
        if sort_by not in self._SORT_FIELDS:
            raise ValueError(f"不支持的排序字段: {sort_by}，支持: {sorted(self._SORT_FIELDS)}")
        checkpoints = self.load_meta()
        return sorted(checkpoints, key=lambda cp: getattr(cp, sort_by), reverse=descending)

    def get_best(self, metric: str = "mean_reward") -> CheckpointMeta | None:
        """获取指定指标最优的检查点。

        以指标取值最大者为最优（``mean_reward`` 越大越好，``timesteps`` 越大
        表示训练越充分）。

        Args:
            metric: 指标字段，支持 ``"mean_reward"`` / ``"timesteps"``，
                默认 ``"mean_reward"``。

        Returns:
            最优检查点；管理器为空时返回 ``None``。

        Raises:
            ValueError: ``metric`` 不在支持指标中。
        """
        if metric not in self._BEST_METRICS:
            raise ValueError(f"不支持的指标: {metric}，支持: {sorted(self._BEST_METRICS)}")
        checkpoints = self.load_meta()
        if not checkpoints:
            return None
        return max(checkpoints, key=lambda cp: getattr(cp, metric))

    def compare(self, version_a: str, version_b: str) -> dict[str, Any]:
        """对比两个版本的检查点性能。

        计算版本 A 相对于版本 B 的奖励差、训练步数差与改进百分比。改进百分比
        以版本 B 平均奖励的绝对值为基准，避免负奖励导致符号反转；当基准为 0
        时，返回 ``float('inf')`` / ``float('-inf')`` / ``0.0``。

        Args:
            version_a: 版本 A 的版本号。
            version_b: 版本 B 的版本号（作为对比基线）。

        Returns:
            包含以下键的字典：
                - ``version_a`` / ``version_b``：版本号
                - ``reward_diff``：平均奖励差（A - B）
                - ``timestep_diff``：训练步数差（A - B）
                - ``improvement_pct``：改进百分比

        Raises:
            ValueError: 任一版本不存在。
        """
        checkpoints = self.load_meta()
        cp_a = self._find(checkpoints, version_a)
        cp_b = self._find(checkpoints, version_b)
        if cp_a is None:
            raise ValueError(f"版本 {version_a} 不存在")
        if cp_b is None:
            raise ValueError(f"版本 {version_b} 不存在")

        reward_diff = cp_a.mean_reward - cp_b.mean_reward
        timestep_diff = cp_a.timesteps - cp_b.timesteps
        baseline = abs(cp_b.mean_reward)
        if baseline != 0:
            improvement_pct = (reward_diff / baseline) * 100.0
        elif reward_diff > 0:
            improvement_pct = float("inf")
        elif reward_diff < 0:
            improvement_pct = float("-inf")
        else:
            improvement_pct = 0.0

        return {
            "version_a": version_a,
            "version_b": version_b,
            "reward_diff": reward_diff,
            "timestep_diff": timestep_diff,
            "improvement_pct": improvement_pct,
        }

    # ------------------------------------------------------------------
    # 删除
    # ------------------------------------------------------------------
    def delete(self, version: str) -> bool:
        """删除检查点（文件 + 元数据）。

        若文件删除失败（如权限不足），仅记录警告，仍会从元数据中移除条目。

        Args:
            version: 待删除检查点的版本号。

        Returns:
            删除成功返回 ``True``；版本不存在返回 ``False``。
        """
        checkpoints = self.load_meta()
        target = self._find(checkpoints, version)
        if target is None:
            logger.warning(f"[CheckpointManager] 删除失败：版本 {version} 不存在")
            return False

        if target.path and os.path.exists(target.path):
            try:
                os.remove(target.path)
                logger.info(f"[CheckpointManager] 删除检查点文件: {target.path}")
            except OSError as e:
                logger.warning(f"[CheckpointManager] 删除检查点文件失败 {target.path}: {e}")

        survivors = [cp for cp in checkpoints if cp.version != version]
        self.save_meta(survivors)
        logger.info(f"[CheckpointManager] 删除检查点元数据: {version}")
        return True

    # ------------------------------------------------------------------
    # 标签
    # ------------------------------------------------------------------
    def tag(self, version: str, tag: str) -> None:
        """为指定版本添加标签。

        标签已存在时为幂等操作，不重复添加。

        Args:
            version: 目标版本号。
            tag: 待添加的标签。

        Raises:
            ValueError: 版本不存在。
        """
        checkpoints = self.load_meta()
        target = self._find(checkpoints, version)
        if target is None:
            raise ValueError(f"版本 {version} 不存在")
        if tag not in target.tags:
            target.tags.append(tag)
            self.save_meta(checkpoints)
            logger.info(f"[CheckpointManager] 为版本 {version} 添加标签: {tag}")

    def untag(self, version: str, tag: str) -> None:
        """从指定版本移除标签。

        标签不存在时为幂等操作，不报错。

        Args:
            version: 目标版本号。
            tag: 待移除的标签。

        Raises:
            ValueError: 版本不存在。
        """
        checkpoints = self.load_meta()
        target = self._find(checkpoints, version)
        if target is None:
            raise ValueError(f"版本 {version} 不存在")
        if tag in target.tags:
            target.tags.remove(tag)
            self.save_meta(checkpoints)
            logger.info(f"[CheckpointManager] 从版本 {version} 移除标签: {tag}")

    # ------------------------------------------------------------------
    # 清理
    # ------------------------------------------------------------------
    def cleanup_orphans(self) -> list[str]:
        """清理元数据中引用了但文件不存在的孤立条目。

        Returns:
            被清理的版本号列表。
        """
        checkpoints = self.load_meta()
        orphan_versions = {cp.version for cp in checkpoints if not os.path.exists(cp.path)}
        if not orphan_versions:
            logger.debug("[CheckpointManager] 无孤立条目")
            return []
        survivors = [cp for cp in checkpoints if cp.version not in orphan_versions]
        self.save_meta(survivors)
        result = sorted(orphan_versions)
        logger.info(f"[CheckpointManager] 清理 {len(result)} 个孤立检查点: {result}")
        return result


__all__ = ["CheckpointManager", "CheckpointMeta"]
