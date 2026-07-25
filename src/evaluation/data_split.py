"""数据分割模块，防止训练/测试数据泄漏。

本模块提供 ``DataSplitter``，将实验种子（seeds）列表划分为互斥的训练集与
测试集（或 k 折交叉验证分组），确保模型评估不接触训练阶段使用过的种子，
从数据划分层面杜绝训练/测试数据泄漏。

核心保证：
    - 训练集 ∩ 测试集 = ∅（无重叠）
    - 训练集 ∪ 测试集 = 全集（无遗漏）
    - 固定 ``random_state`` 时分割结果可复现
"""

from __future__ import annotations

import numpy as np
from loguru import logger


class DataSplitter:
    """将随机种子列表分割为训练集和测试集，防止数据泄漏。

    本类用于评估流程中将实验种子（seeds）划分为互斥的训练集与测试集，
    确保模型评估不接触训练阶段使用过的种子，从而防止数据泄漏。
    支持固定 ``random_state`` 以保证分割结果可复现。

    Args:
        random_state: 随机种子，固定后分割结果可复现；None 表示不固定。
    """

    def __init__(self, random_state: int | None = None) -> None:
        """初始化数据分割器。

        Args:
            random_state: 随机种子，固定后分割结果可复现；None 表示不固定。
        """
        self.random_state = random_state

    def split(
        self,
        seeds: list[int],
        train_ratio: float = 0.7,
    ) -> tuple[list[int], list[int]]:
        """将种子列表按比例分割为训练集和测试集。

        分割结果保证训练集与测试集互斥（无重叠），且并集等于输入全集，
        从而防止训练/测试数据泄漏。

        Args:
            seeds: 待分割的种子列表。
            train_ratio: 训练集占比，取值范围 (0, 1)，默认 0.7。

        Returns:
            元组 ``(train_seeds, test_seeds)``，两者互斥且并集为全集。

        Raises:
            ValueError: ``seeds`` 为空或 ``train_ratio`` 不在 (0, 1) 区间。
        """
        if not seeds:
            raise ValueError("seeds 列表不能为空")
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"train_ratio 必须在 (0, 1) 区间内，当前为 {train_ratio}")

        shuffled = self._shuffle_seeds(seeds)
        total = len(shuffled)
        split_point = round(total * train_ratio)
        # 保证训练集和测试集都非空
        split_point = max(1, min(total - 1, split_point))

        train_seeds = shuffled[:split_point]
        test_seeds = shuffled[split_point:]

        logger.debug(
            "数据分割完成：总数={}，训练集={}，测试集={}",
            total,
            len(train_seeds),
            len(test_seeds),
        )
        return train_seeds, test_seeds

    def kfold_split(
        self,
        seeds: list[int],
        k: int = 5,
    ) -> list[tuple[list[int], list[int]]]:
        """将种子列表进行 k 折交叉验证分割。

        将种子随机打乱后等分为 k 份（不均分时前若干份多一个元素），
        依次以每份作为测试集、其余作为训练集，共产生 k 组分割。
        每组分割的训练集与测试集互斥，所有测试集的并集为全集。

        Args:
            seeds: 待分割的种子列表。
            k: 折数，须满足 ``2 <= k <= len(seeds)``，默认 5。

        Returns:
            长度为 k 的列表，每个元素为 ``(train_seeds, test_seeds)`` 元组。

        Raises:
            ValueError: ``seeds`` 为空，或 ``k`` 不在 ``[2, len(seeds)]`` 区间。
        """
        if not seeds:
            raise ValueError("seeds 列表不能为空")
        if k < 2:
            raise ValueError(f"k 必须不小于 2，当前为 {k}")
        if k > len(seeds):
            raise ValueError(f"k={k} 不能大于 seeds 数量 {len(seeds)}")

        shuffled = self._shuffle_seeds(seeds)
        folds = self._split_into_folds(shuffled, k)

        splits: list[tuple[list[int], list[int]]] = []
        for i in range(k):
            test_seeds = list(folds[i])
            train_seeds: list[int] = []
            for j in range(k):
                if j != i:
                    train_seeds.extend(folds[j])
            splits.append((train_seeds, test_seeds))

        logger.debug("k 折分割完成：k={}，总数={}", k, len(seeds))
        return splits

    def _shuffle_seeds(self, seeds: list[int]) -> list[int]:
        """使用固定随机状态打乱种子列表。

        Args:
            seeds: 待打乱的种子列表。

        Returns:
            打乱后的新列表（不修改原列表）。
        """
        rng = np.random.default_rng(self.random_state)
        permutation = rng.permutation(len(seeds))
        return [seeds[int(idx)] for idx in permutation]

    @staticmethod
    def _split_into_folds(shuffled: list[int], k: int) -> list[list[int]]:
        """将打乱后的列表等分为 k 份。

        不均分时，前 ``len % k`` 份各多一个元素，保证全覆盖且无重叠。

        Args:
            shuffled: 已打乱的列表。
            k: 份数。

        Returns:
            长度为 k 的列表，每个元素为一个子列表。
        """
        total = len(shuffled)
        base_size = total // k
        remainder = total % k

        folds: list[list[int]] = []
        start = 0
        for i in range(k):
            size = base_size + (1 if i < remainder else 0)
            folds.append(shuffled[start : start + size])
            start += size
        return folds
