"""防泄漏 / OOD 泛化评估模块。

提供数据分割（防泄漏）、留出盲测评估与分布外泛化验证能力，
确保模型评估的严谨性与跨分布鲁棒性可量化。

子模块：
    - ``data_split``: 训练/测试种子分割（防数据泄漏）
    - ``blind_test``: 留出测试集盲测评估
    - ``ood_generalization``: 分布外泛化验证
"""

from __future__ import annotations

from src.evaluation.blind_test import BlindTestEvaluator, PredictableModel, extract_action
from src.evaluation.data_split import DataSplitter
from src.evaluation.ood_generalization import OODGeneralizationTester

__all__ = [
    "BlindTestEvaluator",
    "DataSplitter",
    "OODGeneralizationTester",
    "PredictableModel",
    "extract_action",
]
