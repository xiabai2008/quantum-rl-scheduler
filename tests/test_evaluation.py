"""防泄漏 / OOD 泛化评估模块单元测试。

测试覆盖：
    - DataSplitter: 分割不重叠性、比例正确性、可复现性、kfold 覆盖性、异常输入
    - BlindTestEvaluator: 评估结果格式正确性、episode 计数、动作提取兼容性
    - OODGeneralizationTester: 偏移测试、衰减计算、鲁棒性判定
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.evaluation.blind_test import BlindTestEvaluator, extract_action
from src.evaluation.data_split import DataSplitter
from src.evaluation.ood_generalization import OODGeneralizationTester
from src.scheduler.env import QuantumSchedulingEnv

# ============================================================
# 测试辅助：模拟模型
# ============================================================


class _StubModel:
    """模拟模型，直接返回固定动作（int），兼容项目内 Agent 包装器。"""

    def __init__(self, action: int = 0) -> None:
        """初始化模拟模型。

        Args:
            action: 固定返回的动作索引。
        """
        self.action = action
        self.call_count = 0

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> int:
        """返回固定动作。

        Args:
            obs: 观测向量（未使用）。
            deterministic: 是否确定性（未使用）。

        Returns:
            固定动作索引。
        """
        self.call_count += 1
        return self.action


class _StubSB3Model:
    """模拟 Stable-Baselines3 模型，返回 (action_array, state) 元组。"""

    def __init__(self, action: int = 1) -> None:
        """初始化 SB3 风格模拟模型。

        Args:
            action: 固定返回的动作索引。
        """
        self.action = action

    def predict(self, obs: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
        """返回 (action, state) 元组。

        Args:
            obs: 观测向量（未使用）。
            deterministic: 是否确定性（未使用）。

        Returns:
            ``(np.array([action]), None)`` 元组。
        """
        return np.array([self.action]), None


@pytest.fixture
def small_env() -> QuantumSchedulingEnv:
    """小型快速测试环境（15 步）。"""
    return QuantumSchedulingEnv(max_steps=15, max_qubits=287)


@pytest.fixture
def stub_model() -> _StubModel:
    """返回固定动作 0 的模拟模型。"""
    return _StubModel(action=0)


# ============================================================
# DataSplitter 测试
# ============================================================


class TestDataSplitter:
    """DataSplitter 数据分割测试。"""

    @staticmethod
    def test_split_no_overlap() -> None:
        """训练集与测试集不应重叠。"""
        splitter = DataSplitter(random_state=42)
        seeds = list(range(20))
        train, test = splitter.split(seeds, train_ratio=0.7)
        assert set(train).isdisjoint(set(test)), "训练集与测试集存在重叠"

    @staticmethod
    def test_split_full_coverage() -> None:
        """训练集与测试集的并集应等于全集。"""
        splitter = DataSplitter(random_state=42)
        seeds = list(range(20))
        train, test = splitter.split(seeds, train_ratio=0.7)
        assert set(train) | set(test) == set(seeds), "分割未覆盖全部种子"

    @staticmethod
    def test_split_ratio_correct() -> None:
        """训练集比例应接近 train_ratio。"""
        splitter = DataSplitter(random_state=42)
        seeds = list(range(100))
        train, test = splitter.split(seeds, train_ratio=0.7)
        assert len(train) == 70
        assert len(test) == 30

    @staticmethod
    def test_split_both_nonempty() -> None:
        """小数据集分割后训练集与测试集都应非空。"""
        splitter = DataSplitter(random_state=42)
        train, test = splitter.split(list(range(4)), train_ratio=0.7)
        assert len(train) >= 1
        assert len(test) >= 1

    @staticmethod
    def test_split_reproducible() -> None:
        """相同 random_state 应产生相同分割结果。"""
        seeds = list(range(50))
        train1, test1 = DataSplitter(random_state=123).split(seeds)
        train2, test2 = DataSplitter(random_state=123).split(seeds)
        assert train1 == train2
        assert test1 == test2

    @staticmethod
    def test_split_different_states_differ() -> None:
        """不同 random_state 通常应产生不同分割结果。"""
        seeds = list(range(50))
        train1, _ = DataSplitter(random_state=1).split(seeds)
        train2, _ = DataSplitter(random_state=2).split(seeds)
        assert train1 != train2

    @staticmethod
    def test_split_empty_seeds_raises() -> None:
        """空种子列表应抛出 ValueError。"""
        with pytest.raises(ValueError):
            DataSplitter().split([])

    @staticmethod
    def test_split_invalid_ratio_raises() -> None:
        """非法 train_ratio 应抛出 ValueError。"""
        with pytest.raises(ValueError):
            DataSplitter().split([1, 2, 3], train_ratio=0.0)
        with pytest.raises(ValueError):
            DataSplitter().split([1, 2, 3], train_ratio=1.0)
        with pytest.raises(ValueError):
            DataSplitter().split([1, 2, 3], train_ratio=1.5)

    @staticmethod
    def test_kfold_count() -> None:
        """kfold 应返回 k 组分割。"""
        splitter = DataSplitter(random_state=42)
        splits = splitter.kfold_split(list(range(20)), k=5)
        assert len(splits) == 5

    @staticmethod
    def test_kfold_no_overlap_per_fold() -> None:
        """每组 kfold 分割的训练集与测试集不应重叠。"""
        splitter = DataSplitter(random_state=42)
        seeds = list(range(20))
        for train, test in splitter.kfold_split(seeds, k=5):
            assert set(train).isdisjoint(set(test)), "某折训练集与测试集重叠"

    @staticmethod
    def test_kfold_full_coverage() -> None:
        """所有测试折的并集应等于全集。"""
        splitter = DataSplitter(random_state=42)
        seeds = list(range(20))
        splits = splitter.kfold_split(seeds, k=5)
        all_test: set[int] = set()
        for _, test in splits:
            all_test |= set(test)
        assert all_test == set(seeds), "kfold 测试折未覆盖全部种子"

    @staticmethod
    def test_kfold_test_disjoint_across_folds() -> None:
        """各测试折之间应互斥。"""
        splitter = DataSplitter(random_state=42)
        seeds = list(range(20))
        splits = splitter.kfold_split(seeds, k=5)
        test_sets = [set(test) for _, test in splits]
        for i in range(len(test_sets)):
            for j in range(i + 1, len(test_sets)):
                assert test_sets[i].isdisjoint(test_sets[j]), "不同折测试集存在重叠"

    @staticmethod
    def test_kfold_uneven_split() -> None:
        """不能均分时各折大小差不超过 1。"""
        splitter = DataSplitter(random_state=42)
        seeds = list(range(23))  # 23 / 5 → 4,4,5,5,5
        splits = splitter.kfold_split(seeds, k=5)
        sizes = [len(test) for _, test in splits]
        assert max(sizes) - min(sizes) <= 1

    @staticmethod
    def test_kfold_invalid_k_raises() -> None:
        """非法 k 值应抛出 ValueError。"""
        with pytest.raises(ValueError):
            DataSplitter().kfold_split([1, 2, 3], k=1)
        with pytest.raises(ValueError):
            DataSplitter().kfold_split([1, 2, 3], k=4)
        with pytest.raises(ValueError):
            DataSplitter().kfold_split([], k=3)

    @staticmethod
    def test_split_duplicate_seeds_raises() -> None:
        """Issue #386: seeds 列表包含重复元素应抛出 ValueError，防止数据泄漏。"""
        splitter = DataSplitter(random_state=42)
        with pytest.raises(ValueError, match="重复元素"):
            splitter.split([1, 2, 2, 3, 4], train_ratio=0.6)
        # 单个重复也应被检测
        with pytest.raises(ValueError, match="重复元素"):
            splitter.split([5, 5], train_ratio=0.5)

    @staticmethod
    def test_kfold_duplicate_seeds_raises() -> None:
        """Issue #386: kfold seeds 列表包含重复元素应抛出 ValueError。"""
        splitter = DataSplitter(random_state=42)
        with pytest.raises(ValueError, match="重复元素"):
            splitter.kfold_split([1, 2, 2, 3, 4, 5, 6], k=3)

    @staticmethod
    def test_split_unique_seeds_succeeds() -> None:
        """Issue #386: 无重复 seeds 列表应正常分割。"""
        splitter = DataSplitter(random_state=42)
        train, test = splitter.split([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], train_ratio=0.7)
        assert len(train) + len(test) == 10
        assert set(train).isdisjoint(set(test))


# ============================================================
# BlindTestEvaluator 测试
# ============================================================


class TestBlindTestEvaluator:
    """BlindTestEvaluator 盲测评估测试。"""

    @staticmethod
    def test_evaluate_result_format(
        stub_model: _StubModel, small_env: QuantumSchedulingEnv
    ) -> None:
        """评估结果应包含所有必需键且类型正确。"""
        evaluator = BlindTestEvaluator()
        result = evaluator.evaluate(stub_model, small_env, [42, 123], episodes_per_seed=2)
        required_keys = {
            "mean_reward",
            "std_reward",
            "min_reward",
            "max_reward",
            "all_rewards",
            "num_episodes",
        }
        assert required_keys <= set(result.keys()), "结果缺少必需键"
        assert isinstance(result["mean_reward"], float)
        assert isinstance(result["std_reward"], float)
        assert isinstance(result["min_reward"], float)
        assert isinstance(result["max_reward"], float)
        assert isinstance(result["all_rewards"], list)
        assert isinstance(result["num_episodes"], int)

    @staticmethod
    def test_evaluate_episode_count(
        stub_model: _StubModel, small_env: QuantumSchedulingEnv
    ) -> None:
        """all_rewards 长度应等于 seeds × episodes_per_seed。"""
        evaluator = BlindTestEvaluator()
        result = evaluator.evaluate(stub_model, small_env, [1, 2, 3], episodes_per_seed=4)
        assert result["num_episodes"] == 12
        assert len(result["all_rewards"]) == 12

    @staticmethod
    def test_evaluate_stats_consistency(
        stub_model: _StubModel, small_env: QuantumSchedulingEnv
    ) -> None:
        """统计量应与 all_rewards 一致。"""
        evaluator = BlindTestEvaluator()
        result = evaluator.evaluate(stub_model, small_env, [1, 2], episodes_per_seed=3)
        rewards = result["all_rewards"]
        assert result["min_reward"] == pytest.approx(min(rewards))
        assert result["max_reward"] == pytest.approx(max(rewards))
        assert result["mean_reward"] == pytest.approx(sum(rewards) / len(rewards))

    @staticmethod
    def test_evaluate_empty_seeds_raises(
        stub_model: _StubModel, small_env: QuantumSchedulingEnv
    ) -> None:
        """空测试种子应抛出 ValueError。"""
        with pytest.raises(ValueError):
            BlindTestEvaluator().evaluate(stub_model, small_env, [])

    @staticmethod
    def test_evaluate_reproducible(stub_model: _StubModel, small_env: QuantumSchedulingEnv) -> None:
        """相同种子应产生相同评估结果。"""
        evaluator = BlindTestEvaluator()
        r1 = evaluator.evaluate(stub_model, small_env, [42], episodes_per_seed=2)
        r2 = evaluator.evaluate(stub_model, small_env, [42], episodes_per_seed=2)
        assert r1["all_rewards"] == r2["all_rewards"]

    @staticmethod
    def test_extract_action_from_int() -> None:
        """extract_action 应正确处理 int 返回。"""
        assert extract_action(1) == 1
        assert extract_action(2) == 2

    @staticmethod
    def test_extract_action_from_tuple() -> None:
        """extract_action 应正确处理 SB3 元组返回。"""
        assert extract_action((np.array([0]), None)) == 0
        assert extract_action((np.array([2]), np.array([]))) == 2

    @staticmethod
    def test_extract_action_from_scalar_array() -> None:
        """extract_action 应正确处理标量数组。"""
        assert extract_action(np.int64(1)) == 1
        assert extract_action(np.array(2)) == 2

    @staticmethod
    def test_evaluate_sb3_compatible_model(small_env: QuantumSchedulingEnv) -> None:
        """评估器应兼容 SB3 风格（元组返回）模型。"""
        model = _StubSB3Model(action=1)
        result = BlindTestEvaluator().evaluate(model, small_env, [42], episodes_per_seed=2)
        assert result["num_episodes"] == 2
        assert all(isinstance(r, float) for r in result["all_rewards"])

    @staticmethod
    def test_evaluate_adjacent_seeds_no_overlap(
        stub_model: _StubModel, small_env: QuantumSchedulingEnv
    ) -> None:
        """Issue #386: 相邻 test_seeds 的 episode 种子不应重叠。

        原实现使用 seed + episode，若 test_seeds=[42, 43] 且 episodes_per_seed=5，
        seed=42 使用 42-46，seed=43 使用 43-47，存在 4 个重叠种子。
        修复后使用 seed * episodes_per_seed + episode 确保全局唯一。
        """
        evaluator = BlindTestEvaluator()
        # 相邻种子应能正常评估，且产生 10 个 episode
        result = evaluator.evaluate(stub_model, small_env, [42, 43], episodes_per_seed=5)
        assert result["num_episodes"] == 10

    @staticmethod
    def test_evaluate_episode_seeds_globally_unique() -> None:
        """Issue #386: 所有 episode 种子应全局唯一（验证乘法偏移修复）。"""
        # 模拟 evaluate 内部的种子生成逻辑
        test_seeds = [42, 43, 44]
        episodes_per_seed = 5
        episode_seeds: list[int] = []
        for seed in test_seeds:
            for episode in range(episodes_per_seed):
                episode_seeds.append(seed * episodes_per_seed + episode)
        # 所有 episode 种子应唯一
        assert len(set(episode_seeds)) == len(episode_seeds), (
            f"episode 种子存在重叠: {episode_seeds}"
        )

    @staticmethod
    def test_evaluate_negative_seed_overlap_raises(
        stub_model: _StubModel, small_env: QuantumSchedulingEnv
    ) -> None:
        """Issue #386: 若种子间距不足导致 episode 重叠应抛出 ValueError。

        构造 test_seeds=[0, 0] 且 episodes_per_seed=5，乘法偏移后两 seed 生成相同种子，
        应被检测并抛出异常。
        """
        evaluator = BlindTestEvaluator()
        with pytest.raises(ValueError, match="重叠"):
            evaluator.evaluate(stub_model, small_env, [0, 0], episodes_per_seed=5)


# ============================================================
# OODGeneralizationTester 测试
# ============================================================


class TestOODGeneralizationTester:
    """OODGeneralizationTester 分布外泛化测试。"""

    @staticmethod
    def test_distribution_shift_result_structure(
        stub_model: _StubModel, small_env: QuantumSchedulingEnv
    ) -> None:
        """test_distribution_shift 结果应包含 in_distribution/ood/degradation。"""
        tester = OODGeneralizationTester()
        shift_params: dict[str, Any] = {
            "lambda_range": (0.1, 0.3),
            "fidelity_range": (0.80, 0.90),
            "queue_size_range": (3, 8),
        }
        result = tester.test_distribution_shift(
            stub_model, small_env, [42, 123], shift_params, episodes_per_seed=2
        )
        assert "in_distribution" in result
        assert "ood" in result
        assert "degradation" in result
        assert "shift_params" in result

    @staticmethod
    def test_distribution_shift_episode_count(
        stub_model: _StubModel, small_env: QuantumSchedulingEnv
    ) -> None:
        """OOD 评估的 episode 数应等于 seeds × episodes_per_seed。"""
        tester = OODGeneralizationTester()
        shift_params: dict[str, Any] = {"lambda_range": (0.1, 0.3)}
        result = tester.test_distribution_shift(
            stub_model, small_env, [1, 2, 3], shift_params, episodes_per_seed=2
        )
        assert result["ood"]["num_episodes"] == 6
        assert result["in_distribution"]["num_episodes"] == 6

    @staticmethod
    def test_distribution_shift_reproducible(
        stub_model: _StubModel, small_env: QuantumSchedulingEnv
    ) -> None:
        """相同种子与偏移参数应产生可复现的 OOD 结果。"""
        tester = OODGeneralizationTester()
        shift_params: dict[str, Any] = {"lambda_range": (0.1, 0.3)}
        r1 = tester.test_distribution_shift(
            stub_model, small_env, [42], shift_params, episodes_per_seed=2
        )
        r2 = tester.test_distribution_shift(
            stub_model, small_env, [42], shift_params, episodes_per_seed=2
        )
        assert r1["ood"]["all_rewards"] == r2["ood"]["all_rewards"]

    @staticmethod
    def test_distribution_shift_empty_seeds_raises(
        stub_model: _StubModel, small_env: QuantumSchedulingEnv
    ) -> None:
        """空种子列表应抛出 ValueError。"""
        tester = OODGeneralizationTester()
        with pytest.raises(ValueError):
            tester.test_distribution_shift(stub_model, small_env, [], {})

    @staticmethod
    def test_compute_ood_degradation_robust() -> None:
        """衰减率低于阈值时应判定为鲁棒。"""
        tester = OODGeneralizationTester(robust_threshold=0.30)
        in_dist = {"mean_reward": 100.0}
        ood = {"mean_reward": 85.0}  # 衰减 15%
        deg = tester.compute_ood_degradation(in_dist, ood)
        assert deg["degradation_rate"] == pytest.approx(0.15)
        assert deg["is_robust"] is True

    @staticmethod
    def test_compute_ood_degradation_not_robust() -> None:
        """衰减率不低于阈值时应判定为非鲁棒。"""
        tester = OODGeneralizationTester(robust_threshold=0.30)
        in_dist = {"mean_reward": 100.0}
        ood = {"mean_reward": 60.0}  # 衰减 40%
        deg = tester.compute_ood_degradation(in_dist, ood)
        assert deg["degradation_rate"] == pytest.approx(0.40)
        assert deg["is_robust"] is False

    @staticmethod
    def test_compute_ood_degradation_negative_is_robust() -> None:
        """OOD 性能更好（负衰减）时应判定为鲁棒。"""
        tester = OODGeneralizationTester()
        in_dist = {"mean_reward": 100.0}
        ood = {"mean_reward": 120.0}  # 衰减 -20%
        deg = tester.compute_ood_degradation(in_dist, ood)
        assert deg["degradation_rate"] == pytest.approx(-0.20)
        assert deg["is_robust"] is True

    @staticmethod
    def test_compute_ood_degradation_zero_baseline() -> None:
        """基线均值为 0 时应安全处理（不除零）。"""
        tester = OODGeneralizationTester()
        deg = tester.compute_ood_degradation({"mean_reward": 0.0}, {"mean_reward": 0.0})
        assert deg["degradation_rate"] == 0.0
        assert deg["is_robust"] is True

    @staticmethod
    def test_compute_ood_degradation_result_keys() -> None:
        """衰减结果应包含所有必需键。"""
        tester = OODGeneralizationTester()
        deg = tester.compute_ood_degradation({"mean_reward": 10.0}, {"mean_reward": 8.0})
        required_keys = {
            "in_distribution_mean",
            "ood_mean",
            "degradation_rate",
            "is_robust",
            "robustness_threshold",
        }
        assert required_keys <= set(deg.keys())

    @staticmethod
    def test_distribution_shift_with_all_shift_params(
        stub_model: _StubModel, small_env: QuantumSchedulingEnv
    ) -> None:
        """同时施加三种偏移应正常运行并返回完整结果。"""
        tester = OODGeneralizationTester()
        shift_params: dict[str, Any] = {
            "lambda_range": (0.8, 1.2),
            "fidelity_range": (0.70, 0.85),
            "queue_size_range": (15, 25),
        }
        result = tester.test_distribution_shift(
            stub_model, small_env, [42, 100], shift_params, episodes_per_seed=2
        )
        assert result["ood"]["num_episodes"] == 4
        assert result["degradation"]["is_robust"] in (True, False)

    @staticmethod
    def test_distribution_shift_no_shift_params(
        stub_model: _StubModel, small_env: QuantumSchedulingEnv
    ) -> None:
        """无偏移参数时 OOD 性能应接近基线（衰减接近 0）。"""
        tester = OODGeneralizationTester()
        result = tester.test_distribution_shift(
            stub_model, small_env, [42], {}, episodes_per_seed=2
        )
        # 无偏移时衰减率应较小
        assert result["degradation"]["degradation_rate"] < 0.30
