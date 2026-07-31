"""分布外（OOD）泛化验证模块。

本模块提供 ``OODGeneralizationTester``，通过对环境分布参数施加偏移
（泊松到达率、量子保真度、初始队列大小），对比模型在原始分布与偏移分布下
的性能衰减，量化模型的分布外（Out-of-Distribution）泛化鲁棒性。

判定准则（升级为双条件）：衰减率低于阈值 且 衰减不具统计显著性时视为鲁棒。
"""

from __future__ import annotations

from typing import Any

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from src.evaluation.blind_test import (
    BlindTestEvaluator,
    PredictableModel,
    extract_action,
    summarize_rewards,
)
from src.scheduler.env import QuantumSchedulingEnv

#: 鲁棒性判定阈值：性能衰减率低于此值视为鲁棒（默认 30%）。
ROBUST_DEGRADATION_THRESHOLD = 0.30

#: 偏移参数采样 RNG 的种子偏移量（避免与 episode 种子冲突）。
_SHIFT_RNG_OFFSET = 7777


class OODGeneralizationTester:
    """测试模型在分布外场景的泛化能力。

    通过对环境分布参数（泊松到达率、量子保真度、队列大小）施加偏移，
    对比模型在原始分布与偏移分布下的性能衰减，量化模型的 OOD 鲁棒性。

    统计增强：当提供两组逐 episode 奖励列表时，衰减分析额外包含
    Bootstrap 95% CI、Bonferroni 校正 p 值、Cohen's d 效应量及等级，
    并将 ``is_robust`` 判定升级为「阈值 + 非显著衰减」双条件。

    Args:
        evaluator: 盲测评估器实例；None 时使用默认评估器（确定性推理）。
        robust_threshold: 鲁棒性判定阈值，性能衰减率低于此值视为鲁棒，
            默认 0.30（30%）。
    """

    def __init__(
        self,
        evaluator: BlindTestEvaluator | None = None,
        robust_threshold: float = ROBUST_DEGRADATION_THRESHOLD,
    ) -> None:
        """初始化 OOD 泛化测试器。

        Args:
            evaluator: 盲测评估器实例；None 时使用默认评估器。
            robust_threshold: 鲁棒性判定阈值，默认 0.30。
        """
        self.evaluator = evaluator if evaluator is not None else BlindTestEvaluator()
        self.robust_threshold = robust_threshold

    def test_distribution_shift(
        self,
        model: PredictableModel,
        env: QuantumSchedulingEnv,
        seeds: list[int],
        shift_params: dict[str, Any],
        episodes_per_seed: int = 5,
    ) -> dict[str, Any]:
        """测试模型在分布偏移下的性能。

        首先在原始分布（提供的 ``env``）上评估基线性能，然后构造偏移分布
        环境并在其上评估模型性能，最后计算性能衰减（含统计增强）。

        偏移参数采样由 episode 种子确定性驱动，保证结果可复现。

        Args:
            model: 待评估模型。
            env: 原始分布的调度环境（作为基线与偏移环境模板）。
            seeds: 评估种子列表。
            shift_params: 偏移参数字典，可包含：
                - ``lambda_range``: ``(min, max)`` 泊松到达率偏移范围
                - ``fidelity_range``: ``(min, max)`` 量子保真度偏移范围
                - ``queue_size_range``: ``(min, max)`` 初始队列大小偏移范围
            episodes_per_seed: 每个种子的 episode 数，默认 5。

        Returns:
            包含以下键的字典：
            - ``in_distribution``: 原始分布评估结果
            - ``ood``: 偏移分布评估结果
            - ``degradation``: 性能衰减分析结果（含统计显著性）
            - ``shift_params``: 使用的偏移参数

        Raises:
            ValueError: ``seeds`` 为空。
        """
        if not seeds:
            raise ValueError("seeds 列表不能为空")

        # 基线：原始分布
        in_dist_results = self.evaluator.evaluate(model, env, seeds, episodes_per_seed)

        # OOD：偏移分布
        ood_rewards: list[float] = []
        max_steps = env.max_steps
        for seed in seeds:
            for episode in range(episodes_per_seed):
                # Issue #681: 与 blind_test 保持一致的种子计算
                episode_seed = seed * episodes_per_seed + episode
                episode_reward = self._run_shifted_episode(
                    model, env, shift_params, episode_seed, max_steps
                )
                ood_rewards.append(episode_reward)

        ood_results = summarize_rewards(ood_rewards)
        degradation = self.compute_ood_degradation(
            in_dist_results,
            ood_results,
            in_distribution_rewards=in_dist_results["all_rewards"],
            ood_rewards=ood_rewards,
        )

        logger.info(
            "OOD 测试完成：in_dist={:.2f}, ood={:.2f}, degradation={:.2%}, robust={}",
            in_dist_results["mean_reward"],
            ood_results["mean_reward"],
            degradation["degradation_rate"],
            degradation["is_robust"],
        )

        return {
            "in_distribution": in_dist_results,
            "ood": ood_results,
            "degradation": degradation,
            "shift_params": dict(shift_params),
        }

    def compute_ood_degradation(
        self,
        in_distribution_results: dict[str, Any],
        ood_results: dict[str, Any],
        in_distribution_rewards: list[float] | None = None,
        ood_rewards: list[float] | None = None,
    ) -> dict[str, Any]:
        """计算分布外性能衰减。

        衰减率定义为 ``(原始均值 - OOD均值) / |原始均值|``，正值表示性能下降，
        负值表示 OOD 性能更好。当原始均值接近 0 时退化为符号判定。

        当同时提供 ``in_distribution_rewards`` 和 ``ood_rewards`` 时，
        额外返回衰减率的 Bootstrap 95% CI、In-Dist vs OOD 差异显著性
        （p 值 + Cohen's d 效应量），并将 ``is_robust`` 判定从纯阈值升级为
        「衰减率 < 阈值 且 衰减不具统计显著性」双条件。

        Args:
            in_distribution_results: 原始分布评估结果（须含 ``mean_reward``）。
            ood_results: 偏移分布评估结果（须含 ``mean_reward``）。
            in_distribution_rewards: 原始分布的逐 episode 奖励列表，用于
                统计显著性检验；默认 ``None`` 不执行统计增强。
            ood_rewards: 偏移分布的逐 episode 奖励列表，用于统计检验；
                默认 ``None`` 不执行统计增强。

        Returns:
            包含以下键的衰减结果字典：
            - ``in_distribution_mean``: 原始分布平均奖励
            - ``ood_mean``: 偏移分布平均奖励
            - ``degradation_rate``: 衰减率，正值表示性能下降
            - ``is_robust``: 是否鲁棒（阈值 + 非显著衰减双条件）
            - ``robustness_threshold``: 鲁棒性阈值
            - ``degradation_ci95_lower``: 衰减率 95% CI 下界（nan 表示未计算）
            - ``degradation_ci95_upper``: 衰减率 95% CI 上界（nan 表示未计算）
            - ``significance_p_value``: In-Dist vs OOD 显著性 p 值（nan）
            - ``effect_size_cohens_d``: 效应量 Cohen's d（nan 表示未计算）
            - ``effect_size_level``: 效应量等级（None/字符串）
            - ``statistically_significant_degradation``: 是否统计显著衰减
        """
        in_mean = float(in_distribution_results["mean_reward"])
        ood_mean = float(ood_results["mean_reward"])

        if abs(in_mean) < 1e-9:
            # 基线接近 0 时：OOD 也接近 0 视为无衰减，否则视为完全衰减
            degradation_rate = 0.0 if abs(ood_mean) < 1e-9 else 1.0
        else:
            degradation_rate = (in_mean - ood_mean) / abs(in_mean)

        # ---- 统计增强：CI / 显著性 / 效应量 ----
        deg_ci_lo: float = float("nan")
        deg_ci_hi: float = float("nan")
        sig_p: float = float("nan")
        es_d: float = float("nan")
        es_level: str | None = None
        sig_deg: bool = False

        both_rewards_provided = (
            in_distribution_rewards is not None
            and ood_rewards is not None
            and len(in_distribution_rewards) >= 2
            and len(ood_rewards) >= 2
        )

        if both_rewards_provided:
            from src.utils.stats_significance import (
                _effect_level,
                bootstrap_improvement_ci,
                cohen_d,
                compare_strategies,
            )

            # mypy 类型推断：显式缩窄为 list[float]（已在 both_rewards_provided 校验非 None）
            assert in_distribution_rewards is not None
            assert ood_rewards is not None
            idr: list[float] = in_distribution_rewards
            odr: list[float] = ood_rewards

            # 1) 衰减率 Bootstrap 95% CI
            # bootstrap_improvement_ci 计算 (target - baseline) / |baseline| * 100
            # 这里 target = ood, baseline = in_distribution
            # 衰减率 degradation_rate = -improvement_pct / 100
            # 因此 CI 上下界取反并除以 100
            _imp_pct, imp_ci_lo, imp_ci_hi = bootstrap_improvement_ci(odr, idr, confidence=0.95)
            if not (np.isnan(imp_ci_lo) or np.isnan(imp_ci_hi)):
                deg_ci_lo = -imp_ci_hi / 100.0
                deg_ci_hi = -imp_ci_lo / 100.0

            # 2) In-Dist vs OOD 差异显著性 + 效应量
            cmp_res = compare_strategies({"In-Dist": idr, "OOD": odr})
            pair_key = "In-Dist vs OOD"
            if pair_key in cmp_res:
                pair = cmp_res[pair_key]
                sig_p = float(pair.get("p_value", float("nan")))
                # compare_strategies 可能用 Cohen's d 或 rank-biserial
                if pair.get("effect_size_type") == "Cohen's d":
                    es_d = float(pair.get("effect_size", float("nan")))
                else:
                    # 直接用 cohen_d 计算一次确保 d 值
                    es_d = cohen_d(idr, odr)
                if not np.isnan(es_d):
                    es_level = _effect_level(es_d, "Cohen's d")
                significant = bool(pair.get("significant", False))
                mean_diff_positive = in_mean > ood_mean  # 正衰减
                sig_deg = significant and mean_diff_positive

        # ---- is_robust 双条件：阈值 AND 非显著衰减
        is_robust = (degradation_rate < self.robust_threshold) and (not sig_deg)

        return {
            "in_distribution_mean": in_mean,
            "ood_mean": ood_mean,
            "degradation_rate": float(degradation_rate),
            "is_robust": bool(is_robust),
            "robustness_threshold": float(self.robust_threshold),
            "degradation_ci95_lower": deg_ci_lo,
            "degradation_ci95_upper": deg_ci_hi,
            "significance_p_value": sig_p,
            "effect_size_cohens_d": es_d,
            "effect_size_level": es_level,
            "statistically_significant_degradation": sig_deg,
        }

    def _run_shifted_episode(
        self,
        model: PredictableModel,
        template_env: QuantumSchedulingEnv,
        shift_params: dict[str, Any],
        seed: int,
        max_steps: int,
    ) -> float:
        """运行单个偏移分布 episode 并返回累计奖励。

        根据模板环境构造偏移分布环境，在 ``reset`` 后施加保真度与队列大小偏移，
        然后用模型运行完整 episode。

        Args:
            model: 待评估模型。
            template_env: 模板环境（读取其配置构造偏移环境）。
            shift_params: 偏移参数字典。
            seed: 随机种子（同时驱动偏移参数采样，保证可复现）。
            max_steps: 最大步数。

        Returns:
            本 episode 累计奖励。
        """
        shifted_env = self._make_shifted_env(template_env, shift_params, seed)
        try:
            obs, _ = shifted_env.reset(seed=seed)
            rng = np.random.default_rng(seed + _SHIFT_RNG_OFFSET)

            obs = self._apply_shifts(shifted_env, shift_params, rng)

            episode_reward = 0.0
            for _ in range(max_steps):
                action = extract_action(
                    model.predict(obs, deterministic=self.evaluator.deterministic)
                )
                obs, reward, terminated, truncated, _ = shifted_env.step(action)
                episode_reward += float(reward)
                if terminated or truncated:
                    break
            return episode_reward
        finally:
            shifted_env.close()

    def _make_shifted_env(
        self,
        template: QuantumSchedulingEnv,
        shift_params: dict[str, Any],
        seed: int,
    ) -> QuantumSchedulingEnv:
        """根据偏移参数从模板环境构造偏移分布环境。

        当提供 ``lambda_range`` 时，从中采样偏移后的泊松到达率；
        其余环境配置（步数、量子比特、机器列表、量子任务占比）继承自模板。

        Args:
            template: 模板环境。
            shift_params: 偏移参数字典。
            seed: 随机种子（驱动到达率采样）。

        Returns:
            偏移分布的新环境实例。
        """
        arrival_lambda: float | Any = template.arrival_lambda
        if "lambda_range" in shift_params:
            lo, hi = shift_params["lambda_range"]
            rng = np.random.default_rng(seed + _SHIFT_RNG_OFFSET)
            arrival_lambda = float(rng.uniform(lo, hi))

        machine_configs = [
            {
                "name": m.name,
                "total_qubits": m.total_qubits,
                "supported_gates": m.supported_gates,
                "is_real": m.is_real,
            }
            for m in template._machines
        ]

        return QuantumSchedulingEnv(
            max_steps=template.max_steps,
            max_qubits=template.max_qubits,
            machine_configs=machine_configs,
            arrival_lambda=arrival_lambda,
            quantum_task_ratio=template.quantum_task_ratio,
        )

    def _apply_shifts(
        self,
        env: QuantumSchedulingEnv,
        shift_params: dict[str, Any],
        rng: np.random.Generator,
    ) -> NDArray[Any]:
        """在 reset 后对环境施加保真度与队列大小偏移。

        Args:
            env: 已 reset 的调度环境实例。
            shift_params: 偏移参数字典。
            rng: 随机数生成器。

        Returns:
            偏移后的最新观测向量。
        """
        # 施加保真度偏移
        if "fidelity_range" in shift_params:
            lo, hi = shift_params["fidelity_range"]
            for m in env._machines:
                m.fidelity = float(rng.uniform(lo, hi))
                m.update_noise_features(rng)
            env._recompute_aggregate()

        # 施加队列大小偏移
        if "queue_size_range" in shift_params:
            lo, hi = shift_params["queue_size_range"]
            target_size = int(rng.integers(lo, hi + 1))
            self._adjust_queue(env, target_size, rng)

        return env._get_observation()

    @staticmethod
    def _adjust_queue(
        env: QuantumSchedulingEnv,
        target_size: int,
        rng: np.random.Generator,
    ) -> None:
        """调整环境任务队列至目标大小。

        队列过长时从末尾移除；过短时用环境任务生成器补充。
        调整后重新选取队首任务。

        Args:
            env: 调度环境实例。
            target_size: 目标队列大小。
            rng: 随机数生成器。
        """
        queue = env._task_queue
        while len(queue) > target_size:
            queue.pop()
        while len(queue) < target_size:
            queue.append(env._generate_random_task(rng, len(queue)))
        env._pick_next_task()
