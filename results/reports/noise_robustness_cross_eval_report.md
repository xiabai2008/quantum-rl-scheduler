# 噪声鲁棒性交叉评估报告（预注册式，D 档结论）

> 日期：2026-08-09 | 状态：完成（48 seeds，QA 对账通过）
> 实验脚本：`scripts/evaluation/noise_robustness_cross_eval.py`
> 原始数据：`results/reports/noise_robustness_cross_eval_20260809.json`

## 1. 实验目的

在"量子赋能 AI"主方向下，验证用真机噪声分布训练的 PPO 是否能带来可检测的鲁棒性增益。
若成立，可将噪声敏感性负向证据（-12.43%, p=2.98e-08）反转为正向的"策略选择准则"。

## 2. 实验设计（预注册）

- **零重训**：复用 `models/noise_feedback_v2/` 下 48 对 (standard, noise) 模型
- **6 格子交叉评估**：SS / SN / NS / NN（噪声 profile）+ SM / NM（MBS 乘子注入）
- **评估路径与 v2 权威版完全一致**：`PPOAgent.evaluate(deterministic=True)`，
  env 用 `seed+10000` + `DEFAULT_MACHINE_CONFIGS` + `max_steps=500`，5 episodes/seed
- 假设：H1 带噪环境 noise 模型更优；H1′ 跨噪声类型鲁棒；H2 标准环境非劣效；H3 鲁棒缺口缩小

## 3. QA 对账（可信度关键步骤）

| 格子 | 实测均值 | v2 权威 | 判定 |
|:--|:--|:--|:--|
| A (SS) | 4105.7 | 4115.5 | OK（偏差 0.2%） |
| D (NN) | 4102.9 | 4176.2 | OK（偏差 1.8%） |

评估路径与权威版完全一致，结果可信。

## 4. 六格子结果（N=48）

| 格子 | 均值 | 说明 |
|:--|:--|:--|
| SS | 4105.7 | standard × standard |
| SN | 4115.3 | standard × 真机噪声 |
| NS | 3857.9 | noise × standard |
| NN | 4102.9 | noise × 真机噪声 |
| SM | 3678.5 | standard × MBS 注入 |
| NM | 3405.0 | noise × MBS 注入 |

## 5. 假设检验结果

| 假设 | 成功标准 | 实测 | 判定 |
|:--|:--|:--|:--|
| H1 (NN>SN) | p<0.05 且 d_z≥0.2 | p=0.583, d_z=-0.01, mean_diff=-12.4 | 不成立 |
| H1′ (NM>SM) | p<0.05 | p=0.967, d_z=-0.29（NM 反低 7.4%） | 不成立 |
| H2 (NS 非劣效 SS) | CI 下界 > -2% | NS 比 SS 低 6.0%（p=0.059, d_z=0.20） | 不成立 |
| H3 (缺口缩小) | G_std > G_noise | G_std=9.6 vs G_noise=245.0（方向相反） | 不成立 |

## 6. 结论（诚实披露）

1. **100K 噪声训练未带来可检测的鲁棒性增益**：四个假设全不显著，H1′/H3 甚至方向相反。
2. 边缘信号 NS<SS（p=0.059）提示噪声训练在标准环境可能有轻微 trade-off。
3. 与 v2 权威结论自洽（两条件区间内差距仅 +1.5% 不显著）：真机噪声对 RL 是挑战，
   噪声感知训练尚不足以反转该结论。
4. **定位**：不宣称"策略选择准则"成立；作为"噪声感知训练工程闭环 + 方向性证据"（沿用 v2
   诚实定位）。答辩中可作为诚实性加分项，不作为正向性能证据。

## 7. 复现命令

```bash
python scripts/evaluation/noise_robustness_cross_eval.py --episodes 5 --canonical
```
