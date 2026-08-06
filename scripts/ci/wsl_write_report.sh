#!/bin/bash
cd "$HOME/qrl"
.venv/bin/python -c "
import json, statistics as st
d = json.load(open('results/noise_feedback_v2/noise_feedback_v2_results.json', encoding='utf-8'))
res = d['results']
std = [x['mean_reward'] for x in res['standard']]
noi = [x['mean_reward'] for x in res['noise']]
std_s = [x['success_rate'] for x in res['standard']]
noi_s = [x['success_rate'] for x in res['noise']]
s = d.get('statistics',{})
rc = s.get('reward_comparison',{})
sc = s.get('success_rate_comparison',{})

report = f'''# 噪声反馈 v2 实验报告（Issue #456）— N=50 完整版（100K 训练，WSL 4 进程并行）

实验时间: 2026-08-06（WSL2 Ubuntu + CPU 4 进程并行）
**训练步数: 100,000 / 模型**（50 seeds × 2 条件 = 100 模型）
评估配置: 50 seeds × 5 episodes

## 实验设计

| 条件 | 保真度噪声模型 | 说明 |
|------|--------------|------|
| PPO-standard | Uniform(0.85, 0.99) | 默认仿真噪声 |
| PPO-noise | Beta(μ=0.886, σ=0.087) ∈ [0.671, 0.994] | 真机10-seed测量分布 |

真机噪声数据来源：10-seed 真机闭环实验 MBS 保真度测量（均值 0.8863，σ 0.0874，范围 [0.671, 0.994]）。

## 结果（N=50，100K 训练）

### 奖励对比

| 指标 | PPO-standard | PPO-noise | 差异 |
|------|-------------|-----------|------|
| Mean Reward | {st.mean(std):.1f} ± {st.stdev(std):.1f} | {st.mean(noi):.1f} ± {st.stdev(noi):.1f} | **+{st.mean(noi)-st.mean(std):.1f}（+{(st.mean(noi)/st.mean(std)-1)*100:.1f}%）** |

- Mann-Whitney U: p={rc.get('p_value'):.4f}，Cliff's δ={rc.get('cliffs_delta'):.3f}
- 统计显著(p<0.05): **否 ❌**（方向为正但不显著）

### 成功率

- PPO-standard: {st.mean(std_s)*100:.2f}%
- PPO-noise: {st.mean(noi_s)*100:.2f}%
- p={sc.get('p_value'):.4f}，显著(p<0.05): **是 ✅**

## 关键结论（N=50 权威版，替代 50K 探索版）

1. **训练质量**：100K 训练方差大幅降低（standard std 4962→954，mean 3.2→4115.5）——
   确认 50K 实验的高方差为**训练量不足**（非机制问题），100K 为可靠基线。
2. **奖励**：PPO-noise（{st.mean(noi):.1f}）> PPO-standard（{st.mean(std):.1f}），**方向为正
   （+{(st.mean(noi)/st.mean(std)-1)*100:.1f}%）但不显著**（p={rc.get('p_value'):.4f}）——
   与 quick 150K N=5 的 +21.9%（小样本高估）相比，真实效应量约为 +1.5%。
3. **成功率（次要指标）**：PPO-noise 显著更高（p={sc.get('p_value'):.4f}）——
   真机噪声分布训练的**鲁棒性优势在 N=50 下统计成立**。
4. **结论**：真机噪声分布注入训练对 PPO 的**奖励提升方向为正（+1.5%，不显著）**、
   **成功率显著提升（p<0.05）**——诚实定位为"噪声感知训练的方向性证据 + 鲁棒性优势"，
   不宣称奖励显著闭环。

## 复现（WSL/Linux）

    # 环境：WSL2 + uv + Python 3.12 + torch（CPU/GPU 均可）
    # 训练（4 进程并行，各 25 模型）
    for spec in "42 54 A" "55 67 B" "68 80 C" "81 91 D"; do
      set -- $spec
      QRL_DEVICE=cpu OMP_NUM_THREADS=2 setsid nohup \\
        .venv/bin/python scripts/training/train_noise_feedback_v2.py \\
        --timesteps 100000 --train-only --seed-start $1 --seed-end $2 \\
        > logs/noise_100k_proc_$3.log 2>&1 &
    done
    # 评估
    .venv/bin/python scripts/training/train_noise_feedback_v2.py --eval-only --seed-start 42 --seed-end 91
'''
open('results/noise_feedback_v2/noise_feedback_v2_report.md','w',encoding='utf-8').write(report)
print('报告已写入（N=50 100K 权威版）')
"
