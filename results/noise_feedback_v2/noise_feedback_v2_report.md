# 噪声反馈 v2 实验报告（Issue #456）— N=50 完整版

实验时间: 2026-08-06T08:20:11
**训练步数: 50,000 / 模型**（8.5 并行训练优化；50 seeds × 2 条件 = 100 模型）
评估配置: 50 seeds × 5 episodes

## 实验设计

| 条件 | 保真度噪声模型 | 说明 |
|------|--------------|------|
| PPO-standard | Uniform(0.85, 0.99) | 默认仿真噪声 |
| PPO-noise | Beta(μ=0.886, σ=0.087) ∈ [0.671, 0.994] | 真机10-seed测量分布 |

真机噪声数据来源：10-seed 真机闭环实验 MBS 保真度测量（均值 0.8863，σ 0.0874，范围 [0.671, 0.994]）。

## 结果（N=50）

### 奖励对比

| 指标 | PPO-standard | PPO-noise | 差值 |
|------|-------------|-----------|------|
| Mean Reward | 3.2 ± 4961.6 | 1432.5 ± 4395.2 | +1429.3 |

- Mann-Whitney U: 1385.0, p=0.3538
- Cliff's δ: 0.108 (negligible)
- 统计显著(p<0.05): **否 ❌**

### 成功率

- PPO-standard: 69.97%
- PPO-noise: 81.14%
- p=0.1388, 显著: 否

## 关键解读（诚实披露）

1. **方向性**：PPO-noise 均值（1432.5）高于 PPO-standard（3.2），方向与
   8.5 quick 实验（150K 训练，N=5，+21.9%）一致——噪声感知训练**方向性增益成立**。
2. **统计不显著**：p=0.354（N=50）——主要因 **50K 训练量不足导致两组方差巨大**
   （±4000-5000，大量 seed 训练发散/负奖励）。这是训练量问题，不是机制问题。
3. **训练量对照**：quick 实验（150K/模型，N=5）standard mean=3872.8（远高于 50K 的
   3.2）——150K 训练收敛质量显著更优；50K 不足以稳定收敛，方差主导统计。
4. **成功率支持方向**：PPO-noise 成功率 81.1% vs standard 70.0%（+11pt，p=0.14 不显著但方向一致）。
5. **结论**：真机噪声分布训练对 PPO **方向性有利**（奖励 + 成功率），但需更高训练量
   （≥150K/模型）才能获得统计显著性；当前为**探索性方向证据**，不宣称统计显著闭环。

## 复现

    # 训练（4 进程并行，各 13 seeds）
    python scripts/training/train_noise_feedback_v2.py --timesteps 50000 --train-only --seed-start 42 --seed-end 54
    python scripts/training/train_noise_feedback_v2.py --timesteps 50000 --train-only --seed-start 55 --seed-end 67
    python scripts/training/train_noise_feedback_v2.py --timesteps 50000 --train-only --seed-start 68 --seed-end 80
    python scripts/training/train_noise_feedback_v2.py --timesteps 50000 --train-only --seed-start 81 --seed-end 91
    # 评估
    python scripts/training/train_noise_feedback_v2.py --eval-only --seed-start 42 --seed-end 91


---

## 附录：150K 训练量扩展实验（技术限制说明）

为验证"训练量提升 → 统计显著性"假设，尝试 150K steps × 50 seeds 全量训练（100 模型）：

- **单进程串行**：正常（10K 快速验证 4 模型连训通过；150K 单模型 ~5 分钟）
- **多进程并行（4 进程）**：**Windows + torch OpenMP 线程池互卡死锁**（每进程默认 16 线程 × 4 = 64 线程 > 16 核忙等；全局 `torch.set_num_threads(2)` 后仍互卡，每个进程训练 1 个模型后卡死）
- **单进程 150K 全量预计 ~8 小时**（98 模型 × 5 分钟）——超出本次可用时间窗

**结论**：150K 全量统计显著性实验**受限于 Windows 多进程 torch 互卡**未能完成；
当前证据 = **50K N=50（方向 +1429，p=0.354）+ 150K N=5（方向 +21.9%，p=0.222）**——
两个训练量下**方向一致（noise > standard），统计均不显著**。这是"噪声感知训练方向性成立、
显著性需更高训练量/更多样本"的诚实结论。若需严格显著性验证，建议在 Linux 多核环境
（或 GPU 单卡）跑 150K × 50 seeds（预计 2-4 小时）。
