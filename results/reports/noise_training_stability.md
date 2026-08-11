# 噪声训练稳定性分析（量子→AI 方向性证据）

> 日期：2026-08-11 | 数据：`results/noise_feedback_v2/noise_feedback_v2_results.json`（50 seeds）
> 目的：检验"真机噪声分布训练的模型是否更稳定"——量子→AI 方向的稳健性方向性证据

## 1. 方法

- **48 对 (standard, noise) 模型**（noise_feedback_v2 实验，同一训练管道仅噪声注入不同）
- 指标：per-seed 评估 std（std_reward，5 episodes 评估的标准差）
- 检验：配对 t + Wilcoxon（同 seed 配对）

## 2. 结果

| 指标 | standard | noise | 差异 |
|:--|--:|--:|--:|
| per-seed std 均值 | 1943.3 | 1799.6 | **-7.4%** |
| 变异系数 CV（std/mean） | 0.505 | 0.464 | -8.1% |
| Wilcoxon p | — | — | **0.0876**（边缘） |
| 配对 t p | — | — | 0.1411 |

## 3. 结论（方向性，诚实披露）

1. **噪声训练模型稳定性方向性提升**：std 降低 7.4%（CV 降低 8.1%），
   Wilcoxon p=0.0876 处于边缘（未达 0.05，样本量 N=48 功效有限）。
2. **定位**：作为**方向性证据**——量子噪声数据训练让 AI 策略略更稳定；
   **不宣称统计显著**（与噪声训练 +1.5% p=0.584 同级别定位）。
3. 与噪声敏感性机制证据（p=2.98e-08）和跨机器异质性（p=0.0197）一起，
   构成量子→AI"评估可信度 + 稳健性建模赋能"的三角证据链。

## 4. 复现

```python
# noise_feedback_v2_results.json results.standard/noise 的 std_reward 字段
# 配对 Wilcoxon: scipy.stats.wilcoxon(std_list, nse_list)
```
