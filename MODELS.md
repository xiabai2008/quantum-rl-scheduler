# 权威模型检查点归档（MODELS.md）

> 本文件说明本项目**可提交的训练好的模型检查点**，用于保证评审在克隆仓库后能复现论文中的实验结果（PPO vs FCFS +88.3% 等）。
> 最后更新：2026-07-21

## 为什么需要本目录

原 `models/` 与 `results/` 被 `.gitignore` 整体忽略，**不会被提交到仓库**。若直接克隆，评估脚本将因加载不到模型而失败，实验数字无法复现。

为此，本目录 `deliverable_models/` 存放**官方锁定的权威模型副本**（体积小，约 0.5 MB），已通过 `.gitignore` 例外规则纳入版本控制，并登记在 `config/submission_manifest.yaml`（项 `MODEL_PPO` / `MODEL_DQN`）。

## 权威模型清单

| 策略 | 提交路径（可复现） | 训练说明 | 体积 | 复现指标 |
|------|-------------------|----------|------|----------|
| **PPO（14维）** | `deliverable_models/ppo_best_model_14dim.zip` | 14维原生环境，50000 steps，seed=42 | ~267 KB | PPO 奖励 **2746.94**（50 seed × 5 ep，+88.3% vs FCFS，d=1.70） |
| **DQN（14维）** | `deliverable_models/dqn_best_model_14dim.zip` | 14维原生环境，50000 steps，Double DQN + reward clip | ~216 KB | DQN 奖励 **-897.08**（排名 6/8，在14维环境退化为近Quantum-Only） |
| **DQN（10维）** | `deliverable_models/dqn_best_model_10dim.zip` | 10维环境（Obs10Wrapper），50000 steps | ~216 KB | DQN 奖励 **-897.08**（排名 6/8，在14维环境退化为近Quantum-Only） |

## 训练配置（复现前提）

- **PPO 观测空间**：14 维（原生 `QuantumSchedulingEnv`，含队列长度、量子保真度、等待时间、拓扑连接度等）
- **DQN 观测空间**：10 维（`Obs10Wrapper` 截断，仅用于基线对比，不作权威提交）
- **随机种子**：42
- **评测规模**：200 步/episode、泊松到达 λ=0.5、10 seeds × 5 episodes
- **PPO 训练量**：50,000 timesteps
- **DQN 训练量**：50,000 timesteps

## 复现命令

```bash
# 1) 多 seed 评估 + 统计显著性（14维 PPO，10 seed × 5 ep）
python scripts/evaluation/run_multiseed_evaluation.py --seeds 10 --episodes 5 \
    --ppo-model deliverable_models/ppo_best_model_14dim.zip

# 2) 统计显著性检验
python scripts/evaluation/statistical_significance.py \
    --input results/multiseed_evaluation/rewards_multiseed_14dim.json

# 3) 压力测试（自动加载 deliverable_models/ 下的 PPO 14维模型）
python scripts/benchmarking/stress_test.py

# 4) 多机器演示（加载 PPO 14维模型）
python scripts/demo/demo_multi_machine.py --ppo-model deliverable_models/ppo_best_model_14dim.zip
```

## 最终权威指标（答辩统一口径）

| 指标 | 数值 | 统计检验 |
|:--|:--|:--|
| PPO vs FCFS 提升 | **+88.3%** | Welch t, p=3.04×10⁻¹¹, d=1.70 |
| PPO 平均奖励 | 2746.94（SD=1160.72，SE=73.41，N=250：50 seeds × 5 episodes） | — |
| FCFS 平均奖励 | 1457.0 ± 30（SE=10） | 基线 |
| 多机器 MAPPO | +86.3% vs 单机 PPO | — |
| 真机验证 | 32 任务 100% 成功率 | 可用性验证 |

## 注意事项

- PPO 权威模型已从 10 维升级到 14 维（v8→v9），旧 `ppo_best_model_10dim.zip` 保留作为历史参考但不再作为主要口径
- DQN 14维版本已归档：`deliverable_models/dqn_best_model_14dim.zip`（Double DQN + reward clip，Issue #46）。10维版本 `dqn_best_model_10dim.zip` 保留作为历史参考。在 14 维评估环境中 DQN 退化为近 Quantum-Only 行为（已知限制，归因于观测空间不匹配退化）
- 如需重新训练并替换权威模型：训练完成后将 `best_model.zip` 复制至 `deliverable_models/` 并同步更新本文件
- `models/` 目录已删除（旧版训练 artifacts），所有交付模型统一在 `deliverable_models/`
