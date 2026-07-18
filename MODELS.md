# 权威模型检查点归档（MODELS.md）

> 本文件说明本项目**可提交的训练好的模型检查点**，用于保证评审在克隆仓库后能复现论文中的实验结果（PPO vs FCFS +86.9% 等）。
> 最后更新：2026-07-10

## 为什么需要本目录

原 `models/` 与 `results/` 被 `.gitignore` 整体忽略，**不会被提交到仓库**。若直接克隆，评估脚本将因加载不到模型而失败，实验数字无法复现。

为此，本目录 `deliverable_models/` 存放**官方锁定的权威模型副本**（体积小，约 0.5 MB），已通过 `.gitignore` 例外规则纳入版本控制，并登记在 `config/submission_manifest.yaml`（项 `MODEL_PPO` / `MODEL_DQN`）。

## 权威模型清单

| 策略 | 提交路径（可复现） | 来源（训练产物，不入库） | 体积 | 复现指标 |
|------|-------------------|--------------------------|------|----------|
| **PPO** | `deliverable_models/ppo_best_model_10dim.zip` | `models/ppo_seed_42_v4/best_model.zip` | ~261 KB | PPO 奖励 **2723.0**（10 seed × 5 ep，+86.9% vs FCFS） |
| **DQN** | `deliverable_models/dqn_best_model_10dim.zip` | `models/dqn_fair_v2/seed_42/best_model.zip` | ~216 KB | DQN 奖励 **-897.08**（排名第 6/8） |

## 训练配置（复现前提）

- **观测空间**：10 维（`Obs10Wrapper`，将 14 维环境观测截断为 10 维供旧模型使用）
- **随机种子**：42
- **评测规模**：200 步/episode、泊松到达 λ=0.5、多 seed 平均
- **PPO 训练量**：约 50,000 steps（`models/ppo_seed_42_v4/` 下含完整 checkpoint 序列）
- **DQN 训练量**：约 50,000 steps（`models/dqn_fair_v2/seed_42/` 下含完整 checkpoint 序列）

## 复现命令

```bash
# 1) 8 策略对比（加载 PPO/DQN 权威模型，复现 +86.9%）
python scripts/real_machine/strategy_comparison.py

# 2) 多 seed 评估 + 统计显著性（10 seed × 5 ep）
python scripts/evaluation/run_multiseed_evaluation.py --seeds 10 --episodes 5
python scripts/evaluation/statistical_significance.py \
    --input results/multiseed_evaluation/rewards_multiseed.json

# 3) 压力测试（自动优先加载 deliverable_models/ 下的 PPO 模型）
python scripts/benchmarking/stress_test.py

# 4) 多机器演示（加载 PPO 权威模型）
python scripts/demo/demo_multi_machine.py --ppo-model deliverable_models/ppo_best_model_10dim.zip
```

## 注意事项

- 评估脚本（`strategy_comparison.py`、`stress_test.py`、`demo_multi_machine.py`）的模型路径常量已指向本目录；本地开发若使用 `models/` 下的其他实验检查点，可临时改回，但**提交前请确保 `deliverable_models/` 中的两个权威副本存在**。
- 若需重新训练并替换权威模型：训练完成后将 `best_model.zip` 复制至 `deliverable_models/` 并同步更新本文件与 `submission_manifest.yaml` 中的路径/体积。
- `models/`、`results/` 下的其余实验产物（消融、真机闭环、多 seed 原始数据等）仍按 `.gitignore` 规则不入库，仅作本地研发留档。
