# 权威模型检查点归档（MODELS.md）

> 本文件说明本项目**可提交的训练好的模型检查点**，用于保证评审在克隆仓库后能复现论文中的实验结果（PPO vs FCFS +123.4% 等）。
> 最后更新：2026-07-31

## 为什么需要本目录

原 `models/` 与 `results/` 被 `.gitignore` 整体忽略，**不会被提交到仓库**。若直接克隆，评估脚本将因加载不到模型而失败，实验数字无法复现。

为此，本目录 `deliverable_models/` 存放**官方锁定的权威模型副本**（体积小，约 0.5 MB），已通过 `.gitignore` 例外规则纳入版本控制，并登记在 `config/submission_manifest.yaml`（项 `MODEL_PPO`）。

## 权威模型清单

| 策略 | 提交路径（可复现） | 训练说明 | 体积 | 复现指标 |
|------|-------------------|----------|------|----------|
| **PPO（16维MLP）** | `deliverable_models/ppo_best_model_16dim.zip` | 16维原生环境，100K steps，标准PPO(MLP)，seed=42，use_lstm=False | ~282 KB | 调度层PPO（权威实验见statistics.yaml） |
| **PPO编译优化Agent** | `deliverable_models/ppo_compilation_agent.zip` | 量子比特映射任务专用PPO Agent，4×4 2D网格训练200k steps | ~160 KB | 公平对比v2：深电路(14-16q, 20电路)SWAP减少+45.5%（方向性优势，样本量不足），全60电路p=5.95e-01不显著，中+深40电路子集p=2.56e-01不显著（Issue #451） |

## 训练配置（复现前提）

- **PPO 观测空间**：16 维（原生 `QuantumSchedulingEnv`，含队列长度、量子保真度、等待时间、拓扑连接度、串扰风险、到达率MA等）
- **DQN 观测空间**：10 维（`Obs10Wrapper` 截断，仅用于基线对比，不作提交口径）
- **耦合图拓扑**：4×4 2D网格（匹配天衍-287真机nearest-neighbor拓扑，非旧版线性链）
- **模型架构**：标准 PPO MLP（`use_lstm=False`），兼容 Stable-Baselines3 `PPO.load()` 接口
- **随机种子**：42
- **评测规模**：200 步/episode、泊松到达 λ=0.5
- **PPO 训练量**：100,000 timesteps（16维模型已收敛至最优策略）

## 模型兼容性说明

交付模型采用**标准 PPO (MLP)** 架构而非 RecurrentPPO (LSTM)，以确保：
1. 官方评估脚本 `scripts/evaluation/run_simulation.py` 使用 `PPO.load()` 可直接加载
2. 无需额外LSTM状态管理即可推理
3. 消融实验验证：MLP与LSTM在本任务上收敛到相同策略，MLP训练速度更快

## 复现命令

```bash
# 1) 官方仿真评估（默认加载16维模型）
python scripts/evaluation/run_simulation.py --episodes 5 --tasks-per-episode 100

# 2) 多seed评估 + 统计显著性（16维 PPO）
python scripts/evaluation/run_multiseed_evaluation.py --seeds 10 --episodes 5 \
    --ppo-model deliverable_models/ppo_best_model_16dim.zip

# 3) 电路编译公平对比（PPO vs SABRE，4×4 2D网格，同池配对）
python scripts/evaluation/compilation_fair_v2.py

# 4) 消融实验：MLP vs LSTM 对比
python scripts/evaluation/ablation_ppo_variants.py
```

## 最终权威指标（答辩统一口径）

> **统计口径权威源**: `config/statistics.yaml`（Issue #141 建立单一权威统计源）
> 所有文档中的统计数字必须可追溯至该源，由 `scripts/ci/check_stats_consistency.py` 验证一致性。

| 指标 | 数值 | 统计检验 |
|:--|:--|:--|
| PPO vs FCFS 提升 | **+123.4%** | Welch t, p=1.449e-66, Cohen's d=-2.1353（16维权威实验, N=250, config/statistics.yaml） |
| 编译层PPO vs SABRE | 公平对比v2（4×4 2D网格, 60电路同池配对, Issue #451） | 深电路(14-16q, 20电路)SWAP减少+45.5%（方向性优势，样本量不足），全60电路p=5.95e-01不显著，中+深40电路子集p=2.56e-01不显著 |
| 2D网格拓扑优势 | SABRE在2D网格上SWAP比线性链少~62% | 拓扑消融实验（ablation_compilation_env.py） |
| 真机验证 | 315次SDK调用100%成功率 | 可用性验证 |

> **注意**：+123.4% 权威数字基于16维交付模型 ppo_best_model_16dim.zip（v9.1+，OBS_DIM=16，50seed×5episodes=250次独立运行），已在 config/statistics.yaml 中锁定。

## 训练脚本索引

| 脚本 | 用途 | 模型类型 |
|------|------|----------|
| `scripts/training/train_16dim_ppo.py` | 快速训练16维PPO-MLP交付模型 | MLP |
| `scripts/training/train_agent.py` | 标准PPO训练（支持16维） | MLP/LSTM |
| `scripts/training/train_lstm_15m_earlystop.py` | 带早停的PPO-LSTM训练 | LSTM |
| `scripts/training/train_ablation_variant.py` | 消融实验变体训练 | MLP/LSTM |
| `scripts/training/train_compilation_agent_2dgrid.py` | 编译层PPO训练（4×4 2D网格） | MLP |

## 注意事项

- PPO 权威调度模型为**16维标准MLP**（`ppo_best_model_16dim.zip`），是v9.1交付版本
- 旧的10维/14维调度模型已清理，仅保留当前最优16维模型和编译优化Agent
- 编译层Agent（`ppo_compilation_agent.zip`）为独立模型，使用14维观测空间（量子编译环境专用）
- **PR #759 兼容性提示（Issue #772）**：2026-07-31 PR #759 将 `compilation_env.py` 观测维度 11-13 从冗余反义特征替换为非冗余指标（维度数仍为14，shape 不变），但 `ppo_compilation_agent.zip` 是在替换前训练的，行为上可能与新环境存在分布偏移。维度数兼容可正常加载，但若需严格复现公平对比v2结果，建议基于新观测定义重训编译层模型（由 #772 跟踪）
- 如需重新训练并替换权威模型：使用对应训练脚本训练后将模型复制至 `deliverable_models/` 并同步更新本文件
- `run_simulation.py` 默认加载 `deliverable_models/ppo_best_model_16dim.zip`
- `models/` 目录（.gitignore忽略）存放训练过程中的临时checkpoint，不作为交付物
