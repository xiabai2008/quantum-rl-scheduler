# 权威模型检查点归档（MODELS.md）

> 本文件说明项目**已提交的训练模型检查点**，用于保证在克隆仓库后能复现所有核心实验结果（PPO vs FCFS +20.2% 等）。
> Version: 9.1.0
> 最后更新：2026-07-31
> 8.13 注记：`ppo_best_model_16dim.zip` 与 `ppo_fairness17dim.zip` 已于 8.13 重存——学习率调度由 cloudpickle 函数（lambda）改为 SB3 原生常数调度（ConstantSchedule），消除跨 Python 版本加载 SIGSEGV（3.10/3.11/3.12 均已实测可加载）；模型权重哈希不变（逐层 state_dict 逐位一致），权威评估结果不受影响。

## 为什么需要本目录

原 `models/` 与 `results/` 被 `.gitignore` 整体忽略，**不会被提交到仓库**。若直接克隆，评估脚本将因加载不到模型而失败，实验数字无法复现。

为此，本目录 `deliverable_models/` 存放**官方锁定的权威模型副本**（体积小，约 0.5 MB），已通过 `.gitignore` 例外规则纳入版本控制，并登记在 `config/submission_manifest.yaml`（项 `MODEL_PPO`）。

## 权威模型清单

| 策略 | 提交路径（可复现） | 训练说明 | 体积 | 复现指标 |
|------|-------------------|----------|------|----------|
| **PPO（16维MLP）** | `deliverable_models/ppo_best_model_16dim.zip` | 16维原生环境，100K steps，标准PPO(MLP)，seed=42，use_lstm=False | ~282 KB | 调度层PPO（权威实验见statistics.yaml） |
| **PPO公平感知（17维）** | `deliverable_models/ppo_fairness17dim.zip` | 17维原生环境（include_fairness_obs=True + 租户偏斜 80/10/10），100K steps，标准PPO(MLP)，seed=42 | ~285 KB | 不公平负载消融：效率 +5.0%（Jain -0.012，权衡非双赢，8.3 审查修正）；公平观测维实时反映 Jain 完成率公平指数 |
| **MAPPO 多智能体** | `deliverable_models/mappo/mappo.pt` | 3 机协同 MAPPO（共享 Critic 投票仲裁），50K steps，seed=42 | ~323 KB | 权威口径（同训练量收敛严格对比，mappo_strict_strict_comparison）：5507.7 vs 独立PPO 4033.6（协同优势+36.5%，N=20，配对 Wilcoxon p=0.024）vs 同环境 FCFS 917.7（+500.1%）。⚠️ FCFS 923.9 为固定 hybrid 基线（非真实 FCFS），+500.1% 仅作同 wrapper 严格对比，不对外作为权威性能提升；旧 +84.6%（p=0.019）为未收敛(5000步)+训练量不均等的探索值，已废弃。3 独立 PPO（independent_ppo_m0/1/2.zip，各 16K 步单机训练）随附 |
| **PPO编译优化Agent** | `deliverable_models/ppo_compilation_agent.zip` | 量子比特映射任务专用PPO Agent，4×4 2D网格训练200k steps | ~160 KB | 公平对比v2：全60电路+16.5%（p=8.40e-01不显著）；**深电路N=200规模化扩充（8.14, Issue #559）：+52.1%（Wilcoxon p=2.30e-10, 配对t p=6.26e-13, d_z=0.54, 95%CI[+43.9%,+72.1%]），规模化效应：sabre>=10子集+67.5%（p=1.21e-21）——深电路优于SABRE（原N=80 +38.5%为事后方向性已被N=200取代；全60电路不显著，浅/中电路无优势，诚实披露）** |

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
| PPO vs FCFS 提升 | **+20.2%** | Welch t, p=7.56e-12, rank-biserial=-0.3642（中效应）（16维权威实验, N=250, config/statistics.yaml） |
| 编译层PPO vs SABRE | 公平对比v2（4×4 2D网格, 60电路同池配对, Issue #451）+ 深电路N=200规模化扩充（8.14, Issue #559） | **深电路(14-16q)N=200：+52.1%（p=2.30e-10高度显著, d_z=0.54, CI[+43.9%,+72.1%]）**；规模化效应sabre>=10：+67.5%（p=1.21e-21）；全60电路p=8.40e-01不显著（简单/混合电路无优势，诚实披露；原N=80 +38.5%为事后方向性已被取代） |
| 2D网格拓扑优势 | SABRE在2D网格上SWAP比线性链少~62% | 拓扑消融实验（ablation_compilation_env.py） |
| 真机验证 | 315次SDK调用100%成功率 | 可用性验证 |

> **注意**：+20.2% 权威数字基于16维交付模型 ppo_best_model_16dim.zip（v9.1+，OBS_DIM=16，50seed×5episodes=250次独立运行），已在 config/statistics.yaml 中锁定。

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
- 如需重新训练并替换权威模型：使用对应训练脚本训练后将模型复制至 `deliverable_models/` 并同步更新本文件
- `run_simulation.py` 默认加载 `deliverable_models/ppo_best_model_16dim.zip`
- `models/` 目录（.gitignore忽略）存放训练过程中的临时checkpoint，不作为交付物

## 噪声反馈实验模型（8.11 补入库）

- models/noise_feedback_v2/：**48 对 (standard, noise) 模型**（交叉评估复用，96 个 zip）
- 用途：
oise_robustness_cross_eval（48×6 格子交叉评估）+ 
oise_training_stability（稳健性分析）
- 训练：scripts/training/train_noise_feedback_v2.py（500K 步，真实噪声分布注入）
- 8.11 审查补：此前 models/ 被 gitignore 未入库，评审无法复现交叉评估 → 已补入 48 对代表模型
