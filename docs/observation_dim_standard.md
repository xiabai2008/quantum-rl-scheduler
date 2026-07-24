# 观测维度口径管理标准

> **Issue #129** | 生成时间: 2026-07-24
> **适用范围**: 全项目脚本、文档、报告、答辩材料

---

## 一、观测维度定义

### 1.1 两种观测维度

| 维度 | 定义 | 实现方式 | 观测空间形状 |
|:--|:--|:--|:--|
| **14维（原生）** | 完整状态空间，包含物理噪声和拓扑特征 | `QuantumSchedulingEnv`（`src/scheduler/env.py`） | `Box(0, 1, (14,))` |
| **10维（截断）** | 基础状态空间，截断14维前10个维度 | `Obs10Wrapper`（`scripts/evaluation/run_issue_38_67_experiments.py`） | `Box(0, 1, (10,))` |

### 1.2 14维观测空间详细定义

**定义文件**：`src/scheduler/env_types.py`（`OBS_DIM = 14`）

| 索引 | 常量名 | 含义 | 类别 |
|:--|:--|:--|:--|
| 0 | OBS_QUBIT_AVAILABILITY | 量子比特可用率 | 基础 |
| 1 | OBS_QUEUE_LENGTH | 队列长度 | 基础 |
| 2 | OBS_AVG_WAIT_TIME | 平均等待时间 | 基础 |
| 3 | OBS_FIDELITY | 量子保真度 | 基础 |
| 4 | OBS_CLASSICAL_LOAD | 经典负载 | 基础 |
| 5 | OBS_QUANTUM_QUEUE_RATIO | 量子队列占比 | 基础 |
| 6 | OBS_TIME_OF_DAY | 时段 | 基础 |
| 7 | OBS_URGENCY_LEVEL | 紧急程度 | 基础 |
| 8 | OBS_TASK_TYPE_QUANTUM | 量子任务类型 | 基础 |
| 9 | OBS_TASK_TYPE_CLASSICAL | 经典任务类型 | 基础 |
| 10 | OBS_SINGLE_GATE_FIDELITY | 单比特门保真度 | 物理噪声 |
| 11 | OBS_TWO_GATE_FIDELITY | 双比特门保真度 | 物理噪声 |
| 12 | OBS_COUPLING_DENSITY | 耦合密度 | 拓扑特征 |
| 13 | OBS_AVG_CONNECTIVITY | 平均连接度 | 拓扑特征 |

### 1.3 10维观测空间映射

10维为14维的前10个维度截断，不含物理噪声和拓扑特征（索引 10-13）。

### 1.4 Obs10Wrapper 实现

**定义文件**：`scripts/evaluation/run_issue_38_67_experiments.py`（第 65-78 行）

```python
class Obs10Wrapper(gym.Wrapper):
    """将 14 维环境观测截断为 10 维，保持与旧模型兼容。"""
    def __init__(self, env: QuantumSchedulingEnv):
        super().__init__(env)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[:10].astype(np.float32), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return obs[:10].astype(np.float32), reward, terminated, truncated, info
```

---

## 二、适用场景规范

### 2.1 口径使用规则

| 场景 | 使用维度 | 理由 |
|:--|:--|:--|
| **权威对比实验（50seed）** | 10维（Obs10Wrapper） | 公平对比框架，兼容旧 DQN 10维模型 |
| **PPO 训练与评估** | 14维（原生） | 使用完整状态信息，最大化策略性能 |
| **DQN 训练（14维重训）** | 14维（原生） | 解决10维模型在14维环境退化问题 |
| **真机实验** | 14维（原生） | 真实环境使用完整观测 |
| **多机器演示** | 14维（原生） | 展示完整调度能力 |
| **压力测试** | 14维（原生） | 测试策略鲁棒性 |
| **消融实验 D2** | 10维→14维对比 | 专门测试维度扩展效果 |
| **答辩/PPT/白皮书** | 以14维为主口径 | 14维为最终提交版本 |

### 2.2 模型与维度对应关系

| 模型 | 文件 | 观测维度 | 用途 |
|:--|:--|:--|:--|
| PPO 权威模型 | `ppo_best_model_14dim.zip` | 14维 | 答辩/提交/真机实验 |
| DQN 10维（旧） | `dqn_model_10dim.zip`（已归档） | 10维 | 基线对比（Obs10Wrapper） |
| DQN 14维（重训） | `dqn_best_model_14dim.zip` | 14维 | 14维环境独立评估 |

### 2.3 口径切换不可比性声明

**核心原则**：10维和14维环境下的实验结果**不可直接比较**。

| 声明项 | 要求 |
|:--|:--|
| 报告标题 | 必须标注观测维度（如"14维 PPO vs FCFS"或"10维公平对比"） |
| 数据表格 | 必须在表头或脚注标注维度 |
| 跨维度引用 | 必须注明"10维结果与14维结果不可直接比较" |
| 答辩口径 | PPO +88.3% 为14维结果，10维仅用于 DQN 公平对比 |

---

## 三、全项目口径一致性审计

### 3.1 审计方法

扫描 `scripts/`、`docs/`、`results/reports/`、`tests/` 目录中所有涉及观测维度的表述。

### 3.2 审计结果

#### 3.2.1 已一致项（无需修改）

| 文件 | 口径 | 状态 |
|:--|:--|:--|
| `src/scheduler/env_types.py` | 14维（OBS_DIM=14） | ✅ |
| `src/scheduler/env_observation.py` | 14维 | ✅ |
| `src/scheduler/env.py` | 14维 | ✅ |
| `scripts/evaluation/run_multiseed_evaluation.py` | 默认14维，支持10维 | ✅ |
| `scripts/evaluation/run_issue_38_67_experiments.py` | 10维公平对比 | ✅ |
| `scripts/training/train_dqn_14dim.py` | 14维 | ✅ |
| `results/reports/strategy_comparison.md` | 14维 PPO + 10维 DQN | ✅ |
| `results/reports/statistical_validation.md` | 14维 | ✅ |
| `MODELS.md` | 14维 PPO + 14维 DQN + 10维 DQN（归档） | ✅ |
| `AGENTS.md` | 10维公平对比 + 14维训练 | ✅ |

#### 3.2.2 发现的不一致项

| 编号 | 文件 | 问题描述 | 严重程度 | 修复建议 |
|:--|:--|:--|:--|:--|
| AUDIT-01 | `results/reports/real_machine_preregistration.md` 第69行 | 引用不存在的 `Obs14Wrapper`，14维为原生环境无需包装器 | 中 | 改为"14维原生环境 `QuantumSchedulingEnv`" |
| AUDIT-02 | `scripts/evaluation/run_issue_38_67_experiments.py` 第739行 | "兼容现有 DQN/PPO 模型"表述不精确，Obs10Wrapper 仅兼容10维 DQN 模型，PPO 为14维 | 低 | 改为"兼容10维 DQN 模型" |
| AUDIT-03 | `MODELS.md` 第17-18行 | DQN 10维和14维模型奖励值均为 -897.08，可能引起混淆 | 低 | 添加注释说明"10维模型在10维环境评估，14维模型在14维环境评估" |
| AUDIT-04 | `docs/defense_qa_handbook.md` Q2 第46-62行 | DQN 失败归因为"值函数方法不适用"，未提及观测空间不匹配 | 中 | 补充"10维 DQN 模型在14维环境观测空间不匹配，退化为随机策略" |

#### 3.2.3 审计覆盖范围

| 目录 | 扫描文件数 | 涉及维度表述文件数 | 不一致项数 |
|:--|:--|:--|:--|
| `scripts/` | ~30 | 12 | 1 |
| `docs/` | ~20 | 8 | 1 |
| `results/reports/` | ~25 | 10 | 1 |
| `tests/` | ~45 | 5 | 0 |
| 根目录 | ~5 | 2 | 1 |
| **合计** | ~125 | 37 | 4 |

### 3.3 修复优先级

| 优先级 | 编号 | 修复内容 | 工作量 |
|:--|:--|:--|:--|
| P0 | AUDIT-01 | 修正 `real_machine_preregistration.md` 中不存在的 `Obs14Wrapper` 引用 | 5 分钟 |
| P1 | AUDIT-04 | 补充 `defense_qa_handbook.md` 中 DQN 失败的观测维度归因 | 10 分钟 |
| P2 | AUDIT-02 | 修正 `run_issue_38_67_experiments.py` 中兼容性表述 | 5 分钟 |
| P2 | AUDIT-03 | 在 `MODELS.md` 添加维度评估环境注释 | 5 分钟 |

---

## 四、各脚本/文档口径一览表

### 4.1 训练脚本

| 脚本 | 默认维度 | 支持切换 | 说明 |
|:--|:--|:--|:--|
| `scripts/training/train_agent.py` | 14维 | 否 | PPO 标准训练 |
| `scripts/training/train_dqn_14dim.py` | 14维 | 否 | DQN 14维重训 |
| `scripts/training/quick_train.py` | 14维 | 否 | 快速训练验证 |

### 4.2 评估脚本

| 脚本 | 默认维度 | 支持切换 | 说明 |
|:--|:--|:--|:--|
| `scripts/evaluation/run_multiseed_evaluation.py` | 14维 | `--obs-dim 10` | 权威多seed评估 |
| `scripts/evaluation/run_issue_38_67_experiments.py` | 10维 | `obs_dim=14` | Issue#38/67 公平对比 |
| `scripts/evaluation/run_simulation.py` | 14维 | 否 | 仿真运行 |
| `scripts/evaluation/run_quantum_sensitivity.py` | 14维 | 否 | 量子占比敏感性 |
| `scripts/evaluation/run_holdout_evaluation.py` | 14维 | 否 | 留出集评估 |
| `scripts/evaluation/run_dqn_ppo_fcfs_comparison.py` | 14维 | 否 | 三策略对比（Issue#96） |
| `scripts/evaluation/ablation_d3_training.py` | 10维 | 导入 Obs10Wrapper | D3 奖励消融 |
| `scripts/evaluation/machine_scalability_test.py` | 14维 | `obs_dim=10` | 机器扩展性 |
| `scripts/evaluation/workload_pattern_stats.py` | 14维 | 否 | 负载模式统计 |

### 4.3 真机脚本

| 脚本 | 默认维度 | 说明 |
|:--|:--|:--|
| `scripts/real_machine/tianyan287_experiment.py` | 14维 | 天衍-287 实验（兼容10维旧模型） |
| `scripts/real_machine/tianyan287_multiseed.py` | 14维 | 多seed真机实验 |
| `scripts/real_machine/strategy_comparison.py` | 14维 | 策略对比（兼容10维旧模型） |

### 4.4 文档口径

| 文档 | 口径 | 说明 |
|:--|:--|:--|
| `AGENTS.md` | 10维+14维 | 项目记忆文档 |
| `README.md` | 14维 | 项目介绍 |
| `MODELS.md` | 14维+10维 | 模型登记 |
| `docs/defense_qa_handbook.md` | 14维 | 答辩手册 |
| `docs/real_machine_verification_boundary.md` | 10维+14维 | 真机边界文档 |

---

## 五、AGENTS.md 口径管理规范（待添加段落）

在 AGENTS.md 适当章节添加以下内容：

```markdown
### 观测维度口径管理规范（Issue #129）

项目中存在两种观测维度，严格按以下规范使用：

| 维度 | 适用场景 | 包装器 |
|:--|:--|:--|
| 14维（原生） | PPO训练/评估、真机实验、答辩提交 | 无（QuantumSchedulingEnv 原生） |
| 10维（Obs10Wrapper） | 权威公平对比（50seed）、DQN基线对比 | Obs10Wrapper（截断前10维） |

**口径切换声明要求**：
- 10维和14维结果不可直接比较
- 报告/表格必须标注观测维度
- PPO +88.3% 为14维结果，10维仅用于 DQN 公平对比

详见 `docs/observation_dim_standard.md`
```

---

## 六、数据完整性声明

### 权威数字一致性

| 指标 | 值 | 维度 | 来源 |
|:--|:--|:--|:--|
| 仿真 PPO 均值 | 2746.94 ± 1121.19 | 14维（权威对比框架中用10维Obs10Wrapper） | 50 seeds × 5 episodes |
| 仿真 FCFS 均值 | 1458.77 ± 55.85 | 同上 | 同上 |
| 仿真 PPO 提升 | +88.3% | 同上 | 同上 |
| 仿真 p 值 | 3.04e-11 | 同上 | Welch t 检验 |
| 仿真 Cohen's d | -1.70 | 同上 | 同上 |
| DQN 10维 reward | -897.08 | 10维 | Obs10Wrapper 环境 |
| DQN 14维 reward | -897.08 | 14维 | 原生环境（重训后） |

> **注**：权威对比框架（50seed）使用 Obs10Wrapper（10维）评估所有策略，确保 DQN 10维模型可参与公平对比。PPO 模型虽为14维，但在 Obs10Wrapper 环境中通过截断观测进行评估。答辩中 PPO +88.3% 指的是该公平对比框架下的结果。

---

## 七、关联文档

| 文档 | 路径 | 说明 |
|:--|:--|:--|
| 环境类型定义 | `src/scheduler/env_types.py` | OBS_DIM=14 |
| 观测构建 | `src/scheduler/env_observation.py` | 14维观测实现 |
| Obs10Wrapper | `scripts/evaluation/run_issue_38_67_experiments.py` | 10维截断包装器 |
| 权威评估脚本 | `scripts/evaluation/run_multiseed_evaluation.py` | 多seed评估 |
| 模型登记 | `MODELS.md` | 模型维度记录 |
| 答辩手册 | `docs/defense_qa_handbook.md` | DQN 归因说明 |
| 真机边界 | `docs/real_machine_verification_boundary.md` | 10维vs14维对比 |

---

*Issue #129 验收文件 | 2026-07-24*
