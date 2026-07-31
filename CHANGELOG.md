# 变更日志 (CHANGELOG)

本文件记录项目的所有重要变更，按日期倒序排列。

## [2026-07-31] - v9.1.1 文档一致性修复

### 修复
- **#783**: step() crosstalk_penalty 双重扣减修复（_compute_execution_reward 内部已扣减，外部重复扣减导致4x惩罚）
- **#787**: MAPPO action_dim 默认值 3→4 修正（QuantumSchedulingEnv 定义4个动作）
- **#788**: compilation_fair_v2 报告结论文本方向反转（3处 较低→较高）
- **#790/#793/#796**: PPO 早停 callback 修复 + nosec 标注 + include_fairness_obs 默认值回退
- **#797**: RoundRobin 指针修正（非 last_idx 弹出时指针偏移）
- **#798**: 文档测试用例数 2824+→3500+（实测 3508）
- **#800**: Code_Wiki.md 观测维度 14→16、测试文件数 64→111
- **#803**: strategy_comparison_report_v4 添加 DQN 已删除声明
## [2026-07-31] - v9.1.1 文档一致性修复

### 修复
- **#783**: step() crosstalk_penalty 双重扣减修复（_compute_execution_reward 内部已扣减，外部重复扣减导致4x惩罚）
- **#787**: MAPPO action_dim 默认值 3→4 修正（QuantumSchedulingEnv 定义4个动作）
- **#788**: compilation_fair_v2 报告结论文本方向反转（3处 较低→较高）
- **#790/#793/#796**: PPO 早停 callback 修复 + nosec 标注 + include_fairness_obs 默认值回退
- **#797**: RoundRobin 指针修正（非 last_idx 弹出时指针偏移）
- **#798**: 文档测试用例数 2824+→3500+（实测 3508）
- **#800**: Code_Wiki.md 观测维度 14→16、测试文件数 64→111
- **#803**: strategy_comparison_report_v4 添加 DQN 已删除声明
## [2026-07-29] - v9.1.0 权威数字更新 + 评审报告P0/P1修复

### v9.1 关键变更总览（相对于v8.x）
- **16维观测空间**：新增串扰风险（OBS_CROSSTALK_RISK）、到达率滑动平均（OBS_ARRIVAL_RATE_MA）两维，OBS_DIM=14→16
- **第17维可选公平性指数观测**：`include_fairness_obs`开关控制Jain公平性指数是否纳入状态（默认关闭保持向后兼容）
- **新增circuit_templates.py**：Bell/GHZ/VQE4/QAOA5标准量子电路模板
- **新增noise_extractor.py**：NoiseModelExtractor真机噪声画像提取与注入
- **噪声感知奖励整形**：低保真度惩罚/高保真度加成机制（NOISE_AWARD_*）
- **SHAP可解释性集成**：PPOExplainer决策特征贡献度分析
- **LearnableMachineScorer可学习路由**：数据驱动的机器评分与路由
- **QAOAScheduler基线**：QAOA专用调度器基线策略
- **非阻塞轮转轮询**：真机结果轮询改为非阻塞异步，避免step()阻塞
- **观测缓存消除重复计算**：ObservationCache复用帧间不变特征
- **Prometheus /metrics端点**：7个核心指标暴露（任务队列/等待时间/利用率/API延迟/失败率/退火耗时）
- **DAG QUBO numpy向量化**：退火模块QUBO矩阵构建numpy加速
- **AnnealingConfig修复**：Metropolis接受阈值bug修复
- **MARL evaluate()修复**：训练-评估一致性修复，多智能体评估结果可复现
- **Property-based测试 + 性能基准测试**：Hypothesis策略测试+ASV基准
- **27个issues全部关闭**：所有P0/P1/P2 issues已处理
- **测试3359通过，ruff/mypy 0 errors**

### 变更 (v9.0.0 → 9.1.0)
- **权威数字更新**：基于16维交付模型 `ppo_best_model_16dim.zip` 重跑 N=250 多seed评估，PPO vs FCFS 提升 +88.3% → **+123.4%**（Welch t p=1.449e-66，Cohen's d=-2.14，CI[+113.3%, +133.5%]）；旧14维数字移入 `config/statistics.yaml` deprecated 段留痕
- **量子→AI 统计证据**：新增 N=25 配对检验（Wilcoxon signed-rank p=2.98e-08，d_z=7.71，事后功效 1.0），真机噪声分布对PPO策略奖励显著影响，噪声感知闭环统计成立（`results/reports/quantum_noise_paired_canonical.md`），取代旧10seeds探索性实验

### 修复 (fix，对应外部评审报告)
- **P0**：`pre_freeze_check.sh` 模型清单改为现存交付模型；`compilation_fair_v2.py` Windows GBK 编码崩溃（stdout 强制 UTF-8）
- **P1/P2**：43%/v8.0 旧数字清除；编译层观测维度文档统一为14维（与 `compilation_env.py` shape=(14,) 一致）；`export.py` 默认输入形状 14→16 维；`statistical_validation.md` 环境描述修正并标注 DQN 行为 Random 占位；失效脚本加 DELETED 注释+优雅退出

### 工程 (chore)
- CI 新增 Linux 冒烟测试矩阵（#618）；真机密钥+CI验证、teammate 临时文档归档 `archive/`（#617）
- 版本号 `pyproject.toml` 9.0.0 → 9.1.0

## [2026-07-29] - 公平性特性 + PR审查 + 文档更新

### 新增 (feat)
- **#587 公平性奖励**：新增 `compute_fairness_penalty` 函数，当租户等待时间偏离均值超过阈值(0.3)时施加惩罚，惩罚因子=2.0
- **#588 公平性观测**：新增 `include_fairness_obs` 开关，可选第17维 Jain公平性指数观测（默认OBS_DIM=16不变，保持向后兼容）
- **#585 观测维度消融**：新增 `observation_dim` 参数，支持截断观测空间用于D2消融实验
- **#594 编译环境可配置**：支持自定义物理比特数和耦合图，天衍-287预设10x11网格拓扑(PR #616)
- **#576 真机配置统一**：RealMachineConfig 统一管理，真机提交概率默认值0.0→0.15(PR #601)
- **#513 /metrics端点安全**：严格API密钥认证，不豁免GET请求(PR #609)
- **#586 维度注释修正**：env.py/simulator.py/compilation_env.py中14维→16维注释修正(PR #609)
- **#593 真机测试标记**：8个真机测试文件添加@pytest.mark.real_machine(PR #609)

### 修复 (fix)
- **#592 tomllib兼容**：Python 3.10添加tomli fallback(PR #611)
- **#589 FutureWarning**：退火模块警告优化+ppo_agent日志修复(PR #610)
- **#590 退火开关**：dag_scheduler退火关闭时强制回退经典调度(PR #612)
- **统计口径一致性**：pr_patrol文档Cohen's d=5.64→5.33

### PR审查结果
- 已合并(6)：#611, #610, #612, #609, #601, #616
- 已关闭(7)：#571, #608, #607, #606, #551, #557, #558（main分支已有等效或更优实现）
- 需修改(7)：#614(巨型PR需拆分), #615(OBS_DIM破坏性变更), #613(回退#522优化), #604(删除已上线功能), #603(删除statevector), #602(penalty未使用), #600(QAOA物理不正确)

## [2026-07-28] - P0/P1修复 + 消融实验 + 交付模型

### 新增 (feat)
- 16维PPO-MLP交付模型（100K步训练，收敛至最优策略）
- 六维度消融实验全量完成（D1-D5+架构消融MLP/LSTM）
- 4×4 2D网格耦合图（匹配天衍-287拓扑，SWAP减少62%）
- Dynamic QEM（误差缓释动作）
- 串扰感知空间并发

### 修复 (fix)
- P0算法层Bug：耦合图拓扑(#404)、保真度模拟(#405)、SWAP距离(#406)、退火回调(#410)
- 环境终止语义Bug：terminated/truncated语义反转修复
- completion_rate缺失修复
