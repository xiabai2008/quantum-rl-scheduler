# 变更日志 (CHANGELOG)

本文件记录项目的所有重要变更，按日期倒序排列。

## [2026-07-31] - v9.1 文档全面同步 + 自动文档同步检查机制

### 文档同步 (docs)
- **测试用例数统一为 3523**：AGENTS.md / README.md / docs/authoritative_numbers.md / docs/code_freeze.md / docs/requirements_traceability.md / 答辩PPT大纲.md 全部同步（旧值 2824+/3106/3359/3400+/3467 已清除）
- **AGENTS.md 进度条更新**：PR审查 80%→100%（7.31合并8个PR含原7个待修改PR的等效修复）；提交校验 90%→80%（4 ERROR + 1 WARN 待8/15前补齐）；移除"39个open issues""7个待修改PR"等过时表述
- **AGENTS.md 最后更新日期**：2026-07-29 → 2026-07-31
- **docs/项目记忆_给AI.md**：从 v7（2026-07-01）整体刷新至 v9.1（2026-07-31）
- **docs/defense_qa_handbook.md**：新增 Q52 编译层观测维度替换（PR #759）答辩话术，版本 v1.6→v1.7
- **MODELS.md**：新增 PR #759 compilation_env 观测维度11-13替换说明 + 模型兼容性提示（#772）
- **SECURITY.md**：更新 bandit 状态说明（剩1处 B614 Medium，#771跟踪）

### 新增 (feat)
- **scripts/ci/check_doc_sync.py**：CI 文档同步自动检查脚本，检测文档测试数 vs 实际 pytest 收集数、文档版本号 vs pyproject.toml、AGENTS.md open PR/issue vs gh CLI、最后更新日期 vs 当天
- **.github/workflows/ci.yml**：新增 doc-sync job，调用 check_doc_sync.py 作为 CI 门禁

## [2026-07-31] - 7.30-7.31 合并8个PR (#758-#765) 关闭所有历史issue

### 合并的 PR
- **#758** fix: tianyan_cqlib login_key私有化(#735)+wait_for_task错误计数修复(#719)
- **#760** feat(#677,#738,#737): PPO早停机制+cache锁粒度优化+TTL主动清理
- **#761** fix(#694,#734,#724): 线程泄漏+QUBO向量化+HEFT迭代化
- **#762** fix: models qubit_count一致性(#732)+RoundRobin指针修复(#693)+multi_obj封装(#696)
- **#763** fix: websocket异步I/O(#739)+skip DQN pretrain(#733)+CI benchmark阻断(#697)
- **#759** fix: 替换compilation_env观测维度11-13冗余反义特征为非冗余指标 (#656)
- **#764** perf: env.step热路径crosstalk_risk参数化(#746)+simulator状态访问器(#723)
- **#765** refactor: test_visualization拆分(#730)+benchmark阈值断言(#729)
- **#756** feat: 新增金融衍生品定价与药物分子模拟ROI分析场景 (Closes #741)
- **#754** feat: 统一AGENTS.md真机口径为N=10 v2权威+CI黑名单防旧数据回归 (Closes #691)
- **#755** fix(#715): _worker_loop区分致命异常(MemoryError/RuntimeError)与可恢复异常
- **#753** fix(#725): 统一动作常量来源，消除baselines/hybrid_scheduler重复定义
- **#752** fix(#695): ONNX导出后校验完整性+validate_export使用固定种子RNG
- **#751** fix(#679): metrics.py Gauge从未set赋值，simulator同步运行时Gauge

### 关闭的 issue
- #741, #691, #715, #725, #695, #679, #735, #719, #677, #738, #737, #694, #734, #724, #732, #693, #696, #739, #733, #697, #656, #746, #723, #730, #729 等历史 issue 全部关闭

### 当前状态
- 0 open PR / 40 open issue（#766+批次，v9.1定稿前跟踪项持续创建中）
- 3523 测试用例收集通过，ruff/mypy 0 errors，bandit 剩1处 B614 Medium（marl.py:722，#771跟踪）

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
