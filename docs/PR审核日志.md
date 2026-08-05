# PR 审核日志

> 按 PR 审核专家角色（Regression-Gatekeeper v2）维护：每次审核决策必须追加记录。
> 格式：PR 编号 / 决策 / 日期 / 关键发现 / 预判回归风险。

---

## 2026-08-02 批量审核（7 个 open PR 清零）

### #927 — ✅ 已合并（squash, ef86c38）
- **内容**：批量处理 #878/#887/#888/#889/#886（alerts 线程安全 / config fail-fast / training_logger / compilation_env 向量化+奖励参数化 / Docker 安全加固），18 文件 +715/-80
- **审核**：compilation_env.py `_get_obs` 向量化逐项验证数学等价（连通性对称双计、frag 上三角单计、avg_swap_dist fill_diagonal+min、free_n 矩阵乘 mask），观测 14 维不变、奖励系数默认值保持既有行为、旧反义维度保留 → **模型兼容无损**
- **CI**：修复 doc-sync 两处不一致（测试数 3612→3681、AGENTS.md 快照 0→8 open PR）后双绿
- **回归风险**：低。5 个 issue 已随合并关闭（附验证评论）

### #926 — ✅ 已合并（squash, abdf0b7）
- **内容**：pre_freeze_check.sh 从 black/isort 迁移至 ruff，新增 mypy/bandit/open PR 检查（Closes #925），1 文件 +80/-27
- **审核**：与 CI 工具栈（ruff/mypy/bandit）一致，bash -n 语法通过
- **回归风险**：低（冻结前检查脚本与流水线对齐）

### #921 — ✅ 已合并（本地 merge, 755d255）
- **内容**：+88.3% 权威残留清理统一 16 维口径（#766）+ 噪声 N=25（#831）+ **#873 HybridScheduler 线程安全+RL 熔断** + submission 打包修复（15GB→5.5MB），30 文件 +524/-119
- **审核**：hybrid_scheduler.py 锁设计审查（_increment_stats/_stats_lock 短临界区、RL 熔断阈值 10、半开探针、成功即恢复，无死锁）；test_hybrid_scheduler 41/46 用例通过；合并冲突（测试数 3681 vs 3656）统一为实测 **3689**（pytest --co 实测）
- **回归风险**：低。唯一小瑕疵：fallback 路径 `_degraded_decision_count += 1` 未加锁（仅影响探针间隔统计，非安全）

### #916 — ❌ 已拒绝（重复 PR）
- **原因**：14 个文件中 13 个与 #921 完全重叠（results/reports/*.md 的 +88.3% 替换），#921 覆盖面更全（含 docs 根 + submission 打包）

### #914 — ❌ 已拒绝（过时 + noise）
- **原因**：①核心诉求 #903 已核实为误报（8.1 四审全仓 grep）②数据过时——写"深电路 33% 无统计检验"，与 #559 深电路 N=80 显著（+38.5%, p=2.75e-02）矛盾，合并会回退已升级结论 ③夹带 noise（8.1审查报告.html、真机测试记录/ 个人目录入仓库）

### #922 / #923 / #924 — ✅ 已合并（dependabot CI action 升级）
- actions/cache 4→6（0b3ea59）、softprops/action-gh-release 2→3（d207ed2）、upload-artifact 4→7（本地 merge 0120a7a，token 无 workflow scope 用本地合并）
- **审核**：gh-release 参数 name/body_path v3 兼容；upload-artifact v4-v7 同新 artifact 后端，与 download-artifact@v4 配套；CI failure 仅为 doc-sync 快照过时（main 已修）
- **回归风险**：低（基础 CI 依赖升级，合并后 main doc-sync 验证通过）

### 批后状态
- open PR：8 → **0**
- open issue：14 → **4**（#766/#873/#903 关闭——均为假 Open，附验证评论；剩 #846 演示视频人工项）
- 测试数统一：3689（pytest --co 实测，5 个文档同步，doc-sync 11/11 全绿）

---

## 2026-08-05 — 8.5-v2 审查批次（13 份外部审查结果 → 数字诚信大修）

### 触发
外部 8.5-v2 审查（13 份报告）聚焦"数字诚信"：+123.4% 残留、利用率 +7.9% 假数据、N=4 中间品提交、奖励偏向（Reward Hopping）、噪声反馈恒等式。

### 处理（5 个 commit）
| Commit | 内容 | 关键风险 |
|:--|:--|:--|
| 32b71bd | **A 类数字诚信**：+123.4→+20.2 全仓清理 69 处（README/AGENTS/白皮书/QA/PPT 生成器/archive）；利用率 +7.9→-3.3（N=250 真实）；等待时间换 N=250（25.46/-14%）；删 N=4 中间报告 2 份；编译层 README 同步 N=80（+38.5% p=2.75e-02）；check_stats_consistency 加内部冲突检测 | CI Lint 失败（ruff format/E741）→ c6bb333 修复 |
| 5c32f8f | **B 类话术**：Q58/Q59（奖励函数诚实披露 + 噪声反馈证据边界）；30vs315 口径澄清；MAPPO 未收敛注；CI QUBO/Codecov 条件 3.11→3.12 | 同上 Lint 失败 |
| 33f7cca | **C 类真修**：C2 classical.queue 恒 0 修复（到达累加）+ C1 FCFS 命名语义注明 + train_noise_feedback_v2 --timesteps | 低 |
| 7601d49 | train_noise_feedback_v2 参数传递链修复 | 低 |
| c6bb333 | Lint 修复（ruff format 2 文件 + E741） | 无 |

### 预判回归风险
- **中**：数字口径全仓替换（+123.4→+20.2）——若 statistics.yaml/历史报告被外部引用旧口径会不一致；已在 statistics.yaml history 段保留旧值并加"已废弃"标注
- **中**：check_stats_consistency 新内部冲突检查——已测冲突样本 exit=1、诚实披露豁免、真实仓 0 冲突
- **低**：C2 改 env_dynamics（到达计数）——224 相关测试通过；C1 纯注释
- **待验证**：B2 噪声注入训练（train_noise_feedback_v2 后台跑，~90min）——若结果支持则"双向赋能"升级

### 批后状态
- CI：5c32f8f/32b71bd 因 Lint 失败 → c6bb333 修复推送，CI 重跑中
- 权威数字：+20.2%（vs 真实 FCFS，N=250）、利用率 -3.3%、等待 -14%、编译层 +38.5%（N=80）

### 追加（同日，C 类收尾）
- **C3**（保真度漂移方向）：模拟设计（衰减+恢复），有 noise_profile 时零漂移；判定不改（低价值、改动影响所有基准）
- **C4**（QEM 动作语义）：N_ACTIONS=4 但默认 env 无 QEM 分支，action=3 静默=经典执行 → env_types/env.py 显式语义注明（不改行为避免模型兼容风险）；QEM 为真机模块专用设计
- **残留清理**：Code_Wiki/annealing_significance-defense +123.4 清零（7 处）；AGENTS/Code_Wiki p 值 1.449e-66→7.56e-12（vs 真实 FCFS 口径，8 处）
