# Issues 并行会话任务提示词

> **仓库**: xiabai2008/quantum-rl-scheduler
> **Open Issues 总数**: 22 个
> **会话数**: 5 个（可并行执行）
> **生成时间**: 2026-07-02

---

## 分组总览

| 会话 | 主题 | Issues | 优先级 | 涉及文件范围 | 与其他会话冲突 |
|:--:|:--|:--|:--:|:--|:--|
| **会话 1** | 参赛文档与流程 | #102 #104 #105 #107 #39 | 高 | docs/、results/reports/ | 无 ✅ |
| **会话 2** | 实验验证与演示 | #38 #67 #37 | 高 | scripts/、results/ | 无 ✅ |
| **会话 3** | 核心代码重构+真机+测试 | #64 #46 #23 | 高/中 | src/scheduler/、src/api/、tests/ | 与会话5在app.py有轻微交叉 ⚠️ |
| **会话 4** | CI/CD与工程化 | #110 #101 #111 #94 #93 | 高/中 | .github/、pyproject.toml、Dockerfile | 无 ✅ |
| **会话 5** | 功能增强+前端+监控 | #103 #98 #96 #97 #95 #22 | 高/中 | src/（新模块）、src/visualization/ | 与会话3在app.py有轻微交叉 ⚠️ |

### Milestone 紧急度排序

| Milestone | 截止日期 | Open Issues | 紧急度 |
|:--|:--|:--:|:--|
| M6: 代码质量与工程韧性 | 2026-07-31 | 2 | 🔴 最紧急 |
| M7: 安全与可观测性 | 2026-08-14 | 1 | 🟠 紧急 |
| M8: 实验验证与参赛冲刺 | 2026-08-31 | 9 | 🟡 重要 |
| M9: 功能增强与性能优化 | 2026-09-13 | 3 | 🟢 一般 |
| M10: 工程化与开发者体验 | 2026-09-13 | 7 | 🟢 一般 |

---

## 会话 1：参赛文档与流程

### 提示词

```
你是量子RL调度系统项目的文档工程师。仓库地址：https://github.com/xiabai2008/quantum-rl-scheduler，工作目录为项目根目录。

## 你的任务
完成以下 5 个 GitHub Issues（全部为文档撰写，不涉及代码修改）：

1. **#102 比赛需求追溯矩阵**（高优先级，M8）
   - 制作"发榜方需求 ↔ 代码实现"对照表
   - 遍历 src/ 所有模块，将每个功能点映射到比赛需求
   - 输出：docs/requirements_traceability.md

2. **#104 head_only 退火策略有效性验证报告**（高优先级，M8）
   - 这是关键科学性证明文档
   - 分析 src/quantum/annealing.py 中 head_only 参数的作用
   - 对比 head_only=True vs False 的退火效果
   - 引用 results/ablation_annealing_multiseed_*.json 数据
   - 输出：results/reports/head_only_validation.md

3. **#105 答辩模拟演练与评委 Q&A 预案库**（高优先级，M8）
   - 编写 30+ 预设问题和标准答案
   - 覆盖：技术原理、实验设计、创新点、工程实现、商业价值、团队分工
   - 参考答辩PPT大纲.md 和 results/reports/ 下的实验报告
   - 输出：docs/defense_qa_handbook.md

4. **#107 8/15 代码冻结流程与例外审批机制**（高优先级，M8）
   - 定义代码冻结日期（8/15）、冻结范围、例外审批流程
   - 包括：冻结检查清单、例外申请模板、审批角色定义
   - 输出：docs/code_freeze_policy.md

5. **#39 编写 API 接口文档**（中优先级，M10）
   - 为 src/api/ 下所有公开接口编写文档
   - 包括：tianyan_client.py、tianyan_cqlib.py、mock_client.py 的所有公开方法
   - 格式：接口签名、参数说明、返回值、异常、使用示例
   - 输出：docs/api_reference.md

## 工作流程
1. 先用 `gh issue view <编号> --repo xiabai2008/quantum-rl-scheduler` 阅读每个 issue 的完整描述
2. 阅读相关源码和已有文档
3. 逐个完成文档撰写
4. 每完成一个 issue，用 `gh issue close <编号> --repo xiabai2008/quantum-rl-scheduler --comment "已完成：<文件路径>"` 关闭

## 注意事项
- 所有文档使用中文
- 引用数据时标注来源文件路径
- 不修改 src/ 下任何代码文件
- 遵循项目 AGENTS.md 中的规范
```

---

## 会话 2：实验验证与演示

### 提示词

```
你是量子RL调度系统项目的实验工程师。仓库地址：https://github.com/xiabai2008/quantum-rl-scheduler，工作目录为项目根目录。

## 你的任务
完成以下 3 个 GitHub Issues（运行实验脚本，产出数据和报告）：

1. **#38 完善实验对比数据与可视化报告**（高优先级，M8）
   - 运行 8 策略对比仿真，确保数据完整
   - 命令：python scripts/evaluation/run_simulation.py --mock-mode --num-tasks 200 --episodes 50
   - 生成可视化图表（matplotlib）：8策略奖励柱状图、等待时间散点图、利用率对比图
   - 更新 results/reports/strategy_comparison.md（如已存在则补充完善）
   - 确保所有图表有中文标题和坐标轴标注（设置字体：Noto Sans CJK SC）

2. **#67 压力测试梯度报告**（中优先级，M8）
   - 运行任务规模梯度测试：100 / 500 / 1000 / 5000 / 10000 任务
   - 命令示例：
     python scripts/evaluation/run_simulation.py --tasks-per-episode 100 --episodes 50
     python scripts/evaluation/run_simulation.py --tasks-per-episode 500 --episodes 50
     python scripts/evaluation/run_simulation.py --tasks-per-episode 1000 --episodes 30
     python scripts/evaluation/run_simulation.py --tasks-per-episode 5000 --episodes 10
     python scripts/evaluation/run_simulation.py --tasks-per-episode 10000 --episodes 5
   - 记录各规模下的：平均奖励、完成率、等待时间、资源利用率
   - 绘制吞吐量曲线图（任务数 vs 每步奖励）
   - 输出：results/reports/stress_test_gradient.md + 对应 JSON 数据

3. **#37 制作系统演示视频分镜脚本**（高优先级，M8）
   - 参考 演示视频分镜脚本.md（已存在），完善为可执行版本
   - 补充实际录屏操作步骤（Web面板启动、任务提交、监控查看）
   - 撰写 AI 配音稿（可直接用于 Azure TTS 或讯飞配音）
   - 输出：docs/demo_video_script.md

## 工作流程
1. 先用 `gh issue view <编号>` 阅读每个 issue 的完整描述
2. 检查 scripts/ 目录下可用的脚本
3. 运行实验（注意：训练可能耗时较长，先小规模验证再扩大）
4. 生成图表和报告
5. 每完成一个 issue，用 `gh issue close <编号> --comment "已完成：<说明>"` 关闭

## 注意事项
- Python 环境：D:\tools\Python 3.12.9\python.exe
- 所有图表中文标注需设置字体：matplotlib.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
- 实验数据保存到 results/ 目录，文件名包含时间戳
- 不修改 src/ 下任何代码文件（只运行脚本和生成报告）
- 如脚本有 bug 导致无法运行，记录问题但不要修改 src/ 代码
```

---

## 会话 3：核心代码重构+真机+测试

### 提示词

```
你是量子RL调度系统项目的核心开发工程师。仓库地址：https://github.com/xiabai2008/quantum-rl-scheduler，工作目录为项目根目录。

## 你的任务
按以下顺序完成 3 个 GitHub Issues（涉及 src/ 核心代码修改，必须按顺序执行）：

### 第一步：#64 PPO 真机闭环对接（高优先级，M8）
- 将 cqlib 真机客户端注入 PPO 调度循环
- 修改文件：src/api/tianyan_cqlib.py、src/scheduler/agent.py、src/scheduler/env.py
- 实现真机任务提交 → 结果轮询 → 奖励反馈的完整闭环
- 确保降级机制：真机不可用时自动回退到 Mock
- 验证：运行 python scripts/demo_multi_machine.py --real --real-prob 0.05 --episodes 5

### 第二步：#46 拆分大模块（中优先级，M6）
- 在 #64 完成后进行（避免合并冲突）
- 拆分目标：
  - src/scheduler/agent.py（~750行）→ 拆分为 agent_base.py + ppo_agent.py + dqn_agent.py
  - src/scheduler/env.py（~1100行）→ 拆分为 env_core.py + env_tasks.py + env_machines.py
  - src/visualization/app.py（1164行）→ 拆分为 app_main.py + routes/ + websocket_handler.py
- 保持所有公开接口不变（__init__.py 重新导出）
- 拆分后运行全部测试确保无回归

### 第三步：#23 提升测试覆盖率到 80%+（中优先级，M10）
- 在 #46 拆分完成后进行（针对新模块结构编写测试）
- 当前覆盖率约 85%（根据最近 pytest --cov 结果），确认是否已达 80%
- 如已达标，补充边界用例和异常路径测试
- 重点覆盖：src/scheduler/agent.py（当前约 44%）、拆分后的新模块
- 运行：python -m pytest tests/ --cov=src --cov-report=html

## 工作流程
1. 先用 `gh issue view <编号>` 阅读每个 issue 的完整描述
2. 阅读现有代码结构，制定拆分计划
3. 按顺序执行（#64 → #46 → #23）
4. 每完成一个 issue，运行测试验证无回归
5. 用 `gh issue close <编号> --comment "已完成：<说明>"` 关闭

## 注意事项
- 代码规范：Black（line-length=100）、isort、ruff、mypy（disallow_untyped_defs）
- 注释使用中文，函数必须有 docstring
- 类型标注必须完整（mypy strict）
- ⚠️ 与会话 5 的潜在冲突：会话 5 可能修改 src/visualization/app.py 的前端部分
  - 你的 app.py 拆分应聚焦后端路由拆分，前端相关改动让会话 5 先完成
  - 如发现冲突，优先完成 #64 和 #46 的 scheduler 部分拆分
- 每次修改后运行：ruff check src/ && mypy src/ && pytest tests/ --cov=src
```

---

## 会话 4：CI/CD与工程化

### 提示词

```
你是量子RL调度系统项目的 DevOps 工程师。仓库地址：https://github.com/xiabai2008/quantum-rl-scheduler，工作目录为项目根目录。

## 你的任务
完成以下 5 个 GitHub Issues（主要修改配置文件和 CI 脚本，不涉及 src/ 代码逻辑）：

1. **#110 M5 最终提交物一键打包与版本校验脚本**（高优先级，M8）
   - 创建 config/submission_manifest.yaml（提交物清单）
   - 创建 scripts/ci/validate_submission.py（校验脚本）
   - 校验项：文件存在性、版本一致性、PDF页数、PPT页数、视频时长/分辨率、ZIP大小
   - 支持 --check 和 --pack 两种模式
   - 用 issue 描述中的 YAML 模板作为起点

2. **#101 Benchmark 结果跨版本追踪与性能回归检测**（M6）
   - 创建 scripts/ci/benchmark_tracker.py
   - 每次运行 benchmark 后将结果存入 results/benchmark_history.jsonl
   - 对比上一版本，如性能下降超过 10% 则输出警告
   - 在 CI 中添加 benchmark 作业（可选，受时间限制可标记为 manual）

3. **#111 CI 增加 Windows / macOS 测试矩阵**（M10）
   - 修改 .github/workflows/ci.yml
   - 在 test job 中添加 os 矩阵：ubuntu-latest, windows-latest, macos-latest
   - 注意 Windows 下的 multiprocessing 需要 if __name__ == "__main__" 保护
   - 路径处理统一使用 pathlib
   - 如某些测试在 Windows/macOS 失败，添加 skip 标记并记录原因

4. **#94 依赖版本锁定与定期更新策略**（M10）
   - 用 pip-compile 生成 requirements.lock（精确版本锁定）
   - 创建 .github/workflows/dependency-update.yml（每周自动检查更新）
   - 在 requirements.txt 中区分核心依赖和可选依赖
   - 文档：docs/dependency_management.md

5. **#93 DevContainer 添加 GPU 支持**（M10）
   - 修改 .devcontainer/Dockerfile.dev，添加 CUDA/cuDNN 支持
   - 修改 .devcontainer/devcontainer.json，添加 GPU 相关特性
   - 创建 .devcontainer/docker-compose.gpu.yml（GPU 版本）
   - 保持非 GPU 版本可用（向后兼容）

## 工作流程
1. 先用 `gh issue view <编号>` 阅读每个 issue 的完整描述
2. 按优先级执行：#110 → #101 → #111 → #94 → #93
3. 每完成一个 issue，验证配置文件语法正确
4. 用 `gh issue close <编号> --comment "已完成：<说明>"` 关闭

## 注意事项
- 不修改 src/ 下任何代码文件
- CI 配置修改后，如可能用 `act` 工具本地验证（可选）
- YAML 文件注意缩进（2空格）
- 脚本必须有 docstring 和类型标注
- 遵循项目 AGENTS.md 中的规范
```

---

## 会话 5：功能增强+前端+监控

### 提示词

```
你是量子RL调度系统项目的功能开发工程师。仓库地址：https://github.com/xiabai2008/quantum-rl-scheduler，工作目录为项目根目录。

## 你的任务
完成以下 6 个 GitHub Issues（新功能开发和前端改进）：

1. **#103 天衍云真机配额追踪与预警机制**（高优先级，M7）
   - 在 src/api/tianyan_client.py 中添加配额追踪功能
   - 记录每日真机任务提交数、剩余配额、预计耗尽时间
   - 配额低于阈值时发出告警（日志 + Web 面板通知）
   - 在 src/visualization/app.py 添加 /api/quota 端点
   - 输出：src/api/quota_tracker.py（新文件）

2. **#98 任务依赖图支持（DAG 调度）**（M9）
   - 在 src/scheduler/ 中新增 dag_scheduler.py
   - 支持定义任务间依赖关系（前置任务完成后才能执行后继）
   - 与现有 QuantumSchedulingEnv 集成
   - 拓扑排序 + 资源约束调度
   - 新增测试：tests/test_dag_scheduler.py

3. **#96 混合调度模式：规则引擎 + RL 兜底**（M9）
   - 在 src/scheduler/ 中新增 hybrid_scheduler.py
   - 规则引擎处理确定性场景（如：高优先级量子任务直接分配）
   - RL（PPO）处理不确定性场景（多资源竞争、负载均衡）
   - 可配置规则引擎与 RL 的切换阈值
   - 新增测试：tests/test_hybrid_scheduler.py

4. **#97 多租户资源配额隔离**（M10）
   - 在 src/scheduler/env.py 中添加租户维度
   - 每个租户有独立的量子/经典资源配额
   - 配置文件 config/tenants.yaml 定义租户配额
   - 确保调度策略不违反配额约束

5. **#95 前端工程化：Vue3 迁移 Vite**（M10）
   - 将 src/visualization/frontend/index.html 迁移为 Vite 项目
   - 创建 src/visualization/frontend/package.json、vite.config.ts
   - 拆分组件：TaskQueue.vue、ResourceDashboard.vue、DecisionLog.vue、MachineStatus.vue
   - 保持与后端 FastAPI 的 WebSocket/API 接口不变

6. **#22 完善 Web 监控面板交互功能**（M9）
   - 添加任务详情弹窗（点击任务查看完整信息）
   - 添加决策过程回放功能（时间轴滑动）
   - 添加资源利用率历史趋势图（Echarts 折线图）
   - 添加多机器对比视图
   - 修改 src/visualization/frontend/ 和 src/visualization/app.py 的相关路由

## 工作流程
1. 先用 `gh issue view <编号>` 阅读每个 issue 的完整描述
2. 按优先级执行：#103 → #98 → #96 → #97 → #95 → #22
3. 每完成一个 issue，运行相关测试
4. 用 `gh issue close <编号> --comment "已完成：<说明>"` 关闭

## 注意事项
- 代码规范：Black（line-length=100）、isort、ruff、mypy（disallow_untyped_defs）
- 注释使用中文，函数必须有 docstring
- 新增 Python 文件必须有对应测试
- ⚠️ 与会话 3 的潜在冲突：会话 3 正在拆分 src/visualization/app.py
  - 你修改 app.py 时只添加新路由（如 /api/quota），不改现有路由结构
  - 前端改动（#95 #22）集中在 src/visualization/frontend/ 目录，不冲突
  - 如发现 app.py 合并冲突，你的改动优先（新功能添加），会话 3 的拆分后续适配
- 每次修改后运行：ruff check src/ && mypy src/ && pytest tests/ --cov=src
```

---

## 执行建议

### 并行执行顺序

```
时间轴 ──────────────────────────────────────────────→

会话1 [文档] ████████████████████ 完成
会话2 [实验] ████████████████████████ 完成
会话3 [重构] ████████████████████████████████ 完成（耗时最长）
会话4 [CI/CD] ██████████████████ 完成
会话5 [功能] ████████████████████████████████████ 完成（耗时最长）
```

### 冲突协调方案

| 冲突点 | 涉及会话 | 解决方案 |
|:--|:--|:--|
| src/visualization/app.py | 会话3（拆分） vs 会话5（新增路由） | 会话5只添加新路由不改现有结构；会话3的 app.py 拆分最后做，适配会话5的新路由 |
| results/reports/*.md | 会话1（写报告） vs 会话2（写报告） | 会话1写 docs/ 和非实验类报告；会话2写 results/ 下的实验类报告；如文件重叠，会话2的数据优先 |

### 每个会话通用注意事项

1. **开始前**：`git pull origin main` 获取最新代码
2. **分支策略**：每个会话创建独立分支 `feature/session-N-<主题>`
3. **完成后**：推送分支 → 创建 PR → 在 PR 描述中列出关闭的 issue 编号
4. **Issue 关闭**：PR 合并后自动关闭 issue（使用 `closes #编号` 语法），或手动 `gh issue close`
5. **项目规范**：遵循 AGENTS.md 中的所有约定（Black line-length=100、中文注释、mypy strict 等）
