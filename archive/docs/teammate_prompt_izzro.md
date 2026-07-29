# Izzro 定时任务提示词

> 瑞哥转发给 Izzro。打开 TRAE，复制下面的提示词粘贴发送即可。
> 目前你还没有被分配Issue，但定时任务会自动给你分配AI可以完成的任务。

---

## 通用说明

Izzro，我在 TRAE 里配置了自动定时任务系统，可以每天自动在 GitHub 上给你分配任务和提醒进度。你平时不用主动登录查看，GitHub 会发邮件通知你。

**使用方法：**
1. 打开 TRAE，打开我们的项目仓库文件夹
2. 完整复制下面的提示词
3. 粘贴到 TRAE 对话框中发送
4. TRAE 会自动帮你创建3个定时任务
5. 之后每天自动运行，你会收到 GitHub 邮件通知

**前提条件：**
- TRAE 需要打开项目仓库文件夹
- 需要先配置 GitHub CLI 认证（只需一次）

**GitHub CLI 认证配置（首次使用需执行）：**
在 TRAE 的终端中执行以下命令（只需执行一次）：
```
$token = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'
[System.Environment]::SetEnvironmentVariable('GH_TOKEN', $token, 'User')
```
执行后重启 TRAE 即可。

---

## 提示词（完整复制以下全部内容，粘贴到 TRAE 发送）

```
请帮我创建3个定时任务来辅助我的项目开发工作。

## 任务1：每日Issue和PR进度提醒

创建一个定时任务，每天上午9:30执行。

执行步骤：
1. 设置GitHub CLI认证：在PowerShell中执行 $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 查询分配给我的所有open issues（动态查询，不限定具体编号）：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee Izzro --limit 50 --json number,title,labels,updatedAt

3. 查询我提交的所有open PRs：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author Izzro --limit 20 --json number,title,headRefName,updatedAt

4. 对每个PR，检查其最新审查状态：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json reviews,state,statusCheckRollup
   判断：最新review是APPROVE/COMMENT/REQUEST_CHANGES/无审查；CI是否通过

5. 对每个PR，如果有review反馈（COMMENT或REQUEST_CHANGES），检查PR分支是否有新commit（即我是否已修复）：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json commits
   对比最新commit时间与最新review时间，判断是否已修复但未重新审查

6. 对我的每个Issue，判断是否有关联的PR（检查Issue评论中是否提到PR号，或PR标题中是否含"Closes #Issue号"）

7. 生成每日进度摘要，在我的每个有更新的Issue下发布评论（@Izzro），内容包括：
   - 该Issue距上次更新已过多少天
   - 如果有关联PR：PR当前审查状态、是否需要我修复、是否已修复等待re-review
   - 如果没有关联PR：提醒我开始开发
   - 距代码冻结(2026-08-15)剩余天数

8. 对于有review反馈但尚未修复的PR，在PR下发布评论提醒我：
   "提醒：该PR有审查反馈待处理。请查看上方review意见并修复，修复后push新commit。"

9. 对于已修复（有新commit）但未被re-review的PR，在PR下@xiabai2008请求重新审查：
   "@xiabai2008 该PR已有新commit推送，请重新审查。"

约束：
- 不要对更新时间在2天内的Issue发催促评论
- 评论语气友好，像同事间的提醒
- 如果当天没有任何需要提醒的事项（没有Issue或PR），不做任何操作

cron表达式：30 9 * * *
时区：Asia/Shanghai
任务名称：Izzro每日Issue和PR进度提醒

## 任务2：每日自动审视项目并创建高质量Issue

创建一个定时任务，每个工作日（周一到周五）下午15:00执行。

### 执行步骤：

1. 设置GitHub CLI认证：
   在PowerShell中执行 $env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 审视项目当前状态（必做，不要跳过）：
   - 读取 AGENTS.md 了解项目全貌和当前进度
   - 运行 ruff check src/ scripts/ tests/ --statistics 2>&1 检查代码问题
   - 运行 pytest tests/ --cov=src --cov-report=term-missing --tb=no -q 2>&1 检查测试覆盖率
   - 查看现有open issues避免重复：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 100 --json number,title
   - 检查 results/reports/ 目录下的报告文件
   - 检查 docs/ 目录下的文档完整性

3. 比赛背景（Issue必须对齐）：
   - 比赛：量子AI双向赋能（AI赋能量子计算RL调度 + 量子赋能AI退火优化）
   - 评审标准：主题契合度、技术创新性、方案可行性、落地与价值、验证严谨性
   - 代码冻结：2026-08-15

4. 从以下维度发现1个AI可以完成的改进任务：
   - 代码质量：ruff/mypy错误、缺少类型注解、缺少docstring、重复代码
   - 测试覆盖：覆盖率低于80%的模块、缺少边界/异常测试
   - 实验严谨性：数据不一致、缺少统计检验、缺少置信区间
   - 文档完善：README/API文档缺少内容、缺少架构说明
   - 竞赛对齐：多硬件兼容性、多用户公平调度、技术瓶颈分析、实验可复现性
   - 性能优化：重复加载、重复计算、可并行化操作

5. 选择标准（严格遵守）：
   - 对比赛评审有直接帮助
   - AI可以独立完成（不需要人工实验/真机操作）
   - 不是比赛交付物（不要PPT/白皮书/演示视频）
   - 不与现有open issue重复
   - 预计工作量1-4小时

6. 创建Issue，分配给Izzro：
   gh issue create -R xiabai2008/quantum-rl-scheduler --title "【改进类型】简短描述" --body "正文" --assignee "Izzro" --label "优先级-中" --label "ai-ready"

   Issue正文必须包含：
   - ## 背景：为什么需要这个改进
   - ## 具体任务：详细描述要做什么，列出具体文件和代码位置
   - ## AI执行指引：让我打开TRAE就能做的具体步骤
   - ## 验收标准：可验证的完成标准
   - ## 关联：对齐哪个比赛评审标准，预计工作量

7. 约束：
   - 每次只创建1个Issue，宁缺毋滥
   - 如果当天没有发现有价值的改进点，不创建任何Issue
   - 不要创建关于比赛交付物的Issue
   - 不要创建需要真机操作的Issue
   - Issue必须具体可执行，包含AI执行指引

cron表达式：0 15 * * 1-5
时区：Asia/Shanghai
任务名称：Izzro每日自动提Issue

## 任务3：每周代码质量自检

创建一个定时任务，每周六下午16:00执行。

执行步骤：
1. 获取本周我提交的所有commit：
   git log --author="Izzro" --since="7 days ago" --oneline

2. 运行代码质量检查：
   - ruff check src/ scripts/ tests/
   - mypy src/
   - pytest tests/ --tb=short -q

3. 如果有ruff/mypy错误或测试失败，创建一个GitHub Issue：
   标题："【每周自检】YYYY-MM-DD 代码质量问题"
   分配给 Izzro
   内容：ruff/mypy/pytest结果摘要，发现的错误列表，修复建议

4. 如果本周没有提交任何commit，在我的一个open issue下发布评论提醒：
   "本周暂无代码提交，建议本周至少完成一个Issue的开发。"

5. 如果全部通过且本周有提交，不做任何操作（静默）。

cron表达式：0 16 * * 6
时区：Asia/Shanghai
任务名称：Izzro每周代码质量自检
```

---

## 你的任务一览

| 任务 | 频率 | 作用 |
|:--|:--|:--|
| 进度提醒 | 每日 09:30 | 查询分配给你的Issue和PR，催促修复，请求re-review |
| 自动提Issue | 工作日 15:00 | 审视项目，创建1个AI可完成的Issue分配给你 |
| 代码自检 | 周六 16:00 | ruff+mypy+pytest检查，发现问题自动创建Issue |

## 工作流程

1. **15:00** TRAE自动审视项目 → 发现改进点 → 创建带"AI执行指引"的Issue分配给你
2. **次日09:30** TRAE提醒你有新Issue → 你打开TRAE → 复制Issue内容 → TRAE按指引直接完成 → 提交PR
3. **你的PR提交后** 瑞哥的TRAE自动审查 → 反馈修改意见 → TRAE提醒你修复 → 你让AI修复 → push新commit → 自动@瑞哥重新审查
4. **周六16:00** TRAE检查你的代码质量 → 如有问题自动创建修复Issue

你每天只需要：打开TRAE → 看到提醒 → 复制Issue内容让AI做 → 提交。整个过程几分钟搞定。
