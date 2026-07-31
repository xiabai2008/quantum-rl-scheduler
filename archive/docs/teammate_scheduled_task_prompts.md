> ⚠️ **已废弃文档** — 本文档中的 +88.3%/76.4% 等数字为旧版 14 维模型数据，
> 已被 v9.1 16 维模型的 +123.4% 取代。详见 `config/statistics.yaml`。
> 保留本文档仅用于历史追溯，不作为数据基准。

# 队友定时任务提示词（优化版）

> 瑞哥发给各队友。每人找到自己的部分，复制代码块粘贴到 TRAE 即可。
> 每人3个任务：每日进度提醒 + 每周自动提Issue + 每周专项检查。
> 6人扫描时间完全错开，相邻两次间隔至少7小时。

---

## 通用说明（发到团队群）

各位，我在 TRAE 里配置了自动定时任务系统。每天/每周会自动在 GitHub 上提醒你的任务进度，你不用主动登录查看，GitHub 会发邮件通知你。

**使用方法：**
1. 打开 TRAE，打开项目仓库文件夹
2. 找到下面属于你的提示词，完整复制代码块
3. 粘贴到 TRAE 对话框中发送
4. TRAE 会自动帮你创建3个定时任务
5. 之后自动运行，你会收到 GitHub 邮件通知

**前提条件（只需做一次）：**
- TRAE 需要打开项目仓库文件夹，且工作时间内保持 TRAE 后台运行（定时任务需要 TRAE 在线才能执行）
- 配置 GitHub CLI 认证：在 TRAE 终端中执行：
```
$token = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'
[System.Environment]::SetEnvironmentVariable('GH_TOKEN', $token, 'User')
```
- 执行后重启 TRAE

**注意事项：**
- 如果 TRAE 没有打开，定时任务不会执行，所以请工作时间内保持 TRAE 运行
- 如果 GitHub 邮件通知突然停止，可能是 Token 过期，在 TRAE 终端重新执行上面的认证命令
- 在 TRAE 中说"列出我的定时任务"可以管理（暂停/修改/删除）

---

## DUMNOX 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建3个定时任务来辅助我的项目开发工作。

=========================================
任务1：每日Issue和PR进度提醒
=========================================

每天上午10:30执行。

执行步骤：
1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 查询分配给我的所有open issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee DUMNOX --limit 50 --json number,title,labels,updatedAt

3. 查询我提交的所有open PRs：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author DUMNOX --limit 20 --json number,title,headRefName,updatedAt

4. 对每个PR检查最新审查状态：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json reviews,state,statusCheckRollup

5. 对有review反馈的PR，检查是否有新commit（是否已修复）：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json commits
   对比最新commit时间与最新review时间

6. 对我的每个Issue，判断是否有关联PR（检查Issue评论或PR标题中是否含"Closes #Issue号"）

7. 在需要提醒的Issue下发布评论（@DUMNOX），内容包括：
   - 距上次更新天数
   - 有关联PR：PR审查状态、是否需要修复
   - 无关联PR：提醒开始开发
   - 距代码冻结(2026-08-15)剩余天数

8. 有review反馈但未修复的PR：在PR下评论提醒我修复
9. 已修复(有新commit)但未被re-review的PR：在PR下@xiabai2008请求重新审查

约束：不对2天内更新过的Issue催促；无事项则不操作。

cron表达式：30 10 * * *
时区：Asia/Shanghai
任务名称：DUMNOX每日进度提醒

=========================================
任务2：每周自动提Issue（AI可执行任务）
=========================================

每周一和周四上午09:00执行（与heka-ky的16:00错开7小时）。

1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 审视项目（必做）：
   - 读取 AGENTS.md 了解项目全貌
   - 运行 ruff check src/ scripts/ tests/ --statistics 2>&1
   - 运行 pytest tests/ --cov=src --cov-report=term-missing --tb=no -q 2>&1
   - 查看现有open issues避免重复：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 100 --json number,title
   - 检查 docs/ 目录文档完整性

3. 比赛背景：量子AI双向赋能（RL调度+量子退火），评审标准：主题契合度/技术创新性/方案可行性/落地价值/验证严谨性，代码冻结2026-08-15

4. 从以下维度发现1个AI可完成的改进任务：
   - 代码质量：ruff/mypy错误、缺少类型注解/docstring、重复代码
   - 测试覆盖：覆盖率低于80%的模块、缺少边界/异常测试
   - 实验严谨性：数据不一致、缺少统计检验/置信区间
   - 文档完善：README/API/架构说明缺失
   - 竞赛对齐：多硬件兼容、多用户公平调度、瓶颈分析、实验可复现
   - 性能优化：重复加载/计算、可并行化操作

5. 严格约束：
   - 只创建1个Issue，宁缺毋滥，无改进点则不创建
   - 不创建比赛交付物Issue（PPT/白皮书/演示视频）
   - 不创建需要真机操作的Issue
   - 不与现有open issue重复
   - 不连续两次创建相同改进类型的Issue（如上次是"测试覆盖"，这次不要也是"测试覆盖"）
   - 先检查我名下已有多少个ai-ready标签的open issue，如果已有3个以上则不创建新Issue（先完成现有的）
   - 预计工作量1-4小时

6. 创建Issue：
   gh issue create -R xiabai2008/quantum-rl-scheduler --title "【改进类型】简短描述" --body "正文" --assignee "DUMNOX" --label "优先级-中" --label "ai-ready"

   Issue正文必须包含：## 背景、## 具体任务（列出文件和代码位置）、## AI执行指引（打开TRAE就能做的步骤）、## 验收标准、## 关联（对齐哪个评审标准，预计工作量）

cron表达式：0 9 * * 1,4
时区：Asia/Shanghai
任务名称：DUMNOX每周自动提Issue

=========================================
任务3：每周代码质量自检
=========================================

每周日下午15:00执行。

1. 获取本周我的commit：git log --author="DUMNOX" --since="7 days ago" --oneline

2. 验证GH_TOKEN是否有效：gh auth status 2>&1，如果失败则在Issue #58下评论提醒我重新配置Token

3. 运行代码质量检查：
   - ruff check src/ scripts/ tests/
   - mypy src/
   - pytest tests/ --tb=short -q

4. 如果有错误或测试失败，创建Issue：
   标题："【每周自检】YYYY-MM-DD 代码质量问题"
   分配给 DUMNOX
   内容：错误列表和修复建议

5. 如果本周无commit，在Issue #58下评论："本周暂无代码提交，建议本周至少完成一个Issue。"

6. 全部通过且有提交则不操作（静默）。

cron表达式：0 15 * * 0
时区：Asia/Shanghai
任务名称：DUMNOX每周代码自检
```

---

## heka-ky 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建3个定时任务来辅助我的项目开发工作。

=========================================
任务1：每日Issue和PR进度提醒
=========================================

每天上午9:30执行。

执行步骤：
1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 查询分配给我的所有open issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee heka-ky --limit 50 --json number,title,labels,updatedAt

3. 查询我提交的所有open PRs：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author heka-ky --limit 20 --json number,title,headRefName,updatedAt

4. 对每个PR检查最新审查状态：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json reviews,state,statusCheckRollup

5. 对有review反馈的PR，检查是否有新commit（是否已修复）：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json commits

6. 对我的每个Issue，判断是否有关联PR

7. 在需要提醒的Issue下发布评论（@heka-ky），内容包括：
   - 距上次更新天数
   - 有关联PR：PR审查状态、是否需要修复
   - 无关联PR：提醒开始开发
   - 距代码冻结(2026-08-15)剩余天数
   - 按优先级标签排序建议优先处理哪个

8. 有review反馈但未修复的PR：在PR下评论提醒我修复
9. 已修复但未被re-review的PR：在PR下@xiabai2008请求重新审查

约束：不对2天内更新过的Issue催促；无事项则不操作。

cron表达式：30 9 * * *
时区：Asia/Shanghai
任务名称：heka-ky每日进度提醒

=========================================
任务2：每周自动提Issue（AI可执行任务）
=========================================

每周一和周四下午16:00执行（与DUMNOX的09:00错开7小时）。

1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 审视项目（必做）：
   - 读取 AGENTS.md
   - 运行 ruff check src/ scripts/ tests/ --statistics 2>&1
   - 运行 pytest tests/ --cov=src --cov-report=term-missing --tb=no -q 2>&1
   - 查看现有open issues避免重复：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 100 --json number,title
   - 检查 docs/ 目录文档完整性
   - 检查最近7小时创建的Issue（避免与DUMNOX重复）：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 10 --json number,title,createdAt

3. 比赛背景：量子AI双向赋能（RL调度+量子退火），评审标准：主题契合度/技术创新性/方案可行性/落地价值/验证严谨性，代码冻结2026-08-15

4. 从以下维度发现1个AI可完成的改进任务：
   - 代码质量、测试覆盖、实验严谨性、文档完善、竞赛对齐、性能优化

5. 严格约束：
   - 只创建1个Issue，宁缺毋滥，无改进点则不创建
   - 不创建比赛交付物/真机操作Issue
   - 不与现有open issue重复，不与最近7小时创建的Issue重复
   - 不连续两次创建相同改进类型
   - 先检查我名下ai-ready标签的open issue数，超过3个则不创建
   - 预计工作量1-4小时

6. 创建Issue：
   gh issue create -R xiabai2008/quantum-rl-scheduler --title "【改进类型】简短描述" --body "正文" --assignee "heka-ky" --label "优先级-中" --label "ai-ready"

   Issue正文必须包含：## 背景、## 具体任务、## AI执行指引、## 验收标准、## 关联

cron表达式：0 16 * * 1,4
时区：Asia/Shanghai
任务名称：heka-ky每周自动提Issue

=========================================
任务3：每周模块健康检查
=========================================

每周六上午10:00执行。

1. 验证GH_TOKEN：gh auth status 2>&1，如果失败则创建Issue提醒我重新配置Token

2. 运行项目测试套件：pytest tests/ --tb=short -q

3. 获取本周我的commit：git log --author="heka-ky" --since="7 days ago" --oneline

4. 如果有测试失败，创建Issue：
   标题："【每周健康检查】YYYY-MM-DD 测试失败"
   分配给 heka-ky
   内容：失败测试名称、错误信息、相关源码文件

5. 如果本周无commit，在我的一个open issue下评论："本周暂无代码提交，建议本周至少完成一个Issue。"

6. 全部通过且有提交则不操作（静默）。

cron表达式：0 10 * * 6
时区：Asia/Shanghai
任务名称：heka-ky每周模块健康检查
```

---

## NN2914 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建3个定时任务来辅助我的项目开发工作。

=========================================
任务1：每日Issue和PR进度提醒
=========================================

每天上午9:30执行。

执行步骤：
1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 查询分配给我的所有open issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee NN2914 --limit 50 --json number,title,labels,updatedAt

3. 查询我提交的所有open PRs：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author NN2914 --limit 20 --json number,title,headRefName,updatedAt

4. 对每个PR检查最新审查状态：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json reviews,state,statusCheckRollup

5. 对有review反馈的PR，检查是否有新commit（是否已修复）：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json commits

6. 对我的每个Issue，判断是否有关联PR

7. 在需要提醒的Issue下发布评论（@NN2914），内容包括：
   - 距上次更新天数
   - 有关联PR：PR审查状态、是否需要修复
   - 无关联PR：提醒开始开发
   - 距代码冻结(2026-08-15)剩余天数
   - 按优先级标签排序建议优先处理哪个

8. 有review反馈但未修复的PR：在PR下评论提醒我修复
9. 已修复但未被re-review的PR：在PR下@xiabai2008请求重新审查

约束：不对2天内更新过的Issue催促；无事项则不操作。

cron表达式：30 9 * * *
时区：Asia/Shanghai
任务名称：NN2914每日进度提醒

=========================================
任务2：每周自动提Issue（AI可执行任务）
=========================================

每周二和周五上午09:00执行（与Jackhock-1的16:00错开7小时，与周一/周四的DUMNOX和heka-ky错开整天）。

1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 审视项目（必做）：
   - 读取 AGENTS.md
   - 运行 ruff check src/ scripts/ tests/ --statistics 2>&1
   - 运行 pytest tests/ --cov=src --cov-report=term-missing --tb=no -q 2>&1
   - 查看现有open issues避免重复：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 100 --json number,title
   - 检查 docs/ 目录文档完整性
   - 检查最近7小时创建的Issue（避免与其他队友重复）：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 10 --json number,title,createdAt

3. 比赛背景：量子AI双向赋能（RL调度+量子退火），评审标准：主题契合度/技术创新性/方案可行性/落地价值/验证严谨性，代码冻结2026-08-15

4. 从以下维度发现1个AI可完成的改进任务：
   - 代码质量、测试覆盖、实验严谨性、文档完善、竞赛对齐、性能优化

5. 严格约束：
   - 只创建1个Issue，宁缺毋滥，无改进点则不创建
   - 不创建比赛交付物/真机操作Issue
   - 不与现有open issue重复，不与最近7小时创建的Issue重复
   - 不连续两次创建相同改进类型
   - 先检查我名下ai-ready标签的open issue数，超过3个则不创建
   - 预计工作量1-4小时

6. 创建Issue：
   gh issue create -R xiabai2008/quantum-rl-scheduler --title "【改进类型】简短描述" --body "正文" --assignee "NN2914" --label "优先级-中" --label "ai-ready"

   Issue正文必须包含：## 背景、## 具体任务、## AI执行指引、## 验收标准、## 关联

cron表达式：0 9 * * 2,5
时区：Asia/Shanghai
任务名称：NN2914每周自动提Issue

=========================================
任务3：每周CI状态与测试覆盖率检查
=========================================

每周六上午11:00执行。

1. 验证GH_TOKEN：gh auth status 2>&1，如果失败则创建Issue提醒我重新配置Token

2. 检查CI最近运行状态：gh run list -R xiabai2008/quantum-rl-scheduler --limit 10

3. 在本地运行测试覆盖率检查：pytest tests/ --cov=src --cov-report=term-missing --tb=short -q

4. 获取本周我的commit：git log --author="NN2914" --since="7 days ago" --oneline

5. 如果CI有失败或覆盖率低于70%，创建Issue：
   标题："【每周检查】YYYY-MM-DD CI/覆盖率问题"
   分配给 NN2914
   内容：CI失败详情或覆盖率数据，低于70%的模块列表

6. 如果本周无commit，在我的一个open issue下评论提醒。

7. 全部正常且有提交则不操作（静默）。

cron表达式：0 11 * * 6
时区：Asia/Shanghai
任务名称：NN2914每周CI与覆盖率检查
```

---

## Jackhock-1 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建3个定时任务来辅助我的项目开发工作。

=========================================
任务1：每日Issue和PR进度提醒
=========================================

每天上午9:30执行。

执行步骤：
1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 查询分配给我的所有open issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee Jackhock-1 --limit 50 --json number,title,labels,updatedAt

3. 查询我提交的所有open PRs：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author Jackhock-1 --limit 20 --json number,title,headRefName,updatedAt

4. 对每个PR检查最新审查状态：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json reviews,state,statusCheckRollup

5. 对有review反馈的PR，检查是否有新commit（是否已修复）：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json commits

6. 对我的每个Issue，判断是否有关联PR

7. 在需要提醒的Issue下发布评论（@Jackhock-1），内容包括：
   - 距上次更新天数
   - 有关联PR：PR审查状态、是否需要修复
   - 无关联PR：提醒开始开发
   - 距代码冻结(2026-08-15)剩余天数
   - 按优先级标签排序建议优先处理哪个

8. 有review反馈但未修复的PR：在PR下评论提醒我修复
9. 已修复但未被re-review的PR：在PR下@xiabai2008请求重新审查

约束：不对2天内更新过的Issue催促；无事项则不操作。

cron表达式：30 9 * * *
时区：Asia/Shanghai
任务名称：Jackhock-1每日进度提醒

=========================================
任务2：每周自动提Issue（AI可执行任务）
=========================================

每周二和周五下午16:00执行（与NN2914的09:00错开7小时）。

1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 审视项目（必做）：
   - 读取 AGENTS.md
   - 运行 ruff check src/ scripts/ tests/ --statistics 2>&1
   - 运行 pytest tests/ --cov=src --cov-report=term-missing --tb=no -q 2>&1
   - 查看现有open issues避免重复：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 100 --json number,title
   - 检查 docs/ 目录文档完整性
   - 检查最近7小时创建的Issue（避免与NN2914重复）：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 10 --json number,title,createdAt

3. 比赛背景：量子AI双向赋能（RL调度+量子退火），评审标准：主题契合度/技术创新性/方案可行性/落地价值/验证严谨性，代码冻结2026-08-15

4. 从以下维度发现1个AI可完成的改进任务：
   - 代码质量、测试覆盖、实验严谨性、文档完善、竞赛对齐、性能优化

5. 严格约束：
   - 只创建1个Issue，宁缺毋滥，无改进点则不创建
   - 不创建比赛交付物/真机操作Issue
   - 不与现有open issue重复，不与最近7小时创建的Issue重复
   - 不连续两次创建相同改进类型
   - 先检查我名下ai-ready标签的open issue数，超过3个则不创建
   - 预计工作量1-4小时

6. 创建Issue：
   gh issue create -R xiabai2008/quantum-rl-scheduler --title "【改进类型】简短描述" --body "正文" --assignee "Jackhock-1" --label "优先级-中" --label "ai-ready"

   Issue正文必须包含：## 背景、## 具体任务、## AI执行指引、## 验收标准、## 关联

cron表达式：0 16 * * 2,5
时区：Asia/Shanghai
任务名称：Jackhock-1每周自动提Issue

=========================================
任务3：每周实验数据完整性检查
=========================================

每周六下午14:00执行。

1. 验证GH_TOKEN：gh auth status 2>&1，如果失败则创建Issue提醒我重新配置Token

2. 检查实验数据文件是否存在：
   - results/multiseed_evaluation/ 目录
   - results/reports/ 目录下的报告文件

3. 如果存在 results/multiseed_evaluation/rewards_multiseed.json，读取并验证：
   - 计算PPO和FCFS的均值和标准差
   - 与AGENTS.md中的权威数字对比（PPO=2746.94±1160.72, FCFS=1458.77±60.47, +88.3%）
   - 如果偏差超过5%，创建GitHub Issue报告数据不一致，分配给Jackhock-1

4. 获取本周我的commit：git log --author="Jackhock-1" --since="7 days ago" --oneline

5. 如果本周无commit，在我的一个open issue下评论提醒。

6. 全部正常且有提交则不操作（静默）。

cron表达式：0 14 * * 6
时区：Asia/Shanghai
任务名称：Jackhock-1每周实验数据检查
```

---

## qpqpalalzmzm112 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建3个定时任务来辅助我的项目开发工作。

=========================================
任务1：每日Issue和PR进度提醒
=========================================

每天上午9:30执行。

执行步骤：
1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 查询分配给我的所有open issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee qpqpalalzmzm112 --limit 50 --json number,title,labels,updatedAt

3. 查询我提交的所有open PRs：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author qpqpalalzmzm112 --limit 20 --json number,title,headRefName,updatedAt

4. 对每个PR检查最新审查状态：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json reviews,state,statusCheckRollup

5. 对有review反馈的PR，检查是否有新commit（是否已修复）：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json commits

6. 对我的每个Issue，判断是否有关联PR

7. 在需要提醒的Issue下发布评论（@qpqpalalzmzm112），内容包括：
   - 距上次更新天数
   - 有关联PR：PR审查状态、是否需要修复
   - 无关联PR：提醒开始开发
   - 距代码冻结(2026-08-15)剩余天数
   - 按优先级标签排序建议优先处理哪个

8. 有review反馈但未修复的PR：在PR下评论提醒我修复
9. 已修复但未被re-review的PR：在PR下@xiabai2008请求重新审查

约束：不对2天内更新过的Issue催促；无事项则不操作。

cron表达式：30 9 * * *
时区：Asia/Shanghai
任务名称：qpqpalalzmzm112每日进度提醒

=========================================
任务2：每周自动提Issue（AI可执行任务）
=========================================

每周三和周六上午09:00执行（与Izzro的16:00错开7小时，与周一/二/四/五的队友错开整天）。

1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 审视项目（必做）：
   - 读取 AGENTS.md
   - 运行 ruff check src/ scripts/ tests/ --statistics 2>&1
   - 运行 pytest tests/ --cov=src --cov-report=term-missing --tb=no -q 2>&1
   - 查看现有open issues避免重复：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 100 --json number,title
   - 检查 docs/ 目录文档完整性
   - 检查最近7小时创建的Issue（避免与Izzro重复）：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 10 --json number,title,createdAt

3. 比赛背景：量子AI双向赋能（RL调度+量子退火），评审标准：主题契合度/技术创新性/方案可行性/落地价值/验证严谨性，代码冻结2026-08-15

4. 从以下维度发现1个AI可完成的改进任务：
   - 代码质量、测试覆盖、实验严谨性、文档完善、竞赛对齐、性能优化

5. 严格约束：
   - 只创建1个Issue，宁缺毋滥，无改进点则不创建
   - 不创建比赛交付物/真机操作Issue
   - 不与现有open issue重复，不与最近7小时创建的Issue重复
   - 不连续两次创建相同改进类型
   - 先检查我名下ai-ready标签的open issue数，超过3个则不创建
   - 预计工作量1-4小时

6. 创建Issue：
   gh issue create -R xiabai2008/quantum-rl-scheduler --title "【改进类型】简短描述" --body "正文" --assignee "qpqpalalzmzm112" --label "优先级-中" --label "ai-ready"

   Issue正文必须包含：## 背景、## 具体任务、## AI执行指引、## 验收标准、## 关联

cron表达式：0 9 * * 3,6
时区：Asia/Shanghai
任务名称：qpqpalalzmzm112每周自动提Issue

=========================================
任务3：每周代码质量自检
=========================================

每周六下午15:00执行。

1. 验证GH_TOKEN：gh auth status 2>&1，如果失败则创建Issue提醒我重新配置Token

2. 获取本周我的commit：git log --author="qpqpalalzmzm112" --since="7 days ago" --oneline

3. 运行代码质量检查：
   - ruff check src/ scripts/ tests/
   - mypy src/
   - pytest tests/ --tb=short -q

4. 如果有错误或测试失败，创建Issue：
   标题："【每周自检】YYYY-MM-DD 代码质量问题"
   分配给 qpqpalalzmzm112
   内容：错误列表和修复建议

5. 如果本周无commit，在我的一个open issue下评论提醒。

6. 全部通过且有提交则不操作（静默）。

cron表达式：0 15 * * 6
时区：Asia/Shanghai
任务名称：qpqpalalzmzm112每周代码自检
```

---

## Izzro 的提示词

> 复制以下代码块全部内容，粘贴到 TRAE 发送：

```
请帮我创建3个定时任务来辅助我的项目开发工作。

=========================================
任务1：每日Issue和PR进度提醒
=========================================

每天上午9:30执行。

执行步骤：
1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 查询分配给我的所有open issues：
   gh issue list -R xiabai2008/quantum-rl-scheduler --state open --assignee Izzro --limit 50 --json number,title,labels,updatedAt

3. 查询我提交的所有open PRs：
   gh pr list -R xiabai2008/quantum-rl-scheduler --state open --author Izzro --limit 20 --json number,title,headRefName,updatedAt

4. 对每个PR检查最新审查状态：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json reviews,state,statusCheckRollup

5. 对有review反馈的PR，检查是否有新commit（是否已修复）：
   gh pr view <PR号> -R xiabai2008/quantum-rl-scheduler --json commits

6. 对我的每个Issue，判断是否有关联PR

7. 在需要提醒的Issue下发布评论（@Izzro），内容包括：
   - 距上次更新天数
   - 有关联PR：PR审查状态、是否需要修复
   - 无关联PR：提醒开始开发
   - 距代码冻结(2026-08-15)剩余天数
   - 按优先级标签排序建议优先处理哪个

8. 有review反馈但未修复的PR：在PR下评论提醒我修复
9. 已修复但未被re-review的PR：在PR下@xiabai2008请求重新审查

约束：不对2天内更新过的Issue催促；无事项则不操作。

cron表达式：30 9 * * *
时区：Asia/Shanghai
任务名称：Izzro每日进度提醒

=========================================
任务2：每周自动提Issue（AI可执行任务）
=========================================

每周三和周六下午16:00执行（与qpqpalalzmzm112的09:00错开7小时）。

1. 设置GitHub CLI认证：$env:GH_TOKEN = (git remote get-url origin) -replace 'https://(ghp_[^@]+)@.*','$1'

2. 审视项目（必做）：
   - 读取 AGENTS.md
   - 运行 ruff check src/ scripts/ tests/ --statistics 2>&1
   - 运行 pytest tests/ --cov=src --cov-report=term-missing --tb=no -q 2>&1
   - 查看现有open issues避免重复：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 100 --json number,title
   - 检查 docs/ 目录文档完整性
   - 检查最近7小时创建的Issue（避免与qpqpalalzmzm112重复）：gh issue list -R xiabai2008/quantum-rl-scheduler --state open --limit 10 --json number,title,createdAt

3. 比赛背景：量子AI双向赋能（RL调度+量子退火），评审标准：主题契合度/技术创新性/方案可行性/落地价值/验证严谨性，代码冻结2026-08-15

4. 从以下维度发现1个AI可完成的改进任务：
   - 代码质量、测试覆盖、实验严谨性、文档完善、竞赛对齐、性能优化

5. 严格约束：
   - 只创建1个Issue，宁缺毋滥，无改进点则不创建
   - 不创建比赛交付物/真机操作Issue
   - 不与现有open issue重复，不与最近7小时创建的Issue重复
   - 不连续两次创建相同改进类型
   - 先检查我名下ai-ready标签的open issue数，超过3个则不创建
   - 预计工作量1-4小时

6. 创建Issue：
   gh issue create -R xiabai2008/quantum-rl-scheduler --title "【改进类型】简短描述" --body "正文" --assignee "Izzro" --label "优先级-中" --label "ai-ready"

   Issue正文必须包含：## 背景、## 具体任务、## AI执行指引、## 验收标准、## 关联

cron表达式：0 16 * * 3,6
时区：Asia/Shanghai
任务名称：Izzro每周自动提Issue

=========================================
任务3：每周代码质量自检
=========================================

每周六下午16:00执行。

1. 验证GH_TOKEN：gh auth status 2>&1，如果失败则创建Issue提醒我重新配置Token

2. 获取本周我的commit：git log --author="Izzro" --since="7 days ago" --oneline

3. 运行代码质量检查：
   - ruff check src/ scripts/ tests/
   - mypy src/
   - pytest tests/ --tb=short -q

4. 如果有错误或测试失败，创建Issue：
   标题："【每周自检】YYYY-MM-DD 代码质量问题"
   分配给 Izzro
   内容：错误列表和修复建议

5. 如果本周无commit，在我的一个open issue下评论："本周暂无代码提交，建议本周至少完成一个Issue。"

6. 全部通过且有提交则不操作（静默）。

cron表达式：0 16 * * 6
时区：Asia/Shanghai
任务名称：Izzro每周代码自检
```

---

## 扫描时间排布表

| 日期 | 09:00 | 16:00 | 间隔 |
|:--|:--|:--|:--|
| 周一 | DUMNOX | heka-ky | 7小时 |
| 周二 | NN2914 | Jackhock-1 | 7小时 |
| 周三 | qpqpalalzmzm112 | Izzro | 7小时 |
| 周四 | DUMNOX | heka-ky | 7小时 |
| 周五 | NN2914 | Jackhock-1 | 7小时 |
| 周六 | qpqpalalzmzm112 | Izzro | 7小时 |

- 同日两人间隔7小时（09:00 → 16:00）
- 跨日间隔17小时（16:00 → 次日09:00）
- 每人每周2次，6人共12次/周
- 每次扫描前检查最近7小时创建的Issue，避免重复

## 任务总览

| 队友 | 每日进度提醒 | 自动提Issue | 每周专项检查 |
|:--|:--|:--|:--|
| DUMNOX | 10:30 | 周一/四 09:00 | 周日15:00 代码自检 |
| heka-ky | 09:30 | 周一/四 16:00 | 周六10:00 模块健康 |
| NN2914 | 09:30 | 周二/五 09:00 | 周六11:00 CI+覆盖率 |
| Jackhock-1 | 09:30 | 周二/五 16:00 | 周六14:00 实验数据 |
| qpqpalalzmzm112 | 09:30 | 周三/六 09:00 | 周六15:00 代码自检 |
| Izzro | 09:30 | 周三/六 16:00 | 周六16:00 代码自检 |

## 工作流程闭环

1. **进度提醒**（09:30/10:30）：查询我的Issue/PR → 没有PR的提醒开发 → 有review反馈的提醒修复 → 已修复的@xiabai2008请求re-review
2. **自动提Issue**（各人错开时间）：TRAE审视项目 → 发现AI可完成的改进点 → 创建带"AI执行指引"的Issue分配给自己
3. **做Issue**：看到新Issue → 复制内容发给TRAE → TRAE按指引完成 → 提交PR
4. **每周自检**（周六/周日）：ruff/mypy/pytest检查 → 发现问题自动创建Issue → 零提交提醒

## 注意事项

- TRAE 需要工作时间内保持运行，否则定时任务不会执行
- GitHub 邮件通知突然停止 = Token可能过期，重新执行认证命令
- 每人每周2次自动提Issue，6人共12次/周，避免Issue洪水
- 每人最多3个ai-ready open Issue，超过则先完成现有的
- 6人扫描时间按天+时段双重错开，相邻间隔至少7小时
- 在TRAE中说"列出我的定时任务"可以管理（暂停/修改/删除）
