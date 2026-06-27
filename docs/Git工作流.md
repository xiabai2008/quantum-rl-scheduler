# Git 工作流指南

## 项目：量子RL驱动的天衍云平台智能调度系统

---

## 一、分支策略：简化版 GitHub Flow

我们采用 **简化版 GitHub Flow**（不是复杂的 Git Flow），因为：
- 团队10人、开发周期仅4个月，不需要复杂分支
- 每天合并、持续集成，避免长时间分支的合并冲突

```
main（保护分支，只接受PR合并）
  │
  ├── feature/rl-agent          ← 算法A开发RL智能体
  ├── feature/quantum-annealing  ← 算法B开发量子退火
  ├── feature/task-parser        ← 算法C开发任务解析
  ├── feature/tianyan-api        ← 后端A开发天衍API
  ├── feature/scheduler-core     ← 后端B开发调度核心
  ├── feature/web-ui             ← 前端开发监控界面
  ├── feature/visualization      ← 前端开发可视化
  ├── fix/xxx                    ← 修Bug分支（命名：fix/问题简述）
  └── docs/xxx                   ← 文档更新分支
```

---

## 二、第一次使用：克隆仓库 + 安装 Hooks + 创建分支

```bash
# 1. 克隆仓库（所有人）
git clone https://github.com/xiabai2004/quantum-rl-scheduler.git
cd quantum-rl-scheduler

# 2. 安装 Git Hooks（必做！否则可以直接推 main，违反规范）
bash scripts/install-hooks.sh

# 3. 安装 Python 依赖
pip install -r requirements.txt

# 4. 创建自己的功能分支
git checkout main
git pull origin main
git checkout -b feature/你的模块名

# 示例：算法A创建RL智能体分支
git checkout -b feature/rl-agent
```

> **⚠️ 重要：Hook 安装说明**
> - `bash scripts/install-hooks.sh` 会配置两个本地拦截：
>   - **commit-msg**：提交信息必须符合 `<type>: <描述>` 格式，否则拒绝提交
>   - **pre-push**：直接推送 main 分支会被拦截，必须走 PR 流程
> - Hook 通过 `git config core.hooksPath githooks` 实现，仅影响本地仓库
> - 卸载：`git config --unset core.hooksPath`
> - 绕过（仅紧急情况）：`git push --no-verify`（禁止日常使用）

---

## 三、日常工作流（每个人每天的操作）

```bash
# ====== 早上：同步最新代码 ======
git checkout main
git pull origin main          # 拉取最新main
git checkout feature/你的分支
git merge main                # 把main合入自己的分支（避免累积冲突）

# ====== 白天：写代码 ======
# ... 用 TRAE 写代码 ...

# ====== 提交代码 ======
git add src/scheduler/agent.py    # 只加自己改的文件！
git commit -m "feat: 实现DQN网络架构，完成训练循环"

# ====== 晚上：推送并提PR ======
git push origin feature/你的分支
# 然后去 GitHub 网页创建 Pull Request
```

---

## 四、Commit 规范

### 格式
```
<type>: <简短描述>

<详细说明（可选）>
```

### type 类型

| type | 含义 | 示例 |
|------|------|------|
| `feat` | 新功能 | `feat: 实现DQN智能体训练循环` |
| `fix` | 修Bug | `fix: 修复量子退火QUBO矩阵维度不匹配` |
| `refactor` | 重构（不改功能） | `refactor: 提取奖励计算为独立函数` |
| `test` | 添加测试 | `test: 添加任务解析器边界条件测试` |
| `docs` | 文档 | `docs: 更新API接口文档` |
| `style` | 格式（空格、逗号等） | `style: 用Black格式化所有Python文件` |
| `perf` | 性能优化 | `perf: 优化任务队列数据结构，降低查找复杂度` |

### 好的 Commit 示例
```bash
✅ feat: 实现DQN智能体的Duelling网络架构
✅ fix: 修复训练时epsilon衰减溢出bug
✅ refactor: 将状态编码从8维扩展到12维，增加量子噪声特征
❌ update code
❌ fix bug
❌ 修改了一些东西
```

---

## 五、Pull Request 规范

### 创建 PR 前
```bash
# 1. 确保代码能跑
python -m pytest tests/

# 2. 格式化代码
black src/

# 3. 同步main到自己的分支
git checkout main && git pull origin main
git checkout feature/你的分支 && git merge main
```

### PR 标题
```
[模块] 简要描述
```
示例：`[RL智能体] 实现Duelling DQN网络，训练50000步收敛`

### PR 描述模板（GitHub会自动填充）
```markdown
## 做了什么
- 实现了Duelling DQN网络架构
- 训练50000步后在仿真环境达到平均奖励+150

## 测试结果
- 单元测试全部通过
- 仿真测试：任务完成率82%（FCFS为60%）

## 截图/日志
（如果有Web界面变化，贴截图）

## Review 要点
- [ ] 奖励函数设计是否合理
- [ ] 超参数选择是否有依据
```

### Review 规则
- 每个PR至少 **1人 Review + 1人 Approve** 才能合并
- 项目经理合并所有PR（或在Review通过后由作者自行合并）
- 禁止直接 push 到 main 分支！

---

## 六、冲突解决

```bash
# 如果 merge main 时出现冲突：
git checkout feature/你的分支
git merge main

# Git 会提示冲突文件，打开冲突文件，手动编辑
# 冲突标记长这样：
<<<<<<< HEAD
你的代码
=======
main分支的代码
>>>>>>> main

# 编辑后保存，然后：
git add .
git commit -m "merge: 解决与main的冲突"
git push origin feature/你的分支
```

---

## 七、常用命令速查

```bash
# 查看状态
git status

# 查看分支
git branch -a

# 暂存修改（写到一半需要切分支）
git stash          # 暂存
git stash pop      # 恢复

# 撤销修改
git checkout -- 文件名         # 撤销单个文件
git reset --soft HEAD~1        # 撤销最近一次commit（保留修改）

# 查看历史
git log --oneline --graph      # 彩色分支图
git log --author="你的名字"     # 只看自己的提交

# 同步fork（如果用的是fork而非直接clone）
git remote add upstream https://github.com/原始仓库/quantum-rl-scheduler.git
git fetch upstream
git merge upstream/main
```

---

## 八、紧急情况处理

| 情况 | 操作 |
|------|------|
| 不小心push了敏感信息 | 立即告诉项目经理，不要继续操作 |
| 合错了分支 | `git revert` 不要 `git reset --hard` |
| 电脑坏了 | 在另一台设备 `git clone`，代码不会丢 |
| 不知道自己做了什么修改 | `git diff` 看差异 |

---

## 九、GitHub 仓库设置（项目经理操作）

### 1. 创建仓库
浏览器打开 https://github.com/new
- Repository name: `quantum-rl-scheduler`
- Description: 量子RL驱动的天衍云平台智能调度系统 - 2026"揭榜挂帅"擂台赛参赛作品
- Public / Private: **建议 Private**（比赛结束前不公开）
- 不要勾选 "Initialize with README"（我们已有代码）

### 2. 保护 main 分支
Settings → Branches → Add branch protection rule
- Branch name pattern: `main`
- ☑ Require a pull request before merging
- ☑ Require approvals (1)
- ☑ Dismiss stale pull request approvals when new commits are pushed

### 3. 添加协作者
Settings → Collaborators → 添加团队成员的 GitHub 账号
