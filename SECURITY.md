# 安全策略

> 本文件定义 **量子RL驱动的天衍云平台智能调度系统** 的安全规范与漏洞报告流程。
> GitHub 会在仓库 Security tab 自动展示此文件内容。

**最后更新**：2026-07-20

---

## 1. 支持的版本

| 版本 | 支持状态 | 说明 |
|------|:--------:|------|
| 8.0 | :white_check_mark: | 当前版本（比赛提交版本） |
| < 8.0 | :x: | 不支持，请升级至最新版本 |

代码冻结日（2026-08-15）后，仅 8.0 版本接受安全修复。

---

## 2. 漏洞报告流程

### 2.1 报告渠道

**请勿通过公开 GitHub Issue 报告安全漏洞**，以避免漏洞被恶意利用。

发送邮件至项目负责人邮箱：

- **收件人**：xiabai2004（瑞哥）— 见 GitHub 个人主页
- **邮件主题**：`[SECURITY] 量子RL调度系统 — 简要描述`
- **加密**：如涉及敏感信息，建议使用 PGP 加密（公钥可向负责人索取）

### 2.2 报告内容

请尽量包含以下信息，便于我们快速定位和修复：

1. 漏洞类型（如：API 密钥泄露、依赖漏洞、注入攻击、权限提升）
2. 受影响的文件或模块（如：`src/api/tianyan_cqlib.py`）
3. 复现步骤（含最小化复现代码）
4. 影响评估（可导致的最大损害）
5. 建议的修复方案（如有）

### 2.3 响应承诺

| 阶段 | 时间 | 行动 |
|------|------|------|
| 确认接收 | 48 小时内 | 邮件回复确认收到报告 |
| 初步评估 | 7 天内 | 评估漏洞严重程度（CVSS 评分） |
| 修复发布 | 30 天内 | 发布修复版本（严重漏洞优先） |
| 公开披露 | 修复后 90 天 | 在 GitHub Security Advisory 公开披露 |

### 2.4 漏洞严重程度分级

| 等级 | CVSS | 描述 | 示例 |
|:----:|:----:|------|------|
| :red_circle: 严重 | 9.0-10.0 | 可导致真机 API 密钥泄露或系统完全控制 | 硬编码密钥、远程代码执行 |
| :orange_circle: 高 | 7.0-8.9 | 可导致敏感信息泄露或服务中断 | 未授权的真机调用、配额绕过 |
| :yellow_circle: 中 | 4.0-6.9 | 有限的敏感信息泄露或功能滥用 | 路径遍历、CSRF |
| :green_circle: 低 | 0.1-3.9 | 影响有限的轻量级问题 | 信息泄露（日志含调试信息） |

---

## 3. 密钥与凭证管理

### 3.1 核心原则

- **永不硬编码**：API 密钥、Token、密码等凭证**禁止**出现在源代码中
- **环境变量加载**：所有凭证通过 `.env` 环境变量加载（参考 `.env.example`）
- **版本控制排除**：`.env` 已在 `.gitignore` 中，不会进入版本控制
- **最小权限原则**：每个组件仅持有其必需的最小权限

### 3.2 涉及的凭证清单

| 凭证 | 用途 | 存储位置 | 轮换周期 |
|------|------|----------|----------|
| `TIANYAN_API_KEY` | 天衍云 API 主密钥 | `.env` | 90 天 |
| `TIANYAN_API_TOKEN` | API 密钥别名 | `.env` | 90 天 |
| `TIANYAN_API_SECRET` | 双因子认证密钥 | `.env` | 90 天 |
| `TIANYAN_APP_ID` | 应用级鉴权 ID | `.env` | 按需 |

### 3.3 凭证泄露应急

若发现凭证泄露（如误提交到 Git 仓库）：

1. **立即**在天衍云平台吊销并重新生成密钥
2. 清理 Git 历史（使用 `git filter-repo` 或 BFG Repo-Cleaner）
3. 强制推送受影响分支（仅在负责人批准后执行）
4. 在 GitHub Security Advisory 记录事件
5. 通知所有团队成员更换本地凭证

### 3.4 相关代码模块

- `.env.example` — 环境变量模板（不含真实值）
- `src/config/settings.py` — 配置加载（从环境变量读取）
- `src/api/tianyan_cqlib.py` — 真机 API 客户端（使用凭证）
- `src/api/quota_tracker.py` — 配额追踪（防止资源滥用）

---

## 4. 依赖安全

### 4.1 自动化监控

| 工具 | 作用 | 频率 | 配置文件 |
|------|------|------|----------|
| **Dependabot** | 监控 GitHub 依赖漏洞 | 每周 | `.github/dependabot.yml` |
| **pip-audit** | 扫描 PyPI 依赖漏洞 | 每次 PR | `.github/workflows/ci.yml`（Job 4） |
| **Bandit** | 检测代码层面安全问题 | 每次 PR | `pyproject.toml` + CI Job 1 |

### 4.2 依赖更新策略

- Dependabot 发现漏洞后，自动创建 PR 升级依赖
- 维护者需在 7 天内 review 并合并安全相关 PR
- 依赖版本锁定在 `requirements.txt`，避免隐式升级

### 4.3 依赖分层

| 文件 | 用途 | 安装命令 |
|------|------|----------|
| `requirements.txt` | 基础依赖（含 dimod/dwave-neal） | `pip install -r requirements.txt` |
| `requirements-quantum.txt` | 真机可选依赖（cqlib） | `pip install -r requirements-quantum.txt` |

---

## 5. 安全最佳实践

### 5.1 输入验证

- **任务描述验证**：`src/scheduler/parser.py` 对量子任务 QCIS 电路进行语法校验
- **API 响应验证**：所有外部 API 响应都经过类型和字段验证
- **配置校验**：`src/config/schema.py` 使用 Pydantic schema 验证配置

### 5.2 服务韧性

- **熔断器**（`src/api/circuit_breaker.py`）：CLOSED/OPEN/HALF_OPEN 三态转换，防止级联故障
- **配额追踪**（`src/api/quota_tracker.py`）：跟踪真机 API 调用次数，防止资源耗尽
- **重试机制**：API 调用失败时指数退避重试，避免雪崩

### 5.3 可观测性

- **Prometheus 指标**：7 个指标覆盖调度/API/退火三个维度（`src/utils/metrics.py`）
- **结构化日志**：所有关键操作都有可追溯的日志记录
- **`/metrics` 端点**：实时暴露指标，便于监控告警

### 5.4 代码安全规范

- 禁止使用 `eval()` / `exec()` / `pickle.loads()` 处理不可信输入
- 禁止使用 `shell=True` 调用子进程
- 文件操作必须使用 `Path` 对象，避免路径遍历
- Bandit 扫描必须在 CI 中通过（严格阻断）

---

## 6. 比赛提交相关

### 6.1 代码冻结前检查清单

在 2026-08-15 代码冻结前，需确认：

- [ ] `.env` 不在版本控制中（`.gitignore` 已配置）
- [ ] 源代码中无硬编码凭证（Bandit 扫描通过）
- [ ] `requirements.txt` 依赖无已知漏洞（pip-audit 通过）
- [ ] 所有 API 调用都通过熔断器保护
- [ ] 所有真机调用都通过配额追踪
- [ ] `SECURITY.md` 已就位（本文件）

### 6.2 提交清单对应

本文件对应 `config/submission_manifest.yaml` 中的工程完整性要求，是比赛评审的加分项。

---

## 7. 联系方式

- **项目负责人**：瑞哥（GitHub: [@xiabai2004](https://github.com/xiabai2004)）
- **团队**：8 人团队（详见 [AGENTS.md](AGENTS.md) 第 10 节）
- **安全报告邮箱**：见负责人 GitHub 主页

---

## 8. 致谢

感谢以下安全工具与平台的支持：

- [Dependabot](https://docs.github.com/en/code-security/dependabot)
- [Bandit](https://bandit.readthedocs.io/)
- [pip-audit](https://github.com/pypa/pip-audit)
- [GitHub Security Lab](https://securitylab.github.com/)
