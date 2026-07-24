# 生产落地路径规划文档

> **Issue #130** | 生成时间: 2026-07-24
> **适用范围**: 项目规划、答辩材料、技术白皮书

---

## 一、当前状态评估

### 1.1 成熟度评级

| 维度 | 评级 | 评分 | 依据 |
|:--|:--|:--|:--|
| **功能完整性** | 研究原型（高级） | 8/10 | PPO/DQN/MAPPO 三算法 + 8基线 + 量子退火 + 真机接入 + 可视化 |
| **工程化程度** | 工程原型（高级） | 8/10 | CI 6 Job + Docker 一键部署 + 500+测试 + 91%覆盖率 + ruff/mypy/bandit |
| **生产就绪度** | 试点就绪 | 5/10 | 监控/告警/熔断已实现，但状态持久化/多租户集成/高可用待完善 |
| **可复现性** | 研究级（优秀） | 9/10 | 50seed N=250 + 统计显著性 + Bonferroni校正 + 权威数字锁定 |
| **文档完备性** | 竞赛级（优秀） | 9/10 | AGENTS.md + 10份实验报告 + 答辩QA手册 + 代码Wiki |

**综合评级**：**研究原型向工程原型过渡阶段，具备试点部署能力**

### 1.2 已具备的生产能力

| 能力 | 实现状态 | 代码位置 |
|:--|:--|:--|
| CI/CD 流水线 | 6 Job（lint→test→typecheck→security→mutation→benchmark） | `.github/workflows/ci.yml` |
| Docker 容器化 | 多阶段构建 + docker-compose + 健康检查 | `Dockerfile`, `docker-compose.yml` |
| Prometheus 监控 | 7类指标 + /metrics端点 + 告警规则 | `src/utils/metrics.py`, `config/prometheus.yml` |
| API 熔断器 | CLOSED/OPEN/HALF_OPEN 三态 | `src/api/circuit_breaker.py` |
| 配额追踪 | JSON持久化 + 线程安全 | `src/api/quota_tracker.py` |
| 检查点管理 | 版本管理 + 性能对比 | `src/scheduler/checkpoint_manager.py` |
| 多租户框架 | 配额管理 + 4租户预配置 | `src/scheduler/tenant.py` |
| 统一异常体系 | 8类异常 + code/retryable语义 | `src/exceptions.py` |
| FastAPI 监控面板 | /health + /ready + /ws + /api/status | `src/visualization/app.py` |
| 真机接入 | 天衍云 cqlib SDK + 14后端 + 自动故障切换 | `src/api/tianyan_cqlib.py` |

### 1.3 关键差距

| 差距 | 当前状态 | 生产要求 | 优先级 |
|:--|:--|:--|:--|
| 状态持久化 | Web状态纯内存 | Redis/SQLite 持久化 | P0 |
| Redis 缓存接入 | 配置已声明，SchedulerCache 为内存实现 | SchedulerCache 接入 Redis | P0 |
| SQLite 数据层 | 配置已声明，无建表/迁移代码 | ORM + 迁移脚本 | P1 |
| 多租户调度集成 | TenantQuotaManager 已实现，调度引擎未调用 | 调度引擎集成配额检查 | P1 |
| Alertmanager | 告警规则已定义，无 Alertmanager 配置 | Alertmanager + 通知渠道 | P1 |
| Grafana Dashboard | Provider 已配置，无 Dashboard JSON | 至少1个监控仪表盘 | P2 |
| 跨平台测试 | 仅 Windows runner | Linux + macOS 矩阵 | P2 |
| 安全审计阻断 | pip-audit 为 continue-on-error | 安全漏洞阻断 CI | P2 |

---

## 二、落地路径分阶段规划

### 阶段1：竞赛交付（截止 2026-08-15）

**目标**：代码冻结，交付物完善，答辩准备

| 任务 | 状态 | 截止 | 负责人 |
|:--|:--|:--|:--|
| 代码冻结（v8.0-submission 标签） | 待执行 | 08/15 | 瑞哥 |
| CI 全绿（lint/test/typecheck/security） | 待验证 | 08/15 | 全员 |
| `validate_submission.py --check` 通过 | 待验证 | 08/15 | 全员 |
| PPT/白皮书数字与权威数字一致 | 待验证 | 08/15 | 瑞哥 |
| 演示视频录制（4-5分钟，1080p） | 待录制 | 08/15 | 瑞哥 |
| 提交清单打包 `validate_submission.py --pack` | 待执行 | 08/15 | 全员 |

**交付物清单**：
- 源代码（v8.0-submission 标签）
- 答辩PPT（v5+）
- 技术白皮书（v5+）
- 演示视频
- 实验数据（results/ 全目录）
- 模型文件（deliverable_models/）

### 阶段2：试点部署（2026-08 ~ 2026-10）

**目标**：在受控环境中验证系统端到端可用性

| 任务 | 优先级 | 工作量 | 关联Issue |
|:--|:--|:--|:--|
| Web 监控状态 Redis 持久化 | P0 | 3-5天 | #114 |
| SchedulerCache 接入 Redis | P0 | 2-3天 | — |
| SQLite 数据层建表 + 迁移 | P1 | 3-5天 | — |
| Alertmanager 配置 + 通知渠道 | P1 | 2天 | — |
| Grafana Dashboard JSON | P1 | 2-3天 | — |
| 日志规范化（结构化JSON日志） | P1 | 2天 | — |
| Docker 生产配置优化（secrets管理） | P1 | 1-2天 | — |

**试点验证指标**：
- 系统连续运行 72 小时无崩溃
- API 响应时间 P99 < 500ms
- 真机任务提交成功率 > 95%
- 监控指标完整暴露（7类 Prometheus 指标）
- 告警链路验证（告警触发 → Alertmanager → 通知）

### 阶段3：生产部署（2026-10 ~ 2027-01）

**目标**：面向真实量子计算平台的生产级部署

| 任务 | 优先级 | 工作量 | 关联Issue |
|:--|:--|:--|:--|
| 多租户调度引擎集成 | P0 | 5-7天 | #113 |
| 租户认证与隔离（API Key + 资源隔离） | P0 | 5-7天 | — |
| 高可用架构（主备切换 + 负载均衡） | P1 | 1-2周 | — |
| 性能调优（推理延迟 < 100ms） | P1 | 1周 | — |
| 数据备份与恢复（Redis AOF + SQLite WAL） | P1 | 2-3天 | — |
| 安全加固（TLS + API认证 + 审计日志） | P1 | 1周 | — |
| 跨平台 CI（Linux + macOS 矩阵） | P2 | 2天 | — |
| 安全审计阻断（pip-audit 硬门禁） | P2 | 1天 | — |

**生产部署指标**：
- SLA: 99.5% 可用性
- 推理延迟 P99 < 100ms
- 支持至少 4 个租户并发
- 真机任务排队时间 < 5 分钟（优先队列）
- 数据零丢失（Redis AOF + SQLite WAL）

### 阶段4：规模化（2027-01 ~ 2027-06）

**目标**：多硬件适配与云原生部署

| 任务 | 优先级 | 工作量 | 关联Issue |
|:--|:--|:--|:--|
| 多硬件适配（天衍-287/176/504 + 其他厂商） | P0 | 2-3周 | #100 |
| Kubernetes 云原生部署 | P1 | 2周 | — |
| SLA 保障体系（99.9% + 监控 + 告警 + 故障恢复） | P1 | 1-2周 | — |
| 水平扩展（多实例 + 共享状态） | P1 | 1-2周 | — |
| 量子退火改进（分块退火 + 编码精度提升） | P2 | 4-6周 | #127 |
| 量子特征映射探索 | P2 | 3-4月 | #127 |

**规模化指标**：
- SLA: 99.9% 可用性
- 支持 10+ 量子硬件后端
- 支持 100+ 并发租户
- 退火贡献提升至 +12-18%（Issue #127 改进方案）

---

## 三、技术债务清单

### 3.1 按优先级排序

| 编号 | 技术债务 | 影响 | 优先级 | 修复工作量 | 阶段 |
|:--|:--|:--|:--|:--|:--|
| TD-01 | Web 监控状态纯内存存储 | 进程重启丢失所有调度状态 | P0 | 3-5天 | 阶段2 |
| TD-02 | Redis 配置与实现不匹配 | 缓存配置无效，实际为内存缓存 | P0 | 2-3天 | 阶段2 |
| TD-03 | SQLite 数据层未落地 | 配置声明但无建表/迁移代码 | P1 | 3-5天 | 阶段2 |
| TD-04 | 多租户调度未集成 | TenantQuotaManager 已实现但调度引擎未调用 | P1 | 5-7天 | 阶段3 |
| TD-05 | Alertmanager 未配置 | 告警仅 loguru 输出，无外部通知 | P1 | 2天 | 阶段2 |
| TD-06 | Grafana Dashboard 缺失 | Provider 已配置但无仪表盘 JSON | P2 | 2-3天 | 阶段2 |
| TD-07 | 跨平台测试缺失 | 仅 Windows runner，Linux/macOS 未验证 | P2 | 2天 | 阶段3 |
| TD-08 | 安全审计未阻断 | pip-audit 为 continue-on-error | P2 | 1天 | 阶段3 |
| TD-09 | CI 并发配置问题 | ci.yml 中重复 concurrency 块 | P2 | 0.5天 | 阶段1 |
| TD-10 | Docker secrets 管理不足 | 环境变量直接写入 compose 文件 | P1 | 1-2天 | 阶段2 |

### 3.2 技术债务趋势

| 阶段 | 新增债务 | 偿还债务 | 净债务 |
|:--|:--|:--|:--|
| v1-v8 开发期 | 10 | 0 | 10 |
| 阶段1（竞赛交付） | 0-1 | 1（TD-09） | 9-10 |
| 阶段2（试点部署） | 2-3 | 6（TD-01~03, 05, 06, 10） | 5-7 |
| 阶段3（生产部署） | 1-2 | 4（TD-04, 07, 08 + 高可用） | 2-5 |
| 阶段4（规模化） | 0-1 | 2-3 | 0-3 |

---

## 四、风险评估

### 4.1 主要风险及缓解措施

| 风险 | 概率 | 影响 | 缓解措施 |
|:--|:--|:--|:--|
| **真机机时不足** | 高 | 高 | 多机器故障切换已实现；申请更大机时包；使用仿真为主验证手段 |
| **退火改进未达预期** | 中 | 中 | 分阶段验证（编码精度→分块退火）；保留退火可关闭开关 |
| **多租户集成复杂度** | 中 | 中 | 先实现配额检查，后实现资源隔离；分租户灰度上线 |
| **高可用架构成本** | 中 | 中 | 主备切换优先于集群；利用 Docker restart policy |
| **跨硬件兼容性** | 中 | 高 | 统一 cqlib 接口；硬件适配层抽象；逐硬件验证 |
| **安全漏洞** | 低 | 高 | pip-audit 阻断 CI；定期依赖更新（Dependabot）；安全审计 |
| **性能退化** | 中 | 高 | 基准测试 CI 集成；性能回归告警；P99 延迟监控 |

### 4.2 风险矩阵

```
影响 ↑
 高  │  真机机时不足    │  跨硬件兼容性    │
     │  安全漏洞        │  性能退化        │
 中  │  退火改进未达预期 │  多租户集成复杂度│
     │                  │  高可用架构成本  │
 低  │                  │                  │
     └──────────────────┴──────────────────→ 概率
        低                中                高
```

---

## 五、README.md 生产就绪度摘要（待添加段落）

在 README.md 项目概述部分添加以下内容：

```markdown
## 生产就绪度

| 维度 | 评级 | 状态 |
|:--|:--|:--|
| CI/CD | ★★★★★ | 6 Job 流水线（lint/test/typecheck/security/mutation/benchmark） |
| 容器化 | ★★★★☆ | Docker 多阶段构建 + docker-compose + 健康检查 |
| 监控告警 | ★★★★☆ | Prometheus 7指标 + 告警规则 + FastAPI 健康检查 |
| 测试覆盖 | ★★★★★ | 500+ 用例，91% 覆盖率，property-based + mutation testing |
| 真机接入 | ★★★★☆ | 天衍云 cqlib SDK，14后端，284次调用100%成功 |
| 状态持久化 | ★★☆☆☆ | 纯内存存储，Redis/SQLite 待落地（阶段2） |
| 多租户 | ★★★☆☆ | 框架已实现，调度引擎集成待完成（阶段3） |

**落地路径**：竞赛交付(08/15) → 试点部署(08-10月) → 生产部署(10-01月) → 规模化(01-06月)

详见 `docs/production_roadmap.md`
```

---

## 六、数据完整性声明

### 权威数字一致性

| 指标 | 值 | 来源 | 本文档引用 |
|:--|:--|:--|:--|
| 仿真 PPO 提升 | +88.3% | 50 seeds × 5 episodes | ✅ 一致 |
| 仿真 p 值 | 3.04e-11 | Welch t 检验 | ✅ 一致 |
| 仿真 Cohen's d | -1.70 | 同上 | ✅ 一致 |
| 多seed真机 PPO | 1665.22 ± 324.51 | 5 seeds × 3 策略 | ✅ 一致 |
| 多seed真机 p 值 | 6.83e-04 | Bonferroni 校正后显著 | ✅ 一致 |
| 测试覆盖率 | 91% | CI pytest --cov | ✅ 一致 |
| CI Job 数 | 6 | ci.yml | ✅ 一致 |
| 测试用例数 | 500+ | tests/ 目录 | ✅ 一致 |

---

## 七、关联文档

| 文档 | 路径 | 说明 |
|:--|:--|:--|
| 项目记忆 | `AGENTS.md` | 项目全貌 |
| CI 配置 | `.github/workflows/ci.yml` | 6 Job 流水线 |
| Docker 配置 | `Dockerfile`, `docker-compose.yml` | 容器化部署 |
| 监控配置 | `config/prometheus.yml`, `config/alerts.yml` | Prometheus + 告警 |
| 非对称性分析 | `docs/dual_empowerment_asymmetry_analysis.md` | Issue #127 退火改进方案 |
| 真机验证边界 | `docs/real_machine_verification_boundary.md` | Issue #128 真机验证结论 |
| 观测维度标准 | `docs/observation_dim_standard.md` | Issue #129 口径管理 |

---

*Issue #130 验收文件 | 2026-07-24*
