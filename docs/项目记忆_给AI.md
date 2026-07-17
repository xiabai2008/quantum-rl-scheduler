# 揭榜挂帅擂台赛 - 项目记忆文档

> **同步给 TRAE AI 助手 | 更新时间**：2026-07-01 (v7)
> **项目负责人**：瑞哥（xiabai2004）

---

## 项目背景

### 比赛信息
- **赛事**：2026年"揭榜挂帅"擂台赛 | **选题编号**：XA-202609
- **发榜单位**：中国电信集团有限公司 | **执行单位**：中电信量子
- **作品名称**：量子RL驱动的天衍云平台智能调度系统
- **真机**：天衍云287量子比特超导量子计算机（3台：tianyan_s/sw/tn）
- **报名状态**：6/30 已通过审核

### 关键节点
| 日期 | 事项 | 状态 |
|------|------|------|
| 2026-06-30 | 报名截止 | 已通过 |
| 2026-09-15 | 作品提交 | 进行中 |
| 2026-09-30 | 初审 | — |
| 2026-11 | 终审擂台赛 | — |

### 核心创新
- **AI 赋能量子**：PPO强化学习智能调度量子/经典任务，PPO vs FCFS +92.4%
- **量子赋能 AI**：量子退火(QUBO映射)加速RL策略搜索
- **量化目标**：资源利用率+30%，等待时间-40%

---

## 项目当前状态（v7）

| 指标 | 数值 |
|------|------|
| src/ 代码 | 22文件，~11,000行 |
| tests/ | 14文件，~4,485行，100+用例 |
| CI覆盖率 | 60%（目标80%） |
| mypy类型检查 | 8项严格配置（2模块豁免） |
| 真机任务 | 32个（3台天衍云超导） |
| 实验成果 | PPO +92.4%，多机 +86.3%，消融全完成 |
| 比赛材料 | PPT 15页 + 白皮书 10章 + 视频脚本完成 |

### 完成进度
```
v1 技术提升   ████████████████░   83%
Track A       ████████████████████ 100%
Track B       ████████████████████ 100%
Track C       ████████░░░░░░░░░░   40%（待修复退步）
真机闭环      ░░░░░░░░░░░░░░░░░░   0%（待开发）
```

---

## 项目结构（v7 精简版）

```
quantum-rl-scheduler/
├── src/                      # 22文件
│   ├── exceptions.py         # 8类统一异常
│   ├── scheduler/            # env + agent + parser + marl + multi_objective
│   ├── api/                  # tianyan_client + cqlib + mock + circuit_breaker
│   ├── quantum/              # annealing + annealing_loop
│   ├── visualization/        # FastAPI + Vue3 + Echarts
│   └── utils/                # helpers + Prometheus metrics
├── tests/                    # 14文件（含 benchmarks/）
├── scripts/                  # 6子目录 + cli.py
│   ├── training/ evaluation/ demo/ testing/ benchmarking/ reporting/
├── results/reports/          # B1实验数据（4份）
├── docs/                     # 团队文档
├── config/                   # config.yaml + .env.example
└── .github/                  # CI(4Job) + PR自动化 + Dependabot
```

### v1 技术提升新增模块
| 模块 | 用途 |
|------|------|
| `src/exceptions.py` | 8类异常（code + retryable语义） |
| `src/api/circuit_breaker.py` | 熔断器（CLOSED/OPEN/HALF_OPEN） |
| `src/utils/metrics.py` | Prometheus 7指标 |
| `scripts/cli.py` | Click统一入口（train/simulate/serve/demo） |
| `mypy.ini` | 8项严格配置 |
| `.pre-commit-config.yaml` | Git自动检查 |
| `.github/dependabot.yml` | 自动依赖更新 |
| 测试扩展 | 5→14文件，+82%代码量 |

---

## 比赛材料

| 材料 | 路径 | 状态 |
|------|------|------|
| 答辩PPT（15页） | `../答辩PPT_量子RL调度系统.pptx` | 已完成 |
| 技术白皮书（10章） | `../技术白皮书_量子RL调度系统_v2.docx` | 已完成 |
| 演示视频脚本（5分钟） | `演示视频分镜脚本.md` | 已完成 |
| 实验报告（4份） | `results/reports/` | 已完成 |

---

## 实验核心数据（引用 B1 报告）

| 实验 | 结果 |
|------|------|
| 8策略对比 | PPO 2814 vs FCFS 1462（+92.4%） |
| 消融D1-D5 | D4多机+86.3% > D1+92.4% > D5+6.4% > D2+2.1% |
| 压力4场景 | PPO综合最强；量子波动PPO +91.4% |
| 真机 | 32任务100%成功；Mock偏差<5% |

---

## 待完成

### P0 立即
- Track C: mypy 豁免 6→2 清理
- Track C: CI 覆盖率 40→60% 恢复

### P1 7月
- Mutation testing 基线
- pre-commit 迁移 ruff+bandit
- PPO 真机闭环（cqlib 注入调度循环）
- 演示视频录制
- 创建 8 个 GitHub Issues

### P2 8月
- 覆盖率 80%
- 8/15 代码冻结
- 材料终审定稿

---

## 关键路径

```
项目仓库: C:\Users\HZR\Desktop\揭榜挂帅擂台赛\quantum-rl-scheduler
Python: D:\tools\Python 3.12.9\python.exe
推送: PR流程（main分支保护）

提示词文件：
  TrackB_提示词_给TRAE.md  → B1-B4（已完成）
  TrackC_提示词_给TRAE.md  → C1-C3（待执行）
  GitHub_Issues_待创建.md  → 8个issue模板

重要提醒：
  - .env 不推送
  - TRAE 只写代码，不碰比赛材料
  - tianyan_s 可用门：H, CZ, M
```

---

**文档结束**
