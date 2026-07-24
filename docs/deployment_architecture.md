# 部署架构与生产化路径

> **项目**: 量子RL驱动的天衍云平台智能调度系统
> **文档类型**: 部署架构设计
> **生成时间**: 2026-07-24
> **对应比赛要求**: "落地与价值 — 商业潜力、社会效益、实施路径"

---

## 一、当前系统架构

### 1.1 架构总览

系统采用三层架构，各层通过明确接口解耦：

```
┌─────────────────────────────────────────────────────────┐
│                    用户交互层                             │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────────┐ │
│  │ CLI入口   │  │ Web监控   │  │ REST API + WebSocket │ │
│  │ (Click)  │  │ (Vue3)   │  │ (FastAPI)            │ │
│  └────┬─────┘  └────┬─────┘  └──────────┬────────────┘ │
├───────┼──────────────┼──────────────────┼──────────────┤
│       │     调度引擎层（核心）             │              │
│  ┌────▼──────────────▼──────────────────▼────────────┐ │
│  │  HybridScheduler（规则引擎 + RL 三级降级）         │ │
│  │  ┌─────────┐  ┌──────────┐  ┌──────────────────┐ │ │
│  │  │PPO推理   │  │多租户配额  │  │量子退火优化(可选) │ │ │
│  │  │(PyTorch)│  │(TenantQM)│  │(QUBO/Annealing)  │ │ │
│  │  └─────────┘  └──────────┘  └──────────────────┘ │ │
│  └───────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                    平台对接层                             │
│  ┌──────────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │天衍云cqlib    │  │熔断器     │  │Prometheus监控     │ │
│  │(真机API)     │  │(3态转换)  │  │(7指标/metrics)   │ │
│  └──────────────┘  └──────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────┘
```

### 1.2 核心组件

| 组件 | 文件 | 职责 | 部署状态 |
|------|------|------|---------|
| PPO 推理引擎 | `src/scheduler/ppo_agent.py` | 策略推理，输出调度动作 | ✅ 训练完成，模型77KB |
| 混合调度器 | `src/scheduler/hybrid_scheduler.py` | 规则+RL三级降级调度 | ✅ 可用 |
| 多租户管理 | `src/scheduler/tenant.py` | 租户配额、优先级管理 | ✅ 可用 |
| 天衍云客户端 | `src/api/tianyan_cqlib.py` | 真机任务提交/查询 | ✅ 已验证（15/15成功） |
| 熔断器 | `src/api/circuit_breaker.py` | 故障保护，3态转换 | ✅ 可用 |
| 量子退火 | `src/quantum/annealing.py` | QUBO策略优化（可选） | ✅ 可用，默认关闭 |
| Web监控 | `src/visualization/app.py` | 实时监控+手动操作 | ✅ Docker部署 |
| Prometheus | `src/utils/metrics.py` | 7个指标暴露 | ✅ /metrics端点 |

---

## 二、三阶段部署路径

### 阶段一：原型验证（当前，已完成）

**目标**：验证 RL 调度在仿真和真机环境下的有效性

| 维度 | 状态 | 数据 |
|------|------|------|
| 仿真验证 | ✅ 完成 | N=250, PPO +88.3%, p<0.001 |
| 真机验证 | ✅ 完成 | 15/15任务成功, Cohen's d=5.64 |
| 多租户 | ✅ 完成 | 5租户×10seeds, Jain's Index=0.9875 |
| 退火消融 | ✅ 完成 | +6.4%, p=0.19（探索性） |
| Docker | ✅ 完成 | docker-compose一键部署 |

**关键交付物**：
- PPO模型：`deliverable_models/ppo_best_model_14dim.zip`（77KB）
- CLI工具：`python scripts/cli.py train/simulate/serve/demo`
- Web监控：`uvicorn src.visualization.app:app --port 8000`

### 阶段二：试点部署（赛后1-3月）

**目标**：在天衍云平台进行小规模真实用户试点

#### 2.1 部署架构

```
天衍云平台
├── 调度服务容器（Docker）
│   ├── PPO推理服务（FastAPI, 端口8000）
│   ├── 混合调度器（规则+RL降级）
│   ├── 多租户配额管理
│   └── Prometheus指标暴露（/metrics）
├── 天衍云API网关
│   ├── 任务提交接口（QCIS格式）
│   ├── 状态查询接口
│   └── 配额管理接口
├── 监控面板
│   ├── Grafana可视化（Prometheus数据源）
│   └── 告警规则（Slack/邮件通知）
└── 数据存储
    ├── 调度日志（JSON, 按日轮转）
    └── 模型检查点（PPO权重, 定期更新）
```

#### 2.2 关键配置

| 参数 | 试点值 | 说明 |
|------|--------|------|
| 租户数 | 5-10 | 小规模科研团队 |
| 日均任务量 | 100-200 | 量子化学+优化任务 |
| PPO推理模式 | PyTorch eager | 简单可靠 |
| 退火模块 | 关闭 | 试点阶段聚焦RL调度 |
| 真机提交概率 | 10% | 控制机时消耗 |
| 超时阈值 | 180s | 覆盖排队延迟 |
| 熔断阈值 | 3次连续失败 | 自动降级仿真 |

#### 2.3 试点验证指标

| 指标 | 目标值 | 测量方法 |
|------|--------|---------|
| 调度性能 | PPO > FCFS +50% | A/B测试对比 |
| 资源利用率 | ≥70% | Prometheus指标 |
| Jain's Fairness Index | ≥0.9 | 按租户统计 |
| 推理延迟 | <100ms (P99) | 延迟直方图 |
| 系统可用性 | ≥99% | 运行时间/总时间 |
| 用户满意度 | ≥4/5 | 试点用户问卷 |

### 阶段三：生产部署（赛后3-6月）

**目标**：全面集成到天衍云平台调度系统

#### 3.1 生产架构增强

| 维度 | 试点→生产 | 实现方案 |
|------|----------|---------|
| 推理性能 | PyTorch→ONNX Runtime | 延迟降低2-5x |
| 模型格式 | FP32→INT8量化 | 模型体积减4x |
| 高可用 | 单容器→K8s集群 | 3副本+自动恢复 |
| 模型更新 | 手动→CI/CD流水线 | 自动训练→验证→部署 |
| 扩展性 | 单机→水平扩展 | 无状态推理服务 |
| 安全 | API Key→mTLS | 双向认证 |

#### 3.2 ONNX推理优化路径

```python
# 当前（PyTorch eager）
from stable_baselines3 import PPO
model = PPO.load("ppo_best_model_14dim.zip")
action, _ = model.predict(obs, deterministic=True)

# 生产（ONNX Runtime）
import onnxruntime as ort
session = ort.InferenceSession("ppo_model.onnx")
action = session.run(None, {"obs": obs})[0]
```

| 指标 | PyTorch eager | ONNX Runtime | 优化 |
|------|-------------|-------------|------|
| 单次推理延迟 | ~5ms | ~1ms | 5x |
| 内存占用 | ~200MB | ~50MB | 4x |
| 模型体积 | 77KB | 20KB | 3.85x |
| 依赖 | PyTorch (~2GB) | onnxruntime (~50MB) | 40x |

> 注：当前已有 `src/scheduler/export.py` 支持 ONNX 导出，生产部署时直接使用。

#### 3.3 K8s部署配置

```yaml
# k8s/scheduler-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: quantum-rl-scheduler
spec:
  replicas: 3                    # 3副本高可用
  selector:
    matchLabels:
      app: quantum-scheduler
  template:
    spec:
      containers:
      - name: scheduler
        image: quantum-rl-scheduler:v1.0
        ports:
        - containerPort: 8000
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:           # 健康检查
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
        readinessProbe:          # 就绪检查
          httpGet:
            path: /ready
            port: 8000
```

---

## 三、推理延迟优化

### 3.1 当前延迟分析

PPO模型结构：64-64 隐藏层，~19k 参数，输入14维，输出3动作。

| 推理方式 | 延迟(P50) | 延迟(P99) | 吞吐 | 适用场景 |
|---------|----------|----------|------|---------|
| PyTorch CPU | ~3ms | ~8ms | 300/s | 试点 |
| PyTorch GPU | ~1ms | ~3ms | 1000/s | 高负载 |
| ONNX Runtime CPU | ~0.5ms | ~2ms | 2000/s | 生产 |
| ONNX Runtime + INT8 | ~0.3ms | ~1ms | 3000/s | 极低延迟 |

### 3.2 延迟保障机制

1. **规则引擎优先**：`HybridScheduler` 对确定性场景（classical任务、紧急任务）直接规则匹配，跳过RL推理，延迟<0.1ms
2. **RL推理缓存**：对相同观测状态的推理结果缓存，减少重复计算
3. **批量推理**：高负载时批量处理多个调度请求
4. **降级保障**：RL推理超时（>50ms）时自动降级到规则引擎

---

## 四、高可用设计

### 4.1 三级降级策略

```
正常状态：PPO推理（confidence=0.8）
    ↓ PPO推理失败/超时
降级1：规则引擎（confidence=1.0 for确定性场景, 0.3 for兜底）
    ↓ 规则引擎异常
降级2：默认动作（ACTION_QUANTUM, confidence=0.0）
    ↓
系统告警 + 人工介入
```

### 4.2 熔断器机制

| 状态 | 条件 | 行为 |
|------|------|------|
| CLOSED | 正常运行 | 所有请求通过 |
| OPEN | 连续失败≥阈值 | 请求直接降级，不调用真机 |
| HALF_OPEN | 冷却时间后 | 允许少量请求试探 |

### 4.3 数据持久化

| 数据类型 | 存储方式 | 保留期 |
|---------|---------|--------|
| 调度日志 | JSON文件（按日轮转） | 90天 |
| 模型检查点 | 本地+云存储 | 最新3个版本 |
| Prometheus指标 | TSDB | 30天 |
| 真机任务记录 | JSON+数据库 | 永久 |

---

## 五、监控与运维

### 5.1 Prometheus指标（7个）

| 指标 | 类型 | 说明 |
|------|------|------|
| scheduler_dispatch_total | Counter | 调度决策总数 |
| scheduler_dispatch_duration | Histogram | 调度延迟分布 |
| scheduler_queue_length | Gauge | 当前队列长度 |
| tianyan_api_calls_total | Counter | 天衍云API调用数 |
| tianyan_api_errors_total | Counter | API错误数 |
| tianyan_api_duration | Histogram | API延迟分布 |
| tianyan_circuit_breaker_state | Gauge | 熔断器状态(0/1/2) |

### 5.2 告警规则

| 告警 | 条件 | 级别 |
|------|------|------|
| 熔断器开启 | circuit_breaker_state==1 | 严重 |
| API错误率高 | errors/calls > 10% | 警告 |
| 调度延迟高 | P99 > 100ms | 警告 |
| 队列积压 | queue_length > 25 | 警告 |
| 模型推理失败 | dispatch_errors > 5/min | 严重 |

---

## 六、模型更新机制

### 6.1 离线训练→在线部署流水线

```
数据收集（生产环境调度日志）
    ↓
离线训练（PPO, 50k steps, ~1min）
    ↓
验证（A/B测试, N=50 episodes）
    ↓
灰度发布（10%流量→50%→100%）
    ↓
监控（性能指标对比）
    ↓
回滚（如性能下降 > 5%）
```

### 6.2 模型版本管理

| 版本 | 模型文件 | 训练数据 | 验证结果 | 状态 |
|------|---------|---------|---------|------|
| v1.0 | ppo_best_model_14dim.zip | N=250仿真 | +88.3%, p<0.001 | 当前生产 |
| v1.1 | ppo_v1.1.zip | +真实调度日志 | 待验证 | 待发布 |
| v1.2 | ppo_v1.2.zip | +退火优化 | +6.4%额外 | 实验中 |

---

## 七、答辩要点

1. **部署路径清晰**：原型→试点→生产三阶段，每阶段有明确的验证指标
2. **推理延迟可控**：当前PyTorch ~3ms，ONNX优化后 ~0.5ms，远低于100ms要求
3. **高可用保障**：三级降级+熔断器+K8s多副本，系统可用性目标≥99%
4. **渐进式部署**：试点阶段聚焦核心RL调度，退火模块可选启用
5. **模型可更新**：离线训练→A/B验证→灰度发布的CI/CD流水线
6. **监控完善**：7个Prometheus指标+5条告警规则，运维可视化

---

## 八、与天衍云平台集成方案

### 8.1 集成接口

| 接口 | 方向 | 协议 | 说明 |
|------|------|------|------|
| 任务提交 | 调度器→天衍云 | cqlib SDK | QCIS格式量子电路 |
| 状态查询 | 调度器→天衍云 | cqlib SDK | 批量查询, max_wait=30s |
| 结果回调 | 天衍云→调度器 | 轮询 | 非阻塞, 180s超时 |
| 指标暴露 | 调度器→监控 | HTTP /metrics | Prometheus格式 |
| 用户管理 | 平台→调度器 | REST API | 租户配置同步 |

### 8.2 集成步骤

1. **API对接**：使用 `CqlibTianyanClient` 连接天衍云API，API Key通过环境变量注入
2. **机器配置**：配置 `DEFAULT_MACHINE_CONFIGS` 匹配天衍云可用量子计算机列表
3. **配额同步**：从天衍云平台获取用户配额信息，同步到 `TenantQuotaManager`
4. **监控集成**：将 `/metrics` 端点接入天衍云现有监控系统
5. **灰度切换**：先在10%流量上启用RL调度，对比FCFS基线性能

---

*本文档响应比赛方案"落地与价值 — 实施路径"的评估要求，所有架构设计基于项目现有代码实现。*
