# 部署指南（Issue #214）

> 本文档覆盖量子RL调度系统的生产部署，包括环境分级配置、健康检查、Prometheus 告警规则与 Docker 部署流程。
>
> **相关文档**：
> - 架构设计与生产化路径：见 [deployment_architecture.md](deployment_architecture.md)（Issue #210 文档碎片化整理：本文档聚焦实操部署，架构设计见另一份文档）

## 1. 环境分级配置

项目支持通过 `APP_ENV` 环境变量切换 dev / prod 配置：

| `APP_ENV` | 加载的配置文件 | 适用场景 |
|---|---|---|
| 未设置 | `config/config.yaml` | 默认（开发 + mock） |
| `dev` | `config/config.dev.yaml` | 开发环境（debug + mock） |
| `staging` | `config/config.staging.yaml` | 预发布（需自行创建） |
| `prod` | `config/config.prod.yaml` | 生产环境（真机 + INFO 日志） |

### 1.1 关键差异（dev vs prod）

| 维度 | dev | prod |
|---|---|---|
| `tianyan.mock_mode` | `true` | `false` |
| `web.debug` | `true` | `false` |
| `system.log_level` | `DEBUG` | `INFO` |
| `quantum.shots` | `1024` | `2048` |
| `scheduler.batch_size` | `64` | `128` |
| `scheduler.replay_buffer_size` | `10000` | `50000` |
| `system.max_queue_size` | `100` | `500` |
| `cache.db` | `0` | `1`（独立 Redis DB） |

### 1.2 使用示例

```bash
# 开发环境
export APP_ENV=dev
python scripts/cli.py serve --port 8000

# 生产环境（需先注入 TIANYAN_API_KEY / TIANYAN_API_SECRET）
export APP_ENV=prod
export TIANYAN_API_KEY="your_key"
export TIANYAN_API_SECRET="your_secret"
python scripts/cli.py serve --port 8000
```

### 1.3 配置优先级

`load_settings()` 按以下优先级合并（高 → 低）：

1. 环境变量（`os.environ`）
2. `.env` 文件
3. `config/config.{APP_ENV}.yaml`
4. `Settings` dataclass 默认值

详见 [src/config/settings.py](../src/config/settings.py) 中 `load_settings` 函数。

## 2. 健康检查端点

FastAPI 暴露两个健康检查端点，供 Kubernetes / Docker / 负载均衡器使用：

### 2.1 `/health` — 存活探针（Liveness）

```bash
curl http://localhost:8000/health
# {"status":"alive"}
```

只要进程在运行就返回 200，**不依赖任何外部资源**，避免因外部抖动导致进程被重启。

### 2.2 `/ready` — 就绪探针（Readiness）

```bash
curl http://localhost:8000/ready
```

返回各关键依赖的就绪状态：

```json
{
  "ready": true,
  "checks": {
    "app": {"ok": true},
    "metrics": {"ok": true},
    "ppo_model": {"ok": true, "required": false},
    "quota_tracker": {"ok": false, "required": false}
  },
  "required_ok": true,
  "timestamp": "2026-07-21T10:30:00.000000"
}
```

`required: false` 的检查失败不会让 `ready=false`，仅作为信息暴露。

### 2.3 Kubernetes 探针配置

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 10
  periodSeconds: 30
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /ready
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
  failureThreshold: 3
```

### 2.4 Docker Compose 健康检查

```yaml
services:
  web:
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 30s
```

## 3. Prometheus 告警规则

告警规则文件位于 [config/alerts.yml](../config/alerts.yml)，共 7 条规则，覆盖 6 个维度：

| 规则名 | 维度 | 严重度 | 触发条件 |
|---|---|---|---|
| `SchedulerQueueBacklog` | 调度 | warning | 队列长度 > 50 持续 2 分钟 |
| `SchedulerQueueCriticalBacklog` | 调度 | critical | 队列长度 > 200 持续 5 分钟 |
| `TianyanApiErrorRateHigh` | API | warning | API 错误率 > 10% 持续 5 分钟 |
| `TianyanApiErrorRateCritical` | API | critical | API 错误率 > 50% 持续 2 分钟 |
| `CircuitBreakerOpen` | 韧性 | critical | 熔断器 OPEN 持续 1 分钟 |
| `TianyanCircuitBreakerOpen` | 韧性 | critical | 天衍云熔断器 OPEN 持续 1 分钟 |
| `TaskWaitTimeHigh` | 调度 | warning | P95 等待时间 > 120s 持续 5 分钟 |
| `QubitUtilizationLow` | 资源 | warning | 利用率 < 20% 持续 10 分钟 |
| `QubitUtilizationSaturated` | 资源 | warning | 利用率 > 95% 持续 5 分钟 |
| `ServiceDown` | 可用性 | critical | Prometheus 抓取失败持续 2 分钟 |
| `ManyAlertsFiring` | 运维 | warning | firing 告警 > 20 条持续 10 分钟 |

### 3.1 加载告警规则

在 `config/prometheus.yml` 中添加 `rule_files`：

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - alerts.yml

scrape_configs:
  - job_name: 'qrls-scheduler'
    static_configs:
      - targets: ['scheduler:9090']
    metrics_path: '/metrics'
```

### 3.2 告警接收（Alertmanager）

将告警路由到不同接收器（邮件 / 钉钉 / PagerDuty），按 `severity` 分级：

- `critical` → PagerDuty / 电话
- `warning` → 邮件 / 工单


---

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
│  │  │PPO推理   │  │多租户配额  │  │量子启发式退火(可选)│ │ │
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
| 量子启发式退火 | `src/quantum/annealing.py` | QUBO策略优化（经典模拟退火，可选） | ✅ 可用，默认关闭 |
| Web监控 | `src/visualization/app.py` | 实时监控+手动操作 | ✅ Docker部署 |
| Prometheus | `src/utils/metrics.py` | 7个指标暴露 | ✅ /metrics端点 |

---

## 二、三阶段部署路径

### 阶段一：原型验证（当前，已完成）

**目标**：验证 RL 调度在仿真和真机环境下的有效性

| 维度 | 状态 | 数据 |
|------|------|------|
| 仿真验证 | ✅ 完成 | N=250, PPO +20.2%, p<0.001（16维交付模型） |
| 真机验证 | ✅ 完成 | 15/15任务成功, Cohen's d=5.33（效应量异常大，小样本探索性结果，需进一步验证） |
| 多租户 | ✅ 完成 | 5租户×10seeds, Jain's Index=0.9875 |
| 退火消融 | ✅ 完成 | +6.4%, p=0.9430（20seeds权威值，探索性；旧5seed的p=0.190已废弃） |
| Docker | ✅ 完成 | docker-compose一键部署 |
| 公平性特性 | ✅ 完成 | compute_fairness_penalty（阈值0.3, 因子2.0）+ 可选第17维Jain公平性观测 |
| 编译环境可配置 | ✅ 完成 | 自定义物理比特数和耦合图，天衍-287预设10x11网格拓扑（PR #616） |

**关键交付物**：
- PPO模型：`deliverable_models/ppo_best_model_16dim.zip`（77KB）
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
model = PPO.load("ppo_best_model_16dim.zip")
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

PPO模型结构：64-64 隐藏层，~19k 参数，输入16维，输出3动作。

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
| v1.0 | ppo_best_model_16dim.zip | N=250仿真 | +20.2%, p<0.001（16维交付模型） | 当前生产 |
| v1.1 | ppo_v1.1.zip | +真实调度日志 | 待验证 | 待发布 |
| v1.2 | ppo_v1.2.zip | +退火优化 | 退火20seed权威方向 -5.6%（p=0.9430 不显著，旧 +6.4% 已废弃），不额外计入 | 探索性 |

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

---

## Docker 容器化部署（完整说明）

# Docker 部署说明

## 快速开始

### 1. 构建并启动服务

```bash
# 构建镜像
docker build -t quantum-rl-scheduler:latest .

# 启动服务
docker-compose up -d
```

### 2. 访问服务

- **Web 监控界面**: http://localhost:8000
- **API 文档**: http://localhost:8000/docs
- **TensorBoard** (可选): http://localhost:6006

```bash
# 启动包含 TensorBoard 的完整服务
docker-compose --profile monitoring up -d
```

## 常用命令

```bash
# 查看日志
docker-compose logs -f web

# 停止服务
docker-compose down

# 重新构建并启动
docker-compose up -d --build

# 进入容器
docker exec -it quantum-rl-web bash

# 查看容器状态
docker-compose ps
```

## 数据持久化

日志和模型文件会持久化到宿主机的 `logs/` 和 `models/` 目录。

```bash
# 查看训练日志
ls -la logs/

# 查看保存的模型
ls -la models/
```

## 生产环境部署

对于生产环境，建议：

1. 启动 Redis 缓存：
```bash
docker-compose --profile production up -d
```

2. 配置环境变量：
```bash
# 创建 .env 文件
echo "LOG_LEVEL=INFO" > .env
echo "REDIS_URL=redis://redis:6379" >> .env
```

3. 使用 Nginx 反向代理

## GPU 支持（可选）

如果需要在 Docker 中使用 GPU：

```bash
# NVIDIA GPU
docker build -t quantum-rl-scheduler:latest . \
  --build-arg CUDA_VERSION=11.8

# 运行带 GPU 支持的容器
docker run --gpus all -p 8000:8000 quantum-rl-scheduler:latest
```

## 故障排除

### 端口被占用
```bash
# 查看端口占用
netstat -tulpn | grep 8000

# 修改 docker-compose.yml 中的端口映射
```

### 内存不足
```bash
# 增加 Docker 内存限制
# Docker Desktop -> Settings -> Resources -> Memory
```

### 构建失败
```bash
# 清理 Docker 缓存
docker builder prune -a

# 重新构建
docker-compose build --no-cache
```

---

## 5. 生产部署检查清单

部署前请逐项确认：

- [ ] `APP_ENV=prod` 已设置
- [ ] `TIANYAN_API_KEY` / `TIANYAN_API_SECRET` 已通过环境变量注入（**不要**写入代码或 git）
- [ ] `config/config.prod.yaml` 中 `tianyan.mock_mode: false`
- [ ] `config/config.prod.yaml` 中 `web.debug: false`
- [ ] 日志目录 `logs/` 已挂载到持久化卷
- [ ] 模型目录 `deliverable_models/` 已挂载到持久化卷
- [ ] `/health` 返回 200
- [ ] `/ready` 返回 `ready: true`
- [ ] `/metrics` 可被 Prometheus 抓取
- [ ] `config/alerts.yml` 已通过 `rule_files` 加载
- [ ] 关键告警（critical）已路由到值班手机 / PagerDuty
