# 部署指南（Issue #214）

> 本文档覆盖量子RL调度系统的生产部署，包括环境分级配置、健康检查、Prometheus 告警规则与 Docker 部署流程。

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

## 4. Docker 部署

详见 [docker-deploy.md](docker-deploy.md)，简要流程：

```bash
# 1. 配置环境变量
cp .env.example .env
# 编辑 .env 注入 TIANYAN_API_KEY / TIANYAN_API_SECRET

# 2. 设置 APP_ENV
export APP_ENV=prod

# 3. 构建并启动
docker-compose up -d --build

# 4. 健康检查
curl http://localhost:8000/health
curl http://localhost:8000/ready
curl http://localhost:8000/metrics
```

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
