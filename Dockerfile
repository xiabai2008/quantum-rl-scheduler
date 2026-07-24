# =============================================================================
# 量子RL调度系统 - 多阶段 Dockerfile（v2：含前端构建 + 依赖瘦身）
# =============================================================================
#
# 构建镜像：
#   docker build -t quantum-rl-scheduler:latest .
#
# 运行容器（一键复现，推荐）：
#   docker compose up
#
# 单独运行 Web 服务（不跑仿真）：
#   docker run -p 8000:8000 -p 6006:6006 quantum-rl-scheduler:latest
#
# =============================================================================

# ---------- 阶段 0：构建 Vue3 前端 ----------
FROM node:20-slim AS frontend-builder

WORKDIR /frontend

# 先复制 package 文件以利用 Docker 层缓存
COPY src/visualization/frontend/package*.json ./
RUN npm ci --production=false

# 复制前端源码并构建
COPY src/visualization/frontend/ ./
RUN npm run build

# ---------- 阶段 1：构建 Python 依赖 ----------
FROM python:3.11-slim AS builder

WORKDIR /app

# 安装系统构建依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖清单并安装到用户目录（便于多阶段复制）
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# ---------- 阶段 2：运行时镜像 ----------
FROM python:3.11-slim

# 环境变量
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装运行时系统依赖（OpenMP for numpy, GLib for matplotlib）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 复制构建阶段安装的 Python 包
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# 复制项目代码
COPY . .

# 复制前端构建产物到 dist/ 目录
COPY --from=frontend-builder /frontend/dist /app/src/visualization/frontend/dist

# 创建运行时目录
RUN mkdir -p logs models results

# 复制入口脚本
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# 暴露端口
# 8000: FastAPI Web 服务
# 6006: TensorBoard（可选）
EXPOSE 8000 6006

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/api/status || exit 1

# 默认启动命令：entrypoint 脚本（后台仿真 + 前台 Web）
ENTRYPOINT ["/docker-entrypoint.sh"]

# 备用启动命令（用于扩展）：
# - 训练模式: docker-compose run --rm web python scripts/cli.py train --timesteps 100000
# - 快速训练: docker-compose run --rm web python scripts/cli.py quick-train
# - 仿真模式: docker-compose run --rm web python scripts/cli.py simulate --num-tasks 200 --strategies all
