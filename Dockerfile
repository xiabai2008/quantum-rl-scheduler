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
# 注意：src/visualization/frontend/ 现已落地正式 package.json + package-lock.json（Issue #177 修复）
COPY src/visualization/frontend/package*.json ./
# npm ci 默认即安装 devDependencies（vite / @vitejs/plugin-vue / typescript 等构建所需）
RUN npm ci

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

# 创建非 root 运行用户（最小权限原则，避免容器以 root 身份运行）
RUN useradd -m -s /bin/bash appuser

# 复制构建阶段安装的 Python 包到非 root 用户 home，避免依赖 root 的 /root/.local，
# 确保 appuser 可读取/导入依赖（Python 非 root 运行时自动启用 user site）。
COPY --from=builder /root/.local /home/appuser/.local
ENV PATH=/home/appuser/.local/bin:$PATH

# 复制项目代码
COPY . .

# 复制前端构建产物到 dist/ 目录
COPY --from=frontend-builder /frontend/dist /app/src/visualization/frontend/dist

# 创建运行时目录
RUN mkdir -p logs models results

# 复制入口脚本
COPY docker-entrypoint.sh /docker-entrypoint.sh
RUN chmod +x /docker-entrypoint.sh

# 将应用目录与用户目录归属改为 appuser，确保非 root 用户可写可读
# （logs/models/results 及 entrypoint 中 mkdir -p 的目录均位于 /app 下）
RUN chown -R appuser /app /home/appuser

# 暴露端口
# 8000: FastAPI Web 服务
# 6006: TensorBoard（可选）
EXPOSE 8000 6006

# 健康检查（使用无认证的 /health 存活探针，见 src/visualization/routes.py:193）
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 默认启动命令：entrypoint 脚本（后台仿真 + 前台 Web）
ENTRYPOINT ["/docker-entrypoint.sh"]

# 以非 root 用户运行（若挂载 ./logs ./models ./results 卷，须确保宿主目录对
# appuser(UID) 可写，否则启动后写 IO 会报错——详见 PR 描述部署须知）
USER appuser

# 备用启动命令（用于扩展）：
# - 训练模式: docker-compose run --rm web python scripts/cli.py train --timesteps 100000
# - 快速训练: docker-compose run --rm web python scripts/cli.py quick-train
# - 仿真模式: docker-compose run --rm web python scripts/cli.py simulate --num-tasks 200 --strategies all
