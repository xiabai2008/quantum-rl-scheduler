#!/usr/bin/env python
"""
量子RL调度系统 - 一键启动脚本

功能：
1. 自动检查Python版本和依赖
2. 检查deliverable_models模型文件是否存在
3. 自动启动uvicorn服务器
4. 自动打开浏览器访问Dashboard
5. 打印访问地址和操作指引

使用方法：
    python scripts/demo_one_click.py
    或双击运行（如果.py关联了Python）
"""

import os
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path


def print_banner():
    """打印启动横幅"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║            量子RL智能调度系统 - Demo 一键启动                ║
║                                                              ║
║     AI赋能量子计算 · PPO强化学习 · 8策略对比                 ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def check_python_version():
    """检查Python版本"""
    print("[1/5] 检查Python版本...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        print(f"  ❌ Python版本过低: {version.major}.{version.minor}")
        print("     需要Python 3.9或更高版本")
        return False
    print(f"  ✅ Python {version.major}.{version.minor}.{version.micro}")
    return True


def check_dependencies():
    """检查关键依赖是否已安装"""
    print("\n[2/5] 检查依赖包...")
    required = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "stable_baselines3": "stable-baselines3",
        "gymnasium": "gymnasium",
        "numpy": "numpy",
        "pydantic": "pydantic",
    }
    missing = []
    for import_name, pkg_name in required.items():
        try:
            __import__(import_name)
            print(f"  ✅ {pkg_name}")
        except ImportError:
            print(f"  ❌ {pkg_name} 未安装")
            missing.append(pkg_name)

    if missing:
        print(f"\n  ⚠️  缺少 {len(missing)} 个依赖包，正在自动安装...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", *missing],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            print("  ✅ 依赖安装完成")
        except subprocess.CalledProcessError as e:
            print(f"  ❌ 自动安装失败: {e}")
            print(f"     请手动运行: pip install {' '.join(missing)}")
            return False
    return True


def check_model():
    """检查模型文件"""
    print("\n[3/5] 检查PPO模型文件...")
    project_root = Path(__file__).parent.parent
    model_dir = project_root / "deliverable_models"
    model_path = model_dir / "ppo_best_model_14dim.zip"

    if model_path.exists():
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ 找到模型: {model_path.name} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ⚠️  未找到模型: {model_path}")
        # 搜索其他可能的模型文件
        models_dir = project_root / "models"
        if models_dir.exists():
            zips = list(models_dir.glob("**/ppo*14dim*.zip"))
            if zips:
                print(f"  ✅ 找到备选模型: {zips[0].name}")
                return True
        print("     系统将在首次请求时懒加载模型，可能需要等待...")
        return True  # 不阻止启动，懒加载会处理


def find_free_port(start_port=8000, max_tries=10):
    """寻找可用端口"""
    import socket

    for port in range(start_port, start_port + max_tries):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(("0.0.0.0", port))
                return port
        except OSError:
            continue
    return start_port


def start_server(port):
    """启动uvicorn服务器"""
    print(f"\n[4/5] 启动服务器 (端口 {port})...")
    project_root = Path(__file__).parent.parent
    os.chdir(str(project_root))

    # 设置环境变量
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root) + os.pathsep + env.get("PYTHONPATH", "")

    # 启动uvicorn
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "src.visualization.app:app",
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "--log-level",
        "warning",
    ]

    print(f"  执行: {' '.join(cmd)}")
    print("  服务器启动中，首次加载模型可能需要10-30秒...")

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return proc
    except Exception as e:
        print(f"  ❌ 启动失败: {e}")
        return None


def wait_for_server(port, timeout=60):
    """等待服务器启动"""
    import urllib.error
    import urllib.request

    url = f"http://127.0.0.1:{port}/health"
    start = time.time()

    while time.time() - start < timeout:
        try:
            req = urllib.request.urlopen(url, timeout=3)
            if req.getcode() == 200:
                return True
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass
        time.sleep(1)

        # 检查进程是否还在运行
        if hasattr(wait_for_server, "proc") and wait_for_server.proc.poll() is not None:
            return False

    return False


def open_browser_delayed(port, delay=5):
    """延迟打开浏览器，等待服务器就绪"""

    def _open():
        time.sleep(delay)
        url = f"http://127.0.0.1:{port}"
        print(f"\n[5/5] 自动打开浏览器: {url}")
        webbrowser.open(url)

    t = threading.Thread(target=_open, daemon=True)
    t.start()


def print_usage_info(port):
    """打印使用指引"""
    info = f"""
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│  🎉 量子RL调度系统已启动！                                   │
│                                                              │
│  📊 Dashboard 地址:                                          │
│     http://127.0.0.1:{port}                                   │
│     http://localhost:{port}                                   │
│                                                              │
│  📋 功能面板:                                                │
│     • PPO vs FCFS 实时策略对比（自动对战）                    │
│     • 资源利用率趋势图                                        │
│     • 最近决策日志                                            │
│     • 决策可解释性（特征贡献度）                              │
│     • 8种策略性能排名                                        │
│     • 任务队列管理                                            │
│     • 调度策略切换                                            │
│                                                              │
│  ⌨️  操作:                                                    │
│     • 按 Ctrl+C 停止服务器                                    │
│     • 在右侧控制面板提交任务观察调度效果                      │
│     • 切换不同调度策略对比性能                                │
│                                                              │
│  📌 核心数据:                                                │
│     PPO vs FCFS: +88.3% (N=250, p<0.001)                    │
│     量子利用率: +48.9% (33.6% → 50%)                         │
│     退火模块: 探索性方向（训练开销+74.5%, p=0.190不显著）     │
│                                                              │
└──────────────────────────────────────────────────────────────┘
    """
    print(info)


def main():
    print_banner()

    # 检查环境
    if not check_python_version():
        input("\n按回车键退出...")
        sys.exit(1)

    if not check_dependencies():
        input("\n按回车键退出...")
        sys.exit(1)

    check_model()

    # 找端口
    port = find_free_port(8000)
    if port != 8000:
        print(f"\n  ℹ️  端口8000被占用，使用端口 {port}")

    # 启动服务器
    proc = start_server(port)
    if proc is None:
        input("\n按回车键退出...")
        sys.exit(1)

    wait_for_server.proc = proc

    # 等待服务器就绪
    print("  等待服务器响应...", end="", flush=True)
    ready = wait_for_server(port, timeout=45)
    if ready:
        print(" ✅")
    else:
        print(" ⚠️")
        print("  服务器可能仍在加载模型，浏览器将在5秒后打开...")

    # 打印信息并打开浏览器
    print_usage_info(port)
    open_browser_delayed(port, delay=3 if ready else 8)

    # 等待用户中断
    try:
        while True:
            line = proc.stdout.readline()
            if line:
                # 只打印错误和关键信息
                line_lower = line.lower()
                if "error" in line_lower or "warning" in line_lower or "started" in line_lower:
                    print(f"  [server] {line.rstrip()}")
            if proc.poll() is not None:
                print("\n  ❌ 服务器进程已退出")
                remaining = proc.stdout.read()
                if remaining:
                    print(remaining)
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n\n  🛑 收到停止信号，正在关闭服务器...")
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("  ✅ 服务器已停止")

    print("\n感谢使用量子RL智能调度系统！")


if __name__ == "__main__":
    main()
