#!/usr/bin/env python
"""
量子RL调度系统 — 统一命令行入口
Quantum RL Scheduler - Unified CLI

Usage:
    qs train [options]          # 训练智能体
    qs simulate [options]       # 运行仿真对比
    qs serve [options]          # 启动 Web 服务
    qs demo [options]           # 一键演示

示例:
    python scripts/cli.py train --timesteps 100000
    python scripts/cli.py simulate --episodes 50
    python scripts/cli.py serve --port 8000
    python scripts/cli.py demo
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import click


@click.group(name="qs")
@click.version_option(version="1.0.0", prog_name="Quantum RL Scheduler")
def cli():
    """量子RL调度系统 — 统一命令行入口"""
    pass


@cli.command(name="train")
@click.option("--timesteps", type=int, default=100000, help="训练总步数")
@click.option("--resume", is_flag=True, help="从检查点恢复训练")
@click.option("--checkpoint", type=str, default=None, help="检查点路径")
@click.option("--patience", type=int, default=0, help="早停耐心值")
@click.option("--eval-freq", type=int, default=1000, help="评估频率")
@click.option("--eval-episodes", type=int, default=10, help="每次评估的 episode 数")
@click.option("--save-path", type=str, default="./models/", help="模型保存路径")
@click.option("--log-dir", type=str, default="./logs/", help="TensorBoard 日志目录")
@click.option("--seed", type=int, default=None, help="随机种子")
@click.option("--seeds", type=int, multiple=True, default=None, help="多种子对比实验")
@click.option("--max-qubits", type=int, default=287, help="最大量子比特数")
@click.option("--max-steps", type=int, default=500, help="每个 episode 的最大步数")
@click.option("--learning-rate", type=float, default=3e-4, help="学习率")
@click.option("--verbose", type=int, default=1, help="详细程度 0-2")
def train(
    timesteps,
    resume,
    checkpoint,
    patience,
    eval_freq,
    eval_episodes,
    save_path,
    log_dir,
    seed,
    seeds,
    max_qubits,
    max_steps,
    learning_rate,
    verbose,
):
    """训练智能体"""
    import argparse

    from scripts.training.train_agent import main

    args_dict = {
        "timesteps": timesteps,
        "resume": resume,
        "checkpoint": checkpoint,
        "patience": patience,
        "eval_freq": eval_freq,
        "eval_episodes": eval_episodes,
        "save_path": save_path,
        "log_dir": log_dir,
        "seed": seed,
        "seeds": list(seeds) if seeds else None,
        "max_qubits": max_qubits,
        "max_steps": max_steps,
        "learning_rate": learning_rate,
        "verbose": verbose,
    }

    args = argparse.Namespace(**args_dict)
    main.__wrapped__(args) if hasattr(main, "__wrapped__") else main()


@cli.command(name="simulate")
@click.option("--episodes", type=int, default=100, help="仿真 episode 数")
@click.option("--tasks-per-episode", type=int, default=100, help="每个 episode 的任务数")
@click.option("--model-path", type=str, default=None, help="训练好的 DQN 模型路径")
@click.option("--ppo-model-path", type=str, default=None, help="训练好的 PPO 模型路径")
@click.option("--output-dir", type=str, default="./results/", help="结果输出目录")
@click.option("--verbose", is_flag=True, help="打印详细日志")
@click.option("--real-prob", type=float, default=0.0, help="真机抽样概率")
@click.option("--real-machine", type=str, default="tianyan_s", help="真机抽样目标机器名")
def simulate(
    episodes,
    tasks_per_episode,
    model_path,
    ppo_model_path,
    output_dir,
    verbose,
    real_prob,
    real_machine,
):
    """运行仿真对比实验"""
    from scripts.evaluation.run_simulation import run_simulation

    run_simulation(
        episodes=episodes,
        tasks_per_episode=tasks_per_episode,
        model_path=model_path,
        ppo_model_path=ppo_model_path,
        output_dir=output_dir,
        verbose=verbose,
        real_prob=real_prob,
        real_machine=real_machine,
    )


@cli.command(name="serve")
@click.option("--host", type=str, default="0.0.0.0", help="服务绑定地址")
@click.option("--port", type=int, default=8000, help="服务端口")
@click.option("--reload", is_flag=True, help="开发模式：代码修改自动重载")
def serve(host, port, reload):
    """启动 Web 监控服务"""
    try:
        import uvicorn

        from src.visualization.app import app

        click.echo(f"启动服务: http://{host}:{port}")
        click.echo("按 Ctrl+C 停止服务")
        uvicorn.run(app, host=host, port=port, reload=reload, log_level="info")
    except ImportError as e:
        click.echo(f"启动失败: {e}", err=True)
        click.echo("请安装依赖: pip install uvicorn fastapi", err=True)


@cli.command(name="demo")
@click.option("--skip-train", is_flag=True, help="跳过训练步骤")
@click.option("--skip-simulation", is_flag=True, help="跳过仿真步骤")
@click.option("--skip-web", is_flag=True, help="跳过 Web 界面")
@click.option("--port", type=int, default=8000, help="Web 端口")
def demo(skip_train, skip_simulation, skip_web, port):
    """一键演示"""
    import argparse

    from scripts.demo.demo import main

    args = argparse.Namespace(
        skip_train=skip_train,
        skip_simulation=skip_simulation,
        skip_web=skip_web,
        port=port,
    )
    main.__wrapped__(args) if hasattr(main, "__wrapped__") else main()


if __name__ == "__main__":
    cli()
