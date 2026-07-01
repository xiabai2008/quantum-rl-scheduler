"""
端到端集成测试脚本 — 验证完整 pipeline
End-to-End Integration Test - Verify Complete Pipeline

测试流程：
    1. TaskParser.parse_qasm() → Task 对象
    2. QuantumSchedulingEnv → (obs, reward, done)
    3. SchedulerAgent → action
    4. QuantumAnnealingOptimizer → QUBO 最优解

验证各环节输出格式正确，无 ImportError / TypeError / AttributeError
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_parser():
    """测试任务解析器"""
    print("\n--- [1/4] 测试任务解析器 ---")
    try:
        from src.scheduler.parser import Task, TaskParser

        task_dict = {
            "task_id": "TEST001",
            "task_type": "quantum",
            "qubits_required": 3,
            "priority": 3,
            "urgency": 0.8,
        }

        parser = TaskParser()
        task = parser.parse(task_dict)

        assert isinstance(task, Task), f"TaskParser 返回类型错误: {type(task)}"
        assert hasattr(task, "qubits_required"), "Task 缺少 qubits_required 属性"
        assert hasattr(task, "priority"), "Task 缺少 priority 属性"
        assert hasattr(task, "task_type"), "Task 缺少 task_type 属性"
        assert task.qubits_required == 3, f"qubits_required 错误: {task.qubits_required}"

        print("  ✓ TaskParser 输出正确")
        print(f"    - task_id: {task.task_id}")
        print(f"    - task_type: {task.task_type}")
        print(f"    - qubits_required: {task.qubits_required}")
        print(f"    - priority: {task.priority}")
        return task

    except Exception as e:
        print(f"  ✗ TaskParser 测试失败: {type(e).__name__}: {e}")
        raise


def test_env(task):
    """测试调度环境"""
    print("\n--- [2/4] 测试调度环境 ---")
    try:
        from src.scheduler.env import QuantumSchedulingEnv

        env = QuantumSchedulingEnv(max_qubits=287, max_steps=100, seed=42)
        obs, info = env.reset()

        assert obs is not None, "reset() 返回 None"
        assert len(obs) == 10, f"状态维度错误: {len(obs)}"

        action = 0
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None, "step() 返回 None obs"
        assert isinstance(reward, (int, float)), f"reward 类型错误: {type(reward)}"
        assert isinstance(terminated, bool), f"terminated 类型错误: {type(terminated)}"
        assert isinstance(truncated, bool), f"truncated 类型错误: {type(truncated)}"
        assert isinstance(info, dict), f"info 类型错误: {type(info)}"

        print("  ✓ QuantumSchedulingEnv 接口完整")
        print(f"    - 状态维度: {len(obs)}")
        print(f"    - reward: {reward:.2f}")
        print(f"    - terminated: {terminated}")
        print(f"    - truncated: {truncated}")
        return env

    except Exception as e:
        print(f"  ✗ QuantumSchedulingEnv 测试失败: {type(e).__name__}: {e}")
        raise


def test_agent(env):
    """测试 RL 智能体"""
    print("\n--- [3/4] 测试 RL 智能体 ---")
    try:
        from src.scheduler.agent import SchedulerAgent

        agent = SchedulerAgent(env=env, seed=42, verbose=0)

        agent.train(total_timesteps=500, eval_freq=250, log_dir=None)

        obs, _ = env.reset()
        action = agent.predict(obs, deterministic=True)

        assert action is not None, "predict() 返回 None"
        assert isinstance(action, int), f"action 类型错误: {type(action)}"
        assert 0 <= action < 3, f"action 范围错误: {action}"

        print("  ✓ SchedulerAgent 接口完整")
        print(f"    - action: {action}")
        print("    - 动作空间: Discrete(3)")
        return agent

    except Exception as e:
        print(f"  ✗ SchedulerAgent 测试失败: {type(e).__name__}: {e}")
        raise


def test_annealing():
    """测试量子退火"""
    print("\n--- [4/4] 测试量子退火 ---")
    try:
        import numpy as np

        from src.quantum.annealing import QuantumAnnealingOptimizer

        opt = QuantumAnnealingOptimizer(num_qubits=2)

        Q = np.array([[-1, 0.5], [0.5, -1]])

        bitstring = opt.anneal(Q)

        assert bitstring is not None, "anneal() 返回 None"
        assert isinstance(bitstring, str), f"result 类型错误: {type(bitstring)}"
        assert len(bitstring) == Q.shape[0], f"比特串长度错误: {len(bitstring)} != {Q.shape[0]}"

        print("  ✓ QuantumAnnealingOptimizer 接口完整")
        print(f"    - bitstring: {bitstring}")
        print(f"    - length: {len(bitstring)}")
        return opt

    except Exception as e:
        print(f"  ✗ QuantumAnnealingOptimizer 测试失败: {type(e).__name__}: {e}")
        raise


def main():
    print(f"{'=' *60}")
    print("端到端集成测试")
    print(f"{'=' *60}")

    try:
        task = test_parser()
        env = test_env(task)
        test_agent(env)
        test_annealing()

        print(f"\n{'=' *60}")
        print("✓ 所有端到端测试通过！")
        print(f"{'=' *60}")
        print("测试流程:")
        print("  1. TaskParser → Task 对象 ✓")
        print("  2. QuantumSchedulingEnv → (obs, reward, done) ✓")
        print("  3. SchedulerAgent → action ✓")
        print("  4. QuantumAnnealingOptimizer → QUBO 最优解 ✓")
        print(f"{'=' *60}")
        return 0

    except Exception:
        print(f"\n{'=' *60}")
        print("✗ 端到端测试失败")
        print(f"{'=' *60}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
