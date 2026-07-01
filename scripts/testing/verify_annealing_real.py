"""任务四验证脚本：退火真机路径 + 退火前后权重差异日志

验证两件事：
1. 退火回调被触发时，anneal() 输出"降级为仿真"日志（cqlib 无退火接口）
2. optimize_policy() 输出退火前后网络权重确实不同（L2 差异 > 0）

为避免 PPO MlpPolicy 的 1.9 万参数生成超大 QUBO 矩阵导致仿真退火过慢，
本脚本直接构造一个微型 nn.Module 调用 optimize_policy()，快速验证日志路径。

运行方式（项目根目录）：
    python scripts/verify_annealing_real.py
    python scripts/verify_annealing_real.py --real   # 绑定真机客户端触发降级日志
"""

import logging
import os
import sys
from pathlib import Path

# 项目根目录注入 sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# 启用量子加速总开关（annealing.py 内部检查此环境变量）
os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"

# 配置日志：让 annealing.py 的 logger 输出到 stdout
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    stream=sys.stdout,
)


class TinyPolicyNet:
    """微型策略网络容器，模拟 agent.policy_net 属性。

    用极小网络（58 参数）避免生成超大 QUBO 矩阵，专注于验证退火日志路径。
    """

    def __init__(self):
        from torch import nn

        self.policy_net = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 2),
        )


def main(real: bool = False):
    """运行退火验证。

    Args:
        real: 是否尝试绑定真机 cqlib 客户端（用于触发"降级为仿真"日志）
    """
    from src.quantum.annealing import QuantumAnnealingOptimizer

    print("=" * 70)
    print("任务四验证：退火真机路径 + 退火前后权重差异")
    print("=" * 70)

    # 可选：绑定真机客户端以触发降级日志
    cqlib_client = None
    simulation_mode = True
    if real:
        from dotenv import load_dotenv

        load_dotenv()
        api_key = os.getenv("TIANYAN_API_KEY", "")
        if api_key:
            try:
                from src.api.tianyan_cqlib import CqlibTianyanClient

                cqlib_client = CqlibTianyanClient(
                    login_key=api_key,
                    machine_name="tianyan_s",
                    auto_retry_machine=True,
                )
                simulation_mode = False
                print("[验证] 已绑定真机客户端，simulation_mode=False（将触发降级日志）")
            except Exception as e:
                print(f"[验证] 真机客户端创建失败 ({e})，回退到纯仿真模式")
                cqlib_client = None
                simulation_mode = True
        else:
            print("[验证] 未设置 TIANYAN_API_KEY，使用纯仿真模式")
    else:
        # 不带真机时用一个空对象模拟"cqlib 无退火接口"场景
        class _FakeCqlib:
            """模拟 cqlib 客户端：有方法但没有 submit_annealing_task。"""

            machine_name = "tianyan_s"

            def submit_quantum_task(self, **kwargs):
                return "fake_task_id"

        cqlib_client = _FakeCqlib()
        simulation_mode = False  # 强制走真机分支，触发"无退火接口"降级日志
        print(
            "[验证] 注入 FakeCqlib（无 submit_annealing_task），"
            "simulation_mode=False（将触发降级日志）"
        )

    # 构造微型网络 + 退火器
    agent = TinyPolicyNet()
    optimizer = QuantumAnnealingOptimizer(
        num_qubits=16,  # n_bits_per_weight=4，58 参数 × 4 = 232 比特
        annealing_time=10.0,
        shots=100,
        simulation_mode=simulation_mode,
        cqlib_client=cqlib_client,
    )

    print(
        f"\n[验证] simulation_mode={simulation_mode}, "
        f"cqlib_client={type(cqlib_client).__name__}"
    )
    print("[验证] 调用 optimize_policy()（3 次迭代），观察退火日志...\n")

    # 直接调用 optimize_policy，3 次迭代足够验证日志
    optimizer.optimize_policy(agent, num_iterations=3, learning_rate=0.05)

    print("\n" + "=" * 70)
    print("验证完成。请检查上方日志是否包含：")
    print("  1. [退火] '降级为仿真' 日志（cqlib 无退火接口）")
    print("  2. [退火] '权重差异 L2=...' 日志（每次迭代）")
    print("  3. [退火] '退火前后权重差异汇总' 日志（最终汇总）")
    print("  4. [退火] '✅ ... 权重确实不同' 验收通过标记")
    print("=" * 70)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="任务四退火验证")
    parser.add_argument(
        "--real",
        action="store_true",
        help="绑定真机 cqlib 客户端以触发降级为仿真日志",
    )
    args = parser.parse_args()
    main(real=args.real)
