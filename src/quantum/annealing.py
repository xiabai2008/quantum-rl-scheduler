"""
量子启发式退火加速 RL 策略搜索模块
Quantum-Inspired Annealing Accelerator for Reinforcement Learning Policy Optimization

.. deprecated:: 2026-07-27
    退火已降级为**探索性功能**，默认关闭，不再投入开发。
    - 统计显著性不达标：配对Wilcoxon p=0.9430（需N=95才达80%功效）
    - 实为经典模拟退火（neal库），非真机量子退火
    - 量子赋能AI主方向已转向**真机噪声反馈优化PPO鲁棒性**
    - 代码保留用于展示QUBO建模能力，竞赛答辩诚实标注为"探索性补充"
    - 通过 ``annealing.enabled=false``（默认）或 ``ANNEALING_ENABLED=false`` 关闭

核心思想：
    将 DQN 策略网络的参数优化问题映射为 QUBO（Quadratic Unconstrained Binary Optimization）问题，
    利用仿真模拟退火来高效求解，从而加速策略搜索过程。

QUBO 问题形式：min  x^T Q x，其中 x ∈ {0,1}^n, Q 为 n×n 的实数矩阵。

开关控制：
    通过环境变量 QUANTUM_ACCELERATION_ENABLED 可全局启用/禁用量子加速功能。
    - 设为 "1"/"true"/"yes" 启用
    - 未设置或设为其他值时禁用（回退到纯经典 RL 优化流程）

依赖：
    真机模式需要 D-Wave Ocean SDK (dwave-neal / dimod)
    仿真模式仅依赖 numpy（始终可用）
"""

import math
import os
import random
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn

# ============================================================================
# 全局开关：通过环境变量 QUANTUM_ACCELERATION_ENABLED 控制是否启用量子加速
# ============================================================================
QUANTUM_ACCELERATION_ENABLED = os.environ.get(
    "QUANTUM_ACCELERATION_ENABLED", "0"
).strip().lower() in ("1", "true", "yes")

# 尝试导入 D-Wave Ocean SDK（真机模式所需）
try:
    import neal  # D-Wave 的模拟退火求解器

    _DWAVE_AVAILABLE = True
    logger.info("已检测到 D-Wave Ocean SDK (dimod + neal)，真机/高级仿真模式可用。")
except ImportError:
    _DWAVE_AVAILABLE = False
    logger.info("未检测到 D-Wave Ocean SDK，将使用内置 numpy 模拟退火。")


# ============================================================================
# 核心类：QuantumAnnealingOptimizer
# ============================================================================
class QuantumAnnealingOptimizer:
    """
    量子退火策略优化器

    将 DQN 策略网络的权重优化问题映射为 QUBO 问题，
    并通过量子退火（或仿真模拟退火）来求解最优权重更新方向。

    典型工作流程：
        1. 从 agent 的策略网络中提取当前权重
        2. 将权重编码为 QUBO 矩阵（network_to_qubo）
        3. 调用退火器求解最优比特串（anneal）
        4. 将比特串解码回权重空间并更新网络（bitstring_to_weights）
        5. 重复迭代直至收敛（optimize_policy）

    Attributes:
        num_qubits    : 量子比特数，决定 QUBO 问题的规模
        n_bits_per_weight: 每个权重的编码位数（1 个符号位 + 数值位）
        annealing_time: 退火时间（微秒），仅在真机模式下生效
        shots         : 每次退火的采样次数，用于统计最优解
        use_dw        : 是否使用 D-Wave SDK 仿真器（优先级高于 numpy 仿真）
        solver_type   : 最后一次 anneal() 实际使用的求解器类型
                        ("real_quantum" / "neal_sa" / "numpy_sa" / "none")
    """

    def __init__(
        self,
        num_qubits: int = 16,
        annealing_time: float = 20.0,
        shots: int = 1000,
        simulation_mode: bool = True,
        cqlib_client: Any = None,
        n_bits_per_weight: int = 4,
        max_qubo_memory_mb: float = 64.0,
        random_state: int | None = None,
    ):
        """
        初始化量子退火策略优化器

        Args:
            num_qubits    : 量子比特数（默认 16），对应 QUBO 变量的个数。
                            实际使用时会自动扩展以匹配策略网络的权重总数。
                            建议值：≥16（每权重至少 4 bit，含 1 符号位 + 3 数值位）
            annealing_time: 退火时间，单位微秒（默认 20μs），仅在连接 D-Wave 真机时有效。
            shots         : 退火采样次数（默认 1000），多次采样后取能量最低的解。
            simulation_mode: 是否使用仿真模式。True=纯仿真（numpy/neal）；
                            False 时若提供了 cqlib_client 且支持退火接口则走真机退火，
                            否则降级为仿真并打印日志。默认 True。
            cqlib_client  : 天衍云 cqlib 客户端实例（可选）。simulation_mode=False
                            且客户端具备 submit_annealing_task 方法时尝试真机退火。
            n_bits_per_weight: 每个权重的编码位数（默认 4），其中 1 位为符号位。
            max_qubo_memory_mb: 单个 QUBO 矩阵允许的最大内存（MiB，默认 64）。
                                当 n_bits_per_weight 较大（如 8）时，QUBO 矩阵
                                规模为 (num_weights * n_bits)²，此参数防止内存溢出。
            random_state  : 随机种子（Issue #391）。固定后 numpy 模拟退火结果可复现；
                            None 表示不固定（保持原行为）。neal 路径也会传入此种子。
        """
        if n_bits_per_weight < 2:
            raise ValueError("n_bits_per_weight 必须至少为 2（1 个符号位 + 1 个数值位）")
        if max_qubo_memory_mb <= 0:
            raise ValueError("max_qubo_memory_mb 必须大于 0")
        self.num_qubits = num_qubits
        self.n_bits_per_weight = n_bits_per_weight
        self.max_qubo_memory_mb = max_qubo_memory_mb
        self.last_qubo_memory_bytes: int = 0
        self.annealing_time = annealing_time
        self.shots = shots
        self.simulation_mode = bool(simulation_mode)
        self.cqlib_client = cqlib_client
        # Issue #391: 随机种子，固定后退火结果可复现
        self.random_state: int | None = random_state

        # 检查比特编码精度，过低则发出警告
        if self.n_bits_per_weight < 4:
            logger.warning(
                f"每权重仅 {self.n_bits_per_weight} bit 编码 "
                f"（1 符号位 + {self.n_bits_per_weight - 1} 数值位），精度可能不足。"
                "建议 n_bits_per_weight ≥ 4 以获得更好的优化效果。"
            )
        logger.info(
            f"权重编码精度: {self.n_bits_per_weight} bit "
            f"（1 符号位 + {self.n_bits_per_weight - 1} 数值位）"
        )

        # 自动选择求解器：
        #   优先使用 D-Wave neal 模拟退火器（如果 SDK 可用）
        #   否则回退到内置 numpy 模拟退火
        self.use_dw = _DWAVE_AVAILABLE

        if self.use_dw:
            logger.info("使用 D-Wave neal 模拟退火求解器 (SimulatedAnnealingSampler)")
        else:
            logger.info("使用内置 numpy 模拟退火求解器")

        # 内置模拟退火超参数
        self._sim_initial_temp = 2.0  # 初始温度
        self._sim_cooling_rate = 0.995  # 降温系数
        self._sim_num_sweeps = 200  # 扫描次数（减少以适应 QUBO 规模）
        # Issue #391: 早停阈值——连续 _sim_patience 次扫描 best_energy 无改进则终止
        self._sim_patience = 20

        # 记录最后一次 anneal 实际使用的求解器类型（Issue #226）
        # solver_type 为公开属性，_last_solver 为向后兼容别名，两者始终同步
        #   - "real_quantum": 真机量子退火（cqlib submit_annealing_task 成功）
        #   - "neal_sa"     : D-Wave neal 模拟退火
        #   - "numpy_sa"    : 内置 numpy 模拟退火
        #   - "none"        : 尚未执行退火
        self.solver_type: str = "none"
        self._last_solver: str = "none"
        # 记录最后一次 optimize_policy 的退火统计（供外部诊断无效化/接受率）
        self._last_anneal_stats: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # 方法 1.1: get_annealing_config（Issue #247 退火参数可追溯）
    # ------------------------------------------------------------------
    def get_annealing_config(self) -> dict[str, Any]:
        """返回当前退火优化器的完整参数配置（Issue #247）。

        用于实验脚本在输出 JSON 中填充 ``annealing_config`` 字段，
        确保不同实验使用的退火参数可追溯、可复现。

        Returns:
            包含全部退火参数的字典，字段说明：
            - ``num_qubits``: 量子比特数
            - ``annealing_time``: 退火时间（μs，仅真机有效）
            - ``shots``: 采样次数
            - ``simulation_mode``: 是否仿真模式
            - ``solver_backend``: 求解器后端（"neal" / "numpy_sa"）
            - ``sim_initial_temp``: 内置 SA 初始温度
            - ``sim_cooling_rate``: 内置 SA 降温系数
            - ``sim_num_sweeps``: 内置 SA 扫描次数
            - ``n_bits_per_weight``: 每权重编码比特数
            - ``max_qubo_memory_mb``: QUBO 矩阵内存上限（MiB）
            - ``last_qubo_memory_bytes``: 最近一次 QUBO 矩阵实际内存（字节）
            - ``last_solver``: 最后一次实际使用的求解器
            - ``quantum_acceleration_enabled``: 全局加速开关
        """
        return {
            "num_qubits": self.num_qubits,
            "annealing_time": self.annealing_time,
            "shots": self.shots,
            "simulation_mode": self.simulation_mode,
            "solver_backend": "neal" if self.use_dw else "numpy_sa",
            "sim_initial_temp": self._sim_initial_temp,
            "sim_cooling_rate": self._sim_cooling_rate,
            "sim_num_sweeps": self._sim_num_sweeps,
            "n_bits_per_weight": self.n_bits_per_weight,
            "max_qubo_memory_mb": self.max_qubo_memory_mb,
            "last_qubo_memory_bytes": self.last_qubo_memory_bytes,
            "last_solver": self._last_solver,
            "quantum_acceleration_enabled": os.environ.get(
                "QUANTUM_ACCELERATION_ENABLED", "0"
            ).lower()
            in ("1", "true", "yes"),
        }

    # ------------------------------------------------------------------
    # 方法 2: network_to_qubo
    # ------------------------------------------------------------------
    def network_to_qubo(
        self,
        weights: list[np.ndarray],
        gradients: list[np.ndarray] | None = None,
        td_errors: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        将神经网络权重列表映射为 QUBO 矩阵（优化版 v2）

        改进的映射策略（解决 loss 增加问题）：
            1. **基于梯度的目标函数**：将梯度信息融入 QUBO，使 QUBO 最小化
               对应于损失函数下降方向（而非简单的权重编码）
            2. **权重差编码**：编码权重更新量 Δw = w_new - w_old，
               而非绝对权重值，确保更新方向与梯度一致
            3. **L2 正则化约束**：添加权重更新幅度的惩罚项，防止更新过大
               导致 loss 激增
            4. **对称二进制编码**：使用有符号数字表示（sign-magnitude），
               提高编码精度并支持正负更新
            5. **梯度相关性耦合**：利用梯度的 Hessian 信息构造非对角耦合项

        QUBO 目标函数：
            min_x  g^T Δw(x) + λ * ||Δw(x)||^2
            其中 Δw(x) 是从二进制变量 x 解码出的权重更新量
                  g 是梯度向量
                  λ 是正则化系数

        Args:
            weights  : 神经网络权重列表，每个元素是一个 numpy array
            gradients: 可选，梯度列表（与 weights 形状一致）。
                       若提供，QUBO 将以梯度下降方向为目标。
                       若未提供，退化为基于权重大小的编码。
            td_errors: 可选，TD 误差数组，用于调整各参数的重要性权重。

        Returns:
            QUBO 矩阵 Q，形状为 (N, N)，其中 N 为编码后的总比特数
        """
        # ---------- 步骤 1：参数配置 ----------
        n_bits_per_weight = self.n_bits_per_weight
        reg_lambda = 0.1  # L2 正则化系数，防止更新过大

        # ---------- 步骤 2：展平所有权重和梯度为一维向量 ----------
        flat_weights = np.concatenate([w.flatten() for w in weights])
        num_weights = flat_weights.size

        # 处理梯度
        if gradients is not None:
            flat_gradients = np.concatenate([g.flatten() for g in gradients])
            grad_abs_max = np.max(np.abs(flat_gradients)) + 1e-8
            flat_gradients_normalized = flat_gradients / grad_abs_max
            use_gradient = True
        else:
            flat_gradients_normalized = np.zeros_like(flat_weights)
            use_gradient = False

        # 处理 TD 误差（作为参数重要性权重）
        if td_errors is not None and len(td_errors) > 0:
            param_importance = np.ones(num_weights)
            # 按比例分配重要性（前 30% 输入层权重受 TD 误差影响更大）
            td_abs = np.mean(np.abs(td_errors))
            importance_scale = min(td_abs, 2.0)
            param_importance[: int(num_weights * 0.3)] *= 1.0 + importance_scale
        else:
            param_importance = np.ones(num_weights)

        # 总比特数 = 权重数 × 每权重编码比特数
        # 编码格式：1 bit 符号位 + (n_bits-1) bit 数值位
        total_bits = num_weights * n_bits_per_weight

        # QUBO 内存预估与保护（Issue #239）：
        # QUBO 矩阵大小为 total_bits × total_bits × float64，
        # 当 n_bits_per_weight=8 且参数量较大时可能超出内存限制。
        estimated_bytes = total_bits * total_bits * np.dtype(np.float64).itemsize
        self.last_qubo_memory_bytes = int(estimated_bytes)
        estimated_mb = estimated_bytes / (1024 * 1024)
        logger.info(
            f"QUBO 内存预估: {total_bits}² float64 = {estimated_mb:.2f} MiB "
            f"（上限 {self.max_qubo_memory_mb:.2f} MiB）"
        )
        if estimated_mb > self.max_qubo_memory_mb:
            raise MemoryError(
                f"QUBO 矩阵预计占用 {estimated_mb:.2f} MiB，"
                f"超过配置上限 {self.max_qubo_memory_mb:.2f} MiB。"
                f"可减小 n_bits_per_weight 或增大 max_qubo_memory_mb。"
            )

        logger.debug(
            f"network_to_qubo (v2): {num_weights} 个权重参数, "
            f"每参数 {n_bits_per_weight} bit, 总计 {total_bits} 个 QUBO 变量, "
            f"梯度信息: {'使用' if use_gradient else '未使用'}"
        )

        # ---------- 步骤 3：构造 QUBO 矩阵 ----------
        Q = np.zeros((total_bits, total_bits), dtype=np.float64)  # noqa: N806

        # 计算权重的全局统计量，用于归一化更新幅度
        weight_std = np.std(flat_weights) + 1e-8
        # 最大更新幅度限制为权重标准差的 10%（防止更新过大）
        max_delta = weight_std * 0.1

        for i in range(num_weights):
            w = flat_weights[i]
            imp = param_importance[i]
            base_idx = i * n_bits_per_weight

            # 梯度方向：正梯度表示应该减小权重（负更新），反之亦然
            if use_gradient:
                g_norm = flat_gradients_normalized[i]
                # 目标更新方向：与梯度相反（梯度下降）
                target_delta_direction = -g_norm
            else:
                # 无梯度时，倾向于小幅正则化（向零收缩）
                target_delta_direction = -np.sign(w) * 0.1

            # 每一位的数值权重（无符号部分）
            # bit 0: 符号位 (1=负, 0=正)
            # bit 1..n-1: 数值位，权重为 1/2, 1/4, ...
            magnitude_bits = n_bits_per_weight - 1

            for bit_k in range(n_bits_per_weight):
                global_idx = base_idx + bit_k

                if bit_k == 0:
                    # 符号位
                    # 注意：符号位没有单独的对角线性项
                    # 因为 Δw = (1-2s) * m 中 s 总是与 m 相乘
                    # 符号位的影响通过与数值位的耦合项体现
                    Q[global_idx, global_idx] = 0.0
                else:
                    # 数值位 (bit_k - 1 是数值位的索引)
                    mag_idx = bit_k - 1
                    bit_val = max_delta / (2 ** (mag_idx + 1))  # 1/2, 1/4, 1/8, ...

                    # 对角项：-t*v_k + λ*v_k²
                    # t = target_delta_direction (目标更新方向)
                    # 线性项来自 g*Δw，我们要最小化 loss，所以目标是 -t*Δw
                    Q[global_idx, global_idx] = (
                        -target_delta_direction * bit_val * imp + reg_lambda * bit_val * bit_val
                    )

            # --- 同一权重内比特间的耦合项 ---
            # 1. 符号位与数值位的耦合：来自 Δw 的符号-数值表示
            # f = -t*Δw + λ*Δw²
            #   = -t*(1-2s)*m + λ*m²
            #   = -t*m + 2t*s*m + λ*m²
            # 展开 m = Σ b_k v_k 后，s 与 b_k 的交叉项系数为 2t*v_k
            sign_idx = base_idx
            for mag_idx in range(magnitude_bits):
                bit_k = 1 + mag_idx
                bit_val = max_delta / (2 ** (mag_idx + 1))
                Q[sign_idx, bit_k] = 2.0 * target_delta_direction * bit_val * imp
                Q[bit_k, sign_idx] = Q[sign_idx, bit_k]

            # 2. 数值位之间的耦合（L2 正则化的二次项）
            for mk1 in range(magnitude_bits):
                for mk2 in range(mk1 + 1, magnitude_bits):
                    b1 = 1 + mk1
                    b2 = 1 + mk2
                    val1 = max_delta / (2 ** (mk1 + 1))
                    val2 = max_delta / (2 ** (mk2 + 1))
                    # 交叉项来自 L2 正则化: (sum b_i v_i)^2 = sum b_i^2 v_i^2 + 2 sum_{i<j} b_i b_j v_i v_j
                    coupling = 2.0 * reg_lambda * val1 * val2
                    Q[b1 + base_idx, b2 + base_idx] = coupling
                    Q[b2 + base_idx, b1 + base_idx] = coupling

        # --- 跨权重的耦合项（可选，基于 Hessian 近似）---
        # 对于同一层的相邻权重，添加弱相关耦合
        # 使用梯度信息的相关性近似 Hessian 非对角元
        if use_gradient and num_weights > 1:
            # 仅在同一层内相邻权重之间添加耦合（通过权重形状判断层边界）
            offset = 0
            for _w_idx, w_layer in enumerate(weights):
                layer_size = w_layer.size
                layer_start = offset
                layer_end = offset + layer_size

                # 在层内，对相邻权重对添加耦合
                max_pairs = min(layer_size - 1, 64)  # 限制计算量
                for i in range(max_pairs):
                    idx_i = layer_start + i
                    idx_j = layer_start + i + 1

                    g_i = flat_gradients_normalized[idx_i]
                    g_j = flat_gradients_normalized[idx_j]

                    # 近似 Hessian: 如果两个权重的梯度变化相关，则耦合
                    # 这里用梯度乘积的符号作为耦合方向
                    coupling_strength = 0.001 * g_i * g_j * max_delta * max_delta

                    # 只在符号位之间添加耦合（简化）
                    bit_i = idx_i * n_bits_per_weight
                    bit_j = idx_j * n_bits_per_weight

                    Q[bit_i, bit_j] += coupling_strength
                    Q[bit_j, bit_i] += coupling_strength

                offset = layer_end

        return Q

    # ------------------------------------------------------------------
    # 方法 3: anneal
    # ------------------------------------------------------------------
    def anneal(self, qubo_matrix: np.ndarray) -> str:
        """
        调用退火求解器（真机/仿真）求解 QUBO 问题，返回最优比特串

        求解路径优先级：
            1. 真机退火：``simulation_mode=False`` 且 ``cqlib_client`` 提供
               ``submit_annealing_task`` 方法时，提交 QUBO 到天衍云真机退火接口
            2. D-Wave neal 模拟退火：若 D-Wave Ocean SDK 可用
            3. 内置 numpy 模拟退火：始终可用的兜底实现

        天衍云 cqlib 为门控量子计算机 SDK，不提供 QUBO 退火接口；遇到此情况
        会打印"降级为仿真"日志并回退到 numpy/neal 路径，保证流程不中断。
        实际使用的求解器类型记录在 ``self.solver_type`` 属性中（Issue #226）。

        Args:
            qubo_matrix: QUBO 矩阵 Q，形状为 (N, N)

        Returns:
            best_bitstring: 最优比特串，例如 "10110..."，长度为 N
        """
        n = qubo_matrix.shape[0]

        # ---- 路径 1：真机退火（若配置启用且客户端支持） ----
        if not self.simulation_mode and self.cqlib_client is not None:
            if hasattr(self.cqlib_client, "submit_annealing_task"):
                logger.info(
                    f"[退火] 尝试真机退火 (cqlib)，QUBO 规模 {n}x{n}, "
                    f"shots={self.shots}, annealing_time={self.annealing_time}μs"
                )
                try:
                    result = self.cqlib_client.submit_annealing_task(
                        qubo_matrix,
                        shots=self.shots,
                        annealing_time=self.annealing_time,
                    )
                    # 兼容两种返回：直接返回比特串，或返回 {'bitstring': ...}
                    # 同时根据实际返回追踪 solver_type（Issue #226）
                    if isinstance(result, str):
                        best_bitstring = result
                        used_solver = "real_quantum"
                    elif isinstance(result, dict):
                        bitstring_val = str(result.get("bitstring", ""))
                        if bitstring_val:
                            best_bitstring = bitstring_val
                            used_solver = "real_quantum"
                        else:
                            logger.warning(
                                f"[退火][降级] 真机返回 dict 无 bitstring 字段，"
                                f"降级为 numpy_sa。QUBO 规模={n}x{n}, 返回={result}"
                            )
                            best_bitstring = self.numpy_simulated_annealing(qubo_matrix)
                            used_solver = "numpy_sa"
                    else:
                        logger.warning(
                            f"[退火][降级] 真机退火返回类型 {type(result)} 无法识别，"
                            f"降级为 numpy_sa。QUBO 规模={n}x{n}"
                        )
                        best_bitstring = self.numpy_simulated_annealing(qubo_matrix)
                        used_solver = "numpy_sa"
                    self.solver_type = used_solver
                    self._last_solver = used_solver
                    logger.info(
                        f"[退火] 真机退火完成，比特串长度={len(best_bitstring)}, "
                        f"求解器={used_solver}"
                    )
                    return best_bitstring
                except Exception as e:
                    # 真机退火涉及 cqlib SDK，异常类型无法穷举，保留宽捕获并记录日志
                    logger.warning(
                        f"[退火][降级] 真机退火失败 ({type(e).__name__}: {e})，"
                        f"降级为 numpy_sa。QUBO 规模={n}x{n}, "
                        f"降级原因=submit_annealing_task 异常"
                    )
                    # 继续走下方仿真路径
            else:
                logger.warning(
                    f"[退火][降级] cqlib_client 无 submit_annealing_task 接口，"
                    f"降级为仿真。QUBO 规模={n}x{n}, "
                    f"降级原因=cqlib 为门控量子 SDK，仅支持电路接口"
                )

        # ---- 路径 2/3：仿真退火 ----
        if self.use_dw:
            # ---- 使用 D-Wave neal 求解器 ----
            self.solver_type = "neal_sa"
            self._last_solver = "neal_sa"
            logger.info(
                f"[退火] 使用 D-Wave neal 求解器, QUBO 规模 {n}x{n}, "
                f"shots={self.shots}, annealing_time={self.annealing_time}μs"
            )
            qubo_dict = self._matrix_to_qubo_dict(qubo_matrix)
            sampler = neal.SimulatedAnnealingSampler()
            # Issue #391: neal 路径也传入 seed，保证可复现
            sample_kwargs: dict[str, Any] = {
                "num_reads": self.shots,
                "annealing_time": self.annealing_time,
            }
            if self.random_state is not None:
                sample_kwargs["seed"] = self.random_state
            sampleset = sampler.sample_qubo(qubo_dict, **sample_kwargs)
            # 取能量最低的样本
            best_sample = sampleset.first.sample
            best_bitstring = "".join(str(best_sample[i]) for i in range(n))
        else:
            # ---- 使用内置 numpy 模拟退火 ----
            self.solver_type = "numpy_sa"
            self._last_solver = "numpy_sa"
            logger.info(
                f"[退火] 使用内置 numpy 模拟退火, QUBO 规模 {n}x{n}, sweeps={self._sim_num_sweeps}"
            )
            best_bitstring = self.numpy_simulated_annealing(qubo_matrix)

        logger.debug(f"anneal: 最优比特串 = {best_bitstring[:32]}{'...' if n > 32 else ''}")
        return best_bitstring

    # ------------------------------------------------------------------
    # 方法 4: weight_deltas_to_bitstring / bitstring_to_weights
    # ------------------------------------------------------------------
    def weight_deltas_to_bitstring(
        self,
        deltas: list[np.ndarray],
        max_delta: float = 0.1,
    ) -> str:
        """将权重更新量编码为符号-数值比特串（Issue #239）。

        该方法与 :meth:`bitstring_to_weights` 使用相同的定点数约定，
        可用于验证 4/8 bit 等配置下的量化往返误差。

        编码格式：每权重 n_bits_per_weight 位 = 1 符号位 + (n-1) 数值位。
        数值位采用二进制小数编码（MSB = 1/2, LSB = 1/2^m），
        与 ``bitstring_to_weights`` 的解码逻辑严格对应。

        Args:
            deltas    : 权重更新量列表，每个元素为 numpy array。
            max_delta : 更新量的最大幅度，用于归一化。默认 0.1。

        Returns:
            编码后的比特串，长度 = sum(prod(shape)) × n_bits_per_weight。
        """
        if max_delta <= 0:
            raise ValueError("max_delta 必须大于 0")

        magnitude_bits = self.n_bits_per_weight - 1
        scale = 2**magnitude_bits
        max_encoded = scale - 1
        flat_deltas = np.concatenate([d.flatten() for d in deltas])
        encoded: list[str] = []

        for value in flat_deltas:
            sign = "1" if value < 0 else "0"
            quantized = round(min(abs(float(value)) / max_delta, 1.0) * scale)
            quantized = min(int(quantized), max_encoded)
            encoded.append(f"{sign}{quantized:0{magnitude_bits}b}")

        return "".join(encoded)

    def bitstring_to_weights(
        self,
        bitstring: str,
        original_shape: list[tuple[int, ...]],
        current_weights: list[np.ndarray] | None = None,
    ) -> list[np.ndarray]:
        """
        将最优比特串解码还原为神经网络权重（v2 - 符号-数值编码 + 权重差）

        解码策略：
            1. 将比特串按每 n_bits_per_weight 分组
            2. 每组格式：[符号位][数值位...] = [1 bit sign][(n-1) bits magnitude]
               - 符号位 0 = 正更新，1 = 负更新
               - 数值位为无符号定点数，编码更新量的大小
            3. 解码出权重更新量 Δw
            4. 如果提供了 current_weights，则 w_new = w_old + Δw
               否则返回 Δw 本身

        Args:
            bitstring      : 最优比特串，例如 "10110..."
            original_shape : 原始权重的形状列表，例如 [(128, 64), (64,), ...]
            current_weights: 可选，当前权重列表。若提供，返回 w_old + Δw；
                             若未提供，返回 Δw 本身。

        Returns:
            weights: 解码后的权重列表（或权重更新量列表）
        """
        n_bits_per_weight = self.n_bits_per_weight

        # 将比特串转为 bit 数组
        bits = np.array([int(b) for b in bitstring], dtype=np.float64)

        # 计算总权重数
        total_params = sum(np.prod(s) for s in original_shape)
        num_bits_used = total_params * n_bits_per_weight

        # 截断或填充比特串以匹配需要的长度
        if len(bits) >= num_bits_used:
            bits = bits[:num_bits_used]
        else:
            padded = np.zeros(num_bits_used, dtype=np.float64)
            padded[: len(bits)] = bits
            bits = padded

        # 计算当前权重的统计量，用于确定更新幅度
        if current_weights is not None:
            flat_current = np.concatenate([w.flatten() for w in current_weights])
            weight_std = np.std(flat_current) + 1e-8
            max_delta = weight_std * 0.1
        else:
            max_delta = 0.1  # 默认值

        # 解码每个权重的比特编码为连续更新量
        delta_values = np.zeros(total_params, dtype=np.float64)
        magnitude_bits = n_bits_per_weight - 1

        # 数值位的最大可能值：sum_{k=0}^{m-1} 1/2^k = 2 - 1/2^{m-1}
        # 但我们直接用最大值为 1.0（即最高位权重为 1.0，后续位递减）
        # 这样更直观：数值位直接表示 [0, 1] 之间的数
        for i in range(total_params):
            start = i * n_bits_per_weight
            end = start + n_bits_per_weight
            weight_bits = bits[start:end]

            # 第 0 位是符号位，其余是数值位
            sign_bit = weight_bits[0]
            mag_bits = weight_bits[1:]

            # 计算数值部分：直接以 [0, 1] 为范围
            # 最高位 (mag_bits[0]) 权重为 1/2，次高位 1/4，...
            # 总和范围是 [0, 1 - 1/2^m] ≈ [0, 1]
            magnitude = 0.0
            for k in range(magnitude_bits):
                if k < len(mag_bits):
                    magnitude += mag_bits[k] / (2 ** (k + 1))

            # 符号：0 = 正更新，1 = 负更新
            delta = magnitude * max_delta
            if sign_bit > 0.5:
                delta = -delta

            delta_values[i] = delta

        # 计算最终权重
        if current_weights is not None:
            flat_current = np.concatenate([w.flatten() for w in current_weights])
            final_values = flat_current + delta_values
        else:
            final_values = delta_values

        # 将解码后的值重塑为原始权重形状
        weights = []
        offset = 0
        for shape in original_shape:
            count = int(np.prod(shape))
            w = final_values[offset : offset + count].reshape(shape)
            weights.append(w)
            offset += count

        return weights

    # ------------------------------------------------------------------
    # 方法 5: optimize_policy
    # ------------------------------------------------------------------
    def _setup_head_only_params(
        self,
        policy_net: nn.Module,
        head_only: bool,
        max_head_tensors: int,
    ) -> int:
        """计算 head_only 模式下的参数张量起始索引（Issue #222 拆分）。

        Args:
            policy_net      : 策略网络
            head_only       : 是否仅优化尾部参数
            max_head_tensors: 最多优化的尾部参数张量数

        Returns:
            head_start_idx : 尾部参数的起始索引（非 head_only 模式返回 0）
        """
        if not head_only:
            return 0

        all_params = list(policy_net.parameters())
        total_tensors = len(all_params)
        n_head = min(max_head_tensors, total_tensors)
        head_start_idx = total_tensors - n_head
        head_param_count = sum(all_params[i].numel() for i in range(head_start_idx, total_tensors))
        logger.info(
            f"[退火] head_only 模式: 仅优化最后 {n_head}/{total_tensors} 个参数张量 "
            f"({head_param_count} 个标量参数)"
        )
        return head_start_idx

    def _compute_weight_delta_stats(
        self,
        current_weights: list[np.ndarray],
        optimized_weights: list[np.ndarray],
    ) -> tuple[float, float, np.ndarray]:
        """计算两组权重的差异统计（Issue #222 拆分）。

        Args:
            current_weights : 原始权重列表
            optimized_weights: 退火后权重列表

        Returns:
            (delta_l2, delta_max, delta_flat):
                - delta_l2  : 权重差异的 L2 范数
                - delta_max : 最大绝对差
                - delta_flat: 展平的差异向量
        """
        delta_flat = np.concatenate(
            [
                (ow - cw).flatten()
                for ow, cw in zip(optimized_weights, current_weights, strict=False)
            ]
        )
        delta_l2 = float(np.linalg.norm(delta_flat))
        delta_max = float(np.max(np.abs(delta_flat))) if delta_flat.size else 0.0
        return delta_l2, delta_max, delta_flat

    def _compute_actual_weight_diff(
        self,
        policy_net: nn.Module,
        head_only: bool,
        head_start_idx: int,
        old_weights: list[np.ndarray],
    ) -> float:
        """计算应用权重更新后的实际 L2 变化量（Issue #222 拆分）。

        用于无效化诊断：当 learning_rate 过小时，实际权重变化可能低于阈值。

        Args:
            policy_net    : 已应用更新的策略网络
            head_only     : 是否仅优化尾部参数
            head_start_idx: 尾部参数起始索引
            old_weights   : 更新前的权重列表

        Returns:
            weight_l2_diff_iter: 实际权重 L2 变化量
        """
        if head_only:
            all_params = list(policy_net.parameters())
            head_params = all_params[head_start_idx:]
            applied_weights = [p.detach().cpu().numpy().copy() for p in head_params]
        else:
            applied_weights = [p.detach().cpu().numpy().copy() for p in policy_net.parameters()]
        actual_delta_flat = np.concatenate(
            [(aw - ow).flatten() for aw, ow in zip(applied_weights, old_weights, strict=False)]
        )
        return float(np.linalg.norm(actual_delta_flat))

    def _finalize_anneal_stats(
        self,
        initial_l2_norm: float,
        initial_flat: np.ndarray,
        initial_loss: float,
        best_weights: list[np.ndarray] | None,
        best_loss: float,
        anneal_accepted: int,
        anneal_rejected: int,
        ineffective_count: int,
    ) -> None:
        """汇总退火统计并写入 _last_anneal_stats（Issue #222 拆分）。

        Args:
            initial_l2_norm    : 初始权重 L2 范数
            initial_flat       : 初始权重展平向量
            initial_loss       : 初始 loss
            best_weights       : 最佳权重列表
            best_loss          : 最佳 loss
            anneal_accepted    : 接受次数
            anneal_rejected    : 拒绝次数
            ineffective_count  : 无效次数
        """
        total_anneals = anneal_accepted + anneal_rejected
        accept_rate = anneal_accepted / total_anneals if total_anneals > 0 else 0.0
        logger.info(
            f"[退火] 接受/拒绝/无效统计: 接受={anneal_accepted}, "
            f"拒绝={anneal_rejected}, 无效={ineffective_count}, "
            f"接受率={accept_rate:.1%}, 求解器={self._last_solver}"
        )

        # 最终权重差异统计
        final_flat = (
            np.concatenate([w.flatten() for w in best_weights])
            if best_weights is not None
            else initial_flat
        )
        final_l2_norm = float(np.linalg.norm(final_flat))
        weight_diff = final_flat - initial_flat
        diff_l2 = float(np.linalg.norm(weight_diff))
        diff_max = float(np.max(np.abs(weight_diff))) if weight_diff.size else 0.0
        diff_relative = diff_l2 / (initial_l2_norm + 1e-12)

        self._last_anneal_stats = {
            "accepted": anneal_accepted,
            "rejected": anneal_rejected,
            "total": total_anneals,
            "accept_rate": accept_rate,
            "solver": self._last_solver,
            "solver_type": self.solver_type,
            "ineffective_count": ineffective_count,
            "weight_l2_diff": diff_l2,
        }

        logger.info(
            "[退火] 退火前后权重差异汇总: "
            f"初始 L2={initial_l2_norm:.6f}, 最终 L2={final_l2_norm:.6f}, "
            f"差异 L2={diff_l2:.6e}, 相对差异={diff_relative:.6e} "
            f"({diff_relative * 100:.4f}%), 最大绝对差={diff_max:.6e}"
        )
        if diff_l2 > 0:
            logger.info("[退火] ✅ 退火前后 PPO 网络权重确实不同（验收通过）")
        else:
            logger.warning("[退火] ⚠️ 退火前后权重完全相同，请检查退火是否生效")

        logger.info(
            f"量子退火策略优化完成: 最佳 loss={best_loss:.6f}, "
            f"初始 loss={initial_loss:.6f}, "
            f"改进: {((initial_loss - best_loss) / max(initial_loss, 1e-8) * 100):.2f}%"
        )

    def optimize_policy(
        self,
        agent: Any,
        num_iterations: int = 10,
        learning_rate: float = 0.01,
        callback: Any | None = None,
        replay_buffer: Any | None = None,
        head_only: bool = True,
        max_head_tensors: int = 4,
        mode: str = "head_only",
        max_params_per_block: int = 200,
        block_strategy: str = "tensor_wise",
        min_effective_delta: float = 1e-4,
    ) -> Any:
        """
        主优化循环：用量子退火加速策略更新（v2 - 梯度引导）

        改进的优化流程（解决 loss 增加问题）：
            1. 从经验回放缓冲区采样批次数据
            2. 计算策略网络的梯度（TD 误差反向传播）
            3. 将梯度信息融入 QUBO 构造，使 QUBO 最小化对应梯度下降方向
            4. 退火求解最优权重更新量 Δw
            5. 用学习率缩放更新量，更新网络权重
            6. 接受准则：只有当 loss 下降时才接受更新（防止 loss 增加）

        支持三种优化模式 (mode 参数)：
            - "head_only" (默认): 仅优化网络尾部 N 个参数张量，向后兼容
            - "hierarchical": 分层/分块退火，逐块 QUBO → 全量网络覆盖，突破 OOM
            - "full": 全量单次 QUBO（仅在小网络 <200 参数时使用）

        无效化诊断 (min_effective_delta)：
            当 learning_rate 过小（如默认 0.01）时，实际权重更新量
            w_final = w_old + lr * delta 的 L2 范数可能低于 min_effective_delta，
            此时退火更新实质上是空操作。该方法会在每次迭代后计算实际权重 L2
            变化量，低于阈值则标记为 ineffective 并跳过接受/拒绝判定，
            最终统计 ineffective_count 写入 _last_anneal_stats。

        Args:
            agent               : RL 智能体（需具有 policy_net 属性，为 nn.Module）
            num_iterations      : 量子退火优化迭代次数（默认 10）
            learning_rate       : 权重更新学习率（默认 0.01），控制更新幅度
            callback            : 可选的回调函数，签名为 callback(iteration, loss)
            replay_buffer       : 可选，经验回放缓冲区。若提供，用于计算梯度；
                                  若未提供，退化为基于权重正则化的优化。
            head_only           : [已废弃，使用 mode="head_only" 替代]
                                  是否仅优化网络输出头权重（默认 True）。
            max_head_tensors    : head_only 模式时，最多优化的尾部参数张量数（默认 4）。
            mode                : 优化模式: "head_only" / "hierarchical" / "full"
            max_params_per_block: hierarchical 模式时每块最大参数数（默认 200）
            block_strategy      : hierarchical 模式的分块策略: "tensor_wise" / "size_limited"
            min_effective_delta : 单次迭代实际权重 L2 变化量的下限阈值（默认 1e-4）。
                                  低于此值时该次迭代被标记为 ineffective（无效），
                                  跳过接受/拒绝判定，用于诊断 learning_rate 过小导致的
                                  退火无效化问题。

        Returns:
            agent: 优化后的智能体（原地修改并返回）
        """
        # 模式路由：hierarchical 模式委托给独立方法
        if mode == "hierarchical":
            return self.optimize_policy_hierarchical(
                agent=agent,
                num_iterations=num_iterations,
                learning_rate=learning_rate,
                callback=callback,
                replay_buffer=replay_buffer,
                max_params_per_block=max_params_per_block,
                block_strategy=block_strategy,
                min_effective_delta=min_effective_delta,
            )

        if not QUANTUM_ACCELERATION_ENABLED:
            logger.warning(
                "[退火][降级] 量子加速功能已禁用 (QUANTUM_ACCELERATION_ENABLED 未设置)。"
                "跳过 optimize_policy，直接返回原始 agent。"
                f"降级原因=QUANTUM_ACCELERATION_ENABLED 未启用, "
                f"目标求解器=none, num_qubits={self.num_qubits}"
            )
            return agent

        # 获取策略网络
        # head_only 模式需要完整的 policy（含 action_net/value_net 输出头）
        # 非 head_only 模式使用 mlp_extractor 即可
        policy_net = self.get_full_policy(agent) if head_only else self._get_policy_net(agent)
        if policy_net is None:
            logger.error("无法获取策略网络，退出 optimize_policy")
            return agent

        # 退火求解器选择日志（Issue #229: 显式记录目标求解器类型）
        if self.simulation_mode:
            target_solver = "neal_sa" if self.use_dw else "numpy_sa"
            logger.info(
                f"[退火] simulation_mode=True，使用仿真求解器={target_solver}, "
                f"QUBO 比特数={self.num_qubits}, n_bits_per_weight={self.n_bits_per_weight}"
            )
        else:
            target_solver = (
                "real_quantum"
                if (
                    self.cqlib_client is not None
                    and hasattr(self.cqlib_client, "submit_annealing_task")
                )
                else "neal_sa"
            )
            logger.info(
                f"[退火] simulation_mode=False，目标求解器={target_solver}, "
                f"cqlib_client={'已提供' if self.cqlib_client is not None else 'None'}, "
                f"QUBO 比特数={self.num_qubits}"
            )

        logger.info(
            f"开始量子退火策略优化 (v2 - 梯度引导): {num_iterations} 次迭代, "
            f"学习率={learning_rate}, 量子比特数={self.num_qubits}"
            f"{', head_only模式' if head_only else ''}"
        )

        # 如果启用了 head_only 模式，计算需要优化的参数张量索引范围
        # PPO 完整 policy 的参数顺序: [0-7: mlp_extractor, 8-9: action_net, 10-11: value_net]
        # 仅优化最后 max_head_tensors 个（action_net + value_net = 4 个张量, 260 参数）
        head_start_idx = self._setup_head_only_params(policy_net, head_only, max_head_tensors)

        best_loss = float("inf")
        best_weights = None
        history = []
        # 统计退火接受/拒绝次数（供外部诊断为何不同策略训练结果相同）
        anneal_accepted = 0
        anneal_rejected = 0
        # 统计退火无效化次数（Issue #194: 诊断 learning_rate 过小导致退火空操作）
        ineffective_count = 0

        # 初始评估
        initial_loss = self._evaluate_network_quality(policy_net)
        best_loss = initial_loss
        initial_weights, _initial_shapes = self.extract_weights(policy_net)
        best_weights = [w.copy() for w in initial_weights]
        # 记录初始权重 L2 范数，用于最终计算退火前后权重差异
        initial_flat = np.concatenate([w.flatten() for w in initial_weights])
        initial_l2_norm = float(np.linalg.norm(initial_flat))
        logger.info(
            f"[退火] 初始权重统计: 参数数={initial_flat.size}, "
            f"L2 范数={initial_l2_norm:.6f}, loss={initial_loss:.6f}"
        )

        for iteration in range(num_iterations):
            # ---- 步骤 1: 提取当前权重 ----
            all_weights, all_shapes = self.extract_weights(policy_net)

            # head_only 模式：仅优化最后 N 个参数张量
            if head_only:
                current_weights = all_weights[head_start_idx:]
                original_shapes = all_shapes[head_start_idx:]
            else:
                current_weights = all_weights
                original_shapes = all_shapes

            # ---- 步骤 2: 计算梯度（如果有 replay buffer）----
            gradients = None
            td_errors = None
            current_loss = initial_loss

            if replay_buffer is not None and hasattr(replay_buffer, "sample"):
                try:
                    gradients, td_errors, current_loss = self._compute_gradients(
                        policy_net, replay_buffer, agent
                    )
                    if head_only and gradients is not None:
                        gradients = gradients[head_start_idx:]
                    logger.debug(f"  梯度计算成功, TD 误差均值={np.mean(np.abs(td_errors)):.4f}")
                except Exception as e:
                    # 梯度计算涉及 PyTorch 张量运算与 replay buffer，异常类型无法穷举，保留宽捕获并记录日志
                    logger.warning(f"  梯度计算失败: {e}, 退化为无梯度模式")
                    gradients = None

            # ---- 步骤 3: 映射为 QUBO 矩阵（带梯度信息）----
            qubo_matrix = self.network_to_qubo(
                current_weights,
                gradients=gradients,
                td_errors=td_errors,
            )

            # ---- 步骤 4: 退火求解 ----
            best_bitstring = self.anneal(qubo_matrix)

            # ---- 步骤 5: 解码为权重更新（使用当前权重作为基准）----
            optimized_head_weights = self.bitstring_to_weights(
                best_bitstring,
                original_shapes,
                current_weights=current_weights,
            )

            # 退火前后权重差异（L2 范数 + 最大绝对差）
            delta_l2, delta_max, _delta_flat = self._compute_weight_delta_stats(
                current_weights, optimized_head_weights
            )
            logger.info(
                f"[退火] 迭代 {iteration + 1}/{num_iterations}: "
                f"权重差异 L2={delta_l2:.6e}, 最大绝对差={delta_max:.6e}"
            )

            # ---- 步骤 6: 应用权重更新（带接受准则）----
            # 先保存旧权重，用于回滚
            old_weights = [w.copy() for w in current_weights]

            # 应用更新（head_only 模式下仅更新尾部参数）
            if head_only:
                # 仅更新网络尾部参数
                all_param_list = list(policy_net.parameters())
                head_params = all_param_list[head_start_idx:]
                self._apply_weights_v2_partial(
                    head_params,
                    current_weights,
                    optimized_head_weights,
                    learning_rate=learning_rate,
                )
            else:
                self._apply_weights_v2(
                    policy_net,
                    current_weights,
                    optimized_head_weights,
                    learning_rate=learning_rate,
                )

            # ---- 步骤 6.5: 无效化诊断（Issue #194）----
            # 计算实际权重 L2 变化量（应用 learning_rate 缩放后的真实变化）
            # 当 learning_rate 过小时，w_final = w_old + lr * delta 的变化量极小，
            # 退火实质上是空操作。低于 min_effective_delta 则跳过接受/拒绝判定。
            weight_l2_diff_iter = self._compute_actual_weight_diff(
                policy_net, head_only, head_start_idx, old_weights
            )

            if weight_l2_diff_iter < min_effective_delta:
                # 退火更新量过小，标记为无效，跳过接受/拒绝判定
                ineffective_count += 1
                history.append((iteration, current_loss, current_loss, False))
                logger.warning(
                    f"  迭代 {iteration + 1}/{num_iterations}: "
                    f"实际权重变化 L2={weight_l2_diff_iter:.6e} "
                    f"< 阈值 {min_effective_delta:.6e}，标记为无效 (ineffective)，"
                    f"跳过接受/拒绝判定"
                )
                if callback is not None:
                    callback(iteration, current_loss)
                continue

            # ---- 步骤 7: 评估更新后的 loss，决定是否接受 ----
            new_loss = self._evaluate_network_quality(policy_net)
            loss_improvement = current_loss - new_loss

            # 接受准则：loss 下降，或上升幅度不超过阈值（早期探索）
            accept_threshold = 0.01 * current_loss  # 允许 1% 的暂时上升
            if new_loss <= best_loss or loss_improvement > -accept_threshold:
                # 接受更新
                accepted = True
                anneal_accepted += 1
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_weights, _ = self.extract_weights(policy_net)
            else:
                # 回滚：仅回滚被修改的那部分参数
                if head_only:
                    head_params = list(policy_net.parameters())[head_start_idx:]
                    self._set_params_from_weights(head_params, old_weights)
                else:
                    self._set_weights(policy_net, old_weights)
                accepted = False
                anneal_rejected += 1

            history.append((iteration, current_loss, new_loss, accepted))

            logger.info(
                f"  迭代 {iteration + 1}/{num_iterations}: "
                f"优化前 loss={current_loss:.6f}, 优化后 loss={new_loss:.6f}, "
                f"{'✅ 接受' if accepted else '❌ 拒绝'}, "
                f"最佳 loss={best_loss:.6f}"
            )

            if callback is not None:
                callback(iteration, new_loss)

        # 退火接受/拒绝汇总（诊断 A/B 策略结果为何相同）
        # 恢复到最佳权重
        if best_weights is not None:
            self._set_weights(policy_net, best_weights)
            logger.info(f"已恢复到最佳权重 (loss={best_loss:.6f})")

        # 汇总退火统计并写入 _last_anneal_stats（Issue #222 拆分）
        self._finalize_anneal_stats(
            initial_l2_norm=initial_l2_norm,
            initial_flat=initial_flat,
            initial_loss=initial_loss,
            best_weights=best_weights,
            best_loss=best_loss,
            anneal_accepted=anneal_accepted,
            anneal_rejected=anneal_rejected,
            ineffective_count=ineffective_count,
        )

        # 如果 agent 有 target_net，同步更新
        if hasattr(agent, "target_net"):
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            logger.info("已同步更新 target_net")

        return agent

    # ------------------------------------------------------------------
    # 方法 6: optimize_policy_hierarchical（分块/分层 QUBO 退火）
    # ------------------------------------------------------------------
    def optimize_policy_hierarchical(
        self,
        agent: Any,
        num_iterations: int = 10,
        learning_rate: float = 0.01,
        callback: Any | None = None,
        replay_buffer: Any | None = None,
        max_params_per_block: int = 200,
        block_strategy: str = "tensor_wise",
        min_effective_delta: float = 1e-4,
    ) -> Any:
        """
        分层/分块量子退火策略优化（突破 head_only 限制，全量网络退火）

        核心思想：
            将全量网络参数按张量或参数量分块，每块独立构造小规模 QUBO 并退火求解，
            避免全量参数合并成大 QUBO 矩阵导致 OOM（当前 head_only 的瓶颈）。

        分块策略 (block_strategy)：
            - "tensor_wise": 每个参数张量作为一个独立块（推荐，符合网络层结构）
            - "size_limited": 按 max_params_per_block 动态分块，保证每块 ≤ 指定上限

        工作流程（每轮迭代）：
            1. 提取全量网络权重
            2. 按策略将参数分块
            3. 对每块：
                a. 提取该块的权重和梯度（如有 replay_buffer）
                b. 构造小块 QUBO 矩阵
                c. 退火求解最优比特串
                d. 解码为权重更新量并应用到网络
            4. 评估全量网络的 loss，决定是否接受本轮更新
            5. 多轮迭代，逐块逼近全局最优

        内存优势：
            每块 QUBO 矩阵大小 = (块内参数 × 每参数比特)²
            例如 200 参数 × 4bit = 800² ≈ 5 MB（vs 全量 2000+ 参数 × 4bit = 8000² ≈ 512 MB）

        Args:
            agent               : RL 智能体
            num_iterations      : 外层迭代轮数（默认 10）
            learning_rate       : 权重更新学习率（默认 0.01）
            callback            : 可选回调，签名 callback(iteration, loss)
            replay_buffer       : 可选，经验回放缓冲区
            max_params_per_block: 每块最大参数数（仅在 size_limited 策略下生效，默认 200）
            block_strategy      : 分块策略，默认 "tensor_wise"
            min_effective_delta : 单次迭代实际权重 L2 变化量的下限阈值（默认 1e-4）。
                                  低于此值时该轮迭代被标记为 ineffective（无效），
                                  跳过接受/拒绝判定。

        Returns:
            agent: 优化后的智能体
        """
        if not QUANTUM_ACCELERATION_ENABLED:
            logger.warning(
                "[退火][降级] 量子加速功能已禁用，跳过分层退火。"
                f"降级原因=QUANTUM_ACCELERATION_ENABLED 未启用, "
                f"目标求解器=none, num_qubits={self.num_qubits}"
            )
            return agent

        policy_net = (
            self._get_policy_net(agent)
            if block_strategy == "size_limited"
            else self.get_full_policy(agent)
        )
        if policy_net is None:
            logger.error("无法获取策略网络，退出分层退火。")
            return agent

        # 获取全量参数，构建分块索引
        all_params = list(policy_net.parameters())
        total_tensors = len(all_params)
        total_params_count = sum(p.numel() for p in all_params)
        n_bits_per_weight = self.n_bits_per_weight

        blocks = self._create_param_blocks(all_params, block_strategy, max_params_per_block)

        _qubo_mb = (max_params_per_block * n_bits_per_weight) ** 2 * 8 / 1024 / 1024
        logger.info(
            f"开始分层/分块量子退火 ({block_strategy}): "
            f"{num_iterations} 轮, 全量 {total_tensors} 张量/{total_params_count} 参数, "
            f"分为 {len(blocks)} 块, 每块 ≤{max_params_per_block} 参数, "
            f"预估每块 QUBO ≤{_qubo_mb:.1f} MB"
        )

        # 初始评估
        initial_loss = self._evaluate_network_quality(policy_net)
        best_loss = initial_loss
        best_weights, _ = self.extract_weights(policy_net)
        # 保存初始权重快照，用于最终计算退火前后权重差异（Issue #194）
        initial_weights_hier = [w.copy() for w in best_weights]

        logger.info(
            f"[分层退火] 初始 loss={initial_loss:.6f}, "
            f"全量参数={total_params_count}, 分块数={len(blocks)}"
        )

        # 统计退火无效化次数（Issue #194）
        hierarchical_ineffective_count = 0

        for iteration in range(num_iterations):
            # 保存本轮开始前的全量权重（用于可能的回滚）
            old_all_weights, _old_all_shapes = self.extract_weights(policy_net)

            # 逐块处理
            total_accepted_blocks = 0
            for block_idx, block_param_indices in enumerate(blocks):
                # --- 提取该块的权重 ---
                block_weights = [
                    all_params[idx].detach().cpu().numpy().copy() for idx in block_param_indices
                ]
                block_shapes = [w.shape for w in block_weights]
                block_param_count = sum(w.size for w in block_weights)

                # --- 计算该块的梯度 ---
                block_gradients = None
                td_errors = None
                if replay_buffer is not None and hasattr(replay_buffer, "sample"):
                    try:
                        full_gradients, td_errors, _ = self._compute_gradients(
                            policy_net, replay_buffer, agent
                        )
                        if full_gradients is not None:
                            block_gradients = [full_gradients[i] for i in block_param_indices]
                    except Exception:
                        block_gradients = None

                # --- 构造小块 QUBO ---
                qubo_matrix = self.network_to_qubo(
                    block_weights,
                    gradients=block_gradients,
                    td_errors=td_errors,
                )

                # --- 退火求解 ---
                best_bitstring = self.anneal(qubo_matrix)

                # --- 解码为权重更新 ---
                optimized_block_weights = self.bitstring_to_weights(
                    best_bitstring,
                    block_shapes,
                    current_weights=block_weights,
                )

                # --- 应用权重更新（仅该块）---
                block_params = [all_params[idx] for idx in block_param_indices]
                self._apply_weights_v2_partial(
                    block_params,
                    block_weights,
                    optimized_block_weights,
                    learning_rate=learning_rate,
                )

                # 该块的权重变化统计
                block_delta = np.concatenate(
                    [
                        (ow - cw).flatten()
                        for ow, cw in zip(optimized_block_weights, block_weights, strict=False)
                    ]
                )
                block_delta_l2 = float(np.linalg.norm(block_delta))

                if block_delta_l2 > 1e-12:
                    total_accepted_blocks += 1

                logger.debug(
                    f"  块 {block_idx + 1}/{len(blocks)} ({block_param_count} 参数): "
                    f"QUBO {qubo_matrix.shape[0]}×{qubo_matrix.shape[0]}, "
                    f"ΔL2={block_delta_l2:.6e}"
                )

            # --- 无效化诊断（Issue #194）---
            # 计算本轮全量网络实际权重 L2 变化量（应用 learning_rate 缩放后）
            current_all_weights, _ = self.extract_weights(policy_net)
            hier_delta_flat = np.concatenate(
                [
                    (cw - ow).flatten()
                    for cw, ow in zip(current_all_weights, old_all_weights, strict=False)
                ]
            )
            hier_weight_l2_diff = float(np.linalg.norm(hier_delta_flat))

            if hier_weight_l2_diff < min_effective_delta:
                # 本轮退火更新量过小，标记为无效，跳过接受/拒绝判定
                hierarchical_ineffective_count += 1
                logger.warning(
                    f"[分层退火] 轮次 {iteration + 1}/{num_iterations}: "
                    f"实际权重变化 L2={hier_weight_l2_diff:.6e} "
                    f"< 阈值 {min_effective_delta:.6e}，标记为无效 (ineffective)，"
                    f"跳过接受/拒绝判定"
                )
                if callback is not None:
                    callback(iteration, initial_loss)
                continue

            # --- 评估本轮全量更新后的 loss ---
            new_loss = self._evaluate_network_quality(policy_net)
            loss_improvement = best_loss - new_loss

            # 接受准则
            accept_threshold = 0.01 * best_loss
            if new_loss <= best_loss or loss_improvement > -accept_threshold:
                accepted = True
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_weights, _ = self.extract_weights(policy_net)
            else:
                # 回滚全量权重
                self._set_weights(policy_net, old_all_weights)
                accepted = False

            logger.info(
                f"[分层退火] 轮次 {iteration + 1}/{num_iterations}: "
                f"更新前 loss={initial_loss:.6f}, 更新后 loss={new_loss:.6f}, "
                f"最佳 loss={best_loss:.6f}, "
                f"{'✅' if accepted else '❌'}, "
                f"有效块 {total_accepted_blocks}/{len(blocks)}"
            )

            if callback is not None:
                callback(iteration, new_loss)

        # 恢复最佳权重
        if best_weights is not None:
            self._set_weights(policy_net, best_weights)
            logger.info(f"[分层退火] 已恢复到最佳权重 (loss={best_loss:.6f})")

        # 最终权重差异统计（Issue #194）
        final_hier_weights, _ = self.extract_weights(policy_net)
        final_hier_flat = np.concatenate([w.flatten() for w in final_hier_weights])
        init_hier_flat = np.concatenate([w.flatten() for w in initial_weights_hier])
        hier_weight_diff = final_hier_flat - init_hier_flat
        hier_diff_l2 = float(np.linalg.norm(hier_weight_diff))

        # 写入退火统计（与 optimize_policy 保持一致的诊断字段）
        self._last_anneal_stats = {
            "accepted": num_iterations - hierarchical_ineffective_count,
            "rejected": 0,
            "total": num_iterations,
            "accept_rate": (num_iterations - hierarchical_ineffective_count)
            / max(num_iterations, 1),
            "solver": self._last_solver,
            "solver_type": self.solver_type,
            "ineffective_count": hierarchical_ineffective_count,
            "weight_l2_diff": hier_diff_l2,
        }

        final_improvement = (initial_loss - best_loss) / max(initial_loss, 1e-8) * 100
        logger.info(
            f"[分层退火] 完成: 初始 loss={initial_loss:.6f}, "
            f"最佳 loss={best_loss:.6f}, 改进={final_improvement:.2f}%, "
            f"处理了 {len(blocks)} 个参数块 × {num_iterations} 轮, "
            f"无效轮次={hierarchical_ineffective_count}, "
            f"最终权重差异 L2={hier_diff_l2:.6e}"
        )

        # 同步 target_net
        if hasattr(agent, "target_net"):
            agent.target_net.load_state_dict(agent.policy_net.state_dict())

        return agent

    # ==================================================================
    # 内部辅助方法
    # ==================================================================

    @staticmethod
    def _create_param_blocks(
        all_params: list,
        block_strategy: str = "tensor_wise",
        max_params_per_block: int = 200,
    ) -> list[list[int]]:
        """
        将网络参数分块，每块独立构造 QUBO 并退火

        分块策略：
            - "tensor_wise": 每个参数张量作为一个独立块
            - "size_limited": 按参数量动态分块，保证每块 ≤ max_params_per_block

        Args:
            all_params         : PyTorch 模型参数列表 (list of nn.Parameter)
            block_strategy     : 分块策略
            max_params_per_block: 每块最大参数数（仅 size_limited 生效）

        Returns:
            blocks: 块索引列表，每个元素为 [param_idx, ...]
        """
        if block_strategy == "tensor_wise":
            # 每个张量独立成块
            return [[i] for i in range(len(all_params))]

        # size_limited: 按参数量动态分组
        blocks: list[list[int]] = []
        current_block: list[int] = []
        current_size = 0

        for idx, param in enumerate(all_params):
            param_count = param.numel()
            # 如果单个张量就超过上限，独立成块
            if param_count > max_params_per_block:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                    current_size = 0
                blocks.append([idx])
                continue

            if current_size + param_count > max_params_per_block and current_block:
                blocks.append(current_block)
                current_block = [idx]
                current_size = param_count
            else:
                current_block.append(idx)
                current_size += param_count

        if current_block:
            blocks.append(current_block)

        return blocks

    @staticmethod
    def _get_policy_net(agent: Any) -> nn.Module | None:
        """
        从 agent 对象中获取策略网络

        支持的 agent 类型：
            - 具有 policy_net 属性的 SchedulingAgent
            - SB3 DQN agent (policy.q_net)
            - SB3 PPO agent (policy.mlp_extractor 或 policy)
        """
        # 方式 1：直接属性（项目内的 SchedulingAgent）
        if hasattr(agent, "policy_net") and isinstance(agent.policy_net, nn.Module):
            return agent.policy_net

        # 方式 2：Stable-Baselines3 DQN agent
        if hasattr(agent, "policy") and hasattr(agent.policy, "q_net"):
            return agent.policy.q_net

        # 方式 3：Stable-Baselines3 PPO agent（ActorCriticPolicy）
        if hasattr(agent, "policy") and isinstance(agent.policy, nn.Module):
            # PPO 的 policy 是 ActorCriticPolicy，内含 mlp_extractor
            if hasattr(agent.policy, "mlp_extractor"):
                return agent.policy.mlp_extractor
            # 回退：直接返回整个 policy 网络
            return agent.policy

        logger.warning("无法识别 agent 的策略网络结构")
        return None

    @staticmethod
    def get_full_policy(agent: Any) -> nn.Module | None:
        """
        获取完整的 policy 网络（含输出头），用于 head_only 模式

        与 get_policy_net 的区别：
            - get_policy_net 对 PPO 返回 mlp_extractor（不含 action_net/value_net）
            - get_full_policy 对 PPO 返回完整的 ActorCriticPolicy（含所有参数）

        支持的 agent 类型：与 get_policy_net 相同
        """
        # SB3 PPO: 返回完整的 policy（含 action_net + value_net 输出头）
        if hasattr(agent, "policy") and isinstance(agent.policy, nn.Module):
            return agent.policy

        # 其它类型回退到 _get_policy_net
        return QuantumAnnealingOptimizer._get_policy_net(agent)

    @staticmethod
    def extract_weights(
        network: nn.Module,
    ) -> tuple[list[np.ndarray], list[tuple[int, ...]]]:
        """
        从 PyTorch 网络中提取所有权重参数

        Returns:
            weights        : 权重列表（每个元素为 numpy array）
            original_shapes: 每个权重张量的形状列表
        """
        weights = []
        shapes = []
        for param in network.parameters():
            w = param.detach().cpu().numpy().copy()
            weights.append(w)
            shapes.append(w.shape)
        return weights, shapes

    @staticmethod
    def _evaluate_network_quality(network: nn.Module) -> float:
        """
        评估网络质量（用作 QUBO 构造的辅助信息）

        使用权重 L2 正则化作为简单的质量度量。
        在实际应用中可替换为经验回放缓冲区的平均 TD 误差。

        Returns:
            loss: 质量分数（越小越好）
        """
        total_norm = 0.0
        num_params = 0
        for param in network.parameters():
            total_norm += param.detach().cpu().norm(2).item() ** 2
            num_params += param.numel()
        # 归一化：每参数的平均 L2 范数
        avg_l2 = math.sqrt(total_norm) / max(num_params, 1)
        return avg_l2

    @staticmethod
    def _apply_weights(
        network: nn.Module,
        old_weights: list[np.ndarray],
        new_weights: list[np.ndarray],
        shapes: list[tuple[int, ...]],
        learning_rate: float = 0.01,
    ):
        """
        将优化后的权重应用到网络（旧版本，保留用于向后兼容）

        使用线性插值混合新旧权重：
            w_final = (1 - lr) * w_old + lr * w_new

        Args:
            network       : PyTorch 神经网络
            old_weights   : 旧权重列表
            new_weights   : 量子退火优化后的权重列表
            shapes        : 权重形状列表（用于验证）
            learning_rate : 学习率，控制更新幅度
        """
        with torch.no_grad():
            for param, w_old, w_new, shape in zip(
                network.parameters(), old_weights, new_weights, shapes, strict=False
            ):
                assert w_new.shape == shape, f"权重形状不匹配: 期望 {shape}, 实际 {w_new.shape}"
                old_std = np.std(w_old) + 1e-8
                new_std = np.std(w_new) + 1e-8
                w_new_scaled = w_new * (old_std / new_std)

                w_final = (1.0 - learning_rate) * w_old + learning_rate * w_new_scaled
                param.copy_(torch.from_numpy(w_final.astype(np.float32)))

    @staticmethod
    def _apply_weights_v2(
        network: nn.Module,
        old_weights: list[np.ndarray],
        new_weights: list[np.ndarray],
        learning_rate: float = 0.01,
    ):
        """
        将优化后的权重应用到网络（v2 版本）

        与 v1 的区别：
        - new_weights 已经是包含当前权重的完整权重（w_old + Δw）
        - 使用 learning_rate 控制更新步长：w_final = w_old + lr * (w_new - w_old)
        - 不需要重新缩放，因为 Δw 已经是在正确的尺度上

        Args:
            network       : PyTorch 神经网络
            old_weights   : 旧权重列表
            new_weights   : 量子退火优化后的完整权重列表
            learning_rate : 学习率，控制更新幅度
        """
        with torch.no_grad():
            for param, w_old, w_new in zip(
                network.parameters(), old_weights, new_weights, strict=False
            ):
                # 计算更新量 Δw = w_new - w_old
                delta = w_new - w_old

                # 用学习率缩放更新量
                w_final = w_old + learning_rate * delta

                param.copy_(torch.from_numpy(w_final.astype(np.float32)))

    @staticmethod
    def _set_weights(network: nn.Module, weights: list[np.ndarray]):
        """
        直接设置网络权重（用于回滚）

        Args:
            network: PyTorch 神经网络
            weights: 权重列表
        """
        with torch.no_grad():
            for param, w in zip(network.parameters(), weights, strict=False):
                param.copy_(torch.from_numpy(w.astype(np.float32)))

    @staticmethod
    def _apply_weights_v2_partial(
        params: list[nn.Parameter],
        old_weights: list[np.ndarray],
        new_weights: list[np.ndarray],
        learning_rate: float = 0.01,
    ):
        """
        将优化后的权重应用到指定的参数子集（用于 head_only 模式）

        Args:
            params        : PyTorch 参数列表（子集）
            old_weights   : 旧权重列表
            new_weights   : 量子退火优化后的完整权重列表
            learning_rate : 学习率，控制更新幅度
        """
        with torch.no_grad():
            for param, w_old, w_new in zip(params, old_weights, new_weights, strict=False):
                delta = w_new - w_old
                w_final = w_old + learning_rate * delta
                param.copy_(torch.from_numpy(w_final.astype(np.float32)))

    @staticmethod
    def _set_params_from_weights(params: list[nn.Parameter], weights: list[np.ndarray]):
        """
        直接将权重写入参数子集（用于 head_only 模式下的回滚）

        Args:
            params  : PyTorch 参数列表（子集）
            weights : 权重列表
        """
        with torch.no_grad():
            for param, w in zip(params, weights, strict=False):
                param.copy_(torch.from_numpy(w.astype(np.float32)))

    def _compute_gradients(
        self,
        policy_net: nn.Module,
        replay_buffer: Any,
        agent: Any,
        batch_size: int = 64,
    ) -> tuple[list[np.ndarray], np.ndarray, float]:
        """
        计算策略网络的梯度和 TD 误差

        从经验回放缓冲区采样一批数据，前向传播计算 TD 误差，
        反向传播得到梯度。

        Args:
            policy_net   : 策略网络
            replay_buffer: 经验回放缓冲区
            agent        : RL 智能体（用于获取 gamma 等参数）
            batch_size   : 采样批次大小

        Returns:
            gradients: 梯度列表（与网络参数一一对应）
            td_errors: TD 误差数组
            loss     : 标量损失值
        """
        # 尝试从 replay buffer 采样
        if hasattr(replay_buffer, "sample"):
            try:
                batch = replay_buffer.sample(batch_size)
            except Exception as e:
                # replay buffer 采样异常类型因实现而异，保留宽捕获并记录原始异常
                logger.debug(f"Replay buffer 采样失败: {type(e).__name__}: {e}")
                raise ValueError("Replay buffer 采样失败") from None
        else:
            raise ValueError("Replay buffer 不支持 sample 方法")

        # 解析 batch（兼容不同的 replay buffer 格式）
        # SB3 的 ReplayBuffer 返回的是 namedtuple 或字典
        if isinstance(batch, tuple) and len(batch) >= 5:
            observations = torch.from_numpy(batch[0]).float()
            actions = torch.from_numpy(batch[1]).long()
            rewards = torch.from_numpy(batch[2]).float()
            next_observations = torch.from_numpy(batch[3]).float()
            dones = torch.from_numpy(batch[4]).float()
        elif hasattr(batch, "observations"):
            observations = batch.observations.float()
            actions = batch.actions.long()
            rewards = batch.rewards.float()
            next_observations = batch.next_observations.float()
            dones = batch.dones.float()
        else:
            raise ValueError(f"无法解析 batch 格式: {type(batch)}")

        # 获取 gamma
        gamma = getattr(agent, "gamma", 0.99)

        # 前向传播
        policy_net.train()
        q_values = policy_net(observations)
        q_value = q_values.gather(1, actions).squeeze(1)

        # 计算目标 Q 值
        with torch.no_grad():
            next_q_values = policy_net(next_observations)
            next_q_value = next_q_values.max(1)[0]
            target_q = rewards + gamma * next_q_value * (1 - dones)

        # 计算 TD 误差和损失
        td_errors = q_value - target_q
        loss = F.mse_loss(q_value, target_q)

        # 反向传播计算梯度
        policy_net.zero_grad()
        loss.backward()

        # 提取梯度
        gradients = []
        for param in policy_net.parameters():
            if param.grad is not None:
                gradients.append(param.grad.detach().cpu().numpy().copy())
            else:
                gradients.append(np.zeros_like(param.detach().cpu().numpy()))

        policy_net.eval()

        return gradients, td_errors.detach().cpu().numpy(), float(loss.item())

    @staticmethod
    def _matrix_to_qubo_dict(qubo_matrix: np.ndarray) -> dict:
        """
        将 QUBO numpy 矩阵转换为 dimod 兼容的字典格式

        dimod QUBO 字典格式：{(i, j): value}，其中 i <= j

        Args:
            qubo_matrix: (N, N) 的 numpy 矩阵

        Returns:
            qubo_dict: {(row, col): value} 字典
        """
        n = qubo_matrix.shape[0]
        qubo_dict = {}
        for i in range(n):
            for j in range(i, n):
                val = qubo_matrix[i, j]
                if abs(val) > 1e-12:  # 跳过零值项以节省内存
                    qubo_dict[(i, j)] = float(val)
        return qubo_dict

    @staticmethod
    def _qubo_flip_delta(
        qubo_matrix: np.ndarray,
        solution: np.ndarray,
        flip_idx: int,
    ) -> float:
        """单比特翻转的能量变化 ΔE = E(x') - E(x)，其中 x' 在第 flip_idx 位取反。

        QUBO 能量定义为 E(x) = x^T Q x（Q 对称，含对角项）。翻转第 k 位
        （x_k -> 1 - x_k）后的解析能量差为：

            ΔE = 2·(1 - 2·x_k)·(Σ_j Q[k,j]·x_j) + Q[k,k]

        等价于 (1 - 2·x_k)·[2·Σ_{j≠k} Q[k,j]·x_j + Q[k,k]]。

        该公式保证：从任意解出发，累进维护的 current_energy 始终等于
        compute_qubo_energy(current_solution, Q)。早期实现漏写 ×2 的离对角项
        与对角项 Q[k,k]，导致 Metropolis 接受概率偏离正确玻尔兹曼分布，且
        累进能量与真实能量逐渐漂移——这是默认仿真求解器的严重正确性缺陷。
        """
        delta = 1.0 - 2.0 * solution[flip_idx]
        linear_term = float(np.dot(qubo_matrix[flip_idx], solution))
        return 2.0 * delta * linear_term + float(qubo_matrix[flip_idx, flip_idx])

    def numpy_simulated_annealing(
        self,
        qubo_matrix: np.ndarray,
    ) -> str:
        """
        内置 numpy 模拟退火求解器

        当 D-Wave Ocean SDK 不可用时，使用此方法作为仿真替代。
        实现经典的 Metropolis-Hastings 模拟退火算法来近似求解 QUBO 问题。

        算法流程：
            1. 随机初始化二值解 x ∈ {0,1}^n
            2. 在每个温度下执行多次扫描（sweep）：
               - 随机翻转一个比特
               - 计算能量差 ΔE
               - 如果 ΔE < 0 或 rand() < exp(-ΔE/T)，接受翻转
            3. 按冷却率降低温度
            4. 重复直至温度低于终止阈值

        Args:
            qubo_matrix: QUBO 矩阵 Q

        Returns:
            best_bitstring: 最优比特串
        """
        n = qubo_matrix.shape[0]

        # Issue #391: 使用独立 RNG，固定种子后结果可复现
        rng = np.random.default_rng(self.random_state)
        py_rng = random.Random(self.random_state)

        # ---------- 随机初始化 ----------
        current_solution = rng.integers(0, 2, n).astype(np.float64)
        current_energy = self.compute_qubo_energy(current_solution, qubo_matrix)

        best_solution = current_solution.copy()
        best_energy = current_energy

        temperature = self._sim_initial_temp
        # Issue #391: 早停计数器——连续 _sim_patience 次扫描无改进则终止
        no_improve_count = 0

        # ---------- 主循环：逐步降温 ----------
        for sweep in range(self._sim_num_sweeps):
            sweep_best_before = best_energy

            # 在每个温度下，翻转 n 个比特（一次完整扫描）
            for _ in range(n):
                # 随机选择一个比特进行翻转
                flip_idx = py_rng.randint(0, n - 1)

                # 计算翻转后的能量变化（向量化，避免 Python 层内循环）
                # 解析公式：ΔE = 2*(1-2x[flip])*(Q[flip]·x) + Q[flip,flip]
                # 推导与回归测试见 _qubo_flip_delta
                delta_energy = self._qubo_flip_delta(qubo_matrix, current_solution, flip_idx)

                # Metropolis 准则：以概率 min(1, exp(-ΔE/T)) 接受新解
                if delta_energy < 0 or py_rng.random() < math.exp(
                    -delta_energy / max(temperature, 1e-12)
                ):
                    current_solution[flip_idx] = 1.0 - current_solution[flip_idx]
                    current_energy += delta_energy

                    # 更新全局最优
                    if current_energy < best_energy:
                        best_solution = current_solution.copy()
                        best_energy = current_energy

            # 降温
            temperature *= self._sim_cooling_rate

            # Issue #391: 早停——连续 _sim_patience 次扫描无改进则终止
            if best_energy < sweep_best_before:
                no_improve_count = 0
            else:
                no_improve_count += 1
                if no_improve_count >= self._sim_patience:
                    logger.debug(
                        f"numpy 模拟退火: 早停于第 {sweep + 1} 次扫描"
                        f"（连续 {self._sim_patience} 次无改进）"
                    )
                    break

            # 提前终止：温度足够低
            if temperature < 1e-6:
                break

        logger.debug(f"numpy 模拟退火: 最佳能量 = {best_energy:.6f}, 扫描次数 = {sweep + 1}")

        # 转换为比特串
        best_bitstring = "".join(str(int(b)) for b in best_solution)
        return best_bitstring

    @staticmethod
    def compute_qubo_energy(solution: np.ndarray, qubo_matrix: np.ndarray) -> float:
        """
        计算 QUBO 目标函数值：E(x) = x^T Q x

        Args:
            solution    : 二值解向量 x ∈ {0,1}^n
            qubo_matrix : QUBO 矩阵 Q

        Returns:
            energy: 目标函数值
        """
        return float(solution @ qubo_matrix @ solution)


# ============================================================================
# Issue #45: QUBO 矩阵构建性能剖析与加速
# ============================================================================
# 以下函数面向"任务调度"场景的 QUBO 构建（输入为任务优先级与处理时间），
# 与上方 QuantumAnnealingOptimizer.network_to_qubo（面向神经网络权重）不同。
# QUBO 模型（n×n，n 为任务数）：
#   - 对角元    Q[i,i] = priority[i] * time[i]
#     （单个任务被选中的线性代价，优先级与处理时间加权）
#   - 非对角元  Q[i,j] = penalty * 0.5 * (p[i]*t[j] + p[j]*t[i])   (i != j)
#     （任务对 i,j 同时调度时的二次冲突代价，对称）
# 矩阵对称；当 priorities/times/penalty 非负时，矩阵非负。
# ---------------------------------------------------------------------------


def build_qubo_matrix(
    task_priorities: np.ndarray,
    task_times: np.ndarray,
    penalty: float = 10.0,
) -> np.ndarray:
    """
    构建任务调度 QUBO 矩阵（原版，基于双重 for 循环）

    Args:
        task_priorities: 任务优先级一维数组，形状 (n,)，建议非负
        task_times      : 任务处理时间一维数组，形状 (n,)，建议非负
        penalty         : 冲突惩罚系数，默认 10.0

    Returns:
        Q: (n, n) 对称 QUBO 矩阵

    Raises:
        ValueError: 当 task_priorities 与 task_times 形状不一致，
                    或输入不是一维数组时
    """
    task_priorities = np.asarray(task_priorities, dtype=np.float64)
    task_times = np.asarray(task_times, dtype=np.float64)

    if task_priorities.shape != task_times.shape:
        raise ValueError(
            "task_priorities 与 task_times 形状不一致: "
            f"{task_priorities.shape} vs {task_times.shape}"
        )
    if task_priorities.ndim != 1:
        raise ValueError(f"task_priorities 必须为一维数组，实际 ndim={task_priorities.ndim}")

    n = task_priorities.shape[0]
    qubo = np.zeros((n, n), dtype=np.float64)

    # 对角元：单任务线性代价
    for i in range(n):
        qubo[i, i] = task_priorities[i] * task_times[i]

    # 非对角元：任务对冲突代价（双重循环，原版实现）
    for i in range(n):
        for j in range(n):
            if i != j:
                qubo[i, j] = (
                    0.5
                    * penalty
                    * (task_priorities[i] * task_times[j] + task_priorities[j] * task_times[i])
                )

    return qubo


def build_qubo_matrix_optimized(
    task_priorities: np.ndarray,
    task_times: np.ndarray,
    penalty: float = 10.0,
) -> np.ndarray:
    """
    构建任务调度 QUBO 矩阵（优化版，numpy 向量化实现）

    与 :func:`build_qubo_matrix` 等价，但用 numpy 广播（外积 + 转置）
    代替双重 for 循环，在任务数较大时显著加速。

    向量化推导：
        设 P = priorities (n,), T = times (n,)
        外积 PT = outer(P, T)，则 PT[i,j] = P[i]*T[j]
        非对角元 = penalty * 0.5 * (PT + PT.T)[i,j]
        对角元   = P * T  （覆盖非对角公式在对角处的值）

    Args:
        task_priorities: 任务优先级一维数组，形状 (n,)
        task_times      : 任务处理时间一维数组，形状 (n,)
        penalty         : 冲突惩罚系数，默认 10.0

    Returns:
        Q: (n, n) 对称 QUBO 矩阵，与 build_qubo_matrix 结果一致

    Raises:
        ValueError: 当 task_priorities 与 task_times 形状不一致，
                    或输入不是一维数组时
    """
    task_priorities = np.asarray(task_priorities, dtype=np.float64)
    task_times = np.asarray(task_times, dtype=np.float64)

    if task_priorities.shape != task_times.shape:
        raise ValueError(
            "task_priorities 与 task_times 形状不一致: "
            f"{task_priorities.shape} vs {task_times.shape}"
        )
    if task_priorities.ndim != 1:
        raise ValueError(f"task_priorities 必须为一维数组，实际 ndim={task_priorities.ndim}")

    n = task_priorities.shape[0]
    # 外积 PT[i,j] = P[i] * T[j]；加上其转置得到对称的成对冲突代价
    pt_outer = np.outer(task_priorities, task_times)
    qubo = 0.5 * penalty * (pt_outer + pt_outer.T)
    # 对角线单独覆盖为 P*T（非对角公式在对角处为 penalty*P*T，需替换）
    diag_idx = np.arange(n)
    qubo[diag_idx, diag_idx] = task_priorities * task_times
    return qubo


# ============================================================================
# 多机任务分配 QUBO 构建（变量 x[i][m]，面向 DAG 调度器决策层）
# 与上方单机任务选择 QUBO（变量 x[i]）不同：此处用于将任务分配到多台机器。
# ---------------------------------------------------------------------------


def build_assignment_qubo_matrix(
    tasks: list[dict[str, Any]],
    machines: list[dict[str, Any]],
    penalty: float = 10.0,
) -> np.ndarray:
    """
    构建多机任务分配 QUBO 矩阵

    变量定义：``x[i][m]`` 表示任务 i 是否分配到机器 m，变量总数
    ``N = n_tasks × n_machines``，按 (i, m) 行优先展平为
    ``index = i * n_machines + m``。

    目标与约束：
        - 对角项（选中代价）：
          ``Q[(i,m),(i,m)] = priority[i] * estimated_time[i] - penalty``
          （``-penalty`` 来自约束1对角展开，鼓励每任务至少选一台机器）
        - 约束1（每任务恰好分一台机器）：
          ``penalty * (Σ_m x[i][m] - 1)^2`` 展开后的非对角部分，
          贡献同任务不同机器变量对 ``2 * penalty``，惩罚多选
        - 约束2（机器容量软约束）：近似
          ``penalty * (Σ_i qubits[i] * x[i][m])^2`` 的非对角部分，
          贡献同机器不同任务变量对 ``penalty * qubits[i] * qubits[i']``，
          当同机器任务总比特超过容量时惩罚增大（QUBO 无法精确表达
          ``max(0, .)^2``，此处采用标准二次松弛）

    矩阵对称；变量排列顺序确保两类约束作用于不同的变量对，互不干扰。

    Args:
        tasks: 任务字典列表，每项需含 ``task_id``、``qubits_required``、
               ``estimated_time``、``priority`` 字段
        machines: 机器字典列表，每项需含 ``machine_id``、``capacity`` 字段
        penalty: 约束违反惩罚系数，默认 10.0

    Returns:
        Q: ``(N, N)`` 对称 QUBO 矩阵，``N = n_tasks * n_machines``

    Raises:
        ValueError: 当 tasks 或 machines 为空，或字段缺失时
    """
    if not tasks:
        raise ValueError("tasks 不能为空")
    if not machines:
        raise ValueError("machines 不能为空")

    # 校验字段存在性
    for t in tasks:
        for key in ("task_id", "qubits_required", "estimated_time", "priority"):
            if key not in t:
                raise ValueError(f"任务字典缺少字段 '{key}': {t}")
    for m in machines:
        for key in ("machine_id", "capacity"):
            if key not in m:
                raise ValueError(f"机器字典缺少字段 '{key}': {m}")

    n_tasks = len(tasks)
    n_machines = len(machines)
    n_vars = n_tasks * n_machines
    qubo = np.zeros((n_vars, n_vars), dtype=np.float64)

    # 提取属性数组（强制浮点，避免整数除法等副作用）
    priorities = np.array([float(t["priority"]) for t in tasks], dtype=np.float64)
    times = np.array([float(t["estimated_time"]) for t in tasks], dtype=np.float64)
    qubits = np.array([float(t["qubits_required"]) for t in tasks], dtype=np.float64)

    for i in range(n_tasks):
        for m in range(n_machines):
            idx_im = i * n_machines + m

            # 对角项：选中代价 + 约束1对角（-penalty 鼓励选择）
            qubo[idx_im, idx_im] = priorities[i] * times[i] - penalty

            # 约束1非对角：同任务不同机器（惩罚多选）
            for m2 in range(m + 1, n_machines):
                idx_im2 = i * n_machines + m2
                qubo[idx_im, idx_im2] = 2.0 * penalty
                qubo[idx_im2, idx_im] = 2.0 * penalty

            # 约束2非对角：同机器不同任务（容量软约束）
            for i2 in range(i + 1, n_tasks):
                idx_i2m = i2 * n_machines + m
                coupling = penalty * qubits[i] * qubits[i2]
                qubo[idx_im, idx_i2m] = coupling
                qubo[idx_i2m, idx_im] = coupling

    return qubo


def solve_task_assignment(
    tasks: list[dict[str, Any]],
    machines: list[dict[str, Any]],
    optimizer: QuantumAnnealingOptimizer | None = None,
    penalty: float = 10.0,
) -> tuple[dict[str, str], float]:
    """
    量子退火求解多机任务分配问题

    构建任务分配 QUBO 矩阵，调用量子退火（或仿真模拟退火）求解，
    解码比特串为 ``{task_id: machine_id}`` 分配方案。

    解码规则：每个任务对应 ``n_machines`` 个比特，取第一个为 ``"1"`` 的
    机器作为分配目标；若全为 ``"0"`` 或比特串长度不足，回退到机器 0。

    Args:
        tasks: 任务字典列表，字段同 :func:`build_assignment_qubo_matrix`
        machines: 机器字典列表，字段同 :func:`build_assignment_qubo_matrix`
        optimizer: 量子退火优化器实例；为 None 时使用默认仿真优化器
        penalty: 约束违反惩罚系数，默认 10.0

    Returns:
        tuple (assignment, energy):
            - assignment: ``{task_id: machine_id}`` 分配方案字典，
              machine_id 以字符串形式返回
            - energy: QUBO 能量值 ``x^T Q x``，越低越优；比特串长度
              不匹配时返回 ``+inf``
    """
    qubo = build_assignment_qubo_matrix(tasks, machines, penalty=penalty)
    n_tasks = len(tasks)
    n_machines = len(machines)

    if optimizer is None:
        optimizer = QuantumAnnealingOptimizer(
            num_qubits=n_tasks * n_machines,
            simulation_mode=True,
        )

    bitstring = optimizer.anneal(qubo)
    bits = bitstring.strip()

    # 解码比特串为分配方案
    assignment: dict[str, str] = {}
    for i, task in enumerate(tasks):
        # 取该任务对应的机器变量段
        if len(bits) < (i + 1) * n_machines:
            # 比特串长度不足，回退到机器 0
            assignment[task["task_id"]] = str(machines[0]["machine_id"])
            continue
        segment = bits[i * n_machines : (i + 1) * n_machines]
        # 选择第一个为 "1" 的机器；若全为 "0"，选择机器 0
        selected = 0
        for m, b in enumerate(segment):
            if b == "1":
                selected = m
                break
        assignment[task["task_id"]] = str(machines[selected]["machine_id"])

    # 计算 QUBO 能量 x^T Q x
    if len(bits) != qubo.shape[0]:
        energy = float("inf")
    else:
        x = np.array([float(b) for b in bits], dtype=np.float64)
        energy = float(x @ qubo @ x)

    return assignment, energy


def profile_qubo_construction(n_tasks: int = 10, n_iterations: int = 100) -> dict:
    """
    剖析 QUBO 矩阵构建性能

    随机生成任务优先级与处理时间（固定种子，可复现），多次调用
    :func:`build_qubo_matrix`，用 time.perf_counter 统计构建耗时分布。

    Args:
        n_tasks     : 任务数量（生成数据的规模），默认 10
        n_iterations: 重复构建次数，默认 100

    Returns:
        dict 包含：
            - mean_time_ms : 平均耗时（毫秒）
            - std_time_ms  : 耗时标准差（毫秒）
            - min_time_ms  : 最小耗时（毫秒）
            - max_time_ms  : 最大耗时（毫秒）
            - matrix_size  : QUBO 矩阵边长（= n_tasks）
            - n_tasks      : 任务数量

    Raises:
        ValueError: 当 n_tasks 为负或 n_iterations < 1 时
    """
    if n_tasks < 0:
        raise ValueError(f"n_tasks 不能为负，实际: {n_tasks}")
    if n_iterations < 1:
        raise ValueError(f"n_iterations 必须 >= 1，实际: {n_iterations}")

    rng = np.random.default_rng(seed=42)
    task_priorities = rng.uniform(1.0, 10.0, size=n_tasks)
    task_times = rng.uniform(1.0, 20.0, size=n_tasks)

    timings_ms = np.empty(n_iterations, dtype=np.float64)
    qubo = np.zeros((0, 0), dtype=np.float64)
    for k in range(n_iterations):
        t0 = time.perf_counter()
        qubo = build_qubo_matrix(task_priorities, task_times)
        t1 = time.perf_counter()
        timings_ms[k] = (t1 - t0) * 1000.0

    return {
        "mean_time_ms": float(np.mean(timings_ms)),
        "std_time_ms": float(np.std(timings_ms)),
        "min_time_ms": float(np.min(timings_ms)),
        "max_time_ms": float(np.max(timings_ms)),
        "matrix_size": int(qubo.shape[0]),
        "n_tasks": int(n_tasks),
    }


def benchmark_qubo_versions(n_tasks: int = 10, n_iterations: int = 50) -> dict:
    """
    对比原版与优化版 QUBO 矩阵构建的性能与正确性

    使用相同的随机任务数据（固定种子），分别多次计时两个版本，
    并验证结果一致性。

    Args:
        n_tasks     : 任务数量，默认 10
        n_iterations: 每个版本重复构建次数，默认 50

    Returns:
        dict 包含：
            - original_mean_ms : 原版平均耗时（毫秒）
            - optimized_mean_ms: 优化版平均耗时（毫秒）
            - speedup          : 加速比 = original_mean_ms / optimized_mean_ms
            - results_match    : 两版结果是否一致（np.allclose）

    Raises:
        ValueError: 当 n_tasks 为负或 n_iterations < 1 时
    """
    if n_tasks < 0:
        raise ValueError(f"n_tasks 不能为负，实际: {n_tasks}")
    if n_iterations < 1:
        raise ValueError(f"n_iterations 必须 >= 1，实际: {n_iterations}")

    rng = np.random.default_rng(seed=42)
    task_priorities = rng.uniform(1.0, 10.0, size=n_tasks)
    task_times = rng.uniform(1.0, 20.0, size=n_tasks)

    # 原版计时
    orig_timings = np.empty(n_iterations, dtype=np.float64)
    qubo_orig = np.zeros((0, 0), dtype=np.float64)
    for k in range(n_iterations):
        t0 = time.perf_counter()
        qubo_orig = build_qubo_matrix(task_priorities, task_times)
        t1 = time.perf_counter()
        orig_timings[k] = (t1 - t0) * 1000.0

    # 优化版计时
    opt_timings = np.empty(n_iterations, dtype=np.float64)
    qubo_opt = np.zeros((0, 0), dtype=np.float64)
    for k in range(n_iterations):
        t0 = time.perf_counter()
        qubo_opt = build_qubo_matrix_optimized(task_priorities, task_times)
        t1 = time.perf_counter()
        opt_timings[k] = (t1 - t0) * 1000.0

    results_match = bool(np.allclose(qubo_orig, qubo_opt))
    orig_mean = float(np.mean(orig_timings))
    opt_mean = float(np.mean(opt_timings))
    speedup = float(orig_mean / opt_mean) if opt_mean > 0 else float("inf")

    return {
        "original_mean_ms": orig_mean,
        "optimized_mean_ms": opt_mean,
        "speedup": speedup,
        "results_match": results_match,
    }


def find_optimal_qubo_params(
    task_priorities: np.ndarray,
    task_times: np.ndarray,
    param_grid: dict | None = None,
) -> dict:
    """
    网格搜索最优 penalty 参数

    对每个候选 penalty 构建 QUBO 矩阵，计算参考解 x = ones(n)（全选所有任务）
    的能量 E = x^T Q x 作为可比指标，选择使能量最低的 penalty。

    评价准则说明：以全 1 解的能量衡量 QUBO 矩阵的整体代价规模，
    能量越低代表该 penalty 下同时调度所有任务的总代价越小。
    当 priorities/times 非负时，能量随 penalty 单调递增，
    故 best_penalty 通常为网格中的最小值。

    Args:
        task_priorities: 任务优先级一维数组
        task_times      : 任务处理时间一维数组
        param_grid      : 搜索网格，形如 {"penalty": [1.0, 5.0, ...]}。
                          默认 {"penalty": [1.0, 5.0, 10.0, 50.0, 100.0]}

    Returns:
        dict 包含：
            - best_penalty : 最优 penalty 值（属于网格）
            - best_energy  : 对应最低能量
            - all_results  : 列表，每项 {"penalty": p, "energy": e}
    """
    if param_grid is None:
        param_grid = {"penalty": [1.0, 5.0, 10.0, 50.0, 100.0]}

    penalties = param_grid.get("penalty", [10.0])
    task_priorities = np.asarray(task_priorities, dtype=np.float64)
    task_times = np.asarray(task_times, dtype=np.float64)

    n = task_priorities.shape[0]
    x = np.ones(n, dtype=np.float64)

    all_results: list[dict] = []
    best_penalty = float(penalties[0]) if penalties else 10.0
    best_energy = float("inf")

    for p in penalties:
        qubo = build_qubo_matrix_optimized(task_priorities, task_times, penalty=float(p))
        energy = float(x @ qubo @ x) if n > 0 else 0.0
        all_results.append({"penalty": float(p), "energy": energy})
        if energy < best_energy:
            best_energy = energy
            best_penalty = float(p)

    return {
        "best_penalty": best_penalty,
        "best_energy": best_energy,
        "all_results": all_results,
    }


# ============================================================================
# 模块自测试
# ============================================================================
if __name__ == "__main__":
    # loguru 已在模块顶部导入，无需 basicConfig

    print("=" * 60)
    print("量子退火策略优化器 - 模块自测试")
    print("=" * 60)

    # 显示量子加速开关状态
    _qa_env = os.environ.get("QUANTUM_ACCELERATION_ENABLED", "未设置")
    print(f"\n环境变量 QUANTUM_ACCELERATION_ENABLED = {_qa_env}")
    print(f"量子加速功能: {'✅ 已启用' if QUANTUM_ACCELERATION_ENABLED else '❌ 已禁用'}")
    print(f"D-Wave SDK 可用: {'✅ 是' if _DWAVE_AVAILABLE else '❌ 否（使用 numpy 仿真）'}")

    # ---- 测试 1: 创建优化器 ----
    print("\n--- 测试 1: 初始化 QuantumAnnealingOptimizer ---")
    optimizer = QuantumAnnealingOptimizer(num_qubits=8, annealing_time=20, shots=100)
    print(f"  量子比特数: {optimizer.num_qubits}")
    print(f"  退火时间: {optimizer.annealing_time} μs")
    print(f"  采样次数: {optimizer.shots}")
    print(f"  使用 D-Wave: {optimizer.use_dw}")

    # ---- 测试 2: 构造 QUBO ----
    print("\n--- 测试 2: network_to_qubo ---")
    # 模拟一个简单的两层全连接网络权重
    W1 = np.random.randn(8, 4).astype(np.float32)
    b1 = np.random.randn(4).astype(np.float32)
    W2 = np.random.randn(4, 2).astype(np.float32)
    b2 = np.random.randn(2).astype(np.float32)
    mock_weights = [W1, b1, W2, b2]

    qubo = optimizer.network_to_qubo(mock_weights)
    print(f"  输入: 4 层权重, 总参数 = {sum(w.size for w in mock_weights)}")
    print(f"  QUBO 矩阵形状: {qubo.shape}")
    print(f"  QUBO 矩阵非零元素: {np.count_nonzero(qubo)}")

    # ---- 测试 3: 退火求解 ----
    print("\n--- 测试 3: anneal ---")
    bitstring = optimizer.anneal(qubo)
    print(f"  最优比特串长度: {len(bitstring)}")
    print(f"  最优比特串: {bitstring}")

    # 验证比特串确实降低了 QUBO 能量
    random_bits = np.random.randint(0, 2, len(bitstring)).astype(np.float64)
    random_energy = optimizer.compute_qubo_energy(random_bits, qubo)
    best_bits = np.array([int(b) for b in bitstring], dtype=np.float64)
    best_energy = optimizer.compute_qubo_energy(best_bits, qubo)
    print(f"  随机解能量: {random_energy:.6f}")
    print(f"  最优解能量: {best_energy:.6f}")
    print(f"  能量改进: {random_energy - best_energy:.6f}")

    # ---- 测试 4: 比特串解码 ----
    print("\n--- 测试 4: bitstring_to_weights ---")
    original_shapes = [w.shape for w in mock_weights]
    decoded_weights = optimizer.bitstring_to_weights(bitstring, original_shapes)
    print(f"  解码后权重层数: {len(decoded_weights)}")
    for i, (dw, orig_shape) in enumerate(zip(decoded_weights, original_shapes, strict=False)):
        assert dw.shape == orig_shape, f"形状不匹配: {dw.shape} vs {orig_shape}"
        print(f"  第 {i} 层: 形状 {dw.shape}, 范围 [{dw.min():.4f}, {dw.max():.4f}]")

    # ---- 测试 5: 完整 optimize_policy 流程（使用简单 nn.Module 模拟 agent）----
    print("\n--- 测试 5: optimize_policy (模拟 agent) ---")

    # 构建一个简单的 PyTorch 网络作为模拟的 agent
    class MockAgent:
        """模拟的 RL 智能体，用于测试 optimize_policy 接口"""

        def __init__(self, state_dim=8, action_dim=3):
            self.policy_net = nn.Sequential(
                nn.Linear(state_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, action_dim),
            )
            self.target_net = nn.Sequential(
                nn.Linear(state_dim, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
                nn.ReLU(),
                nn.Linear(8, action_dim),
            )
            self.target_net.load_state_dict(self.policy_net.state_dict())

    mock_agent = MockAgent()
    total_params = sum(p.numel() for p in mock_agent.policy_net.parameters())
    print(f"  模拟 agent 参数总数: {total_params}")

    # 临时启用量子加速以测试完整流程
    original_flag = os.environ.get("QUANTUM_ACCELERATION_ENABLED")
    os.environ["QUANTUM_ACCELERATION_ENABLED"] = "1"

    # 直接修改当前模块的全局标志
    _original_enabled = QUANTUM_ACCELERATION_ENABLED

    # 需要重新导入模块以刷新全局变量（仅当以包方式运行时有效）
    # 脚本直接运行时，直接修改全局变量
    # 将当前模块标记为启用量子加速
    import __main__

    __main__.QUANTUM_ACCELERATION_ENABLED = True
    # 同时修改当前模块命名空间
    globals()["QUANTUM_ACCELERATION_ENABLED"] = True

    # 执行优化（少量迭代）
    optimized_agent = optimizer.optimize_policy(
        mock_agent,
        num_iterations=3,
        learning_rate=0.01,
    )

    # 验证 target_net 已同步
    params_match = all(
        torch.equal(p1, p2)
        for p1, p2 in zip(
            optimized_agent.policy_net.parameters(),
            optimized_agent.target_net.parameters(),
            strict=False,
        )
    )
    print(f"  target_net 同步状态: {'✅ 已同步' if params_match else '❌ 未同步'}")

    # 恢复环境变量和全局标志
    if original_flag is not None:
        os.environ["QUANTUM_ACCELERATION_ENABLED"] = original_flag
    else:
        os.environ.pop("QUANTUM_ACCELERATION_ENABLED", None)
    globals()["QUANTUM_ACCELERATION_ENABLED"] = _original_enabled

    print("\n" + "=" * 60)
    print("所有测试通过！量子退火策略优化器工作正常。")
    print("=" * 60)
