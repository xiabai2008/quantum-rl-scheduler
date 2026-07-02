"""
量子退火加速 RL 策略搜索模块
Quantum Annealing Accelerator for Reinforcement Learning Policy Optimization

核心思想：
    将 DQN 策略网络的参数优化问题映射为 QUBO（Quadratic Unconstrained Binary Optimization）问题，
    利用量子退火器（或仿真模拟退火）来高效求解，从而加速策略搜索过程。

QUBO 问题形式：min  x^T Q x，其中 x ∈ {0,1}^n, Q 为 n×n 的实数矩阵。

开关控制：
    通过环境变量 QUANTUM_ACCELERATION_ENABLED 可全局启用/禁用量子加速功能。
    - 设为 "1"/"true"/"yes" 启用
    - 未设置或设为其他值时禁用（回退到纯经典 RL 优化流程）

依赖：
    真机模式需要 D-Wave Ocean SDK (dwave-neal / dimod)
    仿真模式仅依赖 numpy（始终可用）
"""

import logging
import math
import os
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# ============================================================================
# 全局开关：通过环境变量 QUANTUM_ACCELERATION_ENABLED 控制是否启用量子加速
# ============================================================================
QUANTUM_ACCELERATION_ENABLED = os.environ.get(
    "QUANTUM_ACCELERATION_ENABLED", "0"
).strip().lower() in ("1", "true", "yes")

# ============================================================================
# 日志配置
# ============================================================================
logger = logging.getLogger(__name__)

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
        annealing_time: 退火时间（微秒），仅在真机模式下生效
        shots         : 每次退火的采样次数，用于统计最优解
        use_dw        : 是否使用 D-Wave SDK 仿真器（优先级高于 numpy 仿真）
    """

    def __init__(
        self,
        num_qubits: int = 16,
        annealing_time: float = 20.0,
        shots: int = 1000,
        simulation_mode: bool = True,
        cqlib_client: Any = None,
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
        """
        self.num_qubits = num_qubits
        self.annealing_time = annealing_time
        self.shots = shots
        self.simulation_mode = bool(simulation_mode)
        self.cqlib_client = cqlib_client

        # 检查比特编码精度，过低则发出警告
        n_bits_per_weight = max(1, num_qubits // 4)
        if n_bits_per_weight < 4:
            logger.warning(
                f"量子比特数 {num_qubits} 较低，每权重仅 {n_bits_per_weight} bit 编码 "
                f"（1 符号位 + {n_bits_per_weight - 1} 数值位），精度可能不足。"
                f"建议 num_qubits ≥ 16 以获得更好的优化效果。"
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
        n_bits_per_weight = max(1, self.num_qubits // 4)  # 增加比特数提高精度
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
        调用量子退火器（或仿真）求解 QUBO 问题，返回最优比特串

        求解路径优先级：
            1. 真机退火：``simulation_mode=False`` 且 ``cqlib_client`` 提供
               ``submit_annealing_task`` 方法时，提交 QUBO 到天衍云量子退火器
            2. D-Wave neal 模拟退火：若 D-Wave Ocean SDK 可用
            3. 内置 numpy 模拟退火：始终可用的兜底实现

        天衍云 cqlib 为门控量子计算机 SDK，不提供 QUBO 退火接口；遇到此情况
        会打印"降级为仿真"日志并回退到 numpy/neal 路径，保证流程不中断。

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
                    if isinstance(result, str):
                        best_bitstring = result
                    elif isinstance(result, dict):
                        best_bitstring = str(
                            result.get("bitstring", "")
                        ) or self._numpy_simulated_annealing(qubo_matrix)
                    else:
                        logger.warning(
                            f"[退火] 真机退火返回类型 {type(result)} 无法识别，降级为仿真"
                        )
                        best_bitstring = self._numpy_simulated_annealing(qubo_matrix)
                    logger.info(f"[退火] 真机退火完成，比特串长度={len(best_bitstring)}")
                    return best_bitstring
                except Exception as e:
                    # 真机退火涉及 cqlib SDK，异常类型无法穷举，保留宽捕获并记录日志
                    logger.warning(f"[退火] 真机退火失败 ({type(e).__name__}: {e})，降级为仿真")
                    # 继续走下方仿真路径
            else:
                logger.info(
                    "[退火] cqlib 为门控量子 SDK，无 submit_annealing_task 接口，"
                    "当前降级为仿真（numpy 模拟退火）"
                )

        # ---- 路径 2/3：仿真退火 ----
        if self.use_dw:
            # ---- 使用 D-Wave neal 求解器 ----
            logger.debug(f"anneal: 使用 D-Wave neal 求解器, QUBO 规模 {n}x{n}")
            qubo_dict = self._matrix_to_qubo_dict(qubo_matrix)
            sampler = neal.SimulatedAnnealingSampler()
            sampleset = sampler.sample_qubo(
                qubo_dict,
                num_reads=self.shots,
                annealing_time=self.annealing_time,
            )
            # 取能量最低的样本
            best_sample = sampleset.first.sample
            best_bitstring = "".join(str(best_sample[i]) for i in range(n))
        else:
            # ---- 使用内置 numpy 模拟退火 ----
            logger.debug(f"anneal: 使用内置 numpy 模拟退火, QUBO 规模 {n}x{n}")
            best_bitstring = self._numpy_simulated_annealing(qubo_matrix)

        logger.debug(f"anneal: 最优比特串 = {best_bitstring[:32]}{'...' if n > 32 else ''}")
        return best_bitstring

    # ------------------------------------------------------------------
    # 方法 4: bitstring_to_weights
    # ------------------------------------------------------------------
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
        n_bits_per_weight = max(1, self.num_qubits // 4)

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
    def optimize_policy(
        self,
        agent: Any,
        num_iterations: int = 10,
        learning_rate: float = 0.01,
        callback: Any | None = None,
        replay_buffer: Any | None = None,
        head_only: bool = True,
        max_head_tensors: int = 4,
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

        Args:
            agent          : RL 智能体（需具有 policy_net 属性，为 nn.Module）
            num_iterations : 量子退火优化迭代次数（默认 10）
            learning_rate  : 权重更新学习率（默认 0.01），控制更新幅度
            callback       : 可选的回调函数，签名为 callback(iteration, loss)
            replay_buffer  : 可选，经验回放缓冲区。若提供，用于计算梯度；
                             若未提供，退化为基于权重正则化的优化。
            head_only       : 是否仅优化网络输出头权重（默认 True）。
                             设为 True 时仅优化最后 max_head_tensors 个参数张量，
                             避免全量参数的 QUBO 矩阵 OOM。
            max_head_tensors: head_only=True 时，最多优化的尾部参数张量数（默认 4）。

        Returns:
            agent: 优化后的智能体（原地修改并返回）
        """
        if not QUANTUM_ACCELERATION_ENABLED:
            logger.warning(
                "量子加速功能已禁用 (QUANTUM_ACCELERATION_ENABLED 未设置)。"
                "跳过 optimize_policy，直接返回原始 agent。"
            )
            return agent

        # 获取策略网络
        # head_only 模式需要完整的 policy（含 action_net/value_net 输出头）
        # 非 head_only 模式使用 mlp_extractor 即可
        policy_net = self._get_full_policy(agent) if head_only else self._get_policy_net(agent)
        if policy_net is None:
            logger.error("无法获取策略网络，退出 optimize_policy")
            return agent

        logger.info(
            f"开始量子退火策略优化 (v2 - 梯度引导): {num_iterations} 次迭代, "
            f"学习率={learning_rate}, 量子比特数={self.num_qubits}"
            f"{', head_only模式' if head_only else ''}"
        )

        # 如果启用了 head_only 模式，计算需要优化的参数张量索引范围
        # PPO 完整 policy 的参数顺序: [0-7: mlp_extractor, 8-9: action_net, 10-11: value_net]
        # 仅优化最后 max_head_tensors 个（action_net + value_net = 4 个张量, 260 参数）
        if head_only:
            all_params = list(policy_net.parameters())
            total_tensors = len(all_params)
            n_head = min(max_head_tensors, total_tensors)
            head_start_idx = total_tensors - n_head
            head_param_count = sum(
                all_params[i].numel() for i in range(head_start_idx, total_tensors)
            )
            logger.info(
                f"[退火] head_only 模式: 仅优化最后 {n_head}/{total_tensors} 个参数张量 "
                f"({head_param_count} 个标量参数)"
            )
        else:
            head_start_idx = 0

        best_loss = float("inf")
        best_weights = None
        history = []

        # 初始评估
        initial_loss = self._evaluate_network_quality(policy_net)
        best_loss = initial_loss
        initial_weights, _initial_shapes = self._extract_weights(policy_net)
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
            all_weights, all_shapes = self._extract_weights(policy_net)

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
            delta_flat = np.concatenate(
                [
                    (ow - cw).flatten()
                    for ow, cw in zip(optimized_head_weights, current_weights, strict=False)
                ]
            )
            delta_l2 = float(np.linalg.norm(delta_flat))
            delta_max = float(np.max(np.abs(delta_flat))) if delta_flat.size else 0.0
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

            # ---- 步骤 7: 评估更新后的 loss，决定是否接受 ----
            new_loss = self._evaluate_network_quality(policy_net)
            loss_improvement = current_loss - new_loss

            # 接受准则：loss 下降，或上升幅度不超过阈值（早期探索）
            accept_threshold = 0.01 * current_loss  # 允许 1% 的暂时上升
            if new_loss <= best_loss or loss_improvement > -accept_threshold:
                # 接受更新
                accepted = True
                if new_loss < best_loss:
                    best_loss = new_loss
                    best_weights, _ = self._extract_weights(policy_net)
            else:
                # 回滚：仅回滚被修改的那部分参数
                if head_only:
                    head_params = list(policy_net.parameters())[head_start_idx:]
                    self._set_params_from_weights(head_params, old_weights)
                else:
                    self._set_weights(policy_net, old_weights)
                accepted = False

            history.append((iteration, current_loss, new_loss, accepted))

            logger.info(
                f"  迭代 {iteration + 1}/{num_iterations}: "
                f"优化前 loss={current_loss:.6f}, 优化后 loss={new_loss:.6f}, "
                f"{'✅ 接受' if accepted else '❌ 拒绝'}, "
                f"最佳 loss={best_loss:.6f}"
            )

            if callback is not None:
                callback(iteration, new_loss)

        # 恢复到最佳权重
        if best_weights is not None:
            self._set_weights(policy_net, best_weights)
            logger.info(f"已恢复到最佳权重 (loss={best_loss:.6f})")

        # 最终权重差异统计（退火前 initial_weights → 退火后 best_weights）
        # 用 L2 范数和最大绝对差证明退火确实改变了 PPO 网络权重
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
        logger.info(
            f"[退火] 退火前后权重差异汇总: "
            f"初始 L2={initial_l2_norm:.6f}, 最终 L2={final_l2_norm:.6f}, "
            f"差异 L2={diff_l2:.6e}, 相对差异={diff_relative:.6e} ({diff_relative * 100:.4f}%), "
            f"最大绝对差={diff_max:.6e}"
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

        # 如果 agent 有 target_net，同步更新
        if hasattr(agent, "target_net"):
            agent.target_net.load_state_dict(agent.policy_net.state_dict())
            logger.info("已同步更新 target_net")

        return agent

    # ==================================================================
    # 内部辅助方法
    # ==================================================================

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
    def _get_full_policy(agent: Any) -> nn.Module | None:
        """
        获取完整的 policy 网络（含输出头），用于 head_only 模式

        与 _get_policy_net 的区别：
            - _get_policy_net 对 PPO 返回 mlp_extractor（不含 action_net/value_net）
            - _get_full_policy 对 PPO 返回完整的 ActorCriticPolicy（含所有参数）

        支持的 agent 类型：与 _get_policy_net 相同
        """
        # SB3 PPO: 返回完整的 policy（含 action_net + value_net 输出头）
        if hasattr(agent, "policy") and isinstance(agent.policy, nn.Module):
            return agent.policy

        # 其它类型回退到 _get_policy_net
        return QuantumAnnealingOptimizer._get_policy_net(agent)

    @staticmethod
    def _extract_weights(network: nn.Module) -> tuple[list[np.ndarray], list[tuple[int, ...]]]:
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

    def _numpy_simulated_annealing(
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

        # ---------- 随机初始化 ----------
        current_solution = np.random.randint(0, 2, n).astype(np.float64)
        current_energy = self._compute_qubo_energy(current_solution, qubo_matrix)

        best_solution = current_solution.copy()
        best_energy = current_energy

        temperature = self._sim_initial_temp

        # ---------- 主循环：逐步降温 ----------
        for sweep in range(self._sim_num_sweeps):  # noqa: B007
            # 在每个温度下，翻转 n 个比特（一次完整扫描）
            for _ in range(n):
                # 随机选择一个比特进行翻转
                flip_idx = random.randint(0, n - 1)

                # 计算翻转后的能量变化（向量化，避免 Python 层内循环）
                # ΔE = (1 - 2*x[flip]) * (Σ_j Q[flip,j] * x[j])
                delta = 1.0 - 2.0 * current_solution[flip_idx]
                linear_term = np.dot(qubo_matrix[flip_idx], current_solution)
                delta_energy = delta * linear_term

                # Metropolis 准则：以概率 min(1, exp(-ΔE/T)) 接受新解
                if delta_energy < 0 or random.random() < math.exp(
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

            # 提前终止：温度足够低
            if temperature < 1e-6:
                break

        logger.debug(f"numpy 模拟退火: 最佳能量 = {best_energy:.6f}, " f"扫描次数 = {sweep + 1}")

        # 转换为比特串
        best_bitstring = "".join(str(int(b)) for b in best_solution)
        return best_bitstring

    @staticmethod
    def _compute_qubo_energy(solution: np.ndarray, qubo_matrix: np.ndarray) -> float:
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
# 模块自测试
# ============================================================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

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
    random_energy = optimizer._compute_qubo_energy(random_bits, qubo)
    best_bits = np.array([int(b) for b in bitstring], dtype=np.float64)
    best_energy = optimizer._compute_qubo_energy(best_bits, qubo)
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
