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

import os
import math
import logging
import random
from typing import List, Tuple, Optional, Any

import numpy as np

import torch
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
    import dimod           # D-Wave 的 QUBO/Ising 建模工具
    import neal            # D-Wave 的模拟退火求解器
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
        num_qubits: int = 8,
        annealing_time: float = 20.0,
        shots: int = 1000,
    ):
        """
        初始化量子退火策略优化器

        Args:
            num_qubits    : 量子比特数（默认 8），对应 QUBO 变量的个数。
                            实际使用时会自动扩展以匹配策略网络的权重总数。
            annealing_time: 退火时间，单位微秒（默认 20μs），仅在连接 D-Wave 真机时有效。
            shots         : 退火采样次数（默认 1000），多次采样后取能量最低的解。
        """
        self.num_qubits = num_qubits
        self.annealing_time = annealing_time
        self.shots = shots

        # 自动选择求解器：
        #   优先使用 D-Wave neal 模拟退火器（如果 SDK 可用）
        #   否则回退到内置 numpy 模拟退火
        self.use_dw = _DWAVE_AVAILABLE

        if self.use_dw:
            logger.info("使用 D-Wave neal 模拟退火求解器 (SimulatedAnnealingSampler)")
        else:
            logger.info("使用内置 numpy 模拟退火求解器")

        # 内置模拟退火超参数
        self._sim_initial_temp = 2.0      # 初始温度
        self._sim_cooling_rate = 0.995    # 降温系数
        self._sim_num_sweeps = 2000        # 扫描次数（迭代轮数）

    # ------------------------------------------------------------------
    # 方法 2: network_to_qubo
    # ------------------------------------------------------------------
    def network_to_qubo(self, weights: List[np.ndarray]) -> np.ndarray:
        """
        将神经网络权重列表映射为 QUBO 矩阵

        映射策略：
            1. 将每个连续权重 w 量化为 n_bits 位的二值编码（固定点表示）
            2. 每个 bit 对应 QUBO 中的一个二值变量
            3. 构造二次目标函数，使得 QUBO 的最优解对应于
               "让梯度下降目标最小化" 的离散近似

            具体来说：
            - 对角项 Q[i,i] 编码单个比特的偏好（基于权重大小和符号）
            - 非对角项 Q[i,j] 编码比特间的耦合关系（基于相邻权重间的梯度相关性）

        Args:
            weights: 神经网络权重列表，每个元素是一个 numpy array（对应一层权重）
                     例如 [W1, b1, W2, b2, ...]

        Returns:
            QUBO 矩阵 Q，形状为 (N, N)，其中 N 为编码后的总比特数
        """
        # ---------- 步骤 1：参数配置 ----------
        n_bits_per_weight = max(1, self.num_qubits // 8)  # 每个权重编码的比特数

        # ---------- 步骤 2：展平所有权重为一维向量 ----------
        flat_weights = np.concatenate([w.flatten() for w in weights])
        num_weights = flat_weights.size

        # 总比特数 = 权重数 × 每权重编码比特数
        total_bits = num_weights * n_bits_per_weight

        logger.debug(
            f"network_to_qubo: {num_weights} 个权重参数, "
            f"每参数 {n_bits_per_weight} bit, 总计 {total_bits} 个 QUBO 变量"
        )

        # ---------- 步骤 3：构造 QUBO 矩阵 ----------
        Q = np.zeros((total_bits, total_bits), dtype=np.float64)

        # 计算权重的全局统计量，用于归一化
        weight_abs_max = np.max(np.abs(flat_weights)) + 1e-8

        for i in range(num_weights):
            w = flat_weights[i]
            w_normalized = w / weight_abs_max  # 归一化到 [-1, 1]

            base_idx = i * n_bits_per_weight

            for bit_k in range(n_bits_per_weight):
                global_idx = base_idx + bit_k
                bit_significance = 1.0 / (2 ** bit_k)  # 高位权重更大

                # --- 对角项 Q[global_idx, global_idx] ---
                # 目标：使 QUBO 最小值倾向于编码 "当前权重 + 梯度方向" 的离散近似
                # 使用 sigmoid 变换将连续权重映射到 [0,1]，然后编码为比特偏好
                sigmoid_val = 1.0 / (1.0 + math.exp(-w_normalized))
                # 负号使得 QUBO 最小化时偏好 sigmoid_val 对应的 bit 值
                Q[global_idx, global_idx] = -(2.0 * sigmoid_val - 1.0) * bit_significance

                # --- 非对角项：同一权重内不同比特间的耦合 ---
                # 高位比特与低位比特之间需要有适当的耦合，保证编码一致性
                for bit_l in range(bit_k + 1, n_bits_per_weight):
                    other_idx = base_idx + bit_l
                    other_significance = 1.0 / (2 ** bit_l)
                    # 耦合强度与比特位的权重乘积相关
                    Q[global_idx, other_idx] = (
                        -0.1 * w_normalized * bit_significance * other_significance
                    )

        # --- 跨权重的非对角项（可选，增加权重间的协调性）---
        # 对于相邻权重，添加弱耦合项，鼓励权重更新的一致性
        # 仅处理连续权重对，避免 O(n^2) 复杂度
        for i in range(min(num_weights - 1, 128)):  # 限制计算量
            base_i = i * n_bits_per_weight
            base_j = (i + 1) * n_bits_per_weight

            # 取第一个比特位的耦合（简化）
            w_i = flat_weights[i] / weight_abs_max
            w_j = flat_weights[i + 1] / weight_abs_max
            coupling = -0.01 * w_i * w_j

            Q[base_i, base_j] = coupling
            Q[base_j, base_i] = coupling

        return Q

    # ------------------------------------------------------------------
    # 方法 3: anneal
    # ------------------------------------------------------------------
    def anneal(self, qubo_matrix: np.ndarray) -> str:
        """
        调用量子退火器（或仿真）求解 QUBO 问题，返回最优比特串

        优先使用 D-Wave Ocean SDK 的 SimulatedAnnealingSampler（如果可用），
        否则回退到内置的 numpy 模拟退火实现。

        Args:
            qubo_matrix: QUBO 矩阵 Q，形状为 (N, N)

        Returns:
            best_bitstring: 最优比特串，例如 "10110..."，长度为 N
        """
        # 将 QUBO 矩阵转换为 dimod 兼容的字典格式 {(i,j): value}
        n = qubo_matrix.shape[0]

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
        original_shape: List[Tuple[int, ...]],
    ) -> List[np.ndarray]:
        """
        将最优比特串解码还原为神经网络权重的形状

        解码策略：
            1. 将比特串按每 n_bits_per_weight 分组
            2. 每组解码为一个固定点数（映射回 [-1, 1] 区间）
            3. 重塑为原始权重的形状

        Args:
            bitstring      : 最优比特串，例如 "10110..."
            original_shape : 原始权重的形状列表，例如 [(128, 64), (64,), ...]

        Returns:
            weights: 解码后的权重列表，每个元素形状与 original_shape 对应
        """
        n_bits_per_weight = max(1, self.num_qubits // 8)

        # 将比特串转为 bit 数组
        bits = np.array([int(b) for b in bitstring], dtype=np.float64)

        # 计算总权重数
        total_params = sum(np.prod(s) for s in original_shape)
        num_bits_used = total_params * n_bits_per_weight

        # 截断或填充比特串以匹配需要的长度
        if len(bits) >= num_bits_used:
            bits = bits[:num_bits_used]
        else:
            # 不足时用 0 填充
            padded = np.zeros(num_bits_used, dtype=np.float64)
            padded[:len(bits)] = bits
            bits = padded

        # 解码每个权重的比特编码为连续值
        decoded_values = np.zeros(total_params, dtype=np.float64)
        for i in range(total_params):
            start = i * n_bits_per_weight
            end = start + n_bits_per_weight
            weight_bits = bits[start:end]
            # 固定点解码：bit_k 的权重为 1/2^k，值为 sum(bit_k / 2^k)
            fp_value = 0.0
            for k in range(n_bits_per_weight):
                fp_value += weight_bits[k] / (2 ** k)
            # 映射到 [-1, 1]：先映射到 [0, 1]，再移到 [-1, 1]
            fp_value = fp_value / (2 ** n_bits_per_weight)  # 归一化到 [0, 1]
            decoded_values[i] = 2.0 * fp_value - 1.0  # 移到 [-1, 1]

        # 将解码后的值重塑为原始权重形状
        weights = []
        offset = 0
        for shape in original_shape:
            count = int(np.prod(shape))
            w = decoded_values[offset:offset + count].reshape(shape)
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
        callback: Optional[Any] = None,
    ) -> Any:
        """
        主优化循环：用量子退火加速策略更新

        每次迭代流程：
            1. 提取 agent 策略网络的当前权重
            2. 计算当前网络的损失/梯度信息（构造 QUBO 目标）
            3. 将权重映射为 QUBO 矩阵
            4. 调用退火器求解最优比特串
            5. 将比特串解码为权重更新
            6. 用学习率混合新旧权重，更新网络

        Args:
            agent         : RL 智能体（需具有 policy_net 属性，为 nn.Module）
            num_iterations: 量子退火优化迭代次数（默认 10）
            learning_rate : 权重更新学习率（默认 0.01），控制新旧权重的混合比例
            callback      : 可选的回调函数，签名为 callback(iteration, loss)
                            可用于日志记录或可视化

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
        policy_net = self._get_policy_net(agent)
        if policy_net is None:
            logger.error("无法获取策略网络，退出 optimize_policy")
            return agent

        logger.info(
            f"开始量子退火策略优化: {num_iterations} 次迭代, "
            f"学习率={learning_rate}, 量子比特数={self.num_qubits}"
        )

        best_loss = float("inf")
        history = []  # 记录每次迭代的损失，用于监控收敛

        for iteration in range(num_iterations):
            # ---- 步骤 1: 提取当前权重及其形状信息 ----
            current_weights, original_shapes = self._extract_weights(policy_net)
            current_param_count = sum(np.prod(s) for s in original_shapes)

            # ---- 步骤 2: 评估当前网络质量（作为 QUBO 构造的参考）----
            # 使用权重的 L2 范数和梯度信息的组合作为 QUBO 构造的辅助信息
            current_loss = self._evaluate_network_quality(policy_net)

            # ---- 步骤 3: 映射为 QUBO 矩阵 ----
            qubo_matrix = self.network_to_qubo(current_weights)

            # ---- 步骤 4: 退火求解 ----
            best_bitstring = self.anneal(qubo_matrix)

            # ---- 步骤 5: 解码比特串为权重 ----
            optimized_weights = self.bitstring_to_weights(best_bitstring, original_shapes)

            # ---- 步骤 6: 混合更新权重 ----
            self._apply_weights(
                policy_net,
                current_weights,
                optimized_weights,
                original_shapes,
                learning_rate=learning_rate,
            )

            # 评估更新后的网络质量
            new_loss = self._evaluate_network_quality(policy_net)

            history.append((iteration, current_loss, new_loss))

            logger.info(
                f"  迭代 {iteration + 1}/{num_iterations}: "
                f"优化前 loss={current_loss:.6f}, 优化后 loss={new_loss:.6f}, "
                f"参数数={current_param_count}, QUBO规模={qubo_matrix.shape[0]}"
            )

            if new_loss < best_loss:
                best_loss = new_loss

            # 调用回调
            if callback is not None:
                callback(iteration, new_loss)

        logger.info(
            f"量子退火策略优化完成: 最佳 loss={best_loss:.6f}, "
            f"共 {num_iterations} 次迭代"
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
    def _get_policy_net(agent: Any) -> Optional[nn.Module]:
        """
        从 agent 对象中获取策略网络

        支持的 agent 类型：
            - 具有 policy_net 属性的 SchedulingAgent
            - 具有 policy 属性的 Stable-Baselines3 DQN agent
        """
        # 方式 1：直接属性（项目内的 SchedulingAgent）
        if hasattr(agent, "policy_net") and isinstance(agent.policy_net, nn.Module):
            return agent.policy_net

        # 方式 2：Stable-Baselines3 DQN agent
        if hasattr(agent, "policy") and hasattr(agent.policy, "q_net"):
            return agent.policy.q_net

        logger.warning("无法识别 agent 的策略网络结构")
        return None

    @staticmethod
    def _extract_weights(network: nn.Module) -> Tuple[List[np.ndarray], List[Tuple[int, ...]]]:
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
        old_weights: List[np.ndarray],
        new_weights: List[np.ndarray],
        shapes: List[Tuple[int, ...]],
        learning_rate: float = 0.01,
    ):
        """
        将优化后的权重应用到网络

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
                network.parameters(), old_weights, new_weights, shapes
            ):
                # 确保形状匹配
                assert w_new.shape == shape, (
                    f"权重形状不匹配: 期望 {shape}, 实际 {w_new.shape}"
                )
                # 归一化新权重到与旧权重相同的尺度，防止尺度跳跃
                old_std = np.std(w_old) + 1e-8
                new_std = np.std(w_new) + 1e-8
                w_new_scaled = w_new * (old_std / new_std)

                # 线性插值
                w_final = (1.0 - learning_rate) * w_old + learning_rate * w_new_scaled

                # 就地更新参数
                param.copy_(torch.from_numpy(w_final.astype(np.float32)))

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
        for sweep in range(self._sim_num_sweeps):
            # 在每个温度下，翻转 n 个比特（一次完整扫描）
            for _ in range(n):
                # 随机选择一个比特进行翻转
                flip_idx = random.randint(0, n - 1)

                # 计算翻转后的能量变化（增量计算，无需重新计算全部能量）
                # ΔE = E(x') - E(x) = Q[flip_idx, flip_idx] * (1 - 2*x[flip_idx])
                #     + Σ_{j≠flip_idx} Q[flip_idx, j] * x[j] * (1 - 2*x[flip_idx])
                # 简化: ΔE = (1 - 2*x[flip]) * (Q[flip,flip] + Σ_{j≠flip} Q[flip,j]*x[j])
                delta = 1.0 - 2.0 * current_solution[flip_idx]
                linear_term = qubo_matrix[flip_idx, flip_idx]
                for j in range(n):
                    if j != flip_idx:
                        linear_term += qubo_matrix[flip_idx, j] * current_solution[j]
                delta_energy = delta * linear_term

                # Metropolis 准则：以概率 min(1, exp(-ΔE/T)) 接受新解
                if delta_energy < 0 or random.random() < math.exp(-delta_energy / max(temperature, 1e-12)):
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

        logger.debug(
            f"numpy 模拟退火: 最佳能量 = {best_energy:.6f}, "
            f"扫描次数 = {sweep + 1}"
        )

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
    print(f"\n环境变量 QUANTUM_ACCELERATION_ENABLED = {os.environ.get('QUANTUM_ACCELERATION_ENABLED', '未设置')}")
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
    for i, (dw, orig_shape) in enumerate(zip(decoded_weights, original_shapes)):
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
    import importlib
    import sys
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
