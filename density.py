"""
密度计算模块

术语约定：
- rho: Voronoi 密度场
- beta: 边界锐度参数
- soft: softmax 归一化后的距离
"""

import jax
import jax.numpy as np
import numpy as onp   # 仅用于构造常量掩码/索引 (jax 的 bool 索引须为具体值)


def soft_density(distances: np.ndarray, beta: float, temperature: float = 1.0) -> np.ndarray:
    """
    软 Voronoi 密度

    rho = 1 - sum(soft_i^beta)
    其中 soft_i = exp(-d_i/T) / sum(exp(-d_j/T))

    Args:
        distances: 距离数组，形状 (..., N)
        beta: 边界锐度，保持在 2 或 2.66 附近（控制密度幂次）
        temperature: 温度系数 T（控制 softmax 锐度）
            - T < 1: 分布更集中，边界更锐利
            - T > 1: 分布更分散，边界更平滑
            - T = 1: 标准 softmax

    Returns:
        密度值，形状 (...)
    """
    # run47 数值稳定修复 (2026-10-09): softmax 平移不变性 — 减去最小距离后
    # exp 输入恒 ≤0, 分母 s 恒 ≥1, 永不下溢。
    # 旧版 exp(-d/T) 在 T=0.002 下: 最近种子距离 d>0.0888 时 s<5.4e-20,
    # 反向 1/s² 溢出 float32 → inf×0 → nan (CPU 复现 0.085✓/0.090✗ 与
    # 理论破界 0.0888 精确吻合); d≥0.176 时 exp 归零, 前向 0/0 也 nan。
    # 平移不改变 softmax 值, 前向数学恒等; stop_gradient 截断 min 的梯度
    # (平移量本就是辅助量, 不参与梯度)。
    d0 = jax.lax.stop_gradient(np.min(distances, axis=-1, keepdims=True))
    exp_dist = np.exp(-(distances - d0) / temperature)
    soft = exp_dist / np.sum(exp_dist, axis=-1, keepdims=True)
    return 1.0 - np.sum(soft ** beta, axis=-1)


def soft_density_with_eta(distances: np.ndarray, beta: float, eta: float, temperature: float = 1.0) -> np.ndarray:
    """
    带 eta 补充项的软密度

    在 softmax 中加入 eta 项，用于处理边界情况。

    Args:
        distances: 距离数组，形状 (..., N)
        beta: 边界锐度（保持在 2 或 2.66 附近）
        eta: 补充项
        temperature: 温度系数 T（控制 softmax 锐度）

    Returns:
        密度值，形状 (...)
    """
    # run47: 同 soft_density — softmax 平移减最小距离, 防 1/s² 反向溢出
    d0 = jax.lax.stop_gradient(np.min(distances, axis=-1, keepdims=True))
    exp_dist = np.exp(-(distances - d0) / temperature)
    eta_term = np.full_like(exp_dist[..., :1], eta)
    exp_dist_with_eta = np.concatenate([exp_dist, eta_term], axis=-1)
    soft = exp_dist_with_eta / np.sum(exp_dist_with_eta, axis=-1, keepdims=True)
    return 1.0 - np.sum(soft[..., :-1] ** beta, axis=-1)


def compute_gamma(epoch: int) -> float:
    """
    根据迭代次数计算 Heaviside 投影的 gamma 值

    Args:
        epoch: 当前迭代次数

    Returns:
        gamma 值，随 epoch 增大而增大
    """
    return 2.0 ** (epoch // 2)


def heaviside(rho: np.ndarray, eta: float, gamma: float) -> np.ndarray:
    """
    Heaviside 投影函数（最后一步）

    将连续密度场投影为近似二值场。

    Args:
        rho: 连续密度场
        eta: 投影阈值（典型范围 0.3-0.7）
        gamma: 投影锐度（由 compute_gamma 计算）

    Returns:
        投影后的密度场
    """
    tanh_gamma_eta = np.tanh(gamma * eta)
    tanh_gamma_1_eta = np.tanh(gamma * (1 - eta))
    return (tanh_gamma_eta + np.tanh(gamma * (rho - eta))) / (tanh_gamma_eta + tanh_gamma_1_eta)


# =========================================================================== #
# 分组融合 (group fusion, 2026-11)
#
# 背景: rho = 1 - sum_i soft_i^beta 并非"覆盖率", 而是竞争者个数的函数:
#       sum_i soft_i^2 = 1/N_eff,  N_eff = 1/sum_i soft_i^2,  rho = 1 - N_eff^(1-beta)
#   于是 1 颗 -> rho=0; 2 颗等距 -> 0.5; M 颗挤在 ~2T 内 -> rho -> 1。
#   投影阈值 eta 一压, 局部种子簇就变成一块**实心材料** (退化块)。这与距离如何
#   归一化无关 (任何单调重标定都改不了 N_eff), 只能从"求和项数"上治。
#
# 修法: 把"距离 < delta0 的种子组"看成同一个特征, 即距离场取并集。
#   对覆盖率泛函, 并集的权重恒等式为
#       d_{A∪B} = min(d_A, d_B)  <=>  组权重 = sum_{i in g} soft_i
#   故   C_g = (1-w_g) * sum_{i in g} soft_i^beta  +  w_g * (sum_{i in g} soft_i)^beta
#        rho = 1 - sum_g C_g
#        w_g = sigmoid((delta0 - R_g)/eps),  R_g = 组内 RMS 互距
#
# 三条极限:
#   R_g >> delta0 : w->0, C_g = sum soft_i^beta  => 逐位等于原式 (不影响既有行为)
#   R_g -> 0      : w->1, C_g = 1               => rho=0, 内部假壁消失;
#                                                与外部仍出 0.5 的正常壁
#   中间          : 内部壁连续变淡, 不跳变
#
# 用 RMS 而非 max、用求和而非 soft-min: max/min 在一阶有折点; 求和与 sigmoid
# 都解析, 整条链 C-infinity。组内两颗种子都拿到非零梯度 (硬 min 会让较远那颗
# 梯度恒为 0, 变成自由漂移的死变量)。
# =========================================================================== #


def group_ids(n_seeds: int, group: int) -> np.ndarray:
    """固定索引分组: 连续 group 颗种子为一组。

    固定分组是整条链保持光滑的前提 (数据驱动的动态聚类会在阈值处跳变)。
    """
    if group <= 1:
        return onp.arange(n_seeds)
    if n_seeds % group:
        raise ValueError(f'n_seeds={n_seeds} 不能被 group={group} 整除')
    return onp.repeat(onp.arange(n_seeds // group), group)


def fusion_weight(seeds: np.ndarray, group: int, delta0: float,
                  eps: float) -> np.ndarray:
    """每组的融合权重 w_g in (0,1): 组内互距远小于 delta0 -> 1 (取并集)。

    Args:
        seeds: (N, dim) 种子坐标
        group: 每组种子数 (group<=1 时全部返回 1, 即不融合)
        delta0: 融合尺度, 建议取 6*temperature (=0.5x 配对图间距)
        eps: 过渡软硬, 建议取 0.4*temperature

    Returns:
        (G,) 每组的权重; group<=1 时返回 (N,) 的全 1
    """
    if group <= 1:
        return np.ones((seeds.shape[0],), seeds.dtype)
    s = seeds.reshape(-1, group, seeds.shape[-1])
    diff = s[:, :, None, :] - s[:, None, :, :]
    d2 = np.sum(diff ** 2, axis=-1)                       # (G, group, group)
    iu = onp.triu(onp.ones((group, group), bool), 1)     # 上三角 → 无序对 (须为具体 bool)
    iu = onp.asarray(iu)
    rms = np.sqrt(np.maximum(np.mean(d2[:, iu], axis=-1), 0.0))
    return jax.nn.sigmoid((delta0 - rms) / eps)


def soft_density_fused(distances: np.ndarray, beta: float, temperature: float = 1.0,
                       weight: np.ndarray = None, group: int = 1) -> np.ndarray:
    """带分组融合的软 Voronoi 密度。

    Args:
        distances: (..., N) 场点到各种子的距离
        beta: 边界锐度
        temperature: softmax 温度 T
        weight: (G,) 每组融合权重; 省略或 group<=1 时退化为 soft_density
        group: 每组种子数, 必须整除 N

    Returns:
        (...,) 密度场
    """
    d0 = jax.lax.stop_gradient(np.min(distances, axis=-1, keepdims=True))
    exp_dist = np.exp(-(distances - d0) / temperature)
    soft = exp_dist / np.sum(exp_dist, axis=-1, keepdims=True)

    n = soft.shape[-1]
    if group <= 1 or weight is None:
        return 1.0 - np.sum(soft ** beta, axis=-1)

    if n % group:
        raise ValueError(f'N={n} 不能被 group={group} 整除')
    g = soft.reshape(soft.shape[:-1] + (n // group, group))
    a_g = np.sum(g ** beta, axis=-1)                      # 分开算 (原式)
    b_g = np.sum(g, axis=-1) ** beta                      # 并集算 (组权重之和)
    # w: (G,) -> 只对齐 g 的倒数第二维 (组维), 其余维保持可广播
    w = np.asarray(weight).reshape((1,) * (g.ndim - 2) + (-1,))
    return 1.0 - np.sum((1.0 - w) * a_g + w * b_g, axis=-1)


def soft_density_grouped(distances: np.ndarray, seeds: np.ndarray, beta: float,
                         temperature: float = 1.0, group: int = 2,
                         delta0: float = None, eps: float = None) -> np.ndarray:
    """便捷封装: 由种子坐标自动算融合权重再求密度。

    delta0 缺省 6*temperature (≈0.5x 配对图间距; 2T 实测几乎不触发), eps 缺省 0.4*temperature。
    """
    if delta0 is None:
        delta0 = 6.0 * temperature
    if eps is None:
        eps = 0.4 * temperature
    if group <= 1:
        return soft_density_fused(distances, beta, temperature)
    w = fusion_weight(seeds, group, delta0, eps)
    return soft_density_fused(distances, beta, temperature, weight=w, group=group)
