"""
密度计算模块

术语约定：
- rho: Voronoi 密度场
- beta: 边界锐度参数
- soft: softmax 归一化后的距离
"""

import jax
import jax.numpy as np


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
