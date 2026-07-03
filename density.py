"""
密度计算模块

术语约定：
- rho: Voronoi 密度场
- beta: 边界锐度参数
- soft: softmax 归一化后的距离
"""

import jax.numpy as np


def soft_density(distances: np.ndarray, beta: float) -> np.ndarray:
    """
    软 Voronoi 密度

    rho = 1 - sum(soft_i^beta)
    其中 soft_i = exp(-d_i) / sum(exp(-d_j))

    Args:
        distances: 距离数组，形状 (..., N)
        beta: 边界锐度，值越大边界越锐利

    Returns:
        密度值，形状 (...)
    """
    exp_dist = np.exp(-distances)
    soft = exp_dist / np.sum(exp_dist, axis=-1, keepdims=True)
    return 1.0 - np.sum(soft ** beta, axis=-1)


def soft_density_with_eta(distances: np.ndarray, beta: float, eta: float) -> np.ndarray:
    """
    带 eta 补充项的软密度

    在 softmax 中加入 eta 项，用于处理边界情况。

    Args:
        distances: 距离数组，形状 (..., N)
        beta: 边界锐度
        eta: 补充项

    Returns:
        密度值，形状 (...)
    """
    exp_dist = np.exp(-distances)
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
