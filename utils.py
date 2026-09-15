"""
工具函数模块

术语约定：
- seed: Voronoi 种子点（原 site）
- unit: 待求点/场点（原 cell）
"""

import jax.numpy as np
import jax


def generate_random_seeds(
    n_seeds: int,
    bounds: tuple,
    key: jax.random.PRNGKey = None
) -> np.ndarray:
    """
    生成随机种子点

    Args:
        n_seeds: 种子点数量
        bounds: 边界 ((x_min, y_min), (x_max, y_max))
        key: JAX 随机密钥

    Returns:
        种子点坐标，形状 (n_seeds, 2)
    """
    if key is None:
        key = jax.random.PRNGKey(0)

    low, high = bounds
    return jax.random.uniform(
        key,
        shape=(n_seeds, 2),
        minval=np.array(low),
        maxval=np.array(high)
    )


def generate_identity_D(n_seeds: int, dim: int = 2) -> np.ndarray:
    """
    生成单位变换矩阵（对应欧几里得距离）

    Args:
        n_seeds: 种子点数量
        dim: 维度

    Returns:
        变换矩阵，形状 (n_seeds, dim, dim)
    """
    return np.tile(np.eye(dim), (n_seeds, 1, 1))


def interpolate_field(
    field: np.ndarray,
    coords: np.ndarray,
    query_points: np.ndarray
) -> np.ndarray:
    """
    场值插值（最近邻）

    Args:
        field: 场值，形状 (Nx, Ny)
        coords: 场坐标，形状 (Nx, Ny, 2)
        query_points: 查询点，形状 (N, 2)

    Returns:
        插值结果，形状 (N,)
    """
    dx = coords[1, 0, 0] - coords[0, 0, 0]
    dy = coords[0, 1, 1] - coords[0, 0, 1]

    x_idx = ((query_points[:, 0] - coords[0, 0, 0]) / dx).astype(int)
    y_idx = ((query_points[:, 1] - coords[0, 0, 1]) / dy).astype(int)

    x_idx = np.clip(x_idx, 0, field.shape[0] - 1)
    y_idx = np.clip(y_idx, 0, field.shape[1] - 1)

    return field[x_idx, y_idx]


def compute_field_gradient(field: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    计算场梯度

    Args:
        field: 场值，形状 (Nx, Ny)
        dx: 网格间距

    Returns:
        梯度场，形状 (Nx, Ny, 2)
    """
    grad_x = np.gradient(field, dx, axis=0)
    grad_y = np.gradient(field, dx, axis=1)
    return np.stack([grad_x, grad_y], axis=-1)


def D_params_to_matrix(
    theta: np.ndarray,
    lambda1: np.ndarray,
    lambda2: np.ndarray
) -> np.ndarray:
    """
    将 D 矩阵参数 (θ, λ₁, λ₂) 转换为变换矩阵

    D = R(θ) @ diag(λ₁, λ₂) @ R(-θ)

    其中 R(θ) 是旋转矩阵。通过约束 λ₁, λ₂ 接近 1，可以：
    - 保持 D 接近单位矩阵（欧几里得距离）
    - 允许轻微的各向异性以增加结构变化
    - 避免过度拉伸导致的异形结构

    Args:
        theta: 旋转角度，形状 (n_seeds,)
        lambda1: 第一主轴缩放因子，形状 (n_seeds,)
        lambda2: 第二主轴缩放因子，形状 (n_seeds,)

    Returns:
        D: 变换矩阵，形状 (n_seeds, 2, 2)

    Example:
        >>> theta = np.array([0.0, np.pi/4])
        >>> lambda1 = np.array([1.1, 1.0])
        >>> lambda2 = np.array([0.9, 1.0])
        >>> D = D_params_to_matrix(theta, lambda1, lambda2)
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # R(θ) = [[cos, -sin], [sin, cos]]
    # D = R @ diag(λ1, λ2) @ R^T
    # 展开计算每个元素
    D_00 = lambda1 * cos_t**2 + lambda2 * sin_t**2
    D_01 = (lambda1 - lambda2) * cos_t * sin_t
    D_10 = D_01  # 对称
    D_11 = lambda1 * sin_t**2 + lambda2 * cos_t**2

    # 堆叠为 (n_seeds, 2, 2)
    D = np.stack([
        np.stack([D_00, D_01], axis=-1),
        np.stack([D_10, D_11], axis=-1)
    ], axis=-2)

    return D


def angles_to_cp(
    seeds: np.ndarray,
    angles: np.ndarray,
    radius: float = 0.1
) -> np.ndarray:
    """
    将角度参数转换为控制点坐标

    控制点 (cp) 定义了每个种子点的参考方向，用于 masked Mahalanobis 距离。
    使用角度参数化而非直接优化 cp 坐标，可以：
    - 消除距离自由度的冗余（只有方向有意义）
    - 减少设计变量数量（每个种子点只需 1 个角度而非 2 个坐标）
    - 让梯度信息更干净，加速收敛

    Args:
        seeds: 种子点坐标，形状 (n_seeds, dim)
        angles: 角度参数，形状 (n_seeds,)，范围 [0, 2π)
        radius: 种子到控制点的固定距离（默认 0.1）

    Returns:
        cp: 控制点坐标，形状 (n_seeds, dim)

    Example:
        >>> seeds = np.array([[0.5, 0.5], [0.3, 0.7]])
        >>> angles = np.array([0.0, np.pi/2])  # 第一个向右，第二个向上
        >>> cp = angles_to_cp(seeds, angles, radius=0.1)
        >>> # cp ≈ [[0.6, 0.5], [0.3, 0.8]]
    """
    direction = np.stack([np.cos(angles), np.sin(angles)], axis=-1)
    cp = seeds + radius * direction
    return cp


def find_boundary(field: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    查找边界

    Args:
        field: 场值，形状 (Nx, Ny)
        threshold: 梯度阈值

    Returns:
        边界点坐标，形状 (N, 2)
    """
    grad = compute_field_gradient(field)
    grad_mag = np.sqrt(np.sum(grad ** 2, axis=-1))
    boundary_mask = grad_mag > threshold
    return np.argwhere(boundary_mask)
