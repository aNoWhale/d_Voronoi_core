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
