"""
距离计算模块

术语约定：
- seed: Voronoi 种子点
- unit: 待求点/场点
- D: 变换矩阵（定义马氏距离）
- cp: 控制点（影响方向敏感性）

距离类型：
- 欧几里得 = D=I 的马氏距离
- 马氏距离 = 一般情况
- 带遮罩马氏 = 考虑方向敏感性
"""

import jax.numpy as np


def mahalanobis(unit: np.ndarray, seed: np.ndarray, D: np.ndarray) -> np.ndarray:
    """
    马氏距离（欧几里得是 D=I 的特例）

    d = sqrt((x-s)^T D^T D (x-s))

    Args:
        unit: 场点坐标，形状 (..., n, dim)
        seed: 种子点坐标，形状 (N, dim)
        D: 变换矩阵，形状 (N, dim, dim)
           D=I 时为欧几里得距离

    Returns:
        距离矩阵，形状 (..., N, n)
    """
    diff = unit[..., np.newaxis, :, np.newaxis, :] - seed[..., np.newaxis, :, np.newaxis, :, :]
    nor = np.einsum(
        '...ijklm,...imn,...ijkln->...ijk',
        np.einsum('...ijklm,...imn->...ijkln', diff, D.swapaxes(-1, -2)),
        D
    )
    return np.sqrt(np.maximum(nor, 0.0))


def masked_mahalanobis(
    unit: np.ndarray,
    seed: np.ndarray,
    cp: np.ndarray,
    D: np.ndarray,
    sigma: float,
    scale: float,
    eps: float = 1e-9
) -> np.ndarray:
    """
    带方向遮罩的马氏距离

    遮罩基于 seed->cp 方向与 seed->unit 方向的相似度。

    Args:
        unit: 场点坐标，形状 (..., n, dim)
        seed: 种子点坐标，形状 (N, dim)
        cp: 控制点坐标，形状 (N, dim)
        D: 变换矩阵，形状 (N, dim, dim)
        sigma: 遮罩宽度
        scale: 遮罩强度
        eps: 数值稳定性常数

    Returns:
        距离矩阵，形状 (..., N, n)
    """
    # 马氏距离
    d_mahal = mahalanobis(unit, seed, D)

    # 方向向量
    diff_ref = seed - cp  # seed -> cp 参考方向
    diff_cur = unit[..., np.newaxis, :, :] - seed[..., np.newaxis, :, :, :]  # seed -> unit

    # 余弦相似度
    dot = np.sum(diff_ref[..., np.newaxis, :, :] * diff_cur, axis=-1)
    norm_ref = np.linalg.norm(diff_ref, axis=-1)[..., np.newaxis, :]
    norm_cur = np.linalg.norm(diff_cur, axis=-1)
    cos_theta = np.abs(dot / (norm_ref * norm_cur + eps))

    # 高斯遮罩
    mask = scale * np.exp(-0.5 * ((cos_theta - 1.0) / sigma) ** 2) + 1.0

    return mask * d_mahal


# 单点版本（用于 vmap 批处理）

def mahalanobis_single(unit: np.ndarray, seeds: np.ndarray, D: np.ndarray) -> np.ndarray:
    """单点到多种子的马氏距离"""
    diff = seeds - unit
    transformed = np.einsum('nij,nj->ni', D, diff)
    return np.sqrt(np.sum(transformed ** 2, axis=-1))


def masked_mahalanobis_single(
    unit: np.ndarray,
    seeds: np.ndarray,
    D: np.ndarray,
    cp: np.ndarray,
    sigma: float,
    scale: float,
    eps: float = 1e-9
) -> np.ndarray:
    """单点到多种子的带遮罩马氏距离"""
    # 马氏距离
    d_mahal = mahalanobis_single(unit, seeds, D)

    # 方向计算
    diff_sx = unit - seeds  # seed -> unit
    diff_sc = cp - seeds    # seed -> cp

    # 余弦相似度
    dot = np.sum(diff_sc * diff_sx, axis=-1)
    norm_sc = np.linalg.norm(diff_sc, axis=-1) + eps
    norm_sx = np.linalg.norm(diff_sx, axis=-1) + eps
    cos_theta = np.abs(dot / (norm_sc * norm_sx))

    # 遮罩
    mask = scale * np.exp(-0.5 * ((cos_theta - 1.0) / sigma) ** 2) + 1.0

    # 方向衰减
    direction_weight = 1.0 / (1.0 + np.exp(-0.1 * (norm_sc - norm_sx)))

    return (mask ** direction_weight) * d_mahal


def masked_mahalanobis_single_by_angle(
    unit: np.ndarray,
    seeds: np.ndarray,
    D: np.ndarray,
    angle: np.ndarray,
    sigma: float,
    scale: float,
    eps: float = 1e-9
) -> np.ndarray:
    """
    单点到多种子的带遮罩马氏距离（角度参数化版本）

    直接使用角度参数定义方向，无需控制点坐标。
    方向向量 = [cos(angle), sin(angle)]

    Args:
        unit: 单个场点，形状 (dim,)
        seeds: 种子点坐标，形状 (N, dim)
        D: 变换矩阵，形状 (N, dim, dim)
        angle: 方向角度，形状 (N,)，弧度
        sigma: 遮罩宽度
        scale: 遮罩强度
        eps: 数值稳定性常数

    Returns:
        距离向量，形状 (N,)
    """
    # 马氏距离
    d_mahal = mahalanobis_single(unit, seeds, D)

    # 方向向量：从角度直接计算
    direction = np.stack([np.cos(angle), np.sin(angle)], axis=-1)  # (N, 2)

    # 方向计算
    diff_sx = unit - seeds  # seed -> unit

    # 余弦相似度（使用方向向量而非 cp 坐标）
    dot = np.sum(direction * diff_sx, axis=-1)
    norm_dir = np.linalg.norm(direction, axis=-1) + eps  # ≈ 1，但保持通用
    norm_sx = np.linalg.norm(diff_sx, axis=-1) + eps
    cos_theta = np.abs(dot / (norm_dir * norm_sx))

    # 遮罩
    mask = scale * np.exp(-0.5 * ((cos_theta - 1.0) / sigma) ** 2) + 1.0

    # 方向衰减（使用固定参考距离，避免半径概念）
    # 使用单位距离作为参考
    ref_dist = 1.0
    direction_weight = 1.0 / (1.0 + np.exp(-0.1 * (ref_dist - norm_sx)))

    return (mask ** direction_weight) * d_mahal
