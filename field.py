"""
Voronoi 场计算模块

所有参数直接暴露在函数签名中。

术语约定：
- seed: Voronoi 种子点
- unit: 待求点/场点
- D: 变换矩阵（定义马氏距离，D=I 时为欧几里得）
- cp: 控制点（影响方向敏感性）

距离类型切换：
- 传入 D, 不传 cp → 马氏距离（D=I 时为欧几里得）
- 传入 D 和 cp → 带方向遮罩的马氏距离
"""

from typing import Optional, Callable
from functools import partial

import jax
import jax.numpy as np

from density import soft_density, heaviside
from distances import mahalanobis_single, masked_mahalanobis_single


# =============================================================================
# 核心计算函数（JIT 优化）
# =============================================================================

@partial(jax.jit, static_argnums=(4, 5, 6, 7))
def _compute_unit_density(
    unit: np.ndarray,
    seeds: np.ndarray,
    D: np.ndarray,
    cp: Optional[np.ndarray],
    beta: float,
    sigma: float,
    scale: float,
    temperature: float = 1.0
) -> np.ndarray:
    """计算单点的 Voronoi 密度"""
    if cp is None:
        dist = mahalanobis_single(unit, seeds, D)
    else:
        dist = masked_mahalanobis_single(unit, seeds, D, cp, sigma, scale)
    return soft_density(dist, beta, temperature)


# =============================================================================
# 批处理场计算
# =============================================================================

def voronoi_field(
    coords: np.ndarray,
    seeds: np.ndarray,
    D: np.ndarray,
    cp: Optional[np.ndarray] = None,
    beta: float = 2.0,
    sigma: float = 0.1,
    scale: float = 3.0,
    temperature: float = 1.0,
    batch_size: int = 20
) -> np.ndarray:
    """
    计算 Voronoi 场（支持批处理和多设备）

    Args:
        coords: 场坐标，形状 (Nx, Ny, dim)
        seeds: 种子点坐标，形状 (N, dim)
        D: 变换矩阵，形状 (N, dim, dim)
            D=I 时为欧几里得距离
        cp: 控制点坐标，形状 (N, dim)，可选
            传入时使用带方向遮罩的马氏距离
        beta: 边界锐度（保持在 2 或 2.66 附近）
        sigma: 遮罩宽度（典型范围 0.03-0.3），仅在 cp 不为 None 时生效
        scale: 遮罩强度（典型范围 1-5），仅在 cp 不为 None 时生效
        temperature: 温度系数 T（控制 softmax 锐度）
            - T < 1: 分布更集中，边界更锐利
            - T > 1: 分布更分散，边界更平滑
            - T = 1: 标准 softmax
        batch_size: 批处理大小

    Returns:
        密度场，形状 (Nx, Ny)
    """
    original_shape = coords.shape[:-1]
    flat_coords = coords.reshape(-1, coords.shape[-1])
    n_units = flat_coords.shape[0]

    compute_fn = partial(_compute_unit_density, beta=beta, sigma=sigma, scale=scale, temperature=temperature)
    vectorized = jax.vmap(compute_fn, in_axes=(0, None, None, None))

    devices = jax.devices()
    num_devices = len(devices)

    if num_devices == 1:
        def process_batch(batch):
            return vectorized(batch, seeds, D, cp)

        n_batches = (n_units + batch_size - 1) // batch_size
        padded_size = n_batches * batch_size
        padding = padded_size - n_units
        padded = np.pad(flat_coords, ((0, padding), (0, 0)))
        batched = padded.reshape(n_batches, batch_size, -1)
        results = jax.lax.map(process_batch, batched)
        return results.reshape(-1)[:n_units].reshape(original_shape)
    else:
        effective_batch = batch_size * num_devices
        n_batches = (n_units + effective_batch - 1) // effective_batch
        padded_size = n_batches * effective_batch
        padding = padded_size - n_units
        padded = np.pad(flat_coords, ((0, padding), (0, 0)))
        batched = padded.reshape(n_batches, effective_batch, -1)
        device_coords = batched.reshape(n_batches, num_devices, batch_size, -1)

        def process_device(batch):
            return vectorized(batch, seeds, D, cp)

        pmap_fn = jax.pmap(process_device)
        results = pmap_fn(device_coords)
        return results.reshape(-1)[:n_units].reshape(original_shape)


# =============================================================================
# 主入口函数
# =============================================================================

def compute_voronoi(
    coords: np.ndarray,
    seeds: np.ndarray,
    D: np.ndarray,
    cp: Optional[np.ndarray] = None,
    beta: float = 2.0,
    sigma: float = 0.1,
    scale: float = 3.0,
    temperature: float = 1.0,
    eta: float = 0.5,
    gamma: float = 1.0,
    batch_size: int = 20
) -> np.ndarray:
    """
    计算 Voronoi 场

    计算流程：
    1. 计算距离矩阵（马氏或带遮罩马氏）
    2. Softmax 归一化 + 密度计算
    3. [可选] Heaviside 投影（最后一步）

    Args:
        coords: 场坐标，形状 (Nx, Ny, dim)
        seeds: 种子点坐标，形状 (N, dim)
        D: 变换矩阵，形状 (N, dim, dim)
            - D=I 时为欧几里得距离
            - D 非对角时为各向异性
        cp: 控制点坐标，形状 (N, dim)，可选
            - 不传：普通马氏距离
            - 传入：带方向遮罩的马氏距离
        beta: 边界锐度（保持在 2 或 2.66 附近，控制密度幂次）
        sigma: 遮罩宽度（典型范围 0.03-0.3），仅 cp 不为 None 时生效
        scale: 遮罩强度（典型范围 1-5），仅 cp 不为 None 时生效
        temperature: 温度系数 T（控制 softmax 锐度，调整结构宽度）
            - T < 1: 分布更集中，边界更锐利，结构更窄
            - T > 1: 分布更分散，边界更平滑，结构更宽
            - T = 1: 标准 softmax
        eta: Heaviside 投影阈值（典型范围 0.3-0.7）
        gamma: Heaviside 投影锐度（由 compute_gamma 计算）
               gamma=1.0 时不做投影
        batch_size: 批处理大小

    Returns:
        密度场，形状 (Nx, Ny)
    """
    # 步骤1-2: 计算连续场
    field = voronoi_field(coords, seeds, D, cp, beta, sigma, scale, temperature, batch_size)

    # 步骤3: Heaviside 投影（最后一步）
    if gamma > 1.0:
        field = heaviside(field, eta, gamma)

    return field


# =============================================================================
# 便捷工厂函数
# =============================================================================

def make_voronoi_fn(
    beta: float = 2.0,
    sigma: float = 0.1,
    scale: float = 3.0,
    batch_size: int = 20
) -> Callable:
    """
    创建 Voronoi 场计算函数（预绑定静态参数）

    Args:
        beta: 边界锐度（保持在 2 或 2.66 附近）
        sigma: 遮罩宽度
        scale: 遮罩强度
        batch_size: 批处理大小

    Returns:
        voronoi_fn(coords, seeds, D, cp=None, temperature=1.0, eta=0.5, gamma=1.0)

    Example:
        >>> # 欧几里得距离（D=I），标准温度
        >>> voronoi = make_voronoi_fn(beta=2.0)
        >>> field = voronoi(coords, seeds, np.eye(2)[None].repeat(N, 0))
        >>>
        >>> # 低温 → 锐利边界
        >>> field = voronoi(coords, seeds, D, temperature=0.5)
        >>>
        >>> # 高温 → 平滑边界
        >>> field = voronoi(coords, seeds, D, temperature=2.0)
    """
    def voronoi_fn(
        coords: np.ndarray,
        seeds: np.ndarray,
        D: np.ndarray,
        cp: Optional[np.ndarray] = None,
        temperature: float = 1.0,
        eta: float = 0.5,
        gamma: float = 1.0
    ) -> np.ndarray:
        return compute_voronoi(
            coords, seeds, D, cp,
            beta=beta, sigma=sigma, scale=scale,
            temperature=temperature,
            eta=eta, gamma=gamma,
            batch_size=batch_size
        )
    return voronoi_fn


def make_optimizer_fn(
    n_seeds: int,
    dim: int,
    use_cp: bool = False,
    beta: float = 2.0,
    sigma: float = 0.1,
    scale: float = 3.0,
    batch_size: int = 20
) -> Callable:
    """
    创建优化器友好的函数（接收参数向量 p）

    Args:
        n_seeds: 种子点数量
        dim: 维度
        use_cp: 是否使用控制点
        beta: 边界锐度（保持在 2 或 2.66 附近）
        sigma: 遮罩宽度
        scale: 遮罩强度
        batch_size: 批处理大小

    Returns:
        optimizer_fn(p, coords, temperature=1.0, eta=0.5, gamma=1.0)

    Example:
        >>> opt_fn = make_optimizer_fn(n_seeds=5, dim=2, beta=2.0)
        >>> p = np.concatenate([seeds.ravel(), D.ravel()])
        >>> field = opt_fn(p, coords, temperature=0.8, eta=0.5, gamma=4.0)
    """
    def optimizer_fn(
        p: np.ndarray,
        coords: np.ndarray,
        temperature: float = 1.0,
        eta: float = 0.5,
        gamma: float = 1.0
    ) -> np.ndarray:
        idx = 0
        seeds = p[idx:idx + n_seeds * dim].reshape(n_seeds, dim)
        idx += n_seeds * dim

        D = p[idx:idx + n_seeds * dim * dim].reshape(n_seeds, dim, dim)
        idx += n_seeds * dim * dim

        cp = None
        if use_cp:
            cp = p[idx:idx + n_seeds * dim].reshape(n_seeds, dim)

        return compute_voronoi(
            coords, seeds, D, cp,
            beta=beta, sigma=sigma, scale=scale,
            temperature=temperature,
            eta=eta, gamma=gamma,
            batch_size=batch_size
        )
    return optimizer_fn
