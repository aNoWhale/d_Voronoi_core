"""
Voronoi Core 模块

基于 JAX 的高性能 Voronoi 场计算。

术语约定：
- seed: Voronoi 种子点
- unit: 待求点/场点
- D: 变换矩阵（定义马氏距离，D=I 时为欧几里得）
- cp: 控制点（影响方向敏感性）
"""

from density import (
    soft_density,
    soft_density_with_eta,
    soft_density_fused,
    soft_density_grouped,
    fusion_weight,
    group_ids,
    block_perm,
    compute_gamma,
    heaviside
)
from distances import (
    mahalanobis,
    masked_mahalanobis,
    mahalanobis_single,
    masked_mahalanobis_single,
    masked_mahalanobis_single_by_angle
)

from field import (
    compute_voronoi,
    voronoi_field,
    make_voronoi_fn,
    make_optimizer_fn
)
from utils import (
    generate_random_seeds,
    generate_identity_D,
    interpolate_field,
    compute_field_gradient,
    find_boundary,
    angles_to_cp,
    D_params_to_matrix
)

__version__ = "0.6.0"

__all__ = [
    # 密度计算
    "soft_density",
    "soft_density_with_eta",
    "soft_density_fused",
    "soft_density_grouped",
    "fusion_weight",
    "group_ids",
    "block_perm",
    "compute_gamma",
    "heaviside",

    # 距离计算
    "mahalanobis",
    "masked_mahalanobis",
    "mahalanobis_single",
    "masked_mahalanobis_single",
    "masked_mahalanobis_single_by_angle",

    # 场计算
    "compute_voronoi",
    "voronoi_field",
    "make_voronoi_fn",
    "make_optimizer_fn",

    # 工具函数
    "generate_random_seeds",
    "generate_identity_D",
    "interpolate_field",
    "compute_field_gradient",
    "find_boundary",
    "angles_to_cp",
    "D_params_to_matrix",
]
