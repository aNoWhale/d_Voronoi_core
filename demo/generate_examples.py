"""
生成示例图片：普通 Voronoi 和马氏距离 Voronoi
使用均匀分布的种子点（网格+抖动），更符合实际 Voronoi 结构
"""

import sys
sys.path.insert(0, '..')

import numpy as np
import jax
import matplotlib.pyplot as plt
from field import compute_voronoi
from utils import generate_identity_D

# 通用设置
Nx, Ny = 150, 150
coords = np.stack(np.indices((Nx, Ny)), axis=-1).astype(float)

# 均匀分布的种子点（网格 + 随机抖动）
N_per_side = 4  # 每边 4 个点
N = N_per_side * N_per_side  # 总共 16 个点
margin = 20  # 边距

# 在网格上均匀分布（坐标范围 0-149）
grid_x = np.linspace(margin, Nx - 1 - margin, N_per_side)
grid_y = np.linspace(margin, Ny - 1 - margin, N_per_side)
xx, yy = np.meshgrid(grid_x, grid_y)
seeds_grid = np.stack([xx.ravel(), yy.ravel()], axis=-1)

# 添加随机抖动
key = jax.random.PRNGKey(42)
jitter = jax.random.uniform(key, shape=(N, 2), minval=-10, maxval=10)
seeds = seeds_grid + np.array(jitter)

print(f"种子点位置:\n{seeds}")

# =============================================================================
# 案例 1: 普通 Voronoi（欧几里得距离）
# =============================================================================
print("\n生成案例 1: 普通 Voronoi...")
D_euclidean = generate_identity_D(N, 2)

# 不做 Heaviside 投影（gamma=1.0）
field_euclidean = compute_voronoi(coords, seeds, D_euclidean, beta=2.0, temperature=1.0, gamma=1.0)

# 做 Heaviside 投影（gamma=4.0）
field_euclidean_proj = compute_voronoi(coords, seeds, D_euclidean, beta=2.0, temperature=1.0, eta=0.5, gamma=4.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：连续场
im1 = axes[0].imshow(field_euclidean.T, origin='lower',
                      extent=[0, Nx-1, 0, Ny-1], cmap='viridis')
axes[0].scatter(seeds[:, 0], seeds[:, 1], c='red', s=120, marker='x', linewidths=3, zorder=5)
axes[0].set_title('Euclidean Distance (Continuous)', fontsize=12)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# 右图：投影后
im2 = axes[1].imshow(field_euclidean_proj.T, origin='lower',
                      extent=[0, Nx-1, 0, Ny-1], cmap='viridis')
axes[1].scatter(seeds[:, 0], seeds[:, 1], c='red', s=120, marker='x', linewidths=3, zorder=5)
axes[1].set_title('Euclidean Distance (Heaviside Projected)', fontsize=12)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
plt.colorbar(im2, ax=axes[1], shrink=0.8)

plt.suptitle(f'Euclidean Voronoi Field (N={N}, β=2.0, T=1.0)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('voronoi_euclidean.png', dpi=150, bbox_inches='tight')
print("已保存: voronoi_euclidean.png")

# =============================================================================
# 案例 2: 马氏距离 Voronoi（各向异性）
# =============================================================================
print("\n生成案例 2: 马氏距离 Voronoi...")
# 为每个种子点生成随机但可控的变换矩阵
D_anisotropic = np.zeros((N, 2, 2), dtype=float)
keys = jax.random.split(key, N)

for i in range(N):
    # 生成随机角度
    angle = jax.random.uniform(keys[i], minval=0, maxval=np.pi)
    # 生成随机缩放比
    scale_x = jax.random.uniform(keys[i], minval=1.5, maxval=3.0)
    scale_y = jax.random.uniform(keys[i], minval=0.4, maxval=0.8)

    # 旋转矩阵
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    R = np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # 缩放矩阵
    S = np.array([[scale_x, 0], [0, scale_y]])

    # D = R @ S（旋转 + 缩放）
    D_anisotropic[i] = R @ S

# 不做 Heaviside 投影
field_aniso = compute_voronoi(coords, seeds, D_anisotropic, beta=2.0, temperature=1.0, gamma=1.0)

# 做 Heaviside 投影
field_aniso_proj = compute_voronoi(coords, seeds, D_anisotropic, beta=2.0, temperature=1.0, eta=0.5, gamma=4.0)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 左图：连续场
im1 = axes[0].imshow(field_aniso.T, origin='lower',
                      extent=[0, Nx-1, 0, Ny-1], cmap='magma')
axes[0].scatter(seeds[:, 0], seeds[:, 1], c='cyan', s=120, marker='x', linewidths=3, zorder=5)
axes[0].set_title('Mahalanobis Distance (Continuous)', fontsize=12)
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')
plt.colorbar(im1, ax=axes[0], shrink=0.8)

# 右图：投影后
im2 = axes[1].imshow(field_aniso_proj.T, origin='lower',
                      extent=[0, Nx-1, 0, Ny-1], cmap='magma')
axes[1].scatter(seeds[:, 0], seeds[:, 1], c='cyan', s=120, marker='x', linewidths=3, zorder=5)
axes[1].set_title('Mahalanobis Distance (Heaviside Projected)', fontsize=12)
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
plt.colorbar(im2, ax=axes[1], shrink=0.8)

plt.suptitle(f'Mahalanobis Voronoi Field (N={N}, β=2.0, T=1.0, Anisotropic D)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('voronoi_mahalanobis.png', dpi=150, bbox_inches='tight')
print("已保存: voronoi_mahalanobis.png")

plt.close('all')
print("\n示例图片生成完成！")
