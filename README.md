# Voronoi Core

基于 JAX 的高性能 Voronoi 场计算库。

## 安装

### 方式1: pip install

```bash
pip install -e .
```

然后直接导入模块：

```python
from density import soft_density
from field import compute_voronoi
from utils import generate_identity_D
```

### 方式2: git submodule

```bash
git submodule add <repo-url> d_Voronoi_core
```

然后将 `d_Voronoi_core` 的父目录添加到 Python 路径，或直接导入：

```python
import sys
sys.path.append('path/to/parent')
from d_Voronoi_core import compute_voronoi
```

## 依赖

- `jax>=0.4.0`
- `jaxlib>=0.4.0`
- `numpy>=1.24.0`

## 术语

| 术语 | 说明 |
|------|------|
| **seed** | Voronoi 种子点 |
| **unit** | 待求点/场点 |
| **D** | 变换矩阵（D=I 时为欧几里得距离） |
| **cp** | 控制点（影响方向敏感性） |

## 快速开始

```python
from field import compute_voronoi, make_voronoi_fn
from density import compute_gamma
import numpy as np

# 创建网格
Nx, Ny = 100, 100
coords = np.stack(np.indices((Nx, Ny)), axis=-1)

# 种子点
seeds = np.array([[25, 25], [75, 25], [50, 75]], dtype=float)
N = seeds.shape[0]

# 变换矩阵（D=I 时为欧几里得距离）
D = np.tile(np.eye(2), (N, 1, 1))

# 方式1: 直接调用
field = compute_voronoi(coords, seeds, D, beta=4.0)

# 方式2: 预绑定静态参数（推荐用于优化循环）
voronoi = make_voronoi_fn(beta=4.0)
for epoch in range(100):
    eta = 0.3 + 0.4 * (epoch / 100)
    gamma = compute_gamma(epoch)
    field = voronoi(coords, seeds, D, eta=eta, gamma=gamma)
```

## 距离类型

通过 D 矩阵和 cp 参数切换：

```python
from distances import mahalanobis_single

# 1. 欧几里得距离（D=I）
D_I = np.tile(np.eye(2), (N, 1, 1))
dist = mahalanobis_single(unit, seeds, D_I)

# 2. 马氏距离（各向异性）
D = np.array([[[2, 0.5], [0.5, 1]], ...])
dist = mahalanobis_single(unit, seeds, D)

# 3. 带遮罩马氏距离（考虑方向）
cp = seeds + np.array([5, -5])
dist = masked_mahalanobis_single(unit, seeds, D, cp, sigma=0.1, scale=3.0)
```

## 函数签名

### `compute_voronoi`

```python
def compute_voronoi(
    coords,      # 场坐标 (Nx, Ny, dim)
    seeds,       # 种子点 (N, dim)
    D,           # 变换矩阵 (N, dim, dim)
    cp=None,     # 控制点 (N, dim)，可选
    beta=4.0,    # 边界锐度（2-10）
    sigma=0.1,   # 遮罩宽度（0.03-0.3）
    scale=3.0,   # 遮罩强度（1-5）
    eta=0.5,     # Heaviside 阈值（0.3-0.7）
    gamma=1.0,   # 投影锐度（compute_gamma(epoch)）
    batch_size=20
)
```

### `make_voronoi_fn`

```python
voronoi = make_voronoi_fn(
    beta=4.0,
    sigma=0.1,
    scale=3.0,
    batch_size=20
)

# 返回的函数：
field = voronoi(coords, seeds, D, cp=None, eta=0.5, gamma=1.0)
```

## 数学原理

### 软 Voronoi 密度

$$soft_i = \frac{\exp(-d_i)}{\sum_j \exp(-d_j)}, \quad \rho = 1 - \sum_i soft_i^\beta$$

### Heaviside 投影（最后一步）

$$\rho' = \frac{\tanh(\gamma\eta) + \tanh(\gamma(\rho-\eta))}{\tanh(\gamma\eta) + \tanh(\gamma(1-\eta))}$$

## 文件结构

```
d_Voronoi_core/
├── __init__.py      # 公共接口
├── density.py       # soft_density, heaviside, compute_gamma
├── distances.py     # mahalanobis, masked_mahalanobis
├── field.py         # compute_voronoi, make_voronoi_fn
├── utils.py         # generate_random_seeds, generate_identity_D
├── README.md
├── pyproject.toml
└── requirements.txt
```
