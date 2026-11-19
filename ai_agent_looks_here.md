# AI Agent 完整参考文档

本文档为 AI agent 提供 Voronoi Core 库的详细技术文档，包括所有函数的签名、实现细节和使用示例。本文档设计为可被 RAG（检索增强生成）系统使用，包含结构化的技术信息和完整的代码示例。

## 快速索引

- **核心函数**：[compute_voronoi](#compute_voronoi) | [make_voronoi_fn](#make_voronoi_fn) | [make_optimizer_fn](#make_optimizer_fn)
- **密度计算**：[soft_density](#soft_density) | [soft_density_grouped](#soft_density_groupeddistances-seeds-beta-temperature-group2-delta0none-epsnone) | [heaviside](#heaviside)
- **距离计算**：[mahalanobis](#mahalanobis) | [masked_mahalanobis](#masked_mahalanobis)
- **工具函数**：[generate_identity_D](#generate_identity_D) | [compute_field_gradient](#compute_field_gradient)
- **关键概念**：[开胞/闭胞转换](#开胞闭胞转换) | [方向敏感 Voronoi](#方向敏感-voronoi)

---

## 项目概述

Voronoi Core 是一个基于 JAX 的高性能 Voronoi 场计算库，用于生成连续的 Voronoi 密度场。所有核心计算使用 `jax.numpy`，支持 GPU/TPU 加速。

**学术背景：** 本项目是论文 "Differentiable Voronoi diagram towards a new design method for the additive manufacturing of 3D disordered cellular structure" (Engineering Structures, 2026) 的核心计算模块。

**论文引用：**
```bibtex
@article{sun2026differentiable,
  title={Differentiable Voronoi diagram towards a new design method for the additive manufacturing of 3D disordered cellular structure},
  author={Sun, Changkun and Deng, Yongbo and Li, Nanya},
  journal={Engineering Structures},
  volume={364},
  pages={123138},
  year={2026},
  doi={10.1016/j.engstruct.2026.123138}
}
```

**核心计算流程：**
1. 计算场点到种子点的距离（Mahalanobis 或方向敏感的 masked Mahalanobis）
2. 应用 softmax 归一化和密度函数，生成连续 Voronoi 密度场
3. （可选）应用 Heaviside 投影，将连续场锐化为近似二值场

---

## 模块结构

### 1. density.py — 密度计算

#### `soft_density(distances, beta, temperature)`

**功能：** 计算软 Voronoi 密度，这是库的核心密度公式。

**数学公式：**
```
soft_i = exp(-d_i/T) / sum(exp(-d_j/T))
rho = 1 - sum(soft_i^beta)
```

**参数：**
| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `distances` | ndarray | `(..., N)` | 场点到 N 个种子点的距离 |
| `beta` | float | - | 边界锐度，控制密度幂次。推荐值：2.0 或 2.66 |
| `temperature` | float | - | 温度系数 T，控制 softmax 锐度。T<1 锐利，T>1 平滑 |

**返回值：** 密度值，形状 `(...)`

**实现细节：**
- 使用 `exp(-distances/T)` 计算未归一化的 softmax
- 对 softmax 结果取 `beta` 次幂后求和
- 最终密度 `rho = 1 - sum(soft^beta)`，范围 [0, 1]
- 当所有距离相等时，rho 接近 1；当一个距离远小于其他时，rho 接近 0

---

#### `soft_density_with_eta(distances, beta, eta, temperature)`

**功能：** 带 eta 补充项的软密度变体，用于处理边界情况。

**数学公式：**
```
exp_dist = exp(-d_i/T)
eta_term = eta  # 常数项
soft_i = exp_dist_i / (sum(exp_dist) + eta_term)
rho = 1 - sum(soft_i^beta)
```

**参数：** 同 `soft_density`，额外参数：
| 参数 | 类型 | 说明 |
|------|------|------|
| `eta` | float | 补充项，添加到 softmax 分母中 |

**实现细节：**
- 在 softmax 分母中添加一个常数 `eta` 项
- 这会略微降低所有种子点的 softmax 权重
- 用于调整边界行为

---

#### `soft_density_grouped(distances, seeds, beta, temperature, group=2, delta0=None, eps=None)`

**功能：** 带**分组融合**的软密度。把间距小于 `delta0` 的种子组视为同一特征
（距离场取并集），从而消除"两颗种子靠得太近 → 一整块实心材料"的退化。

**为什么需要它：** `soft_density` 里的 `rho` 并不是覆盖率，而是**竞争者个数**的函数

```
sum_i soft_i^2 = 1/N_eff ,  N_eff = 1/sum_i soft_i^2 ,  rho = 1 - N_eff^(1-beta)
```

- 1 颗主导 → `rho = 0`
- 2 颗等距 → `rho = 0.5`（`eta = 0.5` 时正好出材料）
- M 颗挤在 ~2T 以内 → `rho → 1`（整块）

该退化对任何单调重标定 `g(d)` 都不变（`N_eff` 不变），因此只能从"求和项数"上治。

**数学公式：**

```
A_g = sum_{i in g} soft_i^beta                  # 分开算（原式）
B_g = ( sum_{i in g} soft_i )^beta              # 并集算（组权重可加）
C_g = (1 - w_g) * A_g + w_g * B_g
rho = 1 - sum_g C_g

w_g = sigmoid( (delta0 - R_g) / eps )
R_g = sqrt( mean_{i<j in g} |s_i - s_j|^2 )     # 组内 RMS 互距
```

并集恒等式：`d_{A∪B} = min(d_A, d_B)` ⇔ `w_{A∪B} = soft_A + soft_B`。两子重合时
`soft_i + soft_j = 1` ⇒ `C_g = 1` ⇒ `rho = 0`（组内无材料），而组与外部交界仍是
`rho = 0.5` 的正常壁。

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `distances` | (..., N) | 场点到各种子的距离 |
| `seeds` | (N, dim) | 种子坐标，仅用于算融合权重 |
| `group` | int | 每组种子数，必须整除 N；`1` 退化为 `soft_density` |
| `delta0` | float | 融合尺度，缺省 `6*temperature`（≈0.5×配对图间距） |
| `eps` | float | 过渡软硬，缺省 `0.4*temperature` |

**实现细节：**
- **固定索引分组**（不随迭代变化）——这是整条链保持 C^∞ 的前提；数据驱动的动态
  聚类会在阈值处跳变。
- 用 RMS 互距而非 `max`、用求和而非 soft-min：`max/min` 在一阶有折点，而求和与
  `sigmoid` 都解析，**组内两颗种子都拿到非零梯度**（硬 `min` 会让较远那颗梯度
  恒为 0，成为自由漂移的死变量）。
- `w_g = 0` 时与原式**逐位一致**（实测 `max|Δrho| = 1.2e-7`，float32 精度）。
- **δ0 必须与配对图的实际间距同量级**（不是与 T 同量级）：18×18 点阵标称间距 0.56 mm，
  塌缩对 0.15 mm，健康对 0.37–0.56 mm ⇒ δ0≈0.5×间距（≈6T）可清晰区分；
  δ0=2T 时 w≈σ(−15)≈0，机制不触发（实测 1T–16T 所有指标不变）。

**相关函数：** `fusion_weight(seeds, group, delta0, eps)` 只返回每组权重 `(G,)`；
`soft_density_fused(distances, beta, temperature, weight, group)` 接受外部给定的权重。
演示与自检：`python demo/demo_fusion.py`。

**实测（demo）：** 两颗种子间距 0.0015 ≪ δ0 时，连线中点的 `rho` 由 `0.485`
（原式，η=0.5 出材料）降到 `0.010`（融合，无材料）；规则点阵下两式
`max|Δrho| = 6e-8`。

---

#### `compute_gamma(epoch)`

**功能：** 根据迭代次数计算 Heaviside 投影的 gamma 值。

**数学公式：** `gamma = 2^(epoch//2)`

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `epoch` | int | 当前迭代次数 |

**返回值：** gamma 值（随 epoch 指数增长）

**使用场景：** 在优化循环中，随着迭代进行逐渐增大 gamma，使投影越来越锐利。

**示例：**
```python
gamma_schedule = [compute_gamma(e) for e in range(10)]
# [1, 1, 2, 2, 4, 4, 8, 8, 16, 16]
```

---

#### `heaviside(rho, eta, gamma)`

**功能：** Heaviside 投影函数，将连续密度场投影为近似二值场。

**数学公式：**
```
rho' = (tanh(gamma*eta) + tanh(gamma*(rho-eta))) / (tanh(gamma*eta) + tanh(gamma*(1-eta)))
```

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `rho` | ndarray | 连续密度场 |
| `eta` | float | 投影阈值（典型范围 0.3-0.7） |
| `gamma` | float | 投影锐度。gamma=1 时几乎不投影，gamma 越大投影越锐利 |

**返回值：** 投影后的密度场

**行为：**
- `rho > eta` 的区域被推向 1
- `rho < eta` 的区域被推向 0
- `gamma` 越大，过渡越锐利
- 当 `gamma=1.0` 时，投影效果很弱

---

### 2. distances.py — 距离计算

#### `mahalanobis(unit, seed, D)`

**功能：** 批量计算马氏距离（欧几里得是 D=I 的特例）。

**数学公式：** `d = sqrt((x-s)^T D^T D (x-s))`

**参数：**
| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `unit` | ndarray | `(..., n, dim)` | 场点坐标 |
| `seed` | ndarray | `(N, dim)` | 种子点坐标 |
| `D` | ndarray | `(N, dim, dim)` | 变换矩阵。D=I 时为欧几里得距离 |

**返回值：** 距离矩阵，形状 `(..., N, n)`

**实现细节：**
- 使用 `einsum` 进行高效的批量矩阵运算
- 结果取 `sqrt` 并用 `maximum(..., 0.0)` 保证数值稳定性

---

#### `masked_mahalanobis(unit, seed, cp, D, sigma, scale, eps)`

**功能：** 带方向遮罩的马氏距离，基于 seed->cp 方向与 seed->unit 方向的相似度。

**数学公式：**
```
d_mahal = mahalanobis(unit, seed, D)
cos_theta = |dot(seed->cp, seed->unit)| / (|seed->cp| * |seed->unit|)
mask = scale * exp(-0.5 * ((cos_theta - 1) / sigma)^2) + 1
d_masked = mask * d_mahal
```

**参数：**
| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `unit` | ndarray | `(..., n, dim)` | 场点坐标 |
| `seed` | ndarray | `(N, dim)` | 种子点坐标 |
| `cp` | ndarray | `(N, dim)` | 控制点坐标 |
| `D` | ndarray | `(N, dim, dim)` | 变换矩阵 |
| `sigma` | float | - | 遮罩宽度（典型 0.03-0.3） |
| `scale` | float | - | 遮罩强度（典型 1-5） |
| `eps` | float | - | 数值稳定性常数（默认 1e-9） |

**返回值：** 距离矩阵，形状 `(..., N, n)`

**行为说明：**
- 当 `unit` 与 `cp` 在 `seed` 的同一方向时，`cos_theta ≈ 1`，mask ≈ scale+1，距离被放大
- 当方向不同时，mask 趋近于 1，距离基本不变
- 这实现了方向敏感的距离度量

---

#### `mahalanobis_single(unit, seeds, D)`

**功能：** 单点到多种子的马氏距离（用于 `jax.vmap` 批处理）。

**参数：**
| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `unit` | ndarray | `(dim,)` | 单个场点 |
| `seeds` | ndarray | `(N, dim)` | N 个种子点 |
| `D` | ndarray | `(N, dim, dim)` | 变换矩阵 |

**返回值：** 距离向量，形状 `(N,)`

**实现细节：**
- `diff = seeds - unit`
- `transformed = einsum('nij,nj->ni', D, diff)` 对每个种子点应用变换
- 最终距离 = `sqrt(sum(transformed^2, axis=-1))`

---

#### `masked_mahalanobis_single(unit, seeds, D, cp, sigma, scale, eps)`

**功能：** 单点版本的带遮罩马氏距离。

**重要差异：** 比批量版本多一个 `direction_weight` sigmoid 衰减项。

**数学公式：**
```
d_mahal = mahalanobis_single(unit, seeds, D)
cos_theta = |dot(seed->cp, seed->unit)| / (|seed->cp| * |seed->unit|)
mask = scale * exp(-0.5 * ((cos_theta - 1) / sigma)^2) + 1
direction_weight = 1 / (1 + exp(-0.1 * (|seed->cp| - |seed->unit|)))
d_masked = mask^direction_weight * d_mahal
```

**direction_weight 的作用：**
- 当 `|seed->cp| > |seed->unit|`（unit 比 cp 更靠近 seed）时，weight ≈ 1
- 当 `|seed->cp| < |seed->unit|`（unit 比 cp 更远离 seed）时，weight < 1
- 这提供了额外的距离衰减，使得远离 cp 的 unit 点距离被压缩

---

### 3. field.py — 主入口模块

#### `_compute_unit_density(unit, seeds, D, cp, beta, sigma, scale, temperature)`

**功能：** JIT 编译的单点密度计算核心。

**JIT 优化：** 使用 `static_argnums=(4,5,6,7)` 将 beta/sigma/scale/temperature 作为编译时常量。

**距离分派逻辑：**
```python
if cp is None:
    dist = mahalanobis_single(unit, seeds, D)
else:
    dist = masked_mahalanobis_single(unit, seeds, D, cp, sigma, scale)
return soft_density(dist, beta, temperature)
```

---

#### `voronoi_field(coords, seeds, D, cp, beta, sigma, scale, temperature, batch_size)`

**功能：** 批处理场计算，支持单设备和多设备并行。

**参数：**
| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `coords` | ndarray | `(Nx, Ny, dim)` | 场坐标网格 |
| `seeds` | ndarray | `(N, dim)` | 种子点 |
| `D` | ndarray | `(N, dim, dim)` | 变换矩阵 |
| `cp` | ndarray | `(N, dim)` 或 None | 控制点 |
| `beta` | float | - | 边界锐度 |
| `sigma` | float | - | 遮罩宽度 |
| `scale` | float | - | 遮罩强度 |
| `temperature` | float | - | 温度系数 |
| `batch_size` | int | - | 批处理大小 |

**返回值：** 密度场，形状 `(Nx, Ny)`

**实现细节：**
1. 将 coords 展平为 `(n_units, dim)`
2. 使用 `jax.vmap` 向量化单点计算
3. 单设备：使用 `jax.lax.map` 逐批处理
4. 多设备：使用 `jax.pmap` 并行处理，自动分配到各设备
5. 处理 padding 以适应批大小
6. 最后 reshape 回原始形状

---

#### `compute_voronoi(coords, seeds, D, cp, beta, sigma, scale, temperature, eta, gamma, batch_size)`

**功能：** 完整的 Voronoi 场计算主入口。

**计算流程：**
1. 调用 `voronoi_field` 计算连续场
2. 如果 `gamma > 1.0`，应用 Heaviside 投影

**参数：** 同 `voronoi_field`，额外参数：
| 参数 | 类型 | 说明 |
|------|------|------|
| `eta` | float | Heaviside 投影阈值（典型 0.3-0.7） |
| `gamma` | float | 投影锐度。gamma=1.0 时不投影 |

**返回值：** 密度场，形状 `(Nx, Ny)`

---

#### `make_voronoi_fn(beta, sigma, scale, batch_size)`

**功能：** 创建预绑定静态参数的 Voronoi 场计算函数（闭包）。

**返回值：** 函数 `voronoi_fn(coords, seeds, D, cp=None, temperature=1.0, eta=0.5, gamma=1.0)`

**使用场景：** 在优化循环中重复调用，避免重复传递静态参数。

**示例：**
```python
voronoi = make_voronoi_fn(beta=2.0, sigma=0.1, scale=3.0)

for epoch in range(100):
    gamma = compute_gamma(epoch)
    field = voronoi(coords, seeds, D, temperature=0.8, eta=0.5, gamma=gamma)
```

---

#### `make_optimizer_fn(n_seeds, dim, use_cp, beta, sigma, scale, batch_size)`

**功能：** 创建优化器友好的函数，接收扁平参数向量 `p`。

**参数向量 p 的布局：**
```
p = [seeds.ravel(), D.ravel(), cp.ravel()]  # cp 仅在 use_cp=True 时
```

**返回值：** 函数 `optimizer_fn(p, coords, temperature=1.0, eta=0.5, gamma=1.0)`

**示例：**
```python
import jax.numpy as np

opt_fn = make_optimizer_fn(n_seeds=5, dim=2, use_cp=False, beta=2.0)

# 构造参数向量
seeds = np.array([[25, 25], [75, 25], [50, 75], [25, 75], [75, 75]], dtype=float)
D = np.tile(np.eye(2), (5, 1, 1))
p = np.concatenate([seeds.ravel(), D.ravel()])

# 计算场
field = opt_fn(p, coords, temperature=0.8, eta=0.5, gamma=4.0)
```

---

### 4. utils.py — 工具函数

#### `generate_random_seeds(n_seeds, bounds, key)`

**功能：** 在指定边界内生成随机种子点。

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `n_seeds` | int | 种子点数量 |
| `bounds` | tuple | 边界 `((x_min, y_min), (x_max, y_max))` |
| `key` | PRNGKey | JAX 随机密钥（默认使用 key(0)） |

**返回值：** 种子点坐标，形状 `(n_seeds, 2)`

**注意：** 仅支持 2D，如需 3D 需自行扩展。

---

#### `generate_identity_D(n_seeds, dim)`

**功能：** 生成单位变换矩阵（对应欧几里得距离）。

**返回值：** 形状 `(n_seeds, dim, dim)` 的单位矩阵堆叠

**示例：**
```python
D_2d = generate_identity_D(10, 2)  # 10 个 2D 种子点
D_3d = generate_identity_D(20, 3)  # 20 个 3D 种子点
```

---

#### `interpolate_field(field, coords, query_points)`

**功能：** 场值的最近邻插值。

**参数：**
| 参数 | 类型 | 形状 | 说明 |
|------|------|------|------|
| `field` | ndarray | `(Nx, Ny)` | 场值 |
| `coords` | ndarray | `(Nx, Ny, 2)` | 场坐标网格 |
| `query_points` | ndarray | `(N, 2)` | 查询点 |

**返回值：** 插值结果，形状 `(N,)`

**实现细节：**
- 假设均匀网格
- 将查询点映射到最近的网格索引
- 使用 `np.clip` 处理边界

---

#### `compute_field_gradient(field, dx)`

**功能：** 计算场的梯度。

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `field` | ndarray | 场值，形状 `(Nx, Ny)` |
| `dx` | float | 网格间距（默认 1.0） |

**返回值：** 梯度场，形状 `(Nx, Ny, 2)`

---

#### `find_boundary(field, threshold)`

**功能：** 查找边界点（梯度幅值超过阈值的位置）。

**参数：**
| 参数 | 类型 | 说明 |
|------|------|------|
| `field` | ndarray | 场值，形状 `(Nx, Ny)` |
| `threshold` | float | 梯度阈值（默认 0.5） |

**返回值：** 边界点坐标，形状 `(N, 2)`

---

## 关键设计细节

### 距离分派机制

```python
if cp is None:
    # 普通 Mahalanobis 距离
    dist = mahalanobis_single(unit, seeds, D)
else:
    # 方向敏感的 masked Mahalanobis
    dist = masked_mahalanobis_single(unit, seeds, D, cp, sigma, scale)
```

- `cp=None`：标准 Voronoi
- `cp` 有值：方向敏感的 Voronoi，适合生成有方向性的结构

### JIT 编译策略

`_compute_unit_density` 使用 `static_argnums=(4,5,6,7)`：
- beta, sigma, scale, temperature 在 JIT 编译时确定
- 改变这些参数会触发重新编译
- 这允许编译器对这些常量进行特化优化

### 批量 vs 单点实现的差异

`masked_mahalanobis_single` 比 `masked_mahalanobis` 多一个 `direction_weight` 项：

```python
direction_weight = 1.0 / (1.0 + np.exp(-0.1 * (norm_sc - norm_sx)))
return (mask ** direction_weight) * d_mahal
```

这是一个 sigmoid 函数，根据 unit 与 cp 的相对距离提供额外的衰减。

---

## 关键概念详解

### 开胞/闭胞转换

当 beta=2 时，三维 Voronoi 结构中不同部分的密度分布具有特殊性质：

**密度分布特征：**
- **面（shell）**：密度约 0.5
- **杆（rod）**：密度约 0.6

**转换方法：**
通过 `heaviside` 函数在 0.5-0.6 之间选择一个 eta 阈值，将密度低于该值的单元过滤掉，即可从三维闭胞 Voronoi 获得三维开胞 Voronoi。

**示例：**
```python
# 闭胞 Voronoi（默认 eta=0.5）
field_closed = compute_voronoi(coords_3d, seeds_3d, D_3d, beta=2.0, eta=0.5)

# 开胞 Voronoi（eta 在 0.5-0.6 之间）
field_open = compute_voronoi(coords_3d, seeds_3d, D_3d, beta=2.0, eta=0.55)

# 更稀疏的开胞结构
field_sparse = compute_voronoi(coords_3d, seeds_3d, D_3d, beta=2.0, eta=0.6)
```

---

### 方向敏感 Voronoi

通过 `cp`（控制点）参数，可以实现方向敏感的 Voronoi 结构。

**原理：**
- `cp` 定义了每个种子点的"参考方向"
- 当场点与 `cp` 在种子点的同一方向时，距离被放大
- 当方向不同时，距离基本不变

**参数调节：**
- `sigma`（0.03-0.3）：控制遮罩的宽度，sigma 越小遮罩越窄
- `scale`（1-5）：控制遮罩的强度，scale 越大方向敏感性越强

**示例：**
```python
# 所有种子点的控制点向同一方向偏移
cp = seeds + np.array([10, -10])

# 生成有方向性的 Voronoi 结构
field = compute_voronoi(coords, seeds, D, cp=cp, beta=2.0, sigma=0.1, scale=3.0)
```

---

### 温度系数的影响

温度系数 T 控制 softmax 的锐度，直接影响 Voronoi 结构的边界特性：

| T 值 | 效果 | 适用场景 |
|------|------|----------|
| T < 1 | 边界更锐利，结构更窄 | 需要清晰边界的场景 |
| T = 1 | 标准 softmax | 一般用途 |
| T > 1 | 边界更平滑，结构更宽 | 需要平滑过渡的场景 |

---

### Heaviside 投影策略

Heaviside 投影将连续密度场转换为近似二值场，是生成最终结构的关键步骤。

**参数作用：**
- `eta`：投影阈值，决定哪些区域保留
- `gamma`：投影锐度，决定过渡的陡峭程度

**优化循环中的使用：**
```python
from density import compute_gamma

for epoch in range(100):
    # gamma 随迭代指数增长
    gamma = compute_gamma(epoch)  # 1, 1, 2, 2, 4, 4, 8, 8, ...
    
    # eta 可以固定或动态调整
    eta = 0.5  # 固定阈值
    
    field = voronoi(coords, seeds, D, eta=eta, gamma=gamma)
```

---

## 使用示例

### 基础 2D Voronoi 场

```python
import numpy as np
from field import compute_voronoi
from utils import generate_identity_D

# 创建 100x100 网格
Nx, Ny = 100, 100
coords = np.stack(np.indices((Nx, Ny)), axis=-1).astype(float)

# 定义种子点
seeds = np.array([[25, 25], [75, 25], [50, 75]], dtype=float)
N = seeds.shape[0]

# 单位变换矩阵（欧几里得距离）
D = generate_identity_D(N, 2)

# 计算场（beta=2，不做投影）
field = compute_voronoi(coords, seeds, D, beta=2.0, temperature=1.0)
```

### 3D 开胞 Voronoi

```python
# 创建 50x50x50 网格
coords_3d = np.stack(np.indices((50, 50, 50)), axis=-1).astype(float)

# 随机种子点
import jax
key = jax.random.PRNGKey(42)
seeds_3d = jax.random.uniform(key, shape=(20, 3)) * 50

# 单位变换矩阵
D_3d = generate_identity_D(20, 3)

# beta=2 时：shell 密度 ≈ 0.5，rod 密度 ≈ 0.6
# eta=0.55 可获得开胞结构
field_open = compute_voronoi(coords_3d, seeds_3d, D_3d, beta=2.0, eta=0.55, gamma=4.0)
```

### 优化循环

```python
from field import make_voronoi_fn, make_optimizer_fn
from density import compute_gamma
import jax

# 方法1：使用 make_voronoi_fn
voronoi = make_voronoi_fn(beta=2.0)

for epoch in range(100):
    eta = 0.3 + 0.4 * (epoch / 100)
    gamma = compute_gamma(epoch)
    field = voronoi(coords, seeds, D, temperature=0.8, eta=eta, gamma=gamma)

# 方法2：使用 make_optimizer_fn（用于梯度优化）
opt_fn = make_optimizer_fn(n_seeds=5, dim=2, use_cp=False, beta=2.0)

# 初始参数
seeds_init = np.array([[25, 25], [75, 25], [50, 75], [25, 75], [75, 75]], dtype=float)
D_init = generate_identity_D(5, 2)
p_init = np.concatenate([seeds_init.ravel(), D_init.ravel()])

# 梯度优化
grad_fn = jax.grad(lambda p: np.mean(opt_fn(p, coords)))
for epoch in range(100):
    grads = grad_fn(p_init)
    p_init = p_init - 0.01 * grads
```

### 方向敏感的 Voronoi

```python
# 控制点：影响方向敏感性
cp = seeds + np.array([10, -10])  # 每个种子点的控制点

# 使用 masked Mahalanobis
field_directional = compute_voronoi(
    coords, seeds, D, cp=cp,
    beta=2.0,
    sigma=0.1,   # 遮罩宽度
    scale=3.0    # 遮罩强度
)
```

### 计算梯度和边界

```python
from utils import compute_field_gradient, find_boundary

# 计算梯度
gradient = compute_field_gradient(field, dx=1.0)
grad_magnitude = np.sqrt(np.sum(gradient**2, axis=-1))

# 查找边界
boundary_points = find_boundary(field, threshold=0.3)
```

---

## 参数范围建议

| 参数 | 推荐范围 | 说明 |
|------|----------|------|
| `beta` | 2.0 或 2.66 | 边界锐度，2.0 是常用值 |
| `temperature` | 0.5 - 2.0 | T<1 锐利，T>1 平滑 |
| `eta` | 0.3 - 0.7 | Heaviside 阈值。beta=2 时，0.5-0.6 可控制开胞程度 |
| `gamma` | compute_gamma(epoch) | 投影锐度，随迭代指数增长 |
| `sigma` | 0.03 - 0.3 | 遮罩宽度（仅 cp 不为 None 时生效） |
| `scale` | 1 - 5 | 遮罩强度（仅 cp 不为 None 时生效） |
| `batch_size` | 10 - 50 | 根据内存调整 |

---

## 重要注意事项

1. **所有核心计算使用 JAX**：支持 GPU/TPU 加速，首次调用会有编译开销
2. **多设备并行**：`voronoi_field` 自动检测设备数量，使用 `jax.pmap` 并行处理
3. **内存控制**：通过 `batch_size` 参数控制同时处理的点数
4. **D 矩阵语义**：D=I 时为欧几里得距离；D 非对角时为各向异性
5. **cp 参数**：仅在需要方向敏感性时传入，否则保持 None
6. **beta=2 的特殊性质**：三维 Voronoi 中 shell 密度约 0.5，rod 密度约 0.6
7. **静态参数编译**：beta/sigma/scale/temperature 改变会触发 JIT 重新编译
8. **单点 vs 批量实现**：`masked_mahalanobis_single` 多一个 `direction_weight` 衰减项
