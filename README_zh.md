# Voronoi Core

中文 | [English](README_en.md)

基于 JAX 的高性能 Voronoi 场计算库。

## 论文引用

本项目是以下论文的核心计算模块：

> Changkun Sun, Yongbo Deng, Nanya Li. **Differentiable Voronoi diagram towards a new design method for the additive manufacturing of 3D disordered cellular structure**. *Engineering Structures*, 364:123138, July 2026. DOI: [10.1016/j.engstruct.2026.123138](https://doi.org/10.1016/j.engstruct.2026.123138)

## 许可证

本项目采用 [MIT 许可证](LICENSE) 开源。您可以自由使用、修改和分发本代码，但需保留版权声明。

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

## 示例展示

### 欧几里得 Voronoi 场

使用欧几里得距离（D=I）的标准 Voronoi。左图：连续密度场；右图：Heaviside 投影后。

![欧几里得 Voronoi](demo/voronoi_euclidean.png)

### 马氏距离 Voronoi 场

使用马氏距离的各向异性 Voronoi，每个种子点有不同的 D 矩阵，产生方向性拉伸效果。

![马氏距离 Voronoi](demo/voronoi_mahalanobis.png)

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
field = compute_voronoi(coords, seeds, D, beta=2.0, temperature=1.0)

# 方式2: 预绑定静态参数（推荐用于优化循环）
voronoi = make_voronoi_fn(beta=2.0)
for epoch in range(100):
    eta = 0.3 + 0.4 * (epoch / 100)
    gamma = compute_gamma(epoch)
    field = voronoi(coords, seeds, D, temperature=1.0, eta=eta, gamma=gamma)
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
    coords,        # 场坐标 (Nx, Ny, dim)
    seeds,         # 种子点 (N, dim)
    D,             # 变换矩阵 (N, dim, dim)
    cp=None,       # 控制点 (N, dim)，可选
    beta=2.0,      # 边界锐度（保持在 2 或 2.66 附近）
    sigma=0.1,     # 遮罩宽度（0.03-0.3）
    scale=3.0,     # 遮罩强度（1-5）
    temperature=1.0,  # 温度系数 T（控制结构宽度）
    eta=0.5,       # Heaviside 阈值（0.3-0.7）
    gamma=1.0,     # 投影锐度（compute_gamma(epoch)）
    batch_size=20
)
```

### `make_voronoi_fn`

```python
voronoi = make_voronoi_fn(
    beta=2.0,
    sigma=0.1,
    scale=3.0,
    batch_size=20
)

# 返回的函数：
field = voronoi(coords, seeds, D, cp=None, temperature=1.0, eta=0.5, gamma=1.0)
```

## 数学原理

### 软 Voronoi 密度

$$soft_i = \frac{\exp(-d_i/T)}{\sum_j \exp(-d_j/T)}, \quad \rho = 1 - \sum_i soft_i^\beta$$

其中：
- **T (temperature)**: 温度系数，控制 softmax 的锐度
  - T < 1: 分布更集中，边界更锐利，结构更窄
  - T > 1: 分布更分散，边界更平滑，结构更宽
  - T = 1: 标准 softmax
- **beta**: 边界锐度，保持在 2 或 2.66 附近（控制密度幂次）

### 分组融合（Group Fusion，v0.6）

$\rho = 1-\sum_i \mathrm{soft}_i^{\beta}$ 严格来说并非"覆盖率"，而是**竞争者个数**的函数：

$$\sum_i \mathrm{soft}_i^{2}=\frac{1}{N_{\mathrm{eff}}},\qquad
N_{\mathrm{eff}}=\frac{1}{\sum_i \mathrm{soft}_i^{2}},\qquad
\rho = 1-N_{\mathrm{eff}}^{\,1-\beta}$$

因此 2 颗种子等距时 $\rho=0.5$（$\eta=0.5$ 即出材料），$M$ 颗挤在 $\sim 2T$ 以内时
$\rho\to 1$——局部种子簇会退化成一块**实心材料**。该退化与距离如何归一化无关
（任何单调重标定都改不了 $N_{\mathrm{eff}}$），只能从"求和项数"上治。

分组融合把"间距小于 $\delta_0$ 的种子组"视为同一特征，即距离场取并集。对覆盖率泛函，
并集满足 $d_{A\cup B}=\min(d_A,d_B)\iff w_{A\cup B}=\mathrm{soft}_A+\mathrm{soft}_B$，故

$$C_g=(1-w_g)\sum_{i\in g}\mathrm{soft}_i^{\beta}
+w_g\Big(\sum_{i\in g}\mathrm{soft}_i\Big)^{\beta},
\qquad \rho = 1-\sum_g C_g$$

$$w_g=\sigma\Big(\frac{\delta_0-R_g}{\varepsilon}\Big),\qquad
R_g=\sqrt{\frac{1}{|\mathcal{P}_g|}\sum_{(i,j)\in\mathcal{P}_g}
\|\mathbf{s}_i-\mathbf{s}_j\|^{2}}$$

三条极限：$R_g\gg\delta_0$ 时 $w_g\to 0$，**逐位退化为原式**（不影响既有行为）；
$R_g\to 0$ 时 $C_g\to 1$，组内无材料（假壁消失），与外部仍给出 $0.5$ 的正常壁；
中间连续过渡。用 RMS 互距而非 $\max$、用求和而非 soft-min：$\max/\min$ 在一阶有折点，
而 $\sum_i\mathrm{soft}_i$ 与 $\sigma$ 都解析，整条链 $C^{\infty}$，组内两颗种子都得到
非零梯度。

```python
from density import soft_density_grouped, soft_density_fused, fusion_weight

# 便捷封装（delta0 缺省 2T，eps 缺省 0.4T）
rho = soft_density_grouped(distances, seeds, beta=2.0, temperature=T,
                           group=2, delta0=2 * T, eps=0.4 * T)

# 或分两步（自定权重）
w = fusion_weight(seeds, group=2, delta0=2 * T, eps=0.4 * T)      # (G,)
rho = soft_density_fused(distances, beta=2.0, temperature=T, weight=w, group=2)
```

注意：`group` 必须整除种子数；分组为**固定索引分组**（不随迭代变化），这是保持
$C^{\infty}$ 的前提。`group=1` 或 `weight=None` 时逐位等于 `soft_density`。
演示与自检：`python demo/demo_fusion.py`（输出 `demo/fusion_demo.png`）。

### Heaviside 投影（最后一步）

$$\rho' = \frac{\tanh(\gamma\eta) + \tanh(\gamma(\rho-\eta))}{\tanh(\gamma\eta) + \tanh(\gamma(1-\eta))}$$

### 开胞/闭胞转换（beta=2）

当 beta=2 时，三维 Voronoi 结构中不同部分的密度分布：
- **面（shell）**：密度约 0.5
- **杆（rod）**：密度约 0.6

通过在 0.5-0.6 之间选择一个 eta 阈值，将密度低于该值的单元过滤掉，即可从三维闭胞 Voronoi 获得三维开胞 Voronoi。

```python
# 闭胞 Voronoi（默认 eta=0.5）
field_closed = compute_voronoi(coords, seeds, D, beta=2.0, eta=0.5)

# 开胞 Voronoi（eta 在 0.5-0.6 之间）
field_open = compute_voronoi(coords, seeds, D, beta=2.0, eta=0.55)
```

## 文件结构

```
d_Voronoi_core/
├── __init__.py      # 公共接口
├── density.py       # soft_density, soft_density_grouped/fused, fusion_weight, heaviside
├── distances.py     # mahalanobis, masked_mahalanobis
├── field.py         # compute_voronoi, make_voronoi_fn
├── utils.py         # generate_random_seeds, generate_identity_D
├── demo/            # 示例脚本和图片
├── README.md        # 英文文档
├── README_zh.md     # 中文文档
├── ai_agent_looks_here.md  # AI agent 详细技术文档
├── LICENSE          # MIT 许可证
├── pyproject.toml
└── requirements.txt
```

## AI Agent 文档

本项目为 AI agent 提供了详细的技术参考文档：[ai_agent_looks_here.md](ai_agent_looks_here.md)

该文档包含：
- 所有函数的完整签名、参数说明和返回值
- 数学原理的详细解释
- 实现细节和设计决策
- 常见任务的完整代码示例
- 参数范围建议和注意事项
