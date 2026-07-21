# Voronoi Core

**English** | [中文](README_zh.md)

High-performance Voronoi field computation library based on JAX.

## Citation

This project is the core computational module of the following paper:

> Changkun Sun, Yongbo Deng, Nanya Li. **Differentiable Voronoi diagram towards a new design method for the additive manufacturing of 3D disordered cellular structure**. *Engineering Structures*, 364:123138, July 2026. DOI: [10.1016/j.engstruct.2026.123138](https://doi.org/10.1016/j.engstruct.2026.123138)

## License

This project is open-sourced under the [MIT License](LICENSE). You are free to use, modify, and distribute the code, provided that the copyright notice is retained.

## Installation

### Option 1: pip install

```bash
pip install -e .
```

Then import the modules directly:

```python
from density import soft_density
from field import compute_voronoi
from utils import generate_identity_D
```

### Option 2: git submodule

```bash
git submodule add <repo-url> d_Voronoi_core
```

Then add the parent directory of `d_Voronoi_core` to the Python path, or import directly:

```python
import sys
sys.path.append('path/to/parent')
from d_Voronoi_core import compute_voronoi
```

## Dependencies

- `jax>=0.4.0`
- `jaxlib>=0.4.0`
- `numpy>=1.24.0`

## Terminology

| Term | Description |
|------|-------------|
| **seed** | Voronoi seed point |
| **unit** | Query point / field point |
| **D** | Transformation matrix (D=I for Euclidean distance) |
| **cp** | Control point (affects direction sensitivity) |

## Demo

### Euclidean Voronoi Field

Standard Voronoi using Euclidean distance (D=I). Left: continuous density field; Right: after Heaviside projection.

![Euclidean Voronoi](demo/voronoi_euclidean.png)

### Mahalanobis Voronoi Field

Anisotropic Voronoi using Mahalanobis distance with different D matrices. Each seed point has its own transformation matrix, creating directional stretching effects.

![Mahalanobis Voronoi](demo/voronoi_mahalanobis.png)

## Quick Start

```python
from field import compute_voronoi, make_voronoi_fn
from density import compute_gamma
import numpy as np

# Create grid
Nx, Ny = 100, 100
coords = np.stack(np.indices((Nx, Ny)), axis=-1)

# Seed points
seeds = np.array([[25, 25], [75, 25], [50, 75]], dtype=float)
N = seeds.shape[0]

# Transformation matrix (D=I for Euclidean distance)
D = np.tile(np.eye(2), (N, 1, 1))

# Option 1: Direct call
field = compute_voronoi(coords, seeds, D, beta=2.0, temperature=1.0)

# Option 2: Pre-bind static parameters (recommended for optimization loops)
voronoi = make_voronoi_fn(beta=2.0)
for epoch in range(100):
    eta = 0.3 + 0.4 * (epoch / 100)
    gamma = compute_gamma(epoch)
    field = voronoi(coords, seeds, D, temperature=1.0, eta=eta, gamma=gamma)
```

## Distance Types

Switch between distance types using D matrix and cp parameter:

```python
from distances import mahalanobis_single

# 1. Euclidean distance (D=I)
D_I = np.tile(np.eye(2), (N, 1, 1))
dist = mahalanobis_single(unit, seeds, D_I)

# 2. Mahalanobis distance (anisotropic)
D = np.array([[[2, 0.5], [0.5, 1]], ...])
dist = mahalanobis_single(unit, seeds, D)

# 3. Masked Mahalanobis distance (direction-aware)
cp = seeds + np.array([5, -5])
dist = masked_mahalanobis_single(unit, seeds, D, cp, sigma=0.1, scale=3.0)
```

## Function Signatures

### `compute_voronoi`

```python
def compute_voronoi(
    coords,        # Field coordinates (Nx, Ny, dim)
    seeds,         # Seed points (N, dim)
    D,             # Transformation matrix (N, dim, dim)
    cp=None,       # Control points (N, dim), optional
    beta=2.0,      # Boundary sharpness (keep around 2 or 2.66)
    sigma=0.1,     # Mask width (0.03-0.3)
    scale=3.0,     # Mask strength (1-5)
    temperature=1.0,  # Temperature coefficient T (controls structure width)
    eta=0.5,       # Heaviside threshold (0.3-0.7)
    gamma=1.0,     # Projection sharpness (compute_gamma(epoch))
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

# Returned function:
field = voronoi(coords, seeds, D, cp=None, temperature=1.0, eta=0.5, gamma=1.0)
```

## Mathematical Principles

### Soft Voronoi Density

$$soft_i = \frac{\exp(-d_i/T)}{\sum_j \exp(-d_j/T)}, \quad \rho = 1 - \sum_i soft_i^\beta$$

Where:
- **T (temperature)**: Temperature coefficient, controls softmax sharpness
  - T < 1: More concentrated distribution, sharper boundaries, narrower structures
  - T > 1: More dispersed distribution, smoother boundaries, wider structures
  - T = 1: Standard softmax
- **beta**: Boundary sharpness, keep around 2 or 2.66 (controls density power)

### Heaviside Projection (Final Step)

$$\rho' = \frac{\tanh(\gamma\eta) + \tanh(\gamma(\rho-\eta))}{\tanh(\gamma\eta) + \tanh(\gamma(1-\eta))}$$

### Open/Closed Cell Conversion (beta=2)

When beta=2, the density distribution in 3D Voronoi structures has special properties:
- **Shell**: density approximately 0.5
- **Rod**: density approximately 0.6

By choosing an eta threshold between 0.5-0.6, you can filter out units with density below that value, converting 3D closed-cell Voronoi to open-cell Voronoi.

```python
# Closed-cell Voronoi (default eta=0.5)
field_closed = compute_voronoi(coords, seeds, D, beta=2.0, eta=0.5)

# Open-cell Voronoi (eta between 0.5-0.6)
field_open = compute_voronoi(coords, seeds, D, beta=2.0, eta=0.55)
```

## File Structure

```
d_Voronoi_core/
├── __init__.py      # Public interface
├── density.py       # soft_density, heaviside, compute_gamma
├── distances.py     # mahalanobis, masked_mahalanobis
├── field.py         # compute_voronoi, make_voronoi_fn
├── utils.py         # generate_random_seeds, generate_identity_D
├── demo/            # Example scripts and images
├── README.md        # English documentation
├── README_zh.md     # Chinese documentation
├── ai_agent_looks_here.md  # Detailed technical documentation for AI agents
├── LICENSE          # MIT License
├── pyproject.toml
└── requirements.txt
```

## AI Agent Documentation

This project provides detailed technical reference documentation for AI agents: [ai_agent_looks_here.md](ai_agent_looks_here.md)

This documentation includes:
- Complete function signatures, parameter descriptions, and return values
- Detailed explanations of mathematical principles
- Implementation details and design decisions
- Complete code examples for common tasks
- Parameter range suggestions and important notes
