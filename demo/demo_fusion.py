#!/usr/bin/env python3
"""分组融合 (group fusion) 演示与自检。

原式 rho = 1 - sum_i soft_i^beta 是"竞争者计数"而非覆盖率:
    sum_i soft_i^2 = 1/N_eff ,  rho = 1 - N_eff^(1-beta)
于是两颗种子靠到 < ~2T 时, 交界处 rho -> 0.5 (eta=0.5 即出材料), 局部种子簇
就变成一块实心。分组融合把近距种子的距离场取并集, 从"求和项数"上消除该退化。

运行: python demo/demo_fusion.py
输出: demo/fusion_demo.png + 四条自检
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import jax                                                  # noqa: E402
import jax.numpy as jnp                                     # noqa: E402

from density import (soft_density, soft_density_fused,        # noqa: E402
                     soft_density_grouped, fusion_weight)

T, BETA, LAT = 0.0045, 2.0, 8
DELTA0, EPS = 2 * T, 0.4 * T


def grid(n=320):
    g = np.linspace(0, 1, n)
    x, y = np.meshgrid(g, g, indexing='ij')
    return np.stack([x.ravel(), y.ravel()], -1).astype(np.float32), n


def dists(pts, seeds):
    return jnp.sqrt(jnp.sum((jnp.asarray(pts)[..., None, :]
                             - jnp.asarray(seeds)[None, ...]) ** 2, -1) + 1e-30)


def rho_at(rho, n, a, b, m=41):
    rho = np.asarray(rho).reshape(n, n)
    t = np.linspace(0, 1, m)[:, None]
    p = np.asarray(a) + (np.asarray(b) - np.asarray(a)) * t
    ix = np.clip((p[:, 0] * n).astype(int), 0, n - 1)
    iy = np.clip((p[:, 1] * n).astype(int), 0, n - 1)
    return rho[ix, iy]


def lattice(lat=LAT):
    return np.array([[(i + .5) / lat, (j + .5) / lat]
                     for i in range(lat) for j in range(lat)], np.float32)


def main():
    pts, n = grid()
    S = lattice()
    S2 = S.copy()
    S2[1] = S2[0] + np.array([0.0015, 0.0015], np.float32)   # 制造一处重合

    d0 = dists(pts, S)
    print('=' * 72)
    print(f'分组融合自检   T={T}  beta={BETA}  delta0=2T={DELTA0}  eps=0.4T={EPS}')
    print('=' * 72)

    r_ref = soft_density(d0, BETA, T)
    r_grp = soft_density_grouped(d0, jnp.asarray(S), BETA, T, group=2)
    dmax = float(jnp.max(jnp.abs(r_ref - r_grp)))
    print(f'\n① 规则点阵 (间距 1 胞元 >> delta0):')
    print(f'   max|rho(grouped) - rho(原式)| = {dmax:.2e}  ->  '
          + ('逐位等价 ✓' if dmax < 1e-6 else '不等价 ✗'))
    print(f'   中点 rho = {_mid(r_ref, n, S):.3f} (正常壁, 两式相同)')

    d2 = dists(pts, S2)
    a = soft_density(d2, BETA, T)
    b = soft_density_grouped(d2, jnp.asarray(S2), BETA, T, group=2)
    va, vb = _mid(a, n, S2), _mid(b, n, S2)
    print(f'\n② 两颗种子几乎重合 (间距 0.0015 << delta0):')
    print(f'   原式  rho_mid = {va:.3f}   (eta=0.5 -> 实心块)')
    print(f'   融合  rho_mid = {vb:.3f}   (eta=0.5 -> 无材料)  '
          + ('✓ 修复' if vb < 0.1 else '✗'))

    pfix = jnp.asarray(0.5 * (S2[0] + S2[1]))

    def probe(seeds, grouped):
        dd = dists(pfix.reshape(1, 2), seeds)[0]
        return (soft_density_grouped(dd, seeds, BETA, T, group=2) if grouped
                else soft_density(dd, BETA, T))

    g_ref = jax.grad(lambda s: probe(s, False))(jnp.asarray(S2))
    g_grp = jax.grad(lambda s: probe(s, True))(jnp.asarray(S2))
    print('\n③ 重合对的梯度 (固定场点求导, 两颗种子范数之和):')
    vals = {}
    for nm, gg in (('原式', g_ref), ('融合', g_grp)):
        v = float(jnp.linalg.norm(gg[0]) + jnp.linalg.norm(gg[1]))
        vals[nm] = v
        print(f'   {nm}: {v:.3e}' + ('   两台都活 ✓' if v > 1e-6 else '   ✗ 死变量'))
    if vals['原式'] > 0:
        print(f'   融合 / 原式 = {vals["融合"]/vals["原式"]:.2e}  '
              f'(退化区被解锁; 原式那里近乎平顶)')

    print('\n④ 融合权重 w(R/delta0):')
    for k in (0.0, .5, 1.0, 2.0, 4.0):
        w = float(1 / (1 + np.exp(-(1 - k) / (EPS / DELTA0))))
        print(f'   R={k:3.1f}*delta0 -> w={w:.4f}')

    _figure(S2, n)
    print('\n结论: 远处不动、近处并集、全程可微、两颗都有梯度。')


def _mid(rho, n, S):
    return float(rho_at(rho, n, S[0], S[1]).max())


def _figure(S, n):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    pts, _ = grid(300)
    d = dists(pts, S)
    fig, ax = plt.subplots(1, 3, figsize=(12.6, 4.3))
    for k, (rho, ttl) in enumerate((
            (soft_density(d, BETA, T), 'baseline: coincident pair -> solid block'),
            (soft_density_grouped(d, jnp.asarray(S), BETA, T, group=2),
             'group fusion: coincident pair -> void'))):
        im = ax[k].imshow(np.asarray(rho).reshape(300, 300).T, origin='lower',
                          cmap='magma', extent=[0, 1, 0, 1], vmin=0, vmax=0.6)
        ax[k].set_title(ttl)
        ax[k].set_xlabel('x / L')
        ax[k].set_ylabel('y / L')
        plt.colorbar(im, ax=ax[k], fraction=.046, label=r'$\rho$')
    ax[2].plot(S[:, 0], S[:, 1], 'k.', ms=4)
    for i in (0, 1):
        ax[2].plot(S[i, 0], S[i, 1], '+', color='c', ms=16, mew=2.5)
    ax[2].set_xlim(0, 1)
    ax[2].set_ylim(0, 1)
    ax[2].set_aspect('equal')
    ax[2].set_title('seeds (cyan +: the coincident pair)')
    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fusion_demo.png')
    plt.savefig(out, dpi=115)
    print(f'\n图 -> {out}')


if __name__ == '__main__':
    main()
