"""M1b refit stability: frozen-probe coordinates of H against an operator refitted on probe + H.

One orthogonal K x K alignment fitted on the probe images over all sites (eigenvectors rescaled for N).
Per site: share of squared coordinate norm reproduced, for the probe images themselves (how far the refit
moved the probe) and for the held-out images (refit drift plus extension error).
Run: uv run --with numpy python invert/refit.py
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / 'data/invert'
sp = np.load(ROOT / 'splits.npz')
P, H = sp['P2'], sp['H']
U = np.sort(np.r_[P, H]); posP, posH = np.searchsorted(U, P), np.searchsorted(U, H)
out = {}
for a, b in (('ea30', 'ea60'), ('ea100', 'ea200'), ('ea300', 'ea600'), ('pr100', 'pr200'), ('pr300', 'pr600'), ('knn15', 'knn30')):
    try:
        zp = np.load(ROOT / f'joint/P2__{a}__L12/modes.npz'); zr = np.load(ROOT / f'joint/P2+H__{b}__L12/modes.npz')
        cz = np.load(ROOT / f'joint/P2__{a}__L12/coords_H.npz')
    except FileNotFoundError:
        continue
    scale = np.sqrt(len(U) / len(P))
    for K in (16, 64):
        A = zp['vecs'][:, :, :K].astype(float); Y = cz['y'][:, :, :K].astype(float); Y1 = cz['onestep'][:, :, :K].astype(float)
        Bp = zr['vecs'][:, posP, :K].astype(float) * scale; Bh = zr['vecs'][:, posH, :K].astype(float) * scale
        u, _, vt = np.linalg.svd(np.einsum('lnk,lnj->kj', A, Bp))
        Q = u @ vt
        share = lambda X, B: 1 - ((X @ Q - B) ** 2).sum((1, 2)) / (B ** 2).sum((1, 2))
        sp_, sh, s1 = share(A, Bp), share(Y, Bh), share(Y1, Bh)
        out[f'{a}|{K}'] = dict(probe=sp_.tolist(), held=sh.tolist(), onestep=s1.tolist(), cond_max=float(cz['cond'][:K].max()),
                               ratio=float(np.sqrt((Y ** 2).sum() / (A ** 2).sum() * len(P) / len(H))))
        print(f'{a:6s} K={K:3d} probe ' + ' '.join(f'{v:5.2f}' for v in sp_) + f'   cond<= {cz["cond"][:K].max():.1f}')
        print(f'{"":6s}       held  ' + ' '.join(f'{v:5.2f}' for v in sh) + f'   rms(held)/rms(probe) {out[f"{a}|{K}"]["ratio"]:.2f}')
        print(f'{"":6s}       1step ' + ' '.join(f'{v:5.2f}' for v in s1))
(ROOT / 'refit.json').write_text(json.dumps(out))
