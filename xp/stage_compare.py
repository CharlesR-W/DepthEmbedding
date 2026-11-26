"""X1 (xp/SPEC.md): pairwise comparison of networks on one probe.  Unit {'probe': p} -> /results/compare/<p>.npz

All arrays are indexed by network in the order of `nets` (A = first index, B = second), sites in common.SITES order.

  nets (n,)                     network names that had an embedding for this probe
  labels (N,)                   probe labels
  nn (n, n, 12, 12)             neighbour overlap: mean_i |N_A,l(i) & N_B,m(i)| / 15   (chance 15 / (N - 1))
  nn_perm, nn_within (n, n, 12) same-site overlap with B's images permuted (freely / within class)
  cka (n, n, 12, 12)            linear CKA of the centred, row-normalised features (double-centred Grams)
  cka_perm, cka_within (n, n, 12)   same-site CKA under the two permutation nulls
  readout (n, 12)               share of an image's 15 nearest probe neighbours with its own label
  readout_coarse (n, 12)        the same with CIFAR-100 coarse labels (c100 probe only)
  per kernel k in (knn15, ea100):
    <k>_walk8, <k>_walk16 (n, n, 12)      |U_A^T U_B|_F^2 / r for the first r nontrivial eigenvectors of each site's walk
    <k>_walk8_perm, <k>_walk8_within      the same under the two permutation nulls
    <k>_lam2 (n, 12)                      second eigenvalue of each site's lazy walk
    per chain c in (trunk, full), prefix <k>_<c>_ :
      share (n, n, 4)           |V_A^T V_B|_F^2 / K for K = 4, 8, 16, 32 (joint modes stacked over sites)
      share_perm, share_within  the same under the nulls
      sv (n, n, 16)             singular values of V_B^T V_A at K = 16 (cosines of the principal angles)
      modes (n, n, 16, 16)      |<u_j^B, u_k^A>|, rows j (B), columns k (A)
      site (n, n, L)            held-out share of A's squared coordinates at each site reproduced by z_B Q, one
                                orthogonal Q for all sites fitted on half of the probe images (K = 16)
      cos (n, n, L)             the same fit scored as a cosine: <z_A, z_B Q> / (|z_A| |z_B|) at each site, held out
                                (1 = same embedding up to the rotation, 0 = unrelated; symmetric up to the fit)
      site_perm, site_within, cos_perm, cos_within    the same under the nulls
      vals (n, 32), t (n,), c (n, L)       spectrum, coupling and site scales of each network's operator
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

KERNELS = ['knn15', 'ea100']
CHAINS = ['trunk', 'full']


def within_class_perm(labels, rng):
    perm = np.arange(len(labels))
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        perm[idx] = idx[rng.permutation(len(idx))]
    return perm


def site_share(va, vb, train, test):
    """va, vb (L, N, K).  Orthogonal Q from the train images (all sites); per-site share on the test images."""
    L, _, K = va.shape
    u, _, wt = np.linalg.svd(vb[:, train].reshape(-1, K).T @ va[:, train].reshape(-1, K))
    q = u @ wt
    a, b = va[:, test], vb[:, test] @ q
    na, nb = (a ** 2).sum((1, 2)), (b ** 2).sum((1, 2))
    return 1 - ((a - b) ** 2).sum((1, 2)) / na, (a * b).sum((1, 2)) / np.sqrt(na * nb)


def run(unit, cfg):
    import scipy.sparse as sp
    import torch
    from common import R, X1_NETS
    t0 = time.time()
    probe = unit['probe']
    out = Path(R) / cfg.get('out', 'compare'); out.mkdir(parents=True, exist_ok=True)
    root = Path(R) / 'embed'
    nets = cfg.get('nets', X1_NETS)
    missing = [m for m in nets if not (root / m / probe / 'DONE').exists()]
    assert not missing, f'no embedding on probe {probe} for {missing}; pass an explicit nets list for a reduced run'
    n = len(nets)
    labels = np.load(root / nets[0] / probe / 'meta.npz')['labels']
    N = len(labels)
    rng = np.random.default_rng(0)
    perms = dict(perm=rng.permutation(N), within=within_class_perm(labels, rng))
    half = rng.permutation(N)
    train, test = np.sort(half[:N // 2]), np.sort(half[N // 2:])
    res = dict(nets=np.array(nets), labels=labels, perm=perms['perm'], within=perms['within'])

    # ---- neighbour overlap (all site pairs) and the class readout
    knn = np.stack([np.load(root / m / probe / 'knn.npy').astype(np.int64) for m in nets])        # (n, 12, N, 15)
    rows_i = np.repeat(np.arange(N), 15)

    def indicator(k, perm=None):
        """Sparse (n * 12, N * N) indicator of directed neighbour pairs; perm relabels images i -> perm[i]."""
        mats = []
        for a in range(n):
            for l in range(12):
                i, j = rows_i, k[a, l].ravel()
                if perm is not None:
                    i, j = perm[i], perm[j]
                mats.append(sp.csr_matrix((np.ones(len(i), np.float32), (np.zeros(len(i), np.int64), i * N + j)),
                                          shape=(1, N * N)))
        return sp.vstack(mats).tocsr()

    M = indicator(knn)
    res['nn'] = np.asarray((M @ M.T).todense()).reshape(n, 12, n, 12).transpose(0, 2, 1, 3) / (15 * N)
    diag = np.arange(12)
    for name, p in perms.items():
        inv = np.argsort(p)                       # B's image p[i] plays the role of image i
        G = np.asarray((M @ indicator(knn, inv).T).todense()).reshape(n, 12, n, 12).transpose(0, 2, 1, 3) / (15 * N)
        res[f'nn_{name}'] = G[:, :, diag, diag]
    res['readout'] = (labels[knn] == labels[None, None, :, None]).mean((2, 3))
    if probe == 'c100':
        coarse = np.load(Path(R) / 'probes.npz')['c100_coarse']
        res['readout_coarse'] = (coarse[knn] == coarse[None, None, :, None]).mean((2, 3))
    del M

    # ---- linear CKA (all site pairs)
    torch.set_num_threads(16)
    F = torch.empty((n * 12, N * N), dtype=torch.float32)
    for a, m in enumerate(nets):
        g = torch.from_numpy(np.load(root / m / probe / 'gram.npy')).double()                     # (12, N, N)
        g = g - g.mean(1, keepdim=True) - g.mean(2, keepdim=True) + g.mean((1, 2), keepdim=True)
        F[a * 12:(a + 1) * 12] = (g / g.flatten(1).norm(dim=1)[:, None, None]).flatten(1).float()
    G = torch.empty((n * 12, n * 12), dtype=torch.float64)
    for at in range(0, n * 12, 24):
        G[at:at + 24] = (F[at:at + 24] @ F.T).double()
    res['cka'] = G.numpy().reshape(n, 12, n, 12).transpose(0, 2, 1, 3)
    for name, p in perms.items():                 # B's images permuted on both axes of its Gram (same-site only)
        pt = torch.from_numpy(p)
        Fp = F.view(n * 12, N, N)[:, pt][:, :, pt].reshape(n * 12, N * N)
        Gp = torch.empty((n * 12, n * 12), dtype=torch.float64)
        for at in range(0, n * 12, 24):
            Gp[at:at + 24] = (F[at:at + 24] @ Fp.T).double()
        res[f'cka_{name}'] = Gp.numpy().reshape(n, 12, n, 12).transpose(0, 2, 1, 3)[:, :, diag, diag]
        del Fp, Gp
    del F, G

    # ---- operators
    for kname in KERNELS:
        ops = {c: [np.load(root / m / probe / f'op_{kname}_{c}.npz') for m in nets] for c in CHAINS}
        U = np.stack([o['site_vec'].astype(np.float64) for o in ops['full']])                     # (n, 12, N, 16)
        res[f'{kname}_lam2'] = np.stack([o['lam2'] for o in ops['full']])
        for r in (8, 16):
            res[f'{kname}_walk{r}'] = (np.einsum('alik,blij->ablkj', U[..., :r], U[..., :r]) ** 2).sum((3, 4)) / r
        for name, p in perms.items():
            res[f'{kname}_walk8_{name}'] = (np.einsum('alik,blij->ablkj', U[..., :8], U[:, :, p, :8]) ** 2).sum((3, 4)) / 8
        del U
        for chain in CHAINS:
            V = np.stack([o['vecs'] for o in ops[chain]])                                         # (n, L, N, 32)
            L = V.shape[1]
            pre = f'{kname}_{chain}_'
            res[pre + 'vals'] = np.stack([o['vals'] for o in ops[chain]])
            res[pre + 't'] = np.array([float(o['t']) for o in ops[chain]])
            res[pre + 'c'] = np.stack([o['c'] for o in ops[chain]])
            Vf = V.reshape(n, L * N, 32)
            cross = np.einsum('bij,aik->abjk', Vf, Vf)                                            # (A, B, j of B, k of A)
            res[pre + 'share'] = np.stack([(cross[:, :, :K, :K] ** 2).sum((2, 3)) / K for K in (4, 8, 16, 32)], -1)
            res[pre + 'sv'] = np.linalg.svd(cross[:, :, :16, :16], compute_uv=False)
            res[pre + 'modes'] = np.abs(cross[:, :, :16, :16]).astype(np.float32)
            ss = np.array([[site_share(V[a, :, :, :16], V[b, :, :, :16], train, test) for b in range(n)] for a in range(n)])
            res[pre + 'site'], res[pre + 'cos'] = ss[:, :, 0], ss[:, :, 1]
            for name, p in perms.items():
                Vp = V[:, :, p]
                cp = np.einsum('bij,aik->abjk', Vp.reshape(n, L * N, 32), Vf)
                res[pre + f'share_{name}'] = np.stack([(cp[:, :, :K, :K] ** 2).sum((2, 3)) / K for K in (4, 8, 16, 32)], -1)
                ss = np.array([[site_share(V[a, :, :, :16], Vp[b, :, :, :16], train, test) for b in range(n)] for a in range(n)])
                res[pre + f'site_{name}'], res[pre + f'cos_{name}'] = ss[:, :, 0], ss[:, :, 1]
            del V, Vf, cross
    np.savez(out / f'{probe}.npz', **res)
    info = dict(probe=probe, nets=nets, seconds=time.time() - t0)
    (out / f'{probe}.json').write_text(json.dumps(info))
    return info
