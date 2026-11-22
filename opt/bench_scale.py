"""Bounded throughput/fill probes. Run from repo root, with BLAS pinned.

OPENBLAS_NUM_THREADS=1 uv run --with numpy --with scipy python -B opt/bench_scale.py --fill
OPENBLAS_NUM_THREADS=1 uv run --with numpy --with scipy python -B opt/bench_scale.py --throughput
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy import sparse

from bench import ROOT, rss
import sparse_core as O


def direct_probe():
    for n in (250, 500, 1000):
        walks = []
        for l in range(12):
            gram = np.load(ROOT / f'data/grams/r20s0b/gram_64000_{l}.npy', mmap_mode='r')[:n, :n]
            a = O.adjacency(O.knn(gram))
            d, _, _ = O.balance_weights(a)
            walks.append(O.make_walk(a, d))
        joint = O.Joint(walks, np.ones(12))
        t = .65
        q = time.perf_counter()
        lu = O.SparseDirect(joint, t)
        setup = time.perf_counter() - q
        factors = lu.lu.nnz
        measured_nnz = lu.lu.L.nnz + lu.lu.U.nnz
        b = O.centre(np.random.default_rng(0).normal(size=(*joint.shape, 1)))
        q = time.perf_counter()
        y = lu.solve(b)
        secs = time.perf_counter() - q
        r = np.linalg.norm((joint.apply(y, t, lu.eps) - b).ravel()) / np.linalg.norm(b.ravel())
        print('fill: N', n, 'setup s', setup, 'solve s', secs, 'L+U actual/allocated nnz',
              measured_nnz, factors, 'relative residual', r, 'RSS MiB', rss(), flush=True)
        del lu


def synthetic_walk(n, rng):
    # Twelve random directed permutations yield approximately the observed
    # mean undirected degree (24.2). Loops are dropped after union.
    i = np.tile(np.arange(n), 12)
    j = np.concatenate([rng.permutation(n) for _ in range(12)])
    a = sparse.csr_matrix((np.ones(len(i)), (i, j)), shape=(n, n))
    a = a.maximum(a.T)
    a.setdiag(0)
    a.eliminate_zeros()
    a.data.fill(1.)
    d, _, _ = O.balance_weights(a)
    return O.make_walk(a, d)


def throughput():
    for n in (2000, 10000, 50000):
        q = time.perf_counter()
        rng = np.random.default_rng(27399)
        walks = [synthetic_walk(n, rng) for _ in range(12)]
        t, eps = 8.36, 1e-3
        joint = O.Joint(walks, np.linspace(.03, .30, 12))
        del walks
        pre = joint.path_preconditioner(t, eps)
        b = O.centre(rng.normal(size=joint.shape))
        print('throughput: N', n, 'build', time.perf_counter()-q, 'nnz', joint.base.nnz,
              'HWM MiB', rss(), flush=True)
        for tol in (5e-13,):
            q = time.perf_counter()
            steps = sum(O.pcg(lambda v: joint.apply(v, t, eps), b, pre, tol=tol)[1] for _ in range(3))
            print('throughput: N', n, 'cost per CG step', (time.perf_counter()-q)/steps,
                  'iterations (synthetic, not a prediction)', steps//3, 'HWM MiB', rss(), flush=True)
        del joint


def ritz_probe():
    rng = np.random.default_rng(5)
    for k in (16, 256):
        q, aq = (rng.normal(size=(12, 2000, k)) / np.sqrt(24000.) for _ in range(2))
        results = []
        for name, fn in [('einsum', lambda: np.einsum('lni,lnj->ij', q, aq)),
                         ('gemm', lambda: q.reshape(-1, k).T @ aq.reshape(-1, k))]:
            elapsed = []
            for _ in range(3):
                start = time.perf_counter()
                h = fn()
                elapsed.append(time.perf_counter() - start)
            results.append((name, np.median(elapsed), h))
        error = np.linalg.norm(results[0][2] - results[1][2]) / np.linalg.norm(results[0][2])
        print('Ritz contraction: K', k, [(name, t) for name, t, _ in results],
              'relative difference', error, 'HWM MiB', rss(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--fill', action='store_true')
    g.add_argument('--throughput', action='store_true')
    g.add_argument('--ritz', action='store_true')
    a = p.parse_args()
    if a.fill:
        direct_probe()
    elif a.throughput:
        throughput()
    else:
        ritz_probe()
