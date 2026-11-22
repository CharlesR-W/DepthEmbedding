"""Exact spectral-band diffusion wavelets with pivoted-QR bases (wavelets/SPEC_DW.md, "Conventions").

Input: eigenpairs of a symmetric operator T, T U = U diag(lam), lam descending and positive, U with
Euclidean-orthonormal columns (n, r).  A *band* is a contiguous run of eigen-indices [s, e).  The localised basis
of a band is Phi = U[:, s:e] Q with Q orthogonal, from the column-pivoted QR of the band-filtered deltas
C = diag(lam[s:e]) U[:, s:e]^T (one column per point).  The spaces are those of Coifman & Maggioni (2006),
Eqs. 4.9-4.11; the bases are not their recursive compressed QR (see `cm_tree` for a small-n port used as a control).

Out-of-sample values for a kernel row p (m, n):   z_Phi = (p U[:, s:e] / lam[s:e]) Q      (divide, then rotate).

Everything here is numpy float64; bands of a few thousand functions take seconds to minutes (LAPACK geqp3).
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def cut_index(lam: np.ndarray, c: float, rtol: float = 1e-9, edge: bool = False) -> int:
    """Number of leading eigenvalues kept by the cut-off lam >= c, never splitting a (numerically) repeated one.

    The tolerance is rtol * |lam[0]| by default; edge=True makes it rtol * |lam[k-1]| (relative to the eigenvalue at
    the edge), which keeps distinct small eigenvalues apart near a low floor (with lam[0] = 1 and a floor of 1e-10,
    the default tolerance 1e-9 would merge every eigenvalue below ~1e-9 into one block)."""
    k = int((lam >= c).sum())
    while 0 < k < len(lam) and lam[k - 1] - lam[k] <= rtol * abs(lam[k - 1] if edge else lam[0]):
        k += 1                                              # keep the whole multiplicity block
    return k


def levels(lam: np.ndarray, cuts, ranks=(), edge: bool = False) -> list[dict]:
    """Nested bands V = [0, k) for cut-offs `cuts` (any order) and fixed `ranks`; empty and repeated ones dropped.

    Returns dicts sorted by rank: k, cut (smallest kept eigenvalue's threshold, or None for a fixed rank),
    lam_min, amp (= 1 / lam_min, the largest Nystrom reciprocal), gap (lam[k-1] - lam[k], relative to lam[0]).
    """
    out = {}
    for c in cuts:
        k = cut_index(lam, c, edge=edge)
        if k > 0:
            out.setdefault(k, dict(k=k, cut=float(c), fixed=False))
    for k in ranks:
        if 0 < k <= len(lam):
            k = cut_index(lam, lam[k - 1], edge=edge)
            out.setdefault(k, dict(k=k, cut=None, fixed=True))
    res = []
    for k in sorted(out):
        d = out[k]
        d.update(lam_min=float(lam[k - 1]), amp=float(1 / lam[k - 1]),
                 gap=float((lam[k - 1] - lam[k]) / abs(lam[0])) if k < len(lam) else float('nan'))
        res.append(d)
    return res


def band_basis(U: np.ndarray, lam: np.ndarray, s: int, e: int, weights: np.ndarray | None = None):
    """Localised orthonormal basis of the band [s, e): returns Q (r, r) orthogonal and piv (r,) seed points.

    Phi = U[:, s:e] @ Q.  Column j of Phi is the orthonormalised band-filtered delta at point piv[j], in the greedy
    order of largest remaining column norm; its sign makes Phi[piv[j], j] >= 0.  The filter is `weights` (e - s,)
    when given (e.g. exp(-t mu) for a heat bump at time t), else lam[s:e]: one application of T, whatever
    diffusion time the band's edge corresponds to.
    """
    wgt = lam[s:e] if weights is None else weights
    C = wgt[:, None] * U[:, s:e].T                           # (r, n): column x = coefficients of f(T)_band delta_x
    Q, _, piv = qr(C, mode='economic', pivoting=True)        # C[:, piv] = Q R
    piv = piv[:e - s]
    sgn = np.sign(np.einsum('ij,ij->j', U[piv, s:e].T, Q))   # Phi[piv[j], j]
    sgn[sgn == 0] = 1
    return Q * sgn, piv


def random_basis(r: int, rng) -> np.ndarray:
    """Haar-random orthogonal (r, r): the control for 'does localisation matter'."""
    q, t = np.linalg.qr(rng.standard_normal((r, r)))
    return q * np.sign(np.diag(t))


class Multiscale:
    """Coarsest scaling band [0, k_0) plus wavelet bands [k_i, k_{i+1}), each with its own localised basis.

    `ks` ascending ranks (k_0 < k_1 < ... <= r).  `kind`: 'qr' (pivoted-QR localisation), 'rand' (random rotation
    within each band) or 'eig' (identity: the eigenvectors themselves).  Atoms are ordered band by band, so atom
    index a in [k_{i-1}, k_i) belongs to band i; `rot` applies the block rotation to eigen-coordinates.
    """

    def __init__(self, U, lam, ks, kind='qr', seed=0):
        ks = [int(k) for k in ks]
        assert all(a < b for a, b in zip(ks, ks[1:])) and 0 < ks[0] and ks[-1] <= U.shape[1]
        self.ks, self.kind = ks, kind
        self.bounds = list(zip([0] + ks[:-1], ks))
        rng = np.random.default_rng(seed)
        self.Q, self.piv = [], []
        for s, e in self.bounds:
            if kind == 'qr':
                q, p = band_basis(U, lam, s, e)
            elif kind == 'rand':
                q, p = random_basis(e - s, rng), np.full(e - s, -1)
            else:
                q, p = np.eye(e - s), np.full(e - s, -1)
            self.Q.append(q); self.piv.append(p)
        self.r = ks[-1]
        self.band = np.concatenate([np.full(e - s, i) for i, (s, e) in enumerate(self.bounds)])
        self.pivot = np.concatenate(self.piv)

    def rot(self, A: np.ndarray) -> np.ndarray:
        """Eigen-coordinates A (m, >= r) -> atom coordinates (m, r): A[:, band] @ Q_band, band by band."""
        return np.concatenate([A[:, s:e] @ q for (s, e), q in zip(self.bounds, self.Q)], 1)

    def basis(self, U: np.ndarray) -> np.ndarray:
        """Phi (n, r) on the probe points."""
        return self.rot(U)


def extend(pU: np.ndarray, lam: np.ndarray, ms: Multiscale | None = None, t1: bool = False) -> np.ndarray:
    """Out-of-sample coordinates from pU = p @ U (m, r): divide by eigenvalues in the eigenbasis, then rotate.

    t1=True skips the division (the smoothed, one-kernel-step convention)."""
    A = pU if t1 else pU / lam[None, :pU.shape[1]]
    return A if ms is None else ms.rot(A)


def localisation(Phi: np.ndarray, nbr: np.ndarray | None = None, pivot: np.ndarray | None = None, share: float = 0.9):
    """Per column: n90 = number of points holding `share` of the squared mass (centre-free); near_peak = share of
    squared mass on the function's own absolute peak and that point's neighbour set nbr[peak] (n, q), the same
    rule for every basis; near_seed = the same about the QR seed `pivot` (nan where pivot < 0), a diagnostic of
    the seed as a label, not a comparison between bases."""
    e = Phi ** 2
    e = e / e.sum(0, keepdims=True)
    srt = -np.sort(-e, 0)
    n90 = (np.cumsum(srt, 0) < share).sum(0) + 1
    r = Phi.shape[1]
    near_peak, near_seed = np.full(r, np.nan), np.full(r, np.nan)
    if nbr is not None:
        peak = e.argmax(0)
        for j in range(r):
            near_peak[j] = e[nbr[peak[j]], j].sum() + e[peak[j], j]
            if pivot is not None and pivot[j] >= 0:
                near_seed[j] = e[nbr[pivot[j]], j].sum() + e[pivot[j], j]
    return n90, near_peak, near_seed


def best_n(B2: np.ndarray) -> np.ndarray:
    """Atom order by decreasing coefficient energy B2 (r,) = sum over target dimensions of B^2."""
    return np.argsort(-B2, kind='stable')


# ------------------------------------------------------------------ control: the paper's recursive construction

def cm_tree(T: np.ndarray, eps: float, levels_max: int = 30, symm: bool = True):
    """Small-n port of Coifman & Maggioni's diffusion wavelet tree (dense, no entry thresholding).

    At level j the operator T^(2^j) is held in the scaling basis Phi_j; a column-pivoted QR keeps the columns whose
    *remaining column norm* exceeds eps (as gsqr.m does), giving Phi_{j+1} inside span(Phi_j).  The next operator is
    Op Op^T with Op = Q^T T_j (symm=True: the reference DWPTree.m symmetric branch, `T = Op*Op'`), or the square of
    Q^T T_j Q (symm=False: its general branch).  Returns the scaling bases Phi_j (n, k_j), j = 1.., in point coordinates.
    """
    n = len(T)
    Phi, Tj, out = np.eye(n), (T + T.T) / 2, []
    for _ in range(levels_max):
        Q, R, _ = qr(Tj, mode='economic', pivoting=True)
        d = np.abs(np.diag(R))                              # remaining column norm at each pivot step
        k = int((d > eps).sum())
        if k == 0:
            break
        Q = Q[:, :k]
        Phi = Phi @ Q
        if symm:
            Op = Q.T @ Tj
            Tj = Op @ Op.T
        else:
            Tn = Q.T @ Tj @ Q
            Tj = Tn @ Tn
        Tj = (Tj + Tj.T) / 2
        out.append(Phi)
        if k == 1:
            break
    return out
