"""Double diffusion maps and latent harmonics (Evangelou et al., J. Comput. Phys. 2023, arXiv:2204.12536).

Torch, float64, device-agnostic.  The pieces:

  first stage     any diffusion-map coordinates Z (n, d) of the probe; `site_coords` gives the classic ones from a
                  doubly stochastic walk P, and `pr_oos_rows` the walk's out-of-sample row for the restriction.
  selection       `llr_residuals`: the local-linear-regression test of Dsilva et al. (2018) for which coordinates
                  are new directions and which are harmonics (functions of earlier ones).
  second stage    `LatentHarmonics`: a kernel on Z, its eigenpairs (sigma_j, psi_j), and the geometric-harmonics
                  extension.  The lift of a function f on the probe to a new latent point z is
                      f(z) = sum_{j <= m} sigma_j^-1 (k*(z) . psi_j) <psi_j, f>,
                  i.e. weights W(z) = k*(z) Psi_m Sigma_m^-1 Psi_m^T on the probe values.
  cut-off         `cut_scores`: held-out scores and generalised cross-validation for every cut-off m at once.

Kernels: 'gauss' is the paper's K*_ij = exp(-|z_i - z_j|^2 / (2 h^2)).  'pr' is this repo's first-stage rule applied
to latent distances (per-point width set by participation ratio, symmetrised, Sinkhorn): `pr_kernel` /
`pr_oos_rows` are torch ports of dm.ea_kernel(renyi=2) / dm.ea_oos_rows and are tested against them.
"""
from __future__ import annotations

import math

import torch

DT = torch.float64
INF = float('inf')


def sqd(a, b):
    return torch.cdist(a, b).pow(2)


def knn_scale(z, k=15):
    """Median distance to the k-th nearest other point (same convention as stage_decode.knn_scale)."""
    return float(torch.cdist(z, z).kthvalue(k + 1, dim=1).values.median())


# ------------------------------------------------------------------ per-point-width kernel (port of dm.py)

def cond_rows(d, perp, mask_self=False, renyi=2):
    """Per-row Gaussian conditionals on distances d (m, n) with effective neighbour count `perp`.

    renyi=2: participation ratio 1 / sum p^2;  renyi=1: Shannon perplexity.  Returns beta, dmin, logz, logp."""
    d = d.clone()
    if mask_self:
        d.fill_diagonal_(INF)
    dmin = d.min(1).values
    ds = d - dmin[:, None]
    fin = torch.where(torch.isfinite(ds), ds, torch.zeros_like(ds))
    target = math.log(perp)

    def cond(beta):
        lg = -beta[:, None] * ds
        logz = torch.logsumexp(lg, 1)
        logp = lg - logz[:, None]
        if renyi == 2:
            ent = -torch.logsumexp(2 * logp, 1)
        else:
            ent = logz + beta * (logp.exp() * fin).sum(1)
        return ent, logp, logz

    lo = torch.zeros(len(d), dtype=d.dtype, device=d.device)
    hi = torch.ones_like(lo)
    for _ in range(60):
        bad = cond(hi)[0] > target
        if not bool(bad.any()):
            break
        hi = torch.where(bad, hi * 2, hi)
    for _ in range(50):
        beta = (lo + hi) / 2
        ent = cond(beta)[0]
        lo = torch.where(ent > target, beta, lo)
        hi = torch.where(ent > target, hi, beta)
    beta = (lo + hi) / 2
    _, logp, logz = cond(beta)
    return beta, dmin, logz, logp


def sinkhorn_sym(K, tol=1e-11, max_iter=400):
    """v > 0 with v_i (K v)_i = 1 for a symmetric nonnegative K (damped fixed point, then Newton)."""
    v = torch.full((len(K),), 1 / math.sqrt(float(K.sum(1).mean())), dtype=K.dtype, device=K.device)
    err = INF
    for it in range(max_iter):
        r = v * (K @ v)
        err = float((r - 1).abs().max())
        if err < tol:
            break
        if it < 30 or err > 0.5:
            v = v / r.sqrt()
            continue
        A = v[:, None] * K * v[None, :] + torch.diag(r)
        du = torch.linalg.solve((A + A.T) / 2, -r * r.log())
        step, ok, v2 = 1.0, False, v
        while step > 1e-4:
            v2 = v * torch.exp((step * du).clamp(-50, 50))
            e2 = float((v2 * (K @ v2) - 1).abs().max())
            if math.isfinite(e2) and e2 < err:
                ok = True
                break
            step /= 2
        v = v2 if ok else v / r.sqrt()
    return v, err


def pr_kernel(d, perp, renyi=2):
    """Doubly stochastic symmetric walk with zero diagonal from distances d (n, n); see dm.ea_kernel."""
    beta, dmin, logz, logp = cond_rows(d, perp, True, renyi)
    p = logp.exp()
    K = (p + p.T) / 2
    K.fill_diagonal_(0)
    v, err = sinkhorn_sym(K)
    P = v[:, None] * K * v[None, :]
    return dict(P=(P + P.T) / 2, v=v, beta=beta, dmin=dmin, logz=logz, target=float(perp), renyi=renyi, err=err)


def pr_oos_rows(d, ker):
    """Out-of-sample rows (m, n) of new points at distances d (m, n) from the probe; see dm.ea_oos_rows.
    A probe point is replayed exactly when its distance to itself is set to +inf."""
    _, _, _, logp_x = cond_rows(d, float(ker['target']), False, int(ker['renyi']))
    logp_j = -ker['beta'][None, :] * (d - ker['dmin'][None, :]) - ker['logz'][None, :]
    logp_j = torch.where(torch.isfinite(d), logp_j, torch.full_like(logp_j, -INF))
    lw = torch.logaddexp(logp_x, logp_j) + ker['v'].log()[None, :]
    return torch.softmax(lw, 1)


def ker_to_torch(ker, device):
    """A kernel dict as stored by dm.build_kernel (numpy) -> the fields pr_oos_rows needs, as tensors."""
    t = lambda a: torch.as_tensor(a, dtype=DT, device=device)
    return dict(v=t(ker['v']), beta=t(ker['beta']), dmin=t(ker['dmin']), logz=t(ker['logz']),
                target=float(ker['target']), renyi=int(ker['renyi']))


# ------------------------------------------------------------------ first stage

def site_coords(P, K):
    """Leading K non-constant eigenpairs of a doubly stochastic walk P: lam (K,), Phi (n, K) with unit variance.

    Restriction (Nystrom) of a new point with out-of-sample row p: phi_k(x) = p . Phi[:, k] / lam[k]."""
    w, V = torch.linalg.eigh((P + P.T) / 2)
    w, V = w.flip(0), V.flip(1)
    assert abs(float(w[0]) - 1) < 1e-6, 'top eigenvalue of a doubly stochastic walk must be 1'
    return w[1:K + 1], V[:, 1:K + 1] * math.sqrt(len(P))


# ------------------------------------------------------------------ selection of non-harmonic coordinates

def llr_residuals(Z, n_anchor=1000, bw=1 / 3, seed=0, chunk=200, ridge=1e-10):
    """Dsilva et al. (2018) local linear regression residuals r_k for the columns of Z (n, K).

    r_k is the normalised leave-one-out error of column k predicted from columns 0..k-1 by a local linear fit with
    Gaussian weights exp(-d^2 / eps^2), eps = bw * median distance in those columns.  r_k near 1: a new direction;
    near 0: a harmonic (a function of the earlier ones).  r_0 = 1 by convention.  Evaluated at `n_anchor` random
    points against all n.  Also returns the mean effective number of neighbours (participation ratio of weights).
    """
    n, K = Z.shape
    g = torch.Generator().manual_seed(seed)
    A = torch.randperm(n, generator=g)[:n_anchor].to(Z.device)
    a = len(A)
    r = torch.ones(K, dtype=Z.dtype)
    neff = torch.full((K,), float(n), dtype=Z.dtype)
    D2 = torch.zeros(a, n, dtype=Z.dtype, device=Z.device)
    ar = torch.arange(a, device=Z.device)
    for k in range(1, K):
        D2 += (Z[A, k - 1][:, None] - Z[None, :, k - 1]) ** 2
        notself = torch.ones_like(D2, dtype=torch.bool)
        notself[ar, A] = False
        eps = max(bw * float(D2[notself].sqrt().median()), 1e-150)
        lw = (-D2 / eps ** 2).masked_fill(~notself, -INF)
        Wt = torch.exp(lw - lw.max(1, keepdim=True).values)                 # a row's scale does not change its fit
        y = Z[:, k]
        pred = torch.empty(a, dtype=Z.dtype, device=Z.device)
        for c in ar.split(chunk):
            F = torch.cat([torch.ones(len(c), n, 1, dtype=Z.dtype, device=Z.device),
                           Z[None, :, :k] - Z[A[c], None, :k]], 2)                      # (b, n, k + 1), centred at the anchor
            FW = F * Wt[c][:, :, None]
            M = FW.transpose(1, 2) @ F
            rhs = FW.transpose(1, 2) @ y[None, :, None].expand(len(c), n, 1)
            tr = M.diagonal(dim1=1, dim2=2).sum(1) / (k + 1)
            M = M + (ridge * tr)[:, None, None] * torch.eye(k + 1, dtype=Z.dtype, device=Z.device)
            pred[c] = torch.linalg.solve(M, rhs)[:, 0, 0]                               # intercept = value at the anchor
        r[k] = float(((y[A] - pred) ** 2).sum().sqrt() / (y[A] ** 2).sum().sqrt())
        neff[k] = float((Wt.sum(1) ** 2 / (Wt ** 2).sum(1).clamp_min(1e-300)).mean())
    return r, neff


def select_parsimonious(r, d):
    """Indices of the d coordinates with the largest residual, the first always kept, in their original order."""
    score = r.clone()
    score[0] = INF
    return torch.sort(torch.topk(score, d).indices).values


# ------------------------------------------------------------------ second stage

class LatentHarmonics:
    """Second diffusion map on latent coordinates Z (n, d) and its geometric-harmonics extension.

    kind='gauss': K* = exp(-|z - z'|^2 / (2 h^2)).  kind='pr': the per-point-width doubly stochastic walk with
    participation ratio k2 on squared latent distances.  Eigenpairs are sorted by descending eigenvalue; only
    those above `floor` times the largest are usable (the others are set to zero in `coef`)."""

    def __init__(self, Z, kind='gauss', h=None, k2=None, floor=1e-13):
        self.Z, self.kind, self.h, self.k2 = Z, kind, h, k2
        D = sqd(Z, Z)
        if kind == 'gauss':
            Ks = torch.exp(-D / (2 * h * h))
        else:
            self.ker = pr_kernel(D, k2)
            Ks = self.ker['P']
        w, U = torch.linalg.eigh(Ks)
        self.w, self.U = w.flip(0), U.flip(1)
        self.m_max = int((self.w > floor * self.w[0]).sum())

    def cross(self, Zs):
        D = sqd(Zs, self.Z)
        return torch.exp(-D / (2 * self.h * self.h)) if self.kind == 'gauss' else pr_oos_rows(D, self.ker)

    def n_keep(self, delta):
        """Number of harmonics with sigma_j > delta * sigma_1."""
        return min(int((self.w > delta * self.w[0]).sum()), self.m_max)

    def proj(self, Zs):
        """C (m, n): C[s, j] = k*(z_s) . psi_j."""
        return self.cross(Zs) @ self.U

    def coef(self, C):
        """A (m, n) from C = proj(Zs): A[s, j] = C[s, j] / sigma_j, the extended harmonic j at new point s."""
        A = C.clone()
        A[:, :self.m_max] /= self.w[:self.m_max]
        A[:, self.m_max:] = 0
        return A

    def weights(self, Zs, m, C=None):
        """W (m_s, n): the lift of any probe function f to the new points is W @ f (spectral cut-off at m)."""
        return self.coef(self.proj(Zs) if C is None else C)[:, :m] @ self.U[:, :m].T

    def ridge_weights(self, Zs, ridge, C=None):
        """W (m_s, n) of kernel ridge with the same kernel: k*(z) (K* + ridge I)^-1."""
        C = self.proj(Zs) if C is None else C
        return (C / (self.w.clamp_min(0) + ridge)) @ self.U.T


def vec_scores(Xh, Xs):
    """cos and captured share 1 - |x - x_hat|^2 / |x|^2 of decoded vectors Xh against true vectors Xs (m, D)."""
    n2 = (Xs ** 2).sum(1)
    cos = (Xh * Xs).sum(1) / ((Xh ** 2).sum(1) * n2).clamp_min(1e-300).sqrt()
    return cos, 1 - ((Xh - Xs) ** 2).sum(1) / n2


def trend_overlap(lh, Fp):
    """o (n,): o_j = |Q^T psi_j|^2 for an orthonormal basis Q of the trend space span(Fp).  The two-stage fit
    'trend, then lift of the residual' has hat matrix H + S (I - H), whose trace is rank(H) + sum_j s_j (1 - o_j)."""
    Q = torch.linalg.qr(Fp).Q
    return ((lh.U.T @ Q) ** 2).sum(1)


def cut_scores(lh, X, cuts, evals, df0=0, overlap=None):
    """Scores of the spectral cut-off lift at every cut-off in `cuts` (ascending ints).

    X (n, D): the target vectors at the latent points (centred activations, or their residual from a trend).
    evals: name -> (C, Xs, base): C = lh.proj(Zs); Xs (m, D) the true vectors; base (m, D) or None is added to the
    lift before scoring (the trend's own prediction).  Everything is computed from vectors, never from a Gram, so
    a decoded squared norm cannot go negative.
    Returns train (c,) = RSS_m / |X|^2 with RSS_m the energy of the discarded harmonics; df (c,) = m + df0, or
    m - sum_{j<=m} overlap_j + df0 when X is the residual of a trend (see trend_overlap); gcv (c,) =
    train / (1 - df / n)^2, infinite where df >= n (undefined at interpolation); per eval cos, cap (c, m)."""
    n = len(X)
    Y = lh.U.T @ X                                                              # (n, D) harmonic coefficients
    e = (Y ** 2).sum(1)
    tail = torch.cat([e.flip(0).cumsum(0).flip(0), e.new_zeros(1)])             # tail[m] = energy beyond the first m
    idx = torch.as_tensor(cuts, device=X.device)
    train = tail[idx] / e.sum()
    df = idx.to(X.dtype) + df0
    if overlap is not None:
        df = df - torch.cat([overlap.new_zeros(1), overlap.cumsum(0)])[idx]
    dof = 1 - df / n
    ok = (dof > 0) & (idx < n)
    out = dict(train=train, df=df, gcv=torch.where(ok, train / dof.clamp_min(1e-300) ** 2, torch.full_like(train, INF)))
    for name, (C, Xs, base) in evals.items():
        A = lh.coef(C)
        cos, cap = [], []
        for m in cuts:
            Xh = A[:, :m] @ Y[:m]
            c, p = vec_scores(Xh if base is None else Xh + base, Xs)
            cos.append(c); cap.append(p)
        out[name] = dict(cos=torch.stack(cos), cap=torch.stack(cap))
    return out


def ridge_scores(lh, X, ridges, evals, df0=0, overlap=None):
    """The same for kernel ridge with the same kernel, K* (K* + rho I)^-1, at every rho in `ridges`.
    Returns train, gcv (with df = trace of the hat matrix + df0), df, and per eval cos, cap."""
    n = len(X)
    Y = lh.U.T @ X
    e = (Y ** 2).sum(1)
    w = lh.w.clamp_min(0)
    out = dict(train=[], gcv=[], df=[])
    sc = {name: dict(cos=[], cap=[]) for name in evals}
    for rho in ridges:
        shrink = w / (w + rho)
        train = float((((1 - shrink) ** 2) * e).sum() / e.sum())
        df = float((shrink if overlap is None else shrink * (1 - overlap)).sum()) + df0
        out['train'].append(train); out['df'].append(df)
        out['gcv'].append(train / (1 - df / n) ** 2 if df < n else INF)
        for name, (C, Xs, base) in evals.items():
            Xh = (C / (w + rho)) @ Y
            c, p = vec_scores(Xh if base is None else Xh + base, Xs)
            sc[name]['cos'].append(c); sc[name]['cap'].append(p)
    for name in evals:
        out[name] = dict(cos=torch.stack(sc[name]['cos']), cap=torch.stack(sc[name]['cap']))
    return out


def cosines(Xh, Xn):
    """Cosines (m, n) of vectors Xh (m, D) to unit vectors Xn (n, D): what the first stage needs to restrict them."""
    return ((Xh @ Xn.T) / Xh.norm(dim=1, keepdim=True).clamp_min(1e-300)).clamp(-1, 1)
