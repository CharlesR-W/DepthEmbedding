"""Double diffusion maps: ports agree with dm.py, identities of the cut-off scores, and a manifold with a known answer.

Run: uv run --with numpy --with scipy --with torch --with pytest python -m pytest -q invert/test_ddm.py
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ddm
import dm

DT = torch.float64


def T(a):
    return torch.as_tensor(a, dtype=DT)


# ------------------------------------------------------------------ ports of the per-point-width kernel

@pytest.fixture(scope='module')
def cloud():
    rng = np.random.default_rng(0)
    x = rng.standard_normal((260, 10)) * np.linspace(3, 0.3, 10)
    x -= x[:200].mean(0)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x[:200] @ x[:200].T, x[200:] @ x[:200].T


@pytest.mark.parametrize('renyi', [1, 2])
def test_kernel_port_matches_numpy(cloud, renyi):
    C, Cs = cloud
    ref = dm.ea_kernel(C, 20, renyi=renyi)
    ker = ddm.pr_kernel(T(1 - C), 20, renyi=renyi)
    assert abs(ker['P'].numpy() - ref['P']).max() < 1e-9
    assert abs(ker['P'].sum(1) - 1).abs().max() < 1e-9
    rows_ref = dm.ea_oos_rows(Cs, ref)
    rows = ddm.pr_oos_rows(T(1 - Cs), ddm.ker_to_torch(ref, 'cpu'))
    assert abs(rows.numpy() - rows_ref).max() < 1e-9
    d = T(1 - C); d.fill_diagonal_(float('inf'))                     # a probe point, masked, replays its own row
    assert abs(ddm.pr_oos_rows(d, ker) - ker['P']).max() < 1e-9


# ------------------------------------------------------------------ identities

def test_cut_and_ridge_scores_equal_explicit_weights():
    g = torch.Generator().manual_seed(0)
    Z = torch.randn(150, 3, generator=g, dtype=DT)
    Zs = torch.randn(40, 3, generator=g, dtype=DT)
    X = torch.randn(150, 30, generator=g, dtype=DT); Xs = torch.randn(40, 30, generator=g, dtype=DT)
    base = torch.randn(40, 30, generator=g, dtype=DT)
    for kind, kw in (('gauss', dict(h=1.5)), ('pr', dict(k2=15))):
        lh = ddm.LatentHarmonics(Z, kind, **kw)
        C = lh.proj(Zs)
        cuts = [3, 20, min(90, lh.m_max)]
        for b0, df0 in ((None, 0), (base, 4)):
            sc = ddm.cut_scores(lh, X, cuts, dict(e=(C, Xs, b0)), df0=df0)
            for i, m in enumerate(cuts):
                Xh = lh.weights(Zs, m) @ X + (0 if b0 is None else b0)
                cos = (Xh * Xs).sum(1) / Xh.norm(dim=1) / Xs.norm(dim=1)
                cap = 1 - ((Xh - Xs) ** 2).sum(1) / (Xs ** 2).sum(1)
                assert abs(sc['e']['cap'][i] - cap).max() < 1e-8 and abs(sc['e']['cos'][i] - cos).max() < 1e-8
                # on the probe the truncated lift is the projection on the first m harmonics
                Pm = lh.U[:, :m] @ lh.U[:, :m].T
                rss = ((X - Pm @ X) ** 2).sum() / (X ** 2).sum()
                assert abs(float(sc['train'][i]) - float(rss)) < 1e-9
                assert abs(float(sc['gcv'][i]) - float(rss) / (1 - (m + df0) / 150) ** 2) < 1e-9
    lh = ddm.LatentHarmonics(Z, 'gauss', h=1.5)
    Kst = torch.exp(-ddm.sqd(Z, Z) / (2 * 1.5 ** 2)); ks = torch.exp(-ddm.sqd(Zs, Z) / (2 * 1.5 ** 2))
    rs = ddm.ridge_scores(lh, X, [1e-3, 1.0], dict(e=(lh.proj(Zs), Xs, None)))
    for i, rho in enumerate([1e-3, 1.0]):
        Wd = ks @ torch.linalg.inv(Kst + rho * torch.eye(150, dtype=DT))
        assert abs(lh.ridge_weights(Zs, rho) - Wd).max() < 1e-6
        hat = Kst @ torch.linalg.inv(Kst + rho * torch.eye(150, dtype=DT))
        rss = float(((X - hat @ X) ** 2).sum() / (X ** 2).sum())
        assert abs(rs['train'][i] - rss) < 1e-9 and abs(rs['df'][i] - float(hat.trace())) < 1e-7
        assert abs(rs['gcv'][i] - rss / (1 - float(hat.trace()) / 150) ** 2) < 1e-7
        assert abs(rs['e']['cap'][i] - ddm.vec_scores(Wd @ X, Xs)[1]).max() < 1e-6


def test_trend_overlap_gives_the_hat_trace():
    """trend, then lift of the residual: df from trend_overlap equals the trace of H + S (I - H)."""
    g = torch.Generator().manual_seed(7)
    Z = torch.randn(120, 3, generator=g, dtype=DT); X = torch.randn(120, 9, generator=g, dtype=DT)
    Fp = torch.cat([torch.ones(120, 1, dtype=DT), Z], 1)
    H = Fp @ torch.linalg.pinv(Fp); I = torch.eye(120, dtype=DT)
    lh = ddm.LatentHarmonics(Z, 'gauss', h=2.0)
    o = ddm.trend_overlap(lh, Fp)
    R = X - H @ X
    cuts = [5, 30, 70]
    sc = ddm.cut_scores(lh, R, cuts, {}, df0=4, overlap=o)
    for i, m in enumerate(cuts):
        Pm = lh.U[:, :m] @ lh.U[:, :m].T
        assert abs(float(sc['df'][i]) - float((H + Pm @ (I - H)).trace())) < 1e-8
    rs = ddm.ridge_scores(lh, R, [1e-2], {}, df0=4, overlap=o)
    S = lh.U @ torch.diag(lh.w.clamp_min(0) / (lh.w.clamp_min(0) + 1e-2)) @ lh.U.T
    assert abs(rs['df'][0] - float((H + S @ (I - H)).trace())) < 1e-8


def test_gcv_is_undefined_at_interpolation():
    """At m = n the fit interpolates: GCV is 0/0 and must not be selectable."""
    g = torch.Generator().manual_seed(4)
    Z = torch.randn(60, 2, generator=g, dtype=DT); X = torch.randn(60, 5, generator=g, dtype=DT)
    lh = ddm.LatentHarmonics(Z, 'gauss', h=0.3)
    assert lh.m_max == 60
    sc = ddm.cut_scores(lh, X, [10, 59, 60], {})
    assert torch.isfinite(sc['gcv'][:2]).all() and sc['gcv'][2] == float('inf') and float(sc['train'][2]) == 0
    assert int(sc['gcv'].argmin()) != 2
    sc = ddm.cut_scores(lh, X, [10, 57, 58], {}, df0=3)
    assert sc['gcv'][1] == float('inf')


def test_scores_stay_valid_with_near_duplicates():
    """A near-duplicate pair makes the kernel ill-conditioned; scores from vectors remain a true cosine."""
    g = torch.Generator().manual_seed(5)
    Z = torch.randn(80, 2, generator=g, dtype=DT); Z[1] = Z[0] + 1e-7
    X = torch.randn(80, 12, generator=g, dtype=DT); Zs = torch.randn(20, 2, generator=g, dtype=DT)
    Xs = torch.randn(20, 12, generator=g, dtype=DT)
    lh = ddm.LatentHarmonics(Z, 'gauss', h=0.5)
    sc = ddm.cut_scores(lh, X, list(range(1, lh.m_max + 1)), dict(e=(lh.proj(Zs), Xs, None)))
    assert torch.isfinite(sc['e']['cos']).all() and float(sc['e']['cos'].abs().max()) <= 1 + 1e-9
    assert float(sc['e']['cap'].max()) <= 1 + 1e-9


def test_llr_survives_an_isolated_anchor():
    g = torch.Generator().manual_seed(6)
    Z = torch.randn(300, 4, generator=g, dtype=DT)
    Z[0] = 1e4                                                           # every other point is ~1e4 bandwidths away
    r, _ = ddm.llr_residuals(Z, n_anchor=300)
    assert torch.isfinite(r).all()


def test_full_rank_gauss_interpolates():
    g = torch.Generator().manual_seed(1)
    Z = torch.randn(80, 2, generator=g, dtype=DT)
    f = torch.sin(Z[:, :1]) + Z[:, 1:] ** 2
    lh = ddm.LatentHarmonics(Z, 'gauss', h=0.3)
    assert lh.m_max == 80
    assert abs(lh.weights(Z, 80) @ f - f).max() < 1e-6


def test_cosines():
    g = torch.Generator().manual_seed(2)
    X = torch.randn(50, 20, generator=g, dtype=DT); W = torch.randn(7, 50, generator=g, dtype=DT)
    Xh = W @ X
    Xn = X / X.norm(dim=1, keepdim=True)
    ref = (Xh / Xh.norm(dim=1, keepdim=True)) @ Xn.T
    assert abs(ddm.cosines(Xh, Xn) - ref).max() < 1e-10


# ------------------------------------------------------------------ a manifold with a known answer

@pytest.fixture(scope='module')
def strip():
    """A 4 x 1 strip, curved into R^6.  Intrinsic coordinates (u, v); diffusion eigenvectors along u come first
    (cos(pi u / 4), then its harmonics cos(2 pi u / 4), ...) and the first one along v arrives after several."""
    rng = np.random.default_rng(3)
    n, m = 1400, 300
    u = rng.uniform(0, 4, n + m); v = rng.uniform(0, 1, n + m)
    X = np.c_[np.cos(u * 0.9), np.sin(u * 0.9), v, 0.3 * u * v, 0.2 * np.sin(2 * v) * np.cos(u), 0.1 * u ** 2]
    D = T(X); D2 = ddm.sqd(D, D)
    scale = float(D2[:n, :n].median())
    ker = ddm.pr_kernel(D2[:n, :n] / scale, 25)
    lam, Phi = ddm.site_coords(ker['P'], 12)
    rows = ddm.pr_oos_rows(D2[n:, :n] / scale, ker)
    Phis = rows @ Phi / lam
    return dict(X=D[:n], Xs=D[n:], u=u, v=v, n=n, ker=ker, lam=lam, Phi=Phi, Phis=Phis, scale=scale)


def test_llr_flags_harmonics(strip):
    """Exactly two of the leading eigenvectors are new directions; the rest are functions of earlier ones."""
    Phi, u, v, n = strip['Phi'], strip['u'][:strip['n']], strip['v'][:strip['n']], strip['n']
    r, _ = ddm.llr_residuals(Phi[:, :8], n_anchor=400)
    corr = lambda a, b: abs(np.corrcoef(a, b)[0, 1])
    kv = int(np.argmax([corr(Phi[:, k].numpy(), np.cos(np.pi * v)) for k in range(8)]))   # first mode across the strip
    assert corr(Phi[:, 0].numpy(), np.cos(np.pi * u / 4)) > 0.95 and kv >= 2
    assert float(r[kv]) > 0.5, (kv, r)
    others = [float(r[k]) for k in range(1, 8) if k != kv]
    assert max(others) < 0.35, (kv, r)
    sel = ddm.select_parsimonious(r, 2)
    assert sel.tolist() == [0, kv]


def test_double_dm_lifts_and_round_trips(strip):
    """From the two non-harmonic coordinates: the lift reconstructs held-out points far better than a linear map,
    and lifting then restricting returns the coordinates."""
    Phi, Phis, X, Xs, n = strip['Phi'], strip['Phis'], strip['X'], strip['Xs'], strip['n']
    r, _ = ddm.llr_residuals(Phi[:, :8], n_anchor=400)
    sel = ddm.select_parsimonious(r, 2)
    Z, Zs = Phi[:, sel], Phis[:, sel]
    mu = X.mean(0); Xc, Xsc = X - mu, Xs - mu
    s0 = ddm.knn_scale(Z)
    best = None
    for hm in (2, 4, 8):                                              # cut-off by generalised cross-validation
        lh = ddm.LatentHarmonics(Z, 'gauss', h=hm * s0)
        cuts = sorted({lh.n_keep(10 ** (-j / 2)) for j in range(1, 25)} - {0})
        cuts = [m for m in cuts if m <= n // 2]
        sc = ddm.cut_scores(lh, Xc, cuts, dict(e=(lh.proj(Zs), Xsc, None)))
        i = int(sc['gcv'].argmin())
        if best is None or float(sc['gcv'][i]) < best[0]:
            best = (float(sc['gcv'][i]), lh, cuts[i], float(sc['e']['cap'][i].mean()))
    _, lh, m, cap = best
    W = lh.weights(Zs, m)
    err = ((W @ Xc - Xsc) ** 2).sum() / (Xsc ** 2).sum()
    F = torch.cat([torch.ones(n, 1, dtype=DT), Z], 1); Fs = torch.cat([torch.ones(len(Zs), 1, dtype=DT), Zs], 1)
    err_lin = ((Fs @ torch.linalg.pinv(F) @ Xc - Xsc) ** 2).sum() / (Xsc ** 2).sum()
    assert float(err) < 0.01 and float(err) < 0.1 * float(err_lin), (float(err), float(err_lin), m)
    assert abs(cap - (1 - float(err))) < 0.02
    # restriction of the lifted points (this first stage is a kernel on Euclidean distance, so it needs the points)
    Xh = W @ Xc + mu
    rows = ddm.pr_oos_rows(ddm.sqd(Xh, X) / strip['scale'], strip['ker'])
    back = (rows @ Phi / strip['lam'])[:, sel]
    miss = (((back - Zs) ** 2).mean(0) / Zs.var(0)).mean()
    assert float(miss) < 0.03, float(miss)          # 2 = two unrelated points
