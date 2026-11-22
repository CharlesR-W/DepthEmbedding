"""Controls with known answers for the diffusion-wavelet inversion (wavelets/SPEC_DW.md, "Controls", 1-7), plus the
torch localisation port and a CPU smoke run of invert/stage_dw.py on synthetic inputs.

Run: uv run --with numpy --with scipy --with torch --with pytest python -m pytest -q wavelets/test_dw.py
"""
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / 'invert')]
import dw  # noqa: E402
import ddm  # noqa: E402
import stage_decode as sd  # noqa: E402
import stage_dw  # noqa: E402

DT = torch.float64


def T(a):
    return torch.as_tensor(a, dtype=DT)


def proj(B):
    """Orthogonal projector onto span(B) (B with orthonormal columns)."""
    return B @ B.T


# ------------------------------------------------------------------ 1. divide in the eigenbasis, then rotate

def test_1_extension_divides_then_rotates():
    rng = np.random.default_rng(0)
    n = 12
    M = rng.standard_normal((n, n))
    M[:, 0], M[:, 1] = 0, 0
    M[:2, 0], M[:2, 1] = (1, 1), (1, -1)            # two eigenvectors sharing their largest entries
    U, _ = np.linalg.qr(M + 1e-3 * rng.standard_normal((n, n)))
    lam = np.r_[0.9, 0.4, np.full(n - 2, 0.01)]
    ms = dw.Multiscale(U, lam, [2], 'qr')
    Q = ms.Q[0]
    assert 0.2 < abs(Q[0, 0]) < 0.95, ('need a nontrivial rotation', Q)
    assert np.allclose(Q.T @ Q, np.eye(2), atol=1e-12)
    p = rng.random((5, n))
    want = (p @ U[:, :2] / lam[:2]) @ Q
    got = dw.extend(p @ U[:, :2], lam, ms)
    assert np.abs(got - want).max() < 1e-12
    Phi = ms.basis(U)
    wrong = (p @ Phi) / lam[:2]                    # dividing the rotated coordinates by eigenvalues is not the same
    assert np.abs(wrong - want).max() > 1e-2


# ------------------------------------------------------------------ 2. masked probe replay

def test_2_masked_probe_replay():
    rng = np.random.default_rng(1)
    x = rng.standard_normal((150, 6)) * np.linspace(2, .5, 6)
    x -= x.mean(0)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    d = T(1 - x @ x.T)
    ker = ddm.pr_kernel(d, 15)
    P = ker['P']
    w, V = torch.linalg.eigh(P)
    w, V = w.flip(0).numpy(), V.flip(1).numpy()
    lam, U = w[1:], V[:, 1:]
    r = int((lam > 1e-10).sum())
    lam, U = lam[:r], U[:, :r]
    lv = stage_dw.partition(lam)
    ks = [e['k'] for e in lv]
    ms = dw.Multiscale(U, lam, ks, 'qr')
    dm_ = d.clone()
    dm_.fill_diagonal_(float('inf'))
    rows = ddm.pr_oos_rows(dm_, ker).numpy()        # each probe point, itself masked
    assert np.abs(rows - P.numpy()).max() < 1e-9
    z = dw.extend(rows @ U, lam, ms)
    Phi = ms.basis(U)
    # on bands with moderate amplification the replay is exact to rounding times 1/lam
    k = ks[[i for i, e in enumerate(lv) if e['A'] <= 1e4][-1]]
    assert np.abs(z[:, :k] - Phi[:, :k]).max() < 1e-8


# ------------------------------------------------------------------ 3. ring graph

def ring(N, width):
    th = 2 * np.pi * np.arange(N) / N
    dd = np.angle(np.exp(1j * (th[:, None] - th[None, :])))
    K = np.exp(-dd ** 2 / (2 * width ** 2))
    return th, K / K[0].sum()


def test_3_ring_graph():
    N = 64
    width = 0.35
    th, Tm = ring(N, width)
    w, V = np.linalg.eigh(Tm)
    w, V = w[::-1], V[:, ::-1]
    r = int((w >= stage_dw.FLOOR * w[0]).sum())       # the positive sector above the floor
    lam_abs, w, V = w[:r].copy(), w[:r] / w[0], V[:, :r]
    # analytic eigenvalues of the circulant: DFT of its first row
    lam_k = np.real(np.fft.fft(Tm[0]))                # lam_k for frequency k
    freq = np.minimum(np.arange(N), N - np.arange(N))
    w = w / w[0]
    lam_rel = lam_k / lam_k[0]
    for A in (1.5, 4, 16, 1e2):
        k = dw.cut_index(w, 1 / A)
        kk = int((lam_rel >= 1 / A - 1e-12).sum())
        assert k == kk                                  # pairs (k, -k) kept together
        fs = sorted(set(freq[lam_rel >= 1 / A - 1e-12]))
        Pa = np.zeros((N, N))
        for f in fs:                                    # analytic Fourier projector
            Pa += (1 if f == 0 or 2 * f == N else 2) * np.cos(f * (th[:, None] - th[None, :])) / N
        Pn = proj(V[:, :k])
        assert np.abs(Pn - Pa).max() < 1e-9
        assert np.abs(np.roll(np.roll(Pn, 1, 0), 1, 1) - Pn).max() < 1e-9   # shift-equivariant
    lv = stage_dw.partition(w)
    ks = [e['k'] for e in lv if e['A'] <= 1e4]
    ms = dw.Multiscale(V, w, ks, 'qr')
    Phi = ms.basis(V)
    assert np.abs(Phi.T @ Phi - np.eye(ks[-1])).max() < 1e-10                 # orthonormal
    for i in range(1, len(ks)):                       # V_i = V_{i-1} (+) W_i
        s, e = ks[i - 1], ks[i]
        assert np.abs(proj(Phi[:, :e]) - proj(Phi[:, :s]) - proj(Phi[:, s:e])).max() < 1e-10
        assert np.abs(Phi[:, :s].T @ Phi[:, s:e]).max() < 1e-10
        assert np.abs(proj(Phi[:, :e]) - proj(V[:, :e])).max() < 1e-10
    f = np.random.default_rng(0).standard_normal(N)                       # Parseval with the omitted tail
    c = Phi.T @ f
    tail = f - Phi @ c
    assert abs((c ** 2).sum() + (tail ** 2).sum() - (f ** 2).sum()) < 1e-10
    # a band-limited signal: reconstructed on the ring and extended exactly to new points
    kb = ks[1]
    fsig = V[:, :kb] @ np.random.default_rng(1).standard_normal(kb)
    B = Phi[:, :kb].T @ fsig
    assert np.abs(Phi[:, :kb] @ B - fsig).max() < 1e-12
    thn = np.linspace(0, 2 * np.pi, 37, endpoint=False) + 0.05
    dd = np.angle(np.exp(1j * (thn[:, None] - th[None, :])))
    c0 = np.exp(-np.angle(np.exp(1j * (th[0] - th))) ** 2 / (2 * width ** 2)).sum()
    pn = np.exp(-dd ** 2 / (2 * width ** 2)) / c0            # kernel rows of new points, same kernel
    ms1 = dw.Multiscale(V, lam_abs, ks, 'qr')
    zn = dw.extend(pn @ V[:, :ks[-1]], lam_abs, ms1)[:, :kb]
    fh = np.fft.fft(fsig)                                    # truth: the trigonometric polynomial through the values
    truth = np.real(sum(fh[k] * np.exp(1j * (k if k <= N // 2 else k - N) * thn) for k in range(N) if abs(fh[k]) > 1e-9)) / N
    assert np.abs(zn @ (ms1.basis(V)[:, :kb].T @ fsig) - truth).max() < 1e-8


# ------------------------------------------------------------------ 4. the paper's recursive QR against exact bands

def test_4_cm_tree_matches_exact_bands():
    rng = np.random.default_rng(2)
    sizes = (8, 10, 12)
    n = sum(sizes)
    W = np.zeros((n, n))
    o = 0
    for s in sizes:                                    # dense clusters, weak links
        W[o:o + s, o:o + s] = 1
        o += s
    W += 0.01 * rng.random((n, n))
    W = (W + W.T) / 2
    np.fill_diagonal(W, 0)
    dg = W.sum(1)
    Tm = (np.eye(n) + W / np.sqrt(dg[:, None] * dg[None, :])) / 2   # lazy symmetric walk, spectrum in [0, 1]
    w, V = np.linalg.eigh(Tm)
    w, V = w[::-1], V[:, ::-1]
    eps = 1e-3
    report = {}
    for symm in (True, False):
        rows = []
        for j, Phi in enumerate(dw.cm_tree(Tm, eps, symm=symm)):
            pw = w ** (2 ** j)
            k_exact = int((pw > eps).sum())
            err = np.abs(proj(Phi) - proj(V[:, :k_exact])).max() if Phi.shape[1] == k_exact else np.inf
            sep = np.abs(np.log(pw / eps)).min() > np.log(10)          # every eigenvalue a factor 10 from eps
            rows.append((j, Phi.shape[1], k_exact, err, sep))
        report[symm] = rows
    print('cm_tree (level, rank, exact rank, projector error, separated):', report)
    # its threshold is on remaining column norms, not eigenvalues: ranks agree where the spectrum is separated at eps
    for symm, rows in report.items():
        sep = [r_ for r_ in rows if r_[4]]
        assert any(0 < r_[2] < n for r_ in sep), 'need a separated level that actually compresses'
        for j, k, ke, e, _ in sep:
            assert k == ke, (symm, j, k, ke)
            assert e < 1e-2, (symm, j, e)                  # its own tolerance: O(eps) leakage across the edge


# ------------------------------------------------------------------ 5, 6. span invariance; the second stage

@pytest.fixture(scope='module')
def latent():
    g = torch.Generator().manual_seed(0)
    Zp = torch.randn(300, 3, generator=g, dtype=DT)
    Zh = torch.randn(60, 3, generator=g, dtype=DT)
    f = lambda Z: torch.stack([torch.sin(Z[:, 0]), Z[:, 1] * Z[:, 2], torch.cos(Z.sum(1)), Z[:, 0] ** 2], 1)
    Xp = f(Zp) @ torch.randn(4, 20, generator=g, dtype=DT)
    return Zp, Zh, Xp - Xp.mean(0)


def test_5_span_invariance_and_linear_decode():
    rng = np.random.default_rng(3)
    x = rng.standard_normal((200, 8)) * np.linspace(2, .3, 8)
    xh = rng.standard_normal((40, 8)) * np.linspace(2, .3, 8)
    mu = x.mean(0)
    xc, xhc = x - mu, xh - mu
    nrm = lambda a: a / np.linalg.norm(a, axis=1, keepdims=True)
    ker = ddm.pr_kernel(T(1 - nrm(xc) @ nrm(xc).T), 20)
    rows = ddm.pr_oos_rows(T(1 - nrm(xhc) @ nrm(xc).T), ker)
    w, V = torch.linalg.eigh(ker['P'])
    w, V = w.flip(0), V.flip(1)
    r = int((w[1:] > 1e-10).sum())
    lam, U = w[1:1 + r], V[:, 1:1 + r]
    ks = [e['k'] for e in stage_dw.partition(lam.numpy())]
    X = T(xc)
    B = U.T @ X
    zH = rows @ U / lam
    R = stage_dw.BandRot(U.numpy(), lam.numpy(), ks, 'qr')
    zP, BP = R.rot(zH), R.rotT(B)
    for k in ks:
        Xu = zH[:, :k] @ B[:k]
        Xp_ = zP[:, :k] @ BP[:k]
        assert (Xu - Xp_).abs().max() / Xu.abs().max() < 1e-8
    for K in (4, 16):                                  # = the stored `site|K` linear decode (ddm.site_coords scaling)
        lam_s, Phi_s = ddm.site_coords(ker['P'], K)
        Xl = sd.w_linear(Phi_s, rows @ Phi_s / lam_s) @ X
        Xb = zH[:, :K] @ B[:K]
        assert (Xl - Xb).abs().max() / Xl.abs().max() < 1e-8


def test_6_fullband_equals_latent_harmonics_weights(latent):
    Zp, Zh, Xp = latent
    lift = stage_dw.Lift(Zp, Xp, h=1.0, rand_seeds=(0, 1))
    W = lift.lh.weights(Zh, lift.r) @ Xp
    for kind in ('lead', 'ebest', 'wbest', 'rbest'):
        Xh = lift.decode(Zh, kind, lift.r)
        assert (Xh - W).abs().max() / W.abs().max() < 1e-8, kind
    Xh = lift.decode(Zh, 'lead', 17)
    assert (Xh - lift.lh.weights(Zh, 17) @ Xp).abs().max() < 1e-10
    rho = 1e-3
    Wr = lift.lh.ridge_weights(Zh, rho) @ Xp
    assert (lift.ridge(Zh, rho) - Wr).abs().max() < 1e-10


def test_7_decoder_sees_only_latent_points(latent):
    Zp, Zh, Xp = latent
    lift = stage_dw.Lift(Zp, Xp, h=1.0)
    assert list(inspect.signature(lift.decode).parameters) == ['Zq', 'kind', 'n', 'adm']
    assert list(inspect.signature(lift.ridge).parameters) == ['Zq', 'rho']
    XH = torch.randn(len(Zh), Xp.shape[1], dtype=DT)
    before = {k: lift.decode(Zh, k, 30) for k in ('lead', 'wbest', 'rbest', 'ebest')}
    XH.fill_(float('nan'))                             # poison the query activations after encoding
    for k, v in before.items():
        assert torch.equal(lift.decode(Zh, k, 30), v)
    held = {id(t) for t in vars(lift).values() if torch.is_tensor(t)}
    assert id(XH) not in held


# ------------------------------------------------------------------ helpers of the stage

def test_loc_port_matches_numpy():
    rng = np.random.default_rng(4)
    Phi = np.linalg.qr(rng.standard_normal((120, 30)))[0]
    nbr = np.stack([rng.choice(np.delete(np.arange(120), i), 10, replace=False) for i in range(120)])
    piv = rng.integers(0, 120, 30)
    piv[3] = -1
    ref = dw.localisation(Phi, nbr, piv)
    got = stage_dw.loc_t(T(Phi), torch.as_tensor(nbr), piv, chunk=7)
    for a, b in zip(ref, got):
        assert np.allclose(np.asarray(a, float), b.numpy(), equal_nan=True)


def test_top_sv_both_sides():
    g = torch.Generator().manual_seed(6)
    for r, D in ((30, 8), (12, 40)):
        L = torch.randn(r, D, generator=g, dtype=DT)
        ks = [1, 3, 8, 9, 20, 30] if r == 30 else [1, 5, 12]
        for k, sv in stage_dw.top_sv(L, ks):
            ref = torch.linalg.svdvals(L[:k])[:min(5, k, D)]
            assert len(sv) == len(ref) and (sv - ref).abs().max() < 1e-10, (r, D, k)


def test_partition_frozen_and_admissible():
    # a pair straddling the cut 1/2 (A = 2) within the multiplicity tolerance, and a pair straddling the floor
    lam = np.r_[1.0, 0.9, 0.5 * (1 + 4e-10), 0.5 * (1 - 4e-10), 0.3, 0.05, 1e-3, 1e-6, 1e-9,
                1e-10 * (1 + 4e-10), 1e-10 * (1 - 4e-10), 1e-12, -0.2]
    lv = stage_dw.partition(lam)
    ks = [d['k'] for d in lv]
    assert ks == sorted(set(ks))
    assert 3 not in ks and 4 in ks                    # the block at 1/2 is kept whole
    assert ks[-1] == 11                               # the block at the floor wins over the hard floor
    assert lv[-1]['gap'] > 0 and lv[-1]['lam_min'] < 1e-10
    plain = dw.levels(lam, [0.5])                     # the default tolerance (relative to lam[0]) also keeps it
    assert plain[0]['k'] == 4
    assert dw.cut_index(lam, 2e-10) == 12             # ...but merges eigenvalues that differ by < 1e-9 lam[0]
    assert dw.cut_index(lam, 2e-10, edge=True) == 9
    assert stage_dw.rank_at(lam, 1e2, ks[-1]) == 6 and stage_dw.rank_at(lam, 1e4, ks[-1]) == 7
    for d in lv[:-1]:
        assert d['A'] >= d['amp'] - 1e-9


# ------------------------------------------------------------------ smoke run of the whole stage on synthetic inputs

class FakeModel(torch.nn.Module):
    def __init__(self, D):
        super().__init__()
        self.lin = torch.nn.Linear(D, 10).double()


def make_fake(tmp, n=240, m=60, D=24, L=12):
    rng = np.random.default_rng(5)
    R = Path(tmp)
    sites = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
    N = n + m
    lat = rng.standard_normal((N, 3))
    acts = np.tanh(lat @ rng.standard_normal((3, D)) + 0.3 * rng.standard_normal((N, D))).astype(np.float32)
    idx, hid = np.arange(n), np.arange(n, N)
    np.savez(R / 'splits.npz', P8=idx, H=hid)
    (R / 'acts/c10').mkdir(parents=True)
    np.save(R / 'acts/c10/b4.npy', acts)
    model = FakeModel(D)
    with torch.no_grad():
        lg = model.lin(torch.tensor(acts, dtype=DT)).numpy()
    np.save(R / 'acts/c10/logits.npy', lg)
    mu = acts[idx].astype(np.float64).mean(0)
    x = acts - mu
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    (R / 'gram/P8').mkdir(parents=True)
    np.save(R / 'gram/P8/b4.npy', (x @ x.T).astype(np.float32))
    np.savez(R / 'gram/P8/b4_norms.npz', mean=mu.astype(np.float32), c10=np.linalg.norm(acts - mu, axis=1))
    C = (x[idx] @ x[idx].T)
    ker = ddm.pr_kernel(T(1 - C), 30)
    (R / 'kernel/P8__pr400').mkdir(parents=True)
    np.savez(R / 'kernel/P8__pr400/b4.npz', P=ker['P'].numpy(), v=ker['v'].numpy(), beta=ker['beta'].numpy(),
             dmin=ker['dmin'].numpy(), logz=ker['logz'].numpy(), target=30.0, renyi=2)
    tag = 'P8__pr400__L12'
    (R / 'joint' / tag).mkdir(parents=True)
    vecs = rng.standard_normal((L, n, 64)).astype(np.float32)
    vals = np.arange(64.0)
    np.savez(R / 'joint' / tag / 'modes.npz', vecs=vecs, vals=vals, sites=np.array(sites))
    rows = ddm.pr_oos_rows(T((1 - x @ x.T).astype(np.float32)[hid][:, idx]), ker).numpy()
    on = np.zeros((L, m, 64), np.float32)
    on[4] = rows @ vecs[4]
    np.savez(R / 'joint' / tag / 'coords_H.npz', y=rng.standard_normal((L, m, 64)).astype(np.float32), onestep=on,
             basis=float(np.abs(vals).sum()), rowmax=np.full((L, m), 0.01))
    # stored DDM references computed the way stage_ddm computed them
    Xc = T(acts[idx] - acts[idx].mean(0)) if False else (torch.tensor(acts[idx]) - torch.tensor(mu.astype(np.float32))).double()
    XH = (torch.tensor(acts[hid]) - torch.tensor(mu.astype(np.float32))).double()
    F = T(vecs[4]); Cst = T((x @ x.T).astype(np.float32)[hid][:, idx])
    rowsS = ddm.pr_oos_rows(1 - Cst, ker)
    cells = {'joint|64': (F, T(np.load(R / 'joint' / tag / 'coords_H.npz')['y'][4])), 'source|64': (ker['P'] @ F, rowsS @ F)}
    combos, arrs = {}, {}
    for name, (Zp, Zh) in cells.items():
        s0 = sd.knn_scale(Zp)
        hm = 64 if name == 'source|64' else 8                     # 64: the stored rank lands above the 1e-10 floor
        lh = ddm.LatentHarmonics(Zp, 'gauss', h=hm * s0)
        r_dw = stage_dw.partition((lh.w / lh.w[0]).numpy())[-1]['k']
        mg = min(lh.m_max, r_dw + 3) if lh.m_max > r_dw else max(2, r_dw // 2)
        ksel = dict(h=2, ridge=1e-3)
        gk = sd.Gauss(Zp, 2 * s0)
        cl, _ = ddm.vec_scores(lh.weights(Zh, mg) @ Xc, XH)
        ck, _ = ddm.vec_scores(gk.w_krr(Zh, 1e-3) @ Xc, XH)
        combos[name] = dict(s0=s0, gcv_sel=dict(lh=dict(h=hm, m=mg)), krr_sel=ksel, finals=dict(krr=dict(cos=[float(ck.mean())])))
        arrs[f'{name}|lh|cos'] = cl.float().numpy()
        arrs[f'{name}|krr|cos'] = ck.float().numpy()
    for K in (4, 16, 64):
        lam_s, Phi_s = ddm.site_coords(ker['P'], K)
        cs, _ = ddm.vec_scores(sd.w_linear(Phi_s, rowsS @ Phi_s / lam_s) @ Xc, XH)
        combos[f'site|{K}'] = dict(finals=dict(linear=dict(cos=[float(cs.mean())])))
        arrs[f'site|{K}|linear|cos'] = cs.float().numpy()
    (R / 'ddm' / tag).mkdir(parents=True)
    (R / 'ddm' / tag / 'b4.json').write_text(json.dumps(dict(combos=combos)))
    np.savez(R / 'ddm' / tag / 'b4.npz', **arrs)
    mi = SimpleNamespace(R=str(R), index_set=lambda sp, name: sp[name], centre_set=lambda name: 'P8',
                         load_model=lambda: (model, None), SHAPES={'b4': (D,)},
                         suffix=lambda mod, site, h: mod.lin(h.double()))
    return mi


def test_stage_smoke_and_isolation(tmp_path, monkeypatch):
    """The whole stage on synthetic inputs: in-run controls pass, stored references are reproduced (including a stored
    GCV rank above the dictionary's floor), the JSON is strict, and changing the held-out activations while the
    encoded points stay fixed changes the scores but not one decoded vector (control 7 on the production path)."""
    mi = make_fake(tmp_path)
    monkeypatch.setattr(stage_dw, 'NSEEDS', 2)
    out = stage_dw.run(dict(pset='P8', kname='pr400', site='b4'), dict(device='cpu'), mi)
    assert out['site'] == 'b4'
    f = tmp_path / 'dw/P8__pr400__L12/b4.json'
    res = json.loads(f.read_text(), parse_constant=lambda c: pytest.fail(f'non-strict JSON constant {c}'))
    assert res['failed'] == [], res['failed']
    c = res['checks']
    assert set(c['q2_vs_linear']) == {str(K) for K in (4, 16, 64) if K <= res['site']['r']} and len(c['q2_vs_linear']) >= 2
    assert any(q['checks']['m_gcv_above_floor'] for q in res['q1'].values()), 'fake must exercise a stored rank above the floor'
    for name in ('joint|64', 'source|64'):
        fin = res['q1'][name]['finals']
        assert set(fin) >= {'lead|None', 'wbest|None', 'wbest|100.0', 'ebest|10000.0', 'rbest|None', 'ridge', 'krr', 'lead@gcv'}
        assert fin['lead@gcv']['n_eff'] == res['q1'][name]['m_gcv']
        for e in fin.values():
            assert e['n_eff'] is None or e['n_eff'] <= e['n_req']
        grid = [p['n'] for p in res['q1'][name]['curves']['lead|None']]
        assert {4, 16, 64} <= set(grid)
    for v in res['q3']['energy'].values():
        assert len(v['n']) == len(v['share']) and max(v['share']) <= 1 + 1e-9
    assert stage_dw.run(dict(pset='P8', kname='pr400', site='b4'), dict(device='cpu'), mi) == 'cached'
    o = json.loads(f.read_text())
    o['failed'] = ['x']
    f.write_text(json.dumps(o))
    assert not stage_dw.cached(f, f.with_suffix('.npz'), ['q1'], [])           # a failed unit is never a cache hit
    f.write_text(json.dumps(res))
    assert stage_dw.run(dict(pset='P8', kname='pr400', site='b4'), dict(device='cpu', parts=['q1'], suffix='_x'), mi) != 'cached'

    # control 7: perturb the held-out activations, keep the stored coordinates and cosines (the encoded points)
    a = np.load(tmp_path / 'acts/c10/b4.npy')
    hid = np.load(tmp_path / 'splits.npz')['H']
    a[hid] += np.random.default_rng(9).standard_normal(a[hid].shape).astype(np.float32)
    np.save(tmp_path / 'acts/c10/b4.npy', a)
    stage_dw.run(dict(pset='P8', kname='pr400', site='b4'), dict(device='cpu', parts=['q1'], suffix='_iso'), mi)
    res2 = json.loads((tmp_path / 'dw/P8__pr400__L12/b4_iso.json').read_text())
    for name in ('joint|64', 'source|64'):
        for k, e in res['q1'][name]['finals'].items():
            e2 = res2['q1'][name]['finals'][k]
            assert e2['fp'] == e['fp'], (name, k)
            assert e2['cos'] != e['cos'], (name, k)
