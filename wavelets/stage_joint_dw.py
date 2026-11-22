"""Diffusion-wavelet scaling functions of the viewer's own joint operator (wavelets/SPEC_DW.md, "Viewer", amendment 5).

Operator: `L_t` of spacetime.py, scale `own`, t = t* (coupling factor 1), kernel rule 0 (union 15-NN Sinkhorn walk), probe
P2 (N = 2,000, the viewer's order), L = 12 sites, ResNet-20 seed 0 step 64,000: exactly the shipped viewer modes.

Two parts:
  solve(ms, c, t, k)   (Modal, wavelets/modal_joint_dw.py): the k smallest eigenpairs of L_t on slice-mean-zero functions with
                       the projected solver `spacetime.modes` (never an ambient solve: at balance coupling the first temporal
                       eigenvalue equals the first slice-mean-zero one), residuals, slice means, and the level edges chosen
                       from that spectrum, each checked for completeness by an independent Sylvester inertia count of the
                       block LDL^T of L_t - sigma I (minus the analytic temporal eigenvalues below sigma).
  stage(eig)           (local): compares the leading 16 modes with the shipped viewer modes, builds each level's pivoted-QR
                       scaling functions filtered by exp(-t_j mu), t_j = ln 2 / mu_c (dw.band_basis weights), and per-function
                       metadata; writes data/wavelets/joint_dw_<step>.npz (float64) and .json (checks and level table).

    uv run --with numpy --with scipy python wavelets/stage_joint_dw.py stage      # after modal_joint_dw.py run
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(ROOT), str(HERE)):
    if p not in sys.path:
        sys.path.insert(0, p)

import spacetime as S  # noqa: E402

STEP = 64000
TAG = 'r20s0b'
FACTOR_INDEX = 3            # t / t* = 1 in spacetime_stage.FACTORS
SCALE_INDEX = 0             # 'own'
K_SOLVE = 200               # modes computed: brackets every window below with room for a multiplicity block
TARGETS = (3, 6, 12, 24, 48, 96)
SHIP = 16                   # scaling functions shipped per level (pivot order = greedy residual-norm coverage)
NBR = 15                    # neighbour set for the localisation diagnostic (the walk's own k)
OUT = ROOT / 'data' / 'wavelets'


def window(r: int) -> range:
    """Candidate ranks for a target r: ceil(r 2^-1/4) .. floor(r 2^1/4), a quarter octave either side (disjoint for the
    doubling targets). Half-octave windows were tried first: log gaps shrink with rank, so every level snapped to its
    window's lower edge (ranks 4, 5, 9, 18, 34, 68), far from the targets."""
    return range(int(np.ceil(r * 2 ** -.25 - 1e-12)), int(np.floor(r * 2 ** .25 + 1e-12)) + 1)


def choose_levels(mu: np.ndarray, targets=TARGETS) -> list[dict]:
    """For each target rank, the rank k in its window with the largest log gap ln(mu_{k+1} / mu_k) (1-based k: mu[k-1] is
    the largest kept eigenvalue); cut-off mu_c = sqrt(mu_k mu_{k+1}) (geometric midpoint), heat time t = ln 2 / mu_c."""
    out = []
    for r in targets:
        ks = [k for k in window(r) if k + 1 <= len(mu)]
        assert ks and max(ks) + 1 < len(mu), f'target {r}: window beyond the computed spectrum'
        g = np.array([np.log(mu[k] / mu[k - 1]) for k in ks])
        k = ks[int(np.argmax(g))]
        mu_c = float(np.sqrt(mu[k - 1] * mu[k]))
        out.append(dict(target=r, window=[ks[0], ks[-1]], k=k, mu_last=float(mu[k - 1]), mu_next=float(mu[k]),
                        mu_c=mu_c, t=float(np.log(2) / mu_c), log_gap=float(g.max()),
                        rel_gap=float((mu[k] - mu[k - 1]) / mu[k - 1])))
    return out


def count_below(ms: np.ndarray, c: np.ndarray, a2: float, sigma: float) -> tuple[int, float]:
    """Number of eigenvalues of L_t (whole space) below sigma, by Sylvester's law of inertia on the block LDL^T of the
    block-tridiagonal L_t - sigma I. Returns (count, smallest |eigenvalue| of any Schur block: conditioning)."""
    nl, n, _ = ms.shape
    deg = np.r_[1.0, np.full(nl - 2, 2.0), 1.0]
    eye = np.eye(n)
    neg, tiny, prev_inv = 0, np.inf, None
    for l in range(nl):
        s = (eye - ms[l]) / c[l] + (deg[l] * a2 - sigma) * eye
        if prev_inv is not None:
            s = s - a2 * a2 * prev_inv
        s = (s + s.T) / 2
        w, v = np.linalg.eigh(s)
        neg += int((w < 0).sum())
        tiny = min(tiny, float(np.abs(w).min()))
        prev_inv = (v / w) @ v.T
    return neg, tiny


def solve(ms: np.ndarray, c: np.ndarray, t: float, k: int = K_SOLVE, log=print) -> dict:
    nl, n, _ = ms.shape
    vals, vecs = S.modes(ms, c, t, k=k, tol=1e-12)
    log('solved', k, 'modes; first', np.round(vals[:4], 6), 'last', vals[-1])
    resid = S.residual(ms, c, t, vals, vecs)
    smean = np.abs(vecs.mean(1)).max(0)                       # (k,) largest slice mean
    gram = np.einsum('lni,lnj->ij', vecs, vecs)
    levels = choose_levels(vals)
    temporal = S.temporal_eigs(nl, t)
    checks = []
    sig = [lv['mu_c'] for lv in levels] + [float(np.sqrt(vals[-2] * vals[-1]))]
    for s in sig:
        cnt, tiny = count_below(ms, c, t, s)
        nt = int((temporal < s).sum())
        checks.append(dict(sigma=s, ambient=cnt, temporal=nt, mean_zero=cnt - nt, computed=int((vals < s).sum()),
                           min_schur_abs=tiny))
        log('inertia at', round(s, 6), checks[-1])
    return dict(vals=vals, vecs=vecs.astype(np.float64), resid=resid, slice_mean=smean,
                orth_err=float(np.abs(gram - np.eye(k)).max()), levels=json.dumps(levels), inertia=json.dumps(checks),
                temporal=temporal, c=c, t=np.float64(t), nbr=knn_sets(ms))


# ---------------------------------------------------------------- local part

def shipped_compare(vecs: np.ndarray, vals: np.ndarray) -> dict:
    """Leading 16 modes against (a) the shipped viewer unit slices (data/<tag>/spacetime/st_<step>_0.npz, int16) and (b) the
    packaged raw joint modes (data/sitex/ops/r20s0-c10-knn15, int16 with per-(site, mode) scale)."""
    import gzip
    z = np.load(ROOT / 'data' / TAG / 'spacetime' / f'st_{STEP}_0.npz')
    sv = z['vecs'][SCALE_INDEX, FACTOR_INDEX].astype(np.float64)          # (K, L, N) int16 of unit slices
    K = sv.shape[0]
    ref_vals = z['vals'][SCALE_INDEX, FACTOR_INDEX][:K]
    g = vecs[:, :, :K]
    unit = S.unit_slices(g).transpose(2, 0, 1)                            # (K, L, N)
    mx = np.abs(unit).max(-1, keepdims=True)
    mine_q = np.round(32767 * unit / mx)
    w = S.depth_weight(g)                                                 # (K, L)
    cos = (unit * sv).sum(-1) / np.linalg.norm(sv, axis=-1).clip(1e-300)  # unit slices: cosine per (mode, site)
    signs = np.sign(np.einsum('kl,kl->k', cos, w))
    exact = (mine_q * signs[:, None, None] == sv).mean()
    maxq = np.abs(mine_q * signs[:, None, None] - sv).max()
    out = dict(K=K, val_maxrel=float(np.abs(vals[:K] / ref_vals - 1).max()), signs=signs.tolist(),
               min_cos_weighted=float((np.abs(cos) * w).sum(1).min()), min_abs_cos_heavy=float(np.abs(cos)[w > 0.01].min()),
               int16_equal_share=float(exact), int16_max_diff=float(maxq))
    op = ROOT / 'data' / 'sitex' / 'ops' / 'r20s0-c10-knn15'
    if (op / 'op.json').exists():
        meta = json.loads((op / 'op.json').read_text())
        kd = meta['kinds']['joint']
        raw = np.frombuffer(gzip.decompress((op / kd['file']).read_bytes()), '<i2').reshape(kd['shape']).astype(np.float64)
        raw = raw * np.asarray(kd['scale'])[:, None, :]                   # (L, N, K) raw values
        d = g * signs[None, None, :] - raw
        out['sitex_raw_max_abs_diff'] = float(np.abs(d).max())
        out['sitex_raw_max_abs'] = float(np.abs(raw).max())
        out['sitex_raw_max_rel_per_slice'] = float((np.abs(d).max(1) / np.abs(raw).max(1).clip(1e-300)).max())
        out['sitex_t_match'] = abs(meta['t'] - float(z['t'][SCALE_INDEX, FACTOR_INDEX])) < 1e-9
    return out


def knn_sets(ms: np.ndarray, q: int = NBR) -> np.ndarray:
    """(L*N, q) flat neighbour indices: each point's q heaviest walk neighbours at its own site (diagonal excluded)."""
    nl, n, _ = ms.shape
    out = np.empty((nl * n, q), np.int64)
    for l in range(nl):
        m = ms[l] - np.diag(np.diag(ms[l]))
        nb = np.argsort(-m, 1)[:, :q]
        out[l * n:(l + 1) * n] = nb + l * n
    return out


def stage(eig_path: Path, out_dir: Path = OUT) -> dict:
    import dw
    e = np.load(eig_path)
    vals, vecs = e['vals'], e['vecs']
    nl, n, k = vecs.shape
    levels, inertia = json.loads(str(e['levels'])), json.loads(str(e['inertia']))
    U = vecs.reshape(nl * n, k)                                            # flat index = site * N + image
    probe = np.load(ROOT / 'data' / TAG / 'probe.npz')
    sites = json.loads((ROOT / 'data' / TAG / 'meta.json').read_text())['sites']
    lam = np.exp(-vals)
    checks = dict(resid_max=float(e['resid'].max()), slice_mean_max=float(e['slice_mean'].max()),
                  orth_err=float(e['orth_err']), inertia=inertia, ship=shipped_compare(vecs, vals))
    checks['inertia_ok'] = all(r['mean_zero'] == r['computed'] for r in inertia)
    nbr = e['nbr']                                                          # localisation diagnostic only
    arrays, table = {}, []
    for j, lv in enumerate(levels):
        kk = lv['k']
        assert dw.cut_index(lam, float(np.exp(-lv['mu_c']))) == kk, 'cut-off splits a multiplicity block'
        wgt = np.exp(-lv['t'] * vals[:kk])
        Q, piv = dw.band_basis(U, lam, 0, kk, weights=wgt)
        Phi = U[:, :kk] @ Q                                                # (L*N, kk) unit columns
        ns = min(SHIP, kk)
        ph = Phi[:, :ns].reshape(nl, n, ns)
        assert np.all(Phi[piv[:ns], np.arange(ns)] > 0)
        n90, near_peak, near_seed = dw.localisation(Phi[:, :ns], nbr, piv[:ns])
        prof = (ph ** 2).sum(1).T                                          # (ns, L)
        peak = np.abs(Phi[:, :ns]).argmax(0)
        fns = []
        for i in range(ns):
            ps, pi = divmod(int(piv[i]), n)
            ks_, ki = divmod(int(peak[i]), n)
            fns.append(dict(id=f'w{kk}.{i + 1}', order=i + 1, pivot_site=ps, pivot_site_name=sites[ps], pivot_image=pi,
                            pivot_cifar=int(probe['idx'][pi]), pivot_label=int(probe['labels'][pi]),
                            pivot_value=float(Phi[piv[i], i]), peak_site=ks_, peak_image=ki,
                            peak_value=float(Phi[peak[i], i]),
                            profile=[float(x) for x in prof[i]], n90=int(n90[i]),
                            near_peak=None if np.isnan(near_peak[i]) else float(near_peak[i]),
                            near_seed=None if np.isnan(near_seed[i]) else float(near_seed[i])))
        # a complete band: Phi is an orthogonal rotation of the band's eigenmodes
        span_err = float(np.abs(Q.T @ Q - np.eye(kk)).max())
        proj_err = float(np.abs(Phi @ Phi.T[:, :50] - U[:, :kk] @ U[:50, :kk].T).max())
        eig_n90 = dw.localisation(U[:, :kk])[0]
        table.append(dict(lv, level=j, rank=kk, shipped=ns, weight_min=float(wgt.min()), weight_next=float(np.exp(-lv['t'] * vals[kk])),
                          q_orth_err=span_err, proj_err_sample=proj_err, n90_median=float(np.median(n90)),
                          eig_n90_median=float(np.median(eig_n90)), functions=fns))
        arrays[f'phi{j}'] = ph                                             # (L, N, ns) float64 raw values
        arrays[f'piv{j}'] = piv[:ns]
    rec = dict(step=STEP, tag=TAG, operator=dict(scale='own', factor=1.0, t_star=float(e['t']), c=[float(x) for x in e['c']],
                                                  rule=0, kernel='union 15-NN Sinkhorn walk', n=n, sites=sites, k_solved=k),
               targets=list(TARGETS), ship=SHIP, vals=[float(x) for x in vals], checks=checks, levels=table)
    bad = accept(rec)
    rec['accepted'], rec['rejected_because'] = not bad, bad
    if bad:            # nothing the build reads is touched: the last accepted pair (if any) stays; the rejection is reported
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / f'joint_dw_{STEP}.rejected.json').write_text(json.dumps(rec, indent=1))
        return rec
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f'joint_dw_{STEP}.npz', vals=vals, probe_idx=probe['idx'], c=e['c'], t=e['t'], **arrays)
    (out_dir / f'joint_dw_{STEP}.json').write_text(json.dumps(rec, indent=1))
    (out_dir / f'joint_dw_{STEP}.rejected.json').unlink(missing_ok=True)
    return rec


def accept(rec: dict) -> list[str]:
    c = rec['checks']
    bad = []
    if c['resid_max'] > 1e-8:
        bad.append(f"residual {c['resid_max']:.2e}")
    if c['slice_mean_max'] > 1e-10:
        bad.append(f"slice mean {c['slice_mean_max']:.2e}")
    if c['orth_err'] > 1e-9:
        bad.append(f"orthonormality {c['orth_err']:.2e}")
    if not c['inertia_ok']:
        bad.append('inertia count differs from the computed spectrum (incomplete band)')
    s = c['ship']
    if s['val_maxrel'] > 1e-8:
        bad.append(f"eigenvalues differ from shipped {s['val_maxrel']:.2e}")
    if s['min_abs_cos_heavy'] < 0.9999:
        bad.append(f"modes differ from shipped (min |cos| {s['min_abs_cos_heavy']:.6f})")
    for lv in rec['levels']:
        if lv['q_orth_err'] > 1e-10 or lv['proj_err_sample'] > 1e-10:
            bad.append(f"level {lv['rank']}: basis is not a rotation of its band")
    return bad


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'stage'
    if cmd == 'stage':
        rec = stage(OUT / f'joint_eig_{STEP}.npz')
        c = rec['checks']
        print(json.dumps({k: v for k, v in c.items() if k != 'inertia'}, indent=1))
        for r in c['inertia']:
            print('inertia', r)
        for lv in rec['levels']:
            print(f"rank {lv['rank']:4d} (target {lv['target']}, window {lv['window']}) mu_c {lv['mu_c']:.5f} t {lv['t']:.4f} "
                  f"log gap {lv['log_gap']:.4f} shipped {lv['shipped']} n90 median {lv['n90_median']:.0f} (eig {lv['eig_n90_median']:.0f})")
        bad = rec['rejected_because']
        print('ACCEPT' if not bad else f'FAILED (accepted files left unchanged; see joint_dw_{STEP}.rejected.json): {bad}')
        sys.exit(1 if bad else 0)
