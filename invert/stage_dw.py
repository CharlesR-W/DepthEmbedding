"""Diffusion-wavelet inversion stage (wavelets/SPEC_DW.md, questions Q1-Q3, revision 3): one site per unit.

Operators and dictionaries (`wavelets/dw.py` conventions; eigenpairs T U = U diag(lam), U orthonormal, lam descending):
  site    the `pr400` Sinkhorn walk P_l of this site: positive spectral sector above the floor, constant removed.
  second  T = K* / sigma_1, the Gaussian Gram of ddm.LatentHarmonics on the 64 input coordinates of a cell
          (`joint|64`, `source|64`) at the width the stored DDM GCV `lh` picked (frozen in absolute units).
Partition: V_i = V(1/A_i) for the frozen list A_LIST plus the floor lam >= FLOOR (relative to the top eigenvalue);
W_i = V_i minus V_{i-1}.  Cuts are resolved on the whole (unsliced) spectrum with a multiplicity tolerance relative to
the eigenvalue at the edge; where a block straddles the floor the whole block is kept (the block wins over the hard
floor).  Every band's pivoted-QR basis filters by one application of T (diag(lam_B)).  Fixed ranks (FIXED) are exact,
unsnapped evaluation points only.  Out-of-sample: z_Phi = (p U / lam) Q.

Q1  decoders from the 64 coordinates: lead (leading n harmonics), ebest / wbest / rbest (best n eigenvectors /
    QR wavelets / within-band random rotations, by coefficient energy), each unconstrained and inside the admissible
    bands A <= 1e2, 1e4; ridge at the same width; the stored tuned krr; the stored DDM GCV lift (`lead@gcv`, through
    the stored rank on the whole spectrum, not clipped at the floor).  n (and the ridge) chosen on an inner 80/20 split
    that rebuilds kernel, bands, bases, rankings and coefficients on the 80% (targets centred on the 80% mean); only
    the budget transfers.  Every prediction goes through `Lift`, whose methods take latent points only.
Q2  band(A): all of V(1/A) of the site operator decoded linearly from the site's own Nystrom coordinates.
Q3  compression: in-sample energy of the first / best n eigenvectors, best n wavelets, best n random-rotation atoms;
    held-out decode from those n features (a new encoding, computed from the true activation's kernel row).
Also: spectrum tables, error amplification (coordinate -> activation, and kernel row -> activation for Q2), the
Q2 round trip, localisation of every basis, in-run controls with tolerances (`failed` lists any that miss), timing
and peak memory per part.

Precision: centring follows stage_ddm (float32 subtraction of the stored float32 mean, then float64) so that the
stored scores are reproduced; patching runs the float32 network.

Outputs: dw/<tag>/<site>.json (summaries, curves, tables; strict JSON, nulls for undefined values) and <site>.npz
(per-image scores of finals and of every curve point, per-atom localisation).
"""
from __future__ import annotations

import json
import math
import resource
import time
from pathlib import Path

import numpy as np
import torch

import ddm
import dw
import stage_decode as sd

PROTOCOL = 'dw-inv-3'
DT = torch.float64
FLOOR = 1e-10                                     # numeric positive floor, relative to the top eigenvalue
A_LIST = tuple(sorted({1.5, 1e2, 1e4} | {2.0 ** j for j in range(1, 34)}))   # allowed amplifications, frozen
ADM = (None, 1e2, 1e4)                            # admissible bands for the *best decoders (None: down to the floor)
FIXED = (4, 16, 64, 256, 1024)                    # exact evaluation ranks only; never part of the partition
FRACS = (0.01, 0.05, 0.2)                         # perturbation sizes, share of the probe RMS cloud radius
CELLS = ('joint|64', 'source|64')
NNB = 400                                         # neighbourhood size for localisation
NSEEDS = 5                                        # random rotations per band
KINDS = ('lead', 'ebest', 'wbest', 'rbest')
BASIS = dict(lead='eig', first='eig', ebest='eig', wbest='qr', rbest='rand0')
RIDGES = tuple(10.0 ** j for j in range(-8, 3))   # as stage_ddm
TOL = dict(patch_true_kl=1e-9, onestep_max_abs_diff=1e-6, q2_qr_vs_eig_rel=1e-8, q2_rand_vs_eig_rel=1e-8,
           q2_fd_vs_exact_rel=1e-8, lin_vec_rel=1e-5, lin_cos_vs_stored=1e-4, band_cos_vs_stored=1e-4, lead_gcv_vs_weights=1e-8,
           fullband_vs_weights=1e-8, lead_gcv_cos_vs_stored=1e-4, krr_cos_vs_stored=1e-4, complete_band_rel=1e-8,
           s0_rel=1e-9)


# ------------------------------------------------------------------ dictionaries

def partition(lam: np.ndarray) -> list[dict]:
    """Bands V_i = V(1/A_i), A_i in A_LIST, plus the floor, on the whole spectrum lam (descending, scaled so the
    operator's top eigenvalue is 1; may contain zero and negative values).  Empty and repeated bands dropped (a
    repeated rank keeps its smallest A).  The last level is the floor; its rank r is the dictionary size."""
    lv = dw.levels(lam, [1 / A for A in A_LIST] + [FLOOR], edge=True)
    for d in lv:
        d['A'] = 1 / d['cut']
    return lv


def rank_at(lam: np.ndarray, A: float | None, r: int) -> int:
    """Rank of V(1/A) (whole multiplicity blocks), at most r; r for A = None."""
    return r if A is None else min(r, dw.cut_index(lam, 1 / A, edge=True))


class BandRot:
    """Block rotation of a dw.Multiscale on the GPU.  kind 'qr' (pivoted-QR localisation) or 'rand' (Haar per band)."""

    def __init__(self, U_np, lam, ks, kind, seed=0, device='cpu'):
        ms = dw.Multiscale(U_np if U_np is not None else np.empty((1, ks[-1])), lam, ks, kind, seed)
        self.bounds, self.pivot, self.band = ms.bounds, ms.pivot, ms.band
        self.Q = [torch.as_tensor(q, dtype=DT, device=device) for q in ms.Q]

    def rot(self, A):
        """Eigen-coordinates (m, >= r) -> atom coordinates (m, r)."""
        return torch.cat([A[:, s:e] @ q for (s, e), q in zip(self.bounds, self.Q)], 1)

    def rotT(self, B):
        """Eigen-coefficients (r, D) -> atom coefficients (r, D) (Phi^T X = Q^T U^T X, band by band)."""
        return torch.cat([q.T @ B[s:e] for (s, e), q in zip(self.bounds, self.Q)], 0)


class Lift:
    """Second-stage dictionary on latent coordinates (spec Q1), fit on probe values Xp only.

    Every prediction method takes latent query points Zq and nothing else about the query: a decoder never sees a
    query activation (spec control 7)."""

    def __init__(self, Zp, Xp, h, qr=True, rand_seeds=(0,), coef_seeds=(0,)):
        t0 = time.time()
        self.lh = lh = ddm.LatentHarmonics(Zp, 'gauss', h=h)
        self.t_eig = time.time() - t0
        self.lam_full = (lh.w / lh.w[0]).cpu().numpy()
        self.levels = partition(self.lam_full)
        self.ks = [d['k'] for d in self.levels]
        self.r = r = self.ks[-1]
        self.lam = self.lam_full[:r]
        self.Y = lh.U.T @ Xp                                                    # all n eigen-coefficients
        dev = Zp.device
        t0 = time.time()
        self.rot = {'eig': None}
        if qr:
            self.rot['qr'] = BandRot(lh.U[:, :r].cpu().numpy(), self.lam, self.ks, 'qr', device=dev)
        self.t_qr = time.time() - t0
        for s in rand_seeds:
            self.rot[f'rand{s}'] = BandRot(None, self.lam, self.ks, 'rand', s, dev)
        self.B, self.E2 = {}, {}
        for k, R in self.rot.items():
            if k.startswith('rand') and int(k[4:]) not in coef_seeds:
                continue
            self.B[k] = self.Y[:r] if R is None else R.rotT(self.Y[:r])
            self.E2[k] = (self.B[k] ** 2).sum(1)

    def kA(self, adm):
        return rank_at(self.lam_full, adm, self.r)

    def order(self, kind, adm=None):
        if kind in ('lead', 'first'):
            return torch.arange(self.r, device=self.Y.device)
        return torch.argsort(-self.E2[BASIS[kind]][:self.kA(adm)], stable=True)

    def n_eff(self, kind, n, adm=None):
        return min(int(n), len(self.order(kind, adm)))

    def _zU(self, Zq):
        return self.lh.proj(Zq)[:, :self.r] / self.lh.w[:self.r]

    def _atoms(self, zU, basis):
        R = self.rot[basis]
        return zU if R is None else R.rot(zU)

    def decode(self, Zq, kind, n, adm=None):
        """Centred activation decoded from latent points Zq by the best (or leading) n atoms."""
        b = BASIS[kind]
        o = self.order(kind, adm)[:n]
        return self._atoms(self._zU(Zq), b)[:, o] @ self.B[b][o]

    def curve(self, Zq, kind, adm, grid):
        """Yield (n, decode with the first n atoms of the ranking) for ascending n in grid (built incrementally)."""
        b = BASIS[kind]
        yield from cum_decode(self._atoms(self._zU(Zq), b), self.B[b], self.order(kind, adm), grid)

    def band_prefix(self, Zq, basis, ks):
        """Yield (k, decode by all atoms of the complete bands below rank k) in the given basis."""
        yield from cum_decode(self._atoms(self._zU(Zq), basis), self.B[basis], torch.arange(self.r, device=self.Y.device), ks)

    def lead_raw(self, Zq, m):
        """The stored DDM lift: leading m harmonics of the whole spectrum (LatentHarmonics.weights), not clipped
        at this dictionary's floor."""
        m = min(int(m), self.lh.m_max)
        return (self.lh.proj(Zq)[:, :m] / self.lh.w[:m]) @ self.Y[:m]

    def ridge_curve(self, Zq, rhos):
        """Yield (rho, kernel ridge with the same kernel, k*(z) (K* + rho I)^-1 X), one projection for all rho."""
        C = self.lh.proj(Zq)
        w = self.lh.w.clamp_min(0)
        for rho in rhos:
            yield rho, (C / (w + rho)) @ self.Y

    def ridge(self, Zq, rho):
        return next(self.ridge_curve(Zq, [rho]))[1]

    def ridge_df(self, rho):
        w = self.lh.w.clamp_min(0)
        return float((w / (w + rho)).sum())


def cum_decode(A, B, order, grid):
    """Yield (n, Xh) for ascending n in grid: Xh = A[:, order[:n]] @ B[order[:n]], built incrementally (n capped at
    len(order)).  Xh is updated in place: use it before advancing."""
    Xh = A.new_zeros(len(A), B.shape[1])
    prev = 0
    for n in grid:
        o = order[prev:n]
        if len(o):
            Xh += A[:, o] @ B[o]
        prev = max(prev, min(n, len(order)))
        yield n, Xh


def geom_grid(r, npts, extra=()):
    g = set(np.unique(np.round(np.geomspace(1, r, npts)).astype(int)).tolist()) | {int(e) for e in extra if 0 < e <= r}
    return sorted(g | {r})


def loc_t(Phi, nbr, piv=None, share=0.9, chunk=1024):
    """Torch port of dw.localisation: n90, near_peak (own absolute peak + its nbr), near_seed (QR seed; nan if none)."""
    out = [[], [], []]
    for c in range(0, Phi.shape[1], chunk):
        e = Phi[:, c:c + chunk] ** 2
        e = e / e.sum(0, keepdim=True)
        srt = e.sort(0, descending=True).values
        out[0].append((srt.cumsum(0) < share).sum(0) + 1)
        peak = e.argmax(0)
        idx = torch.cat([peak[:, None], nbr[peak]], 1)
        out[1].append(e.T.gather(1, idx).sum(1))
        if piv is None:
            out[2].append(torch.full((e.shape[1],), float('nan'), dtype=DT, device=e.device))
        else:
            p = torch.as_tensor(piv[c:c + chunk], device=e.device)
            ok = p >= 0
            idx = torch.cat([p.clamp_min(0)[:, None], nbr[p.clamp_min(0)]], 1)
            v = e.T.gather(1, idx).sum(1)
            out[2].append(torch.where(ok, v, torch.full_like(v, float('nan'))))
    return [torch.cat(o) for o in out]


def top_sv(L, ks, nsv=5):
    """Leading singular values of L[:k] (rows of Lam^-1 B) for ascending k in ks, on the smaller side: the row Gram
    built once and sliced when k <= D, an incrementally accumulated D x D Gram when k > D.  At most min(nsv, k, D)."""
    r, D = L.shape
    kmax = max(ks)
    G = L[:kmax] @ L[:kmax].T if kmax <= D else None
    GD, prev = None, 0
    for k in ks:
        if k <= D:
            Gk = G[:k, :k] if G is not None else L[:k] @ L[:k].T
        else:
            GD = L[:k].T @ L[:k] if GD is None else GD + L[prev:k].T @ L[prev:k]
            prev = k
            Gk = GD
        yield k, torch.linalg.eigvalsh((Gk + Gk.T) / 2).clamp_min(0).sqrt().flip(0)[:min(nsv, k, D)]


def upd(cur, v):
    """Running maximum that keeps a non-finite value once seen."""
    if not math.isfinite(cur):
        return cur
    return v if not math.isfinite(v) else max(cur, v)


def stat(a):
    a = a.double()
    return [float(a.mean()), float(a.median()), float(a.std() / max(1, len(a)) ** .5)]


def paired(a, b):
    d = (a.double() - b.double())
    return [float(d.mean()), float(d.std() / len(d) ** .5)]


def clean(o):
    """Strict JSON: non-finite floats become null."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (float, np.floating)):
        return float(o) if math.isfinite(o) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


class Parts:
    """Wall-clock and peak GPU memory of each part (CUDA synchronised at the boundaries), and the process's
    cumulative peak resident memory at the end of each part."""

    def __init__(self):
        self.log, self.t, self.name = {}, None, None

    def _sync(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self, name):
        self.stop()
        self._sync()
        self.name, self.t = name, time.time()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def stop(self):
        if self.name is None:
            return
        self._sync()
        self.log[self.name] = dict(seconds=time.time() - self.t,
                                   gpu_peak_gb=torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else 0.0,
                                   rss_peak_cum_gb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20)
        print('part', self.name, {k: round(v, 2) for k, v in self.log[self.name].items()}, flush=True)
        self.name = None


def spectrum_table(lam_full, levels, energy, total, fixed=FIXED):
    """Rows per partition level and per fixed (exact, unsnapped) evaluation rank: rank, A, cut, smallest kept
    eigenvalue, largest reciprocal, relative gap to the first excluded eigenvalue, share of |X_c|^2 inside the band.
    lam_full: the whole spectrum; energy (r,) = per-eigenvector energy inside the dictionary."""
    cum = np.concatenate([[0.0], np.cumsum(energy)])
    rows = [dict(k=d['k'], A=d['A'], cut=d['cut'], lam_min=d['lam_min'], amp=d['amp'], gap=d['gap'],
                 lam_next=float(lam_full[d['k']]) if d['k'] < len(lam_full) else None,
                 share=float(cum[d['k']] / total)) for d in levels]
    fx = [dict(k=K, exact=True, lam_min=float(lam_full[K - 1]), amp=float(1 / lam_full[K - 1]),
               gap=float((lam_full[K - 1] - lam_full[K]) / lam_full[0]) if K < len(lam_full) else None,
               share=float(cum[K] / total)) for K in fixed if K <= len(energy)]
    return dict(levels=rows, fixed=fx)


def cached(fout, fnpz, parts, cells):
    if not (fout.exists() and fnpz.exists()):
        return False
    try:
        o = json.loads(fout.read_text())
    except ValueError:
        return False
    return (o.get('protocol') == PROTOCOL and o.get('failed') == [] and set(parts) <= set(o.get('parts_run', []))
            and set(cells) <= set(o.get('cells_run', [])))


# ------------------------------------------------------------------ the unit

def run(unit: dict, cfg: dict, mi):
    t_all = time.time()
    R = Path(mi.R)
    dev = cfg.get('device', 'cuda')
    torch.backends.cuda.matmul.allow_tf32 = False
    pset, kname, site = unit['pset'], unit['kname'], unit['site']
    tag = f'{pset}__{kname}__L12'
    out = R / 'dw' / tag
    out.mkdir(parents=True, exist_ok=True)
    sfx = cfg.get('suffix', '')
    fout, fnpz = out / f'{site}{sfx}.json', out / f'{site}{sfx}.npz'
    parts = sorted(cfg.get('parts', ['q1', 'q2', 'q3', 'loc']))
    cells = list(cfg.get('cells', list(CELLS))) if 'q1' in parts else []
    if not cfg.get('force') and cached(fout, fnpz, parts, cells):
        return 'cached'
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=DT, device=dev)
    pt = Parts()
    pt.start('load')

    sp = dict(np.load(R / 'splits.npz'))
    idx, hid = mi.index_set(sp, pset), mi.index_set(sp, 'H')
    z = np.load(R / 'joint' / tag / 'modes.npz')
    sites = [str(s) for s in z['sites']]
    li = sites.index(site)
    F = T(z['vecs'][li][:, :64])
    cz = np.load(R / 'joint' / tag / 'coords_H.npz')
    assert abs(float(cz['basis']) - float(np.abs(z['vals']).sum())) < 1e-12, 'stale coordinates'
    FH, SH_stored = T(cz['y'][li][:, :64]), T(cz['onestep'][li][:, :64])
    n, m = len(idx), len(hid)

    g = R / 'gram' / mi.centre_set(pset)
    Call = np.load(g / f'{site}.npy', mmap_mode='r')
    Ctt, CsH = T(Call[idx][:, idx]), T(Call[hid][:, idx])
    nz = np.load(g / f'{site}_norms.npz')
    mu = torch.tensor(nz['mean'], device=dev)
    acts = np.load(R / 'acts/c10' / f'{site}.npy', mmap_mode='r')
    Xc = (torch.tensor(np.asarray(acts[idx]), device=dev) - mu).double()       # as stage_ddm (float32 subtraction)
    XH = (torch.tensor(np.asarray(acts[hid]), device=dev) - mu).double()
    Xn = Xc / Xc.norm(dim=1, keepdim=True)
    nH = XH.norm(dim=1)
    D = Xc.shape[1]
    sX = float(((Xc - Xc.mean(0)) ** 2).sum(1).mean().sqrt())                    # probe RMS cloud radius of X
    E_tot = float((Xc ** 2).sum())
    res = dict(unit=unit, protocol=PROTOCOL, parts_run=parts, cells_run=cells, n=n, m=m, D=D, A_list=list(A_LIST),
               floor=FLOOR, adm=list(ADM), fracs=list(FRACS), fixed=list(FIXED), sX=sX, tol=TOL, checks={}, spectrum={},
               q1={}, parts={})
    arrays = {}

    kerd = dict(np.load(R / 'kernel' / f'{pset}__{kname}' / f'{site}.npz'))
    P = T(kerd['P'])
    kt = ddm.ker_to_torch(kerd, dev)
    del kerd

    def fwd(X):
        """The site's forward map: centred activations -> first-stage out-of-sample kernel rows (from vectors)."""
        return ddm.pr_oos_rows(1 - ddm.cosines(X, Xn), kt)

    rowsH = fwd(XH)
    rowsH_stored = ddm.pr_oos_rows(1 - CsH, kt)                                  # as stage_ddm (stored cosines)
    res['checks']['rows_vec_vs_stored_cos'] = float((rowsH - rowsH_stored).abs().max())
    SH = rowsH_stored @ F
    res['checks']['onestep_max_abs_diff'] = float((SH - SH_stored).abs().max())
    near = rowsH.max(1).values > 0.5                                             # near-copy of a probe image, this site
    near_any = torch.as_tensor(np.asarray(cz['rowmax']) > 0.5, device=dev).any(0) if 'rowmax' in cz.files else near
    res['near_probe'], res['near_probe_any_site'] = int(near.sum()), int(near_any.sum())
    arrays['near_probe'], arrays['near_probe_any'] = near.cpu().numpy(), near_any.cpu().numpy()
    far, far_any = ~near, ~near_any

    lg_all = np.load(R / 'acts/c10/logits.npy')
    logp = torch.log_softmax(torch.tensor(lg_all[hid], device=dev).double(), 1)
    model, _ = mi.load_model()
    shape = mi.SHAPES[site]

    def patched(Xh):
        xh = mu + Xh.float()
        with torch.inference_mode():
            lp = torch.cat([mi.suffix(model, site, xh[a:a + 500].reshape(-1, *shape)) for a in range(0, len(xh), 500)])
        lp = torch.log_softmax(lp.double(), 1)
        return (lp.argmax(1) == logp.argmax(1)).double(), (logp.exp() * (logp - lp)).sum(1)

    ag, kl = patched(XH)
    res['checks']['patch_true_agree'], res['checks']['patch_true_kl'] = float(ag.mean()), float(kl.abs().max())

    ddm_dir = R / 'ddm' / tag
    ddm_json = json.loads((ddm_dir / f'{site}.json').read_text())
    ddm_npz = np.load(ddm_dir / f'{site}.npz')

    def amp_stats(a):
        return dict(med=float(a.median()), p95=float(a.quantile(.95)), med_far=float(a[far].median()),
                    p95_far=float(a[far].quantile(.95)), med_far_any=float(a[far_any].median()),
                    p95_far_any=float(a[far_any].quantile(.95)))

    def score(Xh, dX=None, key=None):
        """Reconstruction and function of decodes Xh (m, D); dX: {f: decode at z + delta_f minus Xh}."""
        cos, cap = ddm.vec_scores(Xh, XH)
        agr, kld = patched(Xh)
        e = dict(cos=stat(cos), cap=stat(cap), norm=stat(Xh.norm(dim=1) / nH), agree=stat(agr), kl=stat(kld),
                 fp=[float(Xh.sum()), float((Xh ** 2).sum())])                  # fingerprint of the decode itself
        per = dict(cos=cos, cap=cap, agree=agr, kl=kld)
        if dX is not None:
            e['amp'] = {}
            for f, d in dX.items():
                a = d.norm(dim=1) / (f * sX)
                per[f'amp{f}'] = a
                e['amp'][str(f)] = amp_stats(a)
        if key is not None:
            for k_, v in per.items():
                arrays[f'{key}|{k_}'] = (v.bool() if k_ == 'agree' else v.float()).cpu().numpy()
        return e, per

    def curve_rec(store, key):
        """Per-image scores of a curve's points: arrays[key|cos|kl|agree] (npts, m)."""
        for k_ in ('cos', 'kl', 'agree'):
            arrays[f'{key}|{k_}'] = np.stack(store[k_]) if store[k_] else np.zeros((0, m))

    def curve_point(Xh, dX=None, store=None):
        cos, cap = ddm.vec_scores(Xh, XH)
        agr, kld = patched(Xh)
        o = dict(cos=float(cos.mean()), cos_sem=float(cos.std() / m ** .5), cap_med=float(cap.median()),
                 norm_med=float((Xh.norm(dim=1) / nH).median()), agree=float(agr.mean()), agree_sem=float(agr.std() / m ** .5),
                 kl_mean=float(kld.mean()), kl_med=float(kld.median()))
        if dX is not None:
            for f, d in dX.items():
                a = d.norm(dim=1) / (f * sX)
                o[f'amp{f}_med'], o[f'amp{f}_p95'] = float(a.median()), float(a.quantile(.95))
        if store is not None:
            store['cos'].append(cos.float().cpu().numpy())
            store['kl'].append(kld.float().cpu().numpy())
            store['agree'].append(agr.bool().cpu().numpy())
        return o

    new_store = lambda: dict(cos=[], kl=[], agree=[])

    # ================================================================== site operator (Q2, Q3)
    S = {}
    if set(parts) & {'q2', 'q3', 'loc'}:
        pt.start('site_eig')
        wP, V = torch.linalg.eigh((P + P.T) / 2)
        wP, V = wP.flip(0), V.flip(1)
        res['checks']['site_top_eig'] = float(wP[0])
        res['checks']['site_top_vec_const'] = float((V[:, 0].abs() * math.sqrt(n) - 1).abs().max())
        assert abs(float(wP[0]) - 1) < 1e-6
        rest = wP[1:]
        lam_full = (rest / wP[0]).cpu().numpy()
        levels = partition(lam_full)
        ks = [d['k'] for d in levels]
        r = ks[-1]
        lam = lam_full[:r]
        lam_t = rest[:r]
        U = V[:, 1:1 + r]
        Beig = U.T @ Xc
        e_eig = (Beig ** 2).sum(1)
        res['site'] = dict(r=r, n_neg=int((rest < 0).sum()), n_below_floor=int((rest < FLOOR).sum()) - int((rest < 0).sum()),
                           lam1=float(lam[0]), complement_share=1 - float(e_eig.sum()) / E_tot,
                           const_share=float((Xc.sum(0) ** 2).sum() / n) / E_tot)
        res['spectrum']['site'] = spectrum_table(lam_full, levels, e_eig.cpu().numpy(), E_tot)
        del wP, V
        pt.start('site_qr')
        t0 = time.time()
        rots = dict(qr=BandRot(U.cpu().numpy(), lam, ks, 'qr', device=dev))
        res['site']['qr_seconds'] = time.time() - t0
        res['site']['max_band'] = max(e - s for s, e in rots['qr'].bounds)
        for s in range(NSEEDS):
            rots[f'rand{s}'] = BandRot(None, lam, ks, 'rand', s, dev)
        Bat, E2 = {'eig': Beig}, {'eig': e_eig}
        for k_, Rr in rots.items():
            b = Rr.rotT(Beig)
            E2[k_] = (b ** 2).sum(1)
            if k_ in ('qr', 'rand0'):
                Bat[k_] = b
            del b
        zH = rowsH @ U / lam_t                                                   # Nystrom coordinates (m, r)
        zat = {'eig': zH, 'qr': rots['qr'].rot(zH), 'rand0': rots['rand0'].rot(zH)}
        S = dict(levels=levels, ks=ks, r=r, lam=lam, lam_full=lam_full, lam_t=lam_t, U=U, Beig=Beig, e_eig=e_eig,
                 rots=rots, Bat=Bat, E2=E2, zH=zH, zat=zat)
        del U, Beig, rots, Bat, E2, zH, zat

    # ------------------------------------------------------------------ Q2: band(A), all of V(1/A)
    def q2_part():
        levels, ks, r, lam, lam_t, U, Beig, zH, zat, Bat = (S[k] for k in ('levels', 'ks', 'r', 'lam', 'lam_t', 'U', 'Beig',
                                                                          'zH', 'zat', 'Bat'))
        q2 = dict(levels=[])
        evalk = sorted(set(ks) | {K for K in FIXED if K <= r})
        g2 = torch.Generator().manual_seed(2)
        a_, b_ = torch.randint(n, (m,), generator=g2), torch.randint(n - 1, (m,), generator=g2)
        b_ = b_ + (b_ >= a_)                                                     # a != b
        dU = U[a_.to(dev)] - U[b_.to(dev)]                                       # probe-coordinate differences (m, r)
        sZ2 = ((U - U.mean(0)) ** 2).sum(0).cumsum(0) / n                        # sZ(k)^2: probe mean |z[:k] - mean|^2
        d2 = (dU ** 2).cumsum(1)                                                 # |delta[:k]|^2 per image
        s_p = float(((P - P.mean(0)) ** 2).sum(1).mean().sqrt())                 # probe RMS radius of kernel rows
        svs = top_sv(Beig / lam_t[:, None], evalk)                              # rows of Lam^-1 B
        q2['n_zero_delta'] = int((d2[:, 0] == 0).sum())                          # zero-length directions at k = 1
        perm = torch.randperm(m, generator=torch.Generator().manual_seed(1)).to(dev)
        lin_ref = {}
        cumv = (zH ** 2).sum(0)
        zvar = ((zH - zH.mean(0)) ** 2).sum(0)
        ar = torch.arange(r, device=dev)
        gq = cum_decode(zH, Beig, ar, evalk)
        gw = cum_decode(zat['qr'], Bat['qr'], ar, evalk)
        gr = cum_decode(zat['rand0'], Bat['rand0'], ar, evalk)
        gd = cum_decode(dU, Beig, ar, evalk)
        eq_qr = eq_rand = fd_err = 0.0
        for (k, Xh), (_, Xw), (_, Xr), (_, dX), (_, sv) in zip(gq, gw, gr, gd, svs):
            sc = float(Xh.abs().max())
            nr = Xh.norm(dim=1).clamp_min(1e-300)
            if k in ks:                                                          # complete bands only, per image
                eq_qr = upd(eq_qr, float(((Xw - Xh).norm(dim=1) / nr).max()))
                eq_rand = upd(eq_rand, float(((Xr - Xh).norm(dim=1) / nr).max()))
            e, per = score(Xh, key=f'q2|{k}')
            # coordinate -> activation: delta = f sZ(k) dU[:, :k] / |dU[:, :k]|; linear, so the ratio is f-free
            resp = dX.norm(dim=1) / d2[:, k - 1].sqrt() * sZ2[k - 1].sqrt() / sX
            e['amp_coord'] = amp_stats(resp)
            if k in FIXED:                                                       # finite difference against exact delta B
                for f in FRACS:
                    dl = f * sZ2[k - 1].sqrt() * dU[:, :k] / d2[:, k - 1:k].sqrt()
                    ex = dl @ Beig[:k]
                    fd = (zH[:, :k] + dl) @ Beig[:k] - Xh
                    fd_err = upd(fd_err, float((fd - ex).abs().max() / ex.abs().max()))
            # kernel row -> activation: leading singular values of p -> p U_k Lam_k^-1 B_k (top_sv, smaller side)
            e['amp_row'] = dict(sv=sv.tolist(), sv_max_scaled=float(sv[0]) * s_p / sX)
            # round trip: band coordinates of the decode against the requested ones, same forward map both sides
            zr = fwd(Xh) @ U[:, :k]
            tv = float(zvar[:k].sum())
            miss = ((zr / lam_t[:k] - zH[:, :k]) ** 2).sum(1)
            e['rt'] = dict(total=float(miss.sum()) / tv, med=float(miss.median()) / (tv / m),
                           t1_total=float(((zr - zH[:, :k]) ** 2).sum()) / tv,
                           ref_zero=float(cumv[:k].sum()) / tv, ref_other=float(((zH[perm, :k] - zH[:, :k]) ** 2).sum()) / tv)
            arrays[f'q2|{k}|rt'] = miss.float().cpu().numpy()
            e.update(k=k, lam_min=float(lam[k - 1]), amp=float(1 / lam[k - 1]), share=float(S['e_eig'][:k].sum()) / E_tot,
                     in_partition=k in ks, fixed=k in FIXED, A=next((d['A'] for d in levels if d['k'] == k), None))
            q2['levels'].append(e)
            if k in (4, 16, 64, 256) and f'site|{k}|linear|cos' in ddm_npz.files:   # span invariance vs the linear decode
                Phi_u = U[:, :k] * math.sqrt(n)
                Xl = sd.w_linear(Phi_u, rowsH_stored @ Phi_u / lam_t[:k]) @ Xc
                st = torch.as_tensor(ddm_npz[f'site|{k}|linear|cos'], device=dev).double()
                cl, _ = ddm.vec_scores(Xl, XH)
                lin_ref[k] = dict(vec_rel=float(((Xl - Xh).norm(dim=1) / nr).max()), cos_vs_stored=float((cl - st).abs().max()),
                                  band_cos_vs_stored=float((per['cos'] - st).abs().max()),
                                  stored_mean=ddm_json['combos'][f'site|{k}']['finals']['linear']['cos'][0])
        res['checks']['q2_lin_expected'] = [K for K in (4, 16, 64, 256) if K <= r]
        res['checks'].update(q2_qr_vs_eig_rel=eq_qr, q2_rand_vs_eig_rel=eq_rand, q2_fd_vs_exact_rel=fd_err, q2_vs_linear=lin_ref)
        res['q2'] = q2

    # ------------------------------------------------------------------ Q3: compression
    def q3_part():
        r, lam_full, e_eig, E2, Bat, zat = (S[k] for k in ('r', 'lam_full', 'e_eig', 'E2', 'Bat', 'zat'))
        q3 = dict(energy={}, decode={})
        eg = geom_grid(r, 80, list(S['ks']) + list(FIXED))
        egi = torch.as_tensor(eg, device=dev) - 1
        q3['energy']['first'] = dict(n=eg, share=(e_eig.cumsum(0)[egi] / E_tot).tolist())
        for k_ in E2:
            for adm in ADM:
                kA = rank_at(lam_full, adm, r)
                c = E2[k_][:kA].sort(descending=True).values.cumsum(0) / E_tot
                gg = [x for x in eg if x <= kA] + ([kA] if kA not in eg else [])
                q3['energy'][f'{k_}|{adm}'] = dict(n=gg, share=c[torch.as_tensor(gg, device=dev) - 1].tolist())
        dg = geom_grid(r, 25, FIXED)
        combos = [('first', None)] + [(k_, adm) for k_ in ('ebest', 'wbest', 'rbest') for adm in ADM]
        for kind, adm in combos:
            b = BASIS[kind]
            kA = rank_at(lam_full, adm, r)
            o = torch.arange(r, device=dev) if kind == 'first' else torch.argsort(-E2[b][:kA], stable=True)
            grid = sorted({x for x in dg if x <= kA} | {kA})
            st = new_store()
            pts = []
            for nn, Xh in cum_decode(zat[b], Bat[b], o, grid):
                pts.append(dict(n=nn, **curve_point(Xh, store=st)))
            q3['decode'][f'{kind}|{adm}'] = pts
            curve_rec(st, f'q3|{kind}|{adm}')
        res['q3'] = q3

    def localise(Ub, rotd, nbr, bands, key):
        o = {}
        for nm, Rr in rotd.items():
            Phi = Ub if Rr is None else Rr.rot(Ub)
            n90, npk, nsd = loc_t(Phi, nbr, Rr.pivot if nm == 'qr' else None)
            del Phi
            arrays[f'{key}|{nm}|n90'] = n90.int().cpu().numpy()
            arrays[f'{key}|{nm}|near_peak'] = npk.float().cpu().numpy()
            if nm == 'qr':
                arrays[f'{key}|{nm}|near_seed'] = nsd.float().cpu().numpy()
            per = [dict(band=i, s=s, e=e, n90_med=float(n90[s:e].double().median()), near_peak_med=float(npk[s:e].median()),
                        near_seed_med=float(nsd[s:e].median()) if nm == 'qr' else None) for i, (s, e) in enumerate(bands)]
            o[nm] = dict(bands=per, n90_med=float(n90.double().median()), near_peak_med=float(npk.median()),
                         near_seed_med=float(nsd.median()) if nm == 'qr' else None)
        return o

    if 'q2' in parts:
        pt.start('q2')
        q2_part()
    if 'q3' in parts:
        pt.start('q3')
        q3_part()
    if 'loc' in parts:
        pt.start('loc_site')
        Cm = Ctt.clone()
        Cm.fill_diagonal_(-2)
        nbr = Cm.topk(min(NNB, n - 1), dim=1).indices
        del Cm
        ks = S['ks']
        res['loc_site'] = localise(S['U'], {'eig': None, **S['rots']}, nbr, list(zip([0] + ks[:-1], ks)), 'loc|site')
        res['loc_site']['bands_A'] = [d['A'] for d in S['levels']]
    S.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ================================================================== Q1: the second stage on 64 coordinates
    if 'q1' in parts:
        Zs = {'joint|64': (F, FH, far_any), 'source|64': (P @ F, SH, far)}
        for name in cells:
            Zp, Zh, far_cell = Zs[name]
            o = dict()
            stored = ddm_json['combos'][name]
            s0 = sd.knn_scale(Zp)
            o['s0'], o['s0_stored'] = s0, stored['s0']
            gl = stored['gcv_sel']['lh']
            h = gl['h'] * stored['s0']                                           # frozen absolute width
            o.update(h_mult=gl['h'], h=h, m_gcv=gl['m'], far_mask='any-site' if name.startswith('joint') else 'site')

            # shared perturbations: random probe-coordinate differences, rescaled to f x probe RMS radius
            g1 = torch.Generator().manual_seed(3)
            a_, b_ = torch.randint(n, (m,), generator=g1), torch.randint(n - 1, (m,), generator=g1)
            b_ = b_ + (b_ >= a_)
            dz = Zp[a_.to(dev)] - Zp[b_.to(dev)]
            dz = dz / dz.norm(dim=1, keepdim=True)
            sZ = float(((Zp - Zp.mean(0)) ** 2).sum(1).mean().sqrt())
            Zq = torch.cat([Zh] + [Zh + f * sZ * dz for f in FRACS])            # (4m, 64): encoded points only
            o['sZ'] = sZ

            def split4(Xq):
                return Xq[:m], {f: Xq[(i + 1) * m:(i + 2) * m] - Xq[:m] for i, f in enumerate(FRACS)}

            # --- the stored practical benchmark: krr at the stored selection (stage_decode's rule), parity-checked
            pt.start(f'{name}|krr')
            ksel = stored['krr_sel']
            gk = sd.Gauss(Zp, ksel['h'] * s0)
            Xk = gk.w_krr(Zq, ksel['ridge']) @ Xc
            df_k = Zp.shape[1] + 1 + (0.0 if ksel['ridge'] >= sd.LINEAR_ONLY else float((gk.K * gk.krr_A(ksel['ridge']).T).sum()))
            del gk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- inner split: rebuild everything on the 80%, transfer only the budget n (and the ridge)
            pt.start(f'{name}|inner')
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(0)).to(dev)
            tr, va = perm[:int(0.8 * n)], perm[int(0.8 * n):]
            mtr = Xc[tr].mean(0)
            inner = Lift(Zp[tr], Xc[tr] - mtr, h)
            Xva = Xc[va] - mtr
            tot = float((Xva ** 2).sum())
            sel = {}
            for kind in KINDS:
                for adm in (ADM if kind != 'lead' else (None,)):
                    kA = len(inner.order(kind, adm))
                    grid = geom_grid(kA, 120, inner.ks)
                    se = [float(((Xh - Xva) ** 2).sum()) / tot for _, Xh in inner.curve(Zp[va], kind, adm, grid)]
                    i = int(np.argmin(se))
                    sel[f'{kind}|{adm}'] = dict(n=grid[i], rse=se[i], kA_inner=kA, grid=grid, curve=se)
            se = [float(((Xh - Xva) ** 2).sum()) / tot for _, Xh in inner.ridge_curve(Zp[va], RIDGES)]
            i = int(np.argmin(se))
            sel['ridge'] = dict(rho=RIDGES[i], rse=se[i], curve=se)
            o['inner'] = dict(r=inner.r, ks=inner.ks, t_eig=inner.t_eig, t_qr=inner.t_qr)
            o['sel'] = sel
            del inner, Xva
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # --- the full probe
            pt.start(f'{name}|full')
            full = Lift(Zp, Xc, h, rand_seeds=tuple(range(NSEEDS)))
            o.update(r=full.r, ks=full.ks, t_eig=full.t_eig, t_qr=full.t_qr, m_max_1e13=full.lh.m_max,
                     max_band=max(e - s for s, e in full.rot['qr'].bounds))
            res['spectrum'][name] = spectrum_table(full.lam_full, full.levels, (full.Y[:full.r] ** 2).sum(1).cpu().numpy(), E_tot)
            res['spectrum'][name]['complement_share'] = 1 - float((full.Y[:full.r] ** 2).sum()) / E_tot
            # controls: the stored GCV lift and the full band equal LatentHarmonics.weights
            Xg = full.lead_raw(Zh, gl['m'])
            Wg = full.lh.weights(Zh, gl['m']) @ Xc
            Xf = full.decode(Zh, 'lead', full.r)
            Wf = full.lh.weights(Zh, full.r) @ Xc
            cs, _ = ddm.vec_scores(Xg, XH)
            o['checks'] = dict(lead_gcv_vs_weights=float((Xg - Wg).abs().max() / Wg.abs().max()),
                               fullband_vs_weights=float((Xf - Wf).abs().max() / Wf.abs().max()),
                               lead_gcv_cos_vs_stored=float((cs - torch.as_tensor(ddm_npz[f'{name}|lh|cos'], device=dev)).abs().max()),
                               s0_rel=abs(s0 - stored['s0']) / stored['s0'], m_gcv_above_floor=gl['m'] > full.r)
            del Xg, Wg, Xf, Wf

            pt.start(f'{name}|finals')
            fin, per_all = {}, {}

            def final(key, Xq, n_req=None, n_eff=None, df=None):
                Xh, dX = split4(Xq)
                e, per = score(Xh, dX, key=f'q1|{name}|{key}')
                for f in FRACS:
                    a = per[f'amp{f}']
                    e['amp'][str(f)].update(med_far_cell=float(a[far_cell].median()), p95_far_cell=float(a[far_cell].quantile(.95)))
                e.update(n_req=n_req, n_eff=n_eff, df=df)
                fin[key], per_all[key] = e, per

            for kind in KINDS:
                for adm in (ADM if kind != 'lead' else (None,)):
                    nn = sel[f'{kind}|{adm}']['n']
                    ne = full.n_eff(kind, nn, adm)
                    final(f'{kind}|{adm}', full.decode(Zq, kind, nn, adm), nn, ne, ne)
            nw = sel['wbest|None']['n']
            final('lead@wbest_n', full.decode(Zq, 'lead', nw), nw, full.n_eff('lead', nw), full.n_eff('lead', nw))
            mg = min(gl['m'], full.lh.m_max)
            final('lead@gcv', full.lead_raw(Zq, gl['m']), gl['m'], mg, mg)
            rho = sel['ridge']['rho']
            final('ridge', full.ridge(Zq, rho), df=full.ridge_df(rho))
            final('krr', Xk, df=df_k)
            o['krr_sel'] = ksel
            o['checks']['krr_cos_vs_stored'] = float((per_all['krr']['cos'] - torch.as_tensor(ddm_npz[f'{name}|krr|cos'], device=dev)).abs().max())
            base = per_all['lead|None']
            for k_, e in fin.items():
                e['d_vs_lead'] = dict(cos=paired(per_all[k_]['cos'], base['cos']), agree=paired(per_all[k_]['agree'], base['agree']),
                                      kl=paired(per_all[k_]['kl'], base['kl']))
            for a_k, b_k in (('wbest|None', 'rbest|None'), ('wbest|None', 'ebest|None'), ('ridge', 'lead|None'), ('krr', 'lead|None')):
                fin[a_k].setdefault('d_vs', {})[b_k] = dict(cos=paired(per_all[a_k]['cos'], per_all[b_k]['cos']),
                                                             agree=paired(per_all[a_k]['agree'], per_all[b_k]['agree']))
            o['finals'] = fin
            del Xk, per_all

            # --- curves on H (descriptive), with function and amplification; lead at every band boundary
            pt.start(f'{name}|curves')
            curves = {}
            chosen = [v['n'] for k_, v in sel.items() if k_ != 'ridge']
            for kind in KINDS:
                for adm in (ADM if kind != 'lead' else (None,)):
                    kA = len(full.order(kind, adm))
                    extra = chosen + list(FIXED) + (full.ks + [gl['m']] if kind == 'lead' else [])
                    grid = geom_grid(kA, 28, extra)
                    pts, st = [], new_store()
                    for nn, Xq in full.curve(Zq, kind, adm, grid):
                        Xh, dX = split4(Xq)
                        p = curve_point(Xh, dX, store=st)
                        p['n'] = nn
                        if kind == 'lead' and nn in full.ks:
                            d = full.levels[full.ks.index(nn)]
                            p.update(A=d['A'], amp=d['amp'], boundary=True)
                        pts.append(p)
                    curves[f'{kind}|{adm}'] = pts
                    curve_rec(st, f'q1c|{name}|{kind}|{adm}')
            eqc = {}
            for b in ('qr', 'rand0'):                                            # wavelet / eigen equality, complete bands
                mx = 0.0
                for (k, Xa), (_, Xe) in zip(full.band_prefix(Zh, b, full.ks), full.band_prefix(Zh, 'eig', full.ks)):
                    mx = upd(mx, float(((Xa - Xe).norm(dim=1) / Xe.norm(dim=1).clamp_min(1e-300)).max()))
                eqc[b] = mx
            o['checks']['complete_band_rel'] = eqc
            rc, st = [], new_store()
            for rh, Xq in full.ridge_curve(Zq, RIDGES):
                Xh, dX = split4(Xq)
                p = curve_point(Xh, dX, store=st)
                p.update(rho=rh, df=full.ridge_df(rh))
                rc.append(p)
            curves['ridge'] = rc
            curve_rec(st, f'q1c|{name}|ridge')
            o['curves'] = curves

            if 'loc' in parts:
                pt.start(f'{name}|loc')
                d = torch.cdist(Zp, Zp)
                d.fill_diagonal_(float('inf'))
                nbr = d.topk(min(NNB, n - 1), dim=1, largest=False).indices
                del d
                o['loc'] = localise(full.lh.U[:, :full.r], dict(full.rot), nbr, list(zip([0] + full.ks[:-1], full.ks)), f'loc|{name}')
                o['loc']['bands_A'] = [d_['A'] for d_ in full.levels]
            res['q1'][name] = o
            del full, Zq
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(site, name, 'done', f'{time.time() - t_all:.0f}s', flush=True)

    pt.stop()
    res['parts'] = pt.log
    res['failed'] = gate(res)
    res['seconds'] = time.time() - t_all
    res['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
    np.savez(fnpz, **arrays)
    fout.write_text(json.dumps(clean(res), allow_nan=False))
    return dict(tag=tag, site=site, seconds=res['seconds'], gpu=res['gpu'], failed=res['failed'], parts=pt.log)


def gate(res):
    """Names of in-run controls outside their tolerance (non-finite counts as a failure)."""
    bad = []

    def chk(name, v, tol, lo=False):
        ok = v is not None and math.isfinite(v) and (v >= tol if lo else v <= tol)
        if not ok:
            bad.append(f'{name}={v}')

    c = res['checks']
    chk('patch_true_agree', c['patch_true_agree'], 1.0, lo=True)
    chk('patch_true_kl', c['patch_true_kl'], TOL['patch_true_kl'])
    chk('onestep_max_abs_diff', c['onestep_max_abs_diff'], TOL['onestep_max_abs_diff'])
    for k in ('q2_qr_vs_eig_rel', 'q2_rand_vs_eig_rel', 'q2_fd_vs_exact_rel'):
        if k in c:
            chk(k, c[k], TOL[k])
    for K in c.get('q2_lin_expected', []):
        v = c['q2_vs_linear'].get(str(K), c['q2_vs_linear'].get(K))
        if v is None:
            bad.append(f'q2_lin{K}_missing')
            continue
        chk(f'q2_lin{K}_vec_rel', v['vec_rel'], TOL['lin_vec_rel'])
        chk(f'q2_lin{K}_cos_vs_stored', v['cos_vs_stored'], TOL['lin_cos_vs_stored'])
        chk(f'q2_lin{K}_band_cos_vs_stored', v['band_cos_vs_stored'], TOL['band_cos_vs_stored'])
    req = lambda e, name: [chk(f'{name}:{k}', e[k][0], math.inf) for k in ('cos', 'agree', 'kl')]
    fin = lambda name, v: chk(name, v, math.inf)
    for e in res.get('q2', {}).get('levels', []):
        req(e, f'q2|{e["k"]}')
        for k in ('med', 'p95', 'med_far', 'p95_far'):
            fin(f'q2|{e["k"]}:amp_coord_{k}', e['amp_coord'][k])
        fin(f'q2|{e["k"]}:amp_row', e['amp_row']['sv_max_scaled'])
        fin(f'q2|{e["k"]}:rt', e['rt']['total'])
    for key, pts in res.get('q3', {}).get('decode', {}).items():
        for p in pts:
            for k in ('cos', 'agree', 'kl_mean'):
                fin(f'q3|{key}|{p["n"]}:{k}', p[k])
    for key, v in res.get('q3', {}).get('energy', {}).items():
        if not all(x is not None and math.isfinite(x) for x in v['share']):
            bad.append(f'q3_energy|{key}')
    if res.get('q2', {}).get('n_zero_delta', 0):
        bad.append(f'q2_n_zero_delta={res["q2"]["n_zero_delta"]}')
    for name, q in res['q1'].items():
        qc = q['checks']
        for k in ('lead_gcv_vs_weights', 'fullband_vs_weights', 'lead_gcv_cos_vs_stored', 'krr_cos_vs_stored', 's0_rel'):
            chk(f'{name}:{k}', qc.get(k), TOL[k])
        for b, v in qc['complete_band_rel'].items():
            chk(f'{name}:complete_band_{b}', v, TOL['complete_band_rel'])
        for k, e in q['finals'].items():
            req(e, f'{name}|{k}')
            for f, a in e['amp'].items():
                for kk in ('med', 'p95', 'med_far_cell', 'p95_far_cell'):
                    fin(f'{name}|{k}:amp{f}_{kk}', a[kk])
        for key, pts in q['curves'].items():
            for p in pts:
                for kk in ('cos', 'agree', 'kl_mean') + tuple(f'amp{f}_{x}' for f in FRACS for x in ('med', 'p95')):
                    fin(f'{name}|c|{key}|{p.get("n", p.get("rho"))}:{kk}', p[kk])
    return bad
