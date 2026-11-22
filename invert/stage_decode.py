"""Decode stage: from K joint coordinates at `site` to the activation at `tsite` (default: the same site).

Every decoder is a weight matrix W (m, n) on the probe activations, so held-out scores come from Grams and
norms, and the decoded activation itself (for patching) is mean + W (X_probe - mean).

Decoders (weights depend only on coordinates; tuned ones are tuned on an inner split of the probe, never on
the held-out images):
  mean      W = 0                                      reference: the probe mean
  linear    least-squares projection on span{1, coordinates}                 no parameters
  nn        mean of the 15 nearest probe images in coordinates               k = 15
  krr       linear + Gaussian kernel ridge on its residual                   (h, ridge) on the inner split
  gh        geometric harmonics on the coordinates (spectral cut-off)        (h, delta) on the inner split
  lp        Laplacian pyramid, stopping level on the inner split
  lpz       Laplacian pyramid with the zero-diagonal stopping rule of Fernandez et al. (no split)
References that see the target activation: pca (K+1 components of the probe Gram), step (one kernel step in
activation space at tsite).  Benchmark: mlp (trained; coordinates -> normalised activation).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

DT = torch.float64
HS = (0.5, 1, 2, 4, 8, 16, 32)
RIDGES = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
DELTAS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-6, 1e-8)


def sqd(a, b):
    return torch.cdist(a, b).pow(2)


def knn_scale(z, k=15):
    d = torch.cdist(z, z)
    return float(d.kthvalue(k + 1, dim=1).values.median())


def score(W, Ctt, Cst, nt=None, ns=None):
    """cos and captured share per decoded point.  With norms: raw (unnormalised) targets."""
    if nt is not None:
        W = W * nt[None, :]
    dot = (W * Cst).sum(1)
    n2 = ((W @ Ctt) * W).sum(1)
    cos = dot / n2.clamp_min(1e-300).sqrt()
    if ns is not None:
        cap = (2 * ns * dot - n2) / ns ** 2
    else:
        cap = 2 * dot - n2
    return cos, cap


def feats(z):
    return torch.cat([torch.ones(len(z), 1, dtype=z.dtype, device=z.device), z], 1)


def w_linear(zt, zs):
    return feats(zs) @ torch.linalg.pinv(feats(zt))


def w_nn(zt, zs, k=15):
    nn = torch.cdist(zs, zt).topk(k, largest=False).indices
    W = torch.zeros(len(zs), len(zt), dtype=zt.dtype, device=zt.device)
    W.scatter_(1, nn, 1.0 / k)
    return W


LINEAR_ONLY = 1e30          # ridge value meaning "no kernel part"


class Gauss:
    """Gaussian kernel on coordinates.  Kernel ridge by direct solves; eigendecomposition only for geometric harmonics."""

    def __init__(self, zt, h, eig=False):
        self.zt, self.h = zt, h
        self.K = torch.exp(-sqd(zt, zt) / (2 * h * h))
        self.ft = feats(zt)
        self.pin = torch.linalg.pinv(self.ft)
        self._A = {}
        if eig:
            self.w, self.U = torch.linalg.eigh(self.K)
            self.w = self.w.clamp_min(0)

    def cross(self, zs):
        return torch.exp(-sqd(zs, self.zt) / (2 * self.h * self.h))

    def krr_A(self, ridge):
        """(K + ridge I)^-1 (I - H): maps probe values to kernel coefficients of the residual of the linear fit."""
        if ridge not in self._A:
            n = len(self.zt)
            eye = torch.eye(n, dtype=self.K.dtype, device=self.K.device)
            self._A = {ridge: torch.linalg.solve(self.K + ridge * eye, eye - self.ft @ self.pin)}
        return self._A[ridge]

    def w_krr(self, zs, ridge):
        lin = feats(zs) @ self.pin
        return lin if ridge >= LINEAR_ONLY else lin + self.cross(zs) @ self.krr_A(ridge)

    def w_gh(self, zs, delta):
        r = self.w > delta * self.w.max()
        return ((self.cross(zs) @ self.U[:, r]) / self.w[r]) @ self.U[:, r].T


def lp_weights(zt, zs_list, levels, zero_diag=False, Ctt=None, max_levels=16):
    """Laplacian pyramid as weights.  If levels is None and zero_diag: stop at the minimum of the
    zero-diagonal training error (needs Ctt).  Returns list of W (one per zs), levels used, error history."""
    n = len(zt)
    dtt = sqd(zt, zt)
    dts = [sqd(zs, zt) for zs in zs_list]
    h = float(dtt.max().sqrt())
    Rt = torch.eye(n, dtype=zt.dtype, device=zt.device)
    Ws = [torch.zeros(len(zs), n, dtype=zt.dtype, device=zt.device) for zs in zs_list]
    best, keep, hist = None, None, []
    nlev = levels if levels is not None else max_levels
    for i in range(nlev):
        K = torch.exp(-dtt / (2 * h * h))
        if zero_diag:
            K.fill_diagonal_(0)
        rs = K.sum(1, keepdim=True)
        if float(rs.min()) < 1e-200:
            break
        K = K / rs
        for j, d in enumerate(dts):
            Ks = torch.exp(-d / (2 * h * h))
            Ws[j] = Ws[j] + (Ks / Ks.sum(1, keepdim=True).clamp_min(1e-300)) @ Rt
        Rt = Rt - K @ Rt
        if levels is None:
            err = float(((Rt @ Ctt) * Rt).sum(1).mean())
            hist.append(err)
            if best is None or err < best:
                best, keep = err, ([w.clone() for w in Ws], i + 1)
            elif err > 1.05 * best:
                break
        h /= 2
    if levels is None:
        return keep[0], keep[1], hist
    return Ws, nlev, hist


def w_pca(eig, Cst, k):
    """Projection of the unit target on the top-k principal directions of the unit probe activations."""
    k = min(k, int((eig[0] > 1e-9 * eig[0].max()).sum()))
    w, U = eig[0][-k:], eig[1][:, -k:]
    return ((Cst @ U) / w) @ U.T


def select(zt, Ctt, nt, seed=0, max_n=4000):
    """Choose (h, ridge), (h, delta) and the pyramid level on an inner 80/20 split of (a subsample of) the probe."""
    n = len(zt)
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)[:max_n].to(zt.device)
    cut = int(0.8 * len(perm))
    tr, va = perm[:cut], perm[cut:]
    ztr, zva = zt[tr], zt[va]
    Ctr, Cva = Ctt[tr][:, tr], Ctt[va][:, tr]
    ntr, nva = nt[tr], nt[va]                     # scores below are for the raw (unnormalised) target
    s0 = knn_scale(ztr)
    lin = float(score(w_linear(ztr, zva), Ctr, Cva, ntr, nva)[1].mean())
    best_k, best_g = (lin, HS[0], LINEAR_ONLY), None          # the linear decoder is always a candidate
    for hm in HS:
        gk = Gauss(ztr, hm * s0, eig=True)
        for rg in RIDGES:
            v = float(score(gk.w_krr(zva, rg), Ctr, Cva, ntr, nva)[1].mean())
            if best_k is None or v > best_k[0]:
                best_k = (v, hm, rg)
        for dl in DELTAS:
            v = float(score(gk.w_gh(zva, dl), Ctr, Cva, ntr, nva)[1].mean())
            if best_g is None or v > best_g[0]:
                best_g = (v, hm, dl)
        del gk
    # pyramid: validation error after each level
    best_l, nlev = None, 0
    for lev in range(1, 17):
        try:
            (Wv,), _, _ = lp_weights(ztr, [zva], lev)
        except Exception:  # noqa: BLE001
            break
        v = float(score(Wv, Ctr, Cva, ntr, nva)[1].mean())
        if best_l is None or v > best_l[0]:
            best_l = (v, lev)
        elif v < best_l[0] - 0.02:
            break
    return dict(krr=dict(val=best_k[0], h=best_k[1], ridge=best_k[2], linear_val=lin),
                gh=dict(val=best_g[0], h=best_g[1], delta=best_g[2]),
                lp=dict(val=best_l[0], level=best_l[1]))


def train_mlp(zt, Xn, zs_list, seed=0, epochs=150):
    """Benchmark: MLP from standardised coordinates to the normalised activation; early stopping on 10% of the probe."""
    torch.manual_seed(seed)
    dev = Xn.device
    mu, sd = zt.mean(0), zt.std(0).clamp_min(1e-12)
    f = lambda z: ((z - mu) / sd).float()
    n = len(zt)
    perm = torch.randperm(n, device=dev)
    va, tr = perm[: n // 10], perm[n // 10:]
    net = torch.nn.Sequential(torch.nn.Linear(zt.shape[1], 1024), torch.nn.GELU(), torch.nn.Linear(1024, 1024),
                              torch.nn.GELU(), torch.nn.Linear(1024, Xn.shape[1])).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=epochs * ((len(tr) + 255) // 256))
    zin, best, state = f(zt), -1.0, None
    for ep in range(epochs):
        net.train()
        for b in tr[torch.randperm(len(tr), device=dev)].split(256):
            loss = ((net(zin[b]) - Xn[b]) ** 2).sum(1).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if ep % 5 == 4 or ep == epochs - 1:
            net.eval()
            with torch.no_grad():
                p = net(zin[va])
                cap = float((2 * (p * Xn[va]).sum(1) - (p * p).sum(1)).mean())
            if cap > best:
                best, state = cap, {k: v.clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state); net.eval()
    with torch.no_grad():
        return [net(f(zs)) for zs in zs_list], best


def run(unit: dict, cfg: dict, mi):
    """unit: pset, kname, L (12|13), site, tsite (optional), targets (optional)."""
    t0 = time.time()
    R = Path(mi.R)
    dev = 'cuda'
    pset, kname, site = unit['pset'], unit['kname'], unit['site']
    tsite = unit.get('tsite', site)
    tag = f'{pset.replace("/", "_")}__{kname.replace("/", "_")}__L{unit.get("L", 12)}'
    out = R / 'decode' / tag; out.mkdir(parents=True, exist_ok=True)
    fout = out / f'{site}__{tsite}.npz'
    if fout.exists() and not cfg.get('force'):
        return 'cached'
    sp = dict(np.load(R / 'splits.npz'))
    idx = mi.index_set(sp, pset)
    cen = mi.centre_set(pset)
    z = np.load(R / 'joint' / tag / 'modes.npz')
    sites = [str(s) for s in z['sites']]
    li = sites.index(site)
    Z = torch.tensor(z['vecs'][li], dtype=DT, device=dev)                       # (n, Kmax)
    Ks = [k for k in cfg.get('Ks', [4, 16, 64, 256]) if k <= Z.shape[1]]
    targets = [t for t in unit.get('targets', cfg.get('targets', ['H'] + mi.OOD))
               if (R / 'joint' / tag / f'coords_{t.replace("/", "_")}.npz').exists()]

    g = R / 'gram' / cen
    Call = np.load(g / f'{tsite}.npy', mmap_mode='r')
    Ctt = torch.tensor(np.asarray(Call[idx][:, idx]), dtype=DT, device=dev)
    nz = np.load(g / f'{tsite}_norms.npz')
    mu = torch.tensor(nz['mean'], device=dev)
    nt = torch.tensor(nz['c10'][idx], dtype=DT, device=dev)
    acts10 = np.load(R / 'acts/c10' / f'{tsite}.npy', mmap_mode='r')
    Xc = torch.tensor(np.asarray(acts10[idx]), device=dev) - mu                   # raw centred probe activations
    Xn = Xc / nt.float()[:, None]
    model, _ = mi.load_model()
    shape = mi.SHAPES[tsite]

    T = {}
    for t in targets:
        cz = np.load(R / 'joint' / tag / f'coords_{t.replace("/", "_")}.npz')
        assert abs(float(cz['basis']) - float(np.abs(z['vals']).sum())) < 1e-12 and cz['y'].shape[2] == Z.shape[1], 'stale coordinates'
        if t in mi.OOD:
            Cst = np.asarray(np.load(g / f'{tsite}_x_{t}.npy', mmap_mode='r')[:, idx])
            ns = nz[t]
            xt = np.load(R / 'acts' / t / f'{tsite}.npy')
            lg = np.load(R / 'acts' / t / 'logits.npy')
        else:
            tid = mi.index_set(sp, t)
            Cst = np.asarray(Call[tid][:, idx]); ns = nz['c10'][tid]
            xt = np.asarray(acts10[tid]); lg = np.load(R / 'acts/c10/logits.npy')[tid]
        T[t] = dict(Y=torch.tensor(cz['y'][li], dtype=DT, device=dev), Cst=torch.tensor(Cst, dtype=DT, device=dev),
                    ns=torch.tensor(ns, dtype=DT, device=dev), x=torch.tensor(xt, device=dev),
                    logp=torch.log_softmax(torch.tensor(lg, device=dev).double(), 1))
    res, meta = {}, dict(unit=unit, Ks=Ks, targets=targets, sel={})

    def patch(xhat):
        with torch.inference_mode():
            outs = [mi.suffix(model, tsite, xhat[a:a + 500].reshape(-1, *shape)) for a in range(0, len(xhat), 500)]
        return torch.log_softmax(torch.cat(outs).double(), 1)

    def record(name, K, t, W=None, direct=None, unit=False):
        """Scores for weights W (m, n), or for a directly decoded normalised activation `direct` (m, D).

        unit=True: W acts on unit activations only (PCA reference); its raw decode uses the true norm."""
        d = T[t]
        key = f'{t}|{K}|{name}'
        if W is not None:
            cn, pn = score(W, Ctt, d['Cst'])
            Wf = W.float()
            dirn = Wf @ Xn
            if unit:
                cr, pr = cn, pn
                raw = d['ns'].float()[:, None] * dirn
            else:
                cr, pr = score(W, Ctt, d['Cst'], nt, d['ns'])
                raw = Wf @ Xc                                                    # decoded centred activation (scale predicted)
        else:
            xr = d['x'] - mu
            xs = xr / xr.norm(dim=1, keepdim=True).clamp_min(1e-12)
            dot = (direct * xs).sum(1).double(); n2 = (direct * direct).sum(1).double()
            cn, pn = dot / n2.clamp_min(1e-300).sqrt(), 2 * dot - n2
            raw = direct * nt.float().mean(); dirn = direct                      # scale: the probe's mean norm
            cr = cn
            pr = 1 - ((raw - xr) ** 2).sum(1).double() / d['ns'] ** 2
        res[key + '|cos'] = cn.float().cpu().numpy(); res[key + '|cap'] = pn.float().cpu().numpy()
        res[key + '|cos_raw'] = cr.float().cpu().numpy(); res[key + '|cap_raw'] = pr.float().cpu().numpy()
        if cfg.get('patch', True) and tsite != 'logits':
            for nm, xh in (('', mu + raw), ('_on', mu + d['ns'].float()[:, None] * dirn / dirn.norm(dim=1, keepdim=True).clamp_min(1e-12))):
                lp = patch(xh)
                res[key + f'|kl{nm}'] = (d['logp'].exp() * (d['logp'] - lp)).sum(1).float().cpu().numpy()
                res[key + f'|agree{nm}'] = (lp.argmax(1) == d['logp'].argmax(1)).cpu().numpy()

    # references independent of K
    for t in targets:
        d = T[t]
        record('mean', 0, t, W=torch.zeros_like(d['Cst']))
        if cfg.get('patch', True) and tsite != 'logits':          # control: patching the true activation changes nothing
            lp = patch(d['x'])
            res[f'{t}|0|true|kl'] = (d['logp'].exp() * (d['logp'] - lp)).sum(1).float().cpu().numpy()
    eigC = torch.linalg.eigh(Ctt)
    Graw = Ctt * nt[:, None] * nt[None, :]          # Gram of the raw centred probe activations
    for K in Ks:
        zt = Z[:, :K]
        zs = {t: T[t]['Y'][:, :K] for t in targets}
        sel = select(zt, Ctt, nt)
        meta['sel'][K] = sel
        s0 = knn_scale(zt)
        Wl = {t: w_linear(zt, zs[t]) for t in targets}
        gk = Gauss(zt, sel['krr']['h'] * s0)
        Wk = {t: gk.w_krr(zs[t], sel['krr']['ridge']) for t in targets}
        del gk; gk = Gauss(zt, sel['gh']['h'] * s0, eig=True)
        Wg = {t: gk.w_gh(zs[t], sel['gh']['delta']) for t in targets}
        del gk
        Wlp, _, _ = lp_weights(zt, [zs[t] for t in targets], sel['lp']['level'])
        Wlz, lz, _ = lp_weights(zt, [zs[t] for t in targets], None, zero_diag=True, Ctt=Graw)
        meta['sel'][K]['lpz'] = dict(level=lz)
        for j, t in enumerate(targets):
            record('linear', K, t, W=Wl[t]); record('krr', K, t, W=Wk[t]); record('gh', K, t, W=Wg[t])
            record('lp', K, t, W=Wlp[j]); record('lpz', K, t, W=Wlz[j])
            record('nn', K, t, W=w_nn(zt, zs[t]))
            record('pca', K, t, W=w_pca(eigC, T[t]['Cst'], K + 1), unit=True)
        del Wl, Wk, Wg, Wlp, Wlz
        if cfg.get('mlp', True) and K in cfg.get('mlp_Ks', [16, 64]):
            preds, val = train_mlp(zt, Xn, [zs[t] for t in targets])
            meta['sel'][K]['mlp'] = dict(val=val)
            for t, p in zip(targets, preds):
                record('mlp', K, t, direct=p)
        torch.cuda.empty_cache()
    # one kernel step in activation space at tsite (uses the target activation itself)
    try:
        import dm
        ker = dict(np.load(R / 'kernel' / f'{pset.replace("/", "_")}__{kname.replace("/", "_")}' / f'{tsite}.npz'))
        for t in targets:
            rows = torch.tensor(dm.rows(T[t]['Cst'].cpu().numpy(), ker), dtype=DT, device=dev)
            record('step', 0, t, W=rows)
    except FileNotFoundError:
        pass
    meta['seconds'] = time.time() - t0
    np.savez(fout, **res)
    (out / f'{site}__{tsite}.json').write_text(json.dumps(meta))
    return dict(tag=tag, site=site, tsite=tsite, seconds=meta['seconds'])
