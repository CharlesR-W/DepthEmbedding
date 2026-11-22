"""Round trip, matching pre-image and interventions at one site (perplexity-rule kernels only).

Source coordinates.  For an activation x at site l,  s_l(x) = p_l(x) . f_l  (K numbers: the kernel average of
the mode slices over x's neighbours in the probe).  It depends on the activation at site l only.  The joint
coordinates of an image are a fixed linear filter along depth of its source trajectory, per mode k:
    y_k = T_k^-1 (s_k / 2c),    T_k = diag(1/(2 c_l)) + t D_path - mu_k I     (L x L).
So a joint coordinate at site l is NOT a function of the activation at site l: it mixes the sources of the
neighbouring sites.  On the probe  s_l = P_l f_l  exactly.  The single-site inverse problem is therefore posed
in source coordinates:
  matching pre-image   the activation nearest x0 with  s_l(x) = s',  by damped Gauss-Newton with a trust region
                       (sequential quadratic programming on  min |x - x0|^2  s.t.  s_l(x) = s').
A requested joint coordinate psi' at site l is turned into a source target with the decoder's own weights:
s'(psi') = W(psi') (P_l f_l), i.e. the source coordinates decoded like any other function on the probe.

Where a patched run lands ("D"): sites before l keep the reference image's sources, site l uses x, and the
later sites use the activations the network computes from x; then the depth filter.

Experiments (m held-out images of H):
  E0 control   torch-path joint coordinates of true activations equal the stored (numpy-path) ones
  E1 sources   how the source coordinates relate to the joint ones; accuracy of s'(psi) on held-out images
  E2 decode    decode the image's own coordinates (kernel ridge and linear), re-encode (sources; patched run)
  E3 match     matching pre-image started from the decode, target the image's true sources
  E4 edit      give image a the sources of image b (another predicted class): residual form, matching from
               x_a, matching from the residual guess, decode alone, full swap, a random edit of the same size
  E5 request   source midpoint and 1.5x extrapolation, by matching from x_a
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import stage_decode as sd

DT = torch.float64


def crit_and_slope(beta, ds, renyi):
    """Log effective neighbour count of the conditional exp(-beta ds) and its derivative in beta."""
    lg = -beta[:, None] * ds
    logz = torch.logsumexp(lg, 1)
    p = torch.exp(lg - logz[:, None])
    m1 = (p * ds).sum(1)
    if renyi == 2:
        p2 = p * p
        s2 = p2.sum(1)
        return -torch.log(s2), 2 * ((p2 * ds).sum(1) / s2 - m1)
    return logz + beta * m1, -beta * ((p * ds * ds).sum(1) - m1 ** 2)


def bisect_beta(d, target, renyi=1):
    """Per-row Gaussian width on distances d (m, n) so that the conditional has log neighbour count `target`."""
    dmin = d.min(1).values
    ds = d - dmin[:, None]

    def ent(beta):
        return crit_and_slope(beta, ds, renyi)[0]

    lo = torch.zeros(len(d), dtype=d.dtype, device=d.device); hi = torch.ones_like(lo)
    for _ in range(60):
        bad = ent(hi) > target
        if not bool(bad.any()):
            break
        hi = torch.where(bad, hi * 2, hi)
    for _ in range(50):
        mid = (lo + hi) / 2
        e = ent(mid) > target
        lo = torch.where(e, mid, lo); hi = torch.where(e, hi, mid)
    return (lo + hi) / 2, dmin


class Site:
    def __init__(self, R, mi, sp, pset, kname, site, f, dev):
        idx = mi.index_set(sp, pset)
        nz = np.load(R / 'gram' / mi.centre_set(pset) / f'{site}_norms.npz')
        self.mu = torch.tensor(nz['mean'], device=dev)
        self.nt = torch.tensor(nz['c10'][idx], dtype=torch.float32, device=dev)
        acts = np.load(R / 'acts/c10' / f'{site}.npy', mmap_mode='r')
        self.Xc = torch.tensor(np.asarray(acts[idx]), device=dev) - self.mu
        self.Xn = self.Xc / self.nt[:, None]
        k = np.load(R / 'kernel' / f'{pset.replace("/", "_")}__{kname}' / f'{site}.npz')
        self.beta = torch.tensor(k['beta'], dtype=DT, device=dev); self.dmin = torch.tensor(k['dmin'], dtype=DT, device=dev)
        self.logz = torch.tensor(k['logz'], dtype=DT, device=dev); self.logv = torch.tensor(np.log(k['v']), dtype=DT, device=dev)
        self.target = float(np.log(k['target']))
        self.renyi = int(k['renyi']) if 'renyi' in k else 1
        self.f = torch.tensor(f, dtype=DT, device=dev)                    # (n, K) slice of the joint modes
        self.site, self.shape = site, mi.SHAPES[site]

    def cos(self, x):
        xc = x - self.mu
        return ((xc @ self.Xn.T) / xc.norm(dim=1, keepdim=True).clamp_min(1e-12)).double().clamp(-1, 1)

    def rows(self, x):
        d = 1 - self.cos(x)
        with torch.no_grad():
            bx, dmx = bisect_beta(d.detach(), self.target, self.renyi)
        ds = d - dmx[:, None]
        # The width is an implicit function of the distances (neighbour count = target).  One Newton correction
        # around the bisected value leaves the value unchanged and carries the exact derivative -H_d / H_beta.
        ent, slope = crit_and_slope(bx, ds, self.renyi)
        beta = bx - (ent - self.target) / slope.detach().clamp(max=-1e-300)
        lg = -beta[:, None] * ds
        logp_x = lg - torch.logsumexp(lg, 1, keepdim=True)
        logp_j = -self.beta[None] * (d - self.dmin[None]) - self.logz[None]
        return torch.softmax(torch.logaddexp(logp_x, logp_j) + self.logv[None], 1)


def later_acts(model, mi, site, h):
    """[(site name, flat activation)] for every site after `site`, computed by the network from h."""
    out = []
    i = mi.SITES.index(site)
    if i <= 9:
        for j in range(i, 9):
            h = model.blocks[j](h); out.append((mi.SITES[j + 1], h.flatten(1)))
        pool = h.mean((2, 3)); out.append(('pool', pool)); out.append(('logits', model.fc(pool)))
    elif site == 'pool':
        out.append(('logits', model.fc(h)))
    return out


def run(unit: dict, cfg: dict, mi):
    t0 = time.time()
    R = Path(mi.R); dev = 'cuda'
    pset, kname, site, K = unit['pset'], unit['kname'], unit['site'], unit.get('K', 16)
    m = unit.get('m', 256)
    tag = f'{pset.replace("/", "_")}__{kname}__L12'
    save_points = cfg.get('save_points', False)
    out = R / (cfg.get('points_dir', 'rt_points') if save_points else 'rt') / tag
    out.mkdir(parents=True, exist_ok=True)
    fout = out / f'{site}__K{K}.npz'
    if fout.exists() and (save_points or not cfg.get('force')):
        return 'cached'
    sp = dict(np.load(R / 'splits.npz'))
    modefile = R / 'joint' / tag / 'modes.npz'
    if save_points:
        with modefile.open('rb') as f:
            modes_sha256 = hashlib.file_digest(f, 'sha256').hexdigest()
        if modes_sha256 != cfg['modes_sha256']:
            raise ValueError(f'Modes hash mismatch: {modes_sha256} != {cfg["modes_sha256"]}')
    z = np.load(modefile)
    sites = [str(s) for s in z['sites']]
    L, li = len(sites), sites.index(site)
    vals, c, t = z['vals'][:K], z['c'], float(z['t'])
    S = [Site(R, mi, sp, pset, kname, s, z['vecs'][j][:, :K], dev) for j, s in enumerate(sites)]
    deg = np.r_[1.0, np.full(L - 2, 2.0), 1.0]
    adj = np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
    Tinv = torch.tensor(np.stack([np.linalg.inv(np.diag(1 / (2 * c) + t * deg - mu) - t * adj) for mu in vals]),
                        dtype=DT, device=dev)                              # (K, L, L)
    c2 = torch.tensor(2 * c, dtype=DT, device=dev)
    g = Tinv[:, li, li] / c2[li]                                           # (K,)
    model, _ = mi.load_model()

    hid = mi.index_set(sp, 'H')[:m]
    X = [torch.tensor(np.asarray(np.load(R / 'acts/c10' / f'{s}.npy', mmap_mode='r')[hid]), device=dev) for s in sites]
    logits = X[-1].double()
    pred = logits.argmax(1)
    stored = torch.tensor(np.load(R / 'joint' / tag / 'coords_H.npz')['y'][:, :m, :K], dtype=DT, device=dev)

    def joint_from_rows(b):
        """b (L, m, K) = p_l . f_l  ->  coordinates (L, m, K)."""
        return torch.einsum('kij,jmk->imk', Tinv, b / c2[:, None, None])

    with torch.no_grad():
        P = [S[j].rows(X[j]) for j in range(L)]                            # true rows at every site
        B = torch.stack([P[j] @ S[j].f for j in range(L)])                 # (L, m, K)
        Y = joint_from_rows(B)
    res = {}
    res['E0|maxdiff_rel'] = np.array(float((Y - stored).abs().max() / stored.abs().max()))
    psi = Y[li]                                                            # (m, K) coordinates at this site
    sdv = psi.std(0)                                                       # per-mode spread, the unit for errors

    def relerr(y, target):
        """Squared error of coordinates relative to their spread over the held-out images, per image."""
        return (((y - target) / sdv) ** 2).mean(1)

    def FS(x):
        """Source coordinates of activations x at this site (m, K)."""
        return S[li].rows(x) @ S[li].f

    def FD(x, ref):
        """All-site coordinates of the patched run: prefix from the reference, suffix recomputed by the network."""
        with torch.inference_mode():
            later = later_acts(model, mi, site, x.reshape(-1, *S[li].shape))
        b = B[:, ref].clone()
        b[li] = S[li].rows(x) @ S[li].f
        for name, a in later:
            j = sites.index(name)
            b[j] = S[j].rows(a.clone()) @ S[j].f
        lg = later[-1][1].double() if later else x.double()
        return joint_from_rows(b), torch.log_softmax(lg, 1)

    scale0 = (X[li] - S[li].mu).norm(dim=1).mean()

    def match(x0, target, iters=20, radius=3.0, probe_metric=False):
        """Activation near x0 with FS(x) = target: damped Gauss-Newton steps of the problem min |x - x0| s.t.
        FS(x) = target, accepted when the constraint residual falls.  Steps capped at 0.5 and the total move at
        `radius` (units: mean centred norm of the held-out activations), so infeasible targets stay bounded.
        Not guaranteed to be the nearest solution; `tang` (returned) is the share of the final move that lies
        along the constraint surface, 0 at a true nearest point.

        probe_metric: 'nearest' in the Mahalanobis distance of the probe cloud instead of the Euclidean one,
        i.e. x = x0 + G^T w with |w| minimal, G = centred probe activations / sqrt(n): the edit is a combination
        of probe activations with the smallest coefficients, so it moves along directions the probe varies in."""
        x = x0.clone()
        eye = torch.eye(K, dtype=DT, device=dev)[None]
        G = S[li].Xc / len(S[li].Xc) ** 0.5 if probe_metric else None
        w = torch.zeros(len(x0), len(S[li].Xc), dtype=DT, device=dev) if probe_metric else None
        for _ in range(iters):
            x.requires_grad_(True)
            y = FS(x)
            J = torch.stack([torch.autograd.grad(y[:, k].sum(), x, retain_graph=k < K - 1)[0] for k in range(K)], 1)
            x = x.detach(); y = y.detach()
            r = target - y
            Jd = (J @ G.T).double() if probe_metric else J.double()          # Jacobian in the search coordinates
            cur = w if probe_metric else (x - x0).double()
            A = Jd @ Jd.transpose(1, 2)
            lam = 1e-6 * A.diagonal(dim1=1, dim2=2).mean(1)
            rhs = r + (Jd @ cur[:, :, None])[:, :, 0]
            sol = torch.linalg.solve(A + lam[:, None, None] * eye, rhs)
            new = (Jd.transpose(1, 2) @ sol[:, :, None])[:, :, 0]
            proj = (Jd.transpose(1, 2) @ torch.linalg.solve(A + lam[:, None, None] * eye, (Jd @ cur[:, :, None])[:, :, 0])[:, :, None])[:, :, 0]
            tang = (cur - proj).norm(dim=1) / cur.norm(dim=1).clamp_min(1e-30)   # of the iterate entering this step
            xn = x0 + (new.float() @ G if probe_metric else new.float())
            step = xn - x
            sn = step.norm(dim=1, keepdim=True)
            cap = (0.5 * scale0 / sn.clamp_min(1e-30)).clamp(max=1.0)
            a = torch.ones(len(x), device=dev)
            r0 = r.norm(dim=1)
            with torch.no_grad():
                for _ in range(6):
                    f = a[:, None] * cap                                          # fraction of the full step taken
                    xt = x + f * step
                    off = xt - x0; on = off.norm(dim=1, keepdim=True)
                    sh = (radius * scale0 / on.clamp_min(1e-30)).clamp(max=1.0)
                    xt = x0 + off * sh
                    bad = (target - FS(xt)).norm(dim=1) > r0
                    if not bool(bad.any()):
                        break
                    a = torch.where(bad, a / 2, a)
                x = torch.where(bad[:, None], x, xt)
                if probe_metric:                                                  # keep w consistent with x = x0 + w G
                    wt = (cur + f.double() * (new - cur)) * sh.double()
                    w = torch.where(bad[:, None], cur, wt)
            del J, Jd, A
        return x, tang

    print(f'[rt {site} K{K}] E0 done {time.time() - t0:.0f}s', flush=True)
    # E0b control: the autograd Jacobian (with the implicit width derivative) against a central finite difference
    xq = X[li][:16].clone().requires_grad_(True)
    yq = FS(xq)
    Jq = torch.stack([torch.autograd.grad(yq[:, k].sum(), xq, retain_graph=k < K - 1)[0] for k in range(K)], 1)
    torch.manual_seed(1)
    dq = torch.randn_like(xq); dq = dq / dq.norm(dim=1, keepdim=True) * 0.02 * scale0
    with torch.no_grad():
        fd = (FS(xq.detach() + dq) - FS(xq.detach() - dq)) / 2
        lin = (Jq.double() @ dq.double()[:, :, None])[:, :, 0]
    res['E0|jac_fd_rel'] = np.array(float((fd - lin).norm() / fd.norm()))
    del Jq, xq, yq

    ar = torch.arange(m, device=dev)
    logp_clean = torch.log_softmax(logits, 1)

    def kl(lp_ref, lp):
        return (lp_ref.exp() * (lp_ref - lp)).sum(1)

    def cosc(a, b):
        a = a - S[li].mu; b = b - S[li].mu
        return (a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1)).clamp_min(1e-12)

    def put(key, v):
        res[key] = v.detach().float().cpu().numpy()

    with torch.no_grad():
        src = B[li]                                                        # (m, K) true source coordinates at this site
    ssd = src.std(0)

    def srel(y, target):
        return (((y - target) / ssd) ** 2).mean(1)

    # decoders at this site: weights on probe activations
    zt = S[li].f
    idx = mi.index_set(sp, pset)
    Ctt = torch.tensor(np.asarray(np.load(R / 'gram' / mi.centre_set(pset) / f'{site}.npy', mmap_mode='r')[idx][:, idx]),
                       dtype=DT, device=dev)
    sel = sd.select(zt, Ctt, S[li].nt.double())
    gk = sd.Gauss(zt, sel['krr']['h'] * sd.knn_scale(zt))
    pin = torch.linalg.pinv(sd.feats(zt))
    Pl = np.load(R / 'kernel' / f'{pset.replace("/", "_")}__{kname}' / f'{site}.npz')['P']
    s_probe = torch.tensor(Pl @ z['vecs'][li][:, :K].astype(np.float64), dtype=DT, device=dev)   # sources of the probe
    del Pl

    def Wd(y, kind='krr'):
        return gk.w_krr(y, sel['krr']['ridge']) if kind == 'krr' else sd.feats(y) @ pin

    def D(y, kind='krr'):
        return S[li].mu + Wd(y, kind).float() @ S[li].Xc

    print(f'[rt {site} K{K}] decoders ready {time.time() - t0:.0f}s', flush=True)
    x_true = X[li]
    if not save_points:
        # E1: sources against joint coordinates, and the source target computed from a joint coordinate
        put('E1|slope', (src * psi).sum(0) / (psi * psi).sum(0))
        put('E1|corr', ((src - src.mean(0)) * (psi - psi.mean(0))).mean(0) / (ssd * sdv))
        wts = (Tinv[:, li, :] / c2[None, :]).abs()                              # (K, L) weight of each site's source in y_l
        put('E1|share_ll', wts[:, li] / wts.sum(1))
        for kind in ('krr', 'linear'):
            put(f'E1|{kind}|srel', srel(Wd(psi, kind) @ s_probe, src))

        # E2/E3: decode own coordinates, re-encode, match to the true sources
        for kind in ('krr', 'linear'):
            xh = D(psi, kind)
            put(f'E2|{kind}|cos', cosc(xh, x_true))
            with torch.no_grad():
                put(f'E2|{kind}|srel', srel(FS(xh), src))
            yD, lpD = FD(xh, ar)
            put(f'E2|{kind}|relerr_D', relerr(yD[li], psi)); put(f'E2|{kind}|kl', kl(logp_clean, lpD))
            for j in range(li + 1, L):
                put(f'E2|{kind}|relerr_D@{sites[j]}', (((yD[j] - Y[j]) / Y[j].std(0)) ** 2).mean(1))
            if kind == 'krr':
                # E3: oracle target (the image's true sources: a solver control) and the inferred target s'(psi)
                for nm, tgt in (('E3', src), ('E3n', Wd(psi) @ s_probe)):
                    xm, tg = match(xh, tgt)
                    put(f'{nm}|cos', cosc(xm, x_true)); put(f'{nm}|tang', tg)
                    with torch.no_grad():
                        sm = FS(xm)
                    put(f'{nm}|miss', srel(sm, tgt)); put(f'{nm}|srel', srel(sm, src))
                    yD, lpD = FD(xm, ar)
                    put(f'{nm}|relerr_D', relerr(yD[li], psi)); put(f'{nm}|kl', kl(logp_clean, lpD))
                    put(f'{nm}|move', (xm - xh).norm(dim=1) / (x_true - S[li].mu).norm(dim=1))
                    for j in range(li + 1, L):
                        put(f'{nm}|relerr_D@{sites[j]}', (((yD[j] - Y[j]) / Y[j].std(0)) ** 2).mean(1))

        print(f'[rt {site} K{K}] E2/E3 done {time.time() - t0:.0f}s', flush=True)
    # E4: edits a -> b
    order = torch.arange(m, device=dev)
    bidx = torch.roll(order, 1)
    for shift in range(2, 40):                                             # ensure a different predicted class
        same = pred[bidx] == pred
        if not bool(same.any()):
            break
        bidx = torch.where(same, torch.roll(order, shift), bidx)
    res['E4|pair_ok'] = (pred[bidx] != pred).cpu().numpy()
    xa, xb = x_true, x_true[bidx]
    ya, yb = psi, psi[bidx]
    sa, sb = src, src[bidx]
    dab = (xb - xa).norm(dim=1)
    torch.manual_seed(0)
    edits = {'swap': xb, 'decode': D(yb), 'resid': xa + D(yb) - D(ya)}
    tangs = {}
    edits['match'], tangs['match'] = match(xa, sb)
    edits['resid+match'], tangs['resid+match'] = match(edits['resid'], sb)
    edits['match_p'], tangs['match_p'] = match(xa, sb, probe_metric=True)
    edits['match_req'], tangs['match_req'] = match(xa, Wd(yb) @ s_probe)      # target inferred from b's joint coordinates
    edits['resid+match_req'], _ = match(edits['resid'], Wd(yb) @ s_probe)
    for k_, v_ in tangs.items():
        put(f'E4|{k_}|tang', v_)
    for nm in ('resid', 'match', 'match_p', 'resid+match', 'decode', 'swap'):
        rnd = torch.randn_like(xa); rnd = rnd / rnd.norm(dim=1, keepdim=True)
        edits[f'rand~{nm}'] = xa + rnd * (edits[nm] - xa).norm(dim=1, keepdim=True)
    edits['none'] = xa

    def progress(y, y0, y1, sc):
        dlt = (y1 - y0) / sc
        return (((y - y0) / sc) * dlt).sum(1) / (dlt * dlt).sum(1)

    n_points = int(cfg.get('n_points', 64)) if save_points else 0
    if save_points and not 0 < n_points <= m:
        raise ValueError(f'n_points must be 1..{m}, got {n_points}')
    s_edit, y_patched = [], []
    for name, xp in edits.items():
        put(f'E4|{name}|size', (xp - xa).norm(dim=1) / dab)
        put(f'E4|{name}|cos_b', cosc(xp, xb)); put(f'E4|{name}|cos_a', cosc(xp, xa))
        with torch.no_grad():
            sp_ = FS(xp)
        yD, lp = FD(xp, ar)
        put(f'E4|{name}|prog_S', progress(sp_, sa, sb, ssd)); put(f'E4|{name}|miss_S', srel(sp_, sb))
        put(f'E4|{name}|prog_D', progress(yD[li], ya, yb, sdv)); put(f'E4|{name}|miss_D', relerr(yD[li], yb))
        put(f'E4|{name}|p_b', lp.exp()[ar, pred[bidx]]); put(f'E4|{name}|p_a', lp.exp()[ar, pred])
        put(f'E4|{name}|kl_b', kl(logp_clean[bidx], lp))
        if save_points:
            s_edit.append(sp_[:n_points].detach().cpu().numpy())
            y_patched.append(yD[:, :n_points].detach().cpu().numpy())
        for j in range(li + 1, L):
            put(f'E4|{name}|prog_D@{sites[j]}', progress(yD[j], Y[j], Y[j][bidx], Y[j].std(0)))

    if save_points:
        # Keep the original m-image pairing, trust-region scale and shape of RNG draws. Slicing
        # before constructing pairs or running the edits would not be the same E4 experiment.
        ia = ar[:n_points].cpu().numpy()
        ib = bidx[:n_points].cpu().numpy()
        methods = list(edits)
        meta = dict(seed=0, commit=cfg['commit'], stage_rt_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    modes_sha256=modes_sha256, vals=vals.tolist(), sites=sites, edit_site=site, tag=tag, m_full=m,
                    E0=float(res['E0|maxdiff_rel']), jac=float(res['E0|jac_fd_rel']),
                    gpu=torch.cuda.get_device_name(), seconds=time.time() - t0)
        with fout.open('xb') as f:
            np.savez(f, pair_pos=np.column_stack((ia, ib)), pair_ok=res['E4|pair_ok'][:n_points],
                     methods=np.array(methods), s_edit=np.stack(s_edit), y_patched=np.stack(y_patched),
                     y_a=Y[:, ia].cpu().numpy(), y_b=Y[:, ib].cpu().numpy(),
                     p_b=np.stack([res[f'E4|{nm}|p_b'][:n_points] for nm in methods]),
                     size=np.stack([res[f'E4|{nm}|size'][:n_points] for nm in methods]), meta=np.array(json.dumps(meta)))
        return meta

    print(f'[rt {site} K{K}] E4 done {time.time() - t0:.0f}s', flush=True)
    # E5: requests with no known pre-image
    for tau in (0.5, 1.5):
        tgt = sa + tau * (sb - sa)
        xm, _ = match(xa, tgt)
        with torch.no_grad():
            put(f'E5|{tau}|miss_S', srel(FS(xm), tgt))
        _, lp = FD(xm, ar)
        put(f'E5|{tau}|size', (xm - xa).norm(dim=1) / dab)
        put(f'E5|{tau}|p_b', lp.exp()[ar, pred[bidx]]); put(f'E5|{tau}|p_a', lp.exp()[ar, pred])

    okp = res['E4|pair_ok'].astype(bool)
    for k_ in list(res):
        if k_.startswith(('E4|', 'E5|')) and k_ != 'E4|pair_ok' and res[k_].shape == okp.shape:
            res[k_] = np.where(okp, res[k_], np.nan)
    res['gpu_peak_gb'] = np.array(torch.cuda.max_memory_allocated() / 2 ** 30)
    np.savez(fout, **res)
    summ = {k: float(np.nanmean(v)) for k, v in res.items()}
    summ['seconds'] = time.time() - t0; summ['sel'] = sel
    (out / f'{site}__K{K}.json').write_text(json.dumps(summ))
    return dict(site=site, K=K, seconds=summ['seconds'], E0=summ['E0|maxdiff_rel'], jac=summ['E0|jac_fd_rel'], gpu=summ['gpu_peak_gb'])
