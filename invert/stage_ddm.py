"""Double-diffusion stage (invert/SPEC_DDM.md): one site per unit.

For each coordinate system and K, decoders from K coordinates to the activation, all as weights on the centred
probe activations and all scored from the decoded vectors themselves:

  linear, krr     stage_decode's decoders, chosen by its code, grid and split (the joint numbers reproduce the
                  stored ones: a control).
  lh_split        latent harmonics (Gaussian second kernel, spectral cut-off), (h, delta) on the inner split.
  lh, lhlin,      the matched 2 x 2: {spectral cut-off, ridge} x {no trend, linear trend}, same kernel, same h grid,
  kr, krlin       each chosen by generalised cross-validation on the whole probe (never at interpolation).
  nn1, nn15       the nearest probe activation(s) in the coordinates;  nn1+krr: the nearest one moved by the
                  difference of the krr decodes (a real activation plus a smooth correction).

Also: the whole (h, m) surface on H, with patched agreement and round trips along it for a few cells; the
local-linear-regression residuals r_k; how well later coordinates are predicted from the first K by a lift tuned
for them; the second stage with the first stage's kernel rule (`pr2`).

Outputs: ddm/<tag>/<site>.json (summaries, surfaces) and <site>.npz (per-image arrays of the final decoders).
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch

import ddm
import stage_decode as sd
import stage_rt as srt

DT = torch.float64
HS = (1, 2, 4, 8, 16, 32, 64)                       # second-kernel width, in units of the median 15-NN distance
DELTAS = tuple(10 ** (-j / 2) for j in range(1, 25))
COUNTS = (8, 16, 32, 64, 128, 256, 512, 800, 1024, 2048, 4096)
RIDGES = tuple(10.0 ** j for j in range(-8, 3))
KFULL, LLR_K, KJ, MJ = 256, 64, 64, 256
FUNC = ('joint|16', 'joint|64', 'site|64', 'source|64')     # cells with function measured along the cut-off


def inner_split(n, device, seed=0, max_n=4000):
    """The inner 80/20 split of stage_decode.select."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)[:max_n].to(device)
    cut = int(0.8 * len(perm))
    return perm[:cut], perm[cut:]


def select_krr(zt, Ctt, nt):
    """(h, ridge) exactly as stage_decode.select chooses them."""
    tr, va = inner_split(len(zt), zt.device)
    ztr, zva = zt[tr], zt[va]
    Ctr, Cva, ntr, nva = Ctt[tr][:, tr], Ctt[va][:, tr], nt[tr], nt[va]
    s0 = sd.knn_scale(ztr)
    lin = float(sd.score(sd.w_linear(ztr, zva), Ctr, Cva, ntr, nva)[1].mean())
    best = (lin, sd.HS[0], sd.LINEAR_ONLY)
    for hm in sd.HS:
        gk = sd.Gauss(ztr, hm * s0)
        for rg in sd.RIDGES:
            v = float(sd.score(gk.w_krr(zva, rg), Ctr, Cva, ntr, nva)[1].mean())
            if v > best[0]:
                best = (v, hm, rg)
        del gk
    return dict(val=best[0], h=best[1], ridge=best[2], linear_val=lin)


def select_split(zt, X, hs, fn=None):
    """(h, delta) of the latent harmonics maximising mean captured share on the inner split.
    fn(va, Xh) -> per-image KL of the patched run to the clean one: if given, also the (h, delta) minimising it."""
    tr, va = inner_split(len(zt), zt.device)
    ztr, zva, Xtr, Xva = zt[tr], zt[va], X[tr], X[va]
    s0 = ddm.knn_scale(ztr)
    best, fbest, table, ftable = None, None, {}, {}
    for hm in hs:
        lh = ddm.LatentHarmonics(ztr, 'gauss', h=hm * s0)
        ms = [lh.n_keep(d) for d in DELTAS]
        cuts = sorted(set(ms))
        C = lh.proj(zva)
        sc = ddm.cut_scores(lh, Xtr, cuts, dict(v=(C, Xva, None)))
        cap = {m: float(c.mean()) for m, c in zip(cuts, sc['v']['cap'])}
        table[hm] = [cap[m] for m in ms]
        for d, m in zip(DELTAS, ms):
            if best is None or cap[m] > best[0]:
                best = (cap[m], hm, d, m)
        if fn is not None:
            A, Y = lh.coef(C), lh.U.T @ Xtr
            kl = {m: float(fn(va, A[:, :m] @ Y[:m]).mean()) for m in cuts}
            ftable[hm] = [kl[m] for m in ms]
            for d, m in zip(DELTAS, ms):
                if fbest is None or kl[m] < fbest[0]:
                    fbest = (kl[m], hm, d, m)
            del A, Y
        del lh
    fsel = None if fbest is None else dict(val=fbest[0], h=fbest[1], delta=fbest[2], m_inner=fbest[3], table=ftable)
    return dict(val=best[0], h=best[1], delta=best[2], m_inner=best[3]), table, fsel


def stat(a):
    a = a.double()
    return [float(a.mean()), float(a.median()), float(a.std() / max(1, len(a)) ** .5)]


def run(unit: dict, cfg: dict, mi):
    t0 = time.time()
    R = Path(mi.R)
    dev = 'cuda'
    pset, kname, site = unit['pset'], unit['kname'], unit['site']
    tag = f'{pset}__{kname}__L12'
    out = R / 'ddm' / tag
    out.mkdir(parents=True, exist_ok=True)
    fout = out / f'{site}{cfg.get("suffix", "")}.json'
    if fout.exists() and not cfg.get('force'):
        return 'cached'
    hs = tuple(cfg.get('hs', HS))
    want = cfg.get('combos')
    T = lambda a: torch.as_tensor(np.asarray(a), dtype=DT, device=dev)

    sp = dict(np.load(R / 'splits.npz'))
    idx, hid = mi.index_set(sp, pset), mi.index_set(sp, 'H')
    z = np.load(R / 'joint' / tag / 'modes.npz')
    sites = [str(s) for s in z['sites']]
    L, li = len(sites), sites.index(site)
    F = T(z['vecs'][li][:, :KFULL])                                             # joint mode slices on the probe
    cz = np.load(R / 'joint' / tag / 'coords_H.npz')
    assert abs(float(cz['basis']) - float(np.abs(z['vals']).sum())) < 1e-12, 'stale coordinates'
    FH, SH_stored = T(cz['y'][li][:, :KFULL]), T(cz['onestep'][li][:, :KFULL])
    n = len(idx)

    g = R / 'gram' / mi.centre_set(pset)
    Call = np.load(g / f'{site}.npy', mmap_mode='r')
    Ctt, CsH = T(Call[idx][:, idx]), T(Call[hid][:, idx])                        # stored cosines: first stage, krr parity
    nz = np.load(g / f'{site}_norms.npz')
    nt_stored = T(nz['c10'][idx])
    mu = torch.tensor(nz['mean'], device=dev)
    acts = np.load(R / 'acts/c10' / f'{site}.npy', mmap_mode='r')
    Xc = (torch.tensor(np.asarray(acts[idx]), device=dev) - mu).double()         # centred probe activations (n, D)
    XH = (torch.tensor(np.asarray(acts[hid]), device=dev) - mu).double()
    Xn = Xc / Xc.norm(dim=1, keepdim=True)
    nH = XH.norm(dim=1)
    res = dict(unit=unit, n=n, D=Xc.shape[1], hs=list(hs), deltas=list(DELTAS), ridges=list(RIDGES), combos={}, checks={})
    res['checks']['cos_vec_vs_stored'] = float((ddm.cosines(XH, Xn) - CsH).abs().max())

    kerd = dict(np.load(R / 'kernel' / f'{pset}__{kname}' / f'{site}.npz'))
    P = T(kerd['P'])
    kt = ddm.ker_to_torch(kerd, dev)
    rowsH = ddm.pr_oos_rows(1 - CsH, kt)                                        # first-stage rows of the true held-out activations
    SH = rowsH @ F
    res['checks']['onestep_max_abs_diff'] = float((SH - SH_stored).abs().max())     # control: equals the stored sources
    res['checks']['onestep_scale'] = float(SH_stored.abs().max())
    lam, Phi = ddm.site_coords(P, KFULL)
    PhiH = rowsH @ Phi / lam
    res['site_lam'] = lam.tolist()
    SP = P @ F
    del P, kerd

    def restrict(Xh, mask=None):
        """First-stage out-of-sample rows of decoded (centred) activations.  mask (m,): a probe index per row to
        leave out, for a decode that *is* that probe activation (the kernel has a zero diagonal)."""
        d = 1 - ddm.cosines(Xh, Xn)
        if mask is not None:
            d[torch.arange(len(d), device=dev), mask] = float('inf')
        return ddm.pr_oos_rows(d, kt)

    # patching
    lg_all = np.load(R / 'acts/c10/logits.npy')
    logp = torch.log_softmax(torch.tensor(lg_all[hid], device=dev).double(), 1)
    logp_probe = torch.log_softmax(torch.tensor(lg_all[idx], device=dev).double(), 1)
    model, _ = mi.load_model()
    shape = mi.SHAPES[site]

    def patched(Xh, ref=None):
        ref = logp if ref is None else ref
        xh = mu + Xh.float()
        with torch.inference_mode():
            lp = torch.cat([mi.suffix(model, site, xh[a:a + 500].reshape(-1, *shape)) for a in range(0, len(xh), 500)])
        lp = torch.log_softmax(lp.double(), 1)
        return (lp.argmax(1) == ref.argmax(1)).double(), (ref.exp() * (ref - lp)).sum(1)

    probe_kl = lambda va, Xh: patched(Xh, logp_probe[va])[1]

    ag, kl = patched(XH)
    res['checks']['patch_true_agree'], res['checks']['patch_true_kl'] = float(ag.mean()), float(kl.abs().max())

    # joint re-embedding of a patched run: earlier sites from the image, this site decoded, later sites recomputed
    vals, cc, tt = z['vals'][:KJ], z['c'], float(z['t'])
    deg = np.r_[1.0, np.full(L - 2, 2.0), 1.0]
    adj = np.diag(np.ones(L - 1), 1) + np.diag(np.ones(L - 1), -1)
    Tinv = T(np.stack([np.linalg.inv(np.diag(1 / (2 * cc) + tt * deg - m_) - tt * adj) for m_ in vals]))   # (KJ, L, L)
    c2 = T(2 * cc)
    Btrue = T(cz['onestep'][:, :MJ, :KJ])                                        # (L, MJ, KJ) true sources at every site
    S = {j: srt.Site(R, mi, sp, pset, kname, sites[j], z['vecs'][j][:, :KJ], dev) for j in range(li + 1, L)}
    ktj = {j: dict(v=s_.logv.exp(), beta=s_.beta, dmin=s_.dmin, logz=s_.logz, target=math.exp(s_.target), renyi=s_.renyi)
           for j, s_ in S.items()}
    varJ = FH[:, :KJ].var(0)
    arJ = torch.arange(MJ, device=dev)

    def joint_miss(Xh, K, mask=None):
        """Squared miss of the patched run's joint coordinate at this site from the requested one (first MJ images).
        mask: as in restrict, applied at this site and at the recomputed later ones."""
        with torch.inference_mode():
            later = srt.later_acts(model, mi, site, (mu + Xh[:MJ].float()).reshape(-1, *shape))
            b = Btrue[:, :, :K].clone()
            b[li] = restrict(Xh[:MJ], None if mask is None else mask[:MJ]) @ F[:, :K]
            for nm, a in later:
                j = sites.index(nm)
                d = 1 - S[j].cos(a.float().clone())
                if mask is not None:
                    d[arJ, mask[:MJ]] = float('inf')
                b[j] = ddm.pr_oos_rows(d, ktj[j]) @ S[j].f[:, :K]
            y = torch.einsum('kij,jmk->imk', Tinv[:K], b / c2[:, None, None])[li]
            return (((y - FH[:MJ, :K]) ** 2) / varJ[:K]).mean(1)

    res['checks']['joint_true_miss'] = float(joint_miss(XH, KJ).max())             # control: the true activation returns its own

    # non-harmonic selection (Dsilva et al.)
    t1 = time.time()
    r_site, ne_site = ddm.llr_residuals(Phi[:, :LLR_K])
    r_joint, ne_joint = ddm.llr_residuals(F[:, :LLR_K])
    sel_site, sel_joint = ddm.select_parsimonious(r_site, 16).to(dev), ddm.select_parsimonious(r_joint, 16).to(dev)
    res['llr'] = dict(site=r_site.tolist(), joint=r_joint.tolist(), neff_site=ne_site.tolist(), neff_joint=ne_joint.tolist(),
                      sel_site=sel_site.tolist(), sel_joint=sel_joint.tolist(), seconds=time.time() - t1,
                      joint_rms=F[:, :LLR_K].pow(2).mean(0).sqrt().tolist())

    # cells: name -> probe coordinates, held-out coordinates, round trips [(label, basis, target)], later coordinates
    first = lambda K: torch.arange(K, device=dev)
    rest64 = lambda sel: torch.tensor([k for k in range(LLR_K) if k not in set(sel.tolist())], device=dev)
    cells = {}
    for K in (4, 16, 64, 256):
        c = first(K)
        later = torch.arange(K, KFULL, device=dev) if K < KFULL else None
        cells[f'joint|{K}'] = dict(Zp=F[:, c], Zh=FH[:, c], rts=[('src', F[:, c], SH[:, c]), ('req', F[:, c], FH[:, c])],
                                   prem=None if later is None else (F[:, later], FH[:, later]))
        cells[f'site|{K}'] = dict(Zp=Phi[:, c], Zh=PhiH[:, c], rts=[('req', Phi[:, c] / lam[c], PhiH[:, c]), ('one', Phi[:, c], PhiH[:, c])],
                                  prem=None if later is None else (Phi[:, later], PhiH[:, later]))
        cells[f'source|{K}'] = dict(Zp=SP[:, c], Zh=SH[:, c], rts=[('req', F[:, c], SH[:, c])], prem=None)
    c = first(64)
    cells['site-t1|64'] = dict(Zp=Phi[:, c] * lam[c], Zh=PhiH[:, c] * lam[c], rts=[('req', Phi[:, c], PhiH[:, c] * lam[c])], prem=None)
    c, o = sel_joint, rest64(sel_joint)
    cells['joint-llr16|16'] = dict(Zp=F[:, c], Zh=FH[:, c], rts=[('src', F[:, c], SH[:, c]), ('req', F[:, c], FH[:, c])],
                                   prem=(F[:, o], FH[:, o]))
    c, o = sel_site, rest64(sel_site)
    cells['site-llr16|16'] = dict(Zp=Phi[:, c], Zh=PhiH[:, c], rts=[('req', Phi[:, c] / lam[c], PhiH[:, c]), ('one', Phi[:, c], PhiH[:, c])],
                                  prem=(Phi[:, o], PhiH[:, o]))
    arrays = {}
    gperm = torch.randperm(len(hid), generator=torch.Generator().manual_seed(1)).to(dev)

    def rt_of(Xh, rts, mask=None):
        rows = restrict(Xh, mask)
        return {lab: (((rows @ basis - target) ** 2) / target.var(0).clamp_min(1e-300)).mean(1) for lab, basis, target in rts}

    def rt_refs(rts):
        """What the round-trip unit means here: returning another image's coordinates, and returning zero."""
        return {lab: dict(shuffled=float((((target[gperm] - target) ** 2) / target.var(0)).mean()),
                          zero=float(((target ** 2) / target.var(0)).mean())) for lab, _, target in rts}

    def finals(name, Ws, rts, jK=None, masks=None):
        o = {}
        for dec, W in Ws.items():
            mask = (masks or {}).get(dec)
            Xh = W @ Xc
            cos, cap = ddm.vec_scores(Xh, XH)
            ag, kl = patched(Xh)
            e = dict(cos=stat(cos), cap=stat(cap), norm=stat(Xh.norm(dim=1) / nH), agree=stat(ag), kl=stat(kl))
            arrays[f'{name}|{dec}|cos'] = cos.float().cpu().numpy()
            arrays[f'{name}|{dec}|cap'] = cap.float().cpu().numpy()
            arrays[f'{name}|{dec}|agree'] = ag.bool().cpu().numpy()
            for lab, miss in rt_of(Xh, rts, mask).items():
                e[f'rt_{lab}'] = stat(miss)
                arrays[f'{name}|{dec}|rt_{lab}'] = miss.float().cpu().numpy()
            if jK is not None:
                e['rt_joint'] = stat(joint_miss(Xh, jK, mask))
            o[dec] = e
        if 'krr' in Ws:
            ck = torch.as_tensor(arrays[f'{name}|krr|cos'], device=dev).double()
            for dec in Ws:
                d = torch.as_tensor(arrays[f'{name}|{dec}|cos'], device=dev).double() - ck
                o[dec]['dcos_vs_krr'] = [float(d.mean()), float(d.std() / len(d) ** .5)]
        return o

    def pack(sc):
        return dict(train=[float(v) for v in sc['train']], gcv=[float(v) if np.isfinite(float(v)) else None for v in sc['gcv']],
                    df=[float(v) for v in sc['df']], H_cos=sc['H']['cos'].mean(1).tolist(), H_cap=sc['H']['cap'].mean(1).tolist(),
                    H_cap_med=sc['H']['cap'].median(1).values.tolist())

    def fpoint(f, Xh, rts, jK):
        cos, cap = ddm.vec_scores(Xh, XH)
        ag, kl = patched(Xh)
        f['cos'].append(float(cos.mean())); f['cap_med'].append(float(cap.median()))
        f['norm_med'].append(float((Xh.norm(dim=1) / nH).median()))
        f['agree'].append(float(ag.mean())); f['kl_med'].append(float(kl.median())); f['kl_mean'].append(float(kl.mean()))
        for lab, miss in rt_of(Xh, rts).items():
            f['rt'][lab].append([float(miss.mean()), float(miss.median())])
        if jK is not None:
            jm = joint_miss(Xh, jK)
            f['rt_joint'].append([float(jm.mean()), float(jm.median())])

    fnew = lambda rts, **kw: dict(cos=[], cap_med=[], norm_med=[], agree=[], kl_med=[], kl_mean=[],
                                  rt={lab: [] for lab, _, _ in rts}, rt_joint=[], **kw)

    def better(cur, v):
        return np.isfinite(v) and (cur is None or v < cur['gcv'])

    # reference that sees the activation: one kernel step in activation space
    res['ref'] = finals('ref', dict(step=rowsH), [('src', F[:, :16], SH[:, :16])])

    for name, cell in cells.items():
        if want and name not in want:
            continue
        tc = time.time()
        Zp, Zh, rts = cell['Zp'], cell['Zh'], cell['rts']
        K = Zp.shape[1]
        s0 = sd.knn_scale(Zp)
        Fp = sd.feats(Zp)
        pin = torch.linalg.pinv(Fp)
        Wl = sd.feats(Zh) @ pin
        Rres = Xc - Fp @ (pin @ Xc)                                             # residual of the linear trend on the probe
        baseH = Wl @ Xc
        detrend = lambda Wr: Wl + Wr - (Wr @ Fp) @ pin                          # weights of trend + lift of the residual
        ksel = select_krr(Zp, Ctt, nt_stored)
        gk = sd.Gauss(Zp, ksel['h'] * s0)
        Wk = gk.w_krr(Zh, ksel['ridge'])
        i1 = torch.cdist(Zh, Zp).argmin(1)
        W1 = torch.zeros(len(Zh), n, dtype=DT, device=dev)
        W1[torch.arange(len(Zh), device=dev), i1] = 1.0
        Ws = dict(linear=Wl, krr=Wk, nn1=W1, nn15=sd.w_nn(Zp, Zh), **{'nn1+krr': W1 + Wk - gk.w_krr(Zp[i1], ksel['ridge'])})
        del gk
        isfunc = name in FUNC and cfg.get('func', True)
        ssel, vtable, fsel = select_split(Zp, Xc, hs, probe_kl if isfunc else None)
        prem = cell['prem']
        if prem is not None:
            pm, ps = prem[0].mean(0), prem[0].std(0).clamp_min(1e-300)
            Lp, LH = (prem[0] - pm) / ps, (prem[1] - pm) / ps                    # standardised on the probe
            pbest = dict(lh=None, kr=None)
        gsel = dict(lh=None, lhlin=None, kr=None, krlin=None)
        ocos = ocap = None
        surf, fsurf = {}, {}
        for hm in hs:
            te = time.time()
            lh = ddm.LatentHarmonics(Zp, 'gauss', h=hm * s0)
            C = lh.proj(Zh)
            ms = [lh.n_keep(d) for d in DELTAS]
            cuts = sorted(set(ms) | {c_ for c_ in COUNTS if c_ <= lh.m_max} | {lh.m_max})
            ov = ddm.trend_overlap(lh, Fp)
            sc = dict(lh=ddm.cut_scores(lh, Xc, cuts, dict(H=(C, XH, None))),
                      lhlin=ddm.cut_scores(lh, Rres, cuts, dict(H=(C, XH, baseH)), df0=K + 1, overlap=ov),
                      kr=ddm.ridge_scores(lh, Xc, RIDGES, dict(H=(C, XH, None))),
                      krlin=ddm.ridge_scores(lh, Rres, RIDGES, dict(H=(C, XH, baseH)), df0=K + 1, overlap=ov))
            surf[hm] = dict(m_of_delta=ms, cuts=cuts, m_max=lh.m_max, **{a: pack(s_) for a, s_ in sc.items()})
            for arm in ('lh', 'lhlin'):                                         # generalised cross-validation, df <= n / 2
                ok = [i for i, d in enumerate(sc[arm]['df'].tolist()) if d <= n / 2]
                i = min(ok, key=lambda i_: float(sc[arm]['gcv'][i_]))
                v = float(sc[arm]['gcv'][i])
                if better(gsel[arm], v):
                    gsel[arm] = dict(gcv=v, h=hm, m=cuts[i], H_cos=float(sc[arm]['H']['cos'][i].mean()))
                    Wr = lh.weights(Zh, cuts[i], C)
                    Ws[arm] = Wr if arm == 'lh' else detrend(Wr)
            for arm in ('kr', 'krlin'):
                ok = [i for i, d in enumerate(sc[arm]['df']) if d <= n / 2]
                if ok:
                    i = min(ok, key=lambda i_: sc[arm]['gcv'][i_])
                    v = sc[arm]['gcv'][i]
                    if better(gsel[arm], v):
                        gsel[arm] = dict(gcv=v, h=hm, ridge=RIDGES[i], df=sc[arm]['df'][i], H_cos=float(sc[arm]['H']['cos'][i].mean()))
                        Wr = lh.ridge_weights(Zh, RIDGES[i], C)
                        Ws[arm] = Wr if arm == 'kr' else detrend(Wr)
            hc, hp = sc['lh']['H']['cos'].mean(1), sc['lh']['H']['cap'].mean(1)
            j = int(hc.argmax())
            if ocos is None or float(hc[j]) > ocos['H_cos']:
                ocos = dict(H_cos=float(hc[j]), H_cap=float(hp[j]), h=hm, m=cuts[j])
            j = int(hp.argmax())
            if ocap is None or float(hp[j]) > ocap['H_cap']:
                ocap = dict(H_cos=float(hc[j]), H_cap=float(hp[j]), h=hm, m=cuts[j])
            if hm == ssel['h']:
                ssel['m'] = lh.n_keep(ssel['delta'])
                Ws['lh_split'] = lh.weights(Zh, ssel['m'], C)
            if fsel is not None and hm == fsel['h']:
                fsel['m'] = lh.n_keep(fsel['delta'])
                Ws['lh_fsplit'] = lh.weights(Zh, fsel['m'], C)
            if prem is not None:                                                # later coordinates from these, tuned for them
                ok = [i for i, m in enumerate(cuts) if m <= n // 2]
                pc = ddm.cut_scores(lh, Lp, cuts, {})
                i = min(ok, key=lambda i_: float(pc['gcv'][i_]))
                if better(pbest['lh'], float(pc['gcv'][i])):
                    pbest['lh'] = dict(gcv=float(pc['gcv'][i]), h=hm, m=cuts[i], pred=lh.weights(Zh, cuts[i], C) @ Lp)
                pr_ = ddm.ridge_scores(lh, Lp, RIDGES, {})
                ok = [i for i, d in enumerate(pr_['df']) if d <= n / 2]
                if ok:
                    i = min(ok, key=lambda i_: pr_['gcv'][i_])
                    if better(pbest['kr'], pr_['gcv'][i]):
                        pbest['kr'] = dict(gcv=pr_['gcv'][i], h=hm, ridge=RIDGES[i], pred=lh.ridge_weights(Zh, RIDGES[i], C) @ Lp)
            if isfunc:                                                          # function along the cut-off and the ridge
                A, Y = lh.coef(C), lh.U.T @ Xc
                jK = K if name.startswith('joint') else None
                fc = [c_ for c_ in COUNTS if c_ < lh.m_max] + [lh.m_max]
                f = fnew(rts, cuts=fc)
                for m in fc:
                    fpoint(f, A[:, :m] @ Y[:m], rts, jK)
                fr = fnew(rts, ridges=list(RIDGES))
                wpos = lh.w.clamp_min(0)
                for rho in RIDGES:
                    fpoint(fr, (C / (wpos + rho)) @ Y, rts, jK)
                fsurf[hm] = dict(cut=f, ridge=fr)
                del A, Y
            surf[hm]['seconds'] = time.time() - te
            del lh, sc, C
            torch.cuda.empty_cache()
        o = dict(K=K, s0=s0, krr_sel=ksel, split_sel=ssel, split_table=vtable, fsplit_sel=fsel, gcv_sel=gsel, oracle_cos=ocos,
                 oracle_cap=ocap, surface=surf, func=fsurf, rt_refs=rt_refs(rts),
                 finals=finals(name, Ws, rts, jK=K if name.startswith('joint|') and K <= KJ else None, masks=dict(nn1=i1)))
        if prem is not None:
            r2 = lambda pred: (1 - ((pred - LH) ** 2).sum(0) / ((LH - LH.mean(0)) ** 2).sum(0)).tolist()

            def rho2(pred):                                                     # squared correlation: blind to a scale mismatch
                a_, b_ = pred - pred.mean(0), LH - LH.mean(0)
                return ((a_ * b_).sum(0) ** 2 / ((a_ ** 2).sum(0) * (b_ ** 2).sum(0)).clamp_min(1e-300)).tolist()

            o['later'] = dict(linear=r2(Wl @ Lp), linear_rho2=rho2(Wl @ Lp),
                              **{a: dict(h=b['h'], m=b.get('m'), ridge=b.get('ridge'), r2=r2(b['pred']), rho2=rho2(b['pred']))
                                 for a, b in pbest.items() if b})
            del Lp, LH, pbest
        o['seconds'] = time.time() - tc
        res['combos'][name] = o
        del Ws, Rres, baseH
        torch.cuda.empty_cache()
        print(site, name, f'{time.time() - tc:.0f}s', flush=True)

    # exploratory: the second stage built with the first stage's own kernel rule on latent distances.  A normalised
    # walk with zero diagonal has negative eigenvalues; only its positive spectrum is used.
    for K in (16, 64):
        for k2 in (30, 100, 400):
            name = f'joint-pr2_{k2}|{K}'
            if want and name not in want:
                continue
            tc = time.time()
            c = first(K)
            Zp, Zh = F[:, c], FH[:, c]
            rts = [('src', F[:, c], SH[:, c]), ('req', F[:, c], FH[:, c])]
            lh = ddm.LatentHarmonics(Zp, 'pr', k2=k2)
            C = lh.proj(Zh)
            ms = [lh.n_keep(d) for d in DELTAS]
            cuts = sorted(set(ms) | {c_ for c_ in COUNTS if c_ <= lh.m_max} | {lh.m_max})
            sc = ddm.cut_scores(lh, Xc, cuts, dict(H=(C, XH, None)))
            ok = [i for i, m in enumerate(cuts) if m <= n // 2]
            i = min(ok, key=lambda i_: float(sc['gcv'][i_]))
            hc = sc['H']['cos'].mean(1)
            j = int(hc.argmax())
            res['combos'][name] = dict(K=K, k2=k2, sinkhorn_err=lh.ker['err'],
                                       gcv_sel=dict(lh=dict(gcv=float(sc['gcv'][i]), m=cuts[i])),
                                       oracle_cos=dict(H_cos=float(hc[j]), m=cuts[j]),
                                       surface={0: dict(m_of_delta=ms, cuts=cuts, m_max=lh.m_max, lh=pack(sc))},
                                       finals=finals(name, dict(lh=lh.weights(Zh, cuts[i], C)), rts, jK=K),
                                       seconds=time.time() - tc)
            del lh, sc, C
            torch.cuda.empty_cache()
            print(site, name, f'{time.time() - tc:.0f}s', flush=True)

    res['seconds'] = time.time() - t0
    res['gpu'] = torch.cuda.get_device_name(0)
    np.savez(out / f'{site}{cfg.get("suffix", "")}.npz', **arrays)
    fout.write_text(json.dumps(res))
    return dict(tag=tag, site=site, seconds=res['seconds'], gpu=res['gpu'])
