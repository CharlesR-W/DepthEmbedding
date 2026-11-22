"""Operators on a probe and out-of-sample readouts for xp/SPEC.md.

fit_op      (numpy, CPU)  kernels at each site, the joint modes of a chain of sites, each site's own walk modes.
Ext         (torch, GPU)  new points -> site-local readouts and joint coordinates, in chunks, without storing rows.

The row rules are ports of dm.ea_oos_rows / dm.knn_oos_rows (checked against them in modal_xp.selftest).
"""
from __future__ import annotations

import numpy as np

import dm
import spacetime as st

KER_KEYS = {'ea': ('v', 'beta', 'dmin', 'logz', 'target'), 'knn': ('v', 'k', 'floor', 'rk')}
KER_KEYS['pr'] = (*KER_KEYS['ea'], 'renyi')


def site_kernels(Cs, kname):
    """Per-site kernel dicts (with P) for Grams Cs (L, n, n)."""
    return [dm.build_kernel(np.asarray(C, dtype=np.float64), kname) for C in Cs]


def site_spectra(kers):
    """Full eigendecomposition of each lazy walk, once: (lam (L, n) descending, vec (L, n, n))."""
    n = len(kers[0]['P'])
    return st.site_eigs(np.stack([(k['P'] + np.eye(n)) / 2 for k in kers]))


def fit_op(kers, eigs, sel, K=32, R=16):
    """Joint modes on the chain of sites `sel` (indices into kers), relative site units, t = t*.

    Returns a flat dict of arrays (np.savez-able): vals, vecs (L, n, K), c, t, resid, site_lam (L, R) eigenvalues
    of the non-lazy walk, site_vec (L, n, R), and the kernel parameters needed for out-of-sample rows.
    """
    lam, vec = eigs[0][sel], eigs[1][sel]
    n = lam.shape[1]
    ms = np.stack([(kers[i]['P'] + np.eye(n)) / 2 for i in sel])
    c = (1 - lam[:, 1:1 + st.SCALE_R]).mean(1)
    out = dict(c=c, site_lam=2 * lam[:, 1:R + 1] - 1, site_vec=vec[:, :, 1:R + 1].astype(np.float32),
               lam2=lam[:, 1].copy())
    if len(sel) > 1:
        t = st.balance_direct(ms, c, eigs=(lam, vec))
        vals, vecs = st.modes(ms, c, t, k=K)
        out.update(t=t, vals=vals, vecs=vecs, resid=st.residual(ms, c, t, vals, vecs))
    kind = 'pr' if kers[0].get('renyi') == 2 else ('ea' if 'beta' in kers[0] else 'knn')
    for key in KER_KEYS[kind]:
        out[f'ker_{key}'] = np.stack([np.asarray(kers[i][key], dtype=np.float64) for i in sel])
    out['kind'] = np.array(kind)
    return out


# ------------------------------------------------------------------ torch side

def _cond_rows_t(d, perp, renyi=1):
    """Port of dm._cond_rows (no self-masking: mask by passing inf).  Returns logp (m, n)."""
    import torch
    dmin = d.min(1).values
    ds = d - dmin[:, None]
    fin = torch.where(torch.isfinite(ds), ds, torch.zeros_like(ds))
    target = float(np.log(perp))

    def cond(beta):
        lg = -beta[:, None] * ds
        logz = torch.logsumexp(lg, 1)
        logp = lg - logz[:, None]
        if renyi == 2:
            return -(2 * logp).exp().sum(1).log(), logp
        return logz + beta * (logp.exp() * fin).sum(1), logp

    lo, hi = torch.zeros(len(d), dtype=d.dtype, device=d.device), torch.ones(len(d), dtype=d.dtype, device=d.device)
    for _ in range(60):
        bad = cond(hi)[0] > target
        if not bad.any():
            break
        hi = torch.where(bad, hi * 2, hi)
    for _ in range(50):
        beta = (lo + hi) / 2
        big = cond(beta)[0] > target
        lo, hi = torch.where(big, beta, lo), torch.where(big, hi, beta)
    return cond((lo + hi) / 2)[1]


def ea_rows_t(c_new, ker, renyi=None):
    """Rows (m, n) of new points for the perplexity rule.  c_new: cosines to the probe; -inf masks an entry."""
    import torch
    d = 1 - c_new
    renyi = int(ker.get('renyi', 1)) if renyi is None else renyi
    logp_x = _cond_rows_t(d, float(ker['target']), renyi)
    logp_j = -ker['beta'][None, :] * (d - ker['dmin'][None, :]) - ker['logz'][None, :]
    logp_j = torch.where(torch.isfinite(d), logp_j, torch.full_like(d, -float('inf')))
    lw = torch.logaddexp(logp_x, logp_j) + ker['v'].log()[None, :]
    return torch.softmax(lw, 1)


def knn_rows_t(c_new, ker):
    import torch
    k = int(ker['k'])
    W = torch.full_like(c_new, float(ker['floor']))
    W.scatter_(1, c_new.topk(k, dim=1).indices, 1.0)
    W[(1 - c_new) <= ker['rk'][None, :]] = 1.0
    W[~torch.isfinite(c_new)] = 0.0
    W = W * ker['v'][None, :]
    return W / W.sum(1, keepdim=True)


class Ext:
    """Out-of-sample readouts against a fitted operator (the dict from fit_op, or its npz)."""

    def __init__(self, op, labels=None, device='cuda'):
        import torch
        T = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float64), device=device)
        self.kind = str(op['kind'])
        self.L = len(op['c'])
        self.kers = [{k: T(op[f'ker_{k}'][l]) for k in KER_KEYS[self.kind]} for l in range(self.L)]
        self.c = T(op['c'])
        self.site_lam, self.site_vec = T(op['site_lam']), T(op['site_vec'])
        self.joint = 'vecs' in op
        if self.joint:
            self.vals, self.vecs, self.t = T(op['vals']), T(op['vecs']), float(op['t'])
            L, c, t = self.L, self.c, self.t
            deg = torch.tensor([1.0] + [2.0] * (L - 2) + [1.0], dtype=torch.float64, device=device)
            adj = torch.diag(torch.ones(L - 1, dtype=torch.float64, device=device), 1)
            adj = adj + adj.T
            # one L x L system per mode: T_j = diag(1/(2c) + t deg - mu_j) - t adj
            self.Tm = torch.diag_embed((1 / (2 * c) + t * deg)[None, :] - self.vals[:, None]) - t * adj[None]
            sv = torch.linalg.svdvals(self.Tm)
            self.cond, self.smin = (sv[:, 0] / sv[:, -1]).cpu().numpy(), sv[:, -1].cpu().numpy()
        self.onehot = None
        if labels is not None:
            lab = torch.as_tensor(np.asarray(labels), device=device).long()
            self.onehot = torch.nn.functional.one_hot(lab, int(lab.max()) + 1).double()

    def rows(self, l, c_new):
        if self.kind in ('ea', 'pr'):
            return ea_rows_t(c_new, self.kers[l], renyi=2 if self.kind == 'pr' else 1)
        return knn_rows_t(c_new, self.kers[l])

    def site(self, l, c_new):
        """Site-local readouts of new points with cosines c_new (m, n) to the probe at chain position l."""
        import torch
        p = self.rows(l, c_new)
        fin = torch.where(torch.isfinite(c_new), c_new, torch.full_like(c_new, -2.0))
        out = dict(site_y=p @ self.site_vec[l] / self.site_lam[l][None, :],
                   cmax=fin.max(1).values,
                   perp=torch.exp(-(p * torch.log(p.clamp_min(1e-300))).sum(1)))
        if self.joint:
            out['b'] = p @ self.vecs[l] / (2 * self.c[l])
        if self.onehot is not None:
            out['mass'] = p @ self.onehot
        return out

    def solve(self, b):
        """b (L, m, K) from site() -> joint coordinates y (L, m, K)."""
        import torch
        # for mode j: T_j y[:, :, j] = b[:, :, j]
        rhs = b.permute(2, 0, 1)                          # (K, L, m)
        return torch.linalg.solve(self.Tm, rhs).permute(1, 2, 0)

    def embed(self, cross, chunk=4000):
        """cross: list over the chain's sites of cosines (m, n) (torch, any device).  Returns numpy dict:
        y (L, m, K), site_y (L, m, R), cmax, perp (L, m), mass (L, m, classes)."""
        import torch
        m = len(cross[0])
        acc = {}
        for at in range(0, m, chunk):
            parts = [self.site(l, cross[l][at:at + chunk].to(self.c.device, torch.float64)) for l in range(self.L)]
            res = {k: torch.stack([p[k] for p in parts]) for k in parts[0]}
            if self.joint:
                res['y'] = self.solve(res.pop('b'))
            for k, v in res.items():
                acc.setdefault(k, []).append(v.float().cpu())
        return {k: torch.cat(v, 1).numpy() for k, v in acc.items()}


def probe_gram(a):
    """a (n, d) torch float64 on GPU: centre on the mean, normalise rows.  Returns C, mean, norms, unit rows."""
    mu = a.mean(0)
    x = a - mu
    nrm = x.norm(dim=1)
    x = x / nrm.clamp_min(1e-12)[:, None]
    return (x @ x.T).clamp(-1, 1), mu, nrm, x


def cross_cos(y, mu, xn):
    """New activations y (m, d) against the probe's unit rows xn (n, d): cosines (m, n) and centred norms."""
    y = y.to(xn.dtype) - mu
    ny = y.norm(dim=1)
    return ((y / ny.clamp_min(1e-12)[:, None]) @ xn.T).clamp(-1, 1), ny
