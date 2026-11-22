"""X4 of xp/SPEC.md (with its "Revisions after review"): straight paths between the activations of two real images
at a start site l0, pushed through the rest of ResNet-20 `c10-s0`; where the motion concentrates, exact Jacobian
gains, ReLU pattern changes, outputs, and the paths' coordinates in a suffix embedding.

Called by modal_xp.x4 (GPU container) as run(unit, cfg).  Units and outputs (under /results/<cfg['out']>/, default x4):
    {'part': 'paths', 'l0': <site>}   l0 in pixels, stem, b3, b6, b8, b9      -> paths_<l0>.npz
    {'part': 'probe_gain'}                                                    -> probe_gain.npz
    {'part': 'checks'}                                                        -> checks.json
    pairs.npz is written by the first paths unit (and reused by later ones).
cfg: n_pairs (per kind, default 150), out (default 'x4'), force (recompute existing outputs), grid_paths (paths per
chunk on the T = 64 grid, default 16).  Every unit returns a small dict (timings and check numbers).

Conventions used by every file
------------------------------
* Network `c10-s0` (ModelZoo resnet20-relu-seed0, step 64000), eval mode, float32 network passes (TF32 off); Gram,
  kernel and embedding algebra in float64.
* Sites, in order (common.ALL_SITES): pixels (normalised input, (x/255 - mean)/std per channel, 3x32x32), stem, b1..b9
  (post-ReLU block outputs: 16x32x32 for stem..b3, 32x16x16 for b4..b6, 64x8x8 for b7..b9), pool (64), logits (10).
  Activations are flattened.  "Activation units" = the network's own values at that site.
* Path: x(s) = (1 - s) a + s b at l0, a and b the raw (uncentred) activations of the two endpoint images at l0.
  s is dimensionless in [0, 1].  The downstream sites m of a unit are all sites after l0 (`sites`).
* Probe: /results/probes.npz c10 (2,000 CIFAR-10 test images).  mu_m = probe mean activation at site m; "centred
  radius" = |x_m - mu_m|; rad_med[m] = median centred radius of the probe at m.
* "Raw" quantities use x_m itself.  "Unit" (angular) quantities use u_m = (x_m - mu_m) / |x_m - mu_m|, the
  centred normalised activation the Gram and the embedding see; unit step norms are chord lengths on the unit
  sphere (radians, up to the chord/arc difference, negligible for small steps).
* Log gain: natural log of the largest singular value, log sigma_max(J).  Raw gain: J = J_{l0->m}(x), units
  log(activation units at m per activation unit at l0).  Angular gain (a fixed-radius derivative): unit tangent t of
  the sphere around mu_l0 at x (t orthogonal to x - mu_l0), lifted to activation units as r t with r = |x - mu_l0|,
  pushed through J, then through the derivative of y -> (y - mu_m)/|y - mu_m|; dimensionless (radians per radian).
  "FTLE" (log gain / number of sites crossed) is not stored: divide by (index(m) - index(l0)) in ALL_SITES.
  Logits target: exact (J J^T from 10 VJPs; angular from J J^T, J x_hat and the 10x10 output derivative).
  Other targets: power iteration on J^T J (JVP then VJP), 20 iterations from a fixed random start.  The stored
  gain is the Rayleigh-Ritz estimate over the span of all 20 iterates (= the Krylov space the power method visits,
  i.e. the Lanczos estimate at no extra network cost); it is a lower bound on sigma_max and never below the plain
  power estimate |J v_20| (stored separately as gain_pow_* / pow_*).  Convergence = relative change of the Ritz
  estimate between using the first 17 and all 20 iterates, (ritz_20 - ritz_17) / ritz_20 (>= 0; small = converged).
  In the checks the Ritz estimate is far closer to an explicit SVD than the plain power estimate (see checks.json).

paths_<l0>.npz  (P = 2 n_pairs paths: rows 0..n-1 same-class pairs, rows n..2n-1 different-class pairs;
                 M = number of downstream sites; NaN where a quantity does not exist)
------------------------------------------------------------------------------------------------------------
identity and data
  l0               ()          start site name
  sites            (M,)        downstream site names (ALL_SITES after l0)
  taps             (Mr,)       downstream tap sites up to b9 used for ReLU patterns (empty for l0 = b9)
  chain            (L,)        sites of the suffix operator, l0..logits (stem..logits for l0 = pixels)
  pair_idx         (P, 2)      CIFAR-10 test-set indices of the endpoint images [a (s=0), b (s=1)]
  kind             (P,)        0 = same-class pair, 1 = different-class pair
  labels           (P, 2)      true labels of a and b (both images are classified correctly by c10-s0)
  images           (P, 2, 3, 32, 32) uint8 endpoint images
  s256, s64, s1024 grids k/T, T = 256 / 64 / 1024
  rad_med          (M,)        probe median centred radius at each downstream site (activation units)
  rad_med_l0       ()          same at l0
  chord_l0         (P,)        |b - a| at l0 (activation units); stratify same/different-class pairs on this
  cos_l0           (P,)        cosine between the centred endpoints (a - mu_l0, b - mu_l0) at l0
step norms on the T = 256 grid (257 points)
  d_raw, d_unit    (P, M, 256) |x_m(s_{k+1}) - x_m(s_k)| for the raw / unit activation
  d64_raw, d64_unit (P, M, 64) the same on the T = 64 grid (every 4th point)
  chord_raw, chord_unit (P, M) |x_m(1) - x_m(0)| (raw: activation units; unit: chord on the sphere)
  d_l0_unit        (P, 256)    unit step norms at l0 itself (the raw path is straight there; this is the purely
                               geometric angular stretching from passing near the probe mean)
  share_l0_unit    (P,)        window share (below) of d_l0_unit
path statistics, for v in {raw, unit}, each (P, M), from the T = 256 step norms
  share_v          share of arc length in the fastest window of width 1/16 in s (16 steps): 1/16 for a uniform path,
                   1 for a step.  PRIMARY measure of localised stretching.
  win_v            centre (in s) of that fastest window
  peak_v           peak speed = max step norm * 256 (units per unit s)
  arc_v            arc length = sum of step norms
  sigma_v          log(m_64 / m_256) / log 4, m_T = largest step on the T grid (1 for a smooth path, 0 for a jump)
  S_v              m_256 / chord
  ev_v             excess length arc / chord - 1
  width_v          transition width: (k90 - k10 + 1)/256, k_q the first step where cumulative arc >= q of total
  tau_v            location (in s, step centre) of the largest step
T = 1024 refinement on the first min(25, n) pairs of each kind (Q paths)
  ref_rows         (Q,)        rows of P that were refined
  d1024_raw, d1024_unit (Q, M, 1024) step norms on the T = 1024 grid
  share1024_v, peak1024_v, sigma1024_v, S1024_v, width1024_v, tau1024_v  (Q, M), v in {raw, unit}: as above on the
                   T = 1024 grid; window still 1/16 in s (64 steps); sigma1024 = log(m_256 / m_1024) / log 4
ReLU patterns at tap sites (T = 256 grid).  Taps are post-ReLU, so pattern = activation > 0.  This covers the
ReLU at each tap (block output) only, NOT the ReLU inside each block (after its first conv).
  relu_from0       (P, Mr, 257) fraction of units at the tap whose pattern differs from s = 0
  relu_prev        (P, Mr, 256) fraction differing from the previous grid point (cell crossings per step)
outputs (T = 256 grid)
  logits           (P, 257, 10)
  pred             (P, 257) int8 predicted class
  margin           (P, 257) top logit minus second logit (>= 0)
  n_switch         (P,)        number of prediction changes along the path
  s_switch         (P,)        s of the first prediction change (midpoint of the step), NaN if none
exact speed on the T = 64 grid (65 points), forward mode, v = b - a
  speed            (P, M, 65)  |J_{l0->m}(x(s)) (b - a)| = |dx_m/ds| (activation units per unit s); compare peak_raw
  speed_unit       (P, M, 65)  |du_m/ds| exact (radians per unit s)
  rad_out          (P, M, 65)  centred radius at m divided by rad_med (small = near the probe mean, unit
                               quantities ill-conditioned)
  rad_in           (P, 65)     centred radius at l0 divided by rad_med_l0
  jvp_fd_med, jvp_fd_p99, jvp_fd_frac (M,)  relative discrepancy |speed - fd| / max(speed, fd) between the exact speed
                   and the forward finite difference of the T = 256 path at the same s (backward at s = 1): median,
                   99th percentile, fraction above 1e-2.  Nonzero partly because ReLU cells are crossed inside a step.
  jvp_fd1024_med, jvp_fd1024_p99, jvp_fd1024_frac (M,)  the same against T = 1024 (refined paths); should be smaller
log gain on the T = 64 grid, targets `gain_targets` = [logits, b9] ([logits] when l0 = b9)
  gain_raw, gain_ang   (P, G, 65)  raw / angular log gain l0 -> target
  gain_conv_raw, gain_conv_ang (P, G, 65)  convergence of the Ritz estimate (NaN for the exact logits target)
  gain_pow_raw, gain_pow_ang   (P, G, 65)  plain power-iteration estimate log |A v_20| (NaN for logits)
  gain_rad_out     (P, G, 65)  centred radius at the target / probe median there
  small_flag       (P, G, 65)  bool, rad_in < 0.1 or gain_rad_out < 0.1: angular gain unreliable
  align_raw        (P, 65)     |J v| / (sigma_max |v|), v = b - a, target logits (exact); in [0, 1]
  align_ang        (P, 65)     angular analogue: v replaced by its tangential part, J by the angular map
  align_raw_b9     (P, 65)     |J v| / (sigma_max_est |v|), target b9 (power-iteration sigma: may exceed 1 slightly)
suffix embedding (kernel ea100, joint operator fitted on the clean probe over `chain`, t = t*, frozen-probe
extension of the path points; T = 64 grid)
  emb_y            (L, P, 65, 16) joint coordinates, first 16 modes
  emb_site_y       (L, P, 65, 16) site-local coordinates (Nystrom extension of each site's own walk modes)
  emb_mass         (L, P, 65, 10) kernel-row weight on probe images of each class
  emb_cmax         (L, P, 65)  largest cosine to a probe image (support)
  emb_perp         (L, P, 65)  perplexity of the kernel row
  ref_cmax         (L, 1000)   emb_cmax for 1,000 clean held-out images (reference distribution for support scores)
  ref_idx          (1000,)     their CIFAR-10 test indices (held-out, correctly classified, not in any pair)
  probe_y          (L, 2000, 16) the probe's own joint coordinates (op vecs), the clean cloud
  probe_site_y     (L, 2000, 16) the probe's own site-local coordinates (site walk eigenvectors)
  probe_labels, probe_idx (2000,)
  op_vals (32,), op_c (L,), op_t (), op_site_lam (L, 16): joint eigenvalues, relative site units, coupling,
                   site walk eigenvalues
  ext_cond, ext_smin (32,)  per-mode condition number / smallest singular value of the L x L extension system
  endpoint_err     ()  max |emb_y(path at s=0 or 1) - emb_y(endpoint images embedded as plain images)|
  replay_err       ()  max |masked replay of the probe - probe vecs| (all 32 modes)
  meta             ()  JSON string: timings, GPU, check numbers, pair source
pairs.npz: pair_idx (2N, 2), kind (2N,), labels (2N, 2), N per kind (prefix-stable: the first n of each kind are the
same for any N >= n), held (held-out correctly classified non-probe test indices with margin > 1e-3), seed (the
pair-drawing seed; same-class pairs use seed, different-class seed + 1).

probe_gain.npz  (N = 2000 probe images; starts = pixels, stem, b3, b6; targets = [logits, b9, next])
  starts (4,), targets (3,), next_sites (4,) the "next" target per start (stem, b1, b4, b7)
  gain_raw, gain_ang   (4, 3, N)  raw / angular log gain (logits exact; b9 and next by power iteration)
  conv_raw, conv_ang   (4, 3, N)  convergence of the Ritz estimate (NaN for logits)
  pow_raw, pow_ang     (4, 3, N)  plain power-iteration estimate (NaN for logits)
  rad_in               (4, N)     centred radius at the start / probe median there
  rad_out              (4, 3, N)  centred radius at the target / probe median there
  logits (N, 10), pred (N,), margin (N,) top-1 minus top-2 logit, labels (N,), probe_idx (N,), meta (JSON)
  Probe coordinates to colour by gain: probe_y in paths_<l0>.npz (same image order).

checks.json: named numeric checks with known answers (asserted) and comparisons (recorded); see checks().
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

R = '/results'
NET = 'c10-s0'
STARTS = ['pixels', 'stem', 'b3', 'b6', 'b8', 'b9']
GAIN_STARTS = ['pixels', 'stem', 'b3', 'b6']
T_HI, T_LO, T_REF = 256, 64, 1024
N_PAIRS, N_REF, N_REFCMAX = 150, 25, 1000
K_KEEP = 16
KERNEL = 'ea100'
PAIR_SEED = 4040
MARGIN_MIN = 1e-3            # "correct" needs top-1 minus top-2 logit above this, so the held-out set is GPU-independent
SMALL_RAD = 0.1
ITERS = 20
B9, LOGITS = 10, 12          # indices in ALL_SITES


# ------------------------------------------------------------------ network pieces

def step(model, i, h):
    """Map the activation at site i (ALL_SITES index) to site i + 1."""
    if i == 0:
        return model.act(model.bn1(model.conv1(h)))
    if i <= 9:
        return model.blocks[i - 1](h)
    if i == 10:
        return h.mean((2, 3))
    if i == 11:
        return model.fc(h)
    raise ValueError(i)


def downstream(model, i0, h, i1=LOGITS):
    out = []
    for i in range(i0, i1):
        h = step(model, i, h)
        out.append(h)
    return out


def seg(model, i0, i1):
    def f(h):
        for i in range(i0, i1):
            h = step(model, i, h)
        return h
    return f


def _bshape(t, n):
    return t.view((-1,) + (1,) * (n - 1))


def _unitize(w):
    return w / _bshape(w.flatten(1).norm(dim=1).clamp_min(1e-30), w.dim())


# ------------------------------------------------------------------ gains

def gain_exact(model, i0, X, mu_in, mu_out, v=None):
    """Exact raw and angular log gain l0 -> logits.  X (B, ...) float32 at site i0.  mu_in/mu_out float64 flat."""
    import torch
    f = seg(model, i0, LOGITS)
    xr = X.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        y = f(xr)
        k = y.shape[1]
        G = torch.stack([torch.autograd.grad(y[:, j].sum(), xr, retain_graph=j < k - 1)[0].flatten(1)
                         for j in range(k)], 1).double()                       # (B, k, D): rows of J
    y = y.detach().double()
    JJ = G @ G.transpose(1, 2)
    xt = X.flatten(1).double() - mu_in
    r = xt.norm(dim=1)
    xh = xt / r[:, None]
    Jx = (G @ xh[:, :, None])[..., 0]
    yt = y - mu_out
    rho = yt.norm(dim=1)
    uh = yt / rho[:, None]
    eye = torch.eye(k, dtype=torch.float64, device=X.device)
    D = (eye[None] - uh[:, :, None] * uh[:, None, :]) / rho[:, None, None]
    M = (r ** 2)[:, None, None] * (D @ (JJ - Jx[:, :, None] * Jx[:, None, :]) @ D)
    s_raw = torch.linalg.eigvalsh(JJ)[:, -1].clamp_min(1e-300).sqrt()
    s_ang = torch.linalg.eigvalsh((M + M.transpose(1, 2)) / 2)[:, -1].clamp_min(1e-300).sqrt()
    out = dict(raw=s_raw.log(), ang=s_ang.log(), r=r, rho=rho, y=y)
    if v is not None:
        vf = v.flatten(1).double()
        Jv = (G @ vf[:, :, None])[..., 0]
        xv = (xh * vf).sum(1, keepdim=True)
        Pv = vf - xh * xv
        JPv = Jv - Jx * xv
        At = r * (D @ JPv[:, :, None])[..., 0].norm(dim=1) / Pv.norm(dim=1)
        out.update(align_raw=Jv.norm(dim=1) / (s_raw * vf.norm(dim=1)), align_ang=At / s_ang, Jv=Jv)
    return out


def _ritz(Vs, Zs, k, tol=1e-6, sub=128):
    """Largest Ritz value of |A c| / |c| over span of the first k iterates.  Vs (B, n, D) iterates, Zs (B, n, E) their
    images A v.  Directions whose Gram eigenvalue is below tol * largest are dropped (float32 noise)."""
    import torch
    out = []
    for at in range(0, len(Vs), sub):
        V, Z = Vs[at:at + sub, :k].double(), Zs[at:at + sub, :k].double()
        lam, U = torch.linalg.eigh(V @ V.transpose(1, 2))
        keep = lam > tol * lam[:, -1:]
        W = U * torch.where(keep, lam.clamp_min(1e-300).rsqrt(), torch.zeros_like(lam))[:, None, :]
        M = W.transpose(1, 2) @ (Z @ Z.transpose(1, 2)) @ W
        out.append(torch.linalg.eigvalsh((M + M.transpose(1, 2)) / 2)[:, -1].clamp_min(1e-300).sqrt())
    return torch.cat(out)


def gain_power(model, i0, i1, X, mu_in, mu_out, angular, iters=ITERS, seed=0):
    """Power iteration on J^T J (raw) or A^T A (angular, A = D J P r) for l0 = i0 -> i1, 20 iterations (one JVP and
    one VJP each), with Rayleigh-Ritz extraction on the span of all iterates (the Krylov space the power method
    visits, i.e. the Lanczos estimate; still a lower bound on sigma_max, never below the plain power estimate).
    Returns log Ritz estimate, its convergence (ritz_20 - ritz_17)/ritz_20, log plain power estimate |A v_20|, the
    plain estimate's convergence (est_20 - est_17)/est_20, input and output centred radii (float64 tensors)."""
    import torch
    f = seg(model, i0, i1)
    nd = X.dim()
    xt = X.flatten(1).double() - mu_in
    r = xt.norm(dim=1)
    xh = (xt / r[:, None]).float()
    rs = _bshape(r.float(), nd)
    xr = X.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        y = f(xr)
    yt = y.detach().flatten(1).double() - mu_out
    rho = yt.norm(dim=1)
    uh = (yt / rho[:, None]).float()
    rhof = rho.float()[:, None]

    def proj(w):
        wf = w.flatten(1)
        return (wf - xh * (xh * wf).sum(1, keepdim=True)).view_as(w)

    gen = torch.Generator(device=X.device).manual_seed(seed)
    v = torch.randn(X.shape, generator=gen, device=X.device, dtype=X.dtype)
    v = _unitize(proj(v) if angular else v)
    hist, Vs, Zs = [], [], []
    for _ in range(iters):
        _, z = torch.func.jvp(f, (X,), (v * rs if angular else v,))
        zf = z.flatten(1)
        if angular:
            zf = (zf - uh * (uh * zf).sum(1, keepdim=True)) / rhof          # D J (r v)
            w = zf / rhof                                                   # D D J (r v) (zf is already tangent)
        else:
            w = zf
        hist.append(zf.double().norm(dim=1))
        Vs.append(v.flatten(1)); Zs.append(zf)
        g = torch.autograd.grad(y, xr, grad_outputs=w.view_as(y), retain_graph=True)[0]
        v = _unitize(proj(g * rs) if angular else g)
    del xr, y
    hist = torch.stack(hist, 1)
    est = hist[:, -1]
    conv_pow = (hist[:, -1] - hist[:, -4]) / est
    Vs, Zs = torch.stack(Vs, 1), torch.stack(Zs, 1)
    rz = torch.maximum(_ritz(Vs, Zs, iters), est)
    rz3 = torch.maximum(_ritz(Vs, Zs, iters - 3), hist[:, -4])
    return dict(log=rz.log(), conv=(rz - rz3) / rz, log_pow=est.clamp_min(1e-300).log(), conv_pow=conv_pow, r=r, rho=rho)


# ------------------------------------------------------------------ path statistics (numpy)

def path_stats(d, dlo, chord, win):
    """d (..., T) step norms; dlo (..., T/4) step norms of the 4x coarser grid; chord (...); win = window in steps."""
    d, dlo, chord = (np.asarray(a, np.float64) for a in (d, dlo, chord))
    T = d.shape[-1]
    with np.errstate(divide='ignore', invalid='ignore'):
        arc = d.sum(-1)
        cs = np.concatenate([np.zeros(d.shape[:-1] + (1,)), np.cumsum(d, -1)], -1)
        w = cs[..., win:] - cs[..., :-win]
        m_hi, m_lo = d.max(-1), dlo.max(-1)
        c = cs[..., 1:] / arc[..., None]
        k10, k90 = (c >= 0.1).argmax(-1), (c >= 0.9).argmax(-1)
        return dict(share=w.max(-1) / arc, win=(w.argmax(-1) + win / 2) / T, peak=m_hi * T, arc=arc,
                    sigma=np.log(m_lo / m_hi) / np.log(4), S=m_hi / chord, ev=arc / chord - 1,
                    width=(k90 - k10 + 1) / T, tau=(d.argmax(-1) + 0.5) / T)


def _steps(H):
    """H (q, T+1, D) -> step norms (q, T), 4x coarser step norms (q, T/4), chord (q,), in float64."""
    Hd = H.double()
    d = (Hd[:, 1:] - Hd[:, :-1]).norm(dim=-1)
    c = Hd[:, ::4]
    return d, (c[:, 1:] - c[:, :-1]).norm(dim=-1), (Hd[:, -1] - Hd[:, 0]).norm(dim=-1)


def _unit(H, mu):
    U = H.double() - mu
    return U / U.norm(dim=-1, keepdim=True).clamp_min(1e-30)


def _relerr(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-30)


# ------------------------------------------------------------------ context: network, probe, held-out, pairs

class Ctx:
    def __init__(self, gram_sites=()):
        import torch
        from common import ALL_SITES, load_net, normalize
        self.t0 = time.time()
        self.model, mean, std = load_net(NET)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.norm = lambda u8: normalize(torch.as_tensor(np.ascontiguousarray(u8), device='cuda').float() / 255, mean, std)
        pz = np.load(Path(R) / 'probes.npz')
        self.probe_x, self.probe_y, self.probe_idx = pz['c10_x'], pz['c10_y'], pz['c10_idx']
        self.test_x = np.load(Path(R) / 'data/c10/test_x.npy')
        self.test_y = np.load(Path(R) / 'data/c10/test_y.npy')
        # probe statistics at every site; Grams and unit rows at gram_sites
        acts = [[] for _ in ALL_SITES]
        with torch.no_grad():
            for at in range(0, len(self.probe_x), 250):
                x = self.norm(self.probe_x[at:at + 250])
                for dst, a in zip(acts, [x] + downstream(self.model, 0, x)):
                    dst.append(a.flatten(1).double())
        import xlib
        self.mu, self.rad_med, self.C, self.xn = {}, {}, {}, {}
        for i, parts in enumerate(acts):
            a = torch.cat(parts)
            parts.clear()
            C, mu, nrm, xn = xlib.probe_gram(a)
            del a
            self.mu[i], self.rad_med[i] = mu, float(nrm.median())
            if i in gram_sites:
                self.C[i], self.xn[i] = C.cpu().numpy(), xn
            del C, xn
        torch.cuda.empty_cache()
        # held-out: test images outside the probe that c10-s0 classifies correctly with margin > MARGIN_MIN
        logits = []
        with torch.no_grad():
            for at in range(0, len(self.test_x), 1000):
                logits.append(self.model(self.norm(self.test_x[at:at + 1000])).double().cpu().numpy())
        lg = np.concatenate(logits)
        top2 = np.sort(lg, 1)[:, -2:]
        ok = (lg.argmax(1) == self.test_y) & (top2[:, 1] - top2[:, 0] > MARGIN_MIN)
        ok[self.probe_idx] = False
        self.held = np.flatnonzero(ok)
        self.setup_s = time.time() - self.t0


def make_pairs(held, y, N):
    """N same-class and N different-class ordered pairs of distinct held-out images; prefix-stable in N."""
    out = []
    for kind in (0, 1):
        rng = np.random.default_rng(PAIR_SEED + kind)
        seen, P = set(), []
        while len(P) < N:
            i = int(rng.choice(held))
            pool = held[(y[held] == y[i]) if kind == 0 else (y[held] != y[i])]
            j = int(rng.choice(pool))
            key = (min(i, j), max(i, j))
            if i == j or key in seen:
                continue
            seen.add(key)
            P.append((i, j))
        out.append(np.array(P, dtype=np.int64))
    return out


def get_pairs(ctx, odir, n):
    """(pair_idx (2n, 2), kind (2n,), source).  pairs.npz in odir is authoritative once it exists."""
    f = odir / 'pairs.npz'
    if f.exists():
        z = np.load(f)
        N = int(z['N'])
        if N >= n:
            pi = z['pair_idx']
            same, diff = pi[:N][:n], pi[N:][:n]
            src = 'file'
            mine = make_pairs(ctx.held, ctx.test_y, n)
            if not (np.array_equal(mine[0], same) and np.array_equal(mine[1], diff)):
                src = 'file (differs from a fresh derivation)'
            return np.concatenate([same, diff]), np.repeat([0, 1], n), src
    N = max(n, N_PAIRS)
    same, diff = make_pairs(ctx.held, ctx.test_y, N)
    pi = np.concatenate([same, diff])
    tmp = odir / f'.pairs.{os.getpid()}.{time.time_ns()}.npz'
    with open(tmp, 'wb') as fh:
        np.savez(fh, pair_idx=pi, kind=np.repeat([0, 1], N), labels=ctx.test_y[pi], N=N, held=ctx.held,
                 seed=PAIR_SEED)
    os.replace(tmp, f)
    return np.concatenate([same[:n], diff[:n]]), np.repeat([0, 1], n), 'derived'


def site_acts(ctx, u8, i0, bs=500):
    """Activations (float32, unflattened) at site i0 for uint8 images."""
    import torch
    out = []
    with torch.no_grad():
        for at in range(0, len(u8), bs):
            x = ctx.norm(u8[at:at + bs])
            out.append(x if i0 == 0 else downstream(ctx.model, 0, x, i0)[-1])
    return torch.cat(out)


def _save(path, **arrs):
    tmp = path.with_name(f'.{path.name}.{os.getpid()}.tmp')
    with open(tmp, 'wb') as fh:
        np.savez(fh, **arrs)
    os.replace(tmp, path)


# ------------------------------------------------------------------ paths

def trace(ctx, i0, A, B, T, budget=4096, extras=True):
    """Push the T-step paths between rows of A and B (activations at i0) through the network.  Returns numpy dict:
    d/d4/chord for raw and unit at every downstream site, d_l0_unit, and (extras) ReLU patterns and logits."""
    import torch
    model = ctx.model
    ms = list(range(i0 + 1, LOGITS + 1))
    taps = [m for m in ms if m <= B9]
    s = torch.arange(T + 1, device='cuda', dtype=torch.float32) / T
    pc = max(1, budget // (T + 1))
    acc = {}

    def put(k, v):
        acc.setdefault(k, []).append(v.float().cpu().numpy() if hasattr(v, 'cpu') else v)

    for at in range(0, len(A), pc):
        a, b = A[at:at + pc], B[at:at + pc]
        q = len(a)
        sh = (1, T + 1) + (1,) * (a.dim() - 1)
        X = ((1 - s).view(sh) * a[:, None] + s.view(sh) * b[:, None]).reshape((q * (T + 1),) + a.shape[1:])
        with torch.no_grad():
            acts = downstream(model, i0, X)
            d0, _, _ = _steps(_unit(X.reshape(q, T + 1, -1), ctx.mu[i0]))
            put('d_l0_unit', d0)
            per = {k: [] for k in ('d_raw', 'd4_raw', 'chord_raw', 'd_unit', 'd4_unit', 'chord_unit', 'from0', 'prev')}
            for m, h in zip(ms, acts):
                H = h.reshape(q, T + 1, -1)
                for tag, src in (('raw', H), ('unit', _unit(H, ctx.mu[m]))):
                    d, d4, ch = _steps(src)
                    per[f'd_{tag}'].append(d); per[f'd4_{tag}'].append(d4); per[f'chord_{tag}'].append(ch)
                if extras and m in taps:
                    pat = H > 0
                    per['from0'].append((pat != pat[:, :1]).float().mean(-1))
                    per['prev'].append((pat[:, 1:] != pat[:, :-1]).float().mean(-1))
            for k, v in per.items():
                if v:
                    put(k, torch.stack(v, 1))
            if extras:
                put('logits', acts[-1].reshape(q, T + 1, -1))
        del acts, X
    return {k: np.concatenate(v) for k, v in acc.items()}


def run_paths(ctx, l0, odir, cfg):
    import torch
    import xlib
    from common import ALL_SITES
    t0 = time.time()
    i0 = ALL_SITES.index(l0)
    model = ctx.model
    n = int(cfg.get('n_pairs', N_PAIRS))
    meta = dict(l0=l0, gpu=torch.cuda.get_device_name(), n_pairs=n, setup_s=ctx.setup_s)
    pair_idx, kind, meta['pairs_source'] = get_pairs(ctx, odir, n)
    P = len(pair_idx)
    ms = list(range(i0 + 1, LOGITS + 1))
    taps = [m for m in ms if m <= B9]
    chain = list(range(max(i0, 1), LOGITS + 1))
    imgs = ctx.test_x[pair_idx]                                        # (P, 2, 3, 32, 32)
    A = site_acts(ctx, imgs[:, 0], i0)
    Bm = site_acts(ctx, imgs[:, 1], i0)
    out = dict(l0=np.array(l0), sites=np.array([ALL_SITES[m] for m in ms]), taps=np.array([ALL_SITES[m] for m in taps]),
               chain=np.array([ALL_SITES[c] for c in chain]), pair_idx=pair_idx, kind=kind,
               labels=ctx.test_y[pair_idx], images=imgs,
               s256=np.arange(T_HI + 1) / T_HI, s64=np.arange(T_LO + 1) / T_LO, s1024=np.arange(T_REF + 1) / T_REF,
               rad_med=np.array([ctx.rad_med[m] for m in ms]), rad_med_l0=np.array(ctx.rad_med[i0]))
    at_, bt_ = A.flatten(1).double() - ctx.mu[i0], Bm.flatten(1).double() - ctx.mu[i0]
    out['chord_l0'] = (Bm - A).flatten(1).double().norm(dim=1).cpu().numpy()
    out['cos_l0'] = ((at_ * bt_).sum(1) / (at_.norm(dim=1) * bt_.norm(dim=1))).cpu().numpy()

    # ---- 1, 3, 5: T = 256 step norms, ReLU patterns, outputs
    t1 = time.time()
    tr = trace(ctx, i0, A, Bm, T_HI)
    for tag in ('raw', 'unit'):
        out[f'd_{tag}'], out[f'd64_{tag}'], out[f'chord_{tag}'] = tr[f'd_{tag}'], tr[f'd4_{tag}'], tr[f'chord_{tag}']
        st = path_stats(tr[f'd_{tag}'], tr[f'd4_{tag}'], tr[f'chord_{tag}'], T_HI // 16)
        for k, v in st.items():
            out[f'{k}_{tag}'] = v.astype(np.float32)
    out['d_l0_unit'] = tr['d_l0_unit']
    out['share_l0_unit'] = path_stats(tr['d_l0_unit'], tr['d_l0_unit'][:, :T_HI // 4], tr['d_l0_unit'].sum(1),
                                      T_HI // 16)['share'].astype(np.float32)
    if taps:
        out['relu_from0'], out['relu_prev'] = tr['from0'], tr['prev']
    else:
        out['relu_from0'], out['relu_prev'] = np.zeros((P, 0, T_HI + 1), np.float32), np.zeros((P, 0, T_HI), np.float32)
    lg = tr['logits']
    out['logits'] = lg
    pred = lg.argmax(-1)
    srt = np.sort(lg, -1)
    out['pred'], out['margin'] = pred.astype(np.int8), (srt[..., -1] - srt[..., -2]).astype(np.float32)
    ch = np.diff(pred, axis=1) != 0
    out['n_switch'] = ch.sum(1)
    out['s_switch'] = np.where(ch.any(1), (ch.argmax(1) + 0.5) / T_HI, np.nan)
    meta['end_pred_ok'] = float(np.mean((pred[:, 0] == out['labels'][:, 0]) & (pred[:, -1] == out['labels'][:, 1])))
    del tr
    meta['t256_s'] = time.time() - t1

    # ---- T = 1024 refinement
    t1 = time.time()
    nr = min(N_REF, n)
    ref_rows = np.r_[np.arange(nr), n + np.arange(nr)]
    rr = torch.as_tensor(ref_rows, device='cuda')
    tr = trace(ctx, i0, A[rr], Bm[rr], T_REF, extras=False)
    out['ref_rows'] = ref_rows
    for tag in ('raw', 'unit'):
        out[f'd1024_{tag}'] = tr[f'd_{tag}']
        st = path_stats(tr[f'd_{tag}'], tr[f'd4_{tag}'], tr[f'chord_{tag}'], T_REF // 16)
        for k in ('share', 'peak', 'sigma', 'S', 'width', 'tau'):
            out[f'{k}1024_{tag}'] = st[k].astype(np.float32)
    del tr
    meta['t1024_s'] = time.time() - t1

    # ---- suffix operator on the clean probe
    t1 = time.time()
    Cs = [ctx.C[c] for c in chain]
    kers = xlib.site_kernels(Cs, KERNEL)
    eigs = xlib.site_spectra(kers)
    op = xlib.fit_op(kers, eigs, list(range(len(chain))), K=32, R=16)
    meta['sinkhorn_err'] = max(float(k['err']) for k in kers)
    del kers, eigs
    ext = xlib.Ext(op, labels=ctx.probe_y)
    meta['op_s'] = time.time() - t1
    L = len(chain)

    def cross_of(acts_by_site):
        return [xlib.cross_cos(acts_by_site[c].flatten(1).double(), ctx.mu[c], ctx.xn[c])[0] for c in chain]

    # ---- 2, 4, 6 on the T = 64 grid
    t1 = time.time()
    s = torch.arange(T_LO + 1, device='cuda', dtype=torch.float32) / T_LO
    pc = int(cfg.get('grid_paths', 16))
    targets = [LOGITS] + ([B9] if i0 < B9 else [])
    G = len(targets)
    acc = {}

    def put(k, v):
        acc.setdefault(k, []).append(v.float().cpu().numpy() if hasattr(v, 'cpu') else np.asarray(v))

    f_all = lambda h: tuple(downstream(model, i0, h))
    for ci, at in enumerate(range(0, P, pc)):
        a, b = A[at:at + pc], Bm[at:at + pc]
        q = len(a)
        sh = (1, T_LO + 1) + (1,) * (a.dim() - 1)
        X = ((1 - s).view(sh) * a[:, None] + s.view(sh) * b[:, None]).reshape((q * (T_LO + 1),) + a.shape[1:])
        V = (b - a)[:, None].expand((q, T_LO + 1) + a.shape[1:]).reshape(X.shape).contiguous()
        prim, tang = torch.func.jvp(f_all, (X,), (V,))
        sp, spu, ro = [], [], []
        for m, y, z in zip(ms, prim, tang):
            yt = y.flatten(1).double() - ctx.mu[m]
            rho = yt.norm(dim=1)
            uh = yt / rho[:, None]
            zf = z.flatten(1).double()
            sp.append(zf.norm(dim=1))
            spu.append((zf - uh * (uh * zf).sum(1, keepdim=True)).norm(dim=1) / rho)
            ro.append(rho / ctx.rad_med[m])
        rs = lambda lst: torch.stack(lst, 1).reshape(q, T_LO + 1, -1).permute(0, 2, 1)
        put('speed', rs(sp)); put('speed_unit', rs(spu)); put('rad_out', rs(ro))
        jv_logits = tang[-1].double()
        # embedding (chain acts: the path itself at l0 when l0 is in the chain)
        by_site = {m: y for m, y in zip(ms, prim)}
        by_site[i0] = X
        e = ext.embed(cross_of(by_site))
        for k in ('y', 'site_y'):
            put(f'emb_{k}', e[k][..., :K_KEEP].reshape(L, q, T_LO + 1, K_KEEP))
        put('emb_mass', e['mass'].reshape(L, q, T_LO + 1, -1))
        for k in ('cmax', 'perp'):
            put(f'emb_{k}', e[k].reshape(L, q, T_LO + 1))
        del prim, tang, by_site
        # gains
        g_raw, g_ang, g_cr, g_ca, g_ro, g_pr, g_pa = [], [], [], [], [], [], []
        ge = gain_exact(model, i0, X, ctx.mu[i0], ctx.mu[LOGITS], v=V)
        put('jvp_vjp_logits', ((ge['Jv'] - jv_logits).norm(dim=1) / jv_logits.norm(dim=1).clamp_min(1e-30)))
        r_in = ge['r'] / ctx.rad_med[i0]
        put('rad_in', r_in.reshape(q, -1))
        put('align_raw', ge['align_raw'].reshape(q, -1)); put('align_ang', ge['align_ang'].reshape(q, -1))
        nan = torch.full_like(ge['raw'], float('nan'))
        g_raw.append(ge['raw']); g_ang.append(ge['ang']); g_cr.append(nan); g_ca.append(nan)
        g_pr.append(nan); g_pa.append(nan)
        g_ro.append(ge['rho'] / ctx.rad_med[LOGITS])
        del ge
        if B9 in targets:
            pr_ = gain_power(model, i0, B9, X, ctx.mu[i0], ctx.mu[B9], False, seed=1000 + ci)
            pa_ = gain_power(model, i0, B9, X, ctx.mu[i0], ctx.mu[B9], True, seed=2000 + ci)
            g_raw.append(pr_['log']); g_ang.append(pa_['log']); g_cr.append(pr_['conv']); g_ca.append(pa_['conv'])
            g_pr.append(pr_['log_pow']); g_pa.append(pa_['log_pow']); g_ro.append(pr_['rho'] / ctx.rad_med[B9])
            jv9 = sp[B9 - i0 - 1]                                   # exact speed at b9 from the JVP above
            put('align_raw_b9', (jv9 / (pr_['log'].exp() * V.flatten(1).double().norm(dim=1))).reshape(q, -1))
            del pr_, pa_
        else:
            put('align_raw_b9', torch.full((q, T_LO + 1), float('nan')))
        rs2 = lambda lst: torch.stack(lst, 1).reshape(q, T_LO + 1, G).permute(0, 2, 1)
        put('gain_raw', rs2(g_raw)); put('gain_ang', rs2(g_ang))
        put('gain_conv_raw', rs2(g_cr)); put('gain_conv_ang', rs2(g_ca))
        put('gain_pow_raw', rs2(g_pr)); put('gain_pow_ang', rs2(g_pa))
        gro = rs2(g_ro)
        put('gain_rad_out', gro)
        put('small_flag', ((r_in.reshape(q, 1, -1) < SMALL_RAD) | (gro < SMALL_RAD)).cpu().numpy())
        del X, V
    for k, v in acc.items():
        out[k] = np.concatenate(v, 1 if k.startswith('emb_') else 0)
    out['small_flag'] = out['small_flag'].astype(bool)
    out['gain_targets'] = np.array(['logits', 'b9'][:G])
    meta['jvp_vs_vjp_logits_max_relerr'] = float(out.pop('jvp_vjp_logits').max())
    meta['grid_s'] = time.time() - t1

    # JVP speed vs finite differences of the T = 256 and T = 1024 paths
    fd = np.concatenate([out['d_raw'][:, :, ::4], out['d_raw'][:, :, -1:]], 2) * T_HI
    e = _relerr(out['speed'], fd)
    out['jvp_fd_med'], out['jvp_fd_p99'] = np.median(e, (0, 2)), np.quantile(e, 0.99, axis=(0, 2))
    out['jvp_fd_frac'] = (e > 1e-2).mean((0, 2))
    fd = np.concatenate([out['d1024_raw'][:, :, ::16], out['d1024_raw'][:, :, -1:]], 2) * T_REF
    e = _relerr(out['speed'][ref_rows], fd)
    out['jvp_fd1024_med'], out['jvp_fd1024_p99'] = np.median(e, (0, 2)), np.quantile(e, 0.99, axis=(0, 2))
    out['jvp_fd1024_frac'] = (e > 1e-2).mean((0, 2))
    for k in ('jvp_fd_med', 'jvp_fd_p99', 'jvp_fd_frac', 'jvp_fd1024_med', 'jvp_fd1024_p99', 'jvp_fd1024_frac'):
        meta[k] = dict(zip(out['sites'].tolist(), np.round(out[k], 6).tolist()))

    # ---- embedding checks: endpoints as plain images; masked replay; reference support
    t1 = time.time()
    flat = imgs.reshape((2 * P,) + imgs.shape[2:])                  # a0, b0, a1, b1, ...
    ys = []
    with torch.no_grad():
        for at in range(0, len(flat), 500):
            x = ctx.norm(flat[at:at + 500])
            acts = dict(zip(range(1, LOGITS + 1), downstream(model, 0, x)))
            ys.append(ext.embed(cross_of(acts))['y'][..., :K_KEEP])
    y_img = np.concatenate(ys, 1).reshape(L, P, 2, K_KEEP)
    ends = out['emb_y'][:, :, [0, T_LO]]
    out['endpoint_err'] = np.array(float(np.abs(ends - y_img).max()))
    scale = float(np.abs(op['vecs'][..., :K_KEEP]).max())
    meta['endpoint_err'], meta['emb_scale'] = float(out['endpoint_err']), scale
    masked = []
    for c in chain:
        Cm = torch.as_tensor(ctx.C[c], device='cuda', dtype=torch.float64)
        Cm.fill_diagonal_(-float('inf'))
        masked.append(Cm)
    rep = ext.embed(masked)
    del masked
    out['replay_err'] = np.array(float(np.abs(rep['y'] - op['vecs']).max()))
    meta['replay_err'] = float(out['replay_err'])
    meta['replay_site_err'] = float(np.abs(rep['site_y'] - op['site_vec']).max())
    meta['ext_cond_max'] = float(ext.cond.max())
    rng = np.random.default_rng(PAIR_SEED + 2)
    cand = np.setdiff1d(ctx.held, pair_idx.ravel())
    ref = np.sort(rng.choice(cand, min(N_REFCMAX, len(cand)), replace=False))
    cm = []
    with torch.no_grad():
        for at in range(0, len(ref), 500):
            x = ctx.norm(ctx.test_x[ref[at:at + 500]])
            acts = dict(zip(range(1, LOGITS + 1), downstream(model, 0, x)))
            cm.append(torch.stack([c.max(1).values for c in cross_of(acts)]).cpu().numpy())
    out['ref_cmax'] = np.concatenate(cm, 1).astype(np.float32)
    out['ref_idx'] = ref
    out.update(probe_y=op['vecs'][..., :K_KEEP].astype(np.float32), probe_site_y=op['site_vec'][..., :K_KEEP],
               probe_labels=ctx.probe_y, probe_idx=ctx.probe_idx, op_vals=op['vals'], op_c=op['c'],
               op_t=np.array(op['t']), op_site_lam=op['site_lam'], ext_cond=ext.cond, ext_smin=ext.smin)
    meta['emb_check_s'] = time.time() - t1
    assert float(out['replay_err']) < 1e-4 * max(1.0, meta['ext_cond_max']), meta
    assert meta['endpoint_err'] < 1e-2 * scale, meta

    # ---- summary numbers for the log
    rawc = {ALL_SITES[m]: [float(np.median(out['share_raw'][kind == k, j])) for k in (0, 1)] for j, m in enumerate(ms)}
    meta['median_share_raw_same_diff'] = rawc
    meta['median_sigma_raw'] = dict(zip(out['sites'].tolist(), np.round(np.median(out['sigma_raw'], 0), 4).tolist()))
    meta['gain_finite'] = bool(np.isfinite(out['gain_raw']).all() and np.isfinite(out['gain_ang']).all())
    meta['max_gain_conv_raw'] = float(np.nanmax(out['gain_conv_raw'])) if G > 1 else None
    meta['max_gain_conv_ang'] = float(np.nanmax(out['gain_conv_ang'])) if G > 1 else None
    if G > 1:
        meta['q99_gain_conv'] = [float(np.quantile(out[f'gain_conv_{t}'][:, 1], 0.99)) for t in ('raw', 'ang')]
        meta['median_ritz_minus_pow'] = [float(np.median(out[f'gain_{t}'][:, 1] - out[f'gain_pow_{t}'][:, 1]))
                                         for t in ('raw', 'ang')]
    meta['frac_small_flag'] = float(out['small_flag'].mean())
    meta['seconds'] = time.time() - t0 + ctx.setup_s
    out['meta'] = np.array(json.dumps(meta))
    return out, meta


# ------------------------------------------------------------------ probe gain

def run_probe_gain(ctx, cfg):
    import torch
    from common import ALL_SITES
    t0 = time.time()
    model = ctx.model
    N = len(ctx.probe_x)
    S = len(GAIN_STARTS)
    gr, ga = np.full((S, 3, N), np.nan), np.full((S, 3, N), np.nan)
    cr, ca = np.full((S, 3, N), np.nan), np.full((S, 3, N), np.nan)
    pw = {False: np.full((S, 3, N), np.nan), True: np.full((S, 3, N), np.nan)}
    rin, rout = np.full((S, N), np.nan), np.full((S, 3, N), np.nan)
    nexts = []
    bs = int(cfg.get('gain_batch', 500))
    for si, l0 in enumerate(GAIN_STARTS):
        i0 = ALL_SITES.index(l0)
        nexts.append(ALL_SITES[i0 + 1])
        for at in range(0, N, bs):
            sl = slice(at, at + bs)
            X = site_acts(ctx, ctx.probe_x[sl], i0)
            ge = gain_exact(model, i0, X, ctx.mu[i0], ctx.mu[LOGITS])
            gr[si, 0, sl], ga[si, 0, sl] = ge['raw'].cpu().numpy(), ge['ang'].cpu().numpy()
            rin[si, sl] = (ge['r'] / ctx.rad_med[i0]).cpu().numpy()
            rout[si, 0, sl] = (ge['rho'] / ctx.rad_med[LOGITS]).cpu().numpy()
            del ge
            for ti, i1 in ((1, B9), (2, i0 + 1)):
                for ang, g_, c_ in ((False, gr, cr), (True, ga, ca)):
                    gp = gain_power(model, i0, i1, X, ctx.mu[i0], ctx.mu[i1], ang,
                                    seed=10_000 * si + 100 * ti + at // bs + (50 if ang else 0))
                    g_[si, ti, sl], c_[si, ti, sl] = gp['log'].cpu().numpy(), gp['conv'].cpu().numpy()
                    pw[ang][si, ti, sl] = gp['log_pow'].cpu().numpy()
                    rout[si, ti, sl] = (gp['rho'] / ctx.rad_med[i1]).cpu().numpy()
                    del gp
    with torch.no_grad():
        lg = torch.cat([model(ctx.norm(ctx.probe_x[at:at + 500])) for at in range(0, N, 500)]).double().cpu().numpy()
    srt = np.sort(lg, 1)
    meta = dict(gpu=torch.cuda.get_device_name(), setup_s=ctx.setup_s, seconds=time.time() - t0 + ctx.setup_s,
                finite=bool(np.isfinite(gr).all() and np.isfinite(ga).all()),
                max_conv_raw=float(np.nanmax(cr)), max_conv_ang=float(np.nanmax(ca)),
                q99_conv_raw=float(np.nanquantile(cr, 0.99)), q99_conv_ang=float(np.nanquantile(ca, 0.99)),
                max_ritz_minus_pow_raw=float(np.nanmax(gr - pw[False])), max_ritz_minus_pow_ang=float(np.nanmax(ga - pw[True])),
                median_ritz_minus_pow_raw=float(np.nanmedian(gr[:, 1:] - pw[False][:, 1:])),
                median_gain_raw={f'{l0}->{t}': float(np.median(gr[si, ti]))
                                 for si, l0 in enumerate(GAIN_STARTS) for ti, t in enumerate(['logits', 'b9', 'next'])},
                median_gain_ang={f'{l0}->{t}': float(np.median(ga[si, ti]))
                                 for si, l0 in enumerate(GAIN_STARTS) for ti, t in enumerate(['logits', 'b9', 'next'])})
    out = dict(starts=np.array(GAIN_STARTS), targets=np.array(['logits', 'b9', 'next']), next_sites=np.array(nexts),
               gain_raw=gr.astype(np.float32), gain_ang=ga.astype(np.float32), conv_raw=cr.astype(np.float32),
               conv_ang=ca.astype(np.float32), pow_raw=pw[False].astype(np.float32),
               pow_ang=pw[True].astype(np.float32), rad_in=rin.astype(np.float32), rad_out=rout.astype(np.float32),
               logits=lg.astype(np.float32), pred=lg.argmax(1), margin=(srt[:, -1] - srt[:, -2]).astype(np.float32),
               labels=ctx.probe_y, probe_idx=ctx.probe_idx, meta=np.array(json.dumps(meta)))
    return out, meta


# ------------------------------------------------------------------ checks

def run_checks(ctx, cfg):
    import copy
    import torch
    from common import ALL_SITES, suffix_all
    t0 = time.time()
    model = ctx.model
    res = dict(gpu=torch.cuda.get_device_name())
    x8 = ctx.probe_x[:8]

    # (a) my site composition equals common.suffix_all
    dmax = 0.0
    with torch.no_grad():
        for l0 in STARTS + ['pool']:
            i0 = ALL_SITES.index(l0)
            h = site_acts(ctx, x8, i0)
            for u, v in zip(downstream(model, i0, h), suffix_all(model, l0, h)):
                dmax = max(dmax, float((u - v).abs().max()))
    res['suffix_vs_common_maxabs'] = dmax
    assert dmax < 1e-4, res

    W = model.fc.weight.double()
    sW = float(torch.linalg.svdvals(W)[0])
    res['log_sigma_W'] = float(np.log(sW))
    # (b) pool -> logits exact = sigma_max(W)
    i_pool = ALL_SITES.index('pool')
    ge = gain_exact(model, i_pool, site_acts(ctx, x8, i_pool), ctx.mu[i_pool], ctx.mu[LOGITS])
    res['pool_logits_exact_err'] = float((ge['raw'] - np.log(sW)).abs().max())
    assert res['pool_logits_exact_err'] < 1e-5, res
    # (c) b9 -> logits exact = sigma_max(W)/8; power iteration on the same
    X9 = site_acts(ctx, x8, B9)
    ge = gain_exact(model, B9, X9, ctx.mu[B9], ctx.mu[LOGITS])
    res['b9_logits_exact_err'] = float((ge['raw'] - np.log(sW / 8)).abs().max())
    assert res['b9_logits_exact_err'] < 1e-5, res
    rel = lambda a, b: float(((a - b).exp() - 1).abs().max())
    gp = gain_power(model, B9, LOGITS, X9, ctx.mu[B9], ctx.mu[LOGITS], False)
    gpa = gain_power(model, B9, LOGITS, X9, ctx.mu[B9], ctx.mu[LOGITS], True)
    res['b9_logits_vs_exact_relerr'] = dict(ritz_raw=rel(gp['log'], ge['raw']), pow_raw=rel(gp['log_pow'], ge['raw']),
                                            ritz_ang=rel(gpa['log'], ge['ang']), pow_ang=rel(gpa['log_pow'], ge['ang']))

    def explicit(i0, i1, x):
        """Explicit Jacobian (one image) by a loop of VJPs; raw and angular sigma_max by SVD in float64."""
        f = seg(model, i0, i1)
        J = torch.autograd.functional.jacobian(f, x[None]).reshape(-1, x.numel()).double()
        xt = x.flatten().double() - ctx.mu[i0]
        r = xt.norm()
        xh = xt / r
        y = f(x[None]).flatten().double() - ctx.mu[i1]
        rho = y.norm()
        uh = y / rho
        D = (torch.eye(len(y), dtype=torch.float64, device='cuda') - uh[:, None] * uh[None]) / rho
        Pm = torch.eye(len(xt), dtype=torch.float64, device='cuda') - xh[:, None] * xh[None]
        return float(torch.linalg.svdvals(J)[0].log()), float(torch.linalg.svdvals(r * D @ J @ Pm)[0].log())

    # (d) b8 -> b9 power iteration against explicit SVD
    i8 = ALL_SITES.index('b8')
    X8 = site_acts(ctx, ctx.probe_x[:4], i8)
    ex = [explicit(i8, B9, X8[k]) for k in range(len(X8))]
    gp = gain_power(model, i8, B9, X8, ctx.mu[i8], ctx.mu[B9], False)
    gpa = gain_power(model, i8, B9, X8, ctx.mu[i8], ctx.mu[B9], True)
    b8 = dict(svd_raw=[e[0] for e in ex], svd_ang=[e[1] for e in ex])
    for tag, g, j in (('raw', gp, 0), ('ang', gpa, 1)):
        for est in ('log', 'log_pow'):
            b8[f'relerr_{"ritz" if est == "log" else "pow"}_{tag}'] = [float(np.exp(p - e[j]) - 1)
                                                                     for p, e in zip(g[est].tolist(), ex)]
        b8[f'conv_ritz_{tag}'], b8[f'conv_pow_{tag}'] = g['conv'].tolist(), g['conv_pow'].tolist()
    res['b8_b9'] = b8
    # both estimates are lower bounds (up to float32 noise)
    assert max(max(v) for k, v in b8.items() if k.startswith('relerr')) < 1e-4, res

    # (e) b6 -> logits: exact method vs explicit SVD (asserted), power iteration vs exact (recorded)
    i6 = ALL_SITES.index('b6')
    X6 = site_acts(ctx, ctx.probe_x[:4], i6)
    ge = gain_exact(model, i6, X6, ctx.mu[i6], ctx.mu[LOGITS])
    ex = [explicit(i6, LOGITS, X6[k]) for k in range(len(X6))]
    gp = gain_power(model, i6, LOGITS, X6, ctx.mu[i6], ctx.mu[LOGITS], False)
    gpa = gain_power(model, i6, LOGITS, X6, ctx.mu[i6], ctx.mu[LOGITS], True)
    res['b6_logits'] = dict(exact_raw=ge['raw'].tolist(), svd_raw=[e[0] for e in ex], ritz_raw=gp['log'].tolist(),
                            pow_raw=gp['log_pow'].tolist(), exact_ang=ge['ang'].tolist(), svd_ang=[e[1] for e in ex],
                            ritz_ang=gpa['log'].tolist(), pow_ang=gpa['log_pow'].tolist())
    e_exact = max(max(abs(a - e[0]), abs(b - e[1])) for a, b, e in zip(ge['raw'].tolist(), ge['ang'].tolist(), ex))
    res['b6_logits_exact_vs_svd_maxabs_log'] = e_exact
    res['b6_logits_vs_exact_maxabs_log'] = dict(
        ritz=float(max((gp['log'] - ge['raw']).abs().max(), (gpa['log'] - ge['ang']).abs().max())),
        pow=float(max((gp['log_pow'] - ge['raw']).abs().max(), (gpa['log_pow'] - ge['ang']).abs().max())))
    assert e_exact < 1e-4, res

    # (f) a b9-start path is affine at pool and logits: window share 1/16 and sigma 1 (raw)
    pair_idx = make_pairs(ctx.held, ctx.test_y, 4)
    pair_idx = np.concatenate(pair_idx)
    imgs = ctx.test_x[pair_idx]
    A, Bm = site_acts(ctx, imgs[:, 0], B9), site_acts(ctx, imgs[:, 1], B9)
    tr = trace(ctx, B9, A, Bm, T_HI, extras=False)
    st = path_stats(tr['d_raw'], tr['d4_raw'], tr['chord_raw'], T_HI // 16)
    stu = path_stats(tr['d_unit'], tr['d4_unit'], tr['chord_unit'], T_HI // 16)
    res['b9_affine'] = dict(share_x16_minus1_maxabs=float(np.abs(st['share'] * 16 - 1).max()),
                            sigma_minus1_maxabs=float(np.abs(st['sigma'] - 1).max()),
                            ev_maxabs=float(np.abs(st['ev']).max()),
                            unit_share_x16_range=[float(stu['share'].min() * 16), float(stu['share'].max() * 16)])
    assert res['b9_affine']['share_x16_minus1_maxabs'] < 1e-3 and res['b9_affine']['sigma_minus1_maxabs'] < 1e-3, res

    # (g) forward-mode speed (float32) against a float64 central difference, pixels start
    m64 = copy.deepcopy(model).double()
    A0, B0 = site_acts(ctx, imgs[:2, 0], 0), site_acts(ctx, imgs[:2, 1], 0)
    sv = torch.tensor([0.0, 0.25, 0.5, 0.77, 1.0], device='cuda')
    X = ((1 - sv)[None, :, None, None, None] * A0[:, None] + sv[None, :, None, None, None] * B0[:, None]).reshape(-1, 3, 32, 32)
    V = (B0 - A0)[:, None].expand(2, len(sv), 3, 32, 32).reshape(X.shape).contiguous()
    prim, tang = torch.func.jvp(lambda h: tuple(downstream(model, 0, h)), (X,), (V,))
    prim64, tang64 = torch.func.jvp(lambda h: tuple(downstream(m64, 0, h)), (X.double(),), (V.double(),))
    # A pre-activation can sit within ~1e-6 of a kink at a grid point (seen at a real endpoint image in block b7),
    # so the reference is the better of the central / forward / backward difference at a tiny step.
    h = 1e-9
    nrm = lambda t: t.flatten(1).double().norm(dim=1).cpu().numpy()
    with torch.no_grad():
        up = downstream(m64, 0, X.double() + h * V.double())
        dn = downstream(m64, 0, X.double() - h * V.double())
    e64, e32, e32med, flips = {}, {}, {}, {}
    for k, (z, z64, u, d, p64) in enumerate(zip(tang, tang64, up, dn, prim64)):
        site = ALL_SITES[k + 1]
        n64 = nrm(z64)
        fds = [nrm((u - d) / (2 * h)), nrm((u - p64) / h), nrm((p64 - d) / h)]
        e64[site] = float(np.min([_relerr(n64, fd) for fd in fds], 0).max())    # implementation: float64 JVP vs FD
        e = _relerr(nrm(z), n64)                                                  # float32 vs float64 JVP
        e32[site], e32med[site] = float(e.max()), float(np.median(e))
        if k + 1 <= B9:
            flips[site] = int(((prim[k] > 0) != (p64 > 0)).sum())
    res['jvp64_vs_fd64_max_relerr'] = e64
    res['jvp32_vs_jvp64_max_relerr'] = e32
    res['jvp32_vs_jvp64_median_relerr'] = e32med
    res['relu_tap_flips_f32_vs_f64'] = flips
    assert max(e64.values()) < 1e-5, res
    res['seconds'] = time.time() - t0 + ctx.setup_s
    return res


# ------------------------------------------------------------------ entry

def run(unit: dict, cfg: dict) -> dict:
    from common import ALL_SITES
    odir = Path(R) / cfg.get('out', 'x4')
    odir.mkdir(parents=True, exist_ok=True)
    part = unit['part']
    if part == 'paths':
        l0 = unit['l0']
        assert l0 in STARTS, l0
        target = odir / f'paths_{l0}.npz'
    elif part == 'probe_gain':
        target = odir / 'probe_gain.npz'
    elif part == 'checks':
        target = odir / 'checks.json'
    else:
        raise ValueError(part)
    if target.exists() and not cfg.get('force'):
        return dict(unit=unit, status='cached')
    if part == 'paths':
        i0 = ALL_SITES.index(unit['l0'])
        ctx = Ctx(gram_sites=set(range(max(i0, 1), LOGITS + 1)))
        out, meta = run_paths(ctx, unit['l0'], odir, cfg)
        _save(target, **out)
    elif part == 'probe_gain':
        ctx = Ctx()
        out, meta = run_probe_gain(ctx, cfg)
        _save(target, **out)
    else:
        ctx = Ctx()
        meta = run_checks(ctx, cfg)
        tmp = target.with_name(f'.{target.name}.{os.getpid()}.tmp')
        tmp.write_text(json.dumps(meta, indent=1))
        os.replace(tmp, target)
    return dict(unit=unit, status='done', **meta)
