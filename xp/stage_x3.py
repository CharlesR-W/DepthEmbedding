"""X3 (xp/SPEC.md): adversarial images through depth.  One GPU unit; writes /results/<out>/ (default x3).

Networks: A = c10-s0 (attacked), B = c10-s1 (transfer).  Probe: c10 (2,000 images).  S = the first 1,000 CIFAR-10 test
images outside the probe that both networks classify correctly.

Image sets (all pushed through both networks):
  clean                 S
  u1 u2 u4 u8           untargeted PGD on A, L-inf eps/255, 50 steps, step 2.5 eps / 50, random start
  t8                    targeted PGD on A, eps 8/255, target = (label + r) mod 10, r uniform in 1..9 (`target` in sets.npz)
  n1 n2 n4 n8           random sign noise at the same eps (measured baseline)
  p1..p7                clean + (k/8) * (u8 - clean): a straight line in pixel space from clean (p0) to u8 (p8)
  natA, natB            held-out test images that A (resp. B) misclassifies (all of them; not aligned with S)

Files:
  sets.npz      S (indices into the CIFAR-10 test set), labels, target, natA, natB (indices), natA_y, natB_y,
                img_clean, img_u8, img_t8 (1000, 3, 32, 32) uint8
  op_<net>_<kernel>.npz   the fitted operator on the probe (xlib.fit_op, full 12-site chain, K = 32) plus
                probe_labels, cond, smin (per-mode condition number and smallest singular value of the extension)
  <net>_<set>.npz, per network and image set (m images, 12 sites in common.SITES order):
      logits (m, 10)
      norm (12, m)      centred norm of the activation (centre = probe mean at the site)
      actcos (12, m)    cosine between the centred activations of this image and of its clean version  [aligned sets]
      relpert (12, m)   |a - a_clean| / |a_clean - mu|                                             [aligned sets]
      and for each kernel k in (ea100, knn15), prefix `<k>_`:
      y (12, m, 16)       joint coordinates (frozen-probe extension); uses the rows of every site
      site_y (12, m, 16)  site-local coordinates: Nystrom extension of that site's own walk modes
      mass (12, m, 10)    site-local class mass: the point's kernel-row weight on probe images of each class
      cmax (12, m)        largest cosine to a probe image;  perp (12, m) perplexity of the row
  check_logits_only.npz   clean S in A with only the logits row replaced by the u8 row (kernel ea100):
      dy (12, m) |y_mixed - y_clean| per site, dy_full (12, m) |y_u8 - y_clean|, local_max_change (must be 0)
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

KERNELS = ['ea100', 'knn15']
ALIGNED = ['clean'] + [f'{a}{e}' for a in 'un' for e in (1, 2, 4, 8)] + ['t8'] + [f'p{k}' for k in range(1, 8)]


def run(unit, cfg):
    import torch
    import torch.nn.functional as F
    import xlib
    from common import R, SITES, activations, load_net, normalize
    t0 = time.time()
    out = Path(R) / cfg.get('out', 'x3'); out.mkdir(parents=True, exist_ok=True)
    if (out / 'DONE').exists() and not cfg.get('force'):
        return 'cached'
    n_src = int(cfg.get('n_src', 1000))
    info = {}
    pz = np.load(Path(R) / 'probes.npz')
    tx = torch.from_numpy(np.load(Path(R) / 'data/c10/test_x.npy')).float() / 255
    ty = np.load(Path(R) / 'data/c10/test_y.npy')
    probe_idx, probe_y = pz['c10_idx'], pz['c10_y']
    assert (ty[probe_idx] == probe_y).all()
    nets = {k: load_net(k) for k in ('c10-s0', 'c10-s1')}

    def logits_of(net, x01):
        model, mean, std = nets[net]
        with torch.inference_mode():
            return torch.cat([model(normalize(x01[i:i + 1000].cuda(), mean, std)) for i in range(0, len(x01), 1000)]).cpu()

    pred = {k: logits_of(k, tx).argmax(1).numpy() for k in nets}
    info['clean_acc'] = {k: float((pred[k] == ty).mean()) for k in nets}
    held = np.setdiff1d(np.arange(len(ty)), probe_idx)
    both = held[(pred['c10-s0'][held] == ty[held]) & (pred['c10-s1'][held] == ty[held])]
    S = both[:n_src]
    natA, natB = held[pred['c10-s0'][held] != ty[held]], held[pred['c10-s1'][held] != ty[held]]
    xs, ys = tx[S].cuda(), torch.from_numpy(ty[S]).cuda()
    rng = np.random.RandomState(0)
    target = torch.from_numpy((ty[S] + rng.randint(1, 10, len(S))) % 10).cuda()

    model, mean, std = nets['c10-s0']

    def pgd(eps, targeted=False, steps=50, seed=0):
        g = torch.Generator(device='cuda').manual_seed(seed)
        adv = (xs + (torch.rand(xs.shape, device='cuda', generator=g) * 2 - 1) * eps).clamp(0, 1)
        for _ in range(steps):
            adv.requires_grad_(True)
            lg = model(normalize(adv, mean, std))
            loss = -F.cross_entropy(lg, target) if targeted else F.cross_entropy(lg, ys)
            gr, = torch.autograd.grad(loss, adv)
            adv = torch.min(torch.max(adv.detach() + 2.5 * eps / steps * gr.sign(), xs - eps), xs + eps).clamp(0, 1)
        return adv.detach()

    sets = {'clean': xs}
    for e in (1, 2, 4, 8):
        sets[f'u{e}'] = pgd(e / 255, seed=e)
        g = torch.Generator(device='cuda').manual_seed(100 + e)
        sets[f'n{e}'] = (xs + (torch.randint(0, 2, xs.shape, device='cuda', generator=g) * 2 - 1) * e / 255).clamp(0, 1)
    sets['t8'] = pgd(8 / 255, targeted=True, seed=50)
    for k in range(1, 8):
        sets[f'p{k}'] = (xs + k / 8 * (sets['u8'] - xs)).clamp(0, 1)
    assert float((sets['u8'] - xs).abs().max()) <= 8 / 255 + 1e-6
    sets['natA'], sets['natB'] = tx[natA].cuda(), tx[natB].cuda()
    u8 = lambda a: (a * 255).round().clamp(0, 255).byte().cpu().numpy()
    np.savez(out / 'sets.npz', S=S, labels=ty[S], target=target.cpu().numpy(), natA=natA, natB=natB,
             natA_y=ty[natA], natB_y=ty[natB], img_clean=u8(xs), img_u8=u8(sets['u8']), img_t8=u8(sets['t8']))
    info['attack_s'] = time.time() - t0

    def acts(net, x01):
        """List over SITES of (m, d) float64 cuda activations."""
        model, mean, std = nets[net]
        store = [[] for _ in SITES]
        with torch.inference_mode():
            for i in range(0, len(x01), 250):
                for dst, a in zip(store, activations(model, normalize(x01[i:i + 250], mean, std))[1:]):
                    dst.append(a.flatten(1).double())
        return [torch.cat(p) for p in store]

    xp = torch.from_numpy(pz['c10_x']).float().cuda() / 255
    for net in nets:
        t1 = time.time()
        pa = acts(net, xp)
        mus, xns, Cs, pnorm = [], [], [], []
        for a in pa:
            C, mu, nrm, xn = xlib.probe_gram(a)
            Cs.append(C.cpu().numpy()); mus.append(mu); xns.append(xn); pnorm.append(nrm.median().item())
        del pa
        exts = {}
        for kname in KERNELS:
            kers = xlib.site_kernels(Cs, kname)
            op = xlib.fit_op(kers, xlib.site_spectra(kers), list(range(12)), K=32, R=16)
            exts[kname] = xlib.Ext(op, labels=probe_y)
            np.savez(out / f'op_{net}_{kname}.npz', probe_labels=probe_y, cond=exts[kname].cond, smin=exts[kname].smin,
                     probe_norm_median=np.array(pnorm), **op)
            # masked replay of the probe: must return the modes
            masked = [torch.from_numpy(np.where(np.eye(len(C), dtype=bool), -np.inf, C)) for C in Cs]
            rep = exts[kname].embed(masked)
            info[f'{net}_{kname}_replay'] = float(np.abs(rep['y'] - op['vecs']).max())
            info[f'{net}_{kname}_cond_max'] = float(exts[kname].cond.max())
            del kers
        info[f'{net}_fit_s'] = time.time() - t1
        clean_acts, crosses = None, {}
        for name, x01 in sets.items():
            if name == 'natA' and net != 'c10-s0' or name == 'natB' and net != 'c10-s1':
                pass                                             # still embed: the other network's errors are a reference
            a = acts(net, x01)
            rec = dict(logits=a[-1].float().cpu().numpy())
            cross, norms = [], []
            for l in range(12):
                c, ny = xlib.cross_cos(a[l], mus[l], xns[l])
                cross.append(c); norms.append(ny.cpu().numpy())
            rec['norm'] = np.stack(norms)
            if name == 'clean':
                clean_acts = a
            if name in ALIGNED:
                ac, rp = [], []
                for l in range(12):
                    u, v = a[l] - mus[l], clean_acts[l] - mus[l]
                    ac.append(F.cosine_similarity(u, v, dim=1).cpu().numpy())
                    rp.append(((a[l] - clean_acts[l]).norm(dim=1) / v.norm(dim=1).clamp_min(1e-12)).cpu().numpy())
                rec['actcos'], rec['relpert'] = np.stack(ac), np.stack(rp)
            for kname, ext in exts.items():
                e = ext.embed(cross)
                for k, v in e.items():
                    rec[f'{kname}_{k}'] = v[..., :16] if k in ('y', 'site_y') else v
            if name in ('clean', 'u8') and net == 'c10-s0':
                crosses[name] = cross
            np.savez(out / f'{net}_{name}.npz', **rec)
            del a
            if name != 'clean':
                torch.cuda.empty_cache()
        if net == 'c10-s0':                                      # logits-only row change
            ext = exts['ea100']
            mixed = crosses['clean'][:11] + [crosses['u8'][11]]
            e0, e1, e8 = ext.embed(crosses['clean']), ext.embed(mixed), ext.embed(crosses['u8'])
            local = max(float(np.abs(e1[k][:11] - e0[k][:11]).max()) for k in ('site_y', 'mass', 'cmax', 'perp'))
            assert local == 0.0, local
            np.savez(out / 'check_logits_only.npz', dy=np.linalg.norm(e1['y'] - e0['y'], axis=2),
                     dy_full=np.linalg.norm(e8['y'] - e0['y'], axis=2), local_max_change=local)
        del clean_acts, crosses, mus, xns, exts
        torch.cuda.empty_cache()
        info[f'{net}_s'] = time.time() - t1
    for name in sets:
        lg = np.load(out / f'c10-s0_{name}.npz')['logits']
        if name in ALIGNED:
            info[f'acc_A_{name}'] = float((lg.argmax(1) == ty[S]).mean())
            info[f'acc_B_{name}'] = float((np.load(out / f'c10-s1_{name}.npz')['logits'].argmax(1) == ty[S]).mean())
    info['t8_target_hit_A'] = float((np.load(out / 'c10-s0_t8.npz')['logits'].argmax(1) == target.cpu().numpy()).mean())
    info['n_natA'], info['n_natB'] = int(len(natA)), int(len(natB))
    info['seconds'] = time.time() - t0
    (out / 'DONE').write_text(json.dumps(info))
    return info
