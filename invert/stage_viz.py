"""Decoded images: the image as a function on the probe, decoded from K coordinates at a site (13-site operator).

For held-out images, their PGD versions, and two other datasets: kernel-ridge decode of the pixels from the
coordinates at each chosen site; and decodes along straight paths between pairs of held-out images' coordinates.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

import stage_decode as sd

DT = torch.float64


def run(unit: dict, cfg: dict, mi):
    R = Path(mi.R); dev = 'cuda'
    pset, kname = unit['pset'], unit['kname']
    tag = f'{pset}__{kname}__L13'
    n = unit.get('n', 32)
    sp = dict(np.load(R / 'splits.npz'))
    idx = mi.index_set(sp, pset)
    z = np.load(R / 'joint' / tag / 'modes.npz')
    sites = [str(s) for s in z['sites']]
    g = R / 'gram' / mi.centre_set(pset)
    Ctt = torch.tensor(np.asarray(np.load(g / 'pixels.npy', mmap_mode='r')[idx][:, idx]), dtype=DT, device=dev)
    nz = np.load(g / 'pixels_norms.npz')
    mu = torch.tensor(nz['mean'], device=dev); nt = torch.tensor(nz['c10'][idx], dtype=DT, device=dev)
    Xc = torch.tensor(np.asarray(np.load(R / 'acts/c10/pixels.npy', mmap_mode='r')[idx]), device=dev) - mu
    mean = torch.tensor(mi.MEAN, device=dev).view(1, 3, 1, 1); std = torch.tensor(mi.STD, device=dev).view(1, 3, 1, 1)

    def to_img(x):
        im = (x.reshape(-1, 3, 32, 32) * std + mean).clamp(0, 1)
        return (im * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()

    sets = {t: n for t in ('H', 'pgd8', 'pgd2', 'c100', 'svhn', 'noise20')}
    have = [t for t in sets if (R / 'joint' / tag / f'coords_{t}.npz').exists()]
    coords = {t: np.load(R / 'joint' / tag / f'coords_{t}.npz')['y'][:, :sets[t]] for t in have}
    srcs = {t: np.load(R / 'joint' / tag / f'coords_{t}.npz')['onestep'][:, :sets[t]] for t in have}   # source coordinates
    out = {}
    hid = mi.index_set(sp, 'H')[:n]
    nhwc = lambda a: a.transpose(0, 2, 3, 1) if a.shape[1] == 3 else a       # stored originals are NCHW
    out['orig|H'] = nhwc(np.load(R / 'acts/c10/meta.npz')['images'][hid])
    out['label|H'] = sp['labels'][hid]
    for t in coords:
        if t != 'H':
            m = np.load(R / 'acts' / t / 'meta.npz')
            out[f'orig|{t}'] = nhwc(m['images'][:n]); out[f'pred|{t}'] = m['logits'][:n].argmax(1)
    out['pred|H'] = np.load(R / 'acts/c10/logits.npy')[hid].argmax(1)
    for site in unit.get('sites', ['pixels', 'stem', 'b3', 'b6', 'b9', 'pool']):
        li = sites.index(site)
        for K in unit.get('Ks', [16, 64]):
            zt = torch.tensor(z['vecs'][li][:, :K], dtype=DT, device=dev)
            sel = sd.select(zt, Ctt, nt)
            gk = sd.Gauss(zt, sel['krr']['h'] * sd.knn_scale(zt))
            dec = lambda y: to_img(mu + gk.w_krr(y, sel['krr']['ridge']).float() @ Xc)
            for t, c in coords.items():
                out[f'dec|{site}|{K}|{t}'] = dec(torch.tensor(c[li][:, :K], dtype=DT, device=dev))
            yh = torch.tensor(coords['H'][li][:, :K], dtype=DT, device=dev)
            a, b = yh[:8], yh[8:16]
            out[f'path|{site}|{K}'] = np.stack([dec(a + tau * (b - a)) for tau in np.linspace(-0.5, 1.5, 9)], 1)
            del gk
            # the same from the site's own source coordinates s_l = p_l(x) . f_l (probe: P_l f_l)
            Pl = np.load(R / 'kernel' / f'{pset}__{kname}' / f'{site}.npz')['P']
            zt = torch.tensor(Pl @ z['vecs'][li][:, :K].astype(np.float64), dtype=DT, device=dev)
            del Pl
            sel = sd.select(zt, Ctt, nt)
            gk = sd.Gauss(zt, sel['krr']['h'] * sd.knn_scale(zt))
            dec = lambda y: to_img(mu + gk.w_krr(y, sel['krr']['ridge']).float() @ Xc)
            for t, c in srcs.items():
                out[f'decs|{site}|{K}|{t}'] = dec(torch.tensor(c[li][:, :K], dtype=DT, device=dev))
            yh = torch.tensor(srcs['H'][li][:, :K], dtype=DT, device=dev)
            a, b = yh[:8], yh[8:16]
            out[f'paths|{site}|{K}'] = np.stack([dec(a + tau * (b - a)) for tau in np.linspace(-0.5, 1.5, 9)], 1)
            del gk
    f = R / 'viz'; f.mkdir(exist_ok=True)
    np.savez_compressed(f / f'{tag}.npz', **out)
    return dict(tag=tag, keys=len(out))
