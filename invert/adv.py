"""Clean and PGD-attacked versions of the same held-out images, followed through depth in the joint embedding.

Per site (first K modes): displacement of the attacked image from its clean version relative to the spread
of held-out images; nearest-class-centroid label (centroids from the probe; labels used for this readout
only) for clean and attacked versions, compared with the true label and the network's attacked prediction.
Run: uv run --with numpy python invert/adv.py [tag] [K]
"""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / 'data/invert'
tag = sys.argv[1] if len(sys.argv) > 1 else 'P8__ea400__L12'
K = int(sys.argv[2]) if len(sys.argv) > 2 else 16
pset = tag.split('__')[0]
sp = np.load(ROOT / 'splits.npz')
z = np.load(ROOT / f'joint/{tag}/modes.npz')
sites = [str(s) for s in z['sites']]
V = z['vecs'][:, :, :K].astype(float)
lab_p = sp['labels'][sp[pset]]
H = sp['H'][:1000]; lab = sp['labels'][H]
yc = np.load(ROOT / f'joint/{tag}/coords_H.npz')['y'][:, :1000, :K].astype(float)
pred_clean = np.load(ROOT / 'meta/c10_logits.npy')[H].argmax(1)
out = {}
for adv in ('pgd2', 'pgd8'):
    ya = np.load(ROOT / f'joint/{tag}/coords_{adv}.npz')['y'][:, :, :K].astype(float)
    pa = np.load(ROOT / f'meta/{adv}.npz')['logits'].argmax(1)
    rows = []
    for l, s in enumerate(sites):
        cen = np.stack([V[l][lab_p == c].mean(0) for c in range(10)])
        sd = yc[l].std(0)
        disp = (((ya[l] - yc[l]) / sd) ** 2).mean()                  # 2 = as far as an unrelated image
        nc = lambda y: ((y[:, None, :] - cen[None]) ** 2).sum(2).argmin(1)
        cc, ca = nc(yc[l]), nc(ya[l])
        rows.append(dict(site=s, disp=float(disp), clean_true=float((cc == lab).mean()), adv_true=float((ca == lab).mean()),
                         adv_follows_attack=float((ca == pa).mean()), clean_matches_pred=float((cc == pred_clean).mean())))
    out[adv] = rows
    print(adv, 'net acc on attacked:', float((pa == lab).mean()))
    print('  site    ' + ' '.join(f'{r["site"]:>6s}' for r in rows))
    for k in ('disp', 'clean_true', 'adv_true', 'adv_follows_attack'):
        print(f'  {k:18s}'[:10] + ' '.join(f'{r[k]:6.2f}' for r in rows) + f'   {k}')
# The same from each site's own source coordinates (stored as 'onestep'); centroids from the second half of H,
# since the probe's sources are not fetched locally.
src = np.load(ROOT / f'joint/{tag}/coords_H.npz')['onestep'][:, :, :K].astype(float)
ref, labr = src[:, 1000:], sp['labels'][sp['H'][1000:]]
sc = src[:, :1000]
for adv in ('pgd2', 'pgd8'):
    sa = np.load(ROOT / f'joint/{tag}/coords_{adv}.npz')['onestep'][:, :, :K].astype(float)
    pa = np.load(ROOT / f'meta/{adv}.npz')['logits'].argmax(1)
    rows = []
    for l, s in enumerate(sites):
        cen = np.stack([ref[l][labr == c].mean(0) for c in range(10)])
        sd = sc[l].std(0)
        nc = lambda y: ((y[:, None, :] - cen[None]) ** 2).sum(2).argmin(1)
        cc, ca = nc(sc[l]), nc(sa[l])
        rows.append(dict(site=s, disp=float((((sa[l] - sc[l]) / sd) ** 2).mean()), clean_true=float((cc == lab).mean()),
                         adv_true=float((ca == lab).mean()), adv_follows_attack=float((ca == pa).mean()),
                         clean_matches_pred=float((cc == pred_clean).mean())))
    out[f'{adv}_src'] = rows
    print(adv, 'source coordinates')
    for k in ('disp', 'clean_true', 'adv_true', 'adv_follows_attack'):
        print(f'  {k:18s}'[:10] + ' '.join(f'{r[k]:6.2f}' for r in rows) + f'   {k}')
(ROOT / f'adv_{tag}_K{K}.json').write_text(json.dumps(out))
