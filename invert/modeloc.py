"""How many of the leading joint modes are carried by a handful of probe images.

For each fetched tag and each of the first 64 modes: the share of the mode's squared norm on its two heaviest
images (summed over sites), and the participation share (inverse participation ratio / number of entries).
A mode with most of its norm on two images is a near-duplicate pair, not a direction of the cloud.
Run: uv run --with numpy python invert/modeloc.py
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / 'data/invert'
out = {}
for d in sorted((ROOT / 'joint').glob('*__L12')):
    f = d / 'modes.npz'
    if not f.exists() or '+' in d.name:
        continue
    V = np.load(f)['vecs'][:, :, :64].astype(float)
    L, n, K = V.shape
    w = (V ** 2).sum(0)                                  # (n, K) weight of each image in each mode
    top2 = np.sort(w, 0)[-2:].sum(0) / w.sum(0)
    flat = V.reshape(L * n, K)
    part = 1 / ((flat ** 4).sum(0) / (flat ** 2).sum(0) ** 2) / (L * n)
    out[d.name] = dict(top2=top2.tolist(), part=part.tolist(), n=n)
    print(f'{d.name:20s} n={n}  modes with top-2 share > 0.5: first16={int((top2[:16] > .5).sum())} first64={int((top2 > .5).sum())}'
          f'   > 0.1: first16={int((top2[:16] > .1).sum())} first64={int((top2 > .1).sum())}   first such mode: {int(np.argmax(top2 > .1)) + 1 if (top2 > .1).any() else None}')
(ROOT / 'modeloc.json').write_text(json.dumps(out))
