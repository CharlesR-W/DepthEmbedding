"""Run it in reverse: decode an image's activation at site b from its neighbours at site a. (ResNet-20 seed 0, final.)

For each image, take its 15 nearest other probe images at site a (cosine) and average their activations at site b;
score the cosine with the image's own activation at site b. Row = site whose neighbourhood is used, column = site
decoded. Below the diagonal is the reverse direction (a later site's neighbourhood decoding an earlier site).
Run: uv run --with numpy python reembed_cross.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
M = 15
sites = json.loads((HERE / "data/r20s0b/meta.json").read_text())["sites"]
Cs = [np.load(HERE / f"data/grams/r20s0b/gram_64000_{l}.npy") for l in range(len(sites))]
nns = []
for C in Cs:
    S = C.copy(); np.fill_diagonal(S, -np.inf); nns.append(np.argpartition(-S, M, 1)[:, :M])
R = np.zeros((len(sites), len(sites)))
for a, nn in enumerate(nns):
    for b, C in enumerate(Cs):
        num = np.take_along_axis(C, nn, 1).mean(1)
        den = np.sqrt(C[nn[:, :, None], nn[:, None, :]].mean((1, 2)))
        R[a, b] = (num / den).mean()
print("nbrs at \\ decode " + " ".join(f"{s:>6s}" for s in sites))
for a, s in enumerate(sites):
    print(f"{s:16s} " + " ".join(f"{x:6.2f}" for x in R[a]))
(HERE / "data/reembed/r20s0_64000_cross.json").write_text(json.dumps({"sites": sites, "cos": R.tolist()}))
