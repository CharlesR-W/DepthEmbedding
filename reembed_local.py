"""Local re-embedding test (leave one out): decode an image's activation from its neighbours in an embedding.

For image i at site l, take its m nearest other images in some coordinate system and decode x_i as their mean
activation. Report the cosine between x_i and that decode (from the stored Gram C = X X^T alone; no fit, one
parameter m). Neighbours taken in:
  act      activation space itself (cosine): the ceiling for this decoder
  td16     the 16 through-depth coordinates at this site, t = t*, relative units (what the viewer plots)
  td16u    the same with each mode's slice scaled to unit norm (shape only)
  pca16    the top 16 principal coordinates of C
  random   m random images
ResNet-20 seed 0, final checkpoint. Run: uv run --with numpy --with scipy python reembed_local.py
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
STEP, M = 64000, 15


def nn_of(Z):
    d = (Z ** 2).sum(1)[:, None] + (Z ** 2).sum(1)[None] - 2 * Z @ Z.T
    np.fill_diagonal(d, np.inf)
    return np.argpartition(d, M, 1)[:, :M]


def decode_cos(C, nn):
    num = np.take_along_axis(C, nn, 1).mean(1)                       # <x_i, mean_j x_j>
    den = np.sqrt(np.array([C[np.ix_(J, J)].mean() for J in nn]))    # |mean_j x_j|
    return float((num / den / np.sqrt(np.diag(C))).mean())


z = np.load(HERE / f"data/r20s0b/spacetime/st_{STEP}_0.npz")
sites = json.loads((HERE / "data/r20s0b/meta.json").read_text())["sites"]
rng, out = np.random.default_rng(0), []
print(f"{'site':7s} {'act':>6s} {'td16':>6s} {'td16u':>6s} {'pca16':>6s} {'random':>6s}")
for l, site in enumerate(sites):
    C = np.load(HERE / f"data/grams/r20s0b/gram_{STEP}_{l}.npy"); N = len(C)
    S = C.copy(); np.fill_diagonal(S, -np.inf)
    u = z["vecs"][0, 3, :, l, :].astype(float); u /= np.linalg.norm(u, axis=1, keepdims=True)
    w = np.sqrt(z["weight"][0, 3, :, l].astype(float))
    ev, U = eigh(C, subset_by_index=[N - 16, N - 1])
    r = {"site": site, "act": decode_cos(C, np.argpartition(-S, M, 1)[:, :M]), "td16": decode_cos(C, nn_of((u * w[:, None]).T)),
         "td16u": decode_cos(C, nn_of(u.T)), "pca16": decode_cos(C, nn_of(U * np.sqrt(ev))),
         "random": decode_cos(C, np.array([rng.choice(N, M, replace=False) for _ in range(N)]))}
    out.append(r)
    print(f"{site:7s} {r['act']:6.3f} {r['td16']:6.3f} {r['td16u']:6.3f} {r['pca16']:6.3f} {r['random']:6.3f}", flush=True)
(HERE / "data/reembed/r20s0_64000_local.json").write_text(json.dumps(out))
