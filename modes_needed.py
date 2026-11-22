"""How many walk modes does a linear decode need? (ResNet-20 seed 0, final checkpoint, in-sample, from the Grams.)

For a symmetric doubly stochastic walk on the 2,000 probe images, order its eigenvectors by eigenvalue (constant
first). captured(K) = share of the activations' squared norm inside the first K (tr(B^T C B) / tr(C)); report the
smallest K reaching 0.5, 0.8, 0.9 (per-image cosine is roughly sqrt(captured): 0.71, 0.89, 0.95). Walks:
  knn      rule 0: 15-nearest-neighbour graph
  g1, g/4, g/16   Gaussian on cosine, exp(-(1 - cos) / s), s = 1, 4, 16 x the median 15th-neighbour distance
  pca      eigenvectors of C itself: the fewest any basis can need (the s -> infinity limit of the Gaussian walks)
In-sample: K = 1,999 is exact for every basis. It says how compact each basis is, not how well it generalises.
Run: uv run --with numpy --with scipy python modes_needed.py
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from reembed_test import walk

HERE = Path(__file__).resolve().parent
TARGETS = (0.5, 0.8, 0.9)


def sinkhorn(W):
    d = np.ones(len(W))
    for _ in range(20000):
        r = d * (W @ d)
        if np.abs(r - 1).max() < 1e-11: break
        d = d / np.sqrt(r)
    assert np.abs(d * (W @ d) - 1).max() < 1e-8
    return d[:, None] * W * d[None, :]


def need(C, U):                      # U columns ordered, first = leading
    cum = np.cumsum(np.einsum("ik,ik->k", U, C @ U)) / np.trace(C)
    return [int(np.searchsorted(cum, t) + 1) for t in TARGETS]


sites = json.loads((HERE / "data/r20s0b/meta.json").read_text())["sites"]
out = []
print(f"{'site':7s} K for captured 0.5 / 0.8 / 0.9:  knn | g1 | g/4 | g/16 | pca")
for l, site in enumerate(sites):
    C = np.load(HERE / f"data/grams/r20s0b/gram_64000_{l}.npy"); N = len(C)
    D = 1 - C; np.fill_diagonal(D, np.inf); s0 = np.median(np.partition(D, 14, 1)[:, 14]); np.fill_diagonal(D, 0)
    rec = {"site": site, "knn": need(C, eigh(walk(C.copy()))[1][:, ::-1])}
    for name, m in (("g1", 1), ("g/4", 4), ("g/16", 16)):
        W = np.exp(-D / (m * s0)); np.fill_diagonal(W, 0)
        rec[name] = need(C, eigh(sinkhorn(W))[1][:, ::-1])
    rec["pca"] = need(C, eigh(C)[1][:, ::-1])
    out.append(rec)
    print(f"{site:7s} " + " | ".join("/".join(f"{k:4d}" for k in rec[n]) for n in ("knn", "g1", "g/4", "g/16", "pca")), flush=True)
(HERE / "data/reembed/r20s0_64000_modes_needed.json").write_text(json.dumps(out))
