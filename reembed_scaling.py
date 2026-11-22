"""How does the nearest neighbour get closer as the probe grows? (ResNet-20 seed 0, final checkpoint, from the Grams.)

For random subsets of n probe images, the mean cosine between an image's activation and its nearest other image.
1 - cos is half the squared distance on the sphere, which for points filling a d-dimensional set falls like
n^(-2/d); the slope over n = 250..2000 gives a rough d, and an extrapolated n for a target cosine. An extrapolation,
not a measurement: d is not constant across scales.
Run: uv run --with numpy python reembed_scaling.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
NS, REPS, TARGETS = (250, 500, 1000, 2000), 8, (0.8, 0.9)
sites = json.loads((HERE / "data/r20s0b/meta.json").read_text())["sites"]
rng, out = np.random.default_rng(0), []
print(f"{'site':7s} " + " ".join(f"n={n:<5d}" for n in NS) + "   d_est   n for cos .8   n for cos .9")
for l, site in enumerate(sites):
    C = np.load(HERE / f"data/grams/r20s0b/gram_64000_{l}.npy"); N = len(C)
    S = C.copy(); np.fill_diagonal(S, -np.inf)
    cos = []
    for n in NS:
        v = []
        for _ in range(1 if n == N else REPS):
            ix = rng.choice(N, n, replace=False)
            v.append(S[np.ix_(ix, ix)].max(1).mean())
        cos.append(float(np.mean(v)))
    slope, icpt = np.polyfit(np.log(NS), np.log(1 - np.array(cos)), 1)
    d = -2 / slope
    need = [float(np.exp((np.log(1 - t) - icpt) / slope)) for t in TARGETS]
    out.append({"site": site, "nn_cos": cos, "d_est": d, "n_for": dict(zip(map(str, TARGETS), need))})
    print(f"{site:7s} " + " ".join(f"{c:7.3f}" for c in cos) + f"   {d:5.1f}   {need[0]:12.3g}   {need[1]:12.3g}", flush=True)
(HERE / "data/reembed/r20s0_64000_scaling.json").write_text(json.dumps(out))
