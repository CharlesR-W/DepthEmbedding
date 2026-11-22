"""Do a site's walk modes reproduce on a disjoint half of the probe? (ResNet-20 seed 0, final checkpoint.)

Split the probe into disjoint halves A, B (1,000 each). Build the rule-0 walk on each and take its eigenvectors.
Extend A's mode k to B's images by one walk step from B into A (the mean of the mode over each B image's 15 nearest
A images, the Nystrom extension up to a per-mode scale). Share of that extended function's squared norm (mean
removed) inside the span of B's own first 2k + 10 modes; a random function gives about (2k + 10) / 1000.
`excess` = (share - chance) / (1 - chance). 3 random splits.
Run: uv run --with numpy --with scipy python modes_stability_disjoint.py
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from reembed_test import walk

HERE = Path(__file__).resolve().parent
KS, REPS = (1, 2, 4, 8, 16, 32, 64, 128, 256), 3
sites = json.loads((HERE / "data/r20s0b/meta.json").read_text())["sites"]
rng, out = np.random.default_rng(0), []
print(f"{'site':7s} excess share at k = " + ", ".join(map(str, KS)) + "   | last k with excess >= 0.5")
for l, site in enumerate(sites):
    C = np.load(HERE / f"data/grams/r20s0b/gram_64000_{l}.npy")
    ex = []
    for _ in range(REPS):
        p = rng.permutation(2000); a, b = p[:1000], p[1000:]
        Ua = eigh(walk(C[np.ix_(a, a)].copy()))[1][:, ::-1][:, 1:]
        Ub = eigh(walk(C[np.ix_(b, b)].copy()))[1][:, ::-1][:, 1:]
        nn = np.argpartition(-C[np.ix_(b, a)], 15, 1)[:, :15]
        E = Ua[nn].mean(1); E -= E.mean(0); E /= np.linalg.norm(E, axis=0) + 1e-30      # (1000 B images, modes of A)
        row = []
        for k in KS:
            m = 2 * k + 10
            share = float(((Ub[:, :m].T @ E[:, k - 1]) ** 2).sum()); ch = m / 1000
            row.append((share - ch) / (1 - ch))
        ex.append(row)
    ex = np.mean(ex, 0)
    last = max([k for k, e in zip(KS, ex) if e >= 0.5], default=0)
    out.append({"site": site, "ks": KS, "excess": ex.tolist()})
    print(f"{site:7s} " + " ".join(f"{e:5.2f}" for e in ex) + f"   | {last}", flush=True)
(HERE / "data/reembed/r20s0_64000_stability_disjoint.json").write_text(json.dumps(out))
