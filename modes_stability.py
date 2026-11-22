"""How many of a site's walk modes are stable under resampling the probe? (ResNet-20 seed 0, final checkpoint.)

Two random subsets A, B of 1,500 probe images sharing 1,000. Build the rule-0 walk (kNN-15, symmetric Sinkhorn) on
each from the stored Gram, take its eigenvectors, restrict both to the shared images. For mode k of A: the share of
its squared norm inside the span of B's first 2k + 10 modes (1 = reproduced; a random function gives about
(2k + 10) / 1000). A mode counts as reproduced at share >= 0.5 above that chance level... reported: the number of
modes reproduced among the first 500, and the largest k such that 90% of modes up to k are reproduced. 3 repeats.
Run: uv run --with numpy --with scipy python modes_stability.py
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from reembed_test import walk

HERE = Path(__file__).resolve().parent
KMAX, REPS = 500, 3
sites = json.loads((HERE / "data/r20s0b/meta.json").read_text())["sites"]
rng, out = np.random.default_rng(0), []
print(f"{'site':7s} share at k = 4, 16, 64, 128, 256 (chance in brackets)   reproduced/500   stable up to k")
for l, site in enumerate(sites):
    C = np.load(HERE / f"data/grams/r20s0b/gram_64000_{l}.npy")
    shares = []
    for _ in range(REPS):
        p = rng.permutation(2000); sh, a, b = p[:1000], p[:1500], np.concatenate([p[:1000], p[1500:]])
        V = []
        for ix in (a, b):
            w, U = eigh(walk(C[np.ix_(ix, ix)].copy()))
            V.append(U[:1000, ::-1][:, 1:KMAX * 2 + 11])          # shared images come first in both; drop the constant
        A = V[0][:, :KMAX] / np.linalg.norm(V[0][:, :KMAX], axis=0)
        s = []
        for k in range(1, KMAX + 1):
            Q, _ = np.linalg.qr(V[1][:, :2 * k + 10]) if k in (1,) or k % 8 == 0 or k <= 32 else (Q, None)
            s.append(float(((Q.T @ A[:, k - 1]) ** 2).sum()))
        shares.append(s)
    s = np.mean(shares, 0); chance = (2 * np.arange(1, KMAX + 1) + 10) / 1000
    ok = (s - chance) / (1 - chance) >= 0.5
    frac = np.cumsum(ok) / np.arange(1, KMAX + 1)
    stable = int(np.max(np.where(frac >= 0.9)[0]) + 1) if (frac >= 0.9).any() else 0
    out.append({"site": site, "share": s.tolist(), "reproduced": int(ok.sum()), "stable_to": stable})
    print(f"{site:7s} " + "  ".join(f"{s[k-1]:.2f} ({chance[k-1]:.2f})" for k in (4, 16, 64, 128, 256)) + f"   {int(ok.sum()):4d}   {stable:4d}", flush=True)
(HERE / "data/reembed/r20s0_64000_stability.json").write_text(json.dumps(out))
