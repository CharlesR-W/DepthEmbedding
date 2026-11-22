"""Geometric harmonics decode (Coifman & Lafon 2006), 10-fold held out. ResNet-20 seed 0, final checkpoint.

Coordinates z_i (16 numbers per image at a site). On the training images: Gaussian kernel k = exp(-|z - z'|^2 / eps),
eigenpairs (mu_m, phi_m), keep mu_m >= delta * mu_0. Each activation coordinate is expanded in the kept phi_m and
extended to a held-out z by Nystrom: x_hat(z) = sum_j w_j(z) x_j, w = k(z, Z) Phi diag(1/mu) Phi^T. Score: mean cosine
between x_i and x_hat(z_i) over held-out images, from the Gram alone. eps and delta are swept and the best pair per
site is reported, chosen on the held-out score itself, so the numbers are slightly optimistic. This is a fit.
Run: uv run --with numpy --with scipy python reembed_gh.py
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
EPS_MULT, DELTAS, FOLDS = (0.25, 1, 4, 16), (1e-1, 1e-2, 1e-3, 1e-4), 10


def sqd(A, B):
    return np.maximum((A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T, 0)


def gh(C, Z, folds):
    N = len(C); D = sqd(Z, Z)
    base = np.median(np.partition(D + np.diag(np.full(N, np.inf)), 15, 1)[:, 15])
    best = (-1, None)
    for em in EPS_MULT:
        cos = {d: np.zeros(N) for d in DELTAS}
        for S in folds:
            T = np.setdiff1d(np.arange(N), S)
            mu, Phi = eigh(np.exp(-D[np.ix_(T, T)] / (em * base)))
            Kst, Cts, Ctt = np.exp(-D[np.ix_(S, T)] / (em * base)), C[np.ix_(T, S)], C[np.ix_(T, T)]
            A = Kst @ Phi
            for d in DELTAS:
                keep = mu >= d * mu[-1]
                W = (A[:, keep] / mu[keep]) @ Phi[:, keep].T
                num = (W * Cts.T).sum(1)
                den = np.sqrt(np.maximum(((W @ Ctt) * W).sum(1), 1e-30))
                cos[d][S] = num / den
        for d in DELTAS:
            if cos[d].mean() > best[0]:
                best = (float(cos[d].mean()), (em, d))
    return best


z = np.load(HERE / "data/r20s0b/spacetime/st_64000_0.npz")
sites = json.loads((HERE / "data/r20s0b/meta.json").read_text())["sites"]
rng = np.random.default_rng(0); folds = np.array_split(rng.permutation(2000), FOLDS)
old = {r["site"]: r for r in json.loads((HERE / "data/reembed/r20s0_64000_local.json").read_text())}
out = []
print(f"{'site':7s} {'GH td16':>8s} {'(eps,delta)':>14s} {'GH pca16':>9s} | neighbour mean: {'td16':>5s} {'act ceiling':>11s}")
for l, site in enumerate(sites):
    C = np.load(HERE / f"data/grams/r20s0b/gram_64000_{l}.npy"); N = len(C)
    u = z["vecs"][0, 3, :, l, :].astype(float); u /= np.linalg.norm(u, axis=1, keepdims=True)
    ev, U = eigh(C, subset_by_index=[N - 16, N - 1])
    a, pa = gh(C, (u * np.sqrt(z["weight"][0, 3, :, l].astype(float))[:, None]).T, folds)
    b, pb = gh(C, U * np.sqrt(ev), folds)
    out.append({"site": site, "gh_td16": a, "params_td16": pa, "gh_pca16": b, "params_pca16": pb})
    print(f"{site:7s} {a:8.3f} {str(pa):>14s} {b:9.3f} | {'':16s}{old[site]['td16']:5.3f} {old[site]['act']:11.3f}", flush=True)
(HERE / "data/reembed/r20s0_64000_gh.json").write_text(json.dumps(out))
