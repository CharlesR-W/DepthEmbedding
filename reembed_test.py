"""Re-embedding test: how much of a site's activations do the low modes carry back?

At site l the stored Gram C = X X^T (X: probe activations, centred over the probe, rows normalised) fixes X up to a
rotation. For an orthonormal set B of functions on the probe images, the decode of X from B is the projection
X_hat = B B^T X, with Gram C_hat = P C P (P = B B^T). Everything below is computed from C alone:
  captured   tr(B^T C B) / tr(C)            share of X's squared norm that the decode keeps (1 = exact)
  nn15       mean overlap of each image's 15 nearest neighbours (cosine) before and after the decode
Bases, each with the constant function added (dimension K + 1):
  td(c)      site-l slices of the first K through-depth modes, coupling factors[c] x t* (relative site units)
  site       first K nontrivial eigenvectors of this site's own rule-0 walk (kNN-15, symmetric Sinkhorn)
  pca        top K + 1 eigenvectors of C: the best any (K + 1)-dimensional basis can do for `captured`
  random     a random (K + 1)-dimensional subspace: expected captured = (K + 1) / N
ResNet-20 seed 0, final checkpoint. Run: uv run --with numpy --with scipy python reembed_test.py
"""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigh

HERE = Path(__file__).resolve().parent
STEP, KNN = 64000, 15
KS_TD, KS_SITE = (4, 8, 16), (4, 8, 16, 64, 256, 1000)


def walk(C):
    N = len(C)
    D = 1 - C; np.fill_diagonal(D, np.inf)
    nn = np.argpartition(D, KNN, 1)[:, :KNN]
    mask = np.zeros((N, N), bool); mask[np.arange(N)[:, None], nn] = True; mask |= mask.T
    W = np.where(mask, 1.0, 1e-6); np.fill_diagonal(W, 0)
    d = np.ones(N)
    for _ in range(5000):                       # symmetric Sinkhorn, damped
        r = d * (W @ d)
        if np.abs(r - 1).max() < 1e-12: break
        d = d / np.sqrt(r)
    assert np.abs(d * (W @ d) - 1).max() < 1e-10
    return d[:, None] * W * d[None, :]


def nn_sets(C):
    n = np.sqrt(np.clip(np.diag(C), 1e-30, None)); S = C / n[:, None] / n[None, :]
    np.fill_diagonal(S, -np.inf)
    return np.argpartition(-S, KNN, 1)[:, :KNN]


def overlap(a, b):
    return float(np.mean([len(np.intersect1d(x, y)) for x, y in zip(a, b)]) / KNN)


def score(C, B, nn0):
    CB = C @ B
    return {"captured": float(np.trace(B.T @ CB) / np.trace(C)), "nn15": overlap(nn0, nn_sets(B @ (B.T @ CB) @ B.T))}


def with_const(V):
    Q, _ = np.linalg.qr(np.column_stack([np.ones(len(V)), V])); return Q


def main():
    z = np.load(HERE / f"data/r20s0b/spacetime/st_{STEP}_0.npz")
    sites = json.loads((HERE / "data/r20s0b/meta.json").read_text())["sites"]
    factors, rng, out = z["factors"], np.random.default_rng(0), []
    for l, site in enumerate(sites):
        C = np.load(HERE / f"data/grams/r20s0b/gram_{STEP}_{l}.npy"); N = len(C)
        assert np.array_equal(np.arange(N), np.arange(N)) and abs(np.trace(C) - N) < 1e-6
        nn0 = nn_sets(C)
        ev, U = eigh(C); U = U[:, ::-1]; ev = ev[::-1]
        pv, Pv = eigh(walk(C)); Pv = Pv[:, ::-1]
        assert abs(pv[-1] - 1) < 1e-8 and np.abs(Pv[:, 0] * np.sqrt(N)).std() < 1e-6      # constant mode
        rec = {"site": site, "pr": float(ev.sum() ** 2 / (ev ** 2).sum()), "rows": []}
        for K in KS_SITE:
            row = {"K": K, "site": score(C, with_const(Pv[:, 1:K + 1]), nn0), "pca": score(C, U[:, :K + 1], nn0),
                   "random": score(C, with_const(rng.standard_normal((N, K))), nn0)}
            if K in KS_TD:
                for c in (0, 3, 5):
                    row[f"td_{factors[c]:g}"] = score(C, with_const(z["vecs"][0, c, :K, l, :].T.astype(float)), nn0)
            rec["rows"].append(row)
        out.append(rec)
        r = {x["K"]: x for x in rec["rows"]}
        print(f"{site:7s} PR {rec['pr']:6.1f} | K=16 captured: td(t*) {r[16]['td_1']['captured']:.3f} site {r[16]['site']['captured']:.3f} "
              f"pca {r[16]['pca']['captured']:.3f} rand {r[16]['random']['captured']:.3f} | nn15: td {r[16]['td_1']['nn15']:.3f} "
              f"site {r[16]['site']['nn15']:.3f} pca {r[16]['pca']['nn15']:.3f} | K=256 site {r[256]['site']['captured']:.3f}/{r[256]['site']['nn15']:.3f} "
              f"pca {r[256]['pca']['captured']:.3f}/{r[256]['pca']['nn15']:.3f} | K=1000 site {r[1000]['site']['captured']:.3f}/{r[1000]['site']['nn15']:.3f}", flush=True)
    (HERE / "data/reembed").mkdir(exist_ok=True)
    (HERE / "data/reembed/r20s0_64000.json").write_text(json.dumps({"step": STEP, "factors": factors.tolist(), "sites": out}))


if __name__ == "__main__":
    main()
