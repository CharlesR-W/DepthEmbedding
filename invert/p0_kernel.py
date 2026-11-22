"""Phase 0a: per-site kernel study on the stored 2,000-image Grams (local, no GPU).

For the k-NN rule and the doubly stochastic Gaussian at several perplexities: hubness (k-occurrence of
the 15 heaviest entries per row), spectrum, and reproducibility of the modes on a disjoint half of the probe
(mode of half A, extended to half B by the kernel's own out-of-sample row, share inside B's first 2k+10
modes minus the chance share).
Run: uv run --with numpy --with scipy python invert/p0_kernel.py
"""
import json, sys, time
from pathlib import Path
import numpy as np
from scipy.linalg import eigh

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dm

ROOT = Path(__file__).resolve().parent.parent
SITES = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
NAMES = ['knn15', 'ea30', 'ea100', 'ea300', 'ds150', 'ds500']
KS = [1, 2, 4, 8, 16, 32, 64]


def top_modes(P, k):
    n = len(P)
    w, v = eigh(P, subset_by_index=[n - k - 1, n - 1])
    return w[::-1][1:], v[:, ::-1][:, 1:]


def build(C, name):
    if name == 'knn15':
        return dm.knn_kernel(C), dm.knn_oos_rows
    if name.startswith(('ea', 'pr')):
        return dm.build_kernel(C, name), dm.ea_oos_rows
    return dm.ds_kernel(C, perp=float(name[2:])), dm.oos_rows


def stability(C, name, rng):
    n = len(C)
    perm = rng.permutation(n); A, B = perm[:n // 2], perm[n // 2:]
    kA, rowfn = build(C[np.ix_(A, A)], name)
    kB, _ = build(C[np.ix_(B, B)], name)
    kmax = max(KS)
    lamA, phiA = top_modes(kA['P'], kmax)
    _, phiB = top_modes(kB['P'], 2 * kmax + 10)
    ext = rowfn(C[np.ix_(B, A)], kA) @ phiA               # one out-of-sample step (scale irrelevant below)
    ext -= ext.mean(0); ext /= np.linalg.norm(ext, axis=0)
    out = {}
    for k in KS:
        m = 2 * k + 10
        share = ((phiB[:, :m].T @ ext[:, k - 1]) ** 2).sum()
        chance = m / (len(B) - 1)
        out[k] = float((share - chance) / (1 - chance))
    return out


def main():
    sites = [int(s) for s in sys.argv[1].split(',')] if len(sys.argv) > 1 else range(12)
    names = sys.argv[2].split(',') if len(sys.argv) > 2 else NAMES
    outf = ROOT / 'data/invert/p0_kernel.json'
    res = json.loads(outf.read_text()) if outf.exists() and len(sys.argv) > 2 else {}
    rng = np.random.default_rng(0)
    for si in sites:
        C = np.load(ROOT / f'data/grams/r20s0b/gram_64000_{si}.npy')
        r = res.setdefault(SITES[si], {})
        r['raw_cos_hub'] = dm.k_occurrence(C)
        for name in names:
            t0 = time.time()
            ker, _ = build(C, name)
            lam, _ = top_modes(ker['P'], 64)
            r[name] = dict(eps=ker.get('eps'), perp=ker['perp'], err=ker['err'], hub=dm.k_occurrence(ker['P']),
                           maxw=float(ker['P'].max()), lam=[float(lam[i]) for i in (0, 1, 3, 7, 15, 31, 63)],
                           stab=stability(C, name, np.random.default_rng(1)), sec=time.time() - t0)
            s = r[name]
            print(f"{SITES[si]:6s} {name:6s} eps={s['eps'] or 0:.4f} perp={s['perp']:.0f} hubmax={s['hub']['max']:.0f} "
                  f"skew={s['hub']['skew']:.2f} lam1,16,64={s['lam'][0]:.3f},{s['lam'][4]:.3f},{s['lam'][6]:.3f} "
                  f"stab(1,4,16,64)={s['stab'][1]:.2f},{s['stab'][4]:.2f},{s['stab'][16]:.2f},{s['stab'][64]:.2f} "
                  f"({s['sec']:.0f}s)", flush=True)
    out = ROOT / 'data/invert'; out.mkdir(parents=True, exist_ok=True)
    (out / 'p0_kernel.json').write_text(json.dumps(res))


if __name__ == '__main__':
    main()
