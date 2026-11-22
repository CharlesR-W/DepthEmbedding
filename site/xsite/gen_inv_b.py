"""Generate decoding, shifted-input and edit entries from archived inversion results.

Run from the repo: uv run --with numpy --with pillow python site/xsite/gen_inv_b.py
Use --without-embedding to regenerate chart/fact assets before the operator is packaged.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json

import numpy as np
from PIL import Image

from gen_inv_a import ARROWS, attack_display, packaged
from opdata import read_array
from ops_inv_edits import METHODS as ENDPOINT_METHODS
from xlib import DATA, Out, lines

RESULTS = DATA / "invert/results.json"
SUMMARY = "data/invert/results.json"
RAW = DATA / "fetch/crw-depth-invert-results"
P2 = "P2__pr300__L12"
P8 = "P8__pr400__L12"
OP = "r20s0-p2-pr300"
SETS = {"H": "Held-out CIFAR-10", "c100": "CIFAR-100", "svhn": "SVHN",
        "noise20": "Pixel noise, σ = 0.2", "pgd2": "PGD 2/255", "pgd8": "PGD 8/255"}
KS = (4, 16, 64, 256)


def fact(out, name, value, tier, source, **fmt):
    out.fact(name, value, tier=tier, source=source, **fmt)
    return "{{fact:x." + out.id + "." + name + "}}"


def card(out, answer, *numbers):
    record = dict(answer=answer, numbers=numbers)
    if out.fig_path("thumb").exists():
        record["thumb"] = f"assets/fig/x_{out.id}_thumb.webp"
    out.chart("card", record)
    out.write()


def site_chart(sites, series, label, **kw):
    return lines(list(range(len(sites))), series, "Site", label, xticklabels=sites, **kw)


def trace(name, y, **style):
    return dict(name=name, y=list(y), **style)


def mean_decode(results, tag, site, key):
    return results["decode"][tag][site].get(key, [None])[0]


def decode_fact(out, results, name, site, key, tag=P8, **fmt):
    return fact(out, name, mean_decode(results, tag, site, key), "S",
                f"{SUMMARY}:decode/{tag}/{site}/{key}[0]", **fmt)


def study_shapes():
    # Read the literal without importing the GPU/Modal stage.
    path = DATA.parent / 'invert/modal_invert.py'
    return next(ast.literal_eval(node.value) for node in ast.parse(path.read_text()).body
                if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'SHAPES' for t in node.targets))


def coords(set_id):
    return RAW / "joint" / P2 / f"coords_{set_id}.npz"


def mesh_sets():
    root = DATA / "sitex"
    manifest = json.loads((root / "MANIFEST.json").read_text())
    available = manifest["ops"][OP]["overlays"]
    if missing := set(SETS) - set(available):
        raise ValueError(f"{OP} is missing overlays {sorted(missing)}; ask package B1a")
    return root / "ops" / OP / "ov"


def decode_embedding(out, ov_dir, near):
    held = json.loads((ov_dir / "H.json").read_text())
    scalars = held["meta"]["scalars"]
    choices = [{"scalar": f"decode_cos_K{k}", "label": f"K = {k}"}
               for k in (16, 64, 256) if f"decode_cos_K{k}" in scalars]
    if not any(item["scalar"] == "decode_cos_K16" for item in choices):
        raise ValueError("H overlay is missing per-site decode_cos_K16")
    positions = np.flatnonzero(near[np.asarray(held["sample"]["pos"], dtype=int)]).tolist()
    if not positions:
        raise ValueError("The H overlay sample has no high-row-max images to ring")
    fact(out, 'near.shown', len(positions), 'R', 'Packaged H sample.pos intersected with rowmax flags')
    fact(out, 'near.sample', held['n'], 'S', 'Packaged H.json:n')
    op = packaged(OP)
    axes = np.asarray(op['axes']['pooled'])[:, :2]
    b4 = read_array(ov_dir, held['kinds']['joint'])[op['chain'].index('b4')]
    radii = np.square(b4 @ axes).sum(axis=1)
    fact(out, 'near.projected_top', len(set(positions) & set(np.argsort(radii)[-len(positions):])), 'R',
         'Packaged H overlay b4; count of rings among the largest projected radii, with top count equal to ring count')
    fact(out, 'near.projected_energy', radii.sum() / np.square(b4).sum(), 'R',
         'Packaged H overlay b4; squared distance from origin retained by shipped joint PCA axes', pct=True, digits=1)
    spec = {"type": "emb", "schema": 1, "panels": [{"op": OP}], "site": "b4",
            "kinds": ["joint"], "kind": "joint", "K": 16, "dims": 2, "axes": "pooled:0",
            "norm": "site", "height": 480, "overlays": [{"set": "H", "draw": "points"}],
            "colour": {"by": "scalar", "set": "H", "scalar": choices[0]["scalar"], "options": choices,
                       "title": "Decoder modes", "fit": [2, 98]},
            "highlight": {"overlay": {"set": "H", "positions": positions},
                          "label": "Ringed: row max > 0.5 at any site"}}
    out.chart("emb_lead", spec)


def decode(results, ov_dir):
    out = Out("inv-decode")
    sites = results["sites"]
    with np.load(coords("H"), allow_pickle=False) as saved:
        rowmax = saved["rowmax"]
        near = (rowmax > 0.5).any(axis=0)
        b4 = sites.index("b4")
        sq = np.square(saved["y"][b4, :, :64]).sum(axis=1)
    raw_src = str(coords("H").relative_to(DATA.parent))
    fact(out, "near.count", int(near.sum()), "P", f"{raw_src}:rowmax; sum(any(rowmax > 0.5, axis=0))")
    fact(out, "near.top20", int(near[np.argsort(sq)[-20:]].sum()), "R",
         f"{raw_src}:y[b4,:,:64],rowmax; count with rowmax > 0.5 at any site among 20 largest b4 joint L2 norms")
    with np.load(RAW / "decode" / P2 / "b4__b4.npz", allow_pickle=False) as saved:
        v = saved["H|64|krr|cos_raw"]
        decode_archive = f'data/fetch/crw-depth-invert-results/decode/{P2}/b4__b4.npz'
        p2cos = fact(out, "small.b4", v.mean(dtype=float), "P", f'{decode_archive}:H|64|krr|cos_raw; mean', digits=2)
        for group, mask in [('flagged', near), ('remainder', ~near)]:
            fact(out, f'tail.{group}.n', int(mask.sum()), 'P', f'{raw_src}:rowmax; count in {group} stratum')
            for key, metric in [('cos_raw', 'cosine'), ('cap_raw', 'captured')]:
                fact(out, f'tail.{group}.{metric}', saved[f'H|64|krr|{key}'][mask].mean(dtype=float), 'P',
                     f'{decode_archive}:H|64|krr|{key}; mean conditional on {group} stratum of rowmax from {raw_src}')
    with np.load(RAW / 'splits.npz', allow_pickle=False) as split, \
         np.load(RAW / 'acts/c10/meta.npz', allow_pickle=False) as metadata:
        images = metadata['images']
        digests = {hashlib.sha256(images[i].tobytes()).digest() for i in split['P2']}
        identical = sum(hashlib.sha256(images[i].tobytes()).digest() in digests for i in split['H'])
        fact(out, 'tail.identical', identical, 'R', 'CIFAR-10 meta.npz:images; compare P2/H SHA-256 pixel digests in splits order')
    shapes = study_shapes()
    for site in ('b4', 'pool', 'logits'):
        fact(out, f'dimension.{site}', int(np.prod(shapes[site])), 'S', f'invert/modal_invert.py:SHAPES[{site}] product')
    decode_fact(out, results, 'large.pool_pca', 'pool', 'H|64|pca|cos_raw', digits=3)
    path = f'invert/decode/{P8}/summary.json'
    selected = json.loads((DATA / path).read_text())['b4__b4']['_meta']['sel']['64']['krr']
    fact(out, 'selected.width', selected['h'], 'S', f'data/{path}:b4__b4._meta.sel.64.krr.h')
    fact(out, 'selected.ridge', selected['ridge'], 'S', f'data/{path}:b4__b4._meta.sel.64.krr.ridge', digits=4)
    b4cos = decode_fact(out, results, "large.b4", "b4", "H|64|krr|cos_raw", digits=2)
    poolcos = decode_fact(out, results, "large.pool", "pool", "H|64|krr|cos_raw", digits=2)
    b4top = decode_fact(out, results, "large.b4_agree", "b4", "H|64|krr|agree", pct=True)
    decode_fact(out, results, "large.b4_agree_oracle", "b4", "H|64|krr|agree_on", pct=True)
    decode_fact(out, results, "large.b4_mlp", "b4", "H|64|mlp|agree", pct=True)
    decode_fact(out, results, "large.b4_share_mean", "b4", "H|64|krr|cap_raw", digits=2)
    decode_fact(out, results, "large.b4_share_median", "b4", "H|64|krr|cap_raw_med", digits=2)

    methods = {"krr": "Kernel ridge", "gh": "Latent harmonics", "linear": "Linear regression", "lp": "Laplacian pyramid",
               "lpz": "Zero-diagonal pyramid", "nn": "15-neighbour mean", "pca": "PCA (K + 1)", "mlp": "Trained MLP"}

    def across_sites(method, metric, k):
        return (mean_decode(results, P8, s, f"H|{k if method != 'mean' else 0}|{method}|{metric}") for s in sites)

    def decoder_chart(k):
        return site_chart(sites, [trace(label, across_sites(method, "cos_raw", k), dash=method in {"pca", "mlp"})
                                  for method, label in methods.items()], "Mean centred cosine")

    out.chart("cos_site", decoder_chart(64))
    out.chart("cos_site_16", decoder_chart(16))
    out.chart("cos_budget", {"panels": [dict(title=s, **lines(KS, [
        trace("P2 pr300", (mean_decode(results, P2, s, f"H|{k}|krr|cos_raw") for k in KS), dash=True, dots=True),
        trace("P8 pr400", (mean_decode(results, P8, s, f"H|{k}|krr|cos_raw") for k in KS), dots=True)],
        "Joint modes K (log scale)", "Mean centred cosine", xscale="log", xticks=KS)) for s in ("stem", "b4", "b9")], "shared": True})
    out.chart("captured", site_chart(sites, [trace("Median, kernel ridge", across_sites("krr", "cap_raw_med", 64)),
                                            trace("Mean, kernel ridge", across_sites("krr", "cap_raw", 64)),
                                            trace("Mean, linear", across_sites("linear", "cap_raw", 64), dash=True)],
                                     "Captured share", yrange=None, refline={"y": 0}))
    out.chart("patched", site_chart(sites, [trace("Kernel ridge", across_sites("krr", "agree", 64)),
                                           trace("Trained MLP", across_sites("mlp", "agree", 64)),
                                           trace("Kernel ridge with true norm", across_sites("krr", "agree_on", 64), dash=True),
                                           trace("Probe mean", across_sites("mean", "agree", 64), dash="dot")],
                                    "Top-1 agreement", fmt="%"))
    for src, dest in [("depth_K16", "joint16"), ("depth_K64", "joint64"), ("depthsrc_K64", "source64")]:
        with Image.open(DATA / "invert" / "figs" / f"P8__pr400__L13_{src}.png") as im:
            im.save(out.fig_path(dest), format="WEBP", lossless=True, method=6)
    if ov_dir is not None:
        decode_embedding(out, ov_dir, near)
    card(out, "Given the clean trajectory, kernel ridge recovers activation direction better at pool than at b4, a much larger target.",
         f"{b4cos} b4 cosine (P8)", f"{poolcos} pool cosine (P8)",
         f"{b4top} b4 top-1 agreement (P8)", f"{p2cos} b4 cosine (P2)")


def shifted(results, ov_dir):
    out = Out("inv-ood")
    sites = results["sites"]
    svhn = decode_fact(out, results, "svhn.b4", "b4", "svhn|64|krr|cos_raw", digits=3)
    h = decode_fact(out, results, "h.b4", "b4", "H|64|krr|cos_raw", digits=3)
    comparisons = {}
    for key, prefix in [('svhn', 'svhn'), ('H', 'h')]:
        comparisons[prefix] = (
            decode_fact(out, results, f'{prefix}.b4_captured', 'b4', f'{key}|64|krr|cap_raw'),
            decode_fact(out, results, f'{prefix}.b4_agree', 'b4', f'{key}|64|krr|agree', pct=True, digits=1),
        )
    with np.load(RAW / 'splits.npz', allow_pickle=False) as split, \
         np.load(RAW / 'acts/c10/meta.npz', allow_pickle=False) as clean:
        for budget in (2, 8):
            path = f'acts/pgd{budget}/meta.npz'
            with np.load(RAW / path, allow_pickle=False) as adv:
                labels = adv['labels']
                positions = split['H'][:len(labels)]
                assert np.array_equal(clean['labels'][positions], labels)
                base, pred = clean['logits'][positions].argmax(axis=1), adv['logits'].argmax(axis=1)
            wrong, after, changed = base != labels, pred != labels, base != pred
            counts = {'misclassified': after.sum(), 'wrong_unchanged': (wrong & ~changed).sum(), 'changed': changed.sum()}
            if budget == 2:
                for key, value in [('n', len(labels)), ('wrong_clean', wrong.sum())]:
                    fact(out, f'attack.{key}', int(value), 'R', 'CIFAR-10 logits and labels on the first len(PGD2) H rows')
            for key, value in counts.items():
                fact(out, f'attack.pgd{budget}.{key}', int(value), 'R', f'{path}:logits,labels; compare with aligned clean logits')
    decode_fact(out, results, "c100.b9", "b9", "c100|64|krr|cos_raw", digits=2)
    decode_fact(out, results, "pgd8.b9", "b9", "pgd8|64|krr|cos_raw", digits=2)
    out.chart("dataset_cos", site_chart(sites, [trace(label,
        (mean_decode(results, P8, s, f"{set_id}|64|krr|cos_raw") for s in sites), dash=set_id.startswith("pgd"))
        for set_id, label in SETS.items()], "Mean centred cosine"))
    support = []
    for set_id, label in SETS.items():
        with np.load(coords(set_id), allow_pickle=False) as saved:
            support.append(trace(label, np.median(saved["rowmax"], axis=1), dash=set_id.startswith("pgd")))
    out.chart("rowmax", site_chart(sites, support, "Median largest row weight", yrange=None))
    disp = results["adv"][P8 + "_K16"]
    out.chart("displacement", site_chart(sites, [trace(label, (r["disp"] for r in disp[key]), dash=key.endswith("_src"))
        for key, label in {"pgd2": "2/255 joint", "pgd2_src": "2/255 source", "pgd8": "8/255 joint",
                           "pgd8_src": "8/255 source"}.items()], "Mean squared displacement", yrange=None))
    if ov_dir is not None:
        op = packaged(OP)
        svhn_overlay = packaged(OP, 'ov/svhn.json')
        li = op['chain'].index('b9')
        axes = np.asarray(op['axes']['pooled'])[:, :2]
        probe = read_array(ov_dir.parent, op['kinds']['source'])[li]
        shifted_rows = read_array(ov_dir, svhn_overlay['kinds']['source'])[li]
        for name, p, s in [('full', probe, shifted_rows), ('plane', probe @ axes, shifted_rows @ axes)]:
            fact(out, f'display.svhn.{name}', (s.var(axis=0) / p.var(axis=0)).mean(), 'R',
                 f'Shipped b9 SVHN and P2 source arrays; average per-axis ratio of centred variances ({name})', digits=2)
        attack_display(out, 'b9', (2, 8))
        out.chart("emb_lead", {"type": "emb", "schema": 1, "panels": [{"op": OP}], "site": "b9",
            "kinds": ["source", "joint"], "kind": "source", "K": 16, "dims": 2, "axes": "pooled:0",
            "norm": "site", "height": 480, "overlays": [{"set": "H", "draw": "points", "main": True}],
            "overlayMenu": [{"set": key, "label": label} for key, label in SETS.items()], "overlayMenuLabel": "Dataset",
            "colour": {"by": "class"}, "filmstrip": {"sites": ["stem", "b4", "b6", "b9"]}})
        out.chart("emb_attacks", {"type": "emb", "schema": 1, "panels": [{"op": OP}], "site": "b9",
            "kinds": ["source", "joint"], "kind": "source", "K": 16, "dims": 2, "axes": "pooled:0",
            "norm": "site", "overlays": [{"set": "pgd2", "draw": "arrows", "from": "H-pgd"}],
            "overlayMenu": [{"set": key, "label": SETS[key]} for key in ("pgd2", "pgd8")],
            "overlayMenuLabel": "Attack", "colour": {"by": "class"}, "arrows": ARROWS})
    card(out, "At b4, SVHN has higher reconstruction cosine and captured share than CIFAR-10, but patched agreement is similar.",
         f"{svhn} vs {h} cosine (SVHN vs CIFAR-10)",
         f"{comparisons['svhn'][0]} vs {comparisons['h'][0]} captured share",
         f"{comparisons['svhn'][1]} vs {comparisons['h'][1]} patched top-1 agreement")


def edit_embedding(out, ov_dir):
    axes = np.asarray(packaged(OP)['axes']['pooled'])[:, :2]
    fact(out, 'display.arrows', ARROWS, 'S', 'gen_inv_a.py:ARROWS, initial display setting')
    for site in ("b4", "b6"):
        stem = f"edit-{site}-"
        a, b, residual = (stem + s for s in ("a", "b", "residual"))
        target = json.loads((ov_dir / f"{b}.json").read_text())
        fact(out, f'display.{site}.n', target['n'], 'S', f'Packaged {b}.json:n')
        fact(out, f'display.{site}.donors', len(set(target['images']['ids'])), 'R', f'Packaged {b}.json: distinct images.ids')
        out.chart(f"emb_{site}", {"type": "emb", "schema": 1, "panels": [{"op": OP, "label": f"Edit at {site}"}],
            "site": site, "kinds": ["source", "joint"], "kind": "source", "K": 16, "dims": 2, "axes": "pooled:0",
            "norm": "site", "height": 480, "overlays": [{"set": residual, "draw": "arrows", "from": a,
                        "verb": "edited"}, {"set": b, "draw": "points", "style": "control"}],
            "overlayMenu": [{"set": stem + slug, "label": label} for slug, label in ENDPOINT_METHODS.values()],
            "overlayMenuLabel": "Edit method", "arrows": ARROWS,
            "highlight": {"overlay": {"set": b, "positions": list(range(target["n"]))}, "label": "Ringed: target b"}})

    # These facts describe the rerun slice; keep them separate from the P8 archive facts.
    with np.load(coords("H"), allow_pickle=False) as held:
        spreads = held["onestep"][:, :256, :16].astype(float).std(axis=1, ddof=1)
    for site in ("b4", "b6"):
        path = DATA / "stretch" / "rt_points" / P2 / f"{site}__K16.npz"
        with np.load(path, allow_pickle=False) as raw:
            label = str(path.relative_to(DATA.parent))
            li = json.loads(str(raw["meta"]))["sites"].index(site)
            method = list(raw["methods"])
            first, matched, swap = (method.index(m) for m in ("resid", "resid+match", "swap"))
            keep = raw["pair_ok"]
            key0 = f"replay.{site}"
            for kind, start, end in [('source', raw['s_edit'][method.index('none')], raw['s_edit'][first]),
                                     ('joint', raw['y_a'][li], raw['y_patched'][first, li])]:
                delta = (end - start)[keep]
                fact(out, f'display.{site}.{kind}', np.square(delta @ axes).sum() / np.square(delta).sum(), 'R',
                     f'{label}: residual endpoint minus original, {kind}; squared fraction retained by shipped joint PCA axes',
                     pct=True, digits=1)
            for key, i in [("residual", first), ("match", matched)]:
                fact(out, f"{key0}.{key}", raw["p_b"][i, keep].mean(dtype=float), "P",
                     f"{label}:p_b[methods.index({method[i]}),pair_ok]; mean", pct=True, digits=1)
                miss = np.square((raw["s_edit"][i, keep] - raw["s_edit"][swap, keep]) / spreads[li]).mean()
                fact(out, f"{key0}.{key}_miss", miss, "R",
                     f"{label}:s_edit[methods.index({method[i]}),pair_ok] vs swap, scaled by sample std of first 256 H onestep rows; mean square",
                     digits=2 if key == "residual" else 9)


def edits(results, ov_dir):
    out = Out("inv-edits")
    sites, table = results["sites"], results["rt"][P8]
    methods = {"resid": "Residual guess", "resid+match": "Residual + matching", "match": "Euclidean matching",
               "match_p": "Probe-metric matching", "decode": "Decode b alone", "swap": "Full swap"}

    def val(site, k, key):
        return table[f"{site}__K{k}"][key]

    def edit_fact(name, site, k, key, **fmt):
        return fact(out, name, val(site, k, key), "S", f"{SUMMARY}:rt/{P8}/{site}__K{k}/{key}", **fmt)

    # P8 per-pair rt arrays are not staged; use the per-site producer archive and verify the notebook summary.
    archive = f'data/invert/rt/{P8}/b4__K64.json'
    b4 = json.loads((DATA.parent / archive).read_text())

    def headline(name, method, key, **fmt):
        key = f'E4|{method}|{key}'
        assert b4[key] == val('b4', 64, key)
        return fact(out, name, b4[key], 'S', f'{archive}:{key} (archived per-pair mean)', **fmt)

    resid = headline('b4.residual', 'resid', 'p_b', digits=3)
    match = headline('b4.match', 'resid+match', 'p_b', digits=3)
    edit_fact('b4.random', 'b4', 64, 'E4|rand~resid|p_b', pct=True)
    edit_fact('b4.swap', 'b4', 64, 'E4|swap|p_b', pct=True)
    source_before = headline('b4.miss_before', 'resid', 'miss_S', digits=3)
    source_after = headline('b4.miss_after', 'resid+match', 'miss_S', digits=8)
    joint_before = headline('b4.joint_before', 'resid', 'miss_D')
    joint_after = headline('b4.joint_after', 'resid+match', 'miss_D')
    headline('b4.inferred_miss', 'resid+match_req', 'miss_S', digits=5)
    headline('b4.inferred_prob', 'resid+match_req', 'p_b', digits=5)
    headline('b4.size_before', 'resid', 'size')
    headline('b4.size_after', 'resid+match', 'size')
    shapes = study_shapes()
    for site in ('pool', 'logits'):
        edit_fact(f'rank.{site}_miss', site, 64, 'E4|match|miss_S')
        fact(out, f'rank.{site}', int(np.prod(shapes[site])) - 1, 'S', f'invert/modal_invert.py:SHAPES[{site}] product minus radial degree')

    path = DATA / 'stretch' / 'rt_points' / P2 / 'b4__K16.npz'
    with np.load(path, allow_pickle=False) as saved:
        m = json.loads(str(saved['meta']))['m_full']
        rerun_pair = saved['pair_pos']
    with np.load(RAW / 'splits.npz', allow_pickle=False) as split, \
         np.load(RAW / 'acts/c10/meta.npz', allow_pickle=False) as clean:
        held = split['H'][:m]
        pred = clean['logits'][held].argmax(axis=1)
    order = np.arange(m)
    donor = np.roll(order, 1)
    for shift in range(2, 40):
        same = pred[donor] == pred
        if not same.any():
            break
        donor = np.where(same, np.roll(order, shift), donor)
    np.testing.assert_array_equal(rerun_pair[:, 1], donor[rerun_pair[:, 0]])
    valid = pred[donor] != pred
    reuse = np.unique(donor[valid], return_counts=True)[1]
    for name, count in [('cohort', len(held)), ('pairs', valid.sum()), ('donors', len(reuse)), ('max_reuse', reuse.max()), ('last_index', held[-1])]:
        fact(out, f'sample.{name}', int(count), 'R', 'Rerun m_full; splits.npz:H and clean logits; cyclic donor rule from invert/stage_rt.py')

    check = json.loads((DATA / 'stretch/rt_consistency.json').read_text())
    for key, step in [('p_b', 0.00001), ('size', 0.000001)]:
        worst = max(item['max_abs'] for site in check.values() for method, item in site[key].items() if not method.startswith('rand~'))
        fact(out, f'replay.bound_{key}', (np.floor(worst / step) + 1) * step, 'S',
             f'data/stretch/rt_consistency.json:*:{key}.*.max_abs; strict upper bound rounded up, excluding rand~',
             digits=5 if key == 'p_b' else 6)

    def by_site(k, key):
        return (val(s, k, key) for s in sites)

    def prob(k):
        return site_chart(sites, [trace(label, by_site(k, f"E4|{m}|p_b")) for m, label in methods.items()] + [
            trace("Random, sized to residual guess", by_site(k, "E4|rand~resid|p_b"), dash=True),
            trace("No edit", by_site(k, "E4|none|p_b"), dash="dot")], "Mean probability of b's class", fmt="%")

    out.chart("prob_64", prob(64))
    out.chart("prob_16", prob(16))
    # The shared line renderer supports unconnected points on a common x array. Each point is one site × K mean,
    # not a per-pair sample. Keep exact x values (do not bin sizes to make a categorical chart).
    for k in (16, 64):
        points = {m: {val(s, k, f"E4|{m}|size"): val(s, k, f"E4|{m}|p_b") for s in sites}
                  for m in (*methods, "rand~resid") if m != "swap"}
        xx = sorted({x for p in points.values() for x in p} | {0, 1})
        out.chart(f"size_{k}", lines(xx, [trace(methods.get(m, "Random, sized to residual guess"), (p.get(x) for x in xx),
                    noline=True, dots=4) for m, p in points.items() if p] + [trace("Full swap", (val("b4", k, "E4|swap|p_b") if x == 1 else None for x in xx), noline=True, dots=4)],
                    "Mean edit size (full swap = 1)", "Mean probability of b's class", fmt="%"))
    out.chart("miss", site_chart(sites, [trace(methods[m], by_site(64, f"E4|{m}|miss_S"))
        for m in ("resid", "resid+match", "match", "match_p")], "Mean squared source miss", yrange=None))
    out.chart("requests", {"panels": [dict(title=f"K = {k}", **site_chart(sites, [
        trace("Midpoint", by_site(k, "E5|0.5|miss_S")), trace("1.5× extrapolation", by_site(k, "E5|1.5|miss_S"))],
        "Mean squared source miss", yrange=[0, 3.25])) for k in (16, 64)], "shared": True})
    if ov_dir is not None:
        edit_embedding(out, ov_dir)
    card(out, "At b4, matching the donor’s source nearly exactly made the patched joint match slightly worse.",
         f"{source_before} → {source_after} source miss", f"{joint_before} → {joint_after} joint miss",
         f"{resid} → {match} donor probability (P8, K = 64)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--without-embedding", action="store_true")
    opt = p.parse_args()
    results = json.loads(RESULTS.read_text())
    overlays = None if opt.without_embedding else mesh_sets()
    decode(results, overlays)
    shifted(results, overlays)
    edits(results, overlays)


if __name__ == "__main__":
    main()
