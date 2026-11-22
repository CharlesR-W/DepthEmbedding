"""Build X3 chart/figure specs, facts and card from the saved run and packaged operators.

The browser checker captures the card thumbnail from the rendered lead figure;
rerun this generator after that capture to add it to the card.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from opdata import read_array
from xlib import DATA, PALETTE, ROOT, Out, bars, heat, lines

ID = "x3-adversarial"
RAW = ROOT / "xp/out/x3"
SITEOPS = DATA / "sitex/ops"
A, B = "c10-s0", "c10-s1"
EA, KNN = "ea100", "knn15"
PREFIX = "xp/out/x3/"
ATTACK, CONTROL = PALETTE["accent"], PALETTE["mute"]
BLUE, TEAL = PALETTE["classes"][0:2]
STRETCH = "data/stretch/x3/"
# Pin this one matched regenerated run independently of the archived embedding bundle.
STRETCH_PINS = {
    STRETCH + "sets.npz": "7fb62156ef30946d72aa2da6a02839e18a9fd5c8b8cbed67f63f7176ea2ac8c7",
    STRETCH + "c10-s0_clean_pr100.npz": "2d44be10696753bb15821528930c58a64f4a56516176f7bc779fa539765a2815",
    STRETCH + "c10-s0_u8_pr100.npz": "727050cfe17522e7d65cda6ff13d4fc9652d5e9232dcab47fa85545604828bc2",
    STRETCH + "c10-s0_n8_pr100.npz": "fc9a7ad848698ada1f0342ff23098d06809f596753d2de17f38339e9e7845d40",
    STRETCH + "op_c10-s0_ea100.npz": "ca2a3e7690b9dc19cb76dc4f6005dfbadaf1bf384c9cb1d3e095d4ce83fc14e0",
    STRETCH + "op_c10-s0_pr100.npz": "9ff9bed423a530c4aaa6ddc3ddb187ee405969011009cea54426f16239a6acc1",
    STRETCH + "op_c10-s0_knn15.npz": "345cf118f2c7f3b57d0b73bf6d7def5ef9de366a39f3c152a2ed23e2e5b7cdb3",
    PREFIX + "sets.npz": "3e98c866e04c1ba3a6775fb3e36bfaf7641f43db4598dfc5908591c856b60d63",
    PREFIX + "c10-s0_clean.npz": "50a15ed2f2f7fcd64339abfdf72b0cfc87db2c839ee2c97850c46cb86d28b86b",
    PREFIX + "c10-s0_u8.npz": "4051a2ab38d9eb2ff5a6b0d1239fef5a7374717cff9629bab110dfcf428d15cc",
    PREFIX + "c10-s0_n8.npz": "1b2b7edb354b1a4c883a1148446821f40a70cad4d1efb62900f8bd915cfe1718",
}


def load_at(path: Path, *keys: str) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in keys}


def load(name: str, *keys: str) -> dict:
    return load_at(RAW / f"{name}.npz", *keys)


def token(key: str) -> str:
    return "{{fact:x." + ID + "." + key + "}}"


def norm(x, axis=-1):
    return np.linalg.norm(x, axis=axis)


def top_two(weights, axis):
    return np.take(np.partition(weights, -2, axis=axis), [-2, -1], axis=axis).sum(axis) / weights.sum(axis)


def localisation(root: Path, op: dict) -> tuple[np.ndarray, np.ndarray]:
    """Squared-norm fraction on the two largest images, from decoded shipped modes."""
    joint = read_array(root, op["kinds"]["joint"])
    site = read_array(root, op["kinds"]["site"])
    return top_two((joint**2).sum(axis=0), 0), top_two(site**2, 1)


def switches(mass, src, dst, top=False):
    """Read each trajectory backwards: one past the last site failing the rule."""
    i = np.arange(len(src))
    d = mass[:, i, dst]
    good = d >= mass.max(2) - 1e-12 if top else d > mass[:, i, src]
    return np.where(~good, np.arange(1, mass.shape[0] + 1)[:, None], 0).max(axis=0)


def line(name, y, color, dash=False):
    return dict(name=name, y=y, color=color, dash=dash, dots=2)


def matched_rerun(out, sites, k, archive, archived_med, si, sj):
    for rel, expected in STRETCH_PINS.items():
        with (ROOT / rel).open("rb") as f:
            actual = hashlib.file_digest(f, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"matched re-run cohort file {rel}: SHA-256 {actual}, expected {expected}")

    def get(name, *keys):
        return load_at(ROOT / (STRETCH + name + ".npz"), *keys)

    def fact(key, value, source, *, tier="P", digits=None, pct=False):
        out.fact(f"rerun.{key}", value, tier=tier, digits=digits, pct=pct,
                 source=f"matched re-run cohort: {source}")

    ids = get("sets", "S", "labels", "img_u8")
    if any(not np.array_equal(ids[key], archive[key]) for key in ("S", "labels")):
        raise ValueError("matched re-run cohort must use the archived clean image IDs and labels in the same order")
    y = ids["labels"]
    srck = STRETCH + "c10-s0_{clean,u8,n8}_pr100.npz"
    for s in ("u8", "n8"):
        recomputed = get(f"{A}_{s}_pr100", "logits")["logits"].argmax(1)
        original = load(f"{A}_{s}", "logits")["logits"].argmax(1)
        fact(f"agreement.{s}.predictions", (recomputed == original).mean(),
             f"{STRETCH}c10-s0_{s}_pr100.npz vs {PREFIX}c10-s0_{s}.npz: share of matching argmax predictions", digits=1, pct=True)
    fact("agreement.u8.pixels", (ids["img_u8"] == load("sets", "img_u8")["img_u8"]).mean(),
         f"{STRETCH}sets.npz vs {PREFIX}sets.npz: fraction of equal rounded img_u8 channel values", digits=1, pct=True)

    pred = get(f"{A}_u8_pr100", "logits")["logits"].argmax(1)
    ok = pred != y
    ratios, fractions = {}, {}
    ordered = [*sites, "Never"]
    for kernel in (EA, "pr100", KNN):
        coords = f"{kernel}_site_y"
        clean, attack, noise = (get(f"{A}_{s}_pr100", coords)[coords] for s in ("clean", "u8", "n8"))
        raw_op = get(f"op_{A}_{kernel}", "vecs", "site_vec")
        joint_share = top_two((raw_op["vecs"][..., :k] ** 2).sum(0), 0)
        walk_share = top_two(raw_op["site_vec"][..., :k].astype(float) ** 2, 1)

        def displacement(keep):
            unit = np.median(norm((clean[:, si] - clean[:, sj]) * keep), 1)
            return [np.median(norm((z - clean) * keep) / unit[:, None], 1) for z in (attack, noise)]

        a, n = displacement(True)
        ratios[kernel] = a / n
        if kernel == KNN:
            continue  # k-NN contributes its matched displacement curve; the table compares Shannon and PR.
        for site in ("b6", "b7", "b8", "b9"):
            fact(f"ratio.{kernel}.{site}", ratios[kernel][sites.index(site)],
                 f"{srck}:{coords}; median u8 displacement divided by median n8 displacement at {site}", digits=3)
        for name, val, arr in (("joint", joint_share, "vecs"), ("site", walk_share, "site_vec")):
            fact(f"concentration.{kernel}.{name}", val.max(),
                 f"{STRETCH}op_{A}_{kernel}.npz:{arr}; maximum two-image fraction over the first 16 modes", tier="R", digits=2, pct=True)
        if kernel == EA:
            ma, mn = displacement((walk_share <= 0.1)[:, None, :])
            b7 = sites.index("b7")
            fact("ratio.ea100.b7_masked", (ma / mn)[b7], f"{srck}:ea100_site_y at b7 after dropping modes with two-image share > 0.1 and renormalising", digits=3)
            for s, new in (("u8", a), ("n8", n)):
                fact(f"archive_delta.{s}", np.abs(new - archived_med[s]).max(),
                     f"{srck} vs {PREFIX}c10-s0 set files; maximum sitewise difference in median ea100_site_y displacement", digits=3)
        m = get(f"{A}_u8_pr100", f"{kernel}_mass")[f"{kernel}_mass"][:, ok]
        src = f"{STRETCH}c10-s0_u8_pr100.npz:{kernel}_mass and logits; successful untargeted 8/255 attempts"
        for name, top in (("pair", False), ("top", True)):
            v = switches(m, y[ok], pred[ok], top=top)
            bins = np.bincount(v, minlength=len(ordered))
            fractions[kernel, name] = bins / len(v)
            fact(f"switch.{kernel}.{name}_median", ordered[np.searchsorted(bins.cumsum(), len(v) / 2)], src + f"; {name} rule median")
            fact(f"switch.{kernel}.{name}_mode", ordered[bins.argmax()], src + f"; {name} rule mode")
        dst = m[:, np.arange(ok.sum()), pred[ok]].mean(1)
        for site in ("pool", "logits"):
            fact(f"mass.{kernel}.{site}", dst[sites.index(site)], src + f"; mean final-class mass at {site}", digits=1, pct=True)
    for name in ("pair", "top"):
        fact(f"switch.{name}_tv", np.abs(fractions[EA, name] - fractions["pr100", name]).sum() / 2,
             f"{STRETCH}c10-s0_u8_pr100.npz:ea100_mass,pr100_mass; {name} switch-rule histogram total variation, shown in percentage points",
             digits=1, pct=True)
    out.chart("kernel_sensitivity", lines(list(range(len(sites))), [
        line("Shannon 100", ratios[EA], ATTACK), line("Participation ratio 100", ratios["pr100"], BLUE),
        line("k-NN 15", ratios[KNN], CONTROL, True),
    ], "Site", "Attack ÷ noise", xticklabels=sites,
        yrange=[0, 1.08 * max(float(v.max()) for v in ratios.values())],
        refline={"y": 1, "label": "Same median distance"}))


def main():
    out = Out(ID)
    ea_root = SITEOPS / f"r20s0-c10-{EA}"
    knn_root = SITEOPS / f"r20s0-c10-{KNN}"
    ea, knn = [json.loads((d / "op.json").read_text()) for d in (ea_root, knn_root)]
    sites = ea["chain"]
    K = ea["K"]
    assert sites == knn["chain"] and K == knn["K"] == 16
    nsite = len(sites)
    x = list(range(nsite))
    source = load("sets", "S", "labels", "target", "natA_y")
    y, target, natural_y = (source[k] for k in ("labels", "target", "natA_y"))
    n = len(y)
    assert len(source["S"]) == n

    # Confirm the chosen sets really are the same paired sample across choices and operators.
    first = None
    for d, op in ((ea_root, ea), (knn_root, knn)):
        for name in ("clean", "u2", "u8", "t8", "n8"):
            overlay = json.loads((d / "ov" / f"{name}.json").read_text())
            if first is None:
                first = overlay["sample"]["pos"]
            assert overlay["op_fp"] == op["fp"] and overlay["sample"]["pos"] == first
            assert overlay["images"]["ids"] == source["S"][first].tolist()
            assert overlay["pairs"] == (None if name == "clean" else {"set": "clean", "by": "index"})

    def fact(key, value, source, *, tier="P", digits=None, pct=False):
        out.fact(key, value, digits=digits, pct=pct, tier=tier, source=source)

    fact("cohort.n", n, PREFIX + "sets.npz:S length", tier="R")
    fact("probe.n", ea["probe"]["n"], "data/sitex/ops/r20s0-c10-ea100/op.json:probe.n (also the joint probe array's image dimension)", tier="R")
    fact("sample.n", len(first), "data/sitex/ops/r20s0-c10-ea100/ov/u8.json:sample.pos length", tier="R")
    fact("metric.k", K, PREFIX + "c10-s0_clean.npz:ea100_site_y last dimension", tier="R")

    clean = load(f"{A}_clean", f"{EA}_site_y", f"{EA}_mass", f"{EA}_y")
    attack = {s: load(f"{A}_{s}", "logits", f"{EA}_site_y", f"{EA}_mass") for s in ("u2", "u8", "t8")}
    pred = {s: d["logits"].argmax(1) for s, d in attack.items()}
    success = {s: p == target if s == "t8" else p != y for s, p in pred.items()}
    noise = load(f"{A}_n8", f"{EA}_site_y")
    i, j = np.random.default_rng(0).integers(0, n, (40_000, 2)).T
    same = (i != j) & (y[i] == y[j])
    z = clean[f"{EA}_site_y"]
    unit = np.median(norm(z[:, i[same]] - z[:, j[same]]), axis=1)
    disp = {s: norm(d[f"{EA}_site_y"] - z) / unit[:, None] for s, d in (*attack.items(), ("n8", noise))}
    med = {s: np.median(d, 1) for s, d in disp.items()}
    for s in ("u8", "n8"):
        for site in ("b6", "b7", "b9"):
            fact(f"disp.{s}.{site}", med[s][sites.index(site)],
                 f"{PREFIX}c10-s0_{{clean,{s}}}.npz:ea100_site_y; median of all attempts at {site}, divided by median clean same-class pair distance", digits=2)
    def displacement(curves):
        return lines(x, [line("Untargeted PGD", curves["u8"], ATTACK), line("Sign noise", curves["n8"], CONTROL, True)],
                     "Site", "Median distance (same-class units)", yrange=[0, max(float(v.max()) for v in curves.values()) * 1.08],
                     xticklabels=sites, refline={"y": 1, "label": "Median clean same-class distance"})

    out.chart("displacement", displacement(med))
    # An additional sensitivity check from the saved k-NN coordinates, not part of the original report.
    zk = load(f"{A}_clean", f"{KNN}_site_y")[f"{KNN}_site_y"]
    unitk = np.median(norm(zk[:, i[same]] - zk[:, j[same]]), axis=1)
    kmed = {}
    for s in ("u8", "n8"):
        kmed[s] = np.median(norm(load(f"{A}_{s}", f"{KNN}_site_y")[f"{KNN}_site_y"] - zk) / unitk[:, None], axis=1)
        for site in ("b6", "b7", "b9"):
            fact(f"disp.knn15.{s}.{site}", kmed[s][sites.index(site)],
                 f"{PREFIX}c10-s0_{{clean,{s}}}.npz:knn15_site_y; median of all attempts at {site}, divided by median clean same-class pair distance", digits=2)
    out.chart("displacement_knn", displacement(kmed))
    matched_rerun(out, sites, K, source, med, i[same], j[same])

    cohorts = {s: (d[f"{EA}_mass"][:, success[s]], y[success[s]], pred[s][success[s]]) for s, d in attack.items()}
    natural = load(f"{A}_natA", "logits", f"{EA}_mass")
    npred = natural["logits"].argmax(1)
    err = npred != natural_y
    cohorts["natA"] = (natural[f"{EA}_mass"][:, err], natural_y[err], npred[err])
    check = load(f"{A}_u8", f"{KNN}_mass")
    cohorts[KNN] = (check[f"{KNN}_mass"][:, success["u8"]], y[success["u8"]], pred["u8"][success["u8"]])
    ordered = [*sites, "Never"]
    sw = {s: switches(*data) for s, data in cohorts.items()}
    sw_top = switches(*cohorts["u8"], top=True)

    def switch_source(s="u8", kernel=EA):
        return f"{PREFIX}c10-s0_{s}.npz:{kernel}_mass and logits; sets.npz labels/target; persistent class-mass rule"

    def shares(v):
        return np.bincount(v, minlength=nsite + 1) / len(v)

    def med_site(v):
        return ordered[np.searchsorted(np.cumsum(np.bincount(v, minlength=nsite + 1)), len(v) / 2)]

    def modal_site(v):
        return ordered[np.argmax(np.bincount(v, minlength=nsite + 1))]

    for s in ("u8", "u2", "t8"):
        fact(f"switch.{s}.median", med_site(sw[s]), switch_source(s))
    fact("switch.u8.mode", modal_site(sw["u8"]), switch_source())
    fact("switch.u8.by_b6", (sw["u8"] <= sites.index("b6")).mean(), switch_source(), pct=True)
    fact("switch.u8.stem", shares(sw["u8"])[0], switch_source(), pct=True)
    fact("switch.u8.top_median", med_site(sw_top), switch_source() + "; destination at row maximum within 1e-12")
    fact("switch.natural.never", shares(sw["natA"])[-1], switch_source("natA") + "; natA_y and holdout natural errors", pct=True)
    fact("switch.knn15.mode", modal_site(sw[KNN]), switch_source(kernel=KNN))
    fact("switch.kernel_tv", np.abs(shares(sw["u8"]) - shares(sw[KNN])).sum() / 2,
         switch_source() + " vs knn15_mass; half L1 distance between binned shares", pct=True)

    def bar(name, values, color, control=False):
        return dict(name=name, values=values, color=color, control=control)

    def hist_panel(title, series):
        return bars(ordered, series, "Share of cohort", "First persistent switch", title=title, yrange=[0, 0.5], fmt="pct")

    u8_bar = bar("Untargeted 8/255", shares(sw["u8"]), ATTACK)
    out.chart("switches", hist_panel("Final class outweighs true class", [u8_bar, bar("Natural errors", shares(sw["natA"]), CONTROL, True)]))
    # Separate the rule change from the kernel change: comparisons use exactly the same attempts.
    out.chart("switch_checks", {"panels": [
        hist_panel("Same rule, different kernel", [bar("Shannon", u8_bar["values"], ATTACK), bar("k-NN", shares(sw[KNN]), CONTROL, True)]),
        hist_panel("Same kernel, different rule", [bar("Beats true class", u8_bar["values"], ATTACK), bar("Highest mass", shares(sw_top), CONTROL, True)]),
        hist_panel("Other attack settings", [bar("Untargeted 2/255", shares(sw["u2"]), BLUE), bar("Targeted 8/255", shares(sw["t8"]), TEAL)]),
    ]})

    mass_panels = []
    clean_own = clean[f"{EA}_mass"][:, np.arange(n), y].mean(1)
    for s, title in (("u8", "Untargeted 8/255"), ("t8", "Targeted 8/255"), ("natA", "Natural errors")):
        m, src, dst = cohorts[s]
        i = np.arange(len(src))
        a, b = m[:, i, src].mean(1), m[:, i, dst].mean(1)
        if s == "u8":
            for nm, v in (("true", a), ("destination", b), ("clean_true", clean_own)):
                fact(f"mass.u8.b6.{nm}", v[sites.index("b6")], switch_source() + "; mean b6 mass, clean reference from c10-s0_clean.npz", digits=2)
            fact("mass.u8.logits.destination", b[-1], switch_source() + "; mean logits mass on destination", pct=True)
        if s == "natA":
            fact("mass.natural.logits.destination", b[-1], switch_source("natA") + "; mean logits mass on destination", pct=True)
        mass_panels.append(lines(x, [line("True class", a, BLUE), line("Final class", b, ATTACK), line("Other classes", 1 - a - b, TEAL),
                                      line("True class, correct clean images", clean_own, CONTROL, True)], "Site", "Mean class mass", title=title, xticklabels=sites))
    out.chart("class_mass", {"panels": mass_panels})

    geometry = {s: load(f"{A}_{s}", "relpert", "actcos") for s in ("u8", "n8")}
    geom = {s: {k: np.median(v, 1) for k, v in d.items()} for s, d in geometry.items()}
    for s in geom:
        fact(f"activation.{s}.b6.cosine", geom[s]["actcos"][sites.index("b6")],
             f"{PREFIX}c10-s0_{s}.npz:actcos; median at b6 over all attempts", digits=2)
    out.chart("geometry", {"panels": [
        lines(x, [line("Untargeted PGD", geom["u8"][k], ATTACK), line("Sign noise", geom["n8"][k], CONTROL, True)],
              "Site", axis, yrange=lim, xticklabels=sites, title=title)
        for k, title, axis, lim in (
            ("relpert", "Relative activation change", "Median relative norm", [0, 1.08 * float(max(geom[s]["relpert"].max() for s in geom))]),
            ("actcos", "Activation direction", "Median cosine", [min(0., float(min(geom[s]["actcos"].min() for s in geom))), 1]),
        )
    ]})

    # The saved intervention norms cover 32 joint modes. Reconstruct the same row swap in the plotted 16.
    op = load(f"op_{A}_{EA}", "vals", "c", "t")
    vals, c, t = op["vals"][:K], op["c"], op["t"]
    adj = np.eye(nsite, k=1) + np.eye(nsite, k=-1)
    diagonal = 0.5 / c[None, :] + t * adj.sum(1)[None, :] - vals[:, None]
    T = np.eye(nsite) * diagonal[:, :, None] - t * adj
    full = load(f"{A}_u8", f"{EA}_y")[f"{EA}_y"] - clean[f"{EA}_y"]
    delta = full.transpose(2, 0, 1)
    rhs = np.zeros_like(delta, dtype=float)
    rhs[:, -1] = (T @ delta)[:, -1]
    mix = np.linalg.solve(T, rhs).transpose(1, 2, 0)
    ratio = np.median(norm(mix), 1) / np.median(norm(full), 1)
    for site in ("stem", "b9"):
        fact(f"logits_only.{site}", ratio[sites.index(site)],
             f"{PREFIX}c10-s0_{{clean,u8}}.npz:ea100_y and op_c10-s0_ea100.npz:c,t,vals; apply T to full joint-coordinate change, keep only logits RHS, solve; ratio of median norms in first 16 modes at {site}", digits=1, pct=True)
    ck = load("check_logits_only", "local_max_change")
    fact("logits_only.earlier_site_local", float(ck["local_max_change"]), PREFIX + "check_logits_only.npz:local_max_change; saved scalar maximum absolute pre-logits change", tier="S", digits=0)

    loc = [localisation(r, d) for r, d in ((ea_root, ea), (knn_root, knn))]
    for name, (joint, site) in zip((EA, KNN), loc):
        which = np.unravel_index(site.argmax(), site.shape)
        locsrc = f"data/sitex/ops/r20s0-c10-{name}/op.json kinds arrays; fraction of squared norm on two largest images"
        fact(f"localisation.{name}.joint16", joint[-1], locsrc + "; joint mode 16, summed over sites per image", tier="R", digits=1, pct=True)
        fact(f"localisation.{name}.site_max", site[which], locsrc + "; maximum across sites and their first 16 nonconstant walk modes", tier="R", digits=1, pct=True)
        fact(f"localisation.{name}.site_at", sites[which[0]], locsrc + "; site of that maximum", tier="R")
        fact(f"localisation.{name}.site_mode", int(which[1]) + 1, locsrc + "; walk-mode index of that maximum, counting from first nonconstant", tier="R")
    out.chart("localisation", {"panels": [
        bars(labels, [bar("Shannon", q0, ATTACK), bar("k-NN", q1, CONTROL, True)],
             "Share on two images", "", title=title, yrange=[0, max(float(q0.max()), float(q1.max())) * 1.1], fmt="pct")
        for labels, q0, q1, title in (
            ([str(k + 1) for k in range(K)], loc[0][0], loc[1][0], "Each joint mode"),
            (sites, loc[0][1].max(1), loc[1][1].max(1), "Largest walk-mode share at each site"),
        )
    ]})
    out.chart("walk_localisation", heat([loc[1][1], loc[0][1]], sites, list(range(1, K + 1)),
                                        "share on two images", "Site", "Walk mode", fmt="pct", zrange=[0, 1],
                                        slider={"label": "Kernel", "values": ["k-NN", "Shannon"]}, ml=57))

    for s in ("u8", "t8"):
        guess = load(f"{B}_{s}", "logits")["logits"].argmax(1)
        ok = success[s]
        trans = f"{PREFIX}{{c10-s0,c10-s1}}_{s}.npz:logits; sets.npz:labels,target; argmax on attempted images"
        fact(f"success.{s}", ok.mean(), trans + "; success on seed 0", pct=True)
        fact(f"transfer.{s}.flip", (guess[ok] != y[ok]).mean(), trans + "; seed 1 any flip conditional on seed 0 success", digits=1, pct=True)
        fact(f"transfer.{s}.same", (guess[ok] == pred[s][ok]).mean(), trans + "; same destination conditional on seed 0 success", digits=1, pct=True)

    def filmstrip(curves):
        # Fixed 8/255 reference, not the projected lengths or the current menu selection.
        return {
            "sites": ["stem", "b3", "b6", "b7", "b8", "b9"],
            "stat": {"label": "Local A÷N", "kind": "site", "values": curves["u8"] / curves["n8"]},
        }

    out.chart("emb_lead", {
        "type": "emb", "schema": 1, "switchLabel": "Kernel", "switch": [
            {"label": "Shannon", "panels": [{"op": ea["id"], "label": "Shannon perplexity 100"}], "site": "b6"},
            {"label": "k-NN", "panels": [{"op": knn["id"], "label": "k-NN 15"}], "site": "b6", "filmstrip": filmstrip(kmed)},
        ],
        "kinds": ["site", "source", "joint"], "kind": "site", "site": "b6", "K": K,
        "dims": 2, "axes": "pooled:0", "norm": "site", "colour": {"by": "class"},
        "overlays": [{"set": "u8", "draw": "arrows", "from": "clean"}], "overlayMenuLabel": "Perturbation",
        "overlayMenu": ["u2", "u8", "t8"],
        "control": {"set": "n8", "label": "Sign noise ε 8/255", "mode": "side"},
        "arrows": 60, "filmstrip": filmstrip(med), "height": 330,
    })
    card = {
        "answer": "With untargeted PGD at 8/255, class switches span depth and displacement pulls away from sign noise after b6.",
        "numbers": [f"{token('switch.u8.median')} median persistent switch", f"{token('disp.u8.b9')} vs {token('disp.n8.b9')} same-class units at b9",
                    f"{token('transfer.u8.flip')} flip seed 1"],
    }
    if out.fig_path("thumb").is_file():
        card["thumb"] = f"assets/fig/x_{ID}_thumb.webp"
    out.chart("card", card)
    out.write()


if __name__ == "__main__":
    main()
