"""Build the static Depth embedding site from the through-depth data.

    uv run --with numpy --with pillow python site/build_site.py [--tag r20s0b]
    python3 -m http.server 8765 --bind 127.0.0.1 -d build/site          # then open http://127.0.0.1:8765/

Reads (all under data/, which is not in git; see README.md):
  data/<tag>/meta.json, probe.npz, curves.json   reference network (ResNet-20 seed 0): steps, sites, probe images, accuracy
  data/<tag>/spacetime/st_<step>_0.npz           its through-depth units (spacetime_modal.py)
  data/stnets/<net>/st_<step>_0.npz, info.json   the other networks' units (spacetime_nets_modal.py, spacetime_vgg_modal.py)
  data/stbase/<net>/{pca,umap}_<step>.npz        per-site PCA / UMAP baselines (baseline_embed_modal.py)
  data/wavelets/joint_dw_<step>.{npz,json}       diffusion-wavelet scaling functions of the reference operator (wavelets/stage_joint_dw.py)
and site/ sources. Writes build/site/ (deploy root, .nojekyll):
  d/meta.json     steps, sites, classes, palette, probe digests, and descriptions of d/st, d/stn, d/stb (js/embedding.js netList)
  d/labels.bin    i8 (N) CIFAR labels; d/sprite.webp: the 2,000 probe images, 50 per row
  d/st, d/stn/<net>, d/stb/<net>   see Builder.spacetime and Builder.stnets
  d/stw/<net>                      see Builder.wavelets
The export code is the Depth embedding part of CoherentModes' explore/cifar_coherent/site/build_site.py, unchanged in
what it writes: d/st, d/stn and d/stb are byte-identical to that build's.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

import build_x
from xsite.opdata import ids_bytes, probe_labels_bytes

HERE = Path(__file__).resolve().parent          # site/
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "build" / "site"
ST_FORMAT = "st-v1"            # bump when d/st changes layout (SPEC_SPACETIME.md section 6)
ST_RULE = 0                    # kernel rule 0 (kNN graph, k = 15) only
STN_FSEL = (3,)                # couplings shipped for the other networks (index into the factors: t/t* = 1)
STB_K = 16                     # PCA baseline dimensions shipped
DW_STEP = 64000                # the wavelet view: reference network, final checkpoint (wavelets/SPEC_DW.md, "Viewer")
ST_LOW = 0.01                  # slices holding less of their mode's squared norm are drawn grey (not a direction)


def log(*a):
    print(*a, flush=True)


def write(path: Path, data: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def pack_nibbles(lab: np.ndarray) -> np.ndarray:
    """lab int8 (..., N) in -1..14 -> uint8 (..., ceil(N/2)); value+1 stored, low nibble = even index."""
    v = (lab.astype(np.int16) + 1).astype(np.uint8)
    if v.shape[-1] % 2:
        v = np.concatenate([v, np.zeros(v.shape[:-1] + (1,), np.uint8)], -1)
    return (v[..., 0::2] | (v[..., 1::2] << 4)).astype(np.uint8)


ST_SETS = [{"k": k, "readout": r} for k in (5, 10) for r in ("a", "s")]      # order of the label planes


def st_q8(v):      # int16 -> int8 as in coarse(): a non-zero entry never rounds to 0 (signs stay exact)
    q = np.round(v.astype(np.float64) * 127.0 / 32767.0)
    return np.clip(np.where((q == 0) & (v != 0), np.sign(v), q), -127, 127).astype(np.int8)


def st_by_size(lab):   # (L, N) labels with -1; renumber sets by decreasing size over all sites (ties: first seen)
    flat = lab.ravel()
    ok = flat >= 0
    if not ok.any():
        return lab.copy()
    u, first_i, cnt = np.unique(flat[ok], return_index=True, return_counts=True)
    order = u[np.lexsort((first_i, -cnt))]
    m_ = np.full(int(flat.max()) + 1, -1, np.int16); m_[order] = np.arange(len(order))
    out = np.where(lab >= 0, m_[np.clip(lab, 0, None)], -1)
    return out.astype(np.int8)


def net_label(net: str) -> str:      # "resnet32-relu-seed1" -> "ResNet-32 seed 1"; "vgg16-bn-relu-seed0" -> "VGG-16-BN seed 0"
    m = re.match(r"(resnet|vgg)(\d+)(-bn)?-relu-seed(\d+)", net)
    if not m:
        return net
    return f"{'ResNet' if m[1] == 'resnet' else 'VGG'}-{m[2]}{'-BN' if m[3] else ''} seed {m[4]}"


def net_short(net: str) -> str:
    m = re.match(r"(resnet|vgg)(\d+)(-bn)?-relu-seed(\d+)", net)
    return f"{'R' if m[1] == 'resnet' else 'VGG'}{m[2]} s{m[4]}" if m else net


def st_export_step(z, step, root, variants, sizes, rule=None, fsel=None):
    """One checkpoint's spacetime_stage npz -> <root>/<step>_<variant>.bin.gz for each requested variant (own | none |
    paper) and <root>/<step>.json.gz (see Builder.spacetime for the layout). Adds gzip sizes to `sizes`; returns files.
    fsel: coupling indices to keep (own / none only; None = all)."""
    rule = ST_RULE if rule is None else rule
    g4 = lambda x: None if x is None or not np.isfinite(x) else float(f"{float(x):.4g}")
    L4 = lambda a: g4(a) if np.ndim(a) == 0 else [L4(x) for x in a]
    K = z["vecs"].shape[2]
    summary = {"step": step, "rule": rule, "tstar": {"own": g4(z["tstar_own"]), "none": g4(z["tstar_none"])}, "variants": {}}
    n_files = 0
    for v, si in (("own", 0), ("none", 1), ("paper", None)):
        if v not in variants:
            continue
        p = "" if si is not None else "paper_"
        cs = slice(None) if fsel is None else list(fsel)
        pick = (lambda a: np.asarray(a[si])[cs]) if si is not None else (lambda a: a[None])     # leading coupling axis
        vecs = pick(z[p + "vecs"])                                               # (C', K, L, N) int16
        labs = np.stack([np.stack([st_by_size(x) for x in pick(z[f"{p}labels_{s['readout']}{(5, 10).index(s['k'])}"])])
                         for s in ST_SETS])                                      # (4, C', L, N)
        assert labs.max() < 15, "set labels outside 0..14"
        blob = st_q8(vecs).tobytes() + pack_nibbles(labs).tobytes()
        gz = gzip.compress(blob, compresslevel=6, mtime=0)
        write(root / f"{step}_{v}.bin.gz", gz)
        sizes[v] = sizes.get(v, 0) + len(gz); n_files += 1
        summary["variants"][v] = {
            "t": L4(np.asarray(z["t"][si])[cs]) if si is not None else None,
            "vals": L4(pick(z[p + "vals"])[..., :K]),
            "weight": L4(pick(z[p + "weight"])), "adj_cos": L4(pick(z[p + "adj_cos"])),
            "retention": L4(pick(z[p + "retention"])), "own_share": L4(pick(z[p + "own_share"])),
            "occupancy": L4(pick(z[p + "occupancy"])),
            "scalar_weight": L4(np.asarray(z["scalar_weight"][si])[cs]) if si is not None else None,
            "sets": [{"groups": L4(pick(z[f"{p}set_{s['readout']}{(5, 10).index(s['k'])}_groups"])),
                      "churn": L4(pick(z[f"{p}set_{s['readout']}{(5, 10).index(s['k'])}_churn"]))} for s in ST_SETS],
            "link": L4(pick(z[p + "link"])),
        }
    gz = gzip.compress(json.dumps(summary, separators=(",", ":"), allow_nan=False).encode(), compresslevel=9, mtime=0)
    write(root / f"{step}.json.gz", gz)
    sizes["summary"] = sizes.get("summary", 0) + len(gz)
    return n_files + 1


class Builder:
    def __init__(self, tag: str, out: Path = OUT, viewer_from: Path | None = None, strict: bool = False):
        requested_out = out.absolute()
        self.out = out.resolve()
        if (requested_out.is_symlink() or self.out == ROOT or self.out in ROOT.parents or self.out.is_file()
                or (self.out.is_dir() and any(self.out.iterdir()) and not (self.out / ".nojekyll").is_file())):
            sys.exit(f"unsafe output directory: {out} (expected an empty directory or a previous site build)")
        self.strict = strict
        self.viewer_from = (viewer_from / "d").resolve() if viewer_from is not None else None
        if self.viewer_from is not None:
            if self.out == self.viewer_from or self.out in self.viewer_from.parents or self.viewer_from in self.out.parents:
                sys.exit("--out must not overlap the --viewer-from build's d/ directory")
            for name in ("meta.json", "labels.bin", "sprite.webp"):
                if not (self.viewer_from / name).is_file():
                    sys.exit(f"--viewer-from: missing {self.viewer_from / name}")
            build_x.viewer_probe(self.viewer_from)
        self.tag, self.src = tag, DATA / tag
        if self.viewer_from is None:
            if not (self.src / "meta.json").exists():
                sys.exit(f"no {self.src}/meta.json")
            m = self.meta = json.loads((self.src / "meta.json").read_text())
            self.steps, self.sites, self.modes, self.n = m["steps"], m["sites"], m["modes"], m["n"]
            self.S = len(self.sites)
        self.d = self.out / "d"
        self.st_meta = self.stn_meta = self.dw_meta = None

    def copy_static(self):
        for name in ["css", "js", "vendor"]:
            shutil.copytree(HERE / name, self.out / name, dirs_exist_ok=True)
        pal = self.palette = json.loads((HERE / "palette.json").read_text())
        lines = [":root {"]
        for group in ("classes", "sets"):
            lines += [f"  --c-{group[:-2] if group == 'classes' else 'set'}-{i}: {c};" for i, c in enumerate(pal[group])]
        lines += [f"  --c-{k}: {pal['diverging'][k]};" for k in ("neg", "mid", "pos")]
        lines += [f"  --c-seq-{i}: {c};" for i, c in enumerate(pal["sequential"])]
        lines += [f"  --c-ord-{i}: {c};" for i, c in enumerate(pal["ordinal"])]
        lines += [f"  --{k}: {pal[k]};" for k in ("unassigned", "accent", "bg", "surface", "fg", "mute", "line")]
        write(self.out / "css" / "tokens.css", ("\n".join(lines) + "\n}\n").encode())

    def probe(self):
        with np.load(self.src / "probe.npz") as p:
            ids = ids_bytes(p["idx"])
            labels = probe_labels_bytes(p["labels"])
            imgs = p["images"]
        if len(ids) != 8 * self.n or len(labels) != self.n or len(imgs) != self.n:
            sys.exit(f"{self.src / 'probe.npz'}: probe IDs, labels and images must match meta.json's n={self.n}")
        self.probe_ids_sha256 = hashlib.sha256(ids).hexdigest()
        self.probe_labels_sha256 = hashlib.sha256(labels).hexdigest()
        write(self.d / "labels.bin", labels)
        cols = 50
        sheet = np.zeros((int(np.ceil(self.n / cols)) * 32, cols * 32, 3), np.uint8)
        for i in range(self.n):
            r, c = divmod(i, cols)
            sheet[r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = imgs[i]
        buf = io.BytesIO()
        Image.fromarray(sheet).save(buf, format="WEBP", quality=92, method=4)
        write(self.d / "sprite.webp", buf.getvalue())
        self.sprite_cols = cols

    def spacetime(self):
        """Kernel rule ST_RULE. From data/<tag>/spacetime/st_<step>_<rule>.npz (spacetime_stage.py)
        write the reference network's units, fetched lazily per checkpoint and variant:
          st/<step>_<variant>.bin.gz   variant own (relative site units) | none (native walk units) | paper (the
                                       paper's transition matrix). i8 vecs (C, K, L, N): each site's signed unit slice
                                       of each mode, round(127 u / max|u|) with a non-zero floor (C = 6 couplings, 1
                                       for paper); then u4 labels (4, C, L, ceil(N/2)) of the post-hoc set readouts in
                                       the order of meta.spacetime.sets, value + 1 (0 = no label), low nibble = even
                                       image; set numbers renumbered by decreasing size over all sites (display only)
          st/<step>.json.gz            per variant: t, values, depth weight, neighbour cosines, retention, own share,
                                       occupancy, scalar control, and per-site set scores (4 significant digits)
        The probe order is verified against probe.npz (the build fails on a mismatch)."""
        sdir, root = self.src / "spacetime", self.d / "st"
        files = {t: sdir / f"st_{s}_{ST_RULE}.npz" for t, s in enumerate(self.steps)}
        if not all(f.exists() for f in files.values()):
            shutil.rmtree(root, ignore_errors=True)
            log(f"spacetime: no data/<tag>/spacetime/st_<step>_{ST_RULE}.npz for every checkpoint: through-depth colouring not built")
            return
        probe_idx = np.load(self.src / "probe.npz")["idx"]
        for f in files.values():          # every unit must use the site's probe images in the site's order
            with np.load(f) as z:
                if not np.array_equal(z["probe_idx"], probe_idx):
                    sys.exit(f"{f}: probe_idx differs from {self.src / 'probe.npz'} (idx); the through-depth data do not "
                             f"match the site's probe order")
        stamp = f"{ST_FORMAT}|{ST_LOW}|{(self.src / 'probe.npz').stat().st_mtime_ns}|" + "|".join(
            f"{f.stat().st_mtime_ns}|{f.stat().st_size}" for f in files.values())
        index = root / "index.json"
        if index.exists():
            old = json.loads(index.read_text())
            if old.get("stamp") == stamp:
                self.st_meta = old["meta"]
                log(f"spacetime add-on: unchanged, kept ({self.st_meta['gz_bytes'] / 1e6:.1f} MB)")
                return
        t0 = time.time()
        shutil.rmtree(root, ignore_errors=True)
        sets = ST_SETS
        sizes, n_files, K = {"own": 0, "none": 0, "paper": 0, "summary": 0}, 0, None
        for t, f in files.items():
            step = self.steps[t]
            z = np.load(f)
            C, K, L, n = z["vecs"].shape[1:]
            assert (L, n) == (self.S, self.n), f"{f}: vecs {z['vecs'].shape} do not match {self.S} sites x {self.n} images"
            assert C == len(z["factors"]) and z["vecs"].shape[0] == 2, f"{f}: unexpected vecs shape {z['vecs'].shape}"
            n_files += st_export_step(z, step, root, ("own", "none", "paper"), sizes)
            log(f"  spacetime {step}: {time.time() - t0:.0f} s")
        total = sum(sizes.values())
        self.st_meta = {
            "format": ST_FORMAT, "rule": ST_RULE, "rule_name": self.modes[ST_RULE], "kernel": "cos", "steps": self.steps,
            "K": int(K), "L": self.S, "factors": [float(x) for x in z["factors"]], "low_weight": ST_LOW,
            "variants": [{"id": "own", "label": "relative site units", "C": int(C)},
                         {"id": "none", "label": "native walk units", "C": int(C)},
                         {"id": "paper", "label": "paper transition", "C": 1}],
            "sets": sets, "readouts": {"a": "shape-aligned", "s": "shared coordinates"},
            "quant": "int8 round(127 u / max|u|), non-zero floor; u = the unit slice f_l / |f_l| of each mode at each site",
            "gz_bytes": total, "gz_bytes_by_part": sizes,
        }
        write(index, json.dumps({"stamp": stamp, "meta": self.st_meta}).encode())
        log(f"spacetime add-on ({time.time() - t0:.0f} s): {total / 1e6:.1f} MB gzipped in {n_files} files "
            f"(own {sizes['own'] / 1e6:.1f}, none {sizes['none'] / 1e6:.1f}, paper {sizes['paper'] / 1e6:.1f}, "
            f"summaries {sizes['summary'] / 1e6:.2f} MB)")

    def stnets(self):
        """Other networks' through-depth units (data/stnets/<net>/st_<step>_0.npz + info.json, spacetime_nets_modal.py and
        spacetime_vgg_modal.py) -> d/stn/<net>/ in the d/st format, variant own (relative site units) only, couplings
        STN_FSEL only (the site size cap). Per-site PCA / UMAP baselines (data/stbase/<net>/{pca,umap}_<step>.npz,
        baseline_embed_modal.py) -> d/stb/<net>/<step>_{pca,umap}.bin.gz:
          pca   i8 (L, KB, N): coordinate k at site l = q * scale[l][k] (scale in <step>_base.json.gz)
          umap  i8 (L, 2, N): same, scale per (site, axis)
        and <step>_base.json.gz {sites, evals (L, KB), trace (L), scale_pca (L, KB), scale_umap (L, 2), init}.
        Probe order is verified (the build fails on a mismatch)."""
        sroot, broot = DATA / "stnets", DATA / "stbase"
        out_n, out_b = self.d / "stn", self.d / "stb"
        shutil.rmtree(out_n, ignore_errors=True); shutil.rmtree(out_b, ignore_errors=True)
        self.stn_meta = None
        if not self.st_meta:
            return
        probe_idx = np.load(self.src / "probe.npz")["idx"]
        t0, sizes, nets, base = time.time(), {}, [], {}
        for d in sorted(p for p in sroot.glob("*") if (p / "info.json").exists()) if sroot.exists() else []:
            info = json.loads((d / "info.json").read_text())
            steps = [s for s in info["steps"] if (d / f"st_{s}_{ST_RULE}.npz").exists()]
            if not steps:
                continue
            for step in steps:
                with np.load(d / f"st_{step}_{ST_RULE}.npz") as z:
                    if not np.array_equal(z["probe_idx"], probe_idx):
                        sys.exit(f"{d}/st_{step}_{ST_RULE}.npz: probe order differs from {self.src / 'probe.npz'}")
                    assert z["vecs"].shape[3] == len(info["sites"]), f"{d}: {z['vecs'].shape} vs {len(info['sites'])} sites"
                    st_export_step(z, step, out_n / d.name, ("own",), sizes, fsel=STN_FSEL)
                    K = int(z["vecs"].shape[2])
            nets.append({"id": d.name, "label": net_label(info["net"]), "short": net_short(info["net"]), "root": f"d/stn/{d.name}",
                         "sites": list(info["sites"]), "steps": steps, "K": K, "acc": info.get("test_acc"), "nsets": len(ST_SETS),
                         "variants": [{"id": "own", "label": "relative site units", "C": len(STN_FSEL)}], "fidx": list(STN_FSEL)})
        for d in sorted(broot.glob("*")) if broot.exists() else []:
            steps = sorted(int(p.stem.split("_")[1]) for p in d.glob("pca_*.npz") if (d / f"umap_{p.stem.split('_')[1]}.npz").exists())
            for step in steps:
                with np.load(d / f"pca_{step}.npz") as zp, np.load(d / f"umap_{step}.npz") as zu:
                    for z in (zp, zu):
                        if not np.array_equal(z["probe_idx"], probe_idx):
                            sys.exit(f"{d}: probe order differs from {self.src / 'probe.npz'}")
                    rec = {"sites": [str(x) for x in zp["sites"]], "evals": np.round(zp["evals"].astype(float), 6).tolist(),
                           "trace": np.round(zp["trace"].astype(float), 6).tolist(), "init": [str(x) for x in zu["init"]]}
                    blob = b""
                    for key, arr in (("pca", zp["coords"][:, :, :STB_K]), ("umap", zu["coords"])):
                        a = np.asarray(arr, np.float64).transpose(0, 2, 1)                 # (L, k, N)
                        sc = np.abs(a).max(2) / 127.0; sc[sc == 0] = 1.0
                        q = np.clip(np.round(a / sc[:, :, None]), -127, 127).astype(np.int8)
                        rec[f"scale_{key}"] = [[float(f"{x:.6g}") for x in row] for row in sc]
                        gz = gzip.compress(q.tobytes(), compresslevel=6, mtime=0)
                        write(out_b / d.name / f"{step}_{key}.bin.gz", gz); sizes["base"] = sizes.get("base", 0) + len(gz)
                    gz = gzip.compress(json.dumps(rec, separators=(",", ":")).encode(), compresslevel=9, mtime=0)
                    write(out_b / d.name / f"{step}_base.json.gz", gz); sizes["base"] += len(gz)
            if steps:
                base[d.name] = {"root": f"d/stb/{d.name}", "steps": steps, "K": STB_K}
        self.export_x1(probe_idx, out_n, nets, sizes)
        if not nets and not base:
            return
        self.stn_meta = {"nets": nets, "base": base, "gz_bytes": sum(sizes.values()),
                         "note": "other networks: variant own, couplings " + ", ".join(f"{self.st_meta['factors'][i]:g}" for i in STN_FSEL) + " x t* only"}
        log(f"depth embedding: {len(nets)} other networks, baselines for {len(base)}: {sum(sizes.values()) / 1e6:.1f} MB gzipped "
            f"({', '.join(f'{k} {v / 1e6:.1f}' for k, v in sizes.items())}) ({time.time() - t0:.0f} s)")


    def export_x1(self, probe_idx, out_n, nets, sizes):
        """Ship the full-chain X1 modes as st-v1 unit slices, with no label planes."""
        from xsite.validate import Inputs, check_probe, validate_operator

        fetch = DATA / "fetch"
        if not (fetch / "READY.json").exists() and not self.strict:
            log("X1 viewer networks not built: no staged data/fetch/READY.json")
            return
        inputs = Inputs(fetch)
        pipelines = {
            "c100": ("CIFAR-100", "100 fine labels"),
            "svhn": ("SVHN", "10 digit labels; crops, no flips"),
            "in32": ("ImageNet-32", "1,000 labels"),
            "c100c": ("CIFAR-100 coarse", "same CIFAR-100 images, 20 superclass labels"),
            "anim": ("CIFAR-10 animals", "6 animal classes only"),
            "veh": ("CIFAR-10 vehicles", "4 vehicle classes only"),
            "c10x": ("CIFAR-10 (retrained)", "10 labels; X1 trainer control"),
            "rand": ("CIFAR-10 (shuffled labels)", "10 classes; shuffled training labels, no augmentation; test uses true labels"),
        }
        labels = np.fromfile(self.d / "labels.bin", dtype=np.int8)
        for train_id, (train, detail) in pipelines.items():
            seeds = (0,) if train_id in {"c10x", "rand"} else (0, 1)
            for seed in seeds:
                net = f"{train_id}-s{seed}"
                op_id = f"x1-{net}-full"
                op = validate_operator(inputs, op_id)
                check_probe(op["ids"], op["labels"], probe_idx, labels)
                if op["chain"] != self.sites:
                    raise ValueError(f"{op_id}: full chain does not match reference sites")
                info_key = f"crw-depth-xp-results/nets/{net}/info.json"
                pin = inputs.manifest["ops"][op_id]["bundle"][info_key]
                info = json.loads(inputs.file(info_key, pin=pin).read_text())
                step = info["steps"]
                final = info["log"][-1]
                if info["name"] != net or final["step"] != step:
                    raise ValueError(f"{op_id}: final training step is inconsistent")

                y = op["joint"].transpose(2, 0, 1)  # (K,L,N); each slice has its own int8 scale
                mx = np.max(np.abs(y), axis=2, keepdims=True)
                q = np.round(127 * np.divide(y, mx, out=np.zeros_like(y), where=mx != 0)).astype(np.int8)
                q[(q == 0) & (y != 0)] = np.sign(y[(q == 0) & (y != 0)])
                weight = np.sum(y ** 2, axis=2)
                root = out_n / net
                blob = gzip.compress(q.tobytes(), compresslevel=6, mtime=0)
                write(root / f"{step}_own.bin.gz", blob)
                summary = {"step": step, "rule": ST_RULE, "tstar": {"own": op["tstar"], "none": None},
                           "variants": {"own": {"t": [op["t"]], "vals": [op["vals"].tolist()],
                                                "weight": [weight.tolist()], "sets": []}}}
                stats = gzip.compress(json.dumps(summary, separators=(",", ":"), allow_nan=False).encode(), mtime=0)
                write(root / f"{step}.json.gz", stats)
                sizes["x1"] = sizes.get("x1", 0) + len(blob) + len(stats)
                aug = "" if train_id in {"svhn", "rand"} else "; crops and horizontal flips"
                nets.append({"id": net, "group": "x1", "train": train, "label": f"{train} · ResNet-20 seed {seed}",
                             "short": f"{train} s{seed}", "root": f"d/stn/{net}", "sites": op["chain"], "steps": [step],
                             "K": y.shape[0], "acc": final["test_acc"], "nsets": 0, "sets": [],
                             "training": f"Trained on {train}: {detail}{aug}. Normalised by its own training-set channel statistics.",
                             "variants": [{"id": "own", "label": "relative site units", "C": 1}], "fidx": list(STN_FSEL)})

    def wavelets(self):
        """Diffusion-wavelet scaling functions of the reference network's own operator (L_t, scale own, t = t*, rule 0) at
        step DW_STEP, from data/wavelets/joint_dw_<step>.{npz,json} (wavelets/stage_joint_dw.py) -> d/stw/r20s0/:
          <step>_j<j>.bin.gz   i8 (L, S_j, N): value of scaling function s at site l, image i = q * scale[l][s] (raw values:
                               the functions are unit-norm over all L x N entries; one int8 scale per (site, function))
          <step>_dw.json.gz    per level: rank, shipped, cut-off, heat time, edge gap, scale (L, S_j), and per function its
                               id, QR seed (pivot image and site), absolute peak, depth profile and 90%-mass support
        Probe order is verified (the build fails on a mismatch). Skipped (with a log line) when the staged data are absent,
        unless --strict."""
        src = DATA / "wavelets" / f"joint_dw_{DW_STEP}"
        out = self.d / "stw" / "r20s0"
        shutil.rmtree(self.d / "stw", ignore_errors=True)
        if not (src.with_suffix(".npz").exists() and src.with_suffix(".json").exists()):
            if self.strict:
                sys.exit(f"--strict: no {src}.npz/.json (run wavelets/modal_joint_dw.py run, then wavelets/stage_joint_dw.py stage)")
            log("wavelets: no staged data/wavelets/joint_dw_*.npz: diffusion-wavelet view not built")
            return
        rec = json.loads(src.with_suffix(".json").read_text())
        probe_idx = np.load(self.src / "probe.npz")["idx"]
        g4 = lambda x: float(f"{float(x):.4g}")
        with np.load(src.with_suffix(".npz")) as z:
            if not np.array_equal(z["probe_idx"], probe_idx):
                sys.exit(f"{src}.npz: probe order differs from {self.src / 'probe.npz'}")
            with np.load(self.src / "spacetime" / f"st_{DW_STEP}_{ST_RULE}.npz") as st:
                ship = st["vals"][0, 3][:rec["checks"]["ship"]["K"]]
                if abs(float(st["t"][0, 3]) - float(z["t"])) > 1e-9 * float(z["t"]) or np.abs(z["vals"][:len(ship)] / ship - 1).max() > 1e-8:
                    sys.exit(f"{src}.npz: operator differs from the shipped reference modes (t or eigenvalues)")
            c = rec["checks"]
            bad = [k for k, ok in (("accepted by the stage", rec.get("accepted") is True and not rec.get("rejected_because")),
                                   ("inertia", c["inertia_ok"]), ("residual", c["resid_max"] < 1e-8), ("slice means", c["slice_mean_max"] < 1e-10),
                                   ("orthonormality", c["orth_err"] < 1e-9), ("modes", c["ship"]["min_abs_cos_heavy"] > 0.9999)) if not ok]
            for lv in rec["levels"]:      # the shipped functions themselves: orthonormal over all sites and images, positive at the seed
                f = np.asarray(z[f"phi{lv['level']}"], np.float64).reshape(-1, lv["shipped"])
                if np.abs(f.T @ f - np.eye(lv["shipped"])).max() > 1e-9:
                    bad.append(f"level {lv['rank']}: shipped functions not orthonormal")
                if not all(f[x["pivot_site"] * self.n + x["pivot_image"], i] > 0 for i, x in enumerate(lv["functions"])):
                    bad.append(f"level {lv['rank']}: a function is not positive at its seed")
            if bad:
                sys.exit(f"{src}.json: stage checks failed: {bad}")
            levels, size = [], 0
            for lv in rec["levels"]:
                j = lv["level"]
                a = np.asarray(z[f"phi{j}"], np.float64).transpose(0, 2, 1)         # (L, S, N)
                assert a.shape[0] == self.S and a.shape[2] == self.n and a.shape[1] == lv["shipped"], a.shape
                sc = np.abs(a).max(2) / 127.0; sc[sc == 0] = 1.0
                q = np.clip(np.round(a / sc[:, :, None]), -127, 127).astype(np.int8)
                assert np.abs(q * sc[:, :, None] - a).max() <= 0.5001 * sc.max(), "quantisation error above half a step"
                gz = gzip.compress(q.tobytes(), compresslevel=6, mtime=0)
                write(out / f"{DW_STEP}_j{j}.bin.gz", gz); size += len(gz)
                fns = [{"id": f["id"], "seed_site": f["pivot_site"], "seed_image": f["pivot_image"], "seed_cifar": f["pivot_cifar"],
                        "peak_site": f["peak_site"], "peak_image": f["peak_image"], "profile": [g4(x) for x in f["profile"]],
                        "n90": f["n90"]} for f in lv["functions"]]
                levels.append({"j": j, "target": lv["target"], "rank": lv["rank"], "shipped": lv["shipped"], "mu_c": g4(lv["mu_c"]),
                               "t": g4(lv["t"]), "mu_last": g4(lv["mu_last"]), "mu_next": g4(lv["mu_next"]),
                               "rel_gap": g4(lv["rel_gap"]), "scale": [[float(f"{x:.6g}") for x in row] for row in sc], "functions": fns})
        body = {"step": DW_STEP, "levels": levels}
        gz = gzip.compress(json.dumps(body, separators=(",", ":"), allow_nan=False).encode(), compresslevel=9, mtime=0)
        write(out / f"{DW_STEP}_dw.json.gz", gz); size += len(gz)
        op = rec["operator"]
        self.dw_meta = {"nets": {"r20s0": {"root": "d/stw/r20s0", "step": DW_STEP,
                                           "levels": [{k: lv[k] for k in ("j", "target", "rank", "shipped", "mu_c", "t")} for lv in levels]}},
                        "tstar": g4(op["t_star"]), "ship": rec["ship"], "k_solved": op["k_solved"],
                        "quant": "int8 per (site, function): value = q * scale[l][s]; functions unit-norm over all sites and images",
                        "gz_bytes": size}
        log(f"wavelets: {len(levels)} levels (ranks {', '.join(str(lv['rank']) for lv in levels)}), {size / 1e6:.2f} MB gzipped")

    # ------------------------------------------------------------------ meta + html
    def write_meta(self):
        acc = [None] * len(self.steps)
        p = self.src / "curves.json"
        if p.exists():
            by = {c["step"]: c for c in json.loads(p.read_text()).get("checkpoints", [])}
            acc = [by.get(s, {}).get("val_acc") for s in self.steps]
        meta = {"tag": self.tag, "steps": self.steps, "sites": self.sites, "n": self.n, "classes": self.meta["classes"],
                "spacetime": self.st_meta, **({"stnets": self.stn_meta} if self.stn_meta else {}),
                **({"wavelets": self.dw_meta} if self.dw_meta else {}),
                "val_acc": acc, "sprite_cols": self.sprite_cols, "palette": self.palette,
                "probe_ids_sha256": self.probe_ids_sha256, "probe_labels_sha256": self.probe_labels_sha256}
        write(self.d / "meta.json", json.dumps(meta, separators=(",", ":"), allow_nan=False).encode())

    def assemble_html(self):
        page = (HERE / "index.html").read_text()
        for key, val in (("{{EMBEDDING}}", (HERE / "embedding_tab.html").read_text()),
                         ("{{CONTENT_METHOD}}", (HERE / "content" / "method.html").read_text()),
                         ("{{CONTENT_EXPERIMENTS}}", build_x.experiments_html(self)),
                         ("{{BUILD_DATE}}", dt.date.today().isoformat())):
            page = page.replace(key, val)
        left = re.findall(r"\{\{[^}]*\}\}", page)
        if left:
            sys.exit(f"index.html: unresolved placeholders {sorted(set(left))}")
        build_x.check_local_resources(page, "index.html")
        write(self.out / "index.html", page.encode())

    def run(self):
        t0 = time.time()
        if self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True)
        (self.out / ".nojekyll").write_text("")  # mark this directory as ours, even if a later check fails
        self.copy_static()
        if self.viewer_from is not None:
            shutil.copytree(self.viewer_from, self.d)
        else:
            self.probe()
            self.spacetime()
            if not self.st_meta:
                sys.exit("no through-depth data for the reference network: nothing to show")
            self.stnets()
            self.wavelets()
            self.write_meta()
        build_x.build_experiments(self)
        for folder in ("css", "vendor", "assets"):
            for css in (self.out / folder).rglob("*.css"):
                build_x.check_css(css.read_text(encoding="utf-8"), css)
        self.assemble_html()
        files = [p for p in self.out.rglob("*") if p.is_file()]
        tot = sum(p.stat().st_size for p in files)
        log(f"wrote {self.out}: {tot / 1e6:.1f} MB in {len(files)} files ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tag", default="r20s0b", help="data tag of the reference network (data/<tag>/)")
    ap.add_argument("--out", type=Path, default=OUT, help="output directory, replaced on each build (default: build/site)")
    ap.add_argument("--viewer-from", type=Path, help="copy d/ from this existing build instead of exporting")
    ap.add_argument("--strict", action="store_true", help="fail if any experiment fragment or release input is missing")
    a = ap.parse_args()
    try:
        Builder(a.tag, a.out, a.viewer_from, a.strict).run()
    except (build_x.BuildError, OSError) as e:
        ap.exit(1, f"build failed: {e}\n")
