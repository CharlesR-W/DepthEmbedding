"""Append the pinned O1 inversion edit overlays to an already-packaged P2 pr300 operator.

Run via `python site/xsite/ops.py --entry inv_edits`; for a clean build, package inv first.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np

if __package__:
    from .opdata import read_array, read_operator, require, source_from_joint, write_overlay
    from .ops import write_sprite
else:
    from opdata import read_array, read_operator, require, source_from_joint, write_overlay
    from ops import write_sprite

ROOT = Path(__file__).resolve().parents[2]
VOL = "crw-depth-invert-results"
METHODS = {
    "resid": ("residual", "Residual guess"),
    "resid+match": ("residual-match", "Residual + matching"),
    "match": ("match", "Euclidean matching from a"),
    "rand~resid": ("random", "Random, sized to residual guess"),
}


def package(inputs, root):
    rec = inputs.manifest["stretch"]["inv_edits"]
    op_id = rec["op"]
    raw_op = inputs.bundle(op_id)
    op_path = root / "ops" / op_id / "op.json"
    op = read_operator(op_path)
    stencil = lambda y: source_from_joint(y, op["c"], op["t"], op["vals"])
    held = inputs.value(f"{VOL}/splits.npz:H")
    clean_file = inputs.file(inputs.manifest["overlays"][f"{op_id}/H"]["file"])
    with np.load(clean_file, allow_pickle=False) as z:
        clean_y = z["y"][:, :256, :op["K"]]
        clean_s = z["onestep"][:, :256, :op["K"]]
    with np.load(inputs.file(f"{VOL}/acts/c10/meta.npz"), allow_pickle=False) as a:
        images = a["images"][held[:256]]
        labels = a["labels"][held[:256]]
        pred = a["logits"][held[:256]].argmax(axis=1)

    # Validate both raw files against the same signed operator before publishing either.
    sources = {}
    for site, pin in rec["files"].items():
        path = ROOT / pin["file"]
        blob = path.read_bytes()
        require(hashlib.sha256(blob).hexdigest() == pin["sha256"], f"{path}: O1 hash mismatch")
        with np.load(io.BytesIO(blob), allow_pickle=False) as z:
            sources[site] = {k: z[k] for k in z.files}

    for site, z in sources.items():
        pin = rec["files"][site]
        row = op["chain"].index(site)
        meta = json.loads(str(z["meta"]))
        require(meta["modes_sha256"] == raw_op["bundle"][raw_op["file"]]
                and meta["sites"] == op["chain"] and meta["edit_site"] == site
                and np.allclose(meta["vals"], op["vals"], rtol=0, atol=1e-12), f"{site}: O1 operator identity mismatch")
        require(meta["m_full"] == 256 and z["pair_pos"].shape == (64, 2) and z["pair_ok"].all(),
                f"{site}: unexpected O1 slice")
        a, b = z["pair_pos"].T
        require(np.array_equal(a, np.arange(len(a))) and (pred[a] != pred[b]).all(), f"{site}: O1 pairing mismatch")
        for key, pos in (("y_a", a), ("y_b", b)):
            np.testing.assert_allclose(z[key], clean_y[:, pos], rtol=0, atol=3e-9)
        method = {str(m): i for i, m in enumerate(z["methods"])}
        sa, sb = stencil(z["y_a"]), stencil(z["y_b"])
        np.testing.assert_allclose(sa[row], z["s_edit"][method["none"]], rtol=0, atol=4e-9)
        np.testing.assert_allclose(sb[row], z["s_edit"][method["swap"]], rtol=0, atol=4e-9)
        source_patched = np.stack([stencil(y) for y in z["y_patched"]])
        np.testing.assert_allclose(source_patched[:, row], z["s_edit"], rtol=0, atol=1e-14)
        np.testing.assert_allclose(sb[row], clean_s[row, b], rtol=0, atol=4e-9)

        provenance = dict(input=f"inputs.json#stretch/inv_edits/files/{site}", sha256=pin["sha256"], code=rec["producer"])
        def put(suffix, title, pos, y, s, *, pair=None, index=None):
            sprite = f"inv-edit-{site}-{'a' if pair else suffix}.webp"
            if pair is None:
                sprite_path = write_sprite(root, sprite, images[pos], cols=8)
            else:
                sprite_path = a_sprite
            scalar = {}
            if index is not None:
                scalar = {key: dict(label=label, values=np.tile(z[raw][index], (len(op["chain"]), 1)).tolist(), range=limits)
                          for key, label, raw, limits in (("pb", "Patched probability of b’s class", "p_b", [0, 1]),
                                                          ("size", "Edit size (full swap = 1)", "size", [0, 1]))}
            set_id = f"edit-{site}-{suffix}"
            directory = op_path.parent / "ov"
            desc = write_overlay(root, op, set_id, coords=dict(joint=y, source=s), label=title,
                                 dataset="cifar10-test", ids=held[pos], pos=a, sample_of=256,
                                 sample_rule="first 64 valid E4 pairs in the held-out batch's original order",
                                 meta=dict(label=labels[pos].tolist(), pred=pred[pos].tolist() if pair is None else None,
                                           scalars=scalar), provenance=provenance, pair=pair,
                                 sprite=sprite_path, sprite_cols=8)
            for kind, exact in [("joint", y), ("source", s)]:
                spec = desc["kinds"][kind]
                err = np.abs(read_array(directory, spec) - exact)
                require(np.all(err <= np.asarray(spec["scale"])[:, None, :] * 0.50001), f"{set_id}/{kind}: quantization mismatch")
            return sprite_path

        a_sprite = put("a", "Original a", a, z["y_a"], sa)
        put("b", "Target b", b, z["y_b"], sb)
        for raw, (suffix, label) in METHODS.items():
            i = method[raw]
            put(suffix, label, a, z["y_patched"][i], source_patched[i], pair=f"edit-{site}-a", index=i)
        print(f"{op_id}/{site}: O1 signed anchors, source stencil, and shipped quantization checked")
