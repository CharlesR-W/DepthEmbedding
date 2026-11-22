"""Entry-generator helpers adapted from CoherentModes. No access to raw experiment data."""
from __future__ import annotations

import json
import math
import os
import re
import tempfile
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parent
DATA = ROOT / "data"
PALETTE = json.loads((SITE / "palette.json").read_text(encoding="utf-8"))
SITES = ["stem", *(f"b{i}" for i in range(1, 10)), "pool", "logits"]


def clean(x):
    """JSON-safe data: NumPy -> Python, non-finite chart values -> null (a gap)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return clean(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def fmt(value, digits=None, pct=False):
    """Format a finite number (pct takes a fraction), or pass a plain string through."""
    if isinstance(value, str):
        if digits is not None or pct:
            raise ValueError("digits and pct are only for numerical facts")
        return value
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("expected a finite number or a string")
    if type(value) is int and digits is None and not pct:
        return f"{value:,}"
    number = value * 100 if pct else value
    if digits is None:
        digits = 0 if pct else 3
    text = f"{number:,.{digits}f}"
    if float(text.replace(",", "")) == 0:
        text = text.lstrip("-")
    return text + ("%" if pct else "")


def _write_json(path, obj, **kw):
    text = json.dumps(obj, allow_nan=False, ensure_ascii=False, **kw) + "\n"
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as f:
        name = Path(f.name)
        try:
            f.write(text)
            f.close()
            os.replace(name, path)
        finally:
            name.unlink(missing_ok=True)


class Out:
    """Collect an entry's chart/spec/card records and independently checkable facts."""

    def __init__(self, xid):
        if not re.fullmatch(r"[a-z][a-z0-9-]*", xid):
            raise ValueError(f"experiment id must be a lower-case slug: {xid!r}")
        self.id, self.charts, self.facts = xid, {}, {}

    def chart(self, key, obj):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", key) or key in self.charts or not isinstance(obj, dict):
            raise ValueError(f"expected a unique JSON object key, got {key!r}")
        self.charts[key] = clean(obj)
        return key

    def fact(self, name, value, digits=None, pct=False, *, tier, source=None):
        """Register a raw value and display string. The tier is R (raw), P (per case), or S (summary)."""
        if not re.fullmatch(r"[a-z0-9_]+(?:\.[a-z0-9_]+)*", name) or name in self.facts:
            raise ValueError(f"expected a unique lower-case dotted fact name, got {name!r}")
        if tier not in ("R", "P", "S"):
            raise ValueError(f"{name}: expected tier R, P, or S")
        value = clean(value)
        text = fmt(value, digits, pct)
        self.facts[name] = {"value": value, "text": text, "tier": tier,
                            **({"source": source} if source is not None else {})}
        return text

    def fig_path(self, name, suffix=".webp"):
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", name) or suffix not in (".webp", ".png", ".svg"):
            raise ValueError(f"invalid figure filename: {name}{suffix}")
        p = SITE / "assets" / "fig" / f"x_{self.id}_{name}{suffix}"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def write(self):
        dest = SITE / "assets" / "x"
        dest.mkdir(parents=True, exist_ok=True)
        _write_json(dest / f"{self.id}.json", self.charts, separators=(",", ":"))
        _write_json(dest / f"{self.id}.facts.json", self.facts, indent=2)


def lines(x, series, xlabel, ylabel, xscale="linear", yrange=(0, 1), colors="categorical", **kw):
    return dict(x=x, xlabel=xlabel, ylabel=ylabel, xscale=xscale, yrange=yrange, series=series, colors=colors, **kw)


def heat(z, rows, cols, zlabel, rowlabel, collabel, **kw):
    kw.setdefault("zrange", [0, 1])
    return dict(rows=rows, cols=cols, z=z, zlabel=zlabel, rowlabel=rowlabel, collabel=collabel, **kw)


def bars(categories, series, ylabel, xlabel="", **kw):
    return dict(categories=categories, series=series, ylabel=ylabel, xlabel=xlabel, **kw)


def dots(rows, series, xlabel, **kw):
    return dict(rows=rows, series=series, xlabel=xlabel, **kw)


def hist(edges, series, xlabel, ylabel, norm="count", **kw):
    return dict(edges=edges, series=series, xlabel=xlabel, ylabel=ylabel, norm=norm, **kw)
