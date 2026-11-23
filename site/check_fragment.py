"""Check one Experiments fragment before building: python site/check_fragment.py <id>."""
from __future__ import annotations

import argparse
import re

from build_x import BuildError, Catalog, SITEX, asset_path, manifest, read_entry


def check(entry):
    _, parsed, specs, card = read_entry(entry)
    catalog = Catalog(SITEX, bool(specs))
    for key, spec in specs:
        catalog.check(spec, f"assets/x/{entry['id']}.json#{key}")
    errors, warnings = [], []
    ids = set()
    for tag, attrs, line in parsed.tags:
        if tag in ("script", "style", "link"):
            errors.append(f"line {line}: no <{tag}> in fragments")
        if tag != "figure" and ("data-live" in attrs or "data-emb" in attrs):
            errors.append(f"line {line}: data-live and data-emb belong on figures")
        if "id" in attrs:
            ident = attrs["id"]
            if ident in ids:
                errors.append(f"line {line}: duplicate id {ident!r}")
            ids.add(ident)
        if tag == "img":
            src = attrs.get("src") or ""
            try:
                if not src.startswith("data:image/") and not asset_path(src, f"line {line}").is_file():
                    errors.append(f"line {line}: missing image {src!r}")
            except BuildError as exc:
                errors.append(str(exc))
            if "alt" not in attrs:
                errors.append(f"line {line}: image needs alt text (empty for decorative images)")
        if re.search(r"#[0-9a-fA-F]{3,8}\b|rgba?\(", attrs.get("style") or ""):
            warnings.append(f"line {line}: literal inline colour; use site palette tokens")
    for _, line, caption in parsed.figures:
        if not caption.strip():
            errors.append(f"line {line}: figure needs a non-empty figcaption")
    for name in ("x-scope", "x-answer"):
        if not any(name in (a.get("class") or "").split() for _, a, _ in parsed.tags):
            warnings.append(f"missing .{name}")
    if card is None:
        warnings.append("missing at-a-glance card")
    elif not card.get("thumb"):
        warnings.append("missing card thumbnail (capture after building)")
    return errors, warnings


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("id", help="experiment id without the x- section prefix")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    args = ap.parse_args()
    try:
        entry = next((e for e in manifest() if e["id"] == args.id), None)
        if entry is None:
            raise BuildError(f"unknown experiment {args.id!r}")
        errors, warnings = check(entry)
    except (BuildError, OSError) as exc:
        errors, warnings = [str(exc)], []
    for level, messages in (("error", errors), ("warning", warnings)):
        for msg in messages:
            print(f"{args.id}: {level}: {msg}")
    print(f"{args.id}: {len(errors)} errors, {len(warnings)} warnings")
    return int(bool(errors or (args.strict and warnings)))


if __name__ == "__main__":
    raise SystemExit(main())
