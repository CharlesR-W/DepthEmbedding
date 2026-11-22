"""Build and validate the Experiments tab and its locally hosted appendix reports."""
from __future__ import annotations

import hashlib
import html
import json
import math
import re
import shutil
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath

from xsite.opdata import check_labels

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
XDIR = HERE / "content" / "experiments"
SITEX = ROOT / "data" / "sitex"
GROUPS = {
    "inverting": "Inverting the embedding",
    "probing": "Probing networks with the embedding (X1, X3, X4)",
    "engineering": "Engineering note",
}
ID = re.compile(r"[a-z][a-z0-9-]*")
SET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")
FACT = re.compile(r"\{\{\s*fact:([^{}]+)\}\}")
LIVE_KINDS = ("lines", "heatmap", "bars", "dots", "hist")
REPORTS = {"inversion": "invert/pages/invert.html", "xp": "xp/pages/notebook.html"}
REPORT_SCRIPTS = {
    "https://cdn.plot.ly/plotly-2.35.2.min.js": "../vendor/plotly/plotly.min.js",
    "https://cdn.jsdelivr.net/npm/mathjax@3.2.2/es5/tex-chtml.js": "../vendor/mathjax/es5/tex-chtml.js",
}


class BuildError(ValueError):
    pass


def require(condition, where, message):
    if not condition:
        raise BuildError(f"{where}: {message}")


def read_json(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise BuildError(f"{path}: {e}") from e


def viewer_probe(directory):
    path = directory / "meta.json"
    meta = read_json(path)
    require(isinstance(meta, dict) and all(isinstance(meta.get(key), str) and re.fullmatch(r"[0-9a-f]{64}", meta[key])
            for key in ("probe_ids_sha256", "probe_labels_sha256")), path,
            "missing or invalid viewer probe digests; re-export without --viewer-from")
    labels = (directory / "labels.bin").read_bytes()
    require(type(meta.get("n")) is int and 0 < meta["n"] == len(labels), path, "viewer probe label count mismatch")
    require(hashlib.sha256(labels).hexdigest() == meta["probe_labels_sha256"], path,
            "viewer probe label digest does not match labels.bin")
    return meta


def manifest():
    path = XDIR / "index.json"
    entries = read_json(path)
    require(isinstance(entries, list), path, "expected an ordered list of entries")
    seen = set()
    for e in entries:
        require(isinstance(e, dict), path, f"not an entry: {e!r}")
        xid = e.get("id")
        require(isinstance(xid, str) and ID.fullmatch(xid) and xid not in seen, path,
                f"expected a unique lower-case slug, got {xid!r}")
        require(isinstance(e.get("title"), str) and e["title"].strip(), path, f"{xid}: missing title")
        require(e.get("group") in GROUPS, path, f"{xid}: unknown group {e.get('group')!r}")
        require(isinstance(e.get("pending", False), bool), path, f"{xid}: pending must be a boolean")
        seen.add(xid)
    return entries


def fact_sub(text, xid, facts, where, *, plain=False):
    def replace(m):
        key = m[1].strip()
        prefix = f"x.{xid}."
        record = facts.get(key[len(prefix):]) if xid and key.startswith(prefix) else None
        require(isinstance(record, dict) and isinstance(record.get("text"), str), where,
                f"unresolved fact {key!r}")
        return record["text"] if plain else html.escape(record["text"])

    result = FACT.sub(replace, text)
    require(not re.search(r"\{\{\s*fact\b", result), where, "unresolved or malformed fact token")
    return html.escape(result) if plain else result


def external(url):
    url = re.sub(r"[\t\r\n]", "", url).strip()
    return bool(re.match(r"(?:[a-z][a-z0-9+.-]*:|[/\\]{2})", url, re.I))


def check_css(text, where):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    for m in re.finditer(r"(?:@import\s+(?:url\(\s*)?|url\(\s*)['\"]?([^'\"\s;)]+)", text, re.I):
        url = m[1]
        require(url.startswith("data:") or not external(url), where, f"external CSS URL {url!r}")


class Markup(HTMLParser):
    """The small subset of HTML needed for fragment validation and local resources."""
    def __init__(self, text, where):
        super().__init__()
        self.where = where
        self.tags = []
        self.h2 = []
        self.figures = []
        self.in_h2 = self.in_style = self.in_caption = False
        self.style = ""
        self.figure = None
        self.feed(text)
        self.close()
        check_css(self.style, self.where)

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        line = self.getpos()[0]
        self.tags.append((tag, a, line))
        rel = (a.get("rel") or "").lower().split()
        spec = (a.get("as") or "").lower()
        dest = (a.get("src") if tag == "script" else a.get("href") if tag == "link" and
                ("stylesheet" in rel or "modulepreload" in rel or spec in ("style", "script")) else None)
        if dest:
            require(not external(dest), f"{self.where}:{line}", f"external script or stylesheet {dest!r}")
        if a.get("style"):
            check_css(a["style"], f"{self.where}:{line}")
        if tag == "style":
            self.in_style = True
        if tag == "h2":
            self.in_h2 = True
            self.h2.append("")
        if tag == "figure":
            self.figure = [a, line, ""]
            self.figures.append(self.figure)
        if tag == "figcaption" and self.figure:
            self.in_caption = True

    def handle_endtag(self, tag):
        if tag == "style":
            self.in_style = False
            self.style += "\n"
        if tag == "h2":
            self.in_h2 = False
        if tag in ("figure", "figcaption"):
            self.in_caption = False
        if tag == "figure":
            self.figure = None

    def handle_data(self, data):
        if self.in_h2:
            self.h2[-1] += data
        if self.in_style:
            self.style += data
        if self.in_caption:
            self.figure[2] += data


def check_local_resources(text, where):
    Markup(text, where)


def asset_path(url, where):
    path = PurePosixPath(url)
    require(path.parts[:1] == ("assets",) and ".." not in path.parts and not any(c in url for c in "\\?#"),
            where, f"expected an assets/ path, got {url!r}")
    return HERE.joinpath(*path.parts)


def ref_target(ref, xid, payload, where):
    url, _, key = (ref or "").partition("#")
    require(url == f"assets/x/{xid}.json" and key, where,
            f"expected assets/x/{xid}.json#<key>, got {ref!r}")
    obj = payload
    for part in key.split("."):
        if isinstance(obj, dict):
            obj = obj.get(part)
        elif isinstance(obj, list) and part.isdigit() and int(part) < len(obj):
            obj = obj[int(part)]
        else:
            obj = None
    require(isinstance(obj, dict), where, f"JSON object not found at {ref!r} (run the generator)")
    return obj


def check_live_shape(obj, kind, where):
    required = {"lines": ("x", "series"), "heatmap": ("rows", "cols", "z"),
                "bars": ("categories", "series"), "dots": ("rows", "series"), "hist": ("edges", "series")}[kind]
    panels = obj.get("panels", [obj])
    require(isinstance(panels, list) and panels and all(isinstance(p, dict) and all(k in p for k in required)
                                                       for p in panels), where,
            f"{kind} payload requires {', '.join(required)} (in each panel for small multiples)")


def read_entry(entry):
    xid = entry["id"]
    where = XDIR / f"{xid}.html"
    def optional(path):
        obj = read_json(path) if path.exists() else {}
        require(isinstance(obj, dict), path, "expected a JSON object")
        return obj

    facts = optional(HERE / "assets" / "x" / f"{xid}.facts.json")
    payload = optional(HERE / "assets" / "x" / f"{xid}.json")
    source = where.read_text(encoding="utf-8")
    text = fact_sub(source, xid, facts, where)
    parsed = Markup(text, where)
    require(parsed.tags and parsed.tags[0][0] == "section" and
            parsed.tags[0][1].get("id") == f"x-{xid}" and
            "experiment" in (parsed.tags[0][1].get("class") or "").split(), where,
            f'fragment must start with <section class="experiment" id="x-{xid}">')
    titles = [" ".join(h.split()) for h in parsed.h2]
    require(titles == [entry["title"]], where, f"expected one h2 matching {entry['title']!r}, got {titles!r}")
    for a, line, _ in parsed.figures:
        loc = f"{where}:{line}"
        classes = (a.get("class") or "").split()
        if "emb-fig" in classes or "data-emb" in a:
            require("emb-fig" in classes, loc, "data-emb requires figure.emb-fig")
            spec = ref_target(a.get("data-emb"), xid, payload, loc)
            require(spec.get("type") == "emb", loc, "data-emb must reference type: emb")
        if "live" in classes or "data-live" in a:
            kind = a.get("data-live")
            require("live" in classes and kind in LIVE_KINDS, loc, f"invalid live figure kind {kind!r}")
            check_live_shape(ref_target(a.get("data-src"), xid, payload, loc), kind, loc)

    def specs(obj, path=()):
        if isinstance(obj, dict):
            if obj.get("type") == "emb":
                yield ".".join(path), obj
            else:
                for key, value in obj.items():
                    yield from specs(value, (*path, key))
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                yield from specs(value, (*path, str(i)))

    card = payload.get("card")
    if card is not None:
        require(isinstance(card, dict) and isinstance(card.get("answer"), str) and card["answer"].strip()
                and isinstance(card.get("numbers"), list) and 3 <= len(card["numbers"]) <= 5
                and all(isinstance(n, str) and n.strip() for n in card["numbers"]), where,
                "card needs a nonempty answer and 3–5 number strings")
        card = dict(card, answer=fact_sub(card["answer"], xid, facts, where, plain=True),
                    numbers=[fact_sub(n, xid, facts, where, plain=True) for n in card["numbers"]])
        thumb = card.get("thumb")
        if thumb is not None:
            require(thumb == f"assets/fig/x_{xid}_thumb.webp", where, f"unexpected card thumbnail {thumb!r}")
            require(asset_path(thumb, where).is_file(), where, f"missing card thumbnail {thumb}")
    return text, parsed, list(specs(payload)), card


class Catalog:
    """A verified snapshot of packaged operator, overlay and path metadata."""
    def __init__(self, root, needed, manifest_path=None):
        self.ops, self.sets = {}, {}
        path = manifest_path if manifest_path is not None else root / "MANIFEST.json"
        actual = {p.relative_to(root).as_posix() for folder in ("ops", "sprites")
                  for p in (root / folder).rglob("*") if p.is_file()}
        if not needed and not path.exists() and not actual:
            return  # No packaged data to validate or copy in a stub build.
        doc = read_json(path)
        require(isinstance(doc, dict) and doc.get("schema") == 1 and isinstance(doc.get("files"), dict)
                and isinstance(doc.get("ops"), dict), path, "invalid manifest schema")
        files = doc["files"]
        require(set(files) == actual, path,
                f"stale file inventory; missing: {sorted(set(files) - actual)}; unlisted: {sorted(actual - set(files))}")
        for rel, rec in files.items():
            p = root / rel
            require(isinstance(rec, dict) and p.stat().st_size == rec.get("bytes"), path, f"stale file size: {rel}")
            with p.open("rb") as f:
                sha = hashlib.file_digest(f, "sha256").hexdigest()
            require(sha == rec.get("sha256"), path, f"stale file digest: {rel}")
        expected_meta = set()
        for op_id, rec in doc["ops"].items():
            require(SET_ID.fullmatch(op_id) and isinstance(rec, dict), path, f"invalid operator record {op_id!r}")
            rel = f"ops/{op_id}/op.json"
            expected_meta.add(rel)
            op = self.ops[op_id] = read_json(root / rel)
            require(isinstance(op, dict) and op.get("schema") == 1 and op.get("id") == op_id
                    and isinstance(op.get("fp"), str) and op["fp"] == rec.get("fp"), path,
                    f"manifest fingerprint or identity mismatch for {rel}")
            for field, folder in (("overlays", "ov"), ("paths", "paths")):
                ids = rec.get(field)
                require(isinstance(ids, list), path, f"{op_id}: missing {field} list")
                for set_id in ids:
                    require(isinstance(set_id, str) and SET_ID.fullmatch(set_id), path, f"invalid set id {set_id!r}")
                    rel = f"ops/{op_id}/{folder}/{set_id}.json"
                    expected_meta.add(rel)
                    obj = self.sets[op_id, folder, set_id] = read_json(root / rel)
                    require(isinstance(obj, dict) and obj.get("schema") == 1 and obj.get("id") == set_id
                            and obj.get("op") == op_id and obj.get("op_fp") == op["fp"], path,
                            f"op_fp or identity mismatch for {rel}")
                    try:
                        if folder == "ov":
                            check_labels(rel, obj.get("labels"), obj.get("meta"), obj.get("n"), ("label",))
                        else:
                            check_labels(rel, obj.get("labels"), obj.get("meta"), obj.get("P"), ("label_a", "label_b"))
                    except ValueError as e:
                        raise BuildError(str(e)) from e
        shipped_meta = {p for p in actual if p.startswith("ops/") and p.endswith(".json")}
        require(expected_meta == shipped_meta, path, f"stale operator or set inventory: {sorted(shipped_meta ^ expected_meta)}")

    def check_viewer(self, directory):
        meta = viewer_probe(directory)
        path = directory / "meta.json"
        for op_id, op in self.ops.items():
            probe = op.get("probe")
            require(isinstance(probe, dict) and type(probe.get("viewer_order")) is bool,
                    path, f"{op_id}: missing or invalid operator probe viewer_order")
            if not probe["viewer_order"]:
                continue
            require(probe.get("n") == meta["n"] and probe.get("ids_sha256") == meta["probe_ids_sha256"],
                    path, f"{op_id}: viewer probe ID order or count mismatch; use the viewer export for this probe")
            require(probe.get("labels_sha256") == meta["probe_labels_sha256"],
                    path, f"{op_id}: viewer probe label digest mismatch; use the viewer export for this probe")

    def check(self, spec, where):
        def seq(obj, key):
            v = obj.get(key)
            require(v is None or isinstance(v, list), where, f"{key} must be a list or null")
            return v or []

        def string(obj, key):
            v = obj.get(key)
            require(isinstance(v, str) and v, where, f"{key} must be a nonempty string")
            return v

        def overlay(obj):
            require(isinstance(obj, dict), where, "overlay must be an object")
            return string(obj, "set")

        def positions(items, n, label):
            require(isinstance(items, list) and all(type(i) is int and 0 <= i < n for i in items),
                    where, f"{label}: positions must be integers from 0 through {n - 1}")

        require(spec.get("schema") == 1, where, "unsupported embedding figure schema (expected 1)")
        kinds = seq(spec, "kinds") or [spec.get("kind") or "joint"]
        require(all(k in ("joint", "source", "site") for k in kinds), where,
                "kinds must list available coordinate kinds: joint, source, site")
        require((spec.get("kind") or kinds[0]) in kinds, where, f"default kind {spec.get('kind')!r} is not offered")
        site = string(spec, "site") if spec.get("site") is not None else None
        control = spec.get("control")
        if control is not None:
            overlay(control)
            require(control.get("mode", "side") in ("side", "over", "off"), where, "invalid control mode")

        def check_variant(variant, initial):
            panels = seq(variant, "panels")
            require(panels, where, "embedding figure needs at least one panel")
            paths = seq(variant if "paths" in variant else spec, "paths")
            draws = seq(variant if "overlays" in variant else spec, "overlays")
            options = [overlay(o) if isinstance(o, dict) else o for o in seq(spec, "overlayMenu")]
            overlays = [overlay(o) for o in draws] + options
            main = next((draw for draw in draws if draw.get("style") != "control"), {})
            all_draws = draws
            if control is not None:
                added = dict(control, draw=control.get("draw") or main.get("draw", "arrows"),
                             **({"from": main.get("from")} if control.get("from") is None else {}))
                overlays.append(added["set"])
                all_draws = [*draws, added]
            overlays.extend(string(o, "from") for o in all_draws if o.get("from") is not None)
            colour = variant.get("colour", spec.get("colour")) or {}
            require(isinstance(colour, dict), where, "colour must be an object")
            target = colour.get("set") or "probe"
            scalar_options = []
            if colour.get("by") == "scalar":
                require(isinstance(target, str), where, "scalar set must be an overlay id or 'probe'")
                scalar_options = seq(colour, "options") or [colour]
                if target != "probe":
                    overlays.append(target)
            arrow_count = variant.get("arrows", spec.get("arrows"))
            require(arrow_count is None or (type(arrow_count) in (int, float) and math.isfinite(arrow_count)
                                             and arrow_count >= 0), where, "arrows must be a nonnegative number or null")
            require(variant.get("arrowColour", spec.get("arrowColour", "one")) in ("one", "pred"), where,
                    "arrowColour must be 'one' or 'pred'")
            film = variant.get("filmstrip", spec.get("filmstrip"))
            if film is not None:
                require(isinstance(film, (list, dict)), where, "filmstrip must be a site list or object")
                film = {"sites": film} if isinstance(film, list) else film
            if variant.get("kind") is not None:
                require(variant["kind"] in kinds, where, f"variant kind {variant['kind']!r} is not offered")
            start = string(variant, "site") if variant.get("site") is not None else (site if initial else None)
            menu = seq(variant if "panelMenu" in variant else spec, "panelMenu")
            for p in panels + menu:
                require(isinstance(p, dict), where, "panel must be an object")
                op_id = string(p, "op")
                op = self.ops.get(op_id)
                require(op is not None, where, f"missing operator {op_id!r}")
                for kind in kinds:
                    require(kind in op.get("kinds", {}), where, f"operator {op_id}: missing kind {kind!r}")
                chain = op.get("chain", [])
                if start is not None and p in panels:
                    require(start in chain, where, f"operator {op_id}: missing site {start!r}")
                modes = variant.get("K", spec.get("K"))
                require(modes is None or (type(modes) is int and modes == op.get("K")), where,
                        f"operator {op_id}: requested K {modes!r}, packaged K {op.get('K')!r}")
                if variant.get("axes", spec.get("axes")) == "viewer":
                    require(op.get("viewer_signs"), where, f"operator {op_id}: no viewer-matched axes")
                if film is not None:
                    for frame in seq(film, "sites"):
                        require((type(frame) is int and 0 <= frame < len(chain)) or frame in chain, where,
                                f"operator {op_id}: filmstrip names missing site {frame!r}")
                    stat = film.get("stat")
                    if stat is not None:
                        require(isinstance(stat, dict) and len(seq(stat, "values")) == len(chain), where,
                                f"operator {op_id}: filmstrip stat needs one value per chain site")
                highlight = p.get("highlight") or variant.get("highlight", spec.get("highlight"))
                if highlight is not None:
                    require(isinstance(highlight, dict), where, "highlight must be an object")
                    if highlight.get("probe") is not None:
                        positions(highlight["probe"], op["probe"]["n"], f"operator {op_id}: probe highlight")
                    if highlight.get("overlay") is not None:
                        mark = highlight["overlay"]
                        mark_id = overlay(mark)
                        marked_set = self.sets.get((op_id, "ov", mark_id))
                        require(marked_set is not None, where, f"operator {op_id}: missing highlight overlay {mark_id!r}")
                        positions(mark.get("positions"), marked_set["n"], f"operator {op_id}: highlight overlay {mark_id!r}")
                for folder, sets in (("ov", overlays), ("paths", paths)):
                    for set_id in sets:
                        require(isinstance(set_id, str), where, f"{folder} set name must be a string")
                        obj = self.sets.get((op_id, folder, set_id))
                        require(obj is not None, where, f"operator {op_id}: missing {folder} set {set_id!r}")
                        for kind in kinds:
                            require(kind in obj.get("kinds", {}), where,
                                    f"operator {op_id}: set {set_id!r} lacks kind {kind!r}")
                        if folder == "paths":
                            require(obj.get("chain") == op.get("chain"), where,
                                    f"operator {op_id}: path set {set_id!r} has a different coordinate chain")
                if scalar_options:
                    target_obj = op if target == "probe" else self.sets[op_id, "ov", target]
                    scalars = op.get("scalars", {}) if target == "probe" else target_obj.get("meta", {}).get("scalars", {})
                    n = op["probe"]["n"] if target == "probe" else target_obj["n"]
                    for opt in scalar_options:
                        require(isinstance(opt, dict), where, "each scalar option must be an object")
                        values = opt.get("values")
                        if values is None:
                            key = string(opt, "scalar")
                            require(key in scalars, where, f"operator {op_id}: {target!r} lacks scalar {key!r}")
                            values = scalars[key].get("values")
                        def vector(items, length):
                            return isinstance(items, list) and len(items) == length and all(
                                v is None or type(v) in (int, float) for v in items)
                        require(vector(values, n) or (isinstance(values, list) and len(values) == len(chain)
                                                      and all(vector(row, n) for row in values)), where,
                                f"operator {op_id}: scalar values for {target!r} need [{n}] or [{len(chain)} × {n}]")

                def arrows(draw, set_id):
                    dst = self.sets[op_id, "ov", set_id]
                    pair = (dst.get("pairs") or {}).get("set")
                    origin = draw.get("from") or pair
                    src = self.sets.get((op_id, "ov", origin))
                    require(src is not None, where, f"operator {op_id}: overlay {set_id!r} names missing paired set {origin!r}")
                    require((pair is None or pair == origin) and src.get("n") == dst.get("n")
                            and src.get("sample", {}).get("pos") == dst.get("sample", {}).get("pos"), where,
                            f"operator {op_id}: mismatched paired samples for {origin!r} → {set_id!r}")
                    for kind in kinds:
                        require(kind in src.get("kinds", {}), where, f"operator {op_id}: paired set {origin!r} lacks kind {kind!r}")

                for draw in all_draws:
                    if draw.get("draw") == "arrows":
                        arrows(draw, draw["set"])
                first = main or (draws[0] if draws else {})
                if first.get("draw") == "arrows":
                    for set_id in options:
                        arrows(first, set_id)

        switches = seq(spec, "switch")
        if seq(spec, "panels") or not switches:
            check_variant(spec, not switches)
        for i, variant in enumerate(switches):
            require(isinstance(variant, dict), where, "switch item must be an object")
            check_variant(variant, i == 0)


def report_pages(builder):
    out = builder.out / "reports"
    for slug, rel in REPORTS.items():
        src = ROOT / rel
        if not src.is_file() and not builder.strict:
            print(f"note: report appendix unavailable ({rel})")
            continue
        page = src.read_text(encoding="utf-8")
        for cdn, local in REPORT_SCRIPTS.items():
            page = page.replace(f'src="{cdn}"', f'src="{local}"')
            require((out / local).resolve().is_file(), rel, f"missing vendored script {local}")
        page = page.replace("both loaded from a CDN", "both hosted locally")
        bar = ('<nav aria-label="Back to experiments" style="padding:.5rem 1rem; border-bottom:1px solid currentColor">'
               '<a href="../index.html#experiments" style="color:inherit">← Back to Depth Embedding experiments</a></nav>')
        page, count = re.subn(r"(<body\b[^>]*>)", lambda m: m[0] + "\n" + bar, page, count=1, flags=re.I)
        require(count == 1, rel, "report has no body")
        check_local_resources(page, rel)
        out.mkdir(parents=True, exist_ok=True)
        (out / f"{slug}.html").write_text(page, encoding="utf-8")


def build_experiments(builder):
    entries = []
    for e in manifest():
        xid = e["id"]
        src = XDIR / f"{xid}.html"
        if not src.is_file() and e.get("pending") and not builder.strict:
            continue
        require(src.is_file(), src, "experiment fragment missing")
        text, _, specs, card = read_entry(e)
        if builder.strict:
            require(card is not None and card.get("thumb"), src, "missing at-a-glance card or thumbnail")
        entries.append((e, text, specs, card))
    xout = builder.out / "x"
    for folder in ("ops", "sprites"):
        if (SITEX / folder).is_dir():
            shutil.copytree(SITEX / folder, xout / folder)
    catalog = Catalog(xout, builder.strict or bool(entries), SITEX / "MANIFEST.json")
    catalog.check_viewer(builder.d)
    for e, _, specs, _ in entries:
        for key, spec in specs:
            catalog.check(spec, f"assets/x/{e['id']}.json#{key}")
    if (HERE / "assets").is_dir():
        shutil.copytree(HERE / "assets", builder.out / "assets")
    report_pages(builder)
    builder.experiment_entries = entries


def experiments_html(builder):
    lede = XDIR / "_lede.html"
    parts = [fact_sub(lede.read_text(encoding="utf-8"), None, {}, lede) if lede.is_file() else ""]
    entries = builder.experiment_entries
    cards = {group: [] for group in GROUPS}
    for e, _, _, c in entries:
        if c is None:
            continue
        link = f"#experiments/x-{e['id']}"
        thumb = c.get("thumb")
        img = f'<img src="{thumb}" alt="" loading="lazy" decoding="async">' if thumb else ""
        nums = "".join(f"<li>{n}</li>" for n in c["numbers"])
        cards[e["group"]].append(f'<article class="x-card">{img}<h4><a href="{link}">{html.escape(e["title"])}</a></h4>'
                                 f'<p class="x-card-answer"><strong>{c["answer"]}</strong></p>'
                                 f'<ul class="x-card-numbers">{nums}</ul></article>')
    if any(cards.values()):
        parts.append('<section class="x-glance" aria-labelledby="x-glance-title"><h2 id="x-glance-title">At a glance</h2>')
        for group, title in GROUPS.items():
            if cards[group]:
                parts.append(f'<section class="x-card-group"><h3>{title}</h3><div class="x-cards">'
                             + "\n".join(cards[group]) + "</div></section>")
        parts.append("</section>")
    parts.extend(text for _, text, _, _ in entries)
    return re.sub(r"<img\b(?![^>]*\bloading\s*=)", '<img loading="lazy" decoding="async"', "\n".join(parts), flags=re.I)
