# Authoring a Depth Embedding experiment

## Files and workflow

1. Find your entry in `site/content/experiments/index.json`. The ordered array contains `id` (unique lower-case slug), `title` (the question, exactly the fragment's visible `h2` text), `group` (`inverting`, `probing`, or `engineering`), and optional `pending` (boolean). A `pending: true` entry is skipped **only** when its fragment is absent in a non-strict build; an existing fragment is always checked and shown. Do not change ordering or titles without coordinating.
2. Write `site/xsite/gen_<name>.py`. Use the stored experiment outputs; do not copy measured results by hand into prose. The generator writes `site/assets/x/<id>.json` (charts, embedding figure specs, `card`) and `site/assets/x/<id>.facts.json`. Static figures are `site/assets/fig/x_<id>_<name>.webp` (or `.png`/`.svg`). Packaged embedding data are owned separately in `data/sitex/`; see `SCHEMA.md` for operators and sets.
3. Write `site/content/experiments/<id>.html`, using `_template.html`. Add the scope and bolded main answer, then the lead embedding figure if applicable. Name the precise operator and probe wherever results and illustrations use different ones. The entry should be skimmable with details closed.

```html
<section class="experiment" id="x-example">
  <h2>The exact question from index.json?</h2>
  <p class="x-scope">Tested: …</p>
  <p class="x-answer"><strong>Main answer.</strong> Context with {{fact:x.example.metric.mean}}.</p>
  <figure class="emb-fig" data-emb="assets/x/example.json#emb_lead">
    <figcaption><strong>Plotted:</strong> The named operator, probe and overlays; explain these axes.</figcaption>
  </figure>
  <div class="x-numbers"><!-- 3–5 tiles: measured values and their operator/probe --></div>
  <h3>What was computed</h3>
  <!-- Definition table: quantity · equation · symbols and choices · code -->
  <h3>Supporting figures</h3>
  <!-- Named charts with Plotted: captions -->
  <h3>Caveats</h3>
  <ul><li>…</li></ul>
  <details><summary>Decisions made on the owner's behalf (n)</summary>…</details>
  <details><summary>More figures</summary>…</details>
  <p><a href="reports/xp.html">Full report</a> · <a href="…">Code</a></p>
</section>
```

Use `reports/inversion.html` or `reports/xp.html` as appropriate and `#method/new-images` to link to the shared method. External links to code are fine; do not embed anything that fetches from another host. Escape HTML syntax in maths where needed (e.g. `&lt;`).

## Facts and generator API

```python
from xlib import Out, PALETTE, SITES, lines, heat, bars, dots, hist

out = Out("example")
out.fact("metric.mean", measured_value, digits=2, pct=True, tier="P",
         source="file.npz:key; averaging over the held-out images")
out.chart("trend", lines(x, [{"name": "Run", "y": y}], "Site", "Mean cosine"))
out.chart("emb_lead", spec)  # schema v1 spec, not raw embedding arrays
out.chart("card", card)
out.write()
```

- `Out.fact(name, value, digits=None, pct=False, *, tier, source=None)` records `{value, text, tier, source?}` under the local dotted name. Use lower-case letters, digits and underscores within each segment. `R` = independently recomputable from raw arrays; `P` = aggregate of saved per-image/pair measurements; `S` = transcribed summary. Supply a source (file/key and selection or formula) the number checker can follow. Values are finite scalars or plain strings. `pct=True` formats a fraction as a percentage; integers default to grouped whole numbers, floats to three decimal places, percentages to zero. It returns the formatted text.
- Insert `{{fact:x.<id>.<name>}}` in the HTML or in a card text field. The build escapes and inserts the record's `text`; unknown, foreign or malformed tokens fail. Quoted depths and thresholds refer to the named full-dimensional metric, not the embedding figure's display projection.
- `Out.chart(key, obj)` accepts a JSON object, converts NumPy data, and represents non-finite chart cells as `null`. `Out.write()` writes both JSON files. `Out.fig_path(name, suffix=".webp")` creates the parent directory and returns a `Path` for a static figure. `PALETTE` is the site's `palette.json` dictionary, not a separate palette; `SITES` is the usual full-chain site list (always use the packaged `chain` when a figure varies its operator).
- `card` is `{"answer": "One answer sentence.", "numbers": ["3 to 5 short strings"], "thumb": "assets/fig/x_<id>_thumb.webp"}`. `answer` and the 3–5 `numbers` strings are plain text, optionally with fact tokens. Omit `thumb` until screenshot capture, then set this exact path and re-run the generator. Strict builds require both card and thumbnail.

## Markup contract

The builder inserts `_lede.html`, followed by the at-a-glance card groups, then the authored `section.experiment` elements in manifest order. A card is `.x-glance > .x-card-group > .x-cards > article.x-card`, with an optional decorative lazy `<img>`, `<h4><a href="#experiments/x-<id>">…</a></h4>`, `<p class="x-card-answer"><strong>…</strong></p>`, and `<ul class="x-card-numbers"><li>…</li>…</ul>`. The group has an `h3`. Empty groups are omitted.

The authored figures are passed through unchanged except for build-time facts and lazy image attributes:

```html
<figure class="live" data-live="lines" data-src="assets/x/example.json#trend" data-height="280">
  <div class="live-body"></div>
  <figcaption><strong>Plotted:</strong> What was measured and what each trace means.</figcaption>
</figure>
<figure class="emb-fig" data-emb="assets/x/example.json#emb_lead">
  <figcaption><strong>Plotted:</strong> Which operator and coordinate kind are shown.</figcaption>
</figure>
```

`data-height` is optional for live figures. `data-src` and `data-emb` name an object in the entry's own JSON (`#key` or `#nested.key`). Use `data-live="lines|heatmap|bars|dots|hist"` (one value); JS fills `.live-body`. The embedding component creates its controls, canvas(es) and operator line before the caption. Do not add a live body, operator line or placeholder canvas to the embedding figure.

### Embedding figure spec

An embedding object in `site/assets/x/<id>.json` looks like this (use real IDs and available kinds):

```json
{
  "type": "emb", "schema": 1,
  "panels": [{"op": "r20s0-c10-ea100", "label": "Shannon perplexity 100"}],
  "kinds": ["site", "source", "joint"], "kind": "site", "site": "b6",
  "K": 16, "dims": 2, "axes": "pooled:0", "norm": "site",
  "colour": {"by": "class"},
  "overlays": [{"set": "u8", "draw": "arrows", "from": "clean"}],
  "overlayMenu": ["u2", "u8", "t8", "n8"], "height": 420
}
```

`op` names `x/ops/<op>/op.json`; sets are local to that operator. Use `draw: "arrows"` with a paired `from` set, and `style: "control"` for noise/control. `overlayMenu` provides alternate set names (or `{ "set": …, "label": … }` options). `colour: {"by":"scalar", "set":"H", "scalar":"decode_cos_K16"}` reads the named scalar of a drawn overlay. For a probe scalar, use `set: "probe"` (or omit `set`) and supply either `colour.values` in probe order or the name of a packaged `op.scalars` record. `highlight: {"probe":[…]}` rings zero-based positions in the packaged probe order (not dataset IDs). `axes: "viewer"` requires an operator with viewer-matched bases. `viewerLink` can point to a comparable viewer setting.

For panel swaps, `panelMenu` items have the same shape as `panels`: each item's `R` is its own rotation onto the reference, each `stat: {"label": …, "values": […]}` its own curve on its chain. For suffix switches, supply `switch: [{"label": …, "panels": […], "paths": ["<set>"], "site": …}, …]` rather than `panels`. A switch may override `site`, `kind`, `paths`, `overlays`, `colour` and `highlight`; without an explicit starting site, defaults use the first site of the active chain. Switch by site **name**, never by hard-coded slice index. All selectable operators/sets must support the declared `kinds`. The packaged manifest, operator fingerprint, paths' coordinate grid and every explicit starting site are checked before shipping.

### Live chart payloads

Helpers return plain dictionaries for `Out.chart`. They accept optional renderer fields as keyword arguments. Send missing data as `null` or NumPy non-finite values, not zero. Any chart kind can also be `{"panels": [{"title": …, …chart fields…}, …], "shared": true}` for small multiples sharing an axis range.

| HTML `data-live` | Helper and payload |
|---|---|
| `lines` | `lines(x, series, xlabel, ylabel, xscale="linear", yrange=(0,1), colors="categorical", **kw)`; series have `name`, `y`, optional `color`, `dash` (true or `"dot"`), `width`, `dots`, `noline`, `band: [lo, hi]`. `xticklabels` label categorical positions; `refline: {"y": …, "label": …}` adds a reference. |
| `heatmap` | `heat(z, rows, cols, zlabel, rowlabel, collabel, **kw)`; `zrange` defaults to `[0,1]`; optional `slider: {"label": …, "values": […]}`, `diverging` and `fmt`. |
| `bars` | `bars(categories, series, ylabel, xlabel="", **kw)`; series have `name`, `values`, optional `err: [[lo,hi], …]`, `color`, `control` (hollow dashed outlines). Optional `yrange`, `refline: {"y": …, "label": …}`, `fmt`, `colors`. Use this for binned labels such as sites. |
| `dots` | `dots(rows, series, xlabel, **kw)`; series have `name`, `values`, optional `lo`, `hi` (interval whiskers), `marker` (`circle`, `diamond`, `triangle`, `square`), `hollow`, `color`. Optional `rowgroups: [{"label": …, "from": …, "to": …}]`, `xrange`, `refline: {"x": …, "label": …}`, `fmt`. |
| `hist` | `hist(edges, series, xlabel, ylabel, norm="count", **kw)`; B+1 numeric edges, series have `name`, B `counts`, optional `color`, `control`; `norm` is `"share"` or `"count"`; optional `xticklabels`. |

## Build and check in a private directory

From the repository root, replace `example` with your manifest id, and choose your package's output root, loopback port and browser session name:

```sh
uv run --with numpy --with pillow python site/xsite/gen_example.py
python3 site/check_fragment.py example
uv run --with numpy --with pillow python site/build_site.py --out data/build_example --viewer-from build/site
# Start this in a separate process; stop it after the check.
python3 -m http.server 8821 --bind 127.0.0.1 -d data/build_example
uv run --with pillow python site/check_browser.py http://127.0.0.1:8821/ data/shots/example --session example --capture example
```

Run `--capture` once to create the thumbnail in `site/assets/fig/`; then add it to `card` and regenerate and rebuild. Without `--capture` the browser checker never edits source files. It checks both viewports and writes Viewer, Method, Experiments overview (when present), each entry and the two full-report appendix screenshots. Inspect the images as well as the test result. `check_fragment.py --strict` upgrades warnings to errors. `build_site.py --strict` requires all fragments, cards, thumbs, packaged data manifest and both report inputs. A non-strict build with no entries may run before packaging, provided it has no partial `data/sitex/ops` or `sprites` files. Non-strict builds also skip absent generated report pages; the browser checker expects both to be present.

The report build rewrites the two upstream URLs to the pinned npm distributions in `site/vendor/`: `plotly.js-dist-min@2.35.2` and `mathjax@3.2.2` (`es5/tex-chtml.js`, TeX extensions, and its relative CHTML WOFF directory). Each distribution's `LICENSE` is alongside it.

`--out` replaces its entire destination; use your private directory, **not** the shared `build/site`. The destination must be nonexistent, be empty, or contain a previous build's `.nojekyll` marker. `--viewer-from` copies that existing build's `d/` without changing its bytes or reusing its HTML, JS, experiment data or reports.

## Figure spec (what `site/js/embfig.js` reads; authoritative)

This section is the single source of truth for embedding-figure fields; the summary under "Embedding figure spec" above is an overview. Field names are exact. Anything not listed is ignored.

**Top level**

| Field | Type, default | Meaning |
|---|---|---|
| `type`, `schema` | `"emb"`, `1` | required; any other schema shows "Figure not drawn". |
| `root` | string, `"x/"` | where `ops/<op>/…` live, relative to the page (fixtures and demos only). |
| `panels` | list of panel objects | the panels shown (ignored while a `switch` item is active). |
| `panelMenu` | list of panel objects or null | extra items for the network menu that replaces panels 2… (each item brings its own `R` and `stat`). |
| `switch` | list or null | `[{label, panels, paths?, overlays?, site?, kind?, colour?, highlight?, filmstrip?}]`; a segmented control picks one. An item's fields override the top level while it is active. `switchLabel` (default `"Start"`) titles the control. |
| `kinds`, `kind` | list of `joint`/`source`/`site`, first | coordinate kinds offered, and the initial one. A kind missing from a panel's operator is shown disabled. |
| `site` | site name, first site of the chain | initial site. On a switch the current site name is kept if the new chain has it, else the item's `site`, else the first site. |
| `K` | integer or omitted | if given, must equal the operator's `K` (error otherwise). |
| `dims` | `2` | only 2 is supported. |
| `axes` | `"pooled"` (default), `"pooled:0"`, `"viewer"` | joint/source axes: each panel's operator's pooled probe PCA (`op.axes.pooled`, columns 0 and 1); `pooled:0` uses panel 0's axes for every panel (use with `R`); `viewer` uses `op.axes.viewer` and `op.viewer_signs`. Site-local coordinates always use each site's own walk modes 1 and 2 (signs chosen to agree with the previous site), whatever `axes` says. |
| `norm` | `"site"` (default) or `"none"` | `site` divides every site by the probe's Frobenius norm there (`op.site_norm[kind]`), overlays and paths by the same number. |
| `overlays` | list | `[{set, draw: "arrows"|"points", from?, style?: "control", label?, verb?}]`. `from` (or the set's `pairs.set`) is the paired set an arrow starts from; both sets must hold the same sampled images in the same order (`sample.pos`). The first non-control overlay is the *main* overlay. `label` replaces the set's label in legend, captions and tooltips; `verb` replaces "attacked"/"noised" in the legend ("arrow from each clean image to its *verb* version"). |
| `overlayMenu` | list of set ids or `{set, label?, style?}` | a select (titled `overlayMenuLabel`, default "Overlay") that replaces the main overlay's set; an object item's `style`/`label` apply while it is chosen. The choice persists across `switch` items. |
| `control` | `{set, label?, from?, draw?, mode?: "side"|"over"|"off", title?}` or null | a noise/control set with a three-way control (titled `title`, default "Noise control"): hidden, overlaid (grey dashed arrows on the same panel), or side by side (two panels, same operator, same axes and limits, one slider). `mode` is the initial state (default `"side"`). Side by side needs a single base panel. |
| `arrows` | number, `60`; `null` = all | arrows drawn per set by default (a fixed, spread-out subset); the reader can pick 60 / 150 / all. With one class isolated from the legend, all of that class's arrows are drawn. |
| `arrowColour` | `"one"` (default) or `"pred"` | initial arrow colour: the accent, or the class the network predicts for the perturbed image (`meta.pred`). The reader can switch when `meta.pred` exists. Control arrows stay grey. |
| `colour` | `{by: "class"}` (default) or scalar, below | |
| `highlight` | `{probe?: [positions], overlay?: {set, positions: [positions]}, label?}` | rings probe images (positions in the operator's probe order) and/or overlay points (positions within that set, 0…n−1). `label` is the legend text. |
| `filmstrip` | list of site names (or chain indices), or `{sites, stat?: {label, values}}` | small static frames of the first panel at those sites (same limits; the same axes except for site-local), under the panel; clicking one moves the slider. `stat.values` has one value per chain site and is printed under each frame; `stat.kind` (a kind or list of kinds) prints it only while those coordinates are shown. |
| `layout` | `"grid"` (default) or `"row"` | `grid`: 2 panels side by side, 3 in a row, 4 as 2 × 2. `row`: all panels in one row with the legend below, whenever each panel gets at least 250 px; on screens ≥ 1120 px the figure may extend beyond the text column (up to 1360 px). Both fall back to 2 × 2 or one column on narrow screens (below 560 px, more than two panels show one at a time with a Panel select). |
| `height` | px, `520` | maximum height of a single panel (the panel's shape otherwise follows the data's aspect ratio). |
| `viewerLink` | hash or null | "Open the matching view in the viewer" link. |

**Panel objects** (`panels`, `panelMenu`, `switch[].panels`): `{op, label, R?: [[K x K]] | null, stat?: {label, values: [one per chain site]} | null, highlight?: {probe?, overlay?, label?}}`. `R` turns the panel's joint/source rows onto the first panel: displayed row = y · R (R[k][c]); use with `axes: "pooled:0"`. `stat` is shown under the panel label at the current site with a sparkline. A panel's own `highlight` replaces the top-level one for that panel.

**Scalar colouring**: `colour: {by: "scalar", set: "<overlay id>" | "probe", scalar?: "<name>", values?: [n] | [L x n], label?, range?: [lo, hi], options?: [{label, scalar? | values?}], title?}`. For an overlay, `scalar` names `meta.scalars[<name>]` of that set (`{values, label, range}`); for the probe, `op.scalars[<name>]` or inline `values` in probe order. Values are `[n]` (one per image) or `[L x n]` (per site; follows the slider). With `options`, a segmented control (titled `title`, default "Colour") offers each; the colour range is fixed across sites and options (`colour.range`, else the union of all options' values) and the legend says so. While a scalar colours an overlay, the probe is drawn grey.

**Data checks in the browser**: `schema` of spec, operator and sets; an overlay or path set whose `op`/`op_fp` differ from its panel's operator, a pair whose `sample.pos` differ, a path set whose `chain` differs from its operator's, or any shape mismatch makes that panel show "Not drawn: data mismatch" (or "Not drawn") and log to the console; nothing is drawn in that panel.

**What every figure shows by itself**: the site slider (play, site labels), the class key (also a filter: point at a class to show it alone, click to keep it), the legend for each overlay, path and ramp, hover/tap tooltips (thumbnail, true and predicted class, image id, scalar), and the operator line naming network, step, kernel, probe, sites, coordinate kind, projection, and whether panels share axes.
