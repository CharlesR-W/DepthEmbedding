// Embedded diffusion-embedding figures for the Experiments tab: <figure class="emb-fig" data-emb="assets/x/ID.json#key">.
// Each panel draws the coordinates of ONE operator (schema v1, site/xsite/SCHEMA.md): its probe cloud, overlay sets
// (points, or arrows from a paired set), paths along s, ringed probe images. Every overlay and path set must carry the
// fingerprint of its panel's operator (op_fp == op.fp); otherwise the panel shows an error and draws nothing.
// The figure-spec fields read here are listed in site/xsite/AUTHORING.md ("Figure spec").
import { eigSym, crossTN, mark } from "./embedding.js";

const SCHEMA = 1;
const KIND_NAME = { joint: "joint", source: "source", site: "site-local" };
const KIND_HINT = {
  joint: "joint coordinates: the operator's through-depth modes, one set of axes for every site",
  source: "source coordinates: each site's local kernel average p_l·f_l of the joint modes, before depth filtering (divided by 2c_l, it is the right-hand side of the depth solve); drawn on the joint coordinates' axes",
  site: "site-local coordinates: each site's own walk modes; the axes differ from site to site",
};
const CIFAR = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];
const SITE_MS = 650;            // play speed: one site per SITE_MS
const NARROW = 560;             // below this figure width, multi-panel figures show one panel at a time
// True-label spaces. A set's `labels: {space, names}` says what its meta.label (paths: label_a, label_b) means; the
// probe's labels are CIFAR-10, as are predictions (the network's outputs). A set without `labels` has an unknown space:
// its true labels are not named and the class filter does not apply to it.
const PROBE_SPACE = "cifar10";
const SPACE_NAME = { cifar10: "CIFAR-10", cifar100: "CIFAR-100", svhn: "SVHN", imagenet32: "ImageNet-32" };
const spaceOf = set => set?.labels?.space ?? null;
const sameSpace = set => spaceOf(set) === PROBE_SPACE;
function trueName(set, id, classes) {
  if (id == null || id < 0) return "–";
  const sp = spaceOf(set), nm = set?.labels?.names?.[id];
  if (sp === PROBE_SPACE) return nm ?? classes[id] ?? `class ${id}`;
  if (sp === "svhn") return `SVHN digit ${nm ?? id}`;
  if (sp) return nm != null ? `${nm} (${SPACE_NAME[sp] || sp})` : `${SPACE_NAME[sp] || sp} class ${id}`;
  return `label ${id} (label space not recorded)`;
}

// ------------------------------------------------------------------ small helpers
const cssv = (v, d) => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || d;
function hexRgb(h) { h = h.replace("#", ""); if (h.length === 3) h = [...h].map(c => c + c).join(""); return [0, 2, 4].map(o => parseInt(h.slice(o, o + 2), 16)); }
const rgba = ([r, g, b], a) => `rgba(${r | 0},${g | 0},${b | 0},${a})`;
const fmt = x => x == null || !isFinite(x) ? "–" : Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 1 ? x.toPrecision(3) : x.toPrecision(2);
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const num = n => Number(n).toLocaleString("en");
const rfmt = (v, [a, b]) => { const s = Math.abs(b - a); return (+v).toFixed(s >= 20 ? 0 : s >= 2 ? 1 : s >= 0.2 ? 2 : 3); };
const ord = n => `${n}${n % 10 === 1 && n % 100 !== 11 ? "st" : n % 10 === 2 && n % 100 !== 12 ? "nd" : n % 10 === 3 && n % 100 !== 13 ? "rd" : "th"}`;
const predOf = m => m.adv_pred ?? m.pred ?? null;      // the network's prediction for the drawn input (CIFAR-10 output)
const lower = t => /^[A-Z][a-z]/.test(t) ? t[0].toLowerCase() + t.slice(1) : t;
function h(tag, attrs = {}, ...kids) {
  const e = document.createElement(tag);
  for (const k in attrs) {
    const v = attrs[k]; if (v == null || v === false) continue;
    if (k === "text") e.textContent = v; else if (k === "html") e.innerHTML = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v === true ? "" : v);
  }
  for (const c of kids) if (c != null) e.append(c);
  return e;
}
class DataError extends Error { constructor(msg, detail) { super(msg); this.detail = detail || ""; this.name = "DataError"; } }
function palette() {
  return {
    classes: [...Array(10).keys()].map(i => hexRgb(cssv(`--c-class-${i}`, "#888888"))),
    seq: [...Array(8).keys()].map(i => hexRgb(cssv(`--c-seq-${i}`, "#888888"))),
    accent: hexRgb(cssv("--accent", "#c2410c")), fg: hexRgb(cssv("--fg", "#16161a")), mute: hexRgb(cssv("--mute", "#6b6b70")),
    grey: hexRgb(cssv("--unassigned", "#d4d4d4")), surface: cssv("--surface", "#ffffff"), line: cssv("--line", "#e4e3de"),
  };
}
// sequential ramp, starting at 22% so that the lightest values stay visible on white
function seqColour(P, t) {
  const s = P.seq, x = (0.22 + 0.78 * Math.max(0, Math.min(1, t))) * (s.length - 1), i = Math.min(s.length - 2, Math.floor(x)), f = x - i;
  return [0, 1, 2].map(c => s[i][c] + (s[i + 1][c] - s[i][c]) * f);
}
const LEVELS = 24;
const qLevel = t => Math.max(0, Math.min(LEVELS - 1, Math.round(t * (LEVELS - 1))));

// ------------------------------------------------------------------ data (module-level cache)
const cache = new Map();
function once(key, fn) { if (!cache.has(key)) { const p = fn().catch(e => { cache.delete(key); throw e; }); cache.set(key, p); } return cache.get(key); }
const getJSON = url => once("j:" + url, async () => { const r = await fetch(url); if (!r.ok) throw new DataError(`could not load ${url} (HTTP ${r.status})`); return r.json(); });
const getBytes = url => once("b:" + url, async () => { const r = await fetch(url); if (!r.ok) throw new DataError(`could not load ${url} (HTTP ${r.status})`); return r.arrayBuffer(); });
const getGz = url => once("z:" + url, async () => {
  const r = await fetch(url); if (!r.ok) throw new DataError(`could not load ${url} (HTTP ${r.status})`);
  return new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
});
const getImage = url => once("i:" + url, () => new Promise((res, rej) => { const im = new Image(); im.onload = () => res(im); im.onerror = () => rej(new DataError(`could not load ${url}`)); im.src = url; }));
const rel = (file, base) => new URL(file, base).href;
function getPath(obj, path) { return path ? path.split(".").reduce((o, k) => (o == null ? o : o[k]), obj) : obj; }
async function classNames() {
  try { const m = await getJSON(rel("d/meta.json", document.baseURI)); if (Array.isArray(m.classes)) return m.classes; } catch (e) { /* no viewer data: CIFAR-10 names */ }
  return CIFAR;
}

// dequantise one coordinate array: q (int16 / int8) x scale[site][mode]; shape [L, ..., K]
async function loadArray(spec, base) {
  const buf = await getGz(rel(spec.file, base));
  const T = spec.dtype === "i8" ? Int8Array : spec.dtype === "i16" ? Int16Array : null;
  if (!T) throw new DataError(`unknown dtype ${spec.dtype} in ${spec.file}`);
  const shape = spec.shape, n = shape.reduce((a, b) => a * b, 1), L = shape[0], K = shape[shape.length - 1], inner = n / L;
  if (buf.byteLength !== n * T.BYTES_PER_ELEMENT) throw new DataError(`${spec.file}: ${buf.byteLength} bytes, expected ${n * T.BYTES_PER_ELEMENT} for shape [${shape}]`);
  if (!Array.isArray(spec.scale) || spec.scale.length !== L || spec.scale[0].length !== K) throw new DataError(`${spec.file}: scale is not [L x K]`);
  const q = new T(buf), out = new Float32Array(n);
  for (let l = 0; l < L; l++) { const sc = spec.scale[l], o = l * inner; for (let j = 0; j < inner; j++) out[o + j] = q[o + j] * sc[j % K]; }
  return out;
}
function checkSchema(obj, what) { if (!obj || obj.schema !== SCHEMA) throw new DataError(`${what}: schema ${obj && obj.schema} (this page reads schema ${SCHEMA})`); }

const ops = new Map();
async function loadOp(root, id) {
  const url = rel(`${root}ops/${id}/op.json`, document.baseURI);
  return once("op:" + url, async () => {
    const op = await getJSON(url);
    checkSchema(op, `operator ${id}`);
    if (op.id !== id) throw new DataError(`operator file ${url} has id ${op.id}, expected ${id}`);
    const L = op.chain.length, N = op.probe.n, K = op.K;
    for (const [k, s] of Object.entries(op.kinds)) if (s.shape[0] !== L || s.shape[1] !== N || s.shape[2] !== K) throw new DataError(`operator ${id}, ${k}: shape [${s.shape}] does not match chain ${L}, probe ${N}, K ${K}`);
    const labelsP = op.probe.labels ? getBytes(rel(op.probe.labels, url)).then(b => new Uint8Array(b)) : Promise.resolve(null);
    const o = { op, url, id, L, N, K, kind: {}, labels: await labelsP.catch(() => null) };
    if (o.labels && o.labels.length !== N) throw new DataError(`operator ${id}: ${o.labels.length} probe labels for ${N} probe images`);
    ops.set(url, o);
    return o;
  });
}
const kindData = (o, kind) => once(`k:${o.url}|${kind}`, () => { const s = o.op.kinds[kind]; if (!s) throw new DataError(`operator ${o.id} has no ${kind} coordinates`); return loadArray(s, o.url); });
async function loadSet(o, sub, id) {
  const url = rel(`${sub}/${id}.json`, o.url);
  const s = await getJSON(url);
  checkSchema(s, `${sub === "ov" ? "overlay" : "path set"} ${id}`);
  if (s.op !== o.id || s.op_fp !== o.op.fp)
    throw new DataError(`Data mismatch: ${sub === "ov" ? "overlay" : "path set"} “${id}” was packaged for operator ${s.op} (fingerprint ${String(s.op_fp).slice(0, 10)}…); this panel draws ${o.id} (fingerprint ${String(o.op.fp).slice(0, 10)}…).`, url);
  return { s, url };
}

// ------------------------------------------------------------------ the component
const DENSITY = [60, 150];                 // arrow-density choices below "all"
export class EmbFigure {
  constructor(fig, spec) {
    this.fig = fig; this.spec = spec; this.root = spec.root || "x/";
    this.P = palette();
    this.kind = spec.kind || (spec.kinds || ["joint"])[0];
    this.b = 0; this.playing = false; this.sw = 0; this.focus = 0;
    this.ovChoice = null;            // the overlay set chosen in the overlay menu (replaces the main overlay's set)
    this.ctlMode = spec.control ? (spec.control.mode || "side") : "off";      // noise control: off | over | side
    this.density = spec.arrows === null || spec.arrows === 0 ? Infinity : spec.arrows ?? DENSITY[0];
    this.arrowCol = spec.arrowColour === "pred" ? "pred" : "one";
    this.iso = null; this.isoPreview = null;                                   // class isolated from the legend
    this.scalarOpt = 0;                                                        // chosen colour.options item
    this.pathKind = "all";                                                     // paths shown: all | diff | same
    this.thumb = false;                                                        // thumbnail capture: plot only
    this.panels = [];
    this.build();
    this.ready = this.start();
  }

  // ---------------------------------------------------- spec views (a switch item overrides the top level)
  cur() { const s = this.spec, w = s.switch && s.switch.length ? s.switch[this.sw] : null; return w ? { ...s, ...w, panels: w.panels } : s; }
  // the overlays of the figure: the first non-control overlay is the "main" one; the overlay menu replaces its set (an
  // object menu item {set, label, style} also sets its style and label); spec.control adds the noise control
  overlaysNow() {
    const s = this.spec, c = this.cur(), list = (c.overlays || []).map(x => ({ ...x }));
    let main = list.find(x => x.style !== "control");
    if (this.ovChoice) {
      const it = (s.overlayMenu || []).find(m => (typeof m === "string" ? m : m.set) === this.ovChoice);
      if (!main) { main = { draw: list[0]?.draw || "points", from: list[0]?.from }; list.unshift(main); }
      main.set = this.ovChoice;
      if (it && typeof it === "object") { main.style = it.style; if (it.label) main.label = it.label; }
      else { delete main.style; delete main.label; }
    }
    if (main) main.main = true;
    const out = list.filter(x => x.main || x.set !== main?.set);
    if (s.control && s.control.set !== main?.set && !out.some(x => x.set === s.control.set))
      out.push({ set: s.control.set, draw: s.control.draw || (main?.draw ?? "arrows"), from: s.control.from ?? main?.from, style: "control", label: s.control.label, isControl: true });
    return out;
  }
  // which overlays a panel draws: with the control "side by side", the first panel has the rest and the second the control
  panelOverlays(p) {
    const all = this.overlaysNow(), ctl = x => x.style === "control" && !x.main;
    if (this.ctlMode === "off") return all.filter(x => !x.isControl);
    if (this.ctlMode === "side" && this.sideBySide()) return p.role === "control" ? all.filter(ctl) : all.filter(x => !ctl(x));
    return all;
  }
  sideBySide() { const c = this.cur(); return this.ctlMode === "side" && !!this.spec.control && (c.panels || []).length === 1 && this.overlaysNow().some(x => x.isControl); }

  // ---------------------------------------------------- DOM
  build() {
    const s = this.spec, cap = this.fig.querySelector(":scope > figcaption");
    this.el = h("div", { class: "ef" });
    this.fig.insertBefore(this.el, cap || null);
    if (s.layout === "row") this.fig.classList.add("ef-bleed");      // wide screens: the figure may use more than the text column
    this.playBtn = h("button", { type: "button", class: "ef-play", "aria-label": "Play through the sites", onclick: () => this.togglePlay() });
    this.siteName = h("b"); this.siteSub = h("span");
    this.range = h("input", { type: "range", min: 0, max: 1, step: "any", value: 0, "aria-label": "Site" });
    this.range.addEventListener("input", () => { this.stopAnim(); this.setSite(+this.range.value, true); });
    this.range.addEventListener("change", () => this.setSite(Math.round(+this.range.value)));
    this.ticks = h("div", { class: "ef-ticks" });
    this.el.append(h("div", { class: "ef-bar" }, this.playBtn, h("div", { class: "ef-site" }, this.siteName, this.siteSub), h("div", { class: "ef-track" }, this.range, this.ticks)));
    this.ctl = h("div", { class: "ef-ctl" });
    this.el.append(this.ctl);
    this.grid = h("div", { class: "ef-panels" });
    this.film = h("div", { class: "ef-film" });
    this.filmTitle = h("p", { class: "ef-film-title" });
    this.filmWrap = h("div", { class: "ef-filmwrap", hidden: true }, this.filmTitle, this.film);
    this.legend = h("div", { class: "ef-legend" });
    this.display = h("div", { class: "ef-display" });
    this.el.append(h("div", { class: "ef-body" }, h("div", { class: "ef-main" }, this.grid, this.filmWrap), h("div", { class: "ef-side" }, this.legend, this.display)));
    this.opline = h("p", { class: "ef-op" });
    this.el.append(this.opline);
    if (s.viewerLink) this.el.append(h("p", { class: "ef-link" }, h("a", { href: s.viewerLink, text: "Open the matching view in the viewer" })));
    this.status = h("p", { class: "ef-status", role: "status" });
    this.el.append(this.status);
    this.setPlayIcon();
    new ResizeObserver(() => { const w = this.el.clientWidth; if (Math.abs(w - (this.lastW || 0)) > 4) { this.lastW = w; this.layout(); } }).observe(this.el);
    new IntersectionObserver(es => { if (!es.some(e => e.isIntersecting)) this.stop(); }).observe(this.el);
  }
  // thumbnail capture (site/check_browser.py --capture): the plot area only, without labels, rings, hover or notes
  setThumb(on) {
    this.thumb = !!on; this.stop(); this.iso = this.isoPreview = null; this.panels.forEach(p => { p.hot = null; });
    const i = this.chain().indexOf(this.cur().site); if (on && i >= 0) this.setSite(i);       // the figure's opening site
    this.el.classList.toggle("ef-thumb", this.thumb); this.draw();
  }
  setPlayIcon() {
    this.playBtn.innerHTML = this.playing ? '<svg viewBox="0 0 12 12" width="12" height="12"><rect x="2" y="1.5" width="3" height="9" rx=".6"/><rect x="7" y="1.5" width="3" height="9" rx=".6"/></svg>'
      : '<svg viewBox="0 0 12 12" width="12" height="12"><path d="M3 1.6v8.8a.5.5 0 0 0 .76.43l7-4.4a.5.5 0 0 0 0-.86l-7-4.4A.5.5 0 0 0 3 1.6z"/></svg>';
    this.playBtn.setAttribute("aria-label", this.playing ? "Pause" : "Play through the sites");
  }
  seg(label, items, val, on, cls = "") {
    const g = h("div", { class: "ef-seg", role: "radiogroup", "aria-label": label });
    for (const [v, t, dis, tip] of items) {
      const b = h("button", { type: "button", role: "radio", "aria-checked": String(v === val), disabled: dis || null, title: tip || null, text: t });
      b.addEventListener("click", () => { if (v !== val) on(v); });
      g.append(b);
    }
    return h("div", { class: `ef-field ${cls}` }, h("span", { class: "ef-lab", text: label }), g);
  }
  buildControls() {
    const s = this.spec;
    this.ctl.replaceChildren(); this.display.replaceChildren();
    if (s.switch && s.switch.length > 1)
      this.ctl.append(this.seg(s.switchLabel || "Start", s.switch.map((w, i) => [i, w.label]), this.sw, v => this.setSwitch(v)));
    const kinds = s.kinds || [this.kind];
    if (kinds.length > 1) {
      const have = k => this.panels.every(p => !p.o || p.o.op.kinds[k]);
      this.ctl.append(this.seg("Coordinates", kinds.map(k => [k, KIND_NAME[k] || k, !have(k), KIND_HINT[k] + (have(k) ? "" : " (not packaged for this operator)")]), this.kind, v => this.setKind(v)));
    }
    if (s.overlayMenu && s.overlayMenu.length) {
      const sel = h("select", { "aria-label": s.overlayMenuLabel || "Overlay" });
      const cur = this.overlaysNow().find(x => x.main)?.set;
      for (const it of s.overlayMenu) { const id = typeof it === "string" ? it : it.set; sel.append(h("option", { value: id, selected: id === cur || null, text: (typeof it === "object" && it.label) || this.setLabels?.[id] || id })); }
      sel.addEventListener("change", () => { this.ovChoice = sel.value; this.reloadPanels(); });
      this.ctl.append(h("label", { class: "ef-field" }, h("span", { class: "ef-lab", text: s.overlayMenuLabel || "Overlay" }), sel));
    }
    const col = this.cur().colour;
    if (col?.by === "scalar" && col.options?.length > 1)
      this.ctl.append(this.seg(col.title || "Colour", col.options.map((o, i) => [i, o.label || o.scalar]), Math.min(this.scalarOpt, col.options.length - 1), v => { this.scalarOpt = v; this.panels.forEach(p => { try { if (p.o && !p.err) this.colourFor(p); } catch (e) { p.err = e; console.error(e); } }); this.buildControls(); this.describe(); this.redraw(); }));
    if (s.control) {
      const mainIsCtl = this.overlaysNow().find(x => x.main)?.style === "control";
      this.ctl.append(this.seg(s.control.title || "Noise control", [["off", "hidden"], ["over", "overlaid", mainIsCtl], ["side", "side by side", mainIsCtl]], mainIsCtl ? "off" : this.ctlMode, v => { this.ctlMode = v; this.reloadPanels(); }));
    }
    if (this.narrow && this.panels.length > 2) {
      const sel = h("select", { "aria-label": "Panel" });
      this.panels.forEach((p, i) => sel.append(h("option", { value: i, selected: i === this.focus || null, text: p.item.label })));
      sel.addEventListener("change", () => { this.focus = +sel.value; this.layout(); });
      this.ctl.append(h("label", { class: "ef-field" }, h("span", { class: "ef-lab", text: "Panel" }), sel));
    }
    this.ctl.hidden = !this.ctl.children.length;
    // display options (beside the legend): arrow density and arrow colour
    const arrowSets = this.panels.flatMap(p => (p.sets || []).filter(st => st.cfg.draw === "arrows"));
    const nArrows = Math.max(0, ...arrowSets.map(st => st.s.n));
    const opts = DENSITY.filter(d => d < nArrows);
    if (opts.length) this.display.append(this.seg("Arrows per set", [...opts.map(d => [d, String(d)]), [Infinity, `all ${num(nArrows)}`]], opts.includes(this.density) ? this.density : Infinity, v => { this.density = v; this.buildControls(); this.describe(); this.redraw(); }, "ef-stack"));
    if (arrowSets.some(st => st.cfg.style !== "control" && st.s.meta?.pred))
      this.display.append(this.seg("Arrow colour", [["one", "one colour"], ["pred", "predicted class"]], this.arrowCol, v => { this.arrowCol = v; this.buildControls(); this.describe(); this.redraw(); }, "ef-stack"));
    // paths: show both kinds, or one (different-class or same-class endpoints)
    const ps = this.panels.find(p => p.paths?.length)?.paths[0]?.s, pk = ps?.meta?.kind || [];
    const nd = pk.filter(k => k === "diff").length, ns = pk.filter(k => k !== "diff").length;
    if (nd && ns) this.display.append(this.seg("Paths shown (endpoint classes)", [["all", `all ${nd + ns}`], ["diff", `${nd} different`], ["same", `${ns} same`]], this.pathKind, v => { this.pathKind = v; this.buildControls(); this.redraw(); }, "ef-stack"));
  }

  // ---------------------------------------------------- loading
  async start() {
    try {
      checkSchema(this.spec, "figure spec");
      if (this.spec.type && this.spec.type !== "emb") throw new DataError(`figure spec type ${this.spec.type} is not "emb"`);
      if ((this.spec.dims || 2) !== 2) throw new DataError("only dims = 2 is supported");
      this.classes = await classNames();
      await this.loadPanels();
      if (this.spec.overlayMenu && this.panels[0]?.o) {          // labels for the overlay menu
        this.setLabels = {};
        await Promise.all(this.spec.overlayMenu.map(async it => { const id = typeof it === "string" ? it : it.set; try { this.setLabels[id] = (await loadSet(this.panels[0].o, "ov", id)).s.label; } catch (e) { /* shown when drawn */ } }));
        this.buildControls();
      }
    } catch (e) { this.fatal(e); }
  }
  fatal(e) {
    console.error("emb-fig:", e);
    this.el.classList.add("ef-failed");
    this.grid.replaceChildren(h("div", { class: "ef-error" }, h("b", { text: "Figure not drawn. " }), esc(e.message)));
  }
  async loadPanels() {
    const c = this.cur();
    if (!c.panels || !c.panels.length) throw new DataError("figure spec has no panels");
    const keep = this.panels;
    let items = c.panels.map((item, i) => ({ item: keep[i] && keep[i].swapped && keep[i].base === item ? keep[i].item : item, base: item, idx: i, swapped: keep[i]?.swapped && keep[i].base === item }));
    if (this.sideBySide()) {
      const main = this.overlaysNow().find(x => x.main), ctl = this.overlaysNow().find(x => x.isControl);
      items = [{ ...items[0], role: "main", title: main?.label || null }, { ...items[0], idx: 1, role: "control", title: ctl?.label || "control" }];
    }
    const tok = this.loadTok = (this.loadTok || 0) + 1;      // a later switch, overlay or control change supersedes this one
    this.panels = items;
    this.layoutPanels();
    await Promise.all(this.panels.map(p => this.loadPanel(p)));
    if (tok !== this.loadTok) return;
    if (this.sideBySide()) for (const p of this.panels) p.title = this.setTitle(p);
    this.afterLoad();
  }
  setTitle(p) { const st = (p.sets || [])[0]; return st ? (st.cfg.label || st.s.label) + (p.role === "control" ? " (control)" : "") : p.item.label; }
  // A panel being (re)loaded draws nothing: its projected arrays are dropped first and its canvas is cleared, and the
  // new operator and sets are assigned together once all of them have loaded (p.ready).
  async loadPanel(p) {
    const tok = p.loadTok = (p.loadTok || 0) + 1;
    p.err = null; p.ready = false; p.probe = null; p.box = null; p.hot = null;
    if (p.cv) { const r = this.fit(p.cv); if (r) { r.g.fillStyle = this.P.surface; r.g.fillRect(0, 0, r.w, r.h); } }
    try {
      const o = await loadOp(this.root, p.item.op);
      const want = this.kindWanted ?? this.kind;
      if (!o.op.kinds[want]) {
        const k = (this.spec.kinds || []).find(k => o.op.kinds[k]) || "joint";
        if (this.panels.every(q => q === p || !q.o || q.o.op.kinds[k])) this.kindWanted = k;
      }
      const { sets, paths, sprite } = await this.loadPanelSets(p, o);
      if (tok !== p.loadTok) return;
      p.o = o; p.sets = sets; p.paths = paths; p.sprite = sprite; p.ready = true;
    } catch (e) { if (tok !== p.loadTok) return; p.o = null; p.sets = []; p.paths = []; p.err = e; console.error(`emb-fig panel ${p.item.op}:`, e); }
  }
  async loadPanelSets(p, o) {
    const c = this.cur(), sets = [], paths = [];
    for (const ov of this.panelOverlays(p)) {
      const { s, url } = await loadSet(o, "ov", ov.set);
      let from = null;
      if (ov.draw === "arrows") {
        const fid = ov.from || s.pairs?.set;
        if (!fid) throw new DataError(`overlay ${ov.set} is drawn as arrows but names no paired set`);
        if (s.pairs && s.pairs.set !== fid) throw new DataError(`overlay ${ov.set} is paired with ${s.pairs.set}, not ${fid}`);
        from = await loadSet(o, "ov", fid);
        if (from.s.n !== s.n || JSON.stringify(from.s.sample?.pos) !== JSON.stringify(s.sample?.pos))
          throw new DataError(`Data mismatch: overlay ${ov.set} and its paired set ${fid} do not hold the same sampled images in the same order`);
      }
      sets.push({ cfg: ov, s, url, from });
    }
    for (const id of c.paths || []) {
      const { s, url } = await loadSet(o, "paths", id);
      if (JSON.stringify(s.chain) !== JSON.stringify(o.op.chain)) throw new DataError(`path set ${id}: chain [${s.chain}] differs from its operator's chain [${o.op.chain}]`);
      paths.push({ s, url });
    }
    return { sets, paths, sprite: o.op.probe.sprite ? getImage(rel(o.op.probe.sprite, o.url)).catch(() => null) : null };
  }
  afterLoad() {
    const L = this.chain().length;
    // site: keep the current site name if the new chain has it, else the spec's site, else the first
    const want = this.siteWanted ?? this.cur().site;
    const i = want != null ? this.chain().indexOf(want) : -1;
    if (i < 0 && want != null && this.siteWanted == null) console.warn(`emb-fig: site ${want} not in chain [${this.chain()}]`);
    this.b = i >= 0 ? i : 0;
    this.siteWanted = null;
    this.range.max = Math.max(1, L - 1); this.range.disabled = L < 2; this.playBtn.disabled = L < 2; this.range.value = this.b;
    this.ticks.replaceChildren(...this.chain().map((s, l) => h("button", { type: "button", style: `left:${L > 1 ? 100 * l / (L - 1) : 50}%`, text: s, onclick: () => { this.stop(); this.setSite(l); } })));
    this.buildControls();
    this.recompute();
  }
  chain() { const p = this.panels.find(p => p.o); return p ? p.o.op.chain : []; }

  // ---------------------------------------------------- interactions
  // the kind is committed together with its projected coordinates (recompute); until then the old kind stays drawn
  async setKind(k) {
    this.stop(); this.kindWanted = k;
    const req = this.kindReq = (this.kindReq || 0) + 1;
    await Promise.all(this.panels.map(p => p.ready && !p.err ? null : this.loadPanel(p)));
    if (req === this.kindReq) await this.recompute();
  }
  async setSwitch(i) {
    this.stop(); this.siteWanted = this.chain()[Math.round(this.b)];
    this.sw = i; this.panels = [];
    const k = this.spec.switch[i].kind; if (k) this.kindWanted = k;
    this.status.textContent = "Loading…";
    await this.loadPanels();
    this.status.textContent = "";
  }
  async reloadPanels() { this.stop(); this.siteWanted = this.chain()[Math.round(this.b)]; await this.loadPanels(); }
  async swapPanel(p, item) { this.stop(); p.item = item; p.swapped = item !== p.base; const tok = (p.loadTok || 0) + 1; await this.loadPanel(p); if (tok === p.loadTok) this.recompute(); }
  arrowKeep(n) { const k = Math.min(n, this.density); return k >= n ? () => true : j => (j * 7919) % n < k; }
  isoNow() { return this.isoPreview ?? this.iso; }
  togglePlay() { if (this.playing) this.stop(); else this.play(); }
  play() {
    const L = this.chain().length; if (L < 2) return;
    if (this.b >= L - 1 - 1e-6) this.b = 0;
    this.playing = true; this.setPlayIcon();
    let t0 = null; const b0 = this.b;
    const step = t => {
      if (!this.playing) return;
      if (t0 == null) t0 = t;
      const b = Math.min(L - 1, b0 + (t - t0) / SITE_MS);
      this.setSite(b, true);
      if (b >= L - 1) { this.stop(); return; }
      this.raf = requestAnimationFrame(step);
    };
    this.raf = requestAnimationFrame(step);
  }
  stopAnim() { if (this.playing) { this.playing = false; cancelAnimationFrame(this.raf); this.setPlayIcon(); } }
  stop() { if (!this.playing) return; this.stopAnim(); this.setSite(Math.round(this.b)); }
  setSite(b, live) {
    this.b = b;
    if (!live || this.playing) this.range.value = b;
    this.draw();
  }

  // ---------------------------------------------------- geometry
  // Per panel and kind: one K x 2 display map per site, M_l = R V_l / |probe_l|_F (norm "site"), and the probe,
  // overlay and path coordinates projected by it. Axis limits are fixed over all sites and shared by all panels.
  // Every panel is projected for the requested kind first; the kind, the arrays, the limits and the legend are then
  // committed together, and only by the latest request (a superseded one is dropped).
  async recompute(kind = this.kindWanted ?? this.kind) {
    const token = (this.token = (this.token || 0) + 1);
    const commits = [];
    for (const p of this.panels) if (p.o && p.ready && !p.err) {
      try { commits.push(await this.project(p, kind)); } catch (e) { commits.push(() => { p.err = e; p.probe = null; console.error(e); }); }
    }
    if (token !== this.token) return;
    this.kind = kind; this.kindWanted = null;
    for (const c of commits) { try { c(); } catch (e) { console.error(e); } }
    this.fitBox(); this.renderPanels(); this.buildControls(); this.describe(); this.drawFilm(); this.draw();
  }
  axesFor(p, G) {
    const spec = this.cur(), o = p.o, K = o.K, a = spec.axes || "pooled";
    if (a === "viewer") {
      if (!o.op.axes?.viewer) throw new DataError(`operator ${o.id} has no viewer axes (axes: "viewer" needs them)`);
      return o.op.axes.viewer.map(r => r.slice(0, 2));
    }
    const src = a.startsWith("pooled:") ? this.panels[+a.split(":")[1]] : p;
    const so = src && src.o ? src.o : o;
    if (so.op.axes?.pooled && so.K === K) return so.op.axes.pooled.map(r => r.slice(0, 2));
    const C = new Float64Array(K * K);           // not packaged: pooled PCA of this panel's probe, as the packager does
    for (const g of G) { const X = crossTN(g, g, o.N, K); for (let j = 0; j < C.length; j++) C[j] += X[j]; }
    const { V } = eigSym(C, K);
    return [...Array(K).keys()].map(k => [V[k * K], V[k * K + 1]]);
  }
  async project(p, kind) {
    const o = p.o, K = o.K, L = o.L, N = o.N, spec = this.cur();
    if (spec.K != null && spec.K !== K) throw new DataError(`the figure asks for K = ${spec.K}; operator ${o.id} has K = ${K}`);
    const raw = await kindData(o, kind);
    const norms = (spec.norm || "site") === "site" ? (o.op.site_norm?.[kind] || null) : null;
    const signs = spec.axes === "viewer" && o.op.viewer_signs ? o.op.viewer_signs : null;
    const R = kind !== "site" && p.item.R ? p.item.R : null;
    if (R && (R.length !== K || R[0].length !== K)) throw new DataError(`panel ${p.item.label}: R is ${R.length} x ${R[0].length}, operator K = ${K}`);
    const slice = (arr, l) => arr.subarray(l * N * K, (l + 1) * N * K);
    const nrm = l => { if ((spec.norm || "site") !== "site") return 1; if (norms) return norms[l] || 1; let f = 0; const g = slice(raw, l); for (let j = 0; j < g.length; j++) f += g[j] * g[j]; return Math.sqrt(f) || 1; };
    const M = [];
    if (kind === "site") {
      // each site's own axes: that site's walk modes 1 and 2 as they are (their columns are orthonormal, so a PCA of
      // them would be arbitrary). Eigenvector signs are arbitrary: each is chosen to agree with the previous site's
      // picture, so the cloud does not flip between sites. Stated in the operator line.
      let prev = null;
      for (let l = 0; l < L; l++) {
        const g = slice(raw, l), f = nrm(l), sg = [1, 1];
        if (prev) for (let c = 0; c < 2; c++) { let d = 0; for (let i = 0; i < N; i++) d += g[i * K + c] * prev[i * K + c]; sg[c] = d < 0 ? -1 : 1; }
        M.push([...Array(K).keys()].map(k => [k === 0 ? sg[0] / f : 0, k === 1 ? sg[1] / f : 0]));
        const cur = new Float32Array(g); for (let i = 0; i < N; i++) { cur[i * K] *= sg[0]; cur[i * K + 1] *= sg[1]; }
        prev = cur;
      }
    } else {
      const G = [];
      if ((spec.axes || "pooled").startsWith("pooled") && !o.op.axes?.pooled)
        for (let l = 0; l < L; l++) { const g = Float64Array.from(slice(raw, l)), f = nrm(l); for (let j = 0; j < g.length; j++) g[j] /= f; G.push(g); }
      const V = this.axesFor(p, G);
      const RV = R ? R.map(row => [0, 1].map(c => row.reduce((s, x, k) => s + x * V[k][c], 0))) : V;          // rows y -> y R V
      for (let l = 0; l < L; l++) { const f = nrm(l); M.push(RV.map((r, k) => [r[0] * (signs ? signs[k] : 1) / f, r[1] * (signs ? signs[k] : 1) / f])); }
    }
    const proj = (arr, n) => {          // [L, n, K] -> [L, n, 2]
      const out = new Float32Array(L * n * 2);
      for (let l = 0; l < L; l++) {
        const m = M[l], o0 = l * n * K, o1 = l * n * 2;
        for (let i = 0; i < n; i++) { let x = 0, y = 0; const b = o0 + i * K; for (let k = 0; k < K; k++) { const v = arr[b + k]; x += v * m[k][0]; y += v * m[k][1]; } out[o1 + i * 2] = x; out[o1 + i * 2 + 1] = y; }
      }
      return out;
    };
    // everything is projected first and assigned together, so a draw never mixes two kinds or two states
    const probe = proj(raw, N), setXY = [], pathXY = [];
    // share of each site's probe norm (as scaled) that lies in the plotted plane: printed under filmstrip frames
    const plane = [];
    for (let l = 0; l < L; l++) {
      const g = slice(raw, l), f = kind === "site" ? nrm(l) : 1; let a = 0, t = 0;
      for (let j = 0; j < g.length; j++) t += g[j] * g[j];
      for (let j = l * N * 2; j < (l + 1) * N * 2; j++) a += probe[j] * probe[j];
      const den = kind === "site" ? t / (f * f) : t / (nrm(l) ** 2);
      plane.push(den > 0 ? a / den : NaN);
    }
    for (const st of p.sets) {
      const xy = {};
      for (const which of ["s", "from"]) {
        if (which === "from" && !st.from) continue;
        const set = which === "s" ? st.s : st.from.s;
        if (!set.kinds[kind]) throw new DataError(`overlay ${set.id} has no ${kind} coordinates`);
        const sh = set.kinds[kind].shape; if (sh[0] !== L || sh[1] !== set.n || sh[2] !== K) throw new DataError(`overlay ${set.id}: shape [${sh}] does not match the operator`);
        xy[which] = proj(await loadArray(set.kinds[kind], which === "s" ? st.url : st.from.url), set.n);
      }
      setXY.push(xy);
      if (st.s.sprite) st.sprite = getImage(rel(st.s.sprite, st.url)).catch(() => null);
    }
    for (const pt of p.paths) {
      const s = pt.s; if (!s.kinds[kind]) throw new DataError(`path set ${s.id} has no ${kind} coordinates`);
      const sh = s.kinds[kind].shape; if (sh[0] !== L || sh[1] !== s.P || sh[2] !== s.S || sh[3] !== K) throw new DataError(`path set ${s.id}: shape [${sh}] does not match chain ${L}, P ${s.P}, S ${s.S}, K ${K}`);
      pathXY.push(proj(await loadArray(s.kinds[kind], pt.url), s.P * s.S));
    }
    const sets = p.sets, paths = p.paths;
    return () => {
      if (p.sets !== sets || p.paths !== paths) return;          // the panel was reloaded meanwhile
      p.M = M; p.probe = probe; p.plane = plane; p.err = null;
      sets.forEach((st, i) => { st.xy = setXY[i].s; st.fromXY = setXY[i].from || null; });
      paths.forEach((pt, i) => { pt.xy = pathXY[i]; });
      p.box = this.extent(p);
      this.colourFor(p);
    };
  }
  // the panel's data extent over all sites: 0.3% and 99.7% quantiles of x and y over the probe and every overlay and path
  extent(p) {
    const xs = [], ys = [], add = a => { for (let j = 0; j < a.length; j += 2) { xs.push(a[j]); ys.push(a[j + 1]); } };
    add(p.probe); for (const st of p.sets) { add(st.xy); if (st.fromXY) add(st.fromXY); } for (const pt of p.paths) add(pt.xy);
    const q = (a, t) => { const s = Float32Array.from(a).sort(); return s[Math.floor(t * (s.length - 1))]; };
    return { x0: q(xs, .003), x1: q(xs, .997), y0: q(ys, .003), y1: q(ys, .997) };
  }
  // one box for the whole figure (all panels share axes and limits), padded by 4%
  fitBox() {
    const bs = this.panels.filter(p => p.box && !p.err).map(p => p.box);
    if (!bs.length) { this.box = null; return; }
    let x0 = Math.min(...bs.map(b => b.x0)), x1 = Math.max(...bs.map(b => b.x1)), y0 = Math.min(...bs.map(b => b.y0)), y1 = Math.max(...bs.map(b => b.y1));
    const pw = (x1 - x0) * .04 || 1e-3, ph = (y1 - y0) * .04 || 1e-3;
    this.box = { x0: x0 - pw, x1: x1 + pw, y0: y0 - ph, y1: y1 + ph };
    this.aspect = Math.max(0.55, Math.min(1.05, (this.box.y1 - this.box.y0) / (this.box.x1 - this.box.x0)));
  }
  // colours: class colours on the probe by default; a per-image scalar on an overlay set (or the probe) on request
  // colour.options = [{label, scalar | values}, ...] offers several scalars behind a selector; the colour range is fixed
  // across options and sites (colour.range, else the union of every option's values) and is stated in the legend
  colourFor(p) {
    const col = this.cur().colour || { by: "class" };
    p.scalar = null;
    if (col.by !== "scalar") return;
    const target = col.set && col.set !== "probe" ? col.set : "probe";
    const st = target === "probe" ? null : p.sets.find(s => s.s.id === target);
    if (target !== "probe" && !st) return;                 // that set is not drawn in this panel
    // the legend names the quantity (the packaged scalar's own label, else colour.label); an option's label only
    // names the choice in the selector
    const resolve = opt => {
      if (opt.values) return { values: opt.values, label: col.label ?? opt.label, choice: opt.label, range: opt.range };
      const sc = target === "probe" ? p.o.op.scalars?.[opt.scalar] : st.s.meta?.scalars?.[opt.scalar];
      if (!sc) throw new DataError(`${target === "probe" ? `operator ${p.o.id}` : `overlay ${target}`} has no scalar ${opt.scalar}`);
      const n = target === "probe" ? p.o.N : st.s.n, L = p.o.L, v = sc.values;
      if (!(Array.isArray(v[0]) ? v.length === L && v[0].length === n : v.length === n)) throw new DataError(`scalar ${opt.scalar}: expected [${n}] or [${L} x ${n}] values`);
      return { values: v, label: sc.label ?? col.label ?? opt.label, choice: opt.label, range: opt.range ?? sc.range };
    };
    const opts = (col.options && col.options.length ? col.options : [col]).map(resolve);
    const one = opts[Math.min(this.scalarOpt, opts.length - 1)];
    // colour range, fixed over sites and options: colour.range; else colour.fit = [lo, hi] percentiles of every option's
    // values at every site (values beyond are drawn at the end colours); else the packaged ranges or the full extent
    let range = col.range, fit = null;
    if (!range && Array.isArray(col.fit)) {
      const flat = Float64Array.from(opts.flatMap(o => o.values.flat(2)).filter(Number.isFinite)).sort();
      const q = t => flat[Math.max(0, Math.min(flat.length - 1, Math.round(t / 100 * (flat.length - 1))))];
      if (flat.length) { range = [q(col.fit[0]), q(col.fit[1])]; fit = col.fit; }
    }
    if (!range) { const flat = (opts.length > 1 ? opts : [one]).flatMap(o => o.range || o.values.flat(2)).filter(Number.isFinite); range = [Math.min(...flat), Math.max(...flat)]; }
    if (!(range[1] > range[0])) range = [range[0] - 0.5, range[0] + 0.5];
    p.scalar = { target, vals: one.values, perSite: Array.isArray(one.values[0]), label: one.label || col.scalar, choice: opts.length > 1 ? one.choice : null, range, fit, nOpts: opts.length, optLabels: opts.map(o => o.choice), title: col.title };
  }

  // ---------------------------------------------------- panels and layout
  layoutPanels() {
    this.grid.replaceChildren();
    for (const p of this.panels) {
      p.cv = h("canvas", { tabindex: 0 }); p.cap = h("div", { class: "ef-cap" }); p.err_el = h("div", { class: "ef-error", hidden: true });
      p.fig = h("div", { class: "ef-panel" }, p.cap, p.cv, p.err_el);
      this.grid.append(p.fig);
      this.bindHover(p);
    }
    this.layout();
  }
  layout() {
    const w = this.el.clientWidth || 800, n = this.panels.length;
    const narrow = w < NARROW && n > 1;
    if (narrow !== this.narrow) { this.narrow = narrow; if (this.panels.length) this.buildControls(); }
    this.el.dataset.n = n; this.el.classList.toggle("ef-narrow", narrow);
    // layout "row": all panels in one row with the legend below, when each panel gets at least ~240 px
    this.el.classList.toggle("ef-row", this.cur().layout === "row" && n > 1 && w >= 250 * n);
    this.el.style.setProperty("--ef-n", String(n));
    this.panels.forEach((p, i) => { if (p.fig) p.fig.hidden = narrow && n > 2 && i !== this.focus; });
    // canvas shape follows the data's aspect (equal scales on both axes); one panel: up to spec.height tall
    const a = this.aspect || 0.8, maxH = this.cur().height || 520;
    this.el.style.setProperty("--ef-aspect", String(1 / a));
    this.el.style.setProperty("--ef-h", `${Math.round(Math.min(maxH, Math.max(300, this.grid.clientWidth * a)))}px`);
    this.draw(); this.drawFilm();
  }
  renderPanels() {
    const menu = this.panelMenu();
    for (const p of this.panels) {
      p.cv.hidden = !!p.err || !p.o; p.err_el.hidden = !p.err;
      if (p.err) p.err_el.innerHTML = `<b>${p.err.message.startsWith("Data mismatch") ? "Not drawn: data mismatch." : "Not drawn."}</b> ${esc(p.err.message.replace(/^Data mismatch: /, ""))}`;
      p.cap.replaceChildren();
      const canSwap = !p.role && p.idx > 0 && menu.length > 1 && (p.base.R != null || this.spec.panelMenu);
      if (canSwap) {
        const sel = h("select", { "aria-label": `Panel ${p.idx + 1}: network` });
        for (const it of menu) sel.append(h("option", { value: it.op, selected: it.op === p.item.op || null, text: it.label }));
        sel.addEventListener("change", () => this.swapPanel(p, menu.find(it => it.op === sel.value)));
        p.cap.append(sel);
      } else p.cap.append(h("b", { class: "ef-cap-label", text: p.title || p.item.label || p.o?.op.title || p.item.op }));
      p.statEl = h("span", { class: "ef-stat" }); p.spark = h("canvas", { class: "ef-spark", "aria-hidden": "true" });
      if (p.item.stat) p.cap.append(h("div", { class: "ef-statrow", title: p.item.stat.label }, p.spark, p.statEl));
    }
    this.layout();
  }
  panelMenu() {
    const s = this.cur(), out = [], seen = new Set();
    for (const it of [...(s.panels || []).slice(1), ...(s.panelMenu || [])]) if (!seen.has(it.op)) { seen.add(it.op); out.push(it); }
    return out;
  }

  // ---------------------------------------------------- filmstrip: small static panels at chosen sites (first panel)
  filmSites() {
    const f = this.cur().filmstrip; if (!f) return [];
    const list = Array.isArray(f) ? f : f.sites || [], ch = this.chain();
    return list.map(x => typeof x === "number" ? x : ch.indexOf(x)).filter(l => l >= 0 && l < ch.length);
  }
  drawFilm() {
    const sites = this.filmSites(), p = this.panels.find(q => q.o && !q.err && q.probe);
    this.filmWrap.hidden = !sites.length || !p || !this.box;
    if (this.filmWrap.hidden) return;
    const axesTxt = this.kind === "site" ? "each site on its own axes (site-local), same limits" : "same axes and limits";
    this.filmTitle.textContent = `${this.panels.length > 1 ? `${p.title || p.item.label}, at` : "At"} ${sites.length} sites, ${axesTxt}; under each, the share of that site's squared probe norm${this.kind === "site" ? "" : " (as scaled)"} lying in the plotted plane. Click one to show it above.`;
    const f = this.cur().filmstrip, stat = !Array.isArray(f) && f.stat, ch = this.chain();
    if (this.filmKey !== sites.join() + "|" + ch.join()) {
      this.filmKey = sites.join() + "|" + ch.join();
      this.film.replaceChildren(...sites.map(l => {
        const cv = h("canvas"), b = h("button", { type: "button", class: "ef-frame", "aria-label": `Show ${ch[l]}` }, h("span", { class: "ef-frame-site", text: ch[l] }), cv, h("span", { class: "ef-frame-plane" }), stat ? h("span", { class: "ef-frame-stat" }) : null);
        b.addEventListener("click", () => { this.stop(); this.setSite(l); });
        b.dataset.l = l; return b;
      }));
    }
    // stat.kind (a kind or a list of kinds): the stat is printed only while those coordinates are shown
    const statOn = stat && (!stat.kind || [].concat(stat.kind).includes(this.kind)), sv = statOn && stat.values;
    [...this.film.children].forEach(b => {
      const l = +b.dataset.l, cv = b.querySelector("canvas"), r = this.fit(cv);
      if (r) this.paint(p, r.g, r.w, r.h, l, l, true);
      const s = b.querySelector(".ef-frame-stat"); if (s) { s.hidden = !statOn; s.textContent = sv && Number.isFinite(sv[l]) ? `${stat.label ? stat.label + " " : ""}${fmt(sv[l])}` : "–"; }
      const pl = b.querySelector(".ef-frame-plane"), f = p.plane?.[l]; if (pl) pl.textContent = Number.isFinite(f) ? `${f >= 0.095 ? Math.round(100 * f) : (100 * f).toFixed(1)}% of norm in view` : "";
    });
    this.markFilm();
  }
  markFilm() { const lc = Math.round(this.b); [...this.film.children].forEach(b => b.classList.toggle("on", +b.dataset.l === lc)); }

  // ---------------------------------------------------- drawing
  fit(cv) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, hh = cv.clientHeight;
    if (!w || !hh) return null;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(hh * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(hh * dpr); }
    const g = cv.getContext("2d"); g.setTransform(dpr, 0, 0, dpr, 0, 0); return { g, w, h: hh };
  }
  frameXY(arr, n, L, b, out) {          // positions at fractional site b (linear between neighbouring sites)
    const l0 = Math.max(0, Math.min(L - 2, Math.floor(b))), a = L < 2 ? 0 : Math.max(0, Math.min(1, b - l0));
    const A = l0 * n * 2, B = Math.min(L - 1, l0 + 1) * n * 2;
    for (let j = 0; j < n * 2; j++) out[j] = (1 - a) * arr[A + j] + a * arr[B + j];
    return out;
  }
  redraw() { this.draw(); this.drawFilm(); }
  draw() {
    if (!this.panels.length) return;
    const L = this.chain().length; if (!L) return;
    const tween = this.kind !== "site" && this.playing;            // site-local axes differ by site: step, do not blend
    const b = tween ? this.b : Math.round(this.b), lc = Math.round(this.b), ch = this.chain();
    const near = Math.abs(b - lc) < 0.02;
    this.siteName.textContent = near ? ch[lc] : `${ch[Math.floor(b)]} → ${ch[Math.min(L - 1, Math.floor(b) + 1)]}`;
    this.siteSub.textContent = `site ${lc + 1} of ${L}`;
    this.range.setAttribute("aria-valuetext", `${ch[lc]}, site ${lc + 1} of ${L}`);
    // site labels: all when they fit, otherwise every step-th, always the current one and the last, none crowding them
    const step = Math.max(1, Math.ceil(L * 42 / Math.max(1, this.ticks.clientWidth)));
    const far = (l, m) => Math.abs(l - m) >= step;
    [...this.ticks.children].forEach((t, l) => {
      const show = l === lc || (l === L - 1 ? far(l, lc) : l % step === 0 && far(l, lc) && far(l, L - 1));
      t.classList.toggle("on", l === lc); t.classList.toggle("hide", !show);
    });
    for (const p of this.panels) if (p.o && !p.err && p.probe && !p.fig.hidden) this.drawPanel(p, b, lc);
    for (const p of this.panels) this.drawStat(p, lc);
    this.markFilm();
  }
  drawStat(p, lc) {
    const st = p.item.stat; if (!st || !p.statEl) return;
    const v = st.values?.[lc];
    p.statEl.replaceChildren(h("b", { text: fmt(v) }), ` ${st.label} at ${this.chain()[lc] ?? ""}`);       // the value first: a narrow panel truncates the label, not the number
    const r = this.fit(p.spark); if (!r) return;
    const { g, w, h: hh } = r, vals = st.values || [], ok = vals.filter(Number.isFinite);
    g.clearRect(0, 0, w, hh); if (ok.length < 2) return;
    const lo = Math.min(0, ...ok), hi = Math.max(...ok, lo + 1e-9), X = i => 2 + (w - 4) * i / Math.max(1, vals.length - 1), Y = v => hh - 2 - (hh - 4) * (v - lo) / (hi - lo);
    g.strokeStyle = rgba(this.P.mute, 0.9); g.lineWidth = 1.2; g.beginPath();
    vals.forEach((v, i) => { if (Number.isFinite(v)) i ? g.lineTo(X(i), Y(v)) : g.moveTo(X(i), Y(v)); }); g.stroke();
    if (Number.isFinite(v)) { g.fillStyle = rgba(this.P.fg, 1); g.beginPath(); g.arc(X(lc), Y(v), 2.4, 0, 7); g.fill(); }
  }
  drawPanel(p, b, lc) {
    const r0 = this.fit(p.cv); if (!r0) return;
    const scr = this.paint(p, r0.g, r0.w, r0.h, b, lc, false);
    Object.assign(p, scr);
    if (p.hot) {
      const { g } = r0, P = this.P, { x, y, from } = p.hot;
      if (from) { g.lineWidth = 2.4; g.strokeStyle = rgba(P.fg, 0.9); g.beginPath(); g.moveTo(from[0], from[1]); g.lineTo(x, y); g.stroke(); g.fillStyle = "#fff"; g.beginPath(); g.arc(from[0], from[1], 3.2, 0, 7); g.fill(); g.lineWidth = 1.4; g.stroke(); }
      g.lineWidth = 3; g.strokeStyle = "rgba(255,255,255,.95)"; g.beginPath(); g.arc(x, y, 6, 0, 7); g.stroke();
      g.lineWidth = 1.5; g.strokeStyle = rgba(P.fg, 1); g.beginPath(); g.arc(x, y, 6, 0, 7); g.stroke();
    }
  }
  // class isolation of paired overlays: the pairs whose main-overlay image is in the isolated class (null: no pairing)
  pairFilter(p, iso) {
    if (iso == null) return null;
    const main = p.sets.find(st => st.cfg.main && sameSpace(st.s) && st.s.meta?.pair && st.s.meta?.label);
    if (!main) return null;
    const keep = new Set(); main.s.meta.label.forEach((l, j) => { if (l === iso) keep.add(main.s.meta.pair[j]); });
    keep.main = main; return keep;
  }
  // draw one panel's picture at fractional site b into a 2-D context; small = filmstrip frame (no labels, no hover data)
  paint(p, g, w, hh, b, lc, small) {
    const P = this.P, o = p.o, N = o.N, L = o.L, box = this.box, iso = this.isoNow(), thumb = this.thumb;
    g.fillStyle = P.surface; g.fillRect(0, 0, w, hh);
    if (!box) return {};
    const pad = small ? 3 : 8, S = Math.min((w - 2 * pad) / (box.x1 - box.x0), (hh - 2 * pad) / (box.y1 - box.y0));
    const cx = w / 2 - S * (box.x0 + box.x1) / 2, cy = hh / 2 + S * (box.y0 + box.y1) / 2;
    const X = x => cx + S * x, Y = y => cy - S * y;
    // Nothing is moved to stay in view: arrows and paths that leave the panel are cut at its edge (clip region), an
    // arrowhead or endpoint marker outside it is not drawn, and the panel notes how many were cut.
    const inside = (x, y) => x >= 0 && x <= w && y >= 0 && y <= hh;
    const off = { arrows: 0, paths: 0, points: 0 };
    g.save(); g.beginPath(); g.rect(0, 0, w, hh); g.clip();
    // axes: faint lines through the origin, labelled
    if (!thumb) {
      g.strokeStyle = P.line; g.lineWidth = 1; g.beginPath(); g.moveTo(0, Y(0)); g.lineTo(w, Y(0)); g.moveTo(X(0), 0); g.lineTo(X(0), hh); g.stroke();
    }
    if (!small && !thumb) {
      g.fillStyle = rgba(P.mute, 0.9); g.font = "10.5px Inter, system-ui, sans-serif"; g.textAlign = "right";
      g.fillText(this.kind === "site" ? "mode 1" : "PC 1", w - 4, Y(0) - 5); g.textAlign = "left"; g.fillText(this.kind === "site" ? "mode 2" : "PC 2", X(0) + 5, 11);
    }
    const busy = p.sets.length > 0 || p.paths.length > 0, scal = p.scalar, withPaths = p.paths.length > 0;
    const r = small ? Math.max(0.8, Math.min(1.3, w / 160)) : Math.max(1.2, Math.min(2.6, w / 330)) * (busy ? 0.85 : 1);
    const pts = this.frameXY(p.probe, N, L, b, (small ? (p.fbufP ||= new Float32Array(N * 2)) : (p.buf ||= new Float32Array(N * 2))));
    const screen = new Float32Array(N * 2);
    for (let i = 0; i < N; i++) { screen[i * 2] = X(pts[i * 2]); screen[i * 2 + 1] = Y(pts[i * 2 + 1]); }
    const probeScalar = scal && scal.target === "probe", lab = o.labels;
    const groups = new Map();
    for (let i = 0; i < N; i++) {
      let key;
      if (probeScalar) { const v = scal.perSite ? scal.vals[lc][i] : scal.vals[i]; key = Number.isFinite(v) ? qLevel((v - scal.range[0]) / (scal.range[1] - scal.range[0])) : -1; }
      else if (scal) key = -1;
      else key = lab ? lab[i] : -1;
      if (iso != null && !probeScalar && !scal && key !== iso) key = -2;
      (groups.get(key) || groups.set(key, []).get(key)).push(i);
    }
    const alpha = withPaths ? 0.24 : busy ? 0.34 : 0.8;          // under paths the cloud is a quiet backdrop
    for (const [key, idx] of [...groups].sort((a, b) => (a[0] === -2 ? -1 : 0) - (b[0] === -2 ? -1 : 0))) {
      const c = key === -2 ? P.grey : probeScalar ? (key < 0 ? P.grey : seqColour(P, key / (LEVELS - 1))) : key < 0 || key >= P.classes.length ? P.grey : P.classes[key];
      g.fillStyle = rgba(c, key === -2 ? 0.35 : scal && !probeScalar ? 0.5 : probeScalar ? 0.9 : iso != null ? 0.95 : alpha); g.beginPath();
      for (const i of idx) { const x = screen[i * 2], y = screen[i * 2 + 1]; g.moveTo(x + r, y); g.arc(x, y, r, 0, 6.2832); }
      g.fill();
    }
    // rings around chosen probe images
    const hl = p.item.highlight || this.cur().highlight;           // a panel's own highlight replaces the figure's
    const ring = (x, y, rr) => { g.lineWidth = 3.5; g.strokeStyle = "rgba(255,255,255,.95)"; g.beginPath(); g.arc(x, y, rr, 0, 7); g.stroke(); g.lineWidth = 1.6; g.strokeStyle = rgba(P.fg, 1); g.beginPath(); g.arc(x, y, rr, 0, 7); g.stroke(); };
    if (hl && hl.probe && !small && !thumb) for (const i of hl.probe) if (i >= 0 && i < N) ring(screen[i * 2], screen[i * 2 + 1], r + 5);
    // paths: the polyline over s at the current site, coloured by s (light at a, dark at b), with an arrowhead at s = 1/2;
    // a = open circle, b = filled square; different-class solid and wider, same-class dashed and thinner. Hovering a
    // path draws it on top and fades the others.
    const pathScreen = [], hotPath = !small && p.hot?.path ? p.hot.path : null;
    const pathCol = t => seqColour(P, 0.12 + 0.88 * t);
    for (const pt of p.paths) {
      if (!pt.xy) continue;
      const s = pt.s, n = s.P * s.S, xy = this.frameXY(pt.xy, n, L, b, new Float32Array(n * 2)), sv = s.s, s0 = sv[0], s1 = sv[sv.length - 1], m = s.meta || {};
      const sc = new Float32Array(n * 2); for (let j = 0; j < n; j++) { sc[j * 2] = X(xy[j * 2]); sc[j * 2 + 1] = Y(xy[j * 2 + 1]); }
      const isoOK = q => iso == null || !sameSpace(s) || m.label_a?.[q] === iso || m.label_b?.[q] === iso;
      const kindOK = q => this.pathKind === "all" || (this.pathKind === "diff") === (m.kind?.[q] === "diff");
      const order = [...Array(s.P).keys()].filter(q => isoOK(q) && kindOK(q));
      pathScreen.push({ pt, sc, shown: new Set(order) });
      if (hotPath && hotPath.pt === pt) order.sort((a, c) => (a === hotPath.q) - (c === hotPath.q));
      for (const q of order) {
        const diff = m.kind?.[q] === "diff", base = q * s.S;
        const faded = hotPath && !(hotPath.pt === pt && hotPath.q === q), hot = hotPath && !faded;
        const lw = small ? (diff ? 1.3 : 1) : (diff ? 2.5 : 1.8) + (hot ? 0.8 : 0);
        let out = false; for (let j = 0; j < s.S; j++) if (!inside(sc[(base + j) * 2], sc[(base + j) * 2 + 1])) { out = true; break; }
        if (out) off.paths++;
        g.globalAlpha = faded ? 0.16 : 1;
        g.setLineDash([]); g.lineWidth = lw + (small ? 1.4 : 2.6); g.strokeStyle = "rgba(255,255,255,.85)"; g.lineJoin = "round"; g.beginPath();
        for (let j = 0; j < s.S; j++) { const k = (base + j) * 2; j ? g.lineTo(sc[k], sc[k + 1]) : g.moveTo(sc[k], sc[k + 1]); } g.stroke();
        // segments coloured by s; the dash pattern runs on along the whole path (offset = length so far)
        g.lineWidth = lw; g.setLineDash(diff ? [] : small ? [2, 2] : [5, 3.5]); let run = 0;
        for (let j = 1; j < s.S; j++) {
          const k0 = (base + j - 1) * 2, k1 = (base + j) * 2;
          g.lineDashOffset = run; run += Math.hypot(sc[k1] - sc[k0], sc[k1 + 1] - sc[k0 + 1]);
          g.strokeStyle = rgba(pathCol((sv[j] - s0) / (s1 - s0 || 1)), 1); g.beginPath(); g.moveTo(sc[k0], sc[k0 + 1]); g.lineTo(sc[k1], sc[k1 + 1]); g.stroke();
        }
        g.setLineDash([]); g.lineDashOffset = 0;
        if (!small && !thumb) {
          // direction: an arrowhead where s = 1/2, along the path's local direction
          let jm = 0; for (let j = 0; j < s.S; j++) if (Math.abs((sv[j] - s0) / (s1 - s0 || 1) - 0.5) < Math.abs((sv[jm] - s0) / (s1 - s0 || 1) - 0.5)) jm = j;
          const ja = Math.max(0, jm - 2), jb = Math.min(s.S - 1, jm + 2), ka = (base + ja) * 2, kb = (base + jb) * 2, km = (base + jm) * 2;
          const dx = sc[kb] - sc[ka], dy = sc[kb + 1] - sc[ka + 1], len = Math.hypot(dx, dy);
          if (len > 0.5 && inside(sc[km], sc[km + 1])) {
            const ux = dx / len, uy = dy / len, hd = diff ? 7 : 6, x = sc[km] + ux * hd / 2, y = sc[km + 1] + uy * hd / 2;
            g.beginPath(); g.moveTo(x, y); g.lineTo(x - hd * ux + hd * 0.55 * uy, y - hd * uy - hd * 0.55 * ux); g.lineTo(x - hd * ux - hd * 0.55 * uy, y - hd * uy + hd * 0.55 * ux); g.closePath();
            g.lineWidth = 2; g.strokeStyle = "rgba(255,255,255,.9)"; g.stroke(); g.fillStyle = rgba(pathCol(0.5), 1); g.fill();
          }
          for (const [j, shape] of [[0, 0], [s.S - 1, 1]]) {
            const k = (base + j) * 2; if (!inside(sc[k], sc[k + 1])) continue;
            mark(g, shape, sc[k], sc[k + 1], hot ? 4.2 : 3.4); g.lineWidth = 3; g.strokeStyle = "rgba(255,255,255,.9)"; g.stroke();
            g.fillStyle = j ? rgba(P.fg, 1) : "#fff"; g.fill(); g.lineWidth = 1.4; g.strokeStyle = rgba(P.fg, 1); g.stroke();
          }
        }
        g.globalAlpha = 1;
      }
    }
    // overlays: points, or arrows from the paired set (control sets grey and dashed)
    const ovScreen = [], keepPairs = this.pairFilter(p, iso);
    for (const st of p.sets) {
      if (!st.xy || (st.cfg.draw === "arrows" && st.from && !st.fromXY)) continue;
      const n = st.s.n, xy = this.frameXY(st.xy, n, L, b, new Float32Array(n * 2)), m = st.s.meta || {};
      const sc = new Float32Array(n * 2); for (let j = 0; j < n; j++) { sc[j * 2] = X(xy[j * 2]); sc[j * 2 + 1] = Y(xy[j * 2 + 1]); }
      const rec = { st, sc }; ovScreen.push(rec);
      const control = st.cfg.style === "control", main = control ? P.mute : P.accent;
      // the class filter applies to a set only when its true labels are in the probe's label space; a set sharing pair
      // ids with the main overlay (meta.pair) follows the main overlay's surviving pairs instead of its own labels
      const byPair = keepPairs && st !== keepPairs.main && m.pair;
      const filt = !!byPair || (iso != null && sameSpace(st.s));
      const isoOK = j => !filt || (byPair ? keepPairs.has(m.pair[j]) : m.label?.[j] === iso);
      const own = scal && scal.target === st.s.id;
      const colOf = j => { if (!own) return main; const v = scal.perSite ? scal.vals[lc][j] : scal.vals[j]; return Number.isFinite(v) ? seqColour(P, (v - scal.range[0]) / (scal.range[1] - scal.range[0])) : P.grey; };
      if (st.cfg.draw === "arrows" && st.fromXY) {
        const fx = this.frameXY(st.fromXY, n, L, b, new Float32Array(n * 2));
        // with one class isolated, all of that class's arrows are drawn (the density subset applies to the whole set)
        const sub = this.arrowKeep(n), keep = rec.keep = filt ? isoOK : sub, fsc = rec.fsc = new Float32Array(n * 2);
        const shown = filt ? 30 : Math.min(n, this.density);
        const al = Math.max(0.4, Math.min(0.9, 0.9 * Math.sqrt(60 / shown)));            // fewer arrows, stronger ink
        const pr = predOf(m), byPred = !control && this.arrowCol === "pred" && pr;
        g.lineWidth = small ? 0.8 : control ? 0.9 : 1; g.setLineDash(control ? [3, 2.5] : []);
        const head = small ? 2.5 : 4.5;
        for (let j = 0; j < n; j++) {
          const x0 = fsc[j * 2] = X(fx[j * 2]), y0 = fsc[j * 2 + 1] = Y(fx[j * 2 + 1]), x1 = sc[j * 2], y1 = sc[j * 2 + 1];
          if (!keep(j)) continue;
          const in1 = inside(x1, y1);
          if (!in1 || !inside(x0, y0)) off.arrows++;
          const col = byPred ? (P.classes[pr[j]] || P.grey) : main, a = control ? al * 0.85 : al;
          g.strokeStyle = rgba(col, a); g.fillStyle = rgba(col, Math.min(1, a + 0.1));
          const dx = x1 - x0, dy = y1 - y0, len = Math.hypot(dx, dy);
          if (len < 1.5) { if (in1) { g.beginPath(); g.arc(x1, y1, 1.2, 0, 7); g.fill(); } continue; }
          g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
          if (!in1) continue;
          const hl2 = Math.min(head, 0.45 * len), ux = dx / len, uy = dy / len;
          g.save(); g.setLineDash([]); g.beginPath(); g.moveTo(x1, y1); g.lineTo(x1 - hl2 * ux + hl2 * 0.5 * uy, y1 - hl2 * uy - hl2 * 0.5 * ux); g.lineTo(x1 - hl2 * ux - hl2 * 0.5 * uy, y1 - hl2 * uy + hl2 * 0.5 * ux); g.closePath(); g.fill(); g.restore();
        }
        g.setLineDash([]);
      } else {
        rec.keep = isoOK;
        for (let j = 0; j < n; j++) {
          if (!isoOK(j)) continue;
          if (!inside(sc[j * 2], sc[j * 2 + 1])) { off.points++; continue; }
          g.fillStyle = rgba(colOf(j), own ? 1 : control ? 0.75 : 0.9);
          g.strokeStyle = own ? rgba(P.fg, 0.4) : "rgba(255,255,255,.9)"; g.lineWidth = own ? 0.7 : 0.8;
          g.beginPath(); g.arc(sc[j * 2], sc[j * 2 + 1], small ? r + 0.6 : own ? r + 1.6 : r + 0.9, 0, 7); g.fill(); if (!small) g.stroke();
        }
      }
      // rings around chosen overlay points (highlight.overlay: {set, positions})
      const ho = hl && hl.overlay;
      if (ho && ho.set === st.s.id && !small && !thumb) for (const j of ho.positions || []) if (j >= 0 && j < n && isoOK(j)) ring(sc[j * 2], sc[j * 2 + 1], r + 5);
    }
    g.restore();
    // what was cut at the panel's edge (the limits are the pooled 0.3–99.7% extent, so a few can lie outside)
    const cut = [[off.arrows, "arrow", "arrows"], [off.paths, "path", "paths"], [off.points, "point", "points"]].filter(([k]) => k > 0);
    if (!small && !thumb && cut.length) {
      const t = `${cut.map(([k, one, many]) => `${num(k)} ${k === 1 ? one : many}`).join(", ")} beyond the panel edge (cut, not moved)`;
      g.font = "10.5px Inter, system-ui, sans-serif"; g.textAlign = "left";
      const tw = g.measureText(t).width;
      g.fillStyle = "rgba(255,255,255,.88)"; g.fillRect(2, hh - 17, tw + 8, 15);
      g.fillStyle = rgba(P.mute, 1); g.fillText(t, 6, hh - 6);
    }
    return small ? {} : { screen, ovScreen, pathScreen };
  }

  // ---------------------------------------------------- hover
  bindHover(p) {
    const cv = p.cv;
    const move = e => {
      const r = cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      const hit = this.nearest(p, x, y);
      p.hot = hit ? { x: hit.x, y: hit.y, from: hit.from, path: hit.type === "path" ? { pt: hit.pt, q: hit.q } : null } : null;
      this.drawPanelOnly(p);
      if (hit) this.showTip(e.clientX, e.clientY, p, hit); else hideTip();
    };
    cv.addEventListener("pointermove", move);
    cv.addEventListener("pointerdown", e => { if (e.pointerType !== "mouse") move(e); });
    cv.addEventListener("pointerleave", e => { if (e.pointerType === "mouse") { p.hot = null; this.drawPanelOnly(p); hideTip(); } });
    cv.setAttribute("role", "img");
    cv.setAttribute("aria-label", "Scatter plot of the probe images in the embedding; hover a point for its image");
  }
  drawPanelOnly(p) { if (p.o && !p.err && p.probe) { const tween = this.kind !== "site" && this.playing; this.drawPanel(p, tween ? this.b : Math.round(this.b), Math.round(this.b)); } }
  nearest(p, x, y) {
    let best = null, bd = 12 * 12;
    const test = (sx, sy, what) => { const d = (sx - x) ** 2 + (sy - y) ** 2; if (d < bd) { bd = d; best = { ...what, x: sx, y: sy }; } };
    for (const { st, sc, fsc, keep } of p.ovScreen || []) for (let j = 0; j < st.s.n; j++) if (!keep || keep(j)) test(sc[j * 2], sc[j * 2 + 1], { type: "ov", st, j, from: fsc ? [fsc[j * 2], fsc[j * 2 + 1]] : null });
    for (const { pt, sc, shown } of p.pathScreen || []) for (const q of shown) for (let j = 0; j < pt.s.S; j++) { const k = (q * pt.s.S + j) * 2; test(sc[k], sc[k + 1], { type: "path", pt, q, j }); }
    if (best) return best;
    bd = 10 * 10;
    const sc = p.screen, iso = this.isoNow(), lab = p.o.labels;
    if (sc) for (let i = 0; i < p.o.N; i++) if (iso == null || !lab || lab[i] === iso) test(sc[i * 2], sc[i * 2 + 1], { type: "probe", i });
    return best;
  }
  showTip(cx, cy, p, hit) {
    const lc = Math.round(this.b), site = this.chain()[lc], cls = i => i == null || i < 0 ? "–" : this.classes[i] ?? `class ${i}`;
    const scal = p.scalar, rows = [];
    let head = "";
    const thumb = h("canvas", { width: 32, height: 32, class: "tt-thumb" });
    const draw = async (sprP, idx, cols) => { const im = sprP && await sprP; if (im && cols) thumb.getContext("2d").drawImage(im, (idx % cols) * 32, Math.floor(idx / cols) * 32, 32, 32, 0, 0, 32, 32); else thumb.remove(); };
    if (hit.type === "probe") {
      const i = hit.i, o = p.o;
      head = cls(o.labels ? o.labels[i] : -1);
      rows.push(["probe image", `#${i}${o.op.probe.ids ? ` (test id ${o.op.probe.ids[i]})` : ""}`]);
      if (scal && scal.target === "probe") rows.push([scal.label, fmt(scal.perSite ? scal.vals[lc][i] : scal.vals[i])]);
      draw(p.sprite, i, o.op.probe.sprite_cols);
    } else if (hit.type === "ov") {
      const { st, j } = hit, s = st.s, m = s.meta || {};
      head = st.cfg.label || s.label;
      if (m.label?.[j] != null) rows.push(["true class", trueName(s, m.label[j], this.classes)]);
      const yn = v => v == null ? null : v ? "yes" : "no", pr = predOf(m);
      if (pr?.[j] != null) rows.push(["predicted (network output)", cls(pr[j])]);
      for (const [k, t] of [["clean_correct", "correct on the clean image"], ["misclassified", "misclassified on this input"], ["changed", "prediction changed from clean"]])
        if (m[k]?.[j] != null) rows.push([t, yn(m[k][j])]);
      // meta.success (X3 sets): correct on the clean image (both seeds) and wrong on this perturbed input
      if (m.success?.[j] != null) rows.push([st.cfg.style === "control" ? "clean-correct, wrong after noise" : "attack succeeded (clean-correct, now wrong)", yn(m.success[j])]);
      if (s.images?.ids) rows.push(["image", `${s.images.dataset || ""} #${s.images.ids[j]}`]);
      if (st.from) rows.push(["arrow", `from its ${lower(st.from.s.label || st.from.s.id)} version`]);
      if (scal && scal.target === s.id) rows.push([`${scal.label} at ${site}`, fmt(scal.perSite ? scal.vals[lc][j] : scal.vals[j])]);
      if (m.rowmax) rows.push(Array.isArray(m.rowmax[0]) ? [`largest kernel-row entry at ${site}`, fmt(m.rowmax[lc][j])] : ["largest kernel-row entry (max over sites)", fmt(m.rowmax[j])]);
      draw(st.sprite, j, s.sprite_cols);
    } else {
      const { pt, q, j } = hit, s = pt.s, m = s.meta || {};
      head = `Path ${q + 1} of ${s.P} (${m.kind?.[q] === "diff" ? "different classes" : "same class"})`;
      rows.push(["endpoints (true class)", `${trueName(s, m.label_a?.[q], this.classes)} → ${trueName(s, m.label_b?.[q], this.classes)}`], ["s", fmt(s.s[j])]);
      if (m.s_switch && m.s_switch[q] != null) rows.push(["prediction switches at s", fmt(m.s_switch[q])]);
      thumb.remove();
    }
    rows.push(["site", site]);
    showTip(cx, cy, h("div", { class: "tt-img" }, thumb, h("div", {}, h("div", { class: "tt-head", text: head }), ...rows.map(([k, v]) => h("div", { class: "tt-row" }, h("span", { class: "tt-k", text: k }), h("span", { class: "tt-v", text: v }))))));
  }

  // ---------------------------------------------------- legend and operator line
  describe() {
    const P = this.P, c = this.cur(), live = this.panels.filter(p => p.o && !p.err);
    const items = [];
    const sw = (style, label) => h("span", { class: "ef-k" }, h("i", { style }), label);
    const scal = live.find(p => p.scalar)?.scalar;
    if (!scal || scal.target !== "probe") {
      if (scal) items.push(sw(`background:${rgba(P.grey, 1)}`, "probe images"));
      else {
        // the class key doubles as a filter: hover previews one class, click keeps it (click again for all)
        const cl = h("div", { class: "ef-classes", role: "group", "aria-label": "Show one class only" });
        this.classes.slice(0, 10).forEach((n, i) => {
          const b = h("button", { type: "button", class: `ef-k ef-cls${this.iso === i ? " on" : ""}`, "aria-pressed": String(this.iso === i), title: `Show only ${n} (click again for all)` }, h("i", { style: `background:${rgba(P.classes[i], 1)}` }), n);
          b.addEventListener("click", () => { this.iso = this.iso === i ? null : i; this.isoPreview = null; this.describe(); this.redraw(); });
          b.addEventListener("pointerenter", e => { if (e.pointerType === "mouse") { this.isoPreview = i; this.redraw(); } });
          b.addEventListener("pointerleave", e => { if (e.pointerType === "mouse") { this.isoPreview = null; this.redraw(); } });
          cl.append(b);
        });
        const filt = this.iso != null ? h("span", { class: "ef-filter" }, `Showing ${this.classes[this.iso]} only · `, h("button", { type: "button", class: "ef-linkbtn", text: "show all", onclick: () => { this.iso = null; this.describe(); this.redraw(); } })) : h("span", { class: "ef-filter mute", text: "Point at a class to show it alone; click to keep it." });
        const other = new Map();             // sets the class filter leaves alone: true labels not in the probe's space
        for (const p of live) {
          for (const st of p.sets) if (!sameSpace(st.s) && st.s.meta?.label) other.set(st.cfg.label || st.s.label || st.s.id, spaceOf(st.s));
          for (const pt of p.paths) if (!sameSpace(pt.s) && pt.s.meta?.label_a) other.set("paths", spaceOf(pt.s));
        }
        const note = other.size ? h("span", { class: "ef-filter mute", text: `Not filtered by class: ${[...other].map(([n, sp]) => `${n} (${sp ? `${SPACE_NAME[sp] || sp} labels` : "label space not recorded"})`).join("; ")}.` }) : null;
        items.push(h("div", { class: "ef-kgroup" }, h("span", { class: "ef-ktitle", text: `Probe images (${num(live[0]?.o.N || 0)}), coloured by class` }), cl, filt, note));
      }
    }
    const grad = `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map(t => rgba(seqColour(P, t), 1)).join(",")})`;
    if (scal) items.push(h("div", { class: "ef-kgroup" }, h("span", { class: "ef-ktitle", text: `${scal.target === "probe" ? "Probe images" : this.setName(scal.target)}, coloured by ${scal.label}` }),
      h("span", { class: "ef-ramp" }, h("span", { text: rfmt(scal.range[0], scal.range) }), h("i", { style: `background:${grad}` }), h("span", { text: rfmt(scal.range[1], scal.range) })),
      h("span", { class: "ef-k mute", text: scal.fit
        ? `range fixed: ${ord(scal.fit[0])} to ${ord(scal.fit[1])} percentile of the values at all sites${scal.nOpts > 1 ? ` and every choice of ${lower(scal.title || "colour")}` : ""}; values beyond take the end colours`
        : `colour range fixed over all sites${scal.nOpts > 1 ? " and all colour choices" : ""}` })));
    // counts: what is drawn of what exists (density subset, class isolation)
    const shownTxt = (p, st, arrows) => {
      const n = st.s.n, m = st.s.meta || {}, kp = this.pairFilter(p, this.iso);
      const byPair = kp && st !== kp.main && m.pair, filt = byPair || (this.iso != null && sameSpace(st.s));
      if (filt) { let k = 0; for (let j = 0; j < n; j++) if (byPair ? kp.has(m.pair[j]) : m.label?.[j] === this.iso) k++; return `${num(k)} of ${num(n)} shown: ${this.classes[this.iso]}${byPair ? " pairs" : ""}`; }
      const k = arrows ? Math.min(n, this.density) : n; return k < n ? `${num(k)} of ${num(n)} shown` : `${num(n)} images`;
    };
    const seen = new Set();
    for (const p of live) for (const st of p.sets) {
      if (seen.has(st.s.id + st.cfg.style)) continue; seen.add(st.s.id + st.cfg.style);
      const control = st.cfg.style === "control", col = rgba(control ? P.mute : P.accent, 1), lab = st.cfg.label || st.s.label;
      const byPred = !control && this.arrowCol === "pred" && predOf(st.s.meta || {});
      if (st.cfg.draw === "arrows" && st.fromXY) items.push(h("span", { class: "ef-k" }, h("i", { class: `ef-arrow${control ? " ctl" : ""}${byPred ? " pred" : ""}`, style: `--c:${col}` }),
        h("span", { html: `<b>${esc(lab)}</b>${control ? " (control)" : ""}: arrow from each ${esc(lower(st.from.s.label || st.from.s.id))} image to its ${st.cfg.verb || (control ? "noised" : "attacked")} version${byPred ? ", coloured by the class the network predicts for it" : ""} (${shownTxt(p, st, true)})` })));
      else if (!(scal && scal.target === st.s.id)) items.push(sw(`background:${col};border-radius:50%`, `${lab} (${shownTxt(p, st, false)})`));
      else items.push(h("span", { class: "ef-k mute", text: `${lab}: ${shownTxt(p, st, false)}` }));
    }
    const p0 = live[0];
    if (p0 && p0.paths.length) {
      const s = p0.paths[0].s;
      const pgrad = `linear-gradient(90deg, ${[0, .25, .5, .75, 1].map(t => rgba(seqColour(P, 0.12 + 0.88 * t), 1)).join(",")})`;
      items.push(h("div", { class: "ef-kgroup" }, h("span", { class: "ef-ktitle", text: `Paths x(s) = (1−s)a + sb at ${s.l0} (${num(s.P)}), coloured by s` }),
        h("span", { class: "ef-ramp" }, h("span", { html: "<b class=ef-shape>○</b> a, s = 0" }), h("i", { style: `background:${pgrad}` }), h("span", { html: "s = 1, b <b class=ef-shape>■</b>" })),
        h("span", { class: "ef-k" }, h("i", { class: "ef-line" }), "endpoints of different classes"),
        h("span", { class: "ef-k" }, h("i", { class: "ef-line dash" }), "endpoints of the same class"),
        h("span", { class: "ef-k mute", text: "The arrowhead at s = ½ points from a to b. Point at a path to pick it out." })));
    }
    const hl = c.highlight || live.find(p => p.item.highlight)?.item.highlight, perPanel = live.some(p => p.item.highlight);
    const nRing = (hl?.probe?.length || 0) + (hl?.overlay?.positions?.length || 0);
    const ringTxt = (perPanel && !c.highlight ? (live.find(p => p.item.highlight?.label)?.item.highlight.label || "ringed: chosen probe images, different in each panel") : null) || hl?.label || [hl?.probe?.length && `ringed: ${hl.probe.length} probe images`, hl?.overlay?.positions?.length && `ringed: ${hl.overlay.positions.length} images of ${this.setName(hl.overlay.set)}`].filter(Boolean).join("; ");
    // highlight.total: how many images meet the ringed criterion in all (the drawn sample may hold fewer)
    if (nRing) items.push(h("span", { class: "ef-k" }, h("i", { class: "ef-ring" }), ringTxt + (Number.isFinite(hl?.total) ? ` (${num(nRing)} shown of ${num(hl.total)} flagged)` : "")));
    items.push(h("span", { class: "ef-k mute ef-hint", text: matchMedia("(hover: none)").matches ? "Tap a point to see its image." : "Hover a point to see its image." }));
    this.legend.replaceChildren(...(live.length ? items : []));
    this.el.classList.toggle("ef-nolegend", !live.length);
    this.opline.replaceChildren(...this.operatorLine());
  }
  setName(id) { for (const p of this.panels) for (const st of p.sets || []) if (st.s.id === id) return st.cfg.label || st.s.label; return id; }
  operatorLine() {
    const c = this.cur(), live = this.panels.filter(p => p.o);
    if (!live.length) return [];
    const K = live[0].o.K, o0 = live[0].o, pr = o0.op.probe || {};
    const netTxt = o => { const n = o.op.net || {}; return `<b>${esc(n.label || n.id || o.id)}</b>${n.step != null ? ` (step ${num(n.step)})` : ""}`; };
    const kerTxt = o => { const k = o.op.kernel || {}; return `<b>${esc(k.label || `${k.rule} ${k.param}`)}</b> kernel`; };
    const chainTxt = o => { const ch = o.op.chain; return `${ch.length} sites (${esc(ch[0])} … ${esc(ch[ch.length - 1])})`; };
    const probe = pr.label || (pr.viewer_order ? `the viewer's ${num(pr.n)} probe images` : `probe ${esc(pr.id)} (${num(pr.n)} images)`);
    const kindTxt = { joint: `<b>joint</b> (one set of ${K} through-depth modes for every site)`, source: `<b>source</b> (each site's local kernel average p<sub>l</sub>·f<sub>l</sub> of the ${K} joint modes, before depth filtering; divided by 2c<sub>l</sub>, it is the right-hand side of the depth solve)`, site: `<b>site-local</b> (each site's own ${K} walk modes; axes differ between sites)` }[this.kind] || esc(this.kind);
    const ops = [...new Map(live.map(p => [p.o.id, p.o])).values()];
    const same = f => new Set(ops.map(f)).size === 1;
    let who;
    if (ops.length === 1) who = `of ${netTxt(o0)}, ${kerTxt(o0)}, ${probe}, ${chainTxt(o0)}`;
    else {
      const shared = [same(o => o.op.kernel?.label) && kerTxt(o0), same(o => o.op.chain.join()) && chainTxt(o0)].filter(Boolean);
      const each = live.map(p => [!same(o => o.op.net?.label) && netTxt(p.o), !same(o => o.op.kernel?.label) && kerTxt(p.o), !same(o => o.op.chain.join()) && chainTxt(p.o)].filter(Boolean).join(", "));
      who = `of a separate operator per panel (${each.join("; ")})${shared.length ? `, each with ${shared.join(" over ")}` : ""}; ${probe}`;
    }
    const multi = live.length > 1, ax = c.axes || "pooled";
    const axes = this.kind === "site" ? "each site's own walk modes 1 and 2 (signs chosen to agree with the previous site)"
      : ax === "viewer" ? "the viewer's pooled principal axes (same signs and scaling as the viewer)"
      : `${live.every(p => (ax.startsWith("pooled:") ? this.panels[+ax.split(":")[1]]?.o : p.o)?.op.axes?.pooled) ? "the two principal axes of the site-normalised joint probe coordinates, pooled over sites" : `the pooled principal axes of the probe's ${KIND_NAME[this.kind] || this.kind} coordinates`}${multi && ops.length > 1 ? (ax.startsWith("pooled:") ? " of the first panel's operator" : " of each panel's operator") : ""}`;
    const scaled = (c.norm || "site") === "site" ? `; each site divided by the probe's Frobenius norm in ${KIND_NAME[this.kind] || this.kind} coordinates` : "";
    // panels share axes only when they draw one operator, or are turned onto the first panel's axes by a fitted R
    const sameAxes = ops.length === 1 || (this.kind !== "site" && ax.startsWith("pooled:") && live.slice(1).every(p => p.item.R));
    const shared = multi ? (sameAxes ? "; all panels share axes and limits" : "; the panels share axis limits, not axes") : "";
    const rot = live.some(p => p.item.R) && this.kind !== "site" ? ` The other panels are turned onto the first by a fitted ${K} × ${K} orthogonal map (a post-hoc fit).` : "";
    // spec.note (the entry's own statement of which metric its numbers use), else only that the picture is 2-D
    const moves = this.panels.some(p => p.sets?.some(s => s.cfg.draw === "arrows") || p.paths?.length);
    const extra = c.note ? ` ${esc(c.note)}` : moves ? " Lengths in the picture are 2-D; see the text for the metric." : "";
    const display = ` Limits: the pooled 0.3–99.7% extent over all sites, padded; anything beyond is cut at the edge and counted.${this.kind !== "site" ? " While playing, frames between sites are interpolated for display, not measured." : ""}`;
    return [h("span", { class: "ef-op-tag", text: "Operator" }), h("span", { html: `Coordinates: ${kindTxt} ${who}. Shown: 2-D projection on ${axes}${scaled}${shared}.${rot}${extra}` }), h("span", { class: "ef-op-display", text: display })];
  }
}

// ------------------------------------------------------------------ tooltip (the page's one #tooltip element)
function showTip(x, y, content) {
  const t = document.getElementById("tooltip"); if (!t) return;
  t.replaceChildren(content); t.hidden = false;
  const r = t.getBoundingClientRect(), W = innerWidth, H = innerHeight;
  let lx = x + 14, ly = y + 14;
  if (lx + r.width > W - 8) lx = x - r.width - 14;
  if (ly + r.height > H - 8) ly = y - r.height - 14;
  t.style.left = Math.max(8, lx) + "px"; t.style.top = Math.max(8, ly) + "px";
}
function hideTip() { const t = document.getElementById("tooltip"); if (t) t.hidden = true; }

// ------------------------------------------------------------------ entry point used by main.js
export async function mount(fig) {
  if (fig.__emb) return fig.__emb;
  const src = fig.dataset.emb, [url, key] = src.split("#");
  let spec;
  const holder = { ready: null };
  fig.__emb = holder;
  try {
    const obj = await getJSON(rel(url, document.baseURI));
    spec = getPath(obj, key);
    if (!spec) throw new DataError(`${src}: no figure spec at “${key}”`);
  } catch (e) {
    console.error("emb-fig:", e);
    const cap = fig.querySelector(":scope > figcaption");
    fig.insertBefore(h("div", { class: "ef ef-failed" }, h("div", { class: "ef-error" }, h("b", { text: "Figure not drawn. " }), esc(e.message))), cap || null);
    return holder;
  }
  const f = new EmbFigure(fig, spec);
  fig.__emb = f;
  await f.ready;
  return f;
}
