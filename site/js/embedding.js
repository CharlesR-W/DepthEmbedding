// "Depth embedding" tab (embedding_tab.html): the probe images' rows in the low space of each network's through-depth
// operator (SPEC_SPACETIME.md), site by site, for one network or several side by side.
// Data: build_site.py spacetime() (d/st, ResNet-20 seed 0) and stnets() (d/stn/<net>, the other networks), same format:
//   <root>/<step>_<variant>.bin.gz  i8 unit slices (C, K, L, N) scaled to max |u| = 127, then u4 set labels
//   <root>/<step>.json.gz           per variant: t, vals, weight (C, K, L) = share of each mode's squared norm per site, ...
// Rows: G_l[i, k] = sqrt(weight[k][l]) * u_kl(i) / |u_kl|, optionally divided by |G_l|_F ("per site").
// Across networks, nothing is shared by construction; two post-hoc fits put them side by side:
//   depth matching: fractional (site l of L at l / (L - 1)) or DTW on the subspace distance 1 - |Q_A^T Q_B|_F^2 / K;
//   coordinates: one orthogonal K x K rotation per network onto the reference (Procrustes over the matched sites).
import * as D from "./data.js";

const $ = id => document.getElementById(id);
const SEBA_R = [2, 3, 4, 5, 6, 8, 10, 12, 14, 16];     // offered r (at most 16 sets: 8 palette colours + 8 darker)
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim() || ({ "--surface": "#fff", "--fg": "#16161a", "--mute": "#6b6b70", "--line": "#e4e3de", "--accent": "#c2410c" })[v];
function hexRgb(h) { h = h.replace("#", ""); return [0, 2, 4].map(o => parseInt(h.slice(o, o + 2), 16)); }
const rgba = ([r, g, b], a) => `rgba(${r | 0},${g | 0},${b | 0},${a})`;
function hsl2rgb(h, s, l) {
  const f = n => { const k = (n + h * 12) % 12, a = s * Math.min(l, 1 - l); return l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1)); };
  return [255 * f(0), 255 * f(8), 255 * f(4)];
}
const fmt = x => x == null || !isFinite(x) ? "–" : Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 1 ? x.toPrecision(3) : x.toPrecision(2);
const fmtFactor = f => f >= 1 ? String(f) : `1/${Math.round(1 / f)}`;

// one marker shape per network in the overlay (path only; the caller fills or strokes)
export function mark(g, k, x, y, r) {
  g.beginPath();
  switch (k % 7) {
    case 0: g.arc(x, y, r, 0, 7); break;
    case 1: g.rect(x - r * .9, y - r * .9, r * 1.8, r * 1.8); break;
    case 2: g.moveTo(x, y - r * 1.15); g.lineTo(x + r, y + r * .7); g.lineTo(x - r, y + r * .7); g.closePath(); break;
    case 3: g.moveTo(x, y - r * 1.2); g.lineTo(x + r * 1.05, y); g.lineTo(x, y + r * 1.2); g.lineTo(x - r * 1.05, y); g.closePath(); break;
    case 4: g.moveTo(x, y + r * 1.15); g.lineTo(x + r, y - r * .7); g.lineTo(x - r, y - r * .7); g.closePath(); break;
    case 5: { const a = r * .38; g.rect(x - r, y - a, 2 * r, 2 * a); g.rect(x - a, y - r, 2 * a, 2 * r); break; }
    default: for (let t = 0; t < 10; t++) { const rr = t % 2 ? r * .5 : r * 1.2, an = -Math.PI / 2 + t * Math.PI / 5; t ? g.lineTo(x + rr * Math.cos(an), y + rr * Math.sin(an)) : g.moveTo(x + rr * Math.cos(an), y + rr * Math.sin(an)); } g.closePath();
  }
}

// ------------------------------------------------------------------ small dense linear algebra
// symmetric eigendecomposition (cyclic Jacobi), descending; V row-major, columns = eigenvectors (sign: largest entry > 0)
export function eigSym(Ain, n) {
  const A = Float64Array.from(Ain), V = new Float64Array(n * n);
  for (let i = 0; i < n; i++) V[i * n + i] = 1;
  for (let sweep = 0; sweep < 80; sweep++) {
    let off = 0; for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += A[p * n + q] ** 2;
    if (off < 1e-28) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = A[p * n + q]; if (Math.abs(apq) < 1e-300) continue;
      const th = (A[q * n + q] - A[p * n + p]) / (2 * apq), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let r = 0; r < n; r++) { const x = A[r * n + p], y = A[r * n + q]; A[r * n + p] = c * x - s * y; A[r * n + q] = s * x + c * y; }
      for (let r = 0; r < n; r++) { const x = A[p * n + r], y = A[q * n + r]; A[p * n + r] = c * x - s * y; A[q * n + r] = s * x + c * y; }
      for (let r = 0; r < n; r++) { const x = V[r * n + p], y = V[r * n + q]; V[r * n + p] = c * x - s * y; V[r * n + q] = s * x + c * y; }
    }
  }
  const idx = [...Array(n).keys()].sort((a, b) => A[b * n + b] - A[a * n + a]), W = new Float64Array(n * n);
  idx.forEach((j, c) => {
    let big = 0; for (let r = 0; r < n; r++) if (Math.abs(V[r * n + j]) > Math.abs(big)) big = V[r * n + j];
    const sg = big < 0 ? -1 : 1; for (let r = 0; r < n; r++) W[r * n + c] = sg * V[r * n + j];
  });
  return { vals: idx.map(j => A[j * n + j]), V: W };
}
// A^T B for row-major (N x K) arrays -> K x K
export function crossTN(A, B, N, K) {
  const C = new Float64Array(K * K);
  for (let i = 0; i < N; i++) { const o = i * K; for (let a = 0; a < K; a++) { const x = A[o + a]; if (!x) continue; for (let b = 0; b < K; b++) C[a * K + b] += x * B[o + b]; } }
  return C;
}
// orthogonal polar factor of M (K x K): the R minimising |X_B R - X_A| when M = X_B^T X_A. Returns {R, sv}
export function polar(M, K) {
  const MtM = new Float64Array(K * K);
  for (let a = 0; a < K; a++) for (let b = 0; b < K; b++) { let s = 0; for (let r = 0; r < K; r++) s += M[r * K + a] * M[r * K + b]; MtM[a * K + b] = s; }
  const { vals, V } = eigSym(MtM, K), sv = vals.map(v => Math.sqrt(Math.max(v, 0)));
  const tol = 1e-10 * (sv[0] || 1), R = new Float64Array(K * K);
  // R = U V^T with U = M V S^{-1} on the nonzero singular values; columns of U for ~0 singular values are completed
  // to an orthonormal basis (Gram-Schmidt on the standard basis), so R is always orthogonal (rank deficiency: sv)
  const U = new Float64Array(K * K), ok = sv.map(x => x > tol);
  for (let r = 0; r < K; r++) for (let c = 0; c < K; c++) { if (!ok[c]) continue; let s = 0; for (let k = 0; k < K; k++) s += M[r * K + k] * V[k * K + c]; U[r * K + c] = s / sv[c]; }
  let e = 0;
  for (let c = 0; c < K; c++) {
    if (ok[c]) continue;
    for (; e < K; e++) {
      const u = new Float64Array(K); u[e] = 1;
      for (let c2 = 0; c2 < K; c2++) { if (c2 === c || (!ok[c2] && c2 > c)) continue; let d = 0; for (let r = 0; r < K; r++) d += U[r * K + c2] * u[r]; for (let r = 0; r < K; r++) u[r] -= d * U[r * K + c2]; }
      const nr = Math.hypot(...u); if (nr > 1e-6) { for (let r = 0; r < K; r++) U[r * K + c] = u[r] / nr; e++; break; }
    }
  }
  for (let r = 0; r < K; r++) for (let c = 0; c < K; c++) { let s = 0; for (let k = 0; k < K; k++) s += U[r * K + k] * V[c * K + k]; R[r * K + c] = s; }
  return { R, sv };
}
// orthonormal basis of the column space of X (N x K): Q = X V S^-1 over the nonzero part; returns {Q, r}
function colBasis(X, N, K) {
  const { vals, V } = eigSym(crossTN(X, X, N, K), K), tol = 1e-10 * (vals[0] || 1), keep = vals.map(v => v > tol);
  const r = keep.filter(Boolean).length, Q = new Float64Array(N * r);
  let c2 = 0;
  for (let c = 0; c < K; c++) {
    if (!keep[c]) continue;
    const s = 1 / Math.sqrt(vals[c]);
    for (let i = 0; i < N; i++) { let z = 0; for (let k = 0; k < K; k++) z += X[i * K + k] * V[k * K + c]; Q[i * r + c2] = z * s; }
    c2++;
  }
  return { Q, r };
}
// DTW with both ends fixed; cost (La x Lb) row-major. Returns for each a, the mean matched b
export function dtwMap(cost, La, Lb) {
  const Dm = new Float64Array(La * Lb).fill(Infinity);
  for (let a = 0; a < La; a++) for (let b = 0; b < Lb; b++) {
    const c = cost[a * Lb + b];
    if (!a && !b) { Dm[0] = c; continue; }
    const best = Math.min(a ? Dm[(a - 1) * Lb + b] : Infinity, b ? Dm[a * Lb + b - 1] : Infinity, a && b ? Dm[(a - 1) * Lb + b - 1] : Infinity);
    Dm[a * Lb + b] = c + best;
  }
  const sum = new Float64Array(La), cnt = new Float64Array(La);
  let a = La - 1, b = Lb - 1;
  for (;;) {
    sum[a] += b; cnt[a]++;
    if (!a && !b) break;
    const opts = [];
    if (a && b) opts.push([Dm[(a - 1) * Lb + b - 1], a - 1, b - 1]);
    if (a) opts.push([Dm[(a - 1) * Lb + b], a - 1, b]);
    if (b) opts.push([Dm[a * Lb + b - 1], a, b - 1]);
    opts.sort((x, y) => x[0] - y[0]); [, a, b] = opts[0];
  }
  const out = Array.from(sum, (s, i) => s / cnt[i]);
  out[0] = 0; out[La - 1] = Lb - 1;     // both ends fixed: the first and last sites match (averaging would move them)
  return out;
}

// SEBA (Froyland, Rock & Sakellariou 2019, Algorithm 3.1; as explore/cifar_coherent/seba.py) on an orthonormal
// V (p x r, row-major). Returns {S (p x r, reliability order, clipped at 0, max 1), iters} or {error}
export function sebaCore(V, p, r, maxIter = 5000, tol = 1e-12) {
  const mu = 0.99 / Math.sqrt(p), S = new Float64Array(p * r);
  let R = new Float64Array(r * r); for (let a = 0; a < r; a++) R[a * r + a] = 1;
  let it = 0;
  const thresh = () => {
    S.fill(0); const nrm = new Float64Array(r);
    for (let j = 0; j < p; j++) for (let c = 0; c < r; c++) {
      let z = 0; for (let k = 0; k < r; k++) z += V[j * r + k] * R[k * r + c];
      const a = Math.abs(z) - mu; if (a > 0) { const x = Math.sign(z) * a; S[j * r + c] = x; nrm[c] += x * x; }
    }
    if (nrm.some(x => x === 0)) return false;
    for (let j = 0; j < p; j++) for (let c = 0; c < r; c++) S[j * r + c] /= Math.sqrt(nrm[c]);
    return true;
  };
  let converged = false;
  for (; it < maxIter; it++) {          // as seba.py: keep the rotation that produced the last thresholded S
    if (!thresh()) return { error: `the threshold emptied a vector at r = ${r}` };
    const { R: Rn } = polar(crossTN(V, S, p, r), r);
    let dn = 0; for (let k = 0; k < r * r; k++) dn += (Rn[k] - R[k]) ** 2;
    if (Math.sqrt(dn) <= tol) { converged = true; break; }
    if (it !== maxIter - 1) R = Rn;
  }
  it = Math.min(it, maxIter - 1);
  // sign by sum, max scaling, reliability order (decreasing minimum), clip at 0
  const mins = new Float64Array(r);
  for (let c = 0; c < r; c++) {
    let sum = 0; for (let j = 0; j < p; j++) sum += S[j * r + c];
    const sg = sum < 0 ? -1 : 1; let mx = -Infinity, mn = Infinity;
    for (let j = 0; j < p; j++) { S[j * r + c] *= sg; if (S[j * r + c] > mx) mx = S[j * r + c]; }
    for (let j = 0; j < p; j++) { S[j * r + c] /= mx; if (S[j * r + c] < mn) mn = S[j * r + c]; }
    mins[c] = mn;
  }
  const perm = [...Array(r).keys()].sort((a, b) => mins[b] - mins[a] || a - b), out = new Float64Array(p * r);
  for (let j = 0; j < p; j++) for (let c = 0; c < r; c++) out[j * r + c] = Math.max(0, S[j * r + perm[c]]);
  return { S: out, iters: it + 1, converged };
}

// SEBA runs in a Web Worker (built from the functions above) so a slow solve never blocks the page
let sebaWorkerURL = null;
function runSebaWorker(V, p, r) {
  sebaWorkerURL ??= URL.createObjectURL(new Blob([[eigSym, crossTN, polar, sebaCore].map(f => f.toString()).join("\n") +
    "\nonmessage = e => { const { V, p, r } = e.data; postMessage(sebaCore(V, p, r)); };"], { type: "text/javascript" }));
  return new Promise((res, rej) => {
    const w = new Worker(sebaWorkerURL);
    w.onmessage = e => { w.terminate(); res(e.data); };
    w.onerror = e => { w.terminate(); rej(new Error(e.message || "SEBA worker failed")); };
    w.postMessage({ V, p, r }, [V.buffer]);
  });
}

// ------------------------------------------------------------------ data
async function gunzip(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer();
}
const cache = new Map();
function once(key, fn) { if (!cache.has(key)) { const p = fn().catch(e => { cache.delete(key); throw e; }); cache.set(key, p); } return cache.get(key); }
function loadUnit(m, net, step, variant) {
  return once(`u:${net.root}/${step}_${variant}`, async () => {
    const V = net.variants.find(v => v.id === variant), C = V.C, K = net.K, L = net.sites.length, N = m.n, half = (N + 1) >> 1;
    const nsets = net.nsets ?? 4, nv = C * K * L * N, nl = nsets * C * L;
    const b = await gunzip(`${net.root}/${step}_${variant}.bin.gz`);
    if (b.byteLength !== nv + nl * half) throw new Error(`${net.root}/${step}_${variant}: ${b.byteLength} bytes, expected ${nv + nl * half}`);
    const u8 = new Uint8Array(b, nv, nl * half), lab = new Int8Array(nl * N);
    for (let p = 0; p < nl; p++) for (let i = 0; i < N; i++) { const byte = u8[p * half + (i >> 1)]; lab[p * N + i] = ((i & 1) ? byte >> 4 : byte & 15) - 1; }
    return { C, K, L, vecs: new Int8Array(b, 0, nv), lab };
  });
}
// per-site baseline coordinates (build_site.py stnets()): {L, k, q: Int8Array (L, k, N), scale: (L, k), rec}
function loadBase(m, net, step, kind) {
  return once(`b:${net.base.root}/${step}_${kind}`, async () => {
    const rec = JSON.parse(new TextDecoder().decode(await gunzip(`${net.base.root}/${step}_base.json.gz`)));
    const b = await gunzip(`${net.base.root}/${step}_${kind}.bin.gz`), sc = rec[`scale_${kind}`], L = sc.length, k = sc[0].length;
    if (b.byteLength !== L * k * m.n) throw new Error(`${net.base.root}/${step}_${kind}: ${b.byteLength} bytes, expected ${L * k * m.n}`);
    return { L, k, q: new Int8Array(b), scale: sc, rec };
  });
}
const loadSummary = (net, step) => once(`s:${net.root}/${step}`, async () => JSON.parse(new TextDecoder().decode(await gunzip(`${net.root}/${step}.json.gz`))));

// the networks this build ships: ResNet-20 seed 0 from meta.spacetime, the rest from meta.stnets.nets
export function netList(m) {
  const S = m.spacetime, out = [];
  if (S) out.push({ id: "r20s0", label: "ResNet-20 seed 0", short: "R20 s0", root: "d/st", sites: m.sites, steps: S.steps, variants: S.variants, K: S.K, fidx: S.factors.map((_, i) => i),
                    acc: m.val_acc?.[m.val_acc.length - 1] ?? null, sets: S.sets });
  for (const n of m.stnets?.nets || []) out.push({ ...n, sets: n.nsets === 0 ? [] : n.sets ?? S?.sets });
  for (const n of out) n.base = m.stnets?.base?.[n.id] || null;
  return out;
}

// ------------------------------------------------------------------ the view
export class DepthEmbedding {
  constructor(root) {
    this.root = root;
    this.st = { method: "modes", nets: ["r20s0"], ref: "r20s0", step: null, variant: "own", c: 3, K: 8, norm: "site", dmode: "frac", align: true,
                basis: "pooled", ax: [0, 1, 2], three: false, color: "class", r: 6, layout: "grid", ocol: "net", show: "all", links: true, anchor: 0, trails: true, fix: true, depth: 0 };
    this.sel = null; this.nSel = 0; this.hidden = new Set(); this.hiddenNets = new Set(); this.rot = { yaw: 0.6, pitch: 0.35 }; this.playing = false;
    this.panels = []; this.lastHash = null;
    this.ready = this.init();
  }

  async init() {
    this.m = await D.meta();
    this.nets = netList(this.m);
    if (!this.nets.length) { $("emb-status").textContent = "This build has no through-depth data."; return; }
    this.N = this.m.n; this.sel = new Uint8Array(this.N);
    this.byId = Object.fromEntries(this.nets.map(n => [n.id, n]));
    this.st.step = this.byId.r20s0 ? this.byId.r20s0.steps.at(-1) : this.nets[0].steps.at(-1);
    this.st.ref = this.nets[0].id; this.st.nets = [this.nets[0].id];
    [this.labels] = await Promise.all([D.labels()]);
    D.sprite().then(im => { this.sprite = im; });
    this.pal = this.m.palette;
    this.buildControls();
    this.bind();
  }

  // ------------------------------------------------ controls
  buildControls() {
    const accTxt = n => n.acc != null ? ` (${(100 * n.acc).toFixed(1)}%)` : "";
    const box = $("emb-nets"), ref = $("emb-ref"), x1 = document.createElement("details"), group = document.createElement("optgroup");
    const title = "Trained on other data (X1) · final checkpoint only";
    const list = document.createElement("div"); list.className = "emb-nets";
    const summary = document.createElement("summary"); summary.textContent = title; summary.className = "emb-hint";
    x1.append(summary, list); group.label = title;
    box.replaceChildren(); ref.replaceChildren();
    for (const n of this.nets) {
      const isX1 = n.group === "x1";
      const l = document.createElement("label"); l.className = "emb-check"; l.dataset.net = n.id;
      const c = document.createElement("input"); c.type = "checkbox"; c.value = n.id; c.dataset.net = n.id;
      const acc = Object.assign(document.createElement("span"), { className: "emb-hint", textContent: accTxt(n) });
      if (isX1) { l.title = n.training; acc.title = "Test accuracy on its own task"; }
      l.append(c, ` ${n.label}`, acc);
      (isX1 ? list : box).append(l);
      const opt = new Option(`${n.label}${accTxt(n)}`, n.id);
      if (isX1) opt.title = `${n.training} Test accuracy on its own task in brackets.`;
      (isX1 ? group : ref).append(opt);
    }
    if (list.childElementCount) { box.append(x1); ref.append(group); }
    ref.title = list.childElementCount ? "Original networks: final validation accuracy. X1: test accuracy on its own task." : "final validation accuracy in brackets";
  }
  ref() { return this.byId[this.st.ref]; }
  stepFor(n) {
    const { step } = this.st;
    return n.steps.includes(step) ? step : step === this.ref().steps.at(-1) ? n.steps.at(-1) : null;
  }
  noSetPlanes() {
    return this.st.method === "modes" && this.shown().some(n => n.nsets === 0 && this.stepFor(n) !== null && n.variants.some(v => v.id === this.st.variant));
  }
  shown() { const s = this.st.nets.filter(id => this.byId[id]); if (!s.includes(this.st.ref)) s.unshift(this.st.ref); return s.map(id => this.byId[id]); }
  // the options under "More options", and whether any differs from its default (then the section opens and says so)
  advancedChanged() {
    const st = this.st, ref = this.ref();
    return st.method === "modes" && (st.variant !== ref.variants[0].id || st.c !== 3) || (st.method !== "umap" && st.K !== 8) || st.norm !== "site" ||
      st.basis !== "pooled" || st.ax.some((a, d) => a !== d) || st.three || !st.fix;
  }
  syncControls() {
    const st = this.st, ref = this.ref(), multi = this.shown().length > 1;
    document.querySelectorAll("#emb-nets label").forEach(l => { l.hidden = l.dataset.net === st.ref; });
    document.querySelectorAll("#emb-nets input").forEach(c => { c.checked = st.nets.includes(c.value) && c.value !== st.ref; });
    $("emb-ref").value = st.ref;
    $("emb-step").replaceChildren(...ref.steps.map(s => new Option(s.toLocaleString("en"), s))); $("emb-step").value = st.step;
    $("emb-variant").replaceChildren(...ref.variants.map(v => new Option(v.label + (v.id === ref.variants[0].id ? " (default)" : ""), v.id))); $("emb-variant").value = st.variant;
    const V = ref.variants.find(v => v.id === st.variant);
    const nf = this.m.spacetime.factors.length;
    $("emb-coup").max = nf - 1; $("emb-coup").value = st.c; $("emb-coup").disabled = V.C === 1;
    $("emb-K").value = st.K; $("emb-norm").value = st.norm; $("emb-dmode").value = st.dmode; $("emb-align").checked = st.align;
    $("emb-basis").value = st.basis; $("emb-three").checked = st.three; $("emb-color").value = st.color;
    for (const id of ["emb-anchor", "emb-anchor2"]) { $(id).replaceChildren(...ref.sites.map((s, l) => new Option(s, l))); $(id).value = st.anchor; }
    $("emb-trails").checked = st.trails; $("emb-fix").checked = st.fix;
    $("emb-depth").max = ref.sites.length - 1; $("emb-depth").value = st.depth;
    $("emb-cmp").hidden = !multi;
    for (let d = 0; d < 3; d++) {
      const s = $("emb-ax" + d);
      s.replaceChildren(...Array.from({ length: st.K }, (_, k) => new Option(st.basis === "modes" ? `mode ${k + 1}` : `PC${k + 1}`, k)));
      if (st.method === "umap") { s.value = Math.min(d, 1); s.disabled = true; continue; }
      s.disabled = false; st.ax[d] = Math.min(st.ax[d], st.K - 1); s.value = st.ax[d];
    }
    $("emb-ax2").disabled = !st.three || st.method === "umap";
    document.body.classList.toggle("emb-rot", st.three);
    $("emb-reset3d").disabled = !st.three; $("emb-3dhint").hidden = !st.three;
    $("emb-mousehint").textContent = st.three ? "Plot: hover a dot to see its image. Keys: ← → step a site (when the plot is not focused), Space plays."
      : "Plot: hover a dot to see its image; drag a box to select images and follow them through depth. Keys: ← → step a site, Space plays.";
    this.root.querySelectorAll("input[name=emb-method]").forEach(r => {
      r.checked = r.value === st.method; r.disabled = ref.group === "x1" && r.value !== "modes";
    });
    const base = st.method !== "modes", noSets = this.noSetPlanes();
    $("emb-variant").disabled = base; $("emb-coup").disabled = base || V.C === 1; $("emb-K").disabled = st.method === "umap"; $("emb-three").disabled = st.method === "umap";
    [...$("emb-color").options].forEach(o => { o.disabled = o.value === "seba" ? base : noSets || base ? o.value.startsWith("set") : false; });
    $("emb-r-row").hidden = st.color !== "seba"; $("emb-r").value = st.r;
    $("emb-anchor-row").hidden = st.color !== "pos"; $("emb-anchor2-row").hidden = st.basis !== "anchor";      // UMAP's 2-d coordinates are rotated by this basis too
    $("emb-color-hint").textContent = ({
      class: "The class labels colour the dots only; they are not used to build anything.",
      pos: "Hue = direction and depth of colour = distance from the centre in the x/y axes, at the chosen site. Watch which colours stay together.",
      seba: "A post-hoc readout: SEBA run once on the stacked modes. Each network runs its own; colours are matched to the reference by overlap.",
    })[st.color] || (st.color.startsWith("set") ? "A post-hoc readout: k-means sets of the rank-8 readout, computed per network and per site." : "");
    if (base) $("emb-color-hint").textContent += " (The set colourings need the through-depth modes.)";
    if (noSets) $("emb-color-hint").textContent += " K-means colourings are unavailable here: they were not computed for networks trained on other data. SEBA uses the modes and is still available.";
    this.root.querySelectorAll("input[name=emb-layout]").forEach(r => { r.checked = r.value === st.layout; });
    $("emb-ocol").value = st.ocol; $("emb-show").value = st.show; $("emb-links").checked = st.links;
    $("emb-ov-opts").hidden = st.layout !== "overlay";
    const adv = this.advancedChanged();
    $("emb-more-note").textContent = adv ? "(some changed from the defaults)" : "";
    if (adv && !this.moreOpened) { $("emb-more").open = true; this.moreOpened = true; }
  }
  bind() {
    const st = this.st, on = (id, ev, f) => $(id).addEventListener(ev, f);
    // while a load is pending, controls only change the state: the load recomputes with the latest state when it lands
    const re = (f, reload = true) => async () => { f(); this.clamp(); this.syncControls(); this.saveHash(); if (reload) await this.load(); else if (!this.loading && this.data) { this.recompute(); await this.refreshSeba(); } };
    $("emb-nets").addEventListener("change", re(() => { st.nets = [st.ref, ...[...document.querySelectorAll("#emb-nets input:checked")].map(c => c.value).filter(id => id !== st.ref)]; }));
    on("emb-one", "click", re(() => { st.nets = [st.ref]; }));
    on("emb-all", "click", re(() => { st.nets = this.nets.map(n => n.id); }));
    // the main network: alone, it is replaced; when comparing, the new one becomes the reference and the rest stay
    on("emb-ref", "change", re(() => { const single = this.shown().length < 2; st.ref = $("emb-ref").value; if (single) st.nets = [st.ref]; else if (!st.nets.includes(st.ref)) st.nets.push(st.ref); st.anchor = 0; st.depth = 0; }));
    on("emb-step", "change", re(() => { st.step = +$("emb-step").value; }));
    this.root.querySelectorAll("input[name=emb-method]").forEach(r => r.addEventListener("change", re(() => {
      st.method = r.value; if (st.method !== "modes" && (st.color.startsWith("set") || st.color === "seba")) st.color = "class";
    })));
    on("emb-variant", "change", re(() => { st.variant = $("emb-variant").value; }));
    on("emb-coup", "input", re(() => { st.c = +$("emb-coup").value; }));
    on("emb-K", "change", re(() => { st.K = +$("emb-K").value; }, false));
    on("emb-norm", "change", re(() => { st.norm = $("emb-norm").value; }, false));
    on("emb-dmode", "change", re(() => { st.dmode = $("emb-dmode").value; }, false));
    const redraw = f => () => { f(); this.syncControls(); this.saveHash(); if (this.refD) { this.renderPanels(); this.legend(); this.drawAll(); } };
    this.root.querySelectorAll("input[name=emb-layout]").forEach(r => r.addEventListener("change", redraw(() => { st.layout = r.value; })));
    on("emb-ocol", "change", redraw(() => { st.ocol = $("emb-ocol").value; }));
    on("emb-show", "change", redraw(() => { st.show = $("emb-show").value; }));
    on("emb-links", "change", redraw(() => { st.links = $("emb-links").checked; }));
    on("emb-align", "change", re(() => { st.align = $("emb-align").checked; }, false));
    on("emb-basis", "change", re(() => { st.basis = $("emb-basis").value; }, false));
    for (let d = 0; d < 3; d++) on("emb-ax" + d, "change", re(() => { st.ax[d] = +$("emb-ax" + d).value; }, false));
    on("emb-three", "change", re(() => { st.three = $("emb-three").checked; }, false));
    on("emb-reset3d", "click", () => this.resetRotation());
    on("emb-anchor", "change", re(() => { st.anchor = +$("emb-anchor").value; }, false));
    on("emb-anchor2", "change", re(() => { st.anchor = +$("emb-anchor2").value; }, false));
    on("emb-color", "change", async () => { st.color = $("emb-color").value; this.hidden.clear(); this.syncControls(); this.saveHash(); if (this.loading || !this.refD) return; this.buildColours(); this.drawAll(); await this.refreshSeba(); });
    on("emb-r", "change", async () => { st.r = +$("emb-r").value; this.hidden.clear(); this.saveHash(); if (this.loading || !this.refD) return; this.buildColours(); this.drawAll(); await this.refreshSeba(); });
    on("emb-trails", "change", () => { st.trails = $("emb-trails").checked; this.draw(); });
    on("emb-fix", "change", () => { st.fix = $("emb-fix").checked; this.syncControls(); this.drawAll(); });
    on("emb-depth", "input", () => { this.stop(); const prev = Math.round(st.depth); st.depth = +$("emb-depth").value; this.draw(); if (Math.round(st.depth) !== prev) this.legend(); });
    on("emb-depth", "change", () => this.saveHash());
    on("emb-gif", "click", () => { if (this.exporting) this.exportCancel = true; else this.exportGif(); });
    // while a GIF is being rendered, nothing that changes the drawing may run: the frames must match the header the
    // export wrote at its start. Form controls are disabled (lockForExport); this catches the rest (legend, strip,
    // plots, keys), in the capture phase so no handler below sees the event
    for (const ev of ["click", "pointerdown", "keydown", "change", "input"]) this.root.addEventListener(ev, e => {
      if (!this.exporting || e.target.closest?.("#emb-gif, #emb-jump, #emb-back")) return;      // Cancel, and moving around the page
      if (ev === "keydown" && !["Enter", " ", "ArrowLeft", "ArrowRight", "ArrowUp", "ArrowDown", "Home", "End", "PageUp", "PageDown", "r", "R", "0"].includes(e.key)) return;
      if (ev === "click" && e.target.closest?.("summary, a")) return;      // opening a section or following a link changes nothing drawn
      e.preventDefault(); e.stopPropagation();
    }, true);
    on("emb-clear", "click", () => { this.sel.fill(0); this.selChanged(); });
    on("emb-rand", "click", () => { this.sel.fill(0); for (let k = 0; k < 60; k++) this.sel[(Math.random() * this.N) | 0] = 1; this.selChanged(); });
    on("emb-play", "click", () => this.togglePlay());
    on("emb-jump", "click", () => { const c = $("emb-controls"); c.scrollIntoView({ block: "start" }); c.focus({ preventScroll: true }); });
    on("emb-back", "click", () => { this.root.querySelector(".emb-stage").scrollIntoView({ block: "start" }); $("emb-play").focus({ preventScroll: true }); });
    // touch only: whether a swipe on a plot scrolls the page (default) or draws a selection box / rotates the 3D view
    this.root.querySelectorAll("input[name=emb-touch]").forEach(r => r.addEventListener("change", () => {
      this.touchSel = r.value === "select" && r.checked; this.root.classList.toggle("touch-sel", this.touchSel);
    }));
    // keyboard: arrows step one site, space plays or pauses (only when focus is not in a form control)
    document.addEventListener("keydown", e => {
      if (document.body.dataset.tab !== "embedding" || !this.refD || this.exporting || e.ctrlKey || e.metaKey || e.altKey) return;
      const t = e.target instanceof Element ? e.target : document.body;
      if (t.closest("select, textarea, [contenteditable], input[type=radio], input[type=text], input[type=number], input[type=range]:not(#emb-depth)")) return;
      if (e.key === " ") { if (t.closest("input, button, summary, a")) return; e.preventDefault(); this.togglePlay(); }
      else if (e.key === "ArrowRight" || e.key === "ArrowLeft") { e.preventDefault(); this.stepSite(e.key === "ArrowRight" ? 1 : -1); }
    });
    window.addEventListener("scroll", () => { $("emb-tip").style.display = "none"; }, { passive: true });
    let rz; window.addEventListener("resize", () => { clearTimeout(rz); rz = setTimeout(() => this.layout(), 120); });
  }
  togglePlay() {
    const st = this.st;
    if (this.playing) return this.stop();
    if (st.depth >= this.ref().sites.length - 1) st.depth = 0;
    this.playing = true; this.lastT = 0; $("emb-play").textContent = "❚❚ Pause"; $("emb-play").setAttribute("aria-label", "Pause"); requestAnimationFrame(t => this.tick(t));
  }
  stepSite(dir) {
    this.stop(); const L = this.ref().sites.length, s = this.st.depth;
    this.st.depth = Math.max(0, Math.min(L - 1, dir > 0 ? Math.floor(s + 1e-6) + 1 : Math.ceil(s - 1e-6) - 1));
    $("emb-depth").value = this.st.depth; this.legend(); this.draw(); this.saveHash();
  }
  clamp() {
    const st = this.st, ref = this.ref();
    if (ref.group === "x1") st.method = "modes";
    if (!ref.steps.includes(st.step)) st.step = ref.steps.reduce((a, b) => Math.abs(b - st.step) < Math.abs(a - st.step) ? b : a);
    if (!ref.variants.some(v => v.id === st.variant)) st.variant = ref.variants[0].id;
    const base = st.method !== "modes";
    if (st.color === "seba" ? base : st.color.startsWith("set") && (base || this.noSetPlanes())) {
      st.color = "class"; this.hidden.clear();
    }
    st.c = Math.max(0, Math.min(st.c, this.m.spacetime.factors.length - 1));
    if (ref.fidx && !ref.fidx.includes(st.c)) st.c = ref.fidx.reduce((b, x) => Math.abs(x - st.c) < Math.abs(b - st.c) ? x : b, ref.fidx[0]);
    st.anchor = Math.max(0, st.anchor | 0); st.depth = Math.max(0, +st.depth || 0);
    st.ax = st.ax.map(a => Number.isInteger(a) && a >= 0 ? a : 0);
    if (!["pooled", "anchor", "modes"].includes(st.basis)) st.basis = "pooled";
    if (!["site", "raw"].includes(st.norm)) st.norm = "site";
    if (!["frac", "dtw"].includes(st.dmode)) st.dmode = "frac";
    st.anchor = Math.min(st.anchor, ref.sites.length - 1); st.depth = Math.min(st.depth, ref.sites.length - 1);
    st.K = Math.min(st.K, ...this.shown().map(n => st.method === "modes" ? n.K : n.base?.K ?? n.K));
    if (st.method === "umap") { st.K = 2; st.three = false; } else if (st.K < 4) st.K = 8;
  }

  // ------------------------------------------------ hash state (#embedding?...)
  saveHash() {
    const st = this.st, h = new URLSearchParams({ m: st.method, r: st.r, lay: st.layout, oc: st.ocol, sh: st.show, lk: +st.links, nets: this.shown().map(n => n.id).join("."), ref: st.ref, step: st.step, v: st.variant, c: st.c, K: st.K,
      norm: st.norm, dm: st.dmode, al: +st.align, basis: st.basis, ax: st.ax.join("."), d3: +st.three, color: st.color, anchor: st.anchor, depth: (+st.depth).toFixed(2) });
    this.lastHash = "#embedding?" + h.toString();
    if (document.body.dataset.tab === "embedding") history.replaceState(history.state, "", this.lastHash);
  }
  async applyParams(h) {
    await this.ready; if (!this.nets?.length) return;
    // a link opened mid-export waits for the export to end (finish, cancel or error). Then only the newest waiting
    // request proceeds, and it applies the hash that is current at that moment, not the one it was called with:
    // Back/Forward during the export may have moved the address bar since (and if it now shows the state already on
    // screen, lastHash, there is nothing to apply)
    if (this.exporting) {
      const tok = this.queuedTok = (this.queuedTok || 0) + 1;
      await this.exportDone;
      if (tok !== this.queuedTok || document.body.dataset.tab !== "embedding" || location.hash === this.lastHash) return;
      const raw = decodeURIComponent(location.hash.slice(1)), qi = raw.indexOf("?");
      if (raw.slice(0, qi < 0 ? raw.length : qi) !== "embedding" || qi < 0) return;
      h = new URLSearchParams(raw.slice(qi + 1));
    }
    const st = this.st, num = (k, d) => h.has(k) && isFinite(+h.get(k)) ? +h.get(k) : d;
    if (["modes", "pca", "umap"].includes(h.get("m"))) st.method = h.get("m");
    if (SEBA_R.includes(+h.get("r"))) st.r = +h.get("r");
    if (["grid", "overlay"].includes(h.get("lay"))) st.layout = h.get("lay");
    if (["net", "plot"].includes(h.get("oc"))) st.ocol = h.get("oc");
    if (["all", "sample", "centroids"].includes(h.get("sh"))) st.show = h.get("sh");
    if (h.has("lk")) st.links = h.get("lk") === "1";
    if (h.get("ref") && this.byId[h.get("ref")]) st.ref = h.get("ref");
    if (h.get("nets")) st.nets = h.get("nets").split(".").filter(id => this.byId[id]);
    st.step = num("step", st.step); st.variant = h.get("v") || st.variant; st.c = num("c", st.c); st.K = num("K", st.K);
    st.norm = h.get("norm") || st.norm; st.dmode = h.get("dm") || st.dmode; st.align = h.has("al") ? h.get("al") === "1" : st.align;
    st.basis = h.get("basis") || st.basis; st.three = h.get("d3") === "1";
    if (h.get("ax")) st.ax = h.get("ax").split(".").map(Number).slice(0, 3);
    st.color = h.get("color") || st.color; st.anchor = num("anchor", st.anchor); st.depth = num("depth", st.depth);
    this.clamp(); this.syncControls(); await this.load(); this.lastHash = location.hash;
  }
  async layout() {      // called by the router when the tab is shown (canvases have no size while hidden)
    await this.ready; if (!this.nets?.length) return;
    if (!this.data) { this.clamp(); this.syncControls(); await this.load(); this.saveHash(); return; }
    this.drawAll();
  }

  // ------------------------------------------------ loading and geometry
  async load() {
    const st = this.st, nets = this.shown(), token = this.loadToken = (this.loadToken || 0) + 1;
    this.loading = true;
    $("emb-status").textContent = `loading step ${st.step.toLocaleString("en")}…`;
    const got = await Promise.all(nets.map(async n => {
      const step = this.stepFor(n);
      if (step === null) return { net: n, missing: true, reason: n.group === "x1" ? "only its final checkpoint is available; choose the reference's final step" : null };
      if (st.method !== "modes") {
        if (!n.base || !n.base.steps.includes(step)) return { net: n, missing: true, reason: n.group === "x1" ? "no PCA or UMAP baseline is shipped" : null };
        try { return { net: n, step, base: await loadBase(this.m, n, step, st.method) }; }
        catch (e) { console.error(e); return { net: n, missing: true, error: e.message }; }
      }
      if (!n.variants.some(v => v.id === st.variant)) return { net: n, missing: true, reason: n.group === "x1" ? "only the relative site units variant is shipped" : null };
      try { const [unit, summ] = await Promise.all([loadUnit(this.m, n, step, st.variant), loadSummary(n, step)]); return { net: n, step, unit, summ }; }
      catch (e) { console.error(e); return { net: n, missing: true, error: e.message }; }
    }));
    if (token !== this.loadToken) return;      // superseded by a newer request
    this.loading = false;
    this.data = got;
    this.recompute();
    await this.refreshSeba();
  }
  // index into the couplings shipped for d (other networks ship t/t* = 1 only): nearest to the chosen one
  cIdx(d) {
    if (d.unit.C === 1) return 0;
    const f = d.net.fidx || [...Array(d.unit.C).keys()], i = f.indexOf(this.st.c);
    return i >= 0 ? i : f.reduce((b, x, j) => Math.abs(x - this.st.c) < Math.abs(f[b] - this.st.c) ? j : b, 0);
  }
  buildG(d) {
    if (d.base) return this.buildBaseG(d);
    const { st, N } = this, K = st.K, KK = d.unit.K, L = d.unit.L, c = this.cIdx(d), w = d.summ.variants[st.variant].weight[c], v = d.unit.vecs, G = [];
    for (let l = 0; l < L; l++) {
      const g = new Float64Array(N * K);
      for (let k = 0; k < K; k++) {
        const off = ((c * KK + k) * L + l) * N;
        let nrm = 0; for (let i = 0; i < N; i++) nrm += v[off + i] * v[off + i];
        const s = Math.sqrt(Math.max(w[k][l], 0)) / (Math.sqrt(nrm) || 1);
        for (let i = 0; i < N; i++) g[i * K + k] = v[off + i] * s;
      }
      if (st.norm === "site") { let f = 0; for (let j = 0; j < g.length; j++) f += g[j] * g[j]; f = Math.sqrt(f) || 1; for (let j = 0; j < g.length; j++) g[j] /= f; }
      G.push(g);
    }
    return G;
  }
  // baselines: per-site coordinates (centred), per-site scale as chosen, then chained along depth by Procrustes
  // (site l rotated onto the already rotated site l - 1): the usual way to show an independent embedding per site
  buildBaseG(d) {
    const { st, N } = this, B = d.base, K = st.method === "umap" ? 2 : Math.min(st.K, B.k), G = [];
    for (let l = 0; l < B.L; l++) {
      const g = new Float64Array(N * K);
      for (let k = 0; k < K; k++) { const off = (l * B.k + k) * N, sc = B.scale[l][k]; let mu = 0; for (let i = 0; i < N; i++) mu += B.q[off + i]; mu /= N; for (let i = 0; i < N; i++) g[i * K + k] = (B.q[off + i] - mu) * sc; }
      if (st.norm === "site") { let f = 0; for (let j = 0; j < g.length; j++) f += g[j] * g[j]; f = Math.sqrt(f) || 1; for (let j = 0; j < g.length; j++) g[j] /= f; }
      if (l) {
        const { R } = polar(crossTN(g, G[l - 1], N, K), K), y = new Float64Array(N * K);
        for (let i = 0; i < N; i++) for (let c = 0; c < K; c++) { let s = 0; for (let k = 0; k < K; k++) s += g[i * K + k] * R[k * K + c]; y[i * K + c] = s; }
        G.push(y);
      } else G.push(g);
    }
    return G;
  }
  // rows of network d at fractional site b (linear interpolation between neighbouring sites)
  rowsAt(G, b) {
    const L = G.length, l0 = Math.max(0, Math.min(L - 2, Math.floor(b))), a = Math.max(0, Math.min(1, b - l0));
    if (L === 1) return G[0];
    const out = new Float64Array(G[0].length), g0 = G[l0], g1 = G[l0 + 1];
    for (let j = 0; j < out.length; j++) out[j] = (1 - a) * g0[j] + a * g1[j];
    return out;
  }
  recompute() {
    const { st, N } = this, K = st.K, ok = this.data.filter(d => !d.missing);
    const refD = ok.find(d => d.net.id === st.ref);
    if (!refD) { $("emb-status").textContent = `The reference network has no data at step ${st.step} for this variant.`; this.panels = []; this.renderPanels(); return; }
    for (const d of ok) d.G = this.buildG(d);
    const La = refD.G.length;
    // depth matching: map[a] = fractional site of d matched to reference site a
    for (const d of ok) {
      const Lb = d.G.length;
      if (d === refD) d.map = Array.from({ length: La }, (_, a) => a);
      else if (st.dmode === "dtw") {
        refD.Q ??= refD.G.map(g => colBasis(g, N, K)); d.Q = d.G.map(g => colBasis(g, N, K));
        const cost = new Float64Array(La * Lb);
        for (let a = 0; a < La; a++) for (let b = 0; b < Lb; b++) {
          const A = refD.Q[a], B = d.Q[b];
          if (!A.r || !B.r) { cost[a * Lb + b] = 1; continue; }
          const C = new Float64Array(A.r * B.r);
          for (let i = 0; i < N; i++) for (let p = 0; p < A.r; p++) { const x = A.Q[i * A.r + p]; for (let q = 0; q < B.r; q++) C[p * B.r + q] += x * B.Q[i * B.r + q]; }
          let s = 0; for (const x of C) s += x * x;
          cost[a * Lb + b] = 1 - s / Math.max(A.r, B.r);
        }
        d.cost = cost; d.map = dtwMap(cost, La, Lb);
      } else d.map = Array.from({ length: La }, (_, a) => La === 1 ? 0 : a * (Lb - 1) / (La - 1));
    }
    delete refD.Q;
    // coordinates: Procrustes rotation onto the reference over the matched sites (identity for the reference)
    for (const d of ok) {
      let R = null; d.svMin = null;
      if (d !== refD && st.align) {
        const M = new Float64Array(K * K);
        for (let a = 0; a < La; a++) { const C = crossTN(this.rowsAt(d.G, d.map[a]), refD.G[a], N, K); for (let j = 0; j < M.length; j++) M[j] += C[j]; }
        const p = polar(M, K); R = p.R; d.svMin = p.sv[K - 1] / (p.sv[0] || 1);
      }
      d.R = R;
      d.Y = R ? d.G.map(g => { const y = new Float64Array(N * K); for (let i = 0; i < N; i++) for (let c = 0; c < K; c++) { let s = 0; for (let k = 0; k < K; k++) s += g[i * K + k] * R[k * K + c]; y[i * K + c] = s; } return y; }) : d.G;
    }
    // disagreement with the reference per reference site, and overall
    for (const d of ok) {
      if (d === refD) continue;
      let num = 0, den = 0; d.dis = [];
      for (let a = 0; a < La; a++) {
        const y = this.rowsAt(d.Y, d.map[a]), x = refD.Y[a]; let n1 = 0, d1 = 0;
        for (let j = 0; j < x.length; j++) { n1 += (y[j] - x[j]) ** 2; d1 += x[j] * x[j]; }
        d.dis.push(d1 > 0 ? n1 / d1 : null); num += n1; den += d1;
      }
      d.resid = den > 0 ? num / den : null;
    }
    // axes basis from the reference
    let V; this.evals = null;
    // raw scale: the modes' columns are orthonormal over all sites, so the pooled spectrum is flat (1, ..., 1) and its
    // "principal" axes would be set by rounding; use the raw modes instead and say so
    this.flatPooled = st.method === "modes" && st.norm === "raw" && st.basis === "pooled";
    if (st.basis === "modes" || this.flatPooled) { V = new Float64Array(K * K); for (let i = 0; i < K; i++) V[i * K + i] = 1; }
    else {
      const R = new Float64Array(K * K), sites = st.basis === "pooled" ? [...Array(La).keys()] : [st.anchor];
      for (const l of sites) { const C = crossTN(refD.Y[l], refD.Y[l], N, K); for (let j = 0; j < R.length; j++) R[j] += C[j]; }
      const e = eigSym(R, K); V = e.V; this.evals = e.vals;
    }
    const ax = st.method === "umap" ? [0, 1, 1] : st.ax.map(a => Math.max(0, Math.min(a, K - 1))), dims = st.three ? 3 : 2;
    const q995 = arr => { const s = Float32Array.from(arr).sort(); return s[Math.floor(0.995 * (s.length - 1))] || 1e-9; };
    for (const d of ok) {
      d.P = d.Y.map(y => { const p = new Float32Array(N * 3); for (let i = 0; i < N; i++) for (let t = 0; t < 3; t++) { let z = 0; for (let k = 0; k < K; k++) z += y[i * K + k] * V[k * K + ax[t]]; p[i * 3 + t] = z; } return p; });
      d.rowNorm = d.Y.map(y => { const r = new Float32Array(N); for (let i = 0; i < N; i++) { let s = 0; for (let k = 0; k < K; k++) s += y[i * K + k] ** 2; r[i] = Math.sqrt(s); } return r; });
      d.scaleBySite = d.P.map(p => { const r = new Float32Array(N); for (let i = 0; i < N; i++) { let s = 0; for (let t = 0; t < dims; t++) s += p[i * 3 + t] ** 2; r[i] = Math.sqrt(s); } return q995(r); });
    }
    this.scaleAll = Math.max(...ok.flatMap(d => d.scaleBySite));
    this.refD = refD;
    this.panels = this.data;
    this.buildColours();
    this.renderPanels();
    this.drawAll();
  }

  // ------------------------------------------------ colours
  setPal(j) { const s = this.pal.sets; if (j < s.length) return hexRgb(s[j]); const [r, g, b] = hexRgb(s[j % s.length]); return [r * .55, g * .55, b * .55]; }
  plane() { if (this.st.method !== "modes") return -1; const mm = this.st.color.match(/^set(5|10)([as])$/); if (!mm) return -1; return (this.ref().sets || []).findIndex(s => s.k === +mm[1] && s.readout === mm[2]); }
  setLabel(d, p, l, i) { return d.unit.lab[((p * d.unit.C + this.cIdx(d)) * d.unit.L + l) * this.N + i]; }
  isSeba() { return this.st.method === "modes" && this.st.color === "seba"; }
  labelled() { return this.isSeba() || this.plane() >= 0; }
  labelOf(d, l, i) { return this.isSeba() ? (this.sebaReady(d) ? d.seba.lab[l * this.N + i] : -1) : this.setLabel(d, this.plane(), l, i); }
  sebaKey(d) { return `${this.st.r}|${this.st.variant}|${this.cIdx(d)}`; }
  sebaReady(d) { return !!(d.seba && d.seba.lab && d.seba.key === this.sebaKey(d)); }
  // compute (in a worker) whatever SEBA results the current state needs, then recolour; superseded runs are dropped
  async refreshSeba() {
    if (!this.isSeba() || !this.data) return;
    const ds = this.data.filter(d => !d.missing && d.unit), todo = ds.filter(d => !this.sebaReady(d));
    if (!todo.length) return;
    const token = this.sebaToken = (this.sebaToken || 0) + 1, data = this.data;
    $("emb-status").textContent = `computing SEBA sets (r = ${this.st.r}) for ${todo.length} network(s)…`;
    const res = await Promise.all(todo.map(d => this.sebaOf(d)));
    if (token !== this.sebaToken || data !== this.data) return;
    todo.forEach((d, j) => { d.seba = res[j]; });
    this.buildColours(); this.drawAll();
  }

  // ------------------------------------------------ SEBA on the stacked modes (one sparse basis over (site, image))
  // V = [constant, mode 1, ..., mode r-1] as vectors of length L*N (each mode's raw slices sqrt(w_kl) u_kl stacked by
  // site), re-orthonormalised (Gram-Schmidt; int8 rounding); then Froyland, Rock & Sakellariou's Algorithm 3.1 exactly
  // as seba.py (threshold 0.99/sqrt(p), polar update, sign by sum, max scaling, reliability order, clip at 0).
  sebaOf(d) {
    const r = this.st.r, key = this.sebaKey(d);
    d.sebaCache ??= {};
    if (d.sebaCache[key]) return d.sebaCache[key];
    const { N } = this, L = d.unit.L, KK = d.unit.K, c = this.cIdx(d), variant = this.st.variant, w = d.summ.variants[variant].weight[c], v = d.unit.vecs, p = L * N;
    if (r - 1 > KK) return (d.sebaCache[key] = Promise.resolve({ key, error: `r - 1 = ${r - 1} exceeds the ${KK} modes shipped` }));
    const V = new Float64Array(p * r);
    // the constant eigenfunction in the coordinates of the shipped slices: flat for the identity-link variants; for the
    // paper's matrix the slices are of g = B^{1/2} f (B = diag(1, 2, ..., 2, 1)), where the constant is B^{1/2} 1
    const cw = Array.from({ length: L }, (_, l) => variant === "paper" && l > 0 && l < L - 1 ? Math.SQRT2 : 1);
    const cn = Math.sqrt(N * cw.reduce((a, x) => a + x * x, 0));
    for (let l = 0; l < L; l++) for (let i = 0; i < N; i++) V[(l * N + i) * r] = cw[l] / cn;
    for (let k = 0; k < r - 1; k++) for (let l = 0; l < L; l++) {
      const off = ((c * KK + k) * L + l) * N; let nrm = 0; for (let i = 0; i < N; i++) nrm += v[off + i] * v[off + i];
      const sc = Math.sqrt(Math.max(w[k][l], 0)) / (Math.sqrt(nrm) || 1);
      for (let i = 0; i < N; i++) V[(l * N + i) * r + k + 1] = v[off + i] * sc;
    }
    for (let a = 0; a < r; a++) {      // modified Gram-Schmidt (int8 rounding of the shipped slices)
      for (let b = 0; b < a; b++) { let s = 0; for (let j = 0; j < p; j++) s += V[j * r + a] * V[j * r + b]; for (let j = 0; j < p; j++) V[j * r + a] -= s * V[j * r + b]; }
      let n = 0; for (let j = 0; j < p; j++) n += V[j * r + a] ** 2; n = Math.sqrt(n) || 1; for (let j = 0; j < p; j++) V[j * r + a] /= n;
    }
    const pr = runSebaWorker(V, p, r).then(res => {
      if (res.error) return { key, error: res.error };
      const S = res.S, lab = new Int8Array(p), count = Array.from({ length: r + 1 }, () => new Int32Array(L));
      for (let l = 0; l < L; l++) for (let i = 0; i < N; i++) {
        const j0 = l * N + i; let best = -1, bv = 0;
        for (let j = 0; j < r; j++) { const x = S[j0 * r + j]; if (x > bv) { bv = x; best = j; } }
        lab[j0] = best; count[best < 0 ? r : best][l]++;
      }
      return { key, lab, count, iters: res.iters, converged: res.converged, r };
    }).catch(e => { delete d.sebaCache[key]; return { key, error: e.message }; });
    return (d.sebaCache[key] = pr);
  }
  // colours across networks: each network's sets matched to the reference's greedily by overlap at matched sites
  matchSeba() {
    const refS = this.sebaReady(this.refD) ? this.refD.seba : null; if (!refS) return;
    refS.colour = Array.from({ length: refS.r }, (_, j) => j);
    for (const d of this.data) {
      if (d.missing || d === this.refD || !this.sebaReady(d)) continue;
      const r = d.seba.r, ov = Array.from({ length: r }, () => new Float64Array(refS.r)), La = this.refD.G.length;
      for (let a = 0; a < La; a++) { const b = Math.round(d.map[a]); for (let i = 0; i < this.N; i++) { const x = d.seba.lab[b * this.N + i], y = refS.lab[a * this.N + i]; if (x >= 0 && y >= 0) ov[x][y]++; } }
      const pairs = []; for (let x = 0; x < r; x++) for (let y = 0; y < refS.r; y++) pairs.push([ov[x][y], x, y]);
      pairs.sort((u, v) => v[0] - u[0]);
      const col = new Array(r).fill(-1), used = new Set();
      for (const [n, x, y] of pairs) { if (n <= 0 || col[x] >= 0 || used.has(y)) continue; col[x] = y; used.add(y); }
      let next = refS.r; for (let x = 0; x < r; x++) if (col[x] < 0) col[x] = next++;
      d.seba.colour = col;
    }
  }
  buildColours() {
    const { st, N } = this, grey = hexRgb(this.pal.unassigned), p = this.plane(), refD = this.refD, seba = this.isSeba();
    if (seba) this.matchSeba();
    let posCols = null;
    if (st.color !== "class" && p < 0 && !seba) {
      const P = refD.P[st.anchor], sc = refD.scaleBySite[st.anchor]; posCols = new Array(N);
      for (let i = 0; i < N; i++) { const x = P[i * 3] / sc, y = P[i * 3 + 1] / sc, r = Math.min(1, Math.hypot(x, y)), h = (Math.atan2(y, x) / (2 * Math.PI) + 1) % 1; posCols[i] = hsl2rgb(h, 0.25 + 0.7 * r, 0.62 - 0.17 * r); }
    }
    for (const d of this.data) {
      if (d.missing) continue;
      d.colCache = [];
      for (let l = 0; l < d.G.length; l++) {
        const cols = new Array(N), grp = new Int16Array(N);
        for (let i = 0; i < N; i++) {
          if (st.color === "class") { grp[i] = this.labels[i]; cols[i] = hexRgb(this.pal.classes[this.labels[i]]); }
          else if (seba) {     // grp = the displayed (matched) colour id, so legend clicks act on what is shown
            const g = this.sebaReady(d) ? d.seba.lab[l * N + i] : -1, cid = g < 0 ? -1 : d.seba.colour ? d.seba.colour[g] : g;
            grp[i] = cid; cols[i] = cid < 0 ? grey : this.setPal(cid);
          }
          else if (p >= 0) { const g = this.setLabel(d, p, l, i); grp[i] = g; cols[i] = g < 0 ? grey : this.setPal(g); }
          else { grp[i] = -2; cols[i] = posCols[i]; }
        }
        d.colCache.push({ cols, grp });
      }
    }
    this.legend();
  }
  // the legend is rebuilt on every change (depth, colouring, toggles); keyboard focus is carried across by item key
  legend() {
    const el = $("emb-legend"), a = document.activeElement, fk = a && el.contains(a) ? a.dataset.key : null;
    this.buildLegend();
    if (fk) (el.querySelector(`[data-key="${CSS.escape(fk)}"]`) || el.querySelector("button"))?.focus({ preventScroll: true });
  }
  buildLegend() {
    const el = $("emb-legend"), hint = $("emb-leghint"); el.innerHTML = ""; hint.textContent = ""; if (!this.refD) return;
    const item = (sw, text, pressedOff, title, key) => {
      const b = document.createElement("button"); b.type = "button"; b.className = "emb-leg"; b.title = title; b.dataset.key = key;
      b.append(sw, text); if (pressedOff) b.classList.add("off"); el.append(b); return b;
    };
    // groups hidden with Alt-click stay hidden in every layout, so say so (and offer to show them) wherever they are in effect
    const showHidden = () => {
      if (!this.hidden.size) return;
      const b = document.createElement("button"); b.type = "button"; b.className = "emb-leg emb-showhidden"; b.dataset.key = "showhidden";
      b.textContent = `${this.hidden.size} colour group${this.hidden.size > 1 ? "s" : ""} hidden · show all`;
      b.onclick = () => { this.hidden.clear(); this.legend(); this.drawAll(); };
      el.append(b);
    };
    const dot = c => { const i = document.createElement("i"); i.style.background = c; return i; };
    const mute = t => Object.assign(document.createElement("span"), { textContent: t, className: "emb-legnote" });
    if (this.overlayOn()) {
      const row = document.createElement("div"); row.className = "emb-netleg"; el.append(row);
      const byNet = this.st.ocol === "net" && this.st.show !== "centroids";
      this.panels.forEach((d, k) => {
        if (d.missing) return;
        const cv = document.createElement("canvas"); cv.width = cv.height = 14;
        const g = cv.getContext("2d");
        g.fillStyle = byNet ? rgba(this.netColour(k), 1) : "#6b6b70"; mark(g, k, 7, 7, 5); g.fill();
        const b = item(cv, ` ${d.net.label}${d === this.refD ? " (reference)" : ""}`, this.hiddenNets.has(d.net.id), "click: hide or show this network", `net:${d.net.id}`);
        b.setAttribute("aria-pressed", String(!this.hiddenNets.has(d.net.id)));
        b.onclick = () => { this.hiddenNets.has(d.net.id) ? this.hiddenNets.delete(d.net.id) : this.hiddenNets.add(d.net.id); this.legend(); this.draw(); };
        row.append(b);
      });
      hint.textContent = "Shapes mark networks; click a network to hide or show it.";
      if (byNet) { showHidden(); return; }
      if (this.st.show === "centroids") {      // centroids are always coloured by class
        for (const [j, n] of this.m.classes.entries()) {
          const s = document.createElement("span"); s.className = "emb-leg static"; s.append(dot(this.pal.classes[j]), n);
          if (this.st.color === "class" && this.hidden.has(j)) s.classList.add("off");
          el.append(s);
        }
        hint.textContent += " Class centroids: colour = CIFAR class (labels used for display only).";
        showHidden(); return;
      }
    }
    const p = this.plane(), refD = this.refD;
    let items = [];
    if (this.st.color === "class") items = this.m.classes.map((n, j) => ({ j, n, c: this.pal.classes[j] }));
    else if (this.isSeba()) {
      const S = this.sebaReady(refD) ? refD.seba : null, l = Math.round(this.st.depth);
      if (!S) { el.append(mute(refD.seba?.error && refD.seba.key === this.sebaKey(refD) ? `SEBA: ${refD.seba.error}` : "computing SEBA sets…")); return; }
      const seen = new Map(); for (let i = 0; i < this.N; i++) { const g = S.lab[l * this.N + i]; seen.set(g, (seen.get(g) || 0) + 1); }
      items = [...seen.keys()].sort((a, b) => a - b).map(j => ({ j, n: j < 0 ? `unassigned (${seen.get(j)})` : `set ${j + 1} (${seen.get(j)})`, c: j < 0 ? this.pal.unassigned : `rgb(${this.setPal(j).join(",")})` }));
      el.append(mute(`SEBA sets through depth, a post-hoc readout (r = ${S.r}, ${S.iters} iterations${S.converged ? "" : ", NOT converged"}), ${refD.net.short || refD.net.label} at ${refD.net.sites[l]}${this.otherLoaded().length ? "; other networks' sets are matched one-to-one to the reference's by overlap, greedily (a set may get its second choice, or a colour of its own)" : ""}:`));
    }
    else if (p >= 0) {
      const l = Math.round(this.st.depth), seen = new Map();
      for (let i = 0; i < this.N; i++) { const g = this.setLabel(refD, p, l, i); seen.set(g, (seen.get(g) || 0) + 1); }
      items = [...seen.keys()].sort((a, b) => a - b).map(j => ({ j, n: j < 0 ? `no label (${seen.get(j)})` : `set ${j + 1} (${seen.get(j)})`, c: j < 0 ? this.pal.unassigned : `rgb(${this.setPal(j).join(",")})` }));
      el.append(mute(`post-hoc k-means sets of the rank-8 readout (fixed, whatever K is plotted), ${refD.net.short || refD.net.label} at ${refD.net.sites[l]}; set numbers are per network and per site:`));
    } else {
      const sw = document.createElement("i"); sw.className = "emb-wheel";
      const s = document.createElement("span"); s.className = "emb-leg static";
      s.append(sw, `colour = direction and distance from the centre in ${refD.net.label} at ${refD.net.sites[this.st.anchor]}, in the x/y axes. Watch which colours stay together, in each network.`);
      el.append(s); showHidden(); return;
    }
    for (const it of items) {
      const b = item(dot(it.c), it.n, this.hidden.has(it.j), "click: select this group (shift adds); alt-click: hide or show", `grp:${it.j}`);
      const toggleHide = () => { this.hidden.has(it.j) ? this.hidden.delete(it.j) : this.hidden.add(it.j); this.legend(); this.drawAll(); };
      // keyboard: a click synthesised from Enter carries no modifier keys, so Alt+Enter is handled here
      b.onkeydown = e => { if (e.altKey && (e.key === "Enter" || e.key === " ")) { e.preventDefault(); toggleHide(); } };
      b.onclick = e => {
        if (e.altKey) { toggleHide(); return; }
        const grp = refD.colCache[Math.round(this.st.depth)].grp;
        if (!e.shiftKey) this.sel.fill(0);
        for (let k = 0; k < this.N; k++) if (grp[k] === it.j) this.sel[k] = 1;
        this.selChanged();
      };
    }
    hint.textContent += (hint.textContent ? " " : "") + "Legend: click an entry to select those images (Shift adds); Alt-click (or Alt+Enter) hides or shows it.";
    showHidden();
  }

  // ------------------------------------------------ panels
  overlayOn() { return this.st.layout === "overlay" && this.panels.filter(d => !d.missing).length > 1; }
  renderPanels() {
    const wrap = $("emb-plots"), n = this.panels.length;
    wrap.innerHTML = ""; wrap.dataset.n = n; this.ov = null; delete wrap.dataset.ov;
    if (this.overlayOn()) {
      const fig = document.createElement("figure"); fig.className = "emb-panel";
      const cap = document.createElement("figcaption"), cv = document.createElement("canvas");
      fig.append(cap, cv); wrap.append(fig); wrap.dataset.n = 1; wrap.dataset.ov = 1;
      this.ov = { overlay: true, fig, cap, cv };
      this.bindPlot(this.ov);
      $("emb-strip-card").hidden = true; $("emb-dis-card").hidden = false; $("emb-cmp").hidden = false;
      return;
    }
    for (const d of this.panels) {
      const fig = document.createElement("figure"); fig.className = "emb-panel";
      const cap = document.createElement("figcaption"), cv = document.createElement("canvas");
      fig.append(cap, cv); wrap.append(fig); d.fig = fig; d.cap = cap; d.cv = cv;
      if (d.missing) { cv.hidden = true; cap.textContent = `${d.net.label}: ${d.error ? "failed to load" : d.reason || "no data at this checkpoint or variant"}; no comparison fit was run`; continue; }
      this.bindPlot(d);
    }
    const single = n === 1;
    $("emb-strip-card").hidden = !single; $("emb-dis-card").hidden = !this.otherLoaded().length;      // no chart when no comparison fit ran
    $("emb-cmp").hidden = single;
    if (single) this.drawStripInit();
  }
  posOf(d) {          // fractional site of d at the current depth
    const s = this.st.depth, La = this.refD.G.length, a0 = Math.max(0, Math.min(La - 2, Math.floor(s))), t = s - a0;
    return La === 1 ? d.map[0] : (1 - t) * d.map[a0] + t * d.map[a0 + 1];
  }
  frame(d) {
    const b = this.posOf(d), P = d.P, L = P.length, l0 = Math.max(0, Math.min(L - 2, Math.floor(b))), a = Math.max(0, Math.min(1, b - l0)), N = this.N;
    const out = new Float32Array(N * 3), p0 = P[l0], p1 = P[Math.min(L - 1, l0 + 1)];
    for (let j = 0; j < out.length; j++) out[j] = (1 - a) * p0[j] + a * p1[j];
    return { pts: out, b, scale: this.st.fix ? this.scaleAll : (1 - a) * d.scaleBySite[l0] + a * d.scaleBySite[Math.min(L - 1, l0 + 1)] };
  }
  projector(w, h, scale) {
    const R = Math.min(w, h) * 0.46 / scale, cx = w / 2, cy = h / 2;
    if (!this.st.three) return (x, y) => [cx + R * x, cy - R * y, 0];
    const { yaw, pitch } = this.rot, cyw = Math.cos(yaw), syw = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
    return (x, y, z) => { const x1 = cyw * x + syw * z, z1 = -syw * x + cyw * z, y2 = cp * y - sp * z1, z2 = sp * y + cp * z1; return [cx + R * x1, cy - R * y2, z2 / scale]; };
  }
  fit(cv) {
    const dpr = window.devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) { cv.width = Math.round(w * dpr); cv.height = Math.round(h * dpr); }
    const g = cv.getContext("2d", { willReadFrequently: cv.id === "emb-occ" }); g.setTransform(dpr, 0, 0, dpr, 0, 0); return { g, w, h };
  }
  axisNames() {
    const st = this.st, tot = this.evals ? this.evals.reduce((x, y) => x + Math.max(y, 0), 0) : 0;
    if (st.method === "umap") return ["UMAP axis 1 (after rotation)", "UMAP axis 2 (after rotation)", ""];
    return st.ax.map(a => st.basis === "modes" || this.flatPooled ? `mode ${a + 1}` : `PC${a + 1}${this.evals ? ` (${(100 * this.evals[a] / tot).toFixed(0)}%)` : ""}`);
  }
  drawAll() { this.draw(); if (this.panels.length === 1) this.drawStrip(); else this.drawDis(); this.drawOcc(); this.drawSeba(); this.status(); }
  // SEBA: each set's share of images at each site of the reference (unassigned dashed grey)
  drawSeba() {
    const card = $("emb-seba-card"), S = this.refD && this.sebaReady(this.refD) ? this.refD.seba : null, on = this.isSeba() && !!S;
    card.hidden = !on; if (!on || document.body.dataset.tab !== "embedding") return;
    const cv = $("emb-seba"), { g, w, h } = this.fit(cv), sites = this.refD.net.sites, L = sites.length, N = this.N;
    const padL = 34, padR = 90, padT = 8, padB = 18, X = l => padL + (w - padL - padR) * l / Math.max(1, L - 1), Y = v => padT + (h - padT - padB) * (1 - v);
    g.fillStyle = css("--surface"); g.fillRect(0, 0, w, h); g.strokeStyle = css("--line"); g.fillStyle = css("--mute"); g.font = "10px system-ui, sans-serif"; g.lineWidth = 1;
    for (const v of [0, 0.5, 1]) { g.beginPath(); g.moveTo(padL, Y(v)); g.lineTo(w - padR, Y(v)); g.stroke(); g.textAlign = "right"; g.fillText(`${v * 100}%`, padL - 4, Y(v) + 3); }
    g.textAlign = "center"; const every = Math.ceil(L / Math.max(1, Math.floor((w - padL - padR) / 34)));
    sites.forEach((s, l) => { if (l % every === 0 || l === L - 1) g.fillText(s, X(l), h - 4); });
    g.textAlign = "left";
    for (let j = 0; j <= S.r; j++) {
      const un = j === S.r, c = un ? hexRgb(this.pal.unassigned) : this.setPal(j);
      g.strokeStyle = rgba(c, 1); g.lineWidth = un ? 1.2 : 1.8; g.setLineDash(un ? [4, 3] : []); g.beginPath();
      for (let l = 0; l < L; l++) { const v = S.count[j][l] / N; l ? g.lineTo(X(l), Y(v)) : g.moveTo(X(l), Y(v)); }
      g.stroke(); g.setLineDash([]);
      g.fillStyle = rgba(c, 1); g.fillText(un ? "unassigned" : `set ${j + 1}`, w - padR + 8, padT + 10 + 12 * j);
    }
    const x = X(this.st.depth); g.strokeStyle = css("--accent"); g.lineWidth = 1.5; g.beginPath(); g.moveTo(x, padT); g.lineTo(x, h - padB); g.stroke();
  }
  draw() {
    if (!this.refD || document.body.dataset.tab !== "embedding") return;
    if (this.ov) this.drawOverlay(); else for (const d of this.panels) if (!d.missing) this.drawPanel(d);
    const ref = this.refD.net, lc = Math.round(this.st.depth), s = this.st.depth;
    const near = Math.abs(s - lc) < 0.02, L = ref.sites.length;
    $("emb-sitehead").textContent = near ? ref.sites[lc] : `${ref.sites[Math.floor(s)]} → ${ref.sites[Math.min(L - 1, Math.floor(s) + 1)]}`;
    $("emb-sitesub").textContent = `site ${near ? lc + 1 : (s + 1).toFixed(1)} of ${L}${this.panels.length > 1 ? ` (${ref.short || ref.label})` : ""}`;
    $("emb-depth").setAttribute("aria-valuetext", `${near ? ref.sites[lc] : $("emb-sitehead").textContent}, site ${lc + 1} of ${L}`);
    if (this.panels.length === 1) [...$("emb-strip").children].forEach((f, l) => f.classList.toggle("on", l === lc));
    this.drawOccCursor(); this.drawDisCursor(); if (this.isSeba()) this.drawSeba();
  }
  drawPanel(d) {
    const { st, N } = this, { g, w, h } = this.fit(d.cv);
    g.fillStyle = css("--surface"); g.fillRect(0, 0, w, h);
    const { pts, b, scale } = this.frame(d), proj = this.projector(w, h, scale);
    // axes
    g.strokeStyle = css("--line"); g.lineWidth = 1; g.fillStyle = css("--mute"); g.font = "11px system-ui, sans-serif";
    const names = this.axisNames(), ends = st.three ? [[1, 0, 0], [0, 1, 0], [0, 0, 1]] : [[1, 0, 0], [0, 1, 0]];
    ends.forEach((e, k) => { const a = proj(-e[0] * scale, -e[1] * scale, -e[2] * scale), z = proj(e[0] * scale, e[1] * scale, e[2] * scale); g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(z[0], z[1]); g.stroke(); if (this.panels.length < 4 || k < 2) { const tw = g.measureText(names[k]).width; g.fillText(names[k], Math.min(z[0] + 4, w - tw - 2), Math.max(12, z[1] - 4)); } });
    const lc = Math.max(0, Math.min(d.P.length - 1, Math.round(b))), { cols, grp } = d.colCache[lc];
    const sc = new Float32Array(N * 2), zz = new Float32Array(N), order = [...Array(N).keys()];
    for (let i = 0; i < N; i++) { const s = proj(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]); sc[i * 2] = s[0]; sc[i * 2 + 1] = s[1]; zz[i] = s[2]; }
    d.screen = sc;
    const sel = this.sel, nSel = this.nSel;
    if (st.three) order.sort((x, y) => zz[x] - zz[y]); else order.sort((x, y) => sel[x] - sel[y] || ((x * 7919) % N) - ((y * 7919) % N));
    if (st.trails && nSel) {
      let shown = 0;
      for (let i = 0; i < N && shown < 400; i++) {
        if (!sel[i] || this.hidden.has(grp[i])) continue; shown++;
        const c = cols[i];
        for (let l = 0; l < d.P.length - 1; l++) {
          const p0 = d.P[l], p1 = d.P[l + 1], a = proj(p0[i * 3], p0[i * 3 + 1], p0[i * 3 + 2]), z = proj(p1[i * 3], p1[i * 3 + 1], p1[i * 3 + 2]);
          g.strokeStyle = rgba(c, l + 1 <= b ? 0.55 : 0.14); g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(z[0], z[1]); g.stroke();
        }
        const s0 = proj(d.P[0][i * 3], d.P[0][i * 3 + 1], d.P[0][i * 3 + 2]); g.strokeStyle = rgba(c, 0.6); g.beginPath(); g.arc(s0[0], s0[1], 2.2, 0, 7); g.stroke();
      }
    }
    const r = Math.max(1.3, Math.min(3.2, w / 330));
    for (const i of order) {
      if (this.hidden.has(grp[i])) continue;
      const a = nSel && !sel[i] ? 0.12 : st.three ? 0.45 + 0.5 * Math.max(0, Math.min(1, (zz[i] + 1) / 2)) : 0.85;
      g.fillStyle = rgba(cols[i], a); g.beginPath(); g.arc(sc[i * 2], sc[i * 2 + 1], nSel && sel[i] ? r + 0.8 : r, 0, 7); g.fill();
      if (nSel && sel[i]) { g.strokeStyle = css("--fg"); g.lineWidth = 0.6; g.stroke(); g.lineWidth = 1; }
    }
    this.drawBox(g, d);
    // caption: network, matched site, accuracy, fit quality
    const L = d.P.length, near = Math.abs(b - Math.round(b)) < 0.02;
    const site = near ? d.net.sites[Math.round(b)] : `${d.net.sites[Math.floor(b)]} → ${d.net.sites[Math.min(L - 1, Math.floor(b) + 1)]}`;
    const extra = d === this.refD ? (this.panels.length > 1 ? " · reference" : "") : d.resid != null ? ` · residual vs reference ${d.resid.toFixed(2)}${st.align ? "" : " (no rotation)"}${d.svMin != null && d.svMin < 0.05 ? " (weak fit)" : ""}` : "";
    const coup = d.unit && st.variant !== "paper" && d.net.fidx ? ` · t/t* = ${fmtFactor(this.m.spacetime.factors[d.net.fidx[this.cIdx(d)]])}` : "";
    const acc = this.accuracy(d.net), step = this.panelStep(d);
    const cap = `<b>${d.net.label}</b> · ${site}${step}<br><span class="emb-hint">${acc}${coup}${extra}</span>`;
    d.cap.title = `${d.net.label} · ${site}${step} · ${acc}${coup}${extra}${d.net.training ? ` · ${d.net.training}` : ""}`;
    if (d.capHtml !== cap) { d.cap.innerHTML = cap; d.capHtml = cap; }
  }
  drawBox(g, d) { if (d.box) { g.strokeStyle = css("--accent"); g.setLineDash([4, 3]); g.strokeRect(Math.min(d.box.x0, d.box.x1), Math.min(d.box.y0, d.box.y1), Math.abs(d.box.x1 - d.box.x0), Math.abs(d.box.y1 - d.box.y0)); g.setLineDash([]); } }
  siteName(d) {
    const b = this.posOf(d), L = d.P.length, near = Math.abs(b - Math.round(b)) < 0.02;
    return near ? d.net.sites[Math.round(b)] : `${d.net.sites[Math.floor(b)]} → ${d.net.sites[Math.min(L - 1, Math.floor(b) + 1)]}`;
  }
  // ---- all networks in one plot: each network at its matched site, in the reference's (rotated) coordinates
  ovMembers() { return this.panels.map((d, k) => ({ d, k })).filter(({ d }) => !d.missing && !this.hiddenNets.has(d.net.id)); }
  drawOverlay() {
    const { st, N } = this, ov = this.ov, { g, w, h } = this.fit(ov.cv), mem = this.ovMembers();
    g.fillStyle = css("--surface"); g.fillRect(0, 0, w, h);
    const frames = mem.map(({ d }) => this.frame(d)), dims = st.three ? 3 : 2, C = this.m.classes.length;
    const cmean = P => { const c = new Float64Array(C * 3), n = new Float64Array(C); for (let i = 0; i < N; i++) { const y = this.labels[i]; n[y]++; for (let t = 0; t < 3; t++) c[y * 3 + t] += P[i * 3 + t]; } for (let y = 0; y < C; y++) for (let t = 0; t < 3; t++) c[y * 3 + t] /= n[y] || 1; return c; };
    const cent = st.show === "centroids" ? mem.map(({ d }, j) => ({ path: d.P.map(cmean), cur: cmean(frames[j].pts) })) : null;
    const crad = c => { let mx = 0; for (let y = 0; y < C; y++) { let r2 = 0; for (let t = 0; t < dims; t++) r2 += c[y * 3 + t] ** 2; mx = Math.max(mx, Math.sqrt(r2)); } return mx; };
    // centroids sit near the centre: scale to their own extent (over all sites when the axis limits are fixed)
    const scale = cent ? 1.1 * Math.max(1e-9, ...cent.flatMap(c => st.fix ? c.path.map(crad) : [crad(c.cur)]))
      : st.fix ? this.scaleAll : Math.max(...frames.map(f => f.scale), 1e-9);
    const proj = this.projector(w, h, scale);
    g.strokeStyle = css("--line"); g.lineWidth = 1; g.fillStyle = css("--mute"); g.font = "11px system-ui, sans-serif";
    const names = this.axisNames(), ends = st.three ? [[1, 0, 0], [0, 1, 0], [0, 0, 1]] : [[1, 0, 0], [0, 1, 0]];
    ends.forEach((e, k) => { const a = proj(-e[0] * scale, -e[1] * scale, -e[2] * scale), z = proj(e[0] * scale, e[1] * scale, e[2] * scale); g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(z[0], z[1]); g.stroke(); const tw = g.measureText(names[k]).width; g.fillText(names[k], Math.min(z[0] + 4, w - tw - 2), Math.max(12, z[1] - 4)); });
    const sel = this.sel, nSel = this.nSel, r = Math.max(1.6, Math.min(3.4, w / 300));
    ov.screens = [];
    const colOf = (m, i, lc) => st.ocol === "net" ? this.netColour(m.k) : m.d.colCache[lc].cols[i];
    const lcs = mem.map(({ d }) => Math.max(0, Math.min(d.P.length - 1, Math.round(this.posOf(d)))));
    mem.forEach((m, j) => { const sc = new Float32Array(N * 2), p = frames[j].pts; for (let i = 0; i < N; i++) { const q = proj(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]); sc[i * 2] = q[0]; sc[i * 2 + 1] = q[1]; } ov.screens.push({ d: m.d, k: m.k, sc, lc: lcs[j] }); });
    const visible = (j, i) => !this.hidden.has(mem[j].d.colCache[lcs[j]].grp[i]);
    if (st.show === "centroids") {
      // class means (labels used for display only): colour = class, shape = network; path = the mean through depth
      mem.forEach((m, j) => {
        const { path, cur } = cent[j], b = this.posOf(m.d);
        for (let y = 0; y < C; y++) {
          if (this.st.color === "class" && this.hidden.has(y)) continue;
          const col = hexRgb(this.pal.classes[y]);
          if (st.trails) for (let l = 0; l < path.length - 1; l++) { const a = proj(path[l][y * 3], path[l][y * 3 + 1], path[l][y * 3 + 2]), z = proj(path[l + 1][y * 3], path[l + 1][y * 3 + 1], path[l + 1][y * 3 + 2]); g.strokeStyle = rgba(col, l + 1 <= b ? 0.55 : 0.12); g.lineWidth = 1.2; g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(z[0], z[1]); g.stroke(); }
          const q = proj(cur[y * 3], cur[y * 3 + 1], cur[y * 3 + 2]);
          g.fillStyle = rgba(col, 0.95); g.strokeStyle = css("--fg"); g.lineWidth = 0.8; mark(g, m.k, q[0], q[1], r * 2.2); g.fill(); g.stroke();
        }
      });
    } else {
      const order = [...Array(N).keys()].sort((x, y) => sel[x] - sel[y] || ((x * 7919) % N) - ((y * 7919) % N));
      const keep = st.show === "sample" ? (i => (i * 7919) % N < 300) : (() => true);
      // the same selected image in every network: a grey line through its positions (network order)
      if (st.links && nSel) {
        let shown = 0; g.strokeStyle = rgba(hexRgb(this.pal.unassigned), 0.9); g.lineWidth = 0.8;
        for (let i = 0; i < N && shown < 300; i++) { if (!sel[i]) continue; shown++; g.beginPath(); ov.screens.forEach((s, j) => { j ? g.lineTo(s.sc[i * 2], s.sc[i * 2 + 1]) : g.moveTo(s.sc[i * 2], s.sc[i * 2 + 1]); }); g.stroke(); }
      }
      if (st.trails && nSel) mem.forEach((m, j) => {
        let shown = 0; const d = m.d, b = this.posOf(d);
        for (let i = 0; i < N && shown < 150; i++) {
          if (!sel[i] || !visible(j, i)) continue; shown++; const c = colOf(m, i, lcs[j]);
          for (let l = 0; l < d.P.length - 1; l++) { const p0 = d.P[l], p1 = d.P[l + 1], a = proj(p0[i * 3], p0[i * 3 + 1], p0[i * 3 + 2]), z = proj(p1[i * 3], p1[i * 3 + 1], p1[i * 3 + 2]); g.strokeStyle = rgba(c, l + 1 <= b ? 0.5 : 0.12); g.lineWidth = 1; g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(z[0], z[1]); g.stroke(); }
        }
      });
      for (const i of order) {
        if (!keep(i) && !sel[i]) continue;
        ov.screens.forEach((s, j) => {
          if (!visible(j, i)) return;
          const a = nSel && !sel[i] ? 0.08 : st.show === "sample" || nSel ? 0.9 : 0.55;
          g.fillStyle = rgba(colOf(mem[j], i, lcs[j]), a); mark(g, s.k, s.sc[i * 2], s.sc[i * 2 + 1], nSel && sel[i] ? r + 0.8 : r); g.fill();
        });
      }
    }
    this.drawBox(g, ov);
    const sites = mem.map(({ d }) => `${d.net.short || d.net.label} ${this.siteName(d)}${this.panelStep(d)}`).join(" · ");
    const missing = this.panels.filter(d => d.missing).map(d => `${d.net.short || d.net.label}${d.reason ? ` (${d.reason})` : ""}`);
    ov.cap.title = `${this.describe().overlayTitle}: ${sites}`;
    const D = this.describe();
    const cap = `<b>${D.overlayTitle}</b> <span class="emb-hint">(${D.overlaySub})</span><br><span class="emb-hint">${sites}${missing.length ? ` · no data: ${missing.join(", ")}` : ""}</span>`;
    if (ov.capHtml !== cap) { ov.cap.innerHTML = cap; ov.capHtml = cap; }
  }
  // 3D view angle, from the keyboard (arrow keys on a focused plot) or the Reset button; dragging sets it too
  rotateBy(dyaw, dpitch) {
    this.rot.yaw += dyaw; this.rot.pitch = Math.max(-1.5, Math.min(1.5, this.rot.pitch + dpitch));
    this.draw(); if (this.panels.length === 1) this.drawStrip();
  }
  resetRotation() { this.rot = { yaw: 0.6, pitch: 0.35 }; this.draw(); if (this.panels.length === 1) this.drawStrip(); }
  bindPlot(d) {
    const cv = d.cv;
    // focusable, so the 3D view can be rotated from the keyboard; the keys act only while the plot has focus, and only
    // in 3D (in 2D the arrows keep stepping through depth)
    cv.tabIndex = 0;
    cv.setAttribute("aria-label", `${d.overlay ? "All shown networks in one plot" : d.net.label}: scatter plot of the probe images. In 3D, arrow keys rotate the view and R resets it.`);
    cv.addEventListener("keydown", e => {
      if (!this.st.three || e.ctrlKey || e.metaKey || e.altKey) return;
      const step = e.shiftKey ? 0.35 : 0.12, k = e.key;
      if (k === "ArrowLeft") this.rotateBy(-step, 0); else if (k === "ArrowRight") this.rotateBy(step, 0);
      else if (k === "ArrowUp") this.rotateBy(0, -step); else if (k === "ArrowDown") this.rotateBy(0, step);
      else if (k === "r" || k === "R" || k === "0" || k === "Home") this.resetRotation();
      else return;
      e.preventDefault(); e.stopPropagation();
    });
    cv.addEventListener("pointerdown", e => {
      $("emb-tip").style.display = "none";
      // touch, unless "Swipe on plot: Select / rotate" is on: the page scrolls (CSS touch-action), a tap shows an image
      if (e.pointerType === "touch" && !this.touchSel) { this.drag = { d, x: e.clientX, y: e.clientY, scroll: true, moved: false }; return; }
      const r = cv.getBoundingClientRect(); cv.setPointerCapture(e.pointerId);
      this.drag = { d, x: e.clientX, y: e.clientY, yaw: this.rot.yaw, pitch: this.rot.pitch, shift: e.shiftKey, moved: false };
      if (!this.st.three) d.box = { x0: e.clientX - r.left, y0: e.clientY - r.top, x1: e.clientX - r.left, y1: e.clientY - r.top };
    });
    cv.addEventListener("pointermove", e => {
      const dr = this.drag; if (!dr) return this.hover(d, e);
      if (dr.scroll) { if (Math.abs(e.clientX - dr.x) + Math.abs(e.clientY - dr.y) > 8) dr.moved = true; return; }
      const r = cv.getBoundingClientRect();
      if (Math.abs(e.clientX - dr.x) + Math.abs(e.clientY - dr.y) > 3) dr.moved = true;
      if (this.st.three) { this.rot.yaw = dr.yaw + (e.clientX - dr.x) * 0.01; this.rot.pitch = Math.max(-1.5, Math.min(1.5, dr.pitch + (e.clientY - dr.y) * 0.01)); this.draw(); }
      else { d.box.x1 = e.clientX - r.left; d.box.y1 = e.clientY - r.top; d.overlay ? this.drawOverlay() : this.drawPanel(d); }
    });
    const up = e => {
      const dr = this.drag; if (!dr) return; this.drag = null;
      if (dr.scroll) { if (!dr.moved && e.type === "pointerup") this.hover(d, e); return; }
      // a tap (touch or pen, no drag) shows the nearest image, as hovering does with a mouse
      if (!dr.moved && e.type === "pointerup" && e.pointerType !== "mouse" && !this.st.three) { d.box = null; this.draw(); this.hover(d, e); return; }
      if (!this.st.three && d.box && dr.moved) {
        const b = d.box, [x0, x1] = [Math.min(b.x0, b.x1), Math.max(b.x0, b.x1)], [y0, y1] = [Math.min(b.y0, b.y1), Math.max(b.y0, b.y1)];
        if (!dr.shift) this.sel.fill(0);
        const srcs = d.overlay ? d.screens.map(s => ({ sc: s.sc, grp: s.d.colCache[s.lc].grp })) : [{ sc: d.screen, grp: d.colCache[Math.max(0, Math.min(d.P.length - 1, Math.round(this.posOf(d))))].grp }];
        for (const { sc, grp } of srcs) for (let i = 0; i < this.N; i++) { const x = sc[i * 2], y = sc[i * 2 + 1]; if (x >= x0 && x <= x1 && y >= y0 && y <= y1 && !this.hidden.has(grp[i])) this.sel[i] = 1; }
        d.box = null; this.selChanged(); return;
      }
      d.box = null; if (this.st.three && dr.moved && this.panels.length === 1) this.drawStrip(); this.draw();
    };
    cv.addEventListener("pointerup", up); cv.addEventListener("pointercancel", up);
    cv.addEventListener("pointerleave", e => { if (e.pointerType === "mouse") $("emb-tip").style.display = "none"; });
  }
  hover(d0, e) {
    const tip = $("emb-tip"), r = d0.cv.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
    if (d0.overlay && (!d0.screens || this.st.show === "centroids")) { tip.style.display = "none"; return; }
    const srcs = d0.overlay ? d0.screens.map(s => ({ d: s.d, sc: s.sc, lc: s.lc })) : d0.screen ? [{ d: d0, sc: d0.screen, lc: Math.max(0, Math.min(d0.P.length - 1, Math.round(this.posOf(d0)))) }] : [];
    let best = -1, bd = 64, d = null, lc = 0;
    for (const s of srcs) { const grp = s.d.colCache[s.lc].grp; for (let i = 0; i < this.N; i++) { if (this.hidden.has(grp[i])) continue; const q = (s.sc[i * 2] - x) ** 2 + (s.sc[i * 2 + 1] - y) ** 2; if (q < bd) { bd = q; best = i; d = s.d; lc = s.lc; } } }
    if (best < 0) { tip.style.display = "none"; return; }
    const cv = document.createElement("canvas"); cv.width = cv.height = 32;
    if (this.sprite) { const c = this.m.sprite_cols; cv.getContext("2d").drawImage(this.sprite, (best % c) * 32, Math.floor(best / c) * 32, 32, 32, 0, 0, 32, 32); }
    const txt = document.createElement("div");
    const setTxt = this.labelled() ? (() => { const g = this.labelOf(d, lc, best); return g < 0 ? (this.isSeba() ? "unassigned" : "no label") : `set ${g + 1}`; })() : null;
    txt.innerHTML = `<b>${this.m.classes[this.labels[best]]}</b> · image ${best}<br>${d.net.label}, ${d.net.sites[lc]}: row norm ${fmt(d.rowNorm[lc][best])}` + (setTxt ? `<br>${setTxt}` : "");
    tip.replaceChildren(cv, txt); tip.style.display = "flex";
    const host = $("emb-tip").offsetParent?.getBoundingClientRect() || { left: 0, top: 0, width: innerWidth };
    tip.style.left = Math.max(4, Math.min(host.width - 240, e.clientX - host.left + 14)) + "px"; tip.style.top = (e.clientY - host.top + 14) + "px";
  }
  selChanged() { this.nSel = this.sel.reduce((a, b) => a + b, 0); this.drawAll(); }

  // ------------------------------------------------ strip (one network), occupancy, disagreement chart, status
  drawStripInit() {
    const strip = $("emb-strip"), d = this.panels[0]; strip.innerHTML = ""; if (d.missing) return;
    strip.style.setProperty("--n", d.net.sites.length);
    d.net.sites.forEach((s, l) => {
      const f = document.createElement("figure"), cv = document.createElement("canvas"), cap = document.createElement("figcaption");
      cap.textContent = s; f.append(cv, cap); strip.append(f);
      f.onclick = () => { this.stop(); this.st.depth = l; $("emb-depth").value = l; this.legend(); this.draw(); this.saveHash(); };
    });
  }
  drawStrip() {
    const d = this.panels[0]; if (!d || d.missing || document.body.dataset.tab !== "embedding") return;
    [...$("emb-strip").children].forEach((f, l) => {
      const { g, w, h } = this.fit(f.querySelector("canvas"));
      g.fillStyle = css("--surface"); g.fillRect(0, 0, w, h);
      const proj = this.projector(w, h, this.st.fix ? this.scaleAll : d.scaleBySite[l]), p = d.P[l], { cols, grp } = d.colCache[l];
      for (let pass = 0; pass < 2; pass++) for (let i = 0; i < this.N; i++) {
        if (this.hidden.has(grp[i]) || (pass === 0) === !!(this.nSel && this.sel[i])) continue;
        const s = proj(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]); g.fillStyle = rgba(cols[i], this.nSel && !this.sel[i] ? 0.1 : 0.8); g.fillRect(s[0] - 0.9, s[1] - 0.9, 1.8, 1.8);
      }
    });
  }
  occupancy() {
    const d = this.refD;
    if (d.base) {      // PCA: share of the site's variance in the first K principal coordinates; UMAP: none
      if (this.st.method !== "pca") return null;
      return d.base.rec.evals.map((ev, l) => ev.slice(0, this.st.K).reduce((a, b) => a + b, 0) / (d.base.rec.trace[l] || 1));
    }
    const w = d.summ.variants[this.st.variant].weight[this.cIdx(d)], L = d.unit.L, o = new Float64Array(L);
    for (let k = 0; k < this.st.K; k++) for (let l = 0; l < L; l++) o[l] += w[k][l];
    return o;
  }
  drawOcc() {
    if (!this.refD || document.body.dataset.tab !== "embedding") return;
    const cv = $("emb-occ"), { g, w, h } = this.fit(cv), o = this.occupancy() || new Float64Array(this.refD.net.sites.length), L = o.length, mx = Math.max(...o, 1e-9), pad = 8, mute = css("--mute"), sites = this.refD.net.sites;
    g.fillStyle = css("--surface"); g.fillRect(0, 0, w, h); g.font = "10px system-ui, sans-serif";
    // site labels: first and last always, the rest wherever they fit without touching a neighbour
    const bw = Math.max(3, Math.min(12, (w - 2 * pad) / L - 3)), X = l => pad + (w - 2 * pad) * l / Math.max(1, L - 1);
    const span = l => { const tw = g.measureText(sites[l]).width, x = X(l); return l === 0 ? [Math.max(0, x - bw / 2), Math.max(0, x - bw / 2) + tw] : l === L - 1 ? [Math.min(w, x + bw / 2) - tw, Math.min(w, x + bw / 2)] : [x - tw / 2, x + tw / 2]; };
    const lastL = span(L - 1)[0]; let right = -Infinity;
    for (let l = 0; l < L; l++) {
      const x = X(l), bh = (h - 14) * o[l] / mx;
      g.fillStyle = mute; g.globalAlpha = 0.45; g.fillRect(x - bw / 2, h - 12 - bh, bw, bh); g.globalAlpha = 1;
      const [a0, a1] = span(l);
      if (l === 0 || l === L - 1 || (a0 > right + 6 && a1 < lastL - 6)) { g.textAlign = "left"; g.fillText(sites[l], a0, h - 2); right = a1; }
    }
    g.textAlign = "left";
    const K = this.st.K, net = this.refD.net.short || this.refD.net.label;
    $("emb-occ-cap").textContent = this.st.method === "modes" ? `Bars: how much of the first ${K} modes lies at each site of ${net} (total ${K})`
      : this.st.method === "pca" ? `Bars: share of each site's variance in its first ${K} principal coordinates (${net}; max ${fmt(mx)})`
      : `${net}: UMAP per site (no per-site weights)`;
    this.occBase = g.getImageData(0, 0, cv.width, cv.height); this.drawOccCursor();
  }
  drawOccCursor() {
    if (!this.occBase || !this.refD) return;
    const cv = $("emb-occ"), g = cv.getContext("2d", { willReadFrequently: true }); g.putImageData(this.occBase, 0, 0);
    const w = cv.clientWidth, h = cv.clientHeight, pad = 8, L = this.refD.net.sites.length, x = pad + (w - 2 * pad) * this.st.depth / Math.max(1, L - 1);
    g.strokeStyle = css("--accent"); g.lineWidth = 2; g.beginPath(); g.moveTo(x, 12); g.lineTo(x, h - 12); g.stroke();
  }
  netColour(j) { return hexRgb(this.pal.sets[j % this.pal.sets.length]); }
  drawDis() {
    if (!this.otherLoaded().length || document.body.dataset.tab !== "embedding") return;
    $("emb-dis-what").textContent = this.st.align ? "after the Procrustes rotation (a post-hoc fit)" : "without rotation (each network's own coordinates)";
    const cv = $("emb-dis"), { g, w, h } = this.fit(cv), sites = this.refD.net.sites, L = sites.length;
    const padL = 34, padR = Math.min(150, Math.max(96, w * 0.3)), padT = 8, padB = 18, others = this.panels.filter(d => !d.missing && d !== this.refD);
    const ymax = Math.max(0.5, ...others.flatMap(d => d.dis.filter(v => v != null)));
    g.fillStyle = css("--surface"); g.fillRect(0, 0, w, h);
    const X = l => padL + (w - padL - padR) * l / Math.max(1, L - 1), Y = v => padT + (h - padT - padB) * (1 - v / ymax);
    g.strokeStyle = css("--line"); g.fillStyle = css("--mute"); g.font = "10px system-ui, sans-serif"; g.lineWidth = 1;
    for (const v of [0, ymax / 2, ymax]) { g.beginPath(); g.moveTo(padL, Y(v)); g.lineTo(w - padR, Y(v)); g.stroke(); g.textAlign = "right"; g.fillText(v.toFixed(2), padL - 4, Y(v) + 3); }
    g.textAlign = "center"; const every = Math.ceil(L / Math.max(1, Math.floor((w - padL - padR) / 34)));
    sites.forEach((s, l) => { if (l % every === 0 || l === L - 1) g.fillText(s, X(l), h - 4); });
    g.textAlign = "left";
    others.forEach((d, j) => {
      const c = this.netColour(this.panels.indexOf(d)); g.strokeStyle = rgba(c, 0.95); g.lineWidth = 1.8; g.beginPath();
      d.dis.forEach((v, l) => { if (v == null) return; l ? g.lineTo(X(l), Y(v)) : g.moveTo(X(l), Y(v)); }); g.stroke();
      g.fillStyle = rgba(c, 1); g.fillText(`${d.net.short || d.net.label} (${fmt(d.resid)})`, w - padR + 8, padT + 10 + 14 * j);
    });
    this.disBase = g.getImageData(0, 0, cv.width, cv.height); this.disGeom = { X }; this.drawDisCursor();
  }
  drawDisCursor() {
    if (!this.disBase || !this.otherLoaded().length) return;
    const cv = $("emb-dis"), g = cv.getContext("2d"); g.putImageData(this.disBase, 0, 0);
    const x = this.disGeom.X(this.st.depth); g.strokeStyle = css("--accent"); g.lineWidth = 1.5; g.beginPath(); g.moveTo(x, 6); g.lineTo(x, cv.clientHeight - 16); g.stroke();
  }
  // one plain-text description of the view, used by the key above the plots, the one-plot caption, the comparison
  // help and the GIF header, so that all of them name the same fits (honest labelling: every post-hoc step in use)
  describe() {
    const st = this.st, d = this.refD, ref = d.net.label, refS = d.net.short || ref;
    // claims about the comparison fits rest on the networks that loaded: recompute() fits only those
    const others = this.otherLoaded(), multi = others.length > 0, nm = x => x.net.short || x.net.label;
    const missing = this.panels.filter(x => x.missing).map(nm), why = st.method !== "modes" ? "this checkpoint" : "this checkpoint and variant";
    const dm = st.dmode === "dtw" ? "DTW on subspace distance" : "fractional depth";
    const base = st.method !== "modes", bname = st.method === "pca" ? "PCA" : "UMAP";
    const mixedSteps = others.some(x => x.step !== d.step);
    const stepNote = mixedSteps ? " (reference); other networks at their final checkpoints" : "";
    const checkpoint = `training step ${d.step.toLocaleString("en")}${stepNote}`;
    const kind = base ? `${bname} baseline: each site separately, chained along depth by Procrustes (a fit)` : `Through-depth modes (the construction), first ${st.K}`;
    const noData = missing.length ? `no data at ${why} for ${missing.join(", ")}, so no fit was run for ${missing.length > 1 ? "them" : "it"}` : "";
    const fits = !multi ? null : (st.align
      ? `depths matched by ${dm}; ${others.map(nm).join(", ")} rotated onto ${ref} (Procrustes)`
      : `depths matched by ${dm}. No rotation: each network is shown in its own coordinates, which are not comparable across networks`) + (noData ? `. ${noData[0].toUpperCase()}${noData.slice(1)}` : "");
    const noFit = !multi && missing.length ? `no data at ${why} for ${missing.join(", ")}, so no depth matching or rotation was run` : null;
    const overlayTitle = st.align ? "All networks in one space" : "Overlay of separate coordinate systems";
    const overlaySub = st.align ? `each network rotated onto ${ref} by Procrustes, a post-hoc fit; depths matched by ${dm}, a post-hoc fit`
      : `no rotation: each network in its own coordinates, so positions are not comparable across networks; depths matched by ${dm}, a post-hoc fit`;
    const help = "Each network's operator is built on its own; nothing ties two networks together in the construction. " +
      (!multi && missing.length ? `None of the networks chosen for comparison has data at ${why}, so no depth matching or rotation was run.`
        : st.align ? "To show them together, depths are matched and each network is rotated onto the one chosen under Network (the reference). Both are post-hoc fits."
        : "Depths are matched (a post-hoc fit). Rotation is off, so each network keeps its own coordinates and their positions cannot be compared directly.") +
      (base ? ` The ${bname} baseline is also chained along depth by Procrustes within each network (a fit).` : "") +
      (this.panels.some(x => x.net.group === "x1") ? " X1 compares training pipelines: images, labels, class counts, normalisation and augmentation can differ. The viewer's Procrustes rotation and residual are post-hoc comparisons, not X1's joint agreement statistic." : "");
    const ov = this.overlayOn(), sets = st.color.match(/^set(5|10)([as])$/);
    let colour = ov && st.show === "centroids" ? "class centroids and their paths; colour = CIFAR class (labels used for display only); shape = network"
      : ov && st.ocol === "net" ? "network (one colour and shape each)"
      : st.color === "class" ? "CIFAR class (labels used for colour only)"
      : st.color === "pos" ? `position in ${refS} at ${d.net.sites[st.anchor]}, in the x/y axes (hue = direction, depth of colour = distance)`
      : st.color === "seba" ? `SEBA sets through depth, r = ${st.r} (a post-hoc readout of the modes${multi ? "; other networks' sets matched to the reference's by overlap" : ""})`
      : sets ? `k-means sets, k = ${sets[1]}, ${sets[2] === "a" ? "shape-aligned" : "shared-coordinates"} readout (a post-hoc readout, per network and per site)` : st.color;
    if (ov && st.show !== "centroids" && st.ocol !== "net") colour += "; shape = network";
    if (this.hidden.size) colour += `; ${this.hidden.size} colour group${this.hidden.size > 1 ? "s" : ""} hidden`;
    const hiddenNets = ov ? this.panels.filter(x => !x.missing && this.hiddenNets.has(x.net.id)).map(x => x.net.short || x.net.label) : [];
    if (hiddenNets.length) colour += `; hidden: ${hiddenNets.join(", ")}`;
    return { kind, fits, noFit, checkpoint, mixedSteps, fitsBadge: st.align ? "Post-hoc fits" : "Post-hoc fit", overlayTitle, overlaySub, help, colour };
  }
  panelStep(d) {
    return d.step !== this.st.step || this.panels.some(p => !p.missing && p.net.group === "x1") ? ` · step ${d.step.toLocaleString("en")}` : "";
  }
  accuracy(n) { return n.acc == null ? "" : `${n.group === "x1" ? "Test accuracy on its own task:" : "final acc"} ${(100 * n.acc).toFixed(1)}%`; }
  otherLoaded() { return this.refD ? this.panels.filter(x => !x.missing && x !== this.refD) : []; }
  // what is plotted (prominent, above the plots; names every post-hoc fit in use) and the details line under them
  status() {
    if (!this.refD) return;
    const st = this.st, d = this.refD, multi = this.panels.length > 1, ref = d.net.label;
    const who = multi ? `reference ${ref}` : `${ref}${d.net.acc != null ? ` (${this.accuracy(d.net)})` : ""}`;
    const D = this.describe();
    const fits = D.fits ? `<div class="emb-whatfit"><span class="emb-badge fit">${D.fitsBadge}</span> ${D.fits}</div>`
      : D.noFit ? `<div class="emb-whatfit"><span class="emb-badge">No comparison fit</span> ${D.noFit}</div>` : "";
    $("emb-cmp-help").textContent = D.help;
    $("emb-selinfo").textContent = this.nSel ? `${this.nSel} selected` : "";
    if (d.base) {
      $("emb-tlabel").textContent = "(not used by the baselines)";
      $("emb-what").innerHTML = `<div><span class="emb-badge base">Baseline</span> <b>${st.method === "pca" ? "PCA" : "UMAP"} at each site separately</b>, chained along depth by Procrustes (a fit) · ${who} · ${D.checkpoint}</div>${fits}`;
      $("emb-status").textContent = st.method === "pca" ? `PCA: first ${st.K} principal coordinates of the centred cosine activations, at each site independently.` : "UMAP (15 neighbours, min_dist 0.1) on the cosine distance, at each site independently.";
      if (D.mixedSteps) $("emb-status").textContent += ` · ${D.checkpoint}`;
      return;
    }
    const V = d.net.variants.find(v => v.id === st.variant), rec = d.summ.variants[st.variant], c = this.cIdx(d);
    const S = this.m.spacetime, t = rec.t ? rec.t[c] : null;
    const paper = st.variant === "paper", fi = d.net.fidx ? d.net.fidx[c] : c, coup = paper ? "no coupling parameter" : `t/t* = ${fmtFactor(S.factors[fi])}`;
    const off = this.panels.filter(x => !x.missing && x.unit && x.net.fidx && x.net.fidx[this.cIdx(x)] !== st.c && st.variant !== "paper").map(x => x.net.short || x.net.label);
    $("emb-tlabel").textContent = paper ? "(not used by this variant)" : `t/t* = ${fmtFactor(S.factors[fi])} (${d.net.short || d.net.label}: t = ${fmt(t)})`;
    const vals = rec.vals[c].slice(0, st.K).map(x => fmt(x)).join(", ");
    $("emb-what").innerHTML = `<div><span class="emb-badge">The construction</span> <b>Through-depth modes</b>, first ${st.K} · ${who} · ${D.checkpoint}</div>${fits}`;
    $("emb-status").textContent = `${V.label} · ${coup} · ${d.net.label} eigenvalues 1–${st.K}: ${vals}` +
      (D.mixedSteps ? ` · ${D.checkpoint}` : "") +
      (off.length ? ` · ${off.join(", ")}: only t/t* = 1 is shipped, shown instead` : "") +
      (this.flatPooled ? " · raw scale: the pooled spectrum is flat, so the axes are the raw modes" : "");
  }
  // ---- GIF export: the current view (all panels, or the overlay) swept from the first site to the last, with a
  // header naming the reference site and a legend; encoded in the browser with gifenc (vendor/gifenc, MIT)
  // disable every control except the GIF button (which becomes Cancel), and show that the viewer is locked
  lockForExport(on) {
    this.root.classList.toggle("exporting", on);
    for (const el of this.root.querySelectorAll("input, select, button")) {
      if (el.id === "emb-gif" || el.id === "emb-jump" || el.id === "emb-back") continue;
      if (on) { el.dataset.preLock = el.disabled ? "1" : "0"; el.disabled = true; }
      else if (el.dataset.preLock != null) { el.disabled = el.dataset.preLock === "1"; delete el.dataset.preLock; }
    }
    $("emb-gif").textContent = on ? "Cancel" : "Save GIF";
    $("emb-exportnote").hidden = !on;
  }
  async exportGif() {
    if (this.exporting || !this.refD) return;
    const info = $("emb-gifinfo");
    // the colouring must be final before the header is written: wait out a pending load or SEBA solve
    if (this.loading || (this.isSeba() && this.data.some(d => !d.missing && d.unit && !this.sebaReady(d)))) {
      info.textContent = "Still loading or computing; try again in a moment."; return;
    }
    this.exporting = true; this.exportCancel = false; this.stop();
    let done; this.exportDone = new Promise(r => { done = r; });
    const btn = $("emb-gif"), st = this.st, saved = st.depth;
    this.lockForExport(true);
    const progress = t => { info.textContent = t; $("emb-exportnote").textContent = `Saving GIF: ${t} Controls are locked until it is saved (Cancel, under Save as GIF, stops it).`; };
    try {
      const { GIFEncoder, quantize, applyPalette } = await import("../vendor/gifenc/gifenc.esm.js");
      const W = +$("emb-gifw").value, perSite = +$("emb-gifspeed").value, L = this.refD.net.sites.length;
      const D = this.describe(), multi = this.panels.length > 1;
      const cvs = this.ov ? [{ cv: this.ov.cv, title: D.overlayTitle }] : this.panels.filter(d => !d.missing).map(d => ({ cv: d.cv, d }));
      // header text: what is plotted, every post-hoc fit in use, and what the colours are (the GIF travels without the page)
      const wrap = (font, text, maxW) => {
        g.font = font; const out = []; let line = "";
        for (const w of text.split(" ")) { const t = line ? `${line} ${w}` : w; if (line && g.measureText(t).width > maxW) { out.push(line); line = w; } else line = t; }
        if (line) out.push(line); return out;
      };
      const cols = cvs.length === 1 ? 1 : cvs.length <= 4 ? 2 : 3, gap = 8, cellW = Math.floor((W - gap * (cols + 1)) / cols);
      const ar = cvs[0].cv.height / cvs[0].cv.width, cellH = Math.round(cellW * ar), capH = 18, rows = Math.ceil(cvs.length / cols);
      const legend = this.gifLegend(), legH = legend.length ? 22 * Math.ceil(legend.length / Math.max(1, Math.floor(W / 120))) + 8 : 0;
      const out = document.createElement("canvas"), g = out.getContext("2d", { willReadFrequently: true });
      const small = "11px system-ui, sans-serif", tw = W - 2 * gap;
      const notes = [...wrap(small, D.kind + ".", tw),
        ...(D.mixedSteps ? wrap(small, `${D.checkpoint}. ${this.otherLoaded().map(d => `${d.net.short || d.net.label}: ${d.step.toLocaleString("en")}`).join("; ")}.`, tw) : []),
        ...(D.fits ? wrap(small, `${D.fitsBadge}: ${D.fits}.`, tw) : []),
        ...(D.noFit ? wrap(small, `No comparison fit: ${D.noFit}.`, tw) : []),
        ...wrap(small, `Colour: ${D.colour}.`, tw)];
      const trackY = 18 + 6 + 14 * notes.length + 12, headH = trackY + 20;
      const H = headH + rows * (cellH + capH + gap) + gap + legH + 16;
      out.width = W; out.height = H;
      const depths = []; for (let k = 0; k <= (L - 1) * perSite; k++) depths.push(k / perSite);
      const frames = [];
      for (let f = 0; f < depths.length; f++) {
        st.depth = depths[f]; $("emb-depth").value = st.depth; this.draw();
        if (this.exportCancel) throw new Error("cancelled");
        if (document.body.dataset.tab !== "embedding") throw new Error("cancelled: the viewer was hidden");
        progress(`rendering frame ${f + 1} of ${depths.length}…`);
        g.fillStyle = "#ffffff"; g.fillRect(0, 0, W, H);
        // header: reference site and a depth track
        const ref = this.refD.net, lc = Math.round(st.depth);
        g.fillStyle = "#16161a"; g.font = "600 15px system-ui, sans-serif"; g.textAlign = "left";
        g.fillText(`${this.st.method === "modes" ? "Through-depth modes" : (this.st.method === "pca" ? "PCA" : "UMAP") + " baseline"} · ${multi ? "reference " : ""}step ${this.refD.step.toLocaleString("en")} · ${ref.label}: ${ref.sites[lc]}`, gap, 18, tw);
        g.font = small; g.fillStyle = "#4a4a50"; notes.forEach((t, k) => g.fillText(t, gap, 18 + 6 + 14 * (k + 1) - 3));
        const x0 = gap + 4, x1 = W - gap - 4, y = trackY; g.strokeStyle = "#d4d4d4"; g.lineWidth = 2; g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke();
        g.font = "10px system-ui, sans-serif"; g.textAlign = "center";
        const every = Math.ceil(L / Math.max(1, Math.floor((x1 - x0) / 34)));
        ref.sites.forEach((s2, l) => { const x = x0 + (x1 - x0) * l / Math.max(1, L - 1); g.fillStyle = Math.abs(l - st.depth) < 0.5 ? "#c2410c" : "#9a9aa0"; g.beginPath(); g.arc(x, y, 3, 0, 7); g.fill(); if (l % every === 0 || l === L - 1) { g.textAlign = l === 0 ? "left" : l === L - 1 ? "right" : "center"; g.fillText(s2, l === 0 ? x0 - 4 : l === L - 1 ? x1 + 4 : x, y + 11); } });
        const xc = x0 + (x1 - x0) * st.depth / Math.max(1, L - 1); g.strokeStyle = "#c2410c"; g.beginPath(); g.arc(xc, y, 6, 0, 7); g.stroke();
        // panels
        g.textAlign = "left";
        cvs.forEach((c, j) => {
          const cx = gap + (j % cols) * (cellW + gap), cy = headH + Math.floor(j / cols) * (cellH + capH + gap);
          g.fillStyle = "#16161a"; g.font = "600 12px system-ui, sans-serif";
          g.fillText(c.d ? `${c.d.net.label}${multi && c.d === this.refD ? " (reference)" : ""} · ${this.siteName(c.d)}${this.panelStep(c.d)}` : c.title, cx, cy + 12, cellW);
          g.drawImage(c.cv, cx, cy + capH, cellW, cellH);
        });
        // legend
        let lx = gap, ly = H - legH - 4 + 14; g.font = "11px system-ui, sans-serif";
        for (const it of legend) {
          const tw = g.measureText(it.n).width + 26; if (lx + tw > W - gap) { lx = gap; ly += 22; }
          g.fillStyle = it.c; if (it.k != null) { mark(g, it.k, lx + 6, ly - 4, 5); g.fill(); } else { g.beginPath(); g.arc(lx + 6, ly - 4, 5, 0, 7); g.fill(); }
          g.fillStyle = "#16161a"; g.fillText(it.n, lx + 15, ly); lx += tw;
        }
        frames.push(g.getImageData(0, 0, W, H).data);
        await new Promise(r => requestAnimationFrame(r));
      }
      // encoding is interruptible too: Cancel (and leaving the tab) is checked before it starts and every few frames
      const check = () => {
        if (this.exportCancel) throw new Error("cancelled");
        if (document.body.dataset.tab !== "embedding") throw new Error("cancelled: the viewer was hidden");
      };
      progress("encoding…"); await new Promise(r => setTimeout(r, 0)); check();
      // one palette for all frames (from a sample of them), so colours do not flicker
      const pick = [0, Math.floor(frames.length / 2), frames.length - 1], step = 4 * 7, sample = [];
      for (const k of pick) { const f = frames[k]; for (let i = 0; i < f.length; i += step) sample.push(f[i], f[i + 1], f[i + 2], f[i + 3]); }
      const palette = quantize(new Uint8Array(sample), 256), gif = GIFEncoder(), delay = Math.round(1000 / 12);
      for (let k = 0; k < frames.length; k++) {
        if (k % 6 === 0) { progress(`encoding frame ${k + 1} of ${frames.length}…`); await new Promise(r => setTimeout(r, 0)); check(); }
        const idx = applyPalette(frames[k], palette);
        gif.writeFrame(idx, W, H, { palette: k ? undefined : palette, delay: k === 0 ? 800 : k === frames.length - 1 ? 2000 : delay });
      }
      check();
      gif.finish();
      const blob = new Blob([gif.bytes()], { type: "image/gif" }), a = document.createElement("a");
      a.href = URL.createObjectURL(blob); a.download = `depth-embedding_${this.st.method}_${this.refD.net.id}_${this.refD.step}${D.mixedSteps ? "_ref_other-final-checkpoints" : ""}${this.ov ? "_overlay" : ""}.gif`;
      document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 60000);
      info.textContent = `saved ${a.download} (${(blob.size / 1e6).toFixed(1)} MB, ${frames.length} frames)`;
    } catch (e) { if (e.message.startsWith("cancelled")) info.textContent = `GIF export ${e.message}.`; else { console.error(e); info.textContent = "GIF export failed: " + e.message; } }
    finally { st.depth = saved; $("emb-depth").value = saved; this.exporting = false; this.lockForExport(false); this.syncControls(); this.draw(); btn.focus({ preventScroll: true }); done(); }
  }
  gifLegend() {
    const ovl = this.overlayOn(), shapes = ovl && !(this.st.ocol === "net" && this.st.show !== "centroids")
      ? this.panels.map((d, k) => d.missing || this.hiddenNets.has(d.net.id) ? null : { n: d.net.short || d.net.label, c: "#6b6b70", k }).filter(Boolean) : [];
    if (ovl && this.st.show === "centroids") return [...shapes, ...this.m.classes.map((n, j) => this.st.color === "class" && this.hidden.has(j) ? null : { n, c: this.pal.classes[j] }).filter(Boolean)];
    if (ovl && this.st.ocol === "net") return this.panels.map((d, k) => d.missing || this.hiddenNets.has(d.net.id) ? null : { n: d.net.short || d.net.label, c: rgba(this.netColour(k), 1), k }).filter(Boolean);
    if (this.st.color === "class") return [...shapes, ...this.m.classes.map((n, j) => this.hidden.has(j) ? null : { n, c: this.pal.classes[j] }).filter(Boolean)];
    return shapes;
  }
  stop() { const was = this.playing; this.playing = false; $("emb-play").textContent = "▶ Play"; $("emb-play").setAttribute("aria-label", "Play through depth"); if (was) this.saveHash(); }
  tick(t) {
    if (!this.playing) return;
    const dt = this.lastT ? (t - this.lastT) / 1000 : 0; this.lastT = t;
    const L = this.ref().sites.length, prev = Math.round(this.st.depth);
    this.st.depth = Math.min(L - 1, this.st.depth + dt * Math.max(1.2, L / 8));
    $("emb-depth").value = this.st.depth; this.draw(); if (Math.round(this.st.depth) !== prev) this.legend();
    if (this.st.depth >= L - 1) { this.stop(); this.saveHash(); return; }
    requestAnimationFrame(t2 => this.tick(t2));
  }
}
