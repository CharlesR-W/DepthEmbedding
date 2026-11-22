"""Results page for the double-diffusion-map experiment: one self-contained HTML file.

    python3 invert/build_ddm_page.py            # -> invert/pages_ddm/ddm.html (stdlib only)
    bash invert/verify_ddm_page.sh /tmp/ddm-shots

Inputs: data/invert/ddm_results.json (layout in invert/ddm_collect.py) and invert/ddm_page_text.json (all prose,
keyed by tab and figure).  Plots draw in the browser (Plotly) from the numbers embedded here.  Means are shown for
cosine and agreement (bars +-2 s.e. over held-out images), medians for captured share, decoded norm and round-trip
misses (their means are dominated by a few images).
"""
from __future__ import annotations

import ast
import json
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from notebook import Notebook, Radio, Select, arch_svg, table_html  # noqa: E402

ROOT = HERE.parent
SRC = ROOT / 'data/invert/ddm_results.json'
OUT = HERE / 'pages_ddm' / 'ddm.html'
T = json.loads((HERE / 'ddm_page_text.json').read_text())
R = json.loads(SRC.read_text())
SITES = R['sites']
KS = [4, 16, 64, 256]


def r5(v):
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return [r5(x) for x in v]
    if isinstance(v, dict):
        return {k: r5(x) for k, x in v.items()}
    if isinstance(v, float):
        return None if v != v or v in (float('inf'), float('-inf')) else float(f'{v:.5g}')
    return v


def quart(xs):
    s = sorted(x for x in xs if x is not None)
    if not s:
        return None
    q = lambda p: s[min(len(s) - 1, max(0, round(p * (len(s) - 1))))]  # noqa: E731
    return [q(.25), q(.5), q(.75)]


# ------------------------------------------------------------------ one visual grammar --
# hue = decoder family, line style = variant (solid: GCV, no trend; dash: + linear trend; dot: inner-split rule;
# dashdot: chosen for function).  Shades of a hue are a parameter (width h, K, coordinate system).
FAM = dict(lift='#228833', ridge='#d0473f', linear='#4477aa', nn='#aa3377', ref='#7a7a7a')
LG = T['legend']
STY = {
    'lh': dict(color=FAM['lift'], dash='solid', symbol='circle', fam='lift'),
    'lhlin': dict(color=FAM['lift'], dash='dash', symbol='circle', fam='lift'),
    'lh_split': dict(color=FAM['lift'], dash='dot', symbol='circle-open', fam='lift'),
    'lh_fsplit': dict(color=FAM['lift'], dash='dashdot', symbol='diamond', fam='lift'),
    'kr': dict(color=FAM['ridge'], dash='solid', symbol='circle', fam='ridge'),
    'krlin': dict(color=FAM['ridge'], dash='dash', symbol='circle', fam='ridge'),
    'krr': dict(color=FAM['ridge'], dash='dot', symbol='square', fam='ridge'),
    'linear': dict(color=FAM['linear'], dash='solid', symbol='circle', fam='linear'),
    'nn1': dict(color='#aa3377', dash='solid', symbol='circle', fam='nn'),
    'nn15': dict(color='#d68fbb', dash='solid', symbol='triangle-up', fam='nn'),
    'nn1+krr': dict(color='#5e1742', dash='solid', symbol='diamond', fam='nn'),
    'ref': dict(color=FAM['ref'], dash='dash', symbol='diamond-open', fam='ref'),
    'mlp': dict(color='#111111', dash='dot', symbol='x', fam='ref'),
}
for k, s in STY.items():
    s['label'] = LG.get(k, k)
SHADE = {   # 7 shades, light -> dark, for the width h (and a subset for K or coordinate system)
    'lift': ['#a6dba0', '#7cc674', '#53ad52', '#2f9440', '#1b7a32', '#0e5e24', '#063f17'],
    'ridge': ['#f4b2a6', '#ec8a7a', '#de6151', '#c93f31', '#a72a22', '#811c16', '#5a100c'],
    'linear': ['#b4cbe6', '#8eb0d8', '#6893c6', '#4477aa', '#2f5d8c', '#1f456b', '#122d48'],
}
KSHADE = ['#b3c2d9', '#7f97bd', '#4c6b9c', '#1f3a66']
COORD_STY = {'joint': dict(i=5, symbol='circle'), 'source': dict(i=3, symbol='square'), 'site': dict(i=1, symbol='diamond')}

# ------------------------------------------------------------------ payloads --
FIN_KEYS = dict(cos='cos', agree='agree')            # [mean, sem]
MED_KEYS = dict(cap='cap', norm='norm', rtj='rt_joint', rtq='rt_req', rto='rt_one', rts='rt_src')   # median


def fin_payload():
    out = {}
    for cell, bys in R['finals'].items():
        out[cell] = {}
        for s in SITES:
            out[cell][s] = {}
            for dec, e in bys.get(s, {}).items():
                o = {k: r5([e[f][0], e[f][2]]) for k, f in FIN_KEYS.items() if f in e}
                o.update({k: r5(e[f][1]) for k, f in MED_KEYS.items() if f in e})
                out[cell][s][dec] = o
    return out


def sel_payload():
    out = {}
    for cell, bys in R['sel'].items():
        out[cell] = {}
        for s in SITES:
            e = bys.get(s)
            if not e:
                continue
            g = e.get('gcv', {})
            pick = lambda d, *ks: {k: d.get(k) for k in ks} if d else None  # noqa: E731
            out[cell][s] = r5(dict(
                lh=pick(g.get('lh'), 'h', 'k2', 'm', 'H_cos'), kr=pick(g.get('kr'), 'h', 'ridge', 'H_cos'),
                oracle=pick(e.get('oracle_cos'), 'h', 'm', 'H_cos'),
                fsplit=dict(h=e['fsplit'].get('h'), m=e['fsplit'].get('m')) if e.get('fsplit') else None))
    return out


def func_payload():
    out = {}
    for cell, bys in R['func'].items():
        out[cell] = {}
        for s in SITES:
            out[cell][s] = {}
            for h, v in bys[s].items():
                o = {}
                for path in ('cut', 'ridge'):
                    p = v[path]
                    assert path == 'cut' or p['ridges'] == R['ridges']
                    rt = [x[1] for x in p['rt_joint']] if p.get('rt_joint') else [x[1] for x in p['rt'].get('req', [])]
                    # the cut-off grid stops at the number of numerically usable harmonics, which falls with h
                    o[path] = r5(dict(x=p['cuts'] if path == 'cut' else R['ridges'], cos=p['cos'], agree=p['agree'],
                                      norm=p['norm_med'], rt=rt))
                out[cell][s][h] = o
    return out


def surf_payload():
    out = {}
    for cell, bys in R['surface'].items():
        out[cell] = {s: {h: r5(dict(cuts=v['cuts'], cos=v['H_cos'], gcv=v['gcv'], kcos=v['kr_H_cos'], kgcv=v['kr_gcv']))
                         for h, v in bys[s].items()} for s in SITES}
    return out


def later_payload():
    out = {}
    for cell, bys in R['later'].items():
        out[cell] = {}
        for s in SITES:
            L = bys[s]
            out[cell][s] = r5(dict(linear=dict(r2=quart(L['linear']), rho2=quart(L['linear_rho2'])),
                                   **{a: dict(r2=quart(L[a]['r2']), rho2=quart(L[a]['rho2']), n=len(L[a]['r2']))
                                      for a in ('lh', 'kr')}))
    return out


def llr_payload():
    return {s: r5(dict(site=v['site'], joint=v['joint'], sel_site=v['sel_site'], sel_joint=v['sel_joint']))
            for s, v in R['llr'].items()}


def span(vals, pad=0.04, lo=None, hi=None):
    v = [x for x in vals if x is not None]
    a, b = min(v + ([lo] if lo is not None else [])), max(v + ([hi] if hi is not None else []))
    p = (b - a) * pad
    return [round(a - p, 4), round(b + p, 4)]


def ranges(fin, func, surf):
    """Fixed axis ranges, so a selector moves the data, not the axes."""
    rg = {}
    decs = ['linear', 'lh', 'lhlin', 'lh_split', 'kr', 'krlin', 'krr']
    cells = [f'{c}|{K}' for c in ('joint', 'source', 'site') for K in KS]
    for m in ('cos', 'cap', 'agree'):
        vals = []
        for c in cells:
            for s in SITES:
                for d in decs:
                    e = fin[c][s].get(d, {}).get(m)
                    if e is None:
                        continue
                    if isinstance(e, list):
                        vals += [e[0] - 2 * e[1], e[0] + 2 * e[1]]
                    else:
                        vals.append(e)
        rg['cmp_' + m] = span(vals, lo=0 if m == 'agree' else None, hi=1 if m == 'agree' else None)
    gap, rule = [], []
    for c in cells:
        for s in SITES:
            f = fin[c][s]
            gap += [f['kr']['cos'][0] - f['lh']['cos'][0], f['krlin']['cos'][0] - f['lhlin']['cos'][0]]
            sl = R['sel'][c][s]
            rule += [sl['oracle_cos']['H_cos'] - sl['gcv']['lh']['H_cos'], f['lh']['cos'][0] - f['lh_split']['cos'][0],
                     f['krlin']['cos'][0] - f['krr']['cos'][0]]
    rg['gap'] = span(gap, lo=0)
    rg['rule'] = span(rule, lo=0)
    rg['coords'] = span([fin[f'{c}|{K}'][s][d]['cos'][0] for c in ('joint', 'source', 'site') for K in KS for s in SITES
                         for d in ('kr', 'lh', 'linear')] + [R['ref'][s]['cos'][0] for s in SITES], hi=1)
    var = []
    for d in ('kr', 'lh', 'linear'):
        for s in SITES:
            var += [fin['site-t1|64'][s][d]['cos'][0] - fin['site|64'][s][d]['cos'][0],
                    fin['joint-llr16|16'][s][d]['cos'][0] - fin['joint|16'][s][d]['cos'][0],
                    fin['site-llr16|16'][s][d]['cos'][0] - fin['site|16'][s][d]['cos'][0]]
    rg['var'] = span(var, lo=0)
    rg['path'] = {}
    for cell, bys in func.items():
        for path in ('cut', 'ridge'):
            vals = {k: [] for k in ('cos', 'agree', 'norm', 'rt')}
            for s in SITES:
                for v in bys[s].values():
                    for k in vals:
                        vals[k] += [x for x in v[path][k] if x is not None]
            rt = [x for x in vals['rt'] if x > 0]
            rg['path'][f'{cell}|{path}'] = dict(cos=span(vals['cos']), agree=[0, 1.02], norm=span(vals['norm'], lo=0, hi=1),
                                                rt=[round(_log10(min(rt)) - 0.1, 3), round(_log10(max(rt)) + 0.1, 3)] if rt else None)
    rg['surf'] = {}
    for cell, bys in surf.items():
        c, g = [], []
        for s in SITES:
            for v in bys[s].values():
                c += v['cos'] + v['kcos']
                g += v['gcv'] + v['kgcv']
        g = [x for x in g if x is not None and 0 < x]
        rg['surf'][cell] = dict(cos=span([x for x in c if x is not None and x > 0], lo=0),
                                gcv=[round(_log10(min(g)) - 0.03, 3), round(_log10(min(max(g), 10)) + 0.03, 3)])
    rts = [fin[f'{c}|{K}'][s][d].get(k) for c, k in (('joint', 'rtj'), ('site', 'rtq'), ('site', 'rto')) for K in (4, 16, 64)
           for s in SITES for d in fin[f'{c}|{K}'][s]]
    rts = [x for x in rts if x is not None and x > 0]
    rg['rt'] = [round(_log10(min(rts)) - 0.15, 3), round(_log10(max(rts)) + 0.15, 3)]
    return rg


def _log10(x):
    import math
    return math.log10(x)


# ------------------------------------------------------------------ JS --
JS = r"""
const M = DATA.meta, SITES = M.sites, STY = M.sty;
function st(d) { return STY[d] || {color:'#777', dash:'solid', label:d}; }
function siteAxis(extra) { return merge(AXIS, merge({type:'category', categoryorder:'array', categoryarray: SITES,
  range:[-0.5, SITES.length - 0.5], title:{text:'site'}}, extra || {})); }
function empty(el, msg) { try { Plotly.purge(el); } catch (e) {} el.innerHTML = '<p class="empty">' + msg + '</p>'; }
function sub(el, text) { const f = el.closest('figure'); const p = f && f.querySelector('.fsub'); if (p) p.textContent = text; }
function plot(el, traces, layout) {
  traces = traces.filter(t => t && ((t.y && t.y.some(v => v != null)) || t.type === 'heatmap' || t.legendonly));
  if (!traces.length) { empty(el, 'No data for this selection.'); return; }
  layout = Object.assign({}, layout);
  const names = layout.showlegend === false ? [] : traces.filter(t => t.showlegend !== false && t.name && t.type !== 'heatmap').map(t => t.name);
  if (names.length) {   // legend above the plot, in a top margin sized to its estimated rows
    const W = Math.max(280, (el.clientWidth || 900) - 70); let rows = 1, x = 0;
    for (const n of names) { const w = n.length * 6.9 + 46; if (x + w > W && x > 0) { rows++; x = 0; } x += w; }
    layout.legend = {orientation:'h', x:0, xanchor:'left', y:1.0, yanchor:'bottom', font:{size:12}, traceorder:'normal'};
    layout.margin = merge({l:60, r:18, t:36, b:48}, merge(layout.margin || {}, {t: (layout.toppad || 10) + rows * 20 + 6}));
    layout.height = (layout.height || 380) + rows * 20;
    if (layout.toppad) { const plotH = layout.height - layout.margin.t - (layout.margin.b == null ? 48 : layout.margin.b);
      layout.legend.y = 1 + (layout.toppad - 4) / plotH; }
  }
  delete layout.toppad;
  const p = el.querySelector(':scope > p.empty'); if (p) p.remove();
  Plotly.react(el, traces, LAYOUT(layout), CONFIG);
}
function val(v) { return v == null ? null : (Array.isArray(v) ? v[0] : v); }
function sline(name, ys, s, extra) {
  const y = ys.map(val);
  const t = {type:'scatter', mode:'lines+markers', name, x: SITES, y, connectgaps:false,
    line:{color: s.color, dash: s.dash || 'solid', width: s.width || 2},
    marker:{size: s.msize || 6, color: s.color, symbol: s.symbol || 'circle'},
    hovertemplate: name + ' · %{x}: %{y:.3~g}<extra></extra>'};
  const e = ys.map(v => Array.isArray(v) && v[1] != null ? 2 * v[1] : null);
  if (e.some(v => v != null && v > 0)) t.error_y = {type:'data', array: e.map(v => v || 0), thickness:1, width:2, color: s.color};
  return merge(t, extra || {});
}
function legendOnly(name, s) { return {type:'scatter', mode: s.mode || 'lines+markers', x:[null], y:[null], name, legendonly:true,
  line:{color: s.color, dash: s.dash || 'solid', width: s.width || 2}, marker:{size: s.msize || 7, color: s.color, symbol: s.symbol || 'circle'}}; }
function facetAxes(n, gap) { const w = (1 - gap * (n - 1)) / n; return Array.from({length:n}, (_, i) => [i * (w + gap), i * (w + gap) + w]); }
function refLine(y, txt, xref, yref, log) { return {
  shape: {type:'line', xref: xref || 'paper', yref: yref || 'y', x0: 0, x1: 1, y0: y, y1: y, line:{color:'#8a8a8a', dash:'dot', width:1.2}},
  ann: txt ? {xref: xref || 'paper', yref: yref || 'y', x: 1, y: log ? Math.log10(y) : y, xanchor:'right', yanchor: y < 1.5 ? 'top' : 'bottom', text: txt, showarrow:false,
              font:{size:11, color:'#6b6b6b'}} : null}; }
function addRefs(lay, refs) { lay.shapes = (lay.shapes || []).concat(refs.map(r => r.shape));
  lay.annotations = (lay.annotations || []).concat(refs.map(r => r.ann).filter(a => a)); return lay; }
function zeroLine(yref) { return {type:'line', xref:'paper', yref: yref || 'y', x0:0, x1:1, y0:0, y1:0, line:{color:'#6b665e', width:1}}; }
function panelTitle(text, dom, top) { return {xref:'paper', yref:'paper', x: dom[0], y: top == null ? 1.0 : top, xanchor:'left', yanchor:'bottom',
  text:'<b>' + text + '</b>', showarrow:false, font:{size:12.5}}; }
const LOGX = {type:'log', exponentformat:'power'};
function fixLog(ax) { return merge(ax, {type:'log', exponentformat:'power', dtick: 1}); }
// MathJax: the engine typesets a tab when it is switched to, which misses the first tab (and the banner) when
// MathJax finishes loading after DOMContentLoaded.  Typeset those once it is ready.
(function mjKick() {
  if (window.MathJax && MathJax.typesetPromise && MathJax.startup && MathJax.startup.promise) {
    MathJax.startup.promise.then(() => { const els = [document.querySelector('.banner')];
      const sec = document.querySelector('section.tab.active'); if (sec && !sec.dataset.typeset) { sec.dataset.typeset = '1'; els.push(sec); }
      return MathJax.typesetPromise(els.filter(e => e)); }).catch(e => console.error(e));
  } else setTimeout(mjKick, 120);
})();
"""

# ---- tab 1: matched comparison
DRAW_CMP = r"""(el, d, c) => {
  const cell = c.coords + '|' + c.K, F = DATA.fin[cell], decs = ['linear','lh','lhlin','lh_split','kr','krlin','krr'];
  sub(el, c.coords + ' coordinates · K = ' + c.K + ' · held-out set H, 2,000 images');
  const mets = [['cos','cosine (mean)'],['cap','captured share (median)'],['agree','agreement (mean)']];
  const dom = facetAxes(3, 0.075), tr = [], lay = {height: 420, annotations: []};
  mets.forEach(([m, lab], i) => { const ax = i ? String(i + 1) : '';
    for (const dec of decs) { const s = st(dec);
      tr.push(sline(s.label, SITES.map(x => F[x][dec] ? F[x][dec][m] : null), merge(s, {msize: 5, width: 1.8}),
        {xaxis:'x' + ax, yaxis:'y' + ax, legendgroup: dec, showlegend: i === 0, hovertemplate: s.label + ' · %{x}: %{y:.3f}<extra></extra>'})); }
    lay['xaxis' + ax] = siteAxis({domain: dom[i], anchor: 'y' + ax, title:{text:'site'}, tickangle: -45});
    lay['yaxis' + ax] = merge(AXIS, {anchor:'x' + ax, range: M.rng['cmp_' + m], title:{text: lab}});
  });
  plot(el, tr, lay);
}"""

DRAW_GAP = r"""(el, d, c) => {
  sub(el, c.coords + ' coordinates · all K (current K drawn thick) · mean over H');
  const tr = [];
  M.Ks.forEach((K, i) => { const F = DATA.fin[c.coords + '|' + K], col = M.kshade[i], cur = K === c.K;
    tr.push(sline('K = ' + K, SITES.map(s => F[s].kr.cos[0] - F[s].lh.cos[0]), {color: col, width: cur ? 3.2 : 1.8, msize: cur ? 7 : 5},
      {legendgroup:'K' + K, hovertemplate: 'K = ' + K + ' · ridge − cut-off · %{x}: %{y:.4f}<extra></extra>'}));
    tr.push(sline('K = ' + K + ' with trend', SITES.map(s => F[s].krlin.cos[0] - F[s].lhlin.cos[0]), {color: col, dash:'dash', width: cur ? 2 : 1.2, msize: 3},
      {legendgroup:'K' + K, showlegend:false, hovertemplate: 'K = ' + K + ' · with trend · %{x}: %{y:.4f}<extra></extra>'}));
  });
  tr.push(legendOnly('dashed: both with linear trend', {color:'#555', dash:'dash', msize:3}));
  plot(el, tr, {height: 340, shapes:[zeroLine()], xaxis: siteAxis(), yaxis: merge(AXIS, {range: M.rng.gap,
    title:{text:'Δ cosine, ridge − cut-off'}, zeroline:false})});
}"""

DRAW_RULE = r"""(el, d, c) => {
  const cell = c.coords + '|' + c.K, S = DATA.sel[cell], F = DATA.fin[cell];
  sub(el, c.coords + ' coordinates · K = ' + c.K + ' · held-out cosine differences');
  const tr = [
    sline('best grid point on H − GCV (cut-off)', SITES.map(s => S[s].oracle.H_cos - S[s].lh.H_cos), {color:'#7a7a7a', symbol:'circle'}),
    sline('GCV − inner split (cut-off)', SITES.map(s => F[s].lh.cos[0] - F[s].lh_split.cos[0]), st('lh_split')),
    sline('GCV − inner split (ridge, with trend: krlin − krr)', SITES.map(s => F[s].krlin.cos[0] - F[s].krr.cos[0]), st('krr')),
  ];
  plot(el, tr, {height: 320, shapes:[zeroLine()], xaxis: siteAxis(), yaxis: merge(AXIS, {range: M.rng.rule, title:{text:'Δ cosine'}, zeroline:false})});
}"""

DRAW_SURF = r"""(el, d, c) => {
  const S = DATA.surf[c.scell][c.ssite], sel = DATA.sel[c.scell][c.ssite], rg = M.rng.surf[c.scell];
  sub(el, c.scell.replace('|', ' coordinates · K = ') + ' · site ' + c.ssite + ' · cosine on H (2,000 images), GCV on the probe (8,000)');
  const dx = facetAxes(2, 0.08), tr = [], lay = {height: 620, annotations: [], shapes: []};
  const hs = M.hs.map(String);
  let best = {cut: [-9], ridge: [-9]};
  hs.forEach((h, i) => { const v = S[h]; if (!v) return;
    v.cos.forEach((y, j) => { if (y != null && y > best.cut[0]) best.cut = [y, v.cuts[j], v.gcv[j], h]; });
    v.kcos.forEach((y, j) => { if (y != null && y > best.ridge[0]) best.ridge = [y, M.ridges[j], v.kgcv[j], h]; });
    const gc = M.shade.lift[i], rc = M.shade.ridge[i], lw = 1.7;
    const hv = (p, w) => 'h = ' + h + ' · ' + p + ' %{x:.3~g}: %{y:.4f}<extra></extra>';
    tr.push({type:'scatter', mode:'lines', x: v.cuts, y: v.cos, name: 'h = ' + h, legendgroup: h, line:{color: gc, width: lw}, xaxis:'x', yaxis:'y', hovertemplate: hv('m')});
    tr.push({type:'scatter', mode:'lines', x: v.cuts, y: v.gcv, legendgroup: h, showlegend:false, line:{color: gc, width: lw}, xaxis:'x3', yaxis:'y3', hovertemplate: hv('m')});
    tr.push({type:'scatter', mode:'lines', x: M.ridges, y: v.kcos, legendgroup: h, showlegend:false, line:{color: rc, width: lw}, xaxis:'x2', yaxis:'y2', hovertemplate: hv('ρ')});
    tr.push({type:'scatter', mode:'lines', x: M.ridges, y: v.kgcv, legendgroup: h, showlegend:false, line:{color: rc, width: lw}, xaxis:'x4', yaxis:'y4', hovertemplate: hv('ρ')});
  });
  const at = (h, key, xs, x) => { const v = S[String(h)]; if (!v) return null; let j = xs(v).indexOf(x);
    if (j < 0) { let b = 1e99; xs(v).forEach((q, k) => { const e = Math.abs(Math.log(q) - Math.log(x)); if (e < b) { b = e; j = k; } }); } return [xs(v)[j], v[key[0]][j], v[key[1]][j]]; };
  const star = (x, y, ax, name, show, sym) => ({type:'scatter', mode:'markers', x:[x], y:[y], xaxis:'x' + ax, yaxis:'y' + ax, name, legendgroup: name,
    showlegend: show, marker:{symbol: sym, size: sym === 'star' ? 15 : 13, color: '#111', line:{color:'#111', width: sym === 'star' ? 1 : 2}},
    hovertemplate: name + ': %{x:.3~g}, %{y:.4f}<extra></extra>'});
  if (sel.lh) { const p = at(sel.lh.h, ['cos','gcv'], v => v.cuts, sel.lh.m); if (p) { tr.push(star(p[0], p[1], '', 'GCV choice', true, 'star')); tr.push(star(p[0], p[2], '3', 'GCV choice', false, 'star')); }
    lay.annotations.push({xref:'x3 domain', yref:'y3 domain', x:0.02, y:0.97, xanchor:'left', yanchor:'top', showarrow:false, font:{size:11.5, color:'#333'}, bgcolor:'rgba(255,255,255,0.8)',
      text:'GCV: h = ' + sel.lh.h + ', m = ' + sel.lh.m.toLocaleString() + ' · cosine ' + sel.lh.H_cos.toFixed(3) + '<br>best on H: h = ' + best.cut[3] + ', m = ' + best.cut[1].toLocaleString() + ' · cosine ' + best.cut[0].toFixed(3)}); }
  if (sel.kr) { const p = at(sel.kr.h, ['kcos','kgcv'], v => M.ridges, sel.kr.ridge); if (p) { tr.push(star(p[0], p[1], '2', 'GCV choice', false, 'star')); tr.push(star(p[0], p[2], '4', 'GCV choice', false, 'star')); }
    lay.annotations.push({xref:'x4 domain', yref:'y4 domain', x:0.02, y:0.97, xanchor:'left', yanchor:'top', showarrow:false, font:{size:11.5, color:'#333'}, bgcolor:'rgba(255,255,255,0.8)',
      text:'GCV: h = ' + sel.kr.h + ', ρ = ' + sel.kr.ridge.toExponential(0) + ' · cosine ' + sel.kr.H_cos.toFixed(3) + '<br>best on H: h = ' + best.ridge[3] + ', ρ = ' + best.ridge[1].toExponential(0) + ' · cosine ' + best.ridge[0].toFixed(3)}); }
  tr.push(star(best.cut[1], best.cut[0], '', 'best grid point on H', true, 'circle-open'));
  tr.push(star(best.ridge[1], best.ridge[0], '2', 'best grid point on H', false, 'circle-open'));
  if (best.cut[2] != null) tr.push(star(best.cut[1], best.cut[2], '3', 'best grid point on H', false, 'circle-open'));
  if (best.ridge[2] != null) tr.push(star(best.ridge[1], best.ridge[2], '4', 'best grid point on H', false, 'circle-open'));
  const dy = [[0.56, 1], [0, 0.44]];
  lay.xaxis = merge(AXIS, merge(LOGX, {domain: dx[0], anchor:'y', dtick: 1, range:[0, Math.log10(8000) + 0.05], title:{text:''}, showticklabels:false}));
  lay.xaxis2 = merge(AXIS, merge(LOGX, {domain: dx[1], anchor:'y2', dtick: 1, range:[2.1, -8.1], title:{text:''}, showticklabels:false}));
  lay.xaxis3 = merge(AXIS, merge(LOGX, {domain: dx[0], anchor:'y3', dtick: 1, range:[0, Math.log10(8000) + 0.05], title:{text:'latent harmonics kept, m (log)'}, matches:'x'}));
  lay.xaxis4 = merge(AXIS, merge(LOGX, {domain: dx[1], anchor:'y4', dtick: 1, range:[2.1, -8.1], title:{text:'ridge ρ (log, reversed: less regularisation →)'}, matches:'x2'}));
  lay.yaxis = merge(AXIS, {domain: dy[0], anchor:'x', range: rg.cos, title:{text:'held-out cosine (mean)'}});
  lay.yaxis2 = merge(AXIS, {domain: dy[0], anchor:'x2', range: rg.cos, matches:'y', showticklabels:true});
  lay.yaxis3 = fixLog(merge(AXIS, {domain: dy[1], anchor:'x3', range: rg.gcv, title:{text:'GCV on the probe (log; lower is better)'}}));
  lay.yaxis4 = fixLog(merge(AXIS, {domain: dy[1], anchor:'x4', range: rg.gcv, matches:'y3'}));
  delete lay.yaxis3.dtick; delete lay.yaxis4.dtick;
  lay.shapes.push({type:'rect', xref:'x3', yref:'y3 domain', x0: 4000, x1: 8000, y0: 0, y1: 1, fillcolor:'rgba(120,120,120,0.12)', line:{width:0}});
  lay.annotations.push({xref:'x3', yref:'y3 domain', x: Math.log10(4000), y: 0.03, xanchor:'right', yanchor:'bottom', showarrow:false,
    text:'df > n/2 →<br>not eligible', font:{size:10.5, color:'#6b6b6b'}});
  lay.annotations.push(panelTitle('spectral cut-off (green)', dx[0]), panelTitle('ridge (red)', dx[1]));
  lay.toppad = 26;
  plot(el, tr, lay);
}"""

# ---- tab 2: reconstruction against function
DRAW_PATH = r"""(el, d, c) => {
  const F = DATA.func[c.cell][c.site], sel = DATA.sel[c.cell][c.site], rg = M.rng.path[c.cell + '|' + c.path];
  const cut = c.path === 'cut', kind = c.cell.split('|')[0];
  const g = cut ? sel.lh : sel.kr, gx = cut ? g.m : g.ridge;
  sub(el, c.cell.replace('|', ' coordinates · K = ') + ' · site ' + c.site + ' · ' + (cut ? 'spectral cut-off' : 'ridge') + ' · GCV chose h = ' + g.h + ', ' + (cut ? 'm = ' + g.m.toLocaleString() : 'ρ = ' + g.ridge.toExponential(0)));
  const rtName = kind === 'joint' ? 'joint re-embedding miss (median)' : kind === 'site' ? 'restriction miss with 1/λ (median)' : 'restriction miss (median)';
  const mets = [['cos', 'cosine (mean)'], ['agree', 'agreement (mean)'], ['norm', 'decoded norm / true norm (median)'], ['rt', rtName]];
  const dx = facetAxes(2, 0.09), dy = [[0.56, 1], [0, 0.42]], tr = [], lay = {height: 640, shapes: [], annotations: []};
  M.hs.forEach((h, i) => { const v = F[String(h)]; if (!v) return; const p = v[c.path], chosen = h === g.h;
    const col = M.shade[cut ? 'lift' : 'ridge'][i];
    mets.forEach(([m], k) => { const ax = k ? String(k + 1) : '';
      tr.push({type:'scatter', mode:'lines+markers', x: p.x, y: p[m], name: 'h = ' + h + (chosen ? ' (GCV)' : ''), legendgroup: 'h' + h, showlegend: k === 0,
        xaxis:'x' + ax, yaxis:'y' + ax, line:{color: col, width: chosen ? 3.4 : 1.6}, marker:{size: chosen ? 6 : 4, color: col},
        hovertemplate: 'h = ' + h + ' · ' + (cut ? 'm' : 'ρ') + ' %{x:.3~g}: %{y:.3f}<extra></extra>'}); }); });
  mets.forEach(([m, lab], k) => { const ax = k ? String(k + 1) : '', row = k < 2 ? 0 : 1, col = k % 2;
    const xa = merge(AXIS, merge(LOGX, {domain: dx[col], anchor: 'y' + ax, dtick: 1,
      title:{text: row ? (cut ? 'latent harmonics kept, m (log)' : 'ridge ρ (log, reversed)') : ''},
      range: cut ? [Math.log10(6), Math.log10(8000) + 0.08] : [2.2, -8.2]}));
    let ya = merge(AXIS, {domain: dy[row], anchor: 'x' + ax, title:{text: lab}, range: rg[m]});
    if (m === 'rt') { if (!rg.rt) { ya = merge(ya, {range:[-1, 1]}); } ya = fixLog(merge(ya, {range: rg.rt || [-1, 1]})); }
    lay['xaxis' + ax] = xa; lay['yaxis' + ax] = ya;
    lay.shapes.push({type:'line', xref:'x' + ax, yref:'y' + ax + ' domain', x0: gx, x1: gx, y0: 0, y1: 1, line:{color:'#111', dash:'dot', width:1}});
    if (m === 'norm') lay.shapes.push({type:'line', xref:'x' + ax + ' domain', yref:'y' + ax, x0:0, x1:1, y0:1, y1:1, line:{color:'#8a8a8a', dash:'dot', width:1}});
  });
  lay.annotations.push({xref:'x', yref:'y domain', x: Math.log10(gx), y: 0.98, xanchor: 'left', yanchor:'top', xshift: 3, showarrow:false,
    text:'GCV choice', font:{size:11, color:'#111'}});
  plot(el, tr, lay);
}"""

DRAW_FRONT = r"""(el, d, c) => {
  const K = c.cell.split('|')[1], kind = c.cell.split('|')[0], F = DATA.fin[c.cell][c.site], sel = DATA.sel[c.cell][c.site], P = DATA.func[c.cell][c.site];
  sub(el, c.cell.replace('|', ' coordinates · K = ') + ' · site ' + c.site + ' · held-out set H');
  const tr = [];
  const path = (pkey, h, nm, fam, lab) => { const p = P[String(h)][pkey], xs = pkey === 'cut' ? p.x : p.x.map(r => r.toExponential(0));
    const n = xs.length, lbl = xs.map((x, i) => (i === 0 || i === n - 1) ? lab + ' ' + (pkey === 'cut' ? Number(x).toLocaleString() : x) : '');
    tr.push({type:'scatter', mode:'lines+markers+text', x: p.cos, y: p.agree, name: nm, line:{color: M.fam[fam], width: 2}, marker:{size: 4, color: M.fam[fam]},
      text: lbl, textposition: pkey === 'cut' ? 'bottom left' : 'top right', textfont:{size: 10.5, color: M.fam[fam]},
      customdata: xs.map(x => lab + ' ' + (pkey === 'cut' ? Number(x).toLocaleString() : x)),
      hovertemplate: nm + ' · %{customdata}: cosine %{x:.3f}, agreement %{y:.3f}<extra></extra>'}); };
  path('cut', sel.lh.h, 'cut-off path (h = ' + sel.lh.h + ')', 'lift', 'm =');
  path('ridge', sel.kr.h, 'ridge path (h = ' + sel.kr.h + ')', 'ridge', 'ρ =');
  const pt = (dec, label, x, y, extra) => { const s = st(dec); if (x == null || y == null) return;
    tr.push(merge({type:'scatter', mode:'markers', x:[x], y:[y], name: label || s.label,
      marker:{symbol: s.symbol, size: 12, color: s.color, line:{color: s.color, width: 2}},
      hovertemplate: (label || s.label) + ': cosine %{x:.3f}, agreement %{y:.3f}<extra></extra>'}, extra || {})); };
  for (const dec of ['lh', 'kr', 'krr', 'linear', 'lh_fsplit', 'nn1', 'nn15', 'nn1+krr']) if (F[dec]) pt(dec, null, F[dec].cos[0], F[dec].agree[0],
    (dec === 'lh' || dec === 'kr') ? {marker:{size: 15, line:{color:'#111', width: 1.5}}} : {});
  const ref = DATA.ref[c.site]; pt('ref', null, ref.cos, ref.agree);
  const pr = DATA.prior[c.site][(kind === 'joint') ? 'mlp|' + K : ''];
  if (pr && pr.cos != null) pt('mlp', st('mlp').label + ' · joint K = ' + K, pr.cos, pr.agree);
  plot(el, tr, {height: 520, xaxis: merge(AXIS, {range:[-0.05, 1.0], title:{text:'cosine (mean) → better reconstruction'}}),
    yaxis: merge(AXIS, {range:[0, 1.02], title:{text:'agreement (mean) → better function'}})});
}"""

# ---- tab 3: coordinate systems
DRAW_COORDS = r"""(el, d, c) => {
  const dec = c.dec, s0 = st(dec), dom = facetAxes(3, 0.05), tr = [], lay = {height: 430, annotations: []};
  sub(el, s0.label + ' · mean over H, bars ±2 s.e.');
  [16, 64, 256].forEach((K, i) => { const ax = i ? String(i + 1) : '';
    for (const cs of ['joint', 'source', 'site']) { const F = DATA.fin[cs + '|' + K], cst = M.coord[cs];
      tr.push(sline(cs + ' coordinates', SITES.map(s => F[s][dec] ? F[s][dec].cos : null),
        {color: M.shade[s0.fam][cst.i], symbol: cst.symbol, msize: 6}, {xaxis:'x' + ax, yaxis:'y' + ax, legendgroup: cs, showlegend: i === 0})); }
    tr.push(sline(st('ref').label, SITES.map(s => DATA.ref[s].cos), merge(st('ref'), {msize: 5, width: 1.5}),
      {xaxis:'x' + ax, yaxis:'y' + ax, legendgroup:'ref', showlegend: i === 0}));
    lay['xaxis' + ax] = siteAxis({domain: dom[i], anchor:'y' + ax, tickangle:-45});
    lay['yaxis' + ax] = merge(AXIS, {anchor:'x' + ax, range: M.rng.coords, title:{text: i ? '' : 'cosine (mean)'}, showticklabels: i === 0});
    lay.annotations.push(panelTitle('K = ' + K, dom[i]));
  });
  lay.toppad = 24;
  plot(el, tr, lay);
}"""

DRAW_VAR = r"""(el, d, c) => {
  const dec = c.dec, s0 = st(dec), sh = M.shade[s0.fam], F = DATA.fin;
  sub(el, s0.label + ' · variant minus base, mean cosine over H');
  const dd = (a, b) => SITES.map(s => F[a][s][dec] && F[b][s][dec] ? F[a][s][dec].cos[0] - F[b][s][dec].cos[0] : null);
  const tr = [
    sline('joint: 16 selected by r<sub>k</sub> − first 16', dd('joint-llr16|16', 'joint|16'), {color: sh[5], symbol:'circle'}),
    sline('site: 16 selected by r<sub>k</sub> − first 16', dd('site-llr16|16', 'site|16'), {color: sh[1], symbol:'diamond'}),
    sline('site, K = 64: λ-scaled (site-t1) − unit variance', dd('site-t1|64', 'site|64'), {color: sh[3], symbol:'triangle-up'}),
  ];
  plot(el, tr, {height: 330, shapes:[zeroLine()], xaxis: siteAxis(), yaxis: merge(AXIS, {range: M.rng.var, title:{text:'Δ cosine'}, zeroline:false})});
}"""

DRAW_PR2 = r"""(el, d, c) => {
  sub(el, 'cut-off lift (GCV) on joint coordinates · mean cosine over H');
  const dom = facetAxes(2, 0.06), tr = [], lay = {height: 380, annotations: []}, sh = ['#c2d97a', '#8ca252', '#5b6e2c'];
  [16, 64].forEach((K, i) => { const ax = i ? String(i + 1) : '';
    tr.push(sline('Gaussian second kernel', SITES.map(s => DATA.fin['joint|' + K][s].lh.cos), merge(st('lh'), {width: 2.6}),
      {xaxis:'x' + ax, yaxis:'y' + ax, legendgroup:'g', showlegend: i === 0}));
    [30, 100, 400].forEach((k2, j) => tr.push(sline('participation ratio k2 = ' + k2, SITES.map(s => DATA.fin['joint-pr2_' + k2 + '|' + K][s].lh.cos),
      {color: sh[j], symbol: ['triangle-up', 'square', 'x'][j], width: 1.6, msize: 6}, {xaxis:'x' + ax, yaxis:'y' + ax, legendgroup:'k' + k2, showlegend: i === 0})));
    lay['xaxis' + ax] = siteAxis({domain: dom[i], anchor:'y' + ax, tickangle:-45});
    lay['yaxis' + ax] = merge(AXIS, {anchor:'x' + ax, range:[0, 1.02], title:{text: i ? '' : 'cosine (mean)'}, showticklabels: i === 0});
    lay.annotations.push(panelTitle('K = ' + K, dom[i]));
  });
  lay.toppad = 24;
  plot(el, tr, lay);
}"""

# ---- tab 4: premise
DRAW_LLR = r"""(el, d, c) => {
  const key = c.pc, ys = SITES.slice(), z = ys.map(s => DATA.llr[s][key]), ks = z[0].map((_, i) => i + 1);
  sub(el, key + ' coordinates · 1,000 anchors against the 8,000-image probe');
  const sx = [], sy = [];
  ys.forEach(s => DATA.llr[s]['sel_' + key].forEach(i => { sx.push(i + 1); sy.push(s); }));
  const tr = [{type:'heatmap', x: ks, y: ys, z, zmin: 0, zmax: 1, colorscale: M.heat,
      colorbar:{title:{text:'r<sub>k</sub>', side:'top'}, thickness: 12, len: 0.85, x: 1.07, y: 0.45, tickvals:[0, 0.25, 0.5, 0.75, 1]},
      hovertemplate:'%{y} · coordinate %{x}: r<sub>k</sub> = %{z:.3f}<extra></extra>', xgap: 1, ygap: 1},
    {type:'scatter', mode:'markers', x: sx, y: sy, name:'kept by the 16-coordinate selection', marker:{size: 5, color:'#fff', line:{color:'#111', width: 1}},
      hovertemplate:'%{y} · coordinate %{x} kept<extra></extra>'}];
  const ann = ys.map(s => ({xref:'paper', yref:'y', x: 1.005, y: s, xanchor:'left', showarrow:false, font:{size:11.5},
    text: String(DATA.llr[s][key].filter(r => r > 0.5).length)}));
  ann.push({xref:'paper', yref:'paper', x: 1.005, y: 1.0, xanchor:'left', yanchor:'bottom', showarrow:false, font:{size:11, color:'#6b665e'}, text:'# r<sub>k</sub> > 0.5'});
  plot(el, tr, {height: 440, margin:{r: 110}, annotations: ann, showlegend: true,
    xaxis: merge(AXIS, {title:{text:'coordinate k (eigenvector index, 1 = leading)'}, range:[0.5, ks.length + 0.5], dtick: 4, ticks:'', showgrid:false}),
    yaxis: merge(AXIS, {type:'category', autorange:'reversed', title:{text:'site'}, showgrid:false, ticks:''})});
}"""

DRAW_LATER = r"""(el, d, c) => {
  const pc = c.pc, arm = c.arm, stat = c.stat, fam = arm === 'kr' ? 'ridge' : arm === 'lh' ? 'lift' : 'linear', sh = M.shade[fam];
  const armName = {kr:'ridge (GCV)', lh:'cut-off lift (GCV)', linear:'linear'}[arm];
  sub(el, pc + ' coordinates · predictor: ' + armName + ' · ' + (stat === 'r2' ? 'R²' : 'squared correlation ρ²') + ' on H · median and quartiles over target coordinates');
  const cells = [[pc + '|4', 'first 4 → coordinates 5..256', 0, 'circle'], [pc + '|16', 'first 16 → 17..256', 2, 'circle'],
                 [pc + '|64', 'first 64 → 65..256', 5, 'circle'], [pc + '-llr16|16', '16 selected by r<sub>k</sub> → the other 48 of the first 64', 3, 'circle-open']];
  const LO = -1, tr = [], clipped = [];
  for (const [cell, lab, si, sym] of cells) { const L = DATA.later[cell]; if (!L) continue;
    const q = SITES.map(s => L[s] && L[s][arm] ? L[s][arm][stat] : null);
    const y = q.map((v, i) => { if (!v) return null; if (v[1] < LO) clipped.push(SITES[i] + ' (' + lab.split(' →')[0] + ': ' + v[1].toPrecision(3) + ')'); return Math.max(v[1], LO); });
    const up = q.map((v, i) => v ? Math.max(v[2], LO) - y[i] : 0), dn = q.map((v, i) => v ? y[i] - Math.max(v[0], LO) : 0);
    tr.push({type:'scatter', mode:'lines+markers', name: 'K: ' + lab, x: SITES, y, line:{color: sh[si], width: sym.endsWith('open') ? 1.6 : 2},
      marker:{size: 7, symbol: sym, color: sh[si], line:{color: sh[si], width: 1.5}},
      error_y:{type:'data', symmetric:false, array: up, arrayminus: dn, thickness: 1, width: 3, color: sh[si]},
      hovertemplate: lab + ' · %{x}: median %{y:.3f}<extra></extra>'}); }
  const ann = clipped.length ? [{xref:'paper', yref:'paper', x: 0, y: 0.0, xanchor:'left', yanchor:'bottom', showarrow:false, align:'left',
    font:{size:11, color:'#8a4a2a'}, bgcolor:'rgba(255,255,255,0.85)', text: 'median below −1, drawn at −1: ' + clipped.join(', ')}] : [];
  plot(el, tr, {height: 400, annotations: ann, shapes:[zeroLine()], xaxis: siteAxis(),
    yaxis: merge(AXIS, {range:[LO - 0.05, 1.02], title:{text:(stat === 'r2' ? 'held-out R²' : 'held-out ρ²') + ' (median over targets)'}, zeroline:false})});
}"""

# ---- tab 5: round trips
RT_DECS = ['linear', 'lh', 'lh_fsplit', 'kr', 'krr', 'nn1', 'nn15', 'nn1+krr']
DRAW_RTJ = r"""(el, d, c) => {
  const F = DATA.fin['joint|' + c.K], tr = [];
  sub(el, 'joint coordinates · K = ' + c.K + ' · median over the first 256 held-out images');
  for (const dec of M.rtdecs) { const s = st(dec); tr.push(sline(s.label, SITES.map(x => F[x][dec] ? F[x][dec].rtj : null), merge(s, {width: 1.8}))); }
  const lay = {height: 420, xaxis: siteAxis(), yaxis: fixLog(merge(AXIS, {range: M.rng.rt, title:{text:'joint re-embedding miss (median, log)'}}))};
  plot(el, tr, addRefs(lay, [refLine(1, 'returning zero = 1', 'paper', 'y', true), refLine(2, 'another image = 2', 'paper', 'y', true)]));
}"""

DRAW_RTS = r"""(el, d, c) => {
  const F = DATA.fin['site|' + c.K], dom = facetAxes(2, 0.06), tr = [], lay = {height: 440, annotations: [], shapes: []};
  sub(el, 'site coordinates · K = ' + c.K + ' · median over 2,000 held-out images');
  [['rtq', 'with 1/λ (the paper\'s Nyström restriction)'], ['rto', 'without 1/λ']].forEach(([k, lab], i) => { const ax = i ? String(i + 1) : '';
    for (const dec of M.rtdecs) { const s = st(dec);
      tr.push(sline(s.label, SITES.map(x => F[x][dec] ? F[x][dec][k] : null), merge(s, {width: 1.8}),
        {xaxis:'x' + ax, yaxis:'y' + ax, legendgroup: dec, showlegend: i === 0})); }
    lay['xaxis' + ax] = siteAxis({domain: dom[i], anchor:'y' + ax, tickangle:-45});
    lay['yaxis' + ax] = fixLog(merge(AXIS, {anchor:'x' + ax, range: M.rng.rt, title:{text: i ? '' : 'restriction miss (median, log)'}, showticklabels: true}));
    lay.annotations.push(panelTitle(lab, dom[i]));
    for (const [y, t] of [[1, 'zero = 1'], [2, 'another image = 2']]) {
      lay.shapes.push({type:'line', xref:'x' + ax + ' domain', yref:'y' + ax, x0:0, x1:1, y0:y, y1:y, line:{color:'#8a8a8a', dash:'dot', width:1.2}});
      lay.annotations.push({xref:'x' + ax + ' domain', yref:'y' + ax, x:1, y: Math.log10(y), xanchor:'right', yanchor: y < 1.5 ? 'top' : 'bottom', showarrow:false, text:t, font:{size:11, color:'#6b6b6b'}}); }
  });
  lay.toppad = 24;
  plot(el, tr, lay);
}"""

CSS = """
p.empty { color:#8a5a1f; background:#fbf2e6; border:1px solid #ecd3ae; border-radius:6px; padding:8px 12px; font-size:13.5px; }
.eq-title { font-weight:700; font-size:14.5px; margin:16px 0 0; }
.banner .facts { font-size:13px; color:var(--dim); margin:2px 0 6px; }
.banner .defs { font-size:13px; line-height:1.55; margin:6px 0 4px; padding-top:6px; border-top:1px solid var(--line); max-width:none; }
.banner .key { display:flex; flex-wrap:wrap; gap:4px 18px; align-items:center; font-size:12.5px; margin:4px 0 4px; }
.banner .key .klabel { font-weight:650; color:var(--dim); margin-right:2px; }
.banner .key span.it { display:inline-flex; align-items:center; gap:5px; }
.step { border:1px solid var(--border); border-radius:8px; background:white; padding:8px 14px; margin:8px 0; }
.step h4 { margin:0 0 4px; font-size:14px; }
.step p { margin:0; max-width:none; }
ul.answer { padding-left:20px; } ul.answer > li { max-width:100ch; margin:0 0 6px; }
ol.findings { padding-left:22px; } ol.findings > li { max-width:100ch; margin:0 0 10px; }
ol.findings > li > b:first-child { color:#2b2a27; }
ol.findings a.goto { font-size:12.5px; white-space:nowrap; }
table.ftab { width:auto; } table.doc td code { white-space:nowrap; }
pre.repro { font:12.5px ui-monospace,SFMono-Regular,Menlo,monospace; background:var(--soft); border:1px solid var(--line);
  border-radius:8px; padding:10px 12px; overflow-x:auto; }
main { overflow-x:hidden; }
.status { white-space:normal; display:inline-block; }
"""


# ------------------------------------------------------------------ page parts --
def arch_banner():
    shapes = None
    try:
        tree = ast.parse((HERE / 'modal_invert.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'SHAPES':
                shapes = ast.literal_eval(node.value)
    except Exception as e:  # noqa: BLE001
        print(f'  note: SHAPES not read from modal_invert.py ({e}); using the built-in table')
    if not shapes:
        shapes = {'pixels': (3, 32, 32), 'stem': (16, 32, 32), **{f'b{i}': ((16, 32, 32), (32, 16, 16), (64, 8, 8))[(i - 1) // 3] for i in range(1, 10)},
                  'pool': (64,), 'logits': (10,)}
    shp = lambda s: f'{shapes[s][0]}×{shapes[s][1]}²' if len(shapes[s]) == 3 else f'{shapes[s][0]}'  # noqa: E731
    kind = {16: 'mlp', 32: 'attn', 64: 'norm'}
    blocks = [('input', shp('pixels'), 'in'), ('conv·BN·ReLU', shp('stem'), 'act')]
    blocks += [(f'block {i}', shp(f'b{i}'), kind.get(shapes[f'b{i}'][0], 'other')) for i in range(1, 10)]
    blocks += [('avg pool', shp('pool'), 'other'), ('fc', shp('logits'), 'out')]
    cuts = {1: 'stem', **{i + 1: f'b{i}' for i in range(1, 10)}, 11: 'pool', 12: 'logits'}
    svg = arch_svg(blocks, skips=[(i, i + 1) for i in range(1, 10)], cuts=cuts, caption=T['banner']['caption'])
    import re
    svg = re.sub(r'viewBox="0 0 (\d+) (\d+)" width="(\d+)"',
                 lambda m: f'viewBox="0 0 {int(m.group(1)) + 30} {m.group(2)}" width="{int(m.group(3)) + 30}"', svg)
    return f'<div class="facts">{T["banner"]["facts"]}</div><div class="archs">{svg}</div>' + key_html() + \
        f'<p class="defs">{T["banner"]["defs"]}</p>'


def swatch(color, dash='solid', symbol=None):
    da = {'solid': '', 'dash': 'stroke-dasharray="6 4"', 'dot': 'stroke-dasharray="2 3"', 'dashdot': 'stroke-dasharray="7 3 2 3"'}[dash]
    return f'<svg width="30" height="10"><line x1="1" y1="5" x2="29" y2="5" stroke="{color}" stroke-width="2.4" {da}/></svg>'


def key_html():
    fam = LG['family']
    hue = ''.join(f'<span class="it">{swatch(FAM[k])}{fam[k]}</span>' for k in ('lift', 'ridge', 'linear', 'nn', 'ref'))
    dash = ''.join(f'<span class="it">{swatch("#555", k)}{v}</span>' for k, v in LG['dash'].items())
    return (f'<div class="key"><span class="klabel">{T["banner"]["key_hue"]}:</span>{hue}</div>'
            f'<div class="key"><span class="klabel">{T["banner"]["key_dash"]}:</span>{dash}'
            '<span class="it" style="color:var(--dim)">· shade within a hue = width h, K, or coordinate system (named in each legend)</span></div>')


def fig(tab, fid, draw, data='@meta', **kw):
    f = T['fig'][fid]
    tab.figure(fid, f['title'], data=data, draw=draw, caption=f['caption'], subtitle=' ', **kw)


def eq(tab, block):
    tab.prose(f'<div class="eq-title">{block["eq_title"]}</div>')
    tab.equation(block['eq'], block.get('eq_where', ''))


def paras(xs):
    return ''.join(f'<p>{x}</p>' for x in xs)


def controls_table():
    rows = []
    for s in SITES:
        c = R['checks'][s]
        e = lambda v: f'{v:.1e}'  # noqa: E731
        rows.append([s, e(max(abs(v) for v in c['krr_parity'].values())), e(abs(c['cos_vec_vs_stored'])),
                     f'{e(c["onestep_max_abs_diff"])} ({e(c["onestep_scale"])})', f'{c["patch_true_agree"]:.3f}',
                     e(c['patch_true_kl']), e(c['joint_true_miss']),
                     f'{R["gpu"][s].replace("NVIDIA ", "")} · {R["seconds"][s]:.0f}'])
    return rows


# ------------------------------------------------------------------ build --
def main():
    fin, func, surf = fin_payload(), func_payload(), surf_payload()
    meta = dict(sites=SITES, sty=STY, fam=FAM, shade=SHADE, kshade=KSHADE, coord=COORD_STY, Ks=KS, hs=R['hs'],
                ridges=R['ridges'], rtdecs=RT_DECS, rng=ranges(fin, func, surf),
                heat=[[0, '#fbf7ef'], [0.25, '#efdcb2'], [0.5, '#d9a05e'], [0.75, '#a9533a'], [1, '#4f1726']])
    built = datetime.fromtimestamp(SRC.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    pg = T['page']
    nb = Notebook(pg['title'], pg['subtitle'], status=pg['status'], kicker=pg['kicker'],
                  footer=pg['footer'].replace('{built}', built).replace('{today}', datetime.now().strftime('%Y-%m-%d %H:%M')))
    nb.style(CSS)
    nb.banner(arch_banner(), label=T['banner']['label'])
    nb.dataset('meta', meta)
    nb.dataset('fin', fin)
    nb.dataset('sel', sel_payload())
    nb.dataset('func', func)
    nb.dataset('surf', surf)
    nb.dataset('later', later_payload())
    nb.dataset('llr', llr_payload())
    nb.dataset('ref', {s: r5(dict(cos=R['ref'][s]['cos'][0], agree=R['ref'][s]['agree'][0])) for s in SITES})
    nb.dataset('prior', {s: r5(R['prior'][s]) for s in SITES})
    nb.script(JS)
    TB, C = T['tabs'], T['controls']
    site_sel = lambda name, default='b4': Select(name, C['site'], SITES, default=default)  # noqa: E731

    # ---------------------------------------------------------------- 0 overview
    ov = nb.tab('overview', TB['overview'])
    o = T['overview']
    ov.h(o['h_question']).prose(paras(o['question']))
    ov.h(o['h_answer']).prose('<ul class="answer">' + ''.join(f'<li>{a}</li>' for a in o['answer']) + '</ul>')
    ov.h(o['h_method']).prose(f'<p>{o["method_intro"]}</p>')
    for s in o['steps']:
        body = f'<p>{s["html"]}</p>'
        if s.get('tex'):
            body += f'<div class="eq">$$ {s["tex"]} $$</div><div class="where">{s.get("where", "")}</div>'
        ov.prose(f'<div class="step"><h4>{s["name"]}</h4>{body}</div>')
    ov.h(o['h_findings'])
    tabno = {t: i for i, t in enumerate(TB)}
    items = ''.join(f'<li><b>{f["head"]}</b> {f["body"]} <a class="goto" href="#tab={f["tab"]}">→ tab {tabno[f["tab"]]}</a>'
                    f'{f.get("table", "")}</li>' for f in o['findings'])
    ov.prose(f'<ol class="findings">{items}</ol>')
    ov.h(o['h_map']).table(o['map_header'], o['map'])
    ov.h(o['h_controls']).prose(f'<p>{o["controls_intro"]}</p>')
    ov.verdict(o['controls_verdict'], 'pass', label='Controls passed')
    ov.table(o['controls_header'], controls_table(), numeric=set(range(1, 7)))

    # ---------------------------------------------------------------- 1 matched comparison
    mt = nb.tab('matched', TB['matched'])
    m = T['matched']
    mt.prose(f'<p>{m["intro"]}</p>')
    eq(mt, m)
    mt.prose(f'<div class="eq-title">{m["gcv_title"]}</div>')
    mt.equation(m['gcv'], m['gcv_where'])
    mt.controls([Radio('coords', C['coords'], ['joint', 'source', 'site'], default='joint'),
                 Radio('K', C['K'], KS, default=64)])
    fig(mt, 'matched_cmp', DRAW_CMP, '@fin')
    fig(mt, 'matched_gap', DRAW_GAP, '@fin')
    fig(mt, 'matched_rule', DRAW_RULE, '@sel')
    fig(mt, 'matched_surface', DRAW_SURF, '@surf',
        controls=[Select('scell', C['cell'], list(R['surface']), default='joint|64',
                         labels={k: k.replace('|', ' · K = ') for k in R['surface']}), site_sel('ssite')])

    # ---------------------------------------------------------------- 2 reconstruction against function
    ft = nb.tab('function', TB['function'])
    f_ = T['function']
    ft.prose(f'<p>{f_["intro"]}</p>')
    ft.prose(f'<div class="eq-title">{m["eq_title"]}</div>')
    ft.equation(m['eq'], '')
    eq(ft, f_)
    ft.controls([Select('cell', C['cell'], list(R['func']), default='joint|64', labels={k: k.replace('|', ' · K = ') for k in R['func']}),
                 site_sel('site'), Radio('path', C['path'], ['cut', 'ridge'], default='cut', labels={'cut': 'spectral cut-off', 'ridge': 'ridge'})])
    fig(ft, 'path', DRAW_PATH, '@func')
    fig(ft, 'frontier', DRAW_FRONT, '@func')

    # ---------------------------------------------------------------- 3 coordinate systems
    ct = nb.tab('coords', TB['coords'])
    ct.prose(f'<p>{T["coords"]["intro"]}</p>')
    ct.controls([Radio('dec', C['decoder'], ['kr', 'lh', 'linear'], default='kr',
                       labels={'kr': 'kernel ridge (GCV)', 'lh': 'cut-off lift (GCV)', 'linear': 'linear'})])
    fig(ct, 'coords', DRAW_COORDS, '@fin')
    fig(ct, 'coords_var', DRAW_VAR, '@fin')
    fig(ct, 'coords_pr2', DRAW_PR2, '@fin')

    # ---------------------------------------------------------------- 4 premise
    pt = nb.tab('premise', TB['premise'])
    pt.prose(f'<p>{T["premise"]["intro"]}</p>')
    eq(pt, T['premise'])
    pt.controls([Radio('pc', C['coords'], ['site', 'joint'], default='site')])
    fig(pt, 'premise_llr', DRAW_LLR, '@llr')
    fig(pt, 'premise_later', DRAW_LATER, '@later',
        controls=[Radio('arm', C['arm'], ['kr', 'lh', 'linear'], default='kr', labels={'kr': 'ridge', 'lh': 'cut-off lift', 'linear': 'linear'}),
                  Radio('stat', C['stat'], ['r2', 'rho2'], default='r2', labels={'r2': 'R²', 'rho2': 'ρ²'})])

    # ---------------------------------------------------------------- 5 round trips
    rt = nb.tab('roundtrip', TB['roundtrip'])
    rt.prose(f'<p>{T["roundtrip"]["intro"]}</p>')
    eq(rt, T['roundtrip'])
    rt.controls([Radio('K', C['K'], [4, 16, 64], default=64)])
    fig(rt, 'rt_joint', DRAW_RTJ, '@fin')
    fig(rt, 'rt_site', DRAW_RTS, '@fin')

    # ---------------------------------------------------------------- 6 what next
    nx = nb.tab('next', TB['next'])
    n = T['next']
    nx.h(n['h_use']).prose('<ul>' + ''.join(f'<li>{x}</li>' for x in n['use']) + '</ul>')
    nx.h(n['h_related']).prose(f'<p>{n["tested"]}</p><p>{n["proposed_intro"]}</p><ol>' + ''.join(f'<li>{x}</li>' for x in n['proposed']) + '</ol>')
    nx.h(n['h_decisions'])
    nx.methodology(n['decisions'], intro=f'<p>{n["decisions_intro"]}</p>', title='')
    nx.h(n['h_limits']).prose('<ul>' + ''.join(f'<li>{x}</li>' for x in n['limits']) + '</ul>')
    nx.details(n['h_changed'], '<ul>' + ''.join(f'<li>{x}</li>' for x in n['changed']) + '</ul>')
    nx.h(n['h_repro']).prose(f'<p>{n["spend"]}</p><pre class="repro">{n["repro"]}</pre>')

    nb.write(OUT)


if __name__ == '__main__':
    main()
