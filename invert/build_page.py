"""Results page for the inversion experiment: one self-contained HTML file.

    uv run --with pillow python invert/build_page.py            # -> invert/pages/invert.html
    bash invert/verify_page.sh <shot-dir>                      # one screenshot per tab + console errors

Inputs: data/invert/results.json (schema in invert/collect.py), the image grids in data/invert/figs/
(invert/make_figs.py), and invert/page_text.json (all prose).  Everything shown is driven by what is present in
those files at build time: probe tags, K values, sites and edit methods come from the data; missing keys are
skipped.  Numbers quoted in prose are placeholders in page_text.json, filled here:

  {name}                         a named number from NAMED below (listed with current values in
                                 invert/pages/text_keys.json after each build)
  {=dec:<tag>:<K>:<dec>:<metric>:<site>[;fmt]}   decode summary, target H.  <tag> = main | P2 | P8 | full tag
  {=ood:<tag>:<set>:<K>:<dec>:<metric>:<site>[;fmt]}
  {=rt:<tag>:<K>:<key>:<site>[;fmt]}             round-trip / edit summary (keys of stage_rt.py)
  {=ker:<kernel>:<field>:<site>[;fmt]}           field = hubmax | hubskew | rawhub | stab<k>
  {=refit:<kernel>:<K>:<probe|held|onestep>:<site>[;fmt]}
  {=adv:<attack>:<field>:<site>[;fmt]}           first adv entry for the main tag
<site> may be a site name or an aggregate over the sites present: min | max | median | argmin | argmax
(argmin/argmax return the site name), optionally over a range of sites: min@b1-b8.  Also
  {=ker:<kernel>:perp:<site>}  final walk perplexity (effective width);  {=ml:<tag>:max|argmax|count:<thr>:<m>}  fmt: g (default, 3 significant digits) | pct | 2f | 1f | int.
"""
from __future__ import annotations

import ast
import base64
import io
import json
import math
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from notebook import Notebook, Radio, Select, add_figure, arch_svg, clean, render_figure, table_html  # noqa: E402

ROOT = HERE.parent
DATA = ROOT / 'data/invert'
OUT = HERE / 'pages' / 'invert.html'
TEXT = json.loads((HERE / 'page_text.json').read_text())
R = json.loads((DATA / 'results.json').read_text())
SITES = R['sites']
WARN: list[str] = []

# ------------------------------------------------------------------ data helpers --


def probe_of(tag):            # 'P8__ea400__L12' -> 'P8'
    return tag.split('__')[0]


def kernel_of(tag):
    return tag.split('__')[1]


def probe_n(p):
    m = re.match(r'P(\d+)', p)
    return int(m.group(1)) * 1000 if m else None


FAMILIES = ['pr', 'ea', 'ds', 'knn']   # selector order: participation ratio (headline) first


def kernel_family(k):
    m = re.match(r'([a-z]+)(\d+)$', k)
    return m.group(1) if m else k


def kernel_param(k):
    m = re.match(r'([a-z]+)(\d+)$', k)
    return int(m.group(2)) if m else 0


def kernel_label(k):
    m = re.match(r'([a-z]+)(\d+)$', k)
    if not m:
        return k
    kind, v = m.groups()
    return {'pr': f'participation ratio k={v}', 'ea': f'Shannon perplexity ξ={v}', 'ds': f'global width, perplexity {v}',
            'knn': f'k-NN rule k={v}'}.get(kind, k)


def tag_label(tag):
    p = probe_of(tag)
    n = probe_n(p)
    return f'{p} ({n:,} images) · {kernel_label(kernel_of(tag))}' if n else tag


def tag_order(t):
    k = kernel_of(t) if '__' in t else t
    fam = kernel_family(k)
    return (FAMILIES.index(fam) if fam in FAMILIES else 99, -(probe_n(probe_of(t)) or 0), -kernel_param(k), t)


DEC_TAGS = sorted([t for t, v in R['decode'].items() if any(s in v for s in SITES)], key=tag_order)
RT_TAGS = sorted([t for t, v in R['rt'].items() if v], key=tag_order)


def n_sites_dec(tag):
    return sum(1 for s in SITES if s in R['decode'].get(tag, {}))


def n_sites_rt(tag):
    return len({u.split('__K')[0] for u in R['rt'].get(tag, {})})


def complete(tag):
    return n_sites_dec(tag) == len(SITES) and n_sites_rt(tag) == len(SITES)


def pick_default(pool, n_sites):
    """The owner's rule: the first preferred tag (page_text.json config.default_tags) that is present in `pool`;
    the first entry only when complete for all sites in both decode and rt.  Else the most complete tag, preferring
    the participation-ratio family and then the larger probe."""
    if not pool:
        return None
    pref = TEXT.get('config', {}).get('default_tags', [])
    for i, t in enumerate(pref):
        if t in pool and (i > 0 or complete(t)):
            return t
    return max(pool, key=lambda t: (n_sites(t), kernel_family(kernel_of(t)) == 'pr', probe_n(probe_of(t)) or 0))


def main_tag_for(prefix):
    """Main tag of one probe (dec_vsK and the P2/P8 placeholders): participation-ratio family first, then complete,
    then most decoded sites, then the larger kernel parameter."""
    cands = [t for t in DEC_TAGS if probe_of(t) == prefix]
    if not cands:
        return None
    return max(cands, key=lambda t: (kernel_family(kernel_of(t)) == 'pr', n_sites_dec(t) == len(SITES), n_sites_dec(t),
                                     kernel_param(kernel_of(t))))


def t_short(t):
    return f'{probe_of(t)} · {kernel_of(t)}'


KERNEL_NAMES = sorted({n for s in SITES for n, v in R['kernel'].get(s, {}).items() if isinstance(v, dict) and 'hub' in v},
                      key=lambda n: tag_order('X__' + n))
PROBES = sorted({probe_of(t) for t in DEC_TAGS}, key=lambda p: probe_n(p) or 0)
MAIN = pick_default(DEC_TAGS, n_sites_dec)
MAIN_BY_PROBE = {p: main_tag_for(p) for p in PROBES}
RT_MAIN = pick_default(RT_TAGS, n_sites_rt)


def dec_val(tag, site, key):
    v = R['decode'].get(tag, {}).get(site, {}).get(key)
    return v if isinstance(v, list) else None


def dec_Ks(tag=None):
    ks = set()
    for t in ([tag] if tag else DEC_TAGS):
        for s in SITES:
            for k in R['decode'].get(t, {}).get(s, {}):
                p = k.split('|')
                if len(p) == 4 and p[0] == 'H' and p[1] != '0':
                    ks.add(int(p[1]))
    return sorted(ks)


def rt_val(tag, K, site, key):
    v = R['rt'].get(tag, {}).get(f'{site}__K{K}', {}).get(key)
    return v if isinstance(v, (int, float)) else None


def rt_Ks():
    return sorted({int(u.split('__K')[1]) for t in RT_TAGS for u in R['rt'][t]})


def r5(v):
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return None
    return float(f'{v:.5g}')


# ------------------------------------------------------------------ payloads --

DECODERS = ['linear', 'nn', 'lp', 'lpz', 'gh', 'krr', 'pca', 'step', 'mlp', 'mean', 'true']
METRICS = ['cos_raw', 'cap_raw', 'kl', 'kl_on', 'agree', 'agree_on', 'cap_raw_med', 'kl_med', 'kl_on_med']
OOD_SETS = ['H', 'c100', 'svhn', 'noise05', 'noise10', 'noise20', 'pgd2', 'pgd8']


def decode_payload():
    """dec[tag][K][decoder][metric] = per-site [mean, se] or null (target H); K '0' holds mean/step/true."""
    out = {}
    for tag in DEC_TAGS:
        d = out[tag] = {}
        for s_i, s in enumerate(SITES):
            for key, v in R['decode'][tag].get(s, {}).items():
                p = key.split('|')
                if len(p) != 4 or p[0] != 'H' or p[3] not in METRICS or not isinstance(v, list):
                    continue
                arr = d.setdefault(p[1], {}).setdefault(p[2], {}).setdefault(p[3], [None] * len(SITES))
                arr[s_i] = [r5(v[0]), r5(v[1])]
    return out


def ood_payload(decs=('krr', 'mean', 'true')):
    out = {}
    for tag in DEC_TAGS:
        d = out[tag] = {}
        for s_i, s in enumerate(SITES):
            for key, v in R['decode'][tag].get(s, {}).items():
                p = key.split('|')
                if len(p) != 4 or p[2] not in decs or p[3] not in ('cos_raw', 'kl') or not isinstance(v, list):
                    continue
                arr = d.setdefault(p[0], {}).setdefault(p[1], {}).setdefault(p[2], {}).setdefault(p[3], [None] * len(SITES))
                arr[s_i] = [r5(v[0]), r5(v[1])]
    return out


def rt_payload():
    """rt[tag][K][key] = per-site value or null."""
    out = {}
    for tag in RT_TAGS:
        for unit, vals in R['rt'][tag].items():
            site, K = unit.split('__K')
            if site not in SITES:
                continue
            d = out.setdefault(tag, {}).setdefault(K, {})
            for k, v in vals.items():
                if isinstance(v, (int, float)) and (k.startswith('E')):
                    d.setdefault(k, [None] * len(SITES))[SITES.index(site)] = r5(v)
    return out


def kernel_payload():
    ker = R['kernel']
    names = []
    for s in SITES:
        for n, v in ker.get(s, {}).items():
            if isinstance(v, dict) and 'hub' in v and n not in names:
                names.append(n)
    out = {'names': names, 'hubmax': {}, 'hubskew': {}, 'raw': {}, 'stab': {}, 'stab_k': []}
    ks = set()
    for n in names:
        out['hubmax'][n] = [r5(ker.get(s, {}).get(n, {}).get('hub', {}).get('max')) for s in SITES]
        out['hubskew'][n] = [r5(ker.get(s, {}).get(n, {}).get('hub', {}).get('skew')) for s in SITES]
        out['stab'][n] = {s: {k: r5(v) for k, v in ker.get(s, {}).get(n, {}).get('stab', {}).items()} for s in SITES}
        for s in SITES:
            ks |= {int(k) for k in ker.get(s, {}).get(n, {}).get('stab', {})}
    out['raw'] = {'max': [r5(ker.get(s, {}).get('raw_cos_hub', {}).get('max')) for s in SITES],
                  'skew': [r5(ker.get(s, {}).get('raw_cos_hub', {}).get('skew')) for s in SITES]}
    out['stab_k'] = sorted(ks)
    out['names'] = sorted(names, key=lambda n: tag_order('X__' + n))
    return out


def refit_payload():
    out = {}
    for key, v in R.get('refit', {}).items():
        kn, K = key.split('|')
        out.setdefault(K, {})[kn] = {q: [r5(x) for x in v[q]] for q in ('probe', 'held', 'onestep') if q in v}
    return {K: {kn: d[kn] for kn in sorted(d, key=lambda n: tag_order('X__' + n))} for K, d in out.items()}


def adv_payload():
    out = {}
    for key, v in R.get('adv', {}).items():
        out[key] = {att: {f: [None] * len(SITES) for f in rows[0] if f != 'site'} for att, rows in v.items() if rows}
        for att, rows in v.items():
            for row in rows:
                if row.get('site') in SITES:
                    for f, x in row.items():
                        if f != 'site':
                            out[key][att][f][SITES.index(row['site'])] = r5(x)
    return out


# ------------------------------------------------------------------ image grids --

FIG_RE = re.compile(r'^(.+__L\d+)_(.+)_K(\d+)\.png$')
PX_SITES = ['pixels', 'stem', 'b3', 'b6', 'b9', 'pool']
TAUS = [-0.5 + 0.25 * i for i in range(9)]
WEBP_QUALITY = 95


def grid_heads(name):
    """Column headers of make_figs.py's grids, by grid name (mirrors that script)."""
    if name in ('depth', 'depthsrc', 'c100', 'svhn', 'noise20', 'c100src', 'svhnsrc', 'noise20src'):
        return ['original'] + PX_SITES
    m = re.match(r'(pgd\d+)(src)?$', name)
    if m:
        ss = ('pixels', 'b3', 'b6', 'b9') + (('pool',) if m.group(2) else ())
        return ['clean', 'attacked'] + [h for s in ss for h in (f'{s} · clean', f'{s} · att.')]
    if re.match(r'path(src)?_', name):
        return ['image a'] + [f'τ={t:+.2f}' for t in TAUS] + ['image b']
    return None


def retile(path, heads):
    """Crop make_figs' header strip and undo its 3x nearest upscaling: one 32 px tile per cell, 1 px gaps."""
    SC, PAD, HEAD = 3, 2, 14
    cell = 32 * SC
    im = Image.open(path).convert('RGB')
    W, H = im.size
    nc, nr = (W - PAD) / (cell + PAD), (H - HEAD - PAD) / (cell + PAD)
    if nc != int(nc) or nr != int(nr):
        WARN.append(f'{path.name}: unexpected geometry {W}x{H}; embedded as is')
        return dict(src=_uri(path.read_bytes()), cols=None, w=W, scale=1)
    nc, nr = int(nc), int(nr)
    if heads is None or len(heads) != nc:
        WARN.append(f'{path.name}: {nc} columns, header list has {len(heads) if heads else 0}; generic headers used')
        heads = [f'column {j + 1}' for j in range(nc)]
    out = Image.new('RGB', (nc * 33 + 1, nr * 33 + 1), (255, 255, 255))
    for j in range(nc):
        for i in range(nr):
            x, y = PAD + j * (cell + PAD), HEAD + PAD + i * (cell + PAD)
            out.paste(im.crop((x, y, x + cell, y + cell)).resize((32, 32), Image.NEAREST), (1 + 33 * j, 1 + 33 * i))
    buf = io.BytesIO()
    out.save(buf, 'WEBP', quality=WEBP_QUALITY, method=6)   # lossy WebP: ~1/4 of the PNG; see verify screenshots
    return dict(src='data:image/webp;base64,' + base64.b64encode(buf.getvalue()).decode('ascii'), cols=heads, w=out.size[0], rows=nr, scale=3)


def js(x):
    return json.dumps(x, ensure_ascii=False)


def _uri(b):
    return 'data:image/png;base64,' + base64.b64encode(b).decode('ascii')


def image_payload():
    """img[figtag][name][K] = {src, cols, w, rows, scale}."""
    out = {}
    for p in sorted((DATA / 'figs').glob('*.png')):
        m = FIG_RE.match(p.name)
        if not m:
            continue
        tag, name, K = m.groups()
        out.setdefault(tag, {}).setdefault(name, {})[K] = retile(p, grid_heads(name))
    return out


# ------------------------------------------------------------------ prose numbers --

AGG = ('min', 'max', 'median', 'argmin', 'argmax')


def _resolve_tag(t, pool):
    if t == 'main':
        return MAIN if pool is DEC_TAGS else RT_MAIN
    if t in PROBES:
        tg = MAIN_BY_PROBE.get(t)
        if pool is RT_TAGS:
            c = [x for x in RT_TAGS if probe_of(x) == t]
            tg = tg if tg in c else (c[0] if c else None)
        return tg
    return t


def _agg(series, site):
    """series: {site: value}."""
    if site in SITES:
        return series.get(site)
    if '@' in site:                     # e.g. min@b1-b8: aggregate over a contiguous range of sites
        site, rng_ = site.split('@')
        a_, b_ = rng_.split('-')
        keep = SITES[SITES.index(a_):SITES.index(b_) + 1]
        series = {k: v for k, v in series.items() if k in keep}
    vals = {s: v for s, v in series.items() if v is not None and math.isfinite(v)}
    if not vals:
        return None
    if site == 'min':
        return min(vals.values())
    if site == 'max':
        return max(vals.values())
    if site == 'median':
        return statistics.median(vals.values())
    if site == 'argmin':
        return min(vals, key=vals.get)
    if site == 'argmax':
        return max(vals, key=vals.get)
    return None


def lookup(spec):
    kind, *a = spec.split(':')
    if kind == 'dec':
        tag, K, dec, met, site = a
        tag = _resolve_tag(tag, DEC_TAGS)
        ser = {s: (dec_val(tag, s, f'H|{K}|{dec}|{met}') or [None])[0] for s in SITES}
    elif kind == 'ood':
        tag, st, K, dec, met, site = a
        tag = _resolve_tag(tag, DEC_TAGS)
        ser = {s: (dec_val(tag, s, f'{st}|{K}|{dec}|{met}') or [None])[0] for s in SITES}
    elif kind == 'rt':
        tag, K, key, site = a
        tag = _resolve_tag(tag, RT_TAGS)
        ser = {s: rt_val(tag, K, s, key) for s in SITES}
    elif kind == 'ker':
        kn, field, site = a
        ker = R['kernel']
        if field == 'hubmax':
            ser = {s: ker.get(s, {}).get(kn, {}).get('hub', {}).get('max') for s in SITES}
        elif field == 'hubskew':
            ser = {s: ker.get(s, {}).get(kn, {}).get('hub', {}).get('skew') for s in SITES}
        elif field == 'perp':
            ser = {s: ker.get(s, {}).get(kn, {}).get('perp') for s in SITES}
        elif field == 'rawhub':
            ser = {s: ker.get(s, {}).get('raw_cos_hub', {}).get('max') for s in SITES}
        elif field.startswith('stab'):
            ser = {s: ker.get(s, {}).get(kn, {}).get('stab', {}).get(field[4:]) for s in SITES}
        else:
            return None
    elif kind == 'refit':
        kn, K, q, site = a
        v = R.get('refit', {}).get(f'{kn}|{K}', {}).get(q)
        ser = dict(zip(SITES, v)) if v else {}
    elif kind == 'ml':                 # ml:<tag>:max | ml:<tag>:argmax | ml:<tag>:count:<threshold>:<first m modes>
        tag, field, *rest = a
        tag = {'main': MAIN}.get(tag, tag)
        top = R.get('modeloc', {}).get(tag, {}).get('top2')
        if not top:
            return None
        if field == 'max':
            return max(top)
        if field == 'argmax':
            return f'mode {max(range(len(top)), key=lambda j: top[j]) + 1}'
        if field == 'count':
            return sum(1 for x in top[:int(rest[1])] if x > float(rest[0]))
        return None
    elif kind == 'adv':
        att, field, site = a
        keys = sorted(R.get('adv', {}), key=lambda k: tag_order(k.rsplit('_K', 1)[0]) + (k,))
        key = next((k for k in keys if MAIN and k.startswith(MAIN + '_')), keys[0] if keys else None)
        rows = R['adv'].get(key, {}).get(att, []) if key else []
        ser = {r['site']: r.get(field) for r in rows}
    else:
        return None
    return _agg(ser, site)


def fmt_num(v, f='g'):
    if v is None:
        return None
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return f'{v:,}'
    if f == 'pct':
        return f'{100 * v:.0f}%' if abs(v) >= 0.1 else f'{100 * v:.1f}%'
    if f == 'int':
        return f'{v:,.0f}'
    if f in ('2f', '1f', '3f'):
        return f'{v:.{f[0]}f}'
    if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
        m, e = f'{v:.1e}'.split('e')
        return f'{m}×10<sup>{int(e)}</sup>'
    if abs(v) >= 10 and float(v).is_integer():
        return f'{v:,.0f}'
    return f'{v:#.3g}'.rstrip('.')


# Named numbers (the brief's style, e.g. {b4_cos_k16}): spec strings resolved through lookup().
def named_specs():
    sp = {
        'main_tag': None, 'rt_tag': None,
        'b4_cos_k16': 'dec:main:16:krr:cos_raw:b4',
        'krr_cos_k16_min': 'dec:main:16:krr:cos_raw:min', 'krr_cos_k16_argmin': 'dec:main:16:krr:cos_raw:argmin',
        'krr_cos_k16_max': 'dec:main:16:krr:cos_raw:max', 'krr_cos_k16_argmax': 'dec:main:16:krr:cos_raw:argmax',
        'krr_cos_k16_b3': 'dec:main:16:krr:cos_raw:b3', 'krr_cos_k16_b9': 'dec:main:16:krr:cos_raw:b9',
        'krr_cos_k64_b3': 'dec:main:64:krr:cos_raw:b3', 'krr_cos_k256_b3': 'dec:main:256:krr:cos_raw:b3',
        'pca_cos_k16_b3': 'dec:main:16:pca:cos_raw:b3', 'mlp_cos_k16_b3': 'dec:main:16:mlp:cos_raw:b3',
        'linear_cos_k16_b3': 'dec:main:16:linear:cos_raw:b3', 'nn_cos_k16_b3': 'dec:main:16:nn:cos_raw:b3',
        'krr_cap_k16_b3': 'dec:main:16:krr:cap_raw:b3', 'krr_cap_k16_pool': 'dec:main:16:krr:cap_raw:pool',
        'krr_cos_k16_pool': 'dec:main:16:krr:cos_raw:pool',
        'p2_krr_cos_k16_b3': 'dec:P2:16:krr:cos_raw:b3', 'p8_krr_cos_k16_b3': 'dec:P8:16:krr:cos_raw:b3',
        'krr_kl_k16_b3': 'dec:main:16:krr:kl:b3', 'mean_kl_b3': 'dec:main:0:mean:kl:b3',
        'krr_kl_k16_stem': 'dec:main:16:krr:kl:stem', 'mean_kl_stem': 'dec:main:0:mean:kl:stem',
        'krr_kl_k16_b9': 'dec:main:16:krr:kl:b9', 'mean_kl_b9': 'dec:main:0:mean:kl:b9',
        'krr_agree_k16_b9': 'dec:main:16:krr:agree:b9', 'krr_agree_k16_b3': 'dec:main:16:krr:agree:b3',
        'krr_agree_on_k16_b3': 'dec:main:16:krr:agree_on:b3', 'mean_agree_max': 'dec:main:0:mean:agree:max',
        'true_kl_max': 'dec:main:0:true:kl:max',
        'share_ll_min': 'rt:main:16:E1|share_ll:min', 'share_ll_max': 'rt:main:16:E1|share_ll:max',
        'share_ll_argmax': 'rt:main:16:E1|share_ll:argmax',
        'corr_min': 'rt:main:16:E1|corr:min', 'corr_max': 'rt:main:16:E1|corr:max',
        'srel_krr_max': 'rt:main:16:E1|krr|srel:max', 'srel_krr_argmax': 'rt:main:16:E1|krr|srel:argmax',
        'e2_srel_median': 'rt:main:16:E2|krr|srel:median',
        'e3_miss_max': 'rt:main:16:E3|miss:max', 'e3n_miss_max': 'rt:main:16:E3n|miss:max',
        'e3_move_min': 'rt:main:16:E3|move:min', 'e3_move_max': 'rt:main:16:E3|move:max',
        'e3_tang_max': 'rt:main:16:E3|tang:max', 'e3_tang_argmax': 'rt:main:16:E3|tang:argmax',
        'e5_05_miss_max': 'rt:main:16:E5|0.5|miss_S:max', 'e5_15_miss_min': 'rt:main:16:E5|1.5|miss_S:min',
        'e5_15_miss_max': 'rt:main:16:E5|1.5|miss_S:max',
        'match_pb_k16_b9': 'rt:main:16:E4|match|p_b:b9', 'match_pb_k16_b3': 'rt:main:16:E4|match|p_b:b3',
        'match_pb_k16_max': 'rt:main:16:E4|match|p_b:max', 'match_pb_k16_argmax': 'rt:main:16:E4|match|p_b:argmax',
        'randmatch_pb_k16_max': 'rt:main:16:E4|rand~match|p_b:max', 'swap_pb_k16': 'rt:main:16:E4|swap|p_b:median',
        'match_size_k16_median': 'rt:main:16:E4|match|size:median',
        'match_progS_k16_min': 'rt:main:16:E4|match|prog_S:min',
        'hub_pr300_max': 'ker:pr300:hubmax:max', 'stab16_pr300_b6': 'ker:pr300:stab16:b6',
        'refit_held_pr300_k16_min': 'refit:pr300:16:held:min', 'refit_held_pr300_k64_min': 'refit:pr300:64:held:min',
        'ml_p8ea400_max': 'ml:P8__ea400__L12:max', 'ml_p8ea400_argmax': 'ml:P8__ea400__L12:argmax',
        'ml_p8ea400_n01_64': 'ml:P8__ea400__L12:count:0.1:64', 'ml_p8ea400_n05_64': 'ml:P8__ea400__L12:count:0.5:64',
        'ml_p2ea300_n01_64': 'ml:P2__ea300__L12:count:0.1:64', 'ml_p8knn_max': 'ml:P8__knn15__L12:max',
        'ml_p8pr400_max': 'ml:P8__pr400__L12:max', 'ml_p8pr400_n01_64': 'ml:P8__pr400__L12:count:0.1:64',
        'ml_main_max': 'ml:main:max', 'ml_main_n01_64': 'ml:main:count:0.1:64',
        'p2_krr_cos_k16_b3_x': 'dec:P2:16:krr:cos_raw:b3', 'p2_krr_cos_k64_b3': 'dec:P2:64:krr:cos_raw:b3',
        'p2_krr_cos_k256_b3': 'dec:P2:256:krr:cos_raw:b3',
        'mlp_cap_k16_min': 'dec:main:16:mlp:cap_raw:min', 'mlp_cap_k16_argmin': 'dec:main:16:mlp:cap_raw:argmin',
        'adv_pgd8src_disp_stem': 'adv:pgd8_src:disp:stem', 'adv_pgd8_disp_stem': 'adv:pgd8:disp:stem',
        'hub_ea300_max': 'ker:ea300:hubmax:max', 'hub_raw_max': 'ker:knn15:rawhub:max',
        'hub_raw_argmax': 'ker:knn15:rawhub:argmax', 'hub_knn_walk_max': 'ker:knn15:hubmax:max',
        'stab16_ea300_b6': 'ker:ea300:stab16:b6', 'stab16_knn15_b6': 'ker:knn15:stab16:b6',
        'refit_held_ea300_k16_min': 'refit:ea300:16:held:min', 'refit_held_ea300_k16_argmin': 'refit:ea300:16:held:argmin',
        'refit_held_ea300_k64_min': 'refit:ea300:64:held:min',
        'adv_pgd8_disp_max': 'adv:pgd8:disp:max', 'adv_pgd8_disp_argmax': 'adv:pgd8:disp:argmax',
        'adv_pgd2_disp_max': 'adv:pgd2:disp:max',
        'svhn_krr_cos_k16_b3': 'ood:main:svhn:16:krr:cos_raw:b3', 'noise20_krr_cos_k16_stem': 'ood:main:noise20:16:krr:cos_raw:stem',
    }
    return sp


def build_named():
    out = {}
    for name, spec in named_specs().items():
        if name == 'main_tag':
            out[name] = tag_label(MAIN) if MAIN else 'n/a'
            continue
        if name == 'rt_tag':
            out[name] = tag_label(RT_MAIN) if RT_MAIN else 'n/a'
            continue
        try:
            v = lookup(spec)
        except (ValueError, KeyError, TypeError):
            v = None
        out[name] = fmt_num(v) if v is not None else None
    # lowest MLP captured share over every tag (K = 16): the benchmark can do worse than the probe mean
    lows = [(v[0], t, st) for t in DEC_TAGS for st in SITES if (v := dec_val(t, st, 'H|16|mlp|cap_raw'))]
    if lows:
        v, t, st = min(lows)
        out['mlp_cap_k16_min_all'], out['mlp_cap_k16_min_all_where'] = fmt_num(v), f'{t_short(t)}, {st}'
    # facts
    out['n_test_acc'] = test_accuracy()
    for p in PROBES:
        out[f'{p.lower()}_sites'] = str(n_sites_dec(MAIN_BY_PROBE[p]))
    out['n_sites'] = str(len(SITES))
    return out


def test_accuracy():
    try:
        import numpy as np
        sp = np.load(DATA / 'splits.npz')
        lg = np.load(DATA / 'meta/c10_logits.npy')
        return f'{100 * float((lg.argmax(1) == sp["labels"]).mean()):.2f}%'
    except Exception as e:  # noqa: BLE001
        WARN.append(f'test accuracy not computed ({e}); page_text fallback used')
        return None


NAMED = build_named()
PH = re.compile(r'\{(=[^{}\s]+|[A-Za-z][A-Za-z0-9]*_[A-Za-z0-9_]+)\}')
USED: set[str] = set()


def fill(s):
    """Substitute number placeholders in a prose string (TeX is never passed through here)."""
    if not isinstance(s, str):
        return s

    def rep(m):
        key = m.group(1)
        USED.add(key)
        if key.startswith('='):
            spec, _, f = key[1:].partition(';')
            try:
                v = fmt_num(lookup(spec), f or 'g')
            except (ValueError, KeyError, TypeError, IndexError):
                v = None
        elif key not in NAMED:
            WARN.append(f'{{{key}}} is not a known placeholder; left as written')
            return m.group(0)
        else:
            v = NAMED.get(key)
        if v is None:
            WARN.append(f'placeholder {{{key}}} has no value; shown as n/a')
            return 'n/a'
        return v

    return PH.sub(rep, s)


def T(path, default=''):
    """Text from page_text.json by dotted path, placeholders filled."""
    cur = TEXT
    for p in path.split('.'):
        if not isinstance(cur, dict) or p not in cur:
            WARN.append(f'page_text.json has no {path}')
            return default
        cur = cur[p]
    if isinstance(cur, list):
        return [fill(x) if isinstance(x, str) else [fill(y) for y in x] for x in cur]
    return fill(cur)


def FT(fid):
    return T(f'fig.{fid}.title', fid), T(f'fig.{fid}.caption', '')


# ------------------------------------------------------------------ styles (one grammar) --

DEC_STYLE = {
    'krr': dict(color='#d0473f', dash='solid', label='kernel ridge (krr)'),
    'linear': dict(color='#4477aa', dash='solid', label='linear'),
    'gh': dict(color='#228833', dash='solid', label='geometric harmonics (gh)'),
    'lp': dict(color='#c9a227', dash='solid', label='Laplacian pyramid (lp)'),
    'lpz': dict(color='#c9a227', dash='dashdot', label='Laplacian pyramid, zero-diag stop (lpz)'),
    'nn': dict(color='#aa3377', dash='solid', label='15-NN mean (nn)'),
    'pca': dict(color='#555555', dash='dash', label='PCA K+1 (reference)', symbol='square'),
    'step': dict(color='#999999', dash='dash', label='kernel step (reference)', symbol='diamond'),
    'mlp': dict(color='#111111', dash='dot', label='trained MLP (benchmark)', symbol='x'),
    'mean': dict(color='#9a9a9a', dash='dot', label='probe mean (control)'),
    'true': dict(color='#2f2f2f', dash='dot', label='true activation (control)'),
}
KER_STYLE = {   # hue = kernel family (participation ratio: purples; Shannon perplexity: blues; global width: greens;
    #               k-NN: orange); shade = the family's parameter
    'pr100': dict(color='#c77dc0', label='participation ratio k=100'), 'pr300': dict(color='#8e2a8b', label='participation ratio k=300'),
    'pr400': dict(color='#5b0f63', label='participation ratio k=400'),
    'ea30': dict(color='#9ecae1', label='Shannon ξ=30'), 'ea100': dict(color='#4a90c2', label='Shannon ξ=100'),
    'ea300': dict(color='#08519c', label='Shannon ξ=300'), 'ea400': dict(color='#08306b', label='Shannon ξ=400'),
    'ds150': dict(color='#74c476', label='global width, perp 150'), 'ds500': dict(color='#238b45', label='global width, perp 500'),
    'knn15': dict(color='#e6862c', label='k-NN k=15'), 'knn30': dict(color='#b35806', label='k-NN k=30'),
}
SET_STYLE = {
    'H': dict(color='#111111', label='CIFAR-10 held-out (H)'), 'c100': dict(color='#2a9d8f', label='CIFAR-100'),
    'svhn': dict(color='#7b5ea7', label='SVHN'), 'noise05': dict(color='#f4a259', label='H + noise σ=0.05'),
    'noise10': dict(color='#e07a2f', label='H + noise σ=0.10'), 'noise20': dict(color='#b5541c', label='H + noise σ=0.20'),
    'pgd2': dict(color='#e56b9f', label='PGD ε=2/255'), 'pgd8': dict(color='#a4255e', label='PGD ε=8/255'),
}
METH_STYLE = {
    'swap': dict(color='#111111', label='swap (x_b itself)'), 'decode': dict(color='#d0473f', label='decode D(ψ_b)'),
    'resid': dict(color='#c9a227', label='residual x_a + D(ψ_b) − D(ψ_a)'), 'match': dict(color='#4477aa', label='match from x_a'),
    'match_p': dict(color='#66ccee', label='match from x_a, probe metric'),
    'resid+match': dict(color='#228833', label='match from residual guess'),
    'match_req': dict(color='#aa3377', label='match from x_a, inferred target'),
    'resid+match_req': dict(color='#8c6d46', label='match from residual guess, inferred target'),
    'none': dict(color='#bbbbbb', label='no edit'),
}
RT_STYLE = {
    'E3': dict(color='#0f7c7c', dash='solid', label='matching, oracle target (true sources)'),
    'E3n': dict(color='#0f7c7c', dash='dash', label='matching, inferred target s′(ψ)'),
}
for _grp, _sty in (('decoder', DEC_STYLE), ('kernel', KER_STYLE), ('set', SET_STYLE), ('method', METH_STYLE), ('matching', RT_STYLE)):
    for _k, _lab in TEXT.get('legend', {}).get(_grp, {}).items():   # legend labels live in page_text.json too
        if _k in _sty:
            _sty[_k]['label'] = _lab
KER_ORDER = list(KER_STYLE)
for k in list(KER_STYLE):
    KER_STYLE[k].setdefault('dash', 'solid')


def style_for_kernel(k):
    if k in KER_STYLE:
        return KER_STYLE[k]
    fam = {'pr': '#7b3294', 'ea': '#2166ac', 'ds': '#1b7837', 'knn': '#d95f02'}.get(kernel_family(k), '#6b6b6b')
    return dict(color=fam, dash='solid', label=kernel_label(k))


# ------------------------------------------------------------------ JS --

JS = r"""
const META = DATA.meta;
const NS = META.sites.length;
function siteAxis(extra) { return merge(AXIS, merge({type:'category', categoryorder:'array', categoryarray: META.sites,
  range:[-0.5, NS - 0.5], title:{text:'site'}}, extra || {})); }
function clearEmpty(el) { el.querySelectorAll(':scope > p.empty').forEach(p => p.remove()); }
function empty(el, msg) { try { Plotly.purge(el); } catch (e) {} el.innerHTML = '<p class="empty">' + msg + '</p>'; }
function plot(el, traces, layout) {
  traces = traces.filter(t => t && t.y && t.y.some(v => v != null) || (t && t.type === 'heatmap') || (t && t.legendonly));
  if (!traces.length) { empty(el, 'No data for this selection yet (runs in progress).'); return; }
  layout = Object.assign({}, layout);
  const names = layout.showlegend === false ? [] : traces.filter(t => t.showlegend !== false && t.name && t.type !== 'heatmap').map(t => t.name);
  if (names.length) {   // legend above the plot, in a top margin sized to its estimated rows
    const W = Math.max(280, (el.clientWidth || 900) - 70); let rows = 1, x = 0;
    for (const n of names) { const w = n.length * 6.9 + 46; if (x + w > W && x > 0) { rows++; x = 0; } x += w; }
    layout.legend = {orientation:'h', x:0, xanchor:'left', y:1.0, yanchor:'bottom', font:{size:12}};
    layout.margin = merge({l:56, r:18, t:36, b:48}, merge(layout.margin || {}, {t: (layout.title ? 30 : 10) + rows * 20 + 6}));
    layout.height = (layout.height || 380) + rows * 20;
  }
  for (const k of Object.keys(layout)) if (/^yaxis\d*$/.test(k) && layout[k] && layout[k].type === 'log') {
    const r = layout[k].range; layout[k] = merge(layout[k], {exponentformat:'power'}); if (!r || r[1] - r[0] > 1.6) layout[k].dtick = 1; }
  clearEmpty(el); Plotly.react(el, traces, LAYOUT(layout), CONFIG);
}
function val(v) { return v == null ? null : (Array.isArray(v) ? v[0] : v); }
function sline(name, ys, st, extra) {
  const y = ys.map(val);
  const t = {type:'scatter', mode:'lines+markers', name, x: META.sites, y, connectgaps:false,
    line:{color: st.color, dash: st.dash || 'solid', width: st.width || 2}, marker:{size: st.msize || (st.symbol ? 7 : 5), color: st.color, symbol: st.symbol || 'circle'},
    hovertemplate: name + ' · %{x}: %{y:.3~g}<extra></extra>'};
  const e = ys.map(v => Array.isArray(v) && v[1] != null ? 2 * v[1] : null);
  if (e.some(v => v != null && v > 0)) t.error_y = {type:'data', array: e.map(v => v || 0), thickness:1, width:2, color: st.color};
  return merge(t, extra || {});
}
function logPos(ys, floor) { return ys.map(v => { const x = val(v); if (x == null) return null;
  if (floor) return Math.max(x, floor); return x > 0 ? x : null; }); }
const FLOOR = 1e-9;   // log-axis floor for residuals: values at or below it (exact zeros included) are drawn on it
function clampNote(all) { return all.some(s => s.some(v => { const x = val(v); return x != null && x <= FLOOR; })) ? ' · values ≤ 1e-9 drawn at 1e-9' : ''; }
function hline(y, txt, color) { return {shape:{type:'line', xref:'paper', x0:0, x1:1, y0:y, y1:y, line:{color: color || '#999', dash:'dot', width:1}},
  ann: txt ? {xref:'paper', x:1, y:y, xanchor:'right', yanchor:'bottom', text:txt, showarrow:false, font:{size:11, color: color || '#777'}} : null}; }
function withRefs(layout, refs) { layout.shapes = (layout.shapes || []).concat(refs.map(r => r.shape));
  layout.annotations = (layout.annotations || []).concat(refs.map(r => r.ann).filter(a => a)); return layout; }
function rng(vals, lo, hi, pad) { const v = vals.flat(3).map(val).filter(x => x != null && isFinite(x));
  if (!v.length) return [lo, hi]; let a = Math.min(lo == null ? Infinity : lo, ...v), b = Math.max(hi == null ? -Infinity : hi, ...v);
  const p = (b - a) * (pad == null ? 0.05 : pad); return [a - p, b + p]; }
function logRange(vals, floor) { const v = vals.flat(3).map(val).filter(x => x != null && x > (floor || 0));
  if (floor === FLOOR && vals.flat(3).map(val).some(x => x != null && x <= FLOOR)) v.push(FLOOR * 1.0001);
  if (!v.length) return [-3, 1]; return [Math.log10(Math.min(...v)) - 0.15, Math.log10(Math.max(...v)) + 0.15]; }
function subTitle(text) { return {text: text, font:{size:12, color:'#6b665e'}, x:0, xanchor:'left', y:0.995, yanchor:'top'}; }
function tagName(t) { return META.tag_label[t] || t; }
function decSeries(tag, K, dec, met) { const d = DATA.dec[tag]; if (!d) return null;
  const kk = (dec === 'mean' || dec === 'step' || dec === 'true') ? '0' : String(K);
  return d[kk] && d[kk][dec] && d[kk][dec][met] ? d[kk][dec][met] : null; }
function rtSeries(tag, K, key) { const d = DATA.rt[tag]; return d && d[String(K)] && d[String(K)][key] ? d[String(K)][key] : null; }
function facetAxes(n, gap) { const w = (1 - gap * (n - 1)) / n; return Array.from({length:n}, (_, i) => [i * (w + gap), i * (w + gap) + w]); }
function imgGrid(el, g, note) {
  if (!g) { el.innerHTML = '<p class="empty">' + (note || 'No image grid for this selection.') + '</p>'; return; }
  const n = g.cols ? g.cols.length : 0;
  const head = g.cols ? '<div class="pxhead" style="grid-template-columns:repeat(' + n + ',1fr);padding-left:' + (100 / g.w) + '%">' +
    g.cols.map(c => '<span>' + c + '</span>').join('') + '</div>' : '';
  el.innerHTML = '<div class="pxgrid" style="max-width:' + (g.w * g.scale) + 'px">' + head +
    '<img src="' + g.src + '" alt="image grid"></div>';
}
"""

CSS = """
p.empty { color:#8a5a1f; background:#fbf2e6; border:1px solid #ecd3ae; border-radius:6px; padding:8px 12px; font-size:13.5px; }
.pxgrid { margin:4px auto 0; }
.pxgrid img { display:block; width:100%; height:auto; image-rendering:pixelated; image-rendering:crisp-edges; }
.pxhead { display:grid; font-size:11.5px; color:#2b2a27; font-weight:600; margin-bottom:3px; }
.pxhead span { text-align:center; overflow:hidden; white-space:nowrap; text-overflow:ellipsis; padding:0 1px; }
.fig .plot.imgplot { min-height:60px; }
.eq-title { font-weight:700; font-size:14.5px; margin:16px 0 0; }
.banner .facts { font-size:13px; color:var(--dim); margin:2px 0 6px; }
.legend-note { font-size:12.5px; color:var(--dim); margin:2px 0 8px; }
table.doc td code { white-space:nowrap; }
.status { white-space:normal; display:inline-block; }
main { overflow-x:hidden; }
"""


# ------------------------------------------------------------------ figure helper --


def fig(tab, fid, draw, data, *, height=None, controls=None, subtitle=''):
    title, cap = FT(fid)
    tab.figure(fid, title, data=data, draw=draw, caption=cap, height=height, controls=controls, subtitle=subtitle)


def grid(nb, tab, items):
    """items: list of (fid, data, draw, height)."""
    shells = []
    for fid, data, draw, h in items:
        title, cap = FT(fid)
        shells.append(render_figure(fid, title, caption=cap, height=h))
    tab.grid(*shells)
    for fid, data, draw, h in items:
        add_figure(nb, tab, fid, data, draw)


def prose_block(tab, key):
    v = T(key, '')
    if isinstance(v, list):
        tab.prose(''.join(f'<p>{x}</p>' for x in v))
    elif v:
        tab.prose(v if v.lstrip().startswith('<') else f'<p>{v}</p>')


# ------------------------------------------------------------------ build --


def arch_banner():
    run, step, shapes = 'resnet20-relu-seed0', 64000, None
    try:
        tree = ast.parse((HERE / 'modal_invert.py').read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'SHAPES':
                shapes = ast.literal_eval(node.value)
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Tuple):
                names = [t.id for t in node.targets[0].elts if isinstance(t, ast.Name)]
                if names == ['RUN', 'STEP']:
                    run, step = ast.literal_eval(node.value)
    except Exception as e:  # noqa: BLE001
        WARN.append(f'could not read SHAPES from modal_invert.py ({e})')
    if not shapes:
        shapes = {'pixels': (3, 32, 32), 'stem': (16, 32, 32), **{f'b{i}': ((16, 32, 32), (32, 16, 16), (64, 8, 8))[(i - 1) // 3] for i in range(1, 10)},
                  'pool': (64,), 'logits': (10,)}

    def shp(s):
        t = shapes[s]
        return f'{t[0]}×{t[1]}²' if len(t) == 3 else f'{t[0]}'

    stage_kind = {16: 'mlp', 32: 'attn', 64: 'norm'}
    blocks = [('input', shp('pixels'), 'in'), ('conv·BN·ReLU', shp('stem'), 'act')]
    blocks += [(f'block {i}', shp(f'b{i}'), stage_kind.get(shapes[f'b{i}'][0], 'other')) for i in range(1, 10)]
    blocks += [('avg pool', shp('pool'), 'other'), ('fc', shp('logits'), 'out')]
    cuts = {1: 'stem', **{i + 1: f'b{i}' for i in range(1, 10)}, 11: 'pool', 12: 'logits'}
    skips = [(i, i + 1) for i in range(1, 10)]
    acc = NAMED.get('n_test_acc') or T('banner.acc_fallback', '')
    svg = arch_svg(blocks, skips=skips, cuts=cuts, caption=T('banner.caption'))
    svg = re.sub(r'viewBox="0 0 (\d+) (\d+)" width="(\d+)"',          # room for the last cut label (logits)
                 lambda m: f'viewBox="0 0 {int(m.group(1)) + 30} {m.group(2)}" width="{int(m.group(3)) + 30}"', svg)
    facts = T('banner.facts').replace('{run}', run).replace('{step}', f'{step:,}').replace('{acc}', acc)
    return f'<div class="facts">{facts}</div><div class="archs">{svg}</div>'


def main():
    meta = dict(sites=SITES, tag_label={t: tag_label(t) for t in DEC_TAGS + RT_TAGS},
                dec_style=DEC_STYLE, ker_style={k: style_for_kernel(k) for k in
                                               set(KER_STYLE) | {kernel_of(t) for t in DEC_TAGS} | set(KERNEL_NAMES)
                                               | {kernel_of(t) for t in R.get('modeloc', {})}},
                set_style=SET_STYLE, meth_style=METH_STYLE, rt_style=RT_STYLE,
                probes=PROBES, main_by_probe=MAIN_BY_PROBE,
                dec_tags=DEC_TAGS,
                adv_label={k: f'{tag_label(k.rsplit("_K", 1)[0])} · K = {k.rsplit("_K", 1)[1]}' for k in R.get('adv', {}) if '_K' in k})
    built = datetime.fromtimestamp((DATA / 'results.json').stat().st_mtime).strftime('%Y-%m-%d %H:%M')
    p8 = [t for t in DEC_TAGS if probe_of(t) != 'P2']
    status_bits = [T('page.status')]
    for t in DEC_TAGS:
        if n_sites_dec(t) < len(SITES):
            status_bits.append(f'{t.split("__L")[0].replace("__", " · ")}: {n_sites_dec(t)}/{len(SITES)} sites')
    for t in sorted({*R.get('modeloc', {}), *(k.rsplit('_K', 1)[0] for k in R.get('adv', {}))} - set(DEC_TAGS), key=tag_order):
        status_bits.append(f'{t.split("__L")[0].replace("__", " · ")}: decode pending')
    if MAIN:
        status_bits.append('default ' + t_short(MAIN))
    nb = Notebook(T('page.title'), T('page.subtitle'), status=' | '.join(status_bits), kicker=T('page.kicker'),
                  footer=T('page.footer').replace('{built}', built).replace('{today}', datetime.now().strftime('%Y-%m-%d %H:%M')))
    nb.style(CSS)
    nb.banner(arch_banner(), label=T('banner.label', 'Network and sites'))
    nb.dataset('meta', clean(meta))
    nb.dataset('dec', clean(decode_payload()))
    nb.dataset('ood', clean(ood_payload()))
    nb.dataset('rt', clean(rt_payload()))
    nb.dataset('ker', clean(kernel_payload()))
    nb.dataset('refit', clean(refit_payload()))
    nb.dataset('adv', clean(adv_payload()))
    nb.dataset('modeloc', clean({t: R['modeloc'][t] for t in sorted(R.get('modeloc', {}), key=tag_order)}))
    imgs = image_payload()
    nb.dataset('img', imgs)
    nb.script(JS)

    dec_ks = dec_Ks()
    k_want = TEXT.get('config', {}).get('decode_K_default', 16)
    k_def = k_want if k_want in dec_ks else (16 if 16 in dec_ks else (dec_ks[0] if dec_ks else 16))
    tag_ctl = lambda name='tag', tags=DEC_TAGS, default=MAIN: Select(  # noqa: E731
        name, T('controls.probe', 'Probe · kernel'), tags, default=default, labels={t: tag_label(t) for t in tags})

    # ---------------------------------------------------------------- 0 overview
    ov = nb.tab('overview', T('tabs.overview', 'Overview'))
    ov.h(T('overview.h_question'))
    prose_block(ov, 'overview.question')
    ov.h(T('overview.h_objects'))
    prose_block(ov, 'overview.objects_intro')
    for eq in TEXT.get('overview', {}).get('equations', []):
        ov.prose(f'<div class="eq-title">{fill(eq.get("name", ""))}</div>')
        ov.equation(eq['tex'], fill(eq.get('where', '')))
    ov.h(T('overview.h_map'))
    prose_block(ov, 'overview.map_intro')
    rows = []
    for row in TEXT.get('overview', {}).get('map', []):
        rows.append([fill(row['tab']), fill(row['question']), fill(row['readout']), status_html(row.get('status_key', ''))])
    ov.table(T('overview.map_header'), rows)
    ov.h(T('overview.h_data'))
    prose_block(ov, 'overview.data')

    # ---------------------------------------------------------------- 1 kernel
    kt = nb.tab('kernel', T('tabs.kernel', 'Kernel'))
    prose_block(kt, 'kernel.intro')
    stab_sites = [s for s in TEXT.get('config', {}).get('stab_sites', ['b3', 'b6', 'b9']) if s in SITES]
    if R.get('modeloc'):
        ml_probes = sorted({probe_of(t) for t in R['modeloc']}, key=lambda p: probe_n(p) or 0)
        thr = TEXT.get('config', {}).get('modeloc_thresholds', [0.1, 0.5])
        ml_n = {p: next((R['modeloc'][t].get('n') for t in R['modeloc'] if probe_of(t) == p), probe_n(p)) for p in ml_probes}
        fig(kt, 'kernel_modeloc', r"""(el, d, c) => {
  const probes = %s, thr = %s, ns = %s, dom = facetAxes(probes.length, 0.06), tr = [], lay = {height:420, shapes:[], annotations:[]};
  const all = Object.values(d).map(o => o.top2);
  const yr = logRange(all, 0); yr[1] = Math.max(yr[1], 0.05);
  probes.forEach((p, i) => { const ax = i ? String(i + 1) : '';
    for (const t of Object.keys(d).filter(t => t.split('__')[0] === p)) { const kn = t.split('__')[1], st = META.ker_style[kn] || {color:'#777', label:kn};
      const y = d[t].top2, x = y.map((_, j) => j + 1);
      tr.push({type:'scatter', mode:'lines+markers', x, y, name: st.label, legendgroup: kn, showlegend: !tr.some(q => q.legendgroup === kn),
        xaxis:'x' + ax, yaxis:'y' + ax, line:{color: st.color, width: kn.startsWith('pr') ? 2.4 : 1.6}, marker:{size:4, color: st.color},
        hovertemplate: st.label + ' · ' + p + ' · mode %%{x}: %%{y:.3f}<extra></extra>'}); }
    lay['xaxis' + ax] = merge(AXIS, {domain: dom[i], anchor:'y' + ax, range:[0, 65], title:{text:'<b>' + p + '</b> (N = ' + (ns[p] || '?').toLocaleString() + ') · mode index'}});
    lay['yaxis' + ax] = merge(AXIS, {type:'log', range: yr, anchor:'x' + ax, showticklabels: i === 0, title:{text: i === 0 ? 'share of squared norm on the two heaviest images (log)' : ''}});
    for (const h of thr) { lay.shapes.push({type:'line', xref:'x' + ax, yref:'y' + ax, x0:0, x1:65, y0:h, y1:h, line:{color:'#b04a4a', dash:'dot', width:1}});
      lay.annotations.push({xref:'x' + ax, yref:'y' + ax, x:64, y: Math.log10(h), xanchor:'right', yanchor:'bottom', showarrow:false, text: String(h), font:{size:11, color:'#b04a4a'}}); } });
  plot(el, tr, lay);
}""" % (json.dumps(ml_probes), json.dumps(thr), json.dumps(ml_n)), '@modeloc')
        rows = []
        for t in sorted(R['modeloc'], key=tag_order):
            a = R['modeloc'][t]['top2']
            jm = max(range(len(a)), key=lambda j: a[j])
            row = [tag_label(t), f"{R['modeloc'][t].get('n', ''):,}" if isinstance(R['modeloc'][t].get('n'), int) else '']
            for m in (16, 64):
                for h in thr:
                    row.append(str(sum(1 for x in a[:m] if x > h)))
            row.append(f'{a[jm]:.2f} (mode {jm + 1})')
            rows.append(row)
        hdr = [T('labels.ml_tag'), 'N'] + [T('labels.ml_count').replace('{m}', str(m)).replace('{h}', str(h)) for m in (16, 64) for h in thr] + [T('labels.ml_max')]
        kt.prose(f'<div class="ftitle">{T("labels.ml_table_title")}</div>')
        kt.table(hdr, rows, numeric=set(range(1, len(hdr))))
    fig(kt, 'kernel_hub', r"""(el, d, c) => {
  const st = c.hubstat, tr = [];
  for (const n of d.names) {
    const s = META.ker_style[n] || {color:'#777'};
    if (n === 'knn15') {
      tr.push(sline(s.label + ' · raw-cosine 15-NN graph', d.raw[st === 'max' ? 'max' : 'skew'], {color: s.color, dash:'solid'}));
      tr.push(sline(s.label + ' · walk top-15 (arbitrary under ties)', d['hub' + st][n], {color: s.color, dash:'dot', width:1.3, msize:3}, {visible:'legendonly'}));
    } else tr.push(sline(s.label, d['hub' + st][n], {color: s.color, dash: s.dash}));
  }
  plot(el, tr, {height:400, xaxis: siteAxis(), yaxis: merge(AXIS, {title:{text: st === 'max' ? 'max k-occurrence (15 heaviest entries per row)' : 'skew of k-occurrence'}, rangemode:'tozero'})});
}""", '@ker', controls=[Radio('hubstat', T('controls.hubstat', 'Statistic'), ['max', 'skew'], labels={'max': 'max', 'skew': 'skew'})])
    fig(kt, 'kernel_stab', r"""(el, d, c) => {
  const sites = %s, dom = facetAxes(sites.length, 0.05), tr = [], lay = {height:380, annotations:[], showlegend:true};
  sites.forEach((s, i) => {
    const ax = i ? String(i + 1) : '';
    for (const n of d.names) { const st = META.ker_style[n] || {color:'#777'};
      const ks = d.stab_k.filter(k => d.stab[n][s] && d.stab[n][s][k] != null);
      tr.push({type:'scatter', mode:'lines+markers', x: ks, y: ks.map(k => d.stab[n][s][k]), name: st.label, legendgroup: n, showlegend: i === 0,
        xaxis:'x' + ax, yaxis:'y' + ax, line:{color: st.color, width:2, dash: st.dash}, marker:{size:5, color: st.color},
        hovertemplate: st.label + ' · ' + s + ' · mode %%{x}: %%{y:.3f}<extra></extra>'}); }
    lay['xaxis' + ax] = merge(AXIS, {type:'log', domain: dom[i], tickvals: d.stab_k, ticktext: d.stab_k.map(String), title:{text:'<b>' + s + '</b> · mode index k (log)'}, anchor:'y' + ax});
    lay['yaxis' + ax] = merge(AXIS, {range:[-0.05, 1.05], anchor:'x' + ax, showticklabels: i === 0, title:{text: i === 0 ? 'reproducibility on a disjoint half' : ''}});
  });
  lay.legend = {orientation:'h', y:1.16, x:0};
  plot(el, tr, lay);
}""" % json.dumps(stab_sites), '@ker')
    rk = sorted(R.get('refit', {}).keys())
    refit_K = sorted({int(k.split('|')[1]) for k in rk})
    if refit_K:
        fig(kt, 'kernel_refit', r"""(el, d, c) => {
  const dd = d[String(c.refitK)] || {}, tr = [];
  const dash = {held:'solid', onestep:'dash', probe:'dot'}, lab = %s;
  for (const n of Object.keys(dd)) { const st = META.ker_style[n] || {color:'#777'};
    for (const q of (c.refitq === 'all' ? ['held', 'onestep', 'probe'] : ['held'])) if (dd[n][q])
      tr.push(sline(st.label + ' · ' + lab[q], dd[n][q], {color: st.color, dash: dash[q], width: q === 'held' ? 2.2 : 1.5}, {legendgroup: n})); }
  plot(el, tr, {height:420, xaxis: siteAxis(), yaxis: merge(AXIS, {title:{text:'share of squared coordinate norm reproduced'}, range: rng(Object.values(dd).map(o => Object.values(o)), null, 1.0)}),
    title: subTitle('K = ' + c.refitK), legend:{orientation:'h', y:1.2, x:0}});
}""" % json.dumps({'held': T('labels.refit_held', 'held-out, frozen-probe extension'),
                   'onestep': T('labels.refit_onestep', 'held-out, one kernel step'),
                   'probe': T('labels.refit_probe', 'probe images themselves')}),
            '@refit', controls=[Radio('refitK', 'K', refit_K, default=16 if 16 in refit_K else refit_K[0]),
                                Radio('refitq', T('controls.refitq', 'Lines'), ['held', 'all'],
                                      labels={'held': T('labels.refit_only_held'), 'all': T('labels.refit_all')})])
    prose_block(kt, 'kernel.outro')

    # ---------------------------------------------------------------- 2 decoding
    dt = nb.tab('decoding', T('tabs.decoding', 'Decoding'))
    prose_block(dt, 'decoding.intro')
    dt.controls([tag_ctl(), Radio('K', 'K', dec_ks, default=k_def)])
    dec_draw = r"""(el, d, c) => {
  const met = %s, med = met.endsWith('_med'), tr = [], all = [];
  const order = ['linear', 'nn', 'lp', 'lpz', 'gh', 'krr', 'pca', 'step', 'mlp'];
  for (const dec of order) { const s = decSeries(c.tag, c.K, dec, met); if (!s) continue; all.push(s);
    const st = META.dec_style[dec]; tr.push(sline(st.label + (dec === 'step' ? ' (K-free)' : ''), s, st, {width: dec === 'krr' ? 3 : 2})); }
  plot(el, tr, {height:440, xaxis: siteAxis(), yaxis: merge(AXIS, {title:{text: med ? %s : %s}, range: rng(all, 0, null)}),
    title: subTitle(tagName(c.tag) + ' · K = ' + c.K + (med ? ' · median over held-out images' : ' · mean, bars ±2 s.e. over held-out images')), legend:{orientation:'h', y:1.22, x:0}});
}"""
    fig(dt, 'dec_cos', dec_draw % ("'cos_raw'", js(T('labels.cos_raw')), js(T('labels.cos_raw'))), '@dec')
    fig(dt, 'dec_cap', dec_draw % ("(c.capstat === 'mean' ? 'cap_raw' : 'cap_raw_med')", js(T('labels.cap_raw_med')), js(T('labels.cap_raw'))), '@dec',
        controls=[Radio('capstat', T('controls.stat', 'Statistic'), ['median', 'mean'],
                        labels={'median': T('labels.stat_median'), 'mean': T('labels.stat_mean')})])
    vsk_sites = [s for s in TEXT.get('config', {}).get('vsk_sites', ['b3', 'b6', 'b9']) if s in SITES]
    fig(dt, 'dec_vsK', r"""(el, d, c) => {
  const sites = %s, dom = facetAxes(sites.length, 0.05), tr = [], all = [], lay = {height:400, annotations:[]};
  const decs = ['linear', 'nn', 'gh', 'krr', 'pca', 'mlp'], dashP = ['solid', 'dash', 'dot', 'dashdot'];
  const probes = META.probes.filter(p => META.main_by_probe[p]), vm = c.vmet === 'cap_raw' ? 'cap_raw_med' : c.vmet;
  sites.forEach((s, i) => { const ax = i ? String(i + 1) : '', si = META.sites.indexOf(s);
    probes.forEach((p, pi) => { const tag = META.main_by_probe[p], D = DATA.dec[tag] || {};
      for (const dec of decs) { const ks = Object.keys(D).map(Number).filter(k => k > 0 && D[k][dec] && D[k][dec][vm] && D[k][dec][vm][si] != null).sort((a, b) => a - b);
        if (!ks.length) continue; const st = META.dec_style[dec], y = ks.map(k => val(D[k][dec][vm][si])); all.push(y);
        tr.push({type:'scatter', mode:'lines+markers', x: ks, y, name: st.label + ' · ' + p, legendgroup: dec + p, showlegend: i === 0,
          xaxis:'x' + ax, yaxis:'y' + ax, line:{color: st.color, dash: dashP[pi %% 4], width: dec === 'krr' ? 2.6 : 1.8}, marker:{size: st.symbol ? 7 : 5, color: st.color, symbol: st.symbol || 'circle'},
          hovertemplate: st.label + ' · ' + tagName(tag) + ' · ' + s + ' · K=%%{x}: %%{y:.3f}<extra></extra>'}); } });
    lay['xaxis' + ax] = merge(AXIS, {type:'log', domain: dom[i], tickvals:[4, 16, 64, 256], ticktext:['4', '16', '64', '256'], title:{text:'<b>' + s + '</b> · K (log)'}, anchor:'y' + ax});
  });
  const r = rng(all, 0, null);
  sites.forEach((s, i) => { const ax = i ? String(i + 1) : ''; lay['yaxis' + ax] = merge(AXIS, {range: r, anchor:'x' + ax, showticklabels: i === 0,
    title:{text: i === 0 ? (c.vmet === 'cos_raw' ? 'held-out cosine (mean)' : 'captured share (median)') : ''}}); });
  lay.legend = {orientation:'h', y:1.24, x:0};
  plot(el, tr, lay);
}""" % json.dumps(vsk_sites), '@dec', controls=[Radio('vmet', T('controls.metric', 'Metric'), ['cos_raw', 'cap_raw'], labels={'cos_raw': 'cosine', 'cap_raw': 'captured share'})])
    fig(dt, 'dec_kernels', r"""(el, d, c) => {
  const tr = [], all = [], dashP = {}, probes = META.probes, dl = ['solid', 'dash', 'dot', 'dashdot'];
  probes.forEach((p, i) => dashP[p] = dl[i % 4]);
  const ref = p => META.dec_tags.find(t => t.split('__')[0] === p && t.split('__')[1].startsWith('knn'));
  for (const tag of META.dec_tags) { const p = tag.split('__')[0], kn = tag.split('__')[1], st = META.ker_style[kn] || {color:'#777', label: kn};
    const km = c.kmet === 'cap_raw' ? 'cap_raw_med' : c.kmet;
    let s = decSeries(tag, c.K, 'krr', km); if (!s) continue;
    if (c.kmode === 'diff') { const r = ref(p) ? decSeries(ref(p), c.K, 'krr', km) : null; if (!r) continue;
      s = s.map((v, i) => val(v) != null && val(r[i]) != null ? val(v) - val(r[i]) : null); }
    all.push(s); tr.push(sline(st.label + ' · ' + p, s, {color: st.color, dash: dashP[p], width: kn.startsWith('pr') ? 2.6 : 1.7}, c.kmode === 'diff' ? {error_y: undefined} : {})); }
  const yt = (c.kmet === 'cos_raw' ? 'held-out cosine (mean)' : 'captured share (median)') + ', kernel ridge' + (c.kmode === 'diff' ? ' − k-NN rule (same probe)' : '');
  const lay = {height:420, xaxis: siteAxis(), yaxis: merge(AXIS, {title:{text: yt}, range: c.kmode === 'diff' ? rng(all, 0, 0) : rng(all, 0, null)}),
    title: subTitle('K = ' + c.K + ' · hue: kernel family and parameter · ' + probes.map(p => p + ' ' + dashP[p]).join(', '))};
  if (c.kmode === 'diff') withRefs(lay, [hline(0, 'k-NN rule')]);
  plot(el, tr, lay);
}""", '@dec', controls=[Radio('kmet', T('controls.metric', 'Metric'), ['cos_raw', 'cap_raw'], labels={'cos_raw': 'cosine', 'cap_raw': 'captured share'}),
                        Radio('kmode', T('controls.kmode', 'Show'), ['value', 'diff'], labels={'value': 'value', 'diff': T('labels.kmode_diff')})])
    prose_block(dt, 'decoding.outro')

    # ---------------------------------------------------------------- 3 function
    ft = nb.tab('function', T('tabs.function', 'Function'))
    prose_block(ft, 'function.intro')
    ft.controls([tag_ctl(), Radio('K', 'K', dec_ks, default=k_def),
                 Radio('scale', T('controls.scale', 'Decoded norm'), ['pred', 'oracle', 'both'],
                       labels={'pred': 'predicted', 'oracle': 'oracle norm', 'both': 'both'})])
    fn_draw = r"""(el, d, c) => {
  const isKL = %s, med = isKL && c.klstat === 'median', sfx = med ? '_med' : '', met = isKL ? 'kl' : 'agree', tr = [], all = [];
  const fl = med ? 1e-3 : 1e-6, tx = s => isKL ? logPos(s, fl) : s;
  const order = ['linear', 'nn', 'lp', 'lpz', 'gh', 'krr', 'pca', 'mlp'];
  for (const dec of order) { const st = META.dec_style[dec];
    if (c.scale !== 'oracle') { const s = decSeries(c.tag, c.K, dec, met + sfx); if (s) { all.push(s); tr.push(sline(st.label, tx(s), {color: st.color, dash: st.dash}, {legendgroup: dec, error_y: undefined})); } }
    if (c.scale !== 'pred') { const s = decSeries(c.tag, c.K, dec, met + '_on' + sfx); if (s) { all.push(s); tr.push(sline(st.label + ' · oracle norm', tx(s), {color: st.color, dash: 'dash', width: 1.5}, {legendgroup: dec, error_y: undefined})); } } }
  const m = decSeries(c.tag, 0, 'mean', met + sfx); if (m) { all.push(m); tr.push(sline(META.dec_style.mean.label, tx(m), META.dec_style.mean, {error_y: undefined})); }
  const clamped = isKL && all.some(s => s.some(v => val(v) != null && val(v) <= fl));
  const lay = {height:440, xaxis: siteAxis(), title: subTitle(tagName(c.tag) + ' · K = ' + c.K + (isKL ? (med ? ' · median over held-out images' : ' · mean over held-out images') : '') + (clamped ? ' · values ≤ ' + fl + ' drawn at ' + fl : '')), legend:{orientation:'h', y:1.24, x:0}};
  if (isKL) { const t = decSeries(c.tag, 0, 'true', 'kl'); const tv = t ? t.map(val).filter(v => v != null) : [];
    const lr = logRange(all, fl); if (clamped) lr[0] = Math.min(lr[0], Math.log10(fl) - 0.15);
    lay.yaxis = merge(AXIS, {type:'log', range: lr, title:{text:'KL(clean ‖ patched), nats (log)'}});
    if (tv.length) lay.annotations = [{xref:'paper', yref:'paper', x:0.01, y:0.02, xanchor:'left', yanchor:'bottom', showarrow:false, font:{size:11.5, color:'#555'},
      text:'control: true activation patched back, |KL| ≤ ' + fmt(Math.max(...tv.map(Math.abs))) + ' at every site'}]; }
  else lay.yaxis = merge(AXIS, {range:[0, 1.02], title:{text:'top-1 agreement with the clean run'}});
  plot(el, tr, lay);
}"""
    fig(ft, 'fn_kl', fn_draw % 'true', '@dec', controls=[Radio('klstat', T('controls.stat', 'Statistic'), ['mean', 'median'],
                                                                 labels={'median': T('labels.stat_median'), 'mean': T('labels.stat_mean')})])
    fig(ft, 'fn_agree', fn_draw % 'false', '@dec')
    prose_block(ft, 'function.outro')

    # ---------------------------------------------------------------- 4 sources and round trip
    st_ = nb.tab('sources', T('tabs.sources', 'Sources and round trip'))
    prose_block(st_, 'sources.intro')
    rks = rt_Ks()
    rk_want = TEXT.get('config', {}).get('rt_K_default', 16)
    rk_def = rk_want if rk_want in rks else (rks[0] if rks else 16)
    if RT_TAGS:
        st_.controls([tag_ctl('rtag', RT_TAGS, RT_MAIN), Radio('rK', 'K', rks, default=rk_def)])
        one = r"""(el, d, c) => {
  const s = rtSeries(c.rtag, c.rK, '%s');
  if (!s) { empty(el, 'No data for this selection yet.'); return; }
  plot(el, [sline(%s, s, {color:'#0f7c7c'})], {height:330, showlegend:false, xaxis: siteAxis(),
    yaxis: merge(AXIS, {title:{text:%s}, range: %s}), title: subTitle(tagName(c.rtag) + ' · K = ' + c.rK)});
}"""
        grid(nb, st_, [
            ('src_share', '@rt', one % ('E1|share_ll', js(T('labels.share_ll')), js(T('labels.share_ll_axis')), '[0, 1]'), None),
            ('src_corr', '@rt', one % ('E1|corr', js(T('labels.corr')), js(T('labels.corr_axis')), '[0, 1.02]'), None),
        ])
        multi = r"""(el, d, c) => {
  const items = %s, tr = [], all = [];
  for (const it of items) { const s = rtSeries(c.rtag, c.rK, it.key); if (!s) continue; all.push(s);
    tr.push(sline(it.label, %s, {color: it.color, dash: it.dash, width: it.width || 2})); }
  const lay = {height:%d, xaxis: siteAxis(), title: subTitle(tagName(c.rtag) + ' · K = ' + c.rK + clampNote(all)), legend:{orientation:'h', y:1.25, x:0}};
  lay.yaxis = %s;
  %s
  plot(el, tr, lay);
}"""
        dsty = DEC_STYLE
        r3 = RT_STYLE
        src_items = [dict(key='E1|krr|srel', label=T('labels.srel_krr'), color=dsty['krr']['color'], dash='solid'),
                     dict(key='E1|linear|srel', label=T('labels.srel_linear'), color=dsty['linear']['color'], dash='solid')]
        ref1 = "withRefs(lay, [hline(1, " + js(T('labels.ref_mean_unit')) + ")]);"
        log_y = lambda title: "merge(AXIS, {type:'log', range: logRange(all.concat([[1]]), FLOOR), title:{text:%s}})" % js(title)  # noqa: E731
        log_y0 = lambda title: "merge(AXIS, {type:'log', range: logRange(all, FLOOR), title:{text:%s}})" % js(title)  # noqa: E731
        lin_y = lambda title, r='rng(all, 0, null)': "merge(AXIS, {range: %s, title:{text:%s}})" % (r, js(title))  # noqa: E731
        fig(st_, 'src_target', multi % (json.dumps(src_items), 'logPos(s, FLOOR)', 360, log_y(T('labels.srel_axis')), ref1), '@rt')
        e3 = lambda q: [dict(key=f'E3|{q}', label=r3['E3']['label'], color=r3['E3']['color'], dash='solid'),  # noqa: E731
                        dict(key=f'E3n|{q}', label=r3['E3n']['label'], color=r3['E3n']['color'], dash='dash')]
        err_items = e3('srel') + [dict(key='E2|krr|srel', label=T('labels.e2_krr'), color=dsty['krr']['color'], dash='dot'),
                                  dict(key='E2|linear|srel', label=T('labels.e2_linear'), color=dsty['linear']['color'], dash='dot')]
        grid(nb, st_, [
            ('rt_miss', '@rt', multi % (json.dumps(e3('miss')), 'logPos(s, FLOOR)', 340, log_y0(T('labels.miss_axis')), ''), None),
            ('rt_err', '@rt', multi % (json.dumps(err_items), 'logPos(s, FLOOR)', 340, log_y(T('labels.srel_axis')), ref1), None),
        ])
        joint_items = [dict(key='E3|relerr_D', label=r3['E3']['label'], color=r3['E3']['color'], dash='solid'),
                       dict(key='E3n|relerr_D', label=r3['E3n']['label'], color=r3['E3n']['color'], dash='dash'),
                       dict(key='E2|krr|relerr_D', label=T('labels.e2_krr'), color=dsty['krr']['color'], dash='dot'),
                       dict(key='E2|linear|relerr_D', label=T('labels.e2_linear'), color=dsty['linear']['color'], dash='dot')]
        fig(st_, 'rt_joint', multi % (json.dumps(joint_items), 'logPos(s, FLOOR)', 380, log_y(T('labels.joint_err_axis')), ref1), '@rt')
        grid(nb, st_, [
            ('rt_move', '@rt', multi % (json.dumps(e3('move')), 's', 340, lin_y(T('labels.move_axis')), ''), None),
            ('rt_tang', '@rt', multi % (json.dumps(e3('tang')), 's', 340, lin_y(T('labels.tang_axis'), '[0, 1]'), ''), None),
        ])
        req_items = [dict(key='E5|0.5|miss_S', label=T('labels.req05'), color='#0f7c7c', dash='solid'),
                     dict(key='E5|1.5|miss_S', label=T('labels.req15'), color='#0f7c7c', dash='dash'),
                     dict(key='E4|match|miss_S', label=T('labels.req10'), color=METH_STYLE['match']['color'], dash='dot')]
        fig(st_, 'rt_req', multi % (json.dumps(req_items), 'logPos(s, FLOOR)', 380, log_y(T('labels.miss_axis')), ref1), '@rt')
    else:
        st_.prose('<p class="empty">No round-trip data in results.json yet.</p>')
    prose_block(st_, 'sources.outro')

    # ---------------------------------------------------------------- 5 edits
    et = nb.tab('edits', T('tabs.edits', 'Edits'))
    prose_block(et, 'edits.intro')
    meths_all = []
    for t in RT_TAGS:
        for e in R['rt'][t].values():
            for k in e:
                p = k.split('|')
                if p[0] == 'E4' and len(p) == 3 and not p[1].startswith('rand~') and p[1] not in meths_all:
                    meths_all.append(p[1])
    order = [m for m in METH_STYLE if m in meths_all] + [m for m in meths_all if m not in METH_STYLE]
    meta_m = {m: METH_STYLE.get(m, dict(color='#777777', label=m)) for m in order}
    if RT_TAGS:
        et.controls([tag_ctl('rtag', RT_TAGS, RT_MAIN), Radio('rK', 'K', rks, default=rk_def),
                     Radio('rand', T('controls.rand', 'Random controls'), ['show', 'hide'])])
        edit_draw = r"""(el, d, c) => {
  const q = '%s', methods = %s, styles = %s, tr = [], all = [], withRand = %s && c.rand === 'show';
  for (const m of methods) { const st = styles[m], s = rtSeries(c.rtag, c.rK, 'E4|' + m + '|' + q); if (!s) continue; all.push(s);
    tr.push(sline(st.label, %s, {color: st.color, dash: m === 'none' ? 'dot' : 'solid', width: m === 'swap' ? 1.6 : 2.2}, {legendgroup: m}));
    if (withRand) { const r = rtSeries(c.rtag, c.rK, 'E4|rand~' + m + '|' + q); if (r) { all.push(r);
      tr.push(sline('random, same size', %s, {color: st.color, dash:'dot', width:1.3, msize:3}, {legendgroup: m, showlegend: false, hovertemplate: 'random edit, size of ' + st.label + ' · %%{x}: %%{y:.3~g}<extra></extra>'})); } } }
  const lay = {height:%d, xaxis: siteAxis(), title: subTitle(tagName(c.rtag) + ' · K = ' + c.rK + (withRand ? ' · dotted: random edit of the same size' : '') + (q === 'miss_S' ? clampNote(all) : '')), legend:{orientation:'h', y:1.26, x:0}};
  lay.yaxis = %s;
  %s
  plot(el, tr, lay);
}"""
        ms = json.dumps(order)
        sty = json.dumps(meta_m)
        no_swap = json.dumps([m for m in order if m not in ('swap', 'none')])
        fig(et, 'edit_pb', edit_draw % ('p_b', ms, sty, 'true', 's', 'r', 460, "merge(AXIS, {range:[0, 1.02], title:{text:" + js(T('labels.p_b')) + "}})", ''), '@rt')
        grid(nb, et, [
            ('edit_size', '@rt', edit_draw % ('size', no_swap, sty, 'false', 's', 'r', 380, "merge(AXIS, {range: rng(all, 0, null), title:{text:" + js(T('labels.size')) + "}})", ''), None),
            ('edit_progS', '@rt', edit_draw % ('prog_S', no_swap, sty, 'true', 's', 'r', 380, "merge(AXIS, {range: rng(all, -0.05, 1.05), title:{text:" + js(T('labels.prog_S')) + "}})", ''), None),
        ])
        fig(et, 'edit_pa', edit_draw % ('p_a', ms, sty, 'true', 's', 'r', 460, "merge(AXIS, {range:[0, 1.02], title:{text:" + js(T('labels.p_a')) + "}})", ''), '@rt')
        fig(et, 'edit_missS', edit_draw % ('miss_S', no_swap, sty, 'true', 'logPos(s, FLOOR)', 'logPos(r, FLOOR)', 400,
                                           "merge(AXIS, {type:'log', range: logRange(all.concat([[2]]), FLOOR), title:{text:" + js(T('labels.miss_S')) + "}})",
                                           "withRefs(lay, [hline(2, " + js(T('labels.ref_unrelated')) + ")]);"), '@rt')
        hm_opts = [m for m in order if m not in ('none',)]
        hm_a = 'match' if 'match' in hm_opts else hm_opts[0]
        hm_b = 'resid' if 'resid' in hm_opts else hm_opts[-1]
        fig(et, 'edit_heat', r"""(el, d, c) => {
  const S = META.sites, styles = %s, tr = [], lay = {height:470, annotations:[], margin:{l:64, r:70, t:60, b:56}};
  const ms = [c.hmA, c.hmB], dom = facetAxes(2, 0.1); let lo = Infinity, hi = -Infinity;
  ms.forEach((m, i) => { const ax = i ? '2' : '';
    const z = S.map((l, li) => S.map((j, ji) => { if (ji < li) return null;
      const s = rtSeries(c.rtag, c.rK, 'E4|' + m + '|' + (ji === li ? 'prog_D' : 'prog_D@' + j)); const v = s ? s[li] : null;
      if (v != null) { lo = Math.min(lo, v); hi = Math.max(hi, v); } return v; }));
    tr.push({type:'heatmap', x: S, y: S, z, zmin:0, zmax:1, zauto:false, colorscale:'Viridis', xaxis:'x' + ax, yaxis:'y' + ax,
      showscale: i === 1, colorbar:{title:{text:'progress', side:'right'}, x:1.01, len:0.9}, xgap:1, ygap:1,
      hovertemplate: styles[m].label + '<br>edited at %%{y}, read at %%{x}: %%{z:.3f}<extra></extra>'});
    lay['xaxis' + ax] = merge(AXIS, {type:'category', domain: dom[i], anchor:'y' + ax, title:{text:'site where the patched run is read'}, tickangle:-45, showgrid:false});
    lay['yaxis' + ax] = merge(AXIS, {type:'category', autorange:'reversed', anchor:'x' + ax, title:{text: i === 0 ? 'edited site' : ''}, showticklabels: true, showgrid:false});
    lay.annotations.push({xref:'paper', yref:'paper', x: (dom[i][0] + dom[i][1]) / 2, y: 1.02, xanchor:'center', yanchor:'bottom', text:'<b>' + styles[m].label + '</b>', showarrow:false}); });
  if (isFinite(lo)) lay.annotations.push({xref:'paper', yref:'paper', x:0, y:-0.24, xanchor:'left', showarrow:false, font:{size:11.5, color:'#555'},
    text:'colour capped at [0, 1]; values shown range from ' + fmt(lo) + ' to ' + fmt(hi) + ' (hover for each cell) · ' + tagName(c.rtag) + ' · K = ' + c.rK});
  lay.margin.b = 110;
  plot(el, tr, lay);
}""" % sty, '@rt', controls=[Select('hmA', T('controls.hm_left', 'Left'), hm_opts, default=hm_a, labels={m: meta_m[m]['label'] for m in hm_opts}),
                             Select('hmB', T('controls.hm_right', 'Right'), hm_opts, default=hm_b, labels={m: meta_m[m]['label'] for m in hm_opts})])
    prose_block(et, 'edits.outro')

    # ---------------------------------------------------------------- 6 out of distribution
    ot = nb.tab('ood', T('tabs.ood', 'Out of distribution'))
    prose_block(ot, 'ood.intro')
    ot.controls([tag_ctl(), Radio('K', 'K', dec_ks, default=k_def)])
    ood_draw = r"""(el, d, c) => {
  const met = '%s', D = d[c.tag] || {}, tr = [], all = [];
  for (const set of Object.keys(META.set_style)) { const st = META.set_style[set];
    const s = D[set] && D[set][String(c.K)] && D[set][String(c.K)].krr ? D[set][String(c.K)].krr[met] : null; if (!s) continue; all.push(s);
    tr.push(sline(st.label, met === 'kl' ? logPos(s) : s, {color: st.color, width: set === 'H' ? 3 : 2}, met === 'kl' ? {error_y: undefined} : {})); }
  if (met === 'kl' && D.H && D.H['0'] && D.H['0'].mean) { const m = D.H['0'].mean.kl; all.push(m);
    tr.push(sline('probe mean patched (H)', logPos(m), META.dec_style.mean, {error_y: undefined})); }
  const lay = {height:440, xaxis: siteAxis(), title: subTitle(tagName(c.tag) + ' · kernel ridge · K = ' + c.K), legend:{orientation:'h', y:1.22, x:0}};
  lay.yaxis = met === 'kl' ? merge(AXIS, {type:'log', range: logRange(all, 1e-8), title:{text:'KL(clean ‖ patched), nats (log)'}})
                           : merge(AXIS, {range: rng(all, 0, null), title:{text:'cosine (true, decoded)'}});
  plot(el, tr, lay);
}"""
    fig(ot, 'ood_cos', ood_draw % 'cos_raw', '@ood')
    fig(ot, 'ood_kl', ood_draw % 'kl', '@ood')
    prose_block(ot, 'ood.outro')

    # ---------------------------------------------------------------- 7 adversarial
    at = nb.tab('adversarial', T('tabs.adversarial', 'Adversarial'))
    prose_block(at, 'adversarial.intro')
    adv_keys = sorted(R.get('adv', {}), key=lambda k: tag_order(k.rsplit('_K', 1)[0]) + (k,))
    if adv_keys:
        adv_def = next((k for k in adv_keys if MAIN and k.startswith(MAIN + '_')), adv_keys[0])
        adv_lab = {k: f'{tag_label(k.rsplit("_K", 1)[0])} · K = {k.rsplit("_K", 1)[1]}' for k in adv_keys}
        atts = sorted({a.replace('_src', '') for v in R['adv'].values() for a in v})
        at.controls([Select('akey', T('controls.adv', 'Embedding'), adv_keys, default=adv_def, labels=adv_lab)])
        fig(at, 'adv_disp', r"""(el, d, c) => {
  const D = d[c.akey] || {}, tr = [], all = [[2]], lab = %s;
  for (const a of %s) for (const v of ['', '_src']) { const s = D[a + v]; if (!s) continue; const st = META.set_style[a] || {color:'#777', label:a};
    all.push(s.disp); tr.push(sline(st.label + ' · ' + (v ? lab.src : lab.joint), logPos(s.disp), {color: st.color, dash: v ? 'dash' : 'solid'})); }
  const lay = withRefs({height:400, xaxis: siteAxis(), yaxis: merge(AXIS, {type:'log', range: logRange(all, 0), title:{text:%s}}),
    title: subTitle(%s.replace('{k}', META.adv_label[c.akey] || c.akey) + ' · solid: joint, dashed: source')}, [hline(2, %s)]);
  plot(el, tr, lay);
}""" % (js({'joint': T('labels.adv_joint'), 'src': T('labels.adv_src')}), json.dumps(atts), js(T('labels.disp')), js(T('labels.adv_sub')),
        js(T('labels.ref_unrelated'))), '@adv')
        fig(at, 'adv_nc', r"""(el, d, c) => {
  const D = d[c.akey] || {}, tr = [], lab = %s, cl = %s, col = {clean_true:'#111111', adv_true:'#4477aa', adv_follows_attack:'#d0473f', clean_matches_pred:'#999999'};
  for (const v of ['', '_src']) { const s = D[c.natt + v]; if (!s) continue;
    for (const f of Object.keys(lab)) if (s[f]) tr.push(sline(lab[f] + ' · ' + (v ? cl.src : cl.joint), s[f], {color: col[f], dash: v ? 'dash' : 'solid', width: f.startsWith('clean') ? 1.5 : 2.2})); }
  const lay = withRefs({height:440, xaxis: siteAxis(), yaxis: merge(AXIS, {range:[0, 1.02], title:{text:%s}}),
    title: subTitle(%s.replace('{k}', META.adv_label[c.akey] || c.akey) + ' · ' + c.natt + ' · solid: joint, dashed: source')}, [hline(0.1, %s)]);
  plot(el, tr, lay);
}""" % (json.dumps({'clean_true': T('labels.nc_clean_true'), 'clean_matches_pred': T('labels.nc_clean_pred'),
                    'adv_true': T('labels.nc_adv_true'), 'adv_follows_attack': T('labels.nc_adv_follow')}),
        js({'joint': T('labels.adv_joint'), 'src': T('labels.adv_src')}), js(T('labels.nc_axis')), js(T('labels.adv_sub')),
        js(T('labels.ref_chance'))), '@adv',
            controls=[Radio('natt', T('controls.attack', 'Attack'), atts, default='pgd2' if 'pgd2' in atts else atts[0],
                            labels={a: SET_STYLE.get(a, {}).get('label', a) for a in atts})])
    else:
        at.prose('<p class="empty">No adversarial summaries in results.json yet.</p>')
    fig_tags = sorted(imgs, key=tag_order)          # participation ratio first, then the larger probe
    if fig_tags:
        fig_def = fig_tags[0]
        img_ks = sorted({int(K) for t in imgs.values() for n in t.values() for K in n}) or [64]
        k_img = 64 if 64 in img_ks else img_ks[-1]
        ftl = {t: tag_label(t) + (' · 13 sites (pixels as a site)' if t.endswith('L13') else '') for t in fig_tags}
        adv_atts = sorted({re.match(r'(pgd\d+)', n).group(1) for t in imgs.values() for n in t if re.match(r'pgd\d+(src)?$', n)})
        if adv_atts:
            img_draw = r"""(el, d, c) => { const g = ((d[c.ftag] || {})['%s'.replace('{a}', c.att)] || {})[String(c.iK)]; imgGrid(el, g); }"""
            at.controls([Select('ftag', T('controls.figtag', 'Image grids from'), fig_tags, default=fig_def, labels=ftl),
                         Radio('iK', 'K', img_ks, default=k_img), Radio('att', T('controls.attack', 'Attack'), adv_atts, default='pgd8' if 'pgd8' in adv_atts else adv_atts[0])])
            fig(at, 'adv_img_src', img_draw % '{a}src', '@img')
            fig(at, 'adv_img_joint', img_draw % '{a}', '@img')
    prose_block(at, 'adversarial.outro')

    # ---------------------------------------------------------------- 8 pixels
    pt = nb.tab('pixels', T('tabs.pixels', 'Pixels'))
    prose_block(pt, 'pixels.intro')
    if fig_tags:
        pt.controls([Select('ftag', T('controls.figtag', 'Image grids from'), fig_tags, default=fig_def, labels=ftl),
                     Radio('iK', 'K', img_ks, default=k_img)])
        g = r"""(el, d, c) => { const g = ((d[c.ftag] || {})[%s] || {})[String(c.iK)]; imgGrid(el, g); }"""
        grid(nb, pt, [('px_depth', '@img', g % "'depth'", None), ('px_depthsrc', '@img', g % "'depthsrc'", None)])
        path_sites = sorted({n.split('_', 1)[1] for t in imgs.values() for n in t if n.startswith('path_')},
                            key=lambda s: (['pixels'] + SITES).index(s) if s in ['pixels'] + SITES else 99)
        if path_sites:
            fig(pt, 'px_path', r"""(el, d, c) => { const g = ((d[c.ftag] || {})[(c.pcoord === 'src' ? 'pathsrc_' : 'path_') + c.psite] || {})[String(c.iK)]; imgGrid(el, g); }""",
                '@img', controls=[Radio('psite', T('controls.path_site', 'Site'), path_sites, default='b9' if 'b9' in path_sites else path_sites[0]),
                                  Radio('pcoord', T('controls.coords', 'Coordinates'), ['joint', 'src'], labels={'joint': 'joint', 'src': 'source'})])
        sets = [s for s in ('c100', 'svhn', 'noise20') if any(s in t for t in imgs.values())]
        if sets:
            pt.h(T('pixels.h_other'), 3)
            pt.controls([Radio('oset', T('controls.dataset', 'Dataset'), sets, labels={s: SET_STYLE[s]['label'] for s in sets})])
            grid(nb, pt, [('px_other', '@img', g % 'c.oset', None), ('px_othersrc', '@img', g % "c.oset + 'src'", None)])
    else:
        pt.prose('<p class="empty">No image grids in data/invert/figs yet.</p>')
    prose_block(pt, 'pixels.outro')

    # ---------------------------------------------------------------- 9 method and caveats
    mt = nb.tab('method', T('tabs.method', 'Method and caveats'))
    mt.h(T('method.h_recipe'))
    prose_block(mt, 'method.recipe_intro')
    rec = TEXT.get('method', {}).get('recipe', {})
    mt.table([fill(h) for h in rec.get('header', [])], [[fill(c) for c in r] for r in rec.get('rows', [])])
    mt.h(T('method.h_decisions'))
    prose_block(mt, 'method.decisions_intro')
    dec_t = TEXT.get('method', {}).get('decisions', {})
    mt.table([fill(h) for h in dec_t.get('header', [])], [[fill(c) for c in r] for r in dec_t.get('rows', [])])
    mt.h(T('method.h_caveats'))
    mt.prose('<ul>' + ''.join(f'<li>{fill(x)}</li>' for x in TEXT.get('method', {}).get('caveats', [])) + '</ul>')
    mt.h(T('method.h_files'))
    prose_block(mt, 'method.files')

    nb.write(OUT)
    # placeholder catalogue for the writer
    cat = {'named': NAMED, 'tags': {'main': MAIN, 'rt_main': RT_MAIN, 'by_probe': MAIN_BY_PROBE, 'decode': DEC_TAGS, 'rt': RT_TAGS},
           'used': sorted(USED), 'grammar': __doc__.split('Numbers quoted in prose are placeholders')[1].split('"""')[0]}
    (OUT.parent / 'text_keys.json').write_text(json.dumps(cat, indent=1, ensure_ascii=False))
    for w in dict.fromkeys(WARN):
        print('  note:', w)


def status_html(key):
    """Experiment-map status, computed from the data present."""
    def badge(txt, done):
        return f'<span class="status {"done" if done else "open"}">{txt}</span>'
    if key == 'kernel':
        n = sum(1 for s in SITES if s in R['kernel'])
        return badge(f'{n}/{len(SITES)} sites' + (' · refit ' + ('done' if R.get('refit') else 'missing')), n == len(SITES) and bool(R.get('refit')))
    if key in ('decode', 'function', 'ood'):
        bits = [f'{probe_of(t)} {kernel_of(t)}: {n_sites_dec(t)}/{len(SITES)}' for t in DEC_TAGS]
        bits += [f'{t_short(t)}: pending' for t in sorted(set(R['decode']) - set(DEC_TAGS), key=tag_order)]
        return badge(' · '.join(bits), all(n_sites_dec(t) == len(SITES) for t in DEC_TAGS))
    if key in ('rt', 'edits'):
        bits = [f'{probe_of(t)} {kernel_of(t)}: {n_sites_rt(t)}/{len(SITES)}' for t in RT_TAGS] or ['no data']
        return badge(' · '.join(bits), bool(RT_TAGS) and all(n_sites_rt(t) == len(SITES) for t in RT_TAGS))
    if key == 'modeloc':
        ml = sorted(R.get('modeloc', {}), key=tag_order)
        return badge(', '.join(t_short(t) for t in ml) or 'no data', bool(ml))
    if key == 'adv':
        return badge(', '.join(sorted(R.get('adv', {}))) or 'no data', bool(R.get('adv')))
    if key == 'pixels':
        tags = sorted({m.group(1) for p in (DATA / 'figs').glob('*.png') if (m := FIG_RE.match(p.name))})
        return badge(', '.join(tags) or 'no grids', bool(tags))
    return ''


if __name__ == '__main__':
    main()
