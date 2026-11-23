#!/usr/bin/env python3
"""Audit experiment headlines without importing site generators or running experiments.

R = derive from saved raw coordinates/parameters (or inspect the actual shipped
sample for a shipping claim); P = aggregate saved scores/measurements; S =
check a transcription only. All cited paths are relative to the repository.
"""
import argparse
import ast
import gzip
import hashlib
import html
import json
import math
import re
from collections import Counter
from functools import cache
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import orthogonal_procrustes, solveh_banded
from scipy.sparse import csr_matrix
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
ASSETS = 'site/assets/x'
FRAG = 'site/content/experiments'
BUILD = 'build/site/index.html'
INV = 'data/invert'
FETCH = 'data/fetch'
IV = 'crw-depth-invert-results'
XV = 'crw-depth-xp-results'
P2 = 'P2__pr300__L12'
P8 = 'P8__pr400__L12'
JOINT = f'{IV}/joint/{P2}'
PKG = 'data/sitex/ops'
X3 = 'xp/out/x3'
X4 = 'xp/out/x4'
SITES = ('stem', *(f'b{i}' for i in range(1, 10)), 'pool', 'logits')
ENTRIES = ('inv-kernel', 'inv-sources', 'inv-decode', 'inv-wavelets', 'inv-ood', 'inv-edits',
           'x1-data', 'x3-adversarial', 'x4-paths', 'solver')
TOKEN = re.compile(r'\{\{\s*fact:([^}]+?)\s*\}\}')


def text(path):
    return (ROOT / path).read_text(encoding='utf-8')


@cache
def read_json(path):
    return json.loads(text(path))


@cache
def staged(rel):
    path = f'{FETCH}/{rel}'
    pin = read_json(f'{FETCH}/READY.json')[rel]
    with (ROOT / path).open('rb') as f:
        if (ROOT / path).stat().st_size != pin['bytes'] or hashlib.file_digest(f, 'sha256').hexdigest() != pin['sha256']:
            raise ValueError(f'{path}: does not match READY.json')
    return path


def arrays(path, *keys):
    with np.load(ROOT / path, allow_pickle=False) as z:
        return tuple(z[k] for k in keys)


def array(path, key):
    return arrays(path, key)[0]


def first_modes(path, key, k=16):
    # Copy only the working subset; do not retain the original backing array.
    return array(path, key)[..., :k].astype(np.float64, copy=True)


def packaged_array(meta_path, kind):
    spec = read_json(meta_path)['kinds'][kind]
    path = str(Path(meta_path).parent / spec['file'])
    with gzip.open(ROOT / path, 'rb') as f:
        a = np.frombuffer(f.read(), dtype={'i16': '<i2', 'i8': 'i1'}[spec['dtype']]).reshape(spec['shape'])
    scale = np.asarray(spec['scale'])
    return a * scale.reshape((a.shape[0],) + (1,) * (a.ndim - 2) + (a.shape[-1],)), path


def source_constant(path, name):
    for s in ast.parse(text(path)).body:
        if isinstance(s, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in s.targets):
            return ast.literal_eval(s.value)
    raise ValueError(f'{path}: configuration variable {name} missing')


def energy_fraction(delta, plane):
    return np.linalg.norm(delta @ plane)**2 / np.linalg.norm(delta)**2


def depth_metrics(c, t, mu):
    a = .5 / c
    deg = np.r_[1, np.full(len(c)-2, 2), 1]
    condition, gain = [], []
    for m in mu:
        diag = a + t*deg - m
        off = np.full(len(c)-1, -t)
        depth = np.diag(diag) + np.diag(off, 1) + np.diag(off, -1)
        spectrum = np.sort(abs(np.linalg.eigvalsh(depth)))
        condition.append(spectrum[-1] / spectrum[0])
        gain.append(np.linalg.norm(np.linalg.solve(depth, np.diag(a)), ord=np.inf))
    return np.array(condition), np.array(gain)


def top_two_shares(path):
    v = first_modes(path, 'vecs')
    energy = np.einsum('lnk,lnk->nk', v, v)
    return np.sort(energy, axis=0)[-2:].sum(axis=0) / energy.sum(axis=0)


def top_two(path):
    return top_two_shares(path).max()


@cache
def probe_gram_metrics():
    """Rebuild the experiment's PR walk and count k-NN walk storage separately."""
    files = [f'data/grams/r20s0b/gram_64000_{l}.npy' for l in range(len(SITES))]
    raw_hubs, pr_hubs, pr_entries = [], [], []
    dense_bytes = compact_bytes = 0
    for f in files:
        cosine = np.load(ROOT / f, mmap_mode='r', allow_pickle=False)
        n = len(cosine)
        if cosine.shape != (n, n):
            raise ValueError(f'{f}: expected a square Gram')
        nn = np.empty((n, 15), dtype=np.int32)
        q = np.empty((n, n), dtype=np.float64)
        for lo in range(0, n, 128):
            rows = slice(lo, min(lo + 128, n))
            d = 1 - np.asarray(cosine[rows])
            d[np.arange(len(d)), np.arange(lo, rows.stop)] = np.inf
            nn[rows] = np.argpartition(d, 15, axis=1)[:, :15]
            d -= np.min(d, axis=1, keepdims=True)
            # PR beta can be found directly from (sum w)^2 / sum(w^2) = 300.
            # The nearest allowed point has w=1, so underflow is harmless.
            low, high = np.zeros((len(d), 1)), np.full((len(d), 1), 512.)
            for _ in range(36):
                mid = (low + high) / 2
                w = np.exp(-mid * d)
                pr = w.sum(axis=1)**2 / np.einsum('ij,ij->i', w, w)
                low = np.where(pr[:, None] > 300, mid, low)
                high = np.where(pr[:, None] > 300, high, mid)
            w /= w.sum(axis=1, keepdims=True)
            np.testing.assert_allclose(1 / np.einsum('ij,ij->i', w, w), 300, rtol=1e-7)
            q[rows] = w
        raw_hubs.append(np.bincount(nn.ravel(), minlength=n).max())
        edge = csr_matrix((np.ones(nn.size, dtype=bool), (np.repeat(np.arange(n), 15), nn.ravel())), shape=(n, n))
        undirected = edge.maximum(edge.T)
        # Sparse lazy walk: one f64 + one column int32 per union edge or
        # diagonal, N+1 int32 CSR offsets and one f64 rank-one vector.
        dense_bytes += n*n*8
        compact_bytes += (undirected.nnz + n)*12 + (n + 1)*4 + n*8
        p = (q + q.T) * .5
        del q
        scale = np.ones(n)
        for _ in range(400):
            row_sums = scale * np.einsum('ij,j->i', p, scale)
            if np.max(np.abs(row_sums - 1)) < 1e-11:
                break
            scale *= row_sums ** -.5
        else:
            raise ValueError(f'{f}: PR scaling did not converge')
        p *= scale[:, None]
        p *= scale[None, :]
        pr_entries.append(p.max())
        np.fill_diagonal(p, -np.inf)
        closest = np.argpartition(-p, 15, axis=1)[:, :15]
        pr_hubs.append(np.bincount(closest.ravel(), minlength=n).max())
        del p
    return files, raw_hubs, pr_hubs, pr_entries, dense_bytes*1e-6, compact_bytes*1e-6


def kernel(add, note):
    split = staged(f'{IV}/splits.npz')
    p2, p8, H, lab = arrays(split, 'P2', 'P8', 'H', 'labels')
    add('probe.p2', p2.size, 'R', split)
    per_class = np.unique(np.bincount(lab[p2]))
    assert len(per_class) == 1
    add('probe.per_class', per_class[0], 'R', split)
    add('probe.p8_extra', len(set(p8) - set(p2)), 'R', split)
    add('probe.held', len(H), 'R', split)
    for name, op_id, rel in [('knn15', 'r20s0-c10-knn15', f'{XV}/x3/op_c10-s0_knn15.npz'),
                             ('ea300', 'r20s0-p2-ea300', f'{IV}/joint/P2__ea300__L12/modes.npz'),
                             ('pr300', 'r20s0-p2-pr300', f'{JOINT}/modes.npz')]:
        source = staged(rel)
        local = top_two_shares(source)
        add(f'top2.p2.{name}', local.max(), 'R', source)
        pkg = f'{PKG}/{op_id}/op.json'
        plane = np.asarray(read_json(pkg)['axes']['pooled'])[:, :2]
        add(f'display.{name}', np.vdot(plane[local.argmax()], plane[local.argmax()]), 'R', source, pkg)
    refit_path = f'{INV}/refit.json'
    refit = read_json(refit_path)
    for k in (16, 64):
        for rule in ('ea300', 'pr300'):
            held = refit[f'{rule}|{k}']['held']
            add(f'refit.k{k}.{rule}.min', min(held), 'S', refit_path)
            add(f'refit.k{k}.{rule}.max', max(held), 'S', refit_path)
    comparison = np.asarray(refit['ea300|16']['held']) - refit['pr300|16']['held']
    add('refit.k16.shannon_wins', np.count_nonzero(comparison > 0), 'S', refit_path)
    loc = f'{INV}/modeloc.json'
    peaks = read_json(loc)['P8__ea400__L12']['top2']
    add('top2.p8.ea400', max(peaks[:16]), 'S', loc)
    add('top2.p8.ea400_count', sum(x > .10 for x in peaks), 'S', loc)
    files, raw_hubs, pr_hubs, pr_entries, _, _ = probe_gram_metrics()
    add('hub.raw_max', max(raw_hubs), 'R', *files)
    add('hub.raw_min', min(raw_hubs), 'R', *files)
    add('hub.pr300_max', max(pr_hubs), 'R', *files)
    add('row.pr300_final_max', max(pr_entries), 'R', *files)


def sources(add, note):
    op = staged(f'{JOINT}/modes.npz')
    c, t, mu, chain = arrays(op, 'c', 't', 'vals', 'sites')
    chain = chain.tolist()
    assert tuple(chain) == SITES
    cond, gain = depth_metrics(c, t, mu)
    for k in (16, 64, 256):
        add(f'depth.k{k}.condition', cond[:k].max(), 'R', op)
        add(f'depth.k{k}.gain', gain[:k].max(), 'R', op)
    before = staged(f'{JOINT}/coords_H.npz')
    after = staged(f'{JOINT}/coords_pgd2.npz')
    y1, s1 = (first_modes(after, k) for k in ('y', 'onestep'))
    n = y1.shape[1]
    y_full, s_full = (first_modes(before, k) for k in ('y', 'onestep'))
    y0, s0 = y_full[:, :n], s_full[:, :n]
    source_delta = s1 - s0
    cut_delta = source_delta.copy()
    cut_delta[chain.index('b7'):] = 0
    # Solve the depth system as a symmetric tridiagonal problem, with identity
    # columns for the filter weights and image columns for the two deltas.
    a = .5 / c
    deg = np.r_[1, np.full(len(c) - 2, 2), 1]
    replay, cut = np.empty_like(y0), np.empty_like(y0)
    shares = []
    for k in range(16):
        band = np.vstack([a + t * deg - mu[k], np.r_[np.full(len(c) - 1, -t), 0]])
        rhs = np.column_stack((np.diag(a), a[:, None] * source_delta[:, :, k],
                               a[:, None] * cut_delta[:, :, k]))
        w, u, v = np.split(solveh_banded(band, rhs, lower=True), [len(c), len(c) + n], axis=1)
        replay[:, :, k], cut[:, :, k] = u, v
        shares.append(np.abs(w.diagonal()) / np.abs(w).sum(axis=1))
    np.testing.assert_allclose(replay, y1 - y0, atol=1e-8, rtol=0)

    def disp(delta, ref, site):
        i = chain.index(site)
        return np.square(delta[i] / np.std(ref[i], axis=0)).mean()

    for kind, delta, base, other, sites in [
        ('joint', y1-y0, y0, y_full, ('b5',)),
        ('source', source_delta, s0, s_full, ('b5', 'b6', 'b8', 'b9')),
        ('truncated', cut, y0, y_full, ('b5',)),
    ]:
        for site in sites:
            val = disp(delta, base, site)
            add(f'attack.{kind}.{site}', val, 'R', before, after, *([op] if kind == 'truncated' else []))
            if site == 'b5':
                note(f'{kind} b5 spread: paired clean H = {val:.8g}; all H = {disp(delta, other, site):.8g}. '
                     'The experiment uses the paired clean reference (invert/adv.py).')
    add('attack.n', n, 'R', after)
    root = f'{PKG}/r20s0-p2-pr300'
    shipped, op_meta = f'{root}/ov/pgd2.json', f'{root}/op.json'
    plane = np.asarray(read_json(op_meta)['axes']['pooled'])[:, :2]
    for kind, delta in [('joint', y1-y0), ('source', source_delta)]:
        add(f'display.pgd2.{kind}', energy_fraction(delta[chain.index('b5')], plane), 'R', before, after, op_meta)
    src = f'{ASSETS}/inv-sources.json'
    add('attack.arrows', read_json(src)['emb_lead']['arrows'], 'S', src)
    add('attack.shown', len(read_json(shipped)['images']['ids']), 'R', shipped)
    conv = SITES[:SITES.index('pool')]
    rt_paths = [f'{INV}/rt/{P8}/{s}__K16.json' for s in conv]
    for tag, weights, paths, tier in [('p2', np.mean(shares, axis=0)[:len(conv)], [op], 'R'),
                                    ('p8', [read_json(p)['E1|share_ll'] for p in rt_paths], rt_paths, 'S')]:
        for name, f in [('min', min), ('max', max)]:
            add(f'share.{tag}.conv_{name}', f(weights), tier, *paths)
    b5 = rt_paths[conv.index('b5')]
    add('patch.b5.swap', read_json(b5)['E4|swap|prog_D'], 'S', b5)


def decode_summary(add, specs):
    p = f'{INV}/decode/{P8}/summary.json'
    j = read_json(p)
    for fact, site, key in specs:
        add(fact, j[f'{site}__{site}'][key][0], 'S', p)


def decode(add, note):
    p = staged(f'{IV}/decode/{P2}/b4__b4.npz')
    cos_raw, cap_raw = arrays(p, 'H|64|krr|cos_raw', 'H|64|krr|cap_raw')
    add('small.b4', np.mean(cos_raw, dtype=np.float64), 'P', p)
    hc = staged(f'{JOINT}/coords_H.npz')
    max_p = array(hc, 'rowmax').max(axis=0)
    flagged = max_p > .5
    add('near.count', np.count_nonzero(flagged), 'P', hc)
    for name, mask in [('flagged', flagged), ('remainder', ~flagged)]:
        add(f'tail.{name}.n', np.count_nonzero(mask), 'P', hc)
        for metric, v in [('cosine', cos_raw), ('captured', cap_raw)]:
            add(f'tail.{name}.{metric}', np.average(v[mask].astype(float)), 'P', p, hc)
    root = f'{PKG}/r20s0-p2-pr300'
    package, h_pkg = f'{root}/op.json', f'{root}/ov/H.json'
    plane = np.asarray(read_json(package)['axes']['pooled'])[:, :2]
    hp, qfile = packaged_array(h_pkg, 'joint')
    b4 = hp[SITES.index('b4')]
    shown = flagged[read_json(h_pkg)['sample']['pos']]
    add('near.sample', len(b4), 'R', h_pkg, qfile)
    add('near.shown', shown.sum(), 'R', h_pkg, hc)
    add('near.projected_energy', energy_fraction(b4, plane), 'R', package, h_pkg, qfile)
    rad = np.linalg.norm(b4 @ plane, axis=1)
    add('near.projected_top', shown[np.argsort(rad)[-shown.sum():]].sum() if shown.any() else 0,
        'R', package, h_pkg, qfile, hc)
    d = array(hc, 'y')[SITES.index('b4'), :, :64]
    add('near.top20', flagged[np.argsort(np.linalg.norm(d, axis=1))[-20:]].sum(), 'R', hc)
    split, image_archive = staged(f'{IV}/splits.npz'), staged(f'{IV}/acts/c10/meta.npz')
    probes, held = arrays(split, 'P2', 'H')
    images = array(image_archive, 'images')
    probe_pixels = {images[j].tobytes() for j in probes}
    add('tail.identical', sum(images[i].tobytes() in probe_pixels for i in held), 'R', split, image_archive)
    experiment = 'invert/modal_invert.py'
    shape = source_constant(experiment, 'SHAPES')
    for site in ('b4', 'pool', 'logits'):
        add(f'dimension.{site}', math.prod(shape[site]), 'S', experiment)
    decode_summary(add, [('large.b4', 'b4', 'H|64|krr|cos_raw'),
                         ('large.pool', 'pool', 'H|64|krr|cos_raw'),
                         ('large.pool_pca', 'pool', 'H|64|pca|cos_raw'),
                         ('large.b4_agree', 'b4', 'H|64|krr|agree'),
                         ('large.b4_mlp', 'b4', 'H|64|mlp|agree')])


def wavelets(add, note):
    """Diffusion-wavelet inversion: recompute from the per-site stage outputs (the generator reads the collected
    data/invert/dw_results.json). Means from the float32 per-image exports carry ~1e-7 rounding."""
    base = f'{INV}/dw/{P8}'
    js = {s: read_json(f'{base}/{s}.json') for s in SITES}
    f32 = dict(raw_atol=2e-6, raw_reason='float32 per-image export')

    def mean(npz, key):
        return float(np.mean(array(npz, key), dtype=np.float64))

    wl, rl, mid = [], [], []
    for s in SITES:
        npz = f'{base}/{s}.npz'
        for cell in ('joint|64', 'source|64'):
            lead = mean(npz, f'q1|{cell}|lead|None|cos')
            wl.append(mean(npz, f'q1|{cell}|wbest|None|cos') - lead)
            d = mean(npz, f'q1|{cell}|ridge|cos') - lead
            rl.append(d)
            if cell == 'joint|64' and s in ('b4', 'b5', 'b6', 'b7', 'b8'):
                mid.append(d)
    p = f'{base}/<site>.npz'
    add('q1.wbest.lo', min(wl), 'P', p, **f32)
    add('q1.wbest.hi', max(wl), 'P', p, **f32)
    add('q1.ridge.lo', min(rl), 'P', p, **f32)
    add('q1.ridge.hi', max(rl), 'P', p, **f32)
    add('q1.ridge.b4b8_lo', min(mid), 'P', p, **f32)
    add('q1.ridge.b4b8_hi', max(mid), 'P', p, **f32)

    def level(s, A):
        return next(lv for lv in js[s]['q2']['levels'] if lv['A'] == A)
    early, conv = SITES[:9], SITES[1:9]
    pj = f'{base}/<site>.json'
    ks = [level(s, 2.0)['k'] for s in early]
    add('q2.a2.k_lo', min(ks), 'S', pj)
    add('q2.a2.k_hi', max(ks), 'S', pj)
    cos2 = [mean(f'{base}/{s}.npz', f"q2|{level(s, 2.0)['k']}|cos") for s in early]
    add('q2.a2.cos_lo', min(cos2), 'P', p, pj, **f32)
    add('q2.a2.cos_hi', max(cos2), 'P', p, pj, **f32)
    rt32 = [level(s, 32.0)['rt']['total'] for s in conv]
    add('q2.a32.rt_lo', min(rt32), 'S', pj)
    add('q2.a32.rt_hi', max(rt32), 'S', pj)

    def e256(s, key):
        e = js[s]['q3']['energy'][key]
        return e['share'][e['n'].index(256)]
    gap = [e256(s, 'eig|None') - e256(s, 'qr|None') for s in SITES]
    qr_rand = [e256(s, 'qr|None') - np.mean([e256(s, f'rand{i}|None') for i in range(5)]) for s in SITES]
    add('q3.eig_wins', sum(g > 0 for g in gap), 'S', pj)
    add('q3.qr_rand.lo', min(qr_rand), 'S', pj)
    add('q3.qr_rand.hi', max(qr_rand), 'S', pj)
    span = max(max(js[s]['checks']['q2_qr_vs_eig_rel'], js[s]['checks']['q2_rand_vs_eig_rel'],
                   *(v for c in ('joint|64', 'source|64') for v in js[s]['q1'][c]['checks']['complete_band_rel'].values()))
               for s in SITES)
    add('ctl.span', span, 'S', pj)
    n90 = [float(np.median(array(f'{base}/{s}.npz', f'loc|{op}|qr|n90'))) for s in SITES for op in ('site', 'joint|64', 'source|64')]
    add('loc.qr.lo', min(n90), 'P', p)
    add('loc.qr.hi', max(n90), 'P', p)
    note('Q1 differences and Q2 cosines are recomputed from per-image float32 scores; round trips, energy shares, band '
         'ranks and controls are read from the per-site stage JSON (not the collected dw_results.json the generator uses).')


def ood(add, note):
    decode_summary(add, [('svhn.b4', 'b4', 'svhn|64|krr|cos_raw'), ('h.b4', 'b4', 'H|64|krr|cos_raw'),
                         ('c100.b9', 'b9', 'c100|64|krr|cos_raw'), ('pgd8.b9', 'b9', 'pgd8|64|krr|cos_raw'),
                         ('svhn.b4_captured', 'b4', 'svhn|64|krr|cap_raw'), ('h.b4_captured', 'b4', 'H|64|krr|cap_raw'),
                         ('svhn.b4_agree', 'b4', 'svhn|64|krr|agree'), ('h.b4_agree', 'b4', 'H|64|krr|agree')])
    root = f'{PKG}/r20s0-p2-pr300'
    op, svhn = f'{root}/op.json', f'{root}/ov/svhn.json'
    probe, pf = packaged_array(op, 'source')
    novel, nf = packaged_array(svhn, 'source')
    i = SITES.index('b9')
    plane = np.asarray(read_json(op)['axes']['pooled'])[:, :2]
    for key, p, s in [('full', probe[i], novel[i]), ('plane', probe[i] @ plane, novel[i] @ plane)]:
        metric = (np.std(s, axis=0) / np.std(p, axis=0)) ** 2
        add(f'display.svhn.{key}', metric.mean(), 'R', op, svhn, pf, nf)
    before = staged(f'{JOINT}/coords_H.npz')
    for budget in (2, 8):
        after = staged(f'{JOINT}/coords_pgd{budget}.npz')
        for key, kind in [('y', 'joint'), ('onestep', 'source')]:
            attacked = first_modes(after, key)[i]
            delta = attacked - first_modes(before, key)[i, :len(attacked)]
            add(f'display.pgd{budget}.{kind}', energy_fraction(delta, plane), 'R', before, after, op)
    p = f'{ASSETS}/inv-ood.json'
    add('attack.arrows', read_json(p)['emb_attacks']['arrows'], 'S', p)


def edits(add, note):
    p = f'{INV}/rt/{P8}/b4__K64.json'
    summary = read_json(p)
    for k, method, stat in [('residual', 'resid', 'p_b'), ('match', 'resid+match', 'p_b'),
                            ('miss_before', 'resid', 'miss_S'), ('miss_after', 'resid+match', 'miss_S'),
                            ('joint_before', 'resid', 'miss_D'), ('joint_after', 'resid+match', 'miss_D'),
                            ('inferred_miss', 'resid+match_req', 'miss_S'), ('inferred_prob', 'resid+match_req', 'p_b')]:
        add(f'b4.{k}', summary[f'E4|{method}|{stat}'], 'S', p)
    root = f'{PKG}/r20s0-p2-pr300'
    f, op, fig = f'{root}/ov/edit-b4-b.json', f'{root}/op.json', f'{ASSETS}/inv-edits.json'
    donors = read_json(f)['images']['ids']
    add('display.b4.n', len(donors), 'R', f)
    add('display.b4.donors', len(set(donors)), 'R', f)
    add('display.arrows', read_json(fig)['emb_b4']['arrows'], 'S', fig)
    plane = np.asarray(read_json(op)['axes']['pooled'])[:, :2]
    p = f'data/stretch/rt_points/{P2}/b4__K16.npz'
    before = staged(f'{JOINT}/coords_H.npz')
    split = staged(f'{IV}/splits.npz')
    logits = f'{INV}/meta/c10_logits.npy'
    pred = np.load(ROOT / logits, allow_pickle=False).argmax(axis=1)[array(split, 'H')]
    methods, se, pb, pairpos, pair_ok, meta = arrays(p, 'methods', 's_edit', 'p_b', 'pair_pos', 'pair_ok', 'meta')
    meta = json.loads(str(meta))
    a, b = pairpos.T
    keep = pred[a] != pred[b]
    np.testing.assert_array_equal(keep, pair_ok)
    li = meta['sites'].index('b4')
    src = first_modes(before, 'onestep')[li]
    methods = methods.tolist()
    joint_a, joint_edited = arrays(p, 'y_a', 'y_patched')
    for kind, delta in [('source', se[methods.index('resid')] - se[methods.index('none')]),
                        ('joint', joint_edited[methods.index('resid'), li] - joint_a[li])]:
        add(f'display.b4.{kind}', energy_fraction(delta[keep], plane), 'R', p, op)
    swap = se[methods.index('swap')]
    np.testing.assert_allclose(swap, src[b], atol=1e-8, rtol=0)
    spread = src[:meta['m_full']].std(axis=0, ddof=1)  # torch.std default in stage_rt.py
    reference = 'data/stretch/ref_rt/b4__K16.npz'
    for k, method in [('residual', 'resid'), ('match', 'resid+match')]:
        i = methods.index(method)
        add(f'replay.b4.{k}', np.mean(pb[i, keep], dtype=float), 'P', p, split, logits)
        miss = np.mean(((se[i, keep] - swap[keep]) / spread) ** 2)
        archived = np.nanmean(array(reference, f'E4|{method}|miss_S')[:len(keep)])
        np.testing.assert_allclose(miss, archived, rtol=1e-6, atol=1e-13)
        add(f'replay.b4.{k}_miss', miss, 'R', p, before, split, logits, reference)
        alt = np.mean(((se[i, keep] - src[b[keep]]) / spread) ** 2)
        note(f'b4 {method} mean source miss: use rerun full swap {miss:.12g}, versus archived clean source '
             f'{alt:.12g}; both round the same. The full swap matches clean source to 1e-8 absolute per coordinate.')
    note('The 64 displayed rerun pairs are a prefix of a separate P2/K16 run, not independent evidence for the P8/K64 tiles.')


def x1(add, note):
    comp = staged(f'{XV}/compare/c10.npz')
    nets, labels = arrays(comp, 'nets', 'labels')
    probes = staged(f'{XV}/probes.npz')
    n = len(array(probes, 'c10_idx'))
    assert len(labels) == n
    add('sample.networks', len(nets), 'R', comp)
    add('sample.probe', n, 'R', probes)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    within = np.zeros(n, dtype=int)
    for c in sorted(set(labels)):
        pos = np.flatnonzero(labels == c)
        within[pos] = rng.permutation(pos)
    train, test = np.split(rng.permutation(n), 2)
    np.testing.assert_array_equal(perm, array(comp, 'perm'))
    np.testing.assert_array_equal(within, array(comp, 'within'))
    k = 16  # analysis K selected in xp/stage_compare.py, not the stored mode count
    groups = ['c10', 'c100', 'in32', 'svhn']
    group = {net: net.split('-')[0].removesuffix('x') for net in nets}
    paths = {net: staged(f'{XV}/embed/{net}/c10/op_knn15_trunk.npz') for net in nets}
    ordered = list(nets)
    add('sample.modes', min(first_modes(paths[ordered[0]], 'vecs', k).shape[-1], k), 'R', paths[ordered[0]])
    add('sample.half', test.size, 'R', comp)
    a_train, a_test = {}, {}
    for net in ordered:
        if group[net] not in groups:
            continue
        v = first_modes(paths[net], 'vecs', k)
        a_train[net] = v[:, train].reshape(-1, k)
        a_test[net] = v[:, test]
    del v
    reference = first_modes(paths['c10-s0'], 'vecs', k)
    per_site = np.linalg.norm(reference.reshape(reference.shape[0], -1), axis=1)
    _, _, axes_t = np.linalg.svd((reference / per_site[:, None, None]).reshape(-1, k), full_matrices=False)
    plane = axes_t[:2].T
    for net, site in [('c100-s0', 'b9'), ('svhn-s0', 'b9'), ('c10-s1', 'stem')]:
        q, _ = orthogonal_procrustes(a_train[net], a_train['c10-s0'])
        l = SITES.index(site)
        a, b = a_test['c10-s0'][l], a_test[net][l] @ q
        key = f'projection.{net.replace("-", "_")}.{site}'
        for label, a_, b_ in [('full', a, b), ('display', a @ plane, b @ plane)]:
            score = np.vdot(a_, b_) / (np.linalg.norm(a_) * np.linalg.norm(b_))
            add(f'{key}.{label}', score, 'R', comp, paths['c10-s0'], paths[net])
    archive = array(comp, 'knn15_trunk_cos')
    stem_measures = {key: array(comp, key)[:, :, 0, 0] for key in ('nn', 'cka')}
    for g in groups:
        vals, indices = [], []
        files = [comp]
        for (i, a), (j, b) in combinations(enumerate(ordered), 2):
            if sorted((group[a], group[b])) != sorted(('c10', g)):
                continue
            q, _ = orthogonal_procrustes(a_train[b], a_train[a])
            aa = a_test[a].reshape(len(SITES)-2, -1)
            bb = (a_test[b] @ q).reshape(aa.shape)
            cos = np.einsum('lm,lm->l', aa, bb) / (np.linalg.norm(aa, axis=1) * np.linalg.norm(bb, axis=1))
            np.testing.assert_allclose(cos, archive[i, j], atol=1e-6, rtol=0)
            vals.append(cos)
            indices.append((i, j))
            files.extend([paths[a], paths[b]])
        # Trunk joint stem agreement mixes all sites; the stem-only controls
        # come separately from comparison-level summaries (not probe rows).
        for ix, site in [(0, 'stem'), (-1, 'b9')]:
            add(f'pair.c10_{g}.{site}', np.mean(vals, axis=0)[ix], 'R', *files)
        if g != 'c10':
            for key, metric in stem_measures.items():
                add(f'stem.{g}.{key}', np.mean([metric[i, j] for i, j in indices]), 'S', comp)


def x3(add, note):
    p = f'{X3}/sets.npz'
    ids, labels, nat_label = arrays(p, 'S', 'labels', 'natA_y')
    op = f'{X3}/op_c10-s0_ea100.npz'
    # xp/stage_x3.py saves depth in xp/common.py:SITES order, not in the NPZ.
    chain, probe = SITES, array(op, 'probe_labels')
    add('cohort.n', len(ids), 'R', p)
    add('probe.n', len(probe), 'R', op)
    sample = f'{PKG}/r20s0-c10-ea100/ov/u8.json'
    sample_pos = np.asarray(read_json(sample)['sample']['pos'])
    add('sample.n', len(sample_pos), 'R', sample)
    fig = f'{ASSETS}/x3-adversarial.json'
    density = read_json(fig)['emb_lead']['arrows']
    # Same deterministic picking rule as the shipped renderer, not stage or
    # generator sampling. A smaller density takes a permutation prefix.
    renderer = 'site/js/embfig.js'
    stride = int(re.search(r'\(j \* (\d+)\) % n < k', text(renderer))[1])
    shown = sample_pos[(np.arange(len(sample_pos)) * stride % len(sample_pos)) < density]
    add('projection.arrows', len(shown), 'R', sample, fig, renderer)
    clean_path = f'{X3}/c10-s0_clean.npz'
    clean = array(clean_path, 'ea100_site_y')
    add('metric.k', clean.shape[-1], 'R', clean_path)
    rng = np.random.default_rng(0)
    same = rng.integers(len(ids), size=(40000, 2))
    i, j = same.T
    good = (i != j) & (labels[i] == labels[j])
    unit = np.median(np.linalg.norm(clean[:, i[good]] - clean[:, j[good]], axis=-1), axis=1)
    distances = {}
    for name in ('u8', 'n8'):
        f = f'{X3}/c10-s0_{name}.npz'
        moved = array(f, 'ea100_site_y') - clean
        full = np.median(np.linalg.norm(moved, axis=-1), axis=1)
        arrows = np.median(np.linalg.norm(moved[:, shown, :2], axis=-1), axis=1)
        distances[name] = (full, arrows, f)
        for site in ('b6', 'b7', 'b9'):
            l = chain.index(site)
            add(f'disp.{name}.{site}', full[l] / unit[l], 'R', p, clean_path, f)
    adv, noise = distances['u8'], distances['n8']
    for l, site in enumerate(chain):
        if site in ('b6', 'b7', 'b9'):
            for axis, jj in [('full', 0), ('arrows', 1)]:
                files = [clean_path, adv[2], noise[2]] + ([sample, fig, renderer] if jj else [])
                add(f'projection.{site}.{axis}', float(adv[jj][l]) / noise[jj][l], 'R', *files,
                    raw_atol=1e-6, raw_reason='fp32 norms and reduction')
    ua, na, ub = (f'{X3}/c10-{net}_{s}.npz' for net, s in [('s0', 'u8'), ('s0', 'natA'), ('s1', 'u8')])
    la, ma = arrays(ua, 'logits', 'ea100_mass')
    dst = la.argmax(axis=1)
    success = dst != labels
    m = ma[:, success].transpose(1, 0, 2)
    a, b, idx = labels[success], dst[success], np.arange(success.sum())
    # The last violation defines where a persistent switch can begin.
    sw = np.where(m[idx, :, b] <= m[idx, :, a], np.arange(len(chain)), -1).max(axis=1) + 1
    lower_median = sorted(sw)[(len(sw) - 1) // 2]
    add('switch.u8.median', (*chain, 'Never')[lower_median], 'P', p, ua, op)
    add('switch.u8.by_b6', np.mean(sw <= chain.index('b6')), 'P', p, ua, op)
    top = np.where(m[idx, :, b] < m.max(axis=2) - 1e-12, np.arange(len(chain)), -1).max(axis=1) + 1
    ordered = (*chain, 'Never')
    b6 = chain.index('b6')
    note(f'Persistent destination>source: lower median {ordered[lower_median]}, by b6 {np.mean(sw <= b6):.6g}; '
         f'persistent destination is top class: lower median {ordered[np.sort(top)[(len(top)-1)//2]]}, '
         f'by b6 {np.mean(top <= b6):.6g}. The tile specifies the true-class rule.')
    # The producer reduces float32 class masses along depth-major, strided
    # rows. Four fp32 eps at unit scale allow for its accumulation order;
    # we independently take the mean in float64.
    mass_atol = 4 * np.finfo(np.float32).eps
    add('mass.u8.logits.destination', np.mean(m[idx, -1, b], dtype=float), 'P', p, ua, raw_atol=mass_atol)
    ln, mn = arrays(na, 'logits', 'ea100_mass')
    dn = ln.argmax(axis=1)
    err = np.flatnonzero(dn != nat_label)
    add('mass.natural.logits.destination', np.mean(mn[-1, err, dn[err]], dtype=float), 'P', p, na, raw_atol=mass_atol)
    pred1 = array(ub, 'logits').argmax(axis=1)
    add('transfer.u8.flip', np.mean(pred1[success] != labels[success]), 'R', p, ua, ub)
    note(f'Seed 1 any flip = {np.mean(pred1[success] != labels[success]):.6g}; same destination = '
         f'{np.mean(pred1[success] == dst[success]):.6g}. The tile says any flip.')
    images = arrays(p, 'img_u8', 'img_clean')
    add('@literal.epsilon_u8', np.max(np.abs(images[0].astype(np.int16) - images[1])).item(), 'R', p,
        literal=('8/255', 8))
    note('The 8/255 budget is consistent with the saved uint8 attack/clean difference; the exact float PGD inputs were not saved.')
    stretch = 'data/stretch/x3'
    attack = f'{stretch}/c10-s0_u8_pr100.npz'
    ss = f'{stretch}/sets.npz'
    truth = array(ss, 'labels')
    pred = array(attack, 'logits').argmax(axis=-1)
    good = truth != pred
    row = np.arange(np.count_nonzero(good))
    switched = {}
    for key in ('ea100', 'pr100', 'knn15'):
        m = array(attack, f'{key}_mass')[:, good].transpose(1, 0, 2)
        # The final failure index of destination > true determines the first
        # persistent switch. Never is one bin past logits.
        beat = m[row, :, pred[good]] > m[row, :, truth[good]]
        switched[key] = len(SITES) - np.argmin(np.column_stack([beat[:, ::-1], np.zeros(len(row), bool)]), axis=1)
    medians = {np.sort(v)[(len(v)-1)//2] for v in switched.values()}
    if len(medians) != 1:
        raise ValueError(f'matched kernel switch medians differ: {medians}')
    add('rerun.switch.common_median', (*SITES, 'Never')[medians.pop()], 'P', ss, attack)
    for key in ('pr100', 'knn15'):
        gap = np.abs(switched[key] - switched['ea100'])
        add(f'rerun.switch.{key}.different', np.mean(gap > 0), 'P', ss, attack)
        add(f'rerun.switch.{key}.two_plus', np.mean(gap >= 2), 'P', ss, attack)
    for key, name in [('ea100', 'ea'), ('knn15', 'knn')]:
        p = f'{X3}/op_c10-s0_{key}.npz'
        c, t, mu = arrays(p, 'c', 't', 'vals')
        add(f'depth.{name}_cond', depth_metrics(c, t, mu[:16])[0].max(), 'R', p)
    p = f'{X3}/DONE'
    done = read_json(p)
    add('depth.replay', max(done[f'c10-s0_{key}_replay'] for key in ('ea100', 'knn15')), 'S', p)
    for kernel_name in ('ea100', 'pr100'):
        run_op = f'{stretch}/op_c10-s0_{kernel_name}.npz'
        add(f'rerun.concentration.{kernel_name}.joint', top_two(run_op), 'R', run_op)
        v = first_modes(run_op, 'site_vec')
        e = np.square(v)
        add(f'rerun.concentration.{kernel_name}.site', (np.sort(e, axis=1)[:, -2:, :].sum(axis=1) / e.sum(axis=1)).max(), 'R', run_op)
        paths = [f'{stretch}/c10-s0_{s}_pr100.npz' for s in ('clean', 'u8', 'n8')]
        clean, adv, noise = (array(p, f'{kernel_name}_site_y').astype(float) for p in paths)
        # The common clean-distance unit cancels within each site's A/N ratio.
        a_med = np.median(np.linalg.norm(adv - clean, axis=-1), axis=1)
        n_med = np.median(np.linalg.norm(noise - clean, axis=-1), axis=1)
        for site in ('b6', 'b7', 'b8', 'b9'):
            l = SITES.index(site)
            add(f'rerun.ratio.{kernel_name}.{site}', a_med[l] / n_med[l], 'R', *paths,
                raw_atol=5e-7, raw_reason='fp32 coordinate norms')


X4_CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# Numbers drawn inside the X4 static figures; the generator takes their text from the fact catalogue, so they are
# audited here like prose tokens.
X4_FIGURE_FACTS = ('pairs.per_kind', 'heim.clip.b3', 'heim.clip.b9', 'embstep.raw_b3', 'embstep.emb_b3', 'embstep.emb_b9',
                   *(f'window.b3.{s}.{q}' for s in ('b4', 'b6', 'b8', 'b9', 'logits') for q in ('median', 'max')),
                   *(f'{k}.{s}.share' for k in ('leaves', 'third') for s in ('pixels', 'stem', 'b3', 'b6', 'b8', 'b9')))
# Values on the X4 page that are definitions or transcriptions rather than measurements, with the reason.
X4_LISTED = {
    'grid.jump': 'definition: a single jump puts the whole distance into one window (100%)',
    'emb.perplexity': 'kernel parameter KERNEL in xp/stage_x4.py (ea<perplexity>)',
    'pairs.margin_min': 'eligibility constant MARGIN_MIN in xp/stage_x4.py',
    'threshold.share': 'summary threshold in xp/analyze_x4.py',
    'gain.affine.pool': 'transcribed from checks.json (the run\'s own closed-form comparison)',
    'gain.affine.b9': 'transcribed from checks.json (the run\'s own closed-form comparison)',
    'speedfd.pixels': 'stored diagnostic jvp_fd_med (computed during the run), read directly',
    'speedfd1024.pixels': 'stored diagnostic jvp_fd1024_med (computed during the run), read directly',
}


def x4_used_facts():
    """Every x4-paths fact referenced by the fragment, the card, the X4 lede bullet or a figure annotation."""
    pre = 'x.x4-paths.'
    card = read_json(f'{ASSETS}/x4-paths.json')['card']
    texts = [text(f'{FRAG}/x4-paths.html'), card['answer'], *card['numbers'], text(f'{FRAG}/_lede.html')]
    keys = {k.removeprefix(pre) for t in texts for k in TOKEN.findall(t) if k.startswith(pre)}
    return keys | set(X4_FIGURE_FACTS)


def x4(add, note):
    """Independent recomputation of every number on the X4 page (own code path, stored arrays only)."""
    from numpy.lib.stride_tricks import sliding_window_view as slide
    used = x4_used_facts()
    done = set()

    def put(key, value, tier, *paths, **kw):
        if key in used:
            add(key, value, tier, *paths, **kw)
            done.add(key)

    def fastest(steps, w):                          # largest share of the total in w consecutive steps (last axis)
        steps = steps.astype(np.float64)
        return slide(steps, w, axis=-1).sum(-1).max(-1) / steps.sum(-1)

    def softmax(z):
        z = z.astype(np.float64)
        e = np.exp(z - z.max(-1, keepdims=True))
        return e / e.sum(-1, keepdims=True)

    p = staged(f'{XV}/x4/pairs.npz')
    kind, pair_ids = arrays(p, 'kind', 'pair_idx')
    kinds = np.bincount(kind)
    if len(kinds) != 2 or kinds[0] != kinds[1]:
        raise ValueError(f'unequal same-/different-class strata: {kinds}')
    same, diff = kind == 0, kind == 1
    put('pairs.per_kind', kinds[0], 'R', p)
    put('pairs.unique_endpoints', np.unique(pair_ids).size, 'R', p)
    stage, analysis, gen = 'xp/stage_x4.py', 'xp/analyze_x4.py', 'site/xsite/gen_x4.py'
    put('pairs.margin_min', source_constant(stage, 'MARGIN_MIN'), 'S', stage)
    put('threshold.share', float(re.search(r"'frac_share_gt_quarter'.*?>\s*([0-9.]+)", text(analysis))[1]), 'S', analysis)
    pk = f'{PKG}/x4-pixels-suffix/paths/x4-pixels-paths.json'
    put('pairs.display', len(read_json(pk)['meta']['pair_pos']), 'R', pk)
    kernel = source_constant(stage, 'KERNEL')                       # e.g. 'ea100': Shannon perplexity 100
    put('emb.perplexity', int(re.fullmatch(r'ea(\d+)', kernel)[1]), 'S', stage)
    put('grid.jump', 1.0, 'R', stage)
    ps = {Path(f).stem.removeprefix('paths_'): str(f.relative_to(ROOT)) for f in sorted((ROOT / X4).glob('paths_*.npz'))}
    starts = [s for s in ('pixels', 'stem', 'b3', 'b6', 'b8', 'b9') if s in ps]
    if sorted(starts) != sorted(ps):
        raise ValueError(f'unexpected X4 starts {sorted(ps)}')
    put('starts.tested', len(starts), 'R', *ps.values())
    put('paths.total', len(starts) * len(kind), 'R', *ps.values())
    top, top_pre, relu, refine_raw, refine_unit, sigma_min, n_ref = 0., 0., [], 0., 0., np.inf, 0
    hits = logit_hits = 0
    thr = float(re.search(r"'frac_share_gt_quarter'.*?>\s*([0-9.]+)", text(analysis))[1])
    for s in starts:
        f = ps[s]
        d, grid, grid64, sites, kind_s, taps = arrays(f, 'd_raw', 's256', 's64', 'sites', 'kind', 'taps')
        np.testing.assert_array_equal(kind_s, kind)
        w = (grid.size - 1) // 16
        share = fastest(d, w)
        top = max(top, share.max())
        put(f'window.{s}.max', share.max(), 'P', f)
        if s == 'pixels':
            put('grid.uniform', w / (grid.size - 1), 'R', f)
            put('grid.points', grid.size, 'R', f)
            put('grid.draw', grid64.size, 'R', f)
            put('probe.n', array(f, 'probe_idx').size, 'R', f)
            put('emb.k', array(f, 'probe_site_y').shape[-1], 'R', f)
            fd, fd1024 = arrays(f, 'jvp_fd_med', 'jvp_fd1024_med')
            li = list(sites).index('logits')
            put('speedfd.pixels', fd[li], 'S', f)
            put('speedfd1024.pixels', fd1024[li], 'S', f)
        if s == 'b3':
            for site in ('b4', 'b6', 'b8', 'b9', 'logits'):
                col = share[:, list(sites).index(site)]
                put(f'window.b3.{site}.median', np.median(col), 'P', f)
                put(f'window.b3.{site}.max', col.max(), 'P', f)
        if len(taps):
            relu.append(fastest(array(f, 'relu_prev'), w).ravel())
        ref, s1024, sraw, s1024u, sunit, sig = arrays(f, 'ref_rows', 'share1024_raw', 'share_raw', 'share1024_unit', 'share_unit', 'sigma_raw')
        n_ref += ref.size
        refine_raw = max(refine_raw, np.nanmax(np.abs(s1024 - sraw[ref])))
        refine_unit = max(refine_unit, np.nanmax(np.nanmedian(np.abs(s1024u - sunit[ref]), axis=0)))
        sigma_min = min(sigma_min, np.nanmin(sig))
        later = np.array([x not in ('logits', s) for x in sites])
        hits += int((sunit[:, later] > thr).any(1).sum())
        logit_hits += int((sunit[:, np.asarray(sites) == 'logits'] > thr).sum())
        pred, lab = arrays(f, 'pred', 'labels')
        out_same = pred[same] != lab[same, :1]
        leave = out_same.any(1)
        put(f'leaves.{s}.share', leave.mean(), 'P', f)
        put(f'leaves.{s}.n', leave.sum(), 'P', f)
        other = (pred[diff] != lab[diff, :1]) & (pred[diff] != lab[diff, 1:])
        put(f'third.{s}.share', other.any(1).mean(), 'P', f)
        if leave.any():
            put(f'leave_stretch.{s}.median', np.median(out_same[leave].sum(1) / pred.shape[1]), 'P', f)
    allp = list(ps.values())
    put('window.max', top, 'P', *allp)
    relu = np.concatenate(relu)
    put('relu.window.median', np.median(relu), 'P', *allp)
    put('relu.window.max', relu.max(), 'P', *allp)
    put('relu.window.n', relu.size, 'P', *allp)
    put('refine.n', n_ref, 'P', *allp)
    put('refine.raw_max', 100 * refine_raw, 'P', *allp)
    put('refine.max_median', refine_unit, 'P', *allp)
    put('sharp.min', sigma_min, 'P', *allp)
    put('pre_logits.n', hits, 'P', *allp)
    put('logits.n', logit_hits, 'P', *allp)

    # outputs: KL from the start of the line, the steepest b3 line, the affine b9 drop, clipped curves
    kl_top = source_constant(gen, 'KL_TOP')
    f3, f9 = ps['b3'], ps['b9']
    probs = {}
    for s, f in (('b3', f3), ('b9', f9)):
        lg, lab = arrays(f, 'logits', 'labels')
        pr = softmax(lg)
        logp = np.log(np.clip(pr, 1e-300, None))
        kl = np.einsum('pc,ptc->pt', pr[:, 0], logp[:, 0][:, None, :] - logp)
        probs[s] = (pr, kl, lab)
        put(f'heim.clip.{s}', (kl.max(1) > kl_top).sum(), 'P', f, gen)
    pr, kl, lab = probs['b3']
    w = 16
    rise = slide(kl, w + 1, axis=1)[..., -1] - slide(kl, w + 1, axis=1)[..., 0]
    i, j = np.unravel_index(np.argmax(rise), rise.shape)
    grid = array(f3, 's256')
    put('steep.b3.a', X4_CLASSES[lab[i, 0]], 'P', f3)
    put('steep.b3.b', X4_CLASSES[lab[i, 1]], 'P', f3)
    put('steep.b3.kl_from', kl[i, j], 'P', f3)
    put('steep.b3.kl_to', kl[i, j + w], 'P', f3)
    put('steep.b3.kl_s_from', grid[j], 'P', f3)
    put('steep.b3.kl_s_to', grid[j + w], 'P', f3)
    d3, sites3 = arrays(f3, 'd_raw', 'sites')
    dl = d3[i, list(sites3).index('logits')].astype(np.float64)
    put('steep.b3.kl_window_share', dl[j:j + w].sum() / dl.sum(), 'P', f3)
    pa = pr[np.arange(len(lab)), :, lab[:, 0]]
    fall = pa[:, :-w] - pa[:, w:]
    i2, j2 = np.unravel_index(np.argmax(fall), fall.shape)
    put('steep.b3.p_from', pa[i2, j2], 'P', f3)
    put('steep.b3.p_to', pa[i2, j2 + w], 'P', f3)
    pr9, _, lab9 = probs['b9']
    pa9 = pr9[np.arange(len(lab9)), :, lab9[:, 0]]
    put('steep.b9.drop_pp', 100 * (pa9[:, :-w] - pa9[:, w:]).max(), 'P', f9)

    # Jacobian quantities
    for s in ('pixels', 'b3', 'b6'):
        f = ps[s]
        g, al, sites = arrays(f, 'gain_raw', 'align_raw', 'sites')
        mid = np.median(g[diff, 0, 32])
        put(f'gain.{s}.s0', np.median(g[diff, 0, 0]), 'P', f)
        put(f'gain.{s}.mid', mid, 'P', f)
        put(f'ftle.{s}.mid', mid / (12 - (0 if s == 'pixels' else SITES.index(s) + 1)), 'P', f)
        put(f'align.{s}.mid', np.median(al[diff, 32]), 'P', f)
    sp, ssw, nsw, g64, sites3 = arrays(f3, 'speed', 's_switch', 'n_switch', 's64', 'sites')
    peak = g64[sp[:, list(sites3).index('logits')].argmax(1)]
    one = diff & (nsw == 1)
    put('timing.all', spearmanr(peak[diff], ssw[diff]).statistic, 'P', f3)
    put('timing.single', spearmanr(peak[one], ssw[one]).statistic, 'P', f3)
    put('timing.single_n', one.sum(), 'P', f3)

    # embedding: bunching at b9, support, localisation of the two drawn modes
    def emb_share(f):
        y, chain = arrays(f, 'emb_site_y', 'chain')
        y = y[list(chain).index('b9')].astype(np.float64)
        return fastest(np.sqrt((np.diff(y, axis=1) ** 2).sum(-1)), 4)
    e3, e9 = emb_share(f3), emb_share(f9)
    d64 = array(f3, 'd64_raw')[:, list(sites3).index('b9')]
    put('embstep.raw_b3', np.median(fastest(d64, 4)[diff]), 'P', f3)
    put('embstep.emb_b3', np.median(e3[diff]), 'P', f3)
    put('embstep.emb_b9', np.median(e9[diff]), 'P', f9)
    put('embstep.half_b3', (e3[diff] > .5).sum(), 'P', f3)
    put('embstep.half_b9', (e9[diff] > .5).sum(), 'P', f9)
    cmax, refc, chain3, psy = arrays(f3, 'emb_cmax', 'ref_cmax', 'chain', 'probe_site_y')
    for site in ('b9', 'logits'):
        l = list(chain3).index(site)
        pct = (refc[l][None, None, :] < cmax[l][..., None]).mean(-1)          # share of clean images below each point
        put(f'support.b3.{site}', np.median(pct[diff].min(1)), 'P', f3)
    q = psy[..., :2].astype(np.float64) ** 2
    put('modes.top2_max', (np.sort(q, axis=1)[:, -2:].sum(1) / q.sum(1)).max(), 'R', f3)

    # worked examples
    for r in (183, 150, 33):
        a_, b_ = array(f3, 'labels')[r]
        put(f'we{r}.a', X4_CLASSES[a_], 'R', f3)
        put(f'we{r}.b', X4_CLASSES[b_], 'R', f3)
        put(f'we{r}.ids', f'{pair_ids[r, 0]} and {pair_ids[r, 1]}', 'R', p)
        for s in ('pixels', 'b3', 'b6', 'b9'):
            f = ps[s]
            pred, lg, spd, g, sites, g64 = arrays(f, 'pred', 'logits', 'speed', 'gain_raw', 'sites', 's64')
            seq = pred[r]
            k = np.flatnonzero(seq[1:] != seq[:-1])
            sw = (k + .5) / (seq.size - 1)
            names = [X4_CLASSES[seq[0]]] + [X4_CLASSES[seq[t + 1]] for t in k]
            put(f'we{r}.{s}.route', ' → '.join(names), 'P', f)
            if k.size:
                put(f'we{r}.{s}.sw1', sw[0], 'P', f)
                put(f'we{r}.{s}.swlast', sw[-1], 'P', f)
            off = (seq != a_) if a_ == b_ else (seq != a_) & (seq != b_)
            put(f'we{r}.{s}.off', off.mean(), 'P', f)
            v = spd[r, list(sites).index('logits')].astype(np.float64)
            put(f'we{r}.{s}.peak_s', g64[v.argmax()], 'P', f)
            put(f'we{r}.{s}.peak_chord', v.max() / np.linalg.norm(lg[r, -1].astype(np.float64) - lg[r, 0]), 'P', f)
            put(f'we{r}.{s}.gain_ratio', np.exp(g[r, 0].max() - g[r, 0].min()), 'P', f)

    # transcribed checks and the probe-gain correlations
    cj = staged(f'{XV}/x4/checks.json')
    checks = read_json(cj)
    for s in ('pool', 'b9'):
        put(f'gain.affine.{s}', checks[f'{s}_logits_exact_err'], 'S', cj)
    f = staged(f'{XV}/x4/probe_gain.npz')
    ss, mm, graw, gang, rad, logits = arrays(f, 'starts', 'targets', 'gain_raw', 'gain_ang', 'rad_out', 'logits')
    i, t = ss.tolist().index('pixels'), mm.tolist().index('logits')
    top2 = np.sort(logits, axis=1)[:, -2:]
    margin = top2[:, 1] - top2[:, 0]
    for key, score in (('raw', graw[i, t]), ('angular', gang[i, t]), ('radius_removed', gang[i, t] + np.log(rad[i, t]))):
        put(f'gain.pixels.{key}', spearmanr(score, margin).statistic, 'P', f)
    put('gain.radius', spearmanr(np.log(rad[i, t]), margin).statistic, 'P', f)

    missing = sorted(used - done)
    note(f'{len(done)} of {len(used)} X4 page facts recomputed or read from their stated source; listed constants and '
         'transcriptions: ' + '; '.join(f'{k} ({v})' for k, v in X4_LISTED.items() if k in used))
    if missing:
        raise ValueError('X4 page facts neither recomputed nor listed: ' + ', '.join(missing))


def solver(add, note):
    p = 'opt/REPORT.md'
    report = text(p)
    wanted = {'Walk storage, 12 sites': ('storage.dense', 'storage.sparse'),
              'Sinkhorn, all sites': ('balance.dense', 'balance.sparse'),
              '16-mode solve, t = t*': (None, 'k16.sparse_modes')}
    for row in report.splitlines():
        if not row.startswith('|'):
            continue
        fields = [s.strip() for s in row.strip('|').split('|')]
        for k, cell in zip(wanted.get(fields[0], ()), fields[1:]):
            if k is not None:
                transcription = float(re.match(r'\d+(?:\.\d+)?', cell)[0])
                if k.startswith('storage.'):
                    files, _, _, _, dense_mb, compact_mb = probe_gram_metrics()
                    val, atol = (dense_mb, 0) if k == 'storage.dense' else (compact_mb, .05)
                    np.testing.assert_allclose(val, transcription, atol=atol, rtol=0)
                    add(k, val, 'R', p, *files, raw_atol=atol, raw_reason='report already rounded')
                else:
                    add(k, transcription, 'S', p)
    for key, pattern, expected in [('probe_n', r'Measured on the real N = ([\d,]+)', 2000),
                                   ('walks', r'Walk storage, (\d+) sites', 12),
                                   ('modes', r'(\d+)-mode solve, t = t\*', 16)]:
        add('@literal.' + key, int(re.search(pattern, report)[1].replace(',', '')), 'S', p,
            literal=(f'{expected:,}', expected))
    note('No benchmark logs archived locally; the three times trace only to opt/REPORT.md. Storage is independently '
         'derived from the Gram/top-15 union and the CSR plus rank-one layout in opt/sparse_core.py.')


CHECKS = (kernel, sources, decode, wavelets, ood, edits, x1, x3, x4, solver)
# Evidence actually consumed by the producer, independently of our own tier.
# A literal matching a shipped sample is a passing value but not a recomputation
# by the generator. Keep that warning distinct from a numerical failure.
PRODUCER_TIER_WARNINGS = {
    ('inv-sources', 'attack.shown'): ('R', 'reads packaged n metadata, rather than counting shipped image IDs'),
    ('inv-decode', 'near.sample'): ('R', 'reads packaged n metadata, rather than inspecting array length'),
    **{('inv-edits', f'display.{site}.n'): ('R', 'reads packaged n metadata, rather than counting shipped donor rows')
       for site in ('b4', 'b6')},
}

# Explicit waivers of raw re-derivation. The stored values are still checked
# by the corresponding S rows; these are not waivers of coverage/transcription.
RAW_WAIVERS = {
    'inv-kernel': {
        'top2.p8.ea400*': 'P8 Shannon modes were not staged.',
        'refit.*': 'Refitted operator and paired coordinates are not local; only the archived held-out shares are saved here.',
    },
    'inv-sources': {
        'share.p8.*': 'P8 depth-system inputs (c, t and eigenvalues) were not staged.',
        'patch.b5.swap': 'P8 individual patched trajectories and progress scores were not staged.',
    },
    'inv-decode': {'large.*': 'Only the P8 decoder summaries are local, not the individual per-image P8 scores.'},
    'inv-wavelets': {'q2.a2.k_*, q2.a32.rt_*, q3.*, ctl.span': 'Eigendecompositions, coordinate variances and band energies were computed on Modal; only per-site summaries and per-image scores are local.'},
    'inv-ood': {'svhn.b4*, h.b4*, c100.b9, pgd8.b9': 'No staged per-image scores for these P8 decoder conditions.'},
    'inv-edits': {'b4.*': 'No staged per-pair P8/K64 source miss, joint miss or donor probabilities; P2/K16 is a separate run.'},
    'x1-data': {'stem.*': 'The comparison matrix already pools images; per-image neighbour lists and stem features for these pipelines were not staged.'},
    'x3-adversarial': {'depth.replay': 'Original replay discrepancy is recorded in DONE; matching unrounded replay inputs were not archived.'},
    'solver': {'balance.*, k16.sparse_modes': 'Historical timings have no local raw timing log; REPORT.md is the only measurement record.'},
}


def block(markup, cls):
    # Extract only the known, non-nested container type. Tile groups contain
    # child divs, so retain the whole interval up to the following h3.
    if cls == 'x-numbers':
        return re.findall(r'<div class="x-numbers"[^>]*>.*?(?=<h3\b|\Z)', markup, flags=re.S)
    # An answer may be a paragraph or a short bulleted list (x4-paths); either is compared whole.
    return [m[0] for m in re.finditer(rf'<(p|ul) class="{cls}"[^>]*>.*?</\1>', markup, flags=re.S)]


def normalize(markup):
    return ' '.join(html.unescape(re.sub(r'<[^>]+>', ' ', markup)).split())


def card_list(markup):
    return re.findall(r'<li[^>]*>(.*?)</li>', markup, flags=re.S)


def resolved(markup, lookup):
    def sub(m):
        return html.escape(record['text']) if (record := lookup(m[1])) is not None else m[0]
    return TOKEN.sub(sub, markup)


def coverage(entry, facts, built, build, lede):
    frag = text(f'{FRAG}/{entry}.html')
    card = read_json(f'{ASSETS}/{entry}.json')['card']
    early = frag.partition('<h3')[0]
    card_pieces = [card['answer'], *card['numbers']]
    prefix = f'x.{entry}.'
    required = {k.removeprefix(prefix) for k in TOKEN.findall(' '.join([early, *card_pieces, lede]))
                if k.startswith(prefix)}
    problems = [f'headline refers to missing fact key {key}' for key in sorted(required - facts.keys())]

    def resolve(markup):
        return resolved(markup, lambda key: facts.get(key.removeprefix(prefix)))

    section = re.search(rf'<section class="experiment" id="x-{re.escape(entry)}">(.*?)</section>', built, re.S)
    cards = re.findall(r'<article class="x-card">(.*?)</article>', built, flags=re.S)
    card_body = next((v for v in cards if f'href="#experiments/x-{entry}"' in v), '')
    if section is None or not card_body:
        problems.append(f'{build}: missing section or at-a-glance card')
    else:
        # Match the entire answer and tile group (including literals), not just
        # a substring such as `0` or `b6` which might occur elsewhere.
        for name in ('x-answer', 'x-numbers'):
            a, b = block(frag, name), block(section[1], name)
            if name == 'x-numbers' and not a and not b:
                continue                     # tiles are optional; an answer paragraph is not
            if not a or [normalize(resolve(x)) for x in a] != [normalize(x) for x in b]:
                problems.append(f'{build}: {name} differs from resolved source')
        a = block(card_body, 'x-card-answer')
        b = card_list(card_body)
        if [normalize(resolve(x)) for x in card_pieces] != [normalize(x) for x in a + b]:
            problems.append(f'{build}: card differs from resolved source')
    return required, problems


NUM = re.compile(r'([-+]?\d[\d,]*(?:\.(\d+))?)(?:[eE]([+-]?\d+))?(%)?\Z')


def judge(actual, record, raw_atol=None, raw_reason='FP arithmetic'):
    value, shown = record['value'], record['text']
    if isinstance(value, str):
        return actual == value == shown, 'exact label'
    m = NUM.fullmatch(shown)
    if m is None:
        return False, 'unrecognized numeric display'
    factor = 100 if m[4] else 1
    places = len(m[2] or '')
    exp = int(m[3] or 0)
    centre = float(m[1].replace(',', '')) * 10.**exp / factor
    # Counts are exact, even though elsewhere integer-like values can be
    # rounded (e.g. 49 s and integer percentages).
    exact = (isinstance(actual, int) or isinstance(value, int)) and not m[4] and not m[3] and not places
    tol = 0 if exact else .5 * 10.**(exp - places) / factor
    epsilon = 1e-12
    dist_ok = abs(actual - centre) <= tol + epsilon and abs(value - centre) <= tol + epsilon
    # Declared full-precision values must also agree with the independent
    # calculation, allowing differences in reduction order and fp32 arrays.
    raw_tol = (0 if exact else max(1e-11, min(tol, 1e-7 * abs(actual)))) if raw_atol is None else raw_atol
    raw_ok = abs(actual - value) <= raw_tol
    why = 'exact integer' if exact else f'±{tol:g} display half-unit; ±{raw_tol:.2g} stored ({raw_reason})'
    return math.isfinite(actual) and dist_ok and raw_ok, why


def main(build=BUILD):
    rows, findings, notes = [], [], []
    required_count = 0
    try:
        built = text(build)
    except OSError as exc:
        print(f'FAIL build: {exc}')
        return 1
    lede = text(f'{FRAG}/_lede.html')
    catalog = {entry: read_json(f'{ASSETS}/{entry}.facts.json') for entry in ENTRIES}
    qualified = {f'x.{entry}.{key}': record for entry, facts in catalog.items() for key, record in facts.items()}
    for key in set(TOKEN.findall(lede)) - qualified.keys():
        findings.append(('lede', f'missing fact key {key}'))
    summary = re.compile(r'<div class="xp-sum">(.*?)</div>', re.S)
    source_text, built_text = summary.search(resolved(lede, qualified.get)), summary.search(built)
    if source_text is None or built_text is None or normalize(source_text[1]) != normalize(built_text[1]):
        findings.append(('lede', f'{build}: summary differs from resolved _lede.html source'))
    for entry, check in zip(ENTRIES, CHECKS, strict=True):
        facts = catalog[entry]
        for key, record in facts.items():
            if record.get('tier') not in {'R', 'P', 'S'}:
                findings.append((entry, f'{key}: invalid or absent tier'))
        required, problems = coverage(entry, facts, built, build, lede)
        required_count += len(required)
        findings.extend((entry, p) for p in problems)
        checked = set()

        def add(key, actual, tier, *paths, literal=None, raw_atol=None, raw_reason='FP arithmetic'):
            checked.add(key)
            if isinstance(actual, np.generic):
                actual = actual.item()
            if literal is None and key not in facts:
                rows.append((entry, key, 'MISSING', actual, 'no fact', tier, 'FAIL', paths))
                return
            record = facts[key] if literal is None else {'text': literal[0], 'value': literal[1], 'tier': '-'}
            ok, why = judge(actual, record, raw_atol, raw_reason) if literal is None else (actual == literal[1], 'exact literal')
            rows.append((entry, key, record['text'], actual, why, f'{record["tier"]}/{tier}', 'PASS' if ok else 'FAIL', paths))

        try:
            check(add, lambda msg: notes.append((entry, msg)))
        except (OSError, KeyError, ValueError, AssertionError) as exc:
            findings.append((entry, f'{type(exc).__name__}: {exc}'))
        findings.extend((entry, f'unchecked headline/lede token {key}') for key in sorted(required - checked))
    print('entry\tfact key\tdisplay\tindependent value\ttolerance\tproducer/audit tier\tresult\tinput files (repo-relative)')
    for entry, key, shown, actual, why, tier, status, paths in rows:
        val = f'{actual:.12g}' if isinstance(actual, float) else str(actual)
        print('\t'.join((entry, key, shown, val, why, tier, status, '; '.join(dict.fromkeys(paths)))))
    print('\nR = raw-derived / inspected shipped sample; P = saved per-image/step aggregate; S = summary transcription.')
    for (entry, key), (claimed, warning) in PRODUCER_TIER_WARNINGS.items():
        if read_json(f'{ASSETS}/{entry}.facts.json').get(key, {}).get('tier') == claimed:
            print(f'TIER-WARN {entry} {key}: claims {claimed} but {warning}.')
    for entry, msg in notes:
        print(f'NOTE {entry}: {msg}')
    for entry, msg in findings:
        print(f'FAIL {entry}: {msg}')
    for entry, waivers in RAW_WAIVERS.items():
        for keys, why in waivers.items():
            print(f'RAW-WAIVER {entry} {keys}: {why} Summary values verified.')
    count = Counter(r[6] for r in rows)
    tiers = Counter(r[5].split('/')[-1] for r in rows if not r[1].startswith('@'))
    print(f'{sum(tiers.values())} fact rows ({dict(sorted(tiers.items()))}), '
          f'{required_count} required headline/lede/prologue fact keys, '
          f'{count["PASS"]} passing rows including literals, {count["FAIL"]} failing rows, '
          f'{len(findings)} coverage/tier-schema/input failures.')
    return int(bool(findings or count['FAIL']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('build', nargs='?', default=BUILD, help=f'built HTML page (default {BUILD})')
    raise SystemExit(main(parser.parse_args().build))
