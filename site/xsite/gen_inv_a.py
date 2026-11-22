"""Generate audited kernel and source-coordinate experiment assets from saved runs."""
from __future__ import annotations

import json

import numpy as np

from ops_inv import EA, PR, VOL
from validate import Inputs
from xlib import DATA, Out, PALETTE, SITES, bars, lines

KNN = 'r20s0-c10-knn15'
P2 = f'{VOL}/joint/P2__pr300__L12'
SUMMARY = 'local/data/invert/results.json'
KERNELS = [('knn15', 'k-NN 15'), ('ea300', 'Shannon 300'), ('pr300', 'Participation ratio 300')]


ARROWS = 60   # Set explicitly in the figure specs; the reader can change this.


def packaged(op, file='op.json'):
    return json.loads((DATA / 'sitex/ops' / op / file).read_text())


def attack_display(out, site, budgets=(2,)):
    op = packaged(PR)
    axes = np.asarray(op['axes']['pooled'])[:, :2]
    li = op['chain'].index(site)
    with np.load(DATA / 'fetch' / P2 / 'coords_H.npz', allow_pickle=False) as clean:
        for budget in budgets:
            path = f'coords_pgd{budget}.npz'
            with np.load(DATA / 'fetch' / P2 / path, allow_pickle=False) as attacked:
                for key, kind in [('y', 'joint'), ('onestep', 'source')]:
                    rows = attacked[key][li, :, :op['K']].astype(float)
                    delta = rows - clean[key][li, :len(rows), :op['K']]
                    out.fact(f'display.pgd{budget}.{kind}', np.square(delta @ axes).sum() / np.square(delta).sum(),
                             pct=True, digits=1, tier='R', source=f'{P2}/{{coords_H.npz,{path}}}:{key}, {site}, all pairs; shipped joint PCA axes')
    out.fact('attack.arrows', ARROWS, tier='S', source='gen_inv_a.py:ARROWS, initial display setting')


def token(out, name):
    return '{{fact:x.' + out.id + '.' + name + '}}'


def card(out, answer, numbers):
    value = dict(answer=answer, numbers=numbers)
    if out.fig_path('thumb').is_file():
        value['thumb'] = f'assets/fig/x_{out.id}_thumb.webp'
    out.chart('card', value)


def emb(panels, kinds, site, **fields):
    return dict(type='emb', schema=1, panels=panels, kinds=kinds, kind=kinds[0], site=site,
                K=16, dims=2, axes='pooled', norm='site', colour={'by': 'class'}, **fields)


def by_site(series, ylabel, **kw):
    return lines(list(range(len(SITES))), series, 'Site', ylabel, xticklabels=SITES, **kw)


def kernel(inputs, report):
    out = Out('inv-kernel')
    fact = lambda name: token(out, name)
    panels, p2, p8, projection = [], [], [], []
    split = report['kernel']
    for (rule, label), op, alt in zip(KERNELS, (KNN, EA, PR), ('knn15', 'ea400', 'pr400'), strict=True):
        rec = inputs.bundle(op)
        with np.load(inputs.file(rec['file']), allow_pickle=False) as z:
            mass = np.square(z['vecs'][..., :16].astype(float)).sum(axis=0)
        top = np.argsort(mass, axis=0, kind='stable')[-2:]
        share = np.take_along_axis(mass, top, axis=0).sum(axis=0) / mass.sum(axis=0)
        m = int(share.argmax())
        axes = np.asarray(packaged(op)['axes']['pooled'])[:, :2]
        projection.append(float(np.square(axes[m]).sum()))
        out.fact(f'display.{rule}', projection[-1], pct=True, digits=2, tier='R',
                 source=f'data/sitex/ops/{op}/op.json:axes.pooled; squared projection of peak-mode unit axis')
        saved = report['modeloc'][f'P2__{rule}__L12']['top2'][:16]
        np.testing.assert_allclose(share, saved, atol=2e-6, rtol=0)
        panels.append(dict(op=op, label=f'{label} · mode {m+1}',
                           highlight=dict(probe=top[:, m].tolist(), label='two images in the peak mode')))
        p2.append(dict(name=label, y=share, dots=3))
        p8.append(dict(name=label if rule == alt else label.replace('300', '400'),
                       y=report['modeloc'][f'P8__{alt}__L12']['top2'][:16], dots=3))
        out.fact(f'top2.p2.{rule}', share[m], pct=True, digits=1, tier='R',
                 source=f"{rec['file']}:vecs, max first 16 top-two shares (depth-summed squared loadings)")
    p8_ea = np.array(report['modeloc']['P8__ea400__L12']['top2'])
    out.fact('top2.p8.ea400', p8_ea[:16].max(), pct=True, digits=1, tier='S',
             source=f'{SUMMARY}:modeloc.P8__ea400__L12.top2, max of first 16')
    out.fact('top2.p8.ea400_count', int((p8_ea > 0.10).sum()), tier='S',
             source=f'{SUMMARY}:modeloc.P8__ea400__L12.top2, count above 0.1 among first 64')
    out.fact('hub.raw_min', min(split[s]['raw_cos_hub']['max'] for s in SITES), digits=0, tier='S',
             source=f'{SUMMARY}:kernel.*.raw_cos_hub.max, min across sites')
    out.fact('hub.raw_max', max(split[s]['raw_cos_hub']['max'] for s in SITES), digits=0, tier='S',
             source=f'{SUMMARY}:kernel.*.raw_cos_hub.max, max across sites')
    for rule, _ in [*KERNELS, ('ds500', 'Global width')]:
        out.fact(f'hub.{rule}_max', max(split[s][rule]['hub']['max'] for s in SITES), digits=0, tier='S',
                 source=f'{SUMMARY}:kernel.*.{rule}.hub.max, max across sites')
    out.fact('row.pr300_final_max', max(split[s]['pr300']['maxw'] for s in SITES), pct=True, digits=1, tier='S',
             source=f'{SUMMARY}:kernel.*.pr300.maxw, max across sites after balancing')
    p2_idx, p8_idx, held, labels = [inputs.value(f'{VOL}/splits.npz:{key}') for key in ('P2', 'P8', 'H', 'labels')]
    class_counts = np.unique(labels[p2_idx], return_counts=True)[1]
    assert np.all(class_counts == class_counts[0]) and np.all(held[:-1] < held[1:])
    for name, value in [('p2', len(p2_idx)), ('per_class', class_counts[0]),
                        ('p8_extra', len(np.setdiff1d(p8_idx, p2_idx))), ('held', len(held))]:
        out.fact(f'probe.{name}', int(value), tier='R', source=f'{VOL}/splits.npz:P2,P8,H,labels; counts')
    out.chart('peak_projection', bars([label for _, label in KERNELS], [dict(name='Peak-mode axis', values=projection)],
                                      'Squared projection onto the plane', fmt='pct'))
    out.chart('emb_lead', emb(panels, ['joint'], 'b9', height=395))
    out.chart('top2', dict(panels=[dict(title=f'N = {n:,}', **lines(list(range(1, 17)), series,
                        'Joint mode', 'Two-image share', fmt='pct', xticks=[1, 4, 8, 12, 16]))
                        for n, series in [(2000, p2), (8000, p8)]], shared=True))
    stab = {rule: [split[s][rule]['stab']['16'] for s in SITES] for rule, _ in KERNELS}
    out.chart('stability', by_site([dict(name=label, y=stab[rule]) for rule, label in KERNELS], 'Reproducibility'))
    widths = {rule: np.array([split[s][rule]['perp'] for s in SITES]) for rule, _ in KERNELS}
    for rule, width in widths.items():
        for bound, value in [('min', width.min()), ('max', width.max())]:
            out.fact(f'width.{rule}.{bound}', value, digits=0, tier='S', source=f'{SUMMARY}:kernel.*.{rule}.perp')
    for name, lhs, rhs in [('global_over_pr', [split[s]['ds500']['stab']['16'] for s in SITES], stab['pr300']),
                           ('pr_over_shannon', stab['pr300'], stab['ea300'])]:
        out.fact(f'split.{name}', int(np.sum(np.asarray(lhs) > rhs)), tier='S', source=f'{SUMMARY}:kernel.*.*.stab.16')
    out.fact('split.sites', len(SITES), tier='R', source='Study chain length, xlib.py:SITES')
    refit = json.loads((DATA / 'invert/refit.json').read_text())
    for k in (16, 64):
        for rule in ('ea300', 'pr300'):
            series = np.asarray(refit[f'{rule}|{k}']['held'])
            for bound, value in [('min', series.min()), ('max', series.max())]:
                out.fact(f'refit.k{k}.{rule}.{bound}', value, tier='S', source=f'data/invert/refit.json:{rule}|{k}.held')
    out.fact('refit.k16.shannon_wins', int(np.sum(np.asarray(refit['ea300|16']['held']) > refit['pr300|16']['held'])),
             tier='S', source='data/invert/refit.json: ea300|16 vs pr300|16 held')
    x = sorted({15, 1000, *(float(v) for w in widths.values() for v in w)})
    scatter = []
    for rule, label in KERNELS:
        xy = dict(zip(widths[rule], stab[rule], strict=True))
        assert len(xy) == len(SITES)
        scatter.append(dict(name=label, y=[xy.get(v) for v in x], dots=5, noline=True))
    out.chart('width', lines(x, scatter, 'Final walk perplexity (log scale)', 'Reproducibility', xscale='log',
                             xticks=[20, 50, 100, 200, 500, 1000]))
    out.chart('hubs', by_site([dict(name=label, y=[split[s][rule]['hub']['max'] for s in SITES])
                               for rule, label in KERNELS] + [dict(name='Raw cosine', dash=True,
                               y=[split[s]['raw_cos_hub']['max'] for s in SITES])], 'Peak top-15 occurrence', yrange=(0, 260)))
    card(out, 'We chose participation ratio for a smooth row without the extreme pair localisation seen with Shannon; it did not win every stability test.',
         [f"{fact('top2.p2.pr300')} vs {fact('top2.p2.ea300')} peak two-image share (PR vs Shannon, P2)",
          f"{fact('top2.p8.ea400')} peak two-image share (Shannon, P8)",
          f"{fact('hub.raw_max')} vs {fact('hub.pr300_max')} peak top-15 occurrence (raw vs PR)"])
    out.write()


def sources(inputs, report):
    out = Out('inv-sources')
    fact = lambda name: token(out, name)
    with np.load(inputs.file(f'{P2}/modes.npz'), allow_pickle=False) as op:
        c, t, vals, sites = op['c'], op['t'], op['vals'][:256], op['sites'].tolist()
    assert sites == list(SITES)
    a = np.diag(np.ones(len(c) - 1), 1)
    a += a.T
    depth = np.stack([np.diag(1 / (2*c) + t*a.sum(1) - mu) - t*a for mu in vals])
    filt = np.linalg.inv(depth) / (2*c)
    abs_weights = abs(filt)
    condition, max_gain = np.linalg.cond(depth), abs_weights.sum(axis=2).max(axis=1)
    for k in (16, 64, 256):
        for name, values, digits in [('condition', condition, 0), ('gain', max_gain, 2)]:
            out.fact(f'depth.k{k}.{name}', values[:k].max(), digits=digits, tier='R',
                     source=f'{P2}/modes.npz:c,t,vals; maximum {name} through mode {k}')
    modes = list(range(1, len(vals) + 1))
    out.chart('sensitivity', dict(panels=[dict(title=label, **lines(modes, [dict(name=label, y=values)],
        'Joint mode', label, yrange=None)) for label, values in [('Condition number of T', condition),
                                                                ('Largest absolute row sum of F', max_gain)]]))
    share = abs_weights.diagonal(axis1=1, axis2=2) / abs_weights.sum(axis=2)
    with np.load(inputs.file(f'{P2}/coords_H.npz'), allow_pickle=False) as hc, \
         np.load(inputs.file(f'{P2}/coords_pgd2.npz'), allow_pickle=False) as pa:
        y, s = [hc[key][:, :1000, :16].astype(float) for key in ('y', 'onestep')]
        dy, ds = [pa[key][..., :16].astype(float) - base for key, base in [('y', y), ('onestep', s)]]
    np.testing.assert_allclose(np.einsum('klj,jik->lik', filt[:16], ds), dy, rtol=0, atol=1e-8)
    ds_cut = ds.copy()
    ds_cut[sites.index('b7'):] = 0
    dy_cut = np.einsum('klj,jik->lik', filt[:16], ds_cut)

    def displacement(delta, base):
        return np.mean((delta / base.std(axis=1)[:, None, :])**2, axis=(1, 2))

    draws = [('Joint', displacement(dy, y)), ('Source', displacement(ds, s)),
             ('Joint, b7 and later source changes set to zero', displacement(dy_cut, y))]
    p8 = report['rt']['P8__pr400__L12']

    def rt(k, key):
        return np.array([p8[f'{s}__K{k}'][key] for s in sites])

    p8_share = rt(16, 'E1|share_ll')
    p2_share = share[:16].mean(axis=0)
    for name, tier, values, src in [('p2', 'R', p2_share, f'{P2}/modes.npz:c,t,vals[:16] via absolute filter row weight'),
                                    ('p8', 'S', p8_share, f'{SUMMARY}:rt.P8__pr400__L12.*__K16.E1|share_ll')]:
        for bound, fn in [('min', min), ('max', max)]:
            out.fact(f'share.{name}.conv_{bound}', fn(values[:10]), pct=True, tier=tier, source=f'{src}, {bound} over stem–b9')
    for metric, values, key, site in [('joint', draws[0][1], 'y', 'b5'), ('source', draws[1][1], 'onestep', 'b5'),
                                      ('source', draws[1][1], 'onestep', 'b6'), ('source', draws[1][1], 'onestep', 'b8'),
                                      ('source', draws[1][1], 'onestep', 'b9'), ('truncated', draws[2][1], 'onestep then filter', 'b5')]:
        out.fact(f'attack.{metric}.{site}', values[sites.index(site)], digits=3, tier='R',
                 source=f'{P2}/coords_{{H,pgd2}}.npz:{key}; first 16 modes, all 1000 pairs, {site}, squared standardized mean; '
                        + ('set Δs=0 at b7 through logits, then apply filter; use clean-H joint scale' if metric == 'truncated' else 'use corresponding clean-H scale'))
    out.fact('patch.b5.swap', rt(16, 'E4|swap|prog_D')[5], pct=True, tier='S',
             source=f'{SUMMARY}:rt.P8__pr400__L12.b5__K16.E4|swap|prog_D')
    out.fact('attack.n', len(y[0]), tier='R', source=f'{P2}/coords_pgd2.npz:y, image count')
    out.fact('attack.shown', packaged(PR, 'ov/pgd2.json')['n'], tier='R', source=f'data/sitex/ops/{PR}/ov/pgd2.json:n')
    path = DATA / 'stretch/rt_points/P2__pr300__L12/b4__K16.npz'
    with np.load(path, allow_pickle=False) as rerun:
        out.fact('patch.cohort', json.loads(str(rerun['meta']))['m_full'], tier='S',
                 source='P2 rerun meta.m_full; same default m in invert/stage_rt.py for P8')
    attack_display(out, 'b5')

    out.chart('emb_lead', emb([dict(op=PR, label='PGD 2/255')], ['joint', 'source'], 'b5', height=465,
                              overlays=[{'set': 'pgd2', 'draw': 'arrows', 'from': 'H-pgd'}],
                              filmstrip=['b5', 'b8', 'b9'], arrows=ARROWS))
    out.chart('share', by_site([dict(name=f'P2 PR 300 · K {k}', y=share[:k].mean(axis=0), dash=k == 64,
                                      color=PALETTE['sets'][0]) for k in (16, 64)] +
                               [dict(name=f'P8 PR 400 · K {k}', y=rt(k, 'E1|share_ll'), dash=k == 64,
                                      color=PALETTE['sets'][1]) for k in (16, 64)], 'Own-source share', fmt='pct'))
    out.chart('attack', by_site([dict(name=label, y=values, dash=i == 2) for i, (label, values) in enumerate(draws)],
                                'Mean squared displacement', yrange=(0, 2)))
    out.chart('slope', by_site([dict(name=f'K = {k}', y=rt(k, 'E1|slope'), dash=k == 64) for k in (16, 64)],
                                'Source-on-joint slope', yrange=None))
    out.chart('progress', by_site([dict(name=label, y=rt(16, f'E4|{key}|prog_D'), dash=key.startswith('rand~')) for key, label in
                                  [('swap', 'True activation swap'), ('match', 'Source match'), ('resid', 'Residual edit'),
                                   ('rand~resid', 'Random, residual-sized')]], 'Mean progress toward target',
                                  yrange=None, refline=dict(y=1, label='target projection')))
    card(out, 'The joint formula needs other sites’ kernel rows; the source formula needs only this row.',
         [f"{fact('share.p8.conv_min')}–{fact('share.p8.conv_max')} relative own-source weight (P8, conv sites)",
          f"{fact('attack.joint.b5')} → {fact('attack.truncated.b5')} b5 joint displacement, suppressing late source changes",
          f"{fact('depth.k16.gain')} max filter row gain (P2, K = 16)"])
    out.write()


if __name__ == '__main__':
    inputs = Inputs()
    report = json.loads(inputs.file(SUMMARY).read_text())
    kernel(inputs, report)
    sources(inputs, report)
