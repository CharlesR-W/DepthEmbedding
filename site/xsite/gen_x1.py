"""Build the X1 entry from pinned trunk modes and the archived comparison outputs."""
from __future__ import annotations

from itertools import combinations
import json

import numpy as np

from ops_x1 import PREFIX, REF, load_operators, op_id, reference_rotations, site_cos, zoo_accuracy
from ops import net_metadata
from validate import Inputs, require, x1_split
from xlib import Out, PALETTE, dots, lines

GROUPS = dict(c10='CIFAR-10', c100='CIFAR-100', c100c='C100 coarse', in32='ImageNet-32',
              svhn='SVHN', anim='Animals', veh='Vehicles', rand='Shuffled labels', init='Untrained')
SUMMARY = 'local/xp/out/x1_summary.json'
RAW = 'data/fetch/crw-depth-xp-results/compare/c10.npz'
SELECT = [0, 3, 6, 9]
VIEWER = '#embedding?m=modes&lay=grid&nets=r20s0.c100-s0.svhn-s0.in32-s0&ref=r20s0&step=64000&v=own&c=3&K=16&norm=site&dm=frac&al=1&basis=pooled&ax=0.1.2&d3=0&color=class&depth=9'


def group(net):
    return net.split('-')[0].removesuffix('x')


def token(key):
    return '{{fact:x.x1-data.' + key + '}}'


def replay(v, perm, train, test):
    """Replay every ordered A/B fit after mapping B's rows by perm (possibly identity)."""
    k = v.shape[-1]
    a, b = v[:, :, train], v[:, :, perm[train]]
    at, bt = v[:, :, test], v[:, :, perm[test]]
    a, b = a.reshape(len(v), -1, k), b.reshape(len(v), -1, k)
    out = np.empty((len(v), len(v), v.shape[1]))
    for i in range(len(v)):
        u, _, vt = np.linalg.svd(b.transpose(0, 2, 1) @ a[i])
        for j, q in enumerate(u @ vt):
            out[i, j] = site_cos(at[i], bt[j] @ q)
    return out


def main():
    out, inputs = Out('x1-data'), Inputs()
    comp, modes, chain = load_operators(inputs)
    nets = comp['nets'].tolist()
    train, test = x1_split(comp['labels'], comp['perm'], comp['within'])
    rots, error, error_i8 = reference_rotations(comp, modes, train, test)
    r = nets.index(REF)
    panels = []
    for j, net in enumerate(nets):
        if net == REF:
            label = 'CIFAR-10 · seed 0 (reference)'
        elif net == 'c10-s1':
            label = 'CIFAR-10 · seed 1 (zoo)'
        elif net == 'c10x-s0':
            label = 'CIFAR-10 · seed 0 (X1 trainer)'
        elif net.startswith('init-'):
            label = f'Untrained · seed {net[-1]}'
        else:
            label = net_metadata(inputs, net)['label']
        panels.append(dict(op=op_id(net), label=label, R=rots[j],
                           stat=dict(label='joint agreement', values=comp[PREFIX + 'cos'][r, j])))
    defaults = [panels[nets.index(n)] for n in (REF, 'c100-s0', 'svhn-s0', 'in32-s0')]
    fig = dict(type='emb', schema=1, layout='row', panels=defaults, panelMenu=panels, kinds=['joint'], kind='joint', site='b9',
               K=16, dims=2, axes='pooled:0', norm='site', colour={'by': 'class'}, height=270,
               filmstrip=[chain[i] for i in SELECT], viewerLink=VIEWER)
    out.chart('emb_lead', fig)

    v = np.stack(modes)
    results = [replay(v, p, train, test) for p in (np.arange(len(comp['labels'])), comp['perm'], comp['within'])]
    for suffix, result in zip(('', '_perm', '_within'), results):
        err = np.max(np.abs(result - comp[PREFIX + 'cos' + suffix]))
        require(err < 1e-6, f'X1 all-pair agreement replay {suffix or "raw"}: {err:.3g}')
        print(f'X1 all-pair agreement replay {suffix or "raw"}: {err:.3g}')
    raw, perm, within = results
    indices = {g: [i for i, n in enumerate(nets) if group(n) == g] for g in GROUPS}
    # Match the archive's convention: for an unordered pair, use A/B in stored network order.
    def cell(a, b, metric=raw):
        pairs = list(combinations(indices[a], 2)) if a == b else [tuple(sorted((i, j))) for i in indices[a] for j in indices[b]]
        return np.stack([metric[i, j] for i, j in pairs]).mean(0)

    off = ~np.eye(len(nets), dtype=bool)
    null = perm[off].mean(0)
    cols = dict(zip(GROUPS, PALETTE['classes']))
    cols['rand'] = PALETTE['classes'][9]
    accent, mute = PALETTE['accent'], PALETTE['mute']

    def trace(a, b, name=None, color=None):
        return dict(name=name or f'{GROUPS[a]} × {GROUPS[b]}', y=cell(a, b), color=color or cols[b],
                    dash='dot' if {a, b} & {'rand', 'init'} else a == b)

    def trend(series, site_names=chain, **kwargs):
        return lines(list(range(len(site_names))), series, 'Site', 'Joint agreement', yrange=(-.1, 1),
                     xticklabels=site_names, **kwargs)

    chance = dict(name='Unrestricted permutation', y=null, dash='dot', color=mute)
    out.chart('trend', trend([trace('c10', g, 'CIFAR-10 run pairs' if g == 'c10' else f'× {GROUPS[g]}')
                              for g in ('c10', 'c100', 'in32', 'svhn', 'init', 'rand')] + [chance]))
    out.chart('coarse', trend([trace('c100', 'c100c', 'Fine × coarse', accent),
                                trace('c100', 'c100', 'Fine seeds'), trace('c100c', 'c100c', 'Coarse seeds')]))
    # All pipeline pairs, with same-pipeline references in their own panel rather than one unreadable legend.
    groups = list(GROUPS)
    out.chart('all_pairs', {'panels': [trend([trace(a, b, f'× {GROUPS[b]}') for b in groups[i:]
                                             if a != b or len(indices[a]) > 1] + [chance],
                                            title=GROUPS[a]) for i, a in enumerate(groups)]})

    # A row for each pair of pipelines, including seed pairs. No use of the word ‘ceiling’:
    # comparisons across pipelines can exceed a within-pipeline seed reference.
    rows = [(a, b) for i, a in enumerate(groups) for b in groups[i:] if a != b or len(indices[a]) > 1]
    rows.sort(key=lambda ab: (ab[0] in ('rand', 'init') or ab[1] in ('rand', 'init'), ab[0] != ab[1],
                              -cell(*ab)[-1]))
    short = dict(c10='C10', c100='C100', c100c='C100 coarse', in32='I32', svhn='SVHN',
                 anim='Animals', veh='Vehicles', rand='Shuffle', init='Untrained')
    labels = [f'{short[a]} × {short[b]}' if a != b else f'{short[a]} seeds' for a, b in rows]
    labels[rows.index(('init', 'init'))] = 'Untrained pair'
    labels[rows.index(('c10', 'c10'))] = 'C10 run pairs'
    ladder = []
    for l in (0, len(chain) - 1):
        ladder.append(dots(labels, [dict(name='Observed', values=[cell(a, b)[l] for a, b in rows], color=accent),
                                     dict(name='Within-class shuffle', values=[cell(a, b, within)[l] for a, b in rows],
                                          marker='diamond', hollow=True, color=cols['svhn']),
                                     dict(name='Unrestricted permutation', values=[cell(a, b, perm)[l] for a, b in rows],
                                          marker='square', hollow=True, color=mute)],
                           'Joint agreement', title=chain[l], xrange=(-.1, 1)))
    out.chart('ladder', {'panels': ladder})

    # Reuse the live line frame as a scatter plot (no connecting segments). Each x is
    # an observed neighbour overlap, so no categorical spacing or binned aggregation.
    b = nets.index('c10-s1')
    diag = np.arange(len(chain))
    scatter_data = [(comp['nn'][r, b, diag, diag], raw[r, b], 'Matched images'),
                    (comp['nn_within'][r, b, :len(chain)], within[r, b], 'Within-class shuffle')]
    xs = sorted({0., 1.} | {float(x) for xx, *_ in scatter_data for x in xx})
    scatter = []
    for xx, yy, label in scatter_data:
        for loc, sites in [('stem', [0]), ('b1–b8', list(range(1, len(chain)-1))), ('b9', [len(chain)-1])]:
            vals = {float(xx[i]): yy[i] for i in sites}
            scatter.append(dict(name=f'{label} · {loc}', y=[vals.get(x) for x in xs], noline=True,
                                dots=5 if len(sites) == 1 else 3,
                                color=cols['svhn'] if label.startswith('Within') else accent))
    out.chart('overlap', lines(xs, scatter, 'Share of the same 15 neighbours', 'Joint agreement',
                               yrange=(0, 1), xticks=[0, .25, .5, .75, 1]))

    def fact(key, value, source, *, tier='R', **opts):
        out.fact(key, value, tier=tier, source=source, **opts)

    for a, b in [('c10', 'c10'), ('c10', 'c100'), ('c10', 'in32'), ('c100', 'in32'), ('c10', 'svhn'),
                 ('c10', 'init'), ('c10', 'rand'), ('c100', 'c100'), ('c100c', 'c100c'), ('c100', 'c100c'),
                 ('svhn', 'svhn'), ('in32', 'in32')]:
        for l in (0, 9):
            fact(f'pair.{a}_{b}.{chain[l]}', cell(a, b)[l],
                 f'{RAW}:knn15_trunk_cos replay, upper triangle in nets order, mean of {a}|{b} at {chain[l]}', digits=2)
    i, j = (nets.index(n) for n in (REF, 'c10-s1'))
    for l in (0, 9):
        for k, m, suffix in [('raw', raw, ''), ('within', within, '_within')]:
            fact(f'zoo.{k}.{chain[l]}', m[i, j, l],
                 f'{RAW}:knn15_trunk_cos{suffix} replay, c10-s0 versus c10-s1 at {chain[l]}', digits=2)
    fact('replay.raw', error, f'{RAW}:knn15_trunk_cos, max absolute replay error over 18 reference rotations × 10 sites', digits=16)
    out.facts['replay.raw']['text'] = f'{error:.2g}'
    fact('replay.i8', error_i8, 'Same comparison with decoded int8 modes, raw rotations retained', digits=6)
    fact('sample.probe', len(comp['labels']), f'{RAW}:len(labels)')
    fact('sample.half', len(test), f'{RAW}:len(labels)//2 following stage_compare.py:run')
    fact('sample.networks', len(nets), f'{RAW}:len(nets)')
    fact('sample.modes', v.shape[-1], 'First 16 from the pinned raw trunk NPZ vecs, as in stage_compare.py:run')
    fact('null.perm.b9', null[-1], f'{RAW}:knn15_trunk_cos_perm replay, mean over off-diagonal ordered pairs at b9', digits=2)

    summary = json.loads(inputs.file(SUMMARY).read_text())
    for net in nets:
        if net.startswith('init-'):
            continue
        zoo = zoo_accuracy(net)
        if zoo:
            acc, source = zoo
        else:
            acc = net_metadata(inputs, net)['acc']
            require(acc == summary['training'][net]['test_acc'], f'{net}: test accuracy differs from X1 summary')
            source = f'xp/out/x1_summary.json:training.{net}.test_acc'
        fact('acc.' + net.replace('-', '_'), acc, source, tier='S', pct=True, digits=2)
    for probe, pairs in {'svhn': [('c10','c10'), ('svhn','svhn'), ('c10','anim'), ('c10','veh'), ('c10','in32'),
                                  ('c10','svhn'), ('c10','init'), ('c10','rand')],
                         'in32': [('c100','in32'), ('c100','c100'), ('in32','in32'), ('c100','c100c'), ('c100c','c100c')]}.items():
        for a, b in pairs:
            z = summary['measures'][probe]['cells']['joint_knn15_trunk'][f'{a}|{b}']['mean']
            fact(f'probe.{probe}.{a}_{b}', z[9], f'xp/out/x1_summary.json:measures.{probe}.cells.joint_knn15_trunk.{a}|{b}.mean[9]', tier='S', digits=2)
    for a, b in [('c10', 'c10'), ('c10', 'c100'), ('c10', 'svhn'), ('c100', 'c100c')]:
        for variant in ('knn15_full', 'ea100_trunk'):
            z = summary['measures']['c10']['cells']['joint_' + variant][f'{a}|{b}']['mean']
            fact(f'{variant}.{a}_{b}.b9', z[9], f'xp/out/x1_summary.json:measures.c10.cells.joint_{variant}.{a}|{b}.mean[9]',
                 tier='S', digits=2)
    card = dict(answer='Several natural-image pipelines gave similar embeddings on the same CIFAR-10 probe; SVHN was further away.',
                numbers=[token('pair.c10_c100.b9') + ' CIFAR-10 × CIFAR-100 at b9', token('pair.c10_in32.b9') + ' CIFAR-10 × ImageNet-32 at b9',
                         token('pair.c10_svhn.b9') + ' CIFAR-10 × SVHN at b9', token('pair.c10_c10.b9') + ' CIFAR-10 run pairs at b9'])
    thumb = out.fig_path('thumb')
    if thumb.exists():
        card['thumb'] = f'assets/fig/{thumb.name}'
    out.chart('card', card)
    out.write()
    print(f'X1 reference max errors: raw {error:.9g}; decoded shipping int8 {error_i8:.9g}')


if __name__ == '__main__':
    main()
