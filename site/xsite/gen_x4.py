"""Build the X4 entry from the saved paths, modes and per-image gains."""
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from opdata import require
from ops_x4 import STARTS, sample_pairs
from validate import Inputs, check_probe
from xlib import Out, bars, heat, lines

ID = 'x4-paths'
ROOT = Path(__file__).resolve().parents[2]
BASE = 'crw-depth-xp-results/x4/'
ALL_STARTS = ('pixels', 'stem', 'b3', 'b6', 'b8', 'b9')
ANSWER = ('Before logits, the sampled paths rarely concentrated angular motion in a short window. '
          'Probe class mass could change steeply even on the affine control.')
SHARE_THRESHOLD = .25


def f(key):
    return '{{fact:x.' + ID + '.' + key + '}}'


def top_two(energy, axis):
    return np.take(np.partition(energy, -2, axis=axis), [-2, -1], axis=axis).sum(axis) / energy.sum(axis)


def chart_lines(sites, runs, key, ylabel, **opts):
    return lines(list(range(len(sites))), [
        dict(name='From b9 (affine)' if start == 'b9' else f'From {start}',
             y=[dict(zip(runs[start][key][0], runs[start][key][1])).get(site) for site in sites],
             dots=True, **({'dash': True, 'width': 3} if start == 'b9' else {}))
        for start in STARTS
    ], 'Network site', ylabel, xticklabels=sites,
                 xticks=[sites.index(s) for s in ('stem', 'b3', 'b6', 'b9', 'logits') if s in sites],
                 colors='ordinal', **opts)


def main():
    inputs, out = Inputs(), Out(ID)
    runs, loc, leaves, switches = {}, {}, [], []
    n_pairs = n_probe = None
    all_sigma_min, refine_max = [], []
    pre_logits, logits_hits = [], {}
    shown_total = 0
    with np.load(inputs.file(BASE + 'pairs.npz'), allow_pickle=False) as pair_file:
        pair_ids, pair_kind = pair_file['pair_idx'], pair_file['kind']
    for start in ALL_STARTS:
        fname = f'paths_{start}.npz'
        if start in STARTS:
            op_id = f'x4-{start}-suffix'
            record = inputs.bundle(op_id)
            path, source = inputs.file(record['file']), record['file']
        else:
            source = f'xp/out/x4/{fname}'
            path = ROOT / source
        with np.load(path, allow_pickle=False) as z:
            sites, chain, kind = z['sites'].tolist(), z['chain'].tolist(), z['kind']
            require(np.array_equal(z['pair_idx'], pair_ids) and np.array_equal(kind, pair_kind),
                    f'{path}: pair order differs')
            same, diff = kind == 0, kind == 1
            count = int(same.sum())
            require(count == diff.sum() and n_pairs in (None, count), f'{path}: unexpected pair counts')
            n_pairs = count
            pred, labels = z['pred'][same], z['labels'][same, 0]
            left = int(np.any(pred != labels[:, None], axis=1).sum())
            leaves.append(left / count)
            require(pred.shape[1] == len(z['s256']), f'{path}: wrong prediction grid')
            out.fact(f'leaves.{start}.n', left, tier='P',
                     source=source + ':pred, labels, kind; same-class pairs with any argmax unequal to label')
            out.fact(f'leaves.{start}.share', leaves[-1], pct=True, tier='P',
                     source=source + ':pred, labels, kind; same-class fraction with any argmax unequal to label')
            sigma = z['sigma_raw']
            all_sigma_min.append(np.nanmin(sigma))
            share = z['share_unit']
            where = (np.asarray(sites) != 'logits') & (np.asarray(sites) != start)
            hits = (share > SHARE_THRESHOLD) & where[None, :]
            ref = set(z['ref_rows'].tolist())
            pre_logits.extend((start, int(i), ', '.join(np.asarray(sites)[hits[i]]), int(i) in ref)
                              for i in np.flatnonzero(hits.any(axis=1)))
            logits_hits[start] = int((share[:, sites.index('logits')] > SHARE_THRESHOLD).sum())
            if start in STARTS:
                shown_total += len(share)
            refine_max.append(np.nanmax(np.nanmedian(abs(z['share1024_unit'] - share[z['ref_rows']]), axis=0)))
            if start not in STARTS:
                continue
            s, ab = z['s64'], z['labels'][diff]
            mass = z['emb_mass'][:, diff]
            path_idx = np.arange(len(ab))
            delta = np.stack([m[path_idx, :, ab[:, 1]] - m[path_idx, :, ab[:, 0]] for m in mass])
            steep = np.max(abs(np.diff(delta, axis=-1) / np.diff(s)), axis=-1)
            runs[start] = dict(share=(sites, np.median(share[diff], axis=0)),
                               sharp=(sites, np.median(sigma[diff], axis=0)),
                               slope=(chain, np.median(steep, axis=1)))
            out.fact(f'share.{start}.logits', runs[start]['share'][1][sites.index('logits')], digits=1, pct=True, tier='P',
                     source=source + ':share_unit, kind, sites; median different-class logits value')
            out.fact(f'slope.{start}.b9', runs[start]['slope'][1][chain.index('b9')], digits=1, tier='P',
                     source=source + ':emb_mass, s64, labels, kind, chain; b9 median different-class max absolute slope')
            joint = top_two(np.square(z['probe_y'], dtype=np.float64).sum(axis=0), 0)
            site = top_two(np.square(z['probe_site_y'], dtype=np.float64), 1)
            for mode, shares, array in (('joint', joint, 'probe_y'), ('site', site, 'probe_site_y')):
                out.fact(f'loc.{start}.{mode}', shares.max(), digits=1, pct=True, tier='R',
                         source=source + f':{array}; max top-two-image squared-norm share over all saved modes'
                         + (' and sites' if mode == 'site' else ', pooling across sites before ranking'))
            n_probe = len(z['probe_idx'])
            label = 'b9 (affine)' if start == 'b9' else 'Pixels' if start == 'pixels' else start
            film_sites = [chain[0], chain[len(chain)//2], chain[-1]] if start == 'b9' else [chain[0], 'b9', chain[-1]]
            switches.append(dict(label=label, panels=[dict(op=op_id)], paths=[f'x4-{start}-paths'],
                                 site='b9', filmstrip=film_sites,
                                 **({'kind': 'site'} if start == 'b9' else {})))
            loc[start] = (joint, site, chain)
            if start == 'pixels':
                out.fact('grid.points', len(z['s256']), tier='R', source=source + ':s256 length')
                out.fact('grid.draw', len(s), tier='R', source=source + ':s64 length')
                out.fact('grid.uniform', (len(z['s256']) - 1) // 16 / (len(z['s256']) - 1),
                         pct=True, digits=2, tier='R', source=source + ':s256; window is one sixteenth of steps')
                # Show the label grid shared by all the line charts, not a hard-coded length.
                plot_sites = chain
    out.fact('pairs.per_kind', n_pairs, tier='P', source=BASE + 'pairs.npz:kind; count per stratum')
    out.fact('pairs.display', len(sample_pairs(pair_kind)), tier='R', source=BASE + 'pairs.npz:kind, replay site/xsite/ops_x4.py:sample_pairs')
    out.fact('pairs.unique_endpoints', np.unique(pair_ids).size, tier='R', source=BASE + 'pairs.npz:pair_idx; distinct test image IDs over all pairs')
    out.fact('pairs.margin_min', 1e-3, digits=3, tier='S', source='xp/stage_x4.py:MARGIN_MIN, fixed eligibility criterion')
    out.fact('probe.n', n_probe, tier='R', source=BASE + 'paths_pixels.npz:probe_idx length')
    hit_src = 'paths_<start>.npz:share_unit, sites, ref_rows; each start–pair with >25% at any strictly downstream pre-logit site'
    out.fact('starts.tested', len(logits_hits), tier='R', source='xp/out/x4/paths_*.npz; number of measured starts')
    out.fact('starts.shown', len(switches), tier='R', source='lead figure switch options from site/xsite/ops_x4.py:STARTS')
    out.fact('threshold.share', SHARE_THRESHOLD, tier='S', source='xp/analyze_x4.py: sharp-window summary threshold', pct=True)
    out.fact('pre_logits.n', len(pre_logits), tier='P', source=hit_src)
    out.fact('pre_logits.shown_n', sum(h[0] in STARTS for h in pre_logits), tier='P', source=hit_src + '; figure-selectable starts only')
    out.fact('pre_logits.shown_total', shown_total, tier='P', source='paths_<start>.npz:share_unit; figure-selectable starts only')
    require(len(pre_logits) == 1, 'pre-logit exception description assumes a single hit')
    for key, value in zip(('start', 'row', 'site', 'refined'), pre_logits[0]):
        if key == 'refined':
            value = 'was refined' if value else 'was not refined'
        out.fact(f'pre_logits.exception.{key}', value, tier='P', source=hit_src)
    out.fact('logits.n', sum(logits_hits.values()), tier='P', source='paths_<start>.npz:share_unit, sites; total count above 25% at logits')
    for start in ('pixels', 'stem', 'b3'):
        out.fact(f'logits.{start}', logits_hits[start], tier='P', source=f'paths_{start}.npz:share_unit, sites; count above 25% at logits')
    out.fact('pre_logits.total', 2 * n_pairs * len(ALL_STARTS), tier='P',
             source='paths_<start>.npz:kind; number of pairs across all starts, counting repeats')
    out.fact('sharp.min', min(all_sigma_min), digits=2, tier='P',
             source='paths_<start>.npz:sigma_raw; minimum across all six starts, sites and pairs')
    out.fact('refine.max_median', max(refine_max), digits=4, tier='P',
             source='paths_<start>.npz:share1024_unit - share_unit[ref_rows]; largest per-site median absolute difference')

    check_src = BASE + 'checks.json'
    checks = json.loads(inputs.file(check_src).read_text())
    for start in ('pool', 'b9'):
        key = f'{start}_logits_exact_err'
        value = checks[key]
        out.fact(f'gain.affine.{start}', value, tier='S', source=f'{check_src}:{key}; max absolute log-gain error versus affine formula')
        out.facts[f'gain.affine.{start}']['text'] = f'{value:.2g}'

    gain_src = BASE + 'probe_gain.npz'
    viewer = inputs.manifest['ops']['r20s0-c10-knn15']['probe']['viewer']
    with np.load(inputs.file(gain_src), allow_pickle=False) as z:
        check_probe(z['probe_idx'], z['labels'], inputs.value(viewer + ':idx'), inputs.value(viewer + ':labels'))
        starts, target = z['starts'].tolist(), z['targets'].tolist().index('logits')
        raw, ang, rad, margin = z['gain_raw'][:, target], z['gain_ang'][:, target], z['rad_out'][:, target], z['margin']
        angular_pixels = ang[starts.index('pixels')].tolist()
        correlations = []
        for key, name, formula, values in (
            ('raw', 'Raw', 'gain_raw', raw),
            ('angular', 'Angular', 'gain_ang', ang),
            ('radius_removed', 'Angular, without radius factor', 'gain_ang + log(rad_out)', ang + np.log(rad)),
        ):
            rho = [float(spearmanr(row, margin).statistic) for row in values]
            correlations.append(dict(name=name, values=rho, **({'control': True} if key == 'raw' else {})))
            for start, value in zip(starts, rho):
                out.fact(f'gain.{start}.{key}', value, digits=2, tier='P',
                         source=f'{gain_src}:{formula}, margin; Spearman, {start} to logits')
        out.fact('gain.radius', float(spearmanr(np.log(rad[0]), margin).statistic), digits=2, tier='P',
                 source=gain_src + ':log(rad_out) vs margin; Spearman, pixels to logits')

    figure = dict(type='emb', schema=1, K=16, dims=2, norm='site')
    out.chart('emb_lead', dict(figure, switch=switches, kinds=['joint', 'site'], kind='joint',
                               axes='pooled:0', colour={'by': 'class'}, height=440))
    out.chart('emb_gain', dict(figure, panels=[dict(op='r20s0-c10-knn15')], kinds=['joint'], kind='joint',
                               site='logits', axes='viewer', height=290,
                               colour=dict(by='scalar', set='probe', label='Angular log gain (pixels to logits)',
                                           values=angular_pixels, fit=[2, 98])))
    out.chart('motion', dict(panels=[
        dict(title='Angular arc length in the fastest window', **chart_lines(plot_sites, runs, 'share', 'Arc length share',
             yrange=(0, .16), fmt='pct', refline={'y': 1/16, 'label': 'Uniform speed'})),
        dict(title='Raw sharpness index', **chart_lines(plot_sites, runs, 'sharp', 'Sharpness index',
             yrange=(.9, 1.02), fmt='.2f', refline={'y': 1, 'label': 'Constant raw speed'})),
    ]))
    out.chart('slope', chart_lines(plot_sites, runs, 'slope', 'Maximum slope magnitude', yrange=(0, 16), fmt='.1f'))
    out.chart('leaves', bars(ALL_STARTS, [dict(name='Left the endpoint class', values=leaves)],
                              'Share of same-class paths', 'Start site', fmt='pct', yrange=(0, .32)))
    out.chart('gains', bars(starts, correlations, 'Spearman rank correlation', 'Start site', fmt='.2f', yrange=(-.85, .1)))
    out.chart('localisation', bars(STARTS, [
        dict(name='Joint: largest across modes', values=[loc[s][0].max() for s in STARTS]),
        dict(name='Site-local: largest across sites and modes', values=[loc[s][1].max() for s in STARTS]),
    ], 'Squared norm on two images', 'Start site', fmt='pct', yrange=(0, 1)))
    local_cells = []
    for start in STARTS:
        joint, walk, chain = loc[start]
        local_cells.append([joint, *(walk[chain.index(s)] if s in chain else [None] * len(joint) for s in plot_sites)])
    out.chart('local_modes', heat(local_cells, ['Joint', *plot_sites], list(range(1, 1 + len(joint))),
                                  'Squared norm on two images', 'Site or joint', 'Mode', fmt='pct',
                                  slider={'label': 'Start site', 'values': list(STARTS)}))
    card = dict(answer=ANSWER, numbers=[f('grid.uniform') + ' uniform arc share',
                                       f('slope.b9.b9') + ' class-mass slope at the affine start',
                                       f('leaves.b8.n') + ' of ' + f('pairs.per_kind') + ' same-class b8 starts left the class on the sampled grid',
                                       f('gain.pixels.angular') + ' vs ' + f('gain.pixels.radius_removed') + ' gain–margin correlation'])
    if out.fig_path('thumb').exists():
        card['thumb'] = 'assets/fig/x_x4-paths_thumb.webp'
    out.chart('card', card)
    out.write()


if __name__ == '__main__':
    main()
