"""Build the ae-objective entry (ae/SPEC.md): the classifier's ResNet-20 trained to reconstruct instead of classify.

Every number on the page is computed here from staged files: neighbour overlap and the label readout are recomputed from
each network's saved 15-nearest-neighbour lists and required to match compare_ae/c10.npz (xp/stage_compare.py); the
thumbnail readout is computed from those lists and the probe images; PSNR is read from each autoencoder's info.json.
"""
from __future__ import annotations

import json

import numpy as np

from ops_ae import AUTOENCODERS, CLASSIFIER, op_id
from validate import Inputs, require, validate_operator, x1_align, x1_split
from xlib import ROOT, Out, PALETTE, lines

XP = 'crw-depth-xp-results'
COMPARE = f'{XP}/compare_ae/c10.npz'
PROBES = f'{XP}/probes.npz'
NETS = ('c10-s0', 'c10-s1', 'c10x-s0', 'c10x-s1', 'init-s0', 'init-s1', *AUTOENCODERS)
SITES = ['stem', *(f'b{i}' for i in range(1, 10)), 'pool', 'last']
HEAD = ['stem', 'b3', 'b6', 'b9', 'pool']           # prespecified headline sites (ae/SPEC.md revision 1)
K = 15
BLOCK = 8                                           # 32 / 8 = 4: 4x4 block-averaged RGB thumbnails (48 numbers)
VIEWER = ('#embedding?m=modes&lay=grid&nets=r20s0.c10x-s0.ae10-s0.ae64-s0&ref=r20s0&step=64000&v=own&c=3&K=16'
          '&norm=site&dm=frac&al=1&basis=pooled&ax=0.1.2&d3=0&color=class&depth=9')
# Pre-flight results (ae_check and the learning-rate pilot, ae/SPEC.md revisions 5 and 6). They were reported by the
# run, not staged as files, so they are transcriptions (tier S).
PREFLIGHT = {'init.tensors_10': 116, 'init.tensors_64': 114, 'batch.mse_from': 1.19, 'batch.mse_to': 0.010,
             'pilot.lr_0_1': 0.364, 'pilot.lr_0_03': 0.370, 'pilot.lr_0_01': 0.380}

# Pair groups (ae/SPEC.md revision 1). Different-seed pairs carry the headline; same-seed pairs are shown apart.
GROUPS = {
    'clf_seeds': [('c10x-s0', 'c10x-s1')],
    'zoo_seeds': [('c10-s0', 'c10-s1')],
    'ae10_seeds': [('ae10-s0', 'ae10-s1')],
    'ae64_seeds': [('ae64-s0', 'ae64-s1')],
    'ae10_clf_diff': [('ae10-s0', 'c10x-s1'), ('ae10-s1', 'c10x-s0')],
    'ae10_clf_same': [('ae10-s0', 'c10x-s0'), ('ae10-s1', 'c10x-s1')],
    'ae64_clf_diff': [('ae64-s0', 'c10x-s1'), ('ae64-s1', 'c10x-s0')],
    'ae64_clf_same': [('ae64-s0', 'c10x-s0'), ('ae64-s1', 'c10x-s1')],
    'ae10_ae64_same': [('ae10-s0', 'ae64-s0'), ('ae10-s1', 'ae64-s1')],
    'ae10_ae64_diff': [('ae10-s0', 'ae64-s1'), ('ae10-s1', 'ae64-s0')],
    'clf_init': [(a, b) for a in ('c10x-s0', 'c10x-s1') for b in ('init-s0', 'init-s1')],
    'ae10_init': [(a, b) for a in ('ae10-s0', 'ae10-s1') for b in ('init-s0', 'init-s1')],
    'init_seeds': [('init-s0', 'init-s1')],
}
KINDS = {'clf': ('c10x-s0', 'c10x-s1'), 'zoo': ('c10-s0', 'c10-s1'), 'ae10': ('ae10-s0', 'ae10-s1'),
         'ae64': ('ae64-s0', 'ae64-s1'), 'init': ('init-s0', 'init-s1')}


def token(key):
    return '{{fact:x.ae-objective.' + key + '}}'


def centred_knn(x, k=K):
    """k nearest rows by cosine after centring on the mean (self excluded), as xp/modal_xp.py:embed does per site."""
    x = x - x.mean(0)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    c = x @ x.T
    np.fill_diagonal(c, -np.inf)
    return np.argsort(-c, axis=1, kind='stable')[:, :k]


def thumbnails(images):
    """uint8 (N, 3, 32, 32) -> (N, 48): 8x8 block means of each channel, pixels in [0, 1]."""
    n = len(images)
    x = images.astype(np.float64).reshape(n, 3, 32 // BLOCK, BLOCK, 32 // BLOCK, BLOCK) / 255
    return x.mean((3, 5)).reshape(n, -1)


def overlap(a, b):
    """mean_i |a(i) ∩ b(i)| / k for neighbour lists a, b (..., N, k); works on any number of leading axes."""
    k = a.shape[-1]
    return (a[..., :, :, None] == b[..., :, None, :]).any(-1).sum(-1).mean(-1) / k


def psnr(recon, images):
    err = ((recon.astype(np.float64) - images.astype(np.float64)) / 255) ** 2
    return -10 * np.log10(err.mean())


def save_strip(path, images, scale=4, gap=6):
    """A row of images, each upscaled by pixel repetition, on a transparent gap."""
    from PIL import Image
    tiles = [np.repeat(np.repeat(im.transpose(1, 2, 0), scale, 0), scale, 1) for im in images]
    h, w = tiles[0].shape[:2]
    canvas = np.zeros((h, len(tiles) * w + (len(tiles) - 1) * gap, 4), np.uint8)
    for j, t in enumerate(tiles):
        canvas[:, j * (w + gap):j * (w + gap) + w, :3] = t
        canvas[:, j * (w + gap):j * (w + gap) + w, 3] = 255
    Image.fromarray(canvas, 'RGBA').save(path, lossless=True, method=6)
    return canvas.shape[1], canvas.shape[0]


def main():
    out, inputs = Out('ae-objective'), Inputs()
    for net in (CLASSIFIER, *AUTOENCODERS):
        inputs.bundle(op_id(net))                  # verifies every pinned file the entry reads
    with np.load(inputs.file(COMPARE), allow_pickle=False) as z:
        comp = {k: z[k] for k in ('nets', 'labels', 'perm', 'within', 'nn', 'nn_perm', 'nn_within', 'cka', 'readout',
                                  'knn15_trunk_cos', 'knn15_full_cos')}
    require(comp['nets'].tolist() == list(NETS), 'compare_ae network order differs from ae/SPEC.md')
    ix = {n: i for i, n in enumerate(NETS)}
    labels = comp['labels']
    n = len(labels)
    with np.load(inputs.file(PROBES), allow_pickle=False) as z:
        images, probe_y, probe_idx = z['c10_x'], z['c10_y'], z['c10_idx']
    require(np.array_equal(probe_y, labels), 'probe labels differ from the comparison labels')
    knn = np.stack([np.load(inputs.file(f'{XP}/embed/{net}/c10/knn.npy')).astype(np.int64) for net in NETS])
    require(knn.shape == (len(NETS), len(SITES), n, K), f'unexpected neighbour-list shape {knn.shape}')
    require(not (knn == np.arange(n)[None, None, :, None]).any(), 'a neighbour list contains its own image')

    # ---- same-site neighbour overlap, recomputed from the lists and checked against stage_compare
    diag = np.arange(len(SITES))
    nn = {}
    for a in NETS:
        for b in NETS:
            if ix[a] < ix[b]:
                nn[a, b] = nn[b, a] = overlap(knn[ix[a]], knn[ix[b]])
                err = np.max(np.abs(nn[a, b] - comp['nn'][ix[a], ix[b]][diag, diag]))
                require(err < 1e-6, f'{a} x {b}: neighbour overlap replay error {err:.3g}')
    readout = (labels[knn] == labels[None, None, :, None]).mean((2, 3))
    require(np.max(np.abs(readout - comp['readout'])) < 1e-9, 'label readout replay failed')

    def group(name, metric=None):
        """(mean, lo, hi) over the group's pairs at every site; metric (n, n, 12[, 12]) from the archive, else nn."""
        vals = []
        for a, b in GROUPS[name]:
            if metric is None:
                vals.append(nn[a, b])
            else:
                m = metric[ix[a], ix[b]]
                vals.append(m[diag, diag] if m.ndim == 2 else m)
        v = np.stack(vals)
        return v.mean(0), v.min(0), v.max(0)

    # ---- thumbnail readout (ae/SPEC.md revision 7)
    thumb = centred_knn(thumbnails(images))
    inv = np.argsort(comp['perm'])
    thumb_perm = inv[thumb[comp['perm']]]       # B's image p(i) plays the role of i, as for stage_compare's nn_perm
    thumb_read = overlap(knn, thumb[None, None])             # (nets, sites)
    thumb_read_perm = overlap(knn, thumb_perm[None, None])
    thumb_purity = (labels[thumb] == labels[:, None]).mean()

    # ---- facts
    def fact(key, value, source, *, tier='R', digits=2, **kw):
        if isinstance(value, str):
            digits = None
        out.fact(key, float(value) if isinstance(value, np.floating) else value, tier=tier, source=source,
                 digits=digits, **kw)

    def kind_mean(kind, values):
        return np.mean([values[ix[m]] for m in KINDS[kind]], axis=0)

    knn_src = 'embed/<net>/c10/knn.npy (12 sites x 2000 images x 15 neighbours)'
    fact('sample.probe', n, f'{PROBES}:c10_idx length', digits=None)
    fact('sample.k', K, 'xp/modal_xp.py:embed topk(15)', digits=None, tier='S')
    fact('sample.chance', K / (n - 1), '15 / (N - 1): expected overlap of two random neighbour lists', digits=4)
    for g in GROUPS:
        mean, lo, hi = group(g)
        pairs = ', '.join(f'{a}|{b}' for a, b in GROUPS[g])
        for l, site in enumerate(SITES):
            fact(f'nn.{g}.{site}', mean[l], f'{knn_src}: mean_i |N_A(i) & N_B(i)| / 15 at {site}, mean over {pairs}; '
                 f'= {COMPARE}:nn[A,B,l,l]')
            if len(GROUPS[g]) > 1 and site in HEAD + ['last']:
                fact(f'nn.{g}.{site}_lo', lo[l], f'min over {pairs} of the same')
                fact(f'nn.{g}.{site}_hi', hi[l], f'max over {pairs} of the same')
    for width in ('ae10', 'ae64'):
        ratio = group(f'{width}_clf_diff')[0] / group('clf_seeds')[0]
        for l, site in enumerate(SITES):
            fact(f'ratio.{width}.{site}', ratio[l], f'nn.{width}_clf_diff.{site} / nn.clf_seeds.{site}')
    same_vs_diff = np.abs(group('ae10_clf_same')[0] - group('ae10_clf_diff')[0])
    fact('nn.same_minus_diff.max', same_vs_diff[[SITES.index(s) for s in HEAD]].max(),
         'max over headline sites of |nn.ae10_clf_same - nn.ae10_clf_diff|', digits=4)
    fact('nn.same_minus_diff.all_max', same_vs_diff.max(),
         'max over all 12 sites of |nn.ae10_clf_same - nn.ae10_clf_diff|', digits=4)
    gap_row = group('clf_seeds')[0] - group('ae10_clf_diff')[0]
    for l, site in enumerate(SITES):
        fact(f'gap.ae10.{site}', gap_row[l], f'nn.clf_seeds.{site} - nn.ae10_clf_diff.{site}')
    for name, key in (('perm', 'nn_perm'), ('within', 'nn_within')):
        for g in ('ae10_clf_diff', 'clf_seeds'):
            mean = group(g, comp[key])[0]
            for site in ('stem', 'b9', 'pool'):
                fact(f'{name}.{g}.{site}', mean[SITES.index(site)], f'{COMPARE}:{key}, mean over {g} pairs at {site}',
                     tier='P', digits=3)
    for g in ('clf_seeds', 'ae10_seeds', 'ae64_seeds', 'ae10_clf_diff', 'ae10_clf_same', 'ae64_clf_diff', 'clf_init',
              'ae10_init', 'ae10_ae64_same', 'zoo_seeds'):
        for metric in ('cka', 'knn15_trunk_cos', 'knn15_full_cos'):
            mean = group(g, comp[metric])[0]
            for site in ('stem', 'b9'):
                l = SITES.index(site)
                fact(f'{metric.split("_")[0] if metric == "cka" else metric.replace("knn15_", "joint_")}.{g}.{site}',
                     mean[l], f'{COMPARE}:{metric}, mean over {g} pairs at {site}', tier='S')
    for kind in KINDS:
        lab, th = kind_mean(kind, readout), kind_mean(kind, thumb_read)
        for l, site in enumerate(SITES):
            fact(f'readout.{kind}.{site}', lab[l], f'{knn_src} and {PROBES}:c10_y; share of the 15 neighbours with the '
                 f'image\'s label at {site}, mean over {"/".join(KINDS[kind])}')
            fact(f'thumb.{kind}.{site}', th[l], f'{knn_src} and {PROBES}:c10_x; share of the 15 neighbours also among '
                 f'the 15 nearest 4x4 thumbnails at {site}, mean over {"/".join(KINDS[kind])}')
    ae = [ix[m] for m in AUTOENCODERS]
    fact('readout.ae.min', readout[ae].min(), 'min over the four autoencoders and all 12 sites of the label readout')
    fact('readout.ae.max', readout[ae].max(), 'max over the four autoencoders and all 12 sites of the label readout')
    fact('readout.ae.head_min', readout[ae][:, [SITES.index(s) for s in HEAD]].min(),
         'min over the four autoencoders and the headline sites of the label readout')
    fact('readout.ae.head_max', readout[ae][:, [SITES.index(s) for s in HEAD]].max(),
         'max over the four autoencoders and the headline sites of the label readout')
    counts = np.bincount(labels)
    fact('readout.chance', ((counts[labels] - 1) / (n - 1)).mean(), f'{PROBES}:c10_y; mean over images of '
         '(images of its class - 1) / (N - 1): the label readout of 15 random neighbours')
    fact('thumb.purity', thumb_purity, f'{PROBES}:c10_x, c10_y; share of each image\'s 15 nearest thumbnails with its label')
    fact('thumb.perm.max', thumb_read_perm.max(), f'{knn_src}; thumbnail readout with thumbnail identities permuted by '
         f'{COMPARE}:perm, max over the ten networks and 12 sites', digits=4)
    for net in AUTOENCODERS + ('c10x-s0', 'c10x-s1'):
        for site in ('stem', 'b3', 'b9', 'last'):
            fact(f'thumb.net.{net.replace("-", "_")}.{site}', thumb_read[ix[net], SITES.index(site)],
                 f'{knn_src}; thumbnail readout of {net} at {site}')

    # ---- training records
    infos = {}
    for net in AUTOENCODERS + ('c10x-s0', 'c10x-s1'):
        key = f'{XP}/nets/{net}/info.json'
        infos[net] = json.loads(inputs.file(key).read_text())
        require(infos[net]['name'] == net and infos[net]['log'][-1]['step'] == infos[net]['steps'] == 64000,
                f'{net}: inconsistent training record')
    for net in ('c10x-s0', 'c10x-s1'):
        fact(f'acc.{net.replace("-", "_")}', infos[net]['log'][-1]['test_acc'],
             f'{XP}/nets/{net}/info.json:log[-1].test_acc', tier='S', digits=1, pct=True)
    recon = {}
    for net in AUTOENCODERS:
        info, name = infos[net], net.replace('-', '_')
        require(info['objective'] == 'ae' and info['lr'] == 0.1, f'{net}: not trained at the classifier rate')
        src = f'{XP}/nets/{net}/info.json'
        fact(f'psnr.{name}', info['log'][-1]['test_psnr'], src + ':log[-1].test_psnr (10,000 test images)', tier='S',
             digits=1)
        fact(f'psnr.train.{name}', info['log'][-1]['train_psnr'], src + ':log[-1].train_psnr', tier='S', digits=1)
        with np.load(inputs.file(f'{XP}/nets/{net}/recon.npz'), allow_pickle=False) as z:
            require(np.array_equal(z['idx'], probe_idx), f'{net}: reconstruction order differs from the probe')
            recon[net] = z['recon']
        fact(f'psnr.probe.{name}', psnr(recon[net], images), f'{XP}/nets/{net}/recon.npz:recon against {PROBES}:c10_x '
             '(uint8, pixels in [0, 1])', digits=1)
        fact(f'code_std_min.{name}', min(info['code_std']), src + ':code_std, min over code coordinates', tier='S')
        fact(f'batch_dep.{name}', info['eval_batch_dependence'], src + ':eval_batch_dependence', tier='S', digits=4)
    for width in (10, 64):
        a, b = (infos[f'ae{width}-s{s}'] for s in (0, 1))
        require(a['pca_psnr'] == b['pca_psnr'] and a['code'] == width, f'ae{width}: PCA baseline differs between seeds')
        fact(f'psnr.pca{width}', a['pca_psnr'], f'{XP}/nets/ae{width}-s0/info.json:pca_psnr (rank {width}, fitted on '
             'the 50,000 training images)', tier='S', digits=1)
        fact(f'params.enc{width}', a['params_encoder'], f'{XP}/nets/ae{width}-s0/info.json:params_encoder', tier='S',
             digits=None)
        fact(f'params.dec{width}', a['params_decoder'], f'{XP}/nets/ae{width}-s0/info.json:params_decoder', tier='S',
             digits=None)
    means = {infos[m]['mean_image_psnr'] for m in AUTOENCODERS}
    require(len(means) == 1, 'mean-image PSNR differs between autoencoders')
    fact('psnr.mean_image', means.pop(), f'{XP}/nets/ae10-s0/info.json:mean_image_psnr', tier='S', digits=1)
    fact('train.steps', infos['ae10-s0']['steps'], f'{XP}/nets/ae10-s0/info.json:steps', tier='S', digits=None)
    fact('train.lr', infos['ae10-s0']['lr'], f'{XP}/nets/ae10-s0/info.json:lr', tier='S', digits=1)
    for key, value in PREFLIGHT.items():
        fact(f'preflight.{key}', value, 'pre-flight ae_check / learning-rate pilot output reported by the run (not '
             'staged); procedure in ae/SPEC.md revisions 5 and 6', tier='S',
             digits=None if isinstance(value, int) else (2 if key == 'batch.mse_from' else 3))

    # ---- charts. x: 11 shared sites, a gap, then the last site (logits or code), which is not comparable.
    x = list(range(len(SITES) + 1))
    ticks = SITES[:-1] + ['', 'last']

    def gap(v):
        v = list(np.asarray(v, dtype=float))
        return v[:-1] + [None] + v[-1:]

    def band(lo, hi):
        # Bands stop at pool: a band polygon would otherwise bridge the gap to the last site.
        return [gap(lo)[:-1] + [None], gap(hi)[:-1] + [None]]

    sets, mute, accent = PALETTE['sets'], PALETTE['mute'], PALETTE['accent']
    colour = dict(clf=sets[0], ae10=sets[2], ae64=sets[6], cross=accent, init=mute)

    def series(g, name, color, **kw):
        mean, lo, hi = group(g)
        banded = kw.pop('banded', True)
        s = dict(name=name, y=gap(mean), color=color, dots=3, **kw)
        if len(GROUPS[g]) > 1 and banded:
            s['band'] = band(lo, hi)
        return s

    def chart(series_list, ylabel, yrange=(0, 1), **kw):
        return lines(x, series_list, 'Site', ylabel, yrange=yrange, xticklabels=ticks, xname='', **kw)

    out.chart('overlap', chart([
        series('ae10_clf_diff', 'Autoencoder × classifier, other seed (2 pairs)', colour['cross'], width=3),
        series('clf_seeds', 'Classifier seed 0 × seed 1', colour['clf']),
        series('ae10_seeds', 'Autoencoder seed 0 × seed 1', colour['ae10']),
        series('clf_init', 'Classifier × untrained (4 pairs)', colour['init'], dash='dot'),
    ], 'Share of the same 15 neighbours'))
    out.chart('same_seed', chart([
        series('ae10_clf_diff', 'Autoencoder × classifier, other seed', colour['cross'], width=3),
        series('ae10_clf_same', 'Autoencoder × classifier, same seed', colour['clf'], dash=True),
    ], 'Share of the same 15 neighbours', yrange=(0, .4)))
    out.chart('width', chart([
        series('ae64_seeds', '64-number autoencoders, seed 0 × seed 1', colour['ae64']),
        series('ae10_ae64_same', '10-number × 64-number code, same seed', colour['ae10'], dash=True),
        series('clf_seeds', 'Classifier seed 0 × seed 1', colour['clf']),
        series('ae64_clf_diff', '64-number autoencoder × classifier, other seed', colour['cross'], width=3),
    ], 'Share of the same 15 neighbours'))
    out.chart('untrained', chart([
        series('ae10_init', 'Autoencoder × untrained (4 pairs)', colour['ae10'], dash='dot'),
        series('clf_init', 'Classifier × untrained (4 pairs)', colour['clf'], dash='dot'),
        series('init_seeds', 'Untrained seed 0 × seed 1', colour['init'], dash='dot'),
        series('ae10_clf_diff', 'Autoencoder × classifier, other seed', colour['cross'], width=3),
    ], 'Share of the same 15 neighbours'))

    def kind_series(values, kind, name, **kw):
        v = np.stack([values[ix[m]] for m in KINDS[kind]])
        return dict(name=name, y=gap(v.mean(0)), band=band(v.min(0), v.max(0)), color=colour[kind], dots=3, **kw)

    def readout_series(values):
        return [kind_series(values, 'clf', 'Classifiers (2 seeds)'), kind_series(values, 'ae10', 'Autoencoders, 10-number code'),
                kind_series(values, 'ae64', 'Autoencoders, 64-number code'), kind_series(values, 'init', 'Untrained', dash='dot')]

    out.chart('readouts', {'panels': [
        chart(readout_series(readout), 'Share of neighbours with the same label',
              title='Label readout: neighbours of the same class', refline={'y': float(thumb_purity)}),
        chart(readout_series(thumb_read), 'Share of neighbours that are also nearest thumbnails',
              title='Thumbnail readout: neighbours that look alike at 4 × 4 pixels'),
    ]})

    def metric_chart(metric, ylabel):
        def s(g, name, color, **kw):
            mean, lo, hi = group(g, comp[metric])
            out_ = dict(name=name, y=gap(mean), color=color, dots=3, **kw)
            if len(GROUPS[g]) > 1:
                out_['band'] = band(lo, hi)
            return out_
        return chart([s('ae10_seeds', 'Autoencoder seed 0 × seed 1', colour['ae10']),
                      s('clf_seeds', 'Classifier seed 0 × seed 1', colour['clf']),
                      s('ae10_clf_diff', 'Autoencoder × classifier, other seed', colour['cross'], width=3),
                      s('clf_init', 'Classifier × untrained', colour['init'], dash='dot')], ylabel,
                     yrange=(-.1 if metric != 'cka' else 0, 1))

    out.chart('cka', metric_chart('cka', 'Linear CKA'))
    out.chart('joint', metric_chart('knn15_full_cos', 'Held-out cosine after one rotation'))

    # ---- aligned view (ae/SPEC.md revision 4): each network's full-chain joint coordinates turned by one rotation onto
    # the classifier's, fitted on half the probe; the rotation replays compare_ae's knn15_full_cos to 1e-6.
    train, test = x1_split(labels, comp['perm'], comp['within'])
    ref = validate_operator(inputs, op_id(CLASSIFIER))['joint']
    aligned = []
    for net, label in ((CLASSIFIER, 'Classifier · seed 0 (reference axes)'),
                       ('ae10-s0', 'Autoencoder, 10-number code · seed 0'),
                       ('ae10-s1', 'Autoencoder, 10-number code · seed 1'),
                       ('ae64-s0', 'Autoencoder, 64-number code · seed 0'),
                       ('ae64-s1', 'Autoencoder, 64-number code · seed 1'), ('init-s0', 'Untrained · seed 0')):
        expected = comp['knn15_full_cos'][ix[CLASSIFIER], ix[net]]
        if net == CLASSIFIER:
            R = np.eye(ref.shape[-1])
        else:
            R, _ = x1_align(ref, validate_operator(inputs, op_id(net))['joint'], train, test, expected)
        aligned.append(dict(op=op_id(net), label=label, R=R.tolist(),
                            stat=dict(label='held-out cosine', values=[float(v) for v in expected])))
    out.chart('emb_joint', dict(type='emb', schema=1, layout='grid', panels=aligned[:2], panelMenu=aligned[1:],
                                kinds=['joint'], kind='joint', site='b9', K=16, dims=2, axes='pooled:0', norm='site',
                                colour={'by': 'class'}, height=340))

    # ---- embedding figure: site-local coordinates only (no rotation between networks is fitted)
    def panel(net, label):
        return dict(op=op_id(net), label=label, R=None,
                    stat=dict(label='label readout', values=[float(v) for v in readout[ix[net]]]))

    menu = [panel('ae10-s0', 'Autoencoder, 10-number code · seed 0'), panel('ae10-s1', 'Autoencoder, 10-number code · seed 1'),
            panel('ae64-s0', 'Autoencoder, 64-number code · seed 0'), panel('ae64-s1', 'Autoencoder, 64-number code · seed 1'),
            panel('init-s0', 'Untrained · seed 0')]
    out.chart('emb', dict(type='emb', schema=1, layout='grid', panels=[panel('c10x-s0', 'Classifier · seed 0'), menu[0]],
                          panelMenu=menu, kinds=['site'], kind='site', site='b9', K=16, dims=2, axes='pooled',
                          norm='site', colour={'by': 'class'}, height=380, viewerLink=VIEWER))

    # ---- reconstructions of fixed probe images: the first probe image of each class, in class order
    names = json.loads((ROOT / 'data/r20s0b/meta.json').read_text())['classes']
    first = [int(np.flatnonzero(labels == c)[0]) for c in range(10)]
    fact('recon.positions', ', '.join(str(p) for p in first), 'first probe position of each class 0..9', tier='R')
    fact('recon.classes', ', '.join(names), 'data/r20s0b/meta.json:classes', tier='S')
    sizes = {}
    for name, stack in (('orig', images), ('ae10', recon['ae10-s0']), ('ae64', recon['ae64-s0'])):
        sizes[name] = save_strip(out.fig_path(f'recon_{name}'), stack[first])
    require(len(set(sizes.values())) == 1, 'reconstruction strips differ in size')

    # ---- card
    card = dict(answer='Reconstruction and classification already disagree at the stem. By b9, '
                       f'{token("readout.clf.b9")} of a classifier\'s neighbours share the image\'s class against '
                       f'{token("readout.ae10.b9")} for an autoencoder, near the {token("thumb.purity")} of images that '
                       'merely look alike as 4 × 4 thumbnails.',
                numbers=[f'{token("nn.ae10_clf_diff.stem")} vs {token("nn.clf_seeds.stem")} shared neighbours at the '
                         'stem (autoencoder × classifier vs classifier seeds)',
                         f'{token("nn.ae10_clf_diff.b9")} vs {token("nn.clf_seeds.b9")} at b9',
                         f'{token("readout.ae10.b9")} vs {token("readout.clf.b9")} label readout at b9',
                         f'{token("psnr.ae10_s0")} dB test PSNR, 10-number code ({token("psnr.pca10")} dB for rank-10 PCA)'])
    thumb_file = out.fig_path('thumb')
    if thumb_file.exists():
        card['thumb'] = f'assets/fig/{thumb_file.name}'
    out.chart('card', card)
    out.write()
    print('strip size', sizes['orig'])
    for g in ('clf_seeds', 'ae10_seeds', 'ae10_clf_diff', 'ae10_clf_same', 'clf_init', 'ae10_init'):
        print(f'{g:16s}', ' '.join(f'{s}:{group(g)[0][SITES.index(s)]:.3f}' for s in HEAD + ['last']))
    print('readout', {k: np.round(kind_mean(k, readout)[[0, 9, 11]], 3).tolist() for k in KINDS})
    print('thumb', {k: np.round(kind_mean(k, thumb_read)[[0, 3, 6, 9, 10, 11]], 3).tolist() for k in KINDS},
          'purity', round(thumb_purity, 3), 'perm max', round(thumb_read_perm.max(), 4))


if __name__ == '__main__':
    main()
