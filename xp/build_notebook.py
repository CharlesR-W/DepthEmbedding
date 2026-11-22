"""Build the experiment notebook for xp/ (X1, X3; X4 when its summary exists) -> xp/pages/notebook.html

    python3 xp/build_notebook.py            (standard library only; reads xp/out/x1_summary.json, x3_summary.json)

Everything on the page is read from the two summary files written by xp/analyze_x1.py and xp/analyze_x3.py.
Networks, training sets and probes are taken from the summaries, so a summary regenerated with more networks
(e.g. the ImageNet-32 pair) is picked up by a rebuild without code changes.

Adding a tab: write a function `tab_<name>(nb, data)` like `tab_x3`, add a row to EXPERIMENTS with the summary
file it needs; the experiment map on the Overview turns the row into a tab link once the file exists.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notebook import Notebook, Radio, Select, Slider, arch_svg, clean, table_html  # noqa: E402

XP = Path(__file__).resolve().parent
OUT = XP / 'out'
PAGE = XP / 'pages' / 'notebook.html'

SITES = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
TRUNK = SITES[:10]

# One colour per training set, used on every tab (validated with the dataviz skill's checker, all pairs, light
# surface; the two reference sets are neutral greys on purpose).
GROUPS = {
    'c10': ('CIFAR-10', '#3366bb'),
    'anim': ('CIFAR-10 animals', '#44b3cc'),
    'veh': ('CIFAR-10 vehicles', '#aa3377'),
    'c100': ('CIFAR-100', '#228833'),
    'c100c': ('CIFAR-100, 20 coarse labels', '#7cbf5a'),
    'in32': ('ImageNet-32', '#c9a800'),
    'svhn': ('SVHN', '#ee6677'),
    'rand': ('shuffled labels', '#57534c'),
    'init': ('untrained', '#a8a29a'),
}
PROBES = {'c10': 'CIFAR-10', 'c100': 'CIFAR-100', 'svhn': 'SVHN', 'in32': 'ImageNet-32'}
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# X1 measures: key in the summary -> (label, short label, trunk-only, chance level or None)
MEASURES = {
    'joint_knn15_trunk': ('Joint agreement: trunk, k-NN kernel (primary)', 'joint agreement', True, 0.0),
    'nn': ('Neighbour overlap (15 nearest by cosine; kernel-free)', 'neighbour overlap', False, 15 / 1999),
    'cka': ('Linear CKA (kernel-free)', 'CKA', False, None),
    'walk8_knn15': ('Site-walk subspace overlap, r = 8, k-NN kernel', 'walk overlap', False, 8 / 2000),
    'joint_ea100_trunk': ('Joint agreement: trunk, smooth kernel (check)', 'joint, smooth kernel', True, 0.0),
    'joint_knn15_full': ('Joint agreement: full chain incl. pool and logits, k-NN (check)', 'joint, full chain', False, 0.0),
    'joint_ea100_full': ('Joint agreement: full chain, smooth kernel (check)', 'joint, full, smooth', False, 0.0),
    'walk8_ea100': ('Site-walk subspace overlap, r = 8, smooth kernel (check)', 'walk overlap, smooth', False, 8 / 2000),
}

EXPERIMENTS = [  # key, tab label, question, readout, summary file
    ('x1', 'X1 Different training sets',
     'How much of the seed-to-seed agreement of the embedding survives a change of training set?',
     'Joint agreement per site (trunk, k-NN kernel); neighbour overlap and CKA as kernel-free references', 'x1_summary.json'),
    ('x3', 'X3 Adversarial images',
     'Where in depth does an adversarial perturbation take hold, and where does the attacked image land?',
     'Site-local displacement and class mass; switch site; arrival and support', 'x3_summary.json'),
    ('x4', 'X4 Interpolation and log gain',
     'Along straight paths between real activations, where does the rest of the network stretch the path, and where does class membership change?',
     'Window share of arc length; class mass along the path; log gain; paths in suffix embeddings', 'x4_summary.json'),
]

f2 = lambda v: f'{v:.2f}'  # noqa: E731
pc = lambda v: f'{100 * v:.0f}%'  # noqa: E731


def th(v):
    """A share as an ordinal percentile: 0.02 -> '2nd'."""
    n = round(100 * v)
    return f'{n}' + ('th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th'))


def swatch(g):
    return f'<span class="sw" style="background:{GROUPS[g][1]}"></span>'


def glabel(g):
    return GROUPS.get(g, (g, '#888'))[0]


# ------------------------------------------------------------------------------------------------ JS helpers --

JS = r"""
const SITES = %SITES%;
const GROUPS = %GROUPS%;
const PROBENAME = %PROBES%;
const CLASSES = %CLASSES%;
const CLASSCOL = ['#4e79a7','#f28e2b','#e15759','#76b7b2','#59a14f','#edc948','#b07aa1','#ff9da7','#9c755f','#bab0ac'];
const GC = g => (GROUPS[g] || ['?', '#888'])[1];
const GL = g => (GROUPS[g] || [g, '#888'])[0];
const ax = i => i === 0 ? '' : String(i + 1);           // axis suffix for panel i
// 2 x 2 facets
function dom4(i) { const col = i % 2, row = Math.floor(i / 2);
  return {x: col ? [0.54, 1] : [0, 0.46], y: row ? [0, 0.42] : [0.58, 1]}; }
function siteAxis(i, extra) { return merge(AXIS, merge({tickvals: SITES.map((_, k) => k), ticktext: SITES, range: [-0.4, 11.4],
  tickfont: {size: 10.5}, anchor: 'y' + ax(i)}, extra || {})); }
function stageLines(i, upto) { const xs = [3.5, 6.5, 9.5].filter(x => x < (upto == null ? 12 : upto));
  return xs.map(x => ({type: 'line', xref: 'x' + ax(i), yref: 'y' + ax(i) + ' domain', x0: x, x1: x, y0: 0, y1: 1,
    line: {color: '#d9d3c9', width: 1, dash: 'dot'}, layer: 'below'})); }
function headShade(i) { return {type: 'rect', xref: 'x' + ax(i), yref: 'y' + ax(i) + ' domain', x0: 9.5, x1: 11.45, y0: 0, y1: 1,
  fillcolor: '#f3f0ea', line: {width: 0}, layer: 'below'}; }
function panelTitle(i, text) { return {text: '<b>' + text + '</b>', xref: 'x' + ax(i) + ' domain', yref: 'y' + ax(i) + ' domain',
  x: 0, y: 1.0, xanchor: 'left', yanchor: 'bottom', showarrow: false, font: {size: 12.5}}; }
function cellKey(order, a, b) { return order.indexOf(a) <= order.indexOf(b) ? a + '|' + b : b + '|' + a; }
function legendOnly(name, color, dash, width, mode, symbol) { return {x: [null], y: [null], name, mode: mode || 'lines',
  line: {color, dash: dash || 'solid', width: width || 2}, marker: {color, symbol: symbol || 'circle', size: 7}, hoverinfo: 'skip'}; }
const BLUES = [[0, '#f7fbff'], [0.25, '#c6dbef'], [0.5, '#6baed6'], [0.75, '#2171b5'], [1, '#08306b']];
"""


# ------------------------------------------------------------------------------------------------- overview --

def overview_found(x1, x3, x4):
    """Three or four sentences per experiment for the top of the Overview; every number is read from a summary."""
    out = []
    S = SITES.index
    if x1:
        J = 'joint_knn15_trunk'
        g = lambda a, b, p='c10': cell(x1, p, J, a, b)  # noqa: E731
        cc, c100, ci, c1, sv, ini = g('c10', 'c10'), g('c10', 'c100'), g('c100', 'in32'), g('c100', 'c100'), g('c10', 'svhn'), g('c10', 'init')
        it = []
        if cc and c100:
            it.append(f'Natural-image training pipelines give high joint agreement on the CIFAR-10 probe, below the matching seed pairs '
                      f'(CIFAR-10 against CIFAR-100: {f2(c100[S("b6")])} at b6 and {f2(c100[S("b9")])} at b9; CIFAR-10 seed pairs, mean of '
                      f'three: {f2(cc[S("b6")])} and {f2(cc[S("b9")])}).')
        if ci and c1:
            it.append(f'The closest cross pair is CIFAR-100 against ImageNet-32 ({f2(ci[0])} at the stem, {f2(ci[S("b9")])} at b9, against '
                      f'{f2(c1[0])} and {f2(c1[S("b9")])} for the CIFAR-100 seeds); the ranking of pipelines changes with probe and depth, and '
                      'the design compares whole pipelines, so it does not isolate label count or image distribution.')
        fc, ff, cc2 = g('c100', 'c100c'), g('c100', 'c100'), g('c100c', 'c100c')
        if fc and ff and cc2:
            it.append(f'Label granularity on fixed images does not move the embedding: CIFAR-100 trained with 20 coarse labels against 100 '
                      f'fine labels agrees at {f2(fc[0])} (stem) and {f2(fc[S("b9")])} (b9), within the seed-pair spread ({f2(ff[S("b9")])} '
                      f'fine, {f2(cc2[S("b9")])} coarse at b9).')
        if sv and ini:
            it.append(f'CIFAR-10 and SVHN pipelines differ at every depth, the stem included ({f2(sv[0])} at the stem, {f2(sv[S("b9")])} at '
                      f'b9; a CIFAR-10 network with the untrained one: {f2(ini[0])}, {f2(ini[S("b9")])}).')
        ss, ss_c = cell(x1, 'svhn', J, 'svhn', 'svhn'), cell(x1, 'c10', J, 'svhn', 'svhn')
        if ss and ss_c:
            it.append(f'Late agreement between the SVHN seeds drops on a probe from the other distribution ({f2(ss[S("b9")])} on SVHN, '
                      f'{f2(ss_c[S("b9")])} on CIFAR-10 at b9); this does not hold for every pipeline.')
        out.append(('X1 Different training pipelines', it))
    if x3:
        A = 'c10-s0'
        at, cm = x3['attacks'], x3['class_mass'][A]
        u, n = at['u8'][A]['disp_site_y'], at['n8'][A]['disp_site_y']
        ac_u, ac_n = at['u8'][A]['actcos'], at['n8'][A]['actcos']
        h = cm['u8_success']['switch']
        BINS = SITES + ['none']
        c_, med = 0, None
        for k, v in enumerate(h):
            c_ += v
            if med is None and c_ >= sum(h) / 2:
                med = BINS[k]
        mode = BINS[max(range(13), key=lambda k: h[k])]
        tr = at['u8']['transfer']
        out.append(('X3 Adversarial images', [
            f'In site-local coordinates an ε 8 PGD attack moves an image about as far as random sign noise of the same L∞ size up to b6 '
            f'({f2(u[S("b6")])} against {f2(n[S("b6")])} same-class units), then much further ({f2(u[S("b9")])} against {f2(n[S("b9")])} at b9); '
            f'raw activations already differ at b6 (cosine to clean {f2(ac_u[S("b6")])} against {f2(ac_n[S("b6")])}).',
            f'The per-image class switch is spread across depth (untargeted ε 8: median {med}, mode {mode}), later for weaker attacks and '
            'latest for natural errors.',
            f'Successful attacks end at typical distances from the destination class with {pc(cm["u8_success"]["dst"][11])} of their '
            f'kernel-row weight on it at the logits; natural errors end between classes ({pc(cm["natural"]["dst"][11])} on the destination).',
            f'The attack flips the other seed on {pc(tr["any_flip"])} of images; there, flipped images separate from unflipped ones before b9.']))
    if x4:
        P = x4['per_start']
        st = x4['starts']
        sig = [v for l in st for k in ('same', 'diff') for v in P[l]['stats']['sigma_raw'][k][1] if v is not None]
        it = [f'Straight paths between real activations show little localised motion through the rest of the network (sharpness index '
              f'{min(sig):.2f}–{max(sig):.2f} in site medians, 1 = smooth).']
        if 'b9' in P:
            it.append(f'Class membership read through the probe changes much more sharply from b9 on, but the affine b9 control shows the same '
                      f'(slope {P["b9"]["embedding"]["mass_slope"][1][0]:.1f}), so this is a property of the kernel readout and the '
                      'class-clustered cloud, not a barrier in the network.')
        leave = {l: 1 - P[l]['switches']['same'][0] / sum(P[l]['switches']['same']) for l in st}
        it.append('Sampled same-class chords stop leaving the class by b8 (leaving at the output: '
                  + ', '.join(f'{pc(leave[l])} from {l}' for l in st if l in ('pixels', 'b6', 'b8')) + ').')
        pg = x4.get('probe_gain')
        if pg and pg.get('rho_ang_radius_removed'):
            it.append(f'Angular log gain is anti-correlated with output margin over the probe ({f2(pg["rho"]["b3>logits_ang"])} from b3), but '
                      f'mostly through the output normalisation ({f2(pg["rho_ang_radius_removed"]["b3"])} with it removed).')
        out.append(('X4 Interpolation and log gain', it))
    return out


def tab_overview(nb, x1, x3, x4, have):
    t = nb.tab('overview', 'Overview')
    nets_c10 = x1['measures'][x1['probes'][0]]['nets'] if x1 else []
    t.prose(
        '<p class="lede">Three overnight experiments on the through-depth embedding of a fixed set of probe images. '
        'All of them use ResNet-20 at 32 px and the repo\'s joint operator. <b>This is exploratory work</b>: one '
        'architecture, two seeds per training set and one probe per dataset, so differences of a few hundredths '
        'between curves should not be read as effects.</p>')
    found = overview_found(x1, x3, x4)
    if found:
        t.takeaway(''.join(f'<h4>{h}</h4><ul>' + ''.join(f'<li>{x}</li>' for x in items) + '</ul>' for h, items in found),
                   label='What was found (numbers from the summaries; details on each tab)')
    t.box('<div class="label oq">Open questions</div><ul>'
          '<li>Does the sharp b9 transition of class membership in X4 survive a bandwidth sweep, or a kernel with matched final perplexity?</li>'
          '<li>Does the late association between peak speed and the output switch in X4 hold on fresh pairs and after accounting for the '
          'centred radius?</li>'
          '<li>X1 cannot separate image distribution, normalisation and augmentation; the coarse-label control addresses only label '
          'granularity, at one pair of granularities (100 against 20) on one image set.</li></ul>')
    t.prose(
        '<h2>The questions</h2>'
        '<p><b>X1.</b> Train the same architecture on different training sets and push the same probe images through '
        'each network. How much of the agreement between two seeds trained on the same data survives a change of '
        'training set, and at which depth is it lost? <b>X3.</b> Follow adversarial images through depth: at which site '
        'does an attack on one network take hold, and does the attacked image end up where real members of the '
        'destination class sit? <b>X4.</b> Walk the straight line between two real images\' activations at some site and run '
        'it through the rest of the network: where downstream does the path stretch, where does class membership change, and '
        'how much does the network stretch nearby directions (log gain)?</p>')

    t.h('The embedding, in one paragraph')
    t.prose(
        '<p>A <b>probe</b> is a fixed set of 2,000 images. At each <b>site</b> (a tap in the network: stem, the nine '
        'residual blocks b1 to b9, the average pool and the logits) the probe\'s activations are flattened, centred '
        'and normalised, and a <b>walk</b> $M_l$ joins each image to its nearest neighbours by cosine. Two kernels '
        'build the walk: <code>knn15</code>, the 15-nearest-neighbour rule behind the public viewer, and '
        '<code>ea100</code>, a smooth Gaussian kernel with per-point width set to perplexity 100. One operator '
        'over all (image, site) pairs charges for roughness within each site and for change between neighbouring '
        'sites:</p>')
    t.equation(r'E_t(f) \;=\; \sum_{l} \frac{f_l^\top (I - M_l)\, f_l}{c_l} \;+\; t \sum_{l} \lVert f_{l+1} - f_l \rVert^2',
               where=r'where $f_l$ is the slice of a function at site $l$, $c_l$ is site $l$’s mean of its first eight '
               r'nontrivial values of $1-\lambda$ (relative site units) and $t = t^*$ is the balance coupling. Its 16 '
               r'lowest nontrivial modes give every image a row of 16 numbers at every site, in one shared space: '
               r'the image’s <b>joint coordinates</b> $z(l)$.')
    t.prose('<p>A new image (an attacked one, say) gets coordinates without refitting: its kernel row against the probe '
            'at each site is pushed through the fitted operator (frozen-probe Nyström extension). A <b>site-local</b> '
            'coordinate uses only that site\'s own walk modes, so it depends on nothing downstream.</p>')

    t.h('What is measured')
    t.cols(
        '<h4>X1: joint agreement (primary)</h4>'
        r'<div class="eq">$$ a_l(A,B) = \frac{\langle z_A(l),\; z_B(l)\,Q\rangle}{\lVert z_A(l)\rVert\,\lVert z_B(l)\,Q\rVert} $$</div>'
        r'<div class="where">where $z_A(l)$ holds the held-out probe images’ coordinates at site $l$ in network $A$’s 16 '
        r'lowest trunk modes (sites stem to b9, $k$-NN kernel), and $Q$ is one $16\times16$ rotation shared by all sites, '
        r'fitted by Procrustes on the other half of the probe. 1 = the same embedding up to that rotation; 0 = unrelated.</div>',
        '<h4>X1: kernel-free references</h4>'
        r'<div class="eq">$$ \mathrm{nn}_l = \frac{1}{N}\sum_i \frac{\lvert \mathcal N^A_l(i)\cap \mathcal N^B_l(i)\rvert}{15} $$</div>'
        r'<div class="where">where $\mathcal N_l(i)$ are image $i$’s 15 nearest probe images by cosine at site $l$ '
        r'(chance $15/1999 \approx 0.008$). Linear CKA of the two centred cosine Grams is shown beside it, and the overlap '
        r'of the first 8 nontrivial eigenvectors of each site’s own walk, $\lVert U_A^\top U_B\rVert_F^2/8$.</div>',
    )
    t.cols(
        '<h4>X3: site-local displacement</h4>'
        r'<div class="eq">$$ D_l = \frac{\lVert y_l(x_{\mathrm{adv}}) - y_l(x)\rVert}{\operatorname{median}_{\,i\ne j,\ \mathrm{same\ class}} \lVert y_l(x_i) - y_l(x_j)\rVert} $$</div>'
        r'<div class="where">where $y_l$ is the site-local embedding at site $l$ (smooth kernel). 1 = as far as two clean '
        r'images of one class are from each other.</div>',
        '<h4>X3: class mass and switch site</h4>'
        r'<div class="eq">$$ m_l(x, c) = \sum_{j:\ \mathrm{label}(j) = c} p_l(x, j) $$</div>'
        r'<div class="where">where $p_l(x,\cdot)$ is the image’s kernel row over the probe at site $l$. The <b>switch '
        r'site</b> is the first site from which the destination class (the network’s final prediction) outweighs the '
        r'source class at every later site.</div>',
    )

    t.h('Experiment map')
    rows = []
    for key, label, q, readout, f in EXPERIMENTS:
        if key in have:
            st = '<span class="status done">measured</span>'
            if key == 'x1':
                st += f' <span class="dim">{len(nets_c10)} networks</span>'
            lab = f'<a href="#tab={key}">{label}</a>'
        else:
            st = '<span class="status open">data not ready</span>'
            lab = label
        rows.append([lab, q, readout, st])
    t.table(['Tab', 'Question', 'Readout', 'Status'], rows)

    if x1:
        t.h('Networks and training')
        t.prose('<p>Every network is a ResNet-20 (banner above) trained with the ModelZoo recipe: SGD with momentum 0.9, '
                'learning rate 0.1 divided by 10 at 50% and 75% of training, weight decay 1e-4, batch 128, 64,000 '
                'steps (250,000 for ImageNet-32), pad-4 random crop and horizontal flip (no flip for SVHN, no '
                'augmentation for the shuffled-label network). Inputs are normalised by each training set\'s own '
                'channel statistics, also on every probe, so the object compared is the map from raw image to '
                'activations. The colour of each training set is the same on every tab.</p>')
        t.table(*training_table(x1, x3), numeric={3, 4, 5, 6, 7})
        t.prose('<p class="note">Train accuracy on 10,000 training images, test accuracy on the network\'s own test split, both in eval mode at the end of training; for CIFAR-10 animals and vehicles, the CIFAR-10 '
                'test images of those classes. The two zoo networks have no stored training log here: their test accuracy '
                'is the one measured in the X3 run (full CIFAR-10 test set). Probes: <code>c10</code> (CIFAR-10 test, 200 '
                'per class, the published probe), <code>c100</code> (CIFAR-100 test, 20 per fine class), <code>svhn</code> '
                '(SVHN test, 200 per digit), <code>in32</code> (ImageNet-32 validation, 2 per class); none is in any '
                'training set.</p>')

    t.h('Design decisions made on the owner\'s behalf')
    t.prose('<p>From <code>xp/SPEC.md</code>, including the revisions after review, which override the original text.</p>')
    t.table(['Decision', 'Default chosen', 'Alternative', 'How to test'], [
        ['Architecture', 'ResNet-20 at 32 px for every training set, including ImageNet-32', 'A pretrained 224 px ImageNet network',
         'Rejected: it would confound resolution and architecture with training data'],
        ['Seeds', 'Two per new training set; three CIFAR-10 networks (two zoo, one retrained here)', 'One seed',
         'Cross-set agreement is read against each set\'s own seed pair'],
        ['What X1 compares', 'Training pipelines (images, labels, number of classes, normalisation, augmentation)',
         'Isolate the images alone', 'Would need networks differing in one factor only'],
        ['X1 joint solve', 'Trunk only, stem to b9 (10 sites)', 'Full 12-site chain',
         'The head changes with the dataset (4 to 1000 outputs) and enters every slice through the coupling; the full chain is shown as a check'],
        ['Kernels', '<code>knn15</code> primary for X1; <code>ea100</code> primary for X3 and X4', 'The other kernel',
         'Both are computed; X1 offers <code>ea100</code> as a check, X3 checks switch sites with <code>knn15</code>'],
        ['Smooth-kernel width', 'Perplexity 100, not tuned', 'Perplexity 30 or 300', 'Not run'],
        ['New points (X3)', 'Frozen-probe extension', 'Add the new points to the probe and refit',
         'Rejected: clean and attacked twins would be each other\'s nearest neighbours and distort the graph'],
        ['Dating effects (X3)', 'Site-local readouts only (activations, site-local class mass, site-local embedding)',
         'Joint coordinates', 'A joint coordinate at site l uses every site\'s row; the logits-only check measures the leak'],
        ['X3 cohorts', 'Fixed in advance: success / failure; transfer as "any flip" and "same destination"', '—', '—'],
        ['Shuffled-label network', 'A trained reference (no augmentation, labels shuffled once)', 'A floor isolating labels',
         'It differs from the others in more than labels'],
        ['X4 outcome', 'Localised stretching: window share of arc length (fastest 1/16 of s); σ and S kept as secondary',
         'Look for jumps', 'The network is continuous and piecewise affine; a 1,025-point refinement checks resolution'],
        ['X4 embedding of path points', 'Suffix operators: one joint operator per start site over start..logits, path points extended into it',
         'Truncated chain of the full operator', 'Masked replay of the probe; endpoints processed as images and as path points agree'],
        ['X4 angular gain', 'Fixed-radius derivative between the unit spheres the embedding uses', 'Raw Jacobian only',
         'Both are stored; points near the probe mean are flagged'],
        ['X4 log gain and FTLE', 'Report log σ<sub>max</sub>; FTLE = log gain / sites crossed, shown only where depths are compared',
         'Divide everywhere', '—'],
        ['X4 pair comparison', 'Same- and different-class pairs compared within terciles of start-site endpoint distance',
         'Pooled comparison', 'Tercile table under F4.1'],
    ])


def training_table(x1, x3):
    tr = x1.get('training', {})
    nets = []
    for p in x1['probes']:
        for n in x1['measures'][p]['nets']:
            if n not in nets:
                nets.append(n)
    grp = lambda n: {'c10x': 'c10'}.get(n.split('-')[0], n.split('-')[0])  # noqa: E731
    order = list(GROUPS)
    nets.sort(key=lambda n: (order.index(grp(n)) if grp(n) in order else 99, n))
    clean_acc = (x3 or {}).get('info', {}).get('clean_acc', {})
    rows = []
    for n in nets:
        g = grp(n)
        if n in tr:
            d = tr[n]
            src = 'trained here' + (' (control for the training recipe)' if n.startswith('c10x') else
                                    ' (control for label granularity: CIFAR-100 images, 20 coarse labels)' if n.startswith('c100c') else '')
            rows.append([f'{swatch(g)}<code>{n}</code>', glabel(g), src, d['classes'], f"{d['n_train']:,}", f"{d['steps']:,}",
                         f"{100 * d['train_acc']:.1f}%", f"{100 * d['test_acc']:.1f}%"])
        elif g == 'init':
            rows.append([f'{swatch(g)}<code>{n}</code>', glabel(g), 'zoo checkpoint at step 0', 10, '—', '0', '—', '—'])
        else:
            ta = clean_acc.get(n)
            rows.append([f'{swatch(g)}<code>{n}</code>', glabel(g), 'zoo checkpoint (ModelZoo)', 10, '50,000', '64,000', '—',
                         f'{100 * ta:.1f}%' if ta is not None else '—'])
    return ['Network', 'Training set', 'Source', 'Classes', 'Training images', 'Steps', 'Train acc.', 'Test acc.'], rows


# ------------------------------------------------------------------------------------------------------- X1 --

def x1_payload(x1):
    present = [g for g in GROUPS if any(g in x1['measures'][p]['group'] for p in x1['probes'])]
    extra = sorted({g for p in x1['probes'] for g in x1['measures'][p]['group']} - set(GROUPS))
    order = present + extra  # unknown groups (if any) go last, grey
    meas = {}
    for p in x1['probes']:
        m = x1['measures'][p]
        meas[p] = dict(nets=m['nets'], group=m['group'], labels_n=m['labels_n'], cells=m['cells'], null=m['null'],
                       readout=m['readout'], readout_coarse=m.get('readout_coarse'), matrix=m['matrix'], modes=m['modes'],
                       spectrum=dict(vals=m['spectrum']['vals']),
                       share={k: v['K16'] for k, v in m['share'].items()})
    return dict(probes=x1['probes'], order=[g for g in x1['groups'] if g in order] + [g for g in order if g not in x1['groups']],
                measures=meas, mlabel={k: v[0] for k, v in MEASURES.items()}, trunk={k: v[2] for k, v in MEASURES.items()},
                chance={k: v[3] for k, v in MEASURES.items()})


def cell(x1, probe, meas, a, b):
    if probe not in x1['measures']:
        return None
    cells = x1['measures'][probe]['cells'][meas]
    return (cells.get(f'{a}|{b}') or cells.get(f'{b}|{a}') or {}).get('mean')


def x1_text(x1):
    """Data-driven sentences for the X1 takeaway (claims from FINDINGS.md); each is skipped when its inputs are missing."""
    S = SITES.index
    J = 'joint_knn15_trunk'
    out = []
    q4 = lambda v: f'{f2(v[0])}, {f2(v[S("b3")])}, {f2(v[S("b6")])}, {f2(v[S("b9")])}'  # noqa: E731
    g = lambda a, b, p='c10': cell(x1, p, J, a, b)  # noqa: E731
    M = x1['measures'].get('c10', {})
    cells = M.get('cells', {}).get(J, {})
    zoo = None
    if 'c10|c10' in cells and 'c10-s0|c10-s1' in cells['c10|c10']['names']:
        zoo = cells['c10|c10']['pairs'][cells['c10|c10']['names'].index('c10-s0|c10-s1')]
    cc, a, b, ci, c1, i1 = g('c10', 'c10'), g('c10', 'c100'), g('c10', 'in32'), g('c100', 'in32'), g('c100', 'c100'), g('in32', 'in32')
    if cc and a:
        t = ('<b>Several natural-image pipelines give high agreement, below the matching seed pairs.</b> Joint agreement on the '
             f'CIFAR-10 probe at stem, b3, b6, b9: CIFAR-10 against CIFAR-100 {q4(a)}')
        if b:
            t += f'; CIFAR-10 against ImageNet-32 {q4(b)}'
        if ci and c1 and i1:
            t += (f'; CIFAR-100 against ImageNet-32 {q4(ci)}, the closest cross pair, against {q4(c1)} for the two CIFAR-100 seeds and '
                  f'{q4(i1)} for the two ImageNet-32 seeds')
        t += f'. CIFAR-10 seed pairs: {q4(cc)} as a mean of {len(cells["c10|c10"]["pairs"])} pairs'
        t += f' ({q4(zoo)} for the zoo pair alone).' if zoo else '.'
        an, ve = g('c10', 'anim'), g('c10', 'veh')
        if an and ve:
            t += f' The class-split networks are further away: animals only against CIFAR-10 {q4(an)}; vehicles only {q4(ve)}.'
        out.append(t)
    sa, sv_, si_ = g('c10', 'anim', 'svhn'), g('c10', 'veh', 'svhn'), g('c10', 'in32', 'svhn')
    ii, c1i, i1i = g('c100', 'in32', 'in32'), g('c100', 'c100', 'in32'), g('in32', 'in32', 'in32')
    if sa and sv_ and si_ and ii and c1i and i1i:
        t = ('<b>The ranking of pipelines depends on probe and depth, and this design does not isolate what drives it.</b> On the '
             f'CIFAR-10 probe the networks with more classes agree better late, but on the SVHN probe at b9 CIFAR-10 agrees more with '
             f'animals only ({f2(sa[S("b9")])}) and vehicles only ({f2(sv_[S("b9")])}) than with ImageNet-32 ({f2(si_[S("b9")])}), and on '
             f'the ImageNet-32 probe CIFAR-100 against ImageNet-32 falls to {f2(ii[S("b9")])} at b9, against {f2(c1i[S("b9")])} and '
             f'{f2(i1i[S("b9")])} for the seed pairs. Label count is confounded with image distribution and training performance.')
        fc, ff, cc2 = g('c100', 'c100c'), g('c100', 'c100'), g('c100c', 'c100c')
        fci, ffi, cci = g('c100', 'c100c', 'in32'), g('c100', 'c100', 'in32'), g('c100c', 'c100c', 'in32')
        if fc and ff and cc2:
            acc = x1.get('training', {})
            ta = [acc[n]['test_acc'] for n in ('c100c-s0', 'c100c-s1') if n in acc]
            t += (f' <b>Control for label granularity: CIFAR-100 images trained with the 20 coarse labels'
                  + (f' ({100 * sum(ta) / len(ta):.0f}% test accuracy)' if ta else '') +
                  f' give the same embedding as with 100 labels, to within the seed-pair spread:</b> fine against coarse {q4(fc)}, '
                  f'against {q4(ff)} for the two fine-label seeds and {q4(cc2)} for the two coarse-label seeds (CIFAR-10 probe')
            if fci and ffi and cci:
                t += (f'; on the ImageNet-32 probe {q4(fci)}, against seed pairs at {f2(ffi[S("b9")])} and {f2(cci[S("b9")])} at b9')
            t += ('). So a fivefold change in label granularity on fixed images does not move the embedding, which points at the '
                  'images (the class-split networks also see different images) as what separates the pipelines, not the number of classes.')
        else:
            t += ' A direct test of label count (CIFAR-100 images with the 20 coarse labels) is being trained.'
        out.append(t)
    s, i_, r_ = g('c10', 'svhn'), g('c10', 'init'), g('c10', 'rand')
    s2, i2, r2 = g('c10', 'svhn', 'svhn'), g('c10', 'init', 'svhn'), g('c10', 'rand', 'svhn')
    if s and i_ and r_:
        t = (f'<b>CIFAR-10 and SVHN pipelines agree much less than seed pairs, at every depth.</b> {f2(s[0])} at the stem and '
             f'{f2(s[S("b9")])} at b9, against {f2(i_[0])} and {f2(i_[S("b9")])} for a CIFAR-10 network with the untrained network, '
             f'and {f2(r_[0])} and {f2(r_[S("b9")])} with the shuffled-label network: the stem already differs.')
        if s2 and i2 and r2:
            t += (f' On the SVHN probe the order of the three reverses at b9 ({f2(s2[S("b9")])}, {f2(i2[S("b9")])}, {f2(r2[S("b9")])}). '
                  'The design cannot say that similar images are necessary: normalisation and augmentation also differ.')
        out.append(t)
    ss, ss_c, cc_s, ic, iI = g('svhn', 'svhn', 'svhn'), g('svhn', 'svhn'), g('c10', 'c10', 'svhn'), g('in32', 'in32'), g('in32', 'in32', 'in32')
    if ss and ss_c and cc and cc_s:
        t = (f'<b>For these two pipelines, a seed pair agrees less on the other\'s probe, most of all late.</b> SVHN seeds: {q4(ss)} on SVHN '
             f'digits against {q4(ss_c)} on CIFAR-10. CIFAR-10 seeds: {f2(cc[S("b9")])} at b9 on CIFAR-10 against {f2(cc_s[S("b9")])} on SVHN.')
        if ic and iI:
            t += (f' This is not a general rule: the two ImageNet-32 seeds agree at {f2(ic[S("b9")])} at b9 on CIFAR-10 images and '
                  f'{f2(iI[S("b9")])} on ImageNet-32 images.')
        out.append(t)
    n_cc, n_cx = cell(x1, 'c10', 'nn', 'c10', 'c10'), cell(x1, 'c10', 'nn', 'c10', 'c100')
    wz = M.get('null', {}).get(J, {}).get('within_pairs', {}).get('c10-s0|c10-s1')
    if n_cc and n_cx:
        t = (f'<b>Exact neighbour overlap falls much more with depth than the joint cosine.</b> CIFAR-10 pairs share {pc(n_cc[0])} of their '
             f'15 nearest neighbours at the stem, {pc(n_cc[S("b9")])} at b9 and {pc(n_cc[S("logits")])} at the logits (mean of three '
             f'pairs); CIFAR-10 against CIFAR-100 {pc(n_cx[0])}, {pc(n_cx[S("b9")])} and {pc(n_cx[S("logits")])}.')
        if wz and zoo:
            t += (f' For the zoo CIFAR-10 pair, shuffling images within class reduces the joint cosine from {f2(zoo[0])} to {f2(wz[0])} at '
                  f'the stem but only from {f2(zoo[S("b9")])} to {f2(wz[S("b9")])} at b9: late joint agreement is largely agreement about '
                  'class structure, early agreement is about individual images.')
        out.append(t)
    if M:
        def ro(gg, site):
            v = [r[S(site)] for r, x in zip(M['readout'], M['group']) if x == gg]
            return sum(v) / len(v) if v else None
        if all(ro(x, 'b9') is not None for x in ('c100', 'c10', 'init', 'svhn')):
            out.append(f'<b>Class readout.</b> A CIFAR-100 network puts {pc(ro("c100", "b9"))} of a CIFAR-10 image\'s 15 neighbours in its own '
                       f'class at b9, against {pc(ro("c10", "b9"))} for CIFAR-10 networks and {pc(ro("init", "b9"))} untrained. SVHN networks '
                       f'are at {pc(ro("svhn", "b9"))} at b9 and {pc(ro("svhn", "pool"))} at pool, below untrained.')
    if zoo and 'c10-s0|c10x-s0' in cells.get('c10|c10', {}).get('names', []):
        x = cells['c10|c10']['pairs'][cells['c10|c10']['names'].index('c10-s0|c10x-s0')]
        out.append(f'<b>Controls.</b> The network retrained here on CIFAR-10 (<code>c10x-s0</code>) agrees with <code>c10-s0</code> slightly '
                   f'better than the zoo pair does early ({f2(x[0])} against {f2(zoo[0])} at the stem); no equivalence test was run. '
                   'Permutation nulls sit at their chance baselines (table below).')
    return out


def tab_x1(nb, x1, x3):
    t = nb.tab('x1', 'X1 Different training sets')
    P = x1_payload(x1)
    nb.dataset('x1', P)
    probes = x1['probes']
    nets0 = x1['measures'][probes[0]]['nets']
    t.prose(
        '<p><b>Question.</b> Two ResNet-20s trained on different training sets see the same 2,000 probe images. How '
        'similar are their through-depth embeddings at each site, compared with two seeds trained on the same data and '
        f'with untrained networks? {len(nets0)} networks from {len(set(x1["measures"][probes[0]]["group"]))} training sets, '
        f'{len(probes)} probes. Site $l$ of one network is compared with site $l$ of the other. What differs between '
        'training sets is the whole pipeline (images, labels, number of classes, normalisation, augmentation), not the '
        'images alone.</p>')
    missing = [g for g in GROUPS if g not in P['order']]
    if missing:
        t.prose(f'<p class="note">Not yet in the summary: {", ".join(glabel(g) for g in missing)}. A rebuild after '
                '<code>analyze_x1.py</code> picks them up.</p>')
    tx = x1_text(x1)
    if tx:
        t.takeaway('<ul>' + ''.join(f'<li>{s}</li>' for s in tx) + '</ul>', label='What the figures show (numbers from the summary)')

    # ---- F1.1
    t.h('F1.1 Agreement against site, by probe')
    order = P['order']
    tr_groups = [g for g in order]
    t.figure(
        'x1_curves', 'Agreement between training sets against site · ResNet-20 · four probes',
        data='@x1',
        controls=[Select('meas', 'Measure', list(MEASURES), labels={k: v[0] for k, v in MEASURES.items()}),
                  Select('ref', 'Reference training set', tr_groups, default='c10', labels={g: glabel(g) for g in tr_groups}),
                  Select('hl', 'Highlight', ['none'] + tr_groups, default='none',
                         labels={'none': 'none', **{g: glabel(g) for g in tr_groups}})],
        draw=DRAW_X1_CURVES, height=700,
        caption='Each panel is one probe. A solid line is the mean, over all network pairs, of the agreement between the '
                'reference training set (selector) and another training set, in that set\'s colour; the small points are the '
                'individual network pairs (2 to 6 per line). The thick dashed line is the reference set\'s own seed pair '
                '(same data, different seed); the grey dotted line is the two untrained networks. Black dotted: image order '
                'permuted (chance); grey dash-dot: images permuted within class, averaged over same-data seed pairs, which '
                'is the agreement class structure alone would give. Choose "Highlight" to bring one training set forward '
                'together with its own seed pair. Shaded: pool and logits, which the trunk solve (the primary measure) '
                'leaves out; faint dotted lines mark the stage boundaries (16, 32, 64 channels). The y-axis is fixed per '
                'measure across references.')

    # ---- F1.2
    t.h('F1.2 Network by network, at three depths')
    t.figure(
        'x1_matrix', 'Network-by-network similarity at b2, b6 and b9',
        data='@x1',
        controls=[Radio('m_probe', 'Probe', probes, labels=PROBES),
                  Radio('m_meas', 'Measure', ['joint_knn15_trunk', 'nn', 'cka'],
                        labels={'joint_knn15_trunk': 'joint agreement', 'nn': 'neighbour overlap', 'cka': 'CKA'}),
                  Radio('m_level', 'Rows', ['net', 'set'], labels={'net': 'networks', 'set': 'training sets (mean)'})],
        draw=DRAW_X1_MATRIX, height=440,
        caption='Each cell is the similarity of two networks on the chosen probe at one site; networks are ordered by '
                'training set and thin lines separate the sets. With "training sets (mean)" each cell is the mean over '
                'network pairs and the diagonal is the set\'s own seed pair (blank for the shuffled-label network, which '
                'has one seed). Colour scale fixed at 0 to 1. Read it for whether networks group by training set late but '
                'not early.')

    # ---- sensitivity
    t.h('Sensitivity to kernel and chain')
    t.figure(
        'x1_sens', 'Primary measure against its checks · every pair of training sets × trunk site',
        data='@x1',
        controls=[Radio('s_probe', 'Probe', probes, labels=PROBES),
                  Radio('s_alt', 'Check', ['joint_ea100_trunk', 'joint_knn15_full', 'joint_ea100_full'],
                        labels={'joint_ea100_trunk': 'smooth kernel, trunk', 'joint_knn15_full': 'k-NN, full chain',
                                'joint_ea100_full': 'smooth kernel, full chain'})],
        draw=DRAW_X1_SENS, height=430,
        caption='Each point is one pair of training sets at one site (stem to b9), mean over network pairs: x is the primary '
                'joint agreement (trunk, k-NN kernel), y the same pair under the chosen check. Colour runs from the stem '
                '(light) to b9 (dark). Points on the diagonal mean the check changes nothing.')

    # ---- F1.3
    t.h('F1.3 Mode by mode')
    pairs = list(x1['measures'][probes[0]]['modes'])
    t.figure(
        'x1_modes', 'Joint modes of two networks: mode-by-mode overlap, principal angles, spectra',
        data='@x1',
        controls=[Radio('d_probe', 'Probe', probes, labels=PROBES),
                  Select('d_pair', 'Pair', pairs, default='c10-s0|c100-s0' if 'c10-s0|c100-s0' in pairs else pairs[0], labels={p: p.replace('|', ' vs ') for p in pairs})],
        draw=DRAW_X1_MODES, height=420,
        caption='Left: $|\\langle u_j^B, u_k^A\\rangle|$ for the first 16 trunk modes (k-NN kernel) of two networks, A = '
                '<code>c10-s0</code> on the columns. A bright diagonal means the modes match one to one; blocks mean a '
                'subspace is shared but its basis is mixed, as expected where eigenvalues are close. Middle: cosines of the '
                'principal angles between the two 16-mode subspaces, for every stored pair (selected pair bold): the number of '
                'values near 1 is the dimension of the shared subspace, independent of the basis. Right: the two operators\' '
                'eigenvalues; nearly equal neighbours (small gaps) mean those individual modes are not identifiable and '
                'should not be read one by one.')

    # ---- F1.4
    t.h('F1.4 Class readout')
    t.figure(
        'x1_readout', 'Share of an image\'s 15 nearest probe neighbours with its own label, against site',
        data='@x1',
        controls=[Radio('r_coarse', 'CIFAR-100 labels', ['fine', 'coarse'], labels={'fine': '100 fine', 'coarse': '20 coarse'}),
                  Radio('r_scale', 'y-axis', ['probe', 'shared'], labels={'probe': 'per probe', 'shared': '0 to 1 in every panel'})],
        draw=DRAW_X1_READOUT, height=700,
        caption='One line per network, in its training set\'s colour; seed 0 solid, seed 1 dashed, the retrained CIFAR-10 '
                'network dotted. The y-axis is scaled per probe by default (printed in the panel title when short); switch to 0 to 1 to compare probes. Black dotted: chance, (images per class − 1)/1999 — about 0.10 for CIFAR-10 and SVHN, 0.01 '
                'for CIFAR-100 fine labels, 0.0005 for ImageNet-32 (2 images per class, so this readout says little '
                'there). This is a property of each network, not of a pair: how well a network trained on one set separates '
                'another set\'s classes.')

    t.prose('<p class="note">F1.5 of the spec (side-by-side scatter of two networks\' embeddings after the rotation) is not '
            'drawn: the summary holds no per-image coordinates for X1. The X3 tab shows what such a scatter looks like for '
            'one network.</p>')

    # controls with known answers
    J = 'joint_knn15_trunk'
    cc = x1['measures']['c10']['cells'][J].get('c10|c10') if 'c10' in x1['measures'] else None
    if cc:
        rows = []
        for nm, v in zip(cc['names'], cc['pairs']):
            rows.append([nm.replace('|', ' vs '), f2(v[0]), f2(v[SITES.index('b6')]), f2(v[SITES.index('b9')])])
        nul = x1['measures']['c10']['null'][J]
        rows.append(['image order permuted (mean over all pairs)', f2(nul['perm'][0]), f2(nul['perm'][6]), f2(nul['perm'][9])])
        t.h('Controls with known answers', 3)
        t.prose('<p>The network retrained here on CIFAR-10 (<code>c10x-s0</code>) should look like a zoo seed; permuted images '
                'should give 0. Joint agreement on the CIFAR-10 probe:</p>')
        t.table(['Pair', 'stem', 'b6', 'b9'], rows, numeric={1, 2, 3})
        t.prose('<p class="note">Also checked when the data were made (reported in FINDINGS.md, not stored in the summary): the reference '
                'network\'s Grams match the published ones to 1.6e-6, its full-chain joint eigenvalues to 1.4e-12 and its coupling to '
                '2e-11. A network against itself gives 1 (diagonal of F1.2). No equivalence test was run for the retrained network.</p>')


DRAW_X1_CURVES = r"""(el, d, c) => {
  const meas = c.meas, A = c.ref, hl = c.hl, order = d.order, trunk = d.trunk[meas];
  let lo = 0, hi = 0;   // fixed y range per measure across references
  for (const p of d.probes) { const M = d.measures[p];
    for (const k in M.cells[meas]) for (const r of M.cells[meas][k].pairs) for (const v of r) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    for (const v of M.null[meas].within_same) hi = Math.max(hi, v); }
  const traces = [], layout = LAYOUT({height: 700, showlegend: true, legend: {orientation: 'h', y: -0.07, x: 0, font: {size: 11.5}},
    margin: {l: 56, r: 14, t: 28, b: 70}}), shapes = [], ann = [];
  const legendSeen = {};
  const once = k => { if (legendSeen[k]) return false; legendSeen[k] = true; return true; };
  d.probes.forEach((p, i) => {
    const M = d.measures[p], present = order.filter(g => M.group.includes(g)), dm = dom4(i);
    layout['xaxis' + ax(i)] = siteAxis(i, {domain: dm.x, showticklabels: i >= 2});
    layout['yaxis' + ax(i)] = merge(AXIS, {domain: dm.y, range: [Math.min(-0.03, lo - 0.02), Math.max(1.0, hi) + 0.02], anchor: 'x' + ax(i),
      title: {text: i % 2 === 0 ? 'agreement' : ''}});
    shapes.push(...stageLines(i)); if (trunk) shapes.push(headShade(i));
    ann.push(panelTitle(i, 'probe: ' + PROBENAME[p] + (M.group.includes(A) ? '' : ' · ' + GL(A) + ' not measured')));
    const L = trunk ? 10 : 12, xs = SITES.slice(0, L).map((_, k) => k);
    const others = present.filter(g => g !== A);
    others.forEach((B, j) => {
      const cl = M.cells[meas][cellKey(order, A, B)]; if (!cl) return;
      const faded = hl !== 'none' && hl !== B, off = (j - (others.length - 1) / 2) * 0.045;
      traces.push({x: xs, y: cl.mean, mode: 'lines', name: 'vs ' + GL(B), legendgroup: B, showlegend: once(B),
        line: {color: GC(B), width: faded ? 1.2 : (hl === B ? 3 : 2)}, opacity: faded ? 0.35 : 1,
        xaxis: 'x' + ax(i), yaxis: 'y' + ax(i),
        hovertemplate: GL(A) + ' vs ' + GL(B) + ' · %{text}<br>mean %{y:.3f} over ' + cl.pairs.length + ' pairs<extra></extra>', text: SITES.slice(0, L)});
      if (!faded) cl.pairs.forEach((r, q) => traces.push({x: xs.map(x => x + off), y: r, mode: 'markers', showlegend: false, legendgroup: B,
        marker: {color: GC(B), size: 4.5, opacity: 0.6}, xaxis: 'x' + ax(i), yaxis: 'y' + ax(i),
        hovertemplate: cl.names[q].replace('|', ' vs ') + ' · %{y:.3f}<extra></extra>'}));
    });
    const own = (g, w) => { const cl = M.cells[meas][g + '|' + g]; if (!cl) return;
      traces.push({x: xs, y: cl.mean, mode: 'lines', name: GL(g) + ': own seed pair', legendgroup: 'own' + g, showlegend: once('own' + g),
        line: {color: GC(g), width: w, dash: 'dash'}, xaxis: 'x' + ax(i), yaxis: 'y' + ax(i),
        hovertemplate: GL(g) + ' seed pair · %{y:.3f}<extra></extra>'}); };
    own(A, 3.2); if (hl !== 'none' && hl !== A) own(hl, 2.4);
    const ii = M.cells[meas]['init|init'];
    if (ii && A !== 'init') traces.push({x: xs, y: ii.mean, mode: 'lines', name: 'untrained pair (two init networks)', legendgroup: 'initpair', showlegend: once('initpair'),
      line: {color: '#8d877c', width: 2.4, dash: 'dot'}, xaxis: 'x' + ax(i), yaxis: 'y' + ax(i), hovertemplate: 'untrained pair · %{y:.3f}<extra></extra>'});
    const nl = M.null[meas];
    traces.push({x: xs, y: nl.perm, mode: 'lines', name: 'null: images permuted', legendgroup: 'perm', showlegend: once('perm'),
      line: {color: '#2b2a27', width: 1.2, dash: 'dot'}, xaxis: 'x' + ax(i), yaxis: 'y' + ax(i), hovertemplate: 'permuted · %{y:.3f}<extra></extra>'});
    traces.push({x: xs, y: nl.within_same, mode: 'lines', name: 'null: permuted within class (same-data pairs)', legendgroup: 'within',
      showlegend: once('within'), line: {color: '#8d877c', width: 1.4, dash: 'dashdot'}, xaxis: 'x' + ax(i), yaxis: 'y' + ax(i),
      hovertemplate: 'within-class permuted · %{y:.3f}<extra></extra>'});
  });
  layout.shapes = shapes; layout.annotations = ann;
  layout.title = {text: d.mlabel[meas] + ' · reference: ' + GL(A), font: {size: 13}, x: 0.0, xanchor: 'left', y: 0.995};
  layout.margin.t = 44;
  Plotly.react(el, traces, layout, CONFIG);
}"""

DRAW_X1_MATRIX = r"""(el, d, c) => {
  const M = d.measures[c.m_probe], order = d.order, sites = ['b2', 'b6', 'b9'];
  let labels, val, seps = [];
  if (c.m_level === 'net') {
    const idx = M.nets.map((n, k) => k).sort((a, b) => (order.indexOf(M.group[a]) - order.indexOf(M.group[b])) || M.nets[a].localeCompare(M.nets[b]));
    labels = idx.map(k => M.nets[k]);
    for (let k = 1; k < idx.length; k++) if (M.group[idx[k]] !== M.group[idx[k - 1]]) seps.push(k - 0.5);
    val = s => idx.map(a => idx.map(b => M.matrix[c.m_meas][s][a][b]));
  } else {
    const gs = order.filter(g => M.group.includes(g)); labels = gs.map(GL);
    const si = s => SITES.indexOf(s);
    val = s => gs.map(a => gs.map(b => { const cl = M.cells[c.m_meas][cellKey(order, a, b)]; return cl ? cl.mean[si(s)] : null; }));
  }
  const traces = [], layout = LAYOUT({height: 440, margin: {l: 70, r: 20, t: 28, b: 70}}), shapes = [], ann = [];
  const w = 0.3, g = 0.035;
  sites.forEach((s, i) => {
    const x0 = i * (w + g);
    traces.push({type: 'heatmap', z: val(s), x: labels, y: labels, zmin: 0, zmax: 1, zauto: false, colorscale: BLUES,
      showscale: i === 2, colorbar: {thickness: 10, len: 0.8, title: {text: 'similarity', side: 'right'}},
      xaxis: 'x' + ax(i), yaxis: 'y' + ax(i), hovertemplate: '%{y} vs %{x}<br>' + s + ': %{z:.3f}<extra></extra>', xgap: 0.5, ygap: 0.5});
    layout['xaxis' + ax(i)] = merge(AXIS, {domain: [x0, x0 + w], tickangle: -60, tickfont: {size: 9.5}, anchor: 'y' + ax(i), ticks: '', showgrid: false});
    layout['yaxis' + ax(i)] = merge(AXIS, {domain: [0, 1], autorange: 'reversed', showticklabels: i === 0, tickfont: {size: 9.5}, anchor: 'x' + ax(i),
      ticks: '', showgrid: false, scaleanchor: 'x' + ax(i)});
    ann.push(panelTitle(i, 'site ' + s));
    for (const sp of seps) { shapes.push({type: 'line', xref: 'x' + ax(i), yref: 'y' + ax(i), x0: sp, x1: sp, y0: -0.5, y1: labels.length - 0.5, line: {color: 'white', width: 2}});
      shapes.push({type: 'line', xref: 'x' + ax(i), yref: 'y' + ax(i), y0: sp, y1: sp, x0: -0.5, x1: labels.length - 0.5, line: {color: 'white', width: 2}}); }
  });
  layout.shapes = shapes; layout.annotations = ann;
  Plotly.react(el, traces, layout, CONFIG);
}"""

DRAW_X1_SENS = r"""(el, d, c) => {
  const M = d.measures[c.s_probe], P = 'joint_knn15_trunk', Q = c.s_alt;
  const xs = [], ys = [], col = [], txt = [];
  for (const k in M.cells[P]) { const a = M.cells[P][k].mean, b = (M.cells[Q][k] || {}).mean; if (!b) continue;
    for (let l = 0; l < 10; l++) { xs.push(a[l]); ys.push(b[l]); col.push(l); txt.push(k.split('|').map(GL).join(' vs ') + ' · ' + SITES[l]); } }
  const tr = [{x: [-0.05, 1.02], y: [-0.05, 1.02], mode: 'lines', line: {color: '#b9b3a8', dash: 'dot', width: 1}, hoverinfo: 'skip', showlegend: false},
    {x: xs, y: ys, text: txt, mode: 'markers', showlegend: false, marker: {size: 7, color: col, cmin: 0, cmax: 9,
      colorscale: [[0, '#c6dbef'], [1, '#08306b']], line: {color: 'white', width: 0.6},
      colorbar: {thickness: 10, len: 0.8, tickvals: [0, 3, 6, 9], ticktext: ['stem', 'b3', 'b6', 'b9'], title: {text: 'site'}}},
      hovertemplate: '%{text}<br>primary %{x:.3f} · check %{y:.3f}<extra></extra>'}];
  Plotly.react(el, tr, LAYOUT({height: 430, width: Math.min(el.clientWidth, 620),
    xaxis: merge(AXIS, {title: {text: 'joint agreement, trunk, k-NN (primary)'}, range: [-0.05, 1.02]}),
    yaxis: merge(AXIS, {title: {text: d.mlabel[Q].replace(' (check)', '')}, range: [-0.05, 1.02], scaleanchor: 'x'})}), CONFIG);
}"""

DRAW_X1_MODES = r"""(el, d, c) => {
  const M = d.measures[c.d_probe];
  const keys = Object.keys(M.modes), pair = keys.includes(c.d_pair) ? c.d_pair : keys[0];
  const [A, B] = pair.split('|'), m = M.modes[pair].m, ks = [...Array(16).keys()].map(k => k + 1);
  const tr = [{type: 'heatmap', z: m, x: ks, y: ks, zmin: 0, zmax: 1, zauto: false, colorscale: BLUES, xaxis: 'x', yaxis: 'y',
    colorbar: {thickness: 10, len: 0.85, x: 0.31, xanchor: 'left', title: {text: '|cos|', side: 'top'}},
    hovertemplate: A + ' mode %{x} · ' + B + ' mode %{y}<br>|cos| %{z:.3f}<extra></extra>'}];
  for (const k of keys) { const sel = k === pair, b = k.split('|')[1], g = M.group[M.nets.indexOf(b)];
    tr.push({x: ks, y: M.modes[k].sv, mode: sel ? 'lines+markers' : 'lines', xaxis: 'x2', yaxis: 'y2', name: k.replace('|', ' vs '),
      line: {color: GC(g), width: sel ? 3 : 1.2}, opacity: sel ? 1 : 0.55, marker: {size: 6}, showlegend: false,
      hovertemplate: k.replace('|', ' vs ') + ' · angle %{x}: %{y:.3f}<extra></extra>'}); }
  for (const [n, dash] of [[A, 'solid'], [B, 'dash']]) { const k = M.nets.indexOf(n), g = M.group[k];
    tr.push({x: ks, y: M.spectrum.vals[k], mode: 'lines+markers', xaxis: 'x3', yaxis: 'y3', name: n, line: {color: GC(g), dash, width: 2},
      marker: {size: 5}, hovertemplate: n + ' mode %{x}: eigenvalue %{y:.3f}<extra></extra>'}); }
  const layout = LAYOUT({height: 420, legend: {x: 0.79, y: 0.02, orientation: 'v', bgcolor: 'rgba(255,255,255,.8)'}, margin: {l: 56, r: 14, t: 34, b: 50},
    xaxis: merge(AXIS, {domain: [0, 0.3], title: {text: 'mode of ' + A}, dtick: 3, showgrid: false}),
    yaxis: merge(AXIS, {domain: [0, 1], title: {text: 'mode of ' + B}, autorange: 'reversed', dtick: 3, showgrid: false}),
    xaxis2: merge(AXIS, {domain: [0.47, 0.69], anchor: 'y2', title: {text: 'principal angle index'}}),
    yaxis2: merge(AXIS, {domain: [0, 1], anchor: 'x2', range: [0, 1.02], title: {text: 'cosine of principal angle'}}),
    xaxis3: merge(AXIS, {domain: [0.79, 1], anchor: 'y3', title: {text: 'mode'}}),
    yaxis3: merge(AXIS, {domain: [0, 1], anchor: 'x3', title: {text: 'eigenvalue (trunk operator)'}}),
    annotations: [panelTitle(0, pair.replace('|', ' vs ') + ' · probe ' + PROBENAME[c.d_probe]), panelTitle(1, 'principal angles, all pairs'),
      panelTitle(2, 'spectra of the two networks')]});
  Plotly.react(el, tr, layout, CONFIG);
}"""

DRAW_X1_READOUT = r"""(el, d, c) => {
  const traces = [], layout = LAYOUT({height: 700, legend: {orientation: 'h', y: -0.07, font: {size: 11.5}}, margin: {l: 56, r: 14, t: 28, b: 70}});
  const shapes = [], ann = [], seen = {};
  d.probes.forEach((p, i) => {
    const M = d.measures[p], dm = dom4(i);
    const coarse = p === 'c100' && c.r_coarse === 'coarse' && M.readout_coarse;
    const R = coarse ? M.readout_coarse : M.readout, ncls = coarse ? 20 : M.labels_n, chance = (2000 / ncls - 1) / 1999;
    layout['xaxis' + ax(i)] = siteAxis(i, {domain: dm.x, showticklabels: i >= 2});
    let top = 1; if (c.r_scale === 'probe') { top = chance; for (const r of R) for (const v of r) top = Math.max(top, v); top *= 1.08; }
    layout['yaxis' + ax(i)] = merge(AXIS, {domain: dm.y, range: [0, top], anchor: 'x' + ax(i), title: {text: i % 2 === 0 ? 'same-label share' : ''},
      tickformat: top < 0.05 ? '.3f' : null});
    shapes.push(...stageLines(i));
    ann.push(panelTitle(i, 'probe: ' + PROBENAME[p] + (p === 'c100' ? (coarse ? ' (20 coarse labels)' : ' (100 fine labels)') : '') +
      (c.r_scale === 'probe' && top < 0.5 ? ' · y-axis to ' + top.toFixed(top < 0.05 ? 4 : 2) : '')));
    const idx = M.nets.map((n, k) => k).sort((a, b) => d.order.indexOf(M.group[a]) - d.order.indexOf(M.group[b]));
    for (const k of idx) { const n = M.nets[k], g = M.group[k];
      const dash = n.startsWith('c10x') ? 'dot' : n.endsWith('-s1') ? 'dash' : 'solid';
      traces.push({x: SITES.map((_, q) => q), y: R[k], mode: 'lines', name: GL(g), legendgroup: g, showlegend: !seen[g],
        line: {color: GC(g), width: 1.8, dash}, xaxis: 'x' + ax(i), yaxis: 'y' + ax(i), text: SITES,
        hovertemplate: n + ' · %{text}: %{y:.3f}<extra></extra>'}); seen[g] = 1; }
    traces.push({x: [0, 11], y: [chance, chance], mode: 'lines', name: 'chance', showlegend: !seen.ch, line: {color: '#2b2a27', dash: 'dot', width: 1.2},
      xaxis: 'x' + ax(i), yaxis: 'y' + ax(i), hovertemplate: 'chance %{y:.4f}<extra></extra>'}); seen.ch = 1;
  });
  layout.shapes = shapes; layout.annotations = ann;
  Plotly.react(el, traces, layout, CONFIG);
}"""


# ------------------------------------------------------------------------------------------------------- X3 --

ATT = {  # name -> (label, colour, dash)
    'u1': ('untargeted ε 1/255', '#f3b37c', 'solid'), 'u2': ('untargeted ε 2/255', '#e8833a', 'solid'),
    'u4': ('untargeted ε 4/255', '#c85a12', 'solid'), 'u8': ('untargeted ε 8/255', '#7f3408', 'solid'),
    't8': ('targeted ε 8/255', '#6250d6', 'solid'),
    'n1': ('noise ε 1/255', '#d2ccc2', 'dash'), 'n2': ('noise ε 2/255', '#b2aba0', 'dash'),
    'n4': ('noise ε 4/255', '#8d877c', 'dash'), 'n8': ('noise ε 8/255', '#57534c', 'dash'),
}


def x3_text(x3):
    A, B = 'c10-s0', 'c10-s1'
    at = x3['attacks']
    S = SITES.index
    u, n = at['u8'][A]['disp_site_y'], at['n8'][A]['disp_site_y']
    ac_u, ac_n = at['u8'][A]['actcos'], at['n8'][A]['actcos']
    cm = x3['class_mass'][A]
    BINS = SITES + ['none']
    mode = lambda h: BINS[max(range(13), key=lambda k: h[k])]  # noqa: E731

    def median(h):
        c, tot = 0, sum(h)
        for k, v in enumerate(h):
            c += v
            if c >= tot / 2:
                return BINS[k]

    by = lambda h, site: sum(h[:S(site) + 1]) / sum(h)  # noqa: E731
    u8s, u8t = cm['u8_success']['switch'], cm['u8_success']['switch_top']
    kn = x3['class_mass']['knn15_u8_success_A']['switch']
    tv = 0.5 * sum(abs(a / sum(u8s) - b / sum(kn)) for a, b in zip(u8s, kn))
    arr, art = x3['arrival'][A]['u8_y']['median'], x3['arrival'][A]['t8_y']['median']
    ars = x3['arrival'][A]['t8_site_y']['median']
    sup_u, sup_n = at['u8'][A]['support'], at['n8'][A]['support']
    mid = [S(s) for s in ('b3', 'b4', 'b5', 'b6', 'b7')]
    tr, tr4, trt = at['u8']['transfer'], at['u4']['transfer'], at['t8']['transfer']
    fl, nf = at['u4'][B]['disp_site_y_flip'], at['u4'][B]['disp_site_y_noflip']
    lo = x3['logits_only']
    nat = cm['natural']
    out = [
        f'<b>In the embedding, an ε 8 attack moves an image about as far as random sign noise of the same L∞ size up to b6; the gap opens '
        f'at b7 and is very large at b9.</b> Site-local displacement, attack / noise (units of the median distance between two clean '
        f'images of one class): {f2(u[0])} / {f2(n[0])} (stem), {f2(u[S("b3")])} / {f2(n[S("b3")])} (b3), {f2(u[S("b6")])} / {f2(n[S("b6")])} (b6), '
        f'{f2(u[S("b7")])} / {f2(n[S("b7")])} (b7), {f2(u[S("b8")])} / {f2(n[S("b8")])} (b8), {f2(u[S("b9")])} / {f2(n[S("b9")])} (b9). The raw '
        f'activations already tell the two apart earlier: cosine to the clean activation at b6 is {f2(ac_u[S("b6")])} for the attack and '
        f'{f2(ac_n[S("b6")])} for noise, and {f2(ac_u[S("b9")])} against {f2(ac_n[S("b9")])} at b9. (Equal L∞ budget does not mean equal L2 size.)',
        f'<b>The class switch, read site by site, is spread across depth; it comes later for weaker attacks and latest for natural '
        f'errors.</b> Untargeted ε 8: median {median(u8s)}, mode {mode(u8s)} ({pc(by(u8s, "b6"))} have switched by b6; with the stricter '
        f'"destination is the top class" rule the median is {median(u8t)}). Targeted ε 8: median {median(cm["t8_success"]["switch"])}, mode '
        f'{mode(cm["t8_success"]["switch"])}; untargeted ε 2: median {median(cm["u2_success"]["switch"])}, mode {mode(cm["u2_success"]["switch"])}. '
        f'Natural errors: mode {mode(nat["switch"])}, and {pc(nat["switch"][12] / sum(nat["switch"]))} never switch. By b6 the ε 8 attacked '
        f'images carry more weight on the destination class ({f2(cm["u8_success"]["dst"][S("b6")])}) than on their own '
        f'({f2(cm["u8_success"]["src"][S("b6")])}), where clean images carry {f2(cm["clean"]["src"][S("b6")])} on their own. About '
        f'{pc(u8s[0] / sum(u8s))} of ε 8 successes "switch" at the stem, where class weights are near uniform: images whose own class never led.',
        f'<b>Successful ε 8 attacks end at typical distances from the destination class, with nearly all their weight on it.</b> At pool and '
        f'logits the untargeted attacked image\'s distance to the destination class centroid is at the {th(arr[10])} and '
        f'{th(arr[11])} percentile of real members\' distances in joint coordinates (targeted: {th(art[10])} and '
        f'{th(art[11])}; {th(ars[11])} at the logits in site-local coordinates), and {pc(cm["u8_success"]["dst"][11])} of its '
        f'kernel-row weight is on the destination class. Natural errors put {pc(nat["dst"][11])} on the destination, {pc(nat["src"][11])} on '
        f'the true class and {pc(nat["other"][11])} elsewhere.'
        + (f' Matching attacks to the same (source, destination) pairs does not change this contrast '
           f'({pc(cm["matched_natural_u8_A"]["natural_dst"])} against {pc(cm["matched_natural_u8_A"]["u8_dst"])} on the destination, '
           f'{cm["matched_natural_u8_A"]["n_combos"]} class pairs).' if 'matched_natural_u8_A' in cm else '')
        + f' In the middle of the network (b3–b7) attacked images have a lower nearest-probe cosine than clean ones (support percentile '
        f'{f2(min(sup_u[i] for i in mid))}–{f2(max(sup_u[i] for i in mid))}); random noise does too ({f2(min(sup_n[i] for i in mid))}–'
        f'{f2(max(sup_n[i] for i in mid))}).',
        f'<b>Transfer.</b> The ε 8 attack built on seed 0 flips seed 1 on {pc(tr["any_flip"])} of images, to the same wrong class on '
        f'{pc(tr["same_destination"])} (ε 4: {pc(tr4["any_flip"])} and {pc(tr4["same_destination"])}; targeted ε 8: {pc(trt["any_flip"])} '
        f'flipped, {pc(trt["same_destination"])} to the target). In seed 1, perturbations that flip it have a larger site-local displacement '
        f'than those that do not already at b7 (ε 4: {f2(fl[S("b7")])} against {f2(nf[S("b7")])}), with by far the largest separation at b9 '
        f'({f2(fl[S("b9")])} against {f2(nf[S("b9")])}). The split is by any flip.',
        f'<b>Method checks.</b> Replacing only the logits row of a clean image by its attacked row changes no site-local readout at earlier '
        f'sites (largest change {lo["local_max_change"]:g}) but moves the joint coordinates at the stem by {pc(lo["ratio"][0])} of the full '
        f'attack\'s displacement and at b9 by {pc(lo["ratio"][9])}: joint coordinates smear late effects backwards, which is why the '
        f'statements above use site-local readouts. The k-NN kernel gives the same modal switch site ({mode(kn)}) as the smooth kernel for '
        f'ε 8 untargeted attacks; the two histograms differ by {100 * tv:.0f} percentage points of total variation.',
    ]
    return out


def x3_payload(x3):
    keep = {k: x3[k] for k in ('sites', 'info', 'n', 'unit_same_class', 'attacks', 'class_mass', 'arrival', 'path', 'logits_only', 'extension')}
    keep['att'] = ATT
    return keep


def tab_x3(nb, x3):
    t = nb.tab('x3', 'X3 Adversarial images')
    nb.dataset('x3', x3_payload(x3))
    pic = x3['picture']
    nb.dataset('x3pic', {k: v for k, v in pic.items() if k != 'ex'} | {'ex': {k: v for k, v in pic['ex'].items() if not k.startswith('img_')}})
    info, A, B = x3['info'], 'c10-s0', 'c10-s1'
    at = x3['attacks']
    t.prose(
        f'<p><b>Question.</b> Where in depth does an adversarial perturbation take hold, and where does the attacked image end '
        f'up? <b>Setup.</b> Attacked network <code>{A}</code> (CIFAR-10, seed 0); transfer network <code>{B}</code> (seed 1). '
        f'Probe: the CIFAR-10 probe (2,000 images), with the smooth kernel (<code>ea100</code>). Scored images: the first '
        f'{x3["n"]:,} CIFAR-10 test images outside the probe that both networks classify correctly. Attacks on seed 0: PGD, '
        f'L∞, 50 steps, random start; untargeted at ε = 1, 2, 4, 8 /255 and targeted at 8/255 (target a random other class). '
        f'Baselines: random sign noise at each ε, and the {info.get("n_natA", "")} held-out test images seed 0 gets wrong '
        f'("natural errors").</p>'
        '<p><b>How to read it.</b> Claims about <i>where</i> an attack takes hold use site-local readouts only: raw activations, '
        'the kernel row at that site (class mass) and the site-local embedding. Joint coordinates use every site\'s row, so a '
        'change at the logits leaks into them at earlier sites (see the method check below); they are shown as retrospective '
        'pictures and for the arrival measure. Colours on this tab code attacks (orange ramp by ε, violet targeted, grey '
        'dashed noise), not training sets.</p>')
    t.takeaway('<ul>' + ''.join(f'<li>{s}</li>' for s in x3_text(x3)) + '</ul>', label='What the figures show (numbers from the summary)')

    rows = []
    for k, (lab, _, _) in ATT.items():
        a = at[k]
        trn = a.get('transfer')
        rows.append([lab, pc(a[A]['acc']), pc(a[B]['acc']),
                     pc(trn['success_A']) if trn else '—', pc(trn['any_flip']) if trn else '—', pc(trn['same_destination']) if trn else '—'])
    t.h('Accuracy under attack', 3)
    t.table(['Image set', f'Accuracy, {A} (attacked)', f'Accuracy, {B} (transfer)', 'Success on seed 0',
             'Seed 1 flipped (of successes)', 'Seed 1 same destination'], rows, numeric={1, 2, 3, 4, 5})
    t.prose('<p class="note">Success is "prediction differs from the label" (untargeted) or "prediction equals the requested target" '
            '(targeted, seed 0). Transfer rates are among the images on which the attack succeeded on seed 0. Clean accuracy is 100% '
            'by construction of the scored set.</p>')

    t.controls([Radio('net', 'Network', [A, B], labels={A: f'{A} (attacked)', B: f'{B} (transfer)'})])
    t.prose('<p class="note">The network switch drives F3.1 to F3.5. On the transfer network "success" means the prediction '
            'differs from the label, and the destination is that network\'s own prediction.</p>')

    t.h('F3.1 How far the image moves, site by site')
    t.figure('x3_disp', 'Displacement of attacked and noisy images from their clean versions, against site', data='@x3',
             controls=[Radio('dq', 'Readout', ['disp_site_y', 'actcos', 'relpert', 'disp_y'],
                             labels={'disp_site_y': 'site-local embedding', 'actcos': 'activation cosine', 'relpert': 'relative activation change',
                                     'disp_y': 'joint coordinates (retrospective)'})],
             draw=DRAW_X3_DISP, height=440,
             caption='Median over the 1,000 images; bands: interquartile range for untargeted ε 8 and noise ε 8 (embedding readouts). '
                     'Site-local embedding and joint coordinates: distance in units of the median distance between two clean images of '
                     'the same class at that site (log scale). Activation cosine: cosine between the centred attacked and clean '
                     'activations (1 = unchanged). Relative activation change: |a − a<sub>clean</sub>| / |a<sub>clean</sub> − μ|. '
                     'Faint dotted lines mark the stage boundaries. Read it for the site where the attack separates from noise of the same size.')

    t.h('F3.2 Class mass: when does the image start to look like the destination class?')
    t.figure('x3_mass', 'Site-local class mass of attacked images, against site', data='@x3', draw=DRAW_X3_MASS, height=640,
             caption='Mean kernel-row weight at each site on probe images of the source class (the true label), of the destination '
                     'class (the network\'s final prediction) and of all other classes; successful attacks only (counts in the panel '
                     'titles). Black dotted: weight on the own class for the clean images, the reference for "looks like its class". '
                     'Grey: source weight of the images on which the attack failed, where there are more than 20. With ten balanced '
                     'classes an uninformative row puts 0.1 on each class (0.8 on "other").')

    t.h('F3.3 Switch site')
    t.figure('x3_switch', 'Where the destination class overtakes the source class for good', data='@x3',
             controls=[Radio('swdef', 'Definition', ['switch', 'switch_top'],
                             labels={'switch': 'destination outweighs source', 'switch_top': 'destination is the top class'})],
             draw=DRAW_X3_SWITCH, height=420,
             caption='Share of each cohort (successful attacks; natural errors) whose switch site is at each site: the first site from '
                     'which the destination class outweighs the source class (or, second definition, is the heaviest class) at '
                     'every later site, in site-local class mass. "none": no persistent switch. A switch at the stem means the source '
                     'class never led after the stem, where class weights are near uniform. Hollow diamonds (seed 0 only): untargeted ε 8 with the k-NN kernel.')

    t.h('F3.4 Along the straight line from the clean image to the attack')
    t.figure('x3_path', 'Perturbation path x + s·δ, δ = the untargeted ε 8/255 perturbation', data='@x3',
             controls=[Radio('pz', 'Colour', ['disp_site', 'src_mass', 'relpert'],
                             labels={'disp_site': 'site-local displacement', 'src_mass': 'source-class mass', 'relpert': 'relative activation change'})],
             draw=DRAW_X3_PATH, height=430,
             caption='Left: each cell is the median (displacement, relative change) or mean (source-class mass) over the 1,000 images '
                     'at fraction s of the perturbation (rows) and one site (columns), in the chosen network. The displacement colour scale '
                     'runs from 0 to 3 in both networks (values above 3 would be capped; the true maximum is printed). Right: accuracy along the path. Read it for how the late sites '
                     'respond once s passes the point where the prediction flips.')

    t.h('F3.5 Arrival and support')
    t.figure('x3_arrive', 'Does the attacked image land where real members of its new class sit?', data='@x3',
             controls=[Radio('acoord', 'Coordinates for arrival', ['y', 'site_y'], labels={'y': 'joint', 'site_y': 'site-local'})],
             draw=DRAW_X3_ARRIVE, height=420,
             caption='Left: median arrival percentile, against site — the distance from a successfully attacked image to the centroid '
                     'of the destination class\'s probe images, as a percentile of clean members\' distances to their own class '
                     'centroid (0.5 = a typical member; 1 = farther than every member). Middle: its distribution at the logits. '
                     'Right: support percentile — the image\'s largest cosine to any probe image, as a percentile of the same quantity '
                     'for clean held-out images (0.5 = as close to the probe cloud as a clean image; low = off the cloud).')

    t.h('F3.6 Transfer to the other seed')
    t.figure('x3_transfer', f'Displacement in {B}, images the attack transfers to against images it does not', data='@x3',
             controls=[Radio('tcoord', 'Coordinates', ['disp_site_y', 'disp_y'], labels={'disp_site_y': 'site-local', 'disp_y': 'joint (retrospective)'})],
             draw=DRAW_X3_TRANSFER, height=420,
             caption=f'Left: median displacement in seed 1 (units of its own same-class distance, log scale) for images whose seed-1 '
                     f'prediction flips (solid) and does not (dashed), for every attack built on seed 0; a group is drawn only when it '
                     f'has more than 20 images; the split is by any flip, not by the same destination. Right: transfer rates. Read it for the site at '
                     f'which flipped and unflipped images separate.')

    t.h('F3.7 Pictures: attacked images in the embedding')
    labels = pic['ex']['label']
    t.figure('x3_scatter', f'Probe cloud and 20 clean → attacked arrows in {A}, site by site', data='@x3pic',
             controls=[Slider('site', 'Site', SITES, default='b9', play_ms=900),
                       Radio('pcoord', 'Coordinates', ['joint', 'site'], labels={'joint': 'joint (retrospective)', 'site': 'site-local'}),
                       Radio('patt', 'Attack', ['u8', 't8'], labels={'u8': 'untargeted ε 8', 't8': 'targeted ε 8'}),
                       Radio('pmodes', 'Modes', ['01', '02', '12'], labels={'01': '1–2', '02': '1–3', '12': '2–3'})],
             draw=DRAW_X3_SCATTER, height=560,
             caption='Small dots: the 2,000 probe images, coloured by class (legend). Each arrow runs from a clean scored '
                     'image (circle, its class colour) to its attacked version (number, coloured by the class seed 0 predicts for it); '
                     'the numbers match the thumbnails below. Joint coordinates share one space across sites and keep fixed axes, so '
                     'the slider shows motion through depth; site-local coordinates are a different basis at every site, so their '
                     'axes are set per site. Joint coordinates at a site depend on later sites too: read them as a picture, not as '
                     'a date. The arrows show attacked images only; random noise of the same size moves images about as far '
                     'through b6 (F3.1).')
    cards = []
    for i in range(len(labels)):
        e = pic['ex']
        cards.append(
            f'<div class="thumb"><div class="tn">{i + 1}</div>'
            f'<div class="ti"><img src="data:image/png;base64,{e["img_clean"][i]}" alt="clean"><span>clean<br><b>{CLASSES[e["label"][i]]}</b></span></div>'
            f'<div class="ti"><img src="data:image/png;base64,{e["img_u8"][i]}" alt="untargeted"><span>untargeted<br>→ <b>{CLASSES[e["pred_u8"][i]]}</b></span></div>'
            f'<div class="ti"><img src="data:image/png;base64,{e["img_t8"][i]}" alt="targeted"><span>targeted<br>→ <b>{CLASSES[e["target"][i]]}</b></span></div></div>')
    t.prose('<div class="thumbs">' + ''.join(cards) + '</div>'
            '<p class="note">The 20 example images (two per class), upscaled from 32 px: clean, untargeted ε 8/255 with the class '
            'seed 0 then predicts, and targeted ε 8/255 with its requested target (hit on every image).</p>')

    t.h('Method checks')
    lo = x3['logits_only']
    t.figure('x3_checks', 'Logits-only leak into joint coordinates, and extension gain', data='@x3', draw=DRAW_X3_CHECKS, height=360,
             caption=f'Left: replace only the logits row of each clean image by its attacked row (ε 8, seed 0) and recompute the joint '
                     f'coordinates; bars give the median change at each site as a share of the full attack\'s median displacement. '
                     f'Site-local readouts at earlier sites must not change: largest change {lo["local_max_change"]:g}. Right: smallest '
                     f'singular value of the 12 × 12 extension system per mode (the extension\'s gain is at most its inverse); the '
                     f'largest condition number over 16 modes is {max(x3["extension"][A]["cond"]):.0f} (seed 0) and '
                     f'{max(x3["extension"][B]["cond"]):.0f} (seed 1). The k-NN kernel check of the switch site is in F3.3.')


DRAW_X3_DISP = r"""(el, d, c) => {
  const q = c.dq, log = q.startsWith('disp'), tr = [], xs = SITES.map((_, k) => k);
  for (const k of Object.keys(d.att)) { const [lab, col, dash] = d.att[k], e = d.attacks[k][c.net];
    if (log && (k === 'u8' || k === 'n8')) { const b = e[q + '_q'];
      tr.push({x: xs.concat(xs.slice().reverse()), y: b[1].concat(b[0].slice().reverse()), fill: 'toself', mode: 'lines', fillcolor: col + '22',
        line: {width: 0}, hoverinfo: 'skip', showlegend: false}); }
    tr.push({x: xs, y: e[q], name: lab, mode: 'lines+markers', line: {color: col, dash, width: k[0] === 'n' ? 1.6 : 2.2},
      marker: {size: 5}, text: SITES, hovertemplate: lab + ' · %{text}: %{y:.3f}<extra></extra>'}); }
  const yt = {disp_site_y: 'displacement (same-class units, log)', disp_y: 'displacement (same-class units, log)',
    actcos: 'cosine to clean activation', relpert: '|a − a_clean| / |a_clean − μ|'}[q];
  Plotly.react(el, tr, LAYOUT({height: 440, shapes: stageLines(0), legend: {orientation: 'v', x: 1.01, y: 1},
    margin: {l: 64, r: 10, t: 20, b: 48}, xaxis: siteAxis(0, {title: {text: 'site'}}),
    yaxis: merge(AXIS, log ? {type: 'log', title: {text: yt}} : {title: {text: yt}, rangemode: 'tozero'})}), CONFIG);
}"""

DRAW_X3_MASS = r"""(el, d, c) => {
  const cm = d.class_mass[c.net], xs = SITES.map((_, k) => k), tr = [], layout = LAYOUT({height: 640, legend: {orientation: 'h', y: -0.09},
    margin: {l: 56, r: 14, t: 30, b: 70}}), shapes = [], ann = [];
  const cohorts = [['u8', 'untargeted ε 8/255'], ['t8', 'targeted ε 8/255'], ['u2', 'untargeted ε 2/255'], ['natural', 'natural errors']];
  const S = {src: ['source class', '#2f6f9f'], dst: ['destination class', '#c85a12'], other: ['other classes', '#a8a29a']};
  cohorts.forEach(([k, lab], i) => { const dm = dom4(i), e = k === 'natural' ? cm.natural : cm[k + '_success'];
    layout['xaxis' + ax(i)] = siteAxis(i, {domain: dm.x, showticklabels: i >= 2});
    layout['yaxis' + ax(i)] = merge(AXIS, {domain: dm.y, range: [0, 1], anchor: 'x' + ax(i), title: {text: i % 2 === 0 ? 'class mass' : ''}});
    shapes.push(...stageLines(i));
    ann.push(panelTitle(i, lab + (e ? ' · ' + (k === 'natural' ? '' : 'successes, ') + 'n = ' + e.n : ' · no successes')));
    if (!e) return;
    for (const s of ['src', 'dst', 'other']) tr.push({x: xs, y: e[s], name: S[s][0], legendgroup: s, showlegend: i === 0, mode: 'lines+markers',
      marker: {size: 4.5}, line: {color: S[s][1], width: 2.2}, xaxis: 'x' + ax(i), yaxis: 'y' + ax(i), text: SITES,
      hovertemplate: S[s][0] + ' · %{text}: %{y:.3f}<extra></extra>'});
    tr.push({x: xs, y: cm.clean.src, name: 'clean images: own class', legendgroup: 'clean', showlegend: i === 0, mode: 'lines',
      line: {color: '#2b2a27', dash: 'dot', width: 1.4}, xaxis: 'x' + ax(i), yaxis: 'y' + ax(i), hovertemplate: 'clean own class · %{y:.3f}<extra></extra>'});
    const f = cm[k + '_failure'];
    if (f) tr.push({x: xs, y: f.src, name: 'failed attacks: source class', legendgroup: 'fail', mode: 'lines', showlegend: true,
      line: {color: '#8d877c', dash: 'dash', width: 1.4}, xaxis: 'x' + ax(i), yaxis: 'y' + ax(i),
      hovertemplate: 'failed (n = ' + f.n + ') source · %{y:.3f}<extra></extra>'});
  });
  layout.shapes = shapes; layout.annotations = ann;
  Plotly.react(el, tr, layout, CONFIG);
}"""

DRAW_X3_SWITCH = r"""(el, d, c) => {
  const cm = d.class_mass[c.net], bins = SITES.concat(['none']), tr = [];
  const co = [['u8', 'untargeted ε 8', '#7f3408'], ['t8', 'targeted ε 8', '#6250d6'], ['u2', 'untargeted ε 2', '#e8833a'],
    ['u1', 'untargeted ε 1', '#f3b37c'], ['natural', 'natural errors', '#228b74']];
  for (const [k, lab, col] of co) { const e = k === 'natural' ? cm.natural : cm[k + '_success']; if (!e) continue;
    const h = e[c.swdef], n = h.reduce((a, b) => a + b, 0);
    tr.push({type: 'bar', x: bins, y: h.map(v => v / n), name: lab + ' (n = ' + n + ')', marker: {color: col},
      customdata: h, hovertemplate: lab + ' · %{x}: %{y:.1%} (%{customdata})<extra></extra>'}); }
  if (c.net === 'c10-s0') { const e = d.class_mass.knn15_u8_success_A, h = e[c.swdef], n = h.reduce((a, b) => a + b, 0);
    tr.push({type: 'scatter', mode: 'markers', x: bins, y: h.map(v => v / n), name: 'untargeted ε 8, k-NN kernel (check)',
      marker: {symbol: 'diamond-open', size: 10, color: '#2b2a27', line: {width: 1.6}}, hovertemplate: 'k-NN check · %{x}: %{y:.1%}<extra></extra>'}); }
  Plotly.react(el, tr, LAYOUT({height: 420, barmode: 'group', bargap: 0.18, bargroupgap: 0.05, legend: {orientation: 'h', y: 1.1},
    xaxis: merge(AXIS, {title: {text: 'switch site'}, type: 'category'}), yaxis: merge(AXIS, {title: {text: 'share of cohort'}, tickformat: '.0%'})}), CONFIG);
}"""

DRAW_X3_PATH = r"""(el, d, c) => {
  const P = d.path[c.net], z = P[c.pz], s = z.map((_, k) => k / 8), cap = c.pz === 'disp_site' ? 3 : null;
  let mx = 0; for (const r of z) for (const v of r) mx = Math.max(mx, v);
  const zz = cap ? z.map(r => r.map(v => Math.min(v, cap))) : z;
  const lab = {disp_site: 'displacement', src_mass: 'source-class mass', relpert: 'relative change'}[c.pz];
  const tr = [{type: 'heatmap', z: zz, x: SITES, y: s, customdata: z, zmin: 0, zmax: cap || (c.pz === 'src_mass' ? 1 : mx), zauto: false,
      colorscale: c.pz === 'src_mass' ? BLUES : [[0, '#fff7ec'], [0.3, '#fdbb84'], [0.65, '#e34a33'], [1, '#7f0000']],
      colorbar: {thickness: 10, len: 0.9, x: 0.66, title: {text: lab, side: 'right'}},
      hovertemplate: 's = %{y:.3f} · %{x}: %{customdata:.3f}<extra></extra>'},
    {x: P.acc, y: s, mode: 'lines+markers', xaxis: 'x2', yaxis: 'y2', line: {color: '#2b2a27', width: 2}, showlegend: false,
      hovertemplate: 's = %{y:.3f}: accuracy %{x:.1%}<extra></extra>'}];
  const ann = [panelTitle(0, lab + (cap ? (mx > cap ? ' (colour capped at ' + cap + '; max ' + mx.toFixed(2) + ')' : ' (colour scale 0 to ' + cap + ' in both networks; max ' + mx.toFixed(2) + ')') : '')), panelTitle(1, 'accuracy along the path')];
  Plotly.react(el, tr, LAYOUT({height: 430, margin: {l: 64, r: 14, t: 30, b: 48}, annotations: ann,
    xaxis: merge(AXIS, {domain: [0, 0.62], title: {text: 'site'}, showgrid: false}),
    yaxis: merge(AXIS, {title: {text: 's (fraction of the ε 8 perturbation)'}, tickvals: s, ticktext: s.map(v => v.toFixed(3).replace(/0+$/, '').replace(/\.$/, '')), showgrid: false}),
    xaxis2: merge(AXIS, {domain: [0.76, 1], anchor: 'y2', range: [0, 1.02], tickformat: '.0%', title: {text: 'accuracy (' + c.net + ')'}}),
    yaxis2: merge(AXIS, {anchor: 'x2', range: [-0.04, 1.04], tickvals: s.filter((_, k) => k % 2 === 0)})}), CONFIG);
}"""

DRAW_X3_ARRIVE = r"""(el, d, c) => {
  const A = d.arrival[c.net], xs = SITES.map((_, k) => k), tr = [];
  const co = [['u8', '#7f3408'], ['t8', '#6250d6'], ['u2', '#e8833a']];
  for (const [k, col] of co) { const e = A[k + '_' + c.acoord]; const lab = d.att[k][0];
    tr.push({x: xs, y: e.median, name: lab, legendgroup: k, mode: 'lines+markers', line: {color: col, width: 2.2}, marker: {size: 5},
      text: SITES, hovertemplate: lab + ' · %{text}: %{y:.2f}<extra></extra>'});
    const n = e.hist_logits.reduce((a, b) => a + b, 0);
    tr.push({type: 'bar', x: [...Array(10).keys()].map(b => b / 10 + 0.05), y: e.hist_logits.map(v => v / n), xaxis: 'x2', yaxis: 'y2',
      name: lab, legendgroup: k, showlegend: false, marker: {color: col}, opacity: 0.85,
      hovertemplate: lab + ' · percentile %{x:.2f}: %{y:.1%}<extra></extra>'}); }
  for (const k of ['u8', 't8', 'u2', 'n8', 'n2']) { const [lab, col, dash] = d.att[k];
    tr.push({x: xs, y: d.attacks[k][c.net].support, xaxis: 'x3', yaxis: 'y3', name: lab, legendgroup: k, showlegend: k[0] === 'n',
      mode: 'lines+markers', marker: {size: 4.5}, line: {color: col, dash, width: 2}, text: SITES, hovertemplate: lab + ' support · %{text}: %{y:.2f}<extra></extra>'}); }
  const half = i => ({type: 'line', xref: 'x' + ax(i) + ' domain', yref: 'y' + ax(i), x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: {color: '#2b2a27', dash: 'dot', width: 1}});
  Plotly.react(el, tr, LAYOUT({height: 420, barmode: 'group', legend: {orientation: 'h', y: 1.16}, margin: {l: 56, r: 14, t: 30, b: 50},
    shapes: [half(0), half(2), ...stageLines(0), ...stageLines(2)],
    annotations: [panelTitle(0, 'arrival percentile (' + (c.acoord === 'y' ? 'joint' : 'site-local') + ')'), panelTitle(1, 'arrival at the logits'), panelTitle(2, 'support percentile (site-local)')],
    xaxis: siteAxis(0, {domain: [0, 0.32]}), yaxis: merge(AXIS, {range: [0, 1], title: {text: 'percentile'}}),
    xaxis2: merge(AXIS, {domain: [0.39, 0.6], anchor: 'y2', range: [0, 1], title: {text: 'arrival percentile at logits'}}),
    yaxis2: merge(AXIS, {anchor: 'x2', tickformat: '.0%', title: {text: 'share of successes'}}),
    xaxis3: siteAxis(2, {domain: [0.68, 1]}), yaxis3: merge(AXIS, {anchor: 'x3', range: [0, 1]})}), CONFIG);
}"""

DRAW_X3_TRANSFER = r"""(el, d, c) => {
  const B = 'c10-s1', xs = SITES.map((_, k) => k), tr = [];
  for (const k of ['u1', 'u2', 'u4', 'u8', 't8']) { const [lab, col] = d.att[k], e = d.attacks[k][B];
    for (const [suf, dash, nm] of [['_flip', 'solid', 'transfers'], ['_noflip', 'dash', 'does not']]) { const y = e[c.tcoord + suf]; if (!y) continue;
      tr.push({x: xs, y, mode: 'lines', name: lab + ': ' + nm, legendgroup: k, line: {color: col, dash, width: 2}, text: SITES,
        hovertemplate: lab + ', ' + nm + ' · %{text}: %{y:.3f}<extra></extra>'}); } }
  const ks = ['u1', 'u2', 'u4', 'u8', 't8'], T = ks.map(k => d.attacks[k].transfer);
  tr.push({type: 'bar', x: ks.map(k => d.att[k][0].replace('untargeted ', 'unt. ').replace('targeted ', 'tgt. ')), y: T.map(t => t.any_flip), xaxis: 'x2', yaxis: 'y2',
    name: 'seed 1 flipped', marker: {color: '#8d877c'}, hovertemplate: '%{x}: flipped %{y:.1%}<extra></extra>'});
  tr.push({type: 'bar', x: ks.map(k => d.att[k][0].replace('untargeted ', 'unt. ').replace('targeted ', 'tgt. ')), y: T.map(t => t.same_destination), xaxis: 'x2', yaxis: 'y2',
    name: 'seed 1 same destination', marker: {color: '#2b2a27'}, hovertemplate: '%{x}: same destination %{y:.1%}<extra></extra>'});
  Plotly.react(el, tr, LAYOUT({height: 420, legend: {orientation: 'v', x: 1.01, y: 1, font: {size: 11}}, barmode: 'group', margin: {l: 60, r: 10, t: 30, b: 70},
    shapes: stageLines(0), annotations: [panelTitle(0, 'displacement in seed 1'), panelTitle(1, 'transfer rate (of seed-0 successes)')],
    xaxis: siteAxis(0, {domain: [0, 0.55]}), yaxis: merge(AXIS, {type: 'log', title: {text: 'displacement (same-class units, log)'}}),
    xaxis2: merge(AXIS, {domain: [0.64, 1], anchor: 'y2', tickangle: -30, tickfont: {size: 10}}),
    yaxis2: merge(AXIS, {anchor: 'x2', range: [0, 1], tickformat: '.0%'})}), CONFIG);
}"""

DRAW_X3_SCATTER = r"""(el, d, c) => {
  const l = SITES.indexOf(c.site), joint = c.pcoord === 'joint', [i, j] = [Number(c.pmodes[0]), Number(c.pmodes[1])];
  const cloud = joint ? d.probe_y : d.probe_site, ex = d.ex;
  const C = joint ? ex.clean_y : ex.clean_site, Z = joint ? ex[c.patt + '_y'] : ex[c.patt + '_site'];
  const dest = c.patt === 'u8' ? ex.pred_u8 : ex.target;
  // axis ranges: joint coordinates fixed over all sites, site-local per site
  const rng = (ls) => { let a = [1e9, -1e9], b = [1e9, -1e9];
    for (const q of ls) { for (const p of cloud[q]) { a = [Math.min(a[0], p[i]), Math.max(a[1], p[i])]; b = [Math.min(b[0], p[j]), Math.max(b[1], p[j])]; }
      for (const S of [C, Z]) for (const p of S[q]) { a = [Math.min(a[0], p[i]), Math.max(a[1], p[i])]; b = [Math.min(b[0], p[j]), Math.max(b[1], p[j])]; } }
    const pa = (a[1] - a[0]) * 0.05, pb = (b[1] - b[0]) * 0.05; return [[a[0] - pa, a[1] + pa], [b[0] - pb, b[1] + pb]]; };
  const key = (joint ? 'J' : 'S' + l) + c.pmodes;
  el._rng = el._rng || {}; if (!el._rng[key]) el._rng[key] = rng(joint ? SITES.map((_, q) => q) : [l]);
  const [rx, ry] = el._rng[key];
  const tr = [];
  for (let k = 0; k < 10; k++) { const pts = cloud[l].filter((_, n) => d.probe_labels[n] === k);
    tr.push({x: pts.map(p => p[i]), y: pts.map(p => p[j]), mode: 'markers', type: 'scattergl', name: CLASSES[k], legendgroup: 'c' + k,
      marker: {color: CLASSCOL[k], size: 3.5, opacity: 0.4}, hoverinfo: 'skip'}); }
  const ann = [];
  for (let n = 0; n < ex.idx.length; n++) { const a = C[l][n], b = Z[l][n];
    ann.push({x: b[i], y: b[j], ax: a[i], ay: a[j], xref: 'x', yref: 'y', axref: 'x', ayref: 'y', showarrow: true, arrowhead: 2,
      arrowsize: 1, arrowwidth: 1.3, arrowcolor: 'rgba(43,42,39,0.55)', text: ''}); }
  tr.push({x: ex.idx.map((_, n) => C[l][n][i]), y: ex.idx.map((_, n) => C[l][n][j]), mode: 'markers', name: 'clean scored image', showlegend: true,
    marker: {color: ex.label.map(k => CLASSCOL[k]), size: 9, line: {color: '#2b2a27', width: 1.2}},
    text: ex.idx.map((_, n) => (n + 1) + ': ' + CLASSES[ex.label[n]]), hovertemplate: '#%{text} (clean)<extra></extra>'});
  tr.push({x: ex.idx.map((_, n) => Z[l][n][i]), y: ex.idx.map((_, n) => Z[l][n][j]), mode: 'markers+text', name: 'attacked (number = thumbnail)',
    marker: {color: dest.map(k => CLASSCOL[k]), size: 13, symbol: 'square', line: {color: '#2b2a27', width: 1}},
    text: ex.idx.map((_, n) => String(n + 1)), textfont: {size: 8.5, color: '#111'},
    customdata: ex.idx.map((_, n) => CLASSES[ex.label[n]] + ' → ' + CLASSES[dest[n]]), hovertemplate: '#%{text}: %{customdata}<extra></extra>'});
  const name = joint ? 'joint mode ' : 'site-local mode ';
  Plotly.react(el, tr, LAYOUT({height: 560, annotations: ann, legend: {orientation: 'v', x: 1.01, y: 1, font: {size: 11}, itemsizing: 'constant'},
    margin: {l: 60, r: 10, t: 36, b: 48}, title: {text: 'site ' + c.site + ' · ' + (joint ? 'joint coordinates' : 'site-local coordinates'), font: {size: 13}, x: 0, xanchor: 'left'},
    xaxis: merge(AXIS, {range: rx, title: {text: name + (i + 1)}, zeroline: false}), yaxis: merge(AXIS, {range: ry, title: {text: name + (j + 1)}, zeroline: false})}), CONFIG);
}"""

DRAW_X3_CHECKS = r"""(el, d, c) => {
  const lo = d.logits_only, tr = [{type: 'bar', x: SITES, y: lo.ratio, marker: {color: '#7f3408'}, showlegend: false,
    hovertemplate: '%{x}: %{y:.1%} of the full displacement<extra></extra>'}];
  const ks = [...Array(16).keys()].map(k => k + 1);
  for (const [n, dash] of [['c10-s0', 'solid'], ['c10-s1', 'dash']]) tr.push({x: ks, y: d.extension[n].smin, xaxis: 'x2', yaxis: 'y2', mode: 'lines+markers',
    name: n, line: {color: GC('c10'), dash, width: 2}, marker: {size: 5}, hovertemplate: n + ' mode %{x}: %{y:.3f}<extra></extra>'});
  Plotly.react(el, tr, LAYOUT({height: 360, legend: {x: 0.9, y: 1, orientation: 'v'}, margin: {l: 60, r: 14, t: 30, b: 50},
    annotations: [panelTitle(0, 'logits-only change / full change, joint coordinates'), panelTitle(1, 'smallest singular value of the extension system')],
    xaxis: merge(AXIS, {domain: [0, 0.5], title: {text: 'site'}}), yaxis: merge(AXIS, {tickformat: '.0%', rangemode: 'tozero'}),
    xaxis2: merge(AXIS, {domain: [0.6, 1], anchor: 'y2', title: {text: 'joint mode'}}), yaxis2: merge(AXIS, {anchor: 'x2', rangemode: 'tozero'})}), CONFIG);
}"""


# ------------------------------------------------------------------------------------------------------- X4 --

ALLS = ['pixels'] + SITES
START_COL = {'pixels': '#0d0887', 'stem': '#6a00a8', 'b3': '#b12a90', 'b6': '#e16462', 'b8': '#e69500', 'b9': '#8d877c'}
KINDLAB = {'same': 'same-class pairs', 'diff': 'different-class pairs'}


def _med(v):
    v = [x for x in v if x is not None]
    return sorted(v)[len(v) // 2] if v else None


def x4_text(x4):
    """Data-driven sentences for the X4 takeaway (claims from FINDINGS.md, numbers from the summary)."""
    P = x4['per_start']
    st = list(x4['starts'])
    out = []
    sig = [v for l in st for k in ('same', 'diff') for v in P[l]['stats']['sigma_raw'][k][1] if v is not None]
    smin = [v for l in st for v in (P[l].get('sigma_min') or {}).values() if v is not None]
    nb9 = [l for l in st if l != 'b9']
    first = [P[l]['stats']['share_unit'][k][1][0] for l in nb9 for k in ('same', 'diff')]
    lg = [P[l]['stats']['share_unit'][k][1][-1] for l in st for k in ('same', 'diff')]
    fq = max(v for l in nb9 for k in ('same', 'diff') for v in P[l]['frac_share_gt_quarter'][k][:-1])
    rf = max(v for l in st for v in P[l]['refine']['share_unit_absdiff'] if v is not None)
    out.append('<b>On these sampled paths there is little localised motion in activation space.</b> The sharpness index (1 for a smooth '
               f'path, 0 for a jump) has site medians of {min(sig):.2f}–{max(sig):.2f}'
               + (f' and a minimum over all paths of {min(smin):.2f}' if smin else '') +
               f'. The fastest sixteenth of a path carries {pc(min(first))}–{pc(max(first))} of its angular arc length at the first site '
               f'after the start (6.25% is uniform) and {pc(min(lg))}–{pc(max(lg))} at the logits; at most {100 * fq:.1f}% of paths exceed '
               f'25% at any site before the logits. Refining 25 paths of each kind to 1,024 steps changes the window shares by at most {rf:.0e}.')
    early = [v for l in st for c_, v in zip(P[l]['chain'], P[l]['embedding']['mass_slope'][1]) if c_ not in ('b9', 'pool', 'logits')]
    late = [v for l in st for c_, v in zip(P[l]['chain'], P[l]['embedding']['mass_slope'][1]) if c_ in ('b9', 'pool', 'logits')]
    ss_e = [v for l in st for c_, v in zip(P[l]['chain'], P[l]['embedding']['site_share']['diff'][1]) if c_ not in ('b9', 'pool', 'logits')]
    ss_9 = [v for l in st for c_, v in zip(P[l]['chain'], P[l]['embedding']['site_share']['diff'][1]) if c_ == 'b9']
    t = ('<b>Read through the probe, the change of class along a path is much sharper from b9 on, but that is a property of the readout '
         'and the cloud, not a barrier in the network.</b> The largest slope of (kernel-row weight on b\'s class − weight on a\'s class) '
         f'along a different-class path is {min(early):.0f}–{max(early):.0f} per unit s at sites before b9 and {min(late):.0f}–{max(late):.0f} '
         f'at b9, pool and logits; in site-local coordinates the fastest sixteenth of the embedded path carries {pc(min(ss_e))}–{pc(max(ss_e))} '
         f'of its motion before b9 and {pc(min(ss_9))}–{pc(max(ss_9))} at b9.')
    if 'b9' in P:
        E9 = P['b9']['embedding']
        t += (f' The same sharpness appears for a path drawn straight at b9 itself (slope {E9["mass_slope"][1][0]:.1f}, share '
              f'{pc(E9["site_share"]["diff"][1][0])}), where the raw path is straight and everything downstream is affine.')
    pm = P.get('pixels', {}).get('embedding', {}).get('perp_mid')
    if pm:
        ch = P['pixels']['chain']
        t += (f' So it reflects how class-clustered the probe cloud is at b9, seen through an adaptive kernel whose row perplexity also drops '
              f'there ({pm["diff"][ch.index("b8")]:.0f} at b8, {pm["diff"][ch.index("b9")]:.0f} at b9, {pm["diff"][ch.index("logits")]:.0f} at the '
              'logits for path midpoints, pixels start); the experiment did not vary the two independently.')
    out.append(t)
    if 'pixels' in P:
        E = P['pixels']['embedding']
        sm, ssm = E['support_min']['diff'][1], [P[l]['embedding']['support_min']['same'][1][-1] for l in st]
        mid = E['support']['diff'][-1][32]
        out.append(f'<b>Different-class paths contain points far from every probe image, more so late.</b> The least-supported point of a '
                   f'path started in pixels has a nearest-probe cosine at the {th(sm[0])} percentile of clean images\' at the stem and '
                   f'the {th(sm[-2])} and {th(sm[-1])} at pool and logits (same-class paths: {th(min(ssm))}–'
                   f'{th(max(ssm))} at the logits, over start sites). This is a minimum over the path; at s = 0.5 exactly the median is '
                   f'the {th(mid)} percentile at the logits.')
    leave = {l: 1 - P[l]['switches']['same'][0] / sum(P[l]['switches']['same']) for l in st}
    third = {l: sum(P[l]['switches']['diff'][2:]) / sum(P[l]['switches']['diff']) for l in st}
    out.append('<b>Sampled same-class chords stop leaving the class by b8.</b> A straight path between two images of one class leaves the '
               'class at the output for ' + ', '.join(f'{pc(leave[l])} ({l})' for l in st if l != 'b9') + ' of pairs (from b9 it cannot: the '
               'suffix is affine and the class regions of the logits are convex). Different-class paths change prediction more than once '
               '(with two changes, through a third class) for ' + ', '.join(f'{pc(third[l])} ({l})' for l in st) + '.')
    A3 = P.get('b3', {}).get('aligned', {})
    if A3.get('site_peak_rho_all') and A3.get('site_peak_rho'):
        ra, r1 = dict(zip(P['b3']['sites'], A3['site_peak_rho_all'])), dict(zip(P['b3']['sites'], A3['site_peak_rho']))
        t = (f'<b>Where a path moves fastest is associated with where the output switches, more strongly at later sites.</b> For all '
             f'{A3["n_all"]} different-class paths started at b3 (aligned on the first switch), the rank correlation between the location of '
             f'peak angular speed at a site and the location of the first output switch is {f2(ra["b4"])} at b4, {f2(ra["b7"])} at b7, '
             f'{f2(ra["b9"])} at b9 and {f2(ra["logits"])} at the logits ({f2(r1["b4"])}, {f2(r1["b7"])}, {f2(r1["b9"])}, {f2(r1["logits"])} on the '
             f'{A3["n"]} single-switch paths).')
        if 'b9' in P and P['b9']['aligned'].get('site_peak_rho'):
            t += (f' Part of this comes from the output normalisation: on the affine b9 control, where raw speed is constant, the angular '
                  f'speed peak at the logits still correlates {f2(P["b9"]["aligned"]["site_peak_rho"][-1])} with the switch (single-switch '
                  'paths), because both sit where the centred logits are smallest.')
        out.append(t)
    g = []
    for l in ('pixels', 'b3', 'b6', 'b8'):
        if l in P:
            v = P[l]['gain']['logits_raw']['diff']
            g.append(f'{l} {f2(v[32])} against {f2((v[0] + v[-1]) / 2)}')
    t = ('<b>Log gain.</b> The raw log gain to the logits (log of the largest singular value of the Jacobian), middle of a different-class '
         'path against the mean of its ends: ' + '; '.join(g) + ' (higher in the middle up to b6, slightly lower from b8, constant from b9). '
         'The interpolation direction has small gain relative to the maximum (ratio ' +
         ', '.join(f'{f2(P[l]["gain"]["align_raw"]["diff"][32])} from {l}' for l in ('pixels', 'b3', 'b6', 'b8') if l in P) + ').')
    pg = x4.get('probe_gain')
    if pg:
        rho, rr = pg['rho'], pg.get('rho_ang_radius_removed', {})
        t += (f' Over the 2,000 probe images, angular log gain to the logits is anti-correlated with the output margin (Spearman '
              f'{f2(rho["pixels>logits_ang"])} from pixels, {f2(rho["stem>logits_ang"])} from the stem, {f2(rho["b3>logits_ang"])} from b3; '
              f'misclassified images {f2(pg["by_correct"]["pixels>logits_ang"][1])} against {f2(pg["by_correct"]["pixels>logits_ang"][0])} '
              f'for correct ones from pixels)')
        if rr:
            t += (f', but <b>most of that comes from the output normalisation</b>: the centred logit radius itself correlates '
                  f'{pg["rho_radius"]["pixels"]:+.2f} with margin, and with that factor removed the correlations are {f2(rr["pixels"])}, '
                  f'{f2(rr["stem"])}, {f2(rr["b3"])}')
        t += f'. Raw gain correlates {f2(rho["pixels>logits_raw"])}. So high gain marks low-margin images weakly, not sharply.'
    out.append(t)
    if 'pixels' in P and 'taps' in P['pixels']:
        taps, fd, fs = P['pixels']['taps'], P['pixels']['relu_from0']['diff'], P['pixels']['relu_from0']['same']
        end = {tp: (fs[i][-1], fd[i][-1]) for i, tp in enumerate(taps)}
        out.append(f'<b>Block-output ReLU masks</b> differ between the two endpoints on {pc(min(end["b3"]))} of units at b3, rising to '
                   f'{pc(end["b9"][0])} (same class) and {pc(end["b9"][1])} (different class) at b9. The ReLU inside each block is not counted, '
                   'and these are mask differences, not counts of linear regions crossed.')
    return out


def x4_payload(x4):
    P = {}
    for l, R in x4['per_start'].items():
        R = dict(R)
        R['picture'] = {k: v for k, v in R['picture'].items() if k != 'img'}
        P[l] = R
    pg = x4.get('probe_gain')
    return dict(starts=x4['starts'], per_start=P, probe_gain=pg, checks=x4.get('checks'), start_col=START_COL, alls=ALLS)


def tab_x4(nb, x4):
    t = nb.tab('x4', 'X4 Interpolation and log gain')
    nb.dataset('x4', clean(x4_payload(x4)))
    P = x4['per_start']
    st = x4['starts']
    n = P[st[0]]['n']
    t.prose(
        '<p><b>Question.</b> Take two real images, read their activations at a <b>start site</b>, walk the straight line between the two '
        'activations and run every point through the rest of the network. Where downstream does the path stretch, where does class '
        'membership change, and how much does the network stretch nearby directions? <b>Setup.</b> Network <code>c10-s0</code> '
        f'(CIFAR-10, seed 0). {n[0]} same-class and {n[1]} different-class pairs of held-out, correctly classified test images, the same pairs '
        'for every start site (' + ', '.join(st) + '). Path $x(s) = (1-s)\\,a + s\\,b$ at the start site, $s \\in [0,1]$, 257 points '
        '(a 1,025-point refinement on 25 pairs of each kind). The b9 start is an exact control: after b9 come average pooling and a '
        'linear head, so the path stays straight.</p>'
        '<p><b>Terms.</b> The network is continuous and piecewise affine, so nothing jumps; the outcome is <b>localised stretching</b>. '
        '<b>Window share</b>: the share of the path\'s arc length at a site that lies in its fastest window of width 1/16 in $s$ '
        '(1/16 = 6.25% for a uniform path, 1 for a step); computed on the <b>angular</b> path (the centred, normalised activation the '
        'embedding sees) unless stated. <b>Sharpness index</b> σ = log(m<sub>64</sub>/m<sub>256</sub>)/log 4 from the largest step on '
        'two grids (1 smooth, 0 a step). <b>Relative speed</b>: |dx/ds| divided by the chord |x(1) − x(0)| at that site (1 everywhere '
        'for uniform motion). <b>Log gain</b>: log of the largest singular value of the Jacobian from the start site to a target site; '
        '<b>FTLE</b> = log gain divided by the number of sites crossed; <b>angular</b> gain is the fixed-radius derivative between the '
        'unit spheres the embedding uses. <b>Suffix embedding</b>: a joint operator fitted on the clean probe over the sites from the '
        'start site to the logits (smooth kernel), into which path points are extended; it is not the public viewer\'s coordinates. '
        '<b>Class mass</b> and <b>support</b> are as on the X3 tab, read from the path point\'s kernel row at each site of that chain.</p>'
        '<p class="note">Colours on this tab code the start site (dark = early); grey is the b9 control. Solid lines: different-class '
        'pairs; dashed: same-class pairs.</p>')
    t.takeaway('<ul>' + ''.join(f'<li>{s}</li>' for s in x4_text(x4)) + '</ul>', label='What the figures show (numbers from the summary)')

    hl = Select('hl4', 'Bands for', st, default='b3' if 'b3' in st else st[0])
    startsel = lambda d='b3': Radio('st4', 'Start site', st, default=d if d in st else st[0])  # noqa: E731
    kindsel = Radio('k4', 'Pairs', ['diff', 'same'], labels={'diff': 'different class', 'same': 'same class'})

    t.h('F4.1 Localised stretching, downstream of each start site')
    t.figure('x4_stretch', 'How concentrated the motion is at each downstream site', data='@x4',
             controls=[Select('f1stat', 'Statistic', ['share_unit', 'share_raw', 'sigma_raw', 'relpeak_raw', 'site_share'],
                              labels={'share_unit': 'window share, angular (primary)', 'share_raw': 'window share, raw activations',
                                      'sigma_raw': 'sharpness index σ (raw)', 'relpeak_raw': 'peak speed / chord (raw)',
                                      'site_share': 'window share in site-local embedding coordinates'}), hl],
             draw=DRAW_X4_STRETCH, height=460,
             caption='Median over pairs; band: interquartile range for the start site chosen in "Bands for". Dotted line: the value for '
                     'uniform motion (1/16 for window shares, 1 for σ and for peak speed / chord). The site-local embedding option reads the '
                     'path in each site\'s own walk coordinates (suffix chain, so it includes the start site itself). The b9 start is the '
                     'affine control: its raw path stays uniform at pool and logits, and its angular share rises only from passing near the '
                     'probe mean.')
    rows = []
    for l in st:
        for i, sr in enumerate(P[l]['strata']):
            sh = sr['share_unit']
            rows.append([l, f'{i + 1}', f'{sr["range"][0]:.3g}–{sr["range"][1]:.3g}', f'{sr["n"][0]} / {sr["n"][1]}',
                         f2(sh['same'][-1]) if sh['same'] else '—', f2(sh['diff'][-1]) if sh['diff'] else '—'])
    t.details('Same- against different-class pairs, within terciles of endpoint distance at the start site',
              '<p class="note">Different-class endpoints are farther apart, so the comparison is repeated within terciles of the chord '
              '|b − a| at the start site (activation units). Window share (angular) at the logits, median.</p>' +
              table_html(['Start', 'Tercile', 'Chord range', 'n same / diff', 'same-class', 'different-class'], rows, {3, 4, 5}))

    t.h('Class membership along the path, read through the probe')
    t.figure('x4_mass', 'Kernel-row weight on the two endpoint classes along different-class paths', data='@x4',
             controls=[startsel()], draw=DRAW_X4_MASS, height=470,
             caption='Left and middle: mean weight of a path point\'s kernel row (smooth kernel, suffix chain) on probe images of class a '
                     '(the s = 0 image\'s class) and of class b, over s (x) and chain site (y), for different-class pairs from the chosen '
                     'start site. Right: largest slope along s of (weight on b − weight on a), per unit s, median over pairs with an '
                     'interquartile band for the chosen start; one line per start site. A gradual change gives a slope near 1; an abrupt '
                     'change a large one. The b9 start (grey) is the affine control: its raw path stays straight downstream, yet its '
                     'slope at b9 is comparable to that of paths from earlier starts, so the sharp change from b9 on is a property of the '
                     'kernel readout and the class-clustered probe cloud there, not a barrier in the network\'s map.')

    t.h('F4.2 Speed along the path')
    exs = ['mean'] + [str(i + 1) for i in range(len(P[st[0]]['picture']['kind']))]
    t.figure('x4_speed', 'Relative speed over s and downstream site', data='@x4',
             controls=[startsel(), kindsel, Radio('spv', 'Activation', ['unit', 'raw'], labels={'unit': 'angular', 'raw': 'raw'}),
                       Select('spex', 'Show', exs, labels={'mean': 'mean over pairs', **{e: f'example {e}' for e in exs[1:]}})],
             draw=DRAW_X4_SPEED, height=430,
             caption='Each cell is the speed |dx/ds| at one downstream site (rows) and one s (columns), divided by that site\'s chord '
                     '|x(1) − x(0)|: 1 is uniform motion. "Mean over pairs" averages the chosen kind; an example shows one pair (angular '
                     'speed; numbers match the thumbnails under F4.5). Colour scale fixed at 0 to 4 for every start (maximum printed).')

    t.h('Support along the path')
    t.figure('x4_support', 'Off-cloud travel: how close path points stay to the probe images', data='@x4',
             controls=[startsel(), kindsel], draw=DRAW_X4_SUPPORT, height=420,
             caption='Support percentile: a path point\'s largest cosine to any probe image, as a percentile of the same quantity for 1,000 '
                     'clean held-out images (0.5 = as close as a typical clean image; near 0 = off the cloud). Left: the least-supported '
                     'point of each path, median over pairs, against site, per start site. Right: median support over s and chain site '
                     'for the chosen start and pair kind.')

    t.h('Straight-line connectedness of classes')
    t.figure('x4_switch', 'How often the output changes along the path', data='@x4', draw=DRAW_X4_SWITCH, height=380,
             caption='Left: share of same-class paths whose prediction leaves the class somewhere along the path, split by the number of '
                     'prediction changes. Right: share of different-class paths that change prediction more than once; with two changes '
                     'the path must pass through a third class, with three or more it may instead go back and forth between a and b. '
                     'Paths drawn from a later start site are paths in a different space, so the bars compare straight lines at different '
                     'depths, not one path.')

    t.h('F4.3 Log gain along the path')
    t.figure('x4_gain', 'Log gain along the path, aligned on the output switch', data='@x4',
             controls=[Radio('gtg', 'Target', ['logits', 'b9']), Radio('gv', 'Gain', ['ang', 'raw'], labels={'ang': 'angular', 'raw': 'raw'}),
                       Radio('gc', 'Display', ['rel', 'abs'], labels={'rel': 'minus the line\'s minimum', 'abs': 'absolute'})],
             draw=DRAW_X4_GAIN, height=720,
             caption='(a) Median log gain from the start site to the target against s − s<sub>switch</sub>, for different-class paths with '
                     'exactly one output switch (n per start shown on hover). (b) Same-class paths, most of which have no switch, '
                     'against s (unaligned). (c) Alignment |Jv| / (σ<sub>max</sub>|v|) of the path direction v = b − a with the most-stretched '
                     'direction, target logits, different-class pairs: 1 = the path runs along the direction of largest gain. (d) Rank '
                     'correlation between the s of peak angular speed at each downstream site and the s of the first output switch: '
                     'solid, all different-class paths with a switch; dotted, the single-switch subset (stronger). Part of the late '
                     'association comes from the output normalisation: the b9 control, whose raw speed is constant, shows it too. With '
                     '"minus the line\'s minimum" each line is shifted to start at 0 so start sites with different levels share one axis. '
                     'The b9 start has only the logits target. With angular gain, (a) still rises at the switch for the b8 start, although the '
                     'median raw gain of different-class paths is not higher mid-path from b8 on (takeaway above).')

    pg = x4.get('probe_gain')
    if pg:
        t.h('F4.4 Log gain at the probe images')
        t.figure('x4_probe', 'Angular log gain to the logits against output margin, 2,000 probe images', data='@x4',
                 controls=[Radio('pgs', 'Start site', pg['starts'], default='b3' if 'b3' in pg['starts'] else pg['starts'][0]),
                           Radio('pgv', 'Gain', ['ang', 'raw'], labels={'ang': 'angular', 'raw': 'raw'}),
                           Radio('pgf', 'Scale', ['gain', 'ftle'], labels={'gain': 'log gain', 'ftle': 'FTLE (÷ sites crossed)'}),
                           Radio('pge', 'Embedding site', list(pg['probe_y']), default='b9' if 'b9' in pg['probe_y'] else list(pg['probe_y'])[0])],
                 draw=DRAW_X4_PROBE, height=470,
                 caption='Left: each dot is a probe image (grey: classified correctly; red: misclassified), log gain from the start site to '
                         'the logits against the output margin (top logit − second logit); black line: medians within margin deciles. '
                         'Spearman correlation in the title. Right: the probe in the suffix embedding of the stem start (operator fitted '
                         'on stem to logits), joint modes 1 and 2 at the chosen site, coloured by the same gain (colour clipped at the 2nd '
                         'and 98th percentiles). FTLE divides by the number of sites crossed, so start sites can be compared. Most of the '
                         'angular correlation comes from the output normalisation: the centred logit radius itself rises with margin, and '
                         'adding back its log (the panel title gives the Spearman correlation then) leaves a weak association.')

    t.h('F4.5 Example paths in the suffix embedding')
    ex_sites = [s for s in ALLS if s != 'pixels']
    t.figure('x4_emb', 'Ten example paths drawn in the suffix embedding, site by site', data='@x4',
             controls=[startsel(), Slider('esite', 'Site', ex_sites, default='b9', play_ms=900),
                       Radio('emodes', 'Modes', ['01', '02', '12'], labels={'01': '1–2', '02': '1–3', '12': '2–3'}),
                       Select('eex', 'Paths', ['all'] + exs[1:], labels={'all': 'all ten', **{e: f'example {e}' for e in exs[1:]}})],
             draw=DRAW_X4_EMB, height=560,
             caption='These are suffix embeddings: for each start site, a joint operator is fitted on the clean probe over the sites from '
                     'the start site to the logits (for the pixels start, stem to logits), and the path points are extended into it. They are '
                     'not the public viewer\'s coordinates, and the spaces of different start sites are different. Small dots: 1,000 probe '
                     'images, coloured by class. Each path runs from image a (circle) to image b (square), points coloured by s; numbers '
                     'match the thumbnails below. Axes are fixed across sites for one start site, so the slider shows the motion through '
                     'depth.')
    pic = P[st[0]]['picture']
    cards = []
    imgs = x4['per_start'][st[0]]['picture']['img']
    for i in range(len(pic['kind'])):
        la, lb = pic['labels'][i]
        cards.append(f'<div class="thumb"><div class="tn">{i + 1}</div>'
                     f'<div class="ti"><img src="data:image/png;base64,{imgs[i][0]}" alt="a"><span>a: <b>{CLASSES[la]}</b></span></div>'
                     f'<div class="ti"><img src="data:image/png;base64,{imgs[i][1]}" alt="b"><span>b: <b>{CLASSES[lb]}</b></span></div>'
                     f'<div class="ti"><span>{"same class" if pic["kind"][i] == 0 else "different class"}</span></div></div>')
    t.prose('<div class="thumbs">' + ''.join(cards) + '</div><p class="note">The ten example pairs (the same pairs for every start '
            'site): image a at s = 0, image b at s = 1.</p>')

    if any('taps' in P[l] for l in st):
        t.h('F4.6 Tap-output ReLU mask differences')
        t.figure('x4_relu', 'Tap-output ReLU mask differences along the path', data='@x4', controls=[startsel(), kindsel],
                 draw=DRAW_X4_RELU, height=400,
                 caption='The taps are block outputs, which are post-ReLU, so a unit\'s mask is "activation > 0". These counts cover the '
                         'block-output masks only, not the ReLU inside each block, and they are not a count of linear-region crossings. '
                         'Left: fraction of units whose mask differs between the two endpoint images, per tap (the same for every start site). '
                         'Middle: fraction differing from s = 0 along s, one line per tap, for the chosen start and pair kind. Right: total '
                         'mask changes along the path (sum over steps of the fraction of units changing, median over pairs), per start site.'
                         + relu_dip_note(P))

    t.h('F4.7 Controls')
    t.figure('x4_ctrl', 'Exact against finite-difference speed', data='@x4', draw=DRAW_X4_CTRL, height=360,
             caption='Median relative gap between the exact speed (forward-mode derivative) and the finite difference of the stored path, '
                     'per downstream site, on the 257-point grid (solid) and on the 1,025-point refinement (dashed). The gap shrinks with '
                     'the step: the path is rough at the scale of a step (ReLU cells are crossed inside a step), not wrong.')
    t.table(['Check', 'Value', 'Expected'], x4_checks_rows(x4))


def relu_dip_note(P):
    """Observation: the endpoint mask difference dips at the b3 tap (end of the first stage)."""
    Q = next((P[l] for l in P if 'taps' in P[l] and 'b3' in P[l]['taps']), None)
    if not Q or not all(t in Q['taps'] for t in ('b2', 'b4')):
        return ''
    e = {t: Q['relu_from0']['diff'][Q['taps'].index(t)][-1] for t in ('b2', 'b3', 'b4')}
    if not (e['b3'] < e['b2'] and e['b3'] < e['b4']):
        return ''
    return (f' Observation, not explained here: the endpoint mask difference dips at the b3 tap, the last block of the first stage '
            f'({pc(e["b2"])} at b2, {pc(e["b3"])} at b3, {pc(e["b4"])} at b4, different-class pairs).')


def x4_checks_rows(x4):
    P, ck = x4['per_start'], x4.get('checks') or {}
    rows = []
    if ck:
        rows.append(['Gain pool → logits: exact against closed form (max abs. error, log)', f'{ck["pool_logits_exact_err"]:.1e}', '≈ 0'])
        rows.append(['Gain b9 → logits: exact against closed form', f'{ck["b9_logits_exact_err"]:.1e}', '≈ 0'])
        rows.append(['Gain b6 → logits: exact against explicit SVD (max abs., log)', f'{ck["b6_logits_exact_vs_svd_maxabs_log"]:.1e}', '≈ 0'])
        rows.append(['Gain b6 → logits: Ritz estimate against exact / plain power iteration against exact',
                     f'{ck["b6_logits_vs_exact_maxabs_log"]["ritz"]:.1e} / {ck["b6_logits_vs_exact_maxabs_log"]["pow"]:.1e}', 'small / larger'])
        ba = ck['b9_affine']
        rows.append(['b9 start, raw path: |16 × window share − 1|, |σ − 1| (max)', f'{ba["share_x16_minus1_maxabs"]:.1e}, {ba["sigma_minus1_maxabs"]:.1e}', '≈ 0 (affine)'])
        rows.append(['Suffix operator against the common one (max abs.)', f'{ck["suffix_vs_common_maxabs"]:.1e}', '0'])
        fl = ck.get('relu_tap_flips_f32_vs_f64', {})
        rows.append(['Tap ReLU masks flipped between float32 and float64 passes', str(sum(fl.values())) + ' unit(s)', '≈ 0'])
    for l in x4['starts']:
        e = P[l]['embedding']['errors']
        rf = P[l]['refine']
        rows.append([f'Start {l}: masked replay of the probe / path endpoint against the image itself (max abs.)',
                     f'{e["replay"]:.1e} / {e["endpoint"]:.1e}', '≈ 0'])
        rows.append([f'Start {l}: largest extension condition number (16 modes); window share change at 1,025 points (max over sites)',
                     f'{e["cond_max"]:.1f}; {max(v for v in rf["share_unit_absdiff"] if v is not None):.1e}', 'moderate; ≈ 0'])
    return rows


DRAW_X4_STRETCH = r"""(el, d, c) => {
  const st = c.f1stat, tr = [], xi = s => d.alls.indexOf(s);
  for (const l0 of d.starts) { const P = d.per_start[l0], emb = st === 'site_share', sites = emb ? P.chain : P.sites, xs = sites.map(xi);
    const col = d.start_col[l0], band = c.hl4 === l0;
    for (const k of ['same', 'diff']) { const q = emb ? P.embedding.site_share[k] : P.stats[st][k];
      if (band) tr.push({x: xs.concat(xs.slice().reverse()), y: q[2].concat(q[0].slice().reverse()), fill: 'toself', mode: 'lines', fillcolor: col + (k === 'diff' ? '30' : '18'),
        line: {width: 0}, hoverinfo: 'skip', showlegend: false});
      tr.push({x: xs, y: q[1], mode: 'lines+markers', name: l0, legendgroup: l0, showlegend: false, marker: {size: 5},
        line: {color: col, width: band ? 2.6 : 1.8, dash: k === 'same' ? 'dash' : 'solid'}, text: sites,
        hovertemplate: 'start ' + l0 + ', ' + k + ' · %{text}: %{y:.3f}<extra></extra>'}); } }
  for (const l0 of d.starts) tr.push(legendOnly('start ' + l0, d.start_col[l0]));
  tr.push(legendOnly('different class', '#2b2a27', 'solid'), legendOnly('same class', '#2b2a27', 'dash'));
  const ref = st.startsWith('share') || st === 'site_share' ? 1 / 16 : 1;
  const yt = {share_unit: 'window share (angular)', share_raw: 'window share (raw)', sigma_raw: 'sharpness index σ', relpeak_raw: 'peak speed / chord',
    site_share: 'window share (site-local embedding)'}[st];
  const yr = st === 'sigma_raw' ? [0, 1.05] : null;
  Plotly.react(el, tr, LAYOUT({height: 460, legend: {orientation: 'h', y: 1.14, x: 0, traceorder: 'normal'}, margin: {l: 60, r: 14, t: 50, b: 48},
    shapes: [{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: ref, y1: ref, line: {color: '#2b2a27', dash: 'dot', width: 1}}],
    xaxis: merge(AXIS, {tickvals: d.alls.map((_, k) => k), ticktext: d.alls, range: [-0.4, d.alls.length - 0.6], title: {text: 'downstream site'}}),
    yaxis: merge(AXIS, yr ? {range: yr, title: {text: yt}} : {title: {text: yt}, rangemode: 'tozero'})}), CONFIG);
}"""

DRAW_X4_MASS = r"""(el, d, c) => {
  const P = d.per_start[c.st4], E = P.embedding, s = [...Array(65).keys()].map(k => k / 64), xi = q => d.alls.indexOf(q);
  const hm = (z, i, name, show) => ({type: 'heatmap', z, x: s, y: P.chain, zmin: 0, zmax: 1, zauto: false, colorscale: BLUES, showscale: show,
    colorbar: {thickness: 10, len: 0.9, x: 0.555, xanchor: 'left', title: {text: 'weight', side: 'top'}}, xaxis: 'x' + ax(i), yaxis: 'y' + ax(i),
    hovertemplate: name + ' · %{y}, s = %{x:.3f}: %{z:.3f}<extra></extra>'});
  const tr = [hm(E.mass_a, 0, 'class a', false), hm(E.mass_b, 1, 'class b', true)];
  for (const l0 of d.starts) { const Q = d.per_start[l0], q = Q.embedding.mass_slope, xs = Q.chain.map(xi), col = d.start_col[l0];
    if (l0 === c.st4) tr.push({x: xs.concat(xs.slice().reverse()), y: q[2].concat(q[0].slice().reverse()), xaxis: 'x3', yaxis: 'y3', fill: 'toself', mode: 'lines',
      fillcolor: col + '26', line: {width: 0}, hoverinfo: 'skip', showlegend: false});
    tr.push({x: xs, y: q[1], xaxis: 'x3', yaxis: 'y3', mode: 'lines+markers', name: 'start ' + l0, line: {color: col, width: l0 === c.st4 ? 2.6 : 1.6},
      marker: {size: 5}, text: Q.chain, hovertemplate: 'start ' + l0 + ' · %{text}: slope %{y:.2f}<extra></extra>'}); }
  Plotly.react(el, tr, LAYOUT({height: 470, legend: {orientation: 'h', x: 0, y: -0.14, font: {size: 10.5}},
    margin: {l: 60, r: 14, t: 30, b: 80},
    annotations: [panelTitle(0, 'weight on class a (start ' + c.st4 + ')'), panelTitle(1, 'weight on class b'), panelTitle(2, 'largest slope of the transition')],
    xaxis: merge(AXIS, {domain: [0, 0.25], title: {text: 's'}, showgrid: false}), yaxis: merge(AXIS, {title: {text: 'chain site'}, showgrid: false}),
    xaxis2: merge(AXIS, {domain: [0.29, 0.54], anchor: 'y2', title: {text: 's'}, showgrid: false}),
    yaxis2: merge(AXIS, {anchor: 'x2', showticklabels: false, showgrid: false, matches: 'y'}),
    xaxis3: merge(AXIS, {domain: [0.7, 1], anchor: 'y3', tickvals: d.alls.map((_, k) => k), ticktext: d.alls, tickfont: {size: 9.5}, range: [0.6, d.alls.length - 0.6]}),
    yaxis3: merge(AXIS, {anchor: 'x3', rangemode: 'tozero', title: {text: 'slope per unit s'}})}), CONFIG);
}"""

DRAW_X4_SPEED = r"""(el, d, c) => {
  const P = d.per_start[c.st4], s = [...Array(65).keys()].map(k => k / 64);
  let z, lab;
  if (c.spex === 'mean') { z = (c.spv === 'unit' ? P.speed_unit_rel : P.speed_rel)[c.k4];
    lab = 'mean ' + (c.spv === 'unit' ? 'angular' : 'raw') + ' relative speed, ' + (c.k4 === 'diff' ? 'different-class' : 'same-class') + ' pairs'; }
  else { const e = Number(c.spex) - 1; z = P.picture.speed_unit[e];
    const kk = P.picture.kind[e] ? 'different class' : 'same class'; lab = 'example ' + c.spex + ' (' + kk + ', angular)'; }
  let mx = 0; for (const r of z) for (const v of r) if (v != null) mx = Math.max(mx, v);
  const cap = 4, zz = z.map(r => r.map(v => v == null ? null : Math.min(v, cap)));
  const tr = [{type: 'heatmap', z: zz, customdata: z, x: s, y: P.sites, zmin: 0, zmax: cap, zauto: false,
    colorscale: [[0, '#ffffff'], [0.25, '#fdd49e'], [0.5, '#fc8d59'], [0.75, '#d7301f'], [1, '#7f0000']],
    colorbar: {thickness: 10, len: 0.9, tickvals: [0, 1, 2, 3, 4], ticktext: ['0', '1 (uniform)', '2', '3', '≥4'], title: {text: 'speed / chord', side: 'right'}},
    hovertemplate: '%{y}, s = %{x:.3f}: %{customdata:.2f}<extra></extra>'}];
  Plotly.react(el, tr, LAYOUT({height: 430, margin: {l: 60, r: 14, t: 34, b: 50},
    annotations: [panelTitle(0, 'start ' + c.st4 + ' · ' + lab + ' · max ' + mx.toFixed(2) + (mx > cap ? ' (colour capped at 4)' : ''))],
    xaxis: merge(AXIS, {title: {text: 's'}, showgrid: false}), yaxis: merge(AXIS, {title: {text: 'downstream site'}, showgrid: false})}), CONFIG);
}"""

DRAW_X4_SUPPORT = r"""(el, d, c) => {
  const xi = q => d.alls.indexOf(q), tr = [], s = [...Array(65).keys()].map(k => k / 64);
  for (const l0 of d.starts) { const Q = d.per_start[l0], xs = Q.chain.map(xi), col = d.start_col[l0];
    for (const k of ['diff', 'same']) { const q = Q.embedding.support_min[k];
      tr.push({x: xs, y: q[1], mode: 'lines+markers', name: 'start ' + l0, legendgroup: l0, showlegend: k === 'diff',
        line: {color: col, dash: k === 'same' ? 'dash' : 'solid', width: l0 === c.st4 ? 2.6 : 1.6}, marker: {size: 4.5}, text: Q.chain,
        hovertemplate: 'start ' + l0 + ', ' + k + ' · %{text}: %{y:.3f}<extra></extra>'}); } }
  const P = d.per_start[c.st4];
  tr.push({type: 'heatmap', z: P.embedding.support[c.k4], x: s, y: P.chain, xaxis: 'x2', yaxis: 'y2', zmin: 0, zmax: 1, zauto: false,
    colorscale: [[0, '#7f0000'], [0.1, '#d7301f'], [0.25, '#fc8d59'], [0.5, '#fdd49e'], [1, '#fff7ec']],
    colorbar: {thickness: 10, len: 0.9, title: {text: 'support pct.', side: 'right'}}, hovertemplate: '%{y}, s = %{x:.3f}: %{z:.3f}<extra></extra>'});
  Plotly.react(el, tr, LAYOUT({height: 420, legend: {orientation: 'h', y: 1.16, font: {size: 11}}, margin: {l: 60, r: 14, t: 34, b: 50},
    annotations: [panelTitle(0, 'least-supported point of the path (median)'), panelTitle(1, 'median support, start ' + c.st4 + ', ' + (c.k4 === 'diff' ? 'different' : 'same') + '-class')],
    shapes: [{type: 'line', xref: 'x domain', yref: 'y', x0: 0, x1: 1, y0: 0.5, y1: 0.5, line: {color: '#2b2a27', dash: 'dot', width: 1}}],
    xaxis: merge(AXIS, {domain: [0, 0.48], tickvals: d.alls.map((_, k) => k), ticktext: d.alls, tickfont: {size: 9.5}, range: [0.6, d.alls.length - 0.6]}),
    yaxis: merge(AXIS, {range: [0, 0.55], title: {text: 'support percentile'}}),
    xaxis2: merge(AXIS, {domain: [0.58, 0.95], anchor: 'y2', title: {text: 's'}, showgrid: false}),
    yaxis2: merge(AXIS, {anchor: 'x2', showgrid: false})}), CONFIG);
}"""

DRAW_X4_SWITCH = r"""(el, d, c) => {
  const st = d.starts, tr = [], greys = ['#2b2a27', '#57534c', '#8d877c', '#b9b3a8', '#d9d3c9'];
  const frac = (k, j) => st.map(l => { const h = d.per_start[l].switches[k], n = h.reduce((a, b) => a + b, 0); return h[j] / n; });
  [[2, '2 changes'], [3, '3'], [4, '4'], [5, '5 or more']].forEach(([j, lab], q) => {
    tr.push({type: 'bar', x: st, y: frac('same', j), name: lab, legendgroup: 'n' + j, marker: {color: greys[q + 1]}, hovertemplate: 'start %{x}: %{y:.1%} (' + lab + ')<extra></extra>'});
    tr.push({type: 'bar', x: st, y: frac('diff', j), name: lab, legendgroup: 'n' + j, showlegend: false, xaxis: 'x2', yaxis: 'y2', marker: {color: greys[q + 1]},
      hovertemplate: 'start %{x}: %{y:.1%} (' + lab + ')<extra></extra>'}); });
  const tot = (k, from) => st.map(l => { const h = d.per_start[l].switches[k]; return h.slice(from).reduce((a, b) => a + b, 0) / h.reduce((a, b) => a + b, 0); });
  const ann = [panelTitle(0, 'same class: path leaves the class'), panelTitle(1, 'different class: more than one change')];
  tot('same', 1).forEach((v, i) => ann.push({x: st[i], y: v, xref: 'x', yref: 'y', text: (100 * v).toFixed(0) + '%', showarrow: false, yanchor: 'bottom', font: {size: 11}}));
  tot('diff', 2).forEach((v, i) => ann.push({x: st[i], y: v, xref: 'x2', yref: 'y2', text: (100 * v).toFixed(0) + '%', showarrow: false, yanchor: 'bottom', font: {size: 11}}));
  Plotly.react(el, tr, LAYOUT({height: 380, barmode: 'stack', legend: {orientation: 'h', y: 1.18, title: {text: 'prediction changes: '}},
    margin: {l: 56, r: 14, t: 40, b: 50}, annotations: ann,
    xaxis: merge(AXIS, {domain: [0, 0.46], title: {text: 'start site'}}), yaxis: merge(AXIS, {range: [0, 0.65], tickformat: '.0%', title: {text: 'share of pairs'}}),
    xaxis2: merge(AXIS, {domain: [0.54, 1], anchor: 'y2', title: {text: 'start site'}}), yaxis2: merge(AXIS, {anchor: 'x2', range: [0, 0.65], tickformat: '.0%'})}), CONFIG);
}"""

DRAW_X4_GAIN = r"""(el, d, c) => {
  const tr = [], key = c.gtg + '_' + c.gv, xi = q => d.alls.indexOf(q), s = [...Array(65).keys()].map(k => k / 64);
  const sh = y => { if (c.gc === 'abs') return y; let m = Infinity; for (const v of y) if (v != null) m = Math.min(m, v); return y.map(v => v == null ? null : v - m); };
  const layout = LAYOUT({height: 720, legend: {orientation: 'h', y: -0.08}, margin: {l: 56, r: 14, t: 30, b: 70}});
  for (let i = 0; i < 4; i++) { const dm = dom4(i); layout['xaxis' + ax(i)] = merge(AXIS, {domain: dm.x, anchor: 'y' + ax(i)});
    layout['yaxis' + ax(i)] = merge(AXIS, {domain: dm.y, anchor: 'x' + ax(i)}); }
  for (const l0 of d.starts) { const P = d.per_start[l0], A = P.aligned, col = d.start_col[l0];
    if (A['gain_' + key]) tr.push({x: A.x, y: sh(A['gain_' + key]), mode: 'lines', name: 'start ' + l0, legendgroup: l0, line: {color: col, width: 2},
      hovertemplate: 'start ' + l0 + ' (n = ' + A.n + ') · s − s_switch = %{x:.3f}: %{y:.3f}<extra></extra>'});
    if (P.gain[key]) tr.push({x: s, y: sh(P.gain[key].same), mode: 'lines', xaxis: 'x2', yaxis: 'y2', showlegend: false, legendgroup: l0,
      line: {color: col, width: 2, dash: 'dash'}, hovertemplate: 'start ' + l0 + ', same-class · s = %{x:.3f}: %{y:.3f}<extra></extra>'});
    const al = P.gain[c.gv === 'raw' ? 'align_raw' : 'align_ang'];
    tr.push({x: s, y: al.diff, mode: 'lines', xaxis: 'x3', yaxis: 'y3', showlegend: false, legendgroup: l0, line: {color: col, width: 2},
      hovertemplate: 'start ' + l0 + ' · s = %{x:.3f}: alignment %{y:.3f}<extra></extra>'});
    if (A.site_peak_rho_all) tr.push({x: P.sites.map(xi), y: A.site_peak_rho_all, mode: 'lines+markers', xaxis: 'x4', yaxis: 'y4', showlegend: false, legendgroup: l0,
      line: {color: col, width: 2}, marker: {size: 5}, text: P.sites, hovertemplate: 'start ' + l0 + ', all ' + A.n_all + ' paths · %{text}: ρ = %{y:.2f}<extra></extra>'});
    if (A.site_peak_rho) tr.push({x: P.sites.map(xi), y: A.site_peak_rho, mode: 'lines', xaxis: 'x4', yaxis: 'y4', showlegend: false, legendgroup: l0,
      line: {color: col, width: 1.4, dash: 'dot'}, text: P.sites, hovertemplate: 'start ' + l0 + ', ' + A.n + ' single-switch paths · %{text}: ρ = %{y:.2f}<extra></extra>'}); }
  const yl = (c.gv === 'ang' ? 'angular' : 'raw') + ' log gain → ' + c.gtg + (c.gc === 'rel' ? ', − min.' : '');
  layout.xaxis.title = {text: 's − s_switch'}; layout.yaxis.title = {text: yl};
  layout.xaxis2.title = {text: 's'};
  layout.xaxis3.title = {text: 's'}; layout.yaxis3.title = {text: 'alignment'}; layout.yaxis3.range = [0, 1];
  layout.xaxis4 = merge(layout.xaxis4, {tickvals: d.alls.map((_, k) => k), ticktext: d.alls, tickfont: {size: 9.5}, range: [0.6, d.alls.length - 0.6]});
  layout.yaxis4.title = {text: 'Spearman ρ'}; layout.yaxis4.range = [-0.1, 1];
  layout.shapes = [{type: 'line', xref: 'x', yref: 'y domain', x0: 0, x1: 0, y0: 0, y1: 1, line: {color: '#2b2a27', dash: 'dot', width: 1}}];
  layout.annotations = [panelTitle(0, '(a) different class, one switch: aligned on it'), panelTitle(1, '(b) same class, unaligned'),
    panelTitle(2, '(c) alignment of the path with the most-stretched direction'), panelTitle(3, '(d) peak speed at each site against the output switch')];
  Plotly.react(el, tr, layout, CONFIG);
}"""

DRAW_X4_PROBE = r"""(el, d, c) => {
  const G = d.probe_gain, key = c.pgs + '>logits_' + c.pgv, g0 = G['gain_' + c.pgv + '_' + c.pgs];
  const crossed = d.alls.length - 1 - d.alls.indexOf(c.pgs), f = c.pgf === 'ftle' ? 1 / crossed : 1;
  const g = g0.map(v => v == null ? null : v * f), m = G.margin;
  const ok = G.correct.map(v => v === 1), tr = [];
  for (const [sel, name, col] of [[true, 'correct', '#9a958c'], [false, 'misclassified', '#d7301f']]) {
    const ix = g.map((_, i) => i).filter(i => ok[i] === sel);
    tr.push({type: 'scattergl', mode: 'markers', x: ix.map(i => m[i]), y: ix.map(i => g[i]), name: name + ' (' + ix.length + ')',
      marker: {color: col, size: sel ? 3.5 : 5, opacity: sel ? 0.45 : 0.85}, hovertemplate: name + ' · margin %{x:.2f}, gain %{y:.3f}<extra></extra>'}); }
  const e = G.decile_edges, mid = e.slice(0, 10).map((v, i) => (v + e[i + 1]) / 2);
  tr.push({x: mid, y: G.deciles[key].map(v => v * f), mode: 'lines+markers', name: 'decile medians', line: {color: '#2b2a27', width: 2.4}, marker: {size: 7},
    hovertemplate: 'decile median · margin %{x:.1f}: %{y:.3f}<extra></extra>'});
  const Y = G.probe_y[c.pge], sorted = g.filter(v => v != null).slice().sort((a, b) => a - b);
  const lo = sorted[Math.floor(0.02 * sorted.length)], hi = sorted[Math.floor(0.98 * sorted.length)];
  tr.push({type: 'scattergl', mode: 'markers', x: Y.map(p => p[0]), y: Y.map(p => p[1]), xaxis: 'x2', yaxis: 'y2', showlegend: false,
    marker: {color: g, cmin: lo, cmax: hi, colorscale: 'Viridis', size: 4.5, colorbar: {thickness: 10, len: 0.85, title: {text: c.pgf === 'ftle' ? 'FTLE' : 'log gain', side: 'right'}}},
    text: g.map(v => v == null ? '' : v.toFixed(3)), hovertemplate: 'gain %{text}<extra></extra>'});
  const rad = (c.pgv === 'ang' && G.rho_ang_radius_removed) ? '<br>output radius removed: ' + G.rho_ang_radius_removed[c.pgs].toFixed(2) +
    ' (radius alone: ' + (G.rho_radius[c.pgs] > 0 ? '+' : '') + G.rho_radius[c.pgs].toFixed(2) + ')' : '';
  const yl = (c.pgv === 'ang' ? 'angular ' : 'raw ') + (c.pgf === 'ftle' ? 'FTLE (log gain / ' + crossed + ' sites)' : 'log gain') + ', ' + c.pgs + ' → logits';
  Plotly.react(el, tr, LAYOUT({height: 490, legend: {orientation: 'h', y: 1.2}, margin: {l: 60, r: 14, t: 70, b: 50},
    annotations: [panelTitle(0, 'Spearman ρ (gain, margin) = ' + G.rho[key].toFixed(2) + rad), panelTitle(1, 'probe in the stem-start suffix embedding, site ' + c.pge)],
    xaxis: merge(AXIS, {domain: [0, 0.5], title: {text: 'output margin (top − second logit)'}}), yaxis: merge(AXIS, {title: {text: yl}}),
    xaxis2: merge(AXIS, {domain: [0.6, 0.95], anchor: 'y2', title: {text: 'joint mode 1'}, zeroline: false}),
    yaxis2: merge(AXIS, {anchor: 'x2', title: {text: 'joint mode 2'}, zeroline: false})}), CONFIG);
}"""

DRAW_X4_EMB = r"""(el, d, c) => {
  const P = d.per_start[c.st4], Q = P.picture, li = P.chain.indexOf(c.esite);
  if (li < 0) { Plotly.purge(el); el.innerHTML = '<p class="note" style="padding:40px 10px">Site ' + c.esite + ' comes before the start site ' + c.st4 +
    ': the path does not exist there. Move the slider to ' + P.chain[0] + ' or later.</p>'; return; }
  const nt = el.querySelector(':scope > p.note'); if (nt) nt.remove();
  const [i, j] = [Number(c.emodes[0]), Number(c.emodes[1])], key = c.st4 + c.emodes;
  el._rng = el._rng || {};
  if (!el._rng[key]) { let a = [1e9, -1e9], b = [1e9, -1e9];
    const upd = p => { a = [Math.min(a[0], p[i]), Math.max(a[1], p[i])]; b = [Math.min(b[0], p[j]), Math.max(b[1], p[j])]; };
    for (const L of Q.probe) for (const p of L) upd(p); for (const L of Q.paths) for (const pth of L) for (const p of pth) upd(p);
    const pa = (a[1] - a[0]) * 0.04, pb = (b[1] - b[0]) * 0.04; el._rng[key] = [[a[0] - pa, a[1] + pa], [b[0] - pb, b[1] + pb]]; }
  const [rx, ry] = el._rng[key], tr = [];
  for (let k = 0; k < 10; k++) { const pts = Q.probe[li].filter((_, n) => Q.probe_labels[n] === k);
    tr.push({type: 'scattergl', mode: 'markers', x: pts.map(p => p[i]), y: pts.map(p => p[j]), name: CLASSES[k], marker: {color: CLASSCOL[k], size: 3.5, opacity: 0.35}, hoverinfo: 'skip'}); }
  const show = c.eex === 'all' ? Q.kind.map((_, n) => n) : [Number(c.eex) - 1];
  for (const n of show) { const pth = Q.paths[li][n];
    tr.push({x: pth.map(p => p[i]), y: pth.map(p => p[j]), mode: 'lines+markers', showlegend: false, line: {color: 'rgba(43,42,39,0.55)', width: 1.2},
      marker: {size: 5, color: Q.s, cmin: 0, cmax: 1, colorscale: [[0, '#2166ac'], [0.5, '#f7f7f7'], [1, '#b2182b']], line: {color: '#555', width: 0.4},
        colorbar: n === show[0] ? {thickness: 10, len: 0.5, y: 0.25, title: {text: 's', side: 'right'}} : undefined, showscale: n === show[0]},
      text: Q.s.map(v => 'path ' + (n + 1) + ' (' + CLASSES[Q.labels[n][0]] + ' → ' + CLASSES[Q.labels[n][1]] + '), s = ' + v.toFixed(3)),
      hovertemplate: '%{text}<extra></extra>'});
    tr.push({x: [pth[0][i]], y: [pth[0][j]], mode: 'markers+text', showlegend: false, marker: {size: 11, color: CLASSCOL[Q.labels[n][0]], line: {color: '#2b2a27', width: 1.2}},
      text: [String(n + 1)], textposition: 'top left', textfont: {size: 10}, hoverinfo: 'skip'});
    tr.push({x: [pth[pth.length - 1][i]], y: [pth[pth.length - 1][j]], mode: 'markers', showlegend: false,
      marker: {size: 11, symbol: 'square', color: CLASSCOL[Q.labels[n][1]], line: {color: '#2b2a27', width: 1.2}}, hoverinfo: 'skip'}); }
  Plotly.react(el, tr, LAYOUT({height: 560, legend: {orientation: 'v', x: 1.01, y: 1, font: {size: 11}, itemsizing: 'constant'}, margin: {l: 60, r: 10, t: 36, b: 48},
    title: {text: 'start ' + c.st4 + ' · suffix embedding · site ' + c.esite, font: {size: 13}, x: 0, xanchor: 'left'},
    xaxis: merge(AXIS, {range: rx, title: {text: 'joint mode ' + (i + 1)}, zeroline: false}), yaxis: merge(AXIS, {range: ry, title: {text: 'joint mode ' + (j + 1)}, zeroline: false})}), CONFIG);
}"""

DRAW_X4_RELU = r"""(el, d, c) => {
  const withTaps = d.starts.filter(l => d.per_start[l].taps), full = d.per_start[withTaps[0]], tr = [], xi = q => d.alls.indexOf(q);
  for (const [k, dash] of [['same', 'dash'], ['diff', 'solid']]) tr.push({x: full.taps.map(xi), y: full.relu_from0[k].map(r => r[r.length - 1]), mode: 'lines+markers',
    name: k === 'same' ? 'same class' : 'different class', line: {color: '#2b2a27', dash, width: 2}, marker: {size: 6}, text: full.taps,
    hovertemplate: k + ' · %{text}: %{y:.1%}<extra></extra>'});
  const P = d.per_start[c.st4];
  if (P.taps) { const s = [...Array(65).keys()].map(q => q / 64), n = P.taps.length;
    P.taps.forEach((tp, q) => { const t = n > 1 ? q / (n - 1) : 1, col = `rgb(${Math.round(198 - 190 * t)},${Math.round(219 - 171 * t)},${Math.round(239 - 132 * t)})`;
      tr.push({x: s, y: P.relu_from0[c.k4][q], mode: 'lines', xaxis: 'x2', yaxis: 'y2', name: 'tap ' + tp, showlegend: false, line: {color: col, width: 2},
        hovertemplate: 'tap ' + tp + ' · s = %{x:.3f}: %{y:.1%}<extra></extra>'});
      tr.push({x: [1.0], y: [P.relu_from0[c.k4][q][64]], mode: 'text', text: [tp], textposition: 'middle right', xaxis: 'x2', yaxis: 'y2', showlegend: false,
        textfont: {size: 9.5, color: '#6b665e'}, hoverinfo: 'skip'}); }); }
  for (const l0 of withTaps) { const Q = d.per_start[l0];
    tr.push({x: Q.taps.map(xi), y: Q.relu_crossings.diff[1], mode: 'lines+markers', xaxis: 'x3', yaxis: 'y3', name: 'start ' + l0, line: {color: d.start_col[l0], width: 2},
      marker: {size: 5}, text: Q.taps, hovertemplate: 'start ' + l0 + ' · %{text}: %{y:.2f}<extra></extra>'}); }
  Plotly.react(el, tr, LAYOUT({height: 400, legend: {orientation: 'h', y: 1.18, font: {size: 11}}, margin: {l: 56, r: 14, t: 40, b: 50},
    annotations: [panelTitle(0, 'endpoint masks differ'), panelTitle(1, P.taps ? 'differs from s = 0, start ' + c.st4 : 'start b9 has no taps downstream'),
      panelTitle(2, 'total mask changes per path (diff.)')],
    xaxis: merge(AXIS, {domain: [0, 0.28], tickvals: d.alls.map((_, k) => k), ticktext: d.alls, tickfont: {size: 9.5}, range: [0.6, 10.4]}),
    yaxis: merge(AXIS, {tickformat: '.0%', rangemode: 'tozero', title: {text: 'share of units'}}),
    xaxis2: merge(AXIS, {domain: [0.36, 0.64], anchor: 'y2', range: [0, 1.12], title: {text: 's'}}), yaxis2: merge(AXIS, {anchor: 'x2', tickformat: '.0%', rangemode: 'tozero'}),
    xaxis3: merge(AXIS, {domain: [0.72, 1], anchor: 'y3', tickvals: d.alls.map((_, k) => k), ticktext: d.alls, tickfont: {size: 9.5}, range: [0.6, 10.4]}),
    yaxis3: merge(AXIS, {anchor: 'x3', rangemode: 'tozero', title: {text: 'sum of changed fractions'}})}), CONFIG);
}"""

DRAW_X4_CTRL = r"""(el, d, c) => {
  const tr = [], xi = q => d.alls.indexOf(q);
  for (const l0 of d.starts) { const P = d.per_start[l0], C = P.controls, xs = P.sites.map(xi), col = d.start_col[l0];
    tr.push({x: xs, y: C.jvp_fd_med, mode: 'lines+markers', name: 'start ' + l0, legendgroup: l0, line: {color: col, width: 2}, marker: {size: 5}, text: P.sites,
      hovertemplate: 'start ' + l0 + ', 257 points · %{text}: %{y:.2e}<extra></extra>'});
    tr.push({x: xs, y: C.jvp_fd1024_med, mode: 'lines', showlegend: false, legendgroup: l0, line: {color: col, width: 1.6, dash: 'dash'}, text: P.sites,
      hovertemplate: 'start ' + l0 + ', 1,025 points · %{text}: %{y:.2e}<extra></extra>'}); }
  Plotly.react(el, tr, LAYOUT({height: 360, legend: {orientation: 'v', x: 1.01, y: 1}, margin: {l: 64, r: 10, t: 20, b: 50},
    xaxis: merge(AXIS, {tickvals: d.alls.map((_, k) => k), ticktext: d.alls, range: [0.6, d.alls.length - 0.6], title: {text: 'downstream site'}}),
    yaxis: merge(AXIS, {type: 'log', exponentformat: 'power', title: {text: 'median |exact − finite diff.| / max'}})}), CONFIG);
}"""


# ------------------------------------------------------------------------------------------------- assemble --

CSS = r"""
.lede { font-size:16px; max-width:85ch; }
.note { font-size:13px; color:var(--dim); max-width:none; }
.dim { color:var(--dim); font-size:12.5px; }
.sw { display:inline-block; width:11px; height:11px; border-radius:3px; margin-right:7px; vertical-align:-1px; border:1px solid rgba(0,0,0,.15); }
.where { font-size:13.5px; color:var(--dim); margin:-6px 0 10px; }
.thumbs { display:grid; grid-template-columns:repeat(auto-fill, minmax(270px, 1fr)); gap:8px; margin:10px 0; }
.thumb { display:flex; align-items:center; gap:8px; border:1px solid var(--line); border-radius:8px; background:white; padding:6px 8px; }
.thumb .tn { font-weight:700; font-size:13px; min-width:20px; color:var(--dim); }
.thumb .ti { display:flex; flex-direction:column; align-items:center; font-size:10.5px; line-height:1.2; color:var(--dim); text-align:center; }
.thumb img { width:64px; height:64px; image-rendering:pixelated; border-radius:3px; margin-bottom:2px; }
.thumb b { color:var(--ink); font-weight:600; }
.banner svg text { font-size:10.5px; }
.box .oq { font-size:11.5px; letter-spacing:.06em; text-transform:uppercase; color:var(--dim); font-weight:700; margin-bottom:4px; }
.box.takeaway h4 { margin:10px 0 2px; }
@media (max-width: 700px) { main { padding:16px 14px 60px; } .grid { grid-template-columns:1fr; } }
"""


def banner():
    widths = [16] * 4 + [32] * 3 + [64] * 3
    res = [32] * 4 + [16] * 3 + [8] * 3
    blocks = [('stem conv', '16 ch · 32²', 'linear')]
    blocks += [(f'block {i}', f'{widths[i]} ch · {res[i]}²', 'mlp') for i in range(1, 10)]
    blocks += [('avg pool', '64', 'norm'), ('linear head', '4–1000', 'out')]
    cuts = {i: s for i, s in enumerate(SITES)}
    skips = [(i - 1, i) for i in range(1, 10)]
    return arch_svg(blocks, skips=skips, cuts=cuts, bw=80, gap=14,
                    caption='ResNet-20: a 3×3 stem convolution, nine residual blocks (two 3×3 convolutions each, green arcs: the '
                            'skip connections) in three stages of width 16, 32 and 64 channels at 32, 16 and 8 px, global average '
                            'pooling and a linear head (4 to 1,000 outputs, by training set). Red ticks: the 12 sites where '
                            'activations are read. The X1 primary solve uses the trunk, stem to b9.')


def main():
    x1p, x3p = OUT / 'x1_summary.json', OUT / 'x3_summary.json'
    x1 = json.loads(x1p.read_text()) if x1p.exists() else None
    x3 = json.loads(x3p.read_text()) if x3p.exists() else None
    x4p = OUT / 'x4_summary.json'
    x4 = json.loads(x4p.read_text()) if x4p.exists() else None
    have = {k for k, *_, f in EXPERIMENTS if (OUT / f).exists()}
    n1 = len(x1['measures'][x1['probes'][0]]['nets']) if x1 else 0
    nb = Notebook('Depth embedding: training sets, adversarial images, interpolation',
                  'Exploratory overnight experiments on the through-depth embedding of a fixed probe set in ResNet-20: how it '
                  'changes with the training set (X1), how adversarial images travel through it (X3), and what happens along '
                  'straight paths between real activations (X4).',
                  status=f'EXPLORATORY | ResNet-20 at 32 px | 2 seeds per training set | 1 probe per dataset | X1: {n1} networks'
                         + ('' if x4 else ' | X4 pending'),
                  kicker='DepthEmbedding · xp · 2026-11-21')
    nb.banner(banner(), label='Network (every experiment)')
    nb.style(CSS)
    nb.script(JS.replace('%SITES%', json.dumps(SITES)).replace('%GROUPS%', json.dumps({k: list(v) for k, v in GROUPS.items()}))
              .replace('%PROBES%', json.dumps(PROBES)).replace('%CLASSES%', json.dumps(CLASSES)))
    tab_overview(nb, x1, x3, x4, have)
    if x1:
        tab_x1(nb, clean(x1), x3)
    if x3:
        tab_x3(nb, clean(x3))
    if x4:
        tab_x4(nb, x4)
    PAGE.parent.mkdir(parents=True, exist_ok=True)
    nb.write(PAGE)


if __name__ == '__main__':
    main()
