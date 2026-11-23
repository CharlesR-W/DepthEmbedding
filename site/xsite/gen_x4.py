"""Build the X4 entry (straight lines between two images' activations): facts, static figures, explorer spec, card.

Run from the repository root:
    uv run --with numpy --with pillow --with scipy --with matplotlib --with fonttools --with brotli \
        python site/xsite/gen_x4.py

Static figures are lossless WebP at two image pixels per CSS pixel (figure size in points = CSS px, saved at 144 dpi),
with text set in the site's Inter. The fragment (site/content/experiments/x4-paths.html) states each image's CSS size.
"""
import io
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from opdata import require  # noqa: E402
from ops_x4 import STARTS, sample_pairs  # noqa: E402
from validate import Inputs, check_probe  # noqa: E402
from xlib import Out, PALETTE  # noqa: E402

ID = 'x4-paths'
ROOT = Path(__file__).resolve().parents[2]
BASE = 'crw-depth-xp-results/x4/'
ALL_STARTS = ('pixels', 'stem', 'b3', 'b6', 'b8', 'b9')
ORDER = ('pixels', 'stem', *(f'b{i}' for i in range(1, 10)), 'pool', 'logits')   # common.ALL_SITES
CLASSES = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
EXAMPLES = (183, 150, 33)          # rows of paths_<start>.npz (the same pair at every start), chosen in the design doc
STEEP = 204                        # the b3-start line with the largest one-sixteenth KL rise (asserted below)
KL_TOP = 32                        # top of the F2 KL axis (nats)
LEAD_SITES = ('b4', 'b6', 'b8', 'b9', 'logits')
EMB_SITES = ('b3', 'b5', 'b7', 'b8', 'b9', 'logits')
CURVE_STARTS = ('pixels', 'b3', 'b6', 'b9')
EMB_PX = 236                       # embedding panel size, CSS px
W16, W16_64 = 16, 4                # one sixteenth of the line on the 256- and 64-step grids
SHARE_THRESHOLD = .25              # the earlier direction-only summary threshold (xp/analyze_x4.py)

INK, INK2, MUTE, GRID = PALETTE['fg'], '#3f3e3b', PALETTE['mute'], PALETTE['line']
BLUE, ORANGE, WALL = PALETTE['sets'][0], PALETTE['sets'][1], PALETTE['diverging']['pos']
ORD = PALETTE['ordinal']
START_COLOUR = {'pixels': ORD[1], 'b3': ORD[3], 'b6': ORD[5], 'b9': INK}
CLASS_COLOUR = PALETTE['classes']
DIFF_KW = dict(color=INK, alpha=.13, lw=.6)        # different-class bundle (F1, F2, F5)
SAME_KW = dict(color=ORANGE, alpha=.28, lw=.6)     # same-class bundle


# ---------------------------------------------------------------- numerics
def window(d, w):
    """Largest share of the total in any w consecutive steps along the last axis (d >= 0)."""
    c = np.pad(np.cumsum(d, axis=-1, dtype=np.float64), [(0, 0)] * (d.ndim - 1) + [(1, 0)])
    return (c[..., w:] - c[..., :-w]).max(-1) / c[..., -1]


def cumulative(d):
    c = np.pad(np.cumsum(d, axis=-1, dtype=np.float64), [(0, 0)] * (d.ndim - 1) + [(1, 0)])
    return c / c[..., -1:]


def kl_from_start(logits):
    z = logits.astype(np.float64)
    z = z - z.max(-1, keepdims=True)
    lp = z - np.log(np.exp(z).sum(-1, keepdims=True))
    return (np.exp(lp[..., :1, :]) * (lp[..., :1, :] - lp)).sum(-1), lp


def route(pred):
    seq = [int(pred[0])]
    for p in pred[1:]:
        if p != seq[-1]:
            seq.append(int(p))
    return seq


def switches(pred, grid):
    return grid[1:][np.flatnonzero(np.diff(pred))] - .5 / (len(grid) - 1)


# ---------------------------------------------------------------- drawing
class Figs:
    """Fixed-size matplotlib canvases in CSS px (1 pt = 1 CSS px), saved as 2x lossless WebP."""

    def __init__(self, out, tmp):
        from fontTools.ttLib import TTFont
        font = TTFont(ROOT / 'site/vendor/fonts/inter-latin-opsz-normal.woff2')
        font.flavor = None
        ttf = Path(tmp) / 'inter.ttf'
        font.save(ttf)
        font_manager.fontManager.addfont(str(ttf))
        name = font_manager.FontProperties(fname=str(ttf)).get_name()
        plt.rcParams.update({
            'font.family': name, 'font.size': 10.5, 'text.color': INK, 'axes.labelcolor': INK2,
            'axes.labelsize': 10.5, 'axes.edgecolor': MUTE, 'axes.linewidth': .7, 'axes.spines.top': False,
            'axes.spines.right': False, 'xtick.color': MUTE, 'ytick.color': MUTE, 'xtick.labelcolor': INK2,
            'ytick.labelcolor': INK2, 'xtick.labelsize': 10, 'ytick.labelsize': 10, 'xtick.major.size': 3,
            'ytick.major.size': 3, 'xtick.major.width': .7, 'ytick.major.width': .7, 'axes.labelpad': 3,
            'lines.solid_capstyle': 'round', 'svg.fonttype': 'path'})
        self.out, self.sizes = out, {}

    @staticmethod
    def canvas(w, h, margins):
        left, bottom, right, top = margins
        fig = plt.figure(figsize=(w / 72, h / 72), dpi=72)      # 1 display px = 1 pt = 1 CSS px
        ax = fig.add_axes([left / w, bottom / h, (w - left - right) / w, (h - bottom - top) / h])
        return fig, ax

    def save(self, fig, name, background=None):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=144, transparent=background is None,
                    **({'facecolor': background} if background else {}))
        plt.close(fig)
        buf.seek(0)
        with Image.open(buf) as im:
            im.load()
            path = self.out.fig_path(name)
            if background:
                im.convert('RGB').save(path, 'WEBP', quality=85, method=6)
            else:
                im.save(path, 'WEBP', lossless=True, method=6)
            self.sizes[name] = (im.width // 2, im.height // 2)
        return path


def s_axis(ax, label=True):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1], ['0', '½', '1'])
    if label:
        ax.set_xlabel('position along the line, s')


def cum_axes(ax, ylabel=True):
    s_axis(ax)
    ax.set_ylim(-.02, 1.02)
    ax.set_yticks([0, .5, 1], ['0%', '50%', '100%'])
    if ylabel:
        ax.set_ylabel('distance covered so far')
    ax.plot([0, 1], [0, 1], color=MUTE, lw=1, ls=(0, (4, 3)), zorder=4)
    ax.plot([0, .5, .5, 1], [0, 0, 1, 1], color=WALL, lw=1.3, ls=(0, (1, 2)), zorder=4)


def note(ax, text, x=.03, y=.97, ha='left', va='top', size=9.5):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=size, color=INK, zorder=6, linespacing=1.3,
            bbox=dict(boxstyle='round,pad=0.25', fc='white', ec='none', alpha=.85))


def bundle(ax, x, curves, kind, highlight=None):
    for i in np.flatnonzero(kind == 0):
        ax.plot(x, curves[i], **SAME_KW, zorder=2)
    for i in np.flatnonzero(kind == 1):
        ax.plot(x, curves[i], **DIFF_KW, zorder=2)
    if highlight is not None:
        ax.plot(x, curves[highlight], color=BLUE, lw=1.7, zorder=5)


# ---------------------------------------------------------------- main
def main():
    inputs, out = Inputs(), Out(ID)
    Z, sources = {}, {}
    with np.load(inputs.file(BASE + 'pairs.npz'), allow_pickle=False) as pair_file:
        pair_ids, pair_kind = pair_file['pair_idx'], pair_file['kind']
    keys = ('sites', 'chain', 'taps', 'kind', 'labels', 'pair_idx', 'images', 's256', 's64', 'd_raw', 'd64_raw', 'share_raw',
            'share_unit', 'sigma_raw', 'ref_rows', 'share1024_unit', 'share1024_raw', 'relu_prev', 'logits', 'pred',
            's_switch', 'n_switch', 'speed', 'gain_raw', 'align_raw', 'emb_site_y', 'probe_site_y', 'probe_labels',
            'probe_idx', 'emb_cmax', 'ref_cmax', 'jvp_fd_med', 'jvp_fd1024_med')
    for start in ALL_STARTS:
        if start in STARTS:
            record = inputs.bundle(f'x4-{start}-suffix')
            path, source = inputs.file(record['file']), record['file']
        else:
            source = f'xp/out/x4/paths_{start}.npz'
            path = ROOT / source
        with np.load(path, allow_pickle=False) as z:
            Z[start] = {k: (z[k].tolist() if k in ('sites', 'chain', 'taps') else z[k]) for k in keys}
        require(np.array_equal(Z[start]['pair_idx'], pair_ids) and np.array_equal(Z[start]['kind'], pair_kind),
                f'{path}: pair order differs')
        sources[start] = source
    kind = pair_kind
    same, diff = kind == 0, kind == 1
    n_pairs = int(same.sum())
    require(n_pairs == diff.sum(), 'unequal pair strata')
    src_all = 'paths_<start>.npz (six starts)'

    def fact(name, value, *a, **kw):
        return out.fact(name, value, *a, **kw)

    # ------------------------------------------------ scope and grid
    z0 = Z['pixels']
    fact('pairs.per_kind', n_pairs, tier='P', source=BASE + 'pairs.npz:kind; count per stratum')
    fact('pairs.display', len(sample_pairs(pair_kind)), tier='R', source=BASE + 'pairs.npz:kind, replay site/xsite/ops_x4.py:sample_pairs')
    fact('pairs.unique_endpoints', np.unique(pair_ids).size, tier='R', source=BASE + 'pairs.npz:pair_idx; distinct test image IDs')
    fact('pairs.margin_min', 1e-3, digits=3, tier='S', source='xp/stage_x4.py:MARGIN_MIN, fixed eligibility criterion')
    fact('probe.n', len(z0['probe_idx']), tier='R', source=sources['pixels'] + ':probe_idx length')
    fact('starts.tested', len(ALL_STARTS), tier='R', source='xp/out/x4/paths_*.npz; number of measured starts')
    fact('paths.total', 2 * n_pairs * len(ALL_STARTS), tier='P', source=src_all + ':kind; pairs x starts')
    fact('grid.points', len(z0['s256']), tier='R', source=sources['pixels'] + ':s256 length')
    fact('grid.draw', len(z0['s64']), tier='R', source=sources['pixels'] + ':s64 length (speed, gain and embedding grid)')
    fact('grid.uniform', W16 / (len(z0['s256']) - 1), pct=True, digits=2, tier='R', source='16 of the 256 steps')
    fact('grid.jump', 1.0, pct=True, tier='R', source='a single jump puts all of the distance into one window')
    fact('emb.k', z0['emb_site_y'].shape[-1], tier='R', source=sources['pixels'] + ':emb_site_y last axis (site-local coordinates kept)')
    fact('emb.perplexity', 100, tier='S', source='operators x4-<start>-suffix: Shannon kernel ea100 (xp/stage_x4.py)')

    # ------------------------------------------------ raw distance concentration (F1)
    win_all, win_pre = [], []
    for start in ALL_STARTS:
        z = Z[start]
        share = window(z['d_raw'], W16)
        require(np.allclose(share, z['share_raw'], atol=1e-5), f'{start}: share_raw does not match d_raw')
        pre = np.asarray(z['sites']) != 'logits'
        win_all.append(share.max())
        win_pre.append(share[:, pre].max())
        fact(f'window.{start}.max', share.max(), pct=True, digits=1 if start != 'b9' else 2, tier='P',
             source=sources[start] + ':d_raw; max over pairs and downstream sites of the fastest 16-of-256-step share')
    fact('window.max', max(win_all), pct=True, digits=1, tier='P',
         source=src_all + ':d_raw; max over 6 starts x 300 pairs x downstream sites of the fastest 16-of-256-step share')
    fact('window.max_pre_logits', max(win_pre), pct=True, digits=1, tier='P', source=src_all + ':d_raw; same, sites before logits')
    zb3 = Z['b3']
    b3_share = window(zb3['d_raw'], W16)
    lead_stats = {}
    for site in LEAD_SITES:
        m = zb3['sites'].index(site)
        lead_stats[site] = (np.median(b3_share[:, m]), b3_share[:, m].max(), int(np.argmax(b3_share[:, m])))
        fact(f'window.b3.{site}.median', lead_stats[site][0], pct=True, digits=1, tier='P', source=sources['b3'] + f':d_raw at {site}; median over 300 pairs')
        fact(f'window.b3.{site}.max', lead_stats[site][1], pct=True, digits=1, tier='P', source=sources['b3'] + f':d_raw at {site}; max over 300 pairs')
    relu = np.concatenate([window(Z[s]['relu_prev'].astype(np.float64), W16).ravel() for s in ALL_STARTS if Z[s]['taps']])
    fact('relu.window.median', np.median(relu), pct=True, tier='P', source=src_all + ':relu_prev; median over start x tap x pair of the fastest-sixteenth share of ReLU flips')
    fact('relu.window.max', relu.max(), pct=True, tier='P', source=src_all + ':relu_prev; max of the same')
    fact('relu.window.n', relu.size, tier='P', source=src_all + ':relu_prev; start x tap x pair combinations')
    ref_raw = max(np.max(abs(Z[s]['share1024_raw'] - Z[s]['share_raw'][Z[s]['ref_rows']])) for s in ALL_STARTS)
    fact('refine.raw_max', 100 * ref_raw, digits=2, tier='P',
         source=src_all + ':share1024_raw - share_raw[ref_rows]; max absolute change, in percentage points')
    fact('refine.n', sum(len(Z[s]['ref_rows']) for s in ALL_STARTS), tier='P', source=src_all + ':ref_rows; refined lines')
    fact('sharp.min', min(np.nanmin(Z[s]['sigma_raw']) for s in ALL_STARTS), digits=2, tier='P',
         source=src_all + ':sigma_raw; minimum across starts, sites and pairs')

    # ------------------------------------------------ the earlier direction-only measure (appendix)
    hits, logit_hits, refine_med = 0, 0, []
    for start in ALL_STARTS:
        z = Z[start]
        sites = np.asarray(z['sites'])
        share = z['share_unit']
        hits += int(np.any(share[:, (sites != 'logits') & (sites != start)] > SHARE_THRESHOLD, axis=1).sum())
        logit_hits += int((share[:, sites == 'logits'] > SHARE_THRESHOLD).sum())
        refine_med.append(np.nanmax(np.nanmedian(abs(z['share1024_unit'] - share[z['ref_rows']]), axis=0)))
    fact('threshold.share', SHARE_THRESHOLD, pct=True, tier='S', source='xp/analyze_x4.py: sharp-window summary threshold')
    fact('pre_logits.n', hits, tier='P', source=src_all + ':share_unit; start-pair lines above 25% at a pre-logit downstream site')
    fact('logits.n', logit_hits, tier='P', source=src_all + ':share_unit; start-pair lines above 25% at the logits')
    fact('refine.max_median', max(refine_med), digits=4, tier='P', source=src_all + ':share1024_unit - share_unit[ref_rows]; largest per-site median')

    # ------------------------------------------------ outputs: KL, steepest flips (F2)
    kl, prob_a = {}, {}
    for start in ALL_STARTS:
        k, lp = kl_from_start(Z[start]['logits'])
        kl[start] = k
        prob_a[start] = np.exp(lp[np.arange(len(k)), :, Z[start]['labels'][:, 0]])
    fact('kl.end_median', np.median(kl['b3'][diff, -1]), digits=1, tier='P', source=sources['b3'] + ':logits; median KL(p(0)||p(1)), different-class pairs')
    rise = kl['b3'][:, W16:] - kl['b3'][:, :-W16]
    row, col = np.unravel_index(np.argmax(rise), rise.shape)
    require(row == STEEP, f'steepest b3 KL rise is row {row}, not {STEEP}')
    drop = prob_a['b3'][:, :-W16] - prob_a['b3'][:, W16:]
    row_p, col_p = np.unravel_index(np.argmax(drop), drop.shape)
    require(row_p == STEEP, f'largest b3 one-sixteenth probability drop is row {row_p}, not {STEEP}')
    s256 = z0['s256']
    lab = Z['b3']['labels']
    src_b3 = sources['b3'] + ':logits, labels'
    fact('steep.b3.row', STEEP, tier='P', source=src_b3 + '; argmax over pairs of the largest 16-step rise of KL(p(0)||p(s))')
    fact('steep.b3.a', CLASSES[lab[STEEP, 0]], tier='P', source=src_b3)
    fact('steep.b3.b', CLASSES[lab[STEEP, 1]], tier='P', source=src_b3)
    fact('steep.b3.p_from', prob_a['b3'][STEEP, col_p], pct=True, digits=1, tier='P', source=src_b3 + '; softmax prob of class a at window start')
    fact('steep.b3.p_to', prob_a['b3'][STEEP, col_p + W16], pct=True, digits=1, tier='P', source=src_b3 + '; same, 16 steps later')
    fact('steep.b3.kl_from', kl['b3'][STEEP, col], digits=1, tier='P', source=src_b3 + '; KL at the start of the largest 16-step KL rise')
    fact('steep.b3.kl_to', kl['b3'][STEEP, col + W16], digits=1, tier='P', source=src_b3 + '; KL 16 steps later')
    fact('steep.b3.kl_s_from', s256[col], digits=3, tier='P', source=src_b3 + '; s at the start of that window')
    fact('steep.b3.kl_s_to', s256[col + W16], digits=3, tier='P', source=src_b3 + '; s at its end')
    d_log = zb3['d_raw'][STEEP, zb3['sites'].index('logits')].astype(np.float64)
    fact('steep.b3.kl_window_share', d_log[col:col + W16].sum() / d_log.sum(), pct=True, digits=1, tier='P',
         source=sources['b3'] + ':d_raw at logits; the same pair, share of raw logit distance inside the KL-rise window')
    d9 = prob_a['b9'][:, :-W16] - prob_a['b9'][:, W16:]
    fact('steep.b9.drop_pp', 100 * d9.max(), digits=0, tier='P',
         source=sources['b9'] + ':logits, labels; largest 16-step drop of p(class a), percentage points, all 300 lines')
    for start in ('b3', 'b9'):
        fact(f'heim.clip.{start}', int((kl[start].max(1) > KL_TOP).sum()), tier='P',
             source=sources[start] + f':logits; lines whose KL from s = 0 exceeds the {KL_TOP:g}-nat axis')

    # ------------------------------------------------ detours (F4)
    detour_same, detour_diff = [], []
    for start in ALL_STARTS:
        z = Z[start]
        pred, lb = z['pred'], z['labels']
        off_same = pred[same] != lb[same, :1]
        leave = off_same.any(1)
        other = np.array([~np.isin(pred[i], lb[i]) for i in np.flatnonzero(diff)])
        detour_same.append(leave.mean())
        detour_diff.append(other.any(1).mean())
        src = sources[start] + ':pred, labels, kind'
        fact(f'leaves.{start}.n', int(leave.sum()), tier='P', source=src + '; same-class pairs with any argmax unequal to label')
        fact(f'leaves.{start}.share', leave.mean(), pct=True, tier='P', source=src + '; same-class share with any argmax unequal to label')
        fact(f'third.{start}.share', other.any(1).mean(), pct=True, tier='P', source=src + '; different-class share with any argmax outside both labels')
        if leave.any():
            fact(f'leave_stretch.{start}.median', np.median(off_same[leave].mean(1)), pct=True, tier='P',
                 source=src + '; leavers, median share of the 257 points outside the class')

    # ------------------------------------------------ Jacobian along the line (appendix numbers)
    for start in ('pixels', 'b3', 'b6'):
        z = Z[start]
        g = z['gain_raw'][:, 0]
        mid = np.median(g[diff, 32])
        fact(f'gain.{start}.s0', np.median(g[diff, 0]), digits=2, tier='P', source=sources[start] + ':gain_raw[:, logits, 0]; median, different-class')
        fact(f'gain.{start}.mid', mid, digits=2, tier='P', source=sources[start] + ':gain_raw[:, logits, 32]; median, different-class')
        fact(f'ftle.{start}.mid', mid / (ORDER.index('logits') - ORDER.index(start)), digits=2, tier='P',
             source=sources[start] + ':gain_raw midpoint median / sites crossed')
        fact(f'align.{start}.mid', np.median(z['align_raw'][diff, 32]), pct=True, tier='P', source=sources[start] + ':align_raw[:, 32]; median, different-class')
    L3 = zb3['sites'].index('logits')
    peak = zb3['s64'][np.argmax(zb3['speed'][:, L3], 1)]
    one = diff & (zb3['n_switch'] == 1)
    fact('timing.all', spearmanr(peak[diff], zb3['s_switch'][diff]).statistic, digits=2, tier='P',
         source=sources['b3'] + ':speed[:, logits], s_switch; Spearman of peak-speed s vs first switch, different-class')
    fact('timing.single', spearmanr(peak[one], zb3['s_switch'][one]).statistic, digits=2, tier='P', source=sources['b3'] + '; same, single-switch pairs')
    fact('timing.single_n', int(one.sum()), tier='P', source=sources['b3'] + ':n_switch; different-class pairs with one switch')
    fact('speedfd.pixels', z0['jvp_fd_med'][z0['sites'].index('logits')], pct=True, digits=1, tier='S', source=sources['pixels'] + ':jvp_fd_med at logits')
    fact('speedfd1024.pixels', z0['jvp_fd1024_med'][z0['sites'].index('logits')], pct=True, digits=1, tier='S', source=sources['pixels'] + ':jvp_fd1024_med at logits')

    # ------------------------------------------------ embedding concentration and support (F5)
    zb9 = Z['b9']
    m9 = zb3['sites'].index('b9')
    raw65 = window(zb3['d64_raw'][:, m9], W16_64)

    def emb_share(z, coords=16):
        y = z['emb_site_y'][z['chain'].index('b9')][..., :coords].astype(np.float64)
        return window(np.linalg.norm(np.diff(y, axis=1), axis=-1), W16_64)
    e3, e9 = emb_share(zb3), emb_share(zb9)
    fact('embstep.raw_b3', np.median(raw65[diff]), pct=True, tier='P', source=sources['b3'] + ':d64_raw at b9; median fastest 4-of-64-step share, different-class')
    fact('embstep.emb_b3', np.median(e3[diff]), pct=True, tier='P', source=sources['b3'] + ':emb_site_y at b9 (16 coords); median fastest 4-of-64-step share, different-class')
    fact('embstep.emb_b9', np.median(e9[diff]), pct=True, tier='P', source=sources['b9'] + ':emb_site_y at b9 (16 coords); same')
    fact('embstep.half_b3', int((e3[diff] > .5).sum()), tier='P', source=sources['b3'] + '; different-class pairs above 50%')
    fact('embstep.half_b9', int((e9[diff] > .5).sum()), tier='P', source=sources['b9'] + '; different-class pairs above 50%')
    for site in ('b9', 'logits'):
        l = zb3['chain'].index(site)
        ref = np.sort(zb3['ref_cmax'][l])
        sup = np.searchsorted(ref, zb3['emb_cmax'][l]) / ref.size
        fact(f'support.b3.{site}', np.median(sup[diff].min(1)), pct=True, tier='P',
             source=sources['b3'] + f':emb_cmax, ref_cmax at {site}; median over different-class pairs of the path-minimum lower-tail percentile')
    p2 = zb3['probe_site_y'][..., :2].astype(np.float64) ** 2
    fact('modes.top2_max', (np.sort(p2, 1)[:, -2:].sum(1) / p2.sum(1)).max(), pct=True, digits=1, tier='R',
         source=sources['b3'] + ':probe_site_y modes 1-2; max two-image squared-norm share over chain sites')

    # ------------------------------------------------ worked examples
    for r in EXAMPLES:
        a, b = lab[r]
        fact(f'we{r}.a', CLASSES[a], tier='P', source=src_b3)
        fact(f'we{r}.b', CLASSES[b], tier='P', source=src_b3)
        fact(f'we{r}.ids', f'{pair_ids[r, 0]} and {pair_ids[r, 1]}', tier='R', source=BASE + 'pairs.npz:pair_idx')
        for start in CURVE_STARTS:
            z = Z[start]
            src = sources[start] + f':row {r}'
            L = z['sites'].index('logits')
            sp, g = z['speed'][r, L].astype(np.float64), z['gain_raw'][r, 0]
            chord = np.linalg.norm(z['logits'][r, -1].astype(np.float64) - z['logits'][r, 0])
            sw = switches(z['pred'][r], s256)
            fact(f'we{r}.{start}.route', ' → '.join(CLASSES[c] for c in route(z['pred'][r])), tier='P', source=src + ' pred')
            if len(sw):
                fact(f'we{r}.{start}.sw1', sw[0], digits=2, tier='P', source=src + ' pred; first switch, step midpoint')
                fact(f'we{r}.{start}.swlast', sw[-1], digits=2, tier='P', source=src + ' pred; last switch')
            off = ~np.isin(z['pred'][r], (a, b)) if a != b else z['pred'][r] != a
            fact(f'we{r}.{start}.off', off.mean(), pct=True, tier='P', source=src + ' pred; share of points outside the endpoint classes')
            fact(f'we{r}.{start}.kl_max', kl[start][r].max(), digits=1, tier='P', source=src + ' logits; max KL from s = 0')
            fact(f'we{r}.{start}.peak_s', z['s64'][sp.argmax()], digits=2, tier='P', source=src + ' speed at logits; s of the maximum')
            fact(f'we{r}.{start}.peak', sp.max(), digits=0, tier='P', source=src + ' speed at logits; maximum')
            fact(f'we{r}.{start}.peak_chord', sp.max() / chord, digits=1, tier='P', source=src + ' speed max / |logits(1) - logits(0)|')
            fact(f'we{r}.{start}.gain0', g[0], digits=2, tier='P', source=src + ' gain_raw[logits] at s = 0')
            fact(f'we{r}.{start}.gain_ratio', np.exp(np.ptp(g)), digits=1, tier='P', source=src + ' exp(max - min) of gain_raw[logits]')
            fact(f'we{r}.{start}.gain_peak_s', z['s64'][g.argmax()], digits=2, tier='P', source=src + ' s of the gain maximum')
        fact(f'we{r}.chord', np.linalg.norm(Z['b3']['logits'][r, -1].astype(np.float64) - Z['b3']['logits'][r, 0]), digits=0, tier='P',
             source=src_b3 + f'; row {r}, distance between the endpoint logits')
        fact(f'we{r}.kl_end', kl['b3'][r, -1], digits=1, tier='P', source=src_b3 + f'; row {r}, KL(p(0)||p(1))')

    # ------------------------------------------------ gain across the probe (appendix)
    check_src = BASE + 'checks.json'
    import json
    checks = json.loads(inputs.file(check_src).read_text())
    for start in ('pool', 'b9'):
        key = f'{start}_logits_exact_err'
        out.fact(f'gain.affine.{start}', checks[key], tier='S', source=f'{check_src}:{key}; max abs log-gain error vs closed form')
        out.facts[f'gain.affine.{start}']['text'] = f'{checks[key]:.2g}'
    gain_src = BASE + 'probe_gain.npz'
    viewer = inputs.manifest['ops']['r20s0-c10-knn15']['probe']['viewer']
    with np.load(inputs.file(gain_src), allow_pickle=False) as z:
        check_probe(z['probe_idx'], z['labels'], inputs.value(viewer + ':idx'), inputs.value(viewer + ':labels'))
        i, t = z['starts'].tolist().index('pixels'), z['targets'].tolist().index('logits')
        raw, ang, rad, margin = z['gain_raw'][i, t], z['gain_ang'][i, t], z['rad_out'][i, t], z['margin']
    for key, values in (('raw', raw), ('angular', ang), ('radius_removed', ang + np.log(rad))):
        fact(f'gain.pixels.{key}', spearmanr(values, margin).statistic, digits=2, tier='P', source=f'{gain_src}; Spearman with margin, pixels to logits')
    fact('gain.radius', spearmanr(np.log(rad), margin).statistic, digits=2, tier='P', source=gain_src + ':log(rad_out) vs margin')

    # ------------------------------------------------ explorer (kept from the earlier page)
    switch_items = []
    for start in STARTS:
        chain = Z[start]['chain']
        label = 'b9 (affine after)' if start == 'b9' else 'Pixels' if start == 'pixels' else start
        film = [chain[0], chain[len(chain) // 2], chain[-1]] if start == 'b9' else [chain[0], 'b9', chain[-1]]
        switch_items.append(dict(label=label, panels=[dict(op=f'x4-{start}-suffix')], paths=[f'x4-{start}-paths'],
                                 site='b9', filmstrip=film))
    out.chart('emb_explore', dict(type='emb', schema=1, K=16, dims=2, norm='site', switch=switch_items, kinds=['site', 'joint'],
                                  kind='site', axes='pooled:0', colour={'by': 'class'}, height=440))

    # ------------------------------------------------ figures
    with tempfile.TemporaryDirectory() as tmp:
        figs = Figs(out, tmp)
        figs.text = lambda key: out.facts[key]['text']     # numbers drawn in figures come from the fact catalogue
        draw_lead(figs, zb3, b3_share, lead_stats, kind)
        draw_heim(figs, kl, kind, (col, col + W16))
        draw_detours(figs, detour_same, detour_diff)
        draw_embstep(figs, zb3, zb9, kind)
        for r in EXAMPLES:
            draw_example(figs, Z, kl, r)
        draw_thumb(figs, zb3, b3_share, kl, kind)
    out.chart('card', dict(
        answer=('No wall in the activations: on ' + f('paths.total') + ' straight lines between two images’ activations, no '
                'recorded later site moved more than ' + f('window.max') + ' of its distance in any sixteenth of the line. '
                'Class probabilities can still change sharply.'),
        numbers=[f('window.max') + ' at most in any sixteenth (' + f('grid.uniform') + ' = even pace)',
                 f('steep.b3.p_from') + ' → ' + f('steep.b3.p_to') + ' start-class probability within one sixteenth (steepest line drawn at b3)',
                 f('leaves.b3.share') + ' of same-class lines drawn at b3 leave their class; ' + f('leaves.b8.n') + ' of '
                 + f('pairs.per_kind') + ' drawn at b8'],
        thumb='assets/fig/x_x4-paths_thumb.webp'))
    out.write()
    for name, size in sorted(figs.sizes.items()):
        print(f'{name}: {size[0]}x{size[1]} css px')


def f(key):
    return '{{fact:x.' + ID + '.' + key + '}}'


# ---------------------------------------------------------------- figure builders
def draw_lead(figs, z, share, stats, kind):
    s = z['s256']
    for k, site in enumerate(LEAD_SITES):
        m = z['sites'].index(site)
        fig, ax = figs.canvas(196, 196, (46, 34, 8, 6))
        cum_axes(ax)
        med, mx, top = stats[site]
        bundle(ax, s, cumulative(z['d_raw'][:, m]), kind, highlight=top)
        if site == 'logits':
            d = z['d_raw'][top, m].astype(np.float64)
            c = np.pad(np.cumsum(d), (1, 0))
            j = int(np.argmax(c[W16:] - c[:-W16]))
            ax.axvspan(s[j], s[j + W16], color=BLUE, alpha=.14, lw=0, zorder=1)
        note(ax, 'fastest 1/16\nmedian ' + figs.text(f'window.b3.{site}.median') + '\nmax ' + figs.text(f'window.b3.{site}.max'))
        figs.save(fig, f'lead_{site}')


def draw_heim(figs, kl, kind, window_cols):
    s = np.linspace(0, 1, 257)
    for start in ('b3', 'b9'):
        fig, ax = figs.canvas(340, 240, (46, 34, 10, 8))
        s_axis(ax)
        ax.set_ylim(-.5, KL_TOP)
        ax.set_yticks([0, 10, 20, 30])
        ax.set_ylabel('KL(start || current), nats')
        bundle(ax, s, np.minimum(kl[start], KL_TOP + .5), kind, highlight=STEEP)
        if start == 'b3':
            ax.axvspan(s[window_cols[0]], s[window_cols[1]], color=BLUE, alpha=.14, lw=0, zorder=1)
        end = np.median(kl[start][kind == 1, -1])
        ax.plot([0, .5, .5, 1], [0, 0, end, end], color=WALL, lw=1.3, ls=(0, (1, 2)), zorder=4)
        ax.text(.515, end + .8, 'illustrative wall', color=WALL, fontsize=9.5, va='bottom', zorder=6)
        note(ax, figs.text(f'heim.clip.{start}') + f' curves above {KL_TOP} nats are cut', x=.02, y=.98, size=9)
        figs.save(fig, f'heim_{start}')


def draw_detours(figs, same_share, diff_share):
    fig, ax = figs.canvas(360, 230, (44, 36, 10, 10))
    x = np.arange(len(ALL_STARTS))
    ax.plot(x, same_share, color=ORANGE, lw=1.6, marker='o', ms=5, zorder=4)
    ax.plot(x, diff_share, color=INK, lw=1.6, marker='o', ms=5, zorder=4)
    ax.set_xticks(x, ALL_STARTS)
    ax.set_xlim(-.3, len(x) - .7)
    ax.set_ylim(0, .62)
    ax.set_yticks([0, .2, .4, .6], ['0%', '20%', '40%', '60%'])
    ax.set_ylabel('share of ' + figs.text('pairs.per_kind') + ' pairs')
    ax.set_xlabel('start site (where the line is drawn)')
    ax.grid(axis='y', color=GRID, lw=.6, zorder=0)
    for xi, (start, a, b) in enumerate(zip(ALL_STARTS, same_share, diff_share)):
        ax.text(xi, a - .035 if a > .05 else a + .025, figs.text(f'leaves.{start}.share'), ha='center',
                va='top' if a > .05 else 'bottom', fontsize=9, color=ORANGE)
        ax.text(xi + .1, b + .02, figs.text(f'third.{start}.share'), ha='left', va='bottom', fontsize=9, color=INK)
    figs.save(fig, 'detours')


def draw_embstep(figs, z3, z9, kind):
    s64 = z3['s64']
    m9 = z3['sites'].index('b9')
    sel = kind == 1

    def emb_steps(z):
        y = z['emb_site_y'][z['chain'].index('b9')].astype(np.float64)
        return np.linalg.norm(np.diff(y, axis=1), axis=-1)
    for name, d in (('raw', z3['d64_raw'][:, m9]), ('b3', emb_steps(z3)), ('b9', emb_steps(z9))):
        fig, ax = figs.canvas(300, 230, (46, 34, 8, 6))
        cum_axes(ax)
        c = cumulative(d.astype(np.float64))
        for i in np.flatnonzero(sel):
            ax.plot(s64, c[i], **DIFF_KW, zorder=2)
        note(ax, 'fastest 1/16: median ' + figs.text({'raw': 'embstep.raw_b3', 'b3': 'embstep.emb_b3', 'b9': 'embstep.emb_b9'}[name]))
        figs.save(fig, f'embstep_{name}')


def draw_thumb(figs, z, share, kl, kind):
    fig = plt.figure(figsize=(400 / 72, 200 / 72))
    m = z['sites'].index('logits')
    for k, (curves, x, ylim) in enumerate(((cumulative(z['d_raw'][:, m]), z['s256'], (-.02, 1.02)),
                                           (np.minimum(kl['b3'], 32.5), z['s256'], (-.5, 32)))):
        ax = fig.add_axes([.04 + .5 * k, .08, .42, .84])
        bundle(ax, x, curves, kind, highlight=int(np.argmax(share[:, m])) if k == 0 else STEEP)
        if k == 0:
            ax.plot([0, 1], [0, 1], color=MUTE, lw=1, ls=(0, (4, 3)))
            ax.plot([0, .5, .5, 1], [0, 0, 1, 1], color=WALL, lw=1.4, ls=(0, (1, 2)))
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
    figs.save(fig, 'thumb', background='white')


def draw_example(figs, Z, kl, r):
    s256 = Z['b3']['s256']
    lab = Z['b3']['labels'][r]
    # endpoint images (32x32, enlarged 4x nearest-neighbour; the page shows them pixelated)
    for j, end in enumerate('ab'):
        im = Image.fromarray(np.ascontiguousarray(Z['b3']['images'][r, j].transpose(1, 2, 0))).resize((128, 128), Image.NEAREST)
        im.save(figs.out.fig_path(f'we{r}_{end}'), 'WEBP', lossless=True, method=6)
    # output change with the predicted-class strip underneath
    fig = plt.figure(figsize=(330 / 72, 270 / 72))
    w, h = 330, 270
    ax = fig.add_axes([46 / w, 96 / h, (330 - 56) / w, (270 - 104) / h])
    strip = fig.add_axes([46 / w, 34 / h, (330 - 56) / w, 54 / h])
    for start in CURVE_STARTS:
        ax.plot(s256, kl[start][r], color=START_COLOUR[start], lw=1.6, ls=(0, (4, 2.5)) if start == 'b9' else '-',
                zorder=3 if start != 'b9' else 4)
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .5, 1], [])
    ax.set_ylim(bottom=-.02 * max(kl[s][r].max() for s in CURVE_STARTS))
    ax.set_ylabel('KL(start || current), nats')
    rows = np.array([[matplotlib.colors.to_rgb(CLASS_COLOUR[p]) for p in Z[s]['pred'][r]] for s in CURVE_STARTS])
    strip.imshow(rows, aspect='auto', extent=(0, 1, len(CURVE_STARTS), 0), interpolation='nearest')
    strip.set_yticks(np.arange(len(CURVE_STARTS)) + .5, CURVE_STARTS)
    strip.tick_params(axis='y', length=0, labelsize=9.5)
    for sp in strip.spines.values():
        sp.set_visible(False)
    s_axis(strip)
    for k, start in enumerate(CURVE_STARTS):
        pred = Z[start]['pred'][r]
        edges = np.r_[0, np.flatnonzero(np.diff(pred)) + 1, len(pred)]
        for e0, e1 in zip(edges[:-1], edges[1:]):
            if (e1 - e0) / len(pred) >= .17:
                c = CLASS_COLOUR[pred[e0]]
                light = np.mean(matplotlib.colors.to_rgb(c)) > .55
                strip.text((s256[e0] + s256[e1 - 1]) / 2, k + .5, CLASSES[pred[e0]], ha='center', va='center', fontsize=9.5,
                           color=INK if light else 'white')
    figs.save(fig, f'we{r}_kl')
    # logit speed and gain
    for name, ylabel in (('speed', 'logit speed (logit units per unit s)'), ('gain', 'change in log gain since s = 0')):
        fig, ax = figs.canvas(330, 230, (50, 34, 10, 8))
        s_axis(ax)
        top = 0
        for start in CURVE_STARTS:
            z = Z[start]
            if name == 'speed':
                y = z['speed'][r, z['sites'].index('logits')]
            else:
                y = z['gain_raw'][r, 0] - z['gain_raw'][r, 0, 0]
            top = max(top, np.max(y))
            ax.plot(z['s64'], y, color=START_COLOUR[start], lw=1.6, ls=(0, (4, 2.5)) if start == 'b9' else '-',
                    zorder=4 if start == 'b9' else 3)
        if name == 'speed':
            ax.set_ylim(0, top * 1.06)
        else:
            ax.axhline(0, color=GRID, lw=.8, zorder=1)
        ax.set_ylabel(ylabel)
        figs.save(fig, f'we{r}_{name}')
    # the line in each site's own diffusion map (modes 1 and 2), drawn at b3; the b9 line from b9 on. The classes the b3
    # line's prediction visits are drawn in colour with their centroids labelled; other probe images are grey.
    z3, z9 = Z['b3'], Z['b9']
    prev = None
    sw = z3['s_switch'][r]
    keep = list(dict.fromkeys(route(z3['pred'][r])))
    for site in EMB_SITES:
        l = z3['chain'].index(site)
        probe = z3['probe_site_y'][l, :, :2].astype(np.float64)
        path = z3['emb_site_y'][l, r, :, :2].astype(np.float64)
        if prev is not None:
            sign = np.where((probe * prev).sum(0) < 0, -1., 1.)
            probe, path = probe * sign, path * sign
        else:
            sign = np.ones(2)
        prev = probe
        fig, ax = figs.canvas(EMB_PX, EMB_PX, (3, 3, 3, 3))
        ax.set_axis_off()
        labels = z3['probe_labels']
        other = ~np.isin(labels, keep)
        ax.scatter(*probe[other].T, s=4, color='#c9c8c2', alpha=.45, lw=0, zorder=1)
        centres = []
        for c in keep:
            sel = labels == c
            ax.scatter(*probe[sel].T, s=9, color=CLASS_COLOUR[c], alpha=.6, lw=0, zorder=2)
            centres.append((c, np.median(probe[sel], axis=0)))
        pts = [path]
        if site in z9['chain']:
            p9 = z9['emb_site_y'][z9['chain'].index(site), r, :, :2].astype(np.float64) * sign
            ax.plot(*p9.T, color=INK, lw=1.3, ls=(0, (4, 2.5)), zorder=4)
            pts.append(p9)
        ax.plot(*path.T, color=BLUE, lw=1.8, zorder=5)
        ax.scatter(*path[::4].T, s=14, color=BLUE, edgecolor='white', lw=.6, zorder=6)
        if np.isfinite(sw):
            k = int(round(sw * 64))
            ax.scatter(*path[k], s=40, marker='D', color=INK, edgecolor='white', lw=.8, zorder=7)
        ax.scatter(*path[0], s=56, facecolor='none', edgecolor=INK, lw=1.5, zorder=8)
        ax.scatter(*path[-1], s=40, marker='s', color=INK, zorder=8)
        # view: the lines and the visited classes' centres, at least 30% of the probe cloud's extent
        span = np.concatenate([*pts, np.array([m for _, m in centres])])
        centre = (span.min(0) + span.max(0)) / 2
        cloud = np.percentile(probe, 99.5, axis=0) - np.percentile(probe, .5, axis=0)
        half = max(.62 * np.ptp(span, axis=0).max(), .15 * cloud.max())
        ax.set_xlim(centre[0] - half, centre[0] + half)
        ax.set_ylim(centre[1] - half, centre[1] + half)
        ax.set_aspect('equal', adjustable='box')
        boxes = []                                                     # greedy label placement, in points
        for c, m in centres:
            ax.scatter(*m, s=70, marker='X', color=CLASS_COLOUR[c], edgecolor='white', lw=1, zorder=9)
            x0, y0 = ax.transData.transform(m)
            w, h = 7.6 * len(CLASSES[c]) + 6, 18
            options = [(6, 4), (6, -22), (-6, 4), (-6, -22), (6, 21), (-6, -39)]

            def box(dx, dy):
                left = x0 + dx if dx > 0 else x0 + dx - w
                return (left, y0 + dy, left + w, y0 + dy + h)

            def fits(bx):
                inside = bx[0] >= 2 and bx[2] <= EMB_PX - 2 and bx[1] >= 2 and bx[3] <= EMB_PX - 2
                return inside and all(bx[2] < o[0] or bx[0] > o[2] or bx[3] < o[1] or bx[1] > o[3] for o in boxes)
            dx, dy = next((o for o in options if fits(box(*o))), options[0])
            boxes.append(box(dx, dy))
            ax.annotate(CLASSES[c], m, xytext=(dx, dy + 2), textcoords='offset points', fontsize=12.5,
                        ha='left' if dx > 0 else 'right', va='bottom', color=INK, zorder=10,
                        bbox=dict(boxstyle='round,pad=0.18', fc='white', ec=CLASS_COLOUR[c], lw=.9, alpha=.9))
        figs.save(fig, f'we{r}_emb_{site}')

if __name__ == '__main__':
    main()
