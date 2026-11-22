"""Local analysis of X4 (interpolation between real activations): xp/out/x4/* -> xp/out/x4_summary.json

    uv run --with numpy --with scipy --with pillow python xp/analyze_x4.py

Array definitions are in the docstring of xp/stage_x4.py.  Kinds: 0 = same-class pair, 1 = different-class pair.
"Relative speed" is |dx_m/ds| / |x_m(1) - x_m(0)|: 1 everywhere for a path that moves uniformly along its chord.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / 'out/x4'
STARTS = ['pixels', 'stem', 'b3', 'b6', 'b8', 'b9']
ALL = ['pixels', 'stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
KIND = ['same', 'diff']
r = lambda a, n=4: np.where(np.isfinite(a), np.round(np.asarray(a, dtype=float), n), None).tolist() if np.ndim(a) else (None if not np.isfinite(a) else round(float(a), n))


def q3(a, axis=0):
    return [r(np.nanquantile(a, q, axis=axis)) for q in (0.25, 0.5, 0.75)]


def aligned(prof, s_sw, grid, half=0.5, nb=41):
    """Mean of prof (P, T) against s - s_switch on a common grid."""
    xs = np.linspace(-half, half, nb)
    rows = [np.interp(xs, grid - s0, p, left=np.nan, right=np.nan) for p, s0 in zip(prof, s_sw)]
    return xs, np.nanmedian(np.array(rows), 0)


def png(a):
    from PIL import Image
    b = io.BytesIO(); Image.fromarray(a.transpose(1, 2, 0)).resize((64, 64), Image.NEAREST).save(b, 'PNG')
    return base64.b64encode(b.getvalue()).decode()


def main():
    from scipy.stats import spearmanr
    out = dict(starts=[], all_sites=ALL, per_start={})
    for l0 in STARTS:
        f = D / f'paths_{l0}.npz'
        if not f.exists():
            continue
        z = np.load(f, allow_pickle=False)
        sites = [str(s) for s in z['sites']]
        chain = [str(s) for s in z['chain']]
        kind = z['kind']
        M = len(sites)
        s64, s256 = z['s64'], z['s256']
        R = dict(sites=sites, chain=chain, n=[int((kind == k).sum()) for k in (0, 1)], stats={}, meta=json.loads(str(z['meta'])))
        relpeak = z['peak_raw'] / z['chord_raw']
        for name, arr in dict(share_raw=z['share_raw'], share_unit=z['share_unit'], sigma_raw=z['sigma_raw'], sigma_unit=z['sigma_unit'],
                              S_raw=z['S_raw'], relpeak_raw=relpeak, ev_raw=z['ev_raw'], ev_unit=z['ev_unit'],
                              width_raw=z['width_raw'], tau_unit=z['tau_unit'], win_unit=z['win_unit']).items():
            R['stats'][name] = {KIND[k]: q3(arr[kind == k]) for k in (0, 1)}
        R['share_l0_unit'] = {KIND[k]: q3(z['share_l0_unit'][kind == k]) for k in (0, 1)}
        R['frac_sigma_lt_half'] = {KIND[k]: r((z['sigma_raw'][kind == k] < 0.5).mean(0)) for k in (0, 1)}
        R['frac_share_gt_quarter'] = {KIND[k]: r((z['share_unit'][kind == k] > 0.25).mean(0)) for k in (0, 1)}
        # refinement
        rr = z['ref_rows']
        R['refine'] = dict(share_unit_absdiff=r(np.nanmedian(np.abs(z['share1024_unit'] - z['share_unit'][rr]), 0)),
                           sigma1024_raw=q3(z['sigma1024_raw']), peak_ratio=r(np.nanmedian(z['peak1024_raw'] / z['peak_raw'][rr], 0)))
        # strata of start-site chord (terciles of the pooled distribution)
        ch = z['chord_l0']
        edges = np.quantile(ch, [0, 1 / 3, 2 / 3, 1]); edges[-1] += 1e-9
        st = []
        for i in range(3):
            m = (ch >= edges[i]) & (ch < edges[i + 1])
            st.append(dict(range=r(edges[i:i + 2]), n=[int((m & (kind == k)).sum()) for k in (0, 1)],
                           share_unit={KIND[k]: r(np.nanmedian(z['share_unit'][m & (kind == k)], 0)) if (m & (kind == k)).sum() >= 5 else None for k in (0, 1)}))
        R['strata'] = st
        R['chord_l0'] = {KIND[k]: q3(ch[kind == k]) for k in (0, 1)}
        # speed profiles (relative), mean over pairs: (M, 65) per kind; unit version too
        rel = z['speed'] / z['chord_raw'][:, :, None]
        relu_ = z['speed_unit'] / z['chord_unit'][:, :, None]
        R['speed_rel'] = {KIND[k]: r(np.nanmean(rel[kind == k], 0), 3) for k in (0, 1)}
        R['speed_unit_rel'] = {KIND[k]: r(np.nanmean(relu_[kind == k], 0), 3) for k in (0, 1)}
        # outputs
        ns, ss = z['n_switch'], z['s_switch']
        R['switches'] = {KIND[k]: np.bincount(np.minimum(ns[kind == k], 5), minlength=6).tolist() for k in (0, 1)}
        R['s_switch_hist'] = np.histogram(ss[(kind == 1) & (ns >= 1)], 20, (0, 1))[0].tolist()
        R['margin_min'] = {KIND[k]: q3(z['margin'][kind == k].min(1)) for k in (0, 1)}
        R['margin_profile'] = {KIND[k]: r(np.median(z['margin'][kind == k][:, ::4], 0), 3) for k in (0, 1)}
        # gains
        gt = ['logits', 'b9'] if l0 != 'b9' else ['logits']
        G = {}
        for gi, g in enumerate(gt):
            for v in ('raw', 'ang'):
                a = z[f'gain_{v}'][:, gi]
                ok = ~z['small_flag'][:, gi] if v == 'ang' else np.ones_like(a, bool)
                a = np.where(ok, a, np.nan)
                G[f'{g}_{v}'] = {KIND[k]: r(np.nanmedian(a[kind == k], 0), 3) for k in (0, 1)}
                G[f'{g}_{v}_range'] = {KIND[k]: q3(np.nanmax(a[kind == k], 1) - np.nanmin(a[kind == k], 1)) for k in (0, 1)}
        G['flag_frac'] = float(z['small_flag'].mean())
        G['conv_p99'] = r(np.nanquantile(z['gain_conv_raw'][:, -1] if l0 != 'b9' else np.zeros(1), 0.99)) if l0 != 'b9' else None
        G['align_raw'] = {KIND[k]: r(np.nanmedian(z['align_raw'][kind == k], 0), 4) for k in (0, 1)}
        G['align_ang'] = {KIND[k]: r(np.nanmedian(z['align_ang'][kind == k], 0), 4) for k in (0, 1)}
        R['gain'] = G
        # aligned on the first switch (different-class paths with exactly one switch)
        one = (kind == 1) & (ns == 1)
        A = dict(n=int(one.sum()))
        if one.sum() >= 10:
            mi = {s: sites.index(s) for s in ('logits', 'b9', 'b6', 'b3') if s in sites}
            for s, i in mi.items():
                xs, y = aligned(relu_[one, i], ss[one], s64)
                A[f'speed_unit_{s}'] = r(y, 3)
            for gi, g in enumerate(gt):
                for v in ('raw', 'ang'):
                    xs, y = aligned(z[f'gain_{v}'][one, gi], ss[one], s64)
                    A[f'gain_{g}_{v}'] = r(y, 3)
            A['x'] = r(xs, 3)
            pk_gain = s64[np.nanargmax(z['gain_ang'][one, 0], 1)]
            pk_speed = s64[np.nanargmax(relu_[one, sites.index('logits')], 1)]
            A['peak_vs_switch'] = dict(
                gain_minus_switch=q3(pk_gain - ss[one]), speed_minus_switch=q3(pk_speed - ss[one]),
                rho_gain=r(spearmanr(pk_gain, ss[one])[0]), rho_speed=r(spearmanr(pk_speed, ss[one])[0]))
            # peak of unit speed at each downstream site against the output switch
            A['site_peak_minus_switch'] = [q3(s64[np.nanargmax(relu_[one, i], 1)] - ss[one]) for i in range(M)]
            A['site_peak_rho'] = [r(spearmanr(s64[np.nanargmax(relu_[one, i], 1)], ss[one])[0]) for i in range(M)]
        anyd = (kind == 1) & (ns >= 1)                             # every different-class path, aligned on its first switch
        A['n_all'] = int(anyd.sum())
        if anyd.sum() >= 10:
            A['site_peak_rho_all'] = [r(spearmanr(s64[np.nanargmax(relu_[anyd, i], 1)], ss[anyd])[0]) for i in range(M)]
        R['aligned'] = A
        R['sigma_min'] = {KIND[k]: r(np.nanmin(z['sigma_raw'][kind == k])) for k in (0, 1)}
        # ReLU patterns
        if len(z['taps']):
            R['taps'] = [str(t) for t in z['taps']]
            R['relu_from0'] = {KIND[k]: r(np.mean(z['relu_from0'][kind == k][:, :, ::4], 0), 4) for k in (0, 1)}
            R['relu_crossings'] = {KIND[k]: q3(z['relu_prev'][kind == k].sum(2)) for k in (0, 1)}
            R['relu_prev_profile'] = {KIND[k]: r(np.mean(z['relu_prev'][kind == k].reshape((kind == k).sum(), len(z['taps']), 64, 4).sum(3), 0), 5) for k in (0, 1)}
        # embedding: class mass along the path, support, site-local path statistics
        mass, lab = z['emb_mass'], z['labels']
        P = len(kind); ar = np.arange(P)
        ma = mass[:, ar, :, lab[:, 0]] if False else np.stack([mass[l][ar, :, lab[:, 0]] for l in range(len(chain))])   # (L, P, 65)
        mb = np.stack([mass[l][ar, :, lab[:, 1]] for l in range(len(chain))])
        dk = kind == 1
        E = dict(mass_a=r(ma[:, dk].mean(1), 3), mass_b=r(mb[:, dk].mean(1), 3), mass_other=r((1 - ma - mb)[:, dk].mean(1), 3),
                 mass_same=r(ma[:, ~dk].mean(1), 3))
        diff = (mb - ma)[:, dk]                                    # crossing point and width of the mass transition
        cross = np.array([[np.interp(0, d, s64) if (np.diff(d) > 0).all() else s64[np.argmax(d > 0)] if (d > 0).any() else np.nan for d in diff[l]] for l in range(len(chain))])
        steep = np.abs(np.diff(diff, axis=2)).max(2) * 64          # largest slope of (mass_b - mass_a) in s
        E['cross_s'] = q3(cross, 1); E['mass_slope'] = q3(steep, 1)
        sup = np.stack([np.searchsorted(np.sort(z['ref_cmax'][l]), z['emb_cmax'][l]) / z['ref_cmax'].shape[1] for l in range(len(chain))])
        E['support'] = {KIND[k]: r(np.median(sup[:, kind == k], 1), 3) for k in (0, 1)}
        E['support_min'] = {KIND[k]: q3(sup[:, kind == k].min(2), 1) for k in (0, 1)}
        dy = np.linalg.norm(np.diff(z['emb_site_y'], axis=2), axis=3)                      # (L, P, 64)
        chord = np.linalg.norm(z['emb_site_y'][:, :, -1] - z['emb_site_y'][:, :, 0], axis=2)
        win = np.stack([dy[:, :, i:i + 4].sum(2) for i in range(61)], 2).max(2) / dy.sum(2)
        E['site_share'] = {KIND[k]: q3(win[:, kind == k], 1) for k in (0, 1)}
        E['site_excess'] = {KIND[k]: q3((dy.sum(2) / chord - 1)[:, kind == k], 1) for k in (0, 1)}
        E['perp_mid'] = {KIND[k]: r(np.median(z['emb_perp'][:, kind == k, 32], 1), 1) for k in (0, 1)}   # row perplexity at s = 0.5
        E['errors'] = dict(replay=float(z['replay_err']), endpoint=float(z['endpoint_err']), cond_max=float(z['ext_cond'][:16].max()))
        R['embedding'] = E
        # pictures: probe cloud and example paths in joint coordinates (first 3 modes) at every chain site
        ex = np.concatenate([np.where(kind == 0)[0][:4], np.where(kind == 1)[0][:6]])
        R['picture'] = dict(probe=np.round(z['probe_y'][:, ::2, :3], 4).tolist(), probe_labels=z['probe_labels'][::2].tolist(),
                            paths=np.round(z['emb_y'][:, ex][:, :, ::2, :3], 4).tolist(), kind=kind[ex].tolist(), labels=lab[ex].tolist(),
                            s=r(s64[::2], 3), pred=z['pred'][ex][:, ::8].astype(int).tolist(),
                            img=[[png(z['images'][i, 0]), png(z['images'][i, 1])] for i in ex],
                            speed_unit=np.round(relu_[ex], 3).tolist())
        R['controls'] = dict(jvp_fd_med=r(z['jvp_fd_med'], 5), jvp_fd1024_med=r(z['jvp_fd1024_med'], 5))
        out['starts'].append(l0); out['per_start'][l0] = R
        print(f'\n== l0 {l0}: sites {sites[0]}..{sites[-1]}  n {R["n"]}  switches same {R["switches"]["same"]} diff {R["switches"]["diff"]}')
        for nm in ('share_unit', 'share_raw', 'sigma_raw', 'relpeak_raw'):
            print(f'  {nm:12s} same {np.round(R["stats"][nm]["same"][1], 3).tolist()}\n  {"":12s} diff {np.round(R["stats"][nm]["diff"][1], 3).tolist()}')
        print('  frac share_unit>0.25 diff', R['frac_share_gt_quarter']['diff'], ' share_l0_unit', R['share_l0_unit']['diff'][1])
        print('  mass slope by chain site (median, diff)', np.round(E['mass_slope'][1], 2).tolist(), '\n  cross_s median', E['cross_s'][1])
        print('  support min along path, diff', E['support_min']['diff'][1], '\n  site_share diff', np.round(E['site_share']['diff'][1], 3).tolist())
        if 'peak_vs_switch' in A:
            print('  aligned n', A['n'], A['peak_vs_switch'], '\n  site peak rho', A['site_peak_rho'])
        print('  gain logits raw (s=0, .5, 1) diff', [G['logits_raw']['diff'][i] for i in (0, 32, 64)], 'range', G['logits_raw_range']['diff'][1],
              ' ang', [G['logits_ang']['diff'][i] for i in (0, 32, 64)], 'align', G['align_raw']['diff'][32])
        if 'relu_crossings' in R:
            print('  relu from0 at s=1 (diff)', [row[-1] for row in R['relu_from0']['diff']], ' same', [row[-1] for row in R['relu_from0']['same']])
        print('  errors', E['errors'], 'strata n', [s['n'] for s in st])
    # probe gains
    f = D / 'probe_gain.npz'
    if f.exists():
        z = np.load(f)
        starts, targets = [str(s) for s in z['starts']], [str(s) for s in z['targets']]
        margin, correct = z['margin'], z['pred'] == z['labels']
        PG = dict(starts=starts, targets=targets, next_sites=[str(s) for s in z['next_sites']], rho={}, deciles={}, by_correct={},
                  margin=r(margin, 3), correct=correct.astype(int).tolist(), labels=z['labels'].tolist())
        dec = np.quantile(margin, np.linspace(0, 1, 11)); dec[-1] += 1e-9
        for si, s in enumerate(starts):
            for ti, t in enumerate(targets):
                for v in ('raw', 'ang'):
                    g = z[f'gain_{v}'][si, ti]
                    key = f'{s}>{t}_{v}'
                    PG['rho'][key] = r(spearmanr(g, margin)[0])
                    PG['deciles'][key] = [r(np.median(g[(margin >= dec[i]) & (margin < dec[i + 1])])) for i in range(10)]
                    PG['by_correct'][key] = [r(np.median(g[correct])), r(np.median(g[~correct]))]
            lr = np.log(z['rad_out'][si, 0])                       # centred logit radius / probe median (output normalisation)
            PG.setdefault('rho_radius', {})[s] = r(spearmanr(lr, margin)[0])
            PG.setdefault('rho_ang_radius_removed', {})[s] = r(spearmanr(z['gain_ang'][si, 0] + lr, margin)[0])
            PG[f'gain_ang_{s}'] = r(z['gain_ang'][si, 0], 3)
            PG[f'gain_raw_{s}'] = r(z['gain_raw'][si, 0], 3)
        PG['decile_edges'] = r(dec, 3)
        PG['conv_p99'] = r(np.nanquantile(z['conv_raw'][:, 1:], 0.99))
        pz = np.load(D / 'paths_stem.npz')
        PG['probe_y'] = {s: np.round(pz['probe_y'][[str(c) for c in pz['chain']].index(s), :, :3], 4).tolist() for s in ('b3', 'b6', 'b9', 'logits')}
        out['probe_gain'] = PG
        print('\n== probe gain: Spearman with margin'); [print('  ', k, v, 'deciles', PG['deciles'][k][0], PG['deciles'][k][4], PG['deciles'][k][9], 'correct/wrong', PG['by_correct'][k]) for k, v in PG['rho'].items()]
    if (D / 'checks.json').exists():
        out['checks'] = json.loads((D / 'checks.json').read_text())
    (D.parent / 'x4_summary.json').write_text(json.dumps(out))


if __name__ == '__main__':
    main()
