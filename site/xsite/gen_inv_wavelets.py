"""Diffusion-wavelet inversion entry (wavelets/SPEC_DW.md, wavelets/NOTES_DW.md): charts and facts.

Reads only the stored stage outputs: data/invert/dw_results.json (per-site summaries and curves, written by
invert/dw_collect.py from data/invert/dw/P8__pr400__L12/<site>.json) and the per-atom localisation arrays in
data/invert/dw/P8__pr400__L12/<site>.npz.  Nothing is recomputed from activations.
"""
from __future__ import annotations

import json
import math

import numpy as np

from xlib import DATA, PALETTE, SITES, Out, dots, heat, lines

XID = 'inv-wavelets'
RES = 'data/invert/dw_results.json'
NPZ = 'data/invert/dw/P8__pr400__L12/{site}.npz'
CONV = [f'b{i}' for i in range(1, 9)]          # conv sites, b1-b8
EARLY = ['stem', *CONV]                          # stem-b8
CELLS = ('joint|64', 'source|64')
SET = PALETTE['sets']
# One colour per decoder, the same in every chart (colour follows the entity).
COL = dict(lead=SET[0], ebest=SET[2], wbest=SET[1], rbest=SET[6], ridge=SET[3], krr=SET[4], ddm=PALETTE['mute'])
NAME = dict(lead='leading n harmonics', ebest='best n eigenvectors', wbest='best n QR wavelets',
            rbest='best n random-rotation atoms', ridge='ridge, same width', krr='stored kernel ridge')
FRAC = '0.05'                                    # amplification probe size: 5% of the probe RMS coordinate radius


def token(name):
    return '{{fact:x.' + XID + '.' + name + '}}'


def sci(x):
    """Two significant figures in e-notation without zero padding, e.g. 6.8e-6."""
    m, e = f'{x:.1e}'.split('e')
    return f'{m}e{int(e)}'


def decades(xs):
    """Explicit log-axis ticks: powers of ten inside the data range."""
    return [10 ** e for e in range(0, 12) if min(xs) <= 10 ** e <= max(xs)]


SUP = str.maketrans('0123456789-', '⁰¹²³⁴⁵⁶⁷⁸⁹⁻')


def a_label(a):
    """Label of an allowed reciprocal A: 1.5, 2, ..., 512, then 2^k, 10^k, or the floor."""
    if a == 1e10:
        return 'floor (10⁻¹⁰)'
    if a < 1000:
        return f'{a:g}'
    k = round(math.log2(a))
    return f'2{str(k).translate(SUP)}' if 2 ** k == a else f'10{str(round(math.log10(a))).translate(SUP)}'


def legend_last(panels):
    """One legend, under the last panel (the series are the same in every panel)."""
    for p in panels[:-1]:
        p['legend'] = False
    return panels


def main():
    R = json.loads((DATA.parent / RES).read_text())
    S = R['per_site']
    assert R['sites'] == SITES and not R['missing'] and not R['rejected'] and not R['partial']
    assert all(not S[s]['failed'] for s in SITES)
    out = Out(XID)

    def fact(name, value, digits=None, *, tier='S', source, pct=False, text=None):
        out.fact(name, value, digits, pct, tier=tier, source=source)
        if text is not None:
            out.facts[name]['text'] = text

    def final(site, cell, key):
        return S[site]['q1'][cell]['finals'][key]

    # ------------------------------------------------------------------ Q1: same 64 coordinates, other basis
    panels_lead = []
    dw_lead, ridge_lead, eb_lead, rb_lead = [], [], [], []
    for cell in CELLS:
        series = {k: dict(values=[], lo=[], hi=[]) for k in ('wbest', 'ebest', 'rbest', 'ridge')}
        for site in SITES:
            for k, key in (('wbest', 'wbest|None'), ('ebest', 'ebest|None'), ('rbest', 'rbest|None'), ('ridge', 'ridge')):
                mean, se = final(site, cell, key)['d_vs_lead']['cos']
                series[k]['values'].append(mean)
                series[k]['lo'].append(mean - 2 * se)
                series[k]['hi'].append(mean + 2 * se)
            dw_lead.append(final(site, cell, 'wbest|None')['d_vs_lead']['cos'][0])
            eb_lead.append(final(site, cell, 'ebest|None')['d_vs_lead']['cos'][0])
            rb_lead.append(final(site, cell, 'rbest|None')['d_vs_lead']['cos'][0])
            ridge_lead.append((site, cell, final(site, cell, 'ridge')['d_vs_lead']['cos'][0]))
        panels_lead.append(dict(title=('joint coordinates' if cell.startswith('joint') else 'source coordinates') + ', K = 64',
                                **dots(SITES, [dict(name=NAME[k] + ' − leading harmonics', color=COL[k], marker=m, **series[k])
                                               for k, m in (('wbest', 'circle'), ('ebest', 'square'), ('rbest', 'triangle'), ('ridge', 'diamond'))],
                                       'Held-out cosine minus that of the leading harmonics', refline=dict(x=0), fmt='3f')))
    out.chart('q1_diff', dict(panels=legend_last(panels_lead), shared=True))
    src_q1 = f'{RES}:per_site.<site>.q1.<cell>.finals.<decoder>.d_vs_lead.cos[0]; all 12 sites, joint|64 and source|64'
    fact('q1.wbest.lo', min(dw_lead), 3, source=src_q1 + '; min over sites and cells, wbest|None')
    fact('q1.wbest.hi', max(dw_lead), 3, source=src_q1 + '; max over sites and cells, wbest|None')
    fact('q1.ebest_rbest.absmax', max(abs(v) for v in eb_lead + rb_lead), 3,
         source=src_q1 + '; max |difference| over ebest|None and rbest|None')
    rl = [v for _, _, v in ridge_lead]
    fact('q1.ridge.lo', min(rl), 3, source=src_q1 + '; min over sites and cells, ridge')
    fact('q1.ridge.hi', max(rl), 3, source=src_q1 + '; max over sites and cells, ridge')
    mid = [v for s, c, v in ridge_lead if c == 'joint|64' and s in ('b4', 'b5', 'b6', 'b7', 'b8')]
    fact('q1.ridge.b4b8_lo', min(mid), 3, source=src_q1 + '; joint|64, b4-b8, min')
    fact('q1.ridge.b4b8_hi', max(mid), 3, source=src_q1 + '; joint|64, b4-b8, max')
    for key, name in (('lead|None', 'lead'), ('wbest|None', 'wbest'), ('ridge', 'ridge'), ('krr', 'krr')):
        fact(f'q1.b4.{name}', final('b4', 'joint|64', key)['cos'][0], 3,
             source=f'{RES}:per_site.b4.q1.joint|64.finals.{key}.cos[0]; mean over H')
    fact('q1.b4.n_lead', final('b4', 'joint|64', 'lead|None')['n_eff'], source=f'{RES}:per_site.b4.q1.joint|64.finals.lead|None.n_eff')
    fact('q1.b4.n_wbest', final('b4', 'joint|64', 'wbest|None')['n_eff'], source=f'{RES}:per_site.b4.q1.joint|64.finals.wbest|None.n_eff')

    # Admissible bands: A <= 1e2 against sigma_1 of the second-stage Gram.
    ratio = [S[s]['q1']['joint|64'] and S[s]['spectrum']['joint|64']['levels'][0]['lam_next'] for s in EARLY]
    fact('q1.sigma2.lo', min(ratio), source=f'{RES}:per_site.<site>.spectrum.joint|64.levels[0].lam_next (sigma_2/sigma_1); stem-b8, min',
         text=sci(min(ratio)))
    fact('q1.sigma2.hi', max(ratio), source=f'{RES}:per_site.<site>.spectrum.joint|64.levels[0].lam_next; stem-b8, max',
         text=sci(max(ratio)))
    one = sum(final(s, 'joint|64', 'wbest|100.0')['n_eff'] == 1 for s in EARLY)
    fact('q1.adm100.one', one, source=f'{RES}:per_site.<site>.q1.joint|64.finals.wbest|100.0.n_eff == 1; count over stem-b8')
    fact('q1.adm100.wbest_b9', final('b9', 'joint|64', 'wbest|100.0')['cos'][0], 3,
         source=f'{RES}:per_site.b9.q1.joint|64.finals.wbest|100.0.cos[0]')
    fact('q1.adm100.wbest_logits', final('logits', 'joint|64', 'wbest|100.0')['cos'][0], 3,
         source=f'{RES}:per_site.logits.q1.joint|64.finals.wbest|100.0.cos[0]')

    # Q1 curves along n (descriptive, on H), joint|64, with the ridge and stored krr as marks at their df.
    def q1_curves(site, metric):
        q = S[site]['q1']['joint|64']
        cur = q['curves']
        gcv = q['finals']['lead@gcv']
        xs = sorted({p['n'] for p in cur['lead|None']} | {round(q['finals']['ridge']['df'], 1), round(q['finals']['krr']['df'], 1),
                                                         gcv['n_eff']})
        ser = []
        for k in ('lead', 'ebest', 'wbest', 'rbest'):
            pts = {p['n']: p[metric] for p in cur[f'{k}|None']}
            ser.append(dict(name=NAME[k], color=COL[k], y=[pts.get(x) for x in xs], join=True, width=1.6 if k != 'lead' else 2.2))
        for k in ('ridge', 'krr'):
            f = q['finals'][k]
            ser.append(dict(name=NAME[k] + ' (at its effective df)', color=COL[k], noline=True, dots=5,
                            y=[f['cos' if metric == 'cos' else 'agree'][0] if x == round(f['df'], 1) else None for x in xs]))
        ser.append(dict(name='stored double-diffusion GCV lift (at its rank)', color=COL['ddm'], noline=True, dots=5,
                        y=[gcv['cos' if metric == 'cos' else 'agree'][0] if x == gcv['n_eff'] else None for x in xs]))
        return xs, ser

    for site in ('b4', 'pool'):
        pan = []
        for metric, lab in (('cos', 'Mean held-out cosine'), ('agree', 'Patched top-1 agreement')):
            xs, ser = q1_curves(site, metric)
            pan.append(dict(title=f'{site} · {lab.lower()}', **lines(xs, ser, 'n functions kept (log)', lab, xscale='log',
                                                                    yrange=None, xname='n =', fmt='3f', xticks=decades(xs))))
        out.chart('q1_curves' if site == 'b4' else f'q1_curves_{site}', dict(panels=legend_last(pan)))

    # Error amplification at the cosine-best and agreement-best n along lead (joint|64), descriptive.
    amp_c, amp_a, ratio_amp = [], [], {}
    for site in SITES:
        cur = S[site]['q1']['joint|64']['curves']['lead|None']
        ic = max(range(len(cur)), key=lambda i: (cur[i]['cos'], -cur[i]['n']))
        ia = max(range(len(cur)), key=lambda i: (cur[i]['agree'], -cur[i]['n']))
        amp_c.append(cur[ic][f'amp{FRAC}_med'])
        amp_a.append(cur[ia][f'amp{FRAC}_med'])
        ratio_amp[site] = cur[ia][f'amp{FRAC}_med'] / cur[ic][f'amp{FRAC}_med']
    out.chart('amp', dots(SITES, [dict(name='lift with the highest cosine', values=amp_c, color=COL['lead'], marker='circle'),
                                  dict(name='lift with the highest top-1 agreement', values=amp_a, color=COL['lead'], marker='diamond', hollow=True)],
                          'Median error amplification (activation spread per coordinate spread)', fmt='2f'))
    src_amp = (f'{RES}:per_site.<site>.q1.joint|64.curves.lead|None; amp{FRAC}_med at the argmax of agree over argmax of cos '
               '(lowest n on ties)')
    mid = [ratio_amp[s] for s in ('b4', 'b5', 'b6')]
    fact('amp.b4b6.lo', min(mid), 1, source=src_amp + '; b4-b6 min')
    fact('amp.b4b6.hi', max(mid), 1, source=src_amp + '; b4-b6 max')

    # ------------------------------------------------------------------ Q2: band(A) of the site's own walk
    A_all = sorted({lv['A'] for s in SITES for lv in S[s]['q2']['levels'] if lv['A'] is not None})
    metr = (('cos', 'Mean held-out cosine', lambda lv: lv['cos'][0]),
            ('share', 'Share of probe activation energy in the band', lambda lv: lv['share']),
            ('agree', 'Patched top-1 agreement', lambda lv: lv['agree'][0]))
    q2pan = []
    for key, lab, get in metr:
        ser = []
        for site in SITES:
            by = {lv['A']: get(lv) for lv in S[site]['q2']['levels'] if lv['A'] is not None}
            ser.append(dict(name=site, y=[by.get(a) for a in A_all], join=True, width=1.6))
        q2pan.append(dict(title=lab, **lines(A_all, ser, 'Allowed spectral reciprocal A (log)', '', xscale='log', yrange=None,
                                             colors='ordinal', xname='A =', fmt='3f', xticklabels=[a_label(a) for a in A_all],
                                             xticks=[a for a in A_all if a in (2, 32, 2 ** 10, 2 ** 15, 2 ** 20, 2 ** 25, 1e10)])))
    out.chart('q2', dict(panels=legend_last(q2pan)))
    z = []
    for site in SITES:
        by = {lv['A']: lv['rt']['total'] for lv in S[site]['q2']['levels'] if lv['A'] is not None}
        z.append([by.get(a) for a in A_all])
    out.chart('q2_rt', heat(z, SITES, [a_label(a) for a in A_all],
                            'Round trip (squared miss / coordinate variance); 1 = returning the held-out coordinate mean, zero scores about 1; above 2 drawn as 2',
                            'Site', 'Allowed spectral reciprocal A', zrange=[0, 2], diverging=True, fmt='2f'))

    def lv_at(site, A):
        return next(lv for lv in S[site]['q2']['levels'] if lv['A'] == A)

    src_q2 = f'{RES}:per_site.<site>.q2.levels[A=...]'
    k2 = [lv_at(s, 2.0)['k'] for s in EARLY]
    c2 = [lv_at(s, 2.0)['cos'][0] for s in EARLY]
    rt2 = [lv_at(s, 2.0)['rt']['total'] for s in CONV]
    fact('q2.a2.k_lo', min(k2), source=src_q2 + '.k; A = 2, stem-b8 min')
    fact('q2.a2.k_hi', max(k2), source=src_q2 + '.k; A = 2, stem-b8 max')
    fact('q2.a2.cos_lo', min(c2), 2, source=src_q2 + '.cos[0]; A = 2, stem-b8 min')
    fact('q2.a2.cos_hi', max(c2), 2, source=src_q2 + '.cos[0]; A = 2, stem-b8 max')
    fact('q2.a2.rt_lo', min(rt2), 2, source=src_q2 + '.rt.total; A = 2, b1-b8 min')
    fact('q2.a2.rt_hi', max(rt2), 2, source=src_q2 + '.rt.total; A = 2, b1-b8 max')
    fact('q2.a2.rt_stem', lv_at('stem', 2.0)['rt']['total'], 2, source=src_q2 + '.rt.total; A = 2, stem')
    zero = [lv_at(s, 2.0)['rt']['ref_zero'] for s in SITES]
    fact('q2.ref_zero.lo', min(zero), 2, source=src_q2 + '.rt.ref_zero; A = 2, all sites min')
    fact('q2.ref_zero.hi', max(zero), 2, source=src_q2 + '.rt.ref_zero; A = 2, all sites max')
    c32 = [lv_at(s, 32.0)['cos'][0] for s in CONV]
    k32 = [lv_at(s, 32.0)['k'] for s in CONV]
    rt32 = [lv_at(s, 32.0)['rt']['total'] for s in CONV]
    fact('q2.a32.cos_lo', min(c32), 2, source=src_q2 + '.cos[0]; A = 32, b1-b8 min')
    fact('q2.a32.cos_hi', max(c32), 2, source=src_q2 + '.cos[0]; A = 32, b1-b8 max')
    fact('q2.a32.k_lo', min(k32), source=src_q2 + '.k; A = 32, b1-b8 min')
    fact('q2.a32.k_hi', max(k32), source=src_q2 + '.k; A = 32, b1-b8 max')
    fact('q2.a32.rt_lo', min(rt32), 0, source=src_q2 + '.rt.total; A = 32, b1-b8 min')
    fact('q2.a32.rt_hi', max(rt32), 0, source=src_q2 + '.rt.total; A = 32, b1-b8 max')
    # Where each criterion peaks (on the partition grid, descriptive).
    best = {}
    for site in CONV:
        lv = [l for l in S[site]['q2']['levels'] if l['A'] is not None]
        best[site] = dict(cos=max(lv, key=lambda l: l['cos'][0])['A'], agree=max(lv, key=lambda l: (l['agree'][0], -l['A']))['A'],
                          rt=min(lv, key=lambda l: l['rt']['total'])['A'])
    out.chart('q2_best', dict(sites=CONV, **{k: [best[s][k] for s in CONV] for k in ('cos', 'agree', 'rt')}))
    same = sum(len({best[s]['cos'], best[s]['agree'], best[s]['rt']}) == 1 for s in CONV)
    fact('q2.common', same, source=f'{RES}:per_site.<site>.q2.levels; b1-b8 sites where argmax cos, argmax agree and argmin rt.total '
                                   'over the partition levels coincide')

    # ------------------------------------------------------------------ Q3: compression
    def e256(site, key):
        e = S[site]['q3']['energy'][key]
        return e['share'][e['n'].index(256)]

    gap, qr_rand = [], []
    for site in SITES:
        eig, qr = e256(site, 'eig|None'), e256(site, 'qr|None')
        rand = float(np.mean([e256(site, f'rand{i}|None') for i in range(5)]))
        gap.append(eig - qr)
        qr_rand.append(qr - rand)
    out.chart('q3_256', dots(SITES, [dict(name='best n eigenvectors − best n QR wavelets', values=gap, color=COL['ebest'], marker='square'),
                                     dict(name='best n QR wavelets − mean of 5 random rotations', values=qr_rand, color=COL['rbest'], marker='triangle')],
                             'Difference in probe energy share at n = 256', refline=dict(x=0), fmt='4f'))
    src_q3 = f'{RES}:per_site.<site>.q3.energy.<basis>|None at n = 256 (share of |X_c|^2)'
    fact('q3.eig_wins', sum(g > 0 for g in gap), source=src_q3 + '; sites where eig > qr')
    fact('q3.gap_max', max(gap), 3, source=src_q3 + '; max of eig - qr over sites')
    fact('q3.qr_rand.lo', min(qr_rand), 4, source=src_q3 + '; min over sites of qr - mean(rand0..4)')
    fact('q3.qr_rand.hi', max(qr_rand), 4, source=src_q3 + '; max over sites of qr - mean(rand0..4)')
    q3pan = []
    for site in ('b4', 'b7', 'logits'):
        e = S[site]['q3']['energy']
        xs = e['first']['n']
        ser = [dict(name='first n eigenvectors', color=PALETTE['mute'], dash=True, y=e['first']['share'])]
        for key, k in (('eig|None', 'ebest'), ('qr|None', 'wbest')):
            by = dict(zip(e[key]['n'], e[key]['share']))
            ser.append(dict(name=NAME[k], color=COL[k], y=[by.get(x) for x in xs], join=True))
        rmean = [float(np.mean([dict(zip(e[f'rand{i}|None']['n'], e[f'rand{i}|None']['share'])).get(x, np.nan) for i in range(5)]))
                 for x in xs]
        ser.append(dict(name='best n random-rotation atoms (mean of 5)', color=COL['rbest'], y=rmean, join=True))
        q3pan.append(dict(title=site, **lines(xs, ser, 'n atoms kept (log)', 'Share of probe activation energy', xscale='log',
                                              yrange=[0, 1], xname='n =', fmt='3f', xticks=decades(xs))))
    out.chart('q3_curves', dict(panels=legend_last(q3pan), shared=True))

    # ------------------------------------------------------------------ localisation (site operator, per atom)
    n90 = {b: [] for b in ('qr', 'eig', 'rand')}
    near = {b: [] for b in ('qr', 'eig', 'rand')}
    n90_all = {b: [] for b in ('qr', 'eig', 'rand')}
    for site in SITES:
        with np.load(DATA.parent / NPZ.format(site=site)) as zf:
            for b in ('qr', 'eig'):
                n90[b].append(float(np.median(zf[f'loc|site|{b}|n90'])))
                near[b].append(float(np.median(zf[f'loc|site|{b}|near_peak'])))
            n90['rand'].append(float(np.mean([np.median(zf[f'loc|site|rand{i}|n90']) for i in range(5)])))
            near['rand'].append(float(np.mean([np.median(zf[f'loc|site|rand{i}|near_peak']) for i in range(5)])))
            for op in ('site', 'joint|64', 'source|64'):
                for b in ('qr', 'eig'):
                    n90_all[b].append(float(np.median(zf[f'loc|{op}|{b}|n90'])))
                n90_all['rand'].append(float(np.mean([np.median(zf[f'loc|{op}|rand{i}|n90']) for i in range(5)])))
    loc_ser = lambda d: [dict(name='QR wavelets', values=d['qr'], color=COL['wbest'], marker='circle'),
                            dict(name='eigenvectors', values=d['eig'], color=COL['ebest'], marker='square'),
                            dict(name='random rotations (mean of 5)', values=d['rand'], color=COL['rbest'], marker='triangle')]
    out.chart('loc', dict(panels=legend_last([
        dict(title='Points holding 90% of an atom’s squared mass (median over atoms, of 8,000)',
             **dots(SITES, loc_ser(n90), 'Number of probe points', fmt='int')),
        dict(title='Squared mass on the atom’s peak and its 400 nearest probe points (median)',
             **dots(SITES, loc_ser(near), 'Share of squared mass', fmt='2f'))])))
    src_loc = 'data/invert/dw/P8__pr400__L12/<site>.npz:loc|<op>|<basis>|n90; median over atoms; ops site, joint|64, source|64; 12 sites'
    fact('loc.qr.lo', min(n90_all['qr']), 0, tier='P', source=src_loc + '; min over sites and operators, qr')
    fact('loc.qr.hi', max(n90_all['qr']), 0, tier='P', source=src_loc + '; max, qr')
    fact('loc.eig.lo', min(n90_all['eig']), 0, tier='P', source=src_loc + '; min, eig')
    fact('loc.eig.hi', max(n90_all['eig']), 0, tier='P', source=src_loc + '; max, eig')
    fact('loc.near.qr_lo', min(near['qr']), 2, tier='P', source='…/<site>.npz:loc|site|qr|near_peak; median over atoms, site operator, min over sites')
    fact('loc.near.qr_hi', max(near['qr']), 2, tier='P', source='…/<site>.npz:loc|site|qr|near_peak; max over sites')
    fact('loc.near.eig_lo', min(near['eig']), 2, tier='P', source='…/<site>.npz:loc|site|eig|near_peak; min over sites')
    fact('loc.near.eig_hi', max(near['eig']), 2, tier='P', source='…/<site>.npz:loc|site|eig|near_peak; max over sites')

    # ------------------------------------------------------------------ the stored double-diffusion curve (b4, joint|64)
    ddm = R['ddm_curve']['b4']
    cur = S['b4']['q1']['joint|64']['curves']['lead|None']
    xs = sorted({p['n'] for p in cur} | set(ddm['cuts']))
    pan = []
    for metric, lab in (('cos', 'Mean held-out cosine'), ('agree', 'Patched top-1 agreement')):
        this = {p['n']: p[metric] for p in cur}
        old = dict(zip(ddm['cuts'], ddm[metric]))
        pan.append(dict(title=lab, **lines(xs, [dict(name='leading n harmonics (this run)', color=COL['lead'], y=[this.get(x) for x in xs], join=True),
                                                dict(name='stored double-diffusion curve', color=COL['ddm'], dash=True, dots=4,
                                                     y=[old.get(x) for x in xs], join=True)],
                                          'n harmonics kept (log)', lab, xscale='log', yrange=None, xname='n =', fmt='3f',
                                          xticks=decades(xs))))
    out.chart('ddm', dict(panels=legend_last(pan)))
    ic = max(range(len(cur)), key=lambda i: (cur[i]['cos'], -cur[i]['n']))
    ia = max(range(len(cur)), key=lambda i: (cur[i]['agree'], -cur[i]['n']))
    src_c = f'{RES}:per_site.b4.q1.joint|64.curves.lead|None'
    fact('ddm.b4.cos_n', cur[ic]['n'], source=src_c + '; n at argmax cos')
    fact('ddm.b4.cos', cur[ic]['cos'], 3, source=src_c + '; max cos')
    fact('ddm.b4.agree_n', cur[ia]['n'], source=src_c + '; n at argmax agree')
    fact('ddm.b4.agree', cur[ia]['agree'], 3, source=src_c + '; max agree')
    fact('ddm.b4.r', S['b4']['q1']['joint|64']['r'], source=f'{RES}:per_site.b4.q1.joint|64.r (dictionary size at the 1e-10 floor)')

    # ------------------------------------------------------------------ controls
    span = max(max(S[s]['checks']['q2_qr_vs_eig_rel'], S[s]['checks']['q2_rand_vs_eig_rel'],
                   *(v for c in CELLS for v in S[s]['q1'][c]['checks']['complete_band_rel'].values())) for s in SITES)
    fact('ctl.span', span, source=f'{RES}:per_site.<site>.checks.q2_*_vs_eig_rel and q1.<cell>.checks.complete_band_rel; max',
         text=sci(span))

    # ------------------------------------------------------------------ the viewer's wavelet mode (a different operator)
    V = json.loads((DATA / 'wavelets/joint_dw_64000.json').read_text())
    vsrc = 'data/wavelets/joint_dw_64000.json'
    lv = V['levels']
    fact('viewer.ranks', ', '.join(str(l['rank']) for l in lv), source=vsrc + ':levels[].rank')
    fact('viewer.t', ', '.join(f"{l['t']:.3f}" for l in lv), source=vsrc + ':levels[].t (ln 2 / mu_c)')
    fact('viewer.tstar', V['operator']['t_star'], 3, source=vsrc + ':operator.t_star')
    fact('viewer.ship', V['ship'], source=vsrc + ':ship')
    fact('viewer.n', V['operator']['n'] * len(V['operator']['sites']), source=vsrc + ':operator.n × len(operator.sites)')
    for key, name in (('n90_median', 'qr'), ('eig_n90_median', 'eig')):
        vals = [l[key] for l in lv]
        fact(f'viewer.{name}.lo', min(vals), 0, source=vsrc + f':levels[].{key}; min')
        fact(f'viewer.{name}.hi', max(vals), 0, source=vsrc + f':levels[].{key}; max')
    fact('viewer.gap53', lv[4]['rel_gap'], 1, pct=True, source=vsrc + ':levels[4].rel_gap (rank 53)')
    fact('viewer.gap93', lv[5]['rel_gap'], 1, pct=True, source=vsrc + ':levels[5].rel_gap (rank 93)')

    card = dict(answer='Decoding the same 64 coordinates with the best-n diffusion wavelets instead of the leading-n harmonics (each at its own split-chosen n) changes '
                       f'held-out cosine by {token("q1.wbest.lo")} to {token("q1.wbest.hi")}; kernel ridge stays ahead.',
                numbers=[f'Wavelets − harmonics: {token("q1.wbest.lo")} to {token("q1.wbest.hi")} cosine',
                         f'Ridge − harmonics: {token("q1.ridge.lo")} to {token("q1.ridge.hi")}',
                         f'Band at A = 2: cosine {token("q2.a2.cos_lo")}–{token("q2.a2.cos_hi")} at stem–b8',
                         f'Eigenvectors hold more energy than wavelets at {token("q3.eig_wins")} of 12 sites'])
    thumb = out.fig_path('thumb')
    if thumb.exists():
        card['thumb'] = f'assets/fig/{thumb.name}'
    out.chart('card', card)
    out.write()
    for k, v in out.facts.items():
        print(f'{k:24s} {v["text"]:>10s}  {v["value"]}')


if __name__ == '__main__':
    main()
