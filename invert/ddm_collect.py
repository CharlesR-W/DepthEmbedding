"""Collect data/invert/ddm/<tag>/<site>.json into one compact data/invert/ddm_results.json for the page.

Run: python3 invert/ddm_collect.py      (after: modal_invert.py fetch ddm data/invert/ddm P8__pr400__L12)
"""
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / 'data/invert'
TAG = 'P8__pr400__L12'
SITES = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']


def main():
    R = {s: json.loads((ROOT / 'ddm' / TAG / f'{s}.json').read_text()) for s in SITES}
    prior = json.loads((ROOT / 'results.json').read_text())['decode'][TAG]
    out = dict(sites=SITES, tag=TAG, n=R['b4']['n'], hs=R['b4']['hs'], ridges=R['b4']['ridges'], checks={}, finals={}, sel={},
               func={}, surface={}, llr={}, later={}, rt_refs={}, ref={}, site_lam={}, seconds={}, gpu={}, prior={})
    for s in SITES:
        r = R[s]
        out['checks'][s] = dict(r['checks'], krr_parity={K: r['combos'][f'joint|{K}']['finals']['krr']['cos'][0]
                                                             - prior[s][f'H|{K}|krr|cos_raw'][0] for K in (16, 64, 256)})
        out['prior'][s] = {f'{d}|{K}': dict(cos=prior[s][f'H|{K}|{d}|cos_raw'][0], agree=prior[s].get(f'H|{K}|{d}|agree', [None])[0])
                           for K in (16, 64, 256) for d in ('gh', 'krr', 'mlp', 'pca') if f'H|{K}|{d}|cos_raw' in prior[s]}
        out['seconds'][s], out['gpu'][s] = r['seconds'], r['gpu']
        out['llr'][s] = {k: r['llr'][k] for k in ('site', 'joint', 'neff_site', 'neff_joint', 'sel_site', 'sel_joint')}
        out['site_lam'][s] = [r['site_lam'][k] for k in (0, 3, 15, 63, 255)]
        out['ref'][s] = r['ref']['step']
        for name, c in r['combos'].items():
            if 'llr16' in name:                      # stage_ddm re-embeds the first 16 modes, not the selected 16
                for e in c['finals'].values():
                    e.pop('rt_joint', None)
            out['finals'].setdefault(name, {})[s] = c['finals']
            out['sel'].setdefault(name, {})[s] = dict(krr=c.get('krr_sel'), split={k: v for k, v in (c.get('split_sel') or {}).items()},
                                                      fsplit={k: v for k, v in (c.get('fsplit_sel') or {}).items() if k != 'table'},
                                                      gcv=c['gcv_sel'], oracle_cos=c['oracle_cos'], oracle_cap=c.get('oracle_cap'))
            if c.get('rt_refs'):
                out['rt_refs'].setdefault(name, {})[s] = c['rt_refs']
            if c.get('func'):
                out['func'].setdefault(name, {})[s] = c['func']
            if 'later' in c:
                L = c['later']
                out['later'].setdefault(name, {})[s] = {a: (dict(r2=v['r2'], rho2=v['rho2'], h=v['h'], m=v.get('m'), ridge=v.get('ridge'))
                                                            if isinstance(v, dict) else v) for a, v in L.items()}
            if name in ('joint|16', 'joint|64', 'site|64', 'joint|256'):          # cosine and GCV along the cut-off, per h
                out['surface'].setdefault(name, {})[s] = {hm: dict(cuts=v['cuts'], H_cos=v['lh']['H_cos'], H_cap_med=v['lh']['H_cap_med'],
                                                                   gcv=v['lh']['gcv'], kr_H_cos=v['kr']['H_cos'], kr_gcv=v['kr']['gcv'])
                                                          for hm, v in c['surface'].items()}
    (ROOT / 'ddm_results.json').write_text(json.dumps(out))
    print('wrote', ROOT / 'ddm_results.json', f'{(ROOT / "ddm_results.json").stat().st_size / 1e6:.1f} MB',
          'gpu-seconds', round(sum(out['seconds'].values())))
    return out


if __name__ == '__main__':
    main()
