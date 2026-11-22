"""Collect data/invert/dw/<tag>/<site>.json into one compact data/invert/dw_results.json (wavelets/SPEC_DW.md).

Run: uv run --with numpy --with scipy --with torch python invert/dw_collect.py [--partial]   (after: modal_invert.py fetch dw/P8__pr400__L12 data/invert/dw/P8__pr400__L12)

A site is accepted only if its JSON has the current protocol, an empty `failed` list, all parts and both cells, and
its npz exists, and the current stage_dw.gate (re-applied here to the stored JSON) finds no failure; otherwise the
run stops (with --partial: the site is listed under `rejected` with the reason).

Kept per site: checks, spectrum tables, Q1 (selection, finals, curves, localisation summaries), Q2 levels, Q3
energy and decode curves, site localisation summaries, timing and memory per part.  Per-image and per-atom arrays
stay in the npz files.  Also copied: the stored double-diffusion curve along the cut-off (joint|64, at the stored
GCV width) from ddm_results.json, for the reconstruction-against-function comparison.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / 'data/invert'
sys.path[:0] = [str(Path(__file__).resolve().parent), str(ROOT.parent.parent / 'wavelets')]
from stage_dw import gate  # noqa: E402
TAG = 'P8__pr400__L12'
SITES = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
CELLS = ('joint|64', 'source|64')
PROTOCOL = 'dw-inv-3'
PARTS = {'q1', 'q2', 'q3', 'loc'}


def slim_loc(loc):
    """Localisation: overall medians per basis, and per-band medians for the QR basis, eigenvectors and rand0."""
    if not loc:
        return None
    out = dict(bands_A=loc.get('bands_A'))
    for k, v in loc.items():
        if k == 'bands_A':
            continue
        out[k] = dict(n90_med=v['n90_med'], near_peak_med=v['near_peak_med'], near_seed_med=v['near_seed_med'])
        if k in ('qr', 'eig', 'rand0'):
            out[k]['bands'] = v['bands']
    return out


def main():
    partial = '--partial' in sys.argv
    R, missing, rejected = {}, [], {}
    for s in SITES:
        f = ROOT / 'dw' / TAG / f'{s}.json'
        if not f.exists():
            missing.append(s)
            continue
        r = json.loads(f.read_text())
        why = []
        if r.get('protocol') != PROTOCOL:
            why.append(f"protocol {r.get('protocol')}")
        if r.get('failed') != []:
            why.append(f"failed {r.get('failed')}")
        else:
            try:
                regate = gate(r)
            except (KeyError, TypeError) as e:
                regate = [f'gate error {e!r}']
            if regate:
                why.append(f'current gate: {regate[:10]}')
        if not PARTS <= set(r.get('parts_run', [])) or not set(CELLS) <= set(r.get('cells_run', [])):
            why.append('incomplete parts/cells')
        if not f.with_suffix('.npz').exists():
            why.append('no npz')
        if why:
            rejected[s] = why
        else:
            R[s] = r
    if (rejected or missing) and not partial:
        raise SystemExit(f'not collecting: missing {missing}, rejected {rejected} (use --partial for a pilot)')
    sites = [s for s in SITES if s in R]
    if not sites:
        raise SystemExit(f'no acceptable site: missing {missing}, rejected {rejected}')
    ddm = json.loads((ROOT / 'ddm_results.json').read_text()) if (ROOT / 'ddm_results.json').exists() else None
    any_ = R[sites[0]]
    out = dict(tag=TAG, sites=sites, missing=missing, rejected=rejected, partial=bool(missing or rejected), n=any_['n'], m=any_['m'], A_list=any_['A_list'], floor=any_['floor'],
               adm=any_['adm'], fracs=any_['fracs'], fixed=any_['fixed'], per_site={}, ddm_curve={})
    gpu_s = 0.0
    for s in sites:
        r = R[s]
        e = dict(D=r['D'], sX=r['sX'], near_probe=r['near_probe'], near_probe_any_site=r.get('near_probe_any_site'),
                 failed=r['failed'], checks=r['checks'], spectrum=r['spectrum'],
                 site=r.get('site'), parts=r['parts'], seconds=r['seconds'], gpu=r['gpu'], q2=r.get('q2'), q3=r.get('q3'),
                 loc_site=slim_loc(r.get('loc_site')), q1={})
        gpu_s += r['seconds']
        for name, q in r.get('q1', {}).items():
            e['q1'][name] = {k: v for k, v in q.items() if k != 'loc'}
            e['q1'][name]['sel'] = {k: {kk: vv for kk, vv in v.items() if kk not in ('grid', 'curve')} | dict(
                curve=list(zip(v['grid'], v['curve'])) if 'grid' in v else v['curve']) for k, v in q['sel'].items()}
            e['q1'][name]['loc'] = slim_loc(q.get('loc'))
        out['per_site'][s] = e
        if ddm and s in ddm.get('func', {}).get('joint|64', {}):
            h = r['q1']['joint|64']['h_mult'] if 'joint|64' in r.get('q1', {}) else None
            fc = ddm['func']['joint|64'][s]
            key = str(h) if str(h) in fc else (h if h in fc else None)
            if key is not None:
                c = fc[key]['cut']
                out['ddm_curve'][s] = dict(h_mult=h, cuts=c['cuts'], cos=c['cos'], agree=c['agree'], kl_mean=c['kl_mean'])
    out['seconds_total'] = gpu_s
    f = ROOT / 'dw_results.json'
    f.write_text(json.dumps(out))
    print('wrote', f, f'{f.stat().st_size / 1e6:.1f} MB', 'sites', len(sites), 'missing', missing,
          'container-seconds', round(gpu_s))
    return out


if __name__ == '__main__':
    main()
