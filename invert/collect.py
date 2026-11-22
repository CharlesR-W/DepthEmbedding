"""Merge every fetched summary into data/invert/results.json, the single input of the results page.

Run after `modal_invert.py summary` and `fetch decode|rt ... summary.json|.json`:
    uv run --with numpy python invert/collect.py
Schema:
  sites            the 12 site names
  kernel           p0_kernel.json: per site, per kernel: hub{max,skew,zero}, lam (eigenvalues 1,2,4,8,16,32,64),
                   stab{k: reproducibility on a disjoint half}, perp; plus raw_cos_hub
  refit            per '<kernel>|<K>': probe / held / onestep share reproduced per site (N = 2,000 probe)
  decode[tag][site]['<target>|<K>|<decoder>|<metric>'] = [mean, standard error, n]; site '_meta' holds the tuned
                   hyperparameters.  tag = '<probe>__<kernel>__L12'.  Metrics: cos, cap (unit target), cos_raw,
                   cap_raw (raw target, scale predicted), kl / agree (patched, predicted scale), kl_on / agree_on
                   (patched, true norm supplied)
  rt[tag]['<site>__K<K>'][key] = mean over held-out images (keys documented in stage_rt.py)
  adv[tag_K]       adversarial trajectories (adv.py): keys pgd2, pgd8 (joint coordinates) and pgd2_src, pgd8_src
                   (each site's own source coordinates)
  modeloc[tag]     per mode (first 64): top2 = share of squared norm on the two heaviest images, part = participation
                   share; n = probe size (modeloc.py)
"""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / 'data/invert'
S = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
out = dict(sites=S, kernel=json.loads((ROOT / 'p0_kernel.json').read_text()), decode={}, rt={}, adv={})
if (ROOT / 'refit.json').exists():
    out['refit'] = json.loads((ROOT / 'refit.json').read_text())
for d in sorted((ROOT / 'decode').glob('*')):
    f = d / 'summary.json'
    if not f.exists():
        continue
    raw = json.loads(f.read_text())
    out['decode'][d.name] = {}
    for unit, vals in raw.items():
        site, tsite = unit.split('__')
        key = site if site == tsite else f'{site}>{tsite}'
        out['decode'][d.name][key] = vals
for d in sorted((ROOT / 'rt').glob('*')):
    out['rt'][d.name] = {f.stem: json.loads(f.read_text()) for f in sorted(d.glob('*__K*.json'))}
for f in sorted(ROOT.glob('adv_*.json')):
    out['adv'][f.stem[4:]] = json.loads(f.read_text())
if (ROOT / 'modeloc.json').exists():
    out['modeloc'] = json.loads((ROOT / 'modeloc.json').read_text())
(ROOT / 'results.json').write_text(json.dumps(out))
print({k: (len(v) if hasattr(v, '__len__') else v) for k, v in out.items()}, f'{(ROOT / "results.json").stat().st_size / 1e6:.1f} MB')
print('decode tags:', {t: len(v) for t, v in out['decode'].items()})
print('rt tags:', {t: len(v) for t, v in out['rt'].items()})
