"""Print decode tables from fetched summaries.  Usage: tab.py <tag> [target] [metric] [K,...]"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / 'data/invert/decode'
S = ['pixels', 'stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
tag = sys.argv[1]; target = sys.argv[2] if len(sys.argv) > 2 else 'H'; metric = sys.argv[3] if len(sys.argv) > 3 else 'cos_raw'
Ks = [int(k) for k in sys.argv[4].split(',')] if len(sys.argv) > 4 else [4, 16, 64, 256]
d = json.loads((ROOT / tag / 'summary.json').read_text())
sites = [s for s in S if f'{s}__{s}' in d]
print(f'{tag}  target={target}  metric={metric}')
print('K    dec     ' + ' '.join(f'{s:>6s}' for s in sites))
rows = [(0, 'mean'), (0, 'true'), (0, 'step')] + [(K, n) for K in Ks for n in ('linear', 'nn', 'lp', 'lpz', 'gh', 'krr', 'mlp', 'pca')]
for K, n in rows:
    vals = [d[f'{s}__{s}'].get(f'{target}|{K}|{n}|{metric}') for s in sites]
    if all(v is None for v in vals):
        continue
    print(f'{K:<4d} {n:7s} ' + ' '.join('     -' if v is None else f'{v[0]:6.3f}' for v in vals))
