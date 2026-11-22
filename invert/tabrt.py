"""Print round-trip summaries across sites.  Usage: tabrt.py <tag> <K> [prefix ...]"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent / 'data/invert/rt'
S = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
tag, K = sys.argv[1], sys.argv[2]; pre = sys.argv[3:] or ['E']
d = {s: json.loads(f.read_text()) for s in S if (f := ROOT / tag / f'{s}__K{K}.json').exists()}
keys = []
for j in d.values():
    keys += [k for k in j if k not in keys and '@' not in k and k not in ('sel', 'seconds') and any(k.startswith(p) for p in pre)]
print(f'{tag} K={K}'.ljust(26) + ' '.join(f'{s:>7s}' for s in d))
for k in keys:
    print(f'{k:26s}' + ' '.join(f'{j[k]:7.3f}' if k in j and abs(j[k]) < 1e4 else '      -' if k not in j else f'{j[k]:7.0e}' for j in d.values()))
