"""Local analysis of X1: xp/out/<compare dir>/<probe>.npz -> xp/out/x1_summary.json

    uv run --with numpy python xp/analyze_x1.py [compare dir name, default compare]

Networks are grouped by training set (c10x-s0, trained here, counts as a CIFAR-10 network).  For every measure and
every pair of training sets the summary holds the per-site mean over network pairs and the individual pairs.
"""
import json
import sys
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / 'out'
SITES = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
PROBES = ['c10', 'c100', 'svhn', 'in32']
GROUP_ORDER = ['c10', 'anim', 'veh', 'c100', 'c100c', 'in32', 'svhn', 'rand', 'init']
r = lambda a, n=4: np.round(np.asarray(a, dtype=float), n).tolist()
group = lambda name: {'c10x': 'c10'}.get(name.split('-')[0], name.split('-')[0])


def main():
    cdir = OUT / (sys.argv[1] if len(sys.argv) > 1 else 'compare')
    out = dict(sites=SITES, probes=[], groups=GROUP_ORDER, measures={})
    diag = np.arange(12)
    for probe in PROBES:
        f = cdir / f'{probe}.npz'
        if not f.exists():
            continue
        z = np.load(f)
        nets = [str(s) for s in z['nets']]
        n = len(nets)
        g = [group(m) for m in nets]
        sym = lambda a: (a + a.transpose(1, 0, *range(2, a.ndim))) / 2
        # per-site measures (n, n, L), symmetrised
        M = {
            'nn': z['nn'][:, :, diag, diag], 'cka': z['cka'][:, :, diag, diag],
            'walk8_knn15': z['knn15_walk8'], 'walk8_ea100': z['ea100_walk8'],
            'joint_knn15_trunk': z['knn15_trunk_cos'], 'joint_ea100_trunk': z['ea100_trunk_cos'],
            'joint_knn15_full': z['knn15_full_cos'], 'joint_ea100_full': z['ea100_full_cos'],
        }
        NUL = {
            'nn': (z['nn_perm'], z['nn_within']), 'cka': (z['cka_perm'], z['cka_within']),
            'walk8_knn15': (z['knn15_walk8_perm'], z['knn15_walk8_within']),
            'walk8_ea100': (z['ea100_walk8_perm'], z['ea100_walk8_within']),
            **{f'joint_{k}_{c}': (z[f'{k}_{c}_cos_perm'], z[f'{k}_{c}_cos_within'])
               for k in ('knn15', 'ea100') for c in ('trunk', 'full')},
        }
        # scalar joint shares (n, n) at K = 16 (index 2), and K sweep
        S = {f'{k}_{c}': z[f'{k}_{c}_share'] for k in ('knn15', 'ea100') for c in ('trunk', 'full')}
        present = [x for x in GROUP_ORDER if x in g]
        P = dict(nets=nets, group=g, cells={}, null={}, share={}, readout=r(z['readout']),
                 labels_n=int(len(np.unique(z['labels']))))
        if 'readout_coarse' in z.files:
            P['readout_coarse'] = r(z['readout_coarse'])
        off = ~np.eye(n, dtype=bool)
        for name, m in M.items():
            cells = {}
            for ga, gb in combinations_with_replacement(present, 2):
                pairs = [(a, b) for a in range(n) for b in range(a + 1, n)
                         if {g[a], g[b]} == {ga, gb} and (ga != gb or g[a] == g[b] == ga)]
                if not pairs:
                    continue
                vals = np.stack([m[a, b] for a, b in pairs])
                cells[f'{ga}|{gb}'] = dict(mean=r(vals.mean(0)), pairs=r(vals), names=[f'{nets[a]}|{nets[b]}' for a, b in pairs])
            P['cells'][name] = cells
            P['null'][name] = dict(perm=r(NUL[name][0][off].mean(0)), within=r(NUL[name][1][off].mean(0)),
                                   within_same=r(np.mean([NUL[name][1][a, b] for a in range(n) for b in range(n)
                                                          if a != b and g[a] == g[b] and g[a] not in ('init', 'rand')] or [np.zeros(m.shape[2])], 0)))
            P['null'][name]['within_pairs'] = {f'{nets[a]}|{nets[b]}': r(NUL[name][1][a, b]) for a in range(n) for b in range(a + 1, n)
                                               if g[a] == g[b] and g[a] not in ('init', 'rand')}
        for name, s in S.items():
            P['share'][name] = dict(K16=r(s[:, :, 2]), K4=r(s[:, :, 0]), K8=r(s[:, :, 1]), K32=r(s[:, :, 3]),
                                    perm=float(z[f'{name}_share_perm'][:, :, 2][off].mean()),
                                    within=float(z[f'{name}_share_within'][:, :, 2][off].mean()))
        # matrices for the network-by-network figure
        P['matrix'] = {m: {s: r(M[m][:, :, SITES.index(s)], 3) for s in ('b2', 'b6', 'b9')} for m in ('nn', 'cka', 'joint_knn15_trunk')}
        P['matrix']['nn']['logits'] = r(M['nn'][:, :, 11], 3)
        # mode-by-mode matrices for selected pairs (trunk, knn15)
        sel = [('c10-s0', b) for b in ('c10-s1', 'c10x-s0', 'anim-s0', 'veh-s0', 'c100-s0', 'in32-s0', 'svhn-s0', 'rand-s0', 'init-s0') if b in nets]
        P['modes'] = {f'{a}|{b}': dict(m=r(z['knn15_trunk_modes'][nets.index(a), nets.index(b)], 3),
                                       sv=r(z['knn15_trunk_sv'][nets.index(a), nets.index(b)], 3)) for a, b in sel if a in nets}
        P['spectrum'] = dict(vals=r(z['knn15_trunk_vals'][:, :16]), t=r(z['knn15_trunk_t']), c=r(z['knn15_trunk_c']),
                             lam2=r(z['knn15_lam2'], 5))
        out['probes'].append(probe)
        out['measures'][probe] = P
        # console digest
        print(f'\n== probe {probe}  nets {n}')
        for name in ('nn', 'cka', 'joint_knn15_trunk'):
            print(f'-- {name}   (sites stem b3 b6 b9 [pool logits])   null perm/within-same:',
                  np.round(np.array(P['null'][name]['perm'])[[0, 3, 6, 9]], 3).tolist(),
                  np.round(np.array(P['null'][name]['within_same'])[[0, 3, 6, 9]], 3).tolist())
            for cell, v in P['cells'][name].items():
                mm = np.array(v['mean'])
                print(f'   {cell:12s} n={len(v["pairs"])}', np.round(mm[[0, 3, 6, 9] + ([10, 11] if len(mm) > 10 else [])], 3).tolist())
        print('-- trunk share K16 knn15: c10|c10 etc')
        s = np.array(P['share']['knn15_trunk']['K16'])
        for ga, gb in combinations_with_replacement(present, 2):
            v = [s[a, b] for a in range(n) for b in range(a + 1, n) if {g[a], g[b]} == {ga, gb} and (ga != gb or g[a] == g[b])]
            if v:
                print(f'   {ga}|{gb}: {np.mean(v):.3f}', end=';')
        print('\n   null perm', round(P['share']['knn15_trunk']['perm'], 4), 'within', round(P['share']['knn15_trunk']['within'], 4))
        print('-- readout (stem b3 b6 b9 pool logits):')
        for a, m in enumerate(nets):
            print(f'   {m:9s}', np.round(z['readout'][a][[0, 3, 6, 9, 10, 11]], 3).tolist())
    # training table
    info = {}
    for f in sorted((OUT / 'nets').glob('*/info.json')):
        d = json.loads(f.read_text())
        info[d['name']] = dict(classes=d['num_classes'], steps=d['steps'], n_train=d['n_train'], **{k: d['log'][-1][k] for k in ('train_acc', 'test_acc')})
    out['training'] = info
    (OUT / 'x1_summary.json').write_text(json.dumps(out))


if __name__ == '__main__':
    main()
