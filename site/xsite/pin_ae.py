"""Write the `ae-<net>-full` operator records of ae/SPEC.md into inputs.json, pinned to the staged files.

Run once after `fetch.py` has staged the autoencoder outputs: `python3 site/xsite/pin_ae.py`.  The records follow the
X1 full-chain ones; the last site is the code, and the pinned bundle also holds what the Experiments entry reads.
"""
from __future__ import annotations

import json

from fetch import AE_COMPARED, AE_NETS, FETCH, ROOT, SITES, write_json

XP = 'crw-depth-xp-results'


def main():
    path = ROOT / 'site/xsite/inputs.json'
    manifest = json.loads(path.read_text())
    ready = json.loads((FETCH / 'READY.json').read_text())
    for net in AE_NETS:
        d = f'{XP}/embed/{net}/c10'
        files = [f'{d}/op_knn15_full.npz', 'local/data/r20s0b/probe.npz', f'{XP}/probes.npz', f'{d}/meta.npz',
                 f'{XP}/compare_ae/c10.npz', f'{XP}/compare_ae/c10.json', f'{XP}/nets/{net}/info.json',
                 f'{XP}/nets/{net}/recon.npz', f'{d}/DONE', f'{XP}/nets/c10x-s1/info.json',
                 *(f'{XP}/embed/{n}/c10/knn.npy' for n in AE_COMPARED)]
        manifest['ops'][f'ae-{net}-full'] = {
            'file': files[0], 'volume': f'{XP}:embed/{net}/c10/op_knn15_full.npz', 'producer': 'xp/modal_xp.py:embed',
            'net': net, 'kernel': {'rule': 'knn', 'param': 15, 'label': 'k-NN 15'}, 'chain': 'full',
            'sites': [*SITES[:-1], 'code'], 'K': 16,
            'probe': {'id': 'c10', 'dataset': 'cifar10-test', 'ids_from': f'{XP}/probes.npz:c10_idx',
                      'labels_from': f'{d}/meta.npz:labels', 'viewer': 'local/data/r20s0b/probe.npz'},
            'checks': {'compare': f'{XP}/compare_ae/c10.npz', 'stage_done': f'{d}/DONE'},
            'bundle': {f: ready[f]['sha256'] for f in files},
        }
    write_json(path, manifest)
    print(f'pinned {len(AE_NETS)} operators')


if __name__ == '__main__':
    main()
