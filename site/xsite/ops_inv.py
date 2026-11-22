"""Package the inversion P2 operators and their out-of-sample image sets.

Run via `python site/xsite/ops.py --entry inv` for an atomic publication.
"""
from __future__ import annotations

import numpy as np

if __package__:
    from .opdata import require, write_overlay
    from .ops import package_operator, sample, write_sprite
    from .validate import check_inversion_overlay, validate_operator
else:
    from opdata import require, write_overlay
    from ops import package_operator, sample, write_sprite
    from validate import check_inversion_overlay, validate_operator

VOL = 'crw-depth-invert-results'
PR = 'r20s0-p2-pr300'
EA = 'r20s0-p2-ea300'
SETS = {
    'H': 'Held-out CIFAR-10',
    'H-pgd': 'Clean CIFAR-10 paired with PGD',
    'c100': 'CIFAR-100',
    'svhn': 'SVHN',
    'noise20': 'Pixel noise σ = 0.2',
    'pgd2': 'PGD ε 2/255',
    'pgd8': 'PGD ε 8/255',
}


def _decode(inputs, chain, n, pos):
    scalars = {}
    for site in chain:
        with np.load(inputs.file(f'{VOL}/decode/P2__pr300__L12/{site}__{site}.npz'), allow_pickle=False) as z:
            for k in (4, 16, 64, 256):
                for field, stem, label, limits in (
                    ('cos_raw', 'decode_cos', 'kernel-ridge decode cosine', [-1, 1]),
                    ('agree', 'decode_agree', 'patched top-1 agreement', [0, 1]),
                ):
                    key = f'H|{k}|krr|{field}'
                    entry = scalars.setdefault(f'{stem}_K{k}', dict(values=[], label=f'{label} (K = {k})', range=limits))
                    if site == 'logits' and field == 'agree':  # no downstream patch measurement
                        entry['values'].append([None] * len(pos))
                        continue
                    arr = z[key]
                    require(arr.shape == (n,) and np.isfinite(arr).all(), f'{site}: invalid decode scores for {key}')
                    entry['values'].append(arr[pos].astype(float).tolist())
    return scalars


def package(inputs, root):
    package_operator(inputs, root, EA, validate_operator(inputs, EA))
    raw = validate_operator(inputs, PR)
    op = package_operator(inputs, root, PR, raw, source=True)
    held = inputs.value(f'{VOL}/splits.npz:H')
    labels = inputs.value(f'{VOL}/splits.npz:labels')
    require(held.shape == (2000,) and len(np.unique(held)) == len(held)
            and not np.intersect1d(held, raw['ids']).size, 'H is not a distinct held-out set')
    pair_pos = sample(1000)
    for set_id, title in SETS.items():
        clean_pair = set_id == 'H-pgd'
        attack = set_id in ('pgd2', 'pgd8')
        key = f'{PR}/{"H" if clean_pair else set_id}'
        row = inputs.manifest['overlays'][key]
        with np.load(inputs.file(row['file']), allow_pickle=False) as z:
            err = check_inversion_overlay(raw, z)
            n_raw = 1000 if attack else 2000
            n = 1000 if clean_pair else n_raw
            ids = held[:n] if set_id not in ('c100', 'svhn') else np.arange(n)
            pos = pair_pos if attack or clean_pair else sample(n)
            coords = {}
            for kind, field in row['keys'].items():
                arr = z[field]
                require(arr.ndim == 3 and arr.shape[:2] == (len(raw['chain']), n_raw)
                        and arr.shape[2] >= op['K'], f'{key}: invalid {kind} coordinate shape')
                coords[kind] = arr[:, pos, :op['K']]
            rowmax = z['rowmax']
            require(rowmax.shape == (len(raw['chain']), n_raw) and np.isfinite(rowmax).all()
                    and ((rowmax >= 0) & (rowmax <= 1)).all(), f'{key}: invalid row maxima')
            rowmax = rowmax[:, pos].max(axis=0).tolist()
        c10_meta = set_id in ('H', 'H-pgd')
        with np.load(inputs.file(row['images']['meta']), allow_pickle=False) as a:
            image_pos = ids[pos] if c10_meta else pos
            raw_labels = a['labels']
            if row['images']['dataset'] == 'cifar10-test':
                actual = raw_labels[ids] if c10_meta else raw_labels
                require(np.array_equal(actual, labels[ids]), f'{set_id}: held-out label order mismatch')
            display_labels = raw_labels[image_pos].tolist()
            pred = a['logits'][image_pos].argmax(axis=1)
            sprite = write_sprite(root, f'inv_{set_id}.webp', a['images'][image_pos])
        scalars = _decode(inputs, raw['chain'], len(held), pos) if set_id == 'H' else {}
        success = (pred != display_labels).tolist() if attack else None
        meta = dict(label=display_labels, pred=pred.tolist(), success=success, rowmax=rowmax, scalars=scalars)
        write_overlay(root, op, set_id, coords=coords, label=title, dataset=row['images']['dataset'],
                      ids=ids[pos], pos=pos, sample_of=n, meta=meta,
                      provenance=inputs.provenance(PR, key=row['file'], entry=f'inputs.json#overlays/{key}'),
                      pair='H-pgd' if attack else None, sprite=sprite, sprite_cols=20)
        print(f'{PR}/{set_id}: one-step source relative error {err:.3g}')
