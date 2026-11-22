"""Package the X4 suffix operators and shared path sample (ops.py --entry x4)."""
from __future__ import annotations

import numpy as np

if __package__:
    from .opdata import require, write_paths
    from .ops import SEED, package_operator, write_sprite
    from .validate import validate_operator
else:
    from opdata import require, write_paths
    from ops import SEED, package_operator, write_sprite
    from validate import validate_operator

STARTS = ('pixels', 'b3', 'b6', 'b9')
PER_KIND = 12


def sample_pairs(kind):
    """Keep the report's first 4 same-class and 6 different-class examples, then fill each stratum."""
    rng = np.random.default_rng(SEED)
    groups = []
    for value, n_examples in enumerate((4, 6)):
        positions = np.flatnonzero(kind == value)
        groups.append(np.r_[positions[:n_examples],
                            np.sort(rng.choice(positions[n_examples:], PER_KIND - n_examples, replace=False))])
    return np.concatenate(groups)


def package(inputs, root):
    endpoints = None
    for start in STARTS:
        op_id, set_id = f'x4-{start}-suffix', f'x4-{start}-paths'
        raw = validate_operator(inputs, op_id)
        op = package_operator(inputs, root, op_id, raw, site=True)
        key = f'{op_id}/{set_id}'
        record = inputs.manifest['paths'][key]
        with np.load(inputs.file(record['file']), allow_pickle=False) as z:
            kind, s, labels = z['kind'], z['s64'], z['labels']
            pos = sample_pairs(kind)
            images = z['images'][pos]
            if endpoints is None:
                endpoints = images
                sprite = write_sprite(root, 'x4_endpoints.webp', images.reshape(-1, *images.shape[2:]))
            else:
                require(np.array_equal(images, endpoints), f'{key}: shared endpoint image mismatch')
            coords = {}
            for mode, array_key in record['keys'].items():
                coord = z[array_key]
                require(coord.shape == (len(raw['chain']), len(kind), len(s), op['K']),
                        f'{key}: invalid {mode} coordinate grid')
                coords[mode] = coord[:, pos]
            meta = dict(pair_pos=pos.tolist(), kind=np.where(kind[pos] == 0, 'same', 'diff').tolist(),
                        label_a=labels[pos, 0].tolist(), label_b=labels[pos, 1].tolist(),
                        s_switch=[float(v) if np.isfinite(v) else None for v in z['s_switch'][pos]])
            write_paths(root, op, set_id, coords=coords, l0=start, sites=z['sites'], s=s, meta=meta,
                        provenance=inputs.provenance(op_id, key=record['file'], entry=f'inputs.json#paths/{key}'),
                        sprite=sprite)
