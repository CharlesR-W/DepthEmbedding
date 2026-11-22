"""Package the 18 pinned X1 trunk operators and verify the shared rotations."""
from __future__ import annotations

import json

import numpy as np

if __package__:
    from .fetch import ROOT, write_json
    from .opdata import quantize, require
    from .ops import package_operator
    from .validate import validate_operator, x1_align, x1_split
else:
    from fetch import ROOT, write_json
    from opdata import quantize, require
    from ops import package_operator
    from validate import validate_operator, x1_align, x1_split

REF = 'c10-s0'
PREFIX = 'knn15_trunk_'


def zoo_accuracy(net):
    """Use the public viewer's actual zoo evaluation archives, not a default."""
    if net == REF:
        path = 'data/r20s0b/curves.json'
        final = json.loads((ROOT / path).read_text())['checkpoints'][-1]
        require(final['step'] == 64000, 'X1 zoo reference accuracy must be from step 64000')
        return final['val_acc'], path + ':checkpoints[-1].val_acc'
    if net == 'c10-s1':
        path = 'data/stnets/r20s1/info.json'
        return json.loads((ROOT / path).read_text())['test_acc'], path + ':test_acc'
    return None


def op_id(net):
    return f'x1-{net}-trunk'


def load_operators(inputs, root=None):
    """Read one pinned bundle at a time; retain only these operators' first 16 joint modes."""
    rec = inputs.manifest['ops'][op_id(REF)]
    key = rec['checks']['compare']
    with np.load(inputs.file(key, pin=rec['bundle'][key]), allow_pickle=False) as z:
        keys = ('nets', 'labels', 'perm', 'within', 'nn', 'nn_within', 'nn_perm',
                *(PREFIX + k for k in ('vals', 't', 'c', 'cos', 'cos_perm', 'cos_within')))
        comp = {k: z[k] for k in keys}
    nets = comp['nets'].tolist()
    require({op_id(net) for net in nets} == {k for k in inputs.manifest['ops']
                                            if k.startswith('x1-') and k.endswith('-trunk')},
            'X1 comparison network set differs from the pinned trunk operators')
    modes = []
    for j, net in enumerate(nets):
        raw = validate_operator(inputs, op_id(net))
        require(np.array_equal(raw['labels'], comp['labels']) and raw['chain'] == rec['sites']
                and np.array_equal(raw['vals'], comp[PREFIX + 'vals'][j, :len(raw['vals'])])
                and np.array_equal(raw['c'], comp[PREFIX + 'c'][j]) and raw['t'] == comp[PREFIX + 't'][j],
                f'{net}: X1 comparison basis metadata differs')
        if root is not None:
            desc = package_operator(inputs, root, op_id(net), raw, dtype='i8')
            # Net metadata does not participate in the operator fingerprint. Correct
            # the archived zoo accuracy inside the staging directory before its manifest is hashed.
            zoo = zoo_accuracy(net)
            if zoo and desc['net']['acc'] != zoo[0]:
                desc['net']['acc'] = zoo[0]
                write_json(root / 'ops' / op_id(net) / 'op.json', desc, indent=None)
        modes.append(np.ascontiguousarray(raw['joint']))
    return comp, modes, rec['sites']


def site_cos(a, b):
    return np.einsum('lnk,lnk->l', a, b) / (np.linalg.norm(a, axis=(1, 2)) * np.linalg.norm(b, axis=(1, 2)))


def reference_rotations(comp, modes, train, test):
    """Check the score from the original modes and report the display quantization error separately."""
    where = comp['nets'].tolist().index(REF)
    exact, display = 0., 0.
    rotations = []
    aq, scale = quantize(modes[where], 'i8')
    a8 = aq[:, test] * scale[:, None, :]
    for j, b in enumerate(modes):
        expected = comp[PREFIX + 'cos'][where, j]
        q, measured = x1_align(modes[where], b, train, test, expected)
        if j == where:
            q = np.eye(b.shape[-1])
            measured = site_cos(modes[where][:, test], modes[where][:, test] @ q)
        bq, scale = quantize(b, 'i8')
        b8 = bq[:, test] * scale[:, None, :]
        exact = max(exact, float(np.max(np.abs(measured - expected))))
        display = max(display, float(np.max(np.abs(site_cos(a8, b8 @ q) - expected))))
        rotations.append(q)
    require(exact < 1e-6, 'X1 reference rotations fail replay')
    return rotations, exact, display


def package(inputs, root):
    comp, modes, _ = load_operators(inputs, root)
    train, test = x1_split(comp['labels'], comp['perm'], comp['within'])
    _, exact, display = reference_rotations(comp, modes, train, test)
    print(f'X1: max absolute agreement error {exact:.3g} with raw modes; {display:.3g} with quantized display modes')
