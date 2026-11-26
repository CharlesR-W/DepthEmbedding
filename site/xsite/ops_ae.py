"""Package the operators drawn in the ae-objective entry (ae/SPEC.md): the retrained classifier and the autoencoders.

Only site-local coordinates are drawn (each site's own k-NN 15 walk modes), so no rotation between networks is fitted.
`ops.net_metadata` does not know the autoencoders; their metadata is written here from each network's info.json.
"""
from __future__ import annotations

import json

if __package__:
    from .opdata import require, write_operator
    from .validate import validate_operator
else:
    from opdata import require, write_operator
    from validate import validate_operator

XP = 'crw-depth-xp-results'
CLASSIFIER = 'c10x-s0'
AUTOENCODERS = ('ae10-s0', 'ae10-s1', 'ae64-s0', 'ae64-s1')
UNTRAINED = 'init-s0'


def op_id(net):
    return f'x1-{net}-full' if net in (CLASSIFIER, UNTRAINED) else f'ae-{net}-full'


def drawn():
    return (CLASSIFIER, *AUTOENCODERS, UNTRAINED)


def metadata(inputs, net):
    """Network record for op.json; accuracy for classifiers, None otherwise (the autoencoders' PSNR is on the page)."""
    if net == UNTRAINED:
        return dict(id='r20s0', label='Untrained ResNet-20 · seed 0', step=0, train='none', acc=None)
    info = json.loads(inputs.file(f'{XP}/nets/{net}/info.json').read_text())
    require(info['name'] == net and info['log'][-1]['step'] == info['steps'], f'{net}: inconsistent training record')
    seed = net.rsplit('-s', 1)[1]
    if net == CLASSIFIER:
        return dict(id=net, label=f'Classifier · seed {seed}', step=info['steps'], train='CIFAR-10',
                    acc=info['log'][-1]['test_acc'])
    require(info.get('objective') == 'ae', f'{net}: not an autoencoder record')
    return dict(id=net, label=f'Autoencoder, {info["code"]}-number code · seed {seed}', step=info['steps'],
                train='CIFAR-10 (reconstruction)', acc=None)


def package(inputs, root):
    for net in drawn():
        key = op_id(net)
        rec = inputs.manifest['ops'][key]
        raw = validate_operator(inputs, key)
        require(raw['site'] is not None, f'{key}: no site-local coordinates in the raw operator')
        write_operator(root, key, joint=raw['joint'], ids=raw['ids'], probe_labels=raw['labels'], chain=raw['chain'],
                       c=raw['c'], t=raw['t'], vals=raw['vals'], tstar=raw['tstar'], legacy_basis=raw['legacy_basis'],
                       viewer_order=True, probe_id=rec['probe']['id'], net=metadata(inputs, net), kernel=rec['kernel'],
                       provenance=inputs.provenance(key), dtype='i8', site=raw['site'])
        print(f'{key}: packaged ({len(raw["chain"])} sites, K {raw["joint"].shape[-1]})')
