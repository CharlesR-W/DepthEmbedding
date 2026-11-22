"""Shared by the Modal stages of xp/SPEC.md: the network table, loading, taps.  Runs on Modal (paths are mounts)."""
from __future__ import annotations

from pathlib import Path

R = '/results'                       # volume crw-depth-xp-results
SITES = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
ALL_SITES = ['pixels'] + SITES
C10_MEAN, C10_STD = (.4914, .4822, .4465), (.2470, .2435, .2616)
ANIMALS, VEHICLES = [2, 3, 4, 5, 6, 7], [0, 1, 8, 9]

# Networks.  'zoo': an existing ModelZoo checkpoint on cm-cifar-coherent.  Otherwise trained by modal_train.py.
NETS = {
    'c10-s0': dict(zoo='resnet20-relu-seed0', step=64000),
    'c10-s1': dict(zoo='resnet20-relu-seed1', step=64000),
    'init-s0': dict(zoo='resnet20-relu-seed0', step=0),
    'init-s1': dict(zoo='resnet20-relu-seed1', step=0),
    'c10x-s0': dict(data='c10', seed=0),
    'c100-s0': dict(data='c100', seed=0), 'c100-s1': dict(data='c100', seed=1),
    'svhn-s0': dict(data='svhn', seed=0, flip=False), 'svhn-s1': dict(data='svhn', seed=1, flip=False),
    'anim-s0': dict(data='c10', seed=0, classes=ANIMALS), 'anim-s1': dict(data='c10', seed=1, classes=ANIMALS),
    'veh-s0': dict(data='c10', seed=0, classes=VEHICLES), 'veh-s1': dict(data='c10', seed=1, classes=VEHICLES),
    'rand-s0': dict(data='c10', seed=0, shuffle_labels=True, augment=False),
    'c100c-s0': dict(data='c100', seed=0, labels='coarse'), 'c100c-s1': dict(data='c100', seed=1, labels='coarse'),
    'in32-s0': dict(data='in32', seed=0, steps=250000), 'in32-s1': dict(data='in32', seed=1, steps=250000),
}
TRAINED = [n for n, s in NETS.items() if 'zoo' not in s]
PROBES = ['c10', 'c100', 'svhn', 'in32']


def build(num_classes):
    from zoo.models import build_model
    return build_model({'model': {'arch': 'resnet20', 'activation': 'relu', 'num_classes': int(num_classes)}})


def load_net(name):
    """(model on cuda in eval mode, mean (3,), std (3,)) with mean/std in [0,1] pixel units."""
    import torch
    spec = NETS[name]
    if 'zoo' in spec:
        from zoo.io import load_checkpoint
        ck = load_checkpoint(Path(f'/out/runs/{spec["zoo"]}/checkpoints/step_{spec["step"]}.pt'))
        model = build(10); model.load_state_dict(ck['model_state'])
        mean, std = C10_MEAN, C10_STD
    else:
        ck = torch.load(Path(R) / 'nets' / name / 'final.pt', map_location='cpu', weights_only=False)
        model = build(ck['num_classes']); model.load_state_dict(ck['model_state'])
        mean, std = ck['mean'], ck['std']
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return model.eval().cuda(), tuple(float(v) for v in mean), tuple(float(v) for v in std)


def normalize(x01, mean, std):
    import torch
    m = torch.tensor(mean, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    s = torch.tensor(std, device=x01.device, dtype=x01.dtype).view(1, 3, 1, 1)
    return (x01 - m) / s


def activations(model, x):
    """[pixels, stem, b1..b9, pool, logits] for normalised input x."""
    h = model.act(model.bn1(model.conv1(x))); acts = [x, h]
    for block in model.blocks:
        h = block(h); acts.append(h)
    pool = h.mean((2, 3))
    return acts + [pool, model.fc(pool)]


def suffix_all(model, site, h):
    """Activations at every site after `site` (ALL_SITES order), given the tap h at `site`."""
    i = ALL_SITES.index(site)
    out = []
    if i == 0:
        return activations(model, h)[1:]
    if site == 'logits':
        return out
    if site == 'pool':
        return [model.fc(h)]
    for block in model.blocks[i - 1:]:           # stem (i=1) -> blocks[0:], b_j (i=j+1) -> blocks[j:]
        h = block(h); out.append(h)
    pool = h.mean((2, 3))
    return out + [pool, model.fc(pool)]


SHAPES = {'pixels': (3, 32, 32), 'stem': (16, 32, 32), 'b1': (16, 32, 32), 'b2': (16, 32, 32), 'b3': (16, 32, 32),
          'b4': (32, 16, 16), 'b5': (32, 16, 16), 'b6': (32, 16, 16), 'b7': (64, 8, 8), 'b8': (64, 8, 8),
          'b9': (64, 8, 8), 'pool': (64,)}
