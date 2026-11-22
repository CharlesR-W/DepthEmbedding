"""Package pinned raw data into schema v1. Entry modules expose `package(inputs, root)`.

Run `python site/xsite/ops.py` (all available entries), `--entry x3` (one entry), or
`--fixtures` (regenerate the small frozen integration fixtures). No network access.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import importlib
import os
from pathlib import Path
import shutil

import numpy as np

if __package__:
    from .fetch import ROOT, FETCH, sha256, write_json
    from .opdata import (_axes, read_operator, read_overlay, read_paths, require, source_from_joint,
                         write_operator, write_overlay, write_paths)
    from .validate import Inputs, check_x3_logits, validate_operator, viewer_alignment, x1_split, x1_align
else:
    from fetch import ROOT, FETCH, sha256, write_json
    from opdata import (_axes, read_operator, read_overlay, read_paths, require, source_from_joint,
                        write_operator, write_overlay, write_paths)
    from validate import Inputs, check_x3_logits, validate_operator, viewer_alignment, x1_split, x1_align

OUT = ROOT/'data/sitex'
FIXTURES = Path(__file__).with_name('fixtures')
SEED = 20261122
X3 = {'clean': 'Clean', 'u2': 'PGD ε 2/255, untargeted', 'u8': 'PGD ε 8/255, untargeted',
      't8': 'PGD ε 8/255, targeted', 'n8': 'Sign noise ε 8/255'}


def sample(n, k=300):
    return np.sort(np.random.default_rng(SEED).choice(n, k, replace=False))


def write_sprite(root, filename, images, cols=20):
    from PIL import Image

    images = np.asarray(images)
    if images.ndim == 4 and images.shape[1] == 3:
        images = images.transpose(0, 2, 3, 1)
    require(images.ndim == 4 and images.shape[-1] == 3 and images.dtype == np.uint8 and len(images) > 0,
            'Sprite images must be a nonempty uint8 RGB stack')
    require(Path(filename).name == filename and filename.endswith('.webp'), 'Invalid sprite filename')
    n, h, w, _ = images.shape
    atlas = np.zeros(((n+cols-1)//cols*h, cols*w, 3), dtype=np.uint8)
    for j, im in enumerate(images):
        y, x = divmod(j, cols)
        atlas[y*h:(y+1)*h, x*w:(x+1)*w] = im
    path = Path(root)/'sprites'/filename
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(atlas).save(path, quality=82, method=4)
    return f'../../../sprites/{filename}'


def net_metadata(inputs, net):
    if net in ('c10-s0', 'c10-s1', 'init-s0', 'init-s1'):
        seed = int(net[-1])
        return dict(id=f'r20s{seed}', label=f'ResNet-20 seed {seed}', step=0 if net.startswith('init') else 64000,
                    train='CIFAR-10', acc=None if net.startswith('init') else (0.9156 if seed == 0 else 0.9151))
    key = f'crw-depth-xp-results/nets/{net}/info.json'
    import json
    info = json.loads(inputs.file(key).read_text())
    data = {'c10x': 'CIFAR-10', 'c100': 'CIFAR-100', 'c100c': 'CIFAR-100 coarse', 'svhn': 'SVHN',
            'in32': 'ImageNet-32', 'anim': 'CIFAR-10 animals', 'veh': 'CIFAR-10 vehicles',
            'rand': 'CIFAR-10 shuffled labels'}[net.rsplit('-', 1)[0]]
    return dict(id=net, label=f'{data} · seed {net[-1]}', step=info['steps'], train=data,
                acc=info['log'][-1]['test_acc'])


def package_operator(inputs, root, op_id, raw, *, probe_pos=None, K=None, source=False, site=False, dtype='i16', **kwargs):
    """Use canonical arrays from validate_operator, optionally subsetted for a frozen fixture."""
    rec = inputs.manifest['ops'][op_id]
    K = raw['joint'].shape[-1] if K is None else K
    idx = slice(None) if probe_pos is None else probe_pos
    joint, vals = raw['joint'][:, idx, :K], raw['vals'][:K]
    labels = None if probe_pos is None else raw['labels'][idx]
    s = source_from_joint(joint, raw['c'], raw['t'], vals) if source else None
    return write_operator(root, op_id, joint=joint, ids=raw['ids'][idx], chain=raw['chain'], c=raw['c'], t=raw['t'],
                          vals=vals, tstar=raw['tstar'], legacy_basis=raw['legacy_basis'], fixture_labels=labels,
                          probe_id=rec['probe']['id'], net=net_metadata(inputs, rec['net']), kernel=rec['kernel'],
                          provenance=inputs.provenance(op_id), dtype=dtype, source=s,
                          site=raw['site'][:, idx, :K] if site else None, **kwargs)


def _viewer_axes(raw):
    # Site export quantizes int16 -> int8, keeps the sign of every nonzero entry,
    # and rounds mode weights to 4 significant figures. See st_q8 and buildG.
    v = raw['vecs'][0, 3]
    q = np.round(v.astype(np.float64) * 127 / 32767)
    q = np.where((q == 0) & (v != 0), np.sign(v), q)
    w = np.vectorize(lambda x: float(f'{x:.4g}'))(raw['weight'][0, 3])
    modes = q / np.linalg.norm(q, axis=-1, keepdims=True) * np.sqrt(w)[..., None]
    y = modes.transpose(1, 2, 0)
    y = y / np.linalg.norm(y, axis=(1, 2), keepdims=True)
    return _axes(y)


def x3(inputs, root, *, fixture=False):
    for kernel in (('ea100',) if fixture else ('ea100', 'knn15')):
        op_id = f'r20s0-c10-{kernel}'
        op = validate_operator(inputs, op_id)
        rec = inputs.manifest['ops'][op_id]
        kwargs = {}
        if kernel == 'knn15':
            with np.load(inputs.file(rec['checks']['viewer_modes']), allow_pickle=False) as ref:
                signs, _ = viewer_alignment(op, ref)
                kwargs.update(viewer_signs=signs, axes_viewer=_viewer_axes(ref))
        out = package_operator(inputs, root, op_id, op, site=True, source=True,
                               probe_pos=sample(len(op['ids']), 24) if fixture else None, **kwargs)
        sets_path = rec['file'].rsplit('/', 1)[0] + '/sets.npz'
        with np.load(inputs.file(sets_path), allow_pickle=False) as sets:
            ids, labels, target = sets['S'], sets['labels'], sets['target']
            require(len(ids) == len(labels) == len(target), f'{op_id}: X3 set labels mismatch')
            require(not np.intersect1d(ids, op['ids']).size, f'{op_id}: X3 held-out set overlaps probe')
            pos = sample(len(ids), 20 if fixture else 300)
            sprite = None if fixture else write_sprite(root, 'x3_sample.webp', sets['img_clean'][pos])
        ref_logits = inputs.value(rec['checks']['logits_from'])
        with np.load(inputs.file(rec['file']), allow_pickle=False) as raw:
            for set_id in (('clean','u8','n8') if fixture else X3):
                key = f'{op_id}/{set_id}'
                row = inputs.manifest['overlays'][key]
                with np.load(inputs.file(row['file']), allow_pickle=False) as overlay:
                    joint, site = (overlay[row['keys'][k]] for k in ('joint', 'site'))
                    L, K = len(op['chain']), len(op['vals'])
                    require(joint.shape == site.shape == (L, len(ids), K), f'{key}: X3 raw overlay shape mismatch')
                    err = check_x3_logits(op, raw, ref_logits, overlay, kernel, pos[::max(1, len(pos)//8)])
                    pred = overlay['logits'][pos].argmax(1)
                    success = None if set_id == 'clean' else (pred == target[pos] if set_id == 't8' else pred != labels[pos]).tolist()
                    src = source_from_joint(joint[:, pos], op['c'], op['t'], op['vals'])
                    meta = dict(label=labels[pos].tolist(), pred=pred.tolist(), success=success,
                                rowmax=None, scalars={})
                    if set_id == 't8':
                        meta['target'] = target[pos].tolist()
                    write_overlay(root, out, set_id, coords=dict(joint=joint[:, pos], site=site[:, pos], source=src),
                                  label=X3[set_id], dataset='cifar10-test', ids=ids[pos], pos=pos, sample_of=len(ids),
                                  meta=meta, provenance=inputs.provenance(op_id, key=row['file'], entry=f'inputs.json#overlays/{key}'),
                                  pair=row['pairs'], sprite=sprite, sprite_cols=20 if sprite else None)
                    print(f'{key}: logits replay max abs {err:.3g}')


def _panel(op, label=None, R=None, stat=None):
    return dict(op=op, label=label or op, R=R, stat=stat)


def _spec(panels, kinds, *, K=16, site='b6', **kwargs):
    return dict(type='emb', schema=1, panels=panels, kinds=kinds, kind=kinds[0], site=site, K=K,
                dims=2, axes='pooled:0', norm='site', colour={'by':'class'}, height=420, **kwargs)


def fixtures(inputs, root):
    root.mkdir(parents=True, exist_ok=True)
    x3(inputs, root, fixture=True)
    write_json(root/'F1.json', _spec([_panel('r20s0-c10-ea100')], ['site', 'source', 'joint'],
                                    overlays=[{'set':'u8', 'draw':'arrows', 'from':'clean'},
                                              {'set':'n8', 'draw':'arrows', 'from':'clean', 'style':'control'}],
                                    overlayMenu=['u8', 'n8']))
    switches = []
    for l0 in ('b3', 'b9'):
        op_id, set_id = f'x4-{l0}-suffix', f'x4-{l0}-paths'
        raw = validate_operator(inputs, op_id)
        K = 8
        op = package_operator(inputs, root, op_id, raw, probe_pos=sample(len(raw['ids']), 24), K=K, site=True)
        with np.load(inputs.file(inputs.manifest['ops'][op_id]['file']), allow_pickle=False) as z:
            kind = z['kind']
            idx = np.array([np.flatnonzero(kind == i)[0] for i in (0, 1)])
            labels = z['labels'][idx]
            switch = [float(s) if np.isfinite(s) else None for s in z['s_switch'][idx]]
            coords = dict(joint=z['emb_y'][:, idx, :, :K], site=z['emb_site_y'][:, idx, :, :K])
            meta = dict(pair_pos=idx.tolist(), kind=['same', 'diff'], label_a=labels[:, 0].tolist(),
                        label_b=labels[:, 1].tolist(), s_switch=switch)
            write_paths(root, op, set_id, coords=coords, l0=l0, sites=z['sites'], s=z['s64'], meta=meta,
                        provenance=inputs.provenance(op_id, entry=f'inputs.json#paths/{op_id}/{set_id}'))
        switches.append(dict(label=f'Start at {l0}', panels=[_panel(op_id)], paths=[set_id]))
    write_json(root/'F2.json', _spec(switches[0]['panels'], ['joint', 'site'], K=8, site='b9', switch=switches))
    net_ids = ('c10-s0', 'c100-s0', 'svhn-s0')
    op_ids = [f'x1-{net}-trunk' for net in net_ids]
    raws = [validate_operator(inputs, i) for i in op_ids]
    check = inputs.manifest['ops'][op_ids[0]]['checks']['compare']
    menu = []
    with np.load(inputs.file(check), allow_pickle=False) as compare:
        train, test = x1_split(compare['labels'], compare['perm'], compare['within'])
        names = compare['nets'].tolist()
        a = names.index(net_ids[0])
        vals, ts, cs, cos = (compare[f'knn15_trunk_{k}'] for k in ('vals', 't', 'c', 'cos'))
        for j, (net, op_id, raw) in enumerate(zip(net_ids, op_ids, raws)):
            b = names.index(net)
            require(np.array_equal(raw['vals'], vals[b, :len(raw['vals'])]) and np.array_equal(raw['c'], cs[b])
                    and raw['t'] == ts[b], f'{op_id}: X1 comparison basis metadata differs')
            op = package_operator(inputs, root, op_id, raw, probe_pos=sample(len(raw['ids']), 40), dtype='i8')
            if j:
                Q, result = x1_align(raws[0]['joint'], raw['joint'], train, test, cos[a, b])
                print(f'{op_id}: X1 agreement max abs error {np.max(np.abs(result-cos[a,b])):.3g}')
                menu.append(_panel(op_id, op['net']['label'], Q.tolist(),
                                   dict(label='joint agreement', values=cos[a, b].tolist())))
    write_json(root/'F3.json', _spec([_panel(op_ids[0], 'CIFAR-10 · seed 0 (reference)'), menu[0]],
                                    ['joint'], panelMenu=menu))
    require(sum(p.stat().st_size for p in root.rglob('*') if p.is_file()) <= 300_000, 'Frozen fixtures exceed 300 KB')


def manifest_for(root):
    root = Path(root)
    files = {p.relative_to(root).as_posix(): dict(bytes=p.stat().st_size, sha256=sha256(p))
             for p in sorted(root.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'}
    ops = {}
    for desc in sorted((root/'ops').glob('*/op.json')):
        op = read_operator(desc)
        sets, paths = ([reader(p, op)['id'] for p in sorted((desc.parent/sub).glob('*.json'))]
                       for sub, reader in (('ov', read_overlay), ('paths', read_paths)))
        ops[op['id']] = dict(fp=op['fp'], overlays=sets, paths=paths, provenance=op['provenance'],
                            bytes=sum(v['bytes'] for k,v in files.items() if k.startswith(f"ops/{op['id']}/")))
    return dict(schema=1, ops=ops, files=files)


def _swap(tmp, out):
    if not out.exists():
        os.replace(tmp, out)
        return
    # POSIX rename cannot replace a nonempty directory. Linux RENAME_EXCHANGE
    # makes swapping directory roots truly atomic (same-volume sibling paths).
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.renameat2(-100, os.fsencode(tmp), -100, os.fsencode(out), 2):
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err), str(out))
    shutil.rmtree(tmp)


def _entry_modules():
    return {p.stem.removeprefix('ops_'): p.stem for p in Path(__file__).parent.glob('ops_*.py')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', action='append', help='package one entry, keeping other entries; repeatable')
    parser.add_argument('--fixtures', action='store_true', help='regenerate fixtures instead of production data')
    parser.add_argument('--out', type=Path, help='override output directory')
    parser.add_argument('--fetch', type=Path, default=FETCH, help='override staged input root')
    args = parser.parse_args()
    if args.fixtures and args.entry:
        parser.error('--fixtures cannot be combined with --entry')
    modules = _entry_modules()
    for entry in args.entry or []:
        if entry not in {'x3', *modules}:
            parser.error(f'Unknown entry {entry}')
    inputs = Inputs(args.fetch)
    out = (args.out or (FIXTURES if args.fixtures else OUT)).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + '.tmp')
    # Fixtures use the same gitignored lock as production.
    lockpath = (ROOT/'data/.sitex.lock') if args.fixtures else out.with_name('.'+out.name+'.lock')
    with lockpath.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if tmp.exists():
            shutil.rmtree(tmp)
        if args.entry and out.exists():
            shutil.copytree(out, tmp, ignore=shutil.ignore_patterns('MANIFEST.json'))
        else:
            tmp.mkdir()
        try:
            if args.fixtures:
                fixtures(inputs, tmp)
                _swap(tmp, out)
            else:
                (tmp/'ops').mkdir(exist_ok=True)
                (tmp/'sprites').mkdir(exist_ok=True)
                for entry in args.entry or ['x3', *sorted(modules)]:
                    if entry == 'x3':
                        x3(inputs, tmp)
                    else:
                        module = importlib.import_module(f'{__package__}.{modules[entry]}' if __package__ else modules[entry])
                        module.package(inputs, tmp)
                manifest = manifest_for(tmp)
                _swap(tmp, out)
                write_json(out/'MANIFEST.json', manifest)
        finally:
            if tmp.exists():
                shutil.rmtree(tmp)
    print(f'Published {out}')


if __name__ == '__main__':
    main()
