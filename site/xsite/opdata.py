"""Schema-v1 array codec, source stencil, and operator/overlay/path readers and writers."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
import re

import numpy as np

if __package__:
    from .fetch import write_json
else:
    from fetch import write_json

DTYPES = {'i16': np.dtype('<i2'), 'i8': np.dtype('i1')}
KINDS = {'joint', 'source', 'site'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def name(s):
    require(isinstance(s, str) and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_-]*', s), f'Invalid id: {s!r}')
    return s


def source_from_joint(y, c, t, vals):
    """Apply s = 2 c T y; axis 0 is site and the last axis is mode."""
    y, c, vals = (np.asarray(x, dtype=np.float64) for x in (y, c, vals))
    L, K = y.shape[0], y.shape[-1]
    require(c.shape == (L,) and vals.shape == (K,) and np.all(c > 0), 'Invalid shape or c for source stencil')
    deg = np.zeros(L)
    deg[1:] += 1
    deg[:-1] += 1
    tail = (1,) * (y.ndim - 2)
    diag = ((1 / (2 * c) + t * deg)[:, None] - vals).reshape(L, *tail, K)
    out = y * diag
    out[1:] -= t * y[:-1]
    out[:-1] -= t * y[1:]
    return out * (2 * c).reshape(L, *tail, 1)


def quantize(x, dtype='i16'):
    x = np.asarray(x, dtype=np.float64)
    require(x.ndim in (3, 4) and all(x.shape) and np.isfinite(x).all(), 'Invalid coordinate array')
    qtype = DTYPES[dtype]
    cap = np.iinfo(qtype).max
    mx = np.max(np.abs(x), axis=tuple(range(1, x.ndim-1)))
    scale = np.where(mx > 0, mx / cap, 1.)
    q = np.rint(x / scale.reshape(x.shape[0], *([1] * (x.ndim-2)), x.shape[-1]))
    return q.astype(qtype), scale


def write_array(directory, stem, x, dtype='i16'):
    q, scale = quantize(x, dtype)
    path = Path(directory) / f'{name(stem)}.{dtype}.gz'
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('wb') as out, gzip.GzipFile(fileobj=out, mode='wb', filename='', mtime=0, compresslevel=6) as z:
        z.write(q.tobytes(order='C'))
    return dict(file=path.name, dtype=dtype, shape=list(q.shape), scale=scale.tolist())


def _array(path, spec):
    dtype = DTYPES.get(spec['dtype'])
    shape = spec['shape']
    scale = np.asarray(spec['scale'], dtype=np.float64)
    require(dtype is not None and len(shape) in (3, 4) and all(type(v) is int and v > 0 for v in shape),
            'Invalid coordinate dtype or shape')
    require(scale.shape == (shape[0], shape[-1]) and np.isfinite(scale).all() and (scale > 0).all(),
            'Invalid coordinate scale')
    file = spec['file']
    require(Path(file).name == file and file.endswith(f".{spec['dtype']}.gz"), f'Invalid coordinate filename: {file}')
    raw = gzip.decompress((Path(path) / file).read_bytes())
    from math import prod
    require(len(raw) == dtype.itemsize * prod(shape), f'{file}: coordinate byte length mismatch')
    return raw, dtype, scale


def read_array(directory, spec):
    raw, dtype, scale = _array(directory, spec)
    shape = spec['shape']
    q = np.frombuffer(raw, dtype=dtype).reshape(shape)
    return q * scale.reshape(shape[0], *([1] * (len(shape)-2)), shape[-1])


def ids_bytes(ids):
    ids = np.asarray(ids)
    require(ids.ndim == 1 and ids.dtype.kind in 'iu' and np.all(ids >= 0) and np.unique(ids).size == ids.size,
            'Probe IDs must be distinct nonnegative integers')
    return ids.astype('<i8').tobytes()


def probe_labels_bytes(labels):
    labels = np.asarray(labels)
    require(labels.ndim == 1 and labels.dtype.kind in 'iu' and ((labels >= 0) & (labels <= 127)).all(),
            'Probe labels must be nonnegative int8 class IDs')
    return labels.astype('i1').tobytes()


def _canon(v):
    return json.dumps(v, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def fingerprint(directory, op, ids):
    """SHA256(canonical parameters || ordered <i8 IDs || signed coordinate bytes/scales)."""
    def rounded(x):
        y = np.round(np.asarray(x, dtype=np.float64), 12)
        return np.where(y == 0, 0., y).tolist()

    h = hashlib.sha256(_canon(dict(kernel={k: op['kernel'][k] for k in ('rule', 'param')},
                                  chain=op['chain'], units=op['units'], c=rounded(op['c']),
                                  t=rounded(op['t']), vals=rounded(op['vals']))))
    b = ids_bytes(ids)
    require(len(b) == 8 * op['probe']['n'] and hashlib.sha256(b).hexdigest() == op['probe']['ids_sha256'],
            'Operator probe order or count mismatch')
    h.update(b)
    for kind, spec in sorted(op['kinds'].items()):
        raw, _, scale = _array(directory, spec)
        h.update(_canon(dict(kind=kind, dtype=spec['dtype'], shape=spec['shape'])))
        h.update(raw)
        h.update(scale.astype('<f8').tobytes(order='C'))
    return h.hexdigest()


def _axes(x):
    K = x.shape[-1]
    x = x.reshape(-1, K)
    # No recentering: projection is linear through the coordinate origin.
    _, vecs = np.linalg.eigh(x.T @ x)
    vecs = vecs[:, ::-1][:, :3].copy()
    sign = np.sign(vecs[np.argmax(np.abs(vecs), axis=0), np.arange(vecs.shape[1])])
    return (vecs * sign).tolist()


def write_operator(root, op_id, *, joint, ids, probe_labels, chain, c, t, vals, kernel, net, provenance, title=None,
                   probe_id='c10', units='relative site units', tstar=None, legacy_basis=None, source=None,
                   site=None, dtype='i16', viewer_signs=None, axes_viewer=None, viewer_order=True):
    """Write under root/ops/id; accepts already-validated physical (unscaled) coordinates."""
    directory = Path(root) / 'ops' / name(op_id)
    directory.mkdir(parents=True, exist_ok=True)
    joint = np.asarray(joint, dtype=np.float64)
    chain, c, vals = list(chain), np.asarray(c), np.asarray(vals)
    L, N, K = joint.shape
    require(L == len(chain) == c.size and K == vals.size and N == len(ids), f'{op_id}: operator shape mismatch')
    labels = probe_labels_bytes(probe_labels)
    require(len(labels) == N, f'{op_id}: probe labels do not match IDs')
    probe = dict(id=probe_id, n=N, ids_sha256=hashlib.sha256(ids_bytes(ids)).hexdigest(),
                 labels_sha256=hashlib.sha256(labels).hexdigest(), viewer_order=viewer_order,
                 labels='../../../d/labels.bin', sprite='../../../d/sprite.webp', sprite_cols=50)
    if not viewer_order:
        (directory/'labels.bin').write_bytes(labels)
        probe.update(ids=np.asarray(ids).tolist(), labels='labels.bin', sprite=None, sprite_cols=None)
    op = dict(schema=1, id=op_id, title=title or f"{net['label']} · {kernel['label']}", net=net,
              kernel=kernel, probe=probe, chain=chain, units=units, c=c.tolist(), t=float(t),
              tstar=float(t if tstar is None else tstar), vals=vals.tolist(), K=K, legacy_basis=legacy_basis,
              kinds={}, site_norm={}, axes={}, viewer_signs=None if viewer_signs is None else np.asarray(viewer_signs).tolist(),
              provenance=provenance)
    x = dict(joint=joint)
    if source is not None:
        x['source'] = source
    if site is not None:
        x['site'] = site
    for kind, value in x.items():
        value = np.asarray(value, dtype=np.float64)
        require(value.shape == joint.shape, f'{op_id}/{kind}: shape mismatch')
        op['kinds'][kind] = write_array(directory, f'probe_{kind}', value, dtype)
        norm = np.sqrt((value**2).sum(axis=(1, 2)))
        require((norm > 0).all(), f'{op_id}/{kind}: zero probe site norm')
        op['site_norm'][kind] = norm.tolist()
        if kind == 'joint':
            op['axes']['pooled'] = _axes(value / norm[:, None, None])
        if kind == 'site':
            op['kinds'][kind]['axes'] = 'own'
    if axes_viewer is not None:
        op['axes']['viewer'] = np.asarray(axes_viewer).tolist()
    op['fp'] = fingerprint(directory, op, ids)
    write_json(directory/'op.json', op, indent=None)
    return op


def _read_json(path):
    obj = json.loads(Path(path).read_text())
    require(obj.get('schema') == 1, f'{path}: expected schema 1')
    return obj


def read_operator(path, *, ids=None):
    path = Path(path)
    op = _read_json(path)
    require(op['id'] == path.parent.name and 'joint' in op['kinds'] and set(op['kinds']) <= KINDS,
            f'{path}: invalid operator id/kinds')
    require(len(op['chain']) == len(op['c']) and len(set(op['chain'])) == len(op['chain'])
            and op['K'] == len(op['vals']), f'{path}: invalid operator metadata shapes')
    require(re.fullmatch(r'[0-9a-f]{64}', op['fp']), f'{path}: invalid fingerprint')
    probe = op['probe']
    L, N, K = len(op['chain']), probe['n'], op['K']
    require(all(type(dim) is int and dim > 0 for dim in (L, N, K)), f'{path}: invalid operator dimensions')
    require(type(probe.get('viewer_order')) is bool and
            all(isinstance(probe.get(key), str) and re.fullmatch(r'[0-9a-f]{64}', probe[key])
                for key in ('ids_sha256', 'labels_sha256')), f'{path}: invalid probe digests or viewer_order')
    if not probe['viewer_order']:
        labels = (path.parent/'labels.bin').read_bytes()
        require(len(labels) == N and hashlib.sha256(labels).hexdigest() == probe['labels_sha256'],
                f'{path}: fixture probe label digest or count mismatch')
    shape = [L, N, K]
    for axis in ('pooled', 'viewer'):
        if axis == 'pooled' or axis in op['axes']:
            matrix = np.asarray(op['axes'][axis], dtype=np.float64)
            require(matrix.shape == (K, min(K, 3)) and np.isfinite(matrix).all(), f'{path}: invalid {axis} axes')
    signs = op['viewer_signs']
    require(signs is None or (len(signs) == K and set(signs) <= {-1, 1}), f'{path}: invalid viewer signs')
    for kind, spec in op['kinds'].items():
        norms = np.asarray(op['site_norm'][kind])
        require(spec['shape'] == shape and norms.shape == (shape[0],) and np.isfinite(norms).all()
                and (norms > 0).all(), f'{path}: {kind} shape or norm mismatch')
        _array(path.parent, spec)
    if ids is None:
        ids = op['probe'].get('ids')
    if ids is not None:
        require(op['fp'] == fingerprint(path.parent, op, ids), f'{path}: operator fingerprint mismatch')
    return op


def _child_kinds(directory, op, set_id, coords, prefix):
    require(coords and coords.keys() <= op['kinds'].keys(), f'{set_id}: invalid coordinate kinds')
    kinds = {}
    for kind, value in coords.items():
        value = np.asarray(value)
        require(value.shape == (len(op['chain']), *prefix, op['K']), f'{set_id}/{kind}: coordinate shape mismatch')
        kinds[kind] = write_array(directory, f'{set_id}_{kind}', value)
    return kinds


def check_labels(where, labels, meta, n, fields):
    """Validate the true-label namespace; predictions are in the network's output space."""
    require(isinstance(labels, dict) and isinstance(labels.get('space'), str)
            and re.fullmatch(r'[a-z][a-z0-9_-]*', labels['space']) and 'names' in labels,
            f'{where}: missing or invalid labels namespace (expected space and names)')
    names = labels['names']
    require(names is None or (isinstance(names, list) and names
                             and all(isinstance(s, str) and s.strip() for s in names)),
            f'{where}: labels.names must be a nonempty class-name list or null')
    for key in fields:
        ids = meta.get(key) if isinstance(meta, dict) else None
        require(isinstance(ids, list) and len(ids) == n and all(type(i) is int and i >= 0 for i in ids),
                f'{where}: invalid true-label IDs in meta.{key}')
        require(names is None or all(i < len(names) for i in ids),
                f'{where}: meta.{key} ID outside labels.names')


def write_overlay(root, op, set_id, *, coords, label, labels, dataset, ids, pos, sample_of, meta, provenance,
                  pair=None, sprite=None, sprite_cols=None, sample_rule='sorted choice without replacement, seed 20261122'):
    directory = Path(root)/'ops'/name(op['id'])/'ov'
    n = len(ids)
    check_labels(set_id, labels, meta, n, ('label',))
    value = dict(schema=1, id=name(set_id), label=label, labels=labels, op=op['id'], op_fp=op['fp'], n=n,
                 pairs=dict(set=name(pair), by='index') if pair else None,
                 images=dict(dataset=dataset, ids=np.asarray(ids).tolist()),
                 sample=dict(rule=sample_rule, of=int(sample_of), pos=np.asarray(pos).tolist()),
                 kinds=_child_kinds(directory, op, set_id, coords, (n,)), sprite=sprite, sprite_cols=sprite_cols,
                 meta=meta, provenance=provenance)
    path = directory/f'{set_id}.json'
    write_json(path, value, indent=None)
    return read_overlay(path, op)


def write_paths(root, op, set_id, *, coords, l0, sites, s, labels, meta, provenance, sprite=None):
    directory = Path(root)/'ops'/name(op['id'])/'paths'
    P, S = len(meta['pair_pos']), len(s)
    check_labels(set_id, labels, meta, P, ('label_a', 'label_b'))
    value = dict(schema=1, id=name(set_id), op=op['id'], op_fp=op['fp'], l0=l0, chain=op['chain'],
                 sites=list(sites), s=np.asarray(s).tolist(), P=P, S=S, labels=labels, meta=meta, provenance=provenance,
                 sprite=sprite, kinds=_child_kinds(directory, op, set_id, coords, (P, S)))
    path = directory/f'{set_id}.json'
    write_json(path, value, indent=None)
    return read_paths(path, op)


def _child(path, op):
    value = _read_json(path)
    require(value.get('op') == op['id'] and value.get('op_fp') == op['fp'], f'{path}: overlay/path fingerprint mismatch')
    require(value['id'] == path.stem and value['kinds'] and value['kinds'].keys() <= op['kinds'].keys(),
            f'{path}: invalid set ID or coordinate kinds')
    return value


def _read_children(path, op, value, prefix):
    for kind, spec in value['kinds'].items():
        require(spec['shape'] == [len(op['chain']), *prefix, op['K']] and spec['dtype'] == 'i16',
                f'{path}: {kind} shape or dtype mismatch')
        _array(path.parent, spec)


def read_overlay(path, op):
    path = Path(path)
    value = _child(path, op)
    n = value['n']
    pos = np.asarray(value['sample']['pos'])
    require(len(value['images']['ids']) == n and pos.shape == (n,) and pos.dtype.kind in 'iu'
            and (pos >= 0).all() and (pos < value['sample']['of']).all() and (np.diff(pos) > 0).all(),
            f'{path}: invalid overlay sampling metadata')
    check_labels(path, value.get('labels'), value.get('meta'), n, ('label',))
    for key in ('pred', 'success', 'rowmax'):
        data = value['meta'].get(key)
        require(data is None or len(data) == n, f'{path}: invalid {key} length')
    for scalar in value['meta'].get('scalars', {}).values():
        arr = np.asarray(scalar['values'], dtype=np.float64)
        require(arr.shape == (len(op['chain']), n) and not np.isinf(arr).any(), f'{path}: invalid overlay scalar grid')
    _read_children(path, op, value, (n,))
    return value


def read_paths(path, op):
    path = Path(path)
    value = _child(path, op)
    P, S = value['P'], value['S']
    require(value['chain'] == op['chain'] and all(s in op['chain'] for s in value['sites'])
            and len(value['s']) == S and np.all(np.diff(value['s']) > 0), f'{path}: invalid path grids')
    check_labels(path, value.get('labels'), value.get('meta'), P, ('label_a', 'label_b'))
    for key in ('pair_pos', 'kind', 's_switch'):
        data = value['meta'][key]
        require((key == 's_switch' and data is None) or len(data) == P, f'{path}: invalid {key} length')
    _read_children(path, op, value, (P, S))
    return value
