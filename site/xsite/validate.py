"""Verify staged raw inputs and independent operator consistency before packaging."""
from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np

if __package__:
    from .fetch import FETCH, sha256
    from .opdata import require, source_from_joint
else:
    from fetch import FETCH, sha256
    from opdata import require, source_from_joint

INPUTS = Path(__file__).with_name('inputs.json')


class Inputs:
    def __init__(self, root=FETCH, manifest=INPUTS):
        self.root = Path(root)
        self.ready = json.loads((self.root/'READY.json').read_text())
        self.manifest = manifest if isinstance(manifest, dict) else json.loads(Path(manifest).read_text())
        require(self.manifest['schema'] == 1, 'Invalid raw input manifest schema')
        self._verified = {}

    def file(self, key, *, pin=None):
        p = Path(key)
        require(not p.is_absolute() and '..' not in p.parts, f'Invalid staged path {key}')
        path = self.root/p
        entry = self.ready.get(key)
        require(entry is not None, f'Unlisted raw input: {key}')
        if pin is not None:
            require(re.fullmatch(r'[0-9a-f]{64}', pin) and pin == entry['sha256'], f'{key}: pinned raw bundle hash mismatch')
        st = path.stat()
        require(st.st_size == entry['bytes'], f'{key}: READY size mismatch')
        revision = (st.st_dev, st.st_ino, st.st_size, st.st_ctime_ns, entry['sha256'])
        if self._verified.get(key) != revision:
            require(sha256(path) == entry['sha256'], f'{key}: READY hash mismatch')
            self._verified[key] = revision
        return path

    def bundle(self, op_id):
        record = self.manifest['ops'][op_id]
        pins = record['bundle']
        expected = {record['file'], record['probe']['viewer']}
        for ref in ('ids_from', 'labels_from'):
            key = record['probe'][ref].split(':', 1)[0]
            if key != 'op':
                expected.add(key)
        for category in ('overlays', 'paths'):
            for child in self.manifest[category].values():
                if child['op'] == op_id:
                    expected.add(child['file'])
        require(expected <= pins.keys(), f'{op_id}: incomplete pinned raw bundle')
        require(self.ready[record['file']]['volume_path'] == record['volume'], f'{op_id}: volume lineage mismatch')
        for key, digest in pins.items():
            self.file(key, pin=digest)
        return record

    def value(self, locator, *, raw=None):
        file, key = locator.rsplit(':', 1)
        if file == 'op':
            return raw[key]
        path = self.file(file)
        if path.suffix == '.npz':
            with np.load(path, allow_pickle=False) as obj:
                m = re.fullmatch(r'(\w+)\[(\w+)\]', key)
                return obj[m[1]][obj[m[2]]] if m else obj[key]
        return json.loads(path.read_text())[key]

    def provenance(self, op_id, *, key=None, entry=None):
        record = self.manifest['ops'][op_id]
        return dict(input=entry or f'inputs.json#ops/{op_id}', sha256=record['bundle'][key or record['file']],
                    code=record['producer'])


def check_probe(ids, labels, ref_ids, ref_labels):
    require(np.array_equal(ids, ref_ids), 'Probe ID order mismatch against viewer')
    require(np.array_equal(labels, ref_labels), 'Probe label order mismatch against viewer')


def validate_operator(inputs, op_id):
    """Return canonical unscaled operator data, after checking the complete pinned bundle."""
    record = inputs.bundle(op_id)
    suffix = record['chain'] == 'suffix'
    with np.load(inputs.file(record['file']), allow_pickle=False) as raw:
        ids = inputs.value(record['probe']['ids_from'], raw=raw)
        labels = inputs.value(record['probe']['labels_from'], raw=raw)
        ref = record['probe']['viewer']
        check_probe(ids, labels, inputs.value(ref+':idx'), inputs.value(ref+':labels'))
        chain_key = 'chain' if suffix else 'sites'
        chain = raw[chain_key].tolist() if chain_key in raw else record['sites']
        require(chain == record['sites'], f'{op_id}: raw site order mismatch')
        L, K = len(chain), record['K']
        prefix = 'op_' if suffix else ''
        c, vals, t = (raw[prefix+k] for k in ('c', 'vals', 't'))
        y = raw['probe_y' if suffix else 'vecs']
        require(y.ndim == 3 and y.shape[:2] == (L, len(ids)) and y.shape[-1] >= K
                and vals.ndim == 1 and len(vals) >= y.shape[-1] and c.shape == (L,)
                and np.isfinite(c).all() and (c > 0).all() and np.isfinite(t) and t > 0
                and np.isfinite(vals).all() and (np.diff(vals) >= 0).all(), f'{op_id}: invalid raw operator shape or values')
        Q = np.asarray(y, dtype=np.float64).reshape(-1, y.shape[-1])
        require(np.max(np.abs(Q.T @ Q - np.eye(y.shape[-1]))) < 1e-6, f'{op_id}: modes not orthonormal')
        require(np.max(np.abs(Q.reshape(y.shape).mean(1))) < 1e-8, f'{op_id}: modes not slice-mean-zero')
        checks = record['checks']
        if suffix:
            for k in ('replay_err', 'endpoint_err'):
                require(0 <= raw[k] < checks[k+'_max'], f'{op_id}: {k} too large')
            with np.load(inputs.file(checks['pairs_file']), allow_pickle=False) as pairs:
                require(all(np.array_equal(raw[k], pairs[k]) for k in ('pair_idx', 'kind', 'labels')),
                        f'{op_id}: raw path pair order mismatch')
            if 'equal_vals_to' in checks:
                other = inputs.manifest['ops'][checks['equal_vals_to']]['file']
                with np.load(inputs.file(other), allow_pickle=False) as z:
                    require(np.allclose(vals, z['vals'], rtol=0, atol=checks['equal_vals_atol']),
                            f'{op_id}: spectrum differs from claimed counterpart')
        else:
            resid = raw['resid']
            require(resid.shape == vals.shape and np.all((0 <= resid) & (resid < 1e-6)), f'{op_id}: mode residual too large')
        if 'stage_done' in checks:
            done = json.loads(inputs.file(checks['stage_done']).read_text())
            stage = done['knn15_'+record['chain']]
            require(done['net'] == record['net'] and done['probe'] == 'c10' and stage['t'] == t
                    and np.array_equal(stage['vals'], vals[:len(stage['vals'])]) and 0 <= stage['resid'] < 1e-6,
                    f'{op_id}: raw operator disagrees with producer DONE')
            if record['net'] == 'c10-s0':
                require(0 <= done['max_abs_diff_published_gram'] < 1e-5, f'{op_id}: published Gram replay failed')
        if 'stage_replay' in record['checks']:
            require(0 <= inputs.value(record['checks']['stage_replay']) < record['checks']['max'],
                    f'{op_id}: producer replay check failed')
        if not suffix and 'kind' in raw:
            require(str(raw['kind']) == record['kernel']['rule'], f'{op_id}: raw kernel rule mismatch')
            param = {'ea': 'ker_target', 'knn': 'ker_k'}.get(str(raw['kind']))
            if param:
                require(np.all(raw[param] == record['kernel']['param']), f'{op_id}: raw kernel parameter mismatch')
        site_key = 'probe_site_y' if suffix else 'site_vec'
        site = raw[site_key][:, :, :K] if site_key in raw else None
        if site is not None:
            require(site.shape == (L, len(ids), K) and np.isfinite(site).all(), f'{op_id}: invalid site-local probe coordinates')
        return dict(joint=y[:, :, :K], ids=ids, labels=labels, chain=chain, c=c, t=float(t), vals=vals[:K],
                    tstar=float(raw['tstar']) if 'tstar' in raw else float(t), site=site,
                    legacy_basis=float(np.abs(vals).sum()) if record['probe']['id'] == 'P2' else None)


def check_onestep(y, c, t, vals, onestep, *, rtol=1e-5):
    expected = source_from_joint(y, c, t, vals)
    require(expected.shape == np.shape(onestep), 'One-step source shape mismatch')
    scale = np.linalg.norm(onestep)
    err = np.linalg.norm(expected - onestep)
    require(np.isfinite(err) and err <= rtol * scale, f'One-step source mismatch (error {err:g}, norm {scale:g})')
    return err / scale if scale else 0.


def check_inversion_overlay(op, raw):
    require(op['legacy_basis'] is not None and np.isclose(raw['basis'], op['legacy_basis'], rtol=1e-12, atol=0),
            'Legacy inversion basis mismatch')
    K = len(op['vals'])
    return check_onestep(raw['y'][..., :K], op['c'], op['t'], op['vals'], raw['onestep'][..., :K])


def viewer_alignment(op, raw):
    """Sign vector against the published full ResNet-20 basis, and sorted-order check."""
    joint, K = op['joint'], len(op['vals'])
    v = raw['vecs'][0, 3, :K].astype(np.float64)
    weight = raw['weight'][0, 3, :K]
    ref = (v / np.linalg.norm(v, axis=-1, keepdims=True) * np.sqrt(weight)[..., None]).transpose(1, 2, 0)
    check_probe(op['ids'], op['labels'], raw['probe_idx'], op['labels'])
    require(np.allclose(raw['scales_own'], op['c'], atol=1e-10, rtol=0)
            and abs(raw['t'][0, 3]-op['t']) < 1e-8
            and np.allclose(raw['vals'][0, 3, :K], op['vals'], atol=1e-8, rtol=0), 'Viewer basis parameters mismatch')
    z, refz = joint.reshape(-1, K), ref.reshape(-1, K)
    cos = (z*refz).sum(0) / (np.linalg.norm(z, axis=0)*np.linalg.norm(refz, axis=0))
    require(np.all(np.abs(cos) > 1-1e-6), 'Viewer basis modes mismatch')
    return np.sign(cos).astype(int), ref


def check_x3_logits(op, raw, ref_logits, overlay, k, pos):
    """Independently replay the last-site row on saved logits (NumPy/SciPy vs producer torch)."""
    from scipy.special import logsumexp

    base = np.asarray(ref_logits, dtype=np.float64)
    base_mean = base.mean(0)
    base = base - base_mean
    query = np.asarray(overlay['logits'][pos], dtype=np.float64) - base_mean
    base /= np.maximum(np.linalg.norm(base, axis=1, keepdims=True), 1e-12)
    query /= np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    d = 1 - query @ base.T
    if k == 'knn15':
        near = np.argpartition(d, 14, axis=1)[:, :15]
        w = np.where(d <= raw['ker_rk'][-1], 1., float(raw['ker_floor'][-1]))
        np.put_along_axis(w, near, 1., axis=1)
        w *= raw['ker_v'][-1]
    else:
        require(k == 'ea100', f'Unknown X3 kernel {k}')
        ds = d - d.min(1, keepdims=True)
        lo, hi = np.zeros((len(d), 1)), np.full((len(d), 1), 1e6)
        for _ in range(60):
            beta = (lo + hi) / 2
            lp = -beta * ds
            logz = logsumexp(lp, axis=1, keepdims=True)
            H = logz + beta * (np.exp(lp-logz) * ds).sum(1, keepdims=True)
            lo, hi = np.where(H > np.log(100), beta, lo), np.where(H > np.log(100), hi, beta)
        lp = -((lo + hi) / 2) * ds
        lp -= logsumexp(lp, axis=1, keepdims=True)
        ref = -raw['ker_beta'][-1] * (d - raw['ker_dmin'][-1]) - raw['ker_logz'][-1]
        w = np.exp(np.logaddexp(lp, ref)) * raw['ker_v'][-1]
    w /= w.sum(1, keepdims=True)
    independent = w @ op['joint'][-1]
    reconstructed = source_from_joint(overlay[k+'_y'][:, pos], op['c'], op['t'], op['vals'])[-1]
    site = w @ op['site'][-1] / raw['site_lam'][-1]
    err = max(np.max(np.abs(independent - reconstructed)), np.max(np.abs(site - overlay[k+'_site_y'][-1, pos])))
    # The producer used float32 GPU activations/cosines and then promoted to float64 for rows.
    require(err < 2e-6, f'X3 {k} logits source/site replay error {err:g}')
    return err


def x1_split(labels, perm, within):
    rng = np.random.default_rng(0)
    N = len(labels)
    require(np.array_equal(rng.permutation(N), perm), 'X1 permutation mismatch')
    w = np.arange(N)
    for c in np.unique(labels):
        g = np.flatnonzero(labels == c)
        w[g] = rng.permutation(g)
    require(np.array_equal(w, within), 'X1 within-class permutation mismatch')
    return tuple(np.sort(half) for half in np.split(rng.permutation(N), 2))


def x1_align(a, b, train, test, expected):
    """One joint Q fitted on train; check stored test-site cosine without quantization."""
    K = a.shape[-1]
    a, b = a.astype(np.float64), b.astype(np.float64)
    U, _, Vt = np.linalg.svd(b[:, train].reshape(-1, K).T @ a[:, train].reshape(-1, K))
    Q = U @ Vt
    l, r = a[:, test], b[:, test] @ Q
    result = (l*r).sum((1, 2)) / (np.linalg.norm(l, axis=(1, 2))*np.linalg.norm(r, axis=(1, 2)))
    require(np.max(np.abs(result-expected)) < 1e-6, 'X1 joint agreement mismatch')
    return Q, result
