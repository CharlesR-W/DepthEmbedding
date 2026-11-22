from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest

if __package__:
    from .fetch import FETCH, sha256
    from .opdata import ids_bytes, read_array, read_operator, read_overlay, read_paths, source_from_joint
    from .ops import FIXTURES, OUT, manifest_for
    from .validate import (Inputs, check_inversion_overlay, check_probe, check_x3_logits,
                           validate_operator, x1_align, x1_split)
else:  # pytest also collects this file without package mode
    from fetch import FETCH, sha256
    from opdata import ids_bytes, read_array, read_operator, read_overlay, read_paths, source_from_joint
    from ops import FIXTURES, OUT, manifest_for
    from validate import (Inputs, check_inversion_overlay, check_probe, check_x3_logits,
                          validate_operator, x1_align, x1_split)


def fixture_op(op_id):
    return read_operator(FIXTURES/'ops'/op_id/'op.json')


def spec(i):
    return json.loads((FIXTURES/f'F{i}.json').read_text())


def test_f1_paired_overlay():
    fig = spec(1)
    assert fig['type'] == 'emb' and fig['schema'] == 1
    op = fixture_op(fig['panels'][0]['op'])
    folder = FIXTURES/'ops'/op['id']/'ov'
    clean = read_overlay(folder/'clean.json', op)
    assert clean['n'] == 20 and clean['pairs'] is None
    for item in fig['overlays']:
        over = read_overlay(folder/f"{item['set']}.json", op)
        assert over['pairs'] == {'set': 'clean', 'by': 'index'}
        assert over['images'] == clean['images'] and over['sample'] == clean['sample']
        for kind in ('joint', 'site', 'source'):
            x = read_array(folder, over['kinds'][kind])
            assert x.shape == (12, 20, 16) and np.isfinite(x).all()


def test_f2_switch_actual_chains():
    fig = spec(2)
    assert fig['site'] == 'b9' and len(fig['switch']) == 2
    lengths, pairs = [], []
    for option in fig['switch']:
        op = fixture_op(option['panels'][0]['op'])
        folder = FIXTURES/'ops'/op['id']/'paths'
        value = read_paths(folder/f"{option['paths'][0]}.json", op)
        assert len(value['sites']) == len(op['chain']) - 1
        assert value['chain'][0] == value['l0'] and fig['site'] in value['chain']
        assert (value['P'], value['S']) == (2, 65) and value['meta']['kind'] == ['same', 'diff']
        lengths.append(len(value['chain']))
        pairs.append(value['meta']['pair_pos'])
    assert lengths == [9, 3] and pairs[0] == pairs[1]


def test_f3_panel_swap():
    fig = spec(3)
    assert fig['panels'][0]['op'] == 'x1-c10-s0-trunk'
    assert len(fig['panelMenu']) == 2
    curves, rotations = [], []
    for item in fig['panelMenu']:
        op = fixture_op(item['op'])
        assert set(op['kinds']) == {'joint'} and op['kinds']['joint']['dtype'] == 'i8'
        R = np.asarray(item['R'])
        assert R.shape == (16, 16)
        np.testing.assert_allclose(R.T @ R, np.eye(16), atol=1e-12)
        assert len(item['stat']['values']) == len(op['chain']) == 10
        curves.append(item['stat']['values'])
        rotations.append(R)
    assert not np.allclose(*rotations) and not np.allclose(*curves)
    assert sum(f.stat().st_size for f in FIXTURES.rglob('*') if f.is_file()) <= 300_000


def test_stale_packaged_overlay_rejected(tmp_path):
    op = fixture_op('r20s0-c10-ea100')
    source = FIXTURES/'ops'/op['id']/'ov/u8.json'
    stale = json.loads(source.read_text())
    stale['op_fp'] = '0'*64  # right operator ID, wrong version of its signed basis
    path = tmp_path/source.name
    path.write_text(json.dumps(stale))
    with pytest.raises(ValueError, match='fingerprint mismatch'):
        read_overlay(path, op)


@pytest.mark.parametrize('op_id,rel,reader,field', [
    ('r20s0-c10-ea100', 'ov/u8.json', read_overlay, 'label'),
    ('x4-b3-suffix', 'paths/x4-b3-paths.json', read_paths, 'label_b'),
])
@pytest.mark.parametrize('bad_ids', [False, True], ids=['missing-labels', 'out-of-range'])
def test_invalid_set_labels_rejected(tmp_path, monkeypatch, op_id, rel, reader, field, bad_ids):
    monkeypatch.syspath_prepend(str(FIXTURES.parents[1]))
    from build_x import BuildError, Catalog

    op = fixture_op(op_id)
    root = tmp_path/'x'
    op_dir = root/'ops'/op_id
    shutil.copytree(FIXTURES/'ops'/op_id, op_dir)
    manifest = manifest_for(root)
    path = op_dir/rel
    value = json.loads(path.read_text())
    if bad_ids:
        value['meta'][field][0] = len(value['labels']['names'])
    else:
        del value['labels']
    path.write_text(json.dumps(value))
    match = 'outside labels.names' if bad_ids else 'labels namespace'
    with pytest.raises(ValueError, match=match):
        reader(path, op)

    # An up-to-date inventory must not make invalid descriptor contents buildable.
    manifest['files'][path.relative_to(root).as_posix()] = dict(bytes=path.stat().st_size, sha256=sha256(path))
    (root/'MANIFEST.json').write_text(json.dumps(manifest))
    with pytest.raises(BuildError, match=match):
        Catalog(root, True)


def test_build_rejects_permuted_viewer_export(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(FIXTURES.parents[1]))
    from build_x import BuildError, Catalog

    # Treat the small fixture's probe as viewer-order for this build test.
    root = tmp_path/'x'
    sample = 'r20s0-c10-ea100'
    op_dir = root/'ops'/sample
    shutil.copytree(FIXTURES/'ops'/sample, op_dir)
    op_path = op_dir/'op.json'
    op = json.loads(op_path.read_text())
    op['probe']['viewer_order'] = True
    op_path.write_text(json.dumps(op))
    (root/'MANIFEST.json').write_text(json.dumps(manifest_for(root)))
    catalog = Catalog(root, True)
    d = tmp_path/'d'
    d.mkdir()
    labels = (op_dir/'labels.bin').read_bytes()
    (d/'labels.bin').write_bytes(labels)
    meta = dict(n=len(labels), probe_ids_sha256=op['probe']['ids_sha256'],
                probe_labels_sha256=op['probe']['labels_sha256'])
    path = d/'meta.json'
    path.write_text(json.dumps(meta))
    catalog.check_viewer(d)

    # Same N and labels; only the export's ordered IDs differ.
    meta['probe_ids_sha256'] = hashlib.sha256(ids_bytes(np.roll(op['probe']['ids'], 1))).hexdigest()
    path.write_text(json.dumps(meta))
    with pytest.raises(BuildError, match='viewer probe ID order or count mismatch'):
        catalog.check_viewer(d)


def test_permuted_probe_rejected():
    path = FIXTURES/'ops/r20s0-c10-ea100/op.json'
    op = read_operator(path)
    ids = op['probe']['ids']
    labels = np.fromfile(path.parent/'labels.bin', dtype=np.uint8)
    perm = np.roll(np.arange(len(ids)), 1)
    with pytest.raises(ValueError, match='Probe ID order mismatch'):
        check_probe(np.asarray(ids)[perm], labels[perm], ids, labels)
    with pytest.raises(ValueError, match='probe order'):
        read_operator(path, ids=np.asarray(ids)[perm])


@pytest.fixture
def inputs():
    if not (FETCH/'READY.json').exists():
        pytest.skip('independent raw-data checks require the staged research outputs')
    return Inputs()


def test_raw_mode_sign_flip_rejected_before_packaging(inputs, tmp_path):
    op_id = 'r20s0-c10-ea100'
    rec = inputs.manifest['ops'][op_id]
    key = rec['file']
    # Place a genuine one-mode sign flip in a staged file and issue a fresh READY
    # record for it. The pinned raw bundle must still reject the replacement.
    changed = tmp_path/key
    changed.parent.mkdir(parents=True)
    with np.load(inputs.file(key), allow_pickle=False) as z:
        arrays = dict(z)
    arrays['vecs'][:, :, 0] *= -1
    np.savez(changed, **arrays)
    ready = {}
    for k in rec['bundle']:
        candidate = tmp_path/k
        if k != key:
            candidate.parent.mkdir(parents=True, exist_ok=True)
            candidate.symlink_to(inputs.file(k))
        ready[k] = {**inputs.ready[k], 'bytes': candidate.stat().st_size, 'sha256': sha256(candidate)}
    (tmp_path/'READY.json').write_text(json.dumps(ready))
    with pytest.raises(ValueError, match='pinned raw bundle hash mismatch'):
        validate_operator(Inputs(tmp_path, inputs.manifest), op_id)


def test_inversion_onestep_and_legacy_basis(inputs):
    op_id = 'r20s0-p2-pr300'
    op = validate_operator(inputs, op_id)
    key = inputs.manifest['overlays'][f'{op_id}/pgd2']['file']
    with np.load(inputs.file(key), allow_pickle=False) as z:
        subset = {k: z[k][:, :33, :16] if k in ('y', 'onestep') else z[k]
                  for k in ('y', 'onestep', 'basis')}
    assert check_inversion_overlay(op, subset) < 1e-6
    with pytest.raises(ValueError, match='Legacy inversion basis mismatch'):
        check_inversion_overlay(op, {**subset, 'basis': subset['basis'] + 1})
    subset['y'][:, :, 0] *= -1
    with pytest.raises(ValueError, match='One-step source mismatch'):
        check_inversion_overlay(op, subset)


@pytest.mark.parametrize('kernel', ('ea100', 'knn15'))
def test_roundtrip_actual_u8_and_independent_logits_replay(inputs, kernel):
    raw_id = f'r20s0-c10-{kernel}'
    rec = inputs.manifest['ops'][raw_id]
    op = validate_operator(inputs, raw_id)
    packed_dir = OUT/'ops'/raw_id
    if not (packed_dir/'op.json').exists():
        pytest.skip('production data not packaged yet')
    desc = read_operator(packed_dir/'op.json', ids=op['ids'])
    ov = read_overlay(packed_dir/'ov/u8.json', desc)
    clean = read_overlay(packed_dir/'ov/clean.json', desc)
    assert ov['images'] == clean['images'] and ov['sample'] == clean['sample']
    pos = ov['sample']['pos']
    key = inputs.manifest['overlays'][f'{raw_id}/u8']['file']
    with np.load(inputs.file(rec['file']), allow_pickle=False) as raw, np.load(inputs.file(key), allow_pickle=False) as u8:
        expected_joint = u8[kernel+'_y'][:, pos]
        expected_site = u8[kernel+'_site_y'][:, pos]
        expected_source = source_from_joint(expected_joint, op['c'], op['t'], op['vals'])
        for kind, expected in [('joint', expected_joint), ('site', expected_site), ('source', expected_source)]:
            zs = ov['kinds'][kind]
            diff = read_array(packed_dir/'ov', zs) - expected
            assert np.all(np.abs(diff) <= np.asarray(zs['scale'])[:, None, :] / 2 + 1e-12)
        zs = desc['kinds']['site']
        diff = read_array(packed_dir, zs) - raw['site_vec']
        assert np.all(np.abs(diff) <= np.asarray(zs['scale'])[:, None, :] / 2 + 1e-12)
        logits = inputs.value(rec['checks']['logits_from'])
        assert check_x3_logits(op, raw, logits, u8, kernel, pos[::37]) < 2e-6
        # Same magnitudes but signs in the probe modes are now inconsistent with
        # the signed source coordinate implied by the archived overlay.
        changed = {**op, 'joint': op['joint'].copy()}
        changed['joint'][..., 0] *= -1
        with pytest.raises(ValueError, match='logits source/site replay error'):
            check_x3_logits(changed, raw, logits, u8, kernel, pos[::37])


def test_x1_agreement_uses_own_rotation_and_real_split(inputs):
    a_id = 'x1-c10-s0-trunk'
    a = validate_operator(inputs, a_id)
    compare = inputs.manifest['ops'][a_id]['checks']['compare']
    with np.load(inputs.file(compare), allow_pickle=False) as z:
        train, test = x1_split(z['labels'], z['perm'], z['within'])
        stored = z['knn15_trunk_cos']
        names = z['nets'].tolist()
        for item in spec(3)['panelMenu']:
            net = inputs.manifest['ops'][item['op']]['net']
            b = validate_operator(inputs, item['op'])
            R, result = x1_align(a['joint'], b['joint'], train, test, stored[0, names.index(net)])
            np.testing.assert_allclose(R, item['R'], atol=1e-12)
            np.testing.assert_allclose(result, item['stat']['values'], atol=1e-6)
