"""Detached rule-0 through-depth runs over the spectral-alignment networks.

Commands: deploy | pilot | launch PILOT_CALL_ID | retry CALL_ID | status CALL_ID | fetch [SHORT ...]
The coordinator returns per-unit timings and preserves them on the Volume.
"""
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import modal

HERE = Path(__file__).resolve().parent
APP = 'cm-cifar-coherent-spacetime-nets'
VOLUME = 'cm-cifar-coherent'
ROOT = Path('/out')
STEPS = [0, 64, 256, 1024, 2560, 6400, 25600, 64000]
NETS = {
    'r20s1': 'resnet20-relu-seed1',
    'r32s0': 'resnet32-relu-seed0',
    'r32s1': 'resnet32-relu-seed1',
    'r56s0': 'resnet56-relu-seed0',
    'r56s1': 'resnet56-relu-seed1',
}
PILOT = (NETS['r56s0'], 64000)
REFERENCE = ('resnet20-relu-seed0', 64000)
CORES, MEM_GIB = 8, 24
# Conservatively price each requested vCPU as one physical core.
RATE = CORES * .0000131 + MEM_GIB * .00000222
BUDGET = 15.0
VERSION = 5
app = modal.App(APP)
volume = modal.Volume.from_name(VOLUME)
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('numpy', 'scipy')
         .pip_install('torch', index_url='https://download.pytorch.org/whl/cpu')
         .env({'PYTHONPATH': '/root/cc', 'OMP_NUM_THREADS': str(CORES),
               'OPENBLAS_NUM_THREADS': str(CORES), 'MKL_NUM_THREADS': str(CORES),
               'SPACETIME_NETS_VERSION': str(VERSION)}))
for name in ('spacetime.py', 'spacetime_stage.py', 'coherent.py', 'seba.py'):
    image = image.add_local_file(str(HERE / name), remote_path='/root/cc/' + name)
image = image.add_local_file(str(HERE / 'upstream/spectral_align/nets.py'), remote_path='/root/cc/nets.py')
exact_image = (modal.Image.debian_slim(python_version='3.12')
               .pip_install('torch', 'numpy', 'scipy', 'dtw-python')
               .env({'PYTHONPATH': '/root/modelzoo:/root/cc', 'OMP_NUM_THREADS': '1'})
               .add_local_dir(str(Path.home() / 'code/ModelZoo/src'), '/root/modelzoo'))
for path in (HERE / 'coherent.py', HERE / 'seba.py', HERE / 'upstream/spectral_align/nets.py'):
    exact_image = exact_image.add_local_file(str(path), f'/root/cc/{path.name}')
zoo = modal.Volume.from_name('crw-modelzoo-train').read_only()


@app.function(name=f'exact_masks_v{VERSION}', image=exact_image, gpu='A100-40GB',
              cpu=2, memory=16384, timeout=900, max_containers=4,
              volumes={'/out': volume, '/zoo': zoo})
def exact_masks(net: str, step: int, indices, labels):
    """Replay the alignment extraction for walks whose support needs full precision."""
    import numpy as np
    import torch
    from coherent import centred_cosine_gram
    from nets import activations, build_model, site_names
    from zoo.data import build_data
    from zoo.io import load_checkpoint

    start = time.monotonic()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    volume.reload()
    ck = load_checkpoint(ROOT / 'runs' / net / 'checkpoints' / f'step_{step}.pt')
    model = build_model(ck['config'])
    model.load_state_dict(ck['model_state'])
    model.eval().cuda()
    data = build_data(ck['config'], Path('/zoo/data'), 'cpu')
    assert np.array_equal(data.test_y.numpy()[indices], labels)
    storage = [[] for _ in site_names(int(net.split('-')[0][6:]))]
    with torch.inference_mode():
        for start_index in range(0, len(indices), 128):
            batch = data.normalize(data.test_x[indices[start_index:start_index + 128]]).cuda()
            for site, activation in zip(storage, activations(model, batch)):
                site.append(activation.cpu())
    packed = []
    for site in storage:
        gram = centred_cosine_gram(torch.cat(site).cuda())
        distance = 1 - gram
        distance.diagonal().fill_(float('inf'))
        neighbors = distance.topk(15, largest=False, dim=1).indices
        mask = torch.zeros_like(gram, dtype=torch.bool)
        mask.scatter_(1, neighbors, True)
        packed.append(np.packbits((mask | mask.T).cpu().numpy()))
    return packed, time.monotonic() - start


def rebuild(source, si, exact_mask=None):
    """Recover the exact binary affinity support from a quantized eigensystem."""
    import numpy as np
    import torch
    from coherent import symmetric_balance

    vectors = np.load(source / f'fullU_0_{si}.npy').astype(np.float64)
    with np.load(source / f'cell_{si}.npz') as cell:
        values = cell['eig'][0]
    n = len(vectors)
    approx = (vectors * values[1:]) @ vectors.T + 1 / n
    # Most entries in each row are floor weights. Their medians are
    # proportional to the endpoint scaling factors, so estimate the
    # entrywise floor before thresholding the two affinity populations.
    symmetric = (approx + approx.T) * .5
    floor_scale = np.median(symmetric, axis=1)
    cutoff = np.maximum(1e-7, 1000 * np.outer(floor_scale, floor_scale) / np.median(floor_scale))
    mask = symmetric > cutoff if exact_mask is None else exact_mask
    np.fill_diagonal(mask, False)
    assert mask.sum(1).min() >= 15, (source, si, 'missing edges')
    weights = np.where(mask, 1.0, 1e-6)
    np.fill_diagonal(weights, 0)
    log_weights = torch.from_numpy(weights)
    log_weights.log_()
    walk = symmetric_balance(log_weights).numpy() * .5
    np.fill_diagonal(walk, .5)
    delta = np.abs(walk - approx)
    error = float(delta.max())
    # Two factors in U diag(lambda) U.T each carry float32 rounding.
    # Use entrywise residuals for the separation check so the rounded
    # diagonal cannot determine the tolerance for tiny off-diagonal entries.
    assert error < 2e-7, (source, si, 'spectral reconstruction mismatch', error)
    if exact_mask is None:
        assert np.all(walk[mask] - cutoff[mask] > 5 * delta[mask]), (source, si, 'ambiguous graph edge')
        np.fill_diagonal(mask, True)
        assert np.all(cutoff[~mask] - walk[~mask] > 5 * delta[~mask]), (source, si, 'ambiguous floor weight')
    return walk, error


def load_walks(net, step, probe, packed=None):
    import numpy as np
    from nets import site_names

    sites = site_names(int(net.split('-')[0][6:]))
    source = ROOT / 'work/salign' / net / str(step)
    with np.load(source / 'probe.npz') as measured:
        for key in ('idx', 'labels'):
            assert np.array_equal(measured[key], probe[key]), (net, step, key)
    n = len(probe['idx'])
    walks = np.empty((len(sites), n, n), dtype=np.float64)
    errors = []
    for si, site in enumerate(sites):
        mask = None if packed is None else np.unpackbits(packed[si], count=n*n).reshape(n, n).astype(bool)
        walk, error = rebuild(source, si, mask)
        assert np.abs(walk.sum(1) - 1).max() < 1e-9, (net, step, site)
        assert np.abs(walk - walk.T).max() < 1e-12, (net, step, site)
        walks[si] = walk
        errors.append(error)
    return sites, walks, max(errors)


def reference_errors(probe):
    import numpy as np

    sites, walks, _ = load_walks(*REFERENCE, probe)
    errors = {}
    for si, (site, walk) in enumerate(zip(sites, walks)):
        stored = np.load(ROOT / f'work/r20s0b/operators_64000_{si}.npy', mmap_mode='r')[0]
        errors[site] = float(np.abs(walk - stored).max())
    return errors


@app.function(name=f'unit_v{VERSION}', image=image, cpu=CORES, memory=MEM_GIB * 1024, timeout=3600,
              max_containers=16, retries=0, volumes={'/out': volume})
def unit(net: str, step: int):
    started = time.monotonic()
    import numpy as np
    import torch
    import spacetime_stage as stage

    torch.set_num_threads(CORES)
    volume.reload()
    target = ROOT / 'analysis/salign' / net / 'spacetime'
    timing = ROOT / 'work/salign' / net / 'spacetime' / f'timing_{step}_0.json'
    output = target / f'st_{step}_0.npz'
    if output.exists() and timing.exists():
        row = json.loads(timing.read_text())
        if row.get('version', 0) >= 3:
            return row

    with np.load(ROOT / 'analysis/r20s0b/probe.npz') as data:
        probe = {key: data[key] for key in ('idx', 'labels')}
    exact_seconds = 0.0
    try:
        sites, walks, error = load_walks(net, step, probe)
    except AssertionError as failure:
        print(net, step, 'rebuilding from full-precision activations:', str(failure), flush=True)
        packed, exact_seconds = exact_masks.remote(net, step, probe['idx'], probe['labels'])
        sites, walks, error = load_walks(net, step, probe, packed)
    rebuilt = time.monotonic()
    validation = reference_errors(probe) if (net, step) == PILOT else None
    if validation is not None:
        print('reference operator errors', validation, flush=True)

    solve_start = time.monotonic()
    result = stage.compute(walks, probe['labels'].astype(int), controls=False,
                           seed=STEPS.index(step),
                           log=lambda *args: print(net, step, *args, flush=True))
    failures = stage.accept(result)
    if failures:
        raise RuntimeError(f'{net} {step} failed acceptance: {failures}')
    target.mkdir(parents=True, exist_ok=True)
    timing.parent.mkdir(parents=True, exist_ok=True)
    result.update(probe_idx=probe['idx'], sites=np.array(sites))
    np.savez_compressed(output, **result)
    finished = time.monotonic()
    seconds = finished - started
    row = dict(net=net, step=step, seconds=seconds, solve_write_seconds=finished - solve_start,
               rebuild_elapsed=rebuilt - started, version=VERSION, exact_seconds=exact_seconds,
               estimated_usd=seconds * RATE + exact_seconds * (.000583 + 2 * .0000131 + 16 * .00000222),
               spectral_error=error, reference_errors=validation,
               worst_resid=float(max(result['resid'].max(), result['paper_resid'].max())))
    timing.write_text(json.dumps(row))
    volume.commit()
    return row


@app.function(name=f'coordinate_v{VERSION}', image=image, cpu=.25, memory=1024,
              timeout=7200, volumes={'/out': volume})
def coordinate(units: list):
    rows = []
    for index, row in enumerate(unit.starmap(units, return_exceptions=True)):
        net, step = units[index]
        if isinstance(row, Exception):
            row = dict(net=net, step=step, error=repr(row))
        rows.append(row)
        print(row, flush=True)
    spent = sum(row.get('estimated_usd', 0) for row in rows)
    return dict(rows=rows, spent=spent)


def fetch(shorts):
    import numpy as np
    import spacetime_stage as stage

    metadata = HERE / 'upstream/spectral_align/metadata'
    accuracies = json.loads((metadata / 'accuracies.json').read_text())
    with np.load(metadata / 'probe.npz') as probe:
        indices = probe['idx']
    for short in shorts:
        net = NETS[short]
        target = HERE / 'data/stnets' / short
        target.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp:
            subprocess.run(['modal', 'volume', 'get', VOLUME, f'/analysis/salign/{net}/spacetime/',
                            temp + '/'], check=True)
            files = Path(temp) / 'spacetime'
            sites = None
            for step in STEPS:
                path = files / f'st_{step}_0.npz'
                with np.load(path) as result:
                    names = result['sites'].tolist()
                    assert sites is None or sites == names, (net, step)
                    sites = names
                    assert np.array_equal(result['probe_idx'], indices), (net, step)
                    assert not stage.accept(result), (net, step)
            for step in STEPS:
                filename = f'st_{step}_0.npz'
                shutil.move(files / filename, target / filename)
            info = dict(net=net, sites=sites, steps=STEPS, test_acc=accuracies.get(net))
            (target / 'info.json').write_text(json.dumps(info, indent=2) + '\n')


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'deploy':
        subprocess.run(['modal', 'deploy', __file__], check=True)
    elif command == 'pilot':
        print(modal.Function.from_name(APP, f'coordinate_v{VERSION}').spawn([PILOT]).object_id)
    elif command == 'launch':
        pilot = modal.FunctionCall.from_id(sys.argv[2]).get(timeout=0)['rows'][0]
        assert (pilot['net'], pilot['step']) == PILOT and pilot.get('version', 0) >= 3
        assert max(pilot['reference_errors'].values()) < 1e-9, pilot['reference_errors']
        projection = (len(NETS) * len(STEPS) + 1) * pilot['estimated_usd']
        if projection > BUDGET:
            sys.exit(f'Projected spend ${projection:.2f} exceeds ${BUDGET:.2f}')
        units = [(net, step) for net in NETS.values() for step in STEPS if (net, step) != PILOT]
        print(f'Projected spend ${projection:.2f}', flush=True)
        print(modal.Function.from_name(APP, f'coordinate_v{VERSION}').spawn([REFERENCE, *units]).object_id)
    elif command == 'status':
        try:
            print(json.dumps(modal.FunctionCall.from_id(sys.argv[2]).get(timeout=0), indent=2))
        except TimeoutError:
            print('running')
    elif command == 'retry':
        previous = modal.FunctionCall.from_id(sys.argv[2]).get(timeout=0)
        failed = [(row['net'], row['step']) for row in previous['rows'] if 'error' in row]
        if failed:
            print(modal.Function.from_name(APP, f'coordinate_v{VERSION}').spawn(failed).object_id)
        else:
            print('All units passed')
    elif command == 'fetch':
        fetch(sys.argv[2:] or NETS)
    else:
        sys.exit(__doc__)
