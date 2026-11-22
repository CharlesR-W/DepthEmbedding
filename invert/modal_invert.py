"""Modal pipeline for invert/SPEC.md.  Stages write named parts to the Volume `crw-depth-invert-results`.

    uv run --with modal python invert/modal_invert.py deploy
    uv run --with modal python invert/modal_invert.py launch <stage> '<json cfg>'   # spawns, detached
    uv run --with modal python invert/modal_invert.py status <call_id>
    uv run --with modal python invert/modal_invert.py fetch <remote path> <local dir>

Stages: extract -> gram -> kernel -> joint -> decode (decode lives in stage_decode.py, shipped in the image).
Every unit skips work whose output exists unless cfg['force'].
Rates (modal.com/pricing as recorded in CoherentModes measure_v2.py, 2026-11-12): CPU $0.0000131/core-s,
memory $0.00000222/GiB-s, GPU bounded above by A100-40GB $0.000583/s.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
import os
BASE = 'crw-depth-invert'
APP = os.environ.get('INVERT_APP', BASE)      # deploy iterating stages under another name; one shared Volume
RUN, STEP = 'resnet20-relu-seed0', 64000
SITES = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
ALL_SITES = ['pixels'] + SITES
MEAN, STD = (.4914, .4822, .4465), (.2470, .2435, .2616)
R = '/results'

image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch', 'numpy', 'scipy')
         .env({'PYTHONPATH': '/root/src:/root/cc'})
         .add_local_dir(str(Path.home() / 'code/ModelZoo/src'), remote_path='/root/src')
         .add_local_file(str(HERE.parent / 'spacetime.py'), remote_path='/root/cc/spacetime.py')
         .add_local_file(str(HERE / 'dm.py'), remote_path='/root/cc/dm.py')
         .add_local_file(str(HERE / 'stage_decode.py'), remote_path='/root/cc/stage_decode.py')
         .add_local_file(str(HERE / 'stage_rt.py'), remote_path='/root/cc/stage_rt.py')
         .add_local_file(str(HERE / 'stage_viz.py'), remote_path='/root/cc/stage_viz.py')
         .add_local_file(str(HERE / 'ddm.py'), remote_path='/root/cc/ddm.py')
         .add_local_file(str(HERE / 'stage_ddm.py'), remote_path='/root/cc/stage_ddm.py'))
app = modal.App(APP)
vol = modal.Volume.from_name(f'{BASE}-results', create_if_missing=True)
src = modal.Volume.from_name('cm-cifar-coherent').read_only()
zoo = modal.Volume.from_name('crw-modelzoo-train').read_only()
VOLS = {R: vol, '/out': src, '/zoo': zoo}


# ------------------------------------------------------------------ model helpers (run on Modal)

def load_model():
    import torch
    from zoo.io import load_checkpoint
    from zoo.models import build_model
    ck = load_checkpoint(Path(f'/out/runs/{RUN}/checkpoints/step_{STEP}.pt'))
    model = build_model(ck['config']); model.load_state_dict(ck['model_state']); model.eval().cuda()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return model, ck['config']


def normalize(x01):
    import torch
    m = torch.tensor(MEAN, device=x01.device).view(1, 3, 1, 1); s = torch.tensor(STD, device=x01.device).view(1, 3, 1, 1)
    return (x01 - m) / s


def activations(model, x):
    """[pixels, stem, b1..b9, pool, logits] for normalised input x."""
    h = model.act(model.bn1(model.conv1(x))); acts = [x, h]
    for block in model.blocks:
        h = block(h); acts.append(h)
    pool = h.mean((2, 3))
    return acts + [pool, model.fc(pool)]


def suffix(model, site: str, h):
    """Logits from an activation h (batch, shaped as the tap) given at `site`."""
    if site == 'logits':
        return h
    if site == 'pool':
        return model.fc(h)
    if site == 'pixels':
        return activations(model, h)[-1]
    start = SITES.index(site)                     # stem -> all blocks; b_i -> blocks[i:]
    for block in model.blocks[start:]:
        h = block(h)
    return model.fc(h.mean((2, 3)))


SHAPES = {'pixels': (3, 32, 32), 'stem': (16, 32, 32), 'b1': (16, 32, 32), 'b2': (16, 32, 32), 'b3': (16, 32, 32),
          'b4': (32, 16, 16), 'b5': (32, 16, 16), 'b6': (32, 16, 16), 'b7': (64, 8, 8), 'b8': (64, 8, 8),
          'b9': (64, 8, 8), 'pool': (64,), 'logits': (10,)}


# ------------------------------------------------------------------ stage: extract

@app.function(image=image, gpu='L4', cpu=4, memory=32768, timeout=3600, volumes=VOLS)
def extract(cfg: dict):
    import io, pickle, tarfile, urllib.request
    import numpy as np
    import torch
    from zoo.data import build_data
    t0 = time.time()
    vol.reload()
    out = Path(R) / 'acts'
    if (out / 'DONE').exists() and not cfg.get('force'):
        return 'cached'
    model, mcfg = load_model()
    data = build_data(mcfg, Path('/zoo/data'), 'cpu')
    tx, ty = data.test_x, data.test_y.numpy()
    assert tx.dtype == torch.uint8 and tx.shape == (10000, 3, 32, 32), (tx.dtype, tx.shape)
    assert torch.allclose(data.normalize(tx[:64]), normalize(tx[:64].float() / 255), atol=1e-5)

    rng = np.random.RandomState(0)
    p2 = np.sort(np.concatenate([rng.choice(np.where(ty == c)[0], 200, replace=False) for c in range(10)]))
    ref = np.load('/out/analysis/r20s0b/probe.npz')['idx']
    assert (p2 == ref).all(), 'probe indices differ from the published probe'
    rest = np.setdiff1d(np.arange(10000), p2)
    perm = np.random.RandomState(1).permutation(rest)
    p8 = np.sort(np.r_[p2, perm[:6000]]); hh = np.sort(perm[6000:])
    np.savez(Path(R) / 'splits.npz', P2=p2, P8=p8, H=hh, labels=ty)

    def run(x01, name, labels=None):
        """x01 float (m,3,32,32) in [0,1] on cpu."""
        store = [[] for _ in ALL_SITES]
        with torch.inference_mode():
            for at in range(0, len(x01), 250):
                for dst, a in zip(store, activations(model, normalize(x01[at:at + 250].cuda()))):
                    dst.append(a.flatten(1).float().cpu())
        d = out / name; d.mkdir(parents=True, exist_ok=True)
        logits = None
        for s, parts in zip(ALL_SITES, store):
            a = torch.cat(parts).numpy(); np.save(d / f'{s}.npy', a)
            if s == 'logits':
                logits = a
        np.savez(d / 'meta.npz', images=(x01.numpy() * 255).round().clip(0, 255).astype(np.uint8),
                 labels=np.asarray(labels if labels is not None else -np.ones(len(x01))), logits=logits)
        return logits

    info = {}
    x10 = tx.float() / 255
    lg = run(x10, 'c10', ty)
    info['clean_acc'] = float((lg.argmax(1) == ty).mean())
    assert abs(info['clean_acc'] - 0.9156) < 1e-4, info

    # CIFAR-100 test, first 2000 (cached npz on the volume, else download)
    try:
        z = np.load('/out/data/cifar100/cifar100_aadb3af77e90.npz')
        info['c100_keys'] = {k: list(z[k].shape) for k in z.files}
        kx = [k for k in z.files if 'test' in k and z[k].ndim >= 2][0]
        ky = [k for k in z.files if 'test' in k and z[k].ndim == 1][0]
        a100, y100 = z[kx][:2000], z[ky][:2000]
        a100 = a100.reshape(len(a100), 3, 32, 32) if a100.shape[-1] != 3 else a100.transpose(0, 3, 1, 2)
    except Exception as e:  # noqa: BLE001
        info['c100_npz_error'] = repr(e)
        raw = urllib.request.urlopen('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', timeout=300).read()
        with tarfile.open(fileobj=io.BytesIO(raw)) as tf:
            d = pickle.load(tf.extractfile('cifar-100-python/test'), encoding='bytes')
        a100, y100 = np.asarray(d[b'data'][:2000]).reshape(-1, 3, 32, 32), np.asarray(d[b'fine_labels'][:2000])
    assert a100.dtype == np.uint8 and a100.shape == (2000, 3, 32, 32), (a100.dtype, a100.shape)
    run(torch.from_numpy(np.ascontiguousarray(a100)).float() / 255, 'c100', y100)

    # SVHN test, first 2000 (network download; skipped if unavailable)
    try:
        from scipy.io import loadmat
        raw = urllib.request.urlopen('http://ufldl.stanford.edu/housenumbers/test_32x32.mat', timeout=120).read()
        m = loadmat(io.BytesIO(raw))
        xs = torch.from_numpy(m['X'].transpose(3, 2, 0, 1)[:2000].copy()).float() / 255
        run(xs, 'svhn', m['y'].ravel()[:2000] % 10)
        info['svhn'] = True
    except Exception as e:  # noqa: BLE001
        info['svhn'] = repr(e)

    # Gaussian pixel noise on H
    xh, yh = x10[hh], ty[hh]
    g = torch.Generator().manual_seed(0)
    for sig in (0.05, 0.1, 0.2):
        run((xh + sig * torch.randn(xh.shape, generator=g)).clamp(0, 1), f'noise{int(sig * 100):02d}', yh)

    # PGD L-inf on the first 1000 images of H (untargeted, 20 steps, step eps/4, random start)
    xa, ya = xh[:1000].cuda(), torch.from_numpy(yh[:1000]).cuda()
    for e255 in (2, 8):
        eps = e255 / 255
        torch.manual_seed(0)
        adv = (xa + (torch.rand_like(xa) * 2 - 1) * eps).clamp(0, 1)
        for _ in range(20):
            adv.requires_grad_(True)
            loss = torch.nn.functional.cross_entropy(activations(model, normalize(adv))[-1], ya)
            gr, = torch.autograd.grad(loss, adv)
            adv = (adv.detach() + eps / 4 * gr.sign()).clamp(xa - eps, xa + eps).clamp(0, 1)
        la = run(adv.detach().cpu(), f'pgd{e255}', yh[:1000])
        info[f'pgd{e255}_acc'] = float((la.argmax(1) == yh[:1000]).mean())
    info['seconds'] = time.time() - t0
    (out / 'DONE').write_text(json.dumps(info))
    vol.commit()
    return info


# ------------------------------------------------------------------ stage: gram

OOD = ['c100', 'svhn', 'noise05', 'noise10', 'noise20', 'pgd2', 'pgd8']


@app.function(image=image, gpu='L4', cpu=4, memory=49152, timeout=3600, volumes=VOLS, max_containers=13)
def gram(site: str, probe: str, cfg: dict):
    """Cosines among all 10,000 CIFAR-10 test images and from each other set to them, centred on `probe`."""
    import numpy as np
    import torch
    t0 = time.time()
    vol.reload()
    d = Path(R) / 'gram' / probe; d.mkdir(parents=True, exist_ok=True)
    if (d / f'{site}.done').exists() and not cfg.get('force'):
        return 'cached'
    sp = np.load(Path(R) / 'splits.npz')
    idx = sp[probe]
    x = torch.from_numpy(np.load(Path(R) / 'acts/c10' / f'{site}.npy')).cuda().double()
    mu = x[idx].mean(0)
    x -= mu
    nrm = x.norm(dim=1)
    x /= nrm.clamp_min(1e-12)[:, None]
    c = (x @ x.T).clamp(-1, 1)
    info = {}
    if probe == 'P2' and site != 'pixels':     # control: reproduce the published Gram
        ref = torch.from_numpy(np.load(f'/out/work/r20s0b/gram_{STEP}_{SITES.index(site)}.npy')).cuda()
        ii = torch.from_numpy(idx).cuda()
        info['max_abs_diff_published'] = float((c[ii][:, ii] - ref).abs().max())
        assert info['max_abs_diff_published'] < 1e-5, info
    np.save(d / f'{site}.npy', c.float().cpu().numpy())
    norms = {'c10': nrm.cpu().numpy()}
    for s in OOD:
        f = Path(R) / 'acts' / s / f'{site}.npy'
        if not f.exists():
            continue
        y = torch.from_numpy(np.load(f)).cuda().double() - mu
        ny = y.norm(dim=1); y /= ny.clamp_min(1e-12)[:, None]
        np.save(d / f'{site}_x_{s}.npy', (y @ x.T).clamp(-1, 1).float().cpu().numpy())
        norms[s] = ny.cpu().numpy()
    np.savez(d / f'{site}_norms.npz', mean=mu.float().cpu().numpy(), **norms)
    info['seconds'] = time.time() - t0
    (d / f'{site}.done').write_text(json.dumps(info))
    vol.commit()
    return info


# ------------------------------------------------------------------ stage: kernel

def index_set(sp, name):
    """'P2', 'P8', 'H', unions 'P2+H', and folds 'P2/f0' (train part) / 'P2/h0' (held-out part) of 5."""
    import numpy as np
    if '+' in name:
        return np.sort(np.concatenate([index_set(sp, n) for n in name.split('+')]))
    if '/' in name:
        base, f = name.split('/')
        idx = sp[base]
        fold = np.random.RandomState(7).permutation(len(idx)) % 5
        k = int(f[1:])
        return idx[fold != k] if f[0] == 'f' else idx[fold == k]
    return sp[name]


def centre_set(name):
    """Which probe's mean centres the Gram used for this index set."""
    return 'P8' if 'P8' in name else 'P2'


@app.function(image=image, cpu=8, memory=32768, timeout=7200, volumes=VOLS, max_containers=40)
def kernel(site: str, pset: str, kname: str, cfg: dict):
    """kname: 'ds<perp>' (calibrated), 'knn15', or 'ds<perp>@<pset0>' = eps taken from pset0's kernel."""
    import numpy as np
    import dm
    t0 = time.time()
    vol.reload()
    d = Path(R) / 'kernel' / f'{pset.replace("/", "_")}__{kname.replace("/", "_")}'; d.mkdir(parents=True, exist_ok=True)
    f = d / f'{site}.npz'
    if f.exists() and not cfg.get('force'):
        return 'cached'
    sp = np.load(Path(R) / 'splits.npz')
    idx = index_set(sp, pset)
    C = np.load(Path(R) / 'gram' / centre_set(pset) / f'{site}.npy', mmap_mode='r')
    C = np.asarray(C[idx][:, idx], dtype=np.float64)
    if '@' in kname:                              # 'ds<perp>@<pset0>': eps borrowed from pset0's kernel
        base, p0 = kname.split('@')
        eps = float(np.load(Path(R) / 'kernel' / f'{p0.replace("/", "_")}__{base}' / f'{site}.npz')['eps'])
        k = dm.ds_kernel(C, eps=eps)
    else:
        k = dm.build_kernel(C, kname)
    np.savez(f, **k)
    vol.commit()
    return dict(site=site, pset=pset, kname=kname, perp=k['perp'], err=k['err'], seconds=time.time() - t0)


# ------------------------------------------------------------------ stage: joint

def load_kernels(pset, kname, sites):
    import numpy as np
    d = Path(R) / 'kernel' / f'{pset.replace("/", "_")}__{kname.replace("/", "_")}'
    return [dict(np.load(d / f'{s}.npz')) for s in sites]


def set_rows(site, pset, target, ker, sp):
    """Out-of-sample rows (m, n) of `target` (an index-set name inside c10, or an OOD set) against pset."""
    import numpy as np
    import dm
    idx = index_set(sp, pset)
    g = Path(R) / 'gram' / centre_set(pset)
    if target in OOD:
        c = np.load(g / f'{site}_x_{target}.npy', mmap_mode='r')
        c = np.asarray(c[:, idx], dtype=np.float64)
    else:
        tid = index_set(sp, target)
        c = np.load(g / f'{site}.npy', mmap_mode='r')
        c = np.asarray(c[tid][:, idx], dtype=np.float64)
    return dm.rows(c, ker)


@app.function(image=image, cpu=16, memory=98304, timeout=6 * 3600, volumes=VOLS, max_containers=12)
def joint(pset: str, kname: str, cfg: dict):
    """Joint modes on pset, and Nystrom coordinates of every target set.  cfg: K, pixels (bool), targets."""
    import numpy as np
    import dm
    t0 = time.time()
    vol.reload()
    K = cfg.get('K', 64)
    sites = ALL_SITES if cfg.get('pixels') else SITES
    tag = f'{pset.replace("/", "_")}__{kname.replace("/", "_")}__L{len(sites)}'
    d = Path(R) / 'joint' / tag; d.mkdir(parents=True, exist_ok=True)
    sp = np.load(Path(R) / 'splits.npz')
    log = {}
    kers = load_kernels(pset, kname, sites)
    have = (d / 'modes.npz').exists() and len(np.load(d / 'modes.npz')['vals']) >= K
    if not have or cfg.get('force'):
        Ps = np.stack([k['P'] for k in kers])
        jm = dm.joint_modes(Ps, k=K)
        np.savez(d / 'modes.npz', vals=jm['vals'], vecs=jm['vecs'].astype(np.float32), c=jm['c'], t=jm['t'],
                 tstar=jm['tstar'], resid=jm['resid'], sites=np.array(sites))
        del Ps
        vol.commit()
        log['modes_s'] = time.time() - t0
    z = np.load(d / 'modes.npz')
    jm = dict(vals=z['vals'], vecs=z['vecs'].astype(np.float64), c=z['c'], t=float(z['t']))
    targets = cfg.get('targets', ['H'] + OOD)
    for tg in targets:
        f = d / f'coords_{tg.replace("/", "_")}.npz'
        basis = float(np.abs(jm['vals']).sum())                  # identifies the mode set the coordinates belong to
        if f.exists() and not (cfg.get('force') or cfg.get('force_coords')):
            if abs(float(np.load(f)['basis']) - basis) < 1e-12:
                continue
        if tg in OOD and not (Path(R) / 'acts' / tg).exists():
            continue
        rows = np.stack([set_rows(s, pset, tg, k, sp) for s, k in zip(sites, kers)])
        y, cond = dm.joint_forward(rows, jm)
        onestep = np.einsum('lmn,lnk->lmk', rows, jm['vecs'])
        # support score: log normaliser of the new point's row, minus the probe's median
        assert np.isfinite(y).all(), (tg, 'non-finite coordinates')
        np.savez(f, y=y.astype(np.float32), cond=cond, onestep=onestep.astype(np.float32), basis=basis,
                 rowmax=rows.max(2), rowperp=np.exp(-(rows * np.log(np.where(rows > 0, rows, 1))).sum(2)))
        vol.commit()
    log['seconds'] = time.time() - t0
    (d / 'log.json').write_text(json.dumps(log)); vol.commit()
    return dict(tag=tag, **log)


# ------------------------------------------------------------------ stage: decode (see stage_decode.py)

GPUS = ['L4', 'A10G', 'L40S', 'A100-40GB', 'A100-80GB', 'H100', 'T4']       # first available


@app.function(image=image, gpu=GPUS, cpu=8, memory=65536, timeout=6 * 3600, volumes=VOLS, max_containers=30)
def decode(unit: dict, cfg: dict):
    import stage_decode
    try:
        vol.reload()
    except RuntimeError:
        pass
    out = stage_decode.run(unit, cfg, sys.modules[__name__])
    vol.commit()
    return out


@app.function(image=image, gpu=GPUS, cpu=8, memory=65536, timeout=6 * 3600, volumes=VOLS, max_containers=30)
def rt(unit: dict, cfg: dict):
    import stage_rt
    try:
        vol.reload()
    except RuntimeError:          # a reused container may still hold memory-mapped inputs; they do not change
        pass
    out = stage_rt.run(unit, cfg, sys.modules[__name__])
    vol.commit()
    return out


@app.function(image=image, gpu='L4', cpu=8, memory=32768, timeout=480, volumes=VOLS, max_containers=2)
def rt_points_s(unit: dict, cfg: dict):
    """Bounded stretch run, writing only to a new rt_points subtree."""
    if not cfg.get('save_points'):
        raise ValueError('rt_points requires save_points; deploy it under a separate -s app')
    if cfg.get('points_dir', 'rt_points') not in ('rt_points', 'rt_points_retry'):
        raise ValueError('Expected a new rt_points output path')
    import stage_rt
    vol.reload()
    out = stage_rt.run(unit, cfg, sys.modules[__name__])
    vol.commit()
    return out


@app.function(image=image, gpu=GPUS, cpu=8, memory=65536, timeout=3600, volumes=VOLS)
def viz(unit: dict, cfg: dict):
    import stage_viz
    try:
        vol.reload()
    except RuntimeError:
        pass
    out = stage_viz.run(unit, cfg, sys.modules[__name__])
    vol.commit()
    return out


# float64 eigendecompositions: A100/H100 run double precision at half rate, the cheaper cards at 1/32 to 1/64
GPUS64 = ['A100-40GB', 'A100-80GB', 'H100']


@app.function(image=image, gpu=GPUS64, cpu=8, memory=65536, timeout=4 * 3600, volumes=VOLS, max_containers=12)
def ddmstage(unit: dict, cfg: dict):
    import stage_ddm
    try:
        vol.reload()
    except RuntimeError:
        pass
    out = stage_ddm.run(unit, cfg, sys.modules[__name__])
    vol.commit()
    return out


# diffusion-wavelet inversion (wavelets/SPEC_DW.md): float64 eigendecompositions on the GPU, pivoted QR on the CPU
image_dw = (image.add_local_file(str(HERE.parent / 'wavelets' / 'dw.py'), remote_path='/root/cc/dw.py')
            .add_local_file(str(HERE / 'stage_dw.py'), remote_path='/root/cc/stage_dw.py'))


@app.function(image=image_dw, gpu=GPUS64, cpu=16, memory=65536, timeout=3 * 3600, volumes=VOLS, max_containers=12)
def dwstage(unit: dict, cfg: dict):
    import stage_dw
    try:
        vol.reload()
    except RuntimeError:
        pass
    out = stage_dw.run(unit, cfg, sys.modules[__name__])
    vol.commit()
    return out


@app.function(image=image, cpu=4, memory=16384, timeout=3600, volumes=VOLS)
def summary(cfg: dict):
    """Means and standard errors over images of every array in decode/<tag>/*.npz and rt/<tag>/*.npz."""
    import numpy as np
    vol.reload()
    done = []
    for kind in ('decode', 'rt'):
        root = Path(R) / kind
        if not root.exists():
            continue
        for d in sorted(root.iterdir()):
            if cfg.get('tags') and d.name not in cfg['tags']:
                continue
            summ = {}
            for f in sorted(d.glob('*.npz')):
                z = np.load(f)
                summ[f.stem] = {k: [float(np.nanmean(z[k])), float(np.nanstd(z[k]) / max(1, z[k].size) ** .5), int(z[k].size)]
                                for k in z.files}
                # medians for quantities whose mean is dominated by a few images (near-duplicates of probe images
                # get overshooting out-of-sample coordinates, hence oversized decodes)
                summ[f.stem].update({k + '_med': [float(np.nanmedian(z[k])), 0.0, int(z[k].size)] for k in z.files
                                     if k.endswith(('|cap_raw', '|cap', '|kl', '|kl_on', '|cos_raw'))})
                j = d / f'{f.stem}.json'
                if j.exists():
                    summ[f.stem]['_meta'] = json.loads(j.read_text())
            (d / 'summary.json').write_text(json.dumps(summ))
            done.append(f'{kind}/{d.name}')
    vol.commit()
    return done


# ------------------------------------------------------------------ coordinator

@app.function(image=image, timeout=24 * 3600, volumes=VOLS)
def coordinator(stage: str, cfg: dict):
    def gather(gen):
        return [repr(r) if isinstance(r, Exception) else r for r in gen]

    out = {}
    if stage in ('extract', 'all'):
        out['extract'] = extract.remote(cfg)
    if stage in ('gram', 'all'):
        units = [(s, p, cfg) for p in cfg.get('centres', ['P2', 'P8']) for s in ALL_SITES]
        out['gram'] = gather(gram.starmap(units, return_exceptions=True))
    if stage in ('kernel', 'all', 'kj'):
        sites = ALL_SITES if cfg.get('pixels') else SITES
        units = [(s, p, k, cfg) for p, k in cfg['pk'] for s in sites]
        # kernels whose eps is borrowed ('@') need their base first
        first = [u for u in units if '@' not in u[2]]; second = [u for u in units if '@' in u[2]]
        out['kernel'] = gather(kernel.starmap(first, return_exceptions=True))
        if second:                                 # an empty starmap never returns
            out['kernel'] += gather(kernel.starmap(second, return_exceptions=True))
    if stage in ('joint', 'all', 'kj'):
        units = [(p, k, cfg) for p, k in cfg['pk']]
        out['joint'] = gather(joint.starmap(units, return_exceptions=True))
    if stage in ('decode',):
        out['decode'] = gather(decode.starmap([(u, cfg) for u in cfg['units']], return_exceptions=True))
    if stage in ('viz',):
        out['viz'] = gather(viz.starmap([(u, cfg) for u in cfg['units']], return_exceptions=True))
    if stage in ('rt',):
        out['rt'] = gather(rt.starmap([(u, cfg) for u in cfg['units']], return_exceptions=True))
    if stage in ('ddm',):
        out['ddm'] = gather(ddmstage.starmap([(u, cfg) for u in cfg['units']], return_exceptions=True))
    if stage in ('dw',) and cfg.get('units'):     # an empty starmap never returns
        out['dw'] = gather(dwstage.starmap([(u, cfg) for u in cfg['units']], return_exceptions=True))
    p = Path(R) / 'logs'; p.mkdir(exist_ok=True)
    (p / f'{stage}_{int(time.time())}.json').write_text(json.dumps(out, default=str))
    vol.commit()
    return out


# ------------------------------------------------------------------ local CLI

def main():
    cmd = sys.argv[1]
    if cmd == 'deploy':
        import subprocess
        subprocess.run([sys.executable, '-m', 'modal', 'deploy', __file__], check=True)
    elif cmd == 'launch':
        stage, cfg = sys.argv[2], json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        call = modal.Function.from_name(APP, 'coordinator').spawn(stage, cfg)
        rec = dict(stage=stage, cfg=cfg, call_id=call.object_id, launched=time.ctime())
        runs = HERE / 'runs'; runs.mkdir(exist_ok=True)
        (runs / f'{time.strftime("%Y%m%d-%H%M%S")}_{stage}.json').write_text(json.dumps(rec, indent=1))
        print(call.object_id)
    elif cmd == 'summary':
        print(modal.Function.from_name(APP, 'summary').remote(json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}))
    elif cmd == 'cancel':
        modal.FunctionCall.from_id(sys.argv[2]).cancel(terminate_containers=True)
    elif cmd == 'status':
        call = modal.FunctionCall.from_id(sys.argv[2])
        try:
            print(json.dumps(call.get(timeout=0), default=str)[:6000])
        except TimeoutError:
            print('running')
    elif cmd == 'fetch':
        remote, dest = sys.argv[2].strip('/'), Path(sys.argv[3])
        n = 0
        for e in vol.listdir('/' + remote, recursive=True):
            if e.type != modal.volume.FileEntryType.FILE:
                continue
            if len(sys.argv) > 4 and not any(pat in e.path for pat in sys.argv[4:]):
                continue
            target = dest / Path(e.path).relative_to(remote)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(b''.join(vol.read_file(e.path)))
            n += 1
        print(f'fetched {n} files to {dest}')


if __name__ == '__main__':
    main()
