"""Analysis stages for xp/SPEC.md.  Volume `crw-depth-xp-results`; training lives in modal_train.py.

    uv run --with modal python xp/modal_xp.py deploy
    uv run --with modal python xp/modal_xp.py launch <stage> ['<json cfg>']     # detached; prints the call id
    uv run --with modal python xp/modal_xp.py status <call_id>
    uv run --with modal python xp/modal_xp.py fetch <remote dir> <local dir> [substring ...]

Stages: selftest | embed | compare | x3 | x4.  Units skip work whose output exists unless cfg['force'].
embed: one unit per (network, probe): activations, Grams, kernels, joint modes (full chain and trunk).
"""
from __future__ import annotations

import json
import sys
import os
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = os.environ.get('XP_APP', 'crw-depth-xp')
R = '/results'
KERNELS = ['knn15', 'ea100']
CHAINS = {'full': list(range(12)), 'trunk': list(range(10))}       # indices into SITES
STAGE_FILES = ['common.py', 'xlib.py', 'dm.py', 'stage_compare.py', 'stage_x3.py', 'stage_x4.py']

image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch', 'numpy', 'scipy')
         .env({'PYTHONPATH': '/root/src:/root/cc'})
         .add_local_dir(str(Path.home() / 'code/ModelZoo/src'), remote_path='/root/src')
         .add_local_file(str(HERE.parent / 'spacetime.py'), remote_path='/root/cc/spacetime.py'))
for f in STAGE_FILES:
    if (HERE / f).exists():
        image = image.add_local_file(str(HERE / f), remote_path=f'/root/cc/{f}')
app = modal.App(APP)
vol = modal.Volume.from_name('crw-depth-xp-results', create_if_missing=True)
src = modal.Volume.from_name('cm-cifar-coherent').read_only()
VOLS = {R: vol, '/out': src}
GPUS = ['L4', 'A10G']


@app.function(image=image, gpu=GPUS, cpu=8, memory=32768, timeout=3600, volumes=VOLS)
def selftest(cfg: dict):
    """Row rules against dm; masked replay of probe images; the batched solve against dm.joint_forward."""
    import numpy as np
    import torch
    import dm, xlib
    rng = np.random.default_rng(0)
    n, L, K = 400, 4, 12
    out = {}
    Cs = []
    for l in range(L):
        a = rng.standard_normal((n, 30)) + 2 * rng.standard_normal((1, 30)) * (rng.integers(0, 3, (n, 1)) - 1)
        a -= a.mean(0); a /= np.linalg.norm(a, axis=1, keepdims=True)
        Cs.append(a @ a.T)
    new = [np.clip(C[:50] * 0.9 + 0.05 * rng.standard_normal((50, n)), -1, 1) for C in Cs]
    for kname in ('ea30', 'knn15'):
        kers = xlib.site_kernels(Cs, kname)
        eigs = xlib.site_spectra(kers)
        op = xlib.fit_op(kers, eigs, list(range(L)), K=K, R=8)
        ext = xlib.Ext(op, labels=rng.integers(0, 3, n))
        # (a) rows equal dm's
        d_rows = max(float(np.abs(ext.rows(l, torch.tensor(new[l]).cuda()).cpu().numpy() - dm.rows(new[l], kers[l])).max())
                     for l in range(L))
        # (b) masked replay returns P's rows and the modes
        masked = [torch.tensor(np.where(np.eye(n, dtype=bool), -np.inf, C)) for C in Cs]
        d_p = max(float(np.abs(ext.rows(l, masked[l].cuda()).cpu().numpy() - kers[l]['P']).max()) for l in range(L))
        e = ext.embed(masked)
        d_y = float(np.abs(e['y'] - op['vecs']).max())
        d_site = float(np.abs(e['site_y'] - op['site_vec']).max())
        # (c) the solve equals dm.joint_forward
        rows = np.stack([dm.rows(new[l], kers[l]) for l in range(L)])
        y_ref, cond = dm.joint_forward(rows, dict(vecs=op['vecs'], vals=op['vals'], c=op['c'], t=op['t']))
        d_fwd = float(np.abs(ext.embed([torch.tensor(c) for c in new])['y'] - y_ref).max())
        out[kname] = dict(rows=d_rows, replay_P=d_p, replay_modes=d_y, replay_site=d_site, forward=d_fwd,
                          vec_scale=float(np.abs(op['vecs']).max()), resid=float(op['resid'].max()))
        assert d_rows < 1e-9 and d_p < 1e-8 and d_fwd < 1e-5, out
        assert d_y < 1e-4 * max(1.0, float(ext.cond.max())) and d_site < 1e-4, out
    return out


@app.function(image=image, gpu=GPUS, cpu=8, memory=32768, timeout=3 * 3600, volumes=VOLS, max_containers=24,
              retries=modal.Retries(max_retries=3, initial_delay=5.0, backoff_coefficient=1.0))
def embed(net: str, probe: str, cfg: dict):
    import numpy as np
    import torch
    import xlib
    from common import SITES, activations, load_net, normalize
    t0 = time.time()
    vol.reload()
    d = Path(R) / 'embed' / net / probe; d.mkdir(parents=True, exist_ok=True)
    if (d / 'DONE').exists() and not cfg.get('force'):
        return 'cached'
    info = dict(net=net, probe=probe)
    pz = np.load(Path(R) / 'probes.npz')
    x01 = torch.from_numpy(pz[f'{probe}_x']).float() / 255
    model, mean, std = load_net(net)
    store = [[] for _ in SITES]
    with torch.inference_mode():
        for at in range(0, len(x01), 250):
            for dst, a in zip(store, activations(model, normalize(x01[at:at + 250].cuda(), mean, std))[1:]):
                dst.append(a.flatten(1).double())
    Cs, norms, knn = [], [], []
    for parts in store:
        C, _, nrm, _ = xlib.probe_gram(torch.cat(parts))
        knn.append((C - 3 * torch.eye(len(C), device='cuda', dtype=C.dtype)).topk(15, dim=1).indices.cpu().numpy().astype(np.int16))
        Cs.append(C.cpu().numpy()); norms.append(nrm.cpu().numpy())
    logits = torch.cat(store[-1]).cpu().numpy()
    del store
    torch.cuda.empty_cache()
    if net == 'c10-s0' and probe == 'c10':                   # control: the published Grams
        diffs = [float(np.abs(Cs[i] - np.load(f'/out/work/r20s0b/gram_64000_{i}.npy')).max()) for i in range(12)]
        info['max_abs_diff_published_gram'] = max(diffs)
        assert max(diffs) < 1e-5, diffs
    np.save(d / 'gram.npy', np.stack(Cs).astype(np.float32))
    np.save(d / 'knn.npy', np.stack(knn))
    np.savez(d / 'meta.npz', norms=np.stack(norms), logits=logits, labels=pz[f'{probe}_y'])
    info['extract_s'] = time.time() - t0
    for kname in cfg.get('kernels', KERNELS):
        t1 = time.time()
        kers = xlib.site_kernels(Cs, kname)
        eigs = xlib.site_spectra(kers)
        info[f'{kname}_sinkhorn_err'] = max(k['err'] for k in kers)
        info[f'{kname}_lam2'] = [float(v) for v in eigs[0][:, 1]]
        for chain, sel in CHAINS.items():
            op = xlib.fit_op(kers, eigs, sel, K=cfg.get('K', 32), R=16)
            np.savez(d / f'op_{kname}_{chain}.npz', sites=np.array([SITES[i] for i in sel]), **op)
            info[f'{kname}_{chain}'] = dict(t=float(op['t']), resid=float(op['resid'].max()), vals=op['vals'][:8].tolist())
        del kers, eigs
        info[f'{kname}_s'] = time.time() - t1
    info['seconds'] = time.time() - t0
    (d / 'DONE').write_text(json.dumps(info))
    vol.commit()
    return info


def _stage(name, unit, cfg):
    import importlib
    vol.reload()
    out = importlib.import_module(f'stage_{name}').run(unit, cfg)
    vol.commit()
    return out


@app.function(image=image, cpu=16, memory=65536, timeout=6 * 3600, volumes=VOLS, max_containers=8)
def compare(unit: dict, cfg: dict):
    return _stage('compare', unit, cfg)


@app.function(image=image, gpu=GPUS, cpu=8, memory=49152, timeout=6 * 3600, volumes=VOLS, max_containers=8)
def x3(unit: dict, cfg: dict):
    return _stage('x3', unit, cfg)


@app.function(image=image, gpu='L4', cpu=8, memory=49152, timeout=660, volumes=VOLS, max_containers=1)
def x3_pr(unit: dict, cfg: dict):
    if not cfg.get('replay_baseline'):
        raise ValueError('x3_pr requires replay_baseline; deploy it under a separate -s app')
    return _stage('x3', unit, cfg)


@app.function(image=image, gpu=GPUS, cpu=8, memory=49152, timeout=6 * 3600, volumes=VOLS, max_containers=12)
def x4(unit: dict, cfg: dict):
    return _stage('x4', unit, cfg)


@app.function(image=image, timeout=24 * 3600, volumes=VOLS)
def coordinator(stage: str, cfg: dict):
    from common import NETS, PROBES

    def gather(gen):
        return [repr(r) if isinstance(r, Exception) else r for r in gen]

    if stage == 'selftest':
        out = selftest.remote(cfg)
    elif stage == 'embed':
        units = [(n, p, cfg) for n in cfg.get('nets', list(NETS)) for p in cfg.get('probes', PROBES)]
        out = gather(embed.starmap(units, return_exceptions=True))
    else:
        fn = dict(compare=compare, x3=x3, x4=x4)[stage]
        out = gather(fn.starmap([(u, cfg) for u in cfg['units']], return_exceptions=True))
    p = Path(R) / 'logs'; p.mkdir(exist_ok=True)
    (p / f'{stage}_{int(time.time())}.json').write_text(json.dumps(out, default=str)); vol.commit()
    return out


def main():
    cmd = sys.argv[1]
    if cmd == 'deploy':
        import subprocess
        subprocess.run([sys.executable, '-m', 'modal', 'deploy', __file__], check=True)
    elif cmd == 'launch':
        stage = sys.argv[2]
        cfg = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        call = modal.Function.from_name(APP, 'coordinator').spawn(stage, cfg)
        runs = HERE / 'runs'; runs.mkdir(exist_ok=True)
        (runs / f'{time.strftime("%Y%m%d-%H%M%S")}_{stage}.json').write_text(
            json.dumps(dict(app=APP, stage=stage, cfg=cfg, call_id=call.object_id, launched=time.ctime()), indent=1))
        print(call.object_id)
    elif cmd == 'status':
        try:
            print(json.dumps(modal.FunctionCall.from_id(sys.argv[2]).get(timeout=0), default=str)[:int(sys.argv[3]) if len(sys.argv) > 3 else 6000])
        except TimeoutError:
            print('running')
    elif cmd == 'cancel':
        modal.FunctionCall.from_id(sys.argv[2]).cancel(terminate_containers=True)
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
