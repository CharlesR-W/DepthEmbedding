"""Datasets and training for xp/SPEC.md X1.  Separate app so that redeploying the analysis app never touches a run.

    uv run --with modal python xp/modal_train.py deploy
    uv run --with modal python xp/modal_train.py launch data
    uv run --with modal python xp/modal_train.py launch train ['["c100-s0", ...]']   # default: all trained nets
    uv run --with modal python xp/modal_train.py status <call_id>

Recipe (ModelZoo's ResNet-20 recipe): SGD momentum 0.9, lr 0.1, wd 1e-4, batch 128, lr /10 at 50% and 75% of the
steps, pad-4 random crop and horizontal flip.  Augmentation is done on the GPU and the run is not bit-reproducible.
Training resumes from `last.pt` after a preemption; batch order and augmentation depend only on (seed, step).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
APP = 'crw-depth-xp-train'
R = '/results'
image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch', 'numpy', 'scipy', 'pyarrow', 'pillow')
         .env({'PYTHONPATH': '/root/src:/root/cc'})
         .add_local_dir(str(Path.home() / 'code/ModelZoo/src'), remote_path='/root/src')
         .add_local_file(str(HERE / 'common.py'), remote_path='/root/cc/common.py'))
app = modal.App(APP)
vol = modal.Volume.from_name('crw-depth-xp-results', create_if_missing=True)
zoo = modal.Volume.from_name('crw-modelzoo-train').read_only()
src = modal.Volume.from_name('cm-cifar-coherent').read_only()        # zoo checkpoints (ae_check only)
VOLS = {R: vol, '/zoo': zoo}


# ------------------------------------------------------------------ data

@app.function(image=image, cpu=8, memory=32768, timeout=3 * 3600, volumes=VOLS)
def data(cfg: dict):
    """Write /results/data/<name>/{train_x,train_y,test_x,test_y}.npy (uint8 N,3,32,32; int64) and probes.npz."""
    import io, pickle, tarfile, urllib.request
    import numpy as np
    t0 = time.time()
    vol.reload()
    root = Path(R) / 'data'
    info = {}

    def save(name, trx, tr_y, tex, te_y, **extra):
        d = root / name; d.mkdir(parents=True, exist_ok=True)
        for k, a in dict(train_x=trx, train_y=tr_y, test_x=tex, test_y=te_y, **extra).items():
            a = np.ascontiguousarray(a)
            if k.endswith('_x'):
                assert a.dtype == np.uint8 and a.shape[1:] == (3, 32, 32), (name, k, a.dtype, a.shape)
            else:
                a = a.astype(np.int64)
            np.save(d / f'{k}.npy', a)
        info[name] = dict(train=len(trx), test=len(tex), classes=int(np.max(tr_y)) + 1,
                          mean=(trx.reshape(len(trx), 3, -1).mean((0, 2)) / 255).tolist() if len(trx) < 2e5 else None)
        (d / 'DONE').write_text('ok'); vol.commit()

    if not (root / 'c10/DONE').exists():
        from zoo.data import load_dataset
        raw = load_dataset('cifar10', Path('/zoo/data'))
        save('c10', raw['train_x'].numpy(), raw['train_y'].numpy(), raw['test_x'].numpy(), raw['test_y'].numpy())

    if not (root / 'c100/DONE').exists():
        b = urllib.request.urlopen('https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz', timeout=600).read()
        with tarfile.open(fileobj=io.BytesIO(b)) as tf:
            tr = pickle.load(tf.extractfile('cifar-100-python/train'), encoding='bytes')
            te = pickle.load(tf.extractfile('cifar-100-python/test'), encoding='bytes')
        save('c100', np.asarray(tr[b'data']).reshape(-1, 3, 32, 32), np.asarray(tr[b'fine_labels']),
             np.asarray(te[b'data']).reshape(-1, 3, 32, 32), np.asarray(te[b'fine_labels']),
             train_coarse=np.asarray(tr[b'coarse_labels']), test_coarse=np.asarray(te[b'coarse_labels']))

    if not (root / 'svhn/DONE').exists():
        from scipy.io import loadmat
        parts = {}
        for split in ('train', 'test'):
            b = urllib.request.urlopen(f'http://ufldl.stanford.edu/housenumbers/{split}_32x32.mat', timeout=900).read()
            m = loadmat(io.BytesIO(b))
            parts[split] = (m['X'].transpose(3, 2, 0, 1).copy(), m['y'].ravel().astype(np.int64) % 10)
        save('svhn', *parts['train'], *parts['test'])

    if not (root / 'in32/DONE').exists():
        import pyarrow.parquet as pq
        from PIL import Image
        base = 'https://huggingface.co/datasets/benjamin-paine/imagenet-1k-32x32/resolve/main/data/'

        def read(fname):
            tmp = Path('/tmp') / fname
            urllib.request.urlretrieve(base + fname, tmp)
            t = pq.read_table(tmp).to_pydict()
            xs = np.stack([np.asarray(Image.open(io.BytesIO(im['bytes'])).convert('RGB')) for im in t['image']])
            assert xs.shape[1:] == (32, 32, 3), xs.shape
            tmp.unlink()
            return xs.transpose(0, 3, 1, 2), np.asarray(t['label'], dtype=np.int64)

        tr = [read(f'train-0000{i}-of-00003.parquet') for i in range(3)]
        vx, vy = read('validation-00000-of-00001.parquet')
        trx, tr_y = np.concatenate([a for a, _ in tr]), np.concatenate([b for _, b in tr])
        assert len(trx) == 1281167 and len(vx) == 50000 and tr_y.min() == 0 and tr_y.max() == 999, (len(trx), len(vx))
        assert np.bincount(vy, minlength=1000).min() == 50
        save('in32', trx, tr_y, vx, vy)

    # probes: 2,000 test images each, stratified by class, fixed seeds
    if not (Path(R) / 'probes.npz').exists():
        out = {}
        y10 = np.load(root / 'c10/test_y.npy')
        rng = np.random.RandomState(0)
        p = np.sort(np.concatenate([rng.choice(np.where(y10 == c)[0], 200, replace=False) for c in range(10)]))
        out['c10_idx'] = p
        for name, per, seed in (('c100', 20, 100), ('svhn', 200, 101), ('in32', 2, 102)):
            y = np.load(root / name / 'test_y.npy')
            rng = np.random.RandomState(seed)
            out[f'{name}_idx'] = np.sort(np.concatenate(
                [rng.choice(np.where(y == c)[0], per, replace=False) for c in range(int(y.max()) + 1)]))
        for name in ('c10', 'c100', 'svhn', 'in32'):
            idx = out[f'{name}_idx']
            assert len(idx) == 2000 and len(set(idx.tolist())) == 2000
            out[f'{name}_x'] = np.load(root / name / 'test_x.npy', mmap_mode='r')[idx]
            out[f'{name}_y'] = np.load(root / name / 'test_y.npy')[idx]
        out['c100_coarse'] = np.load(root / 'c100/test_coarse.npy')[out['c100_idx']]
        np.savez(Path(R) / 'probes.npz', **out)
        vol.commit()
    info['seconds'] = time.time() - t0
    (root / 'info.json').write_text(json.dumps(info)); vol.commit()
    return info


# ------------------------------------------------------------------ train

def lr_at(step, total, base=0.1):
    return base * (0.1 ** sum(step >= m for m in (total // 2, total * 3 // 4)))


@app.function(image=image, gpu='A10G', cpu=4, memory=32768, timeout=10 * 3600, volumes=VOLS, max_containers=16,
              retries=modal.Retries(max_retries=6, initial_delay=5.0, backoff_coefficient=1.0))
def train(name: str, cfg: dict):
    import numpy as np
    import torch
    import torch.nn.functional as F
    import math
    from common import NETS, build, build_ae
    assert torch.cuda.is_available()
    t0 = time.time()
    vol.reload()
    spec = NETS[name]
    # steps, lr and holdout overrides are for pilots: they need their own directory, and never a classifier
    assert not ({'steps', 'lr', 'holdout'} & set(cfg)) or (cfg.get('suffix') and spec.get('objective') == 'ae'), cfg
    out = Path(R) / 'nets' / (name + cfg.get('suffix', '')); out.mkdir(parents=True, exist_ok=True)
    if (out / 'final.pt').exists() and not cfg.get('force'):
        return 'cached'
    seed, total = spec['seed'], int(cfg.get('steps', spec.get('steps', 64000)))
    ae = spec.get('objective') == 'ae'                       # ae/SPEC.md: reconstruct the input instead of classifying
    flip, augment = spec.get('flip', True), spec.get('augment', True)
    d = Path(R) / 'data' / spec['data']
    trx, tr_y = np.load(d / 'train_x.npy'), np.load(d / 'train_y.npy')
    tex, te_y = np.load(d / 'test_x.npy'), np.load(d / 'test_y.npy')
    if spec.get('labels') == 'coarse':                      # CIFAR-100's 20 superclasses on the same images
        tr_y, te_y = np.load(d / 'train_coarse.npy'), np.load(d / 'test_coarse.npy')
    if 'classes' in spec:                                   # keep a subset of classes, relabelled 0..m-1
        remap = -np.ones(int(tr_y.max()) + 1, dtype=np.int64); remap[spec['classes']] = np.arange(len(spec['classes']))
        keep, keep_t = remap[tr_y] >= 0, remap[te_y] >= 0
        trx, tr_y, tex, te_y = trx[keep], remap[tr_y[keep]], tex[keep_t], remap[te_y[keep_t]]
    if spec.get('shuffle_labels'):
        tr_y = tr_y[np.random.RandomState(1234 + seed).permutation(len(tr_y))]
    ncls = int(tr_y.max()) + 1
    hold = int(cfg.get('holdout', 0))                        # pilots only: a validation split of the training images
    if hold:
        assert ae
        hp = np.random.RandomState(4321).permutation(len(trx))
        Xv = torch.from_numpy(trx[np.sort(hp[:hold])]).cuda()
        trx, tr_y = trx[np.sort(hp[hold:])], tr_y[np.sort(hp[hold:])]
    base_lr = float(cfg.get('lr', spec.get('lr', 0.1)))
    n = len(trx)
    X = torch.from_numpy(trx).cuda(); Y = torch.from_numpy(tr_y).cuda()
    Xt = torch.from_numpy(tex).cuda(); Yt = torch.from_numpy(te_y).cuda()
    mean = torch.stack([X[i:i + 100000].float().mean((0, 2, 3)) * len(X[i:i + 100000]) for i in range(0, n, 100000)]).sum(0) / n
    sq = torch.stack([(X[i:i + 100000].float() ** 2).mean((0, 2, 3)) * len(X[i:i + 100000]) for i in range(0, n, 100000)]).sum(0) / n
    std = (sq - mean ** 2).sqrt()
    mean, std = mean / 255, std / 255
    mv, sv = (mean * 255).view(1, 3, 1, 1), (std * 255).view(1, 3, 1, 1)

    torch.manual_seed(seed)
    model = (build_ae(spec['code']) if ae else build(ncls)).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    torch.backends.cudnn.benchmark = True
    bs, spe = 128, n // 128
    step, log = 0, []
    if (out / 'last.pt').exists() and not cfg.get('force'):
        ck = torch.load(out / 'last.pt', map_location='cuda', weights_only=False)
        model.load_state_dict(ck['model_state']); opt.load_state_dict(ck['opt_state']); step, log = ck['step'], ck['log']

    def evaluate():
        model.eval(); res = {}
        with torch.inference_mode():
            sub = torch.randperm(n, generator=torch.Generator().manual_seed(0))[:10000].cuda()
            for nm, (xx, yy) in dict(train=(X[sub], Y[sub]), test=(Xt, Yt), **(dict(val=(Xv, None)) if hold else {})).items():
                if ae:      # loss: MSE in normalised units (the training loss); psnr: of the clipped image, pixels in [0, 1]
                    se = se01 = 0.0
                    for i in range(0, len(xx), 2000):
                        xn = (xx[i:i + 2000].float() - mv) / sv
                        rec = model(xn)
                        se += F.mse_loss(rec, xn, reduction='sum').item()
                        se01 += (((rec * sv + mv).clamp(0, 255) - xx[i:i + 2000].float()) ** 2).sum().item() / 255 ** 2
                    res[f'{nm}_loss'], res[f'{nm}_psnr'] = se / xx.numel(), -10 * math.log10(se01 / xx.numel())
                    continue
                correct = loss = 0.0
                for i in range(0, len(xx), 2000):
                    lg = model((xx[i:i + 2000].float() - mv) / sv)
                    correct += (lg.argmax(1) == yy[i:i + 2000]).sum().item()
                    loss += F.cross_entropy(lg, yy[i:i + 2000], reduction='sum').item()
                res[f'{nm}_acc'], res[f'{nm}_loss'] = correct / len(xx), loss / len(xx)
        model.train()
        return res

    def save_last():
        torch.save(dict(model_state=model.state_dict(), opt_state=opt.state_dict(), step=step, log=log), out / 'last.tmp')
        (out / 'last.tmp').rename(out / 'last.pt'); vol.commit()

    ar = torch.arange(32, device='cuda')
    gc, perm, perm_epoch = torch.Generator(device='cuda'), None, -1
    model.train()
    every = max(4000, total // 16)
    while step < total:
        epoch, pos = divmod(step, spe)
        if epoch != perm_epoch:
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000003 + epoch)).cuda()
            perm_epoch = epoch
        idx = perm[pos * bs:(pos + 1) * bs]
        x, y = X[idx], Y[idx]
        if augment:
            gc.manual_seed(seed * 1000003 + 7919 * step + 17)
            oy = torch.randint(0, 9, (bs,), device='cuda', generator=gc); ox = torch.randint(0, 9, (bs,), device='cuda', generator=gc)
            xp = F.pad(x, (4, 4, 4, 4))
            x = xp[torch.arange(bs, device='cuda')[:, None, None, None], torch.arange(3, device='cuda')[None, :, None, None],
                   (oy[:, None] + ar)[:, None, :, None], (ox[:, None] + ar)[:, None, None, :]]
            if flip:
                f = torch.rand(bs, device='cuda', generator=gc) < 0.5
                x = torch.where(f[:, None, None, None], x.flip(-1), x)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, total, base_lr)
        xn = (x.float() - mv) / sv
        loss = F.mse_loss(model(xn), xn) if ae else F.cross_entropy(model(xn), y)
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        step += 1
        if ae and step % 200 == 0:
            assert torch.isfinite(loss), f'{name}: loss {loss.item()} at step {step}'
        if step % every == 0 or step == total:
            log.append(dict(step=step, loss=loss.item(), seconds=time.time() - t0, **evaluate()))
            print(name, log[-1], flush=True)
            if step < total:
                save_last()
    final = dict(model_state={k: v.cpu() for k, v in model.state_dict().items()}, num_classes=ncls,
                 mean=mean.cpu().tolist(), std=std.cpu().tolist(), spec=spec, steps=total, log=log, n_train=n)
    extra = {}
    if ae:
        model.eval()
        pz = np.load(Path(R) / 'probes.npz')
        px = torch.from_numpy(pz['c10_x']).cuda()
        with torch.inference_mode():        # the probe images as the network reconstructs them (uint8, as probes.npz)
            rec = torch.cat([(model((px[i:i + 500].float() - mv) / sv) * sv + mv).clamp(0, 255).round().byte()
                             for i in range(0, len(px), 500)])
        np.savez_compressed(out / 'recon.npz', idx=pz['c10_idx'], recon=rec.cpu().numpy())
        # control: rank-`code` PCA of the (unaugmented) training images, scored on the test images like test_psnr
        A = X.float().flatten(1) / 255
        mu = A.mean(0)
        A -= mu
        V = torch.linalg.eigh((A.T @ A).double())[1][:, -spec['code']:].float()
        T = Xt.float().flatten(1) / 255
        pca = ((((T - mu) @ V) @ V.T + mu).clamp(0, 1) - T).square().mean().item()
        mean_img = (((mu.view(1, -1)).clamp(0, 1) - T).square().mean().item())        # the mean training image as output
        with torch.inference_mode():
            xn = (Xt[:256].float() - mv) / sv
            codes = torch.cat([model.enc((Xt[i:i + 2000].float() - mv) / sv) for i in range(0, len(Xt), 2000)])
            alone = torch.cat([model(xn[i:i + 1]) for i in range(8)])       # eval mode: no dependence on the batch
            batch_dep = (alone - model(xn)[:8]).abs().max().item()
        extra = dict(objective='ae', code=spec['code'], pca_psnr=-10 * math.log10(pca), lr=base_lr,
                     mean_image_psnr=-10 * math.log10(mean_img), code_std=codes.std(0).cpu().tolist(),
                     eval_batch_dependence=batch_dep,
                     params_encoder=sum(p.numel() for p in model.enc.parameters()),
                     params_decoder=sum(p.numel() for p in model.dec.parameters()))
    torch.save(final, out / 'final.pt')
    (out / 'info.json').write_text(json.dumps(dict(name=name, num_classes=ncls, steps=total, n_train=n,
                                                   mean=final['mean'], std=final['std'], log=log, **extra)))
    vol.commit()
    return log[-1]


@app.function(image=image, gpu='A10G', cpu=4, memory=16384, timeout=1800, volumes={**VOLS, '/out': src})
def ae_check(cfg: dict):
    """ae/SPEC.md pre-flight: (a) an autoencoder's encoder starts from the classifier's initial weights under the same
    seed (and whether those equal the zoo's step-0 checkpoint); (b) the autoencoder can fit one fixed batch: its
    train-mode MSE after 1,500 steps must be under a quarter of the MSE of the batch's mean image."""
    import hashlib
    import numpy as np
    import torch
    import torch.nn.functional as F
    from common import build, build_ae
    out = {}
    for seed in (0, 1):
        torch.manual_seed(seed); ref = build(10).state_dict()
        out[f's{seed}_init_sha256'] = hashlib.sha256(b''.join(v.numpy().tobytes() for v in ref.values())).hexdigest()
        for code in (10, 64):
            torch.manual_seed(seed); enc = build_ae(code).enc.state_dict()
            keys = [k for k in ref if code == 10 or not k.startswith('fc.')]
            out[f's{seed}_ae{code}_encoder_equals_classifier_init'] = all(torch.equal(ref[k], enc[k]) for k in keys)
            assert out[f's{seed}_ae{code}_encoder_equals_classifier_init'], out
            out[f's{seed}_ae{code}_tensors_compared'] = len(keys)
        try:
            from zoo.io import load_checkpoint
            ck = load_checkpoint(Path(f'/out/runs/resnet20-relu-seed{seed}/checkpoints/step_0.pt'))['model_state']
            out[f's{seed}_zoo_step0'] = 'match' if all(torch.equal(ref[k], ck[k].cpu()) for k in ref) else 'mismatch'
        except Exception as e:
            out[f's{seed}_zoo_step0'] = f'unavailable: {e!r}'
    x = torch.from_numpy(np.load(Path(R) / 'data/c10/train_x.npy')[:128]).cuda().float()
    xn = (x - x.mean((0, 2, 3), keepdim=True)) / x.std((0, 2, 3), keepdim=True)
    torch.manual_seed(0); model = build_ae(10).cuda()
    opt = torch.optim.SGD(model.parameters(), lr=float(cfg.get('lr', 0.1)), momentum=0.9, weight_decay=1e-4)
    fit = []
    for step in range(1501):
        loss = F.mse_loss(model(xn), xn)
        if step % 250 == 0:
            fit.append(loss.item())
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    model.eval()
    with torch.inference_mode():
        out['one_batch_eval_mse'] = F.mse_loss(model(xn), xn).item()
    out['one_batch_train_mse_every_250'] = fit
    out['one_batch_mean_image_mse'] = (xn - xn.mean(0, keepdim=True)).square().mean().item()   # constant-output level
    assert fit[-1] < 0.25 * out['one_batch_mean_image_mse'], out
    return out


@app.function(image=image, timeout=24 * 3600, volumes=VOLS)
def coordinator(stage: str, cfg: dict):
    from common import TRAINED
    if stage == 'data':
        return data.remote(cfg)
    if stage == 'ae_check':
        return ae_check.remote(cfg)
    names = cfg.get('nets', TRAINED)
    res = list(train.starmap([(n, cfg) for n in names], return_exceptions=True))
    out = {n: (repr(r) if isinstance(r, Exception) else r) for n, r in zip(names, res)}
    p = Path(R) / 'logs'; p.mkdir(exist_ok=True)
    (p / f'train_{int(time.time())}.json').write_text(json.dumps(out, default=str)); vol.commit()
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
            print(json.dumps(modal.FunctionCall.from_id(sys.argv[2]).get(timeout=0), default=str)[:6000])
        except TimeoutError:
            print('running')
    elif cmd == 'cancel':
        modal.FunctionCall.from_id(sys.argv[2]).cancel(terminate_containers=True)


if __name__ == '__main__':
    main()
