"""Dev runner for xp/stage_x4.py: its own Modal app (crw-depth-xp-x4dev), same image and mounts as modal_xp.py.

    uv run --with modal modal run xp/dev_x4.py::main --units '[{"part":"checks"}]' --cfg '{"out":"x4dev"}'
    uv run --with modal modal run xp/dev_x4.py::main --units '[{"part":"paths","l0":"b9"}]' --cfg '{"out":"x4dev","n_pairs":4}'

Units run in parallel containers.  `modal run xp/dev_x4.py::run_diag` runs the float64 kink diagnostic.  Output goes to /results/<cfg.out>; cfg.out must start with 'x4dev'.
"""
from __future__ import annotations

import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
R = '/results'
FILES = ['common.py', 'xlib.py', 'dm.py', 'stage_x4.py']

image = (modal.Image.debian_slim(python_version='3.12')
         .pip_install('torch', 'numpy', 'scipy')
         .env({'PYTHONPATH': '/root/src:/root/cc'})
         .add_local_dir(str(Path.home() / 'code/ModelZoo/src'), remote_path='/root/src')
         .add_local_file(str(HERE.parent / 'spacetime.py'), remote_path='/root/cc/spacetime.py'))
for f in FILES:
    image = image.add_local_file(str(HERE / f), remote_path=f'/root/cc/{f}')
app = modal.App('crw-depth-xp-x4dev')
vol = modal.Volume.from_name('crw-depth-xp-results')
src = modal.Volume.from_name('cm-cifar-coherent').read_only()


@app.function(image=image, gpu=['L4', 'A10G'], cpu=8, memory=49152, timeout=3 * 3600, volumes={R: vol, '/out': src})
def x4(unit: dict, cfg: dict):
    import time
    import traceback
    import stage_x4
    t0 = time.time()
    vol.reload()
    try:
        out = stage_x4.run(unit, cfg)
    except Exception as e:                       # report rather than retry
        out = dict(unit=unit, status='error', error=repr(e), tb=traceback.format_exc()[-4000:])
    vol.commit()
    out['wall_s'] = time.time() - t0
    return out


@app.function(image=image, gpu=['L4', 'A10G'], cpu=8, memory=49152, timeout=1800, volumes={R: vol, '/out': src})
def diag():
    """Why float64 JVP and float64 central differences disagree from b7 on at the checks' points."""
    import copy
    import numpy as np
    import torch
    import stage_x4 as S
    ctx = S.Ctx()
    m64 = copy.deepcopy(ctx.model).double()
    pairs = np.concatenate(S.make_pairs(ctx.held, ctx.test_y, 4))
    imgs = ctx.test_x[pairs]
    A0, B0 = S.site_acts(ctx, imgs[:2, 0], 0).double(), S.site_acts(ctx, imgs[:2, 1], 0).double()
    sv = torch.tensor([0.0, 0.25, 0.5, 0.77, 1.0], device='cuda', dtype=torch.float64)
    X = ((1 - sv)[None, :, None, None, None] * A0[:, None] + sv[None, :, None, None, None] * B0[:, None]).reshape(-1, 3, 32, 32)
    V = (B0 - A0)[:, None].expand(2, len(sv), 3, 32, 32).reshape(X.shape).contiguous()
    f = lambda h: tuple(S.downstream(m64, 0, h))
    _, tg = torch.func.jvp(f, (X,), (V,))
    out = {}
    with torch.no_grad():
        for h in (1e-6, 1e-9):
            up, dn, mid = f(X + h * V), f(X - h * V), f(X)
            for k in (6, 7, 11):           # b6, b7, logits
                n = lambda t: t.flatten(1).norm(dim=1).cpu().numpy()
                j = n(tg[k])
                out[f'h{h}_site{k + 1}'] = dict(
                    central=np.round(S._relerr(j, n((up[k] - dn[k]) / (2 * h))), 8).tolist(),
                    fwd=np.round(S._relerr(j, n((up[k] - mid[k]) / h)), 8).tolist(),
                    bwd=np.round(S._relerr(j, n((mid[k] - dn[k]) / h)), 8).tolist())
        # smallest pre-activation magnitudes in block b7 (inner ReLU and output ReLU), relative to their median
        h6 = S.downstream(m64, 0, X, 7)[-1]
        blk = m64.blocks[6]
        pre1 = blk.bn1(blk.conv1(h6))
        o = blk.bn2(blk.conv2(blk.act(pre1)))
        sc = torch.nn.functional.pad(h6[:, :, ::2, ::2], (0, 0, 0, 0, 0, blk.pad_channels))
        pre2 = o + sc
        for nm, p in (('inner', pre1), ('out', pre2)):
            a = p.flatten(1).abs()
            out[f'b7_{nm}_min_rel'] = (a.min(1).values / a.median(1).values).cpu().numpy().tolist()
            out[f'b7_{nm}_exact_zero'] = (a == 0).sum(1).cpu().numpy().tolist()
        out['b7_bn2_zero_gamma'] = int((blk.bn2.weight == 0).sum())
        out['b7_bn1_zero_gamma'] = int((blk.bn1.weight == 0).sum())
        out['b7_bn2_gamma_absmin'] = float(blk.bn2.weight.abs().min())
    return out


@app.local_entrypoint()
def main(units: str, cfg: str = '{"out": "x4dev"}'):
    units, cfg = json.loads(units), json.loads(cfg)
    assert str(cfg.get('out', '')).startswith('x4dev'), 'dev runs write under /results/x4dev* only'
    for res in x4.starmap([(u, cfg) for u in units]):
        print(json.dumps(res, indent=1, default=str)[:12000])


@app.local_entrypoint()
def run_diag():
    print(json.dumps(diag.remote(), indent=1))
