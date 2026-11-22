"""Modal entry for wavelets/stage_joint_dw.py: the projected eigensolve of the viewer's L_t (CPU, float64).

    uv run --with modal python wavelets/modal_joint_dw.py run      # solve on Modal, write data/wavelets/joint_eig_64000.npz
Reads the r20s0b walks (rule 0) from the cm-cifar-coherent volume, read-only; new app name, no other app touched.
"""
from pathlib import Path
import io
import json
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
APP = 'depth-joint-dw'
TAG, STEP, RULE, NSITE = 'r20s0b', 64000, 0, 12
CORES, MEM_GB = 8, 32
CPU_RATE, MEM_RATE = .0000131, .00000222

image = (modal.Image.debian_slim(python_version='3.12').pip_install('numpy', 'scipy')
         .env({'PYTHONPATH': '/root/cc', 'OMP_NUM_THREADS': str(CORES), 'OPENBLAS_NUM_THREADS': str(CORES)})
         .add_local_file(str(ROOT / 'spacetime.py'), remote_path='/root/cc/spacetime.py')
         .add_local_file(str(HERE / 'stage_joint_dw.py'), remote_path='/root/cc/stage_joint_dw.py'))
app = modal.App(APP)
vol = modal.Volume.from_name('cm-cifar-coherent')


@app.function(image=image, cpu=CORES, memory=MEM_GB * 1024, timeout=3 * 3600, retries=0, volumes={'/out': vol})
def solve(c: list, t: float, probe_idx: bytes, k: int):
    import numpy as np
    import stage_joint_dw as D
    start = time.time()
    work, root = Path('/out/work') / TAG, Path('/out/analysis') / TAG
    pidx = np.load(root / 'probe.npz')['idx']
    assert np.array_equal(pidx, np.frombuffer(probe_idx, pidx.dtype)), 'volume probe order differs from the local probe'
    ms = np.stack([np.array(np.load(work / f'operators_{STEP}_{si}.npy', mmap_mode='r')[RULE], dtype=np.float64)
                   for si in range(NSITE)])
    assert np.abs(ms.sum(1) - 1).max() < 1e-9 and np.abs(ms - ms.transpose(0, 2, 1)).max() < 1e-12
    import spacetime as S
    c_re = S.site_scales(ms, 'own')
    print('scales recomputed vs given', float(np.abs(c_re / np.asarray(c) - 1).max()), flush=True)
    res = D.solve(ms, np.asarray(c, np.float64), float(t), k=k,
                  log=lambda *a: print(*a, f'{time.time() - start:.0f}s', flush=True))
    res['c_recomputed'] = c_re
    res['seconds'] = np.float64(time.time() - start)
    buf = io.BytesIO()
    np.savez(buf, **res)
    return buf.getvalue()


if __name__ == '__main__':
    import numpy as np
    sys.path.insert(0, str(HERE))
    import stage_joint_dw as D
    if sys.argv[1] == 'run':
        z = np.load(ROOT / 'data' / TAG / 'spacetime' / f'st_{STEP}_{RULE}.npz')
        c = z['scales_own'].tolist()
        t = float(z['t'][D.SCALE_INDEX, D.FACTOR_INDEX])
        pidx = np.load(ROOT / 'data' / TAG / 'probe.npz')['idx']
        assert np.array_equal(z['probe_idx'], pidx)
        t0 = time.time()
        with modal.enable_output(), app.run():
            blob = solve.remote(c, t, pidx.tobytes(), D.K_SOLVE)
        D.OUT.mkdir(parents=True, exist_ok=True)
        (D.OUT / f'joint_eig_{STEP}.npz').write_bytes(blob)
        e = np.load(io.BytesIO(blob))
        sec = float(e['seconds'])
        print(json.dumps(dict(seconds=sec, wall=time.time() - t0, est_usd=sec * (CORES * CPU_RATE + MEM_GB * MEM_RATE),
                              resid=float(e['resid'].max()), slice_mean=float(e['slice_mean'].max()))))
