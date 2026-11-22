"""Real-data comparison; run from the repo root.

OPENBLAS_NUM_THREADS=1 uv run --with numpy --with scipy python -B opt/bench.py --calibrate
OPENBLAS_NUM_THREADS=1 uv run --with numpy --with scipy python -B opt/bench.py --fast-build --methods dense,normal --scales own --factors 0 3 5
Use --k 256 --ncv 384 with --scales own --factors 3 for the larger solve.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import resource
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import spacetime as S
import sparse_core as O


def timer():
    return time.perf_counter()


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def dense_balance(mask):
    """NumPy port of coherent.symmetric_balance (binary kernel; identical init and damped Newton)."""
    w = np.where(mask, 1., O.FLOOR)
    np.fill_diagonal(w, 0.)
    u = -.5 * np.log(w.sum(1))
    w = np.log(w, where=w > 0, out=np.full_like(w, -np.inf))
    for i in range(80):
        p = np.exp(w + u[:, None] + u)
        r = p.sum(1)
        err = np.abs(r - 1).max()
        if err < 2e-12:
            p *= .5
            np.fill_diagonal(p, .5)
            return p, i + 1, err
        if i < 10:
            u -= .5 * np.log(r)
            continue
        h = p.copy()
        h.flat[::len(h)+1] += r
        direction = np.linalg.solve(h, 1 - r)
        objective = .5 * p.sum() - u.sum()
        directional = (r - 1) @ direction
        for j in range(40):
            alpha = 2.**(-j)
            trial = u + alpha * direction
            tp = np.exp(w + trial[:, None] + trial)
            if (np.abs(tp.sum(1) - 1).max() < err or
                .5 * tp.sum() - trial.sum() <= objective + 1e-4 * alpha * directional):
                u = trial
                break
        else:
            raise RuntimeError('Newton line search stalled')
    raise RuntimeError('Newton did not converge')


def build(variant):
    times = defaultdict(float)
    ms, walks, nnz, iterations, errs = [], [], [], [], []
    for l in range(12):
        p = ROOT / f'data/grams/r20s0b/gram_64000_{l}.npy'
        q = timer()
        c = np.load(p, mmap_mode='r' if variant == 'sparse' else None)
        times['load'] += timer() - q
        q = timer()
        if variant == 'sparse':
            nn = O.knn(c)
            a = O.adjacency(nn)
            nnz.append(a.nnz)
        else:
            d = 1 - c
            np.fill_diagonal(d, np.inf)
            nn = np.argpartition(d, 15, axis=1)[:, :15].copy()
            del d
            a = np.zeros(c.shape, dtype=bool)
            a[np.arange(len(c))[:, None], nn] = True
            a |= a.T
            nnz.append(a.sum())
        times['knn'] += timer() - q
        q = timer()
        if variant == 'sparse':
            d, it, err = O.balance_weights(a)
            walks.append(O.make_walk(a, d))
        else:
            m, it, err = dense_balance(a)
            ms.append(m)
        times['balance'] += timer() - q
        iterations.append(it)
        errs.append(err)
    result = walks if variant == 'sparse' else np.stack(ms)
    print('build', variant, dict(times), 'primary directed nnz', sum(nnz), 'steps', iterations,
          'max row err', max(errs), 'HWM MiB', rss(), flush=True)
    return result


def archive_ref(f, s, i):
    g = f['vecs'][s, i].astype(np.float64) * f['vmax'][s, i, :, :, None] / 32767.
    g *= np.sqrt(f['weight'][s, i, :, :, None])
    return f['vals'][s, i, :16], g.transpose(1, 2, 0)


def dense_modes(ms, c, t, k, solver, ncv=None):
    old = S._eig_top
    diag = []

    def traced(mv, *args, **kwargs):
        q = timer()
        count = 0
        op_time = 0.

        def prod(x):
            nonlocal count, op_time
            b = timer()
            y = mv(x)
            op_time += timer() - b
            count += 1
            return y

        if ncv is not None:
            kwargs['ncv'] = ncv
        try:
            return old(prod, *args, **kwargs)
        finally:
            diag.extend([timer() - q, count, op_time])

    S._eig_top = traced
    try:
        return (*S.modes(ms, c, t, k=k, solver=solver), diag)
    finally:
        S._eig_top = old


def run(args):
    archive = dict(np.load(ROOT / 'data/r20s0b/spacetime/st_64000_0.npz'))
    variants = tuple(args.methods.split(',')) if args.methods else (
        ('dense', 'sparse', 'chol', 'float32') if args.variant == 'all' else (args.variant,))
    need_dense = any(v in ('dense', 'chol', 'float32', 'potri') for v in variants)
    need_sparse = args.fast_build or any(v in ('sparse', 'cal', 'lobpcg', 'normal', 'power') for v in variants)
    dense = build('dense') if need_dense and not args.fast_build else None
    walks = build('sparse') if need_sparse else None
    if need_dense and args.fast_build:
        q = timer()
        n = walks[0].u.size
        dense = np.empty((len(walks), n, n))
        for i, w in enumerate(walks):
            dense[i] = w.dense()
        print('dense materialization from sparse', timer() - q, 'HWM MiB', rss(), flush=True)
    elif dense is not None and walks is not None:
        print('max walk delta:', max(np.abs(w.dense() - m).max() for w, m in zip(walks, dense)), flush=True)
    if args.build_only:
        return
    if dense is not None and args.calibrate:
        q = timer()
        full = S.site_eigs(dense)
        print('dense full site eigs', timer() - q, 'HWM MiB', rss(), flush=True)
    else:
        full = None
    if walks is not None and args.calibrate:
        q = timer()
        partial = O.site_spectra(walks)
        print('sparse partial site eigs', timer() - q, 'HWM MiB', rss(),
              'archive eigen max abs delta', np.abs(partial[0] - archive['site_lam']).max(), flush=True)
        if full is not None:
            spec = full[0]
            s_err = abs(spec[:, :33] - partial[0]).max()
            tail_err = abs(spec[:, -9:][:, ::-1] - partial[2]).max()
            fro_err = abs(((2*spec[:, 1:]-1)**2).sum(axis=1) - partial[3]).max()
            print('spectra errors top, tail, Frobenius', s_err, tail_err, fro_err, flush=True)
    else:
        partial = None
    for s, mode in enumerate(('own', 'none')):
        if mode not in args.scales:
            continue
        c = archive['scales_' + mode]
        if full is not None:
            cs = (1 - full[0][:, 1:9]).mean(1) if mode == 'own' else np.ones(12)
            q = timer()
            tstar = S.balance_direct(dense, cs, eigs=full)
            print('dense calibration', mode, timer() - q, 't*', tstar, 't* relative err',
                  tstar / archive['tstar_' + mode] - 1, 'HWM MiB', rss(), flush=True)
        if walks is not None:
            joint = O.Joint(walks, c)
            if args.calibrate:
                cs = (1 - partial[0][:, 1:9]).mean(1) if mode == 'own' else np.ones(12)
                jc = O.Joint(walks, cs)
                for method in args.cal_methods.split(','):
                    if method == 'root':
                        q = timer()
                        tstar, counts = O.balance_root(jc, (1 - partial[0][:, 1]) / cs, partial[1][:, :, 0])
                        print('sparse calibration', mode, method, 'search', timer() - q, 't*', tstar,
                              't* relative err', tstar / archive['tstar_' + mode] - 1,
                              'root calls', len(counts), 'matvecs', sum(counts), 'HWM MiB', rss(), flush=True)
                        continue
                    q = timer()
                    cal = O.Calibration(jc, method=method)
                    setup = timer()-q
                    q = timer()
                    tstar = cal.balance()
                    print('sparse calibration', mode, method, 'setup', setup, 'search', timer() - q, 't*', tstar,
                          't* relative err', tstar / archive['tstar_' + mode] - 1,
                          'minv calls', len(cal.counts), 'total inner iters', sum(cal.counts), 'HWM MiB', rss(), flush=True)
                    del cal
            if args.variant == 'cal':
                continue
        for i, factor in enumerate(archive['factors']):
            if i not in args.factors:
                continue
            t = float(archive['t'][s, i])
            k = args.k
            ref = None
            early = {}
            for variant in variants:
                if variant == 'cal':
                    continue
                try:
                    prep_cpu = time.process_time()
                    q = timer()
                    if variant == 'dense':
                        sv = S.Solver(dense, c, t)
                    elif variant in ('chol', 'float32', 'potri'):
                        sv = O.CholeskyBlocks(dense, c, t, dtype=np.float32 if variant == 'float32' else np.float64,
                                              retain_inverse=variant == 'potri')
                    elif variant == 'sparse':
                        sv = O.SparseInverse(joint, t, tol=args.inner_tol)
                    elif variant not in ('lobpcg', 'normal', 'power'):
                        raise ValueError(f'Unknown method {variant!r}')
                    setup = timer() - q
                    prep_cpu = time.process_time() - prep_cpu
                    modes_cpu = time.process_time()
                    q = timer()
                    if variant == 'sparse':
                        w, u, calls = O.modes(joint.shape, t, k, sv.solve, joint.apply, ncv=args.ncv)
                        diag = ['calls', calls, 'CG total/max', sum(sv.counts), max(sv.counts),
                                'true solve residual', sv.max_residual]
                    elif variant == 'normal':
                        w, u, calls = O.normal_modes(joint, t, k, ncv=args.ncv)
                        diag = ['calls', calls]
                    elif variant == 'power':
                        w, u, calls = O.power_modes(joint, t, k, ncv=args.ncv, degree=args.degree)
                        diag = ['calls', calls, 'joint matvecs', calls*args.degree]
                    elif variant == 'lobpcg':
                        w, u, iterations = O.block_modes(joint, t, k)
                        diag = ['iterations+2', iterations]
                    else:
                        w, u, info = dense_modes(dense, c, t, k, sv, ncv=args.dense_ncv)
                        diag = ['eigsh s', info[0], 'calls', info[1], 'OP s', info[2]]
                    eigtime = timer() - q
                    modes_cpu = time.process_time() - modes_cpu
                    if variant not in ('sparse', 'lobpcg', 'normal', 'power'):
                        del sv
                    if variant == 'dense':
                        ref = w, u
                        for name, pair in early.items():
                            print('comparison to dense', mode, i, name, k, O.error(ref, pair), flush=True)
                        early.clear()
                    elif ref is None and 'dense' in variants:
                        early[variant] = w, u
                    av, au = archive_ref(archive, s, i)
                    m = min(k, 16)
                    arerr = O.error((av[:m], au[:, :, :m]), (w[:m], u[:, :, :m]))
                    mval = min(k, archive['vals'].shape[-1])
                    archive_vals = float(np.max(np.abs(w[:mval] / archive['vals'][s, i, :mval] - 1)))
                    accuracy = O.error(ref, (w, u)) if ref is not None and variant != 'dense' else None
                    if variant in ('sparse', 'lobpcg', 'normal', 'power'):
                        r = np.linalg.norm((joint.apply(u, t) - u*w).reshape(-1, k), axis=0)
                    else:
                        r = S.residual(dense, c, t, w, u)
                    print('modes', mode, i, factor, variant, k, 'prep', setup, 'modes', eigtime,
                          *diag, 'accuracy', accuracy, 'archive', arerr, 'relative eig residual',
                          max(r/abs(w)), 'HWM MiB', rss(), 'CPU s prep/modes', prep_cpu, modes_cpu,
                          'archive relative value error through', mval, archive_vals, flush=True)
                except (RuntimeError, np.linalg.LinAlgError) as e:
                    print('FAILED', mode, i, variant, k, repr(e), flush=True)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant', choices=['all', 'normal', 'dense', 'sparse', 'chol', 'float32', 'cal'], default='normal')
    p.add_argument('--methods', help='comma-separated subset: dense,normal,power,sparse,chol,potri,float32,lobpcg,cal')
    p.add_argument('--degree', type=int, default=12, help='power filter degree (power method only)')
    p.add_argument('--fast-build', action='store_true', help='build dense walks by materializing exact sparse walks')
    p.add_argument('--k', type=int, default=16)
    p.add_argument('--ncv', type=int, help='override the Krylov basis size for sparse modes')
    p.add_argument('--dense-ncv', type=int, help='optional dense comparison override; default is the legacy 4K')
    p.add_argument('--inner-tol', type=float, default=5e-13)
    p.add_argument('--factors', nargs='+', type=int, default=list(range(6)))
    p.add_argument('--scales', nargs='+', default=['own', 'none'])
    p.add_argument('--calibrate', action='store_true')
    p.add_argument('--cal-methods', default='root')
    p.add_argument('--build-only', action='store_true')
    return p


if __name__ == '__main__':
    run(parser().parse_args())
