"""Recompute walk sizes and transcribe archived solver timings and projections."""
from __future__ import annotations

import re
import sys

import numpy as np

from xlib import ROOT, Out

sys.path.insert(0, str(ROOT))
from opt import sparse_core


def token(key):
    return '{{fact:x.solver.' + key + '}}'


def main():
    src = ROOT / 'opt/REPORT.md'
    report = src.read_text()
    out = Out('solver')

    def row(heading):
        return next(line for line in report.splitlines() if line.startswith(f'| {heading} |'))

    def record(key, value, loc, *, scientific=False):
        value = value.replace(',', '')
        numeric = float(value) if any(c in value for c in '.e') else int(value)
        digits = len(value.partition('.')[2]) if '.' in value and 'e' not in value else None
        out.fact(key, numeric, digits=digits, tier='S', source=f'opt/REPORT.md:{loc}')
        if scientific:
            out.facts[key]['text'] = f'{numeric:.2g}'

    phases = {
        'Sinkhorn, all sites': ('balance.dense', 'balance.sparse'),
        'Site spectra for t*': ('spectra.dense', 'spectra.sparse_lo', 'spectra.sparse_hi'),
        't* search': ('calibration.dense', 'calibration.sparse_lo', 'calibration.sparse_hi', 'calibration.error'),
        '16-mode solve, t = t*': ('k16.dense_prep', 'k16.dense_modes', 'k16.sparse_modes', 'k16.speed_lo', 'k16.speed_hi'),
        '256-mode solve': ('k256.dense', 'k256.ncv', 'k256.sparse', 'k256.degree', 'k256.val_error', 'k256.space_error'),
    }
    number = r'\d+(?:\.\d+)?(?:e-\d+)?'
    storage_row = 'Walk storage, 12 sites'
    error = re.findall(number, row(storage_row).split('|', 2)[2])[-1]
    record('storage.entry_error', error, f'Headline table, {storage_row}', scientific=True)
    for label, keys in phases.items():
        cells = row(label).split('|', 2)[2].replace('(12 full eigendecompositions)', '')
        values = re.findall(number, cells)
        for k, value in zip(keys, values, strict=True):
            record(k, value, f'Headline table, {label}', scientific='e' in value)
    # Recover storage without rerunning timed phases. Only one mapped Gram is read at a time.
    grams = sorted((ROOT / 'data/grams/r20s0b').glob('gram_64000_*.npy'))
    assert len(grams) == 12
    dense = compact = 0
    for file in grams:
        gram = np.load(file, mmap_mode='r', allow_pickle=False)
        a = sparse_core.adjacency(sparse_core.knn(gram))
        walk = sparse_core.make_walk(a, sparse_core.balance_weights(a)[0])
        dense += gram.size * np.dtype(np.float64).itemsize
        compact += sum(arr.nbytes for arr in (walk.base.data, walk.base.indices, walk.base.indptr, walk.u))
    gram_src = 'data/grams/r20s0b/gram_64000_*.npy; opt/sparse_core.py:knn,adjacency,make_walk'
    out.fact('storage.dense', dense / 1e6, digits=0, tier='R', source=gram_src + '; float64 dense site walks in decimal MB')
    out.fact('storage.sparse', compact / 1e6, digits=2, tier='R', source=gram_src + '; CSR plus rank-one vector per site in decimal MB')
    for n in (10000, 50000):
        for k in (16, 256):
            label = f'{n:,} | {k}'
            cells = row(label).split('|')[3:-1]
            for version, cell in zip(('dense', 'sparse'), cells, strict=True):
                t, ram = cell.split('/')
                for metric, field in [('time', t), ('ram', ram)]:
                    lo, hi = re.findall(number, field)
                    root = f'projection.n{n}.k{k}.{version}_{metric}'
                    record(root + '_lo', lo, f'Projections (one coupling), N={n}, K={k}, {version} {metric}')
                    record(root + '_hi', hi, f'Projections (one coupling), N={n}, K={k}, {version} {metric}')
    n = 50000  # largest N in the report's projection table
    sites = len(grams)
    k = 256  # higher mode count in the projection table
    measured_ncv = out.facts['k256.ncv']['value']
    default_ncv = max(4*k, 40)  # opt/sparse_core.py:power_modes default
    out.fact('scale.n', n, tier='S', source='opt/REPORT.md: largest N in Projections table')
    out.fact('scale.gram', n**2 * np.dtype(np.float32).itemsize / 1e9, digits=0, tier='R', source='float32 itemsize × N²; decimal GB per Gram at projected N')
    out.fact('scale.grams', sites * out.facts['scale.gram']['value'], digits=0, tier='R', source='site count from Gram inventory × projected float32 Gram bytes; decimal GB')
    out.fact('scale.ncv_default', default_ncv, tier='R', source='opt/sparse_core.py:power_modes; max(4*K, 40) for 256 modes')
    for key, ncv in (('measured', measured_ncv), ('default', default_ncv)):
        out.fact(f'scale.basis_{key}', sites * n * ncv * np.dtype(np.float64).itemsize / 1e9, digits=2, tier='R',
                 source=f'float64 itemsize × chain size × projected N × ncv ({key}); decimal GB, basis alone')
    record('floor_drift', re.search(r'first 16 eigenvalues by up to ([\d.]+)%', report)[1], 'Headline, dropping the floor')
    card = dict(answer='An exact compact k-NN walk cut stored walk size and solver time at 2,000 images; larger runs are projections.',
                numbers=[token('storage.dense') + ' → ' + token('storage.sparse') + ' MB for 12 walks',
                         token('balance.dense') + ' → ' + token('balance.sparse') + ' s to balance walks',
                         token('k16.sparse_modes') + ' s for 16 joint modes'])
    thumb = out.fig_path('thumb')
    if thumb.exists():
        card['thumb'] = f'assets/fig/{thumb.name}'
    out.chart('card', card)
    out.write()


if __name__ == '__main__':
    main()
