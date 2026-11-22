"""Transcribe the archived solver benchmark and projections as summary-tier facts."""
from __future__ import annotations

import re

from xlib import ROOT, Out


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
        'Walk storage, 12 sites': ('storage.dense', 'storage.sparse', 'storage.entry_error'),
        'Sinkhorn, all sites': ('balance.dense', 'balance.sparse'),
        'Site spectra for t*': ('spectra.dense', 'spectra.sparse_lo', 'spectra.sparse_hi'),
        't* search': ('calibration.dense', 'calibration.sparse_lo', 'calibration.sparse_hi', 'calibration.error'),
        '16-mode solve, t = t*': ('k16.dense_prep', 'k16.dense_modes', 'k16.sparse_modes', 'k16.speed_lo', 'k16.speed_hi'),
        '256-mode solve': ('k256.dense', 'k256.ncv', 'k256.sparse', 'k256.degree', 'k256.val_error', 'k256.space_error'),
    }
    number = r'\d+(?:\.\d+)?(?:e-\d+)?'
    for label, keys in phases.items():
        cells = row(label).split('|', 2)[2].replace('(12 full eigendecompositions)', '')
        values = re.findall(number, cells)
        for k, value in zip(keys, values, strict=True):
            record(k, value, f'Headline table, {label}', scientific='e' in value)
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
