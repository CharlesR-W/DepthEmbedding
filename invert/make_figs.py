"""Image grids from the decoded-image arrays (stage_viz).  uv run --with numpy --with pillow python invert/make_figs.py [tag]"""
import sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent / 'data/invert'
tag = sys.argv[1] if len(sys.argv) > 1 else 'P2__ea300__L13'
z = np.load(ROOT / 'viz' / f'{tag}.npz')
OUT = ROOT / 'figs'; OUT.mkdir(exist_ok=True)
SC, PAD, HEAD = 3, 2, 14
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


def grid(cols, heads, name, rows=None):
    """cols: list of (n, 32, 32, 3) uint8 arrays; one column each, with a header."""
    n = len(cols[0]) if rows is None else rows
    cell = 32 * SC
    W = len(cols) * (cell + PAD) + PAD; H = HEAD + n * (cell + PAD) + PAD
    im = Image.new('RGB', (W, H), (255, 255, 255)); dr = ImageDraw.Draw(im)
    for j, (c, h) in enumerate(zip(cols, heads)):
        x = PAD + j * (cell + PAD)
        dr.text((x + 2, 1), h, fill=(0, 0, 0))
        for i in range(n):
            im.paste(Image.fromarray(c[i]).resize((cell, cell), Image.NEAREST), (x, HEAD + PAD + i * (cell + PAD)))
    im.save(OUT / f'{tag}_{name}.png')


sites = ['pixels', 'stem', 'b3', 'b6', 'b9', 'pool']
n = 12
for K in (16, 64):
    if f'decs|b9|{K}|H' in z:
        grid([z['orig|H'][:n]] + [z[f'decs|{s}|{K}|H'][:n] for s in sites], ['original'] + [f'src {s}' for s in sites], f'depthsrc_K{K}')
        for adv in ('pgd8', 'pgd2'):
            if f'orig|{adv}' in z:
                grid([z['orig|H'][:n], z[f'orig|{adv}'][:n]] + [c for s in ('pixels', 'b3', 'b6', 'b9', 'pool') for c in (z[f'decs|{s}|{K}|H'][:n], z[f'decs|{s}|{K}|{adv}'][:n])],
                     ['clean', 'attacked'] + [h for s in ('pixels', 'b3', 'b6', 'b9', 'pool') for h in (f'{s} clean', f'{s} att.')], f'{adv}src_K{K}')
        for t in ('c100', 'svhn', 'noise20'):
            if f'orig|{t}' in z:
                grid([z[f'orig|{t}'][:n]] + [z[f'decs|{s}|{K}|{t}'][:n] for s in sites], ['original'] + [f'src {s}' for s in sites], f'{t}src_K{K}')
        for s in ('pixels', 'b6', 'b9'):
            p = z[f'paths|{s}|{K}']
            grid([z['orig|H'][:8]] + [p[:, j] for j in range(9)] + [z['orig|H'][8:16]],
                 ['image a'] + [f'{t:+.2f}' for t in np.linspace(-0.5, 1.5, 9)] + ['image b'], f'pathsrc_{s}_K{K}')
    grid([z['orig|H'][:n]] + [z[f'dec|{s}|{K}|H'][:n] for s in sites], ['original'] + [f'from {s}' for s in sites], f'depth_K{K}')
    for adv in ('pgd8', 'pgd2'):
        if f'orig|{adv}' in z:
            grid([z['orig|H'][:n], z[f'orig|{adv}'][:n]] + [c for s in ('pixels', 'b3', 'b6', 'b9') for c in (z[f'dec|{s}|{K}|H'][:n], z[f'dec|{s}|{K}|{adv}'][:n])],
                 ['clean', 'attacked'] + [h for s in ('pixels', 'b3', 'b6', 'b9') for h in (f'{s} clean', f'{s} att.')], f'{adv}_K{K}')
    for t in ('c100', 'svhn', 'noise20'):
        if f'orig|{t}' in z:
            grid([z[f'orig|{t}'][:n]] + [z[f'dec|{s}|{K}|{t}'][:n] for s in sites], ['original'] + [f'from {s}' for s in sites], f'{t}_K{K}')
    for s in ('pixels', 'b6', 'b9'):
        p = z[f'path|{s}|{K}']                       # (8 pairs, 9 steps, 32, 32, 3), tau = -0.5 .. 1.5
        grid([z['orig|H'][:8]] + [p[:, j] for j in range(9)] + [z['orig|H'][8:16]],
             ['image a'] + [f'{t:+.2f}' for t in np.linspace(-0.5, 1.5, 9)] + ['image b'], f'path_{s}_K{K}')
print('labels', [CLASSES[i] for i in z['label|H'][:n]])
print('pred clean', [CLASSES[i] for i in z['pred|H'][:n]])
if 'pred|pgd8' in z:
    print('pred pgd8 ', [CLASSES[i] for i in z['pred|pgd8'][:n]])
print(sorted(f.name for f in OUT.glob(f'{tag}_*.png')))
