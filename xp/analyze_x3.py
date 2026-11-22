"""Local analysis of X3 (adversarial images through depth): xp/out/x3/* -> xp/out/x3_summary.json

    uv run --with numpy --with pillow python xp/analyze_x3.py

Cohorts are fixed as in SPEC.md (revisions): success = prediction differs from the label (untargeted) or equals the
requested target (targeted); destination = the network's final prediction.  Site-local quantities date an effect;
joint coordinates are reported as retrospective.
"""
import base64
import io
import json
from pathlib import Path

import numpy as np

D = Path(__file__).resolve().parent / 'out/x3'
SITES = ['stem'] + [f'b{i}' for i in range(1, 10)] + ['pool', 'logits']
A, B = 'c10-s0', 'c10-s1'
K = 'ea100'
r = lambda a, n=4: np.round(np.asarray(a, dtype=float), n).tolist()



def switch_site(mass, src, dst, top=False):
    """First site from which dst outweighs src (or is the top class) at every later site; 12 = no persistent switch."""
    m = mass.transpose(1, 0, 2)                                   # (images, sites, classes)
    i = np.arange(len(m))
    ok = (m[i, :, dst] >= m.max(2) - 1e-12) if top else (m[i, :, dst] > m[i, :, src])
    tail = np.flip(np.cumprod(np.flip(ok, 1), 1), 1).astype(bool)  # ok at this and every later site
    return np.where(tail.any(1), tail.argmax(1), 12)


def pct(x, ref):
    """Percentile of each x within ref."""
    return np.searchsorted(np.sort(ref), x) / len(ref)


def top2_share(energy, axis):
    """Largest two images' squared-norm share, along the specified image axis."""
    pick = np.take(np.partition(energy, -2, axis=axis), [-2, -1], axis=axis)
    return pick.sum(axis) / energy.sum(axis)


def kernel_comparison(D, baseline, y, si, sj, kernels=('ea100', 'pr100', 'knn15'), rerun=False):
    """Same c10-s0 images throughout. Recompute the displacement unit after dropping modes."""
    ans = {}
    for kernel in kernels:
        root = D if kernel == 'pr100' or rerun else baseline
        suffix = '_pr100' if kernel == 'pr100' or rerun else ''
        with np.load(root / f'op_{A}_{kernel}.npz') as z:
            site = top2_share(z['site_vec'].astype(float)[:, :, :16] ** 2, 1)
            joint = top2_share((z['vecs'][:, :, :16].astype(float) ** 2).sum(0), 0)
        with np.load(root / f'{A}_clean{suffix}.npz') as z:
            clean_y = z[f'{kernel}_site_y']
        rec = dict(top2_joint=joint.tolist(), top2_site_local=site.tolist())
        masks = {'all': np.ones_like(site, dtype=bool)}
        if kernel == 'ea100':
            masks['drop_top2_gt_0p1'] = site <= 0.1
        rec['displacement'] = {}
        for name, keep in masks.items():
            scale = np.median(np.linalg.norm((clean_y[:, si] - clean_y[:, sj]) * keep[:, None, :], axis=2), 1)
            disp = dict(kept_modes=keep.sum(1).tolist(), unit=r(scale, 8))
            for setid in ('u8', 'n8', 't8'):
                with np.load(root / f'{A}_{setid}{suffix}.npz') as z:
                    diff = (z[f'{kernel}_site_y'] - clean_y) * keep[:, None, :]
                disp[setid] = r(np.median(np.linalg.norm(diff, axis=2), 1) / scale)
            rec['displacement'][name] = disp
        rec['switch_and_mass'] = {}
        with np.load(baseline / 'sets.npz') as z:
            target, ny = z['target'], z['natA_y']
        for setid in ('u8', 't8', 'u2', 'natA'):
            with np.load(root / f'{A}_{setid}{suffix}.npz') as z:
                mass, pred = z[f'{kernel}_mass'], z['logits'].argmax(1)
            src = ny if setid == 'natA' else y
            ok = (pred == target) if setid == 't8' else (pred != src)
            m, s, dst = mass[:, ok], src[ok], pred[ok]
            i = np.arange(len(dst))
            hist = {}
            for rule, top in (('dst_gt_src', False), ('dst_top', True)):
                sw = switch_site(m, s, dst, top=top)
                bins = np.bincount(sw, minlength=13)
                hist[rule] = dict(hist=bins.tolist(), median=float(np.median(sw)),
                                  modes=np.flatnonzero(bins == bins.max()).tolist())
            ms, md = m[:, i, s], m[:, i, dst]
            rec['switch_and_mass'][setid] = dict(n=int(ok.sum()), **hist,
                pool=dict(src=float(ms[10].mean()), dst=float(md[10].mean()), other=float((1 - ms[10] - md[10]).mean())),
                logits=dict(src=float(ms[11].mean()), dst=float(md[11].mean()), other=float((1 - ms[11] - md[11]).mean())))
        ans[kernel] = rec
    return ans


def main(K=K, D=D, output=None, baseline=D):
    stretch = K == 'pr100'
    nets = (A,) if stretch else (A, B)
    attack_names = ('u2', 'u8', 'n8', 't8') if stretch else [f'{a}{e}' for a in 'un' for e in (1, 2, 4, 8)] + ['t8']

    def load(net, name):
        suffix = '_pr100' if stretch else ''
        with np.load(D / f'{net}_{name}{suffix}.npz') as z:
            return dict(z)

    sets = np.load(D / 'sets.npz')
    y, target = sets['labels'], sets['target']
    info = json.loads((D / 'DONE').read_text())
    out = dict(sites=SITES, info=info, n=len(y))
    ops = {n: np.load(D / f'op_{n}_{K}.npz') for n in nets}
    clean = {n: load(n, 'clean') for n in nets}
    rng = np.random.default_rng(0)
    same = [(i, j) for i, j in rng.integers(0, len(y), (40000, 2)) if y[i] == y[j] and i != j]
    si, sj = np.array(same).T

    def unit(net, key):                                           # median same-class clean distance per site
        z = clean[net][f'{K}_{key}']
        return np.median(np.linalg.norm(z[:, si] - z[:, sj], axis=2), 1)

    units = {n: dict(y=unit(n, 'y'), site_y=unit(n, 'site_y')) for n in nets}
    out['unit_same_class'] = {n: {k: r(v, 5) for k, v in u.items()} for n, u in units.items()}

    # ---- per attack: accuracy, activation measures, displacement, support
    att = {}
    for name in attack_names:
        rec = {}
        for net in nets:
            d = load(net, name)
            pred = d['logits'].argmax(1)
            flip = pred != y
            e = dict(acc=float((~flip).mean()), actcos=r(np.median(d['actcos'], 1)), relpert=r(np.median(d['relpert'], 1)))
            for key in ('y', 'site_y'):
                disp = np.linalg.norm(d[f'{K}_{key}'] - clean[net][f'{K}_{key}'], axis=2) / units[net][key][:, None]
                e[f'disp_{key}'] = r(np.median(disp, 1))
                e[f'disp_{key}_q'] = [r(np.quantile(disp, 0.25, 1)), r(np.quantile(disp, 0.75, 1))]
                if net == B and name[0] in 'ut':
                    e[f'disp_{key}_flip'] = r(np.median(disp[:, flip], 1)) if flip.sum() > 20 else None
                    e[f'disp_{key}_noflip'] = r(np.median(disp[:, ~flip], 1)) if (~flip).sum() > 20 else None
            e['support'] = r(np.median(np.stack([pct(d[f'{K}_cmax'][l], clean[net][f'{K}_cmax'][l]) for l in range(12)]), 1))
            rec[net] = e
        if not stretch and name[0] in 'ut':
            pa, pb = load(A, name)['logits'].argmax(1), load(B, name)['logits'].argmax(1)
            okA = (pa == target) if name == 't8' else (pa != y)
            rec['transfer'] = dict(success_A=float(okA.mean()), any_flip=float((pb != y)[okA].mean()),
                                   same_destination=float((pb == pa)[okA].mean()))
        att[name] = rec
    out['attacks'] = att

    # ---- class mass and switch sites
    def mass_block(net, name, src, dst, sel, kern=K):
        m = load(net, name)[f'{kern}_mass'][:, sel]
        i = np.arange(sel.sum())
        s, d_ = src[sel], dst[sel]
        ms, md = m[:, i, s], m[:, i, d_]
        sw, swt = switch_site(m, s, d_), switch_site(m, s, d_, top=True)
        return dict(n=int(sel.sum()), src=r(ms.mean(1)), dst=r(md.mean(1)), other=r((1 - ms - md).mean(1)),
                    switch=np.bincount(sw, minlength=13).tolist(), switch_top=np.bincount(swt, minlength=13).tolist())

    cm = {}
    for net in nets:
        c = {}
        m = clean[net][f'{K}_mass']
        c['clean'] = dict(n=len(y), src=r(m[:, np.arange(len(y)), y].mean(1)))
        for name in (nm for nm in attack_names if nm[0] in 'ut'):
            pred = load(net, name)['logits'].argmax(1)
            ok = (pred == target) if (name == 't8' and net == A) else (pred != y)
            c[name + '_success'] = mass_block(net, name, y, pred, ok)
            if (~ok).sum() > 20:
                mf = load(net, name)[f'{K}_mass'][:, ~ok]
                c[name + '_failure'] = dict(n=int((~ok).sum()), src=r(mf[:, np.arange((~ok).sum()), y[~ok]].mean(1)))
        nat = 'natA' if net == A else 'natB'
        ny = sets[nat + '_y']
        npred = load(net, nat)['logits'].argmax(1)
        c['natural'] = mass_block(net, nat, ny, npred, npred != ny)
        cm[net] = c
    pred8 = load(A, 'u8')['logits'].argmax(1)
    if not stretch:
        cm['knn15_u8_success_A'] = mass_block(A, 'u8', y, pred8, pred8 != y, kern='knn15')
    # natural errors against eps-8 untargeted successes matched on (source, destination), class mass at the logits (seed 0)
    nd = load(A, 'natA'); ny, npr = sets['natA_y'], nd['logits'].argmax(1); okn = npr != ny
    mN, mU = nd[f'{K}_mass'][11], load(A, 'u8')[f'{K}_mass'][11]
    rows = []
    for s_, d_ in sorted(set(zip(ny[okn].tolist(), npr[okn].tolist()))):
        iN, iU = okn & (ny == s_) & (npr == d_), (pred8 != y) & (y == s_) & (pred8 == d_)
        if iN.sum() >= 3 and iU.sum() >= 3:
            rows.append((iN.sum(), mN[iN, d_].mean(), mU[iU, d_].mean(), mN[iN, s_].mean(), mU[iU, s_].mean(), iU.sum()))
    if rows:
        w = np.array([q[0] for q in rows], float)
        av = lambda j: float(np.sum(w * np.array([q[j] for q in rows])) / w.sum())  # noqa: E731
        cm['matched_natural_u8_A'] = dict(n_combos=len(rows), n_natural=int(w.sum()), n_u8=int(sum(q[5] for q in rows)),
                                         natural_dst=round(av(1), 4), u8_dst=round(av(2), 4), natural_src=round(av(3), 4), u8_src=round(av(4), 4))
    out['class_mass'] = cm

    # ---- arrival: distance to the destination class's probe centroid, as a percentile of clean members' distances
    arr = {}
    for net in nets:
        op = ops[net]
        pl = op['probe_labels']
        cents = {'y': np.stack([op['vecs'][:, pl == c, :16].mean(1) for c in range(10)], 1),          # (12, 10, 16)
                 'site_y': np.stack([op['site_vec'][:, pl == c].mean(1) for c in range(10)], 1)}
        a = {}
        for key in ('y', 'site_y'):
            zc = clean[net][f'{K}_{key}']
            ref = np.linalg.norm(zc - cents[key][:, y], axis=2)                                        # clean to own class
            for name in ('u8', 't8', 'u2'):
                d = load(net, name)
                pred = d['logits'].argmax(1)
                ok = (pred == target) if (name == 't8' and net == A) else (pred != y)
                dist = np.linalg.norm(d[f'{K}_{key}'] - cents[key][:, pred], axis=2)
                p = np.full((12, len(y)), np.nan)
                for l in range(12):
                    for c in range(10):
                        m = ok & (pred == c)
                        if m.any() and (y == c).sum() > 10:
                            p[l, m] = pct(dist[l, m], ref[l, y == c])
                a[f'{name}_{key}'] = dict(median=r(np.nanmedian(p, 1)), hist_logits=np.histogram(p[11][ok], 10, (0, 1))[0].tolist(),
                                          hist_pool=np.histogram(p[10][ok], 10, (0, 1))[0].tolist())
        arr[net] = a
    out['arrival'] = arr

    if not stretch:
        # ---- perturbation path clean -> u8
        path = {}
        for net in nets:
            names = ['clean'] + [f'p{k}' for k in range(1, 8)] + ['u8']
            ds = [load(net, nm) for nm in names]
            path[net] = dict(
                acc=[float((d['logits'].argmax(1) == y).mean()) for d in ds],
                src_mass=[r(d[f'{K}_mass'][:, np.arange(len(y)), y].mean(1)) for d in ds],
                disp_site=[r(np.median(np.linalg.norm(d[f'{K}_site_y'] - clean[net][f'{K}_site_y'], axis=2), 1) / units[net]['site_y']) for d in ds],
                relpert=[r(np.median(d['relpert'], 1)) if 'relpert' in d else [0] * 12 for d in ds])
        out['path'] = path

    # ---- logits-only check
    ck = np.load(D / 'check_logits_only.npz')
    out['logits_only'] = dict(ratio=r(np.median(ck['dy'], 1) / np.median(ck['dy_full'], 1)), local_max_change=float(ck['local_max_change']),
                              dy=r(np.median(ck['dy'], 1), 6), dy_full=r(np.median(ck['dy_full'], 1), 6))
    out['extension'] = {n: dict(cond=r(ops[n]['cond'][:16], 2), smin=r(ops[n]['smin'][:16], 4)) for n in nets}

    # ---- pictures: probe cloud and a few examples in joint and site-local coordinates
    from PIL import Image

    def png(a):
        b = io.BytesIO(); Image.fromarray(a.transpose(1, 2, 0)).resize((64, 64), Image.NEAREST).save(b, 'PNG')
        return base64.b64encode(b.getvalue()).decode()

    op = ops[A]
    ex = np.concatenate([np.where(y == c)[0][:2] for c in range(10)])
    u8d, t8d = load(A, 'u8'), load(A, 't8')
    out['picture'] = dict(
        probe_labels=op['probe_labels'].tolist(),
        probe_y=np.round(op['vecs'][:, :, :4], 4).tolist(), probe_site=np.round(op['site_vec'][:, :, :3].astype(float), 4).tolist(),
        ex=dict(idx=ex.tolist(), label=y[ex].tolist(), pred_u8=u8d['logits'].argmax(1)[ex].tolist(), target=target[ex].tolist(),
                clean_y=np.round(clean[A][f'{K}_y'][:, ex, :4], 4).tolist(), u8_y=np.round(u8d[f'{K}_y'][:, ex, :4], 4).tolist(),
                t8_y=np.round(t8d[f'{K}_y'][:, ex, :4], 4).tolist(),
                clean_site=np.round(clean[A][f'{K}_site_y'][:, ex, :3], 4).tolist(), u8_site=np.round(u8d[f'{K}_site_y'][:, ex, :3], 4).tolist(),
                t8_site=np.round(t8d[f'{K}_site_y'][:, ex, :3], 4).tolist(),
                img_clean=[png(sets['img_clean'][i]) for i in ex], img_u8=[png(sets['img_u8'][i]) for i in ex],
                img_t8=[png(sets['img_t8'][i]) for i in ex]))
    if stretch:
        out['kernel'] = K
        rerun = info.get('comparison_fallback', False)
        out['analysis_scope'] = 'matched regenerated cohort' if rerun else 'original cohort'
        out['kernel_comparison'] = kernel_comparison(D, baseline, y, si, sj, rerun=rerun)
        if rerun:
            out['archived_kernel_comparison'] = kernel_comparison(baseline, baseline, y, si, sj, kernels=('ea100', 'knn15'))
    output = output or D.parent / ('x3_summary_pr100.json' if stretch else 'x3_summary.json')
    output.write_text(json.dumps(out))
    if stretch:
        return
    # console digest
    print('acc A/B', {k: (round(v[A]['acc'], 3), round(v[B]['acc'], 3)) for k, v in att.items()})
    for k in ('u8', 'u2', 'n8', 't8'):
        print(k, 'actcos', att[k][A]['actcos']); print(k, 'disp_site', att[k][A]['disp_site_y']); print(k, 'disp_joint', att[k][A]['disp_y'])
    for k in ('u8_success', 't8_success', 'u2_success', 'natural'):
        c = cm[A][k]; print(k, c['n'], 'src', c['src'], '\n   dst', c['dst'], '\n   switch', c['switch'], 'top', c['switch_top'])
    print('clean src', cm[A]['clean']['src'])
    print('knn u8 switch', cm['knn15_u8_success_A']['switch'])
    print('arrival A', {k: v['median'][9:] for k, v in arr[A].items()})
    print('support u8', att['u8'][A]['support'], 'n8', att['n8'][A]['support'])
    print('transfer', {k: v.get('transfer') for k, v in att.items() if 'transfer' in v})
    print('B u4 flip/noflip site disp', att['u4'][B]['disp_site_y_flip'], att['u4'][B]['disp_site_y_noflip'])
    print('logits-only ratio', out['logits_only']['ratio'])
    print('cond', out['extension'][A]['cond'])


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', choices=('ea100', 'pr100'), default='ea100')
    parser.add_argument('--input', type=Path)
    parser.add_argument('--baseline', type=Path, default=D)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    stretch_input = D.parents[2] / 'data' / 'stretch' / 'x3'
    main(args.kernel, args.input or (stretch_input if args.kernel == 'pr100' else D), args.output, args.baseline)
