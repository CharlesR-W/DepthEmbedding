# xp: three overnight experiments on the through-depth embedding (2026-11-21)

Exploratory. Spec: `SPEC.md` (with the revisions after review at its end). Results in words: `FINDINGS.md`.
Results page: `pages/notebook.html`, published as `DepthEmbedding/xp-notebook` (rebuild with
`python3 xp/build_notebook.py`, check with `python3 xp/verify_notebook.py <screenshot dir>`).

- **X1** the same probe through ResNet-20s trained on different data (18 networks, 4 probes).
- **X3** adversarial images followed through depth.
- **X4** straight paths between real activations, log gain (FTLE), paths in the embedding.

## Layout

| Path | What |
|---|---|
| `common.py` | Network table, loading, taps. |
| `xlib.py` | Operator fitting on a probe and GPU out-of-sample readouts (ports of `dm.py`'s row rules). |
| `dm.py`, `test_dm.py` | Snapshot of `invert/dm.py` taken 2026-11-21 19:45 (kernels, joint modes, extension). |
| `modal_train.py` | Modal app `crw-depth-xp-train`: datasets and training. |
| `modal_xp.py` | Modal app `crw-depth-xp`: self-test, embedding, comparison, X3, X4. |
| `stage_compare.py`, `stage_x3.py`, `stage_x4.py` | The stages; each docstring defines its output arrays. |
| `analyze_x1.py`, `analyze_x3.py`, `analyze_x4.py` | Local analysis: `out/<stage>/` to `out/x*_summary.json`. |
| `build_notebook.py`, `notebook.py`, `verify_notebook.py` | The results page. |
| `dev_x4.py` | Small dev runner used to test `stage_x4.py`. |
| `out/` | Fetched outputs (about 0.5 GB, not for git). `runs/` holds one record per Modal launch. |

Everything else lives on the Modal volume `crw-depth-xp-results`: `data/` (datasets as arrays), `probes.npz`,
`nets/<name>/final.pt`, `embed/<net>/<probe>/` (Grams, neighbour lists, operators), `compare/`, `x3/`, `x4/`.

## Reproduce

```bash
uv run --with modal python xp/modal_train.py deploy
uv run --with modal python xp/modal_train.py launch data
uv run --with modal python xp/modal_train.py launch train
uv run --with modal python xp/modal_xp.py deploy
uv run --with modal python xp/modal_xp.py launch selftest
uv run --with modal python xp/modal_xp.py launch embed
uv run --with modal python xp/modal_xp.py launch compare '{"units":[{"probe":"c10"},{"probe":"c100"},{"probe":"svhn"},{"probe":"in32"}]}'
uv run --with modal python xp/modal_xp.py launch x3 '{"units":[{"part":"main"}]}'
uv run --with modal python xp/modal_xp.py launch x4 '{"units":[{"part":"paths","l0":"pixels"},{"part":"paths","l0":"stem"},{"part":"paths","l0":"b3"},{"part":"paths","l0":"b6"},{"part":"paths","l0":"b8"},{"part":"paths","l0":"b9"},{"part":"probe_gain"},{"part":"checks"}]}'
uv run --with modal python xp/modal_xp.py fetch compare xp/out/compare      # likewise x3, x4, nets (info.json)
uv run --with numpy python xp/analyze_x1.py compare
uv run --with numpy --with pillow python xp/analyze_x3.py
uv run --with numpy --with scipy --with pillow python xp/analyze_x4.py
python3 xp/build_notebook.py
```

After a `deploy`, wait about half a minute before launching: a call spawned immediately can land on a warm
container that still runs the previous code (this happened once with `compare`).

## Status (2026-11-21)

All stages complete. Launch records with call ids are in `runs/`.

| Stage | Units | Result |
|---|---|---|
| data | 1 | CIFAR-10, CIFAR-100, SVHN, ImageNet 32x32 (from `benjamin-paine/imagenet-1k-32x32`), four probes |
| train | 14 | 12 at 64,000 steps (about 9-10 min each on A10G), 2 ImageNet-32 at 250,000 steps (22 min) |
| embed | 72 | 18 networks x 4 probes, about 100 s each |
| compare | 4 | one per probe, about 2.5 min |
| x3 | 1 | 4 min |
| x4 | 8 | 1-5 min each; checks 8 min |

Test accuracies: CIFAR-10 91.4% (retrained control; zoo seeds 91.6%, 91.5%), CIFAR-100 66.5% and 67.1%,
CIFAR-100 coarse labels 77.8% and 77.4%, SVHN 96.3% and 96.3%, animals 90.6% and 90.8%, vehicles 96.0% and 96.2%,
ImageNet-32 24.3% and 24.2%, shuffled labels 100% train and 9.8% test.

Spend: about $12 estimated from unit times and the 2026-11-21 Modal prices (training about $4, embedding about
$3, X4 including development about $2, the rest about $3); not an invoice.

Reviews: spec by GPT-6.2 Sol (plan review); stages by GPT-6.2 Sol (two code reviews); findings audited against
the data by GPT-6.2 Sol. Their points are folded into `SPEC.md` (revisions) and `FINDINGS.md`.
