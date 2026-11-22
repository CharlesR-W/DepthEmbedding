# Depth embedding

An interactive viewer of how a fixed set of 2,000 CIFAR-10 probe images moves through the layers of a trained
network, for seven networks (ResNet-20/32/56, two seeds each, and one VGG-16-BN).

**Site:** <https://charlesr-w.github.io/DepthEmbedding/>

Each network gets one operator over all (image, site) pairs. Its low modes give every image a row of K numbers at
every site, all in one K-dimensional space, and the viewer plots those rows site by site. The site's Method page
defines the operator and says which steps are post-hoc fits (depth matching and one Procrustes rotation per network,
used only to show networks side by side). Per-site PCA and UMAP embeddings are included as baselines.

This repo was split out of [CoherentModes](https://github.com/CharlesR-W/CoherentModes)
(`explore/cifar_coherent`, the "Depth embedding" tab of its [site](https://charlesr-w.github.io/CoherentModes/)) on
2026-11-21. The per-site walks the operator is built from are defined and measured there. This is exploratory work:
one probe set, a handful of networks, no claims beyond what the viewer shows.

## Layout

| Path | What it is |
|---|---|
| `spacetime.py` | The through-depth operator and its eigen-solve (numpy/scipy). |
| `spacetime_stage.py` | One checkpoint's full computation: couplings, variants, set readouts, acceptance checks. |
| `test_spacetime.py` | Tests of the operator's exact limits and identities. |
| `SPEC_SPACETIME.md` | The specification. It was written inside CoherentModes, so `README.md`, `SPEC.md` and section references in it mean that repo's `explore/cifar_coherent`. |
| `coherent.py`, `seba.py` | Shared modules copied from CoherentModes (walk construction, SEBA). |
| `spacetime_modal.py` | Modal runner: ResNet-20 seed 0 (the reference network). |
| `spacetime_nets_modal.py` | Modal runner: the other five ResNets. |
| `spacetime_vgg_modal.py` | Modal runner: VGG-16-BN. |
| `baseline_embed_modal.py` | Modal runner: per-site PCA and UMAP baselines for all seven networks. |
| `upstream/` | Further modules copied unmodified from CoherentModes that the runners need; see `upstream/README.md`. |
| `site/` | The site's sources and `site/build_site.py`. `site/xsite/` holds the Experiments tab's data layer: input pins, operator-dataset packagers, per-entry generators and checks (`AUTHORING.md`, `SCHEMA.md`). |
| `invert/`, `xp/`, `opt/` | The overnight experiments of 2026-11-21/22 (inverting the embedding; X1 training data, X3 adversarial images, X4 interpolation; solver study): specs, code and notes. Their outputs are not in git. |
| `data/`, `build/` | Not in git: the runners' outputs (about 1.1 GB) and the built site (about 100 MB). |

## Build the site

```bash
uv run --with numpy --with scipy --with pillow python site/build_site.py --strict
python3 -m http.server 8765 --bind 127.0.0.1 -d build/site     # http://127.0.0.1:8765/
```

The Experiments tab is assembled by `site/build_x.py` from `site/content/experiments/` (one fragment per entry),
`site/assets/x/` (chart specs and the build-time facts every number on the page comes from) and the packaged operator
datasets in `data/sitex/`. To regenerate those from the staged raw outputs in `data/fetch/`:

```bash
uv run --with numpy --with scipy --with pillow python site/xsite/ops.py                 # package data/sitex (--entry <name> for one)
for g in x1 x3 x4 solver inv_a inv_b; do uv run --with numpy --with scipy --with pillow python site/xsite/gen_$g.py; done
uv run --with numpy --with scipy --with pytest python -m pytest -q site/xsite/test_opdata.py
uv run --with numpy --with scipy python site/xsite/check_numbers.py    # independent recomputation of headline facts
uv run --with pillow python site/check_browser.py http://127.0.0.1:8765/ data/shots/check
```

This needs `data/`, which holds:

- `data/r20s0b/{meta.json,probe.npz,curves.json}` and `data/r20s0b/spacetime/st_<step>_0.npz`: the reference network;
- `data/stnets/<net>/{info.json,st_<step>_0.npz}`: the other networks;
- `data/stbase/<net>/{pca,umap}_<step>.npz`: the baselines.

To deploy, copy `build/site/` into an empty directory, commit it as a single orphan commit on `gh-pages` and
force-push. That branch holds only the built site.

## Tests

```bash
uv run --with numpy --with scipy --with pytest python -m pytest -q test_spacetime.py
```

## Recomputing the data

The runners are not self-contained. They read inputs that CoherentModes' pipeline wrote to the author's Modal
volumes, so they document how the data were made but will not run from a fresh clone:

- `spacetime_modal.py` reads the reference network's stored per-site walks
  (`cm-cifar-coherent:/work/r20s0b/operators_<step>_<site>.npy`) and probe.
- `spacetime_nets_modal.py` reads each ResNet's stored probe activations and spectra
  (`cm-cifar-coherent:/work/salign/<net>/<step>/`), written by CoherentModes' `explore/spectral_align`.
- `spacetime_vgg_modal.py` and the VGG part of `baseline_embed_modal.py` re-extract activations from trained
  checkpoints (`cm-cifar-coherent:/runs/vgg16-relu-seed0/checkpoints/`) and need the author's ModelZoo training
  code (`~/code/ModelZoo/src`) and its data volume (`crw-modelzoo-train`).

Each runner's docstring gives its commands. The Modal app names, volume names and output paths are unchanged from
CoherentModes.
