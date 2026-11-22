# Re-embedding notes: from embedding coordinates back to activations

Notes from an exploratory session on 2026-11-21, written for an agent picking this up with no other context.
Everything here is exploratory and unreviewed: one network, one checkpoint, one probe set, no second-model check of
the scripts or the numbers. Treat each result as a lead to reproduce, not as established.

## The question

The viewer plots each probe image's coordinates in the low modes of a spectral operator. The owner wants to go the
other way: pick a point in that embedding (ideally anywhere, not only at a probe image) and get the activation it
corresponds to, in order to probe the network with it. Sub-questions that came up:

1. Can activations be reconstructed from the embedding at all, and how well?
2. Does it work off the probe set ("out of distribution")?
3. How many coordinates does a proper reconstruction need, and how many are above sampling noise?
4. Does a different kernel (Gaussian on cosine, instead of the kNN graph) help?
5. Can the map be run in reverse through depth (late site to early site, eventually to the image)?

## Setup shared by every result below

- Network: ResNet-20 seed 0 on CIFAR-10, final checkpoint (step 64,000). Nothing else was tested.
- Probe: 2,000 CIFAR-10 test images (200 per class), the same for every site.
- Sites (12): stem, b1..b9, pool, logits.
- Per-site Gram `C = X X^T`, where `X` is the probe activations at the site, flattened, centred over the probe, then
  each row normalised. So `diag(C) = 1`, and only directions are represented: **the norm is not recoverable from
  these Grams**. `C` fixes `X` up to one orthogonal rotation (classical MDS), which is all the scores below need;
  no script touches raw activations.
- Rule-0 walk at a site: union 15-nearest-neighbour graph on cosine (edge 1, non-edge 1e-6, zero diagonal), scaled
  to symmetric doubly stochastic by Sinkhorn. `reembed_test.walk` builds it in numpy. It omits the lazy
  `(I + P)/2` step of the real pipeline, which does not change eigenvectors.
- Through-depth modes: `data/r20s0b/spacetime/st_64000_0.npz`, array `vecs` of shape
  `(2 scalings, 6 couplings, 16 modes, 12 sites, 2000 images)`. The scripts use scaling 0 (relative site units) and
  coupling index 3 (`t = t*`), the viewer's default. The coordinates of image `i` at site `l` are
  `sqrt(weight[k, l]) * u_kl(i) / |u_kl|`.

## Data and scripts

All in `/home/ubuntu/code/DepthEmbedding`. `data/` is gitignored and exists only on this VPS. None of the scripts
or this file is committed.

| Path | What |
|---|---|
| `data/grams/r20s0b/gram_64000_<site index>.npy` | The 12 Grams (float64, 2000 x 2000). Downloaded from Modal volume `cm-cifar-coherent`, path `work/r20s0b/gram_64000_<i>.npy`. Other checkpoints (`gram_<step>_<i>.npy`) are on the volume too. |
| `data/reembed/*.json` | Full outputs of every script below. |
| `reembed_test.py` | Linear decode by projection onto modes: captured norm and neighbour preservation, K = 4..1000. |
| `reembed_local.py` | Leave-one-out decode by the mean of 15 neighbours' activations. |
| `reembed_gh.py` | Geometric harmonics decode, 10-fold held out. Slow (about 20 min on 4 cores). |
| `reembed_scaling.py` | Nearest-neighbour cosine against probe size; rough dimension per site. |
| `modes_stability_disjoint.py` | Which walk modes reproduce on a disjoint half of the probe. |
| `modes_stability.py` | **Discarded.** Overlapping subsets; shared images make every mode look stable. Do not cite. |
| `modes_needed.py` | Number of modes needed for a given captured share, for kNN, Gaussian and PCA bases. |
| `reembed_cross.py` | Decode site b from neighbours at site a (both depth directions). |

Run any of them with `uv run --with numpy --with scipy python <script>` from the repo root. All run locally; none
needs Modal or a GPU.

## Results

Scores: "captured" is the share of the activations' squared norm kept by a decode (1 = exact); "cosine" is the mean
cosine between an image's true activation and its decode. Per-image cosine is roughly `sqrt(captured)`.

### 1. Linear decode from the modes (`reembed_test.py`, in-sample)

Decode `X_hat = B B^T X` for an orthonormal set `B` of functions on the probe (modes plus the constant). This is a
projection, not a fit. With all 1,999 modes any basis is exact.

| Site | Through-depth, K=16: captured / 15-NN kept | PCA, 17 dims (best possible) | Site's own walk modes, K=1000 |
|---|---|---|---|
| stem | 0.29 / 0.27 | 0.41 / 0.44 | 0.73 / 0.68 |
| b1–b8 | 0.12–0.16 / 0.21–0.28 | 0.18–0.23 / 0.29–0.44 | 0.65–0.71 / 0.67–0.81 |
| b9 | 0.37 / 0.30 | 0.50 / 0.56 | 0.81 / 0.75 |
| pool | 0.76 / 0.35 | 0.94 / 0.88 | 0.97 / 0.81 |
| logits | 0.87 / 0.41 | 1.00 / 1.00 | 0.99 / 0.81 |

A random 1,000-dimensional subspace captures 0.50, so 1,000 kNN walk modes are only modestly better than random in
the conv sites. Through-depth and per-site walk modes are about equal at K=16. Participation ratio of `C`: about
150–240 in b1–b8, 40 at stem and b9, 10 at pool, 5.5 at logits.

### 2. Decode from neighbours (`reembed_local.py`, leave one out)

Decode an image as the mean activation of its 15 nearest other images in some coordinate system. One parameter (15),
no fitted map.

| Site | Neighbours in activation space | 16 through-depth coords | 16 PCA coords | Random |
|---|---|---|---|---|
| stem | 0.57 | 0.49 | 0.54 | 0.01 |
| b1–b8 | 0.37–0.50 | 0.28–0.34 | 0.30–0.39 | 0.00 |
| b9 | 0.70 | 0.61 | 0.67 | 0.01 |
| pool | 0.92 | 0.87 | 0.92 | 0.01 |
| logits | 0.97 | 0.93 | 0.97 | 0.02 |

The first column is a ceiling **for neighbour averaging only** (see result 3). Rescaling each mode's slice to unit
norm changes nothing (within 0.002).

### 3. Geometric harmonics (`reembed_gh.py`, 10-fold held out)

Gaussian kernel on the 16 coordinates, expansion of each activation coordinate in that kernel's eigenfunctions,
Nyström extension to held-out images (Coifman and Lafon 2006). This is a fit with two parameters (bandwidth,
truncation), swept and chosen on the held-out score itself, so slightly optimistic.

| Site | GH, 16 through-depth coords | GH, 16 PCA coords |
|---|---|---|
| stem | 0.53 | 0.62 |
| b1–b8 | 0.32–0.38 | 0.38–0.45 |
| b9 | 0.62 | 0.73 |
| pool | 0.89 | 0.98 |
| logits | 0.95 | 1.00 |

GH beats the neighbour mean by 0.02–0.05 from the same coordinates, and from PCA coordinates it beats the
activation-space neighbour mean at several sites, so that column in result 2 is not a general ceiling. The best
bandwidth was the widest tried (16 x the median 15th-neighbour distance) at every site: the sweep hit its edge, and
a wide Gaussian is close to a low-order polynomial fit.

### 4. Density of the probe (`reembed_scaling.py`)

Mean cosine to the nearest other image, on random subsets of 250..2,000 images. `1 - cos` should fall like
`n^(-2/d)`; the slope gives a rough dimension `d` and an extrapolated `n` for a target cosine.

| Site | Nearest-neighbour cosine at 2,000 | Rough d | n for cosine 0.9 |
|---|---|---|---|
| stem | 0.46 | 42 | ~1e18 |
| b1–b8 | 0.23–0.31 | 40–77 | 1e20 or more |
| b9 | 0.59 | 26 | ~1e11 |
| pool | 0.89 | 12 | ~3,000 |
| logits | 0.96 | 8 | ~60 |

The last column is a four-point extrapolation: read it as "more probe images will not fix the conv sites", not as an
estimate. All of CIFAR would move the conv-site cosines by a few hundredths.

### 5. How many modes are above sampling noise (`modes_stability_disjoint.py`)

Split the probe into disjoint halves, build the kNN walk on each, extend each mode of one half to the other by one
walk step (Nyström up to scale), and measure how much of it lies in the other half's first `2k + 10` modes, beyond
chance (1 = reproduced, 0 = random).

| Site | k=1 | k=4 | k=8 | k=16 | k=32 | k=64 | k=256 | Last k with excess >= 0.5 |
|---|---|---|---|---|---|---|---|---|
| stem | 0.96 | 0.87 | 0.73 | 0.48 | 0.38 | 0.37 | 0.31 | 8 |
| b1–b8 | 0.94–0.95 | 0.79–0.88 | 0.51–0.76 | 0.26–0.61 | 0.24–0.47 | 0.21–0.34 | 0.20–0.34 | 8–16 |
| b9 | 0.99 | 0.98 | 0.96 | 0.80 | 0.66 | 0.53 | 0.43 | 64 |
| pool | 1.00 | 0.99 | 0.97 | 0.78 | 0.75 | 0.62 | 0.49 | 128 |
| logits | 1.00 | 0.99 | 0.98 | 0.80 | 0.78 | 0.66 | 0.50 | 256 |

The 0.5 threshold is arbitrary; there is no sharp noise floor (excess levels off at 0.2–0.35). Halves have 1,000
images, so the full probe resolves somewhat more. This is for per-site kNN walk modes only: **not tested for the
through-depth modes or for Gaussian-kernel modes**.

### 6. Modes needed, and the Gaussian kernel (`modes_needed.py`, in-sample)

Smallest K whose first K modes capture 0.5 / 0.8 / 0.9. Gaussian walk: `exp(-(1 - cos) / s)` with `s` the median
15th-neighbour distance, zero diagonal, Sinkhorn-scaled.

| Site | kNN-15 walk | Gaussian on cosine | PCA of activations (fewest possible) |
|---|---|---|---|
| stem | 324 / 1,284 / 1,653 | 45 / 291 / 584 | 38 / 251 / 479 |
| b1–b8 | 450–650 / 1,290–1,430 / 1,640–1,710 | 110–220 / 470–680 / 820–1,010 | 107–217 / 459–667 / 786–996 |
| b9 | 72 / 918 / 1,490 | 20 / 219 / 590 | 17 / 158 / 411 |
| pool | 7 / 34 / 157 | 6 / 16 / 52 | 4 / 8 / 11 |
| logits | 4 / 9 / 34 | 4 / 9 / 27 | 3 / 6 / 7 |

Bandwidths 4 x and 16 x wider move the Gaussian column only slightly further towards PCA. Reading: in the
high-dimensional sites pairwise cosine distances are all similar, so a Gaussian of any reasonable bandwidth is close
to linear in the cosine and its modes are close to principal components. It reconstructs well and mostly stops being
a neighbourhood method. These counts are in-sample and are a large fraction of the probe size (2,000), so they are
set by the probe, not by the network.

### 7. Reverse through depth (`reembed_cross.py`)

Decode an image's activation at one site from its 15 nearest images at another (neighbours in activation space, not
in the through-depth coordinates). Rows: site whose neighbourhood is used; columns: site decoded.

| Neighbours at | stem | b4 | b8 | b9 | pool | logits |
|---|---|---|---|---|---|---|
| stem | 0.56 | 0.29 | 0.19 | 0.28 | 0.42 | 0.48 |
| b4 | 0.49 | 0.46 | 0.39 | 0.50 | 0.67 | 0.72 |
| b8 | 0.33 | 0.39 | 0.49 | 0.65 | 0.82 | 0.87 |
| b9 | 0.25 | 0.28 | 0.37 | 0.70 | 0.88 | 0.93 |
| pool | 0.24 | 0.23 | 0.26 | 0.62 | 0.92 | 0.96 |
| logits | 0.21 | 0.20 | 0.23 | 0.60 | 0.91 | 0.97 |

The full 12 x 12 matrix is in `data/reembed/r20s0_64000_cross.json`. Reverse is as legitimate as forward (the
coupling and the walks are symmetric; on probe images it is a lookup), but from a late site it recovers little of
the early ones. There is a break at b9: sites from b9 on decode each other well and everything earlier poorly.

## What I concluded, and how firmly

- **Two separate limits.** (a) Sampling density: 2,000 images are far apart in the conv sites, so any decoder learned
  from them reproduces only structure coarser than the gaps. (b) Information in K numbers: 16 coordinates cannot
  specify a point in a cloud of dimension 40–77; the best possible output is the average over everything sharing
  those coordinates. More coordinates relax (b), more probe images relax (a). Neither binds at pool and logits.
  The dimension estimates behind this are rough.
- **16 coordinates are roughly what the conv-site kNN modes support** (result 5), yet far too few to reconstruct
  (results 1 and 6). Both are true at once: the cloud is high-dimensional and the probe resolves few directions.
- **kNN modes are a poor reconstruction basis; Gaussian-on-cosine modes are close to optimal** (result 6). Firm for
  in-sample capture. Whether the Gaussian modes generalise is untested.
- **PCA coordinates decode better than through-depth coordinates** at every site (results 2 and 3). Expected for a
  linear target.
- **A correction made during the session:** I first called the truncated-mode decode "a fit that must be labelled";
  it is an exact projection, with coefficients given by inner products. I also first called the activation-space
  neighbour mean a ceiling; geometric harmonics exceeds it.

## Ideas not yet tried

1. **Stability of Gaussian-kernel modes.** Rerun `modes_stability_disjoint.py` with the Gaussian walk of
   `modes_needed.py`. This decides whether "a few hundred Gaussian modes" is real or probe-specific. Local, minutes.
2. **A Gaussian-on-cosine through-depth operator with hundreds of modes.** The full operator is 24,000 x 24,000;
   needs a Modal machine for a dense solve. `spacetime.py` currently solves for the lowest 24 only.
3. **Smooth kernel plus Nyström forward map.** The extension of a mode to any activation vector is exact for a smooth
   kernel (for the through-depth modes it is a small tridiagonal solve over the sites). The kNN indicator kernel is
   piecewise constant, so it has no useful gradients. This gives a canonical activation-to-embedding map everywhere,
   including off the probe.
4. **Round-trip test for any decoder.** Decode a point, push the result back through the Nyström forward map, and
   check it lands where asked. Label-free, and it works for off-probe points, which is the case the owner cares
   about.
5. **Decode directions instead of points.** Each mode has an exact activation-space pattern (`X^T psi_k`) and
   pixel-space pattern (mode-weighted sum of images). Moving along an embedding axis corresponds to adding that
   direction; patch it into the network and watch downstream. A local version uses a tangent map fitted on
   neighbours (a fit).
6. **Diffusion geometry on the samples.** The carré du champ of the walk,
   `Gamma(f, g)(i) = 1/2 sum_j M(i, j) (f(j) - f(i)) (g(j) - g(i))`, gives the pulled-back metric of the embedding,
   and with it gradients and divergence, with no out-of-sample extension. The divergence of the depth displacement
   field `G_{l+1} - G_l` (where the cloud contracts or expands between sites) can be computed from the shipped data.
7. **Norms.** Decode the unnormalised dot-product Gram with the cosine-graph modes to get norms back. Needs the
   unnormalised Grams (tag `r20s0d` on the volume).
8. **The same tests through the 16 through-depth coordinates** for result 7, and for other checkpoints and networks.
9. **Pixel space as a site**, so that reverse-through-depth ends at the image.
10. **A denser probe.** Only worthwhile for b9, pool and logits, per result 4.

## Things an agent should know before continuing

- **Kernels already examined in CoherentModes** (`explore/cifar_coherent/README.md`, "dot" section): the L2 distance
  on unnormalised activations, `|x_i - x_j|^2 / 2`, has severe hubs in stem–b8 (max in-degree 860–1,700 of 2,000)
  and was set aside. Ranking by raw dot product behaved like cosine in a diagnostic (15-NN overlap 0.71, mild hubs)
  but was never run through the full pipeline. Cosine with Sinkhorn scaling has mild hubs (max in-degree about 140).
- **Owner's fixed commitments** (from CoherentModes' `CLAUDE.md`): every object in the construction ends in an SVD
  of a sample-estimable operator; nothing is trained inside the construction; closed-form fits are allowed and must
  be called fits; the only geometric assumption is the dot product (cosine on normalised activations); no labels or
  probes define a kernel or basis; no sparsity assumption; architecture-agnostic. Optimising an input through the
  network, or patching activations, is an intervention and is labelled as one.
- **Terminology.** Describe what is computed (sites, walks, modes, coordinates). Do not describe what the network
  "does", and do not import the vocabulary of CoherentModes' `THEOREMS.md` / `SPARSITY.md`.
- **Compute.** This VPS has 4 cores and 8 GB RAM and is for orchestration. Anything over about 10 minutes, or a
  dense 24,000 x 24,000 solve, goes to Modal (`modal` skill). The Modal volume `cm-cifar-coherent` holds Grams,
  walks (`work/r20s0b/operators_<step>_<site>.npy`, 5 rules each) and, for the other networks,
  `work/salign/<net>/<step>/`.
- **Review rule.** These results have had no cross-model review. Before any of them goes on the site or into
  CoherentModes' `THEORY.md` / `EXPERIMENTS.md`, that repo's rule applies: a critique by an agent of the opposite
  model family.
- **Repo state.** `main` on GitHub has the pre-redesign site. Uncommitted in the working tree: the interface
  redesign (`site/embedding_tab.html`, `site/js/embedding.js`, `site/css/site.css`; GPT-approved, awaiting the
  owner's go-ahead to commit and deploy), the scripts above, and this file.
