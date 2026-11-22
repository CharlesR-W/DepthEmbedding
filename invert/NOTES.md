# Inverting the through-depth embedding: what was built and what it measured

Overnight session, 2026-11-21 to 22. Exploratory: one network (ResNet-20 seed 0, CIFAR-10, step 64,000, test
accuracy 91.56%), one checkpoint, CIFAR-10 test images as probe and held-out set. Spec and its two amendments:
`invert/SPEC.md`. Literature survey: `invert/LITERATURE.md`. Results page: `invert/pages/invert.html`, built by
`invert/build_page.py` from `data/invert/results.json`. A GPT reviewer recomputed every number below from that
file; its corrections are folded in.

Unless stated, numbers are for the 8,000-image probe with the participation-ratio kernel, k = 400
(`P8__pr400__L12`). Decode numbers use all 2,000 held-out images `H`; round-trip and edit numbers use the first
256 of them; adversarial numbers the first 1,000. "Cosine" is `cos_raw`: cosine between the true centred
activation and the decode with its predicted scale. The probe and `H` share no image.

## The method in five lines

1. **Kernel.** Per site, each probe image gets its own Gaussian width on cosine distance so that the
   participation ratio of its conditional over the other images is k; average the conditional with its transpose;
   symmetric Sinkhorn. Smooth, with an out-of-sample row that replays a probe image's own row exactly when the
   image is masked out of its own columns (`dm.ea_kernel(renyi=2)`, `dm.ea_oos_rows`).
2. **Forward map.** A new activation x at site l has **source coordinates** `s_l(x) = p_l(x) . f_l` (its kernel
   row times the mode slices). A whole image's joint coordinates are a fixed linear filter along depth of its
   source trajectory: per mode, `y = T^-1 (s / 2c)`, `T = diag(1/2c_l) + t D_path - mu I` (`dm.joint_forward`).
3. **Decode (an estimate of the conditional mean).** Linear part plus Gaussian kernel ridge from K coordinates to
   the activation, as weights on probe activations; bandwidth and ridge tuned on an inner split of the probe
   (`stage_decode.py`).
4. **Source matching.** An activation near a starting point x0 whose source coordinates equal a requested
   value, by damped Gauss-Newton with a step cap and a trust radius (`stage_rt.py: match`). A requested joint
   coordinate is turned into a source target with the decoder's own weights.
5. **Edits with a reference image.** The residual-form guess `x_a + D(psi_b) - D(psi_a)`, optionally followed by
   source matching.

## Findings

**Kernel.**
- Hubs: the directed 15-NN graph on raw cosine has maximum in-degree 49-253 by site (N = 2,000). The per-point
  kernels' top-15 occurrence is at most 42 (global-width kernel: 53).
- The Shannon-perplexity version of the rule (the repo's existing rules 1-2) produces modes carried by two
  images: at N = 8,000 (perplexity 400), 46 of the first 64 joint modes have more than 10% of their squared norm
  on two images and 9 have more than half; at N = 2,000, 11-14 of 64. The participation-ratio version and the
  k-NN rule have none above 10%, at either size. The pairs were not verified to be duplicate images (one checked
  pair has activation cosine 0.65 at stem, 0.98 at logits). Likely mechanism: a row of Shannon perplexity 400 over
  8,000 images can put up to 0.41 of its weight on one close neighbour (0.33 for 300 over 2,000), whereas with
  participation ratio k no conditional weight exceeds 1/sqrt(k) (a bound on the conditional, before
  symmetrisation and Sinkhorn).
- A single global width with Sinkhorn: at the narrower width tried (perplexity 150) the second eigenvalue is 0.997
  at b1 and b2 and the leading mode does not reproduce on a disjoint half (0.24, 0.18); at perplexity 500 it is
  0.95 and 0.81.
- Reproducibility of mode 16 on a disjoint half (one 1,000/1,000 split, N = 2,000; chance-corrected share inside
  the other half's first 42 modes), by site: participation ratio 300: 0.71-0.98; Shannon 300: 0.63-0.96;
  k-NN 15: 0.24-0.88. The kernels differ in effective width (final walk perplexity 478-953, 314-426 and 17-21),
  so this is not a like-for-like comparison of the rules.

**Forward map.**
- Exactness tests on synthetic sites (`test_dm.py`, 10 pass): masked probe rows reproduce P; the tridiagonal
  system fed them reproduces the modes; an independent check of the equation. On real data the torch and numpy
  paths agree to 4e-7.
- Refit stability (N = 2,000 probe; held-out images' frozen-probe coordinates against an operator refitted with
  them; share of squared coordinate norm reproduced, stem to b9), first 16 / first 64 modes: participation ratio
  100: 0.91-0.98 / 0.75-0.93; participation ratio 300: 0.94-0.98 / 0.78-0.86; Shannon 300: 0.96-0.99 /
  0.65-0.87; k-NN 15: 0.90-0.98 / 0.75-0.98. The refit also moves the probe images' own coordinates (0.87 on
  average for participation ratio 300 at 64 modes).

**The joint coordinates at a site take the other sites' sources as inputs.** The site's own source has 13-27% of
the absolute filter weight in conv sites (K = 16; 44% at pool, 62% at logits). In the pilot, swapping in the true
target activation at one site with the other sites' sources held moved the joint coordinate 17-33% of the way.
So the single-site inverse is posed in source coordinates. Two consequences that were measured: (i) image decodes
from joint coordinates change little with the site (pixel error to the original 0.148 to 0.159 from the pixel
site to pool, 32 images, K = 64), while decodes from source coordinates lose instance detail with depth (0.147 to
0.241); (ii) see the adversarial paragraph.

**Decoding from K joint coordinates (held-out cosine; kernel ridge).**

| K | stem | b1-b8 | b9 | pool | logits |
|---|---|---|---|---|---|
| 16 | 0.56 | 0.36-0.42 | 0.65 | 0.93 | 0.97 |
| 64 | 0.63 | 0.42-0.51 | 0.71 | 0.94 | 0.98 |
| 256 | 0.68 | 0.46-0.59 | 0.73 | 0.93 | 0.97 |

- Among the closed-form decoders kernel ridge has the highest mean cosine at every site and K. At K = 64 geometric
  harmonics on the coordinates trails it by 0.005-0.036 and the parameter-free linear projection by 0.014-0.054;
  the Laplacian pyramid and the neighbour mean are lower; the order of the others varies with site and K.
- A trained MLP on the same coordinates is within 0.01 of kernel ridge in cosine at K = 16 (ahead at 8 of 12
  sites) and 0.00-0.03 behind at K = 64, but its patched top-1 agreement at b4, K = 64 is higher (0.49 against
  0.39). So kernel ridge is at the level of the trained benchmark for reconstruction, not above it for function.
- Reference that sees the activation itself: projecting the unit activation on K + 1 principal directions of the
  probe gives 0.50-0.61 at K = 64 in b1-b8.
- Probe size: N = 2,000 (k = 300) gives 0.50 at b4, K = 64, against 0.51 at N = 8,000 (k = 400); the gap is 0.03
  at K = 256. Width and size change together here, so this is a local comparison, not an isolated effect of N.
- Captured share of squared norm, median over images, K = 64: 0.16-0.25 in b1-b8, 0.51 at b9, 0.88 at pool. Its
  mean is pulled down by a few images (b4: mean 0.15, median 0.25), enough to put kernel ridge below the linear
  decoder on the mean at b4. Of the 2,000 held-out images, 25 have a kernel row with more than half its weight on
  one probe image at some site, and 17 of the 20 largest b4 coordinate norms are among them: unmasked near-copies
  of probe images get overshooting coordinates.
- Patched alone (K = 64), the kernel-ridge decode gives the clean run's top-1 class for 12-39% of images at
  stem-b4, 74-77% at b5-b6, 90-96% from b7 to pool. Replacing its norm by the true norm (which also changes the
  decoded direction) gives 66% at b3 and 72% at b4.

**Source matching (K = 16).**
- Mean miss of the matched activation's sources from the target, in per-mode held-out spread units squared: at
  most 0.002 at every site, for the image's true sources and for the target inferred from its joint coordinates.
  The match moves the decode by 0.19-0.51 of the image's centred norm.
- The inferred source target itself is 0.02-0.07 from the true sources through b8 and under 0.01 from b9.
- What matching does not do: reproduce the requested **joint** coordinate in a patched run. With sites before l
  from the image, site l matched, and later sites recomputed by the network, the joint coordinate at l is off by
  0.80 at stem, 0.26 at b4, 0.13-0.22 at b5-b8 and under 0.01 from b9 (same units), no better than the
  unmatched decode (0.67, 0.20, 0.03-0.12).
- Midpoints between two images' sources are matched (miss under 0.001; 0.04 at logits); 1.5x extrapolations are
  not (0.20-0.94). This is solver performance, not a proof of infeasibility.
- The solver is a bounded residual-reduction scheme, not a certified nearest point (tangential share of the move
  entering the last step: 0.04-0.18). Its Jacobian, with the implicit derivative of the per-point width, agrees
  with a finite difference to 1e-4 to 6e-4 (16 images, one direction each).

**Edits: give image a the source coordinates of image b (pairs with different predicted classes).** Mean softmax
probability of b's class after patching, K = 16 / K = 64. It is probability mass, not a switch rate. A full swap
of the activation gives 0.97 everywhere (b's own confidence).

| Method | stem | b3 | b4 | b6 | b8 | b9 | edit size at b4 |
|---|---|---|---|---|---|---|---|
| residual guess | 0.03 / 0.06 | 0.20 / 0.39 | 0.24 / 0.42 | 0.46 / 0.71 | 0.83 / 0.90 | 0.92 / 0.94 | 0.44 / 0.55 |
| residual guess + matching | 0.04 / 0.09 | 0.20 / 0.38 | 0.26 / 0.43 | 0.48 / 0.68 | 0.83 / 0.83 | 0.94 / 0.94 | 0.47 / 0.57 |
| matching from x_a, Euclidean | 0.02 / 0.05 | 0.10 / 0.15 | 0.14 / 0.16 | 0.35 / 0.38 | 0.64 / 0.56 | 0.81 / 0.72 | 0.37 / 0.45 |
| matching from x_a, probe metric | 0.03 / 0.04 | 0.18 / 0.27 | 0.21 / 0.31 | 0.42 / 0.54 | 0.66 / 0.71 | 0.90 / 0.85 | |
| decode of b's coordinates, no reference | 0.14 / 0.23 | 0.09 / 0.21 | 0.17 / 0.35 | 0.63 / 0.80 | 0.87 / 0.94 | 0.95 / 0.95 | 0.77 / 0.81 |

Edit size is |x' - x_a| / |x_b - x_a|. Isotropic random edits of the same size as each of the first four methods
give under 0.03 outside logits. Readings: (i) the residual guess already does nearly all of the steering; matching
makes the source coordinates exact (miss 0.09-0.17 in conv sites to 0.000) without a consistent gain in b's probability, and the
methods differ in edit size, so they are not compared at equal budget; (ii) from b5 on, the decode with no
reference image steers at least as well, with a larger edit; (iii) Euclidean matching from x_a is a poor edit at
pool and logits (0.40, 0.31 at K = 16; at logits it also misses the target, 0.16); (iv) before b3 no method moves
the output much. A control matched to the probe's covariance, rather than isotropic, was not run.

**Other datasets (decode cosine against the activation actually produced, K = 64).** CIFAR-100: within 0.03 of
CIFAR-10 held-out through b6, lower later (0.57 against 0.71 at b9). SVHN: 0.60-0.75 through b6, higher than
CIFAR-10 (cause not measured). Pixel noise sigma 0.2: 0.25-0.34 at stem-b2, within 0.04 of clean at b3-b8. PGD
8/255: 0.02-0.09 lower than clean through b8, 0.77 at b9.

**Adversarial images in the embedding (PGD 2/255, K = 16).** In source coordinates the attacked image is within
0.10 of its clean version through b6 (mean squared distance in per-mode spread units; 2 is the expected value for
two independent images), 0.38 at b8 and 1.67 at b9. Its nearest class centroid agrees with the attacked network's
prediction for 16-25% of images through b6 and 94% at b9. In joint coordinates the displacement is already 0.35
at b5; zeroing the source changes at b7 and later reduces that to 0.015, so the early joint displacement is depth
mixing, not an early change at that site. Decoded images from late-site sources change visibly under attack; no
classifier was run on the decoded pixels.

## What I would use, and how sure I am

- **Kernel:** the participation-ratio rule. It removes the two-image modes and keeps an exact masked
  out-of-sample row. Width: k = 100 and k = 400 at N = 8,000 are within 0.02 in decode cosine at K = 64; the wider
  one is ahead at K >= 64 before b9 and behind at K = 4. No width sweep was run.
- **Point to activation, no reference image:** kernel-ridge decode, the best tested closed-form decoder by mean
  cosine. It is an estimate of a conditional mean: 0.42-0.51 in b1-b8 at K = 64, 0.71 at b9, 0.94 at pool.
- **Point to activation, with a reference image:** the residual guess; add source matching when the source
  coordinates must be met exactly. Do not expect the joint coordinate of the patched run to match before b9.
- **Site-specific questions:** read source coordinates. Joint coordinates deliberately carry the other sites.

## Limits and open ends

- One network, one checkpoint, one probe draw, CIFAR-10 test images; no variability across probes or seeds was
  measured, and the spec's bootstrap intervals and 5-fold small-probe runs were not done. Labels do not enter any
  kernel, basis or decoder, but they stratify the 2,000-image probe, generate the PGD examples and fit the
  centroid readout.
- The tuning split is not end-to-end held out: its coordinates come from an operator fitted on the whole probe,
  and tuning validates on in-probe coordinates, not out-of-sample ones. `H` informed the recommendations above,
  so they have no fresh confirmation.
- Held-out images that nearly copy a probe image get overshooting coordinates (zero-diagonal kernel; no mode of
  the L x L system is excluded for conditioning, maximum condition number 80 / 178 / 331 at K = 16 / 64 / 256).
- Decoder behaviour off the probe is a convention (kernel ridge with a linear part extrapolates linearly).
- Cross-network transfer was out of scope and is untested; `LITERATURE.md` gives the functional-map link through
  the shared probe.
- Lower coupling t was not explored. The 13-site operator (pixels as a site) was run only for the image figures.
- Kernel comparisons are at different effective widths; refit and half-split tests change several things at once.

## Reproduce

```
uv run --with numpy --with scipy --with pytest python -m pytest -q invert/test_dm.py
uv run --with modal python invert/modal_invert.py deploy            # stable stages: extract, gram, kernel, joint
INVERT_APP=crw-depth-invert-b uv run --with modal python invert/modal_invert.py deploy   # iterating stages
uv run --with modal python invert/modal_invert.py launch all '{"K":256,"pk":[["P8","pr400"]]}'
INVERT_APP=crw-depth-invert-b ... launch decode|rt|viz '{"units":[...]}' ; ... summary '{}'
uv run --with modal python invert/modal_invert.py fetch decode data/invert/decode summary.json
uv run --with modal python invert/modal_invert.py fetch rt data/invert/rt .json
uv run --with modal python invert/modal_invert.py fetch joint data/invert/joint <tag>/modes <tag>/coords_H <tag>/coords_pgd
uv run --with numpy python invert/modeloc.py ; ... adv.py <tag> 16 ; ... collect.py
uv run --with numpy --with pillow python invert/make_figs.py <tag>__L13      # after fetch viz
uv run --with pillow --with numpy python invert/build_page.py
```

`refit.py` also needs the `P2+H__*` modes fetched. Do not redeploy an app while its jobs are in flight (it kills
them); that is why the iterating stages deploy under other app names. All stage outputs are on the Modal volume
`crw-depth-invert-results`.

## Status and spend

Finished 2026-11-22 about 01:40 PST. Modal apps `crw-depth-invert` (stable stages) and `crw-depth-invert-b`
(decode, round trip, images, summary) are left deployed and idle; nothing is running. Logged unit time of the
completed runs prices at about $32 (decode 8.5 GPU-hours, round trip 5.0, joint solves 5.3 CPU-hours at 16
cores / 96 GB); runs that were cancelled and redone (the kernel change, a tuning fix, GPU queueing) are not in
those logs, so the true figure is higher, roughly $45-60. The spec projected under $15 for one pass. The exact
amount is on the Modal dashboard.
