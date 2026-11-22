# Spec: inverting the through-depth embedding (exploratory)

Owner request (2026-11-21): map a point of the joint through-depth embedding back to the activation at that
site, in a principled way with little tuning, good in distribution and reasonable out of distribution
(held-out images, other datasets, interpolation and extrapolation, adversarial images). Kernel, probe size and
number of modes are all free. Also decode to pixels. Cross-network transfer is the eventual use but is out of
scope here. Prior exploration: `REEMBED_NOTES.md` (one agent, unreviewed).

Fixed: ResNet-20 seed 0 on CIFAR-10, step 64,000 (test accuracy 91.56%); 12 sites (stem, b1..b9, pool,
logits); activations flattened, centred over the probe, rows normalised (cosine is the only geometry); the joint
operator of `spacetime.py`, relative site units, coupling t = t*; nothing is trained inside the construction;
closed-form fits are allowed and called fits; labels never enter a kernel, basis or decoder.

## 1. Objects

**Kernel (replaces the 0/1 k-NN graph).** The repo's existing "perplexity" rule (rules 1-2 of the kernel
bank, `sweep_math.perplexity_logkernel`): each probe image i gets its own Gaussian width on cosine distance so
that its conditional distribution `q(j|i)` over the other probe images has perplexity `xi`; the kernel is
`(q(j|i) + q(i|j))/2`, zero diagonal, scaled by symmetric Sinkhorn to a doubly stochastic P. One parameter,
`xi`. It is smooth in the activation, so a new point x has a row: calibrate its own width on its cosines to the
probe, average its conditional with each probe image's frozen conditional towards it, multiply by the probe's
frozen Sinkhorn scalings, normalise. A probe image replayed with itself masked gets its own row of P exactly
(tested); an unmasked new point arbitrarily near a probe image does not converge to that row (it keeps weight on
the image itself), which is inherent to a zero-diagonal kernel. A single global width (one eps, Sinkhorn) was
tried first and rejected in the pilot: at small eps tight groups of images detach (second eigenvalue 0.998-1.000
at b1, b2) and the leading modes do not reproduce on a disjoint half. Compared: xi in {30, 100, 300} at
N = 2,000 (scaled with N for larger probes), the global-width kernel, and the k-NN rule as baseline.

**Forward map (activation to coordinates).** Frozen-probe extension. Writing the eigen-equation of the joint
operator at a new point (one-way affinities to the probe, depth links between the point's own sites) gives, per
mode, a tridiagonal L x L system in the unknown slice values `y_l`:
`(y_l - p_l(x).f_l)/(2 c_l) + t (deg_l y_l - y_{l-1} - y_{l+1}) = mu y_l`. Fed the probe's own rows it returns
the modes (tested); it is not the coordinates the point would get in a refitted, larger operator. Modes whose
system is ill-conditioned (mu near the spectrum of the L x L matrix) are reported and excluded. It needs the new
image's activation at every site. A single-site forward map, for an activation given at one site only: with
the other sites' rows held at a reference image's, the system is affine in the row at site l, so
`y_l(x) = y_l(ref) + [T^-1]_ll (p_l(x) - p_l(ref)).f_l / (2 c_l)`; with no reference, one kernel step
`p_l(x).f_l` is the only parameter-free choice and is reported with its (large) bias.

**Inverse map (coordinates at a site to activation at that site).** Any decoder here is an estimate of the
conditional mean of the activation given K coordinates; the candidates differ in assumptions and tuning.
Every candidate is linear in the probe activations, `x_hat = sum_j W(psi)_j x_j`:

| Name | Definition | Tuning | Status |
|---|---|---|---|
| linear | least-squares projection of the activation coordinates onto span{1, K slice functions} | none | projection |
| geometric harmonics | Gaussian kernel on the K coordinates, eigen-expansion truncated at `delta`, Nystrom (Coifman & Lafon 2006; 'latent harmonics', Evangelou et al. 2022) | (h, delta) on an inner split of the probe | fit |
| Laplacian pyramid | multiscale kernel smoothing of residuals, halving the width from the cloud diameter (Rabin & Coifman 2012) | stopping level: inner split, or the zero-diagonal rule of Fernandez et al. 2020 (no split) | fit |
| kriging | linear part + Gaussian-kernel ridge on its residual | (h, ridge) on an inner split of the probe | fit |
| neighbour mean | mean of 15 nearest probe images in coordinates | k = 15 | baseline |

Tuned decoders are tuned on an inner 80/20 split of the probe; `H` is used only for final scores. The cosine
embedding does not carry the activation's norm, so the decoders' target is the unnormalised centred activation
(scale predicted by the same weights); results with the true norm supplied are shown separately as "oracle
norm". References, not K-coordinate decoders: PCA with K + 1 components and one kernel step in activation space
(both see the target activation itself), and a trained MLP from coordinates to activations as an empirical
same-input benchmark (not an upper bound; trained; not part of the method).

**Matching pre-image (exact round trip; also the form used for interventions).** Given target coordinates psi'
at site l and a starting activation x0, find the activation nearest x0 whose single-site forward map equals
psi': `min |x - x0|` subject to `F_l(x) = psi'`, solved by Gauss-Newton on the smooth kernel (the constraint
Jacobian is a weighted sum of probe activations, so the search stays in their span plus x0). With x0 a real
activation this is the least change that moves it to psi' (the equation-free "lifting with matching" and the
Nystrom-constrained pre-image of Thorstensen et al.); with x0 = the conditional-mean decode it is a decoder
with an exact round trip. The simpler residual form `x' = D(psi') + (x - D(psi(x)))` is its first guess; it
is a reference-conditioned intervention and guarantees nothing about where x' re-embeds, which is measured.

**Pixels.** (a) Decode the image as a function on the probe from the coordinates at any site with the same
decoders (the mean image of what is embedded there); (b) a 13-site operator with centred, normalised pixels as
site 0, so that pixel space has coordinates too.

## 2. Data

CIFAR-10 test set, all 10,000 images. `P2` = the existing 2,000-image probe; `P8` = `P2` plus 6,000 more
(stratified by class only through the inherited probe; the rest random); `H` = the remaining 2,000, held out
from every operator and every fit. Out-of-distribution sets, 2,000 each unless stated: CIFAR-100 test, SVHN
test, `H` with Gaussian pixel noise (sigma 0.05, 0.1, 0.2), and PGD L-inf adversarial versions of 1,000
images of `H` (eps 2/255 and 8/255, 20 steps, untargeted). Replicates: images. Uncertainty: bootstrap over
held-out images; for the small-probe case also 5 folds of `P2`.

## 3. Measurements

- **M0 kernel.** Per site and kernel: k-occurrence of the 15 heaviest entries per row (max, skew), spectrum,
  and reproducibility of modes on a disjoint half (mode of half A extended to half B by the kernel's own
  out-of-sample row; share inside B's first 2k + 10 modes, minus chance). Control: the k-NN rule, whose
  numbers are in `REEMBED_NOTES.md` result 5.
- **M1 forward.** (a) Correctness, in `invert/test_dm.py`: masked probe rows reproduce P; the tridiagonal
  system fed those rows reproduces the modes; an independently assembled check of the equation. (b) Refit
  stability: coordinates of `H` by the frozen-probe extension against their coordinates in an operator refitted
  on probe plus `H` (perplexity scaled with N; c_l and t* recomputed; centring unchanged). Compared by
  projection onto the refitted K-mode space with one orthogonal alignment fitted on the probe images across all
  sites, with eigenvector normalisation rescaled for N; reported per site as the share of squared coordinate
  norm reproduced, and separately for weakly occupied slices.
- **M2 decode.** Held-out cosine and captured share (1 - squared error / squared norm) between true and decoded
  activation, per site, K in {4, 16, 64, 256}, probe in {P2, P8}, perplexity, decoder. Controls with known
  answers: the probe mean (K = 0) has captured share about 0; patching the true activation gives KL 0.
- **M3 round trip.** First the oracle floor: encode the true held-out activation with the single-site map and
  compare with its joint coordinates. Then decode and re-encode with that same map, for held-out points and for
  activation-space interpolations with known pre-images. Requested coordinates with no known pre-image
  (latent midpoints, 1.5x extrapolation) are reported as requests, with the achieved coordinates and a support
  score. The matching pre-image is scored by its residual constraint violation and by |x - x0|.
- **M4 function.** Patch the decoded activation (probe mean added; scale as predicted by the decoder; oracle
  norm shown separately) into the network at the site and compare outputs with the clean run: KL of the softmax,
  top-1 agreement. References: probe mean, PCA-K, neighbour mean; control: the true activation. Interventions:
  move a held-out image's coordinates at site l to those of an image of another class (residual form and
  matching pre-image); report the achieved coordinates, the output, and a matched-size random activation edit
  as a control.
- **M5 out of distribution.** M2 and M4 on the out-of-distribution sets, with a label-free support score per
  point (nearest-probe cosine and row perplexity, calibrated on clean held-out images) and decode error against
  retained coverage. Adversarial: attack success, decode of the attacked activation, and coordinates of clean
  and attacked versions through depth.
- **M6 pixels.** Held-out image reconstruction error in pixel units against the mean image and the
  latent-neighbour mean, originals shown beside decodes; 12-site (pixels as a decoded function) and 13-site
  (pixels as a site) reported separately.

## 4. Figures (page)

1. Kernel: max k-occurrence and half-split reproducibility by site, one line per kernel. *"Kernel X has
   this much hubness and its low modes reproduce this well, against the k-NN rule."*
2. Forward: R^2 of out-of-sample coordinates by mode index, per site group. *"A new image lands where it would
   have if it had been in the probe, for the first m modes."*
3. Decode: held-out cosine by site, one line per decoder, facets K and probe size, with the PCA-K and MLP
   references. *"From K coordinates decoder X recovers this much, this close to the ceiling."*
4. Decode against K and N at three sites. *"Reconstruction changes this much with the number of coordinates
   and this much with the probe size."*
5. Round trip R^2 by site and decoder. *"Decoding then re-embedding returns the point asked for."*
6. Function: KL and top-1 agreement after patching, by site and decoder. *"The decode carries / does not carry
   what the rest of the network uses."*
7. Out of distribution: decode cosine and support score by dataset. *"Off-distribution points decode worse and
   are flagged."*
8. Pixels: grid of decoded images along paths in the embedding. Adversarial: clean and attacked trajectories.

## 5. Decisions made on the owner's behalf

- The repo's per-point perplexity rule rather than a new kernel: already in the kernel bank, one parameter,
  exact masked out-of-sample row. Alternatives: symmetric entropic affinities (Van Assel et al. 2023), local
  scaling (Zelnik-Manor & Perona).
- Zero diagonal, so a new point is treated as a probe point is (by its affinity to the other probe images).
  Alternative: keep the diagonal, which at these distances puts nearly all of a row's weight on itself.
- Held-out images come from the CIFAR-10 test set, not the training set (training images were fitted).
- Recommendation criterion, in order: no fitted parameters or automatic ones, sane behaviour off the probe,
  accuracy.

## 6. Compute

Modal, app `crw-depth-invert`, volume `crw-depth-invert-results`; inputs read-only from `cm-cifar-coherent`
and `crw-modelzoo-train`. Units: one extraction (GPU, minutes); per-site Grams (13 units, GPU, minutes);
kernels (site x probe x perplexity, CPU); joint solves (probe x perplexity, 16 cores / 64 GB, about 30 min at
N = 8,000); decode and patching units. Projected spend under $15, wall-clock about 3 hours for the main pass.

## 7. Amendment after the pilot (2026-11-21, evening): source coordinates

The pilot's round-trip stage showed that a joint coordinate at site l is not a function of the activation at
site l. Per mode k the joint out-of-sample system is `y_k = T_k^-1 (s_k / 2c)` with
`T_k = diag(1/(2 c_l)) + t D_path - mu_k I` and `s_{l,k}(x) = p_l(x) . f_{l,k}`: the joint coordinates are a
fixed linear filter along depth of the per-site **source coordinates** `s_l(x)`, each of which depends on the
activation at its own site only. At t = t* site l's own source carries 16-27% of the weight in `y_l`
(measured), so swapping in the true target activation at one site, with the other sites' sources held, moves
the joint coordinate only 17-33% of the way, and "the activation at site l whose joint coordinate is psi'" has
no solution in general. The earlier "held" convention (section 1, forward map) is dropped.

The single-site inverse is therefore posed in source coordinates:

- **Forward, one site:** `s_l(x) = p_l(x) . f_l`. Exact, no reference image, smooth. On the probe
  `s_l = P_l f_l`.
- **Whole image:** joint coordinates = depth filter of the source trajectory (the Nystrom extension, unchanged).
- **Requested joint coordinate psi' at site l -> source target:** `s'(psi') = W(psi') (P_l f_l)`, the source
  coordinates decoded as a function on the probe with the decoder's weights (a fit; error measured on `H`).
- **Matching pre-image:** nearest activation to x0 with `s_l(x) = s'`, damped Gauss-Newton with a trust region;
  x0 = the conditional-mean decode (a decoder with an exact source round trip) or a real activation (the least
  change that moves it to s'). Sources are kernel averages of probe slice values, so targets outside their
  convex hull have no pre-image; the solver then returns a bounded nearest attempt and the miss is reported.
- **Where a patched run lands:** sources before l from the reference image, at l from the patched activation,
  after l from what the network computes; then the depth filter. Reported as "patched-run coordinates".

Measurements M3/M4 are run in this form (`invert/stage_rt.py`, experiments E0-E5), with a random edit of
matched size as the control for every edit.

## 8. Amendment (2026-11-21, night): near-duplicate images and the participation-ratio rule

Tuning kernel ridge at the 8,000-image probe exposed validation images with enormous decodes. Cause: under the
Shannon-perplexity rule (`ea<xi>`) several leading joint modes are carried by pairs of near-duplicate test images
(e.g. at `P8__ea400` mode 14 has 94% of its squared norm on two images; 46 of the first 64 modes have more than
10% on two images; at `P2` it is 11-14 of 64). A row with Shannon perplexity xi can still put up to about 40% of
its weight on one very close neighbour (one heavy neighbour plus a uniform remainder satisfies the entropy
constraint), so a duplicate pair keeps enough mutual weight to become a low mode of the joint operator. The 0/1
k-NN rule cannot do this (every neighbour has the same weight): it has no such mode at either probe size.
Measured by `invert/modeloc.py`.

Change: calibrate each row by its **participation ratio** `1 / sum_j q(j|i)^2 = k` (Renyi-2 effective number of
neighbours) instead of `exp(entropy)`. Then no single neighbour can hold more than `1/sqrt(k)` of a row. Kernel
names `pr<k>`; everything else (symmetrisation, Sinkhorn, masked out-of-sample row, implicit derivative of the
width) is unchanged and covered by the same tests, plus `test_participation_rule_bounds_duplicates`. The headline
runs are redone with `pr100` and `pr300` at N = 2,000 and `pr100` and `pr400` at N = 8,000; the `ea` runs are kept
as the comparison that shows the failure.

Also changed after the same diagnosis: kernel ridge is solved directly rather than through an eigendecomposition,
the ridge grid extends to 100, and the linear decoder is always a candidate in the tuning.
