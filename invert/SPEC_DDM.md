# Double diffusion maps as the inverse of the depth embedding

2026-11-22. Extends `invert/SPEC.md`; same network (ResNet-20 seed 0, step 64,000), same probe `P8` (8,000
CIFAR-10 test images), same held-out set `H` (2,000), same first-stage kernel (`pr400`), same joint modes
(`P8__pr400__L12`). Exploratory: one network, one probe draw. Revised after a GPT review of the plan and of the
code, and after a one-site pilot (`b4`); what changed is listed at the end.

## Question

On clean held-out images, does the double-diffusion-map lift (Evangelou et al., J. Comput. Phys. 2023,
arXiv:2204.12536) map a point of the diffusion embedding back to the activation at that site better than what
`invert/` already has (kernel ridge on the joint coordinates), in three separate senses: (a) reconstruction,
(b) the decode re-embeds where it was asked to, (c) the decode preserves what the network computes? And which
of the method's premises hold here?

What already existed: the `gh` decoder in `stage_decode.py` is the paper's lifting step (Gaussian kernel on the K
joint coordinates, spectral cut-off, Nystrom), with `(h, delta)` on a coarse grid whose edges were selected at
most sites, scored by reconstruction only.

## Method (`invert/ddm.py`). An adaptation of the paper, not a replication

The paper's first stage is an ordinary Gaussian diffusion map with coordinates picked by eye; ours is the
repo's participation-ratio Sinkhorn walk on cosine distance.

1. **First stage, three coordinate systems for a site l** (probe values; held-out values by Nystrom):
   - `joint`: slices of the joint through-depth modes, `f_k(., l)` (the embedding the viewer shows; held-out
     values from the stored `coords_H`). Not a function of the site's activation alone.
   - `source`: `s = p_l(x) . f_l`, one kernel step of the slices; a function of the site's activation.
   - `site`: eigenvectors of the site's own walk `P_l`, unit variance, constant removed; held-out
     `phi_k(x) = p_l(x) . phi_k / lambda_k` (the paper's setting). Variant `site-t1`: `lambda_k phi_k`.
2. **Non-harmonic selection** (Dsilva et al. 2018): leave-one-out local linear regression of coordinate k on
   coordinates 1..k-1, Gaussian weights of scale (median distance)/3, residual `r_k`, at 1,000 anchors. Reported
   for the first 64 coordinates; variant `*-llr16`: the 16 of the first 64 with the largest `r_k`.
3. **Second stage (latent harmonics)**: `K*_ij = exp(-|z_i - z_j|^2 / (2 h^2))` on the chosen coordinates,
   eigenpairs `(sigma_j, psi_j)`; lift `x_hat(z) = mean + sum_{j<=m} sigma_j^-1 (k*(z) . psi_j) <psi_j, X_c>`.
   The paper cuts off by count (300 of ~2,800 and 400 of ~3,800 points) and gives no rule for `h` or `m`.
4. **Restriction**: the first stage's out-of-sample row of the decoded activation, then the coordinate formula.

**Matched comparison.** {spectral cut-off, ridge} x {no trend, linear trend}, all on the same second kernel and
`h` grid (1..64 x the median 15-NN distance), all chosen by generalised cross-validation on the whole probe:
`gcv = (RSS / |X|^2) / (1 - df/n)^2`, `df` the trace of the hat matrix (for a trend plus a lift of its residual,
`rank(H) + sum_j s_j (1 - o_j)` with `o_j` the share of harmonic j inside the trend space), restricted to
`df <= n/2`; undefined at interpolation and excluded there. Cut-offs: `delta = 10^(-j/2)`, j = 1..24, plus fixed
counts; ridges `10^-8..10^2`. GCV targets pooled squared error; the split rule targets mean per-image captured
share. Also kept: `linear` and `krr` exactly as `stage_decode` chooses them (a control: the joint numbers must
reproduce `results.json`), `lh_split` (cut-off by the inner 80/20 split, the previous rule on the finer grid).

**Realistic decodes.** A reconstruction-optimal decode is a conditional mean. Compared with: `nn1` (the nearest
probe activation in the coordinates), `nn15`, `nn1+krr` (that activation moved by the difference of the krr
decodes at the requested point and at the neighbour's), the lift at high rank, and `lh_fsplit`: the cut-off
chosen on the inner split by the mean KL(clean || patched), instead of by squared error.

## Measured

Per site (12) x coordinate system x K, on `H` (replicate = image; mean, median, s.e.m.), every score computed
from decoded vectors:
- Reconstruction: cosine and captured share `1 - |x - x_hat|^2/|x|^2` to the true centred activation; decoded
  norm over true norm.
- Function: top-1 agreement with the clean run and KL(clean || patched), the decode patched in at the site.
- Round trip, three kinds, in units of the coordinate's variance over `H`, with the measured value for another
  image's coordinates (about 2) and for zero (1) as references:
  - **joint re-embedding** (`joint` cells, K <= 64, first 256 images): earlier sites from the image, this site
    decoded, later sites recomputed by the network, then the joint out-of-sample map; compared with the requested
    joint coordinates. This is the round trip in the viewer's coordinates.
  - **single-site restriction** (`site`, `source`): restrict the decode, compare with the request; for `site` both
    with the Nystrom `1/lambda` and without.
  - **source recovery** (`joint`): the decode's one-step coordinates against the true image's sources.
- Along the cut-off and along the ridge (cells `joint|16`, `joint|64`, `site|64`, `source|64`): all of the above
  at each `m` and each ridge, so reconstruction, function and round trip can be read against each other.
- Predictability of later coordinates: held-out R^2 and squared correlation of coordinate k > K predicted from
  the first K (or the omitted ones from the selected 16), by a cut-off lift and a ridge each tuned by GCV for
  those targets (standardised on the probe). Reported as predictability, not as an intrinsic dimension.

Cells: `joint`, `site`, `source` at K in {4, 16, 64, 256}; `site-t1|64`; `joint-llr16`, `site-llr16`; exploratory
`pr2`: second kernel = the first stage's rule on latent distances (positive spectrum only), `joint` K in {16, 64},
k2 in {30, 100, 400}. Reference that sees the activation: one kernel step.

**Controls with known answers.** (a) Synthetic 2-D strip in R^6 (`test_ddm.py`): LLR flags the harmonics, the
lift from the two selected coordinates reconstructs held-out points, lift-then-restrict returns them.
(b) `krr` on `joint` reproduces the stored `results.json`. (c) Rows of the true held-out activations reproduce
the stored sources; patching the true activation gives agreement 1 and KL 0; its joint re-embedding has miss 0.
(d) Cut-off and ridge scores equal scores from explicit weights; hat-matrix traces equal the closed forms; GCV is
infinite at interpolation; scores stay valid with a near-duplicate pair.

## Figures

1. Matched comparison at K = 64, `joint`, by site: cosine, captured share (median), patched agreement for
   cut-off vs ridge, with and without trend, plus `krr`, `linear`. *"At matched kernel and rule, ridge is X above
   the cut-off; the trend changes nothing."*
2. Along the cut-off: cosine, patched agreement, decoded norm and joint re-embedding against m, a few sites.
   *"Reconstruction peaks at m ~ X; function keeps improving to m ~ Y."*
3. Reconstruction against function for every decoder family (frontier), K = 64.
4. Coordinate systems: cosine by site for `joint` / `source` / `site` at K = 16, 64, 256.
5. Premise: `r_k` by site and k; predictability of later coordinates by site.
6. Round trips by site: joint re-embedding; single-site with and without `1/lambda`.

## Decisions made on the user's behalf

- Main result on the joint coordinates (the embedding asked for); per-site coordinates as the arm closest to the
  paper.
- Probe and kernel fixed at the previous session's recommendation; no kernel sweep.
- `H` is used for the surfaces and the cosine-only grid maximum ("oracle"), which are descriptive; the rules
  `gcv`, `split` and `fsplit` never see `H`.
- One fit per cell, no bootstrap over probes (uncertainty is over held-out images only).

## Compute

Unit: one site (12), A100 (float64 runs at half rate there, 1/32 to 1/64 on the cheaper cards). Pilot `b4`, full
configuration: 365 s. Projected: 12 x 6-12 min, about $5; wall-clock about 15 minutes.
Outputs: Modal volume `crw-depth-invert-results`, `ddm/P8__pr400__L12/<site>.json|npz`.

## Changed after review and pilot

- GCV selected the interpolant (0/0 clamped to 0). Now undefined at interpolation, residual computed from the
  discarded spectral energy, restricted to `df <= n/2`; regression tests added.
- Scores came from a Gram rebuilt from float32 cosines, which is not positive semidefinite to the precision the
  lift needs. All scores now come from decoded vectors.
- The local-linear-regression weights are shifted per row in log space so an isolated anchor cannot zero them.
- The round trip for `joint` compared with sources only; the joint re-embedding of the patched run is now the
  main one, and the source comparison is labelled source recovery.
- Later-coordinate predictability used weights tuned for activations; now tuned for the coordinates themselves.
- Added the matched cut-off/ridge x trend comparison, captured share and KL beside cosine, and function along the
  cut-off (the pilot showed reconstruction and function disagree about the cut-off).
- A nearest-neighbour decode is exactly a probe activation, so it is restricted with that probe image masked
  (the kernel's zero-diagonal convention).
