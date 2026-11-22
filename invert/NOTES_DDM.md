# Double diffusion maps as the inverse: what was built and what it measured

2026-11-22. Exploratory: one network (ResNet-20 seed 0, CIFAR-10, step 64,000), one probe draw (`P8`, 8,000 test
images), 2,000 held-out images `H`, first-stage kernel `pr400`, joint modes `P8__pr400__L12`. Spec with the review
changes: `invert/SPEC_DDM.md`. Code: `invert/ddm.py` (method), `invert/stage_ddm.py` (measurement),
`invert/test_ddm.py` (11 tests), `invert/ddm_collect.py`. Numbers: `data/invert/ddm_results.json`.

"Cosine" is the cosine between the true centred activation and the decode, mean over `H`. "Agreement" is the share
of `H` whose top-1 class is unchanged when the decode is patched in at the site. Round-trip misses are squared
errors in units of each coordinate's variance over `H` (measured references: another image's coordinates 2.0,
zero 1.0), medians over images. "Conv sites" are b1 to b8.

## What double diffusion is here

Evangelou et al. (2023): (1) a diffusion map, keeping only coordinates that are not harmonics of earlier ones;
(2) a second Gaussian kernel on those coordinates, whose eigenvectors ("latent harmonics") are a basis for
functions of the coordinates; (3) lift = expand each activation coordinate in the first m latent harmonics and
extend by Nystrom; (4) restrict = Nystrom of the first map. The previous session's `gh` decoder was step 3 alone on
the joint coordinates. This session adds steps 1, 2 and 4, a cut-off rule with no tuning split, a matched
comparison against ridge, and measurements of function and of the round trip. It is an adaptation: the first
stage is this repo's participation-ratio Sinkhorn walk, not the paper's plain Gaussian map.

## Findings

**1. As a reconstruction, the lift is second to kernel ridge on the joint coordinates; the matched comparison
points at the hard cut-off.** Same kernel, same width grid, same rule (generalised cross-validation, GCV): on the
joint coordinates the spectral cut-off trails ridge in cosine at every site and K. The gap is 0.003-0.009 at
K = 16, 0.013-0.025 in conv sites at K = 64 (0.004-0.016 elsewhere) and 0.005-0.027 at K = 256; paired standard
errors are below 0.001. Adding a linear trend to either changes cosine by at most 0.005 there, so the trend that
the old `krr` had and `gh` lacked is not what separates them. b4, K = 64: linear 0.488, cut-off 0.496, ridge
0.515. (In the single-site coordinate systems the ordering holds almost everywhere; `site|4` at b9 and logits
slightly favours the cut-off.)

**2. GCV was competitive with the tuning split, with no split.** On the joint cells the GCV cut-off is within
0.006 of the best grid point on `H` at every site and K, and between 0.003 below and 0.019 above the inner-split
rule, which usually keeps fewer harmonics. For ridge, GCV and the old inner-split `krr` agree within 0.007.
Outside the joint cells the shortfall from the grid maximum reaches 0.025 (`source|16`, pool). GCV is undefined
at interpolation; the first version selected it there and was fixed.

**3. The cut-off that reconstructs best is not the one that functions best.** On the joint coordinates at
K = 64, cosine peaks at about 200-300 of 8,000 harmonics in b3-b6 (about 500 in b7-b8), while agreement keeps
rising well past that and peaks short of, or at, full rank:

| b4, K = 64, h = 4 | m = 256 | m = 1,024 | m = 2,048 | m = 4,096 | m = 8,000 |
|---|---|---|---|---|---|
| cosine | 0.50 | 0.48 | 0.45 | 0.41 | 0.30 |
| agreement | 0.37 | 0.53 | 0.66 | 0.72 | 0.59 |
| decoded norm / true norm (median) | 0.53 | 0.63 | 0.74 | 0.85 | 1.24 |
| joint re-embedding miss | 0.17 | 0.16 | 0.13 | 0.13 | 0.07 |

The same happens along the ridge (b4, h = 4: ridge 1e-5 gives cosine 0.38, agreement 0.73), so it is the amount
of regularisation, not cut-off against ridge. A reconstruction-optimal decode has about half the true norm in
b3-b8; with less regularisation the norm, the agreement and (at b3-b6) the joint re-embedding all improve. My
reading, not separately measured: the reconstruction optimum is a blurred conditional mean and the high-rank
lift moves towards real nearby probe activations. The paper's own examples keep 300-400 harmonics of about 3,000
points, proportionally more than our reconstruction optimum. Size of the effect at K = 64, agreement at the
cosine-best point of the cut-off sweep against the best in the sweep: b2 0.15 / 0.58, b3 0.23 / 0.70,
b4 0.37 / 0.75, b5 0.73 / 0.84, b6 0.76 / 0.87, b7 0.89 / 0.91; no gain at b8; at b9, pool and logits the
agreement-best point has lower rank than the cosine-best one (pool 0.98 against 0.91). These maxima are read off
`H`, so they describe the trade-off; they are not a held-out score.

**4. Held-out versions: a cut-off chosen for function, and the nearest probe activation.** Choosing the cut-off
on the probe's inner split by the mean KL of the patched run (`lh_fsplit`) picks high rank at b1-b7 (4,665-7,338
of 8,000; width 8 at b1, 2 at b2, 1 at b3-b7) and gives held-out agreement 0.36 / 0.49 / 0.69 / 0.74 / 0.83 / 0.87
at b1-b6 against 0.12 / 0.15 / 0.24 / 0.39 / 0.74 / 0.77 for kernel ridge, at a cosine cost of 0.12-0.21. It
fails at stem, where it selects what is effectively the probe mean (agreement 0.10). The nearest probe
activation in the coordinates (`nn1`), with no fitting, gives 0.58-0.84 at stem-b6: far ahead at stem-b2, and
0.04-0.06 behind `lh_fsplit` at b3-b6. Moved by the difference of the krr decodes (`nn1+krr`) it gives 0.57-0.87
and is within 0.02 of `lh_fsplit` at b3-b6, with cosine 0.21-0.34 in conv sites. Neither wins at every site, and
on mean KL the lift is ahead where they tie on agreement (b4: 1.48 against 2.26). For comparison the previous
session's trained MLP reached 0.49 at b4 and 0.79 at b5. All of this is clean held-out patching; edits and
transfer were not tested.

**5. Round trips.** (a) *In the viewer's joint coordinates* (decode patched in, later sites recomputed, K = 64,
first 256 held-out images): kernel ridge misses by 0.82 at stem, 0.70 at b1, 0.41 at b2, 0.30 at b3, 0.18 at b4,
0.13 at b5, 0.07 at b6 and less after; the GCV cut-off lift is within 0.04 of that. The nearest probe activation,
replayed with itself masked, misses by 0.32, 0.21, 0.16, 0.12, 0.08, 0.05 at stem-b5: 2.2 to 3.4 times closer.
(b) *The paper's round trip* (lift, then Nystrom restriction with 1/lambda; single-site coordinates, K = 64, GCV
ridge): the miss is 1.6-4.1 in conv sites, worse than returning zero. Dropping the 1/lambda from the output gives
0.19-0.21 against the same target. A likely reason, not isolated here: a smoothed decode sits among the images
that share its coordinates, so its kernel row returns phi rather than lambda phi. High rank can close the gap at
some widths (b4, full rank: 0.35 at h = 2, 0.15-2.5 across widths). From b9 on the ridge decode's round trip
closes (0.14, 0.007, 0.004). (c) `nn1+krr` re-embeds badly in joint coordinates (2.7 at b4). It lies next to a
probe image and is not masked, while `nn1` is; the kernel's zero diagonal makes those two conventions differ
sharply, as noted last session for near-duplicate held-out images. This comparison does not separate that from
other causes.

**6. Is the premise true? Partly; and more coordinates did not rescue reconstruction.** The paper lifts from a
few coordinates because the discarded eigenvectors are functions of the kept ones. Dsilva's test on the first 64
single-site coordinates: 5-10 have residual above 0.5 in stem-b9, 2-3 at pool and logits (the first counts by
convention); the median residual of the rest is 0.24-0.43 in stem-b9 and 0.14-0.16 at pool and logits, neither
new directions nor clean harmonics. Held-out predictability of later single-site eigenvectors from the first K
(ridge tuned by GCV for them; median R^2 over coordinates K+1..256): 0.05-0.08 from K = 4 and 0.22-0.48 from
K = 16 in stem-b8; from K = 64, 0.32-0.72 in stem-b8 and 0.82 / 0.93 / 0.96 at b9 / pool / logits. (At K = 16, pool
and logits have negative R^2 with squared correlation 0.3: a scale mismatch in the extension, not a prediction.)
So from b9 the first 64 coordinates predict the rest well, which is the method's setting; in conv sites they
predict about half. Predictability of eigenvectors is not the same as determining the activation. One kernel
step from the full row, which uses every eigenvector and sees the activation, has cosine 0.56 at b4; 256
coordinates reach 0.58-0.60. That is consistent with the first kernel's smoothing (participation ratio 400 of
8,000) limiting what any coordinates carry, but it is one estimator, not a ceiling.

**7. Coordinates.** Single-site coordinates decode better than the joint slices, increasingly with depth: source
against joint at K = 64 (GCV ridge): +0.01-0.02 at stem-b5, +0.03 at b6, +0.05-0.07 at b7-b9, +0.03 at pool; at
K = 16 the site's own eigenvectors are +0.04 at b4 and +0.09-0.10 at b7-b8 over joint. Selecting 16 of the first
64 coordinates by Dsilva's residual gives +0.001 to +0.017 on joint coordinates in stem-b8 and -0.01 to -0.04
from b9; on single-site coordinates it is neutral in conv sites and +0.01 at pool and logits. Scaling coordinates
by lambda helps by 0.007-0.014 in conv sites and not after.

**8. Negative: the second stage with the first stage's own kernel rule** (per-point width, Sinkhorn) is below the
Gaussian second kernel in every tested cell (b4, K = 64: 0.43-0.45 against 0.50). At pool and logits with wide k2
the GCV-chosen cut-off collapses (0.10, 0.25) although smaller cut-offs reach 0.90 and 0.95, so that part is a
failure of the rule on this kernel. Its normalised walk has negative eigenvalues, so only part of the spectrum is
usable. Exploratory.

## What I would use

- **Reconstruction:** kernel ridge, width and ridge by GCV (no split). The double-diffusion lift is the same
  estimator with a hard cut-off and loses 0.01-0.03 cosine at K = 64 in conv sites.
- **Patching a decode back into the network, before b7:** not the reconstruction-optimal decode. Either the
  nearest probe activation with the krr correction, or the lift at high rank (m = n/4 at h = 2 keeps cosine
  within 0.06 of the best cut-off and gives 0.53 / 0.65 / 0.82 / 0.85 agreement at b3-b6). They are close on
  agreement; the lift is better on KL from b3, the neighbour is far better at stem-b2. Whether this carries to
  edits and cross-model transfer is untested.
- **Single-site questions:** source coordinates, as last session; they also decode better.
- **Do not use the paper's restriction (with 1/lambda) on a smoothed decode.**

## Related approaches

Tested here: GCV in place of the split; cut-off chosen for function; nearest probe activation plus correction;
single-site and Dsilva-selected coordinates; the adaptive second kernel.

Proposed, not run:
1. **Nearest probe activation, then source matching.** Last session's Gauss-Newton matching makes source
   coordinates exact starting from a reference image. Starting it from the nearest probe activation needs no
   reference and should give a realistic activation with an exact single-site round trip. This is the equation-free
   "lift from an analog, then constrain" recipe and the most direct next step.
2. **A finer or multiscale first stage.** Finding 6 is consistent with the first kernel's smoothing limiting
   reconstruction. A narrower kernel may recover more but brings back the near-duplicate modes; a multiscale basis (diffusion wavelets, already
   started in `wavelets/`) is the principled way to add fine coordinates.
3. **Embed spatial positions, not whole images.** A conv activation is 16-64 channels at each of 64-1,024
   positions. A diffusion map over positions has clouds of low dimension, where the paper's premise (finding 6)
   should hold and the lift should be near exact. It changes what a "point" is, so it is a different embedding.
4. **A conditional sampler** (a small diffusion model for the activation given the coordinates) is the complete
   answer to one coordinate point having many activations. It is trained, so it is a benchmark for the closed-form
   options, not a replacement.
5. **Mask or smooth the zero diagonal for near-probe queries** so that realistic decodes re-embed continuously
   (finding 5c).

## Limits

- One network, one checkpoint, one probe draw; uncertainty is over held-out images only.
- The function trade-off and its maxima are read on `H`; only `gcv`, `split`, `fsplit`, `nn1` and `nn1+krr`
  numbers are held out.
- Agreement measures the top-1 class only. KL (clean against patched) is in `ddm_results.json`; it agrees on the
  trade-off against reconstruction-tuned decoders but ranks the realistic decoders differently (finding 4).
- Joint re-embedding uses the first 256 held-out images, in units of the coordinate's variance over all 2,000;
  `nn1` is replayed with its own probe image masked, the other decoders are not masked. The "without 1/lambda"
  round trip changes the output only; its target is still the Nystrom coordinate.
- Dsilva's test in 60 dimensions with about 400 effective neighbours is noisy; the predictability test is the
  sturdier of the two. Neither is a dimension estimate.
- No out-of-distribution sets, no extrapolated coordinates, no cross-model transfer in this round.

## Reproduce

```
uv run --with numpy --with scipy --with torch --with pytest python -m pytest -q invert/test_ddm.py
INVERT_APP=crw-depth-invert-d uv run --with modal python invert/modal_invert.py deploy
INVERT_APP=crw-depth-invert-d uv run --with modal python invert/modal_invert.py launch ddm '{"units":[{"pset":"P8","kname":"pr400","site":"b4"}, ...]}'
uv run --with modal python invert/modal_invert.py fetch ddm data/invert/ddm P8__pr400__L12
python3 invert/ddm_collect.py
```

## Status and spend

Finished 2026-11-22. Twelve sites, 3,919 GPU-seconds on A100/H100 plus two pilots: about $5 by my estimate from
logged seconds at list rates (the exact figure is on the Modal dashboard); the full run took about 10 minutes. The app `crw-depth-invert-d` is deployed and idle. Controls at every site: stored
kernel-ridge cosines reproduced to 6e-8, stored sources to 1e-9, patching the true activation gives agreement 1
and KL below 2e-11, its joint re-embedding misses by less than 2e-12.
