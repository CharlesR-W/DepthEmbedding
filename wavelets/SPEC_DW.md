# Diffusion-wavelet embedding and its inverse

2026-11-22, revision 3 (after two rounds of GPT-6.2 Astra review; changes listed at the end; where the
"Amendments" section and the body disagree, the amendments win). Extends `invert/SPEC_DDM.md`:
same network (ResNet-20 seed 0, step 64,000), probe `P8` (8,000 CIFAR-10 test images), held-out `H` (2,000),
first-stage kernel `pr400`, 12 sites. Exploratory: one network, one probe draw.

## What the user asked

1. A reconstruction (embedding point -> activation at that site) "comparable to double diffusion but using the
   wavelet basis", hoped to be more stable; "I may simply be thinking of the geometric harmonics per se". On being
   asked which instability: "something which works principledly, so that the cut-off is more or less
   interpretable as a spectral truncation".
2. A viewer mode showing the wavelet embedding (chosen form: scaling-function coordinates of the joint
   depth-as-time operator, with a scale selector), deployed to the public site with an Experiments entry.

## Three separate questions (kept apart in every figure)

- **Q1, decoding a given point (primary; comparable to double diffusion).** Input fixed: the K = 64 joint
  coordinates (`joint|64`), and separately the 64 source coordinates (`source|64`). Only the function basis on
  that coordinate space changes.
- **Q2, the stable spectral cut-off (single-site control with a known answer).** Input: the site's own Nystrom
  coordinates. A full band decoded linearly is geometric harmonics per se; by span invariance it is identical in
  eigenvectors and in scaling functions, so this arm measures the cut-off, not the basis.
- **Q3, adaptive compression (a different, supervised embedding).** Are activations sparser in localised
  wavelets than in eigenvectors? Its n is a number of transmitted features, not a spectral cut-off, and its
  coordinates cannot be computed from a point given in the first n eigen-coordinates.

## Conventions (`wavelets/dw.py`)

Symmetric operator `T` with eigenpairs `T U = U diag(lam)`, U with Euclidean-orthonormal columns.
- **Band** `V(c) = span{u_k : lam_k >= c}`; a level is a cut-off c. With eps = 1/2 the equivalent diffusion time
  is `t = ln 2 / (-ln c)`, and the Nystrom reciprocal on the band is at most `A = 1/c`. Levels are a list of
  allowed amplifications `A` (dyadic: 1.5, 2, 4, 8, ..., then the positive numerical floor), fixed from the
  spectrum table below before any held-out score is read; empty or repeated bands are dropped; whole
  multiplicity blocks are kept together. Fixed ranks (4, 16, 64, 256, 1,024) are added as comparison points.
  `W_j = V_j (-) V_{j+1}` (coarser = smaller index is not assumed; bands are named by A).
- **Localised basis of a band** with eigen-columns `U_B`: column-pivoted QR of the band-filtered deltas
  `C = diag(lam_B) U_B^T` (columns = probe points) gives `Q`; `Phi = U_B Q`. Signs fixed so the entry at the pivot
  is positive. The pivot is the seed point, not necessarily the peak.
- **Out-of-sample**, for a frozen kernel row `p_x`: `z_Phi(x) = p_x U_B diag(lam_B)^-1 Q` (divide in the
  eigenbasis, then rotate). The smoothed variant is `p_x U_B Q` and is named `t1`.
- **Name.** An exact spectral-band diffusion-wavelet construction with pivoted-QR bases. The spaces are
  Coifman-Maggioni's Eqs. 4.9-4.11; the bases are not their recursive compressed QR, and neither their
  localisation guarantees nor their fast transform are claimed. Localisation is measured, not assumed.

Operators:
- **Site** (Q2, Q3): the `pr400` Sinkhorn walk `P_l` (zero diagonal, so it has negative eigenvalues); positive
  spectral sector only, constant removed; the omitted complement's share of activation energy is reported.
- **Second stage** (Q1): `T = K*/sigma_1` for the Gaussian Gram `K*` of `ddm.LatentHarmonics` at the width the
  stored `lh` GCV picked, no mean removal, same relative floor; extension `k*(z) U Sigma^-1 Q` (raw). A positive
  contraction, not a Markov walk.
- **Joint, viewer** (below): `L_t` of `spacetime.py`; `lam = exp(-mu)` is used only to order and band the modes.

## Decoders

Q1, per site, cells `joint|64` and `source|64`, coefficients `B = Phi^T X_c`, decode `mean + z_Phi(z) B`:
`lead(n)` leading n harmonics (the stored `lh` cut-off; must reproduce `LatentHarmonics.weights`), `ebest(n)` best
n harmonics by `|B_k|^2`, `wbest(n)` best n functions of the multiscale wavelet basis, `rbest(n)` best n of a
random orthogonal rotation within each band (does localisation matter, or only leaving eigenvalue order?),
`ridge` on the same kernel, and the stored tuned `krr` as the practical benchmark.
Atom ranking for `*best` uses an inner 80/20 split of the probe (ranked on the 80%, n chosen on the 20% by
pooled squared error); curves along n on `H` are descriptive.

Q2, per site: `band(A)` = all of `V(1/A)` from the site's own Nystrom coordinates. Controls: equals the stored
`site|K` `linear` decode at K = dim; identical via `Phi` and via `U`.

Q3, per site: in-sample energy captured by the first n eigenvectors, the best n eigenvectors, the best n wavelets
and the best n of within-band random rotations; and the held-out decode from those n features (computed from the
true activation's kernel row; labelled as a new encoding).

## Measured (on H; replicate = image; mean, median, s.e.m.; paired differences between decoders)

- Spectrum table per operator: rank, cut-off, smallest kept eigenvalue, largest reciprocal, gap at the edge,
  share of `|X_c|^2` inside the band.
- Reconstruction: cosine, captured share, decoded/true norm.
- Function: top-1 agreement and KL of the patched run (control: the true activation gives 1 and 0).
- Error amplification: for held-out z and shared perturbations delta (in the input coordinates, 1%, 5%, 20% of
  the coordinate spread; Q1 nonlinear map by paired finite differences, Q2 also the exact singular values of
  `p -> p U Lam^-1 Q B`), `|x_hat(z + delta) - x_hat(z)| / |delta|` in units of activation spread per coordinate
  spread: median and 95th percentile; near-probe held-out images flagged.
- Round trip (Q2): the band coordinates of the decode against the requested ones, the same forward map on both
  sides, total squared error over total coordinate variance; measured references for zero and for another
  image. The `t1`-output-against-exact-target number is kept as a cross-convention diagnostic only.
- Reconstruction against function along the cut-off: both curves with s.e.m.; the two maxima on `H` are
  descriptive (double diffusion at b4, `joint|64`: rank 256 against 4,096).
- Localisation of each basis: number of points holding 90% of squared mass, and share of mass within the pivot's
  400 nearest neighbours, for wavelets, eigenvectors and within-band random rotations.

## Controls with known answers (`wavelets/test_dw.py`)

1. Two unequal eigenvalues and a nontrivial rotation: `z_Phi` is `p U Lam^-1 Q`, and differs from `(p Phi)/lam`.
2. Masked probe replay: `e_i^T P U Lam^-1 Q = e_i^T Phi` on the band.
3. Ring graph: band ranks and projectors equal the analytic Fourier ones; the projector is shift-equivariant;
   orthonormality; `V_j = V_{j+1} (+) W_j`; Parseval including the omitted tail; a band-limited signal is
   reconstructed and extended exactly.
4. Small gapped graph: a port of the paper's recursive QR (`wavelets/ref/dw`) against the exact bands: ranks,
   projector error, with its own tolerance reported (its QR threshold is on column norms).
5. Span invariance: `band(A)` via `Phi` = via `U` to 1e-8, and = stored `site|K` linear decode on `H`.
6. Full-band second-stage prediction = `LatentHarmonics.weights` on held-out queries.
7. A decoder given a latent point never touches the query activation after encoding (interface test).

## Figures

1. Q1: cosine and agreement against n (log x) for `lead`, `ebest`, `wbest`, `rbest`, with `ridge` and `krr` as
   marks at their own df, `joint|64`, a few sites + all-site summary at the split-chosen n. *"On the same 64
   coordinates the best-n wavelets are X above / below the leading-n harmonics; ridge is Y."*
2. Q1/Q2: error amplification against n or A. *"The lift that functions best amplifies input error Z times more
   than the one that reconstructs best."*
3. Q2: cosine, energy share, agreement and round trip against allowed amplification A, by site. *"With gain at
   most 2 the spectral truncation keeps r functions and reaches cosine X; the round trip closes to Y."*
4. Q3: compressibility curves, a few sites. *"Activations are / are not sparser in wavelets than in
   eigenvectors, and a random rotation does as well / worse."*
5. Localisation table.
6. Reconstruction against function along the cut-off, against the stored double-diffusion curve.

## Viewer (`site/`), on the viewer's own operator

The public viewer is a different operator from the inversion experiments: `L_t` at t = t*, scale `own`, on the
union 15-NN Sinkhorn walk, probe `P2` (N = 2,000; not a prefix of `P8`), L = 12. The wavelet mode is built on
exactly that operator (reference network, step 64,000) and says so.
- Dense unquantised eigendecomposition of `L_t` on slice-mean-zero functions (24,000-dimensional, Modal), so
  every advertised band is complete; residuals, the gap at each band edge and whole multiplicity blocks checked.
- Levels: cut-offs `mu_c` chosen from the spectrum for ranks near 3, 6, 12, 24, 48, 96 (snapped to the largest
  nearby gap), reported with `t = ln 2 / mu_c`.
- Shipped per level: at most 16 scaling functions, chosen by a stated coverage rule (pivot order, which is
  greedy residual-norm coverage), with stable ids, pivot image and site, depth profile (share of squared mass
  per site) and 90%-mass support. Rank / shipped / plotted are three separately displayed numbers.
- Axes are held fixed while moving through depth; raw values (no per-site rescaling, Procrustes, pooled PCA,
  SEBA, k-means, coupling, checkpoint or compare controls); on a scale change the axes reset and are relabelled;
  a rank-1 band is drawn as one axis.
- Caption: these are scaling functions of this operator; a complete band's functions are together a rotation of
  that band's eigenmodes (unit-norm, not the viewer's per-site normalised display), which three plotted axes are
  not.
- Checks: decoded browser values against the source arrays, ids and hover images, a deep link, on the deployed
  site; the Experiments entry follows `site/xsite/AUTHORING.md` (facts file, number check), with static charts
  and viewer deep links (the `emb-fig` schema has no wavelet kind).

## Decisions made on the user's behalf

- Q1 (same input, wavelets in place of harmonics) is the headline, because "comparable to double diffusion" was
  read as decoding the same supplied point. Alternative: headline Q3.
- Exact spectral bands, levels named by allowed amplification, instead of a dyadic time grid (which on `pr400`
  gives ranks >= 256, 16, 3, 0) or the paper's recursive QR.
- One atom selection per site by group norm (alternative: per activation dimension).
- Viewer on the public operator, not on `P8`/`pr400`.
- Laplacian pyramids not rerun: `lp`/`lpz` trailed `gh` at all 12 sites and four ranks last session. Those are
  residual Gaussian smoothers, not Coifman-Lafon multiscale geometric harmonics, which remain untested.

## Compute

Unit = site (12): one N = 8,000 float64 eigendecomposition and pivoted QRs on CPU (recomputed; the DDM stage
kept only 256 eigenvalues), a second-stage eigendecomposition per cell, patching on GPU. Pilot `b4` first,
recording peak memory and the time of each part; projected under $10 for 12 sites, about 20 minutes wall-clock.
Viewer solve: one 24,000-dimensional eigendecomposition, about $1.

## Changed after review

- Out-of-sample values divide by eigenvalues before the wavelet rotation; unit-norm columns throughout.
- The three questions are separated; the same-input second-stage comparison is primary and gained the
  best-n-harmonics and within-band-rotation controls.
- Second-stage operator defined (`K*/sigma_1`, leading eigenvector kept).
- Levels chosen from the spectrum by allowed amplification; the dyadic time grid dropped.
- GCV for adaptive selection replaced by an inner split; stability measured as input-to-output error
  amplification; round trips use one forward map on both sides and a rotation-invariant normalisation.
- Ring control no longer expects translates; the paper comparison is by projectors with its own tolerance.
- Viewer built on the public operator with complete bands, fixed axes through depth, no inherited overlays.

## Amendments (revision 3, second review round)

1. **Inner split (Q1).** The 64 input coordinates are fixed. On the inner 80%: build the second kernel (width
   frozen in absolute units at the stored `lh` GCV width), bands, QR bases, atom rankings and coefficients;
   Nystrom-extend to the 20%; choose n by pooled squared error. Then rebuild and re-rank on all of P8 keeping only
   the chosen n (the budget transfers, not the pivot identities) and score on `H`. `lead` gets the same
   split-chosen n; the stored GCV `lh` is kept as a separate mark. `ridge` is refit at this same width with its
   ridge parameter from the same split (the stored `krr` chose another width and stays the practical benchmark).
2. **Admissible bands (Q1).** `lead` is reported at every band boundary with A, rank and held-out scores.
   `wbest`, `ebest`, `rbest` are computed both unconstrained (down to the numerical floor: compression results)
   and restricted to prespecified admissible bands A <= 1e2 and A <= 1e4 (sigma_1/sigma_k), n chosen within the
   band. Wavelet/eigen equality is checked at every complete band. A is "the retained spectral reciprocal", not
   the gain of the coordinate-to-activation map.
3. **The dictionary.** For strictly increasing A_1 < A_2 < ..., `V_i = V(1/A_i)`; V_1 is the coarsest block and
   `W_i = V_i (-) V_{i-1}`. The partition, a numeric positive floor (lam >= 1e-10 lam_1) and the multiplicity
   tolerance (1e-9 relative) are frozen before held-out scores; fixed ranks are evaluation points only and never
   enter the partition. QR weights: in the inversion dictionary every band is filtered by one application of T
   (`diag(lam_B)`), and is described as such; in the viewer a level's scaling functions are filtered by
   `exp(-t_j mu)` with `t_j = ln 2 / mu_c` (`dw.band_basis(weights=...)`), i.e. heat bumps at the captioned time
   truncated to the modes that keep at least half their amplitude.
4. **Error amplification.** Two separately named quantities. *Coordinate -> activation*: delta a random
   direction drawn from centred probe-coordinate differences, rescaled to 1%, 5%, 20% of the probe RMS cloud
   radius, shared draws across decoders; response `(|dx_hat| / s_X) / (|delta| / s_Z)` with s_X, s_Z the probe RMS
   radii; in Q2 checked against the exact `delta B`. *Kernel row -> activation* (Q2): singular values of
   `p -> p U Lam^-1 Q B`, reported as algebraic sensitivity, not as attainable noise. Neither bounds the round
   trip. The whole measure is a synthetic stress test (no application error model is given).
5. **Viewer eigensolve.** The slice-mean-zero space has dimension 12 x 1,999 = 23,988, and at balance coupling
   the first temporal eigenvalue equals the first slice-mean-zero one, so an ambient solve can mix the two. Use
   the projected `spacetime.modes` solver with enough extra modes to bracket the highest band edge through any
   multiplicity block (or reduce to the mean-zero space before a dense solve); check slice means, residuals and
   the first excluded eigenvalue. The scale selector's heat time is labelled separately from the fixed depth
   coupling t*.
6. **Localisation.** n90 (centre-free) for all bases; neighbourhood mass about each function's own absolute peak
   for all bases; mass about the QR seed reported separately as a diagnostic of the pivot label; five random
   rotations per band.
7. The report presents the trade-off along A without recommending a unique cut-off unless one is clearly
   supported.
