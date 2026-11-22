# Diffusion wavelets as the inverse: what was built and what it measured (Q1-Q3)

2026-11-22. Exploratory: one network (ResNet-20 seed 0, CIFAR-10, step 64,000), one probe draw (`P8`, 8,000 test
images), 2,000 held-out images `H`, first-stage kernel `pr400`, joint modes `P8__pr400__L12`, 12 sites. Spec:
`wavelets/SPEC_DW.md` (revision 3; its "Amendments" win over the body). Code: `wavelets/dw.py` (bands, pivoted-QR
bases, extension), `invert/stage_dw.py` (measurement), `wavelets/test_dw.py` (11 tests), `invert/dw_collect.py`.
Numbers: `data/invert/dw_results.json` (summaries), `data/invert/dw/P8__pr400__L12/<site>.npz` (per-image scores of
every final and curve point, per-atom localisation). The viewer part of the spec is not covered here.

Definitions. "Cosine": between the true centred activation and the decode, mean over `H`. "Agreement": share of `H`
whose top-1 class is unchanged when the decode is patched in at the site. "Paired": mean over `H` of the per-image
difference between two decoders, with its standard error (SE). "A": the allowed spectral reciprocal; a band V(1/A)
keeps the eigenvectors with lam >= lam_1 / A (`T = K*/sigma_1` for the second stage; the walk `P_l`, top eigenvalue 1,
for the site operator). The largest reciprocal actually retained, sigma_1 / sigma_k, is reported separately (`amp` in
the spectrum tables). Neither is the gain of the coordinate-to-activation map. Bands use A in {1.5, 2, 4, ..., 2^33}
plus {1e2, 1e4} plus the floor lam >= 1e-10 lam_1, fixed before any held-out score was read. "Amplification
(coordinate)": `(|x_hat(z + delta) - x_hat(z)| / s_X) / (|delta| / s_Z)` with delta a random probe-coordinate
difference rescaled to 5% of the probe RMS cloud radius s_Z, s_X the probe RMS radius of the activations; median over
`H` (1% and 20% are in the JSON). It is a synthetic stress test, not an error model. "Round trip" (Q2): the band
coordinates of the decode, through the same forward map as the request, against the requested ones; total squared
error over total coordinate variance (returning zero scores 1.00, another image's coordinates 1.95-2.07). Conv sites
are b1-b8. Maxima along curves are maxima on the evaluated grid.

## What was built

- **Dictionary.** For each operator, the eigenvectors are cut into bands by A; every band gets an orthonormal basis
  by column-pivoted QR of its band-filtered deltas (one application of T), so a complete band's wavelets are a
  rotation of its eigenvectors. Out-of-sample values divide by the eigenvalues in the eigenbasis, then rotate.
  This is an exact spectral-band construction, not Coifman-Maggioni's recursive compressed QR.
- **Q1 (primary): same 64 input coordinates, different function basis.** Cells `joint|64` (joint coordinates of the
  P8 / `pr400` inversion experiment, not the public viewer's P2 operator) and `source|64` (one kernel step of the
  slices; held-out values from the stored cosines, as in the DDM run). Second stage: the Gaussian Gram of the DDM lift
  at the width its GCV picked. Decoders: `lead` (leading n harmonics; also evaluated at every band boundary), `ebest` /
  `wbest` / `rbest` (best n eigenvectors / QR wavelets / atoms of one within-band random rotation, seed 0, by
  coefficient energy), each with separately tuned arms restricted to A <= 1e2 and A <= 1e4; `ridge` at the same
  width; stored `krr`; stored GCV lift (`lead@gcv`). n and the ridge parameter are chosen on an inner 80/20 split of
  P8 that rebuilds kernel, bands, bases, rankings and coefficients on the 80% (targets centred on the 80% mean); only
  the budget and the ridge parameter transfer. Curves along n on `H` are descriptive.
- **Q2: band(A) of the site operator.** All of V(1/A), decoded linearly from the site's own Nystrom coordinates.
  The input therefore grows with A (1 to 3,517 coordinates), unlike Q1.
- **Q3: compression.** Probe energy of the first n / best n eigenvectors, best n wavelets, best n atoms of five
  within-band random rotations; and the held-out decode from those n features (random rotation: seed 0 only),
  computed from the true activation's kernel row. That is a new encoding: it cannot be computed from a point given in
  the first n eigen-coordinates.

## Q1. Decoding a given 64-coordinate point: the basis changes little; ridge is ahead

**1a. Best-n wavelets are within a few thousandths of the leading-n harmonics.** Fig. 1 sentence: *on the same 64
input coordinates, at each decoder's own split-selected budget, the best-n wavelets are 0.004 below to 0.001 above the
leading-n harmonics in cosine (`joint|64` -0.0037 to +0.0009; `source|64` -0.0006 to +0.0005); ridge at the same
width is 0.001-0.025 above the leading-n harmonics (and 0.001-0.024 above the wavelets).* The differences are small
but not all zero: the largest, `joint|64` logits, is -0.0037 +- 0.0008 (paired SE). Agreement differs by -0.018 to
+0.003 (`joint|64`) and -0.007 to +0.011 (`source|64`), within two paired SEs in 22 of 24 site-cell comparisons; the
exceptions are stem `joint|64` (-0.018 +- 0.005) and b5 `source|64` (+0.011 +- 0.004). `ebest` and `rbest` are within
0.003 cosine of `lead` everywhere. Ridge is above `lead` at every site, paired SE <= 0.0006, most in b4-b8
(`joint|64` +0.018 to +0.025). The stored `krr` is within about 0.004 of `ridge` in cosine. `wbest` and `lead` chose
the same budget in 6 of the 24 comparisons.

| `joint\|64`, cosine / agreement | stem | b2 | b4 | b6 | b8 | b9 | pool | logits |
|---|---|---|---|---|---|---|---|---|
| lead (split n) | 0.615 / 0.15 | 0.418 / 0.14 | 0.495 / 0.35 | 0.474 / 0.77 | 0.463 / 0.95 | 0.694 / 0.95 | 0.931 / 0.92 | 0.974 / 0.94 |
| wbest (split n) | 0.614 / 0.13 | 0.418 / 0.14 | 0.495 / 0.34 | 0.474 / 0.76 | 0.464 / 0.95 | 0.694 / 0.95 | 0.930 / 0.92 | 0.970 / 0.93 |
| ridge (same width) | 0.630 / 0.15 | 0.430 / 0.16 | 0.514 / 0.40 | 0.495 / 0.79 | 0.488 / 0.95 | 0.710 / 0.96 | 0.937 / 0.92 | 0.976 / 0.95 |
| krr (stored) | 0.632 / 0.21 | 0.431 / 0.15 | 0.515 / 0.39 | 0.495 / 0.77 | 0.489 / 0.95 | 0.706 / 0.96 | 0.938 / 0.96 | 0.978 / 0.97 |

Why the basis changes so little was not measured (the overlap of the selected atom sets was not computed). One
possibility: rotating within a band keeps the band's energy, and energy ranking may reselect roughly the leading
set; on the separate Q3 site operator, first- and best-eigenvector energy differ by at most 0.0057 over this budget
range (wavelet budgets 65-1,014).

**1b. The admissible bands are almost empty in stem-b8, because sigma_1 dwarfs the rest.** The second-stage Gram is not
mean-removed (as in the DDM lift), so its leading eigenvector is close to constant and carries almost none of the
centred activation; sigma_2 / sigma_1 is 6.8e-6 to 2.0e-3 in stem-b8 on `joint|64` (8.1e-5 to 1.7e-2 on `source|64`)
and 0.05-0.23 (`joint|64`) / 0.011-0.069 (`source|64`) at b9-logits. A <= 1e2 then keeps only the leading harmonic at
all nine `joint|64` sites in stem-b8 and eight of nine `source|64` ones (source b8 keeps 3; its restricted wavelet
decode reaches 0.195 cosine). A <= 1e4 keeps 62-64 atoms at `joint|64` b4-b6 and b8 (cosine 0.43-0.49, 0.01-0.03
below `lead`) and only the leading harmonic at stem-b3 and b7. At b9-logits the admissible bands are usable:
on `joint|64`, A <= 1e2 admits 14 / 9 / 11 atoms at b9 / pool / logits; `wbest` selects 14 / 8 / 11 and reaches cosine
0.594 / 0.828 / 0.916. Its agreement is higher than `lead`'s at b9 (0.971 against 0.952) and logits (0.976 against
0.940) and lower at pool (0.901 against 0.916). At A <= 1e4, pool reaches 0.892 cosine / 0.977 agreement against
0.931 / 0.916 for `lead`. The
split-chosen `lead` n lies in bands with A up to 1.6e4 at b9 and 4e6 to 1e10 elsewhere (`joint|64`; `source|64` 8e6 to
1e10). The floor binds in a few cells: `joint|64` b3 uses its whole refitted dictionary (195 of 195; the split asked
for 197), and the `source|64` budgets at stem, b3 and b7 enter the last band (295 of 296, 254 of 266, 283 of 284).
Measured against sigma_1, a cut-off with small A in stem-b8 keeps almost nothing. Measured against sigma_2 it
would mean something else; that was not prespecified and was not run.

**1c. Reconstruction and function disagree along the cut-off, and the function end amplifies more** (Fig. 2 sentence).
Along `lead` on `joint|64`, the lift with the highest agreement amplifies coordinate error 4.4-4.9 times more than
the lift with the highest cosine at b4-b6 (2.5-2.8 against 0.57, activation spread per coordinate spread), 1.4-2.0
times at stem-b2 and 1.7 at b7; at b3 the two maxima are at nearby ranks (188 and 195) with essentially the same
amplification (ratio 1.004). From b8 on, the agreement-best lift has lower rank and amplifies less (0.85 times at b8,
0.07-0.27 times at b9-logits). Ridge along its parameter shows the same: 3.8-4.0 times at b4-b6. On `source|64` the
ratio is 1.5-2.8 at b1, b2 and b4-b6; at stem, b3 and b7 the two maxima coincide. Some agreement maxima are ties
(e.g. `joint|64` pool at n = 19 and 138); the lowest rank is used.

**1d. The double-diffusion lift is reproduced at shared ranks** (Fig. 6). At the 56 site-rank points that the `lead`
curve shares with the stored DDM curve on `joint|64`, mean cosine differs by less than 5.3e-11 and mean agreement is
identical. Both show the separation of the cosine and agreement peaks, on different grids: b4 cosine peaks at
n = 243 (0.496) and agreement at the full dictionary, n = 4,051 (0.715); the stored DDM curve gave 0.496 at 256 and
0.712 at 4,096. The dictionary stops at the 1e-10 floor, so the DDM's full-rank point (n = 8,000, agreement 0.5555 at
b4) is outside it.

## Q2. The spectral cut-off with a bounded spectral reciprocal, on the site's own coordinates

Fig. 3 sentence: **with A <= 2 the truncation keeps 2-4 functions in stem-b8 (9-12 at b9-logits) and reaches cosine
0.20-0.47 (0.60 / 0.86 / 0.93 at b9 / pool / logits); its round trip is 0.68-1.20 in b1-b8 and 0.26 at stem, where
returning zero scores 1.00 (0.13 / 0.06 / 0.07 at b9 / pool / logits).** In conv sites the round trip at A = 2 beats
returning zero at b1-b6 (narrowly at b5-b6: 0.95, 0.99) and is worse at b7-b8 (1.14, 1.20). At b1-b4 a smaller
nonempty band (A = 1.5, one function) also exists. The `t1` diagnostic (output without 1/lambda against the exact
target) is 0.04-0.16 at A = 2.

Along A, cosine peaks at the A = 32 partition boundary in b1-b8 (0.54-0.66, with 1,296-1,508 functions); counting
the fixed evaluation rank 1,024, that rank is marginally higher at b7 and b8 (0.6562 against 0.6560; 0.6318 against
0.6313), and it is the maximum at stem (0.72) and b9 (0.81); pool peaks at A = 32 (0.97), logits at A = 16 (0.996).
Agreement in b3-b8 rises to A = 32-64 (0.73-0.91), and at b1-b2 to A = 100 (0.51, 0.61). At A = 1e4 cosine falls to
0.14-0.27 at stem-b9 but is still 0.57 at pool and 0.73 at logits. Round trips are poor around the
reconstruction-optimal bands (25-80 at A = 32 in b1-b8; 1.7-3.9 at A = 8), but not monotonic in A: at b4 the round
trip goes 0.74, 35.4 and 1.11 at A = 2, 32 and 8,192, and at b7 its minimum is at A = 4,096 (0.95, against 0.96 at
A = 4). Coordinate amplification grows slowly (median 0.2-0.3 at A = 2 to 0.7-0.8 at A = 32 in conv sites); the
kernel-row sensitivity (largest singular value of p -> p U Lam^-1 B, scaled by probe radii) is 0.96-2.55 at A <= 2,
2.2-6.0 at A = 32, 170-310 at A = 1e4 and 2e3-4e4 at the floor: an algebraic sensitivity, not attainable noise.

This answers the "interpretable spectral truncation" request in a narrow sense: each cut-off is a band with a stated
maximum reciprocal, but the three criteria (reconstruction, function, round trip) do not select a common cut-off
across sites, so no single cut-off is recommended. Comparison, not the same input: the DDM's best site-coordinate
decoder by cosine at K = 256 (krr) had 0.49-0.64 cosine and 0.20-0.90 agreement in b1-b8; Q2 at A = 32 uses five to
six times as many coordinates.

## Q3. Compression: at n = 256, activations are not sparser in these wavelets

Fig. 4 sentence: **at n = 256 the best eigenvectors capture more probe energy than the best QR wavelets at 11 of 12
sites (by up to 0.028, at b7; at logits the wavelets lead by 0.00003), and the wavelets exceed the mean of five
within-band random rotations by only 0.0004-0.0063.** Probe energy share of the best n atoms, site operator:

| b4 | n = 16 | 64 | 256 | 1,024 |
|---|---|---|---|---|
| first n eigenvectors | 0.175 | 0.266 | 0.388 | 0.549 |
| best n eigenvectors | 0.175 | 0.267 | 0.388 | 0.549 |
| best n QR wavelets | 0.175 | 0.262 | 0.366 | 0.543 |
| best n random-rotation atoms (5 rotations) | 0.175 | 0.260 | 0.363 | 0.541 |

The eigenvector advantage at n = 256 is 0.022-0.028 in b4-b8; the standard deviation over the five rotations is at most
0.002. It is not uniform across budgets: at very small n the wavelets can lead (stem, n = 14: by 0.004 over the best
eigenvectors) or trail random rotations (logits, n = 3: wavelets 0.337, random 0.402, eigenvectors 0.528). At n = 256
the energy gap between the first and best n eigenvectors is at most 0.00112 (over the whole grid the gap reaches 0.025, logits n = 5).
Held-out decode from the n features at n = 256: wavelet features trail the best eigenvectors in cosine at every site
(by 0.0005-0.020; agreement 0.003 higher to 0.037 lower), and differ from the one decoded random rotation by -0.0003
(logits) to +0.0045 (b1). 21-26% of the centred activation energy in conv sites (14% at b9, 2-3% at pool / logits)
lies outside the positive spectral sector the dictionary spans (the walk has 4,482-6,702 negative eigenvalues).

## Localisation (Fig. 5)

The QR atoms are only weakly localised. Median number of probe points holding 90% of an atom's squared mass, over all
atoms: 2,476-3,300 of 8,000 for QR wavelets against 2,674-3,471 for eigenvectors and 2,682-3,472 for random rotations
(site operator, `joint|64`, `source|64`; mean over five rotations). Mass on the atom's own peak and its 400 nearest
other probe points (5% of the points): QR 0.12-0.31 median, eigenvectors 0.05-0.11, random 0.05-0.10. In the site
operator's A = 16 band, QR 0.30-0.45 against 0.06-0.08. Mass about the QR seed has the same median as mass about the
peak at every site, and the two differ by at most 1e-6 for about 99.0-99.9% of atoms: the seed labels a neighbourhood
of similar mass, though it need not be the peak itself.

## Controls (all 12 sites, gated in the run; `failed` is empty everywhere)

- Patching the true activation: agreement 1, KL < 2e-11.
- Stored DDM numbers reproduced, as the largest per-image difference between this run's float64 cosines and the stored
  scores: GCV lift (`lead@gcv`, including the four `source|64` cells, stem, b7, pool and logits, whose stored rank lies
  above the 1e-10 floor) 1.4e-7, krr 3.0e-8, `site|K` linear (K = 4, 16, 64, 256) via the band decode 1.1e-7; stored
  one-step sources to 4.6e-10. (Differencing the exported float32 arrays gives slightly different maxima, consistent
  with rounding.)
- Span invariance: wavelets against eigenvectors over every complete band, per image, at most 7e-15 relative (Q1 and
  Q2, QR and random rotations); full-band second stage against `LatentHarmonics.weights` 2.2e-14; finite differences
  against the exact `delta B` 2.1e-12.
- Spec controls 1-7 and the stage itself on synthetic inputs: `wavelets/test_dw.py`, 11 tests, about 6 s on CPU.

## Fixes to `wavelets/dw.py` made by this task

- `cut_index` / `levels` gained `edge=True` (opt-in, default unchanged): multiplicity tolerance relative to the
  eigenvalue at the edge. With the default (relative to lam_1) and a 1e-10 floor, every eigenvalue below about 1e-9
  merged into one block, so the tail bands were misnamed. `stage_dw` uses `edge=True`.
- `cm_tree` gained `symm=True` (now the default): `T_{j+1} = Op Op^T`, the reference `DWPTree.m` symmetric branch,
  instead of squaring the compressed operator. Only `test_dw.py` uses it.

## Decisions made without guidance

- A grid: dyadic from 2 plus 1.5, 1e2, 1e4; floor 1e-10 relative to the top eigenvalue; a block straddling the floor is
  kept whole.
- Inner split: 80/20 of all 8,000 probe images (seed 0), not `stage_ddm`'s 4,000-image subsample. n is chosen among up
  to 120 log-spaced candidates plus band boundaries (after rounding and deduplication); the ridge parameter is chosen
  on the same split. `lead` gets its own split-chosen n; `lead` at the wavelet n is also reported.
- `rbest` decodes use one rotation (seed 0); energy and localisation use five.
- The stored `krr` selection is reused (parity-checked), not searched again.
- Q2 / Q3 inputs come from kernel rows computed from vectors. Rows from the stored float32 cosines (differing by at
  most 1.5e-7) are used for the parity controls and for the `source|64` query coordinates (as in the DDM run).
  Centring follows `stage_ddm` (float32 subtraction).
- Near-probe flags: kernel row with more than half its weight on one probe image, at this site (0-17 images) and at any
  site (25); amplification is also reported without them (`*_far`, `*_far_any` in the JSON).

## Limits

- One network, one checkpoint, one probe draw; uncertainty is over held-out images only.
- Q1 finals are held out (n and ridge chosen on the probe split); curves and their maxima are read on `H` and are
  descriptive.
- A is measured against sigma_1 of a non-mean-removed Gram (1b); the admissible-band results depend on that convention.
- Q2's input size grows with A, so Q2 and Q1 are not the same comparison; Q3 is a different encoding.
- Atom ranking is by group norm over all activation dimensions, one ranking per site.
- Agreement is top-1 only; KL is in the JSON.

## Status, spend, reproduce

Finished 2026-11-22. A pilot (b4, stem), then the other 10 sites; every site took 28-46 s in-stage on an A100 (468
in-stage worker-seconds in total), peak GPU allocation 10.7 GiB. Spend: a rough allowance of about $1-2 from
container time at list rates, not a measured cost (the Modal dashboard has the figure). The app
`crw-depth-invert-dwinv` was left deployed with no job running.

```
uv run --with numpy --with scipy --with torch --with pytest python -m pytest -q wavelets/test_dw.py
INVERT_APP=crw-depth-invert-dwinv uv run --with modal python invert/modal_invert.py deploy
INVERT_APP=crw-depth-invert-dwinv uv run --with modal python invert/modal_invert.py launch dw '{"units":[{"pset":"P8","kname":"pr400","site":"b4"}, ...]}'
uv run --with modal python invert/modal_invert.py fetch dw/P8__pr400__L12 data/invert/dw/P8__pr400__L12
uv run --with numpy --with scipy --with torch python invert/dw_collect.py
```
