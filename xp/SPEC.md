# Spec: three overnight experiments on the through-depth embedding (exploratory)

Owner request (2026-11-21). Three experiments, all on ResNet-20 at 32 px, all using the repo's joint
through-depth operator (`spacetime.py`) on a fixed probe set. Everything here is exploratory.

- **X1** Train the same architecture on different datasets; push the same probe through each; ask how much of
  the seed-to-seed agreement of the embedding survives a change of training data.
- **X3** Follow adversarial images through depth in the embedding of the clean probe.
- **X4** Interpolate between pairs of real activations at a site, run the rest of the network, and look for
  barriers at intermediate sites; measure finite-time Lyapunov exponents (FTLE); show the paths in the embedding.

(X2, single-class probes, was dropped by the owner. Divergence, Ricci curvature, material derivative: dropped.)

## 0. Shared objects

**Network.** `zoo.models.ResNet20` (ModelZoo), `num_classes` set per dataset. Sites (12): stem, b1..b9, pool,
logits, plus `pixels` (normalised input) where stated. Eval mode throughout (batch-norm uses running statistics).

**Probe and Gram.** A probe is 2,000 fixed images. At a site, activations are flattened, centred on the probe
mean and row-normalised; `C` is their cosine Gram. Only directions enter.

**Kernels (two, both from `invert/dm.py`, copied to `xp/dm.py` as a snapshot).**
- `knn15`: the published rule (union 15-NN graph on cosine, floor 1e-6, symmetric Sinkhorn). Primary for X1,
  since it is the rule behind the viewer and the known seed/architecture agreement.
- `ea100`: per-point Gaussian width calibrated to perplexity 100, symmetrised, Sinkhorn. Smooth in the
  activation, so a new point has a well-defined row. Primary for X3 and X4; second kernel for X1.

**Joint modes.** `dm.joint_modes`: relative site units (`c_l` = mean of the first 8 nontrivial `1 - lambda`),
coupling `t = t*` (balance rule), K = 32 modes stored, first 16 used unless stated. Coordinates of image `i` at
site `l` in mode `k` are `vecs[l, i, k]` (orthonormal over all L x N entries); no reweighting.

**Out-of-sample coordinates.** `dm.rows` gives a new point's row at each site from its cosines to the probe;
`dm.joint_forward` solves the L x L tridiagonal system per mode (frozen-probe Nystrom extension of the joint
operator). Fed a probe image's own masked rows it returns that image's coordinates (tested in `invert/`).
Support score of a new point at a site: its largest cosine to a probe image, as a percentile of the same
quantity for clean held-out images.

## X1. Different training sets, same probe

**Question.** For two ResNet-20s trained on datasets A and B, how similar are the through-depth embeddings of
the same probe, per site, against (ceiling) two seeds trained on the same data and (floor) untrained networks?

**Networks (16).**

| Name | Training data | Classes | Source |
|---|---|---|---|
| `c10-s0`, `c10-s1` | CIFAR-10 | 10 | existing zoo checkpoints (step 64,000) |
| `c10x-s0` | CIFAR-10 | 10 | trained here: control that this trainer reproduces a zoo network |
| `c100-s0`, `c100-s1` | CIFAR-100 | 100 | trained here |
| `in32-s0`, `in32-s1` | ImageNet 32x32 (1.28 M images) | 1000 | trained here |
| `svhn-s0`, `svhn-s1` | SVHN train (73 k) | 10 | trained here |
| `anim-s0`, `anim-s1` | CIFAR-10 animals only (bird, cat, deer, dog, frog, horse) | 6 | trained here |
| `veh-s0`, `veh-s1` | CIFAR-10 vehicles only (airplane, automobile, ship, truck) | 4 | trained here |
| `rand-s0` | CIFAR-10 images, labels shuffled once, no augmentation | 10 | trained here (trained floor) |
| `init-s0`, `init-s1` | none (zoo step 0) | 10 | existing |

Recipe for everything trained here (the zoo's ResNet-20 recipe): SGD momentum 0.9, lr 0.1, weight decay 1e-4,
batch 128, 64,000 steps, lr /10 at 50% and 75%, pad-4 random crop + horizontal flip (no flip for SVHN; no
augmentation for `rand`), inputs normalised by the training set's own channel mean and std. ImageNet-32:
250,000 steps (25 epochs), same schedule shape. Not bit-identical to the zoo trainer (GPU augmentation, no
determinism flags); `c10x-s0` measures whether that matters. Each network uses its own normalisation on every
probe: the object compared is the map from raw image to activations.

**Probes (4, 2,000 images each, none in any training set).** `c10` (the published probe, 200 per class),
`c100` (CIFAR-100 test, 20 per fine class), `svhn` (SVHN test, 200 per digit), `in32` (ImageNet-32 validation,
2 per class). 16 networks x 4 probes = 64 (network, probe) units; each gives 12 Grams, two kernels, two joint
solves.

**Measures, for every pair of networks on the same probe.** Same architecture, so site `l` of A is compared
with site `l` of B (the full site-by-site matrix is stored for measures 1 and 2).

1. **Neighbour overlap** (kernel-free): mean over probe images of `|N_A(i) & N_B(i)| / 15`, 15 nearest by
   cosine. Chance 15/1999.
2. **Linear CKA** of the two centred-cosine Grams. Standard baseline.
3. **Site-walk subspace overlap**: `|U_A^T U_B|_F^2 / r` for the first r = 8 nontrivial eigenvectors of each
   site's own walk. Chance about r/N.
4. **Joint-mode subspace overlap**: `|V_A^T V_B|_F^2 / K`, K = 16, V the (L*N) x K joint modes; and its split
   by site: the share of A's squared coordinates at site l reproduced by `z_B(l) Q`, with one orthogonal Q
   (polar factor of `V_B^T V_A`) for all sites, fitted on a random half of the probe images and scored on the
   other half. Also the K x K matrix of `|<u_k^A, u_j^B>|`.
5. **Class structure readout** (per network, not per pair): share of an image's 15 nearest probe neighbours
   with its own label, per site. Says whether a network separates the probe's classes at all.

Null for 1-4: the same statistic with B's image order permuted (one permutation shared by all sites).
Uncertainty: every network pair is a point; within a (training set, training set) cell there are 4 cross pairs
and 1 same-data pair (3 for CIFAR-10); bootstrap over probe images for single pairs.

**Controls with known answers.** `c10-s0` Grams and `knn15` modes must reproduce the published ones
(`data/grams/r20s0b`, `data/r20s0b/spacetime/st_64000_0.npz`). `c10x-s0` against `c10-s0` must look like
`c10-s0` against `c10-s1`. A network against itself gives 1. Permuted pairs give chance.

**Figures.**
- F1.1 Similarity against site, one panel per measure, one line per training-set pair (mean over network
  pairs, pairs shown as points), ceiling (same data, different seed) and floor (init, permuted) drawn; facet by
  probe. *"Networks trained on A and B agree as well as two seeds up to site X and lose this much after."*
- F1.2 Network-by-network similarity matrix at three depths (early, middle, late), per probe. *"Networks group
  by training set late but not early"* (or not).
- F1.3 Mode-by-mode overlap matrix for selected pairs. *"The first m joint modes are shared, the rest are not."*
- F1.4 Class readout against site, one line per network, per probe. *"A network trained on B separates A's
  classes this well."*
- F1.5 Side-by-side embeddings of one probe in two networks after the rotation Q (viewer-style scatter per site).
- F1.6 Training table: final train and test accuracy per network.

## X3. Adversarial images through depth

**Setup.** Attacked network `c10-s0`; transfer network `c10-s1`. Probe `c10` (P2). Source images: the first
1,000 CIFAR-10 test images outside the probe that both networks classify correctly. Attacks on `c10-s0`, L-inf,
PGD with 50 steps, step 2.5 eps / 50, random start, in [0,1] pixel space:
untargeted at eps in {1, 2, 4, 8}/255; targeted at eps = 8/255 (target = a random other class);
controls: random sign noise at eps = 8/255, and the clean images. Also a perturbation path `x + s * delta`
(delta = the untargeted eps-8 perturbation), s in {0, 1/8, ..., 1}.
Reference groups: naturally misclassified clean test images (outside the probe), and correctly classified
clean images of every class.

Every set is pushed through both networks; rows and joint coordinates in each network's own operator (`ea100`
primary, `knn15` as check).

**Measures, per site.**
1. Activation level (no embedding; the known baseline): cosine between clean and attacked centred activations;
   relative perturbation norm.
2. Displacement in the embedding: `|y_adv(l) - y_clean(l)|` in units of the median distance between two clean
   same-class held-out images at that site.
3. Class mass: row weight on probe images of the source class, of the final predicted class, and elsewhere.
   The **switch site** of an image is the first site from which the predicted class outweighs the source class
   at every later site. Distribution of switch sites for attacks, against naturally misclassified images.
4. Arrival: at pool and logits, distance from the attacked image to the centroid of the adversarial class's
   clean images, as a percentile of clean members' distances. Does the attack land where real members sit?
5. Support score per site (off-cloud travel).
6. Transfer: measures 2-5 in `c10-s1`, split by whether the attack transfers.

Controls: clean images displaced by 0; random noise at equal L-inf norm should barely move; `knn15` extension
should agree on class mass.

**Figures.** F3.1 Displacement against site, one line per attack and eps, noise control. F3.2 Class mass
against site (source / adversarial / other) for eps 8 untargeted, targeted, natural errors. F3.3 Histogram of
switch sites. F3.4 Heatmap site x s for the perturbation path. F3.5 Arrival percentile and support score.
F3.6 Transfer: displacement in `c10-s1`, transferred against not. F3.7 Embedding scatter per site with
clean-to-attacked arrows for a few images (with the images shown).

## X4. Interpolation between real activations, FTLE

Follows `CoherentModes/explore/cifar_coherent/site/harmonic.py` (path statistics) and extends it to every
downstream site and to the embedding.

**Setup.** Network `c10-s0`, probe `c10`. Pairs of held-out, correctly classified CIFAR-10 test images:
150 same-class and 150 different-class. Start sites `l0` in {pixels, stem, b3, b6, b8, b9}. Path
`x(s) = (1 - s) a + s b` between the two raw activations at `l0`, T = 256 steps (and T = 64 by subsampling),
pushed through the rest of the network. `b9` is an exact negative control (its suffix is average pooling and a
linear head, so the path is affine at pool and logits).

**Measures at every downstream site m > l0.**
1. Path statistics on the raw activation path at m (from `harmonic.path_stats`): sharpness
   `sigma = log(m_64 / m_256) / log 4` (0 for a jump, 1 for a smooth path), jump size
   `S = m_256 / |x_m(1) - x_m(0)|`, excess length `sum |dx| / |x(1) - x(0)| - 1`, transition width,
   location of the largest step. The owner's "barrier inside the model" is `sigma` well below 1 or a large S at
   an intermediate site, as opposed to at the logits only.
2. Exact speed `|J_{l0->m}(x(s)) (b - a)|` by forward-mode differentiation at every grid point.
3. ReLU pattern change: fraction of ReLU units between l0 and m whose sign differs from s = 0, and from the
   previous grid point (cell crossings).
4. **FTLE**: `Lambda_{l0->m}(x) = log sigma_max(J_{l0->m}(x))`, by power iteration (JVP then VJP, 20
   iterations, convergence recorded). Two versions: raw, and angular (Jacobian of the centred, normalised map
   between the two unit spheres, restricted to tangent directions, which is the geometry the embedding uses).
   Computed (a) along the paths (T = 64) and (b) at all 2,000 probe images for l0 in {pixels, stem, b3, b6}
   and m in {next site, b9, logits}. Also the alignment of the path direction with the top singular direction
   (`|J v| / (sigma_max |v|)`).
5. Output along the path: logits, predicted class, margin; the s at which the prediction changes.
6. Embedding: coordinates of each path point at sites m >= l0 (T = 64). For l0 after pixels the point does not
   exist at earlier sites; the extension is solved on the truncated chain of sites l0..logits with a free end
   at l0. **Control with a known answer:** real held-out images given the same truncated treatment, compared
   with their full-chain coordinates; reported per l0, and the truncated coordinates are used only where this
   error is small. In the embedding: path length over chord length, largest step, support score along the path.

**Questions the figures answer.**
- F4.1 `sigma` and S against downstream site, one line per start site, same-class and different-class pairs;
  b9 control. *"Paths started at l0 show jumps from site m on / only at the logits."*
- F4.2 Heatmap s x downstream site of speed (mean over pairs, and three single examples), per start site.
- F4.3 FTLE along the path against s, aligned on the output switch point; FTLE peak location against speed
  peak location. *"Stretching peaks where the path moves fastest / where the class changes."*
- F4.4 FTLE at probe images: embedding scatter coloured by FTLE; FTLE against output margin. *"High FTLE marks
  images near class boundaries"* (or not).
- F4.5 Paths drawn in the embedding at several downstream sites, coloured by s.
- F4.6 ReLU pattern change against s and site.
- F4.7 Truncated-chain control error.

## Decisions made on the owner's behalf

- Same architecture and resolution for X1; no pretrained 224 px ImageNet network (it would confound resolution
  and architecture).
- Two seeds per new training set, so that cross-data agreement can be read against same-data agreement.
- `knn15` primary for X1, `ea100` primary for X3/X4. Alternative for X3/X4, adding the new points to the probe
  and refitting: rejected, because path points and clean/attacked twins are each other's nearest neighbours and
  would distort the graph.
- Perplexity 100 fixed (middle of the inversion agent's 30/100/300); not tuned.
- Frozen-probe extension, not refit, for all new points.
- FTLE "time" is the number of sites between l0 and m; values are reported as `log sigma_max` without dividing.

## Compute

Modal app `crw-depth-xp`, volume `crw-depth-xp-results`; read-only inputs from `cm-cifar-coherent` and
`crw-modelzoo-train`. Prices from modal.com/pricing on 2026-11-21: L4 $0.000222/s, A10 $0.000306/s,
CPU $0.0000131/core-s, memory $0.00000222/GiB-s.

| Stage | Units | Hardware | Each | Cost |
|---|---|---|---|---|
| data | 1 | CPU | 20 min | < $0.5 |
| train (64k steps) | 11 | A10 | about 45 min | about $9 |
| train (ImageNet-32, 250k steps) | 2 | A10 | about 3 h | about $7 |
| extract + Grams | 64 | L4 | 2 min | about $2 |
| kernels + joint modes | 128 | 8 cores, 16 GiB | 5 min | about $6 |
| compare | 4 | 8 cores, 32 GiB | 15 min | < $1 |
| X3 | 2 | L4 + CPU | 15 min | < $1 |
| X4 | about 8 | L4 + CPU | 20 min | about $3 |

Projected total about $30, wall-clock about 5 hours, limited by ImageNet-32 training.

## Revisions after review (GPT-6.2 Sol, 2026-11-21); these override the text above

**X1.**
- The comparison is between **training pipelines** (images, labels, number of classes, normalisation,
  augmentation, exposure), not an isolated effect of the images.
- **Primary joint solve: trunk only, stem..b9 (10 sites).** The head changes with the dataset (4 to 1000
  outputs) and the logits walk enters every slice of a joint solve through the coupling. The full 12-site
  solve is stored as a sensitivity check and for the published-modes control.
- Reference for a cross pair (A, B) is each dataset's own seed pair, shown separately, not a single ceiling.
  The unpermuted init pair is a separate line from the permutation null. A second null permutes images
  **within class**, to tell agreement on individual images from agreement on class structure.
- Mode-by-mode overlap is reported as principal angles and with eigengaps; individual near-degenerate modes
  are not interpreted. Probe-bootstrap intervals are conditional on the fitted networks.
- Measure 2 is linear CKA on centred, row-normalised features. `rand-s0` is a different training regime (no
  augmentation, labels shuffled), a trained reference rather than a floor isolating the role of labels.

**X3.**
- A joint coordinate at site l uses the rows of the whole trajectory, so **joint displacement cannot date an
  effect**. Claims about where an attack takes hold use site-local readouts only: raw activation measures,
  site-local class mass (the point's kernel row at that site), and a site-only embedding (Nystrom extension of
  that site's own walk modes). Joint coordinates are used for pictures and for the arrival measure.
- Check: replacing only the logits row of a clean image by its attacked row must leave every site-local
  readout at earlier sites unchanged, and the size of the change in the joint coordinates at earlier sites is
  reported. Per-mode extension gain (smallest singular value of the L x L system) is recorded.
- Cohorts, fixed in advance: every attempt enters success rates and displacement; class-mass, switch and
  arrival figures are split into success / failure. Targeted success = prediction equals the requested target.
  Transfer: "any flip" and "same destination" separately. Switch site = first site from which the destination
  class outweighs the source class at every later site (destination = final prediction of that network);
  "destination is the top class" is reported as a second definition; images with no persistent switch are
  counted. Natural errors are summarised overall and matched by (source, destination) where counts allow.
  Centroids for the arrival measure come from probe images only (the scored images are never in the probe).
  Random noise is a measured baseline at each eps, not a control with a known answer.

**X4.**
- The network is continuous and piecewise affine, so nothing jumps; the outcome is **localised stretching**:
  the share of a path's arc length inside its fastest window of width 1/16 in s (1/16 for a uniform path, 1
  for a step), with peak speed and chord length reported separately. `sigma` and S are kept as secondary
  (comparability with `harmonic.py`), with a T = 1024 refinement of 50 paths. Same- and different-class pairs
  are compared within strata of start-site endpoint distance, and the overlap is reported.
- **Suffix operators replace the truncated-chain extension.** For each start site l0, a separate joint operator
  is fitted on the clean probe over sites l0..logits; the clean cloud and the paths are both expressed in that
  basis ("suffix embedding", not the viewer's coordinates). Checks: masked replay of probe images, and
  held-out endpoints (real images) recovered identically whether processed as images or as path endpoints.
  Site-local coordinates are reported alongside.
- Angular gain is a **fixed-radius** derivative: hold the centred radius of the input fixed, lift a unit
  tangent of the input sphere to activation units, apply the suffix Jacobian, then the derivative of the output
  normalisation. Points with small centred norm are flagged. The quantity is called log gain
  (`log sigma_max`); "FTLE" is log gain divided by the number of sites crossed, shown where depths are
  compared. Power iteration is checked against an explicit SVD on a small case (pool start; b9 start).
- F4.3 gets a panel for paths with no output switch; paths with several switches are aligned on the first.
  F4.4 tests association with output margin, not distance to a boundary.
