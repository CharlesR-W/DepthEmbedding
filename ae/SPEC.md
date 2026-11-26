# AE: the same ResNet-20 trained to reconstruct instead of classify (2026-11-26)

Exploratory. Request: "train one or two autoencoders on CIFAR-10 and add them to the embedding viewer; I want to see
how the different objective causes divergence in their representations in later layers. You can choose what kind."

## Question and measured quantity

At which sites does a network trained to reconstruct its input stop representing the 2,000 probe images the way a
classifier does? Measured per site l (stem, b1..b9, pool, last), for pairs of networks A, B on the CIFAR-10 probe:

- neighbour overlap: mean over images of |N_A,l(i) ∩ N_B,l(i)| / 15 (15 nearest by centred cosine; chance 15/1999);
- linear CKA of the double-centred cosine Grams;
- class readout of one network: share of an image's 15 neighbours with its own label.

All three are computed by the existing `xp/stage_compare.py` (unchanged), with its two permutation nulls. The viewer
shows the through-depth modes of each network (kNN-15 walk, full 12-site chain, relative site units, t = t*), fitted
by the existing `xp/modal_xp.py:embed` (unchanged estimator).

## Design

Reuse the X1 pipeline (`xp/common.py`, `xp/modal_train.py`, `xp/modal_xp.py`), so that a classifier trained by the
same code exists (`c10x-s0`, 91.4% test) and only the loss differs.

| | classifier `c10x-s0` (exists) | `ae10-s{0,1}` | `ae64-s{0,1}` |
|---|---|---|---|
| network up to the last site | zoo ResNet-20, fc 64 -> 10 | identical | identical but fc 64 -> 64 |
| after the last site | cross-entropy | decoder, MSE to the input | decoder, MSE to the input |
| init | seed 0 | same code and seed, so same trunk and fc init | same trunk init (fc replaced after) |
| data, order, augmentation | CIFAR-10, crop + flip, (seed, step) | identical | identical |
| optimiser | SGD 0.9, lr 0.1, wd 1e-4, batch 128, 64k steps, /10 at 50%, 75% | identical | identical |

Decoder (not a measured site): Linear(d, 256·4·4), BN, ReLU; three times [nearest upsample x2, conv3x3, BN, ReLU]
with 128, 64, 32 channels; conv3x3 to 3 channels. Target is the augmented, normalised input the encoder saw.

Sites: stem, b1..b9, pool (global average of b9, 64-d), code (the fc output, 10-d or 64-d; the classifier's logits
site). The operator only sees each site's 2000 x 2000 cosine Gram, so the differing width of the last site is allowed.

- Varies: the objective (classify / reconstruct), the code width (10 / 64), the seed (0 / 1).
- Fixed: architecture to the last site, data, augmentation, optimiser, schedule, probe, kernel, chain.
- Replicate: a seed. Two per autoencoder; classifiers have c10-s0, c10-s1 (zoo trainer) and c10x-s0 (this trainer).
- Compared networks (9): c10-s0, c10-s1, c10x-s0, init-s0, init-s1, ae10-s0, ae10-s1, ae64-s0, ae64-s1.

## Controls with a known answer

1. The new compare run, restricted to the five old networks, must reproduce `compare/c10.npz` (nn, cka, readout).
2. Each autoencoder's test PSNR must exceed that of rank-d PCA of the training images (d = code width).
3. Seed floor: classifier-vs-classifier and autoencoder-vs-autoencoder similarity bound what "the same" looks like;
   trained-vs-init bounds what "unrelated training" looks like.

## Figures (Experiments tab entry, group "probing")

1. Lead embedding figure: classifier and ae10 side by side at a late site, coloured by class. "By b9 the classifier
   has pulled the classes apart and the autoencoder has not."
2. Lines: x = site, y = neighbour overlap with a classifier; one line per pair group (classifier seeds;
   autoencoder vs classifier, same init and different init; autoencoder seeds; trained vs init), band = range over
   pairs. "The autoencoder agrees with the classifier as well as another classifier does up to site X, then falls."
3. Same with CKA.
4. Lines: x = site, y = class readout per network. "Class neighbourhoods grow with depth only under the label loss."
5. Reconstructions of a few probe images (ae10, ae64) next to the originals, with test PSNR and the PCA bound.

Viewer: a new group "Other objectives" listing the four autoencoders (final checkpoint, modes only, like X1).

## Decisions made on the user's behalf

- Plain bottleneck autoencoder on the classifier's exact trunk, rather than a denoising, masked or variational one:
  it isolates the loss. With a global-average-pool bottleneck, masking or noise would change little, since the code
  already cannot carry detail.
- The second autoencoder varies the code width (64) rather than the corruption, because the 10-d code is the obvious
  confound: is the divergence from the loss or from the tight bottleneck?
- Same SGD recipe as the classifier rather than Adam, to avoid an optimiser confound; abort if the loss is not finite.
- Final checkpoint only; no PCA / UMAP baselines for these networks (as for X1).

## Work and cost

Train 4 units (A10G, about 12 min each), embed 4 units (about 2 min each), compare 1 unit (16 CPU, about 3 min).
Projected spend about $1.5, wall-clock about 30 min. New files on `crw-depth-xp-results`: `nets/ae*`, `embed/ae*`,
`compare_ae/`. Nothing existing is overwritten.

## Revisions after review (GPT-6.2 Astra, 2026-11-26)

1. **Matched seed pair.** `c10-s0` and `c10x-s0` are two trainers at one seed, not a seed pair. Train `c10x-s1`
   (classifier, this trainer, seed 1). Pair groups, plotted separately: classifier seeds (`c10x-s0`, `c10x-s1`);
   autoencoder seeds; autoencoder vs classifier at the same seed (2 pairs) and at different seeds (2 pairs);
   trained vs init as a reference, not a bound. Compared networks are now 10. Classifier pairs already fall with depth
   (X1: neighbour overlap 75% at stem, 43% at b9), so the claim is about the *gap* between the autoencoder-classifier
   curve and the classifier-seed curve at the prespecified sites stem, b3, b6, b9 and pool, not about a first crossing.
   Two seeds: descriptive only.
2. **Narrower claim.** The intervention is class prediction versus reconstruction of the augmented crop through a
   learned decoder behind a global-average-pool bottleneck; "only the loss differs" is shorthand for that. `ae64` is a
   width sensitivity check reported separately, not a test of "loss or bottleneck". The sentence in the decisions
   about masking changing little is withdrawn (not established).
3. **Headline sites.** Headline numbers are for stem..b9 and pool, which are the same activations in both networks. The
   last site is shown apart and labelled "10 logits" versus "10 (or 64) code coordinates": the decoder's first linear
   layer can undo any invertible change of the code, so its cosine geometry is not pinned by the objective.
4. **Lead with measurements.** The entry leads with same-site neighbour overlap and the two neighbour readouts, all on
   the original activations. Viewer panels are illustrations: exact site matches (`frac`), fixed reference axes, at
   stem and b9. The stem..b9 trunk-only operators (already fitted by `embed`) check that the picture does not need the
   output sites; the numeric alignment score is `stage_compare`'s held-out rotation cosine, not the viewer's residual.
5. **Training gates.** Before the full runs: `ae_check` (fit one fixed batch of 128; inputs have unit variance, so
   1.0 is the mean-image level) and a learning-rate pilot on `ae10-s0`: 8,000 steps with the same schedule shape, 5,000
   training images held out, lr in {0.1, 0.03, 0.01}. Rule fixed in advance: keep 0.1 unless another rate's held-out MSE
   is more than 5% lower, then use the lowest; the chosen rate is used for all four autoencoders and reported. Final
   record per autoencoder: test PSNR, PSNR of the mean training image, PSNR of rank-d PCA fitted on the training
   images (a baseline reported either way, not a gate: PCA sees pixel positions, the pooled code does not), standard
   deviation of each code coordinate over test images, and eval-mode batch independence.
6. **Initialisation.** `ae_check` compares step-zero tensors of the autoencoder's encoder with `build(10)` under the
   same seed, and with the zoo's step-0 checkpoint. `c10x-s0`'s own step-0 state was not saved, so for it the claim is
   "same initialisation code and seed".
7. **What the autoencoder groups by.** Besides the label readout, a pixel-side readout: the share of a site's 15
   neighbours that are also among the image's 15 neighbours as 4x4 RGB thumbnails (centred cosine), next to its value
   for shuffled images. Captions state what was observed; the sentences under Figures above are hypotheses.
8. **Housekeeping.** Permutation values are single draws and are labelled as reference values; bands are ranges over
   pairs. The viewer shows PSNR in dB for autoencoders and names their last site `code`.

Figures become: (1) lines, neighbour overlap with the matched classifier by site, one line per pair group;
(2) lines, the label readout and the thumbnail readout by site per network; (3) CKA by site (supporting);
(4) embedding panels at stem and b9; (5) reconstructions of fixed probe images with the PSNR table.

Work is now: pre-flight 1 + pilot 3 short units, train 5 (4 autoencoders, 1 classifier), embed 5, compare 1.
Projected spend about $2.5, wall-clock about 45 min.
