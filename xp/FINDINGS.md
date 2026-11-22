# Findings (exploratory; one architecture, two seeds per training set, one probe per dataset)

Numbers are read from `xp/out/x1_summary.json`, `xp/out/x3_summary.json` and `xp/out/x4_summary.json`;
definitions are in `xp/SPEC.md`. This version incorporates an independent audit of the first draft against the
data (GPT-6.2 Sol, 2026-11-21), which corrected several overstatements; the open questions it raised are listed
at the end.

"Joint agreement" is the held-out cosine between two networks' 16-mode trunk embeddings (stem..b9, k-NN kernel)
after one rotation shared by all sites: 1 means the same embedding up to that rotation, 0 unrelated. Sites
quoted in fours are stem, b3, b6, b9.

## X1. Different training pipelines, same probe

The comparison is between training pipelines: the images, labels, number of classes, normalisation and
augmentation all change together. Numbers are on the CIFAR-10 probe unless stated.

1. **Several natural-image pipelines give high agreement, below the matching seed pairs.** CIFAR-10 against
   CIFAR-100: 0.91, 0.90, 0.89, 0.75. CIFAR-10 against ImageNet-32: 0.88, 0.87, 0.89, 0.80. CIFAR-100 against
   ImageNet-32: 0.95, 0.93, 0.94, 0.91, the closest cross pair, against 0.97, 0.96, 0.97, 0.94 for the two
   CIFAR-100 seeds and 0.97, 0.96, 0.97, 0.97 for the two ImageNet-32 seeds. CIFAR-10 seed pairs: 0.95 throughout
   as a mean of three pairs (0.93, 0.92, 0.93, 0.95 for the zoo pair alone). The class-split networks are further
   away: animals-only against CIFAR-10 0.90, 0.87, 0.84, 0.69; vehicles-only 0.78, 0.78, 0.78, 0.68.
2. **The ranking of pipelines depends on probe and depth, and this design does not isolate what drives it.**
   On the CIFAR-10 probe the networks with more classes agree better late, but on the SVHN probe at b9 CIFAR-10
   agrees more with animals-only (0.73) and vehicles-only (0.70) than with ImageNet-32 (0.64), and on the
   ImageNet-32 probe CIFAR-100 against ImageNet-32 falls to 0.76 at b9 against 0.89 and 0.92 for the seed pairs.
   Across pipelines, label count is confounded with image distribution and training performance.
   **Control for label granularity (run after the audit): CIFAR-100 images trained with the 20 coarse labels
   (`c100c`, two seeds, 78% and 77% test accuracy) give the same embedding as CIFAR-100 with 100 labels, to within the
   seed-pair spread:** fine against coarse 0.97, 0.96, 0.96, 0.93, against 0.97, 0.96, 0.97, 0.94 for the two
   fine-label seeds and 0.97, 0.97, 0.97, 0.94 for the two coarse-label seeds (CIFAR-10 probe; on the
   ImageNet-32 probe 0.95, 0.94, 0.92, 0.86 against seed pairs at 0.89 and 0.87 at b9). So a fivefold change in
   label granularity on fixed images does not move the embedding, which points at the images (the class-split
   networks also see different images) as what separates the pipelines, not the number of classes.
3. **CIFAR-10 and SVHN pipelines agree much less than seed pairs, at every depth.** 0.57 at the stem and 0.36 at
   b9, against 0.51 and 0.17 for the untrained pair with a CIFAR-10 network, and 0.56 and 0.24 for the
   shuffled-label network. So the stem already differs. On the SVHN probe the order of the three reverses at b9
   (0.25, 0.30, 0.44). The design cannot say that similar images are necessary: normalisation and augmentation
   also differ.
4. **For these two pipelines, a seed pair agrees less on the other's probe, most of all late.** SVHN seeds: 0.97,
   0.97, 0.98, 0.96 on SVHN digits against 0.93, 0.92, 0.88, 0.76 on CIFAR-10. CIFAR-10 seeds: 0.95 at b9 on
   CIFAR-10 against 0.75 on SVHN. This is not a general rule: the two ImageNet-32 seeds agree at 0.97 at b9 on
   CIFAR-10 images and 0.92 on ImageNet-32 images.
5. **Exact neighbour overlap falls much more with depth than the joint cosine.** CIFAR-10 pairs share 75% of
   15-nearest neighbours at the stem, 43% at b9 and 22% at the logits (mean of three pairs); CIFAR-10 against
   CIFAR-100 shares 72%, 21% and 8%. For the zoo CIFAR-10 pair, shuffling images within class reduces the joint
   cosine from 0.93 to 0.10 at the stem but only from 0.95 to 0.80 at b9: late joint agreement is largely
   agreement about class structure, early agreement is about individual images.
6. **Class readout.** A CIFAR-100 network puts 52% of a CIFAR-10 image's 15 neighbours in its own class at b9,
   against 88% for CIFAR-10 networks and 18% untrained. SVHN networks are at 16% at b9 and 14% at pool, below
   untrained.
7. Controls: the network trained here on CIFAR-10 (`c10x-s0`, 91.4% test) reproduces the zoo networks' late
   readouts closely and agrees with `c10-s0` slightly better than the zoo pair does early (0.97 against 0.93 at
   the stem); no equivalence test was run. The reference network's Grams match the published ones to 1.6e-6, its
   full-chain joint eigenvalues to 1.4e-12 and its coupling to 2e-11. Permutation nulls sit at their chance
   baselines (0.007 neighbour overlap, 0.02 CKA, -0.01 joint cosine at b9).

## X3. Adversarial images through depth

Attacked network: ResNet-20 seed 0; 1,000 held-out test images both seeds classify correctly; PGD, L-inf, 50 steps.
Accuracy under attack: 20.6% at eps 1/255, 0.5% at 2/255, 0% at 4 and 8; targeted at 8/255 hits its target 100%.
"Site-local displacement" is the distance between the attacked and clean image in a site's own diffusion
coordinates, in units of the median distance between two clean images of one class at that site.

1. **In the embedding, an eps-8 attack moves an image about as far as random sign noise of the same L-inf size up
   to b6; the gap opens at b7 and is very large at b9.** Site-local displacement, attack against noise: 0.02 / 0.03
   (stem), 0.46 / 0.52 (b3), 0.56 / 0.56 (b6), 0.68 / 0.46 (b7), 0.98 / 0.34 (b8), 2.6 / 0.42 (b9). The raw
   activations already tell the two apart earlier: cosine to the clean activation at b6 is 0.64 for the attack
   and 0.77 for noise, and 0.13 against 0.81 at b9. (Equal L-inf budget does not mean equal L2 size.)
2. **The class switch, read site by site, is spread across depth; it comes later for weaker attacks and latest
   for natural errors.** Switch site = first site from which the destination class outweighs the source class,
   for good, in the image's kernel row. Eps-8 untargeted: median b6, mode b7 (57% have switched by b6; with the
   stricter "destination is the top class" rule the median is b7). Eps-8 targeted and eps-2 untargeted: median
   b8, mode b9. Naturally misclassified images: mode b9, and 13% never switch. By b6 the eps-8 attacked images
   carry more weight on the destination class (0.28) than on their own (0.23), where clean images carry 0.43 on
   their own. About 13% of eps-8 successes "switch" at the stem, where class weights are near uniform; these are
   images whose own class never led.
3. **Successful eps-8 attacks end at typical distances from the destination class, with nearly all their weight
   on it.** At pool and logits the untargeted attacked image's distance to the destination class centroid is at
   the 63rd and 69th percentile of real members' distances in joint coordinates (targeted: 69th and 85th; 58th
   at the logits in site-local coordinates), and 97% of its kernel-row weight is on the destination class.
   Naturally misclassified images have 57% on the destination, 21% on the true class and 21% elsewhere; matching
   the attacks to the same (source, destination) pairs does not change this contrast. In the middle of the
   network (b3-b7) attacked images have a lower nearest-probe cosine than clean ones (about the 20th percentile);
   random noise does too (30th-45th).
4. **Transfer.** The eps-8 attack built on seed 0 flips seed 1 on 93% of images, to the same wrong class on 91%
   (eps 4: 68% and 66%; targeted eps 8: 44% flipped, 37% to the target). In seed 1, perturbations that flip it
   have a larger site-local displacement than those that do not already at b7 (eps 4: 0.35 against 0.26), with by
   far the largest separation at b9 (2.3 against 0.6). The split is by any flip.
5. Method checks: replacing only the logits row of a clean image by its attacked row changes no site-local
   readout at earlier sites (exactly 0) but moves the joint coordinates at the stem by 3% of the full attack's
   displacement and at b9 by 14%; joint coordinates smear late effects backwards, which is why the statements
   above use site-local readouts. The k-NN kernel gives the same modal switch site (b7) as the smooth kernel for
   eps-8 untargeted attacks; the histograms differ by 5 percentage points of total variation.

## X4. Interpolation between real activations, log gain (FTLE)

Network: ResNet-20 seed 0. 150 same-class and 150 different-class pairs of held-out, correctly classified images
(the same pairs at every start site). Straight path between the two activations at a start site (pixels, stem,
b3, b6, b8, b9), 257 points, pushed through the rest of the network. Medians over pairs unless stated.

1. **On these sampled paths there is little localised motion in activation space.** The sharpness index (1 for
   a smooth path, 0 for a jump) has site medians of 0.97-1.00 and a minimum over all paths of 0.89. The fastest
   sixteenth of a path carries 7-8% of its angular arc length at early sites (6.25% is uniform) and 7-15% at the
   logits; fewer than 1% of paths exceed 25% before the logits. Refining 50 paths to 1,024 steps leaves the
   window shares unchanged to 1e-4. This is unlike the wall-like transitions reported for MLPs on MNIST.
2. **Read through the probe, the change of class along a path is much sharper from b9 on, but that is a property
   of the readout and the cloud, not a barrier in the network.** The largest slope of (kernel-row weight on b's
   class minus weight on a's class) along a different-class path is 1-4 per unit s at sites before b9 and 7-14 at
   b9, pool and logits; in site-local coordinates the fastest sixteenth of the embedded path carries 12-21% of
   its motion before b9 and 26-32% at b9. The same sharpness appears for a path drawn straight at b9 itself
   (slope 9.7, share 32%), where the raw path is straight and everything downstream is affine. So it reflects
   how class-clustered the probe cloud is at b9, seen through an adaptive kernel whose row perplexity also drops
   there (164 at b8, 135 at b9, 90 at the logits for path midpoints); the experiment did not vary the two
   independently.
3. **Different-class paths contain points far from every probe image, more so late.** The least-supported point
   of a path has a nearest-probe cosine at about the 15th percentile of clean images' at early sites and the
   1st-2nd percentile at pool and logits (same-class paths: 9th-25th at the logits). This is a minimum over the
   path, usually near the middle but not at it (at s = 0.5 exactly, the 12th percentile at the logits).
4. **Sampled same-class chords stop leaving the class by b8.** A straight path between two images of one class
   leaves the class at the output for 23% of pairs when drawn in pixel space, 27% at b3, 9% at b6 and 0 of 150
   at b8 (at b9 it cannot, since the suffix is affine and the class regions of the logits are convex).
   Different-class paths visit a third class for 53% (pixels), 54% (b3), 34% (b6), 17% (b8), 9% (b9).
5. **Where a path moves fastest is associated with where the output switches, more strongly at later sites.**
   For all 150 different-class paths started at b3 (aligned on the first switch), the rank correlation between
   the location of peak angular speed at a site and the location of the first output switch is 0.06 at b4, 0.30
   at b7, 0.44 at b9 and 0.50 at the logits (0.08, 0.42, 0.70, 0.76 on the 69 single-switch paths). Part of this
   comes from the output normalisation: on the affine b9 control, where raw speed is constant, the angular speed
   peak still correlates 0.66 with the switch, because both sit where the centred logits are smallest.
6. **Log gain.** The raw log gain to the logits (log of the largest singular value of the Jacobian) is higher
   in the middle of a different-class path than at its ends for starts up to b6 (pixels 3.86 against 3.56; b3
   1.88 against 1.69; b6 1.82 against 1.57), slightly lower from b8 (1.49 against 1.55), and constant from b9.
   The interpolation direction has small gain relative to the maximum (ratio 0.02 from pixels, 0.11-0.15 from
   b3-b8). Over the 2,000 probe images, angular log gain to the logits is anti-correlated with the output margin
   (Spearman -0.65 from pixels, -0.72 from the stem, -0.76 from b3; misclassified images 4.57 against 4.15 for
   correct ones from pixels), but **most of that comes from the output normalisation**: the centred logit radius
   itself correlates +0.76 with margin, and with that factor removed the correlations are -0.29, -0.34, -0.34.
   Raw gain correlates -0.17. So high gain marks low-margin images weakly, not sharply.
7. Block-output ReLU masks differ between the two endpoints on 14% of units at b3 rising to 40% (same class) and
   48% (different class) at b9. The ReLU inside each block is not counted, and these are mask differences, not
   counts of linear regions crossed.
8. Controls: exact gains equal the closed forms (pool to logits, b9 to logits) to 2e-15 and an explicit SVD to
   4e-8; the b9 start is exactly affine downstream; all six start sites use the same 300 pairs; masked replay of
   the probe and endpoint-as-image errors are 1e-8 to 2e-8. Exact speed and finite-difference speed at the logits
   differ by 2.6% (median) at a step of 1/256 and 1.2% at 1/1024: the path is rough at fine scale.

## Open questions the audit raised

- Does the sharp b9 transition in X4.2 survive a bandwidth sweep or a kernel with matched final perplexity?
- Does the late speed-switch association in X4.5 hold on fresh pairs and after accounting for centred radius?
- X1 cannot separate image distribution, normalisation and augmentation; the coarse-label control addresses only
  label granularity, at one pair of granularities (100 against 20) on one image set.
