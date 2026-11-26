# ae: the classifiers' ResNet-20 trained to reconstruct (2026-11-26)

Exploratory. Spec with the review revisions: `SPEC.md`. Results page: the Experiments entry `ae-objective` on the
site; the four networks are in the Viewer under "Trained to reconstruct (autoencoders)".

There is no separate code here. The X1 pipeline in `xp/` gained an `objective='ae'` branch:

| Path | What changed |
|---|---|
| `xp/common.py` | `AE_NETS` (`ae10-s0`, `ae10-s1`, `ae64-s0`, `ae64-s1`), `build_ae`, `site_names`, and the classifier `c10x-s1`. |
| `xp/modal_train.py` | MSE branch of `train`, pilot options (`steps`, `lr`, `holdout`, `suffix`), `ae_check`. |
| `xp/modal_xp.py` | `embed` names an autoencoder's last site `code`. |
| `site/xsite/fetch.py`, `pin_ae.py`, `inputs.json` | Staged and pinned inputs (`ae-<net>-full` operators). |
| `site/build_site.py`, `site/js/embedding.js` | Viewer export and the network group. |
| `site/xsite/gen_ae.py`, `ops_ae.py`, `site/content/experiments/ae-objective.html` | The Experiments entry. |

## Reproduce

```bash
uv run --with modal python xp/modal_train.py deploy && uv run --with modal python xp/modal_xp.py deploy
uv run --with modal python xp/modal_train.py launch ae_check
uv run --with modal python xp/modal_train.py launch train '{"nets":["ae10-s0"],"steps":8000,"holdout":5000,"lr":0.1,"suffix":"__pilot_lr0.1"}'   # also 0.03, 0.01
uv run --with modal python xp/modal_train.py launch train '{"nets":["ae10-s0","ae10-s1","ae64-s0","ae64-s1","c10x-s1"]}'
uv run --with modal python xp/modal_xp.py launch embed '{"nets":["ae10-s0","ae10-s1","ae64-s0","ae64-s1","c10x-s1"],"probes":["c10"]}'
uv run --with modal python xp/modal_xp.py launch compare '{"units":[{"probe":"c10"}],"out":"compare_ae","nets":["c10-s0","c10-s1","c10x-s0","c10x-s1","init-s0","init-s1","ae10-s0","ae10-s1","ae64-s0","ae64-s1"]}'
uv run --with modal python site/xsite/fetch.py && (cd site/xsite && python3 pin_ae.py)
```

Redeploying an app kills its running jobs; wait half a minute after a deploy before launching.

## Status (2026-11-26)

All stages complete. Launch records with call ids are in `xp/runs/20261126-*`.

| Stage | Units | Result |
|---|---|---|
| ae_check | 1 | Encoder initial weights equal the classifier's under the same seed (all four autoencoders) and the zoo's step-0 checkpoints (both seeds). One fixed batch fitted from MSE 1.19 to 0.010. |
| pilot | 3 | 8,000 steps, 5,000 training images held out: held-out MSE 0.364 (lr 0.1), 0.370 (0.03), 0.380 (0.01). Kept 0.1, the classifier's rate. |
| train | 5 | 64,000 steps, about 15 min each on A10G (classifier 10 min). Test PSNR 16.84, 16.86 dB (10-d code), 21.22, 21.25 dB (64-d); rank-10 and rank-64 PCA 16.71 and 20.80 dB; mean image 12.09 dB. `c10x-s1` 91.7% test. |
| embed | 5 | CIFAR-10 probe only, about 75 s each. |
| compare | 1 | 10 networks, 60 s. Restricted to the five networks of the archived `compare/c10.npz` it reproduces `nn`, `cka` and `readout` exactly. |

Spend: about $2.3 estimated from unit times and Modal's 2026-11 prices (training about $1.9); not an invoice.
Left on the volume `crw-depth-xp-results`: `nets/ae*`, `nets/c10x-s1`, `embed/ae*`, `embed/c10x-s1`, `compare_ae/`,
and the three pilot directories `nets/ae10-s0__pilot_lr*`.

Reviews: spec by GPT-6.2 Astra (two rounds; its points are the "Revisions after review" in `SPEC.md`); code by
GPT-6.2 Sol.
