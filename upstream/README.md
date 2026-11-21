# Vendored CoherentModes dependencies

Source checkout HEAD: `4c932fa8d810579fcaaf72e117a2eb66aea9a95c`. All files below were copied byte-for-byte, without edits. Destination paths are relative to this directory; source paths are relative to the root of CoherentModes.

| Destination | Source | Version |
| --- | --- | --- |
| `cifar_more/measure.py` | `explore/cifar_more/measure.py` | HEAD |
| `cifar_more/cm3.py` | `explore/cifar_more/cm3.py` | HEAD |
| `cifar_more/nets.py` | `explore/cifar_more/nets.py` | HEAD |
| `cifar_more/c100.py` | `explore/cifar_more/c100.py` | HEAD |
| `cifar_more/decod.py` | `explore/cifar_more/decod.py` | HEAD |
| `cifar_more/validate3.py` | `explore/cifar_more/validate3.py` | HEAD |
| `cifar_coherent/schema_v2.py` | `explore/cifar_coherent/schema_v2.py` | HEAD |
| `cifar_coherent/sweep_math.py` | `explore/cifar_coherent/sweep_math.py` | HEAD |
| `cifar_coherent/construction.py` | `explore/cifar_coherent/construction.py` | HEAD |
| `spectral_align/nets.py` | `explore/spectral_align/nets.py` | HEAD |
| `spectral_align/metadata/accuracies.json` | `explore/spectral_align/data/accuracies.json` | Ignored working-tree file, not in HEAD (186 bytes) |
| `spectral_align/metadata/probe.npz` | `explore/spectral_align/data/probe.npz` | Ignored working-tree file, not in HEAD (32,504 bytes) |

The metadata goes in `metadata/` so the repository's `data/` ignore rule does not exclude it.

The sibling directory layout preserves `measure.py`'s image assembly and keeps the two different `nets.py` modules separate. `cifar_coherent/coherent.py` and `cifar_coherent/seba.py` are relative symlinks to the root copies of the corresponding `explore/cifar_coherent/` files (both at HEAD). The root through-depth files and spec (other than `coherent.py` and `seba.py`) came from untracked working-tree files, not from HEAD.
