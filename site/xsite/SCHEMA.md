# Experiment embedding schema v1

Datasets live under `data/sitex/` and are copied into the built site's `x/` root. Paths in a descriptor (`file`, `sprite`, `probe.labels`, `probe.sprite`) are relative to **that descriptor**, not the page. All JSON uses UTF-8. Null means unavailable.

| Descriptor | Coordinate files beside it | Shape in C order |
|---|---|---|
| `ops/<op>/op.json` | `probe_<kind>.i16.gz` or `probe_<kind>.i8.gz` | `[L,N,K]`: chain site, probe row, mode |
| `ops/<op>/ov/<set>.json` | `<set>_<kind>.i16.gz` | `[L,n,K]`: chain site, sampled image, mode |
| `ops/<op>/paths/<set>.json` | `<set>_<kind>.i16.gz` | `[L,P,S,K]`: chain site, path, path parameter, mode |

Each `kinds[kind]` descriptor has `file`, `dtype` (`i16` = signed little-endian 16-bit, `i8` = signed 8-bit), `shape`, `scale` (`[L][K]`). Gunzip to **headerless** integers, with mode contiguous. Decode a coordinate as `int_at_flat_index * scale[site][mode]`. Scales are chosen independently per dataset/kind/site/mode, as `max(abs(x)) / 32767` (`i16`) or `/127` (`i8`); all-zero blocks use scale 1. The rounding error is at most half a scale. Signed probe, overlay and path arrays are in their original operator bases; no normalization or projection is baked into the integers.

## Operator

`schema:1`, `id`, `title`, `fp`, `net` (`id,label,step,train,acc`), `kernel` (`rule,param,label`), `probe`, `chain`, `units`, `c`, `t`, `tstar`, `vals`, `K`, `legacy_basis`, `kinds`, `site_norm`, `axes`, `viewer_signs`, `provenance`. `chain` is the full coordinate grid, in order; `L = chain.length` and `K = vals.length`. `legacy_basis` is inversion's sum of absolute **raw** eigenvalues, or null. `probe` contains `id,n,ids_sha256,labels_sha256,viewer_order,labels,sprite,sprite_cols`. `labels_sha256` hashes the ordered one-byte class IDs exactly as written to `labels.bin`. Production viewer-order probes reuse the viewer's labels and WebP sprite; the build requires their probe count and both digests to agree with `n`, `probe_ids_sha256` and `probe_labels_sha256` in `d/meta.json`, and verifies the actual `d/labels.bin` bytes. `--viewer-from` requires a viewer export with these digests. Fixtures have a smaller real probe, `viewer_order:false`, inline `probe.ids`, a local `labels.bin`, and no sprite.

- `joint` stores the signed joint modes/extension. `source` uses the same mode indices, with `source_from_joint(y,c,t,vals) = 2c T y`; `T` has diagonal `1/(2c) + t*degree - vals` and adjacent entries `-t`. For inversion an independently stored one-step array may be used after checking the stencil. `site` uses the raw site-specific walk basis (`kinds.site.axes:"own"`); mode indices here do **not** identify the same axes across sites or correspond to the joint/source mode indices.
- For `norm:"site"`, divide decoded values by the **operator's** `site_norm[kind][site]` for probe, overlays and paths alike. It is the pre-quantization Frobenius norm of that site's probe over all shipped modes. For a display using fewer modes, a consumer may recompute the norm from the same truncated probe for all objects in the panel. These are display units, not X3's median same-class distance or any other experiment metric.
- `axes.pooled` is `[K][min(K,3)]`, columns descending from eigendecomposition of the pooled joint second moment after per-site Frobenius normalization. The largest-magnitude loading of each column is positive. Project a row by right-multiplying this matrix. Use same joint axes for source; use the first coordinate modes directly for `site`. For an X1 panel with `R`, right-multiply its row by `R` first and project into the **reference panel's** axes; each menu item has its own `R` and `stat`.
- For `axes:"viewer"` on `r20s0-c10-knn15`, apply `viewer_signs` per mode before projection. Optional `axes.viewer` stores the K=16 axes computed from the viewer's sign-preserving int8 slices and rounded slice weights at r20s0 step 64000, own coupling index 3. The experiment retains its higher-resolution coordinates, so a viewer-matched projection is not byte-identical to the viewer cloud. A different K needs a corresponding PCA fit.

## Sets

Both `ov` and `paths` descriptors have `schema:1`, `id`, `op`, `op_fp`, `kinds`, `sprite`, `labels`, `provenance`. Reject unless **both** `op == op.id` and `op_fp == op.fp`, and each kind and coordinate shape is supported by that operator. `provenance` is `{input,sha256,code}`, referencing the corresponding pinned record and raw file.

`labels:{space,names}` describes the true label IDs (`meta.label` for overlays, `meta.label_a` and `meta.label_b` for path endpoints). `space` is a stable identifier, e.g. `cifar10`, `svhn`, `cifar100`; `names` is the class-name list indexed by nonnegative integer ID, or `null` when unavailable. If names are supplied every true ID must index that list. This namespace does not describe `meta.pred` or `meta.target`, which are in the network's output space (CIFAR-10 for these overlays).

- Overlay: `label,n,pairs,images,sample,sprite_cols,meta`. `images:{dataset,ids}` identifies the selected images; `sample:{rule,of,pos}` gives the sorted row positions in the original experiment list. If `pairs:{set,by:"index"}`, the matched set's row `i` corresponds to row `i` here; don't join on predictions. `meta` contains per-image `label` and may include `pred,success,rowmax` and `scalars` (`{name:{values:[L][n],label,range:[lo,hi]}}`). X3 clean has `success:null`; u2/u8/n8 record a changed predicted label, t8 records a target hit and also emits `meta.target`. X3 sets for both operators share one sample and one clean-image atlas (`../../../sprites/x3_sample.webp`, 20 columns). Atlas tiles are row-major 32 px square.
- Path set: `l0,chain,sites,s,P,S,meta`. Only `chain` indexes coordinates. `sites` is the producer's separate measurement grid, e.g. b3-start has 9 coordinate sites and 8 downstream measurement sites. `s` contains the S sampled path parameters in ascending order. `meta:{pair_pos,kind,label_a,label_b,s_switch}` describes the P sampled pairs, with `kind` `same`/`diff`; missing individual switches are null.

## Fingerprint and build gate

`fp` is lowercase SHA-256 hex. Hash in this order (no separators beyond the specified serialization):

1. Canonical JSON of `{c,chain,kernel:{param,rule},t,units,vals}` (`c,t,vals` rounded to 12 decimal places as float64, signed zero canonicalized to `0.0`); sort keys, compact `,` / `:` separators, `ensure_ascii=False`.
2. Ordered probe image IDs as contiguous signed little-endian int64 bytes. `probe.ids_sha256` hashes only these bytes.
3. For each kind sorted by key, canonical JSON `{dtype,kind,shape}`, exact uncompressed shipped signed integer bytes, then C-order little-endian float64 bytes of its `[L][K]` scales (parsed from the descriptor).

No labels, sprites, axes, paths or overlay bytes enter `fp`. The staged `MANIFEST.json` is `{schema:1,ops:{id:{fp,overlays,paths,bytes,provenance}},files:{relativePath:{bytes,sha256}}}`. This hashes the serialized output files, including JSON and sprites, excluding itself. It is written last, after swapping in the output directory. Missing or stale manifests are not a buildable state.

## Packaging from the pinned inputs

`fetch.py --inventory` lists but does not transfer. Regular fetch stages the base-plan inputs under `data/fetch/<volume>/<path>` and repo-local inputs under `data/fetch/local/<repo-relative-path>`; `READY.json` records exact sizes and hashes. `inputs.json` pins each operator and its raw bundle. `validate.Inputs.file(key)` verifies a READY-backed file; `Inputs.bundle(op_id)` also verifies all the bundle pins. `validate_operator(inputs, op_id)` returns validated, unscaled arrays and metadata. New entries add `ops_<entry>.py` with `package(inputs, root)` (imports may be relative or script-aware); no entry edits `ops.py` to register itself. The packager discovers these automatically. Run `uv run --with numpy --with scipy --with pillow python site/xsite/ops.py --entry <entry>` to replace one entry while preserving the rest, or omit `--entry` for all entries. `--fixtures` regenerates the frozen F1–F3 integration data. `ops.package_operator` wraps `opdata.write_operator`; `write_overlay` and `write_paths` take the returned operator descriptor and unscaled coordinates.
