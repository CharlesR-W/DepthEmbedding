"""Independent per-site principal coordinates and UMAP on the CIFAR-10 probe.

Commands: deploy | pilot | launch PILOT_CALL_ID | status CALL_ID | fetch
To retry only selected units, append short:step arguments to launch.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import warnings

import modal

HERE = Path(__file__).resolve().parent

APP = "cm-cifar-coherent-baseline-embed"
VOLUME = "cm-cifar-coherent"
ROOT = Path("/out")
REFERENCE = ROOT / "analysis/r20s0b/probe.npz"
TARGET = ROOT / "analysis/stbase"
SCRATCH = ROOT / "work/stbase/v16s0"
STEPS = (0, 64, 256, 1024, 2560, 6400, 25600, 64000)
VGG_STEPS = (0, 256, 2560, 25600, 64000)
NETS = {f"r{d}s{s}": f"resnet{d}-relu-seed{s}" for d in (20, 32, 56) for s in (0, 1)}
ALL = [(short, step) for short in (*NETS, "v16s0")
       for step in (VGG_STEPS if short == "v16s0" else STEPS)]
PILOT = ("r56s0", 64000)
CPU, MEM, TIMEOUT = 2, 3, 600
GPU_CPU, GPU_MEM, GPU_TIMEOUT = 4, 12, 360
OVERHEAD = 30
CPU_RATE = CPU * .0000131 + MEM * .00000222
GPU_RATE = .000583 + GPU_CPU * .0000131 + GPU_MEM * .00000222
COORD_RATE = .25 * .0000131 + .00000222
app = modal.App(APP)
volume = modal.Volume.from_name(VOLUME)
image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("umap-learn==0.5.9.post2")
         .env({"OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
               "MKL_NUM_THREADS": "2", "NUMBA_NUM_THREADS": "1"}))
gpu_image = image
gpu_volumes = {"/out": volume, "/zoo": modal.Volume.from_name("crw-modelzoo-train").read_only()}
if modal.is_local():
    sys.path.insert(0, str(HERE / "upstream/cifar_more"))
    import measure

    assert VGG_STEPS == tuple(measure.STUDIES["v16s0"]["steps"])
    gpu_image = measure.image.add_local_file(
        str(HERE / "upstream/cifar_more/measure.py"), "/root/cm/measure.py")


def reference_probe():
    import numpy as np

    with np.load(REFERENCE) as data:
        idx, labels = data["idx"], data["labels"]
    assert idx.shape == labels.shape == (2000,)
    return idx, labels


def assert_probe(path, idx, labels):
    import numpy as np

    with np.load(path) as data:
        np.testing.assert_array_equal(data["idx"], idx)
        np.testing.assert_array_equal(data["labels"], labels)


def coordinates(gram):
    import numpy as np
    from scipy.linalg import eigh

    n = len(gram)
    values, vectors = eigh(gram.astype(np.float64), subset_by_index=(n - 16, n - 1))
    values = values[::-1].clip(min=0)
    return vectors[:, ::-1] * np.sqrt(values), values


def umap_coordinates(gram):
    import numpy as np
    from scipy.sparse.linalg import ArpackError
    from umap import UMAP

    distances = np.maximum(1 - gram.astype(np.float64), 0).astype(np.float32)
    np.fill_diagonal(distances, 0)

    def fit(init):
        return UMAP(metric="precomputed", n_neighbors=15, min_dist=.1, n_components=2,
                    random_state=0, init=init, n_jobs=1).fit_transform(distances)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r"using precomputed metric.*")
        # UMAP catches many spectral failures internally and warns before fallback.
        warnings.filterwarnings("error", message=r"(?is).*falling back to random initiali[sz]ation.*")
        try:
            return fit("spectral"), "spectral", None
        except (ArpackError, UserWarning) as exc:
            return fit("random"), "random", str(exc)


def reference_errors(source, site_index, gram, values):
    import numpy as np
    from scipy.linalg import eigh

    work = ROOT / "work/r20s0b"
    old_gram = np.load(work / f"gram_64000_{site_index}.npy")
    _, old_values = coordinates(old_gram)
    gram_error = float(np.max(np.abs(gram - old_gram)))
    pca_error = float(np.max(np.abs(values - old_values)))
    # The paired spectral-alignment spectrum was computed before rounding its
    # Gram to float32. Rule 0 is the same union 15-NN walk in both studies.
    with np.load(source / f"cell_{site_index}.npz") as cell:
        salign = cell["eig"][0, 1:17]
    walk = np.load(work / f"operators_64000_{site_index}.npy", mmap_mode="r")[0]
    n = len(walk)
    explorer = eigh(walk, eigvals_only=True, subset_by_index=(n - 17, n - 2))[::-1]
    walk_error = float(np.max(np.abs(salign - explorer)))
    assert gram_error < 1e-6 and pca_error < 1e-4 and walk_error < 1e-7, (
        site_index, gram_error, pca_error, walk_error)
    return dict(gram_max_abs=gram_error, pca_evals_max_abs=pca_error, rule0_evals_max_abs=walk_error)


@app.function(image=gpu_image, gpu="A100-40GB", cpu=GPU_CPU, memory=GPU_MEM * 1024,
              timeout=GPU_TIMEOUT, max_containers=5, scaledown_window=5, retries=0, volumes=gpu_volumes)
def extract_vgg(step):
    import measure as vgg
    import numpy as np
    import torch
    from coherent import centred_cosine_gram

    started = time.monotonic()
    volume.reload()
    vgg.environment()
    idx, labels = reference_probe()
    public = ROOT / "analysis/v16s0"
    assert_probe(public / "probe.npz", idx, labels)
    sites = vgg.site_names("v16s0")
    assert sites == json.loads((public / "meta.json").read_text())["sites"] and len(sites) == 11

    # Activation extraction follows spacetime_vgg_modal.rebuild.
    model, data = vgg.load("v16s0", step)
    np.testing.assert_array_equal(data["idx"], idx)
    np.testing.assert_array_equal(data["test_y"][idx].numpy(), labels)
    extract_sites = vgg.sites_fn("v16s0")
    n = len(idx)
    store = None
    dest = SCRATCH / str(step)
    dest.mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for at in range(0, n, 250):
            batch = data["norm"](data["test_x"][idx[at:at + 250]])
            acts = extract_sites(model, batch)
            assert len(acts) == len(sites)
            if store is None:
                store = [torch.empty((n,) + tuple(a.shape[1:])) for a in acts]
            for buffer, a in zip(store, acts):
                buffer[at:at + len(a)] = a.float().cpu()
        for si, activations in enumerate(store):
            gram = centred_cosine_gram(activations.flatten(1).cuda())
            np.save(dest / f"gram_{si}.npy", gram.cpu().numpy())
    np.savez(dest / "probe.npz", idx=idx, labels=labels, sites=sites)
    volume.commit()
    return dict(step=step, seconds=time.monotonic() - started)


@app.function(image=image, cpu=CPU, memory=MEM * 1024, timeout=TIMEOUT,
              max_containers=18, scaledown_window=5, retries=0, volumes={"/out": volume})
def unit(short, step):
    started = time.monotonic()
    import numpy as np
    from threadpoolctl import threadpool_limits

    volume.reload()
    idx, labels = reference_probe()
    if short == "v16s0":
        source = SCRATCH / str(step)
        with np.load(source / "probe.npz") as probe:
            sites = probe["sites"].tolist()
        assert len(sites) == 11
    else:
        source = ROOT / "work/salign" / NETS[short] / str(step)
        blocks = (int(short[1:3]) - 2) // 2
        sites = ["stem", *(f"b{i}" for i in range(1, blocks + 1)), "pool", "logits"]
    assert_probe(source / "probe.npz", idx, labels)
    assert len(list(source.glob("gram_*.npy"))) == len(sites)

    pcs, evals, traces, layouts, inits = [], [], [], [], []
    fallbacks, validation = {}, {}
    with threadpool_limits(limits=CPU, user_api="blas"):
        for si, site in enumerate(sites):
            gram = np.load(source / f"gram_{si}.npy")
            assert gram.shape == (len(idx), len(idx))
            assert gram.dtype == (np.float64 if short == "v16s0" else np.float32)
            assert np.isfinite(gram).all()
            pc, values = coordinates(gram)
            layout, init, error = umap_coordinates(gram)
            assert np.isfinite(layout).all(), (short, step, site)
            pcs.append(pc)
            evals.append(values)
            traces.append(np.trace(gram, dtype=np.float64))
            layouts.append(layout)
            inits.append(init)
            if error:
                fallbacks[site] = error
            if (short, step) == ("r20s0", 64000):
                validation[site] = reference_errors(source, si, gram, values)

    dest = TARGET / short
    dest.mkdir(parents=True, exist_ok=True)
    common = dict(sites=np.array(sites), probe_idx=idx)
    np.savez_compressed(dest / f"pca_{step}.npz", coords=np.asarray(pcs, dtype="f4"),
                        evals=np.asarray(evals, dtype="f4"), trace=np.asarray(traces, dtype="f4"), **common)
    np.savez_compressed(dest / f"umap_{step}.npz", coords=np.asarray(layouts, dtype="f4"),
                        init=np.array(inits), **common)
    volume.commit()
    if short == "v16s0":
        for path in source.iterdir():
            path.unlink()
        source.rmdir()
        volume.commit()
    return dict(short=short, step=step, sites=len(sites), seconds=time.monotonic() - started,
                fallbacks=fallbacks, validation=validation)


@app.function(image=image, cpu=.25, memory=1024, timeout=1800, scaledown_window=5, retries=0)
def coordinate(units):
    started = time.monotonic()
    assert units and len(set(units)) == len(units) and set(units) <= set(ALL)
    gpu_steps = [step for short, step in units if short == "v16s0"]
    gpu_calls = [extract_vgg.spawn(step) for step in gpu_steps]
    cpu_calls = [(key, unit.spawn(*key)) for key in units if key[0] != "v16s0"]
    gpu_rows, rows = [], []
    for step, call in zip(gpu_steps, gpu_calls):
        try:
            gpu_rows.append(call.get())
            cpu_calls.append((("v16s0", step), unit.spawn("v16s0", step)))
        except Exception as exc:
            gpu_rows.append(dict(step=step, error=repr(exc), seconds=GPU_TIMEOUT))
    for (short, step), call in cpu_calls:
        try:
            rows.append(call.get())
        except Exception as exc:
            rows.append(dict(short=short, step=step, error=repr(exc), seconds=TIMEOUT))
        print(rows[-1], flush=True)
    seconds = time.monotonic() - started
    estimated = sum(row["seconds"] + OVERHEAD for row in rows) * CPU_RATE
    estimated += sum(row["seconds"] + OVERHEAD for row in gpu_rows) * GPU_RATE
    estimated += (seconds + OVERHEAD) * COORD_RATE
    return dict(rows=rows, gpu=gpu_rows, seconds=seconds, estimated_usd=estimated)


def fetch():
    import numpy as np

    dest = HERE / "data/stbase"
    dest.mkdir(parents=True, exist_ok=True)
    with np.load(HERE / "data/r20s0b/probe.npz") as probe:
        idx = probe["idx"]
    # Stage a complete download, then validate every file before installing it.
    with tempfile.TemporaryDirectory() as temp:
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", VOLUME,
                        "/analysis/stbase/", temp + "/"], check=True)
        staging = Path(temp) / "stbase"
        site_order = {}
        for short, step in ALL:
            for method, k in (("pca", 16), ("umap", 2)):
                with np.load(staging / short / f"{method}_{step}.npz") as data:
                    sites = data["sites"].tolist()
                    np.testing.assert_array_equal(data["probe_idx"], idx)
                    assert sites == site_order.setdefault(short, sites)
                    expected = 11 if short == "v16s0" else (int(short[1:3]) - 2) // 2 + 3
                    assert len(sites) == expected and len(set(sites)) == expected
                    assert data["coords"].shape == (expected, 2000, k)
                    assert data["coords"].dtype == np.float32 and np.isfinite(data["coords"]).all()
                    if method == "pca":
                        vals, trace = data["evals"], data["trace"]
                        assert vals.shape == (expected, 16) and trace.shape == (expected,)
                        assert vals.dtype == trace.dtype == np.float32
                        assert np.isfinite(vals).all() and np.isfinite(trace).all()
                        assert (vals >= 0).all() and (np.diff(vals, axis=1) <= 0).all()
                        assert (vals.sum(axis=1) <= trace + .001).all()
                        norms = np.square(data["coords"], dtype=np.float64).sum(axis=1)
                        np.testing.assert_allclose(norms, vals, rtol=5e-6)
                    else:
                        assert data["init"].shape == (expected,)
                        assert set(data["init"].tolist()) <= {"spectral", "random"}
        import shutil
        for short, step in ALL:
            (dest / short).mkdir(exist_ok=True)
            for method in ("pca", "umap"):
                name = f"{method}_{step}.npz"
                shutil.move(staging / short / name, dest / short / name)
    print(f"Validated and installed {2 * len(ALL)} files under {dest}")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "deploy":
        subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)
    elif command == "pilot":
        print(modal.Function.from_name(APP, "coordinate").spawn([PILOT]).object_id)
    elif command == "launch":
        pilot = modal.FunctionCall.from_id(args[0]).get(timeout=0)
        main = pilot["rows"][0]
        assert len(pilot["rows"]) == 1 and (main["short"], main["step"]) == PILOT and "error" not in main
        # Give every CPU unit the full 30-site pilot time, and reserve the full
        # timeout for every GPU extraction. Include a separate $1 contingency.
        projected = (main["seconds"] + OVERHEAD) * CPU_RATE * len(ALL)
        projected += (GPU_TIMEOUT + OVERHEAD) * GPU_RATE * len(VGG_STEPS) + 1
        print(f"Projected total: ${projected:.2f}", flush=True)
        if projected > 10:
            raise SystemExit("Projected total exceeds the $10 budget")
        units = [(short, int(step)) for short, step in (arg.split(":") for arg in args[1:])]
        print(modal.Function.from_name(APP, "coordinate").spawn(units or [u for u in ALL if u != PILOT]).object_id)
    elif command == "status":
        try:
            print(json.dumps(modal.FunctionCall.from_id(args[0]).get(timeout=20), indent=2))
        except TimeoutError:
            print("running")
    elif command == "fetch":
        fetch()
    else:
        raise SystemExit(__doc__)
