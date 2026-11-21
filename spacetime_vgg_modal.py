"""Rebuild VGG rule-0 walks and run the existing through-depth stage on Modal.

Commands: deploy | launch [steps ...] | status CALL_ID | fetch
Each launch has one worker per requested checkpoint and no automatic retries.
"""
from pathlib import Path
import json
import subprocess
import sys
import time

import modal

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "upstream/cifar_more"))
import measure

APP = "cm-cifar-coherent-spacetime-vgg"
TAG = "v16s0"
STEPS = tuple(measure.STUDIES[TAG]["steps"])
REFERENCE_STEPS = (0, 64, 256, 1024, 2560, 6400, 25600, 64000)
TEST_ACC = 0.9307  # final checkpoint in /analysis/v16s0/curves.json
CPU, MEM_GIB = 8, 24
UNIT_TIMEOUT = 1200
COORD_TIMEOUT = 1800
STARTUP = 60
UNIT_RATE = measure.A100 + CPU * measure.CPU + MEM_GIB * measure.MEM
COORD_RATE = .25 * measure.CPU + measure.MEM
SPECTRUM_TOL = 1e-7  # the explorer stores eigenvalues in float32
VOLUME = "cm-cifar-coherent"
REMOTE_DIR = f"/analysis/{TAG}/spacetime"
image = measure.image.add_local_file(str(HERE / "upstream/cifar_more" / "measure.py"), "/root/cm/measure.py")
for name in ("spacetime.py", "spacetime_stage.py"):
    image = image.add_local_file(str(HERE / name), "/root/cc/" + name)
app = modal.App(APP)


def rule_zero_walk(gram):
    """The rule-0 branch of cm3.kernel_bank and cm3.lazy_operators."""
    import torch
    from coherent import symmetric_balance

    distances = 1 - gram.double()
    distances.diagonal().fill_(float("inf"))
    neighbours = distances.topk(15, largest=False, dim=1).indices
    mask = torch.zeros_like(gram, dtype=torch.bool)
    mask.scatter_(1, neighbours, True)
    weights = torch.where(mask | mask.T, torch.ones_like(gram), torch.full_like(gram, 1e-6))
    weights.diagonal().fill_(0)
    walk = symmetric_balance(torch.log(weights)) * .5
    walk.diagonal().add_(.5)
    return walk


def rebuild(step, probe_idx, probe_labels, sites):
    import cm3
    import numpy as np
    import torch
    from coherent import block_modes, centred_cosine_gram

    model, data = measure.load(TAG, step)
    n = len(probe_idx)
    idx = data["idx"]
    np.testing.assert_array_equal(idx, probe_idx)
    np.testing.assert_array_equal(data["test_y"][idx].numpy(), probe_labels)
    extract_sites = measure.sites_fn(TAG)
    store = None
    walks = []
    differences = []
    with torch.inference_mode():
        for at in range(0, n, 250):
            batch = data["norm"](data["test_x"][idx[at:at + 250]])
            acts = extract_sites(model, batch)
            assert len(acts) == len(sites)
            if store is None:
                store = [torch.empty((n,) + tuple(a.shape[1:])) for a in acts]
            for dest, a in zip(store, acts):
                dest[at:at + len(a)] = a.float().cpu()
        root = Path("/out/analysis") / TAG
        with np.load(root / f"layer_{step}.npz") as explorer:
            original = explorer["vals"][0]
        pairs = cm3.pairs(len(sites))
        for site, activations in enumerate(store):
            gram = centred_cosine_gram(activations.flatten(1).cuda())
            walk = rule_zero_walk(gram)
            expected = original[pairs.index((site, site))]
            actual, _ = block_modes([walk], len(expected), shared_stationary=True)
            differences.append(float(np.abs(actual.cpu().numpy() - expected).max()))
            walks.append(walk.cpu().numpy())
    return np.stack(walks), differences


@app.function(image=image, gpu="A100-40GB", cpu=CPU, memory=MEM_GIB * 1024,
              timeout=UNIT_TIMEOUT, retries=0, max_containers=len(STEPS), volumes=measure.VOLS)
def unit(step):
    import numpy as np
    import spacetime_stage as stage
    import torch
    from threadpoolctl import threadpool_limits

    start = time.monotonic()
    row = dict(step=step)
    try:
        measure.vol.reload()
        measure.environment()
        assert torch.cuda.is_available()
        root = Path("/out/analysis")
        with np.load(root / "r20s0b/probe.npz") as ref, np.load(root / TAG / "probe.npz") as vgg:
            idx, labels = ref["idx"], ref["labels"].astype(int)
            assert idx.shape == labels.shape == (2000,)
            np.testing.assert_array_equal(vgg["idx"], idx)
            np.testing.assert_array_equal(vgg["labels"], labels)
        sites = measure.site_names(TAG)
        meta = json.loads((root / TAG / "meta.json").read_text())
        assert sites == meta["sites"] and len(sites) == 11
        walks, errors = rebuild(step, idx, labels, sites)
        row.update(rebuild_seconds=time.monotonic() - start, eigenvalue_max_error=max(errors),
                   eigenvalue_errors_by_site=dict(zip(sites, errors)))
        print(step, row, flush=True)
        assert max(errors) < SPECTRUM_TOL, row
        assert walks.shape == (11, 2000, 2000) and walks.dtype == np.float64
        assert np.abs(walks.sum(1) - 1).max() < 1e-9
        assert np.abs(walks - walks.transpose(0, 2, 1)).max() < 1e-12

        compute_start = time.monotonic()
        with threadpool_limits(limits=CPU, user_api="blas"):
            result = stage.compute(walks, labels, controls=False, seed=REFERENCE_STEPS.index(step),
                                   log=lambda *args: print(step, *args, flush=True))
        row.update(compute_seconds=time.monotonic() - compute_start,
                   residual=float(max(result["resid"].max(), result["paper_resid"].max())),
                   balance=float(max(result["balance_check_own"], result["balance_check_none"])))
        failures = stage.accept(result)
        if failures:
            raise RuntimeError(f"Acceptance failed: {failures}")
        result.update(probe_idx=idx, sites=np.array(sites))
        directory = Path("/out" + REMOTE_DIR)
        directory.mkdir(exist_ok=True)
        path = directory / f"st_{step}_0.npz"
        pending = path.with_suffix(".pending.npz")
        np.savez_compressed(pending, **result)
        pending.replace(path)
        measure.vol.commit()
        row["status"] = "ok"
    except Exception as error:
        row.update(status="failed", error=repr(error))
        print(step, row, flush=True)
    row["seconds"] = time.monotonic() - start
    row["estimated_usd"] = (row["seconds"] + STARTUP) * UNIT_RATE
    return row


@app.function(image=image, cpu=.25, memory=1024, timeout=COORD_TIMEOUT, retries=0)
def coordinate(steps):
    start = time.monotonic()
    assert steps and len(set(steps)) == len(steps) and set(steps) <= set(STEPS)
    compute_bound = len(steps) * (UNIT_TIMEOUT + STARTUP) * UNIT_RATE
    compute_bound += (COORD_TIMEOUT + STARTUP) * COORD_RATE
    assert compute_bound < 8
    rows = []
    for index, result in enumerate(unit.map(steps, return_exceptions=True)):
        step = steps[index]
        if isinstance(result, Exception):
            result = dict(step=step, status="failed", error=repr(result),
                          estimated_usd=(UNIT_TIMEOUT + STARTUP) * UNIT_RATE)
        print(result, flush=True)
        rows.append(result)
    assert len(rows) == len(steps)
    coordinator_seconds = time.monotonic() - start
    spend = sum(row["estimated_usd"] for row in rows) + (coordinator_seconds + STARTUP) * COORD_RATE
    return dict(rows=rows, estimated_usd=spend, coordinator_seconds=coordinator_seconds)


def fetch():
    import numpy as np

    dest = HERE / "data/stnets" / TAG
    dest.mkdir(parents=True, exist_ok=True)
    for step in STEPS:
        name = f"st_{step}_0.npz"
        subprocess.run([sys.executable, "-m", "modal", "volume", "get", "--force", VOLUME,
                        f"{REMOTE_DIR}/{name}", str(dest / name)], check=True)
    with np.load(dest / "st_0_0.npz") as first:
        sites = first["sites"].tolist()
        idx = first["probe_idx"]
    for step in STEPS:
        with np.load(dest / f"st_{step}_0.npz") as data:
            np.testing.assert_array_equal(data["probe_idx"], idx)
            assert data["sites"].tolist() == sites
    info = dict(net="vgg16-bn-relu-seed0", sites=sites, steps=STEPS, test_acc=TEST_ACC)
    (dest / "info.json").write_text(json.dumps(info, indent=2) + "\n")


if __name__ == "__main__":
    command, *args = sys.argv[1:]
    if command == "deploy":
        subprocess.run([sys.executable, "-m", "modal", "deploy", __file__], check=True)
    elif command == "launch":
        steps = [int(arg) for arg in args] if args else list(STEPS)
        call = modal.Function.from_name(APP, "coordinate").spawn(steps)
        print(call.object_id)
    elif command == "status":
        try:
            print(json.dumps(modal.FunctionCall.from_id(args[0]).get(timeout=0), indent=2))
        except TimeoutError:
            print("running")
    elif command == "fetch":
        fetch()
    else:
        raise SystemExit(__doc__)
