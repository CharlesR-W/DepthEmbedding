"""Exact identities of the kernel rows and the joint out-of-sample map (small synthetic sites).

Run: uv run --with numpy --with scipy --with pytest python -m pytest -q xp/test_dm.py
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dm
import spacetime as st


def cos_gram(x):
    x = x - x.mean(0)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    return x @ x.T


@pytest.fixture(scope='module')
def sites():
    rng = np.random.default_rng(0)
    base = rng.standard_normal((160, 12)) * np.linspace(3, 0.3, 12)
    return [cos_gram(base @ rng.standard_normal((12, 30)) + 0.5 * l * rng.standard_normal((160, 30))) for l in range(4)]


@pytest.mark.parametrize('name', ['ea20', 'pr20', 'ds20', 'knn8'])
def test_masked_row_replays_probe_row(sites, name):
    """A probe image, treated as a new point with itself masked out, gets its own row of P."""
    C = sites[0]
    ker = dm.build_kernel(C, name)
    assert abs(ker['P'].sum(1) - 1).max() < 1e-9 and abs(ker['P'] - ker['P'].T).max() < 1e-12
    Cm = C.copy(); np.fill_diagonal(Cm, -np.inf)
    assert abs(dm.rows(Cm, ker) - ker['P']).max() < 1e-9


@pytest.mark.parametrize('name', ['ea20', 'pr20', 'knn8'])
def test_joint_forward_replays_modes(sites, name):
    """The tridiagonal system, fed the probe's own rows, returns the modes: it is the eigen-equation."""
    kers = [dm.build_kernel(C, name) for C in sites]
    jm = dm.joint_modes(np.stack([k['P'] for k in kers]), k=10)
    assert jm['resid'].max() < 1e-8
    y, cond = dm.joint_forward(np.stack([k['P'] for k in kers]), jm)
    assert abs(y - jm['vecs']).max() < 1e-7 * max(1, cond.max())


def test_joint_forward_matches_block_operator(sites):
    """Independent assembly: append one node per site with one-way affinities and check the dense equation."""
    kers = [dm.build_kernel(C, 'ea20') for C in sites]
    Ps = np.stack([k['P'] for k in kers])
    jm = dm.joint_modes(Ps, k=6)
    rng = np.random.default_rng(1)
    r = rng.random((4, 1, 160)); r /= r.sum(2, keepdims=True)          # arbitrary stochastic rows of a new point
    y, _ = dm.joint_forward(r, jm)
    L, c, t = 4, jm['c'], jm['t']
    for k in range(6):
        f, mu, yk = jm['vecs'][:, :, k], jm['vals'][k], y[:, 0, k]
        lhs = (yk - np.einsum('ln,ln->l', r[:, 0], f)) / (2 * c)       # (I - M) part at the new node, M = (I + P)/2
        lhs += t * st.path_lap(yk[:, None, None])[:, 0, 0]
        assert abs(lhs - mu * yk).max() < 1e-9


def test_path_lap_sign():
    f = np.arange(4.0)[:, None, None] ** 2
    d = np.array([[1, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1.0]])
    got = st.path_lap(f)[:, 0, 0]
    assert np.allclose(got, d @ f[:, 0, 0])


def test_near_duplicate_query_is_finite(sites):
    """A new point far nearer to a probe image than that image's fitted neighbours still gets a finite row."""
    C = sites[0]
    ker = dm.build_kernel(C, 'ea20')
    c = C[:3].copy()
    c[np.arange(3), np.arange(3)] = 1 - 1e-9                 # unmasked near-duplicates of probe images 0..2
    r = dm.rows(c, ker)
    assert np.isfinite(r).all() and abs(r.sum(1) - 1).max() < 1e-12
    assert (r.argmax(1) == np.arange(3)).all()


@pytest.mark.parametrize('kernel', ['ea20', 'pr20'])
def test_torch_agrees_on_masked_rows_and_new_points(sites, kernel):
    torch = pytest.importorskip('torch')
    import xlib

    C = sites[0]
    ker = dm.build_kernel(C, kernel)
    op = xlib.fit_op([ker], xlib.site_spectra([ker]), [0])
    ext = xlib.Ext(op, device='cpu')
    masked = C.copy(); np.fill_diagonal(masked, -np.inf)
    c_new = C[:3].copy()
    c_new[np.arange(3), np.arange(3)] = 1 - 1e-9
    c_new[1, 2] = -np.inf
    cross = np.concatenate([masked, c_new])
    rows = ext.rows(0, torch.from_numpy(cross)).numpy()
    np.testing.assert_allclose(rows, dm.rows(cross, ker), rtol=0, atol=1e-12)
    np.testing.assert_allclose(rows[:len(C)], ker['P'], rtol=0, atol=1e-9)
    np.testing.assert_array_equal(np.diag(rows[:len(C)]), 0)


def test_pr_conditional_has_target_participation_ratio(sites):
    perp = 20
    _, _, _, logp = dm._cond_rows(1 - sites[0], perp, True, renyi=2)
    np.testing.assert_allclose(1 / np.exp(2 * logp).sum(1), perp, rtol=1e-12)
    assert np.exp(logp).max() <= perp ** -0.5
