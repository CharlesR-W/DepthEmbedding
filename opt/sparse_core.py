"""Isolated numerical prototypes. Walks preserve the positive off-edge floor."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import cholesky, cho_solve, eigh, get_lapack_funcs
from scipy.optimize import brentq
from scipy.sparse.linalg import LinearOperator, eigsh, lobpcg, splu

FLOOR = 1e-6


def knn(gram, k=15, batch=128):
    """Exact row partition; same distance arithmetic as reembed_test.walk.

    Input may be a memory map. Does not change the tie-breaking policy of
    np.argpartition; it is not a promise to match other libraries on ties.
    """
    n = len(gram)
    result = np.empty((n, k), dtype=np.int32)
    for lo in range(0, n, batch):
        hi = min(lo + batch, n)
        d = 1 - gram[lo:hi]
        d[np.arange(hi - lo), np.arange(lo, hi)] = np.inf
        result[lo:hi] = np.argpartition(d, k, axis=1)[:, :k]
    return result


def adjacency(nn):
    n, k = nn.shape
    a = sparse.csr_matrix((np.ones(n * k), (np.repeat(np.arange(n), k), nn.ravel())),
                          shape=(n, n))
    return a.maximum(a.T).tocsr()


def balance_weights(a, floor=FLOOR, tol=1e-12, maxiter=5000):
    """Balance W=(1-floor)*adjacency - floor*I + floor*11^T."""
    d = np.ones(a.shape[0])
    for i in range(maxiter):
        wd = (1 - floor) * (a @ d) + floor * (d.sum() - d)
        r = d * wd
        err = np.max(np.abs(r - 1))
        if err < tol:
            return d, i + 1, err
        d /= np.sqrt(r)
    raise RuntimeError(f'Sinkhorn residual {err:g} after {maxiter} steps')


@dataclass
class Walk:
    base: sparse.csr_matrix
    u: np.ndarray

    def __matmul__(self, x):
        if x.ndim == 1:
            return self.base @ x + self.u * (self.u @ x)
        return self.base @ x + self.u[:, None] * (self.u @ x)

    def dense(self):
        return self.base.toarray() + np.outer(self.u, self.u)


def make_walk(a, d, floor=FLOOR):
    dd = sparse.diags(d)
    m = 0.5 * (1 - floor) * (dd @ a @ dd) + sparse.diags(0.5 - 0.5 * floor * d**2)
    return Walk(m.tocsr(), np.sqrt(0.5 * floor) * d)


def centre(x):
    return x - x.mean(axis=1, keepdims=True)


def path_lap(x):
    y = np.zeros_like(x)
    d = x[1:] - x[:-1]
    y[1:] += d
    y[:-1] -= d
    return y


class Joint:
    def __init__(self, walks, scales):
        self.shape = (len(walks), len(walks[0].u))
        self.upper = 1. / np.min(scales)
        self.min_inv_scale = 1. / np.max(scales)
        self.h = [(sparse.eye(len(w.u), format='csr') - w.base) / c
                  for w, c in zip(walks, scales)]
        self.base = sparse.block_diag(self.h, format='csr')
        self.u = np.stack([w.u / np.sqrt(c) for w, c in zip(walks, scales)])
        self.d = np.array([np.mean(b.diagonal()) for b in self.h])
        self.dim = np.prod(self.shape)
        self.path = path_lap(np.eye(self.shape[0]))

    def apply(self, x, t, eps=0.):
        if x.ndim == 2:
            out = (self.base @ x.ravel()).reshape(self.shape)
            out -= self.u * np.einsum('ln,ln->l', self.u, x)[:, None]
        else:
            out = (self.base @ x.reshape(self.dim, -1)).reshape(x.shape)
            out -= self.u[:, :, None] * np.einsum('ln,lnk->lk', self.u, x)[:, None, :]
        if t:
            out += t * path_lap(x)
        if eps:
            out += eps * x
        return out

    def path_preconditioner(self, t, eps):
        inv = np.linalg.inv(t * self.path + np.diag(self.d + eps))
        return lambda x: inv @ x


def pcg(apply, b, precondition, tol=5e-13, maxiter=1600):
    """One RHS. Avoid threaded BLAS dispatch for the small reductions."""
    x = np.zeros_like(b)
    r = b.copy()
    dot = lambda u, v: np.einsum('ij,ij->', u, v)
    target = tol**2 * dot(b, b)
    if not target:
        return x, 0
    z = precondition(r)
    s = dot(r, z)
    p = z.copy()
    for i in range(1, maxiter + 1):
        ap = apply(p)
        alpha = s / dot(p, ap)
        x += alpha * p
        r -= alpha * ap
        if dot(r, r) <= target:
            return x, i
        z = precondition(r)
        s1 = dot(r, z)
        p *= s1 / s
        p += z
        s = s1
    raise RuntimeError(f'PCG failed after {maxiter} steps (recursive relative residual '
                       f'{np.sqrt(dot(r, r) / dot(b, b)):g})')


class SparseInverse:
    def __init__(self, joint, t, tol=5e-13):
        self.joint, self.t, self.tol = joint, t, tol
        self.eps = 1e-3 * min(1., t if t > 0 else 1., joint.min_inv_scale)
        self.precondition = joint.path_preconditioner(t, self.eps)
        self.counts = []
        self.max_residual = 0.

    def solve(self, b):
        assert b.ndim == 3 and b.shape[-1] == 1
        rhs = centre(b[:, :, 0])
        apply = lambda v: self.joint.apply(v, self.t, self.eps)
        x, count = pcg(apply, rhs, self.precondition, tol=self.tol)
        self.counts.append(count)
        self.max_residual = max(self.max_residual, np.linalg.norm((apply(x) - rhs).ravel()) /
                                np.linalg.norm(rhs.ravel()))
        return x[:, :, None]


class SparseDirect:
    """Small-N diagnostic: exact joint SuperLU plus a rank-L update.

    Sparse does not imply a sparse factor. Do not use at large N without a
    fill estimate or a separately enforced memory budget.
    """
    def __init__(self, joint, t):
        l, n = self.shape = joint.shape
        self.eps = 1e-3 * min(1., t if t > 0 else 1., joint.min_inv_scale)
        base = joint.base + sparse.kron(t * joint.path, sparse.eye(n), format='csr')
        base += self.eps * sparse.eye(l*n, format='csr')
        self.lu = splu(base.tocsc(), permc_spec='MMD_AT_PLUS_A',
                       diag_pivot_thresh=0., options={'SymmetricMode': True})
        u = np.zeros((l, n, l))
        u[np.arange(l), :, np.arange(l)] = joint.u
        self.u = u.reshape(l*n, l)
        self.z = self.lu.solve(self.u)
        self.factor = cholesky(np.eye(l) - self.u.T @ self.z, lower=True)

    def solve(self, b):
        l, n = self.shape
        y = self.lu.solve(b.reshape(l*n, -1))
        return (y + self.z @ cho_solve((self.factor, True), self.u.T @ y)).reshape(b.shape)


class CholeskyBlocks:
    """Use SPD factor + triangular solves; still form each previous inverse
    transiently for the next dense Schur complement (coupling block is -tI)."""
    def __init__(self, ms, c, t, dtype=np.float64, retain_inverse=False):
        l, n, _ = ms.shape
        self.shape, self.t = (l, n), t
        self.eps = 1e-3 * min(1., t if t > 0 else 1., (1 / c).min())
        self.factors = []
        self.inverse = [] if retain_inverse else None
        idx = np.diag_indices(n)
        lower = np.tril_indices(n, -1) if retain_inverse else None
        for i in range(l):
            d = np.array(-ms[i] / c[i], dtype=dtype, order='F')
            d[idx] += 1 / c[i] + (int(i > 0) + int(i < l - 1)) * t + self.eps
            if i:
                d -= t * t * inverse
            f = cholesky(d, lower=True, overwrite_a=True, check_finite=False)
            if not retain_inverse:
                self.factors.append(f)
            if i < l - 1 or retain_inverse:
                inverse, info = get_lapack_funcs('potri', (f,))(f, lower=True,
                                                               overwrite_c=retain_inverse)
                if info != 0:
                    raise np.linalg.LinAlgError(f'potri: {info}')
                # Only the lower triangle is used by the next cholesky call.
                if retain_inverse:
                    inverse[lower[1], lower[0]] = inverse[lower]
                    self.inverse.append(inverse)

    def solve(self, b):
        def subsolve(l, x):
            if self.inverse is not None:
                return self.inverse[l] @ x
            return cho_solve((self.factors[l], True), x, check_finite=False)

        t = self.t
        y = [subsolve(0, b[0])]
        for l in range(1, len(b)):
            y.append(subsolve(l, b[l] + t * y[-1]))
        x = [y[-1]]
        for l in range(len(b) - 2, -1, -1):
            x.append(y[l] + t * subsolve(l, x[-1]))
        return np.stack(x[::-1])


def ritz(v, shape, t, apply, batch=16):
    """In-place centering and a small generalized Ritz problem.

    ARPACK vectors are already very nearly orthonormal. The Gram matrix
    accounts for centering without a tall QR, and the operator is applied
    to only one batch of RHS at a time.
    """
    l, n = shape
    k = v.shape[1]
    q = v.reshape(l, n, k)
    q -= q.mean(axis=1, keepdims=True)
    flat = q.reshape(l*n, k)
    gram = flat.T @ flat
    h = np.empty((k, k))
    for lo in range(0, k, batch):
        hi = min(lo + batch, k)
        h[:, lo:hi] = flat.T @ apply(q[:, :, lo:hi], t).reshape(l*n, -1)
    w, r = eigh((h + h.T) * .5, (gram + gram.T) * .5, check_finite=False)
    return w, (flat @ r).reshape(l, n, k)


def modes(shape, t, k, solve, apply, ncv=None, tol=1e-10):
    l, n = shape
    v0 = centre(np.random.default_rng(0).standard_normal((l, n, 1))).ravel()
    calls = 0

    def mv(x):
        nonlocal calls
        calls += 1
        return centre(solve(centre(x.reshape(l, n, 1)))).ravel()

    op = LinearOperator((l*n, l*n), matvec=mv, dtype=np.float64)
    _, v = eigsh(op, k=k, which='LA', tol=tol, v0=v0, ncv=ncv or max(4*k, 40))
    w, u = ritz(v, shape, t, apply)
    return w, u, calls


def normal_modes(joint, t, k, ncv=None, tol=1e-10):
    """No nested solve; keep all temporal-only eigenvalues above the target."""
    l, n = joint.shape
    barrier = (joint.upper + 4*t) * 1.01
    v0 = centre(np.random.default_rng(0).standard_normal((l, n))).ravel()
    calls = 0

    def mv(a):
        nonlocal calls
        calls += 1
        x = a.reshape(l, n)
        xc = centre(x)
        return (centre(joint.apply(xc, t)) + barrier * (x - xc)).ravel()

    w, v = eigsh(LinearOperator((l*n, l*n), matvec=mv, dtype=np.float64), v0=v0,
                  k=k, ncv=ncv or max(4*k, 40), which='SA', tol=tol)
    return w, v.reshape(l, n, k), calls


def power_modes(joint, t, k, ncv=None, degree=12, tol=2e-12):
    """Exact-spectrum polynomial transform; trade SpMVs for fewer restarts.

    (I - L/b)^degree is monotone on the spatial spectrum in [0,b].
    Projection sets the temporal-only modes to zero. No kernel approximation.
    """
    l, n = joint.shape
    bound = 1.01 * (joint.upper + 4*t)
    mvcount = 0
    v0 = centre(np.random.default_rng(0).standard_normal((l, n))).ravel()

    def mv(a):
        nonlocal mvcount
        mvcount += 1
        x = centre(a.reshape(l, n))
        for _ in range(degree):
            x -= joint.apply(x, t) / bound
        return centre(x).ravel()

    _, v = eigsh(LinearOperator((l*n, l*n), matvec=mv, dtype=np.float64), v0=v0,
                  k=k, ncv=ncv or max(4*k, 40), which='LA', tol=tol)
    w, u = ritz(v, joint.shape, t, joint.apply)
    return w, u, mvcount


def block_modes(joint, t, k, maxiter=200, guard=4):
    l, n = joint.shape
    x = centre(np.random.default_rng(0).standard_normal((l, n, k + guard))).reshape(l*n, -1)
    y = np.repeat(np.eye(l), n, axis=0) / np.sqrt(n)
    prec = joint.path_preconditioner(t, 0.)

    def op(a):
        return centre(joint.apply(a.reshape(l, n, -1), t)).reshape(a.shape)

    def pre(a):
        return prec(a.reshape(l, -1)).reshape(a.shape)

    w, v, resid = lobpcg(op, x, M=pre, Y=y, tol=5e-11, maxiter=maxiter,
                          largest=False, retResidualNormsHistory=True)
    return w[:k], v[:, :k].reshape(l, n, k), len(resid)


def site_spectra(walks, k=32, tail=9, tol=1e-11):
    """Nonconstant top eigenpairs + bottom eigenvalues + exact nonlazy Frobenius."""
    tops, vecs, tails, fro = [], [], [], []
    for w in walks:
        n = w.u.size
        op = LinearOperator((n, n), matvec=lambda x: w @ x, dtype=np.float64)
        rng = np.random.default_rng(1)
        v0 = rng.standard_normal(n)
        value, vec = eigsh(op, k=k + 1, which='LA', tol=tol, v0=v0)
        bot = eigsh(op, k=tail, which='SA', tol=tol, v0=v0, return_eigenvectors=False)
        base = 2 * w.base - sparse.eye(n, format='csr')
        u = w.u * np.sqrt(2.)
        fro.append(np.dot(base.data, base.data) + 2 * (u @ (base @ u)) + (u @ u)**2 - 1.)
        tops.append(value[::-1]); vecs.append(vec[:, -2::-1]); tails.append(np.sort(bot))
    return np.array(tops), np.array(vecs), np.array(tails), np.array(fro)


def balance_root(joint, gaps, leading):
    """Bracket t*=lambda_min(H + t*path)/tau using spatial trial functions.

    Lower bound: min site gap. Upper bound: test depth-constant centered
    copies of each leading site mode, so the depth energy is zero.
    """
    l, n = joint.shape
    tau = 4 * np.sin(np.pi / (2*l))**2
    f = np.broadcast_to(leading.T[None, :, :], (l, n, l)).copy()
    f = centre(f)
    energy = np.einsum('lnk,lnk->k', f, joint.apply(f, 0.))
    energy /= np.einsum('lnk,lnk->k', f, f)
    lo, hi = 0.5 * np.min(gaps) / tau, 1.01 * np.min(energy) / tau
    if not 0 < lo < hi:
        raise ValueError('Need connected sites and a positive bracket')
    counts = []

    def fun(t):
        w, _, calls = normal_modes(joint, t, 1, ncv=24, tol=2e-11)
        counts.append(calls)
        return t * tau - w[0]

    return brentq(fun, lo, hi, xtol=1e-13, rtol=3e-13), counts


class Calibration:
    """Solve B v = mu H v with H the site Laplacians and B=tau*I-path.

    Regularize constants in H's mass, zero them in B.  Invert H on a
    centered RHS by solving with its sparse part then centering the solution:
    Hbase*1=u*(u.T*1), so u.T*(Hbase^-1 b)=0 when 1.T*b=0.
    """
    def __init__(self, joint, method='cg'):
        self.joint, self.method = joint, method
        self.counts = []
        if method == 'lu':
            self.factors = [splu(b.tocsc(), permc_spec='MMD_AT_PLUS_A',
                                 diag_pivot_thresh=0., options={'SymmetricMode': True}) for b in joint.h]
        elif method != 'cg':
            raise ValueError(method)

    def balance(self, tol=1e-10, ncv=48):
        l, n = self.joint.shape
        tau = 4 * np.sin(np.pi / (2*l))**2
        joint = self.joint
        shape = (joint.dim, joint.dim)

        def bprod(x):
            f = centre(x.reshape(l, n))
            return (tau * f - path_lap(f)).ravel()

        def mass(x):
            f = x.reshape(l, n)
            fc = centre(f)
            return (centre(joint.apply(fc, 0.)) + (f - fc)).ravel()

        def minv(x):
            f = x.reshape(l, n)
            fc = centre(f)
            if self.method == 'lu':
                y = np.stack([a.solve(v) for a, v in zip(self.factors, fc)])
                self.counts.append(1)
            else:
                qinv = joint.path_preconditioner(0., 0.)
                y, nstep = pcg(lambda z: centre(joint.apply(z, 0.)), fc, qinv)
                self.counts.append(nstep)
            return (centre(y) + (f - fc)).ravel()

        v0 = centre(np.random.default_rng(17).standard_normal((l, n))).ravel()
        w = eigsh(LinearOperator(shape, bprod, dtype=np.float64),
                  M=LinearOperator(shape, mass, dtype=np.float64),
                  Minv=LinearOperator(shape, minv, dtype=np.float64),
                  k=1, which='LA', ncv=ncv, tol=tol, v0=v0, return_eigenvectors=False)
        return 1. / w[0]


def error(ref, test, cluster_tol=1e-8):
    """Max relative eigenvalue error, sin theta max (all and by cluster)."""
    w, v = ref
    z, u = test
    k = len(w)
    v, u = v.reshape(-1, k), u.reshape(-1, k)
    gap = np.diff(w) / np.maximum(np.abs(w[:-1]), np.abs(w[1:]))
    stops = np.r_[0, np.flatnonzero(gap > cluster_tol) + 1, k]
    def max_sin(a, b):
        qa = np.linalg.qr(a, mode='reduced')[0]
        qb = np.linalg.qr(b, mode='reduced')[0]
        # Project explicitly rather than sqrt(1-cos(theta)**2), which
        # cannot resolve the small angles of interest in this benchmark.
        residual = qb - qa @ (qa.T @ qb)
        return float(np.sqrt(max(0., np.linalg.eigvalsh(residual.T @ residual)[-1])))

    group = max(max_sin(v[:, a:b], u[:, a:b]) for a, b in zip(stops[:-1], stops[1:]))
    overall = max_sin(v, u)
    return float(np.max(np.abs(z / w - 1))), overall, group, float(np.min(gap)) if k > 1 else np.inf
