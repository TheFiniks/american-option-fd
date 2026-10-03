"""Solvers for the tridiagonal linear complementarity problem (LCP)

    A w >= b,   w >= g,   (A w - b)^T (w - g) = 0,

that arises at every time step of an American option (Seydel §4.6).  ``A`` is
tridiagonal and given by three arrays of length n:

    lower[i] * w[i-1] + diag[i] * w[i] + upper[i] * w[i+1]      (lower[0], upper[-1] unused)

Three solvers:

* :func:`psor` — projected SOR (Cryer 1971; Seydel Alg. 4.11/4.13).  Iterative,
  works for any monotone exercise region; this is the method of the project.
* :func:`brennan_schwartz` — direct elimination with projection (Brennan &
  Schwartz 1977; Seydel Exercise 4.12).  Exact for a single contiguous
  exercise region (true for vanilla puts/calls), O(n) per step, no iterations.
* :func:`thomas` — plain tridiagonal solve, used for European options.

The kernels are written so that they run unchanged either under ``numba.njit``
(``pip install americanfd[fast]``) or as plain Python on lists.
"""

from __future__ import annotations

import numpy as np

try:  # optional JIT
    from numba import njit

    HAVE_NUMBA = True
except ImportError:  # pragma: no cover - depends on the environment
    HAVE_NUMBA = False


# --------------------------------------------------------------------------- kernels


def _psor_kernel(lower, diag, upper, b, g, v, omega, tol, max_iter):
    """In-place projected SOR on ``v`` (initial guess, must satisfy v >= g).

    Returns the number of sweeps; ``max_iter + 1`` signals non-convergence.
    """
    n = len(b)
    for k in range(1, max_iter + 1):
        err = 0.0
        scale = 1.0
        for i in range(n):
            s = b[i]
            if i > 0:
                s -= lower[i] * v[i - 1]  # already updated: Gauss–Seidel
            if i < n - 1:
                s -= upper[i] * v[i + 1]
            rho = s / diag[i]  # Gauss–Seidel value
            new = v[i] + omega * (rho - v[i])  # over-relaxation
            if new < g[i]:  # projection onto w >= g
                new = g[i]
            d = abs(new - v[i])
            if d > err:
                err = d
            if abs(new) > scale:
                scale = abs(new)
            v[i] = new
        if err <= tol * scale:
            return k
    return max_iter + 1


def _brennan_schwartz_kernel(lower, diag, upper, b, g, w, exercise_at_left, project):
    """Direct solve with projection, written into ``w``.

    Put (exercise region at small indices): eliminate the super-diagonal from
    the bottom up, then substitute forward from i = 0 and project as we go.
    Call: classical Thomas (eliminate sub-diagonal), substitute backward.
    """
    n = len(b)
    dd = [0.0] * n
    bb = [0.0] * n
    if exercise_at_left:
        dd[n - 1] = diag[n - 1]
        bb[n - 1] = b[n - 1]
        for i in range(n - 2, -1, -1):
            f = upper[i] / dd[i + 1]
            dd[i] = diag[i] - f * lower[i + 1]
            bb[i] = b[i] - f * bb[i + 1]
        prev = 0.0
        for i in range(n):
            val = bb[i] / dd[i] if i == 0 else (bb[i] - lower[i] * prev) / dd[i]
            if project and val < g[i]:
                val = g[i]
            w[i] = val
            prev = val
    else:
        dd[0] = diag[0]
        bb[0] = b[0]
        for i in range(1, n):
            f = lower[i] / dd[i - 1]
            dd[i] = diag[i] - f * upper[i - 1]
            bb[i] = b[i] - f * bb[i - 1]
        nxt = 0.0
        for i in range(n - 1, -1, -1):
            val = bb[i] / dd[i] if i == n - 1 else (bb[i] - upper[i] * nxt) / dd[i]
            if project and val < g[i]:
                val = g[i]
            w[i] = val
            nxt = val


if HAVE_NUMBA:
    try:  # compile once on tiny inputs; fall back to pure Python if numba chokes
        _psor_jit = njit(cache=True)(_psor_kernel)
        _bs_jit = njit(cache=True)(_brennan_schwartz_kernel)
        _o, _t = np.ones(3), np.full(3, 4.0)
        _psor_jit(_o, _t, _o, _o, np.zeros(3), np.zeros(3), 1.0, 1e-12, 50)
        _bs_jit(_o, _t, _o, _o, np.zeros(3), np.empty(3), True, True)
    except Exception as exc:  # pragma: no cover
        import warnings

        warnings.warn(f"numba JIT failed ({exc}); using pure Python kernels", stacklevel=1)
        HAVE_NUMBA = False


# --------------------------------------------------------------------------- public API


def optimal_omega(alpha: float, size: int) -> float:
    """Optimal SOR factor for tridiag(-alpha, 1 + 2 alpha, -alpha) of order ``size``.

    The Jacobi iteration matrix has spectral radius
    mu = 2 alpha cos(pi / (size + 1)) / (1 + 2 alpha), and for this consistently
    ordered matrix Young's formula gives omega* = 2 / (1 + sqrt(1 - mu^2)).
    With the obstacle active and a warm start from the previous time level PSOR
    is no longer exactly SOR: for moderate lam a fixed omega of 1.2-1.4 can need
    fewer sweeps, but omega* is the robust default - for large lam it is close
    to the best fixed value, while omega = 1 needs several times more sweeps
    (see scripts/psor_omega.py and results/psor_omega.md).
    """
    mu = 2.0 * alpha * np.cos(np.pi / (size + 1)) / (1.0 + 2.0 * alpha)
    return float(2.0 / (1.0 + np.sqrt(1.0 - mu * mu)))


class PSORNotConverged(RuntimeWarning):
    pass


def psor(lower, diag, upper, b, g, x0=None, omega=1.2, tol=1e-10, max_iter=10_000):
    """Projected SOR.  Returns ``(w, n_sweeps)``.

    ``x0`` is the warm start (previous time level); it is projected onto
    ``w >= g`` first.  ``omega`` in [1, 2) is the relaxation factor; the stopping
    rule is ``max|w_new - w_old| <= tol * max(1, max|w|)``.
    """
    if not 0.0 < omega < 2.0:
        raise ValueError("omega must lie in (0, 2)")
    g = np.asarray(g, dtype=float)
    v = np.maximum(np.asarray(b if x0 is None else x0, dtype=float), g)
    if HAVE_NUMBA:
        sweeps = _psor_jit(
            np.asarray(lower, float), np.asarray(diag, float), np.asarray(upper, float),
            np.asarray(b, float), g, v, float(omega), float(tol), int(max_iter),
        )  # fmt: skip
        w = v
    else:
        vl = v.tolist()
        sweeps = _psor_kernel(
            np.asarray(lower, float).tolist(), np.asarray(diag, float).tolist(),
            np.asarray(upper, float).tolist(), np.asarray(b, float).tolist(), g.tolist(),
            vl, float(omega), float(tol), int(max_iter),
        )  # fmt: skip
        w = np.array(vl)
    if sweeps > max_iter:
        import warnings

        warnings.warn(f"PSOR did not converge in {max_iter} sweeps", PSORNotConverged, stacklevel=2)
    return w, int(sweeps)


def brennan_schwartz(lower, diag, upper, b, g, exercise_at_left: bool):
    """Brennan–Schwartz direct LCP solve.  ``exercise_at_left=True`` for a put."""
    return _direct(lower, diag, upper, b, g, exercise_at_left, project=True)


def thomas(lower, diag, upper, b):
    """Plain tridiagonal solve (no constraint)."""
    n = len(b)
    return _direct(lower, diag, upper, b, np.zeros(n), False, project=False)


def _direct(lower, diag, upper, b, g, exercise_at_left, project):
    arrays = [np.asarray(a, dtype=float) for a in (lower, diag, upper, b, g)]
    if HAVE_NUMBA:
        w = np.empty(len(b))
        _bs_jit(*arrays, w, bool(exercise_at_left), bool(project))
        return w
    w = [0.0] * len(b)
    _brennan_schwartz_kernel(*(a.tolist() for a in arrays), w, exercise_at_left, project)
    return np.array(w)
