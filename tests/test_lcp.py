"""Tests of the LCP solvers: projected SOR, Brennan–Schwartz and Thomas."""

import numpy as np
import pytest

from americanfd.lcp import brennan_schwartz, optimal_omega, psor, thomas


def _random_lcp(n=60, seed=0):
    rng = np.random.default_rng(seed)
    alpha = 3.0
    lower = np.full(n, -alpha)
    diag = np.full(n, 1 + 2 * alpha)
    b = rng.normal(size=n)
    g = np.maximum(1.0 - np.linspace(0, 2, n), 0.0)  # obstacle active on the left
    return lower, diag, lower.copy(), b, g


def _check_complementarity(lower, diag, upper, b, g, w, tol=1e-9):
    Aw = diag * w
    Aw[1:] += lower[1:] * w[:-1]
    Aw[:-1] += upper[:-1] * w[1:]
    assert np.all(w >= g - tol)
    assert np.all(Aw - b >= -tol)
    assert np.abs((Aw - b) * (w - g)).max() < tol


def test_thomas_solves_tridiagonal_system():
    lower, diag, upper, b, _ = _random_lcp()
    A = np.diag(diag) + np.diag(lower[1:], -1) + np.diag(upper[:-1], 1)
    np.testing.assert_allclose(thomas(lower, diag, upper, b), np.linalg.solve(A, b), atol=1e-12)


def test_psor_solves_a_general_lcp():
    # random right-hand side: the active set is scattered, PSOR still works
    lower, diag, upper, b, g = _random_lcp()
    w, sweeps = psor(lower, diag, upper, b, g, tol=1e-14)
    _check_complementarity(lower, diag, upper, b, g, w)
    assert sweeps < 200


@pytest.mark.parametrize("left", [True, False])
def test_brennan_schwartz_matches_psor_for_one_contact_region(left):
    # obstacle active on one contiguous block only (put: left, call: right):
    # the situation in which Brennan–Schwartz is exact
    lower, diag, upper, _, _ = _random_lcp()
    s = np.linspace(0.0, 2.0, len(diag))
    g = np.maximum(1.0 - s, 0.0) if left else np.maximum(s - 1.0, 0.0)
    b = 0.1 + 0.2 * np.exp(-((s - 1.0) ** 2))
    w_bs = brennan_schwartz(lower, diag, upper, b, g, exercise_at_left=left)
    w_ps, _ = psor(lower, diag, upper, b, g, tol=1e-15)
    _check_complementarity(lower, diag, upper, b, g, w_bs)
    np.testing.assert_allclose(w_bs, w_ps, atol=1e-11)
    assert (w_bs == g).any()


def test_optimal_omega_speeds_up_psor():
    lower, diag, upper, b, g = _random_lcp(n=200)
    om = optimal_omega(3.0, 200)
    assert 1.0 < om < 2.0
    _, it_gs = psor(lower, diag, upper, b, g, omega=1.0, tol=1e-12)
    _, it_opt = psor(lower, diag, upper, b, g, omega=om, tol=1e-12)
    assert it_opt < it_gs
