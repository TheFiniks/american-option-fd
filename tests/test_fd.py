"""Tests of the finite-difference solver against analytic and binomial references."""

import numpy as np
import pytest

from americanfd import Option
from americanfd.binomial import binomial_price
from americanfd.black_scholes import bs_greeks, bs_price, perpetual_american_put
from americanfd.fd import solve_fd

EX_1_6 = Option(K=50.0, T=5 / 12, r=0.1, sigma=0.4, kind="put", style="american")  # Seydel
TAB_4_1 = Option(K=10.0, T=1.0, r=0.25, sigma=0.6, div=0.2, kind="put", style="american")


# ----------------------------------------------------------------------------- European


@pytest.mark.parametrize("kind", ["call", "put"])
@pytest.mark.parametrize("div", [0.0, 0.05])
def test_crank_nicolson_second_order_for_european(kind, div):
    opt = Option(K=100.0, T=0.5, r=0.05, sigma=0.25, div=div, kind=kind, style="european")
    exact = bs_price(100.0, opt)
    err = [abs(solve_fd(opt, m, m).price(100.0) - exact) for m in (100, 200)]
    assert err[1] < 5e-3
    assert 3.3 < err[0] / err[1] < 4.7  # halving the grid -> error / 4


def test_european_greeks_match_black_scholes():
    opt = Option(K=100.0, T=0.5, r=0.05, sigma=0.25, div=0.02, kind="put", style="european")
    res = solve_fd(opt, 400, 400)
    mask = (res.S > 70) & (res.S < 140)
    exact = bs_greeks(res.S[mask], opt)
    assert np.abs(res.V[mask] - exact.price).max() < 1e-3
    assert np.abs(res.delta[mask] - exact.delta).max() < 1e-4
    assert np.abs(res.gamma[mask] - exact.gamma).max() < 1e-4
    assert np.abs(res.theta[mask] - exact.theta).max() < 2e-3


def test_rannacher_removes_gamma_oscillations():
    # large lam = dtau/dx^2: pure Crank–Nicolson rings at the strike
    opt = Option(K=100.0, T=0.25, r=0.05, sigma=0.2, kind="put", style="european")
    pure = solve_fd(opt, 800, 25, rannacher=0)
    smooth = solve_fd(opt, 800, 25, rannacher=2)
    mask = (pure.S > 80) & (pure.S < 120)
    exact = bs_greeks(pure.S[mask], opt).gamma
    err_pure = np.abs(pure.gamma[mask] - exact).max()
    err_smooth = np.abs(smooth.gamma[mask] - exact).max()
    assert err_pure > 0.1  # bigger than Gamma itself (~0.08)
    assert err_smooth < 1e-4


# ----------------------------------------------------------------------------- American


@pytest.mark.parametrize("m, expected", [(50, 1.8562637), (100, 1.8752110), (200, 1.8800368)])
def test_reproduces_seydel_table_4_1(m, expected):
    # Seydel's setting: x in [-5, 5], m = nu_max, plain Crank–Nicolson
    res = solve_fd(TAB_4_1, m, m, half_width=5.0, rannacher=0)
    assert res.price(10.0) == pytest.approx(expected, abs=1e-7)


def test_psor_matches_brennan_schwartz():
    a = solve_fd(EX_1_6, 200, 200, lcp="psor")
    b = solve_fd(EX_1_6, 200, 200, lcp="brennan-schwartz")
    np.testing.assert_allclose(a.V, b.V, atol=1e-8)
    np.testing.assert_allclose(a.boundary_S, b.boundary_S)


def test_american_put_matches_binomial():
    ref = binomial_price(50.0, EX_1_6, n=4000)  # 4.28422
    assert solve_fd(EX_1_6, 400, 400).price(50.0) == pytest.approx(ref, abs=1e-3)


def test_american_put_dominates_european_and_payoff():
    res = solve_fd(EX_1_6, 200, 200)
    eur = bs_price(res.S, EX_1_6)
    inner = slice(1, -1)
    assert np.all(res.V[inner] >= eur[inner] - 1e-6)
    assert np.all(res.V[inner] >= EX_1_6.payoff(res.S[inner]) - 1e-12)


def test_american_call_without_dividends_is_european():
    opt = Option(K=100.0, T=1.0, r=0.05, sigma=0.2, kind="call", style="american")
    am = solve_fd(opt, 200, 200)
    eu = solve_fd(opt.with_(style="european"), 200, 200)
    np.testing.assert_allclose(am.V, eu.V, atol=1e-8)
    assert np.isnan(am.exercise_boundary_t0)


def test_put_call_symmetry():
    # C(S, K, r, div) = P(K, S, div, r)  (McDonald–Schroder; Seydel A5.3), here S = K.
    # Holds to round-off when both problems use the same x-grid.
    put = solve_fd(TAB_4_1, 400, 400, half_width=5.0).price(10.0)
    call = solve_fd(TAB_4_1.with_(kind="call", r=0.2, div=0.25), 400, 400, half_width=5.0)
    assert put == pytest.approx(call.price(10.0), abs=1e-8)


def test_seydel_american_call_value():
    # Seydel Fig. 4.9: K=10, r=0.25, sigma=0.6, T=1, div=0.2 -> V(K, 0) = 2.18728
    opt = TAB_4_1.with_(kind="call")
    assert solve_fd(opt, 400, 400).price(10.0) == pytest.approx(2.18728, abs=3e-4)


def test_long_dated_put_tends_to_perpetual():
    opt = Option(K=10.0, T=100.0, r=0.06, sigma=0.3, kind="put", style="american")
    res = solve_fd(opt, 1600, 400, half_width=8.0, lcp="brennan-schwartz")
    value, s_f = perpetual_american_put(10.0, 10.0, 0.06, 0.3)
    assert res.price(10.0) == pytest.approx(float(value), abs=3e-4)
    assert res.exercise_boundary_t0 == pytest.approx(s_f, abs=0.05)


def test_exercise_region_greeks_and_smooth_pasting():
    res = solve_fd(EX_1_6, 400, 400)
    s_f = res.exercise_boundary_t0
    assert 34.0 < s_f < 38.0  # Seydel reports S_f(0) = 36.3
    deep = (res.S < 0.95 * s_f) & (res.S > 20.0)
    np.testing.assert_allclose(res.delta[deep], -1.0, atol=1e-8)
    np.testing.assert_allclose(res.gamma[deep], 0.0, atol=1e-8)
    np.testing.assert_allclose(res.theta[deep], 0.0, atol=1e-8)
    # smooth pasting: Delta is continuous and equals -1 at the boundary
    near = np.argmin(np.abs(res.S - s_f))
    assert res.delta[near + 1] == pytest.approx(-1.0, abs=0.02)


def test_exercise_boundary_shape():
    res = solve_fd(EX_1_6, 400, 400)
    t, s_f = res.boundary_t, res.boundary_S
    assert np.all(np.diff(t) <= 0)  # levels go backwards in calendar time
    assert np.all(s_f <= EX_1_6.K)
    assert s_f[-1] < s_f[0]  # boundary drops away from expiry


def test_american_greeks_match_binomial_bump():
    S, h = 50.0, 0.5
    up, mid, dn = (binomial_price(s, EX_1_6, n=4000) for s in (S + h, S, S - h))
    g = solve_fd(EX_1_6, 800, 400, lcp="brennan-schwartz").greeks(S)
    assert g.delta == pytest.approx((up - dn) / (2 * h), abs=2e-3)
    assert g.gamma == pytest.approx((up - 2 * mid + dn) / h**2, abs=2e-3)


def test_input_validation():
    with pytest.raises(ValueError):
        Option(K=-1.0, T=1.0, r=0.05, sigma=0.2)
    with pytest.raises(ValueError):
        solve_fd(EX_1_6, 100, 100, lcp="gauss")
    with pytest.raises(ValueError):
        solve_fd(EX_1_6, 100, 100, theta=1.5)
