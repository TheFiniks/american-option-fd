"""Tests for the reference methods: Black–Scholes formula and binomial trees."""

import numpy as np
import pytest

from americanfd import Option
from americanfd.binomial import binomial_price
from americanfd.black_scholes import bs_greeks, bs_price, perpetual_american_put

EUR_PUT = Option(K=100.0, T=1.0, r=0.05, sigma=0.2, div=0.0, kind="put", style="european")
EUR_CALL = EUR_PUT.with_(kind="call")


def test_bs_textbook_value():
    # Hull's classic example: S=K=100, r=5%, sigma=20%, T=1  ->  call 10.4506, put 5.5735
    assert bs_price(100.0, EUR_CALL) == pytest.approx(10.450584, abs=1e-5)
    assert bs_price(100.0, EUR_PUT) == pytest.approx(5.573526, abs=1e-5)


@pytest.mark.parametrize("div", [0.0, 0.03, 0.1])
def test_put_call_parity(div):
    S = np.linspace(50.0, 150.0, 11)
    call = bs_price(S, EUR_CALL.with_(div=div))
    put = bs_price(S, EUR_PUT.with_(div=div))
    rhs = S * np.exp(-div * 1.0) - 100.0 * np.exp(-0.05 * 1.0)
    np.testing.assert_allclose(call - put, rhs, atol=1e-12)


@pytest.mark.parametrize("kind", ["call", "put"])
def test_bs_greeks_match_finite_differences(kind):
    opt = EUR_PUT.with_(kind=kind, div=0.02)
    S, h, k = 95.0, 1e-3, 1e-5
    g = bs_greeks(S, opt)
    assert g.price == pytest.approx(bs_price(S, opt))
    assert g.delta == pytest.approx(
        (bs_price(S + h, opt) - bs_price(S - h, opt)) / (2 * h), abs=1e-7
    )
    assert g.gamma == pytest.approx(
        (bs_price(S + h, opt) - 2 * bs_price(S, opt) + bs_price(S - h, opt)) / h**2, abs=1e-4
    )
    # dV/dt = -dV/dtau
    theta_fd = -(bs_price(S, opt, tau=1.0 + k) - bs_price(S, opt, tau=1.0 - k)) / (2 * k)
    assert g.theta == pytest.approx(theta_fd, abs=1e-6)


def test_bs_greeks_satisfy_the_pde():
    # Theta + 1/2 sigma^2 S^2 Gamma + (r - div) S Delta - r V = 0
    opt = EUR_CALL.with_(div=0.04)
    S = np.linspace(60.0, 140.0, 9)
    g = bs_greeks(S, opt)
    residual = (
        g.theta
        + 0.5 * opt.sigma**2 * S**2 * g.gamma
        + (opt.r - opt.div) * S * g.delta
        - opt.r * g.price
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-10)


@pytest.mark.parametrize("method", ["crr", "bbs", "bbsr"])
@pytest.mark.parametrize("kind", ["call", "put"])
def test_binomial_european_converges_to_bs(method, kind):
    opt = EUR_PUT.with_(kind=kind, div=0.02)
    assert binomial_price(100.0, opt, n=2000, method=method) == pytest.approx(
        bs_price(100.0, opt), abs=2e-3
    )


def test_bbsr_american_put_seydel_example_1_6():
    # Seydel, Example 1.6 / Fig. 4.11: K=50, r=0.1, sigma=0.4, T=5/12, S=50 -> 4.2842
    opt = Option(K=50.0, T=5 / 12, r=0.1, sigma=0.4, kind="put", style="american")
    assert binomial_price(50.0, opt, n=4000) == pytest.approx(4.2842, abs=2e-4)


def test_american_call_without_dividends_equals_european():
    opt = Option(K=100.0, T=1.0, r=0.05, sigma=0.2, div=0.0, kind="call", style="american")
    assert binomial_price(100.0, opt, n=1000, method="crr") == pytest.approx(
        binomial_price(100.0, opt.with_(style="european"), n=1000, method="crr"), abs=1e-12
    )


def test_american_put_dominates_european_and_payoff():
    opt = Option(K=100.0, T=1.0, r=0.08, sigma=0.25, kind="put", style="american")
    for S in (70.0, 90.0, 100.0, 120.0):
        am = binomial_price(S, opt, n=1000)
        assert am >= bs_price(S, opt) - 1e-6
        assert am >= opt.K - S - 1e-6


def test_perpetual_put_smooth_pasting():
    K, r, sigma = 10.0, 0.06, 0.3
    _, s_f = perpetual_american_put(1.0, K, r, sigma)
    h = 1e-6
    v_plus, _ = perpetual_american_put(s_f + h, K, r, sigma)
    v0, _ = perpetual_american_put(s_f, K, r, sigma)
    assert v0 == pytest.approx(K - s_f)
    assert (v_plus - v0) / h == pytest.approx(-1.0, abs=1e-5)  # high contact
    assert s_f == pytest.approx(K * (2 * r / sigma**2) / (1 + 2 * r / sigma**2))
