"""Closed-form Black–Scholes–Merton prices and Greeks for European options.

Continuous dividend yield ``div``:

    d1 = (ln(S/K) + (r - div + sigma^2/2) T) / (sigma sqrt T),   d2 = d1 - sigma sqrt T
    call = S e^{-div T} N(d1) - K e^{-r T} N(d2)
    put  = K e^{-r T} N(-d2) - S e^{-div T} N(-d1)

Theta is returned as dV/dt (calendar time, per year), i.e. negative for a long
vanilla option in most cases.  This matches the convention used by the finite
difference solver.

These formulas are the "ground truth" for the European tests (Seydel, App. A4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import ndtr

from .option import Option

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _pdf(x):
    return np.exp(-0.5 * x * x) / _SQRT_2PI


def _d1_d2(S, opt: Option, tau: float):
    S = np.asarray(S, dtype=float)
    vol_sqrt = opt.sigma * np.sqrt(tau)
    d1 = (np.log(S / opt.K) + (opt.r - opt.div + 0.5 * opt.sigma**2) * tau) / vol_sqrt
    return d1, d1 - vol_sqrt


def bs_price(S, opt: Option, tau: float | None = None):
    """European price at spot ``S`` with time to expiry ``tau`` (default ``opt.T``).

    ``opt.style`` is ignored: the formula is always the European one.
    """
    tau = opt.T if tau is None else tau
    d1, d2 = _d1_d2(S, opt, tau)
    S = np.asarray(S, dtype=float)
    disc_S = S * np.exp(-opt.div * tau)
    disc_K = opt.K * np.exp(-opt.r * tau)
    if opt.is_call:
        return disc_S * ndtr(d1) - disc_K * ndtr(d2)
    return disc_K * ndtr(-d2) - disc_S * ndtr(-d1)


@dataclass(frozen=True)
class Greeks:
    price: np.ndarray | float
    delta: np.ndarray | float
    gamma: np.ndarray | float
    theta: np.ndarray | float  # dV/dt, per year


def bs_greeks(S, opt: Option, tau: float | None = None) -> Greeks:
    """Price, Delta, Gamma and Theta (= dV/dt) of a European option."""
    tau = opt.T if tau is None else tau
    d1, d2 = _d1_d2(S, opt, tau)
    S = np.asarray(S, dtype=float)
    eq = np.exp(-opt.div * tau)
    er = np.exp(-opt.r * tau)
    sqrt_tau = np.sqrt(tau)

    gamma = eq * _pdf(d1) / (S * opt.sigma * sqrt_tau)
    decay = -S * eq * _pdf(d1) * opt.sigma / (2.0 * sqrt_tau)
    if opt.is_call:
        price = S * eq * ndtr(d1) - opt.K * er * ndtr(d2)
        delta = eq * ndtr(d1)
        theta = decay - opt.r * opt.K * er * ndtr(d2) + opt.div * S * eq * ndtr(d1)
    else:
        price = opt.K * er * ndtr(-d2) - S * eq * ndtr(-d1)
        delta = -eq * ndtr(-d1)
        theta = decay + opt.r * opt.K * er * ndtr(-d2) - opt.div * S * eq * ndtr(-d1)
    return Greeks(price=price, delta=delta, gamma=gamma, theta=theta)


def perpetual_american_put(S, K: float, r: float, sigma: float, div: float = 0.0):
    """Perpetual (T = infinity) American put — closed form (Seydel, Exercise 4.8).

    Returns ``(value, S_f)``.  Useful as a test for the free-boundary solver at
    large maturities: the exercise boundary tends to ``S_f`` as T grows.
    """
    q = 2.0 * r / sigma**2
    q_div = 2.0 * (r - div) / sigma**2
    lam = 0.5 * (1.0 - q_div - np.sqrt((q_div - 1.0) ** 2 + 4.0 * q))  # negative root
    s_f = K * lam / (lam - 1.0)
    S = np.asarray(S, dtype=float)
    value = np.where(S <= s_f, K - S, (K - s_f) * (S / s_f) ** lam)
    return value, s_f
