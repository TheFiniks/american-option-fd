"""Binomial reference prices: Cox–Ross–Rubinstein and BBSR.

* ``crr``  — classical CRR tree (Seydel §1.4, Shiryaev ch. VI §4–5).  Converges
  at rate O(1/N) but oscillates (odd/even N), which makes it a noisy reference.
* ``bbs``  — Broadie–Detemple "binomial Black–Scholes": the last step uses the
  Black–Scholes formula instead of the payoff, which smooths the convergence.
* ``bbsr`` — BBS plus Richardson extrapolation, ``2 V(N) - V(N/2)``.  This is a
  standard, accurate benchmark for American options and is what the tests and
  convergence study use as "exact" American value.

Early exercise is ``max(continuation, payoff)`` at every node.
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from .black_scholes import bs_price
from .option import Option

Method = Literal["crr", "bbs", "bbsr"]


def _tree(S0: float, opt: Option, n: int, smooth_last_step: bool) -> float:
    if n < 1:
        raise ValueError("need at least one time step")
    dt = opt.T / n
    u = np.exp(opt.sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((opt.r - opt.div) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError(f"risk-neutral probability p={p:.4f} outside (0,1): increase n")
    disc = np.exp(-opt.r * dt)

    def stock(i: int) -> np.ndarray:  # stock prices at time step i, j = 0..i up-moves
        return S0 * u ** (2.0 * np.arange(i + 1) - i)

    if smooth_last_step:
        # values at step n-1 from the European formula with one step to expiry
        S = stock(n - 1)
        V = bs_price(S, opt, tau=dt)
        if opt.is_american:
            V = np.maximum(V, opt.payoff(S))
        start = n - 2
    else:
        V = opt.payoff(stock(n))
        start = n - 1

    for i in range(start, -1, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        if opt.is_american:
            V = np.maximum(V, opt.payoff(stock(i)))
    return float(V[0])


def binomial_price(S0: float, opt: Option, n: int = 1000, method: Method = "bbsr") -> float:
    """Price of ``opt`` at spot ``S0`` on a binomial tree with ``n`` steps."""
    if method == "crr":
        return _tree(S0, opt, n, smooth_last_step=False)
    if method == "bbs":
        return _tree(S0, opt, n, smooth_last_step=True)
    if method == "bbsr":
        n = n + (n % 2)  # need an even number of steps
        return 2.0 * _tree(S0, opt, n, True) - _tree(S0, opt, n // 2, True)
    raise ValueError(f"unknown method {method!r}")
