"""Contract and market parameters shared by every pricing method."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

Kind = Literal["call", "put"]
Style = Literal["european", "american"]


@dataclass(frozen=True)
class Option:
    """A vanilla option in the Black–Scholes market with a continuous dividend yield.

    Parameters
    ----------
    K : strike
    T : time to expiry in years
    r : risk-free rate (continuous compounding)
    sigma : volatility
    div : continuous dividend yield (Seydel's delta)
    kind : "call" or "put"
    style : "european" or "american"
    """

    K: float
    T: float
    r: float
    sigma: float
    div: float = 0.0
    kind: Kind = "put"
    style: Style = "american"

    def __post_init__(self) -> None:
        if self.K <= 0 or self.T <= 0 or self.sigma <= 0:
            raise ValueError("K, T and sigma must be positive")
        if self.kind not in ("call", "put"):
            raise ValueError(f"kind must be 'call' or 'put', got {self.kind!r}")
        if self.style not in ("european", "american"):
            raise ValueError(f"style must be 'european' or 'american', got {self.style!r}")

    @property
    def is_call(self) -> bool:
        return self.kind == "call"

    @property
    def is_american(self) -> bool:
        return self.style == "american"

    def payoff(self, S):
        S = np.asarray(S, dtype=float)
        return np.maximum(S - self.K, 0.0) if self.is_call else np.maximum(self.K - S, 0.0)

    def with_(self, **changes) -> Option:
        """Copy with some fields changed, e.g. ``opt.with_(style="european")``."""
        return replace(self, **changes)
