"""American option pricing by finite differences (Crank–Nicolson + PSOR)."""

from .binomial import binomial_price
from .black_scholes import Greeks, bs_greeks, bs_price, perpetual_american_put
from .fd import FDResult, fd_price, solve_fd
from .option import Option

__all__ = [
    "FDResult",
    "Greeks",
    "Option",
    "binomial_price",
    "bs_greeks",
    "bs_price",
    "fd_price",
    "perpetual_american_put",
    "solve_fd",
]
__version__ = "0.1.0"
