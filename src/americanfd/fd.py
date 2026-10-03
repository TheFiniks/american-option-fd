"""Finite-difference solver for European and American options (Seydel, ch. 4).

Pipeline
--------
1. Transform the Black–Scholes equation with dividend yield ``div``

       V_t + 1/2 sigma^2 S^2 V_SS + (r - div) S V_S - r V = 0

   into the heat equation  y_tau = y_xx  by (Seydel 4.3)

       S = K e^x,   t = T - 2 tau / sigma^2,
       V = K exp(-a x - b tau) y,   a = (q_d - 1)/2,   b = a^2 + q,
       q = 2 r / sigma^2,   q_d = 2 (r - div) / sigma^2.

   Constant coefficients make the scheme simple; the uniform x-grid is a
   log-grid in S, dense near small S.

2. Discretise with the theta-method on a uniform (x, tau) grid; theta = 1/2 is
   Crank–Nicolson, theta = 1 fully implicit.  Each step solves

       (1 + 2 lam theta) w_i - lam theta (w_{i-1} + w_{i+1})
           = w_i + lam (1 - theta) (w_{i+1} - 2 w_i + w_{i-1}) + boundary terms,

   with lam = dtau / dx^2.

3. American options: the step is the linear complementarity problem
   A w >= b, w >= g, (A w - b)^T (w - g) = 0 with g the transformed payoff
   (obstacle).  Solved by projected SOR or by Brennan–Schwartz.

4. Rannacher start-up: the payoff kink at the strike makes pure Crank–Nicolson
   produce slowly decaying oscillations, visible mostly in Gamma.  The first
   ``rannacher`` CN steps are replaced by ``2 * rannacher`` fully implicit half
   steps (Rannacher 1984; Giles & Carter 2006), which keeps second order.

5. Transform back and compute Delta, Gamma (three-point formulas on the
   non-uniform S-grid) and Theta (second-order one-sided difference in time).

Boundary conditions at the truncated ends x = +-L are taken in the original
variables and transformed: European values from put–call-parity asymptotics,
American values ``max(European asymptote, payoff)``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.interpolate import CubicSpline

from .black_scholes import Greeks
from .lcp import brennan_schwartz, optimal_omega, psor, thomas
from .option import Option

LCPSolver = Literal["psor", "brennan-schwartz"]


# --------------------------------------------------------------------------- transform


@dataclass(frozen=True)
class HeatTransform:
    """Change of variables (S, t, V) <-> (x, tau, y) of Seydel (4.3)."""

    opt: Option

    @property
    def a(self) -> float:
        return 0.5 * (2.0 * (self.opt.r - self.opt.div) / self.opt.sigma**2 - 1.0)

    @property
    def b(self) -> float:
        return self.a**2 + 2.0 * self.opt.r / self.opt.sigma**2

    @property
    def tau_max(self) -> float:
        return 0.5 * self.opt.sigma**2 * self.opt.T

    def remaining_time(self, tau):
        """T - t for a given tau."""
        return 2.0 * np.asarray(tau) / self.opt.sigma**2

    def to_y(self, V, x, tau):
        return V / self.opt.K * np.exp(self.a * x + self.b * tau)

    def to_V(self, y, x, tau):
        return self.opt.K * y * np.exp(-self.a * x - self.b * tau)

    def obstacle(self, x, tau):
        """Payoff in y-variables: g(x, tau) of Seydel, Problem 4.7."""
        return self.to_y(self.opt.payoff(self.opt.K * np.exp(x)), x, tau)

    def boundary_values(self, x_left: float, x_right: float, tau: float) -> tuple[float, float]:
        """y at the truncated ends of the grid at time level tau."""
        o = self.opt
        s = float(self.remaining_time(tau))
        S_lo, S_hi = o.K * np.exp(x_left), o.K * np.exp(x_right)
        disc_K, disc_div = o.K * np.exp(-o.r * s), np.exp(-o.div * s)
        if o.is_call:
            v_lo, v_hi = 0.0, S_hi * disc_div - disc_K
        else:
            v_lo, v_hi = disc_K - S_lo * disc_div, 0.0
        if o.is_american:
            v_lo = max(v_lo, float(o.payoff(S_lo)))
            v_hi = max(v_hi, float(o.payoff(S_hi)))
        return float(self.to_y(v_lo, x_left, tau)), float(self.to_y(v_hi, x_right, tau))


def default_half_width(opt: Option) -> float:
    """Half-width L of the x-interval [-L, L].

    Six standard deviations of ln S over the life of the option plus the drift,
    at least 1.  Truncation error is then far below the discretisation error.
    Seydel uses a fixed L = 5, which wastes nodes for short-dated options.
    """
    sd = opt.sigma * np.sqrt(opt.T)
    drift = abs(opt.r - opt.div - 0.5 * opt.sigma**2) * opt.T
    return float(max(1.0, 6.0 * sd + drift))


# --------------------------------------------------------------------------- result


@dataclass
class FDResult:
    """Solution on the grid at t = 0 plus diagnostics.

    Arrays ``S, V, delta, gamma, theta`` are aligned with the x-grid (including
    both truncated ends; the Greeks are NaN there).  ``theta`` is dV/dt per year.
    """

    option: Option
    x: np.ndarray
    S: np.ndarray
    V: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    theta: np.ndarray
    boundary_t: np.ndarray  # calendar times t of the stored time levels
    boundary_S: np.ndarray  # early-exercise boundary S_f(t) (NaN = no exercise)
    settings: dict = field(default_factory=dict)
    psor_sweeps: list[int] = field(default_factory=list)
    surface: np.ndarray | None = None  # V on all time levels (rows: t from T down to 0)
    surface_t: np.ndarray | None = None
    elapsed: float = 0.0

    def _spline(self, values):
        inner = slice(1, -1)
        return CubicSpline(self.x[inner], values[inner])

    def price(self, S0):
        """Option value at spot ``S0`` (cubic spline in x = ln(S/K))."""
        return self._spline(self.V)(np.log(np.asarray(S0, float) / self.option.K))[()]

    def greeks(self, S0) -> Greeks:
        """Price, Delta, Gamma, Theta at ``S0`` (Gamma linearly interpolated: it jumps at S_f)."""
        xq = np.log(np.asarray(S0, float) / self.option.K)
        inner = slice(1, -1)
        return Greeks(
            price=self.price(S0),
            delta=self._spline(self.delta)(xq)[()],
            gamma=np.interp(xq, self.x[inner], self.gamma[inner])[()],
            theta=np.interp(xq, self.x[inner], self.theta[inner])[()],
        )

    @property
    def exercise_boundary_t0(self) -> float:
        """Early-exercise point S_f at t = 0 (NaN if the option is never exercised)."""
        return float(self.boundary_S[-1])


# --------------------------------------------------------------------------- solver


def solve_fd(
    opt: Option,
    m: int = 400,
    n: int = 400,
    *,
    theta: float = 0.5,
    lcp: LCPSolver = "psor",
    rannacher: int = 2,
    half_width: float | None = None,
    omega: float | None = None,
    tol: float = 1e-12,
    store_surface: bool = False,
) -> FDResult:
    """Solve for V(S, 0) on the whole grid.

    Parameters
    ----------
    m : number of x-intervals (rounded up to even so that the strike is a node)
    n : number of tau-steps
    theta : 0.5 Crank–Nicolson, 1.0 fully implicit (0 explicit: only if lam <= 1/2)
    lcp : "psor" (projected SOR) or "brennan-schwartz" (direct); American only
    rannacher : number of initial CN steps replaced by two implicit half-steps each
    half_width : L for x in [-L, L]; default :func:`default_half_width`
    omega : PSOR relaxation factor; ``None`` picks the optimal SOR value for the
        unconstrained system (see :func:`optimal_omega`) at every step
    tol : PSOR relative stopping tolerance
    store_surface : keep V on every time level (for 3-D plots)
    """
    start = time.perf_counter()
    if m < 4 or n < 3:
        raise ValueError("grid too coarse: need m >= 4 and n >= 3")
    if not 0.0 <= theta <= 1.0:
        raise ValueError("theta must lie in [0, 1]")
    if lcp not in ("psor", "brennan-schwartz"):
        raise ValueError(f"unknown LCP solver {lcp!r}")
    m += m % 2
    rannacher = int(min(max(rannacher, 0), n - 2))  # keep >= 2 regular steps for Theta

    tr = HeatTransform(opt)
    L = default_half_width(opt) if half_width is None else float(half_width)
    x = np.linspace(-L, L, m + 1)
    dx = x[1] - x[0]
    x_in = x[1:-1]
    dtau = tr.tau_max / n

    # time-step schedule: (step size, theta)
    schedule = [(0.5 * dtau, 1.0)] * (2 * rannacher) + [(dtau, theta)] * (n - rannacher)

    y = tr.obstacle(x, 0.0)  # initial condition = payoff
    tau = 0.0
    history = [(tau, y.copy())]  # last three levels, for Theta
    levels_tau, levels_Sf = [], []
    surface = [tr.to_V(y, x, tau)] if store_surface else None
    surface_tau = [tau] if store_surface else None
    sweeps: list[int] = []
    exercise_at_left = not opt.is_call

    for dt, th in schedule:
        tau_new = tau + dt
        lam = dt / dx**2
        y_lo, y_hi = tr.boundary_values(x[0], x[-1], tau_new)

        w = y[1:-1]
        rhs = w + lam * (1.0 - th) * (y[2:] - 2.0 * w + y[:-2])
        rhs[0] += lam * th * y_lo
        rhs[-1] += lam * th * y_hi

        k = m - 1
        off = np.full(k, -lam * th)
        diag = np.full(k, 1.0 + 2.0 * lam * th)

        if opt.is_american:
            g = tr.obstacle(x_in, tau_new)
            if lcp == "psor":
                om = optimal_omega(lam * th, m - 1) if omega is None else omega
                w_new, it = psor(off, diag, off, rhs, g, x0=w, omega=om, tol=tol)
                sweeps.append(it)
            else:
                w_new = brennan_schwartz(off, diag, off, rhs, g, exercise_at_left)
            levels_tau.append(tau_new)
            levels_Sf.append(_exercise_point(w_new, g, opt.K * np.exp(x_in), exercise_at_left))
        else:
            w_new = thomas(off, diag, off, rhs)

        y = np.concatenate(([y_lo], w_new, [y_hi]))
        tau = tau_new
        history = (history + [(tau, y.copy())])[-3:]
        if store_surface:
            surface.append(tr.to_V(y, x, tau))
            surface_tau.append(tau)

    # ---- back to financial variables at t = 0
    S = opt.K * np.exp(x)
    V = tr.to_V(y, x, tau)

    delta, gamma = _delta_gamma(S, V)

    # Theta = dV/dt = -(sigma^2 / 2) dV/dtau, BDF2-type one-sided difference
    (t2, _), (_, _), (t0, _) = history
    V2, V1, V0 = (tr.to_V(yy, x, tt) for tt, yy in history)
    dV_dtau = (3.0 * V0 - 4.0 * V1 + V2) / (t0 - t2)
    theta_arr = np.full_like(V, np.nan)
    theta_arr[1:-1] = (-0.5 * opt.sigma**2 * dV_dtau)[1:-1]

    t_levels = opt.T - tr.remaining_time(np.array(levels_tau))
    res = FDResult(
        option=opt,
        x=x,
        S=S,
        V=V,
        delta=delta,
        gamma=gamma,
        theta=theta_arr,
        boundary_t=t_levels,
        boundary_S=np.array(levels_Sf) if levels_Sf else np.array([np.nan]),
        settings=dict(
            m=m, n=n, theta=theta, lcp=lcp, rannacher=rannacher, half_width=L, omega=omega, tol=tol
        ),  # fmt: skip
        psor_sweeps=sweeps,
        elapsed=time.perf_counter() - start,
    )
    if store_surface:
        res.surface = np.array(surface)
        res.surface_t = opt.T - tr.remaining_time(np.array(surface_tau))
    return res


def _delta_gamma(S, V):
    """Delta and Gamma by three-point formulas on the non-uniform S-grid.

    With h1 = S_i - S_{i-1}, h2 = S_{i+1} - S_i:

        V'  ~ (-h2^2 V_{i-1} + (h2^2 - h1^2) V_i + h1^2 V_{i+1}) / (h1 h2 (h1 + h2))
        V'' ~ 2 (h2 V_{i-1} - (h1 + h2) V_i + h1 V_{i+1}) / (h1 h2 (h1 + h2))

    Second order on the geometric grid S = K e^x, and exact for linear V, so in
    the stopping region Delta = -1 (put) / +1 (call) and Gamma = 0 to round-off.
    """
    h1 = S[1:-1] - S[:-2]
    h2 = S[2:] - S[1:-1]
    den = h1 * h2 * (h1 + h2)
    delta = np.full_like(V, np.nan)
    gamma = np.full_like(V, np.nan)
    delta[1:-1] = (-(h2**2) * V[:-2] + (h2**2 - h1**2) * V[1:-1] + h1**2 * V[2:]) / den
    gamma[1:-1] = 2.0 * (h2 * V[:-2] - (h1 + h2) * V[1:-1] + h1 * V[2:]) / den
    return delta, gamma


def _exercise_point(w, g, S, exercise_at_left: bool) -> float:
    """Grid point of the early-exercise boundary at one time level.

    A node is in the stopping region when the obstacle is active (w == g,
    both PSOR and Brennan–Schwartz set it exactly) and the payoff is positive.
    Put: the largest such S; call: the smallest.  This is the step-function
    approximation of S_f(t) described in Seydel, Alg. 4.14.
    """
    stopped = (g > 0.0) & (w <= g)
    if not stopped.any():
        return float("nan")
    idx = np.flatnonzero(stopped)
    return float(S[idx[-1]] if exercise_at_left else S[idx[0]])


def fd_price(S0, opt: Option, **kwargs):
    """Convenience: ``solve_fd(opt, **kwargs).price(S0)``."""
    return solve_fd(opt, **kwargs).price(S0)
