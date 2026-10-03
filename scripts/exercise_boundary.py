"""Early-exercise boundary S_f(t).

* Left: American put of Seydel Example 1.6 on three grids, the asymptotic
  behaviour near expiry  S_f(t) ~ K (1 - sigma sqrt((T - t) |ln(T - t)|))
  and the perpetual level K q / (1 + q), q = 2 r / sigma^2.
* Right: the analogue of Seydel Fig. 4.7: put with r = 6%, sigma = 30%, K = 10,
  T = 1 and dividend yields 12%, 8%, 4%.  As t -> T the boundary tends to
  min(K, r K / div)  (Seydel 4.23P).

Run:  python scripts/exercise_boundary.py
"""

from __future__ import annotations

import numpy as np
from _common import AQUA, BLUE, INK_2, MUTED, ORANGE, plt, save

from americanfd import Option, perpetual_american_put, solve_fd

am = Option(K=50.0, T=5 / 12, r=0.1, sigma=0.4, kind="put")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

for m, color in ((200, AQUA), (800, ORANGE), (3200, BLUE)):
    res = solve_fd(am, m, m, lcp="brennan-schwartz")
    ax1.plot(res.boundary_t, res.boundary_S, color=color, lw=1.4 if m < 3200 else 2,
             label=f"FD grid {m}×{m}")  # fmt: skip

t = np.linspace(am.T - 0.02, am.T - 1e-5, 200)
rem = am.T - t
asym = am.K * (1 - am.sigma * np.sqrt(rem * np.abs(np.log(rem))))
ax1.plot(t, asym, color=INK_2, ls="--", lw=1.2, label="asymptote near expiry")
_, s_perp = perpetual_american_put(1.0, am.K, am.r, am.sigma)
ax1.axhline(s_perp, color=MUTED, ls=":", lw=1.2)
ax1.annotate(f"perpetual put: S_f = {s_perp:.2f}", (0.01, s_perp), xytext=(0, 4),
             textcoords="offset points", color=INK_2, fontsize=8.5)  # fmt: skip
ax1.set(xlabel="calendar time t", ylabel="S_f(t)", ylim=(s_perp - 2, am.K + 1),
        title="Put, K = 50 (Seydel Ex. 1.6)")  # fmt: skip
ax1.annotate("stop (exercise)", (0.24, 33.0), color=INK_2, fontsize=8.5)
ax1.annotate("hold", (0.24, 44.0), color=INK_2, fontsize=8.5)
ax1.legend(loc="upper left")

for div, color in ((0.12, BLUE), (0.08, ORANGE), (0.04, AQUA)):
    opt = Option(K=10.0, T=1.0, r=0.06, sigma=0.3, div=div, kind="put")
    res = solve_fd(opt, 1600, 1600, lcp="brennan-schwartz")
    ax2.plot(res.boundary_t, res.boundary_S, color=color)
    ax2.annotate(f"δ = {div:.0%}", (0.02, res.boundary_S[-1]), xytext=(0, 6),
                 textcoords="offset points", color=INK_2, fontsize=8.5)  # fmt: skip
    lim = min(opt.K, opt.r * opt.K / div)
    ax2.plot([opt.T], [lim], marker="o", color=color, mfc="white", mew=1.5)
ax2.set(xlabel="calendar time t", ylabel="S_f(t)",
        title="Put with dividends (r = 6%, σ = 30%, K = 10)")  # fmt: skip
ax2.annotate("○ = limit min(K, rK/δ) at t = T", (0.02, 9.6), color=INK_2, fontsize=8.5)
save(fig, "exercise_boundary.png")
