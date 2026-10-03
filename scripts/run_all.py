"""Regenerate every table in results/ and every figure in figures/.

Run:  python scripts/run_all.py      (a few minutes without numba)
"""

import runpy
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
for name in ("convergence", "compare_methods", "greeks", "exercise_boundary", "psor_omega"):
    start = time.perf_counter()
    print(f"== {name}")
    runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
    print(f"   {time.perf_counter() - start:.1f}s")
