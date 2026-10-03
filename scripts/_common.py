"""Shared plotting style and paths for the scripts in this folder."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "figures"
RES = ROOT / "results"
FIG.mkdir(exist_ok=True)
RES.mkdir(exist_ok=True)

# categorical slots in fixed order (validated reference palette, light mode)
BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#4a3aa7"
INK, INK_2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK_2,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.grid": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "legend.fontsize": 8.5,
        "legend.frameon": False,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "font.family": "DejaVu Sans",
    }
)


def save(fig, name: str) -> Path:
    path = FIG / name
    fig.savefig(path)
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def write_md(name: str, text: str) -> Path:
    path = RES / name
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    print(f"wrote {path.relative_to(ROOT)}")
    return path


def md_table(header: list[str], rows: list[list]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---:" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)
