"""
Shared style, formatting, and IO for all figures.

Every figure in the repo (fig_gen scripts and the annotation analysis) imports
its style from here so all charts share the same fonts, grid, colors, and
number formatting — never set rcParams or hex colors inline. Reusable chart
builders (bars, histograms, heatmaps) live in plots.py, next to this module.

Usage (from any script in fig_gen/):
    from utils import COLORS, PALETTE, apply_style, ecdf, fmt_count, iter_jsonl, save_figure
    from plots import bar_counts, hist, heatmap

    apply_style()                      # once, before building the figure
    ax.bar(..., color=COLORS["blue_face"], edgecolor=COLORS["blue_edge"])
    save_figure(fig, out_path)
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Paired face (fill) / edge (outline) colors. Use one pair per data family
# within a figure: face for the area, edge for its outline and related lines.
COLORS = {
    "blue_face":   "#b7d4ea",
    "blue_edge":   "#0b3c6d",
    "pink_face":   "#f7c6d9",
    "pink_edge":   "#d81b60",
    "green_face":  "#bfe6dc",
    "green_edge":  "#00695c",
    "orange_face": "#fde4c8",
    "orange_edge": "#b45309",
    "purple_face": "#ddd6fe",
    "purple_edge": "#5b21b6",
    "gray_edge":   "#4d4d4d",
}

# Categorical palette (the edge colors, in order) for lines, word clouds, …
PALETTE = [COLORS[k] for k in
           ("blue_edge", "pink_edge", "green_edge", "orange_edge", "purple_edge")]

# Publication theme: white canvas, open frame (top/right spines removed),
# hairline grid, Times/STIX typography matching LaTeX body text, and
# editable Type 42 fonts at 300 dpi as required by most venues.
_RC = {
    # Canvas & export
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "pdf.fonttype": 42,            # embed TrueType, not Type 3 (IEEE/ACM rule)
    "ps.fonttype": 42,

    # Typography
    "font.family": "serif",
    "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlepad": 8,

    # Axes frame: only left + bottom spines, thin and near-black
    "axes.facecolor": "white",
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,

    # Grid: hairlines that sit behind the data without competing with it
    "axes.grid": True,
    "grid.color": "#d4d4d4",
    "grid.linewidth": 0.6,
    "grid.alpha": 1.0,

    # Ticks: short, outward, matching the spines
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,

    # Legend & details
    "legend.frameon": False,
    "legend.fontsize": 9,
    "legend.handlelength": 1.6,
    "lines.linewidth": 1.8,
    "hatch.linewidth": 0.6,        # fine hatch texture, readable in B/W print
}


def apply_style(overrides: dict | None = None) -> None:
    """Apply the shared paper style; call once before building a figure."""
    plt.rcParams.update({**_RC, **(overrides or {})})


def fmt_count(x, _=None) -> str:
    """Axis tick formatter: 500 → '500', 1500 → '1.5k', 2000 → '2k'."""
    if x >= 1000:
        return f"{x/1000:.0f}k" if x % 1000 == 0 else f"{x/1000:.1f}k"
    return str(int(x))


def ecdf(data) -> tuple[np.ndarray, np.ndarray]:
    """Empirical CDF: returns (sorted values, cumulative fractions)."""
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def iter_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_figure(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Saved → {out_path}")
    plt.close(fig)
