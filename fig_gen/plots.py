"""
Shared axes-level chart builders for all figures.

Style (rcParams, colors, formatters, IO) lives in utils.py; this module holds
the reusable chart shapes built on top of it — hatch bars, histograms with
reference lines, grouped bars, and annotated heatmaps. Every figure script and
the annotation analysis modules draw through these helpers so all charts share
the exact same look; never restyle bars or titles inline.

Usage (from any script in fig_gen/):
    from utils import COLORS, apply_style, save_figure
    from plots import bar_counts, hist, heatmap, set_panel_title

    apply_style()                      # once, before building the figure
    bar_counts(ax, ["PERS", "ORG"], [120, 40],
               COLORS["blue_face"], COLORS["blue_edge"], "Entity types")
"""

from __future__ import annotations

from statistics import mean, median

import matplotlib.pyplot as plt
import numpy as np
from utils import COLORS, fmt_count

# Unified hatch-bar look shared by every bar chart (splat into ax.bar/ax.barh
# for bespoke panels that the helpers below don't cover).
BAR_KW = {"linewidth": 2.0, "hatch": "//", "zorder": 3}


def set_panel_title(ax, text: str, fontsize: float = 10.5) -> None:
    """Panel title with the shared weight/padding."""
    ax.set_title(text, fontsize=fontsize, fontweight="bold", pad=6)


def bar_counts(ax, labels, vals, face, edge, title, ylabel="Instances",
               total=None, pct=True, width=0.6):
    """Vertical bar chart with value (and optional %) labels above each bar."""
    bars = ax.bar(labels, vals, color=face, edgecolor=edge, width=width, **BAR_KW)
    ymax = max(vals) if len(vals) else 1
    ax.set_ylim(0, ymax * 1.25)
    for b, v in zip(bars, vals, strict=True):
        label = f"{v:,}"
        if pct and total:
            label += f"\n({100*v/total:.0f}%)"
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02, label,
                ha="center", va="bottom", fontsize=7.5, color="#333")
    set_panel_title(ax, title)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_count))
    return bars


def barh_counts(ax, labels, vals, face, edge, title, xlabel="Instances"):
    """Horizontal bar chart (labels listed top-to-bottom in given order)."""
    labels = list(reversed(labels))
    vals = list(reversed(vals))
    bars = ax.barh(labels, vals, color=face, edgecolor=edge, height=0.65,
                   **{**BAR_KW, "linewidth": 1.8})
    xmax = max(vals) * 1.2 if vals else 1
    ax.set_xlim(0, xmax)
    for b, v in zip(bars, vals, strict=True):
        ax.text(v + xmax * 0.012, b.get_y() + b.get_height() / 2,
                f"{v:,}", va="center", fontsize=7.5, color="#333")
    ax.set_xlabel(xlabel)
    set_panel_title(ax, title)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_count))
    ax.tick_params(axis="y", labelsize=8)
    return bars


def pct_bars(ax, labels, pcts, faces, edge, title, ylabel="%"):
    """Percentage bar chart (0–100 scale) with a % label above each bar."""
    bars = ax.bar(labels, pcts, color=faces, edgecolor=edge, width=0.6, **BAR_KW)
    for b, v in zip(bars, pcts, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(ylabel)
    set_panel_title(ax, title)
    return bars


def hist(ax, data, bins, face, edge, title, xlabel, ylabel="Instances",
         clip_pct=None, vlines=True):
    """Histogram with mean/median reference lines and a legend."""
    data = np.asarray(data, dtype=float)
    if clip_pct is not None:
        data = data[data <= np.percentile(data, clip_pct)]
    ax.hist(data, bins=bins, color=face, edgecolor=edge,
            **{**BAR_KW, "linewidth": 1.4})
    if vlines:
        mu, med = mean(data), median(data)
        ax.axvline(mu, color=COLORS["gray_edge"], linewidth=1.6,
                   linestyle="-", label=f"Mean = {mu:.1f}")
        ax.axvline(med, color=COLORS["gray_edge"], linewidth=1.6,
                   linestyle="--", label=f"Median = {med:.1f}")
        ax.legend(loc="upper right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_count))
    set_panel_title(ax, title)


def grouped_bars(ax, group_labels, series, title, ylabel="Instances",
                 colors=None):
    """Grouped vertical bars: one bar per (group, series) pair.

    series maps a series name to its per-group values; colors maps a series
    name to a (face, edge) pair, defaulting to the shared color families.
    """
    default_pairs = [("blue_face", "blue_edge"), ("pink_face", "pink_edge"),
                     ("green_face", "green_edge"), ("orange_face", "orange_edge"),
                     ("purple_face", "purple_edge")]
    n_series = len(series)
    x = np.arange(len(group_labels))
    width = 0.8 / max(n_series, 1)
    ymax = max((max(v) for v in series.values() if len(v)), default=1)
    for k, (name, vals) in enumerate(series.items()):
        if colors and name in colors:
            face, edge = colors[name]
        else:
            fk, ek = default_pairs[k % len(default_pairs)]
            face, edge = COLORS[fk], COLORS[ek]
        offset = (k - (n_series - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width * 0.92, color=face,
                      edgecolor=edge, label=name, **{**BAR_KW, "linewidth": 1.6})
        for b, v in zip(bars, vals, strict=True):
            ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02, f"{v:,}",
                    ha="center", va="bottom", fontsize=7, color="#333")
    ax.set_xticks(x)
    ax.set_xticklabels(group_labels)
    ax.set_ylim(0, ymax * 1.25)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_count))
    ax.legend()
    set_panel_title(ax, title)


def heatmap(ax, values, row_labels, col_labels, title, fmt=".2f",
            cbar_label=None, cmap="RdYlGn", vmin=None, vmax=None):
    """Annotated matrix heatmap (pure matplotlib imshow, no grid lines)."""
    values = np.asarray(values, dtype=float)
    ax.grid(False)
    im = ax.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    lo = np.nanmin(values) if vmin is None else vmin
    hi = np.nanmax(values) if vmax is None else vmax
    mid, span = (lo + hi) / 2, (hi - lo) or 1.0
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            v = values[r, c]
            if np.isnan(v):
                continue
            # Dark ink on mid-range cells, white on the color extremes
            color = "white" if abs(v - mid) > 0.35 * span else "#1a1a1a"
            ax.text(c, r, format(v, fmt), ha="center", va="center",
                    fontsize=8.5, color=color)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if cbar_label:
        cbar.set_label(cbar_label)
    cbar.outline.set_visible(False)
    set_panel_title(ax, title)
    return im
