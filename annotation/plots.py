"""
Figure builders for the annotation analysis, in the shared fig_gen style.

Pure matplotlib on top of fig_gen (no seaborn, no inline rcParams/hex): colors
come from fig_gen.utils.COLORS/PALETTE and chart shapes from fig_gen.plots.
Label colors: YES = green, NO = pink, TIE/UNCERTAIN = orange, missing = gray.
Call fig_gen's apply_style() once (notebook setup cell) before plotting.

Usage (from the analysis notebook, cwd = annotation/):
    import plots

    plots.plot_volume(stats.annotator_volume(master, annotators))
    plots.plot_kappa_matrix(stats.pairwise_kappa(master, annotators))
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_ROOT), str(_ROOT / "fig_gen")):
    if _p not in sys.path:
        sys.path.append(_p)

from fig_gen.plots import (  # noqa: E402
    BAR_KW,
    bar_counts,
    grouped_bars,
    heatmap,
    pct_bars,
    set_panel_title,
)
from fig_gen.utils import COLORS  # noqa: E402

LABEL_COLORS = {
    "YES": (COLORS["green_face"], COLORS["green_edge"]),
    "NO": (COLORS["pink_face"], COLORS["pink_edge"]),
    "TIE": (COLORS["orange_face"], COLORS["orange_edge"]),
    "UNCERTAIN": (COLORS["orange_face"], COLORS["orange_edge"]),
    "not annotated": ("#e8e8e8", COLORS["gray_edge"]),
}


def _annotator_pair(k: int) -> tuple[str, str]:
    """(face, edge) pair for the k-th annotator, cycling the shared palette."""
    families = ["blue", "pink", "green", "orange", "purple"]
    fam = families[k % len(families)]
    return COLORS[f"{fam}_face"], COLORS[f"{fam}_edge"]


def plot_volume(vol: pd.Series):
    """Instances labeled per annotator (bar)."""
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bar_counts(ax, list(vol.index), list(vol.values),
               COLORS["blue_face"], COLORS["blue_edge"],
               "Volume annotated per annotator", ylabel="Instances")
    fig.tight_layout()
    return fig


def plot_majority(master: pd.DataFrame):
    """Majority label distribution on multi-annotated instances."""
    multi = master[master["is_multi"]]
    order = ["YES", "NO", "TIE"]
    vals = [int((multi["majority"] == m).sum()) for m in order]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(order, vals, color=[LABEL_COLORS[m][0] for m in order],
                  edgecolor=[LABEL_COLORS[m][1] for m in order], width=0.6, **BAR_KW)
    ymax = max(vals) or 1
    for b, v in zip(bars, vals, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02, f"{v:,}",
                ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_ylim(0, ymax * 1.25)
    ax.set_ylabel("Instances")
    set_panel_title(ax, "Majority label (multi-annotated instances)")
    fig.tight_layout()
    return fig


def plot_label_counts(counts: pd.DataFrame):
    """YES / NO / not-annotated counts per annotator (grouped bars)."""
    fig, ax = plt.subplots(figsize=(7.5, 4))
    grouped_bars(ax, list(counts.index),
                 {label: list(counts[label]) for label in counts.columns},
                 f"Labels per annotator (out of {int(counts.iloc[0].sum())} instances)",
                 ylabel="Instances", colors=LABEL_COLORS)
    fig.tight_layout()
    return fig


def plot_label_totals(counts: pd.DataFrame):
    """Overall YES / NO / not-annotated totals across all annotators."""
    totals = counts.sum()
    fig, ax = plt.subplots(figsize=(6, 3.6))
    bars = ax.bar(list(totals.index), list(totals.values),
                  color=[LABEL_COLORS[c][0] for c in totals.index],
                  edgecolor=[LABEL_COLORS[c][1] for c in totals.index],
                  width=0.6, **BAR_KW)
    ymax = max(totals.values) or 1
    for b, v in zip(bars, totals.values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02, f"{v:,}",
                ha="center", va="bottom", fontsize=7.5, color="#333")
    ax.set_ylim(0, ymax * 1.25)
    ax.set_ylabel("Labels")
    set_panel_title(ax, "Overall labels (all annotators)")
    fig.tight_layout()
    return fig


def plot_rate_and_bias(per_ann: pd.DataFrame, label: str = "NO"):
    """1×2 panel: per-annotator rate vs group mean, and signed bias bars."""
    key = label.lower()
    rate, bias = per_ann[f"{key}_rate"], per_ann[f"bias_{key}"]
    group = per_ann.attrs.get(f"group_{key}", rate.mean())
    face, edge = LABEL_COLORS[label]

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    pct_bars(axes[0], list(per_ann.index), list(rate), face, edge,
             f"{label} rate (group mean = {group:.0f}%)", ylabel=f"% {label}")
    axes[0].axhline(group, ls="--", c=COLORS["gray_edge"], lw=1.2)
    axes[0].set_ylim(0, max(105, rate.max() * 1.15))

    ax = axes[1]
    faces = [COLORS["green_face"] if b >= 0 else COLORS["pink_face"] for b in bias]
    edges = [COLORS["green_edge"] if b >= 0 else COLORS["pink_edge"] for b in bias]
    bars = ax.bar(list(per_ann.index), list(bias), color=faces, edgecolor=edges,
                  width=0.6, **BAR_KW)
    span = max(abs(bias.min()), abs(bias.max()), 1)
    for b, v in zip(bars, bias, strict=True):
        va = "bottom" if v >= 0 else "top"
        ax.text(b.get_x() + b.get_width() / 2, v + np.sign(v or 1) * span * 0.04,
                f"{v:+.1f}", ha="center", va=va, fontsize=7.5, color="#333")
    ax.axhline(0, c=COLORS["gray_edge"], lw=1.0)
    ax.set_ylim(-span * 1.3, span * 1.3)
    ax.set_ylabel("Gap (pts)")
    set_panel_title(ax, f"{label} bias vs group")
    fig.tight_layout()
    return fig


def plot_kappa_matrix(K: pd.DataFrame):
    """Pairwise Cohen's kappa heatmap (κ ∈ [-1, 1])."""
    fig, ax = plt.subplots(figsize=(4.8, 4.0))
    heatmap(ax, K.values, list(K.index), list(K.columns),
            "Cohen's κ (pairwise, on overlap)", fmt=".2f",
            cbar_label="κ", cmap="RdYlGn", vmin=-1, vmax=1)
    fig.tight_layout()
    return fig


def plot_confusion(cm: pd.DataFrame, a: str, b: str):
    """Label confusion matrix between two annotators."""
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    heatmap(ax, cm.values, list(cm.index), list(cm.columns),
            f"Confusion {a} × {b}  (n = {int(cm.values.sum())})",
            fmt=".0f", cmap="Blues")
    ax.set_xlabel(b)
    ax.set_ylabel(a)
    fig.tight_layout()
    return fig


def plot_category_rates(cat: pd.DataFrame):
    """1×2 panel: YES rate and multi-annotator agreement per category."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    pct_bars(axes[0], list(cat.index), list(cat["yes_rate"]),
             COLORS["green_face"], COLORS["green_edge"],
             "YES rate per category", ylabel="% YES")
    pct_bars(axes[1], list(cat.index), list(cat["agreement_multi_%"]),
             COLORS["blue_face"], COLORS["blue_edge"],
             "Agreement (multi) per category", ylabel="% unanimous")
    fig.tight_layout()
    return fig


def plot_agreement_by_category(master: pd.DataFrame):
    """Agreement vs disagreement counts per category (multi-annotated)."""
    multi = master[master["is_multi"]]
    cats = [c for c in ["PERS", "ORG", "LOC"] if c in set(multi["category"])]
    series = {
        "agreement": [int(((multi["category"] == c) & ~multi["disagree"]).sum())
                      for c in cats],
        "disagreement": [int(((multi["category"] == c) & multi["disagree"]).sum())
                         for c in cats],
    }
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    grouped_bars(ax, cats, series,
                 "Agreement / disagreement per category (multi-annotated)",
                 ylabel="Instances",
                 colors={"agreement": LABEL_COLORS["YES"],
                         "disagreement": LABEL_COLORS["NO"]})
    fig.tight_layout()
    return fig


def plot_labels_by_category(long: pd.DataFrame, annotators: list[str]):
    """Label counts per category: one overall panel + one panel per annotator."""
    cats = [c for c in ["PERS", "ORG", "LOC"] if c in set(long["category"])]
    labels = [label for label in ["YES", "NO", "UNCERTAIN"]
              if label in set(long["label"])]

    def _series(sub: pd.DataFrame) -> dict[str, list[int]]:
        return {label: [int(((sub["category"] == c) & (sub["label"] == label)).sum())
                        for c in cats] for label in labels}

    n_panels = 1 + len(annotators)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 3.6))
    axes = np.atleast_1d(axes)
    grouped_bars(axes[0], cats, _series(long), "All annotators",
                 ylabel="Labels", colors=LABEL_COLORS)
    for k, u in enumerate(annotators):
        grouped_bars(axes[k + 1], cats, _series(long[long["annotator"] == u]),
                     u, ylabel="Labels", colors=LABEL_COLORS)
    fig.suptitle("Labels per category and per annotator", y=1.02,
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig


def plot_time_matrix(matrix: pd.DataFrame):
    """Mean decision time heatmap (annotator × category, ALL margins)."""
    fig, ax = plt.subplots(figsize=(6.8, 3.4))
    heatmap(ax, matrix.values, list(matrix.index), list(matrix.columns),
            "Mean decision time (s) — annotator × category",
            fmt=".0f", cbar_label="mean time (s)", cmap="YlOrRd")
    fig.tight_layout()
    return fig


def plot_time_distribution(times: pd.DataFrame,
                           categories: tuple[str, ...] = ("PERS", "ORG", "LOC")):
    """1×2 panel: mean ± std bars and boxplots of decision time, per
    category × annotator (outliers capped upstream by stats.cap_outliers)."""
    cats = [c for c in categories if c in set(times["category"])]
    annotators = sorted(times["annotator"].unique())
    n_ann = len(annotators)
    x = np.arange(len(cats))
    width = 0.8 / max(n_ann, 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    # Mean ± std grouped bars
    ax = axes[0]
    for k, u in enumerate(annotators):
        face, edge = _annotator_pair(k)
        sub = times[times["annotator"] == u]
        means = [sub.loc[sub["category"] == c, "t"].mean() for c in cats]
        stds = [sub.loc[sub["category"] == c, "t"].std() for c in cats]
        offset = (k - (n_ann - 1) / 2) * width
        ax.bar(x + offset, means, yerr=stds, capsize=3, width=width * 0.92,
               color=face, edgecolor=edge, label=u,
               error_kw={"ecolor": COLORS["gray_edge"], "elinewidth": 1.2},
               **{**BAR_KW, "linewidth": 1.6})
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Time (s)")
    ax.legend()
    set_panel_title(ax, "Mean decision time ± std (capped)")

    # Boxplots
    ax = axes[1]
    for k, u in enumerate(annotators):
        face, edge = _annotator_pair(k)
        sub = times[times["annotator"] == u]
        data = [sub.loc[sub["category"] == c, "t"].values for c in cats]
        offset = (k - (n_ann - 1) / 2) * width
        bp = ax.boxplot(data, positions=x + offset, widths=width * 0.8,
                        patch_artist=True, showfliers=False,
                        medianprops={"color": edge, "linewidth": 1.6},
                        boxprops={"facecolor": face, "edgecolor": edge},
                        whiskerprops={"color": edge}, capprops={"color": edge})
        bp["boxes"][0].set_label(u)
        for c_idx, vals in enumerate(data):
            jitter = (np.random.default_rng(0).random(len(vals)) - 0.5) * width * 0.5
            ax.plot(np.full(len(vals), x[c_idx] + offset) + jitter, vals, "o",
                    color=COLORS["gray_edge"], markersize=3, alpha=0.7, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(cats)
    ax.set_ylabel("Time (s)")
    ax.legend()
    set_panel_title(ax, "Decision time distribution (capped)")

    fig.tight_layout()
    return fig
