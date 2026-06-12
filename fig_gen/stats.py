#!/usr/bin/env python3
"""
Dataset statistics for the final MEL dataset.

Usage:
    python scripts/stats.py output/final/instances.jsonl
    python scripts/stats.py output/final/instances.jsonl --kb output/final/kb.jsonl
    python scripts/stats.py output/final/instances.jsonl --out figures/stats.pdf
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np


COLORS = {
    "blue_face":   "#b7d4ea",
    "blue_edge":   "#0b3c6d",
    "pink_face":   "#f7c6d9",
    "pink_edge":   "#d81b60",
    "green_face":  "#bfe6dc",
    "green_edge":  "#00695c",
    "orange_face": "#fde4c8",
    "orange_edge": "#b45309",
    "gray_edge":   "#4d4d4d",
}


def _style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "figure.facecolor": "#ffffff",
        "savefig.facecolor": "#ffffff",
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["Times New Roman", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.facecolor": "#ececec",
        "axes.edgecolor": "#666666",
        "axes.linewidth": 1.6,
        "axes.labelsize": 13,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#c9c9c9",
        "grid.linewidth": 1.0,
        "grid.alpha": 1.0,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "hatch.linewidth": 1.5,
    })


def _fmt(x, _=None):
    if x >= 1000:
        return f"{x/1000:.0f}k" if x % 1000 == 0 else f"{x/1000:.1f}k"
    return str(int(x))


def _label_bars(ax, bars, total, fontsize=7):
    ymax = ax.get_ylim()[1]
    for b in bars:
        h = b.get_height()
        if h == 0:
            continue
        ax.text(b.get_x() + b.get_width() / 2, h + ymax * 0.012,
                f"{int(h):,}\n({100*h/total:.0f}%)",
                ha="center", va="bottom", fontsize=fontsize, color="#333")


def _bar(ax, labels, vals, face, edge, title, ylabel=None, total=None):
    bars = ax.bar(labels, vals, color=face, edgecolor=edge,
                  linewidth=2.0, hatch="//", zorder=3, width=0.55)
    if total is not None:
        _label_bars(ax, bars, total)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt))
    return bars


def _ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--kb", type=Path, default=None)
    ap.add_argument("--out", "-o", type=Path, default=None)
    ap.add_argument("--top", type=int, default=15, metavar="N",
                    help="number of top surface forms to show (default: 15)")
    args = ap.parse_args()

    kb_path = args.kb or (args.input.parent / "kb.jsonl")
    if not kb_path.exists():
        raise SystemExit(f"KB not found: {kb_path}")
    out = args.out or args.input.with_name(args.input.stem + "_stats.pdf")

    kb = {e["qid"]: e for e in _iter_jsonl(kb_path)}
    instances = list(_iter_jsonl(args.input))
    n_kb, N = len(kb), len(instances)
    if N == 0:
        raise SystemExit("No instances.")

    # ── KB ───────────────────────────────────────────────────
    kb_types       = Counter(e.get("type", "OTHER") for e in kb.values())
    kb_pct_intro   = 100 * sum(1 for e in kb.values() if e.get("intro"))        / n_kb
    kb_pct_desc    = 100 * sum(1 for e in kb.values() if e.get("desc"))         / n_kb
    kb_pct_infobox = 100 * sum(1 for e in kb.values() if e.get("infobox_img"))  / n_kb
    kb_pct_wiki    = 100 * sum(1 for e in kb.values() if e.get("url_wikipedia")) / n_kb

    # ── Instances ─────────────────────────────────────────────
    n_text = [len(i["text_candidates"])   for i in instances]
    n_vis  = [len(i["visual_candidates"]) for i in instances]
    n_ub   = [i["image"].get("n_used_by", 0) for i in instances]
    img_w  = [i["image"].get("width", 0)  for i in instances]
    img_h  = [i["image"].get("height", 0) for i in instances]

    answer_types   = Counter()
    mention_counts = Counter(i["mention"] for i in instances)
    for i in instances:
        a = kb.get(i.get("answer"))
        if a:
            answer_types[a.get("type", "OTHER")] += 1

    inst_per_mention  = list(mention_counts.values())
    n_unique_mentions = len(mention_counts)
    n_unique_answers  = len({i.get("answer") for i in instances} - {None})

    # ── Console ───────────────────────────────────────────────
    W = 46
    print(f"\n{'─'*W}")
    print("  Knowledge Base")
    print(f"{'─'*W}")
    print(f"  {'Unique entities':<30} {n_kb:>9,}")
    for t in ["PERS", "ORG", "LOC"]:
        c = kb_types.get(t, 0)
        print(f"    {'— '+t:<28} {c:>9,}  ({100*c/n_kb:.0f}%)")
    print(f"  {'With intro text':<30} {kb_pct_intro:>8.0f}%")
    print(f"  {'With Wikidata desc':<30} {kb_pct_desc:>8.0f}%")
    print(f"  {'With infobox image':<30} {kb_pct_infobox:>8.0f}%")
    print(f"  {'With Wikipedia URL':<30} {kb_pct_wiki:>8.0f}%")
    print(f"{'─'*W}")
    print("  Instances")
    print(f"{'─'*W}")
    print(f"  {'Total instances':<30} {N:>9,}")
    print(f"  {'Unique surface forms':<30} {n_unique_mentions:>9,}")
    print(f"  {'Unique answer entities':<30} {n_unique_answers:>9,}")
    print(f"  {'Instances/mention  μ / max':<30} {mean(inst_per_mention):>5.1f}  /  {max(inst_per_mention)}")
    print(f"  {'Text candidates  μ / med':<30} {mean(n_text):>5.1f}  /  {median(n_text):.0f}")
    print(f"  {'Visual candidates  μ / med':<30} {mean(n_vis):>5.1f}  /  {median(n_vis):.0f}")
    print(f"  {'Image n_used_by  μ / med':<30} {mean(n_ub):>5.1f}  /  {median(n_ub):.0f}")
    print(f"  {'Image width   μ / med px':<30} {mean(img_w):>5.0f}  /  {median(img_w):.0f}")
    print(f"  {'Image height  μ / med px':<30} {mean(img_h):>5.0f}  /  {median(img_h):.0f}")
    print(f"  Answer type:")
    for t in ["PERS", "ORG", "LOC"]:
        c = answer_types.get(t, 0)
        print(f"    {'— '+t:<28} {c:>9,}  ({100*c/N:.0f}%)")
    print(f"{'─'*W}")
    print(f"\n  Top {args.top} surface forms by instance count:")
    for mention, cnt in mention_counts.most_common(args.top):
        print(f"    {mention:<32} {cnt:>5,}")
    print()

    # ── Figure ────────────────────────────────────────────────
    # Layout: 3 rows × 2 cols; bottom-left spans full row for top mentions
    _style()

    top_n   = min(args.top, len(mention_counts))
    row_h   = [1, 1, max(1.0, top_n * 0.12)]  # taller row for top mentions

    fig = plt.figure(figsize=(10, sum(row_h) * 3.2))
    gs  = gridspec.GridSpec(3, 2, figure=fig,
                            height_ratios=row_h,
                            hspace=0.55, wspace=0.38)

    fig.text(0.5, 1.002,
             f"{args.input.name}  ·  {n_kb:,} KB entities  ·  {N:,} instances",
             fontsize=10, ha="center", va="bottom", color="#555")

    types = ["PERS", "ORG", "LOC"]

    # ── (A) KB entity types ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    vals = [kb_types.get(t, 0) for t in types]
    _bar(ax, types, vals, COLORS["blue_face"], COLORS["blue_edge"],
         "KB — Entity types", ylabel="Entities", total=n_kb)
    ax.set_ylim(0, max(vals) * 1.4)

    # ── (B) KB completeness ───────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    fields = ["Intro", "Desc", "Infobox", "Wiki URL"]
    fvals  = [kb_pct_intro, kb_pct_desc, kb_pct_infobox, kb_pct_wiki]
    faces  = [COLORS["blue_face"], COLORS["pink_face"], COLORS["green_face"], "#ddd6fe"]
    bars = ax.bar(fields, fvals, color=faces, edgecolor=COLORS["gray_edge"],
                  linewidth=2.0, hatch="//", zorder=3, width=0.55)
    for b, v in zip(bars, fvals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=9, color="#333")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_title("KB — Entity completeness", fontsize=11, fontweight="bold", pad=6)
    ax.set_ylabel("% of KB entities")

    # ── (C) CDF — text vs. visual candidate set size ──────────
    ax = fig.add_subplot(gs[1, 0])
    xt, yt = _ecdf(n_text)
    xv, yv = _ecdf(n_vis)
    # clip x to 99th percentile for readability
    x_max = int(np.percentile(np.concatenate([xt, xv]), 99)) + 1

    ax.step(xt, yt, where="post",
            color=COLORS["blue_edge"], linewidth=2.2,
            label=f"Text  (μ={mean(n_text):.1f}, med={median(n_text):.0f})")
    ax.fill_between(xt, 0, yt, step="post",
                    color=COLORS["blue_face"], alpha=0.45)

    ax.step(xv, yv, where="post",
            color=COLORS["pink_edge"], linewidth=2.2, linestyle="--",
            label=f"Visual  (μ={mean(n_vis):.1f}, med={median(n_vis):.0f})")
    ax.fill_between(xv, 0, yv, step="post",
                    color=COLORS["pink_face"], alpha=0.35)

    # median reference lines
    for val, col in [(median(n_text), COLORS["blue_edge"]),
                     (median(n_vis),  COLORS["pink_edge"])]:
        ax.axvline(val, color=col, linewidth=1.0, linestyle=":", alpha=0.7)

    ax.set_xlim(1, x_max)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("# candidates")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("Candidate set size — CDF", fontsize=11, fontweight="bold", pad=6)
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

    # ── (D) Image n_used_by distribution ─────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ub_counter = Counter(n_ub)
    xs = sorted(ub_counter.keys())
    ys = [ub_counter[x] for x in xs]
    bars = ax.bar(xs, ys, color=COLORS["green_face"], edgecolor=COLORS["green_edge"],
                  linewidth=2.0, hatch="//", zorder=3, width=0.7)
    ax.plot(xs, ys, "o-", color=COLORS["green_edge"], linewidth=1.6,
            markersize=4, zorder=4)
    ax.set_xlabel("n_used_by")
    ax.set_ylabel("Instances")
    ax.set_title(f"Image article reuse  (μ={mean(n_ub):.1f})",
                 fontsize=11, fontweight="bold", pad=6)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_fmt))

    # ── (E) Top surface forms ─────────────────────────────────
    ax = fig.add_subplot(gs[2, 0])
    top  = mention_counts.most_common(top_n)
    mlabels = [m for m, _ in reversed(top)]
    mvals   = [c for _, c in reversed(top)]
    bars = ax.barh(mlabels, mvals,
                   color=COLORS["orange_face"], edgecolor=COLORS["orange_edge"],
                   linewidth=1.8, hatch="//", zorder=3, height=0.65)
    xmax = max(mvals) * 1.18
    for b, v in zip(bars, mvals):
        ax.text(v + xmax * 0.01, b.get_y() + b.get_height() / 2,
                str(v), va="center", fontsize=8, color="#333")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("# instances")
    ax.set_title(f"Top {top_n} surface forms by instance count",
                 fontsize=11, fontweight="bold", pad=6)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(_fmt))
    ax.tick_params(axis="y", labelsize=9)

    # ── (F) Answer entity types ───────────────────────────────
    ax = fig.add_subplot(gs[2, 1])
    avals = [answer_types.get(t, 0) for t in types]
    _bar(ax, types, avals, COLORS["orange_face"], COLORS["orange_edge"],
         "Answer entity types", ylabel="Instances", total=N)
    ax.set_ylim(0, max(avals, default=1) * 1.4)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print(f"Saved → {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
