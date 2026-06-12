#!/usr/bin/env python3
"""
Extended dataset statistics for the final MEL dataset.

A larger panel of simple, clearly-labelled plots (histograms and bar
charts only — no fancy/compound visualisations) covering the KB,
candidate pools, images, mentions and answer entities.

Usage:
    python fig_gen/stats_extended.py output/split_10_text/instances.jsonl
    python fig_gen/stats_extended.py output/split_10_text/instances.jsonl --kb output/split_10_text/kb.jsonl
    python fig_gen/stats_extended.py output/split_10_text/instances.jsonl --out output/figures/stats_extended.pdf
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from utils import COLORS, apply_style, fmt_count, iter_jsonl, save_figure


def _title(ax, text):
    ax.set_title(text, fontsize=10.5, fontweight="bold", pad=6)


def _bar_counts(ax, labels, vals, face, edge, title, ylabel="Instances",
                total=None, pct=True):
    """Vertical bar chart with value (and optional %) labels above each bar."""
    bars = ax.bar(labels, vals, color=face, edgecolor=edge,
                   linewidth=2.0, hatch="//", zorder=3, width=0.6)
    ymax = max(vals) if vals else 1
    ax.set_ylim(0, ymax * 1.25)
    for b, v in zip(bars, vals):
        label = f"{v:,}"
        if pct and total:
            label += f"\n({100*v/total:.0f}%)"
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.02, label,
                ha="center", va="bottom", fontsize=7.5, color="#333")
    _title(ax, title)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_count))
    return bars


def _barh_counts(ax, labels, vals, face, edge, title, xlabel="Instances"):
    """Horizontal bar chart (labels listed top-to-bottom in given order)."""
    labels = list(reversed(labels))
    vals = list(reversed(vals))
    bars = ax.barh(labels, vals, color=face, edgecolor=edge,
                    linewidth=1.8, hatch="//", zorder=3, height=0.65)
    xmax = max(vals) * 1.2 if vals else 1
    ax.set_xlim(0, xmax)
    for b, v in zip(bars, vals):
        ax.text(v + xmax * 0.012, b.get_y() + b.get_height() / 2,
                f"{v:,}", va="center", fontsize=7.5, color="#333")
    ax.set_xlabel(xlabel)
    _title(ax, title)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_count))
    ax.tick_params(axis="y", labelsize=8)
    return bars


def _hist(ax, data, bins, face, edge, title, xlabel, ylabel="Instances",
          clip_pct=None, vlines=True):
    """Histogram with mean/median reference lines and a legend."""
    data = np.asarray(data, dtype=float)
    if clip_pct is not None:
        data = data[data <= np.percentile(data, clip_pct)]
    ax.hist(data, bins=bins, color=face, edgecolor=edge,
            linewidth=1.4, hatch="//", zorder=3)
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
    _title(ax, title)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("--kb", type=Path, default=None)
    ap.add_argument("--out", "-o", type=Path, default=None)
    args = ap.parse_args()

    kb_path = args.kb or (args.input.parent / "kb.jsonl")
    if not kb_path.exists():
        raise SystemExit(f"KB not found: {kb_path}")
    out = args.out or args.input.with_name(args.input.stem + "_stats_extended.pdf")

    kb = {e["qid"]: e for e in iter_jsonl(kb_path)}
    instances = list(iter_jsonl(args.input))
    n_kb, N = len(kb), len(instances)
    if N == 0:
        raise SystemExit("No instances.")

    types = ["PERS", "ORG", "LOC"]

    # ── KB ────────────────────────────────────────────────────
    kb_types = Counter(e.get("type", "OTHER") for e in kb.values())
    kb_completeness = {
        "Intro":    100 * sum(1 for e in kb.values() if e.get("intro")) / n_kb,
        "Desc":     100 * sum(1 for e in kb.values() if e.get("desc")) / n_kb,
        "Infobox":  100 * sum(1 for e in kb.values() if e.get("infobox_img")) / n_kb,
        "Wiki URL": 100 * sum(1 for e in kb.values() if e.get("url_wikipedia")) / n_kb,
    }

    # ── Per-instance basics ───────────────────────────────────
    n_text = [len(i["text_candidates"]) for i in instances]
    n_vis  = [len(i["visual_candidates"]) for i in instances]
    n_ub   = [i["image"].get("n_used_by", 0) for i in instances]
    img_w  = [i["image"].get("width", 0) for i in instances]
    img_h  = [i["image"].get("height", 0) for i in instances]
    aspect = [w / h for w, h in zip(img_w, img_h) if w and h]
    mpix   = [w * h / 1e6 for w, h in zip(img_w, img_h) if w and h]
    licenses = Counter(i["image"].get("license", "Unknown") for i in instances)

    mention_counts = Counter(i["mention"] for i in instances)
    mention_lens   = [len(i["mention"]) for i in instances]
    mention_words  = [len(i["mention"].split()) for i in instances]

    answer_types = Counter()
    desc_lens, has_infobox = [], 0
    for i in instances:
        a = kb.get(i.get("answer"))
        if a:
            answer_types[a.get("type", "OTHER")] += 1
            text = a.get("intro") or a.get("desc") or ""
            desc_lens.append(len(text))
            if a.get("infobox_img"):
                has_infobox += 1
    pct_answer_infobox = 100 * has_infobox / N

    # ── Candidate pool composition ────────────────────────────
    pool_n_types = Counter()
    answer_is_majority = 0
    for i in instances:
        pool_types = Counter(kb[q]["type"] for q in i["text_candidates"] if q in kb)
        pool_n_types[len(pool_types)] += 1
        ans = kb.get(i.get("answer"))
        if ans and pool_types:
            majority = pool_types.most_common(1)[0][0]
            if ans.get("type") == majority:
                answer_is_majority += 1
    pct_majority = 100 * answer_is_majority / N
    pct_minority = 100 - pct_majority

    # instances per mention, bucketed
    inst_per_mention = list(mention_counts.values())
    bucket_labels = ["1", "2", "3", "4-5", "6+"]
    bucket_vals = [
        sum(1 for c in inst_per_mention if c == 1),
        sum(1 for c in inst_per_mention if c == 2),
        sum(1 for c in inst_per_mention if c == 3),
        sum(1 for c in inst_per_mention if 4 <= c <= 5),
        sum(1 for c in inst_per_mention if c >= 6),
    ]

    # mention word-count buckets
    word_labels = ["1", "2", "3", "4+"]
    word_vals = [
        sum(1 for w in mention_words if w == 1),
        sum(1 for w in mention_words if w == 2),
        sum(1 for w in mention_words if w == 3),
        sum(1 for w in mention_words if w >= 4),
    ]

    # visual candidate size buckets
    vis_dist = Counter(n_vis)
    vis_labels = sorted(vis_dist)
    vis_max_label = max(vis_labels)
    vis_labels_disp = [str(v) if v < 6 else "6+" for v in range(2, 7)]
    vis_vals_disp = [vis_dist.get(v, 0) for v in range(2, 6)]
    vis_vals_disp.append(sum(c for v, c in vis_dist.items() if v >= 6))

    # top licenses
    top_licenses = licenses.most_common(8)
    lic_labels = [l for l, _ in top_licenses]
    lic_vals = [c for _, c in top_licenses]

    # top surface forms
    top_mentions = mention_counts.most_common(10)
    tm_labels = [m for m, _ in top_mentions]
    tm_vals = [c for _, c in top_mentions]

    # ── Console summary ───────────────────────────────────────
    W = 50
    print(f"\n{'─'*W}")
    print(f"  {args.input.name}  —  {n_kb:,} KB entities, {N:,} instances")
    print(f"{'─'*W}")
    print(f"  Candidate pool type homogeneity:")
    for k in sorted(pool_n_types):
        print(f"    {k} type(s): {pool_n_types[k]:>7,}  ({100*pool_n_types[k]/N:.1f}%)")
    print(f"  Answer = pool majority type: {pct_majority:.1f}%")
    print(f"  Answer entity has infobox image: {pct_answer_infobox:.1f}%")
    print(f"  Image licenses (top 8):")
    for l, c in top_licenses:
        print(f"    {l:<28} {c:>7,}  ({100*c/N:.1f}%)")
    print(f"{'─'*W}\n")

    # ── Figure: 4x4 grid ───────────────────────────────────────
    apply_style()
    fig = plt.figure(figsize=(20, 18.5))
    gs = gridspec.GridSpec(4, 4, figure=fig, hspace=0.5, wspace=0.42, top=0.96)

    fig.text(0.5, 0.99,
             f"{args.input.name}  ·  {n_kb:,} KB entities  ·  {N:,} instances",
             fontsize=12, ha="center", va="bottom", color="#444", fontweight="bold")

    # Row 1 -----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _bar_counts(ax, types, [kb_types.get(t, 0) for t in types],
                COLORS["blue_face"], COLORS["blue_edge"],
                "KB — Entity types", ylabel="KB entities", total=n_kb)

    ax = fig.add_subplot(gs[0, 1])
    fields = list(kb_completeness)
    fvals = list(kb_completeness.values())
    bars = ax.bar(fields, fvals, color=COLORS["purple_face"], edgecolor=COLORS["purple_edge"],
                   linewidth=2.0, hatch="//", zorder=3, width=0.6)
    for b, v in zip(bars, fvals):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=8, color="#333")
    ax.set_ylim(0, 115)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("% of KB entities")
    _title(ax, "KB — Field completeness")

    ax = fig.add_subplot(gs[0, 2])
    _bar_counts(ax, types, [answer_types.get(t, 0) for t in types],
                COLORS["orange_face"], COLORS["orange_edge"],
                "Answer entity — type", ylabel="Instances", total=N)

    ax = fig.add_subplot(gs[0, 3])
    _barh_counts(ax, tm_labels, tm_vals, COLORS["pink_face"], COLORS["pink_edge"],
                 "Top 10 surface forms", xlabel="Instances")

    # Row 2 -------------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    _hist(ax, n_text, bins=range(2, 32), face=COLORS["blue_face"], edge=COLORS["blue_edge"],
          title="Text candidates per instance", xlabel="# text candidates")

    ax = fig.add_subplot(gs[1, 1])
    _bar_counts(ax, vis_labels_disp, vis_vals_disp, COLORS["pink_face"], COLORS["pink_edge"],
                "Visual candidates per instance", ylabel="Instances", total=N)
    ax.set_xlabel("# visual candidates")

    ax = fig.add_subplot(gs[1, 2])
    _bar_counts(ax, bucket_labels, bucket_vals, COLORS["green_face"], COLORS["green_edge"],
                "Instances per surface form", ylabel="Surface forms", total=n_unique_mentions if (n_unique_mentions := len(mention_counts)) else N)
    ax.set_xlabel("# instances sharing the mention")

    ax = fig.add_subplot(gs[1, 3])
    ub_dist = Counter(n_ub)
    ub_xs = sorted(ub_dist)
    ub_ys = [ub_dist[x] for x in ub_xs]
    bars = ax.bar(ub_xs, ub_ys, color=COLORS["green_face"], edgecolor=COLORS["green_edge"],
                   linewidth=2.0, hatch="//", zorder=3, width=0.7)
    ax.plot(ub_xs, ub_ys, "o-", color=COLORS["green_edge"], linewidth=1.6, markersize=4, zorder=4)
    ax.set_xlabel("Wikipedia article reuse (n_used_by)")
    ax.set_ylabel("Instances")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_count))
    _title(ax, f"Image reuse across articles  (mean = {mean(n_ub):.1f})")

    # Row 3 -------------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    _barh_counts(ax, lic_labels, lic_vals, COLORS["purple_face"], COLORS["purple_edge"],
                 "Image license (top 8)", xlabel="Instances")

    ax = fig.add_subplot(gs[2, 1])
    _hist(ax, aspect, bins=np.arange(0.2, 2.61, 0.1), face=COLORS["orange_face"],
          edge=COLORS["orange_edge"], title="Image aspect ratio (width / height)",
          xlabel="Aspect ratio  (1.0 = square)", vlines=False)
    ax.axvline(1.0, color=COLORS["gray_edge"], linewidth=1.6, linestyle="--",
               label="Square (1:1)")
    ax.legend(loc="upper right")

    ax = fig.add_subplot(gs[2, 2])
    _hist(ax, mpix, bins=np.arange(0, 16.1, 1), face=COLORS["blue_face"],
          edge=COLORS["blue_edge"], title="Image resolution",
          xlabel="Megapixels", clip_pct=99)

    ax = fig.add_subplot(gs[2, 3])
    _hist(ax, mention_lens, bins=range(2, 32), face=COLORS["pink_face"],
          edge=COLORS["pink_edge"], title="Mention length",
          xlabel="# characters")

    # Row 4 -------------------------------------------------------
    ax = fig.add_subplot(gs[3, 0])
    _bar_counts(ax, word_labels, word_vals, COLORS["green_face"], COLORS["green_edge"],
                "Mention length", ylabel="Instances", total=N)
    ax.set_xlabel("# words")

    ax = fig.add_subplot(gs[3, 1])
    homog_labels = ["1 type\n(homogeneous)", "2 types", "3 types\n(all PERS/ORG/LOC)"]
    homog_vals = [pool_n_types.get(k, 0) for k in (1, 2, 3)]
    _bar_counts(ax, homog_labels, homog_vals, COLORS["blue_face"], COLORS["blue_edge"],
                "Candidate pool — entity types present", ylabel="Instances", total=N)

    ax = fig.add_subplot(gs[3, 2])
    _bar_counts(ax, ["Matches majority\ntype in pool", "Differs from\npool majority"],
                [answer_is_majority, N - answer_is_majority],
                COLORS["orange_face"], COLORS["orange_edge"],
                "Answer type vs. candidate pool", ylabel="Instances", total=N)

    ax = fig.add_subplot(gs[3, 3])
    _hist(ax, desc_lens, bins=np.arange(0, 1001, 50), face=COLORS["purple_face"],
          edge=COLORS["purple_edge"],
          title=f"Answer description length  ({pct_answer_infobox:.0f}% have an infobox photo)",
          xlabel="# characters (intro or Wikidata desc)", clip_pct=99)

    save_figure(fig, out)


if __name__ == "__main__":
    main()
