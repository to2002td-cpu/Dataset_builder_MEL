#!/usr/bin/env python3
"""
Word frequency visualisation for mentions in the final MEL dataset.

Left panel:  word cloud of the words appearing in mentions.
Right panel: top-N words as a horizontal bar chart with exact counts.

Usage:
    python fig_gen/mention_wordcloud.py output/split_10_text/instances.jsonl
    python fig_gen/mention_wordcloud.py output/split_10_text/instances.jsonl --out output/figures/mention_wordcloud.pdf
    python fig_gen/mention_wordcloud.py output/split_10_text/instances.jsonl --unique   # count each distinct mention once
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud

from utils import COLORS, PALETTE, apply_style, iter_jsonl, save_figure

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # alphabetic runs only


def count_words(instances_path: Path, unique: bool) -> tuple[Counter, int]:
    """Word frequencies across mentions; returns (counter, n_mentions)."""
    mentions = (m["mention"] for m in iter_jsonl(instances_path))
    if unique:
        mentions = set(mentions)

    counts: Counter = Counter()
    n_mentions = 0
    for mention in mentions:
        n_mentions += 1
        for w in _WORD_RE.findall(mention.lower()):
            if len(w) >= 2 and w not in STOPWORDS:
                counts[w] += 1
    return counts, n_mentions


def _color_func(word, *args, **kwargs):
    return PALETTE[hash(word) % len(PALETTE)]


def plot(counts: Counter, n_mentions: int, out_path: Path, top_n: int, unique: bool) -> None:
    apply_style()
    fig, (ax_cloud, ax_bar) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.7, 1]})

    # ── Word cloud ────────────────────────────────────────────────────────────
    wc = WordCloud(width=1600, height=900, background_color="white",
                   color_func=_color_func, max_words=200,
                   prefer_horizontal=0.95, random_state=42)
    wc.generate_from_frequencies(counts)
    ax_cloud.imshow(wc, interpolation="bilinear")
    ax_cloud.axis("off")
    ax_cloud.grid(False)
    scope = "unique mentions" if unique else "mentions"
    ax_cloud.set_title(f"Words in {scope} ({n_mentions:,} {scope}, "
                       f"{len(counts):,} distinct words)",
                       fontsize=10.5, fontweight="bold", pad=6)

    # ── Top-N bar chart ───────────────────────────────────────────────────────
    top = counts.most_common(top_n)
    labels = [w for w, _ in reversed(top)]
    vals   = [c for _, c in reversed(top)]
    bars = ax_bar.barh(labels, vals, color=COLORS["blue_face"], edgecolor=COLORS["blue_edge"],
                       linewidth=1.8, hatch="//", zorder=3, height=0.65)
    xmax = max(vals) * 1.18 if vals else 1
    ax_bar.set_xlim(0, xmax)
    for b, v in zip(bars, vals):
        ax_bar.text(v + xmax * 0.012, b.get_y() + b.get_height() / 2,
                    f"{v:,}", va="center", fontsize=7.5, color="#333")
    ax_bar.set_xlabel("Occurrences")
    ax_bar.set_title(f"Top {len(top)} words", fontsize=10.5, fontweight="bold", pad=6)
    ax_bar.tick_params(axis="y", labelsize=8)

    fig.tight_layout()
    save_figure(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", type=Path, nargs="?",
                    default=Path("output/split_10_text/instances.jsonl"),
                    help="instances.jsonl (default: output/split_10_text/instances.jsonl)")
    ap.add_argument("--out", type=Path, default=Path("output/figures/mention_wordcloud.pdf"),
                    help="output figure path (default: output/figures/mention_wordcloud.pdf)")
    ap.add_argument("--top", type=int, default=25,
                    help="number of words in the bar chart (default: 25)")
    ap.add_argument("--unique", action="store_true",
                    help="count each distinct mention once instead of once per instance")
    args = ap.parse_args()

    if not args.instances.exists():
        raise SystemExit(f"Not found: {args.instances}")

    counts, n_mentions = count_words(args.instances, args.unique)
    if not counts:
        raise SystemExit("No words found in mentions.")
    plot(counts, n_mentions, args.out, args.top, args.unique)


if __name__ == "__main__":
    main()
