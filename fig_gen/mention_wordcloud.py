#!/usr/bin/env python3
"""
Word frequency visualisation for mentions in the final MEL dataset.

Left panel:  word cloud of the words appearing in mentions.
Right panel: top-N words as a horizontal bar chart with exact counts.

Usage:
    python scripts/mention_wordcloud.py output/final/instances.jsonl
    python scripts/mention_wordcloud.py output/final/instances.jsonl --out figures/mention_wordcloud.pdf
    python scripts/mention_wordcloud.py output/final/instances.jsonl --unique   # count each distinct mention once
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from wordcloud import STOPWORDS, WordCloud

PALETTE = ["#0b3c6d", "#d81b60", "#00695c", "#b45309", "#5b21b6"]

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)  # alphabetic runs only


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
        "axes.labelsize": 11,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": "#c9c9c9",
        "grid.linewidth": 1.0,
        "grid.alpha": 1.0,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "hatch.linewidth": 1.5,
    })


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def count_words(instances_path: Path, unique: bool) -> tuple[Counter, int]:
    """Word frequencies across mentions; returns (counter, n_mentions)."""
    mentions = (m["mention"] for m in _iter_jsonl(instances_path))
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
    _style()
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
    bars = ax_bar.barh(labels, vals, color="#b7d4ea", edgecolor="#0b3c6d",
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
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    print(f"Figure written → {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", type=Path, nargs="?",
                    default=Path("output/final/instances.jsonl"),
                    help="instances.jsonl (default: output/final/instances.jsonl)")
    ap.add_argument("--out", type=Path, default=Path("figures/mention_wordcloud.pdf"),
                    help="output figure path (default: figures/mention_wordcloud.pdf)")
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
