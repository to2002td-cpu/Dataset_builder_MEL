#!/usr/bin/env python3
"""
Word frequency visualisation for mentions in the final MEL dataset.

Left panel:  word cloud of the words appearing in mentions.
Right panel: top-N words as a horizontal bar chart with exact counts.

Usage:
    python fig_gen/mention_wordcloud.py output/pilot/instances.jsonl
    python fig_gen/mention_wordcloud.py output/pilot/instances.jsonl --out out.pdf
    python fig_gen/mention_wordcloud.py output/pilot/instances.jsonl --unique
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from plots import barh_counts, set_panel_title
from utils import COLORS, PALETTE, apply_style, iter_jsonl, save_figure
from wordcloud import STOPWORDS, WordCloud

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
    set_panel_title(ax_cloud, f"Words in {scope} ({n_mentions:,} {scope}, "
                              f"{len(counts):,} distinct words)")

    # ── Top-N bar chart ───────────────────────────────────────────────────────
    top = counts.most_common(top_n)
    barh_counts(ax_bar, [w for w, _ in top], [c for _, c in top],
                COLORS["blue_face"], COLORS["blue_edge"],
                f"Top {len(top)} words", xlabel="Occurrences")

    fig.tight_layout()
    save_figure(fig, out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("instances", type=Path, nargs="?",
                    default=Path("output/pilot/instances.jsonl"),
                    help="instances.jsonl (default: output/pilot/instances.jsonl)")
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
