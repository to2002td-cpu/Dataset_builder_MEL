#!/usr/bin/env python3
"""
Show all Wikipedia disambiguation categories present in a JSONL file and
the number of mentions (= disambiguation pages) per category, sorted by count.

Works with any file that has a `categories` field per line:
  disam_index.jsonl   — all crawled pages (pre-filtering)
  entity_links.jsonl  — after S2 (has mention field)
  dataset.jsonl       — final assembled dataset

Usage:
    python split_gen/category_stats.py output/scrape_data/disam_index.jsonl
    python split_gen/category_stats.py output/raw_dataset/dataset.jsonl --min-count 10
    python split_gen/category_stats.py output/scrape_data/disam_index.jsonl --tsv > cats.tsv
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

try:
    import orjson
    def _loads(b: bytes) -> dict:
        return orjson.loads(b)
except ImportError:
    import json
    def _loads(b: bytes) -> dict:
        return json.loads(b)

try:
    from rich.console import Console
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


def _iter_jsonl(path: Path):
    with tqdm(total=path.stat().st_size, desc="Reading",
              unit="B", unit_scale=True, unit_divisor=1024,
              file=sys.stderr) as bar, path.open("rb") as f:
            for line in f:
                bar.update(len(line))
                if not line.isspace():
                    yield _loads(line)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("input", type=Path,
                    help="JSONL file with a `categories` field per line")
    ap.add_argument("--min-count", "-m", type=int, default=1,
                    help="hide categories with fewer mentions (default: 1 = show all)")
    ap.add_argument("--top", "-n", type=int, default=0,
                    help="show only top N categories (default: 0 = all)")
    ap.add_argument("--tsv", action="store_true",
                    help="output as TSV instead of a rich table (useful for piping)")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")

    cat_counter: Counter = Counter()
    uncategorised = 0
    total = 0

    for record in _iter_jsonl(args.input):
        total += 1
        cats = record.get("categories") or []
        if not cats:
            uncategorised += 1
        for c in cats:
            cat_counter[c] += 1

    rows = [(cat, cnt) for cat, cnt in cat_counter.most_common() if cnt >= args.min_count]
    if args.top > 0:
        rows = rows[: args.top]

    if args.tsv:
        print("category\tcount")
        for cat, cnt in rows:
            print(f"{cat}\t{cnt}")
        print(f"(no category)\t{uncategorised}", file=sys.stderr)
        return

    # ── Rich table ─────────────────────────────────────────────
    print(f"\nTotal records : {total:,}", file=sys.stderr)
    print(f"Uncategorised : {uncategorised:,}  ({100*uncategorised/max(total,1):.1f}%)",
          file=sys.stderr)
    print(f"Unique categories : {len(cat_counter):,}\n", file=sys.stderr)

    if _RICH:
        console = Console()
        table = Table(title=f"Category stats — {args.input.name}", show_lines=False)
        table.add_column("#", style="dim", width=5, justify="right")
        table.add_column("Category", style="bold")
        table.add_column("Mentions", justify="right", style="cyan")
        table.add_column("% of total", justify="right", style="green")

        for rank, (cat, cnt) in enumerate(rows, 1):
            pct = f"{100 * cnt / max(total, 1):.1f}%"
            # strip "Category:" prefix for display, keep it in tooltip / raw value
            label = cat.removeprefix("Category:")
            table.add_row(str(rank), label, f"{cnt:,}", pct)

        if uncategorised:
            table.add_row("—", "(no category)", f"{uncategorised:,}",
                          f"{100*uncategorised/max(total,1):.1f}%")

        console.print(table)
    else:
        # plain text fallback
        col_w = max((len(c) for c, _ in rows), default=20)
        print(f"{'#':>5}  {'Category':<{col_w}}  {'Count':>8}  {'%':>6}")
        print("-" * (col_w + 25))
        for rank, (cat, cnt) in enumerate(rows, 1):
            pct = f"{100 * cnt / max(total, 1):.1f}%"
            print(f"{rank:>5}  {cat:<{col_w}}  {cnt:>8,}  {pct:>6}")
        if uncategorised:
            print(f"{'—':>5}  {'(no category)':<{col_w}}  {uncategorised:>8,}  "
                  f"{100*uncategorised/max(total,1):>5.1f}%")


if __name__ == "__main__":
    main()
