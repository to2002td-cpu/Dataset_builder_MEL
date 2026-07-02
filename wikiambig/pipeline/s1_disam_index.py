"""
S1 — Disambiguation page index.

Crawls Category:Disambiguation pages and all of its subcategories
(breadth-first, recursively) and writes one JSON object per line to
disam_index.jsonl. Each entry records every (sub)category the page was
found in, so category-based filtering can be done offline in later stages
instead of re-querying Wikipedia.

Output format (one per line):
    {"title": "Barack (disambiguation)",
     "url": "https://en.wikipedia.org/wiki/Barack_(disambiguation)",
     "categories": ["Category:Disambiguation pages",
                    "Category:Human name disambiguation pages", ...]}
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm

from wikiambig.api_clients.wikipedia import get_category_members
from wikiambig.config import PipelineConfig

logger = logging.getLogger(__name__)

_NUMERIC_RE = re.compile(r"^\d+$")

ROOT_CATEGORY = "Category:Disambiguation pages"


def _is_valid_title(title: str) -> bool:
    return not _NUMERIC_RE.match(title)


def _title_to_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")


def run(config: PipelineConfig) -> None:
    output_path = config.stage_path("disam_index.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    visited_cats: set[str] = set()
    page_categories: dict[str, set[str]] = {}
    lock = Lock()
    rate = config.wikipedia_rate_limit

    def _crawl(category: str) -> tuple[str, list[dict]]:
        members = get_category_members(category, cmtype="page|subcat")
        time.sleep(rate)
        return category, members

    frontier = [ROOT_CATEGORY]
    visited_cats.add(ROOT_CATEGORY)
    pbar = tqdm(desc="S1 categories")

    with ThreadPoolExecutor(max_workers=config.n_workers) as pool:
        while frontier:
            futures = {pool.submit(_crawl, cat): cat for cat in frontier}
            frontier = []

            for fut in as_completed(futures):
                cat = futures[fut]
                try:
                    _, members = fut.result()
                except Exception as exc:
                    logger.error("Failed to crawl %s: %s", cat, exc)
                    continue
                pbar.update(1)

                with lock:
                    for m in members:
                        title = m["title"]
                        if m["ns"] == 14:  # subcategory
                            if title not in visited_cats:
                                visited_cats.add(title)
                                frontier.append(title)
                        elif m["ns"] == 0 and _is_valid_title(title):  # article
                            page_categories.setdefault(title, set()).add(cat)

    pbar.close()

    tmp = str(output_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for title in sorted(page_categories):
            entry = {
                "title": title,
                "url": _title_to_url(title),
                "categories": sorted(page_categories[title]),
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp, str(output_path))

    logger.info(
        "S1 done: %d disambiguation pages across %d categories → %s",
        len(page_categories), len(visited_cats), output_path,
    )
