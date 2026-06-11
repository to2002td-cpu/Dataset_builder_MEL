"""
S1 — Disambiguation page index.

Crawls the configured Wikipedia disambiguation sub-categories and writes
one JSON object per line to disam_index.jsonl.

Output format (one per line):
    {"title": "Barack (disambiguation)", "url": "https://en.wikipedia.org/wiki/Barack_(disambiguation)"}
"""

from __future__ import annotations

import json
import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm

from wikiambig.api_clients.wikipedia import get_category_members
from wikiambig.config import PipelineConfig

logger = logging.getLogger(__name__)

_NUMERIC_RE = re.compile(r"^\d+$")


def _is_valid_title(title: str) -> bool:
    return not _NUMERIC_RE.match(title)


def _title_to_url(title: str) -> str:
    return "https://en.wikipedia.org/wiki/" + title.replace(" ", "_")


def run(config: PipelineConfig) -> None:
    output_path = config.stage_path("disam_index.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    seen_titles: set[str] = set()
    entries: list[dict[str, str]] = []
    lock = Lock()

    def _crawl(category: str) -> tuple[str, list[dict]]:
        members = get_category_members(category)
        return category, members

    n_cats = len(config.disam_categories)
    with ThreadPoolExecutor(max_workers=n_cats) as pool:
        futures = {pool.submit(_crawl, cat): cat for cat in config.disam_categories}
        for fut in tqdm(as_completed(futures), total=n_cats, desc="S1 categories"):
            cat = futures[fut]
            try:
                _, members = fut.result()
            except Exception as exc:
                logger.error("Failed to crawl %s: %s", cat, exc)
                continue

            added = 0
            with lock:
                for m in members:
                    title = m["title"]
                    if title in seen_titles or not _is_valid_title(title):
                        continue
                    seen_titles.add(title)
                    entries.append({"title": title, "url": _title_to_url(title)})
                    added += 1
            logger.info("  %s: +%d pages", cat, added)

    tmp = str(output_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp, str(output_path))

    logger.info("S1 done: %d disambiguation pages → %s", len(entries), output_path)
