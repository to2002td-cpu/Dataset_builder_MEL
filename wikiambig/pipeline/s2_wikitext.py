"""
S2 — Wikitext parse + QID resolution.

For each disambiguation page from disam_index.jsonl:
  1. Fetch raw wikitext (batch_size pages per API call, threaded).
  2. Parse wikitext with regex to extract entity link targets.
  3. Resolve entity titles to Wikidata QIDs (batch_size titles per call, threaded).

Output: entity_links.jsonl — one JSON object per line:
    {"mention": "Barack", "disam_title": "Barack (disambiguation)", "qids": ["Q76", ...]}
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from wikiambig.api_clients.wikipedia import (
    extract_entity_links,
    get_qids_from_titles,
    get_wikitext_batch,
)
from wikiambig.config import PipelineConfig

logger = logging.getLogger(__name__)

_DISAM_SUFFIX_RE = re.compile(r"\s*\([^)]*\)\s*$")


def _clean_mention(title: str) -> str:
    return _DISAM_SUFFIX_RE.sub("", title).strip()


def _load_disam_index(path: Path) -> list[dict[str, str]]:
    entries = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def run(config: PipelineConfig) -> None:
    input_path = config.stage_path("disam_index.jsonl")
    output_path = config.stage_path("entity_links.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        raise FileNotFoundError(f"S1 output not found: {input_path}. Run S1 first.")

    pages = _load_disam_index(input_path)
    logger.info("S2: processing %d disambiguation pages", len(pages))

    batch_size = config.api_batch_size
    rate = config.wikipedia_rate_limit

    # ── Phase 1: fetch wikitext for all pages (threaded) ──────────────────────
    def _fetch_wikitext(batch: list[dict]) -> dict[str, str]:
        titles = [p["title"] for p in batch]
        result = get_wikitext_batch(titles)
        time.sleep(rate)
        return result

    batches = [pages[i : i + batch_size] for i in range(0, len(pages), batch_size)]
    title_to_links: dict[str, list[str]] = {}

    with ThreadPoolExecutor(max_workers=config.n_workers) as pool:
        futures = {pool.submit(_fetch_wikitext, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="S2 wikitext"):
            try:
                wikitext_map = fut.result()
            except Exception as exc:
                logger.error("Wikitext fetch failed: %s", exc)
                wikitext_map = {}
            for title, wikitext in wikitext_map.items():
                title_to_links[title] = extract_entity_links(wikitext) if wikitext else []

    # ── Phase 2: resolve all unique entity titles → QIDs (threaded) ───────────
    all_entity_titles: list[str] = list(
        {link for links in title_to_links.values() for link in links}
    )
    logger.info("S2: resolving %d unique entity titles to QIDs", len(all_entity_titles))

    def _resolve_qids(batch: list[str]) -> dict[str, str]:
        result = get_qids_from_titles(batch)
        time.sleep(rate)
        return result

    qid_batches = [
        all_entity_titles[i : i + batch_size]
        for i in range(0, len(all_entity_titles), batch_size)
    ]
    title_to_qid: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=config.n_workers) as pool:
        futures = {pool.submit(_resolve_qids, b): b for b in qid_batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="S2 QIDs"):
            try:
                title_to_qid.update(fut.result())
            except Exception as exc:
                logger.error("QID resolution failed: %s", exc)

    # ── Phase 3: assemble entity_links.jsonl ──────────────────────────────────
    results: list[dict] = []
    for page in pages:
        disam_title = page["title"]
        links = title_to_links.get(disam_title, [])
        qids = [title_to_qid[t] for t in links if t in title_to_qid]
        if not qids:
            continue
        results.append(
            {
                "mention": _clean_mention(disam_title),
                "disam_title": disam_title,
                "qids": qids,
            }
        )

    tmp = str(output_path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for entry in results:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    os.replace(tmp, str(output_path))

    total_qids = sum(len(r["qids"]) for r in results)
    logger.info(
        "S2 done: %d mentions, %d entity links → %s",
        len(results),
        total_qids,
        output_path,
    )
