"""
S3 — Wikidata entity enrichment.

A single combined SPARQL query per batch (200 QIDs) fetches entity data
(name, description, Wikipedia URL, infobox image) AND coarse type
(PERS / ORG / LOC via P31/P279* root traversal) in one round-trip —
half the Wikidata SPARQL traffic of running the two as separate passes.

Inputs:  entity_links.jsonl
Outputs: entity_data.json   {QID: {name, desc, url_wikipedia, infobox_img}}
         entity_types.json  {QID: "PERS"|"ORG"|"LOC"|None}
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm

from wikiambig.api_clients.wikidata import get_entities_data_and_types
from wikiambig.checkpoint import Checkpoint
from wikiambig.config import PipelineConfig
from wikiambig.pipeline.utils import atomic_write, load_qids_from_jsonl

logger = logging.getLogger(__name__)


def _load_json(path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    return {}


def run(config: PipelineConfig) -> None:
    """Fetch entity data and types for every QID via one combined SPARQL pass."""
    input_path = config.stage_path("entity_links.jsonl")
    if not input_path.exists():
        raise FileNotFoundError(f"S2 output not found: {input_path}. Run S2 first.")

    data_path = config.stage_path("entity_data.json")
    types_path = config.stage_path("entity_types.json")

    entity_data: dict[str, dict] = _load_json(data_path)
    entity_types: dict[str, str | None] = _load_json(types_path)

    cp = Checkpoint(config.stage_path("s3.checkpoint.json"))
    if cp.n_done == 0 and entity_types:
        # Migration from the old two-phase S3 (or resume of an interrupted
        # combined run that wrote outputs before the checkpoint flushed):
        # entity_types.json gets an entry — even None — for every QID that
        # was ever fully processed, so its keys are a reliable "done" set.
        cp.mark_done_batch(list(entity_types))
        cp.flush()
        logger.info("S3: seeded checkpoint with %d already-enriched QIDs", len(entity_types))

    all_qids = load_qids_from_jsonl(input_path)
    pending = cp.pending(all_qids)
    logger.info("S3: %d unique QIDs, %d done, %d pending", len(all_qids), cp.n_done, len(pending))

    output_lock = Lock()
    batch_size = config.entity_data_batch_size
    rate = config.wikidata_rate_limit

    def _sparql_batch(batch: list[str]) -> dict[str, dict]:
        result = get_entities_data_and_types(batch)
        time.sleep(rate)
        return result

    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    batches_done = 0

    with ThreadPoolExecutor(max_workers=config.n_workers) as pool:
        futures = {pool.submit(_sparql_batch, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="S3 Wikidata"):
            batch = futures[fut]
            try:
                results = fut.result()
            except Exception as exc:
                logger.error("S3 batch failed: %s", exc)
                results = {}

            with output_lock:
                for qid in batch:
                    sparql = results.get(qid, {})
                    url = sparql.get("url_wikipedia") or ""
                    if url:
                        entity_data[qid] = {
                            "name": sparql.get("name") or qid,
                            "desc": sparql.get("desc", ""),
                            "url_wikipedia": url,
                            "infobox_img": sparql.get("infobox_img"),
                        }
                    entity_types[qid] = sparql.get("type")
                    cp.mark_done(qid)

                batches_done += 1
                if batches_done % config.save_every == 0:
                    atomic_write(data_path, entity_data)
                    atomic_write(types_path, entity_types)

    cp.flush()
    atomic_write(data_path, entity_data)
    atomic_write(types_path, entity_types)

    counts: dict[str | None, int] = {}
    for t in entity_types.values():
        counts[t] = counts.get(t, 0) + 1
    logger.info(
        "S3 done: %d entities → %s; PERS=%d ORG=%d LOC=%d OTHER/None=%d → %s",
        len(entity_data), data_path,
        counts.get("PERS", 0), counts.get("ORG", 0), counts.get("LOC", 0), counts.get(None, 0),
        types_path,
    )
