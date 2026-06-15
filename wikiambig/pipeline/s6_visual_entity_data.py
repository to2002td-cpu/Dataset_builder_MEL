"""
S6 — Visual entity data collection.

For every QID that appears in image_data.json's used_by lists but was never
fully enriched in S3 (no entry in entity_types.json — entity_types.json gets
one for every QID it ever processes, even untyped ones, so its keys are a
reliable "already enriched" marker), fetch name, description, Wikipedia URL
AND coarse type in a single combined SPARQL query per batch (the same
get_entities_data_and_types call S3 uses).

Results are merged into entity_data.json and entity_types.json so the viewer
can display names and descriptions for visual candidates. Run S4 again
afterwards to fetch intros/infobox images for these newly-added entities.

Runs after S5 and before S7.
Input:  image_data.json, entity_data.json, entity_types.json
Output: entity_data.json (augmented), entity_types.json (augmented)
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

from tqdm import tqdm

from wikiambig.api_clients.wikidata import get_entities_data_and_types
from wikiambig.checkpoint import Checkpoint
from wikiambig.config import PipelineConfig
from wikiambig.pipeline.utils import atomic_write

logger = logging.getLogger(__name__)


def _collect_visual_qids(image_data_path: Path) -> set[str]:
    image_data: dict = json.loads(image_data_path.read_text(encoding="utf-8"))
    qids: set[str] = set()
    for img_info in image_data.values():
        qids.update(img_info.get("used_by", []))
    return qids


def run(config: PipelineConfig) -> None:
    image_data_path = config.stage_path("image_data.json")
    entity_data_path = config.stage_path("entity_data.json")
    entity_types_path = config.stage_path("entity_types.json")

    for p in [image_data_path, entity_data_path, entity_types_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required input missing: {p}. Run S3/S4/S5 first.")

    logger.info("S6: scanning image_data.json for visual QIDs…")
    all_visual_qids = _collect_visual_qids(image_data_path)
    logger.info("S6: %d unique QIDs found in used_by fields", len(all_visual_qids))

    entity_data: dict = json.loads(entity_data_path.read_text(encoding="utf-8"))
    entity_types: dict = json.loads(entity_types_path.read_text(encoding="utf-8"))

    pending = sorted(q for q in all_visual_qids if q not in entity_types)
    logger.info("S6: %d visual-only QIDs need enrichment (%d already known)",
                len(pending), len(all_visual_qids) - len(pending))

    if not pending:
        logger.info("S6: nothing to do — all visual QIDs already enriched.")
        return

    cp = Checkpoint(config.stage_path("s6.checkpoint.json"))
    pending = cp.pending(pending)

    batch_size = config.entity_data_batch_size
    rate = config.wikidata_rate_limit
    lock = Lock()
    batches_done = 0

    def _sparql_batch(batch: list[str]) -> dict[str, dict]:
        result = get_entities_data_and_types(batch)
        time.sleep(rate)
        return result

    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]

    with ThreadPoolExecutor(max_workers=config.n_workers) as pool:
        futures = {pool.submit(_sparql_batch, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="S6 Wikidata"):
            batch = futures[fut]
            try:
                results = fut.result()
            except Exception as exc:
                logger.error("S6 batch failed: %s", exc)
                results = {}

            with lock:
                for qid in batch:
                    sparql = results.get(qid, {})
                    url = sparql.get("url_wikipedia") or ""
                    if url:
                        entity_data[qid] = {
                            "name": sparql.get("name") or qid,
                            "desc": sparql.get("desc", ""),
                            "url_wikipedia": url,
                        }
                    entity_types[qid] = sparql.get("type")
                    cp.mark_done(qid)

                batches_done += 1
                if batches_done % config.save_every == 0:
                    atomic_write(entity_data_path, entity_data)
                    atomic_write(entity_types_path, entity_types)

    cp.flush()
    atomic_write(entity_data_path, entity_data)
    atomic_write(entity_types_path, entity_types)
    logger.info(
        "S6 done: %d total in entity_data.json, %d total in entity_types.json",
        len(entity_data), len(entity_types),
    )
