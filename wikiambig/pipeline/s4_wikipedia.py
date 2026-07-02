"""
S4 — Wikipedia entity enrichment.

For every entity with a Wikipedia URL — regardless of type, so OTHER-typed
entities carry full multimodal data too — fetches the intro paragraph and
image filename list in a single API call (``prop=extracts|images``), halving
the number of Wikipedia requests compared to separate passes.

Inputs:  entity_data.json
Outputs: entity_intros.json         {QID: "full first paragraph, verbatim, or ''"}
         image_lists.json            {QID: [filename, …]}
         entity_infobox_images.json
             {QID: "https://en.wikipedia.org/wiki/Special:FilePath/<file>" | None}
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from urllib.parse import quote

from tqdm import tqdm

from wikiambig.api_clients.wikipedia import get_wiki_entity_data_batch
from wikiambig.checkpoint import Checkpoint
from wikiambig.config import PipelineConfig
from wikiambig.pipeline.utils import atomic_write, title_from_url

logger = logging.getLogger(__name__)


def run(config: PipelineConfig) -> None:
    entity_data_path  = config.stage_path("entity_data.json")
    intros_path       = config.stage_path("entity_intros.json")
    images_path       = config.stage_path("image_lists.json")
    infobox_path      = config.stage_path("entity_infobox_images.json")
    cp = Checkpoint(config.stage_path("s4.checkpoint.json"))

    if not entity_data_path.exists():
        raise FileNotFoundError(f"S3 output not found: {entity_data_path}. Run S3 first.")

    entity_data: dict[str, dict] = json.loads(entity_data_path.read_text(encoding="utf-8"))

    qid_to_title: dict[str, str] = {
        qid: title_from_url(d["url_wikipedia"])
        for qid, d in entity_data.items()
        if d.get("url_wikipedia")
    }

    all_qids = sorted(qid_to_title)
    pending  = cp.pending(all_qids)
    logger.info("S4: %d entities, %d done, %d pending", len(all_qids), cp.n_done, len(pending))

    entity_intros: dict[str, str] = {}
    if intros_path.exists():
        with contextlib.suppress(json.JSONDecodeError):
            entity_intros = json.loads(intros_path.read_text(encoding="utf-8"))

    image_lists: dict[str, list[str]] = {}
    if images_path.exists():
        with contextlib.suppress(json.JSONDecodeError):
            image_lists = json.loads(images_path.read_text(encoding="utf-8"))

    entity_infobox_images: dict[str, str | None] = {}
    if infobox_path.exists():
        with contextlib.suppress(json.JSONDecodeError):
            entity_infobox_images = json.loads(infobox_path.read_text(encoding="utf-8"))

    batch_size = 50  # MediaWiki hard limit for extracts + images
    rate = config.wikipedia_rate_limit
    output_lock = Lock()

    def _fetch_batch(
        batch_qids: list[str],
    ) -> tuple[dict[str, str], dict[str, list[str]], dict[str, str | None]]:
        titles = [qid_to_title[q] for q in batch_qids]
        wiki_result = get_wiki_entity_data_batch(titles)
        time.sleep(rate)
        intros = {q: wiki_result.get(qid_to_title[q], {}).get("intro", "") for q in batch_qids}
        images = {q: wiki_result.get(qid_to_title[q], {}).get("images", []) for q in batch_qids}
        infobox_images: dict[str, str | None] = {}
        for q in batch_qids:
            fname = wiki_result.get(qid_to_title[q], {}).get("infobox_image")
            infobox_images[q] = (
                f"https://en.wikipedia.org/wiki/Special:FilePath/{quote(fname)}" if fname else None
            )
        return intros, images, infobox_images

    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    batches_done = 0

    with ThreadPoolExecutor(max_workers=config.n_workers) as pool:
        futures = {pool.submit(_fetch_batch, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="S4 Wikipedia"):
            batch = futures[fut]
            try:
                intros, images, infobox_images = fut.result()
                with output_lock:
                    entity_intros.update(intros)
                    image_lists.update(images)
                    entity_infobox_images.update(infobox_images)
                cp.mark_done_batch(batch)
            except Exception as exc:
                logger.error("S4 batch failed: %s", exc)

            batches_done += 1
            if batches_done % config.save_every == 0:
                with output_lock:
                    atomic_write(intros_path, entity_intros)
                    atomic_write(images_path, image_lists)
                    atomic_write(infobox_path, entity_infobox_images)

    cp.flush()
    atomic_write(intros_path, entity_intros)
    atomic_write(images_path, image_lists)
    atomic_write(infobox_path, entity_infobox_images)

    n_intros    = sum(1 for v in entity_intros.values() if v)
    n_infobox   = sum(1 for v in entity_infobox_images.values() if v)
    total_imgs  = sum(len(v) for v in image_lists.values())
    logger.info(
        "S4 done: %d intros (%.0f%% non-empty), %d infobox images (%.0f%% non-empty), "
        "%d entities × avg %.1f images",
        n_intros, 100 * n_intros / max(len(entity_intros), 1),
        n_infobox, 100 * n_infobox / max(len(entity_infobox_images), 1),
        len(image_lists), total_imgs / max(len(image_lists), 1),
    )
