"""
S4b — Commons gallery image source.

Second, independent image source for each entity, parallel to the en.wikipedia
article images from S4. For every QID with a Commons gallery page (the
``commonswiki`` sitelink), fetches the list of image filenames embedded on that
gallery page. Both sources are kept side by side (image_lists.json vs
image_lists_commons.json) so they can be compared downstream; the rest of the
pipeline (S5 file usage, S7 assembly) consumes the union of the two.

Galleries only — entities with no Commons gallery sitelink are skipped (no
P373 category fallback, by design).

Inputs:  entity_data.json
Outputs: image_lists_commons.json     {QID: [filename, …]}
         entity_commons_pages.json    {QID: "<gallery title>" | None}
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from tqdm import tqdm

from wikiambig.api_clients.commons import (
    get_commons_gallery_titles,
    get_commons_images_batch,
)
from wikiambig.checkpoint import Checkpoint
from wikiambig.config import PipelineConfig
from wikiambig.pipeline.utils import atomic_write

logger = logging.getLogger(__name__)

BATCH_SIZE = 50  # wbgetentities ids limit and prop=images titles limit


def run(config: PipelineConfig) -> None:
    entity_data_path = config.stage_path("entity_data.json")
    images_path = config.stage_path("image_lists_commons.json")
    pages_path = config.stage_path("entity_commons_pages.json")
    cp = Checkpoint(config.stage_path("s4b.checkpoint.json"))

    if not entity_data_path.exists():
        raise FileNotFoundError(f"S3 output not found: {entity_data_path}. Run S3 first.")

    entity_data: dict[str, dict] = json.loads(entity_data_path.read_text(encoding="utf-8"))
    all_qids = sorted(entity_data)
    pending = cp.pending(all_qids)
    logger.info("S4b: %d entities, %d done, %d pending", len(all_qids), cp.n_done, len(pending))

    image_lists: dict[str, list[str]] = {}
    if images_path.exists():
        try:
            image_lists = json.loads(images_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    commons_pages: dict[str, str | None] = {}
    if pages_path.exists():
        try:
            commons_pages = json.loads(pages_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass

    rate = config.wikidata_rate_limit
    output_lock = Lock()

    def _fetch_batch(batch_qids: list[str]) -> tuple[dict[str, str | None], dict[str, list[str]]]:
        # 1. QID → Commons gallery title (Wikidata sitelinks)
        qid_to_title = get_commons_gallery_titles(batch_qids)
        time.sleep(rate)
        pages: dict[str, str | None] = {q: qid_to_title.get(q) for q in batch_qids}

        # 2. gallery title → embedded image filenames (Commons prop=images)
        images: dict[str, list[str]] = {q: [] for q in batch_qids}
        titles = list(dict.fromkeys(qid_to_title.values()))
        if titles:
            title_imgs = get_commons_images_batch(titles)
            time.sleep(rate)
            for q, t in qid_to_title.items():
                images[q] = title_imgs.get(t, [])
        return pages, images

    batches = [pending[i : i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)]
    batches_done = 0

    with ThreadPoolExecutor(max_workers=config.n_workers) as pool:
        futures = {pool.submit(_fetch_batch, b): b for b in batches}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="S4b Commons"):
            batch = futures[fut]
            try:
                pages, images = fut.result()
                with output_lock:
                    commons_pages.update(pages)
                    image_lists.update(images)
                cp.mark_done_batch(batch)
            except Exception as exc:
                logger.error("S4b batch failed: %s", exc)

            batches_done += 1
            if batches_done % config.save_every == 0:
                with output_lock:
                    atomic_write(images_path, image_lists)
                    atomic_write(pages_path, commons_pages)

    cp.flush()
    atomic_write(images_path, image_lists)
    atomic_write(pages_path, commons_pages)

    n_gallery = sum(1 for v in commons_pages.values() if v)
    total_imgs = sum(len(v) for v in image_lists.values())
    logger.info(
        "S4b done: %d/%d entities have a Commons gallery, %d entities × avg %.1f images → %s",
        n_gallery, len(commons_pages),
        len(image_lists), total_imgs / max(len(image_lists), 1),
        images_path,
    )
