"""
S5 — Image usage scrape.

For each unique image filename from image_lists.json:
  Step B: Wikipedia imageinfo + fileusage (direct URL + which articles embed it).
  Step C: Wikidata wbgetentities to resolve article titles → QIDs.

Global caches (thread-safe) ensure each image and each title is resolved
at most once per run, regardless of how many entities reference it.

Output: image_data.json — {filename: {url, used_by, width, height, mime, license}}
"""

from __future__ import annotations

import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Optional

from tqdm import tqdm

from wikiambig.api_clients.wikipedia import get_image_info_batch
from wikiambig.api_clients.wikidata import get_titles_to_qids
from wikiambig.checkpoint import Checkpoint
from wikiambig.config import PipelineConfig
from wikiambig.pipeline.utils import atomic_write, title_from_url

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global caches (shared across all worker threads for the entire run)
# ---------------------------------------------------------------------------

# image cache: filename → {"url": str, "used_by": [QID, ...]} | None on failure
_image_cache: dict[str, Optional[dict]] = {}
_image_cache_lock = Lock()

# title→QID resolution cache
_title_qid_cache: dict[str, Optional[str]] = {}
_title_cache_lock = Lock()


def _resolve_titles(titles: list[str], batch_size: int = 50) -> dict[str, str]:
    """Resolve article titles → QIDs, using the global cache."""
    with _title_cache_lock:
        unknown = [t for t in titles if t not in _title_qid_cache]

    if unknown:
        fetched: dict[str, str] = {}
        for i in range(0, len(unknown), batch_size):
            batch = unknown[i : i + batch_size]
            try:
                fetched.update(get_titles_to_qids(batch))
            except Exception as exc:
                logger.warning("S5 title resolution batch failed: %s", exc)

        with _title_cache_lock:
            for t in unknown:
                _title_qid_cache[t] = fetched.get(t)

    with _title_cache_lock:
        return {t: _title_qid_cache[t] for t in titles if _title_qid_cache.get(t)}


def _process_image_batch(
    filenames: list[str],
    config: PipelineConfig,
) -> dict[str, dict]:
    """
    Fetch imageinfo + fileusage for a batch of filenames (Step B),
    then resolve article titles to QIDs (Step C).
    Returns {filename: {url, used_by}}.
    """
    with _image_cache_lock:
        cached = {f: _image_cache[f] for f in filenames if f in _image_cache}
        uncached = [f for f in filenames if f not in _image_cache]

    if not uncached:
        return {f: v for f, v in cached.items() if v is not None}

    # Step B
    try:
        raw = get_image_info_batch(uncached)
    except Exception as exc:
        logger.error("S5 imageinfo batch failed: %s", exc)
        raw = {f: {"url": "", "usage": []} for f in uncached}

    # Step C — resolve all unique usage titles to QIDs
    all_titles = list({t for info in raw.values() for t in info["usage"]})
    title_to_qid = _resolve_titles(all_titles, batch_size=config.api_batch_size)

    new_entries: dict[str, Optional[dict]] = {}
    for fname, info in raw.items():
        url = info["url"]
        used_by = list(
            dict.fromkeys(title_to_qid[t] for t in info["usage"] if t in title_to_qid)
        )
        new_entries[fname] = (
            {
                "url": url,
                "used_by": used_by,
                "width": info.get("width"),
                "height": info.get("height"),
                "mime": info.get("mime", ""),
                "license": info.get("license", ""),
            }
            if url
            else None
        )

    with _image_cache_lock:
        _image_cache.update(new_entries)
        for f, v in new_entries.items():
            if v is not None:
                cached[f] = v

    return {f: v for f, v in cached.items() if v is not None}


def run(config: PipelineConfig) -> None:
    """
    Fetch image URL + usage for all unique filenames in image_lists.json.
    Resumable via checkpoint on filenames.
    """
    image_lists_path = config.stage_path("image_lists.json")
    output_path = config.stage_path("image_data.json")
    cp = Checkpoint(config.stage_path("s5.checkpoint.json"))

    if not image_lists_path.exists():
        raise FileNotFoundError(
            f"S5 output not found: {image_lists_path}. Run S4 first."
        )

    image_lists: dict[str, list[str]] = json.loads(
        image_lists_path.read_text(encoding="utf-8")
    )
    all_filenames = sorted({f for fnames in image_lists.values() for f in fnames})
    pending = cp.pending(all_filenames)

    logger.info(
        "S5: %d unique image filenames, %d already done, %d pending",
        len(all_filenames),
        cp.n_done,
        len(pending),
    )

    # Pre-populate the title→QID cache from entity_data so S6 doesn't re-resolve
    # titles we already fetched in S3 (covers most of the high-frequency article titles).
    entity_data_path = config.stage_path("entity_data.json")
    if entity_data_path.exists():
        try:
            _ed = json.loads(entity_data_path.read_text(encoding="utf-8"))
            with _title_cache_lock:
                for _qid, _d in _ed.items():
                    _url = _d.get("url_wikipedia", "")
                    if _url:
                        _title = title_from_url(_url)
                        _title_qid_cache[_title] = _qid
            logger.info("S5: pre-seeded title cache with %d known entities", len(_ed))
        except Exception:
            pass

    # Load existing image data into cache.
    image_data: dict[str, dict] = {}
    if output_path.exists():
        try:
            image_data = json.loads(output_path.read_text(encoding="utf-8"))
            with _image_cache_lock:
                _image_cache.update(image_data)
        except json.JSONDecodeError:
            pass

    batch_size = config.api_batch_size
    batches = [pending[i : i + batch_size] for i in range(0, len(pending), batch_size)]
    batches_done = 0
    output_lock = Lock()

    def _run_batch(batch: list[str]) -> dict[str, dict]:
        result = _process_image_batch(batch, config)
        time.sleep(config.wikipedia_rate_limit)
        return result

    with ThreadPoolExecutor(max_workers=config.n_workers) as executor:
        future_to_batch = {executor.submit(_run_batch, b): b for b in batches}
        with tqdm(
            total=len(pending), desc="S5 image data", unit="file"
        ) as pbar:
            for future in as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    result = future.result()
                    with output_lock:
                        image_data.update(result)
                    cp.mark_done_batch(batch)
                except Exception as exc:
                    logger.error("S5 batch failed: %s", exc)

                pbar.update(len(batch))
                batches_done += 1

                if batches_done % config.save_every == 0:
                    with output_lock:
                        atomic_write(output_path, image_data)
                    logger.info("S5 checkpoint: %d images written", len(image_data))

    cp.flush()
    atomic_write(output_path, image_data)

    total_used = sum(len(v.get("used_by", [])) for v in image_data.values())
    logger.info(
        "S5 done: %d images, avg used_by=%.1f → %s",
        len(image_data),
        total_used / max(len(image_data), 1),
        output_path,
    )
