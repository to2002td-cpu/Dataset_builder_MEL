"""
S7 — Dataset assembly (offline, deterministic).

Joins all intermediate stage outputs into the final dataset.json and entity_kb.json.

Assembly algorithm:
  1. Load entity_links.jsonl, entity_data.json, entity_types.json,
     image_lists.json, image_data.json.
  2. Build the entity KB: one Entity object per QID.
  3. For each Entity, populate page_imglist:
       - Resolve each filename → Image(url, used_by, n_used_by, is_infobox,
         width, height, mime, license).
       - is_infobox: True when the image filename matches the entity's Wikipedia infobox image
         (entity_infobox_images.json from S4, derived from PageImages).
  4. For each disambiguation page entry in entity_links.jsonl, build a MentionEntry:
       - Populate ambiguities with Entity objects (in original link order).
       - Carry over the page's categories (from S1, via S2) for offline filtering.
       - Compute n_entities and n_visual_ambiguities.
  5. Write dataset.json, dataset.jsonl, entity_kb.json, manifest.json (all atomic).

manifest.json records what produced the dataset and when — package version,
UTC assembly timestamp, the resolved config, and headline counts. Wikipedia
and Wikidata are live, continuously-edited sources with no fixed dump
version, so this timestamp is the only way a future reader can know which
snapshot of the encyclopedia a given dataset.json reflects.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
from datetime import datetime, timezone

from tqdm import tqdm

from wikiambig import __version__
from wikiambig.config import PipelineConfig
from wikiambig.models import Dataset, Entity, Image, MentionEntry
from wikiambig.pipeline.utils import atomic_write

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Visual-signal helpers (used to populate n_visual_ambiguities / n_grounded)
# ---------------------------------------------------------------------------

def _entity_has_external_anchor(entity: Entity, candidate_qids: set[str]) -> bool:
    """True if entity has ≥1 image that is shared with non-candidates but no other candidate."""
    other_candidates = candidate_qids - {entity.qid}
    for img in entity.page_imglist:
        if img.is_infobox:
            continue
        used_by_set = set(img.used_by)
        if used_by_set & other_candidates:
            continue
        if used_by_set - candidate_qids:
            return True
    return False


def _compute_visual_counts(entities: list[Entity]) -> tuple[int, int]:
    """Return (n_visual_ambiguities, n_grounded_entities) for a candidate set."""
    candidate_qids = {e.qid for e in entities}
    shared_urls: set[str] = set()
    for e in entities:
        for img in e.page_imglist:
            if len(set(img.used_by) & candidate_qids) >= 2:
                shared_urls.add(img.url)
    n_grounded = sum(
        1 for e in entities if _entity_has_external_anchor(e, candidate_qids)
    )
    return len(shared_urls), n_grounded


# ---------------------------------------------------------------------------
# is_infobox cross-reference helpers
# ---------------------------------------------------------------------------

def _normalise_image_name(url_or_filename: str) -> str:
    """
    Extract and normalise the filename from a Wikimedia URL or a 'File:...' string.

    Handles:
      - 'File:Charles IV duc de Lorraine.jpg'
      - 'http://commons.wikimedia.org/wiki/Special:FilePath/Charles%20IV%20duc%20de%20Lorraine.jpg'
      - 'https://upload.wikimedia.org/wikipedia/commons/a/ab/Charles_IV_duc_de_Lorraine.jpg'
    """
    s = url_or_filename

    if s.startswith("File:"):
        s = s[5:]
    elif "Special:FilePath/" in s:
        s = s.split("Special:FilePath/", 1)[-1]
    elif "/wikipedia/commons/" in s:
        # CDN URL: take the last path component
        s = s.rstrip("/").split("/")[-1]

    # URL-decode percent-encoding
    s = urllib.parse.unquote(s)
    # Normalise underscores and spaces, lowercase for comparison
    s = s.replace("_", " ").strip().lower()
    return s


def _is_infobox(filename: str, infobox_img: str | None) -> bool:
    if not infobox_img:
        return False
    return _normalise_image_name(filename) == _normalise_image_name(infobox_img)


# ---------------------------------------------------------------------------
# Main assembler
# ---------------------------------------------------------------------------

def run(config: PipelineConfig) -> None:
    """
    Offline assembly: load all stage outputs and produce the final dataset.
    """
    # --- Load stage outputs ---
    entity_links_path = config.stage_path("entity_links.jsonl")
    entity_data_path = config.stage_path("entity_data.json")
    entity_types_path = config.stage_path("entity_types.json")
    image_lists_path = config.stage_path("image_lists.json")
    image_data_path = config.stage_path("image_data.json")

    for p in [entity_links_path, entity_data_path, entity_types_path,
              image_lists_path, image_data_path]:
        if not p.exists():
            raise FileNotFoundError(f"Required stage output missing: {p}")

    entity_links: list[dict] = []
    with open(entity_links_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entity_links.append(json.loads(line))

    entity_data: dict[str, dict] = json.loads(
        entity_data_path.read_text(encoding="utf-8")
    )
    entity_types: dict[str, str | None] = json.loads(
        entity_types_path.read_text(encoding="utf-8")
    )
    image_lists: dict[str, list[str]] = json.loads(
        image_lists_path.read_text(encoding="utf-8")
    )
    image_data: dict[str, dict] = json.loads(
        image_data_path.read_text(encoding="utf-8")
    )

    # Load Wikipedia intro paragraphs if S4 has been run (optional).
    entity_intros: dict[str, str] = {}
    intros_path = config.stage_path("entity_intros.json")
    if intros_path.exists():
        try:
            entity_intros = json.loads(intros_path.read_text(encoding="utf-8"))
            logger.info("S7: loaded %d Wikipedia intros from S4", len(entity_intros))
        except json.JSONDecodeError:
            pass

    # Load Wikipedia infobox images if S4 has been run (optional).
    entity_infobox_images: dict[str, str | None] = {}
    infobox_images_path = config.stage_path("entity_infobox_images.json")
    if infobox_images_path.exists():
        try:
            entity_infobox_images = json.loads(infobox_images_path.read_text(encoding="utf-8"))
            logger.info("S7: loaded %d Wikipedia infobox images from S4", len(entity_infobox_images))
        except json.JSONDecodeError:
            pass

    logger.info(
        "S7: assembling — %d mentions, %d entities, %d images",
        len(entity_links),
        len(entity_data),
        len(image_data),
    )

    # --- Build set of "special page" QIDs to strip from used_by ---
    # funamespace=0 already excludes talk/template/category pages, but within
    # namespace 0 there are still disambiguation pages and name-list pages.
    # We identify them two ways:
    #   1. QIDs explicitly typed None by S4 (disambiguation pages fail the
    #      P31/P279*/Q4167410 FILTER NOT EXISTS check and come back None).
    #   2. QIDs in entity_data whose Wikipedia URL contains "(disambiguation)".
    _DISAM_RE = re.compile(r"\(disambiguation\)", re.IGNORECASE)
    special_qids: set[str] = set()
    for qid, t in entity_types.items():
        if t is None:
            special_qids.add(qid)
    for qid, d in entity_data.items():
        if _DISAM_RE.search(d.get("url_wikipedia", "")):
            special_qids.add(qid)
    logger.info("S7: %d special-page QIDs will be excluded from used_by", len(special_qids))

    # --- Build Entity KB ---
    kb: dict[str, Entity] = {}
    for qid, data in tqdm(entity_data.items(), desc="S7 build KB"):
        infobox_img = entity_infobox_images.get(qid)
        filenames = image_lists.get(qid, [])

        page_imglist: list[Image] = []
        for fname in filenames:
            img_info = image_data.get(fname)
            if not img_info:
                continue
            raw_used_by: list[str] = img_info.get("used_by", [])
            # Strip disambiguation pages and other special pages from used_by so they
            # don't falsely count as "external entity" usages in the anchor criterion.
            used_by = [q for q in raw_used_by if q not in special_qids]
            img = Image(
                url=img_info["url"],
                used_by=used_by,
                n_used_by=len(used_by),
                is_infobox=_is_infobox(fname, infobox_img),
                width=img_info.get("width"),
                height=img_info.get("height"),
                mime=img_info.get("mime", ""),
                license=img_info.get("license", ""),
            )
            page_imglist.append(img)

        entity = Entity(
            qid=qid,
            name=data.get("name") or qid,
            desc=data.get("desc", ""),
            intro=entity_intros.get(qid, ""),
            type=entity_types.get(qid) or "OTHER",
            infobox_img=infobox_img,
            url_wikipedia=data.get("url_wikipedia", ""),
            page_imglist=page_imglist,
        )
        kb[qid] = entity

    logger.info("S7: built KB with %d entities", len(kb))

    # --- Build MentionEntry list ---
    entries: list[MentionEntry] = []
    skipped_no_entity = 0

    for link_entry in tqdm(entity_links, desc="S7 build mentions"):
        mention = link_entry["mention"]
        qids = link_entry["qids"]

        # Deduplicate QIDs while preserving order (a QID may appear twice in raw links).
        seen: set[str] = set()
        unique_qids: list[str] = []
        for q in qids:
            if q in kb and q not in seen:
                seen.add(q)
                unique_qids.append(q)
        ambiguities = [kb[q] for q in unique_qids]
        if len(ambiguities) < 2:
            skipped_no_entity += 1
            continue

        n_visual, n_grounded = _compute_visual_counts(ambiguities)

        entry = MentionEntry(
            mention=mention,
            categories=link_entry.get("categories", []),
            ambiguities=ambiguities,
            n_entities=len(ambiguities),
            n_visual_ambiguities=n_visual,
            n_grounded_entities=n_grounded,
        )
        entries.append(entry)

    logger.info(
        "S7: %d mention entries built, %d skipped (< 2 entities in KB)",
        len(entries),
        skipped_no_entity,
    )

    # --- Write outputs ---
    config.output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset(entries=entries, kb=kb)
    dataset.save(config.output_path("dataset.json"))
    dataset.save_kb(config.output_path("entity_kb.json"))

    manifest = {
        "wikiambig_version": __version__,
        "assembled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": (
            "Wikipedia and Wikidata are live, continuously-edited sources with "
            "no fixed dump version — assembled_at is the only record of which "
            "snapshot this dataset reflects. Re-running the pipeline today will "
            "yield a different (typically larger) result."
        ),
        "config": json.loads(config.model_dump_json()),
        "counts": {
            "n_mentions": len(entries),
            "n_kb_entities": len(kb),
            "n_images": len({img.url for e in kb.values() for img in e.page_imglist}),
        },
    }
    atomic_write(config.output_path("manifest.json"), manifest)

    logger.info(
        "S7 done: dataset.json (%d mentions), entity_kb.json (%d entities)",
        len(entries),
        len(kb),
    )
