#!/usr/bin/env python3
"""
Build the final MEL dataset from the raw pipeline output.

Produces two files in output/final/:
  kb.jsonl        — filtered entity KB, one entity per line
  instances.jsonl — MEL instances, candidates as QID lists (not embedded objects)

Task definition:
  Given a mention (ambiguous surface form) and a body image (a photograph that
  appears in multiple Wikipedia articles), identify the correct entity in the KB.

Usage:
    python scripts/make_dataset.py output/dataset.jsonl output/final/
"""

from __future__ import annotations

import argparse
import json
import urllib.parse
from pathlib import Path

from tqdm import tqdm

_MIN_DIM       = 250   # minimum image dimension (px)
_MAX_N_USED_BY = 10   # maximum Wikipedia image reuse
_MIN_N_USED_BY = 2    # minimum Wikipedia image reuse
_KB_FIELDS     = ("qid", "name", "type", "desc", "intro", "infobox_img", "url_wikipedia")
_IMG_FIELDS    = ("url", "n_used_by", "width", "height", "mime", "license")


def keep_entity(e: dict) -> bool:
    """Entity qualifies for the KB."""
    if not (e.get("intro") or e.get("infobox_img")):
        return False

    return (e.get("type") or "OTHER") in {"PERS", "ORG", "LOC"}


def keep_image(img: dict, n_in_pool: int) -> bool:
    """Image qualifies as a mention image."""
    return (
        img.get("mime") == "image/jpeg"
        and (img.get("width") or 0) >= _MIN_DIM
        and (img.get("height") or 0) >= _MIN_DIM
        and  (img.get("n_used_by") or 0) <= _MAX_N_USED_BY
    and  (img.get("n_used_by") or 0) >= _MIN_N_USED_BY
        and n_in_pool >= 2
    )


_MIN_VISUAL_CANDIDATES = 2  # min KB entities sharing an image (visual ambiguity)
_MAX_TEXT_CANDIDATES = 50   # max KB entities sharing a mention (drops generic-fragment mentions, e.g. "Cerro")


def keep_mention(mention: str) -> bool:
    """Mention qualifies as a genuine ambiguous surface form."""
    if len(mention) <= 2:
        return False
    return any(c.isalpha() for c in mention)


def _normalise_image_name(url_or_filename: str) -> str:
    """Extract and normalise a filename from a Wikimedia URL or 'File:...' string."""
    s = url_or_filename
    if s.startswith("File:"):
        s = s[5:]
    elif "Special:FilePath/" in s:
        s = s.split("Special:FilePath/", 1)[-1]
    elif "/wikipedia/commons/" in s:
        s = s.rstrip("/").split("/")[-1]
    s = urllib.parse.unquote(s)
    return s.replace("_", " ").strip().lower()


# ============================================================
# PIPELINE
# ============================================================

def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def build(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    kb_path        = output_dir / "kb.jsonl"
    instances_path = output_dir / "instances.jsonl"

    # ── Pass 1: build KB and image index ──────────────────────────────────────
    # kb_entities : QID → output fields (no page_imglist)
    # image_pool  : image URL → set of QIDs from KB entities that use this image
    # image_meta  : image URL → output fields
    kb_entities: dict[str, dict] = {}
    image_pool:  dict[str, set[str]] = {}
    image_meta:  dict[str, dict] = {}

    for m in tqdm(_iter_jsonl(input_path), desc="Pass 1/2"):
        if not keep_mention(m["mention"]):
            continue
        pool = [e for e in m["ambiguities"] if keep_entity(e)]
        if len(pool) < 2 or len(pool) > _MAX_TEXT_CANDIDATES:
            continue

        for e in pool:
            qid = e["qid"]
            if qid not in kb_entities:
                kb_entities[qid] = {k: e.get(k) for k in _KB_FIELDS}
            for img in e.get("page_imglist", []):
                if img.get("is_infobox", False):
                    continue
                url = img.get("url")
                if not url:
                    continue
                image_pool.setdefault(url, set()).add(qid)
                if url not in image_meta:
                    image_meta[url] = {k: img.get(k) for k in _IMG_FIELDS}

    # Images that are themselves another KB entity's reference photo (P18 infobox)
    # are excluded: a query image that IS some entity's canonical portrait lets a
    # model "recognise" that entity directly, bypassing the cooccurrence signal
    # the task is meant to test.
    infobox_filenames = {
        _normalise_image_name(e["infobox_img"])
        for e in kb_entities.values()
        if e.get("infobox_img")
    }

    # Filter images
    n_images_total = len(image_meta)
    n_infobox_excluded = 0
    qualifying: set[str] = set()
    for url, img in image_meta.items():
        if not keep_image(img, len(image_pool[url])):
            continue
        if _normalise_image_name(url) in infobox_filenames:
            n_infobox_excluded += 1
            continue
        qualifying.add(url)

    image_pool = {url: image_pool[url] for url in qualifying}
    image_meta = {url: image_meta[url] for url in qualifying}

    print(f"KB entities:        {len(kb_entities):>10,}")
    print(f"Qualifying images:  {len(qualifying):>10,}  (of {n_images_total:,} total)")
    print(f"  excluded as another entity's infobox: {n_infobox_excluded:,}")

    # ── Write kb.jsonl ────────────────────────────────────────────────────────
    with kb_path.open("w", encoding="utf-8") as f:
        for entity in kb_entities.values():
            f.write(json.dumps(entity, ensure_ascii=False) + "\n")
    print(f"KB written:         {len(kb_entities):>10,}  → {kb_path}")

    # ── Pass 2: generate instances ────────────────────────────────────────────
    seen_pair:   set[tuple[str, str]] = set()   # (url, mention)
    seen_answer: set[tuple[str, str]] = set()   # (mention, answer_qid)
    n_written          = 0
    n_rej_visual_pool  = 0
    n_rej_intersection = 0
    n_rej_answer_dedup = 0

    with instances_path.open("w", encoding="utf-8") as out:
        for m in tqdm(_iter_jsonl(input_path), desc="Pass 2/2"):
            if not keep_mention(m["mention"]):
                continue
            pool = [e for e in m["ambiguities"] if keep_entity(e)]
            if len(pool) < 2 or len(pool) > _MAX_TEXT_CANDIDATES:
                continue

            pool_qids = {e["qid"] for e in pool if e["qid"] in kb_entities}

            for e in pool:
                for img in e.get("page_imglist", []):
                    if img.get("is_infobox", False):
                        continue
                    url = img.get("url")
                    if not url or url not in qualifying:
                        continue

                    pair_key = (url, m["mention"])
                    if pair_key in seen_pair:
                        continue
                    seen_pair.add(pair_key)

                    # Visual ambiguity must be symmetric to text ambiguity: the image
                    # has to be shared by >= 2 KB entities, otherwise it identifies the
                    # answer trivially and the instance only tests text disambiguation.
                    visual_qids = image_pool[url]
                    visual_candidates = sorted(visual_qids & kb_entities.keys())
                    if len(visual_candidates) < _MIN_VISUAL_CANDIDATES:
                        n_rej_visual_pool += 1
                        continue

                    # The answer is the unique entity that is both a text candidate
                    # AND uses this image — combining both modalities is required.
                    intersection = pool_qids & visual_qids
                    if len(intersection) != 1:
                        n_rej_intersection += 1
                        continue

                    answer_qid = next(iter(intersection))
                    answer_key = (m["mention"], answer_qid)
                    if answer_key in seen_answer:
                        n_rej_answer_dedup += 1
                        continue
                    seen_answer.add(answer_key)

                    out.write(json.dumps({
                        "mention":          m["mention"],
                        "image":            image_meta[url],
                        "answer":           answer_qid,
                        "text_candidates":  sorted(pool_qids),
                        "visual_candidates": visual_candidates,
                    }, ensure_ascii=False) + "\n")
                    n_written += 1

    total = n_written + n_rej_visual_pool + n_rej_intersection + n_rej_answer_dedup
    print(f"\nInstance generation cascade:")
    print(f"  (mention, image) candidates:      {total:>10,}")
    print(f"  – visual pool < {_MIN_VISUAL_CANDIDATES}:                {n_rej_visual_pool:>10,}  ({100*n_rej_visual_pool/max(total,1):.1f}%)")
    print(f"  – intersection ≠ 1:               {n_rej_intersection:>10,}  ({100*n_rej_intersection/max(total,1):.1f}%)")
    print(f"  – (mention, answer) duplicate:    {n_rej_answer_dedup:>10,}  ({100*n_rej_answer_dedup/max(total,1):.1f}%)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Instances written:                {n_written:>10,}  → {instances_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input",      type=Path, help="dataset.jsonl (pipeline output)")
    ap.add_argument("output_dir", type=Path, help="output directory (e.g. output/final/)")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")

    build(args.input, args.output_dir)


if __name__ == "__main__":
    main()
