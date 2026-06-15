#!/usr/bin/env python3
"""
Build a MEL split (filtered instances + KB) from the raw scrape output.

Produces two files in the output directory:
  kb.jsonl        — filtered entity KB, one entity per line
  instances.jsonl — MEL instances, candidates as QID lists (not embedded objects)

Task definition:
  Given a mention (ambiguous surface form) and a body image (a photograph that
  appears in multiple Wikipedia articles), identify the correct entity in the KB.

Filtering thresholds live in configs/split_gen/.

Usage:
    python split_gen/make_split.py output/raw_dataset/dataset.jsonl output/split_10_text/
    python split_gen/make_split.py output/raw_dataset/dataset.jsonl output/split_10_text/ --config configs/split_gen/default.yaml
"""

from __future__ import annotations

import argparse
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml
from tqdm import tqdm

try:
    import orjson

    def _loads(data: bytes) -> dict:
        return orjson.loads(data)

    def _dumps(obj: dict) -> bytes:
        return orjson.dumps(obj)

except ImportError:
    import json

    def _loads(data: bytes) -> dict:
        return json.loads(data)

    def _dumps(obj: dict) -> bytes:
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")


_DEFAULT_CONFIG = Path(__file__).resolve().parent.parent / "configs" / "split_gen" / "default.yaml"

_KB_FIELDS  = ("qid", "name", "type", "desc", "intro", "infobox_img", "url_wikipedia")
_IMG_FIELDS = ("url", "n_used_by", "width", "height", "mime", "license")


@dataclass(frozen=True)
class Config:
    entity_types:          frozenset
    require_intro:         bool
    require_image:         bool
    image_mime:            str
    image_min_dim:         int
    image_min_used_by:     int
    image_max_used_by:     int
    mention_min_len:       int
    min_visual_candidates: int
    max_text_candidates:   int
    banwords:              set[str]

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls(
            entity_types          = frozenset(raw["entity"]["types"]),
            require_intro         = raw["entity"].get("intro", False),
            require_image         = raw["entity"].get("image", False),
            image_mime            = raw["image"]["mime"],
            image_min_dim         = raw["image"]["min_dim"],
            image_min_used_by     = raw["image"]["min_used_by"],
            image_max_used_by     = raw["image"]["max_used_by"],
            mention_min_len       = raw["mention"]["min_len"],
            min_visual_candidates = raw["candidates"]["min_visual"],
            max_text_candidates   = raw["candidates"]["max_text"],
            banwords              = set(raw["mention"].get("banwords", [])),
        )


# ============================================================
# FILTERS
# ============================================================

def keep_entity(e: dict, cfg: Config) -> bool:
    """Entity qualifies for the KB."""
    has_intro = bool(e.get("intro"))
    has_image = bool(e.get("infobox_img"))
    if not (has_intro or has_image):
        return False
    if cfg.require_intro and not has_intro:
        return False
    if cfg.require_image and not has_image:
        return False
    return (e.get("type") or "OTHER") in cfg.entity_types


def keep_image(img: dict, n_in_pool: int, cfg: Config) -> bool:
    """Image qualifies as a mention image."""
    return (
        img.get("mime") == cfg.image_mime
        and (img.get("width") or 0) >= cfg.image_min_dim
        and (img.get("height") or 0) >= cfg.image_min_dim
        and cfg.image_min_used_by <= (img.get("n_used_by") or 0) <= cfg.image_max_used_by
        and n_in_pool >= 2
    )


def keep_mention(mention: str, cfg: Config) -> bool:
    """Mention qualifies as a genuine ambiguous surface form."""
    if any(b in mention.lower() for b in cfg.banwords):
        return False
    if len(mention) < cfg.mention_min_len:
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

def _iter_jsonl(path: Path, desc: str) -> Iterator[dict]:
    """Stream JSONL records, with a byte-accurate progress bar."""
    with tqdm(total=path.stat().st_size, desc=desc,
              unit="B", unit_scale=True, unit_divisor=1024) as bar:
        with path.open("rb") as f:
            for line in f:
                bar.update(len(line))
                if not line.isspace():
                    yield _loads(line)


def build(input_path: Path, output_dir: Path, cfg: Config) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    kb_path        = output_dir / "kb.jsonl"
    instances_path = output_dir / "instances.jsonl"

    # ── Single pass: build KB, image index, and a compact mention cache ──────
    # kb_entities   : QID → output fields (no page_imglist)
    # image_pool    : image URL → set of QIDs from KB entities that use this image
    # image_meta    : image URL → output fields
    # mention_cache : (mention, [(qid, [image URLs])]) for instance generation,
    #                 so the 25 GB input is parsed only once.
    kb_entities:   dict[str, dict] = {}
    image_pool:    dict[str, set[str]] = {}
    image_meta:    dict[str, dict] = {}
    mention_cache: list[tuple[str, list[tuple[str, list[str]]]]] = []

    for m in _iter_jsonl(input_path, desc="Scanning"):
        mention = m["mention"]
        if not keep_mention(mention, cfg):
            continue
        pool = [e for e in m["ambiguities"] if keep_entity(e, cfg)]
        if not (2 <= len(pool) <= cfg.max_text_candidates):
            continue

        entity_urls: list[tuple[str, list[str]]] = []
        for e in pool:
            qid = e["qid"]
            if qid not in kb_entities:
                kb_entities[qid] = {k: e.get(k) for k in _KB_FIELDS}
            urls: list[str] = []
            for img in e.get("page_imglist") or ():
                if img.get("is_infobox", False):
                    continue
                url = img.get("url")
                if not url:
                    continue
                urls.append(url)
                image_pool.setdefault(url, set()).add(qid)
                if url not in image_meta:
                    image_meta[url] = {k: img.get(k) for k in _IMG_FIELDS}
            entity_urls.append((qid, urls))
        mention_cache.append((mention, entity_urls))

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
        if not keep_image(img, len(image_pool[url]), cfg):
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
    with kb_path.open("wb") as f:
        for entity in kb_entities.values():
            f.write(_dumps(entity) + b"\n")
    print(f"KB written:         {len(kb_entities):>10,}  → {kb_path}")

    # ── Generate instances from the cache ─────────────────────────────────────
    seen_pair:   set[tuple[str, str]] = set()   # (url, mention)
    seen_answer: set[tuple[str, str]] = set()   # (mention, answer_qid)
    n_written          = 0
    n_rej_visual_pool  = 0
    n_rej_intersection = 0
    n_rej_answer_dedup = 0

    with instances_path.open("wb") as out:
        for mention, entity_urls in tqdm(mention_cache, desc="Instances"):
            pool_qids = {qid for qid, _ in entity_urls}
            text_candidates = sorted(pool_qids)

            for _, urls in entity_urls:
                for url in urls:
                    if url not in qualifying:
                        continue

                    pair_key = (url, mention)
                    if pair_key in seen_pair:
                        continue
                    seen_pair.add(pair_key)

                    # Visual ambiguity must be symmetric to text ambiguity: the image
                    # has to be shared by >= 2 KB entities, otherwise it identifies the
                    # answer trivially and the instance only tests text disambiguation.
                    visual_qids = image_pool[url]
                    if len(visual_qids) < cfg.min_visual_candidates:
                        n_rej_visual_pool += 1
                        continue

                    # The answer is the unique entity that is both a text candidate
                    # AND uses this image — combining both modalities is required.
                    intersection = pool_qids & visual_qids
                    if len(intersection) != 1:
                        n_rej_intersection += 1
                        continue

                    answer_qid = next(iter(intersection))
                    answer_key = (mention, answer_qid)
                    if answer_key in seen_answer:
                        n_rej_answer_dedup += 1
                        continue
                    seen_answer.add(answer_key)

                    out.write(_dumps({
                        "mention":           mention,
                        "image":             image_meta[url],
                        "answer":            answer_qid,
                        "text_candidates":   text_candidates,
                        "visual_candidates": sorted(visual_qids),
                    }) + b"\n")
                    n_written += 1

    total = n_written + n_rej_visual_pool + n_rej_intersection + n_rej_answer_dedup
    print(f"\nInstance generation cascade:")
    print(f"  (mention, image) candidates:      {total:>10,}")
    print(f"  – visual pool < {cfg.min_visual_candidates}:                {n_rej_visual_pool:>10,}  ({100*n_rej_visual_pool/max(total,1):.1f}%)")
    print(f"  – intersection ≠ 1:               {n_rej_intersection:>10,}  ({100*n_rej_intersection/max(total,1):.1f}%)")
    print(f"  – (mention, answer) duplicate:    {n_rej_answer_dedup:>10,}  ({100*n_rej_answer_dedup/max(total,1):.1f}%)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Instances written:                {n_written:>10,}  → {instances_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input",      type=Path, help="dataset.jsonl (scrape output)")
    ap.add_argument("output_dir", type=Path, help="split output directory (e.g. output/split_10_text/)")
    ap.add_argument("--config",   type=Path, default=_DEFAULT_CONFIG,
                    help=f"filter thresholds YAML (default: {_DEFAULT_CONFIG})")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")
    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")

    build(args.input, args.output_dir, Config.load(args.config))


if __name__ == "__main__":
    main()
