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
    python split_gen/make_split.py output/raw_dataset/dataset.jsonl output/split_10_text/ --workers 8
"""

from __future__ import annotations

import argparse
import os
import re
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import yaml
from tqdm import tqdm

from p_scrapping import fetch_related_qids, prefetch_qids, get_stats as get_sparql_stats

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
_RE_STARTS_NUM = re.compile(r"^\s*\d")


@dataclass(frozen=True)
class Config:
    entity_types:              frozenset
    require_intro:             bool
    require_image:             bool
    image_mime:                str
    image_min_dim:             int
    image_min_used_by:         int
    image_max_used_by:         int
    mention_min_len:           int
    min_text_candidates:       int   # min entities sharing the mention (default 2)
    max_text_candidates:       int   # max entities sharing the mention
    min_visual_candidates:     int   # min KB entities sharing the body image
    max_visual_candidates:     int   # max KB entities sharing the body image (0 = unlimited)
    require_answer_type_shared: bool  # answer's type must appear ≥1× among other text candidates
    require_unique_candidate_infoboxes: bool  # no two text candidates may share the same infobox portrait
    max_instances_per_answer:  int            # cap instances per answer entity (0 = unlimited)
    max_instances_per_image:   int            # cap instances per body image URL (0 = unlimited)
    forbidden_properties:      list[str]
    banwords:                  set[str]
    category_include:          frozenset
    category_exclude:          frozenset
    drop_if_start_with_num:    bool

    @classmethod
    def load(cls, path: Path) -> "Config":
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        categories = raw.get("categories", {})
        cands = raw["candidates"]
        return cls(
            entity_types               = frozenset(raw["entity"]["types"]),
            require_intro              = raw["entity"].get("intro", False),
            require_image              = raw["entity"].get("image", False),
            image_mime                 = raw["image"]["mime"],
            image_min_dim              = raw["image"]["min_dim"],
            image_min_used_by          = raw["image"]["min_used_by"],
            image_max_used_by          = raw["image"]["max_used_by"],
            mention_min_len            = raw["mention"]["min_len"],
            min_text_candidates        = cands.get("min_text", 2),
            max_text_candidates        = cands["max_text"],
            min_visual_candidates      = cands["min_visual"],
            max_visual_candidates      = cands.get("max_visual", 0),
            require_answer_type_shared      = cands.get("require_answer_type_shared", False),
            require_unique_candidate_infoboxes = cands.get("require_unique_candidate_infoboxes", False),
            max_instances_per_answer        = cands.get("max_instances_per_answer", 0),
            max_instances_per_image         = cands.get("max_instances_per_image", 0),
            forbidden_properties           = raw.get("candidates", {}).get("forbidden_properties", []),
            banwords                        = set(raw["mention"].get("banwords", [])),
            category_include           = frozenset(categories.get("include", [])),
            category_exclude           = frozenset(categories.get("exclude", [])),
            drop_if_start_with_num     = raw["mention"].get("drop_if_start_with_num", False),
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
    if cfg.drop_if_start_with_num and _RE_STARTS_NUM.match(mention):
        return False
    return any(c.isalpha() for c in mention)


def keep_categories(categories: list[str], cfg: Config) -> bool:
    """Disambiguation page's categories pass the include/exclude filters."""
    cats = set(categories)
    if cfg.category_include and not (cats & cfg.category_include):
        return False
    if cfg.category_exclude and (cats & cfg.category_exclude):
        return False
    return True


def keep_forbidden_properties(qids: list[str], cfg: Config) -> bool:
    """Return False if any pair of QIDs is linked by a forbidden property. Cache must be warm."""
    if not cfg.forbidden_properties:
        return True
    qid_set = set(qids)
    for qid in qids:
        related = fetch_related_qids(qid, cfg.forbidden_properties)
        matche = related&(qid_set- {qid})
        if related & (qid_set - {qid}):
            print(f"le qid vérifier {qid} en lien avec {matche} (les autres : {qid_set})")
            return False
    return True


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
# PARALLEL SCAN
# ============================================================

def _chunk_file(path: Path, n: int) -> list[tuple[int, int]]:
    """Split path into n byte ranges aligned to newline boundaries."""
    size = path.stat().st_size
    approx = size // n
    offsets: list[tuple[int, int]] = []
    with path.open("rb") as f:
        start = 0
        for _ in range(n - 1):
            f.seek(start + approx)
            f.readline()  # advance to the next newline so records stay whole
            end = f.tell()
            if end >= size:
                break
            offsets.append((start, end))
            start = end
        offsets.append((start, size))
    return offsets


def _scan_chunk(
    path_str: str, byte_start: int, byte_end: int, cfg: Config
) -> tuple[dict, dict, dict, list]:
    """
    Parse and filter one byte range of a JSONL file.
    Returns (kb_entities, image_pool, image_meta, mention_cache).
    Must be a module-level function to be picklable by ProcessPoolExecutor.
    """
    kb_entities:   dict[str, dict]      = {}
    image_pool:    dict[str, set[str]]  = {}
    image_meta:    dict[str, dict]      = {}
    mention_cache: list                  = []

    with open(path_str, "rb", buffering=1 << 23) as f:  # 8 MB read buffer
        f.seek(byte_start)
        while f.tell() < byte_end:
            line = f.readline()
            if not line or line.isspace():
                continue
            try:
                m = _loads(line)
            except Exception:
                continue

            mention = m.get("mention", "")
            if not keep_mention(mention, cfg):
                continue
            if not keep_categories(m.get("categories", []), cfg):
                continue
            pool = [e for e in m.get("ambiguities", []) if keep_entity(e, cfg)]
            if not (cfg.min_text_candidates <= len(pool) <= cfg.max_text_candidates):
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
                    s = image_pool.get(url)
                    if s is None:
                        image_pool[url] = {qid}
                    else:
                        s.add(qid)
                    if url not in image_meta:
                        image_meta[url] = {k: img.get(k) for k in _IMG_FIELDS}
                entity_urls.append((qid, urls))
            mention_cache.append((mention, entity_urls))

    return kb_entities, image_pool, image_meta, mention_cache


def _scan_sequential(input_path: Path, cfg: Config) -> tuple[dict, dict, dict, list]:
    """Single-threaded scan with a byte-accurate progress bar."""
    kb_entities:   dict[str, dict]      = {}
    image_pool:    dict[str, set[str]]  = {}
    image_meta:    dict[str, dict]      = {}
    mention_cache: list                  = []

    with tqdm(total=input_path.stat().st_size, desc="Scanning",
              unit="B", unit_scale=True, unit_divisor=1024) as bar:
        with input_path.open("rb") as f:
            for line in f:
                bar.update(len(line))
                if line.isspace():
                    continue
                m = _loads(line)
                mention = m["mention"]
                if not keep_mention(mention, cfg):
                    continue
                if not keep_categories(m.get("categories", []), cfg):
                    continue
                pool = [e for e in m["ambiguities"] if keep_entity(e, cfg)]
                if not (cfg.min_text_candidates <= len(pool) <= cfg.max_text_candidates):
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

    return kb_entities, image_pool, image_meta, mention_cache


def _scan_parallel(input_path: Path, cfg: Config, n_workers: int) -> tuple[dict, dict, dict, list]:
    """Parallel scan: split file into n_workers chunks, parse each in a subprocess."""
    offsets = _chunk_file(input_path, n_workers)
    print(f"Scanning {input_path.stat().st_size / 1e9:.1f} GB with {len(offsets)} workers…")

    kb_entities:   dict[str, dict]      = {}
    image_pool:    dict[str, set[str]]  = {}
    image_meta:    dict[str, dict]      = {}
    mention_cache: list                  = []

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futs = [
            ex.submit(_scan_chunk, str(input_path), s, e, cfg)
            for s, e in offsets
        ]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Chunks done"):
            pk, pp, pm, pc = fut.result()
            # merge kb_entities: same QID → same data across chunks
            kb_entities.update(pk)
            # merge image_pool: union QID sets per URL
            for url, qids in pp.items():
                s = image_pool.get(url)
                if s is None:
                    image_pool[url] = set(qids)
                else:
                    s |= qids
            # merge image_meta: same URL → same metadata
            image_meta.update(pm)
            mention_cache.extend(pc)

    return kb_entities, image_pool, image_meta, mention_cache


# ============================================================
# PIPELINE
# ============================================================

def build(input_path: Path, output_dir: Path, cfg: Config, n_workers: int = 1) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    kb_path        = output_dir / "kb.jsonl"
    instances_path = output_dir / "instances.jsonl"

    # ── Scan phase ───────────────────────────────────────────────────────────
    if n_workers > 1:
        kb_entities, image_pool, image_meta, mention_cache = _scan_parallel(
            input_path, cfg, n_workers
        )
    else:
        kb_entities, image_pool, image_meta, mention_cache = _scan_sequential(
            input_path, cfg
        )

    # ── Forbidden-property filter ───────────────────────────────────────────
    if cfg.forbidden_properties and mention_cache:
        all_qids = list({qid for _, eurls in mention_cache for qid, _ in eurls})
        print(f"\nForbidden-property prefetch: {len(all_qids):,} unique QIDs, "
              f"properties: {cfg.forbidden_properties}")
        prefetch_qids(all_qids, cfg.forbidden_properties, max_workers=30)
        stats = get_sparql_stats()
        print(f"  SPARQL requests:      {stats['requests']:>10,}")
        print(f"  Errors:               {stats['errors']:>10,}")

        before = len(mention_cache)
        mention_cache = [
            (mention, eurls) for mention, eurls in mention_cache
            if keep_forbidden_properties([qid for qid, _ in eurls], cfg)
        ]
        after = len(mention_cache)
        print(f"  Mentions checked:     {before:>10,}")
        print(f"  Mentions dropped:     {before - after:>10,}")
        print(f"  Mentions remaining:   {after:>10,}")

    # Images that are themselves another KB entity's reference photo (P18 infobox)
    # are excluded: a query image that IS some entity's canonical portrait lets a
    # model "recognise" that entity directly, bypassing the cooccurrence signal
    # the task is meant to test.
    infobox_filenames = {
        _normalise_image_name(e["infobox_img"])
        for e in kb_entities.values()
        if e.get("infobox_img")
    }

    # ── Filter images ────────────────────────────────────────────────────────
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

    # ── Write kb.jsonl ───────────────────────────────────────────────────────
    with kb_path.open("wb") as f:
        for entity in kb_entities.values():
            f.write(_dumps(entity) + b"\n")
    print(f"KB written:         {len(kb_entities):>10,}  → {kb_path}")

    # ── Generate instances from the cache ────────────────────────────────────
    seen_pair:    set[tuple[str, str]] = set()   # (url, mention)
    seen_answer:  set[tuple[str, str]] = set()   # (mention, answer_qid)
    answer_counts: dict[str, int]      = {}      # answer_qid → instance count
    image_counts:  dict[str, int]      = {}      # body image url → instance count
    n_written             = 0
    n_rej_visual_min      = 0
    n_rej_visual_max      = 0
    n_rej_intersection    = 0
    n_rej_type_not_shared = 0
    n_rej_shared_infobox  = 0
    n_rej_answer_dedup    = 0
    n_rej_answer_cap      = 0
    n_rej_image_cap       = 0

    with instances_path.open("wb") as out:
        for mention, entity_urls in tqdm(mention_cache, desc="Instances"):
            pool_qids = {qid for qid, _ in entity_urls}

            # If any two text candidates share the same infobox portrait, the KB
            # is visually degenerate for this mention: two entities would look
            # identical, making visual disambiguation impossible.
            if cfg.require_unique_candidate_infoboxes:
                seen_imgs: set[str] = set()
                collision = False
                for qid in pool_qids:
                    img = (kb_entities.get(qid) or {}).get("infobox_img")
                    if img:
                        norm = _normalise_image_name(img)
                        if norm in seen_imgs:
                            collision = True
                            break
                        seen_imgs.add(norm)
                if collision:
                    n_rej_shared_infobox += 1
                    continue

            text_candidates = sorted(pool_qids)

            for _, urls in entity_urls:
                for url in urls:
                    if url not in qualifying:
                        continue

                    pair_key = (url, mention)
                    if pair_key in seen_pair:
                        continue
                    seen_pair.add(pair_key)

                    # Image must be shared by enough KB entities that vision alone
                    # cannot trivially identify the answer.
                    visual_qids = image_pool[url]
                    n_vis = len(visual_qids)
                    if n_vis < cfg.min_visual_candidates:
                        n_rej_visual_min += 1
                        continue

                    # Cap the visual pool: images shared by too many entities are
                    # generic (maps, flags, stock photos) and carry no identity signal.
                    if cfg.max_visual_candidates > 0 and n_vis > cfg.max_visual_candidates:
                        n_rej_visual_max += 1
                        continue

                    # The answer is the unique entity that is both a text candidate
                    # AND uses this image — combining both modalities is required.
                    intersection = pool_qids & visual_qids
                    if len(intersection) != 1:
                        n_rej_intersection += 1
                        continue

                    answer_qid = next(iter(intersection))

                    # The answer entity's semantic type must not be unique among the
                    # text candidates: if it were, a model could resolve the instance
                    # using type information alone (e.g., the only LOC in a list of
                    # PERS entities), bypassing the need for cross-modal reasoning.
                    if cfg.require_answer_type_shared:
                        answer_type = (kb_entities.get(answer_qid) or {}).get("type") or "OTHER"
                        type_shared = any(
                            ((kb_entities.get(q) or {}).get("type") or "OTHER") == answer_type
                            for q in pool_qids if q != answer_qid
                        )
                        if not type_shared:
                            n_rej_type_not_shared += 1
                            continue

                    answer_key = (mention, answer_qid)
                    if answer_key in seen_answer:
                        n_rej_answer_dedup += 1
                        continue
                    seen_answer.add(answer_key)

                    # Cap: prevent a single answer entity from dominating the
                    # dataset (the model would over-learn its visual signature).
                    if cfg.max_instances_per_answer > 0:
                        if answer_counts.get(answer_qid, 0) >= cfg.max_instances_per_answer:
                            n_rej_answer_cap += 1
                            continue

                    # Cap: prevent a single body image from appearing in too many
                    # instances (the model would memorise image→entity clusters).
                    if cfg.max_instances_per_image > 0:
                        if image_counts.get(url, 0) >= cfg.max_instances_per_image:
                            n_rej_image_cap += 1
                            continue

                    out.write(_dumps({
                        "mention":           mention,
                        "image":             image_meta[url],
                        "answer":            answer_qid,
                        "text_candidates":   text_candidates,
                        "visual_candidates": sorted(visual_qids),
                    }) + b"\n")
                    n_written += 1
                    if cfg.max_instances_per_answer > 0:
                        answer_counts[answer_qid] = answer_counts.get(answer_qid, 0) + 1
                    if cfg.max_instances_per_image > 0:
                        image_counts[url] = image_counts.get(url, 0) + 1

    # n_rej_shared_infobox is counted per MENTION (not per url-mention pair),
    # so we exclude it from the url-level cascade total.
    total = (n_written + n_rej_visual_min + n_rej_visual_max + n_rej_intersection
             + n_rej_type_not_shared + n_rej_answer_dedup
             + n_rej_answer_cap + n_rej_image_cap)
    w = max(total, 1)
    print(f"\nInstance generation cascade:")
    print(f"  (mention, image) candidates:      {total:>10,}")
    if cfg.require_unique_candidate_infoboxes:
        print(f"  – shared infobox (mentions):      {n_rej_shared_infobox:>10,}  (mention-level)")
    print(f"  – visual pool < {cfg.min_visual_candidates}:                {n_rej_visual_min:>10,}  ({100*n_rej_visual_min/w:.1f}%)")
    if cfg.max_visual_candidates > 0:
        print(f"  – visual pool > {cfg.max_visual_candidates}:               {n_rej_visual_max:>10,}  ({100*n_rej_visual_max/w:.1f}%)")
    print(f"  – intersection ≠ 1:               {n_rej_intersection:>10,}  ({100*n_rej_intersection/w:.1f}%)")
    if cfg.require_answer_type_shared:
        print(f"  – answer type unique in text:     {n_rej_type_not_shared:>10,}  ({100*n_rej_type_not_shared/w:.1f}%)")
    print(f"  – (mention, answer) duplicate:    {n_rej_answer_dedup:>10,}  ({100*n_rej_answer_dedup/w:.1f}%)")
    if cfg.max_instances_per_answer > 0:
        print(f"  – answer cap (>{cfg.max_instances_per_answer}/entity):       {n_rej_answer_cap:>10,}  ({100*n_rej_answer_cap/w:.1f}%)")
    if cfg.max_instances_per_image > 0:
        print(f"  – image cap (>{cfg.max_instances_per_image}/image):         {n_rej_image_cap:>10,}  ({100*n_rej_image_cap/w:.1f}%)")
    print(f"  ─────────────────────────────────────────────────────")
    print(f"  Instances written:                {n_written:>10,}  → {instances_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input",      type=Path, help="dataset.jsonl (scrape output)")
    ap.add_argument("output_dir", type=Path, help="split output directory (e.g. output/split_10_text/)")
    ap.add_argument("--config",   type=Path, default=_DEFAULT_CONFIG,
                    help=f"filter thresholds YAML (default: {_DEFAULT_CONFIG})")
    ap.add_argument("--workers",  type=int, default=None,
                    help="parallel scan workers (default: cpu_count; use 1 for sequential with progress bar)")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")
    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")

    n_workers = args.workers if args.workers is not None else (os.cpu_count() or 4)
    build(args.input, args.output_dir, Config.load(args.config), n_workers=n_workers)


if __name__ == "__main__":
    main()
