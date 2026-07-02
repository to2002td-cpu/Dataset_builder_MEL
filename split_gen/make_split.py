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
    python split_gen/make_split.py output/raw_dataset/dataset.jsonl output/split_10_text/ \\
        --config configs/split_gen/default.yaml
    python split_gen/make_split.py output/raw_dataset/dataset.jsonl output/split_10_text/ \\
        --workers 8
"""

from __future__ import annotations

import argparse
import os
import re
import urllib.parse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm import tqdm
from wikidata_enrich import (
    fetch_instanceof_matches,
    fetch_related_qids,
    prefetch_qids,
)
from wikidata_enrich import (
    get_stats as get_sparql_stats,
)

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
    # drop KB entities that are P31/P279* of any of these classes (e.g. states)
    entity_exclude_instance_of: list[str]
    require_intro:             bool
    require_image:             bool
    image_mime:                str
    image_min_dim:             int
    image_min_used_by:         int
    image_max_used_by:         int
    mention_min_len:           int
    mention_min_used_by:       int   # min entities sharing the mention (text candidate pool size)
    mention_max_used_by:       int   # max entities sharing the mention (0 = unlimited)
    require_answer_type_shared: bool  # answer's type must appear ≥1× among other text candidates
    # no two text candidates may share the same infobox portrait
    require_unique_candidate_infoboxes: bool
    visual_min:                int    # min KB entities sharing the query image
    visual_max:                int    # max KB entities sharing the query image (0 = unlimited)
    # drop mention if any candidate pair is linked by these Wikidata P-IDs
    forbidden_properties:      list[str]
    banwords:                  set[str]
    category_include:          frozenset
    category_exclude:          frozenset
    drop_if_start_with_num:    bool

    @classmethod
    def load(cls, path: Path) -> Config:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        categories = raw.get("categories", {})
        cands = raw["candidates"]
        return cls(
            entity_types               = frozenset(raw["entity"]["types"]),
            entity_exclude_instance_of = list(raw["entity"].get("exclude_instance_of", [])),
            require_intro              = raw["entity"].get("intro", False),
            require_image              = raw["entity"].get("image", False),
            image_mime                 = raw["image"]["mime"],
            image_min_dim              = raw["image"]["min_dim"],
            image_min_used_by          = raw["image"]["min_used_by"],
            image_max_used_by          = raw["image"]["max_used_by"],
            mention_min_len            = raw["mention"]["min_len"],
            mention_min_used_by        = raw["mention"].get("min_used_by", 2),
            mention_max_used_by        = raw["mention"].get("max_used_by", 0),
            require_answer_type_shared      = cands.get("require_answer_type_shared", False),
            require_unique_candidate_infoboxes = cands.get(
                "require_unique_candidate_infoboxes", False),
            visual_min                      = cands.get("min_visual", 2),
            visual_max                      = cands.get("max_visual", 0),
            forbidden_properties            = cands.get("forbidden_properties", []),
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
    name = e.get("name") or ""
    if re.fullmatch(r"Q\d+", name):
        return False
    has_intro = bool(e.get("intro"))
    has_image = bool(e.get("infobox_img"))
    if not (has_intro or has_image):
        return False
    if cfg.require_intro and not has_intro:
        return False
    if cfg.require_image and not has_image:
        return False
    return (e.get("type") or "OTHER") in cfg.entity_types


def keep_image(img: dict, cfg: Config) -> bool:
    """Image qualifies as a mention image (quality only — no candidate-count gate)."""
    return (
        img.get("mime") == cfg.image_mime
        and (img.get("width") or 0) >= cfg.image_min_dim
        and (img.get("height") or 0) >= cfg.image_min_dim
        and cfg.image_min_used_by <= (img.get("n_used_by") or 0) <= cfg.image_max_used_by
    )


def keep_mention(mention: str, cfg: Config) -> bool:
    """Mention qualifies as a genuine ambiguous surface form."""
    if any(re.search(rf"\b{re.escape(b)}\b", mention, re.IGNORECASE) for b in cfg.banwords):
        return False
    if len(mention) < cfg.mention_min_len:
        return False
    if cfg.drop_if_start_with_num and _RE_STARTS_NUM.match(mention):
        return False
    return any(c.isalpha() for c in mention)


def keep_candidate_count(n: int, cfg: Config) -> bool:
    """Mention kept only if its number of text candidates is within bounds."""
    if n < cfg.mention_min_used_by:
        return False
    return not (cfg.mention_max_used_by > 0 and n > cfg.mention_max_used_by)


def keep_categories(categories: list[str], cfg: Config) -> bool:
    """Disambiguation page's categories pass the include/exclude filters."""
    cats = set(categories)
    if cfg.category_include and not (cats & cfg.category_include):
        return False
    return not (cfg.category_exclude and (cats & cfg.category_exclude))


def keep_forbidden_properties(qids: list[str], cfg: Config) -> bool:
    """
    Return False if any candidate is linked to another candidate by a forbidden
    Wikidata property (e.g. P131 "located in" — "New York City" → "New York"
    state). Such a pair is nested/irresolvable, so the whole mention is dropped.

    The SPARQL cache must already be warm (see prefetch_qids in build()).
    """
    if not cfg.forbidden_properties:
        return True
    others = set(qids)
    for qid in qids:
        related = fetch_related_qids(qid, cfg.forbidden_properties)
        if related & (others - {qid}):
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


def _has_shared_infobox(qids, kb_entities: dict) -> bool:
    """True if any two of the given entities share the same (normalised) infobox
    portrait — i.e. the set is visually degenerate (two entities look identical)."""
    seen: set[str] = set()
    for qid in qids:
        img = (kb_entities.get(qid) or {}).get("infobox_img")
        if img:
            norm = _normalise_image_name(img)
            if norm in seen:
                return True
            seen.add(norm)
    return False


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
            if not keep_candidate_count(len(pool), cfg):
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
              unit="B", unit_scale=True, unit_divisor=1024) as bar, input_path.open("rb") as f:
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
            if not keep_candidate_count(len(pool), cfg):
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

    # ── Exclude-by-instance-of filter ────────────────────────────────────────
    # Drop KB entities that are administrative territorial entities (states,
    # regions, …) — configured as a list of Wikidata classes in
    # entity.exclude_instance_of. One batched SPARQL query flags every matching
    # QID (P31/P279* traversal); flagged entities are removed from the KB and
    # from every mention's candidate pool. A mention whose pool then falls below
    # the candidate-count floor is dropped entirely.
    if cfg.entity_exclude_instance_of and kb_entities:
        kb_qids = list(kb_entities)
        print(f"\nExclude-by-instance-of: {len(kb_qids):,} KB QIDs, "
              f"classes: {cfg.entity_exclude_instance_of}")
        excluded = fetch_instanceof_matches(
            kb_qids, cfg.entity_exclude_instance_of, max_workers=30
        )
        print(f"  Entities flagged:     {len(excluded):>10,}")

        for qid in excluded:
            kb_entities.pop(qid, None)

        # Also strip flagged QIDs from the visual pools, so they can't appear as
        # visual_candidates (dangling refs to a QID no longer in the KB) or
        # inflate the visual-pool count.
        for qids in image_pool.values():
            qids -= excluded

        before_m = len(mention_cache)
        pruned: list = []
        for mention, eurls in mention_cache:
            kept = [(qid, urls) for qid, urls in eurls if qid not in excluded]
            if keep_candidate_count(len(kept), cfg):
                pruned.append((mention, kept))
        mention_cache = pruned
        print(f"  KB entities dropped:  {len(kb_qids) - len(kb_entities):>10,}")
        print(f"  Mentions dropped:     {before_m - len(mention_cache):>10,}  "
              f"(pool fell below floor)")
        print(f"  Mentions remaining:   {len(mention_cache):>10,}")

    # ── Forbidden-property filter ────────────────────────────────────────────
    # Drop a mention if any two of its candidates are linked by a located-in /
    # part-of property: the pair is nested ("New York City" inside "New York"
    # state) and so irresolvable. One batched SPARQL prefetch warms the cache,
    # then the check is a set-intersection per mention.
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
        if not keep_image(img, cfg):
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
    # Every distinct qualifying (image, mention) couple becomes its own instance:
    # the same answer may appear with several body images. seen_pair only guards
    # against emitting the exact same (image, mention) twice.
    seen_pair:    set[tuple[str, str]] = set()   # (url, mention)
    n_written             = 0
    n_rej_visual          = 0
    n_rej_intersection    = 0
    n_rej_type_not_shared = 0
    n_rej_shared_infobox  = 0
    n_rej_shared_infobox_visual = 0

    with instances_path.open("wb") as out:
        for mention, entity_urls in tqdm(mention_cache, desc="Instances"):
            pool_qids = {qid for qid, _ in entity_urls}

            # If any two text candidates share the same infobox portrait, the KB
            # is visually degenerate for this mention: two entities would look
            # identical, making visual disambiguation impossible.
            if (cfg.require_unique_candidate_infoboxes
                    and _has_shared_infobox(pool_qids, kb_entities)):
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

                    # Visual candidate pool = every KB entity using this image.
                    # Require it to be genuinely ambiguous (≥ visual_min) and not
                    # degenerate/over-shared (≤ visual_max, 0 = unlimited).
                    visual_qids = image_pool[url]
                    n_vis = len(visual_qids)
                    if n_vis < cfg.visual_min or (cfg.visual_max > 0 and n_vis > cfg.visual_max):
                        n_rej_visual += 1
                        continue

                    # Same rule as for text candidates, applied to the visual pool:
                    # if two entities sharing this image also share an infobox
                    # portrait, they are visually indistinguishable here.
                    if (cfg.require_unique_candidate_infoboxes
                            and _has_shared_infobox(visual_qids, kb_entities)):
                        n_rej_shared_infobox_visual += 1
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

                    out.write(_dumps({
                        "mention":           mention,
                        "image":             image_meta[url],
                        "answer":            answer_qid,
                        "text_candidates":   text_candidates,
                        "visual_candidates": sorted(visual_qids),
                    }) + b"\n")
                    n_written += 1

    # n_rej_shared_infobox is counted per MENTION (not per url-mention pair),
    # so we exclude it from the url-level cascade total.
    total = (n_written + n_rej_visual + n_rej_shared_infobox_visual
             + n_rej_intersection + n_rej_type_not_shared)
    w = max(total, 1)
    print("\nInstance generation cascade:")
    print(f"  (mention, image) candidates:      {total:>10,}")
    if cfg.require_unique_candidate_infoboxes:
        print(f"  – shared infobox (mentions):      {n_rej_shared_infobox:>10,}  (mention-level)")
    print(f"  – visual pool out of [{cfg.visual_min},{cfg.visual_max or '∞'}]:        "
          f"{n_rej_visual:>10,}  ({100*n_rej_visual/w:.1f}%)")
    if cfg.require_unique_candidate_infoboxes:
        print(f"  – shared infobox (visual pool):   {n_rej_shared_infobox_visual:>10,}  "
              f"({100*n_rej_shared_infobox_visual/w:.1f}%)")
    print(f"  – intersection ≠ 1:               {n_rej_intersection:>10,}  "
          f"({100*n_rej_intersection/w:.1f}%)")
    if cfg.require_answer_type_shared:
        print(f"  – answer type unique in text:     {n_rej_type_not_shared:>10,}  "
              f"({100*n_rej_type_not_shared/w:.1f}%)")
    print("  ─────────────────────────────────────────────────────")
    print(f"  Instances written:                {n_written:>10,}  → {instances_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input",      type=Path, help="dataset.jsonl (scrape output)")
    ap.add_argument("output_dir", type=Path,
                    help="split output directory (e.g. output/split_10_text/)")
    ap.add_argument("--config",   type=Path, default=_DEFAULT_CONFIG,
                    help=f"filter thresholds YAML (default: {_DEFAULT_CONFIG})")
    ap.add_argument("--workers",  type=int, default=None,
                    help="parallel scan workers (default: cpu_count; "
                         "use 1 for sequential with progress bar)")
    args = ap.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Not found: {args.input}")
    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")

    n_workers = args.workers if args.workers is not None else (os.cpu_count() or 4)
    build(args.input, args.output_dir, Config.load(args.config), n_workers=n_workers)


if __name__ == "__main__":
    main()
