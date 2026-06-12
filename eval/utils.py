"""
Shared helpers for the evaluation loop: prompt assembly, image caching,
response parsing, and metrics.
"""

from __future__ import annotations

import hashlib
import random
import re
import time
from io import BytesIO
from pathlib import Path

import requests

_CACHE_DIR = Path(__file__).resolve().parent / "image_cache"
# Same UA as wikiambig/api_clients — Wikimedia rate-limits unknown agents (429)
_HEADERS = {"User-Agent": "wikiambig/2.0 (research project; https://github.com)"}
_NUMBER_RE = re.compile(r'"number"\s*:\s*(-?\d+)')
_RANKING_RE = re.compile(r'"ranking"\s*:\s*\[([^\]]*)\]')


def fetch_image(url: str, max_side: int = 1280, timeout: int = 30,
                retries: int = 3) -> Path | None:
    """Download an image to the local cache (resized to max_side); None on failure."""
    from PIL import Image

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _CACHE_DIR / (hashlib.sha1(url.encode()).hexdigest() + ".jpg")
    if path.exists():
        return path
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=_HEADERS, timeout=timeout)
            if r.status_code == 429:
                time.sleep(int(r.headers.get("Retry-After", 2 ** attempt)))
                continue
            r.raise_for_status()
            img = Image.open(BytesIO(r.content)).convert("RGB")
            img.thumbnail((max_side, max_side))
            img.save(path, "JPEG", quality=90)
            return path
        except Exception:
            if attempt == retries - 1:
                return None
            time.sleep(2 ** attempt)
    return None


def _fields_sentence(use_desc: bool, use_img: bool) -> str:
    """Wording of the {fields} slot, derived from the prompt config flags."""
    if use_desc and use_img:
        return "a name, a brief description, and an image"
    if use_desc:
        return "a name and a brief description"
    if use_img:
        return "a name and an image"
    return "a name"


def format_prompt(template: dict, mention: str, candidates: list[dict], answer_qid: str,
                  prompt_cfg: dict, image_max_side: int = 1280, rng=None):
    """
    Build the candidate list into a contrastive template and return the label
    (the correct answer's position in the list, 1-based).

    template   : parsed eval/prompts/<name>.yaml (header/candidate_line/…).
    prompt_cfg : the config `prompt:` section; its flags drive both the
                 content and the {fields} wording, so prompt text and config
                 cannot drift apart.
    Returns (parts, label, order): parts is a list of ("text"|"image", value)
    for Model.generate (candidate images interleaved after their line), and
    order is the candidate list as QIDs in presentation order.
    """
    use_desc = prompt_cfg.get("use_entity_description", False)
    use_img = prompt_cfg.get("use_entity_image", False)
    rng = rng or random.Random(0)
    pool = list(candidates)
    if prompt_cfg.get("shuffle_candidates", False):
        rng.shuffle(pool)
    else:
        pool.sort(key=lambda e: e["qid"] != answer_qid)  # answer first → label 1

    header = template["header"].format(
        mention=mention, fields=_fields_sentence(use_desc, use_img))
    parts: list[tuple[str, object]] = [("text", header)]
    order: list[str] = []
    label = -1
    for i, e in enumerate(pool, start=1):
        line = template["candidate_line"].format(number=i, name=e["name"])
        if use_desc:
            # Wikipedia intro (first sentence) when available, Wikidata desc as fallback
            desc = (e.get("intro") or "").split(". ")[0].strip() or e.get("desc")
            if desc:
                line += template.get("candidate_desc", " — {desc}").format(desc=desc)
        parts.append(("text", line))
        if use_img and e.get("infobox_img"):
            img_path = fetch_image(e["infobox_img"], max_side=image_max_side)
            if img_path:
                parts.append(("image", img_path))
        order.append(e["qid"])
        if e["qid"] == answer_qid:
            label = i
    parts.append(("text", template["footer"]))
    if prompt_cfg.get("answer_none", False):
        parts.append(("text", template["none_hint"]))
    return parts, label, order


def parse_number(response: str) -> int | None:
    """Extract the predicted option number from the model response."""
    m = _NUMBER_RE.search(response)
    if m:
        return int(m.group(1))
    m = re.search(r"-?\d+", response)
    return int(m.group(0)) if m else None


def parse_ranking(response: str) -> list[int] | None:
    """Extract the predicted candidate ranking from the model response."""
    m = _RANKING_RE.search(response)
    if not m:
        m = re.search(r"\[([\d\s,]+)\]", response)
    if not m:
        return None
    try:
        return [int(x) for x in m.group(1).replace(",", " ").split()]
    except ValueError:
        return None


def compute_metrics(preds: list[int | None], labels: list[int],
                    pool_sizes: list[int] | None = None) -> dict:
    """Accuracy over option-number predictions; unparsed responses count as wrong."""
    n = len(labels)
    metrics = {
        "n": n,
        "n_correct": sum(p == l for p, l in zip(preds, labels)),
        "n_unparsed": sum(p is None for p in preds),
        "n_none": sum(p == -1 for p in preds),
    }
    metrics["accuracy"] = metrics["n_correct"] / n if n else 0.0
    if pool_sizes:
        metrics["random_baseline"] = sum(1 / s for s in pool_sizes) / len(pool_sizes)
    return metrics


def compute_ranking_metrics(gold_ranks: list[int | None],
                            ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict:
    """
    Retrieval metrics from the gold entity's position in each predicted
    ranking (1-based; None = unparsed response or gold missing from ranking,
    scored as not retrieved). With a single relevant item per instance,
    recall@k equals hit@k and precision@k is hit@k / k.
    """
    from math import log2

    n = len(gold_ranks)
    metrics: dict = {
        "n": n,
        "n_unparsed": sum(r is None for r in gold_ranks),
        "mrr": sum(1 / r for r in gold_ranks if r) / n if n else 0.0,
    }
    for k in ks:
        hits = sum(1 for r in gold_ranks if r and r <= k)
        metrics[f"hit@{k}"] = hits / n if n else 0.0
        metrics[f"precision@{k}"] = hits / (n * k) if n else 0.0
        metrics[f"recall@{k}"] = metrics[f"hit@{k}"]
        metrics[f"ndcg@{k}"] = (
            sum(1 / log2(r + 1) for r in gold_ranks if r and r <= k) / n if n else 0.0)
    return metrics
