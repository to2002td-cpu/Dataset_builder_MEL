"""
Fetch related QIDs from Wikidata for a given set of properties.

Uses batched SPARQL SELECT queries (multiple QIDs per request) with caching
and threading.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "wikiambig/2.0 (research project; https://github.com)"}

_local = threading.local()
_cache: dict[str, set[str]] = {}
_cache_lock = threading.Lock()

_stats_lock = threading.Lock()
_stats = {"requests": 0, "cache_hits": 0, "errors": 0}

BATCH_SIZE = 50


def _session() -> requests.Session:
    if not hasattr(_local, "sparql"):
        s = requests.Session()
        retry = Retry(
            total=5,
            backoff_factor=2.0,
            respect_retry_after_header=True,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.headers.update(HEADERS)
        _local.sparql = s
    return _local.sparql


def get_stats() -> dict:
    with _stats_lock:
        return dict(_stats)


def _fetch_batch(qids: list[str], forbidden_props: list[str]) -> dict[str, set[str]]:
    """
    Fetch related QIDs for a batch of QIDs in a single SPARQL query.
    Returns {qid: {related_qids}}.
    """
    items_str = " ".join(f"wd:{q}" for q in qids)
    props_str = " ".join(f"wdt:{p}" for p in forbidden_props)
    query = f"""SELECT ?item ?related WHERE {{
  VALUES ?item {{ {items_str} }}
  VALUES ?p {{ {props_str} }}
  {{ ?item ?p ?related }} UNION {{ ?related ?p ?item }}
}}"""

    result: dict[str, set[str]] = {q: set() for q in qids}
    try:
        resp = _session().get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=60,
        )
        resp.raise_for_status()
        for row in resp.json()["results"]["bindings"]:
            item = row["item"]["value"].split("/")[-1]
            related = row["related"]["value"].split("/")[-1]
            if item in result:
                result[item].add(related)
    except Exception:
        with _stats_lock:
            _stats["errors"] += 1

    with _stats_lock:
        _stats["requests"] += 1
    with _cache_lock:
        _cache.update(result)

    return result


def fetch_related_qids(qid: str, forbidden_props: list[str]) -> set[str]:
    """
    Return all QIDs directly linked to *qid* by any of the given properties
    (in both directions).  Reads from cache (populated by prefetch_qids).
    """
    with _cache_lock:
        if qid in _cache:
            with _stats_lock:
                _stats["cache_hits"] += 1
            return _cache[qid]
    return _fetch_batch([qid], forbidden_props).get(qid, set())


def prefetch_qids(
    all_qids: list[str],
    forbidden_props: list[str],
    max_workers: int = 5,
) -> None:
    """
    Pre-populate the cache for a batch of QIDs using batched SPARQL + threading.
    """
    if not forbidden_props:
        return

    with _cache_lock:
        to_fetch = [q for q in all_qids if q not in _cache]
    if not to_fetch:
        return

    batches = [to_fetch[i:i + BATCH_SIZE] for i in range(0, len(to_fetch), BATCH_SIZE)]

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_batch, batch, forbidden_props): batch for batch in batches}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Prefetch SPARQL"):
            fut.result()
