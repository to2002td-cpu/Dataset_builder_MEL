"""
Wikidata SPARQL enrichment for the split filters in make_split.py:
forbidden-property relations and instance-of class membership.

Relations — fetch, for a set of QIDs, the QIDs they are directly linked to by
any of a given list of Wikidata properties. Used by the candidate "forbidden
properties" filter: a mention whose candidate pool contains two entities linked
by a located-in / part-of property (e.g. "New York City" P131 "New York" state)
is irresolvable and dropped. Only the *forward* direction (?item ?p ?related)
is queried. That is sufficient for pairwise detection: the filter iterates over
every candidate, so an edge A ?p B between two candidates is caught from
whichever endpoint stores the truthy claim. Avoiding the reverse direction
keeps the per-QID result set small (a place's parents, not its thousands of
children).

Instance-of — fetch which of a set of QIDs satisfy P31/P279* membership of
given Wikidata classes. Used by the entity ``exclude_instance_of`` filter
(e.g. drop administrative territorial entities from the KB).

Both use batched SPARQL SELECT queries (multiple QIDs per request) with
caching and threading.
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
    Fetch the forward-linked QIDs for a batch of QIDs in a single SPARQL query.
    Returns {qid: {related_qids}}.
    """
    items_str = " ".join(f"wd:{q}" for q in qids)
    props_str = " ".join(f"wdt:{p}" for p in forbidden_props)
    query = f"""SELECT ?item ?related WHERE {{
  VALUES ?item {{ {items_str} }}
  VALUES ?p {{ {props_str} }}
  ?item ?p ?related
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
    Return all QIDs *qid* links to via any of the given properties (forward
    direction). Reads from cache (populated by prefetch_qids).
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


# ---------------------------------------------------------------------------
# instance-of / subclass-of class membership (KB entity exclusion filter)
# ---------------------------------------------------------------------------

def _fetch_instanceof_batch(qids: list[str], classes: list[str]) -> set[str]:
    """
    Return the subset of *qids* that are an instance or subclass of any class in
    *classes*, i.e. that satisfy  ?item wdt:P31/wdt:P279* ?class  for some
    ?class in the set. One batched SPARQL query per call.
    """
    items_str = " ".join(f"wd:{q}" for q in qids)
    classes_str = " ".join(f"wd:{c}" for c in classes)
    query = f"""SELECT DISTINCT ?item WHERE {{
  VALUES ?item {{ {items_str} }}
  VALUES ?class {{ {classes_str} }}
  ?item wdt:P31/wdt:P279* ?class
}}"""

    matched: set[str] = set()
    try:
        resp = _session().get(
            SPARQL_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={"Accept": "application/sparql-results+json"},
            timeout=90,
        )
        resp.raise_for_status()
        for row in resp.json()["results"]["bindings"]:
            matched.add(row["item"]["value"].split("/")[-1])
    except Exception:
        with _stats_lock:
            _stats["errors"] += 1

    with _stats_lock:
        _stats["requests"] += 1

    return matched


def fetch_instanceof_matches(
    all_qids: list[str],
    classes: list[str],
    max_workers: int = 5,
) -> set[str]:
    """
    Return every QID in *all_qids* that is an instance/subclass (P31/P279*) of at
    least one Wikidata class in *classes*. Batched SPARQL + threading.

    Used by make_split's ``entity.exclude_instance_of`` filter to drop
    administrative territorial entities (states, regions, …) from the KB.
    """
    if not classes or not all_qids:
        return set()

    batches = [all_qids[i:i + BATCH_SIZE] for i in range(0, len(all_qids), BATCH_SIZE)]
    matched: set[str] = set()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_fetch_instanceof_batch, batch, classes): batch for batch in batches}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Exclude instance-of"):
            matched |= fut.result()
    return matched
