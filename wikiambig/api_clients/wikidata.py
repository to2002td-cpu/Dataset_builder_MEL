"""
Wikidata API client (wbgetentities + SPARQL).

Uses wikidata.org — separate quota from en.wikipedia.org.
All SPARQL queries run via https://query.wikidata.org/sparql.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WIKIDATA_API = "https://www.wikidata.org/w/api.php"
SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
HEADERS = {"User-Agent": "wikiambig/2.0 (research project; https://github.com)"}

# Entity type roots and their coarse labels.
ROOT_MAP: dict[str, str] = {
    "Q5": "PERS",        # human
    "Q43229": "ORG",     # organization
    "Q618123": "LOC",    # geographical entity
}
TYPE_PRIORITY = ["PERS", "ORG", "LOC"]

# ---------------------------------------------------------------------------
# Thread-local session
# ---------------------------------------------------------------------------

_local = threading.local()


def _make_session(backoff_factor: float = 1.0) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=backoff_factor,
        respect_retry_after_header=True,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(HEADERS)
    return session


def _wd_session() -> requests.Session:
    if not hasattr(_local, "wikidata"):
        _local.wikidata = _make_session()
    return _local.wikidata


def _sparql_session() -> requests.Session:
    if not hasattr(_local, "sparql"):
        _local.sparql = _make_session(backoff_factor=2.0)
    return _local.sparql


# ---------------------------------------------------------------------------
# wbgetentities — bulk QID resolution to Wikipedia URL
# ---------------------------------------------------------------------------

def get_titles_to_qids(titles: list[str]) -> dict[str, str]:
    """
    Resolve up to 50 Wikipedia article titles to Wikidata QIDs.
    Uses wbgetentities?sites=enwiki&titles=...&props=info|sitelinks.
    Returns {title: QID}. Titles without a Wikidata entity are omitted.
    """
    params: dict[str, Any] = {
        "action": "wbgetentities",
        "sites": "enwiki",
        "titles": "|".join(titles),
        "props": "info|sitelinks",
        "sitefilter": "enwiki",
        "format": "json",
    }
    resp = _wd_session().get(WIKIDATA_API, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    result: dict[str, str] = {}
    for entity in payload.get("entities", {}).values():
        qid = entity.get("id", "")
        if not qid or qid.startswith("-"):
            continue
        enwiki_title = entity.get("sitelinks", {}).get("enwiki", {}).get("title", "")
        if enwiki_title:
            result[enwiki_title] = qid
    return result


# ---------------------------------------------------------------------------
# SPARQL — combined entity data + type classification (S3, S6)
# ---------------------------------------------------------------------------

def sparql_get(query: str, timeout: int = 60) -> list[dict[str, Any]]:
    """Execute a SPARQL SELECT query and return the bindings list."""
    resp = _sparql_session().get(
        SPARQL_ENDPOINT,
        params={"query": query, "format": "json"},
        headers={"Accept": "application/sparql-results+json"},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()["results"]["bindings"]


def get_entities_data_and_types(qids: list[str]) -> dict[str, dict[str, Any]]:
    """
    Fetch name, English description, Wikipedia URL AND coarse type
    (PERS/ORG/LOC) for a batch of QIDs in a single SPARQL query.

    Returns {QID: {name, desc, url_wikipedia, type}}, where ``type`` is
    "PERS", "ORG", "LOC", or None (no root match).

    Replaces what used to be two separate round-trips (entity data, then
    entity type) over the same QID batch — halving Wikidata SPARQL traffic,
    the same "fetch more per call" pattern as ``get_wiki_entity_data_batch``.

    SERVICE wikibase:label automatically binds ?itemLabel and ?itemDescription.
    ?article gives the full enwiki URL directly — no separate sitelinks call needed.
    The type root match (P31/P279* traversal, PERS > ORG > LOC priority,
    disambiguation pages excluded) is scoped inside its own OPTIONAL so that
    entity data is still returned for QIDs that match no root.
    """
    qids_str = " ".join(f"wd:{q}" for q in qids)
    roots_str = " ".join(f"wd:{r}" for r in ROOT_MAP)
    query = f"""
    SELECT ?item ?itemLabel ?itemDescription ?article ?root WHERE {{
      VALUES ?item {{ {qids_str} }}
      OPTIONAL {{
        ?article schema:about ?item ;
                 schema:isPartOf <https://en.wikipedia.org/> .
      }}
      OPTIONAL {{
        VALUES ?root {{ {roots_str} }}
        ?item wdt:P31/wdt:P279* ?root .
        FILTER NOT EXISTS {{ ?item wdt:P31/wdt:P279* wd:Q4167410 }}
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en" }}
    }}
    """

    for attempt in range(3):
        try:
            bindings = sparql_get(query, timeout=90)
            break
        except Exception:
            if attempt < 2:
                time.sleep(5 * (attempt + 1))
            else:
                return {q: {"name": None, "desc": "", "url_wikipedia": None,
                            "type": None} for q in qids}

    data: dict[str, dict[str, Any]] = {}
    roots: dict[str, set[str]] = {}
    for row in bindings:
        qid = row["item"]["value"].split("/")[-1]
        if qid not in data:
            data[qid] = {
                "name": None,
                "desc": "",
                "url_wikipedia": None,
            }
            roots[qid] = set()

        if "itemLabel" in row:
            data[qid]["name"] = row["itemLabel"]["value"]

        if "itemDescription" in row and not data[qid]["desc"]:
            data[qid]["desc"] = row["itemDescription"]["value"]

        if "article" in row and data[qid]["url_wikipedia"] is None:
            data[qid]["url_wikipedia"] = row["article"]["value"]

        if "root" in row:
            roots[qid].add(ROOT_MAP[row["root"]["value"].split("/")[-1]])

    for qid, tags in roots.items():
        matched: str | None = None
        for p in TYPE_PRIORITY:
            if p in tags:
                matched = p
                break
        data[qid]["type"] = matched

    return data
