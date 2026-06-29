"""
Wikimedia Commons API client.

Provides the *second* image source for an entity, parallel to the en.wikipedia
article images collected in S4/S5: the files listed on the entity's Commons
**gallery** page (the main-namespace ``commonswiki`` sitelink, e.g.
``https://commons.wikimedia.org/wiki/Carolus_V,_Imperator_Romanus_Sacer`` for
Charles V). Both sources are kept side by side so they can be compared
downstream — this client never touches the Wikipedia-side artifacts.

Two calls:
  - ``get_commons_gallery_titles`` : QID → ``commonswiki`` sitelink title
    (Wikidata ``wbgetentities``, 50 QIDs/call). Galleries only — entities with
    no gallery sitelink are simply omitted (no P373 category fallback).
  - ``get_commons_images_batch`` : gallery title → embedded image filenames
    (Commons ``prop=images``, 50 titles/call), reusing the same
    ``is_non_image_format`` scope filter as the Wikipedia side.
"""

from __future__ import annotations

import threading
import time
from typing import Any
from urllib.parse import unquote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from wikiambig.api_clients.wikipedia import is_non_image_format

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"
HEADERS = {"User-Agent": "wikiambig/2.0 (research project; https://github.com)"}

_local = threading.local()


def _make_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=1.0,
        respect_retry_after_header=True,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update(HEADERS)
    return session


def _session() -> requests.Session:
    if not hasattr(_local, "commons"):
        _local.commons = _make_session()
    return _local.commons


def _api_get(url: str, params: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    """GET a MediaWiki API endpoint with maxlag handling (HTTP 200 + JSON error)."""
    params = dict(params)
    params.setdefault("format", "json")
    params["maxlag"] = "5"

    resp = _session().get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()

    if data.get("error", {}).get("code") == "maxlag":
        lag = float(data["error"].get("lag", 5))
        time.sleep(lag + 1)
        resp = _session().get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

    return data


# ---------------------------------------------------------------------------
# QID → Commons gallery title (commonswiki sitelink)
# ---------------------------------------------------------------------------

def get_commons_gallery_titles(qids: list[str]) -> dict[str, str]:
    """
    Resolve up to 50 QIDs to their Commons **gallery** page title via the
    ``commonswiki`` sitelink. Returns ``{QID: gallery_title}`` only for
    entities that have one (galleries are sparse — most entities have only a
    Commons *category*, which is intentionally not used here).
    """
    params: dict[str, Any] = {
        "action": "wbgetentities",
        "ids": "|".join(qids),
        "props": "sitelinks",
        "sitefilter": "commonswiki",
    }
    data = _api_get(WIKIDATA_API, params)

    result: dict[str, str] = {}
    for qid, entity in data.get("entities", {}).items():
        if qid.startswith("-") or "missing" in entity:
            continue
        title = entity.get("sitelinks", {}).get("commonswiki", {}).get("title", "")
        if title:
            result[qid] = title
    return result


# ---------------------------------------------------------------------------
# Commons gallery title → embedded image filenames
# ---------------------------------------------------------------------------

def get_commons_images_batch(titles: list[str]) -> dict[str, list[str]]:
    """
    Fetch the embedded image filenames (``prop=images``) for up to 50 Commons
    gallery titles in one call, following ``imcontinue`` pagination. Non-image
    formats (svg/audio/video) are dropped with the same filter the Wikipedia
    side uses. Returns ``{original_title: [filename, …]}``.
    """
    decoded: dict[str, str] = {unquote(t).replace("_", " "): t for t in titles}
    result: dict[str, list[str]] = {t: [] for t in titles}

    base_params: dict[str, Any] = {
        "action": "query",
        "titles": "|".join(decoded),
        "prop": "images",
        "imlimit": "max",
        "redirects": "1",
    }
    params = dict(base_params)
    resolved: dict[str, str] = {}
    first_call = True

    while True:
        data = _api_get(COMMONS_API, params)
        query = data.get("query", {})

        if first_call:
            for n in query.get("normalized", []):
                resolved[n["to"]] = decoded.get(n["from"], n["from"])
            for r in query.get("redirects", []):
                src, dst = r["from"], r["to"]
                resolved[dst] = resolved.pop(src, decoded.get(src, src))
            first_call = False

        for page in query.get("pages", {}).values():
            page_title = page.get("title", "")
            original = resolved.get(page_title, decoded.get(page_title, page_title))
            if original not in result:
                continue
            for img in page.get("images", []):
                fname = img.get("title", "")
                if fname and not is_non_image_format(fname):
                    result[original].append(fname)

        cont = data.get("continue")
        if not cont:
            break
        params = dict(base_params)
        params.update(cont)

    return result
