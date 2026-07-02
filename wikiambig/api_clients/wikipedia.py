"""
Wikipedia API client.

All calls go to https://en.wikipedia.org/w/api.php.
Handles:
  - maxlag detection (HTTP 200 with JSON error body — sleep + one retry)
  - Retry adapter (429, 5xx) via urllib3
  - Thread-local sessions (one per worker, reused across calls)
"""

from __future__ import annotations

import re
import threading
import time
from typing import Any
from urllib.parse import unquote

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

WIKI_API = "https://en.wikipedia.org/w/api.php"
HEADERS = {"User-Agent": "wikiambig/2.0 (research project; https://github.com)"}

# ---------------------------------------------------------------------------
# Non-photographic media formats — out of scope for an image-reuse dataset
# (icons/diagrams as vector graphics, animations, audio, video). This is a
# scope boundary, not a quality heuristic: everything that passes it is
# collected, and the data-driven `max_image_usage` filter (applied at filter
# time, on actual reuse statistics) is what separates illustrative photos
# from generic stock images — no hand-curated filename blocklist needed.
# ---------------------------------------------------------------------------

NON_IMAGE_EXTENSIONS: tuple[str, ...] = (
    ".svg",
    ".gif",
    ".ogg",
    ".ogv",
    ".webm",
    ".midi",
    ".mp3",
    ".wav",
)

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


def _session() -> requests.Session:
    if not hasattr(_local, "wiki"):
        _local.wiki = _make_session()
    return _local.wiki


# ---------------------------------------------------------------------------
# Core GET with maxlag handling
# ---------------------------------------------------------------------------

def wiki_get(params: dict[str, Any], timeout: int = 30) -> dict[str, Any]:
    """
    GET the Wikipedia API. Handles maxlag responses (HTTP 200 + JSON error).
    Always sets format=json and maxlag=5.
    """
    params = dict(params)
    params.setdefault("format", "json")
    params["maxlag"] = "5"

    resp = _session().get(WIKI_API, params=params, timeout=timeout)
    resp.raise_for_status()
    data: dict[str, Any] = resp.json()

    if data.get("error", {}).get("code") == "maxlag":
        lag = float(data["error"].get("lag", 5))
        time.sleep(lag + 1)
        resp = _session().get(WIKI_API, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()

    return data


# ---------------------------------------------------------------------------
# S1: category member listing
# ---------------------------------------------------------------------------

def get_category_members(
    category: str, limit: int = 500, cmtype: str = "page"
) -> list[dict[str, Any]]:
    """
    Yield all members of a Wikipedia category.

    ``cmtype`` selects member kinds: "page", "subcat", "file", or a
    "|"-separated combination (e.g. "page|subcat" for a recursive crawl).

    Returns list of {title, pageid, ns} dicts — ``ns`` is 0 for articles
    and 14 for subcategories, letting callers tell them apart.
    """
    members: list[dict[str, Any]] = []
    params: dict[str, Any] = {
        "action": "query",
        "list": "categorymembers",
        "cmtitle": category,
        "cmlimit": str(min(limit, 500)),
        "cmtype": cmtype,
    }
    while True:
        data = wiki_get(params)
        for m in data.get("query", {}).get("categorymembers", []):
            members.append({"title": m["title"], "pageid": m["pageid"], "ns": m["ns"]})
        cont = data.get("continue")
        if not cont:
            break
        params.update(cont)
    return members


# ---------------------------------------------------------------------------
# S2: wikitext retrieval
# ---------------------------------------------------------------------------

def get_wikitext_batch(titles: list[str]) -> dict[str, str]:
    """
    Fetch raw wikitext for up to 50 page titles in one API call.
    Returns {title: wikitext} keyed by the original requested title.
    Follows redirects transparently (redirects=1).
    Missing or missing-revision pages map to "".
    """
    params: dict[str, Any] = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "revisions",
        "rvprop": "content",
        "rvslots": "main",
        "redirects": "1",
    }
    data = wiki_get(params)
    query = data.get("query", {})

    # Build a chain: resolved_title → original_submitted_title
    # normalized: submitted → api-canonical  (e.g. lowercase first letter fix)
    # redirects:  api-canonical/redirect → redirect target
    # We need: final_page_title → original submitted title
    resolved: dict[str, str] = {}

    # Step 1: normalisation map (api title → original submitted title)
    for n in query.get("normalized", []):
        resolved[n["to"]] = n["from"]

    # Step 2: follow redirect chain (redirect source → target)
    for r in query.get("redirects", []):
        src = r["from"]
        dst = r["to"]
        # Carry the original title forward through the redirect chain
        original = resolved.pop(src, src)
        resolved[dst] = original

    result: dict[str, str] = {}
    for page in query.get("pages", {}).values():
        page_title = page.get("title", "")
        original = resolved.get(page_title, page_title)
        revisions = page.get("revisions", [])
        if revisions:  # noqa: SIM108 — kept explicit for readability
            wikitext = revisions[0].get("slots", {}).get("main", {}).get("*", "")
        else:
            wikitext = ""
        result[original] = wikitext

    return result


# ---------------------------------------------------------------------------
# S2: QID resolution from page titles
# ---------------------------------------------------------------------------

def get_qids_from_titles(titles: list[str]) -> dict[str, str]:
    """
    Resolve up to 50 Wikipedia page titles to their Wikidata QIDs.
    Uses prop=pageprops&ppprop=wikibase_item.
    Returns {title: QID} for titles that have a linked Wikidata entity.
    """
    params: dict[str, Any] = {
        "action": "query",
        "titles": "|".join(titles),
        "prop": "pageprops",
        "ppprop": "wikibase_item",
    }
    data = wiki_get(params)
    result: dict[str, str] = {}

    norm_map: dict[str, str] = {}
    for n in data.get("query", {}).get("normalized", []):
        norm_map[n["to"]] = n["from"]

    for page in data.get("query", {}).get("pages", {}).values():
        title = page.get("title", "")
        original = norm_map.get(title, title)
        qid = page.get("pageprops", {}).get("wikibase_item")
        if qid:
            result[original] = qid

    return result


# ---------------------------------------------------------------------------
# S5: image list per article
# ---------------------------------------------------------------------------

def is_non_image_format(filename: str) -> bool:
    """Return True for vector/audio/video files — not photographic images."""
    lower = filename.lower()
    return any(lower.endswith(ext) for ext in NON_IMAGE_EXTENSIONS)


# ---------------------------------------------------------------------------
# S5: image info + file usage
# ---------------------------------------------------------------------------

def get_image_info_batch(
    filenames: list[str],
) -> dict[str, dict[str, Any]]:
    """
    Fetch imageinfo (URL, dimensions, MIME type, license) and fileusage
    (article titles) for up to 50 filenames in a single API call, paginating
    through *every* fileusage page so heavily-reused images (flags, seals,
    maps) aren't silently truncated at the API's ~500-per-title page size.
    funamespace=0 restricts usage to article pages only (~40% fewer results).

    Scope note: this is en.wikipedia *local* article-namespace usage only —
    it does not include other-language Wikipedias, sister projects, or
    Commons-side reuse (what a Commons File page's "File usage on other
    wikis" section shows via the GlobalUsage extension). See the "Visual
    signal" limitations in README for what that means for `used_by`/
    `n_used_by` and the filters/stats derived from them.

    License is read from Commons ``extmetadata`` — ``LicenseShortName`` when
    present (e.g. "CC BY-SA 4.0", "Public domain"), falling back to the
    machine-readable ``License`` slug; "" when neither is recorded.

    Returns ``{filename: {"url", "width", "height", "mime", "license", "usage": [title, …]}}``.
    """
    params: dict[str, Any] = {
        "action": "query",
        "titles": "|".join(filenames),
        "prop": "imageinfo|fileusage",
        "iiprop": "url|size|mime|extmetadata",
        "fulimit": "max",
        "funamespace": "0",
    }
    data = wiki_get(params, timeout=60)

    result: dict[str, dict[str, Any]] = {}
    for page in data.get("query", {}).get("pages", {}).values():
        fname = page.get("title", "")
        if not fname:
            continue
        ii = page.get("imageinfo", [])
        info = ii[0] if ii else {}
        extmeta = info.get("extmetadata", {})
        license_ = (
            extmeta.get("LicenseShortName", {}).get("value")
            or extmeta.get("License", {}).get("value")
            or ""
        )
        result[fname] = {
            "url": info.get("url", ""),
            "width": info.get("width"),
            "height": info.get("height"),
            "mime": info.get("mime", ""),
            "license": license_,
            "usage": [u["title"] for u in page.get("fileusage", []) if u.get("title")],
        }

    # MediaWiki continuation advances one title's fileusage at a time within
    # a multi-title batch — keep following `continue` until every title's
    # usage list is fully drained, merging each page into its existing entry.
    cont = data.get("continue")
    while cont:
        cont_data = wiki_get({**params, **cont}, timeout=60)
        for page in cont_data.get("query", {}).get("pages", {}).values():
            fname = page.get("title", "")
            if fname in result:
                result[fname]["usage"].extend(
                    u["title"] for u in page.get("fileusage", []) if u.get("title")
                )
        cont = cont_data.get("continue")

    return result


# ---------------------------------------------------------------------------
# S4: combined intro + image list per article (single API call)
# ---------------------------------------------------------------------------

def get_wiki_entity_data_batch(page_titles: list[str]) -> dict[str, dict]:
    """
    Fetch the full first paragraph and image filename list for up to 50
    Wikipedia titles in a single API call (``prop=extracts|images``).

    The first paragraph is returned verbatim — never truncated — so every
    consumer sees exactly what Wikipedia published.

    Titles may be URL-encoded (``%C3%A9`` etc.) — decoded automatically.
    Image continuation is handled internally (rare for pages with >500 images).

    Also fetches each page's PageImages-derived lead/infobox image
    (``piprop=name``) — the image actually displayed in the en.wikipedia
    infobox, as opposed to Wikidata's (possibly different) P18 claim.

    Returns ``{original_title: {"intro": str, "images": [filename, …],
    "infobox_image": str | None}}``.
    """
    decoded: dict[str, str] = {unquote(t).replace("_", " "): t for t in page_titles}
    result: dict[str, dict] = {
        t: {"intro": "", "images": [], "infobox_image": None} for t in page_titles
    }

    base_params: dict[str, Any] = {
        "action": "query",
        "titles": "|".join(decoded),
        "prop": "extracts|images|pageimages",
        "exintro": "1",
        "explaintext": "1",
        "imlimit": "max",
        "piprop": "name",
        "redirects": "1",
    }
    params = dict(base_params)
    resolved: dict[str, str] = {}
    first_call = True

    while True:
        data = wiki_get(params)
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

            if "extract" in page:
                result[original]["intro"] = (page.get("extract") or "").split("\n\n")[0].strip()

            if "pageimage" in page:
                result[original]["infobox_image"] = page["pageimage"]

            for img in page.get("images", []):
                fname = img.get("title", "")
                if fname and not is_non_image_format(fname):
                    result[original]["images"].append(fname)

        cont = data.get("continue")
        if not cont:
            break
        # Continuation can carry `excontinue` as well as `imcontinue` — a 50-title
        # batch routinely needs more than one page of extracts (exlimit defaults
        # to 20), so keep requesting both props or later titles' intros are
        # silently dropped.
        params = dict(base_params)
        params.update(cont)

    return result


# ---------------------------------------------------------------------------
# Wikitext link extraction (used by S2)
# ---------------------------------------------------------------------------

_LIST_ITEM_RE = re.compile(
    r"^([*#]+)\s*['\`_~]*\[\[([^\]|#]+)(?:\|[^\]]+)?\]\](.*)",
    re.UNICODE,
)
_SEE_ALSO_RE = re.compile(r"==\s*[Ss]ee also\s*==")


def extract_entity_links(wikitext: str) -> list[str]:
    """
    Parse a disambiguation page's wikitext and return a list of entity link targets.

    Rules (derived from the working old-code implementation):
      - Stop before ==See also==.
      - Level-1 list items (*): include only when followed by a non-empty description
        (at least one alphanumeric char after the link, not starting with ':').
      - Level-2+ list items (**, ***, …): include unconditionally.
      - Skip links whose target contains ':' (non-article namespaces).
    """
    see_also = _SEE_ALSO_RE.search(wikitext)
    if see_also:
        wikitext = wikitext[: see_also.start()]

    links: list[str] = []
    for line in wikitext.splitlines():
        line = line.strip()
        m = _LIST_ITEM_RE.match(line)
        if not m:
            continue

        bullets = m.group(1)
        target = m.group(2).strip()
        suffix = m.group(3).strip()

        if ":" in target:
            continue

        level = len(bullets)
        if level == 1:
            # Only include level-1 items that have a descriptive suffix.
            if any(c.isalnum() for c in suffix) and not suffix.startswith(":"):
                links.append(target)
        else:
            links.append(target)

    return links
