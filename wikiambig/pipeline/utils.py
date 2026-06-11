"""Shared helpers used across pipeline stages."""

from __future__ import annotations

import json
import os
from pathlib import Path


def atomic_write(path: Path, data: dict) -> None:
    """Write *data* as JSON to *path* via a tmp file — safe on interrupt."""
    tmp = str(path) + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, str(path))


def title_from_url(url: str) -> str:
    """Extract the Wikipedia article title from a full en.wikipedia.org URL."""
    return url.split("/wiki/", 1)[-1].replace("_", " ")


def load_qids_from_jsonl(path: Path) -> list[str]:
    """Return a sorted, deduplicated list of all QIDs from entity_links.jsonl."""
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                seen.update(json.loads(line).get("qids", []))
    return sorted(seen)
