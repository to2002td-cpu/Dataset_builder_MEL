"""Pydantic v2 data models for the wikiambig dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

try:
    import orjson as _json

    def _loads(data: bytes) -> Any:
        return _json.loads(data)

except ImportError:
    def _loads(data: bytes) -> Any:  # type: ignore[misc]
        return json.loads(data)

from pydantic import BaseModel, Field, model_validator


class Image(BaseModel):
    """A single image appearing in a Wikipedia article body."""

    url: str
    used_by: list[str] = Field(default_factory=list)
    """QIDs of all entities whose Wikipedia articles embed this image."""
    n_used_by: int = 0
    is_infobox: bool = False
    """True when this image is the Wikipedia infobox image of its parent entity."""
    width: Optional[int] = None
    height: Optional[int] = None
    mime: str = ""
    license: str = ""
    """Short license name from Commons extmetadata, e.g. "CC BY-SA 4.0"; "" when unrecorded."""

    @model_validator(mode="after")
    def _sync_count(self) -> "Image":
        if self.n_used_by == 0 and self.used_by:
            object.__setattr__(self, "n_used_by", len(self.used_by))
        return self


class Entity(BaseModel):
    """A candidate entity for a disambiguation mention."""

    qid: str
    name: str
    desc: str = ""
    intro: str = ""
    """Full first paragraph of the Wikipedia article, verbatim; empty when not fetched."""
    type: str = "OTHER"
    """Coarse semantic type: PERS | ORG | LOC | OTHER."""
    infobox_img: Optional[str] = None
    """Special:FilePath URL of the image shown in the en.wikipedia infobox
    (PageImages-derived); None when absent."""
    url_wikipedia: str
    page_imglist: list[Image] = Field(default_factory=list)
    """All images appearing in the entity's Wikipedia article body."""
    commons_imglist: list[Image] = Field(default_factory=list)
    """Second image source (S4b): all images on the entity's Commons gallery
    page (the commonswiki sitelink). Empty when the entity has no gallery.
    Kept alongside page_imglist so the two sources can be compared."""


class MentionEntry(BaseModel):
    """Top-level dataset entry — one per Wikipedia disambiguation page."""

    mention: str
    """Clean surface form (disambiguation page title with ' (disambiguation)' stripped)."""
    categories: list[str] = Field(default_factory=list)
    """Wikipedia categories (and subcategories) the disambiguation page belongs to,
    as logged by S1 — used for offline category-based filtering in split_gen."""
    ambiguities: list[Entity] = Field(default_factory=list)
    n_entities: int = 0
    n_visual_ambiguities: int = 0
    """Count of *unique* images shared by ≥2 of this mention's candidate entities."""
    n_grounded_entities: int = 0
    """Count of entities that have ≥1 valid external anchor image:
    an image that is (a) not the entity's infobox, (b) not shared with any other
    candidate in this mention, and (c) shared with at least one non-candidate entity.
    When n_grounded_entities == n_entities, the multimodal invariant holds:
    neither modality alone can disambiguate, but together they can."""

    @model_validator(mode="after")
    def _sync_counts(self) -> "MentionEntry":
        if self.n_entities == 0 and self.ambiguities:
            object.__setattr__(self, "n_entities", len(self.ambiguities))
        return self


class Dataset:
    """
    Container for the assembled dataset.

    Attributes:
        entries: List of MentionEntry objects (one per disambiguation page).
        kb: Flat dict {QID → Entity} with all collected entities deduplicated.
    """

    def __init__(self, entries: list[MentionEntry], kb: dict[str, Entity]) -> None:
        self.entries = entries
        self.kb = kb

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Write dataset.jsonl, streaming one entry per line (atomic)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(e.model_dump_json() + "\n")
        tmp.rename(path)

    def save_kb(self, path: str | Path) -> None:
        """Write entity_kb.jsonl, one entity per line (atomic).

        Each line is a self-contained JSON object — qid is embedded so the file
        can be read without any surrounding dict structure.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for e in self.kb.values():
                f.write(e.model_dump_json() + "\n")
        tmp.rename(path)

    @classmethod
    def load(cls, path: str | Path, kb_path: str | Path | None = None) -> "Dataset":
        """
        Load from dataset.jsonl (or dataset.json for legacy files).

        Prefers .jsonl — line-by-line parsing avoids loading a multi-GB JSON
        array as a single string. Falls back to json.loads on .json for compat.
        entity_kb.jsonl (new) and entity_kb.json (legacy) are both supported.
        """
        path = Path(path)
        jsonl_path = path if path.suffix == ".jsonl" else path.with_suffix(".jsonl")

        if jsonl_path.exists():
            entries: list[MentionEntry] = []
            with jsonl_path.open("rb") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(MentionEntry.model_validate(_loads(line)))
        else:
            entries = [
                MentionEntry.model_validate(e)
                for e in _loads(path.read_bytes())
            ]

        kb: dict[str, Entity] = {}
        if kb_path is not None:
            kb_path = Path(kb_path)
            # prefer .jsonl (new format), fall back to .json dict (legacy)
            jsonl_kb = kb_path if kb_path.suffix == ".jsonl" else kb_path.with_suffix(".jsonl")
            if jsonl_kb.exists():
                with jsonl_kb.open("rb") as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            e = Entity.model_validate(_loads(line))
                            kb[e.qid] = e
            else:
                for qid, raw_e in _loads(kb_path.read_bytes()).items():
                    kb[qid] = Entity.model_validate(raw_e)

        return cls(entries=entries, kb=kb)

    # ------------------------------------------------------------------
    # Downstream operations
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.entries)

    def __repr__(self) -> str:
        n_qids = len(self.kb) or sum(len(e.ambiguities) for e in self.entries)
        return f"Dataset(mentions={len(self.entries)}, entities≈{n_qids})"

    def to_dict(self) -> list[dict[str, Any]]:
        return [e.model_dump() for e in self.entries]
