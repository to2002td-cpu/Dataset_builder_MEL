"""Pydantic v2 data models for the wikiambig dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

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
        """Write dataset.json and dataset.jsonl, streaming entry-by-entry."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.entries)

        # dataset.jsonl — one JSON object per line
        jsonl_path = path.with_suffix(".jsonl")
        tmp_jsonl = jsonl_path.with_suffix(".tmp")
        with tmp_jsonl.open("w", encoding="utf-8") as f:
            for e in self.entries:
                f.write(e.model_dump_json() + "\n")
        tmp_jsonl.rename(jsonl_path)

        # dataset.json — valid JSON array written entry-by-entry to avoid a
        # full-dataset intermediate dict + json.dumps string in RAM
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write("[\n")
            for i, e in enumerate(self.entries):
                comma = "," if i < n - 1 else ""
                f.write(e.model_dump_json() + comma + "\n")
            f.write("]\n")
        tmp.rename(path)

    def save_kb(self, path: str | Path) -> None:
        """Write entity_kb.json keyed by QID, streaming entry-by-entry."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        n = len(self.kb)
        tmp = path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            f.write("{\n")
            for i, (qid, e) in enumerate(self.kb.items()):
                comma = "," if i < n - 1 else ""
                f.write(f"  {json.dumps(qid)}: {e.model_dump_json()}{comma}\n")
            f.write("}\n")
        tmp.rename(path)

    @classmethod
    def load(cls, path: str | Path, kb_path: str | Path | None = None) -> "Dataset":
        """
        Load from dataset.json or dataset.jsonl (and optionally entity_kb.json).

        Prefers the .jsonl sidecar when it exists — line-by-line parsing avoids
        loading a multi-GB JSON array as a single string. Falls back to
        json.loads on the .json file for compatibility.
        """
        path = Path(path)
        jsonl_path = path.with_suffix(".jsonl")

        if jsonl_path.exists():
            entries: list[MentionEntry] = []
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        entries.append(MentionEntry.model_validate_json(line))
        else:
            raw = json.loads(path.read_text(encoding="utf-8"))
            entries = [MentionEntry.model_validate(e) for e in raw]

        kb: dict[str, Entity] = {}
        if kb_path is not None:
            kb_path = Path(kb_path)
            with kb_path.open("r", encoding="utf-8") as f:
                kb_raw = json.load(f)
            kb = {qid: Entity.model_validate(e) for qid, e in kb_raw.items()}

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
