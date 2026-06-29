"""Pipeline orchestration — run one or more stages by name.

Stage overview:
  S1  disam_index      — Wikipedia disambiguation page index
  S2  entity_links     — entity QIDs per disambiguation page (wikitext parse)
  S3  wikidata         — entity data + types via one combined SPARQL query (Wikidata)
  S4  wikipedia        — intro paragraphs + image lists in one API call (Wikipedia)
  S4b commons          — second image source: files on the entity's Commons gallery page
  S5  image_data       — image URL + used_by QIDs (Wikipedia imageinfo + fileusage)
  S6  visual_entity_data — Wikidata enrichment for used_by QIDs not in S3
  S7  assemble         — offline join → dataset.jsonl + entity_kb.jsonl
"""

from __future__ import annotations

import logging

from wikiambig.config import PipelineConfig
from wikiambig.pipeline import (
    s1_disam_index,
    s2_wikitext,
    s3_wikidata,
    s4_wikipedia,
    s4b_commons,
    s5_image_data,
    s6_visual_entity_data,
    s7_assemble,
)

logger = logging.getLogger(__name__)

STAGES: dict[str, object] = {
    "s1": s1_disam_index,
    "s2": s2_wikitext,
    "s3": s3_wikidata,
    "s4": s4_wikipedia,
    "s4b": s4b_commons,
    "s5": s5_image_data,
    "s6": s6_visual_entity_data,
    "s7": s7_assemble,
}

# s4 / s4b run twice: once for S3's candidates, and again after S6 adds
# visual-only entities to entity_data.json (so they also get intros, infobox
# images, and Commons gallery images — both stages skip QIDs already in their
# checkpoint, so the second pass only fetches the newly-added ones).
ALL_STAGES = ["s1", "s2", "s3", "s4", "s4b", "s5", "s6", "s4", "s4b", "s7"]


def run_stages(config: PipelineConfig, stages: list[str] | None = None) -> None:
    """Execute the given pipeline stages in dependency order."""
    if stages is None:
        stages = ALL_STAGES

    unknown = [s for s in stages if s not in STAGES]
    if unknown:
        raise ValueError(f"Unknown stage(s): {unknown}. Valid: {ALL_STAGES}")

    for stage in stages:
        logger.info("=== Running %s ===", stage.upper())
        STAGES[stage].run(config)  # type: ignore[union-attr]
        logger.info("=== %s complete ===", stage.upper())
