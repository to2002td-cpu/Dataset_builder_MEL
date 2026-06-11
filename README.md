# Multimodal Named Entity Disambiguation Dataset Builder

> A large-scale pipeline that automatically builds a **Multimodal Entity Linking (MEL)** benchmark from Wikipedia and Wikidata, combining textual disambiguation with visual evidence.

---

## Overview

The key insight driving the dataset design is a **bipartite grounding criterion**: an instance is valid only when the correct answer sits at the intersection of two independent candidate sets: one derived from the textual surface form, the other from which Wikipedia articles share the query image. Neither modality alone is sufficient to resolve the ambiguity; combining both is required.

The pipeline is fully automatic, reproducible, and resumable. It queries the live Wikipedia and Wikidata APIs with no dependency on offline dumps.

---

## Task Definition

Each instance presents a model with:

| Field | Description |
|---|---|
| `mention` | Ambiguous surface form (e.g., `"John Smith"`) |
| `image` | A JPEG photograph appearing in ≥ 2 Wikipedia articles |
| `text_candidates` | QIDs of entities sharing this surface form |
| `visual_candidates` | QIDs of entities whose Wikipedia articles embed this image |
| `answer` | The unique QID at the intersection `text_candidates ∩ visual_candidates` |

Formally, the task is: given `(mention, image)`, retrieve `answer` from the knowledge base.

```json
{
  "mention": "Alexandra Bridge",
  "image": {
    "url": "https://upload.wikimedia.org/wikipedia/commons/8/83/State_Highway_8_bridge_Alexandra%2C_New_Zealand.jpg",
    "n_used_by": 2,
    "width": 2048,
    "height": 1536,
    "mime": "image/jpeg",
    "license": "CC BY-SA 3.0"
  },
  "answer": "Q19875502",
  "text_candidates": ["Q19875502", "Q22329496", "Q4720595"],
  "visual_candidates": ["Q19875502", "Q2058593"]
}
```

---

## Dataset Statistics

The following figures are from the June 2026 snapshot.

### Raw pipeline output (`output/`)

| Metric | Count |
|---|---|
| Disambiguation mentions | 138,306 |
| KB entities | 2,198,397 |
| Unique images | 1,094,733 |

### Final MEL dataset (`output/final/`)

| Metric | Count |
|---|---|
| Instances | 24,212 |
| KB entities (filtered) | 416,238 |

**Entity type breakdown (KB):** persons (PERS, 62%), organisations (ORG, 22%), locations (LOC, 16%).

**Mention filters applied:** longer than 2 characters · contains at least one alphabetic character · not an ordinal-numbered military unit designator (e.g. "1st Cavalry", "55th Regiment of Foot" — generic, visually-indistinguishable formations; peerage titles like "1st Earl Temple" are exempt).

**Entity filters applied:** must have an intro paragraph or an infobox image · must be typed PERS, ORG, or LOC · text candidate pool size ≤ 50 (drops generic-fragment mentions, e.g. "Cerro").

**Image filters applied:** JPEG only · minimum 100 × 100 px · article reuse 2 ≤ `n_used_by` ≤ 10 · shared across ≥ 2 KB entities · not itself another entity's infobox image.

**Instance filters applied:** the image's visual candidate set (KB entities sharing that image) must have ≥ 2 members · the intersection of text and visual candidates must contain exactly one entity (the answer) · each `(mention, answer)` pair is kept only once.

---

## Pipeline Architecture

The pipeline runs in seven sequential stages. Each stage writes its output atomically to `data/` before the next stage reads it; any stage can be re-run independently.

```
Wikipedia categories
        │
        ▼
   S1 — Disambiguation index        disam_index.jsonl
        │
        ▼
   S2 — Wikitext entity links       entity_links.jsonl
        │
        ▼
   S3 — Wikidata enrichment         entity_data.json · entity_types.json
   (combined SPARQL: name, desc,
    Wikipedia URL, infobox image,
    coarse type PERS/ORG/LOC)
        │
        ▼
   S4 — Wikipedia intro + images    entity_intros.json · image_lists.json
        │
        ▼
   S5 — Image metadata              image_data.json
   (URL, dimensions, license,
    used_by QID list)
        │
        ▼
   S6 — Visual entity enrichment    (extends entity_data for used_by QIDs)
        │
        ▼
   S7 — Assembly (offline)          dataset.jsonl · entity_kb.json · manifest.json
```

After S7, run `scripts/make_dataset.py` to apply quality filters and produce the final `instances.jsonl` and `kb.jsonl`.

---

## Installation

Requires Python 3.10+.

```bash
pip install -e .
# With optional analysis dependencies (stats, plots):
pip install -e ".[full]"
```

---

## Quick Start

### 1. Run the full pipeline

```bash
wikiambig scrape
```

This runs all stages S1 → S7 using `configs/config.yaml`. Intermediate outputs are written to `data/` and the assembled dataset to `output/`.

### 2. Run specific stages

```bash
# Run only stages 3 and 4
wikiambig scrape --stages s3,s4

# Re-assemble from existing intermediate files
wikiambig build
```

### 3. Build the final MEL dataset

```bash
python scripts/make_dataset.py output/dataset.jsonl output/final/
```

### 4. Inspect statistics

```bash
python scripts/stats.py output/final/instances.jsonl
python scripts/stats.py output/final/instances.jsonl --out figures/stats.pdf
```

### 5. Browse instances visually

```bash
python scripts/view_split.py output/final/instances.jsonl --open
```

This generates a self-contained HTML viewer showing each disambiguation task with its image, candidate entities, and highlighted answer.

---

## Configuration

All parameters live in `configs/config.yaml`:

```yaml
data_dir: ./data
output_dir: ./output

disam_categories:
  - "Category:Human name disambiguation pages"
  - "Category:Place name disambiguation pages"
  # ... (10 categories total)

n_workers: 10              # parallel API threads
api_batch_size: 50         # titles per Wikipedia REST call
entity_data_batch_size: 200  # QIDs per Wikidata SPARQL query
wikipedia_rate_limit: 0.5  # seconds between requests
wikidata_rate_limit: 0.5
save_every: 200
```

Pass a custom config at runtime:

```bash
wikiambig scrape --config configs/my_config.yaml
```

---

## Output File Reference

| File | Description |
|---|---|
| `data/disam_index.jsonl` | Wikipedia disambiguation page titles and URLs |
| `data/entity_links.jsonl` | Per-mention QID lists (raw links from wikitext) |
| `data/entity_data.json` | Wikidata name, description, Wikipedia URL, infobox image per QID |
| `data/entity_types.json` | Coarse entity type (`PERS` / `ORG` / `LOC` / `null`) per QID |
| `data/image_lists.json` | Per-entity list of image filenames on their Wikipedia article |
| `data/image_data.json` | Per-image URL, dimensions, MIME, license, `used_by` QID list |
| `output/dataset.jsonl` | Assembled dataset (one `MentionEntry` per line) |
| `output/entity_kb.json` | Full entity knowledge base keyed by QID |
| `output/manifest.json` | Pipeline version, assembly timestamp, and headline counts |
| `output/final/instances.jsonl` | Final filtered MEL instances |
| `output/final/kb.jsonl` | Final filtered entity KB |

---

## Reproducibility

Wikipedia and Wikidata are continuously-edited live sources with no fixed dump version. `output/manifest.json` records the exact UTC timestamp at which S7 was run — this is the only stable provenance record for a given dataset snapshot. Re-running the pipeline on a different date will yield a different (typically larger) result.

---