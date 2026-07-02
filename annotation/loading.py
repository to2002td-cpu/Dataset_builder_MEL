"""
Load annotation campaign data (KB, items, annotator labels, behavioral logs).

All I/O for the annotation analysis lives here; the returned objects are plain
dicts and pandas DataFrames consumed by stats.py, plots.py and display.py.
Annotator labels are read generically from ``<annot_dir>/<user>/user_state.json``
(Potato exports): add a new annotator folder and every analysis recomputes.

Usage (from the analysis notebook, cwd = annotation/):
    from loading import (DEFAULT_ANNOT_DIR, DEFAULT_DATA_DIR, build_master,
                         load_annotations, load_decision_times, load_items,
                         load_kb_maps)

    kb = load_kb_maps(DEFAULT_DATA_DIR / "kb.jsonl")
    items = load_items(DEFAULT_DATA_DIR, kb)
    annotations = load_annotations(DEFAULT_ANNOT_DIR)
    master = build_master(items, annotations)
    times = load_decision_times(DEFAULT_ANNOT_DIR, items)
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = _ROOT / "output" / "annot_output" / "data"
DEFAULT_ANNOT_DIR = _ROOT / "output" / "annot_output" / "annot"

LABELS = ("YES", "NO", "UNCERTAIN")


def load_kb_maps(kb_path: Path) -> dict[str, dict]:
    """Stream kb.jsonl into {qid: {type, name, desc, infobox_img}} (memory-light)."""
    kb: dict[str, dict] = {}
    with open(kb_path, encoding="utf-8") as f:
        for line in f:
            o = json.loads(line)
            kb[o["qid"]] = {
                "type": o["type"],
                "name": o.get("name"),
                "desc": o.get("desc"),
                "infobox_img": o.get("infobox_img"),
            }
    return kb


def load_items(data_dir: Path, kb: dict[str, dict]) -> dict[str, dict]:
    """Index annotation items by instance id, joining items.jsonl (what the
    annotator saw) with instances.jsonl (candidate pools) line by line."""
    by_id: dict[str, dict] = {}
    with open(data_dir / "items.jsonl", encoding="utf-8") as f_items, \
         open(data_dir / "instances.jsonl", encoding="utf-8") as f_insts:
        for line_item, line_inst in zip(f_items, f_insts, strict=True):
            it = json.loads(line_item)
            ins = json.loads(line_inst)
            item_id = it.get("id")
            if item_id is None:
                continue
            entity = kb.get(it.get("qid"), {})
            by_id[item_id] = {
                "instance_id": item_id,
                "mention": it.get("mention"),
                "entity_name": it.get("entity_name"),
                "qid": it.get("qid"),
                "category": entity.get("type"),
                "desc": entity.get("desc") or it.get("entity_desc"),
                "image_url": it.get("image_url"),
                "n_text_candidates": len(ins.get("text_candidates", [])),
                "n_visual_candidates": len(ins.get("visual_candidates", [])),
                "n_alternatives": len(it.get("alternatives", [])),
                "candidate_qids": list(dict.fromkeys(
                    ins.get("text_candidates", []) + ins.get("visual_candidates", [])
                )),
            }
    return by_id


def load_annotations(annot_dir: Path) -> dict[str, dict[str, str]]:
    """Read {annotator: {instance_id: label}} from every user_state.json."""
    out: dict[str, dict[str, str]] = {}
    for path in sorted(annot_dir.glob("*/user_state.json")):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        user = d.get("user_id") or path.parent.name
        labels: dict[str, str] = {}
        for iid, val in d["instance_id_to_label_to_value"].items():
            with contextlib.suppress(IndexError, KeyError, TypeError):
                labels[iid] = val[0][0]["name"]  # 'YES' / 'NO' / 'UNCERTAIN'
        out[user] = labels
    return out


def build_master(items: dict[str, dict],
                 annotations: dict[str, dict[str, str]]) -> pd.DataFrame:
    """One row per annotated instance: item fields, one label column per
    annotator, plus vote aggregates (n_yes/n_no, majority, disagree, …)."""
    annotators = sorted(annotations)
    all_ids = sorted(set().union(*(set(v) for v in annotations.values())))

    rows = []
    for iid in all_ids:
        r = dict(items[iid])
        for u in annotators:
            r[u] = annotations[u].get(iid, np.nan)
        votes = [r[u] for u in annotators if isinstance(r[u], str)]
        r["n_votes"] = len(votes)
        r["n_yes"] = votes.count("YES")
        r["n_no"] = votes.count("NO")
        r["is_multi"] = r["n_votes"] >= 2
        r["disagree"] = r["n_yes"] > 0 and r["n_no"] > 0
        if r["n_yes"] > r["n_no"]:
            r["majority"] = "YES"
        elif r["n_no"] > r["n_yes"]:
            r["majority"] = "NO"
        else:
            r["majority"] = "TIE"
        r["unanimous"] = not r["disagree"] and r["n_votes"] >= 1
        r["margin"] = abs(r["n_yes"] - r["n_no"])
        rows.append(r)
    return pd.DataFrame(rows)


def _decision_time(behavioral: dict) -> float:
    """Seconds from the first instance_load to the first annotation_change."""
    inter = behavioral.get("interactions", [])
    loads = [e["timestamp"] for e in inter
             if e["event_type"] == "navigation" and e.get("target") == "instance_load"]
    changes = [e["timestamp"] for e in inter if e["event_type"] == "annotation_change"]
    if not loads or not changes:
        return np.nan
    t0 = min(loads)
    after = [c for c in changes if c >= t0]
    return (min(after) - t0) if after else np.nan


def load_decision_times(annot_dir: Path, items: dict[str, dict]) -> pd.DataFrame:
    """Per (annotator, instance) decision time from the behavioral logs.
    Columns: annotator, instance_id, category, t_raw (seconds, uncapped)."""
    rows = []
    for path in sorted(annot_dir.glob("*/user_state.json")):
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
        user = d.get("user_id") or path.parent.name
        for iid, beh in d["instance_id_to_behavioral_data"].items():
            rows.append({
                "annotator": user,
                "instance_id": iid,
                "category": items.get(iid, {}).get("category"),
                "t_raw": _decision_time(beh),
            })
    return pd.DataFrame(rows).dropna(subset=["t_raw"]).reset_index(drop=True)
