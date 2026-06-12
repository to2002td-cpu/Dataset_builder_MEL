#!/usr/bin/env python3
"""
Evaluate a model on a MEL split.

Reads an experiment config from configs/eval/ and writes results to
eval/results/<config_stem>/: predictions.jsonl, metrics.json, and a copy
of the config alongside the git commit hash for provenance.

Usage:
    python eval/run_eval.py --config configs/eval/qwen_contrastive_10_text.yaml
    python eval/run_eval.py --config ... --limit 2   # debug: cap instance count
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path

import yaml
from tqdm import tqdm

from models.model import load_model
from utils import (compute_metrics, compute_ranking_metrics, fetch_image,
                   format_prompt, parse_number, parse_ranking)

_ROOT = Path(__file__).resolve().parent.parent
_RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_template(name: str) -> dict:
    """Load eval/prompts/<name>.yaml (a path is also accepted)."""
    path = Path(name) if "/" in name else \
        Path(__file__).resolve().parent / "prompts" / f"{name}.yaml"
    if not path.exists():
        raise SystemExit(f"Prompt template not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="experiment config (configs/eval/<model>_<prompt>_<split>.yaml)")
    ap.add_argument("--limit", type=int, default=None,
                    help="debug only: cap the number of instances (recorded in metrics)")
    args = ap.parse_args()

    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    data_cfg, prompt_cfg = cfg["data"], cfg["prompt"]
    rng = random.Random(cfg.get("seed", 0))

    # ── Load split ────────────────────────────────────────────────────────────
    instances = list(_iter_jsonl(_ROOT / data_cfg["instances"]))
    kb = {e["qid"]: e for e in _iter_jsonl(_ROOT / data_cfg["kb"])}
    n_samples = data_cfg.get("n_samples")
    if n_samples and n_samples < len(instances):
        instances = rng.sample(instances, n_samples)
    if args.limit:
        instances = instances[: args.limit]

    template = _load_template(prompt_cfg["template"])
    mode = template.get("mode", "contrastive")
    image_max_side = data_cfg.get("image_max_side", 1280)

    model = load_model(cfg["model"])

    # ── Results dir + provenance ──────────────────────────────────────────────
    results_dir = _RESULTS_ROOT / args.config.stem
    results_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(args.config, results_dir / "config.yaml")
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "device": model.device if hasattr(model, "device") else None,
        "template": prompt_cfg["template"],
        "limit": args.limit,
    }
    print(f"Model:   {cfg['model']['name']}   Split: {data_cfg['instances']}")
    print(f"Results: {results_dir}   ({len(instances)} instances)")

    # ── Evaluation loop ───────────────────────────────────────────────────────
    preds, labels, pool_sizes, gold_ranks = [], [], [], []
    n_skipped = 0
    with (results_dir / "predictions.jsonl").open("w", encoding="utf-8") as out:
        for inst in tqdm(instances, desc="Evaluating"):
            record = {"mention": inst["mention"], "answer": inst["answer"],
                      "image_url": inst["image"]["url"]}

            query_img = fetch_image(inst["image"]["url"], max_side=image_max_side)
            if query_img is None:
                n_skipped += 1
                record["error"] = "query image download failed"
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            if mode in ("contrastive", "ranking"):
                candidates = [kb[q] for q in inst["text_candidates"] if q in kb]
                parts, label, order = format_prompt(
                    template, inst["mention"], candidates, inst["answer"],
                    prompt_cfg, image_max_side=image_max_side, rng=rng,
                )
                parts.insert(0, ("image", query_img))
                response = model.generate(parts)
                record.update({"options": order, "label": label, "response": response})

            if mode == "contrastive":
                pred = parse_number(response)
                pred_qid = order[pred - 1] if pred and 1 <= pred <= len(order) else None
                record.update({"pred_number": pred, "pred_qid": pred_qid,
                               "correct": pred == label})
                preds.append(pred)
                labels.append(label)
                pool_sizes.append(len(order))
            elif mode == "ranking":
                ranking = parse_ranking(response)
                gold_rank = ranking.index(label) + 1 if ranking and label in ranking \
                    else None
                record.update({"ranking": ranking, "gold_rank": gold_rank})
                gold_ranks.append(gold_rank)
            else:  # free: scored by gold-name match in the response
                parts = [("image", query_img),
                         ("text", template["header"].format(mention=inst["mention"]))]
                response = model.generate(parts)
                gold_name = kb.get(inst["answer"], {}).get("name", "")
                record.update({"response": response, "gold_name": gold_name,
                               "correct": bool(gold_name)
                                          and gold_name.lower() in response.lower()})
                preds.append(1 if record["correct"] else 0)
                labels.append(1)

            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ── Metrics ───────────────────────────────────────────────────────────────
    if mode == "ranking":
        metrics = compute_ranking_metrics(gold_ranks)
    else:
        metrics = compute_metrics(preds, labels, pool_sizes or None)
    metrics["n_skipped_no_image"] = n_skipped
    metrics["meta"] = meta
    (results_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if mode == "ranking":
        print(f"\nMRR: {metrics['mrr']:.3f}  hit@1: {metrics['hit@1']:.3f}  "
              f"hit@3: {metrics['hit@3']:.3f}  ndcg@5: {metrics['ndcg@5']:.3f}  "
              f"({metrics['n_unparsed']} unparsed, {n_skipped} skipped)")
    else:
        print(f"\nAccuracy: {metrics['accuracy']:.3f}  "
              f"({metrics['n_correct']}/{metrics['n']}, "
              f"{metrics['n_unparsed']} unparsed, {n_skipped} skipped)")
        if "random_baseline" in metrics:
            print(f"Random baseline: {metrics['random_baseline']:.3f}")
    print(f"Results → {results_dir}")


if __name__ == "__main__":
    main()
