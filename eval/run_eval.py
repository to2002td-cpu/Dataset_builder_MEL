#!/usr/bin/env python3
"""
Evaluate a model on a MEL split.

Reads an experiment config from configs/eval/ and writes results to
eval/results/<config_stem>/: predictions.jsonl, metrics.json, and a copy
of the config alongside the git commit hash for provenance.

Usage:
    python eval/run_eval.py --config configs/eval/qwen_contrastive_10_text.yaml
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

_RESULTS_ROOT = Path(__file__).resolve().parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="experiment config (configs/eval/<model>_<prompt>_<split>.yaml)")
    args = ap.parse_args()

    if not args.config.exists():
        raise SystemExit(f"Config not found: {args.config}")
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    results_dir = _RESULTS_ROOT / args.config.stem
    print(f"Model:   {cfg['model']['name']}")
    print(f"Split:   {cfg['data']['instances']}")
    print(f"Results: {results_dir}")

    # TODO: load split, build prompts (cfg["prompt"]["template"]), run model,
    # write predictions.jsonl + metrics.json + config copy to results_dir.
    raise NotImplementedError("Evaluation loop not implemented yet.")


if __name__ == "__main__":
    main()
