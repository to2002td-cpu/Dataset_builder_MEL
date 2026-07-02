"""
CLI entry-point for the wikiambig pipeline.

Commands:
  scrape   Run one or more pipeline stages (S1–S7).
  build    Run the offline assembly stage (S7) only.

Downstream stages live in their own top-level directories:
  split_gen/make_split.py   — create a filtered split from dataset.jsonl
  split_gen/view_split.py   — browse a split as a self-contained HTML page
  fig_gen/stats.py          — compute split statistics and figures
  eval/run_eval.py          — evaluate a model on a split
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import typer

if TYPE_CHECKING:
    from wikiambig.config import PipelineConfig

app = typer.Typer(
    name="wikiambig",
    help="Wikipedia-based multimodal named-entity disambiguation dataset builder.",
    no_args_is_help=True,
)


def _load_config(config_path: Path | None) -> PipelineConfig:
    from wikiambig.config import PipelineConfig

    if config_path:
        return PipelineConfig.from_yaml(config_path)
    return PipelineConfig()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# scrape
# ---------------------------------------------------------------------------

@app.command()
def scrape(
    config: Path | None = typer.Option(
        None, "--config", "-c", help="Path to config.yaml", exists=True
    ),
    stages: str | None = typer.Option(
        None,
        "--stages",
        "-s",
        help="Comma-separated stage list, e.g. s1,s2,s3. Default: all.",
    ),
    data_dir: Path | None = typer.Option(None, help="Override data_dir from config."),
    output_dir: Path | None = typer.Option(None, help="Override output_dir from config."),
) -> None:
    """Run pipeline stages (default: full pipeline s1→s7)."""
    cfg = _load_config(config)
    if data_dir:
        cfg = cfg.model_copy(update={"data_dir": data_dir})
    if output_dir:
        cfg = cfg.model_copy(update={"output_dir": output_dir})

    _setup_logging(cfg.log_level)
    logger = logging.getLogger("wikiambig.cli")

    stage_list = [s.strip().lower() for s in stages.split(",")] if stages else None

    from wikiambig.pipeline import run_stages

    try:
        run_stages(cfg, stage_list)
    except Exception as exc:
        logger.error("Pipeline failed: %s", exc)
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

@app.command()
def build(
    config: Path | None = typer.Option(None, "--config", "-c", exists=True),
    data_dir: Path | None = typer.Option(None),
    output_dir: Path | None = typer.Option(None),
) -> None:
    """Run the offline assembly stage (S7) only."""
    cfg = _load_config(config)
    if data_dir:
        cfg = cfg.model_copy(update={"data_dir": data_dir})
    if output_dir:
        cfg = cfg.model_copy(update={"output_dir": output_dir})

    _setup_logging(cfg.log_level)

    from wikiambig.pipeline import run_stages

    run_stages(cfg, ["s7"])


if __name__ == "__main__":
    app()
