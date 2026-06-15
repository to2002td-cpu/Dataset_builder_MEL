"""Pipeline configuration via Pydantic settings + YAML file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import field_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource

class PipelineConfig(BaseSettings):
    """
    All pipeline settings in one place.

    Loaded in priority order (highest to lowest):
      1. Environment variables prefixed with WIKIAMBIG_
      2. YAML config file (path set via --config CLI flag or WIKIAMBIG_CONFIG_FILE env var)
      3. Defaults below
    """

    # Paths
    data_dir: Path = Path("./output/scrape_data")
    output_dir: Path = Path("./output/raw_dataset")

    # Rate limiting (seconds between batch calls, per worker)
    wikipedia_rate_limit: float = 1.0
    wikidata_rate_limit: float = 1.0

    # Parallelism
    n_workers: int = 5

    # Batch sizes
    api_batch_size: int = 50
    entity_data_batch_size: int = 200

    # Checkpointing
    save_every: int = 200

    log_level: str = "INFO"

    model_config = {"env_prefix": "WIKIAMBIG_", "extra": "ignore"}

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v = v.upper()
        if v not in allowed:
            raise ValueError(f"log_level must be one of {allowed}")
        return v

    # Derived path helpers
    def stage_path(self, filename: str) -> Path:
        """Return a path under data_dir for an intermediate stage file."""
        return self.data_dir / filename

    def output_path(self, filename: str) -> Path:
        return self.output_dir / filename

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls(**raw)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        **kwargs: Any,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Only use init kwargs and env vars; skip dotenv / secrets / any future sources.
        return (init_settings, env_settings)
