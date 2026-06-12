"""
Abstract model interface for evaluation.

Every model wrapper implements `Model` and is instantiated through
`load_model()` from the `model:` section of the eval config. A prompt is a
list of (kind, value) parts, in order:
  ("text", str)         — a text segment
  ("image", Path)       — a local image file (already downloaded/cached)
so multimodal prompts can interleave candidate text and candidate images.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

Part = tuple[str, "str | Path"]


class Model(ABC):
    """Base class for all evaluated models; cfg is the config `model:` section."""

    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.name = cfg["name"]

    @abstractmethod
    def generate(self, parts: list[Part]) -> str:
        """Run one multimodal prompt and return the raw text response."""


def load_model(cfg: dict) -> Model:
    """Instantiate the wrapper matching cfg['name'] (prefix-based registry)."""
    name = cfg["name"].lower()
    if name.startswith("qwen3vl") or name.startswith("qwen3-vl"):
        from models.qwen3 import Qwen3VL

        return Qwen3VL(cfg)
    raise SystemExit(f"No model wrapper registered for: {cfg['name']}")
