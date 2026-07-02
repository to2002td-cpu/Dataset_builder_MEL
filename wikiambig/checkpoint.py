"""Resumable run state: tracks which item IDs a stage has already processed."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock


class Checkpoint:
    """
    Thread-safe checkpoint for a single pipeline stage.

    Persists a set of completed item IDs (strings) to a JSON file.
    On resume, any ID already in the checkpoint is skipped by the stage.

    Usage::

        cp = Checkpoint(config.stage_path("s3.checkpoint.json"))
        for qid in pending:
            if cp.contains(qid):
                continue
            result = process(qid)
            cp.mark_done(qid)
    """

    def __init__(self, path: str | Path, flush_every: int = 100) -> None:
        self._path = Path(path)
        self._flush_every = flush_every
        self._lock = Lock()
        self._completed: set[str] = self._load()
        self._dirty = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def contains(self, item_id: str) -> bool:
        with self._lock:
            return item_id in self._completed

    def mark_done(self, item_id: str) -> None:
        with self._lock:
            self._completed.add(item_id)
            self._dirty += 1
            if self._dirty >= self._flush_every:
                self._write()
                self._dirty = 0

    def mark_done_batch(self, item_ids: list[str]) -> None:
        with self._lock:
            self._completed.update(item_ids)
            self._dirty += len(item_ids)
            if self._dirty >= self._flush_every:
                self._write()
                self._dirty = 0

    def flush(self) -> None:
        with self._lock:
            if self._dirty > 0:
                self._write()
                self._dirty = 0

    @property
    def n_done(self) -> int:
        with self._lock:
            return len(self._completed)

    def pending(self, all_ids: list[str]) -> list[str]:
        """Return IDs not yet in the checkpoint."""
        with self._lock:
            done = self._completed
        return [i for i in all_ids if i not in done]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> set[str]:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                return set(data.get("completed", []))
            except (json.JSONDecodeError, KeyError):
                return set()
        return set()

    def _write(self) -> None:
        """Atomic write (must be called under self._lock)."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "completed": sorted(self._completed),
            "n_done": len(self._completed),
            "last_updated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        tmp = str(self._path) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, str(self._path))

    def __enter__(self) -> Checkpoint:
        return self

    def __exit__(self, *_: object) -> None:
        self.flush()
