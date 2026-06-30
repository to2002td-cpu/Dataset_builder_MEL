"""Load and normalise annotation data from user_state.json files.

Each annotator's folder must contain a single user_state.json produced by the
annotation tool.  The tool appends one entry to annotation_changes for every
keystroke, so only the *last* entry per schema per instance is meaningful.
"""

import json
from pathlib import Path


# Raw label values stored by the tool → human-readable display labels.
LABEL_MAP: dict[str, str] = {"y": "YES", "n": "NO", "u": "UNCERTAIN"}


def load_annotators(annotations_dir: Path) -> dict[str, "AnnotatorData"]:
    """Return one AnnotatorData per annotator found in *annotations_dir*.

    Each sub-folder is expected to contain a user_state.json file.
    Sub-folders without that file are silently skipped.
    """
    annotators: dict[str, AnnotatorData] = {}

    for annotator_dir in sorted(annotations_dir.iterdir()):
        state_file = annotator_dir / "user_state.json"
        if not state_file.exists():
            continue

        with open(state_file, encoding="utf-8") as f:
            raw = json.load(f)

        user_id = raw.get("user_id", annotator_dir.name)
        annotators[user_id] = AnnotatorData.from_user_state(user_id, raw)

    return annotators


class AnnotatorData:
    """Holds the cleaned annotation data for one annotator."""

    def __init__(
        self,
        user_id: str,
        labels: dict[str, str | None],
        comments: dict[str, str],
    ) -> None:
        self.user_id: str = user_id
        self.short_name: str = _short_name(user_id)
        self.labels: dict[str, str | None] = labels          # {instance_id: "YES"|"NO"|"UNCERTAIN"|None}
        self.comments: dict[str, str] = comments             # {instance_id: comment_text}
        self.label_counts: dict[str, int] = {
            label: sum(1 for v in labels.values() if v == label)
            for label in ("YES", "NO", "UNCERTAIN")
        }

    @classmethod
    def from_user_state(cls, user_id: str, user_state: dict) -> "AnnotatorData":
        behavioral_data = user_state.get("instance_id_to_behavioral_data", {})
        labels   = {iid: _last_label(data)   for iid, data in behavioral_data.items()}
        comments = {iid: text for iid, data in behavioral_data.items()
                    if (text := _last_comment(data))}
        return cls(user_id, labels, comments)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _last_label(instance_data: dict) -> str | None:
    """Return the final entity_is_illustrated label, or None if unannotated."""
    label = None
    for change in instance_data.get("annotation_changes", []):
        if change.get("schema_name") == "entity_is_illustrated":
            label = LABEL_MAP.get(change["new_value"], change["new_value"])
    return label


def _last_comment(instance_data: dict) -> str | None:
    """Return the final non-empty comment, or None."""
    comment = None
    for change in instance_data.get("annotation_changes", []):
        if change.get("schema_name") == "comment" and change.get("new_value"):
            comment = change["new_value"].strip()
    return comment or None


def _short_name(user_id: str) -> str:
    """Derive a short display name from an email or dot-separated identifier.

    Examples:
        "thomas.derrien@irisa.fr" → "derrien"
        "laurent.amsaleg"         → "laurent.a"
        "Louis"                   → "Louis"
    """
    if "@" in user_id:
        return user_id.split("@")[0].split(".")[-1]
    if "." in user_id:
        parts = user_id.split(".")
        return parts[0] + "." + parts[-1][0]
    return user_id
