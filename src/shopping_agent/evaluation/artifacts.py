"""Shared evaluation artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path


class ArtifactError(ValueError):
    """Raised for malformed, duplicate, or unsafe dataset artifacts."""


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSONL file into a list of objects; blank lines are ignored."""
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
