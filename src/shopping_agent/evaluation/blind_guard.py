"""Task-ID protection for the frozen blind final test.

The Final-200 evaluation set must never leak into training. Before SFT the training
files are scanned and rejected as soon as a single ``task_id`` overlaps the frozen
blind list shipped in ``shopping_agent.resources``.
"""

from __future__ import annotations

import gzip
import json
from collections.abc import Iterable, Mapping
from importlib.resources import files
from pathlib import Path

from shopping_agent.evaluation.artifacts import ArtifactError

BLIND_TASK_IDS_SCHEMA = "shopping-blind-task-ids-v1"
_TASK_IDS_RESOURCE = "blind_final_task_ids.json"
_RESOURCE_PACKAGE = "shopping_agent.resources"


def _load_final_task_ids() -> set[int]:
    """Return the frozen Final-200 task IDs packaged with this project."""

    try:
        resource = files(_RESOURCE_PACKAGE).joinpath(_TASK_IDS_RESOURCE)
        payload = json.loads(resource.read_text(encoding="utf-8"))
    except (ModuleNotFoundError, OSError, json.JSONDecodeError) as exc:
        raise ArtifactError("cannot read packaged blind task-ID list") from exc
    if not isinstance(payload, Mapping):
        raise ArtifactError("packaged blind task-ID list must be an object")
    if payload.get("schema_version") != BLIND_TASK_IDS_SCHEMA:
        raise ArtifactError("unsupported blind task-ID schema")
    raw_task_ids = payload.get("task_ids")
    if not isinstance(raw_task_ids, list) or not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in raw_task_ids
    ):
        raise ArtifactError("packaged blind task_ids must be integers")
    task_ids = set(raw_task_ids)
    if len(task_ids) != len(raw_task_ids):
        raise ArtifactError("packaged blind task_ids contain duplicates")
    return task_ids


def _row_task_id(row: Mapping) -> int | None:
    value = row.get("task_id")
    if value is None:
        extra = row.get("extra_info")
        if isinstance(extra, Mapping):
            value = extra.get("task_id")
    if value is None:
        normalized = row.get("normalized_trajectory")
        if isinstance(normalized, Mapping):
            value = normalized.get("task_id")
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ArtifactError(f"invalid task_id in {row!r}") from exc


def _jsonl_task_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    opener = gzip.open if path.name.endswith(".gz") else open
    task_ids = set()
    try:
        with opener(path, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise ArtifactError(
                        f"{path}:{line_number}: JSONL row must be an object"
                    )
                task_id = _row_task_id(value)
                if task_id is not None:
                    task_ids.add(task_id)
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"{path}: invalid JSONL during blind guard") from exc
    return task_ids


def guard_blind_final(
    paths: Iterable[Path],
    *,
    allowed: bool,
) -> None:
    """Reject any artifact whose task IDs overlap the frozen blind final test."""

    if allowed:
        return
    final_task_ids = _load_final_task_ids()
    blocked = {}
    for raw_path in paths:
        path = Path(raw_path)
        if not path.is_file():
            continue
        overlap = sorted(_jsonl_task_ids(path) & final_task_ids)
        if overlap:
            blocked[str(path)] = {
                "overlap_count": len(overlap),
                "sample_task_ids": overlap[:10],
            }
    if blocked:
        raise ArtifactError(
            "refusing to consume frozen blind-final tasks without "
            "--allow-blind-final: "
            + json.dumps(blocked, ensure_ascii=False, sort_keys=True)
        )
