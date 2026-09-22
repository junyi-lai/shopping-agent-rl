#!/usr/bin/env python3
"""Validate the SFT data and build the fixed three-stage curriculum manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


SCHEMA_VERSION = "shopping-sft-curriculum-v1"
# 每个难度桶只在自己那一阶段训练一次(基础 → 约束 → 长程策略)。三个阶段互不重叠,
# 合计正好覆盖全部训练 task,所以每道题只被曝光一次,不会产生隐性加权重。
STAGE_CONFIG = {
    "a": {"buckets": ["foundation"], "epochs": 1.0, "learning_rate": 1e-4},
    "b": {"buckets": ["constraints"], "epochs": 1.0, "learning_rate": 7e-5},
    "c": {"buckets": ["strategy"], "epochs": 1.0, "learning_rate": 5e-5},
}
GUARD_MARKERS = ("guard rejected", "action guard", "动作守卫拒绝", "guard_violation")
ASIN_PATTERN = re.compile(r"(?:^|\n)asin:\s*([^\n]+)")


def _portable_path(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _tool_names(row: dict) -> set[str]:
    return {
        str((tool.get("function") or {}).get("name") or "")
        for tool in row.get("tools") or []
    }


def _validate_row(row: dict) -> list[tuple[str, str]]:
    task_id = int(row["task_id"])
    known_tools = _tool_names(row)
    calls = []
    terminal_call_id = None
    for index, message in enumerate(row.get("messages") or []):
        if message.get("role") == "tool":
            folded = str(message.get("content") or "").casefold()
            if any(marker in folded for marker in GUARD_MARKERS):
                raise ValueError(f"task {task_id}: guard rejection at message {index}")
            continue
        if message.get("role") != "assistant":
            continue
        tool_calls = message.get("tool_calls") or []
        if len(tool_calls) > 1:
            raise ValueError(f"task {task_id}: multiple tool calls at message {index}")
        for call in tool_calls:
            function = call.get("function") or {}
            name = str(function.get("name") or "")
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"task {task_id}: invalid tool arguments") from exc
            if not isinstance(arguments, dict):
                raise ValueError(f"task {task_id}: tool arguments must be an object")
            if name not in known_tools:
                raise ValueError(f"task {task_id}: unknown tool {name!r}")
            signature = (name, json.dumps(arguments, ensure_ascii=False, sort_keys=True))
            calls.append(signature)
            if name == "buy_now":
                terminal_call_id = call.get("id")

    if sum(name == "buy_now" for name, _ in calls) != 1 or not calls or calls[-1][0] != "buy_now":
        raise ValueError(f"task {task_id}: trajectory must end with exactly one buy_now")
    messages = row.get("messages") or []
    if not messages or messages[-1].get("role") != "tool":
        raise ValueError(f"task {task_id}: missing terminal tool response")
    if messages[-1].get("tool_call_id") != terminal_call_id:
        raise ValueError(f"task {task_id}: terminal tool response mismatch")
    terminal_content = str(messages[-1].get("content") or "").strip()
    if terminal_content != "购买已完成。" and not re.search(
        r"(?:^|\n)page_type:\s*terminal(?:\n|$)", terminal_content
    ):
        raise ValueError(f"task {task_id}: terminal reward text was not sanitized")
    return calls


def _bucket(label: dict) -> str:
    if label["difficulty"] == "simple":
        return "foundation"
    if (
        label["difficulty"] == "medium"
        and not label["needs_query_rewrite"]
        and not label["needs_candidate_comparison"]
    ):
        return "constraints"
    return "strategy"


def _split(ids: list[int], *, seed: int, validation_ratio: float) -> tuple[list[int], list[int]]:
    ordered = sorted(
        ids,
        key=lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(),
    )
    validation_count = int(len(ordered) * validation_ratio + 0.5)
    validation = sorted(ordered[:validation_count])
    train = sorted(ordered[validation_count:])
    return train, validation


def build_manifest(
    rows: list[dict],
    labels: list[dict],
    *,
    evaluation_ids: set[int],
    seed: int = 20260814,
    validation_ratio: float = 0.1,
) -> dict:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    row_ids = [int(row["task_id"]) for row in rows]
    label_ids = [int(label["task_id"]) for label in labels]
    if len(set(row_ids)) != len(row_ids):
        raise ValueError("duplicate SFT task IDs")
    if len(set(label_ids)) != len(label_ids):
        raise ValueError("duplicate difficulty-label task IDs")
    if set(row_ids) != set(label_ids):
        raise ValueError("SFT rows and difficulty labels must have identical task IDs")
    overlap = sorted(set(row_ids) & {int(task_id) for task_id in evaluation_ids})
    if overlap:
        raise ValueError(f"evaluation overlap: {overlap[:10]}")

    labels_by_id = {int(label["task_id"]): label for label in labels}
    bucket_ids = {name: [] for name in ("foundation", "constraints", "strategy")}
    review = {
        "candidate_comparison_under_evidenced": [],
        "consecutive_exact_action": [],
        "search_heavy": [],
        "trajectory_long": [],
    }
    limits = {
        "simple": {"searches": 2, "steps": 10},
        "medium": {"searches": 4, "steps": 16},
        "hard": {"searches": 6, "steps": 25},
    }
    for row in rows:
        task_id = int(row["task_id"])
        label = labels_by_id[task_id]
        if label.get("difficulty") not in limits:
            raise ValueError(f"task {task_id}: invalid difficulty")
        calls = _validate_row(row)
        bucket_ids[_bucket(label)].append(task_id)
        opened_asins = set()
        for message in row.get("messages") or []:
            if message.get("role") != "tool":
                continue
            match = ASIN_PATTERN.search(str(message.get("content") or ""))
            if match:
                opened_asins.add(match.group(1).strip())
        if label.get("needs_candidate_comparison") and len(opened_asins) < 2:
            review["candidate_comparison_under_evidenced"].append(task_id)
        if any(left == right for left, right in zip(calls, calls[1:])):
            review["consecutive_exact_action"].append(task_id)
        difficulty_limits = limits[label["difficulty"]]
        if sum(name == "search_products" for name, _ in calls) > difficulty_limits["searches"]:
            review["search_heavy"].append(task_id)
        if len(calls) > difficulty_limits["steps"]:
            review["trajectory_long"].append(task_id)

    buckets = {}
    for name, ids in bucket_ids.items():
        train, validation = _split(ids, seed=seed, validation_ratio=validation_ratio)
        buckets[name] = {
            "rows": len(ids),
            "train_task_ids": train,
            "validation_task_ids": validation,
        }
    stages = {}
    for name, config in STAGE_CONFIG.items():
        selected = config["buckets"]
        stages[name] = {
            **config,
            "train_rows": sum(len(buckets[bucket]["train_task_ids"]) for bucket in selected),
            "validation_rows": sum(
                len(buckets[bucket]["validation_task_ids"]) for bucket in selected
            ),
        }
    train_count = sum(len(bucket["train_task_ids"]) for bucket in buckets.values())
    validation_count = sum(
        len(bucket["validation_task_ids"]) for bucket in buckets.values()
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "split_seed": int(seed),
        "validation_ratio": float(validation_ratio),
        "counts": {
            "rows": len(rows),
            "train": train_count,
            "validation": validation_count,
            "evaluation_overlap": 0,
        },
        "buckets": buckets,
        "stages": stages,
        "review_flags": {key: sorted(value) for key, value in review.items()},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root / "data/sft/all.jsonl")
    parser.add_argument(
        "--labels", type=Path, default=root / "data/sft/difficulty_labels.jsonl"
    )
    parser.add_argument(
        "--evaluation", type=Path, default=root / "data/evaluation/tasks.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=root / "data/sft/curriculum.json"
    )
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    rows = _read_jsonl(args.source)
    labels = _read_jsonl(args.labels)
    evaluation_ids = {int(row["task_id"]) for row in _read_jsonl(args.evaluation)}
    manifest = build_manifest(
        rows,
        labels,
        evaluation_ids=evaluation_ids,
        seed=args.seed,
        validation_ratio=args.validation_ratio,
    )
    manifest["source"] = {"path": _portable_path(args.source, root)}
    manifest["labels"] = {"path": _portable_path(args.labels, root)}
    manifest["evaluation"] = {"path": _portable_path(args.evaluation, root)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **manifest["counts"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
