#!/usr/bin/env python3
"""Score accepted SFT trajectories with the Reward v4 process score.

``02_filter`` drives this stage: it passes the collection output, reads
``quality.jsonl`` back and does the selection itself (best trajectory per task,
honouring the ``--min-total`` / ``--keep-top`` knobs). The score distribution is
printed so the knobs can be chosen with real numbers; the authoritative run
summary is written by the pipeline into ``works/filter/summary.json``.

The module can also be run directly to inspect the scores:

    python -m shopping_agent.collection.filter \
        --input works/collect/accepted.jsonl \
        --output-dir works/filter \
        --max-steps 35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shopping_agent.reward import (
    compute_shaped_reward,
    count_consecutive_duplicate_actions,
    count_evidence_actions,
)


def tool_steps(messages) -> list[dict]:
    """Extract the ordered tool-call sequence from an OpenAI chat trajectory."""
    steps = []
    for message in messages or []:
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function") or {}
            name = function.get("name")
            raw_arguments = function.get("arguments")
            if isinstance(raw_arguments, str):
                try:
                    arguments = json.loads(raw_arguments or "{}")
                except json.JSONDecodeError:
                    arguments = {}
            else:
                arguments = raw_arguments or {}
            if not isinstance(arguments, dict):
                arguments = {}
            steps.append({"tool_name": name, "parameters": arguments})
    return steps


def score_row(row: dict, *, max_steps: int) -> dict:
    steps = tool_steps(row.get("messages"))
    shaped = compute_shaped_reward(
        1.0,  # 所有已验收轨迹都是 gold_purchase,终局分恒为 1.0
        purchase_success=True,
        steps=len(steps),
        evidence_actions=count_evidence_actions(steps),
        repeat_action_count=count_consecutive_duplicate_actions(steps),
        reward_type="gold_purchase",
        reward_valid=True,
        max_steps=max_steps,
    )
    return {
        "task_id": row.get("task_id"),
        "trajectory_id": row.get("trajectory_id"),
        "v4_total": shaped["total"],
        "steps": len(steps),
        "evidence_actions": count_evidence_actions(steps),
        "repeat_actions": count_consecutive_duplicate_actions(steps),
        "shaping": shaped["shaping"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("works/filter"))
    parser.add_argument("--min-total", type=float, default=None)
    parser.add_argument("--keep-top", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=35)
    args = parser.parse_args(argv)

    rows = [
        json.loads(line)
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    scored = [score_row(row, max_steps=args.max_steps) for row in rows]
    scored.sort(key=lambda item: item["v4_total"], reverse=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "quality.jsonl").open("w", encoding="utf-8") as handle:
        for item in scored:
            handle.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")

    totals = [item["v4_total"] for item in scored]
    summary = {
        "rows": len(scored),
        "v4_total": {
            "min": min(totals),
            "median": sorted(totals)[len(totals) // 2],
            "max": max(totals),
            "mean": sum(totals) / len(totals),
        },
        "mean_shaping": {
            "efficiency_bonus": sum(i["shaping"]["efficiency_bonus"] for i in scored) / len(scored),
            "evidence_bonus": sum(i["shaping"]["evidence_bonus"] for i in scored) / len(scored),
            "repeat_penalty": sum(i["shaping"]["repeat_penalty"] for i in scored) / len(scored),
        },
    }

    if args.keep_top is not None and args.keep_top > 0:
        summary["selection"] = {
            "mode": "keep_top",
            "count": min(args.keep_top, len(scored)),
        }
    elif args.min_total is not None:
        summary["selection"] = {
            "mode": "min_total",
            "threshold": args.min_total,
            "count": sum(item["v4_total"] >= args.min_total for item in scored),
        }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
