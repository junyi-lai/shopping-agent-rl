#!/usr/bin/env python3
"""Re-validate the Final-200 Clean benchmark under Reward v4.

Reward v4 keeps the deterministic terminal ladder and adds trajectory-level
shaping (efficiency / evidence / exploration / anti-repeat). The terminal ladder
is unchanged, so the 200 tasks remain gold-reachable. This audit re-proves that
property statically for every task and records the static evidence-coverage upper
bound used by the v4 evidence bonus.

Run from the repository root:

    python scripts/05_evaluate.py <模型目录> --recurate
"""

from __future__ import annotations

import argparse
import gzip
import json
from copy import deepcopy
from pathlib import Path

from shopping_agent.reward.budget import explicit_budget_from_instruction
from shopping_agent.reward.features import compile_reward_features
from shopping_agent.reward.terminal import evaluate_purchase
from shopping_agent.reward.variant_price import (
    candidate_options_for_evaluation,
    resolve_variant_price,
)

ROOT = Path(__file__).resolve().parents[3]
SHOP_ENV_ROOT = ROOT / "environments/ShopSimulator/shop_env"


def load_tasks(path: Path) -> list[int]:
    return [
        int(json.loads(line)["task_id"])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_products() -> list[dict]:
    with gzip.open(
        SHOP_ENV_ROOT / "data/fine_items_eval_train_all.json.gz", "rt", encoding="utf-8"
    ) as stream:
        return json.load(stream)


def static_gold_check(product: dict, instruction: dict) -> dict:
    product = deepcopy(product)
    product.update(
        {
            "Title": product["title"],
            "Description": product.get("full_description", ""),
            "BulletPoints": [],
            "Attributes": product.get("attribute", []),
            "pricing": product.get("pricing", []),
        }
    )
    goal = {
        "asin": product["asin"],
        "category": product["category"],
        "price_upper": explicit_budget_from_instruction(instruction["instruction"]),
    }
    goal.update(compile_reward_features(instruction, product))
    selected, _ = candidate_options_for_evaluation(
        product, goal["required_options_by_key"]
    )
    price = resolve_variant_price(product, selected)
    result = evaluate_purchase(
        product, goal, selected_options=selected, price_resolution=price
    )
    evidence_coverage = float(
        result.evidence.get("preference_scoring", {}).get("evidence_coverage", 0.0)
    )
    return {
        "gold_reachable": result.reward_type == "gold_purchase",
        "reward_type": result.reward_type,
        "price_status": price["status"],
        "unresolved_options": goal["unresolved_option_requirements"],
        "price_upper": goal["price_upper"],
        "evidence_coverage": evidence_coverage,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tasks", type=Path, default=ROOT / "data/evaluation/tasks.jsonl"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "works/eval/recuration/reward_v4_audit.jsonl"
    )
    args = parser.parse_args(argv)

    task_ids = load_tasks(args.tasks)
    products = load_products()
    rows = []
    failures = []
    for task_id in task_ids:
        target = products[task_id]
        instruction = target["instructions"][0]
        check = static_gold_check(target, instruction)
        check["task_id"] = task_id
        rows.append(check)
        if not check["gold_reachable"] or check["price_status"] != "pass":
            failures.append(task_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    coverage = sum(row["evidence_coverage"] for row in rows)
    print(
        json.dumps(
            {
                "tasks": len(rows),
                "gold_reachable": sum(row["gold_reachable"] for row in rows),
                "price_resolvable": sum(row["price_status"] == "pass" for row in rows),
                "mean_static_evidence_coverage": coverage / len(rows) if rows else 0.0,
                "failures": failures,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
