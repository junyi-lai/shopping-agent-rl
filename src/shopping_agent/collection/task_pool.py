#!/usr/bin/env python3
"""Deterministically sample a held-out task pool for collection.

The ShopSimulator task pool is ``[0, len(products))`` (each product carries one
instruction). ``sample_pool`` picks ``count`` task ids with a fixed seed,
excluding the frozen evaluation tasks and any other task lists passed in, so the
pool is reproducible and never overlaps the test set. It is driven by
``01_collect`` through :func:`shopping_agent.pipeline.ensure_task_pool`; the
``main`` below is for ad-hoc inspection only.

    python -m shopping_agent.collection.task_pool \
        --count 2000 \
        --output works/collect/task_pool.jsonl \
        --exclude data/evaluation/tasks.jsonl data/sft/all.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PRODUCTS_GZ = ROOT / "environments/ShopSimulator/shop_env/data/fine_items_eval_train_all.json.gz"
EVAL_TASKS = ROOT / "data/evaluation/tasks.jsonl"


def jsonl_task_ids(path: Path | None) -> set[int]:
    """Read the ``task_id`` set of a JSONL file (missing file means no exclusions)."""

    if path is None or not path.exists():
        return set()
    result = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            result.add(int(json.loads(line)["task_id"]))
    return result



def pool_size() -> int:
    with gzip.open(PRODUCTS_GZ, "rt", encoding="utf-8") as stream:
        products = json.load(stream)
    return len(products)


def sample_pool(count: int, *, seed: int, excluded: set[int]) -> list[int]:
    available = sorted(set(range(pool_size())) - excluded)
    if count > len(available):
        raise SystemExit(
            f"requested {count} tasks but only {len(available)} are available after exclusions"
        )
    ordered = sorted(
        available,
        key=lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(),
    )
    return sorted(ordered[:count])


def top_level_category(category) -> str:
    """Reduce a catalog path like ``数码相机/单反›全景相机`` to its first segment."""

    if not category:
        return "unknown"
    text = str(category)
    for separator in ("›", ">", "/"):
        if separator in text:
            return text.split(separator, 1)[0].strip()
    return text.strip() or "unknown"


def product_categories() -> dict[int, str]:
    """Map every task id (product index) to its top-level category."""

    with gzip.open(PRODUCTS_GZ, "rt", encoding="utf-8") as stream:
        products = json.load(stream)
    return {
        index: top_level_category(product.get("category"))
        for index, product in enumerate(products)
    }


def sample_pool_stratified(
    count: int,
    *,
    seed: int,
    excluded: set[int],
    category_of: dict[int, str] | None = None,
) -> list[int]:
    """Uniformly allocate ``count`` tasks across top-level categories.

    Every category gets ``count // n_categories`` slots (plus one for the first
    ``count % n_categories``), capped by how many tasks that category has after
    exclusions.  Selection inside a category is deterministic by seed, so raising
    ``count`` later only adds tasks and never drops previously selected ones.
    """

    if category_of is None:
        category_of = product_categories()
    available = sorted(set(category_of) - excluded)
    if count > len(available):
        raise SystemExit(
            f"requested {count} tasks but only {len(available)} are available after exclusions"
        )
    buckets: dict[str, list[int]] = {}
    for task_id in available:
        buckets.setdefault(category_of[task_id], []).append(task_id)
    categories = sorted(buckets)
    base, extra = divmod(int(count), len(categories))
    selected: list[int] = []
    for index, category in enumerate(categories):
        quota = base + (1 if index < extra else 0)
        pool = buckets[category]
        take = min(quota, len(pool))
        ordered = sorted(
            pool,
            key=lambda task_id: hashlib.sha256(
                f"{seed}:{category}:{task_id}".encode()
            ).hexdigest(),
        )
        selected.extend(ordered[:take])
    return sorted(selected)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "works/collect/task_pool.jsonl")
    parser.add_argument("--seed", type=int, default=20260814)
    parser.add_argument(
        "--exclude",
        type=Path,
        nargs="*",
        default=[],
        help="额外排除的 task JSONL(默认永远排除 Final-200 Clean)",
    )
    args = parser.parse_args(argv)

    excluded = jsonl_task_ids(EVAL_TASKS)
    for path in args.exclude:
        excluded |= jsonl_task_ids(path)

    selected = sample_pool(args.count, seed=args.seed, excluded=excluded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for task_id in selected:
            handle.write(json.dumps({"task_id": task_id}) + "\n")

    print(
        json.dumps(
            {
                "pool_size": pool_size(),
                "excluded": len(excluded),
                "selected": len(selected),
                "overlap_with_eval": len(set(selected) & jsonl_task_ids(EVAL_TASKS)),
                "seed": args.seed,
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
