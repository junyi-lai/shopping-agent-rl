#!/usr/bin/env python3
"""Build the RL task dataset (parquet) used by GRPO and RLOO.

RL prompts are raw ShopSimulator tasks: the shared agent system prompt plus the
instruction the environment returns for that task id, so training sees exactly
what evaluation sees.  The task pool excludes every SFT task and the frozen
evaluation set, keeping SFT, RL and the benchmark disjoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from shopping_agent.collection.task_pool import jsonl_task_ids, pool_size
from shopping_agent.environment.client import ShopAgentEnv
from shopping_agent.evaluation.rollout import SYSTEM_PROMPT
from shopping_agent.reward import REWARD_VERSION

SCHEMA_VERSION = "shopping-agent-dataset-v1"
DATA_SOURCE = "shopsimulator"
ABILITY = "shopping"


def select_task_ids(
    count: int,
    *,
    seed: int,
    excluded: set[int],
    pool_total: int | None = None,
) -> list[int]:
    """Deterministically pick tasks that avoid every excluded id."""

    total = pool_size() if pool_total is None else int(pool_total)
    available = sorted(set(range(total)) - {int(task_id) for task_id in excluded})
    if count > len(available):
        raise SystemExit(
            f"需要 {count} 个 RL 任务,但排除 SFT/评测后只剩 {len(available)} 个"
        )
    ordered = sorted(
        available,
        key=lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(),
    )
    return sorted(ordered[:count])


def split_train_validation(
    task_ids: list[int], *, seed: int, validation_ratio: float
) -> tuple[list[int], list[int]]:
    if not 0 <= validation_ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    ordered = sorted(
        task_ids,
        key=lambda task_id: hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest(),
    )
    validation_count = int(len(ordered) * validation_ratio + 0.5)
    return sorted(ordered[validation_count:]), sorted(ordered[:validation_count])


def build_rows(
    task_ids: list[int],
    *,
    base_url: str,
    split: str,
    env_factory=ShopAgentEnv,
) -> list[dict]:
    """Ask the environment for each instruction and build veRL-ready prompt rows."""

    rows: list[dict] = []
    with env_factory(base_url=base_url) as env:
        for index, task_id in enumerate(task_ids):
            reset = env.reset(int(task_id))
            instruction = str(reset.get("instruction") or "")
            if not instruction:
                raise SystemExit(f"环境未返回任务 {task_id} 的 instruction")
            env.release()
            rows.append(
                {
                    "data_source": DATA_SOURCE,
                    "prompt": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": instruction},
                    ],
                    "ability": ABILITY,
                    "reward_model": {"style": "rule", "ground_truth": None},
                    "extra_info": {"split": split, "index": index, "task_id": int(task_id)},
                }
            )
    return rows


def write_rows(rows: list[dict], parquet_path: Path) -> dict:
    """Write the parquet veRL reads; ``extra_info.task_id`` keeps the task list."""

    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    import pandas as pd

    frame = pd.DataFrame(rows)
    frame["prompt"] = frame["prompt"].apply(lambda value: list(value))
    frame.to_parquet(parquet_path, index=False)
    return {
        "parquet": str(parquet_path),
        "tasks": len(rows),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=1000)
    parser.add_argument("--validation-count", type=int, default=50)
    parser.add_argument("--source", type=Path, nargs="*", default=[], help="需要排除的任务 JSONL")
    parser.add_argument("--exclude", type=Path, nargs="*", default=[], help="额外排除的任务 JSONL")
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--base-url", default="http://127.0.0.1:5700")
    parser.add_argument("--env-version", default="shopsimulator-environment-v2.1")
    parser.add_argument("--metadata", type=Path, default=None, help="数据来源审计 JSON")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    excluded: set[int] = set()
    for path in list(args.source) + list(args.exclude):
        excluded |= jsonl_task_ids(path)
    if not excluded:
        raise SystemExit("必须通过 --source/--exclude 指定要排除的 SFT 与评测任务")

    total = args.train_count + args.validation_count
    task_ids = select_task_ids(total, seed=args.seed, excluded=excluded)
    train_ids, validation_ids = split_train_validation(
        task_ids, seed=args.seed, validation_ratio=args.validation_count / total
    )

    output_dir = args.output_dir
    train = write_rows(
        build_rows(train_ids, base_url=args.base_url, split="train"),
        output_dir / "train.parquet",
    )
    validation = write_rows(
        build_rows(validation_ids, base_url=args.base_url, split="validation"),
        output_dir / "validation.parquet",
    )

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "environment": args.env_version,
        "reward": REWARD_VERSION,
        "provenance": {
            "seed": int(args.seed),
            "excluded": "SFT 全部任务 + 冻结评测集 + 额外排除",
            "source": [str(path) for path in args.source],
            "extra_exclude": [str(path) for path in args.exclude],
            "note": "任务池与 SFT/评测不重叠;prompt 来自环境 reset 的真实 instruction",
        },
        "train": train,
        "validation": validation,
    }
    metadata_path = args.metadata or (output_dir / "metadata.json")
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"train": train["tasks"], "validation": validation["tasks"], "metadata": str(metadata_path)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
