#!/usr/bin/env python3
"""04 RL:GRPO 或 RLOO 训练,训练结束后自动合并检查点到 models/rl/<算法>。

用法:
    python scripts/04_rl.py grpo                 # GRPO(configs/grpo.yaml)
    python scripts/04_rl.py rloo                 # RLOO(configs/rloo.yaml)
    python scripts/04_rl.py grpo --dry-run       # 只打印训练命令与预检
    python scripts/04_rl.py rloo --resume        # 从 works/rl/rloo 的最新 checkpoint 接着训
    python scripts/04_rl.py rloo --label rloo_v6_lr3e-5_200step   # 给本次 run 命名
    python scripts/04_rl.py grpo -- --trainer.total_training_steps=200   # 透传 veRL Hydra 覆盖

约定:`--label <名>` 让三处产物同名 —— `works/rl/<名>`(训练)、`models/rl/<名>`(合并模型)、
`works/eval/rl/<名>`(评测,由模型目录名自动推出)。默认要求产物目录为空(防止误覆盖上次结果),
要接着训就显式加 `--resume`。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopping_agent import config, pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("algo", nargs="?", default="grpo", choices=("grpo", "rloo"))
    parser.add_argument("--model", help="RL 起点模型目录,默认 models/sft")
    parser.add_argument("--output", help="训练产物目录,默认 works/rl/<算法>")
    parser.add_argument("--logger", choices=("console", "swanlab"), default="console")
    parser.add_argument(
        "--label",
        help="本次 run 的名字,使 works/rl/<label> 与 models/rl/<label> 同名"
        "(评测默认 label 也会同名);不填则用 --output 目录名,最后回落到算法名",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从产物目录的最新 checkpoint 续训(默认要求目录为空,防误覆盖)",
    )
    options, extra = parser.parse_known_args()

    config.bootstrap()
    settings = config.load_config(options.algo)
    extra_args = [item for item in extra if item != "--"]
    return pipeline.run_rl(
        settings,
        algo=options.algo,
        model=options.model,
        output=options.output,
        logger=options.logger,
        extra_args=extra_args,
        dry_run=options.dry_run,
        resume=options.resume,
        label=options.label,
    )


if __name__ == "__main__":
    raise SystemExit(main())
