#!/usr/bin/env python3
"""05 测评:在 Final-200 Clean 盲测集上评测任意模型目录,并生成 HTML 报告。

用法:
    python scripts/05_evaluate.py models/sft        # 评测 SFT 模型 → works/eval/sft
    python scripts/05_evaluate.py models/rl/<标签>  # 评测 RL 模型 → works/eval/rl/<标签>
    python scripts/05_evaluate.py models/rl/rloo --label rloo
    python scripts/05_evaluate.py --recurate        # 只重新校验盲测集(不出报告)
    python scripts/05_evaluate.py models/base --no-serve   # 复用已有服务(LLM_BASE_URL)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopping_agent import config, pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model_dir", nargs="?", help="被测模型目录(如 models/sft)")
    parser.add_argument("--config", default=None, help="默认 configs/eval.yaml")
    parser.add_argument("--label", help="结果子目录名,默认取模型目录名")
    parser.add_argument("--no-serve", action="store_true", help="不自动起 vLLM,复用已有服务")
    parser.add_argument("--recurate", action="store_true", help="只重新校验盲测集")
    options, extra = parser.parse_known_args()

    config.bootstrap()
    settings = config.load_config("eval", options.config)
    if not options.recurate and not options.model_dir:
        parser.error("需要提供被测模型目录,或使用 --recurate")
    return pipeline.run_evaluate(
        settings,
        options.model_dir or "data/evaluation",
        label=options.label,
        no_serve=options.no_serve,
        extra_args=[item for item in extra if item != "--"],
        recurate_only=options.recurate,
    )


if __name__ == "__main__":
    raise SystemExit(main())
