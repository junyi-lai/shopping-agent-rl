#!/usr/bin/env python3
"""02 筛选:V4 重打分 → 过滤 → 难度标签 → 课程清单 → RL 任务集。

用法:
    python scripts/02_filter.py                    # 按 configs/filter.yaml 执行
    python scripts/02_filter.py --keep-top 800     # 只保留质量最高的 800 条
    python scripts/02_filter.py --skip-labels      # 跳过难度标签(标签已齐全时)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopping_agent import config, pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="默认 configs/filter.yaml")
    parser.add_argument("--min-total", type=float, help="只保留 V4 总分 ≥ 该值的轨迹")
    parser.add_argument("--keep-top", type=int, help="只保留质量最高的前 N 条")
    parser.add_argument("--skip-labels", action="store_true", help="不补难度标签")
    parser.add_argument("--skip-rl-data", action="store_true", help="不重建 RL 任务集")
    options = parser.parse_args()

    config.bootstrap()
    settings = config.load_config("filter", options.config)
    if options.min_total is not None:
        settings.setdefault("rescoring", {})["min-total"] = options.min_total
    if options.keep_top is not None:
        settings.setdefault("rescoring", {})["keep-top"] = options.keep_top
    return pipeline.run_filter(
        settings,
        skip_labels=options.skip_labels,
        skip_rl_data=options.skip_rl_data,
    )


if __name__ == "__main__":
    raise SystemExit(main())
