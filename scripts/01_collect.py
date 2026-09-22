#!/usr/bin/env python3
"""01 采集:教师模型(API)在 ShopSimulator 里生成轨迹,支持增量补采。

用法:
    python scripts/01_collect.py                        # 按 configs/collect.yaml 采集
    python scripts/01_collect.py --target-accepted 1200 # 加大目标 → 只补差额,老数据保留
    python scripts/01_collect.py --dry-run              # 只打印将要执行的参数
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopping_agent import config, pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="默认 configs/collect.yaml")
    parser.add_argument("--target-accepted", type=int, help="累计验收目标条数(增量)")
    parser.add_argument("--target-tasks", type=int, help="任务池规模(增量扩容)")
    parser.add_argument("--workers", type=int, help="并发请求数")
    parser.add_argument("--attempts-per-task", type=int, help="每任务尝试次数")
    parser.add_argument("--build-only", action="store_true", help="不调用教师,只用已有轨迹重建统计")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()

    config.bootstrap()
    settings = config.load_config("collect", options.config)
    return pipeline.run_collect(
        settings,
        target_accepted=options.target_accepted,
        target_tasks=options.target_tasks,
        workers=options.workers,
        attempts_per_task=options.attempts_per_task,
        build_only=options.build_only,
        dry_run=options.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
