#!/usr/bin/env python3
"""03 SFT:课程 a/b/c 逐级 LoRA 训练,并把最终合并模型归档到 models/sft。

用法:
    python scripts/03_sft.py                 # 按 configs/sft.yaml 跑完整课程
    python scripts/03_sft.py --dry-run       # 只打印将要执行的训练/合并命令
    python scripts/03_sft.py --start-stage b # 从 b 阶段继续(需要 stage-a/merged 已存在)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopping_agent import config, pipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="默认 configs/sft.yaml")
    parser.add_argument("--start-stage", choices=("a", "b", "c"), help="起始课程阶段")
    parser.add_argument("--stop-after-stage", choices=("a", "b", "c"), help="结束课程阶段")
    parser.add_argument("--swanlab", action="store_true", help="启用 SwanLab 监控")
    parser.add_argument("--dry-run", action="store_true")
    options = parser.parse_args()

    config.bootstrap()
    settings = config.load_config("sft", options.config)
    return pipeline.run_sft(
        settings,
        start_stage=options.start_stage,
        stop_after_stage=options.stop_after_stage,
        swanlab=True if options.swanlab else None,
        dry_run=options.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
