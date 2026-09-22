#!/usr/bin/env python3
"""00 环境:安装 / 启动 / 停止 ShopSimulator(其余环节的前置条件)。

用法:
    python scripts/00_environment.py setup              # 依赖 + 环境 + 索引 + veRL 补丁 + 契约校验 + 环境自检
    python scripts/00_environment.py start --background # 后台启动环境服务
    python scripts/00_environment.py status             # 查看是否在运行
    python scripts/00_environment.py smoke              # 跑一条最小动作链自检
    python scripts/00_environment.py stop               # 停止环境服务
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from shopping_agent import config  # noqa: E402
from shopping_agent.environment import service  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("action", nargs="?", default="start", choices=("setup", "start", "stop", "status", "smoke"))
    parser.add_argument("--background", action="store_true", help="start 时后台运行并等待就绪")
    parser.add_argument("--skip-index", action="store_true", help="setup 时跳过搜索索引构建")
    parser.add_argument("--skip-patch", action="store_true", help="setup 时跳过 veRL 动态采样补丁")
    parser.add_argument("--task-id", type=int, default=0, help="smoke 使用的任务号")
    options = parser.parse_args()

    config.bootstrap()
    base_url = str(config.section(config.load_pipeline(), "environment", "base_url", default="http://127.0.0.1:5700"))
    argv = [options.action, "--base-url", base_url, "--task-id", str(options.task_id)]
    if options.background:
        argv.append("--background")
    if options.skip_index:
        argv.append("--skip-index")
    if options.skip_patch:
        argv.append("--skip-patch")
    return service.main(argv) or 0


if __name__ == "__main__":
    raise SystemExit(main())
