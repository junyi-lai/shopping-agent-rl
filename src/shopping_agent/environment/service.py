#!/usr/bin/env python3
"""ShopSimulator lifecycle: install, start, stop, health check and smoke test.

The environment ships as source under ``environments/ShopSimulator`` and needs its
own Python 3.10 virtual environment, so the shell steps that used to live in
``scripts/setup.sh`` and ``scripts/start_environment.sh`` are reproduced here in
Python and driven by ``scripts/00_environment.py``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import shutil
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

from shopping_agent.config import PROJECT_ROOT
from shopping_agent.environment.client import ShopAgentEnv

SHOPSIM_ROOT = PROJECT_ROOT / "environments/ShopSimulator"
SHOP_ENV_ROOT = SHOPSIM_ROOT / "shop_env"
ENV_VENV_DIR = SHOPSIM_ROOT / ".venv-shopsim"
ENV_PYTHON = ENV_VENV_DIR / "bin/python"
PRODUCTS_GZ = SHOP_ENV_ROOT / "data/fine_items_eval_train_all.json.gz"
PRODUCTS = SHOP_ENV_ROOT / "data/items_eval_train.json"
START_SCRIPT = SHOP_ENV_ROOT / "start.sh"
WORK_DIR = PROJECT_ROOT / "works/environment"
PID_FILE = WORK_DIR / "environment.pid"
LOG_FILE = WORK_DIR / "environment.log"
MAIN_PYTHON = os.environ.get("MAIN_PYTHON", "3.12")
SHOPSIM_PYTHON = os.environ.get("SHOPSIM_PYTHON", "3.10")
DEFAULT_BASE_URL = "http://127.0.0.1:5700"


def _require_tool(name: str) -> str:
    found = shutil.which(name)
    if not found:
        raise SystemExit(f"需要 {name} 命令,请先安装(ShopSimulator 依赖它)")
    return found


def ensure_products() -> Path:
    """Decompress the embedded product archive once."""

    if not PRODUCTS_GZ.is_file():
        raise SystemExit(f"缺少内置商品数据:{PRODUCTS_GZ}")
    if not PRODUCTS.is_file():
        temporary = PRODUCTS.with_suffix(".json.preparing")
        with gzip.open(PRODUCTS_GZ, "rb") as source, temporary.open("wb") as target:
            shutil.copyfileobj(source, target)
        temporary.replace(PRODUCTS)
    return PRODUCTS


def build_index() -> None:
    if not ENV_PYTHON.is_file():
        raise SystemExit("ShopSimulator 虚拟环境不存在,请先运行 setup")
    environment = {**os.environ, "PYTHONPATH": "."}
    subprocess.run(
        [str(ENV_PYTHON), "scripts/build_index.py"],
        check=True,
        cwd=str(SHOP_ENV_ROOT),
        env=environment,
    )
    print(f"搜索索引:{SHOP_ENV_ROOT / 'search_engine/products.sqlite3'}")


def validate_environment_contract() -> None:
    """Check that the frozen manifest and the environment's config agree."""

    from shopping_agent.environment.manifest import (
        load_manifest,
        load_runtime_config,
        validate_runtime_config,
    )

    try:
        manifest = load_manifest()
        validate_runtime_config(manifest, load_runtime_config())
    except ValueError as exc:
        raise SystemExit(f"环境契约校验失败:{exc}") from exc
    print("环境契约校验通过(manifest 与运行期配置一致)")


def run_environment_tests() -> None:
    """Run the environment's own unit suite with its own interpreter."""

    if not ENV_PYTHON.is_file():
        raise SystemExit("ShopSimulator 虚拟环境不存在,请先运行 setup")
    status = subprocess.call(
        [str(ENV_PYTHON), "-m", "unittest", "discover", "-s", "tests", "-t", "."],
        cwd=str(SHOP_ENV_ROOT),
    )
    if status:
        raise SystemExit(f"ShopSimulator 自检失败(exit {status})")


def setup_environment(*, with_index: bool = True, with_patch: bool = True) -> int:
    """Install the training environment, the ShopSimulator environment and the index."""

    uv = _require_tool("uv")
    print("== 安装训练依赖(uv sync --extra sft --extra grpo)==")
    subprocess.run(
        [uv, "sync", "--python", MAIN_PYTHON, "--extra", "sft", "--extra", "grpo"],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    print(f"== 准备 ShopSimulator 独立环境(Python {SHOPSIM_PYTHON})==")
    if not ENV_PYTHON.is_file():
        subprocess.run(
            [uv, "venv", "--python", SHOPSIM_PYTHON, str(ENV_VENV_DIR)],
            check=True,
            cwd=str(PROJECT_ROOT),
        )
    subprocess.run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(ENV_PYTHON),
            "-r",
            str(SHOP_ENV_ROOT / "requirements.txt"),
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )

    print("== 解压商品数据 ==")
    products = ensure_products()
    print(f"商品数据:{products}")

    if with_index:
        print("== 构建搜索索引 ==")
        build_index()

    if with_patch:
        print("== 应用 veRL 动态采样补丁 ==")
        from shopping_agent.training.rl import patch

        patch.main([])

    print("== 校验环境契约 ==")
    validate_environment_contract()

    print("== 环境自检(自带单元测试)==")
    run_environment_tests()

    print("环境安装完成。")
    return 0


def environment_ready(base_url: str = DEFAULT_BASE_URL, *, timeout: float = 3.0) -> bool:
    """Return True when the ShopSimulator HTTP service answers."""

    url = base_url.rstrip("/") + "/"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 500
    except (urllib.error.URLError, OSError, ValueError):
        pass
    try:
        host, _, port = base_url.split("//", 1)[-1].partition(":")
        with socket.create_connection((host, int(port.split("/")[0])), timeout=timeout):
            return True
    except (OSError, ValueError):
        return False


def ensure_environment(base_url: str = DEFAULT_BASE_URL) -> None:
    """Fail with an actionable message when the environment is not running."""

    if not environment_ready(base_url):
        raise SystemExit(
            f"ShopSimulator 未在 {base_url} 运行;请先执行:"
            " python scripts/00_environment.py start --background"
        )


def start_environment(
    *,
    background: bool = False,
    base_url: str = DEFAULT_BASE_URL,
    ready_timeout: float = 300.0,
) -> int:
    if not ENV_PYTHON.is_file():
        raise SystemExit("ShopSimulator 未安装,请先执行:python scripts/00_environment.py setup")
    if environment_ready(base_url):
        print(f"ShopSimulator 已在运行:{base_url}")
        return 0

    environment = dict(os.environ)
    environment["PATH"] = os.pathsep.join(
        [str(ENV_VENV_DIR / "bin"), environment.get("PATH", "")]
    )
    command = ["bash", str(START_SCRIPT)]
    if not background:
        return subprocess.call(command, cwd=str(SHOP_ENV_ROOT), env=environment)

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_FILE.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            cwd=str(SHOP_ENV_ROOT),
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    PID_FILE.write_text(f"{process.pid}\n", encoding="utf-8")
    print(f"ShopSimulator 启动中(pid={process.pid},日志={LOG_FILE})")
    deadline = time.monotonic() + float(ready_timeout)
    while time.monotonic() < deadline:
        if environment_ready(base_url):
            print(f"ShopSimulator 已就绪:{base_url}")
            return 0
        if process.poll() is not None:
            raise SystemExit(f"ShopSimulator 启动失败,请查看日志:{LOG_FILE}")
        time.sleep(3)
    raise SystemExit(f"ShopSimulator 在 {int(ready_timeout)} 秒内未就绪,请查看日志:{LOG_FILE}")


def stop_environment(*, base_url: str = DEFAULT_BASE_URL) -> int:
    if not PID_FILE.is_file():
        if environment_ready(base_url):
            print(f"没有 pid 记录;请手动结束占用 {base_url} 的进程")
            return 1
        print("ShopSimulator 未在运行。")
        return 0
    pid = int(PID_FILE.read_text(encoding="utf-8").strip() or 0)
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            print(f"进程 {pid} 已不存在。")
    PID_FILE.unlink(missing_ok=True)
    print(f"已停止 ShopSimulator(pid={pid})")
    return 0


def write_run_result(output_dir, task_id, base_url, steps, reset=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"task_{task_id:04d}_{timestamp}.json"
    payload = {
        "task_id": task_id,
        "base_url": base_url,
        "reset": reset or {},
        "steps": steps,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path


def run_smoke(base_url, task_id, actions, output_dir, env_factory=ShopAgentEnv):
    steps = []
    with env_factory(base_url=base_url) as env:
        reset = env.reset(task_id)
        for action in actions:
            result = env.step(action)
            steps.append({"action": action, "result": result})
            if result.get("done", False):
                break
    return write_run_result(output_dir, task_id, base_url, steps, reset=reset)


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=("setup", "start", "stop", "status", "smoke"),
        default="smoke",
    )
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--background", action="store_true", help="start 时后台运行")
    parser.add_argument("--skip-index", action="store_true", help="setup 时跳过索引构建")
    parser.add_argument("--skip-patch", action="store_true", help="setup 时跳过 veRL 补丁")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--actions", nargs="+", default=["search[乳胶枕]"])
    parser.add_argument("--output-dir", type=Path, default=WORK_DIR / "smoke")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.action == "setup":
        return setup_environment(with_index=not args.skip_index, with_patch=not args.skip_patch)
    if args.action == "start":
        return start_environment(background=args.background, base_url=args.base_url)
    if args.action == "stop":
        return stop_environment(base_url=args.base_url)
    if args.action == "status":
        ready = environment_ready(args.base_url)
        print(f"{args.base_url} -> {'运行中' if ready else '未运行'}")
        return 0 if ready else 1
    path = run_smoke(args.base_url, args.task_id, args.actions, args.output_dir)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
