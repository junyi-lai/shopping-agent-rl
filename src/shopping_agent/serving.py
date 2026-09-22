"""Serve a local model with vLLM for evaluation, and wait until it is ready.

Only the evaluation stage needs a served model: collection talks to the Teacher
API and RL training lets veRL manage its own rollout engine.  MiniCPM5 emits XML
tool calls and enables thinking by default, so the served model is always started
with ``--tool-call-parser minicpm5`` and ``enable_thinking=false``.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterator

DEFAULT_MAX_MODEL_LEN = 24576
DEFAULT_TOOL_PARSER = "minicpm5"
READY_TIMEOUT_SECONDS = 1800


def _vllm_command() -> list[str]:
    console = Path(sys.executable).with_name("vllm")
    if console.is_file():
        return [str(console)]
    found = shutil.which("vllm")
    if found:
        return [found]
    return [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]


def build_serve_command(
    model: str,
    *,
    port: int = 8000,
    host: str = "127.0.0.1",
    served_model_name: str = "shopping-agent",
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float | None = None,
    tool_call_parser: str = DEFAULT_TOOL_PARSER,
    enable_thinking: bool = False,
) -> list[str]:
    """Build the ``vllm serve`` command line used by the evaluation entry point."""

    command = _vllm_command()
    if command[-1].endswith("api_server"):  # module form takes "serve" implicitly
        pass
    else:
        command.append("serve")
    command += [
        str(model),
        "--host",
        str(host),
        "--port",
        str(port),
        "--served-model-name",
        str(served_model_name),
        "--max-model-len",
        str(int(max_model_len)),
        "--tensor-parallel-size",
        str(int(tensor_parallel_size)),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        str(tool_call_parser),
        "--default-chat-template-kwargs",
        json.dumps({"enable_thinking": bool(enable_thinking)}),
    ]
    if gpu_memory_utilization is not None:
        command += ["--gpu-memory-utilization", str(float(gpu_memory_utilization))]
    return command


def endpoint_ready(base_url: str, *, timeout: float = 3.0) -> bool:
    """Return True when an OpenAI-compatible endpoint answers ``GET /models``."""

    url = base_url.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 300
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_until_ready(base_url: str, *, timeout: float = READY_TIMEOUT_SECONDS) -> None:
    deadline = time.monotonic() + float(timeout)
    while time.monotonic() < deadline:
        if endpoint_ready(base_url):
            return
        time.sleep(5)
    raise RuntimeError(f"模型服务在 {int(timeout)} 秒内未就绪:{base_url}")


def stop_process(process: subprocess.Popen, *, grace: float = 20.0) -> None:
    if process.poll() is not None:
        return
    process.send_signal(signal.SIGTERM)
    try:
        process.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=grace)


@contextlib.contextmanager
def served_model(
    model: str | Path,
    *,
    port: int = 8000,
    host: str = "127.0.0.1",
    served_model_name: str = "shopping-agent",
    max_model_len: int = DEFAULT_MAX_MODEL_LEN,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float | None = None,
    log_path: Path | None = None,
    api_key: str = "EMPTY",
    ready_timeout: float = READY_TIMEOUT_SECONDS,
) -> Iterator[str]:
    """Start vLLM, yield the OpenAI-compatible base URL, then stop the server."""

    base_url = f"http://{host}:{int(port)}/v1"
    if endpoint_ready(base_url):
        print(f"复用已在运行的学生模型服务:{base_url}")
        yield base_url
        return

    command = build_serve_command(
        str(model),
        port=port,
        host=host,
        served_model_name=served_model_name,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    print("启动模型服务:" + " ".join(command))
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
    else:
        handle = None
    process = subprocess.Popen(
        command,
        stdout=handle or None,
        stderr=subprocess.STDOUT if handle else None,
        env={**os.environ, "VLLM_LOGGING_LEVEL": os.environ.get("VLLM_LOGGING_LEVEL", "INFO")},
    )
    try:
        wait_until_ready(base_url, timeout=ready_timeout)
        print(f"模型服务已就绪:{base_url}(api_key={api_key})")
        yield base_url
    finally:
        stop_process(process)
        if handle is not None:
            handle.close()
        print("模型服务已停止")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="本地模型目录或 Hugging Face 模型 ID")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--served-model-name", default="shopping-agent")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_MAX_MODEL_LEN)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_serve_command(
        args.model,
        port=args.port,
        host=args.host,
        served_model_name=args.served_model_name,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )
    print(" ".join(command))
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
