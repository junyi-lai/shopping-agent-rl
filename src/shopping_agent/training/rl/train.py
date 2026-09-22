#!/usr/bin/env python3
"""Run the repository's single supported Shopping Agent GRPO recipe."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = ROOT / "configs/grpo.yaml"
DEFAULT_AGENT_CONFIG = ROOT / "configs/agent_loop.yaml"
DEFAULT_TOOL_CONFIG = ROOT / "configs/tools.json"
DEFAULT_MANIFEST = ROOT / "configs/environment.json"
DEFAULT_MODEL = ROOT / "models/sft"
DEFAULT_TRAIN_DATA = ROOT / "data/rl/train.parquet"
DEFAULT_VAL_DATA = ROOT / "data/rl/validation.parquet"


def _model_has_weights(path: Path) -> bool:
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return any((path / name).is_file() for name in candidates)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--train-data", type=Path, default=DEFAULT_TRAIN_DATA)
    parser.add_argument("--val-data", type=Path, default=DEFAULT_VAL_DATA)
    parser.add_argument("--env-url", default="http://127.0.0.1:5700")
    parser.add_argument("--output", type=Path, default=Path("works/rl/grpo"))
    parser.add_argument(
        "--logger",
        choices=("console", "swanlab"),
        default="console",
    )
    parser.add_argument("--experiment-name", default="shopping-agent-grpo")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从 --output 下最新的 global_step_* checkpoint 续训(允许目录非空)",
    )
    parser.add_argument(
        "hydra_overrides",
        nargs=argparse.REMAINDER,
        help="additional veRL Hydra overrides after --",
    )
    return parser.parse_args(argv)


def _validated_path(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise SystemExit(f"{description} does not exist: {resolved}")
    return resolved


def build_overrides(args: argparse.Namespace) -> list[str]:
    """训练命令与运行时预检共用的 Hydra override 列表。

    预检会用同一批 override 组出配置再校验,所以两边必须一致。
    """
    logger_override = (
        "trainer.logger=[console,swanlab]"
        if args.logger == "swanlab"
        else "trainer.logger=[console]"
    )
    # Hydra 默认把配置快照写到仓库根的 outputs/<日期>/<时间>/,还会在 run 目录里留 .hydra/ 与
    # main_ppo.log。下面四条把它管死,使一次 run 的产物目录里**只有真产物**:
    #   run.dir       → 快照与日志不落仓库根,归到本次 run 名下
    #   output_subdir → null:不写 .hydra/
    #   job_logging   → stdout:只往控制台打日志(仍是 INFO),不写 main_ppo.log
    #   job.chdir     → false:不切换进程 CWD,否则相对路径全移位
    run_dir = args.output.expanduser().resolve()
    overrides = [
        f"hydra.run.dir={run_dir}",
        "hydra.job.chdir=false",
        "hydra.output_subdir=null",
        "hydra/job_logging=stdout",
        logger_override,
        f"trainer.experiment_name={args.experiment_name}",
    ]
    if args.resume:
        # 显式声明续训语义:veRL 默认就是 auto(有 checkpoint 就接着训),写出来
        # 是为了即使配置将来改了也不会悄悄变成从零开始。
        overrides.append("trainer.resume_mode=auto")
    extra = list(args.hydra_overrides)
    if extra[:1] == ["--"]:
        extra = extra[1:]
    return [*overrides, *extra]


def build_command(args: argparse.Namespace) -> tuple[list[str], dict[str, str]]:
    model = _validated_path(args.model, "model directory")
    if not model.is_dir() or not (model / "config.json").is_file():
        raise SystemExit(f"model directory is missing config.json: {model}")
    if not _model_has_weights(model):
        raise SystemExit(
            "model directory has no supported weight file or sharded index: "
            f"{model}"
        )
    train_data = _validated_path(args.train_data, "train parquet")
    val_data = _validated_path(args.val_data, "validation parquet")
    config = _validated_path(args.config, "GRPO example config")
    output = args.output.expanduser().resolve()
    if output.exists():
        if not output.is_dir():
            raise SystemExit(f"output must be a directory: {output}")
        if any(output.iterdir()):
            if not args.resume:
                raise SystemExit(
                    f"output directory must be new or empty: {output} "
                    "(pass --resume to continue from its latest checkpoint)"
                )
            checkpoints = sorted(
                output.glob("global_step_*"),
                # 按步数数值排序:字符串排序会把 global_step_50 排到 global_step_200 之后。
                key=lambda path: int(path.name.rsplit("_", 1)[-1]),
            )
            if checkpoints:
                print(
                    f"续训:从 {output}/{checkpoints[-1].name} 继续"
                    f"(该目录共 {len(checkpoints)} 个 checkpoint)"
                )
            else:
                print(
                    f"续训:已指定 --resume,但 {output} 下没有 global_step_* checkpoint,"
                    "将从零开始"
                )
    if args.logger == "swanlab" and not os.environ.get("SWANLAB_API_KEY"):
        raise SystemExit("--logger swanlab requires SWANLAB_API_KEY")

    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONPATH": str(ROOT / "src"),
            "SHOPPING_AGENT_ROOT": str(ROOT),
            "SHOPPING_ENVIRONMENT_VERSION": "shopsimulator-environment-v2.1",
            "SHOPPING_ENV_MANIFEST": str(DEFAULT_MANIFEST),
            "RL_MODEL_PATH": str(model),
            "RL_TRAIN_FILE": str(train_data),
            "RL_VAL_FILE": str(val_data),
            "RL_OUTPUT_DIR": str(output),
            "SHOPPING_AGENT_DIAGNOSTICS_PATH": str(
                output / "training_diagnostics.jsonl"
            ),
            "SHOPSIM_BASE_URL": str(args.env_url),
            "SHOPPING_AGENT_LOOP_CONFIG": str(DEFAULT_AGENT_CONFIG),
            "SHOPPING_TOOL_CONFIG": str(DEFAULT_TOOL_CONFIG),
            "RL_CONFIG_NAME": config.stem,
        }
    )
    if args.logger == "swanlab":
        environment.update(
            {
                "SWANLAB_MODE": "online",
                "SWANLAB_LOG_DIR": str(output / "swanlab"),
            }
        )
    hydra_overrides = build_overrides(args)
    command = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"--config-path={config.parent}",
        f"--config-name={config.stem}",
        *hydra_overrides,
    ]
    return command, environment


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    command, environment = build_command(args)
    audit = {
        "command": command,
        "model": environment["RL_MODEL_PATH"],
        "train_data": environment["RL_TRAIN_FILE"],
        "val_data": environment["RL_VAL_FILE"],
        "env_url": environment["SHOPSIM_BASE_URL"],
        "output": environment["RL_OUTPUT_DIR"],
        "logger": args.logger,
        "config": str(args.config.resolve()),
    }
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    Path(environment["RL_OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    preflight = [
        sys.executable,
        "-m",
        "shopping_agent.training.rl.preflight",
        *build_overrides(args),
    ]
    preflight_status = subprocess.call(preflight, cwd=ROOT, env=environment)
    if preflight_status:
        raise SystemExit(preflight_status)
    raise SystemExit(subprocess.call(command, cwd=ROOT, env=environment))


if __name__ == "__main__":
    main()
