#!/usr/bin/env python3
"""Run the fixed SFT curriculum with the LoRA train and merge modules."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
STAGES = ("a", "b", "c")


def build_stage_commands(
    manifest,
    *,
    manifest_path,
    source,
    base_model,
    output_root,
    python,
    start_stage="a",
    stop_after_stage="c",
    swanlab=False,
    swanlab_project="shopping-agent-sft-curriculum",
    qlora=False,
    liger_kernel=False,
    resume_from_checkpoint=None,
):
    start = STAGES.index(start_stage)
    stop = STAGES.index(stop_after_stage)
    if stop < start:
        raise ValueError("--stop-after-stage must be at or after --start-stage")

    commands = []
    for index, stage in enumerate(STAGES[start : stop + 1]):
        stage_config = manifest["stages"][stage]
        stage_root = Path(output_root) / f"stage-{stage}"
        model = (
            base_model
            if stage == "a"
            else str(Path(output_root) / f"stage-{STAGES[STAGES.index(stage) - 1]}" / "merged")
        )
        train = [
            str(python),
            "-m",
            "shopping_agent.training.sft.lora",
            "--model",
            str(model),
            "--train",
            str(source),
            "--validation",
            str(source),
            "--curriculum-manifest",
            str(manifest_path),
            "--curriculum-stage",
            stage,
            "--output",
            str(stage_root / "adapter"),
            "--epochs",
            str(stage_config["epochs"]),
            "--learning-rate",
            str(stage_config["learning_rate"]),
            "--max-length",
            "24576",
            "--dtype",
            "bf16",
            "--attention-implementation",
            "sdpa",
            "--gradient-checkpointing",
            "--swanlab-run-name",
            f"sft-stage-{stage}",
        ]
        if swanlab:
            train.extend(["--swanlab", "--swanlab-project", swanlab_project])
        if qlora:
            train.append("--qlora")
        if liger_kernel:
            train.append("--liger-kernel")
        if index == 0 and resume_from_checkpoint:
            train.extend(["--resume-from-checkpoint", str(resume_from_checkpoint)])
        merge = [
            str(python),
            "-m",
            "shopping_agent.training.sft.merge",
            "--base-model",
            str(model),
            "--adapter",
            str(stage_root / "adapter"),
            "--output",
            str(stage_root / "merged"),
            "--bf16",
        ]
        commands.append({"stage": stage, "train": train, "merge": merge})
    return commands


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="openbmb/MiniCPM5-2B")
    parser.add_argument(
        "--source", type=Path, default=ROOT / "data/sft/all.jsonl"
    )
    parser.add_argument(
        "--manifest", type=Path, default=ROOT / "data/sft/curriculum.json"
    )
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "works/sft"
    )
    parser.add_argument("--start-stage", choices=STAGES, default="a")
    parser.add_argument("--stop-after-stage", choices=STAGES, default="c")
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--swanlab", action="store_true")
    parser.add_argument("--swanlab-project", default="shopping-agent-sft-curriculum")
    parser.add_argument("--qlora", action="store_true")
    parser.add_argument("--liger-kernel", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "shopping-sft-curriculum-v1":
        raise SystemExit("不支持的课程清单 schema_version")
    try:
        commands = build_stage_commands(
            manifest,
            manifest_path=args.manifest,
            source=args.source,
            base_model=args.base_model,
            output_root=args.output_root,
            python=sys.executable,
            start_stage=args.start_stage,
            stop_after_stage=args.stop_after_stage,
            swanlab=args.swanlab,
            swanlab_project=args.swanlab_project,
            qlora=args.qlora,
            liger_kernel=args.liger_kernel,
            resume_from_checkpoint=args.resume_from_checkpoint,
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    for command in commands:
        print(f"\n[stage {command['stage'].upper()}] train")
        print(shlex.join(command["train"]))
        print(f"[stage {command['stage'].upper()}] merge")
        print(shlex.join(command["merge"]))
        if args.dry_run:
            continue
        merged = args.output_root / f"stage-{command['stage']}" / "merged"
        if merged.exists() and any(merged.iterdir()):
            raise SystemExit(f"拒绝覆盖已完成阶段：{merged}")
        subprocess.run(command["train"], check=True)
        subprocess.run(command["merge"], check=True)
    last_stage = commands[-1]["stage"]
    label = "最终 GRPO 起点" if last_stage == "c" else "本次最后阶段输出"
    print(f"\n{label}：{args.output_root / f'stage-{last_stage}/merged'}")


if __name__ == "__main__":
    main()
