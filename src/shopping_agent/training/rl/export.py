#!/usr/bin/env python3
"""Merge a veRL FSDP actor checkpoint into a standalone Hugging Face model.

RL training writes sharded FSDP checkpoints under ``works/rl/<algo>/global_step_*``;
this module turns the newest one into a plain model directory (``models/rl/<algo>``)
that the evaluation entry point can serve.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def latest_actor_checkpoint(output_dir: str | Path) -> Path:
    """Return the newest ``global_step_*/actor`` directory under an RL run dir."""

    candidates = [
        path
        for path in Path(output_dir).glob("global_step_*/actor")
        if path.is_dir()
    ]
    if not candidates:
        raise SystemExit(f"{output_dir} 下找不到 global_step_*/actor 检查点")
    return max(candidates, key=lambda path: int(path.parent.name.split("_")[-1]))


def merge_lora_adapter(target: Path) -> bool:
    """把 veRL 导出的 LoRA 适配器合并进 base 权重。

    对 LoRA 检查点,``verl.model_merger`` 输出的是"base 权重 + ``lora_adapter/`` 子目录",
    而 vLLM 直接 serve 这个目录时**只会加载 base** —— 等于把 RL 训出来的适配器丢掉,
    评测定出来的其实是 SFT 模型(实测两者张量逐位相同)。这里按 SFT 归档的同一套做法
    (``PeftModel.merge_and_unload``)把适配器并成独立模型,并删除适配器目录避免再次误用。
    """

    adapter = target / "lora_adapter"
    if not adapter.is_dir():
        return False

    import torch  # 延迟导入:只有真的存在适配器时才需要重依赖
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"检测到 LoRA 适配器,合并进 base:{adapter}")
    base = AutoModelForCausalLM.from_pretrained(
        str(target), torch_dtype=torch.bfloat16, trust_remote_code=True
    )
    merged = PeftModel.from_pretrained(base, str(adapter)).merge_and_unload()
    merged.save_pretrained(str(target), safe_serialization=True)
    AutoTokenizer.from_pretrained(str(target), trust_remote_code=True).save_pretrained(
        str(target)
    )
    shutil.rmtree(adapter)
    print("LoRA 适配器已合并,适配器目录已删除")
    return True


def export_checkpoint(
    actor_checkpoint: str | Path,
    target_dir: str | Path,
    *,
    backend: str = "fsdp",
) -> int:
    """Run veRL's model merger and record an audit manifest."""

    target = Path(target_dir)
    if target.exists() and any(target.iterdir()):
        raise SystemExit(f"拒绝覆盖非空输出目录:{target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "verl.model_merger",
        "merge",
        "--backend",
        backend,
        "--local_dir",
        str(actor_checkpoint),
        "--target_dir",
        str(target),
        "--trust-remote-code",
    ]
    print(" ".join(command))
    status = subprocess.call(command)
    if status == 0:
        merged_adapter = merge_lora_adapter(target)
        manifest = {
            "operation": "verl.model_merger",
            "backend": backend,
            "source": str(actor_checkpoint),
            "target": str(target),
            "lora_adapter_merged": merged_adapter,
            "merged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
        (target / "export_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return status


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, help="RL 运行目录(自动取最新检查点)")
    parser.add_argument("--actor-checkpoint", type=Path, help="显式指定 global_step_*/actor")
    parser.add_argument("--target", type=Path, required=True, help="导出的模型目录")
    parser.add_argument("--backend", default="fsdp")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.actor_checkpoint is not None:
        checkpoint = args.actor_checkpoint
    elif args.run_dir is not None:
        checkpoint = latest_actor_checkpoint(args.run_dir)
    else:
        raise SystemExit("需要 --run-dir 或 --actor-checkpoint")
    return export_checkpoint(checkpoint, args.target, backend=args.backend)


if __name__ == "__main__":
    raise SystemExit(main())
