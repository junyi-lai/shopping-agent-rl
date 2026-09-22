#!/usr/bin/env python3
"""把完成 SFT 的 LoRA adapter 合并为 GRPO 的独立起点。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_merge_manifest(base_model, adapter_path, output_path, model_type):
    """输出可审计清单；GRPO 必须新挂 adapter，不能覆盖这份 checkpoint。"""
    return {
        "operation": "peft_merge_and_unload",
        "source": {"base_model": str(base_model), "adapter": str(adapter_path), "model_type": str(model_type)},
        "output": str(output_path),
        "next_step": "load this standalone checkpoint as GRPO base and attach a new LoRA adapter",
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description="合并 LoRA SFT adapter，为 GRPO 创建独立 BF16 起点")
    parser.add_argument("--base-model", required=True, help="与 SFT 完全一致的原始模型路径或 Hugging Face 名称")
    parser.add_argument("--adapter", type=Path, required=True, help="SFT LoRA adapter 目录")
    parser.add_argument("--output", type=Path, required=True, help="新的 merged checkpoint 目录，必须为空")
    parser.add_argument("--bf16", action="store_true", help="以 bf16 合并；4090/RTX PRO 6000 建议开启")
    parser.add_argument("--max-shard-size", default="5GB")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None):
    args = parse_args(argv)
    if args.output.exists() and any(args.output.iterdir()):
        raise SystemExit(f"拒绝覆盖非空输出目录：{args.output}")
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("缺少合并依赖；请执行：uv sync --extra sft") from exc

    config = AutoConfig.from_pretrained(args.base_model, trust_remote_code=True)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    print(f"加载 base={args.base_model} model_type={config.model_type} dtype={dtype}")
    base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=dtype, trust_remote_code=True)
    merged = PeftModel.from_pretrained(base, str(args.adapter)).merge_and_unload()
    args.output.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(args.output), safe_serialization=True, max_shard_size=args.max_shard_size)
    # 学生模型是纯文本模型：只带 tokenizer，与 LoRA adapter 目录保持一致。
    AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True).save_pretrained(
        str(args.output)
    )
    manifest = build_merge_manifest(args.base_model, args.adapter, args.output, config.model_type)
    (args.output / "merge_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
