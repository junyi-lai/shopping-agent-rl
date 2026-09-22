"""验证 LoRA SFT 入口的关键默认值。"""

import os
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shopping_agent.training.sft.lora import (
    DEFAULT_TARGET_MODULES,
    _load_preprocessing_components,
    _model_load_kwargs,
    _prepare_model_for_training,
    _resolve_dtype,
    _swanlab_config,
    _curriculum_task_ids,
    parse_args,
)


class _FakeTokenizer:
    pass


class _FakeAutoTokenizer:
    called = False

    @classmethod
    def from_pretrained(cls, model_name, trust_remote_code):
        del model_name, trust_remote_code
        cls.called = True
        return _FakeTokenizer()


class _FakeBitsAndBytesConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _FakeModel:
    def __init__(self):
        self.config = type("Config", (), {"use_cache": True})()
        self.input_grads_enabled = False

    def enable_input_require_grads(self):
        self.input_grads_enabled = True


class TrainLoraSftCliTest(unittest.TestCase):
    def test_curriculum_manifest_expands_cumulative_stage_ids(self):
        manifest = {
            "stages": {"b": {"buckets": ["foundation", "constraints"]}},
            "buckets": {
                "foundation": {
                    "train_task_ids": [1],
                    "validation_task_ids": [2],
                },
                "constraints": {
                    "train_task_ids": [3],
                    "validation_task_ids": [4],
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(_curriculum_task_ids(path, "b", "train"), {1, 3})
            self.assertEqual(_curriculum_task_ids(path, "b", "validation"), {2, 4})

    def test_defaults_are_suitable_for_small_minicpm5_lora_warmup(self):
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "/models/MiniCPM5-2B",
                "--train",
                "outputs/batch/train.jsonl",
                "--output",
                "checkpoints/minicpm5-shopping-lora",
                "--curriculum-manifest",
                "data/sft/curriculum.json",
                "--curriculum-stage",
                "b",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.model, "/models/MiniCPM5-2B")
        self.assertEqual(args.train, Path("outputs/batch/train.jsonl"))
        self.assertEqual(args.max_length, 24576)
        self.assertEqual(args.epochs, 3)
        self.assertEqual(args.lora_r, 16)
        self.assertEqual(args.lora_alpha, 32)
        self.assertEqual(args.gradient_accumulation_steps, 8)
        self.assertEqual(args.save_total_limit, 3)
        self.assertEqual(args.dtype, "auto")
        self.assertFalse(args.bf16)
        self.assertFalse(args.swanlab)
        self.assertEqual(args.swanlab_project, "shopping-agent")
        self.assertEqual(
            args.curriculum_manifest,
            Path("data/sft/curriculum.json"),
        )
        self.assertEqual(args.curriculum_stage, "b")

    def test_swanlab_flags_are_opt_in_and_keep_a_stable_run_name(self):
        """国内监控必须显式启用，且实验名可由调用方固定以便对比。"""
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "openbmb/MiniCPM5-2B",
                "--train",
                "outputs/train.jsonl",
                "--output",
                "outputs/adapter",
                "--swanlab",
                "--swanlab-project",
                "shopping-agent",
                "--swanlab-run-name",
                "minicpm5-2b-lora-v1",
            ],
        ):
            args = parse_args()

        self.assertTrue(args.swanlab)
        self.assertEqual(args.swanlab_project, "shopping-agent")
        self.assertEqual(args.swanlab_run_name, "minicpm5-2b-lora-v1")

    def test_swanlab_config_returns_a_stable_default_run_name(self):
        """SwanLab 由 main 中的显式 init 配置；此处只验证纯配置函数。"""
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "openbmb/MiniCPM5-2B",
                "--train",
                "outputs/train.jsonl",
                "--output",
                "outputs/run/adapter",
                "--swanlab",
                "--swanlab-mode",
                "local",
            ],
        ):
            args = parse_args()

        with patch.dict(sys.modules, {"swanlab": object()}), patch.dict(os.environ, {}, clear=True):
            report_to, run_name = _swanlab_config(args)
            self.assertEqual(report_to, "swanlab")
            self.assertIn("lora-r16", run_name)

    def test_text_model_uses_tokenizer_as_chat_template(self):
        """MiniCPM5-2B 是纯文本模型，tokenizer 即 chat template 持有者。"""
        tokenizer = _load_preprocessing_components(
            "openbmb/MiniCPM5-2B",
            auto_tokenizer=_FakeAutoTokenizer,
        )

        self.assertIsInstance(tokenizer, _FakeTokenizer)
        self.assertTrue(_FakeAutoTokenizer.called)

    def test_default_lora_targets_cover_minicpm5_llama_layers(self):
        """MiniCPM5-2B 是标准 Llama 架构，目标层为 q/k/v/o + gate/up/down。"""
        for name in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
            self.assertIn(name, DEFAULT_TARGET_MODULES)
        self.assertNotIn("in_proj_qkv", DEFAULT_TARGET_MODULES)
        self.assertNotIn("out_proj", DEFAULT_TARGET_MODULES)

    def test_acceleration_flags_build_liger_sdpa_and_standard_qlora_configuration(self):
        """D 组必须在 C 的 SDPA 基础上显式添加 NF4 QLoRA，而非传递未验证的 dict。"""
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model", "openbmb/MiniCPM5-2B",
                "--train", "outputs/train.jsonl",
                "--output", "outputs/adapter",
                "--liger-kernel",
                "--attention-implementation", "sdpa",
                "--qlora",
            ],
        ):
            args = parse_args()

        kwargs = _model_load_kwargs(args, dtype="bf16", bits_and_bytes_config=_FakeBitsAndBytesConfig)
        self.assertTrue(args.liger_kernel)
        self.assertEqual(kwargs["attn_implementation"], "sdpa")
        self.assertIsInstance(kwargs["quantization_config"], _FakeBitsAndBytesConfig)
        self.assertEqual(kwargs["quantization_config"].kwargs["bnb_4bit_quant_type"], "nf4")
        self.assertEqual(kwargs["quantization_config"].kwargs["bnb_4bit_compute_dtype"], "bf16")

    def test_dtype_auto_prefers_bf16_then_fp16_and_cpu_fp32(self):
        class FakeCuda:
            available = True
            bf16_supported = True

            @classmethod
            def is_available(cls):
                return cls.available

            @classmethod
            def is_bf16_supported(cls):
                return cls.bf16_supported

        fake_torch = type(
            "FakeTorch",
            (),
            {
                "cuda": FakeCuda,
                "bfloat16": "bf16",
                "float16": "fp16",
                "float32": "fp32",
            },
        )
        args = type("Args", (), {"dtype": "auto", "bf16": False})()

        self.assertEqual(_resolve_dtype(args, fake_torch), ("bf16", "bf16"))
        FakeCuda.bf16_supported = False
        self.assertEqual(_resolve_dtype(args, fake_torch), ("fp16", "fp16"))
        FakeCuda.available = False
        self.assertEqual(_resolve_dtype(args, fake_torch), ("fp32", "fp32"))

    def test_model_revision_is_forwarded_to_loader(self):
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model",
                "openbmb/MiniCPM5-2B",
                "--train",
                "outputs/train.jsonl",
                "--output",
                "outputs/adapter",
                "--revision",
                "frozen-revision",
            ],
        ):
            args = parse_args()

        kwargs = _model_load_kwargs(
            args,
            dtype="bf16",
            bits_and_bytes_config=_FakeBitsAndBytesConfig,
        )
        self.assertEqual(kwargs["revision"], "frozen-revision")

    def test_qlora_prepares_model_before_lora_and_keeps_gradient_checkpointing_compatible(self):
        """量化基座必须先做 PEFT 标准预处理，再由后续 LoRA 注入 adapter。"""
        with patch.object(
            sys,
            "argv",
            [
                "train_lora_sft.py",
                "--model", "openbmb/MiniCPM5-2B",
                "--train", "outputs/train.jsonl",
                "--output", "outputs/adapter",
                "--qlora",
                "--gradient-checkpointing",
            ],
        ):
            args = parse_args()
        model = _FakeModel()
        prepared = _FakeModel()
        prepare = unittest.mock.MagicMock(return_value=prepared)

        result = _prepare_model_for_training(model, args, prepare)

        self.assertIs(result, prepared)
        prepare.assert_called_once_with(model, use_gradient_checkpointing=True)
        self.assertFalse(result.config.use_cache)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
