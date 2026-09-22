"""Package-level contracts: public imports stay importable and the veRL shim works."""

import importlib
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

import shopping_agent

PUBLIC_MODULES = (
    "shopping_agent.config",
    "shopping_agent.pipeline",
    "shopping_agent.serving",
    "shopping_agent.smoke",
    "shopping_agent.collection.collect",
    "shopping_agent.collection.curriculum",
    "shopping_agent.collection.filter",
    "shopping_agent.collection.labels",
    "shopping_agent.collection.rl_dataset",
    "shopping_agent.collection.sft",
    "shopping_agent.collection.task_pool",
    "shopping_agent.environment.actions",
    "shopping_agent.environment.client",
    "shopping_agent.environment.context",
    "shopping_agent.environment.manifest",
    "shopping_agent.environment.observation",
    "shopping_agent.environment.projection",
    "shopping_agent.environment.service",
    "shopping_agent.environment.tools",
    "shopping_agent.evaluation.artifacts",
    "shopping_agent.evaluation.benchmark",
    "shopping_agent.evaluation.blind_guard",
    "shopping_agent.evaluation.comparison_report",
    "shopping_agent.evaluation.model_client",
    "shopping_agent.evaluation.recurate",
    "shopping_agent.evaluation.report",
    "shopping_agent.evaluation.rollout",
    "shopping_agent.evaluation.summary",
    "shopping_agent.reward",
    "shopping_agent.reward.budget",
    "shopping_agent.reward.comparators",
    "shopping_agent.reward.contract",
    "shopping_agent.reward.features",
    "shopping_agent.reward.score",
    "shopping_agent.reward.terminal",
    "shopping_agent.reward.v4",
    "shopping_agent.reward.v5",
    "shopping_agent.reward.v6",
    "shopping_agent.reward.variant_price",
    "shopping_agent.reward.versions",
    "shopping_agent.training.sft.dataset",
    "shopping_agent.training.sft.lora",
    "shopping_agent.training.sft.merge",
    "shopping_agent.training.sft.train",
    "shopping_agent.training.rl.compat",
    "shopping_agent.training.rl.dynamic_sampling",
    "shopping_agent.training.rl.export",
    "shopping_agent.training.rl.patch",
    "shopping_agent.training.rl.preflight",
    "shopping_agent.training.rl.train",
)


class PublicModulesTest(unittest.TestCase):
    def test_package_exposes_a_version(self):
        self.assertTrue(shopping_agent.__version__)

    def test_every_public_module_imports(self):
        """每个公开模块都必须能在没有 veRL/GPU 的解释器里导入。"""
        for name in PUBLIC_MODULES:
            with self.subTest(module=name):
                self.assertIsNotNone(importlib.import_module(name))


class VerlCompatTest(unittest.TestCase):
    """veRL 不应为了纯 padding 操作强制依赖 FlashAttention。"""

    def test_installs_verl_builtin_padding_functions(self):
        attention = ModuleType("verl.utils.attention_utils")
        fallback = ModuleType("verl.utils.npu_flash_attn_utils")
        expected = tuple(object() for _ in range(4))
        (
            fallback.index_first_axis,
            fallback.pad_input,
            fallback.rearrange,
            fallback.unpad_input,
        ) = expected
        utils = ModuleType("verl.utils")
        utils.attention_utils = attention
        utils.npu_flash_attn_utils = fallback
        verl = ModuleType("verl")
        verl.utils = utils
        trainer = ModuleType("verl.trainer")
        ppo = ModuleType("verl.trainer.ppo")
        ray_trainer = ModuleType("verl.trainer.ppo.ray_trainer")

        class RayPPOTrainer:
            def _update_actor(self, batch):
                return batch

        ray_trainer.RayPPOTrainer = RayPPOTrainer

        with patch.dict(
            sys.modules,
            {
                "verl": verl,
                "verl.utils": utils,
                "verl.utils.attention_utils": attention,
                "verl.utils.npu_flash_attn_utils": fallback,
                "verl.trainer": trainer,
                "verl.trainer.ppo": ppo,
                "verl.trainer.ppo.ray_trainer": ray_trainer,
            },
        ):
            from shopping_agent.training.rl.compat import install_torch_padding_fallback

            install_torch_padding_fallback()

        self.assertEqual(attention._get_attention_functions(), expected)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
