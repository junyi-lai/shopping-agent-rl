"""Reproducibility tests for the vendored veRL source patch."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import torch
    from verl import DataProto
except ImportError as exc:  # torch/veRL only exist inside the GRPO environment
    raise unittest.SkipTest(f"GRPO dependencies are unavailable: {exc}") from exc
from shopping_agent.training.rl import patch as patcher
from shopping_agent.training.rl.dynamic_sampling import (
    extract_shopping_group_signals,
    select_reward_varying_groups,
)


ROOT = Path(__file__).resolve().parents[1]
PATCH_MODULE = "shopping_agent.training.rl.patch"


def original_source() -> Path:
    installed = patcher.resolve_installed_ray_trainer()
    if not patcher.has_current_marker(installed):
        return installed
    backup = Path(str(installed) + patcher.BACKUP_SUFFIX)
    if not backup.is_file():
        raise AssertionError("patched veRL source has no original backup")
    return backup


class VerlPatchScriptTest(unittest.TestCase):
    def run_script(self, target: Path, *args: str) -> subprocess.CompletedProcess[str]:
        import os

        environment = dict(os.environ)
        source_root = str(ROOT / "src")
        environment["PYTHONPATH"] = os.pathsep.join(
            [source_root, environment.get("PYTHONPATH", "")]
        ).strip(os.pathsep)
        return subprocess.run(
            [sys.executable, "-m", PATCH_MODULE, "--target", str(target), *args],
            cwd=ROOT,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )

    def test_apply_is_idempotent_and_restore_recovers_original(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            shutil.copy2(original_source(), target)
            pristine = target.read_bytes()

            first = self.run_script(target)
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertIn(patcher.PATCH_MARKER, target.read_text(encoding="utf-8"))
            patched = target.read_bytes()
            self.assertNotEqual(patched, pristine)

            second = self.run_script(target)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertIn("already applied", second.stdout)
            self.assertEqual(target.read_bytes(), patched)

            checked = self.run_script(target, "--check")
            self.assertEqual(checked.returncode, 0, checked.stderr)

            restored = self.run_script(target, "--restore")
            self.assertEqual(restored.returncode, 0, restored.stderr)
            self.assertEqual(target.read_bytes(), pristine)

    def test_check_rejects_a_runtime_without_the_patch_marker(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            shutil.copy2(original_source(), target)

            unchecked = self.run_script(target, "--check")
            self.assertNotEqual(unchecked.returncode, 0)
            self.assertIn("missing marker", unchecked.stderr)

    def test_failed_apply_rolls_back_the_target(self):
        """补丁上下文不匹配时必须回滚,不留半个补丁。"""
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            target.write_text("# not the real ray_trainer.py\n", encoding="utf-8")
            before = target.read_bytes()

            result = self.run_script(target)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(target.read_bytes(), before)

    def test_patched_fit_preserves_bypass_and_defers_reference_and_update(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            target = Path(temp_dir) / "ray_trainer.py"
            shutil.copy2(original_source(), target)
            result = self.run_script(target)
            self.assertEqual(result.returncode, 0, result.stderr)

            fit_source = target.read_text(encoding="utf-8").split("    def fit(self):", 1)[1]
            generation = fit_source.index("generate_sequences(combined_gen_batch)")
            reward_filter = fit_source.index("SHOPPING_AGENT_DYNAMIC_SAMPLING_BATCH")
            skipped = fit_source.index("SHOPPING_AGENT_DYNAMIC_SAMPLING_SKIPPED")
            ready = fit_source.index("SHOPPING_AGENT_DYNAMIC_SAMPLING_READY")
            sleep_before_training = fit_source.index(
                "self.checkpoint_manager.sleep_replicas()", ready
            )
            bypass = fit_source.index("apply_bypass_mode", sleep_before_training)
            reference = fit_source.index("self._compute_ref_log_prob(batch)", bypass)
            advantage = fit_source.index("batch = compute_advantage(", reference)
            update = fit_source.index("actor_output = self._update_actor(batch)", advantage)

            self.assertLess(generation, reward_filter)
            self.assertLess(reward_filter, skipped)
            self.assertLess(reward_filter, ready)
            self.assertLess(ready, sleep_before_training)
            self.assertLess(sleep_before_training, bypass)
            self.assertLess(bypass, reference)
            self.assertLess(reference, advantage)
            self.assertLess(advantage, update)
            self.assertIn(
                "if not dynamic_sampling_enabled:\n"
                "                            self.checkpoint_manager.sleep_replicas()",
                fit_source,
            )
            self.assertIn(
                "if bypass_recomputing_logprobs:  # Use `rollout_log_probs`",
                fit_source,
            )
            self.assertIn(
                "else:  # Recompute old_log_probs\n"
                "                        with marked_timer(\"old_log_prob\"",
                fit_source,
            )
            self.assertIn("extract_shopping_group_signals", fit_source)
            self.assertIn("aggregate_shopping_metrics", fit_source)
            self.assertIn("terminal_utilities=terminal_utilities", fit_source)
            self.assertIn("sampling_invalid=sampling_invalid", fit_source)
            self.assertIn('"drop_reason": group["drop_reason"]', fit_source)
            self.assertIn('"group/all_zero_utility_ratio"', fit_source)
            self.assertIn('"group/no_purchase_success_ratio"', fit_source)
            self.assertIn('"group/all_purchase_success_ratio"', fit_source)
            self.assertIn('"group/sampling_invalid"', fit_source)
            self.assertIn('"training/optimizer_updated": 0', fit_source)
            self.assertIn(
                "logger.log(data=skipped_metrics, step=self.global_steps)",
                fit_source,
            )
            self.assertIn(
                "dynamic_accepted_batches = []",
                fit_source[skipped:ready],
            )
            self.assertIn("dynamic_consecutive_skips", fit_source)
            self.assertIn(">= dynamic_max_consecutive_skips", fit_source)
            self.assertNotIn(
                "exhausted max_num_gen_batches=",
                fit_source,
            )
            self.assertLess(
                fit_source.index("SHOPPING_AGENT_DYNAMIC_SAMPLING_SKIPPED"),
                fit_source.index("self.checkpoint_manager.sleep_replicas()", ready),
            )

    def test_select_and_concat_keep_all_trajectory_fields_aligned(self):
        def make_batch(offset: int, uid_prefix: str) -> DataProto:
            row_ids = torch.arange(offset, offset + 8, dtype=torch.int64)
            uids = np.array([f"{uid_prefix}-drop"] * 4 + [f"{uid_prefix}-keep"] * 4)
            rewards = torch.tensor([0.0] * 4 + [2 / 7, 4 / 7, 2 / 7, 2 / 7])
            return DataProto.from_dict(
                tensors={
                    "responses": row_ids[:, None],
                    "response_mask": row_ids[:, None],
                    "rollout_log_probs": row_ids[:, None].float(),
                    "token_level_scores": rewards[:, None],
                    "rm_scores": rewards[:, None],
                    "attention_mask": row_ids[:, None],
                    "position_ids": row_ids[:, None],
                },
                non_tensors={
                    "uid": uids,
                    "extra_info": np.array(
                        [{"task_id": int(row_id)} for row_id in row_ids], dtype=object
                    ),
                    "shopping": np.array(
                        [
                            {
                                "infrastructure_invalid": False,
                                "reward": {
                                    "terminal_utility": float(reward),
                                    "purchase_success": bool(reward > 0),
                                    "sampling_invalid": False,
                                },
                            }
                            for reward in rewards
                        ],
                        dtype=object,
                    ),
                },
                meta_info={"reward_extra_keys": []},
            )

        selected_batches = []
        for batch in (make_batch(0, "a"), make_batch(8, "b")):
            rewards = batch.batch["rm_scores"].sum(dim=-1).tolist()
            utility, success, invalid, reasons = extract_shopping_group_signals(
                batch.non_tensor_batch["shopping"].tolist()
            )
            indices, _ = select_reward_varying_groups(
                batch.non_tensor_batch["uid"].tolist(),
                rewards,
                terminal_utilities=utility,
                purchase_success=success,
                sampling_invalid=invalid,
                sampling_invalid_reasons=reasons,
            )
            selected_batches.append(batch.select_idxs(indices))

        combined = DataProto.concat(selected_batches)
        expected_ids = [4, 5, 6, 7, 12, 13, 14, 15]
        self.assertEqual(combined.batch["responses"].flatten().tolist(), expected_ids)
        for key in (
            "response_mask",
            "rollout_log_probs",
            "attention_mask",
            "position_ids",
        ):
            self.assertEqual(combined.batch[key].flatten().tolist(), expected_ids)
        self.assertEqual(
            [item["task_id"] for item in combined.non_tensor_batch["extra_info"]],
            expected_ids,
        )
        self.assertEqual(
            combined.non_tensor_batch["uid"].tolist(),
            ["a-keep"] * 4 + ["b-keep"] * 4,
        )
        self.assertTrue(
            torch.allclose(
                combined.batch["token_level_scores"].flatten(),
                torch.tensor([2 / 7, 4 / 7, 2 / 7, 2 / 7] * 2),
            )
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
