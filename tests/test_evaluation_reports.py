"""Evaluation aggregation and reporting contracts.

Covers the fixed-denominator Reward v4 summary, the benchmark CLI defaults, the
single-model HTML report and the cross-model comparison report.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shopping_agent.evaluation.benchmark import parse_args
from shopping_agent.evaluation.comparison_report import build_comparison_data
from shopping_agent.evaluation.report import build_data
from shopping_agent.evaluation.summary import summarize_trajectories


def _trajectory(task_id, strict=False, steps=3, status="done", blocked=None):
    reward_detail = {
        "reward_version": "shopping-reward-v4",
        "reward_type": "gold_purchase" if strict else "wrong_purchase",
        "reward_valid": True,
        "purchase_success": strict,
        "termination_reason": "gold_purchase" if strict else "wrong_purchase",
        "terminal_utility": 1.0 if strict else -0.85,
        "weighted_score": 1.0 if strict else 0.0,
    }
    return {
        "task_id": task_id,
        "status": status,
        "done": status == "done",
        "final_reward": 1.0 if strict else -0.85,
        "steps": [{"tool_name": "search_products"}] * steps,
        "blocked_tool_calls": blocked or [],
        "terminal_result": {"done": status == "done", "over": status == "done", "reward_detail": reward_detail},
    }


class BenchmarkTest(unittest.TestCase):
    def test_summary_uses_expected_tasks_as_v4_strict_success_denominator(self):
        """缺失或非严格成功 task 都计入失败，避免只统计已跑完的容易样本。"""
        summary = summarize_trajectories(
            expected_task_ids=[10, 11, 12],
            trajectories=[
                _trajectory(10, strict=True, steps=4),
                _trajectory(
                    11,
                    strict=False,
                    steps=6,
                    blocked=[{"reason": "schema_extra_arguments:asin"}],
                ),
            ],
        )

        self.assertEqual(summary["expected_tasks"], 3)
        self.assertEqual(summary["completed_tasks"], 2)
        self.assertEqual(summary["strict_successes"], 1)
        self.assertAlmostEqual(summary["strict_success_rate"], 1 / 3)
        self.assertEqual(summary["gold_purchases"], 1)
        self.assertAlmostEqual(summary["gold_purchase_rate"], 1 / 3)
        self.assertEqual(summary["reward_contract"], "shopping-reward-v4")
        self.assertEqual(summary["shaped_reward_version"], "shopping-reward-v4")
        self.assertEqual(summary["reward_type_counts"]["wrong_purchase"], 1)
        # Reward v4 = terminal + shaping. Gold (4 identical steps, no evidence)
        # gains an efficiency bonus minus a repeat penalty; wrong_purchase only
        # takes the repeat penalty. mean_terminal_utility stays raw Reward v4.
        self.assertAlmostEqual(summary["mean_terminal_utility"], (1.0 - 0.85) / 2)
        self.assertAlmostEqual(summary["mean_final_reward"], 0.039285714285714285)
        self.assertAlmostEqual(summary["shaping"]["efficiency_bonus"], 0.10 * 31 / 35 / 2)
        self.assertAlmostEqual(summary["shaping"]["repeat_penalty"], (0.06 + 0.10) / 2)
        self.assertEqual(summary["missing_tasks"], [12])
        self.assertAlmostEqual(summary["average_steps"], 5.0)
        self.assertEqual(summary["guard_reason_counts"]["schema_extra_arguments:asin"], 1)

    def test_summary_separates_successes_by_projection_bucket(self):
        projected = _trajectory(10, strict=True, steps=1)
        projected["steps"][0]["projection"] = {
            "truncated": True,
            "raw_tokens": 1000,
            "visible_tokens": 700,
            "visible_asin_count": 10,
            "visible_button_count": 12,
            "critical_footer_preserved": True,
        }
        projected["context_turn_tokens"] = [{"input_tokens": 17000}]
        projected["blocked_tool_calls"] = [
            {"reason": "click", "latest_observation_truncated": True}
        ]
        plain = _trajectory(11, strict=False, steps=1)

        summary = summarize_trajectories([10, 11], [projected, plain])
        projection = summary["context_projection"]

        self.assertEqual(projection["truncated_tool_observations"], 1)
        self.assertEqual(projection["guard_rejections_after_truncation"], 1)
        self.assertEqual(projection["max_context_input_tokens"], 17000)
        self.assertEqual(
            projection["success_by_truncation_bucket"]["any"],
            {"tasks": 1, "strict_successes": 1},
        )
        self.assertEqual(
            projection["success_by_truncation_bucket"]["none"],
            {"tasks": 1, "strict_successes": 0},
        )


class BenchmarkCliTest(unittest.TestCase):
    def test_evaluation_defaults_match_frozen_protocol(self):
        """Base、SFT、GRPO 必须默认使用同一 35 步上限。"""
        with patch.object(
            sys,
            "argv",
            [
                "evaluate_shop_benchmark.py",
                "--benchmark",
                "data/evaluation/tasks.jsonl",
                "--output",
                "outputs/eval/base/raw.jsonl",
                "--summary",
                "outputs/eval/base/summary.json",
                "--model",
                "openbmb/MiniCPM5-2B",
                "--llm-base-url",
                "http://127.0.0.1:8000/v1",
                "--api-key",
                "EMPTY",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.max_steps, 35)
        self.assertEqual(args.max_tokens, 512)
        self.assertEqual(args.temperature, 0.0)
        self.assertEqual(args.context_window, 24576)
        self.assertEqual(args.context_safety_margin, 512)
        self.assertFalse(args.context_compaction)
        self.assertEqual(args.observation_token_budget, 1536)
        self.assertEqual(args.observation_detail_token_budget, 4096)
        self.assertEqual(args.observation_generic_token_budget, 768)
        self.assertEqual(args.observation_search_top_k, 20)


class EvaluationReportTest(unittest.TestCase):
    def test_report_reads_any_evaluation_directory_and_model_name(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            summary = {
                "completed_tasks": 1,
                "done_tasks": 1,
                "done_rate": 1.0,
                "strict_successes": 1,
                "strict_success_rate": 1.0,
                "strict_success_task_ids": [42],
                "purchase_successes": 1,
                "purchase_success_rate": 1.0,
                "mean_final_reward": 1.0,
                "mean_weighted_score": 1.0,
                "average_steps": 1.0,
                "protocol": {"model": "new-model-1"},
            }
            trajectory = {
                "task_id": 42,
                "status": "done",
                "done": True,
                "final_reward": 1.0,
                "steps": [],
                "blocked_tool_calls": [],
                "initial_result": {"instruction": "test"},
                "terminal_result": {
                    "reward_detail": {"reward_type": "gold_purchase", "purchase_success": True},
                    "termination_reason": "gold_purchase",
                },
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "trajectories.jsonl").write_text(
                json.dumps(trajectory) + "\n", encoding="utf-8"
            )

            data = build_data(run_dir)

            self.assertEqual(data["meta"]["model"], "new-model-1")
            self.assertEqual(data["summary"]["total"], 1)

    def test_report_includes_early_abstain_and_guard_per_task(self):
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            summary = {
                "completed_tasks": 1,
                "done_tasks": 1,
                "done_rate": 1.0,
                "strict_successes": 0,
                "strict_success_rate": 0.0,
                "strict_success_task_ids": [],
                "purchase_successes": 0,
                "purchase_success_rate": 0.0,
                "mean_final_reward": 0.0,
                "mean_weighted_score": 0.0,
                "average_steps": 1.0,
                "protocol": {"model": "new-model-1"},
            }
            trajectory = {
                "task_id": 42,
                "status": "done",
                "done": True,
                "final_reward": 0.0,
                "steps": [],
                "blocked_tool_calls": [{"reason": "test_guard"}],
                "initial_result": {"instruction": "test"},
                "terminal_result": {
                    "reward_detail": {
                        "reward_type": "early_abstain",
                        "purchase_success": False,
                    },
                    "termination_reason": "early_abstain",
                },
            }
            (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
            (run_dir / "trajectories.jsonl").write_text(
                json.dumps(trajectory) + "\n", encoding="utf-8"
            )

            data = build_data(run_dir)

            early_abstain = next(
                item for item in data["charts"]["outcomes"]
                if item["key"] == "early_abstain"
            )
            self.assertEqual(early_abstain["value"], 1)
            self.assertEqual(data["summary"]["guard_per_task"], 1.0)


class ComparisonReportTest(unittest.TestCase):
    def _write_run(self, run, protocol_model):
        run.mkdir(parents=True)
        (run / "summary.json").write_text(
            json.dumps(
                {
                    "completed_tasks": 2,
                    "strict_successes": 1,
                    "strict_success_rate": 0.5,
                    "purchase_successes": 1,
                    "purchase_success_rate": 0.5,
                    "reward_valid_tasks": 2,
                    "reward_valid_rate": 1.0,
                    "average_steps": 3.5,
                    "reward_type_counts": {"gold_purchase": 1, "repeat_loop": 1},
                    "protocol": {"model": protocol_model},
                }
            ),
            encoding="utf-8",
        )
        rows = [
            {"task_id": 1, "final_reward": 1.0},
            {"task_id": 2, "final_reward": -0.5},
        ]
        (run / "trajectories.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )

    def test_discovers_runs_and_computes_reward_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_run(root / "model-a", "Model A")

            data = build_comparison_data(root)

            self.assertEqual([model["key"] for model in data["models"]], ["model-a"])
            self.assertEqual(data["models"][0]["reward"]["mean"], 0.25)
            self.assertEqual(data["models"][0]["reward"]["median"], 0.25)
            self.assertEqual(sum(data["models"][0]["reward"]["histogram"]), 2)

    def test_labels_come_from_run_directories_and_nested_runs_are_found(self):
        """回归:展示名不能取协议里的 ``model``(那是所有模型共用的 vLLM 服务名)。

        同时也验证分层存放(``works/eval/rl/<label>``)能被递归发现,且报告链接嵌一层。
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            # 两个 run 的协议名相同——曾经导致对比报告 8 行标签全是 shopping-agent。
            self._write_run(root / "base", "shopping-agent")
            self._write_run(root / "rl" / "rloo_v6_lr3e-5_200step", "shopping-agent")

            data = build_comparison_data(root)

            self.assertEqual(
                [model["key"] for model in data["models"]],
                ["base", "rl/rloo_v6_lr3e-5_200step"],
            )
            self.assertEqual(
                [model["name"] for model in data["models"]],
                ["base", "rloo_v6_lr3e-5_200step"],
            )
            self.assertEqual(
                [model["report"] for model in data["models"]],
                ["base/report.html", "rl/rloo_v6_lr3e-5_200step/report.html"],
            )
            self.assertIn(data["best_model"], {"base", "rloo_v6_lr3e-5_200step"})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
