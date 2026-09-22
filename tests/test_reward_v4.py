"""Reward v4 process-shaping unit tests."""

import unittest

from shopping_agent.reward import (
    REWARD_VERSION,
    compute_shaped_reward,
    count_consecutive_duplicate_actions,
    count_evidence_actions,
    is_strict_success,
    score_reward_detail,
)


def _step(tool, **params):
    return {"tool_name": tool, "parameters": params}


class RewardV4Test(unittest.TestCase):
    def test_gold_purchase_gains_efficiency_and_evidence(self):
        shaped = compute_shaped_reward(
            1.0,
            purchase_success=True,
            steps=10,
            evidence_actions=2,
            repeat_action_count=0,
            reward_type="gold_purchase",
            reward_valid=True,
        )
        self.assertEqual(shaped["version"], REWARD_VERSION)
        self.assertTrue(shaped["success"])
        self.assertAlmostEqual(shaped["shaping"]["efficiency_bonus"], 0.10 * 25 / 35)
        self.assertAlmostEqual(shaped["shaping"]["evidence_bonus"], 0.10)
        self.assertAlmostEqual(shaped["shaping"]["repeat_penalty"], 0.0)
        self.assertAlmostEqual(shaped["total"], 1.0 + 0.10 * 25 / 35 + 0.10)

    def test_non_purchase_gets_no_efficiency_or_evidence(self):
        shaped = compute_shaped_reward(
            -0.85,
            purchase_success=False,
            steps=5,
            evidence_actions=3,
            repeat_action_count=0,
            reward_type="wrong_purchase",
            reward_valid=True,
        )
        self.assertFalse(shaped["success"])
        self.assertEqual(shaped["shaping"]["efficiency_bonus"], 0.0)
        self.assertEqual(shaped["shaping"]["evidence_bonus"], 0.0)
        self.assertAlmostEqual(shaped["total"], -0.85)

    def test_evidence_bonus_scales_with_verification_actions(self):
        half = compute_shaped_reward(
            1.0, purchase_success=True, steps=5, evidence_actions=1,
            repeat_action_count=0,
        )
        full = compute_shaped_reward(
            1.0, purchase_success=True, steps=5, evidence_actions=2,
            repeat_action_count=0,
        )
        self.assertAlmostEqual(half["shaping"]["evidence_bonus"], 0.05)
        self.assertAlmostEqual(full["shaping"]["evidence_bonus"], 0.10)

    def test_repeat_penalty_is_capped(self):
        shaped = compute_shaped_reward(
            1.0, purchase_success=True, steps=1, evidence_actions=0,
            repeat_action_count=100,
        )
        self.assertAlmostEqual(shaped["shaping"]["repeat_penalty"], 0.20)

    def test_valid_alternative_is_not_strict_success(self):
        self.assertTrue(is_strict_success("gold_purchase", True, True))
        self.assertFalse(is_strict_success("valid_alternative_purchase", True, True))
        self.assertFalse(is_strict_success("gold_purchase", False, True))

    def test_score_reward_detail_reads_environment_facts(self):
        detail = {
            "reward_version": "shopping-reward-v4",
            "reward_type": "gold_purchase",
            "reward_valid": True,
            "purchase_success": True,
            "terminal_utility": 1.0,
        }
        steps = [
            _step("search_products", query="a"),
            _step("open_product", asin="1"),
            _step("view_features"),
            _step("buy_now"),
        ]
        result = score_reward_detail(detail, steps=steps)
        self.assertTrue(result["success"])
        self.assertEqual(result["version"], REWARD_VERSION)
        self.assertAlmostEqual(result["shaping"]["efficiency_bonus"], 0.10 * 31 / 35)
        self.assertAlmostEqual(result["shaping"]["evidence_bonus"], 0.05)

    def test_count_evidence_actions_ignores_non_view_tools(self):
        steps = [
            _step("search_products", query="a"),
            _step("view_description"),
            _step("select_option", value="x"),
            _step("view_attributes"),
            _step("buy_now"),
        ]
        self.assertEqual(count_evidence_actions(steps), 2)

    def test_consecutive_duplicate_count_skips_think(self):
        steps = [
            _step("search_products", query="a"),
            _step("search_products", query="a"),
            _step("think", note="x"),
            _step("open_product", asin="1"),
            _step("open_product", asin="1"),
            _step("open_product", asin="2"),
        ]
        self.assertEqual(count_consecutive_duplicate_actions(steps), 2)


if __name__ == "__main__":
    unittest.main()
