"""Reward v5 (opt-in) selection and behaviour tests.

v5 keeps every v4 terminal bucket and only changes: the ``reward_unverifiable``
terminal value, the evidence/repeat shaping density and the exploration bonus on
non-purchase trajectories. v4 remains the default; these tests pin the version
explicitly so they never mutate the frozen v4 contract.
"""

import json
import os
import unittest

from shopping_agent.reward import (
    active_terminal_overrides,
    active_version_name,
    active_version_string,
    compute_shaped_reward,
)
from shopping_agent.reward.score import compute_terminal_reward_detail
from shopping_agent.training.rl.adapter.runtime import (
    make_runtime_state,
    reward_breakdown,
    validate_reward,
)

ENV_VERSION = "SHOPPING_AGENT_REWARD_VERSION"
ENV_TERMINAL = "SHOPPING_AGENT_REWARD_TERMINAL"
ENV_CONFIG = "SHOPPING_AGENT_REWARD_CONFIG"


def _gold_terminal_result():
    return {
        "instruction": "terminal",
        "done": True,
        "over": True,
        "reward": 1.0,
        "termination_reason": "purchase",
        "purchased_product": {
            "asin": "B0GOLD0001",
            "title": "乳胶枕",
            "category": "家居›床上用品›枕头",
            "pricing": [{"price": 99.0}],
            "attribute": [],
            "small_description": [],
        },
        "goal": {
            "asin": "B0GOLD0001",
            "instruction_text": "买一个乳胶枕，预算在200元以下",
            "goal_options": [],
            "attributes": [],
        },
        "target_product": {
            "asin": "B0GOLD0001",
            "title": "乳胶枕",
            "category": "家居›床上用品›枕头",
            "pricing": [{"price": 99.0}],
        },
        "options": {},
        "price_resolution": {
            "status": "pass",
            "price": 99.0,
            "version": "variant-price-v1",
            "method": "explicit_test_price",
            "evidence": {},
        },
    }


def _unverifiable_terminal_result():
    result = _gold_terminal_result()
    result["purchased_product"] = dict(result["purchased_product"], pricing=[])
    result["price_resolution"] = {"status": "unverifiable", "price": None}
    return result


def _env_guard():
    """Save-and-remove the reward env vars, restoring them on teardown.

    The reward version/shaping travel through ``os.environ``, so a test that
    writes them must remove them afterwards or they leak into later tests (which
    run in the same process and expect the v4 default).
    """

    saved = {}
    for key in (ENV_VERSION, ENV_TERMINAL, ENV_CONFIG):
        if key in os.environ:
            saved[key] = os.environ[key]
            del os.environ[key]
    return saved


class RewardVersionSelectionTest(unittest.TestCase):
    def setUp(self):
        self._saved = _env_guard()

    def tearDown(self):
        for key in (ENV_VERSION, ENV_TERMINAL, ENV_CONFIG):
            os.environ.pop(key, None)
        os.environ.update(self._saved)

    def test_defaults_to_v4(self):
        self.assertEqual(active_version_name(), "v4")
        self.assertEqual(active_version_string(), "shopping-reward-v4")
        self.assertEqual(active_terminal_overrides(), {})

    def test_selects_v5_by_short_name(self):
        os.environ[ENV_VERSION] = "v5"
        self.assertEqual(active_version_name(), "v5")
        self.assertEqual(active_version_string(), "shopping-reward-v5")

    def test_accepts_full_contract_string(self):
        os.environ[ENV_VERSION] = "shopping-reward-v5"
        self.assertEqual(active_version_name(), "v5")

    def test_rejects_unknown_version(self):
        os.environ[ENV_VERSION] = "v9"
        with self.assertRaisesRegex(ValueError, "unsupported reward version"):
            active_version_string()

    def test_reads_terminal_overrides_from_environment(self):
        os.environ[ENV_TERMINAL] = json.dumps({"reward_unverifiable": -0.40})
        self.assertEqual(
            active_terminal_overrides(), {"reward_unverifiable": -0.40}
        )


class RewardV5BehaviourTest(unittest.TestCase):
    def setUp(self):
        self._saved = _env_guard()

    def tearDown(self):
        for key in (ENV_VERSION, ENV_TERMINAL, ENV_CONFIG):
            os.environ.pop(key, None)
        os.environ.update(self._saved)

    def test_terminal_emits_v5_and_fixes_unverifiable_value(self):
        detail = compute_terminal_reward_detail(
            _unverifiable_terminal_result(),
            version="shopping-reward-v5",
            terminal_overrides={"reward_unverifiable": -0.40},
        )
        self.assertEqual(detail["reward_version"], "shopping-reward-v5")
        self.assertEqual(detail["reward_type"], "reward_unverifiable")
        self.assertFalse(detail["reward_valid"])
        self.assertEqual(detail["terminal_utility"], -0.40)

    def test_v4_terminal_still_defaults_to_zero_for_unverifiable(self):
        detail = compute_terminal_reward_detail(_unverifiable_terminal_result())
        self.assertEqual(detail["reward_version"], "shopping-reward-v4")
        self.assertEqual(detail["terminal_utility"], 0.0)

    def test_v5_shaping_densifies_evidence(self):
        v5_shaping = {
            "evidence_cap": 0.15,
            "evidence_threshold": 4,
            "exploration_cap": 0.10,
            "repeat_per_action": 0.03,
            "repeat_cap": 0.25,
        }
        shaped = compute_shaped_reward(
            1.0,
            purchase_success=True,
            steps=5,
            evidence_actions=2,
            repeat_action_count=0,
            config=v5_shaping,
        )
        # evidence = 0.15 * min(1, 2/4) = 0.075 (v4 would be 0.10 at threshold 2)
        self.assertAlmostEqual(shaped["shaping"]["evidence_bonus"], 0.075)

    def test_v5_enables_exploration_on_failed_trajectories(self):
        v5_shaping = {
            "evidence_cap": 0.15,
            "evidence_threshold": 4,
            "exploration_cap": 0.10,
            "repeat_per_action": 0.03,
            "repeat_cap": 0.25,
        }
        shaped = compute_shaped_reward(
            -0.15,
            purchase_success=False,
            steps=5,
            evidence_actions=0,
            repeat_action_count=0,
            distinct_searches=3,
            distinct_opened_candidates=2,
            config=v5_shaping,
        )
        # exploration = 0.10 * 0.5 * (min(1,3/3) + min(1,2/2)) = 0.10
        self.assertAlmostEqual(shaped["shaping"]["exploration_bonus"], 0.10)
        self.assertAlmostEqual(shaped["total"], -0.15 + 0.10)

    def test_validate_reward_and_breakdown_accept_v5(self):
        os.environ[ENV_VERSION] = "v5"
        os.environ[ENV_CONFIG] = json.dumps(
            {
                "evidence_cap": 0.15,
                "evidence_threshold": 4,
                "exploration_cap": 0.10,
                "repeat_per_action": 0.03,
                "repeat_cap": 0.25,
            }
        )
        detail = compute_terminal_reward_detail(_gold_terminal_result())
        self.assertEqual(detail["reward_version"], "shopping-reward-v5")
        public = validate_reward(detail)
        self.assertEqual(public["reward_version"], "shopping-reward-v5")

        state = make_runtime_state(task_id=1, max_steps=35)
        state["done"] = True
        state["terminal_result"] = {"done": True, "over": True}
        state["reward_version"] = public["reward_version"]
        state["reward_type"] = public["reward_type"]
        state["reward_valid"] = public["reward_valid"]
        state["reward_unverifiable"] = not public["reward_valid"]
        state["reward_detail"] = public
        state["final_reward"] = public["terminal_utility"]
        state["steps"] = [
            {"tool_name": "view_features"},
            {"tool_name": "view_attributes"},
            {"tool_name": "buy_now"},
        ]
        breakdown = reward_breakdown(state)
        self.assertEqual(breakdown["shaped_reward_version"], "shopping-reward-v5")
        self.assertTrue(breakdown["success"])
        # evidence = 0.15 * min(1, 2/4) = 0.075
        self.assertAlmostEqual(breakdown["shaping"]["evidence_bonus"], 0.075)


if __name__ == "__main__":
    unittest.main()
