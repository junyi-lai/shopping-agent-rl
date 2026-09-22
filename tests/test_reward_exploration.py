"""探索项(Reward v4 过程分第 4 子项)的行为测试。"""

from __future__ import annotations

import unittest

from shopping_agent.reward.contract import (
    count_distinct_opened_candidates,
    count_distinct_searches,
    compute_shaped_reward,
)

ON = {"exploration_cap": 0.05, "exploration_search_target": 3, "exploration_open_target": 2}
OFF = {"exploration_cap": 0.0}


def failed(total=-0.30, *, searches=0, opened=0, config=None):
    return compute_shaped_reward(
        total,
        purchase_success=False,
        steps=20,
        evidence_actions=0,
        repeat_action_count=0,
        distinct_searches=searches,
        distinct_opened_candidates=opened,
        reward_type="abstain",
        reward_valid=True,
        config=config,
    )


def succeeded(**kwargs):
    return compute_shaped_reward(
        1.0,
        purchase_success=True,
        steps=10,
        evidence_actions=2,
        repeat_action_count=0,
        reward_type="gold_purchase",
        reward_valid=True,
        **kwargs,
    )


class ExplorationTermTest(unittest.TestCase):
    def test_failure_that_explored_beats_failure_that_gave_up(self):
        idle = failed(searches=0, opened=0, config=ON)
        worked = failed(searches=3, opened=2, config=ON)
        self.assertGreater(worked["total"], idle["total"])
        self.assertEqual(idle["shaping"]["exploration_bonus"], 0.0)
        self.assertAlmostEqual(worked["shaping"]["exploration_bonus"], 0.05)

    def test_exploration_saturates_at_targets(self):
        at_target = failed(searches=3, opened=2, config=ON)
        beyond = failed(searches=9, opened=7, config=ON)
        self.assertEqual(at_target["total"], beyond["total"])

    def test_successful_trajectory_never_gets_exploration(self):
        shaped = succeeded(distinct_searches=5, distinct_opened_candidates=4, config=ON)
        self.assertEqual(shaped["shaping"]["exploration_bonus"], 0.0)
        self.assertGreater(shaped["shaping"]["efficiency_bonus"], 0.0)
        self.assertGreater(shaped["shaping"]["evidence_bonus"], 0.0)

    def test_disabled_by_default_is_unchanged(self):
        shaped = failed(searches=3, opened=2, config=OFF)
        self.assertEqual(shaped["shaping"]["exploration_bonus"], 0.0)
        self.assertEqual(shaped["total"], -0.30)

    def test_counts_read_the_trajectory_shape(self):
        steps = [
            {"tool_name": "search_products", "parameters": {"query": "木梳"}},
            {"tool_name": "search_products", "parameters": {"query": "木梳"}},
            {"tool_name": "search_products", "parameters": {"query": " 木梳 "}},
            {"tool_name": "search_products", "parameters": {"query": "卡通木梳"}},
            {"tool_name": "open_product", "parameters": {"asin": "A1"}},
            {"tool_name": "open_product", "parameters": {"asin": "A1"}},
            {"tool_name": "open_product", "parameters": {"asin": "A2"}},
            {"tool_name": "view_description", "parameters": {}},
        ]
        self.assertEqual(count_distinct_searches(steps), 2)
        self.assertEqual(count_distinct_opened_candidates(steps), 2)


if __name__ == "__main__":
    unittest.main()
