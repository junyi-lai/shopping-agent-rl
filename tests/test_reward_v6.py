"""Reward v6 (continuous terminal reward) tests.

v6 keeps the reward_type classification and strict-success gate unchanged, but
replaces the discrete gold/alternative/partial buckets with one continuous
formula for every hard-gate-passing purchase:

    reward = 0.5 * asin_match + match_score - 0.5

so "right product but wrong variant" earns a continuous, informative signal.
"""

import json
import os
import unittest

from shopping_agent.reward import active_version_name, active_version_string
from shopping_agent.reward.terminal import evaluate_purchase
from shopping_agent.reward.score import compute_terminal_reward_detail

ENV_VERSION = "SHOPPING_AGENT_REWARD_VERSION"
ENV_TERMINAL = "SHOPPING_AGENT_REWARD_TERMINAL"
ENV_CONFIG = "SHOPPING_AGENT_REWARD_CONFIG"

_CATEGORY = "家居›床上用品›枕头"
_PASS_PRICE = {
    "status": "pass",
    "price": 99.0,
    "version": "variant-price-v1",
    "method": "explicit_test_price",
    "evidence": {},
}


def _goal(asin, category=_CATEGORY, expected_brand=()):
    return {
        "asin": asin,
        "category": category,
        "price_upper": None,  # 无预算声明 → budget 硬门 pass
        "expected_brand": list(expected_brand),
        "expected_model": [],
        "expected_core_functions": [],
        "required_options_by_key": {},
        "unresolved_option_requirements": [],
    }


def _product(asin, brand="", category=_CATEGORY):
    return {
        "asin": asin,
        "Title": "乳胶枕",
        "Description": "",
        "BulletPoints": [],
        "Attributes": [],
        "brand": brand,
        "shop_name": "",
        "category": category,
        "pricing": [{"price": 99.0}],
    }


def _gold_terminal_result():
    return {
        "instruction": "terminal",
        "done": True,
        "over": True,
        "reward": 1.0,
        "termination_reason": "purchase",
        "purchased_product": {
            "asin": "B1",
            "title": "乳胶枕",
            "category": _CATEGORY,
            "pricing": [{"price": 99.0}],
            "attribute": [],
            "small_description": [],
        },
        "goal": {
            "asin": "B1",
            "instruction_text": "买一个乳胶枕，预算在200元以下",
            "goal_options": [],
            "attributes": [],
        },
        "target_product": {
            "asin": "B1",
            "title": "乳胶枕",
            "category": _CATEGORY,
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


class RewardV6FormulaTest(unittest.TestCase):
    def test_gold_is_1_0(self):
        r = evaluate_purchase(
            _product("B1"), _goal("B1"),
            selected_options={}, price_resolution=_PASS_PRICE, version="v6",
        )
        self.assertEqual(r.reward_type, "gold_purchase")
        self.assertEqual(r.reward, 1.0)

    def test_alternative_is_0_5(self):
        r = evaluate_purchase(
            _product("B2"), _goal("B1"),
            selected_options={}, price_resolution=_PASS_PRICE, version="v6",
        )
        self.assertEqual(r.reward_type, "valid_alternative_purchase")
        self.assertEqual(r.reward, 0.5)

    def test_partial_with_asin_match_is_continuous(self):
        # 品牌一半命中 → match_score 0.5,asin 命中 → 0.5*1 + 0.5 - 0.5 = 0.5
        r = evaluate_purchase(
            _product("B1", brand="匹配牌"),
            _goal("B1", expected_brand=["匹配牌", "不匹配牌"]),
            selected_options={}, price_resolution=_PASS_PRICE, version="v6",
        )
        self.assertEqual(r.reward_type, "partial_alternative_purchase")
        self.assertAlmostEqual(r.weighted_score, 0.5)
        self.assertAlmostEqual(r.reward, 0.5 * 1.0 + r.weighted_score - 0.5)

    def test_wrong_asin_low_match_is_negative(self):
        # 买错商品 + 品牌全不匹配 → 0.5*0 + 0 - 0.5 = -0.5,比放弃(-0.15)更差
        r = evaluate_purchase(
            _product("B2"), _goal("B1", expected_brand=["某品牌"]),
            selected_options={}, price_resolution=_PASS_PRICE, version="v6",
        )
        self.assertAlmostEqual(r.reward, -0.5)

    def test_v4_default_is_unchanged(self):
        # 不传 version → v4: alternative 仍是 0.55
        r = evaluate_purchase(
            _product("B2"), _goal("B1"),
            selected_options={}, price_resolution=_PASS_PRICE,
        )
        self.assertEqual(r.reward_type, "valid_alternative_purchase")
        self.assertEqual(r.reward, 0.55)


class RewardV6WiringTest(unittest.TestCase):
    def setUp(self):
        self._saved = {
            key: os.environ.pop(key)
            for key in (ENV_VERSION, ENV_TERMINAL, ENV_CONFIG)
            if key in os.environ
        }

    def tearDown(self):
        for key in (ENV_VERSION, ENV_TERMINAL, ENV_CONFIG):
            os.environ.pop(key, None)
        os.environ.update(self._saved)

    def test_version_string_registered(self):
        os.environ[ENV_VERSION] = "v6"
        self.assertEqual(active_version_name(), "v6")
        self.assertEqual(active_version_string(), "shopping-reward-v6")

    def test_compute_terminal_reward_detail_uses_v6(self):
        os.environ[ENV_VERSION] = "v6"
        os.environ[ENV_TERMINAL] = json.dumps({"reward_unverifiable": -0.40})
        detail = compute_terminal_reward_detail(_gold_terminal_result())
        self.assertEqual(detail["reward_version"], "shopping-reward-v6")
        self.assertEqual(detail["reward_type"], "gold_purchase")
        self.assertEqual(detail["terminal_utility"], 1.0)


if __name__ == "__main__":
    unittest.main()
