"""Reward v6: continuous terminal value for hard-gate-passing purchases.

v6 keeps v4's classification and process shaping; only the purchase **value**
changes to one continuous formula:

    ``reward = 0.5 * asin_match + match_score - 0.5``

That gives "right product but wrong variant" a continuous, informative signal
instead of v4's flat capped partial bucket, while ``reward_type`` (which drives
strict success) stays byte-identical to v4. Measured **+5.5pp over v4**
(76.5% vs 71.0% on the Final-200 blind set; the champion configuration).

Non-purchase outcomes (``wrong_purchase`` / ``reward_unverifiable``) keep the
discrete table values, exactly as in v4.
"""

from __future__ import annotations

from shopping_agent.reward.contract import PURCHASE_REWARD_TYPES

NAME = "v6"
CONTRACT = "shopping-reward-v6"


def terminal_value(reward_type, values, *, asin_match, match_score):
    """v6 终局取值:硬门全过的购买用连续公式;其余沿用离散值表。"""
    if reward_type in PURCHASE_REWARD_TYPES:
        return 0.5 * float(asin_match) + match_score - 0.5
    return values[reward_type]
