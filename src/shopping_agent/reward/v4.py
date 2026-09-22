"""Reward v4: the frozen baseline — discrete terminal buckets.

v4 is the contract the project is frozen on: collection, filtering, SFT and
evaluation all score with it, so the strict-success metric and the frozen
datasets stay stable no matter which RL reward experiment is selected.

Only the terminal *value* lives here (the discrete buckets: gold 1.0 /
alternative 0.55 / partial capped at 0.25 ...). Process shaping, the
strict-success gate and the frozen contract string are version-independent and
live in ``contract.py``, shared by ``v4.py`` / ``v5.py`` / ``v6.py``.
"""

from __future__ import annotations

from shopping_agent.reward.contract import REWARD_VERSION

NAME = "v4"
CONTRACT = REWARD_VERSION


def terminal_value(reward_type, values, *, asin_match, match_score):
    """离散终局桶;partial 按 match_score 线性插值后封顶。

    ``values`` 是终局值表(``terminal.DEFAULT_REWARDS`` 经版本覆盖后的结果)。
    """
    if reward_type == "partial_alternative_purchase":
        return min(
            values["partial_purchase_cap"],
            values["partial_purchase_base"]
            + values["partial_purchase_scale"] * match_score,
        )
    return values[reward_type]
