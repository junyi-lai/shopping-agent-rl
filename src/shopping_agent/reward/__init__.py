"""Environment-independent reward package.

The reward function lives here, fully decoupled from the embedded ShopSimulator.
Training, evaluation and collection all consume it through ``score_reward_detail``.
"""

from shopping_agent.reward.contract import (
    DEFAULT_SHAPING_CONFIG,
    REWARD_VERSION,
    compute_shaped_reward,
    count_consecutive_duplicate_actions,
    count_evidence_actions,
    is_strict_success,
)
from shopping_agent.reward.score import score_reward_detail
from shopping_agent.reward.versions import (
    DEFAULT_VERSION,
    REWARD_VERSION_STRINGS,
    active_terminal_overrides,
    active_version_name,
    active_version_string,
)

__all__ = [
    "DEFAULT_SHAPING_CONFIG",
    "DEFAULT_VERSION",
    "REWARD_VERSION",
    "REWARD_VERSION_STRINGS",
    "active_terminal_overrides",
    "active_version_name",
    "active_version_string",
    "compute_shaped_reward",
    "count_consecutive_duplicate_actions",
    "count_evidence_actions",
    "is_strict_success",
    "score_reward_detail",
]
