"""Reward v5: v4's terminal mapping with denser process shaping.

v5 changes **no terminal logic**. It differs from v4 purely through its complete
``reward.v5`` block in ``configs/pipeline.yaml``:

  - denser evidence shaping (``evidence_cap`` 0.10 → 0.15, threshold 2 → 4);
  - a stronger repeat penalty (0.02 → 0.03 per action, cap 0.20 → 0.25);
  - an exploration bonus on non-purchase trajectories (``exploration_cap`` → 0.10);
  - a negative ``reward_unverifiable`` (0.0 → -0.40).

Measured no gain over v4 (69.0% vs 69.5% on the Final-200 blind set), but the
version stays selectable so that experiment remains reproducible.

The terminal value mapping is v4's, imported verbatim, so the two cannot drift
apart.
"""

from __future__ import annotations

from shopping_agent.reward.v4 import terminal_value

NAME = "v5"
CONTRACT = "shopping-reward-v5"
