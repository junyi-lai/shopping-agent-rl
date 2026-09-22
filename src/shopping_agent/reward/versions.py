"""Reward version selection shared by every reward consumer.

The pipeline runs on one reward contract at a time. All three versions share
**one code path** and differ only in (a) their complete ``reward.<version>``
block in ``pipeline.yaml`` and (b) the terminal value mapping in
``terminal.TERMINAL_VALUE_BY_VERSION``:

  - ``v4`` (frozen baseline): discrete terminal buckets + original process
    shaping. Collection, filtering, SFT and evaluation always use v4, so the
    strict success metric and the frozen datasets stay stable no matter which
    RL experiment is selected.
  - ``v5`` (RL-only experiment, measured no gain): denser evidence shaping, a
    small exploration bonus and a negative ``reward_unverifiable``; the terminal
    mapping is identical to v4.
  - ``v6`` (current champion, +5.5pp over v4): keeps v4's process shaping but
    scores every hard-gate-passing purchase with one continuous formula.

Selection travels through two environment variables that the pipeline entry
points export before launching veRL workers (which inherit the environment):

  - ``SHOPPING_AGENT_REWARD_VERSION``  : "v4" | "v5" | "v6"
  - ``SHOPPING_AGENT_REWARD_TERMINAL`` : JSON of terminal-value overrides
    (e.g. ``{"reward_unverifiable": -0.40}``).

This module imports nothing from the reward package, so it can be read by
``score.py``, the training adapter and the tests without a cycle.
"""

from __future__ import annotations

import json
import os

REWARD_VERSION_STRINGS = {
    "v4": "shopping-reward-v4",
    "v5": "shopping-reward-v5",
    "v6": "shopping-reward-v6",
}
NAME_BY_VERSION_STRING = {
    value: key for key, value in REWARD_VERSION_STRINGS.items()
}
DEFAULT_VERSION = "v4"
ENV_VERSION = "SHOPPING_AGENT_REWARD_VERSION"
ENV_TERMINAL = "SHOPPING_AGENT_REWARD_TERMINAL"


def version_name_for(version_string: str) -> str:
    """Map a version to its short name, accepting either form.

    ``"v6"`` and ``"shopping-reward-v6"`` both resolve to ``"v6"``, mirroring
    :func:`active_version_name` so callers never silently fall back to the
    default on a valid short name.
    """
    raw = str(version_string)
    if raw in REWARD_VERSION_STRINGS:
        return raw
    return NAME_BY_VERSION_STRING.get(raw, DEFAULT_VERSION)


def active_version_name() -> str:
    """Return the active version name ("v4" or "v5"), defaulting to v4.

    Accepts either the short name ("v5") or the full contract string
    ("shopping-reward-v5"), so a worker seeded from either form behaves the same.
    """

    raw = os.environ.get(ENV_VERSION, "").strip()
    name = raw or DEFAULT_VERSION
    if name in REWARD_VERSION_STRINGS:
        return name
    by_string = {value: key for key, value in REWARD_VERSION_STRINGS.items()}
    if name in by_string:
        return by_string[name]
    raise ValueError(f"unsupported reward version: {raw!r}")


def active_version_string() -> str:
    """Return the active contract string, e.g. "shopping-reward-v4"."""

    return REWARD_VERSION_STRINGS[active_version_name()]


def active_terminal_overrides() -> dict:
    """Return terminal-value overrides for the active version ({} for v4)."""

    raw = os.environ.get(ENV_TERMINAL, "").strip()
    if not raw:
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}
