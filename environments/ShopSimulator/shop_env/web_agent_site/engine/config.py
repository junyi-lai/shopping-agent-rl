"""Load and validate the minimal ShopSimulator environment config.

The environment is a pure simulator: the config only carries search, liveness
termination and version markers. All reward logic lives in the project's
``shopping_agent.reward`` package.
"""

from __future__ import annotations

import json
from pathlib import Path

from web_agent_site.engine.observation import OBSERVATION_VERSION
from web_agent_site.engine.search import DEFAULT_FIELD_WEIGHTS, SEARCH_VERSION


ENVIRONMENT_VERSION = "shopsimulator-environment-v2.1"
TOOL_VERSION = "shopping-tools-v2"
SEARCH_TOP_K = 150
SEARCH_PAGE_SIZE = 20
_POSITIVE_TERMINATION_FIELDS = (
    "exact_repeat_limit",
    "max_steps",
)


def load_config(path):
    config_path = Path(path).resolve()
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"cannot load environment config {config_path}: {exc}"
        ) from exc
    validate_config(config)
    return config


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("environment config must be an object")
    if config.get("environment_version") != ENVIRONMENT_VERSION:
        raise ValueError("environment config has the wrong environment_version")
    search = config.get("search")
    if not isinstance(search, dict) or search.get("version") != SEARCH_VERSION:
        raise ValueError("environment config has the wrong search version")
    if int(search.get("top_k", 0)) != SEARCH_TOP_K:
        raise ValueError(f"environment search top_k must equal {SEARCH_TOP_K}")
    if int(search.get("page_size", 0)) != SEARCH_PAGE_SIZE:
        raise ValueError(
            f"environment search page_size must equal {SEARCH_PAGE_SIZE}"
        )
    if search.get("field_weights") != DEFAULT_FIELD_WEIGHTS:
        raise ValueError(
            "environment search field weights differ from the index contract"
        )
    termination = config.get("termination")
    if not isinstance(termination, dict):
        raise ValueError("environment config is missing termination")
    for name in _POSITIVE_TERMINATION_FIELDS:
        if int(termination.get(name, 0)) <= 0:
            raise ValueError(f"environment termination.{name} must be positive")
    expected_versions = {
        "observation_version": OBSERVATION_VERSION,
        "tool_version": TOOL_VERSION,
    }
    for name, expected in expected_versions.items():
        if config.get(name) != expected:
            raise ValueError(
                f"environment config has the wrong {name}: expected {expected!r}"
            )
    return config
