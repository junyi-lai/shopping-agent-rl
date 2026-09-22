"""Frozen ShopSimulator Environment v2.1 contract."""

from __future__ import annotations

import json

from shopping_agent import config as project_config


MANIFEST_VERSION = "shopping-environment-manifest-v1"
MANIFEST_PATH = "configs/environment.json"
RUNTIME_CONFIG_PATH = "environments/ShopSimulator/shop_env/configs/environment.json"
REQUIRED_KEYS = {
    "manifest_version",
    "shopsimulator_commit",
    "search",
    "reward",
    "observation_version",
    "tool_version",
    "max_steps",
    "seed",
}


def validate_manifest(manifest):
    if not isinstance(manifest, dict):
        raise ValueError("environment manifest must be an object")
    missing = REQUIRED_KEYS - set(manifest)
    if missing:
        raise ValueError(
            "environment manifest is missing: " + ", ".join(sorted(missing))
        )
    if manifest["manifest_version"] != MANIFEST_VERSION:
        raise ValueError("unsupported environment manifest version")
    if manifest["observation_version"] != "shopping-observation-v2":
        raise ValueError("manifest does not select Observation v2")
    if manifest["tool_version"] != "shopping-tools-v2":
        raise ValueError("manifest does not select Tool v2")
    environment_version = manifest.get(
        "environment_version",
        "shopsimulator-environment-v2.1",
    )
    if environment_version != "shopsimulator-environment-v2.1":
        raise ValueError("manifest has an unsupported environment_version")
    if manifest["reward"].get("version") != "shopping-reward-v4":
        raise ValueError(
            "shopsimulator-environment-v2.1 requires shopping-reward-v4"
        )
    if manifest["search"].get("version") != "shopsimulator-multifield-bm25-v2":
        raise ValueError("manifest does not select multi-field BM25 v2")
    if int(manifest["search"].get("page_size", 0)) != 20:
        raise ValueError("Environment v2 page_size must equal 20")
    if int(manifest["max_steps"]) <= 0:
        raise ValueError("max_steps must be positive")
    commit = manifest["shopsimulator_commit"]
    if (
        not isinstance(commit, str)
        or len(commit) != 40
        or any(character not in "0123456789abcdef" for character in commit)
    ):
        raise ValueError("manifest shopsimulator_commit is not a lowercase Git SHA")
    return manifest


def load_manifest() -> dict:
    """Read and validate the frozen manifest committed in ``configs/``."""
    path = project_config.project_path(MANIFEST_PATH)
    if not path.is_file():
        raise ValueError(f"environment manifest is missing: {path}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid environment manifest {path}: {exc}") from exc
    return validate_manifest(manifest)


def load_runtime_config() -> dict:
    """Read the environment's own runtime config shipped inside the snapshot."""
    path = project_config.project_path(RUNTIME_CONFIG_PATH)
    if not path.is_file():
        raise ValueError(f"environment runtime config is missing: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid environment runtime config {path}: {exc}") from exc


def validate_runtime_config(manifest, runtime_config) -> dict:
    """Fail when the frozen manifest and the environment's runtime config drift.

    The manifest (`configs/environment.json`) is what the project pins; the
    snapshot also ships its own copy under ``environments/ShopSimulator``. Both
    describe the same environment, so every shared field must agree.
    """

    if not isinstance(runtime_config, dict):
        raise ValueError("environment runtime config must be an object")
    termination = runtime_config.get("termination") or {}
    search = runtime_config.get("search") or {}
    pairs = [
        ("environment_version", manifest.get("environment_version"), runtime_config.get("environment_version")),
        ("observation_version", manifest["observation_version"], runtime_config.get("observation_version")),
        ("tool_version", manifest["tool_version"], runtime_config.get("tool_version")),
        ("max_steps", int(manifest["max_steps"]), int(termination.get("max_steps", 0))),
        ("search.version", manifest["search"].get("version"), search.get("version")),
        ("search.top_k", int(manifest["search"].get("top_k", 0)), int(search.get("top_k", 0))),
        ("search.page_size", int(manifest["search"].get("page_size", 0)), int(search.get("page_size", 0))),
    ]
    mismatched = [
        f"{name}: manifest={expected!r} runtime={actual!r}"
        for name, expected, actual in pairs
        if expected != actual
    ]
    if manifest["search"].get("field_weights") != search.get("field_weights"):
        mismatched.append("search.field_weights differ between manifest and runtime config")
    if mismatched:
        raise ValueError(
            "environment runtime config disagrees with the frozen manifest: "
            + "; ".join(mismatched)
        )
    return runtime_config
