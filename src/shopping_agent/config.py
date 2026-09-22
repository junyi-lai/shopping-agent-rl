"""Unified configuration loading for the pipeline entry points.

``configs/pipeline.yaml`` holds the global settings (models, directories, ports,
reward shaping) and ``configs/<stage>.yaml`` holds one stage.  Every stage file
exposes an ``args`` mapping whose keys mirror the underlying command-line options;
:func:`stage_argv` turns that mapping into the ``argv`` a stage module consumes,
so a stage never has to know where its settings came from.
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "configs"
PIPELINE_CONFIG = CONFIG_DIR / "pipeline.yaml"
ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Load ``.env`` into ``os.environ`` without overriding real environment variables."""

    dotenv_path = resolve_path(path) if path is not None else PROJECT_ROOT / ".env"
    loaded: dict[str, str] = {}
    if not dotenv_path.is_file():
        return loaded
    for line in dotenv_path.read_text(encoding="utf-8-sig").splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#") or "=" not in entry:
            continue
        key, _, value = entry.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        loaded[key] = value
        os.environ.setdefault(key, value)
    return loaded


def bootstrap() -> Path:
    """Make ``src`` importable for this process and its children, then cd to the repo root."""

    source_root = PROJECT_ROOT / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    parts = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    if str(source_root) not in parts:
        os.environ["PYTHONPATH"] = os.pathsep.join([str(source_root), *parts])
    os.environ.setdefault("SHOPPING_AGENT_ROOT", str(PROJECT_ROOT))
    os.chdir(PROJECT_ROOT)
    load_dotenv()
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:  # Windows 控制台默认 GBK,统一按 UTF-8 输出中文日志
                reconfigure(encoding="utf-8", errors="replace")
            except (ValueError, OSError):
                pass
    return PROJECT_ROOT


def project_path(*parts: str) -> Path:
    """Return an absolute path inside the repository root."""

    return PROJECT_ROOT.joinpath(*parts)


def resolve_path(value: str | Path) -> Path:
    """Resolve a configured path against the repository root."""

    candidate = Path(value).expanduser()
    return candidate if candidate.is_absolute() else (PROJECT_ROOT / candidate)


def _expand_environment(value: Any) -> Any:
    """Substitute ``${VAR}`` / ``${VAR:-default}`` references.

    An unset variable without a default becomes an empty string instead of an
    error: loading a stage config must not require every credential, and each
    stage validates the values it truly needs (for example the collection stage
    reports a missing Teacher API key itself).
    """

    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            name, default = match.group(1), match.group(2)
            found = os.environ.get(name)
            if found:
                return found
            if default is not None:
                return default
            return ""

        return ENV_REFERENCE.sub(replace, value)
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, dict):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def load_yaml(path: str | Path) -> dict:
    import yaml  # 延迟导入:评测/采集之外的使用方不必安装 pyyaml

    resolved = resolve_path(path)
    if not resolved.is_file():
        raise SystemExit(f"配置文件不存在:{resolved}")
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise SystemExit(f"配置文件必须是映射:{resolved}")
    return payload


def load_pipeline() -> dict:
    """Load only ``configs/pipeline.yaml`` (used by the environment entry point)."""

    return _expand_environment(load_yaml(PIPELINE_CONFIG))


def load_config(stage: str, path: str | Path | None = None) -> dict:
    """Return ``configs/<stage>.yaml`` with the global pipeline settings attached."""

    pipeline = load_pipeline()
    stage_path = resolve_path(path) if path is not None else CONFIG_DIR / f"{stage}.yaml"
    stage_config = _expand_environment(load_yaml(stage_path))
    stage_config.setdefault("stage", stage)
    stage_config["pipeline"] = pipeline
    stage_config["config_path"] = str(stage_path)
    return stage_config


def stage_argv(
    config: Mapping[str, Any],
    *,
    inject: Mapping[str, Any] | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> list[str]:
    """Turn a stage's ``args`` mapping (plus injected/overridden values) into ``argv``."""

    values: dict[str, Any] = dict(config.get("args") or {})
    for extra in (inject, overrides):
        if extra:
            values.update({key: value for key, value in extra.items() if value is not None})
    return args_to_argv(values)


def args_to_argv(values: Mapping[str, Any]) -> list[str]:
    """Convert ``{"key": value}`` into ``["--key", "value"]`` with flag/list support."""

    argv: list[str] = []
    for key, value in values.items():
        option = f"--{key}"
        if value is None or value is False:
            continue
        if value is True:
            argv.append(option)
        elif isinstance(value, (list, tuple)):
            argv.append(option)
            argv.extend(str(item) for item in value)
        else:
            argv.extend([option, str(value)])
    return argv


def section(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    """Read a nested value from a config mapping without raising on missing keys."""

    current: Any = config
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            return default
        current = current[key]
    return current


def pipeline_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    """Return the absolute ``data``/``models``/``works`` directories."""

    paths = section(config, "pipeline", "paths", default={}) or {}
    return {
        "data": resolve_path(paths.get("data", "data")),
        "models": resolve_path(paths.get("models", "models")),
        "works": resolve_path(paths.get("works", "works")),
    }


def works_dir(config: Mapping[str, Any], *parts: str, create: bool = True) -> Path:
    """Return ``works/<parts...>`` relative to the configured works directory."""

    directory = pipeline_paths(config)["works"].joinpath(*parts)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def models_dir(config: Mapping[str, Any], *parts: str, create: bool = False) -> Path:
    """Return ``models/<parts...>`` relative to the configured models directory."""

    directory = pipeline_paths(config)["models"].joinpath(*parts)
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    return directory


def environment_url(config: Mapping[str, Any]) -> str:
    return str(section(config, "pipeline", "environment", "base_url", default="http://127.0.0.1:5700"))


def base_model_reference(config: Mapping[str, Any]) -> str:
    """Prefer the downloaded local model directory, else fall back to the hub id."""

    local = resolve_path(section(config, "pipeline", "models", "base", default="models/base"))
    if (local / "config.json").is_file():
        return str(local)
    return str(section(config, "pipeline", "models", "base_id", default="openbmb/MiniCPM5-2B"))


def max_steps(config: Mapping[str, Any]) -> int:
    return int(section(config, "pipeline", "reward", "max_steps", default=35) or 35)


def reward_version(config: Mapping[str, Any]) -> str:
    """Return the selected Reward version name (``v4`` / ``v5`` / ``v6``)."""

    return str(section(config, "pipeline", "reward", "version", default="v4") or "v4")


def reward_block(
    config: Mapping[str, Any], *, version: str | None = None
) -> dict[str, Any]:
    """Return the selected version's complete block from ``pipeline.yaml``.

    Every version owns one full block under ``reward.<version>`` with the same
    keys, so reading a contract never requires merging a "baseline" with a
    per-version delta. Non-versioned knobs (``version``, ``max_steps``) stay at
    the ``reward`` top level.
    """

    active = version or reward_version(config)
    return dict(section(config, "pipeline", "reward", active, default={}) or {})


def reward_shaping(
    config: Mapping[str, Any], *, version: str | None = None
) -> dict[str, Any]:
    """Return the selected version's process-shaping parameters.

    ``reward_unverifiable`` is a terminal value rather than a process shaping
    knob, so it is stripped here and served by
    :func:`reward_terminal_overrides`. ``max_steps`` is version-independent and
    read by :func:`max_steps`.
    """

    block = reward_block(config, version=version)
    block.pop("reward_unverifiable", None)
    return block


def reward_terminal_overrides(
    config: Mapping[str, Any], *, version: str | None = None
) -> dict[str, Any]:
    """Return the selected version's terminal-value override.

    Only ``reward_unverifiable`` is configurable per version; v6's continuous
    purchase formula lives in ``terminal.evaluate_purchase`` (selected by
    ``version``), not here.
    """

    value = reward_block(config, version=version).get("reward_unverifiable")
    return {"reward_unverifiable": value} if value is not None else {}


def write_json(path: str | Path, payload: Any) -> Path:
    import json

    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved


def read_jsonl(path: str | Path) -> list[dict]:
    import json

    resolved = resolve_path(path)
    if not resolved.is_file():
        return []
    return [
        json.loads(line)
        for line in resolved.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    import json

    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return resolved
