"""Derive lightweight Reward v4 features without a per-task constraint contract."""

from __future__ import annotations

import re
import unicodedata

from shopping_agent.reward.comparators import (
    load_brand_aliases,
    normalize_text,
)


REWARD_FEATURE_VERSION = "shopping-reward-features-v1"
OPTION_AXIS_VERSION = "option-axis-v1"
_AXIS_ALIASES = {
    "color": {"颜色", "颜色分类"},
    "size": {"尺码", "鞋码"},
    "dimensions": {"尺寸", "大小"},
    "net_content": {"净含量", "总净含量"},
    "flavor": {"口味", "食品口味"},
    "specification": {"规格", "规格描述", "规格类型"},
    "bundle": {"套餐", "套餐类型", "组合套餐"},
    "capacity": {"容量", "规格容量"},
}
_MODEL_TOKEN = re.compile(
    r"(?<![a-z0-9])(?=[a-z0-9._+-]{2,24}(?![a-z0-9]))"
    r"(?=[a-z0-9._+-]*\d)[a-z0-9._+-]+",
    flags=re.IGNORECASE,
)
_SHOP_SUFFIXES = (
    "官方旗舰店",
    "旗舰店",
    "专卖店",
    "专营店",
    "企业店",
    "店",
)


def normalize_option_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold()
    text = text.replace("/", "|")
    return re.sub(r"\s+", "", text)


def canonicalize_option_axis(value: object) -> str:
    normalized = normalize_option_text(value)
    for canonical, aliases in _AXIS_ALIASES.items():
        if normalized in {
            normalize_option_text(alias) for alias in aliases
        }:
            return canonical
    return normalized


def _clean_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value:
        text = re.sub(r"\s+", " ", str(item or "")).strip()
        if text and text not in cleaned:
            cleaned.append(text)
    return cleaned


def _target_option_axes(target_product: dict) -> dict[str, list[str]]:
    axes = {}
    for raw_axis, entries in (
        target_product.get("customization_options") or {}
    ).items():
        values = []
        for entry in entries or []:
            if (
                isinstance(entry, dict)
                and normalize_option_text(entry.get("value"))
            ):
                values.append(str(entry["value"]))
        axes[str(raw_axis)] = values
    return axes


def _resolve_required_options(
    option_values: list[str],
    target_product: dict,
) -> tuple[dict, list[dict]]:
    axes = _target_option_axes(target_product)
    resolved = {}
    unresolved = []
    for required_value in option_values:
        normalized_required = normalize_option_text(required_value)
        matches = [
            raw_axis
            for raw_axis, values in axes.items()
            if normalized_required
            in {normalize_option_text(value) for value in values}
        ]
        if len(matches) != 1:
            unresolved.append(
                {
                    "value": required_value,
                    "reason": (
                        "axis_not_found"
                        if not matches
                        else "axis_ambiguous"
                    ),
                    "axes": matches,
                }
            )
            continue
        raw_axis = matches[0]
        canonical_axis = canonicalize_option_axis(raw_axis)
        if canonical_axis in resolved:
            unresolved.append(
                {
                    "value": required_value,
                    "reason": "canonical_axis_collision",
                    "axes": [raw_axis],
                }
            )
            continue
        resolved[canonical_axis] = {
            "value": required_value,
            "source_axis": raw_axis,
            "source": "instruction.instruction_options",
        }
    return resolved, unresolved


def _explicit_brand(instruction: str, target_product: dict) -> list[str]:
    instruction_text = normalize_text(instruction)
    target_text = normalize_text(
        " ".join(
            str(value)
            for value in (
                target_product.get("title"),
                target_product.get("shop_name"),
            )
            if value
        )
    )
    aliases = load_brand_aliases()
    matches = {
        canonical
        for alias, canonical in aliases.items()
        if len(alias) >= 2
        and alias in instruction_text
        and alias in target_text
    }
    shop_name = normalize_text(target_product.get("shop_name"))
    for suffix in _SHOP_SUFFIXES:
        normalized_suffix = normalize_text(suffix)
        if shop_name.endswith(normalized_suffix):
            shop_name = shop_name[: -len(normalized_suffix)]
            break
    title = normalize_text(target_product.get("title"))
    for length in range(min(len(shop_name), 12), 1, -1):
        prefix = shop_name[:length]
        if prefix in instruction_text and prefix in title:
            matches.add(prefix)
            break
    return sorted(matches)


def _explicit_models(instruction: str, target_product: dict) -> list[str]:
    instruction_tokens = {
        token.casefold() for token in _MODEL_TOKEN.findall(instruction)
    }
    target_text = " ".join(
        str(value)
        for value in (
            target_product.get("title"),
            target_product.get("full_description"),
        )
        if value
    )
    target_tokens = {
        token.casefold() for token in _MODEL_TOKEN.findall(target_text)
    }
    return sorted(instruction_tokens.intersection(target_tokens))


def compile_reward_features(
    instruction_record: object,
    target_product: object,
) -> dict:
    """Build fixed scoring inputs from existing task annotations and Gold metadata."""
    instruction = (
        instruction_record if isinstance(instruction_record, dict) else {}
    )
    product = target_product if isinstance(target_product, dict) else {}
    instruction_text = str(instruction.get("instruction") or "")
    option_values = _clean_list(instruction.get("instruction_options"))
    required_options, unresolved_options = _resolve_required_options(
        option_values,
        product,
    )
    return {
        "reward_feature_version": REWARD_FEATURE_VERSION,
        "category": product.get("category"),
        "expected_brand": _explicit_brand(instruction_text, product),
        "expected_model": _explicit_models(instruction_text, product),
        "expected_core_functions": _clean_list(
            instruction.get("attributes")
        ),
        "required_options_by_key": required_options,
        "unresolved_option_requirements": unresolved_options,
        "option_axis_version": OPTION_AXIS_VERSION,
        "feature_sources": {
            "category": "task.target_product.category",
            "brand": "instruction_explicit_alias",
            "model": "instruction_target_token_intersection",
            "core_functions": "instruction.attributes",
            "options": "instruction.instruction_options",
        },
    }
