"""Field-aware, tri-state comparators for ShopSimulator Reward v4."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import re
import unicodedata


COMPARATOR_VERSION = "shopping-comparators-v1"
BRAND_ALIAS_VERSION = "brand-aliases-v1"
PASS = "pass"
FAIL = "fail"
UNVERIFIABLE = "unverifiable"
_NEGATION_PREFIXES = (
    "不支持",
    "不含",
    "不具备",
    "没有",
    "无",
    "非",
    "不",
)


def normalize_text(value: object, *, remove_space: bool = True) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).casefold().strip()
    if remove_space:
        return re.sub(r"\s+", "", text)
    return re.sub(r"\s+", " ", text)


def comparison(
    status: str,
    *,
    comparator: str,
    required: object,
    actual: object,
    source_field: str,
    evidence: object = None,
) -> dict:
    if status not in {PASS, FAIL, UNVERIFIABLE}:
        raise ValueError(f"invalid comparator status: {status}")
    return {
        "status": status,
        "passed": status == PASS,
        "verifiable": status != UNVERIFIABLE,
        "comparator": comparator,
        "required": required,
        "actual": actual,
        "source_field": source_field,
        "evidence": evidence,
    }


@lru_cache(maxsize=1)
def load_brand_aliases() -> dict[str, str]:
    path = Path(__file__).resolve().parents[1] / "resources" / "brand_aliases.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("version") != BRAND_ALIAS_VERSION:
        raise ValueError("brand alias table has the wrong version")
    result = {}
    for canonical, aliases in (payload.get("aliases") or {}).items():
        canonical_normalized = normalize_text(canonical)
        result[canonical_normalized] = canonical_normalized
        for alias in aliases or []:
            result[normalize_text(alias)] = canonical_normalized
    return result


def _model_boundary_match(expected: object, actual: object) -> bool:
    needle = normalize_text(expected, remove_space=False)
    haystack = normalize_text(actual, remove_space=False)
    if not needle or not haystack:
        return False
    escaped = re.escape(needle)
    if re.fullmatch(r"[a-z0-9._+-]+", needle):
        return re.search(
            rf"(?<![a-z0-9]){escaped}(?![a-z0-9])",
            haystack,
        ) is not None
    return needle == haystack


def compare_model(required: object, product: dict) -> dict:
    required_values = required if isinstance(required, list) else [required]
    actual = product.get("model") or product.get("title") or product.get("Title")
    if not actual or not any(normalize_text(value) for value in required_values):
        return comparison(
            UNVERIFIABLE,
            comparator="model_token_boundary",
            required=required_values,
            actual=actual,
            source_field="model|title",
        )
    passed = all(_model_boundary_match(value, actual) for value in required_values)
    return comparison(
        PASS if passed else FAIL,
        comparator="model_token_boundary",
        required=required_values,
        actual=actual,
        source_field="model|title",
    )


def _category_parts(value: object) -> list[str]:
    return [
        normalize_text(part)
        for part in str(value or "").split("›")
        if normalize_text(part)
    ]


def compare_category(required: object, product: dict) -> dict:
    actual = product.get("category")
    actual_parts = _category_parts(actual)
    required_values = required if isinstance(required, list) else [required]
    required_chains = [
        parts for parts in map(_category_parts, required_values) if parts
    ]
    if not required_chains or not actual_parts:
        return comparison(
            UNVERIFIABLE,
            comparator="category_leaf_ancestor_chain",
            required=required_values,
            actual=actual,
            source_field="category",
        )
    comparisons = []
    for required_parts in required_chains:
        leaf_match = required_parts[-1] == actual_parts[-1]
        required_ancestors = set(required_parts[:-1])
        actual_ancestors = set(actual_parts[:-1])
        ancestor_match = (
            not required_ancestors
            or not actual_ancestors
            or bool(required_ancestors.intersection(actual_ancestors))
        )
        comparisons.append(
            {
                "required": required_parts,
                "leaf_match": leaf_match,
                "ancestor_match": ancestor_match,
            }
        )
    passed = any(
        item["leaf_match"] and item["ancestor_match"]
        for item in comparisons
    )
    return comparison(
        PASS if passed else FAIL,
        comparator="category_leaf_ancestor_chain",
        required=required_values,
        actual=actual,
        source_field="category",
        evidence={"comparisons": comparisons},
    )


def _flatten_text(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, dict):
        result = []
        for key, nested in value.items():
            result.append(str(key))
            result.extend(_flatten_text(nested))
        return result
    if isinstance(value, (list, tuple, set)):
        result = []
        for nested in value:
            result.extend(_flatten_text(nested))
        return result
    return [str(value)]


def _positive_occurrence(text: str, expected: str) -> bool | None:
    positions = [match.start() for match in re.finditer(re.escape(expected), text)]
    if not positions:
        return False
    for position in positions:
        prefix = text[max(0, position - 4):position]
        if not any(prefix.endswith(negative) for negative in _NEGATION_PREFIXES):
            return True
    return None


def compare_core_functions(required: object, product: dict) -> dict:
    required_values = required if isinstance(required, list) else [required]
    required_values = [
        str(value) for value in required_values if normalize_text(value)
    ]
    structured = _flatten_text(
        product.get("attribute") or product.get("Attributes")
    )
    fallback = _flatten_text(
        [
            product.get("title"),
            product.get("Title"),
            product.get("small_description"),
            product.get("BulletPoints"),
            product.get("full_description"),
            product.get("Description"),
        ]
    )
    if not required_values or not structured and not any(fallback):
        return comparison(
            UNVERIFIABLE,
            comparator="structured_attribute_with_polarity_fallback",
            required=required_values,
            actual=structured or fallback,
            source_field="attribute|description",
        )
    structured_normalized = [normalize_text(value) for value in structured]
    fallback_normalized = normalize_text(" ".join(fallback))
    missing = []
    negated = []
    evidence = []
    for raw_expected in required_values:
        expected = normalize_text(raw_expected)
        structured_matches = [
            _positive_occurrence(actual, expected)
            for actual in structured_normalized
            if expected in actual
        ]
        if any(match is True for match in structured_matches):
            evidence.append({"value": raw_expected, "source": "attribute"})
            continue
        if any(match is None for match in structured_matches):
            negated.append(raw_expected)
            continue
        fallback_match = _positive_occurrence(fallback_normalized, expected)
        if fallback_match is True:
            evidence.append({"value": raw_expected, "source": "text_fallback"})
        elif fallback_match is None:
            negated.append(raw_expected)
        else:
            missing.append(raw_expected)
    status = PASS if not missing and not negated else FAIL
    return comparison(
        status,
        comparator="structured_attribute_with_polarity_fallback",
        required=required_values,
        actual=structured or fallback,
        source_field="attribute|description",
        evidence={
            "matched": evidence,
            "missing": missing,
            "negated": negated,
        },
    )

