"""Lightweight, consumer-oriented terminal Reward v4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from shopping_agent.reward.comparators import (
    COMPARATOR_VERSION,
    FAIL,
    PASS,
    UNVERIFIABLE,
    compare_category,
    compare_core_functions,
    compare_model,
    comparison,
    load_brand_aliases,
    normalize_text,
)
from shopping_agent.reward.features import (
    REWARD_FEATURE_VERSION,
    normalize_option_text,
)
from shopping_agent.reward.variant_price import (
    VARIANT_PRICE_VERSION,
    compare_required_options,
    resolve_variant_price,
)

# 每版一个模块(reward/v4.py、v5.py、v6.py),接口统一:
# ``terminal_value(reward_type, values, *, asin_match, match_score)``。
import shopping_agent.reward.v4 as reward_v4
import shopping_agent.reward.v5 as reward_v5
import shopping_agent.reward.v6 as reward_v6


REWARD_VERSION = "shopping-reward-v4"
DIMENSION_WEIGHTS = {
    "brand": 0.35,
    "model": 0.25,
    "core_functions": 0.25,
    "key_options": 0.15,
}
DEFAULT_REWARDS = {
    "gold_purchase": 1.0,
    "valid_alternative_purchase": 0.55,
    "partial_purchase_base": -0.30,
    "partial_purchase_scale": 0.55,
    "partial_purchase_cap": 0.25,
    "graceful_stop": -0.15,
    "early_abstain": -0.35,
    "max_steps": -0.50,
    "repeat_loop": -0.65,
    "wrong_purchase": -0.85,
    "reward_unverifiable": 0.0,
}


@dataclass(frozen=True)
class RewardResult:
    reward: float
    reward_type: str
    reward_valid: bool
    termination_reason: str
    target_asin_match: bool
    hard_gates: dict
    weighted_score: float
    evidence: dict

    def to_dict(self):
        payload = asdict(self)
        preference_scoring = self.evidence.get("preference_scoring") or {}
        dimensions = preference_scoring.get("dimensions") or {}
        payload.update(
            {
                "reward_version": REWARD_VERSION,
                "terminal_utility": self.reward,
                "purchase_success": self.reward_type
                in {"gold_purchase", "valid_alternative_purchase"},
                "sampling_invalid": not self.reward_valid,
                "evidence_coverage": float(
                    preference_scoring.get("evidence_coverage", 0.0)
                ),
                "dimension_scores": {
                    name: float(result.get("score", 0.0))
                    for name, result in dimensions.items()
                },
            }
        )
        return payload


def _price_gate(price_resolution: object, goal: dict) -> dict:
    upper = goal.get("price_upper")
    if upper is None:
        return comparison(
            PASS,
            comparator="budget_not_declared_v1",
            required=None,
            actual=(
                price_resolution.get("price")
                if isinstance(price_resolution, dict)
                else None
            ),
            source_field="instruction",
            evidence=price_resolution,
        )
    if (
        not isinstance(price_resolution, dict)
        or price_resolution.get("status") != PASS
    ):
        return comparison(
            UNVERIFIABLE,
            comparator="variant_price_budget_v1",
            required=upper,
            actual=(
                price_resolution.get("price")
                if isinstance(price_resolution, dict)
                else None
            ),
            source_field="variant_price",
            evidence=price_resolution,
        )
    try:
        actual = float(price_resolution["price"])
        limit = float(upper)
    except (KeyError, TypeError, ValueError):
        status = UNVERIFIABLE
        actual = price_resolution.get("price")
        limit = upper
    else:
        if (
            not math.isfinite(actual)
            or not math.isfinite(limit)
            or actual < 0
            or limit <= 0
        ):
            status = UNVERIFIABLE
        else:
            status = PASS if actual <= limit else FAIL
    return comparison(
        status,
        comparator="variant_price_budget_v1",
        required=limit,
        actual=actual,
        source_field="variant_price",
        evidence=price_resolution,
    )


def _brand_gate(expected: str, product: dict) -> dict:
    aliases = load_brand_aliases()
    canonical = aliases.get(normalize_text(expected), normalize_text(expected))
    candidate_brand_text = normalize_text(
        " ".join(
            str(value)
            for value in (
                product.get("brand"),
                product.get("shop_name"),
            )
            if value
        )
    )
    candidate_aliases = {
        candidate_canonical
        for alias, candidate_canonical in aliases.items()
        if len(alias) >= 2 and alias in candidate_brand_text
    }
    matched = (
        canonical in candidate_aliases
        or normalize_text(expected) in candidate_brand_text
    )
    return comparison(
        PASS if matched else FAIL,
        comparator="explicit_brand_alias_text_v1",
        required=expected,
        actual=candidate_brand_text,
        source_field="brand|shop_name",
        evidence={"alias_match": matched},
    )


def _score_requirements(requirements, evaluator) -> dict:
    results = [evaluator(requirement) for requirement in requirements]
    total = len(results)
    passed = sum(result["status"] == PASS for result in results)
    verifiable = sum(
        result["status"] != UNVERIFIABLE for result in results
    )
    return {
        "active": total > 0,
        "score": passed / total if total else 0.0,
        "coverage": verifiable / total if total else 0.0,
        "required_count": total,
        "passed_count": passed,
        "verifiable_count": verifiable,
        "results": results,
    }


def _option_dimension(
    product: dict,
    goal: dict,
    selected_options: object,
) -> dict:
    required = goal.get("required_options_by_key") or {}
    unresolved = goal.get("unresolved_option_requirements") or []
    results = [
        compare_required_options(
            product,
            {axis: requirement},
            selected_options,
        )
        for axis, requirement in required.items()
    ]
    selected_values = {
        normalize_option_text(value)
        for value in (
            selected_options.values()
            if isinstance(selected_options, dict)
            else ()
        )
    }
    results.extend(
        comparison(
            (
                PASS
                if normalize_option_text(item.get("value"))
                in selected_values
                else FAIL
            ),
            comparator="exact_option_value_fallback_v1",
            required=item.get("value"),
            actual=list(
                selected_options.values()
                if isinstance(selected_options, dict)
                else ()
            ),
            source_field="instruction_options",
            evidence=item,
        )
        for item in unresolved
    )
    total = len(results)
    passed = sum(result["status"] == PASS for result in results)
    verifiable = sum(
        result["status"] != UNVERIFIABLE for result in results
    )
    return {
        "active": total > 0,
        "score": passed / total if total else 0.0,
        "coverage": verifiable / total if total else 0.0,
        "required_count": total,
        "passed_count": passed,
        "verifiable_count": verifiable,
        "results": results,
    }


def _preference_dimensions(
    product: dict,
    goal: dict,
    selected_options: object,
) -> dict:
    dimensions = {
        "brand": _score_requirements(
            goal.get("expected_brand") or [],
            lambda value: _brand_gate(value, product),
        ),
        "model": _score_requirements(
            goal.get("expected_model") or [],
            lambda value: compare_model(value, product),
        ),
        "core_functions": _score_requirements(
            goal.get("expected_core_functions") or [],
            lambda value: compare_core_functions([value], product),
        ),
        "key_options": _option_dimension(
            product,
            goal,
            selected_options,
        ),
    }
    active_weight = sum(
        DIMENSION_WEIGHTS[name]
        for name, result in dimensions.items()
        if result["active"]
    )
    if active_weight:
        match_score = sum(
            DIMENSION_WEIGHTS[name] * result["score"]
            for name, result in dimensions.items()
            if result["active"]
        ) / active_weight
        evidence_coverage = sum(
            DIMENSION_WEIGHTS[name] * result["coverage"]
            for name, result in dimensions.items()
            if result["active"]
        ) / active_weight
    else:
        match_score = 1.0
        evidence_coverage = 1.0
    all_satisfied = (
        math.isclose(match_score, 1.0, abs_tol=1.0e-8)
        and math.isclose(evidence_coverage, 1.0, abs_tol=1.0e-8)
    )
    return {
        "weights": DIMENSION_WEIGHTS,
        "active_weight": active_weight,
        "match_score": match_score,
        "evidence_coverage": evidence_coverage,
        "all_satisfied": all_satisfied,
        "dimensions": dimensions,
    }


def _evaluate(
    product: dict,
    goal: dict,
    *,
    selected_options: object,
    price_resolution: object,
) -> tuple[dict, dict]:
    hard_gates = {
        "category": compare_category(goal.get("category"), product),
        "budget": _price_gate(price_resolution, goal),
    }
    preferences = _preference_dimensions(
        product,
        goal,
        selected_options,
    )
    return hard_gates, preferences


def _explicit_price_resolution(price: object) -> dict:
    try:
        value = float(price)
    except (TypeError, ValueError):
        value = None
    return {
        "status": (
            PASS
            if value is not None
            and math.isfinite(value)
            and value >= 0
            else UNVERIFIABLE
        ),
        "price": value,
        "version": VARIANT_PRICE_VERSION,
        "method": "explicit_test_price",
        "evidence": {},
    }


# 终局取值映射:三版各一个模块,接口一致;分类只有一份(见 evaluate_purchase),
# 这里只决定"同一份分类在不同版本下给多少分"。
TERMINAL_VALUE_BY_VERSION = {
    reward_v4.NAME: reward_v4.terminal_value,
    reward_v5.NAME: reward_v5.terminal_value,
    reward_v6.NAME: reward_v6.terminal_value,
}


def evaluate_purchase(
    product: dict,
    goal: dict,
    *,
    selected_options: object,
    price_resolution: dict | None = None,
    price: object = None,
    rewards: dict[str, float] | None = None,
    version: str = "v4",
) -> RewardResult:
    """Score a purchase using two hard gates and continuous soft matching.

    ``version`` selects the terminal **value** mapping (``TERMINAL_VALUE_BY_VERSION``):

    - ``"v4"`` (default) / ``"v5"``: the frozen discrete buckets
      (gold 1.0 / alternative 0.55 / partial capped 0.25 ...). v5 differs from
      v4 only in configuration (process shaping + terminal override).
    - ``"v6"``: one continuous formula for every hard-gate-passing purchase,
      ``0.5 * asin_match + match_score - 0.5``, so "right product but wrong
      variant" earns a continuous, informative signal instead of the flat
      partial cap.

    The ``reward_type`` classification is **shared by every version** (it drives
    strict success, so it must not depend on the reward experiment); only the
    value differs.
    """
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if price_resolution is None:
        price_resolution = (
            _explicit_price_resolution(price)
            if price is not None
            else resolve_variant_price(product, selected_options)
        )
    hard_gates, preferences = _evaluate(
        product,
        goal,
        selected_options=selected_options,
        price_resolution=price_resolution,
    )
    asin_match = str(product.get("asin")) == str(goal.get("asin"))
    hard_unverifiable = any(
        gate["status"] == UNVERIFIABLE for gate in hard_gates.values()
    )
    hard_failed = any(
        gate["status"] == FAIL for gate in hard_gates.values()
    )
    # 分类只此一份:任何版本都不改变分类(分类驱动严格成功率,必须版本无关)。
    if hard_unverifiable:
        reward_type = "reward_unverifiable"
        reward_valid = False
    elif hard_failed:
        reward_type = "wrong_purchase"
        reward_valid = True
    elif preferences["all_satisfied"]:
        reward_type = (
            "gold_purchase" if asin_match else "valid_alternative_purchase"
        )
        reward_valid = True
    else:
        reward_type = "partial_alternative_purchase"
        reward_valid = True
    # 取值按版本查表:同一份分类,不同版本给不同分值。
    reward = TERMINAL_VALUE_BY_VERSION[version](
        reward_type,
        values,
        asin_match=asin_match,
        match_score=float(preferences["match_score"]),
    )
    return RewardResult(
        reward=float(reward),
        reward_type=reward_type,
        reward_valid=reward_valid,
        termination_reason=reward_type,
        target_asin_match=asin_match,
        hard_gates=hard_gates,
        weighted_score=float(preferences["match_score"]),
        evidence={
            "reward_feature_version": goal.get("reward_feature_version"),
            "expected_reward_feature_version": REWARD_FEATURE_VERSION,
            "comparator_version": COMPARATOR_VERSION,
            "variant_price_version": VARIANT_PRICE_VERSION,
            "price_resolution": price_resolution,
            "preference_scoring": preferences,
        },
    )


def evaluate_abstain(
    *,
    effective_result_sets: int,
    opened_candidates: int,
    known_acceptable_candidates: int | None = None,
    known_valid_candidates: int | None = None,
    rewards: dict[str, float] | None = None,
) -> RewardResult:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    known_count = int(
        known_acceptable_candidates
        if known_acceptable_candidates is not None
        else known_valid_candidates or 0
    )
    eligible = (
        int(effective_result_sets) >= 2
        and int(opened_candidates) >= 2
        and known_count == 0
    )
    reward_type = "graceful_stop" if eligible else "early_abstain"
    return RewardResult(
        reward=float(values[reward_type]),
        reward_type=reward_type,
        reward_valid=True,
        termination_reason=reward_type,
        target_asin_match=False,
        hard_gates={},
        weighted_score=0.0,
        evidence={
            "eligible": eligible,
            "effective_result_sets": int(effective_result_sets),
            "opened_candidates": int(opened_candidates),
            "known_acceptable_candidates": known_count,
        },
    )


def fixed_termination(
    reason: str,
    rewards: dict[str, float] | None = None,
) -> RewardResult:
    values = {**DEFAULT_REWARDS, **(rewards or {})}
    if reason not in {"repeat_loop", "max_steps"}:
        raise ValueError(f"unsupported fixed termination reason: {reason}")
    return RewardResult(
        reward=float(values[reason]),
        reward_type=reason,
        reward_valid=True,
        termination_reason=reason,
        target_asin_match=False,
        hard_gates={},
        weighted_score=0.0,
        evidence={},
    )
