"""Environment-independent reward scoring entry points."""

from __future__ import annotations

from shopping_agent.reward.budget import explicit_budget_from_instruction
from shopping_agent.reward.features import compile_reward_features
from shopping_agent.reward.terminal import (
    evaluate_abstain,
    evaluate_purchase,
    fixed_termination,
)
from shopping_agent.reward.contract import (
    compute_shaped_reward,
    count_consecutive_duplicate_actions,
    count_evidence_actions,
)
from shopping_agent.reward.versions import (
    active_terminal_overrides,
    active_version_string,
    version_name_for,
)


def score_reward_detail(reward_detail, *, steps, max_steps=None, config=None):
    """Score one terminal step from a reward-detail facts dict.

    Returns the Reward v4 result including ``success`` and ``total``.
    """
    reward_detail = reward_detail if isinstance(reward_detail, dict) else {}
    return compute_shaped_reward(
        float(reward_detail.get("terminal_utility", 0.0)),
        purchase_success=reward_detail.get("purchase_success") is True,
        steps=len(steps or []),
        evidence_actions=count_evidence_actions(steps or []),
        repeat_action_count=count_consecutive_duplicate_actions(steps or []),
        reward_type=reward_detail.get("reward_type"),
        reward_valid=reward_detail.get("reward_valid"),
        max_steps=max_steps,
        config=config,
    )


def _normalize_product_for_scoring(product):
    """Fill the field aliases ``evaluate_purchase`` expects from an env product."""
    product = dict(product or {})
    product.setdefault("Title", product.get("title") or product.get("Title") or "")
    product.setdefault(
        "Description",
        product.get("full_description") or product.get("Description") or "",
    )
    product.setdefault("BulletPoints", product.get("small_description") or [])
    product.setdefault(
        "Attributes",
        product.get("attribute") or product.get("Attributes") or [],
    )
    product.setdefault("pricing", product.get("pricing") or [])
    return product


def compute_terminal_reward_detail(result, *, version=None, terminal_overrides=None):
    """Compute the terminal reward detail from the environment's raw facts.

    The environment is a pure simulator: at the terminal step it returns only
    raw facts (purchased product, goal, target product, options, price
    resolution, abstention facts). This function compiles the reward features,
    classifies the terminal outcome (gold / alternative / partial / wrong /
    abstain / repeat / max steps) and returns the ``reward_detail`` structure
    training, evaluation and collection all consume.

    The classification bucket is version-stable; only the terminal reward
    values and the emitted ``reward_version`` string follow the active version
    (v4 by default, v5 opt-in via environment). ``version`` and
    ``terminal_overrides`` let callers pin one version explicitly (used by
    tests); when omitted they fall back to the process environment.
    """
    result = result if isinstance(result, dict) else {}
    reason = str(result.get("termination_reason") or "environment_done")
    version_string = version or active_version_string()
    overrides = (
        terminal_overrides
        if terminal_overrides is not None
        else active_terminal_overrides()
    )
    rewards = overrides or None
    version_name = version_name_for(version_string)

    if reason == "purchase":
        product = _normalize_product_for_scoring(result.get("purchased_product"))
        goal = result.get("goal") or {}
        target = result.get("target_product") or {}
        options = result.get("options") or {}
        price_resolution = result.get("price_resolution")
        instruction_record = {
            "instruction": goal.get("instruction_text") or "",
            "attributes": goal.get("attributes") or [],
            "instruction_options": goal.get("goal_options") or [],
        }
        reward_goal = {
            "asin": goal.get("asin"),
            "category": goal.get("category"),
            "price_upper": explicit_budget_from_instruction(
                goal.get("instruction_text") or ""
            ),
        }
        reward_goal.update(compile_reward_features(instruction_record, target))
        detail = evaluate_purchase(
            product,
            reward_goal,
            selected_options=options,
            price_resolution=price_resolution,
            rewards=rewards,
            version=version_name,
        ).to_dict()
        detail["reward_version"] = version_string
        return detail

    if reason == "abstain":
        facts = result.get("abstain_facts") or {}
        queries = facts.get("distinct_normalized_queries") or []
        opened = facts.get("opened_asins") or []
        # 简化环境不再追踪"已知可接受候选",按 0 处理(与简化后的语义一致)。
        detail = evaluate_abstain(
            effective_result_sets=min(len(queries), 3),
            opened_candidates=len(opened),
            known_acceptable_candidates=0,
            rewards=rewards,
        ).to_dict()
        detail["reward_version"] = version_string
        return detail

    if reason in {"repeat_loop", "max_steps"}:
        detail = fixed_termination(reason, rewards=rewards).to_dict()
        detail["reward_version"] = version_string
        return detail

    raise ValueError(f"unsupported terminal reason: {reason!r}")
