"""Pure reward-group selection used by the bounded veRL sampling patch."""

from __future__ import annotations

import json
import math
from collections.abc import Hashable, Mapping, Sequence
from pathlib import Path
from typing import Any


def build_rollout_diagnostics(
    uids: Sequence[Hashable], shopping_infos: Sequence[object]
) -> list[dict[str, Any]]:
    """Attach stable group/rollout identities to public AgentLoop diagnostics."""
    if len(uids) != len(shopping_infos):
        raise ValueError("uids and shopping_infos must have equal length")
    rollout_counts: dict[Hashable, int] = {}
    records = []
    for index, (uid, info) in enumerate(zip(uids, shopping_infos, strict=True)):
        if not isinstance(info, Mapping):
            raise ValueError(f"shopping extra field at index {index} is not an object")
        rollout_index = rollout_counts.get(uid, 0)
        rollout_counts[uid] = rollout_index + 1
        records.append({"uid": uid, "rollout_index": rollout_index, **dict(info)})
    return records


def append_training_diagnostic(
    path: str | Path | None,
    event: str,
    global_step: int,
    **payload: object,
) -> None:
    """Append one driver-side training event; an unset path disables persistence."""
    if not path:
        return
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def scalar(value):
        item = getattr(value, "item", None)
        if callable(item):
            return item()
        raise TypeError(f"{type(value).__name__} is not JSON serializable")

    record = {
        "schema_version": 1,
        "event": str(event),
        "global_step": int(global_step),
        **payload,
    }
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True, default=scalar))
        handle.write("\n")


def aggregate_shopping_metrics(shopping_infos: Sequence[object]) -> dict[str, float]:
    """把 AgentLoop 轨迹诊断聚合为 veRL 每步指标。"""
    if not shopping_infos:
        return {}

    reward_keys = (
        "full",
        "strict",
        "native",
        "semantic",
        "total",
        "efficiency",
        "penalty_unfinished",
        "penalty_repeat",
        "repeat_action_rate",
        "r_type",
        "r_att",
        "r_option",
        "r_price",
    )
    rewards = {key: [] for key in reward_keys}
    steps = []
    done = []
    max_steps = []
    repeat_loop = []
    infrastructure_invalid = []
    reward_unverifiable = []
    terminal_utilities = []
    purchase_success = []
    sampling_invalid = []
    match_scores = []
    evidence_coverage = []
    partial_purchase = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        reward = info["reward"]
        for key in reward_keys:
            try:
                value = float(reward[key])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"shopping reward at index {index} is missing numeric {key}"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(f"shopping reward {key} at index {index} is not finite")
            rewards[key].append(value)
        steps.append(float(info.get("steps", 0)))
        done.append(float(info.get("done") is True))
        max_steps.append(float(info.get("termination_reason") == "max_steps"))
        repeat_loop.append(float(info.get("reward_type") == "repeat_loop"))
        infrastructure_invalid.append(float(bool(info.get("infrastructure_invalid"))))
        reward_unverifiable.append(float(bool(info.get("reward_unverifiable"))))
        terminal_utilities.append(
            float(reward.get("terminal_utility", reward["total"]))
        )
        purchase_success.append(
            float(bool(reward.get("purchase_success", reward["full"])))
        )
        sampling_invalid.append(
            float(
                bool(
                    reward.get(
                        "sampling_invalid",
                        info.get("infrastructure_invalid")
                        or info.get("reward_unverifiable"),
                    )
                )
            )
        )
        match_scores.append(float(reward.get("match_score", reward["r_att"])))
        evidence_coverage.append(
            float(reward.get("evidence_coverage", 0.0))
        )
        partial_purchase.append(
            float(info.get("reward_type") == "partial_alternative_purchase")
        )

    def mean(values):
        return sum(values) / len(values)

    return {
        "reward/full_mean": mean(rewards["full"]),
        "reward/strict_mean": mean(rewards["strict"]),
        "reward/native_mean": mean(rewards["native"]),
        "reward/semantic_mean": mean(rewards["semantic"]),
        "reward/shaped_min": min(rewards["total"]),
        "reward/shaped_mean": mean(rewards["total"]),
        "reward/shaped_max": max(rewards["total"]),
        "reward/terminal_utility_min": min(terminal_utilities),
        "reward/terminal_utility_mean": mean(terminal_utilities),
        "reward/terminal_utility_max": max(terminal_utilities),
        "reward/purchase_success_rate": mean(purchase_success),
        "reward/partial_purchase_rate": mean(partial_purchase),
        "reward/match_score_mean": mean(match_scores),
        "reward/evidence_coverage_mean": mean(evidence_coverage),
        "reward/efficiency_mean": mean(rewards["efficiency"]),
        "penalty/unfinished_mean": mean(rewards["penalty_unfinished"]),
        "penalty/repeat_mean": mean(rewards["penalty_repeat"]),
        "component/r_type_mean": mean(rewards["r_type"]),
        "component/r_att_mean": mean(rewards["r_att"]),
        "component/r_option_mean": mean(rewards["r_option"]),
        "component/r_price_mean": mean(rewards["r_price"]),
        "trajectory/average_steps": mean(steps),
        "trajectory/done_rate": mean(done),
        "trajectory/max_steps_rate": mean(max_steps),
        "trajectory/repeat_loop_rate": mean(repeat_loop),
        "trajectory/repeat_action_rate": mean(rewards["repeat_action_rate"]),
        "trajectory/infrastructure_invalid_rate": mean(infrastructure_invalid),
        "trajectory/reward_unverifiable_rate": mean(reward_unverifiable),
        "trajectory/sampling_invalid_rate": mean(sampling_invalid),
    }


def extract_shopping_group_signals(
    shopping_infos: Sequence[object],
) -> tuple[list[float], list[bool], list[bool], list[tuple[str, ...]]]:
    """Return terminal utility, success metrics, and explicit invalid reasons."""
    terminal_utilities = []
    purchase_success = []
    sampling_invalid = []
    invalid_reasons = []
    for index, info in enumerate(shopping_infos):
        if not isinstance(info, Mapping) or not isinstance(info.get("reward"), Mapping):
            raise ValueError(f"shopping extra field at index {index} is missing reward diagnostics")
        try:
            terminal_utility = float(info["reward"]["terminal_utility"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"shopping extra field at index {index} is missing terminal_utility"
            ) from exc
        if not math.isfinite(terminal_utility):
            raise ValueError(
                f"shopping terminal_utility at index {index} is not finite"
            )
        raw_purchase_success = info["reward"].get("purchase_success")
        if not isinstance(raw_purchase_success, (bool, int, float)):
            raise ValueError(
                f"shopping extra field at index {index} is missing purchase_success"
            )
        if "infrastructure_invalid" not in info:
            raise ValueError(
                f"shopping extra field at index {index} is missing infrastructure_invalid"
            )
        reasons = []
        if bool(info["infrastructure_invalid"]):
            reasons.append("infrastructure_invalid")
        if bool(info.get("reward_unverifiable")):
            reasons.append("reward_unverifiable")
        reward_sampling_invalid = bool(
            info["reward"].get("sampling_invalid", False)
        )
        if reward_sampling_invalid and not reasons:
            reasons.append("reward_sampling_invalid")
        terminal_utilities.append(terminal_utility)
        purchase_success.append(bool(raw_purchase_success))
        sampling_invalid.append(bool(reasons))
        invalid_reasons.append(tuple(reasons))
    return (
        terminal_utilities,
        purchase_success,
        sampling_invalid,
        invalid_reasons,
    )


# 组内"可信样本"门槛(达到才保留整组):
#   * 下限 2:1 条样本构不成组内基线(优势恒为 0,原始 GRPO 还要除以组内 std 会除零);
#   * 半数:组内均值方差 ~ σ²/k,取半数(GRPO n=8 时 k=4)对应优势噪声
#     (k/(k-1))² ≤ 1.78×,避免放行 k=2 这种 4.0× 噪声的组。
# 丢组时 insufficient_valid_group_count 会记录,不会静默丢数据。
MIN_VALID_GROUP_MEMBERS = 2
MIN_VALID_GROUP_RATIO = 0.5


def select_reward_varying_groups(
    uids: Sequence[Hashable],
    seq_rewards: Sequence[float],
    *,
    terminal_utilities: Sequence[float] | None = None,
    purchase_success: Sequence[bool] | None = None,
    sampling_invalid: Sequence[bool] | None = None,
    sampling_invalid_reasons: Sequence[Sequence[str]] | None = None,
    tolerance: float = 1.0e-8,
) -> tuple[list[int], dict[str, Any]]:
    """Return trajectory indices belonging to groups with non-constant reward.

    Group order follows the first occurrence of each uid. Returned trajectory
    indices preserve their original order, so callers can safely apply the same
    selection to every aligned tensor and non-tensor batch field.
    """

    if len(uids) != len(seq_rewards):
        raise ValueError(
            f"uids and seq_rewards must have equal length, got {len(uids)} and {len(seq_rewards)}"
        )
    optional_sequences = {
        "terminal_utilities": terminal_utilities,
        "purchase_success": purchase_success,
        "sampling_invalid": sampling_invalid,
        "sampling_invalid_reasons": sampling_invalid_reasons,
    }
    for name, values in optional_sequences.items():
        if values is not None and len(values) != len(uids):
            raise ValueError(f"{name} must have the same length as uids")
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError(f"tolerance must be a finite non-negative number, got {tolerance!r}")

    utility_values = (
        terminal_utilities if terminal_utilities is not None else seq_rewards
    )
    success_values = (
        purchase_success if purchase_success is not None else [False] * len(uids)
    )
    invalid_values = (
        sampling_invalid if sampling_invalid is not None else [False] * len(uids)
    )
    reason_values = (
        sampling_invalid_reasons
        if sampling_invalid_reasons is not None
        else [()] * len(uids)
    )
    grouped: dict[Hashable, dict[str, Any]] = {}
    for index, (
        uid,
        raw_reward,
        raw_utility,
        raw_success,
        raw_invalid,
        raw_reasons,
    ) in enumerate(
        zip(
            uids,
            seq_rewards,
            utility_values,
            success_values,
            invalid_values,
            reason_values,
            strict=True,
        )
    ):
        try:
            hash(uid)
        except TypeError as exc:
            raise ValueError(f"uid at index {index} is not hashable: {uid!r}") from exc

        reward = float(raw_reward)
        if not math.isfinite(reward):
            raise ValueError(f"seq_reward at index {index} is not finite: {raw_reward!r}")
        utility = float(raw_utility)
        if not math.isfinite(utility):
            raise ValueError(
                f"terminal_utility at index {index} is not finite: {raw_utility!r}"
            )

        group = grouped.setdefault(
            uid,
            {
                "uid": uid,
                "indices": [],
                "rewards": [],
                "terminal_utilities": [],
                "purchase_success": [],
                "sampling_invalid": [],
                "sampling_invalid_reasons": [],
            },
        )
        group["indices"].append(index)
        group["rewards"].append(reward)
        group["terminal_utilities"].append(utility)
        group["purchase_success"].append(bool(raw_success))
        group["sampling_invalid"].append(bool(raw_invalid))
        group["sampling_invalid_reasons"].extend(str(reason) for reason in raw_reasons)

    kept_uids: list[Hashable] = []
    dropped_uids: list[Hashable] = []
    groups: list[dict[str, Any]] = []
    kept_trajectory_indices: list[int] = []
    invalid_trajectory_count = 0
    for uid, group in grouped.items():
        indices = group["indices"]
        utilities = group["terminal_utilities"]
        invalid_flags = group["sampling_invalid"]
        has_sampling_invalid = any(invalid_flags)
        # 无效样本逐条掩码,不再"一票否决"整组:n=8 时 any() 会把同组里其余
        # 可用样本一起丢掉(实测丢弃率随 n 上升,44% 的组因此被浪费)。
        valid_positions = [
            position for position, bad in enumerate(invalid_flags) if not bad
        ]
        invalid_count = len(indices) - len(valid_positions)
        invalid_trajectory_count += invalid_count
        basis = [utilities[position] for position in valid_positions] or list(utilities)
        utility_min = min(basis)
        utility_max = max(basis)
        utility_varying = utility_max - utility_min > tolerance
        reasons = tuple(sorted(set(group["sampling_invalid_reasons"])))
        required_valid = max(
            MIN_VALID_GROUP_MEMBERS, math.ceil(len(indices) * MIN_VALID_GROUP_RATIO)
        )
        if len(valid_positions) < required_valid:
            # 可信样本太少(低于半数或低于下限):构不成可靠基线,整组丢弃。
            drop_reason = "sampling_invalid"
            kept_indices: list[int] = []
        elif utility_varying:
            # veRL 断言 batch 必须是完整组的整数倍("batch size % rollout.n != 0"),
            # 所以只能整组保留:个别不可信样本随组进入 batch,其低奖励会自然拿到负优势。
            drop_reason = None
            kept_indices = list(indices)
        else:
            drop_reason = "constant_reward"
            kept_indices = []
        kept_trajectory_indices.extend(kept_indices)
        keep = bool(kept_indices)
        if keep:
            kept_uids.append(uid)
        else:
            dropped_uids.append(uid)
        groups.append(
            {
                "uid": uid,
                "indices": tuple(group["indices"]),
                "rewards": tuple(group["rewards"]),
                "terminal_utilities": tuple(utilities),
                "purchase_success": tuple(group["purchase_success"]),
                "utility_min": utility_min,
                "utility_max": utility_max,
                "reward_varying": utility_varying,
                "sampling_invalid": has_sampling_invalid,
                "sampling_invalid_reasons": reasons,
                "drop_reason": drop_reason,
                "kept": keep,
                "kept_indices": tuple(kept_indices),
                "valid_count": len(valid_positions),
                "required_valid": required_valid,
            }
        )

    # 逐条返回并保持原始顺序:无效样本已被单独掩码,不再连坐整组。
    trajectory_indices = sorted(kept_trajectory_indices)
    stats = {
        "num_trajectories": len(uids),
        "num_groups": len(grouped),
        "kept_group_count": len(kept_uids),
        "dropped_group_count": len(dropped_uids),
        "kept_uids": tuple(kept_uids),
        "dropped_uids": tuple(dropped_uids),
        "all_equal_group_count": sum(
            not group["reward_varying"] for group in groups
        ),
        "all_zero_utility_group_count": sum(
            max(abs(value) for value in group["terminal_utilities"]) <= tolerance
            for group in groups
        ),
        "all_purchase_success_group_count": sum(
            all(group["purchase_success"])
            for group in groups
        ),
        "no_purchase_success_group_count": sum(
            not any(group["purchase_success"]) for group in groups
        ),
        "sampling_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "invalid_trajectory_count": invalid_trajectory_count,
        "insufficient_valid_group_count": sum(
            group["drop_reason"] == "sampling_invalid" for group in groups
        ),
        "sampling_invalid_reason_counts": {
            reason: sum(
                reason in group["sampling_invalid_reasons"] for group in groups
            )
            for reason in sorted(
                {
                    reason
                    for group in groups
                    for reason in group["sampling_invalid_reasons"]
                }
            )
        },
        # Compatibility aliases for existing monitoring code.
        "infrastructure_invalid_group_count": sum(
            group["sampling_invalid"] for group in groups
        ),
        "groups": tuple(groups),
    }
    return trajectory_indices, stats
