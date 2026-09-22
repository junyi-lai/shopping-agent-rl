"""Frozen reward contract shared by every reward version.

This module is **version-independent**: v4, v5 and v6 all run the exact same
process shaping, strict-success gate and step accounting. A version only changes
the terminal *value* (see ``terminal.py`` and the ``v4.py`` / ``v5.py`` /
``v6.py`` specs) plus its ``reward.<version>`` block in ``configs/pipeline.yaml``.

Contents:

  - ``REWARD_VERSION``   : the frozen environment contract string;
  - ``is_strict_success``: the correctness gate (gold_purchase + valid + purchased);
  - ``compute_shaped_reward``: terminal utility + process shaping;
  - the step-accounting helpers the training adapter and the evaluator share.

It is deliberately decoupled from the embedded ShopSimulator: it imports nothing
from ``environments/`` and consumes only the structured facts the environment
returns at the terminal step plus the executed tool steps.
"""

from __future__ import annotations

import json
import os

# 项目层与环境终局必须同名:环境返回的 reward_detail.reward_version 就是这一字符串,
# 采集/筛选/测评/校验都拿它做相等比较(见 reward/__init__.py 的导出)。
REWARD_VERSION = "shopping-reward-v4"

DEFAULT_SHAPING_CONFIG = {
    "efficiency_cap": 0.10,
    "evidence_cap": 0.10,
    "evidence_threshold": 2,  # 核验≥2 次详情/规格页即得满额证据奖励
    "repeat_per_action": 0.02,
    "repeat_cap": 0.20,
    "max_steps": 35,
    # 探索项:只在"没买成"的轨迹上给,用来区分"认真找过才放弃"和"敷衍放弃"。
    # 默认 0.0(关闭);开启后四个环节共用同一套公式。
    "exploration_cap": 0.0,
    "exploration_search_target": 3,   # 换够 3 个不同搜索词即拿满搜索那一半
    "exploration_open_target": 2,     # 打开核验过 2 个候选即拿满候选那一半
}

# 入口把 configs/pipeline.yaml 的 reward 段通过这个变量传进来(内联 JSON 或文件路径),
# 使训练、筛选、评测三处共用同一套塑造参数。
SHAPING_CONFIG_ENV = "SHOPPING_AGENT_REWARD_CONFIG"

# 这些工具调用代表"在买之前核验了证据",是证据充分性的轨迹级信号。
EVIDENCE_TOOLS = frozenset(
    {"view_description", "view_features", "view_attributes", "view_reviews"}
)

# 硬门全过的三种购买结果(即"真的买到了东西"),终局取值映射据此区分。
PURCHASE_REWARD_TYPES = frozenset(
    {
        "gold_purchase",
        "valid_alternative_purchase",
        "partial_alternative_purchase",
    }
)


def resolve_shaping_config(overrides: dict | None = None) -> dict:
    """Return the shaping config: defaults, then the environment, then explicit overrides."""

    resolved = dict(DEFAULT_SHAPING_CONFIG)
    raw = os.environ.get(SHAPING_CONFIG_ENV, "").strip()
    if raw:
        payload = raw
        if not raw.startswith("{"):
            try:
                payload = open(raw, encoding="utf-8").read()  # noqa: SIM115 (配置只读一次)
            except OSError:
                payload = ""
        if payload:
            try:
                loaded = json.loads(payload)
            except json.JSONDecodeError:
                loaded = None
            if isinstance(loaded, dict):
                resolved.update(
                    {
                        key: value
                        for key, value in loaded.items()
                        if key in DEFAULT_SHAPING_CONFIG and value is not None
                    }
                )
    if overrides:
        resolved.update(
            {
                key: value
                for key, value in overrides.items()
                if key in DEFAULT_SHAPING_CONFIG and value is not None
            }
        )
    return resolved


def is_strict_success(reward_type, reward_valid, purchase_success) -> bool:
    """Strict success: a complete, valid gold purchase.

    This is the correctness gate. It depends only on the terminal facts, never
    on the shaping magnitude, so it stays stable no matter how the shaping
    weights are tuned.
    """
    return bool(
        reward_type == "gold_purchase"
        and reward_valid is True
        and purchase_success is True
    )


def canonical_parameters(parameters) -> dict:
    """Deterministically normalize tool arguments; ``think.note`` is excluded."""
    if not isinstance(parameters, dict):
        return {}
    return {
        str(key): value
        for key, value in sorted(parameters.items(), key=lambda item: str(item[0]))
        if str(key) != "note"
    }


def _step_tool(step) -> str | None:
    tool = step.get("tool_name") if isinstance(step, dict) else None
    if tool is None:
        tool = step.get("tool") if isinstance(step, dict) else None
    return str(tool).strip() if tool else None


def count_consecutive_duplicate_actions(steps) -> int:
    """Count consecutive identical (tool, arguments) pairs across executed steps.

    ``think`` is ignored because it is not an environment action. The training
    adapter and the evaluation summarizer both call this helper so the repeat
    penalty is identical in both paths.
    """
    count = 0
    previous = None
    for step in steps or []:
        if not isinstance(step, dict):
            continue
        tool = _step_tool(step)
        if not tool or tool == "think":
            continue
        signature = (
            tool,
            json.dumps(
                canonical_parameters(step.get("parameters")),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
        )
        if signature == previous:
            count += 1
        previous = signature
    return count


def count_evidence_actions(steps) -> int:
    """Count detail/subpage verification actions (view_*) in the trajectory."""
    return sum(
        1
        for step in steps or []
        if isinstance(step, dict) and _step_tool(step) in EVIDENCE_TOOLS
    )


def count_distinct_searches(steps) -> int:
    """Count distinct (normalized) search queries in the trajectory."""
    queries = set()
    for step in steps or []:
        if not isinstance(step, dict) or _step_tool(step) != "search_products":
            continue
        parameters = step.get("parameters") or {}
        query = parameters.get("query") if isinstance(parameters, dict) else None
        if query is None:
            continue
        queries.add(str(query).strip().casefold())
    return len(queries)


def count_distinct_opened_candidates(steps) -> int:
    """Count distinct products the agent actually opened for inspection."""
    opened = set()
    for step in steps or []:
        if not isinstance(step, dict) or _step_tool(step) != "open_product":
            continue
        parameters = step.get("parameters") or {}
        asin = parameters.get("asin") if isinstance(parameters, dict) else None
        if asin is None:
            continue
        opened.add(str(asin).strip())
    return len(opened)


def compute_shaped_reward(
    terminal_utility: float,
    *,
    purchase_success: bool,
    steps: int,
    evidence_actions: int,
    repeat_action_count: int,
    distinct_searches: int = 0,
    distinct_opened_candidates: int = 0,
    reward_type=None,
    reward_valid=None,
    max_steps: int | None = None,
    config: dict | None = None,
) -> dict:
    """Return the shaped reward (终端取值 + 过程塑造)、拆解明细与成功门。

    ``terminal_utility`` 由版本决定(见 ``v4.py`` / ``v5.py`` / ``v6.py``);
    本函数对任何版本都一样,故与版本无关。
    """
    cfg = resolve_shaping_config(config)
    limit = int(max_steps or cfg["max_steps"])
    terminal = float(terminal_utility)

    efficiency = 0.0
    evidence = 0.0
    if purchase_success:
        executed = max(0, int(steps))
        efficiency = float(cfg["efficiency_cap"]) * max(
            0.0, (limit - executed) / limit
        )
        evidence = float(cfg["evidence_cap"]) * max(
            0.0,
            min(1.0, int(evidence_actions) / float(cfg["evidence_threshold"])),
        )

    exploration = 0.0
    if not purchase_success:
        search_ratio = min(
            1.0,
            int(distinct_searches) / float(cfg["exploration_search_target"]),
        )
        open_ratio = min(
            1.0,
            int(distinct_opened_candidates) / float(cfg["exploration_open_target"]),
        )
        exploration = float(cfg["exploration_cap"]) * 0.5 * (search_ratio + open_ratio)

    repeat_penalty = min(
        float(cfg["repeat_cap"]),
        float(cfg["repeat_per_action"]) * max(0, int(repeat_action_count)),
    )

    total = terminal + efficiency + evidence + exploration - repeat_penalty
    return {
        "version": REWARD_VERSION,
        "total": total,
        "terminal_utility": terminal,
        "success": is_strict_success(reward_type, reward_valid, purchase_success),
        "shaping": {
            "efficiency_bonus": efficiency,
            "evidence_bonus": evidence,
            "exploration_bonus": exploration,
            "repeat_penalty": repeat_penalty,
        },
    }
