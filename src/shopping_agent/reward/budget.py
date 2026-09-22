"""Pure, reproducible budget parsing for Environment v2.

Only fields that are actually stated in the user instruction are used: the
budget gate never invents an upper bound for a task that does not declare one.
"""

from __future__ import annotations

import re


def explicit_budget_from_instruction(instruction):
    """Extract a clearly stated upper budget; return None when ambiguous."""
    text = str(instruction or "").replace(",", "")

    def scaled(number, unit):
        value = float(number)
        normalized_unit = str(unit or "").casefold()
        if normalized_unit == "万":
            value *= 10000
        elif normalized_unit in {"千", "k"}:
            value *= 1000
        return value

    shorthand = re.search(
        r"预算(?:控制)?在?\s*(\d+)\s*万\s*(\d+)\s*(?:千)?\s*(以内|以下|内|左右)?",
        text,
    )
    if shorthand:
        value = float(shorthand.group(1)) * 10000 + float(shorthand.group(2)) * 1000
        if shorthand.group(3) == "左右":
            value *= 1.1
        return value

    # Ranges are explicit upper bounds. Parse them before single-value
    # patterns so “10-20” is never mistaken for an upper bound of 10.
    price_range = re.search(
        r"(?:预算|价格)(?:控制)?在?\s*"
        r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?\s*"
        r"(?:-|~|～|至|到)\s*"
        r"(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?"
        r"(?:之间|以内|以下|左右)?",
        text,
    )
    if price_range:
        low = scaled(price_range.group(1), price_range.group(2))
        high = scaled(price_range.group(3), price_range.group(4))
        if low > 0 and high >= low:
            return high

    # “4k+” describes a lower/open-ended price, not a maximum.
    if re.search(
        r"(?:预算|价格)(?:控制)?在?\s*\d+(?:\.\d+)?\s*[kK]\s*\+",
        text,
    ):
        return None

    patterns = (
        r"预算(?:控制)?在?\s*(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?(以内|以下|内|左右)?(?!\s*[-~～至到+kK])",
        r"价格(?:控制)?在?\s*(\d+(?:\.\d+)?)\s*(万|千|[kK])?\s*元?(以内|以下|内|左右)?(?!\s*[-~～至到+kK])",
        r"(?:不超过|不高于|最高)\s*(\d+(?:\.\d+)?)\s*(万|千)?\s*元",
        r"(\d+(?:\.\d+)?)\s*(万|千)?\s*元(以内|以下)",
    )
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = scaled(match.group(1), match.group(2))
            qualifier = match.group(3) if match.lastindex and match.lastindex >= 3 else None
            # “左右” is not a strict upper bound. Environment v2 freezes a
            # small deterministic tolerance rather than letting target items
            # just above the round-number budget become impossible.
            if qualifier == "左右":
                value *= 1.1
            if value > 0:
                return value
    return None
