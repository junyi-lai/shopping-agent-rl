"""Canonical validation helpers for ShopSimulator product identifiers."""

from __future__ import annotations

import re


PRODUCT_ID_CAPTURE = r"\d{8,12}"
PRODUCT_ID_PATTERN = re.compile(rf"^{PRODUCT_ID_CAPTURE}$")


def is_product_id(value: object) -> bool:
    """Return whether *value* is an exact product ID used by the frozen catalog."""
    return PRODUCT_ID_PATTERN.fullmatch(str(value or "").strip()) is not None
