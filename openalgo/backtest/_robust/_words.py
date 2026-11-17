"""Figures as the plain text of verdicts, rules, notes and warnings: percentages, rupees
with Indian digit grouping and counts. HTML pages escape this text; it holds no markup.
An undefined figure reads "n/a"."""

from __future__ import annotations

import math

from .._messages import counted
from ..report._format import MISSING, indian_digits


def _undefined(value: float) -> str | None:
    if value is None or math.isnan(value):
        return MISSING
    if math.isinf(value):
        return "infinite" if value > 0 else "minus infinite"
    return None


def percent(fraction: float, decimals: int = 1) -> str:
    """A fraction as a percentage: 0.875 is "87.5%"."""
    special = _undefined(fraction)
    return special if special is not None else f"{fraction * 100:.{decimals}f}%"


def rupees(amount: float) -> str:
    """Rupees with Indian digit grouping and no paise: "Rs 1,23,457"; "Rs -500"."""
    special = _undefined(amount)
    return special if special is not None else f"Rs {indian_digits(amount, 0)}"


def number(value: float, decimals: int = 2) -> str:
    """A plain figure (a ratio, a Sharpe ratio) to `decimals` places."""
    special = _undefined(value)
    return special if special is not None else indian_digits(value, decimals)


def whole(count: int, singular: str, plural: str | None = None) -> str:
    """A count with Indian digit grouping and its noun: "1,000 runs", "1 run"."""
    text = counted(count, singular, plural)
    return indian_digits(count, 0) + text[len(str(count)) :]


__all__ = ["number", "percent", "rupees", "whole"]
