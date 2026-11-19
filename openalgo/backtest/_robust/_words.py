"""Figures as the plain text of verdicts, rules, notes and warnings: percentages, ordinal
percentiles, rupees with Indian digit grouping and counts. HTML pages escape this text;
it holds no markup. An undefined figure reads "n/a"."""

from __future__ import annotations

import math

from .._messages import counted
from ..report._format import MISSING, indian_digits

#: The most decimal places of a percentage: a figure below a ten-billionth of a percent
#: is a rounding residue of summed rupees, not a figure to show.
MOST_DECIMALS = 10


def _undefined(value: float) -> str | None:
    if value is None or math.isnan(value):
        return MISSING
    if math.isinf(value):
        return "infinite" if value > 0 else "minus infinite"
    return None


def percent(fraction: float, decimals: int = 1) -> str:
    """A fraction as a percentage to `decimals` places: 0.875 is "87.5%". A nonzero
    fraction that would round to zero there shows more places, up to its first
    significant digit (-0.0004 is "-0.04%"), so a shallow drawdown never reads "-0.0%";
    one too small to show in `MOST_DECIMALS` places, a rounding residue, and a zero of
    either sign read as zero ("0.0%")."""
    special = _undefined(fraction)
    if special is not None:
        return special
    value = fraction * 100
    shown = decimals
    while round(value, shown) == 0 and shown < MOST_DECIMALS:
        shown += 1
    if round(value, shown) == 0:
        return f"{0.0:.{decimals}f}%"
    return f"{value:.{shown}f}%"


def ordinal(fraction: float, decimals: int = 0) -> str:
    """A fraction as an ordinal percentile, rounded to `decimals` places with trailing
    zeros dropped: 0.66 is "66th", 0.21 "21st", 0.975 to one place "97.5th" (a part of
    a hundred takes "th", as do 11 to 13). A fraction short of 0 or 1 that would round to
    either there shows more places, up to its first digit that keeps it apart (0.996 is
    "99.6th", 0.004 "0.4th"), so only 0 and 1 themselves read "0th" and "100th"."""
    special = _undefined(fraction)
    if special is not None:
        return special
    value = fraction * 100
    places = decimals
    while value not in (0, 100) and round(value, places) in (0, 100) and places < MOST_DECIMALS:
        places += 1
    shown = f"{value:.{places}f}"
    if "." in shown:
        shown = shown.rstrip("0").rstrip(".")
    if "." in shown or shown[-2:] in ("11", "12", "13"):
        return f"{shown}th"
    return shown + {"1": "st", "2": "nd", "3": "rd"}.get(shown[-1], "th")


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


__all__ = ["MOST_DECIMALS", "number", "ordinal", "percent", "rupees", "whole"]
