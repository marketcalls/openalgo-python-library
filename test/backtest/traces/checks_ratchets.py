"""Consistency rules for the ratchets in the level timeline: trail and breakeven state.

Every rule takes a case and returns its violations; each violation names the fixture,
the slice and the rule that is broken.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from . import ledger
from .bars import bars_by_slice, rupees
from .timegrid import Case, parse_ist
from .timeline import is_long, stop_level

Violations = list[str]


def _opening_fill(case: Case, instrument: str, slice_ist: str) -> dict[str, Any] | None:
    """The fill that opened the position an instrument holds at the end of a slice."""
    opening = None
    held = 0
    for fill, after in ledger.position_history(case):
        if fill["instrument"] != instrument:
            continue
        if parse_ist(fill["slice_ist"]) > parse_ist(slice_ist):
            break
        if held == 0:
            opening = fill
        held = after.quantity_units
    return opening


def _best_price_since(case: Case, opening: dict[str, Any], slice_ist: str, long: bool) -> Decimal:
    """Best price for the position from its entry to the end of a slice.

    The entry bar counts: the entry is at its open, so the whole bar follows it (the
    recommended reading of specification review item SPEC-01).
    """
    first, last = parse_ist(opening["slice_ist"]), parse_ist(slice_ist)
    best = rupees(opening["price_rupees"])
    for time_ist, bar in bars_by_slice(case)[opening["instrument"]].items():
        if first <= parse_ist(time_ist) <= last:
            best = max(best, bar.high) if long else min(best, bar.low)
    return best


def favourable_extremes_match_the_bars(case: Case) -> Violations:
    """Rule: the favourable extreme of a trail is the best price since the entry, the
    entry bar included. Positions opened by a roll are covered by the roll rule."""
    problems = []
    for row in case["expected"]["levels"]:
        if row["trail"] is None:
            continue
        opening = _opening_fill(case, row["instrument"], row["slice_ist"])
        if opening is None or opening["reason"] == "roll":
            continue
        best = _best_price_since(case, opening, row["slice_ist"], is_long(row))
        stated = rupees(row["trail"]["favourable_extreme_rupees"])
        if stated != best:
            problems.append(
                f"{case['id']} {row['slice_ist']}: favourable extreme of {row['instrument']} "
                f"is {stated}, the bars since the entry give {best}"
            )
    return problems


def armed_breakeven_is_reflected_in_the_stop(case: Case) -> Violations:
    """Rule: once breakeven is armed the stop stands at the breakeven level or tighter."""
    problems = []
    for row in case["expected"]["levels"]:
        breakeven = row["breakeven"]
        if breakeven is None or not breakeven["armed"]:
            continue
        level, stop = rupees(breakeven["level_rupees"]), stop_level(row)
        behind = stop is None or (stop < level if is_long(row) else stop > level)
        if behind:
            problems.append(
                f"{case['id']} {row['slice_ist']}: breakeven is armed at {level} but the "
                f"stop of {row['instrument']} is {stop}"
            )
    return problems
