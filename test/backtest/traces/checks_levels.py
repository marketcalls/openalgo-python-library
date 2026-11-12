"""Consistency rules for the level timeline of a case. One function per rule.

Every rule takes a case and returns its violations; each violation names the fixture,
the slice and the rule that is broken.
"""

from __future__ import annotations

from . import ledger
from .bars import is_on_tick, rupees, tick_size_of
from .timegrid import Case, parse_ist
from .timeline import LevelRow, is_long, rows_of, stop_level, working_targets

Violations = list[str]


def _instruments(case: Case) -> list[str]:
    return [instrument["symbol"] for instrument in case["instruments"]]


def level_rows_are_in_slice_order(case: Case) -> Violations:
    """Rule: the rows of one instrument are strictly increasing in time."""
    problems = []
    for instrument in _instruments(case):
        rows = rows_of(case, instrument)
        for earlier, later in zip(rows, rows[1:], strict=False):
            if parse_ist(later["slice_ist"]) <= parse_ist(earlier["slice_ist"]):
                problems.append(
                    f"{case['id']} {later['slice_ist']}: level row of {instrument} is not "
                    "later than the row before it"
                )
    return problems


def level_rows_follow_the_fills(case: Case) -> Violations:
    """Rule: the position and average price of a row are those the fills give at the end
    of that slice."""
    problems = []
    for row in case["expected"]["levels"]:
        position = ledger.position_after(case, row["instrument"], row["slice_ist"])
        stated = row["average_price_rupees"]
        average = None if stated is None else rupees(stated)
        if (position.quantity_units, position.average_price) != (
            row["position_quantity_units"],
            average,
        ):
            problems.append(
                f"{case['id']} {row['slice_ist']}: level row states "
                f"{row['position_quantity_units']} units at {stated}, the fills give "
                f"{position.quantity_units} at {position.average_price}"
            )
    return problems


def every_position_change_has_a_level_row(case: Case) -> Violations:
    """Rule: a slice with a fill has a level row for that instrument."""
    with_rows = {(row["instrument"], row["slice_ist"]) for row in case["expected"]["levels"]}
    problems = []
    for fill in case["expected"]["fills"]:
        if (fill["instrument"], fill["slice_ist"]) not in with_rows:
            problems.append(
                f"{case['id']} {fill['slice_ist']}: {fill['instrument']} has a fill "
                "but no level row at the end of the slice"
            )
    return sorted(set(problems))


def stops_never_loosen(case: Case) -> Violations:
    """Rule: while a position stays open its stop only tightens and never disappears."""
    problems = []
    for instrument in _instruments(case):
        rows = rows_of(case, instrument)
        for earlier, later in zip(rows, rows[1:], strict=False):
            if not earlier["position_quantity_units"] or not later["position_quantity_units"]:
                continue
            problems += _loosening(case, earlier, later)
    return problems


def _loosening(case: Case, earlier: LevelRow, later: LevelRow) -> Violations:
    before, after = stop_level(earlier), stop_level(later)
    where = f"{case['id']} {later['slice_ist']}: stop of {later['instrument']}"
    if before is None:
        return []
    if after is None:
        return [f"{where} disappears while the position is open"]
    loosened = after < before if is_long(later) else after > before
    return [f"{where} loosens from {before} to {after}"] if loosened else []


def working_exit_quantity_never_exceeds_the_position(case: Case) -> Violations:
    """Rule: the stop quantity equals the position, working targets never exceed it, and
    the remainder is the position less the working targets."""
    problems = []
    for row in case["expected"]["levels"]:
        held = abs(row["position_quantity_units"])
        where = f"{case['id']} {row['slice_ist']}: {row['instrument']}"
        if row["stop"] is not None and row["stop"]["quantity_units"] != held:
            problems.append(
                f"{where} stop quantity {row['stop']['quantity_units']} differs from "
                f"the position {held}"
            )
        working = sum(target["quantity_units"] for target in working_targets(row))
        if working > held:
            problems.append(f"{where} working targets {working} exceed the position {held}")
        if row["remainder_quantity_units"] != held - working:
            problems.append(
                f"{where} remainder {row['remainder_quantity_units']} is not the position "
                f"{held} less the working targets {working}"
            )
    return problems


def flat_rows_carry_no_levels(case: Case) -> Violations:
    """Rule: a flat row has no stop, targets, trail or breakeven."""
    problems = []
    for row in case["expected"]["levels"]:
        leftovers = row["stop"] or row["targets"] or row["trail"] or row["breakeven"]
        if row["position_quantity_units"] == 0 and leftovers:
            problems.append(
                f"{case['id']} {row['slice_ist']}: flat row of {row['instrument']} still has levels"
            )
    return problems


def levels_are_on_the_tick_grid(case: Case) -> Violations:
    """Rule: every stop, target and breakeven level is a whole number of ticks."""
    problems = []
    for row in case["expected"]["levels"]:
        tick_size = tick_size_of(case, row["instrument"])
        levels = [target["level_rupees"] for target in row["targets"]]
        if row["stop"] is not None:
            levels.append(row["stop"]["level_rupees"])
        if row["breakeven"] is not None:
            levels.append(row["breakeven"]["level_rupees"])
        for level in levels:
            if not is_on_tick(rupees(level), tick_size):
                problems.append(
                    f"{case['id']} {row['slice_ist']}: level {level} of {row['instrument']} "
                    f"is not a multiple of the tick {tick_size}"
                )
    return problems
