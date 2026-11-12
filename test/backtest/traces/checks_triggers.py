"""Consistency rules that tie the fills to the levels in force. One function per rule.

These rules re-test step 3 of the sequence (03-architecture 4.2) from the level timeline
and the bars alone, without the fixture's derivation: a bar that reaches a level in
force must produce a fill, and a level fill must be priced from the level in force.

Slices in which the instrument has a fill at the open (an entry, an add, a square-off, a
roll) are left out: there the levels change in step 2 and the timeline, which records
the end of each slice, does not show the levels step 3 used.

On a CNC or NRML position the levels are a GTT's triggers (04-execution 4.4): a bar that
reaches one fires the GTT, and its exit is either the child's fill in that slice or, when
the child cannot fill, the engine's exit at the next bar open. The price of a GTT's exit
is checked in `checks_gtt_children`, not here.
"""

from __future__ import annotations

from collections.abc import Iterator
from decimal import Decimal

from .bars import Bar, bars_by_slice, rupees
from .timegrid import Case
from .timeline import (
    LevelRow,
    fills_at,
    has_fill_at_the_open,
    is_beyond,
    is_long,
    level_for_exit,
    row_in_force,
    stop_level,
    working_targets,
)

Violations = list[str]
Trigger = tuple[str, int | None]

LEVEL_REASONS = ("leg_sl", "leg_target")
REASON_BY_GTT_LEG = {"stop": "leg_sl", "target": "leg_target"}


def _carried_positions(case: Case) -> Iterator[tuple[str, str, Bar, LevelRow]]:
    """Bars that an open position enters with levels unchanged by a fill at the open."""
    for instrument, bars in bars_by_slice(case).items():
        for slice_ist, bar in bars.items():
            row = row_in_force(case, instrument, slice_ist)
            if row is None or row["position_quantity_units"] == 0:
                continue
            if has_fill_at_the_open(case, instrument, slice_ist):
                continue
            yield instrument, slice_ist, bar, row


def _expected_triggers(bar: Bar, row: LevelRow) -> set[Trigger]:
    """What step 3 must fill for one bar: a level crossed at the open first, then the stop
    before any target inside the range."""
    long = is_long(row)
    stop = stop_level(row)
    targets = [(target["index"], rupees(target["level_rupees"])) for target in working_targets(row)]
    if stop is not None and is_beyond(bar.open, stop, upward=not long):
        return {("leg_sl", None)}
    crossed_at_open = {
        ("leg_target", index) for index, level in targets if is_beyond(bar.open, level, long)
    }
    if crossed_at_open:
        return crossed_at_open
    adverse, favourable = (bar.low, bar.high) if long else (bar.high, bar.low)
    if stop is not None and is_beyond(adverse, stop, upward=not long):
        return {("leg_sl", None)}
    return {("leg_target", index) for index, level in targets if is_beyond(favourable, level, long)}


def _gtt_fires_left_unfilled(
    case: Case, instrument: str, slice_ist: str, row: LevelRow
) -> set[Trigger]:
    """Levels whose GTT fired in this slice but whose child was cancelled unfilled; the exit
    then comes at the next bar open, so the slice has no fill for them."""
    unfilled = {
        event["gtt_id"]
        for event in case["expected"]["events"]
        if event["type"] == "gtt"
        and event["name"] == "GttChildCanceled"
        and event["slice_ist"] == slice_ist
    }
    triggers = set()
    for event in case["expected"]["events"]:
        fired = event["type"] == "gtt" and event["name"] == "GttTriggered"
        if not fired or event["gtt_id"] not in unfilled or event["instrument"] != instrument:
            continue
        reason = REASON_BY_GTT_LEG[event["fired_leg"]]
        index = None if reason == "leg_sl" else working_targets(row)[0]["index"]
        triggers.add((reason, index))
    return triggers


def no_level_in_force_is_missed(case: Case) -> Violations:
    """Rule: a bar that reaches a stop or target in force has exactly the exits step 3
    prescribes (a fill, or a fired GTT whose child could not fill), and a bar that reaches
    none has no level exit."""
    problems = []
    for instrument, slice_ist, bar, row in _carried_positions(case):
        expected = _expected_triggers(bar, row)
        actual = {
            (fill["reason"], fill["target_index"])
            for fill in fills_at(case, instrument, slice_ist)
            if fill["reason"] in LEVEL_REASONS
        } | _gtt_fires_left_unfilled(case, instrument, slice_ist, row)
        if expected != actual:
            problems.append(
                f"{case['id']} {slice_ist}: {instrument} bar {bar.low} to {bar.high} against "
                f"the levels in force requires {sorted(expected, key=str)}, the fixture has "
                f"{sorted(actual, key=str)}"
            )
    return problems


def level_fills_are_priced_from_the_level_in_force(case: Case) -> Violations:
    """Rule: an engine-held stop or target (MIS) fills at its level when the bar opens short
    of it, and at the open when the open has already reached it."""
    problems = []
    bars = bars_by_slice(case)
    for fill in case["expected"]["fills"]:
        instrument, slice_ist = fill["instrument"], fill["slice_ist"]
        if fill["reason"] not in LEVEL_REASONS or fill["gtt_id"] is not None:
            continue
        if has_fill_at_the_open(case, instrument, slice_ist):
            continue
        where = f"{case['id']} {slice_ist}: {fill['reason']} fill"
        row = row_in_force(case, instrument, slice_ist)
        level = None if row is None else level_for_exit(row, fill["reason"], fill["target_index"])
        if level is None:
            problems.append(f"{where} has no level in force in the timeline")
            continue
        toward_higher = is_long(row) == (fill["reason"] == "leg_target")
        open_reached = is_beyond(bars[instrument][slice_ist].open, level, toward_higher)
        price = rupees(fill["price_rupees"])
        if fill["fill_at"] == "open" and not open_reached:
            problems.append(f"{where} is at the open, but the open has not reached {level}")
        if fill["fill_at"] == "level" and (open_reached or price != level):
            problems.append(f"{where} at {price} is not a fill at the level in force {level}")
    return problems


def roll_shifts_levels_by_the_spread(case: Case) -> Violations:
    """Rule: after a roll the stop and the favourable extreme of the new contract are those
    of the old one shifted by the roll spread, never looser."""
    problems = []
    rolls = [fill for fill in case["expected"]["fills"] if fill["reason"] == "roll"]
    for closing, opening in zip(rolls[0::2], rolls[1::2], strict=True):
        slice_ist = opening["slice_ist"]
        before = row_in_force(case, closing["instrument"], slice_ist)
        after = [
            row
            for row in case["expected"]["levels"]
            if row["instrument"] == opening["instrument"] and row["slice_ist"] == slice_ist
        ]
        if before is None or not after or after[0]["position_quantity_units"] == 0:
            continue
        spread = rupees(opening["price_rupees"]) - rupees(closing["price_rupees"])
        bar = bars_by_slice(case)[opening["instrument"]][slice_ist]
        problems += _shift_violations(case, before, after[0], spread, bar)
    return problems


def _shift_violations(
    case: Case, before: LevelRow, after: LevelRow, spread: Decimal, bar: Bar
) -> Violations:
    where = f"{case['id']} {after['slice_ist']}: roll with spread {spread}"
    long = is_long(after)
    problems = []
    extreme_moved = False
    if before["trail"] is not None and after["trail"] is not None:
        shifted = rupees(before["trail"]["favourable_extreme_rupees"]) + spread
        expected = max(shifted, bar.high) if long else min(shifted, bar.low)
        stated = rupees(after["trail"]["favourable_extreme_rupees"])
        extreme_moved = expected != shifted
        if stated != expected:
            problems.append(f"{where}: favourable extreme is {stated}, expected {expected}")
    old_stop, new_stop = stop_level(before), stop_level(after)
    if old_stop is not None and new_stop is not None:
        shifted_stop = old_stop + spread
        looser = new_stop < shifted_stop if long else new_stop > shifted_stop
        if looser or (not extreme_moved and new_stop != shifted_stop):
            problems.append(f"{where}: stop is {new_stop}, the shifted stop is {shifted_stop}")
    return problems
