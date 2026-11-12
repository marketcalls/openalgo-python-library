"""Consistency rules for the GTTs that hold positional stops and targets. One per rule.

04-execution 4.4: on a CNC or NRML position a stop or target is a GTT, MIS holds none, a
trail or breakeven modifies the GTT at most once per base bar, and every exit the engine
makes itself cancels the GTT first (cancel-then-exit, 08-live-trading 6.2). The child
order a fired GTT places is checked in `checks_gtt_children`.

Every rule takes a case and returns its violations; each violation names the fixture,
the slice and the rule that is broken.
"""

from __future__ import annotations

import re

from . import ledger
from . import vocabulary as vocab
from .bars import rupees, tick_size_of
from .gtt_state import (
    Event,
    active_gtts,
    end_of_slice,
    events_of,
    gtt_events,
    is_positional,
    moment,
    trigger_of,
)
from .timegrid import Case
from .timeline import LevelRow, stop_level, working_targets

Violations = list[str]

# One letter per event name; a GTT's life is placed, modified any number of times, then
# either cancelled, or fired and its child filled or cancelled (or still working at the end).
_LETTERS = {
    "GttPlaced": "P",
    "GttModified": "M",
    "GttCanceled": "C",
    "GttTriggered": "T",
    "GttChildFilled": "F",
    "GttChildCanceled": "X",
}
_LIFE_CYCLE = re.compile(r"PM*(C|T[FX])?")
# The step in which each event happens (03-architecture 4.2): placed with the opening
# fill (step 2), modified by a ratchet (step 4), fired and resolved while the bar is
# tested (step 3). A cancellation happens in the step of the engine's exit.
_STEP_OF = {
    "GttPlaced": vocab.STEP_FILLS_AT_OPEN,
    "GttModified": 4,
    "GttTriggered": vocab.STEP_LEVELS,
    "GttChildFilled": vocab.STEP_LEVELS,
    "GttChildCanceled": vocab.STEP_LEVELS,
}


def intraday_positions_hold_no_gtts(case: Case) -> Violations:
    """Rule: a GTT exists only on a CNC or NRML instrument (MIS cannot hold one)."""
    problems = [
        f"{case['id']} {event['slice_ist']}: {event['name']} on {event['instrument']}, "
        "whose product cannot hold a GTT"
        for event in gtt_events(case)
        if not is_positional(case, event["instrument"])
    ]
    problems += [
        f"{case['id']} {fill['slice_ist']}: fill on {fill['instrument']} names GTT "
        f"{fill['gtt_id']}, but the product cannot hold one"
        for fill in case["expected"]["fills"]
        if fill["gtt_id"] is not None and not is_positional(case, fill["instrument"])
    ]
    return problems


def gtt_life_cycles_are_complete(case: Case) -> Violations:
    """Rule: a GTT is placed, then modified any number of times, then either cancelled or
    fired with its child filled or cancelled; each event in its own step, on one
    instrument, and a fired GTT's events in one slice."""
    problems = []
    for gtt_id in dict.fromkeys(event["gtt_id"] for event in gtt_events(case)):
        events = events_of(case, gtt_id)
        where = f"{case['id']} {events[0]['slice_ist']}: GTT {gtt_id}"
        letters = "".join(_LETTERS[event["name"]] for event in events)
        if not _LIFE_CYCLE.fullmatch(letters):
            problems.append(f"{where} has the life cycle {[e['name'] for e in events]}")
        if len({event["instrument"] for event in events}) != 1:
            problems.append(f"{where} names more than one instrument")
        for event in events:
            step = _STEP_OF.get(event["name"])
            if step is not None and event["step"] != step:
                problems.append(f"{where}: {event['name']} must be in step {step}")
        fired = [
            e for e in events if e["name"] in ("GttTriggered", "GttChildFilled", "GttChildCanceled")
        ]
        if len({event["slice_ist"] for event in fired}) > 1:
            problems.append(f"{where}: a fired GTT's child is resolved in the slice it fired")
    return problems


def gtt_modifications_follow_the_rules(case: Case) -> Violations:
    """Rule: a GTT is modified at most once per base bar and only when a trigger moves by at
    least one tick (decision GTT-3)."""
    problems = []
    for gtt_id in dict.fromkeys(event["gtt_id"] for event in gtt_events(case)):
        events = [e for e in events_of(case, gtt_id) if e["name"] in ("GttPlaced", "GttModified")]
        for earlier, later in zip(events, events[1:], strict=False):
            where = f"{case['id']} {later['slice_ist']}: GttModified of {gtt_id}"
            if earlier["name"] == "GttModified" and earlier["slice_ist"] == later["slice_ist"]:
                problems.append(f"{where} is the second modification in one base bar")
            tick = tick_size_of(case, later["instrument"])
            moves = [
                abs(trigger_of(later, leg) - trigger_of(earlier, leg))
                for leg in vocab.GTT_LEGS
                if trigger_of(later, leg) is not None and trigger_of(earlier, leg) is not None
            ]
            if not moves or max(moves) < tick:
                problems.append(f"{where} does not move a trigger by at least one tick")
    return problems


def gtts_are_placed_with_the_fill_that_opens_the_position(case: Case) -> Violations:
    """Rule: a GTT is placed in the step of the fill that opened its position (an entry or
    the next contract's roll fill), for the whole position."""
    problems = []
    openings = {}
    held: dict[str, int] = {}
    for fill, after in ledger.position_history(case):
        if held.get(fill["instrument"], 0) == 0 and after.quantity_units != 0:
            openings[(fill["slice_ist"], fill["step"], fill["instrument"])] = after.quantity_units
        held[fill["instrument"]] = after.quantity_units
    for event in gtt_events(case):
        if event["name"] != "GttPlaced":
            continue
        quantity = openings.get((event["slice_ist"], event["step"], event["instrument"]))
        if quantity is None or abs(quantity) != event["quantity_units"]:
            problems.append(
                f"{case['id']} {event['slice_ist']}: GttPlaced {event['gtt_id']} does not "
                f"follow a fill that opened a position of {event['quantity_units']} units"
            )
    return problems


def _held_levels(row: LevelRow) -> tuple[object, object] | None:
    """(stop, target) a GTT must hold for a level row, or None when the row needs several
    target GTTs, which no trace models."""
    targets = working_targets(row)
    if len(targets) > 1:
        return None
    target = rupees(targets[0]["level_rupees"]) if targets else None
    return stop_level(row), target


def positional_levels_are_held_by_gtts(case: Case) -> Violations:
    """Rule: at the end of every slice with a level row, a CNC or NRML position's stop and
    target are exactly the triggers of the GTT in force, and a flat position holds none."""
    problems = []
    for row in case["expected"]["levels"]:
        instrument = row["instrument"]
        if not is_positional(case, instrument):
            continue
        where = f"{case['id']} {row['slice_ist']}: {instrument}"
        active = active_gtts(case, end_of_slice(row["slice_ist"])).get(instrument)
        held = None if row["position_quantity_units"] == 0 else _held_levels(row)
        if row["position_quantity_units"] != 0 and held is None:
            problems.append(f"{where} has several targets, which one OCO GTT cannot hold")
            continue
        if held in (None, (None, None)):
            if active is not None:
                problems.append(f"{where} holds no levels but GTT {active['gtt_id']} is in force")
            continue
        triggers = (
            None if active is None else tuple(trigger_of(active, leg) for leg in vocab.GTT_LEGS)
        )
        if triggers != held:
            problems.append(f"{where}: the levels {held} are not the triggers in force {triggers}")
        elif active["quantity_units"] != abs(row["position_quantity_units"]):
            problems.append(f"{where}: GTT {active['gtt_id']} is not for the whole position")
    return problems


def engine_exits_cancel_the_gtt_first(case: Case) -> Violations:
    """Rule: before the engine closes or reduces a CNC or NRML position itself (any exit that
    is not a GTT's own: the roll, the daily loss limit, a signal), the GTT in force on it is
    cancelled in the same step, before the exit's position event."""
    problems = []
    events = case["expected"]["events"]
    held: dict[str, int] = {}
    for fill, after in ledger.position_history(case):
        instrument = fill["instrument"]
        before, held[instrument] = held.get(instrument, 0), after.quantity_units
        reduces = abs(after.quantity_units) < abs(before)
        if not reduces or fill["gtt_id"] is not None or not is_positional(case, instrument):
            continue
        active = active_gtts(case, moment(fill)).get(instrument)
        if active is None:
            continue
        cancel = _event_index(events, fill, "gtt", "GttCanceled", active["gtt_id"])
        position = _event_index(events, fill, "position", None, None)
        if cancel is None or position is None or cancel > position:
            problems.append(
                f"{case['id']} {fill['slice_ist']}: the {fill['reason']} exit on {instrument} "
                f"is not preceded by GttCanceled of {active['gtt_id']} in step {fill['step']}"
            )
    return problems


def _event_index(
    events: list[Event], fill: Event, event_type: str, name: str | None, gtt_id: str | None
) -> int | None:
    """Index of the first event of a type (and name, GTT) in the fill's slice and step on
    its instrument."""
    for index, event in enumerate(events):
        same_place = (event["slice_ist"], event["step"]) == (fill["slice_ist"], fill["step"])
        if not same_place or event["type"] != event_type:
            continue
        if event.get("instrument") != fill["instrument"]:
            continue
        if name is not None and (event["name"], event["gtt_id"]) != (name, gtt_id):
            continue
        return index
    return None
