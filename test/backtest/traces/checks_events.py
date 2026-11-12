"""Consistency rules for the expected events of a case. One function per rule.

The expected events hold order events for orders the strategy placed, position events
and risk events (see `vocabulary.EVENT_NAMES`). This module holds the rules for their
order and for order and position events; `checks_risk_events` holds those for risk
events. Every rule takes a case and returns its violations; each violation names the
fixture, the slice and the rule that is broken.
"""

from __future__ import annotations

from typing import Any

from . import ledger
from . import vocabulary as vocab
from .timegrid import Case, parse_ist

Violations = list[str]
Event = dict[str, Any]

# Order events written when the strategy places the order (SPEC-04: an entry outside the
# entry window is rejected at submission).
SUBMISSION_EVENTS = ("Submitted", "Rejected")


def _events(case: Case, event_type: str) -> list[Event]:
    return [event for event in case["expected"]["events"] if event["type"] == event_type]


def events_are_in_sequence_order(case: Case) -> Violations:
    """Rule: events are listed in slice order and, inside a slice, in step order."""
    problems = []
    events = case["expected"]["events"]
    keys = [(parse_ist(event["slice_ist"]), event["step"]) for event in events]
    for earlier, later, event in zip(keys, keys[1:], events[1:], strict=False):
        if later < earlier:
            problems.append(
                f"{case['id']} {event['slice_ist']}: event {event['name']} is listed after "
                "a later slice or step"
            )
    return problems


def _position_events_from_fills(case: Case) -> list[tuple[Any, ...]]:
    """(slice, step, name, instrument, position) for every fill, in fill order."""
    derived = []
    held: dict[str, int] = {}
    for fill, after in ledger.position_history(case):
        before = held.get(fill["instrument"], 0)
        held[fill["instrument"]] = after.quantity_units
        name = "PositionChanged"
        if before == 0:
            name = "PositionOpened"
        elif after.quantity_units == 0:
            name = "PositionClosed"
        derived.append(
            (fill["slice_ist"], fill["step"], name, fill["instrument"], after.quantity_units)
        )
    return derived


def position_events_follow_the_fills(case: Case) -> Violations:
    """Rule: every fill produces one position event (opened from flat, closed to flat,
    changed otherwise) with the position it leaves, in fill order."""
    derived = _position_events_from_fills(case)
    stated = [
        (
            event["slice_ist"],
            event["step"],
            event["name"],
            event["instrument"],
            event["position_quantity_units"],
        )
        for event in _events(case, "position")
    ]
    for index in range(max(len(stated), len(derived))):
        from_fills = derived[index] if index < len(derived) else None
        in_fixture = stated[index] if index < len(stated) else None
        if from_fills != in_fixture:
            where = (from_fills or in_fixture)[0]
            return [
                f"{case['id']} {where}: position event {index}: the fills give "
                f"{from_fills}, the fixture has {in_fixture}"
            ]
    return []


def order_events_match_the_decisions(case: Case) -> Violations:
    """Rule: an order that passes submission is Submitted in step 7 of its decision slice
    and Filled where its fill is; a rejected order has a Rejected event and nothing else."""
    problems = []
    order_events = _events(case, "order")
    outcomes = case["expected"]["decisions"]
    known_orders = {outcome["order_id"] for outcome in outcomes}
    for decision, outcome in zip(case["strategy"]["decisions"], outcomes, strict=True):
        order_id = outcome["order_id"]
        if order_id is None:
            continue
        names = [event["name"] for event in order_events if event["order_id"] == order_id]
        expected_names = ["Submitted", "Filled"] if outcome["outcome"] == "filled" else ["Rejected"]
        if names != expected_names:
            problems.append(
                f"{case['id']} {decision['decision_slice_ist']}: order {order_id} is "
                f"{outcome['outcome']} and needs the events {expected_names}, it has {names}"
            )
    for event in order_events:
        if event["order_id"] not in known_orders:
            problems.append(
                f"{case['id']} {event['slice_ist']}: event {event['name']} names the "
                f"unknown order {event['order_id']}"
            )
    return problems + _order_event_placement(case, order_events)


def _order_event_placement(case: Case, order_events: list[Event]) -> Violations:
    problems = []
    decision_slices = {
        outcome["order_id"]: decision["decision_slice_ist"]
        for decision, outcome in zip(
            case["strategy"]["decisions"], case["expected"]["decisions"], strict=True
        )
    }
    fills = {fill["order_id"]: fill for fill in case["expected"]["fills"] if fill["order_id"]}
    for event in order_events:
        where = f"{case['id']} {event['slice_ist']}: {event['name']} event of {event['order_id']}"
        placed = (decision_slices.get(event["order_id"]), vocab.STEP_CALLBACKS)
        if event["name"] in SUBMISSION_EVENTS and (event["slice_ist"], event["step"]) != placed:
            problems.append(f"{where} must be in step 7 of the decision slice")
        fill = fills.get(event["order_id"])
        if event["name"] == "Filled" and fill is not None:
            stated = (
                event["slice_ist"],
                event["step"],
                event["quantity_units"],
                event["price_rupees"],
            )
            actual = (fill["slice_ist"], fill["step"], fill["quantity_units"], fill["price_rupees"])
            if stated != actual:
                problems.append(f"{where} does not repeat its fill {actual}")
    return problems
