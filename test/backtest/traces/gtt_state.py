"""The GTT orders of a case, replayed from its expected events (04-execution 4.4).

On a CNC or NRML position the stop and the target are held as a GTT: a SINGLE GTT for a
stop or a target alone, one OCO GTT for both. A GTT is placed (GttPlaced), may be replaced
by a trail or breakeven (GttModified, in force from the next base bar), and ends either
cancelled by the engine (GttCanceled) or fired by a bar (GttTriggered), after which its
child order fills (GttChildFilled) or is cancelled unfilled (GttChildCanceled).
"""

from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Any

from . import vocabulary as vocab
from .bars import rupees
from .timegrid import Case, parse_ist

Event = dict[str, Any]
Moment = tuple[datetime, int]

# Events that set the triggers a GTT holds, and events after which it holds none.
SETTING_NAMES = ("GttPlaced", "GttModified")
ENDING_NAMES = ("GttTriggered", "GttCanceled")


def moment(item: dict[str, Any]) -> Moment:
    """When an event or a fill happened: (slice open as an instant, step of the sequence)."""
    return parse_ist(item["slice_ist"]), item["step"]


def end_of_slice(slice_ist: str) -> Moment:
    """A moment after every step of a slice."""
    return parse_ist(slice_ist), vocab.LAST_STEP + 1


def is_positional(case: Case, instrument: str) -> bool:
    """True when the instrument's product holds its stops and targets as GTTs (CNC, NRML)."""
    for declared in case["instruments"]:
        if declared["symbol"] == instrument:
            return declared["product"] in vocab.GTT_PRODUCTS
    raise KeyError(f"{case['id']}: instrument {instrument!r} is not declared")


def gtt_events(case: Case) -> list[Event]:
    """The GTT events of a case, in the order listed."""
    return [event for event in case["expected"]["events"] if event["type"] == "gtt"]


def events_of(case: Case, gtt_id: str) -> list[Event]:
    """The events of one GTT, in the order listed."""
    return [event for event in gtt_events(case) if event["gtt_id"] == gtt_id]


def active_gtts(case: Case, before: Moment) -> dict[str, Event]:
    """GTTs in force by instrument: the latest GttPlaced or GttModified of every GTT not
    yet fired or cancelled, counting only events that happened strictly before `before`."""
    active: dict[str, Event] = {}
    for event in gtt_events(case):
        if moment(event) >= before:
            continue
        if event["name"] in SETTING_NAMES:
            active[event["instrument"]] = event
        elif event["name"] in ENDING_NAMES:
            current = active.get(event["instrument"])
            if current is not None and current["gtt_id"] == event["gtt_id"]:
                del active[event["instrument"]]
    return active


def trigger_of(event: Event, leg: str) -> Decimal | None:
    """The trigger price in rupees of one leg ("stop" or "target") of a GTT event."""
    value = event[f"{leg}_trigger_rupees"]
    return None if value is None else rupees(value)
