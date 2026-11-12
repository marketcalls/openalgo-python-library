"""Consistency rules for the expected risk events of a case. One function per rule.

A risk event is written when an engine-held rule fires: a stop or target is hit, a trail
or breakeven arms, the daily loss limit is breached. Every rule takes a case and returns
its violations; each violation names the fixture, the slice and the rule that is broken.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from . import ledger
from .bars import bars_by_slice, rupees
from .timegrid import Case
from .timeline import has_fill_at_the_open, level_for_exit, row_in_force, rows_of, stop_level

Violations = list[str]
Event = dict[str, Any]

# Risk event written when an engine-held rule closes a position, by fill reason.
HIT_EVENT_BY_REASON = {
    "leg_sl": "leg_sl_hit",
    "leg_target": "leg_target_hit",
    "daily_loss_limit": "daily_loss_limit_hit",
}
# Exit reason of a leg level, by the risk event that reports it.
LEVEL_REASON_BY_EVENT = {"leg_sl_hit": "leg_sl", "leg_target_hit": "leg_target"}
# Level-row field that an arming event switches on.
ARMED_FIELD_BY_EVENT = {"leg_trail_armed": "trail", "leg_breakeven_armed": "breakeven"}
DAILY_LOSS_EVENT = "daily_loss_limit_hit"


def _risk_events(case: Case) -> list[Event]:
    return [event for event in case["expected"]["events"] if event["type"] == "risk"]


def _rule_fired_at(case: Case, fill: dict[str, Any]) -> tuple[str, int]:
    """Slice and step in which the rule behind an exit fired: the fill's own, except for the
    engine's exit after a GTT child did not fill, whose rule fired when the GTT did."""
    if "gtt_child_unfilled" not in fill["flags"]:
        return fill["slice_ist"], fill["step"]
    for event in case["expected"]["events"]:
        fired = event["type"] == "gtt" and event["name"] == "GttTriggered"
        if fired and event["gtt_id"] == fill["gtt_id"]:
            return event["slice_ist"], event["step"]
    return fill["slice_ist"], fill["step"]


def rule_exits_have_matching_risk_events(case: Case) -> Violations:
    """Rule: a stop, target or daily-loss exit has a risk event in the slice and step in
    which its rule fired (for a GTT exit, the step in which the GTT fired), and every such
    risk event has its exit."""
    exits = sorted(
        (*_rule_fired_at(case, fill), HIT_EVENT_BY_REASON[fill["reason"]], fill["target_index"])
        for fill in case["expected"]["fills"]
        if fill["reason"] in HIT_EVENT_BY_REASON
    )
    hits = sorted(
        (event["slice_ist"], event["step"], event["name"], event["target_index"])
        for event in _risk_events(case)
        if event["name"] in HIT_EVENT_BY_REASON.values()
    )
    problems = [
        f"{case['id']} {item[0]}: exit by rule has no risk event {item[2]} in step {item[1]}"
        for item in exits
        if item not in hits
    ]
    problems += [
        f"{case['id']} {item[0]}: risk event {item[2]} in step {item[1]} has no matching exit"
        for item in hits
        if item not in exits
    ]
    return problems


def hit_events_report_the_level_in_force(case: Case) -> Violations:
    """Rule: a stop or target event reports the level that was in force, which on a gap is
    not the fill price."""
    problems = []
    for event in _risk_events(case):
        reason = LEVEL_REASON_BY_EVENT.get(event["name"])
        instrument, slice_ist = event["instrument"], event["slice_ist"]
        if reason is None or has_fill_at_the_open(case, instrument, slice_ist):
            continue
        row = row_in_force(case, instrument, slice_ist)
        level = None if row is None else level_for_exit(row, reason, event["target_index"])
        if level is None or level != rupees(event["level_rupees"]):
            problems.append(
                f"{case['id']} {slice_ist}: {event['name']} reports {event['level_rupees']}, "
                f"the level in force is {level}"
            )
    return problems


def arming_events_match_the_timeline(case: Case) -> Violations:
    """Rule: a trail or breakeven arming event sits in the slice whose level row first
    shows it armed, and reports the stop level of that row."""
    problems = []
    for event in _risk_events(case):
        field = ARMED_FIELD_BY_EVENT.get(event["name"])
        if field is None:
            continue
        where = f"{case['id']} {event['slice_ist']}: {event['name']}"
        rows = [
            row
            for row in rows_of(case, event["instrument"])
            if row["slice_ist"] == event["slice_ist"]
        ]
        before = row_in_force(case, event["instrument"], event["slice_ist"])
        was_armed = bool(before and before[field] and before[field]["armed"])
        if not rows or not rows[0][field] or not rows[0][field]["armed"] or was_armed:
            problems.append(f"{where} is not where the timeline first shows it armed")
        elif stop_level(rows[0]) != rupees(event["level_rupees"]):
            problems.append(f"{where} reports {event['level_rupees']}, the row has another stop")
    return problems


def _run_mark_to_market(case: Case, slice_ist: str, before_step: int) -> Decimal:
    """Gross P&L of the run at a slice's close: realized so far plus open positions marked
    at that close, counting the fills of earlier steps only."""
    total = ledger.ZERO
    bars = bars_by_slice(case)
    for instrument in case["instruments"]:
        symbol = instrument["symbol"]
        position = ledger.position_after(case, symbol, slice_ist, last_step=before_step - 1)
        total += position.realized
        if position.quantity_units:
            close = bars[symbol][slice_ist].close
            total += position.quantity_units * (close - position.average_price)
    return total


def daily_loss_events_report_the_run_mark_to_market(case: Case) -> Violations:
    """Rule: a daily loss event reports minus the limit and, as the observed value, the
    run's gross P&L at that close, which is at or beyond the limit (the recommended
    reading of specification review item SPEC-16)."""
    problems = []
    for event in _risk_events(case):
        if event["name"] != DAILY_LOSS_EVENT:
            continue
        where = f"{case['id']} {event['slice_ist']}: {event['name']}"
        limit = -rupees(case["settings"]["max_daily_loss"])
        observed = _run_mark_to_market(case, event["slice_ist"], event["step"])
        if rupees(event["level_rupees"]) != limit:
            problems.append(f"{where} reports the limit {event['level_rupees']}, it is {limit}")
        if rupees(event["observed_rupees"]) != observed or observed > limit:
            problems.append(
                f"{where} observes {event['observed_rupees']}; the run's gross P&L at that "
                f"close is {observed} against the limit {limit}"
            )
    return problems
