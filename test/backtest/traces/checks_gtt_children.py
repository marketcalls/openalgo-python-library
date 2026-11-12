"""Consistency rules for the child order a fired GTT places. One function per rule.

04-execution 4.4 item 3: when a base bar reaches a GTT's trigger, or opens beyond it, the
GTT fires and places a child LIMIT beyond the trigger by the segment buffer (1 percent for
cash and futures, decision GTT-1), rounded to the tick. The child fills only if the price
trades through the limit after the trigger. When it has not filled through one base bar,
the engine raises a risk event, cancels the child, and then exits at the next base-bar
open, flagged (lead ruling LEAD-10).

Where the documents leave a value open the rules follow the recommended readings of the
specification review: a child that is marketable when placed fills at the trigger, or at
the open when the bar opened beyond the trigger (GTT-5); the limit is rounded away from
the trader (GTT-6); a bar that opens beyond the limit and later trades back to it fills
the child at the limit (GTT-7). Every trace runs at zero slippage.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

from openalgo.backtest.reference import round_to_tick

from . import ledger
from . import vocabulary as vocab
from .bars import Bar, bars_by_slice, rupees, tick_size_of
from .gtt_state import Event, active_gtts, events_of, gtt_events, moment, trigger_of
from .timegrid import Case, parse_ist
from .timeline import is_beyond

Violations = list[str]
# Price and where it filled, or None when the child cannot fill in its bar.
Outcome = tuple[str, Decimal] | None

EXIT_REASON_BY_LEG = {"stop": "leg_sl", "target": "leg_target"}
UNFILLED = "gtt_child_unfilled"


def _instrument_type(case: Case, instrument: str) -> str:
    return next(item["type"] for item in case["instruments"] if item["symbol"] == instrument)


def _sells(case: Case, trigger: Event) -> bool:
    """True when the child order sells, that is when the position it protects is long."""
    position = ledger.position_after(
        case, trigger["instrument"], trigger["slice_ist"], last_step=vocab.STEP_FILLS_AT_OPEN
    )
    return position.quantity_units > 0


def expected_child_limit(case: Case, trigger: Event, sells: bool) -> Decimal:
    """The child LIMIT: the fired trigger moved by the buffer against the child's side,
    rounded to the tick away from the trader (down for a sell, up for a buy)."""
    buffer = Decimal(vocab.GTT_CHILD_BUFFER_PERCENT[_instrument_type(case, trigger["instrument"])])
    level = trigger_of(trigger, trigger["fired_leg"])
    raw = level * (100 - buffer) / 100 if sells else level * (100 + buffer) / 100
    tick = tick_size_of(case, trigger["instrument"])
    return rupees(round_to_tick(float(raw), float(tick), mode="down" if sells else "up"))


def child_outcome(bar: Bar, trigger: Event, sells: bool) -> Outcome:
    """Where and at what price the child fills in the bar its GTT fired in, or None."""
    level = trigger_of(trigger, trigger["fired_leg"])
    limit = rupees(trigger["child_limit_rupees"])
    approached_upward = (trigger["fired_leg"] == "target") == sells
    if not is_beyond(bar.open, level, approached_upward):
        return "level", level
    if is_beyond(bar.open, limit, upward=sells):
        return "open", bar.open
    best_after_trigger = bar.high if sells else bar.low
    if is_beyond(best_after_trigger, limit, upward=sells):
        return "level", limit
    return None


def gtt_children_carry_the_buffer(case: Case) -> Violations:
    """Rule: a fired GTT fires on a trigger it holds and the bar reaches, and its child is a
    LIMIT beyond that trigger by the segment buffer, rounded to the tick away from the
    trader."""
    problems = []
    for trigger in _fired(case):
        where = f"{case['id']} {trigger['slice_ist']}: GttTriggered {trigger['gtt_id']}"
        problems += _trigger_in_force_violations(case, trigger, where)
        sells = _sells(case, trigger)
        limit = expected_child_limit(case, trigger, sells)
        if rupees(trigger["child_limit_rupees"]) != limit:
            problems.append(
                f"{where}: child limit {trigger['child_limit_rupees']}, expected {limit}"
            )
    return problems


def _trigger_in_force_violations(case: Case, trigger: Event, where: str) -> Violations:
    """The fired trigger is the one the GTT held, and the bar of that slice reaches it."""
    in_force = active_gtts(case, moment(trigger)).get(trigger["instrument"])
    level = trigger_of(trigger, trigger["fired_leg"])
    if in_force is None or in_force["gtt_id"] != trigger["gtt_id"]:
        return [f"{where} fires a GTT that is not in force"]
    if trigger_of(in_force, trigger["fired_leg"]) != level:
        return [f"{where} fires at {level}, not at the trigger in force"]
    bar = bars_by_slice(case)[trigger["instrument"]].get(trigger["slice_ist"])
    approached_upward = (trigger["fired_leg"] == "target") == _sells(case, trigger)
    if bar is None:
        return [f"{where} fires in a slice without a bar"]
    reached = bar.high if approached_upward else bar.low
    if not is_beyond(reached, level, approached_upward):
        return [f"{where}: the bar does not reach the trigger {level}"]
    return []


def gtt_children_fill_by_the_limit_rule(case: Case) -> Violations:
    """Rule: the child fills at the trigger when the bar reached it inside its range, at the
    open when the bar opened beyond the trigger but not beyond the limit, at the limit
    when the bar opened beyond the limit and later traded back to it, and otherwise is
    cancelled unfilled; the fill and the GTT's events say the same."""
    problems = []
    bars = bars_by_slice(case)
    for trigger in _fired(case):
        bar = bars[trigger["instrument"]].get(trigger["slice_ist"])
        if bar is None:
            continue
        sells = _sells(case, trigger)
        outcome = child_outcome(bar, trigger, sells)
        if outcome is None:
            problems += _unfilled_violations(case, trigger, bar, sells)
        else:
            problems += _filled_violations(case, trigger, outcome, sells)
    return problems


def _fired(case: Case) -> list[Event]:
    return [event for event in gtt_events(case) if event["name"] == "GttTriggered"]


def _resolution(case: Case, trigger: Event) -> Event | None:
    for event in events_of(case, trigger["gtt_id"]):
        if event["name"] in ("GttChildFilled", "GttChildCanceled"):
            return event
    return None


def _gtt_fills(case: Case, gtt_id: str) -> list[dict[str, Any]]:
    return [fill for fill in case["expected"]["fills"] if fill["gtt_id"] == gtt_id]


def _filled_violations(case: Case, trigger: Event, outcome: Outcome, sells: bool) -> Violations:
    where = f"{case['id']} {trigger['slice_ist']}: child of {trigger['gtt_id']}"
    fill_at, price = outcome
    resolution = _resolution(case, trigger)
    if resolution is None or resolution["name"] != "GttChildFilled":
        return [f"{where} must fill at {price} ({fill_at}), the fixture does not fill it"]
    problems = []
    if rupees(resolution["price_rupees"]) != price:
        problems.append(
            f"{where}: GttChildFilled at {resolution['price_rupees']}, expected {price}"
        )
    expected_fill = (
        trigger["slice_ist"],
        vocab.STEP_LEVELS,
        "SELL" if sells else "BUY",
        trigger["quantity_units"],
        price,
        fill_at,
        EXIT_REASON_BY_LEG[trigger["fired_leg"]],
    )
    fills = [
        (
            f["slice_ist"],
            f["step"],
            f["side"],
            f["quantity_units"],
            rupees(f["price_rupees"]),
            f["fill_at"],
            f["reason"],
        )
        for f in _gtt_fills(case, trigger["gtt_id"])
    ]
    if fills != [expected_fill]:
        problems.append(f"{where}: the fill must be {expected_fill}, the fixture has {fills}")
    return problems


def _unfilled_violations(case: Case, trigger: Event, bar: Bar, sells: bool) -> Violations:
    where = f"{case['id']} {trigger['slice_ist']}: child of {trigger['gtt_id']}"
    resolution = _resolution(case, trigger)
    if resolution is None or resolution["name"] != "GttChildCanceled":
        return [f"{where} cannot fill in its bar and must be cancelled (GttChildCanceled)"]
    problems = []
    best = bar.high if sells else bar.low
    risk = [
        (rupees(e["level_rupees"]), rupees(e["observed_rupees"]))
        for e in case["expected"]["events"]
        if e["type"] == "risk" and e["name"] == UNFILLED and e["slice_ist"] == trigger["slice_ist"]
    ]
    if risk != [(rupees(trigger["child_limit_rupees"]), best)]:
        problems.append(f"{where}: needs one {UNFILLED} risk event at its limit, observing {best}")
    later = [
        t
        for t in bars_by_slice(case)[trigger["instrument"]]
        if parse_ist(t) > parse_ist(trigger["slice_ist"])
    ]
    if not later:
        return problems
    next_bar = bars_by_slice(case)[trigger["instrument"]][later[0]]
    expected_fill = (later[0], vocab.STEP_FILLS_AT_OPEN, next_bar.open, "open", [UNFILLED])
    fills = [
        (f["slice_ist"], f["step"], rupees(f["price_rupees"]), f["fill_at"], f["flags"])
        for f in _gtt_fills(case, trigger["gtt_id"])
    ]
    if fills != [expected_fill]:
        problems.append(
            f"{where}: the engine's exit must be {expected_fill}, the fixture has {fills}"
        )
    return problems
