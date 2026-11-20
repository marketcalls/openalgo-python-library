"""A held cancel beside another member's unheard fill, on the fake OpenAlgo
(`_live/_stops_held.py`, `_live/_stops_held_group.py`, `_live/_worker_send.py`;
18-multi-strategy 4.5 and row F103 of 5.2, the lead's decision M1, unit M7).

A's target cancels its stop to exit; the broker cancels it with nothing filled, but the
trade book cannot be read, so the cancel is held. B's stop meanwhile fills at the broker,
its word held on the stream. The position book shows the group's net, which B's fill
moved: the targeted read takes the order book after it, sees B's stop complete while B's
engine works it, and decides nothing until B's fill reaches B's engine; then the net
against the members' sum says A's stop filled nothing, and A exits.

The market and the strategies are `shared_samples`': SBIN at 500.00; A (`SbinTrend`) buys
100 at 14:42:01 IST with a stop at 497.00 and a target at 502.00, B (`SbinBreakout`) 50
with a stop at 497.00. SBIN trades at 502.00 from 14:44, reaching A's target at 14:44:05.
Expected values come from the fake's books and the scenario.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist
from shared_samples import A_UNITS, B_UNITS, SbinBreakout, SbinTrend, sbin_orders
from stops_samples import at_price, stop_world

from openalgo.backtest._live._types import LOGGER_NAME

#: The tick at which SBIN reaches A's target, and when B's stop fills unheard.
TARGET_TICK = ist(14, 44, 5)
B_FILLS = TARGET_TICK + NS // 2
#: When A's cancel word, with no fill, reaches the session, and when B's held word does.
A_CANCELLED = TARGET_TICK + 3 * NS
B_HEARD = TARGET_TICK + 12 * NS


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def stop_of(world, name: str):
    return next(o for o in sbin_orders(world, name) if o.price_type == "SL-M")


def b_fills_unheard(world, *, lag_positions: bool = False) -> None:
    """B's stop fills whole at the broker, its word held on the stream, and the trade book
    fails from now on; with `lag_positions`, the position book answers as before the fill
    (the fill lands between the read's two books)."""
    world.fake.faults.hold_order_updates()
    for _ in range(60):
        world.fake.faults.gateway_error("tradebook", status=503)
    if lag_positions:
        world.fake.lag_positionbook()
    world.fake.fill(stop_of(world, "SbinBreakout").order_id, price=497.0)


def a_cancel_heard_with_no_fill(world) -> None:
    """A's stop's cancel word reaches the session saying nothing filled (a polled stream
    always says 0 filled), while B's fill stays held."""
    stop = stop_of(world, "SbinTrend")
    held = [h.detail["message"] for h in world.fake.log.of("order_update", "held")]
    (message,) = [dict(m) for m in held if m["orderid"] == stop.order_id]
    assert message["order_status"] == "cancelled"
    message.update(filled_quantity=0, pending_quantity=0, average_price=0.0)
    world.feed.deliver_update(message)


def run(world, *, lag_positions: bool = False):
    at_price(world, ist(14, 44), 502.0)
    world.script.at(B_FILLS, lambda: b_fills_unheard(world, lag_positions=lag_positions))
    world.script.at(A_CANCELLED, lambda: a_cancel_heard_with_no_fill(world))
    world.script.at(B_HEARD, world.fake.faults.release_order_updates)
    if lag_positions:
        world.script.at(B_HEARD, world.fake.catch_up_positionbook)
    world.stop_at(ist(14, 46))
    return world.run([SbinTrend, SbinBreakout], on_stop="keep")


def events(outcome, kind: str) -> list[dict]:
    """The run's event-stream lines of `kind`, in order."""
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        lines = [json.loads(line) for line in stream]
    return [event for event in lines if event["kind"] == kind]


def a_exits(world) -> list:
    return [
        o
        for o in sbin_orders(world, "SbinTrend")
        if o.price_type == "MARKET" and o.action == "SELL"
    ]


def test_F103_a_held_cancel_waits_while_another_members_fill_is_unheard_then_exits(world):
    outcome = run(world)
    # The two-book read decided it, after B's fill was heard: A's stop filled nothing.
    stop_a = stop_of(world, "SbinTrend").order_id
    (released,) = [e for e in events(outcome, "held_released") if e["order_id"] == stop_a]
    assert (released["filled"], released["ts_ns"] >= B_HEARD) == (0, True)
    (exit_,) = a_exits(world)
    # No exit of A before B's fill reached B's engine; then the whole 100, once.
    assert exit_.placed_ns >= B_HEARD
    assert (exit_.quantity, exit_.status) == (A_UNITS, "complete")
    assert stop_of(world, "SbinTrend").status == "cancelled"
    assert world.fake.position("SBIN", "NSE") == 0


def test_F103_the_read_takes_the_order_book_after_the_position_book(world):
    run(world)
    positions = [c.ts_ns for c in world.fake.log.calls("positionbook")]
    orders = [c.ts_ns for c in world.fake.log.calls("orderbook")]
    waited = [ts for ts in positions if A_CANCELLED + 3 * NS <= ts < B_HEARD]
    assert waited and all(any(ts <= o for o in orders) for ts in waited)


def test_F103_an_error_alert_says_why_the_cancel_waits(world, caplog):
    run(world)
    stop_a, stop_b = stop_of(world, "SbinTrend"), stop_of(world, "SbinBreakout")
    waiting = [t for t in alerts(caplog, "error") if f"stop {stop_a.order_id} of SbinTrend" in t]
    assert len(waiting) == 1
    assert (
        f"its fills not confirmed (the stop {stop_b.order_id} of SbinBreakout shows a fill its "
        "engine has not taken); no exit of SbinTrend is sent until the books tell"
    ) in waiting[0]


def test_F103_the_wait_counts_unprotected_as_a_held_cancel(world):
    outcome = run(world)
    reasons = {(w.strategy, w.reason) for w in outcome.unprotected}
    assert ("SbinTrend", "held cancel") in reasons


def test_F103_variant_a_fill_landing_between_the_two_books_stops_the_decision_too(world):
    # The position book answers as before B's fill, the order book after it: B's stop
    # shows complete, so the read decides nothing (never the reverse).
    run(world, lag_positions=True)
    (exit_,) = a_exits(world)
    assert exit_.placed_ns >= B_HEARD
    assert exit_.quantity == A_UNITS
    assert world.fake.position("SBIN", "NSE") == 0
    assert stop_of(world, "SbinBreakout").quantity == B_UNITS
