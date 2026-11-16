"""Held fills and held cancels, on the fake OpenAlgo (`_live/_stops_held.py`;
16-session-design 4.11 and 7.2 rows F34 and F68): a fill whose price the tradebook never
gives is released at the feed's price after 2 seconds, so the stop is placed; a stop's
cancel the tradebook cannot confirm is decided from the position book after 3 seconds,
with the units it filled before the cancel when the book shows fewer.

`StopHold` and `TargetHold` buy 10 SBIN at 14:42:01 IST with a stop at 497.00, the
latter with a target at 502.00, which SBIN reaches at 14:44:05 (`stops_samples`).
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, PRICE, alerts, ist
from stops_samples import StopHold, TargetHold, at_price, check_invariants, events, stop_world

from openalgo.backtest._live._stops_held import (
    HELD_CANCEL_DECIDE_AFTER_S,
    HELD_FILL_RELEASE_AFTER_S,
)
from openalgo.backtest._live._types import LOGGER_NAME

#: The tick at which SBIN reaches `TargetHold`'s target.
TARGET_TICK = ist(14, 44, 5)


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


def tradebook_fails(world, times: int) -> None:
    for _ in range(times):
        world.fake.faults.gateway_error("tradebook", status=503)


def test_F34_a_fill_the_tradebook_never_prices_is_released_at_the_feeds_price(world):
    world.fake.faults.zero_price_next_fill()
    tradebook_fails(world, 3)
    world.stop_at(ist(14, 44))
    outcome = world.run(StopHold, on_stop="keep")

    entry = world.orders()[0]
    (fill,) = [e for e in events(outcome, "fill") if e["order_id"] == entry.order_id]
    assert (fill["price"], fill["estimated"]) == (PRICE, True)
    (estimate,) = outcome.estimated
    assert (estimate.order_id, estimate.filled_quantity) == (entry.order_id, 10)
    assert estimate.at_ns - ist(14, 42, 3) >= HELD_FILL_RELEASE_AFTER_S * NS
    assert [(o.price_type, o.quantity) for o in world.orders()][1] == ("SL-M", 10)
    check_invariants(world.timeline, strategy="StopHold")


def test_a_fill_the_tradebook_prices_in_time_is_not_an_estimate(world):
    world.fake.faults.zero_price_next_fill()
    world.stop_at(ist(14, 44))
    outcome = world.run(StopHold, on_stop="keep")

    assert outcome.estimated == ()
    assert all(not event["estimated"] for event in events(outcome, "fill"))


def test_F68_a_held_cancel_the_position_book_shows_unfilled_is_released_and_exits(world):
    at_price(world, ist(14, 44), 502.0)
    world.script.at(TARGET_TICK, lambda: tradebook_fails(world, 40))
    outcome = world.run(TargetHold)

    stop, exit_ = [o for o in world.orders() if o.action == "SELL"]
    assert (stop.status, exit_.price_type, exit_.quantity) == ("cancelled", "MARKET", 10)
    (read,) = world.fake.log.calls("positionbook")
    assert read.ts_ns - TARGET_TICK >= HELD_CANCEL_DECIDE_AFTER_S * NS
    assert read.ts_ns <= exit_.placed_ns
    assert "held cancel" in [w.reason for w in outcome.unprotected]
    assert world.fake.position("SBIN", "NSE") == 0
    check_invariants(world.timeline, strategy="TargetHold")


def test_F68_a_held_cancel_of_a_stop_that_filled_part_releases_those_units_estimated(world, caplog):
    # The stop fills 4 units at the broker unseen (its words held and never sent); the
    # stream then reports its cancel with no fill, as a polled stream does.
    at_price(world, ist(14, 44), 502.0)

    def part_fill_unseen():
        world.fake.faults.hold_order_updates()
        tradebook_fails(world, 40)
        stop = next(o for o in world.orders() if o.price_type == "SL-M")
        world.fake.fill(stop.order_id, 4, price=499.0)

    def cancel_word_with_no_fill():
        stop = next(o for o in world.orders() if o.price_type == "SL-M")
        held = world.fake.log.of("order_update", "held")
        message = dict(held[-1].detail["message"])
        assert (message["orderid"], message["order_status"]) == (stop.order_id, "cancelled")
        message.update(filled_quantity=0, pending_quantity=0, average_price=0.0)
        world.feed.deliver_update(message)

    # After the target's tick, before its cancel reaches the broker: the stop, triggered
    # by nothing, keeps its other 6 units resting.
    world.script.at(TARGET_TICK + NS // 2, part_fill_unseen)
    world.script.at(TARGET_TICK + 3 * NS, cancel_word_with_no_fill)
    outcome = world.run(TargetHold)

    stop, exit_ = [o for o in world.orders() if o.action == "SELL"]
    assert (stop.filled, stop.status) == (4, "cancelled")
    assert (exit_.price_type, exit_.quantity, exit_.status) == ("MARKET", 6, "complete")
    # The exit's own fill is an estimate too: its words stay held and the tradebook fails.
    estimates = [(e.order_id, e.filled_quantity) for e in outcome.estimated]
    assert estimates == [(stop.order_id, 4), (exit_.order_id, 6)]
    assert world.fake.position("SBIN", "NSE") == 0
    assert alerts(caplog, "reconcile_mismatch") == []
    check_invariants(world.timeline, strategy="TargetHold")


def test_a_held_cancel_the_position_book_cannot_tell_is_alerted_and_read_again(world, caplog):
    at_price(world, ist(14, 44), 502.0)

    def books_fail():
        tradebook_fails(world, 40)
        for _ in range(3):
            world.fake.faults.gateway_error("positionbook", status=503)

    world.script.at(TARGET_TICK, books_fail)
    world.run(TargetHold)

    reads = world.fake.log.calls("positionbook")
    assert len(reads) >= 4
    assert any("cancelled, its fills not confirmed" in t for t in alerts(caplog, "error"))
    assert world.fake.position("SBIN", "NSE") == 0
    check_invariants(world.timeline, strategy="TargetHold")
