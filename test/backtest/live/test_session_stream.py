"""A live session's fills through faults of the order-updates stream, on the fake OpenAlgo
(`_live/session.py`; 16-session-design 2.6, 6.6 and 7.2 rows F29 to F33, F59, F89 and
F90, unit 8, run without leg rules): a silent stream, lost, repeated, late and
out-of-order updates, a price-0 fill, other clients' orders, and an entry whose fill is
reported after the next decision.

The strategy is `BuyAndHold` (buys 10 SBIN at its first decision, 14:42:01 IST, sells
three bars later) unless a test names another; a baseline run without the fault gives
the trades and fills each faulted run must match.
"""

from __future__ import annotations

import json
import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import BuyAndHold, BuyWhenFlat, alerts, ist, rig

import openalgo.backtest as bt
from openalgo.backtest._live._types import LOGGER_NAME


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return rig(make_fake_openalgo, monkeypatch, tmp_path)


@pytest.fixture
def baseline(make_fake_openalgo, monkeypatch, tmp_path_factory):
    """The trades of a run with no fault, by (units, entry price, exit price, exit
    reason)."""
    folder = tmp_path_factory.mktemp("baseline")
    return trades_of(rig(make_fake_openalgo, monkeypatch, folder).run(BuyAndHold))


def trades_of(outcome, name: str = "BuyAndHold") -> list[tuple]:
    table = outcome.outputs[name]["trades"]
    columns = ("entry_units", "entry_price_rupees", "exit_price_rupees", "exit_reason")
    return list(zip(*(table[column] for column in columns), strict=True))


def fills_of(outcome, name: str = "BuyAndHold") -> list[tuple]:
    table = outcome.outputs[name]["fills"]
    return list(zip(table["quantity_units"], table["price_rupees"], strict=True))


def fill_events(outcome) -> list[dict]:
    with open(outcome.paths["events"], encoding="utf-8") as stream:
        return [event for line in stream if (event := json.loads(line))["kind"] == "fill"]


def test_F29_with_the_order_stream_silent_all_day_polls_give_the_same_trades(world, baseline):
    world.fake.faults.silence_order_updates()
    outcome = world.run(BuyAndHold)
    assert trades_of(outcome) == baseline
    assert {event["source"] for event in fill_events(outcome)} == {"poll"}


def test_F90_when_the_stream_resumes_the_trades_are_the_baseline_ones(world, baseline):
    world.fake.faults.silence_order_updates()
    world.script.at(ist(14, 43, 30), world.fake.faults.resume_order_updates)
    outcome = world.run(BuyAndHold)
    assert trades_of(outcome) == baseline
    sources = [event["source"] for event in fill_events(outcome)]
    assert sources == ["poll", "stream"]


def test_F30_an_entry_fill_whose_update_is_lost_is_found_by_polling(world, baseline):
    # The entry's two updates (open, then complete) are both lost.
    world.fake.faults.drop_next_order_update()
    world.fake.faults.drop_next_order_update()
    outcome = world.run(BuyAndHold)
    assert trades_of(outcome) == baseline
    assert fill_events(outcome)[0]["source"] == "poll"


def test_F31_an_update_that_arrives_twice_is_one_fill(world, caplog):
    world.fake.faults.duplicate_next_order_update()
    world.fake.faults.duplicate_next_order_update()
    outcome = world.run(BuyAndHold)
    assert fills_of(outcome) == [(10.0, 500.0), (10.0, 500.0)]
    assert len(alerts(caplog, "entry")) == 1
    assert alerts(caplog, "reconcile_mismatch") == []


def test_F32_updates_released_newest_first_fill_the_whole_order_at_its_average(world):
    world.fake.faults.hold_order_updates()
    world.fake.split_next_fill(4)
    world.script.price = lambda now_ns: 501.0 if now_ns >= ist(14, 42, 5) else 500.0
    world.script.at(
        ist(14, 42, 30), lambda: world.fake.faults.release_order_updates(newest_first=True)
    )
    outcome = world.run(BuyAndHold)
    entry = outcome.outputs["BuyAndHold"]["trades"]
    assert entry["entry_units"] == [10.0]
    assert entry["entry_price_rupees"][0] == pytest.approx((4 * 500.0 + 6 * 501.0) / 10)


def test_F33_a_fill_reported_at_price_0_is_priced_from_the_tradebook(world):
    world.fake.faults.zero_price_next_fill()
    outcome = world.run(BuyAndHold)
    assert fills_of(outcome)[0] == (10.0, 500.0)
    assert fill_events(outcome)[0]["derived_from_average"] is False


def test_F59_another_clients_orders_on_the_stream_change_nothing(world, baseline):
    other = world.fake.client()
    world.script.at(
        ist(14, 43, 10),
        lambda: other.placeorder(
            strategy="manual",
            symbol="SBIN",
            action="BUY",
            exchange="NSE",
            price_type="MARKET",
            product="MIS",
            quantity=7,
        ),
    )
    outcome = world.run(BuyAndHold)
    assert trades_of(outcome) == baseline
    assert outcome.mode_breach is None
    assert world.fake.position("SBIN", "NSE") == 7


def test_F89_an_entry_whose_fill_is_late_blocks_a_second_entry_live_in_flight(world):
    world.fake.faults.hold_order_updates()
    for _ in range(90):
        world.fake.faults.gateway_error("orderstatus", status=503)
    world.script.at(ist(14, 43, 30), world.fake.faults.release_order_updates)
    outcome = world.run(BuyWhenFlat)
    buys = [order for order in world.orders() if order.action == "BUY"]
    assert len(buys) == 1
    orders = outcome.outputs["BuyWhenFlat"]["orders"]
    assert "live_in_flight" in orders["reject_code"]
    assert world.fake.position("SBIN", "NSE") == 0


class RestingBuy(bt.Strategy):
    """Rests a LIMIT buy of 10 SBIN far below the market at its first decision."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def on_bar(self, bar):
        if not self.orders and not self.position.qty:
            self.buy(10, price=400.0)


def test_a_fill_the_kernel_refuses_locks_the_strategys_entries_and_is_alerted(world, caplog):
    def overfill():
        # The stream reports 15 units filled of the resting order of 10.
        sent = world.fake.log.of("order_update", "sent")
        message = dict(sent[-1].detail["message"])
        message.update(quantity=15, filled_quantity=15, pending_quantity=0, average_price=400.0)
        message["order_status"] = "complete"
        world.feed.deliver_update(message)

    world.script.at(ist(14, 43, 30), overfill)
    world.stop_at(ist(14, 45))
    world.run(RestingBuy)
    first = alerts(caplog, "reconcile_mismatch")[0]
    assert "differ on order" in first and "15 filled" in first
    locks = [r.getMessage() for r in caplog.records if "too_many_units" in r.getMessage()]
    assert locks
    assert [order.action for order in world.orders()] == ["BUY"]


def test_after_a_refused_fill_the_order_still_resting_at_the_broker_is_cancelled(world, caplog):
    # The stream's word ended the order in the engine while it rests at the broker: an
    # orphan of the session's own, which reconciliation cancels (16-session-design 5.3).
    def overfill():
        sent = world.fake.log.of("order_update", "sent")
        message = dict(sent[-1].detail["message"])
        message.update(quantity=15, filled_quantity=15, pending_quantity=0, average_price=400.0)
        message["order_status"] = "complete"
        world.feed.deliver_update(message)

    world.script.at(ist(14, 43, 30), overfill)
    world.stop_at(ist(14, 45))
    outcome = world.run(RestingBuy)
    assert [order.status for order in world.orders()] == ["cancelled"]
    assert "orphan cancelled" in [row.action for row in outcome.reconciliations]
    assert outcome.summary.reconcile_mismatches == 2
