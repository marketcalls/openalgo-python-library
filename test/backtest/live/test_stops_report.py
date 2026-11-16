"""What the operator is told of a position left open, on the fake OpenAlgo
(`_live/_stops_report.py`; 16-session-design 1.4 `open_positions`, 3.3 step 4; the unit 8
hand-off on the left-open note): each order still working toward the position named by
its broker order ID, else its side, units, purpose and intent; an order sent and
unanswered, or a market order accepted with no final word, told from one resting at the
broker; an entry still working named too; the stop, confirmed or not; and, while any
order works, the order book before closing it by hand.

The session starts at 14:40:05 IST; SBIN trades at 500.00 every 10 seconds.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, ist
from stops_samples import STOP_POINTS, StopHold, stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._stops_report import CHECK_FIRST
from openalgo.backtest._live._types import LOGGER_NAME


@pytest.fixture
def world(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return stop_world(make_fake_openalgo, monkeypatch, tmp_path)


class RestsAnExit(bt.Strategy):
    """Buys 10 SBIN at its first decision (no stop) and rests a LIMIT sale of them at 600
    at its second."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10)
        elif self.calls == 2:
            self.sell(10, price=600.0)


def test_a_resting_sale_is_named_as_resting_and_the_order_book_comes_first(world):
    # The safe stop cancels the resting sale (not reduce-only, it is classed with the
    # entries), and the cancel never reaches the broker: the sale rests on.
    world.script.at(
        ist(14, 44) - NS // 2, lambda: world.fake.faults.bad_gateway_page("cancelorder")
    )
    world.stop_at(ist(14, 44))
    outcome = world.run(RestsAnExit, on_stop="keep")

    resting = next(o for o in world.orders() if o.action == "SELL")
    (left,) = outcome.open_positions
    assert resting.status == "open"
    assert f"order {resting.order_id} resting at the broker" in left.note
    assert "in flight" not in left.note
    # A sale by hand beside the resting one would sell short: never a bare "close it".
    assert left.note.endswith(CHECK_FIRST)


class AddsResting(bt.Strategy):
    """Buys 10 SBIN with a stop at its first decision and rests a LIMIT add of 5 at 400 at
    its second."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.calls = 0

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(10, stop=bt.points(STOP_POINTS))
        elif self.calls == 2:
            self.buy(5, price=400.0, stop=bt.points(STOP_POINTS))


def test_an_entry_still_working_is_named_with_the_confirmed_stop(world):
    # The cancel of the resting add never reaches the broker: the add still works.
    world.script.at(
        ist(14, 44) - NS // 2, lambda: world.fake.faults.bad_gateway_page("cancelorder")
    )
    world.stop_at(ist(14, 44))
    outcome = world.run(AddsResting, on_stop="keep")

    add = next(o for o in world.orders() if (o.action, o.price_type) == ("BUY", "LIMIT"))
    stop = next(o for o in world.orders() if o.price_type == "SL-M")
    (left,) = outcome.open_positions
    assert add.status == "open"
    assert f"entry order {add.order_id} resting at the broker" in left.note
    assert f"stop {stop.order_id} confirmed at the broker" in left.note


def test_an_exit_sent_and_unanswered_is_named_by_side_units_and_intent(world):
    class Holds(bt.Strategy):
        symbol, exchange, interval = "SBIN", "NSE", "1m"

        def init(self):
            self.calls = 0

        def on_bar(self, bar):
            self.calls += 1
            if self.calls == 1:
                self.buy(10, stop=bt.points(STOP_POINTS))

    # The safe stop's exit never reaches the broker and its answer is lost.
    world.script.at(
        ist(14, 43, 31) + NS // 2, lambda: world.fake.faults.bad_gateway_page("placeorder")
    )
    world.stop_at(ist(14, 43, 30))
    world.stop_at(ist(14, 43, 34))
    outcome = world.run(Holds)

    (left,) = outcome.open_positions
    assert "exit order SELL 10 (risk_exit, intent 20261116-live-1-Holds-o3)" in left.note
    assert "in flight, sent and unanswered" in left.note
    assert "check the order book before closing it by hand" in left.note


def test_an_exit_accepted_with_no_final_word_is_in_flight_and_never_beside_close_it_by_hand(
    world,
):
    def stream_held_and_reads_failing():
        world.fake.faults.hold_order_updates()
        for _ in range(20):
            world.fake.faults.gateway_error("orderstatus", status=503)
            world.fake.faults.gateway_error("tradebook", status=503)

    world.stop_at(ist(14, 43, 30))
    world.script.at(ist(14, 43, 32) + NS // 2, stream_held_and_reads_failing)
    world.stop_at(ist(14, 43, 33) + NS // 2)
    outcome = world.run(StopHold)

    exit_ = next(o for o in world.orders() if (o.action, o.price_type) == ("SELL", "MARKET"))
    (left,) = outcome.open_positions
    assert f"exit order {exit_.order_id} in flight, accepted with no final word" in left.note
    assert "resting at the broker" not in left.note
    assert left.note.endswith(CHECK_FIRST)
    assert "close it by hand" not in left.note.replace(CHECK_FIRST, "")
