"""What the operator is told of a position left open, on the fake OpenAlgo
(`_live/_stops_report.py`; 16-session-design 1.4 `open_positions`, 3.3 step 4; the unit 8
hand-off on the left-open note): each order still working toward the position named by
its broker order ID, else its side, units, purpose and intent; an order sent and
unanswered told from one resting at the broker; an entry still working named too; and
the stop, confirmed or not.

The session starts at 14:40:05 IST; SBIN trades at 500.00 every 10 seconds.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, ist
from stops_samples import STOP_POINTS, stop_world

import openalgo.backtest as bt
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


def test_a_resting_sale_is_named_by_its_order_id_as_resting_at_the_broker(world):
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
    assert left.note.endswith("no stop at the broker, close it by hand")


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
