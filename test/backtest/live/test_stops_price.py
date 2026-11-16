"""Triggered but unfilled: the price rule, on the fake OpenAlgo (`_live/_stops_price.py`;
16-session-design 4.6 and 7.2 rows F23 and F72): an option's stop-limit the price runs
through is repriced to a marketable LIMIT 2 seconds after the crossing, a partly filled
one for its total units; a reprice the broker refuses exits the rest at a limit.

The strategy is `OptionHold`: one lot (65 units) of the option bought at 120.00 at 14:42:01
IST with a stop 10 points below, an SL at 110.00 whose limit is 104.50 (`stops_samples`).
The option trades every 10 seconds at the test's price.
"""

from __future__ import annotations

import logging

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_samples import NS, alerts, ist
from stops_samples import OPTION, OPTION_LOT, OptionHold, check_invariants, events, option_world

from openalgo.backtest._live._stops_price import TRIGGERED_UNFILLED_AFTER_S
from openalgo.backtest._live._types import LOGGER_NAME

#: The tick at which the option first trades through its stop.
CROSSED_AT = ist(14, 44, 5)


@pytest.fixture
def option(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    return option_world(make_fake_openalgo, monkeypatch, tmp_path)


def falls_to(option, price: float, at_ns: int = CROSSED_AT) -> None:
    option.script.at(at_ns - NS // 2, lambda: setattr(option, "option_price", price))


def sells(option) -> list[tuple]:
    return [
        (o.quantity, o.price_type, o.filled, o.status)
        for o in option.orders()
        if o.action == "SELL"
    ]


def test_F23_a_stop_limit_run_through_is_repriced_to_a_marketable_limit_after_2_seconds(option):
    falls_to(option, 100.0)
    option.run(OptionHold)

    (modify,) = option.fake.log.calls("modifyorder")
    request = modify.detail["request"]
    assert modify.ts_ns >= CROSSED_AT + TRIGGERED_UNFILLED_AFTER_S * NS
    # 5 percent below the last price of 100.00, on the tick: 95.00.
    assert (request["pricetype"], float(request["price"])) == ("LIMIT", 95.0)
    assert int(request["quantity"]) == OPTION_LOT
    assert sells(option) == [(OPTION_LOT, "LIMIT", OPTION_LOT, "complete")]
    assert option.fake.position(OPTION, "NFO") == 0
    check_invariants(option.timeline, strategy="OptionHold")


def test_a_stop_reached_and_filled_at_once_is_never_repriced(option):
    falls_to(option, 108.0)
    outcome = option.run(OptionHold)

    assert option.fake.log.calls("modifyorder") == []
    assert sells(option) == [(OPTION_LOT, "SL", OPTION_LOT, "complete")]
    assert events(outcome, "repriced") == []


class TwoLots(OptionHold):
    """OptionHold that buys two lots."""

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(2 * OPTION_LOT, **type(self).rules)


def part_fill_then_through(option) -> None:
    """The stop fills one lot of two at 108.00 as the price crosses it, then the price
    falls to 100.00, below its limit, a second later."""
    option.script.at(CROSSED_AT - NS // 2, lambda: option.fake.split_next_fill(OPTION_LOT))
    falls_to(option, 108.0)
    falls_to(option, 100.0, CROSSED_AT + NS)
    option.script.at(
        CROSSED_AT + NS, lambda: option.fake.trade(OPTION, "NFO", 100.0, volume=OPTION_LOT)
    )


def test_F72_a_partly_filled_stop_is_repriced_for_its_total_units(option):
    part_fill_then_through(option)
    option.run(TwoLots)

    (modify,) = option.fake.log.calls("modifyorder")
    request = modify.detail["request"]
    assert (request["pricetype"], int(request["quantity"])) == ("LIMIT", 2 * OPTION_LOT)
    assert sells(option) == [(2 * OPTION_LOT, "LIMIT", 2 * OPTION_LOT, "complete")]
    assert option.fake.position(OPTION, "NFO") == 0
    check_invariants(option.timeline, strategy="TwoLots")


def test_F72_a_reprice_refused_exits_the_rest_at_a_marketable_limit(option, caplog):
    part_fill_then_through(option)
    option.script.at(CROSSED_AT + NS, lambda: option.fake.faults.rate_limit("modifyorder"))
    option.run(TwoLots)

    stop, exit_ = [o for o in option.orders() if o.action == "SELL"]
    assert (stop.price_type, stop.filled, stop.status) == ("SL", OPTION_LOT, "cancelled")
    assert (exit_.price_type, exit_.quantity, exit_.status) == ("LIMIT", OPTION_LOT, "complete")
    assert exit_.price == 95.0
    assert any("did not fill" in text for text in alerts(caplog, "error"))
    assert option.fake.position(OPTION, "NFO") == 0
    check_invariants(option.timeline, strategy="TwoLots")
