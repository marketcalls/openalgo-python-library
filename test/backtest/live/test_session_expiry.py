"""Derivatives held into the last trading day of their contract in a live session, on the
fake OpenAlgo (design 20-positional-live section 9, decision 11.2; the lead's ruling N5b-3;
the 13.3 rows "stock future and an option exited on expiry day" and "an index future settled
overnight"): the session's kernel takes each instrument's positional exit day as
`bt.backtest` does, so an option bought NRML is exited at the NFO square-off (15:10 IST) of
its last trading day, its stop GTT cancelled first, and an entry after it is refused
`contract_expiring`; an index future is left to the exchange, its stop GTT cancelled at the
close with no alert of a missing stop. An MCX contract settled by delivery whose tender
period the reference tables do not give is refused at start for CNC and NRML.

The markets are `expiry_samples`'. Expected values come from the scenario and the calendar.
"""

from __future__ import annotations

import logging

import pandas as pd
import pytest
from expiry_samples import (
    FUTURE,
    FUTURE_DAY,
    LOT,
    OPTION,
    OPTION_DAY,
    PRICE,
    STOCK_FUTURE,
    STOCK_LOT,
    expiry_world,
    ist,
    stock_future_tables,
)
from fake_openalgo.fixtures import *  # noqa: F403
from plan_samples import specify
from positional_samples import MODES, calls
from session_samples import alerts

import openalgo.backtest as bt
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME
from openalgo.backtest.errors import ConfigError

#: The strategies' capital and daily loss limit, rupees.
LIMITS = {"capital": 100_000, "max_daily_loss": 5_000}


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


class OptionHold(bt.Strategy):
    """Buys one lot of the option NRML at its first decision with a stop 10 points below;
    flat again, it buys once more (refused after the contract's exit)."""

    symbol, exchange, interval, product = OPTION, "NFO", "1m", "NRML"

    def init(self):
        self.calls = 0
        self.refused: list[str] = []

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1 or (self.position.is_flat and not self.refused):
            order = self.buy(LOT, stop=bt.points(10))
            if order is not None and order.reject_code is not None:
                self.refused.append(order.reject_code)


class FutureHold(OptionHold):
    """OptionHold on the index future of its last trading day, a stop 50 points below."""

    symbol = FUTURE

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(LOT, stop=bt.points(50))


class GoldHold(bt.Strategy):
    """NRML on MCX's GOLD of December 2026, settled by delivery, whose tender period the
    reference table mcx_contracts does not give."""

    symbol, exchange, interval, product = "GOLD05DEC26FUT", "MCX", "1m", "NRML"

    def on_bar(self, bar):
        pass


def ist_time(ts_ns: int) -> pd.Timestamp:
    return pd.Timestamp(ts_ns, tz="UTC").tz_convert("Asia/Kolkata")


def test_an_mcx_delivery_contract_with_no_tender_period_is_refused_for_cnc_and_nrml():
    with pytest.raises(ConfigError) as refused:
        specify(GoldHold)

    assert str(refused.value) == (
        "GoldHold: GOLD05DEC26FUT is settled by delivery, and the reference table "
        "mcx_contracts gives no tender period for it, so a live session cannot tell the day "
        "to exit a CNC or NRML position on it before MCX marks it for delivery. Trade it in "
        "MIS, or trade a contract the table gives a tender period for"
    )


def test_an_mis_strategy_on_that_contract_is_not_refused():
    class GoldDay(GoldHold):
        product = "MIS"

    (each,) = specify(GoldDay).strategies
    assert each.product == "MIS"


@MODES
def test_an_option_held_into_its_last_trading_day_exits_at_the_square_off_gtt_cancelled_first(
    make_fake_openalgo, monkeypatch, tmp_path, caplog, analyzer
):
    world = expiry_world(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        symbol=OPTION,
        day=OPTION_DAY,
        start=(15, 0, 5),
        analyzer=analyzer,
    )
    world.stop_at(ist(OPTION_DAY, 15, 14))

    outcome = world.run(OptionHold, **LIMITS)

    result = session_result(outcome).results["OptionHold"]
    (trade,) = result.trades.itertuples()
    # Bought at the first decision (15:02:01), exited at 15:10, the NFO square-off.
    assert trade.exit_reason == "expiry_exit"
    assert (trade.entry_units, trade.exit_price_rupees) == (LOT, PRICE)
    assert ist_time(trade.exit_ts).strftime("%H:%M") == "15:10"
    # The stop GTT's cancel went before the exit order.
    order = [c.what for c in calls(world, "placegttorder", "cancelgttorder", "placeorder")]
    assert order == ["placeorder", "placegttorder", "cancelgttorder", "placeorder"]
    sells = [o for o in world.orders() if o.action == "SELL"]
    assert [(o.price_type, o.quantity, o.status) for o in sells] == [("MARKET", LOT, "complete")]
    mode = "SANDBOX" if analyzer else "LIVE"
    assert any(
        text.startswith(
            f"{mode} (OptionHold): SELL {LOT} {OPTION} NFO at 120.00 (expiry_exit: exit "
            "before the contract's expiry"
        )
        for text in alerts(caplog, "exit")
    )
    # Flat after the exit, the strategy buys again: refused at submission, nothing sent.
    assert outcome.plans["OptionHold"].strategy.refused == ["contract_expiring"]
    assert len([o for o in world.orders() if o.action == "BUY"]) == 1
    # The result says why, in words that hold for a live session too (it takes no roll=).
    (told,) = [w for w in result.warnings if "contract_expiring" in w]
    assert told.endswith(
        "this is the expiry rule working, the contract being in its last trading minutes or "
        "past them: trade the next contract by naming it as the strategy's symbol (in a "
        "backtest, a futures strategy with roll= moves to it for you)"
    )


class StockFutureHold(OptionHold):
    """Buys one lot of the stock future NRML at its first decision, a stop 10 points below."""

    symbol = STOCK_FUTURE

    def on_bar(self, bar):
        self.calls += 1
        if self.calls == 1:
            self.buy(STOCK_LOT, stop=bt.points(10))


@MODES
def test_a_stock_future_held_into_its_last_trading_day_exits_at_the_square_off_gtt_cancelled_first(
    make_fake_openalgo, monkeypatch, tmp_path, analyzer
):
    # Settled by delivery, so exited as an option is (13.3 "a stock future and an option
    # exited on expiry day"); SBIN's derivatives are planned on the test's own reference rows.
    stock_future_tables(monkeypatch, tmp_path)
    world = expiry_world(
        make_fake_openalgo,
        monkeypatch,
        tmp_path,
        symbol=STOCK_FUTURE,
        day=FUTURE_DAY,
        start=(15, 0, 5),
        analyzer=analyzer,
        lot=STOCK_LOT,
    )
    world.stop_at(ist(FUTURE_DAY, 15, 14))

    outcome = world.run(StockFutureHold, **LIMITS)

    (trade,) = session_result(outcome).results["StockFutureHold"].trades.itertuples()
    assert trade.exit_reason == "expiry_exit"
    assert (trade.entry_units, trade.exit_price_rupees) == (STOCK_LOT, PRICE)
    assert ist_time(trade.exit_ts).strftime("%H:%M") == "15:10"
    order = [c.what for c in calls(world, "placegttorder", "cancelgttorder", "placeorder")]
    assert order == ["placeorder", "placegttorder", "cancelgttorder", "placeorder"]
    sells = [o for o in world.orders() if o.action == "SELL"]
    assert [(o.price_type, o.quantity, o.status) for o in sells] == [
        ("MARKET", STOCK_LOT, "complete")
    ]


def test_an_index_future_on_its_last_trading_day_is_left_to_the_exchange_at_the_close(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    world = expiry_world(
        make_fake_openalgo, monkeypatch, tmp_path, symbol=FUTURE, day=FUTURE_DAY, start=(15, 20, 5)
    )

    outcome = world.run(FutureHold, **LIMITS)

    assert outcome.stopped_by == "session_end"
    assert world.fake.position(FUTURE, "NFO") == LOT
    # The kernel cancels the stop GTT at the close of the last trading day.
    assert [g.status for g in world.fake.gtts()] == ["cancelled"]
    assert (
        f"FutureHold {FUTURE} NFO {LOT} units: the exchange settles it at its expiry today, so "
        "its stop GTT was cancelled at the close; the next start, on the contract traded then, "
        "names the settlement and today's last close"
    ) in world.lines
    assert alerts(caplog, "error") == []
    # The close's Carried alert and summary line say so too (the lead's ruling N5c-4 (h)).
    (carried,) = [a for a in alerts(caplog, "session") if "Carried overnight:" in a]
    assert "the exchange settles it at its expiry today, so its stop GTT was cancelled" in carried
    assert "no stop at the broker" not in carried
    (summary,) = alerts(caplog, "daily_summary")
    assert "the exchange settles it at its expiry today" in summary


def test_an_index_future_under_rearm_daily_on_its_last_day_is_left_to_the_exchange_unwarned(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # Its stop is a day stop, not a GTT, and the exchange settles it: neither the line nor
    # the close's WARN alert of legs without a stop until the next start names a GTT or it.
    world = expiry_world(
        make_fake_openalgo, monkeypatch, tmp_path, symbol=FUTURE, day=FUTURE_DAY, start=(15, 20, 5)
    )

    outcome = world.run(FutureHold, positional_stops="rearm_daily", **LIMITS)

    assert outcome.stopped_by == "session_end"
    assert world.fake.gtts() == []
    assert (
        f"FutureHold {FUTURE} NFO {LOT} units: the exchange settles it at its expiry today, so "
        "it needs no stop from the close; the next start, on the contract traded then, names "
        "the settlement and today's last close"
    ) in world.lines
    assert alerts(caplog, "warn") == []
    (carried,) = [a for a in alerts(caplog, "session") if "Carried overnight:" in a]
    assert "the exchange settles it at its expiry today, so it needs no stop" in carried
    assert "until your next start" not in carried
    warned = session_result(outcome).warnings
    assert not [w for w in warned if w.endswith("no stop at the broker until your next start")]
    assert alerts(caplog, "error") == []
