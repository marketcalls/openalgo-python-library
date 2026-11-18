"""Parity of a live session and a backtest of the bars it ran on (15-phase2-plan section 4:
"the same strategy on a recorded session and on the bars built from it in a backtest takes
the same decisions"; 16-session-design section 8 unit 16).

The session runs on the fake OpenAlgo under the recorder (`session_replay`): SBIN on NSE
from 14:40:05 IST, trading 10 units every 10 s at 500.00, then at 497.50 from 14:46:20.
`Parity` decides from the 14:42 close on: it buys 10 with a stop 2 points below, sells
them two bars later, buys 10 again with the same stop, and that stop is hit. Ctrl+C at
14:49 keeps the leg. The record's bars (the warm-up's candles, today's candles of the
catch-up and the bars the session built from ticks) are then run as a backtest of the
same strategy with the same settings (`bt.backtest(..., data=...)`).

The two take the same decisions: the same entries and exits, at the same bars, for the
same units and reasons, with stops at the same distance from each run's entry. What the
design makes differ is asserted as that difference:

- **Fill prices.** Live fills at the broker's price (the fake's trade), with no modelled
  slippage; the backtest fills at the next bar's open, or at the stop's level, moved by
  its slippage model against the trader.
- **Where the stop is held.** Live, each leg's stop is an SL-M order at the broker (D31),
  placed with the entry's fill and cancelled before a signal exit (one exit owner); the
  backtest holds it as a level of the leg, with no order.
- **An exit's units.** The live exit names the units it closes, as a broker order must;
  the backtest's closes whatever the position holds (0 units asked, the position filled).

Expected values come from the scenario, the fake's books and the backtest's own bars.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_replay import record_sessions
from session_samples import CAPITAL, DAY, MAX_DAILY_LOSS, ist
from stops_samples import stop_world

import openalgo.backtest as bt
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME

#: The stop's distance below each entry, points, and the units of each entry.
STOP_POINTS = 2.0
UNITS = 10
#: The price SBIN falls to at `FALL_AT`, through the second entry's stop.
FALL_PRICE, FALL_AT = 497.5, ist(14, 46, 20)


class Parity(bt.Strategy):
    """From the bar closing at `trade_from` on: buys `UNITS` with a stop `STOP_POINTS`
    below when flat, at most twice; sells the first position two bars after it filled."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"
    trade_from = ist(14, 42)

    def init(self):
        self.state.setdefault("entries", 0)
        self.held = 0

    def on_bar(self, bar):
        if bar.ts_close < type(self).trade_from:
            return
        if self.position.qty:
            self.held += 1
            if self.held >= 2 and self.state["entries"] == 1:
                self.exit()
            return
        if not self.orders and self.state["entries"] < 2:
            self.state["entries"] += 1
            self.held = 0
            self.buy(UNITS, stop=bt.points(STOP_POINTS))


@pytest.fixture
def runs(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """(the live session's Result, the backtest's Result of its bars, the bars, the
    fake)."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.script.price = lambda now_ns: FALL_PRICE if now_ns >= FALL_AT else 500.0
    recorder = record_sessions(monkeypatch)
    world.stop_at(ist(14, 49))
    outcome = world.run(Parity, on_stop="keep")
    (record,) = recorder.finish()
    bars = record.bars("Parity")
    backtest = bt.backtest(
        Parity, data=bars, capital=CAPITAL, max_daily_loss=MAX_DAILY_LOSS, start=DAY, end=DAY
    )
    return session_result(outcome).results["Parity"], backtest, bars, world.fake


def rows(frame: Any, *columns: str) -> list[tuple]:
    return [tuple(row) for row in frame.loc[:, list(columns)].itertuples(index=False)]


def strategy_orders(result: Any) -> Any:
    """The orders the strategy decided: all but the broker-held stops."""
    orders = result.orders
    return orders[orders["order_type"] != "SL-M"]


def test_the_backtest_of_the_sessions_bars_makes_the_same_trades_on_the_same_bars(runs):
    live, backtest, _, _ = runs
    columns = ("entry_slice", "exit_slice", "direction", "entry_units", "max_units", "exit_reason")
    assert rows(live.trades, *columns) == rows(backtest.trades, *columns)
    assert [row[-1] for row in rows(live.trades, *columns)] == ["signal", "leg_sl"]


def test_the_strategys_orders_are_decided_at_the_same_bars_with_the_same_terms(runs):
    live, backtest, _, _ = runs
    columns = ("decided_at", "side", "order_type", "filled_units", "status")
    assert rows(strategy_orders(live), *columns) == rows(strategy_orders(backtest), *columns)


def test_each_fill_is_on_the_same_slice_and_side_for_the_same_reason(runs):
    live, backtest, _, _ = runs
    columns = ("slice", "side", "quantity_units", "reason")
    assert rows(live.fills, *columns) == rows(backtest.fills, *columns)


def test_the_stops_are_the_same_distance_below_each_runs_entry(runs):
    live, backtest, _, _ = runs
    risk = [STOP_POINTS * UNITS] * 2
    assert list(live.trades["initial_risk_rupees"]) == risk
    assert list(backtest.trades["initial_risk_rupees"]) == risk


def test_live_fills_are_at_the_brokers_prices_with_no_modelled_slippage(runs):
    live, _, _, fake = runs
    broker = [fill.price for fill in fake.fills()]
    assert list(live.fills["price_rupees"]) == broker == [500.0, 500.0, 500.0, FALL_PRICE]
    assert live.fills["slippage_rupees"].isna().all()


def test_backtest_fills_are_the_next_open_or_the_stops_level_moved_by_its_slippage(runs):
    _, backtest, bars, _ = runs
    fills = backtest.fills
    moved = fills["slippage_rupees"] / fills["quantity_units"]
    against = fills["side"].map({"BUY": 1.0, "SELL": -1.0}) * moved
    at_open = fills["fill_at"] == "open"
    opens = bars["open"].reindex(fills.loc[at_open, "time"]).to_numpy()
    assert list(fills.loc[at_open, "price_rupees"]) == pytest.approx(opens + against[at_open])
    # The stop's level is its entry's fill less the stop's distance.
    entry = backtest.trades["entry_price_rupees"].iloc[-1]
    (stop_fill,) = fills.loc[fills["reason"] == "leg_sl", "price_rupees"]
    assert stop_fill == pytest.approx(
        entry - STOP_POINTS + against[fills["reason"] == "leg_sl"].item()
    )


def test_live_holds_each_stop_at_the_broker_as_an_sl_m_order_the_backtest_as_a_level(runs):
    live, backtest, _, _ = runs
    entries = live.trades["entry_price_rupees"].tolist()
    stops = live.orders[live.orders["order_type"] == "SL-M"]
    assert rows(stops, "side", "quantity_units", "status") == [
        ("SELL", UNITS, "Canceled"),
        ("SELL", UNITS, "Filled"),
    ]
    assert list(stops["trigger_rupees"]) == [price - STOP_POINTS for price in entries]
    assert "SL-M" not in set(backtest.orders["order_type"])


def test_an_exit_names_its_units_live_and_closes_the_position_in_the_backtest(runs):
    live, backtest, _, _ = runs
    columns = ("quantity_units", "filled_units")
    exits = [strategy_orders(result) for result in (live, backtest)]
    assert [rows(orders[orders["side"] == "SELL"], *columns) for orders in exits] == [
        [(UNITS, UNITS)],
        [(0.0, UNITS)],
    ]
