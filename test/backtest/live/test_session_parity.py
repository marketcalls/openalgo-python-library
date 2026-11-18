"""Parity of a live session and a backtest of the bars its strategy decided on
(15-phase2-plan section 4: "the same strategy on a recorded session and on the bars built
from it in a backtest takes the same decisions"; 16-session-design section 8 unit 16).

The session runs on the fake OpenAlgo under the recorder (`session_recorder`): SBIN on
NSE from 14:40:05 IST, trading 10 units every 10 s at 500.00 in the minute the session
starts in, then at one price a minute (`CLOSES`) from 14:41, until it falls to
`FALL_PRICE` at `FALL_AT`. Ctrl+C at 14:49 keeps whatever is open.

`Parity` reads a 3-bar average of the closes: flat, it buys 10 with a stop 5 points below
when a bar closes above the average (at most twice); long, it exits when a bar closes
below it. Its decisions turn on the bars right before the session's start: the first, a
buy at 14:42, is made because the minute the session started in (14:40, whose broker
candle the strategy's series took) closed at 500.00, which pulls the average below
14:41's close. It exits at 14:44 on a signal, buys again at 14:45, and that leg's stop
is hit in the fall.

The backtest runs the same strategy with the same settings on the bars its series held
(`SessionRecord.bars`: the warm-up's candles, then each slice's last bar), from the
first slice the session decided on (bt.live trades from the first whole minute it
watched, bt.backtest from `start`). The two take the same decisions: the same entries
and exits, decided at the same bars, filled on the same slices, for the same units and
reasons, with stops at the same distance from each run's entry. The strategy's last
decision (14:45) is bars before the record's last (14:47): an order decided on a
record's last bar has no next bar to fill on in a backtest, while live it fills, and the
scenario keeps that end out. What the design makes differ is asserted as that
difference:

- **The seed slice.** The minute the session started in is released to the kernel with
  no bar (its first ticks only seed the bar builder) and marked missing until its candle
  is read; the strategy's series then holds that candle. The kernel's bars lack it.
- **Fill prices.** Live fills at the broker's price (the fake's last trade), with no
  modelled slippage; the backtest fills at the next bar's open, or at the stop's level,
  moved by its slippage model against the trader.
- **Where the stop is held.** Live, each leg's stop is an SL-M order at the broker (D31),
  placed with the entry's fill and cancelled before a signal exit (one exit owner); the
  backtest holds it as a level of the leg, with no order.
- **An exit's units.** The live exit names the units it closes, as a broker order must;
  the backtest's closes whatever the position holds (0 units asked, the position filled).
- **A bar replaced after its decision.** After a tick gap the strategy's series takes the
  broker's candle in place of the bar built from ticks once the candle is read, after
  the decision at that slice's close (16-session-design 2.5 C). `SessionRecord.bars`
  gives the candle, the bar every later decision saw, so a backtest of them may decide
  that slice otherwise than the session did (`SessionRecord.replaced` names such
  slices). This scenario has no tick gap and no replaced bar (asserted); the tick gap of
  `test_session_candles` (F42) is recorded to show the slice where they differ.

Expected values come from the scenario, the fake's books and the backtest's own bars.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import pytest
from fake_openalgo.fixtures import *  # noqa: F403
from session_recorder import record_sessions
from session_samples import CAPITAL, DAY, MAX_DAILY_LOSS, PRICE, ist, rig
from stops_samples import stop_world
from test_session_candles import InfyTicks, Opens, climbing

import openalgo.backtest as bt
from openalgo import ta
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME

#: The stop's distance below each entry, points, and the units of each entry.
STOP_POINTS = 5.0
UNITS = 10
#: The bars of the average the strategy decides on.
AVERAGE_BARS = 3
#: SBIN's price in each minute from 14:41 IST (minute: rupees).
CLOSES = {41: 510.0, 42: 512.0, 43: 509.0, 44: 513.0, 45: 515.0, 46: 516.0}
#: The price SBIN falls to at `FALL_AT`, through the second entry's stop.
FALL_PRICE, FALL_AT = 505.0, ist(14, 46, 20)
#: The minute the session starts in (its seed slice) and SBIN's price in it, rupees.
SEED_MINUTE, SEED_PRICE = ist(14, 40), 500.0
IST = "Asia/Kolkata"


def price(now_ns: int) -> float:
    """SBIN's price at `now_ns` in the scenario (module docstring)."""
    if now_ns >= FALL_AT:
        return FALL_PRICE
    started = [minute for minute in CLOSES if now_ns >= ist(14, minute)]
    return CLOSES[max(started)] if started else SEED_PRICE


class Parity(bt.Strategy):
    """Buys `UNITS` with a stop `STOP_POINTS` below when a bar closes above the average
    of the last `AVERAGE_BARS` closes, at most twice; exits when a bar closes below it."""

    symbol, exchange, interval = "SBIN", "NSE", "1m"

    def init(self):
        self.average = self.I(ta.sma, self.close, AVERAGE_BARS)
        self.state.setdefault("entries", 0)

    def on_bar(self, bar):
        above = bar.close > self.average[-1]
        if self.position.qty:
            if not above:
                self.exit()
        elif not self.orders and above and self.state["entries"] < 2:
            self.state["entries"] += 1
            self.buy(UNITS, stop=bt.points(STOP_POINTS))


def ist_time(ts_ns: int) -> pd.Timestamp:
    """An instant (UTC nanoseconds) in IST, as the Results show times."""
    return pd.Timestamp(ts_ns, tz="UTC").tz_convert(IST)


@pytest.fixture
def runs(make_fake_openalgo, monkeypatch, tmp_path, caplog):
    """(the live session's Result, the backtest's Result of its bars, those bars, the
    session's record, the fake)."""
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    world = stop_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.script.price = price
    recorder = record_sessions(monkeypatch)
    world.stop_at(ist(14, 49))
    outcome = world.run(Parity, on_stop="keep")
    (record,) = recorder.finish()
    bars = record.bars("Parity")
    # The slice that closes at the session's first decision, naive in IST.
    first = ist_time(record.decisions_from("Parity")) - pd.Timedelta(minutes=1)
    backtest = bt.backtest(
        Parity,
        data=bars,
        capital=CAPITAL,
        max_daily_loss=MAX_DAILY_LOSS,
        start=first.tz_localize(None),
        end=DAY,
    )
    return session_result(outcome).results["Parity"], backtest, bars, record, world.fake


def rows(frame: Any, *columns: str) -> list[tuple]:
    return [tuple(row) for row in frame.loc[:, list(columns)].itertuples(index=False)]


def strategy_orders(result: Any) -> Any:
    """The orders the strategy decided: all but the broker-held stops."""
    orders = result.orders
    return orders[orders["order_type"] != "SL-M"]


def test_the_backtest_of_the_bars_the_strategy_held_makes_the_same_trades(runs):
    live, backtest, _, _, _ = runs
    columns = ("entry_time", "exit_time", "direction", "entry_units", "max_units", "exit_reason")
    assert rows(live.trades, *columns) == rows(backtest.trades, *columns)
    assert [row[-1] for row in rows(live.trades, *columns)] == ["signal", "leg_sl"]


def test_the_strategys_orders_are_decided_at_the_same_bars_with_the_same_terms(runs):
    live, backtest, _, _, _ = runs
    columns = ("decided_at", "side", "order_type", "filled_units", "status")
    assert rows(strategy_orders(live), *columns) == rows(strategy_orders(backtest), *columns)
    decided = ((42, "BUY"), (44, "SELL"), (45, "BUY"))
    assert rows(strategy_orders(live), "decided_at", "side") == [
        (ist_time(ist(14, minute)), side) for minute, side in decided
    ]


def test_the_first_decision_turns_on_the_seed_slices_candle(runs):
    # The buy at 14:42: with the seed minute's close the average of the bar closing then
    # is below 14:41's close; without it, above.
    _, _, bars, _, _ = runs
    seed = ist_time(SEED_MINUTE)

    def average(closes: pd.Series) -> float:
        return (sum(closes.loc[:seed].tail(AVERAGE_BARS - 1)) + CLOSES[41]) / AVERAGE_BARS

    assert bars.loc[seed, "close"] == SEED_PRICE
    assert average(bars["close"]) < CLOSES[41] < average(bars["close"].drop(index=seed))


def test_the_kernels_bars_lack_only_the_seed_slice_which_the_strategy_held_as_its_candle(runs):
    _, _, bars, record, _ = runs
    kernel = record.kernel_bars("Parity")
    seed = ist_time(SEED_MINUTE)
    assert list(bars.index.difference(kernel.index)) == [seed]
    assert kernel.index.difference(bars.index).empty
    assert bars.drop(index=seed).equals(kernel)


def test_no_bar_the_strategy_decided_on_was_replaced_after_its_decision(runs):
    # With no tick gap, the series never takes a candle in place of a bar it decided on,
    # so each decision saw the bar the backtest is given.
    _, _, _, record, _ = runs
    assert record.replaced("Parity") == []


def test_a_tick_gaps_slice_is_decided_on_its_tick_bar_and_given_to_a_backtest_as_its_candle(
    make_fake_openalgo, monkeypatch, tmp_path, caplog
):
    # F42's gap: SBIN's first three trades of 14:43 send no tick, so the bar built from
    # ticks opens at the fourth (PRICE + 1.50) while the broker's candle opens at PRICE.
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)
    Opens.seen = []
    world = rig(make_fake_openalgo, monkeypatch, tmp_path, symbols=("SBIN", "INFY"))
    world.script.price = climbing
    world.script.at(ist(14, 42, 59), lambda: world.fake.faults.tick_gap("SBIN", "NSE", trades=3))
    world.stop_at(ist(14, 46))
    recorder = record_sessions(monkeypatch)
    world.run([Opens, InfyTicks])
    (record,) = recorder.finish()
    gap = ist(14, 43)
    assert record.replaced("Opens") == [gap]
    takes = [
        at
        for at, line in enumerate(record.lines)
        if (line["strategy"], line["kind"], line.get("call")) == ("Opens", "series", "take")
        and line["args"][0]["ts_open_ns"] == gap
    ]
    decided = [line for line in record.lines[takes[0] : takes[-1]] if line["kind"] == "drive"]
    assert any(line["strategy"] == "Opens" and line["deciding"] for line in decided)
    at = ist_time(gap)
    assert record.kernel_bars("Opens").loc[at, "open"] == pytest.approx(PRICE + 1.5)
    assert record.bars("Opens").loc[at, "open"] == pytest.approx(PRICE)


def test_the_strategys_last_decision_is_bars_before_the_records_last(runs):
    # An order decided on a record's last bar has no next bar to fill on in a backtest.
    live, _, bars, _, _ = runs
    assert strategy_orders(live)["decided_at"].max() < bars.index[-1]


def test_each_fill_is_on_the_same_slice_and_side_for_the_same_reason(runs):
    live, backtest, _, _, _ = runs
    columns = ("time", "side", "quantity_units", "reason")
    assert rows(live.fills, *columns) == rows(backtest.fills, *columns)


def test_the_stops_are_the_same_distance_below_each_runs_entry(runs):
    live, backtest, _, _, _ = runs
    risk = [STOP_POINTS * UNITS] * 2
    assert list(live.trades["initial_risk_rupees"]) == risk
    assert list(backtest.trades["initial_risk_rupees"]) == risk


def test_live_fills_are_at_the_brokers_prices_with_no_modelled_slippage(runs):
    # A market order sent as a minute closes fills at that minute's last trade.
    live, _, _, _, fake = runs
    broker = [fill.price for fill in fake.fills()]
    expected = [CLOSES[41], CLOSES[43], CLOSES[44], FALL_PRICE]
    assert list(live.fills["price_rupees"]) == broker == expected
    assert live.fills["slippage_rupees"].isna().all()


def test_backtest_fills_are_the_next_open_or_the_stops_level_moved_by_its_slippage(runs):
    _, backtest, bars, _, _ = runs
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
    live, backtest, _, _, _ = runs
    entries = live.trades["entry_price_rupees"].tolist()
    stops = live.orders[live.orders["order_type"] == "SL-M"]
    assert rows(stops, "side", "quantity_units", "status") == [
        ("SELL", UNITS, "Canceled"),
        ("SELL", UNITS, "Filled"),
    ]
    assert list(stops["trigger_rupees"]) == [price - STOP_POINTS for price in entries]
    assert "SL-M" not in set(backtest.orders["order_type"])


def test_an_exit_names_its_units_live_and_closes_the_position_in_the_backtest(runs):
    live, backtest, _, _, _ = runs
    columns = ("quantity_units", "filled_units")
    exits = [strategy_orders(result) for result in (live, backtest)]
    assert [rows(orders[orders["side"] == "SELL"], *columns) for orders in exits] == [
        [(UNITS, UNITS)],
        [(0.0, UNITS)],
    ]
