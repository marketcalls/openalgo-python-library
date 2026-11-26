"""Parity of positional live sessions with `bt.backtest` on the same bars (15-phase2-plan
section 4; design 20-positional-live 2, 6 and 8; the lead's ruling N5b-4, after unit N3b's
departure "no comparison of a positional live session with bt.backtest").

Each test runs a strategy live on the fake OpenAlgo and the same strategy, with the same
settings, in `bt.backtest` on the bars the live strategy decided on, then compares the
trades both close: entry and exit times, direction, units, exit reason and flags. Live
fills are at the broker's prices with no modelled slippage, so prices are checked by hand
for the live run only, and where the two paths cannot match a test says why and compares
what does.

- **CNC, a stop GTT and a target** (`positional_samples`' market: SBIN from 14:40:05 IST,
  `CncTarget` buying 10 at 500.00 at 14:42:01, a stop at 497.00 and a target at 502.00):
  SBIN at 503.00 from 14:44:20 takes the target.
- **The same with `positional_stops="rearm_daily"`**: the stop is a day SL-M at 497.00;
  the target as above, then SBIN at 496.00 through the stop of a second strategy.
- **A daily strategy** (`daily_samples`' market: a start at 09:05 on Monday 2026-11-16):
  it decides Friday's daily candle before the open and buys 10 at the open, as the
  backtest of the daily candles decides at Friday's close and fills at Monday's open.
"""

from __future__ import annotations

import datetime as dt
import logging

import pandas as pd
import pytest
from daily_samples import SESSIONS, UNITS, Recorded, close_of, daily_candle, daily_world
from fake_openalgo.fixtures import *  # noqa: F403
from positional_samples import CncHold, CncTarget, gtt_world
from session_recorder import record_sessions
from session_samples import CAPITAL, DAY, MAX_DAILY_LOSS, ist
from stops_samples import at_price

import openalgo.backtest as bt
from openalgo.backtest._live._result_build import session_result
from openalgo.backtest._live._types import LOGGER_NAME

IST = "Asia/Kolkata"
#: What each trade is compared by.
COLUMNS = (
    "entry_time",
    "exit_time",
    "direction",
    "entry_units",
    "max_units",
    "exit_reason",
    "flags",
)
REARM = {"positional_stops": "rearm_daily"}


@pytest.fixture(autouse=True)
def _logs(caplog):
    caplog.set_level(logging.INFO, logger=LOGGER_NAME)


def rows(frame, *columns):
    return [tuple(row) for row in frame.loc[:, list(columns)].itertuples(index=False)]


def both_runs(make_fake_openalgo, monkeypatch, tmp_path, strategy, moves, **settings):
    """(the live session's Result, the backtest's Result of the bars its strategy held) of
    `strategy` with `settings`, SBIN moved to each (instant, price) of `moves`; Ctrl+C at
    14:49 keeps what is open."""
    world = gtt_world(make_fake_openalgo, monkeypatch, tmp_path)
    for instant, price in moves:
        at_price(world, instant, price)
    recorder = record_sessions(monkeypatch)
    world.stop_at(ist(14, 49))
    outcome = world.run(strategy, **settings)
    (record,) = recorder.finish()
    name = strategy.__name__
    first = pd.Timestamp(record.decisions_from(name), tz="UTC").tz_convert(IST)
    backtest = bt.backtest(
        strategy,
        data=record.bars(name),
        capital=CAPITAL,
        max_daily_loss=MAX_DAILY_LOSS,
        start=(first - pd.Timedelta(minutes=1)).tz_localize(None),
        end=DAY,
        **settings,
    )
    return session_result(outcome).results[name], backtest


def test_a_cnc_target_beside_a_stop_gtt_closes_the_same_trade_as_the_backtest(
    make_fake_openalgo, monkeypatch, tmp_path
):
    live, backtest = both_runs(
        make_fake_openalgo, monkeypatch, tmp_path, CncTarget, [(ist(14, 44, 20), 503.0)]
    )

    assert rows(live.trades, *COLUMNS) == rows(backtest.trades, *COLUMNS)
    assert rows(live.trades, "exit_reason", "flags") == [("leg_target", [])]
    # The stop GTT was cancelled, then the exit sold at the market's 503.00.
    assert list(live.trades["exit_price_rupees"]) == [503.0]


def test_a_rearm_daily_target_closes_the_same_trade_as_the_backtest(
    make_fake_openalgo, monkeypatch, tmp_path
):
    live, backtest = both_runs(
        make_fake_openalgo, monkeypatch, tmp_path, CncTarget, [(ist(14, 44, 20), 503.0)], **REARM
    )

    assert rows(live.trades, *COLUMNS) == rows(backtest.trades, *COLUMNS)
    assert list(live.trades["exit_price_rupees"]) == [503.0]


def test_a_rearm_daily_day_stop_closes_the_same_trade_as_the_backtests_leg_stop(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # SBIN falls to 496.00 at 14:44:20, through the day SL-M at 497.00, which the broker
    # fills at the market's 496.00; the backtest's leg stop, an MIS one's under
    # rearm_daily, closes the trade in the same slice, reason "leg_sl".
    live, backtest = both_runs(
        make_fake_openalgo, monkeypatch, tmp_path, CncHold, [(ist(14, 44, 20), 496.0)], **REARM
    )

    assert rows(live.trades, *COLUMNS) == rows(backtest.trades, *COLUMNS)
    assert rows(live.trades, "exit_reason", "exit_price_rupees") == [("leg_sl", 496.0)]


def daily_frame(monday: tuple[float, float, float, float]) -> pd.DataFrame:
    """The daily candles of the sessions before Monday 2026-11-16, as the fake's broker
    makes them, and Monday's (open, high, low, close), dated by session."""
    days = [*SESSIONS, DAY]
    values = [daily_candle(day) for day in SESSIONS]
    values.append((*monday, 1_000.0))
    index = pd.DatetimeIndex([pd.Timestamp(day) for day in days])
    return pd.DataFrame(values, index=index, columns=["open", "high", "low", "close", "volume"])


def test_a_daily_strategy_buys_what_the_backtest_of_its_daily_candles_buys(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # The live strategy decides Friday's candle before Monday's open and its MARKET buy of
    # 10 goes at the open, filled at the market's 500.00. The backtest of the daily candles
    # from Friday decides at Friday's close and fills the buy at Monday's first open. A
    # daily backtest fills at the bar's open plus its slippage, so the prices are not
    # compared; the decision, the side, the units and the open the order went at are.
    world = daily_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(9, 30))
    outcome = world.run(Recorded)
    live_decided = list(Recorded.decided)
    Recorded.decided.clear()

    backtest = bt.backtest(
        Recorded,
        data=daily_frame((500.0, 500.0, 500.0, 500.0)),
        capital=CAPITAL,
        max_daily_loss=MAX_DAILY_LOSS,
        start=dt.date(2026, 11, 13),
        end=DAY,
    )

    # The backtest also decides Monday's candle at Monday's close, as the next live start
    # would before Tuesday's open.
    friday, monday = close_of(dt.date(2026, 11, 13)), close_of(DAY)
    assert [bar_close for _, bar_close, _ in live_decided] == [friday]
    assert [bar_close for _, bar_close, _ in Recorded.decided] == [friday, monday]
    live = session_result(outcome).results["Recorded"].fills
    columns = ("side", "quantity_units")
    assert rows(live, *columns) == rows(backtest.fills, *columns) == [("BUY", UNITS)]
    opened = pd.Timestamp(ist(9, 15), tz="UTC").tz_convert(IST)
    assert list(backtest.fills["time"]) == [opened]
    assert [t.floor("min") for t in live["time"]] == [opened]


#: The rupees `SizedDaily` buys for.
VALUE = 20_000


class SizedDaily(Recorded):
    """Recorded sized by value: it buys `VALUE` rupees of SBIN (`bt.size.value`)."""

    decided: list[tuple[int, int, int]] = []

    def on_bar(self, bar):
        type(self).decided.append((bar.ts_open, bar.ts_close, len(self.close)))
        if not self.position.qty and not self.orders:
            self.buy(size=bt.size.value(VALUE))


def test_a_daily_strategy_sized_by_value_buys_the_units_its_backtest_buys(
    make_fake_openalgo, monkeypatch, tmp_path
):
    # Both size at the close of the bar they decide, Friday's daily candle: whole shares
    # of 20,000 rupees at that close, worked out by hand from the candle.
    SizedDaily.decided.clear()
    world = daily_world(make_fake_openalgo, monkeypatch, tmp_path)
    world.stop_at(ist(9, 30))
    outcome = world.run(SizedDaily)

    backtest = bt.backtest(
        SizedDaily,
        data=daily_frame((500.0, 500.0, 500.0, 500.0)),
        capital=CAPITAL,
        max_daily_loss=MAX_DAILY_LOSS,
        start=dt.date(2026, 11, 13),
        end=DAY,
    )

    units = int(VALUE // daily_candle(dt.date(2026, 11, 13))[3])
    live = session_result(outcome).results["SizedDaily"].fills
    columns = ("side", "quantity_units")
    assert rows(live, *columns) == rows(backtest.fills, *columns) == [("BUY", units)]
